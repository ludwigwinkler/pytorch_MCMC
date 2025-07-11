# %%
import functools
import copy
from tqdm import tqdm

import torch
from tensordict import TensorDict, NonTensorData
import matplotlib.pyplot as plt
from dataclasses import dataclass
from numbers import Number

from mcmc.sampler import (
    MALASampler,
    MetropolisHastingsAcceptance,
    SGLDSampler,
    MHSampler,
)
from mcmc.energy import Energy, GaussianMixture1D, GaussianMixture2D
from mcmc.utils import EMA, RepeatedCosineSchedule
from mcmc.data import generate_nonstationary_data, generate_multimodal_linear_regression

from torch.nn import Sequential, Linear, ReLU, Tanh, BatchNorm1d

import os


plt.style.use("default")
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["text.color"] = "black"
plt.rcParams["axes.labelcolor"] = "black"
plt.rcParams["xtick.color"] = "black"
plt.rcParams["ytick.color"] = "black"
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.titlecolor"] = "black"
plt.rcParams["figure.edgecolor"] = "white"
plt.rcParams["legend.edgecolor"] = "black"
plt.rcParams["legend.facecolor"] = "white"


x, y = generate_nonstationary_data(
    num_samples=1_000,
    plot=False,
    y_nonstationary_noise_std=0.3,
    y_constant_noise_std=0.01,
)


class ProbModel(Energy):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            BatchNorm1d(1),
            Linear(1, 32),
            Tanh(),
            Linear(32, 64),
            Tanh(),
            Linear(64, 64),
            ReLU(),
            Linear(64, 2),
        )

    def forward(self, x):
        out = self.model(x)
        mu, log_std = out.chunk(2, dim=-1)
        # return self.model(x) + std * torch.randn_like(x)
        return mu, torch.nn.functional.softplus(log_std)

    # def energy(self, mu, std, y):
    #     # Assuming a simple energy function for demonstration
    #     NLL = -torch.distributions.Normal(mu, std).log_prob(y)
    #     return NLL

    @staticmethod
    def energy(probmodel, params, buffers, data, target, other=None):
        mu, std = torch.func.functional_call(probmodel, (params, buffers), (data,))
        energy = -torch.distributions.Normal(mu, std).log_prob(target).mean(dim=-2)
        return energy

    @staticmethod
    def predict(prob_model, params, buffers, data):
        # params, buffers = sample["params"], sample["buffers"]
        mu, std = torch.func.functional_call(prob_model, (params, buffers), (data,))
        return mu, std

    def pretrain(self, x, y, num_steps=100, lr=1e-3):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # prog_bar = tqdm(range(num_steps), desc="Pretraining")
        for step in range(num_steps):
            optimizer.zero_grad()
            mu, std = self.forward(x)
            loss = -torch.distributions.Normal(mu, std).log_prob(y).mean(dim=-2)
            mse = torch.nn.functional.mse_loss(mu, y)
            loss.backward()
            optimizer.step()
            if step % 100 == 0 or step < 5 or step == num_steps - 1:
                # prog_bar.set_postfix({"Energy": loss.item(), "MSE": mse.item()})
                print(f"Step {step}, Loss: {loss.item()}, MSE: {mse.item()}")


probmodel = ProbModel()
# probmodel.pretrain(x, y, num_steps=100, lr=1e-3)

num_chains = 11
models = [copy.deepcopy(probmodel) for _ in range(num_chains)]

params, buffers = torch.func.stack_module_state(models)
init_samples = TensorDict(
    {"params": params, "buffers": buffers, "data": x, "target": y, "aux": "abc"},
)


# %%
energy1 = lambda params, buffers, data, target, aux: probmodel.energy(
    probmodel.train(), params, buffers, data, target
)
vmap_energy = torch.vmap(energy1, (0, 0, None, None, None), randomness="different")
# init_args = list(init_samples.values())
init_args = [
    arg.to_dict() if isinstance(arg, TensorDict) else arg
    for arg in list(init_samples.values())
]
init_energy = vmap_energy(
    *init_args,
)

print(init_energy)


# %%


def plot_uncertainty(params, buffers, title=""):
    x_test = torch.linspace(-4, 4, 100).unsqueeze(-1)
    mu, std = torch.vmap(probmodel.predict, (None, 0, 0, None), randomness="different")(
        probmodel.eval(), params.to_dict(), buffers.to_dict(), x_test
    )
    mu, std = mu.detach().numpy(), std.detach().numpy()
    plt.figure(figsize=(12, 6))
    plt.scatter(x.squeeze(-1), y.squeeze(-1), label="Data", color="blue", s=1)
    for i in range(num_chains):
        plt.plot(x_test.squeeze(-1), mu[i].squeeze(-1), color="red", alpha=0.1)

    # Plot mean prediction and uncertainty bands
    mean_pred = mu.mean(axis=0).squeeze(-1)
    mean_std = std.mean(axis=0).squeeze(-1)
    plt.plot(x_test.squeeze(-1), mean_pred, color="black", label="Mean Prediction")
    for k, alpha in zip([1, 2, 3], [0.2, 0.1, 0.05]):
        plt.fill_between(
            x_test.squeeze(-1),
            mean_pred - k * mean_std,
            mean_pred + k * mean_std,
            color="red",
            alpha=alpha,
            label=f"{k} std" if k == 1 else None,
        )
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim(-2, 2)
    plt.title("Model Prediction " + title)
    plt.show()


# %%

# SGLD Sampling
print("Running SGLD sampling...")

# Create SGLD sampler
# Sampler = SGLDSampler(step_size=0.01, dampening=0.001)
# Sampler = MALASampler(step_size=0.01, dampening=0.0)
Sampler = MHSampler(std=0.01)

# Run SGLD sampling
num_steps = 500
samples, energies = Sampler(
    sample=init_samples,
    energy_fn=vmap_energy,  # Use the existing vmap_energy directly
    steps=num_steps,
    verbose=True,
    buffer=50,
    burn_in=100,
)

print(
    f"SGLD sampling completed. Final energy: {torch.stack(energies).mean().item():.4f}"
)
params = torch.cat([sample["params"] for sample in samples])
buffers = torch.cat([sample["buffers"] for sample in samples])

# Plot results from SGLD sampling
plot_uncertainty(params, buffers, title="SGLD Sampling")

# %%
