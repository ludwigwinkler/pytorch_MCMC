# %%
import torch
import copy
from tensordict import TensorDict
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass
from numbers import Number

from mcmc.sampler import MetropolisHastingsAcceptance
from mcmc.energy import Energy, GaussianMixture1D, GaussianMixture2D
from mcmc.utils import EMA, RepeatedCosineSchedule
from mcmc.data import generate_nonstationary_data, generate_multimodal_linear_regression

from torch.nn import Sequential, Linear, ReLU, Tanh, BatchNorm1d

import os
import psutil

process = psutil.Process(os.getpid())

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
    num_samples=250,
    plot=False,
    y_nonstationary_noise_std=0.3,
    y_constant_noise_std=0.01,
)


class ProbModel(Energy):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        mu, log_std = out.chunk(2, dim=-1)
        # return self.model(x) + std * torch.randn_like(x)
        return mu, torch.nn.functional.softplus(log_std)

    def energy(self, mu, std, y):
        # Assuming a simple energy function for demonstration
        NLL = -torch.distributions.Normal(mu, std).log_prob(y)
        return NLL

    def pretrain(self, x, y, num_steps=100, lr=1e-3):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # prog_bar = tqdm(range(num_steps), desc="Pretraining")
        for step in range(num_steps):
            optimizer.zero_grad()
            mu, std = self.forward(x)
            loss = self.energy(mu, std, y).mean(dim=-2)
            mse = torch.nn.functional.mse_loss(mu, y)
            loss.backward()
            optimizer.step()
            if step % 100 == 0 or step < 5 or step == num_steps - 1:
                # prog_bar.set_postfix({"Energy": loss.item(), "MSE": mse.item()})
                print(f"Step {step}, Loss: {loss.item()}, MSE: {mse.item()}")


num_chains = 10
nn = torch.nn.Sequential(
    BatchNorm1d(1),
    Linear(1, 32),
    Tanh(),
    Linear(32, 64),
    Tanh(),
    Linear(64, 64),
    ReLU(),
    Linear(64, 2),
)

probmodel = ProbModel(nn)
# probmodel.pretrain(x, y, num_steps=100, lr=1e-3)


models = [copy.deepcopy(probmodel) for _ in range(num_chains)]

params, buffers = torch.func.stack_module_state(models)
init_params, init_buffers = copy.deepcopy(params), copy.deepcopy(buffers)


def single_predict(params, buffers, data):
    return torch.func.functional_call(probmodel.eval(), (params, buffers), (data,))


def single_energy(prob_model, params, buffers, data, target):
    mu, std = torch.func.functional_call(prob_model, (params, buffers), (data,))
    energy = probmodel.energy(mu, std, target).mean(dim=-2)
    return energy


vmap_energy = torch.vmap(
    single_energy, (None, 0, 0, None, None), randomness="different"
)
init_energy = vmap_energy(probmodel.eval(), params, buffers, x, y)

# %%


def plot_uncertainty(params, buffers, str=""):
    x_test = torch.linspace(-4, 4, 100).unsqueeze(-1)
    mu, std = torch.vmap(single_predict, (0, 0, None), randomness="different")(
        params, buffers, x_test
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
    plt.title("Model Prediction " + str)
    plt.show()


# plot_uncertainty(params, buffers)


# %%
# Sampling

proposal_std = 0.001

chain = [
    (
        (
            TensorDict(params, batch_size=num_chains),
            TensorDict(buffers, batch_size=num_chains),
        ),
        init_energy,
    )
]

num_steps = [100, 1000, 10000][1]
accept_ema = EMA(ema_weight=0.99)
energy_ema = EMA(ema_weight=0.9)
pbar = tqdm(range(num_steps))
# pbar = range(num_steps)
schedule = RepeatedCosineSchedule(steps=num_steps // 2, cycles=1, min=0.001, max=0.01)

for step in pbar:
    (params, buffers), energy = chain[-1]

    proposal_std_ = schedule(step=step)

    # Compute new parameters
    with torch.enable_grad():
        grad, energy_ = torch.func.grad_and_value(
            lambda model, p, b, x, y: torch.sum(vmap_energy(model, p, b, x, y)),
            argnums=(1,),
        )(probmodel.train(), params.to_dict(), buffers.to_dict(), x, y)
        energy_.detach()
        grad = TensorDict(grad[0], batch_size=num_chains).detach()
    proposal_params = copy.deepcopy(params).apply(
        lambda x, grad: x
        - proposal_std_ * grad
        + 0.1 * torch.randn_like(x) * (2 * proposal_std_) ** 0.5,
        grad,
    )
    # Evaluate proposal energies
    with torch.no_grad():
        proposal_energy = vmap_energy(
            probmodel.eval(), params.to_dict(), buffers.to_dict(), x, y
        ).detach()
    accept: torch.Tensor = MetropolisHastingsAcceptance(
        energy, proposal_energy
    ).detach()

    # Update parameters based on acceptance
    proposal_params.auto_batch_size_(1)
    params.auto_batch_size_(1)
    next_params = []
    for accept_, p_, p in zip(
        accept,
        proposal_params.chunk(num_chains, dim=0),
        params.chunk(num_chains, dim=0),
    ):
        next_params.append(p_) if accept_.item() else next_params.append(p)
    new_params = torch.cat(next_params, dim=0)

    # new_params = torch.concat([p_ for accept, p_, p in zip(accept, new_params.chunk(num_chains, dim=0), params.chunk(num_chains, dim=0)) if accept.item() else p], dim=0)

    chain = [((TensorDict(new_params), TensorDict(buffers)), proposal_energy)]
    accept_ratio = accept.sum() / accept.numel()
    accept_ema(accept_ratio.detach().item())
    energy_ema(energy.mean().detach().item())
    # del grad, next_params, proposal_params, proposal_energy, accept, buffers, params, energy

    pbar.set_postfix(
        {
            "Accept": f"{accept_ema.val:.3f}",
            "PropStd": f"{proposal_std_:.3f}",
            "Energy": f"{energy_ema.val:.3f}",
        }
    )

    if step % (num_steps // 5) == 0 or step == num_steps - 1:
        plot_uncertainty(new_params.to_dict(), buffers.to_dict(), str=f"Step {step}")
        # plt.savefig(f"MH_NeuralNetwork_step_{step}.png", dpi=300)
        plt.close()
        # plot_uncertainty(new_params.to_dict(), buffers.to_dict(), str=f"Step {step}")
