# %%
import torch
import copy
from tensordict import TensorDict
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass
from numbers import Number

from mcmc.sampler import MetropolisHastingsAcceptance
from mcmc.energy import GaussianMixture1D, GaussianMixture2D
from mcmc.utils import EMA, RepeatedCosineSchedule
from mcmc.data import generate_nonstationary_data, generate_multimodal_linear_regression

from torch.nn import Sequential, Linear, ReLU, Tanh, BatchNorm1d

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


class ProbModel(torch.nn.Module):
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


num_chains = 25
nn = torch.nn.Sequential(
    # BatchNorm1d(1),
    Linear(1, 32),
    Tanh(),
    Linear(32, 64),
    Tanh(),
    Linear(64, 64),
    ReLU(),
    Linear(64, 2),
)

probmodel = ProbModel(nn)
probmodel.pretrain(x, y, num_steps=100, lr=1e-3)


models = [copy.deepcopy(probmodel) for _ in range(num_chains)]

params, buffers = torch.func.stack_module_state(models)


def single_forward(params, buffers, data):
    return torch.func.functional_call(probmodel, (params, buffers), (data,))


def single_energy(params, buffers, data, target):
    mu, std = torch.func.functional_call(probmodel, (params, buffers), (data,))
    energy = probmodel.energy(mu, std, target).mean(dim=-2)
    return energy


vmap_energy = torch.vmap(single_energy, (0, 0, None, None), randomness="different")
init_energy = vmap_energy(params, buffers, x, y)


# %%

proposal_std = 0.1
chain = [((TensorDict(params), TensorDict(buffers)), init_energy)]


num_steps = [100, 1000][1]
accept_ema = EMA(ema_weight=0.99)
energy_ema = EMA(ema_weight=0.99)
pbar = tqdm(range(num_steps))
schedule = RepeatedCosineSchedule(steps=num_steps, cycles=1)
MH = MetropolisHastingsAcceptance()
for step in pbar:
    (params, buffers), energy = chain[-1]
    proposal_std_ = schedule(step=step, min=0.001, max=0.01)
    grad, _ = torch.func.grad_and_value(
        lambda p, b, x, y: torch.sum(vmap_energy(p, b, x, y)),
        argnums=(0,),
    )(params.to_dict(), buffers.to_dict(), x, y)
    grad = TensorDict(grad[0])  # .apply(lambda x: torch.clip(x, min=-1.0, max=1.0))
    proposal_params = params.clone().apply(
        lambda x, grad: x
        - proposal_std_ * grad
        + torch.randn_like(x) * (2 * proposal_std_ * 0.01) ** 0.5,
        grad,
    )
    proposal_energy = vmap_energy(params.to_dict(), buffers.to_dict(), x, y)
    accept: torch.Tensor = MH(energy, proposal_energy)
    new_params = params.apply(
        lambda state_, proposal_state_: torch.where(
            accept[(...,) + (None,) * (state_.dim() - 2)], proposal_state_, state_
        ),
        proposal_params,
    )
    chain = [((TensorDict(new_params), TensorDict(buffers)), proposal_energy)]
    accept_ratio = accept.sum() / accept.numel()
    accept_ema(accept_ratio.item())
    energy_ema(energy.mean().item())
    del grad
    pbar.set_postfix(
        {
            "Accept": f"{accept_ema.val:.3f}",
            "PropStd": f"{proposal_std_:.3f}",
            "Energy": f"{energy_ema.val:.3f}",
        }
    )

# %%

(params, buffers), energy = chain[-1]
params, buffers = params.to_dict(), buffers.to_dict()
x_test = torch.linspace(-5, 5, 100).unsqueeze(-1)
mu, std = torch.vmap(single_forward, (0, 0, None), randomness="different")(
    params, buffers, x_test
)
mu, std = mu.detach().numpy(), std.detach().numpy()
plt.figure(figsize=(8, 6))
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
plt.ylim(-3, 3)
plt.title("Model Prediction vs Data")

# %%
print("Done")
