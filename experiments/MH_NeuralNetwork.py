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

from torch.nn import Sequential, Linear, ReLU

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
    plot=True,
    y_nonstationary_noise_std=0.1,
    y_constant_noise_std=0.1,
)


class ProbModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def energy(self, x, y):
        # Assuming a simple energy function for demonstration
        return 0.5 * (self.model(x) - y) ** 2


num_chains = 123
base_model = torch.nn.Sequential(
    Linear(1, 32),
    ReLU(),
    Linear(32, 32),
    ReLU(),
    Linear(32, 32),
    ReLU(),
    Linear(32, 32),
    ReLU(),
    Linear(32, 1),
)


models = [copy.deepcopy(base_model) for _ in range(num_chains)]

params, buffers = torch.func.stack_module_state(models)


def single_forward(params, buffers, data):
    return torch.func.functional_call(base_model, (params, buffers), (data,))


def single_energy(params, buffers, data, target):
    vmap_pred = torch.func.functional_call(base_model, (params, buffers), (data,))
    return torch.nn.functional.mse_loss(vmap_pred, target).unsqueeze(-1)


vmap_energy = torch.vmap(single_energy, (0, 0, None, None))
init_energy = torch.vmap(single_energy, (0, 0, None, None))(params, buffers, x, y)


# %%

proposal_std = 0.1
chain = [((TensorDict(params), TensorDict(buffers)), init_energy)]


num_steps = [300, 1_000][1]
accept_ema = EMA(ema_weight=0.99)
pbar = tqdm(range(num_steps))
schedule = RepeatedCosineSchedule(steps=num_steps, cycles=5)
MH = MetropolisHastingsAcceptance()
for step in pbar:
    (params, buffers), energy = chain[-1]
    proposal_std_ = schedule(step=step, min=0.01, max=0.1)
    grad, _ = torch.func.grad_and_value(
        lambda p, b, x, y: torch.sum(vmap_energy(p, b, x, y)),
        argnums=(0,),
    )(params.to_dict(), buffers.to_dict(), x, y)
    grad = TensorDict(grad[0])
    proposal_params = params.clone().apply(
        lambda x, grad: x
        - proposal_std_ * grad
        + 0.01 * torch.randn_like(x) * (2 * proposal_std_) ** 0.5,
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
    pbar.set_postfix(
        {
            "Accept": f"{accept_ema.val:.3f}",
            "PropStd": f"{proposal_std_:.3f}",
            "Energy": f"{energy.mean().item():.3f}",
        }
    )

# %%

(params, buffers), energy = chain[-1]
params, buffers = params.to_dict(), buffers.to_dict()
x_test = torch.linspace(-0.35, 0.45, 100).unsqueeze(-1)
y_pred = torch.vmap(single_forward, (0, 0, None))(params, buffers, x_test).detach()
plt.figure(figsize=(8, 6))
plt.scatter(x.squeeze(-1), y.squeeze(-1), label="Data", color="blue", s=1)
for i in range(num_chains):
    plt.plot(x_test.squeeze(-1), y_pred[i].squeeze(-1), color="red", alpha=0.1)
# plt.plot(x.squeeze(-1), y_pred.squeeze(-1), label="Model Prediction", color="red")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Model Prediction vs Data")
