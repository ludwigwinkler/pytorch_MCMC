# %%
import torch
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

model = torch.nn.Sequential(torch.)


# %%
num_chains = 50_000
proposal_std = 0.5
vmap_energy = torch.vmap(gmm.energy, (0,))
init_sample = TensorDict({"x": 3 * torch.randn((num_chains, 2)).clamp(-5, 5)})
init_energy = vmap_energy(init_sample["x"])
chain = [(init_sample, init_energy)]


num_steps = [100, 1_000][1]
accept_ema = EMA(ema_weight=0.99)
pbar = tqdm(range(num_steps))
schedule = RepeatedCosineSchedule(steps=num_steps, cycles=5)
MH = MetropolisHastingsAcceptance()
for step in pbar:
    state, energy = chain[-1]
    proposal_std_ = schedule(step=step, min=0.1, max=1.0)
    proposal_state = state.clone().apply(
        lambda x: x + torch.randn_like(x) * proposal_std_
    )
    proposal_energy = vmap_energy(proposal_state["x"])
    accept: torch.Tensor = MH(energy, proposal_energy)
    new_state = state.apply(
        lambda state_, proposal_state_: torch.where(accept, proposal_state_, state_),
        proposal_state,
    )
    chain += [(new_state, proposal_energy)]
    accept_ratio = accept.sum() / accept.numel()
    accept_ema(accept_ratio.item())
    pbar.set_postfix(
        {
            "Accept": f"{accept_ema.val:.3f}",
            "PropStd": f"{proposal_std_:.3f}",
        }
    )

samples = chain[-1][0]["x"]
x = samples[:, 0].numpy()
y = samples[:, 1].numpy()
plt.figure(figsize=(6, 5))
plt.hist2d(x, y, bins=100, density=True, cmap="viridis")
plt.colorbar(label="Density")
plt.xlabel("x")
plt.ylabel("y")
plt.title("2D Gaussian Mixture Samples")
plt.show()
