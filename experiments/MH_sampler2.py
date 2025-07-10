# %%
import torch
from typing import Callable, List
from tensordict import TensorDict
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass
from numbers import Number

# import mcmc.sampler.MetropolisHastingAcceptance
# from mcmc.sampler import MetropolisHastingAcceptance

from mcmc.sampler import MHSampler, ImportanceSampler
from mcmc.energy import Gaussian1D, GaussianMixture1D
from mcmc.utils import RepeatedCosineSchedule

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

# %%


# %%
num_steps = 1_000
num_chains = 500
Energy = Gaussian1D(mean=1, std=0.5)
energy_fn = torch.vmap(lambda td: Energy.energy(td["x"]), in_dims=(0,))

# schedule = RepeatedCosineSchedule(steps=num_steps // 2, cycles=1, min=0.005, max=1.0)
sampler = MHSampler(std=0.25)

init_samples = TensorDict(
    {"x": 0.5 + 0.5 * torch.randn((num_chains, 1)).clamp(-10, 10)},
    batch_size=num_chains,
)
samples, energy = sampler(
    sample=init_samples, energy_fn=energy_fn, steps=num_steps, verbose=True
)

# %%


# %%


_ = plt.hist(
    init_samples["x"].numpy(),
    bins=100,
    density=True,
    alpha=0.5,
    label="Init Distribution",
)
_ = plt.hist(
    samples["x"].numpy(),
    bins=50,
    density=True,
    alpha=0.5,
    label="Final Sampled Distribution",
)
_ = plt.hist(
    Energy.sample(50_000).numpy(),
    bins=100,
    density=True,
    alpha=0.5,
    label="True Distribution",
)
plt.legend()
plt.xlim(-5, 5)
plt.ylim(0, 2)

# %%

proposal_distribution = torch.distributions.Normal(loc=0.0, scale=2.0)
IS_energy_fn = Energy.energy
IS = ImportanceSampler()
_, _, Z_est, _ = IS(
    energy_fn=IS_energy_fn,
    samples=10_000,
    proposal_distribution=proposal_distribution,
)

print(f"Estimated Partition Function Z: {Z_est.item()} vs {Energy.Z.item()}")
