# %%
import torch
from tensordict import TensorDict
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass
from numbers import Number

# import mcmc.sampler.MetropolisHastingAcceptance
# from mcmc.sampler import MetropolisHastingAcceptance

from mcmc.sampler import MALASampler, MHSampler, SGLDSampler
from mcmc.energy import GaussianMixture1D, GaussianMixture2D
from mcmc.utils import EMA, RepeatedCosineSchedule

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


gmm = GaussianMixture2D()
samples = gmm.sample(500_000)

# 2D contour plot of samples2d
x = samples[:, 0].numpy()
y = samples[:, 1].numpy()
# 2D histogram of the Gaussian Mixture Model samples
plt.figure(figsize=(6, 5))
plt.hist2d(x, y, bins=100, density=True, cmap="viridis")
plt.colorbar(label="Density")
plt.xlabel("x")
plt.ylabel("y")
plt.title("2D Gaussian Mixture Samples")
plt.show()

# %%


# %%
num_chains = 500
num_steps = 2000
proposal_std = 0.5
energy_fn = torch.vmap(gmm.energy, (0,))
init_sample = TensorDict({"x": 3 * torch.randn((num_chains, 2)).clamp(-5, 5)})
init_energy = energy_fn(init_sample["x"])

schedule = RepeatedCosineSchedule(steps=num_steps, cycles=1, min=0.01, max=1.0)
# Sampler = MALASampler(step_size=1.0, dampening=1.0, step_size_schedule=schedule)
# Sampler = SGLDSampler(step_size=0.05, dampening=1.0)
Sampler = MHSampler(std=1.0)
samples, energy = Sampler(
    sample=init_sample,
    energy_fn=energy_fn,
    steps=num_steps,
    verbose=True,
    buffer=500,
    burn_in=50,
)

x = samples["x"][:, 0].numpy()
y = samples["x"][:, 1].numpy()
plt.figure(figsize=(6, 5))
plt.hist2d(x, y, bins=100, density=True, cmap="viridis")
plt.colorbar(label="Density")
plt.xlabel("x")
plt.ylabel("y")
plt.title("2D Gaussian Mixture Samples")
plt.show()
