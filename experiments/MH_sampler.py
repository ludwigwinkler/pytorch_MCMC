# %%
import torch
from tensordict import TensorDict
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass
from numbers import Number

# import mcmc.sampler.MetropolisHastingAcceptance
# from mcmc.sampler import MetropolisHastingAcceptance

from mcmc.sampler import MHSampler
from mcmc.energy import GaussianMixture1D, GaussianMixture2D
from mcmc.utils import EMA

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


# gmm2d = GaussianMixture2D()
# samples2d = gmm2d.sample(50_000)

# # 2D contour plot of samples2d
# x = samples2d[:, 0].numpy()
# y = samples2d[:, 1].numpy()
# plt.figure(figsize=(6, 5))
# plt.hist2d(x, y, bins=100, density=True, cmap="viridis")
# plt.colorbar(label="Density")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("2D Gaussian Mixture Samples")
# plt.show()

# %%
Energy = GaussianMixture1D()

data = Energy.sample(50_000)

# Create initial sample: batch of 100 chains, each with 1D x
num_chains = 500
num_steps = 1000
x_init = torch.randn(num_chains, 1) * 3
init_sample = TensorDict({"x": x_init}, batch_size=[num_chains])

# Vectorize the energy function using torch.func.vmap
energy_fn = torch.vmap(lambda td: Energy.energy(td["x"]), in_dims=(0,))

# Run the sampler for a small number of steps
Sampler = MHSampler(std=1.0)
samples, energy = Sampler(
    sample=init_sample, energy_fn=energy_fn, steps=num_steps, verbose=True
)

print(samples)

# samples = [s for s, e in chain]
# samples = torch.cat(samples, dim=0)

plt.hist(
    init_sample["x"].numpy(),
    density=True,
    bins=50,
    color="green",
    alpha=0.5,
    label="Initial Samples",
)
plt.hist(
    samples["x"].numpy(),
    density=True,
    bins=100,
    color="blue",
    alpha=0.5,
    label="Final Samples",
)
plt.legend()

plt.plot(torch.linspace(-5, 5, 100), Energy.prob(torch.linspace(-5, 5, 100)))
plt.ylim(0, 1)
