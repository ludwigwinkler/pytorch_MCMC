from mcmc.energy import GaussianMixture2D
from mcmc.sampler import HMCSampler
from tensordict import TensorDict
import torch
import numpy as np
import matplotlib.pyplot as plt

# Set matplotlib background to white for clarity
plt.style.use("default")
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
Energy = GaussianMixture2D()

# Create initial sample: batch of 200 chains, each with 2D x
num_chains = 200
num_steps = 1000
x_init = torch.randn(num_chains, 2)
init_sample = TensorDict({"sample": x_init}, batch_size=[num_chains])

# Vectorize the energy function using torch.func.vmap
energy_fn = torch.vmap(lambda x: Energy.energy(x), in_dims=(0,))

# Run the HMC sampler for a number of steps
Sampler = HMCSampler(step_size=0.05, num_steps=20, mass=1.0)
samples, energy = Sampler(
    sample=init_sample,
    energy_fn=energy_fn,
    steps=num_steps,
    verbose=True,
    burn_in=50,
    buffer=100,
)

x = samples["sample"][:, 0].numpy()
y = samples["sample"][:, 1].numpy()
plt.figure(figsize=(6, 5))
plt.hist2d(x, y, bins=100, density=True, cmap="viridis")
plt.colorbar(label="Density")
plt.xlabel("x")
plt.ylabel("y")
plt.title("2D Gaussian Mixture Samples")
plt.show()
