from mcmc.energy import GaussianMixture1D, Gaussian1D
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
Energy = GaussianMixture1D(
    weights=torch.tensor([0.5, 0.1, 0.25]),
    means=torch.tensor([-2.5, -0.5, 1.0]),
    stds=torch.tensor([0.5, 0.25, 0.5]),
)
# Energy = Gaussian1D(mean=-1, std=2.0)

# Create initial sample: batch of 200 chains, each with 1D x
num_chains = 200
num_steps = 3000
x_init = torch.randn(num_chains, 1)
init_sample = TensorDict({"sample": x_init}, batch_size=[num_chains])

# Vectorize the energy function using torch.func.vmap
energy_fn = torch.vmap(lambda x: Energy.energy(x), in_dims=(0,))

# Run the HMC sampler for a number of steps
Sampler = HMCSampler(step_size=0.1, num_steps=20, mass=1.0)
samples, energy = Sampler(
    sample=init_sample,
    energy_fn=energy_fn,
    steps=num_steps,
    verbose=True,
    burn_in=50,
    buffer=100,
)

data = Energy.sample(50_000)
target_mean = data.mean()
target_std = data.std()

# Make the matplotlib figure larger for better visibility
plt.figure(figsize=(8, 6))
plt.hist(
    init_sample["sample"].numpy(),
    density=True,
    bins=100,
    color="green",
    alpha=0.5,
    label="Initial Samples",
)
plt.hist(
    samples["sample"].numpy(),
    density=True,
    bins=100,
    color="blue",
    alpha=0.5,
    label="Final Samples",
)
# Plot the data samples from the target distribution for comparison
plt.hist(
    data.numpy(),
    density=True,
    bins=100,
    color="red",
    alpha=0.5,
    label="Data Samples",
)
# Plot the energy function over the range x = [-4, 4]
x_plot = torch.linspace(-4, 4, 500).unsqueeze(-1)
energy_plot = -torch.func.vmap(lambda x: Energy.log_prob(x))(x_plot)
energy_plot = energy_plot - energy_plot.min()  # Normalize to start at 0
plt.plot(
    x_plot.squeeze().numpy(),
    energy_plot.numpy(),
    color="yellow",
    label="Energy",
    linewidth=2,
)
plt.ylim(0, 2)
plt.legend()

plt.plot(x_plot.squeeze(0), energy_plot)

print(f"Target Mean: {target_mean.item():.3f}, Target Std: {target_std.item():.3f}")
print(
    f"Sample Mean: {samples['sample'].mean().item():.3f}, Sample Std: {samples['sample'].std().item():.3f}"
)
