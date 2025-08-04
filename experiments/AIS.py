from scipy.integrate import quad
import copy
from mcmc.energy import GaussianMixture1D
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from tensordict import TensorDict
from mcmc.energy import Gaussian1D
from mcmc.sampler import MHSampler, SGLDSampler

# Define base and target energies (e.g., two Gaussians)
base_energy = Gaussian1D(mean=0.0, std=1.0)

target_energy = GaussianMixture1D(
    means=[-2.0, 0.0, 2.0],
    stds=[0.5, 0.5, 0.5],
    weights=[0.5, 0.2, 0.3],
)

# Annealing schedule (betas from 0 to 1)
num_intermediate = 3
betas = torch.linspace(0, 1, num_intermediate + 1)

# Create interpolated energy functions
energy_fns = []
for beta in betas:

    def make_interpolated_energy(beta):
        def energy(x):
            return (1 - beta) * base_energy.energy(x) + beta * target_energy.energy(x)

        return energy

    energy_fns.append(make_interpolated_energy(beta))

# Vectorize energy functions
energy_fns = [torch.vmap(fn, in_dims=(0,)) for fn in energy_fns]

# Initial sample from base distribution
num_chains = 500
x_init = base_energy.sample(num_chains).unsqueeze(-1)
sample = TensorDict({"sample": x_init}, batch_size=[num_chains])

# MCMC kernel (Metropolis-Hastings)
sampler = SGLDSampler(step_size=0.01)

# AIS log weights
logw = torch.zeros(num_chains)

# Run AIS
intermediate_samples = [copy.deepcopy(sample)]
for t in tqdm(range(len(betas) - 1), desc="AIS annealing"):
    # Current and next energy functions
    energy_fn_t = energy_fns[t]
    energy_fn_tp1 = energy_fns[t + 1]

    # Compute incremental weight
    # x = sample["sample"]
    # logw += -energy_fn_tp1(x).squeeze() + energy_fn_t(x).squeeze()

    # Run a few MCMC steps at the next beta
    sample_all, _ = sampler(
        sample=sample,
        energy_fn=energy_fn_tp1,
        steps=500,
        burn_in=0,
        verbose=False,
        buffer=None,
    )
    intermediate_samples.append(copy.deepcopy(sample_all))
    # Only keep the last num_chains samples for the next step
    sample["sample"] = TensorDict({"sample": sample_all["sample"]}).auto_batch_size_(1)[
        -num_chains:
    ]["sample"]


# Compute partition function via quadrature for an arbitrary energy function using scipy


def compute_partition_function(energy_fn, x_min=-10, x_max=10):
    """
    Numerically compute the partition function Z = ∫ exp(-E(x)) dx
    for a given energy function energy_fn over [x_min, x_max].
    """

    def integrand(x):
        # x is a float, so wrap in tensor for energy_fn
        x_tensor = torch.tensor([[x]], dtype=torch.float32)
        e = energy_fn(x_tensor)
        return np.exp(-e.item())

    Z, err = quad(integrand, x_min, x_max, epsabs=1e-8)
    return Z


# Example: compute partition function for base and target energy
Z_base = compute_partition_function(base_energy.energy)
Z_target = compute_partition_function(target_energy.energy)
print(f"Partition function (base): {Z_base:.6f}")
print(f"Partition function (target): {Z_target:.6f}")


print(betas)
fig, axes = plt.subplots(
    len(intermediate_samples), 1, figsize=(12, 2.5 * len(intermediate_samples))
)
for i, (beta, sample_td, energy_fn, ax) in enumerate(
    zip(betas, intermediate_samples, energy_fns, axes)
):
    # Plot histogram of samples
    x_samples = sample_td["sample"].detach().cpu().numpy().squeeze()
    ax.hist(
        x_samples,
        bins=100,
        density=True,
        alpha=0.6,
        color="blue",
        label=f"Samples (beta={beta:.2f})",
    )
    # Plot energy function
    x_plot = torch.linspace(x_samples.min() - 1, x_samples.max() + 1, 500).unsqueeze(-1)
    energy_plot = energy_fn(x_plot).detach().cpu().numpy().squeeze()
    # Convert energy to unnormalized density for visualization
    density = np.exp(-energy_plot) / compute_partition_function(energy_fn)
    # density = density / (density.max() + 1e-8) * plt.gca().get_ylim()[1] * 0.8
    ax.plot(
        x_plot.squeeze().numpy(), density, color="red", lw=2, label="Unnorm. density"
    )
    ax.set_title(f"Intermediate {i + 1} (beta={beta:.2f})")
    ax.set_xlabel("x")
    # plt.ylabel("Density / Energy")
    # plt.legend()
plt.tight_layout()
plt.show()


# Estimate log partition function (up to base Z)
logZ = torch.logsumexp(logw, dim=0) - torch.log(
    torch.tensor(num_chains, dtype=logw.dtype)
)
print(f"Estimated log partition function (up to base Z): {logZ.item():.4f}")
