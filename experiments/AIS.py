import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import to_rgba
from tqdm import tqdm
from scipy.integrate import quad
import numpy as np


class Normal1D:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def sample(self, num_samples=1):
        return torch.normal(self.mean, self.std, size=(num_samples,))

    def prob(self, x):
        var = self.std**2
        return (
            1
            / (self.std * torch.sqrt(2 * torch.tensor(torch.pi)))
            * torch.exp(-0.5 * (x - self.mean) ** 2 / var)
        )

    def energy(self, x):
        return 0.5 * (x - self.mean) ** 2 / self.std**2

    @property
    def Z(self):
        """Partition function for the normal distribution Z = (sqrt(2 * pi * std**2)"""
        return self.std * torch.sqrt(2 * torch.tensor(torch.pi))

    def log_prob(self, x):
        var = self.std**2
        return (
            -0.5 * torch.log(2 * torch.tensor(torch.pi) * var)
            - 0.5 * (x - self.mean) ** 2 / var
        )


class GaussianMixture1D:
    def __init__(self, means, stds, weights):
        self.means = torch.tensor(means, dtype=torch.float32)
        self.stds = torch.tensor(stds, dtype=torch.float32)
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.weights = self.weights / self.weights.sum()  # Normalize weights

    def sample(self, num_samples=1):
        component = torch.multinomial(self.weights, num_samples, replacement=True)
        samples = torch.normal(self.means[component], self.stds[component])
        return samples

    def prob(self, x):
        x = x.unsqueeze(-1)  # Shape (N, 1)
        probs = (
            1
            / (self.stds * torch.sqrt(2 * torch.tensor(torch.pi)))
            * torch.exp(-0.5 * (x - self.means) ** 2 / (self.stds**2))
        )
        weighted_probs = probs * self.weights
        return weighted_probs.sum(dim=-1)

    def log_prob(self, x):
        x = x.unsqueeze(-1)
        log_probs = (
            -0.5 * torch.log(2 * torch.tensor(torch.pi) * self.stds**2)
            - 0.5 * (x - self.means) ** 2 / (self.stds**2)
            + torch.log(self.weights)
        )
        return torch.logsumexp(log_probs, dim=-1)

    def energy(self, x):
        # Negative log probability (up to constant)
        return -self.log_prob(x) + 1


def sgl_sampler(energy_fn, x_init, lr=1e-2, n_steps=1000, noise_scale=1.0):
    """
    Stochastic Gradient Langevin Dynamics (SGLD) sampler.
    Args:
        energy_fn: Callable, computes energy for input x (requires_grad=True).
        x_init: Initial sample (torch.tensor, requires_grad=False).
        lr: Learning rate (step size).
        n_steps: Number of sampling steps.
        noise_scale: Multiplier for injected noise (default 1.0).
    Returns:
        samples: Tensor of shape (n_steps,).
    """
    x = x_init.clone().detach().requires_grad_(True)
    samples = []
    pbar = tqdm(range(n_steps))
    for _ in pbar:
        lr_t = lr * 0.5 * (1 + torch.cos(torch.tensor(_ / n_steps * torch.pi)))
        pbar.set_description(f"lr={lr_t:.5f}")
        x.requires_grad_(True)
        energy = energy_fn(x)
        grad = torch.autograd.grad(energy.sum(), x)[0]
        noise = torch.randn_like(x) * lr**0.5
        lr_t = lr / 2 * (1 + torch.cos(torch.tensor(_ / n_steps * torch.pi)))
        x = (x - lr_t * grad + noise).detach()
        samples.append(x.detach().clone())
    return x


def metropolishastings_sampler(energy_fn, x_init, n_steps=1000, proposal_std=0.5):
    """
    Metropolis-Hastings sampler.
    Args:
        energy_fn: Callable, computes energy for input x (requires_grad=True).
        x_init: Initial sample (torch.tensor, requires_grad=False).
        n_steps: Number of sampling steps.
        proposal_std: Standard deviation for the proposal distribution.
    Returns:
        samples: Tensor of shape (n_steps,).
    """
    x = x_init
    samples = []
    # progress_bar = tqdm(total=n_steps, desc="MH")
    acceptance_ratio_ema, ema_weight = None, 0.9999
    pbar = tqdm(range(n_steps), desc="MH")
    for _ in pbar:
        x_new = x + torch.randn_like(x) * proposal_std
        log_acceptance_ratio = energy_fn(x) - energy_fn(x_new)
        acceptance_ratio = torch.exp(log_acceptance_ratio)
        accept = torch.rand_like(acceptance_ratio) < acceptance_ratio

        # Debiased running average (corrects for initial bias)
        if acceptance_ratio_ema is None:
            acceptance_ratio_ema = accept.int().float().mean()
            ema_correction = 1.0
        else:
            acceptance_ratio_ema = (
                acceptance_ratio_ema * ema_weight
                + accept.int().float().mean() * (1 - ema_weight)
            )
            ema_correction = 1 - ema_weight ** (_ + 1)
        debiased_acceptance = acceptance_ratio_ema / ema_correction
        pbar.set_postfix({"Accept": float(acceptance_ratio_ema.mean())})
        x = torch.where(accept, x_new, x)
        samples.append(x.detach().clone())
    return x


def compute_partition_function_1d(energy_fn, x_min, x_max):
    integrand = lambda x: np.exp(-energy_fn(torch.tensor(x)).item())
    Z, _ = quad(integrand, x_min, x_max)
    return Z


p0 = Normal1D(torch.tensor(0.0), torch.tensor(5.0))

N = 10_000
samples = p0.sample(N)
prob = p0.prob(samples)
log_prob = p0.log_prob(samples)
Z = p0.Z

p1 = Normal1D(torch.tensor(0.0), torch.tensor(1.0))
print(p0.Z, p1.Z)

log_w = -p1.energy(samples) - p0.log_prob(samples)
logZ1 = torch.logsumexp(log_w, dim=0) - torch.log(torch.tensor(N, dtype=torch.float32))
print(f"Estimated Z1: {torch.exp(logZ1)}")
print(f"True Z1: {p1.Z}")

samples = samples[samples < 3]
samples = samples[samples > -3]
logprob1 = -logZ1 - p1.energy(samples)
prob1 = torch.exp(logprob1)

plt.figure(figsize=(8, 4))
plt.hist(
    samples.numpy(),
    bins=50,
    weights=prob1.numpy(),
    density=True,
    alpha=0.6,
    label="Weighted Histogram",
)
# sns.kdeplot(samples.numpy(), weights=prob1.numpy(), color='red', label='KDE')
plt.title("Histogram of prob1 at sample locations")
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.legend()
plt.show()

gmm = GaussianMixture1D(
    means=[-3.0, 0.0, 3.0],
    stds=[0.5, 0.5, 0.5],
    weights=[2, 0.3, 0.1],
)
x = torch.linspace(-5, 5, 200)


plt.figure(figsize=(8, 4))
plt.plot(
    x.numpy(),
    p1.energy(x).numpy(),
    label="True Normal(0, 1) Energy",
    color="blue",
)
plt.plot(
    x.numpy(),
    gmm.energy(x).numpy(),
    label="GMM Energy",
    color="orange",
)

# Define colors for interpolation
color1 = to_rgba("blue")
color2 = to_rgba("orange")

for t in [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0]:
    # Linear interpolation in color space
    interp_color = tuple((1 - t) * c1 + t * c2 for c1, c2 in zip(color1, color2))
    energy_t = gmm.energy(x) ** t * p1.energy(x) ** (1 - t)
    plt.plot(
        x.numpy(),
        energy_t.numpy(),
        label=f"t={t:.2f}",
        linestyle="--",
        color=interp_color,
    )

plt.title("Energy Interpolation between Normal and GMM")
plt.xlabel("x")
plt.ylabel("Energy")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()

t = 0.0
energy_t = lambda x: gmm.energy(x) ** t * p1.energy(x) ** (1 - t)
energy_t = lambda x: p1.energy(x)
samples = metropolishastings_sampler(
    energy_fn=energy_t, x_init=torch.randn(5_000), n_steps=2_000, proposal_std=0.1
)

print(f"{samples.shape=}")
# Count unique samples and their frequencies
log_weights = -energy_t(samples)
bins = torch.histc(samples, bins=50, min=samples.min(), max=samples.max())
bin_edges = torch.linspace(samples.min(), samples.max(), steps=101)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
bin_probs = bins / bins.sum()
Z_binned = (torch.exp(-energy_t(bin_centers)) * (bin_edges[1:] - bin_edges[:-1])).sum()
print(f"Z_binned: {Z_binned}")


# Weight by frequency of each unique sample
# logZ_est = torch.logsumexp(log_weights, dim=0) - torch.log(
#     torch.tensor(len(samples), dtype=torch.float32)
# )

Z_est = torch.exp(log_weights).sum() / len(samples)
print(
    f"Estimated partition function Z_t: {Z_est.item()} vs {compute_partition_function_1d(energy_t, -4, 4)}"
)

_ = plt.hist(
    samples.detach().numpy(),
    bins=100,
    density=True,
    alpha=0.6,
    label="SGLD Samples",
)
# plt.plot(x.numpy(), gmm.prob(x).numpy(), label="GMM PDF", color="orange")
# plt.plot(x.numpy(), energy_t(x).numpy(), label="Normal PDF", color="blue")
