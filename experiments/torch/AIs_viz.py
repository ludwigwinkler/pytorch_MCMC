import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensordict import TensorDict
from tqdm import tqdm

# Set style and random seeds for reproducibility
plt.style.use("default")
torch.manual_seed(42)
np.random.seed(42)

# =============================================================================
# GLOBAL PARAMETERS - MODIFY THESE TO CHANGE DISTRIBUTIONS
# =============================================================================

# Simulation Configuration
BATCH_SIZE = 5000
N_STEPS = 10
N_PLOTS = N_STEPS

# Grid Configuration
X_MIN, X_MAX = -8, 8
GRID_POINTS = 10000
PLOT_MIN, PLOT_MAX = -6, 6
PLOT_POINTS = 1000

# Initial Distribution Parameters
INITIAL_DIST = {"type": "gaussian", "mean": 0.0, "std": 3.0, "name": "N(0,3)"}

# Target Distribution Parameters
TARGET_DIST = {
    "type": "mixture_gaussian",
    "components": [
        {"mean": -2.0, "std": 0.5, "weight": 0.6},
        {"mean": 2.0, "std": 0.5, "weight": 0.4},
    ],
    "name": "0.7×N(-2,0.5) + 0.3×N(2,0.5)",
}

# =============================================================================
# DISTRIBUTION FRAMEWORK
# =============================================================================


def create_energy_function(dist_params):
    """Create energy function from distribution parameters"""
    if dist_params["type"] == "gaussian":
        mean = dist_params["mean"]
        std = dist_params["std"]

        def gaussian_energy_fn(x):
            return 0.5 * ((x - mean) / std) ** 2 + 0.5 * np.log(2 * np.pi * std**2)

        return gaussian_energy_fn

    elif dist_params["type"] == "mixture_gaussian":
        components = dist_params["components"]

        def mixture_energy_fn(x):
            # For mixture, compute -log(p(x)) where p(x) = sum_k w_k * N(x; mu_k, sigma_k)
            weights = torch.tensor([c["weight"] for c in components])
            means = torch.tensor([c["mean"] for c in components])
            stds = torch.tensor([c["std"] for c in components])

            # Compute log probabilities for each component
            log_probs = []
            for w, mu, sigma in zip(weights, means, stds):
                log_prob_k = (
                    torch.log(w)
                    - 0.5 * ((x - mu) / sigma) ** 2
                    - 0.5 * np.log(2 * np.pi * sigma**2)
                )
                log_probs.append(log_prob_k)

            # Use log-sum-exp for numerical stability
            log_probs = torch.stack(log_probs, dim=0)
            max_log_prob = log_probs.max(dim=0)[0]
            log_sum_exp = max_log_prob + torch.log(
                torch.sum(torch.exp(log_probs - max_log_prob), dim=0)
            )

            return -log_sum_exp  # Return negative log probability (energy)

        return mixture_energy_fn

    else:
        raise ValueError(f"Unknown distribution type: {dist_params['type']}")


def create_pdf_function(dist_params):
    """Create analytical PDF function for plotting"""
    if dist_params["type"] == "gaussian":
        mean = dist_params["mean"]
        std = dist_params["std"]

        def gaussian_pdf_fn(x):
            return (1.0 / np.sqrt(2 * np.pi * std**2)) * torch.exp(
                -0.5 * ((x - mean) / std) ** 2
            )

        return gaussian_pdf_fn

    elif dist_params["type"] == "mixture_gaussian":
        components = dist_params["components"]

        def mixture_pdf_fn(x):
            pdf_total = torch.zeros_like(x)
            for comp in components:
                mean, std, weight = comp["mean"], comp["std"], comp["weight"]
                pdf_comp = (weight) * torch.exp(-0.5 * ((x - mean) / std) ** 2)
                pdf_total += pdf_comp
            return pdf_total

        return mixture_pdf_fn

    else:
        raise ValueError(f"Unknown distribution type: {dist_params['type']}")


def compute_analytical_stats(dist_params):
    """Compute analytical mean and std for distribution"""
    if dist_params["type"] == "gaussian":
        return dist_params["mean"], dist_params["std"]

    elif dist_params["type"] == "mixture_gaussian":
        components = dist_params["components"]
        # Compute mixture mean
        mean = sum(comp["weight"] * comp["mean"] for comp in components)
        # Compute mixture variance
        var = sum(
            comp["weight"] * (comp["std"] ** 2 + (comp["mean"] - mean) ** 2)
            for comp in components
        )
        return mean, np.sqrt(var)

    else:
        raise ValueError(f"Unknown distribution type: {dist_params['type']}")


# =============================================================================
# CORE SAMPLING FUNCTIONS
# =============================================================================


# Define x grid for computing partition function and sampling
x_grid = torch.linspace(X_MIN, X_MAX, GRID_POINTS)
dx = x_grid[1] - x_grid[0]

# Create energy and PDF functions from global parameters
initial_energy = create_energy_function(INITIAL_DIST)
target_energy_fn = create_energy_function(TARGET_DIST)
target_pdf = create_pdf_function(TARGET_DIST)


def interpolated_energy(x, beta):
    """Linearly interpolated energy between initial and target"""
    return (1 - beta) * initial_energy(x) + beta * target_energy_fn(x)


def compute_boltzmann_distribution(x_grid, energy_fn):
    """Compute normalized Boltzmann distribution from energy function"""
    energies = energy_fn(x_grid)
    # Convert to probabilities (Boltzmann distribution)
    log_probs = -energies
    # Normalize using log-sum-exp trick for numerical stability
    max_log_prob = log_probs.max()
    log_partition = max_log_prob + torch.log(
        torch.sum(torch.exp(log_probs - max_log_prob)) * dx
    )
    log_normalized_probs = log_probs - log_partition
    return torch.exp(log_normalized_probs)


def compute_boltzmann_distribution_for_plotting(x_values, energy_fn):
    """Compute normalized Boltzmann distribution for plotting on arbitrary x_values"""
    energies = energy_fn(x_values)
    # Convert to probabilities (Boltzmann distribution)
    log_probs = -energies
    # Normalize using log-sum-exp trick for numerical stability
    max_log_prob = log_probs.max()
    dx_plot = x_values[1] - x_values[0]  # Use the spacing of the plotting grid
    log_partition = max_log_prob + torch.log(
        torch.sum(torch.exp(log_probs - max_log_prob)) * dx_plot
    )
    log_normalized_probs = log_probs - log_partition
    return torch.exp(log_normalized_probs)


def sample_from_empirical_distribution(x_grid, probs, n_samples):
    """Sample from empirical distribution using inverse transform sampling"""
    # Create cumulative distribution
    cdf = torch.cumsum(probs * dx, dim=0)
    cdf = cdf / cdf[-1]  # Normalize to ensure CDF[-1] = 1

    # Generate uniform random samples
    u = torch.rand(n_samples)

    # Use searchsorted for inverse transform sampling
    indices = torch.searchsorted(cdf, u, right=True)
    indices = torch.clamp(indices, 0, len(x_grid) - 1)

    # Return corresponding x values with small noise for continuous appearance
    samples = x_grid[indices] + dx * (torch.rand(n_samples) - 0.5)
    return samples.unsqueeze(-1)


# =============================================================================
# MAIN EXECUTION
# =============================================================================


print("Computing Annealed Importance Sampling via Direct Boltzmann Sampling...")
print(f"Grid resolution: {len(x_grid)} points from {x_grid[0]:.1f} to {x_grid[-1]:.1f}")
print(f"Initial distribution: {INITIAL_DIST['name']}")
print(f"Target distribution: {TARGET_DIST['name']}")

# Prepare for plotting
plot_steps = torch.linspace(0, N_STEPS, N_PLOTS + 1, dtype=torch.int)[1:]  # Skip step 0
x_plot = torch.linspace(PLOT_MIN, PLOT_MAX, PLOT_POINTS)
target_density = target_pdf(x_plot)

# Storage for samples at each step
all_samples = []

# Initial samples
print("Generating initial samples...")
initial_probs = compute_boltzmann_distribution(x_grid, initial_energy)
initial_samples = sample_from_empirical_distribution(x_grid, initial_probs, BATCH_SIZE)
all_samples.append(initial_samples.clone())

# Create initial plot
# plt.figure(figsize=(12, 8))
# plt.hist(
#     initial_samples.numpy().flatten(),
#     bins=100,
#     density=True,
#     alpha=0.7,
#     color='skyblue',
#     label='Initial Samples (β=0.000)'
# )
# plt.plot(x_plot.numpy(), target_density.numpy(), 'r-', linewidth=3, alpha=0.5,
#          label=f'Target: {TARGET_DIST["name"]}')
# target_energies = target_energy_fn(x_plot) - target_energy_fn(x_plot).min()  # Normalize for plotting
# plt.plot(x_plot.numpy(), target_energies, 'g--', linewidth=2,)
# initial_density_plot = compute_boltzmann_distribution_for_plotting(x_plot, initial_energy)
# plt.plot(x_plot.numpy(), initial_density_plot.numpy(), 'g--', linewidth=2,
#          label=f'Initial: {INITIAL_DIST["name"]}')
# plt.xlabel('x')
# plt.ylabel('Density')
# plt.title('Annealed Importance Sampling - Step 0/10 (β=0.000)')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.ylim(0, 1)
# plt.xlim(PLOT_MIN, PLOT_MAX)
# sample_mean = initial_samples.mean().item()
# sample_std = initial_samples.std().item()
# plt.text(0.02, 0.98, f'Sample Stats:\nMean: {sample_mean:.3f}\nStd: {sample_std:.3f}\nN: {BATCH_SIZE}',
#          transform=plt.gca().transAxes, verticalalignment='top',
#          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
# plt.tight_layout()
# plt.show()
# print(f"Step 0: β=0.000, Sample mean={sample_mean:.3f}, Sample std={sample_std:.3f}")


def beta_schedule(step, n_steps):
    """
    Quadratic beta schedule: increases slowly at first, then faster.
    Returns beta in [0, 1] for step in [0, n_steps].
    """
    t = step / n_steps
    beta = t**5  # Quadratic interpolation
    return beta


betas = [beta_schedule(step, N_STEPS) for step in range(N_STEPS + 1)]
# Main AIS loop - sample from interpolated distributions
for step in tqdm(range(0, N_STEPS + 1), desc="AIS Steps"):
    # Linear interpolation schedule
    beta = beta_schedule(step, N_STEPS)

    # Define current interpolated energy function
    current_energy_fn = lambda x: interpolated_energy(x, beta)

    # Compute Boltzmann distribution for current β
    current_probs = compute_boltzmann_distribution(x_grid, current_energy_fn)

    # Sample from current distribution
    current_samples = sample_from_empirical_distribution(
        x_grid, current_probs, BATCH_SIZE
    )
    all_samples.append(current_samples.clone())

    # Create visualization
    plt.figure(figsize=(12, 8))

    # Plot histogram of current samples
    plt.hist(
        current_samples.numpy().flatten(),
        bins=100,
        density=True,
        alpha=0.7,
        color="skyblue",
        label=f"Samples (β={beta:.3f})",
    )

    current_energy = (
        current_energy_fn(x_plot) - current_energy_fn(x_plot).min()
    )  # Normalize for plotting

    # Plot target distribution
    plt.plot(
        x_plot.numpy(),
        target_density.numpy(),
        "r-",
        linewidth=3,
        alpha=0.5,
        label=f'Target: {TARGET_DIST["name"]}',
    )
    plt.plot(
        x_plot.numpy(),
        current_energy.numpy(),
        "g",
        linewidth=2,
        label="Current: (1-β)×Initial + β×Target",
    )

    # Plot current interpolated distribution
    current_density_plot = compute_boltzmann_distribution_for_plotting(
        x_plot, current_energy_fn
    )
    plt.plot(
        x_plot.numpy(),
        current_density_plot.numpy(),
        "g--",
        linewidth=2,
        label="Current: (1-β)×Initial + β×Target",
    )

    plt.xlabel("x")
    plt.ylabel("Density")
    plt.title(f"Annealed Importance Sampling - Step {step}/{N_STEPS} (β={beta:.3f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.xlim(PLOT_MIN, PLOT_MAX)

    # Add statistics
    sample_mean = current_samples.mean().item()
    sample_std = current_samples.std().item()
    plt.text(
        0.02,
        0.98,
        f"Sample Stats:\nMean: {sample_mean:.3f}\nStd: {sample_std:.3f}\nN: {BATCH_SIZE}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.show()

    print(
        f"Step {step}: β={beta:.3f}, Sample mean={sample_mean:.3f}, Sample std={sample_std:.3f}"
    )

# Final analysis
print("\n" + "=" * 60)
print("ANNEALED IMPORTANCE SAMPLING COMPLETED")
print("=" * 60)

final_samples = all_samples[-1]
final_mean = final_samples.mean().item()
final_std = final_samples.std().item()

# Target distribution statistics (analytical)
target_mean, target_std = compute_analytical_stats(TARGET_DIST)

print(f"Initial distribution: {INITIAL_DIST['name']}")
print(f"Target distribution: {TARGET_DIST['name']}")
print("Final sample statistics:")
print(f"  Mean: {final_mean:.4f} (target: {target_mean:.4f})")
print(f"  Std:  {final_std:.4f} (target: {target_std:.4f}")

# Final comprehensive plot
plt.figure(figsize=(15, 10))

# Plot 1: Sample evolution
plt.subplot(2, 2, 1)
colors = cm.get_cmap("viridis")(np.linspace(0, 1, len(all_samples)))
for i, samples_at_step in enumerate(all_samples):
    beta = i / N_STEPS
    plt.hist(
        samples_at_step.numpy().flatten(),
        bins=50,
        alpha=0.4,
        density=True,
        color=colors[i],
        label=f"β={beta:.2f}" if i % 2 == 0 or i == len(all_samples) - 1 else "",
    )
plt.plot(x_plot.numpy(), target_density.numpy(), "r-", linewidth=3, label="Target")
plt.xlabel("x")
plt.ylabel("Density")
plt.title("Sample Evolution During AIS")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Final samples vs target
plt.subplot(2, 2, 2)
plt.hist(
    final_samples.numpy().flatten(),
    bins=100,
    density=True,
    alpha=0.7,
    color="lightgreen",
    label="Final Samples",
)
plt.plot(
    x_plot.numpy(),
    target_density.numpy(),
    "r-",
    linewidth=3,
    label="Target Distribution",
)
plt.xlabel("x")
plt.ylabel("Density")
plt.title("Final Samples vs Target Distribution")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Beta schedule
plt.subplot(2, 2, 3)
beta_schedule = torch.linspace(0, 1, N_STEPS + 1)
plt.plot(range(N_STEPS + 1), beta_schedule.numpy(), "b-", linewidth=2, marker="o")
plt.xlabel("Step")
plt.ylabel("β (Interpolation Parameter)")
plt.title("Linear Annealing Schedule")
plt.grid(True, alpha=0.3)

# Plot 4: Energy landscape evolution
plt.subplot(2, 2, 4)
x_energy = torch.linspace(PLOT_MIN, PLOT_MAX, 200)
for i in [0, N_STEPS // 4, N_STEPS // 2, 3 * N_STEPS // 4, N_STEPS]:
    beta = i / N_STEPS
    energy_vals = interpolated_energy(x_energy, beta)
    plt.plot(x_energy.numpy(), energy_vals.numpy(), label=f"β={beta:.2f}", alpha=0.8)
plt.xlabel("x")
plt.ylabel("Energy")
plt.title("Energy Landscape Evolution")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nParameterized AIS visualization complete! 🎉")
print(
    "Modify the INITIAL_DIST and TARGET_DIST parameters at the top to explore different scenarios."
)
print(
    "This approach directly samples from exact interpolated distributions without MCMC."
)
