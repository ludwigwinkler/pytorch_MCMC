# Enhanced AIS with Energy Function Visualization
# This version shows both probability distributions and their corresponding energy functions

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
import os
from PIL import Image
import io

plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

# Set style and random seeds for reproducibility
plt.style.use("default")
torch.manual_seed(42)
np.random.seed(42)

# =============================================================================
# GLOBAL PARAMETERS - MODIFY THESE TO CHANGE DISTRIBUTIONS
# =============================================================================

# Simulation Configuration
BATCH_SIZE = 5000  # Reduced for faster rendering
N_STEPS = 150  # Reduced for cleaner visualization

# Grid Configuration
X_MIN, X_MAX = -8, 8
GRID_POINTS = 8000
PLOT_MIN, PLOT_MAX = -6, 6
PLOT_POINTS = 1000

# Color scheme configuration - red to blue interpolation
ENERGY_ALPHA = 0.7  # Transparency for energy functions
HIST_COLOR = "blue"  # Fixed blue color for histogram samples

# Initial Distribution Parameters
INITIAL_DIST = {"type": "gaussian", "mean": 0.0, "std": 2.5, "name": "N(0,2.5)"}

# Target Distribution Parameters
TARGET_DIST = {
    "type": "mixture_gaussian",
    "components": [
        {"mean": -2.0, "std": 0.6, "weight": 0.6},
        {"mean": 2.5, "std": 0.8, "weight": 0.4},
    ],
    "name": "0.6×N(-2,0.6) + 0.4×N(2.5,0.8)",
}

# =============================================================================
# COLOR UTILITIES
# =============================================================================


def interpolate_red_to_blue(t):
    """Interpolate from red (t=0) to blue (t=1)"""
    # Red RGB: (1, 0, 0), Blue RGB: (0, 0, 1)
    red = np.array([1.0, 0.0, 0.0])
    blue = np.array([0.0, 0.0, 1.0])
    return (1 - t) * red + t * blue


def get_step_colors(step, n_steps):
    """Get colors for current step in the AIS evolution using red-to-blue interpolation"""
    t = min(max(step / n_steps, 0.0), 1.0)  # Normalized step (clipped to [0, 1])

    # Get colors from red-to-blue interpolation
    current_color = interpolate_red_to_blue(t)  # Returns RGB array
    hist_color = HIST_COLOR  # Fixed blue color for histogram
    energy_color = interpolate_red_to_blue(t)  # Same color for energy
    hist_color = interpolate_red_to_blue(
        t
    )  # Use same interpolation for histogram color
    return current_color, hist_color, energy_color


def get_source_target_colors():
    """Get the source (red) and target (blue) colors"""
    source_color = interpolate_red_to_blue(0.0)  # Red end
    target_color = interpolate_red_to_blue(1.0)  # Blue end
    return (
        source_color,
        target_color,
    )  # =============================================================================


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
            weights = torch.tensor([c["weight"] for c in components])
            means = torch.tensor([c["mean"] for c in components])
            stds = torch.tensor([c["std"] for c in components])

            log_probs = []
            for w, mu, sigma in zip(weights, means, stds):
                log_prob_k = (
                    torch.log(w)
                    - 0.5 * np.log(2 * np.pi * sigma**2)
                    - 0.5 * ((x - mu) / sigma) ** 2
                )
                log_probs.append(log_prob_k)

            log_probs = torch.stack(log_probs, dim=0)
            max_log_prob = log_probs.max(dim=0)[0]
            log_sum_exp = max_log_prob + torch.log(
                torch.sum(torch.exp(log_probs - max_log_prob), dim=0)
            )
            return -log_sum_exp

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
                pdf_comp = (weight / np.sqrt(2 * np.pi * std**2)) * torch.exp(
                    -0.5 * ((x - mean) / std) ** 2
                )
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
        mean = sum(comp["weight"] * comp["mean"] for comp in components)
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


# Define x grid for computing and sampling
x_grid = torch.linspace(X_MIN, X_MAX, GRID_POINTS)
dx = x_grid[1] - x_grid[0]

# Create energy and PDF functions
initial_energy = create_energy_function(INITIAL_DIST)
target_energy_fn = create_energy_function(TARGET_DIST)
target_pdf = create_pdf_function(TARGET_DIST)


def interpolated_energy(x, beta):
    return (1 - beta) * initial_energy(x) + beta * target_energy_fn(x)


def compute_boltzmann_distribution(x_grid, energy_fn):
    energies = energy_fn(x_grid)
    log_probs = -energies
    max_log_prob = log_probs.max()
    log_partition = max_log_prob + torch.log(
        torch.sum(torch.exp(log_probs - max_log_prob)) * dx
    )
    return torch.exp(log_probs - log_partition)


def compute_boltzmann_distribution_for_plotting(x_values, energy_fn):
    energies = energy_fn(x_values)
    log_probs = -energies
    max_log_prob = log_probs.max()
    dx_plot = x_values[1] - x_values[0]
    log_partition = max_log_prob + torch.log(
        torch.sum(torch.exp(log_probs - max_log_prob)) * dx_plot
    )
    return torch.exp(log_probs - log_partition)


def sample_from_empirical_distribution(x_grid, probs, n_samples):
    cdf = torch.cumsum(probs * dx, dim=0)
    cdf = cdf / cdf[-1]
    u = torch.rand(n_samples)
    indices = torch.searchsorted(cdf, u, right=True)
    indices = torch.clamp(indices, 0, len(x_grid) - 1)
    return (x_grid[indices] + dx * (torch.rand(n_samples) - 0.5)).unsqueeze(-1)


def save_plot_as_image():
    """Save current plot to memory buffer and return PIL Image"""
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    return img


# =============================================================================
# MAIN EXECUTION WITH ENHANCED PLOTTING
# =============================================================================


print("🔥 Enhanced AIS with Energy Function Visualization 🔥")
print(f"Initial: {INITIAL_DIST['name']} → Target: {TARGET_DIST['name']}")
print(f"Steps: {N_STEPS}, Samples: {BATCH_SIZE}")
print("Color Scheme: Red → Blue (linear interpolation), Samples: Blue")

# Prepare plotting
x_plot = torch.linspace(PLOT_MIN, PLOT_MAX, PLOT_POINTS)
target_density = target_pdf(x_plot)
all_samples = []
gif_images = []  # Store images for GIF creation

# Precompute energy values for consistent y-axis scaling
initial_energy_vals = initial_energy(x_plot)
target_energy_vals = target_energy_fn(x_plot)

# Generate and visualize each step
STEPS = (
    [0 for _ in range(10)]
    + [step for step in range(N_STEPS + 1)]
    + [N_STEPS + 1 for _ in range(10)]
)
for step in STEPS:
    beta = step / N_STEPS
    current_energy_fn = lambda x: interpolated_energy(x, beta)

    # Get colors for current step
    current_color, hist_color, energy_color = get_step_colors(step, N_STEPS)
    source_color, target_color = get_source_target_colors()

    # Sample from current distribution
    current_probs = compute_boltzmann_distribution(x_grid, current_energy_fn)
    current_samples = sample_from_empirical_distribution(
        x_grid, current_probs, BATCH_SIZE
    )
    all_samples.append(current_samples.clone())

    # Create dual-axis plot
    fig, ax1 = plt.subplots(figsize=(20, 10))

    # LEFT AXIS: Probability Distributions
    ax1.hist(
        current_samples.numpy().flatten(),
        bins=150,
        density=True,
        alpha=0.6,
        color=hist_color,
    )

    # Plot probability densities with color-coded evolution
    ax1.plot(
        x_plot.numpy(),
        target_density.numpy(),
        color=target_color,
        linestyle="-",
        linewidth=3,
    )

    current_density = compute_boltzmann_distribution_for_plotting(
        x_plot, current_energy_fn
    )
    ax1.plot(
        x_plot.numpy(),
        current_density.numpy(),
        color=current_color,
        linestyle="--",
        linewidth=3,
    )

    ax1.set_xlim(PLOT_MIN, PLOT_MAX)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.spines["left"].set_visible(False)

    # RIGHT AXIS: Energy Functions
    ax2 = ax1.twinx()

    # Plot energy functions with dotted lines
    ax2.plot(
        x_plot.numpy(),
        initial_energy_vals.numpy(),
        ":",
        color=source_color,
        linewidth=2,
        alpha=ENERGY_ALPHA,
    )
    ax2.plot(
        x_plot.numpy(),
        target_energy_vals.numpy(),
        ":",
        color=target_color,
        linewidth=2,
        alpha=ENERGY_ALPHA,
    )

    # Current interpolated energy (solid line with evolving color)
    current_energy_vals = current_energy_fn(x_plot)
    ax2.plot(
        x_plot.numpy(),
        current_energy_vals.numpy(),
        "-",
        color=energy_color,
        linewidth=2.5,
    )

    ax2.set_yticks([])
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    # Set y-limits for each axis separately
    ax1.set_ylim(0, 0.5)  # Probability density axis (left)

    # For energy axis, let's set reasonable bounds based on the energy range
    energy_min = min(
        initial_energy_vals.min().item(),
        target_energy_vals.min().item(),
        current_energy_vals.min().item(),
    )
    energy_max = max(
        initial_energy_vals.max().item(),
        target_energy_vals.max().item(),
        current_energy_vals.max().item(),
    )
    ax2.set_ylim(energy_min - 0.5, energy_max + 0.5)  # Energy axis (right)

    # Add energy label at x=-5 that rises with interpolated energy
    x_label_pos = -5.0
    # Find the energy value at x=-5 by interpolating from x_plot
    x_label_idx = torch.argmin(torch.abs(x_plot - x_label_pos))
    energy_at_x = current_energy_vals[x_label_idx].item()

    # Place the text annotation with top-right corner anchored just below the curve
    progress = min(max(step / N_STEPS, 0.0), 1.0)
    ax2.annotate(
        f"$E_{{{progress:.02f}}}$",
        xy=(x_label_pos, energy_at_x - 0.2),  # Anchor point slightly below curve
        xytext=(x_label_pos, energy_at_x - 0.2),  # Text position same as anchor
        fontsize=30,
        color=energy_color,
        ha="right",
        va="top",  # Top-right corner alignment
        weight="bold",
    )

    # Add LaTeX formula at specified position
    ax1.text(
        -3.75,
        0.45,
        r"$p_1(x) \propto e^{-E_1(x)}$",
        fontsize=30,
        color="blue",
        ha="center",
        va="center",
        weight="bold",
    )

    # Add MCMC text that rises with histogram maximum
    # Find the maximum density of the current histogram
    hist, bin_edges = np.histogram(
        current_samples.numpy().flatten(), bins=150, density=True
    )
    max_hist_density = hist.max()

    # Place text at x=-0.5 and at the height of maximum histogram density + small offset
    mcmc_label = str(f"{progress:.02f}")
    y_pos = 0.25 + progress * (0.43 - 0.25)
    x_pos = -0 + progress * (-0.5)
    ais_text = r"$x_{" + mcmc_label + r"} \sim \text{AIS}(E_{" + mcmc_label + "})$"
    ais_text = r"$x \sim \text{AIS}(E_{" + mcmc_label + "})$"
    ax1.text(
        x_pos,
        y_pos,
        ais_text,
        fontsize=30,
        color=hist_color,
        ha="center",
        va="bottom",
        weight="bold",
    )

    plt.tight_layout()

    # Save plot as image for GIF
    gif_images.append(save_plot_as_image())

    plt.show()
    plt.close(fig)  # Close figure to free memory

    # Calculate stats for print output
    mean_val = current_samples.mean().item()
    std_val = current_samples.std().item()
    print(
        f"Step {step}: β={beta:.3f}, μ={mean_val:.3f}, σ={std_val:.3f}, t={step / N_STEPS:.3f}"
    )

# Save as animated GIF after all plots are generated
if gif_images:
    # Convert to GIF with proper optimization
    gif_path = "ais_energy_evolution.gif"
    gif_images[0].save(
        gif_path,
        save_all=True,
        append_images=gif_images[1:],
        duration=5 / N_STEPS,  # 800ms per frame
        loop=0,  # Infinite loop
        optimize=True,
    )
    print(f"Animated GIF saved as: {gif_path}")
else:
    print("No images generated for GIF")

# Final summary
target_mean, target_std = compute_analytical_stats(TARGET_DIST)
final_mean = all_samples[-1].mean().item()
final_std = all_samples[-1].std().item()

print(f"\n{'=' * 70}")
print("🎯 FINAL RESULTS")
print(f"{'=' * 70}")
print(f"Target Analytics: μ={target_mean:.4f}, σ={target_std:.4f}")
print(f"Final Samples:    μ={final_mean:.4f}, σ={final_std:.4f}")
print(
    f"Error:           Δμ={abs(final_mean - target_mean):.4f}, Δσ={abs(final_std - target_std):.4f}"
)
print("\n✨ Key Insight: Energy minima → probability maxima!")
print("   Notice how low energy regions correspond to high probability regions.")
print("🎨 Color Evolution: Red (source) → Blue (target), Samples: Blue")
