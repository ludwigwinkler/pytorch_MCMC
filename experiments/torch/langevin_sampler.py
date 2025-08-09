import torch
import matplotlib.pyplot as plt
import numpy as np


def sgl_sampler(energy_fn, x_init, lr=1e-2, n_steps=1000, noise_scale=1.0):
    """
    Stochastic Gradient Langevin Dynamics (SGLD) sampler for 2D.
    Args:
        energy_fn: Callable, computes energy for input x (requires_grad=True).
        x_init: Initial sample (torch.tensor, shape (2,), requires_grad=False).
        lr: Learning rate (step size).
        n_steps: Number of sampling steps.
        noise_scale: Multiplier for injected noise (default 1.0).
    Returns:
        samples: Tensor of shape (n_steps, 2).
    """
    x = x_init.clone().detach().requires_grad_(True)
    samples = []
    for _ in range(n_steps):
        x.requires_grad_(True)
        energy = energy_fn(x)
        grad = torch.autograd.grad(energy, x)[0]
        noise = torch.randn_like(x) * (lr**0.5) * noise_scale
        x = (x - 0.5 * lr * grad + noise).detach()
        samples.append(x.detach().cpu().numpy())
    return np.stack(samples)


def double_well_energy(x):
    # x: shape (..., 2)
    # Double well in 2D: sum of two 1D double wells
    return 0.25 * ((x[0] ** 2 - 1) ** 2 + (x[1] ** 2 - 1) ** 2)


def test_sgld_on_2d():
    # Initial point
    x_init = torch.tensor([2.0, -2.0], dtype=torch.float32)
    samples = sgl_sampler(
        double_well_energy, x_init, lr=1e-2, n_steps=5000, noise_scale=1.0
    )
    samples = np.array(samples)

    # Plot samples
    plt.figure(figsize=(8, 8))
    plt.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.2, label="SGLD Samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("SGLD Sampling on 2D Double Well Energy Surface")

    # Plot energy contours
    x = np.linspace(-2.5, 2.5, 100)
    y = np.linspace(-2.5, 2.5, 100)
    X, Y = np.meshgrid(x, y)
    Z = 0.25 * ((X**2 - 1) ** 2 + (Y**2 - 1) ** 2)
    plt.contour(X, Y, Z, levels=20, cmap="viridis", alpha=0.7)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_sgld_on_2d()
