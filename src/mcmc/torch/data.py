import torch
import numpy as np
import matplotlib.pyplot as plt


def generate_linear_regression_data(
    num_samples=100, m=1.0, b=-1.0, y_noise=1.0, x_noise=0.01, plot=False
):
    x = torch.linspace(-2, 2, num_samples).reshape(-1, 1)
    x += x_noise * torch.randn_like(x)
    y = m * x + b
    y += y_noise * torch.randn_like(y)

    if plot:
        plt.scatter(x, y)
        plt.show()

    return x, y


def generate_multimodal_linear_regression(
    num_samples, y_noise=1, x_noise=1, plot=False
):
    x1, y1 = generate_linear_regression_data(
        num_samples=num_samples // 10 * int(3),
        m=-1.0,
        b=0,
        y_noise=0.1,
        x_noise=0.1,
        plot=False,
    )
    x2, y2 = generate_linear_regression_data(
        num_samples=num_samples // 10 * int(7),
        m=2,
        b=0,
        y_noise=0.1,
        x_noise=0.1,
        plot=False,
    )

    x = torch.cat([x1, x2], dim=0).float()
    y = torch.cat([y1, y2], dim=0).float()

    if plot:
        plt.scatter(x, y, s=1)
        plt.show()

    return x, y


def generate_nonstationary_data(
    num_samples=1000,
    y_constant_noise_std=0.1,
    y_nonstationary_noise_std=1.0,
    plot=False,
):
    x = np.linspace(-0.35, 0.45, num_samples)
    x_noise = np.random.normal(0.0, 0.01, size=x.shape)

    constant_noise = np.random.normal(0, y_constant_noise_std, size=x.shape)
    std = np.linspace(0, y_nonstationary_noise_std, num_samples)  # * _y_noise_std
    non_stationary_noise = np.random.normal(loc=0, scale=std)

    y = (
        x
        + 0.3 * np.sin(2 * np.pi * (x + x_noise))
        + 0.3 * np.sin(4 * np.pi * (x + x_noise))
        + non_stationary_noise
        + constant_noise
    )

    x = torch.from_numpy(x).reshape(-1, 1).float()
    y = torch.from_numpy(y).reshape(-1, 1).float()

    x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-3)
    y = (y - y.mean(dim=0)) / (y.std(dim=0) + 1e-3)

    if plot:
        plt.scatter(x, y, s=1, alpha=0.5)
        plt.grid()
        # plt.xlim(-0.75, 1.0)
        # plt.ylim(-1.5, 1.5)
        plt.show()

    return x, y
