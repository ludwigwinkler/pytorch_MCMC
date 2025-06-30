import math
import torch
import numpy as np
from scipy.integrate import quad
from numbers import Number
from dataclasses import dataclass


class EMA:
    def __init__(self, ema_weight=0.999):
        self.ema_weight = ema_weight
        self.step = 0
        self.val_ = None

    def __call__(self, val: Number):
        if self.val_ is None:
            self.val_ = val
            self.ema_correction = 1.0
        else:
            self.val_ = self.val_ * self.ema_weight + val * (1 - self.ema_weight)
            self.ema_correction = 1 - self.ema_weight ** (self.step + 1)
            self.step += 1
        return self.val_ / self.ema_correction

    @property
    def val(self):
        return self.val_


def tune(scale, acceptance):
    """Borrowed from PyMC3"""

    # Switch statement
    if acceptance < 0.001:
        # reduce by 90 percent
        scale *= 0.1
    elif acceptance < 0.05:
        # reduce by 50 percent
        scale *= 0.5
    elif acceptance < 0.2:
        # reduce by ten percent
        scale *= 0.9
    elif acceptance > 0.95:
        # increase by factor of ten
        scale *= 10.0
    elif acceptance > 0.75:
        # increase by double
        scale *= 2.0
    elif acceptance > 0.5:
        # increase by ten percent
        scale *= 1.1

    return scale


@dataclass
class RepeatedCosineDecaySchedule:
    steps: int
    cycles: int

    def __call__(self, step, min, max):
        cycle_length = self.steps // self.cycles
        cycle_pos = step % cycle_length
        cosine = 0.5 * (1 + math.cos(math.pi * cycle_pos / (cycle_length - 1)))
        return min + (max - min) * cosine


# TODO compute autocorrelation of samples with lag tau

# TODO Gelman Rubin statistic for convergence diagnostics


@dataclass
class RepeatedCosineSchedule:
    steps: int
    cycles: int
    min: float = 0.0001
    max: float = 0.01

    def __call__(self, step):
        if step < self.steps:
            cycle_pos = (step / self.steps) * self.cycles * math.pi
            cosine = 0.5 * (1 + math.cos(cycle_pos))
            return self.min + (self.max - self.min) * cosine
        else:
            return self.min


def compute_partition_function_1d(energy_fn, x_min, x_max):
    integrand = lambda x: np.exp(-energy_fn(torch.tensor(x)).item())
    Z, _ = quad(integrand, x_min, x_max)
    return Z


# def estimate_1D_partition_function(samples, energy_fn, bins=100):
#     """Estimate the partition function Z from MCMC samples using importance sampling.
#     Args:
#         samples (TensorDict): Samples from the MCMC chain.
#         energy_fn (Callable): Energy function to evaluate the samples.
#         bins (int): Number of bins for histogramming the samples.
#     Returns:
#         Z (float): Estimated partition function.
#     """
#     # counts occurences of samples in bins
#     bins = torch.histc(samples, bins=50, min=samples.min(), max=samples.max())
#     # Define bin edges and centers
#     bin_edges = torch.linspace(samples.min(), samples.max(), steps=101)
#     bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
#     bin_probs = bins / bins.sum()
#     Z_binned = (
#         torch.exp(-energy_fn(bin_centers)) * (bin_edges[1:] - bin_edges[:-1])
#     ).sum()
#     return Z_binned
