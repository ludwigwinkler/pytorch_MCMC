import pytest
import torch
from tensordict import TensorDict
from mcmc.sampler import MHSampler, MALASampler, SGLDSampler
from mcmc.energy import Gaussian1D, GaussianMixture1D, Gaussian1DwithTemperature


@pytest.mark.parametrize(
    "SamplerClass, sampler_kwargs",
    [
        (MHSampler, {"std": 0.5}),
        (MALASampler, {"step_size": 0.01, "dampening": 1.0}),
        (SGLDSampler, {"step_size": 0.01}),
    ],
)
@pytest.mark.parametrize(
    "EnergyClass, energy_kwargs",
    [
        (Gaussian1D, {"mean": 0.0, "std": 1.0}),
        (GaussianMixture1D, {}),
    ],
)
def test_sampler_functionality(
    SamplerClass, sampler_kwargs, EnergyClass, energy_kwargs
):
    torch.manual_seed(42)
    num_chains = 8
    num_steps = 5
    x = torch.randn(num_chains, 1)
    init_sample = TensorDict({"x": x}, batch_size=[num_chains])
    energy = EnergyClass(**energy_kwargs)
    energy_fn = torch.vmap(lambda x: energy.energy(x), in_dims=(0,))
    sampler = SamplerClass(**sampler_kwargs)
    samples, energies = sampler(
        sample=init_sample,
        energy_fn=energy_fn,
        steps=num_steps,
        burn_in=0,
        verbose=False,
        buffer=None,  # Store all samples
    )
    # Check output types and shapes
    assert isinstance(samples, TensorDict)
    assert isinstance(energies, torch.Tensor)
    assert samples.batch_size[0] == num_chains * num_steps
    assert energies.shape[0] == num_chains * num_steps


@pytest.mark.parametrize(
    "SamplerClass, sampler_kwargs",
    [
        (MHSampler, {"std": 0.5}),
        (MALASampler, {"step_size": 0.01, "dampening": 1.0}),
        (SGLDSampler, {"step_size": 0.01}),
    ],
)
@pytest.mark.parametrize(
    "EnergyClass, energy_kwargs",
    [
        (Gaussian1DwithTemperature, {"mean": 0.0, "std": 1.0}),
    ],
)
def test_sampler_functionality_with_extra_inputs(
    SamplerClass, sampler_kwargs, EnergyClass, energy_kwargs
):
    torch.manual_seed(42)
    num_chains = 8
    num_steps = 5
    x = torch.randn(num_chains, 1)
    T = torch.randn(num_chains, 1).abs() + 1
    init_sample = TensorDict({"x": x, "T": T, "other": "abc"})
    energy = EnergyClass(**energy_kwargs)
    energy_fn = torch.vmap(
        lambda x, T, other: energy.energy(x, T, other), in_dims=(0, 0, None)
    )
    sampler = SamplerClass(**sampler_kwargs)
    samples, energies = sampler(
        sample=init_sample,
        energy_fn=energy_fn,
        steps=num_steps,
        burn_in=0,
        verbose=False,
        buffer=None,  # Store all samples
    )
    # Check output types and shapes
    assert isinstance(samples, TensorDict)
    assert isinstance(energies, torch.Tensor)
    assert samples.batch_size[0] == num_chains * num_steps
    assert energies.shape[0] == num_chains * num_steps
