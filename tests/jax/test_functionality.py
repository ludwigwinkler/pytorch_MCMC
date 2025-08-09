import pytest
import jax
import jax.numpy as jnp

pytest.importorskip("mcmc.jax")

from mcmc.jax.sampler import MHSampler, MALASampler, SGLDSampler
from mcmc.jax.energy import Gaussian1D, GaussianMixture1D


@pytest.mark.parametrize(
    "SamplerClass, sampler_kwargs",
    [
        (MHSampler, {"std": 0.5, "compile": True}),
        (MALASampler, {"step_size": 0.01, "dampening": 1.0, "compile": True}),
        (SGLDSampler, {"step_size": 0.01, "compile": True}),
    ],
)
@pytest.mark.parametrize(
    "EnergyClass, energy_kwargs",
    [
        (Gaussian1D, {"mean": 0.0, "std": 1.0}),
        (GaussianMixture1D, {}),
    ],
)
def test_sampler_functionality(SamplerClass, sampler_kwargs, EnergyClass, energy_kwargs):
    key = jax.random.PRNGKey(0)
    num_chains = 8
    num_steps = 5
    x = jax.random.normal(key, shape=(num_chains, 1))
    init_sample = {"sample": x}
    energy = EnergyClass(**energy_kwargs)

    def energy_fn(s):
        return energy.energy(s["sample"])  # returns (B,1)

    sampler = SamplerClass(**sampler_kwargs)
    samples, energies = sampler(
        sample=init_sample,
        energy_fn=energy_fn,
        key=key,
        steps=num_steps,
        burn_in=0,
        buffer=None,
    )

    assert samples["sample"].shape[0] == num_chains * num_steps
    assert energies.shape[0] == num_chains * num_steps


