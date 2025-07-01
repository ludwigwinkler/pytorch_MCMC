import pytest
import torch
import numpy as np
from tensordict import TensorDict
from matplotlib import pyplot as plt

from mcmc.sampler import MHSampler, MALASampler
from mcmc.energy import Gaussian1D, GaussianMixture1D, GaussianMixture2D


@pytest.fixture
def simple_gaussian_energy():
    """Fixture for simple 1D Gaussian energy function."""
    return Gaussian1D(mean=0.0, std=1.0)


@pytest.fixture
def multivariate_gaussian_energy():
    """Fixture for 2D Gaussian energy function."""
    mean = torch.tensor([0.0, 0.0])
    cov = torch.tensor([[1.0, 0.5], [0.5, 2.0]])
    return GaussianMixture2D(mean, cov)


@pytest.fixture
def mh_sampler():
    """Fixture for MH sampler with default proposal."""
    return MHSampler()


class TestMHSampler:
    """Test suite for Metropolis-Hastings sampler."""

    @pytest.mark.parametrize(
        "mean,std",
        [
            (0.0, 1.0),
            (1.0, 0.8),
            (-1.0, 0.5),
            (1.5, 1.5),
        ],
    )
    def test_MH_gaussian1d(self, mean, std):
        # Create energy function for given mean and std
        Energy = Gaussian1D(mean=mean, std=std)

        # Create initial sample: batch of 100 chains, each with 1D x
        num_chains = 500
        num_steps = 1000
        x_init = torch.randn(num_chains, 1)
        init_sample = TensorDict({"x": x_init}, batch_size=[num_chains])

        # Vectorize the energy function using torch.func.vmap
        energy_fn = torch.vmap(lambda td: Energy.energy(td["x"]), in_dims=(0,))

        # Run the sampler for a small number of steps
        Sampler = MHSampler(std=0.5)
        samples, energy = Sampler(
            sample=init_sample, energy_fn=energy_fn, steps=num_steps, verbose=False
        )
        # Check output shapes
        # assert samples["x"].shape == (num_chains, 1)
        # assert energy.shape == (num_chains, 1)

        # Check that the mean is close to the target mean and std close to target std
        sample_mean = samples["x"].mean().item()
        sample_std = samples["x"].std().item()
        assert abs(sample_mean - mean) < 0.1
        assert abs(sample_std - std) < 0.1

    def test_MH_gaussianmixture1d(self):
        Energy = GaussianMixture1D(
            weights=torch.tensor([0.5, 0.25, 0.25]),
            means=torch.tensor([-2.5, -0.5, 0.5]),
        )
        data = Energy.sample(50_000)

        # Create initial sample: batch of 100 chains, each with 1D x
        num_chains = 500
        num_steps = 1000
        x_init = torch.randn(num_chains, 1) * 3
        init_sample = TensorDict({"x": x_init}, batch_size=[num_chains])

        # Vectorize the energy function using torch.func.vmap
        energy_fn = torch.vmap(lambda td: Energy.energy(td["x"]), in_dims=(0,))

        # Run the sampler for a small number of steps
        Sampler = MHSampler(std=1.0)
        samples, energy = Sampler(
            sample=init_sample, energy_fn=energy_fn, steps=num_steps, verbose=False
        )
        target_samples = Energy.sample(50_000)
        target_mean = target_samples.mean()
        target_std = target_samples.std()

        assert torch.allclose(samples["x"].mean(), target_mean, atol=0.1)
        assert torch.allclose(samples["x"].std(), target_std, atol=0.1)

        # plt.hist(
        #     init_sample["x"].numpy(),
        #     density=True,
        #     bins=50,
        #     color="green",
        #     alpha=0.5,
        #     label="Initial Samples",
        # )
        # plt.hist(
        #     samples["x"].numpy(),
        #     density=True,
        #     bins=50,
        #     color="blue",
        #     alpha=0.5,
        #     label="Final Samples",
        # )
        # plt.legend()

        # plt.plot(torch.linspace(-5, 5, 100), Energy.prob(torch.linspace(-5, 5, 100)))
        # plt.ylim(0, 1)


class TestMALASampler:
    """Test suite for Metropolis-Hastings sampler."""

    @pytest.mark.parametrize(
        "mean,std",
        [
            (0.0, 1.0),
            (1.0, 0.8),
            (-1.0, 0.5),
            (1.5, 1.5),
        ],
    )
    def test_MALA_gaussian1d(self, mean, std):
        # Create energy function for given mean and std
        Energy = Gaussian1D(mean=mean, std=std)

        # Create initial sample: batch of 100 chains, each with 1D x
        num_chains = 500
        num_steps = 2000
        x_init = torch.randn(num_chains, 1)
        init_sample = TensorDict({"x": x_init}, batch_size=[num_chains])

        # Vectorize the energy function using torch.func.vmap
        energy_fn = torch.vmap(lambda td: Energy.energy(td["x"]), in_dims=(0,))

        # Run the sampler for a small number of steps
        Sampler = MALASampler(step_size=0.1, dampening=1.0)
        samples, energy = Sampler(
            sample=init_sample, energy_fn=energy_fn, steps=num_steps, verbose=False
        )
        # Check output shapes
        # assert samples["x"].shape == (num_chains, 1)
        # assert energy.shape == (num_chains, 1)

        # Check that the mean is close to the target mean and std close to target std
        sample_mean = samples["x"].mean().item()
        sample_std = samples["x"].std().item()
        assert abs(sample_mean - mean) < 0.1, f"Expected mean {mean}, got {sample_mean}"
        assert abs(sample_std - std) < 0.1, f"Expected std {std}, got {sample_std}"

    def test_MALA_gaussianmixture1d(self):
        Energy = GaussianMixture1D()

        # Create initial sample: batch of 100 chains, each with 1D x
        num_chains = 500
        num_steps = 1000
        x_init = torch.randn(num_chains, 1) * 3
        init_sample = TensorDict({"x": x_init}, batch_size=[num_chains])

        # Vectorize the energy function using torch.func.vmap
        energy_fn = torch.vmap(lambda td: Energy.energy(td["x"]), in_dims=(0,))

        # Run the sampler for a small number of steps
        Sampler = MALASampler(step_size=0.1, dampening=1.0)
        samples, energy = Sampler(
            sample=init_sample, energy_fn=energy_fn, steps=num_steps, verbose=False
        )
        data = Energy.sample(50_000)
        target_mean = data.mean()
        target_std = data.std()

        assert torch.allclose(samples["x"].mean(), target_mean, atol=0.1)
        assert torch.allclose(samples["x"].std(), target_std, atol=0.1)

        # plt.hist(
        #     init_sample["x"].numpy(),
        #     density=True,
        #     bins=50,
        #     color="green",
        #     alpha=0.5,
        #     label="Initial Samples",
        # )
        # plt.hist(
        #     samples["x"].numpy(),
        #     density=True,
        #     bins=50,
        #     color="blue",
        #     alpha=0.5,
        #     label="Final Samples",
        # )
        # plt.legend()

        # plt.plot(torch.linspace(-5, 5, 100), Energy.prob(torch.linspace(-5, 5, 100)))
        # plt.ylim(0, 1)
