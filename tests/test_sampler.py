import pytest
import torch
import numpy as np
from tensordict import TensorDict
from matplotlib import pyplot as plt
from mcmc.sampler import MHSampler, MALASampler, SGLDSampler
from mcmc.energy import (
    Gaussian1D,
    GaussianMixture1D,
    GaussianMixture2D,
    NeuralNetworkEnergy,
)

import copy
from mcmc.data import generate_nonstationary_data


@pytest.fixture(autouse=True)
def set_seed():
    """Automatically set seed before each test."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Also set random seed for Python's random module if needed
    import random

    random.seed(seed)


@pytest.fixture
def simple_gaussian_energy():
    """Fixture for simple 1D Gaussian energy function."""
    return Gaussian1D(mean=0.0, std=1.0)


@pytest.fixture
def multivariate_gaussian_energy():
    """Fixture for 2D Gaussian energy function."""
    mean = torch.tensor([0.0, 0.0])
    cov = torch.tensor([[1.0, 0.5], [0.5, 2.0]])
    weights = torch.tensor([0.5, 0.5])
    return GaussianMixture2D(weights=weights, means=mean, covs=cov)


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
        init_sample = TensorDict({"sample": x_init}, batch_size=[num_chains])

        # Vectorize the energy function using torch.func.vmap
        energy_fn = torch.vmap(lambda x: Energy.energy(x), in_dims=(0,))

        # Run the sampler for a small number of steps
        Sampler = MHSampler(std=0.5)
        samples, energy = Sampler(
            sample=init_sample, energy_fn=energy_fn, steps=num_steps, verbose=False
        )
        # Check output shapes
        # assert samples["x"].shape == (num_chains, 1)
        # assert energy.shape == (num_chains, 1)

        # Check that the mean is close to the target mean and std close to target std
        sample_mean = samples["sample"].mean().item()
        sample_std = samples["sample"].std().item()
        assert abs(sample_mean - mean) < 0.1
        assert abs(sample_std - std) < 0.1

    def test_MH_gaussianmixture1d(self):
        Energy = GaussianMixture1D(
            weights=torch.tensor([0.5, 0.25, 0.25]),
            means=torch.tensor([-2.5, -0.5, 0.5]),
        )

        # Create initial sample: batch of 100 chains, each with 1D x
        num_chains = 500
        num_steps = 2000
        x_init = torch.randn(num_chains, 1)
        init_sample = TensorDict({"sample": x_init}, batch_size=[num_chains])

        # Vectorize the energy function using torch.func.vmap
        energy_fn = torch.vmap(lambda x: Energy.energy(x), in_dims=(0,))

        # Run the sampler for a small number of steps
        Sampler = MHSampler(std=0.5)
        samples, energy = Sampler(
            sample=init_sample, energy_fn=energy_fn, steps=num_steps, verbose=False
        )
        target_samples = Energy.sample(50_000)
        target_mean = target_samples.mean()
        target_std = target_samples.std()

        assert torch.allclose(samples["sample"].mean(), target_mean, atol=0.1)
        assert torch.allclose(samples["sample"].std(), target_std, atol=0.1)

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

    def test_MH_gaussianmixture2d(self):
        """Test MH sampler on 2D Gaussian mixture."""
        Energy = GaussianMixture2D()

        num_chains = 500
        num_steps = 2000
        buffer = 50  # Default buffer size
        x_init = 3 * torch.randn((num_chains, 2)).clamp(-5, 5)
        init_sample = TensorDict({"sample": x_init}, batch_size=[num_chains])

        energy_fn = torch.vmap(Energy.energy, (0,))

        Sampler = MHSampler(std=1.0)
        samples, energy = Sampler(
            sample=init_sample,
            energy_fn=energy_fn,
            steps=num_steps,
            verbose=False,
            buffer=buffer,
        )

        # Compare with target distribution
        # target_samples = Energy.sample(50_000)

        # Check that sample statistics are reasonable
        # The sampler returns buffer * num_chains samples
        expected_samples = buffer * num_chains
        assert samples["sample"].shape == (expected_samples, 2)
        assert energy.shape == (expected_samples, 1)  # Energy has an extra dimension

        # Check that samples are within reasonable bounds
        assert samples["sample"].min() > -10
        assert samples["sample"].max() < 10


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
        ids=[
            "mean=0.0, std=1.0",
            "mean=1.0, std=0.8",
            "mean=-1.0, std=0.5",
            "mean=1.5, std=1.5",
        ],
    )
    @pytest.mark.parametrize(
        "step_size, dampening",
        [
            (0.05, 1.0),
            (0.1, 1.0),
            (0.5, 1.0),
        ],
        ids=[
            "step_size=0.05, dampening=1.0",
            "step_size=0.1, dampening=1.0",
            "step_size=0.5, dampening=1.0",
        ],
    )
    def test_MALA_gaussian1d(self, mean, std, step_size, dampening):
        # Create energy function for given mean and std
        Energy = Gaussian1D(mean=mean, std=std)

        # Create initial sample: batch of 100 chains, each with 1D x
        num_chains = 200
        num_steps = 1000
        x_init = torch.randn(num_chains, 1)
        init_sample = TensorDict({"sample": x_init}, batch_size=[num_chains])

        # Vectorize the energy function using torch.func.vmap
        energy_fn = torch.vmap(lambda x: Energy.energy(x), in_dims=(0,))

        # Run the sampler for a small number of steps
        Sampler = MALASampler(step_size=step_size, dampening=dampening)
        samples, energy = Sampler(
            sample=init_sample,
            energy_fn=energy_fn,
            steps=num_steps,
            verbose=False,
            buffer=200,
            burn_in=0,
        )
        # Check output shapes
        # assert samples["x"].shape == (num_chains, 1)
        # assert energy.shape == (num_chains, 1)

        # Check that the mean is close to the target mean and std close to target std
        sample_mean = samples["sample"].mean().item()
        sample_std = samples["sample"].std().item()
        assert abs(sample_mean - mean) < 0.1, f"Expected mean {mean}, got {sample_mean}"
        assert abs(sample_std - std) < 0.1, f"Expected std {std}, got {sample_std}"

    def test_MALA_gaussianmixture1d(self):
        Energy = GaussianMixture1D(
            weights=torch.tensor([0.5, 0.1, 0.25]),
            means=torch.tensor([-2.5, -0.5, 1.0]),
            stds=torch.tensor([0.5, 0.5, 0.5]),
        )

        # Create initial sample: batch of 100 chains, each with 1D x
        num_chains = 500
        num_steps = 2000
        x_init = torch.randn(num_chains, 1)
        init_sample = TensorDict({"sample": x_init}, batch_size=[num_chains])

        # Vectorize the energy function using torch.func.vmap
        energy_fn = torch.vmap(lambda x: Energy.energy(x), in_dims=(0,))

        # Run the sampler for a small number of steps
        Sampler = MALASampler(step_size=1.0, dampening=1.0)
        samples, energy = Sampler(
            sample=init_sample, energy_fn=energy_fn, steps=num_steps, verbose=False
        )
        data = Energy.sample(50_000)
        target_mean = data.mean()
        target_std = data.std()

        plt.hist(
            init_sample["sample"].numpy(),
            density=True,
            bins=100,
            color="green",
            alpha=0.5,
            label="Initial Samples",
        )
        plt.hist(
            samples["sample"].numpy(),
            density=True,
            bins=100,
            color="blue",
            alpha=0.5,
            label="Final Samples",
        )
        # Plot the data samples from the target distribution for comparison
        plt.hist(
            data.numpy(),
            density=True,
            bins=100,
            color="red",
            alpha=0.5,
            label="Data Samples",
        )
        plt.legend()

        assert torch.allclose(samples["sample"].mean(), target_mean, atol=0.1)
        assert torch.allclose(samples["sample"].std(), target_std, atol=0.1)

        # plt.plot(torch.linspace(-5, 5, 100), Energy.prob(torch.linspace(-5, 5, 100)))
        # plt.ylim(0, 1)

    def test_MALA_gaussianmixture2d(self):
        """Test MALA sampler on 2D Gaussian mixture."""
        Energy = GaussianMixture2D()

        num_chains = 1000
        num_steps = 2000
        buffer = 50  # Default buffer size
        x_init = 3 * torch.randn((num_chains, 2)).clamp(-5, 5)
        init_sample = TensorDict({"sample": x_init}, batch_size=[num_chains])

        energy_fn = torch.vmap(Energy.energy, (0,))

        Sampler = MALASampler(step_size=1.0, dampening=1.0)
        samples, energy = Sampler(
            sample=init_sample,
            energy_fn=energy_fn,
            steps=num_steps,
            verbose=False,
            buffer=buffer,
        )

        # Compare with target distribution
        # target_samples = Energy.sample(50_000)

        # Check that sample statistics are reasonable
        # The sampler returns buffer * num_chains samples
        expected_samples = buffer * num_chains
        assert samples["sample"].shape == (expected_samples, 2)
        assert energy.shape == (expected_samples, 1)  # Energy has an extra dimension

        # Check that samples are within reasonable bounds
        # assert samples["sample"].min() > -10
        # assert samples["sample"].max() < 10

        # x = samples["sample"][:, 0].numpy()
        # y = samples["sample"][:, 1].numpy()
        # plt.figure(figsize=(6, 5))
        # plt.hist2d(x, y, bins=100, density=True, cmap="viridis")
        # plt.colorbar(label="Density")
        # plt.xlabel("x")
        # plt.ylabel("y")
        # plt.title("2D Gaussian Mixture Samples")
        # plt.show()

    def test_mala_neuralnetwork(self):
        x, y = generate_nonstationary_data(
            num_samples=1_000,
            plot=False,
            y_nonstationary_noise_std=0.3,
            y_constant_noise_std=0.01,
        )
        probmodel = NeuralNetworkEnergy()
        num_chains = 11
        models = [copy.deepcopy(probmodel) for _ in range(num_chains)]
        params, buffers = torch.func.stack_module_state(models)
        init_samples = TensorDict(
            {
                "sample": params,
                "buffers": buffers,
                "data": x,
                "target": y,
                "aux": "abc",
            },  # type: ignore
        )
        energy = lambda params, buffers, data, target, aux: NeuralNetworkEnergy.energy(
            probmodel.train(), params, buffers, data, target
        )
        vmap_energy = torch.vmap(
            energy, (0, 0, None, None, None), randomness="different"
        )
        Sampler = MALASampler(step_size=0.001, dampening=0.01)
        samples, energies = Sampler(
            sample=init_samples,
            energy_fn=vmap_energy,
            steps=1_000,
            verbose=False,
            buffer=50,
            burn_in=20,
        )
        assert (
            energies.mean().item() < 0.3
        ), f"Mean energy too high: {energies.mean().item()}"


class TestSGLDSampler:
    """Test suite for SGLD sampler."""

    @pytest.mark.parametrize(
        "mean,std,step_size",
        [
            (0.0, 1.0, 0.1),
            (1.0, 0.8, 0.1),
            (-1.0, 0.5, 0.1),
            (0.5, 0.25, 0.01),
        ],
    )
    def test_sgld_gaussian1d(self, mean, std, step_size):
        Energy = Gaussian1D(mean=mean, std=std)
        num_chains = 200
        num_steps = 2000
        x_init = torch.randn(num_chains, 1)
        init_sample = TensorDict({"sample": x_init}, batch_size=[num_chains])
        energy_fn = torch.vmap(lambda x: Energy.energy(x), in_dims=(0,))
        sampler = SGLDSampler(step_size=step_size)
        samples, energy = sampler(
            sample=init_sample,
            energy_fn=energy_fn,
            steps=num_steps,
            verbose=True,
            buffer=100,
            burn_in=0,
        )
        # generate a histogram of the data samples and the samples from the MCMC sampler
        # data = Energy.sample(50_000)

        # plt.hist(data.numpy(), bins=50, alpha=0.5, label="Data", density=True)
        # plt.hist(
        #     samples["x"].numpy(), bins=50, alpha=0.5, label="MCMC Samples", density=True
        # )
        # plt.legend()
        # plt.show()
        # print(samples["x"].mean(), samples["x"].std())

        assert abs(samples["sample"].mean() - mean) < 0.1
        assert abs(samples["sample"].std() - std) < 0.1

    def test_sgld_gaussian_mixture(self):
        Energy = GaussianMixture1D(
            weights=torch.tensor([0.5, 0.1, 0.25]),
            means=torch.tensor([-2.5, -0.5, 1.0]),
            stds=torch.tensor([0.5, 0.25, 0.5]),
        )
        num_chains = 200
        num_steps = 3000
        x_init = torch.randn(num_chains, 1) * 3
        init_sample = TensorDict({"sample": x_init}, batch_size=[num_chains])
        energy_fn = torch.vmap(lambda x: Energy.energy(x), in_dims=(0,))
        sampler = SGLDSampler(step_size=0.01)
        samples, energy = sampler(
            sample=init_sample,
            energy_fn=energy_fn,
            steps=num_steps,
            verbose=False,
            burn_in=50,
            buffer=200,
        )
        data = Energy.sample(50_000)
        target_mean = data.mean()
        target_std = data.std()

        # # generate a histogram of the data samples and the samples from the MCMC sampler
        # plt.hist(data.numpy(), bins=50, alpha=0.5, label="Data", density=True)
        # plt.hist(
        #     samples["x"].numpy(), bins=50, alpha=0.5, label="MCMC Samples", density=True
        # )
        # plt.legend()
        # plt.show()

        assert torch.allclose(samples["sample"].mean(), target_mean, atol=0.1)
        assert torch.allclose(samples["sample"].std(), target_std, atol=0.1)

    def test_sgld_gaussianmixture2d(self):
        """Test SGLD sampler on 2D Gaussian mixture."""
        Energy = GaussianMixture2D()

        num_chains = 500
        num_steps = 2000
        buffer = 50  # Default buffer size
        x_init = 3 * torch.randn((num_chains, 2)).clamp(-5, 5)
        init_sample = TensorDict({"sample": x_init}, batch_size=[num_chains])

        energy_fn = torch.vmap(Energy.energy, (0,))

        sampler = SGLDSampler(step_size=0.05)
        samples, energy = sampler(
            sample=init_sample,
            energy_fn=energy_fn,
            steps=num_steps,
            verbose=False,
            buffer=buffer,
        )

        # Compare with target distribution
        # target_samples = Energy.sample(50_000)

        # Check that sample statistics are reasonable
        # The sampler returns buffer * num_chains samples
        expected_samples = buffer * num_chains
        assert samples["sample"].shape == (expected_samples, 2)
        assert energy.shape == (expected_samples, 1)  # Energy has an extra dimension

        # Check that samples are within reasonable bounds
        assert samples["sample"].min() > -10
        assert samples["sample"].max() < 10

    def test_sgld_neuralnetwork(self):
        x, y = generate_nonstationary_data(
            num_samples=1_000,
            plot=False,
            y_nonstationary_noise_std=0.3,
            y_constant_noise_std=0.01,
        )
        probmodel = NeuralNetworkEnergy()
        num_chains = 11
        models = [copy.deepcopy(probmodel) for _ in range(num_chains)]
        params, buffers = torch.func.stack_module_state(models)
        init_samples = TensorDict(
            {
                "sample": params,
                "buffers": buffers,
                "data": x,
                "target": y,
                "aux": "abc",
            },  # type: ignore
        )
        energy = lambda params, buffers, data, target, aux: NeuralNetworkEnergy.energy(
            probmodel.train(), params, buffers, data, target
        )
        vmap_energy = torch.vmap(
            energy, (0, 0, None, None, None), randomness="different"
        )
        Sampler = SGLDSampler(step_size=0.001, dampening=0.01)
        samples, energies = Sampler(
            sample=init_samples,
            energy_fn=vmap_energy,
            steps=3_000,
            verbose=False,
            buffer=50,
            burn_in=20,
        )
        assert (
            energies.mean().item() < 0.3
        ), f"Mean energy too high: {energies.mean().item()}"
