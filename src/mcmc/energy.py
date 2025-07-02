from abc import abstractmethod
import torch
import einops

__all__ = ["GaussianMixture1D", "GaussianMixture2D"]


class Energy(torch.nn.Module):
    @abstractmethod
    def energy(self, **kwargs):
        """
        Compute the energy of the model at point x.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class Gaussian1D(Energy):
    def __init__(self, mean=0.0, std=1.0):
        super().__init__()
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def sample(self, num_samples=1):
        return torch.normal(self.mean, self.std, size=(num_samples,))

    def prob(self, x):
        return (
            1
            / (self.std * torch.sqrt(2 * torch.tensor(torch.pi)))
            * torch.exp(-0.5 * ((x - self.mean) / self.std) ** 2)
        )

    # def log_prob(self, x):
    # return 0.5 * ((x - self.mean) / self.std) ** 2

    @property
    def Z(self):
        # Normalization constant for the Gaussian distribution
        return self.std * torch.sqrt(2 * torch.tensor(torch.pi))

    def energy(self, x):
        # Negative log probability (up to constant)
        return 0.5 * ((x - self.mean) / self.std) ** 2


class GaussianMixture1D(Energy):
    def __init__(
        self,
        means=[-2.0, -0.5, 1.5],
        stds=[0.25, 0.25, 0.25],
        weights=[0.5, 0.3, 0.1],
    ):
        super().__init__()
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
        return -self.log_prob(x)


means_2d = torch.tensor([[-2, -2], [-2, 2], [2, 2], [2, -2]], dtype=torch.float32)
covs_2d = einops.repeat(
    torch.tensor([[[1, 0.6], [0.6, 1]], [[1, 0.0], [0.0, 1]]], dtype=torch.float32),
    "k ... -> (2 k) ...",
)
weights_2d = torch.tensor([0.5, 0.5, 0.25, 0.75])


class GaussianMixture2D(Energy):
    def __init__(self, means=means_2d, covs=covs_2d, weights=weights_2d):
        super().__init__()
        assert type(means) == type(covs) == type(weights) == torch.Tensor, (
            f"{type(means)=} {type(covs)=} {type(weights)=}"
        )
        self.means = means  # shape: (K, 2)
        self.covs = covs  # shape: (K, 2, 2)
        self.weights = weights
        self.weights = self.weights / self.weights.sum()  # Normalize weights
        self.K = self.means.shape[0]
        self.dists = [
            torch.distributions.MultivariateNormal(self.means[k], self.covs[k])
            for k in range(self.K)
        ]
        self.combined_dist = torch.distributions.MixtureSameFamily(
            torch.distributions.Categorical(self.weights),
            torch.distributions.MultivariateNormal(self.means, self.covs),
        )

    def sample(self, num_samples=1):
        component = torch.multinomial(self.weights, num_samples, replacement=True)
        samples = torch.stack([self.dists[c].sample() for c in component])
        return samples

    def prob(self, x):
        # x: (N, 2)
        x = x.unsqueeze(1)  # (N, 1, 2)
        means = self.means.unsqueeze(0)  # (1, K, 2)
        covs = self.covs.unsqueeze(0)  # (1, K, 2, 2)
        diffs = x - means  # (N, K, 2)
        inv_covs = torch.inverse(covs)  # (1, K, 2, 2)
        exponents = -0.5 * torch.einsum("nki,nkj, nkj->nk", diffs, inv_covs, diffs)
        det_covs = torch.det(covs)  # (1, K)
        norm_consts = 1.0 / (2 * torch.pi * torch.sqrt(det_covs))  # (1, K)
        probs = norm_consts * torch.exp(exponents)  # (N, K)
        weighted_probs = probs * self.weights  # (N, K)
        return weighted_probs.sum(dim=-1)  # (N,)

    def log_prob(self, x):
        # x: (N, 2)

        return self.combined_dist.log_prob(x).unsqueeze(-1)

    def energy(self, x):
        return self.log_prob(x) + 1


class LinearRegressionEnergy(Energy):
    def __init__(self, m=1.0, b=-1, sigma=1.0):
        super().__init__()
        self.m = torch.tensor(m, dtype=torch.float32)
        self.b = torch.tensor(b, dtype=torch.float32)
        self.sigma = torch.tensor(sigma, dtype=torch.float32)

    def sample(self, num_samples=100, x_range=(-5, 5)):
        x = torch.empty(num_samples, 1).uniform_(*x_range)
        noise = torch.randn(num_samples, 1) * self.sigma
        y = self.m * x + self.b + noise
        return x, y

    def energy(self, x, y):
        # Negative log-likelihood for Gaussian noise
        pred = self.m * x + self.b
        return 0.5 * ((y - pred) / self.sigma) ** 2

    def log_prob(self, x, y):
        pred = self.m * x + self.b
        return -0.5 * ((y - pred) / self.sigma) ** 2 - torch.log(
            self.sigma * torch.sqrt(2 * torch.tensor(torch.pi))
        )
