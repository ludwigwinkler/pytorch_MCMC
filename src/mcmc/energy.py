import torch
import einops

__all__ = ["GaussianMixture1D", "GaussianMixture2D"]


class GaussianMixture1D:
    def __init__(self, means, stds, weights):
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
        return self.log_prob(x) + 1

    def neg_energy(self, x):
        # Negative log probability (up to constant)
        return -self.log_prob(x) + 1


means_2d = torch.tensor([[-2, -2], [-2, 2], [2, 2], [2, -2]], dtype=torch.float32)
covs_2d = einops.repeat(
    torch.tensor([[[1, 0.6], [0.6, 1]], [[1, 0.0], [0.0, 1]]], dtype=torch.float32),
    "k ... -> (2 k) ...",
)
weights_2d = torch.tensor([0.5, 0.5, 0.25, 0.75])


class GaussianMixture2D:
    def __init__(self, means=means_2d, covs=covs_2d, weights=weights_2d):
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

    # def neg_energy(self, x):
    #     return -self.log_prob(x) + 1
