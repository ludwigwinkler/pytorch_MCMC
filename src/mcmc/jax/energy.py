from __future__ import annotations

import jax
import jax.numpy as jnp


class Energy:
    def energy(self, *args, **kwargs):  # pragma: no cover - abstract-like
        raise NotImplementedError


class Gaussian1D(Energy):
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = jnp.asarray(mean, dtype=jnp.float32)
        self.std = jnp.asarray(std, dtype=jnp.float32)

    def sample(self, key: jax.Array, num_samples: int = 1) -> jax.Array:
        return self.mean + self.std * jax.random.normal(key, shape=(num_samples, 1))

    def log_prob(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x)
        return -0.5 * jnp.log(2 * jnp.pi * self.std**2) - 0.5 * (
            (x - self.mean) / self.std
        ) ** 2

    def energy(self, x: jax.Array) -> jax.Array:
        e = 0.5 * ((x - self.mean) / self.std) ** 2
        e = jnp.asarray(e)
        if e.ndim == 1:
            e = e[:, None]
        return e


class GaussianMixture1D(Energy):
    def __init__(
        self,
        means=( -2.0, -0.5, 1.5 ),
        stds=( 0.25, 1.0, 0.25 ),
        weights=( 0.2, 0.1, 0.1 ),
    ):
        means = jnp.asarray(means, dtype=jnp.float32)
        stds = jnp.asarray(stds, dtype=jnp.float32)
        weights = jnp.asarray(weights, dtype=jnp.float32)
        self.means = means
        self.stds = stds
        self.weights = weights / jnp.sum(weights)

    def log_prob(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x)
        x_ = x[..., None]
        log_comp = -0.5 * jnp.log(2 * jnp.pi * self.stds**2) - 0.5 * (
            (x_ - self.means) / self.stds
        ) ** 2 + jnp.log(self.weights)
        return jax.scipy.special.logsumexp(log_comp, axis=-1)

    def energy(self, x: jax.Array) -> jax.Array:
        lp = self.log_prob(x)
        return (-lp).reshape(x.shape[0], 1)


class GaussianMixture2D(Energy):
    def __init__(self, means=None, covs=None, weights=None):
        if means is None:
            means = jnp.asarray([[-2, -2], [-2, 2], [2, 2], [2, -2]], dtype=jnp.float32)
        if covs is None:
            covs = jnp.asarray(
                [
                    [[1, 0.6], [0.6, 1]],
                    [[1, 0.0], [0.0, 1]],
                    [[1, 0.6], [0.6, 1]],
                    [[1, 0.0], [0.0, 1]],
                ],
                dtype=jnp.float32,
            )
        if weights is None:
            weights = jnp.asarray([0.5, 0.5, 0.25, 0.75], dtype=jnp.float32)
        self.means = means
        self.covs = covs
        self.weights = weights / jnp.sum(weights)

        # Precompute inverses and determinants
        self.inv_covs = jnp.linalg.inv(self.covs)
        self.det_covs = jnp.linalg.det(self.covs)

    def log_prob(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x)
        # x: (N,2)
        diffs = x[:, None, :] - self.means[None, :, :]  # (N,K,2)
        exponents = -0.5 * jnp.einsum(
            "nki, kij, nkj -> nk", diffs, self.inv_covs, diffs
        )  # (N,K)
        norm_consts = -jnp.log(2 * jnp.pi) - 0.5 * jnp.log(self.det_covs)  # (K,)
        log_comp = norm_consts[None, :] + exponents + jnp.log(self.weights)[None, :]
        return jax.scipy.special.logsumexp(log_comp, axis=-1)

    def energy(self, x: jax.Array) -> jax.Array:
        return (-self.log_prob(x)).reshape(x.shape[0], 1)


