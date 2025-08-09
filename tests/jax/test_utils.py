import pytest
import jax
import jax.numpy as jnp

pytest.importorskip("mcmc.jax")


def test_key_split_reproducibility():
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    k1b, k2b = jax.random.split(jax.random.PRNGKey(42))
    assert jnp.all(k1 == k1b)
    assert jnp.all(k2 == k2b)


