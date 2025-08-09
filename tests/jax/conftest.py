import pytest
import jax
import jax.numpy as jnp
import numpy as np
import random


@pytest.fixture(autouse=True)
def set_seed():
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    key = jax.random.PRNGKey(seed)
    return key


