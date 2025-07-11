import pytest
import torch
import numpy as np
import random


@pytest.fixture(autouse=True)
def set_global_seed():
    """Set seed before each test across all test files."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
