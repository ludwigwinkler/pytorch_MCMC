import torch

from tensordict import TensorDict


class MetropolisHastings(torch.nn.Module):
    def __init__(self):
        super().__init__()
