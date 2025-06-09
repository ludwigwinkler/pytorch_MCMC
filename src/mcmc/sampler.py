import torch

from tensordict import TensorDict

__all__ = ["MetropolisHastingsAcceptance"]


class MetropolisHastingsAcceptance(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, energy, proposal_energy):
        """
        a(x' | x)   = min( 1, exp(-Energy(x'))/exp(-E(x)))
                    = min( 1, exp(-Energy(x') - -E(x)))

        All calculations done in log space for numerical precision

        """
        assert energy.shape[-1] == 1, f"{energy.shape=}"
        assert energy.shape == proposal_energy.shape, (
            f"{energy.shape=} != {proposal_energy.shape=}"
        )
        log_ratio = proposal_energy - energy
        log_ratio = torch.min(log_ratio, torch.zeros_like(log_ratio))  # log(1) = 0
        log_u = torch.zeros_like(log_ratio).uniform_(0, 1).log()
        log_accept = torch.gt(log_ratio, log_u)

        return log_accept
