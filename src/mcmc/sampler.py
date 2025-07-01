from turtle import st
from typing import Callable
import copy
import torch
import functools
from torch import Tensor
from tqdm import tqdm
from dataclasses import dataclass
from tensordict import TensorDict
from mcmc.utils import EMA
from mcmc.utils import RepeatedCosineSchedule

from typing import Tuple, List


__all__ = ["MHSampler", "MALASampler", "ImportanceSampler"]


class MetropolisHastingsAcceptance(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, energy, proposal_energy, forward_proposal=None, reverse_proposal=None
    ):
        """
        a(x' | x)   = min( 1, exp(-Energy(x'))/exp(-E(x)))
                    = min( 1, exp(-Energy(x') - -E(x)))

        All calculations done in log space for numerical precision

        """
        assert energy.shape[-1] == 1, f"{energy.shape=}"
        assert energy.shape == proposal_energy.shape, (
            f"{energy.shape=} != {proposal_energy.shape=}"
        )
        log_ratio = -proposal_energy + energy
        log_ratio = torch.min(log_ratio, torch.zeros_like(log_ratio))  # log(1) = 0
        log_u = torch.zeros_like(log_ratio).uniform_(0, 1).log()
        log_accept = torch.gt(log_ratio, log_u)

        return log_accept


@dataclass
class Sampler(torch.nn.Module):
    """
    Base class for MCMC samplers.
    """

    def proposal_step(
        self,
        sample: TensorDict,
        energy_fn: Callable,
        step: int | None = None,
    ) -> Tuple[TensorDict, TensorDict]:
        """
        Perform a proposal step for the sampler.

        Args:
            sample (TensorDict): Current sample.
            energy (Callable): Energy function to evaluate the proposal.

        Returns:
            Tuple[TensorDict, TensorDict]: Proposed sample and its energy.
        """
        raise NotImplementedError

    def __call__(
        self,
        sample: TensorDict,
        energy_fn: Callable,
        burn_in: int = 100,
        steps: int = 1000,
        verbose: bool = True,
    ):
        assert hasattr(energy_fn, "__wrapped__"), (
            "energy_fn must be wrapped with torch.func.vmap for vectorized evaluation."
        )
        accept_ema = EMA(ema_weight=0.99)
        if verbose:
            pbar = tqdm(range(steps))
        else:
            pbar = range(steps)
        sample = sample
        energy = energy_fn(sample)

        chain = []

        for step in pbar:
            proposal_dict, metrics = self.proposal_step(
                sample=sample, energy_fn=energy_fn, step=step
            )

            if hasattr(self, "MH_Acceptance"):
                accept: Tensor = self.MH_Acceptance(
                    energy, proposal_dict["proposal_energy"]
                )
                accept_ratio = accept.sum() / accept.numel()
                sample, energy = self.accept_proposal(
                    sample,
                    proposal_dict["proposal_sample"],
                    energy,
                    proposal_dict["proposal_energy"],
                    accept,
                )  # filters according to accept
            else:
                accept_ratio = torch.scalar_tensor(1.0)
                sample = proposal_sample
                energy = proposal_energy

            if step >= burn_in:
                chain += [
                    (
                        copy.deepcopy(sample),
                        copy.deepcopy(energy),
                    )
                ]
                if len(chain) > 250:  # ring buffer
                    chain.pop(0)

            # Running Average of Acceptance Ratio

            accept_ema(accept_ratio.detach().item())

            if verbose:
                print_str = {
                    "Accept": f"{accept_ema.val:.3f} Mean: {sample['x'].mean():.3f} Std: {sample['x'].std():.3f}"
                }
                for key, value in metrics.items():
                    print_str["Accept"] += f" {key}: {value:.3f}"
                pbar.set_postfix(print_str)
        samples = [s for s, e in chain]
        samples = torch.cat(samples, dim=0)
        return samples, energy_fn(samples)

    def accept_proposal(self, sample, proposal_sample, energy, proposal_energy, accept):
        """
        Accept or reject the proposal based on the acceptance criteria.

        Args:
            sample (TensorDict): Current sample.
            proposal_sample (TensorDict): Proposed sample.
            accept (Tensor): Acceptance decision.

        Returns:
            TensorDict: Updated sample after accepting or rejecting the proposal.
        """
        proposal_sample.auto_batch_size_(1)
        sample.auto_batch_size_(1)
        num_chains = proposal_sample.batch_size[0]
        new_sample: List[TensorDict] = []
        new_energy = []
        for accept_, s_, s, e_, e in zip(
            accept,
            proposal_sample.chunk(num_chains, dim=0),
            sample.chunk(num_chains, dim=0),
            proposal_energy,
            energy,
        ):
            new_sample.append(s_) if accept_.item() else new_sample.append(s)
            new_energy.append(e_) if accept_.item() else new_energy.append(e)
        sample: TensorDict = torch.cat(new_sample, dim=0)  # type: ignore
        energy = torch.stack(new_energy, dim=0)
        return sample, energy


@dataclass
class ImportanceSampler:
    def __call__(
        self,
        energy_fn: Callable,
        proposal_distribution: torch.distributions.Distribution,
        samples: int = 10_000,
    ):
        metrics = {}

        samples = proposal_distribution.sample((samples,))
        log_prob = proposal_distribution.log_prob(samples)

        log_weights = -energy_fn(samples) - log_prob
        # Use log-sum-exp trick for numerical stability
        max_logw = torch.max(log_weights)
        Z_est = torch.exp(max_logw) * torch.mean(torch.exp(log_weights - max_logw))

        return samples, energy_fn(samples), Z_est, metrics


@dataclass
class MHSampler(Sampler):
    std: float = 0.1
    proposal_fn: Callable = lambda x, std: x + torch.randn_like(x) * std
    schedule: Callable | None = None
    MH_Acceptance = MetropolisHastingsAcceptance()

    def proposal_step(
        self,
        sample: TensorDict,
        energy_fn: Callable,
        step: int | None = None,
    ):
        std = self.std if self.schedule is None else self.schedule(step=step)
        metrics = {"std": std}
        proposal_fn = functools.partial(self.proposal_fn, std=std)
        proposal_state = sample.apply(proposal_fn)
        proposal_energy = energy_fn(proposal_state)
        return {
            "proposal_sample": proposal_state,
            "proposal_energy": proposal_energy,
            "metrics": metrics,
        }

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(std={self.std}, "
            f"proposal_fn={self.proposal_fn.__name__})"
        )


@dataclass
class SGLDSampler(Sampler):
    step_size: float = 0.01
    dampening: float = 1.0
    step_size_schedule: Callable | None = None
    dampening_schedule: Callable | None = None

    def proposal_step(
        self, sample: TensorDict, energy_fn: Callable, step: int | None = None
    ):
        metrics = {}
        step_size = (
            self.step_size
            if self.step_size_schedule is None
            else self.step_size_schedule(step=step)
        )
        dampening = (
            self.dampening
            if self.dampening_schedule is None
            else self.dampening_schedule(step=step)
        )
        with torch.enable_grad():
            grad, energy_ = torch.func.grad_and_value(
                lambda args: energy_fn(args).sum(), argnums=(0,)
            )(sample.to_dict())
            energy_.detach()
            grad = TensorDict(grad[0], batch_size=sample.batch_size).detach()

        proposal_state = copy.deepcopy(sample).apply(
            lambda x, grad: x
            - step_size * grad
            + torch.randn_like(x) * (2 * step_size) ** 0.5,
            grad,
        )
        with torch.no_grad():
            proposal_energy = energy_fn(proposal_state)
        return proposal_state, proposal_energy, metrics


@dataclass
class MALASampler(SGLDSampler):
    MH_Acceptance = MetropolisHastingsAcceptance()
    # TODO: respect nonsymmetric proposal probability
    '''
    class MetropolisHastingsAcceptance(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, energy, proposal_energy, log_q_reverse, log_q_forward):
        """
        Computes Metropolis-Hastings acceptance for asymmetric proposal.

        Args:
            energy:           shape [batch, 1]  — E(x)
            proposal_energy:  shape [batch, 1]  — E(x')
            log_q_reverse:    log q(x | x')
            log_q_forward:    log q(x' | x)

        Returns:
            accept_mask: shape [batch, 1], bool
        """
        assert energy.shape == proposal_energy.shape
        log_ratio = -proposal_energy + energy + log_q_reverse - log_q_forward
        log_ratio = torch.minimum(log_ratio, torch.zeros_like(log_ratio))
        log_u = torch.log(torch.rand_like(log_ratio))  # log U(0,1)
        accept = log_ratio > log_u
        return accept
    '''

    def proposal_step(
        self, sample: TensorDict, energy_fn: Callable, step: int | None = None
    ):
        metrics = {}
        step_size = (
            self.step_size
            if self.step_size_schedule is None
            else self.step_size_schedule(step=step)
        )
        dampening = (
            self.dampening
            if self.dampening_schedule is None
            else self.dampening_schedule(step=step)
        )
        with torch.enable_grad():
            grad, energy_ = torch.func.grad_and_value(
                lambda args: energy_fn(args).sum(), argnums=(0,)
            )(sample.to_dict())
            energy_.detach()
            grad = TensorDict(grad[0], batch_size=sample.batch_size).detach()

        """Calculate next stochastic sample"""
        proposal_sample = sample.apply(
            lambda x, grad: x
            - step_size * grad
            + torch.randn_like(x) * (2 * step_size) ** 0.5,
            grad,
        )
        with torch.no_grad():
            proposal_energy = energy_fn(proposal_sample)

        """Obtain the forward transition log probability"""
        deterministic_forward_proposal = sample - step_size * grad
        forward_transition_log_prob = (
            -1
            / (4 * step_size)
            * ((proposal_sample - deterministic_forward_proposal) ** 2).sum(
                dim=-1, keepdim=True
            )
        )
        """Obtain reverse transition log probability"""
        with torch.enable_grad():
            grad, energy_ = torch.func.grad_and_value(
                lambda args: energy_fn(args).sum(), argnums=(0,)
            )(proposal_sample.to_dict())
            energy_.detach()
            grad = TensorDict(grad[0], batch_size=sample.batch_size).detach()
        deterministic_backward_proposal = proposal_sample - step_size * grad
        backward_transition_log_prob = (
            -1
            / (4 * step_size)
            * ((sample - deterministic_backward_proposal) ** 2).sum(
                dim=-1, keepdim=True
            )
        )
        return {
            "proposal_sample": proposal_sample,
            "proposal_energy": proposal_energy,
            "forward_transition_log_prob": forward_transition_log_prob,
            "backward_transition_log_prob": backward_transition_log_prob,
            "metrics": metrics,
        }
