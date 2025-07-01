from turtle import st
from typing import Callable, Optional
import copy
import torch
import functools
from torch import Tensor
from tqdm import tqdm
from dataclasses import dataclass, field
from tensordict import TensorDict
from mcmc.utils import EMA
from mcmc.utils import RepeatedCosineSchedule

from typing import Tuple, List


__all__ = ["MHSampler", "MALASampler", "ImportanceSampler"]


def MetropolisHastingsAcceptance(
    energy,
    proposal_energy,
    forward_log_prob=None,
    reverse_log_prob=None,
    asymmetric=False,
):
    """
    Metropolis-Hastings acceptance function (stateless).
    Args:
        energy: Current energy.
        proposal_energy: Proposed energy.
        forward_log_prob: Log-probability of forward transition (for asymmetric proposals).
        reverse_log_prob: Log-probability of reverse transition (for asymmetric proposals).
        asymmetric: Whether to use asymmetric acceptance (e.g., for MALA).
    Returns:
        accept: Boolean tensor indicating acceptance.
    """
    assert energy.shape == proposal_energy.shape
    log_ratio = -proposal_energy + energy
    if asymmetric and forward_log_prob is not None and reverse_log_prob is not None:
        log_ratio = log_ratio + reverse_log_prob - forward_log_prob
    log_ratio = torch.minimum(log_ratio, torch.zeros_like(log_ratio))
    log_u = torch.log(torch.rand_like(log_ratio))  # log U(0,1)
    accept = log_ratio > log_u
    return accept


@dataclass
class Sampler:
    """
    Base class for MCMC samplers.
    """

    def proposal_step(
        self,
        sample: TensorDict,
        energy_fn: Callable,
        step: int | None = None,
    ) -> dict:
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
        pbar = tqdm(range(steps)) if verbose else range(steps)
        energy = energy_fn(sample)
        chain = []
        for step in pbar:
            proposal_dict = self.proposal_step(
                sample=sample, energy_fn=energy_fn, step=step
            )
            proposal_sample = proposal_dict["proposal_sample"]
            proposal_energy = proposal_dict["proposal_energy"]
            energy = proposal_dict["energy"]
            forward_log_prob = proposal_dict["forward_transition_log_prob"]
            backward_log_prob = proposal_dict["backward_transition_log_prob"]
            metrics = proposal_dict.get("metrics", {})

            accept = MetropolisHastingsAcceptance(
                energy,
                proposal_energy,
                forward_log_prob,
                backward_log_prob,
                getattr(self, "asymmetric", False),
            )
            accept_ratio = accept.float().mean()
            sample, energy = self.accept_proposal(
                sample,
                proposal_sample,
                energy,
                proposal_energy,
                accept,
            )
            if step >= burn_in:
                chain.append((copy.deepcopy(sample), copy.deepcopy(energy)))
                if len(chain) > 250:
                    chain.pop(0)
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
        new_sample: list = []
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
        sample = torch.cat(new_sample, dim=0)
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
    schedule: Optional[Callable] = None
    asymmetric: bool = False

    def proposal_step(
        self,
        sample: TensorDict,
        energy_fn: Callable,
        step: int | None = None,
    ) -> dict:
        std = self.std if self.schedule is None else self.schedule(step=step)
        metrics = {"std": std}
        proposal_fn = functools.partial(self.proposal_fn, std=std)
        proposal_state = sample.apply(proposal_fn)
        proposal_energy = energy_fn(proposal_state)
        energy = energy_fn(sample)
        return {
            "energy": energy,
            "proposal_sample": proposal_state,
            "proposal_energy": proposal_energy,
            "forward_transition_log_prob": None,
            "backward_transition_log_prob": None,
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
    step_size_schedule: Optional[Callable] = None
    dampening_schedule: Optional[Callable] = None
    asymmetric: bool = False

    def proposal_step(
        self, sample: TensorDict, energy_fn: Callable, step: int | None = None
    ) -> dict:
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
        proposal_state = sample.apply(
            lambda x, grad: x
            - step_size * grad
            + torch.randn_like(x) * (2 * step_size * dampening) ** 0.5,
            grad,
        )
        with torch.no_grad():
            proposal_energy = energy_fn(proposal_state)
            energy = energy_fn(sample)
        return {
            "energy": energy,
            "proposal_sample": proposal_state,
            "proposal_energy": proposal_energy,
            "forward_transition_log_prob": None,
            "backward_transition_log_prob": None,
            "metrics": metrics,
        }


@dataclass
class MALASampler(Sampler):
    step_size: float = 0.01
    dampening: float = 1.0
    step_size_schedule: Optional[Callable] = None
    dampening_schedule: Optional[Callable] = None
    asymmetric: bool = True

    def proposal_step(
        self, sample: TensorDict, energy_fn: Callable, step: int | None = None
    ) -> dict:
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
        proposal_sample = sample.apply(
            lambda x, grad: x
            - step_size * grad
            + torch.randn_like(x) * (2 * step_size * dampening) ** 0.5,
            grad,
        )
        with torch.no_grad():
            proposal_energy = energy_fn(proposal_sample)
            energy = energy_fn(sample)
        # Forward transition log-probability
        deterministic_forward = sample.apply(lambda x, grad: x - step_size * grad, grad)
        squared_diffs_forward = proposal_sample.apply(
            lambda x, y: ((x - y) ** 2).sum(dim=-1, keepdim=True), deterministic_forward
        )
        forward_log_prob = -sum(squared_diffs_forward.values()) / (4 * step_size)
        # Reverse transition log-probability
        with torch.enable_grad():
            grad_prop, _ = torch.func.grad_and_value(
                lambda args: energy_fn(args).sum(), argnums=(0,)
            )(proposal_sample.to_dict())
            grad_prop = TensorDict(grad_prop[0], batch_size=sample.batch_size).detach()
        deterministic_backward = proposal_sample.apply(
            lambda x, grad: x - step_size * grad, grad_prop
        )
        squared_diffs_backward = sample.apply(
            lambda x, y: ((x - y) ** 2).sum(dim=-1, keepdim=True),
            deterministic_backward,
        )
        backward_log_prob = -sum(squared_diffs_backward.values()) / (4 * step_size)
        return {
            "energy": energy,
            "proposal_sample": proposal_sample,
            "proposal_energy": proposal_energy,
            "forward_transition_log_prob": forward_log_prob,
            "backward_transition_log_prob": backward_log_prob,
            "metrics": metrics,
        }
