from typing import Callable, Optional
import copy
import torch
import functools

from tqdm import tqdm
from dataclasses import dataclass
from tensordict import TensorDict
from mcmc.utils import EMA


__all__ = ["MHSampler", "MALASampler", "ImportanceSampler"]


@dataclass
class ImportanceSampler:
    def __call__(
        self,
        energy_fn: Callable,
        proposal_distribution: torch.distributions.Distribution,
        samples: int = 10_000,
    ):
        metrics = {}

        samples: torch.Tensor = proposal_distribution.sample((samples,))
        log_prob = proposal_distribution.log_prob(samples)

        log_weights = -energy_fn(samples) - log_prob
        # Use log-sum-exp trick for numerical stability
        max_logw = torch.max(log_weights)
        Z_est = torch.exp(max_logw) * torch.mean(torch.exp(log_weights - max_logw))

        return samples, energy_fn(samples), Z_est, metrics


def MetropolisHastingsAcceptance(
    energy,
    proposal_energy,
    forward_energy=None,
    reverse_energy=None,
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
    Notes:
        π(x) ∝ exp(-E(x))
        α = min(1, exp(-E(x'))q(x|x')/exp(-E(x))q(x'|x))
        log(α) = min(0, -E(x') + E(x) + log(q(x|x')) - log(q(x'|x)))
        log(α) = min(0, -E(x') + E(x) - E(x|x') + E(x'|x)))
    """
    assert (
        energy.shape == proposal_energy.shape
    ), f"Shape mismatch: {energy.shape} vs {proposal_energy.shape}"
    log_ratio = -proposal_energy + energy
    if forward_energy is not None and reverse_energy is not None:
        assert (
            reverse_energy.shape == forward_energy.shape == log_ratio.shape
        ), f"Shape mismatch: {reverse_energy.shape}, {forward_energy.shape}, {log_ratio.shape}"
        log_ratio = log_ratio - reverse_energy + forward_energy
    log_ratio = torch.minimum(log_ratio, torch.zeros_like(log_ratio))
    log_u = torch.log(torch.rand_like(log_ratio))  # log U(0,1)
    accept = log_ratio > log_u
    return accept


@dataclass
class Sampler:
    """
    Base class for MCMC samplers.
    """

    has_accept_step: bool

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
        buffer: Optional[int] = 50,
    ):
        """
        Run the MCMC sampler.

        Args:
            sample: Initial sample (TensorDict).
            energy_fn: Vectorized energy function.
            burn_in: Number of burn-in steps.
            steps: Total number of steps.
            verbose: Whether to print progress.
            buffer: If None, store all samples after burn-in. If a positive integer, only the last 'buffer' samples are stored. Default is 50.
        """
        assert hasattr(
            energy_fn, "__wrapped__"
        ), "energy_fn must be wrapped with torch.func.vmap for vectorized evaluation."
        accept_ema = EMA(ema_weight=0.99)
        pbar = tqdm(range(steps)) if verbose else range(steps)
        energy = energy_fn(*sample.values())
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

            if self.has_accept_step:
                accept = MetropolisHastingsAcceptance(
                    energy,
                    proposal_energy,
                    forward_log_prob,
                    backward_log_prob,
                )
                # If the sampler has an accept step, we need to call it
                # to update the sample based on the acceptance decision.
                sample, energy = self.accept_proposal(
                    sample,
                    proposal_sample,
                    energy,
                    proposal_energy,
                    accept,
                )
            else:
                # If not, we just update the sample and energy directly
                # without an additional accept step.
                sample = proposal_sample
                accept = torch.ones(
                    energy.shape[0], dtype=torch.bool, device=sample.device
                )

            accept_ratio = accept.float().mean()
            if step >= burn_in:
                chain.append(
                    (copy.deepcopy(sample).auto_batch_size_(), copy.deepcopy(energy))
                )
                if buffer is not None and buffer > 0 and len(chain) > buffer:
                    chain.pop(0)
            accept_ema(accept_ratio.detach().item())
            if verbose:
                first_key = list(sample.keys())[0]
                print_str = {
                    "Accept": f"{accept_ema.val:.3f} Mean: {sample[first_key].mean():.3f} Std: {sample[first_key].std():.3f}"
                }
                for key, value in metrics.items():
                    print_str["Accept"] += f" {key}: {value:.3f}"
                pbar.set_postfix(print_str)  # type: ignore
        samples, energy = zip(
            *chain
        )  # List[(sample,energy)] -> List[sample], List[energy]
        samples = torch.cat(samples, dim=0)
        energy = torch.cat(energy, dim=0)
        return samples, energy

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
        first_key = list(sample.keys())[0]
        for accept_, s_, s, e_, e in zip(
            accept,
            proposal_sample.chunk(num_chains, dim=0),
            sample.chunk(num_chains, dim=0),
            proposal_energy,
            energy,
        ):
            # Only update the first field, keep the rest as-is
            s_dict = s.to_dict()
            s__dict = s_.to_dict()
            if accept_.item():
                s_dict[first_key] = s__dict[first_key]
                new_sample.append(TensorDict(s_dict, batch_size=[1]))
                new_energy.append(e_)
            else:
                new_sample.append(s)
                new_energy.append(e)
        sample = torch.cat(new_sample, dim=0)
        energy = torch.stack(new_energy, dim=0)
        return sample, energy


@dataclass
class MHSampler(Sampler):
    std: float = 0.1
    proposal_fn: Callable = lambda x, std: x + torch.randn_like(x) * std
    schedule: Optional[Callable] = None
    has_accept_step: bool = True

    def proposal_step(
        self,
        sample: TensorDict,
        energy_fn: Callable,
        step: int | None = None,
    ) -> dict:
        std = self.std if self.schedule is None else self.schedule(step)
        metrics = {"std": std}
        first_key = list(sample.keys())[0]
        proposal_fn = functools.partial(self.proposal_fn, std=std)
        proposal_sample = sample.clone()
        if isinstance(sample[first_key], torch.Tensor):
            proposal_sample[first_key] = proposal_fn(sample[first_key])
        elif isinstance(sample[first_key], TensorDict):
            proposal_sample[first_key] = sample[first_key].apply(proposal_fn)
        else:
            raise ValueError(f"Unsupported type: {type(sample[first_key])}")
        proposal_energy = energy_fn(*proposal_sample.values())
        energy = energy_fn(*sample.values())
        return {
            "energy": energy,
            "proposal_sample": proposal_sample,
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
    has_accept_step: bool = False

    def proposal_step(
        self, sample: TensorDict, energy_fn: Callable, step: int | None = None
    ) -> dict:
        metrics = {}
        step_size = (
            self.step_size
            if self.step_size_schedule is None
            else self.step_size_schedule(step)
        )
        dampening = (
            self.dampening
            if self.dampening_schedule is None
            else self.dampening_schedule(step)
        )
        # Find the first tensor field
        first_key = list(sample.keys())[0]
        first_value = sample[first_key]
        sampled_td = TensorDict({"sampled_param": first_value})
        args = list(sample.values())
        args = [arg.to_dict() if isinstance(arg, TensorDict) else arg for arg in args]

        def energy_sum(*args):
            return torch.sum(energy_fn(*args))

        with torch.enable_grad():
            grad, energy_ = torch.func.grad_and_value(energy_sum, argnums=(0,))(*args)
            energy_.detach()
            grad_tensor = grad[0]
            grad_td = TensorDict({"sampled_param": grad_tensor})
        # TODO Ludi: theres something wrong with the forward and backward transition log-probabilities
        proposal_sampled_td = sampled_td.apply(
            lambda x, grad: x
            - step_size * grad
            + torch.randn_like(x) * (2 * step_size * dampening) ** 0.5,
            grad_td,
        )
        proposal_sample = copy.deepcopy(sample)
        proposal_sample[first_key] = proposal_sampled_td["sampled_param"]
        proposal_args = list(proposal_sample.values())
        proposal_args = [
            arg.to_dict() if isinstance(arg, TensorDict) else arg
            for arg in proposal_args
        ]
        with torch.no_grad():
            proposal_energy = energy_fn(*proposal_args)
            energy = energy_fn(*args)
        return {
            "energy": energy,
            "proposal_sample": proposal_sample,
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
    has_accept_step: bool = True

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
        # Find the first tensor field
        first_key = list(sample.keys())[0]
        first_value = sample[first_key]
        # Pack into a TensorDict for apply
        sample_td = TensorDict({"sampled_param": first_value})
        # Prepare args for grad_and_value
        args = list(sample.values())
        args = [arg.to_dict() if isinstance(arg, TensorDict) else arg for arg in args]

        def energy_sum(*args):
            return torch.sum(energy_fn(*args))

        with torch.enable_grad():
            grad, energy_ = torch.func.grad_and_value(energy_sum, argnums=(0,))(*args)
            energy_.detach()
            grad_tensor = grad[0]
            grad_td = TensorDict({"sampled_param": grad_tensor})
        # Proposal using apply
        proposal_sample_td = sample_td.apply(
            lambda x, grad: x
            - step_size * grad
            + torch.randn_like(x) * (2 * step_size * dampening) ** 0.5,
            grad_td,
        )
        # Update the original TensorDict
        proposal_sample = sample.clone()
        proposal_sample[first_key] = proposal_sample_td["sampled_param"]
        # Forward transition log-probability
        # q(x'|x)   \propto exp(-||x' - x - step_size * \nabla log pi(x)||^2 / (4 * step_size))
        #           \propto exp(-||x' - x - step_size * \nabla log exp(-E(x)||^2 / (4 * step_size))
        #           \propto exp(-||x' - x + step_size * \nabla E(x)||^2 / (4 * step_size))
        forward_energy = sample_td.apply(
            lambda x, grad, proposal: (proposal - x + step_size * grad)
            .pow(2)
            .sum(dim=-1)
            / (4 * step_size),
            grad_td,
            proposal_sample_td,
        )
        forward_energy = forward_energy.apply(
            lambda x: torch.einsum("b ... -> b", x)
        )  # Sum over all dimensions
        forward_energy = sum(list(forward_energy.values())).unsqueeze(
            -1
        )  # Sum over all entries in TensorDict
        # Reverse transition log-probability
        proposal_args = list(proposal_sample.values())
        proposal_args = [
            arg.to_dict() if isinstance(arg, TensorDict) else arg
            for arg in proposal_args
        ]
        with torch.enable_grad():
            grad_proposal, _ = torch.func.grad_and_value(energy_sum, argnums=(0,))(
                *proposal_args
            )
            grad_proposal_tensor = grad_proposal[0]
        grad_proposal_td = TensorDict({"sampled_param": grad_proposal_tensor})

        # Forward transition log-probability
        backward_energy = sample_td.apply(
            lambda x, proposal_grad, proposal: (
                x - proposal + step_size * proposal_grad
            )
            .pow(2)
            .sum(dim=-1)
            / (4 * step_size),
            grad_proposal_td,
            proposal_sample_td,
        )
        backward_energy = backward_energy.apply(
            lambda x: torch.einsum("b ... -> b", x)
        )  # Sum over all dimensions
        backward_energy = sum(list(backward_energy.values())).unsqueeze(
            -1
        )  # Sum over all entries in TensorDict

        with torch.no_grad():
            proposal_energy = energy_fn(*proposal_args)
            energy = energy_fn(*args)
        return {
            "energy": energy,
            "proposal_sample": proposal_sample,
            "proposal_energy": proposal_energy,
            "forward_transition_log_prob": forward_energy,
            "backward_transition_log_prob": backward_energy,
            "metrics": metrics,
        }
