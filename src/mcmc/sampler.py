from typing import Callable, Optional
import copy
import torch
import functools

from tqdm import tqdm
from dataclasses import dataclass
from tensordict import TensorDict
from mcmc.utils import EMA

from torch import Tensor


__all__ = ["MHSampler", "MALASampler", "ImportanceSampler"]


def _flatten_args(td):
    return [v.to_dict() if isinstance(v, TensorDict) else v for v in td.values()]


@dataclass
class ImportanceSampler:
    def __call__(
        self,
        energy_fn: Callable,
        proposal_distribution: torch.distributions.Distribution,
        num_samples: int = 10_000,
    ):
        metrics = {}

        samples: torch.Tensor = proposal_distribution.sample((num_samples,))
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
    ) -> tuple[TensorDict, torch.Tensor]:
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
        init_args = [
            arg.to_dict() if isinstance(arg, TensorDict) else arg
            for arg in list(sample.values())
        ]
        energy = energy_fn(*init_args)
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
                    (
                        copy.deepcopy(sample.detach()).auto_batch_size_(),
                        copy.deepcopy(energy.detach()),
                    )
                )
                if buffer is not None and buffer > 0 and len(chain) > buffer:
                    chain.pop(0)
            accept_ema(float(accept_ratio.detach().item()))
            if verbose:
                print_str = {"Accept": f"{accept_ema.val:.3f} {energy.mean():.3f}"}
                for key, value in metrics.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    print_str["Accept"] += f" {key}: {value:.3f} "
                pbar.set_postfix(print_str)  # type: ignore
        samples, energies = zip(
            *chain
        )  # List[(sample,energy)] -> List[sample], List[energy]

        # Stack 'sample' and 'buffers' keys, preserve other keys from first sample
        vmap_keys = ["sample", "buffers"] if "buffers" in samples[0] else ["sample"]
        vmap_entries = [
            TensorDict(s.select(*vmap_keys), batch_size=[energy.shape[0]])
            for s in samples
        ]
        non_vmap_entries = samples[0].exclude(*vmap_keys)
        stacked_vmapped = torch.cat(vmap_entries, dim=0)
        samples = TensorDict(
            {k: v for k, v in stacked_vmapped.items()}
            | {k: v for k, v in non_vmap_entries.items()}
        )
        energy = torch.cat(energies, dim=0)
        return samples, energy

    def accept_proposal(
        self,
        sample: TensorDict,
        proposal_sample: TensorDict,
        energy: Tensor,
        proposal_energy,
        accept,
    ):
        """
        Accept or reject the proposal based on the acceptance criteria.
        Only processes the first vmapped parameter key and optionally 'buffers' key.

        Args:
            sample (TensorDict): Current sample.
            proposal_sample (TensorDict): Proposed sample.
            accept (Tensor): Acceptance decision.

        Returns:
            TensorDict: Updated sample after accepting or rejecting the proposal.
        """
        # Filter to only the first vmapped parameter key and optionally 'buffers' key
        vmapped_keys = ["sample"]  # First key (e.g., "params")
        if "buffers" in sample:
            vmapped_keys.append("buffers")

        filtered_sample = TensorDict({key: sample[key] for key in vmapped_keys})
        filtered_proposal = TensorDict(
            {key: proposal_sample[key] for key in vmapped_keys}
        )

        # Set batch size to match energy dimension
        num_chains = energy.shape[0]
        filtered_sample.batch_size = [num_chains]  # type: ignore
        filtered_proposal.batch_size = [num_chains]  # type: ignore

        # Your existing chunking logic
        new_sample: list = []
        new_energy = []

        for accept_, s_, s, e_, e in zip(
            accept,
            filtered_proposal.chunk(num_chains, dim=0),
            filtered_sample.chunk(num_chains, dim=0),
            proposal_energy,
            energy,
        ):
            if accept_.item():
                new_sample.append(s_)
                new_energy.append(e_)
            else:
                new_sample.append(s)
                new_energy.append(e)

        # Reconstruct full TensorDict with updated vmapped parameters
        result_sample = copy.deepcopy(sample)
        result_filtered = torch.cat(new_sample, dim=0)

        # Update only the vmapped keys
        for key in vmapped_keys:
            result_sample.set(key, result_filtered[key])  # type: ignore

        energy = torch.stack(new_energy, dim=0)
        return result_sample, energy


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

        # Apply proposal function to the first vmapped parameter key
        proposal_fn = functools.partial(self.proposal_fn, std=std)
        proposal_sample = copy.deepcopy(sample)
        proposal_sample["sample"] = proposal_fn(sample["sample"])

        proposal_energy = energy_fn(*_flatten_args(proposal_sample))
        energy = energy_fn(*_flatten_args(sample))
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

        def energy_sum(*args):
            return torch.sum(energy_fn(*args))

        # Find the first tensor field
        args = [
            arg.to_dict() if isinstance(arg, TensorDict) else arg
            for arg in list(sample.values())
        ]

        with torch.enable_grad():
            grad, energy_ = torch.func.grad_and_value(energy_sum, argnums=(0,))(*args)
            energy_.detach()
            grad_tensor = grad[0]
            grad_td = TensorDict({"sample": grad_tensor})
        proposal_sample = copy.deepcopy(sample.detach().select("sample")).apply(
            lambda x, grad: x
            - step_size * grad
            + torch.randn_like(x) * (2 * step_size * dampening) ** 0.5,
            grad_td,
        )
        proposal_sample = copy.deepcopy(sample).update(proposal_sample).detach()
        proposal_args = [
            arg.to_dict() if isinstance(arg, TensorDict) else arg
            for arg in list(proposal_sample.values())
        ]
        with torch.no_grad():
            proposal_energy = energy_fn(*proposal_args).detach()
            energy = energy_fn(*args).detach()
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
        proposal_sample = copy.deepcopy(sample.detach())
        proposal_sample[first_key] = proposal_sample_td["sampled_param"]  # type: ignore
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
            proposal_sample_td,  # type: ignore
        )
        forward_energy = forward_energy.apply(
            lambda x: torch.einsum("b ... -> b", x)
        )  # Sum over all dimensions
        forward_energy = sum(list(forward_energy.values(True, True))).unsqueeze(-1)  # type: ignore
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
            proposal_sample_td,  # type: ignore
        )
        backward_energy = backward_energy.apply(
            lambda x: torch.einsum("b ... -> b", x)
        )  # Sum over all dimensions
        backward_energy = sum(list(backward_energy.values(True, True))).unsqueeze(-1)  # type: ignore

        with torch.no_grad():
            proposal_energy = energy_fn(*proposal_args)
            energy = energy_fn(*args)
        return {
            "energy": energy.detach(),
            "proposal_sample": proposal_sample.detach(),
            "proposal_energy": proposal_energy.detach(),
            "forward_transition_log_prob": forward_energy.detach(),
            "backward_transition_log_prob": backward_energy.detach(),
            "metrics": metrics,
        }

        # INSERT_YOUR_CODE


@dataclass
class HMCSampler(Sampler):
    step_size: float = 0.01
    num_steps: int = 10
    mass: float = 1.0
    step_size_schedule: Optional[Callable] = None
    num_steps_schedule: Optional[Callable] = None
    has_accept_step: bool = True

    def proposal_step(
        self, sample: TensorDict, energy_fn: Callable, step: int | None = None
    ) -> dict:
        # metrics = {}
        step_size = (
            self.step_size
            if self.step_size_schedule is None
            else self.step_size_schedule(step)
        )
        num_steps = (
            self.num_steps
            if self.num_steps_schedule is None
            else self.num_steps_schedule(step)
        )

        # Assume 'sample' key contains the parameter tensor
        q_init = sample["sample"]
        # Draw momentum from N(0, mass)
        p_init = torch.randn_like(q_init) * self.mass**0.5

        def energy_sum(*args):
            return torch.sum(energy_fn(*args))

        # Leapfrog integration
        q = q_init.clone().detach()
        p = p_init.clone().detach()
        q.requires_grad_(True)
        p.requires_grad_(True)

        # Half step for momentum
        args = [
            arg.to_dict() if isinstance(arg, TensorDict) else arg
            for arg in list(sample.values())
        ]
        with torch.enable_grad():
            grad_q, _ = torch.func.grad_and_value(energy_sum, argnums=(0,))(*args)
            grad_q_tensor = grad_q[0]
        p = p - 0.5 * step_size * grad_q_tensor

        for _ in range(num_steps):
            # Full step for position
            q = q + step_size * p / self.mass
            # Prepare new args for gradient
            proposal_sample = copy.deepcopy(sample)
            proposal_sample["sample"] = q
            proposal_args = [
                arg.to_dict() if isinstance(arg, TensorDict) else arg
                for arg in list(proposal_sample.values())
            ]
            # Full step for momentum, except at end of trajectory
            with torch.enable_grad():
                grad_q, _ = torch.func.grad_and_value(energy_sum, argnums=(0,))(
                    *proposal_args
                )
                grad_q_tensor = grad_q[0]
            if _ != num_steps - 1:
                p = p - step_size * grad_q_tensor
        # Final half step for momentum
        p = p - 0.5 * step_size * grad_q_tensor

        # Negate momentum for symmetry
        p_prop = -p

        # Build proposal sample
        proposal_sample = copy.deepcopy(sample)
        proposal_sample["sample"] = q.detach()

        # Compute energies
        def hamiltonian(q, p):
            sample_dict = copy.deepcopy(sample)
            sample_dict["sample"] = q
            args = [
                arg.to_dict() if isinstance(arg, TensorDict) else arg
                for arg in list(sample_dict.values())
            ]
            potential = energy_fn(*args)
            kinetic = (p**2).sum(dim=-1, keepdim=True) / (2 * self.mass)
            return potential + kinetic

        with torch.no_grad():
            current_H = hamiltonian(q_init, p_init).unsqueeze(-1)
            proposal_H = hamiltonian(q.detach(), p_prop.detach()).unsqueeze(-1)

        # No explicit transition log-probabilities for HMC
        return {
            "energy": current_H.detach(),
            "proposal_sample": proposal_sample.detach(),
            "proposal_energy": proposal_H.detach(),
            "forward_transition_log_prob": None,
            "backward_transition_log_prob": None,
            # "metrics": {
            #     **metrics,
            #     "current_H": current_H.detach(),
            #     "proposal_H": proposal_H.detach(),
            #     "accept_prob": torch.exp(current_H - proposal_H).clamp(max=1.0),
            # },
        }
