from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp


def MetropolisHastingsAcceptance(
    energy: jax.Array,
    proposal_energy: jax.Array,
    forward_energy: jax.Array | None = None,
    backward_energy: jax.Array | None = None,
    key: jax.Array | None = None,
) -> jax.Array:
    log_ratio = -proposal_energy + energy
    if (forward_energy is not None) and (backward_energy is not None):
        log_ratio = log_ratio - backward_energy + forward_energy
    log_ratio = jnp.minimum(log_ratio, 0.0)
    if key is None:
        key = jax.random.PRNGKey(0)
    u = jax.random.uniform(key, shape=log_ratio.shape)
    accept = jnp.log(u) < log_ratio
    return accept


@dataclass
class Sampler:
    has_accept_step: bool
    compile: bool = False

    def proposal_step(self, sample: dict, energy_fn: Callable, key: jax.Array, step: int = 0):
        raise NotImplementedError

    def __call__(
        self,
        sample: dict,
        energy_fn: Callable,
        key: jax.Array,
        burn_in: int = 100,
        steps: int = 1000,
        buffer: Optional[int] = 50,
    ):
        energy = energy_fn(sample)
        # Optionally JIT the proposal step
        if self.compile:
            proposal_f = jax.jit(
                lambda s, k, t: self.proposal_step(sample=s, energy_fn=energy_fn, key=k, step=t)
            )
        else:
            proposal_f = lambda s, k, t: self.proposal_step(sample=s, energy_fn=energy_fn, key=k, step=t)
        chain_samples = []
        chain_energies = []
        k = key
        for t in range(steps):
            k, subk = jax.random.split(k)
            subk_prop, subk_acc = jax.random.split(subk)
            out = proposal_f(sample, subk_prop, t)
            proposal = out["proposal_sample"]
            prop_energy = out["proposal_energy"]
            fwd = out.get("forward_transition_log_prob")
            bwd = out.get("backward_transition_log_prob")
            if self.has_accept_step:
                accept = MetropolisHastingsAcceptance(
                    energy, prop_energy, fwd, bwd, key=subk_acc
                )
                sample = jax.tree.map(lambda a, b: jnp.where(accept, a, b), proposal, sample)
                energy = jnp.where(accept, prop_energy, energy)
            else:
                sample = proposal
                energy = prop_energy
            if t >= burn_in:
                chain_samples.append(sample)
                chain_energies.append(energy)
                if buffer is not None and buffer > 0 and len(chain_samples) > buffer:
                    chain_samples.pop(0)
                    chain_energies.pop(0)
        # stack results
        stacked = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *chain_samples)
        energies = jnp.concatenate(chain_energies, axis=0)
        return stacked, energies


@dataclass
class MHSampler(Sampler):
    std: float = 0.1
    has_accept_step: bool = True

    def proposal_step(self, sample: dict, energy_fn: Callable, key: jax.Array, step: int = 0):
        x = sample["sample"]
        noise = jax.random.normal(key, shape=x.shape) * self.std
        prop = {**sample, "sample": x + noise}
        prop_energy = energy_fn(prop)
        energy = energy_fn(sample)
        return {
            "energy": energy,
            "proposal_sample": prop,
            "proposal_energy": prop_energy,
            "forward_transition_log_prob": None,
            "backward_transition_log_prob": None,
        }


@dataclass
class SGLDSampler(Sampler):
    step_size: float = 0.01
    dampening: float = 1.0
    has_accept_step: bool = False

    def proposal_step(self, sample: dict, energy_fn: Callable, key: jax.Array, step: int = 0):
        def energy_sum(s):
            e = energy_fn(s)
            return jnp.sum(e)

        grad_energy = jax.grad(energy_sum)(sample)
        x = sample["sample"]
        g = grad_energy["sample"]
        noise = jax.random.normal(key, shape=x.shape) * jnp.sqrt(2 * self.step_size * self.dampening)
        x_prop = x - self.step_size * g + noise
        prop = {**sample, "sample": x_prop}
        prop_energy = energy_fn(prop)
        energy = energy_fn(sample)
        return {
            "energy": energy,
            "proposal_sample": prop,
            "proposal_energy": prop_energy,
            "forward_transition_log_prob": None,
            "backward_transition_log_prob": None,
        }


@dataclass
class MALASampler(Sampler):
    step_size: float = 0.01
    dampening: float = 1.0
    has_accept_step: bool = True

    def proposal_step(self, sample: dict, energy_fn: Callable, key: jax.Array, step: int = 0):
        def energy_sum(s):
            e = energy_fn(s)
            return jnp.sum(e)

        grad_energy = jax.grad(energy_sum)(sample)
        x = sample["sample"]
        g = grad_energy["sample"]
        noise = jax.random.normal(key, shape=x.shape) * jnp.sqrt(2 * self.step_size * self.dampening)
        x_prop = x - self.step_size * g + noise
        prop = {**sample, "sample": x_prop}

        # transition energies (quadratic forms)
        forward_energy = jnp.sum((x_prop - x + self.step_size * g) ** 2, axis=-1, keepdims=True) / (
            4 * self.step_size
        )
        grad_energy_prop = jax.grad(energy_sum)(prop)
        g_prop = grad_energy_prop["sample"]
        backward_energy = jnp.sum((x - x_prop + self.step_size * g_prop) ** 2, axis=-1, keepdims=True) / (
            4 * self.step_size
        )

        prop_energy = energy_fn(prop)
        energy = energy_fn(sample)
        return {
            "energy": energy,
            "proposal_sample": prop,
            "proposal_energy": prop_energy,
            "forward_transition_log_prob": forward_energy,
            "backward_transition_log_prob": backward_energy,
        }


@dataclass
class HMCSampler(Sampler):
    step_size: float = 0.01
    num_steps: int = 10
    mass: float = 1.0
    has_accept_step: bool = True

    def proposal_step(self, sample: dict, energy_fn: Callable, key: jax.Array, step: int = 0):
        def energy_sum(s):
            e = energy_fn(s)
            return jnp.sum(e)

        x0 = sample["sample"]
        k1, k2 = jax.random.split(key)
        p0 = jax.random.normal(k1, shape=x0.shape) * jnp.sqrt(self.mass)

        def grad_energy(s):
            return jax.grad(energy_sum)(s)["sample"]

        q = x0
        p = p0 - 0.5 * self.step_size * grad_energy(sample)
        for _ in range(self.num_steps):
            q = q + self.step_size * p / self.mass
            prop = {**sample, "sample": q}
            p = p - (self.step_size * grad_energy(prop))
        # final half step
        p = p - 0.5 * self.step_size * grad_energy({**sample, "sample": q})
        # negate momentum
        p_prop = -p

        def hamiltonian(q, p):
            prop = {**sample, "sample": q}
            potential = energy_fn(prop)
            kinetic = jnp.sum((p**2), axis=-1, keepdims=True) / (2 * self.mass)
            return potential + kinetic

        current_H = hamiltonian(x0, p0)
        proposal_H = hamiltonian(q, p_prop)
        prop = {**sample, "sample": q}
        return {
            "energy": current_H,
            "proposal_sample": prop,
            "proposal_energy": proposal_H,
            "forward_transition_log_prob": None,
            "backward_transition_log_prob": None,
        }


