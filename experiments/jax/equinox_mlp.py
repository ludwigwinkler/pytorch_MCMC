"""
Minimal Equinox MLP example in JAX.

Usage:
  .venv/bin/python experiments/jax/equinox_mlp.py --steps 2000 --lr 1e-2 --batch-size 256
Exit code will be 0 if final loss < initial loss, otherwise non-zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import argparse
import sys

import jax
import jax.numpy as jnp

try:
    import equinox as eqx
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "This example requires the 'equinox' package. Install with `uv add equinox` or `pip install equinox`."
    ) from e


@dataclass
class TrainConfig:
    seed: int = 0
    num_samples: int = 2_000
    batch_size: int = 256
    steps: int = 2_000
    learning_rate: float = 1e-2


class MLP(eqx.Module):
    layers: list

    def __init__(self, key: jax.Array, in_dim: int = 1, hidden: tuple[int, ...] = (64, 64), out_dim: int = 1):
        keys = jax.random.split(key, len(hidden) + 1)
        dims = (in_dim,) + hidden + (out_dim,)
        self.layers = [
            eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i]) for i in range(len(dims) - 1)
        ]

    def __call__(self, x: jax.Array) -> jax.Array:
        y = x
        for layer in self.layers[:-1]:
            y = jax.nn.tanh(layer(y))
        return self.layers[-1](y)


def make_synthetic_regression(key: jax.Array, num: int) -> tuple[jax.Array, jax.Array]:
    x = jax.random.uniform(key, shape=(num, 1), minval=-jnp.pi, maxval=jnp.pi)
    f = jnp.sin(3.0 * x) + 0.3 * jnp.cos(5.0 * x)
    noise = 0.1 * jax.random.normal(jax.random.split(key)[0], shape=f.shape)
    y = f + noise
    # standardize
    x = (x - jnp.mean(x, axis=0)) / (jnp.std(x, axis=0) + 1e-6)
    y = (y - jnp.mean(y, axis=0)) / (jnp.std(y, axis=0) + 1e-6)
    return x, y


def mse_loss(model: MLP, x: jax.Array, y: jax.Array) -> jax.Array:
    # Apply model per-sample: eqx Linear expects (..., in_dim); we vectorize over batch
    yhat = jax.vmap(model)(x)
    return jnp.mean((yhat - y) ** 2)


@eqx.filter_jit
def train_step(model: MLP, x: jax.Array, y: jax.Array, lr: float) -> tuple[MLP, jax.Array]:
    loss, grads = eqx.filter_value_and_grad(mse_loss)(model, x, y)
    updates = jax.tree.map(lambda g: -lr * g, grads)
    new_model = eqx.apply_updates(model, updates)
    return new_model, loss


def iterate_minibatches(key: jax.Array, x: jax.Array, y: jax.Array, batch_size: int):
    n = x.shape[0]
    idx = jax.random.permutation(key, n)
    for i in range(0, n, batch_size):
        sl = idx[i : i + batch_size]
        yield x[sl], y[sl]


def main(cfg: TrainConfig = TrainConfig()):
    print("Building data and model (Equinox MLP)...")
    k = jax.random.PRNGKey(cfg.seed)
    k_data, k_model, k_batch = jax.random.split(k, 3)
    x, y = make_synthetic_regression(k_data, cfg.num_samples)
    model = MLP(k_model, in_dim=1, hidden=(64, 64), out_dim=1)

    @eqx.filter_jit
    def eval_loss(m: MLP) -> jax.Array:
        return mse_loss(m, x, y)

    initial = float(eval_loss(model))
    print(f"initial_loss={initial:.6f}")

    for step in range(1, cfg.steps + 1):
        # simple SGD over shuffled minibatches each iteration
        k_batch, k_iter = jax.random.split(k_batch)
        for xb, yb in iterate_minibatches(k_iter, x, y, cfg.batch_size):
            model, loss = train_step(model, xb, yb, cfg.learning_rate)
        if step % 100 == 0 or step == 1:
            full_loss = float(eval_loss(model))
            print(f"step={step:04d} loss={full_loss:.4f}")

    # quick sanity prediction
    xs = jnp.linspace(-2.0, 2.0, 5).reshape(-1, 1)
    ys = jax.vmap(model)(xs)
    print("example preds:")
    for xi, yi in zip(xs, ys):
        print(f"x={float(xi[0]): .3f} -> y={float(yi[0]): .3f}")

    final = float(eval_loss(model))
    print(f"final_loss={final:.6f}")
    improved = final < initial
    print(f"improved={improved}")
    if not improved:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=TrainConfig.steps)
    parser.add_argument("--lr", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--num-samples", type=int, default=TrainConfig.num_samples)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    args = parser.parse_args()
    cfg = TrainConfig(
        seed=args.seed,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        steps=args.steps,
        learning_rate=args.lr,
    )
    main(cfg)


