import torch
from tensordict import TensorDict
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass
from numbers import Number

plt.style.use("default")
plt.rcParams["axes.facecolor"] = (
    0.1171,
    0.1171,
    0.1171,
)
plt.rcParams["figure.facecolor"] = (
    0.1171,
    0.1171,
    0.1171,
)
plt.rcParams["text.color"] = "white"
plt.rcParams["axes.labelcolor"] = "white"
plt.rcParams["xtick.color"] = "white"
plt.rcParams["ytick.color"] = "white"
plt.rcParams["axes.edgecolor"] = "white"
plt.rcParams["axes.titlecolor"] = "white"
plt.rcParams["figure.edgecolor"] = "white"
plt.rcParams["legend.edgecolor"] = "white"
plt.rcParams["legend.facecolor"] = (0.1171, 0.1171, 0.1171)


class GaussianMixture1D:
    def __init__(self, means, stds, weights):
        self.means = torch.tensor(means, dtype=torch.float32)
        self.stds = torch.tensor(stds, dtype=torch.float32)
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.weights = self.weights / self.weights.sum()  # Normalize weights

    def sample(self, num_samples=1):
        component = torch.multinomial(self.weights, num_samples, replacement=True)
        samples = torch.normal(self.means[component], self.stds[component])
        return samples

    def prob(self, x):
        x = x.unsqueeze(-1)  # Shape (N, 1)
        probs = (
            1
            / (self.stds * torch.sqrt(2 * torch.tensor(torch.pi)))
            * torch.exp(-0.5 * (x - self.means) ** 2 / (self.stds**2))
        )
        weighted_probs = probs * self.weights
        return weighted_probs.sum(dim=-1)

    def log_prob(self, x):
        x = x.unsqueeze(-1)
        log_probs = (
            -0.5 * torch.log(2 * torch.tensor(torch.pi) * self.stds**2)
            - 0.5 * (x - self.means) ** 2 / (self.stds**2)
            + torch.log(self.weights)
        )
        return torch.logsumexp(log_probs, dim=-1)

    def energy(self, x):
        # Negative log probability (up to constant)
        return self.log_prob(x) + 1

    def neg_energy(self, x):
        # Negative log probability (up to constant)
        return -self.log_prob(x) + 1


def tune(scale, acceptance):
    """Borrowed from PyMC3"""

    # Switch statement
    if acceptance < 0.001:
        # reduce by 90 percent
        scale *= 0.1
    elif acceptance < 0.05:
        # reduce by 50 percent
        scale *= 0.5
    elif acceptance < 0.2:
        # reduce by ten percent
        scale *= 0.9
    elif acceptance > 0.95:
        # increase by factor of ten
        scale *= 10.0
    elif acceptance > 0.75:
        # increase by double
        scale *= 2.0
    elif acceptance > 0.5:
        # increase by ten percent
        scale *= 1.1

    return scale


class EMA:
    def __init__(self, ema_weight=0.999):
        self.ema_weight = ema_weight
        self.step = 0
        self.val_ = None

    def __call__(self, val: Number):
        if self.val_ is None:
            self.val_ = val
            self.ema_correction = 1.0
        else:
            self.val_ = self.val_ * self.ema_weight + val * (1 - self.ema_weight)
            self.ema_correction = 1 - self.ema_weight ** (self.step + 1)
            self.step += 1
        return self.val_ / self.ema_correction

    @property
    def val(self):
        return self.val_


num_chains = 1000
proposal_std = 0.1
gmm = GaussianMixture1D(
    means=[-3.0, 0.0, 3.0],
    stds=[0.5, 0.5, 0.5],
    weights=[2, 0.3, 0.1],
)
vmap_energy = torch.vmap(gmm.energy, (0,))
init_sample = TensorDict({"x": 3 * torch.randn((num_chains, 1))})
init_energy = vmap_energy(init_sample["x"])
chain = [(init_sample, init_energy)]

num_steps = 10_000
accept_ratio_ema = EMA(ema_weight=0.99)
pbar = tqdm(range(num_steps))
for step in pbar:
    state, energy = chain[-1]
    proposal_state = state.clone().apply(
        lambda x: x + torch.randn_like(x) * proposal_std
    )
    proposal_energy = vmap_energy(proposal_state["x"])
    log_ratio = proposal_energy - energy
    log_ratio = torch.min(log_ratio, torch.zeros_like(log_ratio))
    log_u = torch.zeros_like(log_ratio).uniform_(0, 1).log()
    log_accept = torch.gt(log_ratio, log_u)
    # new_state = torch.where(log_accept, proposal_state, state)
    new_state = state.apply(
        lambda state_, proposal_state_: torch.where(
            log_accept, proposal_state_, state_
        ),
        proposal_state,
    )
    chain += [(new_state, proposal_energy)]
    accept_ratio = log_accept.sum() / log_accept.numel()
    accept_ratio_ema(accept_ratio.item())
    # print(accept_ratio_ema.val)
    # if step % 100 == 0:
    # proposal_std = tune(proposal_std, accept_ratio)
    pbar.set_postfix({"Accept": float(accept_ratio_ema.val), "PropStd": proposal_std})

data = chain[-1][0]["x"]
data = data[-5 <= data]
data = data[data <= 5]
plt.hist(chain[0][0]["x"], density=True, bins=100, color="green", alpha=0.5)
plt.hist(data, density=True, bins=100, color="b", alpha=0.5)

plt.plot(torch.linspace(-5, 5, 100), gmm.prob(torch.linspace(-5, 5, 100)))
# plt.plot(torch.linspace(-5, 5, 100), -gmm.energy(torch.linspace(-5, 5, 100)))
