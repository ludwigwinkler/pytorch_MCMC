# %%
import torch
from tensordict import TensorDict
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass
from numbers import Number

# import mcmc.sampler.MetropolisHastingAcceptance
# from mcmc.sampler import MetropolisHastingAcceptance

from mcmc.sampler import MetropolisHastingsAcceptance
from mcmc.energy import GaussianMixture1D, GaussianMixture2D
from mcmc.utils import EMA

plt.style.use("default")
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["text.color"] = "black"
plt.rcParams["axes.labelcolor"] = "black"
plt.rcParams["xtick.color"] = "black"
plt.rcParams["ytick.color"] = "black"
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.titlecolor"] = "black"
plt.rcParams["figure.edgecolor"] = "white"
plt.rcParams["legend.edgecolor"] = "black"
plt.rcParams["legend.facecolor"] = "white"


gmm2d = GaussianMixture2D()
samples2d = gmm2d.sample(50_000)

# 2D contour plot of samples2d
x = samples2d[:, 0].numpy()
y = samples2d[:, 1].numpy()
plt.figure(figsize=(6, 5))
plt.hist2d(x, y, bins=100, density=True, cmap="viridis")
plt.colorbar(label="Density")
plt.xlabel("x")
plt.ylabel("y")
plt.title("2D Gaussian Mixture Samples")
plt.show()

# %%
num_chains = 5000
proposal_std = 0.5
gmm = GaussianMixture1D(
    means=[-3.0, 0.0, 3.0],
    stds=[0.5, 0.5, 0.5],
    weights=[2, 0.3, 0.1],
)
vmap_energy = torch.vmap(gmm.energy, (0,))
init_sample = TensorDict({"x": 3 * torch.randn((num_chains, 1)).clamp(-5, 5)})
init_energy = vmap_energy(init_sample["x"])
chain = [(init_sample, init_energy)]


num_steps = 5_000
accept_ema = EMA(ema_weight=0.99)
pbar = tqdm(range(num_steps))
MH = MetropolisHastingsAcceptance()
for step in pbar:
    state, energy = chain[-1]
    proposal_state = state.clone().apply(
        lambda x: x + torch.randn_like(x) * proposal_std
    )
    proposal_energy = vmap_energy(proposal_state["x"])
    accept: torch.Tensor = MH(energy, proposal_energy)
    new_state = state.apply(
        lambda state_, proposal_state_: torch.where(accept, proposal_state_, state_),
        proposal_state,
    )
    chain += [(new_state, proposal_energy)]
    accept_ratio = accept.sum() / accept.numel()
    accept_ema(accept_ratio.item())
    pbar.set_postfix({"Accept": float(accept_ema.val), "PropStd": proposal_std})

data = chain[-1][0]["x"]
data = data[-5 <= data]
data = data[data <= 5]
plt.hist(chain[0][0]["x"], density=True, bins=50, color="green", alpha=0.5)
plt.hist(data, density=True, bins=50, color="b", alpha=0.5)

plt.plot(torch.linspace(-5, 5, 100), gmm.prob(torch.linspace(-5, 5, 100)))
# plt.plot(torch.linspace(-5, 5, 100), -gmm.energy(torch.linspace(-5, 5, 100)))
