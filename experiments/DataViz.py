import future, sys, os, datetime, argparse
# print(os.path.dirname(sys.executable))
import torch
import numpy as np
import matplotlib
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.rcParams["figure.figsize"] = [10, 10]

import torch
from torch.nn import Module, Parameter
from torch.nn import Linear, Tanh, ReLU
import torch.nn.functional as F

Tensor = torch.Tensor
FloatTensor = torch.FloatTensor

torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)

sys.path.append("../../..")  # Up to -> KFAC -> Optimization -> PHD

import scipy
import scipy as sp
from scipy.io import loadmat as sp_loadmat
import copy

cwd = os.path.abspath(os.getcwd())
os.chdir(cwd)

from pytorch_MCMC.src.MCMC_ProbModel import ProbModel
from pytorch_MCMC.models.MCMC_Models import GMM, LinReg, RegressionNN
from pytorch_MCMC.src.MCMC_Sampler import SGLD_Sampler, MetropolisHastings_Sampler, MALA_Sampler, HMC_Sampler
from pytorch_MCMC.data.MCMC_SyntheticData import generate_linear_regression_data, generate_multimodal_linear_regression, generate_nonstationary_data
from pytorch_MCMC.src.MCMC_Utils import posterior_dist
from Utils.Utils import RunningAverageMeter, str2bool

def create_supervised_gif(model, chain, data):

	x, y = data
	x_min = 2 * x.min()
	x_max = 2 * x.max()

	data, mu, _ = model.predict(chain)

	gif_frames = []

	samples = [400, 600, 800, 1000]
	samples += range(2000, len(chain)//2, 2000)
	samples += range(len(chain)//2, len(chain), 4000)

	# print(len(samples))
	# exit()

	for i in range(400,len(chain), 500):
		print(f"{i}/{len(samples)}")

		# _, _, std = model.predict(chain[:i])
		fig = plt.figure()
		_, mu, std = model.predict(chain[399:i])
		plt.fill_between(data.squeeze(), mu + std, mu - std, color='red', alpha=0.25)
		plt.fill_between(data.squeeze(), mu + 2 * std, mu - 2 * std, color='red', alpha=0.10)
		plt.fill_between(data.squeeze(), mu + 3 * std, mu - 3 * std, color='red', alpha=0.05)

		plt.plot(data.squeeze(), mu, c='red')
		plt.scatter(x, y, alpha=1, s=1, color='blue')
		plt.ylim(2 * y.min(), 2 * y.max())
		plt.xlim(x_min, x_max)
		plt.grid()

		fig.canvas.draw()  # draw the canvas, cache the renderer
		image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
		image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

		# plt.show()
		gif_frames.append(image)

	import imageio
	imageio.mimsave('HMC_Sampler5.gif', gif_frames, fps=4)

def create_gmm_gif(chains):

	# num_samples = [40, 80, 120, 160, 200, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
	num_samples = [x for x in range(3,2000,40)]
	num_samples += [x for x in range(2000, len(chains[0]), 500)]

	gif_frames = []

	for num_samples_ in num_samples:

		print(f"{num_samples_}/{len(chains[0])}")

		post = []

		for chain in chains:

			for model_state_dict in chain.samples[:num_samples_]:
				post.append(list(model_state_dict.values())[0])

		post = torch.cat(post, dim=0)

		fig = plt.figure()
		hist2d = plt.hist2d(x=post[:, 0].cpu().numpy(), y=post[:, 1].cpu().numpy(), bins=100, range=np.array([[-3, 3], [-3, 3]]),
				    density=True)
		plt.colorbar(hist2d[3])
		# plt.show()


		fig.canvas.draw()  # draw the canvas, cache the renderer
		image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
		image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))


		gif_frames.append(image)

	import imageio
	imageio.mimsave('GMM_HMC1.gif', gif_frames, fps=4)



if True:
	chain = torch.load("hmc_regnn_ss0.01_len10000.chain")
	chain = chain[:50000]
	data = generate_nonstationary_data(num_samples=1000, plot=False, x_noise_std=0.01, y_noise_std=0.1)
	nn = RegressionNN(*data, batch_size=50)

	create_supervised_gif(nn, chain, data)

if False:

	chains = torch.load("GMM_Chains.chain")

	create_gmm_gif(chains)

	posterior_dist(chains[0][:50])
	plt.show()