import future, sys, os, datetime, argparse
# print(os.path.dirname(sys.executable))
import torch
import numpy as np
import matplotlib
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# matplotlib.rcParams["figure.figsize"] = [10, 10]

import torch
from torch.nn import Module, Parameter, Sequential
from torch.nn import Linear, Tanh, ReLU, CELU
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Categorical, Normal

from joblib import Parallel, delayed

Tensor = torch.Tensor
FloatTensor = torch.FloatTensor

torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)

sys.path.append("../../..")  # Up to -> KFAC -> Optimization -> PHD

from pytorch_MCMC.src.MCMC_ProbModel import ProbModel

class GMM(ProbModel):

	def __init__(self):

		dataloader = DataLoader(TensorDataset(torch.zeros(1,2))) # bogus dataloader
		ProbModel.__init__(self, dataloader)

		self.means = FloatTensor([[-1, -1.25], [-1, 1.25], [1.5, 1]])
		# self.means = FloatTensor([[-1,-1.25]])
		self.num_dists = self.means.shape[0]
		I = FloatTensor([[1, 0], [0, 1]])
		I_compl = FloatTensor([[0, 1], [1, 0]])
		self.covars = [I * 0.5, I * 0.5, I * 0.5 + I_compl * 0.3]
		# self.covars = [I * 0.9, I * 0.9, I * 0.9 + I_compl * 0.3]
		self.weights = [0.4, 0.2, 0.4]
		self.dists = []

		for mean, covar in zip(self.means, self.covars):
			self.dists.append(MultivariateNormal(mean, covar))

		self.X_grid = None
		self.Y_grid = None
		self.surface = None

		self.param = torch.nn.Parameter(self.sample())



	def forward(self, x=None):

		log_probs = torch.stack([weight * torch.exp(dist.log_prob(x)) for dist, weight in zip(self.dists, self.weights)], dim=1)
		log_prob = torch.log(torch.sum(log_probs, dim=1))

		return log_prob

	def log_prob(self, *x):

		log_probs = torch.stack([weight * torch.exp(dist.log_prob(self.param)) for dist, weight in zip(self.dists, self.weights)], dim=1)
		log_prob = torch.log(torch.sum(log_probs, dim=1))

		return {'log_prob': log_prob}

	def prob(self, x):

		log_probs = torch.stack([weight * torch.exp(dist.log_prob(x)) for dist, weight in zip(self.dists, self.weights)], dim=1)
		log_prob = torch.sum(log_probs, dim=1)
		return log_prob

	def sample(self, _shape=(1,)):

		probs = torch.ones(self.num_dists) / self.num_dists
		categorical = Categorical(probs)
		sampled_dists = categorical.sample(_shape)

		samples = []
		for sampled_dist in sampled_dists:
			sample = self.dists[sampled_dist].sample((1,))
			samples.append(sample)

		samples = torch.cat(samples)

		return samples

	def reset_parameters(self):

		self.param.data = self.sample()

	def generate_surface(self, plot_min=-3, plot_max=3, plot_res=500, plot=False):

		# print('in surface')

		x = np.linspace(plot_min, plot_max, plot_res)
		y = np.linspace(plot_min, plot_max, plot_res)
		X, Y = np.meshgrid(x, y)

		self.X_grid = X
		self.Y_grid = Y

		points = FloatTensor(np.stack((X.ravel(), Y.ravel())).T)  # .requires_grad_()

		probs = self.prob(points).view(plot_res, plot_res)
		self.surface = probs.numpy()

		area = ((plot_max - plot_min) / plot_res) ** 2
		sum_px = probs.sum() * area  # analogous to integrating cubes: volume is probs are the height times the area

		fig = plt.figure(figsize=(10, 10))

		contour = plt.contourf(self.X_grid, self.Y_grid, self.surface, levels=20)
		plt.xlim(-3, 3)
		plt.ylim(-3, 3)
		plt.grid()
		cbar = fig.colorbar(contour)
		if plot: plt.show()

		return fig

class LinReg(ProbModel):

	def __init__(self, x, y):
		super().__init__()

		self.data = x
		self.target = y

		self.dataloader = DataLoader(TensorDataset(self.data, self.target), shuffle=True, batch_size=self.data.shape[0])

		self.m = Parameter(FloatTensor(1 * torch.randn((1,))))
		self.b = Parameter(FloatTensor(1 * torch.randn((1,))))
		# self.log_noise = Parameter(FloatTensor([-1.]))
		self.log_noise = FloatTensor([0])

	def reset_parameters(self):
		torch.nn.init.normal_(self.m, std=.1)
		torch.nn.init.normal_(self.b, std=.1)

	def sample(self):
		self.reset_parameters()

	def forward(self, x):

		return self.m * x + self.b

	def log_prob(self):

		data, target = next(self.dataloader.__iter__())
		# data, target = self.data, self.target
		mu = self.forward(data)
		log_prob = Normal(mu, F.softplus(self.log_noise)).log_prob(target).mean()

		return {'log_prob': log_prob}

	@torch.no_grad()
	def predict(self, chain):

		x_min = 2*self.data.min()
		x_max = 2*self.data.max()
		data = torch.arange(x_min, x_max).reshape(-1,1)

		pred = []
		for model_state_dict in chain.samples:
			self.load_state_dict(model_state_dict)
			# data.append(self.data)
			pred_i = self.forward(data)
			pred.append(pred_i)

		pred = torch.stack(pred)
		# data = torch.stack(data)

		mu = pred.mean(dim=0).squeeze()
		std = pred.std(dim=0).squeeze()

		# print(f'{data.shape=}')
		# print(f'{pred.shape=}')

		plt.plot(data, mu, alpha=1., color='red')
		plt.fill_between(data.squeeze(), mu+std, mu-std, color='red', alpha=0.25)
		plt.fill_between(data.squeeze(), mu+2*std, mu-2*std, color='red', alpha=0.10)
		plt.fill_between(data.squeeze(), mu+3*std, mu-3*std, color='red', alpha=0.05)
		plt.scatter(self.data, self.target, alpha=1, s=1, color='blue')
		plt.ylim(pred.min(), pred.max())
		plt.xlim(x_min, x_max)
		plt.show()

class RegressionNNHomo(ProbModel):

	def __init__(self, x, y, batch_size=1):

		self.data = x
		self.target = y

		# dataloader = DataLoader(TensorDataset(self.data, self.target), shuffle=True, batch_size=self.data.shape[0], drop_last=False)
		dataloader = DataLoader(TensorDataset(self.data, self.target), shuffle=True, batch_size=batch_size, drop_last=False)

		ProbModel.__init__(self, dataloader)

		num_hidden = 50
		self.model = Sequential(Linear(1, num_hidden),
					ReLU(),
					Linear(num_hidden, num_hidden),
					ReLU(),
					# Linear(num_hidden, num_hidden),
					# ReLU(),
					# Linear(num_hidden, num_hidden),
					# ReLU(),
					Linear(num_hidden, 1))

		self.log_std = Parameter(FloatTensor([-1]))

	def reset_parameters(self):
		for module in self.model.modules():
			if isinstance(module, Linear):
				module.reset_parameters()

		self.log_std.data = FloatTensor([3.])

	def sample(self):
		self.reset_parameters()

	def forward(self, x):
		pred = self.model(x)
		return pred

	def log_prob(self, data, target):

		# if data is None and target is None:
		# 	data, target = next(self.dataloader.__iter__())

		mu = self.forward(data)
		mse = F.mse_loss(mu,target)

		log_prob = Normal(mu, F.softplus(self.log_std)).log_prob(target).mean()*len(self.dataloader.dataset)

		return {'log_prob': log_prob, 'MSE': mse.detach_()}

	def pretrain(self):

		num_epochs = 200
		optim = torch.optim.Adam(self.parameters(), lr=0.01)

		# print(f"{F.softplus(self.log_std)=}")

		progress = tqdm(range(num_epochs))
		for epoch in progress:
			for batch_i, (data, target) in enumerate(self.dataloader):
				optim.zero_grad()
				mu = self.forward(data)
				loss = -Normal(mu, F.softplus(self.log_std)).log_prob(target).mean()
				mse_loss = F.mse_loss(mu, target)
				loss.backward()
				optim.step()

				desc = f'Pretraining: MSE:{mse_loss:.3f}'
				progress.set_description(desc)

		# print(f"{F.softplus(self.log_std)=}")

	@torch.no_grad()
	def predict(self, chains, plot=False):

		x_min = 2*self.data.min()
		x_max = 2*self.data.max()
		data = torch.linspace(x_min, x_max).reshape(-1,1)

		def parallel_predict(parallel_chain):
			parallel_pred = []
			for model_state_dict in parallel_chain.samples[::50]:
				self.load_state_dict(model_state_dict)
				pred_mu_i = self.forward(data)
				parallel_pred.append(pred_mu_i)
			try:
				parallel_pred_mu = torch.stack(parallel_pred) # list [ pred_0, pred_1, ... pred_N] -> Tensor([pred_0, pred_1, ... pred_N])
				return parallel_pred_mu
			except:
				pass


		parallel_pred = Parallel(n_jobs=len(chains))(delayed(parallel_predict)(chain) for chain in chains)

		pred = [parallel_pred_i for parallel_pred_i in parallel_pred if parallel_pred_i is not None] # flatten [ [pred_chain_0], [pred_chain_1] ... [pred_chain_N] ]
		# pred_log_std = [parallel_pred_i for parallel_pred_i in parallel_pred_log_std if parallel_pred_i is not None] # flatten [ [pred_chain_0], [pred_chain_1] ... [pred_chain_N] ]


		pred = torch.cat(pred).squeeze() # cat list of tensors to single prediciton tensor with samples in first dim
		std = F.softplus(self.log_std)


		epistemic = pred.std(dim=0)
		aleatoric = std
		total_std = (epistemic ** 2 + aleatoric ** 2) ** 0.5

		mu = pred.mean(dim=0)
		std = std.mean(dim=0)

		data.squeeze_()

		if plot:
			fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
			axs = axs.flatten()

			axs[0].scatter(self.data, self.target, alpha=1, s=1, color='blue')
			axs[0].plot(data.squeeze(), mu, alpha=1., color='red')
			axs[0].fill_between(data, mu + total_std, mu - total_std, color='red', alpha=0.25)
			axs[0].fill_between(data, mu + 2 * total_std, mu - 2 * total_std, color='red', alpha=0.10)
			axs[0].fill_between(data, mu + 3 * total_std, mu - 3 * total_std, color='red', alpha=0.05)

			[axs[1].plot(data, pred, alpha=0.1, color='red') for pred in pred]
			axs[1].scatter(self.data, self.target, alpha=1, s=1, color='blue')

			axs[2].scatter(self.data, self.target, alpha=1, s=1, color='blue')
			axs[2].plot(data, mu, color='red')
			axs[2].fill_between(data, mu - aleatoric, mu + aleatoric, color='red', alpha=0.25, label='Aleatoric')
			axs[2].legend()

			axs[3].scatter(self.data, self.target, alpha=1, s=1, color='blue')
			axs[3].plot(data, mu, color='red')
			axs[3].fill_between(data, mu - epistemic, mu + epistemic, color='red', alpha=0.25, label='Epistemic')
			axs[3].legend()

			plt.ylim(2 * self.target.min(), 2 * self.target.max())
			plt.xlim(x_min, x_max)
			plt.show()

class RegressionNNHetero(ProbModel):

	def __init__(self, x, y, batch_size=1):

		self.data = x
		self.target = y

		dataloader = DataLoader(TensorDataset(self.data, self.target), shuffle=True, batch_size=self.data.shape[0], drop_last=False)
		# dataloader = DataLoader(TensorDataset(x, y), shuffle=True, batch_size=batch_size, drop_last=False)

		ProbModel.__init__(self, dataloader)

		num_hidden = 50
		self.model = Sequential(Linear(1, num_hidden),
					ReLU(),
					Linear(num_hidden, num_hidden),
					ReLU(),
					Linear(num_hidden, num_hidden),
					ReLU(),
					# Linear(num_hidden, num_hidden),
					# ReLU(),
					Linear(num_hidden, 2))

	def reset_parameters(self):
		for module in self.model.modules():
			if isinstance(module, Linear):
				module.reset_parameters()

	def sample(self):
		self.reset_parameters()

	def forward(self, x):
		pred = self.model(x)
		mu, log_std = torch.chunk(pred, chunks=2, dim=-1)
		return mu, log_std

	def log_prob(self, data, target):

		# if data is None and target is None:
		# 	data, target = next(self.dataloader.__iter__())

		mu, log_std = self.forward(data)
		mse = F.mse_loss(mu,target)

		log_prob = Normal(mu, F.softplus(log_std)).log_prob(target).mean()*len(self.dataloader.dataset)

		return {'log_prob': log_prob, 'MSE': mse.detach_()}

	def pretrain(self):

		num_epochs = 100
		optim = torch.optim.Adam(self.parameters(), lr=0.001)

		progress = tqdm(range(num_epochs))
		for epoch in progress:
			for batch_i, (data, target) in enumerate(self.dataloader):
				optim.zero_grad()
				mu, log_std = self.forward(data)
				loss = -Normal(mu, F.softplus(log_std)).log_prob(target).mean()
				mse_loss = F.mse_loss(mu, target)
				loss.backward()
				optim.step()

				desc = f'Pretraining: MSE:{mse_loss:.3f}'
				progress.set_description(desc)

	@torch.no_grad()
	def predict(self, chains, plot=False):

		x_min = 2*self.data.min()
		x_max = 2*self.data.max()
		data = torch.linspace(x_min, x_max).reshape(-1,1)

		def parallel_predict(parallel_chain):
			parallel_pred_mu = []
			parallel_pred_log_std = []
			for model_state_dict in parallel_chain.samples[::50]:
				self.load_state_dict(model_state_dict)
				pred_mu_i, pred_log_std_i = self.forward(data)
				parallel_pred_mu.append(pred_mu_i)
				parallel_pred_log_std.append(pred_log_std_i)

			try:
				parallel_pred_mu = torch.stack(parallel_pred_mu) # list [ pred_0, pred_1, ... pred_N] -> Tensor([pred_0, pred_1, ... pred_N])
				parallel_pred_log_std = torch.stack(parallel_pred_log_std) # list [ pred_0, pred_1, ... pred_N] -> Tensor([pred_0, pred_1, ... pred_N])
				return parallel_pred_mu, parallel_pred_log_std
			except:
				pass


		parallel_pred_mu, parallel_pred_log_std = zip(*Parallel(n_jobs=len(chains))(delayed(parallel_predict)(chain) for chain in chains))

		pred_mu = [parallel_pred_i for parallel_pred_i in parallel_pred_mu if parallel_pred_i is not None] # flatten [ [pred_chain_0], [pred_chain_1] ... [pred_chain_N] ]
		pred_log_std = [parallel_pred_i for parallel_pred_i in parallel_pred_log_std if parallel_pred_i is not None] # flatten [ [pred_chain_0], [pred_chain_1] ... [pred_chain_N] ]


		pred_mu = torch.cat(pred_mu).squeeze() # cat list of tensors to single prediciton tensor with samples in first dim
		pred_log_std = torch.cat(pred_log_std).squeeze() # cat list of tensors to single prediciton tensor with samples in first dim

		mu 	= pred_mu.squeeze()
		std 	= F.softplus(pred_log_std).squeeze()

		epistemic = mu.std(dim=0)
		aleatoric = (std**2).mean(dim=0)**0.5
		total_std = (epistemic**2 + aleatoric**2)**0.5

		mu = mu.mean(dim=0)
		std = std.mean(dim=0)

		data.squeeze_()

		if plot:

			fig, axs = plt.subplots(2,2, sharex=True, sharey=True)
			axs = axs.flatten()

			axs[0].scatter(self.data, self.target, alpha=1, s=1, color='blue')
			axs[0].plot(data.squeeze(), mu, alpha=1., color='red')
			axs[0].fill_between(data, mu+total_std, mu-total_std, color='red', alpha=0.25)
			axs[0].fill_between(data, mu+2*total_std, mu-2*total_std, color='red', alpha=0.10)
			axs[0].fill_between(data, mu+3*total_std, mu-3*total_std, color='red', alpha=0.05)

			[axs[1].plot(data, pred, alpha=0.1, color='red') for pred in pred_mu]
			axs[1].scatter(self.data, self.target, alpha=1, s=1, color='blue')

			axs[2].scatter(self.data, self.target, alpha=1, s=1, color='blue')
			axs[2].plot(data, mu, color='red')
			axs[2].fill_between(data, mu-aleatoric, mu+aleatoric, color='red', alpha=0.25, label='Aleatoric')
			axs[2].legend()

			axs[3].scatter(self.data, self.target, alpha=1, s=1, color='blue')
			axs[3].plot(data, mu, color='red')
			axs[3].fill_between(data, mu-epistemic, mu+epistemic, color='red', alpha=0.25, label='Epistemic')
			axs[3].legend()

			plt.ylim(2*self.target.min(), 2*self.target.max())
			plt.xlim(x_min, x_max)
			plt.show()


		return data, mu, std


if __name__ == '__main__':
	
	pass