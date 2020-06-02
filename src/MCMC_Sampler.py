
import os, sys, copy, time
from tqdm import tqdm
from collections import MutableSequence, OrderedDict
import matplotlib
import matplotlib.pyplot as plt

import numpy as np

from pytorch_MCMC.src.MCMC_Utils import posterior_dist
from pytorch_MCMC.src.MCMC_Optim import SGLD_Optim, MetropolisHastings_Optim
from pytorch_MCMC.src.MCMC_Acceptance import MetropolisHastingsAcceptance
from pytorch_MCMC.src.MCMC_Chain import Chain, SGLD_Chain, MALA_Chain, HMC_Chain, SGNHT_Chain
from pytorch_MCMC.src.MCMC_Acceptance import MetropolisHastingsAcceptance, SDE_Acceptance
from pytorch_MCMC.src.MCMC_ProbModel import ProbModel

from joblib import Parallel, delayed
import concurrent.futures


import torch
from torch.nn import Parameter
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.distributions.distribution import Distribution
from torch.distributions import MultivariateNormal, Normal
from torch.nn import Module


class Sampler:

	def __init__(self, probmodel, step_size, num_steps, num_chains, burn_in, pretrain, tune):

		self.probmodel		= probmodel
		self.chain 		= None
		self.num_chains		= num_chains

		self.step_size 		= step_size
		self.num_steps 		= num_steps
		self.burn_in 		= burn_in

		self.pretrain		= pretrain
		self.tune		= tune

		test_log_prob = self.probmodel.log_prob(*next(self.probmodel.dataloader.__iter__()))
		assert type(test_log_prob) == dict
		assert list(test_log_prob.keys())[0]=='log_prob'

	def sample_chains(self):
		raise NotImplementedError

	def __str__(self):
		raise NotImplementedError

	def multiprocessing_test(self, wait_time):

		time.sleep(wait_time)
		print(f'Done after {wait_time=} seconds')

	def sample_independent_chain(self):

		probmodel = copy.deepcopy(self.probmodel)
		probmodel.reset_parameters()

		if self.pretrain:
			probmodel.pretrain()

		optim = SGLD_Optim(probmodel, step_size=self.step_size, prior_std=0., addnoise=True)
		chain = Chain(probmodel=probmodel)

		progress = tqdm(range(self.num_steps))
		for step in progress:

			sample_log_prob, sample = self.propose(probmodel, optim)

			accept, log_ratio = self.acceptance(sample_log_prob['log_prob'], chain.state['log_prob'])

			chain += (probmodel, sample_log_prob, accept, step)

			if not accept:
				probmodel.load_state_dict(chain.state['state'])

			desc = f'{str(self)}: Accept: {chain.running_accepts.avg:.2f}/{chain.accept_ratio:.2f} \t'
			for key, running_avg in chain.running_avgs.items():
				# print(f'{key}: {running_avg.avg=}')
				desc += f' {key}: {running_avg.avg:.2f} '
			desc += f'StepSize: {optim.param_groups[0]["lr"]:.3f}'
			progress.set_description(desc=desc)

		# print(list(chain.samples[-1].values())[-1][0])
		'''
		Remove Burn_in
		'''
		assert len(chain.accepted_steps) > self.burn_in, f'{len(chain.accepted_steps)=} <= {self.burn_in=}'
		chain.accepted_steps = chain.accepted_steps[self.burn_in:]

		return chain

	def sample_chain(self, step_size=None):

		if self.pretrain:
			self.probmodel.pretrain()

		self.optim = SGLD_Optim(self.probmodel,
					step_size=step_size,
					prior_std=0.,
					addnoise=True)

		if self.tune: self.tune_step_size()

		self.chain = Chain(probmodel=self.probmodel)

		progress = tqdm(range(self.num_steps))
		for step in progress:

			sample_log_prob, sample = self.propose()

			accept, log_ratio = self.acceptance(sample_log_prob['log_prob'], self.chain.state['log_prob'])

			self.chain += (self.probmodel, sample_log_prob, accept, step)

			if not accept:
				self.probmodel.load_state_dict(self.chain.state['state'])

			desc = f'{str(self)}: Accept: {self.chain.running_accepts.avg:.2f}/{self.chain.accept_ratio:.2f} \t'
			for key, running_avg in self.chain.running_avgs.items():
				# print(f'{key}: {running_avg.avg=}')
				desc+= f' {key}: {running_avg.avg:.2f} '
			desc += f'StepSize: {self.optim.param_groups[0]["lr"]:.3f}'
			progress.set_description(desc=desc)

		print(len(self.chain))
		'''
		Remove Burn_in
		'''
		assert len(self.chain.accepted_steps)>self.burn_in, f'{len(self.chain.accepted_steps)=} <= {self.burn_in=}'
		self.chain.accepted_steps = self.chain.accepted_steps[self.burn_in:]

	def posterior_dist(self, param=None, verbose=False, plot=True):

		if len(self.probmodel.state_dict())==1:
			'''
			We're sampling from a predefined distribution like a GMM and simulating a particle
			'''
			post = []

			accepted_models = self.chain.samples
			for model_state_dict in accepted_models:
				post.append(list(model_state_dict.values())[0])

			post = torch.cat(post, dim=0)

			if plot:
				hist2d = plt.hist2d(x=post[:, 0].cpu().numpy(), y=post[:, 1].cpu().numpy(), bins=100, range=np.array([[-3, 3], [-3, 3]]),
						    density=True)
				plt.colorbar(hist2d[3])
				plt.show()

		elif len(self.probmodel.state_dict()) > 1:
			'''
			There is more than one parameter in the model
			'''

			param_names = list(self.probmodel.state_dict().keys())
			accepted_models = self.chain.samples

			for param_name in param_names:

				post = []

				for model_state_dict in accepted_models:

					post.append(model_state_dict[param_name])

				post = torch.cat(post)
				# print(post)

				if plot:
					plt.hist(x=post, bins=50,
							    range=np.array([-3, 3]),
							    density=True,
						 alpha=0.5)
					plt.title(param_name)
				plt.show()

	def trace(self, param=None, verbose=False, plot=True):

		if len(self.probmodel.state_dict()) >= 1:
			'''
			There is more than one parameter in the model
			'''

			param_names = list(self.probmodel.state_dict().keys())
			accepted_models = [self.chain.samples[idx] for idx in self.chain.accepted_steps]

			for param_name in param_names:

				post = []

				for model_state_dict in accepted_models:
					post.append(model_state_dict[param_name])

				# print(post)

				post = torch.cat(post)
				# print(post)

				if plot:
					plt.plot(np.arange(len(accepted_models)), post)
					plt.title(param_name)

				plt.show()

class MetropolisHastings_Sampler(Sampler):

	def __init__(self, probmodel, step_size=1., num_steps=10000, burn_in=100, pretrain=False, tune=True):
		'''

		:param probmodel: Probmodel() that implements forward, log_prob, prob and sample
		:param step_length:
		:param num_steps:
		:param burn_in:
		'''

		assert isinstance(probmodel, ProbModel)
		super().__init__(probmodel, step_size, num_steps, burn_in, pretrain, tune)

		self.optim = MetropolisHastings_Optim(self.probmodel,
						      step_length=step_size)

		self.acceptance = MetropolisHastingsAcceptance()

	def __str__(self):
		return 'MH'

	@torch.no_grad()
	def propose(self):
		self.optim.step()
		log_prob = self.probmodel.log_prob()

		return log_prob, self.probmodel

class SGLD_Sampler(Sampler):

	def __init__(self, probmodel, step_size=0.01, num_steps=10000, num_chains=7, burn_in=500, pretrain=True, tune=True):
		'''

		:param probmodel: Probmodel() that implements forward, log_prob, prob and sample
		:param step_length:
		:param num_steps:
		:param burn_in:
		'''

		assert isinstance(probmodel, ProbModel)
		Sampler.__init__(self, probmodel, step_size, num_steps, num_chains, burn_in, pretrain, tune)

	def sample_chains(self):

		if self.num_chains > 1:
			self.parallel_chains = [SGLD_Chain(copy.deepcopy(self.probmodel),
							   step_size=self.step_size,
							   num_steps=self.num_steps,
							   burn_in=self.burn_in,
							   pretrain=self.pretrain,
							   tune=False)
						for _ in range(self.num_chains)]

			chains = Parallel(n_jobs=self.num_chains)(delayed(chain.sample_chain)() for chain in self.parallel_chains)

		elif self.num_chains == 1:
			chain = SGLD_Chain(copy.deepcopy(self.probmodel),
					   step_size=self.step_size,
					   num_steps=self.num_steps,
					   burn_in=self.burn_in,
					   pretrain=self.pretrain,
					   tune=False)
			chains = [chain.sample_chain()]

		self.chain = Chain(probmodel=self.probmodel)

		for chain in chains:
			self.chain += chain

		return chains

	def __str__(self):
		return 'SGLD'

class MALA_Sampler(Sampler):

	def __init__(self, probmodel, step_size=0.01, num_steps=10000, num_chains=4, burn_in=500, pretrain=True, tune=True):
		'''

		:param probmodel: Probmodel() that implements forward, log_prob, prob and sample
		:param step_length:
		:param num_steps:
		:param burn_in:
		'''

		assert isinstance(probmodel, ProbModel)
		super().__init__(probmodel, step_size, num_steps, num_chains, burn_in, pretrain, tune)

	def sample_chains(self):

		if self.num_chains>1:
			self.parallel_chains = [MALA_Chain(copy.deepcopy(self.probmodel),
							   step_size=self.step_size,
							   num_steps=self.num_steps,
							   burn_in=self.burn_in,
							   pretrain=self.pretrain,
							   tune=self.tune,
							   num_chain=i)
						for i in range(self.num_chains)]

			chains = Parallel(n_jobs=self.num_chains)(delayed(chain.sample_chain)() for chain in self.parallel_chains)

		elif self.num_chains == 1:
			chain = MALA_Chain(copy.deepcopy(self.probmodel),
					   step_size=self.step_size,
					   num_steps=self.num_steps,
					   burn_in=self.burn_in,
					   pretrain=self.pretrain,
					   tune=self.tune,
					   num_chain=0)
			chains = [chain.sample_chain()]

		self.chain = Chain(probmodel=self.probmodel)


		for chain in chains:
			self.chain += chain

		return chains

	def __str__(self):
		return 'SGLD'

class HMC_Sampler(Sampler):

	def __init__(self, probmodel, step_size=0.01, num_steps=10000, num_chains=7, burn_in=500, pretrain=True, tune=True,
		     traj_length=21):
		'''

		:param probmodel: Probmodel() that implements forward, log_prob, prob and sample
		:param step_length:
		:param num_steps:
		:param burn_in:
		'''

		assert isinstance(probmodel, ProbModel)
		Sampler.__init__(self, probmodel, step_size, num_steps, num_chains, burn_in, pretrain, tune)

		self.traj_length = traj_length

	def __str__(self):
		return 'HMC'

	def sample_chains(self):

		if self.num_chains > 1:
			self.parallel_chains = [HMC_Chain(copy.deepcopy(self.probmodel),
							   step_size=self.step_size,
							   num_steps=self.num_steps,
							   burn_in=self.burn_in,
							   pretrain=self.pretrain,
							   tune=self.tune)
						for i in range(self.num_chains)]

			chains = Parallel(n_jobs=self.num_chains)(delayed(chain.sample_chain)() for chain in self.parallel_chains)

		elif self.num_chains == 1:
			chain = HMC_Chain(copy.deepcopy(self.probmodel),
					   step_size=self.step_size,
					   num_steps=self.num_steps,
					   burn_in=self.burn_in,
					   pretrain=self.pretrain,
					   tune=self.tune)
			chains = [chain.sample_chain()]

		self.chain = Chain(probmodel=self.probmodel) # the aggregating chain

		for chain in chains:
			self.chain += chain

		return chains

class SGNHT_Sampler(Sampler):

	def __init__(self, probmodel, step_size=0.01, num_steps=10000, num_chains=7, burn_in=500, pretrain=True, tune=True,
		     traj_length=21):
		'''

		:param probmodel: Probmodel() that implements forward, log_prob, prob and sample
		:param step_length:
		:param num_steps:
		:param burn_in:
		'''

		assert isinstance(probmodel, ProbModel)
		Sampler.__init__(self, probmodel, step_size, num_steps, num_chains, burn_in, pretrain, tune)

		self.traj_length = traj_length

	def __str__(self):
		return 'SGNHT'

	def sample_chains(self):

		if self.num_chains > 1:
			self.parallel_chains = [SGNHT_Chain(copy.deepcopy(self.probmodel),
							   step_size=self.step_size,
							   num_steps=self.num_steps,
							   burn_in=self.burn_in,
							   pretrain=self.pretrain,
							   tune=self.tune)
						for i in range(self.num_chains)]

			chains = Parallel(n_jobs=self.num_chains)(delayed(chain.sample_chain)() for chain in self.parallel_chains)

		elif self.num_chains == 1:
			chain = SGNHT_Chain(copy.deepcopy(self.probmodel),
					   step_size=self.step_size,
					   num_steps=self.num_steps,
					   burn_in=self.burn_in,
					   pretrain=self.pretrain,
					   tune=self.tune)
			chains = [chain.sample_chain()]

		self.chain = Chain(probmodel=self.probmodel) # the aggregating chain

		for chain in chains:
			self.chain += chain

		return chains