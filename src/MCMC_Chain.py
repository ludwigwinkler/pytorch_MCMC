import future, sys, os, datetime, argparse, copy, warnings, time
from collections import MutableSequence, Iterable, OrderedDict
from itertools import compress
import numpy as np
from tqdm import tqdm

import matplotlib
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
from pytorch_MCMC.src.MCMC_ProbModel import ProbModel
from pytorch_MCMC.src.MCMC_Optim import SGLD_Optim, MetropolisHastings_Optim, MALA_Optim, HMC_Optim, SGNHT_Optim
from pytorch_MCMC.src.MCMC_Acceptance import SDE_Acceptance, MetropolisHastingsAcceptance
from Utils.Utils import RunningAverageMeter

'''
Python Container Time Complexity: https://wiki.python.org/moin/TimeComplexity
'''


class Chain(MutableSequence):

	'''
	A container for storing the MCMC chain conveniently:
	samples: list of state_dicts
	log_probs: list of log_probs
	accepts: list of bools
	state_idx:
		init index of last accepted via np.where(accepts==True)[0][-1]
		can be set via len(samples) while sampling

	@property
	samples: filters the samples


	'''

	def __init__(self, probmodel=None):

		super().__init__()

		if probmodel is None:
			'''
			Create an empty chain
			'''
			self.state_dicts = []
			self.log_probs = []
			self.accepts = []

		if probmodel is not None:
			'''
			Initialize chain with given model
			'''
			assert isinstance(probmodel, ProbModel)

			self.state_dicts = [copy.deepcopy(probmodel.state_dict())]
			log_prob = probmodel.log_prob(*next(probmodel.dataloader.__iter__()))
			log_prob['log_prob'].detach_()
			self.log_probs = [copy.deepcopy(log_prob)]
			self.accepts = [True]
			self.last_accepted_idx = 0

			self.running_avgs = {}
			for key, value in log_prob.items():
				self.running_avgs.update({key: RunningAverageMeter(0.99)})

		self.running_accepts = RunningAverageMeter(0.999)

	def __len__(self):
		return len(self.state_dicts)

	def __iter__(self):
		return zip(self.state_dicts, self.log_probs, self.accepts)

	def __delitem__(self):
		raise NotImplementedError

	def __setitem__(self):
		raise NotImplementedError

	def insert(self):
		raise NotImplementedError

	def __repr__(self):
		return f'MCMC Chain: Length:{len(self)} Accept:{self.accept_ratio:.2f}'

	def __getitem__(self, i):
		chain = copy.deepcopy(self)
		chain.state_dicts = self.samples[i]
		chain.log_probs = self.log_probs[i]
		chain.accepts = self.accepts[i]
		return chain

	def __add__(self, other):

		if type(other) in [tuple, list]:
			assert len(other) == 3, f"Invalid number of information pieces passed: {len(other)} vs len(Iterable(model, log_prob, accept, ratio))==4"
			self.append(*other)
		elif isinstance(other, Chain):
			self.cat(other)

		return self

	def __iadd__(self, other):

		if type(other) in [tuple, list]:
			assert len(other)==3, f"Invalid number of information pieces passed: {len(other)} vs len(Iterable(model, log_prob, accept, ratio))==4"
			self.append(*other)
		elif isinstance(other, Chain):
			self.cat_chains(other)

		return self

	@property
	def state_idx(self):
		'''
		Returns the index of the last accepted sample a.k.a. the state of the chain

		'''
		if not hasattr(self, 'state_idx'):
			'''
			If the chain hasn't a state_idx, compute it from self.accepts by taking the last True of self.accepts
			'''
			self.last_accepted_idx = np.where(self.accepts==True)[0][-1]
			return self.last_accepted_idx
		else:
			'''
			Check that the state of the chain is actually the last True in self.accepts
			'''
			last_accepted_sample_ = np.where(self.accepts == True)[0][-1]
			assert last_accepted_sample_ == self.last_accepted_idx
			assert self.accepts[self.last_accepted_idx]==True
			return self.last_accepted_idx


	@property
	def samples(self):
		'''
		Filters the list of state_dicts with the list of bools from self.accepts
		:return: list of accepted state_dicts
		'''
		return list(compress(self.state_dicts, self.accepts))

	@property
	def accept_ratio(self):
		'''
		Sum the boolean list (=total number of Trues) and divides it by its length
		:return: float valued accept ratio
		'''
		return sum(self.accepts)/len(self.accepts)

	@property
	def state(self):
		return {'state_dict': self.state_dicts[self.last_accepted_idx], 'log_prob': self.log_probs[self.last_accepted_idx]}

	def cat_chains(self, other):

		assert isinstance(other, Chain)
		self.state_dicts += other.state_dicts
		self.log_probs += other.log_probs
		self.accepts += other.accepts

		for key, value in other.running_avgs.items():
			self.running_avgs[key].avg = 0.5*self.running_avgs[key].avg + 0.5 * other.running_avgs[key].avg


	def append(self, probmodel, log_prob, accept):

		if isinstance(probmodel, ProbModel):
			params_state_dict = copy.deepcopy(probmodel.state_dict())
		elif isinstance(probmodel, OrderedDict):
			params_state_dict = copy.deepcopy(probmodel)
		assert isinstance(log_prob, dict)
		assert type(log_prob['log_prob'])==torch.Tensor
		assert log_prob['log_prob'].numel()==1

		log_prob['log_prob'].detach_()


		self.accepts.append(accept)
		self.running_accepts.update(1 * accept)

		if accept:
			self.state_dicts.append(params_state_dict)
			self.log_probs.append(copy.deepcopy(log_prob))
			self.last_accepted_idx = len(self.state_dicts)-1
			for key, value in log_prob.items():
				self.running_avgs[key].update(value.item())

		elif not accept:
			self.state_dicts.append(False)
			self.log_probs.append(False)

class Sampler_Chain:

	def __init__(self, probmodel, step_size, num_steps, burn_in, pretrain, tune):

		self.probmodel = probmodel
		self.chain = Chain(probmodel=self.probmodel)

		self.step_size = step_size
		self.num_steps = num_steps
		self.burn_in = burn_in

		self.pretrain = pretrain
		self.tune = tune

	def propose(self):
		raise NotImplementedError

	def __repr__(self):
		raise NotImplementedError

	def tune_step_size(self):

		tune_interval_length = 100
		num_tune_intervals = int(self.burn_in // tune_interval_length)

		verbose = True

		print(f'Tuning: Init Step Size: {self.optim.param_groups[0]["step_size"]:.5f}')

		self.probmodel.reset_parameters()
		tune_chain = Chain(probmodel=self.probmodel)
		tune_chain.running_accepts.momentum = 0.5

		progress = tqdm(range(self.burn_in))
		for tune_step in progress:



			sample_log_prob, sample = self.propose()
			accept, log_ratio = self.acceptance(sample_log_prob['log_prob'], self.chain.state['log_prob']['log_prob'])
			tune_chain += (self.probmodel, sample_log_prob, accept)

			# if tune_step < self.burn_in and tune_step % tune_interval_length == 0 and tune_step > 0:
			if tune_step > 1:
				# self.optim.dual_average_tune(tune_chain, np.exp(log_ratio.item()))
				self.optim.dual_average_tune(tune_chain.accepts[-tune_interval_length:], tune_step, np.exp(log_ratio.item()))
				# self.optim.tune(tune_chain.accepts[-tune_interval_length:])

			if not accept:

				if torch.isnan(sample_log_prob['log_prob']):
					print(self.chain.state)
					exit()
				self.probmodel.load_state_dict(self.chain.state['state_dict'])

			desc = f'Tuning: Accept: {tune_chain.running_accepts.avg:.2f}/{tune_chain.accept_ratio:.2f} StepSize: {self.optim.param_groups[0]["step_size"]:.5f}'

			progress.set_description(
				desc=desc)



		time.sleep(0.1)  # for cleaner printing in the console

	def sample_chain(self):

		self.probmodel.reset_parameters()

		if self.pretrain:
			try:
				self.probmodel.pretrain()
			except:
				warnings.warn(f'Tried pretraining but couldnt find a probmodel.pretrain() method ... Continuing wihtout pretraining.')

		if self.tune:
			self.tune_step_size()

		# print(f"After Tuning Step Size: {self.optim.param_groups[0]['step_size']=}")

		self.chain = Chain(probmodel=self.probmodel)

		progress = tqdm(range(self.num_steps))
		for step in progress:

			proposal_log_prob, sample = self.propose()
			accept, log_ratio = self.acceptance(proposal_log_prob['log_prob'], self.chain.state['log_prob']['log_prob'])
			self.chain += (self.probmodel, proposal_log_prob, accept)

			if not accept:

				if torch.isnan(proposal_log_prob['log_prob']):
					print(self.chain.state)
					exit()
				self.probmodel.load_state_dict(self.chain.state['state_dict'])

			desc = f'{str(self)}: Accept: {self.chain.running_accepts.avg:.2f}/{self.chain.accept_ratio:.2f} \t'
			for key, running_avg in self.chain.running_avgs.items():
				desc += f' {key}: {running_avg.avg:.2f} '
			desc += f'StepSize: {self.optim.param_groups[0]["step_size"]:.3f}'
			# desc +=f" Std: {F.softplus(self.probmodel.log_std.detach()).item():.3f}"
			progress.set_description(desc=desc)

		self.chain = self.chain[self.burn_in:]

		return self.chain

class SGLD_Chain(Sampler_Chain):

	def __init__(self, probmodel, step_size=0.0001, num_steps=2000, burn_in=100, pretrain=False, tune=False):

		Sampler_Chain.__init__(self, probmodel, step_size, num_steps, burn_in, pretrain, tune)

		self.optim = SGLD_Optim(self.probmodel,
					step_size=step_size,
					prior_std=1.,
					addnoise=True)

		self.acceptance = SDE_Acceptance()

	def __repr__(self):
		return 'SGLD'

	@torch.enable_grad()
	def propose(self):

		self.optim.zero_grad()
		batch = next(self.probmodel.dataloader.__iter__())
		log_prob = self.probmodel.log_prob(*batch)
		(-log_prob['log_prob']).backward()
		self.optim.step()

		return log_prob, self.probmodel

class MALA_Chain(Sampler_Chain):

	def __init__(self, probmodel, step_size=0.1, num_steps=2000, burn_in=100, pretrain=False, tune=False, num_chain=0):

		Sampler_Chain.__init__(self, probmodel, step_size, num_steps, burn_in, pretrain, tune)

		self.num_chain = num_chain

		self.optim = MALA_Optim(self.probmodel,
					step_size=step_size,
					prior_std=1.,
					addnoise=True)

		self.acceptance = MetropolisHastingsAcceptance()
		# self.acceptance = SDE_Acceptance()

	def __repr__(self):
		return 'MALA'

	@torch.enable_grad()
	def propose(self):

		self.optim.zero_grad()
		batch = next(self.probmodel.dataloader.__iter__())
		log_prob = self.probmodel.log_prob(*batch)
		(-log_prob['log_prob']).backward()
		self.optim.step()

		return log_prob, self.probmodel

class HMC_Chain(Sampler_Chain):

	def __init__(self, probmodel, step_size=0.0001, num_steps=2000, burn_in=100, pretrain=False, tune=False,
		     traj_length=20):

		# assert probmodel.log_prob().keys()[:3] == ['log_prob', 'data', ]

		Sampler_Chain.__init__(self, probmodel, step_size, num_steps, burn_in, pretrain, tune)

		self.traj_length = traj_length

		self.optim = HMC_Optim(self.probmodel,
					step_size=step_size,
					prior_std=1.)

		# self.acceptance = SDE_Acceptance()
		self.acceptance = MetropolisHastingsAcceptance()

	def __repr__(self):
		return 'HMC'

	def sample_chain(self):

		self.probmodel.reset_parameters()

		if self.pretrain:
			try:
				self.probmodel.pretrain()
			except:
				warnings.warn(f'Tried pretraining but couldnt find a probmodel.pretrain() method ... Continuing wihtout pretraining.')

		if self.tune: self.tune_step_size()

		self.chain = Chain(probmodel=self.probmodel)

		progress = tqdm(range(self.num_steps))
		for step in progress:

			_ = self.propose() # values are added directly to self.chain

			desc = f'{str(self)}: Accept: {self.chain.running_accepts.avg:.2f}/{self.chain.accept_ratio:.2f} \t'
			for key, running_avg in self.chain.running_avgs.items():
				desc += f' {key}: {running_avg.avg:.2f} '
			desc += f'StepSize: {self.optim.param_groups[0]["step_size"]:.3f}'
			progress.set_description(desc=desc)

		self.chain = self.chain[self.burn_in:]

		return self.chain

	def propose(self):
		'''
		1) sample momentum for each parameter
		2) sample one minibatch for an entire trajectory
		3) solve trajectory forward for self.traj_length steps
		'''

		hamiltonian_solver = ['euler', 'leapfrog'][0]

		self.optim.sample_momentum()
		batch = next(self.probmodel.dataloader.__iter__()) # samples one minibatch from dataloader

		def closure():
			'''
			Computes the gradients once for batch
			'''
			self.optim.zero_grad()
			log_prob = self.probmodel.log_prob(*batch)
			(-log_prob['log_prob']).backward()
			return log_prob

		if hamiltonian_solver=='leapfrog': log_prob = closure() # compute initial grads

		for traj_step in range(self.traj_length):
			if hamiltonian_solver=='euler':
				proposal_log_prob = closure()
				self.optim.step()
			elif hamiltonian_solver=='leapfrog':
				proposal_log_prob = self.optim.leapfrog_step(closure)

		accept, log_ratio = self.acceptance(proposal_log_prob['log_prob'], self.chain.state['log_prob']['log_prob'])

		if not accept:
			if torch.isnan(proposal_log_prob['log_prob']):
				print(f"{proposal_log_prob=}")
				print(self.chain.state)
				exit()
			self.probmodel.load_state_dict(self.chain.state['state_dict'])

		self.chain += (self.probmodel, proposal_log_prob, accept)

class SGNHT_Chain(Sampler_Chain):

	def __init__(self, probmodel, step_size=0.0001, num_steps=2000, burn_in=100, pretrain=False, tune=False,
		     traj_length=20):

		# assert probmodel.log_prob().keys()[:3] == ['log_prob', 'data', ]

		Sampler_Chain.__init__(self, probmodel, step_size, num_steps, burn_in, pretrain, tune)

		self.traj_length = traj_length

		self.optim = SGNHT_Optim(self.probmodel,
					step_size=step_size,
					prior_std=1.)

		# print(f"{self.optim.A=}")
		# print(f"{self.optim.num_params=}")
		# print(f"{self.optim.A=}")
		# exit()

		# self.acceptance = SDE_Acceptance()
		self.acceptance = MetropolisHastingsAcceptance()

	def __repr__(self):
		return 'SGNHT'

	def sample_chain(self):

		self.probmodel.reset_parameters()

		if self.pretrain:
			try:
				self.probmodel.pretrain()
			except:
				warnings.warn(f'Tried pretraining but couldnt find a probmodel.pretrain() method ... Continuing wihtout pretraining.')

		if self.tune: self.tune_step_size()

		self.chain = Chain(probmodel=self.probmodel)
		self.optim.sample_momentum()
		self.optim.sample_thermostat()

		progress = tqdm(range(self.num_steps))
		for step in progress:

			proposal_log_prob, sample = self.propose()
			accept, log_ratio = self.acceptance(proposal_log_prob['log_prob'], self.chain.state['log_prob']['log_prob'])
			self.chain += (self.probmodel, proposal_log_prob, accept)

			desc = f'{str(self)}: Accept: {self.chain.running_accepts.avg:.2f}/{self.chain.accept_ratio:.2f} \t'
			for key, running_avg in self.chain.running_avgs.items():
				desc += f' {key}: {running_avg.avg:.2f} '
			desc += f'StepSize: {self.optim.param_groups[0]["step_size"]:.3f}'
			progress.set_description(desc=desc)

		self.chain = self.chain[self.burn_in:]

		return self.chain

	def propose(self):
		'''
		1) sample momentum for each parameter
		2) sample one minibatch for an entire trajectory
		3) solve trajectory forward for self.traj_length steps
		'''

		hamiltonian_solver = ['euler', 'leapfrog'][0]

		# self.optim.sample_momentum()
		# self.optim.sample_thermostat()
		batch = next(self.probmodel.dataloader.__iter__()) # samples one minibatch from dataloader

		self.optim.zero_grad()
		proposal_log_prob = self.probmodel.log_prob(*batch)
		(-proposal_log_prob['log_prob']).backward()
		self.optim.step()

		# def closure():
		# 	'''
		# 	Computes the gradients once for batch
		# 	'''
		# 	self.optim.zero_grad()
		# 	log_prob = self.probmodel.log_prob(*batch)
		# 	(-log_prob['log_prob']).backward()
		# 	return log_prob
		#
		# if hamiltonian_solver=='leapfrog': log_prob = closure() # compute initial grads
		#
		# for traj_step in range(self.traj_length):
		# 	if hamiltonian_solver=='euler':
		# 		proposal_log_prob = closure()
		# 		self.optim.step()
		# 	elif hamiltonian_solver=='leapfrog':
		# 		proposal_log_prob = self.optim.leapfrog_step(closure)
		#
		# accept, log_ratio = self.acceptance(proposal_log_prob['log_prob'], self.chain.state['log_prob']['log_prob'])
		#
		# if not accept:
		# 	if torch.isnan(proposal_log_prob['log_prob']):
		# 		print(f"{proposal_log_prob=}")
		# 		print(self.chain.state)
		# 		exit()
		# 	self.probmodel.load_state_dict(self.chain.state['state_dict'])

		return proposal_log_prob, self.probmodel

