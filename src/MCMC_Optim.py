import copy
from tqdm import tqdm
from collections import MutableSequence
import matplotlib
import matplotlib.pyplot as plt

import numpy as np

import torch
from torch.optim import Optimizer, SGD
from torch.distributions.distribution import Distribution
from torch.distributions import MultivariateNormal, Normal
from torch.nn import Module

class MCMC_Optim:

	def __init__(self):

		self.tune_params = {'delta': 0.65,
				    't0': 10,
				    'gamma': .05,
				    'kappa': .75,
				    # 'mu': np.log(self.param_groups[0]["step_size"]),
				    'mu': 0.,
				    'H': 0,
				    'log_eps': 1.}

		# print(f"@MCMC_Optim {self.tune_params=}")
		# exit()

	def tune(self, accepts):

		'''
		PyMC:
		# Switch statement
		    if acceptance < 0.001: 	0.1
		    elif acceptance < 0.05: 	0.5
		    elif acceptance < 0.2:	0.9
		    elif acceptance > 0.95:	10
		    elif acceptance > 0.75:	2
		    elif acceptance > 0.5:	1.1
		'''

		avg_acc = sum(accepts)/len(accepts)

		'''
		Switch statement: the first condition that is met exits the switch statement
		'''
		if avg_acc < 0.001:
			scale = 0.1
		elif avg_acc < 0.05:
			scale = 0.5
		elif avg_acc < 0.20:
			# PyMC: 0.9
			scale = 0.5

		elif avg_acc > 0.99:
			scale = 10.
		elif avg_acc > 0.75:
			scale = 2.
		elif avg_acc > 0.5:
			# PyMC: 1.1
			scale = 1.1
		else:
			scale = 0.9

		for group in self.param_groups:

			group['step_size']*=scale
			# print(f'{avg_acc=:.3f} & {scale=} -> {group["lr"]=:.3f}')

	def dual_average_tune(self, accepts, t, alpha):
		'''
		NUTS Sampler p.17 : Algorithm 5

		mu = log(10 * initial_step_size)

		H_m : running difference between target acceptance rate and current acceptance rate
			delta : target acceptance rate
			alpha : (singular) current acceptance rate

		log_eps = mu - t**0.5 / gamma H_m
		running_log_eps = t**(-kappa) log_eps + (1 - t**(-kappa)) running_log_eps
		'''

		# accept_ratio = sum(accepts)/len(accepts)
		assert 0<alpha<=1., f'{alpha=}'
		# print(f"{alpha=}")
		# print(t)

		delta, t0, gamma, kappa, mu, H, log_eps = self.tune_params.values()
		# t = len(chain)
		# alpha = sum(accepts)/len(accepts)
		# print(f'{self.param_groups[0]["step_size"]} {alpha}')



		H = (1 - 1 / (t + t0)) * H + 1 / (t + t0) * (delta - alpha)
		# H = 0.5 * H + 0.5 * (delta - alpha)

		log_eps_t = mu - t**0.5/gamma * H
		# log_eps_t = mu - 1 * (delta - alpha)

		log_eps = t**(-kappa) * log_eps_t + (1 - t**(-kappa)) * log_eps
		# log_eps = 0.5 * log_eps_t + 0.5 * log_eps

		self.tune_params["H"] = H
		self.tune_params["log_eps"] = log_eps
		# print(f"{log_eps=} {self.tune_params['log_eps']=}")
		# exit()

		for group in self.param_groups:
			group["step_size"] = np.exp(log_eps)


class MetropolisHastings_Optim(Optimizer, MCMC_Optim):

	def __init__(self, model, step_length):

		if step_length < 0.0:
			raise ValueError("Invalid learning rate: {}".format(step_length))

		defaults = dict(lr=step_length)

		params = model.parameters()
		self.model = model

		Optimizer.__init__(self, params=params, defaults=defaults)
		MCMC_Optim.__init__(self)

	def step(self):

		log_prob = None

		for group in self.param_groups:

			for p in group['params']:
				p.data.add_(other=torch.randn_like(p), alpha=group['lr'], )

		return log_prob


class SGLD_Optim(Optimizer, MCMC_Optim):

	def __init__(self, model, step_size=0.1, prior_std=1., addnoise=True):
		'''
		log_N(θ|0,1) =
		:param model:
		:param step_size:
		:param norm_sigma:
		:param addnoise:
		'''

		weight_decay = 1 / (prior_std ** 2) if prior_std!=0 else 0
		if weight_decay < 0.0:
			raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
		if step_size < 0.0:
			raise ValueError("Invalid learning rate: {}".format(step_size))

		defaults = dict(step_size=step_size, weight_decay=weight_decay, addnoise=addnoise)

		self.model = model
		params = self.model.parameters()

		Optimizer.__init__(self, params=params, defaults=defaults)
		MCMC_Optim.__init__(self)

	def step(self):

		'''
		d theta
		= eps/2 nabla log_prob + N(0, eps)
		= eps/2 nabla log_prob + N(0, 1) * eps**0.5 # x^0.5 = x^(1-0.5) = x/x^0.5
		= eps/2 nabla log_prob + N(0, 1) * eps / eps**0.5
		= eps/2 nabla log_prob + N(0, 1) * eps / eps**0.5 * 2/2
		= eps/2 nabla log_prob + N(0, 1) * eps/2 * 2/eps**0.5
		= eps/2 ( nabla log_prob + 2/eps**0.5 * N(0, 1) )

		'''

		log_prob = None

		for group in self.param_groups:

			weight_decay = group['weight_decay']

			for p in group['params']:

				if p.grad is None:
					continue

				grad = p.grad.data
				# grad.clamp_(-1000, 1000)
				if weight_decay != 0:
					grad.add_(alpha=weight_decay, other=p.data)

				if group['addnoise']:

					noise = torch.randn_like(p.data).mul_(group['step_size'] ** 0.5)

					p.data.add_(grad, alpha=-0.5 * group['step_size'])
					p.data.add_(noise)

					if torch.isnan(p.data).any(): exit('Nan param')
					if torch.isinf(p.data).any(): exit('inf param')

				else:
					p.data.add_(other=0.5*grad, alpha=-group['step_size'],)

		return log_prob

class MALA_Optim(Optimizer, MCMC_Optim):

	def __init__(self, model, step_size=0.1, prior_std=1., addnoise=True):
		'''
		log_N(θ|0,1) =
		:param model:
		:param step_size:
		:param norm_sigma:
		:param addnoise:
		'''

		weight_decay = 1 / (prior_std ** 2) if prior_std!=0 else 0
		if weight_decay < 0.0:
			raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
		if step_size < 0.0:
			raise ValueError("Invalid learning rate: {}".format(step_size))

		defaults = dict(step_size=step_size, weight_decay=weight_decay, addnoise=addnoise)

		self.model = model
		params = self.model.parameters()

		Optimizer.__init__(self, params=params, defaults=defaults)
		MCMC_Optim.__init__(self)

		# print(self.tune_params)
		# exit()

	def step(self):

		'''
		d theta
		= eps/2 nabla log_prob + N(0, eps)
		= eps/2 nabla log_prob + N(0, 1) * eps**0.5 # x^0.5 = x^(1-0.5) = x/x^0.5
		= eps/2 nabla log_prob + N(0, 1) * eps / eps**0.5
		= eps/2 nabla log_prob + N(0, 1) * eps / eps**0.5 * 2/2
		= eps/2 nabla log_prob + N(0, 1) * eps/2 * 2/eps**0.5
		= eps/2 ( nabla log_prob + 2/eps**0.5 * N(0, 1) )

		'''

		log_prob = None

		for group in self.param_groups:

			weight_decay = group['weight_decay']

			for p in group['params']:

				if p.grad is None:
					continue

				grad = p.grad.data
				# grad.clamp_(-1000,1000)
				if weight_decay != 0:
					grad.add_(alpha=weight_decay, other=p.data)

				if group['addnoise']:

					noise = torch.randn_like(p.data).mul_(group['step_size']**0.5)#.mul_(0.1)

					p.data.add_(grad, alpha=-0.5*group['step_size'])
					p.data.add_(noise)

					if torch.isnan(p.data).any():
						print(grad)
						exit('Nan param')
					if torch.isinf(p.data).any(): exit('inf param')

				else:
					p.data.add_(other=0.5*grad, alpha=-group['step_size'])

		return log_prob


class HMC_Optim(Optimizer, MCMC_Optim):

	def __init__(self, model, step_size=0.1, prior_std=1.):
		'''
		log_N(θ|0,1) =
		:param model:
		:param step_size:
		:param norm_sigma:
		:param addnoise:
		'''

		weight_decay = 1 / (prior_std ** 2) if prior_std != 0 else 0
		if weight_decay < 0.0:
			raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
		if step_size < 0.0:
			raise ValueError("Invalid learning rate: {}".format(step_size))

		defaults = dict(step_size=step_size,
				weight_decay=weight_decay,
				traj_step=0)

		self.model = model
		params = self.model.parameters()

		Optimizer.__init__(self, params=params, defaults=defaults)
		MCMC_Optim.__init__(self)

	def step(self):

		for group in self.param_groups:

			for p in group['params']:

				grad = p.grad.data
				state = self.state[p] # contains state['velocity']

				state['velocity'].add_(other=-group['step_size'] * grad)
				p.data.add_(other=state['velocity'], alpha=group['step_size'])

			group['traj_step'] += 1


	def sample_momentum(self):

		for group in self.param_groups:

			group['traj_step'] = 0

			for p in group['params']:
				# print(p)
				state = self.state[p]
				state['velocity'] = 1.*torch.randn_like(p)
				# print(self.state)
				# state['velocity'] = abs(torch.randn_like(p))
				# state['velocity'] = abs(1.*torch.randn(1)*torch.ones_like(p))
				# state['velocity'] = abs(1.*torch.ones_like(p))
				# state['velocity'] = abs(1.*torch.zeros_like(p))


	def leapfrog_step(self, closure):
		'''
		Leapfrog Integrator can be implemented with closure:
			1) Takes data and computes gradient
			2) moves halfway along the gradient
			3) recomputes the gradient after half step and does another half-step
			4) voila, we're at new sample

		TODO:: let log_prob return data and target as well because for closure we need the same data: https://pytorch.org/docs/stable/optim.html#taking-an-optimization-step
		TODO:: but that's a requirement of prob_model
		'''

		for group in self.param_groups:

			for p in group['params']:
				grad = p.grad.data
				# grad.clamp_(-1000,1000)
				state = self.state[p]  # contains state['velocity']

				state['velocity'].add_(other=-0.5*group['step_size'] * grad)
				p.data.add_(other=state['velocity'], alpha=group['step_size'])

		log_prob = closure()

		for group in self.param_groups:

			for p in group['params']:
				grad = p.grad.data
				# grad.clamp_(-1000, 1000)
				state = self.state[p]  # contains state['velocity']

				state['velocity'].add_(other=-0.5 * group['step_size'] * grad)
				# p.data.add_(other=state['velocity'], alpha=group['step_size'])


		return log_prob

class SGNHT_Optim(Optimizer, MCMC_Optim):

	def __init__(self, model, step_size=0.1, prior_std=1.):
		'''
		log_N(θ|0,1) =
		:param model:
		:param step_size:
		:param norm_sigma:
		:param addnoise:
		'''

		weight_decay = 1 / (prior_std ** 2) if prior_std != 0 else 0
		if weight_decay < 0.0:
			raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
		if step_size < 0.0:
			raise ValueError("Invalid learning rate: {}".format(step_size))

		self.num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
		self.A = 1.

		defaults = dict(step_size=step_size,
				weight_decay=weight_decay,
				traj_step=0,
				num_params=self.num_params,
				A=self.A)

		self.model = model
		params = self.model.parameters()

		Optimizer.__init__(self, params=params, defaults=defaults)
		MCMC_Optim.__init__(self)

	def step(self):

		for group in self.param_groups:

			step_size = group['step_size']

			# '''For velocity^T velocity / num_params over entire model'''
			# group['velocity_squared'] = 0.
			# for p in group['params']:
			# 	'''
			# 	compute 1/#params p^T p over entire model
			# 	'''
			#
			# 	state = self.state[p]
			# 	group['velocity_squared'] += torch.sum(state['velocity']**2)

			for p in group['params']:

				grad = p.grad.data
				state = self.state[p] # contains 'velocity' and 'thermostat'

				'''Update velocity'''
				state['velocity'].add_(other=-step_size * grad - step_size*state['thermostat']*state['velocity'])
				state['velocity'].add_(other=(2*group['A']*step_size)**0.5*torch.randn_like(p))

				'''Update Thermostat'''
				# state['thermostat'].add_(other=step_size*(group['velocity_squared']/group['num_params'] - 1))
				state['thermostat'].add_(other=step_size*(state['velocity']**2 - 1))

				'''Update parameter'''
				p.data.add_(other=state['velocity'], alpha=group['step_size'])

			group['traj_step'] += 1


	def sample_momentum(self):

		for group in self.param_groups:

			group['traj_step'] = 0

			for p in group['params']:
				state = self.state[p]
				state['velocity'] = 1.*torch.randn_like(p)

	def sample_thermostat(self):

		for group in self.param_groups:

			group['traj_step'] = 0

			for p in group['params']:
				state = self.state[p]
				state['thermostat'] = group['A']*torch.ones_like(p)
				# state['thermostat'] = group['A']*torch.ones_like(p).uniform_(0,1)
