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
from torch.utils.data import Dataset
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

class ProbModel(torch.nn.Module):

	'''
	ProbModel:

	'''

	def __init__(self, dataloader):
		super().__init__()
		assert isinstance(dataloader, torch.utils.data.DataLoader)
		self.dataloader = dataloader


	def log_prob(self):
		'''
		If minibatches have to be sampled due to memory constraints,
		a standard PyTorch dataloader can be used.
		"Infinite minibatch sampling" can be achieved by calling:
		data, target = next(dataloader.__iter__())
		next(Iterable.__iter__()) calls a single mini-batch sampling step
		But since it's not in a loop, we can call it add infinum
		'''
		raise NotImplementedError

	def sample_minibatch(self):
		'''
		Idea:
		Hybrid Monte Carlo Samplers require a constant tuple (data, target) to compute trajectories
		'''
		raise NotImplementedError

	def reset_parameters(self):
		raise NotImplementedError

	def predict(self, chain):
		raise NotImplementedError

	def pretrain(self):

		pass