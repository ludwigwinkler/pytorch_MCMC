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

import scipy
import scipy as sp
from scipy.io import loadmat as sp_loadmat
import copy

cwd = os.path.abspath(os.getcwd())
os.chdir(cwd)

params = argparse.ArgumentParser()
params.add_argument('-xyz', type=str, default='test_xyz')

params = params.parse_args()