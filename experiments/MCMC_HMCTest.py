# cleaner interaction


import os, argparse

# print(os.path.dirname(sys.executable))
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["figure.figsize"] = [10, 10]

import torch

SEED = 0
# torch.manual_seed(SEED)
# np.random.seed(SEED)

from pytorch_MCMC.models.MCMC_Models import GMM, LinReg, RegressionNNHomo, RegressionNNHetero
from pytorch_MCMC.src.MCMC_Sampler import SGLD_Sampler, MALA_Sampler, HMC_Sampler, SGNHT_Sampler
from pytorch_MCMC.data.MCMC_SyntheticData import generate_linear_regression_data, generate_nonstationary_data
from Utils.Utils import str2bool

params = argparse.ArgumentParser(description='parser example')
params.add_argument('-logname', 	type=str, default='Tmp')

params.add_argument('-num_samples', 	type=int, default=200)
params.add_argument('-model',	 	choices=['gmm', 'linreg', 'regnn'], default='gmm')
params.add_argument('-sampler', 	choices=['sgld','mala', 'hmc', 'sgnht'], default='sgnht')

params.add_argument('-step_size', 	type=float, default=0.1)
params.add_argument('-num_steps', 	type=int, default=10000)
params.add_argument('-pretrain', 	type=str2bool, default=False)
params.add_argument('-tune', 		type=str2bool, default=False)
params.add_argument('-burn_in', 	type=int, default=2000)
# params.add_argument('-num_chains', 		type=int, 	default=1)
params.add_argument('-num_chains', 	type=int, default=os.cpu_count() - 1)
params.add_argument('-batch_size', 	type=int, default=50)

params.add_argument('-hmc_traj_length', type=int, default=20)

params.add_argument('-val_split', 	type=float, default=0.9)  # first part is train, second is val i.e. val_split=0.8 -> 80% train, 20% val

params.add_argument('-val_prediction_steps', 	type=int, default=50)
params.add_argument('-val_converge_criterion', 	type=int, default=20)
params.add_argument('-val_per_epoch', 		type=int, default=200)

params = params.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	FloatTensor = torch.cuda.FloatTensor
	Tensor = torch.cuda.FloatTensorf
else:
	device = torch.device('cpu')
	FloatTensor = torch.FloatTensor
	Tensor = torch.FloatTensor

if params.model == 'gmm':
	gmm = GMM()
	gmm.generate_surface(plot=True)

	if params.sampler=='sgld':
		sampler = SGLD_Sampler(probmodel=gmm,
				       step_size=params.step_size,
				       num_steps=params.num_steps,
				       burn_in=params.burn_in,
				       pretrain=params.pretrain,
				       tune=params.tune,
				       num_chains=params.num_chains)
	elif params.sampler=='mala':
		sampler = MALA_Sampler(probmodel=gmm,
				       step_size=params.step_size,
				       num_steps=params.num_steps,
				       burn_in=params.burn_in,
				       pretrain=params.pretrain,
				       tune=params.tune,
				       num_chains=params.num_chains)
	elif params.sampler=='hmc':
		sampler = HMC_Sampler(probmodel=gmm,
				      step_size=params.step_size,
				      num_steps=params.num_steps,
				      burn_in=params.burn_in,
				      pretrain=params.pretrain,
				      tune=params.tune,
				      traj_length=params.hmc_traj_length,
				      num_chains=params.num_chains)
	elif params.sampler == 'sgnht':
		sampler = SGNHT_Sampler(probmodel=gmm,
					step_size=params.step_size,
					num_steps=params.num_steps,
					burn_in=params.burn_in,
					pretrain=params.pretrain,
					tune=params.tune,
					traj_length=params.hmc_traj_length,
					num_chains=params.num_chains)
	else:
		raise ValueError('No Sampler defined')

	sampler.sample_chains()
	sampler.posterior_dist()
	# sampler.trace()

	# plt.plot(sampler.chain.accepted_steps)
	plt.show()

elif params.model == 'linreg':

	x, y = generate_linear_regression_data(num_samples=params.num_samples, m=-2., b=-1, y_noise=0.5)
	linreg = LinReg(x, y)
	# sampler = MetropolisHastings_Sampler(probmodel=linreg, step_size=params.step_size, num_steps=params.num_steps, burn_in=params.burn_in, tune=params.tune)
	sampler = SGLD_Sampler(probmodel=linreg, step_size=params.step_size, num_steps=params.num_steps, burn_in=params.burn_in,
			       pretrain=params.pretrain, tune=params.tune)
	sampler.sample_chains()
	sampler.posterior_dist()
	linreg.predict(sampler.chain)

elif params.model == 'regnn':

	model = ['homo', 'hetero'][1]
	if model=='homo':
		x, y = generate_nonstationary_data(num_samples=params.num_samples, plot=False, y_constant_noise_std=0.25, y_nonstationary_noise_std=0.01)
		nn = RegressionNNHomo(x, y, batch_size=params.batch_size)
	elif model=='hetero':
		x, y = generate_nonstationary_data(num_samples=params.num_samples, plot=False, y_constant_noise_std=0.001, y_nonstationary_noise_std=0.5)
		nn = RegressionNNHetero(x, y, batch_size=params.batch_size)

	if params.sampler == 'sgld':
		sampler = SGLD_Sampler(probmodel=nn,
				       step_size=params.step_size,
				       num_steps=params.num_steps,
				       burn_in=params.burn_in,
				       pretrain=params.pretrain,
				       tune=params.tune,
				       num_chains=params.num_chains)
	elif params.sampler == 'mala':
		sampler = MALA_Sampler(probmodel=nn,
				       step_size=params.step_size,
				       num_steps=params.num_steps,
				       burn_in=params.burn_in,
				       pretrain=params.pretrain,
				       tune=params.tune,
				       num_chains=params.num_chains)
	elif params.sampler == 'hmc':
		sampler = HMC_Sampler(probmodel=nn,
				      step_size=params.step_size,
				      num_steps=params.num_steps,
				      burn_in=params.burn_in,
				      pretrain=params.pretrain,
				      tune=params.tune,
				      traj_length=params.hmc_traj_length,
				      num_chains=params.num_chains)
	elif params.sampler == 'sgnht':
		sampler = SGNHT_Sampler(probmodel=nn,
				      step_size=params.step_size,
				      num_steps=params.num_steps,
				      burn_in=params.burn_in,
				      pretrain=params.pretrain,
				      tune=params.tune,
				      traj_length=params.hmc_traj_length,
				      num_chains=params.num_chains)
	else:
		raise ValueError('No Sampler defined')
	chains = sampler.sample_chains()

	nn.predict(chains, plot=True)
