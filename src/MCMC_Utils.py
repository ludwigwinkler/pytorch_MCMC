import numpy as np
import torch
import matplotlib.pyplot as plt

def posterior_dist(chain, param=None, verbose=False, plot=True):

	if len(chain.samples[0]) == 1:
		'''
		We're sampling from a predefined distribution like a GMM and simulating a particle
		'''
		post = []

		# print(list(self.probmodel.state_dict().values())[0])
		# exit()

		# accepted_models = [chain.samples[idx] for idx in chain.accepted_steps]
		for model_state_dict in chain.samples:
			post.append(list(model_state_dict.values())[0])

		post = torch.cat(post, dim=0)

		if plot:
			hist2d = plt.hist2d(x=post[:, 0].cpu().numpy(), y=post[:, 1].cpu().numpy(), bins=100, range=np.array([[-3, 3], [-3, 3]]),
					    density=True)
			plt.colorbar(hist2d[3])
			plt.show()

	elif len(chain.samples[0]) > 1:
		'''
		There is more than one parameter in the model
		'''

		param_names = list(chain.samples[0].keys())
		accepted_models = [chain.samples[idx] for idx in chain.accepted_idxs]

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