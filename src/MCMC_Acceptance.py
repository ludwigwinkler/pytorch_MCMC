import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MetropolisHastingsAcceptance():

	def __init__(self):

		pass

	def __call__(self, log_prob_proposal, log_prob_state):


		if not torch.isnan(log_prob_proposal) or not torch.isinf(log_prob_proposal):
			log_ratio = (log_prob_proposal - log_prob_state)
			log_ratio = torch.min(log_ratio, torch.zeros_like(log_ratio))
			log_u = torch.zeros_like(log_ratio).uniform_(0,1).log()
			accept = torch.gt(log_ratio, log_u)
			accept = accept.bool().item()

			return accept, log_ratio

		elif torch.isnan(log_prob_proposal) or torch.isinf(log_prob_proposal):
			exit(f'log_prob_proposal is nan or inf {log_prob_proposal}')
			return False, torch.Tensor([-1])

class SDE_Acceptance():

	def __init__(self):

		pass

	def __call__(self, log_prob_proposal, log_prob_state):

		return True, torch.Tensor([0.])