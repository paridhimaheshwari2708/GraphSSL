import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class infonce(nn.Module):
	"""
	The InfoNCE (NT-XENT) loss in contrastive learning. The implementation
	follows the paper `A Simple Framework for Contrastive Learning of 
	Visual Representations <https://arxiv.org/abs/2002.05709>`.
	"""

	def __init__(self):
		super(infonce, self).__init__()

		self.tau = 0.5
		self.norm = True

	def forward(self, embed_anchor, embed_positive):
		"""
		Args:
			embed_anchor, embed_positive: Tensor of shape [batch_size, embed_dim]
			tau: Float. Usually in (0,1].
			norm: Boolean. Whether to apply normlization.
		"""

		batch_size = embed_anchor.shape[0]
		sim_matrix = torch.einsum("ik,jk->ij", embed_anchor, embed_positive)

		if self.norm:
			embed_anchor_abs = embed_anchor.norm(dim=1)
			embed_positive_abs = embed_positive.norm(dim=1)
			sim_matrix = sim_matrix / torch.einsum("i,j->ij", embed_anchor_abs, embed_positive_abs)

		sim_matrix = torch.exp(sim_matrix / self.tau)
		pos_sim = sim_matrix[range(batch_size), range(batch_size)]
		loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
		loss = - torch.log(loss).mean()
		return loss


class jensen_shannon(nn.Module):
	"""
	The Jensen-Shannon Estimator of Mutual Information used in contrastive learning. The
	implementation follows the paper `Learning deep representations by mutual information 
	estimation and maximization <https://arxiv.org/abs/1808.06670>`.

	Note: The JSE loss implementation can produce negative values because a :obj:`-2log2` shift is 
		added to the computation of JSE, for the sake of consistency with other f-convergence losses.
	"""

	def __init__(self):
		super(jensen_shannon, self).__init__()

	def get_expectation(self, masked_d_prime, positive=True):
		"""
		Args:
			masked_d_prime: Tensor of shape [n_graphs, n_graphs] for global_global,
							tensor of shape [n_nodes, n_graphs] for local_global.
			positive (bool): Set True if the d_prime is masked for positive pairs,
							set False for negative pairs.
		"""

		log_2 = np.log(2.)
		if positive:
			score = log_2 - F.softplus(-masked_d_prime)
		else:
			score = F.softplus(-masked_d_prime) + masked_d_prime - log_2
		return score

	def forward(self, embed_anchor, embed_positive):
		"""
		Args:
			embed_anchor, embed_positive: Tensor of shape [batch_size, embed_dim].
		"""

		device = embed_anchor.device
		batch_size = embed_anchor.shape[0]

		pos_mask = torch.zeros((batch_size, batch_size)).to(device)
		neg_mask = torch.ones((batch_size, batch_size)).to(device)
		for graphidx in range(batch_size):
			pos_mask[graphidx][graphidx] = 1.
			neg_mask[graphidx][graphidx] = 0.

		d_prime = torch.matmul(embed_anchor, embed_positive.t())

		E_pos = self.get_expectation(d_prime * pos_mask, positive=True).sum()
		E_pos = E_pos / batch_size
		E_neg = self.get_expectation(d_prime * neg_mask, positive=False).sum()
		E_neg = E_neg / (batch_size * (batch_size - 1))
		return E_neg - E_pos
