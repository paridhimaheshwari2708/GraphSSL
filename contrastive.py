import torch
import torch.nn as nn

class infonce():
	def __init__(self):
		self.sigmoid = torch.nn.Sigmoid()
		self.cos = torch.nn.CosineSimilarity()
		self.crossEntropy = torch.nn.BCELoss()

	def __call__(self, embed, embed_pos, embed_neg):
		dtype, device = embed.dtype, embed.device
		posScores = self.cos(embed, embed_pos)
		negScores = self.cos(embed, embed_neg)
		# TODO: Add temperature
		distances = self.sigmoid( (posScores - negScores))
		loss = self.crossEntropy(distances, torch.ones(distances.size(0), device=device, dtype=dtype))
		return loss

class jensen_shannon():
	def __init__(self):
		self.sigmoid = torch.nn.Sigmoid()
		self.cos = torch.nn.CosineSimilarity()
		self.crossEntropy = torch.nn.BCELoss()

	def __call__(self, embed, embed_pos, embed_neg):
		return 0