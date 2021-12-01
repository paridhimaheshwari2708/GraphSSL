import os
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree


DATA_SPLIT = [0.7, 0.2, 0.1]


def get_max_deg(dataset):
	max_deg = 0
	for data in dataset:
		row, col = data.edge_index
		num_nodes = data.num_nodes
		deg = degree(row, num_nodes)
		deg = max(deg).item()
		if deg > max_deg:
			max_deg = int(deg)
	return max_deg


class CatDegOnehot(object):
	"""
	Adds the node degree as one hot encodings to the node features.
	Args:
		max_degree (int): Maximum degree.
		in_degree (bool, optional): If set to :obj:`True`, will compute the
			in-degree of nodes instead of the out-degree.
			(default: :obj:`False`)
		cat (bool, optional): Concat node degrees to node features instead
			of replacing them. (default: :obj:`True`)
	"""

	def __init__(self, max_degree, in_degree=False, cat=True):
		self.max_degree = max_degree
		self.in_degree = in_degree
		self.cat = cat

	def __call__(self, data):
		idx, x = data.edge_index[1 if self.in_degree else 0], data.x
		deg = degree(idx, data.num_nodes, dtype=torch.long)
		deg = F.one_hot(deg, num_classes=self.max_degree + 1).to(torch.float)

		if x is not None and self.cat:
			x = x.view(-1, 1) if x.dim() == 1 else x
			data.x = torch.cat([x, deg.to(x.dtype)], dim=-1)
		else:
			data.x = deg
		return data


def load_dataset(name, expand_features=True):
	if name == "proteins":
		dataset = TUDataset(root='/tmp/PROTEINS', name='PROTEINS', use_node_attr=True)
	elif name == "enzymes":
		dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
	elif name == "collab":
		dataset = TUDataset(root='/tmp/COLLAB', name='COLLAB', use_node_attr=True)
	elif name == "reddit_binary":
		dataset = TUDataset(root='/tmp/REDDIT-BINARY', name='REDDIT-BINARY', use_node_attr=True)
	elif name == "reddit_multi":
		dataset = TUDataset(root='/tmp/REDDIT-MULTI-5K', name='REDDIT-MULTI-5K', use_node_attr=True)
	elif name == "imdb_binary":
		dataset = TUDataset(root='/tmp/IMDB-BINARY', name='IMDB-BINARY', use_node_attr=True)
	elif name == "imdb_multi":
		dataset = TUDataset(root='/tmp/IMDB-MULTI', name='IMDB-MULTI', use_node_attr=True)
	elif name == "dd":
		dataset = TUDataset(root='/tmp/DD', name='DD', use_node_attr=True)
	elif name == "mutag":
		dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG', use_node_attr=True)
	elif name == "nci1":
		dataset = TUDataset(root='/tmp/NCI1', name='NCI1', use_node_attr=True)

	if dataset[0].x is None or expand_features:
		max_degree = get_max_deg(dataset)
		transform = CatDegOnehot(max_degree)
		dataset = [transform(graph) for graph in dataset]
	else:
		dataset = [graph for graph in dataset]
	feat_dim = dataset[0].num_node_features

	return dataset, feat_dim


def split_dataset(dataset):
	random.shuffle(dataset)

	n = len(dataset)
	train_split, val_split, test_split = DATA_SPLIT

	train_end = int(n * DATA_SPLIT[0])
	val_end = train_end + int(n * DATA_SPLIT[1])
	train_dataset, val_dataset, test_dataset = [i for i in dataset[:train_end]], [i for i in dataset[train_end:val_end]], [i for i in dataset[val_end:]]
	return train_dataset, val_dataset, test_dataset


def build_loader(args, dataset, subset):
	loader = DataLoader(custom_dataset(dataset, subset), \
						num_workers=args.num_workers, batch_size=args.batch_size, \
						follow_batch=['x_anchor', 'x_pos', 'x_neg'])
	return loader


class custom_dataset(Dataset):
	def __init__(self, dataset, subset):
		super(custom_dataset, self).__init__()

		self.dataset = dataset
		print('# samples in {} subset: {}'.format(subset, len(self.dataset)))

	# TODO: Jian augmentation functions
	def get_positive_negative(self):
		pass

	def get(self, idx):
		graph_anchor = self.dataset[idx]
		# TODO: Jian augmentations functions
		graph_pos, graph_neg = self.dataset[0], self.dataset[1]
		return TripletData(graph_anchor.edge_index, graph_anchor.x, graph_pos.edge_index, graph_pos.x, graph_neg.edge_index, graph_neg.x)

	def len(self):
		return len(self.dataset)


class TripletData(Data):
	def __init__(self, edge_index_anchor = None, x_anchor = None, edge_index_pos = None, x_pos = None, edge_index_neg = None, x_neg = None):
		super().__init__()
		self.edge_index_anchor = edge_index_anchor
		self.x_anchor = x_anchor
		
		self.edge_index_pos = edge_index_pos
		self.x_pos = x_pos
		
		self.edge_index_neg = edge_index_neg
		self.x_neg = x_neg

	def __inc__(self, key, value, *args, **kwargs):
		if key == 'edge_index_anchor':
			return self.x_anchor.size(0)
		if key == 'edge_index_pos':
			return self.x_pos.size(0)
		if key == 'edge_index_neg':
			return self.x_neg.size(0)
		else:
			return super().__inc__(key, value, *args, **kwargs)
