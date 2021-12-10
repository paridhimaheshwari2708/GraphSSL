import os
import torch
import random
import torch.nn.functional as F
from torch_geometric.utils import degree
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data, Batch, Dataset

from view_functions import *

DATA_SPLIT = [0.7, 0.2, 0.1] # Train / val / test split ratio


def get_max_deg(dataset):
	"""
	Find the max degree across all nodes in all graphs.
	"""
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
		in_degree (bool, optional): If set to :obj:`True`, will compute the in-
			degree of nodes instead of the out-degree. (default: :obj:`False`)
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
	"""
	Load a specific TUDataset and optionally expand the set of
	node features by adding node degrees as one hot encodings.
	Args:
		name (str): name of TUDataset to load
		expand_features (bool, optional): If set to :obj:`True`, will augment
			the node features using their degrees. (default: :obj:`True`)
	"""

	if name == "proteins":
		dataset = TUDataset(root="/tmp/TUDataset/PROTEINS", name="PROTEINS", use_node_attr=True)
	elif name == "enzymes":
		dataset = TUDataset(root="/tmp/TUDataset/ENZYMES", name="ENZYMES", use_node_attr=True)
	elif name == "collab":
		dataset = TUDataset(root="/tmp/TUDataset/COLLAB", name="COLLAB", use_node_attr=True)
	elif name == "reddit_binary":
		dataset = TUDataset(root="/tmp/TUDataset/REDDIT-BINARY", name="REDDIT-BINARY", use_node_attr=True)
	elif name == "reddit_multi":
		dataset = TUDataset(root="/tmp/TUDataset/REDDIT-MULTI-5K", name="REDDIT-MULTI-5K", use_node_attr=True)
	elif name == "imdb_binary":
		dataset = TUDataset(root="/tmp/TUDataset/IMDB-BINARY", name="IMDB-BINARY", use_node_attr=True)
	elif name == "imdb_multi":
		dataset = TUDataset(root="/tmp/TUDataset/IMDB-MULTI", name="IMDB-MULTI", use_node_attr=True)
	elif name == "dd":
		dataset = TUDataset(root="/tmp/TUDataset/DD", name="DD", use_node_attr=True)
	elif name == "mutag":
		dataset = TUDataset(root="/tmp/TUDataset/MUTAG", name="MUTAG", use_node_attr=True)
	elif name == "nci1":
		dataset = TUDataset(root="/tmp/TUDataset/NCI1", name="NCI1", use_node_attr=True)

	num_classes = dataset.num_classes
	if dataset[0].x is None or expand_features:
		max_degree = get_max_deg(dataset)
		transform = CatDegOnehot(max_degree)
		dataset = [transform(graph) for graph in dataset]
	else:
		dataset = [graph for graph in dataset]
	feat_dim = dataset[0].num_node_features

	return dataset, feat_dim, num_classes


def split_dataset(dataset, train_data_percent=1.0):
	"""
	Splits the data into train / val / test sets.
	Args:
		dataset (list): all graphs in the dataset.
		train_data_percent (float): Fraction of training data
			which is labelled. (default 1.0)
	"""

	random.shuffle(dataset)

	n = len(dataset)
	train_split, val_split, test_split = DATA_SPLIT

	train_end = int(n * DATA_SPLIT[0])
	val_end = train_end + int(n * DATA_SPLIT[1])
	train_label_percent = int(train_end * train_data_percent)
	train_dataset, val_dataset, test_dataset = [i for i in dataset[:train_label_percent]], [i for i in dataset[train_end:val_end]], [i for i in dataset[val_end:]]
	return train_dataset, val_dataset, test_dataset


def build_loader(args, dataset, subset):
	shuffle = (subset != "test")
	loader = DataLoader(MyDataset(dataset, subset, args.augment_list),
						num_workers=args.num_workers, batch_size=args.batch_size, 
						shuffle=shuffle, follow_batch=["x_anchor", "x_pos"])
	return loader


def build_classification_loader(args, dataset, subset):
	shuffle = (subset != "test")
	loader = DataLoader(dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=shuffle)
	return loader


class MyDataset(Dataset):
	"""
	Dataset class that returns a graph and its augmented view in get() call.
	Augmentations are applied sequentially based on the augment_list.
	"""

	def __init__(self, dataset, subset, augment_list):
		super(MyDataset, self).__init__()

		self.dataset = dataset
		self.augment_list = augment_list

		self.augment_functions = []
		for augment in self.augment_list:
			if augment == "edge_perturbation":
				function = EdgePerturbation()
			elif augment == "diffusion":
				function = Diffusion()
			elif augment == "diffusion_with_sample":
				function = DiffusionWithSample()
			elif augment == "node_dropping":
				function = UniformSample()
			elif augment == "random_walk_subgraph":
				function = RWSample()
			elif augment == "node_attr_mask":
				function = NodeAttrMask()
			self.augment_functions.append(function)

		print("# samples in {} subset: {}".format(subset, len(self.dataset)))

	def get_positive_sample(self, current_graph):
		"""
		Possible augmentations include the following:
			edge_perturbation()
			diffusion()
			diffusion_with_sample()
			node_dropping()
			random_walk_subgraph()
			node_attr_mask()
		"""

		graph_temp = current_graph
		for function in self.augment_functions:
			graph_temp = function.views_fn(graph_temp)
		return graph_temp

	def get(self, idx):
		graph_anchor = self.dataset[idx]
		graph_pos = self.get_positive_sample(graph_anchor)
		return PairData(graph_anchor.edge_index, graph_anchor.x, graph_pos.edge_index, graph_pos.x)

	def len(self):
		return len(self.dataset)


class PairData(Data):
	"""
	Utility function to return a pair of graphs in dataloader.
	Adapted from https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
	"""

	def __init__(self, edge_index_anchor = None, x_anchor = None, edge_index_pos = None, x_pos = None):
		super().__init__()
		self.edge_index_anchor = edge_index_anchor
		self.x_anchor = x_anchor
		
		self.edge_index_pos = edge_index_pos
		self.x_pos = x_pos

	def __inc__(self, key, value, *args, **kwargs):
		if key == "edge_index_anchor":
			return self.x_anchor.size(0)
		if key == "edge_index_pos":
			return self.x_pos.size(0)
		else:
			return super().__inc__(key, value, *args, **kwargs)
