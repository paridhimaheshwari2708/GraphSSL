import os
import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import TUDataset

DATA_SPLIT = [0.7, 0.2, 0.1]

def load_dataset(args):
	dataset = TUDataset(root='/tmp/PROTEINS', name='PROTEINS', use_node_attr=True)
	return dataset


def split_dataset(dataset):
	dataset.shuffle()

	n = len(dataset)
	train_split, val_split, test_split = DATA_SPLIT

	train_end = int(n * DATA_SPLIT[0])
	val_end = train_end + int(n * DATA_SPLIT[1])
	train_dataset, val_dataset, test_dataset = [i for i in dataset[:train_end]], [i for i in dataset[train_end:val_end]], [i for i in dataset[val_end:]]
	return train_dataset, val_dataset, test_dataset


def build_loader(args, dataset, subset):
	loader = DataLoader(custom_dataset(dataset, subset), num_workers=args.num_workers, batch_size=args.batch_size, follow_batch=['x_anchor', 'x_pos', 'x_neg'])
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
