import os
import torch
import argparse
import numpy as np
import torch.nn as nn
from tqdm import trange
from tensorboardX import SummaryWriter

from data import *
from loss import *
from model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
	"""
	Utility function to set seed values for RNG for various modules
	"""
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False


class Options:
	def __init__(self):

		self.parser = argparse.ArgumentParser(description="Self-Supervised Learning for Graphs")
		self.parser.add_argument("--save", dest="save", action="store", required=True)
		self.parser.add_argument("--lr", dest="lr", action="store", default=0.001, type=float)
		self.parser.add_argument("--epochs", dest="epochs", action="store", default=20, type=int)
		self.parser.add_argument("--batch_size", dest="batch_size", action="store", default=64, type=int)
		self.parser.add_argument("--num_workers", dest="num_workers", action="store", default=8, type=int)
		self.parser.add_argument("--dataset", dest="dataset", action="store", required=True, type=str, \
			choices=["proteins", "enzymes", "reddit_binary", "reddit_multi", "imdb_binary", "imdb_multi", "dd", "mutag", "nci1"])
		self.parser.add_argument("--model", dest="model", action="store", default="gcn", type=str, choices=["gcn", "gin", "resgcn"])
		self.parser.add_argument("--loss", dest="loss", action="store", default="infonce", type=str, choices=["infonce", "jensen_shannon"])
		self.parser.add_argument("--augment_list", dest="augment_list", nargs="*", default=["edge_perturbation", "node_dropping"], type=str, choices=["edge_perturbation", "diffusion", "diffusion_with_sample", "node_dropping", "random_walk_subgraph", "node_attr_mask"])

		self.parse()
		self.check_args()

	def parse(self):
		self.opts = self.parser.parse_args()

	def check_args(self):
		if not os.path.isdir(os.path.join("runs", self.opts.save)):
			os.makedirs(os.path.join("runs", self.opts.save))
		if not os.path.isdir(os.path.join("logs", self.opts.save)):
			os.makedirs(os.path.join("logs", self.opts.save))

	def __str__(self):
		return ("All Options:\n" + "".join(["-"] * 45) + "\n" + "\n".join(["{:<18} -------> {}".format(k, v) for k, v in vars(self.opts).items()]) + "\n" + "".join(["-"] * 45) + "\n")


def run(args, epoch, mode, dataloader, model, optimizer):
	if mode == "train":
		model.train()
	elif mode == "val" or mode == "test":
		model.eval()
	else:
		assert False, "Wrong Mode:{} for Run".format(mode)

	losses = []
	contrastive_fn = eval(args.loss + "()")
	with trange(len(dataloader), desc="{}, Epoch {}: ".format(mode, epoch)) as t:
		for data in dataloader:
			data.to(device)

			readout_anchor = model((data.x_anchor, data.edge_index_anchor, data.x_anchor_batch))
			readout_positive = model((data.x_pos, data.edge_index_pos, data.x_pos_batch))

			loss = contrastive_fn(readout_anchor, readout_positive)

			if mode == "train":
				# Backprop
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			# Keep track of things
			losses.append(loss.item())
			t.set_postfix(loss=losses[-1])
			t.update()

	# Gather the results for the epoch
	epoch_loss = sum(losses) / len(losses)
	return epoch_loss


def main(args):
	dataset, feat_dim = load_dataset(args.dataset)
	train_dataset, val_dataset, test_dataset = split_dataset(dataset)

	train_loader = build_loader(args, train_dataset, "train", args.augment_list)
	val_loader = build_loader(args, val_dataset, "val", args.augment_list)
	test_loader = build_loader(args, test_dataset, "test", args.augment_list)

	model = Encoder(feat_dim, hidden_dim=128, n_layers=3, gnn=args.model).to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

	best_train_loss, best_val_loss = float("inf"), float("inf")

	logger = SummaryWriter(logdir = os.path.join("runs", args.save))

	for epoch in range(args.epochs):
		train_loss = run(args, epoch, "train", train_loader, model, optimizer)
		print('Train Epoch Loss: {}'.format(train_loss))
		logger.add_scalar('Train Loss', train_loss, epoch)

		val_loss = run(args, epoch, "val", val_loader, model, optimizer)
		print('Val Epoch Loss: {}'.format(val_loss))
		logger.add_scalar('Val Loss', val_loss, epoch)

		# Save Model
		is_best_loss = False
		if val_loss < best_val_loss:
			best_train_loss, best_val_loss, is_best_loss = train_loss, val_loss, True

		model.save_checkpoint(os.path.join("logs", args.save), optimizer, epoch, best_train_loss, best_val_loss, is_best_loss)

	################################################################################
	best_epoch, best_train_loss, best_val_loss = model.load_checkpoint(os.path.join("logs", args.save), optimizer)
	model.eval()

	print("Train Loss at epoch {} (best model): {:.3f}".format(best_epoch, best_train_loss))
	print("Val Loss at epoch {} (best model): {:.3f}".format(best_epoch, best_val_loss))

	test_loss = run(args, best_epoch, "test", test_loader, model, optimizer)
	print("Test Loss at epoch {}: {:.3f}".format(best_epoch, test_loss))

if __name__ == "__main__":

	set_seed(0)
	args = Options()
	print(args)

	main(args.opts)
