import os
import torch
import argparse
import numpy as np
import torch.nn as nn
from tqdm import trange
from tensorboardX import SummaryWriter

from data import *
from model import *
# from contrastive import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

def set_seed(seed):
	"""Utility function to set seed values for RNG for various modules"""
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False


class Options:
	def __init__(self):
		self.parser = argparse.ArgumentParser(description="Classification Task on Graphs")
		self.parser.add_argument("--save", dest="save", action="store", required=True)
		self.parser.add_argument("--lr", dest="lr", action="store", default=0.001, type=float)
		self.parser.add_argument("--epochs", dest="epochs", action="store", default=20, type=int)
		self.parser.add_argument("--batch_size", dest="batch_size", action="store", default=64, type=int)
		self.parser.add_argument("--dataset", dest="dataset", action="store", required=True, type=str, \
			choices=["proteins", "enzymes", "reddit_binary", "reddit_multi", "imdb_binary", "imdb_multi", "dd", "mutag", "nci1"])
		self.parser.add_argument("--num_workers", dest="num_workers", action="store", default=8, type=int)
		self.parser.add_argument("--model", dest="model", action="store", default="gcn", type=str, choices=["gcn", "gin", "resgcn"])
		self.parser.add_argument("--task", dest="task", action="store", default="graph", type=str, choices=["graph", "node"])
		self.parser.add_argument("--ss_encoder_model", dest="ss_encoder_model", action="store", default="model", type=str)
		self.parser.add_argument("--ss_task", dest="ss_task", action="store_true", default=False)
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
	
	# criterion = nn.BCEWithLogitsLoss()
	criterion = torch.nn.CrossEntropyLoss()

	losses = []

	correct = 0
	with trange(len(dataloader), desc="{}, Epoch {}: ".format(mode, epoch)) as t:
		for data in dataloader:
			data.to(device)

			data_input = data.x, data.edge_index, data.batch

			out = model(data_input)
			labels = data.y
			# .view(-1,1).float()
			# print(out.shape, labels.shape)
			loss = criterion(out, labels)

			pred = out.argmax(dim=1)

			correct += int((pred == data.y).sum())
			# print(pred.shape, labels.shape)

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
	accuracy = correct / len(dataloader.dataset)
	return epoch_loss, accuracy


def main(args):
	dataset, feat_dim, num_classes = load_dataset(args.dataset)

	train_dataset, val_dataset, test_dataset = split_dataset(dataset)

	train_loader = build_classification_loader(args, train_dataset, "train")
	val_loader = build_classification_loader(args, val_dataset, "val")
	test_loader = build_classification_loader(args, test_dataset, "test")

	# print(train_dataset[0])
	# print(len(train_dataset))
	# print(num_classes)

	# for step, data in enumerate(train_loader):
	# 	print(f'Step {step + 1}:')
	# 	print('=======')
	# 	print(f'Number of graphs in the current batch: {data.num_graphs}')
	# 	print(data)
	# 	print()

	# output_dim = dataset[0].y.shape[0]
	# print(dataset[0].y.shape,dataset[0].y )

	model = PredictionModel(feat_dim, hidden_dim=128, n_layers=3, output_dim = num_classes, args= args)

	model = model.to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

	

	starting_epoch, best_train_loss, best_val_loss = -1, float("inf"), float("inf")

	logger = SummaryWriter(logdir = os.path.join("runs", args.save))

	for epoch in range(starting_epoch + 1, args.epochs):
		train_loss, train_acc = run(args, epoch, "train", train_loader, model, optimizer)
		print('Train Epoch Loss: {}, Train Epoch Accuracy: {}'.format(train_loss, train_acc))
		logger.add_scalar('Train Loss', train_loss, epoch)

		val_loss, val_acc = run(args, epoch, "val", val_loader, model, optimizer)
		print('Val Epoch Loss: {}, Val Epoch Accuracy: {}'.format(val_loss,val_acc))
		logger.add_scalar('Val Loss', val_loss, epoch)

		# Save Model
		is_best_loss = False
		if val_loss < best_val_loss:
			best_train_loss, best_val_loss, is_best_loss = train_loss, val_loss, True

		model.save_checkpoint(os.path.join("logs", args.save), optimizer, epoch, best_train_loss, best_val_loss, is_best_loss)

	################################################################################
	best_epoch, best_train_loss, best_val_loss = model.load_checkpoint(os.path.join("logs", args.save), optimizer)
	model.eval()

	print("Train Loss (best model): {:.3f}".format(best_train_loss))
	print("Val Loss (best model): {:.3f}".format(best_val_loss))

	test_loss, test_acc = run(args, best_epoch, "test", test_loader, model, optimizer)
	print("Test Loss: {:.3f},Test Accuracy: {:.3f}".format(test_loss, test_acc))

if __name__ == "__main__":

	set_seed(0)
	args = Options()
	print(args)

	main(args.opts)
