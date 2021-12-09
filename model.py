import os
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch.nn import Parameter, Sequential, Linear, BatchNorm1d
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn import GCNConv, GINConv, GATConv, SAGEConv, SGConv, global_add_pool, global_mean_pool

"""
Part of the code has been adapted from https://github.com/divelab/DIG.
"""

class Encoder(torch.nn.Module):
	"""
	A wrapper class for easier instantiation of pre-implemented graph encoders.
	
	Args:
		feat_dim (int): The dimension of input node features.
		hidden_dim (int): The dimension of node-level (local) embeddings. 
		n_layers (int, optional): The number of GNN layers in the encoder. (default: :obj:`5`)
		pool (string, optional): The global pooling methods, :obj:`sum` or :obj:`mean`.
			(default: :obj:`sum`)
		gnn (string, optional): The type of GNN layer, :obj:`gcn` or :obj:`gin` or :obj:`gat`
			or :obj:`graphsage` or :obj:`resgcn` or :obj:`sgc`. (default: :obj:`gcn`)
		bn (bool, optional): Whether to include batch normalization. (default: :obj:`True`)
		node_level (bool, optional): If :obj:`True`, the encoder will output node level
			embedding (local representations). (default: :obj:`False`)
		graph_level (bool, optional): If :obj:`True`, the encoder will output graph level
			embeddings (global representations). (default: :obj:`True`)
		edge_weight (bool, optional): Only applied to GCN. Whether to use edge weight to
			compute the aggregation. (default: :obj:`False`)
			
	Note
	----
	For GCN and GIN encoders, the dimension of the output node-level (local) embedding will be 
	:obj:`hidden_dim`, whereas the node-level embedding will be :obj:`hidden_dim` * :obj:`n_layers`. 
	For ResGCN, the output embeddings for boths node and graphs will have dimensions :obj:`hidden_dim`.
			
	Examples
	--------
	>>> feat_dim = dataset[0].x.shape[1]
	>>> encoder = Encoder(feat_dim, 128, n_layers=3, gnn="gin")
	>>> encoder(some_batched_data).shape # graph-level embedding of shape [batch_size, 128*3]
	
	>>> encoder = Encoder(feat_dim, 128, n_layers=5, node_level=True, graph_level=False)
	>>> encoder(some_batched_data).shape # node-level embedding of shape [n_nodes, 128]
	
	>>> encoder = Encoder(feat_dim, 128, n_layers=5, node_level=True, graph_level=False)
	>>> encoder(some_batched_data) # a tuple of graph-level and node-level embeddings
	"""
	def __init__(self, feat_dim, hidden_dim, n_layers=5, pool="sum", 
				 gnn="gcn", bn=True, node_level=False, graph_level=True):
		super(Encoder, self).__init__()

		if gnn == "gcn":
			self.encoder = GCN(feat_dim, hidden_dim, n_layers, pool, bn)
		elif gnn == "gin":
			self.encoder = GIN(feat_dim, hidden_dim, n_layers, pool, bn)
		elif gnn == "resgcn":
			self.encoder = ResGCN(feat_dim, hidden_dim, n_layers, pool)
		elif gnn == "gat":
			self.encoder = GAT(feat_dim, hidden_dim, n_layers, pool, bn)
		elif gnn == "graphsage":
			self.encoder = GraphSAGE(feat_dim, hidden_dim, n_layers, pool, bn)
		elif gnn == "sgc":
			self.encoder = SGC(feat_dim, hidden_dim, n_layers, pool, bn)

		self.node_level = node_level
		self.graph_level = graph_level

	def forward(self, data):
		z_g, z_n = self.encoder(data)
		if self.node_level and self.graph_level:
			return z_g, z_n
		elif self.graph_level:
			return z_g
		else:
			return z_n

	def save_checkpoint(self, save_path, optimizer, epoch, best_train_loss, best_val_loss, is_best):
		ckpt = {}
		ckpt["state"] = self.state_dict()
		ckpt["epoch"] = epoch
		ckpt["optimizer_state"] = optimizer.state_dict()
		ckpt["best_train_loss"] = best_train_loss
		ckpt["best_val_loss"] = best_val_loss
		torch.save(ckpt, os.path.join(save_path, "model.ckpt"))
		if is_best:
			torch.save(ckpt, os.path.join(save_path, "best_model.ckpt"))

	def load_checkpoint(self, load_path, optimizer):
		ckpt = torch.load(os.path.join(load_path, "best_model.ckpt"))
		self.load_state_dict(ckpt["state"])
		epoch = ckpt["epoch"]
		best_train_loss = ckpt["best_train_loss"]
		best_val_loss = ckpt["best_val_loss"]
		optimizer.load_state_dict(ckpt["optimizer_state"])
		return epoch, best_train_loss, best_val_loss


class GCN(torch.nn.Module):
	def __init__(self, feat_dim, hidden_dim, n_layers=3, pool="sum", bn=False, xavier=True):
		super(GCN, self).__init__()

		if bn:
			self.bns = torch.nn.ModuleList()
		self.convs = torch.nn.ModuleList()
		self.acts = torch.nn.ModuleList()
		self.n_layers = n_layers
		self.pool = pool

		a = torch.nn.ReLU()

		for i in range(n_layers):
			start_dim = hidden_dim if i else feat_dim
			conv = GCNConv(start_dim, hidden_dim)
			if xavier:
				self.weights_init(conv)
			self.convs.append(conv)
			self.acts.append(a)
			if bn:
				self.bns.append(BatchNorm1d(hidden_dim))

	def weights_init(self, module):
		for m in module.modules():
			if isinstance(m, GCNConv):
				layer = m.lin
			if isinstance(m, Linear):
				layer = m
			torch.nn.init.xavier_uniform_(layer.weight.data)
			if layer.bias is not None:
				layer.bias.data.fill_(0.0)

	def forward(self, data):
		x, edge_index, batch = data
		xs = []
		for i in range(self.n_layers):
			x = self.convs[i](x, edge_index)
			x = self.acts[i](x)
			if self.bns is not None:
				x = self.bns[i](x)
			xs.append(x)

		if self.pool == "sum":
			xpool = [global_add_pool(x, batch) for x in xs]
		else:
			xpool = [global_mean_pool(x, batch) for x in xs]
		global_rep = torch.cat(xpool, 1)

		return global_rep, x


class GAT(torch.nn.Module):
	def __init__(self, feat_dim, hidden_dim, n_layers=3, pool="sum",
				 heads=1, bn=False, xavier=True):
		super(GAT, self).__init__()

		if bn:
			self.bns = torch.nn.ModuleList()
		self.convs = torch.nn.ModuleList()
		self.acts = torch.nn.ModuleList()
		self.n_layers = n_layers
		self.pool = pool

		a = torch.nn.ELU()

		for i in range(n_layers):
			start_dim = hidden_dim if i else feat_dim
			conv = GATConv(start_dim, hidden_dim, heads=heads, concat=False)
			if xavier:
				self.weights_init(conv)
			self.convs.append(conv)
			self.acts.append(a)
			if bn:
				self.bns.append(BatchNorm1d(hidden_dim))

	def weights_init(self, module):
		for m in module.modules():
			if isinstance(m, GATConv):
				layers = [m.lin_src, m.lin_dst]
			if isinstance(m, Linear):
				layers = [m]
			for layer in layers:
				torch.nn.init.xavier_uniform_(layer.weight.data)
				if layer.bias is not None:
					layer.bias.data.fill_(0.0)

	def forward(self, data):
		x, edge_index, batch = data
		xs = []
		for i in range(self.n_layers):
			x = self.convs[i](x, edge_index)
			x = self.acts[i](x)
			if self.bns is not None:
				x = self.bns[i](x)
			xs.append(x)

		if self.pool == "sum":
			xpool = [global_add_pool(x, batch) for x in xs]
		else:
			xpool = [global_mean_pool(x, batch) for x in xs]
		global_rep = torch.cat(xpool, 1)

		return global_rep, x


class GraphSAGE(torch.nn.Module):
	def __init__(self, feat_dim, hidden_dim, n_layers=3, pool="sum", bn=False, xavier=True):
		super(GraphSAGE, self).__init__()

		if bn:
			self.bns = torch.nn.ModuleList()
		self.convs = torch.nn.ModuleList()
		self.acts = torch.nn.ModuleList()
		self.n_layers = n_layers
		self.pool = pool

		a = torch.nn.ReLU()

		for i in range(n_layers):
			start_dim = hidden_dim if i else feat_dim
			conv = SAGEConv(start_dim, hidden_dim)
			if xavier:
				self.weights_init(conv)
			self.convs.append(conv)
			self.acts.append(a)
			if bn:
				self.bns.append(BatchNorm1d(hidden_dim))

	def weights_init(self, module):
		for m in module.modules():
			if isinstance(m, SAGEConv):
				layers = [m.lin_l, m.lin_r]
			if isinstance(m, Linear):
				layers = [m]
			for layer in layers:
				torch.nn.init.xavier_uniform_(layer.weight.data)
				if layer.bias is not None:
					layer.bias.data.fill_(0.0)

	def forward(self, data):
		x, edge_index, batch = data
		xs = []
		for i in range(self.n_layers):
			x = self.convs[i](x, edge_index)
			x = self.acts[i](x)
			if self.bns is not None:
				x = self.bns[i](x)
			xs.append(x)

		if self.pool == "sum":
			xpool = [global_add_pool(x, batch) for x in xs]
		else:
			xpool = [global_mean_pool(x, batch) for x in xs]
		global_rep = torch.cat(xpool, 1)

		return global_rep, x


class GIN(torch.nn.Module):
	def __init__(self, feat_dim, hidden_dim, n_layers=3, pool="sum", bn=False, xavier=True):
		super(GIN, self).__init__()

		if bn:
			self.bns = torch.nn.ModuleList()
		self.convs = torch.nn.ModuleList()
		self.n_layers = n_layers
		self.pool = pool

		self.act = torch.nn.ReLU()

		for i in range(n_layers):
			start_dim = hidden_dim if i else feat_dim
			mlp = Sequential(Linear(start_dim, hidden_dim),
							self.act,
							Linear(hidden_dim, hidden_dim))
			if xavier:
				self.weights_init(mlp)
			conv = GINConv(mlp)
			self.convs.append(conv)
			if bn:
				self.bns.append(BatchNorm1d(hidden_dim))

	def weights_init(self, module):
		for m in module.modules():
			if isinstance(m, Linear):
				torch.nn.init.xavier_uniform_(m.weight.data)
				if m.bias is not None:
					m.bias.data.fill_(0.0)

	def forward(self, data):
		x, edge_index, batch = data
		xs = []
		for i in range(self.n_layers):
			x = self.convs[i](x, edge_index)
			x = self.act(x)
			if self.bns is not None:
				x = self.bns[i](x)
			xs.append(x)

		if self.pool == "sum":
			xpool = [global_add_pool(x, batch) for x in xs]
		else:
			xpool = [global_mean_pool(x, batch) for x in xs]
		global_rep = torch.cat(xpool, 1)

		return global_rep, x


class SGC(torch.nn.Module):
	def __init__(self, feat_dim, hidden_dim, n_layers=3, pool="sum", bn=False, xavier=True):
		super(SGC, self).__init__()

		self.pool = pool

		a = torch.nn.ReLU()

		conv = SGConv(feat_dim, hidden_dim, n_layers)
		if xavier:
			self.weights_init(conv)
		self.convs = conv
		self.acts = a
		if bn:
			self.bns = BatchNorm1d(hidden_dim)

	def weights_init(self, module):
		for m in module.modules():
			if isinstance(m, SGConv):
				layer = m.lin
			if isinstance(m, Linear):
				layer = m
			torch.nn.init.xavier_uniform_(layer.weight.data)
			if layer.bias is not None:
				layer.bias.data.fill_(0.0)

	def forward(self, data):
		x, edge_index, batch = data

		x = self.convs(x, edge_index)
		x = self.acts(x)
		if self.bns is not None:
			x = self.bns(x)

		if self.pool == "sum":
			global_rep = global_add_pool(x, batch)
		else:
			global_rep = global_mean_pool(x, batch)

		return global_rep, x


class ResGCNConv(MessagePassing):
	def __init__(self, in_channels, out_channels, improved=False, cached=False, edge_norm=True, gfn=False):
		super(ResGCNConv, self).__init__("add")

		self.in_channels = in_channels
		self.out_channels = out_channels
		self.improved = improved
		self.cached = cached
		self.cached_result = None
		self.edge_norm = edge_norm
		self.gfn = gfn

		self.weight = Parameter(torch.Tensor(in_channels, out_channels))
		self.bias = Parameter(torch.Tensor(out_channels))

		self.weights_init()

	def weights_init(self):
		glorot(self.weight)
		zeros(self.bias)
		self.cached_result = None

	@staticmethod
	def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
		if edge_weight is None:
			edge_weight = torch.ones((edge_index.size(1), ),
									 dtype=dtype,
									 device=edge_index.device)
		edge_weight = edge_weight.view(-1)
		assert edge_weight.size(0) == edge_index.size(1)

		edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
		edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
		# Add edge_weight for loop edges.
		loop_weight = torch.full((num_nodes, ),
								 1 if not improved else 2,
								 dtype=edge_weight.dtype,
								 device=edge_weight.device)
		edge_weight = torch.cat([edge_weight, loop_weight], dim=0)
		row, col = edge_index
		
		deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
		deg_inv_sqrt = deg.pow(-0.5)
		deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

		return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

	def forward(self, x, edge_index, edge_weight=None):
		x = torch.matmul(x, self.weight)
		if self.gfn:
			return x

		if not self.cached or self.cached_result is None:
			if self.edge_norm:
				edge_index, norm = ResGCNConv.norm(edge_index, x.size(0), edge_weight, self.improved, x.dtype)
			else:
				norm = None
			self.cached_result = edge_index, norm

		edge_index, norm = self.cached_result
		return self.propagate(edge_index, x=x, norm=norm)

	def message(self, x_j, norm):
		if self.edge_norm:
			return norm.view(-1, 1) * x_j
		else:
			return x_j

	def update(self, aggr_out):
		if self.bias is not None:
			aggr_out = aggr_out + self.bias
		return aggr_out

	def __repr__(self):
		return "{}({}, {})".format(self.__class__.__name__, self.in_channels, self.out_channels)


class ResGCN(torch.nn.Module):
	def __init__(self, feat_dim, hidden_dim, num_conv_layers, pool,
				 num_feat_layers=1, num_fc_layers=2, xg_dim=None, bn=True, gfn=False,
				 collapse=False, residual=False, dropout=0, edge_norm=True):
		super(ResGCN, self).__init__()

		assert num_feat_layers == 1, "more feat layers are not now supported"
		self.conv_residual = residual
		self.fc_residual = False  # no skip-connections for fc layers.
		self.collapse = collapse
		self.bn = bn

		assert "sum" in pool or "mean" in pool, pool
		if "sum" in pool:
			self.pool = global_add_pool
		else:
			self.pool = global_mean_pool
		self.dropout = dropout
		GConv = partial(ResGCNConv, edge_norm=edge_norm, gfn=gfn)

		if xg_dim is not None:  # Utilize graph level features.
			self.use_xg = True
			self.bn1_xg = BatchNorm1d(xg_dim)
			self.lin1_xg = Linear(xg_dim, hidden_dim)
			self.bn2_xg = BatchNorm1d(hidden_dim)
			self.lin2_xg = Linear(hidden_dim, hidden_dim)
		else:
			self.use_xg = False

		if collapse:
			self.bn_feat = BatchNorm1d(feat_dim)
			self.bns_fc = torch.nn.ModuleList()
			self.lins = torch.nn.ModuleList()
			if "gating" in pool:
				self.gating = torch.nn.Sequential(
					Linear(feat_dim, feat_dim),
					torch.nn.ReLU(),
					Linear(feat_dim, 1),
					torch.nn.Sigmoid())
			else:
				self.gating = None
			for i in range(num_fc_layers - 1):
				self.bns_fc.append(BatchNorm1d(hidden_in))
				self.lins.append(Linear(hidden_in, hidden_dim))
				hidden_in = hidden_dim
		else:
			self.bn_feat = BatchNorm1d(feat_dim)
			feat_gfn = True  # set true so GCNConv is feat transform
			self.conv_feat = ResGCNConv(feat_dim, hidden_dim, gfn=feat_gfn)
			if "gating" in pool:
				self.gating = torch.nn.Sequential(
					Linear(hidden_dim, hidden_dim),
					torch.nn.ReLU(),
					Linear(hidden_dim, 1),
					torch.nn.Sigmoid())
			else:
				self.gating = None
			self.bns_conv = torch.nn.ModuleList()
			self.convs = torch.nn.ModuleList()
			for i in range(num_conv_layers):
				self.bns_conv.append(BatchNorm1d(hidden_dim))
				self.convs.append(GConv(hidden_dim, hidden_dim))
			self.bn_hidden = BatchNorm1d(hidden_dim)
			self.bns_fc = torch.nn.ModuleList()
			self.lins = torch.nn.ModuleList()
			for i in range(num_fc_layers - 1):
				self.bns_fc.append(BatchNorm1d(hidden_dim))
				self.lins.append(Linear(hidden_dim, hidden_dim))

		# BN initialization.
		for m in self.modules():
			if isinstance(m, (torch.nn.BatchNorm1d)):
				torch.nn.init.constant_(m.weight, 1)
				torch.nn.init.constant_(m.bias, 0.0001)
		
	def forward(self, data):
		x, edge_index, batch = data
		if self.use_xg:
			# xg is of shape [n_graphs, feat_dim]
			xg = self.bn1_xg(data.xg) if self.bn else xg
			xg = F.relu(self.lin1_xg(xg))
			xg = self.bn2_xg(xg) if self.bn else xg
			xg = F.relu(self.lin2_xg(xg))
		else:
			xg = None
		
		x = self.bn_feat(x) if self.bn else x
		x = F.relu(self.conv_feat(x, edge_index))
		for i, conv in enumerate(self.convs):
			x_ = self.bns_conv[i](x) if self.bn else x
			x_ = F.relu(conv(x_, edge_index))
			x = x + x_ if self.conv_residual else x_
		local_rep = x
		gate = 1 if self.gating is None else self.gating(x)
		x = self.pool(x * gate, batch)
		x = x if xg is None else x + xg
		
		for i, lin in enumerate(self.lins):
			x_ = self.bns_fc[i](x) if self.bn else x
			x_ = F.relu(lin(x_))
			x = x + x_ if self.fc_residual else x_
		
		x = self.bn_hidden(x)
		if self.dropout > 0:
			x = F.dropout(x, p=self.dropout, training=self.training)

		return x, local_rep


class PredictionModel(nn.Module):
	def __init__(self, feat_dim, hidden_dim, n_layers, output_dim, gnn, load=None):
		super(PredictionModel, self).__init__()

		self.encoder = Encoder(feat_dim, hidden_dim, n_layers=n_layers, gnn=gnn)

		if load:
			ckpt = torch.load(os.path.join("logs", load, "best_model.ckpt"))
			self.encoder.load_state_dict(ckpt["state"])
			for param in self.encoder.parameters():
				param.requires_grad = False

		if gnn in ["resgcn", "sgc"]:
			feat_dim = hidden_dim
		else:
			feat_dim = n_layers * hidden_dim
		self.classifier = nn.Linear(feat_dim, output_dim)

	def forward(self, data):
		embeddings = self.encoder(data)
		scores = self.classifier(embeddings)
		return scores

	def save_checkpoint(self, save_path, optimizer, epoch, best_train_loss, best_val_loss, is_best):
		ckpt = {}
		ckpt["state"] = self.state_dict()
		ckpt["epoch"] = epoch
		ckpt["optimizer_state"] = optimizer.state_dict()
		ckpt["best_train_loss"] = best_train_loss
		ckpt["best_val_loss"] = best_val_loss
		torch.save(ckpt, os.path.join(save_path, "pred_model.ckpt"))
		if is_best:
			torch.save(ckpt, os.path.join(save_path, "best_pred_model.ckpt"))

	def load_checkpoint(self, load_path, optimizer):
		ckpt = torch.load(os.path.join(load_path, "best_pred_model.ckpt"))
		self.load_state_dict(ckpt["state"])
		epoch = ckpt["epoch"]
		best_train_loss = ckpt["best_train_loss"]
		best_val_loss = ckpt["best_val_loss"]
		optimizer.load_state_dict(ckpt["optimizer_state"])
		return epoch, best_train_loss, best_val_loss
