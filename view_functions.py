import torch
import random
import numpy as np
from torch_geometric.data import Batch, Data
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.utils import to_dense_adj, dense_to_sparse, subgraph

"""
Part of the code is adapted from https://github.com/divelab/DIG.
"""

class EdgePerturbation():
    """
    Edge perturbation on the given graph or batched graphs. Class objects callable via 
    method :meth:`views_fn`.
    
    Args:
        add (bool, optional): Set :obj:`True` if randomly add edges in a given graph.
            (default: :obj:`True`)
        drop (bool, optional): Set :obj:`True` if randomly drop edges in a given graph.
            (default: :obj:`False`)
        ratio (float, optional): Percentage of edges to add or drop. (default: :obj:`0.1`)
    """

    def __init__(self, add=True, drop=False, ratio=0.1):
        self.add = add
        self.drop = drop
        self.ratio = ratio
        
    def __call__(self, data):
        return self.views_fn(data)
        
    def do_trans(self, data):
        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()
        perturb_num = int(edge_num * self.ratio)

        edge_index = data.edge_index.detach().clone()
        idx_remain = edge_index
        idx_add = torch.tensor([]).reshape(2, -1).long()

        if self.drop:
            idx_remain = edge_index[:, np.random.choice(edge_num, edge_num-perturb_num, replace=False)]

        if self.add:
            idx_add = torch.randint(node_num, (2, perturb_num))

        new_edge_index = torch.cat((idx_remain, idx_add), dim=1)
        new_edge_index = torch.unique(new_edge_index, dim=1)

        return Data(x=data.x, edge_index=new_edge_index)

    def views_fn(self, data):
        """
        Method to be called when :class:`EdgePerturbation` object is called.
        
        Args:
            data (:class:`torch_geometric.data.Data`): The input graph or batched graphs.
            
        :rtype: :class:`torch_geometric.data.Data`.  
        """

        if isinstance(data, Batch):
            dlist = [self.do_trans(d) for d in data.to_data_list()]
            return Batch.from_data_list(dlist)
        elif isinstance(data, Data):
            return self.do_trans(data)


class Diffusion():
    """
    Diffusion on the given graph or batched graphs, used in 
    `MVGRL <https://arxiv.org/pdf/2006.05582v1.pdf>`_. Class objects callable via 
    method :meth:`views_fn`.
    
    Args:
        mode (string, optional): Diffusion instantiation mode with two options:
            :obj:`"ppr"`: Personalized PageRank; :obj:`"heat"`: heat kernel.
            (default: :obj:`"ppr"`)
        alpha (float, optinal): Teleport probability in a random walk. (default: :obj:`0.2`)
        t (float, optinal): Diffusion time. (default: :obj:`5`)
        add_self_loop (bool, optional): Set True to add self-loop to edge_index.
            (default: :obj:`True`)
    """

    def __init__(self, mode="ppr", alpha=0.2, t=5, add_self_loop=True):
        self.mode = mode
        self.alpha = alpha
        self.t = t
        self.add_self_loop = add_self_loop
        
    def __call__(self, data):
        return self.views_fn(data)
    
    def do_trans(self, data):
        node_num, _ = data.x.size()
        if self.add_self_loop:
            sl = torch.tensor([[n, n] for n in range(node_num)]).t()
            edge_index = torch.cat((data.edge_index, sl), dim=1)
        else:
            edge_index = data.edge_index.detach().clone()
        
        orig_adj = to_dense_adj(edge_index)[0]
        orig_adj = torch.where(orig_adj>1, torch.ones_like(orig_adj), orig_adj)
        d = torch.diag(torch.sum(orig_adj, 1))

        if self.mode == "ppr":
            dinv = torch.inverse(torch.sqrt(d))
            at = torch.matmul(torch.matmul(dinv, orig_adj), dinv)
            diff_adj = self.alpha * torch.inverse((torch.eye(orig_adj.shape[0]) - (1 - self.alpha) * at))

        elif self.mode == "heat":
            diff_adj = torch.exp(self.t * (torch.matmul(orig_adj, torch.inverse(d)) - 1))

        else:
            raise Exception("Must choose one diffusion instantiation mode from 'ppr' and 'heat'!")
            
        edge_ind, edge_attr = dense_to_sparse(diff_adj)

        return Data(x=data.x, edge_index=edge_ind, edge_attr=edge_attr)

    def views_fn(self, data):
        """
        Method to be called when :class:`Diffusion` object is called.
        
        Args:
            data (:class:`torch_geometric.data.Data`): The input graph or batched graphs.
            
        :rtype: :class:`torch_geometric.data.Data`.  
        """

        if isinstance(data, Batch):
            dlist = [self.do_trans(d) for d in data.to_data_list()]
            return Batch.from_data_list(dlist)
        elif isinstance(data, Data):
            return self.do_trans(data)


class DiffusionWithSample():
    """
    Diffusion with node sampling on the given graph for node-level datasets, used in 
    `MVGRL <https://arxiv.org/pdf/2006.05582v1.pdf>`_.  Class objects callable via method 
    :meth:`views_fn`.
    
    Args:
        sample_size (int, optional): Number of nodes in the sampled subgraoh from a large graph.
            (default: :obj:`2000`)
        batch_size (int, optional): Number of subgraphs to sample. (default: :obj:`4`)
        mode (string, optional): Diffusion instantiation mode with two options:
            :obj:`"ppr"`: Personalized PageRank; :obj:`"heat"`: heat kernel; 
            (default: :obj:`"ppr"`)
        alpha (float, optinal): Teleport probability in a random walk. (default: :obj:`0.2`)
        t (float, optinal): Diffusion time. (default: :obj:`5`)
        add_self_loop (bool, optional): Set True to add self-loop to edge_index.
            (default: :obj:`True`)
    """

    def __init__(self, sample_size=2000, batch_size=4, mode="ppr", 
                 alpha=0.2, t=5, epsilon=False, add_self_loop=True):
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.mode = mode
        self.alpha = alpha
        self.t = t
        self.epsilon = epsilon
        self.add_self_loop = add_self_loop
        
    def __call__(self, data):
        return self.views_fn(data)
    
    def views_fn(self, data):
        """
        Method to be called when :class:`DiffusionWithSample` object is called.
        
        Args:
            data (:class:`torch_geometric.data.Data`): The input graph or batched graphs.
            
        :rtype: :class:`torch_geometric.data.Data`.  
        """

        node_num, _ = data.x.size()
        if self.add_self_loop:
            sl = torch.tensor([[n, n] for n in range(node_num)]).t()
            edge_index = torch.cat((data.edge_index, sl), dim=1)
        else:
            edge_index = data.edge_index.detach().clone()
        
        orig_adj = to_dense_adj(edge_index)[0]
        orig_adj = torch.where(orig_adj>1, torch.ones_like(orig_adj), orig_adj)
        d = torch.diag(torch.sum(orig_adj, 1))

        if self.mode == "ppr":
            dinv = torch.inverse(torch.sqrt(d))
            at = torch.matmul(torch.matmul(dinv, orig_adj), dinv)
            diff_adj = self.alpha * torch.inverse((torch.eye(orig_adj.shape[0]) - (1 - self.alpha) * at))

        elif self.mode == "heat":
            diff_adj = torch.exp(self.t * (torch.matmul(orig_adj, torch.inverse(d)) - 1))

        else:
            raise Exception("Must choose one diffusion instantiation mode from 'ppr' and 'heat'!")

        if self.epsilon:
            epsilons = [1e-5, 1e-4, 1e-3, 1e-2]
            avg_degree = torch.sum(orig_adj) / orig_adj.shape[0]
            ep = epsilons[np.argmin([abs(avg_degree - torch.sum(diff_adj >= e) / diff_adj.shape[0]) for e in epsilons])]

            diff_adj[diff_adj < ep] = 0.0
            scaler = MinMaxScaler()
            scaler.fit(diff_adj)
            diff_adj = torch.tensor(scaler.transform(diff_adj))

        dlist_orig_x = []
        dlist_diff_x = []
        drop_num = node_num - self.sample_size
        for b in range(self.batch_size):
            idx_drop = np.random.choice(node_num, drop_num, replace=False)
            idx_nondrop = [n for n in range(node_num) if not n in idx_drop]

            sample_orig_adj = orig_adj.clone()
            sample_orig_adj = sample_orig_adj[idx_nondrop, :][:, idx_nondrop]

            sample_diff_adj = diff_adj.clone()
            sample_diff_adj = sample_diff_adj[idx_nondrop, :][:, idx_nondrop]

            sample_orig_x = data.x[idx_nondrop]
            
            edge_ind, edge_attr = dense_to_sparse(sample_diff_adj)

            dlist_orig_x.append(Data(x=sample_orig_x, 
                                     edge_index=dense_to_sparse(sample_orig_adj)[0]))
            dlist_diff_x.append(Data(x=sample_orig_x, 
                                     edge_index=edge_ind, 
                                     edge_attr=edge_attr))

        return Batch.from_data_list(dlist_orig_x), Batch.from_data_list(dlist_diff_x)


class UniformSample():
    """
    Uniformly node dropping on the given graph or batched graphs. 
    Class objects callable via method :meth:`views_fn`.
    
    Args:
        ratio (float, optinal): Ratio of nodes to be dropped. (default: :obj:`0.1`)
    """

    def __init__(self, ratio=0.1):
        self.ratio = ratio
    
    def __call__(self, data):
        return self.views_fn(data)
    
    def do_trans(self, data):
        
        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()
        
        keep_num = int(node_num * (1-self.ratio))
        idx_nondrop = torch.randperm(node_num)[:keep_num]
        mask_nondrop = torch.zeros_like(data.x[:,0]).scatter_(0, idx_nondrop, 1.0).bool()
        
        edge_index, _ = subgraph(mask_nondrop, data.edge_index, relabel_nodes=True, num_nodes=node_num)
        return Data(x=data.x[mask_nondrop], edge_index=edge_index)
    
    def views_fn(self, data):
        """
        Method to be called when :class:`UniformSample` object is called.
        
        Args:
            data (:class:`torch_geometric.data.Data`): The input graph or batched graphs.
            
        :rtype: :class:`torch_geometric.data.Data`.  
        """

        if isinstance(data, Batch):
            dlist = [self.do_trans(d) for d in data.to_data_list()]
            return Batch.from_data_list(dlist)
        elif isinstance(data, Data):
            return self.do_trans(data)


class RWSample():
    """
    Subgraph sampling based on random walk on the given graph or batched graphs.
    Class objects callable via method :meth:`views_fn`.
    
    Args:
        ratio (float, optional): Percentage of nodes to sample from the graph.
            (default: :obj:`0.1`)
        add_self_loop (bool, optional): Set True to add self-loop to edge_index.
            (default: :obj:`False`)
    """

    def __init__(self, ratio=0.1, add_self_loop=False):
        self.ratio = ratio
        self.add_self_loop = add_self_loop
    
    def __call__(self, data):
        return self.views_fn(data)
    
    def do_trans(self, data):
        node_num, _ = data.x.size()
        sub_num = int(node_num * self.ratio)

        if self.add_self_loop:
            sl = torch.tensor([[n, n] for n in range(node_num)]).t()
            edge_index = torch.cat((data.edge_index, sl), dim=1)
        else:
            edge_index = data.edge_index.detach().clone()

        # edge_index = edge_index.numpy()
        idx_sub = [np.random.randint(node_num, size=1)[0]]
        # idx_neigh = set([n for n in edge_index[1][edge_index[0]==idx_sub[0]]])
        idx_neigh = set([n.item() for n in edge_index[1][edge_index[0]==idx_sub[0]]])

        count = 0
        while len(idx_sub) <= sub_num:
            count = count + 1
            if count > node_num:
                break
            if len(idx_neigh) == 0:
                break
            sample_node = np.random.choice(list(idx_neigh))
            if sample_node in idx_sub:
                continue
            idx_sub.append(sample_node)
            # idx_neigh.union(set([n for n in edge_index[1][edge_index[0]==idx_sub[-1]]]))
            idx_neigh.union(set([n.item() for n in edge_index[1][edge_index[0]==idx_sub[-1]]]))

        idx_sub = torch.LongTensor(idx_sub).to(data.x.device)
        mask_nondrop = torch.zeros_like(data.x[:,0]).scatter_(0, idx_sub, 1.0).bool()
        edge_index, _ = subgraph(mask_nondrop, data.edge_index, relabel_nodes=True, num_nodes=node_num)
        return Data(x=data.x[mask_nondrop], edge_index=edge_index)

    def views_fn(self, data):
        """
        Method to be called when :class:`RWSample` object is called.
        
        Args:
            data (:class:`torch_geometric.data.Data`): The input graph or batched graphs.
            
        :rtype: :class:`torch_geometric.data.Data`.  
        """

        if isinstance(data, Batch):
            dlist = [self.do_trans(d) for d in data.to_data_list()]
            return Batch.from_data_list(dlist)
        elif isinstance(data, Data):
            return self.do_trans(data)


class NodeAttrMask():
    """
    Node attribute masking on the given graph or batched graphs. 
    Class objects callable via method :meth:`views_fn`.
    
    Args:
        mode (string, optinal): Masking mode with three options:
            :obj:`"whole"`: mask all feature dimensions of the selected node with a Gaussian distribution;
            :obj:`"partial"`: mask only selected feature dimensions with a Gaussian distribution;
            :obj:`"onehot"`: mask all feature dimensions of the selected node with a one-hot vector.
            (default: :obj:`"whole"`)
        mask_ratio (float, optinal): The ratio of node attributes to be masked. (default: :obj:`0.1`)
        mask_mean (float, optional): Mean of the Gaussian distribution to generate masking values.
            (default: :obj:`0.5`)
        mask_std (float, optional): Standard deviation of the distribution to generate masking values. 
            Must be non-negative. (default: :obj:`0.5`)
    """

    def __init__(self, mode='whole', mask_ratio=0.1, mask_mean=0.5, mask_std=0.5, return_mask=False):
        self.mode = mode
        self.mask_ratio = mask_ratio
        self.mask_mean = mask_mean
        self.mask_std = mask_std
        self.return_mask = return_mask
    
    def __call__(self, data):
        return self.views_fn(data)
    
    def do_trans(self, data):
        
        node_num, feat_dim = data.x.size()
        x = data.x.detach().clone()

        if self.mode == "whole":
            mask = torch.zeros(node_num)
            mask_num = int(node_num * self.mask_ratio)
            idx_mask = np.random.choice(node_num, mask_num, replace=False)
            x[idx_mask] = torch.tensor(np.random.normal(loc=self.mask_mean, scale=self.mask_std, 
                                                        size=(mask_num, feat_dim)), dtype=torch.float32)
            mask[idx_mask] = 1

        elif self.mode == "partial":
            mask = torch.zeros((node_num, feat_dim))
            for i in range(node_num):
                for j in range(feat_dim):
                    if random.random() < self.mask_ratio:
                        x[i][j] = torch.tensor(np.random.normal(loc=self.mask_mean, 
                                                                scale=self.mask_std), dtype=torch.float32)
                        mask[i][j] = 1

        elif self.mode == "onehot":
            mask = torch.zeros(node_num)
            mask_num = int(node_num * self.mask_ratio)
            idx_mask = np.random.choice(node_num, mask_num, replace=False)
            x[idx_mask] = torch.tensor(np.eye(feat_dim)[np.random.randint(0, feat_dim, size=(mask_num))], dtype=torch.float32)
            mask[idx_mask] = 1

        else:
            raise Exception("Masking mode option '{0:s}' is not available!".format(mode))

        if self.return_mask:
            return Data(x=x, edge_index=data.edge_index, mask=mask)
        else:
            return Data(x=x, edge_index=data.edge_index)

    def views_fn(self, data):
        """
        Method to be called when :class:`NodeAttrMask` object is called.
        
        Args:
            data (:class:`torch_geometric.data.Data`): The input graph or batched graphs.
            
        :rtype: :class:`torch_geometric.data.Data`.  
        """

        if isinstance(data, Batch):
            dlist = [self.do_trans(d) for d in data.to_data_list()]
            return Batch.from_data_list(dlist)
        elif isinstance(data, Data):
            return self.do_trans(data)
