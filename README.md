# Self-Supervised Learning For Graphs

## Introduction
This repository serves as a mini-tutorial on using self-supervised learning for graphs. Self-supervised learning is a class of unsupervised machine learning methods where the goal is to learn rich representations of unstructured data when we do not have access to any labels. This repository implements a variety of commonly used methods (augmentations, encoders, loss functions) for self-supervised learning on graphs. The codebase also includes the option of loading commonly used graph datasets for a variety of downstream tasks. It is built using [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) which is a library built on PyTorch for graph machine learning.

## Setting up the environment
Setup the conda environment which ensures the installation of the correct version of PyTorch Geometric (PyG) and all other dependencies.

```
conda env create -f environment.yml
conda activate graphssl
```

Clone this repository.

```
git clone https://github.com/paridhimaheshwari2708/GraphSSL.git
cd GraphSSL/
```

## Self-supervised pretraining 
For training a self-supervised model, we use the <code>run.py</code> script and we need to specify a few important arguments:
```
--save                  Specify where folder name of the experiment where the logs and models are saved
--dataset               Specify the dataset on which you want to train the model
--model                 Specify the model architecture of the GNN Encoder
--feat_dim              Specify the dimension of node features in GNN
--layers                Specify the number of layers of GNN Encoder
--loss                  Specify the loss function for contrastive training
--augment_list          Specify the augmentations to be applied as space separated strings
```

The options supported for above arguments are:
| Argument      | Choices           |
| ------------- |:-------------:|
| dataset      | proteins, enzymes, collab ,reddit_binary, reddit_multi, imdb_binary, imdb_multi, dd, mutag, nci1 |
| model      | gcn, gin, resgcn, gat, graphsage, sgc |
| loss | infonce, jensen_shannon |
| augment_list | edge_perturbation, diffusion, diffusion_with_sample, node_dropping, random_walk_subgraph, node_attr_mask |

As an example, run the following command to train a self-supervised model on the proteins dataset
```
python3 run.py --save proteins_exp --dataset proteins --model gcn --loss infonce --augment_list edge_perturbation node_dropping
```

## Evaluation on graph classification
For training and evaluating the model on the downstream task, here graph classification, we use the <code>run_classification.py</code> script. The arguments are:
```
--save                  Specify where folder name of the experiment where the logs and models are saved
--load                  Specify the folder name from which we want the self-supervised model is loaded
                        If present, the GNN loads the pretrained weights and only the classifier head is trained
                        If left empty, the model will be trained end-to-end without self-supervised learning
--dataset               Specify the dataset on which you want to train the model
--model                 Specify the model architecture of the GNN Encoder
--feat_dim              Specify the dimension of node features in GNN
--layers                Specify the number of layers of GNN Encoder
--train_data_percent    Specify the fraction of training samples which are labelled
```

The options supported for above arguments are:
| Argument      | Choices           |
| ------------- |:-------------:|
| dataset      | proteins, enzymes, collab ,reddit_binary, reddit_multi, imdb_binary, imdb_multi, dd, mutag, nci1 |
| model      | gcn, gin, resgcn, gat, graphsage, sgc |
| augment_list | edge_perturbation, diffusion, diffusion_with_sample, node_dropping, random_walk_subgraph, node_attr_mask |


As an example, run the following command to train the final classifier head of a self-supervised model on the proteins dataset
```
python3 run_classification.py --save proteins_exp_finetuned --load proteins_exp --dataset proteins --model gcn --train_data_percent 1.0
```

For the same dataset, training the model end-to-end (without self-supervised pretraining) can be done as follows
```
python3 run_classification.py --save proteins_exp_finetuned_e2e  --dataset proteins --model gcn --train_data_percent 1.0
```

## Tutorial
We have also created a [Colab Notebook](https://colab.research.google.com/drive/1WyGiVd9z4TqimA6vN_yRNoQH9E1MmsCm?usp=sharing) which combines various techniques in self-supervised learning and provides an easy-to-use interface for training your own models.
