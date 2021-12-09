# All About Self-Supervised Learning For Graphs

## Introduction
This repository serves as a mini-tutorial on using self-supervised learning for graphs. Self-supervised learning is a class of unsupervised machine learning methods where the goal is to learn rich representations of unstructured data when we do not have access to any labels. This repository implements a variety of commonly used methods (augmentations/loss functions) for self-supervised learning on graphs. The codebase also includes the option of loading commonly used graph datasets for a variety of downstream tasks and is built using [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) which is a library built on PyTorch for graph machine learning.

## Setting up the environment
Setup the conda environment which ensures the installation of the correct version of PyTorch Geometric (PyG) and all other dependencies.

```
conda env create -f environment.yml
conda activate cs224w
```

## Training 
For training a self-supervised model, we need to specify a few important arguments to the <code>run.py<\code> script.

```
--save                  Specify where folder name of the experiment where the logs and models shall be save
--dataset               Specify the dataset on which you want to train the model
--model                 Specify the model architecture of the GNN Encoder
--loss                  Specify the loss function for contrastive training
--augment_list          Specify the augmentations to be applied as space separated strings
--train_data_percent    Specify the fraction of training samples which are labelled
```
  
As an example, one run of the train a self-supervised model on the proteins dataset shall be as follows
```
python3 run.py --save proteins_exp --dataset proteins --model gcn --loss infonce --augment_list edge_perturbation node_dropping --train_data_percent 1
```

