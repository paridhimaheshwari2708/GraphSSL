# All About Self-Supervised Learning For Graphs

## Introduction
This repository serves as a mini-tutorial on using self-supervised learning for graphs. Self-supervised learning is a class of unsupervised machine learning methods where the goal is to learn rich representations of unstructured data when we do not have access to any labels. This repository implements a variety of commonly used methods (augmentations/loss functions) for self-supervised learning on graphs. The codebase also includes the option of loading commonly used graph datasets for a variety of downstream tasks and is built using [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) which is a library built on PyTorch for graph machine learning.

## Setting up the environment
Setup the conda environment which ensures the installation of the correct version of PyTorch Geometric (PyG) and all other dependencies.

```
conda env create -f environment.yml
conda activate cs224w
```


