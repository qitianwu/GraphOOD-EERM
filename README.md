# GraphOOD-EERM

Codes and datasets for [Handling Distribution Shifts on Graphs: An Invariance Perspective](https://arxiv.org/abs/2202.02466).
This work focuses on distribution shifts on graph data and proposes a new approach ***Explore-to-Extrapolate Risk Minimization (EERM)*** for out-of-distribution generalization. 

## Datasets

In our experiment, we consider three types of distribution shifts. You can make a directory `.data` and download the datasets according the following details.

- Artificial Transformation: We use Cora and Amazon-Photo datasets to construct spurious node features. The data construction script is provided in `./synthetic/synthetic.py`. The original datasets can easily accessed via Pytorch Geometric package. To download our preprocessed data, please go to the Google drive:

      https://drive.google.com/drive/folders/15YgnsfSV_vHYTXe7I4e_hhGMcx0gKrO8?usp=sharing

- Cross-Domain Transfer: We use Twitch-Explicit and Facebook-100 datasets. These two datasets both contain multiple graphs. We use different graphs for training/validation/testing. To download the dataset:
     - Twitch: https://github.com/CUAI/Non-Homophily-Benchmarks/tree/main/data/twitch
     - Facebook-100: https://archive.org/details/oxford-2005-facebook-matrix

- Temporal Evolution: We use Elliptic and OGBN-Arxiv datasets. One can download the Elliptic data from [Kaggle dataset](https://www.kaggle.com/ellipticco/elliptic-data-set). For OGB dataset, see the [OGB website](https://ogb.stanford.edu/docs/nodeprop/) for more details.

More information will be updated.


      @inproceedings{wu2022eerm,
      title = {Handling Distribution Shifts on Graphs: An Invariance Perspective},
      author = {Qitian Wu and Hengrui Zhang and Junchi Yan and David Wipf},
      booktitle = {International Conference on Learning Representations (ICLR)},
      year = {2022}
      }
