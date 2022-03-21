# GraphOOD-EERM

Codes and datasets for [Handling Distribution Shifts on Graphs: An Invariance Perspective](https://arxiv.org/abs/2202.02466).
This work focuses on distribution shifts on graph data and proposes a new approach ***Explore-to-Extrapolate Risk Minimization (EERM)*** for out-of-distribution generalization. 

![image](https://user-images.githubusercontent.com/22075007/159216692-ebfa0819-003e-4d5b-bd49-51a48aa31ffd.png)

## Dependency

PYTHON 3.7, PyTorch 1.9.0, PyTorch Geometric 1.7.2

## Datasets

In our experiment, we consider three types of distribution shifts with six real-world datasets. The information of experimental datasets is summarized in the following Table.

![image](https://user-images.githubusercontent.com/22075007/159216628-bf02255c-d4b3-43a2-8ff0-ac480d8d967a.png)

You can make a directory `./data` and download all the datasets through the Google drive:

      https://drive.google.com/drive/folders/15YgnsfSV_vHYTXe7I4e_hhGMcx0gKrO8?usp=sharing

Here is a brief introduction for three distribution shifts and the datasets:

- Artificial Transformation: We use Cora and Amazon-Photo datasets to construct spurious node features. The data construction script is provided in `./synthetic/synthetic.py`. The original datasets can easily accessed via Pytorch Geometric package. 

- Cross-Domain Transfer: We use Twitch-Explicit and Facebook-100 datasets. These two datasets both contain multiple graphs. We use different graphs for training/validation/testing. The original Twitch dataset is from [Non-Homophily Benchmark](https://github.com/CUAI/Non-Homophily-Benchmarks/tree/main/data/twitch). For Facebook dataset, we use partial graphs for experiments, and its complete version could be obtained from [Facebook dataset](https://archive.org/details/oxford-2005-facebook-matrix).

- Temporal Evolution: We use Elliptic and OGBN-Arxiv datasets. The raw Elliptic data is from [Kaggle dataset](https://www.kaggle.com/ellipticco/elliptic-data-set). For OGB dataset, see the [OGB website](https://ogb.stanford.edu/docs/nodeprop/) for more details.

## Running the code

We do not provide the trained model since the training cost for each experiment is acceptable. To run the code, please refer to the bash script `run.sh` in each folder. For example, the training script for ***Cora*** and ***Amazon-Photo*** is

```shell
      # cora
      python main.py --method base --dataset cora --gnn gcn --run 20 --lr 0.001 --device 1

      python main.py --method policy --gnn gcn --lr 0.01 --weight_decay 1e-3 --num_layers 2 --K 3 --T 1 --num_sample 5 --beta 2.0 --lr_a 0.005 --device 1 -  -dataset cora --run 20 --device 1

      # amazon-photo
      python main.py --method base --dataset amazon-photo --gnn gcn --run 20 --lr 0.001 --device 1

      python main.py --method policy --gnn gcn --lr 0.01 --weight_decay 1e-3 --num_layers 2 --K 3 --T 1 --num_sample 5 --beta 2.0 --lr_a 0.005 --device 1 --dataset amazon-photo --run 20 --device 1
```


More information will be updated. Welcome to contact me <echo740@sjtu.edu.cn> for any question.

If you found the codes and datasets are useful, please cite our paper

```bibtex
      @inproceedings{wu2022eerm,
      title = {Handling Distribution Shifts on Graphs: An Invariance Perspective},
      author = {Qitian Wu and Hengrui Zhang and Junchi Yan and David Wipf},
      booktitle = {International Conference on Learning Representations (ICLR)},
      year = {2022}
      }
```
