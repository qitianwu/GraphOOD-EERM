from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import scipy
import scipy.io
from sklearn.preprocessing import label_binarize
from ogb.nodeproppred import NodePropPredDataset
from torch_geometric.datasets import Planetoid, Amazon

from data_utils import rand_train_test_idx, even_quantile_labels, to_sparse_tensor, dataset_drive_url

from os import path

import pickle as pkl

class NCDataset(object):
    def __init__(self, name):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        
        Usage after construction: 
        
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]
        
        Where the graph is a dictionary of the following form: 
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/
        
        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):  
        return '{}({})'.format(self.__class__.__name__, len(self))

def load_nc_dataset(data_dir, dataname, sub_dataname='', gen_model='gcn'):
    """ Loader for NCDataset
        Returns NCDataset
    """
    if dataname in  ('cora', 'amazon-photo'):
        dataset = load_synthetic_dataset(data_dir, dataname, sub_dataname, gen_model)
    else:
        raise ValueError('Invalid dataname')
    return dataset

def load_synthetic_dataset(data_dir, name, lang, gen_model='gcn'):
    dataset = NCDataset(lang)

    assert lang in range(0, 10), 'Invalid dataset'

    if name == 'cora':
        node_feat, y = pkl.load(open('{}/Planetoid/cora/gen/{}-{}.pkl'.format(data_dir, lang, gen_model), 'rb'))
        torch_dataset = Planetoid(root='{}/Planetoid'.format(data_dir),
                              name='cora')
    elif name == 'amazon-photo':
        node_feat, y = pkl.load(open('{}/Amazon/Photo/gen/{}-{}.pkl'.format(data_dir, lang, gen_model), 'rb'))
        torch_dataset = Amazon(root='{}/Amazon'.format(data_dir),
                                  name='Photo')
    data = torch_dataset[0]

    edge_index = data.edge_index
    # label = data.y
    label = y
    num_nodes = node_feat.size(0)

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}

    dataset.label = label

    return dataset
