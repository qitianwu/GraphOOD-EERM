import argparse
import sys
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from torch_scatter import scatter

from logger import Logger, SimpleLogger
from dataset import load_nc_dataset
from data_utils import normalize, gen_normalized_adjs, evaluate, evaluate_whole_graph, eval_acc, eval_rocauc, eval_f1, to_sparse_tensor, load_fixed_splits
from parse import parse_method_base, parse_method_ours, parser_add_main_args

# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
fix_seed(0)

### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

def get_dataset(dataset, ratio=None, sub_dataset=None, gen_model=None):
    ### Load and preprocess data ###
    if dataset == 'cora':
        dataset = load_nc_dataset('cora', sub_dataset, gen_model)
    elif dataset == 'amazon-photo':
        dataset = load_nc_dataset('amazon-photo', sub_dataset, gen_model)
    else:
        raise ValueError('Invalid dataname')

    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)

    dataset.n = dataset.graph['num_nodes']
    dataset.c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    dataset.d = dataset.graph['node_feat'].shape[1]

    dataset.graph['edge_index'], dataset.graph['node_feat'] = \
        dataset.graph['edge_index'], dataset.graph['node_feat']
    return dataset

if args.dataset == 'cora':
    tr_sub, val_sub, te_subs = [0], [1], list(range(2, 10))
    gen_model = args.gnn_gen
    dataset_tr = get_dataset(dataset='cora', sub_dataset=tr_sub[0], gen_model=gen_model)
    dataset_val = get_dataset(dataset='cora', sub_dataset=val_sub[0], gen_model=gen_model)
    datasets_te = [get_dataset(dataset='cora', sub_dataset=te_subs[i], gen_model=gen_model) for i in range(len(te_subs))]
elif args.dataset == 'amazon-photo':
    tr_sub, val_sub, te_subs = [0], [1], list(range(2, 10))
    gen_model = args.gnn_gen
    dataset_tr = get_dataset(dataset='amazon-photo', sub_dataset=tr_sub[0], gen_model=gen_model)
    dataset_val = get_dataset(dataset='amazon-photo', sub_dataset=val_sub[0], gen_model=gen_model)
    datasets_te = [get_dataset(dataset='amazon-photo', sub_dataset=te_subs[i], gen_model=gen_model) for i in range(len(te_subs))]
else:
    raise ValueError('Invalid dataname')

print(f"Train num nodes {dataset_tr.n} | num classes {dataset_tr.c} | num node feats {dataset_tr.d}")
print(f"Val num nodes {dataset_val.n} | num classes {dataset_val.c} | num node feats {dataset_val.d}")
for i in range(len(te_subs)):
    dataset_te = datasets_te[i]
    print(f"Test {i} num nodes {dataset_te.n} | num classes {dataset_te.c} | num node feats {dataset_te.d}")

### Load method ###
if args.method == 'erm':
    model = parse_method_base(args, dataset_tr, dataset_tr.n, dataset_tr.c, dataset_tr.d, device)
else:
    model = parse_method_ours(args, dataset_tr, dataset_tr.n, dataset_tr.c, dataset_tr.d, device)

# using rocauc as the eval function
criterion = nn.NLLLoss()
eval_func = eval_acc

logger = Logger(args.runs, args)

model.train()
print('MODEL:', model)
print('DATASET:', args.dataset)

### Training loop ###
for run in range(args.runs):
    # split_idx = split_idx_lst[run]
    # train_idx = split_idx['train'].to(device)
    model.reset_parameters()
    if args.method == 'erm':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.method == 'eerm':
        optimizer_gnn = torch.optim.AdamW(model.gnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_aug = torch.optim.AdamW(model.gl.parameters(), lr=args.lr_a)
    best_val = float('-inf')
    for epoch in range(args.epochs):
        model.train()
        if args.method == 'erm':
            optimizer.zero_grad()
            loss = model(dataset_tr, criterion)
            loss.backward()
            optimizer.step()
        elif args.method == 'eerm':
            model.gl.reset_parameters()
            beta = 1 * args.beta * epoch / args.epochs + args.beta * (1- epoch / args.epochs)
            for m in range(args.T):
                Var, Mean, Log_p = model(dataset_tr, criterion)
                outer_loss = Var + beta * Mean
                reward = Var.detach()
                inner_loss = - reward * Log_p
                if m == 0:
                    optimizer_gnn.zero_grad()
                    outer_loss.backward()
                    optimizer_gnn.step()
                optimizer_aug.zero_grad()
                inner_loss.backward()
                optimizer_aug.step()

        accs, test_outs = evaluate_whole_graph(args, model, dataset_tr, dataset_val, datasets_te, eval_func)
        # accs2, test_outs2 = evaluate_whole_graph(args, model, dataset_tr, dataset_val, datasets_te, eval_func, partial=True)
        # logger.add_result(run, accs+accs2)
        logger.add_result(run, accs)

        if epoch % args.display_step == 0:
            if args.method == 'erm':
                print(f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * accs[0]:.2f}%, '
                      f'Valid: {100 * accs[1]:.2f}%, ')
                test_info = ''
                for test_acc in accs[2:]:
                    test_info += f'Test: {100 * test_acc:.2f}% '
                print(test_info)
            elif args.method == 'eerm':
                print(f'Epoch: {epoch:02d}, '
                      f'Mean Loss: {Mean:.4f}, '
                      f'Var Loss: {Var:.4f}, '
                      f'Train: {100 * accs[0]:.2f}%, '
                      f'Valid: {100 * accs[1]:.2f}%, ')
                test_info = ''
                for test_acc in accs[2:]:
                    test_info += f'Test: {100 * test_acc:.2f}% '
                print(test_info)

    logger.print_statistics(run)

### Save results ###
results = logger.print_statistics()
filename = f'./results/{args.dataset}.csv'
print(f"Saving results to {filename}")
with open(f"{filename}", 'a+') as write_obj:
    # sub_dataset = f'{args.sub_dataset},' if args.sub_dataset else ''
    log = f"{args.method}," + f"{args.gnn},"
    for i in range(results.shape[1]):
        r = results[:, i]
        log += f"{r.mean():.3f} ± {r.std():.3f},"
    write_obj.write(log + f"\n")
    for i in range(3, results.shape[1]):
        log = ''
        for k in range(results.shape[0]):
            log += f"{results[k, i]:.4f} "
        write_obj.write(log + f"\n")
