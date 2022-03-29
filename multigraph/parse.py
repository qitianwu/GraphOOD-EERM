from model import Base, Model, Model_Multi
from data_utils import normalize

def parse_method_base(args, dataset, n, c, d, device):
    if args.gnn == 'gcn':
        model = Base(args, n, c, d, 'gcn', device).to(device)
    elif args.gnn == 'sage':
        model = Base(args, n, c, d, 'sage', device).to(device)
    elif args.gnn == 'gat':
        model = Base(args, n, c, d, 'gat', device).to(device)
    elif args.gnn == 'gpr':
        model = Base(args, n, c, d, 'gpr', device).to(device)
    elif args.gnn == 'gcnii':
        model = Base(args, n, c, d, 'gcnii', device).to(device)
    else:
        raise ValueError('Invalid method')
    return model

def parse_method_ours(args, dataset, n, c, d, device):
    if args.gnn == 'gcn':
        model = Model(args, n, c, d, 'gcn', device).to(device)
    elif args.gnn == 'sage':
        model = Model(args, n, c, d, 'sage', device).to(device)
    elif args.gnn == 'gat':
        model = Model(args, n, c, d, 'gat', device).to(device)
    elif args.gnn == 'gpr':
        model = Model(args, n, c, d, 'gpr', device).to(device)
    elif args.gnn == 'gcnii':
        model = Model(args, n, c, d, 'gcnii', device).to(device)
    else:
        raise ValueError('Invalid method')
    return model


def parse_method_ours_multi(args, datasets, device):
    ns = [dataset.n for dataset in datasets]
    c, d = datasets[0].c, datasets[0].d
    if args.gnn == 'gcn':
        model = Model_Multi(args, ns, c, d, 'gcn', device).to(device)
    elif args.gnn == 'sage':
        model = Model_Multi(args, ns, c, d, 'sage', device).to(device)
    elif args.gnn == 'gat':
        model = Model_Multi(args, ns, c, d, 'gat', device).to(device)
    elif args.gnn == 'gpr':
        model = Model_Multi(args, ns, c, d, 'gpr', device).to(device)
    elif args.gnn == 'gcnii':
        model = Model_Multi(args, ns, c, d, 'gcnii', device).to(device)
    else:
        raise ValueError('Invalid method')
    return model


def parser_add_main_args(parser):
    # dataset and protocol
    parser.add_argument('--data_dir', type=str, default='../data') # need to be specified
    parser.add_argument('--dataset', type=str, default='twitch-e')
    parser.add_argument('--sub_dataset', type=str, default='')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--rocauc', action='store_true',
                        help='set the eval function to rocauc')

    # model
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gnn', type=str, default='gcn')
    parser.add_argument('--method', type=str, default='erm',
                        choices=['erm', 'eerm'])
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers for deep methods')
    parser.add_argument('--no_bn', action='store_true', help='do not use batchnorm')

    # training
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--display_step', type=int,
                        default=1, help='how often to print')
    parser.add_argument('--runs', type=int, default=5,
                        help='number of distinct runs')
    parser.add_argument('--cached', action='store_true',
                        help='set to use faster sgc')
    parser.add_argument('--gat_heads', type=int, default=4,
                        help='attention heads for gat')
    parser.add_argument('--lp_alpha', type=float, default=.1,
                        help='alpha for label prop')
    parser.add_argument('--gpr_alpha', type=float, default=.1,
                        help='alpha for gprgnn')
    parser.add_argument('--gcnii_alpha', type=float, default=.1,
                        help='alpha for gcnii')
    parser.add_argument('--gcnii_lamda', type=float, default=1.0,
                        help='lambda for gcnii')
    parser.add_argument('--directed', action='store_true',
                        help='set to not symmetrize adjacency')

    # for graph edit model
    parser.add_argument('--K', type=int, default=3,
                        help='num of views for data augmentation')
    parser.add_argument('--T', type=int, default=1,
                        help='steps for graph learner before one step for GNN')
    parser.add_argument('--num_sample', type=int, default=5,
                        help='num of samples for each node with graph edit')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='weight for mean of risks from multiple domains')
    parser.add_argument('--lr_a', type=float, default=0.005,
                        help='learning rate for graph learner with graph edit')

