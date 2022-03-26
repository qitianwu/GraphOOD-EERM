import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import to_dense_adj, dense_to_sparse

from nets import *

class Base(nn.Module):
    def __init__(self, args, n, c, d, gnn, device):
        super(Base, self).__init__()
        if gnn == 'gcn':
            self.gnn = GCN(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        use_bn=not args.no_bn)
        elif gnn == 'sage':
            self.gnn = SAGE(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout)
        elif gnn == 'gat':
            self.gnn = GAT(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        heads=args.gat_heads)
        elif gnn == 'gpr':
            self.gnn = GPRGNN(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        dropout=args.dropout,
                        alpha=args.gpr_alpha,
                        )
        self.n = n
        self.device = device
        self.args = args

    def reset_parameters(self):
        self.gnn.reset_parameters()

    def forward(self, data, criterion):
        x, y = data.graph['node_feat'].to(self.device), data.label.to(self.device)
        edge_index = data.graph['edge_index'].to(self.device)
        out = self.gnn(x, edge_index)
        loss = self.sup_loss(y, out, criterion)
        return loss

    def inference(self, data):
        x = data.graph['node_feat'].to(self.device)
        edge_index = data.graph['edge_index'].to(self.device)
        out = self.gnn(x, edge_index)
        return out

    def sup_loss(self, y, pred, criterion):
        out = F.log_softmax(pred, dim=1)
        target = y.squeeze(1)
        loss = criterion(out, target)
        return loss

class Graph_Editer(nn.Module):
    def __init__(self, K, n, device):
        super(Graph_Editer, self).__init__()
        self.B = nn.Parameter(torch.FloatTensor(K, n, n))
        self.device = device

    def reset_parameters(self):
        nn.init.uniform_(self.B)

    def forward(self, edge_index, n, num_sample, k):
        Bk = self.B[k]
        A = to_dense_adj(edge_index, max_num_nodes=n)[0].to(torch.int)
        A_c = torch.ones(n, n, dtype=torch.int).to(self.device) - A
        P = torch.softmax(Bk, dim=0)
        S = torch.multinomial(P, num_samples=num_sample)  # [n, s]
        M = torch.zeros(n, n, dtype=torch.float).to(self.device)
        col_idx = torch.arange(0, n).unsqueeze(1).repeat(1, num_sample)
        M[S, col_idx] = 1.
        C = A + M * (A_c - A)
        edge_index = dense_to_sparse(C)[0]

        log_p = torch.sum(
            torch.sum(Bk[S, col_idx], dim=1) - torch.logsumexp(Bk, dim=0)
        )

        return edge_index, log_p

class Model(nn.Module):
    def __init__(self, args, n, c, d, gnn, device):
        super(Model, self).__init__()
        if gnn == 'gcn':
            self.gnn = GCN(in_channels=d,
                            hidden_channels=args.hidden_channels,
                            out_channels=c,
                            num_layers=args.num_layers,
                            dropout=args.dropout,
                            use_bn=not args.no_bn)
        elif gnn == 'sage':
            self.gnn = SAGE(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout)
        elif gnn == 'gat':
            self.gnn = GAT(in_channels=d,
                           hidden_channels=args.hidden_channels,
                           out_channels=c,
                           num_layers=args.num_layers,
                           dropout=args.dropout,
                           heads=args.gat_heads)
        elif gnn == 'gpr':
            self.gnn = GPRGNN(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        dropout=args.dropout,
                        alpha=args.gpr_alpha,
                        )
        self.p = 0.2
        self.n = n
        self.device = device
        self.args = args

        self.gl = Graph_Editer(args.K, n, device)

    def reset_parameters(self):
        self.gnn.reset_parameters()
        if hasattr(self, 'graph_est'):
            self.gl.reset_parameters()

    def forward(self, data, criterion):
        Loss, Log_p = [], 0
        x, y = data.graph['node_feat'].to(self.device), data.label.to(self.device)
        edge_index = data.graph['edge_index'].to(self.device)
        for k in range(self.args.K):
            edge_index_k, log_p = self.gl(edge_index, self.n, self.args.num_sample, k)
            out = self.gnn(x, edge_index_k)
            loss = self.sup_loss(y, out, criterion)
            Loss.append(loss.view(-1))
            Log_p += log_p
        Loss = torch.cat(Loss, dim=0)
        Var, Mean = torch.var_mean(Loss)
        return Var, Mean, Log_p

    def inference(self, data):
        x = data.graph['node_feat'].to(self.device)
        edge_index = data.graph['edge_index'].to(self.device)
        out = self.gnn(x, edge_index)
        return out

    def sup_loss(self, y, pred, criterion):
        if self.args.rocauc or self.args.dataset in ('twitch-e', 'fb100', 'elliptic'):
            if y.shape[1] == 1:
                true_label = F.one_hot(y, y.max() + 1).squeeze(1)
            else:
                true_label = y
            loss = criterion(pred, true_label.squeeze(1).to(torch.float))
        else:
            out = F.log_softmax(pred, dim=1)
            target = y.squeeze(1)
            loss = criterion(out, target)
        return loss