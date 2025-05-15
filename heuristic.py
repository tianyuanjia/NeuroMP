import torch
import numpy as np
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, Dropout
from torch_geometric.nn.conv import MessagePassing
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian, add_self_loops
from torch import svd_lowrank

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MPNN(MessagePassing):
    def __init__(self, embed_size, aggr: str = 'max', **kwargs):
        super(MPNN, self).__init__(aggr=aggr, **kwargs)
        self.fx = Seq(Lin(embed_size * 4, embed_size), ReLU(), Lin(embed_size, embed_size))

    def forward(self, x, edge_index, edge_attr):
        """"""
        out = self.propagate(edge_index, x=(x, x), edge_attr=edge_attr)
        return torch.max(x, out)

    def message(self, x_i, x_j, edge_attr):
        z = torch.cat([x_j - x_i, x_j, x_i, edge_attr], dim=-1)
        values = self.fx(z)
        return values

    def __repr__(self):
        return '{}({}, dim={})'.format(self.__class__.__name__, self.channels,
                                       self.dim)

class MP(MessagePassing):
    def __init__(self):
        super(MP, self).__init__()

    def message(self, x_j, norm=None):
        if norm != None:
            return norm.view(-1, 1) * x_j
        else:
            return x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K, self.temp.data.tolist())

class BasisGenerator(torch.nn.Module):
    '''
    generate all the feature spaces
    '''

    def __init__(self, K, low_lx=True, norm1=False):
        super(BasisGenerator, self).__init__()
        self.norm1 = norm1
        self.K = K  # for lx
        self.low_lx = low_lx
        self.mp = MP()

    def get_lx_basis(self, x, edge_index):
        # generate all feature spaces
        lxs = []
        num_nodes = x.shape[0]
        lap_edges, lap_norm = get_laplacian(edge_index=edge_index,
                                            normalization='sym',
                                            num_nodes=num_nodes)
        h = F.normalize(x, dim=1)
        lxs = [h]
        edges, norm = add_self_loops(edge_index=lap_edges,
                                     edge_attr=lap_norm,
                                     fill_value=-1.,
                                     num_nodes=num_nodes)

        for k in range(self.K):
            h = self.mp.propagate(edge_index=edges, x=h, norm=norm)
            h = h - lxs[-1]
            if self.norm1:
                h = F.normalize(h, dim=1)
            lxs.append(h)
        #
        normed_lxs = []
        low_lxs = []
        for lx in lxs:
            if self.low_lx:
                q = min(lx.size())
                U, S, V = svd_lowrank(lx, q=q)
                #U, S, V = svd_lowrank(lx)
                low_lx = torch.mm(U, torch.diag(S))
                low_lxs.append(low_lx)
                normed_lxs.append(F.normalize(low_lx, dim=1))
            else:
                normed_lxs.append(F.normalize(lx, dim=1))

        final_lx = [F.normalize(lx, dim=0) for lx in lxs]  # no norm1
        return final_lx


class Explorer(torch.nn.Module):
    def __init__(self, config_size, embed_size, obs_size):
        super(Explorer, self).__init__()

        self.config_size = config_size
        self.embed_size = embed_size
        self.obs_size = obs_size

        self.hx = Seq(Lin(config_size * 4 + 8, embed_size), ReLU(),
                      Lin(embed_size, embed_size))
        self.hy = Seq(Lin(config_size * 3 + 6, embed_size), ReLU(),
                      Lin(embed_size, embed_size))
        self.mpnn = MPNN(embed_size)
        self.fy = Seq(Lin(embed_size * 3, embed_size), ReLU(),
                      Lin(embed_size, embed_size))

        self.feta = Seq(Lin(embed_size, embed_size), ReLU(),  # Dropout(p=0.5),
                        Lin(embed_size, embed_size), ReLU(),  # Dropout(p=0.5),
                        Lin(embed_size, 1, bias=False))

        self.basis_generator = BasisGenerator(K=2)
        self.thetas = torch.nn.Parameter(torch.ones(3), requires_grad=True)
        self.lin_lx = Lin(config_size+2, embed_size)
        self.lin2 = Lin(embed_size, embed_size)


    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()
        for op in self.ops:
            op.reset_parameters()
        self.node_feature.reset_parameters()
        self.edge_feature.reset_parameters()

    def forward(self, v, edge_index, loop, labels):
        self.labels = labels
        v = torch.cat((v, labels), dim=-1)
        goal = v[labels[:, 1] == 1].view(1, -1)
        x = self.hx(torch.cat((v, goal.repeat(len(v), 1), v - goal, (v - goal) ** 2), dim=-1))
        dict_mat = x
        lx_basis = self.basis_generator.get_lx_basis(v, edge_index)[0:]
        lx_dict = 0

        for k in range(3):
            lx_b = self.lin_lx(lx_basis[k]).to(device) * self.thetas[k]
            lx_dict = lx_dict + lx_b
        dict_mat = dict_mat + lx_dict

        vi, vj = v[edge_index[0, :]], v[edge_index[1, :]]
        y = self.hy(torch.cat((vj - vi, vj, vi), dim=-1))
        x = self.lin2(dict_mat)

        # During training, we iterate x and y over a random number of loops between 1 and 10. Intuitively, taking
        # random loops encourages the GNN to converge faster, which also helps propagating the gradient.
        # During evaluation, the GNN explorer will output x and y after 10 loops. For loops larger than 10.
        for _ in range(loop):
            x = self.mpnn(x, edge_index, y)  # node updating q_i^l
            xi, xj = x[edge_index[0, :]], x[edge_index[1, :]]
            y = torch.max(y, self.fy(torch.cat((xj - xi, xj, xi), dim=-1)))

        heuristic = self.feta(x)
        return heuristic

