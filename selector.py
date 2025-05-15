import torch
import numpy as np
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, Dropout
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CollisionNet(torch.nn.Module):
    def __init__(self, config_size, embed_size, obs_size, use_obstacles=True):
        super(CollisionNet, self).__init__()

        self.config_size = config_size
        self.embed_size = embed_size
        self.obs_size = obs_size
        self.use_obstacles = use_obstacles

        self.hx = Seq(Lin(config_size, embed_size),
                      ReLU(),
                      Lin(embed_size, embed_size))
        self.hy = Seq(Lin(config_size * 3, embed_size),
                      ReLU(),
                      Lin(embed_size, embed_size))

        self.obs_node_code = Seq(Lin(obs_size, embed_size), ReLU(), Lin(embed_size, embed_size))
        self.obs_edge_code = Seq(Lin(obs_size, embed_size), ReLU(), Lin(embed_size, embed_size))
        self.node_attentions = torch.nn.ModuleList([Block(embed_size) for _ in range(3)])
        self.edge_attentions = torch.nn.ModuleList([Block(embed_size) for _ in range(3)])

        self.fy = Seq(Lin(embed_size * 3, embed_size), ReLU(),
                      Lin(embed_size, embed_size), ReLU())

        self.feta2 = Seq(Lin(embed_size, embed_size), ReLU(), Dropout(),
                         Lin(embed_size, 64), ReLU(), Dropout(),
                         Lin(64, 32), ReLU(),
                         Lin(32, 2), Sigmoid()
                         )

        self.gate = Lin(embed_size * 2, embed_size)

        self.edge_fusion = Seq(Lin(embed_size * 2, embed_size), ReLU(),
                               Lin(embed_size, embed_size), ReLU())


    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()
        for op in self.ops:
            op.reset_parameters()
        self.node_feature.reset_parameters()
        self.edge_feature.reset_parameters()

    def forward(self, v, edge_index, obstacles):

        x = self.hx(v)
        vi, vj = v[edge_index[0, :]], v[edge_index[1, :]]
        y = self.hy(torch.cat((vj - vi, vj, vi), dim=-1))

        if self.use_obstacles:
            obs_node_code = self.obs_node_code(obstacles.view(-1, self.obs_size))
            obs_edge_code = self.obs_edge_code(obstacles.view(-1, self.obs_size))
            for na, ea in zip(self.node_attentions, self.edge_attentions):
                x = na(x, obs_node_code)
                y = ea(y, obs_edge_code)
        xi, xj = x[edge_index[0, :]], x[edge_index[1, :]]
        transformed_feature = self.fy(torch.cat((xj - xi, xj, xi), dim=-1))
        gate_weight = torch.sigmoid(
            self.gate(torch.cat((transformed_feature, y), dim=-1)))
        edge_f = gate_weight * transformed_feature + (1 - gate_weight) * y
        edge_f = self.edge_fusion(torch.cat((edge_f, y), dim=-1))
        connection = self.feta2(edge_f)

        return connection

class Attention(torch.nn.Module):

    def __init__(self, embed_size, temperature):
        super(Attention, self).__init__()
        self.temperature = temperature
        self.embed_size = embed_size
        self.key = Lin(embed_size, embed_size, bias=False)
        self.query = Lin(embed_size, embed_size, bias=False)
        self.value = Lin(embed_size, embed_size, bias=False)
        self.glu_fusion = torch.nn.GLU(dim=-1)
        self.output_projection = Lin(32, embed_size)
        self.layer_norm = torch.nn.LayerNorm(embed_size, eps=1e-6)

    def forward(self, map_code, obs_code):
        map_value = self.value(map_code)
        obs_value = self.value(obs_code)

        map_query = self.query(map_code)
        map_key = self.key(map_code)
        obs_key = self.key(obs_code)

        obs_attention = (map_query @ obs_key.T)
        self_attention = (map_query.reshape(-1) * map_key.reshape(-1)).reshape(-1, self.embed_size).sum(dim=-1)

        whole_attention = torch.cat((self_attention.unsqueeze(-1), obs_attention), dim=-1)
        whole_attention = (whole_attention / self.temperature).softmax(dim=-1)
        obs_value_expanded = obs_value.unsqueeze(0).repeat(len(map_code), 1, 1)
        glu_input = torch.cat((map_value.unsqueeze(1), obs_value_expanded), dim=1)
        gated_output = self.glu_fusion(glu_input)
        map_code_new = (whole_attention.unsqueeze(-1) * gated_output).sum(dim=1)
        map_code_new = self.output_projection(map_code_new)

        return self.layer_norm(map_code_new + map_code)



class FeedForward(torch.nn.Module):

    def __init__(self, d_in, d_hid):
        super(FeedForward, self).__init__()
        self.w_1 = Lin(d_in, d_hid)
        self.w_2 = Lin(d_hid, d_in)
        self.layer_norm = torch.nn.LayerNorm(d_in, eps=1e-6)

    def forward(self, x):
        residual = x
        x = self.w_2((self.w_1(x)).relu())
        x += residual
        x = self.layer_norm(x)
        return x


class Block(torch.nn.Module):

    def __init__(self, embed_size):
        super(Block, self).__init__()
        self.attention = Attention(embed_size, embed_size ** 0.5)
        self.map_feed = FeedForward(embed_size, embed_size)

    def forward(self, map_code, obs_code):
        map_code = self.attention(map_code, obs_code)
        map_code = self.map_feed(map_code)

        return map_code


