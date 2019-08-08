import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_sparse import spmm
import math


def uniform(size, tensor):
    stdv = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)


class DenseSAGEConv_my(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_embed=True,
                 bias=True):
        super(DenseSAGEConv_my, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_embed = norm_embed
        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)
        uniform(self.in_channels, self.bias)

    def forward(self, x,  edge_index, edge_attr):
        #x = x.unsqueeze(0) if x.dim() == 2 else x
        
        num_nodes = x.size(0)
        out = spmm(edge_index, edge_attr, num_nodes, x)
        #out = torch.matmul(adj, x)

        out = torch.matmul(out, self.weight)

        if self.bias is not None:
            out = out + self.bias

        if self.norm_embed:
            out = F.normalize(out, p=2, dim=-1)

        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)