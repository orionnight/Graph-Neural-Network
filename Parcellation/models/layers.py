import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer
    """

    def __init__(self, in_features, out_features, emb_size, ker_size, bias=True):
        super(GraphConvolution, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.emb_size = emb_size
        self.ker_size = ker_size

        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features))
        self.mu = Parameter(torch.FloatTensor(emb_size))
        self.sig = Parameter(torch.FloatTensor(1))

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_(0, 0.1)
        self.mu.data.normal_(0, 0.005)
        self.sig.data = self.sig.data.zero_() + 60140
        self.bias.data.normal_(0, 0.1)

    def forward(self, data):
        self.support = torch.mm(data._x, self.weight)

        domain = data._x[:, :3]
        diff = domain[data._edge_idx[0, :], :] - domain[data._edge_idx[1, :], :]
        shape = torch.Size((data._x.shape[0], data._x.shape[0]))
        value = torch.exp(self.sig * (-0.5 * torch.sum((diff - self.mu.expand_as(diff)).pow(2), dim=1)))
        #value = data._edge_wht
        #value = value.type(torch.cuda.FloatTensor).squeeze(0)
        adj = torch.sparse.FloatTensor(data._edge_idx, value, shape)
        output = torch.sparse.mm(adj, self.support)

        return output + self.bias

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
