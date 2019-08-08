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

        self.weight = Parameter(torch.FloatTensor(in_features, out_features, ker_size))
        self.bias = Parameter(torch.FloatTensor(out_features))
        self.mu = Parameter(torch.FloatTensor(emb_size, ker_size))
        self.sig = Parameter(torch.FloatTensor(1, ker_size))

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_(0, 0.1)
        self.mu.data.normal_(0, 0.005)
        self.sig.data = self.sig.data.zero_() + 60140
        self.bias.data.normal_(0, 0.1)

    def forward(self, data):
        output = torch.autograd.Variable(torch.tensor((), dtype=torch.float32)).cuda()
        output = output.new_zeros((data._x.shape[0], self.out_features, self.ker_size))
        domain = data._x[:, :3]
        diff = domain[data._edge_idx[0, :], :] - domain[data._edge_idx[1, :], :]
        shape = torch.Size((data._x.shape[0], data._x.shape[0]))
        # index = data.edge_idx.requires_grad_(True)

        for k in range(self.ker_size):
            support = torch.mm(data._x, self.weight[:, :, k])
            value = torch.exp(self.sig[:, k] *
                              (-0.5 * torch.sum((diff - self.mu[:, k].expand_as(diff)).pow(2),
                                                dim=1)))
            temp = torch.sparse.FloatTensor(data._edge_idx, value, shape)
            output[:, :, k] = torch.sparse.mm(temp, support)
        return output.sum(2) + self.bias.expand_as(output.mean(2))

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
