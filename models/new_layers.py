import torch
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from scipy.sparse import coo_matrix
import numpy.matlib
from torch.autograd import Variable


class SparseMM(torch.autograd.Function):
    """
    Sparse x dense matrix multiplication with autograd support.

    Implementation by Soumith Chintala:
    https://discuss.pytorch.org/t/
    does-pytorch-support-autograd-on-sparse-matrix/6156/7
    """

    def forward(self, matrix1, matrix2):
        self.save_for_backward(matrix1, matrix2)
        return torch.mm(matrix1, matrix2)

    def backward(self, grad_output):
        matrix1, matrix2 = self.saved_tensors
        grad_matrix1 = grad_matrix2 = None
        # pdb.set_trace()
        if self.needs_input_grad[0]:
            grad_matrix1 = None

        if self.needs_input_grad[1]:
            grad_matrix2 = torch.mm(matrix1.t(), grad_output)

        return grad_matrix1, grad_matrix2


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, ker_size, bias=True):
        super(GraphConvolution, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.ker_size = ker_size

        self.weight = Parameter(torch.FloatTensor(in_features, out_features, self.ker_size))
        self.bias = Parameter(torch.FloatTensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_(0, 0.1)
        self.bias.data.normal_(0, 0.1)

    def forward(self, data, TT):
        output = torch.tensor((), dtype=torch.float32).cuda()
        output = output.new_zeros((data._x.shape[0], self.out_features, self.ker_size))

        shape = torch.Size((data._x.shape[0], data._x.shape[0]))
        # index = data.edge_idx.requires_grad_(True)

        for k in range(self.ker_size):
            self.support = torch.mm(data._x, self.weight[:, :, k])
            temp = torch.sparse.FloatTensor(data._edge_idx, TT[:, k], shape)
            output[:, :, k] = SparseMM()(temp, self.support)
        return output.sum(2) + self.bias


class CreateKernal(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, emb_size, ker_size, bias=True):
        super(CreateKernal, self).__init__()

        self.emb_size = emb_size
        self.ker_size = ker_size

    def forward(self, diff, mu, sig):
        temp = torch.FloatTensor(diff.shape[0], self.ker_size).zero_().cuda()

        for k in range(self.ker_size):
            DD = diff - mu[:, k].expand_as(diff)
            QQ = -0.5 * torch.sum(DD.pow(2), dim=1)
            temp[:, k] = torch.exp(sig[:, k] * QQ)

        return temp

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.emb_size) + ' -> ' \
               + str(self.ker_size) + ')'
