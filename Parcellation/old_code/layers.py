import math
import pdb
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
        #pdb.set_trace()
        if self.needs_input_grad[0]:
            grad_matrix1 = grad_output

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
        self.bias   = Parameter(torch.FloatTensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):

        self.weight.data.normal_(0, 0.1)
        self.bias.data.normal_(0, 0.1)

    def forward(self, data, TT):
       
        self.inp = data._x
        self.shape = torch.Size((self.inp.shape[0],self.inp.shape[0]))
        self.indices = torch.autograd.Variable(data._edge_idx)


        output = torch.autograd.Variable(torch.FloatTensor(self.inp.shape[0], self.out_features, self.ker_size).zero_(),requires_grad=True).cuda()

        for k in range(self.ker_size):
            #pdb.set_trace()

            self.support = torch.mm(self.inp, self.weight[:,:,k])
            #pdb.set_trace()

            T = torch.autograd.Variable(torch.sparse.FloatTensor(self.indices.cpu(),TT[:,k].data.cpu(),self.shape)).cuda()
            
            output[:,:,k] = SparseMM()(T,self.support)


        return output.sum(2) + self.bias



class CreateKernal(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, emb_size, ker_size, bias=True):
        super(CreateKernal, self).__init__()


        self.emb_size = emb_size
        self.ker_size = ker_size

        self.mu     = Parameter(torch.FloatTensor(self.emb_size, self.ker_size))
        self.sig    = Parameter(torch.FloatTensor(1, self.ker_size))

        self.reset_parameters()

    def reset_parameters(self):

        self.mu.data.normal_(0, 0.005)
        self.sig.data = self.sig.data.zero_() + 60140

    def forward(self, diff):

        T = torch.FloatTensor(diff.shape[0],self.ker_size).zero_().cuda()

        for k in range(self.ker_size):

            DD = diff - self.mu[:,k].expand_as(diff)
            self.QQ = -0.5*torch.sum(DD.pow(2), dim=1)
            T[:,k] = torch.exp(self.sig[:,k]*self.QQ).data
        
        self.T = torch.autograd.Variable(T,requires_grad=True)
        return self.T



def __repr__(self):
    return self.__class__.__name__ + ' (' \
           + str(self.in_features) + ' -> ' \
           + str(self.out_features) + ')'



