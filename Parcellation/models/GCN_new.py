import torch.nn as nn
from old_code.layers import GraphConvolution, CreateKernal
import torch


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, emb_size, ker_size):
        super(GCN, self).__init__()

        self.gc1_0 = CreateKernal(emb_size, ker_size)
        self.gc1_1 = GraphConvolution(4, 256, ker_size)
        self.gc1_2 = nn.LeakyReLU(0.01)

        self.gc2_0 = CreateKernal(emb_size, ker_size)
        self.gc2_1 = GraphConvolution(4 + 256, 128, ker_size)
        self.gc2_2 = nn.LeakyReLU(0.01)

        self.gc3_0 = CreateKernal(emb_size, ker_size)
        self.gc3_1 = GraphConvolution(4 + 256 + 128, 64, ker_size)
        self.gc3_2 = nn.LeakyReLU(0.01)

        self.gc4_0 = CreateKernal(emb_size, ker_size)
        self.gc4_1 = GraphConvolution(4 + 256 + 128 + 64, 32, ker_size)
        self.gc4_2 = nn.LeakyReLU(0.01)

    def forward(self, data):

        x1 = self.gc1_0(diff, mu, sig)
        x1 = self.gc1_1(data, x1)
        x1 = self.gc1_2(x1)
        data._x = torch.cat((x1, data._x), 1)

        x2 = self.gc2_0(diff)
        x2 = self.gc2_1(data, x2)
        x2 = self.gc2_2(x2)
        data._x = torch.cat((x2, data._x), 1)

        x3 = self.gc3_0(diff)
        x3 = self.gc3_1(data, x3)
        x3 = self.gc3_2(x3)
        data._x = torch.cat((x3, data._x), 1)

        x4 = self.gc4_0(diff)
        x4 = self.gc4_1(data, x4)
        x4 = self.gc4_2(x4)

        return x4
