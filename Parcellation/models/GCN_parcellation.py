import torch
import torch.nn as nn
from models.layers_ker import GraphConvolution




class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, emb_size, ker_size):
        super(GCN, self).__init__()

        self.gc1_1 = GraphConvolution(nfeat, nhid, emb_size, ker_size)
        self.gc1_2 = nn.LeakyReLU(0.01)

        nhid1 = torch.div(nhid, 2).item()
        self.gc2_1 = GraphConvolution(nfeat + nhid, nhid1, emb_size, ker_size)
        self.gc2_2 = nn.LeakyReLU(0.01)

        nhid2 = torch.div(nhid, 4).item()
        self.gc3_1 = GraphConvolution(nfeat + nhid + nhid1, nhid2, emb_size, ker_size)
        self.gc3_2 = nn.LeakyReLU(0.01)

        self.gc4_1 = GraphConvolution(nfeat + nhid + nhid1 + nhid2, nclass, emb_size, ker_size)
        self.gc4_2 = nn.LeakyReLU(0.01)

    def forward(self, data):
        x1 = self.gc1_1(data)
        x1 = self.gc1_2(x1)
        data._x = torch.cat((x1, data._x), 1)

        x2 = self.gc2_1(data)
        x2 = self.gc2_2(x2)
        data._x = torch.cat((x2, data._x), 1)

        x3 = self.gc3_1(data)
        x3 = self.gc3_2(x3)
        data._x = torch.cat((x3, data._x), 1)

        x4 = self.gc4_1(data)
        x4 = self.gc4_2(x4)

        return x4
