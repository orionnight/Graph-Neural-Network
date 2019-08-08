import torch
import torch.nn.functional as F
from torch_geometric.nn import max_pool_x, graclus, dense_diff_pool, SplineConv, max_pool, global_mean_pool
from torch_geometric.utils import normalized_cut
import pdb
import torch.nn as nn
from torch_geometric.utils import degree
from torch_sparse import spmm
from model_dense import DenseSAGEConv_my
from torch_sparse import coalesce
import torch_geometric.transforms as T
from pykeops.torch import LazyTensor
def dense_to_sparse(tensor):
    index = tensor.nonzero()
    value = tensor.view(tensor.size(0)*tensor.size(0))
    
    index = index.t().contiguous()
    index, value = coalesce(index, value, tensor.size(0), tensor.size(1))
    return index, value
def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))

def norm(x, edge_index):
    deg = degree(edge_index[0], x.size(0), x.dtype, x.device) + 1
    return x / deg.unsqueeze(-1)
    
def KMeans(x, K, Niter=10, verbose=False):
    N, D = x.shape  # Number of samples, dimension of the ambient space

    # K-means loop:
    # - x  is the point cloud,
    # - cl is the vector of class labels
    # - c  is the cloud of cluster centroids
    
    c = x[:K, :].clone()  # Simplistic random initialization
    x_i = LazyTensor( x[:,None,:] )  # (Npoints, 1, D)

    for i in range(Niter):

        c_j = LazyTensor( c[None,:,:] )  # (1, Nclusters, D)
        D_ij = ((x_i - c_j)**2).sum(-1)  # (Npoints, Nclusters) symbolic matrix of squared distances
        cl  = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        Ncl = torch.bincount(cl).type(torchtype[dtype])  # Class weights
        for d in range(D):  # Compute the cluster centroids with torch.bincount:
            c[:, d] = torch.bincount(cl, weights=x[:, d]) / Ncl

    

    if verbose:
        print("K-means example with {:,} points in dimension {:,}, K = {:,}:".format(N, D, K))
        print('Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n'.format(
                Niter, end - start, Niter, (end-start) / Niter))

    return cl, c



class GNN(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_embed=True):
        super(GNN, self).__init__()

        self.conv1 = SplineConv(in_channels, out_channels, dim=1, kernel_size=1)        
        #self.conv2 = SplineConv(hidden_channels, hidden_channels, dim=1, kernel_size=1)
        


    def forward(self, x, edge_index, edge_attr):

        x0 = x
        x = F.relu(self.conv1(x0, edge_index, edge_attr))
        #x2 = F.relu(self.conv2(x1, edge_index, edge_attr))

        #x = torch.cat([x1, x2], dim=-1)

        return x


class GCNet(torch.nn.Module):
    def __init__(self):
        super(GCNet, self).__init__()

        self.gnn1_pool = GNN(5, 16)
        self.m1 = nn.LeakyReLU(0.1)
        self.gnn2_pool = GNN(16, 32)
        self.m2 = nn.LeakyReLU(0.1)


        self.lin1 = torch.nn.Linear(32, 8)
        self.m3 = nn.LeakyReLU(0.1)
        self.lin2 = torch.nn.Linear(8, 2)
        #self.m4 = nn.Sigmoid()
        


    def forward(self, data):
        x0, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr.squeeze()


        data.x = self.gnn1_pool(x0, edge_index, edge_attr)
        data.x = self.m1(data.x)
        #pdb.set_trace()

        cl, c = KMeans(data.x, 32)
        pdb.set_trace()
        weight = normalized_cut_2d(data.edge_index, x0[:,0].unsqueeze(1))
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data.edge_attr = None
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        #pdb.set_trace()
        data.x = self.gnn2_pool(data.x, data.edge_index, edge_attr)
        data.x = self.m2(data.x)

        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        x3, batch = max_pool_x(cluster, data.x, data.batch)

        x3 = global_mean_pool(x3, batch)
        
        #print(x3)

        x4 = self.lin1(x3)
        x4 = self.m3(x4)

        x = self.lin2(x4)
        #x = self.m4(x)

        #pdb.set_trace()

        return x[0]