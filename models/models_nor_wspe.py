import torch
import torch.nn.functional as F
from torch_geometric.nn import max_pool, graclus, dense_diff_pool, SplineConv
from torch_geometric.utils import normalized_cut
import pdb
import torch.nn as nn
from torch_geometric.utils import degree
from torch_sparse import spmm
from model_dense import DenseSAGEConv_my
from torch_sparse import coalesce


def dense_to_sparse(tensor):
    index = tensor.nonzero()
    value = tensor.view(tensor.size(0)*tensor.size(0))
    
    index = index.t().contiguous()
    index, value = coalesce(index, value, tensor.size(0), tensor.size(1))
    return index, value

def norm(x, edge_index):
    deg = degree(edge_index[0], x.size(0), x.dtype, x.device) + 1
    return x / deg.unsqueeze(-1)
def diff_pool(x, s, edge_index, edge_attr):
    """Differentiable Pooling Operator based on dense learned assignments
    :math:`S` with

    .. math::
        \begin{align}
        F^{(l+1)} &= S^{(l)} X^{(l)}\\
        A^{(l+1)} &= {S^{(l)}}^{\top}A^{(l)} S^{(l)}
        \end{align}

    from the `"Hierarchical Graph Representation Learning with Differentiable
    Pooling" <https://arxiv.org/abs/1806.08804>`_ paper.
    """


    num_nodes = s.size(0)

    s = torch.softmax(s, dim=-1)
    s_t = s.transpose(0,1)


    adj = torch.matmul(s_t,spmm(edge_index, edge_attr, num_nodes, s))
    out = torch.matmul(s_t, x)

    #pdb.set_trace()

    row, col = edge_index
    diag_ent = edge_attr.clone()

    
    #pdb.set_trace()
    ind_dia = (row==col).nonzero()

    #pdb.set_trace()
    diag_ent[ind_dia]=0

    one_mat = torch.ones([num_nodes,1]).cuda()
    row_sum = spmm(edge_index, diag_ent, num_nodes, one_mat) 

    diag_ent = diag_ent * -1
    diag_ent[ind_dia]=row_sum

    adj_temp = diag_ent
    
    adj_reg = torch.trace(torch.matmul(s_t,spmm(edge_index, adj_temp, num_nodes, s)))


    return out, adj, adj_reg


class GNN(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 lin=False,
                 norm_embed=True):
        super(GNN, self).__init__()

        self.conv1 = SplineConv(in_channels, hidden_channels, dim=1, kernel_size=1)        
        #self.conv2 = SplineConv(hidden_channels, hidden_channels, dim=1, kernel_size=1)
        

        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        #pdb.set_trace()
        num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, 'bn{}'.format(i))(x)
        x = x.view(num_nodes, num_channels)
        return x

    def forward(self, x, edge_index, edge_attr):

        x0 = x
        x = F.relu(self.conv1(x0, edge_index, edge_attr))
        #x2 = F.relu(self.conv2(x1, edge_index, edge_attr))

        #x = torch.cat([x1, x2], dim=-1)

        if self.lin is not None:
            x = F.relu(self.lin(x))

        return x


class GCNet(torch.nn.Module):
    def __init__(self):
        super(GCNet, self).__init__()

        self.gnn1_pool = GNN(2, 16, 8, lin=False)
        self.gnn1_embed = GNN(2, 8, 8, lin=False)

        self.gnn2_pool = GNN(8, 1, 1, lin=False)
        self.gnn2_embed = GNN(8, 16, 32, lin=False)

        self.lin1 = torch.nn.Linear(16, 8)
        self.lin2 = torch.nn.Linear(8, 1)
        


    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr.squeeze()

        x = x[:,3:5]
        s = self.gnn1_pool(x, edge_index, edge_attr.unsqueeze(1))
        x = self.gnn1_embed(x, edge_index, edge_attr)

        x, adj, reg1 = diff_pool(x, s, edge_index, edge_attr)
        idx,val = dense_to_sparse(adj)

        s2 = self.gnn2_pool(x, idx,val)
        x  = self.gnn2_embed(x, idx,val)

        x, adj, reg2 = diff_pool(x, s2, idx, val)

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        
        #pdb.set_trace()
        reg = reg1+reg2
        return x, reg