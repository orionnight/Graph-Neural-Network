import os
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset
from inputs.graph_sample import GraphSample
from sklearn.neighbors import NearestNeighbors
import numpy as np

import pdb


# Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")


def make_dataset(root, mode):
    assert mode in ['train', 'val', 'test']
    items = []
    if mode == 'train':
        files_list = os.listdir(os.path.join(root, 'train'))
        files_list.sort()

        for it in range(len(files_list)):
            item = os.path.join(os.path.join(root, 'train'), files_list[it])
            items.append(item)

    elif mode == 'val':
        files_list = os.listdir(os.path.join(root, 'valid'))
        files_list.sort()

        for it in range(len(files_list)):
            item = os.path.join(os.path.join(root, 'valid'), files_list[it])
            items.append(item)
    else:

        files_list = os.listdir(os.path.join(root, 'test'))
        files_list.sort()

        for it in range(len(files_list)):
            item = os.path.join(os.path.join(root, 'test'), files_list[it])
            items.append(item)

    return items


class GeometricDataset(Dataset):

    def __init__(self, mode, root_dir):
        """
        Args:
            mode: 'train', 'valid', or 'test'
            root_dir: path to the dataset
        """
        self.root_dir = root_dir
        self.files = make_dataset(root_dir, mode)
        self.mode = mode

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        files_path = self.files[index]
        temp = torch.load(files_path)

        # Enable this if you want to add Cortical thickness (X: 3 spectral coordinates + C: Sulcul depth + T: Cotical thickness)
        # x = torch.cat((torch.FloatTensor(temp['X'][:, 0:3]),
        #                torch.FloatTensor(temp['C']),
        #                torch.FloatTensor(temp['T'])), 1)
        #


        x = torch.cat((torch.FloatTensor(temp['X'][:, 0:3]),
                       torch.FloatTensor(temp['C'])), 1)
        e1, e2, e3 = sp.find(temp['A'])
        edge_idx = torch.cat((torch.LongTensor(e1).unsqueeze(0), torch.LongTensor(e2).unsqueeze(0)),
                             0)  # index x,y of the sparse adj matrix
        edge_wht = torch.LongTensor(e3).unsqueeze(
            0)  # the weights of the edges. Often used to construct the sparse matrix (adj)
        gt = torch.LongTensor(temp['GT'])  # This is manual label
        xyz = torch.FloatTensor(temp['EUC'])  # This is the xyz of the mesh node location in euclidean coordinates
        face = torch.FloatTensor(temp['F'])  # This is the face of the mesh triangualtion
        age = torch.FloatTensor(temp['AG'][0])  # This is the age of the subject
        sx = torch.FloatTensor(temp['SX'])  # This is the gender of the subject
        lab = torch.FloatTensor(temp['Y'])  # This is FreeSurfer label

        data = GraphSample(x=x, edge_idx=edge_idx, edge_wht=edge_wht, gt=gt, xyz=xyz, face=face, age=age, sx=sx,
                           lab=lab)
        if self.mode == 'train':
        # Karthik is writing this part to sub-sample the graph/.. !!! Incomplete warning !!!
            reduce_graph = 'False'

            if reduce_graph == 'True':
                y = torch.max(gt, 1)[1]
                all_ind = []
                for id in range(gt.size(1)):
                    aa = (y == id).nonzero()
                    ap = torch.randperm(torch.min(torch.sum(gt, 0)))
                    all_ind.append(aa[ap])
                flat_list = [item for sublist in all_ind for item in sublist]
                flattened_tensor = torch.LongTensor(flat_list)
                sm_x = x[flattened_tensor, :]
                nbrs = NearestNeighbors(n_neighbors=15, algorithm='ball_tree').fit(sm_x)
                adj = sp.csr_matrix(nbrs.kneighbors_graph(sm_x).toarray())
                e1, e2, e3 = sp.find(adj)
                edge_idx = torch.cat((torch.LongTensor(e1).unsqueeze(0), torch.LongTensor(e2).unsqueeze(0)), 0)
                edge_wht = torch.LongTensor(e3).unsqueeze(0)
                sm_gt = gt[flattened_tensor, :]
                sm_xyz = xyz[flattened_tensor, :]
                sm_face = face[flattened_tensor, :]
                sm_lab = lab#[flattened_tensor, :]

                data = GraphSample(x=sm_x, edge_idx=edge_idx, edge_wht=edge_wht, gt=sm_gt, xyz=sm_xyz, face=sm_face,
                                   age=age, sx=sx,
                                   lab=sm_lab)

        return data
