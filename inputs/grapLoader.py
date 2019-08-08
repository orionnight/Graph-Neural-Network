import os
import torch
from torch.utils.data import Dataset

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
    """Face Landmarks dataset."""

    def __init__(self, mode, root_dir):
        """
        Args:
            mode: 'train', 'valid', or 'test'
            root_dir: path to the dataset
        """
        self.root_dir = root_dir
        self.files = make_dataset(root_dir, mode)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        files_path = self.files[index]        
        temp = torch.load(files_path)

        pdb.set_trace(quit)

        features = torch.FloatTensor(np.concatenate([temp['small_X'][:,0:3],temp['small_C'],temp['small_T']],axis=1))
        labels = torch.FloatTensor(temp['small_GT'])
        ag = torch.FloatTensor([temp['AG'][0]])
        sx = torch.FloatTensor(temp['SX'])
        #pdb.set_trace()
        E1,E2,E3 = sp.find(temp['small_A'])
        
        E1 = torch.FloatTensor(E1).unsqueeze(0)
        E2 = torch.FloatTensor(E2).unsqueeze(0)
        E3 = torch.FloatTensor(E3).unsqueeze(0) 
        edges = torch.cat((E1, E2), 0).long()

        data = Graph_Sample(x=features, edge_index=edges, y= sx, edge_attr= E3, pos = labels)


    

        return data
