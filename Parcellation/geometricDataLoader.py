#!/usr/env/bin python3.6

import os
from random import random
from torch.utils.data import Dataset
from torch_geometric.data import Data
import pdb
import torch
# Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")


def make_dataset(root, mode):
    assert mode in ['train', 'val', 'test']
    items = []
    if mode == 'train':
        train_path = os.path.join(root, 'train')

        files_list = os.listdir(train_path)

        files_list.sort()
        
        for it in range(len(files_list)):
            item = os.path.join(train_path, files_list[it])
            items.append(item)

    elif mode == 'val':
        train_path = os.path.join(root, 'valid')

        files_list = os.listdir(train_path)

        files_list.sort()

        for it in range(len(files_list)):
            item = os.path.join(train_path, files_list[it])
            items.append(item)
    else:
        train_path = os.path.join(root, 'test')

        files_list = os.listdir(train_path)

        files_list.sort()

        for it in range(len(files_list)):
            item = os.path.join(train_path, files_list[it])
            items.append(item)

    return items


class GeometricImageDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, mode, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.files = make_dataset(root_dir, mode)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        files_path = self.files[index]
        # print("{} and {}".format(img_path,mask_path))
        data = torch.load(files_path)  # .convert('RGB')
    

        return data
