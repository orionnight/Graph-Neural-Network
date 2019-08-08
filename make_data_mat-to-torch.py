from __future__ import division
from __future__ import print_function
import time
import argparse
import numpy as np
import pdb
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from os import listdir
from os.path import isfile, join
from utils.utils import load_data, accuracy

import scipy.io as sio
from tqdm import tqdm


# Training settings
parser = argparse.ArgumentParser()

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1500,
                    help='Number of epochs to train.')
parser.add_argument('--lrgcn', type=float, default=0.1,
                    help='Initial learning rate.')
parser.add_argument('--lr', type=float, default=0.1,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)




#------------------Initializations-----------------------------------------------

wht = Variable(torch.FloatTensor(np.array([ 3.45, 2.05, 4.48])).cuda())

emb_size=3
ker_size=6
learning_rate_mu = 0.000000001;
learning_rate_si = 1000
initial_dic = 3
ifsaveall = 1

#-------------------------- Load data   ----------------------------------------------------
path_train = './data/mindboggle_reduced/train'
path_valid = './data/mindboggle_reduced/valid'
path_test  = './data/mindboggle_reduced/test'

path_proc_train = './data/mindboggle_red_pyt/train'
path_proc_valid = './data/mindboggle_red_pyt/valid'
path_proc_test  = './data/mindboggle_red_pyt/test'


file_train= [f for f in listdir(path_train) if isfile(join(path_train, f))]
file_valid= [f for f in listdir(path_valid) if isfile(join(path_valid, f))]
file_test = [f for f in listdir(path_test) if isfile(join(path_test, f))]


t_total = time.time()


t = time.time()

idx = np.arange(len(file_train))
np.random.shuffle(idx)


for tra in tqdm(range(len(idx))):

    data = load_data(path_train + '/' + file_train[idx[tra]])
    #pdb.set_trace()
    if data.y.item()==0:
        #pdb.set_trace()
        temp = data.y[0]
        data.y = torch.FloatTensor([1,0])
    else:
        #pdb.set_trace()
        temp = data.y[0]
        data.y = torch.FloatTensor([0,1])
    torch.save(data, path_proc_train + '/' + file_train[idx[tra]][:-4])
    #pass




idx = np.arange(len(file_valid))
np.random.shuffle(idx)

for tra in tqdm(range(len(idx))):

    data = load_data(path_valid + '/' + file_valid[idx[tra]])
    if data.y.item()==0:
        #pdb.set_trace()
        temp = data.y[0]
        data.y = torch.FloatTensor([1,0])
    else:
        #pdb.set_trace()
        temp = data.y[0]
        data.y = torch.FloatTensor([0,1])
    #if torch.max(data.pos,1)[1].item()!=2:
    #temp = data.y[0]
    #    data.y = data.pos[:,0:2]
    torch.save(data, path_proc_valid + '/' + file_valid[idx[tra]][:-4])
    #pass




idx = np.arange(len(file_test))
np.random.shuffle(idx)

for tra in tqdm(range(len(idx))):

    data = load_data(path_test + '/' + file_test[idx[tra]])
    if data.y.item()==0:
        #pdb.set_trace()
        temp = data.y[0]
        data.y = torch.FloatTensor([1,0])
    else:
        #pdb.set_trace()
        temp = data.y[0]
        data.y = torch.FloatTensor([0,1])
    #if torch.max(data.pos,1)[1].item()!=2:
    #temp = data.y[0]
    #    data.y = data.pos[:,0:2]
    torch.save(data, path_proc_test + '/' + file_test[idx[tra]][:-4])
    #pass
