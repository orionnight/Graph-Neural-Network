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
from utils import load_data, accuracy
from models import GCN


# Training settings
parser = argparse.ArgumentParser()

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
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



#--------------------------Functions to get the gradients, dice--------------------------

grad_out1=None
def get_lay_grad_1(self, grad_input, grad_output):
    global grad_out1
    grad_out1 = grad_output[0]
    
grad_out2=None
def get_lay_grad_2(self, grad_input, grad_output):
    global grad_out2
    grad_out2 = grad_output[0]


grad_out3=None
def get_lay_grad_3(self, grad_input, grad_output):
    global grad_out3
    grad_out3 = grad_output[0]    

grad_out4=None
def get_lay_grad_4(self, grad_input, grad_output):
    global grad_out4
    grad_out4 = grad_output[0]   

#-------------------------- Load Model and optimizer --------------------------

path = './wht/'


wht_file = [f for f in listdir(path) if isfile(join(path, f))]

model = torch.load(path + wht_file[0])
optimizer = optim.Adam(model.parameters(), lr=args.lr)
print(sum(p.numel() for p in model.parameters()))

