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

def dice_loss(output, target, wht):

    eps = 0.0001

    output = F.softmax(output)
    
    intersection = output * target
    union = output + target

    numerator = 2 * intersection.sum(0)
    denominator = union.sum(0) + eps
    loss = (1 - (numerator / denominator))
    #wht = (labels.sum()/labels.sum(0)).type(torch.cuda.FloatTensor) 
    #pdb.set_trace()
    ret_loss = wht * loss
    #pdb.set_trace()
    #print(1-loss)
    return ret_loss.sum(), loss.sum(), wht

def test(epoch,tes, wht):

    model.eval()

    output = model(features, E1, E2, Diff)
    gt = torch.max(labels, 1)[1]

    wht_loss, loss, wht = dice_loss(output,labels, wht)
    loss_test = wht_loss + F.cross_entropy(output,gt,weight = wht.data)
    
    acc_test1 = 100 * (1-loss/32)
    acc_test2 = accuracy(output, gt)

    return loss_test.data[0], acc_test1.data[0], acc_test2.data[0]    


#-------------------------- Load Model and optimizer --------------------------

path_train = '../../pyGCN_trai_norm'
path_valid = '../../py_valid'
path_test  = '../../pytor_test'


file_train= [f for f in listdir(path_train) if isfile(join(path_train, f))]
file_valid= [f for f in listdir(path_valid) if isfile(join(path_valid, f))]
file_test = [f for f in listdir(path_test) if isfile(join(path_test, f))]



path = './wht/'


wht_file = [f for f in listdir(path) if isfile(join(path, f))]




wht = Variable(torch.FloatTensor(np.array([ 16.97378   ,  85.77480065,  32.16534025,  43.73894593,
       232.04827732,  31.40006501,  20.5252747 ,  26.5694486 ,
        99.08358305,  17.44059628,  32.52226579,  29.0925319 ,
        65.7419286 ,  21.93470876, 136.3009185 ,  54.85942554,
        60.93557943, 126.48053398,  60.71517654,  67.2784416 ,
        18.68115005,  69.16945821,  18.3112146 ,  24.79957415,
        80.78001951,  21.33177316,  11.76718719,  20.24304891,
        18.4545681 ,  24.17785204, 184.53911415,  47.92705005])).cuda())

emb_size=3
ker_size=1
learning_rate_mu = 0.000000001;
learning_rate_si = 1000
initial_dic = 30
ifsaveall = 1




model = torch.load(path + wht_file[-1])
optimizer = optim.Adam(model.parameters(), lr=args.lr)
print(sum(p.numel() for p in model.parameters()))


mn_time=[]
mn_ac1 = []
mn_ac2 = []


for tes in range(len(file_test)):

	t = time.time()
	E1, E2, Diff, features, labels = load_data(path_test + '/' + file_test[tes])
	loss, acc1, acc2 = test(0,tes, wht)

	print('Time: {:.4f}s'.format(time.time() - t))
	mn_time.append(time.time() - t)
print('Mean Time: {:.4f}s'.format(np.mean(mn_time[1:])))
pdb.set_trace()