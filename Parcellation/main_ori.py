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


def get_grad(grad_output, matrix1, matrix2,mu,sig,Diff, QQ, indices, inp):
    #pdb.set_trace()
    grad_matrix1 = torch.mul(grad_output,matrix2)
    dmu = torch.FloatTensor(3).cuda()
    dsig = torch.FloatTensor(1).cuda()    
    shape = torch.Size((inp.shape[0],inp.shape[0]))

    
    for d in range(3):
        #pdb.set_trace()
        values = sig * ((Diff[:,d] * matrix1) - (mu[d] * matrix1))
        F = torch.sparse.FloatTensor(indices.data, values.data.cpu(), shape).cuda()
        temp2 = torch.mm(F, matrix2.data)
        dmu[d] = (grad_output.data * temp2).sum()

    #pdb.set_trace()
    values = matrix1 * QQ
    F = torch.sparse.FloatTensor(indices.data, values.data.cpu(), shape).cuda()
    temp2 = torch.mm(F, matrix2.data)
    dsig[0] = (grad_output.data * temp2).sum()

    return dmu, dsig

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


#------------------Initializations-----------------------------------------------

wht = Variable(torch.FloatTensor(np.array([ 16.97378   ,  85.77480065,  32.16534025,  43.73894593,
       232.04827732,  31.40006501,  20.5252747 ,  26.5694486 ,
        99.08358305,  17.44059628,  32.52226579,  29.0925319 ,
        65.7419286 ,  21.93470876, 136.3009185 ,  54.85942554,
        60.93557943, 126.48053398,  60.71517654,  67.2784416 ,
        18.68115005,  69.16945821,  18.3112146 ,  24.79957415,
        80.78001951,  21.33177316,  11.76718719,  20.24304891,
        18.4545681 ,  24.17785204, 184.53911415,  47.92705005])).cuda())

emb_size=3
ker_size=6
learning_rate_mu = 0.000000001;
learning_rate_si = 1000
initial_dic = 86.55
ifsaveall = 1
# Dependent on the number of ker_size

dmu_1 = torch.autograd.Variable(torch.Tensor(emb_size,ker_size).zero_(),requires_grad=True).cuda()
dsi_1 = torch.autograd.Variable(torch.Tensor(1,ker_size).zero_(),requires_grad=True).cuda()

dmu_2 = torch.autograd.Variable(torch.Tensor(emb_size,ker_size).zero_(),requires_grad=True).cuda()
dsi_2 = torch.autograd.Variable(torch.Tensor(1,ker_size).zero_(),requires_grad=True).cuda()

dmu_3 = torch.autograd.Variable(torch.Tensor(emb_size,ker_size).zero_(),requires_grad=True).cuda()
dsi_3 = torch.autograd.Variable(torch.Tensor(1,ker_size).zero_(),requires_grad=True).cuda()

dmu_4 = torch.autograd.Variable(torch.Tensor(emb_size,ker_size).zero_(),requires_grad=True).cuda()
dsi_4 = torch.autograd.Variable(torch.Tensor(1,ker_size).zero_(),requires_grad=True).cuda()



#-------------------------- Load data   ----------------------------------------------------

path_train = '../../pyGCN_trai_norm'
path_valid = '../../py_valid'
path_test  = '../../pytor_test'


file_train= [f for f in listdir(path_train) if isfile(join(path_train, f))]
file_valid= [f for f in listdir(path_valid) if isfile(join(path_valid, f))]
file_test = [f for f in listdir(path_test) if isfile(join(path_test, f))]



#-------------------------- Create Model and optimizer --------------------------


#model = GCN(nfeat=4,nhid=args.hidden,nclass=32, emb_size=3, ker_size=ker_size)
#optimizer = optim.Adam(model.parameters(), lr=args.lr)
#model.cuda()


#-------------------------- Load Model and optimizer --------------------------

model = torch.load('./wht/weights_0')
optimizer = optim.Adam(model.parameters(), lr=args.lr)
#pdb.set_trace()

def test(epoch,tes, wht):

    model.eval()

    output = model(features, E1, E2, Diff)
    gt = torch.max(labels, 1)[1]

    wht_loss, loss, wht = dice_loss(output,labels, wht)
    loss_test = wht_loss + F.cross_entropy(output,gt,weight = wht.data)
    
    acc_test1 = 100 * (1-loss/32)
    acc_test2 = accuracy(output, gt)

    return loss_test.data[0], acc_test1.data[0], acc_test2.data[0]



def train(epoch,tra, wht):

    model.train()
    model.zero_grad()

    model.gc1_1.register_backward_hook(get_lay_grad_1)
    
    model.gc2_1.register_backward_hook(get_lay_grad_2)
    
    model.gc3_1.register_backward_hook(get_lay_grad_3)
    
    model.gc4_1.register_backward_hook(get_lay_grad_4)
    


    output = model(features, E1, E2, Diff)
    gt = torch.max(labels, 1)[1]

    wht_loss, loss, wht = dice_loss(output,labels, wht)
    loss_train =  wht_loss + F.cross_entropy(output,gt,weight = wht.data)
    acc_train1 = 100 * (1-loss/32)
    acc_train2 = accuracy(output, gt)

    # ----------------------------------- Backprop and update parameters -----------------------------------

    loss_train.backward()
    optimizer.step()


    for k in range(ker_size):

        dmu_1[:,k], dsi_1[:,k] = get_grad(grad_out1, model.gc1_0.T[:,k], model.gc1_1.support, model.gc1_0.mu[:,k], model.gc1_0.sig[:,k], Diff, model.gc1_0.QQ, model.gc1_1.indices, model.gc1_1.inp)
        
        dmu_2[:,k], dsi_2[:,k] = get_grad(grad_out2, model.gc2_0.T[:,k], model.gc2_1.support, model.gc2_0.mu[:,k], model.gc2_0.sig[:,k], Diff, model.gc2_0.QQ, model.gc2_1.indices, model.gc2_1.inp)
        
        dmu_3[:,k], dsi_3[:,k] = get_grad(grad_out3, model.gc3_0.T[:,k], model.gc3_1.support, model.gc3_0.mu[:,k], model.gc3_0.sig[:,k], Diff, model.gc3_0.QQ, model.gc3_1.indices, model.gc3_1.inp)
        
        dmu_4[:,k], dsi_4[:,k] = get_grad(grad_out4, model.gc4_0.T[:,k], model.gc4_1.support, model.gc4_0.mu[:,k], model.gc4_0.sig[:,k], Diff, model.gc4_0.QQ, model.gc4_1.indices, model.gc4_1.inp)
        
    model.gc1_0.mu.data -= learning_rate_mu * dmu_1.data
    
    model.gc2_0.mu.data -= learning_rate_mu * dmu_2.data
    
    model.gc3_0.mu.data -= learning_rate_mu * dmu_3.data
    
    model.gc4_0.mu.data -= learning_rate_mu * dmu_4.data
    

    model.gc1_0.sig.data -= learning_rate_si * dsi_1.data
    
    model.gc2_0.sig.data -= learning_rate_si * dsi_2.data
    
    model.gc3_0.sig.data -= learning_rate_si * dsi_3.data
    
    model.gc4_0.sig.data -= learning_rate_si * dsi_4.data
    
    #print('Epoch: {:04d}'.format(epoch+1),'file:{:04d}'.format(tra+1),
    #      'loss_train: {:.4f}'.format(loss_train.data[0]),
    #      'Dic_train1: {:.4f}'.format(acc_train1.data[0]),
    #      'acc_train1: {:.4f}'.format(acc_train2.data[0]),
    #      'time: {:.4f}s'.format(time.time() - t))

    return loss_train.data[0], acc_train1.data[0], acc_train2.data[0]



t_total = time.time()

for epoch in range(args.epochs):
    t = time.time()
    mn_ls=[]
    mn_ac1 = []
    mn_ac2 = []
    idx = np.arange(len(file_train))
    np.random.shuffle(idx)
    #wht_dc = Variable(np.ceil(wht.data/(epoch+1)).cuda())

    for tra in range(len(file_train)):

        E1, E2, Diff, features, labels = load_data(path_train + '/' + file_train[idx[tra]])
        loss, acc1, acc2 = train(epoch,tra, wht)
        
        mn_ls.append(loss)
        mn_ac1.append(acc1)
        mn_ac2.append(acc2)

    if ifsaveall:
        np.save('./loss_tr/' + str(epoch) + '.npy', np.mean(mn_ls))
        np.save('./acc1_tr/' + str(epoch) + '.npy', np.mean(mn_ac1))
        np.save('./acc2_tr/' + str(epoch) + '.npy', np.mean(mn_ac2))


    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(np.mean(mn_ls)),
          'Dic_train: {:.4f}'.format(np.mean(mn_ac1)),
          'acc_train: {:.4f}'.format(np.mean(mn_ac2)),
          'time: {:.4f}s'.format(time.time() - t))
 


    mn_ls=[]
    mn_ac1 = []
    mn_ac2 = []
    t = time.time()

    for val in range(len(file_valid)):

        E1, E2, Diff, features, labels = load_data(path_valid + '/' + file_valid[val])        
        loss, acc1, acc2 = test(epoch,val, wht)

        mn_ls.append(loss)
        mn_ac1.append(acc1)
        mn_ac2.append(acc2)

    if ifsaveall:
        np.save('./loss_va/' + str(epoch) + '.npy', np.mean(mn_ls))
        np.save('./acc1_va/' + str(epoch) + '.npy', np.mean(mn_ac1))
        np.save('./acc2_va/' + str(epoch) + '.npy', np.mean(mn_ac2))
    '''
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_vali : {:.4f}'.format(np.mean(mn_ls)),
          'Dic_vali : {:.4f}'.format(np.mean(mn_ac1)),
          'acc_vali : {:.4f}'.format(np.mean(mn_ac2)),
          'time: {:.4f}s'.format(time.time() - t))
    '''



    mn_ls=[]
    mn_ac1 = []
    mn_ac2 = []
    t = time.time()

    for tes in range(len(file_test)):

        E1, E2, Diff, features, labels = load_data(path_test + '/' + file_test[tes])        
        loss, acc1, acc2 = test(epoch,tes, wht)

        mn_ls.append(loss)
        mn_ac1.append(acc1)
        mn_ac2.append(acc2)

    if ifsaveall:
        np.save('./loss_te/' + str(epoch) + '.npy', np.mean(mn_ls))
        np.save('./acc1_te/' + str(epoch) + '.npy', np.mean(mn_ac1))
        np.save('./acc2_te/' + str(epoch) + '.npy', np.mean(mn_ac2))

        if np.mean(mn_ac1) > initial_dic:
            initial_dic = np.mean(mn_ac1)
            torch.save(model, './wht/weights_'+ str(epoch))
    
    '''
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_test : {:.4f}'.format(np.mean(mn_ls)),
          'Dic_test : {:.4f}'.format(np.mean(mn_ac1)),
          'acc_test : {:.4f}'.format(np.mean(mn_ac2)),
          'time: {:.4f}s'.format(time.time() - t))
    '''
    

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
