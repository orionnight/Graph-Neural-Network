#############################################################################################
################ Import function here #######################################################
#############################################################################################

import time
import argparse
import numpy as np
import os
import torch
import scipy.sparse as sp
import torch.nn.functional as F
import torch.optim as optim
import json
import pdb

from torch.autograd import Variable
from os import listdir
from os.path import isfile, join
from custom_callbacks.Loss_plotter import LossPlotter
from custom_callbacks.Logger import Logger
from shutil import copyfile
from tqdm import tqdm
from torch import nn
from inputs.dataloader import DataLoader
from inputs.graphLoader import GeometricDataset

from torch.optim.lr_scheduler import StepLR

#from models.GCN_parcellation import GCN
from old_code.models import GCN
#from losses import DiceLoss


# from utils.utils import load_data, accuracy, bin_accuracy


#############################################################################################
################ Get the information from the command line ##################################
#############################################################################################

def _get_config():
    parser = argparse.ArgumentParser(description="Main handler for training",
                                     usage="python ./train.py -j config.json -g 0")
    parser.add_argument("-j", "--json", help="configuration json file", required=True)
    parser.add_argument('-g', '--gpu', help='Cuda Visible Devices', required=True)

    args = parser.parse_args()

    with open(args.json, 'r') as f:
        config = json.loads(f.read())

    initial_weights = config['generator']['initial_epoch']
    directory = os.path.join(config['directories']['outdir'], config['directories']['ConfigName'], 'config',
                             str(initial_weights))
    if not os.path.exists(directory):
        os.makedirs(directory)

    copyfile(args.json, os.path.join(config['directories']['outdir'], config['directories']['ConfigName'], 'config',
                                     str(initial_weights), 'config.json'))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    return config


##############################################################################################

def bin_accuracy(output, target):
    """Computes the accuracy for multiple binary predictions"""
    num = output.eq(target).sum().double().item()
    den = target.numel()
    acc = 100 * num / den
    return acc


def dice_loss(output, target, wht):

    eps = 0.000000001

    output = F.softmax(output, 1)
    
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

def main(config):
    # device
    device = torch.device("cuda")

    # directory configuration
    directory_config = config['directories']

    outdir = directory_config['outdir']  # Full Path
    main_path = directory_config['datafile']  # Full Path
    ConfigName = directory_config['ConfigName']  # Configuration Name to Uniquely Identify this Experiment

    #########################################################################################################

    log_path = join(outdir, ConfigName, 'log')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(join(log_path, 'weights')):
        os.makedirs(join(log_path, 'weights'))

    ##################################################################################################

    generator_config = config['generator']
    initial_epoch = generator_config['initial_epoch']
    num_epochs = generator_config['num_epochs']  # Total Number of Epochs: Ex. 240
    loss_up = generator_config['loss_up']
    lamb = generator_config['lamda']

    optim_config = config['optimizer']

    B1 = optim_config['B1']  # B1 for Adam Optimizer: Ex. 0.9
    B2 = optim_config['B2']  # B2 for Adam Optimizer: Ex. 0.999
    LR = optim_config['LR']  # Learning Rate: Ex. 0.001
    LR_decay = optim_config['LR_decay']  # Learning Rate Decay: Ex. 0.5
    LR_step_epoch = optim_config['LR_step_epoch']  # Epoch after which apply Learning rate decay: Ex. 100+

    # wht = torch.FloatTensor([1.3]).cuda()
    ggtt_all = []

    #####################################################################################################

    model = GCN(nfeat=4, nhid=256, nclass=32, emb_size=3, ker_size=6)

    print("===> Model Defined and Initialized ")

    if initial_epoch > 0:
        print("===> Loading pre-trained weight {}".format(initial_epoch))
        weight_path = 'weights/model-{:04d}.pt'.format(initial_epoch)
        checkpoint = torch.load(join(log_path, weight_path))
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model = model.to(device)

    print("\n\n\n")
    print(model)
    print("\n\n\n")
    # ##################################################################################################

    train_set = GeometricDataset('train', main_path)
    train_loader = DataLoader(train_set,
                              batch_size=1,
                              shuffle=True)

    valid_set = GeometricDataset('val', main_path)
    valid_loader = DataLoader(valid_set,
                              batch_size=1,
                              shuffle=False)

    test_set = GeometricDataset('test', main_path)
    test_loader = DataLoader(test_set,
                             batch_size=1,
                             shuffle=False)

    print("===> Training Test and Validation Generators Initialized")

    ##########################################################################################################

    my_metric = ['Accuracy']
    my_loss = ['Loss']

    # setup our callbacks

    logger = Logger(mylog_path=log_path, mylog_name="training.log", myloss_names=my_loss, mymetric_names=my_metric)
    LP = LossPlotter(mylog_path=log_path, mylog_name="training.log", myloss_names=my_loss, mymetric_names=my_metric)

    print("===> Logger and LossPlotter Initialized")

    #############################################################################################

    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(B1, B2))
    # optimizer = optim.SGD(model.parameters(), lr=LR, momentum=B1)
    # scheduler = StepLR(optimizer, LR_step_epoch, LR_decay)

    print("===> Optimizer Initialized")

    ############################################################################################

    def checkpoint(epoch):
        w_path = 'weights/model-{:04d}.pt'.format(epoch)
        torch.save(
            {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
            join(log_path, w_path))
        print("===> Checkpoint saved to {}".format(w_path))

    #################################################################################################

    emb_size=3
    ker_size=6
    learning_rate_mu = 0.000000001;
    learning_rate_si = 1000

    dmu_1 = torch.autograd.Variable(torch.Tensor(emb_size,ker_size).zero_(),requires_grad=True).cuda()
    dsi_1 = torch.autograd.Variable(torch.Tensor(1,ker_size).zero_(),requires_grad=True).cuda()

    dmu_2 = torch.autograd.Variable(torch.Tensor(emb_size,ker_size).zero_(),requires_grad=True).cuda()
    dsi_2 = torch.autograd.Variable(torch.Tensor(1,ker_size).zero_(),requires_grad=True).cuda()

    dmu_3 = torch.autograd.Variable(torch.Tensor(emb_size,ker_size).zero_(),requires_grad=True).cuda()
    dsi_3 = torch.autograd.Variable(torch.Tensor(1,ker_size).zero_(),requires_grad=True).cuda()

    dmu_4 = torch.autograd.Variable(torch.Tensor(emb_size,ker_size).zero_(),requires_grad=True).cuda()
    dsi_4 = torch.autograd.Variable(torch.Tensor(1,ker_size).zero_(),requires_grad=True).cuda()


    #--------------------------Functions to get the gradients, dice--------------------------




    def get_grad(grad_output, matrix1, matrix2,mu,sig,Diff, QQ, indices, inp):

        grad_matrix1 = torch.mul(grad_output,matrix2)
        dmu = torch.FloatTensor(3).cuda()
        dsig = torch.FloatTensor(1).cuda()    
        shape = torch.Size((inp.shape[0],inp.shape[0]))
        
        for d in range(3):

            values = sig * ((Diff[:,d] * matrix1) - (mu[d] * matrix1))
            F = torch.sparse.FloatTensor(indices.cpu(), values.data.cpu(), shape).cuda()
            temp2 = torch.mm(F, matrix2.data)
            dmu[d] = (grad_output.data * temp2).sum()

        #pdb.set_trace()
        values = matrix1 * QQ
        F = torch.sparse.FloatTensor(indices.cpu(), values.data.cpu(), shape).cuda()
        temp2 = torch.mm(F, matrix2.data)
        dsig[0] = (grad_output.data * temp2).sum()

        return dmu, dsig


    #################################################################################################


    def train(epoch):

        epoch_loss = 0
        acc_all = 0 
        wht = torch.FloatTensor(np.array([16.97378, 85.77480065, 32.16534025, 43.73894593, 232.04827732, 31.40006501, 20.5252747, 26.5694486, 99.08358305, 17.44059628, 32.52226579, 29.0925319, 65.7419286, 21.93470876, 136.3009185, 
            54.85942554, 60.93557943, 126.48053398, 60.71517654, 67.2784416, 18.68115005, 69.16945821, 18.3112146, 24.79957415, 80.78001951, 21.33177316, 11.76718719, 20.24304891, 18.4545681, 24.17785204, 184.53911415, 47.92705005])).cuda()
        model.train()    


        for data in tqdm(train_loader):

            model.zero_grad()
            model.gc1_1.register_backward_hook(get_lay_grad_1)        
            model.gc2_1.register_backward_hook(get_lay_grad_2)        
            model.gc3_1.register_backward_hook(get_lay_grad_3)        
            model.gc4_1.register_backward_hook(get_lay_grad_4)

            optimizer.zero_grad()
            data.to(device)

            domain = data._x[:, :3]
            diff = domain[data._edge_idx[0, :], :] - domain[data._edge_idx[1, :], :]

            output = model(data)


            gt = torch.max(data._gt, 1)[1]

            wht_loss, loss, wht = dice_loss(output, data._gt.float(), wht)
            loss_train =  wht_loss + F.cross_entropy(output, gt, weight = wht)

            acc_dice = 100 * (1-loss/32)            

            loss_train.backward()
            optimizer.step()

            for k in range(ker_size):
                dmu_1[:,k], dsi_1[:,k] = get_grad(grad_out1, model.gc1_0.T[:,k], model.gc1_1.support, model.gc1_0.mu[:,k], model.gc1_0.sig[:,k], diff, model.gc1_0.QQ, model.gc1_1.indices, model.gc1_1.inp)                
                dmu_2[:,k], dsi_2[:,k] = get_grad(grad_out2, model.gc2_0.T[:,k], model.gc2_1.support, model.gc2_0.mu[:,k], model.gc2_0.sig[:,k], diff, model.gc2_0.QQ, model.gc2_1.indices, model.gc2_1.inp)                
                dmu_3[:,k], dsi_3[:,k] = get_grad(grad_out3, model.gc3_0.T[:,k], model.gc3_1.support, model.gc3_0.mu[:,k], model.gc3_0.sig[:,k], diff, model.gc3_0.QQ, model.gc3_1.indices, model.gc3_1.inp)                
                dmu_4[:,k], dsi_4[:,k] = get_grad(grad_out4, model.gc4_0.T[:,k], model.gc4_1.support, model.gc4_0.mu[:,k], model.gc4_0.sig[:,k], diff, model.gc4_0.QQ, model.gc4_1.indices, model.gc4_1.inp)
                
            model.gc1_0.mu.data -= learning_rate_mu * dmu_1.data            
            model.gc2_0.mu.data -= learning_rate_mu * dmu_2.data            
            model.gc3_0.mu.data -= learning_rate_mu * dmu_3.data            
            model.gc4_0.mu.data -= learning_rate_mu * dmu_4.data
            

            model.gc1_0.sig.data -= learning_rate_si * dsi_1.data            
            model.gc2_0.sig.data -= learning_rate_si * dsi_2.data            
            model.gc3_0.sig.data -= learning_rate_si * dsi_3.data            
            model.gc4_0.sig.data -= learning_rate_si * dsi_4.data
            

            epoch_loss += loss_train.item()
            acc_all += acc_dice.item()

        metric = np.array([epoch_loss / len(train_loader), acc_all / len(train_loader)])
        # pdb.set_trace()
        return metric

        ###################################################################################################

    def test(loader):

        model.eval()
        epoch_loss = 0
        acc_all = 0
        wht = torch.FloatTensor(np.array([16.97378, 85.77480065, 32.16534025, 43.73894593, 232.04827732, 31.40006501, 20.5252747, 26.5694486, 99.08358305, 17.44059628, 32.52226579, 29.0925319, 65.7419286, 21.93470876, 136.3009185, 
            54.85942554, 60.93557943, 126.48053398, 60.71517654, 67.2784416, 18.68115005, 69.16945821, 18.3112146, 24.79957415, 80.78001951, 21.33177316, 11.76718719, 20.24304891, 18.4545681, 24.17785204, 184.53911415, 47.92705005])).cuda()

        with torch.no_grad():
            for data in loader:
                data.to(device)

                output = model(data)

                gt = torch.max(data._gt, 1)[1]

                wht_loss, loss, wht = dice_loss(output, data._gt.float(), wht)
                loss_train =  wht_loss + F.cross_entropy(output, gt, weight = wht)

                acc_dice = 100 * (1-loss/32)  

                epoch_loss += loss.item()
                acc_all += acc_dice.item()

            metric = np.array([epoch_loss / len(loader), acc_all / len(loader)])

        return metric

        #########################################################################################################

    total_params = sum(p.numel() for p in model.parameters())

    print("===> Starting Model Training at Epoch: {}".format(initial_epoch))
    print("===> Total Model Parameter: ", total_params)

    inti_val = 0

    for epch in range(initial_epoch, num_epochs):
        start = time.time()

        print("\n\n")
        print("Epoch:{}".format(epch))

        _ = train(epch)
        train_metric = test(train_loader)
        print(
            "===> Training   Epoch {}: Loss = {:.4f}, Accuracy = {:.4f}".format(epch, train_metric[0], train_metric[1]))

        val_metric = test(valid_loader)
        print("===> Validation Epoch {}: Loss = {:.4f}, Accuracy = {:.4f}".format(epch, val_metric[0], val_metric[1]))

        # if val_metric[1] > inti_val:
        # inti_val = val_metric[1]
        test_metric = test(test_loader)
        print("===> Testing    Epoch {}: Loss = {:.4f}, Accuracy = {:.4f}".format(epch, test_metric[0], test_metric[1]))

        logger.to_csv(np.concatenate((train_metric, val_metric, test_metric)), epch)

        print("===> Logged All Metrics")

        LP.plotter()
        print("===> Plotted All Metrics")

        checkpoint(epch)

        end = time.time()
        print("===> Epoch:{} Completed in {:.4f} seconds".format(epch, end - start))

    print("===> Done Training for Total {:.4f} Epochs".format(num_epochs))


###################################################################################

if __name__ == "__main__":
    main(_get_config())
