#############################################################################################
################ Import function here #######################################################
#############################################################################################

from __future__ import division
from __future__ import print_function
import time
import argparse
import numpy as np
import pdb
import os
import torch
import scipy.sparse as sp
import torch.nn.functional as F
import torch.optim as optim
import json
import geometricDataLoader
import scipy.io as sio

from torch.autograd import Variable
from os import listdir
from os.path import isfile, join
from custom_callbacks.Loss_plotter import LossPlotter
from custom_callbacks.Logger import Logger
from shutil import copyfile
from tqdm import tqdm
from torch import nn
from inoput.data import DataLoader
from input.data import Data
from torch.optim.lr_scheduler import StepLR

from models.models_ourpool import GCNet
from utils.utils import load_data, accuracy, bin_accuracy




#############################################################################################
################ Get the information from the command line ##################################
#############################################################################################

def _get_config():
    
    parser = argparse.ArgumentParser(description="Main handler for training", usage="python ./train.py -j config.json -g 0")
    parser.add_argument("-j", "--json", help="configuration json file", required=True)
    parser.add_argument('-g', '--gpu', help='Cuda Visible Devices', required=True)
    
    args = parser.parse_args()
    
    with open(args.json, 'r') as f:
        config = json.loads(f.read())
    
    initial_weights = config['generator']['initial_epoch']
    directory = os.path.join(config['directories']['outdir'], config['directories']['ConfigName'],'config',str(initial_weights))
    if not os.path.exists(directory):
        os.makedirs(directory)

    copyfile(args.json, os.path.join(config['directories']['outdir'], config['directories']['ConfigName'],'config',str(initial_weights),'config.json'))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    return config

##############################################################################################



def main(config):

    # device
    device = torch.device("cuda")

    # directory configuration
    directory_config = config['directories']

    outdir = directory_config['outdir']                                     # Full Path 
    main_path = directory_config['datafile']                                # Full Path 
    ConfigName = directory_config['ConfigName']                             # Configuration Name to Uniquely Identify this Experiment

    #########################################################################################################

    log_path = join(outdir,ConfigName,'log')
    if not os.path.exists(log_path):
    	os.makedirs(log_path)
	if not os.path.exists(join(log_path, 'weights')):
		os.makedirs(join(log_path, 'weights'))    	

    ##################################################################################################
    generator_config = config['generator']
    initial_epoch = generator_config['initial_epoch']
    num_epochs = generator_config['num_epochs']                              # Total Number of Epochs: Ex. 240
    loss_up = generator_config['loss_up']
    lamb = generator_config['lamda']

    optim_config = config['optimizer']

    B1 = optim_config['B1']                                                  # B1 for Adam Optimizer: Ex. 0.9
    B2 = optim_config['B2']                                                  # B2 for Adam Optimizer: Ex. 0.999
    LR = optim_config['LR']                                                  # Learning Rate: Ex. 0.001
    LR_decay = optim_config['LR_decay']                                      # Learning Rate Decay: Ex. 0.5 
    LR_step_epoch = optim_config['LR_step_epoch']                            # Epoch after which apply Learning rate decay: Ex. 100+

    wht = torch.FloatTensor([1.3]).cuda()
    ggtt_all = []

    #####################################################################################################

    model = GCNet()

    print("===> Model Defined and Initialized ")

    if initial_epoch > 0:
        print("===> Loading pre-trained weight {}".format(initial_epoch))
        weight_path = 'weights/model-{:04d}.pt'.format(initial_epoch)
        checkpoint = torch.load(join(log_path, weight_path))
        model.load_state_dict(checkpoint['model_state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model = model.to(device)

    print("\n\n\n")
    print(model)
    print("\n\n\n")
    ##################################################################################################

    train_set = geometricDataLoader.GeometricImageDataset('train',main_path)
    train_loader = DataLoader(train_set,
                              batch_size=1,
                              num_workers=5,
                              shuffle=True)

    valid_set = geometricDataLoader.GeometricImageDataset('val',main_path)
    valid_loader = DataLoader(valid_set,
                              batch_size=1,
                              num_workers=5,
                              shuffle=False)


    test_set = geometricDataLoader.GeometricImageDataset('test',main_path)
    test_loader = DataLoader(test_set,
                              batch_size=1,
                              num_workers=5,
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
    #optimizer = optim.SGD(model.parameters(), lr=LR, momentum=B1)
    scheduler = StepLR(optimizer, LR_step_epoch, LR_decay)

    print("===> Optimizer Initialized")

    ############################################################################################

    def checkpoint(epoch):
        w_path = 'weights/model-{:04d}.pt'.format(epoch)
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, join(log_path, w_path))
        print("===> Checkpoint saved to {}".format(w_path))


    #################################################################################################

    def train(epoch):

        model.train()

        epoch_loss = 0
        acc_all = 0
        
        crite = torch.nn.BCEWithLogitsLoss()

        for data in tqdm(train_loader):

            optimizer.zero_grad()
            model.zero_grad()
            
            data.to(device)

            out, reg = model(data)
            gt = data.y    
            loss = (loss_up * crite(out, gt)) + (lamb * reg)
            loss.backward(retain_graph=True)
            optimizer.step()

            pred = torch.max(out, 0)[1].item()


            epoch_loss += loss.item()
            ggtt = torch.max(data.y, 0)[1].item()
            ggtt_all.append(pred)
            acc_all += bin_accuracy(pred,ggtt)
            
        metric = np.array([epoch_loss / len(train_loader), acc_all / len(train_loader)]) 
        #pdb.set_trace()
        return metric        

    ###################################################################################################

    def test(loader):

        model.eval()

        epoch_loss = 0
        acc_all = 0
       
        crite = torch.nn.BCEWithLogitsLoss()

        with torch.no_grad():

            for data in loader:
            
                data.to(device)

                out, reg = model(data)
                gt = data.y
      
                loss = crite(out, gt) + (lamb * reg)

                pred = torch.max(out, 0)[1].item()

                epoch_loss += loss.item()
                ggtt = torch.max(data.y, 0)[1].item()
                
                acc_all += bin_accuracy(pred,ggtt)
            
            metric = np.array([epoch_loss / len(loader), acc_all / len(loader)]) 
        #pdb.set_trace()
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
        print("===> Training   Epoch {}: Loss = {:.4f}, Accuracy = {:.4f}".format(epch, train_metric[0], train_metric[1]))

        val_metric = test(valid_loader)
        print("===> Validation Epoch {}: Loss = {:.4f}, Accuracy = {:.4f}".format(epch, val_metric[0], val_metric[1]))

        #if val_metric[1] > inti_val:
            #inti_val = val_metric[1]
        test_metric = test(test_loader)
        print("===> Testing    Epoch {}: Loss = {:.4f}, Accuracy = {:.4f}".format(epch, test_metric[0], test_metric[1]))

    	logger.to_csv(np.concatenate((train_metric, val_metric, test_metric)), epch)


        print("===> Logged All Metrics")
 
        LP.plotter()
        print("===> Plotted All Metrics")
 
        checkpoint(epch)
 
        end = time.time()
        print("===> Epoch:{} Completed in {:.4f} seconds".format(epch, end-start))


    print("===> Done Training for Total {:.4f} Epochs".format(num_epochs))

###################################################################################

if __name__ == "__main__":
    main(_get_config())