### to calculate total parameters in pytorch model
### sum(p.numel() for p in model.parameters())


from __future__ import print_function
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import time
from math import log10
from tqdm import tqdm
import os
import matplotlib
matplotlib.use('Agg')
from os.path import join
import json
import argparse
from shutil import copyfile

from models.unet_models import Unet
from datagenerator.DataGenerator import BraTSSegDataGenerator

from custom_callbacks.Visualizer import BraTSSegUnetVisualizer
from custom_callbacks.Loss_plotter import LossPlotter
from custom_callbacks.Logger import Logger

import pandas as pd
import SimpleITK as sitk

############################################################################################


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



#################### Dice #################################

def dice_tumor(y_true, y_pred):
    y_true = np.greater(y_true, 0).astype('float32')
    y_pred = np.greater(y_pred, 0).astype('float32')
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    return (2. * intersection + 1.0) / (union + 1.0)

def dice_enhance(y_true, y_pred):
    y_true = np.equal(y_true, 4).astype('float32')
    y_pred = np.equal(y_pred, 4).astype('float32')
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    return (2. * intersection + 1.0) / (union + 1.0)

def dice_core(y_true, y_pred):
    y_true = np.equal(y_true, 1).astype('float32') + np.greater(y_true, 2).astype('float32')
    y_pred = np.equal(y_pred, 1).astype('float32') + np.greater(y_pred, 2).astype('float32')
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    return (2. * intersection + 1.0) / (union + 1.0)

######################################################################
#############################################################################################

def _get_config():
    
    parser = argparse.ArgumentParser(description="Main handler for training", usage="python ./train.py -j config.json -g 0")
    parser.add_argument("-j", "--json", help="configuration json file", required=True)
    parser.add_argument('-g', '--gpu', help='Cuda Visible Devices', required=True)
    
    args = parser.parse_args()
    
    with open(args.json, 'r') as f:
        config = json.loads(f.read())
    
    initial_weights = config['generator']['initial_epoch']
    os.makedirs(os.path.join(config['directories']['outdir'], config['directories']['ConfigName'],'config',str(initial_weights)), exist_ok=True)
    copyfile(args.json, os.path.join(config['directories']['outdir'], config['directories']['ConfigName'],'config',str(initial_weights),'config.json'))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    return config

###################################################################################


def main(config):

    # device
    device = torch.device("cuda")


    # model configuration
    model_config = config['model']

    n_scales = model_config['n_scales']                                      # Number of scales in Unet: Ex. 3
    n_layers = model_config['n_layers']                                      # Number of Layers in each Step in Unet: Ex. 2

    init_ker = model_config['init_ker']                                      # Number of Initial Kernals in the Architecture: Ex. 8

    norm = model_config['norm_type']                                         # Normalization Type: BatchNorm("bn"), InstanceNorm("in"), LayerNorm("ln"), GroupNorm("gn"), or "none"
    dropout_rate = model_config['dropout_rate']                              # dropout rate: 0 < dr < 0.5

    activ = model_config['activation_type']                                  # Activation Type: "relu", LeakyReLU("lrelu"), "prelu", "selu", "tanh", or "none"
    pad_type = model_config['pad_type']                                      # Padding Type: "zero" or "replicate"

    ds_type = model_config['downsample_type']                                # DownSample Type: AveragePooling("avg") or MaxPooling("max")
    us_type = model_config['upsample_type']                                  # UpSample Type: TransposedConv ("transpose") or UpSampling("upsample")

    merge_type = model_config['merge_type']                                  # Merge Type: Summation("sum"), Multiplication("mul") or Concatination("concate")



    # Optimizer configuration
    optim_config = config['optimizer']

    B1 = optim_config['B1']                                                  # B1 for Adam Optimizer: Ex. 0.9
    B2 = optim_config['B2']                                                  # B2 for Adam Optimizer: Ex. 0.999
    LR = optim_config['LR']                                                  # Learning Rate: Ex. 0.001
    LR_decay = optim_config['LR_decay']                                      # Learning Rate Decay: Ex. 0.5 
    LR_step_epoch = optim_config['LR_step_epoch']                            # Epoch after which apply Learning rate decay: Ex. 100


    # Generator Configuration
    generator_config = config['generator']

    dim_x = generator_config['dim_x']                                        # Size of input and output in x dim. Ex. 192
    dim_y = generator_config['dim_y']                                        # Size of input and output in y dim. Ex. 184
    dim_z = generator_config['dim_z']                                        # Size of input and output in z dim. Ex. 156

    batch_size = generator_config['batch_size']                              # Batch Size: Ex. 1
    num_epochs = generator_config['num_epochs']                              # Total Number of Epochs: Ex. 240
    initial_epoch = generator_config['initial_epoch']                        # Initial Epoch (Useful for Starting Training from middle): Ex. 0

    modalities = generator_config['modalities']                              # All Modalities: Ex. ["t1", "t2","flair", "t1ce", "seg"] make sure that segmentation modality is the last one
    number_of_classes = generator_config['number_of_classes']                # Number of Classes in Segmentation: Ex. 1

    use_weights = generator_config['use_weights']                            # If "1" uses sample weights according to segmentation modalities Otherwise not
    weight_decay = generator_config['weight_decay']                          # Weight Decay for sample weights: Ex. 0.96

    augment = generator_config['augment']                                    # If "1" augments the training data 
    img_cent = generator_config['img_cent']                                  # Center of Image. Ex. (120,120,77)
    trans_range = generator_config['trans_range']                            # Translation Range: Ex. 5.0 (Should be less than 10.0)
    rot_range = generator_config['rot_range']                                # Rotation Range: Ex. 3.0 (Should be less than 10.0)
    scal_range = generator_config['scale_range']                              # Scaling Range - Should be less than 0.2 Ex. 0.1
    shea_range = generator_config['shear_range']                              # Shear Range - Should be less than 0.2 Ex. 0.1
    interpolator = generator_config['interpolator']                          # Interpolator for SimpleITK: sitk.sitkLinear
    default_val = generator_config['default_val']                            # Default Value for pixels out of image range: Ex. 0

    fold = generator_config['fold']                                          # Fold for BraTS data. Ex. 'fold_1'

    # visualize config
    vis_config = config['visualize']    
    vis_slice = vis_config['vis_slice']                                      # Visualization Slice Number: Ex. 26



    # directory configuration
    directory_config = config['directories']

    outdir = directory_config['outdir']                                     # Full Path to Directory where to store all generated files: Ex. "/usr/local/data/raghav/MSLAQ_experiments/Experiments"
    main_path = directory_config['datafile']                                 # Full Path of Input HDf5 file: Ex. "/usr/local/data/raghav/MSLAQ_loader/MSLAQ.hdf5"
    ConfigName = directory_config['ConfigName']                             # Configuration Name to Uniquely Identify this Experiment

    #########################################################################################################

    log_path = join(outdir,ConfigName,'log')

    os.makedirs(log_path,exist_ok=True)
    os.makedirs(join(log_path, 'weights'), exist_ok=True)

    ##################################################################################################

    #####################################################################################################

    model = Unet( n_layers = n_layers,
                       n_scales = n_scales,
                       init_ker = init_ker,
                       inp_chan = len(modalities[:-1]),
                       out_chan = number_of_classes,
                       dropout_rate = dropout_rate,
                       norm = norm,
                       activ = activ,
                       pad_type = pad_type,
                       ds_type = ds_type,
                       us_type = us_type,
                       merge_type = merge_type)


    #model = model.to(device)
    model.apply(weights_init)

    print("===> Model Defined and Initialized with normal distribution")

    if initial_epoch > 0:
        print("===> Loading pre-trained weight {}".format(initial_epoch))
        weight_path = 'weights/model-{:04d}.pt'.format(initial_epoch)
        model = torch.load(join(log_path, weight_path))

    model = model.to(device)

    print("\n\n\n")
    print(model)
    print("\n\n\n")
    ##################################################################################################

    dataframe = pd.read_csv(join(main_path,"training_"+fold+".csv"),header=None,delim_whitespace=True) 
    train_IDs = dataframe.values
    train_IDs = np.squeeze(train_IDs)
    train_IDs = train_IDs
    print("===> Number of Training Data: {}".format(len(train_IDs)))

    dataframe = pd.read_csv(join(main_path,"validation_"+fold+".csv"),header=None,delim_whitespace=True)
    valid_IDs = dataframe.values
    valid_IDs = np.squeeze(valid_IDs)
    valid_IDs = valid_IDs
    print("===> Number of Validation Data: {}".format(len(valid_IDs)))


    tparams = { 'list_IDs': train_IDs,
                'main_path': join(main_path,'Data'),
                'modalities': modalities,
                'num_classes': number_of_classes,
                'weight_decay': weight_decay,
                'init_epoch': initial_epoch, 
                'dim_x': dim_x,
                'dim_y': dim_y,
                'dim_z': dim_z,
                'augment': augment,
                'img_cent': img_cent,
                'trans_range': trans_range,
                'rot_range': rot_range,
                'scal_range': scal_range,
                'shea_range': shea_range,
                'interpolator': sitk.sitkLinear,
                'default_val': default_val}

    vparams = { 'list_IDs': valid_IDs,
                'main_path': join(main_path,'Data'),
                'modalities': modalities,
                'num_classes': number_of_classes,
                'weight_decay': weight_decay,
                'init_epoch': initial_epoch, 
                'dim_x': dim_x,
                'dim_y': dim_y,
                'dim_z': dim_z,
                'augment': False,
                'img_cent': img_cent,
                'trans_range': trans_range,
                'rot_range': rot_range,
                'scal_range': scal_range,
                'shea_range': shea_range,
                'interpolator': sitk.sitkLinear,
                'default_val': default_val}

    training_generator = BraTSSegDataGenerator(**tparams)
    validation_generator = BraTSSegDataGenerator(**vparams)

    training_data_loader = DataLoader(training_generator, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    validation_data_loader = DataLoader(validation_generator, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)

    print("===> Training and Validation Generators Initialized")

    ##########################################################################################################

    vis_IDs = valid_IDs[0:len(valid_IDs):9]

    print("===> Number of Visualize Data: {}".format(len(vis_IDs)))


    visparams = {'write_images': True,
                'list_IDs': vis_IDs, 
                'log_path': log_path, 
                'modalities': modalities,
                'dim_x': dim_x,
                'dim_y': dim_y,
                'dim_z': dim_z,
                'main_path': join(main_path,'Data'),
                'device': device,
                'num_class': number_of_classes,
                'vis_slice': vis_slice}

    Visual = BraTSSegUnetVisualizer(**visparams)

    print("===> Visualizer Initialized")

    ##############################################################################################################


    my_metric = ['dice_tumor', 'dice_core', 'dice_enhance']

    my_loss = ["loss"]

    # setup our callbacks

    logger = Logger(mylog_path=log_path, mylog_name="training.log", myloss_names=my_loss, mymetric_names=my_metric)
    LP = LossPlotter(mylog_path=log_path, mylog_name="training.log", myloss_names=my_loss, mymetric_names=my_metric)

    print("===> Logger and LossPlotter Initialized")

    #############################################################################################

    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(B1, B2))
    #optimizer = optim.SGD(model.parameters(), lr=LR, momentum=B1)
    scheduler = StepLR(optimizer, LR_step_epoch, LR_decay, last_epoch=initial_epoch-1)

    print("===> Optimizer Initialized")

    ############################################################################################

    def checkpoint(epoch):
        w_path = 'weights/model-{:04d}.pt'.format(epoch)
        torch.save(model, join(log_path, w_path))
        print("===> Checkpoint saved to {}".format(w_path))


    #################################################################################################

    def train(epoch):

        model.train()

        scheduler.step()

        epoch_loss = 0

        weight = None

        if use_weights:
            weight = training_generator.get_weight()
            weight = torch.from_numpy(weight).type('torch.FloatTensor').to(device)

        crit = nn.CrossEntropyLoss(weight=weight)

        for iteration, batch in enumerate(tqdm(training_data_loader)):

            optimizer.zero_grad()

            inp, target = batch['input'].to(device), batch['output'].type('torch.LongTensor').to(device)

            outp = model(inp)

            loss = crit(outp,target)

            epoch_loss += loss.item()

            loss.backward()

            optimizer.step()

        print("===> Training Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))

    ###################################################################################################

    def test():

        model.eval()

        metric = np.zeros(len(my_metric)+1)

        crit = nn.CrossEntropyLoss()

        with torch.no_grad():

            for iteration, batch in enumerate(training_data_loader):

                inp, target = batch['input'].to(device), batch['output'].type('torch.LongTensor').to(device)

                outp = model(inp)

                loss = crit(outp,target)

                loss = loss.item()

                _, outp = outp.max(1)

                target = np.squeeze(target.data.cpu().numpy().astype('float32'))
                outp = np.squeeze(outp.data.cpu().numpy().astype('float32'))

                dice_t = dice_tumor(target,outp)
                dice_c = dice_core(target,outp)
                dice_e = dice_enhance(target, outp)

                metric += np.array([loss, dice_t, dice_c, dice_e])

        return metric/len(training_data_loader)

    #####################################################################################################

    def validate():

        model.eval()

        model.eval()

        metric = np.zeros(len(my_metric)+1)

        crit = nn.CrossEntropyLoss()

        with torch.no_grad():

            for iteration, batch in enumerate(validation_data_loader):

                inp, target = batch['input'].to(device), batch['output'].type('torch.LongTensor').to(device)

                outp = model(inp)

                loss = crit(outp,target)

                loss = loss.item()

                _, outp = outp.max(1)

                target = np.squeeze(target.data.cpu().numpy().astype('float32'))
                outp = np.squeeze(outp.data.cpu().numpy().astype('float32'))

                dice_t = dice_tumor(target,outp)
                dice_c = dice_core(target,outp)
                dice_e = dice_enhance(target, outp)

                metric += np.array([loss, dice_t, dice_c, dice_e])

        return metric/len(validation_data_loader)

    #########################################################################################################

    total_params = sum(p.numel() for p in model.parameters())

    print("===> Starting Model Training at Epoch: {}".format(initial_epoch))
    print("===> Total Model Parameter: ", total_params)

    for epch in range(initial_epoch, num_epochs):
        start = time.time()
        print("\n\n")
        print("Epoch:{}".format(epch))
        train(epch)
        train_metric = test()
        print("===> Training   Epoch {}: Loss - {:.4f}, Dice Tumor - {:.4f}, Dice Core - {:.4f}, Dice Enhance - {:.4f}".format(epch, train_metric[0], train_metric[1], train_metric[2], train_metric[3]))
        val_metric = validate()
        print("===> Validation Epoch {}: Loss - {:.4f}, Dice Tumor - {:.4f}, Dice Core - {:.4f}, Dice Enhance - {:.4f}".format(epch, val_metric[0], val_metric[1], val_metric[2], val_metric[3]))
        logger.to_csv(np.concatenate((train_metric, val_metric)), epch)
        print("===> Logged All Metrics")
        LP.plotter()
        print("===> Plotted All Metrics")
        Visual.vis(epch, model)
        print("===> Visualised Some output")
        checkpoint(epch)
        end = time.time()
        print("===> Epoch:{} Completed in {} seconds".format(epch, end-start))

    print("===> Done Training for Total {} Epochs".format(num_epochs))

###################################################################################

if __name__ == "__main__":
    main(_get_config())

