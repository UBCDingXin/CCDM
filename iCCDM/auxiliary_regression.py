'''

Train an independent auxiliary regressor.

'''

print("\n===================================================================================================")

import argparse
import copy
import gc
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib as mpl
import h5py
import os
import random
from tqdm import tqdm, trange
import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
import timeit
from PIL import Image
import sys
from datetime import datetime 
from accelerate import Accelerator
from accelerate.utils import set_seed

from utils import *
from models import resnet18_aux_regre, resnet34_aux_regre, resnet50_aux_regre
from dataset import LoadDataSet

##############################################
''' Settings '''
parser = argparse.ArgumentParser(description='Pre-train CNNs')
parser.add_argument('--root_path', type=str, default='')
parser.add_argument('--data_path', type=str, default='')
parser.add_argument('--seed', type=int, default=2025, metavar='S',
                    help='random seed (default: 2025)')

parser.add_argument('--data_name', type=str, default='RC-49_imb', choices=["RC-49", "UTKFace", "Cell200", "SteeringAngle","RC-49_imb", "Cell200_imb"])
parser.add_argument('--imb_type', type=str, default='unimodal', choices=['unimodal', 'dualmodal', 'trimodal', 'standard', 'none']) #none means using all data
parser.add_argument('--min_label', type=float, default=0.0)
parser.add_argument('--max_label', type=float, default=90.0)
parser.add_argument('--num_channels', type=int, default=3, metavar='N')
parser.add_argument('--img_size', type=int, default=64)
parser.add_argument('--max_num_img_per_label', type=int, default=2**20, metavar='N')
parser.add_argument('--num_img_per_label_after_replica', type=int, default=0, metavar='N')

parser.add_argument('--net_name', type=str, default='resnet18',
                    help='CNN for training; ResNetXX')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train CNNs (default: 100)')
parser.add_argument('--batch_size_train', type=int, default=256, metavar='N',
                    help='input batch size for training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='learning rate, default=0.1')
parser.add_argument('--weight_dacay', type=float, default=1e-4,
                    help='Weigth decay, default=1e-4')

parser.add_argument('--use_amp', action='store_true', default=False) #use mixed precision
parser.add_argument('--mixed_precision_type', type=str, default='fp16', choices=['no', 'fp16', 'bf16'])

args = parser.parse_args()

# seeds
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
cudnn.benchmark = False
np.random.seed(args.seed)


#######################################################################################
'''                                Output folders                                  '''
#######################################################################################
path_to_output = os.path.join(args.root_path, 'output/{}_{}/aux_reg_model'.format(args.data_name, args.img_size))
if args.data_name in ["RC-49_imb"]:
    path_to_output += "/{}".format(args.imb_type)
os.makedirs(path_to_output, exist_ok=True)

setting_log_file = os.path.join(path_to_output, 'setting_info.txt')
if not os.path.isfile(setting_log_file):
    logging_file = open(setting_log_file, "w")
    logging_file.close()
with open(setting_log_file, 'a') as logging_file:
    logging_file.write("\n===================================================================================================")
    print(args, file=logging_file)

#######################################################################################
'''                                Make dataset                                     '''
#######################################################################################

dataset = LoadDataSet(data_name=args.data_name, data_path=args.data_path, min_label=args.min_label, max_label=args.max_label, img_size=args.img_size, max_num_img_per_label=args.max_num_img_per_label, num_img_per_label_after_replica=args.num_img_per_label_after_replica, imbalance_type=args.imb_type)
    
train_images, train_labels, train_labels_norm = dataset.load_train_data()
num_classes = dataset.num_classes

trainset = IMGs_dataset(train_images, train_labels_norm, normalize=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True)

#######################################################################################
'''                                Initialization                                   '''
#######################################################################################

# accelerator
accelerator = Accelerator(mixed_precision = args.mixed_precision_type if args.use_amp else "no")
set_seed(args.seed)
device = accelerator.device

#initialize CNNs
def net_initialization(net_name):
    if net_name.lower() == "resnet18":
        net = resnet18_aux_regre(nc=args.num_channels)
    elif net_name.lower() == "resnet34":
        net = resnet34_aux_regre(nc=args.num_channels)
    elif net_name.lower() == "resnet50":
        net = resnet50_aux_regre(nc=args.num_channels)
    net = net.to(device)
    return net

# model initialization
net = net_initialization(args.net_name)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr = args.base_lr, momentum= 0.9, weight_decay=args.weight_dacay)

## prepare model, dataloader, optimizer with accelerator
net, optimizer, trainloader = accelerator.prepare(net, optimizer, trainloader)


#######################################################################################
'''                                 Training fn                                     '''
#######################################################################################

#adjust CNN learning rate
def adjust_learning_rate(optimizer, epoch, BASE_LR_CNN):
    lr = BASE_LR_CNN
    if epoch >= 50:
        lr /= 10
    if epoch >= 120:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_cnn():
    start_tmp = timeit.default_timer()
    for epoch in range(args.epochs):
        net.train()
        train_loss = 0
        adjust_learning_rate(optimizer, epoch, args.base_lr)
        # for batch_idx, (batch_train_images, batch_train_labels) in enumerate(trainloader):
        for batch_train_images, batch_train_labels in trainloader:
            optimizer.zero_grad()
            batch_train_images = batch_train_images.float().to(device)
            batch_train_labels = batch_train_labels.float().to(device)
            #Forward pass
            outputs = net(batch_train_images)
            loss = criterion(outputs.view(-1), batch_train_labels.view(-1))
            #backward pass
            accelerator.backward(loss)
            optimizer.step()
            #record training loss
            train_loss += loss.cpu().item()
        #end for batch_idx
        train_loss = train_loss / len(trainloader)
        print('PreAuxReg %s: [epoch %d/%d] train_loss:%f Time:%.4f' % (args.net_name, epoch+1, args.epochs, train_loss, timeit.default_timer()-start_tmp))
    #end for epoch
    return net


###########################################################################################################
# Training and validation
###########################################################################################################

filename_ckpt = path_to_output + '/ckpt_{}_epoch_{}.pth'.format(args.net_name, args.epochs)
print(filename_ckpt)

# training
if not os.path.isfile(filename_ckpt):
    # TRAIN CNN
    print("\n Begin training CNN: ")
    start = timeit.default_timer()
    net = train_cnn()
    stop = timeit.default_timer()
    print("Time elapses: {}s".format(stop - start))
    # save model
    torch.save({
    'net_state_dict': net.state_dict(),
    }, filename_ckpt)
else:
    print("\n Ckpt already exists")
    print("\n Loading...")
    checkpoint = torch.load(filename_ckpt)
    net.load_state_dict(checkpoint['net_state_dict'])
##end if


print("\n===================================================================================================")