print("\n===================================================================================================")

import os
import argparse
import shutil
import timeit
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import random
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib as mpl
from torch import autograd
from torchvision.utils import save_image
import csv
from tqdm import tqdm
import gc
import h5py

### import my stuffs ###
from opts import cnn_opts
from models import *
from utils import IMGs_dataset
from train_cnn import train_cnn, test_cnn


#######################################################################################
'''                                   Settings                                      '''
#######################################################################################
args = cnn_opts()
print(args)

#-------------------------------
# seeds
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
cudnn.benchmark = False
np.random.seed(args.seed)

#-------------------------------
# CNN settings
## lr decay scheme
lr_decay_epochs = (args.lr_decay_epochs).split("_")
lr_decay_epochs = [int(epoch) for epoch in lr_decay_epochs]

#-------------------------------
# output folders

output_directory = os.path.join(args.root_path, 'output/CNN')
cnn_info = '{}_lr_{}_decay_{}'.format(args.cnn_name, args.lr_base, args.weight_decay)
    
os.makedirs(output_directory, exist_ok=True)

#-------------------------------
# some functions
def fn_norm_labels(labels):
    '''
    labels: unnormalized labels
    '''
    return labels/float(args.max_label)

def fn_denorm_labels(labels):
    '''
    labels: normalized labels
    '''
    if isinstance(labels, np.ndarray):
        return (labels*args.max_label).astype(int)
    elif torch.is_tensor(labels):
        return (labels*args.max_label).type(torch.int)
    else:
        return int(labels*args.max_label)


#######################################################################################
'''                                Data loader                                      '''
#######################################################################################

data_filename = args.data_path + '/Cell200_{}x{}.h5'.format(args.img_size, args.img_size)
hf = h5py.File(data_filename, 'r')
labels = hf['CellCounts'][:]
labels = labels.astype(float)
images = hf['IMGs_grey'][:]
hf.close()

# subset of UTKFace
selected_labels = np.arange(args.min_label, args.max_label+1)
for i in range(len(selected_labels)):
    curr_label = selected_labels[i]
    index_curr_label = np.where(labels==curr_label)[0]
    if i == 0:
        images_subset = images[index_curr_label]
        labels_subset = labels[index_curr_label]
    else:
        images_subset = np.concatenate((images_subset, images[index_curr_label]), axis=0)
        labels_subset = np.concatenate((labels_subset, labels[index_curr_label]))
# for i
images = images_subset
labels = labels_subset
del images_subset, labels_subset; gc.collect()

# for each label select num_imgs_per_label
selected_labels = np.arange(args.min_label, args.max_label+1, args.label_stepsize)
n_unique_labels = len(selected_labels)

for i in range(n_unique_labels):
    curr_label = selected_labels[i]
    index_curr_label = np.where(labels==curr_label)[0]
    if i == 0:
        images_subset = images[index_curr_label[0:args.num_imgs_per_label]]
        labels_subset = labels[index_curr_label[0:args.num_imgs_per_label]]
    else:
        images_subset = np.concatenate((images_subset, images[index_curr_label[0:args.num_imgs_per_label]]), axis=0)
        labels_subset = np.concatenate((labels_subset, labels[index_curr_label[0:args.num_imgs_per_label]]))
# for i
images = images_subset
labels = labels_subset
del images_subset, labels_subset; gc.collect()

print("\r We have {} images with {} distinct labels".format(len(images), n_unique_labels))

# normalize labels
print("\n Range of unnormalized labels: ({},{})".format(np.min(labels), np.max(labels)))
labels = fn_norm_labels(labels)
print("\n Range of normalized labels: ({},{})".format(np.min(labels), np.max(labels)))


## number of real images
nreal = len(labels)
assert len(labels) == len(images)

## data loader for the training set and test set
trainset = IMGs_dataset(images, labels, normalize=True, transform=args.transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True, num_workers=args.num_workers)



#######################################################################################
'''                                  CNN Training                                   '''
#######################################################################################

### model initialization

net = cnn_dict[args.cnn_name]()
net = nn.DataParallel(net)

### start training
filename_ckpt = os.path.join(output_directory, 'ckpt_{}_epoch_{}_last.pth'.format(args.cnn_name, args.epochs))
print('\n' + filename_ckpt)

# training
if not os.path.isfile(filename_ckpt):
    print("\n Start training the {} >>>".format(args.cnn_name))

    path_to_ckpt_in_train = output_directory + '/ckpts_in_train/{}'.format(cnn_info)    
    os.makedirs(path_to_ckpt_in_train, exist_ok=True)

    train_cnn(net=net, net_name=args.cnn_name, trainloader=trainloader, testloader=trainloader, epochs=args.epochs, resume_epoch=args.resume_epoch, save_freq=args.save_freq, batch_size=args.batch_size_train, lr_base=args.lr_base, lr_decay_factor=args.lr_decay_factor, lr_decay_epochs=lr_decay_epochs, weight_decay=args.weight_decay, path_to_ckpt = path_to_ckpt_in_train, fn_denorm_labels=fn_denorm_labels)

    # store model
    torch.save({
        'net_state_dict': net.state_dict(),
    }, filename_ckpt)
    print("\n End training CNN.")
else:
    print("\n Loading pre-trained {}.".format(args.cnn_name))
    checkpoint = torch.load(filename_ckpt)
    net.load_state_dict(checkpoint['net_state_dict'])
#end if

print("\n===================================================================================================")
