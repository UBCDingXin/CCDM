import numpy as np
import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
import os
import sys

import torch
from torch import nn, einsum
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

# from dataset import IMGs_dataset


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def divisible_by(numer, denom):
    return (numer % denom) == 0

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


    



# number of parameters
def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}




# Progress Bar
class SimpleProgressBar():
    def __init__(self, width=50):
        self.last_x = -1
        self.width = width

    def update(self, x):
        assert 0 <= x <= 100 # `x`: progress in percent ( between 0 and 100)
        if self.last_x == int(x): return
        self.last_x = int(x)
        pointer = int(self.width * (x / 100.0))
        sys.stdout.write( '\r%d%% [%s]' % (int(x), '#' * pointer + '.' * (self.width - pointer)))
        sys.stdout.flush()
        if x == 100:
            print('')


## dataset
class IMGs_dataset(torch.utils.data.Dataset):
    def __init__(self, images, labels=None, transform=False, normalize=False, to_neg_one_to_one=False):
        super(IMGs_dataset, self).__init__()

        self.images = images
        self.n_images = len(self.images)
        self.labels = labels
        if labels is not None:
            if len(self.images) != len(self.labels):
                raise Exception('images (' +  str(len(self.images)) +') and labels ('+str(len(self.labels))+') do not have the same length!!!')
        self.normalize = normalize
        self.transform = transform
        self.degrees = [0, 90, 180, 270]
        self.to_neg_one_to_one = to_neg_one_to_one

    def __getitem__(self, index):

        image = self.images[index]

        if self.transform:
            assert np.max(image)>1
            image = image[0] #CxWxH ----> WxH
            PIL_im = Image.fromarray(np.uint8(image), mode = 'L')

            degrees = np.array(self.degrees)
            np.random.shuffle(degrees)
            degree = degrees[0]
            PIL_im = PIL_im.rotate(degree)

            if np.random.uniform(0,1) < 0.5: #random flip
                PIL_im = PIL_im.transpose(Image.FLIP_LEFT_RIGHT)

            if np.random.uniform(0,1) < 0.5: #random flip
                PIL_im = PIL_im.transpose(Image.FLIP_TOP_BOTTOM)

            image = np.array(PIL_im)
            image = image[np.newaxis,:,:]

        if self.normalize:
            image = image/255.0
            if self.to_neg_one_to_one:
                image = (image-0.5)/0.5
        
        if self.labels is not None:
            return (image.astype(np.float32), self.labels[index])
        else:
            return image.astype(np.float32)

    def __len__(self):
        return self.n_images


# compute entropy of class labels; labels is a numpy array
def compute_entropy(labels, base=None):
    value,counts = np.unique(labels, return_counts=True)
    norm_counts = counts / counts.sum()
    base = np.e if base is None else base
    return -(norm_counts * np.log(norm_counts)/np.log(base)).sum()

def predict_class_labels(net, images, batch_size=500, verbose=False, num_workers=0):
    net = net.cuda()
    net.eval()

    n = len(images)
    if batch_size>n:
        batch_size=n
    dataset_pred = IMGs_dataset(images, normalize=False)
    dataloader_pred = torch.utils.data.DataLoader(dataset_pred, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_labels_pred = np.zeros(n+batch_size)
    with torch.no_grad():
        nimgs_got = 0
        if verbose:
            pb = SimpleProgressBar()
        for batch_idx, batch_images in enumerate(dataloader_pred):
            batch_images = batch_images.type(torch.float).cuda()
            batch_size_curr = len(batch_images)

            outputs,_ = net(batch_images)
            _, batch_class_labels_pred = torch.max(outputs.data, 1)
            class_labels_pred[nimgs_got:(nimgs_got+batch_size_curr)] = batch_class_labels_pred.detach().cpu().numpy().reshape(-1)

            nimgs_got += batch_size_curr
            if verbose:
                pb.update((float(nimgs_got)/n)*100)
        #end for batch_idx
    class_labels_pred = class_labels_pred[0:n]
    return class_labels_pred



## horizontal flip images
def hflip_images(batch_images):
    ''' for numpy arrays '''
    uniform_threshold = np.random.uniform(0,1,len(batch_images))
    indx_gt = np.where(uniform_threshold>0.5)[0]
    batch_images[indx_gt] = np.flip(batch_images[indx_gt], axis=3)
    return batch_images

def hflip_images_tensor(batch_images):
    ''' for torch tensors '''
    uniform_threshold = np.random.uniform(0,1,len(batch_images))
    indx_gt = np.where(uniform_threshold>0.5)[0]
    batch_images[indx_gt] = torch.flip(batch_images[indx_gt], dims=[3])
    return batch_images

## normalize images
def normalize_images(batch_images, to_neg_one_to_one=False):
    batch_images = batch_images/255.0 #to [0,1]
    if to_neg_one_to_one:
        batch_images = (batch_images - 0.5)/0.5 
    return batch_images