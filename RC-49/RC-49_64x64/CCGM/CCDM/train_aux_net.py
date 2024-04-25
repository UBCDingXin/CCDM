"""

Train an auxilary regression network for predicting the regression label of a given noisy image

"""

import math
import numpy as np
import os
import timeit

import torch
from torch import nn, einsum
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from PIL import Image
from tqdm.auto import tqdm
from accelerate import Accelerator

from ema_pytorch import EMA
from utils import cycle, divisible_by, exists, normalize_images


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)




def train_aux_net(net, net_name, train_images, train_labels, epochs, resume_epoch=0, save_freq=40, batch_size=128, lr_base=0.01, lr_decay_factor=0.1, lr_decay_epochs=[150, 250], weight_decay=1e-4, path_to_ckpt = None, use_amp=True):
    
    '''
    train_images: unnormalized images
    train_labels: normalized labels
    timesteps: the timesteps for training diffusion models
    beta_schedule: the beta in DDPM
    '''

    assert train_images.max()>1 and train_images.max()<=255.0 and train_images.min()>=0
    assert train_labels.min()>=0 and train_labels.max()<=1.0
    
    unique_train_labels = np.sort(np.array(list(set(train_labels)))) ##sorted unique labels
    
    indx_all = np.arange(len(train_labels))


    #######################################
    ''' learning rate decay '''
    def adjust_learning_rate(optimizer, epoch):
        """decrease the learning rate """
        lr = lr_base

        num_decays = len(lr_decay_epochs)
        for decay_i in range(num_decays):
            if epoch >= lr_decay_epochs[decay_i]:
                lr = lr * lr_decay_factor
            #end if epoch
        #end for decay_i
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr



    #######################################
    ''' init net '''
    accelerator = Accelerator(
        mixed_precision = 'fp16' if use_amp else 'no'
    )
    device = accelerator.device

    net = net.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr = lr_base, momentum= 0.9, weight_decay=weight_decay)

    if path_to_ckpt is not None and resume_epoch>0:
        save_file = path_to_ckpt + "/{}_checkpoint_epoch_{}.pth".format(net_name, resume_epoch)
        checkpoint = torch.load(save_file)
        net.load_state_dict(checkpoint['net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])
    #end if


    #######################################
    ''' Start training '''

    train_loss_all = []

    start_time = timeit.default_timer()
    for epoch in range(resume_epoch, epochs):
        net.train()
        train_loss = 0
        adjust_learning_rate(optimizer, epoch)
        
        for batch_idx in range(len(train_labels)//batch_size):
            
            batch_train_indx = np.random.choice(indx_all, size=batch_size, replace=True).reshape(-1)
            
            ### get some real images for training
            batch_train_images = train_images[batch_train_indx]
            batch_train_images = normalize_images(batch_train_images) ## normalize real images
            batch_train_images = torch.from_numpy(batch_train_images).type(torch.float).to(device)
            assert batch_train_images.max().item()<=1.0 and batch_train_images.min().item()>=0

            ### get labels
            batch_train_labels = train_labels[batch_train_indx]
            batch_train_labels = torch.from_numpy(batch_train_labels).type(torch.float).to(device)
        
            #Forward pass
            outputs = net(batch_train_images)
            loss = criterion(outputs.view(-1), batch_train_labels.view(-1))
        
            #backward pass
            optimizer.zero_grad()
            # loss.backward()
            accelerator.backward(loss)
            optimizer.step()

            train_loss += loss.cpu().item()
        
        #end for batch_idx
        train_loss = train_loss / (len(train_labels)//batch_size)
        train_loss_all.append(train_loss)
        
        print('%s: [epoch %d/%d] train_loss:%.3f, Time: %.4f' % (net_name, epoch+1, epochs, train_loss, timeit.default_timer()-start_time))

        # save checkpoint
        if path_to_ckpt is not None and ((epoch+1) % save_freq == 0 or (epoch+1) == epochs) :
            save_file = path_to_ckpt + "/{}_checkpoint_epoch_{}.pth".format(net_name, epoch+1)
            torch.save({
                    'net_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'rng_state': torch.get_rng_state()
            }, save_file)
    #end for epoch

    return net








