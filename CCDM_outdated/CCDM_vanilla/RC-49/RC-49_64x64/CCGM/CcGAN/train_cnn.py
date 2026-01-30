''' For CNN training and testing. '''


import os
import timeit
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

## horizontal flipping
def hflip_images(batch_images):
    ''' for numpy arrays '''
    # uniform_threshold = np.random.uniform(0,1,len(batch_images))
    # indx_gt = np.where(uniform_threshold>0.5)[0]
    # batch_images[indx_gt] = np.flip(batch_images[indx_gt], axis=3)
    return batch_images

## normalize images
def normalize_images(batch_images):
    batch_images = batch_images/255.0
    batch_images = (batch_images - 0.5)/0.5
    return batch_images

''' function for cnn training '''
def train_cnn(net, net_name, train_images, train_labels, testloader, epochs, resume_epoch=0, save_freq=40, batch_size=128, lr_base=0.01, lr_decay_factor=0.1, lr_decay_epochs=[150, 250], weight_decay=1e-4, path_to_ckpt = None, fn_denorm_labels=None):
    
    '''
    train_images: unnormalized images
    train_labels: normalized labels
    '''
    
    assert train_images.max()>1 and train_images.max()<=255.0 and train_images.min()>=0
    assert train_labels.min()>=0 and train_labels.max()<=1.0
    
    unique_train_labels = np.sort(np.array(list(set(train_labels)))) ##sorted unique labels
    
    indx_all = np.arange(len(train_labels))

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


    net = net.cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr = lr_base, momentum= 0.9, weight_decay=weight_decay)

    if path_to_ckpt is not None and resume_epoch>0:
        save_file = path_to_ckpt + "/{}_checkpoint_epoch_{}.pth".format(net_name, resume_epoch)
        checkpoint = torch.load(save_file)
        net.load_state_dict(checkpoint['net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])
    #end if


    train_loss_all = []
    test_mae_all = []


    start_time = timeit.default_timer()
    for epoch in range(resume_epoch, epochs):
        net.train()
        train_loss = 0
        adjust_learning_rate(optimizer, epoch)
        
        for batch_idx in range(len(train_labels)//batch_size):
            # ### generate target labels
            # batch_target_labels = np.random.choice(unique_train_labels, size=batch_size, replace=True)
            # batch_unique_train_labels, batch_unique_label_counts = np.unique(batch_target_labels, return_counts=True)

            # batch_train_indx = []
            # for j in range(len(batch_unique_train_labels)):
            #     indx_j = np.where(train_labels==batch_unique_train_labels[j])[0]
            #     indx_j = np.random.choice(indx_j, size=batch_unique_label_counts[j])
            #     batch_train_indx.append(indx_j)
            # batch_train_indx = np.concatenate(batch_train_indx)
            # batch_train_indx = batch_train_indx.reshape(-1)
            
            batch_train_indx = np.random.choice(indx_all, size=batch_size, replace=True).reshape(-1) #for cnn training only
            
            ### get some real images for training
            batch_train_images = train_images[batch_train_indx]
            batch_train_images = hflip_images(batch_train_images) ## randomly flip real images
            batch_train_images = normalize_images(batch_train_images) ## normalize real images
            batch_train_images = torch.from_numpy(batch_train_images).type(torch.float).cuda()
            assert batch_train_images.max().item()<=1.0
            batch_train_labels = train_labels[batch_train_indx]
            batch_train_labels = torch.from_numpy(batch_train_labels).type(torch.float).view(-1,1).cuda()
        
            #Forward pass
            outputs = net(batch_train_images)
            loss = criterion(outputs, batch_train_labels)
        
            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().item()
        
        #end for batch_idx
        train_loss = train_loss / (len(train_labels)//batch_size)
        train_loss_all.append(train_loss)
        
        test_mae = test_cnn(net, testloader, fn_denorm_labels=fn_denorm_labels, verbose=False)
        test_mae_all.append(test_mae)


        print('%s: [epoch %d/%d] train_loss:%.3f, test_mae:%.3f Time: %.4f' % (net_name, epoch+1, epochs, train_loss, test_mae, timeit.default_timer()-start_time))

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



def test_cnn(net, testloader, fn_denorm_labels=None, verbose=False):

    net = net.cuda()
    net.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        abs_diff_avg = 0
        total = 0
        for _, (images, labels) in enumerate(testloader):
            images = images.type(torch.float).cuda()
            labels = labels.type(torch.float).view(-1).cpu().numpy()
            outputs = net(images)
            outputs = outputs.view(-1).cpu().numpy()
            labels = fn_denorm_labels(labels)
            outputs = fn_denorm_labels(outputs)
            abs_diff_avg += np.sum(np.abs(labels-outputs))
            total += len(labels)

    test_mae = abs_diff_avg/total
    if verbose:
        print('\n Test MAE: {}.'.format(test_mae))
    return test_mae

