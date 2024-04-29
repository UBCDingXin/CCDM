import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.nn import functional as F
import sys
import PIL
from PIL import Image

# ### import my stuffs ###
# from models import *


# ################################################################################
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


################################################################################
# torch dataset from numpy array
# class IMGs_dataset(torch.utils.data.Dataset):
#     def __init__(self, images, labels=None, transform=None):
#         super(IMGs_dataset, self).__init__()

#         self.images = images
#         self.n_images = len(self.images)
#         self.labels = labels
#         if labels is not None:
#             if len(self.images) != len(self.labels):
#                 raise Exception('images (' +  str(len(self.images)) +') and labels ('+str(len(self.labels))+') do not have the same length!!!')
#         self.transform = transform

#     def __getitem__(self, index):

#         ## for RGB only
#         image = self.images[index]
#         if self.transform is not None:
#             image = np.transpose(image, (1, 2, 0)) #C * H * W ---->  H * W * C
#             image = Image.fromarray(np.uint8(image), mode = 'RGB') #H * W * C
#             image = self.transform(image)

#         if self.labels is not None:
#             label = self.labels[index]

#             return image, label

#         return image

#     def __len__(self):
#         return self.n_images

class IMGs_dataset(torch.utils.data.Dataset):
    def __init__(self, images, labels=None, normalize=False):
        super(IMGs_dataset, self).__init__()

        self.images = images
        self.n_images = len(self.images)
        self.labels = labels
        if labels is not None:
            if len(self.images) != len(self.labels):
                raise Exception('images (' +  str(len(self.images)) +') and labels ('+str(len(self.labels))+') do not have the same length!!!')
        self.normalize = normalize

    def __getitem__(self, index):

        image = self.images[index]

        if self.normalize:
            image = image/255.0
            # image = (image-0.5)/0.5

        if self.labels is not None:
            label = self.labels[index]
            return (image, label)
        else:
            return image

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
