import math
import random
import os
import gc
import copy

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from tqdm import tqdm

import h5py
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset



def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    num_channels=1,
    min_label=1,
    max_label=200,
    stepsize=2,
    num_imgs_per_label=10,
    num_classes=100, 
    class_cond=True,
    deterministic=False,
    transform=True,
    num_workers=0,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param transform: if True, randomly transform the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    # load data from the h5 file
    data_filename = data_dir + '/Cell200_{}x{}.h5'.format(image_size, image_size)
    hf = h5py.File(data_filename, 'r')
    labels = hf['CellCounts'][:]
    labels = labels.astype(float)
    images = hf['IMGs_grey'][:]
    hf.close()

    raw_labels = copy.deepcopy(labels)

    # for each label select num_imgs_per_label
    selected_labels = np.arange(min_label, max_label+1, stepsize)
    n_unique_labels = len(selected_labels)

    for i in range(n_unique_labels):
        curr_label = selected_labels[i]
        index_curr_label = np.where(labels==curr_label)[0]
        if i == 0:
            images_subset = images[index_curr_label[0:num_imgs_per_label]]
            labels_subset = labels[index_curr_label[0:num_imgs_per_label]]
        else:
            images_subset = np.concatenate((images_subset, images[index_curr_label[0:num_imgs_per_label]]), axis=0)
            labels_subset = np.concatenate((labels_subset, labels[index_curr_label[0:num_imgs_per_label]]))
    # for i
    images = images_subset
    labels = labels_subset
    del images_subset, labels_subset; gc.collect()

    print("\r We have {} images with {} distinct labels".format(len(images), n_unique_labels))

    # treated as classification; convert regression labels to class labels
    unique_labels = np.sort(np.array(list(set(raw_labels)))) #not counts because we want the last element is the max_count
    num_unique_labels = len(unique_labels)
    print("{} distinct labels are split into {} classes".format(num_unique_labels, num_classes))

    ## convert regression labels to class labels and vice versa
    ### step 1: prepare two dictionaries
    label2class = dict()
    class2label = dict()
    num_labels_per_class = num_unique_labels//num_classes
    class_cutoff_points = [unique_labels[0]] #the cutoff points on [min_label, max_label] to determine classes; each interval is a class
    curr_class = 0
    for i in range(num_unique_labels):
        label2class[unique_labels[i]]=curr_class
        if (i+1)%num_labels_per_class==0 and (curr_class+1)!=num_classes:
            curr_class += 1
            class_cutoff_points.append(unique_labels[i+1])
    class_cutoff_points.append(unique_labels[-1])
    assert len(class_cutoff_points)-1 == num_classes

    ### the label of each interval equals to the average of the two end points
    for i in range(num_classes):
        class2label[i] = (class_cutoff_points[i]+class_cutoff_points[i+1])/2

    ### step 2: convert regression label to class labels
    labels_new = -1*np.ones(len(labels))
    for i in range(len(labels)):
        labels_new[i] = label2class[labels[i]]
    assert np.sum(labels_new<0)==0
    labels = labels_new
    del labels_new; gc.collect()
    unique_labels = np.sort(np.array(list(set(labels)))).astype(int)

    print(unique_labels)

    ### make the dataset and data loader
    if class_cond:
        trainset = IMGs_dataset(images, labels, normalize=True, transform=transform)
    else:
        trainset = IMGs_dataset(images, labels=None, normalize=True, transform=transform)
    
    if deterministic:
        loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True, pin_memory=True
        )
    else:
        loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True
        )

    while True:
        yield from loader


class IMGs_dataset(torch.utils.data.Dataset):
    def __init__(self, images, labels=None, normalize=False, transform=False):
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
            image = (image-0.5)/0.5 #to [-1,1]
        
        if self.labels is not None:
            label = {}
            label["y"] = np.array(self.labels[index], dtype=np.int64)
            return (image.astype(np.float32), label)
        else:
            return image.astype(np.float32)

    def __len__(self):
        return self.n_images
