import math
import random
import os
import gc

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
    min_label=1,
    max_label=60,
    num_classes=60, 
    class_cond=True,
    deterministic=False,
    random_flip=True,
    max_num_img_per_label=10000,
    max_num_img_per_label_after_replica=0,
    num_workers=0,
    random_crop=False,
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
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    # load data from the h5 file
    data_filename = data_dir + '/UTKFace_Wild_{}x{}.h5'.format(image_size, image_size)
    hf = h5py.File(data_filename, 'r')
    labels = hf['ages'][:]
    labels = labels.astype(float)
    images = hf['images'][:]
    hf.close()

    # subset of UTKFace
    selected_labels = np.arange(min_label, max_label+1)
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
    
    # for each label, take no more than max_num_img_per_label images
    print("\n Original set has {} images; For each label, take no more than {} images>>>".format(len(images), max_num_img_per_label))
    unique_labels_tmp = np.sort(np.array(list(set(labels))))
    for i in tqdm(range(len(unique_labels_tmp))):
        indx_i = np.where(labels == unique_labels_tmp[i])[0]
        if len(indx_i)>max_num_img_per_label:
            np.random.shuffle(indx_i)
            indx_i = indx_i[0:max_num_img_per_label]
        if i == 0:
            sel_indx = indx_i
        else:
            sel_indx = np.concatenate((sel_indx, indx_i))
    images = images[sel_indx]
    labels = labels[sel_indx]
    print("{} images left.".format(len(images)))

    ## replicate minority samples to alleviate the imbalance
    max_num_img_per_label_after_replica = np.min([max_num_img_per_label_after_replica, max_num_img_per_label])
    if max_num_img_per_label_after_replica>1:
        unique_labels_replica = np.sort(np.array(list(set(labels))))
        num_labels_replicated = 0
        print("Start replicating minority samples >>>")
        for i in tqdm(range(len(unique_labels_replica))):
            # print((i, num_labels_replicated))
            curr_label = unique_labels_replica[i]
            indx_i = np.where(labels == curr_label)[0]
            if len(indx_i) < max_num_img_per_label_after_replica:
                num_img_less = max_num_img_per_label_after_replica - len(indx_i)
                indx_replica = np.random.choice(indx_i, size = num_img_less, replace=True)
                if num_labels_replicated == 0:
                    images_replica = images[indx_replica]
                    labels_replica = labels[indx_replica]
                else:
                    images_replica = np.concatenate((images_replica, images[indx_replica]), axis=0)
                    labels_replica = np.concatenate((labels_replica, labels[indx_replica]))
                num_labels_replicated+=1
        #end for i
        images = np.concatenate((images, images_replica), axis=0)
        labels = np.concatenate((labels, labels_replica))
        print("We replicate {} images and labels \n".format(len(images_replica)))
        del images_replica, labels_replica; gc.collect()

    max_label = np.max(labels)

    unique_labels = np.sort(np.array(list(set(labels))))
    num_unique_labels = len(unique_labels)
    print("{} unique labels are split into {} classes".format(num_unique_labels, num_classes))

    ## convert ages to class labels and vice versa
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

    ### the cell label of each interval equals to the average of the two end points
    for i in range(num_classes):
        class2label[i] = (class_cutoff_points[i]+class_cutoff_points[i+1])/2

    ### step 2: convert ages to class labels
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
        trainset = IMGs_dataset(images, labels, normalize=True, random_flip=random_flip)
    else:
        trainset = IMGs_dataset(images, labels=None, normalize=True, random_flip=random_flip)
    
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
    def __init__(self, images, labels=None, normalize=False, random_flip=True):
        super(IMGs_dataset, self).__init__()

        self.images = images
        self.n_images = len(self.images)
        self.labels = labels
        if labels is not None:
            if len(self.images) != len(self.labels):
                raise Exception('images (' +  str(len(self.images)) +') and labels ('+str(len(self.labels))+') do not have the same length!!!')
        self.normalize = normalize
        self.random_flip = random_flip

    def __getitem__(self, index):

        image = self.images[index]

        if self.normalize:
            image = image/255.0
            image = (image-0.5)/0.5 #to [-1,1]
        
        if self.random_flip and random.random() < 0.5:
            # image = np.transpose(image, axes=[1,2,0]) #CWH -> WHC
            # image = np.fliplr(image) #flip horizontally
            # image = np.transpose(image, axes=[2,0,1]) #CWH <- WHC
            image = np.flip(image, axis=2)
 
        # if self.labels is not None:
        #     label = self.labels[index]
        #     return (image.astype(np.float32), label)
        # else:
        #     return image.astype(np.float32)

        if self.labels is not None:
            label = {}
            label["y"] = np.array(self.labels[index], dtype=np.int64)
            return (image.astype(np.float32), label)
        else:
            return image.astype(np.float32)

    def __len__(self):
        return self.n_images
