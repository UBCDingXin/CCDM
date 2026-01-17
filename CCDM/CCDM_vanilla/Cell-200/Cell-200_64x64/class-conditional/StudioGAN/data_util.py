# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/data_util.py

import os
import random
import math
import copy

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode
from scipy import io
from PIL import ImageOps, Image
import torch
import torchvision.transforms as transforms
import h5py as h5
import numpy as np
import gc
from tqdm import tqdm


resizer_collection = {"nearest": InterpolationMode.NEAREST,
                      "box": InterpolationMode.BOX,
                      "bilinear": InterpolationMode.BILINEAR,
                      "hamming": InterpolationMode.HAMMING,
                      "bicubic": InterpolationMode.BICUBIC,
                      "lanczos": InterpolationMode.LANCZOS}

class RandomCropLongEdge(object):
    """
    this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
    MIT License
    Copyright (c) 2019 Andy Brock
    """
    def __call__(self, img):
        size = (min(img.size), min(img.size))
        # Only step forward along this edge if it's the long edge
        i = (0 if size[0] == img.size[0] else np.random.randint(low=0, high=img.size[0] - size[0]))
        j = (0 if size[1] == img.size[1] else np.random.randint(low=0, high=img.size[1] - size[1]))
        return transforms.functional.crop(img, j, i, size[0], size[1])

    def __repr__(self):
        return self.__class__.__name__


class CenterCropLongEdge(object):
    """
    this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
    MIT License
    Copyright (c) 2019 Andy Brock
    """
    def __call__(self, img):
        return transforms.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__


class Dataset_(Dataset):
    def __init__(self,
                 data_name,
                 data_dir,
                 train,
                 img_size,
                 num_classes,
                 label_lb,
                 label_ub,
                 stepsize=2,
                 normalize=True,
                 transform=True,
                 crop_long_edge=False,
                 resize_size=None,
                 resizer="lanczos",
                 random_flip=False,
                 hdf5_path=None,
                 load_data_in_memory=False):
        super(Dataset_, self).__init__()
        self.data_name = data_name
        self.data_dir = data_dir
        self.train = train
        self.normalize = normalize
        self.transform = transform
        self.degrees = [0, 90, 180, 270]
        self.img_size = img_size
        self.num_classes = num_classes
        self.label_lb = label_lb
        self.label_ub = label_ub
        self.stepsize = stepsize

        self.load_dataset()

    def load_dataset(self):

        assert self.data_name == "Cell200"

        ## dataset info
        num_classes = self.num_classes
        img_size = self.img_size
        num_imgs_per_label = 10 #num of imgs for each distinct label

        ## modified for ccgan's experiment
        print("Start loading {} h5 file...".format(self.data_name))
        data_filename = self.data_dir+ '/Cell200_{}x{}.h5'.format(img_size, img_size)
        print(data_filename)
        
        with h5.File(data_filename, "r") as hf:
            labels = hf['CellCounts'][:]
            labels = labels.astype(float)
            images = hf['IMGs_grey'][:]
        print("\n Successfully load {} dataset.".format(self.data_name))
        print("\n The loaded dataset has {} images, and the range of labels is [{},{}].".format(len(images),np.min(labels), np.max(labels)))

        raw_labels = copy.deepcopy(labels)

        if self.train: ## in train mode
            # subset of dataset
            print("\n Create a subset for training...")
            selected_labels = np.arange(self.label_lb, self.label_ub+1, self.stepsize)
            n_unique_labels = len(selected_labels)

            # for each label select num_imgs_per_label
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

            print("\r We have {} images for {} distinct labels".format(len(images), n_unique_labels))

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
            unique_labels = np.sort(np.array(list(set(labels_new)))).astype(int)
            print("\n The class labels are \r")
            print(unique_labels)
            assert len(unique_labels) == num_classes

            ## return
            self.num_dataset = images.shape[0]
            self.data = images[:]
            self.labels = labels_new[:].astype(int) #class labels; categorical!
            self.class_cutoff_points = class_cutoff_points

        else: #in eval mode
            
            self.data = images[:]
            self.labels = labels[:] 
            
        return

    def _return_data(self):
        return self.data, self.labels

    def _return_cutoff_points(self):
        return self.class_cutoff_points

    def _return_min_max(self):
        return self.label_lb, self.label_ub

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, label = self.data[index], self.labels[index]
        
        if self.transform:
            assert np.max(image)>1
            image = image[0] #CxWxH ----> WxH
            PIL_im = Image.fromarray(np.uint8(image), mode = 'L')

            degrees = np.array(self.degrees)
            np.random.shuffle(degrees)
            degree = degrees[0]
            PIL_im = PIL_im.rotate(degree)

            if random.random() < 0.5:
                PIL_im = PIL_im.transpose(Image.FLIP_LEFT_RIGHT)

            if random.random() < 0.5:
                PIL_im = PIL_im.transpose(Image.FLIP_TOP_BOTTOM)

            image = np.array(PIL_im)
            image = image[np.newaxis,:,:]

        if self.normalize:
            image = image/255.0
            image = (image-0.5)/0.5

        return (image, label)
