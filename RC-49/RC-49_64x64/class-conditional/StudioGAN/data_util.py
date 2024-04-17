# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/data_util.py

import os
import random
import math

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
                 crop_long_edge=False,
                 resize_size=None,
                 resizer="lanczos",
                 random_flip=False,
                 normalize=True,
                 hdf5_path=None,
                 load_data_in_memory=False):
        super(Dataset_, self).__init__()
        self.data_name = data_name
        self.data_dir = data_dir
        self.train = train
        self.normalize = normalize
        # self.random_flip = random_flip
        self.random_flip = False #for SA, RC49
        self.img_size = img_size
        self.num_classes = num_classes
        self.label_lb = label_lb
        self.label_ub = label_ub

        self.load_dataset()

    def load_dataset(self):

        assert self.data_name == "RC49"

        ## dataset info
        q1 = self.label_lb #lower bound for labels
        q2 = self.label_ub #upper bound for labels
        num_classes = self.num_classes
        img_size = self.img_size
        image_num_threshold = 25 #num of imgs for each distinct label; 25 for RC-49
        max_num_img_per_label_after_replica=0
        # max_num_img_per_label=1e30

        ## modified for ccgan's experiment
        print("Start loading {} h5 file...".format(self.data_name))
        data_filename = self.data_dir+ '/RC-49_{}x{}.h5'.format(img_size, img_size)
        print(data_filename)
        
        with h5.File(data_filename, "r") as hf:
            labels = hf['labels'][:]
            labels = labels.astype(float)
            images = hf['images'][:]
            indx_train = hf['indx_train'][:]
        print("\n Successfully load {} dataset.".format(self.data_name))

        print("\n The loaded dataset has {} images, and the range of labels is [{},{}].".format(len(images),np.min(labels), np.max(labels)))

        if self.train: ## in train mode
            #data for training
            images = images[indx_train]
            labels = labels[indx_train]

            # subset of dataset
            # only take images with label in (q1, q2)
            indx = np.where((labels>q1)*(labels<q2)==True)[0]
            labels = labels[indx]
            images = images[indx]
            assert len(labels)==len(images)
            
            # for each label, take no more than max_num_img_per_label images
            print("\n The original dataset has {} images. For each label, take no more than {} images>>>".format(len(images), image_num_threshold))
            unique_labels_tmp = np.sort(np.array(list(set(labels))))
            for i in tqdm(range(len(unique_labels_tmp))):
                indx_i = np.where(labels == unique_labels_tmp[i])[0]
                if len(indx_i)>image_num_threshold:
                    np.random.shuffle(indx_i)
                    indx_i = indx_i[0:image_num_threshold]
                if i == 0:
                    sel_indx = indx_i
                else:
                    sel_indx = np.concatenate((sel_indx, indx_i))
            images = images[sel_indx]
            labels = labels[sel_indx]
            print("\r {} images left.".format(len(images)))


            ## replicate minority samples to alleviate the imbalance
            max_num_img_per_label_after_replica = np.min([max_num_img_per_label_after_replica, image_num_threshold])
            if max_num_img_per_label_after_replica>1:
                unique_labels_replica = np.sort(np.array(list(set(labels))))
                num_labels_replicated = 0
                print("\n Start replicating monority samples >>>")
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
                print("\r We replicate {} images and labels.".format(len(images_replica)))
                del images_replica, labels_replica; gc.collect()
            

            ### convert regression labels into class labels
            unique_labels = np.sort(np.array(list(set(labels))))
            num_unique_labels = len(unique_labels)
            print("{} unique labels are split into {} classes".format(num_unique_labels, num_classes))

            ### step 1: prepare two dictionaries
            label2class = dict()
            class2label = dict()
            num_labels_per_class = num_unique_labels//num_classes
            class_cutoff_points = [unique_labels[0]] #the cutoff points on [min_label, max_label] to determine classes
            curr_class = 0
            for i in range(num_unique_labels):
                label2class[unique_labels[i]]=curr_class
                if (i+1)%num_labels_per_class==0 and (curr_class+1)!=num_classes:
                    curr_class += 1
                    class_cutoff_points.append(unique_labels[i+1])
            class_cutoff_points.append(unique_labels[-1])
            assert len(class_cutoff_points)-1 == num_classes

            for i in range(num_classes):
                class2label[i] = (class_cutoff_points[i]+class_cutoff_points[i+1])/2

            ### step 2: convert angles to class labels
            labels_new = -1*np.ones(len(labels))
            for i in range(len(labels)):
                labels_new[i] = label2class[labels[i]]
            assert np.sum(labels_new<0)==0
            unique_labels = np.sort(np.array(list(set(labels_new)))).astype(int)
            assert len(unique_labels) == num_classes
            print(unique_labels)

            ## return
            self.num_dataset = images.shape[0]
            self.data = images[:]
            self.labels = labels_new[:].astype(int) #class labels; categorical!
            self.class_cutoff_points = class_cutoff_points

        else: #in eval mode
            
            # subset of dataset
            # only take images with label in (q1, q2)
            indx = np.where((labels>q1)*(labels<q2)==True)[0]
            labels = labels[indx]
            images = images[indx]
            assert len(labels)==len(images)

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
        img, label = self.data[index], self.labels[index]
        if self.normalize:
            img = img/255.0
            img = (img-0.5)/0.5

        # if self.random_flip and random.random() < 0.5:
        #     img = np.flip(img, axis=2)
        #     img = np.ascontiguousarray(img)

        return (img, label)
