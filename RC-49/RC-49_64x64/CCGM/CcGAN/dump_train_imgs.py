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

### import my stuffs ###
from opts import parse_opts
args = parse_opts()
wd = args.root_path
os.chdir(wd)
from utils import IMGs_dataset, compute_entropy, predict_class_labels
from models import *
from train_ccgan import train_ccgan, sample_ccgan_given_labels
from train_net_for_label_embed import train_net_embed, train_net_y2h
from eval_metrics import cal_FID, cal_labelscore, inception_score


#######################################################################################
'''                                   Settings                                      '''
#######################################################################################

## set manually
path_to_dump = "F:/LocalWD/CcGAN_TPAMI_NIQE/RC-49/NIQE_filter_64x64/real_data/RC-49_images_all_0_90/angles"
os.makedirs(path_to_dump, exist_ok=True)

#-------------------------------
# seeds
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
cudnn.benchmark = False
np.random.seed(args.seed)     

#-------------------------------
# Embedding
base_lr_x2y = 0.01
base_lr_y2h = 0.01

#-------------------------------
# sampling parameters
assert args.eval_mode in [1,2,3,4] #evaluation mode must be in 1,2,3,4
if args.data_split == "all":
    args.eval_mode != 1

#-------------------------------
# some functions
def fn_norm_labels(labels):
    '''
    labels: unnormalized labels
    '''
    return labels/args.max_label

def fn_denorm_labels(labels):
    '''
    labels: normalized labels
    '''
    if isinstance(labels, np.ndarray):
        return labels*args.max_label
    elif torch.is_tensor(labels):
        return labels*args.max_label
    else:
        return labels*args.max_label


#######################################################################################
'''                                    Data loader                                 '''
#######################################################################################
# data loader
data_filename = args.data_path + '/RC-49_{}x{}.h5'.format(args.img_size, args.img_size)
hf = h5py.File(data_filename, 'r')
labels_all = hf['labels'][:]
labels_all = labels_all.astype(float)
images_all = hf['images'][:]
indx_train = hf['indx_train'][:]
hf.close()
print("\n RC-49 dataset shape: {}x{}x{}x{}".format(images_all.shape[0], images_all.shape[1], images_all.shape[2], images_all.shape[3]))

# data split
if args.data_split == "train":
    images_train = images_all[indx_train]
    labels_train_raw = labels_all[indx_train]
else:
    images_train = copy.deepcopy(images_all)
    labels_train_raw = copy.deepcopy(labels_all)

# only take images with label in (q1, q2)
q1 = args.min_label
q2 = args.max_label
indx = np.where((labels_train_raw>q1)*(labels_train_raw<q2)==True)[0]
labels_train_raw = labels_train_raw[indx]
images_train = images_train[indx]
assert len(labels_train_raw)==len(images_train)

if args.comp_FID or args.niqe_filter:
    indx = np.where((labels_all>q1)*(labels_all<q2)==True)[0]
    labels_all = labels_all[indx]
    images_all = images_all[indx]
    assert len(labels_all)==len(images_all)


# for each angle, take no more than args.max_num_img_per_label images
image_num_threshold = args.max_num_img_per_label
print("\n Original set has {} images; For each angle, take no more than {} images>>>".format(len(images_train), image_num_threshold))
unique_labels_tmp = np.sort(np.array(list(set(labels_train_raw))))
for i in tqdm(range(len(unique_labels_tmp))):
    indx_i = np.where(labels_train_raw == unique_labels_tmp[i])[0]
    if len(indx_i)>image_num_threshold:
        np.random.shuffle(indx_i)
        indx_i = indx_i[0:image_num_threshold]
    if i == 0:
        sel_indx = indx_i
    else:
        sel_indx = np.concatenate((sel_indx, indx_i))
images_train = images_train[sel_indx]
labels_train_raw = labels_train_raw[sel_indx]
print("{} images left and there are {} unique labels".format(len(images_train), len(set(labels_train_raw))))



unique_labels = np.sort(np.array(list(set(labels_train_raw))))
print(unique_labels)

for i in trange(len(unique_labels)):
    label_i = unique_labels[i]
    path_to_dump_i = os.path.join(path_to_dump, str(label_i))
    os.makedirs(path_to_dump_i, exist_ok=True)

    indx_i = np.where(labels_train_raw==label_i)[0]
    images_train_i = images_train[indx_i]

    for j in range(len(images_train_i)):
        path_to_i_j = os.path.join(path_to_dump_i, "{}_{}.png".format(j, label_i))
        img_i_j = images_train_i[j].astype(np.uint8)
        img_i_j_pil = Image.fromarray(img_i_j.transpose(1,2,0))
        img_i_j_pil.save(path_to_i_j)
    ##end for j

##end for i












