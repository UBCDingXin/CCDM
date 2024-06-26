import argparse
import copy
import gc
import numpy as np
import h5py
import glob
import os
import shutil
import random
from tqdm import tqdm, trange
from PIL import Image
import platform
import matplotlib.pyplot as plt
plt.switch_backend('agg')

parser = argparse.ArgumentParser(description='separate fake ages according to ages.')
parser.add_argument('--imgs_dir', type=str, default='', help='imgs dir.')
parser.add_argument('--out_dir_base', type=str, default='./fake_data', help='output dir.')
parser.add_argument('--dataset_name', type=str, default='RC49_badfake')
parser.add_argument('--img_size', type=int, default=64)
parser.add_argument('--quantile', type=float, default=0.9)
args = parser.parse_args()



if platform.system().lower()=="linux":
    split_symbol = '/'
elif platform.system().lower()=="windows":
    split_symbol = '\\'
else:
    raise ValueError('Do not support!!!')


def get_file_list(dataset_dir):
    file_list = glob.glob(os.path.join(dataset_dir, '*.png')) + glob.glob(os.path.join(dataset_dir, '*.jpg'))
    file_list.sort()
    return file_list


assert float((args.imgs_dir).split("_")[-1])==args.quantile

img_lists = os.listdir(args.imgs_dir)
n = len(img_lists)

fake_images = []
fake_labels = []
for i in trange(n):
    path_to_imgs_i = os.path.join(args.imgs_dir, img_lists[i])
    img_lists_i = os.listdir(path_to_imgs_i)

    for j in range(len(img_lists_i)):
        filename_i_j = img_lists_i[j]
        label_i_j = filename_i_j.split('_')
        label_i_j = label_i_j[-1].split('.png')[0]
        assert np.abs(float(label_i_j) - float(img_lists[i]))<1e-8

        path_to_img_i_j = os.path.join(path_to_imgs_i, filename_i_j)

        img_ij = Image.open(path_to_img_i_j).convert('RGB') # 读取图片

        imgData_ij = np.array(img_ij) # 将对象img转化为RGB像素值矩阵
        imgData_ij = np.transpose(imgData_ij, axes=[2,0,1]) #h,w,c --> c,h,w
        imgData_ij = imgData_ij[np.newaxis,:,:,:]

        fake_images.append(imgData_ij)
        fake_labels.append(float(label_i_j))
    ##end for j
##end for i
fake_images = np.concatenate(fake_images, axis=0)
fake_labels = np.array(fake_labels)

print(fake_images.shape)
print(fake_labels.shape)

h5py_file = args.out_dir_base + '/badfake_NIQE{}_nfake{}.h5'.format(args.quantile, len(fake_images))
with h5py.File(h5py_file, "w") as f:
    f.create_dataset('fake_images', data = fake_images, dtype='uint8', compression="gzip", compression_opts=9) #压缩率越大，数据集越小
    f.create_dataset('fake_labels', data = fake_labels, dtype='float', compression="gzip")