print("\n===================================================================================================")

import os
import math
from abc import abstractmethod
import random
import sys

from PIL import Image
import requests
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py
import gc
import copy
import timeit

from models import Unet
from dataset import LoadDataSet
from label_embedding import LabelEmbed
from diffusion import GaussianDiffusion
from trainer import Trainer
from opts import parse_opts
from utils import get_parameter_number, SimpleProgressBar, IMGs_dataset, compute_entropy, predict_class_labels

##############################################
''' Settings '''
args = parse_opts()

# seeds
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
cudnn.benchmark = False
np.random.seed(args.seed)

if args.torch_model_path!="None":
    os.environ['TORCH_HOME']=args.torch_model_path

# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True 


#######################################################################################
'''                                Output folders                                  '''
#######################################################################################
path_to_output = os.path.join(args.root_path, 'output/{}_{}'.format(args.data_name, args.image_size))
os.makedirs(path_to_output, exist_ok=True)

save_setting_folder = os.path.join(path_to_output, "{}".format(args.setting_name))
os.makedirs(save_setting_folder, exist_ok=True)

setting_log_file = os.path.join(save_setting_folder, 'setting_info.txt')
if not os.path.isfile(setting_log_file):
    logging_file = open(setting_log_file, "w")
    logging_file.close()
with open(setting_log_file, 'a') as logging_file:
    logging_file.write("\n===================================================================================================")
    print(args, file=logging_file)

save_results_folder = os.path.join(save_setting_folder, 'results')
os.makedirs(save_results_folder, exist_ok=True)



#######################################################################################
'''                                Make dataset                                     '''
#######################################################################################

dataset = LoadDataSet(data_name=args.data_name, data_path=args.data_path, min_label=args.min_label, max_label=args.max_label, img_size=args.image_size, max_num_img_per_label=args.max_num_img_per_label, num_img_per_label_after_replica=args.num_img_per_label_after_replica)
    
train_images, train_labels, train_labels_norm = dataset.load_train_data()

unique_labels_norm = np.sort(np.array(list(set(train_labels_norm))))


if args.kernel_sigma<0:
    std_label = np.std(train_labels_norm)
    args.kernel_sigma = 1.06*std_label*(len(train_labels_norm))**(-1/5)

    print("\n Use rule-of-thumb formula to compute kernel_sigma >>>")
    print("\r The std of {} labels is {:.4f} so the kernel sigma is {:.4f}".format(len(train_labels_norm), std_label, args.kernel_sigma))
##end if

if args.kappa<0:
    n_unique = len(unique_labels_norm)

    diff_list = []
    for i in range(1,n_unique):
        diff_list.append(unique_labels_norm[i] - unique_labels_norm[i-1])
    kappa_base = np.abs(args.kappa)*np.max(np.array(diff_list))

    if args.threshold_type=="hard":
        args.kappa = kappa_base
    else:
        args.kappa = 1/kappa_base**2
##end if

print("\r Kappa:{:.4f}".format(args.kappa))



#######################################################################################
'''                             label embedding method                              '''
#######################################################################################

if args.data_name == "UTKFace":
    dataset_embed = LoadDataSet(data_name=args.data_name, data_path=args.data_path, min_label=args.min_label, max_label=args.max_label, img_size=args.image_size, max_num_img_per_label=1e30, num_img_per_label_after_replica=200)
elif args.data_name in ["Cell200", "RC-49"]:
    dataset_embed = LoadDataSet(data_name=args.data_name, data_path=args.data_path, min_label=args.min_label, max_label=args.max_label, img_size=args.image_size, max_num_img_per_label=args.max_num_img_per_label, num_img_per_label_after_replica=0)
else:
    dataset_embed = LoadDataSet(data_name=args.data_name, data_path=args.data_path, min_label=args.min_label, max_label=args.max_label, img_size=args.image_size, max_num_img_per_label=1e30, num_img_per_label_after_replica=0)

label_embedding = LabelEmbed(dataset=dataset_embed, path_y2h=path_to_output+'/model_y2h', path_y2cov=path_to_output+'/model_y2cov', y2h_type=args.y2h_embed_type, y2cov_type=args.y2cov_embed_type, h_dim = args.dim_embed, cov_dim = args.image_size**2*args.num_channels, nc=args.num_channels)
fn_y2h = label_embedding.fn_y2h
fn_y2cov = label_embedding.fn_y2cov



#######################################################################################
'''                             Diffusion  training                                 '''
#######################################################################################

channel_mult = (args.channel_mult).split("_")
channel_mult = [int(dim) for dim in channel_mult]
# print(channel_mult)

model = Unet(
        dim=args.model_channels,
        embed_input_dim=args.dim_embed, #embedding dim of regression label
        cond_drop_prob = args.cond_drop_prob,
        dim_mults=channel_mult,
        in_channels = args.num_channels,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        attn_dim_head = args.attn_dim_head,
        attn_heads = args.num_heads,
    )
model = nn.DataParallel(model)
print('\r model size:', get_parameter_number(model))


## build diffusion process
diffusion = GaussianDiffusion(
    model,
    use_Hy=args.use_Hy, # use y dependent covariance matrix
    fn_y2cov=fn_y2cov,
    cond_drop_prob = args.cond_drop_prob,
    image_size = args.image_size,
    timesteps = args.train_timesteps,
    sampling_timesteps = args.sample_timesteps,
    objective = args.pred_objective,
    beta_schedule = args.beta_schedule,
    ddim_sampling_eta = args.ddim_eta,
).cuda()



## for visualization
if args.image_size>128:
    n_row=6
else:
    n_row=10
n_col = n_row
start_label = np.quantile(train_labels_norm, 0.05)
end_label = np.quantile(train_labels_norm, 0.95)
selected_labels = np.linspace(start_label, end_label, num=n_row)
y_visual = np.zeros(n_row*n_col)
for i in range(n_row):
    curr_label = selected_labels[i]
    for j in range(n_col):
        y_visual[i*n_col+j] = curr_label
y_visual = torch.from_numpy(y_visual).type(torch.float).view(-1).cuda()
print(y_visual)


## for training
vicinal_params = {
    "kernel_sigma": args.kernel_sigma,
    "kappa": args.kappa,
    "threshold_type": args.threshold_type,
    "nonzero_soft_weight_threshold": args.nonzero_soft_weight_threshold,
}

trainer = Trainer(
    data_name=args.data_name,
    diffusion_model=diffusion,
    train_images=train_images,
    train_labels=train_labels_norm,
    vicinal_params=vicinal_params,
    train_batch_size = args.train_batch_size,
    gradient_accumulate_every = args.gradient_accumulate_every,
    train_lr = args.train_lr,
    train_num_steps = args.niters,
    ema_update_after_step = 100, #int(args.niters*0.1)
    ema_update_every = 10,
    ema_decay = 0.995,
    adam_betas = (0.9, 0.99),
    sample_every = args.sample_every,
    save_every = args.save_every,
    results_folder = save_results_folder,
    amp = args.train_amp,
    mixed_precision_type = 'fp16',
    split_batches = True,
    max_grad_norm = 1.,
    y_visual = y_visual,
    nrow_visual = n_col,
    cond_scale_visual=args.sample_cond_scale
)
if args.resume_niter>0:
    trainer.load(args.resume_niter)
trainer.train(fn_y2h=fn_y2h)



#######################################################################################
'''                                Sampling                                        '''
#######################################################################################

print("\n Start sampling {} fake images per label from the model >>>".format(args.nfake_per_label))

## get evaluation labels
_, _, eval_labels = dataset.load_evaluation_data()

num_eval_labels = len(eval_labels)
print(eval_labels)


###########################################
''' multiple h5 files '''

dump_fake_images_folder = os.path.join(save_results_folder, 'fake_data_niters{}_nfake{}_{}_sampstep{}'.format(args.niters, int(args.nfake_per_label*num_eval_labels), args.sampler, args.sample_timesteps))
os.makedirs(dump_fake_images_folder, exist_ok=True)

fake_images = []
fake_labels = []
total_sample_time = 0
for i in range(num_eval_labels):
    print('\n [{}/{}]: Generating fake data for label {}...'.format(i+1, num_eval_labels, eval_labels[i]))
    curr_label = eval_labels[i]
    dump_fake_images_filename = os.path.join(dump_fake_images_folder, '{}.h5'.format(curr_label))
    if not os.path.isfile(dump_fake_images_filename):
        fake_labels_i = curr_label*np.ones(args.nfake_per_label)
        start = timeit.default_timer()
        fake_images_i, _ = trainer.sample_given_labels(given_labels = dataset.fn_normalize_labels(fake_labels_i), fn_y2h=fn_y2h, batch_size = args.samp_batch_size, denorm=True, to_numpy=True, verbose=False, sampler=args.sampler, cond_scale=args.sample_cond_scale, sample_timesteps=args.sample_timesteps, ddim_eta=args.ddim_eta)
        stop = timeit.default_timer()
        assert len(fake_images_i)==len(fake_labels_i)
        sample_time_i = stop-start
        if args.dump_fake_data:
            with h5py.File(dump_fake_images_filename, "w") as f:
                f.create_dataset('fake_images_i', data = fake_images_i, dtype='uint8', compression="gzip", compression_opts=6)
                f.create_dataset('fake_labels_i', data = fake_labels_i, dtype='float')
                f.create_dataset('sample_time_i', data = np.array([sample_time_i]), dtype='float')
    else:
        with h5py.File(dump_fake_images_filename, "r") as f:
            fake_images_i = f['fake_images_i'][:]
            fake_labels_i = f['fake_labels_i'][:]
            sample_time_i = f['sample_time_i'][0]
        assert len(fake_images_i) == len(fake_labels_i)
    ##end if
    total_sample_time+=sample_time_i
    fake_images.append(fake_images_i)
    fake_labels.append(fake_labels_i)
    print("\r {}/{}: Got {} fake images for label {}. Time spent {:.2f}, Total time {:.2f}.".format(i+1, num_eval_labels, len(fake_images_i), curr_label, sample_time_i, total_sample_time))
    
    ## dump some imgs for visualization
    img_vis_i = fake_images_i[0:36]/255.0
    img_vis_i = torch.from_numpy(img_vis_i)
    img_filename = os.path.join(dump_fake_images_folder, 'sample_{}.png'.format(curr_label))
    torchvision.utils.save_image(img_vis_i.data, img_filename, nrow=6, normalize=False)
    del fake_images_i, fake_labels_i; gc.collect()
    
##end for i

fake_images = np.concatenate(fake_images, axis=0)
fake_labels = np.concatenate(fake_labels)
print("Sampling finished; Time elapses: {}s".format(total_sample_time))


### dump for computing NIQE
if args.dump_fake_for_NIQE:
    print("\n Dumping fake images for NIQE...")
    if args.niqe_dump_path=="None":
        dump_fake_images_folder = save_setting_folder + '/fake_images'
    else:
        dump_fake_images_folder = args.niqe_dump_path + '/fake_images'
    os.makedirs(dump_fake_images_folder, exist_ok=True)
    for i in tqdm(range(len(fake_images))):
        label_i = fake_labels[i]
        filename_i = dump_fake_images_folder + "/{}_{:.1f}.png".format(i, label_i)
        os.makedirs(os.path.dirname(filename_i), exist_ok=True)
        image_i = fake_images[i]
        image_i_pil = Image.fromarray(image_i.transpose(1,2,0))
        image_i_pil.save(filename_i)
    #end for i
    sys.exit()




print("\n===================================================================================================")