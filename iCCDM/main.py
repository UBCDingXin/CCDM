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
from datetime import datetime 
import yaml

from opts import parse_opts
from models import UNet_EDM, UNet_CCDM, resnet18_aux_regre, DiT_models
from dataset import LoadDataSet
from label_embedding import LabelEmbed
from diffusion import ElucidatedDiffusion
from trainer import Trainer
from utils import get_parameter_number, SimpleProgressBar, IMGs_dataset, compute_entropy, predict_class_labels, cal_std_dask
from evaluation.evaluator import Evaluator


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

# load yaml: detailed network configurations
with open(args.model_config) as f:
    model_config = yaml.safe_load(f)

assert model_config["data"]["image_size"] == args.image_size
assert model_config["data"]["num_channels"] == args.num_channels

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

path_to_fake_data = os.path.join(save_results_folder, 'fake_data')
os.makedirs(path_to_fake_data, exist_ok=True)



#######################################################################################
'''                                Make dataset                                     '''
#######################################################################################

dataset = LoadDataSet(data_name=args.data_name, data_path=args.data_path, min_label=args.min_label, max_label=args.max_label, img_size=args.image_size, max_num_img_per_label=args.max_num_img_per_label, num_img_per_label_after_replica=args.num_img_per_label_after_replica)
    
train_images, train_labels, train_labels_norm = dataset.load_train_data()
num_classes = dataset.num_classes

unique_labels_norm = np.sort(np.array(list(set(train_labels_norm))))

if args.kernel_sigma<0:
    std_label = np.std(train_labels_norm)
    args.kernel_sigma = 1.06*std_label*(len(train_labels_norm))**(-1/5)

    print("\n Use rule-of-thumb formula to compute kernel_sigma >>>")
    print("\n The std of {} labels is {:.4f} so the kernel sigma is {:.4f}".format(len(train_labels_norm), std_label, args.kernel_sigma))
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

print("\n Kappa:{:.4f}".format(args.kappa))


#######################################################################################
'''                                EDM sigma_data                                   '''
#######################################################################################
# calculate the sigma_data for EDM
# output shape: (B)

def _check_label_type(y):
    if isinstance(y, torch.Tensor):
        return "tensor"
    elif isinstance(y, (int, float)):
        return "scalar"
    else:
        raise TypeError("labels `y` must be a scalar or a torch.Tensor.")

if args.edm_sigma_data_type == "default":  # default value for sigma_data

    def fn_y2sigma_data(y):
        kind = _check_label_type(y)
        val = float(args.edm_sigma_data_default)

        if kind == "scalar":
            # Scalar input -> return Python float
            return val
        else:  # tensor
            # Tensor input -> return tensor with same dtype/device/shape as y
            return torch.full_like(y, fill_value=val)

elif args.edm_sigma_data_type == "global":  # global data standard deviation
    sigma_data = float(cal_std_dask(train_images))

    def fn_y2sigma_data(y):
        kind = _check_label_type(y)
        val = sigma_data

        if kind == "scalar":
            return val
        else:  # tensor
            return torch.full_like(y, fill_value=val)

elif args.edm_sigma_data_type == "local":  # y-dependent data standard deviation

    #   y_unique      = [y_(1), ..., y_(M)]
    #   sigma_unique  = [sigma_{y_(1)}, ..., sigma_{y_(M)}]

    sigma_unique = []
    for lbl in unique_labels_norm:
        idx = np.where(train_labels_norm == lbl)[0]
        assert train_images[idx].max()>1.0
        imgs_for_label = (train_images[idx] / 255.0 - 0.5) / 0.5
        per_image_std = np.std(imgs_for_label.reshape(-1))  # standard deviation for this label
        sigma_unique.append(per_image_std)
    sigma_unique = np.array(sigma_unique, dtype=float)
    y_unique_np = np.asarray(unique_labels_norm, dtype=float)

    def fn_y2sigma_data(y, y_unique=y_unique_np, sigma_unique=sigma_unique, shape=None):
        """
        Map label(s) y to the corresponding standard deviation sigma_y using piecewise rules:

        - If y < y_(1): use sigma_{y_(1)}.
        - If y > y_(M): use sigma_{y_(M)}.
        - If y == y_(k): use sigma_{y_(k)}.
        - If y ∈ (y_(k), y_(k+1)): use 0.5 * (sigma_{y_(k)} + sigma_{y_(k+1)}).

        Input:
            y: scalar or torch.Tensor
        Output:
            scalar -> Python float
            tensor -> torch.Tensor with same dtype/device as y
        """
        kind = _check_label_type(y)
        
        len_y_unique = len(y_unique)

        if kind == "scalar":
            # Scalar -> convert to 1D array, apply piecewise rule, convert back to float
            y_np = np.array([float(y)], dtype=float)
            orig_shape = y_np.shape
        else:
            # Tensor -> move to CPU for NumPy operations
            y_np = y.detach().cpu().numpy().astype(float)
            orig_shape = y_np.shape

        y_flat = y_np.ravel()
        idx = np.searchsorted(y_unique, y_flat, side="left")  # interval index
        
        # ensure indices do not go out of bounds
        # np.searchsorted may return len(y_unique), need to constrain to valid range
        idx_clipped = np.minimum(idx, len_y_unique - 1)
        
        res = np.empty_like(y_flat, dtype=float)

        # Case 1: exact matches
        # Use safe index access
        mask_equal = (idx < len_y_unique) & (y_flat == y_unique[idx_clipped])
        res[mask_equal] = sigma_unique[idx_clipped[mask_equal]]

        # Case 2: non-exact matches
        mask_not_equal = ~mask_equal

        # 2.1: y < y_unique[0]
        mask_left = mask_not_equal & (idx == 0)
        res[mask_left] = sigma_unique[0]

        # 2.2: y > y_unique[-1]
        mask_right = mask_not_equal & (idx == len_y_unique)  # idx may equal len_y_unique here
        res[mask_right] = sigma_unique[-1]

        # 2.3: y in between two known points
        # Fix boundary check: ensure idx is within valid range and not a boundary case
        mask_between = mask_not_equal & (idx > 0) & (idx < len_y_unique)
        idx_between = idx[mask_between]
        # For intermediate cases, use idx-1 and idx as indices (both within valid range)
        res[mask_between] = 0.5 * (
            sigma_unique[idx_between - 1] + sigma_unique[idx_between]
        )

        # Reshape to original shape
        res = res.reshape(orig_shape)
        if shape is not None:
            res = res.reshape(shape)

        if kind == "scalar":
            # Return Python float
            return float(res.reshape(-1)[0])
        else:
            # Return tensor with same dtype/device as input tensor
            return torch.from_numpy(res).to(device=y.device, dtype=y.dtype)

else:
    raise ValueError("Invalid data variance type!")




#######################################################################################
'''                             label embedding method                              '''
#######################################################################################

if args.data_name == "UTKFace":
    dataset_embed = LoadDataSet(data_name=args.data_name, data_path=args.data_path, min_label=args.min_label, max_label=args.max_label, img_size=args.image_size, max_num_img_per_label=1e30, num_img_per_label_after_replica=200)
elif args.data_name in ["Cell200", "RC-49"]:
    dataset_embed = LoadDataSet(data_name=args.data_name, data_path=args.data_path, min_label=args.min_label, max_label=args.max_label, img_size=args.image_size, max_num_img_per_label=args.max_num_img_per_label, num_img_per_label_after_replica=0)
else:
    dataset_embed = LoadDataSet(data_name=args.data_name, data_path=args.data_path, min_label=args.min_label, max_label=args.max_label, img_size=args.image_size, max_num_img_per_label=1e30, num_img_per_label_after_replica=0)

label_embedding = LabelEmbed(
                    dataset = dataset_embed, 
                    path_y2h = path_to_output+'/model_y2h', 
                    path_y2cov = path_to_output+'/model_y2cov', 
                    y2h_type = args.y2h_embed_type, 
                    y2cov_type = args.y2cov_embed_type, 
                    y2cov_y2emb_type = args.net_embed_y2cov_y2emb,
                    h_dim = args.dim_embed, 
                    img_size = args.image_size,
                    nc = args.num_channels, 
                    batch_size = 128, 
                    device = "cuda",
                    )
fn_y2h = label_embedding.fn_y2h
if args.use_y2cov:
    fn_y2cov = label_embedding.fn_y2cov
else:
    fn_y2cov = None

#######################################################################################
'''                             Diffusion  training                                 '''
#######################################################################################


#---------------------------
# unet init.

if model_config["name"]=="UNet_CCDM":
    unet_model = UNet_CCDM(**model_config["model"])
elif model_config["name"]=="UNet_EDM":
    unet_model = UNet_EDM(**model_config["model"])
elif model_config["name"]=="DiT":
    unet_model = DiT_models[model_config["setup_name"]](**model_config["model"])
print('Model size:', get_parameter_number(unet_model))


#---------------------------
# auxiliary regression init.
if args.use_aux_reg_loss:
    aux_reg_net = resnet18_aux_regre(nc=args.num_channels)
    path_to_ckpt = os.path.join(path_to_output, "aux_reg_model")
    if args.data_name in ["RC-49_imb"]:
        path_to_ckpt += "/{}".format(args.imb_type)
    path_to_ckpt += "/ckpt_resnet18_epoch_200.pth"
    checkpoint = torch.load(path_to_ckpt, weights_only=True)
    aux_reg_net.load_state_dict(checkpoint['net_state_dict'])
    aux_reg_net.eval()   
else:
    aux_reg_net = None

aux_loss_params = {
    "use_aux_reg_loss":args.use_aux_reg_loss,
    "aux_reg_loss_type": args.aux_reg_loss_type,
    "aux_reg_loss_weight": args.aux_reg_loss_weight,
    "aux_reg_loss_epsilon": args.aux_reg_loss_epsilon,
    "aux_reg_net": aux_reg_net,
}


#---------------------------
# diffusion init.
diffusion_model = ElucidatedDiffusion(
                    net = unet_model,
                    fn_y2sigma_data = fn_y2sigma_data,        # function for coverting y to data's std
                    aux_loss_params = aux_loss_params,
                    image_size = args.image_size,
                    channels = args.num_channels,
                    num_sample_steps = args.num_sample_steps, # number of sampling steps
                    sigma_data_default = args.edm_sigma_data_default, 
                    sigma_min = args.edm_sigma_min,     # min noise level
                    sigma_max = args.edm_sigma_max,        # max noise level
                    rho = args.edm_rho,               # controls the sampling schedule
                    P_mean = args.edm_P_mean,         # mean of log-normal distribution from which noise is drawn for training
                    P_std = args.edm_P_std,           # standard deviation of log-normal distribution from which noise is drawn for training
                    S_churn = args.edm_S_churn,          # parameters for stochastic sampling - depends on dataset
                    S_tmin = args.edm_S_tmin,
                    S_tmax = args.edm_S_tmax,
                    S_noise = args.edm_S_noise, 
                    cond_drop_prob = model_config["model"]["cond_drop_prob"],  
                    use_y2cov = args.use_y2cov, 
                    fn_y2cov = fn_y2cov, 
                    y2cov_hy_weight_train = args.y2cov_hy_weight_train,
                    y2cov_hy_weight_test = args.y2cov_hy_weight_test,
                )


#---------------------------
# trainer init.

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
        "use_ada_vic":args.use_ada_vic,
        "ada_vic_type":args.ada_vic_type,
        "min_n_per_vic": args.min_n_per_vic,
        "ada_eps":1e-5,
        "use_symm_vic": args.use_symm_vic,
    }

trainer = Trainer(
        diffusion_model=diffusion_model,
        fn_y2h=fn_y2h,
        data_name=args.data_name,
        train_images=train_images,
        train_labels=train_labels_norm,
        vicinal_params=vicinal_params,
        train_batch_size = args.train_batch_size,
        gradient_accumulate_every = args.gradient_accumulate_every,
        train_lr = args.train_lr,
        train_num_steps = args.train_num_steps,
        ema_update_after_step = args.ema_update_after_step,
        ema_update_every = args.ema_update_every,
        ema_decay = args.ema_decay,
        adam_betas = (args.opt_adam_beta1, args.opt_adam_beta2),
        save_every = args.save_every,
        sample_every = args.sample_every,
        y_visual = y_visual,
        cond_scale_visual = args.sample_cond_scale,
        cond_rescaled_phi_visual = args.sample_cond_rescaled_phi,
        results_folder = save_results_folder,
        amp = args.train_amp,
        mixed_precision_type = args.train_mixed_precision,
        max_grad_norm = 1.,
    )
    
if args.resume_step>0:
    trainer.load(args.resume_step)
trainer.train()




#######################################################################################
'''                         Sampling and evaluation                                 '''
#######################################################################################

print("\n Start sampling fake images from the model >>>")

## initialize evaluator
evaluator = Evaluator(dataset=dataset, trainer=trainer, args=args, device=trainer.device, save_results_folder=save_results_folder) 

## initialize evaluation models, prepare for evaluation
if args.data_name in ["RC-49","RC-49_imb"]:
    eval_data_name = "RC49"
else:
    eval_data_name = args.data_name
conduct_import_codes = "from evaluation.eval_models.{}.metrics_{}x{} import ResNet34_class_eval, ResNet34_regre_eval, encoder".format(eval_data_name, args.image_size, args.image_size)
print("\r"+conduct_import_codes)
exec(conduct_import_codes)
# for FID
PreNetFID = encoder(dim_bottleneck=512)
PreNetFID = nn.DataParallel(PreNetFID)
# for Diversity
if args.data_name in ["UTKFace", "RC-49", "RC-49_imb", "SteeringAngle"]:
    PreNetDiversity = ResNet34_class_eval(num_classes=num_classes, ngpu = torch.cuda.device_count())
else:
    PreNetDiversity = None
# for LS
PreNetLS = ResNet34_regre_eval(ngpu = torch.cuda.device_count())

# ## dump fake data in h5 files
# if args.dump_fake_for_h5:
#     path_to_h5files = os.path.join(path_to_fake_data, 'h5')
#     os.makedirs(path_to_h5files, exist_ok=True)
#     evaluator.dump_h5_files(output_path=path_to_h5files)

## dump for niqe computation
if args.dump_fake_for_niqe:
    if args.niqe_dump_path=="None":
        dump_fake_images_folder = os.path.join(path_to_fake_data, 'png')
    else:
        dump_fake_images_folder = args.niqe_dump_path + '/fake_images'
    os.makedirs(dump_fake_images_folder, exist_ok=True)
    evaluator.dump_png_images(output_path=dump_fake_images_folder)

## start computing evaluation metrics
if args.do_eval:
    now = datetime.now()
    time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    eval_results_path = os.path.join(save_setting_folder, "eval_{}".format(time_str))
    os.makedirs(eval_results_path, exist_ok=True)
    evaluator.compute_metrics(eval_results_path, PreNetFID, PreNetDiversity, PreNetLS)



print("\n===================================================================================================")