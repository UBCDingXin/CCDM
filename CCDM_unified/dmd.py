'''
 The implementation of DMD2 (Improved Distribution Matching Distillation for Fast Image Synthesis) aims to enhance the sampling efficiency of CCDM.
 Acknowledgement:
    1. https://arxiv.org/abs/2405.14867
    2. https://github.com/tianweiy/DMD2 
 
 We have adapted the original DMD2 algorithm to be compatible with the CCGM task.
 
 !!!!!!!!!! Current Discriminator does not take timesteps as input !!!!
 
'''

print("\n===================================================================================================")

##################################################################################################################
## Load modules
import os
import math
from abc import abstractmethod
import random
import sys

from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py
import gc
import copy
import timeit
from pathlib import Path

from PIL import Image
import requests
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.nn.init import xavier_uniform_
from torchvision import datasets, transforms, utils
from torchvision.utils import save_image


from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from opts import parse_opts_dmd2
from models import Unet, sagan_generator, sagan_discriminator, sngan_generator, sngan_discriminator
from dataset import LoadDataSet
from label_embedding import LabelEmbed
from diffusion import GaussianDiffusion
from trainer import Trainer
from utils import get_parameter_number, SimpleProgressBar, IMGs_dataset, compute_entropy, predict_class_labels, default, identity, unnormalize_to_zero_to_one, normalize_to_neg_one_to_one, cycle, divisible_by, exists, normalize_images, random_hflip, random_rotate, random_vflip
from DiffAugment_pytorch import DiffAugment


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True 

#######################################################################################
'''                                  Settings                                      '''
#######################################################################################

args = parse_opts_dmd2()

# seeds
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
cudnn.benchmark = False
np.random.seed(args.seed)


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
'''                              Load teacher Unet                                  '''
#######################################################################################

channel_mult = (args.channel_mult).split("_")
channel_mult = [int(dim) for dim in channel_mult]

real_unet = Unet(
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
real_unet = nn.DataParallel(real_unet)

diffusion = GaussianDiffusion(
    real_unet,
    use_Hy=args.use_Hy, # use y dependent covariance matrix
    fn_y2cov=fn_y2cov,
    cond_drop_prob = args.cond_drop_prob,
    image_size = args.image_size,
    timesteps = args.train_timesteps,
    sampling_timesteps = 150,
    objective = 'pred_x0',
    beta_schedule = 'cosine',
    ddim_sampling_eta = 1.5,
)

vicinal_params = {
    "kernel_sigma": args.kernel_sigma,
    "kappa": args.kappa,
    "threshold_type": args.threshold_type,
    "nonzero_soft_weight_threshold": args.nonzero_soft_weight_threshold,
}

trainer_t = Trainer(
    data_name=args.data_name,
    diffusion_model = diffusion,
    train_images = train_images,
    train_labels = train_labels_norm,
    vicinal_params= vicinal_params,
    train_batch_size = 16,
    gradient_accumulate_every = 1,
    train_lr = 1e-4,
    train_num_steps = args.niters_t,
    ema_update_after_step = 100,
    ema_update_every = 10,
    ema_decay = 0.995,
    adam_betas = (0.9, 0.99),
    results_folder = args.teacher_ckpt_path,
    amp = args.train_amp,
    mixed_precision_type = 'fp16',
    split_batches = True,
    max_grad_norm = 10.,
    y_visual = None,
    cond_scale_visual=1.5
)
real_unet = trainer_t.load(args.niters_t, return_unet=True) #load teacher model, i.e., real_unet
real_unet = real_unet.module
real_unet.requires_grad_(False)
del diffusion, trainer_t
torch.cuda.empty_cache()
print('\r real_unet size:', get_parameter_number(real_unet))

#######################################################################################
'''                                 Model Config                                    '''
#######################################################################################

## fake_unet, for denoising noisy images
fake_unet = copy.deepcopy(real_unet)
fake_unet.requires_grad_(True)
print('\r fake_unet size:', get_parameter_number(fake_unet))

## generator network
if args.gen_network == "sagan":
    netG = sagan_generator(dim_z=args.z_dim, dim_embed=args.dim_embed, nc=args.num_channels, img_size=args.image_size, gene_ch=args.gene_ch)
else:
    netG = sngan_generator(dim_z=args.z_dim, dim_embed=args.dim_embed, nc=args.num_channels, img_size=args.image_size, gene_ch=args.gene_ch)
netG.requires_grad_(True)
print('\r netG size:', get_parameter_number(netG))

## discriminator network
if args.gen_network == "sagan":
    netD = sagan_discriminator(dim_embed=args.dim_embed, nc=args.num_channels, img_size=args.image_size, disc_ch=args.disc_ch)
else:
    netD = sngan_discriminator(dim_embed=args.dim_embed, nc=args.num_channels, img_size=args.image_size, disc_ch=args.disc_ch)
netD.requires_grad_(True)
print('\r netD size:', get_parameter_number(netD))



#######################################################################################
'''                                  Training                                      '''
#######################################################################################

## helper functions
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class dmd2_trainer(object):
    def __init__(
        self,
        data_name,
        train_images,
        train_labels,
        real_unet,
        fake_unet,
        netG,
        netD,
        fn_y2h,
        vicinal_params,
        cfg,
        *,
        results_folder = './results',
        mixed_precision_type = 'fp16',
        fn_y2cov=None,
        offset_noise_strength = 0.,
    ):
        super().__init__()
        
        # self.flag = 0
        
        # setups in cfg
        self.cfg = cfg
        # dataset
        ## training images are not normalized here !!!
        self.data_name = data_name
        self.train_images = train_images
        self.train_labels = train_labels #labels are normalized to [0,1]
        self.image_size = cfg.image_size
        self.num_channels = cfg.num_channels
        self.unique_train_labels = np.sort(np.array(list(set(train_labels))))
        assert train_images.max()>1.0
        assert train_labels.min()>=0 and train_labels.max()<=1.0
        assert self.num_channels == train_images.shape[1]
        assert self.image_size == train_images.shape[2]
        print("\n Training labels' range is [{},{}].".format(train_labels.min(), train_labels.max()))
        self.z_dim = cfg.z_dim
        
        # vicinal params
        self.kernel_sigma = vicinal_params["kernel_sigma"]
        self.kappa = vicinal_params["kappa"]
        self.threshold_type = vicinal_params["threshold_type"]
        self.nonzero_soft_weight_threshold = vicinal_params["nonzero_soft_weight_threshold"]

        # training params
        self.niters = cfg.niters
        self.train_batch_size = cfg.train_batch_size
        self.train_timesteps = cfg.train_timesteps
        self.fn_y2h = fn_y2h #label embedding funciton
        
        self.min_step = int(cfg.min_step_percent * self.train_timesteps)
        self.max_step = int(cfg.max_step_percent * self.train_timesteps)
        
        ## y dependent covariance
        self.fn_y2cov = fn_y2cov
        if cfg.use_Hy:
            assert self.fn_y2cov is not None
            
        ## output folder
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)
        
        
        ## init accelerator
        self.accelerator = Accelerator(
            # gradient_accumulation_steps=self.cfg.gradient_accumulate_every, 
            mixed_precision = mixed_precision_type if self.cfg.train_amp else 'no',
        )
        set_seed(self.cfg.seed)
        
        print(self.accelerator.state)
        
        ## variance schedule related constants
        if cfg.beta_schedule == 'linear':
            betas = linear_beta_schedule(self.train_timesteps)
        elif cfg.beta_schedule == 'cosine':
            betas = cosine_beta_schedule(self.train_timesteps)
        else:
            raise ValueError(f'unknown beta schedule {cfg.beta_schedule}')
        
        betas = betas.to(self.device)
        
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)
        
        
        # self.alphas = alphas
        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1)
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min =1e-20))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

        # offset noise strength - 0.1 was claimed ideal

        self.offset_noise_strength = offset_noise_strength

        # loss weight
        snr = alphas_cumprod / (1 - alphas_cumprod)
        maybe_clipped_snr = snr.clone()
        self.denosing_loss_weight = maybe_clipped_snr
        
        
        ## models
        self.real_unet = real_unet
        self.fake_unet = fake_unet
        self.netG = netG
        self.netD = netD

        # step counter state
        self.step = 0
        
        # optimizer
        self.optimizer_guidance = torch.optim.AdamW(
            list(self.netD.parameters()) + list(self.fake_unet.parameters()), 
            lr=cfg.train_lr_guidance, 
            betas=(0.9, 0.999),  # pytorch's default 
            weight_decay=0.01  # pytorch's default 
        ) 
        self.optimizer_generator = torch.optim.AdamW(
            [param for param in self.netG.parameters() if param.requires_grad], 
            lr=cfg.train_lr_generator, 
            betas=(0.9, 0.999),  # pytorch's default 
            weight_decay=0.01  # pytorch's default 
        )
        
        # prepare model, dataloader, optimizer with accelerator
        self.fake_unet, self.netD, self.netG, self.optimizer_guidance, self.optimizer_generator = \
            self.accelerator.prepare(self.fake_unet, self.netD, self.netG, self.optimizer_guidance, self.optimizer_generator)
        
        
        ## visualization
        if cfg.image_size>128:
            n_row=6
        else:
            n_row=10
        n_col = n_row
        start_label = np.quantile(self.train_labels, 0.05)
        end_label = np.quantile(self.train_labels, 0.95)
        selected_labels = np.linspace(start_label, end_label, num=n_row)
        y_visual = np.zeros(n_row*n_col)
        for i in range(n_row):
            curr_label = selected_labels[i]
            for j in range(n_col):
                y_visual[i*n_col+j] = curr_label
        self.y_visual = torch.from_numpy(y_visual).type(torch.float).view(-1)
        print(self.y_visual)
    
    @property
    def device(self):
        return self.accelerator.device
    
    ## conduct y to convariance matrix
    @torch.no_grad()
    def convert_y_to_cov(self, labels):
        b, c, h, w = len(labels), self.num_channels, self.image_size, self.image_size
        return torch.exp(-self.fn_y2cov(labels).view(b,c,h,w))
        
    ## save models
    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return
        data = {
            'step': self.step,
            'fake_unet': self.accelerator.get_state_dict(self.fake_unet),
            'netG': self.accelerator.get_state_dict(self.netG),
            'netD': self.accelerator.get_state_dict(self.netD),
            'optimizer_guidance': self.optimizer_guidance.state_dict(),
            'optimizer_generator': self.optimizer_generator.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))
    
    ## load models
    def load(self, milestone):
        
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=self.device, weights_only=True)

        self.fake_unet = self.accelerator.unwrap_model(self.fake_unet)
        self.fake_unet.load_state_dict(data['fake_unet'])
        self.netG = self.accelerator.unwrap_model(self.netG)
        self.netG.load_state_dict(data['netG'])
        self.netD = self.accelerator.unwrap_model(self.netD)
        self.netD.load_state_dict(data['netD'])
        
        self.step = data['step']
        self.optimizer_guidance.load_state_dict(data['optimizer_guidance'])
        self.optimizer_generator.load_state_dict(data['optimizer_generator'])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
     
    ## calcuate xt from x0 and noise
    def calcuate_xt_from_x0_and_noise(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start)) #if noise is not given, then we use randomly generated noise

        if self.offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += self.offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    
    
    ## distribution matching loss
    ## refer to https://github.com/tianweiy/DMD2/blob/1d410b2cf04ceaab49838948ebd7157d232d9d40/main/edm/edm_guidance.py#L73
    def compute_distribution_matching_loss( 
        self, 
        images,
        labels, 
        timesteps = None
    ):
        ## normalized, scalar labels
                
        original_images = images 
        batch_size = images.shape[0]

        with torch.no_grad():
            
            timesteps = (default(timesteps, lambda: torch.randint(self.min_step, min(self.max_step+1, self.train_timesteps), (batch_size,), device=self.device))).long()
            
            # timesteps = torch.randint(self.min_step, min(self.max_step+1, self.train_timesteps), [batch_size, ], device=self.device, dtype=torch.long) 

            # use y dependent covariance matrix or not
            if self.cfg.use_Hy:
                noise = torch.randn_like(images) * torch.sqrt(self.convert_y_to_cov(labels))
            else: 
                noise = torch.randn_like(images)
            
            noisy_images = self.calcuate_xt_from_x0_and_noise(images, timesteps, noise)    
            
            pred_real_image = self.real_unet(noisy_images.float(), timesteps, self.fn_y2h(labels).float(), cond_drop_prob = 0)
            
            pred_fake_image = self.fake_unet(noisy_images.float(), timesteps, self.fn_y2h(labels).float(), cond_drop_prob = 0)
            
            p_real = (images - pred_real_image) 
            p_fake = (images - pred_fake_image) 

            weight_factor = torch.abs(p_real).mean(dim=[1, 2, 3], keepdim=True)    
            grad = (p_real - p_fake) / weight_factor
                
            grad = torch.nan_to_num(grad) 

        # this loss gives the grad as gradient through autodiff, following https://github.com/ashawkey/stable-dreamfusion 
        loss = 0.5 * F.mse_loss(original_images, (original_images-grad).detach(), reduction="mean")         

        dm_log_dict = {
            "dmtrain_noisy_images": noisy_images.detach(),
            "dmtrain_pred_real_image": pred_real_image.detach(),
            "dmtrain_pred_fake_image": pred_fake_image.detach(),
            "dmtrain_grad": grad.detach(),
            "dmtrain_gradient_norm": torch.norm(grad).item(),
            "dmtrain_timesteps": timesteps.detach(),
        }
        return loss, dm_log_dict
    
    ## denoising loss for fake_unet
    def compute_fake_denoising_loss(
        self, 
        images, 
        labels, 
        timesteps=None,
        ):
        
        batch_size = images.shape[0]
        
        images = images.detach() # no gradient to generator 
                
        timesteps = default(timesteps, lambda: torch.randint(0, self.train_timesteps, (batch_size,), device=self.device).long())
                
        # use y dependent covariance matrix or not
        if self.cfg.use_Hy:
            noise = torch.randn_like(images) * torch.sqrt(self.convert_y_to_cov(labels))
        else: 
            noise = torch.randn_like(images)

        # noise sample
        noisy_images = self.calcuate_xt_from_x0_and_noise(x_start = images, t = timesteps, noise = noise)

        model_out = self.fake_unet(noisy_images.float(), timesteps, self.fn_y2h(labels).float(), cond_drop_prob = 0)
        
        loss = F.mse_loss(model_out, images, reduction = 'none')
        if self.cfg.use_Hy:
            loss_divisor = self.convert_y_to_cov(labels)
            loss = loss / loss_divisor
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.denosing_loss_weight, timesteps, loss.shape) 

        loss = loss.mean()
        
        return loss
    
    ## adversarial loss
    def compute_d_out(self, images, labels, timesteps=None):
        
        batch_size = len(images)
        
        timesteps = default(timesteps, lambda: torch.randint(0, self.train_timesteps, (batch_size,), device=self.device).long())
        
        if self.cfg.use_Hy:
            noise = torch.randn_like(images) * torch.sqrt(self.convert_y_to_cov(labels))
        else: 
            noise = torch.randn_like(images)
            
        if self.cfg.gan_DiffAugment:
            images = DiffAugment(images, policy=self.cfg.gan_DiffAugment_policy)
        
        noisy_images = self.calcuate_xt_from_x0_and_noise(images, timesteps, noise)
        
        # if self.cfg.gan_DiffAugment:
        #     noisy_images = DiffAugment(noisy_images, policy=self.cfg.gan_DiffAugment_policy)
        
        output = self.netD(noisy_images.float(), self.fn_y2h(labels).float())
        
        return output
    
    
    ## adversarial loss: G
    def compute_generator_adv_loss(self, fake_images, fake_labels, timesteps=None):
        assert len(fake_images) == len(fake_labels)
        
        d_out_fake = self.compute_d_out(fake_images, fake_labels, timesteps)
        if self.cfg.adv_loss_type == "vanilla":
            d_out_fake = torch.sigmoid(d_out_fake)
            g_loss = - torch.mean(torch.log(d_out_fake+1e-20))
        elif self.cfg.adv_loss_type == "hinge":
            g_loss = - torch.mean(d_out_fake)
        else:
            raise ValueError('Not supported adversarial loss type!!!')

        return g_loss.view(-1)
    
    ## adversarial loss: D and fake_unet
    def compute_guidance_adv_loss(self, real_images, real_labels, fake_images, fake_labels, timesteps=None):
        
        assert len(real_images) == len(real_labels)
        assert len(fake_images) == len(fake_labels)
        
        d_out_real = self.compute_d_out(real_images.detach(), real_labels, timesteps)
        d_out_fake = self.compute_d_out(fake_images.detach(), fake_labels, timesteps)

        log_dict = {
            "d_prob_real": torch.sigmoid(d_out_real).detach(), 
            "d_prob_fake": torch.sigmoid(d_out_fake).detach() 
        }
        
        if self.cfg.adv_loss_type == "vanilla":
            d_out_real = torch.sigmoid(d_out_real)
            d_out_fake = torch.sigmoid(d_out_fake)
            d_loss_real = - torch.log(d_out_real+1e-20)
            d_loss_fake = - torch.log(1-d_out_fake+1e-20)
        elif self.cfg.adv_loss_type == "hinge":
            d_loss_real = torch.relu(1.0 - d_out_real)
            d_loss_fake = torch.relu(1.0 + d_out_fake)
        else:
            raise ValueError('Not supported adversarial loss type!!!')

        d_loss = torch.mean(d_loss_real) + torch.mean(d_loss_fake)

        return d_loss, log_dict
    

    ## training function
    def train(self):
        
        self.real_unet.eval()

        log_filename = os.path.join(self.results_folder, 'log_loss_niters{}.txt'.format(self.niters))
        if not os.path.isfile(log_filename):
            logging_file = open(log_filename, "w")
            logging_file.close()
        with open(log_filename, 'a') as file:
            file.write("\n===================================================================================================")
               
        with tqdm(initial = self.step, total = self.niters, disable = not self.accelerator.is_main_process) as pbar:

            while self.step < self.niters:

                total_generator_loss = 0.
                generator_dm_loss = 0.
                generator_adv_loss = 0.
                total_guidance_loss = 0.    
                guidance_denoising_loss = 0.
                guidance_adv_loss = 0.                    
                
                # Set timestep sigma to a preset value for all images in the batch.
                # batch_timesteps = torch.randint(0, self.train_timesteps, (self.train_batch_size,), device=self.device).long()  
                batch_timesteps = None
                
                
                ############################################
                '''  Train generator   '''
                ############################################
                
                self.netG.train()
                
                for _ in range(self.cfg.gradient_accumulate_every): #gradient accumulation
                    
                    ## w/ or w/o vicinity; the hard vicinity is assumed
                    if self.cfg.kappa==0:
                        ## draw labels from the training set
                        batch_indx = np.random.choice(np.arange(len(self.train_images)), size=self.train_batch_size, replace=True)
                        batch_target_labels = torch.from_numpy(self.train_labels[batch_indx]).type(torch.float).to(self.device)
                            
                    else:
                        ## randomly draw batch_size_disc y's from unique_train_labels
                        batch_target_labels_in_dataset = np.random.choice(self.unique_train_labels, size=self.train_batch_size, replace=True)
                        ## add Gaussian noise; we estimate image distribution conditional on these labels
                        batch_epsilons = np.random.normal(0, self.kernel_sigma, self.train_batch_size)
                        batch_target_labels = batch_target_labels_in_dataset + batch_epsilons
                        batch_target_labels = torch.from_numpy(batch_target_labels).type(torch.float).to(self.device)
                    ##end if kappa
                    
                    z = torch.randn(self.train_batch_size, self.z_dim, dtype=torch.float).to(self.device)
                    batch_fake_images = self.netG(z, self.fn_y2h(batch_target_labels))
                    
                    ## compute loss
                    with self.accelerator.autocast():
                        dm_loss, _ = self.compute_distribution_matching_loss(batch_fake_images, batch_target_labels)
                        adv_loss = self.compute_generator_adv_loss(batch_fake_images, batch_target_labels, timesteps=batch_timesteps)
                        generator_loss = dm_loss + self.cfg.weight_generator_adv * adv_loss
                        generator_loss /= self.cfg.gradient_accumulate_every
                        total_generator_loss += generator_loss.item()
                        generator_dm_loss += (dm_loss.item() / self.cfg.gradient_accumulate_every)
                        generator_adv_loss += (adv_loss.item() / self.cfg.gradient_accumulate_every)
                    
                    self.accelerator.backward(generator_loss)
                    
                ## end for Ga
                
                self.accelerator.wait_for_everyone()
                self.accelerator.clip_grad_norm_(self.netG.parameters(), self.cfg.max_grad_norm)
                self.optimizer_generator.step()     
                self.optimizer_generator.zero_grad() 
                self.optimizer_guidance.zero_grad()   
                # if we also compute gan loss, the gan branch also received gradient 
                # zero out guidance model's gradient avoids undesired gradient accumulation
                
                
                
                
                ############################################
                '''  Train Discriminator + fake_unet   '''
                ############################################
                
                self.netD.train()
                self.fake_unet.train()
                
                for _ in range(self.cfg.num_D_steps): # update D multiple times while update G once
                
                    for _ in range(self.cfg.gradient_accumulate_every): #gradient accumulation
                        
                        ## w/ or w/o vicinity; the hard vicinity is assumed
                        if self.cfg.kappa==0:
    
                            ## draw labels from the training set
                            batch_real_indx = np.random.choice(np.arange(len(self.train_images)), size=self.train_batch_size, replace=True)
                            batch_target_labels = torch.from_numpy(self.train_labels[batch_real_indx]).type(torch.float).to(self.device)
                            
                            ## draw real image/label batch from the training set
                            batch_real_images = self.train_images[batch_real_indx]
                            if self.data_name == "UTKFace":
                                batch_real_images = random_hflip(batch_real_images)
                            if self.data_name == "Cell200":
                                batch_real_images = random_rotate(batch_real_images)
                                batch_real_images = random_hflip(batch_real_images)
                                batch_real_images = random_vflip(batch_real_images)
                            # if self.data_name == "SteeringAngle":
                            #     batch_real_images, batch_flipped_indx = random_hflip(batch_real_images, return_flipped_indx=True) 
                            batch_real_images = torch.from_numpy(normalize_images(batch_real_images, to_neg_one_to_one=True)) #convert to [-1,1]!
                            batch_real_images = batch_real_images.type(torch.float).to(self.device)
                            
                            ## generate fake images
                            z = torch.randn(self.train_batch_size, self.z_dim, dtype=torch.float).to(self.device)
                            batch_fake_images = self.netG(z, self.fn_y2h(batch_target_labels))
                        
                        else:
                            
                            ## randomly draw batch_size_disc y's from unique_train_labels
                            batch_target_labels_in_dataset = np.random.choice(self.unique_train_labels, size=self.train_batch_size, replace=True)
                            ## add Gaussian noise; we estimate image distribution conditional on these labels
                            batch_epsilons = np.random.normal(0, self.kernel_sigma, self.train_batch_size)
                            batch_target_labels = batch_target_labels_in_dataset + batch_epsilons
                        
                            ## find index of real images with labels in the vicinity of batch_target_labels
                            ## generate labels for fake image generation; these labels are also in the vicinity of batch_target_labels
                            batch_real_indx = np.zeros(self.train_batch_size, dtype=int) #index of images in the datata; the labels of these images are in the vicinity
                            batch_fake_labels = np.zeros(self.train_batch_size)
                            
                            for j in range(self.train_batch_size):
                                ## index for real images
                                indx_real_in_vicinity = np.where(np.abs(self.train_labels-batch_target_labels[j])<= self.kappa)[0]
                                ## if the max gap between two consecutive ordered unique labels is large, it is possible that len(indx_real_in_vicinity)<1
                                while len(indx_real_in_vicinity)<1:
                                    batch_epsilons_j = np.random.normal(0, self.kernel_sigma, 1)
                                    batch_target_labels[j] = batch_target_labels_in_dataset[j] + batch_epsilons_j
                                    # batch_target_labels = np.clip(batch_target_labels, 0.0, 1.0)
                                    indx_real_in_vicinity = np.where(np.abs(self.train_labels-batch_target_labels[j])<= self.kappa)[0]
                                #end while len(indx_real_in_vicinity)<1
                                assert len(indx_real_in_vicinity)>=1
                                batch_real_indx[j] = np.random.choice(indx_real_in_vicinity, size=1)[0]
                                
                                ## labels for fake images generation
                                lb = batch_target_labels[j] - self.cfg.kappa
                                ub = batch_target_labels[j] + self.cfg.kappa
                                lb = max(0.0, lb); ub = min(ub, 1.0)
                                assert lb<=ub
                                assert lb>=0 and ub>=0
                                assert lb<=1 and ub<=1
                                batch_fake_labels[j] = np.random.uniform(lb, ub, size=1)[0]
                            ##end for j
                            
                            batch_target_labels = torch.from_numpy(batch_target_labels).type(torch.float).to(self.device)
                            
                            ## draw real image/label batch from the training set
                            batch_real_images = self.train_images[batch_real_indx]
                            if self.data_name == "UTKFace":
                                batch_real_images = random_hflip(batch_real_images)
                            if self.data_name == "Cell200":
                                batch_real_images = random_rotate(batch_real_images)
                                batch_real_images = random_hflip(batch_real_images)
                                batch_real_images = random_vflip(batch_real_images)
                            # if self.data_name == "SteeringAngle":
                            #     batch_real_images, batch_flipped_indx = random_hflip(batch_real_images, return_flipped_indx=True)
                            batch_real_images = torch.from_numpy(normalize_images(batch_real_images, to_neg_one_to_one=True)) 
                            batch_real_images = batch_real_images.type(torch.float).to(self.device)
                            # batch_real_labels = self.train_labels[batch_real_indx]
                            # batch_real_labels = torch.from_numpy(batch_real_labels).type(torch.float).to(self.device)
                            # # if self.data_name == "SteeringAngle":
                            # #     batch_real_labels[batch_flipped_indx] = 1 - batch_real_labels[batch_flipped_indx]
                            
                            ## generate the fake image batch
                            batch_fake_labels = torch.from_numpy(batch_fake_labels).type(torch.float).to(self.device)
                            z = torch.randn(self.train_batch_size, self.z_dim, dtype=torch.float).to(self.device)   
                            batch_fake_images = self.netG(z, self.fn_y2h(batch_fake_labels))
                        
                        ## compute loss
                        with self.accelerator.autocast():                       
                            adv_loss, _ = self.compute_guidance_adv_loss(batch_real_images, batch_target_labels, batch_fake_images, batch_target_labels, timesteps=batch_timesteps)
                            denoising_loss = self.compute_fake_denoising_loss(batch_fake_images, batch_target_labels, timesteps=batch_timesteps)
                            guidance_loss = denoising_loss + self.cfg.weight_guidance_adv * adv_loss
                            guidance_loss /= self.cfg.gradient_accumulate_every
                            total_guidance_loss += guidance_loss.item()
                            guidance_denoising_loss += (denoising_loss.item()/self.cfg.gradient_accumulate_every)
                            guidance_adv_loss += (adv_loss.item()/self.cfg.gradient_accumulate_every)
                            
                        self.accelerator.backward(guidance_loss)
                        
                    self.accelerator.wait_for_everyone()
                    self.accelerator.clip_grad_norm_(list(self.netD.parameters()) + list(self.fake_unet.parameters()), self.cfg.max_grad_norm)
                    self.optimizer_guidance.step()     
                    self.optimizer_guidance.zero_grad()   
                    
                self.optimizer_generator.zero_grad() #refer to DMD code
                
                ##end for Ga 
                
                total_guidance_loss /= self.cfg.num_D_steps
                guidance_denoising_loss /= self.cfg.num_D_steps
                guidance_adv_loss /= self.cfg.num_D_steps
                
                pbar_text = "g_loss: {:.4f} (dm{:.4f}+adv{:.4f}), d_loss: {:.4f} (dn{:.4f}+adv{:.4f}).".format(total_generator_loss, generator_dm_loss, generator_adv_loss, total_guidance_loss, guidance_denoising_loss, guidance_adv_loss)
                pbar.set_description(pbar_text)
                # pbar.set_description(f'g_loss: {total_generator_loss:.3f}, d_loss: {total_guidance_loss:.3f}')
                
                if self.step%250==0:
                    with open(log_filename, 'a') as file:
                        file.write("\n Step: {}, Gen Loss: {:.4f} (dm loss: {:.4f}, adv loss: {:.4f}), Gui Loss: {:.4f} (de loss: {:.4f}, adv loss: {:.4f}).".format(self.step, total_generator_loss, generator_dm_loss, generator_adv_loss, total_guidance_loss, guidance_denoising_loss, guidance_adv_loss))
                
                self.accelerator.wait_for_everyone()

                self.step += 1
                
                ############################################
                '''  Generate images and save ckpt   '''
                ############################################
                
                if self.accelerator.is_main_process:
                    
                    if self.step != 0 and divisible_by(self.step, self.cfg.sample_every):
                        self.netG.eval()
                        with torch.inference_mode():
                            z = torch.randn(len(self.y_visual), self.z_dim, dtype=torch.float).to(self.device)
                            gen_imgs = self.netG(z, self.fn_y2h(self.y_visual))
                            # gen_imgs = torch.clip(gen_imgs,-1,1)
                            assert gen_imgs.size(1)==self.num_channels
                            
                            utils.save_image(gen_imgs.data, str(self.results_folder) + '/sample_{}.png'.format(self.step), nrow=int(math.sqrt(len(self.y_visual))), normalize=True)

                    if self.step !=0 and divisible_by(self.step, self.cfg.save_every):
                        milestone = self.step
                        self.netG.eval()
                        self.netD.eval()
                        self.fake_unet.eval()
                        self.save(milestone)
                
                pbar.update(1)
                
        self.accelerator.print('training complete')
    ## end def
    
    
    def sample_given_labels(self, given_labels, batch_size, denorm=True, to_numpy=False, verbose=False):
        """
        Generate samples based on given labels
        :given_labels: normalized labels
        """
        assert given_labels.min()>=0 and given_labels.max()<=1.0
        nfake = len(given_labels)

        if batch_size>nfake:
            batch_size = nfake
        fake_images = []
        assert nfake%batch_size==0

        tmp = 0
        while tmp < nfake:
            batch_fake_labels = torch.from_numpy(given_labels[tmp:(tmp+batch_size)]).type(torch.float).view(-1).cuda()
            self.netG.eval()
            with torch.inference_mode():
                z = torch.randn(len(batch_fake_labels), self.z_dim, dtype=torch.float).to(self.device)
                batch_fake_images = self.netG(z, self.fn_y2h(batch_fake_labels))
                batch_fake_images = batch_fake_images.detach().cpu()
                
            if denorm: #denorm imgs to save memory
                # batch_fake_images = torch.clip(batch_fake_images, -1, 1)
                batch_fake_images = (batch_fake_images+1)/2
                batch_fake_images = (batch_fake_images*255.0).type(torch.uint8)

            fake_images.append(batch_fake_images.detach().cpu())
            tmp += batch_size
            if verbose:
                # pb.update(min(float(tmp)/nfake, 1)*100)
                print("\r {}/{} complete...".format(tmp, nfake))
        
        fake_images = torch.cat(fake_images, dim=0)
        #remove extra entries
        fake_images = fake_images[0:nfake]

        if to_numpy:
            fake_images = fake_images.numpy()

        return fake_images, given_labels



## end class dmd2_trainer


## initialization     
trainer_s = dmd2_trainer(
        data_name=args.data_name,
        train_images=train_images,
        train_labels=train_labels_norm,
        real_unet=real_unet,
        fake_unet=fake_unet,
        netG=netG,
        netD=netD,
        fn_y2h=fn_y2h,
        vicinal_params=vicinal_params,
        fn_y2cov=fn_y2cov,
        cfg=args,
        results_folder = save_results_folder,
)
if args.resume_niter>0:
    trainer_s.load(args.resume_niter)
trainer_s.train()



#######################################################################################
'''                                  Sampling                                       '''
#######################################################################################


print("\n Start sampling {} fake images per label from the model >>>".format(args.nfake_per_label))

## get evaluation labels
_, _, eval_labels = dataset.load_evaluation_data()

num_eval_labels = len(eval_labels)
print(eval_labels)


###########################################
''' multiple h5 files '''

dump_fake_images_folder = os.path.join(save_results_folder, 'fake_data_niters{}_nfake{}'.format(args.niters, int(args.nfake_per_label*num_eval_labels)))
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
        fake_images_i, _ = trainer_s.sample_given_labels(given_labels = dataset.fn_normalize_labels(fake_labels_i), batch_size = args.samp_batch_size, denorm=True, to_numpy=True, verbose=False)
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















