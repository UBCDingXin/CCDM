## adapted from https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/elucidated_diffusion.py

import numpy as np
import math
from functools import partial
from collections import namedtuple
from tqdm.auto import tqdm
import random

import torch
from torch import nn, einsum
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from utils import default, exists, identity, unnormalize_to_zero_to_one, normalize_to_neg_one_to_one, prob_mask_like, my_log


class ElucidatedDiffusion(nn.Module):
    def __init__(
        self,
        net,                        # UNet
        fn_y2sigma_data,            # function for coverting y to sigma
        aux_loss_params,            # auxiliary penalty stuffs
        *,
        image_size,
        channels = 3,
        num_sample_steps = 32, # number of sampling steps
        sigma_data_default = 0.5,
        sigma_min = 0.002,     # min noise level
        sigma_max = 80,        # max noise level
        rho = 7,               # controls the sampling schedule
        P_mean = -1.2,         # mean of log-normal distribution from which noise is drawn for training
        P_std = 1.2,           # standard deviation of log-normal distribution from which noise is drawn for training
        S_churn = 80,          # parameters for stochastic sampling - depends on dataset
        S_tmin = 0.05,         
        S_tmax = 50,
        S_noise = 1.003,
        cond_drop_prob = 0.5,  # prob of dropping labels in training denoising network
        use_y2cov = False,        # use y dependent covariance matrix
        fn_y2cov = None,         # function for label embedding
        y2cov_hy_weight_train = 1.0,
        y2cov_hy_weight_test = 1.0, 
    ):
        super().__init__()
        # assert net.random_or_learned_sinusoidal_cond
        self.self_condition = net.self_condition

        self.net = net

        # image dimensions

        self.channels = channels
        self.image_size = image_size

        # parameters
        
        self.num_sample_steps = num_sample_steps  # otherwise known as N in the paper
        
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.rho = float(rho)

        self.P_mean = float(P_mean)
        self.P_std = float(P_std)

        self.S_churn = float(S_churn)
        self.S_tmin = float(S_tmin)
        self.S_tmax = float(S_tmax)
        self.S_noise = float(S_noise)
        
        # sigma_data
        self.fn_y2sigma_data = fn_y2sigma_data #calculate the y-dependent sigma_data
        self.sigma_data_default = sigma_data_default #if condition label is not given, use this default value

        # use y dependent covariance matrix
        self.use_y2cov = use_y2cov
        self.fn_y2cov = fn_y2cov
        if self.use_y2cov:
            assert self.fn_y2cov is not None
            assert y2cov_hy_weight_train>0 or y2cov_hy_weight_test>0
        self.y2cov_hy_weight_train = y2cov_hy_weight_train
        self.y2cov_hy_weight_test = y2cov_hy_weight_test

        self.cond_drop_prob = float(cond_drop_prob)
        
        ## auxiliary loss params
        self.aux_loss_params = aux_loss_params
        if self.aux_loss_params["use_aux_reg_loss"]:
            self.aux_reg_net = self.aux_loss_params["aux_reg_net"].to(self.device)
            self.aux_reg_net.eval()
        
        
    @property
    def device(self):
        return next(self.net.parameters()).device
    

    # construct diagonal of the covariance matrix Sigma(t,y) and its derivative \dot(\Sigma)(t,y)
    
    @torch.no_grad()
    def convert_y_to_tilde_h(self, labels):
        b, c, h, w = len(labels), self.channels, self.image_size, self.image_size
        output = self.fn_y2cov(labels)
        # assert output.min().item()>=-1.0 and output.max().item()<=1.0
        return torch.exp(-output.view(b,c,h,w)) #[0,1]

    @torch.no_grad()
    def cal_cov(self, sigma, labels, shape, device, dtype, keep_indx=None, train_mode=True):
        
        if train_mode:
            y2cov_hy_weight = self.y2cov_hy_weight_train
        else:
            y2cov_hy_weight = self.y2cov_hy_weight_test
        
        B, C, H, W = shape
        
        # sigma: (B,) tensor, sampled from log(sigma) ~ N(P_mean, P_std^2)
        if isinstance(sigma, (float, int)):
            sigma = torch.full((B,), float(sigma), device=device, dtype=dtype)
        else:
            sigma = sigma.to(device=device, dtype=dtype)
            
        # Construct cov_diag
        cov_diag = torch.ones((B, C, H, W), device=device, dtype=dtype)
        sigma2 = sigma * sigma                                   # (B,)
        cov_diag = cov_diag * sigma2.view(B, 1, 1, 1)

        # labels_emb_tilde: nonlinear transformation of label_emb
        if self.use_y2cov:
            if keep_indx is None:
                keep_indx = torch.arange(start=0, end=B, step=1, dtype=torch.long, device=device)
            B_keep = len(labels[keep_indx])
            labels_emb_tilde = self.convert_y_to_tilde_h(labels[keep_indx]) # (B_keep, c, h ,w)  
            # print(labels_emb_tilde.min().item(), labels_emb_tilde.mean().item(), labels_emb_tilde.max().item(), (torch.abs(labels_emb_tilde - 1) < 1e-5).float().mean().item())    
            cov_diag[keep_indx, :,:,:] = cov_diag[keep_indx, :,:,:] + y2cov_hy_weight * labels_emb_tilde * sigma[keep_indx].view(B_keep, 1, 1, 1)
        
        return cov_diag
    
    ### derivative of covariance
    @torch.no_grad()
    def cal_dcov(self, sigma, labels, shape, device, dtype, keep_indx=None, train_mode=True): 
        
        if train_mode:
            y2cov_hy_weight = self.y2cov_hy_weight_train
        else:
            y2cov_hy_weight = self.y2cov_hy_weight_test
        
        B, C, H, W = shape
        
        # sigma: (B,) tensor, sampled from log(sigma) ~ N(P_mean, P_std^2)
        if isinstance(sigma, (float, int)):
            sigma = torch.full((B,), float(sigma), device=device, dtype=dtype)
        else:
            sigma = sigma.to(device=device, dtype=dtype)
            
        # Construct cov_diag
        dsigma = 2 * sigma                                   # (B,)
        dcov_diag = torch.ones((B, C, H, W), device=device, dtype=dtype)
        dcov_diag = dcov_diag * dsigma.view(B, 1, 1, 1) #2*sigma(t)

        # labels_emb_tilde: nonlinear transformation of label_emb
        if self.use_y2cov:
            if keep_indx is None:
                keep_indx = torch.arange(start=0, end=B, step=1, dtype=torch.long, device=device)
            labels_emb_tilde = self.convert_y_to_tilde_h(labels[keep_indx]) # (B_keep, c, h ,w)      
            dcov_diag[keep_indx, :,:,:] = dcov_diag[keep_indx, :,:,:] + y2cov_hy_weight * labels_emb_tilde
        
        return dcov_diag
    
        
    
    ###############################
    # preconditioning

    # derived preconditioning params in matrix form
    
    # input should be (B, C, H, W). 
    
    def c_in(self, cov_diag, labels, null_indx=None):
        # (cov + s^2)^(-1/2)
        s = self.fn_y2sigma_data(labels) #std
        if null_indx is not None:
            s[null_indx] = self.sigma_data_default
        assert len(cov_diag)==len(labels)
        s = s.view(len(cov_diag), 1, 1, 1).to(device=cov_diag.device, dtype=cov_diag.dtype)
        s2 = s * s #cov
        out = torch.rsqrt(cov_diag + s2)              
        return out

    def c_skip(self, cov_diag, labels, null_indx=None):
        # s^2 / (s^2 + cov)
        s = self.fn_y2sigma_data(labels) #std
        if null_indx is not None:
            s[null_indx] = self.sigma_data_default
        assert len(cov_diag)==len(labels)
        s = s.view(len(cov_diag), 1, 1, 1).to(device=cov_diag.device, dtype=cov_diag.dtype)
        s2 = s * s #cov
        out = s2 * (cov_diag + s2)**(-1) #s2 * (s2 + cov)^(-1)
        return out

    def c_out(self, cov_diag, labels, null_indx=None):
        # s * sqrt(cov) / sqrt(s^2 + cov)
        s = self.fn_y2sigma_data(labels) #std
        if null_indx is not None:
            s[null_indx] = self.sigma_data_default
        assert len(cov_diag)==len(labels)
        s = s.view(len(cov_diag), 1, 1, 1).to(device=cov_diag.device, dtype=cov_diag.dtype)
        s2 = s * s #cov
        out = s * cov_diag.sqrt() * torch.rsqrt(cov_diag + s2)
        return out 
    
    # Based on the fact that the covariance matrix (Sigma) is constructed from sigma (noise level) and y, and y has already been passed to the UNet, there is no need to feed the complete Sigma into the UNet. 
    def c_noise(self, sigma): 
        return my_log(sigma) * 0.25

    # preconditioned network output
    def preconditioned_network_forward(
        self, 
        noised_images, 
        sigma, 
        labels,
        labels_emb, 
        cond_scale = 1.5, 
        rescaled_phi = 0.7, 
        self_cond = None, 
        clamp = False, 
        train_mode=False,
        keep_mask = None,
        null_indx = None,
        ):
        
        B, C, H, W = noised_images.shape
        device, dtype = self.device, noised_images.dtype

        # sigma: (B,) tensor, sampled from log(sigma) ~ N(P_mean, P_std^2)
        if isinstance(sigma, (float, int)):
            sigma = torch.full((B,), float(sigma), device=device, dtype=dtype)
        else:
            sigma = sigma.to(device=device, dtype=dtype)

        cov_diag_4d = self.cal_cov(sigma=sigma, labels=labels, shape=(B,C,H,W), device=device, dtype=dtype, train_mode=train_mode)

        # compute preconditioning coefficients
        c_in_tensor   = self.c_in(cov_diag_4d, labels, null_indx)
        c_skip_tensor = self.c_skip(cov_diag_4d, labels, null_indx)
        c_out_tensor  = self.c_out(cov_diag_4d, labels, null_indx)

        if train_mode: #training phase
            net_out = self.net(
                x           = c_in_tensor * noised_images,
                time        = self.c_noise(sigma),
                x_self_cond = self_cond,
                labels_emb  = labels_emb,
                keep_mask = keep_mask, 
            )
        else: #sampling phase
            net_out = self.net.forward_with_cond_scale(
                x           = c_in_tensor * noised_images,
                time        = self.c_noise(sigma),
                x_self_cond = self_cond,
                labels_emb  = labels_emb,
                cond_scale  = cond_scale,
                rescaled_phi= rescaled_phi,
            )

        out = c_skip_tensor * noised_images + c_out_tensor * net_out

        if clamp:
            out.clamp_(-1., 1.)  # in-place clamp

        return out
        

    #############################################
    # sampling

    # sample schedule
    # equation (5) in the EDM paper

    def sample_schedule(self, num_sample_steps = None, dtype = torch.float):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        N = num_sample_steps
        inv_rho = 1 / self.rho

        steps = torch.arange(num_sample_steps, device = self.device, dtype = dtype)
        sigmas = (self.sigma_max ** inv_rho + steps / (N - 1) * (self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)) ** self.rho

        sigmas = F.pad(sigmas, (0, 1), value = 0.) # last step is sigma value of 0.
        
        return sigmas

    # sampling via SDE
    @torch.no_grad()
    def sample_using_sde(
        self, 
        labels,
        labels_emb, 
        cond_scale = 1.5, 
        rescaled_phi = 0.7, 
        num_sample_steps = None, 
        clamp = False
        ):
        
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        batch_size = len(labels_emb)

        shape = (batch_size, self.channels, self.image_size, self.image_size)
        dtype = labels_emb.dtype

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma

        sigmas = self.sample_schedule(num_sample_steps, dtype=dtype)

        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(self.S_churn / num_sample_steps, math.sqrt(2) - 1),
            0.
        )

        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        # images is noise at the beginning

        if self.use_y2cov:
            images = self.cal_cov(sigma=sigmas[0].item(), labels=labels, shape=shape, device=self.device, dtype=dtype, train_mode=False).sqrt() * torch.randn(shape, device = self.device)
        else:
            images = sigmas[0] * torch.randn(shape, device = self.device)

        # for self conditioning
        x_start = None

        # gradually denoise

        for sigma, sigma_next, gamma in tqdm(sigmas_and_gammas, desc = 'sampling time step'):
            sigma, sigma_next, gamma = map(lambda t: t.item(), (sigma, sigma_next, gamma))

            eps = self.S_noise * torch.randn(shape, device = self.device) # stochastic sampling

            sigma_hat = sigma + gamma * sigma
            cov_diag = self.cal_cov(sigma=sigma, labels=labels, shape=shape, device=self.device, dtype=dtype, train_mode=False)
            cov_diag_hat = self.cal_cov(sigma=sigma_hat, labels=labels, shape=shape, device=self.device, dtype=dtype, train_mode=False)
            images_hat = images + (cov_diag_hat - cov_diag).sqrt() * eps

            self_cond = x_start if self.self_condition else None

            dcov_diag_hat = self.cal_dcov(sigma=sigma_hat, labels=labels, shape=shape, device=self.device, dtype=dtype, train_mode=False)
            model_output = self.preconditioned_network_forward(noised_images = images_hat, sigma = sigma_hat, labels=labels, labels_emb = labels_emb, cond_scale = cond_scale, rescaled_phi = rescaled_phi, self_cond = self_cond, clamp = clamp, train_mode=False)
            denoised_over_sigma = 0.5 * dcov_diag_hat * cov_diag_hat**(-1) * (images_hat - model_output) 

            images_next = images_hat + (sigma_next - sigma_hat) * denoised_over_sigma

            # second order correction, if not the last timestep

            if sigma_next != 0:
                self_cond = model_output if self.self_condition else None

                cov_diag_hat_next = self.cal_cov(sigma=sigma_next, labels=labels, shape=shape, device=self.device, dtype=dtype, train_mode=False)
                dcov_diag_hat_next = self.cal_dcov(sigma=sigma_next, labels=labels, shape=shape, device=self.device, dtype=dtype, train_mode=False)
                
                model_output_next = self.preconditioned_network_forward(noised_images = images_next, sigma = sigma_next, labels=labels, labels_emb = labels_emb, cond_scale = cond_scale, rescaled_phi = rescaled_phi, self_cond = self_cond, clamp = clamp, train_mode=False)
                
                denoised_prime_over_sigma = 0.5 * dcov_diag_hat_next * cov_diag_hat_next**(-1) * (images_next - model_output_next)
                
                images_next = images_hat + 0.5 * (sigma_next - sigma_hat) * (denoised_over_sigma + denoised_prime_over_sigma)

            images = images_next
            x_start = model_output_next if sigma_next != 0 else model_output
        
        images = images.clamp(-1., 1.)
        return unnormalize_to_zero_to_one(images)

    # sampling via ODE  
    
    @torch.no_grad()
    def sample_using_ode(
        self, 
        labels,
        labels_emb, 
        cond_scale = 1.5, 
        rescaled_phi = 0.7, 
        num_sample_steps = None, 
        clamp = False
        ):
        
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        batch_size = len(labels_emb)

        shape = (batch_size, self.channels, self.image_size, self.image_size)
        dtype = labels_emb.dtype

        sigmas = self.sample_schedule(num_sample_steps, dtype=dtype)
        
        if self.use_y2cov:
            sqrt_cov_diag = self.cal_cov(sigma=sigmas[0].item(), labels=labels, shape=shape, device=self.device, dtype=dtype, train_mode=False).sqrt()
            images = sqrt_cov_diag * torch.randn(shape, device = self.device)
        else:
            images = sigmas[0] * torch.randn(shape, device = self.device)
        
        # for self conditioning
        x_start = None
        
        for i in tqdm(range(len(sigmas) - 1)):     
            
            sigma_curr, sigma_next = sigmas[i].item(), sigmas[i+1].item()
            
            self_cond = x_start if self.self_condition else None
            model_output = self.preconditioned_network_forward(noised_images = images, sigma = sigma_curr, labels=labels, labels_emb = labels_emb, cond_scale = cond_scale, rescaled_phi = rescaled_phi, self_cond = self_cond, clamp = clamp, train_mode=False)
            cov_diag_curr = self.cal_cov(sigma=sigma_curr, labels=labels, shape=shape, device=self.device, dtype=dtype, train_mode=False)
            dcov_diag_curr = self.cal_dcov(sigma=sigma_curr, labels=labels, shape=shape, device=self.device, dtype=dtype, train_mode=False)
            d_i = 0.5 * dcov_diag_curr * cov_diag_curr**(-1) * (images - model_output)
            images_next = images + (sigma_next - sigma_curr) * d_i
            
            if sigma_next != 0:
                self_cond = model_output if self.self_condition else None
                model_output_next = self.preconditioned_network_forward(noised_images = images_next, sigma = sigma_next, labels=labels, labels_emb = labels_emb, cond_scale = cond_scale, rescaled_phi = rescaled_phi, self_cond = self_cond, clamp = clamp, train_mode=False)
                cov_diag_next = self.cal_cov(sigma=sigma_next, labels=labels, shape=shape, device=self.device, dtype=dtype, train_mode=False)
                dcov_diag_next = self.cal_dcov(sigma=sigma_next, labels=labels, shape=shape, device=self.device, dtype=dtype, train_mode=False)
                d_i_prime = 0.5 * dcov_diag_next * cov_diag_next**(-1) * (images_next - model_output_next)
                images_next = images + 0.5 * (sigma_next - sigma_curr) * (d_i + d_i_prime)
            
            images = images_next
            x_start = model_output_next if sigma_next != 0 else model_output
        
        images = images.clamp(-1., 1.)
        return unnormalize_to_zero_to_one(images)


    # sampling via DPM++
    @torch.no_grad()
    def sample_using_dpmpp(
        self, 
        labels,
        labels_emb, 
        cond_scale = 1.5, 
        rescaled_phi = 0.7, 
        num_sample_steps = None, 
        ):
        
        """
        thanks to Katherine Crowson (https://github.com/crowsonkb) for figuring it all out!
        https://arxiv.org/abs/2211.01095
        """

        device, num_sample_steps = self.device, default(num_sample_steps, self.num_sample_steps)
        batch_size = len(labels_emb)
        shape = (batch_size, self.channels, self.image_size, self.image_size)
        dtype = labels_emb.dtype

        sigmas = self.sample_schedule(num_sample_steps, dtype=dtype)
                
        if self.use_y2cov:
            sqrt_cov_diag = self.cal_cov(sigma=sigmas[0].item(), labels=labels, shape=shape, device=self.device, dtype=dtype, train_mode=False).sqrt()
            images = sqrt_cov_diag * torch.randn(shape, device = self.device)
        else:
            images = sigmas[0] * torch.randn(shape, device = self.device)

        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()

        old_denoised = None
        for i in tqdm(range(len(sigmas) - 1)):
            denoised = self.preconditioned_network_forward(noised_images = images, sigma = sigmas[i].item(), labels=labels, labels_emb = labels_emb, cond_scale = cond_scale, rescaled_phi = rescaled_phi, train_mode=False)           
            
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t

            if not exists(old_denoised) or sigmas[i + 1] == 0:
                denoised_d = denoised
            else:
                h_last = t - t_fn(sigmas[i - 1])
                r = h_last / h
                gamma = - 1 / (2 * r)
                denoised_d = (1 - gamma) * denoised + gamma * old_denoised

            images = (sigma_fn(t_next) / sigma_fn(t)) * images - (-h).expm1() * denoised_d
            old_denoised = denoised

        images = images.clamp(-1., 1.)
        return unnormalize_to_zero_to_one(images)

    
    

    #############################################
    # training

    ## \Lambda(\Sigma)^0.5
    def loss_weight(self, cov_diag, labels, null_indx=None):
        # cov_diag is the diagonal of the covariance matrix
        s = self.fn_y2sigma_data(labels) #std
        if null_indx is not None:
            s[null_indx] = self.sigma_data_default
        s2 = s * s
        s2 = s2.view(len(cov_diag), 1, 1, 1).to(device=cov_diag.device, dtype=cov_diag.dtype)
        return torch.sqrt((cov_diag + s2) / (cov_diag * s2))

    def noise_distribution(self, batch_size):
        return (self.P_mean + self.P_std * torch.randn((batch_size,), device = self.device)).exp()

    ## Auxiliary regression loss for better label consistency
    def fn_aux_reg_loss(self, gt_labels, pred_labels, epsilon = 0):
        if self.aux_loss_params['aux_reg_loss_type'].lower() in ['mse']:
            reg_loss = torch.mean( (pred_labels.view(-1) - gt_labels.view(-1))**2 )
        elif self.aux_loss_params['aux_reg_loss_type'].lower() in ['ei_hinge']:
            if isinstance(epsilon, np.ndarray):
                epsilon = torch.from_numpy(epsilon).type(torch.float).to(self.device)
            abs_diff = torch.abs(pred_labels.view(-1) - gt_labels.view(-1))
            reg_loss = torch.mean(torch.clamp(abs_diff - epsilon, min=0))
        else:
            raise ValueError('Not supported loss type!!!') 
        return reg_loss

    ## given a batch of images and labels, compute the training loss
    def forward(self, images, labels, labels_emb, vicinal_weights = None, max_kappa=0):
        batch_size, c, h, w, dtype, image_size, channels = *images.shape, images.dtype, self.image_size, self.channels

        # normalized images from [0,1] to [-1,1]
        assert images.min().item()>=0.0 and images.max().item()<=1.0
        images = normalize_to_neg_one_to_one(images) #[-1,1]

        assert h == image_size and w == image_size, f'height and width of image must be {image_size}'
        assert c == channels, 'mismatch of image channels'

        # generate keep_mask for random label dropping
        keep_mask = prob_mask_like((batch_size,), 1 - self.cond_drop_prob, device = self.device)
        null_indx = torch.where(keep_mask==False)[0]
        keep_indx = torch.where(keep_mask==True)[0]

        # generate sigmas
        sigmas = self.noise_distribution(batch_size).to(dtype)

        # compute Sigma(t, y), t = sigma
        cov_diag_y = self.cal_cov(sigma=sigmas, labels=labels, shape=(batch_size, c, h, w), device=self.device, dtype=dtype, keep_indx=keep_indx, train_mode=True) # (B, C, H, W), already take into account the drop y's    

        # purterbate data with noise
        noised_images = images + cov_diag_y.sqrt() * torch.randn((batch_size, c, h, w), device = self.device)  
        noised_images = noised_images.to(images.dtype)        

        self_cond = None
        
        if self.self_condition and random() < 0.5:
            # from hinton's group's bit diffusion paper
            with torch.no_grad():
                self_cond = self.preconditioned_network_forward(noised_images = noised_images, sigma = sigmas, labels=labels, labels_emb = labels_emb, train_mode=True, keep_mask=keep_mask, null_indx=null_indx)
                self_cond.detach_()

        denoised = self.preconditioned_network_forward(noised_images = noised_images, sigma = sigmas, labels=labels, labels_emb = labels_emb, self_cond=self_cond, train_mode=True, keep_mask=keep_mask, null_indx=null_indx)
        
        loss = (self.loss_weight(cov_diag_y, labels, null_indx).sqrt() * (denoised - images)) ** 2 #batch_size x NC x IMG_SIZE x IMG_SIZE
        
        ## apply vicinal weights or not?
        if vicinal_weights is not None:
            loss = reduce(loss, 'b ... -> b (...)', 'mean') #batch_size x (NC x IMG_SIZE x IMG_SIZE)
            loss = torch.sum(loss, dim=1) #batch_size x 1
            vicinal_weights[null_indx] = 1 #do not apply weighting on Null logits
            assert len(vicinal_weights)==len(loss)
            loss = vicinal_weights.view(-1) * loss.view(-1) / (c*h*w)  # batch_size,    
        else:
            loss = reduce(loss, 'b ... -> b', 'mean') # batch_size,  

        denoise_loss = loss.mean()
        
        if self.aux_loss_params["use_aux_reg_loss"]:
            pred_labels = self.aux_reg_net(denoised).detach()
            if self.aux_loss_params["aux_reg_loss_epsilon"]<0:
                aux_reg_loss_epsilon = max_kappa
            else:
                aux_reg_loss_epsilon = self.aux_loss_params["aux_reg_loss_epsilon"]
            aux_reg_loss = self.fn_aux_reg_loss(gt_labels=labels, pred_labels=pred_labels, epsilon = aux_reg_loss_epsilon)
            aux_reg_weight = self.aux_loss_params["aux_reg_loss_weight"]
        else:
            aux_reg_loss = torch.tensor([0.0]).to(self.device)
            aux_reg_weight = torch.tensor([0.0]).to(self.device)
        
        return denoise_loss, aux_reg_loss, aux_reg_weight