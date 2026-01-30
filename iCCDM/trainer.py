import numpy as np
import math
from functools import partial
from collections import namedtuple
from tqdm.auto import tqdm
import random
from PIL import Image
import warnings
import os
from pathlib import Path
import timeit

import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.utils import save_image, make_grid
from torch.optim import Adam

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from utils import default, exists, identity, unnormalize_to_zero_to_one, normalize_to_neg_one_to_one, prob_mask_like, my_log, has_int_squareroot, cycle, divisible_by, normalize_images, random_hflip, random_rotate, random_vflip

from accelerate import Accelerator
from ema_pytorch import EMA



class Trainer:
    def __init__(
        self,
        diffusion_model,
        fn_y2h,
        data_name,
        train_images,
        train_labels,
        vicinal_params,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_after_step = 1e30,
        ema_update_every = 10,
        ema_decay = 0.999,
        adam_betas = (0.9, 0.999),
        save_every = 1000,
        sample_every = 1000,
        y_visual = None,
        cond_scale_visual=1.5,
        cond_rescaled_phi_visual = 0.7,
        results_folder = './output/results',
        amp = False,
        mixed_precision_type = 'fp16',
        max_grad_norm = 1.,
    ):
        super().__init__()

        # accelerator
        self.mixed_precision_type = mixed_precision_type
        self.accelerator = Accelerator(
            mixed_precision = mixed_precision_type if amp else 'no'
        )
        
        # dataset
        ## training images are not normalized here !!!
        self.data_name = data_name
        self.train_images = train_images
        self.train_labels = train_labels
        self.unique_train_labels, self.counts_train_elements = np.unique(train_labels, return_counts=True) 
        assert train_images.max()>1.0
        assert train_labels.min()>=0 and train_labels.max()<=1.0
        self.min_abs_diff = np.min(np.abs(np.diff(np.sort(self.unique_train_labels))))  # Compute the minimum absolute difference between adjacent elements.
        #counts_train_elements: number of samples for each unique label
        assert train_images.max()>1.0
        assert train_labels.min()>=0 and train_labels.max()<=1.0
        print("\n Training labels' range is [{},{}].".format(train_labels.min(), train_labels.max()))
        # print(self.counts_train_elements)
        
        # vicinal params   
        self.vicinal_params = vicinal_params
        
        # model
        self.model = diffusion_model
        self.channels = diffusion_model.channels
        self.image_size = diffusion_model.image_size
        self.num_sample_steps = diffusion_model.num_sample_steps

        self.fn_y2h = fn_y2h


        # sampling and training hyperparameters
        
        ### visualize
        assert has_int_squareroot(len(y_visual)), 'number of samples must have an integer square root'
        self.y_visual = y_visual
        self.num_samples = len(y_visual)
        self.nrow_visual = int(math.sqrt(self.num_samples))
        self.sample_every = sample_every
        self.cond_scale_visual = cond_scale_visual
        self.cond_rescaled_phi_visual = cond_rescaled_phi_visual   
        
        ### hyper-parameters
        self.save_every = save_every
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        assert (train_batch_size * gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'
        self.train_num_steps = train_num_steps
        self.max_grad_norm = max_grad_norm
        
        # optimizer
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # init. EMA
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, update_after_step=ema_update_after_step, beta = ema_decay, update_every = ema_update_every, coerce_dtype=True) #coerce_dtype make sure EMA is compatible with multi-GPU
            self.ema.to(self.device)
            if ema_update_after_step<train_num_steps:
                print("\n EMA is enabled in training !!!")
            
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    ########################################################################################    
    @property
    def device(self):
        return self.accelerator.device
    
    ########################################################################################    
    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    ########################################################################################    
    def load(self, milestone, return_ema=False, return_unet=False):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device, weights_only=True)

        self.model = self.accelerator.unwrap_model(self.model)
        self.model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])
            if return_ema:
                return self.ema

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
        
        if return_unet:
            return self.model.net #take unet


    ######################################################################################## 
    ## random data augmentation for a batch of real images
    def fn_transform(self, batch_real_images):
        assert isinstance(batch_real_images, np.ndarray)
        if self.data_name == "UTKFace":
            batch_real_images = random_hflip(batch_real_images)
        if self.data_name[0:7] == "Cell200":
            batch_real_images = random_rotate(batch_real_images)
            batch_real_images = random_hflip(batch_real_images)
            batch_real_images = random_vflip(batch_real_images)
        return batch_real_images


    ######################################################################################## 
    ## make vicinity for target labels
    def make_vicinity(self, batch_target_labels, batch_target_labels_in_dataset):
        
        ###########################################
        ## fixed vicinity, conventional hard/soft vicinity
        if not self.vicinal_params["use_ada_vic"]: 
            
            ### Step 1: Retrieve the indices of real images in the dataset whose labels fall within a vicinity of the target labels. Additionally, generate random labels within the same vicinity for synthesizing fake images.
            ## find index of real images with labels in the vicinity of batch_target_labels
            ## generate labels for fake image generation; these labels are also in the vicinity of batch_target_labels
            batch_real_indx = np.zeros(self.batch_size, dtype=int) #the indices of selected real images in the training dataset; the labels of these images are in the vicinity
            batch_fake_labels = np.zeros(self.batch_size) # the fake labels used to genetated fake images
            
            for j in range(self.batch_size):
                ## index for real images
                if self.vicinal_params["threshold_type"] == "hard":
                    indx_real_in_vicinity = np.where(np.abs(self.train_labels-batch_target_labels[j])<= self.vicinal_params["kappa"])[0]
                else:
                    # reverse the weight function for SVDL
                    indx_real_in_vicinity = np.where((self.train_labels-batch_target_labels[j])**2 <= -np.log(self.vicinal_params["nonzero_soft_weight_threshold"])/self.vicinal_params["kappa"])[0]

                ## if the max gap between two consecutive ordered unique labels is large, it is possible that len(indx_real_in_vicinity)<1
                while len(indx_real_in_vicinity)<1:
                    batch_epsilons_j = np.random.normal(0, self.vicinal_params["kernel_sigma"], 1)
                    batch_target_labels[j] = batch_target_labels_in_dataset[j] + batch_epsilons_j
                    ## index for real images
                    if self.vicinal_params["threshold_type"] == "hard":
                        indx_real_in_vicinity = np.where(np.abs(self.train_labels-batch_target_labels[j])<= self.vicinal_params["kappa"])[0]
                    else:
                        # reverse the weight function for SVDL
                        indx_real_in_vicinity = np.where((self.train_labels-batch_target_labels[j])**2 <= -np.log(self.vicinal_params["nonzero_soft_weight_threshold"])/self.vicinal_params["kappa"])[0]
                #end while len(indx_real_in_vicinity)<1

                assert len(indx_real_in_vicinity)>=1
                
                # print(len(indx_real_in_vicinity))
                
                batch_real_indx[j] = np.random.choice(indx_real_in_vicinity, size=1)[0]

                ## labels for fake images generation
                if self.vicinal_params["threshold_type"] == "hard":
                    lb = batch_target_labels[j] - self.vicinal_params["kappa"]
                    ub = batch_target_labels[j] + self.vicinal_params["kappa"]
                else:
                    lb = batch_target_labels[j] - np.sqrt(-np.log(self.vicinal_params["nonzero_soft_weight_threshold"])/self.vicinal_params["kappa"])
                    ub = batch_target_labels[j] + np.sqrt(-np.log(self.vicinal_params["nonzero_soft_weight_threshold"])/self.vicinal_params["kappa"])
                lb = max(0.0, lb); ub = min(ub, 1.0)
                assert lb<=ub
                assert lb>=0 and ub>=0
                assert lb<=1 and ub<=1
                batch_fake_labels[j] = np.random.uniform(lb, ub, size=1)[0]
            #end for j
            batch_real_labels = self.train_labels[batch_real_indx]
            batch_real_labels = torch.from_numpy(batch_real_labels).type(torch.float).to(self.device)
            batch_fake_labels = torch.from_numpy(batch_fake_labels).type(torch.float).to(self.device)
                        
            ### Step 2: compute the vicinal weights
            if self.vicinal_params["threshold_type"]=="hard":
                real_weights = torch.ones(self.batch_size, dtype=torch.float).to(self.device)
                fake_weights = torch.ones(self.batch_size, dtype=torch.float).to(self.device)
            else:
                batch_target_labels = torch.from_numpy(batch_target_labels).type(torch.float).to(self.device)
                real_weights = torch.exp(-self.vicinal_params["kappa"]*(batch_real_labels-batch_target_labels)**2).to(self.device)
                fake_weights = torch.exp(-self.vicinal_params["kappa"]*(batch_fake_labels-batch_target_labels)**2).to(self.device)
             
            kappa_l_all = np.ones(self.batch_size)*self.vicinal_params["kappa"] #the left radii of the vicinity for the target labels
            kappa_r_all = np.ones(self.batch_size)*self.vicinal_params["kappa"] #the right radii of the vicinity for the target labels
            
            return batch_real_indx, batch_fake_labels, batch_real_labels, real_weights, fake_weights, kappa_l_all, kappa_r_all
        
        
        
        ###########################################
        ## adaptive vicinity
        else: 
            ## get the index of real images in the vicinity
            ## determine the labels used to generate fake images
            batch_real_indx = np.zeros(self.batch_size, dtype=int)
            batch_fake_labels = np.zeros(self.batch_size)
            kappa_l_all = np.zeros(self.batch_size) #the left radii of the vicinity for the target labels
            kappa_r_all = np.zeros(self.batch_size) #the right radii of the vicinity for the target labels         
            for j in range(self.batch_size):
                
                target_y = batch_target_labels[j]
                idx_y = np.searchsorted(self.unique_train_labels, target_y, side='left')
                kappa_l, kappa_r = self.vicinal_params["ada_eps"], self.vicinal_params["ada_eps"]
                n_got = 0
                
                ## case 1: target_y is either the first element of unique_train_labels or smaller than it. Only move toward right
                if idx_y <= 0:     
                    idx_l, idx_r = 0, 0
                    n_got = self.counts_train_elements[idx_r]
                    kappa_r = np.abs(target_y-self.unique_train_labels[idx_r]) + self.vicinal_params["ada_eps"]
                    # while n_got<self.vicinal_params["min_n_per_vic"] or (kappa_l+kappa_r)<self.min_abs_diff: #do not have enough samples in the vicinity
                    loop_counter_warning = 0
                    while n_got<self.vicinal_params["min_n_per_vic"]: #do not have enough samples in the vicinity
                        idx_r += 1
                        n_got += self.counts_train_elements[idx_r]
                        kappa_r = np.abs(target_y-self.unique_train_labels[idx_r])
                        if idx_r==(len(self.counts_train_elements)-1):
                            break
                        loop_counter_warning+=1
                        if loop_counter_warning>1e20:
                            print("\n Detected an infinite loop")

                ## case 2: target_y is either the last element of unique_train_labels or larger than it. Only move toward left
                elif idx_y >= (len(self.unique_train_labels)-1): 
                    idx_l, idx_r = len(self.unique_train_labels)-1, len(self.unique_train_labels)-1
                    n_got = self.counts_train_elements[idx_l]
                    kappa_l = np.abs(target_y-self.unique_train_labels[idx_l]) + self.vicinal_params["ada_eps"]
                    # assert target_y+kappa_l > self.unique_train_labels[idx_l]
                    # while n_got<self.vicinal_params["min_n_per_vic"] or (kappa_l+kappa_r)<self.min_abs_diff: #do not have enough samples in the vicinity
                    loop_counter_warning = 0
                    while n_got<self.vicinal_params["min_n_per_vic"]: #do not have enough samples in the vicinity
                        idx_l -= 1
                        n_got += self.counts_train_elements[idx_l]
                        kappa_l = np.abs(target_y-self.unique_train_labels[idx_l])
                        if idx_l==0:
                            break
                        loop_counter_warning+=1
                        if loop_counter_warning>1e20:
                            print("\n Detected an infinite loop")
            
                ## case 3: other cases
                else:
                    if target_y in self.unique_train_labels: #target_y appears in the training set
                        idx_l, idx_r = idx_y-1, idx_y+1
                        n_got = self.counts_train_elements[idx_y]  
                        # if n_got>=self.vicinal_params["min_n_per_vic"]:
                        #     kappa_l, kappa_r = 1e30, 1e30 #Terminate early
                    else:
                        idx_l, idx_r = idx_y-1, idx_y
                        n_got = 0 
                    
                    dist2left = np.abs(target_y-self.unique_train_labels[idx_l]) #In unique_train_labels, the distance from target_y to its nearest left label.
                    dist2right = np.abs(target_y-self.unique_train_labels[idx_r]) #In unique_train_labels, the distance from target_y to its nearest right label.
                    # while n_got<self.vicinal_params["min_n_per_vic"] or (kappa_l+kappa_r)<self.min_abs_diff: 
                    loop_counter_warning = 0
                    while n_got<self.vicinal_params["min_n_per_vic"]: 
                        if dist2left < dist2right: # If closer to the left label, expand to the left.
                            kappa_l = dist2left #update kappa_l
                            n_got += self.counts_train_elements[idx_l] #update n_got
                            idx_l -= 1 #update idx_l
                        elif dist2left > dist2right: #If closer to the right label, expand to the right.
                            kappa_r = dist2right #update kappa_r
                            n_got += self.counts_train_elements[idx_r] #update n_got
                            idx_r += 1 #update idx_r
                        else: #When the distances on both sides are equal, expand in both directions.
                            kappa_l = dist2left #update kappa_l
                            kappa_r = dist2right #update kappa_r
                            n_got += (self.counts_train_elements[idx_l] + self.counts_train_elements[idx_r])
                            idx_l -= 1 #update idx_l
                            idx_r += 1 #update idx_r
                        if idx_l < 0:
                            dist2left = 1e30 #do not move toward left anymore
                        else:
                            dist2left = np.abs(target_y-self.unique_train_labels[idx_l]) #update
                        if idx_r > len(self.unique_train_labels)-1:
                            dist2right = 1e30 #do not move toward right anymore
                        else:
                            dist2right = np.abs(target_y-self.unique_train_labels[idx_r]) #update
                        if dist2left > 1e10 and dist2right > 1e10:
                            break
                        loop_counter_warning+=1
                        if loop_counter_warning>1e20:
                            print("\n Detected an infinite loop")
                    ##end while n_got          
                        
                ##end if idx_y == 0
                
                # symmetric adaptive vicinity
                if self.vicinal_params["use_symm_vic"]:
                    kappa_l, kappa_r = np.max([kappa_l, kappa_r]), np.max([kappa_l, kappa_r])  #larger
                    # kappa_l, kappa_r = np.min([kappa_l, kappa_r]), np.min([kappa_l, kappa_r])  #smaller

                kappa_l_all[j] = kappa_l #left radius for hard vicinity
                kappa_r_all[j] = kappa_r #right radius for hard vicinity
                nu_l = 1/kappa_l**2 #decay weight for the left soft vicinity
                nu_r = 1/kappa_r**2 #decay weight for the right soft vicinity

                ## index for real images
                ### index for HV
                cond_hard = (self.train_labels>=(target_y-kappa_l)) & (self.train_labels<=(target_y+kappa_r))
                indx_real_in_hard_vicinity = np.where(cond_hard)[0]         
                ### index for SV
                indx_left = np.where(self.train_labels<=target_y)[0]
                indx_right = np.where(self.train_labels>target_y)[0]
                indx_real_in_soft_vicinity_left = np.where((self.train_labels-target_y)**2 <= -np.log(self.vicinal_params["nonzero_soft_weight_threshold"])/nu_l)[0]
                indx_real_in_soft_vicinity_left = np.intersect1d(indx_real_in_soft_vicinity_left, indx_left)
                indx_real_in_soft_vicinity_right = np.where((self.train_labels-target_y)**2 <= -np.log(self.vicinal_params["nonzero_soft_weight_threshold"])/nu_r)[0]
                indx_real_in_soft_vicinity_right = np.intersect1d(indx_real_in_soft_vicinity_right, indx_right)
                indx_real_in_soft_vicinity = np.concatenate([indx_real_in_soft_vicinity_left, indx_real_in_soft_vicinity_right])
                if self.vicinal_params["ada_vic_type"].lower()=="vanilla":
                    if self.vicinal_params["threshold_type"] == "hard":
                        indx_real_in_vicinity = indx_real_in_hard_vicinity               
                    else:
                        indx_real_in_vicinity = indx_real_in_soft_vicinity
                elif self.vicinal_params["ada_vic_type"].lower()=="hybrid":
                    # indx_real_in_vicinity = np.union1d(indx_real_in_hard_vicinity, indx_real_in_soft_vicinity) # hard in soft, not working
                    # indx_real_in_vicinity = indx_real_in_hard_vicinity #soft in hard; if soft vicinity with nonzero weights is smaller than hard vicinity, then use hard vicinity and too small soft weights will be replaced by the nonzero_soft_weight_threshold
                    indx_real_in_vicinity = np.intersect1d(indx_real_in_hard_vicinity, indx_real_in_soft_vicinity) #soft in hard, smaller vicinity
                else:
                    raise ValueError('Not supported vicinity type!!!')
                assert len(indx_real_in_vicinity)>=1
                batch_real_indx[j] = np.random.choice(indx_real_in_vicinity, size=1)[0]
                
                ## labels for fake images generation
                lb_hard = batch_target_labels[j] - kappa_l
                ub_hard = batch_target_labels[j] + kappa_r
                lb_hard = max(0.0, lb_hard); ub_hard = min(ub_hard, 1.0)
                assert lb_hard<=ub_hard and lb_hard>=0 and lb_hard<=1 and ub_hard>=0 and ub_hard<=1
                lb_soft = batch_target_labels[j] - np.sqrt(-np.log(self.vicinal_params["nonzero_soft_weight_threshold"])/nu_l)
                ub_soft = batch_target_labels[j] + np.sqrt(-np.log(self.vicinal_params["nonzero_soft_weight_threshold"])/nu_r)
                lb_soft = max(0.0, lb_soft); ub_soft = min(ub_soft, 1.0)
                assert lb_soft<=ub_soft and lb_soft>=0 and lb_soft<=1 and ub_soft>=0 and ub_soft<=1
                if self.vicinal_params["ada_vic_type"].lower()=="vanilla":
                    if self.vicinal_params["threshold_type"] == "hard":
                        lb, ub = lb_hard, ub_hard
                    else:
                        lb, ub = lb_soft, ub_soft
                elif self.vicinal_params["ada_vic_type"].lower()=="hybrid":
                    # lb, ub = min(lb_hard, lb_soft), max(ub_hard, ub_soft) #hard in soft, not working
                    # lb, ub = lb_hard, ub_hard #soft in hard
                    lb, ub = max(lb_hard, lb_soft), min(ub_hard, ub_soft) #soft in hard, smaller vicinity
                else:
                    raise ValueError('Not supported vicinity type!!!') 
                batch_fake_labels[j] = np.random.uniform(lb, ub, size=1)[0]
            
            ##end for j
            batch_real_labels = self.train_labels[batch_real_indx]
            batch_real_labels = torch.from_numpy(batch_real_labels).type(torch.float).to(self.device)
            batch_fake_labels = torch.from_numpy(batch_fake_labels).type(torch.float).to(self.device)
            
            ## determine vicinal weights for real and fake images              
            if self.vicinal_params["threshold_type"].lower()=="soft" or self.vicinal_params["ada_vic_type"].lower()=="hybrid":
                nu_l_all = torch.from_numpy(1/(kappa_l_all)**2).type(torch.float).to(self.device)
                nu_r_all = torch.from_numpy(1/(kappa_r_all)**2).type(torch.float).to(self.device)
                batch_target_labels = torch.from_numpy(batch_target_labels).type(torch.float).to(self.device)
                indx_left_real = torch.where((batch_real_labels-batch_target_labels)<=0)[0]
                indx_right_real = torch.where((batch_real_labels-batch_target_labels)>0)[0]
                indx_left_fake = torch.where((batch_fake_labels-batch_target_labels)<=0)[0]
                indx_right_fake = torch.where((batch_fake_labels-batch_target_labels)>0)[0]
                real_weights = torch.zeros_like(nu_l_all).type(torch.float).to(self.device)
                real_weights[indx_left_real] = torch.exp(-nu_l_all[indx_left_real]*(batch_real_labels[indx_left_real]-batch_target_labels[indx_left_real])**2)
                real_weights[indx_right_real] = torch.exp(-nu_r_all[indx_right_real]*(batch_real_labels[indx_right_real]-batch_target_labels[indx_right_real])**2)
                fake_weights = torch.zeros_like(nu_r_all).type(torch.float).to(self.device)
                fake_weights[indx_left_fake] = torch.exp(-nu_l_all[indx_left_fake]*(batch_fake_labels[indx_left_fake]-batch_target_labels[indx_left_fake])**2)
                fake_weights[indx_right_fake] = torch.exp(-nu_r_all[indx_right_fake]*(batch_fake_labels[indx_right_fake]-batch_target_labels[indx_right_fake])**2)
                # ## For those weights smaller than threshold, we replace them with the threshold.
                # if self.vicinal_params["ada_vic_type"].lower()=="hybrid":
                #     real_weights[real_weights<self.vicinal_params["nonzero_soft_weight_threshold"]] = self.vicinal_params["nonzero_soft_weight_threshold"]
                #     fake_weights[fake_weights<self.vicinal_params["nonzero_soft_weight_threshold"]] = self.vicinal_params["nonzero_soft_weight_threshold"]
            elif self.vicinal_params["threshold_type"]=="hard":
                real_weights = torch.ones(self.batch_size, dtype=torch.float).to(self.device)
                fake_weights = torch.ones(self.batch_size, dtype=torch.float).to(self.device)
            else:
                raise ValueError('Not supported vicinal weight type!!!') 
        
            return batch_real_indx, batch_fake_labels, batch_real_labels, real_weights, fake_weights, kappa_l_all, kappa_r_all


    #######################################################################################    
    ## training function
    
    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        log_filename = os.path.join(self.results_folder, 'log_loss_steps{}.txt'.format(self.train_num_steps))
        if not os.path.isfile(log_filename):
            logging_file = open(log_filename, "w")
            logging_file.close()
        with open(log_filename, 'a') as file:
            file.write("\n===================================================================================================")

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            start_time = timeit.default_timer()

            while self.step < self.train_num_steps:
                
                total_loss = 0.
                total_denoise_loss = 0.
                total_aux_reg_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    
                    ## no vicinity
                    if self.vicinal_params["kappa"]==0:
                        ## draw real image/label batch from the training set
                        batch_indx = np.random.choice(np.arange(len(self.train_images)), size=self.batch_size, replace=True)
                        batch_labels = self.train_labels[batch_indx]
                        batch_labels = torch.from_numpy(batch_labels).type(torch.float).to(device)
                        real_weights = None
                        batch_target_labels = batch_labels
                        max_kappa = 0
                    ## with vicinity
                    else:
                        ## randomly draw batch_size y's from unique_train_labels
                        batch_target_labels_in_dataset = np.random.choice(self.unique_train_labels, size=self.batch_size, replace=True)
                        
                        ## add Gaussian noise; we estimate image distribution conditional on these labels
                        batch_epsilons = np.random.normal(0, self.vicinal_params["kernel_sigma"], self.batch_size)
                        batch_target_labels = batch_target_labels_in_dataset + batch_epsilons
                        
                        ## make vicinity
                        batch_indx, _, batch_labels, real_weights, _, kappa_l_all, kappa_r_all = self.make_vicinity(batch_target_labels, batch_target_labels_in_dataset)
                        
                        batch_target_labels = torch.from_numpy(batch_target_labels).type(torch.float).to(device)
                        max_kappa = np.maximum(kappa_l_all, kappa_r_all)
                    
                    ## draw real image/label batch from the training set
                    batch_images = self.fn_transform(self.train_images[batch_indx])
                    batch_images = torch.from_numpy(normalize_images(batch_images, to_neg_one_to_one=False)).type(torch.float).to(device) #do not normalized images to [-1,1] here!!! will do it in the diffusion's forward.
                    assert batch_images.min().item()>=0 and batch_images.max().item()<=1.0 
                    
                    if self.mixed_precision_type=="bf16":
                        batch_images = batch_images.bfloat16()
                        batch_target_labels = batch_target_labels.bfloat16()
                    
                    with self.accelerator.autocast():
                        denoise_loss, aux_reg_loss, aux_reg_weight = self.model(images=batch_images, labels=batch_target_labels, labels_emb=self.fn_y2h(batch_target_labels), vicinal_weights = real_weights, max_kappa = max_kappa)
                        loss = denoise_loss + aux_reg_loss * aux_reg_weight #denoising loss with auxiliary regression penalty (with weight)
                        loss = loss / float(self.gradient_accumulate_every)
                        total_loss += loss.item()  
                        total_denoise_loss += denoise_loss.item() / float(self.gradient_accumulate_every)
                        total_aux_reg_loss += aux_reg_loss.item() / float(self.gradient_accumulate_every)
                    
                    self.accelerator.backward(loss)
                    
                ##end for _

                # pbar.set_description(f'loss: {total_loss:.4f}')
                
                pbar.set_description("loss: {:.4f} ({:.4f}/{:.4f})".format(total_loss, total_denoise_loss, total_aux_reg_loss))
                
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                accelerator.wait_for_everyone()
                # accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    
                    if self.step%500==0:
                        with open(log_filename, 'a') as file:
                            file.write("\n Step: {}, Loss: {:.4f} ({:.4f}/{:.4f}), Time: {:.4f} sec.".format(self.step, total_loss, total_denoise_loss, total_aux_reg_loss, timeit.default_timer()-start_time))
                    
                    self.ema.update()
                    if self.step != 0 and divisible_by(self.step, self.sample_every):
                        self.ema.ema_model.eval()
                        with torch.inference_mode():
                            #sde sampler
                            gen_imgs = self.ema.ema_model.sample_using_sde(
                                                                 labels = self.y_visual,
                                                                 labels_emb = self.fn_y2h(self.y_visual), 
                                                                 cond_scale = self.cond_scale_visual,
                                                                 rescaled_phi = self.cond_rescaled_phi_visual,
                                                                 num_sample_steps = self.num_sample_steps, 
                                                                 clamp = False,
                                                                 )
                            gen_imgs = gen_imgs.detach().cpu()
                            assert gen_imgs.min()>=0 and gen_imgs.max()<=1
                            assert gen_imgs.size(1)==self.channels
                            torchvision.utils.save_image(gen_imgs.data, str(self.results_folder) + '/sample_sde_{}.png'.format(self.step), nrow=self.nrow_visual, normalize=False, padding=1)
                            
                            #ode sampler
                            gen_imgs = self.ema.ema_model.sample_using_ode(
                                                                 labels = self.y_visual,
                                                                 labels_emb = self.fn_y2h(self.y_visual), 
                                                                 cond_scale = self.cond_scale_visual,
                                                                 rescaled_phi = self.cond_rescaled_phi_visual,
                                                                 num_sample_steps = self.num_sample_steps, 
                                                                 clamp = False,
                                                                )
                            gen_imgs = gen_imgs.detach().cpu()
                            assert gen_imgs.min()>=0 and gen_imgs.max()<=1
                            assert gen_imgs.size(1)==self.channels
                            torchvision.utils.save_image(gen_imgs.data, str(self.results_folder) + '/sample_ode_{}.png'.format(self.step), nrow=self.nrow_visual, normalize=False, padding=1)

                            # #dpmpp sampler
                            # gen_imgs = self.ema.ema_model.sample_using_dpmpp(
                            #                                      labels = self.y_visual,
                            #                                      labels_emb = self.fn_y2h(self.y_visual), 
                            #                                      cond_scale = self.cond_scale_visual,
                            #                                      rescaled_phi = self.cond_rescaled_phi_visual,
                            #                                      num_sample_steps = self.num_sample_steps, 
                            #                                     )
                            # gen_imgs = gen_imgs.detach().cpu()
                            # assert gen_imgs.min()>=0 and gen_imgs.max()<=1
                            # assert gen_imgs.size(1)==self.channels
                            # torchvision.utils.save_image(gen_imgs.data, str(self.results_folder) + '/sample_dpmpp_{}.png'.format(self.step), nrow=self.nrow_visual, normalize=False, padding=1)


                    if self.step !=0 and divisible_by(self.step, self.save_every):
                        milestone = self.step
                        self.ema.ema_model.eval()
                        self.save(milestone)
                        
                pbar.update(1)
                
            ##end while        
        accelerator.print('training complete \n')
    ###end def train
               
               
    #######################################################################################         
    ## sampling function    
    def sample_given_labels(self, given_labels, batch_size, num_sample_steps=None, denorm=True, to_numpy=False, verbose=False, sampler="sde", cond_scale=1.5, rescaled_phi=0.7):
        
        """
        Generate samples based on given labels
        :given_labels: normalized labels
        """
        
        assert isinstance(given_labels, np.ndarray)
        assert given_labels.min()>=0 and given_labels.max()<=1.0
        nfake = len(given_labels)
        
        if batch_size>nfake:
            batch_size = nfake
        fake_images = []
        assert nfake%batch_size==0
        
        self.ema.ema_model.eval()
        with torch.inference_mode():
            
            tmp = 0
            while tmp < nfake:
                
                batch_fake_labels = torch.from_numpy(given_labels[tmp:(tmp+batch_size)]).type(torch.float).view(-1).cuda()
                                
                if sampler=="sde":                    
                    batch_fake_images = self.ema.ema_model.sample_using_sde(
                                                                 labels = batch_fake_labels,
                                                                 labels_emb = self.fn_y2h(batch_fake_labels), 
                                                                 cond_scale = cond_scale,
                                                                 rescaled_phi = rescaled_phi,
                                                                 num_sample_steps = num_sample_steps, 
                                                                 clamp = False,
                                                                 )
                elif sampler=="ode":                       
                    batch_fake_images = self.ema.ema_model.sample_using_ode(
                                                                 labels = batch_fake_labels,
                                                                 labels_emb = self.fn_y2h(batch_fake_labels), 
                                                                 cond_scale = cond_scale,
                                                                 rescaled_phi = rescaled_phi,
                                                                 num_sample_steps = num_sample_steps, 
                                                                 clamp = False,
                                                                )
                
                elif sampler=="dpmpp":                       
                    batch_fake_images = self.ema.ema_model.sample_using_dpmpp(
                                                                 labels = batch_fake_labels,
                                                                 labels_emb = self.fn_y2h(batch_fake_labels), 
                                                                 cond_scale = cond_scale,
                                                                 rescaled_phi = rescaled_phi,
                                                                 num_sample_steps = num_sample_steps, 
                                                                )
                
                else:
                    raise ValueError("Unsupported Sampler!!!")
                    
                batch_fake_images = batch_fake_images.detach().cpu()
                assert batch_fake_images.min()>=0 and batch_fake_images.max()<=1
                
                if denorm: #denorm imgs to save memory
                    # assert batch_fake_images.max().item()<=1.0 and batch_fake_images.min().item()>=0
                    if batch_fake_images.min()<0 or batch_fake_images.max()>1:
                        print("\r Generated images are out of range. (min={}, max={})".format(batch_fake_images.min(), batch_fake_images.max()))
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