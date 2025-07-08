import numpy as np
import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
import os

import torch
from torch import nn, einsum
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from accelerate import Accelerator

from ema_pytorch import EMA
from utils import cycle, divisible_by, exists, normalize_images, random_hflip, random_rotate, random_vflip
# from moviepy.editor import ImageSequenceClip


class Trainer(object):
    def __init__(
        self,
        data_name,
        diffusion_model,
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
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        sample_every = 1000,
        save_every = 1000,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        max_grad_norm = 1.,
        y_visual = None,
        nrow_visual = 6,
        cond_scale_visual=1.5
    ):
        super().__init__()

        # dataset
        ## training images are not normalized here !!!
        self.data_name = data_name
        self.train_images = train_images
        self.train_labels = train_labels
        self.unique_train_labels = np.sort(np.array(list(set(train_labels))))
        assert train_images.max()>1.0
        assert train_labels.min()>=0 and train_labels.max()<=1.0
        print("\n Training labels' range is [{},{}].".format(train_labels.min(), train_labels.max()))
        
        # vicinal params
        self.kernel_sigma = vicinal_params["kernel_sigma"]
        self.kappa = vicinal_params["kappa"]
        self.threshold_type = vicinal_params["threshold_type"]
        self.nonzero_soft_weight_threshold = vicinal_params["nonzero_soft_weight_threshold"]

        # visualize
        self.y_visual = y_visual
        self.cond_scale_visual = cond_scale_visual
        self.nrow_visual = nrow_visual
        
        # accelerator
        self.accelerator = Accelerator(
            # split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model
        self.model = diffusion_model ##diffusion model instead of unet
        self.channels = diffusion_model.channels

        # sampling and training hyperparameters
        # assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        # self.num_samples = num_samples
        self.sample_every = sample_every
        self.save_every = save_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        assert (train_batch_size * gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.max_grad_norm = max_grad_norm


        # optimizer
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, update_after_step=ema_update_after_step, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            # 'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))
        # torch.save(data, str(self.results_folder / f'model-{self.step}.pt'))

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

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
        
        if return_unet:
            return self.model.model #take unet

    def train(self, fn_y2h):
        accelerator = self.accelerator
        device = accelerator.device

        log_filename = os.path.join(self.results_folder, 'log_loss_niters{}.txt'.format(self.train_num_steps))
        if not os.path.isfile(log_filename):
            logging_file = open(log_filename, "w")
            logging_file.close()
        with open(log_filename, 'a') as file:
            file.write("\n===================================================================================================")

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):

                    ## for no vicinity
                    if self.threshold_type=="hard" and self.kappa==0:
                        ## draw real image/label batch from the training set
                        batch_real_indx = np.random.choice(np.arange(len(self.train_images)), size=self.batch_size, replace=True)
                        batch_images = self.train_images[batch_real_indx]
                        if self.data_name == "UTKFace":
                            batch_images = random_hflip(batch_images)
                        if self.data_name == "Cell200":
                            batch_images = random_rotate(batch_images)
                            batch_images = random_hflip(batch_images)
                            batch_images = random_vflip(batch_images)
                        # if self.data_name == "SteeringAngle":
                        #     batch_images, batch_flipped_indx = random_hflip(batch_images, return_flipped_indx=True) 
                        batch_images = torch.from_numpy(normalize_images(batch_images, to_neg_one_to_one=False)) #In the forward method of the diffusion model, normalization is performed before calculating p_loss. Therefore, it is not necessary to normalize the image to [-1,1] here
                        batch_images = batch_images.type(torch.float).to(device)
                        batch_labels = self.train_labels[batch_real_indx]
                        batch_labels = torch.from_numpy(batch_labels).type(torch.float).to(device)
                        # if self.data_name == "SteeringAngle":
                        #     batch_labels[batch_flipped_indx] = 1 - batch_labels[batch_flipped_indx]

                        with self.accelerator.autocast():
                            loss = self.model(batch_images, labels_emb = fn_y2h(batch_labels), labels = batch_labels, vicinal_weights = None)
                            loss = loss / self.gradient_accumulate_every
                            total_loss += loss.item()

                    ## use the vicinal loss
                    else: 
                        ## randomly draw batch_size_disc y's from unique_train_labels
                        batch_target_labels_in_dataset = np.random.choice(self.unique_train_labels, size=self.batch_size, replace=True)
                        ## add Gaussian noise; we estimate image distribution conditional on these labels
                        batch_epsilons = np.random.normal(0, self.kernel_sigma, self.batch_size)
                        batch_target_labels = batch_target_labels_in_dataset + batch_epsilons

                        ## find index of real images with labels in the vicinity of batch_target_labels
                        ## generate labels for fake image generation; these labels are also in the vicinity of batch_target_labels
                        batch_real_indx = np.zeros(self.batch_size, dtype=int) #index of images in the datata; the labels of these images are in the vicinity
                        
                        for j in range(self.batch_size):
                            ## index for real images
                            if self.threshold_type == "hard":
                                indx_real_in_vicinity = np.where(np.abs(self.train_labels-batch_target_labels[j])<= self.kappa)[0]
                            else:
                                # reverse the weight function for SVDL
                                indx_real_in_vicinity = np.where((self.train_labels-batch_target_labels[j])**2 <= -np.log(self.nonzero_soft_weight_threshold)/self.kappa)[0]

                            ## if the max gap between two consecutive ordered unique labels is large, it is possible that len(indx_real_in_vicinity)<1
                            while len(indx_real_in_vicinity)<1:
                                batch_epsilons_j = np.random.normal(0, self.kernel_sigma, 1)
                                batch_target_labels[j] = batch_target_labels_in_dataset[j] + batch_epsilons_j
                                # batch_target_labels = np.clip(batch_target_labels, 0.0, 1.0)
                                ## index for real images
                                if self.threshold_type == "hard":
                                    indx_real_in_vicinity = np.where(np.abs(self.train_labels-batch_target_labels[j])<= self.kappa)[0]
                                else:
                                    # reverse the weight function for SVDL
                                    indx_real_in_vicinity = np.where((self.train_labels-batch_target_labels[j])**2 <= -np.log(self.nonzero_soft_weight_threshold)/self.kappa)[0]
                            #end while len(indx_real_in_vicinity)<1

                            assert len(indx_real_in_vicinity)>=1

                            batch_real_indx[j] = np.random.choice(indx_real_in_vicinity, size=1)[0]
                        ##end for j

                        ## draw real image/label batch from the training set
                        batch_target_labels = torch.from_numpy(batch_target_labels).type(torch.float).cuda()
                        batch_images = self.train_images[batch_real_indx]
                        if self.data_name == "UTKFace":
                            batch_images = random_hflip(batch_images)
                        if self.data_name == "Cell200":
                            batch_images = random_rotate(batch_images)
                            batch_images = random_hflip(batch_images)
                            batch_images = random_vflip(batch_images)
                        # if self.data_name == "SteeringAngle":
                        #     batch_images, batch_flipped_indx = random_hflip(batch_images, return_flipped_indx=True)
                        batch_images = torch.from_numpy(normalize_images(batch_images, to_neg_one_to_one=False)) #In the forward method of the diffusion model, normalization is performed before calculating p_loss. Therefore, it is not necessary to normalize the image to [-1,1] here
                        batch_images = batch_images.type(torch.float).cuda()
                        batch_labels = self.train_labels[batch_real_indx]
                        batch_labels = torch.from_numpy(batch_labels).type(torch.float).cuda()
                        # if self.data_name == "SteeringAngle":
                        #     batch_labels[batch_flipped_indx] = 1 - batch_labels[batch_flipped_indx]

                        ## weight vector
                        if self.threshold_type == "soft":
                            vicinal_weights = torch.exp(-self.kappa*(batch_labels-batch_target_labels)**2).cuda()
                        else:
                            vicinal_weights = torch.ones(self.batch_size, dtype=torch.float).cuda()

                        ## define loss
                        with self.accelerator.autocast():
                            loss = self.model(batch_images, labels_emb = fn_y2h(batch_labels), labels = batch_labels, vicinal_weights = vicinal_weights)
                            loss = loss / self.gradient_accumulate_every
                            total_loss += loss.item()
                
                    ##end if
                    self.accelerator.backward(loss)
                
                ##end for

                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                pbar.set_description(f'loss: {total_loss:.4f}')

                if self.step%500==0:
                    with open(log_filename, 'a') as file:
                        file.write("\n Step: {}, Loss: {:.4f}.".format(self.step, total_loss))

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.sample_every):
                        self.ema.ema_model.eval()
                        with torch.inference_mode():
                            # ## ddpm sampler
                            # gen_imgs = self.ema.ema_model.sample(
                            #             labels_emb=fn_y2h(self.y_visual), 
                            #             labels = self.y_visual,
                            #             cond_scale = self.cond_scale_visual
                            #             )

                            ## ddim sampler
                            gen_imgs = self.ema.ema_model.ddim_sample(
                                        labels_emb = fn_y2h(self.y_visual),
                                        labels = self.y_visual,
                                        shape = (self.y_visual.shape[0], self.channels, self.image_size, self.image_size),
                                        cond_scale = self.cond_scale_visual,
                                        # preset_sampling_timesteps = 250,
                                        # preset_ddim_sampling_eta = 0, # 1 for ddpm, 0 for ddim
                                        )
                            
                            gen_imgs = gen_imgs.detach().cpu()
                            # assert gen_imgs.min()>=0 and gen_imgs.max()<=1
                            if gen_imgs.min()<0 or gen_imgs.max()>1:
                                print("\r Generated images are out of range. (min={}, max={})".format(gen_imgs.min(), gen_imgs.max()))
                            gen_imgs = torch.clip(gen_imgs,0,1)
                            assert gen_imgs.size(1)==self.channels
                            # gen_imgs = gen_imgs[:,[2, 1, 0],:,:] # BGR-->RGB
                            utils.save_image(gen_imgs.data, str(self.results_folder) + '/sample_{}.png'.format(self.step), nrow=self.nrow_visual, normalize=False, padding=1)
                    
                    if self.step !=0 and divisible_by(self.step, self.save_every):
                        milestone = self.step
                        self.ema.ema_model.eval()
                        self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')
    ## end def



    def sample_given_labels(self, given_labels, fn_y2h, batch_size, denorm=True, to_numpy=False, verbose=False, sampler="ddpm", cond_scale=6.0, sample_timesteps=1000, ddim_eta = 0):
        """
        Generate samples based on given labels
        :given_labels: normalized labels
        :fn_y2h: label embedding function
        """
        accelerator = self.accelerator
        device = accelerator.device

        assert given_labels.min()>=0 and given_labels.max()<=1.0
        nfake = len(given_labels)

        if batch_size>nfake:
            batch_size = nfake
        fake_images = []
        assert nfake%batch_size==0

        tmp = 0
        while tmp < nfake:
            batch_fake_labels = torch.from_numpy(given_labels[tmp:(tmp+batch_size)]).type(torch.float).view(-1).cuda()
            self.ema.ema_model.eval()
            with torch.inference_mode():
                if sampler == "ddpm":
                    batch_fake_images = self.ema.ema_model.sample(
                                        labels_emb=fn_y2h(batch_fake_labels),
                                        labels = batch_fake_labels,
                                        cond_scale = cond_scale, 
                                        # preset_sampling_timesteps=sample_timesteps,
                                        )
                elif sampler == "ddim":
                    batch_fake_images = self.ema.ema_model.ddim_sample(
                                        labels_emb=fn_y2h(batch_fake_labels),
                                        labels = batch_fake_labels,
                                        shape = (batch_fake_labels.shape[0], self.channels, self.image_size, self.image_size),
                                        cond_scale = cond_scale,
                                        # preset_sampling_timesteps = sample_timesteps,
                                        # preset_ddim_sampling_eta = ddim_eta, # 1 for ddpm, 0 for ddim
                                        )

                batch_fake_images = batch_fake_images.detach().cpu()

            if denorm: #denorm imgs to save memory
                # assert batch_fake_images.max().item()<=1.0 and batch_fake_images.min().item()>=0
                if batch_fake_images.min()<0 or batch_fake_images.max()>1:
                    print("\r Generated images are out of range. (min={}, max={})".format(batch_fake_images.min(), batch_fake_images.max()))
                batch_fake_images = torch.clip(batch_fake_images, 0, 1)
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

    ## end def


    # def generate_intermediate_gifs(self, path_to_save, given_labels, fn_y2h, fps=20, sampler="ddpm", cond_scale=6.0, sample_timesteps=1000, ddim_eta = 0):
    #     """
    #     Generate intermediate noisy images in the GIF format
    #     :given_labels: normalized labels
    #     :fn_y2h: label embedding function
    #     """
    #     accelerator = self.accelerator
    #     device = accelerator.device

    #     assert given_labels.min()>=0 and given_labels.max()<=1.0
    #     nfake = len(given_labels)

    #     self.ema.ema_model.eval()
    #     with torch.inference_mode():
    #         if sampler == "ddpm":
    #             _, frames = self.ema.ema_model.sample(
    #                                 labels_emb = fn_y2h(given_labels),
    #                                 labels = given_labels,
    #                                 cond_scale = cond_scale,
    #                                 save_intermediate=True
    #                                 )
    #         elif sampler == "ddim":
    #             _, frames = self.ema.ema_model.ddim_sample(
    #                                 labels_emb = fn_y2h(given_labels),
    #                                 labels = given_labels,
    #                                 shape = (given_labels.shape[0], self.channels, self.image_size, self.image_size),
    #                                 cond_scale = cond_scale,
    #                                 # preset_sampling_timesteps = sample_timesteps,
    #                                 # preset_ddim_sampling_eta = ddim_eta, # 1 for ddpm, 0 for ddim
    #                                 save_intermediate=True
    #                                 )

    #     clip = ImageSequenceClip(list(frames), fps=fps)
    #     clip.write_gif(path_to_save, fps=fps)
            
    # ## end def







    