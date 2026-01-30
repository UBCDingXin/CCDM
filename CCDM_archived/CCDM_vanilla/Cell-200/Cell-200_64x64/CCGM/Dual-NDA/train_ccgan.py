import torch
import numpy as np
import os
import timeit
from PIL import Image
from torchvision.utils import save_image
import torch.cuda as cutorch
import torchvision.transforms as transforms
import random
import h5py
import gc

from accelerate import Accelerator

from utils import SimpleProgressBar, IMGs_dataset
from opts import parse_opts

''' Settings '''
args = parse_opts()

# some parameters in opts
gan_arch = args.GAN_arch
loss_type = args.loss_type_gan
niters = args.niters_gan
resume_niters = args.resume_niters_gan
dim_gan = args.dim_gan
lr_g = args.lr_g_gan
lr_d = args.lr_d_gan
save_niters_freq = args.save_niters_freq
batch_size_disc = args.batch_size_disc
batch_size_gene = args.batch_size_gene
# batch_size_max = max(batch_size_disc, batch_size_gene)
num_D_steps = args.num_D_steps

## grad accumulation
num_grad_acc_d = args.num_grad_acc_d
num_grad_acc_g = args.num_grad_acc_g

visualize_freq = args.visualize_freq

num_workers = args.num_workers

threshold_type = args.threshold_type
nonzero_soft_weight_threshold = args.nonzero_soft_weight_threshold

transform = args.transform
num_channels = args.num_channels
img_size = args.img_size
max_label = args.max_label

use_amp = args.use_amp #mixed precision

## NDA
nda_a = args.nda_a # fake loss
nda_b = args.nda_b # real-fake loss
nda_c = args.nda_c # real-wronglable loss
nda_d = args.nda_d # bad fake loss, NIQE filtering
nda_e = args.nda_e # bad fake loss, MAE filtering
assert np.abs(nda_a+nda_b+nda_c+nda_d+nda_e-1)<1e-6
nda_c_quantile = args.nda_c_quantile
nda_d_nfake = args.nda_d_nfake
nda_e_nfake = args.nda_e_nfake

## finetune
do_finetune = args.GAN_finetune
path_ckpt = args.path_GAN_ckpt
nda_start_iter = args.nda_start_iter



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
        return (labels*args.max_label).astype(int)
    elif torch.is_tensor(labels):
        return (labels*args.max_label).type(torch.int)
    else:
        return int(labels*args.max_label)

if nda_d>0 and not args.eval_mode:
    assert "NIQE" in args.path2badfake1
    hf = h5py.File(args.path2badfake1, 'r') #niters=40000, NIQE filtering
    badfake_labels_1 = hf['fake_labels'][:]
    badfake_labels_1 = badfake_labels_1.astype(float)
    badfake_images_1 = hf['fake_images'][:]
    hf.close()

    if args.path2badfake3!="None":
        assert "NIQE" in args.path2badfake3
        hf = h5py.File(args.path2badfake3, 'r') #niters=10000, NIQE filtering
        fake_labels_3 = hf['fake_labels'][:]
        fake_labels_3 = fake_labels_3.astype(float)
        fake_images_3 = hf['fake_images'][:]
        hf.close()
        badfake_images_1 = np.concatenate((badfake_images_1, fake_images_3), axis=0)
        badfake_labels_1 = np.concatenate((badfake_labels_1, fake_labels_3), axis=0)
        del fake_images_3, fake_labels_3; gc.collect()

    if nda_d_nfake>0:
        indx_tmp_1 = np.arange(len(badfake_images_1))
        np.random.shuffle(indx_tmp_1)
        indx_tmp_1 = indx_tmp_1[0:nda_d_nfake]
        badfake_images_1 = badfake_images_1[indx_tmp_1]
        badfake_labels_1 = badfake_labels_1[indx_tmp_1]
    
    badfake_labels_1 = fn_norm_labels(badfake_labels_1)
    indx_badfake_1 = np.arange(len(badfake_images_1))

if nda_e>0 and not args.eval_mode:
    assert "MAE" in args.path2badfake2
    hf = h5py.File(args.path2badfake2, 'r') #niters=40000, MAE filtering
    badfake_labels_2 = hf['fake_labels'][:]
    badfake_labels_2 = badfake_labels_2.astype(float)
    badfake_images_2 = hf['fake_images'][:]
    hf.close()

    if args.path2badfake4!="None":
        assert "MAE" in args.path2badfake4
        hf = h5py.File(args.path2badfake4, 'r') #niters=10000, MAE filtering
        fake_labels_4 = hf['fake_labels'][:]
        fake_labels_4 = fake_labels_4.astype(float)
        fake_images_4 = hf['fake_images'][:]
        hf.close()
        badfake_images_2 = np.concatenate((badfake_images_2, fake_images_4), axis=0)
        badfake_labels_2 = np.concatenate((badfake_labels_2, fake_labels_4), axis=0)
        del fake_images_4, fake_labels_4; gc.collect()

    if nda_e_nfake>0:
        indx_tmp_2 = np.arange(len(badfake_images_2))
        np.random.shuffle(indx_tmp_2)
        indx_tmp_2 = indx_tmp_2[0:nda_e_nfake]
        badfake_images_2 = badfake_images_2[indx_tmp_2]
        badfake_labels_2 = badfake_labels_2[indx_tmp_2]
    
    badfake_labels_2 = fn_norm_labels(badfake_labels_2)
    indx_badfake_2 = np.arange(len(badfake_images_2))
    


#########################################
# Necessary functions
## horizontal flip images
def hflip_images(batch_images):
    ''' for numpy arrays '''
    uniform_threshold = np.random.uniform(0,1,len(batch_images))
    indx_gt = np.where(uniform_threshold>0.5)[0]
    batch_images[indx_gt] = np.flip(batch_images[indx_gt], axis=3)
    return batch_images
# def hflip_images(batch_images):
#     ''' for torch tensors '''
#     uniform_threshold = np.random.uniform(0,1,len(batch_images))
#     indx_gt = np.where(uniform_threshold>0.5)[0]
#     batch_images[indx_gt] = torch.flip(batch_images[indx_gt], dims=[3])
#     return batch_images

## normalize images
def normalize_images(batch_images):
    batch_images = batch_images/255.0
    batch_images = (batch_images - 0.5)/0.5
    return batch_images


def get_perm(l) :
    perm = torch.randperm(l)
    while torch.all(torch.eq(perm, torch.arange(l))) :
        perm = torch.randperm(l)
    return perm

def jigsaw_k(data, k = 2) :
    with torch.no_grad() :
        actual_h = data.size()[2]
        actual_w = data.size()[3]
        h = torch.split(data, int(actual_h/k), dim = 2)
        splits = []
        for i in range(k) :
            splits += torch.split(h[i], int(actual_w/k), dim = 3)
        fake_samples = torch.stack(splits, -1)
        for idx in range(fake_samples.size()[0]) :
            perm = get_perm(k*k)
            # fake_samples[idx] = fake_samples[idx,:,:,:,torch.randperm(k*k)]
            fake_samples[idx] = fake_samples[idx,:,:,:,perm]
        fake_samples = torch.split(fake_samples, 1, dim=4)
        merged = []
        for i in range(k) :
            merged += [torch.cat(fake_samples[i*k:(i+1)*k], 2)]
        fake_samples = torch.squeeze(torch.cat(merged, 3), -1)
        return fake_samples

def stitch(data, k = 2) :
    #  = torch.randperm()
    indices = get_perm(data.size(0))
    data_perm = data[indices]
    actual_h = data.size()[2]
    actual_w = data.size()[3]
    if torch.randint(0, 2, (1,))[0].item() == 0 :
        dim0, dim1 = 2,3
    else :
        dim0, dim1 = 3,2

    h = torch.split(data, int(actual_h/k), dim = dim0)
    h_1 = torch.split(data_perm, int(actual_h/k), dim = dim0)
    splits = []
    for i in range(k) :
        if i < int(k/2) :
            splits += torch.split(h[i], int(actual_w/k), dim = dim1)
        else :
            splits += torch.split(h_1[i], int(actual_w/k), dim = dim1)
    merged = []
    for i in range(k) :
        merged += [torch.cat(splits[i*k:(i+1)*k], dim1)]
    fake_samples = torch.cat(merged, dim0)

    return fake_samples

def mixup(data, alpha = 25.0) :
    lamb = np.random.beta(alpha, alpha)
    # indices = torch.randperm(data.size(0))
    indices = get_perm(data.size(0))
    data_perm = data[indices]
    return data*lamb + (1-lamb)*data_perm

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutout(data):
    min_k, max_k = 10, 20
    data = data.clone()
    h, w = data.size(2), data.size(3)
    b_size = data.size(0)
    for i in range(b_size) :
        k = (min_k + (max_k - min_k) * torch.rand(1)).long().item()
        h_pos = ((h - k) * torch.rand(1)).long().item()
        w_pos = ((w - k) * torch.rand(1)).long().item()
        patch = data[i,:,h_pos:h_pos+k,w_pos:w_pos+k]
        patch_mean = torch.mean(patch, axis = (1,2), keepdim = True)
        data[i,:,h_pos:h_pos+k,w_pos:w_pos+k] = torch.ones_like(patch) * patch_mean

    return data

def cut_mix(data, beta = 1.0) :
    data = data.clone()
    lam = np.random.beta(beta, beta)
    indices = get_perm(data.size(0))
    data_perm = data[indices]
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data_perm[:, :, bbx1:bbx2, bby1:bby2]
    return data

def rotate(data, angle = 60) :
    batch_size = data.size(0)
    new_data = []
    for i in range(batch_size) :
        pil_img = transforms.ToPILImage()(data[i].cpu())
        img_rotated = transforms.functional.rotate(pil_img, angle)
        new_data.append(transforms.ToTensor()(img_rotated))
    new_data = torch.stack(new_data, 0)
    return new_data







#########################################
# The training function
def train_ccgan(kernel_sigma, kappa, train_images, train_labels, netG, netD, net_y2h, save_images_folder, save_models_folder = None, clip_label=False):

    '''
    Note that train_images are not normalized to [-1,1]
    '''

    accelerator = Accelerator(
        mixed_precision = 'fp16' if use_amp else 'no'
    )
    device = accelerator.device

    netG = netG.to(device)
    netD = netD.to(device)
    net_y2h = net_y2h.to(device)
    net_y2h.eval()

    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr_d, betas=(0.5, 0.999))

    netG = accelerator.prepare(netG)
    netD = accelerator.prepare(netD)
    optimizerG = accelerator.prepare(optimizerG)
    optimizerD = accelerator.prepare(optimizerD)

    if do_finetune:
        assert path_ckpt!="None"
        checkpoint = torch.load(path_ckpt)
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])
    #end if

    if save_models_folder is not None and resume_niters>0:
        save_file = save_models_folder + "/ckpts_in_train/ckpt_niters_{}.pth".format(resume_niters)
        checkpoint = torch.load(save_file)
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])
    #end if

    #################
    unique_train_labels = np.sort(np.array(list(set(train_labels))))

    # printed images with labels between the 5-th quantile and 95-th quantile of training labels
    n_row=10; n_col = n_row
    z_fixed = torch.randn(n_row*n_col, dim_gan, dtype=torch.float).to(device)
    start_label = np.quantile(train_labels, 0.05)
    end_label = np.quantile(train_labels, 0.95)
    selected_labels = np.linspace(start_label, end_label, num=n_row)
    y_fixed = np.zeros(n_row*n_col)
    for i in range(n_row):
        curr_label = selected_labels[i]
        for j in range(n_col):
            y_fixed[i*n_col+j] = curr_label
    print(y_fixed)
    y_fixed = torch.from_numpy(y_fixed).type(torch.float).view(-1,1).to(device)


    start_time = timeit.default_timer()
    for niter in range(resume_niters, niters):

        '''  Train Discriminator   '''
        for step_D_index in range(num_D_steps):

            optimizerD.zero_grad()

            for accumulation_index in range(num_grad_acc_d):

                ## randomly draw batch_size_disc y's from unique_train_labels
                batch_target_labels_in_dataset = np.random.choice(unique_train_labels, size=batch_size_disc, replace=True)
                ## add Gaussian noise; we estimate image distribution conditional on these labels
                batch_epsilons = np.random.normal(0, kernel_sigma, batch_size_disc)
                batch_target_labels = batch_target_labels_in_dataset + batch_epsilons

                ## find index of real images with labels in the vicinity of batch_target_labels
                ## generate labels for fake image generation; these labels are also in the vicinity of batch_target_labels
                batch_real_indx = np.zeros(batch_size_disc, dtype=int) #index of images in the datata; the labels of these images are in the vicinity
                batch_fake_labels = np.zeros(batch_size_disc)
                if nda_c>0 and niter>=nda_start_iter:
                    batch_real_wronglabel_indx = np.zeros(batch_size_disc, dtype=int) #index of images in the data; the labels of these images should be outside the vicinity
                if nda_d>0 and niter>=nda_start_iter:
                    batch_badfake_niqe_indx = np.zeros(batch_size_disc, dtype=int)
                if nda_e>0 and niter>=nda_start_iter:
                    batch_badfake_mae_indx = np.zeros(batch_size_disc, dtype=int)

                for j in range(batch_size_disc):
                    
                    ## index for real images
                    if threshold_type == "hard":
                        indx_real_in_vicinity = np.where(np.abs(train_labels-batch_target_labels[j])<= kappa)[0]
                    else:
                        # reverse the weight function for SVDL
                        indx_real_in_vicinity = np.where((train_labels-batch_target_labels[j])**2 <= -np.log(nonzero_soft_weight_threshold)/kappa)[0]

                    ## if the max gap between two consecutive ordered unique labels is large, it is possible that len(indx_real_in_vicinity)<1
                    while len(indx_real_in_vicinity)<1:
                        batch_epsilons_j = np.random.normal(0, kernel_sigma, 1)
                        batch_target_labels[j] = batch_target_labels_in_dataset[j] + batch_epsilons_j
                        if clip_label:
                            batch_target_labels = np.clip(batch_target_labels, 0.0, 1.0)
                        ## index for real images
                        if threshold_type == "hard":
                            indx_real_in_vicinity = np.where(np.abs(train_labels-batch_target_labels[j])<= kappa)[0]
                        else:
                            # reverse the weight function for SVDL
                            indx_real_in_vicinity = np.where((train_labels-batch_target_labels[j])**2 <= -np.log(nonzero_soft_weight_threshold)/kappa)[0]
                    #end while len(indx_real_in_vicinity)<1
                    assert len(indx_real_in_vicinity)>=1
                    batch_real_indx[j] = np.random.choice(indx_real_in_vicinity, size=1)[0]


                    ## labels for fake images generation
                    if threshold_type == "hard":
                        lb = batch_target_labels[j] - kappa
                        ub = batch_target_labels[j] + kappa
                    else:
                        lb = batch_target_labels[j] - np.sqrt(-np.log(nonzero_soft_weight_threshold)/kappa)
                        ub = batch_target_labels[j] + np.sqrt(-np.log(nonzero_soft_weight_threshold)/kappa)
                    lb = max(0.0, lb); ub = min(ub, 1.0)
                    assert lb<=ub
                    assert lb>=0 and ub>=0
                    assert lb<=1 and ub<=1
                    batch_fake_labels[j] = np.random.uniform(lb, ub, size=1)[0]


                    ### use hard vicinity for real_wronglabel and badfake
                    if threshold_type == "hard":
                        original_kappa = kappa
                    else:
                        original_kappa = np.sqrt(1/kappa)
                    
                    #real_wronglabel
                    if nda_c>0 and niter>=nda_start_iter: 
                        # indx_real_out_vicinity = np.where(np.abs(train_labels-batch_target_labels[j])> original_kappa)[0]
                        mae_all = np.abs(train_labels-batch_target_labels[j])
                        mae_cutoff_point = np.quantile(mae_all, q=nda_c_quantile) 
                        indx_real_out_vicinity = np.where(mae_all > mae_cutoff_point)[0]
                        assert len(indx_real_out_vicinity)>=1
                        batch_real_wronglabel_indx[j] = np.random.choice(indx_real_out_vicinity, size=1)[0]

                    # badfake niqe filtering
                    if nda_d>0 and niter>=nda_start_iter:
                        nda_radius = original_kappa
                        indx_badfake_in_vicinity = np.where(np.abs(badfake_labels_1-batch_target_labels[j])<= nda_radius)[0]
                        while len(indx_badfake_in_vicinity)<1:
                            # print(nda_radius,"flag!")
                            nda_radius *= 1.05
                            indx_badfake_in_vicinity = np.where(np.abs(badfake_labels_1-batch_target_labels[j])<= nda_radius)[0]
                        assert len(indx_badfake_in_vicinity)>=1
                        batch_badfake_niqe_indx[j] = np.random.choice(indx_badfake_in_vicinity, size=1)[0]

                    # badfake mae filtering
                    if nda_e>0 and niter>=nda_start_iter:
                        nda_radius = original_kappa
                        indx_badfake_in_vicinity = np.where(np.abs(badfake_labels_2-batch_target_labels[j])<= nda_radius)[0]
                        while len(indx_badfake_in_vicinity)<1:
                            nda_radius *= 1.05
                            indx_badfake_in_vicinity = np.where(np.abs(badfake_labels_2-batch_target_labels[j])<= nda_radius)[0]
                        assert len(indx_badfake_in_vicinity)>=1
                        batch_badfake_mae_indx[j] = np.random.choice(indx_badfake_in_vicinity, size=1)[0]

                #end for j

                ############################################
                ## draw real image/label batch from the training set
                batch_real_images = train_images[batch_real_indx]
                trainset = IMGs_dataset(batch_real_images, labels=None, normalize=True, transform=transform)
                train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_disc, shuffle=False)
                train_dataloader = iter(train_dataloader)
                batch_real_images = next(train_dataloader)
                batch_real_images = batch_real_images.type(torch.float)

                
                batch_real_labels = train_labels[batch_real_indx]
                batch_real_labels = torch.from_numpy(batch_real_labels).type(torch.float).to(device)

                ############################################
                ## generate the fake image batch, nda_a
                batch_fake_labels = torch.from_numpy(batch_fake_labels).type(torch.float).to(device)
                z = torch.randn(batch_size_disc, dim_gan, dtype=torch.float).to(device)
                batch_fake_images = netG(z, net_y2h(batch_fake_labels))
                batch_fake_images = batch_fake_images.cpu()

                ############################################
                ## NDA: build real-fake samples, nda_b
                if nda_b>0 and niter>=nda_start_iter:
                    # indx_rand = random.randint(0,2)
                    # if indx_rand == 0:
                    #     batch_real_fake_images = jigsaw_k(batch_real_images, k = 2)
                    # elif indx_rand == 1:
                    #     batch_real_fake_images = stitch(batch_real_images, k = 2)
                    # elif indx_rand == 2:
                    #     batch_real_fake_images = cutout(batch_real_images)
                    batch_real_fake_images = jigsaw_k(batch_real_images, k = 2)
                    # batch_real_fake_images = stitch(batch_real_images, k = 2)
                    # batch_real_fake_images = mixup(batch_real_images, alpha = 25.0)
                    # batch_real_fake_images = cutout(batch_real_images)
                    # batch_real_fake_images = cut_mix(batch_real_images)
                ##end if

                ############################################
                ## NDA: build real-wronglabel samples, nda_c
                if nda_c>0 and niter>=nda_start_iter:
                    # batch_real_wronglabel_images = torch.from_numpy(normalize_images(hflip_images(train_images[batch_real_wronglabel_indx])))
                    # batch_real_wronglabel_images = batch_real_wronglabel_images.type(torch.float)
                    # # batch_real_wronglabel_labels = train_labels[batch_real_indx] #!!!note: assign wrong labels to the selected images!!!
                    # # batch_real_wronglabel_labels = torch.from_numpy(batch_real_wronglabel_labels).type(torch.float).to(device)

                    batch_real_wronglabel_images = train_images[batch_real_wronglabel_indx]
                    trainset = IMGs_dataset(batch_real_wronglabel_images, labels=None, normalize=True, transform=transform)
                    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=len(batch_real_wronglabel_indx), shuffle=False)
                    train_dataloader = iter(train_dataloader)
                    batch_real_wronglabel_images = next(train_dataloader)
                    batch_real_wronglabel_images = batch_real_wronglabel_images.type(torch.float)


                ############################################
                ## NDA: build bad-fake samples, nda_d
                if nda_d>0 and niter>=nda_start_iter:
                    # batch_badfake_niqe_indx = np.random.choice(indx_badfake_1, size=batch_size_disc, replace=True)
                    # batch_badfake_images_1 = torch.from_numpy(normalize_images(hflip_images(badfake_images_1[batch_badfake_niqe_indx])))
                    # batch_badfake_images_1 = batch_badfake_images_1.type(torch.float)
                    # # batch_badfake_labels_1 = badfake_labels_1[batch_badfake_niqe_indx]
                    # # batch_badfake_labels_1 = torch.from_numpy(batch_badfake_labels_1).type(torch.float).to(device)

                    batch_badfake_images_1 = train_images[batch_badfake_niqe_indx]
                    trainset = IMGs_dataset(batch_badfake_images_1, labels=None, normalize=True, transform=transform)
                    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=len(batch_badfake_niqe_indx), shuffle=False)
                    train_dataloader = iter(train_dataloader)
                    batch_badfake_images_1 = next(train_dataloader)
                    batch_badfake_images_1 = batch_badfake_images_1.type(torch.float)

                if nda_e>0 and niter>=nda_start_iter:
                    # batch_badfake_mae_indx = np.random.choice(indx_badfake_2, size=batch_size_disc, replace=True)
                    batch_badfake_images_2 = torch.from_numpy(normalize_images(hflip_images(badfake_images_2[batch_badfake_mae_indx])))
                    batch_badfake_images_2 = batch_badfake_images_2.type(torch.float)
                    # batch_badfake_labels_2 = badfake_labels_2[batch_badfake_mae_indx]
                    # batch_badfake_labels_2 = torch.from_numpy(batch_badfake_labels_2).type(torch.float).to(device)

                ## target labels on gpu
                batch_target_labels = torch.from_numpy(batch_target_labels).type(torch.float).to(device)

                ## weight vector
                if threshold_type == "soft":
                    real_weights = torch.exp(-kappa*(batch_real_labels-batch_target_labels)**2).to(device)
                    fake_weights = torch.exp(-kappa*(batch_fake_labels-batch_target_labels)**2).to(device)
                else:
                    real_weights = torch.ones(batch_size_disc, dtype=torch.float).to(device)
                    fake_weights = torch.ones(batch_size_disc, dtype=torch.float).to(device)
                #end if threshold type
                if nda_b>0 and niter>=nda_start_iter:
                    real_fake_weights = torch.ones(batch_size_disc, dtype=torch.float).to(device)
                if nda_c>0 and niter>=nda_start_iter:
                    real_wronglabel_weights = torch.ones(batch_size_disc, dtype=torch.float).to(device)
                if nda_d>0 and niter>=nda_start_iter:
                    real_badfake_weights_1 = torch.ones(batch_size_disc, dtype=torch.float).to(device)
                if nda_e>0 and niter>=nda_start_iter:
                    real_badfake_weights_2 = torch.ones(batch_size_disc, dtype=torch.float).to(device)

                # forward pass
                real_dis_out = netD(batch_real_images.to(device), net_y2h(batch_target_labels))
                fake_dis_out = netD(batch_fake_images.to(device).detach(), net_y2h(batch_target_labels))
                if nda_b>0 and niter>=nda_start_iter:
                    real_fake_dis_out = netD(batch_real_fake_images.to(device), net_y2h(batch_target_labels.to(device)))
                if nda_c>0 and niter>=nda_start_iter:
                    real_wronglabel_dis_out = netD(batch_real_wronglabel_images.to(device), net_y2h(batch_target_labels.to(device)))
                if nda_d>0 and niter>=nda_start_iter:
                    badfake_dis_out_1 = netD(batch_badfake_images_1.to(device), net_y2h(batch_target_labels.to(device)))
                if nda_e>0 and niter>=nda_start_iter:
                    badfake_dis_out_2 = netD(batch_badfake_images_2.to(device), net_y2h(batch_target_labels.to(device)))


                if loss_type == "vanilla":
                    real_dis_out = torch.nn.Sigmoid()(real_dis_out)
                    fake_dis_out = torch.nn.Sigmoid()(fake_dis_out)
                    if nda_b>0 and niter>=nda_start_iter:
                        real_fake_dis_out = torch.nn.Sigmoid()(real_fake_dis_out)
                    if nda_c>0 and niter>=nda_start_iter:
                        real_wronglabel_dis_out = torch.nn.Sigmoid()(real_wronglabel_dis_out)
                    if nda_d>0 and niter>=nda_start_iter:
                        badfake_dis_out_1 = torch.nn.Sigmoid()(badfake_dis_out_1)
                    if nda_e>0 and niter>=nda_start_iter:
                        badfake_dis_out_2 = torch.nn.Sigmoid()(badfake_dis_out_2)

                    d_loss_real = - torch.log(real_dis_out+1e-20)
                    d_loss_fake = - torch.log(1-fake_dis_out+1e-20)
                    if nda_b>0 and niter>=nda_start_iter:
                        d_loss_real_fake = - torch.log(1-real_fake_dis_out+1e-20)
                    if nda_c>0 and niter>=nda_start_iter:
                        d_loss_real_wronglabel = - torch.log(1-real_wronglabel_dis_out+1e-20)
                    if nda_d>0 and niter>=nda_start_iter:
                        d_loss_badfake_1 = - torch.log(1-badfake_dis_out_1+1e-20)
                    if nda_e>0 and niter>=nda_start_iter:
                        d_loss_badfake_2 = - torch.log(1-badfake_dis_out_2+1e-20)

                elif loss_type == "hinge":
                    d_loss_real = torch.nn.ReLU()(1.0 - real_dis_out)
                    d_loss_fake = torch.nn.ReLU()(1.0 + fake_dis_out)
                    if nda_b>0 and niter>=nda_start_iter:
                        d_loss_real_fake = torch.nn.ReLU()(1.0 + real_fake_dis_out)
                    if nda_c>0 and niter>=nda_start_iter:
                        d_loss_real_wronglabel = torch.nn.ReLU()(1.0 + real_wronglabel_dis_out)
                    if nda_d>0 and niter>=nda_start_iter:
                        d_loss_badfake_1 = torch.nn.ReLU()(1.0 + badfake_dis_out_1)
                    if nda_e>0 and niter>=nda_start_iter:
                        d_loss_badfake_2 = torch.nn.ReLU()(1.0 + badfake_dis_out_2)

                else:
                    raise ValueError('Not supported loss type!!!')

                d_loss = torch.mean(real_weights.view(-1) * d_loss_real.view(-1)) + nda_a * torch.mean(fake_weights.view(-1) * d_loss_fake.view(-1))
                # loss_show = []
                if nda_b>0 and niter>=nda_start_iter:
                    d_loss_b = nda_b * torch.mean(real_fake_weights.view(-1) * d_loss_real_fake.view(-1))
                    d_loss += d_loss_b
                    # loss_show.append(d_loss_b.cpu().item())
                if nda_c>0 and niter>=nda_start_iter:
                    d_loss_c = nda_c * torch.mean(real_wronglabel_weights.view(-1) * d_loss_real_wronglabel.view(-1))
                    d_loss += d_loss_c
                    # loss_show.append(d_loss_c.cpu().item())
                if nda_d>0 and niter>=nda_start_iter:
                    d_loss_d = nda_d * torch.mean(real_badfake_weights_1.view(-1) * d_loss_badfake_1.view(-1))
                    d_loss += d_loss_d
                    # loss_show.append(d_loss_d.cpu().item())
                if nda_e>0 and niter>=nda_start_iter:
                    d_loss_e = nda_e * torch.mean(real_badfake_weights_2.view(-1) * d_loss_badfake_2.view(-1))
                    d_loss += d_loss_e
                #     loss_show.append(d_loss_e.cpu().item())
                # print(loss_show)

                d_loss /= float(num_grad_acc_d)

                # d_loss.backward()
                accelerator.backward(d_loss)
            ##end for

            optimizerD.step()

        #end for step_D_index



        '''  Train Generator   '''
        netG.train()

        optimizerG.zero_grad()

        for accumulation_index in range(num_grad_acc_g):

            # generate fake images
            ## randomly draw batch_size_gene y's from unique_train_labels
            batch_target_labels_in_dataset = np.random.choice(unique_train_labels, size=batch_size_gene, replace=True)
            ## add Gaussian noise; we estimate image distribution conditional on these labels
            batch_epsilons = np.random.normal(0, kernel_sigma, batch_size_gene)
            batch_target_labels = batch_target_labels_in_dataset + batch_epsilons
            batch_target_labels = torch.from_numpy(batch_target_labels).type(torch.float).to(device)

            z = torch.randn(batch_size_gene, dim_gan, dtype=torch.float).to(device)
            batch_fake_images = netG(z, net_y2h(batch_target_labels))

            # loss
            dis_out = netD(batch_fake_images, net_y2h(batch_target_labels))
            if loss_type == "vanilla":
                dis_out = torch.nn.Sigmoid()(dis_out)
                g_loss = - torch.mean(torch.log(dis_out+1e-20))
            elif loss_type == "hinge":
                g_loss = - dis_out.mean()

            g_loss = g_loss / float(num_grad_acc_g)

            # backward
            # g_loss.backward()
            accelerator.backward(g_loss)

        ##end for accumulation_index

        optimizerG.step()

        # print loss
        if (niter+1) % 20 == 0:
            print ("CcGAN,%s: [Iter %d/%d] [D loss: %.4e] [G loss: %.4e] [real prob: %.3f] [fake prob: %.3f] [Time: %.4f]" % (gan_arch, niter+1, niters, d_loss.item(), g_loss.item(), real_dis_out.mean().item(), fake_dis_out.mean().item(), timeit.default_timer()-start_time))

        if (niter+1) % visualize_freq == 0:
            netG.eval()
            with torch.no_grad():
                gen_imgs = netG(z_fixed, net_y2h(y_fixed))
                gen_imgs = gen_imgs.detach().cpu()
                save_image(gen_imgs.data, save_images_folder + '/{}.png'.format(niter+1), nrow=n_row, normalize=True)

        if save_models_folder is not None and ((niter+1) % save_niters_freq == 0 or (niter+1) == niters):
            save_file = save_models_folder + "/ckpts_in_train/ckpt_niters_{}.pth".format(niter+1)
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            torch.save({
                    'netG_state_dict': netG.state_dict(),
                    'netD_state_dict': netD.state_dict(),
                    'optimizerG_state_dict': optimizerG.state_dict(),
                    'optimizerD_state_dict': optimizerD.state_dict(),
                    'rng_state': torch.get_rng_state()
            }, save_file)
    #end for niter
    return netG, netD


def sample_ccgan_given_labels(netG, net_y2h, labels, batch_size = 500, to_numpy=True, denorm=True, verbose=True, device="cuda"):
    '''
    netG: pretrained generator network
    labels: float. normalized labels.
    '''

    nfake = len(labels)
    if batch_size>nfake:
        batch_size=nfake

    fake_images = []
    fake_labels = np.concatenate((labels, labels[0:batch_size]))
    netG=netG.to(device)
    netG.eval()
    net_y2h = net_y2h.to(device)
    net_y2h.eval()
    with torch.no_grad():
        if verbose:
            pb = SimpleProgressBar()
        n_img_got = 0
        while n_img_got < nfake:
            z = torch.randn(batch_size, dim_gan, dtype=torch.float).to(device)
            y = torch.from_numpy(fake_labels[n_img_got:(n_img_got+batch_size)]).type(torch.float).view(-1,1).to(device)
            batch_fake_images = netG(z, net_y2h(y))
            if denorm: #denorm imgs to save memory
                assert batch_fake_images.max().item()<=1.0 and batch_fake_images.min().item()>=-1.0
                batch_fake_images = batch_fake_images*0.5+0.5
                batch_fake_images = batch_fake_images*255.0
                batch_fake_images = batch_fake_images.type(torch.uint8)
                # assert batch_fake_images.max().item()>1
            fake_images.append(batch_fake_images.cpu())
            n_img_got += batch_size
            if verbose:
                pb.update(min(float(n_img_got)/nfake, 1)*100)
        ##end while

    fake_images = torch.cat(fake_images, dim=0)
    #remove extra entries
    fake_images = fake_images[0:nfake]
    fake_labels = fake_labels[0:nfake]

    if to_numpy:
        fake_images = fake_images.numpy()

    return fake_images, fake_labels
