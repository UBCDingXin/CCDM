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

from models import Unet, ResNet34_class_eval, ResNet34_regre_eval, ResNet34_embed, model_y2h, encoder
from diffusion import GaussianDiffusion
from trainer import Trainer
from opts import parse_opts
from utils import get_parameter_number, SimpleProgressBar, IMGs_dataset, compute_entropy, predict_class_labels
from train_net_for_label_embed import train_net_embed, train_net_y2h
from eval_metrics import cal_FID, cal_labelscore, inception_score
from ema_pytorch import EMA

##############################################
''' Settings '''
args = parse_opts()

# seeds
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
cudnn.benchmark = False
np.random.seed(args.seed)

# Embedding
base_lr_x2y = 0.01
base_lr_y2h = 0.01

if args.torch_model_path!="None":
    os.environ['TORCH_HOME']=args.torch_model_path

#######################################################################################
'''                                Output folders                                  '''
#######################################################################################
path_to_output = os.path.join(args.root_path, 'output')
os.makedirs(path_to_output, exist_ok=True)

path_to_embed_models = os.path.join(path_to_output, 'embed_models')
os.makedirs(path_to_embed_models, exist_ok=True)

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

# load data from h5 file
data_filename = args.data_path + '/UTKFace_{}x{}.h5'.format(args.image_size, args.image_size)
hf = h5py.File(data_filename, 'r')
labels = hf['labels'][:]
labels = labels.astype(float)
images = hf['images'][:]
hf.close()

# subset of UTKFace
selected_labels = np.arange(args.min_label, args.max_label+1)
for i in range(len(selected_labels)):
    curr_label = selected_labels[i]
    index_curr_label = np.where(labels==curr_label)[0]
    if i == 0:
        images_subset = images[index_curr_label]
        labels_subset = labels[index_curr_label]
    else:
        images_subset = np.concatenate((images_subset, images[index_curr_label]), axis=0)
        labels_subset = np.concatenate((labels_subset, labels[index_curr_label]))
# for i
images = images_subset
labels = labels_subset
del images_subset, labels_subset; gc.collect()

# for evaluation
raw_images = copy.deepcopy(images)
raw_labels = copy.deepcopy(labels)


# for each label, take no more than args.max_num_img_per_label images
image_num_threshold = args.max_num_img_per_label
print("\n Original set has {} images; For each label, take no more than {} images>>>".format(len(images), image_num_threshold))
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
print("{} images left.".format(len(images)))


## replicate minority samples to alleviate the imbalance
max_num_img_per_label_after_replica = np.min([args.max_num_img_per_label_after_replica, args.max_num_img_per_label])
if max_num_img_per_label_after_replica>1:
    unique_labels_replica = np.sort(np.array(list(set(labels))))
    num_labels_replicated = 0
    print("Start replicating monority samples >>>")
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
    print("We replicate {} images and labels \n".format(len(images_replica)))
    del images_replica, labels_replica; gc.collect()

# normalize labels
print("\n Range of unnormalized labels: ({},{})".format(np.min(labels), np.max(labels)))
# labels /= args.max_label #normalize to [0,1]
labels = fn_norm_labels(labels)
print("\n Range of normalized labels: ({},{})".format(np.min(labels), np.max(labels)))
unique_labels_norm = np.sort(np.array(list(set(labels))))

# vicinal parameters
if args.kernel_sigma<0:
    std_label = np.std(labels)
    args.kernel_sigma =1.06*std_label*(len(labels))**(-1/5)
    print("\n Use rule-of-thumb formula to compute kernel_sigma >>>")
    print("\n The std of {} labels is {} so the kernel sigma is {}".format(len(labels), std_label, args.kernel_sigma))

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

# build training set
## for training label embedding network
# transform_embedding = transforms.Compose([
#             transforms.RandomHorizontalFlip(0.5),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]) #to [-1,1]
#             ])
# trainset_embedding = IMGs_dataset(images, labels, transform_embedding)
trainset_embedding = IMGs_dataset(images, labels, normalize=True)
trainloader_embed_net = torch.utils.data.DataLoader(trainset_embedding, batch_size=args.batch_size_embed, shuffle=True, num_workers=args.num_workers)



#######################################################################################
'''                              Embedding network                                  '''
#######################################################################################
net_embed_filename_ckpt = os.path.join(path_to_embed_models, 'ckpt_{}_epoch_{}_seed_2023.pth'.format(args.net_embed, args.epoch_cnn_embed))
net_y2h_filename_ckpt = os.path.join(path_to_embed_models, 'ckpt_net_y2h_epoch_{}_seed_2023.pth'.format(args.epoch_net_y2h))

print("\n "+net_embed_filename_ckpt)
print("\n "+net_y2h_filename_ckpt)

net_embed = ResNet34_embed(dim_embed=args.dim_embed)
net_embed = net_embed.cuda()
net_embed = nn.DataParallel(net_embed)

net_y2h = model_y2h(dim_embed=args.dim_embed)
net_y2h = net_y2h.cuda()
net_y2h = nn.DataParallel(net_y2h)

## (1). Train net_embed first: x2h+h2y
if not os.path.isfile(net_embed_filename_ckpt):
    print("\n Start training CNN for label embedding >>>")
    net_embed = train_net_embed(net=net_embed, net_name=args.net_embed, trainloader=trainloader_embed_net, testloader=None, epochs=args.epoch_cnn_embed, resume_epoch = args.resumeepoch_cnn_embed, lr_base=base_lr_x2y, lr_decay_factor=0.1, lr_decay_epochs=[80, 140], weight_decay=1e-4, path_to_ckpt = path_to_embed_models)
    # save model
    torch.save({
    'net_state_dict': net_embed.state_dict(),
    }, net_embed_filename_ckpt)
else:
    print("\n net_embed ckpt already exists")
    print("\n Loading...")
    checkpoint = torch.load(net_embed_filename_ckpt)
    net_embed.load_state_dict(checkpoint['net_state_dict'])
#end not os.path.isfile

## (2). Train y2h
#train a net which maps a label back to the embedding space
if not os.path.isfile(net_y2h_filename_ckpt):
    print("\n Start training net_y2h >>>")
    net_y2h = train_net_y2h(unique_labels_norm, net_y2h, net_embed, epochs=args.epoch_net_y2h, lr_base=base_lr_y2h, lr_decay_factor=0.1, lr_decay_epochs=[150, 250, 350], weight_decay=1e-4, batch_size=128)
    # save model
    torch.save({
    'net_state_dict': net_y2h.state_dict(),
    }, net_y2h_filename_ckpt)
else:
    print("\n net_y2h ckpt already exists")
    print("\n Loading...")
    checkpoint = torch.load(net_y2h_filename_ckpt)
    net_y2h.load_state_dict(checkpoint['net_state_dict'])
#end not os.path.isfile

##some simple test
unique_labels_norm_embed = np.sort(np.array(list(set(labels))))
indx_tmp = np.arange(len(unique_labels_norm_embed))
np.random.shuffle(indx_tmp)
indx_tmp = indx_tmp[:10]
labels_tmp = unique_labels_norm_embed[indx_tmp].reshape(-1,1)
labels_tmp = torch.from_numpy(labels_tmp).type(torch.float).cuda()
epsilons_tmp = np.random.normal(0, 0.2, len(labels_tmp))
epsilons_tmp = torch.from_numpy(epsilons_tmp).view(-1,1).type(torch.float).cuda()
labels_noise_tmp = torch.clamp(labels_tmp+epsilons_tmp, 0.0, 1.0)
net_embed.eval()
net_h2y = net_embed.module.h2y
# net_h2y = net_embed.h2y
net_y2h.eval()
with torch.no_grad():
    labels_hidden_tmp = net_y2h(labels_tmp)
    labels_noise_hidden_tmp = net_y2h(labels_noise_tmp)
    labels_rec_tmp = net_h2y(labels_hidden_tmp).cpu().numpy().reshape(-1,1)
    labels_noise_rec_tmp = net_h2y(labels_noise_hidden_tmp).cpu().numpy().reshape(-1,1)
    labels_hidden_tmp = labels_hidden_tmp.cpu().numpy()
    labels_noise_hidden_tmp = labels_noise_hidden_tmp.cpu().numpy()
labels_tmp = labels_tmp.cpu().numpy()
labels_noise_tmp = labels_noise_tmp.cpu().numpy()
results1 = np.concatenate((labels_tmp, labels_rec_tmp), axis=1)
print("\n labels vs reconstructed labels")
print(results1)

# labels_diff = (labels_tmp-labels_noise_tmp)**2
# hidden_diff = np.mean((labels_hidden_tmp-labels_noise_hidden_tmp)**2, axis=1, keepdims=True)
# results2 = np.concatenate((labels_diff, hidden_diff), axis=1)
# print("\n labels diff vs hidden diff")
# print(results2)

#put models on cpu
net_embed = net_embed.cpu()
net_h2y = net_h2y.cpu()
del net_embed, net_h2y; gc.collect()
net_y2h = net_y2h.cpu()




#######################################################################################
'''                             Diffusion  training                                 '''
#######################################################################################

attention_resolutions = (args.attention_resolutions).split("_")
attention_resolutions = [int(dim) for dim in attention_resolutions]
# print(attention_resolutions)
channel_mult = (args.channel_mult).split("_")
channel_mult = [int(dim) for dim in channel_mult]
# print(channel_mult)

## build unet
model = Unet(
        embed_input_dim=args.dim_embed,
        cond_drop_prob = args.cond_drop_prob, #default 0.5, 1.0 means no condition
        in_channels=args.num_channels,
        model_channels=args.model_channels,
        out_channels=None,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=attention_resolutions,
        dropout=0,
        channel_mult=channel_mult, 
        conv_resample=True,
        num_heads=args.num_heads,
        use_scale_shift_norm=True,
        learned_variance = False,
        num_groups=args.num_groups,
)
model = nn.DataParallel(model)
print('\r model size:', get_parameter_number(model))

## build diffusion process
diffusion = GaussianDiffusion(
    model,
    image_size = args.image_size,
    timesteps = args.train_timesteps,
    sampling_timesteps = args.train_timesteps,
    objective = args.pred_objective,
    beta_schedule = args.beta_schedule,
    ddim_sampling_eta = 1,
).cuda()

## for visualization
n_row=10; n_col = n_row
start_label = 0
end_label = 59
selected_labels = np.linspace(start_label, end_label, num=n_row)
y_visual = np.zeros(n_row*n_col)
for i in range(n_row):
    curr_label = selected_labels[i]
    for j in range(n_col):
        y_visual[i*n_col+j] = curr_label
y_visual = y_visual.astype(int)
y_visual = torch.from_numpy(y_visual).type(torch.float).view(-1).cuda()
print("\n Conditioning labels for visualization during training...")
print(y_visual.cpu().numpy())
y_visual = fn_norm_labels(y_visual)
print("\n")

## for training
vicinal_params = {
    "kernel_sigma": args.kernel_sigma,
    "kappa": args.kappa,
    "threshold_type": args.threshold_type,
    "nonzero_soft_weight_threshold": args.nonzero_soft_weight_threshold,
}

trainer = Trainer(
    diffusion_model=diffusion,
    train_images=images,
    train_labels=labels,
    vicinal_params=vicinal_params,
    train_batch_size = args.train_batch_size,
    gradient_accumulate_every = args.gradient_accumulate_every,
    train_lr = args.train_lr,
    train_num_steps = args.niters,
    ema_update_after_step = 100,
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
)
if args.resume_niter>0:
    trainer.load(args.resume_niter)
trainer.train(net_y2h=net_y2h)


# ## Generate intermediate noisy images
# path_to_save = os.path.join(save_results_folder, "intermediate_images.gif")
# given_label = 0
# _nfake=10
# given_labels = given_label * torch.ones(_nfake).type(torch.float).view(-1).cuda()
# trainer.generate_intermediate_gifs(path_to_save, given_labels, net_y2h, fps=100, sampler="ddim", cond_scale=6.0, sample_timesteps=250, ddim_eta = 0)


#######################################################################################
'''                                   Evaluation                                    '''
#######################################################################################


if args.comp_FID:

    #for FID
    PreNetFID = encoder(dim_bottleneck=512).cuda()
    PreNetFID = nn.DataParallel(PreNetFID)
    Filename_PreCNNForEvalGANs = os.path.join(args.eval_ckpt_path, 'ckpt_AE_epoch_200_seed_2021_CVMode_False.pth')
    checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs)
    PreNetFID.load_state_dict(checkpoint_PreNet['net_encoder_state_dict'])

    # Diversity: entropy of predicted races within each eval center
    PreNetDiversity = ResNet34_class_eval(num_classes=5, ngpu = torch.cuda.device_count()).cuda() #5 races
    Filename_PreCNNForEvalGANs_Diversity = os.path.join(args.eval_ckpt_path, 'ckpt_PreCNNForEvalGANs_ResNet34_class_epoch_200_seed_2021_classify_5_races_CVMode_False.pth')
    checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs_Diversity)
    PreNetDiversity.load_state_dict(checkpoint_PreNet['net_state_dict'])

    # for LS
    PreNetLS = ResNet34_regre_eval(ngpu = torch.cuda.device_count()).cuda()
    Filename_PreCNNForEvalGANs_LS = os.path.join(args.eval_ckpt_path, 'ckpt_PreCNNForEvalGANs_ResNet34_regre_epoch_200_seed_2021_CVMode_False.pth')
    checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs_LS)
    PreNetLS.load_state_dict(checkpoint_PreNet['net_state_dict'])

    #####################
    # generate nfake images
    print("Start sampling {} fake images per label >>>".format(args.nfake_per_label))

    eval_labels = np.arange(1, args.max_label+1) 
    num_eval_labels = len(eval_labels)
    print(eval_labels)

    ###########################################
    ''' multiple h5 files '''
    # print('\n Start generating fake data...')
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
            fake_images_i, _ = trainer.sample_given_labels(given_labels = fn_norm_labels(fake_labels_i), net_y2h=net_y2h, batch_size = args.samp_batch_size, denorm=True, to_numpy=True, verbose=False, sampler=args.sampler, cond_scale=args.sample_cond_scale, sample_timesteps=args.sample_timesteps, ddim_eta=args.ddim_eta)
            stop = timeit.default_timer()
            assert len(fake_images_i)==len(fake_labels_i)
            sample_time_i = stop-start
            if args.dump_fake_data:
                with h5py.File(dump_fake_images_filename, "w") as f:
                    f.create_dataset('fake_images_i', data = fake_images_i, dtype='uint8', compression="gzip", compression_opts=6)
                    f.create_dataset('fake_labels_i', data = fake_labels_i, dtype='int')
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
        
        ## dump 100 imgs for visualization
        img_vis_i = fake_images_i[0:100]/255.0
        img_vis_i = torch.from_numpy(img_vis_i)
        img_filename = os.path.join(dump_fake_images_folder, 'sample_{}.png'.format(curr_label))
        torchvision.utils.save_image(img_vis_i.data, img_filename, nrow=10, normalize=False)
        
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
            label_i = int(fake_labels[i])
            filename_i = dump_fake_images_folder + "/{}_{}.png".format(i, label_i)
            os.makedirs(os.path.dirname(filename_i), exist_ok=True)
            image_i = fake_images[i]
            image_i_pil = Image.fromarray(image_i.transpose(1,2,0))
            image_i_pil.save(filename_i)
        #end for i
        sys.exit()

    
    #####################
    # normalize labels
    real_labels = raw_labels/args.max_label
    fake_labels = fake_labels/args.max_label
    nfake_all = len(fake_images)
    nreal_all = len(raw_images)
    real_images = raw_images

    if args.comp_IS_and_FID_only:
        #####################
        # FID: Evaluate FID on all fake images
        indx_shuffle_real = np.arange(nreal_all); np.random.shuffle(indx_shuffle_real)
        indx_shuffle_fake = np.arange(nfake_all); np.random.shuffle(indx_shuffle_fake)
        FID = cal_FID(PreNetFID, real_images[indx_shuffle_real], fake_images[indx_shuffle_fake], batch_size=args.eval_batch_size, resize = None, norm_img = True)
        print("\n {}: FID of {} fake images: {}.".format(args.GAN_arch, nfake_all, FID))

        #####################
        # IS: Evaluate IS on all fake images
        IS, IS_std = inception_score(imgs=fake_images[indx_shuffle_fake], num_classes=5, net=PreNetDiversity, cuda=True, batch_size=args.eval_batch_size, splits=10, normalize_img=True)
        print("\n {}: IS of {} fake images: {}({}).".format(args.GAN_arch, nfake_all, IS, IS_std))

    else:
    
        #####################
        # Evaluate FID within a sliding window with a radius R on the label's range (i.e., [1,max_label]). The center of the sliding window locate on [R+1,2,3,...,max_label-R].
        center_start = 1+args.FID_radius
        center_stop = args.max_label-args.FID_radius
        centers_loc = np.arange(center_start, center_stop+1)
        FID_over_centers = np.zeros(len(centers_loc))
        entropies_over_centers = np.zeros(len(centers_loc)) # entropy at each center
        labelscores_over_centers = np.zeros(len(centers_loc)) #label score at each center
        num_realimgs_over_centers = np.zeros(len(centers_loc))
        for i in range(len(centers_loc)):
            center = centers_loc[i]
            interval_start = (center - args.FID_radius)/args.max_label
            interval_stop = (center + args.FID_radius)/args.max_label
            indx_real = np.where((real_labels>=interval_start)*(real_labels<=interval_stop)==True)[0]
            np.random.shuffle(indx_real)
            real_images_curr = real_images[indx_real]
            real_images_curr = (real_images_curr/255.0-0.5)/0.5
            num_realimgs_over_centers[i] = len(real_images_curr)
            indx_fake = np.where((fake_labels>=interval_start)*(fake_labels<=interval_stop)==True)[0]
            np.random.shuffle(indx_fake)
            fake_images_curr = fake_images[indx_fake]
            fake_images_curr = (fake_images_curr/255.0-0.5)/0.5
            fake_labels_assigned_curr = fake_labels[indx_fake]
            # FID
            FID_over_centers[i] = cal_FID(PreNetFID, real_images_curr, fake_images_curr, batch_size=args.eval_batch_size, resize = None)
            # Entropy of predicted class labels
            predicted_class_labels = predict_class_labels(PreNetDiversity, fake_images_curr, batch_size=args.eval_batch_size, num_workers=args.num_workers)
            entropies_over_centers[i] = compute_entropy(predicted_class_labels)
            # Label score
            labelscores_over_centers[i], _ = cal_labelscore(PreNetLS, fake_images_curr, fake_labels_assigned_curr, min_label_before_shift=0, max_label_after_shift=args.max_label, batch_size = args.eval_batch_size, resize = None, num_workers=args.num_workers)

            print("\r Center:{}; Real:{}; Fake:{}; FID:{}; LS:{}; ET:{}.".format(center, len(real_images_curr), len(fake_images_curr), FID_over_centers[i], labelscores_over_centers[i], entropies_over_centers[i]))

        # average over all centers
        print("\n SFID: {}({}); min/max: {}/{}.".format(np.mean(FID_over_centers), np.std(FID_over_centers), np.min(FID_over_centers), np.max(FID_over_centers)))
        print("\n LS over centers: {}({}); min/max: {}/{}.".format(np.mean(labelscores_over_centers), np.std(labelscores_over_centers), np.min(labelscores_over_centers), np.max(labelscores_over_centers)))
        print("\n Entropy over centers: {}({}); min/max: {}/{}.".format(np.mean(entropies_over_centers), np.std(entropies_over_centers), np.min(entropies_over_centers), np.max(entropies_over_centers)))

        # dump FID versus number of samples (for each center) to npy
        dump_fid_ls_entropy_over_centers_filename = os.path.join(save_setting_folder, 'fid_ls_entropy_over_centers_sampstep{}'.format(args.sample_timesteps))
        np.savez(dump_fid_ls_entropy_over_centers_filename, fids=FID_over_centers, labelscores=labelscores_over_centers, entropies=entropies_over_centers, nrealimgs=num_realimgs_over_centers, centers=centers_loc)


        #####################
        # FID: Evaluate FID on all fake images
        indx_shuffle_real = np.arange(nreal_all); np.random.shuffle(indx_shuffle_real)
        indx_shuffle_fake = np.arange(nfake_all); np.random.shuffle(indx_shuffle_fake)
        FID = cal_FID(PreNetFID, real_images[indx_shuffle_real], fake_images[indx_shuffle_fake], batch_size=args.eval_batch_size, resize = None, norm_img = True)
        print("\n FID of {} fake images: {}.".format(nfake_all, FID))

        #####################
        # Overall LS: abs(y_assigned - y_predicted)
        ls_mean_overall, ls_std_overall = cal_labelscore(PreNetLS, fake_images, fake_labels, min_label_before_shift=0, max_label_after_shift=args.max_label, batch_size=args.eval_batch_size, resize = None, norm_img = True, num_workers=args.num_workers)
        print("\n Overall LS of {} fake images: {}({}).".format(nfake_all, ls_mean_overall, ls_std_overall))

        #####################
        # Dump evaluation results
        eval_results_logging_fullpath = os.path.join(save_setting_folder, 'eval_results_niters{}.txt'.format(args.niters))
        if not os.path.isfile(eval_results_logging_fullpath):
            eval_results_logging_file = open(eval_results_logging_fullpath, "w")
            eval_results_logging_file.close()
        with open(eval_results_logging_fullpath, 'a') as eval_results_logging_file:
            eval_results_logging_file.write("\n===================================================================================================")
            print(args, file=eval_results_logging_file)
            eval_results_logging_file.write("\n Sampling Steps: {}.".format(args.sample_timesteps))
            eval_results_logging_file.write("\n Sampling Time: {:.3f}.".format(total_sample_time))
            eval_results_logging_file.write("\n SFID: {:.3f} ({:.3f}).".format(np.mean(FID_over_centers), np.std(FID_over_centers)))
            eval_results_logging_file.write("\n LS: {:.3f} ({:.3f}).".format(ls_mean_overall, ls_std_overall))
            eval_results_logging_file.write("\n Diversity: {:.3f} ({:.3f}).".format(np.mean(entropies_over_centers), np.std(entropies_over_centers)))
            eval_results_logging_file.write("\n FID: {:.3f}.".format(FID))





print("\n===================================================================================================")