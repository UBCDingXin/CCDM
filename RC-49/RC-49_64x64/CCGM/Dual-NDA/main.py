print("\n===================================================================================================")

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

if args.comp_FID:
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

# normalize labels_train_raw
print("\n Range of unnormalized labels: ({},{})".format(np.min(labels_train_raw), np.max(labels_train_raw)))

labels_train =  fn_norm_labels(labels_train_raw)

print("\n Range of normalized labels: ({},{})".format(np.min(labels_train), np.max(labels_train)))

unique_labels_norm = np.sort(np.array(list(set(labels_train))))

if args.kernel_sigma<0:
    std_label = np.std(labels_train)
    args.kernel_sigma = 1.06*std_label*(len(labels_train))**(-1/5)

    print("\n Use rule-of-thumb formula to compute kernel_sigma >>>")
    print("\n The std of {} labels is {} so the kernel sigma is {}".format(len(labels_train), std_label, args.kernel_sigma))
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



#######################################################################################
'''                                Output folders                                  '''
#######################################################################################
path_to_embed_models = os.path.join(wd, 'output/embed_models')
os.makedirs(path_to_embed_models, exist_ok=True)

# path_to_output = os.path.join(wd, 'output/CcGAN_{}_{}_si{:.3f}_ka{:.3f}_{}_nDs{}_nDa{}_nGa{}_Dbs{}_Gbs{}_NDAa{:.2f}b{:.2f}c{:.2f}d{:.2f}'.format(args.GAN_arch, args.threshold_type, args.kernel_sigma, args.kappa, args.loss_type_gan, args.num_D_steps, args.num_grad_acc_d, args.num_grad_acc_g, args.batch_size_disc, args.batch_size_gene, args.nda_a, args.nda_b, args.nda_c, args.nda_d))
path_to_output = os.path.join(wd, 'output/CcGAN_NDA')
os.makedirs(path_to_output, exist_ok=True)

save_setting_folder = os.path.join(path_to_output, "exp_{}".format(args.setting_name))
os.makedirs(save_setting_folder, exist_ok=True)

setting_log_file = os.path.join(save_setting_folder, 'setting_info.txt')
if not os.path.isfile(setting_log_file):
    logging_file = open(setting_log_file, "w")
    logging_file.close()
with open(setting_log_file, 'a') as logging_file:
    logging_file.write("\n===================================================================================================")
    print(args, file=logging_file)
    logging_file.write("\r NDA para, a={}, b={}, c={}, d={}, e={}, nda_start_iter={}.".format(args.nda_a, args.nda_b, args.nda_c, args.nda_d, args.nda_e, args.nda_start_iter))
    logging_file.write("\r Finetune={}, GAN_ckpt={}.".format(args.GAN_finetune, args.path_GAN_ckpt))
    logging_file.write("\r niters={}, lr_g={}, lr_d={}.".format(args.niters_gan, args.lr_g_gan, args.lr_d_gan))
    logging_file.write("\r {}, sigma={}, kappa={}.".format(args.threshold_type, args.kernel_sigma, args.kappa))
    logging_file.write("\r nDs={}, nDa={}, nGa={}, Dbs={}, Gbs={}.".format(args.num_D_steps, args.num_grad_acc_d, args.num_grad_acc_g, args.batch_size_disc, args.batch_size_gene))

save_models_folder = os.path.join(save_setting_folder, 'saved_models')
os.makedirs(save_models_folder, exist_ok=True)
save_images_folder = os.path.join(save_setting_folder, 'saved_images')
os.makedirs(save_images_folder, exist_ok=True)



#######################################################################################
'''               Pre-trained CNN and GAN for label embedding                       '''
#######################################################################################
net_embed_filename_ckpt = os.path.join(path_to_embed_models, 'ckpt_{}_epoch_{}_seed_2020.pth'.format(args.net_embed, args.epoch_cnn_embed))
net_y2h_filename_ckpt = os.path.join(path_to_embed_models, 'ckpt_net_y2h_epoch_{}_seed_2020.pth'.format(args.epoch_net_y2h))

print("\n "+net_embed_filename_ckpt)
print("\n "+net_y2h_filename_ckpt)

trainset = IMGs_dataset(images_train, labels_train, normalize=True)
trainloader_embed_net = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_embed, shuffle=True)

if args.net_embed == "ResNet18_embed":
    net_embed = ResNet18_embed(dim_embed=args.dim_embed)
elif args.net_embed == "ResNet34_embed":
    net_embed = ResNet34_embed(dim_embed=args.dim_embed)
elif args.net_embed == "ResNet50_embed":
    net_embed = ResNet50_embed(dim_embed=args.dim_embed)
net_embed = net_embed.cuda()
# net_embed = nn.DataParallel(net_embed)

net_y2h = model_y2h(dim_embed=args.dim_embed)
net_y2h = net_y2h.cuda()
# net_y2h = nn.DataParallel(net_y2h)

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
indx_tmp = np.arange(len(unique_labels_norm))
np.random.shuffle(indx_tmp)
indx_tmp = indx_tmp[:10]
labels_tmp = unique_labels_norm[indx_tmp].reshape(-1,1)
labels_tmp = torch.from_numpy(labels_tmp).type(torch.float).cuda()
epsilons_tmp = np.random.normal(0, 0.2, len(labels_tmp))
epsilons_tmp = torch.from_numpy(epsilons_tmp).view(-1,1).type(torch.float).cuda()
labels_tmp = torch.clamp(labels_tmp+epsilons_tmp, 0.0, 1.0)
net_embed.eval()
net_h2y = net_embed.h2y
net_y2h.eval()
with torch.no_grad():
    labels_rec_tmp = net_h2y(net_y2h(labels_tmp)).cpu().numpy().reshape(-1,1)
results = np.concatenate((labels_tmp.cpu().numpy(), labels_rec_tmp), axis=1)
print("\n labels vs reconstructed labels")
print(results)



#######################################################################################
'''                                    GAN training                                 '''
#######################################################################################
print("CcGAN: {}, {}, Sigma is {:.4f}, Kappa is {:.4f}.".format(args.GAN_arch, args.threshold_type, args.kernel_sigma, args.kappa))
save_images_in_train_folder = save_images_folder + '/images_in_train'
os.makedirs(save_images_in_train_folder, exist_ok=True)


start = timeit.default_timer()
print("\n Begin Training >>>")
#----------------------------------------------
if args.GAN_finetune:
    ckpt_gan_path = save_models_folder + '/ckpt_niter{}_ft.pth'.format(args.niters_gan)
else:
    ckpt_gan_path = save_models_folder + '/ckpt_niter{}.pth'.format(args.niters_gan)
print(ckpt_gan_path)
if not os.path.isfile(ckpt_gan_path):
    if args.GAN_arch=="SNGAN":
        netG = CcGAN_SNGAN_Generator(dim_z=args.dim_gan, dim_embed=args.dim_embed)
        netD = CcGAN_SNGAN_Discriminator(dim_embed=args.dim_embed)
    elif args.GAN_arch=="SAGAN":
        netG = CcGAN_SAGAN_Generator(dim_z=args.dim_gan, dim_embed=args.dim_embed)
        netD = CcGAN_SAGAN_Discriminator(dim_embed=args.dim_embed)
    else:
        raise Exception("Not supported architecture...")
    netG = nn.DataParallel(netG)
    netD = nn.DataParallel(netD)

    # Start training
    netG, netD = train_ccgan(args.kernel_sigma, args.kappa, images_train, labels_train, netG, netD, net_y2h, save_images_folder=save_images_in_train_folder, save_models_folder = save_models_folder)

    # store model
    torch.save({
        'netG_state_dict': netG.state_dict(),
    }, ckpt_gan_path)

else:
    print("Loading pre-trained generator >>>")
    checkpoint = torch.load(ckpt_gan_path)
    if args.GAN_arch=="SNGAN":
        netG = CcGAN_SNGAN_Generator(dim_z=args.dim_gan, dim_embed=args.dim_embed)
    elif args.GAN_arch=="SAGAN":
        netG = CcGAN_SAGAN_Generator(dim_z=args.dim_gan, dim_embed=args.dim_embed)
    else:
        raise Exception("Not supported architecture...")
    netG = nn.DataParallel(netG)
    netG.load_state_dict(checkpoint['netG_state_dict'])

def fn_sampleGAN_given_labels(labels, batch_size, to_numpy=True, denorm=True, verbose=False):
    ## labels:normalized labels
    fake_images, fake_labels = sample_ccgan_given_labels(netG, net_y2h, labels, batch_size = batch_size, to_numpy=to_numpy, denorm=denorm, verbose=verbose)
    return fake_images, fake_labels

stop = timeit.default_timer()
print("GAN training finished; Time elapses: {}s".format(stop - start))





#######################################################################################
'''                                  Evaluation                                     '''
#######################################################################################
if args.comp_FID:
    print("\n Evaluation in Mode {}...".format(args.eval_mode))

    # for FID
    PreNetFID = encoder(dim_bottleneck=512).cuda()
    PreNetFID = nn.DataParallel(PreNetFID)
    Filename_PreCNNForEvalGANs = args.eval_ckpt_path + '/ckpt_AE_epoch_200_seed_2020_CVMode_False.pth'
    checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs)
    PreNetFID.load_state_dict(checkpoint_PreNet['net_encoder_state_dict'])

    # Diversity: entropy of predicted races within each eval center
    PreNetDiversity = ResNet34_class_eval(num_classes=49, ngpu = torch.cuda.device_count()).cuda() # give scenes
    Filename_PreCNNForEvalGANs_Diversity = args.eval_ckpt_path + '/ckpt_PreCNNForEvalGANs_ResNet34_class_epoch_200_seed_2020_classify_49_chair_types_CVMode_False.pth'
    checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs_Diversity)
    PreNetDiversity.load_state_dict(checkpoint_PreNet['net_state_dict'])

    # for LS
    PreNetLS = ResNet34_regre_eval(ngpu = torch.cuda.device_count()).cuda()
    Filename_PreCNNForEvalGANs_LS = args.eval_ckpt_path + '/ckpt_PreCNNForEvalGANs_ResNet34_regre_epoch_200_seed_2020_CVMode_False.pth'
    checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs_LS)
    PreNetLS.load_state_dict(checkpoint_PreNet['net_state_dict'])

    ## dump fake images for visualization
    sel_labels = np.array([0.1, 45, 89])
    n_per_label = 6
    for i in range(len(sel_labels)):
        curr_label = sel_labels[i]
        if i == 0:
            fake_labels_assigned = np.ones(n_per_label)*curr_label
        else:
            fake_labels_assigned = np.concatenate((fake_labels_assigned, np.ones(n_per_label)*curr_label))
    images_show, _ = fn_sampleGAN_given_labels(fn_norm_labels(fake_labels_assigned), batch_size=10, to_numpy=False, denorm=False, verbose=True)
    filename_images_show = save_images_folder + '/visualization_images_grid.png'
    save_image(images_show.data, filename_images_show, nrow=n_per_label, normalize=True)
    sys.exit()

    #####################
    # generate nfake images
    print("\r Start sampling {} fake images per label from GAN >>>".format(args.nfake_per_label))
    
    if args.eval_mode == 1: #Mode 1: eval on unique labels used for GAN training
        eval_labels = np.sort(np.array(list(set(labels_train_raw)))) #not normalized
    elif args.eval_mode in [2, 3]: #Mode 2 and 3: eval on all unique labels in the dataset
        eval_labels = np.sort(np.array(list(set(labels_all)))) #not normalized
    else: #Mode 4: eval on a interval [min_label, max_label] with num_eval_labels labels
        eval_labels = np.linspace(np.min(labels_all), np.max(labels_all), args.num_eval_labels) #not normalized

    unique_eval_labels = np.sort(np.array(list(set(eval_labels))))
    print("\r There are {} unique eval labels.".format(len(unique_eval_labels)))

    for i in range(len(eval_labels)):
        curr_label = eval_labels[i]
        if i == 0:
            fake_labels_assigned = np.ones(args.nfake_per_label)*curr_label
        else:
            fake_labels_assigned = np.concatenate((fake_labels_assigned, np.ones(args.nfake_per_label)*curr_label))
    fake_images, _ = fn_sampleGAN_given_labels(fn_norm_labels(fake_labels_assigned), args.samp_batch_size)
    assert len(fake_images) == args.nfake_per_label*len(eval_labels)
    assert len(fake_labels_assigned) == args.nfake_per_label*len(eval_labels)

    print("\r End sampling! We got {} fake images.".format(len(fake_images)))

    ## dump fake images for computing NIQE
    if args.dump_fake_for_NIQE:
        print("\n Dumping fake images for NIQE...")
        if args.niqe_dump_path=="None":
            dump_fake_images_folder = save_setting_folder + '/fake_images'
        else:
            dump_fake_images_folder = args.niqe_dump_path + '/fake_images'
        # dump_fake_images_folder = save_images_folder + '/fake_images_for_NIQE_nfake_{}'.format(len(fake_images))
        os.makedirs(dump_fake_images_folder, exist_ok=True)
        for i in tqdm(range(len(fake_images))):
            # label_i = fake_labels_assigned[i]*max_label_after_shift-np.abs(min_label_before_shift)
            # label_i = fn_denorm_labels(fake_labels_assigned[i])
            label_i = fake_labels_assigned[i]
            filename_i = dump_fake_images_folder + "/{}_{}.png".format(i, label_i)
            os.makedirs(os.path.dirname(filename_i), exist_ok=True)
            image_i = fake_images[i].astype(np.uint8)
            # image_i = ((image_i*0.5+0.5)*255.0).astype(np.uint8)
            image_i_pil = Image.fromarray(image_i.transpose(1,2,0))
            image_i_pil.save(filename_i)
        #end for i
        sys.exit()


    #####################
    # prepare real/fake images and labels
    if args.eval_mode in [1, 3]:
        real_images = images_train #not normalized
        real_labels = labels_train_raw #not normalized
    else: #for both mode 2 and 4
        real_images = images_all #not normalized
        real_labels = labels_all #not normalized
    # assert real_images.max()>1.0 and fake_images.max()>1.0
    # assert real_labels.max()>1.0 and fake_labels_assigned.max()>1.0
    
    #######################
    # For each label take nreal_per_label images
    unique_labels_real = np.sort(np.array(list(set(real_labels))))
    indx_subset = []
    for i in range(len(unique_labels_real)):
        label_i = unique_labels_real[i]
        indx_i = np.where(real_labels==label_i)[0]
        np.random.shuffle(indx_i)
        if args.nreal_per_label>1:
            indx_i = indx_i[0:args.nreal_per_label]
        indx_subset.append(indx_i)
    indx_subset = np.concatenate(indx_subset)
    real_images = real_images[indx_subset]
    real_labels = real_labels[indx_subset]

    nfake_all = len(fake_images)
    nreal_all = len(real_images)
    
    if args.comp_IS_and_FID_only:
        #####################
        # FID: Evaluate FID on all fake images
        indx_shuffle_real = np.arange(nreal_all); np.random.shuffle(indx_shuffle_real)
        indx_shuffle_fake = np.arange(nfake_all); np.random.shuffle(indx_shuffle_fake)
        FID = cal_FID(PreNetFID, real_images[indx_shuffle_real], fake_images[indx_shuffle_fake], batch_size = 500, resize = None, norm_img = True)
        print("\n FID of {} fake images: {}.".format(nfake_all, FID))

        #####################
        # IS: Evaluate IS on all fake images
        IS, IS_std = inception_score(imgs=fake_images[indx_shuffle_fake], num_classes=49, net=PreNetDiversity, cuda=True, batch_size=500, splits=10, normalize_img = True)
        print("\n IS of {} fake images: {}({}).".format(nfake_all, IS, IS_std))
    
    else:
        #####################
        # Evaluate FID within a sliding window with a radius R on the label's range (not normalized range, i.e., [min_label,max_label]). The center of the sliding window locate on [min_label+R,...,max_label-R].
        if args.eval_mode == 1:
            center_start = np.min(labels_train_raw)+args.FID_radius ##bug???
            center_stop = np.max(labels_train_raw)-args.FID_radius
        else:
            center_start = np.min(labels_all)+args.FID_radius
            center_stop = np.max(labels_all)-args.FID_radius

        if args.FID_num_centers<=0 and args.FID_radius==0: #completely overlap
            centers_loc = eval_labels #not normalized
        elif args.FID_num_centers>0:
            centers_loc = np.linspace(center_start, center_stop, args.FID_num_centers) #not normalized
        else:
            print("\n Error.")

        FID_over_centers = np.zeros(len(centers_loc))
        entropies_over_centers = np.zeros(len(centers_loc)) # entropy at each center
        labelscores_over_centers = np.zeros(len(centers_loc)) #label score at each center
        num_realimgs_over_centers = np.zeros(len(centers_loc))
        for i in range(len(centers_loc)):
            center = centers_loc[i]
            interval_start = center - args.FID_radius
            interval_stop  = center + args.FID_radius
            indx_real = np.where((real_labels>=interval_start)*(real_labels<=interval_stop)==True)[0]
            assert len(indx_real)>0
            np.random.shuffle(indx_real)
            real_images_curr = real_images[indx_real]
            real_images_curr = (real_images_curr/255.0-0.5)/0.5
            num_realimgs_over_centers[i] = len(real_images_curr)
            indx_fake = np.where((fake_labels_assigned>=interval_start)*(fake_labels_assigned<=interval_stop)==True)[0]
            assert len(indx_fake)>0
            np.random.shuffle(indx_fake)
            fake_images_curr = fake_images[indx_fake]
            fake_images_curr = (fake_images_curr/255.0-0.5)/0.5
            fake_labels_assigned_curr = fake_labels_assigned[indx_fake]
            ## FID
            FID_over_centers[i] = cal_FID(PreNetFID, real_images_curr, fake_images_curr, batch_size=500, resize = None)
            ## Entropy of predicted class labels
            predicted_class_labels = predict_class_labels(PreNetDiversity, fake_images_curr, batch_size=500, num_workers=args.num_workers)
            entropies_over_centers[i] = compute_entropy(predicted_class_labels)
            ## Label score
            labelscores_over_centers[i], _ = cal_labelscore(PreNetLS, fake_images_curr, fn_norm_labels(fake_labels_assigned_curr), min_label_before_shift=0, max_label_after_shift=args.max_label, batch_size=500, resize = None, num_workers=args.num_workers)
            ## print
            print("\r [{}/{}] Center:{}; Real:{}; Fake:{}; FID:{}; LS:{}; ET:{}. \n".format(i+1, len(centers_loc), center, len(real_images_curr), len(fake_images_curr), FID_over_centers[i], labelscores_over_centers[i], entropies_over_centers[i]))
        # end for i
        # average over all centers
        print("\n SFID: {}({}); min/max: {}/{}.".format(np.mean(FID_over_centers), np.std(FID_over_centers), np.min(FID_over_centers), np.max(FID_over_centers)))
        print("\r LS over centers: {}({}); min/max: {}/{}.".format(np.mean(labelscores_over_centers), np.std(labelscores_over_centers), np.min(labelscores_over_centers), np.max(labelscores_over_centers)))
        print("\r Entropy over centers: {}({}); min/max: {}/{}.".format(np.mean(entropies_over_centers), np.std(entropies_over_centers), np.min(entropies_over_centers), np.max(entropies_over_centers)))

        # dump FID versus number of samples (for each center) to npy
        dump_fid_ls_entropy_over_centers_filename = os.path.join(save_setting_folder, 'fid_ls_entropy_over_centers')
        np.savez(dump_fid_ls_entropy_over_centers_filename, fids=FID_over_centers, labelscores=labelscores_over_centers, entropies=entropies_over_centers, nrealimgs=num_realimgs_over_centers, centers=centers_loc)

        #####################
        # FID: Evaluate FID on all fake images
        indx_shuffle_real = np.arange(nreal_all); np.random.shuffle(indx_shuffle_real)
        indx_shuffle_fake = np.arange(nfake_all); np.random.shuffle(indx_shuffle_fake)
        FID = cal_FID(PreNetFID, real_images[indx_shuffle_real], fake_images[indx_shuffle_fake], batch_size=200, resize = None, norm_img = True)
        print("\n {}: FID of {} fake images: {}.".format(args.GAN_arch, nfake_all, FID))

        #####################
        # Overall LS: abs(y_assigned - y_predicted)
        ls_mean_overall, ls_std_overall = cal_labelscore(PreNetLS, fake_images, fn_norm_labels(fake_labels_assigned), min_label_before_shift=0, max_label_after_shift=args.max_label, batch_size=200, resize = None, norm_img = True, num_workers=args.num_workers)
        print("\n {}: overall LS of {} fake images: {}({}).".format(args.GAN_arch, nfake_all, ls_mean_overall, ls_std_overall))


        #####################
        # Dump evaluation results
        eval_results_logging_fullpath = os.path.join(save_setting_folder, 'eval_results_{}.txt'.format(args.GAN_arch))
        if not os.path.isfile(eval_results_logging_fullpath):
            eval_results_logging_file = open(eval_results_logging_fullpath, "w")
            eval_results_logging_file.close()
        with open(eval_results_logging_fullpath, 'a') as eval_results_logging_file:
            eval_results_logging_file.write("\n===================================================================================================")
            eval_results_logging_file.write("\n Radius: {}; # Centers: {}.  \n".format(args.FID_radius, args.FID_num_centers))
            print(args, file=eval_results_logging_file)
            eval_results_logging_file.write("\n SFID: {:.3f} ({:.3f}).".format(np.mean(FID_over_centers), np.std(FID_over_centers)))
            # eval_results_logging_file.write("\n LS: {:.3f} ({:.3f}).".format(np.mean(labelscores_over_centers), np.std(labelscores_over_centers)))
            eval_results_logging_file.write("\n LS: {:.3f} ({:.3f}).".format(ls_mean_overall, ls_std_overall))
            eval_results_logging_file.write("\n Diversity: {:.3f} ({:.3f}).".format(np.mean(entropies_over_centers), np.std(entropies_over_centers)))
            eval_results_logging_file.write("\n FID: {:.3f}.".format(FID))


print("\n===================================================================================================")
