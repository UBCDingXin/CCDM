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
import uuid

### import my stuffs ###
from opts import parse_opts
args = parse_opts()
wd = args.root_path
os.chdir(wd)
from utils import *
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

NGPU = torch.cuda.device_count()

if args.torch_model_path!="None":
    os.environ['TORCH_HOME']=args.torch_model_path

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



#######################################################################################
'''                                    Data loader                                 '''
#######################################################################################
# data loader
data_filename = args.data_path + '/UTKFace_Wild_{}x{}.h5'.format(args.img_size, args.img_size)
hf = h5py.File(data_filename, 'r')
labels = hf['ages'][:]
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





#######################################################################################
'''                                Output folders                                  '''
#######################################################################################
path_to_output = os.path.join(wd, 'output/{}_{}_si{:.3f}_ka{:.3f}_{}_nDs{}_nDa{}_nGa{}_Dbs{}_Gbs{}'.format(args.GAN_arch, args.threshold_type, args.kernel_sigma, args.kappa, args.loss_type_gan, args.num_D_steps, args.num_grad_acc_d, args.num_grad_acc_g, args.batch_size_disc, args.batch_size_gene))
os.makedirs(path_to_output, exist_ok=True)
save_models_folder = os.path.join(path_to_output, 'saved_models')
os.makedirs(save_models_folder, exist_ok=True)
save_images_folder = os.path.join(path_to_output, 'saved_images')
os.makedirs(save_images_folder, exist_ok=True)

path_to_embed_models = os.path.join(wd, 'output/embed_models')
os.makedirs(path_to_embed_models, exist_ok=True)

# fake_data_folder = os.path.join(path_to_output, 'bad_fake_data')
# os.makedirs(fake_data_folder, exist_ok=True)
# if args.setting_name=="None":
#     uuid_str = uuid.uuid4().hex
#     setting_name = "exp_{}".format(uuid_str)
#     save_setting_folder = os.path.join(fake_data_folder, setting_name)
#     os.makedirs(save_setting_folder, exist_ok=True)
#     setting_log_file = os.path.join(save_setting_folder, 'setting_info.txt')
#     if not os.path.isfile(setting_log_file):
#         eval_results_logging_file = open(setting_log_file, "w")
#         eval_results_logging_file.close()
#     with open(setting_log_file, 'a') as setting_log_file:
#         print(args, file=setting_log_file)
# else:
#     save_setting_folder = os.path.join(fake_data_folder, "{}".format(args.setting_name))
#     # print(save_setting_folder)
#     # assert os.path.exists(save_setting_folder)
#     os.makedirs(save_setting_folder, exist_ok=True)

if args.niqe_filter:
    fake_data_folder = os.path.join(path_to_output, 'bad_fake_data')
    os.makedirs(fake_data_folder, exist_ok=True)
    save_setting_folder = os.path.join(fake_data_folder, "{}".format(args.setting_name))
    os.makedirs(save_setting_folder, exist_ok=True)


#######################################################################################
'''               Pre-trained CNN and GAN for label embedding                       '''
#######################################################################################
net_embed_filename_ckpt = os.path.join(path_to_embed_models, 'ckpt_{}_epoch_{}_seed_{}.pth'.format(args.net_embed, args.epoch_cnn_embed, args.seed))
net_y2h_filename_ckpt = os.path.join(path_to_embed_models, 'ckpt_net_y2h_epoch_{}_seed_{}.pth'.format(args.epoch_net_y2h, args.seed))

print("\n "+net_embed_filename_ckpt)
print("\n "+net_y2h_filename_ckpt)

trainset = IMGs_dataset(images, labels, normalize=True)
trainloader_embed_net = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_embed, shuffle=True, num_workers=args.num_workers)

if args.net_embed == "ResNet18_embed":
    net_embed = ResNet18_embed(dim_embed=args.dim_embed)
elif args.net_embed == "ResNet34_embed":
    net_embed = ResNet34_embed(dim_embed=args.dim_embed)
elif args.net_embed == "ResNet50_embed":
    net_embed = ResNet50_embed(dim_embed=args.dim_embed)
net_embed = net_embed.cuda()
net_embed = nn.DataParallel(net_embed)

net_y2h = model_y2h(dim_embed=args.dim_embed)
net_y2h = net_y2h.cuda()
net_y2h = nn.DataParallel(net_y2h)

## (1). Train net_embed first: x2h+h2y
if not os.path.isfile(net_embed_filename_ckpt):
    print("\n Start training CNN for label embedding >>>")
    # optimizer_net_embed = torch.optim.SGD(net_embed.parameters(), lr = base_lr_x2y, momentum= 0.9, weight_decay=1e-4)
    # net_embed = train_net_embed(trainloader_embed_net, None, net_embed, optimizer_net_embed, epochs=args.epoch_cnn_embed, base_lr=base_lr_x2y, save_models_folder = save_models_folder, resumeepoch = args.resumeepoch_cnn_embed)
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
    # optimizer_net_y2h = torch.optim.SGD(net_y2h.parameters(), lr = base_lr_y2h, momentum = 0.9, weight_decay=1e-4)
    # net_y2h = train_net_y2h(unique_labels_norm, net_y2h, net_embed, optimizer_net_y2h, epochs=args.epoch_net_y2h, base_lr=base_lr_y2h, batch_size=32)
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

labels_diff = (labels_tmp-labels_noise_tmp)**2
hidden_diff = np.mean((labels_hidden_tmp-labels_noise_hidden_tmp)**2, axis=1, keepdims=True)
results2 = np.concatenate((labels_diff, hidden_diff), axis=1)
print("\n labels diff vs hidden diff")
print(results2)

#put models on cpu
net_embed = net_embed.cpu()
net_h2y = net_h2y.cpu()
del net_embed, net_h2y; gc.collect()
net_y2h = net_y2h.cpu()




#######################################################################################
'''                                    GAN training                                 '''
#######################################################################################
print("CcGAN: {}, {}, Sigma is {:.4f}, Kappa is {:.4f}.".format(args.GAN_arch, args.threshold_type, args.kernel_sigma, args.kappa))
save_images_in_train_folder = save_images_folder + '/images_in_train'
os.makedirs(save_images_in_train_folder, exist_ok=True)


start = timeit.default_timer()
print("\n Begin Training >>>")
#----------------------------------------------
ckpt_gan_path = save_models_folder + '/ckpt_niter_{}.pth'.format(args.niters_gan)
print(ckpt_gan_path)
if not os.path.isfile(ckpt_gan_path):
    if args.GAN_arch == "SAGAN":
        netG = sagan_generator(dim_z=args.dim_gan, dim_embed=args.dim_embed)
        netD = sagan_discriminator(dim_embed=args.dim_embed)
    netG = nn.DataParallel(netG)
    netD = nn.DataParallel(netD)

    # Start training
    netG, netD = train_ccgan(args.kernel_sigma, args.kappa, images, labels, netG, netD, net_y2h, save_images_folder=save_images_in_train_folder, save_models_folder = save_models_folder)

    # store model
    torch.save({
        'netG_state_dict': netG.state_dict(),
        'netD_state_dict': netD.state_dict(),
    }, ckpt_gan_path)

else:
    print("Loading pre-trained generator >>>")
    checkpoint = torch.load(ckpt_gan_path)
    if args.GAN_arch == "SAGAN":
        netG = sagan_generator(dim_z=args.dim_gan, dim_embed=args.dim_embed).cuda()
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
    #for FID
    PreNetFID = encoder(dim_bottleneck=512).cuda()
    PreNetFID = nn.DataParallel(PreNetFID)
    Filename_PreCNNForEvalGANs = os.path.join(args.eval_ckpt_path, 'ckpt_AE_epoch_200_seed_2024_CVMode_False.pth')
    checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs)
    PreNetFID.load_state_dict(checkpoint_PreNet['net_encoder_state_dict'])

    # Diversity: entropy of predicted races within each eval center
    PreNetDiversity = ResNet34_class_eval(num_classes=5, ngpu = torch.cuda.device_count()).cuda() #5 races
    Filename_PreCNNForEvalGANs_Diversity = os.path.join(args.eval_ckpt_path, 'ckpt_PreCNNForEvalGANs_ResNet34_class_epoch_200_seed_2024_classify_5_races_CVMode_False.pth')
    checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs_Diversity)
    PreNetDiversity.load_state_dict(checkpoint_PreNet['net_state_dict'])

    # for LS
    PreNetLS = ResNet34_regre_eval(ngpu = torch.cuda.device_count()).cuda()
    Filename_PreCNNForEvalGANs_LS = os.path.join(args.eval_ckpt_path, 'ckpt_PreCNNForEvalGANs_ResNet34_regre_epoch_200_seed_2024_CVMode_False.pth')
    checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs_LS)
    PreNetLS.load_state_dict(checkpoint_PreNet['net_state_dict'])


    # ## dump fake images for visualization
    # # sel_labels = np.array([5, 30, 60])
    # # n_per_label = 6
    # sel_labels = np.array([1, 5, 15, 30, 45, 60])
    # n_per_label = 10
    # for i in range(len(sel_labels)):
    #     curr_label = sel_labels[i]
    #     if i == 0:
    #         fake_labels_assigned = np.ones(n_per_label)*curr_label
    #     else:
    #         fake_labels_assigned = np.concatenate((fake_labels_assigned, np.ones(n_per_label)*curr_label))
    # images_show, _ = fn_sampleGAN_given_labels(fn_norm_labels(fake_labels_assigned), batch_size=10, to_numpy=False, denorm=False, verbose=True)
    # filename_images_show = save_images_folder + '/visualization_images_grid.png'
    # save_image(images_show.data, filename_images_show, nrow=n_per_label, normalize=True)
    # sys.exit()


    #####################
    # generate nfake images
    print("Start sampling {} fake images per label from GAN >>>".format(args.nfake_per_label))

    ## normalized labels for evaluation
    eval_labels = np.arange(1, args.max_label+1)
    eval_labels_norm = fn_norm_labels(eval_labels)
    num_eval_labels = len(eval_labels_norm)

    for i in tqdm(range(num_eval_labels)):
        label_i = eval_labels_norm[i]
        curr_fake_images, curr_fake_labels = fn_sampleGAN_given_labels(label_i*np.ones([args.nfake_per_label,1]), args.samp_batch_size)
        if i == 0:
            fake_images = curr_fake_images
            fake_labels_assigned = curr_fake_labels.reshape(-1)
        else:
            fake_images = np.concatenate((fake_images, curr_fake_images), axis=0)
            fake_labels_assigned = np.concatenate((fake_labels_assigned, curr_fake_labels.reshape(-1)))
    assert len(fake_images) == args.nfake_per_label*num_eval_labels
    assert len(fake_labels_assigned) == args.nfake_per_label*num_eval_labels


    ## dump fake images for evaluation: NIQE
    if args.dump_fake_for_NIQE:
        print("\n Dumping fake images for NIQE...")
        if args.niqe_dump_path=="None":
            dump_fake_images_folder = save_images_folder + '/fake_images'
        else:
            dump_fake_images_folder = args.niqe_dump_path + '/fake_images'
        os.makedirs(dump_fake_images_folder, exist_ok=True)
        for i in tqdm(range(len(fake_images))):
            label_i = int(fake_labels_assigned[i]*args.max_label)
            filename_i = dump_fake_images_folder + "/{}_{}.png".format(i, label_i)
            os.makedirs(os.path.dirname(filename_i), exist_ok=True)
            image_i = fake_images[i]
            # image_i = ((image_i*0.5+0.5)*255.0).astype(np.uint8)
            image_i_pil = Image.fromarray(image_i.transpose(1,2,0))
            image_i_pil.save(filename_i)
        #end for i
        sys.exit()

    print("End sampling!")
    print("\n We got {} fake images.".format(len(fake_images)))


    #####################
    # normalize labels
    real_labels = raw_labels/args.max_label
    nfake_all = len(fake_images)
    nreal_all = len(raw_images)
    real_images = raw_images
    
    
    if args.comp_IS_and_FID_only:
        #####################
        # FID: Evaluate FID on all fake images
        indx_shuffle_real = np.arange(nreal_all); np.random.shuffle(indx_shuffle_real)
        indx_shuffle_fake = np.arange(nfake_all); np.random.shuffle(indx_shuffle_fake)
        FID = cal_FID(PreNetFID, real_images[indx_shuffle_real], fake_images[indx_shuffle_fake], batch_size = 500, resize = None, norm_img = True)
        print("\n {}: FID of {} fake images: {}.".format(args.GAN_arch, nfake_all, FID))

        #####################
        # IS: Evaluate IS on all fake images
        IS, IS_std = inception_score(imgs=fake_images[indx_shuffle_fake], num_classes=5, net=PreNetDiversity, cuda=True, batch_size=200, splits=10, normalize_img=True)
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
            indx_fake = np.where((fake_labels_assigned>=interval_start)*(fake_labels_assigned<=interval_stop)==True)[0]
            np.random.shuffle(indx_fake)
            fake_images_curr = fake_images[indx_fake]
            fake_images_curr = (fake_images_curr/255.0-0.5)/0.5
            fake_labels_assigned_curr = fake_labels_assigned[indx_fake]
            # FID
            FID_over_centers[i] = cal_FID(PreNetFID, real_images_curr, fake_images_curr, batch_size=200, resize = None)
            # Entropy of predicted class labels
            predicted_class_labels = predict_class_labels(PreNetDiversity, fake_images_curr, batch_size=200, num_workers=args.num_workers)
            entropies_over_centers[i] = compute_entropy(predicted_class_labels)
            # Label score
            labelscores_over_centers[i], _ = cal_labelscore(PreNetLS, fake_images_curr, fake_labels_assigned_curr, min_label_before_shift=0, max_label_after_shift=args.max_label, batch_size = 200, resize = None, num_workers=args.num_workers)

            print("\r Center:{}; Real:{}; Fake:{}; FID:{}; LS:{}; ET:{}.".format(center, len(real_images_curr), len(fake_images_curr), FID_over_centers[i], labelscores_over_centers[i], entropies_over_centers[i]))

        # average over all centers
        print("\n {} SFID: {}({}); min/max: {}/{}.".format(args.GAN_arch, np.mean(FID_over_centers), np.std(FID_over_centers), np.min(FID_over_centers), np.max(FID_over_centers)))
        print("\n {} LS over centers: {}({}); min/max: {}/{}.".format(args.GAN_arch, np.mean(labelscores_over_centers), np.std(labelscores_over_centers), np.min(labelscores_over_centers), np.max(labelscores_over_centers)))
        print("\n {} entropy over centers: {}({}); min/max: {}/{}.".format(args.GAN_arch, np.mean(entropies_over_centers), np.std(entropies_over_centers), np.min(entropies_over_centers), np.max(entropies_over_centers)))

        # dump FID versus number of samples (for each center) to npy
        dump_fid_ls_entropy_over_centers_filename = os.path.join(path_to_output, 'fid_ls_entropy_over_centers')
        np.savez(dump_fid_ls_entropy_over_centers_filename, fids=FID_over_centers, labelscores=labelscores_over_centers, entropies=entropies_over_centers, nrealimgs=num_realimgs_over_centers, centers=centers_loc)

        #####################
        # FID: Evaluate FID on all fake images
        indx_shuffle_real = np.arange(nreal_all); np.random.shuffle(indx_shuffle_real)
        indx_shuffle_fake = np.arange(nfake_all); np.random.shuffle(indx_shuffle_fake)
        FID = cal_FID(PreNetFID, real_images[indx_shuffle_real], fake_images[indx_shuffle_fake], batch_size=200, resize = None, norm_img = True)
        print("\n {}: FID of {} fake images: {}.".format(args.GAN_arch, nfake_all, FID))

        #####################
        # Overall LS: abs(y_assigned - y_predicted)
        ls_mean_overall, ls_std_overall = cal_labelscore(PreNetLS, fake_images, fake_labels_assigned, min_label_before_shift=0, max_label_after_shift=args.max_label, batch_size=200, resize = None, norm_img = True, num_workers=args.num_workers)
        print("\n {}: overall LS of {} fake images: {}({}).".format(args.GAN_arch, nfake_all, ls_mean_overall, ls_std_overall))

        #####################
        # Dump evaluation results
        eval_results_logging_fullpath = os.path.join(path_to_output, 'eval_results_{}.txt'.format(args.GAN_arch))
        if not os.path.isfile(eval_results_logging_fullpath):
            eval_results_logging_file = open(eval_results_logging_fullpath, "w")
            eval_results_logging_file.close()
        with open(eval_results_logging_fullpath, 'a') as eval_results_logging_file:
            eval_results_logging_file.write("\n===================================================================================================")
            eval_results_logging_file.write("\n Radius: {}.  \n".format(args.FID_radius))
            print(args, file=eval_results_logging_file)
            eval_results_logging_file.write("\n SFID: {} ({}).".format(np.mean(FID_over_centers), np.std(FID_over_centers)))
            eval_results_logging_file.write("\n LS: {} ({}).".format(ls_mean_overall, ls_std_overall))
            eval_results_logging_file.write("\n Diversity: {} ({}).".format(np.mean(entropies_over_centers), np.std(entropies_over_centers)))
            eval_results_logging_file.write("\n FID: {}.".format(FID))



#######################################################################################
'''                        Generate bad fake samples                              '''
#######################################################################################

target_labels = np.arange(1, args.max_label+1)
target_labels_norm = fn_norm_labels(target_labels)
assert target_labels_norm.min()>=0 and target_labels_norm.max()<=1
num_target_labels = len(target_labels_norm)


''' Generate bad fake images based on NIQE filtering '''
if args.niqe_filter:
    
    if args.niqe_dump_path=="None":
        dump_fake_images_folder = save_setting_folder + '/fake_images_for_NIQE'
    else:
        dump_fake_images_folder = args.niqe_dump_path + '/fake_images_for_NIQE'
    os.makedirs(dump_fake_images_folder, exist_ok=True)

    print("\n-------------------------------------------")
    print("\r Generate fake images for niqe filtering...")
    for i in trange(len(target_labels_norm)):
        fake_images_i, fake_labels_i = fn_sampleGAN_given_labels(target_labels_norm[i]*np.ones([args.niqe_nfake_per_label_burnin,1]), args.samp_batch_size, to_numpy=True, denorm=True, verbose=False)
        ### denormalize labels
        assert fake_labels_i.min()>=0 and fake_labels_i.max()<=1
        fake_labels_i = fn_denorm_labels(fake_labels_i)
        ### dump fake images
        for j in range(len(fake_images_i)):
            filename_j = dump_fake_images_folder + "/{}_{}.png".format(i*args.samp_batch_size+j, int(fake_labels_i[j]))
            os.makedirs(os.path.dirname(filename_j), exist_ok=True)
            image_j = fake_images_i[j]
            # image_j = ((image_j*0.5+0.5)*255.0).astype(np.uint8)
            image_j = np.uint8(image_j)
            image_j_pil = Image.fromarray(image_j.transpose(1,2,0))
            image_j_pil.save(filename_j)
        #end for i
    ##end for i

print("\n===================================================================================================")
