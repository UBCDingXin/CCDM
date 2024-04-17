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

from torchvision import datasets, transforms
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize

### import my stuffs ###
from opts import parse_opts
args = parse_opts()
from utils import IMGs_dataset, SimpleProgressBar, compute_entropy, predict_class_labels
from classifier_free_guidance import Unet, GaussianDiffusion, Trainer
from eval_models import ResNet34_regre_eval, ResNet34_class_eval, encoder
from eval_metrics import cal_FID, cal_labelscore, inception_score

print("\n Available GPUs: {}.".format(torch.cuda.device_count()))

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



#######################################################################################
'''                                    Data loader                                 '''
#######################################################################################
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
    labels_train = labels_all[indx_train]
else:
    images_train = copy.deepcopy(images_all)
    labels_train = copy.deepcopy(labels_all)

# remove too small angles and too large angles
q1 = args.min_label
q2 = args.max_label
indx = np.where((labels_train>q1)*(labels_train<q2)==True)[0]
labels_train = labels_train[indx]
images_train = images_train[indx]
assert len(labels_train)==len(images_train)

indx = np.where((labels_all>q1)*(labels_all<q2)==True)[0]
labels_all = labels_all[indx]
images_all = images_all[indx]
assert len(labels_all)==len(images_all)


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

# for each angle, take no more than args.max_num_img_per_label images
image_num_threshold = args.max_num_img_per_label
print("\n Original set has {} images; For each angle, take no more than {} images>>>".format(len(images_train), image_num_threshold))
unique_labels_tmp = np.sort(np.array(list(set(labels_train))))
for i in tqdm(range(len(unique_labels_tmp))):
    indx_i = np.where(labels_train == unique_labels_tmp[i])[0]
    if len(indx_i)>image_num_threshold:
        np.random.shuffle(indx_i)
        indx_i = indx_i[0:image_num_threshold]
    if i == 0:
        sel_indx = indx_i
    else:
        sel_indx = np.concatenate((sel_indx, indx_i))
images_train = images_train[sel_indx]
labels_train = labels_train[sel_indx]
print("{} images left and there are {} unique labels".format(len(images_train), len(set(labels_train))))

## print number of images for each label
unique_labels_tmp = np.sort(np.array(list(set(labels_train))))
num_img_per_label_all = np.zeros(len(unique_labels_tmp))
for i in range(len(unique_labels_tmp)):
    indx_i = np.where(labels_train==unique_labels_tmp[i])[0]
    num_img_per_label_all[i] = len(indx_i)
# print(list(num_img_per_label_all))
# data_csv = np.concatenate((unique_labels_tmp.reshape(-1,1), num_img_per_label_all.reshape(-1,1)), 1)
# np.savetxt(args.root_path + '/label_dist.csv', data_csv, delimiter=',')

## replicate minority samples to alleviate the imbalance issue
max_num_img_per_label_after_replica = args.max_num_img_per_label_after_replica
if max_num_img_per_label_after_replica>1:
    unique_labels_replica = np.sort(np.array(list(set(labels_train))))
    num_labels_replicated = 0
    print("Start replicating minority samples >>>")
    for i in tqdm(range(len(unique_labels_replica))):
        curr_label = unique_labels_replica[i]
        indx_i = np.where(labels_train == curr_label)[0]
        if len(indx_i) < max_num_img_per_label_after_replica:
            num_img_less = max_num_img_per_label_after_replica - len(indx_i)
            indx_replica = np.random.choice(indx_i, size = num_img_less, replace=True)
            if num_labels_replicated == 0:
                images_replica = images_train[indx_replica]
                labels_replica = labels_train[indx_replica]
            else:
                images_replica = np.concatenate((images_replica, images_train[indx_replica]), axis=0)
                labels_replica = np.concatenate((labels_replica, labels_train[indx_replica]))
            num_labels_replicated+=1
    #end for i
    images_train = np.concatenate((images_train, images_replica), axis=0)
    labels_train = np.concatenate((labels_train, labels_replica))
    print("We replicate {} images and labels \n".format(len(images_replica)))
    del images_replica, labels_replica; gc.collect()

## convert steering angles to class labels and vice versa
unique_labels = np.sort(np.array(list(set(labels_train))))
num_unique_labels = len(unique_labels)
print("{} unique labels are split into {} classes".format(num_unique_labels, args.num_classes))

### step 1: prepare two dictionaries
label2class = dict()
class2label = dict()
num_labels_per_class = num_unique_labels//args.num_classes
class_cutoff_points = [unique_labels[0]] #the cutoff points on [min_label, max_label] to determine classes
curr_class = 0
for i in range(num_unique_labels):
    label2class[unique_labels[i]]=curr_class
    if (i+1)%num_labels_per_class==0 and (curr_class+1)!=args.num_classes:
        curr_class += 1
        class_cutoff_points.append(unique_labels[i+1])
class_cutoff_points.append(unique_labels[-1])
assert len(class_cutoff_points)-1 == args.num_classes

for i in range(args.num_classes):
    class2label[i] = (class_cutoff_points[i]+class_cutoff_points[i+1])/2

### step 2: convert angles to class labels
labels_new = -1*np.ones(len(labels_train))
for i in range(len(labels_train)):
    labels_new[i] = label2class[labels_train[i]]
assert np.sum(labels_new<0)==0
labels_train = labels_new
del labels_new; gc.collect()
unique_labels = np.sort(np.array(list(set(labels_train)))).astype(int)
assert len(unique_labels) == args.num_classes
print("\n Distinct labels are shown as follows:\r")
print(unique_labels)


#######################################################################################
'''                                Output folders                                  '''
#######################################################################################
path_to_output = os.path.join(args.root_path, 'output')
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
    # logging_file.write("\r niters={}, lr={}, lr_d={}.".format(args.niters_gan, args.lr_g_gan, args.lr_d_gan))

save_results_folder = os.path.join(save_setting_folder, 'results')
os.makedirs(save_results_folder, exist_ok=True)



#######################################################################################
'''                             Diffusion  training                                 '''
#######################################################################################
           
model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        num_classes = args.num_classes,
        cond_drop_prob = 0.5, #default 0.5
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
    )
model = nn.DataParallel(model)


diffusion = GaussianDiffusion(
    model,
    image_size = args.img_size,
    timesteps = args.timesteps,
    sampling_timesteps = args.sampling_timesteps,
    beta_schedule = 'cosine',
    ddim_sampling_eta = 1,
).cuda()


# transform = Compose([
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             # transforms.Lambda(lambda t: (t * 2) - 1)
#             ])
trainset = IMGs_dataset(images_train, labels_train, normalize=True)


## for visualization
n_row=5; n_col = n_row
start_label = 0
end_label = args.num_classes-1
selected_labels = np.linspace(start_label, end_label, num=n_row)
y_visual = np.zeros(n_row*n_col)
for i in range(n_row):
    curr_label = selected_labels[i]
    for j in range(n_col):
        y_visual[i*n_col+j] = curr_label
y_visual = y_visual.astype(int)
print("\n Visualization labels are shown as follows:\r")
print(y_visual)
print("\n")
y_visual = torch.from_numpy(y_visual).type(torch.long).view(-1).cuda()


## for training
trainer = Trainer(
    diffusion_model=diffusion,
    dataset=trainset,
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
    num_workers = args.num_workers,
    y_visual = y_visual,
)
if args.resume_niter>0:
    trainer.load(args.resume_niter)
trainer.train()


# sampled_images = diffusion.sample(
#     classes = y_visual,
#     cond_scale = 6.,
# )
# sampled_images = sampled_images.cpu().numpy()
# assert sampled_images.min()>=0 and sampled_images.max()<=1
# sampled_images = (sampled_images*255.0).astype(np.uint8)
# print(sampled_images.shape) # (8, 3, 192, 192)
# n = len(sampled_images)
# dump_path = os.path.join(save_results_folder, 'tmp')
# os.makedirs(dump_path, exist_ok=True)
# for i in trange(n):
#     image_i = sampled_images[i]
#     image_i = np.transpose(image_i, (1, 2, 0)) #now W * H * C
    
#     image_pil_i = Image.fromarray(np.uint8(image_i), mode = 'RGB')
#     filename_i = os.path.join(dump_path, "{}.png".format(i))
#     image_pil_i.save(filename_i,"PNG")


def sample_given_labels(given_labels, diffusion, class_cutoff_points=class_cutoff_points, batch_size = args.samp_batch_size, denorm=True, to_numpy=False, verbose=False):
    '''
    given_labels: a numpy array; raw label without any normalization; not class label
    class_cutoff_points: the cutoff points to determine the membership of a give label
    '''

    class_cutoff_points = np.array(class_cutoff_points)
    num_classes = len(class_cutoff_points)-1

    nfake = len(given_labels)
    given_class_labels = np.zeros(nfake)
    for i in range(nfake):
        curr_given_label = given_labels[i]
        diff_tmp = class_cutoff_points - curr_given_label
        indx_nonneg = np.where(diff_tmp>=0)[0]
        if len(indx_nonneg)==1: #the last element of diff_tmp is non-negative
            curr_given_class_label = num_classes-1
            assert indx_nonneg[0] == num_classes
        elif len(indx_nonneg)>1:
            if diff_tmp[indx_nonneg[0]]>0:
                curr_given_class_label = indx_nonneg[0] - 1
            else:
                curr_given_class_label = indx_nonneg[0]
        given_class_labels[i] = curr_given_class_label
    given_class_labels = np.concatenate((given_class_labels, given_class_labels[0:batch_size]))

    if batch_size>nfake:
        batch_size = nfake
    fake_images = []
    # if verbose:
    #     pb = SimpleProgressBar()
    tmp = 0
    while tmp < nfake:
        batch_fake_labels = torch.from_numpy(given_class_labels[tmp:(tmp+batch_size)]).type(torch.long).cuda()
        if batch_fake_labels.max().item()>num_classes:
            print("Error: max label {}".format(batch_fake_labels.max().item()))
        batch_fake_images = diffusion.sample(
                                classes = batch_fake_labels,
                                cond_scale = 6.,
                            )
        
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



if args.comp_FID:
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


    #####################
    # generate nfake images
    print("Start sampling {} fake images per label >>>".format(args.nfake_per_label))

    eval_labels = np.sort(np.array(list(set(labels_all)))) #not normalized
    num_eval_labels = len(eval_labels)
    print(eval_labels)

    # ###########################################
    # ''' one h5 file '''
    # fake_labels = []
    # for i in range(num_eval_labels):
    #     curr_label = eval_labels[i]
    #     fake_labels.append(curr_label*np.ones(args.nfake_per_label))
    # fake_labels = np.concatenate(fake_labels)
    # assert len(fake_labels) == int(args.nfake_per_label*num_eval_labels)

    # dump_fake_images_filename = os.path.join(save_results_folder, 'fake_data_niters{}_nfake{}.h5'.format(args.niters, len(fake_labels)))
    # if not os.path.isfile(dump_fake_images_filename):
    #     print('\n Start generating fake data...')
    #     start = timeit.default_timer()
    #     fake_images, _ = sample_given_labels(given_labels=fake_labels, diffusion=diffusion, class_cutoff_points=class_cutoff_points, batch_size = args.samp_batch_size, denorm=True, to_numpy=True, verbose=True)
    #     stop = timeit.default_timer()
    #     print("Sampling finished; Time elapses: {}s".format(stop - start))
    #     assert len(fake_images)==len(fake_labels)
    #     if args.dump_fake_data:
    #         with h5py.File(dump_fake_images_filename, "w") as f:
    #             f.create_dataset('fake_images', data = fake_images, dtype='uint8', compression="gzip", compression_opts=6)
    #             f.create_dataset('fake_labels', data = fake_labels, dtype='int')
    # else:
    #     print('\n Start loading generated fake data...')
    #     with h5py.File(dump_fake_images_filename, "r") as f:
    #         fake_images = f['fake_images'][:]
    #         fake_labels = f['fake_labels'][:]
    #     assert len(fake_images) == len(fake_labels)
    # ##end if
    # print("End sampling!")
    # print("\n We got {} fake images.".format(len(fake_images)))


    ###########################################
    ''' multiple h5 files '''
    print('\n Start generating fake data...')
    dump_fake_images_folder = os.path.join(save_results_folder, 'fake_data_niters{}_nfake{}_sampstep{}'.format(args.niters, int(args.nfake_per_label*num_eval_labels), args.sampling_timesteps))
    os.makedirs(dump_fake_images_folder, exist_ok=True)
    fake_images = []
    fake_labels = []
    total_sample_time = 0
    for i in range(num_eval_labels):
        curr_label = eval_labels[i]
        dump_fake_images_filename = os.path.join(dump_fake_images_folder, '{}.h5'.format(curr_label))
        if not os.path.isfile(dump_fake_images_filename):
            fake_labels_i = curr_label*np.ones(args.nfake_per_label)
            start = timeit.default_timer()
            fake_images_i, _ = sample_given_labels(given_labels=fake_labels_i, diffusion=diffusion, class_cutoff_points=class_cutoff_points, batch_size = args.samp_batch_size, denorm=True, to_numpy=True, verbose=False)
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
                # fake_labels_i = f['fake_labels_i'][:]
                fake_labels_i = float(curr_label)*np.ones(len(fake_images_i))
                sample_time_i = f['sample_time_i'][0]
            assert len(fake_images_i) == len(fake_labels_i)
        ##end if
        total_sample_time+=sample_time_i
        fake_images.append(fake_images_i)
        fake_labels.append(fake_labels_i)
        print("\r {}/{}: Got {} fake images for label {}. Time spent {:.2f}, Total time {:.2f}.".format(i+1, num_eval_labels, len(fake_images_i), curr_label, sample_time_i, total_sample_time))

        ## dump 100 imgs for visualization
        img_vis_i = fake_images_i[0:25]/255.0
        img_vis_i = torch.from_numpy(img_vis_i)
        img_filename = os.path.join(dump_fake_images_folder, 'sample_{}.png'.format(curr_label))
        torchvision.utils.save_image(img_vis_i.data, img_filename, nrow=5, normalize=False)

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
            filename_i = dump_fake_images_folder + "/{}_{}.png".format(i, label_i)
            os.makedirs(os.path.dirname(filename_i), exist_ok=True)
            image_i = fake_images[i]
            image_i_pil = Image.fromarray(image_i.transpose(1,2,0))
            image_i_pil.save(filename_i)
        #end for i
        sys.exit()


    real_labels = labels_all
    nfake_all = len(fake_images)
    nreal_all = len(images_all)
    real_images = images_all

    if args.comp_IS_and_FID_only:
        #####################
        # FID: Evaluate FID on all fake images
        indx_shuffle_real = np.arange(nreal_all); np.random.shuffle(indx_shuffle_real)
        indx_shuffle_fake = np.arange(nfake_all); np.random.shuffle(indx_shuffle_fake)
        FID = cal_FID(PreNetFID, real_images[indx_shuffle_real], fake_images[indx_shuffle_fake], batch_size=args.eval_batch_size, resize = None, norm_img = True)
        print("\n {}: FID of {} fake images: {}.".format(args.GAN_arch, nfake_all, FID))

        #####################
        # IS: Evaluate IS on all fake images
        IS, IS_std = inception_score(imgs=fake_images[indx_shuffle_fake], num_classes=49, net=PreNetDiversity, cuda=True, batch_size=args.eval_batch_size, splits=10, normalize_img=True)
        print("\n {}: IS of {} fake images: {}({}).".format(args.GAN_arch, nfake_all, IS, IS_std))

    else:
    
        #####################
        # Evaluate FID within a sliding window with a radius R on the label's range (i.e., [1,max_label]). The center of the sliding window locate on [R+1,2,3,...,max_label-R].
        # center_start = np.min(labels_all)+args.FID_radius
        # center_stop = np.max(labels_all)-args.FID_radius
        centers_loc = eval_labels #not normalized
        
        labelscores_over_centers = np.zeros(len(centers_loc)) #label score at each center
        FID_over_centers = np.zeros(len(centers_loc))
        entropies_over_centers = np.zeros(len(centers_loc)) # entropy at each center
        num_realimgs_over_centers = np.zeros(len(centers_loc))
        for i in range(len(centers_loc)):
            center = centers_loc[i]
            interval_start = center - args.FID_radius
            interval_stop = center + args.FID_radius
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
            labelscores_over_centers[i], _ = cal_labelscore(PreNetLS, fake_images_curr, fn_norm_labels(fake_labels_assigned_curr), min_label_before_shift=0, max_label_after_shift=args.max_label, batch_size = args.eval_batch_size, resize = None, num_workers=args.num_workers)

            print("\r Center:{}; Real:{}; Fake:{}; FID:{}; LS:{}; ET:{}.".format(center, len(real_images_curr), len(fake_images_curr), FID_over_centers[i], labelscores_over_centers[i], entropies_over_centers[i]))

        # average over all centers
        print("\n SFID: {}({}); min/max: {}/{}.".format(np.mean(FID_over_centers), np.std(FID_over_centers), np.min(FID_over_centers), np.max(FID_over_centers)))
        print("\n LS over centers: {}({}); min/max: {}/{}.".format(np.mean(labelscores_over_centers), np.std(labelscores_over_centers), np.min(labelscores_over_centers), np.max(labelscores_over_centers)))
        print("\n Entropy over centers: {}({}); min/max: {}/{}.".format(np.mean(entropies_over_centers), np.std(entropies_over_centers), np.min(entropies_over_centers), np.max(entropies_over_centers)))

        # dump FID versus number of samples (for each center) to npy
        dump_fid_ls_entropy_over_centers_filename = os.path.join(save_setting_folder, 'fid_ls_entropy_over_centers_sampstep{}'.format(args.sampling_timesteps))
        np.savez(dump_fid_ls_entropy_over_centers_filename, fids=FID_over_centers, labelscores=labelscores_over_centers, entropies=entropies_over_centers, nrealimgs=num_realimgs_over_centers, centers=centers_loc)


        #####################
        # FID: Evaluate FID on all fake images
        indx_shuffle_real = np.arange(nreal_all); np.random.shuffle(indx_shuffle_real)
        indx_shuffle_fake = np.arange(nfake_all); np.random.shuffle(indx_shuffle_fake)
        FID = cal_FID(PreNetFID, real_images[indx_shuffle_real], fake_images[indx_shuffle_fake], batch_size=args.eval_batch_size, resize = None, norm_img = True)
        print("\n FID of {} fake images: {}.".format(nfake_all, FID))

        #####################
        # Overall LS: abs(y_assigned - y_predicted)
        ls_mean_overall, ls_std_overall = cal_labelscore(PreNetLS, fake_images, fn_norm_labels(fake_labels), min_label_before_shift=0, max_label_after_shift=args.max_label, batch_size=args.eval_batch_size, resize = None, norm_img = True, num_workers=args.num_workers)
        print("\n Overall LS of {} fake images: {}({}).".format(nfake_all, ls_mean_overall, ls_std_overall))

        #####################
        # Dump evaluation results
        eval_results_logging_fullpath = os.path.join(save_setting_folder, 'eval_results_niters{}.txt'.format(args.niters))
        if not os.path.isfile(eval_results_logging_fullpath):
            eval_results_logging_file = open(eval_results_logging_fullpath, "w")
            eval_results_logging_file.close()
        with open(eval_results_logging_fullpath, 'a') as eval_results_logging_file:
            eval_results_logging_file.write("\n===================================================================================================")
            eval_results_logging_file.write("\n Radius: {}.  \n".format(args.FID_radius))
            print(args, file=eval_results_logging_file)
            eval_results_logging_file.write("\n Sampling Steps: {}.".format(args.sampling_timesteps))
            eval_results_logging_file.write("\n Sampling Time: {:.3f}.".format(total_sample_time))
            eval_results_logging_file.write("\n SFID: {:.3f} ({:.3f}).".format(np.mean(FID_over_centers), np.std(FID_over_centers)))
            eval_results_logging_file.write("\n LS: {:.3f} ({:.3f}).".format(ls_mean_overall, ls_std_overall))
            eval_results_logging_file.write("\n Diversity: {:.3f} ({:.3f}).".format(np.mean(entropies_over_centers), np.std(entropies_over_centers)))
            eval_results_logging_file.write("\n FID: {:.3f}.".format(FID))



print("\n===================================================================================================")
