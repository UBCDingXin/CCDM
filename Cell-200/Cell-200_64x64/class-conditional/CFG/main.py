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
data_filename = args.data_path + '/Cell200_{}x{}.h5'.format(args.img_size, args.img_size)
hf = h5py.File(data_filename, 'r')
labels = hf['CellCounts'][:]
labels = labels.astype(float)
images = hf['IMGs_grey'][:]
hf.close()

raw_images = copy.deepcopy(images)
raw_labels = copy.deepcopy(labels)

# for each label select num_imgs_per_label
selected_labels = np.arange(args.min_label, args.max_label+1, args.stepsize)
n_unique_labels = len(selected_labels)


for i in range(n_unique_labels):
    curr_label = selected_labels[i]
    index_curr_label = np.where(labels==curr_label)[0]
    if i == 0:
        images_subset = images[index_curr_label[0:args.num_imgs_per_label]]
        labels_subset = labels[index_curr_label[0:args.num_imgs_per_label]]
    else:
        images_subset = np.concatenate((images_subset, images[index_curr_label[0:args.num_imgs_per_label]]), axis=0)
        labels_subset = np.concatenate((labels_subset, labels[index_curr_label[0:args.num_imgs_per_label]]))
# for i
images = images_subset
labels = labels_subset
del images_subset, labels_subset; gc.collect()

print("\r We have {} images with {} distinct labels".format(len(images), n_unique_labels))

# treated as classification; convert regression labels to class labels
unique_labels = np.sort(np.array(list(set(raw_labels)))) #not counts because we want the last element is the max_count
num_unique_labels = len(unique_labels)
print("{} distinct labels are split into {} classes".format(num_unique_labels, args.num_classes))

## convert regression labels to class labels and vice versa
### step 1: prepare two dictionaries
label2class = dict()
class2label = dict()
num_labels_per_class = num_unique_labels//args.num_classes
class_cutoff_points = [unique_labels[0]] #the cutoff points on [min_label, max_label] to determine classes; each interval is a class
curr_class = 0
for i in range(num_unique_labels):
    label2class[unique_labels[i]]=curr_class
    if (i+1)%num_labels_per_class==0 and (curr_class+1)!=args.num_classes:
        curr_class += 1
        class_cutoff_points.append(unique_labels[i+1])
class_cutoff_points.append(unique_labels[-1])
assert len(class_cutoff_points)-1 == args.num_classes

### the label of each interval equals to the average of the two end points
for i in range(args.num_classes):
    class2label[i] = (class_cutoff_points[i]+class_cutoff_points[i+1])/2

### step 2: convert regression label to class labels
labels_new = -1*np.ones(len(labels))
for i in range(len(labels)):
    labels_new[i] = label2class[labels[i]]
assert np.sum(labels_new<0)==0
labels = labels_new
del labels_new; gc.collect()
unique_labels = np.sort(np.array(list(set(labels)))).astype(int)
print("\n The class labels are \r")
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
        dim_mults = (1, 2, 2, 4),
        channels = args.num_channels,
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

trainset = IMGs_dataset(images, labels, normalize=True, transform=True)


## for visualization
n_row=5; n_col = n_row
start_label = 0
end_label = 99
selected_labels = np.linspace(start_label, end_label, num=n_row)
y_visual = np.zeros(n_row*n_col)
for i in range(n_row):
    curr_label = selected_labels[i]
    for j in range(n_col):
        y_visual[i*n_col+j] = curr_label
y_visual = y_visual.astype(int)
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
    #for FID
    PreNetFID = encoder(dim_bottleneck=512).cuda()
    PreNetFID = nn.DataParallel(PreNetFID)
    Filename_PreCNNForEvalGANs = os.path.join(args.eval_ckpt_path, 'ckpt_AE_epoch_50_seed_2020_CVMode_False.pth')
    checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs)
    PreNetFID.load_state_dict(checkpoint_PreNet['net_encoder_state_dict'])

    # for LS
    PreNetLS = ResNet34_regre_eval(ngpu = torch.cuda.device_count()).cuda()
    Filename_PreCNNForEvalGANs_LS = os.path.join(args.eval_ckpt_path, 'ckpt_PreCNNForEvalGANs_ResNet34_regre_epoch_200_seed_2020_Transformation_True_Cell_200.pth')
    checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs_LS)
    PreNetLS.load_state_dict(checkpoint_PreNet['net_state_dict'])


    #####################
    # generate nfake images
    print("Start sampling {} fake images per label >>>".format(args.nfake_per_label))

    eval_labels = np.arange(args.min_label, args.max_label+1) 
    num_eval_labels = len(eval_labels)
    print(eval_labels)

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
            image_i_pil = Image.fromarray(image_i[0])
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
    
    #####################
    # Evaluate FID within a sliding window with a radius R on the label's range (i.e., [1,max_label]). The center of the sliding window locate on [R+1,2,3,...,max_label-R].
    center_start = 1+args.FID_radius
    center_stop = args.max_label-args.FID_radius
    centers_loc = np.arange(center_start, center_stop+1)
    FID_over_centers = np.zeros(len(centers_loc))
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
        # Label score
        labelscores_over_centers[i], _ = cal_labelscore(PreNetLS, fake_images_curr, fake_labels_assigned_curr, min_label_before_shift=0, max_label_after_shift=args.max_label, batch_size = args.eval_batch_size, resize = None, num_workers=args.num_workers)

        print("\r Center:{}; Real:{}; Fake:{}; FID:{}; LS:{}.".format(center, len(real_images_curr), len(fake_images_curr), FID_over_centers[i], labelscores_over_centers[i]))

    # average over all centers
    print("\n SFID: {}({}); min/max: {}/{}.".format(np.mean(FID_over_centers), np.std(FID_over_centers), np.min(FID_over_centers), np.max(FID_over_centers)))
    print("\n LS over centers: {}({}); min/max: {}/{}.".format(np.mean(labelscores_over_centers), np.std(labelscores_over_centers), np.min(labelscores_over_centers), np.max(labelscores_over_centers)))

    # dump FID versus number of samples (for each center) to npy
    dump_fid_ls_entropy_over_centers_filename = os.path.join(save_setting_folder, 'fid_ls_entropy_over_centers_sampstep{}'.format(args.sampling_timesteps))
    np.savez(dump_fid_ls_entropy_over_centers_filename, fids=FID_over_centers, labelscores=labelscores_over_centers, nrealimgs=num_realimgs_over_centers, centers=centers_loc)


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
        eval_results_logging_file.write("\n Sampling Steps: {}.".format(args.sampling_timesteps))
        eval_results_logging_file.write("\n Sampling Time: {:.3f}.".format(total_sample_time))
        eval_results_logging_file.write("\n SFID: {:.3f} ({:.3f}).".format(np.mean(FID_over_centers), np.std(FID_over_centers)))
        eval_results_logging_file.write("\n LS: {:.3f} ({:.3f}).".format(ls_mean_overall, ls_std_overall))
        eval_results_logging_file.write("\n FID: {:.3f}.".format(FID))



print("\n===================================================================================================")
