"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os
import h5py
import timeit
from tqdm import tqdm, trange
import random
import timeit
import sys
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib as mpl
from PIL import Image
import gc
import copy

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)


from utils import IMGs_dataset, SimpleProgressBar, compute_entropy, predict_class_labels
from eval_models import ResNet34_regre_eval, ResNet34_class_eval, encoder
from eval_metrics import cal_FID, cal_labelscore, inception_score


def main():
    args = create_argparser().parse_args()



    #######################################################################################
    '''                                Output folders                                  '''
    #######################################################################################

    os.chdir(args.root_dir)
    path_to_output = os.path.join(args.root_dir, 'output/exp_{}/sampling'.format(args.setup_name))
    os.makedirs(path_to_output, exist_ok=True)



    #######################################################################################
    '''                                    Data loader                                 '''
    #######################################################################################

    # load data from h5 file
    data_filename = args.data_dir + '/Cell200_{}x{}.h5'.format(args.image_size, args.image_size)
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
    '''                                   Load Models                                   '''
    #######################################################################################

    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    def cond_fn(x, t, y=None):
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return torch.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)


    #######################################################################################
    '''                                    Sampling                                     '''
    #######################################################################################

    def sample_given_labels(given_labels, diffusion, model_fn=model_fn, cond_fn=cond_fn, class_cutoff_points=class_cutoff_points, batch_size = args.samp_batch_size, image_size=args.image_size, use_ddim=args.use_ddim, clip_denoised=args.clip_denoised, denorm=True, to_numpy=True, verbose=False):
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

        sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
        )

        if batch_size>nfake:
            batch_size = nfake
        fake_images = []
        tmp = 0
        while tmp < nfake:
            batch_fake_labels = torch.from_numpy(given_class_labels[tmp:(tmp+batch_size)]).type(torch.long).to(dist_util.dev())

            model_kwargs = {}
            model_kwargs["y"] = batch_fake_labels

            batch_fake_images = sample_fn(
                model_fn,
                (batch_size, 1, image_size, image_size),
                clip_denoised=clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
                device=dist_util.dev(),
            )
            batch_fake_images = batch_fake_images.clamp(-1, 1).contiguous().cpu()           

            # batch_fake_images = torch.clamp(batch_fake_images, 0, 1)
            if denorm: #denorm imgs to save memory
                batch_fake_images = ((batch_fake_images + 1) * 127.5).clamp(0, 255).type(torch.uint8)

            fake_images.append(batch_fake_images)
            tmp += batch_size
            if verbose:
                print("\r {}/{} complete...".format(tmp, nfake))

        fake_images = torch.cat(fake_images, dim=0)
        #remove extra entries
        fake_images = fake_images[0:nfake]

        if to_numpy:
            fake_images = fake_images.numpy()
        else:
            given_labels = torch.from_numpy(given_labels)

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
        print("\n Start sampling {} fake images per label >>>".format(args.nfake_per_label))

        eval_labels = np.arange(args.min_label, args.max_label+1) 
        num_eval_labels = len(eval_labels)
        print(eval_labels)
        
        ###########################################
        ''' multiple h5 files '''
        print('\n Start generating fake data...')
        dump_fake_images_folder = os.path.join(path_to_output, 'fake_data_nfake{}'.format( int(args.nfake_per_label*num_eval_labels) ))
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
        print("\r Sampling finished; Time elapses: {}s".format(total_sample_time))

        ### dump for computing NIQE
        if args.dump_fake_for_NIQE:
            print("\n Dumping fake images for NIQE...")
            if args.niqe_dump_path=="None":
                dump_fake_images_folder = path_to_output + '/fake_images'
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
            labelscores_over_centers[i], _ = cal_labelscore(PreNetLS, fake_images_curr, fake_labels_assigned_curr, min_label_before_shift=0, max_label_after_shift=args.max_label, batch_size = args.eval_batch_size, resize = None, num_workers=0)

            print("\r Center:{}; Real:{}; Fake:{}; FID:{}; LS:{}".format(center, len(real_images_curr), len(fake_images_curr), FID_over_centers[i], labelscores_over_centers[i]))

        # average over all centers
        print("\n SFID: {}({}); min/max: {}/{}.".format(np.mean(FID_over_centers), np.std(FID_over_centers), np.min(FID_over_centers), np.max(FID_over_centers)))
        print("\n LS over centers: {}({}); min/max: {}/{}.".format(np.mean(labelscores_over_centers), np.std(labelscores_over_centers), np.min(labelscores_over_centers), np.max(labelscores_over_centers)))

        # dump FID versus number of samples (for each center) to npy
        dump_fid_ls_entropy_over_centers_filename = os.path.join(path_to_output, 'fid_ls_entropy_over_centers')
        np.savez(dump_fid_ls_entropy_over_centers_filename, fids=FID_over_centers, labelscores=labelscores_over_centers, nrealimgs=num_realimgs_over_centers, centers=centers_loc)

        #####################
        # FID: Evaluate FID on all fake images
        indx_shuffle_real = np.arange(nreal_all); np.random.shuffle(indx_shuffle_real)
        indx_shuffle_fake = np.arange(nfake_all); np.random.shuffle(indx_shuffle_fake)
        FID = cal_FID(PreNetFID, real_images[indx_shuffle_real], fake_images[indx_shuffle_fake], batch_size=args.eval_batch_size, resize = None, norm_img = True)
        print("\n FID of {} fake images: {}.".format(nfake_all, FID))

        #####################
        # Overall LS: abs(y_assigned - y_predicted)
        ls_mean_overall, ls_std_overall = cal_labelscore(PreNetLS, fake_images, fake_labels, min_label_before_shift=0, max_label_after_shift=args.max_label, batch_size=args.eval_batch_size, resize = None, norm_img = True, num_workers=0)
        print("\n Overall LS of {} fake images: {}({}).".format(nfake_all, ls_mean_overall, ls_std_overall))

        #####################
        # Dump evaluation results
        eval_results_logging_fullpath = os.path.join(path_to_output, 'eval_results.txt')
        if not os.path.isfile(eval_results_logging_fullpath):
            eval_results_logging_file = open(eval_results_logging_fullpath, "w")
            eval_results_logging_file.close()
        with open(eval_results_logging_fullpath, 'a') as eval_results_logging_file:
            eval_results_logging_file.write("\n===================================================================================================")
            print(args, file=eval_results_logging_file)
            eval_results_logging_file.write("\n Sampling Time: {:.3f}.".format(total_sample_time))
            eval_results_logging_file.write("\n SFID: {:.3f} ({:.3f}).".format(np.mean(FID_over_centers), np.std(FID_over_centers)))
            eval_results_logging_file.write("\n LS: {:.3f} ({:.3f}).".format(ls_mean_overall, ls_std_overall))
            eval_results_logging_file.write("\n FID: {:.3f}.".format(FID))


def create_argparser():
    defaults = dict(
        setup_name="Setup1",
        root_dir="",
        data_dir="",
        min_label=1,
        max_label=200,
        stepsize=2,
        num_classes=100,
        num_imgs_per_label=10,

        clip_denoised=True,
        nfake_per_label=1000,
        samp_batch_size=500,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
        dump_fake_data=False,

        comp_FID=False,
        eval_ckpt_path="None",
        FID_radius=0,
        dump_fake_for_NIQE=False,
        niqe_dump_path="None",
        comp_IS_and_FID_only=False,
        eval_batch_size=200
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    print("\n===================================================================================================")
    main()
    print("\n===================================================================================================")
