# The file for evaluating trained models.
import os
import random
import sys

from PIL import Image
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py
import gc
import copy
import timeit
from datetime import datetime 
import os

from .eval_metrics import cal_FID, cal_labelscore, inception_score, predict_class_labels, compute_entropy

class Evaluator:
    def __init__(self, dataset, trainer, args, device, save_results_folder):
        self.dataset = dataset
        self.trainer = trainer
        self.data_name = args.data_name
        self.img_size = args.image_size
        self.args = args
        self.device = device
        self.save_results_folder = save_results_folder
        
        print("\n ====================================================================================")
        print("\r Start evaluation...")
        
        ## get evaluation labels and real data
        self.real_images, self.real_labels, self.eval_labels = self.dataset.load_evaluation_data()
        self.num_eval_labels = len(self.eval_labels)
        print(self.eval_labels)
        # print("real data shape:", self.real_images.shape)
        
        self.min_label_before_shift, self.max_label_after_shift = dataset.min_label_before_shift, dataset.max_label_after_shift
        print(self.min_label_before_shift, self.max_label_after_shift)
                
        ## num_classes and relative path to checkpoints
        if self.data_name=="UTKFace":
            self.num_classes = 5
            self.nfake_per_label = 1000
            self.nburnin_per_label = 5000
            if args.image_size==64:
                self.fid_net_path = "./evaluation/eval_ckpts/UTKFace/metrics_64x64/ckpt_AE_epoch_200_seed_2020_CVMode_False.pth"
                self.class_net_path = "./evaluation/eval_ckpts/UTKFace/metrics_64x64/ckpt_PreCNNForEvalGANs_ResNet34_class_epoch_200_seed_2020_classify_5_races_CVMode_False.pth"
                self.regre_net_path = "./evaluation/eval_ckpts/UTKFace/metrics_64x64/ckpt_PreCNNForEvalGANs_ResNet34_regre_epoch_200_seed_2020_CVMode_False.pth"
            else:
                self.fid_net_path = "./evaluation/eval_ckpts/UTKFace/metrics_{}x{}/ckpt_AE_epoch_200_seed_2021_CVMode_False.pth".format(args.image_size, args.image_size)
                self.class_net_path = "./evaluation/eval_ckpts/UTKFace/metrics_{}x{}/ckpt_PreCNNForEvalGANs_ResNet34_class_epoch_200_seed_2021_classify_5_races_CVMode_False.pth".format(args.image_size, args.image_size)
                self.regre_net_path = "./evaluation/eval_ckpts/UTKFace/metrics_{}x{}/ckpt_PreCNNForEvalGANs_ResNet34_regre_epoch_200_seed_2021_CVMode_False.pth".format(args.image_size, args.image_size)
        elif self.data_name in ["RC-49", "RC-49_imb"]:
            self.num_classes = 49
            self.nfake_per_label = 200
            self.nburnin_per_label = 1000
            if args.image_size==64:
                self.fid_net_path = "./evaluation/eval_ckpts/RC49/metrics_64x64/ckpt_AE_epoch_200_seed_2020_CVMode_False.pth"
                self.class_net_path = "./evaluation/eval_ckpts/RC49/metrics_64x64/ckpt_PreCNNForEvalGANs_ResNet34_class_epoch_200_seed_2020_classify_49_chair_types_CVMode_False.pth"
                self.regre_net_path = "./evaluation/eval_ckpts/RC49/metrics_64x64/ckpt_PreCNNForEvalGANs_ResNet34_regre_epoch_200_seed_2020_CVMode_False.pth"
            else:
                raise ValueError("Not Supported Resolution!")
        elif self.data_name=="SteeringAngle":
            self.num_classes = 5
            self.nfake_per_label = 50
            self.nburnin_per_label = 250
            self.fid_net_path = "./evaluation/eval_ckpts/SteeringAngle/metrics_{}x{}/ckpt_AE_epoch_200_seed_2020_CVMode_False.pth".format(args.image_size, args.image_size)
            self.class_net_path = "./evaluation/eval_ckpts/SteeringAngle/metrics_{}x{}/ckpt_PreCNNForEvalGANs_ResNet34_class_epoch_20_seed_2020_classify_5_scenes_CVMode_False.pth".format(args.image_size, args.image_size)
            self.regre_net_path = "./evaluation/eval_ckpts/SteeringAngle/metrics_{}x{}/ckpt_PreCNNForEvalGANs_ResNet34_regre_epoch_200_seed_2020_CVMode_False.pth".format(args.image_size, args.image_size)
        elif self.data_name=="Cell200":
            self.num_classes = 0
            self.nfake_per_label = 1000
            self.nburnin_per_label = 5000
            self.class_net_path = None
            self.fid_net_path = "./evaluation/eval_ckpts/Cell200/metrics_64x64/ckpt_AE_epoch_50_seed_2020_CVMode_False.pth"
            self.regre_net_path = "./evaluation/eval_ckpts/Cell200/metrics_64x64/ckpt_PreCNNForEvalGANs_ResNet34_regre_epoch_200_seed_2020_Transformation_True_Cell_200.pth"
        else:
            raise ValueError("Not Supported Dataset!")
        
        print("Dataset: {}; Resolution: {}x{}; num_classes: {}".format(self.data_name, self.img_size, self.img_size, self.num_classes))
        print(self.fid_net_path)
        print(self.class_net_path)
        print(self.regre_net_path)   
        
        
        ### folder for dumping h5 files
        
        dump_fake_images_folder = os.path.join(save_results_folder, 'fake_data_steps{}_nfake{}_{}_scale{}_sampstep{}_trw{}_tew{}'.format(args.train_num_steps, int(args.nfake_per_label*self.num_eval_labels), args.sampler, args.sample_cond_scale, args.num_sample_steps, args.y2cov_hy_weight_train, args.y2cov_hy_weight_test))
        os.makedirs(dump_fake_images_folder, exist_ok=True)

        ## generating fake data
        
        print("\n Start generating fake image-label pairs for evaluation...")
        
        self.fake_images = []
        self.fake_labels = []
        total_sample_time = 0
        for i in range(self.num_eval_labels):
            print('\n [{}/{}]: Generating fake data for label {}...'.format(i+1, self.num_eval_labels, self.eval_labels[i]))
            curr_label = self.eval_labels[i]
            dump_fake_images_filename = os.path.join(dump_fake_images_folder, '{}.h5'.format(curr_label))
            if not os.path.isfile(dump_fake_images_filename):
                fake_labels_i = curr_label*np.ones(args.nfake_per_label)
                start = timeit.default_timer()
                fake_images_i, _ = trainer.sample_given_labels(given_labels = dataset.fn_normalize_labels(fake_labels_i), batch_size = args.samp_batch_size, num_sample_steps=args.num_sample_steps, denorm=True, to_numpy=True, verbose=False, sampler=args.sampler, cond_scale=args.sample_cond_scale, rescaled_phi=args.sample_cond_rescaled_phi)
                
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
                    fake_labels_i = f['fake_labels_i'][:]
                    sample_time_i = f['sample_time_i'][0]
                assert len(fake_images_i) == len(fake_labels_i)
            ##end if
            total_sample_time+=sample_time_i
            self.fake_images.append(fake_images_i)
            self.fake_labels.append(fake_labels_i)
            print("\n [{}/{}]: Got {} fake images for label {}. Time spent {:.2f}, Total time {:.2f}.".format(i+1, self.num_eval_labels, len(fake_images_i), curr_label, sample_time_i, total_sample_time))
            
            ## dump some imgs for visualization
            img_vis_i = fake_images_i[0:36]/255.0
            img_vis_i = torch.from_numpy(img_vis_i)
            img_filename = os.path.join(dump_fake_images_folder, 'sample_{}.png'.format(curr_label))
            torchvision.utils.save_image(img_vis_i.data, img_filename, nrow=6, normalize=False)
            del fake_images_i, fake_labels_i; gc.collect()
            
        ##end for i
        self.sampling_time = total_sample_time
        self.fake_images = np.concatenate(self.fake_images, axis=0)
        assert self.fake_images.max()>1.0
        self.fake_labels = np.concatenate(self.fake_labels)
        print("\n Sampling finished; Time elapses: {}s".format(total_sample_time))
        print("\n Fake images shape:", self.fake_images.shape)
        
        
    
    ## method for computing evaluation metrics: SFID, Diversity, LS, FID, IS
    def compute_metrics(self, output_path, PreNetFID, PreNetDiversity, PreNetLS):
        
        PreNetFID = PreNetFID.to(self.device)
        if PreNetDiversity is not None:
            PreNetDiversity = PreNetDiversity.to(self.device)
        PreNetLS = PreNetLS.to(self.device)
        
        ## load pre-trained evaluation models from checkpoints
        # for FID
        checkpoint_PreNet = torch.load(self.fid_net_path, weights_only=True)
        PreNetFID.load_state_dict(checkpoint_PreNet['net_encoder_state_dict'])
        # for Diversity
        if self.data_name in ["UTKFace", "RC-49", "RC-49_imb", "SteeringAngle"]:
            checkpoint_PreNet = torch.load(self.class_net_path, weights_only=True)
            PreNetDiversity.load_state_dict(checkpoint_PreNet['net_state_dict'])
        # for LS
        checkpoint_PreNet = torch.load(self.regre_net_path, weights_only=True)
        PreNetLS.load_state_dict(checkpoint_PreNet['net_state_dict'])
        
        # normalize labels
        nfake_all = len(self.fake_images)
        nreal_all = len(self.real_images)
        
        FID_radius = 0.0
        if self.data_name=="UTKFace":
            centers_loc = np.arange(1, 61) #not normalized
        elif self.data_name in ["RC-49", "RC-49_imb"]:
            centers_loc = self.eval_labels #not normalized
        elif self.data_name=="SteeringAngle":
            FID_radius=2.0
            center_start = np.min(self.real_labels)+FID_radius
            center_stop = np.max(self.real_labels)-FID_radius
            centers_loc = np.linspace(center_start, center_stop, 1000) #not normalized
        elif self.data_name=="Cell200":
            centers_loc = np.arange(1, 201) #not normalized
        print(centers_loc)
        
        FID_over_centers = np.zeros(len(centers_loc))
        entropies_over_centers = np.zeros(len(centers_loc)) # entropy at each center
        labelscores_over_centers = np.zeros(len(centers_loc)) #label score at each center
        num_realimgs_over_centers = np.zeros(len(centers_loc))
        
        start_time = timeit.default_timer()
        
        for i in range(len(centers_loc)):
            center = centers_loc[i]
            interval_start = center - FID_radius
            interval_stop = center + FID_radius
            indx_real = np.where((self.real_labels>=interval_start)*(self.real_labels<=interval_stop)==True)[0]
            np.random.shuffle(indx_real)
            real_images_i = self.real_images[indx_real]
            real_images_i = (real_images_i/255.0-0.5)/0.5
            num_realimgs_over_centers[i] = len(real_images_i)
            indx_fake = np.where((self.fake_labels>=interval_start)*(self.fake_labels<=interval_stop)==True)[0]
            np.random.shuffle(indx_fake)
            fake_images_i = self.fake_images[indx_fake]
            fake_images_i = (fake_images_i/255.0-0.5)/0.5
            fake_labels_assigned_i = self.fake_labels[indx_fake]
            # FID
            FID_over_centers[i] = cal_FID(PreNetFID, real_images_i, fake_images_i, batch_size=self.args.eval_batch_size, resize = None, device=self.device)
            # Entropy of predicted class labels
            if self.data_name in ["UTKFace", "RC-49", "RC-49_imb", "SteeringAngle"]:
                predicted_class_labels = predict_class_labels(PreNetDiversity, fake_images_i, batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers, device=self.device)
                entropies_over_centers[i] = compute_entropy(predicted_class_labels)
            # Label score
            labelscores_over_centers[i], _ = cal_labelscore(PreNetLS, fake_images_i, self.dataset.fn_normalize_labels(fake_labels_assigned_i), min_label_before_shift=self.min_label_before_shift, max_label_after_shift=self.max_label_after_shift, batch_size = self.args.eval_batch_size, resize = None, num_workers=self.args.num_workers, device=self.device)

            print("\n Center:{}; Real:{}; Fake:{}; FID:{:.3f}; LS:{:.3f}; ET:{:.3f}; Time:{:.3f}.".format(center, len(real_images_i), len(fake_images_i), FID_over_centers[i], labelscores_over_centers[i], entropies_over_centers[i], timeit.default_timer()-start_time))
        ##end for i
        
        # average over all centers
        print("\n SFID: {:.3f} ({:.3f}); min/max: {:.3f}/{:.3f}. \n".format(np.mean(FID_over_centers), np.std(FID_over_centers), np.min(FID_over_centers), np.max(FID_over_centers)))
        print("\n LS over centers: {:.3f} ({:.3f}); min/max: {:.3f}/{:.3f}. \n".format(np.mean(labelscores_over_centers), np.std(labelscores_over_centers), np.min(labelscores_over_centers), np.max(labelscores_over_centers)))
        print("\n Entropy over centers: {:.3f} ({:.3f}); min/max: {:.3f}/{:.3f}. \n".format(np.mean(entropies_over_centers), np.std(entropies_over_centers), np.min(entropies_over_centers), np.max(entropies_over_centers)))

        # dump FID versus number of samples (for each center) to npy
        dump_fid_ls_entropy_over_centers_filename = output_path + '/fid_ls_entropy_over_centers'
        np.savez(dump_fid_ls_entropy_over_centers_filename, fids=FID_over_centers, labelscores=labelscores_over_centers, entropies=entropies_over_centers, nrealimgs=num_realimgs_over_centers, centers=centers_loc)
        
        #####################
        # Overall LS: abs(y_assigned - y_predicted)
        ls_mean_overall, ls_std_overall = cal_labelscore(PreNetLS, self.fake_images, self.dataset.fn_normalize_labels(self.fake_labels), min_label_before_shift=self.min_label_before_shift, max_label_after_shift=self.max_label_after_shift, batch_size=self.args.eval_batch_size, resize = None, norm_img = True, num_workers=self.args.num_workers, device=self.device)
        print("Overall LS of {} fake images: {:.3f} ({:.3f}). \n".format(nfake_all, ls_mean_overall, ls_std_overall))
        
        #####################
        # FID: Evaluate FID on all fake images
        indx_shuffle_real = np.arange(nreal_all); np.random.shuffle(indx_shuffle_real)
        indx_shuffle_fake = np.arange(nfake_all); np.random.shuffle(indx_shuffle_fake)
        FID = cal_FID(PreNetFID, self.real_images[indx_shuffle_real], self.fake_images[indx_shuffle_fake], batch_size=self.args.eval_batch_size, resize = None, norm_img = True, device=self.device)
        print("FID of {} fake images: {:.3f}.\n".format(nfake_all, FID))

        #####################
        # Compute IS
        if self.data_name != "Cell200":
            IS, IS_std = inception_score(imgs=self.fake_images[indx_shuffle_fake], num_classes=self.num_classes, net=PreNetDiversity, batch_size=self.args.eval_batch_size, splits=10, normalize_img=True, device=self.device)
            print("IS of {} fake images: {:.3f} ({:.3f}).\n".format(nfake_all, IS, IS_std))
        
        #####################
        # Dump evaluation results
        now = datetime.now()
        time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        eval_results_logging_fullpath = os.path.join(output_path, 'eval_results_{}.txt'.format(time_str))
        if not os.path.isfile(eval_results_logging_fullpath):
            eval_results_logging_file = open(eval_results_logging_fullpath, "w")
            eval_results_logging_file.close()
        with open(eval_results_logging_fullpath, 'a') as eval_results_logging_file:
            eval_results_logging_file.write("\n===================================================================================================")
            print(self.args, file=eval_results_logging_file)
            eval_results_logging_file.write("\n Setting name: {}.".format(time_str))
            eval_results_logging_file.write("\n Sampling time: {:.3f}".format(self.sampling_time))
            eval_results_logging_file.write("\n SFID: {:.3f} ({:.3f})".format(np.mean(FID_over_centers), np.std(FID_over_centers)))
            eval_results_logging_file.write("\n LS: {:.3f} ({:.3f})".format(ls_mean_overall, ls_std_overall))
            eval_results_logging_file.write("\n Diversity: {:.3f} ({:.3f})".format(np.mean(entropies_over_centers), np.std(entropies_over_centers)))
            eval_results_logging_file.write("\n FID: {:.3f}".format(FID))
            if self.data_name in ["UTKFace", "RC-49", "RC-49_imb", "SteeringAngle"]:
                eval_results_logging_file.write("\n IS (STD): {:.3f} ({:.3f})".format(IS, IS_std)) 
        

    ##end def compute_metrics
        
        
    # ## method for dumping png images
    # def dump_png_images(self, output_path):
    #     print("\n Dumping fake images for NIQE >>>")
    #     os.makedirs(output_path, exist_ok=True)
        
    #     for i in tqdm(range(len(self.fake_images))):
    #         label_i = self.fake_labels[i]
    #         if self.data_name in ["RC-49","RC-49_imb"]:
    #             filename_i = output_path + "/{}_{:.1f}.png".format(i, label_i)
    #         elif self.data_name == "SteeringAngle":
    #             filename_i = output_path + "/{}_{:.6f}.png".format(i, label_i)
    #         else:
    #             # filename_i = output_path + "/{}_{}.png".format(i, int(label_i))
    #             filename_i = output_path + "/{}_{}.png".format(i, round(label_i))
    #         os.makedirs(os.path.dirname(filename_i), exist_ok=True)
    #         image_i = self.fake_images[i].astype(np.uint8)
    #         image_i_pil = Image.fromarray(image_i.transpose(1,2,0))
    #         image_i_pil.save(filename_i)
    #     #end for i
    
    ## method for dumping png images
    def dump_png_images(self, output_path):
        print("\n Dumping fake images for NIQE >>>")
        os.makedirs(output_path, exist_ok=True)
        
        def array_to_pil(img_array):
            """Convert numpy array to PIL image, handling various shapes"""
            # Ensure uint8 type
            if img_array.dtype != np.uint8:
                if img_array.max() <= 1.0:
                    img_array = (img_array * 255).astype(np.uint8)
                else:
                    img_array = img_array.astype(np.uint8)
            
            # Squeeze singleton dimensions
            img_array = img_array.squeeze()
            
            # Determine image mode based on dimensions
            if len(img_array.shape) == 2:
                return Image.fromarray(img_array, mode='L')
            elif len(img_array.shape) == 3 and img_array.shape[2] == 3:
                return Image.fromarray(img_array, mode='RGB')
            elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
                return Image.fromarray(img_array, mode='RGBA')
            elif len(img_array.shape) == 3 and img_array.shape[2] == 1:
                return Image.fromarray(img_array[:,:,0], mode='L')
            else:
                # Try automatic processing
                return Image.fromarray(img_array)
        
        for i in tqdm(range(len(self.fake_images))):
            label_i = self.fake_labels[i]
            if self.data_name in ["RC-49", "RC-49_imb"]:
                filename_i = output_path + "/{}_{:.1f}.png".format(i, label_i)
            elif self.data_name == "SteeringAngle":
                filename_i = output_path + "/{}_{:.6f}.png".format(i, label_i)
            else:
                # filename_i = output_path + "/{}_{}.png".format(i, int(label_i))
                filename_i = output_path + "/{}_{}.png".format(i, round(label_i))
            os.makedirs(os.path.dirname(filename_i), exist_ok=True)
            
            # Convert to PIL image and save
            image_i_pil = array_to_pil(self.fake_images[i].transpose(1,2,0))
            image_i_pil.save(filename_i)
        #end for i
    
    
    ## method for dumping h5 files
    def dump_h5_files(self, output_path):
        print("\n Dumping h5 files...")
        for i in range(self.num_eval_labels):
            label_i = self.eval_labels[i]
            indx_i = np.where(self.fake_labels==label_i)[0]
            fake_images_i = self.fake_images[indx_i]
            fake_labels_i = self.fake_labels[indx_i]
            print('\r [{}/{}]: Got {} fake images for label {} >>>'.format(i+1, self.num_eval_labels, len(indx_i), label_i))
            dump_fake_images_filename = os.path.join(output_path, '{}.h5'.format(label_i))
            with h5py.File(dump_fake_images_filename, "w") as f:
                f.create_dataset('fake_images_i', data = fake_images_i, dtype='uint8', compression="gzip", compression_opts=6)
                f.create_dataset('fake_labels_i', data = fake_labels_i, dtype='float')
                f.create_dataset('sample_time_i', data = np.array([0]), dtype='float') #to avoid error
            ## dump some imgs for visualization
            img_vis_i = fake_images_i[0:36]/255.0
            img_vis_i = torch.from_numpy(img_vis_i)
            img_filename = os.path.join(output_path, 'sample_{}.png'.format(label_i))
            torchvision.utils.save_image(img_vis_i.data, img_filename, nrow=6, normalize=False)
            del fake_images_i, fake_labels_i; gc.collect()
        ##end for i
            














