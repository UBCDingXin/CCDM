import argparse

def parse_opts():
    parser = argparse.ArgumentParser()

    ''' Overall Settings '''
    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--eval_ckpt_path', type=str, default='')
    parser.add_argument('--seed', type=int, default=2024, metavar='S', help='random seed (default: 2020)')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--setting_name', type=str, default='Setup1')


    ''' Dataset '''
    parser.add_argument('--min_label', type=int, default=1, metavar='N')
    parser.add_argument('--max_label', type=int, default=200, metavar='N')
    parser.add_argument('--label_stepsize', type=int, default=2, metavar='N')
    parser.add_argument('--num_channels', type=int, default=1, metavar='N')
    parser.add_argument('--img_size', type=int, default=64, metavar='N')
    parser.add_argument('--num_imgs_per_label', type=int, default=10, metavar='N', help='number of images for each cell count')
    parser.add_argument('--transform', action='store_true', default=False, help='rotate or flip images for GAN training')


    ''' GAN settings '''
    parser.add_argument('--GAN_arch', type=str, default='DCGAN')

    parser.add_argument('--GAN_finetune', action='store_true', default=False)
    parser.add_argument('--path_GAN_ckpt', type=str, default='None') 

    # label embedding setting
    parser.add_argument('--net_embed', type=str, default='ResNet34_embed') #ResNetXX_emebed
    parser.add_argument('--epoch_cnn_embed', type=int, default=400) #epoch of cnn training for label embedding
    parser.add_argument('--resumeepoch_cnn_embed', type=int, default=0) #epoch of cnn training for label embedding
    parser.add_argument('--epoch_net_y2h', type=int, default=500)
    parser.add_argument('--dim_embed', type=int, default=128) #dimension of the embedding space
    parser.add_argument('--batch_size_embed', type=int, default=256, metavar='N')

    parser.add_argument('--loss_type_gan', type=str, default='hinge')
    parser.add_argument('--niters_gan', type=int, default=10000, help='number of iterations')
    parser.add_argument('--resume_niters_gan', type=int, default=0)
    parser.add_argument('--save_niters_freq', type=int, default=2000, help='frequency of saving checkpoints')
    parser.add_argument('--lr_g_gan', type=float, default=1e-4, help='learning rate for generator')
    parser.add_argument('--lr_d_gan', type=float, default=1e-4, help='learning rate for discriminator')
    parser.add_argument('--dim_gan', type=int, default=128, help='Latent dimension of GAN')
    parser.add_argument('--batch_size_disc', type=int, default=256)
    parser.add_argument('--batch_size_gene', type=int, default=256)
    parser.add_argument('--num_D_steps', type=int, default=4, help='number of Ds updates in one iteration')
    parser.add_argument('--visualize_freq', type=int, default=2000, help='frequency of visualization')

    parser.add_argument('--kernel_sigma', type=float, default=-1.0,
                        help='If kernel_sigma<0, then use rule-of-thumb formula to compute the sigma.')
    parser.add_argument('--threshold_type', type=str, default='hard', choices=['soft', 'hard'])
    parser.add_argument('--kappa', type=float, default=-1)
    parser.add_argument('--nonzero_soft_weight_threshold', type=float, default=1e-3,
                        help='threshold for determining nonzero weights for SVDL; we neglect images with too small weights')

    # Gradient accumulation
    parser.add_argument('--num_grad_acc_d', type=int, default=1)
    parser.add_argument('--num_grad_acc_g', type=int, default=1)

    # mixed precision
    parser.add_argument('--use_amp', action='store_true', default=False, help="mixed precision")


    ''' NDA '''
    parser.add_argument('--nda_start_iter', type=int, default=0)

    parser.add_argument('--nda_a', type=float, default=1) #coefficient for the original fake samples' loss
    
    parser.add_argument('--nda_b', type=float, default=0) #coefficient for the real-fake loss; 
    parser.add_argument('--nda_c', type=float, default=0) #coefficient for the real-wrongLabel loss; 
    parser.add_argument('--nda_c_quantile', type=float, default=0.9)
    
    parser.add_argument('--nda_d', type=float, default=0) #coefficient for the bad fake loss; NIQE filtering
    parser.add_argument('--nda_e', type=float, default=0) #coefficient for the bad fake loss; MAE filtering
    parser.add_argument('--nda_d_nfake', type=int, default=-1)
    parser.add_argument('--nda_e_nfake', type=int, default=-1)

    parser.add_argument('--path2badfake1', type=str, default='') 
    parser.add_argument('--path2badfake2', type=str, default='None')
    parser.add_argument('--path2badfake3', type=str, default='None')
    parser.add_argument('--path2badfake4', type=str, default='None')

    
    ''' Sampling and Evaluation '''
    parser.add_argument('--eval_mode', action='store_true', default=False)
    parser.add_argument('--samp_batch_size', type=int, default=200)
    parser.add_argument('--nfake_per_label', type=int, default=1000)
    parser.add_argument('--comp_FID', action='store_true', default=False)
    parser.add_argument('--epoch_FID_CNN', type=int, default=200)
    parser.add_argument('--FID_radius', type=int, default=0)
    parser.add_argument('--dump_fake_for_NIQE', action='store_true', default=False)
    parser.add_argument('--eval_batch_size', type=int, default=200)
    parser.add_argument('--niqe_dump_path', type=str, default='None') 

    args = parser.parse_args()

    return args
