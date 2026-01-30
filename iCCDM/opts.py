import argparse

def parse_opts():
    parser = argparse.ArgumentParser()

    ''' Overall Settings '''
    parser.add_argument('--setting_name', type=str, default='Setup1')
    parser.add_argument('--root_path', type=str, default='./')
    parser.add_argument('--data_name', type=str, default='RC-49', choices=["UTKFace", "RC-49", "Cell200", "SteeringAngle", "RC-49_imb"])
    parser.add_argument('--imb_type', type=str, default='unimodal', choices=['unimodal', 'dualmodal', 'trimodal', 'standard', 'none']) #none means using all data
    parser.add_argument('--data_path', type=str, default='./')
    parser.add_argument('--torch_model_path', type=str, default='None')
    parser.add_argument('--eval_ckpt_path', type=str, default='./')
    parser.add_argument('--seed', type=int, default=111, metavar='S', help='random seed')
    parser.add_argument('--num_workers', type=int, default=0)
    
    ''' Dataset '''
    parser.add_argument('--min_label', type=float, default=0.0)
    parser.add_argument('--max_label', type=float, default=90.0)
    parser.add_argument('--num_channels', type=int, default=3, metavar='N')
    parser.add_argument('--image_size', type=int, default=64, metavar='N')
    parser.add_argument('--max_num_img_per_label', type=int, default=2**20, metavar='N')
    parser.add_argument('--num_img_per_label_after_replica', type=int, default=0, metavar='N')

    ''' Model Config '''
    ## UNet configs
    parser.add_argument('--model_config', type=str)
    
    ## EDM configs
    parser.add_argument('--edm_sigma_min', type=float, default=0.002) # min noise level
    parser.add_argument('--edm_sigma_max', type=float, default=80) # max noise level
    parser.add_argument('--edm_sigma_data_type', type=str, default='default', choices=["default", "local", "global"]) #default: 0.5; global: data variance of all pixel values; local: y-dependent data variance
    parser.add_argument('--edm_sigma_data_default', type=float, default=0.5)
    parser.add_argument('--edm_rho', type=float, default=7) # controls the sampling schedule
    parser.add_argument('--edm_P_mean', type=float, default=-1.2) # mean of log-normal distribution from which noise is drawn for training
    parser.add_argument('--edm_P_std', type=float, default=1.2) # standard deviation of log-normal distribution from which noise is drawn for training
    parser.add_argument('--edm_S_churn', type=float, default=80) # parameters for stochastic sampling - depends on dataset
    parser.add_argument('--edm_S_tmin', type=float, default=0.05) 
    parser.add_argument('--edm_S_tmax', type=float, default=50) 
    parser.add_argument('--edm_S_noise', type=float, default=1.003) 
    
    ''' Training '''   
    ## Diffusion model
    parser.add_argument('--train_num_steps', type=int, default=10, metavar='N')
    parser.add_argument('--resume_step', type=int, default=0, metavar='N')
    parser.add_argument('--train_batch_size', type=int, default=16, metavar='N')
    parser.add_argument('--train_lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--train_amp', action='store_true', default=False)
    parser.add_argument('--train_mixed_precision', type=str, default='fp16')
    parser.add_argument('--gradient_accumulate_every', type=int, default=1, metavar='N')
    parser.add_argument('--sample_every', type=int, default=1000, metavar='N')
    parser.add_argument('--save_every', type=int, default=10000, metavar='N')
    parser.add_argument('--opt_adam_beta1', type=float, default=0.5)
    parser.add_argument('--opt_adam_beta2', type=float, default=0.999)

    ## label embedding setting: 
    # short label embedding
    parser.add_argument('--y2h_embed_type', type=str, default='resnet', choices=['resnet', 'sinusoidal', 'gaussian']) #for y to h
    parser.add_argument('--net_embed', type=str, default='ResNet34_embed') #ResNetXX_emebed
    parser.add_argument('--epoch_cnn_embed', type=int, default=200) #epoch of cnn training for label embedding
    parser.add_argument('--resumeepoch_cnn_embed', type=int, default=0) #epoch of cnn training for label embedding
    parser.add_argument('--epoch_net_y2h', type=int, default=500)
    parser.add_argument('--dim_embed', type=int, default=128) #dimension of the embedding space
    parser.add_argument('--batch_size_embed', type=int, default=256, metavar='N')
    # long label embedding for covariance
    parser.add_argument('--use_y2cov', action='store_true', default=False) #use y-dependent covariance?
    parser.add_argument('--y2cov_hy_weight_train', type=float, default=1.0)
    parser.add_argument('--y2cov_hy_weight_test', type=float, default=1.0)
    parser.add_argument('--y2cov_embed_type', type=str, default='resnet', choices=['resnet', 'sinusoidal', 'gaussian']) #for y to cov
    parser.add_argument('--net_embed_y2cov', type=str, default='ResNet34_embed_y2cov') 
    parser.add_argument('--net_embed_y2cov_y2emb', type=str, default='cnn', choices=["cnn","mlp"])
    parser.add_argument('--epoch_cnn_embed_y2cov', type=int, default=10) 
    parser.add_argument('--resumeepoch_cnn_embed_y2cov', type=int, default=0) 
    parser.add_argument('--epoch_net_y2cov', type=int, default=500)
    parser.add_argument('--batch_size_embed_y2cov', type=int, default=256, metavar='N')

    # Exponential Moving Average
    parser.add_argument('--ema_update_after_step', type=int, default=0) #by default, ema is enabled and starts at 0-th iteration
    parser.add_argument('--ema_update_every', type=int, default=10)
    parser.add_argument('--ema_decay', type=float, default=0.9999)

    ## vicinal loss
    parser.add_argument('--kernel_sigma', type=float, default=-1.0,
                        help='If kernel_sigma<0, then use rule-of-thumb formula to compute the sigma.')
    parser.add_argument('--threshold_type', type=str, default='hard', choices=['soft', 'hard'])
    parser.add_argument('--kappa', type=float, default=-1)
    parser.add_argument('--nonzero_soft_weight_threshold', type=float, default=1e-3,
                        help='threshold for determining nonzero weights for SVDL; we neglect images with too small weights')
    
    ## Adaptive vicinity
    parser.add_argument('--use_ada_vic', action='store_true', default=False) #use adaptive vicinity
    parser.add_argument('--ada_vic_type', type=str, default='vanilla', choices=["vanilla", "hybrid"]) 
    parser.add_argument('--min_n_per_vic', type=int, default=50) #minimum sample size for each vicinity
    parser.add_argument('--use_symm_vic', action='store_true', default=False) #use symmetric adaptive vicinity
    
    # Auxiliary regression loss
    parser.add_argument('--use_aux_reg_loss', action='store_true', default=False) #whether use auxiliary regression penalty?
    parser.add_argument('--aux_reg_loss_type', type=str, default='ei_hinge', choices=['mse', 'ei_hinge']) #mse or epsilon-insensitive hinge loss
    parser.add_argument('--aux_reg_loss_weight', type=float, default=0.0)
    parser.add_argument('--aux_reg_loss_epsilon', type=float, default=-1.0) #epsilon-insensitive loss: if <0, use max_kappa.

    ''' Sampling '''
    parser.add_argument('--sampler', type=str, default='sde', choices=['sde','ode','dpmpp'])
    parser.add_argument('--num_sample_steps', type=int, default=32, metavar='N')
    parser.add_argument('--sample_cond_scale', type=float, default=1.5)
    parser.add_argument('--sample_cond_rescaled_phi', type=float, default=0.7)
    
    parser.add_argument('--do_eval', action='store_true', default=False)
    parser.add_argument('--nfake_per_label', type=int, default=200)
    parser.add_argument('--samp_batch_size', type=int, default=100)
    parser.add_argument('--dump_fake_data', action='store_true', default=False)
    parser.add_argument('--dump_fake_for_niqe', action='store_true', default=False)
    parser.add_argument('--niqe_dump_path', type=str, default='None') 
    parser.add_argument('--eval_batch_size', type=int, default=200)

    args = parser.parse_args()

    return args