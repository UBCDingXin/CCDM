::===============================================================
:: This is a batch script for running the program on windows! 
::===============================================================

@echo off

set CUDA_VISIBLE_DEVICES=0

set ROOT_PREFIX=D:/BaiduSyncdisk/Baidu_WD/CCGM/CcDDPM/RC-49/RC-49_64x64
set ROOT_PATH=%ROOT_PREFIX%/CCGM/Dual-NDA
set DATA_PATH=D:/BaiduSyncdisk/Baidu_WD/datasets/CCGM_or_regression/RC-49
set EVAL_PATH=%ROOT_PREFIX%/evaluation

set SEED=2024
set NUM_WORKERS=0
set MIN_LABEL=0
set MAX_LABEL=90.0
set IMG_SIZE=64
set MAX_N_IMG_PER_LABEL=25
set MAX_N_IMG_PER_LABEL_AFTER_REPLICA=0

set BATCH_SIZE_G=256
set BATCH_SIZE_D=256
set NUM_D_STEPS=2
set SIGMA=-1.0
set KAPPA=-2.0
set LR_G=1e-4
set LR_D=1e-4
set NUM_ACC_D=1
set NUM_ACC_G=1

set GAN_ARCH="SAGAN"
set LOSS_TYPE="hinge"

set DIM_GAN=256
set DIM_EMBED=128

set SETTING="Setup4"

set fake_data_path_1=%ROOT_PREFIX%/CCGM/CcGAN/output/CcGAN_SAGAN_soft_si0.047_ka50625.000_hinge_nDs2_nDa1_nGa1_Dbs256_Gbs256_DiAu/bad_fake_data/exp_niters30000/badfake_NIQE0.4_nfake108000.h5
set fake_data_path_2=None
set fake_data_path_3=None
set fake_data_path_4=None

@REM set gan_ckpt_path=%ROOT_PREFIX%/CCGM/CcGAN/output/CcGAN_SAGAN_soft_si0.047_ka50625.000_hinge_nDs2_nDa1_nGa1_Dbs256_Gbs256_DiAu/saved_models/ckpts_in_train/ckpt_niter_30000.pth
set gan_ckpt_path="None"

set nda_c_quantile=0.9
set nfake_d=18000
set nda_start_iter=30000

set niqe_dump_path="C:/LocalWD/CcGAN_TPAMI_NIQE/RC-49/NIQE_64x64/fake_data"
set NITERS=35000
set resume_niter=30000
python main.py ^
    --setting_name %SETTING% --root_path %ROOT_PATH% --data_path %DATA_PATH% --eval_ckpt_path %EVAL_PATH% --seed %SEED% --num_workers %NUM_WORKERS% ^
    --min_label %MIN_LABEL% --max_label %MAX_LABEL% --img_size %IMG_SIZE% ^
    --max_num_img_per_label %MAX_N_IMG_PER_LABEL% --max_num_img_per_label_after_replica %MAX_N_IMG_PER_LABEL_AFTER_REPLICA% ^
    --GAN_arch %GAN_ARCH% --niters_gan %NITERS% --resume_niters_gan %resume_niter% --loss_type_gan %LOSS_TYPE% --num_D_steps %NUM_D_STEPS% ^
    --save_niters_freq 5000 --visualize_freq 2000 ^
    --batch_size_disc %BATCH_SIZE_D% --batch_size_gene %BATCH_SIZE_G% ^
    --num_grad_acc_d %NUM_ACC_D% --num_grad_acc_g %NUM_ACC_G% ^
    --lr_g %LR_G% --lr_d %LR_D% --dim_gan %DIM_GAN% --dim_embed %DIM_EMBED% ^
    --kernel_sigma %SIGMA% --threshold_type soft --kappa %KAPPA% ^
    --gan_DiffAugment --gan_DiffAugment_policy color,translation,cutout ^
    --nda_start_iter %nda_start_iter% ^
    --nda_a 0.7 --nda_b 0 --nda_c 0.15 --nda_d 0.15 --nda_e 0 --nda_c_quantile %nda_c_quantile% ^
    --nda_d_nfake %nfake_d% ^
    --path2badfake1 %fake_data_path_1% --path2badfake2 %fake_data_path_2% --path2badfake3 %fake_data_path_3% --path2badfake4 %fake_data_path_4% ^
    --comp_FID --samp_batch_size 500 ^ %*

    @REM --GAN_finetune --path_GAN_ckpt %gan_ckpt_path% ^
    @REM --dump_fake_for_NIQE --niqe_dump_path %niqe_dump_path%