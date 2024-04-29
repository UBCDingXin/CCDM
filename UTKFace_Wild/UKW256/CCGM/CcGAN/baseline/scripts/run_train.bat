::===============================================================
:: This is a batch script for running the program on windows! 
::===============================================================

@echo off

set METHOD_NAME=CcGAN
set ROOT_PREFIX=D:/BaiduSyncdisk/Baidu_WD/CCGM/CcDDPM/UTKFace_Wild/UKW256
set ROOT_PATH=%ROOT_PREFIX%/CCGM/%METHOD_NAME%/baseline
set DATA_PATH=D:/BaiduSyncdisk/Baidu_WD/datasets/CCGM_or_regression/UTKFace
set EVAL_PATH=%ROOT_PREFIX%/evaluation/eval_models
set NIQE_PATH=C:/LocalWD/CcGAN_TPAMI_NIQE/UTKFace_Wild/NIQE_256x256/fake_data

set SEED=2024
set NUM_WORKERS=0
set MIN_LABEL=1
set MAX_LABEL=60
set IMG_SIZE=256
set MAX_N_IMG_PER_LABEL=100000
set MAX_N_IMG_PER_LABEL_AFTER_REPLICA=0

set BATCH_SIZE_G=32
set BATCH_SIZE_D=32
set NUM_D_STEPS=4
set SIGMA=-1.0
set KAPPA=-2.0
set LR_G=1e-4
set LR_D=1e-4
set NUM_ACC_D=4
set NUM_ACC_G=4

set GAN_ARCH="SAGAN"
set LOSS_TYPE="hinge"

set DIM_GAN=256
set DIM_EMBED=128

set NITERS=40000
set resume_niter=0
set SETTING=niters40K

set CUDA_VISIBLE_DEVICES=0

python main.py ^
    --setting_name %SETTING% ^
    --root_path %ROOT_PATH% --data_path %DATA_PATH% --eval_ckpt_path %EVAL_PATH% --seed %SEED% --num_workers %NUM_WORKERS% ^
    --min_label %MIN_LABEL% --max_label %MAX_LABEL% --img_size %IMG_SIZE% ^
    --max_num_img_per_label %MAX_N_IMG_PER_LABEL% --max_num_img_per_label_after_replica %MAX_N_IMG_PER_LABEL_AFTER_REPLICA% ^
    --GAN_arch %GAN_ARCH% --niters_gan %NITERS% --resume_niters_gan %resume_niter% --loss_type_gan %LOSS_TYPE% --num_D_steps %NUM_D_STEPS% ^
    --save_niters_freq 2500 --visualize_freq 1000 ^
    --batch_size_disc %BATCH_SIZE_D% --batch_size_gene %BATCH_SIZE_G% ^
    --num_grad_acc_d %NUM_ACC_D% --num_grad_acc_g %NUM_ACC_G% ^
    --lr_g %LR_G% --lr_d %LR_D% --dim_gan %DIM_GAN% --dim_embed %DIM_EMBED% ^
    --kernel_sigma %SIGMA% --threshold_type soft --kappa %KAPPA% ^
    --gan_DiffAugment --gan_DiffAugment_policy color,translation,cutout ^
    --comp_FID --samp_batch_size 100 --FID_radius 0 --nfake_per_label 1000 --dump_fake_for_NIQE --niqe_dump_path %NIQE_PATH% ^ %*

    @REM --dump_fake_for_NIQE --niqe_dump_path %NIQE_PATH%