::===============================================================
:: This is a batch script for running the program on windows 10! 
::===============================================================

@echo off

set ROOT_PATH="D:/BaiduSyncdisk/Baidu_WD/CCGM/CcDDPM/RC-49/RC-49_64x64/CCGM/CcGAN"
set DATA_PATH="D:/BaiduSyncdisk/Baidu_WD/datasets/CCGM_or_regression/RC-49"
set EVAL_PATH="D:/BaiduSyncdisk/Baidu_WD/CCGM/CcDDPM/RC-49/RC-49_64x64/evaluation"

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

set niqe_dump_path="D:/LocalWD/CcGAN_TPAMI_NIQE/RC-49/NIQE_64x64/fake_data"

set NITERS=30000
set Setting=niters%NITERS%
python main.py ^
    --setting_name %Setting% ^
    --root_path %ROOT_PATH% --data_path %DATA_PATH% --eval_ckpt_path %EVAL_PATH% --seed %SEED% --num_workers %NUM_WORKERS% ^
    --min_label %MIN_LABEL% --max_label %MAX_LABEL% --img_size %IMG_SIZE% ^
    --max_num_img_per_label %MAX_N_IMG_PER_LABEL% --max_num_img_per_label_after_replica %MAX_N_IMG_PER_LABEL_AFTER_REPLICA% ^
    --GAN_arch %GAN_ARCH% --niters_gan %NITERS% --resume_niters_gan 0 --loss_type_gan %LOSS_TYPE% --num_D_steps %NUM_D_STEPS% ^
    --save_niters_freq 5000 --visualize_freq 2000 ^
    --batch_size_disc %BATCH_SIZE_D% --batch_size_gene %BATCH_SIZE_G% ^
    --lr_g %LR_G% --lr_d %LR_D% --dim_gan %DIM_GAN% --dim_embed %DIM_EMBED% ^
    --num_grad_acc_d %NUM_ACC_D% --num_grad_acc_g %NUM_ACC_G% ^
    --kernel_sigma %SIGMA% --threshold_type soft --kappa %KAPPA% ^
    --gan_DiffAugment --gan_DiffAugment_policy color,translation,cutout ^
    --comp_FID --samp_batch_size 500 ^ %*

    @REM --dump_fake_for_NIQE --niqe_dump_path %niqe_dump_path%