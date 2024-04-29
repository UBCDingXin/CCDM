::===============================================================
:: This is a batch script for running the program on windows! 
::===============================================================

@echo off

set CUDA_VISIBLE_DEVICES=0

set METHOD_NAME=CcGAN
set ROOT_PREFIX=D:/BaiduSyncdisk/Baidu_WD/CCGM/CcDDPM/UTKFace_Wild/UKW256
set ROOT_PATH=%ROOT_PREFIX%/CCGM/%METHOD_NAME%/NDA
set DATA_PATH=D:/BaiduSyncdisk/Baidu_WD/datasets/CCGM_or_regression/UTKFace
set EVAL_PATH=%ROOT_PREFIX%/evaluation/eval_models
set NIQE_PATH=D:/LocalWD/CcGAN_TPAMI_NIQE/UTKFace_Wild/NIQE_256x256/fake_data

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


set SETTING="Setup2"

set fake_data_path_1=%ROOT_PREFIX%/CCGM/%METHOD_NAME%/baseline/output/SAGAN_soft_si0.037_ka900.000_hinge_nDs4_nDa4_nGa4_Dbs32_Gbs32/bad_fake_data/niters40K/badfake_NIQE0.9_nfake60000.h5
set fake_data_path_2=None
set fake_data_path_3=None
set fake_data_path_4=None

set nda_c_quantile=0.5
set nfake_d=-1
set nda_start_iter=40000

set NITERS=42500
set resume_niter=42500
python main.py ^
    --setting_name %SETTING% ^
    --root_path %ROOT_PATH% --data_path %DATA_PATH% --eval_ckpt_path %EVAL_PATH% --seed %SEED% --num_workers %NUM_WORKERS% ^
    --min_label %MIN_LABEL% --max_label %MAX_LABEL% --img_size %IMG_SIZE% ^
    --max_num_img_per_label %MAX_N_IMG_PER_LABEL% --max_num_img_per_label_after_replica %MAX_N_IMG_PER_LABEL_AFTER_REPLICA% ^
    --GAN_arch %GAN_ARCH% --niters_gan %NITERS% --resume_niters_gan %resume_niter% --loss_type_gan %LOSS_TYPE% --num_D_steps %NUM_D_STEPS% ^
    --save_niters_freq 5000 --visualize_freq 2500 ^
    --batch_size_disc %BATCH_SIZE_D% --batch_size_gene %BATCH_SIZE_G% ^
    --num_grad_acc_d %NUM_ACC_D% --num_grad_acc_g %NUM_ACC_G% ^
    --lr_g %LR_G% --lr_d %LR_D% --dim_gan %DIM_GAN% --dim_embed %DIM_EMBED% ^
    --kernel_sigma %SIGMA% --threshold_type soft --kappa %KAPPA% ^
    --gan_DiffAugment --gan_DiffAugment_policy color,translation,cutout ^
    --nda_start_iter %nda_start_iter% ^
    --nda_a 0.7 --nda_b 0 --nda_c 0.15 --nda_d 0.15 --nda_e 0 --nda_c_quantile %nda_c_quantile% ^
    --nda_d_nfake %nfake_d% ^
    --path2badfake1 %fake_data_path_1% --path2badfake2 %fake_data_path_2% --path2badfake3 %fake_data_path_3% --path2badfake4 %fake_data_path_4% ^
    --samp_batch_size 50 --eval_batch_size 50 ^
    --comp_FID --FID_radius 0 --nfake_per_label 1000 ^ %*


    @REM --dump_fake_for_NIQE --dump_fake_img_path %NIQE_PATH%