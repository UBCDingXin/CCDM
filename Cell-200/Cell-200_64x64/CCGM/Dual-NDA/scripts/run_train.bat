::===============================================================
:: This is a batch script for running the program on windows! 
::===============================================================

@echo off

set ROOT_PREFIX=<YOUR_PATH>/CCDM/Cell-200/Cell-200_64x64
set ROOT_PATH=%ROOT_PREFIX%/CCGM/Dual-NDA
set DATA_PATH=<YOUR_PATH>/CCDM/datasets/Cell200
set EVAL_PATH=%ROOT_PREFIX%/evaluation
set NIQE_PATH=<YOUR_PATH>/CcGAN_TPAMI_NIQE/Cell-200/NIQE_64x64/fake_data

set SEED=2024
set NUM_WORKERS=0
set MIN_LABEL=1
set MAX_LABEL=200
set IMG_SIZE=64

set BATCH_SIZE_D=32
set BATCH_SIZE_G=512
set NUM_D_STEPS=1
set SIGMA=-1.0
set KAPPA=-2.0
set LR_G=1e-4
set LR_D=1e-4

set GAN_ARCH=DCGAN
set LOSS_TYPE=vanilla

set DIM_GAN=256
set DIM_EMBED=128


set SETTING="Setup1"

set fake_data_path_1=%ROOT_PREFIX%/CCGM/CcGAN/output/CcGAN_DCGAN_soft_si0.077_ka2500.000_vanilla_nDs1_Dbs32_Gbs512/bad_fake_data/niters5K/badfake_NIQE0.9_nfake1000.h5

set nda_c_quantile=0.9
set nfake_d=-1
set nda_start_iter=5000

set NITERS=10000
set resume_niter=7500
python main.py ^
    --setting_name %SETTING% ^
    --root_path %ROOT_PATH% --data_path %DATA_PATH% --eval_ckpt_path %EVAL_PATH% --seed %SEED% --num_workers %NUM_WORKERS% ^
    --min_label %MIN_LABEL% --max_label %MAX_LABEL% --img_size %IMG_SIZE% --transform ^
    --GAN_arch %GAN_ARCH% --niters_gan %NITERS% --resume_niters_gan %resume_niter% --loss_type_gan %LOSS_TYPE% --num_D_steps %NUM_D_STEPS% ^
    --save_niters_freq 5000 --visualize_freq 1000 ^
    --batch_size_disc %BATCH_SIZE_D% --batch_size_gene %BATCH_SIZE_G% ^
    --lr_g %LR_G% --lr_d %LR_D% --dim_gan %DIM_GAN% --dim_embed %DIM_EMBED% ^
    --kernel_sigma %SIGMA% --threshold_type soft --kappa %KAPPA% ^
    --nda_start_iter %nda_start_iter% ^
    --nda_a 0.8 --nda_b 0 --nda_c 0.05 --nda_d 0.15 --nda_e 0 --nda_c_quantile %nda_c_quantile% ^
    --nda_d_nfake %nfake_d% ^
    --path2badfake1 %fake_data_path_1% --path2badfake2 None --path2badfake3 None --path2badfake4 None ^
    --comp_FID --FID_radius 0 ^ %*

    @REM --dump_fake_for_NIQE --niqe_dump_path %NIQE_PATH%