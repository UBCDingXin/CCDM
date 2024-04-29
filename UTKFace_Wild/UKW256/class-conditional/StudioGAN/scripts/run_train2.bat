::===============================================================
:: This is a batch script for running the program on windows!
::===============================================================

@echo off

set ROOT_PREFIX=<YOUR_PATH>/CCDM/UTKFace_Wild/UKW256
set DATA_PATH=<YOUR_PATH>/CCDM/datasets/UTKFace
set EVAL_PATH=%ROOT_PREFIX%/evaluation/eval_models
set CKPT_EVAL_FID_PATH=%EVAL_PATH%/ckpt_AE_epoch_200_seed_2024_CVMode_False.pth
set CKPT_EVAL_LS_PATH=%EVAL_PATH%/ckpt_PreCNNForEvalGANs_ResNet34_regre_epoch_200_seed_2024_CVMode_False.pth
set CKPT_EVAL_Div_PATH=%EVAL_PATH%/ckpt_PreCNNForEvalGANs_ResNet34_class_epoch_200_seed_2024_classify_5_races_CVMode_False.pth
set dump_niqe_path=<YOUR_PATH>/CcGAN_TPAMI_NIQE/UTKFace_Wild/NIQE_256x256/fake_data

set GAN_NAME=ADCGAN
set CONFIG_PATH=./configs/%GAN_NAME%.yaml

wandb disabled
@REM wandb online

python main.py --train -metrics none ^
    --data_dir %DATA_PATH% --cfg_file %CONFIG_PATH% --save_dir ./output/%GAN_NAME% ^
    --seed 2024 --num_workers 8 ^
    -sync_bn -mpc ^
    --print_freq 1000 --save_freq 5000 ^ %*


@REM @REM @REM Remember to change the correct path for G ckpt
@REM set CKPT_G=./output/%GAN_NAME%/checkpoints/UKW256-%GAN_NAME%-train-2023_08_07_07_36_55/model=G-current-weights-step=20000.pth
@REM set CKPT_G_EMA=./output/%GAN_NAME%/checkpoints/UKW256-%GAN_NAME%-train-2023_08_07_07_36_55/model=G_ema-current-weights-step=20000.pth

@REM python main.py -metrics none ^
@REM     --data_dir %DATA_PATH% --cfg_file %CONFIG_PATH% --save_dir ./output/%GAN_NAME% ^
@REM     --seed 2024 --num_workers 8 ^
@REM     -sync_bn -mpc ^
@REM     --do_eval ^
@REM     --path_to_G %CKPT_G% --path_to_G_ema %CKPT_G_EMA% ^
@REM     --eval_ckpt_path_FID %CKPT_EVAL_FID_PATH% --eval_ckpt_path_LS %CKPT_EVAL_LS_PATH% --eval_ckpt_path_Div %CKPT_EVAL_Div_PATH% ^
@REM     --samp_batch_size 100 --eval_batch_size 100 ^
@REM     --dump_fake_for_NIQE --dump_fake_img_path %dump_niqe_path% ^ %*
