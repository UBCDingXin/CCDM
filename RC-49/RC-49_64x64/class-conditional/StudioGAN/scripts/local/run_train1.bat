::===============================================================
:: This is a batch script for running the program on windows! 
::===============================================================

@echo off

set ROOT_PREFIX=D:/BaiduSyncdisk/Baidu_WD/CCGM/CcDDPM/RC-49/RC-49_64x64
set DATA_PATH=D:/BaiduSyncdisk/Baidu_WD/datasets/CCGM_or_regression/RC-49
set EVAL_PATH=%ROOT_PREFIX%/evaluation
set CKPT_EVAL_FID_PATH=%EVAL_PATH%/ckpt_AE_epoch_200_seed_2020_CVMode_False.pth
set CKPT_EVAL_LS_PATH=%EVAL_PATH%/ckpt_PreCNNForEvalGANs_ResNet34_regre_epoch_200_seed_2020_CVMode_False.pth
set CKPT_EVAL_Div_PATH=%EVAL_PATH%/ckpt_PreCNNForEvalGANs_ResNet34_class_epoch_200_seed_2020_classify_49_chair_types_CVMode_False.pth

set GAN_NAME=ReACGAN
set CONFIG_PATH=./configs/%GAN_NAME%.yaml

wandb disabled
@REM wandb online

set CUDA_VISIBLE_DEVICES=0

@REM python main.py --train -metrics none ^
@REM     --data_dir %DATA_PATH% --cfg_file %CONFIG_PATH% --save_dir ./output/%GAN_NAME% ^
@REM     --seed 2023 --num_workers 8 ^
@REM     --print_freq 1000 --save_freq 5000 ^ %*


set CKPT_G=./output/%GAN_NAME%/checkpoints/RC49-ReACGAN-train-2024_01_05_23_23_57/model=G-current-weights-step=30000.pth
set CKPT_G_EMA=./output/%GAN_NAME%/checkpoints/RC49-ReACGAN-train-2024_01_05_23_23_57/model=G_ema-current-weights-step=30000.pth
set dump_niqe_path=D:/LocalWD/CcGAN_TPAMI_NIQE/RC-49/NIQE_64x64/fake_data

python main.py -metrics none ^
    --data_dir %DATA_PATH% --cfg_file %CONFIG_PATH% --save_dir ./output/%GAN_NAME% ^
    --seed 2023 --num_workers 8 ^
    --do_eval ^
    --path_to_G %CKPT_G% --path_to_G_ema %CKPT_G_EMA% ^
    --eval_ckpt_path_FID %CKPT_EVAL_FID_PATH% --eval_ckpt_path_LS %CKPT_EVAL_LS_PATH% --eval_ckpt_path_Div %CKPT_EVAL_Div_PATH% ^
    --samp_batch_size 200 --eval_batch_size 200 ^
    --dump_fake_for_NIQE --dump_fake_img_path %dump_niqe_path% ^ %*