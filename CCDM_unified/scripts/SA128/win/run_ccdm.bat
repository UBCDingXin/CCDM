@echo off

@REM set CUDA_VISIBLE_DEVICES=0

set DATA_NAME="SteeringAngle"

@REM set ROOT_PATH=<YOUR_PATH>/CCDM/CCDM_unified
@REM set DATA_PATH=<YOUR_PATH>/datasets/%DATA_NAME%
set ROOT_PATH=D:/local_wd/CCDM_improved/CCDM_unified
set DATA_PATH=C:/Users/DX/BaiduSyncdisk/Baidu_WD/datasets/CCGM_or_regression/%DATA_NAME%

set SETTING="Setup_CCDM"
set SIGMA=-1.0
set KAPPA=-5.0
set TYPE="hard"

python main.py ^
    --setting_name %SETTING% ^
    --root_path %ROOT_PATH% --data_name %DATA_NAME% --data_path %DATA_PATH% ^
    --image_size 128 --train_amp ^
    --min_label -80.0 --max_label 80.0 ^
    --pred_objective pred_x0 ^
    --model_channels 64 --cond_drop_prob 0.1  --channel_mult 1_2_2_4_4_8 ^
    --y2h_embed_type "resnet" --y2cov_embed_type "resnet" --use_Hy ^
    --niters 200000 --resume_niter 0 --train_lr 5e-5 --train_timesteps 1000 ^
    --train_batch_size 32 --gradient_accumulate_every 2 ^
    --kernel_sigma %SIGMA% --threshold_type %TYPE% --kappa %KAPPA% ^
    --sample_every 10000 --save_every 25000 ^
    --sample_timesteps 150 --sample_cond_scale 1.5 ^
    --sampler ddim --samp_batch_size 200 --nfake_per_label 50 ^
    --dump_fake_data ^ %*