@echo off

@REM set CUDA_VISIBLE_DEVICES=0

set DATA_NAME="SteeringAngle"

set ROOT_PATH=<YOUR_PATH>/CCDM/CCDM_unified
set DATA_PATH=<YOUR_PATH>/datasets/%DATA_NAME%

set SETTING="Setup_CcDPM"
set SIGMA=-1.0
set KAPPA=-5.0
set TYPE="soft"

python main.py ^
    --setting_name %SETTING% ^
    --root_path %ROOT_PATH% --data_name %DATA_NAME% --data_path %DATA_PATH% ^
    --image_size 64 --train_amp ^
    --min_label -80.0 --max_label 80.0 ^
    --pred_objective pred_noise ^
    --model_channels 64 --cond_drop_prob 0.1  --channel_mult 1_2_2_4_8 ^
    --y2h_embed_type "resnet" ^
    --niters 50000 --resume_niter 0 --train_lr 1e-4 --train_timesteps 1000 ^
    --train_batch_size 128 --gradient_accumulate_every 1 ^
    --kernel_sigma %SIGMA% --threshold_type %TYPE% --kappa %KAPPA% ^
    --sample_every 10000 --save_every 10000 ^
    --sample_timesteps 250 --sample_cond_scale 1.5 ^
    --sampler ddim --samp_batch_size 200 --nfake_per_label 50 ^
    --dump_fake_data ^ %*