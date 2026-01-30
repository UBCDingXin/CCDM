@echo off

@REM set CUDA_VISIBLE_DEVICES=0

set DATA_NAME="UTKFace"

@set ROOT_PATH=<YOUR_PATH>/CCDM/CCDM_unified
set DATA_PATH=<YOUR_PATH>/datasets/%DATA_NAME%

set SETTING="Setup_CcDPM"
set SIGMA=-1.0
set KAPPA=-1.0
set TYPE="soft"

python main.py ^
    --setting_name %SETTING% ^
    --root_path %ROOT_PATH% --data_name %DATA_NAME% --data_path %DATA_PATH% ^
    --image_size 128 --train_amp ^
    --min_label 1 --max_label 60 --num_img_per_label_after_replica 200 ^
    --pred_objective pred_noise ^
    --model_channels 64 --cond_drop_prob 0.1  --channel_mult 1_2_4_4_8_8 ^
    --y2h_embed_type "resnet" ^
    --niters 200000 --resume_niter 0 --train_lr 1e-5 --train_timesteps 1000 ^
    --train_batch_size 32 --gradient_accumulate_every 2 ^
    --kernel_sigma %SIGMA% --threshold_type %TYPE% --kappa %KAPPA% ^
    --sample_every 25000 --save_every 25000 ^
    --sample_timesteps 150 --sample_cond_scale 2.0 ^
    --sampler ddim --samp_batch_size 200 --nfake_per_label 1000 ^
    --dump_fake_data ^ %*