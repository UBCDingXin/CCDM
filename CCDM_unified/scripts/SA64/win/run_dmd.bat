@echo off

@REM set CUDA_VISIBLE_DEVICES=0

set DATA_NAME="SteeringAngle"

@REM set ROOT_PATH=<YOUR_PATH>/CCDM/CCDM_unified
@REM set DATA_PATH=<YOUR_PATH>/datasets/%DATA_NAME%
set ROOT_PATH=D:/local_wd/CCDM_improved/CCDM_unified
set DATA_PATH=C:/Users/DX/BaiduSyncdisk/Baidu_WD/datasets/CCGM_or_regression/%DATA_NAME%
set TEACHER_PATH=D:/local_wd/CCDM_improved/CCDM_unified/output/SteeringAngle_64/Setup_CCDM/results

set SETTING="Setup_DMD2M"
set SIGMA=-1.0
set KAPPA=-1.0
set TYPE="hard"

python dmd.py ^
    --setting_name %SETTING% ^
    --root_path %ROOT_PATH% --data_name %DATA_NAME% --data_path %DATA_PATH% ^
    --image_size 64 --train_amp ^
    --min_label -80.0 --max_label 80.0 ^
    --y2h_embed_type "resnet" --y2cov_embed_type "resnet" --use_Hy ^
    --teacher_ckpt_path %TEACHER_PATH% --niters_t 50000 ^
    --model_channels 64 --cond_drop_prob 0  --channel_mult 1_2_2_4_8 ^
    --gen_network sngan --gene_ch 64 --disc_ch 64 ^
    --niters 200000 --resume_niter 0 --adv_loss_type hinge --train_timesteps 1000 ^
    --train_batch_size 128 --gradient_accumulate_every 1 ^
    --train_lr_generator 1e-4 --train_lr_guidance 1e-4 --num_D_steps 2 ^
    --kernel_sigma %SIGMA% --threshold_type %TYPE% --kappa %KAPPA% ^
    --weight_guidance_adv 2.0 --weight_generator_adv 0.2 ^
    --gan_DiffAugment ^
    --sample_every 10000 --save_every 10000 ^
    --samp_batch_size 200 --nfake_per_label 50 ^
    --dump_fake_data ^ %*