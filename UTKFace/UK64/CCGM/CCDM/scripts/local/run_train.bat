@echo off

set ROOT_PREFIX=C:/BaiduSyncdisk/Baidu_WD/CCGM/CcDDPM/UTKFace/UK64
set ROOT_PATH=%ROOT_PREFIX%/CCGM/CcDDPM_v0.3
set DATA_PATH=C:/BaiduSyncdisk/Baidu_WD/datasets/CCGM_or_regression/UTKFace
set EVAL_PATH=%ROOT_PREFIX%/evaluation/eval_models

set SETTING="Setup1"
set SIGMA=-1.0
set KAPPA=-1.0

set NIQE_PATH=F:/LocalWD/CcGAN_TPAMI_NIQE/UTKFace/NIQE_64x64/fake_data

set CUDA_VISIBLE_DEVICES=0
python main.py ^
    --setting_name %SETTING% ^
    --root_path %ROOT_PATH% --data_path %DATA_PATH% --eval_ckpt_path %EVAL_PATH% ^
    --image_size 64 --train_amp ^
    --pred_objective pred_x0 ^
    --model_channels 64 --num_res_blocks 2 --num_groups 8 --cond_drop_prob 0.5 ^
    --attention_resolutions 16_32 --channel_mult 1_2_4_8 ^
    --niters 50000 --resume_niter 50000 --train_lr 1e-4 --train_timesteps 1000 ^
    --train_batch_size 64 --gradient_accumulate_every 1 ^
    --kernel_sigma %SIGMA% --threshold_type hard --kappa %KAPPA% ^
    --sample_every 2500 --save_every 10000 ^
    --sample_timesteps 250 --sample_cond_scale 6.0 ^
    --comp_FID --nfake_per_label 1000 --sampler ddim --samp_batch_size 250 ^
    --dump_fake_data ^ %*
    
    @REM --dump_fake_for_NIQE --niqe_dump_path %NIQE_PATH%