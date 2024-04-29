@echo off

set ROOT_PREFIX=D:/BaiduSyncdisk/Baidu_WD/CCGM/CcDDPM/UTKFace_Wild/UKW256
set ROOT_PATH=%ROOT_PREFIX%/CCGM/CcDDPM
set DATA_PATH=D:/BaiduSyncdisk/Baidu_WD/datasets/CCGM_or_regression/UTKFace
set EVAL_PATH=%ROOT_PREFIX%/evaluation/eval_models

set SETTING="Setup5"
set SIGMA=-1.0
set KAPPA=-2.0

set NIQE_PATH=D:/LocalWD/CcGAN_TPAMI_NIQE/UTKFace_Wild/NIQE_256x256/fake_data

set CUDA_VISIBLE_DEVICES=1,0
python main.py ^
    --setting_name %SETTING% ^
    --root_path %ROOT_PATH% --data_path %DATA_PATH% --eval_ckpt_path %EVAL_PATH% ^
    --image_size 256 ^
    --pred_objective pred_x0 ^
    --model_channels 32 --num_res_blocks 4 --num_groups 8 --cond_drop_prob 0.1 ^
    --attention_resolutions 32_64_128 --channel_mult 1_2_4_4_8_8 ^
    --niters 600000 --resume_niter 600000 --train_lr 1e-5 --train_timesteps 1000 ^
    --train_batch_size 28 --gradient_accumulate_every 4 --train_amp ^
    --ema_update_after_step 100 --ema_update_every 100 --ema_decay 0.995 ^
    --kernel_sigma %SIGMA% --threshold_type hard --kappa %KAPPA% ^
    --sample_every 10000 --save_every 10000 ^
    --sample_timesteps 150 --sample_cond_scale 1.5 ^
    --comp_FID --nfake_per_label 1000 --sampler ddim --samp_batch_size 100 ^
    --dump_fake_data --dump_fake_for_NIQE --niqe_dump_path %NIQE_PATH% ^ %*
    @REM --dump_fake_for_NIQE --niqe_dump_path %NIQE_PATH%