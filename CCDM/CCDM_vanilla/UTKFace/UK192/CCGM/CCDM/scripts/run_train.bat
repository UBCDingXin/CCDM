@echo off

set ROOT_PREFIX=<YOUR_PATH>/CCDM/UTKFace/UK192
set ROOT_PATH=%ROOT_PREFIX%/CCGM/CCDM
set DATA_PATH=<YOUR_PATH>/CCDM/datasets/UTKFace
set EVAL_PATH=%ROOT_PREFIX%/evaluation/eval_models
set NIQE_PATH=<YOUR_PATH>/CcGAN_TPAMI_NIQE/UTKFace/NIQE_192x192/fake_data

set SETTING="Setup1"
set SIGMA=-1.0
set KAPPA=-1.0

python main.py ^
    --setting_name %SETTING% ^
    --root_path %ROOT_PATH% --data_path %DATA_PATH% --eval_ckpt_path %EVAL_PATH% ^
    --image_size 192 ^
    --pred_objective pred_x0 ^
    --model_channels 32 --num_res_blocks 4 --num_groups 8 --cond_drop_prob 0.1 ^
    --attention_resolutions 24_48_96 --channel_mult 1_2_4_4_8_8 ^
    --niters 800000 --resume_niter 0 --train_lr 1e-4 --train_timesteps 1000 ^
    --train_batch_size 48 --gradient_accumulate_every 2 --train_amp ^
    --kernel_sigma %SIGMA% --threshold_type hard --kappa %KAPPA% ^
    --sample_every 20000 --save_every 10000 ^
    --sample_timesteps 100 --sample_cond_scale 2.0 ^
    --comp_FID --nfake_per_label 1000 --sampler ddim --samp_batch_size 100 --eval_batch_size 100 ^
    --dump_fake_data ^ %*
    @REM --dump_fake_for_NIQE --niqe_dump_path %NIQE_PATH%