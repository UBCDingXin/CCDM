@echo off

set ROOT_PREFIX=D:/BaiduSyncdisk/Baidu_WD/CCGM/CcDDPM/Cell-200/Cell-200_64x64
set ROOT_PATH=%ROOT_PREFIX%/class-conditional/CFG
set DATA_PATH=D:/BaiduSyncdisk/Baidu_WD/datasets/CCGM_or_regression/Cell200
set EVAL_PATH=%ROOT_PREFIX%/evaluation
set NIQE_PATH=D:/LocalWD/CcGAN_TPAMI_NIQE/Cell-200/NIQE_64x64/fake_data

set SEED=2023
set NUM_WORKERS=0
set MIN_LABEL=1
set MAX_LABEL=200
set IMG_SIZE=64

set BATCH_SIZE=128
set GRAD_ACC=1
set LR=1e-4
set TIMESTEPS=1000
set SAMP_TIMESTEPS=250
@REM set SAMP_TIMESTEPS=100

set SETUP="Setup1"
set NITERS=20000
set RESUME_NITER=20000

set CUDA_VISIBLE_DEVICES=1
python main.py ^
    --root_path %ROOT_PATH% --data_path %DATA_PATH% --eval_ckpt_path %EVAL_PATH% --seed %SEED% --num_workers %NUM_WORKERS% --setting_name %SETUP% ^
    --min_label %MIN_LABEL% --max_label %MAX_LABEL% --img_size %IMG_SIZE% ^
    --niters %NITERS% --resume_niter %RESUME_NITER% --timesteps %TIMESTEPS% --sampling_timesteps %SAMP_TIMESTEPS% ^
    --train_batch_size %BATCH_SIZE% --train_lr %LR% --gradient_accumulate_every %GRAD_ACC% ^
    --train_amp ^
    --sample_every 2000 --save_every 5000 ^
    --comp_FID --nfake_per_label 1000 --samp_batch_size 500 ^
    --dump_fake_data ^ %*

    @REM --dump_fake_for_NIQE --niqe_dump_path %NIQE_PATH%