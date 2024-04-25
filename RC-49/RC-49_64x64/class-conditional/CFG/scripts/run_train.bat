@echo off

set ROOT_PREFIX=<YOUR_PATH>/CCDM/RC-49/RC-49_64x64
set ROOT_PATH=%ROOT_PREFIX%/class-conditional/CFG
set DATA_PATH=<YOUR_PATH>/CCDM/datasets/RC-49
set EVAL_PATH=%ROOT_PREFIX%/evaluation

set SEED=2023
set MIN_LABEL=0
set MAX_LABEL=90.0
set IMG_SIZE=64
set MAX_N_IMG_PER_LABEL=25
set MAX_N_IMG_PER_LABEL_AFTER_REPLICA=0
set N_CLS=150

set BATCH_SIZE=128
set GRAD_ACC=1
set LR=1e-4
set TIMESTEPS=1000
set SAMP_TIMESTEPS=100

set SETUP="Setup1"
set NITERS=50000
set RESUME_NITER=0

python main.py ^
    --root_path %ROOT_PATH% --data_path %DATA_PATH% --eval_ckpt_path %EVAL_PATH% --seed %SEED% --setting_name %SETUP% ^
    --min_label %MIN_LABEL% --max_label %MAX_LABEL% --img_size %IMG_SIZE% --num_classes %N_CLS% ^
    --max_num_img_per_label %MAX_N_IMG_PER_LABEL% --max_num_img_per_label_after_replica %MAX_N_IMG_PER_LABEL_AFTER_REPLICA% ^
    --niters %NITERS% --resume_niter %RESUME_NITER% --timesteps %TIMESTEPS% --sampling_timesteps %SAMP_TIMESTEPS% ^
    --train_batch_size %BATCH_SIZE% --train_lr %LR% --train_amp --gradient_accumulate_every %GRAD_ACC% ^
    --sample_every 2000 --save_every 5000 ^ %*