@echo off

set ROOT_PREFIX=<YOUR_PATH>/CCDM/UTKFace_Wild/UKW256
set ROOT_PATH=%ROOT_PREFIX%/class-conditional/CFG
set DATA_PATH=<YOUR_PATH>/CCDM/datasets/UTKFace
set EVAL_PATH=%ROOT_PREFIX%/evaluation/eval_models
set dump_niqe_path=<YOUR_PATH>/CcGAN_TPAMI_NIQE/UTKFace_Wild/NIQE_256x256/fake_data

set SEED=2023
set NUM_WORKERS=0
set MIN_LABEL=1
set MAX_LABEL=60
set IMG_SIZE=256
set MAX_N_IMG_PER_LABEL=99999
set MAX_N_IMG_PER_LABEL_AFTER_REPLICA=0

set BATCH_SIZE=32
set GRAD_ACC=2
set LR=1e-5
set TIMESTEPS=1000
set SAMP_TIMESTEPS=250

set SETUP="Setup1"
set NITERS=50000
set RESUME_NITER=0

python main.py ^
    --root_path %ROOT_PATH% --data_path %DATA_PATH% --eval_ckpt_path %EVAL_PATH% --seed %SEED% --num_workers %NUM_WORKERS% --setting_name %SETUP% ^
    --min_label %MIN_LABEL% --max_label %MAX_LABEL% --img_size %IMG_SIZE% ^
    --max_num_img_per_label %MAX_N_IMG_PER_LABEL% --max_num_img_per_label_after_replica %MAX_N_IMG_PER_LABEL_AFTER_REPLICA% ^
    --niters %NITERS% --resume_niter %RESUME_NITER% --timesteps %TIMESTEPS% --sampling_timesteps %SAMP_TIMESTEPS% ^
    --train_batch_size %BATCH_SIZE% --train_lr %LR% --gradient_accumulate_every %GRAD_ACC% ^
    --train_amp ^
    --sample_every 50000 --save_every 10000 ^
    --comp_FID --nfake_per_label 1000 --samp_batch_size 100 ^
    --dump_fake_data --dump_fake_for_NIQE --niqe_dump_path %dump_niqe_path% ^ %*