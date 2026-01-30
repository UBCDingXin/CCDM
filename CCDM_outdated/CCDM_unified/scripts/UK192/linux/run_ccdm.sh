#!/bin/bash

# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export CUDA_VISIBLE_DEVICES=0

DATA_NAME="UTKFace"
IMG_SIZE=192

ROOT_PATH="<YOUR_PATH>/CCDM/CCDM_unified"
DATA_PATH="<YOUR_PATH>/datasets/${DATA_NAME}"

SETTING="Setup_CCDM"
SIGMA=-1.0
KAPPA=-1.0
TYPE="hard"

python main.py \
    --setting_name $SETTING \
    --root_path $ROOT_PATH --data_name $DATA_NAME --data_path $DATA_PATH \
    --image_size $IMG_SIZE --train_amp \
    --min_label 1 --max_label 60 \
    --pred_objective "pred_x0" \
    --model_channels 64 --cond_drop_prob 0.1  --channel_mult 1_2_2_4_4_8_8 \
    --y2h_embed_type "resnet" --y2cov_embed_type "resnet" --use_Hy \
    --niters 300000 --resume_niter 0 --train_lr 1e-5 --train_timesteps 1000 \
    --train_batch_size 16 --gradient_accumulate_every 4 \
    --kernel_sigma $SIGMA --threshold_type $TYPE --kappa $KAPPA \
    --sample_every 25000 --save_every 25000 \
    --sample_timesteps 100 --sample_cond_scale 2.0 \
    --sampler ddim --samp_batch_size 100 --nfake_per_label 1000 \
    --dump_fake_data \
    2>&1 | tee output_${DATA_NAME}_${IMG_SIZE}_${SETTING}.txt
