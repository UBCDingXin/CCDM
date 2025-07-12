#!/bin/bash

# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export CUDA_VISIBLE_DEVICES=0

DATA_NAME="RC-49"
IMG_SIZE=64

ROOT_PATH="<YOUR_PATH>/CCDM/CCDM_unified"
DATA_PATH="<YOUR_PATH>/datasets/${DATA_NAME}"

SETTING="Setup_CCDM"
SIGMA=-1.0
KAPPA=-2.0
TYPE="hard"

python main.py \
    --setting_name $SETTING \
    --root_path $ROOT_PATH --data_name $DATA_NAME --data_path $DATA_PATH \
    --image_size $IMG_SIZE --train_amp \
    --min_label 0 --max_label 90.0 --max_num_img_per_label 25 \
    --pred_objective "pred_x0" \
    --model_channels 64 --cond_drop_prob 0.1  --channel_mult 1_2_2_4_8 \
    --y2h_embed_type "resnet" --y2cov_embed_type "resnet" --use_Hy \
    --niters 50000 --resume_niter 0 --train_lr 1e-4 --train_timesteps 1000 \
    --train_batch_size 128 --gradient_accumulate_every 1 \
    --kernel_sigma $SIGMA --threshold_type $TYPE --kappa $KAPPA \
    --sample_every 10000 --save_every 10000 \
    --sample_timesteps 250 --sample_cond_scale 1.5 \
    --sampler ddim --samp_batch_size 200 --nfake_per_label 200 \
    --dump_fake_data \
    2>&1 | tee output_${DATA_NAME}_${IMG_SIZE}_${SETTING}.txt
