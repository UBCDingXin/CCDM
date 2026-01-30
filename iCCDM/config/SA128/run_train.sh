#!/bin/bash

export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0

DATA_NAME="SteeringAngle"
IMG_SIZE=128

ROOT_PATH="<YOU_PATH>"
DATA_PATH="<YOU_PATH>/${DATA_NAME}"

SETTING="setup1"
SIGMA=-1.0
KAPPA=-2.0
TYPE="hard"

python main.py \
    --setting_name $SETTING \
    --root_path $ROOT_PATH --data_name $DATA_NAME --data_path $DATA_PATH \
    --num_channels 3 --image_size $IMG_SIZE \
    --min_label -80.0 --max_label 80.0 \
    --model_config "./config/model_cfg/unet_ccdm_128_v1.yaml" \
    --y2h_embed_type "resnet" \
    --use_y2cov --y2cov_hy_weight_train 0.01 --y2cov_hy_weight_test 0.1 --y2cov_embed_type "resnet" --net_embed_y2cov_y2emb "cnn" \
    --train_num_steps 400000 --resume_step 0 --train_lr 5e-5 \
    --train_batch_size 64 --gradient_accumulate_every 2 \
    --train_amp --train_mixed_precision fp16 \
    --kernel_sigma $SIGMA --threshold_type $TYPE --kappa $KAPPA \
    --use_ada_vic --ada_vic_type vanilla --min_n_per_vic 20 --use_symm_vic \
    --sample_every 10000 --save_every 20000 \
    --sampler sde --num_sample_steps 32 \
    --sample_cond_scale 1.5 --sample_cond_rescaled_phi 0.7 \
    --nfake_per_label 50 --samp_batch_size 200 \
    --dump_fake_data \
    --do_eval \
    2>&1 | tee output_${DATA_NAME}_${IMG_SIZE}_${SETTING}.txt

    # --dump_fake_for_niqe --niqe_dump_path "<YOU_PATH>/evaluation/NIQE/SteeringAngle/NIQE_128x128/fake_data/" \
