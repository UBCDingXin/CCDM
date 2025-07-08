#!/bin/bash

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0

DATA_NAME="RC-49"
IMG_SIZE=64

ROOT_PATH="<YOUR_PATH>/CCDM/CCDM_unified"
DATA_PATH="<YOUR_PATH>/datasets/${DATA_NAME}"
TEACHER_PATH="${ROOT_PATH}/output/RC-49_64/Setup_CCDM/results"

SETTING="Setup_DMD2M"
SIGMA=-1.0
KAPPA=0
TYPE="hard"

python dmd.py \
    --setting_name $SETTING \
    --root_path $ROOT_PATH --data_name $DATA_NAME --data_path $DATA_PATH \
    --image_size $IMG_SIZE --train_amp \
    --min_label 0 --max_label 90.0 --max_num_img_per_label 25 \
    --y2h_embed_type "resnet" --y2cov_embed_type "resnet" --use_Hy \
    --teacher_ckpt_path $TEACHER_PATH --niters_t 50000 \
    --model_channels 64 --cond_drop_prob 0  --channel_mult 1_2_2_4_8 \
    --gen_network sngan --gene_ch 84 --disc_ch 64 \
    --niters 50000 --resume_niter 0 --adv_loss_type hinge --train_timesteps 1000 \
    --train_batch_size 128 --gradient_accumulate_every 1 \
    --train_lr_generator 1e-4 --train_lr_guidance 1e-4 --num_D_steps 2 \
    --kernel_sigma $SIGMA --threshold_type $TYPE --kappa $KAPPA \
    --weight_guidance_adv 10.0 --weight_generator_adv 1.0 \
    --gan_DiffAugment \
    --sample_every 5000 --save_every 10000 \
    --samp_batch_size 200 --nfake_per_label 200 \
    --dump_fake_data \
    2>&1 | tee output_${DATA_NAME}_${IMG_SIZE}_${SETTING}.txt