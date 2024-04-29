
ROOT_PREFIX="<YOUR_PATH>/CCDM/UTKFace/UK192"
ROOT_PATH="${ROOT_PREFIX}/CCGM/CCDM"
DATA_PATH="<YOUR_PATH>/CCDM/datasets/UTKFace"
EVAL_PATH="${ROOT_PREFIX}/evaluation/eval_models"
NIQE_PATH="<YOUR_PATH>/CcGAN_TPAMI_NIQE/UTKFace/NIQE_192x192/fake_data"

SETTING="Setup1"
SIGMA=-1.0
KAPPA=-1.0
TYPE="hard"

python main.py \
    --setting_name $SETTING \
    --root_path $ROOT_PATH --data_path $DATA_PATH --eval_ckpt_path $EVAL_PATH \
    --image_size 192 \
    --pred_objective pred_x0 \
    --model_channels 32 --num_res_blocks 4 --num_groups 8 --cond_drop_prob 0.1 \
    --attention_resolutions 24_48_96 --channel_mult 1_2_4_4_8_8 \
    --niters 500000 --resume_niter 0 --train_lr 1e-5 --train_timesteps 1000 \
    --train_batch_size 48 --gradient_accumulate_every 2 --train_amp \
    --kernel_sigma $SIGMA --threshold_type $TYPE --kappa $KAPPA \
    --sample_every 20000 --save_every 10000 \
    --sample_timesteps 100 --sample_cond_scale 1.5 \
    --comp_FID --nfake_per_label 1000 --sampler ddim --samp_batch_size 200 \
    --dump_fake_data \
    2>&1 | tee output_${SETTING}.txt