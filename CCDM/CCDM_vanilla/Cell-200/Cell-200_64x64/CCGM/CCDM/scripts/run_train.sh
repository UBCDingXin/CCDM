
ROOT_PREFIX="<YOUR_PATH>/CCDM/Cell-200/Cell-200_64x64"
ROOT_PATH="${ROOT_PREFIX}/CCGM/CCDM"
DATA_PATH="<YOUR_PATH>/CCDM/datasets/Cell200"
EVAL_PATH="${ROOT_PREFIX}/evaluation"
NIQE_PATH="<YOUR_PATH>/CcGAN_TPAMI_NIQE/Cell-200/NIQE_64x64/fake_data"

set SETTING="Setup1"
set SIGMA=-1.0
set KAPPA=-1.0

python main.py \
    --setting_name $SETTING \
    --root_path $ROOT_PATH --data_path $DATA_PATH --eval_ckpt_path $EVAL_PATH \
    --image_size 64 --train_amp \
    --pred_objective pred_x0 \
    --model_channels 32 --num_res_blocks 2 --num_groups 8 --cond_drop_prob 0.1 \
    --attention_resolutions 16_32 --channel_mult 1_2_2_4 \
    --niters 20000 --resume_niter 0 --train_lr 1e-4 --train_timesteps 1000 \
    --train_batch_size 128 --gradient_accumulate_every 1 \
    --kernel_sigma $SIGMA --threshold_type hard --kappa $KAPPA \
    --sample_every 1000 --save_every 2500 \
    --sample_timesteps 250 --sample_cond_scale 1.5 \
    --comp_FID --nfake_per_label 1000 --sampler ddim --samp_batch_size 250 \
    --dump_fake_data \
    
    @REM --dump_fake_for_NIQE --niqe_dump_path $NIQE_PATH