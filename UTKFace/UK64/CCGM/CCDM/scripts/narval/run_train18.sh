#!/bin/bash
#SBATCH --account=def-wjwelch
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=1-00:00
#SBATCH --mail-user=dingx92@163.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=UKs_18
#SBATCH --output=%x-%j.out


module load StdEnv/2020 
module load gcc/9.3.0 cuda/11.4 python/3.9 opencv/4.5.5
virtualenv --no-download ~/ENV
source ~/ENV/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r ./requirements_narval.req


METHOD_NAME="CcDDPM_v0.3"
ROOT_PREFIX="/scratch/dingx92/CcDDPM/UTKFace/UK64"
ROOT_PATH="${ROOT_PREFIX}/CCGM/${METHOD_NAME}"
DATA_PATH="/project/6000538/dingx92/datasets/UTKFace"
EVAL_PATH="${ROOT_PREFIX}/evaluation/eval_models"

SETTING="Setup18" #ablation study test pred obj
SIGMA=-1.0
KAPPA=-1.0
TYPE="hard"

python main.py \
    --setting_name $SETTING \
    --root_path $ROOT_PATH --data_path $DATA_PATH --eval_ckpt_path $EVAL_PATH \
    --image_size 64 --train_amp \
    --pred_objective "pred_noise" \
    --model_channels 64 --num_res_blocks 2 --num_groups 8 --cond_drop_prob 0.1 \
    --attention_resolutions 16_32 --channel_mult 1_2_4_8 \
    --niters 50000 --resume_niter 0 --train_lr 1e-4 --train_timesteps 1000 \
    --train_batch_size 128 --gradient_accumulate_every 1 \
    --kernel_sigma $SIGMA --threshold_type $TYPE --kappa $KAPPA \
    --sample_every 10000 --save_every 10000 \
    --sample_timesteps 250 --sample_cond_scale 1.5 \
    --comp_FID --nfake_per_label 1000 --sampler ddim --samp_batch_size 250 \
    --dump_fake_data \
    2>&1 | tee output_${SETTING}.txt
    