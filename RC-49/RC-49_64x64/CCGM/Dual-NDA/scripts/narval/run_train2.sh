#!/bin/bash
#SBATCH --account=def-wjwelch
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=1-00:00
#SBATCH --mail-user=dingx92@163.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=RC_NDA
#SBATCH --output=%x-%j.out


module load StdEnv/2020 gcc/9.3.0 cuda/11.4 python/3.9 opencv/4.5.5
virtualenv --no-download ~/ENV
source ~/ENV/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r ./requirements_narval.req

ROOT_PREFIX="/scratch/dingx92/CcDDPM/RC-49/RC-49_64x64"
ROOT_PATH="${ROOT_PREFIX}/CCGM/Dual-NDA"
DATA_PATH="/project/6000538/dingx92/datasets/RC-49/"
EVAL_PATH="${ROOT_PREFIX}/evaluation"

SEED=2024
NUM_WORKERS=0
MIN_LABEL=0
MAX_LABEL=90
IMG_SIZE=64
MAX_N_IMG_PER_LABEL=25
MAX_N_IMG_PER_LABEL_AFTER_REPLICA=0

BATCH_SIZE_G=256
BATCH_SIZE_D=256
NUM_D_STEPS=2
SIGMA=-1.0
KAPPA=-2.0
LR_G=1e-4
LR_D=1e-4
NUM_ACC_D=1
NUM_ACC_G=1

GAN_ARCH="SAGAN"
LOSS_TYPE="hinge"
DIM_GAN=256
DIM_EMBED=128


SETTING="Setup2"

fake_data_path_1="${ROOT_PREFIX}/CCGM/CcGAN/output/CcGAN_SAGAN_soft_si0.047_ka50625.000_hinge_nDs2_nDa1_nGa1_Dbs256_Gbs256_DiAu/bad_fake_data/exp_niters30000/badfake_NIQE0.9_nfake18000.h5"
fake_data_path_2=None
fake_data_path_3=None
fake_data_path_4=None

# gan_ckpt_path="${ROOT_PREFIX}/CCGM/CcGAN/output/CcGAN_SAGAN_soft_si0.047_ka50625.000_hinge_nDs2_nDa1_nGa1_Dbs256_Gbs256_DiAu/saved_models/ckpts_in_train/ckpt_niter_30000.pth"
gan_ckpt_path="None"

nda_c_quantile=0.5
nfake_d=-1
nda_start_iter=0

NITERS=30000
resume_niter=0
python main.py \
    --setting_name $SETTING --root_path $ROOT_PATH --data_path $DATA_PATH --eval_ckpt_path $EVAL_PATH --seed $SEED --num_workers $NUM_WORKERS \
    --min_label $MIN_LABEL --max_label $MAX_LABEL --img_size $IMG_SIZE \
    --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA \
    --GAN_arch $GAN_ARCH --niters_gan $NITERS --resume_niters_gan $resume_niter --loss_type_gan $LOSS_TYPE --num_D_steps $NUM_D_STEPS \
    --save_niters_freq 5000 --visualize_freq 2000 \
    --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G \
    --num_grad_acc_d $NUM_ACC_D --num_grad_acc_g $NUM_ACC_G \
    --lr_g $LR_G --lr_d $LR_D --dim_gan $DIM_GAN --dim_embed $DIM_EMBED \
    --kernel_sigma $SIGMA --threshold_type soft --kappa $KAPPA \
    --gan_DiffAugment --gan_DiffAugment_policy color,translation,cutout \
    --nda_start_iter $nda_start_iter \
    --nda_a 0.7 --nda_b 0 --nda_c 0.15 --nda_d 0.15 --nda_e 0 --nda_c_quantile $nda_c_quantile \
    --nda_d_nfake $nfake_d \
    --path2badfake1 $fake_data_path_1 --path2badfake2 $fake_data_path_2 --path2badfake3 $fake_data_path_3 --path2badfake4 $fake_data_path_4 \
    --comp_FID --samp_batch_size 500 \
    2>&1 | tee output_${GAN_ARCH}_${NITERS}_${SETTING}.txt


    # --GAN_finetune --path_GAN_ckpt $gan_ckpt_path \