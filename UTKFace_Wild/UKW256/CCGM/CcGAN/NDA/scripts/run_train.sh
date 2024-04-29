

METHOD_NAME="CcGAN"
ROOT_PREFIX="/scratch/dingx92/CcDDPM/UTKFace_Wild/UKW256"
ROOT_PATH="${ROOT_PREFIX}/CCGM/${METHOD_NAME}/NDA"
DATA_PATH="/project/6000538/dingx92/datasets/UTKFace/"
EVAL_PATH="${ROOT_PREFIX}/evaluation/eval_models"

SEED=2024
NUM_WORKERS=0
MIN_LABEL=1
MAX_LABEL=60
IMG_SIZE=256
MAX_N_IMG_PER_LABEL=100000
MAX_N_IMG_PER_LABEL_AFTER_REPLICA=0

BATCH_SIZE_G=32
BATCH_SIZE_D=32
NUM_D_STEPS=4
SIGMA=-1.0
KAPPA=-2.0
LR_G=1e-4
LR_D=1e-4
NUM_ACC_D=4
NUM_ACC_G=4

GAN_ARCH="SAGAN"
LOSS_TYPE="hinge"

DIM_GAN=256
DIM_EMBED=128

SETTING="Setup2"

fake_data_path_1="${ROOT_PREFIX}/CCGM/${METHOD_NAME}/baseline/output/SAGAN_soft_si0.037_ka900.000_hinge_nDs4_nDa4_nGa4_Dbs32_Gbs32/bad_fake_data/niters40K/badfake_NIQE0.9_nfake60000.h5"
fake_data_path_2="None"
fake_data_path_3="None"
fake_data_path_4="None"

nda_c_quantile=0.5
nfake_d=-1
nda_start_iter=40000

NITERS=45000
resume_niter=40000

export CUDA_VISIBLE_DEVICES=0
python main.py \
    --setting_name $SETTING \
    --root_path $ROOT_PATH --data_path $DATA_PATH --eval_ckpt_path $EVAL_PATH --seed $SEED --num_workers $NUM_WORKERS \
    --min_label $MIN_LABEL --max_label $MAX_LABEL --img_size $IMG_SIZE \
    --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA \
    --GAN_arch $GAN_ARCH --niters_gan $NITERS --resume_niters_gan $resume_niter --loss_type_gan $LOSS_TYPE --num_D_steps $NUM_D_STEPS \
    --save_niters_freq 2500 --visualize_freq 2500 \
    --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G \
    --num_grad_acc_d $NUM_ACC_D --num_grad_acc_g $NUM_ACC_G \
    --lr_g $LR_G --lr_d $LR_D --dim_gan $DIM_GAN --dim_embed $DIM_EMBED \
    --kernel_sigma $SIGMA --threshold_type soft --kappa $KAPPA \
    --gan_DiffAugment --gan_DiffAugment_policy color,translation,cutout \
    --nda_start_iter $nda_start_iter \
    --nda_a 0.7 --nda_b 0 --nda_c 0.15 --nda_d 0.15 --nda_e 0 --nda_c_quantile $nda_c_quantile \
    --nda_d_nfake $nfake_d \
    --path2badfake1 $fake_data_path_1 --path2badfake2 $fake_data_path_2 --path2badfake3 $fake_data_path_3 --path2badfake4 $fake_data_path_4 \
    --comp_FID --FID_radius 0 --nfake_per_label 1000 \
    2>&1 | tee output_${GAN_ARCH}_${NITERS}_${SETTING}.txt