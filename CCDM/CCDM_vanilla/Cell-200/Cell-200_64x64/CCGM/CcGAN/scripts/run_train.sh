
ROOT_PREFIX="<YOUR_PATH>/CCDM/Cell-200/Cell-200_64x64"
ROOT_PATH="${ROOT_PREFIX}/CCGM/CcGAN"
DATA_PATH="<YOUR_PATH>/CCDM/datasets/Cell200"
EVAL_PATH="${ROOT_PREFIX}/evaluation"
dump_niqe_path="<YOUR_PATH>/CcGAN_TPAMI_NIQE/Cell-200/NIQE_64x64/fake_data"

SEED=2024
NUM_WORKERS=0
MIN_LABEL=1
MAX_LABEL=200
IMG_SIZE=64

BATCH_SIZE_D=32
BATCH_SIZE_G=512
NUM_D_STEPS=1
SIGMA=-1.0
KAPPA=-2.0
LR_G=1e-4
LR_D=1e-4

GAN_ARCH=DCGAN
LOSS_TYPE=vanilla

DIM_GAN=256
DIM_EMBED=128

NITERS=5000
resume_niter=0
python main.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --eval_ckpt_path $EVAL_PATH --seed $SEED --num_workers $NUM_WORKERS \
    --min_label $MIN_LABEL --max_label $MAX_LABEL --img_size $IMG_SIZE --transform \
    --GAN_arch $GAN_ARCH --niters_gan $NITERS --resume_niters_gan $resume_niter --loss_type_gan $LOSS_TYPE --num_D_steps $NUM_D_STEPS \
    --save_niters_freq 5000 --visualize_freq 1000 \
    --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G \
    --lr_g $LR_G --lr_d $LR_D --dim_gan $DIM_GAN --dim_embed $DIM_EMBED \
    --kernel_sigma $SIGMA --threshold_type soft --kappa $KAPPA \
    --comp_FID --FID_radius 0 --samp_batch_size 500 \

    @REM --dump_fake_for_NIQE --niqe_dump_path $dump_niqe_path