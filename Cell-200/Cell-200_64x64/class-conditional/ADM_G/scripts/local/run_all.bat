@echo off

set ROOT_PATH="D:/BaiduSyncdisk/Baidu_WD/CCGM/CcDDPM/Cell-200/Cell-200_64x64/class-conditional/ADM_G"
set DATA_PATH="D:/BaiduSyncdisk/Baidu_WD/datasets/CCGM_or_regression/Cell200"
@REM set EVAL_PATH="D:/BaiduSyncdisk/Baidu_WD/CCGM/CcDDPM/Cell-200/Cell-200_64x64/evaluation"


set SETTING="Setup1"

set CLASSIFIER_FLAGS=--classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True --classifier_use_fp16 True

python classifier_train.py ^
    --setup_name %SETTING% ^
    --root_dir %ROOT_PATH% --data_dir %DATA_PATH% --image_size 64 ^
    --iterations 20000 --save_interval 5000 ^
    --batch_size 64 --lr 3e-4 --anneal_lr True --weight_decay 0.05 ^
    --log_interval 1000 ^
    %CLASSIFIER_FLAGS% ^ %*


set MODEL_FLAGS=--image_size 64 --attention_resolutions 32,16,8 --num_channels 128 --num_head_channels 64 --num_heads 4 --num_res_blocks 3 --learn_sigma True --class_cond True --resblock_updown True --use_new_attention_order True --use_scale_shift_norm True
set DIFFUSION_FLAGS=--diffusion_steps 1000 --noise_schedule cosine --rescale_learned_sigmas False --rescale_timesteps False
set TRAIN_FLAGS=--lr 1e-5 --weight_decay 1e-4 --batch_size 32 --use_fp16 True --log_interval 500 --lr_anneal_steps 20000 --save_interval 5000

python image_train.py ^
    --setup_name %SETTING% ^
    --root_dir %ROOT_PATH% --data_dir %DATA_PATH% ^
    %MODEL_FLAGS% ^
    %DIFFUSION_FLAGS% ^
    %TRAIN_FLAGS% ^ %*
