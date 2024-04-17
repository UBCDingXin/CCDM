@echo off

set ROOT_PATH=D:/BaiduSyncdisk/Baidu_WD/CCGM/CcDDPM/RC-49/RC-49_64x64/class-conditional/ADM_G
set DATA_PATH=D:/BaiduSyncdisk/Baidu_WD/datasets/CCGM_or_regression/RC-49

set SETTING="Setup1"

set CLASSIFIER_FLAGS=--classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True --classifier_use_fp16 True

set CUDA_VISIBLE_DEVICES=0
python classifier_train.py ^
    --setup_name %SETTING% ^
    --root_dir %ROOT_PATH% --data_dir %DATA_PATH% --image_size 64 ^
    --iterations 20000 --save_interval 5000 ^
    --batch_size 128 --lr 3e-4 --anneal_lr True --weight_decay 0.05 ^
    --log_interval 1000 ^
    %CLASSIFIER_FLAGS% ^ %*

@REM MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
@REM python classifier_sample.py $MODEL_FLAGS --classifier_scale 1.0 --classifier_path models/64x64_classifier.pt --classifier_depth 4 --model_path models/64x64_diffusion.pt $SAMPLE_FLAGS