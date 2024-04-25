@echo off

set ROOT_PATH=<YOUR_PATH>/CCDM/RC-49/RC-49_64x64/class-conditional/ADM_G
set DATA_PATH=<YOUR_PATH>/CCDM/datasets/RC-49

set SETTING="Setup1"

set CLASSIFIER_FLAGS=--classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True --classifier_use_fp16 True

python classifier_train.py ^
    --setup_name %SETTING% ^
    --root_dir %ROOT_PATH% --data_dir %DATA_PATH% --image_size 64 ^
    --iterations 20000 --save_interval 5000 ^
    --batch_size 128 --lr 3e-4 --anneal_lr True --weight_decay 0.05 ^
    --log_interval 1000 ^
    %CLASSIFIER_FLAGS% ^ %*