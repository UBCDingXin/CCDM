@echo off

set ROOT_PATH="<YOUR_PATH>/CCDM/UTKFace_Wild/UKW256/class-conditional/ADM_G"
set DATA_PATH="<YOUR_PATH>/CCDM/datasets/UTKFace"
set EVAL_PATH="<YOUR_PATH>/CCDM/UTKFace_Wild/UKW256/evaluation/eval_models"
set dump_niqe_path=<YOUR_PATH>/CcGAN_TPAMI_NIQE/UTKFace_Wild/NIQE_256x256/fake_data

set SETTING=Setup1
set path_to_model=%ROOT_PATH%/output/exp_%SETTING%/diffusion/model050000.pt
set path_to_classifier=%ROOT_PATH%/output/exp_%SETTING%/classifier/model019999.pt

set CLASSIFIER_FLAGS=--classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True --classifier_scale 1.0 --classifier_use_fp16 True
set MODEL_FLAGS=--attention_resolutions 64,32,16 --num_channels 64 --num_head_channels 64 --num_heads 4 --num_res_blocks 3 --learn_sigma True --class_cond True --resblock_updown True --use_new_attention_order True --use_scale_shift_norm True --use_fp16 True
set SAMPLE_FLAGS=--nfake_per_label 1000 --samp_batch_size 25 --timestep_respacing ddim25 --use_ddim True
set EVAL_CONFIG=--eval_ckpt_path %EVAL_PATH% --dump_fake_data True --comp_FID True --dump_fake_for_NIQE True --niqe_dump_path %dump_niqe_path%
@REM --dump_fake_for_NIQE True --niqe_dump_path %dump_niqe_path%

python classifier_sample.py ^
    --setup_name %SETTING% ^
    --root_dir %ROOT_PATH% --data_dir %DATA_PATH% --image_size 256 ^
    --model_path %path_to_model% ^
    --classifier_path %path_to_classifier% ^
    %CLASSIFIER_FLAGS% ^
    %MODEL_FLAGS% ^
    %SAMPLE_FLAGS% ^
    %EVAL_CONFIG% ^ %*

    