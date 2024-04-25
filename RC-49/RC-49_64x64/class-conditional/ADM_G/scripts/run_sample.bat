@echo off

set ROOT_PATH=<YOUR_PATH>/CCDM/RC-49/RC-49_64x64/class-conditional/ADM_G
set DATA_PATH=<YOUR_PATH>/CCDM/datasets/RC-49
set EVAL_PATH=<YOUR_PATH>/CCDM/RC-49/RC-49_64x64/evaluation


set SETTING="Setup1"
set path_to_model=%ROOT_PATH%/output/exp_%SETTING%/diffusion/model050000.pt
set path_to_classifier=%ROOT_PATH%/output/exp_%SETTING%/classifier/model019999.pt

set dump_niqe_path=C:/LocalWD/CcGAN_TPAMI_NIQE/RC-49/NIQE_64x64

set CLASSIFIER_FLAGS=--classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True --classifier_scale 1.0 --classifier_use_fp16 True
set MODEL_FLAGS=--num_channels 128 --num_res_blocks 3 --learn_sigma True --class_cond True --use_fp16 True
set SAMPLE_FLAGS=--nfake_per_label 200 --samp_batch_size 200 --timestep_respacing ddim25 --use_ddim True
set EVAL_CONFIG=--eval_ckpt_path %EVAL_PATH% --dump_fake_data True --comp_FID True --niqe_dump_path %dump_niqe_path%

python classifier_sample.py ^
    --setup_name %SETTING% ^
    --root_dir %ROOT_PATH% --data_dir %DATA_PATH% --image_size 64 ^
    --model_path %path_to_model% ^
    --classifier_path %path_to_classifier% ^
    %CLASSIFIER_FLAGS% ^
    %MODEL_FLAGS% ^
    %SAMPLE_FLAGS% ^
    %EVAL_CONFIG% ^ %*

    @REM --dump_fake_for_NIQE True