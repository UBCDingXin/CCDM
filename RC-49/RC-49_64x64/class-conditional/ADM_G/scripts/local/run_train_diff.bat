@echo off

set ROOT_PATH=D:/BaiduSyncdisk/Baidu_WD/CCGM/CcDDPM/RC-49/RC-49_64x64/class-conditional/ADM_G
set DATA_PATH=D:/BaiduSyncdisk/Baidu_WD/datasets/CCGM_or_regression/RC-49

set SETTING="Setup1"

@REM @REM Some configs refer to https://github.com/openai/improved-diffusion; Class-conditional ImageNet-64 model (270M parameters, trained for 250K iterations)
set MODEL_FLAGS=--image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True --class_cond True
set DIFFUSION_FLAGS=--diffusion_steps 4000 --noise_schedule cosine --rescale_learned_sigmas False --rescale_timesteps False
set TRAIN_FLAGS=--lr 1e-4 --batch_size 32 --use_fp16 True --log_interval 1000 --lr_anneal_steps 50000 --save_interval 5000 
@REM decay lr from 3e-4 to 1e-4; num_channels from 192 to 128 to avoid OOM

set CUDA_VISIBLE_DEVICES=0
python image_train.py ^
    --setup_name %SETTING% ^
    --root_dir %ROOT_PATH% --data_dir %DATA_PATH% ^
    %MODEL_FLAGS% ^
    %DIFFUSION_FLAGS% ^
    %TRAIN_FLAGS% ^ %*
    
    @REM --resume_checkpoint .\output\exp_%SETTING%\diffusion\model020000.pt ^