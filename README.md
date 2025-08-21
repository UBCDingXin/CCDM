# The Code Repository for "[CCDM: Continuous Conditional Diffusion Models for Image Generation](https://arxiv.org/abs/2405.03546)"

**[UPDATE! 2025-08-18]** CCDM has been accepted by IEEE Transactions on Multimedia. <br />

**[UPDATE! 2025-07-13]** We corrected a coding error that inadvertently allowed CCDM and CcDPM to utilize additional training samples on RC-49, though fortunately our primary conclusion about CCDM's substantial superiority remains unchanged. <br />

**[UPDATE! 2025-02-26]** We offer a unified code repository located at `./CCDM/CCDM_unified`, which supports training CcDPM, CCDM, and DMD2-M on RC-49, Cell-200, UTKFace, and Steering Angle. The original code repository, containing the initial version of CCDM, is now archived in `./CCDM/CCDM_vanilla`. Detailed training and sampling setups are documented in [`./setup_details.pdf`](https://github.com/UBCDingXin/CCDM/blob/main/setup_details.pdf).  <br />

--------------------------------------------------------

This repository provides the source codes for the experiments in our papers for CCDMs. <br />
If you use this code, please cite
```text
@misc{ding2024ccdm,
      title={{CCDM}: Continuous Conditional Diffusion Models for Image Generation}, 
      author={Xin Ding and Yongwei Wang and Kao Zhang and Z. Jane Wang},
      year={2024},
      eprint={2405.03546},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!-- ----------------------------------------------------------------->
--------------------------------------------------------
## Some Illustrative Figures

<p align="center">
  <img src="images/illustration_CCGM.png">
  Illustration of the CCGM task with sample images from the UTKFace and Steering Angle datasets.
</p>

<p align="center">
  <img src="images/overall_workflow.png">
  The overall workflow of CCDMs
</p>




<!-- ----------------------------------------------------------------->
--------------------------------------------------------
## Software Requirements
Here, we provide a list of crucial software environments and python packages employed in the conducted experiments. Please note that we use different computational platforms for our experiments. <br />

**For computing NIQE scores (Support both Windows and Linux):**
| Item | Version | Item | Version |
|---|---|---|---|
| OS | Win11 | OS | Linux |
| Python | 3.11.3 | Python | 3.10.12 |
| Matlab | 2023a | Matlab | 2021b |

**For implementing CCDM (Support both Windows and Linux):**
| Item | Version | Item | Version |
|---|---| ---|---|
| OS | Win11 | OS | Linux |
| Python | 3.11.3 | Python | 3.10.12 |
| CUDA  | 11.8 | CUDA  | 12.1 |
| numpy | 1.23.5 | numpy | 1.26.4 |
| torch | 2.0.1 | torch | 2.2.1 |
| torchvision | 0.15.2 | torchvision | 0.17.1 |
| Pillow | 9.5.0 | Pillow | 9.0.1 |
| accelearate | 0.20.3 | accelearate | 0.27.2 |

**For implementing ReACGAN, ADCGAN, ADM-G, and CFG (Support Windows):**
| Item | Version |
|---|---|
| OS | Win11 |
| Python | 3.11.3 |
| CUDA  | 11.8 |
| numpy | 1.23.5 |
| torch | 2.0.1 |
| torchvision | 0.15.2 |
| Pillow | 9.5.0 |
| accelearate | 0.20.3 |
| wandb | 0.15.7 |

**For implementing CcGAN and Dual-NDA (Support Linux):**
| Item | Version |
|---|---|
| OS | Linux |
| Python | 3.9 |
| CUDA  | 11.4 |
| numpy | 1.23.0 |
| torch | 1.12.1 |
| torchvision | 0.13.1 |
| Pillow | 8.4.0 |
| accelearate | 0.18.0 |


<!-- --------------------------------------------------------------- -->
--------------------------------------------------------
## Datasets

We use the preprocessed datasets provided by [Ding et. al. (2023)](https://github.com/UBCDingXin/improved_CcGAN).

### The RC-49 Dataset (h5 file)
Download the following h5 file and put it in `./datasets/RC-49`.
#### RC-49 (64x64)
[RC-49_64x64_OneDrive_link](https://1drv.ms/u/s!Arj2pETbYnWQstI0OuDMqpEZA80tRQ?e=fJJbWw) <br />
[RC-49_64x64_BaiduYun_link](https://pan.baidu.com/s/1Odd02zraZI0XuqIj5UyOAw?pwd=bzjf) <br />

### The preprocessed UTKFace Dataset (h5 file)
Download the following h5 files and put them in `./datasets/UTKFace`.
#### UTKFace (64x64)
[UTKFace_64x64_Onedrive_link](https://1drv.ms/u/s!Arj2pETbYnWQstIzurW-LCFpGz5D7Q?e=X23ybx) <br />
[UTKFace_64x64_BaiduYun_link](https://pan.baidu.com/s/1fYjxmD3tJG6QKw5jjXxqIg?pwd=ocmi) <br />
#### UTKFace (128x128)
[UTKFace_128x128_OneDrive_link](https://1drv.ms/u/s!Arj2pETbYnWQstJGpTgNYrHE8DgDzA?e=d7AeZq) <br />
[UTKFace_128x128_BaiduYun_link](https://pan.baidu.com/s/17Br49DYS4lcRFzktfSCOyA?pwd=iary) <br />
#### UTKFace (192x192)
[UTKFace_192x192_OneDrive_link](https://1drv.ms/u/s!Arj2pETbYnWQstY8hLN3lWEyX0lNLA?e=BcjUQh) <br />
[UTKFace_192x192_BaiduYun_link](https://pan.baidu.com/s/1KaT_k21GTdLqqJxUi24f-Q?pwd=4yf1) <br />

### The Steering Angle dataset (h5 file)
Download the following h5 files and put them in `./datasets/SteeringAngle`.
#### Steering Angle (64x64)
[SteeringAngle_64x64_OneDrive_link](https://1drv.ms/u/s!Arj2pETbYnWQstIyDTDpGA0CNiONkA?e=Ui5kUK) <br />
[SteeringAngle_64x64_BaiduYun_link](https://pan.baidu.com/s/1ekpMJLC0mE08zVJp5GpFHQ?pwd=xucg) <br />
#### Steering Angle (128x128)
[SteeringAngle_128x128_OneDrive_link](https://1drv.ms/u/s!Arj2pETbYnWQstJ0j7rXhDtm6y4IcA?e=bLQh2e) <br />
[SteeringAngle_128x128_BaiduYun_link](https://pan.baidu.com/s/1JVBccsr5vgsIdzC-uskx-A?pwd=4z5n) <br />

### The Cell-200 Dataset (h5 file)
Download the following h5 file and put it in `./datasets/Cell-200`.
#### Cell-200 (64x64)
[Cell-200_64x64_OneDrive_link](https://1drv.ms/u/s!Arj2pETbYnWQstIt73ZfGOAjBMiTmQ?e=cvxFIN) <br />
[Cell-200_64x64_BaiduYun_link](https://pan.baidu.com/s/1wkXUT6XUfLpKZ_D9fAg__w?pwd=v2r1) <br />

<!-- --------------------------------------------------------------- -->
--------------------------------------------------------
## Preparation (Required!)
Please download the zip file from either [OneDrive](https://1drv.ms/u/s!Arj2pETbYnWQvOQFAot2lzSWwOEgSQ?e=ZokUe5) or [BaiduYun](https://pan.baidu.com/s/1KfyLWXSpaYClRSnXV0hjrQ?pwd=bdwp) and extract its contents into the `./CCDM` directory. The zip archive contains the required checkpoints for the ILI's embedding networks and covariance embedding networks, as well as the corresponding checkpoints for the evaluation models associated with each individual experiment.

<!-- --------------------------------------------------------------- -->
--------------------------------------------------------
## Training
Following [Ding et. al. (2023)](https://github.com/UBCDingXin/improved_CcGAN) and [Ding et. al. (2024)](https://github.com/UBCDingXin/Dual-NDA), distinct training codes have been provided for various datasets. <br />

*For simplicity, we only show how to implement the proposed **CCDM** in each experiment.* <br />

Before running the experiment, navigate to the `./CCDM/CCDM_unifed` directory in your terminal. <br />

### (1) RC-49 (64x64)
To execute the training process, run the script `./scripts/RC64/win/run_ccdm.bat` on Windows or `./scripts/RC64/linux/run_ccdm.sh` on Linux. Make sure to correctly configure the following path: `YOUR_PATH`.

Additionally, we provide training scripts for CcDPM, named `run_ccdpm.bat`(Windows) and `run_ccdpm.sh` (Linux), as well as scripts for DMD2-M, named `run_dmd.bat` (Windows) and `run_dmd.sh` (Linux).

### (2) UTKFace (64x64)
To execute the training process, run the script `./scripts/UK64/win/run_ccdm.bat` on Windows or `./scripts/UK64/linux/run_ccdm.sh` on Linux. Make sure to correctly configure the following path: `YOUR_PATH`.

Additionally, we provide training scripts for CcDPM, named `run_ccdpm.bat`(Windows) and `run_ccdpm.sh` (Linux), as well as scripts for DMD2-M, named `run_dmd.bat` (Windows) and `run_dmd.sh` (Linux).

### (3) UTKFace (128x128)
To execute the training process, run the script `./scripts/UK128/win/run_ccdm.bat` on Windows or `./scripts/UK128/linux/run_ccdm.sh` on Linux. Make sure to correctly configure the following path: `YOUR_PATH`.

Additionally, we provide training scripts for CcDPM, named `run_ccdpm.bat`(Windows) and `run_ccdpm.sh` (Linux), as well as scripts for DMD2-M, named `run_dmd.bat` (Windows) and `run_dmd.sh` (Linux).

### (4) UTKFace (192x192)
To execute the training process, run the script `./scripts/UK192/win/run_ccdm.bat` on Windows or `./scripts/UK192/linux/run_ccdm.sh` on Linux. Make sure to correctly configure the following path: `YOUR_PATH`.

Additionally, we provide training scripts for CcDPM, named `run_ccdpm.bat`(Windows) and `run_ccdpm.sh` (Linux), as well as scripts for DMD2-M, named `run_dmd.bat` (Windows) and `run_dmd.sh` (Linux).

### (5) Steering Angle (64x64)
To execute the training process, run the script `./scripts/SA64/win/run_ccdm.bat` on Windows or `./scripts/SA64/linux/run_ccdm.sh` on Linux. Make sure to correctly configure the following path: `YOUR_PATH`.

Additionally, we provide training scripts for CcDPM, named `run_ccdpm.bat`(Windows) and `run_ccdpm.sh` (Linux), as well as scripts for DMD2-M, named `run_dmd.bat` (Windows) and `run_dmd.sh` (Linux).

### (6) Steering Angle (128x128)
To execute the training process, run the script `./scripts/SA128/win/run_ccdm.bat` on Windows or `./scripts/SA128/linux/run_ccdm.sh` on Linux. Make sure to correctly configure the following path: `YOUR_PATH`.

Additionally, we provide training scripts for CcDPM, named `run_ccdpm.bat`(Windows) and `run_ccdpm.sh` (Linux), as well as scripts for DMD2-M, named `run_dmd.bat` (Windows) and `run_dmd.sh` (Linux).

### (7) Cell-200 (64x64)
To execute the training process, run the script `./scripts/Cell/win/run_ccdm.bat` on Windows or `./scripts/Cell/linux/run_ccdm.sh` on Linux. Ensure that the following paths are correctly configured: `ROOT_PATH`, `DATA_PATH`, `EVAL_PATH`, and `NIQE_PATH`.

Additionally, we provide training scripts for CcDPM, named `run_ccdpm.bat`(Windows) and `run_ccdpm.sh` (Linux), as well as scripts for DMD2-M, named `run_dmd.bat` (Windows) and `run_dmd.sh` (Linux).

### (8) Finetuning Stable Diffusion v1.5

We fine-tuned [Stable Diffusion v1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) using Hugging Face's official checkpoint on both RC-49 and UTKFace. During fine-tuning, all training images were resized to 512×512 to match the model's native resolution. 

For RC-49, we trained the model for 5 epochs with a batch size of 4 and learning rate 1e-5, subsequently resizing generated images to 64×64 for evaluation. On UTKFace (192×192), we conducted 6 epochs of training with identical batch size and learning rate, then evaluated generated images at three resolutions: 64×64, 128×128, and 192×192. 

<!-- --------------------------------------------------------------- -->
--------------------------------------------------------
## Sampling and Evaluation

After the training, the sampling usually automatically starts. The evaluation setups are consistent with [Ding et. al. (2023)](https://github.com/UBCDingXin/improved_CcGAN).

<!------------------------------------>
### (1) SFID, Diversity, and Label Score

Navigate to the `./CCDM/evaluation/other_metrics/<DATA_NAME>/<metrics_??x??>` directory in your terminal, replacing `<DATA_NAME>` with the name of the corresponding dataset and `<metrics_??x??>` with the appropriate resolution. Execute `run_eval.bat` (Windows) or `run_eval.sh` (Linux) to begin the evaluation process. For each script, ensure the following paths are correctly configured: `ROOT_PATH`, `DATA_PATH`, `FAKE_DATA_PATH`, and `NIQE_PATH`.

<!------------------------------------>
### (2) NIQE

In the bash scripts for training each method, enable the flags `--dump_fake_for_NIQE --niqe_dump_path <YOUR_NIQE_PATH>` to dump fake images for computing NIQE scores. Ensure that `<YOUR_NIQE_PATH>` is set correctly. Fake images for NIQE computation are typically stored in `./NIQE/<DATA_NAME>/NIQE_??x??/fake_data`, where `<DATA_NAME>` should be replaced with the corresponding dataset name and `<metrics_??x??>` with the appropriate resolution. To compute the average NIQE scores, execute the batch script `run_test.bat` (Windows) or `run_test.sh` (Linux). <br />


<!-- --------------------------------------------------------------- -->
--------------------------------------------------------
## Acknowledge
- https://github.com/lucidrains/denoising-diffusion-pytorch
- https://github.com/lucidrains/classifier-free-guidance-pytorch
- https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
- https://github.com/openai/guided-diffusion