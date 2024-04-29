# CCDM
Code repository for Continuous Conditional Diffusion Model (CCDM)

<!-- ----------------------------------------------------------------->
## Software Requirements
Here, we provide a list of crucial software environments and python packages employed in the conducted experiments. Please note that we use different computational platforms for our experiments. <br />

**For computing NIQE scores and implementing the NIQE filtering (Support both Windows and Linux):**
| Item | Version |
|---|---|
| OS | Win11 |
| Python | 3.11.3 |
| Matlab | 2023a |

| Item | Version |
|---|---|
| OS | Linux |
| Python | 3.10.12 |
| Matlab | 2021b |


**For implementing CCDM (Support both Windows and Linux):**
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

| Item | Version |
|---|---|
| OS | Linux |
| Python | 3.10.12 |
| CUDA  | 12.1 |
| numpy | 1.26.4 |
| torch | 2.2.1 |
| torchvision | 0.17.1 |
| Pillow | 9.0.1 |
| accelearate | 0.27.2 |

**For implementing ReACGAN, ADCGAN, ADM-G, and CFG (Support both Windows):**
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

**For implementing CcGAN and Dual-NDA (Support both Linux):**
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
## Datasets

We use the preprocessed datasets provided by [Ding et. al. (2023)](https://github.com/UBCDingXin/improved_CcGAN).

### The RC-49 Dataset (h5 file)
Download the following h5 files and put them in `./datasets/RC-49`.
#### RC-49 (64x64)
[RC-49_64x64_download_link](https://1drv.ms/u/s!Arj2pETbYnWQstI0OuDMqpEZA80tRQ?e=fJJbWw) <br />

### The preprocessed UTKFace Dataset (h5 file)
Download the following h5 files and put them in `./datasets/UTKFace`.
#### UTKFace (64x64)
[UTKFace_64x64_download_link](https://1drv.ms/u/s!Arj2pETbYnWQstIzurW-LCFpGz5D7Q?e=X23ybx) <br />
#### UTKFace (128x128)
[UTKFace_128x128_download_link](https://1drv.ms/u/s!Arj2pETbYnWQstJGpTgNYrHE8DgDzA?e=d7AeZq) <br />
#### UTKFace (192x192)
[UTKFace_192x192_download_link](https://1drv.ms/u/s!Arj2pETbYnWQstY8hLN3lWEyX0lNLA?e=BcjUQh) <br />

### The Steering Angle dataset (h5 file)
Download the following h5 files and put them in `./datasets/SteeringAngle`.
#### Steering Angle (64x64)
[SteeringAngle_64x64_download_link](https://1drv.ms/u/s!Arj2pETbYnWQstIyDTDpGA0CNiONkA?e=Ui5kUK) <br />
#### Steering Angle (128x128)
[SteeringAngle_128x128_download_link](https://1drv.ms/u/s!Arj2pETbYnWQstJ0j7rXhDtm6y4IcA?e=bLQh2e) <br />

### The Cell-200 Dataset (h5 file)
Download the following h5 files and put them in `./datasets/Cell-200`.
#### Cell-200 (64x64)
[Cell-200_64x64_download_link](https://1drv.ms/u/s!Arj2pETbYnWQstIt73ZfGOAjBMiTmQ?e=cvxFIN) <br />

### The UTKFace-Wild Dataset (h5 file)
Download the following h5 files and put them in `./datasets/UTKFace-Wild`.
#### UTKFace-Wild (256x256)
[UTK256_download_link]() <br />

<!-- ----------------------------------------------------------------->
## Preparation

### RC-49
Download the following checkpoint. Unzip it and put it in "./CCDM/RC-49/RC-49_64x64" <br />
[RC-49 (64x64) Evaluation](https://1drv.ms/u/s!Arj2pETbYnWQvNINsmYzlBdkgSloTg?e=c7g3lU) <br />

### UTKFace
Download the following checkpoint. Unzip it and put it in "./CCDM/UTKFace/UK64" <br />
[UTKFace (64x64) Evaluation](https://1drv.ms/u/s!Arj2pETbYnWQvNIR8yAfixpzCDQbrQ?e=yGPEZl) <br />

Download the following checkpoint. Unzip it and put it in "./CCDM/UTKFace/UK128" <br />
[UTKFace (128x128) Evaluation](https://1drv.ms/u/s!Arj2pETbYnWQvNIX_mPvqxOHK8Ppyg?e=pM2ShQ) <br />

Download the following checkpoint. Unzip it and put it in "./CCDM/UTKFace/UK192" <br />
[UTKFace (192x192) Evaluation](https://1drv.ms/u/s!Arj2pETbYnWQvNIVcOryy6Rba6obIA?e=AoHB16) <br />

### Steering Angle
Download the following checkpoint. Unzip it and put it in "./CCDM/SteeringAngle/SA64" <br />
[Steering Angle (64x64) Evaluation](https://1drv.ms/u/s!Arj2pETbYnWQvNIa6KuqZN6G4c8hWA?e=qsfYGy) <br />

Download the following checkpoint. Unzip it and put it in "./CCDM/SteeringAngle/SA128" <br />
[Steering Angle (128x128) Evaluation](https://1drv.ms/u/s!Arj2pETbYnWQvNIYO7bd6tivU7VXBQ?e=E3bRGG) <br />

### Cell-200
Download the following checkpoint. Unzip it and put it in "./CCDM/Cell-200/Cell-200_64x64" <br />
[Steering Angle (64x64) Evaluation](https://1drv.ms/u/s!Arj2pETbYnWQvNIZflDROKCaI4f71w?e=ivxSSN) <br />

### UTKFace-Wild
Download the following checkpoint. Unzip it and put it in "./CCDM/UTKFace-Wild/UKW256" <br />
[UTKFace-Wild (256x256) Evaluation](https://1drv.ms/f/s!Arj2pETbYnWQvNIWtkNQnosw_UG35g?e=noRdLV) <br />


### Baidu Yun Link
You can also download the above checkpoints from [HERE](https://1drv.ms/f/s!Arj2pETbYnWQvNIWtkNQnosw_UG35g?e=noRdLV) <br />



<!-- --------------------------------------------------------------- -->
## Training
As illustrated in the aforementioned repository structure, distinct training codes have been provided for various datasets. <br />

