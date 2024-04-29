# CCDM
Code repository for Continuous Conditional Diffusion Model (CCDM)

<!-- ----------------------------------------------------------------->
## Software Requirements
Here, we provide a list of crucial software environments and python packages employed in the conducted experiments. Please note that we use different computational platforms for our experiments. <br />

**For computing NIQE scores and implementing the NIQE filtering:**
| Item | Version |
|---|---|
| OS | Win11 |
| Python | 3.11.3 |
| Matlab | 2023a |

**For implementing CCDM (Win):**
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

**For implementing CCDM (Linux):**
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

**For implementing ReACGAN, ADCGAN, ADM-G, and CFG:**
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

**For implementing CcGAN and Dual-NDA:**
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
Download the following h5 files and put them in `./datasets/UTKFace`.

### The preprocessed UTKFace Dataset (h5 file)
Download the following h5 files and put them in `./datasets/UTKFace`.
#### UTKFace (64x64)
[UTKFace_64x64_download_link](https://1drv.ms/u/s!Arj2pETbYnWQstIzurW-LCFpGz5D7Q?e=X23ybx) <br />
#### UTKFace (128x128)
[UTKFace_128x128_download_link](https://1drv.ms/u/s!Arj2pETbYnWQstJGpTgNYrHE8DgDzA?e=d7AeZq) <br />

### The Steering Angle dataset (h5 file)
Download the following h5 files and put them in `./datasets/SteeringAngle`.
#### Steering Angle (64x64)
[SteeringAngle_64x64_download_link](https://1drv.ms/u/s!Arj2pETbYnWQstIyDTDpGA0CNiONkA?e=Ui5kUK) <br />
#### Steering Angle (128x128)
[SteeringAngle_128x128_download_link](https://1drv.ms/u/s!Arj2pETbYnWQstJ0j7rXhDtm6y4IcA?e=bLQh2e) <br />

### The Cell-200 Dataset (h5 file)


### The UTKFace-Wild Dataset (h5 file)

<!-- ----------------------------------------------------------------->
### Preparation


<!-- --------------------------------------------------------------- -->
## Training
As illustrated in the aforementioned repository structure, distinct training codes have been provided for various datasets. <br />

