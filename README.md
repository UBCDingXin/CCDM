# CCDM
Code repository for Continuous Conditional Diffusion Model (CCDM)

<!-- --------------------------------------------------------------- -->
--------------------------------------------------------
## 1. Repository Structure and Preparation
```
└── CCDM
    ├── UTKFace
    │   ├── UTKFace_64x64
    │   │   ├──CcGAN
    |   |   |  ├──baseline
    |   |   |  └──NDA
    │   │   ├──class-conditional_GAN
    |   |   |  └──StudioGAN
    |   |   ├──diffusion_models
    |   |   |  ├──ADM_G
    |   |   |  └──classifier_free_guidance
    │   │   └──evaluation
    |   |      └──eval_models
    │   └── UTKFace_128x128
    │       ├──CcGAN
    |       |  ├──baseline
    |       |  └──NDA
    │       ├──class-conditional_GAN
    |       |  └──StudioGAN
    |       ├──diffusion_models
    |       |  ├──ADM_G
    |       |  └──classifier_free_guidance
    │       └──evaluation
    |          └──eval_models
    ├── UTKFace
    │   ├── UTKFace_64x64
    │   │   ├──CcGAN
    |   |   |  ├──baseline
    |   |   |  └──NDA
    │   │   ├──class-conditional_GAN
    |   |   |  └──StudioGAN
    |   |   ├──diffusion_models
    |   |   |  ├──ADM_G
    |   |   |  └──classifier_free_guidance
    │   │   └──evaluation
    |   |      └──eval_models
    │   └── UTKFace_128x128
    │       ├──CcGAN
    |       |  ├──baseline
    |       |  └──NDA
    │       ├──class-conditional_GAN
    |       |  └──StudioGAN
    |       ├──diffusion_models
    |       |  ├──ADM_G
    |       |  └──classifier_free_guidance
    │       └──evaluation
    |          └──eval_models
    ├── SteeringAngle
    │   ├── SteeringAngle_64x64
    │   │   ├──CcGAN
    |   |   |  ├──baseline
    |   |   |  └──NDA
    │   │   ├──class-conditional_GAN
    |   |   |  └──StudioGAN
    |   |   ├──diffusion_models
    |   |   |  ├──ADM_G
    |   |   |  └──classifier_free_guidance
    │   │   └──evaluation
    |   |      └──eval_models
    │   └── SteeringAngle_128x128
    │       ├──CcGAN
    |       |  ├──baseline
    |       |  └──NDA
    │       ├──class-conditional_GAN
    |       |  └──StudioGAN
    |       ├──diffusion_models
    |       |  ├──ADM_G
    |       |  └──classifier_free_guidance
    │       └──evaluation
    |          └──eval_models
    ├── UTKFace
    │   ├── UTKFace_64x64
    │   │   ├──CcGAN
    |   |   |  ├──baseline
    |   |   |  └──NDA
    │   │   ├──class-conditional_GAN
    |   |   |  └──StudioGAN
    |   |   ├──diffusion_models
    |   |   |  ├──ADM_G
    |   |   |  └──classifier_free_guidance
    │   │   └──evaluation
    |   |      └──eval_models
    │   └── UTKFace_128x128
    │       ├──CcGAN
    |       |  ├──baseline
    |       |  └──NDA
    │       ├──class-conditional_GAN
    |       |  └──StudioGAN
    |       ├──diffusion_models
    |       |  ├──ADM_G
    |       |  └──classifier_free_guidance
    │       └──evaluation
    |          └──eval_models
    ├── UTKFace
    │   ├── UTKFace_64x64
    │   │   ├──CcGAN
    |   |   |  ├──baseline
    |   |   |  └──NDA
    │   │   ├──class-conditional_GAN
    |   |   |  └──StudioGAN
    |   |   ├──diffusion_models
    |   |   |  ├──ADM_G
    |   |   |  └──classifier_free_guidance
    │   │   └──evaluation
    |   |      └──eval_models
    │   └── UTKFace_128x128
    │       ├──CcGAN
    |       |  ├──baseline
    |       |  └──NDA
    │       ├──class-conditional_GAN
    |       |  └──StudioGAN
    |       ├──diffusion_models
    |       |  ├──ADM_G
    |       |  └──classifier_free_guidance
    │       └──evaluation
    |          └──eval_models
    └── NIQE
        ├── UTKFace
        │   ├── NIQE_64x64
        │   ├── NIQE_128x128
        │   ├── NIQE_filter_64x64
        │   └── NIQE_filter_128x128
        └── SteeringAngle
           ├── NIQE_64x64
           ├── NIQE_128x128     
           ├── NIQE_filter_64x64
           └── NIQE_filter_128x128
```



<!-- --------------------------------------------------------------- -->
## 2. Software Requirements
Here, we provide a list of crucial software environments and python packages employed in the conducted experiments. Please note that we use different computational platforms for our experiments. <br />

**For computing NIQE scores and implementing the NIQE filtering:**
| Item | Version |
|---|---|
| OS | Win11 |
| Python | 3.11.3 |
| Matlab | 2023a |

**For training CcGAN:**
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
| matplotlib | 3.4.2 |

**For training ReACGAN, ADCGAN, ADM-G, and CFG:**
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
| matplotlib | 3.7.1 |

<!-- --------------------------------------------------------------- -->
## 3. Datasets

We use the preprocessed datasets provided by [Ding et. al. (2023)](https://github.com/UBCDingXin/improved_CcGAN).

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


!------------------------------------>
### 4. Preparation


<!-- --------------------------------------------------------------- -->
## 5. Training
As illustrated in the aforementioned repository structure, distinct training codes have been provided for various datasets. <br />

<