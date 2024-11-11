# Enhancing Image Resolution in Solar Physics: A Latent Diffusion Model Approach
Official repository of the paper "Enhancing Image Resolution in Solar Physics: A Latent Diffusion Model Approach" by F.P. Ramunno et al 2024.

## Abstract
The spatial properties of the solar magnetic field are crucial to decoding the physical processes in the solar interior and their interplanetary effects. However, observations from older instruments, such as the Michelson Doppler Imager (MDI), have limited spatial or temporal resolution, which hinders the ability to study small-scale solar features in detail. Super resolving these older datasets is essential for uniform analysis across different solar cycles, enabling better characterization of solar flares, active regions, and magnetic network dynamics. In this work, we introduce a novel diffusion model-based approach for super-resolution, applied to MDI magnetograms to match the higher-resolution capabilities of the Helioseismic and Magnetic Imager (HMI). By training a Latent Diffusion Model (LDM) with residuals on downscaled HMI data and fine-tuning it with paired MDI/HMI data, we can enhance the resolution of MDI observations from 2"/pixel to 0.5"/pixel resolution. We evaluate the quality of the reconstructed images by means of classical metrics (e.g., PSNR, SSIM, FID and LPIPS) and we check if physical properties, such as the unsigned magnetic flux or the size of an active region, are preserved. We compare our model with different variations of LDM and Denoising Diffusion Probabilistic models (DDPMs), but also with two deterministic architectures already used in the past in performing this task. Furthermore, we show with an analysis in the Fourier domain that the LDM with residuals can resolve features smaller than 2", and due to the probabilistic nature of the LDM, we can asses their reliability, in contrast with the deterministic models. Future studies aim to super-resolve the temporal scale of the solar MDI instrument so that we can also have a better overview of the dynamics of the old events.

## Introduction
This repository contains the code for training and testing the "Enhancing Image Resolution in Solar Physics: A Latent Diffusion Model Approach" by F.P. Ramunno et al 2024.
It includes the following files:
- `training.py`: The main script for training the model.
- `diffusion.py`: Contains the DDPM class and utility functions.
- `modules.py`: Includes essential modules and classes used in the model.
- `physics_param.py`: Script to compute the physics metrics.
- `util.py`: Includes all the utility functions for the physics metrics computations.

## Training procedure
![](https://github.com/fpramunno/MAG2MAG/blob/main/algo_training.png)

## Inference procedure
![](https://github.com/fpramunno/MAG2MAG/blob/main/algo_inference.png)

## Example
![](https://github.com/fpramunno/MAG2MAG/blob/main/example.png)

## Latent diffusion model with residuals on MDI
![](https://github.com/fpramunno/MAG2MAG/blob/main/ldm_crop_zoomed.png)

### Prerequisites
List of libraries to run the project:

### Contact
Contact me at francesco.ramunno@fhnw.ch

### References
