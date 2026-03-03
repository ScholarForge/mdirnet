# MDIRNET: Multi-Degradation Image Restoration Network

Official  implementation of **"MDIRNET: Multi-Degradation Image Restoration Network via Deep Unfolding"**
---

## 📋 Overview

MDIRNET is a unified framework for restoring images affected by multiple degradations (noise, blur, rain). It combines model-driven optimization with deep learning through deep unfolding.

### Key Components

| Module | Description |
|--------|-------------|
| **PPM** (Patch Partitioning Module) | Learnable content-aware patch sampling. Allocates more patches to heavily degraded regions. |
| **DRAM** (Dynamic Rank Allocation Module) | Predicts optimal rank per patch group for efficient low-rank approximation. |
| **DU-OVPCA** (Deep Unfolded OVPCA) | Translates iterative OVPCA optimization into learnable network layers. |
| **SAM** (Supervised Attention Module) | Refines output by comparing restored image with original input via attention maps. |

### Training Strategies

| Strategy | Description |
|----------|-------------|
| **One-by-One** | Separate models trained for each degradation type (denoising, deraining, deblurring) |
| **All-in-One** | Single model trained on all degradations together |

---

## 📊 Results

### Table 1: Denoising Results on BSD68, Urban100, and DIV2K

| Method | BSD68 (σ=25) | BSD68 (σ=50) | Urban100 (σ=25) | Urban100 (σ=50) | DIV2K (σ=25) | DIV2K (σ=50) | Avg. Time |
|--------|--------------|--------------|-----------------|-----------------|--------------|--------------|-----------|
| CBM3D | 30.71/0.868 | 27.36/0.763 | 31.91/0.896 | 27.93/0.840 | 30.55/0.872 | 28.04/0.827 | 3.50s |
| DnCNN | 31.23/0.880 | 27.92/0.788 | 30.81/0.893 | 27.59/0.833 | 31.00/0.888 | 28.10/0.833 | 1.02s |
| IRCNN | 31.18/0.880 | 27.88/0.789 | 31.26/0.899 | 27.70/0.840 | 30.90/0.883 | 27.90/0.821 | 1.43s |
| FFDNet | 31.21/0.879 | 27.96/0.789 | 31.40/0.903 | 28.05/0.848 | 31.10/0.889 | 28.00/0.825 | 1.26s |
| BRDNet | 31.43/0.885 | 28.16/0.794 | 31.90/0.912 | 28.56/0.858 | 31.25/0.901 | 28.20/0.846 | 1.16s |
| AirNet | 31.48/0.893 | 28.23/0.806 | 32.10/0.924 | 28.88/0.870 | 31.45/0.908 | 28.40/0.852 | 1.58s |
| Restormer | 31.78/0.909 | 28.59/0.822 | 33.02/0.927 | 28.33/0.851 | 31.55/0.912 | 28.43/0.853 | 1.83s |
| DA-RCOT | 30.12/0.864 | 27.16/0.779 | 32.22/0.921 | 28.22/0.854 | 31.63/0.915 | 28.51/0.855 | 1.20s |
| Perceive-IR | 31.74/0.908 | 28.53/0.821 | 32.55/0.926 | 29.42/0.864 | 31.71/0.922 | 28.55/0.857 | 1.35s |
| **MDIRNET (Ours)** | **31.82/0.918** | **28.62/0.825** | **32.63/0.928** | **29.48/0.868** | **31.68/0.922** | **28.56/0.863** | **0.98s** |

### Table 2: Deraining Results on Rain100L

| Method | PSNR (dB) | SSIM | Inference Time (s) |
|--------|-----------|------|-------------------|
| UMRL | 27.79 | 0.892 | 0.24 |
| SIRR | 32.39 | 0.917 | 0.44 |
| MSPFN | 32.37 | 0.916 | 0.46 |
| LPNet | 33.50 | 0.923 | 0.33 |
| AirNet | 34.90 | 0.966 | 0.34 |
| MFFDNet | 35.61 | 0.941 | 0.33 |
| DeAirRTUR | 38.04 | 0.983 | 0.34 |
| **MDIRNET (Ours)** | **36.53** | **0.974** | **0.23** |

### Table 3: Deblurring Results on GoPro

| Method | PSNR (dB) | SSIM | Inference Time (s) |
|--------|-----------|------|-------------------|
| DBGAN | 31.10 | 0.942 | 0.32 |
| MTRNN | 31.15 | 0.945 | 0.29 |
| DMPHN | 31.20 | 0.940 | 0.28 |
| MPRNet | 32.66 | 0.959 | 0.24 |
| DA-RCOT | 31.41 | 0.936 | 0.27 |
| MWFormer | 31.92 | 0.951 | 0.31 |
| **MDIRNET (Ours)** | **32.68** | **0.959** | **0.23** |

### Table 4: One-by-One vs All-in-One Training Strategies

| Strategy | Denoising (BSD68) | Deraining (Rain100L) | Deblurring (GoPro) | Average |
|----------|-------------------|----------------------|-------------------|---------|
| **One-by-One** | 31.82 dB | 36.53 dB | 32.68 dB | 33.68 dB |
| **All-in-One** | 31.54 dB | 35.22 dB | 31.35 dB | 32.70 dB |
