# MDIRNET: Multi-Degradation Image Restoration Network


Official PyTorch implementation of **"MDIRNET: Multi-Degradation Image Restoration Network via Deep Unfolding"** submitted to IEEE Transactions on Instrumentation and Measurement.

---

## Overview

MDIRNET is a unified framework for restoring images affected by multiple degradations (noise, blur, rain). It combines model-driven optimization with deep learning through deep unfolding.

### Key Components

| Module | Description |
|--------|-------------|
| **PPM** (Patch Partitioning Module) | Learnable content-aware patch sampling. Allocates more patches to heavily degraded regions. |
| **DRAM** (Dynamic Rank Allocation Module) | Adaptive rank selection per patch group (r_max=32). Predicts optimal rank for efficient low-rank approximation. |
| **DU-OVPCA** (Deep Unfolded OVPCA) | Translates iterative OVPCA optimization into learnable network layers (K=6 layers). |
| **SAM** (Supervised Attention Module) | Refines output by comparing restored image with original input via attention maps. |

### Training Strategies

| Strategy | Description |
|----------|-------------|
| **One-by-One** | Separate models trained for each degradation type (denoising, deraining, deblurring) |
| **All-in-One** | Single model trained on all degradations together |

---

## Results

### Table II: Denoising Results on BSD68, Urban100, DIV2K, and SIDD

| Method | BSD68 (σ=25) | BSD68 (σ=50) | Urban100 (σ=25) | Urban100 (σ=50) | DIV2K (σ=25) | DIV2K (σ=50) | SIDD (Real) | Avg. Time |
|--------|--------------|--------------|-----------------|-----------------|--------------|--------------|-------------|-----------|
| CBM3D | 30.71/0.868 | 27.36/0.763 | 31.91/0.896 | 27.93/0.840 | 30.55/0.887 | 28.04/0.827 | 25.65/0.685 | 3.50s |
| DnCNN | 31.23/0.880 | 27.92/0.788 | 30.81/0.893 | 27.59/0.833 | 31.00/0.888 | 28.10/0.833 | 36.98/0.920 | 1.02s |
| IRCNN | 31.18/0.880 | 27.88/0.789 | 31.26/0.899 | 27.70/0.840 | 30.90/0.883 | 27.90/0.821 | 36.40/0.912 | 1.43s |
| FFDNet | 31.21/0.879 | 27.96/0.789 | 31.40/0.903 | 28.05/0.848 | 31.10/0.889 | 28.00/0.825 | 36.81/0.918 | 1.26s |
| BRDNet | 31.43/0.885 | 28.16/0.794 | 31.90/0.912 | 28.56/0.858 | 31.25/0.901 | 28.20/0.846 | 37.42/0.929 | 1.16s |
| AirNet | 31.48/0.893 | 28.23/0.806 | 32.10/0.924 | 28.88/0.870 | 31.45/0.908 | 28.40/0.852 | 38.84/0.951 | 1.58s |
| Restormer | 31.78/0.909 | 28.59/0.822 | 33.02/0.927 | 28.33/0.851 | 31.55/0.912 | 28.43/0.853 | 40.02/0.960 | 1.83s |
| DA-RCOT | 30.12/0.864 | 27.16/0.779 | 32.22/0.921 | 28.22/0.854 | 31.63/0.915 | 28.51/0.855 | 38.61/0.948 | 1.20s |
| Perceive-IR | 31.74/0.908 | 28.53/0.821 | 32.55/0.926 | 29.42/0.864 | 31.71/0.922 | 28.55/0.857 | 39.41/0.956 | 1.35s |
| **MDIRNET (Ours)** | **31.84/0.920** | **28.64/0.826** | **32.61/0.926** | **29.47/0.867** | **31.69/0.923** | **28.57/0.864** | **39.91/0.960** | **0.98s** |

### Table III: Deraining Results on Rain100L

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

### Table V: Deblurring Results on GoPro

| Method | PSNR (dB) | SSIM | Inference Time (s) |
|--------|-----------|------|-------------------|
| DBGAN | 31.10 | 0.942 | 0.32 |
| MTRNN | 31.15 | 0.945 | 0.29 |
| DMPHN | 31.20 | 0.940 | 0.28 |
| MPRNet | 32.66 | 0.959 | 0.24 |
| DA-RCOT | 31.41 | 0.936 | 0.27 |
| MWFormer | 31.92 | 0.951 | 0.31 |
| **MDIRNET (Ours)** | **32.68** | **0.959** | **0.23** |

### Table IV: One-by-One vs All-in-One Training Strategies

| Strategy | Denoising (BSD68) | Deraining (Rain100L) | Deblurring (GoPro) | Params (M) | GFLOPs | Avg. PSNR |
|----------|-------------------|----------------------|-------------------|------------|--------|-----------|
| One-by-One | 31.84 dB | 36.53 dB | 32.67 dB | 0.68 × 3 | 14.0 × 3 | 33.68 dB |
| All-in-One | 31.56 dB | 35.21 dB | 31.34 dB | 0.68 | 14.0 | 32.70 dB |

### Table VI: Mixed Degradation Results

| Degradation Type | PSNR (dB) | SSIM | Time (ms) |
|------------------|-----------|------|-----------|
| Rain + Noise | 29.34 | 0.892 | 185.23 |
| Blur + Noise | 30.12 | 0.901 | 184.89 |
| Rain + Blur | 28.76 | 0.883 | 186.12 |
| Rain + Blur + Noise | 27.45 | 0.865 | 187.34 |

### Table I: Per-module Parameter and FLOP Breakdown

| Module | Architecture | Params (M) | GFLOPs |
|--------|--------------|------------|--------|
| PPM | 3 Conv (3→96→96→2, 3×3) | 0.60 | 11.8 |
| DRAM | Conv(3→32) + FC(32→32→1) | 0.01 | 0.01 |
| DU-OVPCA | 6 unfolded layers (G, Ψ, ω; r≤32) | 0.02 | 0.24 |
| SAM | 3 Conv (3→64, 3→64, 128→1, 3×3) | 0.05 | 1.95 |
| **Total** | | **0.68** | **14.0** |

### Figure 4: Denoising Results on BSD68 and SIDD

![Denoising Results]

*Figure 4: Denoising results. Rows 1-3: synthetic Gaussian noise on BSD68. Row 4: real sensor noise from SIDD dataset.*

### Figure 5: Deraining Results on Rain100L

![Deraining Results]

*Figure 5: Deraining results on Rain100L dataset. From left to right: Rainy input, MDIRNET output, Ground truth.*

### Figure 6: Deblurring Results on GoPro

![Deblurring Results]

*Figure 6: Deblurring results on GoPro dataset. From left to right: Blurry input, MDIRNET output, Ground truth.*
