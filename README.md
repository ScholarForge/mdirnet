# MDIRNET: Multi-Degradation Image Restoration Network via Deep Unfolding

Official PyTorch implementation of **"MDIRNET: Multi-Degradation Image Restoration Network via Deep Unfolding"**.

## Overview

MDIRNET is a unified framework for restoring images affected by multiple degradations (noise, blur, rain). It combines model-driven optimization with deep learning through deep unfolding of Orthogonal Variational PCA (OVPCA).


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



### Architecture

| Module | Architecture | Params | GFLOPs | Runtime (ms) |
|--------|-------------|--------|--------|-------------|
| PPM | 3 Conv (3→96→96→2, 3×3) | 0.60M | 11.8 | 92 |
| DRAM | Conv(3→32) + FC(32→32→1) | 0.01M | 0.01 | 6 |
| DU-OVPCA | 6 unfolded layers (G, Ψ, ω̂; r≤32) | 0.02M | 0.24 | 74 |
| SAM | 3 Conv (3→64, 3→64, 128→1, 3×3) | 0.05M | 1.95 | 24 |
| **Total** | | **0.68M** | **14.0** | **196** |


## Results

### Denoising (BSD68, Urban100, DIV2K, SIDD)

| Method | BSD68 σ=25 | BSD68 σ=50 | Urban100 σ=25 | Urban100 σ=50 | DIV2K σ=25 | DIV2K σ=50 | SIDD | Inf. Time (s) |
|--------|-----------|-----------|--------------|--------------|-----------|-----------|------|-------------|
| DnCNN | 31.23/0.880 | 27.92/0.788 | 30.81/0.893 | 27.59/0.833 | 31.00/0.888 | 28.10/0.833 | 36.98/0.920 | 0.20 |
| AirNet | 31.48/0.893 | 28.23/0.806 | 32.10/0.924 | 28.88/0.870 | 31.45/0.908 | 28.40/0.852 | 38.84/0.951 | 0.33 |
| Restormer | 31.78/0.909 | 28.59/0.822 | 33.02/0.927 | 28.33/0.851 | 31.55/0.912 | 28.43/0.853 | 40.02/0.960 | 0.33 |
| Perceive-IR | 31.74/0.908 | 28.53/0.821 | 32.55/0.926 | 29.42/0.864 | 31.71/0.922 | 28.55/0.857 | 39.41/0.956 | 0.26 |
| **MDIRNET** | **31.84/0.920** | **28.64/0.826** | **32.61/0.926** | **29.47/0.867** | **31.69/0.923** | **28.57/0.864** | **39.91/0.960** | **0.19** |

### Deraining (Rain100L)

| Method | PSNR/SSIM | Inf. Time (s) |
|--------|-----------|-------------|
| AirNet | 34.90/0.966 | 0.34 |
| MFFDNet | 35.61/0.941 | 0.33 |
| DeAirRTUR | 38.04/0.983 | 0.34 |
| **MDIRNET** | **36.53/0.974** | **0.23** |

### Deblurring (GoPro)

| Method | PSNR/SSIM | Inf. Time (s) |
|--------|-----------|-------------|
| MPRNet | 32.66/0.959 | 0.24 |
| DA-RCOT | 31.41/0.936 | 0.27 |
| **MDIRNET** | **32.68/0.959** | **0.23** |

### One-by-One vs All-in-One

| Type | Method | Params (M) | GFLOPs | BSD68 σ=25 | BSD68 σ=50 | Rain100L | GoPro | Avg. PSNR/SSIM |
|------|--------|-----------|--------|-----------|-----------|----------|-------|---------------|
| One-by-One | MDIRNET | 0.68 | 14.0 | 31.84/0.920 | 28.64/0.826 | 36.53/0.974 | 32.68/0.959 | 32.42/0.920 |
| All-in-One | MDIRNET | 0.68 | 14.0 | 31.56/0.913 | 28.33/0.845 | 35.21/0.958 | 31.34/0.914 | 31.61/0.908 |

### Mixed Degradation Robustness

| Noise | Rain | Blur | BSD68 σ=25/σ=50 | Rain100L | GoPro |
|-------|------|------|-----------------|----------|-------|
| ✓ | ✓ | — | 31.72/0.916 / 28.57/0.822 | 36.29/0.971 | — |
| ✓ | — | ✓ | 31.70/0.915 / 28.52/0.821 | — | 32.44/0.955 |
| — | ✓ | ✓ | — | 36.24/0.970 | 32.39/0.954 |
| ✓ | ✓ | ✓ | 31.56/0.913 / 28.33/0.845 | 35.21/0.958 | 31.34/0.914 |


## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
