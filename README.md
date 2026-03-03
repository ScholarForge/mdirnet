# MDIRNET: Multi-Degradation Image Restoration Network

Official  implementation of **"MDIRNET: Multi-Degradation Image Restoration Network via Deep Unfolding"** submitted to IEEE Transactions on Instrumentation and Measurement.

---

## Overview

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
