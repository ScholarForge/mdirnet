#!/usr/bin/env python
"""
Evaluation Script for MDIRNET
Evaluates on individual degradations:
- Denoising on BSD68 
- Deraining on Rain100L 
- Deblurring on GoPro 

"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import sys
import os
from PIL import Image
import glob
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mdirnet.models import MDIRNET
from mdirnet.utils.metrics import psnr, ssim


class StandardEvaluator:
    """Evaluate MDIRNET on standard datasets (single degradations)"""
    
    def __init__(self, model, device, results_dir='standard_results'):
        self.model = model
        self.device = device
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.figures_dir = self.results_dir / 'figures'
        self.figures_dir.mkdir(exist_ok=True)
        
        self.results = {
            'denoising': {'psnr': 0, 'ssim': 0, 'time': 0, 'per_image': []},
            'deraining': {'psnr': 0, 'ssim': 0, 'time': 0, 'per_image': []},
            'deblurring': {'psnr': 0, 'ssim': 0, 'time': 0, 'per_image': []}
        }
    
    def load_bsd68(self, data_dir='data/BSD68', num_images=10):
        """Load BSD68 denoising dataset"""
        print(f"\n   Loading BSD68...")
        
        # Try different structures
        noisy_dir = os.path.join(data_dir, 'noisy')
        clean_dir = os.path.join(data_dir, 'clean')
        
        degraded, clean = [], []
        
        if os.path.exists(noisy_dir) and os.path.exists(clean_dir):
            files = sorted(glob.glob(os.path.join(noisy_dir, '*.png')))[:num_images]
            for f in files:
                cf = os.path.join(clean_dir, os.path.basename(f))
                if os.path.exists(cf):
                    img_n = np.array(Image.open(f).convert('RGB')) / 255.0
                    img_c = np.array(Image.open(cf).convert('RGB')) / 255.0
                    degraded.append(torch.from_numpy(img_n).float().permute(2,0,1))
                    clean.append(torch.from_numpy(img_c).float().permute(2,0,1))
        
        print(f"      Loaded {len(degraded)} image pairs")
        return torch.stack(degraded) if degraded else None, torch.stack(clean) if clean else None
    
    def load_rain100l(self, data_dir='data/Rain100L', num_images=10):
        """Load Rain100L deraining dataset"""
        print(f"\n   Loading Rain100L...")
        
        rainy_dir = os.path.join(data_dir, 'rainy')
        clean_dir = os.path.join(data_dir, 'clean')
        
        if not os.path.exists(rainy_dir):
            return None, None
        
        degraded, clean = [], []
        files = sorted(glob.glob(os.path.join(rainy_dir, '*.png')))[:num_images]
        
        for f in files:
            cf = os.path.join(clean_dir, os.path.basename(f))
            if os.path.exists(cf):
                img_r = np.array(Image.open(f).convert('RGB')) / 255.0
                img_c = np.array(Image.open(cf).convert('RGB')) / 255.0
                degraded.append(torch.from_numpy(img_r).float().permute(2,0,1))
                clean.append(torch.from_numpy(img_c).float().permute(2,0,1))
        
        print(f"      Loaded {len(degraded)} image pairs")
        return torch.stack(degraded), torch.stack(clean)
    
    def load_gopro(self, data_dir='data/GoPro', num_images=10):
        """Load GoPro deblurring dataset"""
        print(f"\n   Loading GoPro...")
        
        blur_dir = os.path.join(data_dir, 'blur')
        sharp_dir = os.path.join(data_dir, 'sharp')
        
        if not os.path.exists(blur_dir):
            return None, None
        
        degraded, clean = [], []
        files = sorted(glob.glob(os.path.join(blur_dir, '*.png')))[:num_images]
        
        for f in files:
            sf = os.path.join(sharp_dir, os.path.basename(f))
            if os.path.exists(sf):
                img_b = np.array(Image.open(f).convert('RGB')) / 255.0
                img_s = np.array(Image.open(sf).convert('RGB')) / 255.0
                degraded.append(torch.from_numpy(img_b).float().permute(2,0,1))
                clean.append(torch.from_numpy(img_s).float().permute(2,0,1))
        
        print(f"      Loaded {len(degraded)} image pairs")
        return torch.stack(degraded), torch.stack(clean)
    
    def evaluate_task(self, degraded, clean, task_name):
        degraded = degraded.to(self.device)
        clean = clean.to(self.device)
        
        psnr_vals, ssim_vals, times = [], [], []
        samples = []
        
        with torch.no_grad():
            for i in range(len(clean)):
                inp = degraded[i:i+1]
                gt = clean[i:i+1]
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    start = time.time()
                    out = self.model(inp)
                    torch.cuda.synchronize()
                    end = time.time()
                else:
                    start = time.time()
                    out = self.model(inp)
                    end = time.time()
                
                # Metrics
                p = psnr(out, gt)
                s = ssim(out, gt)
                t = (end - start) * 1000
                
                psnr_vals.append(p)
                ssim_vals.append(s)
                times.append(t)
                
                if i < 3:  
                    samples.append({
                        'degraded': inp.cpu(),
                        'restored': out.cpu(),
                        'clean': gt.cpu(),
                        'psnr': p,
                        'ssim': s
                    })
        
        return {
            'psnr': np.mean(psnr_vals),
            'ssim': np.mean(ssim_vals),
            'time': np.mean(times),
            'per_image': list(zip(psnr_vals, ssim_vals, times)),
            'samples': samples
        }
    
    def run_evaluation(self, num_images=10):
        """Run evaluation on all datasets"""
        print("\n" + "="*70)
        print(" "*20 + "STANDARD EVALUATION")
        print("="*70)
        
        # 1. Denoising
        print("\nTask 1: Denoising on BSD68")
        print("-"*50)
        deg, clean = self.load_bsd68(num_images=num_images)
        if deg is not None:
            self.results['denoising'] = self.evaluate_task(deg, clean, 'denoising')
            print(f"   PSNR: {self.results['denoising']['psnr']:.2f} dB")
            print(f"   SSIM: {self.results['denoising']['ssim']:.4f}")
            print(f"   Time: {self.results['denoising']['time']:.2f} ms")
        
        # 2. Deraining
        print("\nTask 2: Deraining on Rain100L")
        print("-"*50)
        deg, clean = self.load_rain100l(num_images=num_images)
        if deg is not None:
            self.results['deraining'] = self.evaluate_task(deg, clean, 'deraining')
            print(f"   PSNR: {self.results['deraining']['psnr']:.2f} dB")
            print(f"   SSIM: {self.results['deraining']['ssim']:.4f}")
            print(f"   Time: {self.results['deraining']['time']:.2f} ms")
        
        # 3. Deblurring
        print("\nTask 3: Deblurring on GoPro")
        print("-"*50)
        deg, clean = self.load_gopro(num_images=num_images)
        if deg is not None:
            self.results['deblurring'] = self.evaluate_task(deg, clean, 'deblurring')
            print(f"   PSNR: {self.results['deblurring']['psnr']:.2f} dB")
            print(f"   SSIM: {self.results['deblurring']['ssim']:.4f}")
            print(f"   Time: {self.results['deblurring']['time']:.2f} ms")
        
        return self.results
    
    def generate_figures(self):
        """Generate Figures 4, 5, 6"""
        
        # Figure 4: Denoising results
        if self.results['denoising'].get('samples'):
            self._plot_results('denoising', 'Denoising on BSD68', 'figure4_denoising.png')
        
        # Figure 5: Deraining results
        if self.results['deraining'].get('samples'):
            self._plot_results('deraining', 'Deraining on Rain100L', 'figure5_deraining.png')
        
        # Figure 6: Deblurring results
        if self.results['deblurring'].get('samples'):
            self._plot_results('deblurring', 'Deblurring on GoPro', 'figure6_deblurring.png')



def main():
    print("\n" + "="*70)
    print(" "*20 + "MDIRNET STANDARD EVALUATION")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📋 Device: {device}")
    
    model = MDIRNET(
        in_channels=3, patch_size=64, num_patch_groups=8,
        patches_per_group=16, num_ovpca_iterations=6,
        dram_min_rank=4, dram_max_rank=32
    ).to(device)
    model.eval()
    
    # Evaluate
    evaluator = StandardEvaluator(model, device)
    evaluator.run_evaluation(num_images=10)
    evaluator.generate_figures()

    
    print("\n" + "="*70)
    print(" "*20 + "EVALUATION COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {evaluator.results_dir}")


if __name__ == '__main__':
    main()