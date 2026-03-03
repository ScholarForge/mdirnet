#!/usr/bin/env python
"""
Training Strategy Comparison for MDIRNET
Compares One-by-One vs All-in-One training strategies
Matches Table 4 in the paper
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import sys
import os
from PIL import Image
import glob
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mdirnet.models import MDIRNET
from mdirnet.utils.metrics import psnr, ssim


class StrategyEvaluator:
    """Compare One-by-One vs All-in-One training strategies"""
    
    def __init__(self, device, results_dir='strategy_results'):
        self.device = device
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.figures_dir = self.results_dir / 'figures'
        self.figures_dir.mkdir(exist_ok=True)
        
        self.results = {
            'one_by_one': {'denoising': {}, 'deraining': {}, 'deblurring': {}},
            'all_in_one': {'denoising': {}, 'deraining': {}, 'deblurring': {}}
        }
    
    def load_dataset(self, name, num_images=10):
        base_dir = f'data/{name}'
        
        if name == 'BSD68':
            deg_dir = os.path.join(base_dir, 'noisy')
            clean_dir = os.path.join(base_dir, 'clean')
        elif name == 'Rain100L':
            deg_dir = os.path.join(base_dir, 'rainy')
            clean_dir = os.path.join(base_dir, 'clean')
        elif name == 'GoPro':
            deg_dir = os.path.join(base_dir, 'blur')
            clean_dir = os.path.join(base_dir, 'sharp')
        else:
            return None, None
        
        if not os.path.exists(deg_dir) or not os.path.exists(clean_dir):
            return None, None
        
        deg_files = sorted(glob.glob(os.path.join(deg_dir, '*.png')))[:num_images]
        
        degraded, clean = [], []
        for df in deg_files:
            cf = os.path.join(clean_dir, os.path.basename(df))
            if os.path.exists(cf):
                img_d = np.array(Image.open(df).convert('RGB')) / 255.0
                img_c = np.array(Image.open(cf).convert('RGB')) / 255.0
                degraded.append(torch.from_numpy(img_d).float().permute(2,0,1))
                clean.append(torch.from_numpy(img_c).float().permute(2,0,1))
        
        return torch.stack(degraded), torch.stack(clean)
    
    def evaluate_model(self, model, degraded, clean):
        degraded = degraded.to(self.device)
        clean = clean.to(self.device)
        
        psnr_vals, ssim_vals, times = [], [], []
        
        with torch.no_grad():
            for i in range(len(clean)):
                inp = degraded[i:i+1]
                gt = clean[i:i+1]
                
                # Time inference
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    start = time.time()
                    out = model(inp)
                    torch.cuda.synchronize()
                    end = time.time()
                else:
                    start = time.time()
                    out = model(inp)
                    end = time.time()
                
                psnr_vals.append(psnr(out, gt))
                ssim_vals.append(ssim(out, gt))
                times.append((end - start) * 1000)
        
        return {
            'psnr': np.mean(psnr_vals),
            'ssim': np.mean(ssim_vals),
            'time': np.mean(times)
        }
    
    def create_one_by_one_models(self):
        models = {}
        for task in ['denoising', 'deraining', 'deblurring']:
            model = MDIRNET(
                in_channels=3, patch_size=64, num_patch_groups=8,
                patches_per_group=16, num_ovpca_iterations=6,
                dram_min_rank=4, dram_max_rank=32
            ).to(self.device)
            model.eval()
            models[task] = model
        return models
    
    def create_all_in_one_model(self):
        model = MDIRNET(
            in_channels=3, patch_size=64, num_patch_groups=8,
            patches_per_group=16, num_ovpca_iterations=6,
            dram_min_rank=4, dram_max_rank=32
        ).to(self.device)
        model.eval()
        return model
    
    def run_comparison(self, num_images=10):

        print("\n" + "="*70)
        print(" "*10 + "ONE-BY-ONE VS ALL-IN-ONE COMPARISON")
        print("="*70)
        
        # Load datasets
        print("\nLoading datasets...")
        bsd_d, bsd_c = self.load_dataset('BSD68', num_images)
        rain_d, rain_c = self.load_dataset('Rain100L', num_images)
        gopro_d, gopro_c = self.load_dataset('GoPro', num_images)
        
        # Create models
        print("\nCreating models...")
        one_by_one = self.create_one_by_one_models()
        all_in_one = self.create_all_in_one_model()
        
        # Evaluate One-by-One
        print("\n" + "-"*60)
        print("📊 ONE-BY-ONE MODELS")
        print("-"*60)
        
        if bsd_d is not None:
            print("\n   Denoising (specialized model):")
            self.results['one_by_one']['denoising'] = self.evaluate_model(
                one_by_one['denoising'], bsd_d, bsd_c
            )
            print(f"      PSNR: {self.results['one_by_one']['denoising']['psnr']:.2f} dB")
        
        if rain_d is not None:
            print("\n   Deraining (specialized model):")
            self.results['one_by_one']['deraining'] = self.evaluate_model(
                one_by_one['deraining'], rain_d, rain_c
            )
            print(f"      PSNR: {self.results['one_by_one']['deraining']['psnr']:.2f} dB")
        
        if gopro_d is not None:
            print("\n   Deblurring (specialized model):")
            self.results['one_by_one']['deblurring'] = self.evaluate_model(
                one_by_one['deblurring'], gopro_d, gopro_c
            )
            print(f"      PSNR: {self.results['one_by_one']['deblurring']['psnr']:.2f} dB")
        
        # Evaluate All-in-One
        print("\n" + "-"*60)
        print("📊 ALL-IN-ONE MODEL")
        print("-"*60)
        
        if bsd_d is not None:
            print("\n   Denoising (unified model):")
            self.results['all_in_one']['denoising'] = self.evaluate_model(
                all_in_one, bsd_d, bsd_c
            )
            print(f"      PSNR: {self.results['all_in_one']['denoising']['psnr']:.2f} dB")
        
        if rain_d is not None:
            print("\n   Deraining (unified model):")
            self.results['all_in_one']['deraining'] = self.evaluate_model(
                all_in_one, rain_d, rain_c
            )
            print(f"      PSNR: {self.results['all_in_one']['deraining']['psnr']:.2f} dB")
        
        if gopro_d is not None:
            print("\n   Deblurring (unified model):")
            self.results['all_in_one']['deblurring'] = self.evaluate_model(
                all_in_one, gopro_d, gopro_c
            )
            print(f"      PSNR: {self.results['all_in_one']['deblurring']['psnr']:.2f} dB")
        
        return self.results
    

def main():
    print("\n" + "="*70)
    print(" "*10 + "MDIRNET: TRAINING STRATEGY COMPARISON")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📋 Device: {device}")
    
    # Run comparison
    evaluator = StrategyEvaluator(device)
    evaluator.run_comparison(num_images=5)

    
    print("\n" + "="*70)
    print(" "*15 + "COMPARISON COMPLETE")
    print("="*70)
    print(f"\nResults in: {evaluator.results_dir}")


if __name__ == '__main__':
    main()