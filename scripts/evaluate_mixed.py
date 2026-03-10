
"""
Mixed Degradation Evaluation for MDIRNET
Tests on images with multiple simultaneous degradations:
- Rain + Noise
- Blur + Noise
- Rain + Blur
- Rain + Blur + Noise (all three)
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


class MixedEvaluator:
    """Evaluate MDIRNET on mixed degradations"""
    
    def __init__(self, model, device, results_dir='mixed_results'):
        self.model = model
        self.device = device
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.figures_dir = self.results_dir / 'figures'
        self.figures_dir.mkdir(exist_ok=True)
        
        self.results = {
            'rain_noise': {},
            'blur_noise': {},
            'rain_blur': {},
            'rain_blur_noise': {}
        }
    
    def load_clean_images(self, data_dir='data/BSD68', num_images=5):
        """Load clean images from BSD68"""
        clean_dir = os.path.join(data_dir, 'clean')
        if not os.path.exists(clean_dir):
            return None
        
        files = sorted(glob.glob(os.path.join(clean_dir, '*.png')))[:num_images]
        clean = []
        for f in files:
            img = np.array(Image.open(f).convert('RGB')) / 255.0
            clean.append(torch.from_numpy(img).float().permute(2,0,1))
        
        return torch.stack(clean) if clean else None
    
    def add_noise(self, images, sigma=25):
        """Add Gaussian noise"""
        noise = torch.randn_like(images) * (sigma / 255.0)
        return (images + noise).clamp(0, 1)
    
    def add_blur(self, images, kernel_size=11, sigma=1.5):
        """Add Gaussian blur"""
        # Simple blur via averaging
        blurred = []
        kernel = torch.ones(1,1,kernel_size,kernel_size) / (kernel_size*kernel_size)
        kernel = kernel.to(images.device)
        
        for i in range(images.shape[0]):
            img = images[i:i+1]
            blurred.append(F.conv2d(img, kernel.repeat(3,1,1,1), padding=kernel_size//2, groups=3))
        
        return torch.cat(blurred, dim=0)
    
    def add_rain(self, images, num_streaks=20, intensity=0.3):
        """Add synthetic rain streaks"""
        rainy = images.clone()
        B, C, H, W = images.shape
        
        for b in range(B):
            for _ in range(num_streaks):
                x = np.random.randint(0, W-20)
                y = np.random.randint(0, H-20)
                rainy[b, :, y:y+15, x:x+2] += intensity
        
        return rainy.clamp(0, 1)
    
    def create_mixed(self, clean):
        """Create all mixed degradation types"""
        clean = clean.to(self.device)
        
        mixed = {
            'rain_noise': self.add_noise(self.add_rain(clean, num_streaks=20), sigma=15),
            'blur_noise': self.add_noise(self.add_blur(clean, kernel_size=9), sigma=15),
            'rain_blur': self.add_blur(self.add_rain(clean, num_streaks=20), kernel_size=9),
            'rain_blur_noise': self.add_noise(
                self.add_blur(self.add_rain(clean, num_streaks=25), kernel_size=9), 
                sigma=20
            )
        }
        
        return mixed, clean
    
    def evaluate_mixed(self, degraded, clean, name):
        """Evaluate on one mixed type"""
        degraded = degraded.to(self.device)
        clean = clean.to(self.device)
        
        psnr_vals, ssim_vals, times = [], [], []
        samples = []
        
        with torch.no_grad():
            for i in range(len(clean)):
                inp = degraded[i:i+1]
                gt = clean[i:i+1]
                
                # Time inference
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
                
                psnr_vals.append(psnr(out, gt))
                ssim_vals.append(ssim(out, gt))
                times.append((end - start) * 1000)
                
                if i < 3:
                    samples.append({
                        'degraded': inp.cpu(),
                        'restored': out.cpu(),
                        'clean': gt.cpu(),
                        'psnr': psnr_vals[-1],
                        'ssim': ssim_vals[-1]
                    })
                
                print(f"      Image {i+1}: PSNR={psnr_vals[-1]:.2f}dB")
        
        return {
            'psnr': np.mean(psnr_vals),
            'ssim': np.mean(ssim_vals),
            'time': np.mean(times),
            'samples': samples
        }
    
    def run_evaluation(self, num_images=5):
        """Run all mixed evaluations"""
        print("\n" + "="*70)
        print(" "*15 + "MIXED DEGRADATION EVALUATION")
        print("="*70)
        
        # Load clean images
        clean = self.load_clean_images(num_images=num_images)
        if clean is None:
            print("No clean images found")
            return
        
        mixed, clean = self.create_mixed(clean)
        
        # Evaluate each type
        for name, deg in mixed.items():
            print(f"\nEvaluating: {name.replace('_', ' + ').upper()}")
            print("-"*50)
            self.results[name] = self.evaluate_mixed(deg, clean, name)
            print(f"   → PSNR: {self.results[name]['psnr']:.2f} dB")
        
        return self.results
    
    def generate_figures(self):
        """Generate figures for each mixed type"""
        for name, res in self.results.items():
            if not res or 'samples' not in res:
                continue
            
            samples = res['samples']
            fig, axes = plt.subplots(len(samples), 3, figsize=(10, 3*len(samples)))
            
            if len(samples) == 1:
                axes = axes.reshape(1, -1)
            
            for i in range(len(samples)):
                s = samples[i]
                
                # Degraded
                axes[i,0].imshow(np.clip(s['degraded'][0].permute(1,2,0).numpy(), 0, 1))
                axes[i,0].set_title('Input', fontsize=10)
                axes[i,0].axis('off')
                
                # Restored
                axes[i,1].imshow(np.clip(s['restored'][0].permute(1,2,0).numpy(), 0, 1))
                axes[i,1].set_title(f'MDIRNET\n{s["psnr"]:.2f}dB', fontsize=10)
                axes[i,1].axis('off')
                
                # Clean
                axes[i,2].imshow(np.clip(s['clean'][0].permute(1,2,0).numpy(), 0, 1))
                axes[i,2].set_title('Ground Truth', fontsize=10)
                axes[i,2].axis('off')
            
            title = name.replace('_', ' + ').upper()
            plt.suptitle(f'Mixed Degradation: {title}', fontsize=12)
            plt.tight_layout()
            plt.savefig(self.figures_dir / f'mixed_{name}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def save_results(self):
        """Save results to file"""
        with open(self.results_dir / 'mixed_summary.txt', 'w') as f:
            f.write("="*50 + "\n")
            f.write("MDIRNET MIXED DEGRADATION RESULTS\n")
            f.write(f"Date: {datetime.now()}\n")
            f.write("="*50 + "\n\n")
            
            for name, res in self.results.items():
                if res:
                    f.write(f"{name.replace('_', ' + ').upper()}:\n")
                    f.write(f"  PSNR: {res['psnr']:.2f} dB\n")
                    f.write(f"  SSIM: {res['ssim']:.4f}\n")
                    f.write(f"  Time: {res['time']:.2f} ms\n\n")


def main():
    print("\n" + "="*70)
    print(" "*15 + "MDIRNET MIXED DEGRADATION")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load model
    model = MDIRNET(
        in_channels=3, patch_size=64, num_patch_groups=8,
        patches_per_group=16, num_ovpca_iterations=6,
        dram_min_rank=4, dram_max_rank=32
    ).to(device)
    model.eval()
    
    # Evaluate
    evaluator = MixedEvaluator(model, device)
    evaluator.run_evaluation(num_images=5)
    evaluator.generate_figures()
    evaluator.save_results()
    
    print("\n" + "="*70)
    print(" "*15 + "EVALUATION COMPLETE")
    print("="*70)
    print(f"\n Results in: {evaluator.results_dir}")


if __name__ == '__main__':

    main()
