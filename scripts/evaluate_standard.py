import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import sys
import os
from PIL import Image
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mdirnet.models import MDIRNET
from mdirnet.utils.metrics import psnr, ssim


class StandardEvaluator:
    
    def __init__(self, model, device, results_dir='standard_results'):
        self.model = model
        self.device = device
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.figures_dir = self.results_dir / 'figures'
        self.figures_dir.mkdir(exist_ok=True)
        
        self.results = {
            'denoising_bsd68': {'psnr': 0, 'ssim': 0, 'time': 0, 'samples': []},
            'denoising_sidd': {'psnr': 0, 'ssim': 0, 'time': 0, 'samples': []}, 
            'deraining': {'psnr': 0, 'ssim': 0, 'time': 0, 'samples': []},
            'deblurring': {'psnr': 0, 'ssim': 0, 'time': 0, 'samples': []}
        }
    
    def load_bsd68(self, data_dir='data/BSD68', num_images=10):
        """Load BSD68 denoising dataset"""
        print(f"\n   Loading BSD68...")
        noisy_dir = os.path.join(data_dir, 'noisy')
        clean_dir = os.path.join(data_dir, 'clean')
        
        if not os.path.exists(noisy_dir):
            return None, None
        
        files = sorted(glob.glob(os.path.join(noisy_dir, '*.png')))[:num_images]
        degraded, clean = [], []
        
        for f in files:
            cf = os.path.join(clean_dir, os.path.basename(f))
            if os.path.exists(cf):
                img_n = np.array(Image.open(f).convert('RGB')) / 255.0
                img_c = np.array(Image.open(cf).convert('RGB')) / 255.0
                degraded.append(torch.from_numpy(img_n).float().permute(2,0,1))
                clean.append(torch.from_numpy(img_c).float().permute(2,0,1))
        
        print(f"      Loaded {len(degraded)} BSD68 image pairs")
        return torch.stack(degraded), torch.stack(clean)
    
    def load_sidd(self, data_dir='data/SIDD', num_images=10):
        """NEW: Load SIDD dataset (real smartphone sensor noise)"""
        print(f"\n   Loading SIDD (real sensor noise)...")
        noisy_dir = os.path.join(data_dir, 'noisy')
        clean_dir = os.path.join(data_dir, 'clean')
        
        if not os.path.exists(noisy_dir):
            print("      SIDD dataset not found - skipping")
            return None, None
        
        files = sorted(glob.glob(os.path.join(noisy_dir, '*.png')))[:num_images]
        degraded, clean = [], []
        
        for f in files:
            cf = os.path.join(clean_dir, os.path.basename(f))
            if os.path.exists(cf):
                img_n = np.array(Image.open(f).convert('RGB')) / 255.0
                img_c = np.array(Image.open(cf).convert('RGB')) / 255.0
                degraded.append(torch.from_numpy(img_n).float().permute(2,0,1))
                clean.append(torch.from_numpy(img_c).float().permute(2,0,1))
        
        print(f"      Loaded {len(degraded)} SIDD image pairs")
        return torch.stack(degraded), torch.stack(clean)
    
    def load_rain100l(self, data_dir='data/Rain100L', num_images=10):
        """Load Rain100L deraining dataset"""
        print(f"\n   Loading Rain100L...")
        rainy_dir = os.path.join(data_dir, 'rainy')
        clean_dir = os.path.join(data_dir, 'clean')
        
        if not os.path.exists(rainy_dir):
            return None, None
        
        files = sorted(glob.glob(os.path.join(rainy_dir, '*.png')))[:num_images]
        degraded, clean = [], []
        
        for f in files:
            cf = os.path.join(clean_dir, os.path.basename(f))
            if os.path.exists(cf):
                img_r = np.array(Image.open(f).convert('RGB')) / 255.0
                img_c = np.array(Image.open(cf).convert('RGB')) / 255.0
                degraded.append(torch.from_numpy(img_r).float().permute(2,0,1))
                clean.append(torch.from_numpy(img_c).float().permute(2,0,1))
        
        print(f"      Loaded {len(degraded)} Rain100L image pairs")
        return torch.stack(degraded), torch.stack(clean)
    
    def load_gopro(self, data_dir='data/GoPro', num_images=10):
        """Load GoPro deblurring dataset"""
        print(f"\n   Loading GoPro...")
        blur_dir = os.path.join(data_dir, 'blur')
        sharp_dir = os.path.join(data_dir, 'sharp')
        
        if not os.path.exists(blur_dir):
            return None, None
        
        files = sorted(glob.glob(os.path.join(blur_dir, '*.png')))[:num_images]
        degraded, clean = [], []
        
        for f in files:
            sf = os.path.join(sharp_dir, os.path.basename(f))
            if os.path.exists(sf):
                img_b = np.array(Image.open(f).convert('RGB')) / 255.0
                img_s = np.array(Image.open(sf).convert('RGB')) / 255.0
                degraded.append(torch.from_numpy(img_b).float().permute(2,0,1))
                clean.append(torch.from_numpy(img_s).float().permute(2,0,1))
        
        print(f"      Loaded {len(degraded)} GoPro image pairs")
        return torch.stack(degraded), torch.stack(clean)
    
    def evaluate_task(self, degraded, clean):
        """Evaluate model on a task"""
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
            'samples': samples
        }
    
    def run_evaluation(self, num_images=10):
        """Run evaluation on all datasets"""
        print("\n" + "="*70)
        print(" "*20 + "STANDARD EVALUATION")
        print("="*70)
        
        # 1. BSD68 Denoising
        print("\nTask 1: Denoising on BSD68 (σ=25)")
        print("-"*50)
        deg, clean = self.load_bsd68(num_images=num_images)
        if deg is not None:
            self.results['denoising_bsd68'] = self.evaluate_task(deg, clean)
            print(f"   PSNR: {self.results['denoising_bsd68']['psnr']:.2f} dB")
            print(f"   SSIM: {self.results['denoising_bsd68']['ssim']:.4f}")
            print(f"   Time: {self.results['denoising_bsd68']['time']:.2f} ms")
        
        # 2. SIDD Real Sensor Noise 
        print("\nTask 2: Denoising on SIDD (Real Sensor Noise)")
        print("-"*50)
        deg, clean = self.load_sidd(num_images=num_images)
        if deg is not None:
            self.results['denoising_sidd'] = self.evaluate_task(deg, clean)
            print(f"   PSNR: {self.results['denoising_sidd']['psnr']:.2f} dB")
            print(f"   SSIM: {self.results['denoising_sidd']['ssim']:.4f}")
            print(f"   Time: {self.results['denoising_sidd']['time']:.2f} ms")
        
        # 3. Deraining
        print("\nTask 3: Deraining on Rain100L")
        print("-"*50)
        deg, clean = self.load_rain100l(num_images=num_images)
        if deg is not None:
            self.results['deraining'] = self.evaluate_task(deg, clean)
            print(f"   PSNR: {self.results['deraining']['psnr']:.2f} dB")
            print(f"   SSIM: {self.results['deraining']['ssim']:.4f}")
            print(f"   Time: {self.results['deraining']['time']:.2f} ms")
        
        # 4. Deblurring
        print("\nTask 4: Deblurring on GoPro")
        print("-"*50)
        deg, clean = self.load_gopro(num_images=num_images)
        if deg is not None:
            self.results['deblurring'] = self.evaluate_task(deg, clean)
            print(f"   PSNR: {self.results['deblurring']['psnr']:.2f} dB")
            print(f"   SSIM: {self.results['deblurring']['ssim']:.4f}")
            print(f"   Time: {self.results['deblurring']['time']:.2f} ms")
        
        return self.results
    
    def generate_figure4(self):
        """Figure 4: Denoising with 4 rows (3 synthetic + 1 SIDD real noise)"""
        fig, axes = plt.subplots(4, 3, figsize=(12, 16))
        
        # Rows 1-3: BSD68 synthetic noise
        bsd_samples = self.results['denoising_bsd68'].get('samples', [])
        for i in range(min(3, len(bsd_samples))):
            s = bsd_samples[i]
            axes[i,0].imshow(np.clip(s['degraded'][0].permute(1,2,0).numpy(), 0, 1))
            axes[i,0].set_title('Noisy Input', fontsize=10)
            axes[i,0].axis('off')
            
            axes[i,1].imshow(np.clip(s['restored'][0].permute(1,2,0).numpy(), 0, 1))
            axes[i,1].set_title(f'MDIRNET\n{s["psnr"]:.2f}dB', fontsize=10)
            axes[i,1].axis('off')
            
            axes[i,2].imshow(np.clip(s['clean'][0].permute(1,2,0).numpy(), 0, 1))
            axes[i,2].set_title('Ground Truth', fontsize=10)
            axes[i,2].axis('off')
        
        # Row 4: SIDD real sensor noise
        sidd_samples = self.results['denoising_sidd'].get('samples', [])
        if sidd_samples:
            s = sidd_samples[0]
            axes[3,0].imshow(np.clip(s['degraded'][0].permute(1,2,0).numpy(), 0, 1))
            axes[3,0].set_title('Real Sensor Noise (SIDD)', fontsize=10)
            axes[3,0].axis('off')
            
            axes[3,1].imshow(np.clip(s['restored'][0].permute(1,2,0).numpy(), 0, 1))
            axes[3,1].set_title(f'MDIRNET\n{s["psnr"]:.2f}dB', fontsize=10)
            axes[3,1].axis('off')
            
            axes[3,2].imshow(np.clip(s['clean'][0].permute(1,2,0).numpy(), 0, 1))
            axes[3,2].set_title('Ground Truth', fontsize=10)
            axes[3,2].axis('off')
        
        plt.suptitle('Figure 4: Denoising Results (BSD68 synthetic + SIDD real noise)', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'figure4_denoising.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_3x3_results(self, task_key, title, filename):
        """Helper to plot 3x3 grid for a task"""
        samples = self.results[task_key].get('samples', [])
        if not samples:
            return
        
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        
        for i in range(min(3, len(samples))):
            s = samples[i]
            axes[i,0].imshow(np.clip(s['degraded'][0].permute(1,2,0).numpy(), 0, 1))
            axes[i,0].set_title('Input', fontsize=12)
            axes[i,0].axis('off')
            
            axes[i,1].imshow(np.clip(s['restored'][0].permute(1,2,0).numpy(), 0, 1))
            axes[i,1].set_title(f'MDIRNET\n{s["psnr"]:.2f}dB', fontsize=12)
            axes[i,1].axis('off')
            
            axes[i,2].imshow(np.clip(s['clean'][0].permute(1,2,0).numpy(), 0, 1))
            axes[i,2].set_title('Ground Truth', fontsize=12)
            axes[i,2].axis('off')
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(self.figures_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" Saved: {filename}")
    
    def generate_figures(self):
        """Generate all figures for paper"""
        self.generate_figure4()
        self._plot_3x3_results('deraining', 'Figure 5: Deraining on Rain100L', 'figure5_deraining.png')
        self._plot_3x3_results('deblurring', 'Figure 6: Deblurring on GoPro', 'figure6_deblurring.png')


def main():
    print("\n" + "="*70)
    print(" "*20 + "MDIRNET STANDARD EVALUATION")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n Device: {device}")
    
    model = MDIRNET(
        in_channels=3, patch_size=64, num_patch_groups=8,
        patches_per_group=16, num_ovpca_iterations=6,
        dram_min_rank=4, dram_max_rank=32
    ).to(device)
    model.eval()
    
    evaluator = StandardEvaluator(model, device)
    evaluator.run_evaluation(num_images=10)
    evaluator.generate_figures()
    
    print("\n" + "="*70)
    print(" "*20 + "EVALUATION COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {evaluator.results_dir}")


if __name__ == '__main__':
    main()