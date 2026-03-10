import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import glob
import numpy as np

class DenoisingDataset(Dataset):
    """BSD68 dataset for denoising"""
    def __init__(self, root_dir, split='test', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Load image pairs
        noisy_dir = os.path.join(root_dir, 'noisy')
        clean_dir = os.path.join(root_dir, 'clean')
        
        self.noisy_images = sorted(glob.glob(os.path.join(noisy_dir, '*.png')))
        self.clean_images = sorted(glob.glob(os.path.join(clean_dir, '*.png')))
    
    def __len__(self):
        return len(self.noisy_images)
    
    def __getitem__(self, idx):
        noisy = Image.open(self.noisy_images[idx]).convert('RGB')
        clean = Image.open(self.clean_images[idx]).convert('RGB')
        
        # Convert to tensor
        noisy = torch.from_numpy(np.array(noisy)).float().permute(2,0,1) / 255.0
        clean = torch.from_numpy(np.array(clean)).float().permute(2,0,1) / 255.0
        
        return {
            'degraded': noisy,
            'clean': clean,
            'filename': os.path.basename(self.noisy_images[idx])
        }

class DerainingDataset(Dataset):
    """Rain100L dataset for deraining"""
    def __init__(self, root_dir, split='test', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        rainy_dir = os.path.join(root_dir, 'rainy')
        clean_dir = os.path.join(root_dir, 'clean')
        
        self.rainy_images = sorted(glob.glob(os.path.join(rainy_dir, '*.png')))
        self.clean_images = sorted(glob.glob(os.path.join(clean_dir, '*.png')))
    
    def __len__(self):
        return len(self.rainy_images)
    
    def __getitem__(self, idx):
        rainy = Image.open(self.rainy_images[idx]).convert('RGB')
        clean = Image.open(self.clean_images[idx]).convert('RGB')
        
        rainy = torch.from_numpy(np.array(rainy)).float().permute(2,0,1) / 255.0
        clean = torch.from_numpy(np.array(clean)).float().permute(2,0,1) / 255.0
        
        return {
            'degraded': rainy,
            'clean': clean,
            'filename': os.path.basename(self.rainy_images[idx])
        }

class DeblurringDataset(Dataset):
    """GoPro dataset for deblurring"""
    def __init__(self, root_dir, split='test', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        blur_dir = os.path.join(root_dir, 'blur')
        sharp_dir = os.path.join(root_dir, 'sharp')
        
        self.blur_images = sorted(glob.glob(os.path.join(blur_dir, '*.png')))
        self.sharp_images = sorted(glob.glob(os.path.join(sharp_dir, '*.png')))
    
    def __len__(self):
        return len(self.blur_images)
    
    def __getitem__(self, idx):
        blur = Image.open(self.blur_images[idx]).convert('RGB')
        sharp = Image.open(self.sharp_images[idx]).convert('RGB')
        
        blur = torch.from_numpy(np.array(blur)).float().permute(2,0,1) / 255.0
        sharp = torch.from_numpy(np.array(sharp)).float().permute(2,0,1) / 255.0
        
        return {
            'degraded': blur,
            'clean': sharp,
            'filename': os.path.basename(self.blur_images[idx])
        }

def create_dataset(task, split, config):
    """Factory function to create dataset"""
    if task == 'denoising':
        return DenoisingDataset(
            root_dir=config['denoising']['datasets'][0],
            split=split
        )
    elif task == 'deraining':
        return DerainingDataset(
            root_dir=config['deraining']['datasets'][0],
            split=split
        )
    elif task == 'deblurring':
        return DeblurringDataset(
            root_dir=config['deblurring']['datasets'][0],
            split=split
        )
    else:

        raise ValueError(f"Unknown task: {task}")
