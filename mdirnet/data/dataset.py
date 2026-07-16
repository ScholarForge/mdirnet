import torch
from torch.utils.data import Dataset, ConcatDataset
import os
from PIL import Image
import glob
import numpy as np
import random


class ImagePairDataset(Dataset):

    def __init__(self, degraded_dir, clean_dir, crop_size=None, augment=False):
        self.crop_size = crop_size
        self.augment = augment

        self.degraded_images = sorted(glob.glob(os.path.join(degraded_dir, '*.png')))
        self.clean_images = sorted(glob.glob(os.path.join(clean_dir, '*.png')))
        assert len(self.degraded_images) == len(self.clean_images)

    def __len__(self):
        return len(self.degraded_images)

    def __getitem__(self, idx):
        degraded = np.array(Image.open(self.degraded_images[idx]).convert('RGB'), dtype=np.float32) / 255.0
        clean = np.array(Image.open(self.clean_images[idx]).convert('RGB'), dtype=np.float32) / 255.0

        if self.crop_size is not None:
            h, w, _ = degraded.shape
            if h >= self.crop_size and w >= self.crop_size:
                top = random.randint(0, h - self.crop_size)
                left = random.randint(0, w - self.crop_size)
                degraded = degraded[top:top+self.crop_size, left:left+self.crop_size]
                clean = clean[top:top+self.crop_size, left:left+self.crop_size]

        if self.augment:
            if random.random() > 0.5:
                degraded = np.fliplr(degraded).copy()
                clean = np.fliplr(clean).copy()
            k = random.choice([0, 1, 2, 3])
            degraded = np.rot90(degraded, k).copy()
            clean = np.rot90(clean, k).copy()

        degraded = torch.from_numpy(degraded).permute(2, 0, 1)
        clean = torch.from_numpy(clean).permute(2, 0, 1)

        return {
            'degraded': degraded,
            'clean': clean,
            'filename': os.path.basename(self.degraded_images[idx])
        }


class DenoisingDataset(ImagePairDataset):

    def __init__(self, root_dir, split='test', crop_size=None, augment=False, sigma=25):
        self.sigma = sigma
        clean_dir = os.path.join(root_dir, 'clean')
        noisy_dir = os.path.join(root_dir, 'noisy')

        if os.path.exists(noisy_dir):
            super().__init__(noisy_dir, clean_dir, crop_size, augment)
            self.online_noise = False
        else:
            self.clean_images = sorted(glob.glob(os.path.join(clean_dir, '*.png')))
            self.crop_size = crop_size
            self.augment = augment
            self.online_noise = True

    def __len__(self):
        return len(self.clean_images)

    def __getitem__(self, idx):
        if not self.online_noise:
            return super().__getitem__(idx)

        clean = np.array(Image.open(self.clean_images[idx]).convert('RGB'), dtype=np.float32) / 255.0

        if self.crop_size is not None:
            h, w, _ = clean.shape
            if h >= self.crop_size and w >= self.crop_size:
                top = random.randint(0, h - self.crop_size)
                left = random.randint(0, w - self.crop_size)
                clean = clean[top:top+self.crop_size, left:left+self.crop_size]

        if self.augment:
            if random.random() > 0.5:
                clean = np.fliplr(clean).copy()
            k = random.choice([0, 1, 2, 3])
            clean = np.rot90(clean, k).copy()

        noise = np.random.randn(*clean.shape).astype(np.float32) * (self.sigma / 255.0)
        degraded = np.clip(clean + noise, 0, 1)

        degraded = torch.from_numpy(degraded).permute(2, 0, 1)
        clean = torch.from_numpy(clean).permute(2, 0, 1)

        return {
            'degraded': degraded,
            'clean': clean,
            'filename': os.path.basename(self.clean_images[idx])
        }


class DerainingDataset(ImagePairDataset):

    def __init__(self, root_dir, split='test', crop_size=None, augment=False):
        rainy_dir = os.path.join(root_dir, 'rainy')
        clean_dir = os.path.join(root_dir, 'clean')
        super().__init__(rainy_dir, clean_dir, crop_size, augment)


class DeblurringDataset(ImagePairDataset):

    def __init__(self, root_dir, split='test', crop_size=None, augment=False):
        blur_dir = os.path.join(root_dir, 'blur')
        sharp_dir = os.path.join(root_dir, 'sharp')
        super().__init__(blur_dir, sharp_dir, crop_size, augment)


def create_dataset(task, root_dir, split='test', crop_size=None, augment=False, sigma=25):
    if task == 'denoising':
        return DenoisingDataset(root_dir, split, crop_size, augment, sigma)
    elif task == 'deraining':
        return DerainingDataset(root_dir, split, crop_size, augment)
    elif task == 'deblurring':
        return DeblurringDataset(root_dir, split, crop_size, augment)
    else:
        raise ValueError(f"Unknown task: {task}")


def create_all_in_one_dataset(data_config, split='train', crop_size=64, augment=True):
    datasets = []

    if 'denoising' in data_config:
        for root in data_config['denoising']['datasets']:
            if os.path.exists(root):
                for sigma in data_config['denoising'].get('noise_levels', [25]):
                    datasets.append(DenoisingDataset(root, split, crop_size, augment, sigma))

    if 'deraining' in data_config:
        for root in data_config['deraining']['datasets']:
            if os.path.exists(root):
                datasets.append(DerainingDataset(root, split, crop_size, augment))

    if 'deblurring' in data_config:
        for root in data_config['deblurring']['datasets']:
            if os.path.exists(root):
                datasets.append(DeblurringDataset(root, split, crop_size, augment))

    if not datasets:
        raise ValueError("No valid dataset directories found")

    return ConcatDataset(datasets)
