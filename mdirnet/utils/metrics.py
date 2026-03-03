import torch
import torch.nn.functional as F
import numpy as np

def psnr(img1, img2, data_range=1.0):
    """
    Compute PSNR between two images
    Args:
        img1, img2: [B, C, H, W] or [C, H, W] tensors
        data_range: Maximum pixel value (usually 1.0)
    """
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same shape")
    
    if img1.ndim == 4:
        # Batch of images
        mse = F.mse_loss(img1, img2, reduction='none').mean(dim=[1,2,3])
        psnr_vals = 20 * torch.log10(data_range / torch.sqrt(mse + 1e-8))
        return psnr_vals.mean().item()
    else:
        # Single image
        mse = F.mse_loss(img1, img2).item()
        if mse == 0:
            return 100
        return 20 * np.log10(data_range / np.sqrt(mse))

def ssim(img1, img2, data_range=1.0, window_size=11, size_average=True):
    """
    Simplified SSIM calculation
    Full SSIM requires scikit-image, this is a placeholder
    """
    try:
        from skimage.metrics import structural_similarity
        img1_np = img1.detach().cpu().numpy()
        img2_np = img2.detach().cpu().numpy()
        
        if img1_np.ndim == 4:
            ssim_vals = []
            for i in range(img1_np.shape[0]):
                if img1_np.shape[1] == 3:  # Color
                    ssim_val = structural_similarity(
                        img1_np[i].transpose(1,2,0),
                        img2_np[i].transpose(1,2,0),
                        data_range=data_range,
                        channel_axis=-1
                    )
                else:  # Grayscale
                    ssim_val = structural_similarity(
                        img1_np[i,0], img2_np[i,0],
                        data_range=data_range
                    )
                ssim_vals.append(ssim_val)
            return np.mean(ssim_vals)
        else:
            if img1_np.shape[0] == 3:  # Color
                return structural_similarity(
                    img1_np.transpose(1,2,0),
                    img2_np.transpose(1,2,0),
                    data_range=data_range,
                    channel_axis=-1
                )
            else:  # Grayscale
                return structural_similarity(
                    img1_np[0], img2_np[0],
                    data_range=data_range
                )
    except ImportError:
        print("Warning: scikit-image not installed. Install it for SSIM computation.")
        return 0.0