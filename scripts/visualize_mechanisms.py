
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from PIL import Image
import glob
from scipy.ndimage import gaussian_filter, sobel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mdirnet.models import MDIRNET


def load_image(data_dir, dataset_type, idx=0):
    if dataset_type == "denoising":
        deg_dir = os.path.join(data_dir, "noisy")
        clean_dir = os.path.join(data_dir, "clean")
    elif dataset_type == "deraining":
        deg_dir = os.path.join(data_dir, "rainy")
        clean_dir = os.path.join(data_dir, "clean")
    elif dataset_type == "deblurring":
        deg_dir = os.path.join(data_dir, "blur")
        clean_dir = os.path.join(data_dir, "sharp")
    else:
        return None, None

    files = sorted(glob.glob(os.path.join(deg_dir, "*.png")))
    if not files or idx >= len(files):
        return None, None

    img = np.array(Image.open(files[idx]).convert("RGB"), dtype=np.float32) / 255.0
    degraded = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    clean = None
    cf = os.path.join(clean_dir, os.path.basename(files[idx]))
    if os.path.exists(cf):
        cimg = np.array(Image.open(cf).convert("RGB"), dtype=np.float32) / 255.0
        clean = torch.from_numpy(cimg).permute(2, 0, 1).unsqueeze(0)

    return degraded, clean


def visualize_ppm(model, degraded, output_dir):
    model.eval()

    with torch.inference_mode():
        _, flow_field, deformed_grid = model.ppm(degraded)

    h, w = degraded.shape[2:]
    stride = model.ppm.patch_size

    sampled = deformed_grid[0, ::stride, ::stride].cpu().numpy()

    density = np.zeros((h, w), dtype=np.float32)

    for i in range(sampled.shape[0]):
        for j in range(sampled.shape[1]):
            x = int((sampled[i, j, 0] + 1) * 0.5 * (w - 1))
            y = int((sampled[i, j, 1] + 1) * 0.5 * (h - 1))
            x = np.clip(x, 0, w - 1)
            y = np.clip(y, 0, h - 1)
            np.add.at(density, (y, x), 1)

    density = gaussian_filter(density, sigma=3)

    degraded_np = degraded[0].cpu().permute(1,2,0).numpy()
    gray = degraded_np.mean(axis=2)
    gx = sobel(gray, axis=0)
    gy = sobel(gray, axis=1)
    grad = np.sqrt(gx**2 + gy**2)

    flow_mag = np.sqrt(flow_field[0,0].cpu().numpy()**2 + flow_field[0,1].cpu().numpy()**2)

    fig, ax = plt.subplots(1,4, figsize=(20,5))
    ax[0].imshow(degraded_np); ax[0].set_title("Input"); ax[0].axis("off")
    im=ax[1].imshow(flow_mag,cmap="viridis"); ax[1].set_title("Flow Magnitude"); ax[1].axis("off"); plt.colorbar(im,ax=ax[1],fraction=0.046)
    im=ax[2].imshow(density,cmap="hot"); ax[2].set_title("Patch Density"); ax[2].axis("off"); plt.colorbar(im,ax=ax[2],fraction=0.046)
    im=ax[3].imshow(grad,cmap="gray"); ax[3].set_title("Gradient"); ax[3].axis("off"); plt.colorbar(im,ax=ax[3],fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,"ppm_patch_distribution.png"),dpi=300,bbox_inches="tight")
    plt.close()


def visualize_dram(model, degraded, output_dir):
    model.eval()
    with torch.inference_mode():
        if model.use_ppm:
            patches,_,_ = model.ppm(degraded)
        else:
            patches = model._extract_fixed_patches(degraded)

        patch_groups,_ = model._group_patches_knn(patches)

        ranks=[]
        if model.use_dram:
            for group in patch_groups:
                rank=model.dram(group)
                ranks.append(rank.mean().item())

    fig,ax=plt.subplots(1,2,figsize=(10,5))
    ax[0].imshow(degraded[0].cpu().permute(1,2,0))
    ax[0].set_title("Input")
    ax[0].axis("off")
    ax[1].bar(range(len(ranks)),ranks,color="steelblue",edgecolor="black")
    ax[1].axhline(np.mean(ranks),color="red",ls="--",label=f"Mean {np.mean(ranks):.1f}")
    ax[1].set_title("DRAM Rank Allocation")
    ax[1].set_xlabel("Patch Group")
    ax[1].set_ylabel("Rank")
    ax[1].grid(alpha=0.3)
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,"dram_rank_allocation.png"),dpi=300,bbox_inches="tight")
    plt.close()


def visualize_sam(model,degraded,output_dir):
    model.eval()
    with torch.inference_mode():
        output,inter=model(degraded,return_intermediates=True)

    gate=inter["gate_map"]
    if gate is None:
        return

    gate=gate[0,0].cpu().numpy()
    restored=output[0].cpu().permute(1,2,0).numpy()
    inp=degraded[0].cpu().permute(1,2,0).numpy()

    fig,ax=plt.subplots(1,3,figsize=(15,5))
    ax[0].imshow(inp); ax[0].set_title("Input"); ax[0].axis("off")
    im=ax[1].imshow(gate,cmap="hot",vmin=0,vmax=1); ax[1].set_title("SAM Gate"); ax[1].axis("off"); plt.colorbar(im,ax=ax[1],fraction=0.046)
    ax[2].imshow(restored); ax[2].set_title("Restored"); ax[2].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,"sam_gate_maps.png"),dpi=300,bbox_inches="tight")
    plt.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--data_dir",required=True)
    p.add_argument("--task",default="denoising",choices=["denoising","deraining","deblurring"])
    p.add_argument("--img_index",type=int,default=0)
    p.add_argument("--output_dir",default="paper_figures")
    p.add_argument("--device",default="cuda")
    args=p.parse_args()

    device=torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir,exist_ok=True)

    model=MDIRNET().to(device)
    ckpt=torch.load(args.checkpoint,map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    degraded,_=load_image(args.data_dir,args.task,args.img_index)
    if degraded is None:
        print("Could not load image.")
        return

    degraded=degraded.to(device)

    visualize_ppm(model,degraded,args.output_dir)
    visualize_dram(model,degraded,args.output_dir)
    visualize_sam(model,degraded,args.output_dir)

    print(f"Figures saved to {args.output_dir}")

if __name__=="__main__":
    main()
