
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import time
import sys
import os
from PIL import Image
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mdirnet.models import MDIRNET
from mdirnet.utils.metrics import psnr, ssim
from mdirnet.utils import profile_model


def add_noise(images, sigma=25):
    noise = torch.randn_like(images) * (sigma / 255.0)
    return (images + noise).clamp(0, 1)


def add_blur(images, kernel_size=11):
    c = images.shape[1]
    kernel = torch.ones(
        c, 1, kernel_size, kernel_size,
        device=images.device
    ) / (kernel_size ** 2)

    return F.conv2d(
        images,
        kernel,
        padding=kernel_size // 2,
        groups=c
    )


def add_rain(images, num_streaks=20, intensity=0.3):
    rainy = images.clone()
    b, c, h, w = images.shape

    for i in range(b):
        for _ in range(num_streaks):
            x = np.random.randint(0, max(w - 20, 1))
            y = np.random.randint(0, max(h - 20, 1))
            rainy[i, :, y:y+15, x:x+2] += intensity

    return rainy.clamp(0, 1)


def load_clean_images(data_dir, num_images=None):
    clean_dir = os.path.join(data_dir, "clean")

    if not os.path.exists(clean_dir):
        return None

    files = sorted(glob.glob(os.path.join(clean_dir, "*.png")))

    if num_images:
        files = files[:num_images]

    images = []

    for f in files:
        img = np.array(
            Image.open(f).convert("RGB"),
            dtype=np.float32
        ) / 255.0

        images.append(
            torch.from_numpy(img).permute(2, 0, 1)
        )

    return torch.stack(images) if images else None


def evaluate_mixed(model, degraded, clean, device, num_warmup=10):

    degraded = degraded.to(device)
    clean = clean.to(device)

    psnr_vals = []
    ssim_vals = []
    times = []

    model.eval()

    with torch.inference_mode():

        for _ in range(num_warmup):
            _ = model(degraded[0:1])

        if device.type == "cuda":
            torch.cuda.synchronize()

        for i in range(len(clean)):

            inp = degraded[i:i+1]
            gt = clean[i:i+1]

            if device.type == "cuda":

                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)

                starter.record()
                out = model(inp)
                ender.record()

                torch.cuda.synchronize()

                elapsed = starter.elapsed_time(ender) / 1000.0

            else:

                start = time.time()
                out = model(inp)
                elapsed = time.time() - start

            psnr_vals.append(psnr(out, gt))
            ssim_vals.append(ssim(out, gt))
            times.append(elapsed)

    return {
        "psnr": np.mean(psnr_vals),
        "ssim": np.mean(ssim_vals),
        "time": np.median(times),
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/BSD68")
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )

    model = MDIRNET().to(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    params, flops = profile_model(
        model,
        input_size=(1, 3, 256, 256)
    )

    print("\n========== Model Profile ==========")
    print(f"Parameters : {params/1e6:.2f} M")
    print(f"GFLOPs     : {flops/1e9:.2f}")
    print("===================================\n")

    clean = load_clean_images(
        args.data_dir,
        args.num_images
    )

    if clean is None:
        print("No clean images found.")
        return

    clean = clean.to(device)

    combinations = {
        "Noise + Rain":
            lambda x: add_noise(add_rain(x), sigma=25),

        "Noise + Blur":
            lambda x: add_noise(add_blur(x), sigma=25),

        "Rain + Blur":
            lambda x: add_blur(add_rain(x)),

        "Noise + Rain + Blur":
            lambda x: add_noise(add_blur(add_rain(x)), sigma=25),
    }

    print(f"Evaluating {len(clean)} images...\n")

    for name, degrade_fn in combinations.items():

        degraded = degrade_fn(clean)

        results = evaluate_mixed(
            model,
            degraded,
            clean,
            device
        )

        fps = 1.0 / results["time"]

        print(f"{name}")
        print(f"  PSNR   : {results['psnr']:.2f} dB")
        print(f"  SSIM   : {results['ssim']:.4f}")
        print(f"  Time   : {results['time']:.4f} s")
        print(f"  FPS    : {fps:.2f}")
        print()


if __name__ == "__main__":
    main()
