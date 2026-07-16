import argparse
import torch
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


def load_image_pairs(degraded_dir, clean_dir, num_images=None):
    deg_files = sorted(glob.glob(os.path.join(degraded_dir, '*.png')))

    if num_images:
        deg_files = deg_files[:num_images]

    degraded = []
    clean = []

    for f in deg_files:

        cf = os.path.join(clean_dir, os.path.basename(f))

        if os.path.exists(cf):

            img_d = np.array(
                Image.open(f).convert("RGB"),
                dtype=np.float32
            ) / 255.0

            img_c = np.array(
                Image.open(cf).convert("RGB"),
                dtype=np.float32
            ) / 255.0

            degraded.append(
                torch.from_numpy(img_d).permute(2, 0, 1)
            )

            clean.append(
                torch.from_numpy(img_c).permute(2, 0, 1)
            )

    if len(degraded) == 0:
        return None, None

    return torch.stack(degraded), torch.stack(clean)


def evaluate(model, degraded, clean, device, num_warmup=10):

    degraded = degraded.to(device)
    clean = clean.to(device)

    psnr_vals = []
    ssim_vals = []
    times = []

    model.eval()

    with torch.inference_mode():

        # Warmup
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
        "time": np.median(times)
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=[
            "denoising",
            "deraining",
            "deblurring"
        ]
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--num_images",
        type=int,
        default=None
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
    )

    args = parser.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )

    model = MDIRNET().to(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    ###################################################
    # Model profiling
    ###################################################

    params, flops = profile_model(
        model,
        input_size=(1, 3, 256, 256)
    )

    print("\n========== Model Profile ==========")
    print(f"Parameters : {params/1e6:.2f} M")
    print(f"GFLOPs     : {flops/1e9:.2f}")
    print("===================================\n")

    ###################################################

    if args.task == "denoising":

        deg, clean = load_image_pairs(
            os.path.join(args.data_dir, "noisy"),
            os.path.join(args.data_dir, "clean"),
            args.num_images
        )

    elif args.task == "deraining":

        deg, clean = load_image_pairs(
            os.path.join(args.data_dir, "rainy"),
            os.path.join(args.data_dir, "clean"),
            args.num_images
        )

    else:

        deg, clean = load_image_pairs(
            os.path.join(args.data_dir, "blur"),
            os.path.join(args.data_dir, "sharp"),
            args.num_images
        )

    if deg is None:

        print(f"No images found in {args.data_dir}")
        return

    print(f"Evaluating on {len(deg)} images...\n")

    results = evaluate(
        model,
        deg,
        clean,
        device
    )

    fps = 1.0 / results["time"]

    print("========== Results ==========")
    print(f"PSNR                 : {results['psnr']:.2f} dB")
    print(f"SSIM                 : {results['ssim']:.4f}")
    print(f"Median Runtime       : {results['time']:.4f} s")
    print(f"FPS                  : {fps:.2f}")
    print("=============================")


if __name__ == "__main__":
    main()