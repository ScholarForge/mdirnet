
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mdirnet.models import MDIRNET
from mdirnet.utils.losses import MDIRNETLoss
from mdirnet.utils import profile_model
from mdirnet.data.dataset import create_dataset, create_all_in_one_dataset
from mdirnet.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train MDIRNET")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="denoising",
        choices=[
            "denoising",
            "deraining",
            "deblurring",
            "all_in_one",
        ],
    )
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():

    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )

    model = MDIRNET(**config["model"]).to(device)

    params, flops = profile_model(
        model,
        input_size=(1, 3, 256, 256)
    )

    print("\n========== Model Profile ==========")
    print(f"Parameters : {params/1e6:.2f} M")
    print(f"GFLOPs     : {flops/1e9:.2f}")
    print("===================================\n")

    print(f"Task         : {args.task}")
    print(f"Device       : {device}")
    print(f"Batch size   : {config['training']['batch_size']}")
    print(f"Epochs       : {config['training']['num_epochs']}")
    print(f"Learning rate: {config['training']['learning_rate']}\n")

    criterion = MDIRNETLoss(
        **config["training"]["loss"]
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        betas=(
            config["training"]["beta1"],
            config["training"]["beta2"],
        ),
        weight_decay=config["training"]["weight_decay"],
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["num_epochs"],
        eta_min=config["training"]["min_learning_rate"],
    )

    crop_size = config["data"]["crop_size"]

    if args.task == "all_in_one":

        train_dataset = create_all_in_one_dataset(
            config["data"],
            split="train",
            crop_size=crop_size,
            augment=True,
        )

        val_dataset = create_all_in_one_dataset(
            config["data"],
            split="val",
            crop_size=crop_size,
            augment=False,
        )

    else:

        data_dir = (
            args.data_dir
            or config["data"][args.task]["datasets"][0]
        )

        train_dataset = create_dataset(
            args.task,
            data_dir,
            split="train",
            crop_size=crop_size,
            augment=True,
        )

        val_dataset = create_dataset(
            args.task,
            data_dir,
            split="val",
            crop_size=crop_size,
            augment=False,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["hardware"]["num_workers"],
        pin_memory=config["hardware"]["pin_memory"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["hardware"]["num_workers"],
        pin_memory=config["hardware"]["pin_memory"],
    )

    if args.resume:

        checkpoint = torch.load(
            args.resume,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        print(
            f"Resumed from epoch {checkpoint['epoch']}"
        )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    trainer.train(
        num_epochs=config["training"]["num_epochs"]
    )


if __name__ == "__main__":
    main()
