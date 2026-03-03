#!/usr/bin/env python
"""
Training script for MDIRNET
"""

import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mdirnet.models import MDIRNET
from mdirnet.utils.losses import MDIRNETLoss
from mdirnet.data.dataset import create_dataset
from mdirnet.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train MDIRNET')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--task', type=str, default='denoising',
                        choices=['denoising', 'deraining', 'deblurring', 'all_in_one'],
                        help='Restoration task')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = MDIRNET(**config['model'])
    model = model.to(args.device)
    
    # Create loss function
    criterion = MDIRNETLoss(**config['training']['loss'])
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs']
    )
    
    # Create datasets
    train_dataset = create_dataset(
        task=args.task,
        split='train',
        config=config['data']
    )
    
    val_dataset = create_dataset(
        task=args.task,
        split='val',
        config=config['data']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=args.device
    )
    
    # Train
    trainer.train(num_epochs=config['training']['num_epochs'])


if __name__ == '__main__':
    main()