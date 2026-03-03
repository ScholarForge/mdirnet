import torch
import torch.nn as nn
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler,
                 train_loader, val_loader, config, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
    
        self.writer = SummaryWriter(config['logging']['log_dir'])
        self.checkpoint_dir = config['logging']['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            degraded = batch['degraded'].to(self.device)
            clean = batch['clean'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            restored = self.model(degraded)
            
            # Compute loss
            loss = self.criterion(restored, clean)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_psnr = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                degraded = batch['degraded'].to(self.device)
                clean = batch['clean'].to(self.device)
                
                restored = self.model(degraded)
                
                # Compute PSNR
                mse = nn.functional.mse_loss(restored, clean)
                psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
                total_psnr += psnr.item()
        
        return total_psnr / len(self.val_loader)
    
    def train(self, num_epochs):
        best_psnr = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            
            # Validate
            val_psnr = self.validate()
            self.writer.add_scalar('PSNR/val', val_psnr, epoch)
            
            print(f"Train Loss: {train_loss:.4f}, Val PSNR: {val_psnr:.2f}dB")
            
            # Update scheduler
            self.scheduler.step()
            
            # Save checkpoint
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_psnr': best_psnr,
                }, os.path.join(self.checkpoint_dir, 'best_model.pth'))
                print(f"✓ Saved best model with PSNR: {best_psnr:.2f}dB")
            
            # Regular checkpoint
            if (epoch + 1) % self.config['logging']['save_frequency'] == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                }, os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        self.writer.close()