"""
Main training script for deep learning models.
"""
import os
import time
import json
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from configs.config import Config, DataConfig, ModelConfig, TrainingConfig
from models.custom_cnn import get_model
from data.dataloader import get_data_loaders

class Trainer:
    """Main trainer class for model training and evaluation."""
    
    def __init__(self, config: Config):
        """Initialize the trainer with configuration."""
        self.config = config
        self.device = torch.device(config.training.device)
        
        # Create output directories
        self.save_dir = Path(config.training.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model, criterion, optimizer, and learning rate scheduler
        self.model = get_model(config.model.model_name, config.model.num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=5, verbose=True
        )
        
        # Load data
        self.loaders = get_data_loaders(
            data_dir=config.data.data_dir,
            batch_size=config.data.batch_size,
            img_size=config.data.img_size,
            num_workers=config.data.num_workers,
            train_val_split=config.data.train_val_split
        )
        
        # Initialize TensorBoard writer
        log_dir = Path(config.training.log_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Save config
        self._save_config()
        
    def _save_config(self):
        """Save configuration to file."""
        config_path = self.save_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=4)
    
    def train_epoch(self, epoch: int) -> float:
        """Train the model for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.loaders['train']):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Log training progress
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch} [{batch_idx * len(inputs)}/{len(self.loaders["train"].dataset)} ' \
                      f'({100. * batch_idx / len(self.loaders["train"]):.0f}%)]\tLoss: {loss.item():.6f}')
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.loaders['train'])
        epoch_acc = 100. * correct / total
        
        # Log to TensorBoard
        self.writer.add_scalar('Loss/train', epoch_loss, epoch)
        self.writer.add_scalar('Accuracy/train', epoch_acc, epoch)
        
        return epoch_loss, epoch_acc
    
    def validate(self, epoch: int) -> float:
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.loaders['val']:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # Calculate validation metrics
        val_loss = running_loss / len(self.loaders['val'])
        val_acc = 100. * correct / total
        
        # Log to TensorBoard
        self.writer.add_scalar('Loss/val', val_loss, epoch)
        self.writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        # Update learning rate
        self.scheduler.step(val_loss)
        
        return val_loss, val_acc
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_acc': getattr(self, 'best_acc', 0)
        }
        
        # Save latest checkpoint
        torch.save(state, self.save_dir / 'checkpoint.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(state, self.save_dir / 'model_best.pth')
    
    def train(self):
        """Main training loop."""
        best_acc = 0
        
        print(f"Starting training for {self.config.training.epochs} epochs...")
        print(f"Using device: {self.device}")
        
        for epoch in range(1, self.config.training.epochs + 1):
            start_time = time.time()
            
            # Train for one epoch
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Evaluate on validation set
            val_loss, val_acc = self.validate(epoch)
            
            # Update best model
            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc
            
            # Save checkpoint
            if epoch % self.config.training.checkpoint_freq == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Print epoch statistics
            epoch_time = time.time() - start_time
            print(f'Epoch: {epoch:03d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | ' \
                  f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Time: {epoch_time:.2f}s')
            print('-' * 100)
        
        # Close TensorBoard writer
        self.writer.close()
        
        # Save final model
        torch.save(self.model.state_dict(), self.save_dir / 'model_final.pth')
        print(f'Training complete. Best validation accuracy: {best_acc:.2f}%')

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a deep learning model')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='data/raw', help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load config if provided, else use default
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = Config(
            data=DataConfig(**config_dict.get('data', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {}))
        )
    else:
        # Use default config with command line overrides
        config = Config(
            data=DataConfig(data_dir=args.data_dir, batch_size=args.batch_size),
            training=TrainingConfig(epochs=args.epochs, learning_rate=args.lr, device=args.device)
        )
    
    # Create and run trainer
    trainer = Trainer(config)
    trainer.train()

if __name__ == '__main__':
    main()
