"""
A more modular and extensible trainer class.
"""
import os
import time
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

class Trainer:
    """A modular trainer for deep learning models."""

    def __init__(self, config, model, train_loader, val_loader):
        """Initialize the trainer."""
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # DDP setup
        self.use_ddp = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
        if self.use_ddp:
            dist.init_process_group(backend='nccl')
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.device = torch.device(f'cuda:{self.rank}')
            torch.cuda.set_device(self.device)
        else:
            self.rank = 0
            self.world_size = 1
            self.device = torch.device(config.training.device)

        self.model.to(self.device)
        if self.use_ddp:
            self.model = DDP(self.model, device_ids=[self.rank])

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=5, verbose=True
        )

        # AMP and Gradient Accumulation
        self.use_amp = self.config.training.use_amp
        self.accumulation_steps = self.config.training.accumulation_steps
        self.scaler = GradScaler(enabled=self.use_amp)

        if self.rank == 0:
            self.save_dir = Path(config.training.save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
            log_dir = Path(config.training.log_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')
            self.writer = SummaryWriter(log_dir=log_dir)
            self._save_config()

    def _save_config(self):
        """Save configuration to file."""
        config_path = self.save_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=4)

    def train_epoch(self, epoch):
        """Train the model for one epoch."""
        self.model.train()
        if self.use_ddp:
            self.train_loader.sampler.set_epoch(epoch)

        running_loss = 0.0
        correct = 0
        total = 0

        self.optimizer.zero_grad()
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            with autocast(enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss = loss / self.accumulation_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            running_loss += loss.item() * self.accumulation_steps
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if self.rank == 0 and batch_idx % 10 == 0:
                print(f'Epoch: {epoch} [{batch_idx * len(inputs)}/{len(self.train_loader.dataset)} '
                      f'({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item() * self.accumulation_steps:.6f}')

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        if self.rank == 0:
            self.writer.add_scalar('Loss/train', epoch_loss, epoch)
            self.writer.add_scalar('Accuracy/train', epoch_acc, epoch)

        return epoch_loss, epoch_acc

    def validate(self, epoch):
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                with autocast(enabled=self.use_amp):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total

        if self.rank == 0:
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)

        self.scheduler.step(val_loss)

        return val_loss, val_acc

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        if self.rank == 0:
            state = {
                'epoch': epoch,
                'state_dict': self.model.module.state_dict() if self.use_ddp else self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'best_acc': getattr(self, 'best_acc', 0)
            }

            torch.save(state, self.save_dir / 'checkpoint.pth')

            if is_best:
                torch.save(state, self.save_dir / 'model_best.pth')

    def train(self):
        """Main training loop."""
        best_acc = 0

        if self.rank == 0:
            print(f"Starting training for {self.config.training.epochs} epochs...")
            print(f"Using device: {self.device}")

        for epoch in range(1, self.config.training.epochs + 1):
            start_time = time.time()

            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)

            if self.rank == 0:
                is_best = val_acc > best_acc
                if is_best:
                    best_acc = val_acc

                if epoch % self.config.training.checkpoint_freq == 0 or is_best:
                    self.save_checkpoint(epoch, is_best)

                epoch_time = time.time() - start_time
                print(f'Epoch: {epoch:03d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
                      f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Time: {epoch_time:.2f}s')
                print('-' * 100)

        if self.rank == 0:
            self.writer.close()
            torch.save(self.model.module.state_dict() if self.use_ddp else self.model.state_dict(), self.save_dir / 'model_final.pth')
            print(f'Training complete. Best validation accuracy: {best_acc:.2f}%')

        if self.use_ddp:
            dist.destroy_process_group()
