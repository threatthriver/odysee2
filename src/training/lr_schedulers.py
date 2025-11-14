"""
Learning Rate Schedulers for Language Model Training

This module implements various learning rate scheduling strategies optimized
for large-scale language model training, including warmup schedules, decay
strategies, and cyclical learning rates.
"""

import torch
from torch.optim.lr_scheduler import LRScheduler, _LRScheduler
from typing import Optional, List, Dict, Any, Union
import math
import warnings


def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
    verbose: bool = False
) -> "_LRScheduler":
    """
    Create a schedule that linearly increases the learning rate from 0 to lr
    and then linearly decreases it to 0 at `num_training_steps`.
    
    Args:
        optimizer: Optimizer object
        num_warmup_steps: Number of steps for the warmup phase
        num_training_steps: Total number of training steps
        last_epoch: The index of the last epoch when resuming training
        verbose: Whether to print schedule information
        
    Returns:
        Linear schedule with warmup
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return LambdaLR(optimizer, lr_lambda, last_epoch, verbose)


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    verbose: bool = False
) -> "_LRScheduler":
    """
    Create a schedule that uses a cosine annealing schedule with warmup.
    
    Args:
        optimizer: Optimizer object
        num_warmup_steps: Number of steps for the warmup phase
        num_training_steps: Total number of training steps
        num_cycles: Number of cycles in the cosine schedule
        last_epoch: The index of the last epoch when resuming training
        verbose: Whether to print schedule information
        
    Returns:
        Cosine schedule with warmup
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch, verbose)


def get_polynomial_decay_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    power: float = 1.0,
    lr_end: float = 1e-7,
    last_epoch: int = -1,
    verbose: bool = False
) -> "_LRScheduler":
    """
    Create a schedule with a learning rate that decreases from the initial lr set
    in the optimizer to a final lr_end * initial lr after the total number of training steps.
    
    Args:
        optimizer: Optimizer object
        num_warmup_steps: Number of steps for the warmup phase
        num_training_steps: Total number of training steps
        power: Power of polynomial decay
        lr_end: Final learning rate as a fraction of the initial lr
        last_epoch: The index of the last epoch when resuming training
        verbose: Whether to print schedule information
        
    Returns:
        Polynomial decay schedule with warmup
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps:
            return lr_end
        else:
            lr_range = 1.0 - lr_end
            decay_steps = num_training_steps - num_warmup_steps
            pct_remaining = 1.0 - (current_step - num_warmup_steps) / decay_steps
            return (lr_range * (pct_remaining ** power)) + lr_end
    
    return LambdaLR(optimizer, lr_lambda, last_epoch, verbose)


def get_inverse_sqrt_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    timescale: Optional[int] = None,
    last_epoch: int = -1,
    verbose: bool = False
) -> "_LRScheduler":
    """
    Create a schedule that uses an inverse square root schedule.
    
    Args:
        optimizer: Optimizer object
        num_warmup_steps: Number of steps for the warmup phase
        timescale: Time scale for the inverse square root decay
        last_epoch: The index of the last epoch when resuming training
        verbose: Whether to print schedule information
        
    Returns:
        Inverse square root schedule with warmup
    """
    if timescale is None:
        timescale = num_warmup_steps
    
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            return math.sqrt(timescale) / math.sqrt(max(current_step, timescale))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch, verbose)


def get_exponential_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    gamma: float = 0.95,
    last_epoch: int = -1,
    verbose: bool = False
) -> "_LRScheduler":
    """
    Create a schedule that uses exponential decay.
    
    Args:
        optimizer: Optimizer object
        num_warmup_steps: Number of steps for the warmup phase
        num_training_steps: Total number of training steps
        gamma: Multiplication factor for exponential decay
        last_epoch: The index of the last epoch when resuming training
        verbose: Whether to print schedule information
        
    Returns:
        Exponential schedule with warmup
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            # Exponential decay after warmup
            decay_steps = current_step - num_warmup_steps
            return gamma ** decay_steps
    
    return LambdaLR(optimizer, lr_lambda, last_epoch, verbose)


def get_step_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    step_size: int,
    gamma: float = 0.1,
    last_epoch: int = -1,
    verbose: bool = False
) -> "_LRScheduler":
    """
    Create a schedule with step decay after warmup.
    
    Args:
        optimizer: Optimizer object
        num_warmup_steps: Number of steps for the warmup phase
        step_size: Number of steps between lr reductions
        gamma: Multiplication factor for step decay
        last_epoch: The index of the last epoch when resuming training
        verbose: Whether to print schedule information
        
    Returns:
        Step schedule with warmup
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            # Step decay after warmup
            step_decay = math.floor((current_step - num_warmup_steps) / step_size)
            return gamma ** step_decay
    
    return LambdaLR(optimizer, lr_lambda, last_epoch, verbose)


class OneCycleLRWithWarmup(LRScheduler):
    """
    One Cycle Learning Rate scheduler with warmup.
    
    This implements the one-cycle learning rate policy where the learning rate
    starts at a low value, increases to a maximum value, and then decreases
    to a very low value.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        max_lr: float,
        min_lr: float = None,
        cycle_momentum: bool = True,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        if min_lr is None:
            min_lr = max_lr / 1000
        
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.cycle_momentum = cycle_momentum
        
        # Calculate phases
        self.warmup_ratio = num_warmup_steps / num_training_steps
        self.anneal_ratio = 0.1  # Last 10% for annealing
        
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self) -> List[float]:
        """Calculate learning rate for each parameter group."""
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                         "please use `get_last_lr()`.", UserWarning)
        
        step_num = self.last_epoch
        
        if step_num >= self.num_training_steps:
            return [self.min_lr for _ in self.optimizer.param_groups]
        
        if step_num < self.num_warmup_steps:
            # Warmup phase
            return [self.max_lr * (step_num / self.num_warmup_steps) for _ in self.optimizer.param_groups]
        else:
            # One cycle phase (increase then decrease)
            remaining_steps = self.num_training_steps - step_num
            
            if remaining_steps > self.num_training_steps * self.anneal_ratio:
                # Increasing phase (half of one cycle)
                cycle_progress = (step_num - self.num_warmup_steps) / (self.num_training_steps * 0.5)
                lr = self.min_lr + (self.max_lr - self.min_lr) * (1 - abs(2 * cycle_progress - 1))
            else:
                # Decreasing phase
                cycle_progress = (step_num - self.num_warmup_steps) / (self.num_training_steps * 0.5)
                lr = self.max_lr - (self.max_lr - self.min_lr) * (cycle_progress - 0.5) * 2
            
            return [lr for _ in self.optimizer.param_groups]


class CyclicLRWithWarmup(LRScheduler):
    """
    Cyclic learning rate scheduler with warmup.
    
    Implements cyclic learning rates that oscillate between minimum and maximum values.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        base_lr: float,
        max_lr: float,
        step_size_up: int = 2000,
        step_size_down: Optional[int] = None,
        mode: str = 'triangular',
        gamma: float = 1.0,
        scale_mode: str = 'cycle',
        last_epoch: int = -1,
        verbose: bool = False
    ):
        if step_size_down is None:
            step_size_down = step_size_up
        
        self.num_warmup_steps = num_warmup_steps
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down
        self.mode = mode
        self.gamma = gamma
        self.scale_mode = scale_mode
        
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self) -> List[float]:
        """Calculate learning rate for each parameter group."""
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                         "please use `get_last_lr()`.", UserWarning)
        
        step_num = self.last_epoch
        
        if step_num >= self.num_warmup_steps:
            # Regular cycling after warmup
            cycle, cycle_size = self._get_cycle(step_num - self.num_warmup_steps)
            x = 1.0 + step_num / cycle_size
            return self._get_scale_from_cycle(x, cycle)
        else:
            # Warmup phase
            return [self.max_lr * (step_num / self.num_warmup_steps) for _ in self.optimizer.param_groups]
    
    def _get_cycle(self, step_num: int) -> tuple:
        """Get cycle information."""
        if self.step_size_up == self.step_size_down:
            if self.mode == 'triangular2':
                cycle = 1.0
            elif self.mode == 'exp_range':
                cycle = (self.gamma ** (step_num // self.step_size_up))
            else:
                cycle = 1.0
        else:
            if self.mode == 'triangular2':
                cycle = 1.0 + step_num // (2 * self.step_size_up)
            elif self.mode == 'exp_range':
                cycle = self.gamma ** (step_num // self.step_size_up)
            else:
                cycle = 1.0 + step_num // (2 * self.step_size_up)
        
        cycle_size = self.step_size_up + self.step_size_down
        return cycle, cycle_size
    
    def _get_scale_from_cycle(self, x: float, cycle: float) -> List[float]:
        """Calculate learning rate scale from cycle."""
        if self.scale_mode == 'cycle':
            scale = 1.0 / (cycle ** (x - 1.0))
        else:
            scale = 1.0
        
        lr_scale = []
        for base_lr, max_lr in zip(self.base_lrs, self.max_lrs):
            if self.mode == 'triangular':
                lr_scale.append(scale)
            elif self.mode == 'triangular2':
                lr_scale.append(scale)
            elif self.mode == 'exp_range':
                lr_scale.append(scale)
        
        return [lr_scale[0] * lr for lr in self.base_lrs]


class WarmupScheduler(LRScheduler):
    """
    Simple warmup scheduler that linearly increases learning rate.
    
    Can be combined with any other scheduler for the warmup phase.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        warmup_factor: float = 1.0,
        warmup_method: str = 'linear',
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.num_warmup_steps = num_warmup_steps
        self.warmup_factor = warmup_factor
        self.warmup_method = warmup_method
        
        if warmup_method not in ['linear', 'exp', 'cos']:
            raise ValueError(f"Invalid warmup method: {warmup_method}")
        
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self) -> List[float]:
        """Calculate learning rate for each parameter group."""
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                         "please use `get_last_lr()`.", UserWarning)
        
        if self.last_epoch < self.num_warmup_steps:
            if self.warmup_method == 'linear':
                progress = self.last_epoch / self.num_warmup_steps
                lr_factor = progress
            elif self.warmup_method == 'exp':
                lr_factor = self.warmup_factor ** self.last_epoch
            elif self.warmup_method == 'cos':
                lr_factor = 0.5 * (1 - math.cos(math.pi * self.last_epoch / self.num_warmup_steps))
            
            return [lr_factor * lr for lr in self.base_lrs]
        else:
            return self.base_lrs


class AdaptiveLRScheduler(LRScheduler):
    """
    Adaptive learning rate scheduler that adjusts based on validation metrics.
    
    This scheduler monitors validation performance and adjusts the learning rate
    when performance plateaus or starts degrading.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        metric_fn,
        patience: int = 10,
        factor: float = 0.5,
        min_lr: float = 1e-6,
        window_size: int = 5,
        threshold: float = 1e-4,
        mode: str = 'min',
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.metric_fn = metric_fn
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.window_size = window_size
        self.threshold = threshold
        self.mode = mode
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.wait = 0
        self.best_values = []
        
        super().__init__(optimizer, last_epoch, verbose)
    
    def step(self, metrics: Optional[float] = None, epoch: Optional[int] = None):
        """Step the scheduler with optional metrics."""
        current = float(metrics) if metrics is not None else self.metric_fn()
        self.last_epoch = int(epoch) if epoch is not None else self.last_epoch
        
        if self.last_epoch < 0:
            raise RuntimeError("You seem to have resumed a scheduler without starting it first.")
        
        # Update best value
        if self.mode == 'min':
            if current < self.best_value - self.threshold:
                self.best_value = current
                self.wait = 0
            else:
                self.wait += 1
        else:  # mode == 'max'
            if current > self.best_value + self.threshold:
                self.best_value = current
                self.wait = 0
            else:
                self.wait += 1
        
        # Store best values for windowed evaluation
        self.best_values.append(self.best_value)
        if len(self.best_values) > self.window_size:
            self.best_values.pop(0)
        
        # Check if learning rate should be reduced
        if self.wait >= self.patience:
            self._reduce_lr()
            self.wait = 0
            self.best_values.clear()
        
        super().step()
    
    def _reduce_lr(self):
        """Reduce learning rate by factor."""
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lr)
            if old_lr != new_lr:
                param_group['lr'] = new_lr
                if self.verbose:
                    print(f'Reducing learning rate of group {i} to {new_lr:.4e}.')
    
    def get_last_lr(self) -> List[float]:
        """Get last computed learning rate."""
        return [group['lr'] for group in self.optimizer.param_groups]


class LambdaLR(LRScheduler):
    """
    Learning rate scheduler that applies a lambda function to update the learning rate.
    
    This is a wrapper around PyTorch's built-in LambdaLR with additional functionality.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_lambda,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.lr_lambda = lr_lambda
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self) -> List[float]:
        """Calculate learning rate using lambda function."""
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                         "please use `get_last_lr()`.", UserWarning)
        
        return [self.lr_lambda(self.last_epoch) for group in self.optimizer.param_groups for group in [group]]


class SchedulerManager:
    """
    Manager for handling multiple learning rate schedulers.
    
    Useful when using different schedulers for different parameter groups
    or when combining warmup with decay schedules.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        schedulers: List[LRScheduler],
        active_schedulers: Optional[List[int]] = None,
        **kwargs
    ):
        self.optimizer = optimizer
        self.schedulers = schedulers
        self.active_schedulers = active_schedulers or list(range(len(schedulers)))
        
        if not all(isinstance(s, LRScheduler) for s in schedulers):
            raise TypeError("All schedulers must be instances of LRScheduler")
    
    def step(self, **kwargs):
        """Step all active schedulers."""
        for i in self.active_schedulers:
            if i < len(self.schedulers):
                scheduler = self.schedulers[i]
                # Check if scheduler has step method with kwargs
                if hasattr(scheduler, 'step') and 'metrics' in scheduler.__init__.__code__.co_varnames:
                    scheduler.step(**kwargs)
                else:
                    scheduler.step()
    
    def get_last_lr(self) -> List[float]:
        """Get last computed learning rates from all schedulers."""
        last_lrs = []
        for i in self.active_schedulers:
            if i < len(self.schedulers):
                scheduler = self.schedulers[i]
                if hasattr(scheduler, 'get_last_lr'):
                    last_lrs.extend(scheduler.get_last_lr())
        return last_lrs
    
    def get_lr_history(self) -> Dict[str, List[float]]:
        """Get learning rate history for all schedulers."""
        history = {}
        for i, scheduler in enumerate(self.schedulers):
            if hasattr(scheduler, 'lr_history'):
                history[f'scheduler_{i}'] = scheduler.lr_history
        return history