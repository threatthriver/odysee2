"""
Advanced Optimizers for Large-Scale Training

This module implements state-of-the-art optimizers including LAMB, AdaFactor,
8-bit optimizers, and adaptive learning rate methods specifically designed
for efficient training of large language models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer
import math
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import bitsandbytes as bnb
from collections import defaultdict
import warnings


class LAMBOptimizer(Optimizer):
    """
    LAMB (Layer-wise Adaptive Moments optimizer for Batch training) optimizer.
    
    LAMB is designed for large batch training and provides better convergence
    for very large batches while maintaining the benefits of adaptive methods.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        clamp_value: float = 10.0,
        always_adapt: bool = False
    ):
        """
        Initialize LAMB optimizer.
        
        Args:
            params: Parameters to optimize
            lr: Learning rate
            betas: Coefficients for computing running averages
            eps: Term added to denominator to avoid division by zero
            weight_decay: L2 regularization coefficient
            clamp_value: Maximum value for updates
            always_adapt: Whether to always use adaptive learning rate
        """
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            clamp_value=clamp_value,
            always_adapt=always_adapt
        )
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        """
        Perform a single optimization step with LAMB algorithm.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('LAMB does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                
                m, v = state['m'], state['v']
                beta1, beta2 = group['betas']
                
                # Update first moment estimate
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update second moment estimate
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                m_hat = m / (1 - beta1 ** state['step'])
                v_hat = v / (1 - beta2 ** state['step'])
                
                # L2 regularization
                if group['weight_decay'] > 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                # LAMB update
                update = m_hat / (torch.sqrt(v_hat) + group['eps'])
                
                # Optional clamping
                if group['clamp_value'] > 0:
                    update = torch.clamp(update, -group['clamp_value'], group['clamp_value'])
                
                # Adaptive learning rate
                if group['always_adapt']:
                    ratio = 1.0
                else:
                    p_norm = torch.norm(p.data)
                    update_norm = torch.norm(update)
                    
                    if p_norm == 0 or update_norm == 0:
                        ratio = 1.0
                    else:
                        ratio = p_norm / update_norm
                
                p.data.add_(update, alpha=-group['lr'] * ratio)
                state['step'] += 1
        
        return loss


class AdaFactorOptimizer(Optimizer):
    """
    AdaFactor optimizer for memory-efficient training.
    
    AdaFactor reduces memory usage by factorizing gradients and storing
    running averages in low-rank factorized forms.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: Tuple[float, float] = (1e-30, 1e-3),
        weight_decay: float = 0.0,
        min_dim_size_factor: int = 32,
        decay_rate: float = -0.8,
        clip_threshold: float = 1.0,
        relative_step: bool = True,
        scale_parameter: bool = True,
        warmup_init: bool = True
    ):
        """
        Initialize AdaFactor optimizer.
        
        Args:
            params: Parameters to optimize
            lr: Learning rate
            betas: Coefficients for running averages
            eps: Numerical stability parameters
            weight_decay: L2 regularization coefficient
            min_dim_size_factor: Minimum dimension for factorization
            decay_rate: Rate of decay for learning rate
            clip_threshold: Maximum value for gradient clipping
            relative_step: Whether to use relative step size
            scale_parameter: Whether to scale learning rate
            warmup_init: Whether to use warmup initialization
        """
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            min_dim_size_factor=min_dim_size_factor,
            decay_rate=decay_rate,
            clip_threshold=clip_threshold,
            relative_step=relative_step,
            scale_parameter=scale_parameter,
            warmup_init=warmup_init
        )
        super().__init__(params, defaults)
    
    def _get_lr_scales(self, group: Dict) -> Dict[str, float]:
        """Calculate learning rate scales."""
        if group['relative_step']:
            min_step = 1e-6 * group['step_size'] if group['warmup_init'] else 1e-2
            relative_step_size = min(0.5, math.sqrt(min_step / group['step_size']))
            lr_scale = relative_step_size * math.sqrt(group['step_size'])
        else:
            lr_scale = 1.0
        
        return {'rlr': relative_step_size if group['relative_step'] else None, 'lr_scale': lr_scale}
    
    def _factorize_tensor(self, tensor: torch.Tensor, min_dim_size: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Factorize tensor using low-rank approximation."""
        if tensor.ndim < 2 or min(tensor.shape) <= min_dim_size:
            return None
        
        # Simple SVD factorization
        try:
            U, S, Vh = torch.linalg.svd(tensor, full_matrices=False)
            # Keep only the most important singular values
            k = min(max(min_dim_size, tensor.shape[0] // 4), min(tensor.shape) - 1)
            U = U[:, :k]
            S = S[:k]
            Vh = Vh[:k, :]
            return U, Vh
        except:
            return None
    
    def step(self, closure=None):
        """
        Perform a single optimization step with AdaFactor algorithm.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr_scales = self._get_lr_scales(group)
            relative_step_size = lr_scales['rlr']
            lr_scale = lr_scales['lr_scale']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdaFactor does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    
                    # Initialize factorized forms
                    min_dim_size = group['min_dim_size_factor']
                    if grad.ndim >= 2:
                        u, v = self._factorize_tensor(grad, min_dim_size)
                        if u is not None:
                            state['exp_avg_u'] = torch.zeros_like(u)
                            state['exp_avg_v'] = torch.zeros_like(v)
                            state['exp_avg_sq_u'] = torch.zeros_like(u)
                            state['exp_avg_sq_v'] = torch.zeros_like(v)
                        else:
                            state['exp_avg_u'] = None
                            state['exp_avg_v'] = None
                            state['exp_avg_sq_u'] = None
                            state['exp_avg_sq_v'] = None
                
                # Update step
                state['step'] += 1
                beta1, beta2 = group['betas']
                
                # Update exponential averages
                if 'exp_avg_u' in state and state['exp_avg_u'] is not None:
                    # Factorized form
                    u, v = state['exp_avg_u'], state['exp_avg_v']
                    u_sq, v_sq = state['exp_avg_sq_u'], state['exp_avg_sq_v']
                    
                    u.mul_(beta1).add_(grad @ grad.T @ u, alpha=1 - beta1)
                    v.mul_(beta1).add_(grad.T @ grad @ v, alpha=1 - beta1)
                    u_sq.mul_(beta2).add_((grad @ grad.T @ u) ** 2, alpha=1 - beta2)
                    v_sq.mul_(beta2).add_((grad.T @ grad @ v) ** 2, alpha=1 - beta2)
                    
                    # Compute update
                    update = (u @ (v_sq ** -0.5) * ((1 - beta2) ** 0.5)) + group['eps'][0]
                    update = update / ((u_sq ** 0.5) * ((1 - beta2) ** 0.5) + group['eps'][1])
                    
                else:
                    # Standard form
                    state['exp_avg'].mul_(beta1).add_(grad, alpha=1 - beta1)
                    state['exp_avg_sq'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    
                    # Compute update
                    update = state['exp_avg'] / (torch.sqrt(state['exp_avg_sq']) + group['eps'][0])
                
                # L2 regularization
                if group['weight_decay'] > 0:
                    update.add_(p.data, alpha=group['weight_decay'])
                
                # Gradient clipping
                if group['clip_threshold'] > 0:
                    update = torch.clamp(update, -group['clip_threshold'], group['clip_threshold'])
                
                # Learning rate
                if group['relative_step']:
                    lr = relative_step_size
                else:
                    lr = group['lr']
                
                if group['scale_parameter']:
                    lr *= lr_scale
                
                # Update parameters
                p.data.add_(update, alpha=-lr)
        
        return loss


class BitsAndBytesOptimizer(Optimizer):
    """
    8-bit optimizer using bitsandbytes library for memory-efficient training.
    
    This optimizer stores optimizer states in 8-bit quantized form,
    reducing memory usage by up to 75% while maintaining training quality.
    """
    
    def __init__(
        self,
        params,
        optim_bits: int = 8,
        args: Any = None,
        config: Dict = None
    ):
        """
        Initialize 8-bit optimizer.
        
        Args:
            params: Parameters to optimize
            optim_bits: Number of bits for quantization (8 or 16)
            args: Arguments passed to underlying optimizer
            config: Configuration dictionary for 8-bit quantization
        """
        if config is None:
            config = {}
        
        # Create underlying optimizer
        if args is None:
            args = optim.AdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
        
        # Wrap with 8-bit quantization
        if optim_bits == 8:
            self.optimizer = bnb.optim.AdamW8bit(
                params,
                lr=args.defaults.get('lr', 1e-3),
                betas=args.defaults.get('betas', (0.9, 0.999)),
                eps=args.defaults.get('eps', 1e-8),
                **config
            )
        else:
            # Fallback to standard optimizer
            self.optimizer = args
        
        defaults = dict(
            optim_bits=optim_bits,
            config=config
        )
        super().__init__(params, defaults)
        
        # Store the actual optimizer
        self.actual_optimizer = self.optimizer if hasattr(self.optimizer, 'param_groups') else args
    
    def step(self, closure=None):
        """Perform a single optimization step."""
        if closure is not None:
            return self.actual_optimizer.step(closure)
        return self.actual_optimizer.step()
    
    def zero_grad(self, set_to_none: bool = False):
        """Zero out gradients."""
        self.actual_optimizer.zero_grad(set_to_none=set_to_none)
    
    def state_dict(self):
        """Return optimizer state dict."""
        return self.actual_optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load optimizer state dict."""
        return self.actual_optimizer.load_state_dict(state_dict)


class AdamW8Bit(Optimizer):
    """
    Custom 8-bit AdamW optimizer implementation.
    
    Provides memory-efficient AdamW optimization with 8-bit quantization
    for gradient moments and parameter storage.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        quant_bits: int = 8,
        quant_delay: int = 1000,
        quant_freq: int = 100
    ):
        """
        Initialize AdamW8Bit optimizer.
        
        Args:
            params: Parameters to optimize
            lr: Learning rate
            betas: Coefficients for running averages
            eps: Term for numerical stability
            weight_decay: L2 regularization coefficient
            quant_bits: Number of bits for quantization
            quant_delay: Step to start quantization
            quant_freq: Frequency of quantization updates
        """
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            quant_bits=quant_bits,
            quant_delay=quant_delay,
            quant_freq=quant_freq
        )
        super().__init__(params, defaults)
        
        # Quantization parameters
        self.quant_bits = quant_bits
        self.quant_delay = quant_delay
        self.quant_freq = quant_freq
        
        # Statistics
        self.stats = defaultdict(int)
    
    def _quantize_tensor(self, tensor: torch.Tensor, bits: int) -> torch.Tensor:
        """
        Simple quantization of tensor to specified number of bits.
        
        Args:
            tensor: Tensor to quantize
            bits: Number of bits for quantization
            
        Returns:
            Quantized tensor
        """
        if bits >= 16:
            return tensor
        
        # Dynamic range quantization
        max_val = tensor.abs().max()
        if max_val == 0:
            return tensor
        
        # Calculate scale
        scale = (2**(bits-1) - 1) / max_val
        
        # Quantize and dequantize
        quantized = torch.round(tensor * scale) / scale
        
        self.stats['quantizations'] += 1
        return quantized
    
    def _dequantize_tensor(self, quantized_tensor: torch.Tensor, bits: int) -> torch.Tensor:
        """
        Simple dequantization of tensor.
        
        Args:
            quantized_tensor: Quantized tensor
            bits: Original number of bits
            
        Returns:
            Dequantized tensor
        """
        return quantized_tensor  # Already stored in original scale
    
    def step(self, closure=None):
        """
        Perform a single optimization step with 8-bit AdamW.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW8Bit does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Apply quantization if conditions are met
                if (state['step'] >= self.quant_delay and 
                    state['step'] % self.quant_freq == 0):
                    
                    exp_avg = self._quantize_tensor(exp_avg, self.quant_bits)
                    exp_avg_sq = self._quantize_tensor(exp_avg_sq, self.quant_bits)
                    
                    self.stats['quantized_tensors'] += 1
                
                # Compute AdamW update
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1
                
                # L2 regularization
                if weight_decay > 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)
                
                # Update parameters
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
                # Store quantized states
                state['exp_avg'] = exp_avg
                state['exp_avg_sq'] = exp_avg_sq
        
        return loss


class AdaGradOptimizer(Optimizer):
    """
    AdaGrad optimizer with accumulated squared gradients.
    
    Adapted for large-scale training with memory-efficient implementation.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        lr_decay: float = 0.0,
        weight_decay: float = 0.0,
        initial_accumulator_value: float = 0.1,
        eps: float = 1e-10
    ):
        """
        Initialize AdaGrad optimizer.
        
        Args:
            params: Parameters to optimize
            lr: Learning rate
            lr_decay: Learning rate decay rate
            weight_decay: L2 regularization coefficient
            initial_accumulator_value: Initial value for accumulated squared gradients
            eps: Term for numerical stability
        """
        defaults = dict(
            lr=lr,
            lr_decay=lr_decay,
            weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value,
            eps=eps
        )
        super().__init__(params, defaults)
        
        # Initialize accumulators
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    state = self.state[p]
                    state['sum'] = torch.full_like(
                        p.data, initial_accumulator_value
                    )
    
    def share_memory(self):
        """Share memory across processes."""
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if 'sum' in state:
                    state['sum'].share_memory_()
    
    def step(self, closure=None):
        """
        Perform a single optimization step with AdaGrad algorithm.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            lr_decay = group['lr_decay']
            weight_decay = group['weight_decay']
            initial_accumulator_value = group['initial_accumulator_value']
            eps = group['eps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdaGrad does not support sparse gradients')
                
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['sum'] = torch.full_like(
                        p.data, initial_accumulator_value
                    )
                
                state['step'] += 1
                
                # Update accumulated squared gradients
                state['sum'].addcmul_(grad, grad, value=1)
                
                # Apply weight decay
                if weight_decay > 0:
                    grad = grad.add(p.data, alpha=weight_decay)
                
                # Update parameters
                denom = state['sum'].sqrt().add_(eps)
                p.data.addcdiv_(grad, denom, value=-lr)
        
        return loss


class AdaptiveLRScheduler:
    """
    Adaptive Learning Rate Scheduler that adjusts learning rate based on
    training dynamics and model performance.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        base_lr: float = 1e-3,
        min_lr: float = 1e-6,
        max_lr: float = 1e-1,
        patience: int = 10,
        factor: float = 0.1,
        mode: str = 'min',
        window_size: int = 100,
        smoothing: float = 0.1
    ):
        """
        Initialize adaptive learning rate scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            base_lr: Base learning rate
            min_lr: Minimum learning rate
            max_lr: Maximum learning rate
            patience: Number of epochs to wait for improvement
            factor: Factor to reduce learning rate by
            mode: 'min' or 'max' for monitoring
            window_size: Window size for moving average
            smoothing: Smoothing factor for moving average
        """
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.patience = patience
        self.factor = factor
        self.mode = mode
        self.window_size = window_size
        self.smoothing = smoothing
        
        # State
        self.step_count = 0
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.wait_count = 0
        self.metric_history = []
        self.smoothed_metric = None
        
        # Update learning rates
        self._update_lr(base_lr)
    
    def _update_lr(self, new_lr: float):
        """Update learning rate in optimizer."""
        new_lr = max(self.min_lr, min(self.max_lr, new_lr))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
    
    def update(self, metric: float) -> float:
        """
        Update learning rate based on metric.
        
        Args:
            metric: Current metric value
            
        Returns:
            Updated learning rate
        """
        self.step_count += 1
        self.metric_history.append(metric)
        
        # Keep only recent history
        if len(self.metric_history) > self.window_size:
            self.metric_history.pop(0)
        
        # Calculate smoothed metric
        if self.smoothed_metric is None:
            self.smoothed_metric = metric
        else:
            self.smoothed_metric = (
                self.smoothing * metric + 
                (1 - self.smoothing) * self.smoothed_metric
            )
        
        # Check for improvement
        improved = (
            (self.mode == 'min' and self.smoothed_metric < self.best_metric) or
            (self.mode == 'max' and self.smoothed_metric > self.best_metric)
        )
        
        if improved:
            self.best_metric = self.smoothed_metric
            self.wait_count = 0
        else:
            self.wait_count += 1
        
        # Adjust learning rate
        if self.wait_count >= self.patience:
            current_lr = self.optimizer.param_groups[0]['lr']
            new_lr = current_lr * self.factor
            self._update_lr(new_lr)
            self.wait_count = 0
            
            # Log the adjustment
            print(f"Reducing learning rate to {new_lr:.2e} at step {self.step_count}")
        
        return self.optimizer.param_groups[0]['lr']
    
    def step(self) -> float:
        """
        Perform a step in the scheduler.
        
        Returns:
            Current learning rate
        """
        self.step_count += 1
        return self.optimizer.param_groups[0]['lr']
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
    
    def state_dict(self) -> Dict:
        """Return scheduler state dict."""
        return {
            'step_count': self.step_count,
            'best_metric': self.best_metric,
            'wait_count': self.wait_count,
            'metric_history': self.metric_history,
            'smoothed_metric': self.smoothed_metric
        }
    
    def load_state_dict(self, state_dict: Dict):
        """Load scheduler state dict."""
        self.step_count = state_dict.get('step_count', 0)
        self.best_metric = state_dict.get('best_metric', float('inf') if self.mode == 'min' else float('-inf'))
        self.wait_count = state_dict.get('wait_count', 0)
        self.metric_history = state_dict.get('metric_history', [])
        self.smoothed_metric = state_dict.get('smoothed_metric', None)


# Utility functions for optimizer selection and management

def create_optimizer(
    optimizer_type: str,
    model_params: List[nn.Parameter],
    lr: float = 1e-3,
    **kwargs
) -> Optimizer:
    """
    Factory function to create optimizers.
    
    Args:
        optimizer_type: Type of optimizer ('lamb', 'adafactor', 'adamw8bit', 'adamw', 'adagrad')
        model_params: Model parameters
        lr: Learning rate
        **kwargs: Additional optimizer-specific arguments
        
    Returns:
        Optimizer instance
    """
    if optimizer_type == 'lamb':
        return LAMBOptimizer(model_params, lr=lr, **kwargs)
    elif optimizer_type == 'adafactor':
        return AdaFactorOptimizer(model_params, lr=lr, **kwargs)
    elif optimizer_type == 'adamw8bit':
        return AdamW8Bit(model_params, lr=lr, **kwargs)
    elif optimizer_type == 'adagrad':
        return AdaGradOptimizer(model_params, lr=lr, **kwargs)
    elif optimizer_type == 'adamw':
        return optim.AdamW(model_params, lr=lr, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def get_memory_efficient_optimizer(
    model_size: int,
    available_memory: int,
    training_mode: str = 'pre_training'
) -> Tuple[Optimizer, Dict]:
    """
    Select optimal optimizer based on model size and available memory.
    
    Args:
        model_size: Number of parameters in millions
        available_memory: Available GPU memory in MB
        training_mode: 'pre_training' or 'fine_tuning'
        
    Returns:
        Tuple of (optimizer, config)
    """
    # Memory thresholds in MB
    memory_thresholds = {
        'low': 8000,      # 8GB
        'medium': 16000,  # 16GB
        'high': 32000     # 32GB
    }
    
    # Determine memory level
    if available_memory < memory_thresholds['low']:
        memory_level = 'low'
        optimizer_type = 'adafactor'
        config = {'optim_bits': 8, 'quant_delay': 100, 'quant_freq': 50}
    elif available_memory < memory_thresholds['medium']:
        memory_level = 'medium'
        if training_mode == 'pre_training':
            optimizer_type = 'adamw8bit'
            config = {'optim_bits': 8, 'quant_delay': 500, 'quant_freq': 100}
        else:
            optimizer_type = 'adamw'
            config = {}
    else:
        memory_level = 'high'
        if model_size > 1000:  # 1B+ parameters
            optimizer_type = 'lamb'
            config = {'always_adapt': True}
        else:
            optimizer_type = 'adamw'
            config = {}
    
    return optimizer_type, config


def benchmark_optimizers(
    model_params: List[nn.Parameter],
    optimizer_configs: Dict[str, Dict],
    memory_budget: int
) -> Dict[str, Dict]:
    """
    Benchmark different optimizers on given model parameters.
    
    Args:
        model_params: Model parameters
        optimizer_configs: Dictionary of optimizer configurations
        memory_budget: Available memory budget in MB
        
    Returns:
        Benchmark results
    """
    results = {}
    
    for opt_name, config in optimizer_configs.items():
        try:
            # Create optimizer
            optimizer = create_optimizer(opt_name, model_params, **config)
            
            # Simulate memory usage
            if '8bit' in opt_name or 'adafactor' in opt_name:
                memory_usage = len(model_params) * 4  # Estimated 8-bit usage
            else:
                memory_usage = len(model_params) * 12  # Standard 32-bit usage
            
            results[opt_name] = {
                'memory_usage_mb': memory_usage * 4 / (1024 * 1024),  # Convert to MB
                'fits_in_budget': memory_usage * 4 <= memory_budget * 1024 * 1024,
                'estimated_speed': 'fast' if memory_usage < memory_budget else 'slow'
            }
            
        except Exception as e:
            results[opt_name] = {
                'error': str(e),
                'fits_in_budget': False,
                'estimated_speed': 'unknown'
            }
    
    return results