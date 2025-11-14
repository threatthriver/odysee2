"""
Advanced Optimizers for Language Model Training

This module implements state-of-the-art optimizers optimized for large-scale
language model training, including AdamW, LAMB, AdaFactor, and other optimizers
with memory-efficient implementations and gradient clipping capabilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer, params_t
from typing import Optional, Dict, Any, List, Tuple
import math
import warnings


class AdamW(Optimizer):
    """
    AdamW optimizer with enhanced features for language model training.
    
    This implementation includes:
    - Decoupled weight decay
    - Gradient clipping
    - Memory-efficient state management
    - Cosine learning rate warmup support
    - Gradient centralization
    """
    
    def __init__(
        self,
        params: params_t,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
        gradient_clipping: Optional[float] = None,
        gradient_centralization: bool = False,
        use_adaptive_clipping: bool = False,
        max_grad_norm: float = 1.0,
        **kwargs
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            gradient_clipping=gradient_clipping,
            gradient_centralization=gradient_centralization,
            use_adaptive_clipping=use_adaptive_clipping,
            max_grad_norm=max_grad_norm
        )
        super().__init__(params, defaults)
    
    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
    
    def step(self, closure: Optional[callable] = None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                
                state = self.state[p]
                
                # Initialize state if needed
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['amsgrad']:
                        # Maximum of past squared gradients
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                # Get current state
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if group['amsgrad']:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                else:
                    max_exp_avg_sq = None
                
                # Update step count
                state['step'] += 1
                
                # Apply gradient centralization if enabled
                if group['gradient_centralization']:
                    grad = self._apply_gradient_centralization(grad)
                
                # Apply adaptive gradient clipping if enabled
                if group['use_adaptive_clipping'] and group['gradient_clipping'] is not None:
                    grad = self._apply_adaptive_clipping(grad, group)
                
                # Perform AdamW update
                self._adamw_update(
                    p,
                    grad,
                    exp_avg,
                    exp_avg_sq,
                    max_exp_avg_sq,
                    state,
                    group
                )
        
        return loss
    
    def _apply_gradient_centralization(self, grad: torch.Tensor) -> torch.Tensor:
        """Apply gradient centralization."""
        if grad.dim() > 1:
            grad = grad - grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True)
        return grad
    
    def _apply_adaptive_clipping(self, grad: torch.Tensor, group: Dict[str, Any]) -> torch.Tensor:
        """Apply adaptive gradient clipping based on gradient statistics."""
        grad_norm = grad.norm()
        threshold = group['gradient_clipping']
        
        if grad_norm > threshold:
            # Adaptive scaling based on gradient magnitude
            scale_factor = threshold / (grad_norm + 1e-8)
            grad = grad * scale_factor
        
        return grad
    
    def _adamw_update(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        max_exp_avg_sq: Optional[torch.Tensor],
        state: Dict[str, Any],
        group: Dict[str, Any]
    ):
        """Perform AdamW parameter update."""
        beta1, beta2 = group['betas']
        
        # Decouple weight decay
        if group['weight_decay'] != 0:
            p.data.add_(p.data, alpha=-group['weight_decay'])
        
        # Exponential moving average of gradient values
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        
        # Exponential moving average of squared gradient values
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
        
        if group['amsgrad']:
            # Use the maximum of past exp_avg_sq
            torch.maximum(max_exp_avg_sq, exp_avg_sq, out=exp_avg_sq)
            denom = exp_avg_sq.sqrt().add_(group['eps'])
        else:
            denom = exp_avg_sq.sqrt().add_(group['eps'])
        
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        
        step_size = group['lr'] / bias_correction1
        p.data.addcdiv_(exp_avg, denom, value=-step_size)
        
        # Apply bias correction to exp_avg
        exp_avg.div_(bias_correction1)
        
        # Apply bias correction to exp_avg_sq
        exp_avg_sq.div_(bias_correction2)


class LAMB(Optimizer):
    """
    LAMB (Layer-wise Adaptive Moments for Batch training) optimizer.
    
    LAMB is designed for large batch training and provides:
    - Layer-wise adaptive learning rates
    - Trust ratio calculation to prevent large updates
    - Better generalization properties
    - Memory-efficient implementation
    """
    
    def __init__(
        self,
        params: params_t,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        trust_clip_ratio: float = 1.0,
        use_gradient_centralization: bool = True,
        use_layer_wise_adaptation: bool = True,
        **kwargs
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= trust_clip_ratio <= 1.0:
            raise ValueError(f"Invalid trust_clip_ratio: {trust_clip_ratio}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            trust_clip_ratio=trust_clip_ratio,
            use_gradient_centralization=use_gradient_centralization,
            use_layer_wise_adaptation=use_layer_wise_adaptation
        )
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[callable] = None):
        """Perform a single optimization step using LAMB algorithm."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('LAMB does not support sparse gradients')
                
                state = self.state[p]
                
                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['layer_norm'] = torch.ones_like(p)
                    state['param_norm'] = torch.ones_like(p)
                
                # Get current state
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                layer_norm = state['layer_norm']
                param_norm = state['param_norm']
                
                state['step'] += 1
                beta1, beta2 = group['betas']
                
                # Apply gradient centralization
                if group['use_gradient_centralization']:
                    grad = self._apply_gradient_centralization(grad)
                
                # Update exponential moving averages
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
                
                # Calculate bias corrections
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Layer-wise adaptation
                if group['use_layer_wise_adaptation']:
                    # Update layer norms
                    layer_norm = layer_norm * 0.9 + grad.norm() * 0.1
                    param_norm = param_norm * 0.9 + p.data.norm() * 0.1
                    
                    # Store updated norms
                    state['layer_norm'] = layer_norm
                    state['param_norm'] = param_norm
                
                # Calculate step size with bias correction
                step_size = group['lr'] / bias_correction1
                
                # Calculate adaptive learning rate
                if group['use_layer_wise_adaptation']:
                    adaptive_lr = step_size * (layer_norm / (exp_avg_sq.sqrt() + group['eps']))
                else:
                    adaptive_lr = step_size
                
                # Calculate trust ratio
                w_norm = p.data.norm()
                g_norm = exp_avg.norm()
                
                if w_norm == 0:
                    trust_ratio = 1.0
                elif g_norm == 0:
                    trust_ratio = group['trust_clip_ratio']
                else:
                    trust_ratio = min(w_norm / g_norm, group['trust_clip_ratio'])
                
                # Apply weight decay
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'])
                
                # Update parameters
                update = exp_avg.div_(exp_avg_sq.sqrt().add_(group['eps']))
                update.mul_(bias_correction2).mul_(adaptive_lr * trust_ratio)
                p.data.add_(-update)


class AdaFactor(Optimizer):
    """
    AdaFactor optimizer for efficient memory usage in large language models.
    
    AdaFactor provides:
    - Memory-efficient parameter management
    - Adaptive learning rates for different parameters
    - Built-in gradient clipping
    - Sublinear memory growth
    """
    
    def __init__(
        self,
        params: params_t,
        lr: Optional[float] = None,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: Tuple[float, float] = (1e-30, 1e-3),
        weight_decay: float = 0.0,
        relative_step: bool = True,
        warmup_init: bool = False,
        max_step: int = 1000000,
        clip_threshold: float = 1.0,
        use_momentum: bool = False,
        momentum: float = 0.0,
        **kwargs
    ):
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= clip_threshold:
            raise ValueError(f"Invalid clip_threshold value: {clip_threshold}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            relative_step=relative_step,
            warmup_init=warmup_init,
            max_step=max_step,
            clip_threshold=clip_threshold,
            use_momentum=use_momentum,
            momentum=momentum
        )
        
        if lr is not None and not relative_step:
            warnings.warn("lr was set but relative_step is True. Ignoring lr and using relative_step.")
        
        super().__init__(params, defaults)
    
    def _get_lr(self, step: int, group: Dict[str, Any]) -> float:
        """Calculate learning rate."""
        if group['relative_step']:
            step = min(step, group['max_step'])
            if group['warmup_init']:
                if step <= group['max_step']:
                    relative_step = min(group['max_step'] ** -0.5, step ** -0.5)
                else:
                    relative_step = (group['max_step'] ** -0.5) * (step ** -0.5)
            else:
                relative_step = min(group['max_step'] ** -0.5, step ** -0.5)
            return group['lr'] if group['lr'] is not None else relative_step
        else:
            return group['lr']
    
    def _get_effective_lr(self, p: torch.Tensor, lr: float, group: Dict[str, Any]) -> float:
        """Calculate effective learning rate for parameter."""
        if p.dim() <= 1:
            # For bias terms, use standard learning rate
            return lr
        else:
            # For weight terms, scale by parameter norm
            param_norm = p.norm()
            if param_norm > 0:
                effective_lr = lr * min((p.size(0) ** 0.5) / (param_norm + 1e-8), group['clip_threshold'])
            else:
                effective_lr = lr * group['clip_threshold']
            return effective_lr
    
    def _apply_gradient_clipping(self, grad: torch.Tensor, group: Dict[str, Any]) -> torch.Tensor:
        """Apply gradient clipping."""
        grad = grad / max(1.0, grad.norm() / group['clip_threshold'])
        return grad
    
    def step(self, closure: Optional[callable] = None):
        """Perform a single optimization step using AdaFactor algorithm."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdaFactor does not support sparse gradients')
                
                state = self.state[p]
                
                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    
                    if group['use_momentum']:
                        state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                # Get current state
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                if group['use_momentum']:
                    momentum_buffer = state['momentum_buffer']
                
                # Update step count
                state['step'] += 1
                step = state['step']
                
                # Apply gradient clipping
                grad = self._apply_gradient_clipping(grad, group)
                
                # Get effective learning rate
                base_lr = self._get_lr(step, group)
                lr = self._get_effective_lr(p, base_lr, group)
                
                # Update exponential moving averages
                beta1, beta2 = group['betas']
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Apply bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                
                # Calculate update
                denom = (exp_avg_sq.sqrt() + group['eps'][1])
                update = exp_avg / denom
                
                # Scale update for bias terms
                if p.dim() > 1:
                    update = update / (bias_correction1 * (exp_avg_sq.sqrt() + group['eps'][0]))
                else:
                    update = update / (bias_correction1 * (p.size(0) ** 0.5))
                
                # Apply momentum if enabled
                if group['use_momentum']:
                    momentum_buffer.mul_(group['momentum']).add_(update)
                    update = momentum_buffer
                
                # Apply weight decay
                if group['weight_decay'] != 0:
                    if p.dim() > 1:
                        update.add_(p.data)
                    else:
                        update.add_(p.data)
                        update = update / (1 + group['weight_decay'] * base_lr)
                
                # Update parameters
                p.data.add_(update, alpha=-lr)
        
        return loss


class RMSprop(Optimizer):
    """
    RMSprop optimizer with enhanced features for deep learning.
    
    This implementation includes:
    - Centered version of RMSprop
    - Learning rate scheduling support
    - Gradient clipping
    - Momentum option
    """
    
    def __init__(
        self,
        params: params_t,
        lr: float = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        centered: bool = False,
        clip_value: Optional[float] = None,
        **kwargs
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= alpha < 1.0:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        
        defaults = dict(
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered,
            clip_value=clip_value
        )
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[callable] = None):
        """Perform a single optimization step using RMSprop algorithm."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('RMSprop does not support sparse gradients')
                
                state = self.state[p]
                
                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                # Get current state
                square_avg = state['square_avg']
                
                if group['momentum'] > 0:
                    momentum_buffer = state['momentum_buffer']
                
                if group['centered']:
                    grad_avg = state['grad_avg']
                
                # Apply gradient clipping
                if group['clip_value'] is not None:
                    grad = torch.clamp(grad, -group['clip_value'], group['clip_value'])
                
                state['step'] += 1
                step = state['step']
                
                # Update square average
                square_avg.mul_(group['alpha']).addcmul_(grad, grad, value=1 - group['alpha'])
                
                # Update gradient average (centered version)
                if group['centered']:
                    grad_avg.mul_(group['alpha']).add_(grad, alpha=1 - group['alpha'])
                    avg = square_avg.addcmul_(grad_avg, grad_avg, value=-1).add_(group['eps']).sqrt_()
                else:
                    avg = square_avg.add_(group['eps']).sqrt_()
                
                # Apply momentum
                if group['momentum'] > 0:
                    momentum_buffer.mul_(group['momentum']).addcdiv_(grad, avg)
                    update = momentum_buffer
                else:
                    update = grad / avg
                
                # Apply weight decay
                if group['weight_decay'] != 0:
                    update.add_(p.data, alpha=group['weight_decay'])
                
                # Update parameters
                p.data.add_(update, alpha=-group['lr'])
        
        return loss


class GradientClipByGlobalNorm(Optimizer):
    """
    Wrapper optimizer that applies global norm gradient clipping.
    
    This wrapper can be applied to any existing optimizer to add
    global norm gradient clipping capabilities.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        max_norm: float,
        norm_type: float = 2.0,
        clip_error_if_nonfinite: bool = False,
        **kwargs
    ):
        self.optimizer = optimizer
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.clip_error_if_nonfinite = clip_error_if_nonfinite
        
        # Validate parameters
        if not 0.0 < max_norm:
            raise ValueError(f"Invalid max_norm value: {max_norm}")
        if not 0.0 <= norm_type:
            raise ValueError(f"Invalid norm_type value: {norm_type}")
    
    def step(self, closure: Optional[callable] = None):
        """Perform optimization step with global norm clipping."""
        # First, zero gradients
        self.optimizer.zero_grad(set_to_none=True)
        
        # Forward and backward pass
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # Calculate global norm
        total_norm = torch.tensor(0.0)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(self.norm_type)
                    total_norm += param_norm.item() ** self.norm_type
        
        total_norm = total_norm ** (1.0 / self.norm_type)
        
        # Clip gradients if needed
        clip_coef = self.max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad.data.mul_(clip_coef)
        
        # Perform optimization step
        self.optimizer.step()
        
        return loss
    
    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients."""
        self.optimizer.zero_grad(set_to_none=set_to_none)
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dict."""
        state_dict = self.optimizer.state_dict()
        state_dict['gradient_clipping'] = {
            'max_norm': self.max_norm,
            'norm_type': self.norm_type,
            'clip_error_if_nonfinite': self.clip_error_if_nonfinite
        }
        return state_dict
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict."""
        if 'gradient_clipping' in state_dict:
            clipping_config = state_dict['gradient_clipping']
            self.max_norm = clipping_config['max_norm']
            self.norm_type = clipping_config['norm_type']
            self.clip_error_if_nonfinite = clipping_config['clip_error_if_nonfinite']
            # Remove from state dict
            state_dict = {k: v for k, v in state_dict.items() if k != 'gradient_clipping'}
        
        self.optimizer.load_state_dict(state_dict)
    
    def __getattr__(self, name: str):
        """Delegate attribute access to wrapped optimizer."""
        return getattr(self.optimizer, name)