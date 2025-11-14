"""
Mixed Precision Training Infrastructure

This module implements efficient mixed precision training for large language models,
including automatic mixed precision (AMP), gradient scaling, and memory optimization
techniques specifically designed for transformer architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from typing import Optional, Dict, Any, List, Union
import warnings
import time
from contextlib import contextmanager


class MixedPrecisionTrainer:
    """
    Mixed Precision Trainer for efficient large-scale language model training.
    
    This trainer provides:
    - Automatic Mixed Precision (AMP) with autocast
    - Gradient scaling for numerical stability
    - Memory optimization techniques
    - Performance monitoring and adaptation
    """
    
    def __init__(
        self,
        enabled: bool = True,
        grad_scaler_init_scale: float = 2.0**16,
        grad_scaler_growth_factor: float = 2.0,
        grad_scaler_backoff_factor: float = 0.5,
        grad_scaler_growth_interval: int = 2000,
        grad_clip_max_norm: Optional[float] = 1.0,
        grad_clip_norm_type: float = 2.0,
        opt_level: str = 'O1',  # 'O0', 'O1', 'O2'
        cast_model_outputs: Optional[List[str]] = None,
        memory_efficient: bool = True,
        **kwargs
    ):
        self.enabled = enabled
        self.opt_level = opt_level
        self.memory_efficient = memory_efficient
        self.grad_clip_max_norm = grad_clip_max_norm
        self.grad_clip_norm_type = grad_clip_norm_type
        
        # Initialize gradient scaler
        if enabled:
            self.scaler = GradScaler(
                init_scale=grad_scaler_init_scale,
                growth_factor=grad_scaler_growth_factor,
                backoff_factor=grad_scaler_backoff_factor,
                growth_interval=grad_scaler_growth_interval,
                enabled=True
            )
        else:
            self.scaler = None
        
        # Cast model outputs configuration
        self.cast_model_outputs = cast_model_outputs or []
        
        # Performance tracking
        self.step_count = 0
        self.autocast_time = 0.0
        self.scaler_time = 0.0
        self.memory_saved = 0.0
        
        # Validation
        if opt_level not in ['O0', 'O1', 'O2']:
            raise ValueError(f"Unsupported opt_level: {opt_level}. Choose from 'O0', 'O1', 'O2'")
    
    @contextmanager
    def autocast_context(self):
        """Context manager for automatic mixed precision."""
        if not self.enabled:
            # No autocast, return the same context
            yield
            return
        
        start_time = time.perf_counter()
        
        # Apply opt_level specific optimizations
        if self.opt_level == 'O0':
            # No optimization, pure float32
            yield
        elif self.opt_level == 'O1':
            # Cast only appropriate layers to float16
            with autocast(enabled=True):
                yield
        elif self.opt_level == 'O2':
            # More aggressive casting
            with autocast(enabled=True, dtype=torch.float16):
                yield
        
        self.autocast_time += time.perf_counter() - start_time
    
    def train_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        inputs: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform a single training step with mixed precision.
        
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            loss_fn: Loss function
            inputs: Input tensors
            labels: Target labels
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with training metrics
        """
        model.train()
        optimizer.zero_grad()
        
        start_time = time.perf_counter()
        
        # Forward pass with mixed precision
        with self.autocast_context():
            if labels is not None:
                outputs = model(**inputs)
                if isinstance(outputs, dict):
                    loss = loss_fn(outputs['logits'], labels)
                else:
                    loss = loss_fn(outputs, labels)
            else:
                outputs = model(**inputs)
                if isinstance(outputs, dict) and 'loss' in outputs:
                    loss = outputs['loss']
                else:
                    raise ValueError("Either provide labels or ensure model returns loss")
        
        # Backward pass with gradient scaling
        if self.enabled and self.scaler is not None:
            scaler_start = time.perf_counter()
            
            # Scale loss for gradient accumulation
            self.scaler.scale(loss).backward()
            
            # Unscale gradients and step
            self.scaler.unscale_(optimizer)
            
            # Gradient clipping
            if self.grad_clip_max_norm is not None:
                self.scaler.clip_grad_norm_(
                    model.parameters(),
                    max_norm=self.grad_clip_max_norm,
                    norm_type=self.grad_clip_norm_type
                )
            
            # Step optimizer
            scaler_result = self.scaler.step(optimizer)
            self.scaler.update()
            
            self.scaler_time += time.perf_counter() - scaler_start
        else:
            # Standard FP32 training
            loss.backward()
            
            # Gradient clipping
            if self.grad_clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=self.grad_clip_max_norm,
                    norm_type=self.grad_clip_norm_type
                )
            
            optimizer.step()
            scaler_result = None
        
        step_time = time.perf_counter() - start_time
        self.step_count += 1
        
        # Prepare outputs
        result = {
            'loss': loss.item(),
            'step_time': step_time,
            'autocast_time': self.autocast_time,
            'scaler_time': self.scaler_time if self.enabled else 0.0,
            'step_count': self.step_count,
            'scaler_scale': self.scaler.get_scale() if self.enabled and self.scaler else 1.0,
            'model_outputs': self._process_model_outputs(outputs)
        }
        
        if scaler_result is not None:
            result['scaler_result'] = scaler_result
        
        return result
    
    def validation_step(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        inputs: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform a validation step with mixed precision.
        
        Args:
            model: PyTorch model
            loss_fn: Loss function
            inputs: Input tensors
            labels: Target labels
            
        Returns:
            Dictionary with validation metrics
        """
        model.eval()
        
        with torch.no_grad():
            with self.autocast_context():
                outputs = model(**inputs)
                
                if labels is not None:
                    if isinstance(outputs, dict):
                        loss = loss_fn(outputs['logits'], labels)
                    else:
                        loss = loss_fn(outputs, labels)
                elif isinstance(outputs, dict) and 'loss' in outputs:
                    loss = outputs['loss']
                else:
                    loss = None
        
        result = {
            'loss': loss.item() if loss is not None else 0.0,
            'model_outputs': self._process_model_outputs(outputs)
        }
        
        return result
    
    def _process_model_outputs(self, outputs):
        """Process and cast model outputs as needed."""
        if isinstance(outputs, dict):
            processed = {}
            for key, value in outputs.items():
                if key in self.cast_model_outputs:
                    if torch.is_tensor(value):
                        processed[key] = value.float()
                    else:
                        processed[key] = value
                else:
                    processed[key] = value
            return processed
        else:
            return outputs
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory statistics for mixed precision training."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            
            return {
                'memory_allocated_gb': allocated,
                'memory_reserved_gb': reserved,
                'memory_efficiency': allocated / reserved if reserved > 0 else 1.0
            }
        else:
            return {'error': 'CUDA not available'}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'step_count': self.step_count,
            'avg_autocast_time': self.autocast_time / self.step_count if self.step_count > 0 else 0.0,
            'avg_scaler_time': self.scaler_time / self.step_count if self.step_count > 0 else 0.0,
            'enabled': self.enabled,
            'opt_level': self.opt_level,
            'scaler_scale': self.scaler.get_scale() if self.enabled and self.scaler else 1.0
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.step_count = 0
        self.autocast_time = 0.0
        self.scaler_time = 0.0
        self.memory_saved = 0.0


class MemoryEfficientMixedPrecision(MixedPrecisionTrainer):
    """
    Memory-efficient mixed precision trainer with additional optimizations.
    
    This trainer includes:
    - Gradient checkpointing integration
    - Activation recomputation
    - Memory-aware batch sizing
    - Dynamic precision adjustment
    """
    
    def __init__(
        self,
        enable_gradient_checkpointing: bool = True,
        activation_recompute_frequency: int = 1,
        memory_threshold: float = 0.8,
        adaptive_precision: bool = True,
        low_precision_threshold: float = 1e-5,
        high_precision_threshold: float = 1e3,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.activation_recompute_frequency = activation_recompute_frequency
        self.memory_threshold = memory_threshold
        self.adaptive_precision = adaptive_precision
        self.low_precision_threshold = low_precision_threshold
        self.high_precision_threshold = high_precision_threshold
        
        self.recompute_count = 0
        self.precision_adjustments = 0
        self.memory_pressure_detected = False
        
        if enable_gradient_checkpointing:
            self._setup_gradient_checkpointing()
    
    def _setup_gradient_checkpointing(self):
        """Setup gradient checkpointing for memory efficiency."""
        self.checkpoint_activations = torch.utils.checkpoint.checkpoint_activations
        self.checkpoint_activations.set_activations_recompute
        
        # Enable activation checkpointing
        torch.utils.checkpoint.set_recompute_defaults(
            use_reentrant=False,
            check_trainable_in_checkpoint=True
        )
    
    def train_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        inputs: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform training step with memory-efficient mixed precision.
        """
        # Check memory pressure
        if self.adaptive_precision:
            self._monitor_memory_pressure(model)
        
        # Forward pass with potential activation recomputation
        with self.autocast_context():
            if self.enable_gradient_checkpointing and self.training:
                outputs = self._forward_with_checkpointing(model, inputs)
            else:
                outputs = model(**inputs)
        
        # Continue with standard training step...
        result = super().train_step(model, optimizer, loss_fn, inputs, labels, **kwargs)
        
        # Add memory-efficient specific metrics
        result.update({
            'recompute_count': self.recompute_count,
            'precision_adjustments': self.precision_adjustments,
            'memory_pressure_detected': self.memory_pressure_detected
        })
        
        return result
    
    def _forward_with_checkpointing(self, model: nn.Module, inputs: Dict[str, torch.Tensor]):
        """Forward pass with activation checkpointing."""
        if self.recompute_count % self.activation_recompute_frequency == 0:
            # Compute activations normally
            outputs = model(**inputs)
        else:
            # Use checkpointing to save memory
            if hasattr(model, 'gradient_checkpointing_enable'):
                with model.gradient_checkpointing_enable():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
        
        self.recompute_count += 1
        return outputs
    
    def _monitor_memory_pressure(self, model: nn.Module):
        """Monitor memory pressure and adjust precision if needed."""
        if not torch.cuda.is_available():
            return
        
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        
        if reserved == 0:
            return
        
        memory_ratio = allocated / reserved
        
        if memory_ratio > self.memory_threshold:
            self.memory_pressure_detected = True
            
            # Check if gradients are too small (underflow risk)
            total_grad_norm = 0.0
            param_count = 0
            
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm += grad_norm ** 2
                    param_count += 1
            
            if param_count > 0:
                avg_grad_norm = (total_grad_norm / param_count) ** 0.5
                
                if avg_grad_norm < self.low_precision_threshold:
                    # Gradients too small, reduce precision requirements
                    self._adjust_precision_requirements('reduce')
                elif avg_grad_norm > self.high_precision_threshold:
                    # Gradients too large, increase precision requirements
                    self._adjust_precision_requirements('increase')
    
    def _adjust_precision_requirements(self, direction: str):
        """Adjust precision requirements based on gradient statistics."""
        if not self.adaptive_precision:
            return
        
        if direction == 'reduce' and self.scaler:
            # Reduce the scale to prevent underflow
            current_scale = self.scaler.get_scale()
            new_scale = max(current_scale * 0.5, 1e-4)
            self.scaler._scale = torch.tensor(new_scale, device=self.scaler._scale.device)
            
        elif direction == 'increase' and self.scaler:
            # Increase the scale to prevent overflow
            current_scale = self.scaler.get_scale()
            new_scale = min(current_scale * 2.0, 65536.0)
            self.scaler._scale = torch.tensor(new_scale, device=self.scaler._scale.device)
        
        self.precision_adjustments += 1
        print(f"Precision adjustment ({direction}): new scale = {self.scaler.get_scale() if self.scaler else 'N/A'}")
    
    def get_memory_efficient_stats(self) -> Dict[str, Any]:
        """Get memory-efficient specific statistics."""
        base_stats = self.get_performance_stats()
        
        if torch.cuda.is_available():
            base_stats.update(self.get_memory_stats())
        
        base_stats.update({
            'recompute_count': self.recompute_count,
            'precision_adjustments': self.precision_adjustments,
            'memory_pressure_detected': self.memory_pressure_detected,
            'gradient_checkpointing_enabled': self.enable_gradient_checkpointing,
            'adaptive_precision_enabled': self.adaptive_precision
        })
        
        return base_stats


class AMPOptimizerWrapper:
    """
    Wrapper for optimizers to work seamlessly with mixed precision training.
    
    This wrapper handles:
    - Gradient unscaling
    - Parameter casting
    - Gradient overflow/underflow detection
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[GradScaler] = None,
        **kwargs
    ):
        self.optimizer = optimizer
        self.scaler = scaler
        
        # Cast parameters to appropriate dtype
        self._cast_parameters()
    
    def _cast_parameters(self):
        """Cast optimizer parameters to appropriate dtype."""
        # This is handled automatically by PyTorch's AMP system
        pass
    
    def step(self, closure: Optional[callable] = None, **kwargs):
        """Optimized step function with gradient scaling."""
        if self.scaler is not None:
            # Step with gradient scaler
            if closure is not None:
                self.scaler.step(self.optimizer, closure)
            else:
                self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard step
            self.optimizer.step(closure)
    
    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients."""
        self.optimizer.zero_grad(set_to_none=set_to_none)
    
    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state dict including scaler state."""
        state = self.optimizer.state_dict()
        if self.scaler is not None:
            state['scaler'] = self.scaler.state_dict()
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load optimizer state dict including scaler state."""
        if 'scaler' in state_dict and self.scaler is not None:
            self.scaler.load_state_dict(state_dict['scaler'])
            # Remove scaler from optimizer state to avoid duplicate loading
            state_dict = {k: v for k, v in state_dict.items() if k != 'scaler'}
        
        self.optimizer.load_state_dict(state_dict)
    
    def __getattr__(self, name: str):
        """Delegate attribute access to the wrapped optimizer."""
        return getattr(self.optimizer, name)


class PrecisionProfiler:
    """
    Profiler for mixed precision training metrics.
    
    Tracks precision-related metrics and provides insights for optimization.
    """
    
    def __init__(self):
        self.grad_scales: List[float] = []
        self.gradient_norms: List[float] = []
        self.overflow_count = 0
        self.underflow_count = 0
        self.total_steps = 0
        
        # Memory stats
        self.memory_usage: List[float] = []
        self.memory_peaks: List[float] = []
    
    def update(
        self,
        scaler: Optional[GradScaler] = None,
        gradient_norms: Optional[List[float]] = None,
        memory_usage: Optional[float] = None
    ):
        """Update profiler with current step metrics."""
        self.total_steps += 1
        
        # Track gradient scale
        if scaler is not None:
            scale = scaler.get_scale()
            self.grad_scales.append(scale)
            
            # Detect overflow/underflow
            if scale == float('inf'):
                self.overflow_count += 1
            elif scale == 1e-4:
                self.underflow_count += 1
        
        # Track gradient norms
        if gradient_norms is not None:
            self.gradient_norms.extend(gradient_norms)
        
        # Track memory usage
        if memory_usage is not None:
            self.memory_usage.append(memory_usage)
            if len(self.memory_usage) > 1:
                if memory_usage > self.memory_peaks[-1] if self.memory_peaks else 0:
                    self.memory_peaks.append(memory_usage)
                else:
                    self.memory_peaks.append(self.memory_peaks[-1])
    
    def get_report(self) -> Dict[str, Any]:
        """Get comprehensive precision profiling report."""
        if not self.grad_scales:
            return {'error': 'No data collected yet'}
        
        return {
            'total_steps': self.total_steps,
            'overflow_rate': self.overflow_count / self.total_steps,
            'underflow_rate': self.underflow_count / self.total_steps,
            'avg_grad_scale': sum(self.grad_scales) / len(self.grad_scales),
            'grad_scale_variance': self._calculate_variance(self.grad_scales),
            'avg_gradient_norm': sum(self.gradient_norms) / len(self.gradient_norms) if self.gradient_norms else 0,
            'gradient_norm_variance': self._calculate_variance(self.gradient_norms) if self.gradient_norms else 0,
            'avg_memory_usage': sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0,
            'memory_peak': max(self.memory_peaks) if self.memory_peaks else 0,
            'memory_efficiency': self._calculate_memory_efficiency()
        }
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    def _calculate_memory_efficiency(self) -> float:
        """Calculate memory efficiency ratio."""
        if not self.memory_usage:
            return 0.0
        
        # Calculate how consistently memory usage stays below peak
        peaks = self.memory_peaks
        if not peaks:
            return 1.0
        
        peak = peaks[-1]
        if peak == 0:
            return 1.0
        
        avg_usage = sum(self.memory_usage) / len(self.memory_usage)
        efficiency = avg_usage / peak
        
        return min(efficiency, 1.0)
    
    def reset(self):
        """Reset profiler."""
        self.grad_scales.clear()
        self.gradient_norms.clear()
        self.overflow_count = 0
        self.underflow_count = 0
        self.total_steps = 0
        self.memory_usage.clear()
        self.memory_peaks.clear()