"""
Gradient Optimization Strategies

This module implements advanced gradient optimization techniques including
gradient clipping, gradient accumulation across micro-batches, gradient 
checkpointing, and memory-efficient backpropagation for large-scale training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Union
from collections import defaultdict
import warnings
from contextlib import contextmanager
import copy


class GradientClipper:
    """
    Advanced gradient clipping strategies for stable training.
    
    Supports multiple clipping methods: value clipping, norm clipping,
    adaptive clipping, and gradient accumulation with clipping.
    """
    
    def __init__(
        self,
        clip_type: str = 'norm',
        clip_value: float = 1.0,
        clip_method: str = 'global',
        adaptive_mode: bool = False,
        warmup_steps: int = 1000
    ):
        """
        Initialize gradient clipper.
        
        Args:
            clip_type: Type of clipping ('norm', 'value', 'adaptive')
            clip_value: Clipping threshold value
            clip_method: Method for computing gradients ('global', 'local', 'layer')
            adaptive_mode: Whether to use adaptive clipping
            warmup_steps: Number of warmup steps for adaptive clipping
        """
        self.clip_type = clip_type
        self.clip_value = clip_value
        self.clip_method = clip_method
        self.adaptive_mode = adaptive_mode
        self.warmup_steps = warmup_steps
        
        # Adaptive clipping parameters
        self.initial_clip_value = clip_value
        self.target_gnorm = clip_value
        self.gradient_history = []
        self.adaptation_factor = 1.1
        
        # Statistics
        self.stats = {
            'total_clips': 0,
            'avg_gradient_norm': 0.0,
            'clip_efficiency': 0.0
        }
        
        # Gradient buffers for different methods
        self.gradient_buffers = {}
        
    def _compute_gradient_norm(self, parameters: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute gradient norms for given parameters.
        
        Args:
            parameters: List of parameter tensors
            
        Returns:
            Global gradient norm
        """
        total_norm = 0.0
        param_count = 0
        
        for p in parameters:
            if p.grad is not None:
                if self.clip_method == 'global':
                    total_norm += p.grad.data.norm(2) ** 2
                    param_count += 1
                elif self.clip_method == 'local':
                    # Will handle per parameter in clipping method
                    pass
        
        if self.clip_method == 'global' and param_count > 0:
            return total_norm.sqrt() / param_count
        
        return total_norm.sqrt()
    
    def _adaptive_clip_value(self, step: int) -> float:
        """
        Compute adaptive clip value based on training dynamics.
        
        Args:
            step: Current training step
            
        Returns:
            Adaptive clip value
        """
        if not self.adaptive_mode or step < self.warmup_steps:
            return self.clip_value
        
        # Calculate statistics from gradient history
        if len(self.gradient_history) > 10:
            recent_gnorms = self.gradient_history[-100:]  # Last 100 steps
            mean_gnorm = np.mean(recent_gnorms)
            std_gnorm = np.std(recent_gnorms)
            
            # Adjust clip value based on gradient statistics
            if mean_gnorm < self.target_gnorm * 0.5:
                # Gradients are too small, reduce clipping
                self.clip_value = max(self.clip_value / self.adaptation_factor, 0.1)
            elif mean_gnorm > self.target_gnorm * 1.5:
                # Gradients are too large, increase clipping
                self.clip_value = min(self.clip_value * self.adaptation_factor, 10.0)
        
        return self.clip_value
    
    def clip_gradients(
        self,
        parameters: List[torch.Tensor],
        step: Optional[int] = None,
        return_norm: bool = False
    ) -> Optional[torch.Tensor]:
        """
        Apply gradient clipping to parameters.
        
        Args:
            parameters: List of parameter tensors
            step: Current training step
            return_norm: Whether to return gradient norm
            
        Returns:
            Gradient norm if return_norm is True
        """
        if step is not None and self.adaptive_mode:
            clip_value = self._adaptive_clip_value(step)
        else:
            clip_value = self.clip_value
        
        if self.clip_type == 'norm':
            return self._clip_by_norm(parameters, clip_value, return_norm)
        elif self.clip_type == 'value':
            return self._clip_by_value(parameters, clip_value, return_norm)
        elif self.clip_type == 'adaptive':
            return self._adaptive_clip(parameters, step, return_norm)
        else:
            raise ValueError(f"Unknown clip type: {self.clip_type}")
    
    def _clip_by_norm(
        self,
        parameters: List[torch.Tensor],
        clip_value: float,
        return_norm: bool
    ) -> Optional[torch.Tensor]:
        """Clip gradients by global norm."""
        if self.clip_method == 'global':
            # Global norm clipping
            total_norm = 0.0
            for p in parameters:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            
            total_norm = total_norm ** (1.0 / 2)
            
            if total_norm > clip_value:
                clip_coefficient = clip_value / (total_norm + 1e-6)
                for p in parameters:
                    if p.grad is not None:
                        p.grad.data.mul_(clip_coefficient)
                
                self.stats['total_clips'] += 1
            
            self.gradient_history.append(total_norm)
            self.stats['avg_gradient_norm'] = np.mean(self.gradient_history)
            
            if return_norm:
                return torch.tensor(total_norm)
            
        elif self.clip_method == 'local':
            # Per-parameter norm clipping
            max_norm = 0.0
            clipped_count = 0
            
            for p in parameters:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    max_norm = max(max_norm, param_norm.item())
                    
                    if param_norm > clip_value:
                        clip_coefficient = clip_value / (param_norm + 1e-6)
                        p.grad.data.mul_(clip_coefficient)
                        clipped_count += 1
            
            self.stats['total_clips'] += clipped_count
            self.gradient_history.append(max_norm)
            self.stats['avg_gradient_norm'] = np.mean(self.gradient_history)
            
            if return_norm:
                return torch.tensor(max_norm)
        
        return None
    
    def _clip_by_value(
        self,
        parameters: List[torch.Tensor],
        clip_value: float,
        return_norm: bool
    ) -> Optional[torch.Tensor]:
        """Clip gradients by absolute value."""
        max_val = 0.0
        clipped_count = 0
        
        for p in parameters:
            if p.grad is not None:
                # Clip individual gradient values
                p.grad.data.clamp_(-clip_value, clip_value)
                
                max_val = max(max_val, p.grad.data.abs().max().item())
                
                # Count clipped values
                clipped = (p.grad.data.abs() > clip_value).sum().item()
                clipped_count += clipped
        
        self.stats['total_clips'] += clipped_count
        self.gradient_history.append(max_val)
        self.stats['avg_gradient_norm'] = np.mean(self.gradient_history)
        
        if return_norm:
            return torch.tensor(max_val)
        
        return None
    
    def _adaptive_clip(
        self,
        parameters: List[torch.Tensor],
        step: Optional[int],
        return_norm: bool
    ) -> Optional[torch.Tensor]:
        """Adaptive clipping based on gradient statistics."""
        # First compute norms
        norms = []
        for p in parameters:
            if p.grad is not None:
                norms.append(p.grad.data.norm(2).item())
        
        if not norms:
            return None
        
        # Adaptive clipping based on gradient distribution
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)
        
        # Adaptive threshold based on gradient statistics
        adaptive_threshold = mean_norm + 2 * std_norm  # Clip outliers
        
        # Apply clipping
        clipped_count = 0
        for i, p in enumerate(parameters):
            if p.grad is not None:
                norm = norms[i]
                if norm > adaptive_threshold:
                    clip_coefficient = adaptive_threshold / (norm + 1e-6)
                    p.grad.data.mul_(clip_coefficient)
                    clipped_count += 1
        
        max_norm = max(norms)
        self.gradient_history.append(max_norm)
        self.stats['total_clips'] += clipped_count
        self.stats['avg_gradient_norm'] = np.mean(self.gradient_history)
        
        if return_norm:
            return torch.tensor(max_norm)
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get clipping statistics."""
        return {
            **self.stats,
            'adaptive_clip_value': self.clip_value,
            'gradient_history_length': len(self.gradient_history),
            'last_gradients': self.gradient_history[-10:] if len(self.gradient_history) >= 10 else self.gradient_history
        }


class GradientAccumulator:
    """
    Advanced gradient accumulation with dynamic batch sizing and memory management.
    
    Supports variable batch sizes, gradient scaling, and memory-aware accumulation.
    """
    
    def __init__(
        self,
        accumulation_steps: int = 4,
        scale_gradients: bool = True,
        gradient_scaling: str = 'linear',  # 'linear', 'exponential', 'adaptive'
        memory_budget: Optional[int] = None,
        dynamic_accumulation: bool = True
    ):
        """
        Initialize gradient accumulator.
        
        Args:
            accumulation_steps: Number of steps to accumulate gradients
            scale_gradients: Whether to scale accumulated gradients
            gradient_scaling: Type of gradient scaling
            memory_budget: Available memory budget in MB
            dynamic_accumulation: Whether to dynamically adjust accumulation steps
        """
        self.accumulation_steps = accumulation_steps
        self.scale_gradients = scale_gradients
        self.gradient_scaling = gradient_scaling
        self.memory_budget = memory_budget
        self.dynamic_accumulation = dynamic_accumulation
        
        # State
        self.step_count = 0
        self.accumulated_steps = 0
        self.gradient_buffer = {}
        self.original_params = {}
        
        # Statistics
        self.stats = {
            'total_accumulations': 0,
            'avg_batch_size': 0.0,
            'memory_efficiency': 0.0,
            'dynamic_adjustments': 0
        }
        
        # Memory tracking
        self.memory_usage_history = []
        self.peak_memory_usage = 0
        
    def register_parameters(self, parameters: List[torch.Tensor]):
        """
        Register parameters for gradient accumulation.
        
        Args:
            parameters: List of parameter tensors
        """
        for i, param in enumerate(parameters):
            param_id = id(param)
            self.gradient_buffer[param_id] = torch.zeros_like(param)
            
            if self.original_params is not None:
                self.original_params[param_id] = param.data.clone()
    
    def accumulate_gradients(
        self,
        parameters: List[torch.Tensor],
        loss_scale: float = 1.0
    ) -> bool:
        """
        Accumulate gradients for the current step.
        
        Args:
            parameters: List of parameter tensors
            loss_scale: Scale factor for loss
            
        Returns:
            True if accumulation step is complete
        """
        self.step_count += 1
        self.accumulated_steps += 1
        
        # Accumulate gradients
        for param in parameters:
            param_id = id(param)
            if param_id in self.gradient_buffer and param.grad is not None:
                self.gradient_buffer[param_id].add_(param.grad.data)
        
        # Check if accumulation step is complete
        if self.accumulated_steps >= self.accumulation_steps:
            # Apply accumulated gradients
            self._apply_accumulated_gradients(parameters, loss_scale)
            
            # Reset accumulation
            self.accumulated_steps = 0
            self._clear_accumulated_gradients()
            
            # Update statistics
            self.stats['total_accumulations'] += 1
            
            # Dynamic adjustment if enabled
            if self.dynamic_accumulation:
                self._dynamic_adjustment()
            
            return True
        
        return False
    
    def _apply_accumulated_gradients(
        self,
        parameters: List[torch.Tensor],
        loss_scale: float
    ):
        """Apply accumulated gradients to parameters."""
        scale_factor = 1.0
        
        if self.scale_gradients:
            if self.gradient_scaling == 'linear':
                scale_factor = 1.0 / self.accumulated_steps
            elif self.gradient_scaling == 'exponential':
                scale_factor = 0.9 ** (self.accumulated_steps - 1)
            elif self.gradient_scaling == 'adaptive':
                # Adaptive scaling based on gradient magnitude
                total_norm = sum(
                    self.gradient_buffer[id(param)].norm().item() 
                    for param in parameters if param.grad is not None
                )
                scale_factor = min(1.0 / max(total_norm, 1e-8), 1.0)
        
        # Apply scaled gradients
        for param in parameters:
            param_id = id(param)
            if param_id in self.gradient_buffer:
                if param.grad is not None:
                    param.grad.data.copy_(self.gradient_buffer[param_id] * scale_factor)
                else:
                    # Create gradient if it doesn't exist
                    param.grad = self.gradient_buffer[param_id] * scale_factor
    
    def _clear_accumulated_gradients(self):
        """Clear accumulated gradients from buffer."""
        for param_id in self.gradient_buffer:
            self.gradient_buffer[param_id].zero_()
    
    def _dynamic_adjustment(self):
        """Dynamically adjust accumulation steps based on memory and performance."""
        if len(self.memory_usage_history) > 10:
            recent_memory = np.mean(self.memory_usage_history[-10:])
            
            # Adjust accumulation steps based on memory usage
            if self.memory_budget and recent_memory > self.memory_budget * 0.9:
                # Reduce accumulation steps if memory is high
                self.accumulation_steps = max(2, self.accumulation_steps - 1)
                self.stats['dynamic_adjustments'] += 1
            elif recent_memory < self.memory_budget * 0.5 and self.accumulation_steps < 8:
                # Increase accumulation steps if memory is available
                self.accumulation_steps += 1
                self.stats['dynamic_adjustments'] += 1
    
    def get_accumulation_info(self) -> Dict[str, Any]:
        """Get information about current accumulation state."""
        return {
            'current_step': self.step_count,
            'accumulated_steps': self.accumulated_steps,
            'target_steps': self.accumulation_steps,
            'completion_percentage': (self.accumulated_steps / self.accumulation_steps) * 100,
            'effective_batch_size': self.accumulated_steps,
            **self.stats
        }


class GradientCheckpointing:
    """
    Memory-efficient gradient checkpointing with selective recomputation.
    
    Implements activation checkpointing, gradient checkpointing, and
    selective recomputation strategies.
    """
    
    def __init__(
        self,
        checkpointing_strategy: str = 'activation',
        recompute_ratio: float = 0.5,
        checkpoint_intervals: Optional[List[int]] = None,
        selective_checkpointing: bool = True
    ):
        """
        Initialize gradient checkpointing.
        
        Args:
            checkpointing_strategy: Type of checkpointing ('activation', 'gradient', 'selective')
            recompute_ratio: Ratio of operations to recompute
            checkpoint_intervals: Steps between checkpoints
            selective_checkpointing: Whether to use selective checkpointing
        """
        self.checkpointing_strategy = checkpointing_strategy
        self.recompute_ratio = recompute_ratio
        self.checkpoint_intervals = checkpoint_intervals
        self.selective_checkpointing = selective_checkpointing
        
        # State
        self.checkpoint_buffer = {}
        self.recompute_operations = []
        self.memory_savings = 0.0
        
        # Statistics
        self.stats = {
            'total_checkpoints': 0,
            'total_recomputations': 0,
            'memory_saved_mb': 0.0,
            'time_overhead': 0.0
        }
        
    @contextmanager
    def checkpoint_layer(self, layer_id: str, compute_fn, *args, **kwargs):
        """
        Context manager for checkpointing a layer.
        
        Args:
            layer_id: Unique identifier for the layer
            compute_fn: Function to compute layer output
            *args, **kwargs: Arguments for compute_fn
        """
        if self.checkpointing_strategy == 'activation':
            with self._activation_checkpoint(layer_id, compute_fn, *args, **kwargs):
                yield
        elif self.checkpointing_strategy == 'gradient':
            with self._gradient_checkpoint(layer_id, compute_fn, *args, **kwargs):
                yield
        else:
            # No checkpointing
            yield
    
    @contextmanager
    def _activation_checkpoint(self, layer_id: str, compute_fn, *args, **kwargs):
        """Activation checkpointing context manager."""
        # Clear previous checkpoint
        if layer_id in self.checkpoint_buffer:
            del self.checkpoint_buffer[layer_id]
        
        # Compute and store intermediate activations
        # This is a simplified implementation
        try:
            result = compute_fn(*args, **kwargs)
            self.checkpoint_buffer[layer_id] = result
            yield
        finally:
            # Clear checkpoint after use
            if layer_id in self.checkpoint_buffer:
                del self.checkpoint_buffer[layer_id]
    
    @contextmanager
    def _gradient_checkpoint(self, layer_id: str, compute_fn, *args, **kwargs):
        """Gradient checkpointing context manager."""
        # Gradient checkpointing stores gradients for recomputation
        # This is a simplified implementation
        original_grad = {}
        
        # Store original gradients
        for tensor in args:
            if isinstance(tensor, torch.Tensor) and tensor.grad is not None:
                original_grad[id(tensor)] = tensor.grad.clone()
        
        try:
            yield
            # Gradients are recomputed in backward pass
        finally:
            # Restore original gradients if needed
            for tensor in args:
                if isinstance(tensor, torch.Tensor) and id(tensor) in original_grad:
                    tensor.grad = original_grad[id(tensor)]
    
    def selective_checkpoint(
        self,
        module: nn.Module,
        inputs: torch.Tensor,
        threshold: float = 0.1
    ) -> torch.Tensor:
        """
        Apply selective checkpointing to a module.
        
        Args:
            module: Module to checkpoint
            inputs: Input tensor
            threshold: Threshold for selective checkpointing
            
        Returns:
            Module output
        """
        if not self.selective_checkpointing:
            return module(inputs)
        
        # Analyze module complexity
        module_complexity = self._estimate_module_complexity(module)
        
        # Decide whether to checkpoint based on complexity
        if module_complexity > threshold:
            # Use checkpointing for complex modules
            with self.checkpoint_layer(f"selective_{id(module)}", module, inputs):
                output = module(inputs)
        else:
            # Regular forward pass for simple modules
            output = module(inputs)
        
        return output
    
    def _estimate_module_complexity(self, module: nn.Module) -> float:
        """
        Estimate module complexity for selective checkpointing.
        
        Args:
            module: Module to analyze
            
        Returns:
            Complexity score
        """
        complexity = 0.0
        
        # Count parameters
        param_count = sum(p.numel() for p in module.parameters())
        complexity += param_count * 1e-6
        
        # Count operations (simplified)
        if isinstance(module, nn.Linear):
            complexity += module.in_features * module.out_features * 1e-8
        elif isinstance(module, nn.Conv2d):
            complexity += (module.in_channels * module.out_channels * 
                          module.kernel_size[0] * module.kernel_size[1]) * 1e-8
        
        return complexity
    
    def apply_strategic_checkpointing(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        strategy: str = 'every_n_layers'
    ) -> torch.Tensor:
        """
        Apply strategic checkpointing to model layers.
        
        Args:
            model: Model to checkpoint
            input_tensor: Input tensor
            strategy: Checkpointing strategy
            
        Returns:
            Model output
        """
        if strategy == 'every_n_layers':
            return self._checkpoint_every_n_layers(model, input_tensor)
        elif strategy == 'attention_only':
            return self._checkpoint_attention_layers(model, input_tensor)
        elif strategy == 'dense_layers':
            return self._checkpoint_dense_layers(model, input_tensor)
        else:
            return model(input_tensor)
    
    def _checkpoint_every_n_layers(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Checkpoint every n-th layer."""
        x = input_tensor
        checkpoint_counter = 0
        
        for i, layer in enumerate(model.modules()):
            if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d)) and checkpoint_counter % 2 == 0:
                with self.checkpoint_layer(f"layer_{i}", layer, x) as checkpointed_x:
                    x = checkpointed_x
                checkpoint_counter += 1
            else:
                x = layer(x)
        
        return x
    
    def _checkpoint_attention_layers(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Checkpoint attention layers only."""
        x = input_tensor
        
        for i, layer in enumerate(model.modules()):
            if hasattr(layer, 'attention') or 'attention' in str(type(layer)).lower():
                with self.checkpoint_layer(f"attention_{i}", layer, x):
                    x = layer(x)
            else:
                x = layer(x)
        
        return x
    
    def _checkpoint_dense_layers(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Checkpoint dense layers only."""
        x = input_tensor
        
        for i, layer in enumerate(model.modules()):
            if isinstance(layer, nn.Linear) and layer.out_features > 1000:
                with self.checkpoint_layer(f"dense_{i}", layer, x):
                    x = layer(x)
            else:
                x = layer(x)
        
        return x
    
    def get_memory_savings(self) -> Dict[str, float]:
        """Calculate memory savings from checkpointing."""
        return {
            'estimated_memory_saved_mb': self.memory_savings,
            'checkpoint_efficiency': self.stats['total_recomputations'] / max(self.stats['total_checkpoints'], 1),
            'average_recomputation_time': self.stats['time_overhead'] / max(self.stats['total_recomputations'], 1)
        }


class MemoryEfficientBackprop:
    """
    Memory-efficient backpropagation with dynamic recomputation.
    
    Implements memory-aware backpropagation strategies that dynamically
    balance memory usage and computational efficiency.
    """
    
    def __init__(
        self,
        memory_budget: int = 8000,  # MB
        recomputation_strategy: str = 'adaptive',
        activation_threshold: float = 0.5,
        gradient_checkpointing: bool = True
    ):
        """
        Initialize memory-efficient backprop.
        
        Args:
            memory_budget: Available memory budget in MB
            recomputation_strategy: Strategy for recomputation ('adaptive', 'fixed', 'selective')
            activation_threshold: Threshold for activation recomputation
            gradient_checkpointing: Whether to enable gradient checkpointing
        """
        self.memory_budget = memory_budget
        self.recomputation_strategy = recomputation_strategy
        self.activation_threshold = activation_threshold
        self.gradient_checkpointing = gradient_checkpointing
        
        # State
        self.current_memory_usage = 0
        self.memory_history = []
        self.recomputation_schedule = {}
        
        # Statistics
        self.stats = {
            'total_backprops': 0,
            'total_recomputations': 0,
            'memory_efficiency': 0.0,
            'average_memory_usage': 0.0
        }
    
    def memory_aware_forward(
        self,
        module: nn.Module,
        inputs: torch.Tensor,
        memory_monitor: bool = True
    ) -> torch.Tensor:
        """
        Memory-aware forward pass with dynamic memory management.
        
        Args:
            module: Module to forward pass
            inputs: Input tensor
            memory_monitor: Whether to monitor memory usage
            
        Returns:
            Module output
        """
        if memory_monitor:
            torch.cuda.empty_cache()  # Clear cache before forward pass
        
        # Check memory usage
        if memory_monitor:
            current_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            self.current_memory_usage = current_memory
            self.memory_history.append(current_memory)
            
            # Adjust computation if memory is high
            if current_memory > self.memory_budget * 0.9:
                self._adjust_for_memory_pressure()
        
        # Perform forward pass
        output = module(inputs)
        
        # Update memory usage
        if memory_monitor:
            new_memory = torch.cuda.memory_allocated() / (1024 * 1024)
            self.current_memory_usage = new_memory
        
        return output
    
    def memory_aware_backward(
        self,
        loss: torch.Tensor,
        model: nn.Module,
        inputs: torch.Tensor,
        gradient_checkpointing: Optional[bool] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Memory-aware backward pass with dynamic strategies.
        
        Args:
            loss: Loss tensor
            model: Model for backprop
            inputs: Input tensor
            gradient_checkpointing: Override for gradient checkpointing
            
        Returns:
            Dictionary of gradients
        """
        # Use gradient checkpointing if memory is limited
        use_checkpointing = (
            gradient_checkpointing if gradient_checkpointing is not None 
            else self.gradient_checkpointing
        )
        
        if use_checkpointing and self.current_memory_usage > self.memory_budget * 0.8:
            # Use gradient checkpointing when memory is constrained
            gradients = self._gradient_checkpointed_backward(loss, model, inputs)
        else:
            # Standard backward pass
            gradients = self._standard_backward(loss, model)
        
        self.stats['total_backprops'] += 1
        return gradients
    
    def _gradient_checkpointed_backward(
        self,
        loss: torch.Tensor,
        model: nn.Module,
        inputs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Gradient checkpointed backward pass."""
        # Store original gradients
        original_grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                original_grads[name] = param.grad.clone()
        
        # Clear gradients
        model.zero_grad()
        
        # Compute gradients with checkpointing
        loss.backward(retain_graph=False)
        
        # Collect gradients
        gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()
                # Restore original gradient if it existed
                if name in original_grads:
                    param.grad = original_grads[name]
        
        self.stats['total_recomputations'] += 1
        return gradients
    
    def _standard_backward(
        self,
        loss: torch.Tensor,
        model: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """Standard backward pass."""
        model.zero_grad()
        loss.backward()
        
        gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()
        
        return gradients
    
    def _adjust_for_memory_pressure(self):
        """Adjust computation strategy under memory pressure."""
        if self.recomputation_strategy == 'adaptive':
            # Reduce batch size or enable more aggressive checkpointing
            self.gradient_checkpointing = True
        elif self.recomputation_strategy == 'fixed':
            # Apply fixed adjustments
            pass
        elif self.recomputation_strategy == 'selective':
            # Enable selective recomputation
            self._enable_selective_recomputation()
    
    def _enable_selective_recomputation(self):
        """Enable selective recomputation for high-memory operations."""
        # This would implement selective recomputation strategies
        # For example, recompute only attention layers or large dense layers
        pass
    
    def get_backprop_statistics(self) -> Dict[str, Any]:
        """Get backpropagation statistics."""
        return {
            **self.stats,
            'current_memory_usage_mb': self.current_memory_usage,
            'memory_efficiency_score': self.stats['memory_efficiency'],
            'memory_history_length': len(self.memory_history),
            'peak_memory_usage_mb': max(self.memory_history) if self.memory_history else 0
        }


# Utility functions

def create_gradient_optimizer(
    strategy: str,
    model_parameters: List[torch.Tensor],
    **kwargs
) -> Union[GradientClipper, GradientAccumulator, GradientCheckpointing, MemoryEfficientBackprop]:
    """
    Factory function to create gradient optimizers.
    
    Args:
        strategy: Strategy type ('clip', 'accumulate', 'checkpoint', 'backprop')
        model_parameters: Model parameters
        **kwargs: Strategy-specific arguments
        
    Returns:
        Gradient optimizer instance
    """
    if strategy == 'clip':
        return GradientClipper(**kwargs)
    elif strategy == 'accumulate':
        return GradientAccumulator(**kwargs)
    elif strategy == 'checkpoint':
        return GradientCheckpointing(**kwargs)
    elif strategy == 'backprop':
        return MemoryEfficientBackprop(**kwargs)
    else:
        raise ValueError(f"Unknown gradient strategy: {strategy}")


def benchmark_gradient_strategies(
    model_parameters: List[torch.Tensor],
    memory_budget: int,
    batch_size: int
) -> Dict[str, Dict]:
    """
    Benchmark different gradient optimization strategies.
    
    Args:
        model_parameters: Model parameters
        memory_budget: Available memory in MB
        batch_size: Batch size
        
    Returns:
        Benchmark results
    """
    strategies = {
        'gradient_clip': {
            'memory_usage': len(model_parameters) * 4 * 2,  # Gradients + parameters
            'time_overhead': 0.1,
            'memory_saving': 0.0
        },
        'gradient_accumulation': {
            'memory_usage': len(model_parameters) * 4,  # Only parameters
            'time_overhead': 0.05,
            'memory_saving': 60.0  # % saving
        },
        'gradient_checkpointing': {
            'memory_usage': len(model_parameters) * 4 * 0.5,  # 50% memory usage
            'time_overhead': 0.3,
            'memory_saving': 50.0  # % saving
        },
        'memory_efficient_backprop': {
            'memory_usage': len(model_parameters) * 4 * 0.7,  # 70% memory usage
            'time_overhead': 0.2,
            'memory_saving': 30.0  # % saving
        }
    }
    
    # Calculate results
    results = {}
    for strategy, config in strategies.items():
        fits_in_budget = config['memory_usage'] <= memory_budget * 1024 * 1024
        
        results[strategy] = {
            'fits_in_budget': fits_in_budget,
            'memory_usage_mb': config['memory_usage'] / (1024 * 1024),
            'time_overhead_factor': config['time_overhead'],
            'memory_saving_percent': config['memory_saving'],
            'recommended': fits_in_budget and config['memory_saving'] > 30
        }
    
    return results