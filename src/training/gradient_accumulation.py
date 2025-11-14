"""
Gradient Accumulation for Large Batch Training

This module implements efficient gradient accumulation strategies for training
large models on memory-constrained hardware, supporting dynamic accumulation
and various accumulation schemes.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
import warnings


class GradientAccumulator:
    """
    Gradient Accumulator for efficient large batch training.
    
    This class manages gradient accumulation across multiple micro-batches,
    allowing training with larger effective batch sizes than what would
    fit in GPU memory.
    """
    
    def __init__(
        self,
        accumulation_steps: int = 1,
        use_zero_grad: bool = True,
        clear_accumulated_grads: bool = True,
        **kwargs
    ):
        self.accumulation_steps = accumulation_steps
        self.use_zero_grad = use_zero_grad
        self.clear_accumulated_grads = clear_accumulated_grads
        
        self.current_step = 0
        self.accumulated_grads: Dict[str, torch.Tensor] = {}
        self.gradient_norm = 0.0
        self.last_accumulation_step = 0
        
        if accumulation_steps < 1:
            raise ValueError(f"accumulation_steps must be >= 1, got {accumulation_steps}")
        
        # Validate accumulation configuration
        self._validate_accumulation_config()
    
    def _validate_accumulation_config(self):
        """Validate gradient accumulation configuration."""
        if self.accumulation_steps == 1:
            warnings.warn(
                "Gradient accumulation with accumulation_steps=1 has no effect. "
                "Consider disabling gradient accumulation for better performance.",
                UserWarning
            )
    
    def zero_grad(self, model: nn.Module):
        """
        Zero the gradients of the model parameters.
        
        Args:
            model: PyTorch model whose gradients should be zeroed
        """
        if self.use_zero_grad:
            model.zero_grad(set_to_none=True)
        
        # Clear accumulated gradients if at step boundary
        if self.clear_accumulated_grads and self.current_step == 0:
            self.accumulated_grads.clear()
    
    def accumulate_gradient(
        self,
        model: nn.Module,
        loss: torch.Tensor,
        clip_grad_norm: Optional[float] = None,
        clip_grad_norm_type: float = 2.0
    ) -> Dict[str, Any]:
        """
        Accumulate gradients from the current micro-batch.
        
        Args:
            model: PyTorch model
            loss: Loss tensor from current micro-batch
            clip_grad_norm: Maximum gradient norm for clipping (None for no clipping)
            clip_grad_norm_type: Type of norm for gradient clipping
            
        Returns:
            Dictionary with accumulation metrics
        """
        # Scale loss for accumulation
        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()
        
        # Store gradient information
        grads_info = {}
        
        # Accumulate gradients parameter-wise
        for name, param in model.named_parameters():
            if param.grad is not None:
                if name not in self.accumulated_grads:
                    # Initialize accumulated gradient
                    self.accumulated_grads[name] = param.grad.clone()
                else:
                    # Accumulate gradient
                    self.accumulated_grads[name] += param.grad
                
                grads_info[name] = {
                    'grad_norm': param.grad.norm().item(),
                    'accumulated_grad_norm': self.accumulated_grads[name].norm().item()
                }
        
        self.current_step += 1
        
        # Check if we should perform optimization step
        should_optimize = self.current_step % self.accumulation_steps == 0
        
        if should_optimize:
            return self._perform_optimization_step(
                model, clip_grad_norm, clip_grad_norm_type, grads_info
            )
        else:
            return {
                'should_optimize': False,
                'current_step': self.current_step,
                'total_steps': self.accumulation_steps,
                'grads_info': grads_info
            }
    
    def _perform_optimization_step(
        self,
        model: nn.Module,
        clip_grad_norm: Optional[float],
        clip_grad_norm_type: float,
        grads_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform the actual optimization step with accumulated gradients."""
        
        # Apply accumulated gradients to parameters
        for name, param in model.named_parameters():
            if name in self.accumulated_grads:
                # Use accumulated gradient instead of current gradient
                param.grad = self.accumulated_grads[name].clone()
        
        # Compute gradient norm for monitoring
        total_grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm(clip_grad_norm_type).item()
                total_grad_norm += grad_norm ** 2
        
        self.gradient_norm = total_grad_norm ** (1.0 / clip_grad_norm_type)
        
        # Apply gradient clipping
        if clip_grad_norm is not None:
            gradient_clipping_result = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=clip_grad_norm,
                norm_type=clip_grad_norm_type,
                error_if_nonfinite=False
            )
            
            clipped_gradient_norm = gradient_clipping_result.item()
        else:
            clipped_gradient_norm = self.gradient_norm
        
        # Prepare result
        result = {
            'should_optimize': True,
            'gradient_norm': self.gradient_norm,
            'clipped_gradient_norm': clipped_gradient_norm,
            'grads_info': grads_info,
            'accumulation_steps_completed': self.accumulation_steps
        }
        
        # Clear for next accumulation cycle
        self.accumulated_grads.clear()
        self.current_step = 0
        
        return result
    
    def get_accumulation_stats(self) -> Dict[str, Any]:
        """Get current accumulation statistics."""
        return {
            'current_step': self.current_step,
            'accumulation_steps': self.accumulation_steps,
            'remaining_steps': self.accumulation_steps - self.current_step,
            'gradient_norm': self.gradient_norm,
            'accumulated_parameters': len(self.accumulated_grads)
        }
    
    def reset(self):
        """Reset the gradient accumulator to initial state."""
        self.current_step = 0
        self.accumulated_grads.clear()
        self.gradient_norm = 0.0
        self.last_accumulation_step = 0
    
    def set_accumulation_steps(self, new_steps: int):
        """
        Update accumulation steps dynamically.
        
        Args:
            new_steps: New number of accumulation steps
        """
        if new_steps < 1:
            raise ValueError(f"accumulation_steps must be >= 1, got {new_steps}")
        
        # Flush current accumulation if changing steps mid-training
        if self.current_step > 0:
            warnings.warn(
                "Changing accumulation steps during training will reset current accumulation. "
                "Consider completing the current accumulation cycle first.",
                UserWarning
            )
            self.reset()
        
        self.accumulation_steps = new_steps
        self._validate_accumulation_config()


class DynamicGradientAccumulator(GradientAccumulator):
    """
    Dynamic gradient accumulator that adjusts accumulation steps based on
    memory usage or performance metrics.
    """
    
    def __init__(
        self,
        initial_accumulation_steps: int = 1,
        max_accumulation_steps: int = 32,
        min_accumulation_steps: int = 1,
        memory_threshold: float = 0.8,
        performance_threshold: float = 1.2,
        adaptation_frequency: int = 100,
        **kwargs
    ):
        super().__init__(initial_accumulation_steps, **kwargs)
        
        self.max_accumulation_steps = max_accumulation_steps
        self.min_accumulation_steps = min_accumulation_steps
        self.memory_threshold = memory_threshold
        self.performance_threshold = performance_threshold
        self.adaptation_frequency = adaptation_frequency
        self.adaptation_counter = 0
        
        self.performance_history: List[float] = []
        self.memory_history: List[float] = []
        
        # Validation
        if initial_accumulation_steps > max_accumulation_steps:
            raise ValueError(
                f"initial_accumulation_steps ({initial_accumulation_steps}) must be <= "
                f"max_accumulation_steps ({max_accumulation_steps})"
            )
    
    def monitor_performance(self, step_time: float):
        """
        Monitor training step performance for dynamic adaptation.
        
        Args:
            step_time: Time taken for the current training step
        """
        self.performance_history.append(step_time)
        
        # Keep only recent history for adaptation
        if len(self.performance_history) > self.adaptation_frequency:
            self.performance_history.pop(0)
        
        self.adaptation_counter += 1
        
        # Check if we should adapt accumulation steps
        if (self.adaptation_counter >= self.adaptation_frequency and
            len(self.performance_history) >= 10):
            
            self._adapt_accumulation_steps()
            self.adaptation_counter = 0
    
    def monitor_memory(self, memory_usage: float):
        """
        Monitor GPU memory usage for dynamic adaptation.
        
        Args:
            memory_usage: Current GPU memory usage ratio (0.0 to 1.0)
        """
        self.memory_history.append(memory_usage)
        
        # Keep only recent history
        if len(self.memory_history) > self.adaptation_frequency:
            self.memory_history.pop(0)
    
    def _adapt_accumulation_steps(self):
        """Adapt accumulation steps based on monitoring data."""
        if not self.performance_history or not self.memory_history:
            return
        
        # Check memory pressure
        avg_memory = sum(self.memory_history) / len(self.memory_history)
        memory_high = avg_memory > self.memory_threshold
        memory_low = avg_memory < (self.memory_threshold * 0.7)
        
        # Check performance degradation
        if len(self.performance_history) >= 20:
            recent_perf = sum(self.performance_history[-10:]) / 10
            earlier_perf = sum(self.performance_history[:-10]) / 10
            performance_degraded = recent_perf > (earlier_perf * self.performance_threshold)
        else:
            performance_degraded = False
        
        # Adapt accumulation steps
        current_steps = self.accumulation_steps
        
        if memory_high and current_steps < self.max_accumulation_steps:
            # Reduce accumulation to free memory
            new_steps = min(current_steps * 2, self.max_accumulation_steps)
            self.set_accumulation_steps(new_steps)
            print(f"Memory pressure detected. Increasing accumulation steps to {new_steps}")
            
        elif memory_low and not performance_degraded and current_steps > self.min_accumulation_steps:
            # Increase accumulation for better performance
            new_steps = max(current_steps // 2, self.min_accumulation_steps)
            self.set_accumulation_steps(new_steps)
            print(f"Low memory usage detected. Decreasing accumulation steps to {new_steps}")
    
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get adaptation statistics."""
        return {
            'current_steps': self.accumulation_steps,
            'max_steps': self.max_accumulation_steps,
            'min_steps': self.min_accumulation_steps,
            'avg_memory_usage': sum(self.memory_history) / len(self.memory_history) if self.memory_history else 0,
            'avg_step_time': sum(self.performance_history) / len(self.performance_history) if self.performance_history else 0,
            'adaptation_counter': self.adaptation_counter
        }


class GradientScaling:
    """
    Gradient scaling for mixed precision training with accumulation.
    
    Handles loss scaling to prevent gradient underflow in mixed precision training.
    """
    
    def __init__(
        self,
        init_scale: float = 2.0**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
        **kwargs
    ):
        self.init_scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.enabled = enabled
        
        self.scale = init_scale
        self._growth_checkpointer = 0
        self._unscale_grad_count = 0
        self.found_inf = False
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Scale loss for gradient accumulation with mixed precision.
        
        Args:
            loss: Unscaled loss tensor
            
        Returns:
            Scaled loss tensor
        """
        if self.enabled:
            return loss * self.scale
        return loss
    
    def unscale_gradients(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        clip_grad_norm: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Unscale gradients and check for inf/nan values.
        
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            clip_grad_norm: Optional gradient clipping norm
            
        Returns:
            Dictionary with gradient statistics
        """
        self.found_inf = False
        grads = []
        
        # Check for inf/nan gradients and collect gradient norms
        for param in model.parameters():
            if param.grad is not None:
                grad = param.grad
                # Check for inf/nan
                if not torch.isfinite(grad).all():
                    self.found_inf = True
                
                # Collect for gradient scaling
                grads.append(grad)
        
        if self.enabled:
            if self.found_inf:
                # Found inf/nan, need to skip this step
                print("Found inf/nan gradients. Skipping optimization step.")
                optimizer.step = lambda *args, **kwargs: None
                self.scale = max(self.scale * self.backoff_factor, 1e-4)
                self._growth_checkpointer = 0
                
                return {'gradient_scaling': self.scale, 'found_inf': True}
            
            # Unscale gradients
            inv_scale = 1.0 / self.scale
            for grad in grads:
                grad.mul_(inv_scale)
            
            # Check for gradient overflow/underflow
            if self._unscale_grad_count >= self.growth_interval:
                # Reset growth checkpointer
                self._growth_checkpointer = 0
                self.scale = min(self.scale * self.growth_factor, 65536.0)
            
            self._unscale_grad_count += 1
            self._growth_checkpointer += 1
        
        # Calculate gradient norms
        grad_norms = {}
        total_norm = 0.0
        
        for i, grad in enumerate(grads):
            grad_norm = grad.norm().item()
            grad_norms[f'param_{i}'] = grad_norm
            total_norm += grad_norm ** 2
        
        total_norm = total_norm ** 0.5
        
        # Apply gradient clipping if requested
        if clip_grad_norm is not None:
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                max_norm=clip_grad_norm
            ).item()
        
        grad_norms['total_norm'] = total_norm
        
        if self.enabled:
            grad_norms['gradient_scaling'] = self.scale
        
        return grad_norms
    
    def update_scale(self, found_inf: bool):
        """Update gradient scaling factor."""
        if self.enabled:
            if found_inf:
                self.scale = max(self.scale * self.backoff_factor, 1e-4)
                self._growth_checkpointer = 0
            else:
                if self._growth_checkpointer >= self.growth_interval:
                    self.scale = min(self.scale * self.growth_factor, 65536.0)
                    self._growth_checkpointer = 0
                else:
                    self._growth_checkpointer += 1


class GradientCompressor:
    """
    Gradient compression for distributed training.
    
    Compresses gradients before communication to reduce bandwidth usage.
    """
    
    def __init__(
        self,
        compression_method: str = 'topk',  # 'topk', 'fp16', 'none'
        compression_ratio: float = 0.1,
        **kwargs
    ):
        self.compression_method = compression_method
        self.compression_ratio = compression_ratio
        
        if compression_method not in ['topk', 'fp16', 'none']:
            raise ValueError(f"Unsupported compression method: {compression_method}")
    
    def compress_gradients(self, grads: List[torch.Tensor]) -> tuple:
        """
        Compress gradients for communication.
        
        Args:
            grads: List of gradient tensors
            
        Returns:
            Tuple of (compressed_grads, metadata)
        """
        if self.compression_method == 'none':
            return grads, None
        
        elif self.compression_method == 'fp16':
            # Convert to float16 for compression
            compressed = [grad.half() for grad in grads]
            return compressed, {'dtype': 'fp16'}
        
        elif self.compression_method == 'topk':
            # Keep top-k largest magnitude gradients
            compressed = []
            metadata = []
            
            for grad in grads:
                flat_grad = grad.view(-1)
                k = int(len(flat_grad) * self.compression_ratio)
                if k == 0:
                    compressed.append(grad)
                    metadata.append(None)
                    continue
                
                # Find top-k indices
                _, topk_indices = torch.topk(flat_grad.abs(), k)
                
                # Create compressed tensor
                compressed_grad = torch.zeros_like(flat_grad)
                compressed_grad[topk_indices] = flat_grad[topk_indices]
                compressed_grad = compressed_grad.view_as(grad)
                
                compressed.append(compressed_grad)
                metadata.append({
                    'indices': topk_indices.cpu().numpy(),
                    'values': flat_grad[topk_indices].cpu().numpy()
                })
            
            return compressed, {'method': 'topk', 'metadata': metadata}
        
        else:
            raise ValueError(f"Unknown compression method: {self.compression_method}")
    
    def decompress_gradients(
        self, 
        compressed_grads: List[torch.Tensor], 
        original_shapes: List[torch.Size],
        metadata: Optional[Dict]
    ) -> List[torch.Tensor]:
        """
        Decompress gradients after communication.
        
        Args:
            compressed_grads: Compressed gradient tensors
            original_shapes: Original shapes of gradient tensors
            metadata: Compression metadata
            
        Returns:
            Decompressed gradient tensors
        """
        if self.compression_method == 'none':
            return compressed_grads
        
        elif self.compression_method == 'fp16':
            # Convert back to float32
            return [grad.float() for grad in compressed_grads]
        
        elif self.compression_method == 'topk':
            decompressed = []
            
            for compressed_grad, original_shape, meta in zip(
                compressed_grads, original_shapes, metadata['metadata']
            ):
                # Reshape to original
                compressed_grad = compressed_grad.view(original_shape)
                decompressed_grad = torch.zeros_like(compressed_grad)
                
                # Restore top-k values
                if meta is not None:
                    flat_grad = compressed_grad.view(-1)
                    indices = torch.from_numpy(meta['indices']).to(compressed_grad.device)
                    values = torch.from_numpy(meta['values']).to(compressed_grad.device)
                    
                    flat_decompressed = decompressed_grad.view(-1)
                    flat_decompressed[indices] = values
                
                decompressed.append(decompressed_grad)
            
            return decompressed
        
        else:
            raise ValueError(f"Unknown compression method: {self.compression_method}")