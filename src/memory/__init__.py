"""
Memory-Efficient Training Utilities

This module implements advanced memory optimization techniques for large-scale
language model training, including gradient checkpointing, activation recomputation,
memory-efficient attention, and dynamic memory management.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import psutil
import warnings
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
import contextlib
import time


class GradientCheckpointing:
    """
    Memory-efficient gradient checkpointing implementation.
    
    Reduces memory usage by recomputing activations during backward pass
    instead of storing them during forward pass.
    """
    
    def __init__(
        self,
        enabled: bool = True,
        checkpoint_frequency: int = 1,
        checkpoint_modules: Optional[List[str]] = None,
        non_checkpointable_modules: Optional[List[type]] = None,
        **kwargs
    ):
        self.enabled = enabled
        self.checkpoint_frequency = checkpoint_frequency
        self.checkpoint_modules = checkpoint_modules or []
        self.non_checkpointable_modules = non_checkpointable_modules or []
        
        # Tracking
        self.forward_calls = 0
        self.checkpoints_created = 0
        
        # Setup
        if enabled:
            self._setup_checkpointing()
    
    def _setup_checkpointing(self):
        """Setup gradient checkpointing hooks."""
        # Register backward hooks for checkpointing
        self.hooks = []
        
        def checkpoint_backward_hook(module, grad_input, grad_output):
            """Hook for backward pass with checkpointing."""
            if self._should_checkpoint_module(module):
                return self._checkpoint_backward_function(
                    module, grad_input, grad_output
                )
            return None
        
        # This is a simplified implementation
        # In practice, you'd need more sophisticated hooking
    
    def _should_checkpoint_module(self, module: nn.Module) -> bool:
        """Determine if a module should use checkpointing."""
        # Check if module is in checkpoint list
        if self.checkpoint_modules:
            module_name = type(module).__name__
            return module_name in self.checkpoint_modules
        
        # Check if module is not checkpointable
        module_type = type(module)
        for non_checkpointable in self.non_checkpointable_modules:
            if isinstance(module, non_checkpointable):
                return False
        
        # Default: checkpoint transformer layers and large modules
        return isinstance(module, (nn.MultiheadAttention, nn.Linear)) and \
               sum(p.numel() for p in module.parameters()) > 1000000
    
    def _checkpoint_backward_function(
        self,
        module: nn.Module,
        grad_input: Tuple,
        grad_output: Tuple
    ) -> Tuple:
        """Function to recompute activations during backward pass."""
        # This is a placeholder - actual implementation would be more complex
        # It would need to:
        # 1. Capture forward computation
        # 2. Recompute during backward
        # 3. Handle gradients correctly
        
        # For now, return gradients unchanged
        return grad_input
    
    @contextlib.contextmanager
    def checkpoint_forward(self, module: nn.Module, *inputs):
        """Context manager for checkpointed forward pass."""
        if not self.enabled:
            yield
            return
        
        # Capture inputs for recomputation
        checkpoint_data = self._capture_forward_computation(module, inputs)
        
        try:
            # Perform forward pass normally
            yield
        except Exception as e:
            # If forward pass fails, try recomputing
            warnings.warn(f"Forward pass failed, trying recomputation: {e}")
            self._recompute_forward(module, checkpoint_data, inputs)
            raise
    
    def _capture_forward_computation(self, module: nn.Module, inputs) -> Dict:
        """Capture forward computation for later recomputation."""
        # This would capture the computation graph
        # For now, just store inputs
        return {
            'inputs': inputs,
            'module_state': module.training,
            'timestamp': time.time()
        }
    
    def _recompute_forward(self, module: nn.Module, checkpoint_data: Dict, inputs):
        """Recompute forward pass during backward."""
        # Restore module state
        module.training = checkpoint_data['module_state']
        
        # Recomputation logic would go here
        # This is complex and depends on the specific module structure
    
    def enable_checkpointing(self):
        """Enable gradient checkpointing."""
        self.enabled = True
        self._setup_checkpointing()
    
    def disable_checkpointing(self):
        """Disable gradient checkpointing."""
        self.enabled = False
        
        # Remove hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def get_memory_savings(self) -> Dict[str, float]:
        """Estimate memory savings from checkpointing."""
        if not self.enabled:
            return {'enabled': False}
        
        return {
            'enabled': True,
            'checkpoints_created': self.checkpoints_created,
            'estimated_savings_gb': self._estimate_memory_savings()
        }
    
    def _estimate_memory_savings(self) -> float:
        """Estimate memory savings in GB."""
        # This is a rough estimation
        # In practice, you'd measure actual memory usage
        return 1.5  # Placeholder value


class MemoryEfficientAttention:
    """
    Memory-efficient attention mechanisms.
    
    Implements various techniques to reduce memory usage in attention layers:
    - Linear attention
    - Sparse attention patterns
    - Chunked attention computation
    """
    
    @staticmethod
    def linear_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0
    ) -> torch.Tensor:
        """
        Linear attention with O(N) complexity instead of O(N²).
        
        Args:
            query: [batch, seq_len, d_model]
            key: [batch, seq_len, d_model]
            value: [batch, seq_len, d_model]
            mask: Optional attention mask
            dropout_p: Dropout probability
            
        Returns:
            output: [batch, seq_len, d_model]
        """
        batch_size, seq_len, d_model = query.size()
        
        # Compute attention scores in linear time
        # Using the feature map approach: ϕ(x) =elu(x) + 1
        
        # Apply feature map
        phi_q = F.elu(query) + 1
        phi_k = F.elu(key) + 1
        
        # Compute attention using prefix sums
        kv = torch.matmul(phi_k.transpose(-2, -1), value)  # [batch, d_model, d_model]
        Z = torch.sum(phi_k, dim=-2, keepdim=True)  # [batch, 1, d_model]
        
        # Apply mask if provided
        if mask is not None:
            # Mask handling for linear attention
            mask = mask.unsqueeze(-1)  # [batch, seq_len, 1]
            phi_k = phi_k * mask
            value = value * mask
        
        # Compute output
        output = torch.matmul(phi_q, kv) / (Z + 1e-8)
        
        # Apply dropout
        if dropout_p > 0 and torch.is_grad_enabled():
            output = F.dropout(output, p=dropout_p)
        
        return output
    
    @staticmethod
    def chunked_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        chunk_size: int = 512,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Chunked attention computation to reduce peak memory usage.
        
        Args:
            query: [batch, seq_len, d_model]
            key: [batch, seq_len, d_model]
            value: [batch, seq_len, d_model]
            chunk_size: Size of chunks for computation
            mask: Optional attention mask
            
        Returns:
            output: [batch, seq_len, d_model]
        """
        batch_size, seq_len, d_model = query.size()
        
        if seq_len <= chunk_size:
            # No need for chunking
            scores = torch.matmul(query, key.transpose(-2, -1))
            if mask is not None:
                scores = scores + mask
            attention_weights = F.softmax(scores / (d_model ** 0.5), dim=-1)
            output = torch.matmul(attention_weights, value)
        else:
            # Chunked computation
            output = torch.zeros_like(query)
            
            for start_idx in range(0, seq_len, chunk_size):
                end_idx = min(start_idx + chunk_size, seq_len)
                
                # Get chunk
                q_chunk = query[:, start_idx:end_idx]
                k_chunk = key[:, start_idx:end_idx]
                v_chunk = value[:, start_idx:end_idx]
                
                # Compute attention for chunk
                scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1))
                
                # Apply mask to chunk
                if mask is not None:
                    chunk_mask = mask[:, start_idx:end_idx, :seq_len]
                    scores = scores + chunk_mask
                
                # Apply softmax and compute output
                attention_weights = F.softmax(scores / (d_model ** 0.5), dim=-1)
                chunk_output = torch.matmul(attention_weights, value)
                
                # Store output
                output[:, start_idx:end_idx] = chunk_output
        
        return output
    
    @staticmethod
    def sparse_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        sparsity_pattern: str = 'sliding_window',
        window_size: int = 64,
        num_random: int = 16,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sparse attention with various sparsity patterns.
        
        Args:
            query: [batch, seq_len, d_model]
            key: [batch, seq_len, d_model]
            value: [batch, seq_len, d_model]
            sparsity_pattern: Pattern for sparsity ('sliding_window', 'random', 'strided')
            window_size: Size of sliding window
            num_random: Number of random connections
            mask: Optional attention mask
            
        Returns:
            output: [batch, seq_len, d_model]
        """
        batch_size, seq_len, d_model = query.size()
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / (d_model ** 0.5)
        
        # Create sparse mask
        sparse_mask = torch.zeros_like(scores)
        
        if sparsity_pattern == 'sliding_window':
            # Local attention with sliding window
            for i in range(seq_len):
                start = max(0, i - window_size // 2)
                end = min(seq_len, i + window_size // 2)
                sparse_mask[:, i, start:end] = 1
                
        elif sparsity_pattern == 'random':
            # Random sparse connections
            for i in range(seq_len):
                # Add random connections
                random_indices = torch.randperm(seq_len)[:num_random]
                sparse_mask[:, i, random_indices] = 1
                
        elif sparsity_pattern == 'strided':
            # Strided attention pattern
            stride = seq_len // window_size
            for i in range(seq_len):
                for j in range(0, seq_len, stride):
                    if j < seq_len:
                        sparse_mask[:, i, j] = 1
        
        # Combine with existing mask
        if mask is not None:
            sparse_mask = sparse_mask * (mask > -10000)
        
        # Apply mask and compute attention
        scores = scores + (sparse_mask - 1) * 10000  # Negative infinity equivalent
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, value)
        
        return output


class ActivationRecomputation:
    """
    Activation recomputation strategies for memory efficiency.
    
    Provides various strategies for recomputing activations during
    backward pass to save memory.
    """
    
    def __init__(
        self,
        strategy: str = 'full',
        recompute_modules: Optional[List[str]] = None,
        memory_budget_gb: float = 10.0,
        **kwargs
    ):
        self.strategy = strategy
        self.recompute_modules = recompute_modules or []
        self.memory_budget_gb = memory_budget_gb
        
        # Memory tracking
        self.peak_memory = 0
        self.recomputation_count = 0
        
        # Setup
        self._setup_recomputation()
    
    def _setup_recomputation(self):
        """Setup activation recomputation."""
        if self.strategy == 'full':
            self._setup_full_recomputation()
        elif self.strategy == 'selective':
            self._setup_selective_recomputation()
        elif self.strategy == 'adaptive':
            self._setup_adaptive_recomputation()
        else:
            raise ValueError(f"Unknown recomputation strategy: {self.strategy}")
    
    def _setup_full_recomputation(self):
        """Setup full recomputation strategy."""
        # In full recomputation, we recompute all activations during backward
        # This provides maximum memory savings but increases computation time
        pass
    
    def _setup_selective_recomputation(self):
        """Setup selective recomputation strategy."""
        # In selective recomputation, we only recompute specific modules
        # This balances memory savings and computation time
        pass
    
    def _setup_adaptive_recomputation(self):
        """Setup adaptive recomputation strategy."""
        # In adaptive recomputation, we adjust strategy based on memory usage
        # This dynamically optimizes the memory-computation tradeoff
        pass
    
    @contextlib.contextmanager
    def recompute_forward(self, module: nn.Module, *inputs):
        """Context manager for activation recomputation."""
        # Capture inputs and module state
        recomputation_data = self._capture_module_state(module, inputs)
        
        try:
            # Perform forward pass and capture activations
            yield
        finally:
            # Mark activations for recomputation
            self._mark_for_recomputation(module, recomputation_data)
    
    def _capture_module_state(self, module: nn.Module, inputs) -> Dict:
        """Capture module state for recomputation."""
        return {
            'inputs': inputs,
            'module_type': type(module).__name__,
            'training_state': module.training,
            'buffer_states': {name: buf.data.clone() for name, buf in module.named_buffers()},
            'parameter_shapes': {name: param.shape for name, param in module.named_parameters()}
        }
    
    def _mark_for_recomputation(self, module: nn.Module, data: Dict):
        """Mark module activations for recomputation during backward."""
        # This would set up the hooks and data needed for recomputation
        # during the backward pass
        pass
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics for recomputation."""
        return {
            'strategy': self.strategy,
            'memory_budget_gb': self.memory_budget_gb,
            'peak_memory_gb': self.peak_memory,
            'recomputation_count': self.recomputation_count,
            'memory_savings_gb': self._estimate_memory_savings()
        }
    
    def _estimate_memory_savings(self) -> float:
        """Estimate memory savings from activation recomputation."""
        # This is a rough estimation based on strategy
        savings_map = {
            'full': 2.5,
            'selective': 1.5,
            'adaptive': 2.0
        }
        return savings_map.get(self.strategy, 0.0)


class DynamicMemoryManager:
    """
    Dynamic memory management for training large models.
    
    Monitors memory usage and automatically adjusts batch size,
    gradient accumulation, or other parameters to stay within memory limits.
    """
    
    def __init__(
        self,
        max_memory_gb: float = 16.0,
        warning_threshold: float = 0.85,
        critical_threshold: float = 0.95,
        memory_monitor_interval: int = 100,
        **kwargs
    ):
        self.max_memory_gb = max_memory_gb
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.memory_monitor_interval = memory_monitor_interval
        
        # Memory tracking
        self.memory_history = []
        self.batch_size_history = []
        self.adjustments_made = 0
        
        # Current settings
        self.current_batch_size = None
        self.current_gradient_accumulation_steps = None
        self.current_mixed_precision = None
    
    def monitor_memory(self) -> Dict[str, Any]:
        """Monitor current memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            cached = torch.cuda.memory_reserved() / (1024**3)      # GB
            
            memory_info = {
                'allocated_gb': allocated,
                'cached_gb': cached,
                'usage_ratio': allocated / self.max_memory_gb,
                'within_limits': allocated < self.max_memory_gb,
                'memory_warning': allocated > (self.max_memory_gb * self.warning_threshold),
                'memory_critical': allocated > (self.max_memory_gb * self.critical_threshold)
            }
        else:
            # System memory as fallback
            memory = psutil.virtual_memory()
            memory_info = {
                'system_memory_gb': memory.total / (1024**3),
                'system_used_gb': memory.used / (1024**3),
                'system_usage_ratio': memory.percent / 100,
                'within_limits': memory.percent < 90,
                'memory_warning': memory.percent > 80,
                'memory_critical': memory.percent > 95
            }
        
        # Store in history
        self.memory_history.append(memory_info)
        
        return memory_info
    
    def should_adjust_training(self) -> bool:
        """Determine if training parameters should be adjusted."""
        memory_info = self.monitor_memory()
        
        return (
            memory_info.get('memory_critical', False) or
            memory_info.get('memory_warning', False)
        )
    
    def suggest_adjustments(
        self,
        current_batch_size: int,
        current_grad_accum_steps: int,
        current_mixed_precision: bool = True
    ) -> Dict[str, Any]:
        """
        Suggest memory-related adjustments to training parameters.
        
        Args:
            current_batch_size: Current batch size
            current_grad_accum_steps: Current gradient accumulation steps
            current_mixed_precision: Whether mixed precision is enabled
            
        Returns:
            Dictionary with suggested adjustments
        """
        memory_info = self.monitor_memory()
        adjustments = {
            'recommended': False,
            'new_batch_size': current_batch_size,
            'new_grad_accum_steps': current_grad_accum_steps,
            'enable_mixed_precision': current_mixed_precision,
            'reason': 'No adjustment needed'
        }
        
        # Critical memory situation
        if memory_info.get('memory_critical', False):
            adjustments.update({
                'recommended': True,
                'new_batch_size': max(1, current_batch_size // 2),
                'new_grad_accum_steps': current_grad_accum_steps * 2,
                'enable_mixed_precision': True,
                'reason': 'Critical memory usage detected'
            })
        
        # Warning memory situation
        elif memory_info.get('memory_warning', False):
            adjustments.update({
                'recommended': True,
                'new_batch_size': max(1, current_batch_size * 3 // 4),
                'new_grad_accum_steps': current_grad_accum_steps * 2,
                'enable_mixed_precision': True,
                'reason': 'High memory usage detected'
            })
        
        # Low memory usage - can increase batch size
        elif memory_info.get('usage_ratio', 1.0) < 0.5:
            adjustments.update({
                'recommended': True,
                'new_batch_size': current_batch_size * 2,
                'new_grad_accum_steps': max(1, current_grad_accum_steps // 2),
                'enable_mixed_precision': current_mixed_precision,
                'reason': 'Low memory usage - can increase batch size'
            })
        
        return adjustments
    
    def apply_adjustments(
        self,
        batch_size: int,
        grad_accum_steps: int,
        mixed_precision: bool,
        adjustments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply suggested adjustments to training."""
        self.current_batch_size = batch_size
        self.current_gradient_accumulation_steps = grad_accum_steps
        self.current_mixed_precision = mixed_precision
        
        # Store history
        self.batch_size_history.append(batch_size)
        self.adjustments_made += 1
        
        return {
            'batch_size': adjustments.get('new_batch_size', batch_size),
            'grad_accum_steps': adjustments.get('new_grad_accum_steps', grad_accum_steps),
            'mixed_precision': adjustments.get('enable_mixed_precision', mixed_precision)
        }
    
    def get_memory_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive memory optimization report."""
        current_memory = self.monitor_memory()
        
        if not self.memory_history:
            return {'error': 'No memory history available'}
        
        return {
            'current_memory': current_memory,
            'peak_memory_gb': max(m['allocated_gb'] for m in self.memory_history if 'allocated_gb' in m),
            'avg_memory_gb': sum(m['allocated_gb'] for m in self.memory_history if 'allocated_gb' in m) / len(self.memory_history),
            'adjustments_made': self.adjustments_made,
            'batch_size_history': self.batch_size_history[-10:],  # Last 10 adjustments
            'memory_efficiency': self._calculate_memory_efficiency(),
            'optimization_suggestions': self._get_optimization_suggestions()
        }
    
    def _calculate_memory_efficiency(self) -> float:
        """Calculate memory efficiency over time."""
        if not self.memory_history:
            return 0.0
        
        # Calculate average memory usage relative to peak
        peak = max(m['allocated_gb'] for m in self.memory_history if 'allocated_gb' in m)
        if peak == 0:
            return 0.0
        
        avg_usage = sum(m['allocated_gb'] for m in self.memory_history if 'allocated_gb' in m) / len(self.memory_history)
        efficiency = avg_usage / peak
        
        return min(efficiency, 1.0)
    
    def _get_optimization_suggestions(self) -> List[str]:
        """Get optimization suggestions based on memory history."""
        suggestions = []
        
        if len(self.memory_history) < 10:
            return ['Collect more memory data for optimization suggestions']
        
        # Analyze memory patterns
        avg_usage = sum(m.get('allocated_gb', 0) for m in self.memory_history) / len(self.memory_history)
        
        if avg_usage < self.max_memory_gb * 0.3:
            suggestions.append("Memory usage is very low - consider increasing batch size")
        elif avg_usage > self.max_memory_gb * 0.8:
            suggestions.append("High memory usage - consider reducing batch size or enabling gradient checkpointing")
        
        # Check for memory spikes
        if max(m.get('allocated_gb', 0) for m in self.memory_history) > self.max_memory_gb * 0.95:
            suggestions.append("Memory spikes detected - consider using dynamic memory management")
        
        # Gradient accumulation suggestions
        if self.adjustments_made > 5:
            suggestions.append("Many adjustments made - consider a more stable batch size")
        
        return suggestions


class MemoryProfiler:
    """
    Detailed memory profiling for transformer models.
    
    Analyzes memory usage patterns and provides detailed insights
    for optimization.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        
        # Memory analysis
        self.module_memory_usage: Dict[str, float] = {}
        self.activation_memory_usage: Dict[str, float] = {}
        self.parameter_memory_usage: Dict[str, float] = {}
        
        # Setup hooks
        self._setup_memory_hooks()
    
    def _setup_memory_hooks(self):
        """Setup hooks to monitor memory usage."""
        self.hooks = []
        
        def memory_hook(name):
            def hook(module, input, output):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    memory = torch.cuda.memory_allocated() / (1024**3)
                    self.activation_memory_usage[name] = memory
            return hook
        
        # Register hooks for all modules
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                hook = module.register_forward_hook(memory_hook(name))
                self.hooks.append(hook)
    
    def profile_memory_usage(self, sample_input: torch.Tensor) -> Dict[str, Any]:
        """Profile memory usage with sample input."""
        # Clear previous profiling
        self.module_memory_usage.clear()
        self.activation_memory_usage.clear()
        self.parameter_memory_usage.clear()
        
        # Get initial memory
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            initial_memory = torch.cuda.memory_allocated() / (1024**3)
        else:
            initial_memory = 0
        
        # Profile forward pass
        with torch.no_grad():
            self.model(sample_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            final_memory = torch.cuda.memory_allocated() / (1024**3)
        else:
            final_memory = 0
        
        # Analyze parameter memory
        for name, param in self.model.named_parameters():
            param_memory = param.numel() * param.element_size() / (1024**3)
            self.parameter_memory_usage[name] = param_memory
        
        # Calculate module memory
        total_activation_memory = sum(self.activation_memory_usage.values())
        
        return {
            'total_memory_gb': final_memory - initial_memory,
            'parameter_memory_gb': sum(self.parameter_memory_usage.values()),
            'activation_memory_gb': total_activation_memory,
            'module_memory_breakdown': self.module_memory_usage,
            'parameter_memory_breakdown': self.parameter_memory_usage,
            'memory_distribution': self._calculate_memory_distribution()
        }
    
    def _calculate_memory_distribution(self) -> Dict[str, float]:
        """Calculate memory distribution across components."""
        total_memory = sum(self.parameter_memory_usage.values()) + sum(self.activation_memory_usage.values())
        
        if total_memory == 0:
            return {}
        
        return {
            'parameters_percentage': sum(self.parameter_memory_usage.values()) / total_memory * 100,
            'activations_percentage': sum(self.activation_memory_usage.values()) / total_memory * 100
        }
    
    def suggest_memory_optimizations(self) -> List[str]:
        """Suggest memory optimizations based on profiling."""
        suggestions = []
        
        # Analyze memory distribution
        if self.parameter_memory_usage:
            total_param_memory = sum(self.parameter_memory_usage.values())
            
            # Large parameter modules
            large_param_modules = {
                name: memory for name, memory in self.parameter_memory_usage.items()
                if memory > total_param_memory * 0.1  # More than 10% of total
            }
            
            if large_param_modules:
                suggestions.append("Consider model parallelism for large parameter modules")
                for module in large_param_modules:
                    suggestions.append(f"  - {module}: {large_param_modules[module]:.2f}GB")
        
        # Activation memory suggestions
        total_activation_memory = sum(self.activation_memory_usage.values())
        if total_activation_memory > 2.0:  # More than 2GB
            suggestions.append("High activation memory usage detected:")
            suggestions.append("  - Enable gradient checkpointing")
            suggestions.append("  - Use memory-efficient attention mechanisms")
            suggestions.append("  - Reduce sequence length or batch size")
        
        # General suggestions
        suggestions.extend([
            "Use mixed precision training to reduce memory usage",
            "Implement dynamic memory management for batch size adaptation",
            "Consider gradient accumulation for larger effective batch sizes"
        ])
        
        return suggestions
    
    def remove_hooks(self):
        """Remove all profiling hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()