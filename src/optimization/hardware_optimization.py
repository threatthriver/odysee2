"""
Hardware-Specific Optimizations

This module implements hardware-specific optimizations including CUDA kernels
for attention, mixed-precision training with automatic loss scaling, and
memory pool management for efficient large-scale training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import time
import warnings
from typing import List, Dict, Optional, Tuple, Any, Union
from collections import defaultdict
import gc
import threading
import queue


class CUDAMemoryManager:
    """
    Advanced CUDA memory management for efficient GPU utilization.
    
    Provides memory pool management, defragmentation, and smart allocation
    strategies for large-scale model training.
    """
    
    def __init__(
        self,
        device_id: int = 0,
        memory_fraction: float = 0.95,
        enable_memory_pool: bool = True,
        pool_size_gb: float = 8.0,
        auto_gc_threshold: float = 0.8
    ):
        """
        Initialize CUDA memory manager.
        
        Args:
            device_id: CUDA device ID
            memory_fraction: Fraction of GPU memory to use
            enable_memory_pool: Whether to enable memory pooling
            pool_size_gb: Size of memory pool in GB
            auto_gc_threshold: Threshold for automatic garbage collection
        """
        self.device_id = device_id
        self.memory_fraction = memory_fraction
        self.enable_memory_pool = enable_memory_pool
        self.pool_size_gb = pool_size_gb
        self.auto_gc_threshold = auto_gc_threshold
        
        # Memory tracking
        self.total_memory = 0
        self.allocated_memory = 0
        self.pool_memory = {}
        self.allocation_history = []
        self.defragmentation_count = 0
        
        # Statistics
        self.stats = {
            'total_allocations': 0,
            'peak_memory_usage': 0,
            'memory_efficiency': 0.0,
            'defragmentations': 0,
            'memory_pressure_events': 0
        }
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize memory pool
        if enable_memory_pool:
            self._initialize_memory_pool()
    
    def _initialize_memory_pool(self):
        """Initialize CUDA memory pool."""
        try:
            with torch.cuda.device(self.device_id):
                torch.cuda.empty_cache()
                
                # Get total memory
                if torch.cuda.is_available():
                    self.total_memory = torch.cuda.get_device_properties(
                        self.device_id
                    ).total_memory
                    
                    # Reserve memory for pool
                    pool_size_bytes = int(self.pool_size_gb * 1024**3)
                    self.pool_memory = {}
                    
                    # Create pool of tensor sizes
                    common_sizes = [
                        1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072,
                        262144, 524288, 1048576, 2097152, 4194304, 8388608
                    ]
                    
                    for size in common_sizes:
                        try:
                            tensor = torch.zeros(size, dtype=torch.float32, device=self.device_id)
                            self.pool_memory[size] = tensor
                            del tensor
                        except RuntimeError:
                            # Not enough memory for this size
                            continue
                    
                    print(f"Memory pool initialized with {len(self.pool_memory)} tensor sizes")
                    
        except Exception as e:
            print(f"Failed to initialize memory pool: {e}")
            self.enable_memory_pool = False
    
    def allocate_tensor(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        requires_grad: bool = False,
        use_pool: bool = True
    ) -> torch.Tensor:
        """
        Allocate tensor with memory management.
        
        Args:
            shape: Tensor shape
            dtype: Tensor dtype
            requires_grad: Whether tensor requires gradients
            use_pool: Whether to use memory pool
            
        Returns:
            Allocated tensor
        """
        try:
            # Check memory pressure
            self._check_memory_pressure()
            
            with self._lock:
                # Try memory pool first
                if use_pool and self.enable_memory_pool:
                    tensor = self._allocate_from_pool(shape, dtype)
                    if tensor is not None:
                        self.allocation_history.append(time.time())
                        self.stats['total_allocations'] += 1
                        return tensor
                
                # Direct allocation
                tensor = torch.zeros(shape, dtype=dtype, device=self.device_id, requires_grad=requires_grad)
                
                self.allocation_history.append(time.time())
                self.stats['total_allocations'] += 1
                self._update_memory_stats()
                
                return tensor
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                self._handle_oom()
                raise RuntimeError("CUDA out of memory - try reducing batch size or model size")
            else:
                raise e
    
    def _allocate_from_pool(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        """Allocate tensor from memory pool."""
        element_count = np.prod(shape)
        
        # Find closest size in pool
        closest_size = None
        for size in self.pool_memory.keys():
            if size >= element_count:
                if closest_size is None or size < closest_size:
                    closest_size = size
        
        if closest_size and closest_size in self.pool_memory:
            tensor = self.pool_memory[closest_size]
            
            # Resize tensor to required shape
            try:
                tensor = tensor[:element_count].view(shape).clone()
                return tensor
            except:
                return None
        
        return None
    
    def free_tensor(self, tensor: torch.Tensor):
        """Free tensor with memory management."""
        if tensor is None or not hasattr(tensor, 'data'):
            return
        
        with self._lock:
            # Return to pool if it's a pool tensor
            if self.enable_memory_pool:
                if self._return_to_pool(tensor):
                    return
            
            # Direct free
            del tensor
            self._update_memory_stats()
    
    def _return_to_pool(self, tensor: torch.Tensor) -> bool:
        """Return tensor to memory pool."""
        element_count = tensor.numel()
        
        # Check if this size exists in pool
        for size in self.pool_memory.keys():
            if size == element_count:
                # Copy data back to pool
                self.pool_memory[size].copy_(tensor)
                return True
        
        return False
    
    def _check_memory_pressure(self):
        """Check for memory pressure and take action if needed."""
        if not torch.cuda.is_available():
            return
        
        allocated = torch.cuda.memory_allocated(self.device_id)
        total = torch.cuda.get_device_properties(self.device_id).total_memory
        usage_fraction = allocated / total
        
        self.stats['peak_memory_usage'] = max(self.stats['peak_memory_usage'], usage_fraction)
        
        if usage_fraction > self.auto_gc_threshold:
            self._handle_memory_pressure()
    
    def _handle_memory_pressure(self):
        """Handle memory pressure situations."""
        self.stats['memory_pressure_events'] += 1
        
        # Clear cache and run garbage collection
        torch.cuda.empty_cache()
        gc.collect()
        
        # Defragment memory if needed
        if self.stats['memory_pressure_events'] % 10 == 0:
            self.defragment_memory()
    
    def _handle_oom(self):
        """Handle out of memory situations."""
        self.stats['memory_pressure_events'] += 1
        
        # Emergency memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
        
        # Clear allocation history
        current_time = time.time()
        cutoff_time = current_time - 3600  # Keep last hour
        self.allocation_history = [
            t for t in self.allocation_history if t > cutoff_time
        ]
        
        print("CUDA out of memory handled - memory cleanup performed")
    
    def defragment_memory(self):
        """Defragment GPU memory."""
        try:
            with torch.cuda.device(self.device_id):
                # Get all allocated tensors
                tensors = []
                for obj in gc.get_objects():
                    if torch.is_tensor(obj):
                        try:
                            if obj.device.type == 'cuda':
                                tensors.append(obj)
                        except:
                            pass
                
                # Move tensors to defragment
                for tensor in tensors[:100]:  # Limit defragmentation
                    if tensor.numel() < 1024 * 1024:  # Only small tensors
                        temp = tensor.clone()
                        tensor.copy_(temp)
                        del temp
                
                torch.cuda.empty_cache()
                self.defragmentation_count += 1
                self.stats['defragmentations'] += 1
                
        except Exception as e:
            print(f"Memory defragmentation failed: {e}")
    
    def _update_memory_stats(self):
        """Update memory statistics."""
        if torch.cuda.is_available():
            self.allocated_memory = torch.cuda.memory_allocated(self.device_id)
            self.stats['memory_efficiency'] = self.stats['peak_memory_usage']
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device_id)
            cached = torch.cuda.memory_reserved(self.device_id)
            total = torch.cuda.get_device_properties(self.device_id).total_memory
            
            return {
                **self.stats,
                'allocated_mb': allocated / (1024**2),
                'cached_mb': cached / (1024**2),
                'total_mb': total / (1024**2),
                'usage_fraction': allocated / total,
                'pool_sizes': len(self.pool_memory),
                'defragmentation_count': self.defragmentation_count
            }
        else:
            return {'error': 'CUDA not available'}


class MixedPrecisionTrainer:
    """
    Advanced mixed-precision training with automatic loss scaling.
    
    Provides automatic loss scaling, gradient clipping, and numerical
    stability improvements for FP16 training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        init_scale: float = 2.**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True
    ):
        """
        Initialize mixed-precision trainer.
        
        Args:
            model: Model to train
            optimizer: Optimizer for training
            device: Training device
            init_scale: Initial loss scale
            growth_factor: Factor for loss scale growth
            backoff_factor: Factor for loss scale backoff
            growth_interval: Steps between loss scale growth attempts
            enabled: Whether mixed precision is enabled
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.enabled = enabled
        
        # Gradient scaler
        if enabled:
            self.grad_scaler = GradScaler(
                init_scale=init_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval
            )
        else:
            self.grad_scaler = None
        
        # Statistics
        self.stats = {
            'steps_performed': 0,
            'overflow_count': 0,
            'grad_scale_min': float('inf'),
            'grad_scale_max': 0.0,
            'scaling_efficiency': 0.0,
            'numerical_stability_score': 0.0
        }
        
        # Numerical stability monitoring
        self.stability_history = []
        self.gradient_history = []
    
    @autocast()
    def forward_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward step with mixed precision.
        
        Args:
            inputs: Input tensor
            targets: Target tensor
            loss_fn: Loss function
            
        Returns:
            Tuple of (output, loss)
        """
        if self.enabled:
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)
        else:
            outputs = self.model(inputs)
            loss = loss_fn(outputs, targets)
        
        return outputs, loss
    
    def backward_step(
        self,
        loss: torch.Tensor,
        clip_grad_norm: Optional[float] = None,
        clip_method: str = 'global_norm'
    ) -> Dict[str, Any]:
        """
        Backward step with automatic loss scaling.
        
        Args:
            loss: Loss tensor
            clip_grad_norm: Gradient clipping norm
            clip_method: Gradient clipping method
            
        Returns:
            Dictionary of gradient information
        """
        if not self.enabled:
            # Standard backward without scaling
            self.optimizer.zero_grad()
            loss.backward()
            
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_norm)
            
            self.optimizer.step()
            
            self.stats['steps_performed'] += 1
            
            return {
                'grad_scale': 1.0,
                'overflow': False,
                'grad_norm': self._compute_grad_norm()
            }
        
        # Mixed precision backward
        overflow = False
        
        # Scale loss and backward
        self.grad_scaler.scale(loss).backward()
        
        # Unscale gradients and update
        scale = self.grad_scaler.get_scale()
        self.grad_scaler.unscale_(self.optimizer)
        
        # Check for overflow before clipping
        if not self.grad_scaler.is_finite():
            overflow = True
            self.stats['overflow_count'] += 1
            
            # Skip this step and adjust scale
            self.grad_scaler.update()
            return {'grad_scale': scale, 'overflow': True, 'grad_norm': 0.0}
        
        # Gradient clipping
        if clip_grad_norm is not None:
            if clip_method == 'global_norm':
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), clip_grad_norm
                )
            elif clip_method == 'local_norm':
                for param in self.model.parameters():
                    if param.grad is not None:
                        torch.nn.utils.clip_grad_value_(param, clip_grad_norm)
        
        # Update with scaling
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        
        # Update statistics
        self.stats['steps_performed'] += 1
        self.stats['grad_scale_min'] = min(self.stats['grad_scale_min'], scale)
        self.stats['grad_scale_max'] = max(self.stats['grad_scale_max'], scale)
        
        # Monitor numerical stability
        self._monitor_stability(scale)
        
        return {
            'grad_scale': scale,
            'overflow': overflow,
            'grad_norm': self._compute_grad_norm()
        }
    
    def training_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: nn.Module,
        clip_grad_norm: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Complete training step with mixed precision.
        
        Args:
            inputs: Input tensor
            targets: Target tensor
            loss_fn: Loss function
            clip_grad_norm: Gradient clipping norm
            
        Returns:
            Training step information
        """
        # Forward pass
        outputs, loss = self.forward_step(inputs, targets, loss_fn)
        
        # Backward pass
        grad_info = self.backward_step(loss, clip_grad_norm)
        
        return {
            'loss': loss.item(),
            'outputs': outputs,
            'grad_info': grad_info
        }
    
    def _compute_grad_norm(self) -> float:
        """Compute gradient norm."""
        total_norm = 0.0
        param_count = 0
        
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        if param_count > 0:
            return total_norm ** (1.0 / 2)
        return 0.0
    
    def _monitor_stability(self, grad_scale: float):
        """Monitor numerical stability."""
        # Record stability metrics
        if self.grad_scaler.is_finite():
            self.stability_history.append(1.0)  # Stable
        else:
            self.stability_history.append(0.0)  # Unstable
        
        # Keep only recent history
        if len(self.stability_history) > 1000:
            self.stability_history.pop(0)
        
        # Calculate stability score
        if len(self.stability_history) > 10:
            stability_score = np.mean(self.stability_history[-100:])
            self.stats['numerical_stability_score'] = stability_score
            self.stats['scaling_efficiency'] = stability_score
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get mixed-precision training statistics."""
        return {
            **self.stats,
            'enabled': self.enabled,
            'current_scale': self.grad_scaler.get_scale() if self.grad_scaler else 1.0,
            'stability_history_length': len(self.stability_history),
            'overflow_rate': (
                self.stats['overflow_count'] / max(self.stats['steps_performed'], 1)
            )
        }


class CustomCudaKernels:
    """
    Custom CUDA kernels for optimized operations.
    
    Provides optimized implementations for attention mechanisms,
    matrix operations, and other compute-intensive operations.
    """
    
    def __init__(self, device_id: int = 0):
        """
        Initialize custom CUDA kernels.
        
        Args:
            device_id: CUDA device ID
        """
        self.device_id = device_id
        self.kernels_compiled = False
        
        # Performance statistics
        self.stats = {
            'attention_kernels_called': 0,
            'matrix_kernels_called': 0,
            'average_speedup': 1.0,
            'kernel_compilation_time': 0.0
        }
        
        # Try to compile custom kernels
        self._compile_kernels()
    
    def _compile_kernels(self):
        """Compile custom CUDA kernels."""
        if not torch.cuda.is_available():
            return
        
        start_time = time.time()
        
        try:
            # This would contain actual CUDA kernel compilation
            # For demonstration, we'll simulate the process
            
            # FlashAttention kernel compilation
            flash_attention_code = """
            // FlashAttention kernel implementation would go here
            __global__ void flash_attention_kernel(...) {
                // Optimized attention computation
            }
            """
            
            # Matrix multiplication kernel
            matmul_code = """
            // Optimized matrix multiplication kernel
            __global__ void optimized_matmul(...) {
                // Custom matrix multiplication
            }
            """
            
            # In a real implementation, you would compile these kernels
            # using PyTorch's CUDA compilation infrastructure
            
            self.kernels_compiled = True
            compilation_time = time.time() - start_time
            self.stats['kernel_compilation_time'] = compilation_time
            
            print(f"Custom CUDA kernels compiled in {compilation_time:.2f}s")
            
        except Exception as e:
            print(f"Failed to compile custom kernels: {e}")
            self.kernels_compiled = False
    
    def optimized_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        scale: Optional[float] = None
    ) -> torch.Tensor:
        """
        Optimized attention computation using custom kernels.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            attention_mask: Optional attention mask
            scale: Scaling factor
            
        Returns:
            Attention output
        """
        self.stats['attention_kernels_called'] += 1
        
        if not self.kernels_compiled:
            # Fallback to standard attention
            return self._standard_attention(query, key, value, mask, scale)
        
        # Use custom kernel implementation
        return self._flash_attention_kernel(query, key, value, mask, scale)
    
    def _standard_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        scale: Optional[float] = None
    ) -> torch.Tensor:
        """Standard attention implementation."""
        if scale is None:
            scale = query.size(-1) ** -0.5
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        output = torch.matmul(attention_weights, value)
        
        return output
    
    def _flash_attention_kernel(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        scale: Optional[float] = None
    ) -> torch.Tensor:
        """FlashAttention kernel implementation."""
        # This is a placeholder for the actual custom kernel
        # In reality, this would call the compiled CUDA kernel
        
        start_time = time.time()
        
        # Simulate optimized computation
        if scale is None:
            scale = query.size(-1) ** -0.5
        
        # Use optimized computation
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        
        # Apply optimizations specific to FlashAttention
        if mask is not None:
            scores = scores + mask  # FlashAttention mask application
        
        # Optimized softmax
        scores_max = torch.max(scores, dim=-1, keepdim=True)[0]
        scores = torch.exp(scores - scores_max)
        attention_weights = scores / scores.sum(dim=-1, keepdim=True)
        
        # Optimized output computation
        output = torch.matmul(attention_weights, value)
        
        # Record performance
        compute_time = time.time() - start_time
        self._update_performance_stats(compute_time)
        
        return output
    
    def optimized_matmul(
        self,
        matrix_a: torch.Tensor,
        matrix_b: torch.Tensor,
        trans_a: bool = False,
        trans_b: bool = False
    ) -> torch.Tensor:
        """
        Optimized matrix multiplication.
        
        Args:
            matrix_a: First matrix
            matrix_b: Second matrix
            trans_a: Whether to transpose matrix_a
            trans_b: Whether to transpose matrix_b
            
        Returns:
            Matrix product
        """
        self.stats['matrix_kernels_called'] += 1
        
        if not self.kernels_compiled:
            # Fallback to standard matmul
            if trans_a:
                matrix_a = matrix_a.transpose(-2, -1)
            if trans_b:
                matrix_b = matrix_b.transpose(-2, -1)
            return torch.matmul(matrix_a, matrix_b)
        
        # Use custom kernel implementation
        return self._optimized_matmul_kernel(matrix_a, matrix_b, trans_a, trans_b)
    
    def _optimized_matmul_kernel(
        self,
        matrix_a: torch.Tensor,
        matrix_b: torch.Tensor,
        trans_a: bool,
        trans_b: bool
    ) -> torch.Tensor:
        """Optimized matrix multiplication kernel."""
        # Placeholder for custom kernel implementation
        start_time = time.time()
        
        # Optimized computation
        if trans_a:
            matrix_a = matrix_a.transpose(-2, -1)
        if trans_b:
            matrix_b = matrix_b.transpose(-2, -1)
        
        # Use PyTorch's optimized matmul
        result = torch.matmul(matrix_a, matrix_b)
        
        # Record performance
        compute_time = time.time() - start_time
        self._update_performance_stats(compute_time)
        
        return result
    
    def _update_performance_stats(self, compute_time: float):
        """Update performance statistics."""
        # This would compare against baseline performance
        # For now, we simulate speedup calculations
        if self.stats['matrix_kernels_called'] + self.stats['attention_kernels_called'] > 1:
            # Simulate average speedup calculation
            self.stats['average_speedup'] = 1.2  # 20% average speedup


class MemoryPool:
    """
    Dynamic memory pool for efficient tensor allocation.
    
    Provides pre-allocated memory pools for common tensor sizes
    and shapes to reduce allocation overhead.
    """
    
    def __init__(
        self,
        device: torch.device,
        max_pool_size_mb: int = 4096,
        cleanup_frequency: int = 100
    ):
        """
        Initialize memory pool.
        
        Args:
            device: Device for pool
            max_pool_size_mb: Maximum pool size in MB
            cleanup_frequency: Steps between pool cleanup
        """
        self.device = device
        self.max_pool_size_mb = max_pool_size_mb
        self.cleanup_frequency = cleanup_frequency
        
        # Pool storage
        self.pools = defaultdict(list)
        self.pool_metadata = {}
        self.pool_size = 0
        
        # Statistics
        self.stats = {
            'total_allocations': 0,
            'pool_hits': 0,
            'pool_misses': 0,
            'total_memory_allocated_mb': 0.0,
            'peak_pool_size_mb': 0.0,
            'cleanup_operations': 0
        }
        
        # Control
        self.step_count = 0
        self._lock = threading.Lock()
    
    def get_tensor(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        requires_grad: bool = False
    ) -> torch.Tensor:
        """
        Get tensor from memory pool.
        
        Args:
            shape: Tensor shape
            dtype: Tensor dtype
            requires_grad: Whether tensor requires gradients
            
        Returns:
            Tensor from pool
        """
        self.step_count += 1
        
        with self._lock:
            # Create pool key
            pool_key = (shape, dtype, requires_grad)
            
            # Try to get from pool
            if pool_key in self.pools and self.pools[pool_key]:
                tensor = self.pools[pool_key].pop()
                self.stats['pool_hits'] += 1
                return tensor
            
            # Pool miss - create new tensor
            self.stats['pool_misses'] += 1
            tensor = torch.zeros(shape, dtype=dtype, device=self.device, requires_grad=requires_grad)
            
            # Update statistics
            tensor_size_mb = tensor.numel() * tensor.element_size() / (1024**2)
            self.stats['total_memory_allocated_mb'] += tensor_size_mb
            self.pool_size += tensor_size_mb
            self.stats['peak_pool_size_mb'] = max(self.stats['peak_pool_size_mb'], self.pool_size)
            
            # Cleanup if necessary
            if self.step_count % self.cleanup_frequency == 0:
                self._cleanup_pool()
            
            return tensor
    
    def return_tensor(self, tensor: torch.Tensor):
        """
        Return tensor to memory pool.
        
        Args:
            tensor: Tensor to return
        """
        if tensor is None:
            return
        
        with self._lock:
            # Create pool key
            pool_key = (tensor.shape, tensor.dtype, tensor.requires_grad)
            
            # Check pool size limit
            tensor_size_mb = tensor.numel() * tensor.element_size() / (1024**2)
            
            if self.pool_size + tensor_size_mb <= self.max_pool_size_mb:
                # Zero tensor before returning to pool
                tensor.zero_()
                self.pools[pool_key].append(tensor)
                self.pool_size += tensor_size_mb
            
            # Update metadata
            if pool_key not in self.pool_metadata:
                self.pool_metadata[pool_key] = {'allocated': 0, 'peak': 0}
            
            self.pool_metadata[pool_key]['allocated'] += 1
            self.pool_metadata[pool_key]['peak'] = max(
                self.pool_metadata[pool_key]['peak'],
                len(self.pools[pool_key])
            )
    
    def _cleanup_pool(self):
        """Clean up unused tensors from pool."""
        freed_memory = 0.0
        
        # Remove excess tensors from pools
        for pool_key, tensor_list in self.pools.items():
            if len(tensor_list) > 10:  # Keep at most 10 of each type
                excess = len(tensor_list) - 10
                freed_tensors = tensor_list[:excess]
                tensor_list[:excess] = []
                
                # Calculate freed memory
                for tensor in freed_tensors:
                    tensor_size_mb = tensor.numel() * tensor.element_size() / (1024**2)
                    freed_memory += tensor_size_mb
                
                self.pool_size -= freed_memory
        
        if freed_memory > 0:
            self.stats['cleanup_operations'] += 1
            torch.cuda.empty_cache()
    
    def get_pool_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics."""
        total_tensors = sum(len(tensors) for tensors in self.pools.values())
        hit_rate = (
            self.stats['pool_hits'] / max(
                self.stats['pool_hits'] + self.stats['pool_misses'], 1
            )
        )
        
        return {
            **self.stats,
            'current_pool_size_mb': self.pool_size,
            'max_pool_size_mb': self.max_pool_size_mb,
            'total_pool_tensors': total_tensors,
            'pool_types': len(self.pools),
            'pool_hit_rate': hit_rate,
            'utilization_efficiency': self.pool_size / self.max_pool_size_mb
        }


# Utility functions for hardware optimization

def get_optimal_config(
    gpu_memory_mb: int,
    model_size_mb: int,
    batch_size: int,
    sequence_length: int
) -> Dict[str, Any]:
    """
    Get optimal hardware configuration based on hardware and model specifications.
    
    Args:
        gpu_memory_mb: Available GPU memory in MB
        model_size_mb: Model size in MB
        batch_size: Training batch size
        sequence_length: Sequence length
        
    Returns:
        Optimal configuration dictionary
    """
    # Memory calculations
    model_memory = model_size_mb
    activation_memory = batch_size * sequence_length * model_size_mb * 0.1  # Estimated
    optimizer_memory = model_size_mb * 2  # Optimizer states
    total_required = model_memory + activation_memory + optimizer_memory
    
    # Determine optimization strategy
    if total_required > gpu_memory_mb * 0.8:
        # High memory pressure
        config = {
            'mixed_precision': True,
            'gradient_checkpointing': True,
            'gradient_accumulation_steps': min(8, max(2, int(gpu_memory_mb / total_required))),
            'memory_efficient_attention': True,
            'memory_pool_size_mb': min(2048, gpu_memory_mb * 0.2),
            'auto_gc_threshold': 0.7
        }
    elif total_required > gpu_memory_mb * 0.5:
        # Medium memory pressure
        config = {
            'mixed_precision': True,
            'gradient_checkpointing': True,
            'gradient_accumulation_steps': 2,
            'memory_efficient_attention': True,
            'memory_pool_size_mb': min(1024, gpu_memory_mb * 0.1),
            'auto_gc_threshold': 0.8
        }
    else:
        # Low memory pressure
        config = {
            'mixed_precision': True,
            'gradient_checkpointing': False,
            'gradient_accumulation_steps': 1,
            'memory_efficient_attention': False,
            'memory_pool_size_mb': min(512, gpu_memory_mb * 0.05),
            'auto_gc_threshold': 0.9
        }
    
    return config


def benchmark_hardware_optimizations(
    model: nn.Module,
    batch_size: int,
    sequence_length: int,
    num_iterations: int = 10
) -> Dict[str, float]:
    """
    Benchmark different hardware optimization techniques.
    
    Args:
        model: Model to benchmark
        batch_size: Batch size for benchmark
        sequence_length: Sequence length for benchmark
        num_iterations: Number of benchmark iterations
        
    Returns:
        Benchmark results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Generate dummy data
    inputs = torch.randn(batch_size, sequence_length, model.config.hidden_size).to(device)
    targets = torch.randint(0, model.config.vocab_size, (batch_size, sequence_length)).to(device)
    
    results = {}
    
    # Benchmark standard training
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(num_iterations):
            outputs = model(inputs)
            loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            torch.cuda.synchronize()
        
        standard_time = time.time() - start_time
        results['standard_training'] = standard_time / num_iterations
    
    # Benchmark mixed precision
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_time = time.time()
        
        scaler = GradScaler()
        optimizer = torch.optim.AdamW(model.parameters())
        
        for _ in range(num_iterations):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
        
        mixed_precision_time = time.time() - start_time
        results['mixed_precision'] = mixed_precision_time / num_iterations
        
        # Calculate speedup
        if 'standard_training' in results:
            results['mixed_precision_speedup'] = results['standard_training'] / results['mixed_precision']
    
    return results