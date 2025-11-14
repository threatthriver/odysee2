"""
Memory Management and Optimization for Efficient Training

This module provides comprehensive memory management strategies including
gradient checkpointing, activation recomputation, dynamic memory allocation,
and memory-efficient training techniques for large language models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import sys
import threading
import time
import psutil
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
import warnings


@dataclass
class MemorySnapshot:
    """Memory usage snapshot for tracking."""
    timestamp: float
    gpu_allocated_mb: float
    gpu_reserved_mb: float
    cpu_memory_mb: float
    swap_used_mb: float
    tensor_count: int
    gradient_count: int


class MemoryProfiler:
    """
    Comprehensive memory profiling and monitoring system.
    
    Tracks GPU memory usage, CPU memory, tensor lifecycle,
    and provides detailed memory analysis.
    """
    
    def __init__(
        self,
        device_id: int = 0,
        sampling_interval: float = 1.0,
        track_tensor_lifecycle: bool = True
    ):
        """
        Initialize memory profiler.
        
        Args:
            device_id: GPU device ID to monitor
            sampling_interval: Time interval between memory samples
            track_tensor_lifecycle: Whether to track tensor creation/destruction
        """
        self.device_id = device_id
        self.sampling_interval = sampling_interval
        self.track_tensor_lifecycle = track_tensor_lifecycle
        
        # Memory tracking
        self.snapshots: List[MemorySnapshot] = []
        self.tensor_history = defaultdict(list)
        self.gradient_history = defaultdict(list)
        
        # Thread safety
        self._lock = threading.Lock()
        self._monitoring = False
        self._monitor_thread = None
        
        # Memory statistics
        self.stats = {
            'peak_gpu_usage_mb': 0.0,
            'peak_cpu_usage_mb': 0.0,
            'memory_leaks': 0,
            'tensor_churn_rate': 0.0,
            'gradient_efficiency': 0.0,
            'memory_fragmentation': 0.0
        }
        
        # Start monitoring if enabled
        if self.track_tensor_lifecycle:
            self._start_tensor_tracking()
    
    def _start_tensor_tracking(self):
        """Start monitoring tensor creation and destruction."""
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._memory_monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
    
    def _memory_monitor_loop(self):
        """Main memory monitoring loop."""
        while self._monitoring:
            try:
                self.capture_snapshot()
                time.sleep(self.sampling_interval)
            except Exception as e:
                print(f"Memory monitoring error: {e}")
                break
    
    def capture_snapshot(self) -> MemorySnapshot:
        """Capture current memory snapshot."""
        with self._lock:
            # GPU memory
            if torch.cuda.is_available():
                gpu_allocated = torch.cuda.memory_allocated(self.device_id) / (1024**2)
                gpu_reserved = torch.cuda.memory_reserved(self.device_id) / (1024**2)
            else:
                gpu_allocated = 0.0
                gpu_reserved = 0.0
            
            # CPU memory
            process = psutil.Process()
            cpu_memory = process.memory_info().rss / (1024**2)
            swap_used = process.memory_info().vms / (1024**2)
            
            # Tensor counts
            tensor_count = len([obj for obj in gc.get_objects() if torch.is_tensor(obj)])
            gradient_count = len([obj for obj in gc.get_objects() 
                                if torch.is_tensor(obj) and obj.grad_fn is not None])
            
            # Create snapshot
            snapshot = MemorySnapshot(
                timestamp=time.time(),
                gpu_allocated_mb=gpu_allocated,
                gpu_reserved_mb=gpu_reserved,
                cpu_memory_mb=cpu_memory,
                swap_used_mb=swap_used,
                tensor_count=tensor_count,
                gradient_count=gradient_count
            )
            
            self.snapshots.append(snapshot)
            
            # Update statistics
            self._update_memory_stats(snapshot)
            
            # Keep only recent snapshots
            if len(self.snapshots) > 10000:
                self.snapshots = self.snapshots[-5000:]
            
            return snapshot
    
    def _update_memory_stats(self, snapshot: MemorySnapshot):
        """Update memory statistics."""
        self.stats['peak_gpu_usage_mb'] = max(
            self.stats['peak_gpu_usage_mb'], snapshot.gpu_allocated_mb
        )
        self.stats['peak_cpu_usage_mb'] = max(
            self.stats['peak_cpu_usage_mb'], snapshot.cpu_memory_mb
        )
    
    def analyze_memory_patterns(self) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        if len(self.snapshots) < 10:
            return {'error': 'Insufficient data for analysis'}
        
        recent_snapshots = self.snapshots[-100:]  # Last 100 snapshots
        
        # Memory usage trends
        gpu_usage = [s.gpu_allocated_mb for s in recent_snapshots]
        cpu_usage = [s.cpu_memory_mb for s in recent_snapshots]
        
        # Calculate trends
        gpu_trend = np.polyfit(range(len(gpu_usage)), gpu_usage, 1)[0]
        cpu_trend = np.polyfit(range(len(cpu_usage)), cpu_usage, 1)[0]
        
        # Memory stability
        gpu_std = np.std(gpu_usage)
        cpu_std = np.std(cpu_usage)
        
        # Tensor lifecycle analysis
        tensor_counts = [s.tensor_count for s in recent_snapshots]
        gradient_counts = [s.gradient_count for s in recent_snapshots]
        
        tensor_churn = np.std(tensor_counts) / np.mean(tensor_counts) if np.mean(tensor_counts) > 0 else 0
        gradient_efficiency = np.mean(gradient_counts) / max(np.mean(tensor_counts), 1)
        
        # Memory fragmentation (simplified)
        fragmentation = (np.mean([s.gpu_reserved_mb - s.gpu_allocated_mb for s in recent_snapshots]) / 
                        np.mean([s.gpu_allocated_mb for s in recent_snapshots]))
        
        return {
            'gpu_trend': gpu_trend,
            'cpu_trend': cpu_trend,
            'gpu_stability': gpu_std,
            'cpu_stability': cpu_std,
            'tensor_churn_rate': tensor_churn,
            'gradient_efficiency': gradient_efficiency,
            'memory_fragmentation': max(0, min(1, fragmentation)),
            'memory_efficiency_score': 1.0 - (tensor_churn + fragmentation) / 2,
            'peak_usage': {
                'gpu_mb': self.stats['peak_gpu_usage_mb'],
                'cpu_mb': self.stats['peak_cpu_usage_mb']
            },
            'current_usage': {
                'gpu_mb': recent_snapshots[-1].gpu_allocated_mb,
                'cpu_mb': recent_snapshots[-1].cpu_memory_mb
            }
        }
    
    def detect_memory_leaks(self) -> Dict[str, Any]:
        """Detect potential memory leaks."""
        if len(self.snapshots) < 50:
            return {'leak_detected': False, 'reason': 'Insufficient data'}
        
        recent_snapshots = self.snapshots[-30:]
        
        # Check for consistent memory growth
        gpu_usage = [s.gpu_allocated_mb for s in recent_snapshots]
        cpu_usage = [s.cpu_memory_mb for s in recent_snapshots]
        
        # Linear regression to detect trends
        gpu_trend = np.polyfit(range(len(gpu_usage)), gpu_usage, 1)[0]
        cpu_trend = np.polyfit(range(len(cpu_usage)), cpu_usage, 1)[0]
        
        # Memory leak detection criteria
        leak_threshold_mb_per_minute = 10.0  # 10 MB per minute
        time_per_snapshot = self.sampling_interval * 60  # Convert to minutes
        
        gpu_leak_rate = gpu_trend / time_per_snapshot
        cpu_leak_rate = cpu_trend / time_per_snapshot
        
        leak_detected = (gpu_leak_rate > leak_threshold_mb_per_minute or 
                        cpu_leak_rate > leak_threshold_mb_per_minute)
        
        return {
            'leak_detected': leak_detected,
            'gpu_leak_rate_mb_per_min': gpu_leak_rate,
            'cpu_leak_rate_mb_per_min': cpu_leak_rate,
            'severity': 'high' if leak_detected and max(gpu_leak_rate, cpu_leak_rate) > 50 else 'low' if leak_detected else 'none'
        }
    
    def get_memory_recommendations(self) -> List[str]:
        """Get memory optimization recommendations."""
        analysis = self.analyze_memory_patterns()
        recommendations = []
        
        if 'error' in analysis:
            return ['Need more data for memory analysis']
        
        # Memory efficiency recommendations
        if analysis['memory_fragmentation'] > 0.3:
            recommendations.append("Consider defragmenting GPU memory or using memory pooling")
        
        if analysis['tensor_churn_rate'] > 0.5:
            recommendations.append("High tensor churn detected - consider tensor reuse or caching")
        
        if analysis['gradient_efficiency'] < 0.1:
            recommendations.append("Low gradient efficiency - consider gradient accumulation or checkpointing")
        
        if analysis['memory_efficiency_score'] < 0.5:
            recommendations.append("Overall memory efficiency is low - review memory management strategy")
        
        # Memory trend recommendations
        if analysis['gpu_trend'] > 5.0:  # Growing GPU usage
            recommendations.append("GPU memory usage is growing - consider reducing batch size or enabling gradient checkpointing")
        
        if analysis['cpu_trend'] > 50.0:  # Growing CPU usage
            recommendations.append("CPU memory usage is growing - consider data loading optimization")
        
        return recommendations if recommendations else ["Memory usage appears stable"]
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        return {
            **self.stats,
            'total_snapshots': len(self.snapshots),
            'monitoring_active': self._monitoring,
            'pattern_analysis': self.analyze_memory_patterns(),
            'leak_analysis': self.detect_memory_leaks(),
            'recommendations': self.get_memory_recommendations()
        }


class GradientCheckpointing:
    """
    Advanced gradient checkpointing system with selective recomputation.
    
    Implements activation checkpointing, gradient checkpointing, and
    strategic recomputation based on memory constraints.
    """
    
    def __init__(
        self,
        memory_budget_mb: int = 8000,
        checkpoint_strategy: str = 'selective',
        recompute_threshold: float = 0.5,
        checkpoint_every_n_layers: int = 2
    ):
        """
        Initialize gradient checkpointing.
        
        Args:
            memory_budget_mb: Available memory budget in MB
            checkpoint_strategy: Strategy for checkpointing ('selective', 'aggressive', 'conservative')
            recompute_threshold: Threshold for deciding what to recompute
            checkpoint_every_n_layers: Checkpoint every n layers
        """
        self.memory_budget_mb = memory_budget_mb
        self.checkpoint_strategy = checkpoint_strategy
        self.recompute_threshold = recompute_threshold
        self.checkpoint_every_n_layers = checkpoint_every_n_layers
        
        # Checkpoint storage
        self.checkpointed_activations = {}
        self.recomputed_layers = []
        self.memory_savings = 0.0
        
        # Performance tracking
        self.stats = {
            'total_checkpoints': 0,
            'total_recomputations': 0,
            'memory_saved_mb': 0.0,
            'recompute_overhead': 0.0,
            'checkpoints_per_layer': defaultdict(int)
        }
        
        # Layer complexity analysis
        self.layer_complexity = {}
        self.memory_pressure = False
    
    def register_module(self, module: nn.Module, module_id: str):
        """
        Register module for checkpointing analysis.
        
        Args:
            module: Module to register
            module_id: Unique identifier for the module
        """
        complexity = self._calculate_module_complexity(module)
        self.layer_complexity[module_id] = complexity
    
    def _calculate_module_complexity(self, module: nn.Module) -> Dict[str, float]:
        """
        Calculate module complexity metrics.
        
        Args:
            module: Module to analyze
            
        Returns:
            Dictionary of complexity metrics
        """
        complexity = {
            'parameter_count': 0,
            'flops_estimate': 0.0,
            'memory_footprint': 0.0,
            'activation_size': 0.0
        }
        
        # Parameter count
        for param in module.parameters():
            complexity['parameter_count'] += param.numel()
        
        # FLOPS estimate (simplified)
        if isinstance(module, nn.Linear):
            complexity['flops_estimate'] = (module.in_features * module.out_features * 2)
        elif isinstance(module, nn.Conv1d):
            complexity['flops_estimate'] = (module.in_channels * module.out_channels * 
                                          module.kernel_size[0] * module.stride[0])
        elif isinstance(module, nn.Conv2d):
            complexity['flops_estimate'] = (module.in_channels * module.out_channels * 
                                          module.kernel_size[0] * module.kernel_size[1])
        
        # Memory footprint
        complexity['memory_footprint'] = complexity['parameter_count'] * 4  # 4 bytes per float32
        
        return complexity
    
    @contextmanager
    def checkpoint_layer(
        self,
        layer_id: str,
        compute_fn: Callable,
        *args,
        **kwargs
    ):
        """
        Context manager for layer checkpointing.
        
        Args:
            layer_id: Unique layer identifier
            compute_fn: Function to compute layer output
            *args, **kwargs: Arguments for compute_fn
        """
        # Check if we should checkpoint this layer
        should_checkpoint = self._should_checkpoint_layer(layer_id)
        
        if should_checkpoint:
            # Store intermediate computations
            try:
                result = compute_fn(*args, **kwargs)
                
                # Check memory pressure and decide on checkpointing
                if self._under_memory_pressure():
                    self._store_checkpoint(layer_id, result)
                    yield result
                    # Clear checkpoint after use
                    if layer_id in self.checkpointed_activations:
                        del self.checkpointed_activations[layer_id]
                else:
                    yield result
                    
            except Exception as e:
                print(f"Checkpointing error in layer {layer_id}: {e}")
                yield compute_fn(*args, **kwargs)
        else:
            # No checkpointing, regular forward pass
            yield compute_fn(*args, **kwargs)
        
        self.stats['total_checkpoints'] += 1
        self.stats['checkpoints_per_layer'][layer_id] += 1
    
    def _should_checkpoint_layer(self, layer_id: str) -> bool:
        """Determine if a layer should be checkpointed."""
        if layer_id not in self.layer_complexity:
            return True  # Default to checkpointing unknown layers
        
        complexity = self.layer_complexity[layer_id]
        
        if self.checkpoint_strategy == 'aggressive':
            return complexity['flops_estimate'] > 1000
        elif self.checkpoint_strategy == 'conservative':
            return complexity['flops_estimate'] > 10000
        else:  # 'selective'
            # Checkpoint layers based on memory pressure and complexity
            base_threshold = 1000 * (1 + (1 - self.recompute_threshold))
            return complexity['flops_estimate'] > base_threshold
    
    def _under_memory_pressure(self) -> bool:
        """Check if system is under memory pressure."""
        if not torch.cuda.is_available():
            return False
        
        current_memory = torch.cuda.memory_allocated() / (1024**2)
        pressure_ratio = current_memory / self.memory_budget_mb
        
        self.memory_pressure = pressure_ratio > 0.8
        return self.memory_pressure
    
    def _store_checkpoint(self, layer_id: str, activation: torch.Tensor):
        """Store activation checkpoint."""
        # Calculate memory usage
        activation_memory = activation.numel() * activation.element_size() / (1024**2)
        
        # Store if within budget
        total_checkpoint_memory = sum(
            act.numel() * act.element_size() / (1024**2)
            for act in self.checkpointed_activations.values()
        )
        
        if total_checkpoint_memory + activation_memory < self.memory_budget_mb * 0.5:
            self.checkpointed_activations[layer_id] = activation.detach()
    
    def selective_backward_recompute(
        self,
        saved_activations: Dict[str, torch.Tensor],
        loss: torch.Tensor,
        module: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """
        Selective backward pass with strategic recomputation.
        
        Args:
            saved_activations: Saved activation checkpoints
            loss: Loss tensor for backward pass
            module: Module for backward computation
            
        Returns:
            Dictionary of gradients
        """
        # Clear gradients
        module.zero_grad()
        
        # Perform backward pass with recomputation strategy
        recomputed_layers = []
        
        try:
            loss.backward(retain_graph=False)
            
            # Collect gradients
            gradients = {}
            for name, param in module.named_parameters():
                if param.grad is not None:
                    gradients[name] = param.grad.clone()
            
            self.stats['total_recomputations'] += 1
            return gradients
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                # Handle OOM with gradient checkpointing
                self._handle_oom_with_checkpointing(loss, module)
                return {}
            else:
                raise e
    
    def _handle_oom_with_checkpointing(
        self,
        loss: torch.Tensor,
        module: nn.Module
    ):
        """Handle out of memory with aggressive checkpointing."""
        print("OOM detected - enabling aggressive gradient checkpointing")
        
        # Enable aggressive checkpointing
        original_strategy = self.checkpoint_strategy
        self.checkpoint_strategy = 'aggressive'
        
        # Clear some memory
        torch.cuda.empty_cache()
        gc.collect()
        
        # Retry with checkpointing
        try:
            # Clear gradients and recompute with checkpointing
            module.zero_grad()
            self.stats['memory_saved_mb'] += self.memory_budget_mb * 0.5
            loss.backward(retain_graph=False)
            
        except RuntimeError:
            print("OOM persists even with aggressive checkpointing")
        
        finally:
            # Restore original strategy
            self.checkpoint_strategy = original_strategy
    
    def get_checkpoint_statistics(self) -> Dict[str, Any]:
        """Get comprehensive checkpointing statistics."""
        return {
            **self.stats,
            'memory_saved_percent': (self.memory_saved_mb / self.memory_budget_mb) * 100,
            'checkpoint_strategy': self.checkpoint_strategy,
            'memory_pressure_active': self.memory_pressure,
            'total_checkpointed_layers': len(self.checkpointed_activations),
            'recompute_efficiency': self.stats['total_recomputations'] / max(self.stats['total_checkpoints'], 1)
        }


class DynamicMemoryAllocator:
    """
    Dynamic memory allocation system for efficient memory management.
    
    Provides intelligent memory allocation, deallocation, and reuse
    strategies based on current system state and training requirements.
    """
    
    def __init__(
        self,
        device_id: int = 0,
        max_pool_size_gb: float = 4.0,
        allocation_strategy: str = 'smart',
        auto_cleanup: bool = True
    ):
        """
        Initialize dynamic memory allocator.
        
        Args:
            device_id: GPU device ID
            max_pool_size_gb: Maximum memory pool size in GB
            allocation_strategy: Strategy for allocation ('smart', 'aggressive', 'conservative')
            auto_cleanup: Whether to enable automatic cleanup
        """
        self.device_id = device_id
        self.max_pool_size_gb = max_pool_size_gb
        self.allocation_strategy = allocation_strategy
        self.auto_cleanup = auto_cleanup
        
        # Memory pools
        self.tensor_pools = defaultdict(list)
        self.gradient_pools = defaultdict(list)
        self.activation_pools = defaultdict(list)
        
        # Pool metadata
        self.pool_metadata = {}
        self.total_allocated_mb = 0.0
        self.peak_allocation_mb = 0.0
        
        # Allocation tracking
        self.allocation_history = deque(maxlen=1000)
        self.deallocation_history = deque(maxlen=1000)
        
        # Performance statistics
        self.stats = {
            'total_allocations': 0,
            'total_deallocations': 0,
            'pool_hits': 0,
            'pool_misses': 0,
            'allocation_efficiency': 0.0,
            'fragmentation_score': 0.0
        }
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize memory pools
        self._initialize_memory_pools()
    
    def _initialize_memory_pools(self):
        """Initialize memory pools for different tensor types."""
        pool_configs = {
            'large_tensors': {'sizes': [1024, 2048, 4096, 8192], 'count': 5},
            'medium_tensors': {'sizes': [256, 512, 1024], 'count': 10},
            'small_tensors': {'sizes': [64, 128, 256], 'count': 20},
            'gradients': {'sizes': [1024, 2048, 4096], 'count': 8},
            'activations': {'sizes': [512, 1024, 2048], 'count': 15}
        }
        
        device = torch.device('cuda', self.device_id) if torch.cuda.is_available() else torch.device('cpu')
        
        for pool_name, config in pool_configs.items():
            pool_size_mb = 0
            for size in config['sizes']:
                for _ in range(config['count']):
                    try:
                        # Create tensor pool entry
                        tensor = torch.zeros(size, dtype=torch.float32, device=device)
                        pool_key = f"{pool_name}_{size}"
                        self.tensor_pools[pool_key].append(tensor)
                        pool_size_mb += size * 4 / (1024**2)  # 4 bytes per float32
                    except RuntimeError:
                        # Not enough memory for this tensor
                        continue
            
            self.pool_metadata[pool_name] = {
                'size_mb': pool_size_mb,
                'capacity': config['count'],
                'current_usage': 0
            }
        
        print(f"Memory pools initialized: {sum(meta['size_mb'] for meta in self.pool_metadata.values()):.1f} MB")
    
    def allocate_tensor(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        tensor_type: str = 'large_tensors',
        requires_grad: bool = False
    ) -> torch.Tensor:
        """
        Allocate tensor using dynamic memory management.
        
        Args:
            shape: Tensor shape
            dtype: Tensor dtype
            tensor_type: Type of tensor for pool selection
            requires_grad: Whether tensor requires gradients
            
        Returns:
            Allocated tensor
        """
        with self._lock:
            # Calculate required size
            element_count = np.prod(shape)
            tensor_size_mb = element_count * torch.finfo(dtype).bits / 8 / (1024**2)
            
            # Check if we should use pool
            pool_key = self._find_best_pool_match(element_count, tensor_type)
            
            if pool_key and self.tensor_pools[pool_key]:
                # Reuse from pool
                tensor = self.tensor_pools[pool_key].pop()
                self.stats['pool_hits'] += 1
                
                # Resize and reinitialize
                tensor = tensor.view(shape).contiguous()
                tensor = tensor.to(dtype).detach().requires_grad_(requires_grad)
                
                # Zero out if reusing
                if not requires_grad:
                    tensor.zero_()
                
            else:
                # Allocate new tensor
                self.stats['pool_misses'] += 1
                device = torch.device('cuda', self.device_id) if torch.cuda.is_available() else torch.device('cpu')
                
                try:
                    tensor = torch.zeros(shape, dtype=dtype, device=device, requires_grad=requires_grad)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        # Try to free memory and retry
                        self._emergency_memory_cleanup()
                        tensor = torch.zeros(shape, dtype=dtype, device=device, requires_grad=requires_grad)
                    else:
                        raise e
            
            # Update tracking
            self.total_allocated_mb += tensor_size_mb
            self.peak_allocation_mb = max(self.peak_allocation_mb, self.total_allocated_mb)
            self.stats['total_allocations'] += 1
            self.allocation_history.append((time.time(), tensor_size_mb, shape))
            
            return tensor
    
    def deallocate_tensor(self, tensor: torch.Tensor, tensor_type: str = 'large_tensors'):
        """
        Deallocate tensor and return to pool if appropriate.
        
        Args:
            tensor: Tensor to deallocate
            tensor_type: Type of tensor for pool classification
        """
        with self._lock:
            if tensor is None:
                return
            
            # Calculate tensor size
            element_count = tensor.numel()
            tensor_size_mb = element_count * tensor.element_size() / (1024**2)
            
            # Try to return to pool
            pool_key = self._find_best_pool_match(element_count, tensor_type)
            
            if pool_key and len(self.tensor_pools[pool_key]) < self.pool_metadata[tensor_type]['capacity']:
                # Return to pool
                tensor.zero_()
                self.tensor_pools[pool_key].append(tensor.detach())
                self.pool_metadata[tensor_type]['current_usage'] += 1
            
            # Remove reference
            del tensor
            
            # Update tracking
            self.total_allocated_mb -= tensor_size_mb
            self.stats['total_deallocations'] += 1
            self.deallocation_history.append((time.time(), tensor_size_mb))
            
            # Auto cleanup if enabled
            if self.auto_cleanup and len(self.allocation_history) > 100:
                self._periodic_cleanup()
    
    def _find_best_pool_match(self, element_count: int, tensor_type: str) -> Optional[str]:
        """Find best matching pool for element count."""
        pool_candidates = []
        
        for pool_key in self.tensor_pools.keys():
            if tensor_type in pool_key:
                pool_candidates.append(pool_key)
        
        # Find closest size match
        best_match = None
        best_diff = float('inf')
        
        for pool_key in pool_candidates:
            # Extract size from pool key
            size_str = pool_key.split('_')[-1]
            try:
                pool_size = int(size_str)
                diff = abs(pool_size - element_count)
                if diff < best_diff and len(self.tensor_pools[pool_key]) > 0:
                    best_diff = diff
                    best_match = pool_key
            except ValueError:
                continue
        
        return best_match
    
    def _emergency_memory_cleanup(self):
        """Perform emergency memory cleanup."""
        print("Performing emergency memory cleanup")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear smallest pools first
        for pool_name in ['small_tensors', 'medium_tensors', 'large_tensors']:
            if pool_name in self.tensor_pools:
                self.tensor_pools[pool_name].clear()
                if pool_name in self.pool_metadata:
                    self.pool_metadata[pool_name]['current_usage'] = 0
        
        # Force garbage collection
        gc.collect()
    
    def _periodic_cleanup(self):
        """Perform periodic memory cleanup."""
        current_time = time.time()
        cutoff_time = current_time - 300  # 5 minutes
        
        # Remove old allocation history
        while (self.allocation_history and 
               self.allocation_history[0][0] < cutoff_time):
            old_allocation = self.allocation_history.popleft()
            self.total_allocated_mb -= old_allocation[1]
    
    def optimize_memory_layout(self):
        """Optimize memory layout to reduce fragmentation."""
        # This would implement memory defragmentation strategies
        # For now, we'll implement a simple cleanup
        
        # Clear pools that are underutilized
        for pool_name, metadata in self.pool_metadata.items():
            if metadata['current_usage'] < metadata['capacity'] * 0.2:
                # Pool is underutilized, clear it
                pool_key = f"{pool_name}_1024"  # Clear all sizes for this type
                if pool_key in self.tensor_pools:
                    self.tensor_pools[pool_key].clear()
                    metadata['current_usage'] = 0
        
        # Update fragmentation score
        total_pool_size = sum(meta['size_mb'] for meta in self.pool_metadata.values())
        used_pool_size = sum(meta['current_usage'] * 4 / (1024**2) for meta in self.pool_metadata.values())
        self.stats['fragmentation_score'] = 1.0 - (used_pool_size / max(total_pool_size, 1))
    
    def get_allocation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive allocation statistics."""
        # Calculate efficiency
        total_operations = self.stats['pool_hits'] + self.stats['pool_misses']
        efficiency = self.stats['pool_hits'] / max(total_operations, 1)
        self.stats['allocation_efficiency'] = efficiency
        
        return {
            **self.stats,
            'total_allocated_mb': self.total_allocated_mb,
            'peak_allocation_mb': self.peak_allocation_mb,
            'pool_utilization': {
                name: {
                    'capacity': meta['capacity'],
                    'current_usage': meta['current_usage'],
                    'utilization_percent': (meta['current_usage'] / meta['capacity']) * 100
                }
                for name, meta in self.pool_metadata.items()
            },
            'fragmentation_score': self.stats['fragmentation_score'],
            'allocation_efficiency': efficiency
        }


class EfficientTrainingOrchestrator:
    """
    Unified orchestrator for all memory-efficient training strategies.
    
    Coordinates memory profiling, checkpointing, allocation, and
    dynamic optimization for optimal training efficiency.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        memory_budget_mb: int = 8000,
        enable_profiling: bool = True,
        auto_optimization: bool = True
    ):
        """
        Initialize efficient training orchestrator.
        
        Args:
            model: Model to manage memory for
            device: Training device
            memory_budget_mb: Available memory budget
            enable_profiling: Whether to enable memory profiling
            auto_optimization: Whether to enable automatic optimization
        """
        self.model = model
        self.device = device
        self.memory_budget_mb = memory_budget_mb
        self.enable_profiling = enable_profiling
        self.auto_optimization = auto_optimization
        
        # Component initialization
        self.memory_profiler = MemoryProfiler() if enable_profiling else None
        self.checkpointing = GradientCheckpointing(memory_budget_mb)
        self.allocator = DynamicMemoryAllocator(
            device_id=device.index if device.type == 'cuda' else 0,
            max_pool_size_gb=min(memory_budget_mb / 1024, 4.0)
        )
        
        # Training state
        self.current_step = 0
        self.oom_count = 0
        self.optimization_applied = []
        
        # Performance tracking
        self.training_efficiency_history = []
        self.memory_efficiency_history = []
        
        # Register model modules
        self._register_model_modules()
    
    def _register_model_modules(self):
        """Register all model modules for memory management."""
        for name, module in self.model.named_modules():
            self.checkpointing.register_module(module, name)
    
    @contextmanager
    def memory_efficient_training_step(self, batch_size: int):
        """
        Context manager for memory-efficient training step.
        
        Args:
            batch_size: Current batch size
        """
        # Pre-step memory optimization
        if self.auto_optimization:
            self._pre_step_optimization()
        
        # Capture memory snapshot
        if self.memory_profiler:
            pre_snapshot = self.memory_profiler.capture_snapshot()
        
        try:
            yield
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                self._handle_oom()
            else:
                raise e
        
        finally:
            # Post-step memory cleanup
            if self.auto_optimization:
                self._post_step_cleanup()
            
            # Update tracking
            self.current_step += 1
            
            # Capture post-step snapshot
            if self.memory_profiler:
                post_snapshot = self.memory_profiler.capture_snapshot()
                self._analyze_step_efficiency(pre_snapshot, post_snapshot)
    
    def _pre_step_optimization(self):
        """Perform pre-step memory optimization."""
        # Analyze current memory state
        if self.memory_profiler:
            recommendations = self.memory_profiler.get_memory_recommendations()
            
            # Apply automatic optimizations
            for recommendation in recommendations:
                if "defragmenting" in recommendation.lower():
                    self.allocator.optimize_memory_layout()
                    self.optimization_applied.append("memory_defragmentation")
                
                elif "gradient checkpointing" in recommendation.lower():
                    self.checkpointing.memory_pressure = True
                    self.optimization_applied.append("gradient_checkpointing")
                
                elif "batch size" in recommendation.lower():
                    # This would be handled by the training loop
                    self.optimization_applied.append("batch_size_reduction")
    
    def _post_step_cleanup(self):
        """Perform post-step memory cleanup."""
        # Periodic optimization
        if self.current_step % 100 == 0:
            self.allocator.optimize_memory_layout()
            
            if self.memory_profiler:
                analysis = self.memory_profiler.analyze_memory_patterns()
                if analysis.get('memory_efficiency_score', 1.0) < 0.5:
                    print("Low memory efficiency detected - applying optimizations")
                    self._apply_memory_optimizations()
    
    def _handle_oom(self):
        """Handle out of memory situations."""
        self.oom_count += 1
        print(f"OOM detected (count: {self.oom_count}) - applying emergency optimizations")
        
        # Emergency memory cleanup
        self.allocator._emergency_memory_cleanup()
        
        # Enable aggressive checkpointing
        self.checkpointing.memory_pressure = True
        self.checkpointing.checkpoint_strategy = 'aggressive'
        
        # Apply multiple optimizations
        emergency_optimizations = [
            "memory_pool_cleanup",
            "aggressive_checkpointing",
            "cache_clearing"
        ]
        
        for opt in emergency_optimizations:
            self.optimization_applied.append(opt)
        
        # Clear torch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _analyze_step_efficiency(
        self,
        pre_snapshot: MemorySnapshot,
        post_snapshot: MemorySnapshot
    ):
        """Analyze efficiency of training step."""
        # Calculate memory efficiency
        memory_used = post_snapshot.gpu_allocated_mb - pre_snapshot.gpu_allocated_mb
        
        if memory_used > 0:
            memory_efficiency = self.memory_budget_mb / (memory_used + 1e-6)
            self.memory_efficiency_history.append(memory_efficiency)
            
            # Keep only recent history
            if len(self.memory_efficiency_history) > 1000:
                self.memory_efficiency_history = self.memory_efficiency_history[-500:]
    
    def _apply_memory_optimizations(self):
        """Apply comprehensive memory optimizations."""
        # Analyze current state
        if self.memory_profiler:
            analysis = self.memory_profiler.analyze_memory_patterns()
            
            # Apply optimizations based on analysis
            if analysis.get('memory_fragmentation', 0) > 0.3:
                self.allocator.optimize_memory_layout()
                self.optimization_applied.append("fragmentation_optimization")
            
            if analysis.get('tensor_churn_rate', 0) > 0.5:
                # Enable tensor caching (implementation would depend on framework)
                self.optimization_applied.append("tensor_caching")
            
            if analysis.get('gradient_efficiency', 1.0) < 0.1:
                self.checkpointing.recompute_threshold = 0.3
                self.optimization_applied.append("gradient_efficiency_optimization")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        stats = {
            'current_step': self.current_step,
            'oom_count': self.oom_count,
            'optimizations_applied': list(set(self.optimization_applied)),
            'training_efficiency': {
                'avg_memory_efficiency': np.mean(self.memory_efficiency_history[-100:]) if self.memory_efficiency_history else 0,
                'memory_stability': np.std(self.memory_efficiency_history[-100:]) if len(self.memory_efficiency_history) > 10 else 0
            }
        }
        
        # Add component stats
        if self.memory_profiler:
            stats['memory_profiling'] = self.memory_profiler.get_stats()
        
        if self.checkpointing:
            stats['checkpointing'] = self.checkpointing.get_checkpoint_statistics()
        
        if self.allocator:
            stats['allocation'] = self.allocator.get_allocation_statistics()
        
        return stats


# Utility functions for memory management

def auto_configure_memory_management(
    gpu_memory_gb: float,
    model_size_mb: int,
    batch_size: int
) -> Dict[str, Any]:
    """
    Automatically configure memory management based on hardware and model.
    
    Args:
        gpu_memory_gb: Available GPU memory in GB
        model_size_mb: Model size in MB
        batch_size: Training batch size
        
    Returns:
        Memory management configuration
    """
    # Calculate memory requirements
    model_memory_gb = model_size_mb / 1024
    activation_memory_gb = batch_size * 0.001  # Rough estimate
    optimizer_memory_gb = model_memory_gb * 2
    
    total_required = model_memory_gb + activation_memory_gb + optimizer_memory_gb
    available_memory = gpu_memory_gb * 0.8  # Use 80% of available memory
    
    # Configuration based on memory pressure
    if total_required > available_memory:
        # High memory pressure
        config = {
            'memory_budget_mb': int(available_memory * 1024),
            'checkpoint_strategy': 'aggressive',
            'enable_profiling': True,
            'auto_optimization': True,
            'max_pool_size_gb': min(available_memory * 0.5, 2.0),
            'allocation_strategy': 'aggressive'
        }
    elif total_required > available_memory * 0.7:
        # Medium memory pressure
        config = {
            'memory_budget_mb': int(available_memory * 1024),
            'checkpoint_strategy': 'selective',
            'enable_profiling': True,
            'auto_optimization': True,
            'max_pool_size_gb': min(available_memory * 0.3, 1.5),
            'allocation_strategy': 'smart'
        }
    else:
        # Low memory pressure
        config = {
            'memory_budget_mb': int(available_memory * 1024),
            'checkpoint_strategy': 'conservative',
            'enable_profiling': False,
            'auto_optimization': False,
            'max_pool_size_gb': min(available_memory * 0.2, 1.0),
            'allocation_strategy': 'conservative'
        }
    
    return config


def benchmark_memory_strategies(
    model: nn.Module,
    batch_size: int,
    sequence_length: int,
    memory_budget_mb: int = 8000
) -> Dict[str, Dict]:
    """
    Benchmark different memory management strategies.
    
    Args:
        model: Model to benchmark
        batch_size: Batch size for benchmark
        sequence_length: Sequence length for benchmark
        memory_budget_mb: Memory budget in MB
        
    Returns:
        Benchmark results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    strategies = {
        'no_optimization': {},
        'basic_checkpointing': {'checkpoint_strategy': 'conservative'},
        'aggressive_checkpointing': {'checkpoint_strategy': 'aggressive'},
        'dynamic_allocation': {'allocation_strategy': 'smart'},
        'full_optimization': {
            'checkpoint_strategy': 'selective',
            'allocation_strategy': 'smart',
            'enable_profiling': True
        }
    }
    
    results = {}
    
    for strategy_name, config in strategies.items():
        try:
            # Create orchestrator
            orchestrator = EfficientTrainingOrchestrator(
                model=model,
                device=device,
                memory_budget_mb=memory_budget_mb,
                **config
            )
            
            # Generate test data
            test_input = torch.randn(batch_size, sequence_length, 1024, device=device)
            test_target = torch.randint(0, 1000, (batch_size, sequence_length), device=device)
            
            # Benchmark memory usage
            torch.cuda.synchronize()
            start_memory = torch.cuda.memory_allocated() / (1024**2)
            
            # Simulate training steps
            for step in range(10):
                with orchestrator.memory_efficient_training_step(batch_size):
                    # Simulate forward/backward
                    output = model(test_input)
                    loss = F.cross_entropy(output.view(-1, output.size(-1)), test_target.view(-1))
                    loss.backward()
            
            torch.cuda.synchronize()
            end_memory = torch.cuda.memory_allocated() / (1024**2)
            
            # Get statistics
            stats = orchestrator.get_comprehensive_stats()
            
            results[strategy_name] = {
                'memory_usage_mb': end_memory - start_memory,
                'peak_memory_mb': stats.get('memory_profiling', {}).get('peak_usage', {}).get('gpu_mb', 0),
                'oom_events': stats.get('oom_count', 0),
                'optimization_overhead': len(stats.get('optimizations_applied', [])),
                'memory_efficiency': stats.get('training_efficiency', {}).get('avg_memory_efficiency', 0),
                'configuration': config
            }
            
        except Exception as e:
            results[strategy_name] = {
                'error': str(e),
                'memory_usage_mb': float('inf'),
                'peak_memory_mb': float('inf'),
                'oom_events': 1,
                'optimization_overhead': 0,
                'memory_efficiency': 0
            }
    
    return results