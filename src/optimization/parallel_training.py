"""
Parallel Training Framework

This module implements comprehensive parallel training strategies including
tensor parallelism, pipeline parallelism, and expert parallelism (MoE) for
efficient distributed training of large language models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import numpy as np
import warnings
from typing import List, Dict, Optional, Tuple, Any, Union
from collections import defaultdict
import threading
import time
from contextlib import contextmanager


class TensorParallel:
    """
    Tensor parallelism implementation that splits model parameters across devices.
    
    Supports both row-wise and column-wise parameter splitting with
    efficient communication strategies.
    """
    
    def __init__(
        self,
        model: nn.Module,
        devices: List[torch.device],
        strategy: str = 'row_wise',
        gradient_sync: bool = True
    ):
        """
        Initialize tensor parallel.
        
        Args:
            model: Model to parallelize
            devices: List of devices for parallelism
            strategy: Splitting strategy ('row_wise', 'column_wise', 'hybrid')
            gradient_sync: Whether to synchronize gradients across devices
        """
        self.model = model
        self.devices = devices
        self.strategy = strategy
        self.gradient_sync = gradient_sync
        
        # Parallel state
        self.parallelized_params = {}
        self.gradient_buffers = {}
        self.rank = 0
        self.world_size = len(devices)
        
        # Statistics
        self.stats = {
            'total_splits': 0,
            'sync_operations': 0,
            'communication_overhead': 0.0,
            'parallel_efficiency': 0.0
        }
        
        # Initialize parallelization
        self._initialize_tensor_parallelism()
    
    def _initialize_tensor_parallelism(self):
        """Initialize tensor parallelism across devices."""
        self.rank = torch.cuda.current_device() if torch.cuda.is_available() else 0
        
        # Split model parameters
        for name, param in self.model.named_parameters():
            if self._should_parallelize_param(param):
                self._split_parameter(name, param)
    
    def _should_parallelize_param(self, param: torch.Tensor) -> bool:
        """
        Determine if a parameter should be parallelized.
        
        Args:
            param: Parameter tensor
            
        Returns:
            Whether to parallelize this parameter
        """
        # Only parallelize large parameters (>= 1M elements)
        return param.numel() >= 10**6
    
    def _split_parameter(self, name: str, param: torch.Tensor):
        """
        Split parameter across devices.
        
        Args:
            name: Parameter name
            param: Parameter tensor
        """
        param_size = param.numel()
        split_size = param_size // self.world_size
        splits = []
        
        if self.strategy == 'row_wise' and param.dim() >= 2:
            # Row-wise splitting (split first dimension)
            dim = 0
            split_size = param.size(dim) // self.world_size
            for i in range(self.world_size):
                start_idx = i * split_size
                end_idx = start_idx + split_size if i < self.world_size - 1 else param.size(dim)
                split = param.narrow(dim, start_idx, end_idx - start_idx)
                splits.append(split)
        
        elif self.strategy == 'column_wise' and param.dim() >= 2:
            # Column-wise splitting (split last dimension)
            dim = -1
            split_size = param.size(dim) // self.world_size
            for i in range(self.world_size):
                start_idx = i * split_size
                end_idx = start_idx + split_size if i < self.world_size - 1 else param.size(dim)
                split = param.narrow(dim, start_idx, end_idx - start_idx)
                splits.append(split)
        
        else:
            # Element-wise splitting (split flat tensor)
            split_size = param_size // self.world_size
            for i in range(self.world_size):
                start_idx = i * split_size
                end_idx = start_idx + split_size if i < self.world_size - 1 else param_size
                split = param.view(-1)[start_idx:end_idx].clone()
                splits.append(split)
        
        # Store splits
        self.parallelized_params[name] = splits
        
        # Create gradient buffer for this parameter
        self.gradient_buffers[name] = [
            torch.zeros_like(splits[i]) for i in range(self.world_size)
        ]
        
        self.stats['total_splits'] += 1
    
    def forward(self, *inputs, **kwargs) -> torch.Tensor:
        """
        Parallel forward pass.
        
        Args:
            *inputs: Input tensors
            **kwargs: Input keyword arguments
            
        Returns:
            Parallel output tensor
        """
        # Perform forward pass with tensor parallel
        if self.strategy == 'row_wise':
            return self._row_wise_forward(*inputs, **kwargs)
        elif self.strategy == 'column_wise':
            return self._column_wise_forward(*inputs, **kwargs)
        else:
            return self._hybrid_forward(*inputs, **kwargs)
    
    def _row_wise_forward(self, *inputs, **kwargs) -> torch.Tensor:
        """Row-wise tensor parallel forward pass."""
        # This is a simplified implementation
        # In reality, you would need to implement complex tensor parallel operations
        
        start_time = time.time()
        
        # Split inputs across devices (simplified)
        batch_size = inputs[0].size(0)
        split_size = batch_size // self.world_size
        
        outputs = []
        for i in range(self.world_size):
            # Simulate parallel computation
            device = self.devices[i]
            
            # Split inputs for this device
            device_inputs = []
            for inp in inputs:
                start_idx = i * split_size
                end_idx = start_idx + split_size if i < self.world_size - 1 else batch_size
                device_inp = inp[start_idx:end_idx].to(device)
                device_inputs.append(device_inp)
            
            # Forward pass on device (simplified)
            with torch.cuda.device(device):
                output = self._device_forward(device_inputs, kwargs)
                outputs.append(output)
        
        # Combine outputs
        output = torch.cat(outputs, dim=0)
        
        # Record communication overhead
        comm_time = time.time() - start_time
        self.stats['communication_overhead'] += comm_time
        self.stats['sync_operations'] += 1
        
        return output
    
    def _column_wise_forward(self, *inputs, **kwargs) -> torch.Tensor:
        """Column-wise tensor parallel forward pass."""
        # Similar to row-wise but with column splitting
        return self._row_wise_forward(*inputs, **kwargs)
    
    def _hybrid_forward(self, *inputs, **kwargs) -> torch.Tensor:
        """Hybrid tensor parallel forward pass."""
        # Use combination of row and column splitting
        return self._row_wise_forward(*inputs, **kwargs)
    
    def _device_forward(self, inputs: List[torch.Tensor], kwargs: Dict) -> torch.Tensor:
        """Forward pass on a single device."""
        # Simplified device computation
        # In reality, this would perform actual model computation
        
        x = inputs[0]
        for module in self.model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                x = module(x)
            elif isinstance(module, nn.ReLU):
                x = F.relu(x)
            elif isinstance(module, nn.LayerNorm):
                x = module(x)
        
        return x
    
    def backward(self, loss: torch.Tensor):
        """
        Parallel backward pass.
        
        Args:
            loss: Loss tensor
        """
        if not self.gradient_sync:
            return
        
        start_time = time.time()
        
        # Backward pass on each device
        gradients = []
        for i, device in enumerate(self.devices):
            with torch.cuda.device(device):
                # Perform backward pass
                device_loss = loss / self.world_size  # Distribute loss
                device_loss.backward(retain_graph=True)
                
                # Collect gradients for parallel parameters
                device_grads = {}
                for name in self.parallelized_params.keys():
                    param = list(self.model.parameters())[i]  # Simplified
                    if param.grad is not None:
                        device_grads[name] = param.grad.clone()
                
                gradients.append(device_grads)
        
        # Synchronize gradients across devices
        self._synchronize_gradients(gradients)
        
        # Record communication overhead
        comm_time = time.time() - start_time
        self.stats['communication_overhead'] += comm_time
        self.stats['sync_operations'] += 1
    
    def _synchronize_gradients(self, gradients: List[Dict[str, torch.Tensor]]):
        """Synchronize gradients across devices."""
        for name, param_splits in self.parallelized_params.items():
            # Average gradients across devices
            avg_grad = torch.zeros_like(param_splits[0])
            for device_grads in gradients:
                if name in device_grads:
                    avg_grad += device_grads[name]
            avg_grad /= len(gradients)
            
            # Update all parameter splits with averaged gradient
            for i, param_split in enumerate(param_splits):
                param_split.grad = avg_grad.clone()
    
    def get_parallel_statistics(self) -> Dict[str, Any]:
        """Get tensor parallel statistics."""
        return {
            **self.stats,
            'world_size': self.world_size,
            'rank': self.rank,
            'parallelized_params': len(self.parallelized_params),
            'strategy': self.strategy,
            'gradient_sync': self.gradient_sync
        }


class PipelineParallel:
    """
    Pipeline parallelism that splits model across layers and pipelines
    the computation across multiple devices.
    """
    
    def __init__(
        self,
        model: nn.Module,
        devices: List[torch.device],
        pipeline_stages: Optional[List[int]] = None,
        micro_batch_size: int = 4,
        accumulation_steps: int = 8
    ):
        """
        Initialize pipeline parallel.
        
        Args:
            model: Model to parallelize
            devices: List of devices for pipeline stages
            pipeline_stages: Layer indices to split between devices
            micro_batch_size: Size of micro-batches
            accumulation_steps: Steps for gradient accumulation
        """
        self.model = model
        self.devices = devices
        self.pipeline_stages = pipeline_stages
        self.micro_batch_size = micro_batch_size
        self.accumulation_steps = accumulation_steps
        
        # Pipeline state
        self.stage_models = {}
        self.forward_buffer = {}
        self.backward_buffer = {}
        self.rank = 0
        self.world_size = len(devices)
        
        # Statistics
        self.stats = {
            'total_forward_passes': 0,
            'total_backward_passes': 0,
            'pipeline_bubbles': 0.0,
            'pipeline_efficiency': 0.0,
            'stage_load_imbalance': 0.0
        }
        
        # Initialize pipeline
        self._initialize_pipeline_parallelism()
    
    def _initialize_pipeline_parallelism(self):
        """Initialize pipeline parallelism across devices."""
        self.rank = torch.cuda.current_device() if torch.cuda.is_available() else 0
        
        # Determine pipeline stages
        if self.pipeline_stages is None:
            self.pipeline_stages = self._auto_partition_layers()
        
        # Split model into stages
        self._partition_model_stages()
    
    def _auto_partition_layers(self) -> List[int]:
        """
        Automatically determine layer partitioning for pipeline parallelism.
        
        Returns:
            List of layer indices to split between devices
        """
        # Get all layers that can be parallelized
        parallelizable_layers = []
        layer_count = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)) and layer_count > 0:
                parallelizable_layers.append(layer_count)
            layer_count += 1
        
        # Partition evenly across devices
        layers_per_stage = len(parallelizable_layers) // self.world_size
        pipeline_stages = []
        
        for i in range(1, self.world_size):
            stage_idx = min(i * layers_per_stage, len(parallelizable_layers) - 1)
            pipeline_stages.append(parallelizable_layers[stage_idx])
        
        return pipeline_stages
    
    def _partition_model_stages(self):
        """Partition model into pipeline stages."""
        # Get all parameters
        all_params = list(self.model.parameters())
        
        # Split parameters across stages
        params_per_stage = len(all_params) // self.world_size
        
        for i, device in enumerate(self.devices):
            start_idx = i * params_per_stage
            end_idx = start_idx + params_per_stage if i < self.world_size - 1 else len(all_params)
            
            # Create stage-specific model (simplified)
            stage_params = all_params[start_idx:end_idx]
            self.stage_models[i] = {
                'device': device,
                'parameters': stage_params,
                'compute_time': 0.0
            }
    
    def pipeline_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Perform pipeline parallel forward pass.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Final output tensor
        """
        batch_size = inputs.size(0)
        micro_batches = batch_size // self.micro_batch_size
        
        # Create micro-batches
        micro_batch_outputs = []
        
        for mb_idx in range(micro_batches):
            start_idx = mb_idx * self.micro_batch_size
            end_idx = start_idx + self.micro_batch_size
            micro_batch = inputs[start_idx:end_idx]
            
            # Pipeline execution
            output = self._execute_pipeline_forward(micro_batch)
            micro_batch_outputs.append(output)
        
        # Combine micro-batch outputs
        output = torch.cat(micro_batch_outputs, dim=0)
        self.stats['total_forward_passes'] += 1
        
        return output
    
    def _execute_pipeline_forward(self, micro_batch: torch.Tensor) -> torch.Tensor:
        """Execute forward pass through pipeline."""
        x = micro_batch
        
        # Forward through each stage
        for stage_idx in range(self.world_size):
            stage = self.stage_models[stage_idx]
            device = stage['device']
            
            with torch.cuda.device(device):
                # Move to stage device
                x = x.to(device)
                
                # Forward computation (simplified)
                start_time = time.time()
                
                # This would perform actual stage computation
                for param in stage['parameters']:
                    x = F.linear(x, param)
                    x = F.relu(x)
                
                stage['compute_time'] = time.time() - start_time
        
        return x
    
    def pipeline_backward(self, loss: torch.Tensor):
        """
        Perform pipeline parallel backward pass.
        
        Args:
            loss: Loss tensor
        """
        # Backward pass through pipeline in reverse order
        for stage_idx in range(self.world_size - 1, -1, -1):
            stage = self.stage_models[stage_idx]
            device = stage['device']
            
            with torch.cuda.device(device):
                # Backward computation (simplified)
                for param in stage['parameters']:
                    if param.grad is not None:
                        # Simulate gradient computation
                        pass
        
        self.stats['total_backward_passes'] += 1
    
    def _calculate_pipeline_efficiency(self) -> float:
        """Calculate pipeline efficiency."""
        compute_times = [stage['compute_time'] for stage in self.stage_models.values()]
        
        if not compute_times:
            return 0.0
        
        # Calculate load imbalance
        mean_time = np.mean(compute_times)
        std_time = np.std(compute_times)
        self.stats['stage_load_imbalance'] = std_time / mean_time if mean_time > 0 else 0.0
        
        # Pipeline efficiency based on load balance
        efficiency = 1.0 - min(self.stats['stage_load_imbalance'], 1.0)
        self.stats['pipeline_efficiency'] = efficiency
        
        return efficiency
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get pipeline parallel statistics."""
        self._calculate_pipeline_efficiency()
        
        return {
            **self.stats,
            'world_size': self.world_size,
            'rank': self.rank,
            'micro_batch_size': self.micro_batch_size,
            'pipeline_stages': len(self.pipeline_stages),
            'stage_compute_times': [stage['compute_time'] for stage in self.stage_models.values()]
        }


class ExpertParallel:
    """
    Expert Parallelism for Mixture of Experts (MoE) models.
    
    Implements efficient expert routing and parallel computation
    for large-scale MoE models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        devices: List[torch.device],
        num_experts: int = 8,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        load_balancing: bool = True
    ):
        """
        Initialize expert parallel.
        
        Args:
            model: Model containing MoE layers
            devices: List of devices for expert computation
            num_experts: Total number of experts
            top_k: Number of experts to activate per token
            capacity_factor: Factor to increase expert capacity
            load_balancing: Whether to enable load balancing
        """
        self.model = model
        self.devices = devices
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.load_balancing = load_balancing
        
        # Expert state
        self.experts = {}
        self.routing_weights = {}
        self.expert_indices = {}
        self.load_balancing_loss = 0.0
        
        # Statistics
        self.stats = {
            'total_expert_calls': 0,
            'expert_utilization': defaultdict(float),
            'routing_efficiency': 0.0,
            'load_balance_loss': 0.0,
            'expert_overflow_count': 0
        }
        
        # Initialize experts
        self._initialize_experts()
    
    def _initialize_experts(self):
        """Initialize experts across devices."""
        # Find MoE layers in model
        moe_layers = self._find_moe_layers()
        
        # Create experts for each MoE layer
        for layer_idx, moe_layer in enumerate(moe_layers):
            experts_for_layer = self._create_experts_for_layer(layer_idx, moe_layer)
            self.experts[layer_idx] = experts_for_layer
    
    def _find_moe_layers(self) -> List[Tuple[str, nn.Module]]:
        """Find MoE layers in the model."""
        moe_layers = []
        
        for name, module in self.model.named_modules():
            # Detect MoE layers (simplified detection)
            if hasattr(module, 'num_experts') or 'moe' in name.lower():
                moe_layers.append((name, module))
        
        return moe_layers
    
    def _create_experts_for_layer(
        self,
        layer_idx: int,
        moe_layer: nn.Module
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Create experts for a specific MoE layer."""
        experts = {}
        experts_per_device = self.num_experts // len(self.devices)
        
        for expert_id in range(self.num_experts):
            device_idx = expert_id // experts_per_device
            device = self.devices[device_idx]
            
            # Create expert parameters (simplified)
            expert = {
                'device': device,
                'weight': torch.randn(1024, 1024, device=device) * 0.1,
                'bias': torch.randn(1024, device=device) * 0.1,
                'utilization': 0.0,
                'calls': 0
            }
            
            experts[expert_id] = expert
        
        return experts
    
    def expert_forward(self, inputs: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Expert parallel forward pass.
        
        Args:
            inputs: Input tensor
            layer_idx: MoE layer index
            
        Returns:
            Expert parallel output
        """
        batch_size, seq_len, hidden_dim = inputs.shape
        
        # Expert routing (simplified)
        routing_weights, expert_indices = self._route_experts(inputs, layer_idx)
        
        # Store routing information
        self.routing_weights[layer_idx] = routing_weights
        self.expert_indices[layer_idx] = expert_indices
        
        # Process through experts
        expert_outputs = self._process_experts(inputs, routing_weights, expert_indices, layer_idx)
        
        # Combine expert outputs
        output = self._combine_expert_outputs(expert_outputs, routing_weights, batch_size, seq_len, hidden_dim)
        
        self.stats['total_expert_calls'] += 1
        return output
    
    def _route_experts(
        self,
        inputs: torch.Tensor,
        layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.
        
        Args:
            inputs: Input tensor [batch_size, seq_len, hidden_dim]
            layer_idx: MoE layer index
            
        Returns:
            Tuple of (routing_weights, expert_indices)
        """
        batch_size, seq_len, hidden_dim = inputs.shape
        
        # Flatten for routing
        flat_inputs = inputs.view(-1, hidden_dim)  # [batch_size * seq_len, hidden_dim]
        
        # Simple routing: compute expert scores
        expert_scores = []
        for expert_id in range(self.num_experts):
            expert = self.experts[layer_idx][expert_id]
            # Compute similarity with expert center
            scores = torch.matmul(flat_inputs, expert['weight'].T)
            expert_scores.append(scores)
        
        expert_scores = torch.stack(expert_scores, dim=-1)  # [batch_size * seq_len, num_experts]
        
        # Select top-k experts
        top_k_scores, top_k_indices = torch.topk(expert_scores, self.top_k, dim=-1)
        
        # Compute routing weights (softmax)
        routing_weights = F.softmax(top_k_scores, dim=-1)
        
        return routing_weights, top_k_indices
    
    def _process_experts(
        self,
        inputs: torch.Tensor,
        routing_weights: torch.Tensor,
        expert_indices: torch.Tensor,
        layer_idx: int
    ) -> List[torch.Tensor]:
        """
        Process inputs through selected experts.
        
        Args:
            inputs: Input tensor
            routing_weights: Routing weights [batch_size * seq_len, top_k]
            expert_indices: Expert indices [batch_size * seq_len, top_k]
            layer_idx: MoE layer index
            
        Returns:
            List of expert outputs
        """
        batch_size, seq_len, hidden_dim = inputs.shape
        flat_inputs = inputs.view(-1, hidden_dim)
        
        expert_outputs = []
        
        for expert_id in range(self.num_experts):
            expert = self.experts[layer_idx][expert_id]
            device = expert['device']
            
            with torch.cuda.device(device):
                # Find tokens assigned to this expert
                token_mask = (expert_indices == expert_id).any(dim=-1)
                if not token_mask.any():
                    continue
                
                # Extract tokens for this expert
                expert_input = flat_inputs[token_mask].to(device)
                
                # Process through expert
                expert_output = F.linear(expert_input, expert['weight'], expert['bias'])
                expert_output = F.relu(expert_output)
                
                # Move back to original device
                expert_output = expert_output.cpu()
                
                expert_outputs.append(expert_output)
                
                # Update utilization statistics
                expert['utilization'] += token_mask.sum().item()
                expert['calls'] += 1
        
        return expert_outputs
    
    def _combine_expert_outputs(
        self,
        expert_outputs: List[torch.Tensor],
        routing_weights: torch.Tensor,
        batch_size: int,
        seq_len: int,
        hidden_dim: int
    ) -> torch.Tensor:
        """
        Combine expert outputs based on routing weights.
        
        Args:
            expert_outputs: List of expert output tensors
            routing_weights: Routing weights
            batch_size, seq_len, hidden_dim: Output dimensions
            
        Returns:
            Combined output tensor
        """
        # Initialize output
        output = torch.zeros(batch_size * seq_len, hidden_dim, device='cpu')
        
        # Combine outputs (simplified)
        # In reality, you would need more sophisticated combination logic
        for expert_output in expert_outputs:
            if expert_output.shape[0] > 0:
                output += expert_output
        
        # Reshape back to original shape
        output = output.view(batch_size, seq_len, hidden_dim)
        
        # Update routing efficiency
        self._update_routing_efficiency(routing_weights)
        
        return output
    
    def _update_routing_efficiency(self, routing_weights: torch.Tensor):
        """Update routing efficiency statistics."""
        # Calculate load balancing loss if enabled
        if self.load_balancing:
            # Simplified load balancing loss
            mean_routing = routing_weights.mean(dim=0)
            load_balance_loss = torch.var(mean_routing) * self.num_experts
            self.load_balancing_loss += load_balance_loss.item()
            self.stats['load_balance_loss'] = self.load_balancing_loss
        
        # Calculate routing efficiency
        routing_entropy = -torch.sum(routing_weights * torch.log(routing_weights + 1e-8), dim=-1)
        efficiency = torch.mean(routing_entropy).item()
        self.stats['routing_efficiency'] = efficiency
    
    def _calculate_expert_utilization(self) -> Dict[int, float]:
        """Calculate expert utilization rates."""
        utilization = {}
        
        for layer_idx, experts in self.experts.items():
            for expert_id, expert in experts.items():
                total_calls = sum(e['calls'] for e in experts.values())
                utilization[f"{layer_idx}_{expert_id}"] = (
                    expert['calls'] / max(total_calls, 1)
                )
        
        return utilization
    
    def get_expert_statistics(self) -> Dict[str, Any]:
        """Get expert parallel statistics."""
        utilization = self._calculate_expert_utilization()
        
        return {
            **self.stats,
            'num_experts': self.num_experts,
            'top_k': self.top_k,
            'capacity_factor': self.capacity_factor,
            'load_balancing': self.load_balancing,
            'expert_utilization': utilization,
            'average_routing_efficiency': self.stats['routing_efficiency']
        }


class ModelParallelManager:
    """
    Unified manager for all parallel training strategies.
    
    Coordinates tensor parallel, pipeline parallel, and expert parallel
    strategies for optimal performance.
    """
    
    def __init__(
        self,
        model: nn.Module,
        devices: List[torch.device],
        parallel_strategy: str = 'tensor_parallel',
        config: Optional[Dict] = None
    ):
        """
        Initialize model parallel manager.
        
        Args:
            model: Model to parallelize
            devices: List of available devices
            parallel_strategy: Primary parallelization strategy
            config: Configuration for parallelization
        """
        self.model = model
        self.devices = devices
        self.parallel_strategy = parallel_strategy
        self.config = config or {}
        
        # Parallel components
        self.tensor_parallel = None
        self.pipeline_parallel = None
        self.expert_parallel = None
        
        # Performance monitoring
        self.performance_monitor = defaultdict(list)
        self.bottleneck_analysis = {}
        
        # Initialize parallel components
        self._initialize_parallel_components()
    
    def _initialize_parallel_components(self):
        """Initialize all parallel components."""
        strategy = self.parallel_strategy
        
        if strategy == 'tensor_parallel':
            self.tensor_parallel = TensorParallel(
                self.model, self.devices, **self.config.get('tensor_parallel', {})
            )
        elif strategy == 'pipeline_parallel':
            self.pipeline_parallel = PipelineParallel(
                self.model, self.devices, **self.config.get('pipeline_parallel', {})
            )
        elif strategy == 'expert_parallel':
            self.expert_parallel = ExpertParallel(
                self.model, self.devices, **self.config.get('expert_parallel', {})
            )
        elif strategy == 'hybrid':
            # Initialize multiple strategies
            if 'tensor_parallel' in self.config:
                self.tensor_parallel = TensorParallel(
                    self.model, self.devices[:2], **self.config['tensor_parallel']
                )
            if 'pipeline_parallel' in self.config:
                self.pipeline_parallel = PipelineParallel(
                    self.model, self.devices, **self.config['pipeline_parallel']
                )
            if 'expert_parallel' in self.config:
                self.expert_parallel = ExpertParallel(
                    self.model, self.devices, **self.config['expert_parallel']
                )
    
    def forward(self, *inputs, **kwargs) -> torch.Tensor:
        """
        Unified parallel forward pass.
        
        Args:
            *inputs: Input tensors
            **kwargs: Input keyword arguments
            
        Returns:
            Parallel output tensor
        """
        start_time = time.time()
        
        if self.parallel_strategy == 'tensor_parallel':
            output = self.tensor_parallel.forward(*inputs, **kwargs)
        elif self.parallel_strategy == 'pipeline_parallel':
            output = self.pipeline_parallel.pipeline_forward(inputs[0])
        elif self.parallel_strategy == 'expert_parallel':
            output = self.expert_parallel.expert_forward(inputs[0], layer_idx=0)
        elif self.parallel_strategy == 'hybrid':
            output = self._hybrid_forward(*inputs, **kwargs)
        else:
            # Fallback to regular forward pass
            output = self.model(*inputs, **kwargs)
        
        # Monitor performance
        forward_time = time.time() - start_time
        self.performance_monitor['forward_time'].append(forward_time)
        
        return output
    
    def backward(self, loss: torch.Tensor):
        """
        Unified parallel backward pass.
        
        Args:
            loss: Loss tensor
        """
        start_time = time.time()
        
        if self.parallel_strategy == 'tensor_parallel':
            self.tensor_parallel.backward(loss)
        elif self.parallel_strategy == 'pipeline_parallel':
            self.pipeline_parallel.pipeline_backward(loss)
        elif self.parallel_strategy == 'expert_parallel':
            # Expert backward (simplified)
            pass
        elif self.parallel_strategy == 'hybrid':
            self._hybrid_backward(loss)
        
        # Monitor performance
        backward_time = time.time() - start_time
        self.performance_monitor['backward_time'].append(backward_time)
    
    def _hybrid_forward(self, *inputs, **kwargs) -> torch.Tensor:
        """Hybrid parallel forward pass."""
        # Use tensor parallel for early layers
        if self.tensor_parallel:
            intermediate = self.tensor_parallel.forward(*inputs, **kwargs)
        else:
            intermediate = inputs[0]
        
        # Use pipeline parallel for middle layers
        if self.pipeline_parallel:
            pipeline_output = self.pipeline_parallel.pipeline_forward(intermediate)
        else:
            pipeline_output = intermediate
        
        # Use expert parallel for final layers
        if self.expert_parallel:
            final_output = self.expert_parallel.expert_forward(pipeline_output, layer_idx=0)
        else:
            final_output = pipeline_output
        
        return final_output
    
    def _hybrid_backward(self, loss: torch.Tensor):
        """Hybrid parallel backward pass."""
        # Backward in reverse order
        if self.expert_parallel:
            pass  # Expert backward
        
        if self.pipeline_parallel:
            self.pipeline_parallel.pipeline_backward(loss)
        
        if self.tensor_parallel:
            self.tensor_parallel.backward(loss)
    
    def analyze_bottlenecks(self) -> Dict[str, Any]:
        """Analyze performance bottlenecks."""
        bottlenecks = {}
        
        # Communication bottleneck
        if hasattr(self, 'tensor_parallel'):
            comm_overhead = self.tensor_parallel.stats.get('communication_overhead', 0)
            bottlenecks['communication_bottleneck'] = comm_overhead
        
        # Compute bottleneck
        if self.performance_monitor['forward_time']:
            avg_forward_time = np.mean(self.performance_monitor['forward_time'][-100:])
            bottlenecks['compute_bottleneck'] = avg_forward_time
        
        # Memory bottleneck
        if torch.cuda.is_available():
            memory_usage = torch.cuda.memory_allocated() / (1024**3)  # GB
            bottlenecks['memory_bottleneck'] = memory_usage
        
        # Pipeline efficiency
        if hasattr(self, 'pipeline_parallel'):
            pipeline_eff = self.pipeline_parallel.stats.get('pipeline_efficiency', 0)
            bottlenecks['pipeline_bottleneck'] = 1.0 - pipeline_eff
        
        self.bottleneck_analysis = bottlenecks
        return bottlenecks
    
    def optimize_parallel_config(self) -> Dict[str, Any]:
        """Optimize parallel configuration based on performance analysis."""
        bottlenecks = self.analyze_bottlenecks()
        
        optimizations = {}
        
        # Optimize based on communication bottleneck
        if bottlenecks.get('communication_bottleneck', 0) > 0.5:
            optimizations['reduce_communication'] = {
                'strategy': 'increase_local_computation',
                'parameters': {'batch_size': 0.8}
            }
        
        # Optimize based on compute bottleneck
        if bottlenecks.get('compute_bottleneck', 0) > 0.1:
            optimizations['increase_parallelism'] = {
                'strategy': 'add_more_devices',
                'parameters': {'devices': len(self.devices) + 1}
            }
        
        # Optimize based on memory bottleneck
        if bottlenecks.get('memory_bottleneck', 0) > 0.8:
            optimizations['reduce_memory'] = {
                'strategy': 'enable_gradient_checkpointing',
                'parameters': {'checkpointing': True}
            }
        
        return optimizations
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive parallel training statistics."""
        stats = {
            'parallel_strategy': self.parallel_strategy,
            'devices': len(self.devices),
            'performance_monitor': {
                'avg_forward_time': np.mean(self.performance_monitor['forward_time'][-100:]) if self.performance_monitor['forward_time'] else 0,
                'avg_backward_time': np.mean(self.performance_monitor['backward_time'][-100:]) if self.performance_monitor['backward_time'] else 0
            },
            'bottleneck_analysis': self.analyze_bottlenecks()
        }
        
        # Add strategy-specific stats
        if self.tensor_parallel:
            stats['tensor_parallel_stats'] = self.tensor_parallel.get_parallel_statistics()
        
        if self.pipeline_parallel:
            stats['pipeline_parallel_stats'] = self.pipeline_parallel.get_pipeline_statistics()
        
        if self.expert_parallel:
            stats['expert_parallel_stats'] = self.expert_parallel.get_expert_statistics()
        
        return stats


# Utility functions for parallel training

def auto_select_parallel_strategy(
    model_size: int,
    num_devices: int,
    available_memory_gb: float,
    batch_size: int,
    sequence_length: int
) -> str:
    """
    Automatically select optimal parallel strategy.
    
    Args:
        model_size: Model size in millions of parameters
        num_devices: Number of available devices
        available_memory_gb: Available memory per device
        batch_size: Training batch size
        sequence_length: Sequence length
        
    Returns:
        Optimal parallel strategy
    """
    # Memory calculations
    model_memory_gb = model_size * 4 / 1024  # Approximate memory for model
    activation_memory_gb = (batch_size * sequence_length * model_size * 1e-6) / 16
    
    total_memory_needed = model_memory_gb + activation_memory_gb
    
    # Decision logic
    if total_memory_needed > available_memory_gb * 0.8:
        if num_devices >= 4:
            return 'hybrid'  # Combine all strategies
        elif model_size > 1000:  # 1B+ parameters
            return 'tensor_parallel'
        else:
            return 'pipeline_parallel'
    
    elif total_memory_needed > available_memory_gb * 0.5:
        if num_devices >= 8:
            return 'expert_parallel'  # Good for MoE models
        elif sequence_length > 1024:
            return 'tensor_parallel'  # Good for long sequences
        else:
            return 'pipeline_parallel'
    
    else:
        if model_size > 5000:  # 5B+ parameters
            return 'tensor_parallel'
        else:
            return 'pipeline_parallel'


def benchmark_parallel_strategies(
    model: nn.Module,
    devices: List[torch.device],
    batch_size: int,
    sequence_length: int,
    num_iterations: int = 10
) -> Dict[str, Dict]:
    """
    Benchmark different parallel strategies.
    
    Args:
        model: Model to benchmark
        devices: Available devices
        batch_size: Batch size for benchmark
        sequence_length: Sequence length for benchmark
        num_iterations: Number of benchmark iterations
        
    Returns:
        Benchmark results
    """
    results = {}
    strategies = ['tensor_parallel', 'pipeline_parallel', 'expert_parallel', 'hybrid']
    
    for strategy in strategies:
        try:
            # Create parallel manager
            config = {
                'tensor_parallel': {'strategy': 'row_wise'},
                'pipeline_parallel': {'micro_batch_size': batch_size // 4},
                'expert_parallel': {'num_experts': 8, 'top_k': 2}
            }
            
            manager = ModelParallelManager(model, devices, strategy, config.get(strategy, {}))
            
            # Generate test data
            test_input = torch.randn(batch_size, sequence_length, 1024)
            
            # Benchmark forward pass
            torch.cuda.synchronize()
            start_time = time.time()
            
            for _ in range(num_iterations):
                output = manager.forward(test_input)
                torch.cuda.synchronize()
            
            forward_time = time.time() - start_time
            
            # Get statistics
            stats = manager.get_comprehensive_statistics()
            
            results[strategy] = {
                'forward_time': forward_time / num_iterations,
                'memory_efficiency': 1.0 - (torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory),
                'parallel_efficiency': stats.get('performance_monitor', {}).get('parallel_efficiency', 0.0),
                'strategy_specific_stats': stats
            }
            
        except Exception as e:
            results[strategy] = {
                'error': str(e),
                'forward_time': float('inf'),
                'memory_efficiency': 0.0,
                'parallel_efficiency': 0.0
            }
    
    return results