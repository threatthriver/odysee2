"""
Training Utilities and Metrics

This module provides training metrics tracking, logging, and utilities for
monitoring training progress and model performance during large-scale
language model training.
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, List, Optional, Union, Tuple
import time
import json
import os
import pickle
import math
from collections import defaultdict, deque
import numpy as np
import logging


class TrainingMetrics:
    """
    Comprehensive training metrics tracker for language model training.
    
    Tracks various metrics including loss, accuracy, learning rates,
    memory usage, and performance statistics.
    """
    
    def __init__(
        self,
        log_dir: Optional[str] = None,
        log_frequency: int = 100,
        save_frequency: int = 1000,
        metrics_to_track: Optional[List[str]] = None,
        **kwargs
    ):
        self.log_dir = log_dir
        self.log_frequency = log_frequency
        self.save_frequency = save_frequency
        
        # Initialize tensorboard writer
        if log_dir is not None:
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None
        
        # Define metrics to track
        self.metrics_to_track = metrics_to_track or [
            'loss', 'perplexity', 'learning_rate', 'grad_norm',
            'memory_allocated', 'memory_cached', 'throughput',
            'step_time', 'validation_loss', 'validation_accuracy'
        ]
        
        # Initialize metric storage
        self.metrics_history: Dict[str, List[float]] = defaultdict(list)
        self.step_history: List[int] = []
        self.epoch_history: List[int] = []
        self.timestamp_history: List[float] = []
        
        # Current metrics (most recent values)
        self.current_metrics: Dict[str, float] = {}
        self.current_step = 0
        self.current_epoch = 0
        
        # Aggregated metrics (averaged over recent steps)
        self.aggregated_metrics: Dict[str, float] = {}
        
        # Metrics windows for rolling averages
        self.metrics_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Time tracking
        self.start_time = time.time()
        self.step_times: deque = deque(maxlen=100)
        
        # Performance counters
        self.total_steps = 0
        self.total_tokens_processed = 0
        self.total_forward_passes = 0
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        self.logger = logging.getLogger('TrainingMetrics')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def update_metrics(
        self,
        metrics: Dict[str, Union[float, int, torch.Tensor]],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        tokens_processed: Optional[int] = None,
        step_time: Optional[float] = None,
        **kwargs
    ):
        """Update training metrics."""
        current_time = time.time()
        
        # Update current step and epoch
        if step is not None:
            self.current_step = step
            self.step_history.append(step)
        if epoch is not None:
            self.current_epoch = epoch
            self.epoch_history.append(epoch)
        
        # Store timestamp
        self.timestamp_history.append(current_time)
        
        # Update metrics
        for key, value in metrics.items():
            if torch.is_tensor(value):
                value = value.item()
            
            # Update current metrics
            self.current_metrics[key] = value
            
            # Update history
            if key in self.metrics_to_track:
                self.metrics_history[key].append(value)
            
            # Update rolling window
            self.metrics_windows[key].append(value)
        
        # Update performance metrics
        if tokens_processed is not None:
            self.total_tokens_processed += tokens_processed
        
        self.total_steps += 1
        self.total_forward_passes += 1
        
        if step_time is not None:
            self.step_times.append(step_time)
        
        # Update aggregated metrics
        self._update_aggregated_metrics()
        
        # Log metrics periodically
        if self.current_step % self.log_frequency == 0:
            self._log_metrics()
        
        # Save metrics periodically
        if self.current_step % self.save_frequency == 0:
            self.save_metrics()
        
        # Write to tensorboard
        if self.writer is not None:
            self._write_to_tensorboard()
    
    def _update_aggregated_metrics(self):
        """Update aggregated metrics (rolling averages)."""
        for key, window in self.metrics_windows.items():
            if window:
                self.aggregated_metrics[key] = np.mean(list(window))
        
        # Calculate derived metrics
        if 'loss' in self.current_metrics:
            # Perplexity
            loss = self.current_metrics['loss']
            if loss > 0:
                self.aggregated_metrics['perplexity'] = math.exp(loss)
        
        # Throughput metrics
        if self.step_times:
            avg_step_time = np.mean(list(self.step_times))
            if avg_step_time > 0:
                self.aggregated_metrics['throughput'] = self.total_tokens_processed / (time.time() - self.start_time)
                self.aggregated_metrics['steps_per_second'] = 1.0 / avg_step_time
        
        # Memory metrics
        if torch.cuda.is_available():
            self.aggregated_metrics['memory_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
            self.aggregated_metrics['memory_cached'] = torch.cuda.memory_reserved() / 1024**3     # GB
    
    def _log_metrics(self):
        """Log current metrics to console and file."""
        log_message = f"Step {self.current_step}"
        
        # Add key metrics
        if 'loss' in self.current_metrics:
            log_message += f" | Loss: {self.current_metrics['loss']:.4f}"
            if 'perplexity' in self.aggregated_metrics:
                log_message += f" | PPL: {self.aggregated_metrics['perplexity']:.2f}"
        
        if 'learning_rate' in self.current_metrics:
            log_message += f" | LR: {self.current_metrics['learning_rate']:.2e}"
        
        if self.step_times:
            avg_step_time = np.mean(list(self.step_times))
            log_message += f" | Step time: {avg_step_time*1000:.2f}ms"
        
        if 'grad_norm' in self.current_metrics:
            log_message += f" | Grad norm: {self.current_metrics['grad_norm']:.4f}"
        
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            log_message += f" | GPU Memory: {memory_allocated:.2f}GB"
        
        self.logger.info(log_message)
    
    def _write_to_tensorboard(self):
        """Write metrics to tensorboard."""
        if self.writer is None:
            return
        
        # Write scalar metrics
        for key, value in self.current_metrics.items():
            if torch.is_tensor(value):
                value = value.item()
            self.writer.add_scalar(key, value, self.current_step)
        
        # Write aggregated metrics
        for key, value in self.aggregated_metrics.items():
            if key not in self.current_metrics:  # Don't duplicate
                self.writer.add_scalar(f'avg_{key}', value, self.current_step)
        
        # Write system metrics
        if torch.cuda.is_available():
            self.writer.add_scalar(
                'system/memory_allocated_gb',
                torch.cuda.memory_allocated() / 1024**3,
                self.current_step
            )
            self.writer.add_scalar(
                'system/memory_cached_gb',
                torch.cuda.memory_reserved() / 1024**3,
                self.current_step
            )
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current (most recent) metrics."""
        return {k: v.item() if torch.is_tensor(v) else v 
                for k, v in self.current_metrics.items()}
    
    def get_aggregated_metrics(self) -> Dict[str, float]:
        """Get aggregated metrics (rolling averages)."""
        return self.aggregated_metrics.copy()
    
    def get_step_statistics(self) -> Dict[str, Any]:
        """Get comprehensive step statistics."""
        return {
            'current_step': self.current_step,
            'current_epoch': self.current_epoch,
            'total_steps': self.total_steps,
            'total_tokens_processed': self.total_tokens_processed,
            'elapsed_time': time.time() - self.start_time,
            'avg_step_time': np.mean(list(self.step_times)) if self.step_times else 0,
            'throughput': self.aggregated_metrics.get('throughput', 0),
            'steps_per_second': self.aggregated_metrics.get('steps_per_second', 0)
        }
    
    def calculate_validation_metrics(self, predictions, targets) -> Dict[str, float]:
        """Calculate validation metrics from predictions and targets."""
        metrics = {}
        
        # Convert to numpy if tensors
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.cpu().numpy()
        
        # Accuracy metrics
        if predictions.shape == targets.shape:
            # Classification accuracy
            correct = np.sum(predictions == targets)
            total = targets.size
            metrics['accuracy'] = correct / total
            
            # Top-k accuracy (if applicable)
            if len(predictions.shape) > 1:
                # For classification with logits
                for k in [1, 5]:
                    if predictions.shape[1] >= k:
                        top_k_correct = 0
                        for i in range(len(targets)):
                            if targets[i] in np.argsort(predictions[i])[-k:]:
                                top_k_correct += 1
                        metrics[f'top_{k}_accuracy'] = top_k_correct / total
        
        # Loss (if predictions are logits)
        if len(predictions.shape) > 1:
            # Cross-entropy loss
            log_probs = torch.nn.functional.log_softmax(
                torch.tensor(predictions), dim=-1
            )
            loss = torch.nn.functional.nll_loss(
                log_probs, torch.tensor(targets)
            )
            metrics['validation_loss'] = loss.item()
            metrics['validation_perplexity'] = math.exp(loss.item())
        
        return metrics
    
    def save_metrics(self, filepath: Optional[str] = None):
        """Save metrics to file."""
        if filepath is None:
            filepath = os.path.join(self.log_dir, 'metrics.json') if self.log_dir else 'metrics.json'
        
        metrics_data = {
            'metrics_history': dict(self.metrics_history),
            'step_history': self.step_history,
            'epoch_history': self.epoch_history,
            'timestamp_history': self.timestamp_history,
            'current_step': self.current_step,
            'current_epoch': self.current_epoch,
            'total_steps': self.total_steps,
            'total_tokens_processed': self.total_tokens_processed,
            'elapsed_time': time.time() - self.start_time,
            'aggregated_metrics': self.aggregated_metrics
        }
        
        # Convert numpy arrays to lists for JSON serialization
        for key in metrics_data['metrics_history']:
            metrics_data['metrics_history'][key] = [
                float(x) if not isinstance(x, (int, float)) else x 
                for x in metrics_data['metrics_history'][key]
            ]
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)
    
    def load_metrics(self, filepath: str):
        """Load metrics from file."""
        with open(filepath, 'r') as f:
            metrics_data = json.load(f)
        
        self.metrics_history = defaultdict(list, metrics_data['metrics_history'])
        self.step_history = metrics_data['step_history']
        self.epoch_history = metrics_data['epoch_history']
        self.timestamp_history = metrics_data['timestamp_history']
        self.current_step = metrics_data['current_step']
        self.current_epoch = metrics_data['current_epoch']
        self.total_steps = metrics_data['total_steps']
        self.total_tokens_processed = metrics_data['total_tokens_processed']
        self.aggregated_metrics = metrics_data['aggregated_metrics']
    
    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics_history.clear()
        self.step_history.clear()
        self.epoch_history.clear()
        self.timestamp_history.clear()
        self.current_metrics.clear()
        self.aggregated_metrics.clear()
        self.metrics_windows.clear()
        self.step_times.clear()
        self.total_steps = 0
        self.total_tokens_processed = 0
        self.total_forward_passes = 0
        self.start_time = time.time()
    
    def close(self):
        """Close tensorboard writer and cleanup."""
        if self.writer:
            self.writer.close()
        
        # Save final metrics
        if self.log_dir:
            self.save_metrics()


class ModelProfiler:
    """
    Advanced model profiler for analyzing training performance.
    
    Provides detailed profiling of model operations, memory usage,
    and performance bottlenecks.
    """
    
    def __init__(self, model: nn.Module, input_shape: Optional[tuple] = None):
        self.model = model
        self.input_shape = input_shape
        
        # Profiling data
        self.activation_shapes: List[tuple] = []
        self.memory_usage: List[Dict[str, float]] = []
        self.timing_data: Dict[str, List[float]] = defaultdict(list)
        
        # Setup profiling hooks
        self.hooks = []
        self._setup_profiling_hooks()
    
    def _setup_profiling_hooks(self):
        """Setup hooks to capture model information."""
        def hook_fn(name):
            def hook(module, input, output):
                # Track activation shapes
                if isinstance(output, torch.Tensor):
                    self.activation_shapes.append(output.shape)
                
                # Track memory usage
                if torch.cuda.is_available():
                    memory_info = {
                        'allocated': torch.cuda.memory_allocated() / 1024**3,
                        'cached': torch.cuda.memory_reserved() / 1024**3
                    }
                    self.memory_usage.append(memory_info)
                
                # Track module type and parameters
                if not hasattr(module, '_profiling_info'):
                    module._profiling_info = {
                        'name': name,
                        'type': type(module).__name__,
                        'parameters': sum(p.numel() for p in module.parameters()),
                        'forward_calls': 0
                    }
                
                module._profiling_info['forward_calls'] += 1
                
            return hook
        
        # Register hooks for all modules
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_forward_hook(hook_fn(name))
                self.hooks.append(hook)
    
    def profile_forward(self, *inputs) -> Dict[str, Any]:
        """Profile a forward pass through the model."""
        # Clear previous profiling data
        self.activation_shapes.clear()
        self.memory_usage.clear()
        
        # Get initial memory state
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            initial_memory = torch.cuda.memory_allocated() / 1024**3
        
        # Warmup run
        with torch.no_grad():
            self.model(*inputs)
        
        # Profile forward pass
        start_time = time.time()
        
        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            output = self.model(*inputs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            end_time = time.time()
            final_memory = torch.cuda.memory_allocated() / 1024**3
        else:
            end_time = time.time()
            final_memory = 0
        
        # Analyze profiling results
        profile_stats = prof.key_averages()
        
        # Compile profiling report
        report = {
            'total_time': end_time - start_time,
            'memory_usage': {
                'initial_gb': initial_memory,
                'peak_gb': final_memory,
                'delta_gb': final_memory - initial_memory
            },
            'activation_shapes': self.activation_shapes,
            'top_operations': [],
            'module_info': []
        }
        
        # Get top operations
        if profile_stats:
            for item in profile_stats[:10]:  # Top 10 operations
                report['top_operations'].append({
                    'name': item.key,
                    'cpu_time': item.cpu_time_total,
                    'cuda_time': item.cuda_time_total if item.cuda_time_total > 0 else None,
                    'self_cpu_time': item.self_cpu_time_total,
                    'calls': item.count
                })
        
        # Get module information
        for module in self.model.modules():
            if hasattr(module, '_profiling_info'):
                report['module_info'].append(module._profiling_info)
        
        return report
    
    def analyze_model_complexity(self) -> Dict[str, Any]:
        """Analyze model complexity and parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Analyze parameter distribution
        param_distribution = defaultdict(int)
        for name, param in self.model.named_parameters():
            module_name = name.split('.')[0] if '.' in name else name
            param_distribution[module_name] += param.numel()
        
        # Calculate model size
        model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb,
            'parameter_distribution': dict(param_distribution)
        }
    
    def suggest_optimizations(self) -> List[str]:
        """Suggest optimizations based on profiling results."""
        suggestions = []
        
        # Analyze memory usage
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1024**3
            if current_memory > 10:  # More than 10GB
                suggestions.append("Consider using gradient checkpointing to reduce memory usage")
                suggestions.append("Implement mixed precision training")
        
        # Analyze model complexity
        complexity = self.analyze_model_complexity()
        if complexity['total_parameters'] > 100_000_000:  # More than 100M parameters
            suggestions.append("Consider model parallelism for large models")
            suggestions.append("Use gradient accumulation for effective large batch training")
        
        # Analyze operations
        if len(self.activation_shapes) > 100:
            suggestions.append("Consider efficient attention mechanisms like FlashAttention")
        
        return suggestions
    
    def remove_hooks(self):
        """Remove all profiling hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class EarlyStopping:
    """
    Early stopping mechanism for training.
    
    Monitors validation metrics and stops training when performance
    stops improving for a specified number of epochs.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        monitor: str = 'validation_loss',
        mode: str = 'min',
        restore_best_weights: bool = True,
        verbose: bool = True,
        **kwargs
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.best_weights = None
        self.should_stop = False
        
        # Validation history
        self.history: List[float] = []
        
    def __call__(self, current_value: float, model: Optional[nn.Module] = None) -> bool:
        """Check if training should stop."""
        self.history.append(current_value)
        
        # Determine if this is a better value
        is_better = False
        if self.mode == 'min':
            if current_value < self.best_value - self.min_delta:
                is_better = True
        else:  # mode == 'max'
            if current_value > self.best_value + self.min_delta:
                is_better = True
        
        if is_better:
            self.best_value = current_value
            self.counter = 0
            
            # Save best weights
            if self.restore_best_weights and model is not None:
                self.best_weights = {
                    name: param.clone().detach()
                    for name, param in model.state_dict().items()
                }
        else:
            self.counter += 1
        
        # Check if we should stop
        if self.counter >= self.patience:
            self.should_stop = True
            
            if self.verbose:
                print(f"Early stopping triggered after {len(self.history)} epochs")
                print(f"Best {self.monitor}: {self.best_value:.6f}")
            
            # Restore best weights
            if self.restore_best_weights and model is not None and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                if self.verbose:
                    print("Restored best model weights")
        
        return self.should_stop
    
    def reset(self):
        """Reset early stopping state."""
        self.best_value = float('inf') if self.mode == 'min' else float('-inf')
        self.counter = 0
        self.should_stop = False
        self.best_weights = None
        self.history.clear()


class ModelValidator:
    """
    Model validation utilities for language models.
    
    Provides comprehensive validation metrics and evaluation tools.
    """
    
    def __init__(
        self,
        metrics_to_compute: Optional[List[str]] = None,
        **kwargs
    ):
        self.metrics_to_compute = metrics_to_compute or [
            'loss', 'perplexity', 'accuracy', 'bleu', 'rouge'
        ]
    
    def validate_model(
        self,
        model: nn.Module,
        dataloader,
        loss_fn,
        device: torch.device,
        compute_metrics: bool = True
    ) -> Dict[str, float]:
        """Comprehensive model validation."""
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        correct_predictions = 0
        total_predictions = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                batch = {k: v.to(device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = model(**batch)
                
                # Compute loss
                if 'loss' in outputs:
                    loss = outputs['loss']
                else:
                    predictions = outputs['logits'] if 'logits' in outputs else outputs
                    targets = batch.get('labels', batch.get('target'))
                    loss = loss_fn(predictions, targets)
                
                # Accumulate metrics
                batch_tokens = batch.get('attention_mask', torch.ones_like(batch.get('input_ids'))).sum()
                total_loss += loss.item() * batch_tokens.item()
                total_tokens += batch_tokens.item()
                
                # Accuracy metrics (if applicable)
                if 'logits' in outputs:
                    predictions = outputs['logits']
                    targets = batch.get('labels', batch.get('target'))
                    
                    if predictions.shape == targets.shape:
                        # Classification accuracy
                        pred_classes = torch.argmax(predictions, dim=-1)
                        correct = (pred_classes == targets).sum().item()
                        correct_predictions += correct
                        total_predictions += targets.numel()
                    
                    # Store for later metrics computation
                    all_predictions.append(predictions.cpu())
                    all_targets.append(targets.cpu())
        
        # Calculate aggregate metrics
        metrics = {
            'validation_loss': total_loss / max(total_tokens, 1),
            'validation_perplexity': math.exp(total_loss / max(total_tokens, 1)) if total_loss > 0 else float('inf'),
            'validation_samples': len(dataloader.dataset),
            'validation_tokens': total_tokens
        }
        
        # Add accuracy metrics
        if total_predictions > 0:
            metrics['validation_accuracy'] = correct_predictions / total_predictions
        
        # Compute additional metrics
        if compute_metrics and all_predictions:
            additional_metrics = self._compute_additional_metrics(
                all_predictions, all_targets
            )
            metrics.update(additional_metrics)
        
        return metrics
    
    def _compute_additional_metrics(
        self,
        predictions: List[torch.Tensor],
        targets: List[torch.Tensor]
    ) -> Dict[str, float]:
        """Compute additional metrics from predictions and targets."""
        metrics = {}
        
        # Concatenate all predictions and targets
        all_pred = torch.cat(predictions, dim=0)
        all_target = torch.cat(targets, dim=0)
        
        # Top-k accuracy
        for k in [1, 5, 10]:
            if all_pred.shape[1] >= k:
                top_k_correct = 0
                for i in range(len(all_target)):
                    if all_target[i] in torch.topk(all_pred[i], k).indices:
                        top_k_correct += 1
                metrics[f'validation_top_{k}_accuracy'] = top_k_correct / len(all_target)
        
        return metrics
    
    def compute_perplexity(
        self,
        model: nn.Module,
        dataloader,
        device: torch.device,
        max_batches: Optional[int] = None
    ) -> float:
        """Compute model perplexity on a dataset."""
        model.eval()
        total_log_prob = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                
                # Move batch to device
                batch = {k: v.to(device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = model(**batch)
                
                # Compute log probabilities
                if 'logits' in outputs:
                    logits = outputs['logits']
                    targets = batch.get('labels', batch.get('input_ids'))
                    
                    # Shift targets for causal language modeling
                    if targets.shape == logits.shape:
                        # Standard classification
                        log_probs = torch.log_softmax(logits, dim=-1)
                        token_log_probs = torch.gather(
                            log_probs, dim=-1, 
                            index=targets.unsqueeze(-1)
                        ).squeeze(-1)
                    else:
                        # Causal language modeling
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_targets = targets[..., 1:].contiguous()
                        
                        log_probs = torch.log_softmax(shift_logits, dim=-1)
                        token_log_probs = torch.gather(
                            log_probs, dim=-1,
                            index=shift_targets.unsqueeze(-1)
                        ).squeeze(-1)
                    
                    # Mask out padding tokens
                    if 'attention_mask' in batch:
                        mask = batch['attention_mask'][..., 1:]  # Align with shifted targets
                        token_log_probs = token_log_probs * mask
                    
                    # Sum log probabilities and tokens
                    total_log_prob += token_log_probs.sum().item()
                    total_tokens += mask.sum().item() if 'attention_mask' in batch else token_log_probs.numel()
        
        # Compute perplexity
        avg_log_prob = total_log_prob / max(total_tokens, 1)
        perplexity = math.exp(-avg_log_prob)
        
        return perplexity