"""
Checkpoint Management and Versioning

This module implements comprehensive checkpoint management for language model training,
including versioning, rollback capabilities, and distributed checkpoint handling.
"""

import torch
import torch.nn as nn
import os
import json
import shutil
import glob
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import pickle
import warnings
from pathlib import Path


class CheckpointManager:
    """
    Advanced checkpoint manager for language model training.
    
    Features:
    - Automatic checkpoint saving with versioning
    - Checkpoint recovery and rollback
    - Distributed training checkpoint handling
    - Checkpoint compression and optimization
    - Metadata tracking for checkpoints
    """
    
    def __init__(
        self,
        save_dir: str,
        model_name: str = "model",
        save_frequency: int = 1000,
        keep_n_checkpoints: int = 5,
        max_checkpoint_size_gb: float = 10.0,
        compression: bool = True,
        save_optimizer: bool = True,
        save_lr_scheduler: bool = True,
        save_training_state: bool = True,
        metadata_file: str = "metadata.json",
        **kwargs
    ):
        self.save_dir = Path(save_dir)
        self.model_name = model_name
        self.save_frequency = save_frequency
        self.keep_n_checkpoints = keep_n_checkpoints
        self.max_checkpoint_size_gb = max_checkpoint_size_gb
        self.compression = compression
        self.save_optimizer = save_optimizer
        self.save_lr_scheduler = save_lr_scheduler
        self.save_training_state = save_training_state
        self.metadata_file = metadata_file
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint tracking
        self.checkpoints: Dict[str, Dict[str, Any]] = {}
        self.current_checkpoint = 0
        self.best_metric_value = float('inf')  # or float('-inf') for metrics to maximize
        self.best_checkpoint_path = None
        
        # Load existing checkpoints
        self._load_checkpoint_registry()
        
        # Setup metadata tracking
        self._setup_metadata_tracking()
    
    def _setup_metadata_tracking(self):
        """Setup metadata tracking for checkpoints."""
        self.metadata_path = self.save_dir / self.metadata_file
        
        # Load or create metadata
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                'model_name': self.model_name,
                'creation_date': datetime.now().isoformat(),
                'checkpoints': {},
                'best_checkpoints': {},
                'training_history': []
            }
            self._save_metadata()
    
    def _save_metadata(self):
        """Save metadata to file."""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _load_checkpoint_registry(self):
        """Load existing checkpoint registry."""
        checkpoint_files = glob.glob(str(self.save_dir / f"{self.model_name}_checkpoint_*.pth"))
        
        for checkpoint_file in checkpoint_files:
            checkpoint_name = Path(checkpoint_file).stem
            checkpoint_info = self._analyze_checkpoint_file(checkpoint_file)
            self.checkpoints[checkpoint_name] = checkpoint_info
    
    def _analyze_checkpoint_file(self, filepath: str) -> Dict[str, Any]:
        """Analyze a checkpoint file to extract information."""
        info = {
            'filepath': filepath,
            'size_gb': 0,
            'timestamp': 0,
            'step': 0,
            'epoch': 0,
            'metric_value': None,
            'compressed': False
        }
        
        try:
            # Get file stats
            stat = os.stat(filepath)
            info['size_gb'] = stat.st_size / (1024**3)
            info['timestamp'] = stat.st_mtime
            
            # Try to extract metadata from checkpoint
            checkpoint = torch.load(filepath, map_location='cpu')
            
            if isinstance(checkpoint, dict):
                info['step'] = checkpoint.get('step', 0)
                info['epoch'] = checkpoint.get('epoch', 0)
                info['metric_value'] = checkpoint.get('best_metric_value')
                
                # Check if checkpoint is compressed
                if 'model_state_dict' in checkpoint:
                    sample_state = checkpoint['model_state_dict']
                    if isinstance(sample_state, dict) and any(
                        tensor.dtype == torch.float16 for tensor in sample_state.values() 
                        if torch.is_tensor(tensor)
                    ):
                        info['compressed'] = True
            
            del checkpoint  # Free memory
            
        except Exception as e:
            warnings.warn(f"Could not analyze checkpoint file {filepath}: {e}")
        
        return info
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        epoch: int = 0,
        step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        training_state: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
        additional_state: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a training checkpoint.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer state (optional)
            lr_scheduler: Learning rate scheduler (optional)
            epoch: Current epoch
            step: Current step
            metrics: Training metrics
            training_state: Additional training state
            is_best: Whether this is the best checkpoint so far
            additional_state: Additional state to save
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_name = f"{self.model_name}_checkpoint_{self.current_checkpoint}"
        checkpoint_path = self.save_dir / f"{checkpoint_name}.pth"
        
        # Build checkpoint dictionary
        checkpoint = {
            'model_name': self.model_name,
            'checkpoint_number': self.current_checkpoint,
            'step': step,
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'model_state_dict': model.state_dict(),
            'best_metric_value': self.best_metric_value,
            'model_config': self._extract_model_config(model)
        }
        
        # Add optimizer state
        if optimizer is not None and self.save_optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        # Add LR scheduler state
        if lr_scheduler is not None and self.save_lr_scheduler:
            checkpoint['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
        
        # Add metrics
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        # Add training state
        if training_state is not None and self.save_training_state:
            checkpoint['training_state'] = training_state
        
        # Add additional state
        if additional_state is not None:
            checkpoint.update(additional_state)
        
        # Update best checkpoint tracking
        if is_best and metrics is not None:
            primary_metric = self._get_primary_metric(metrics)
            if self._is_better_metric(primary_metric, self.best_metric_value):
                self.best_metric_value = primary_metric
                self.best_checkpoint_path = str(checkpoint_path)
                checkpoint['is_best'] = True
                
                # Update metadata
                self.metadata['best_checkpoints'][self.model_name] = {
                    'checkpoint_path': str(checkpoint_path),
                    'step': step,
                    'epoch': epoch,
                    'metric_value': primary_metric,
                    'metrics': metrics,
                    'timestamp': checkpoint['timestamp']
                }
        
        # Save checkpoint
        self._save_checkpoint(checkpoint, checkpoint_path)
        
        # Update registry
        checkpoint_info = self._analyze_checkpoint_file(str(checkpoint_path))
        self.checkpoints[checkpoint_name] = checkpoint_info
        
        # Update metadata
        self.metadata['checkpoints'][checkpoint_name] = {
            'path': str(checkpoint_path),
            'step': step,
            'epoch': epoch,
            'metrics': metrics,
            'is_best': checkpoint.get('is_best', False),
            'timestamp': checkpoint['timestamp']
        }
        
        self._save_metadata()
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
        self.current_checkpoint += 1
        
        return str(checkpoint_path)
    
    def _extract_model_config(self, model: nn.Module) -> Dict[str, Any]:
        """Extract model configuration from model."""
        config = {
            'model_class': type(model).__name__,
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'model_size_mb': sum(p.numel() * 4 for p in model.parameters()) / (1024 * 1024)
        }
        
        # Add model-specific configuration if available
        if hasattr(model, 'config'):
            config['config'] = model.config.__dict__
        
        return config
    
    def _get_primary_metric(self, metrics: Dict[str, float]) -> float:
        """Get primary metric for checkpoint ranking."""
        # Priority order for metrics
        priority_metrics = [
            'validation_loss', 'val_loss', 'loss',
            'validation_perplexity', 'val_perplexity', 'perplexity',
            'validation_accuracy', 'val_accuracy', 'accuracy'
        ]
        
        for metric_name in priority_metrics:
            if metric_name in metrics:
                return metrics[metric_name]
        
        # If no priority metric found, use the first metric
        if metrics:
            return list(metrics.values())[0]
        
        return self.best_metric_value
    
    def _is_better_metric(self, current: float, best: float) -> bool:
        """Determine if current metric is better than best metric."""
        # For loss-type metrics, lower is better
        if any(metric in str(current).lower() for metric in ['loss', 'perplexity']):
            return current < best
        else:
            return current > best
    
    def _save_checkpoint(self, checkpoint: Dict[str, Any], filepath: Path):
        """Save checkpoint with optional compression."""
        try:
            # Check file size
            temp_path = filepath.with_suffix('.tmp')
            
            # Use torch.save with appropriate settings
            if self.compression and torch.cuda.is_available():
                # Try to use compression
                torch.save(checkpoint, temp_path, _use_new_zipfile_serialization=False)
            else:
                torch.save(checkpoint, temp_path)
            
            # Move to final location
            shutil.move(str(temp_path), str(filepath))
            
            # Check file size and warn if too large
            file_size_gb = filepath.stat().st_size / (1024**3)
            if file_size_gb > self.max_checkpoint_size_gb:
                warnings.warn(f"Checkpoint size ({file_size_gb:.2f}GB) exceeds "
                            f"max_checkpoint_size_gb ({self.max_checkpoint_size_gb:.2f})")
            
        except Exception as e:
            if filepath.exists():
                filepath.unlink()  # Clean up failed checkpoint
            raise e
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond the keep limit."""
        # Get all checkpoints sorted by step
        checkpoint_items = sorted(
            self.checkpoints.items(),
            key=lambda x: x[1]['step'],
            reverse=True
        )
        
        # Keep only the most recent checkpoints
        checkpoints_to_remove = checkpoint_items[self.keep_n_checkpoints:]
        
        for checkpoint_name, checkpoint_info in checkpoints_to_remove:
            filepath = Path(checkpoint_info['filepath'])
            
            # Remove checkpoint file
            if filepath.exists():
                filepath.unlink()
            
            # Remove from registry
            del self.checkpoints[checkpoint_name]
            
            # Remove from metadata
            if checkpoint_name in self.metadata['checkpoints']:
                del self.metadata['checkpoints'][checkpoint_name]
            
            # If this was the best checkpoint, update tracking
            if checkpoint_info.get('metric_value') == self.best_metric_value:
                if self.best_checkpoint_path == str(filepath):
                    self.best_checkpoint_path = None
                    # Find new best checkpoint
                    self._update_best_checkpoint()
        
        # Save updated metadata
        self._save_metadata()
    
    def _update_best_checkpoint(self):
        """Update best checkpoint based on current checkpoints."""
        self.best_metric_value = float('inf') if self._is_loss_metric() else float('-inf')
        self.best_checkpoint_path = None
        
        for checkpoint_name, checkpoint_info in self.checkpoints.items():
            metric_value = checkpoint_info.get('metric_value')
            if metric_value is not None and self._is_better_metric(metric_value, self.best_metric_value):
                self.best_metric_value = metric_value
                self.best_checkpoint_path = checkpoint_info['filepath']
    
    def _is_loss_metric(self) -> bool:
        """Check if we're optimizing for a loss-type metric."""
        # This is a heuristic - you might want to make this configurable
        return True  # Assume loss metrics by default
    
    def load_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        map_location: Union[str, torch.device] = 'cpu',
        strict: bool = True,
        load_optimizer: bool = True,
        load_scheduler: bool = True
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint (None for best checkpoint)
            model: Model to load weights into
            optimizer: Optimizer to load state into
            lr_scheduler: LR scheduler to load state into
            map_location: Device to map tensors to
            strict: Whether to enforce strict key matching
            load_optimizer: Whether to load optimizer state
            load_scheduler: Whether to load scheduler state
            
        Returns:
            Checkpoint information dictionary
        """
        # Determine which checkpoint to load
        if checkpoint_path is None:
            checkpoint_path = self.best_checkpoint_path
            if checkpoint_path is None:
                raise ValueError("No checkpoint path specified and no best checkpoint available")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Load model weights
        if model is not None and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        # Load optimizer state
        if optimizer is not None and load_optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load LR scheduler state
        if lr_scheduler is not None and load_scheduler and 'lr_scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        # Return checkpoint information
        checkpoint_info = {
            'checkpoint_path': checkpoint_path,
            'step': checkpoint.get('step', 0),
            'epoch': checkpoint.get('epoch', 0),
            'timestamp': checkpoint.get('timestamp', ''),
            'metrics': checkpoint.get('metrics', {}),
            'model_config': checkpoint.get('model_config', {}),
            'training_state': checkpoint.get('training_state', {})
        }
        
        # Clean up
        del checkpoint
        
        return checkpoint_info
    
    def load_best_checkpoint(
        self,
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        map_location: Union[str, torch.device] = 'cpu',
        strict: bool = True
    ) -> Dict[str, Any]:
        """Load the best checkpoint."""
        return self.load_checkpoint(
            checkpoint_path=None,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            map_location=map_location,
            strict=strict
        )
    
    def get_checkpoint_info(self, checkpoint_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific checkpoint."""
        return self.checkpoints.get(checkpoint_name)
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all checkpoints with their information."""
        return [
            {
                'name': name,
                'info': info,
                'selected': self.best_checkpoint_path == info['filepath']
            }
            for name, info in self.checkpoints.items()
        ]
    
    def delete_checkpoint(self, checkpoint_name: str) -> bool:
        """Delete a specific checkpoint."""
        if checkpoint_name not in self.checkpoints:
            return False
        
        checkpoint_info = self.checkpoints[checkpoint_name]
        filepath = Path(checkpoint_info['filepath'])
        
        # Remove file
        if filepath.exists():
            filepath.unlink()
        
        # Remove from registry
        del self.checkpoints[checkpoint_name]
        
        # Remove from metadata
        if checkpoint_name in self.metadata['checkpoints']:
            del self.metadata['checkpoints'][checkpoint_name]
        
        # Update best checkpoint if necessary
        if checkpoint_info.get('metric_value') == self.best_metric_value:
            self._update_best_checkpoint()
        
        # Save metadata
        self._save_metadata()
        
        return True
    
    def get_checkpoint_history(self) -> List[Dict[str, Any]]:
        """Get checkpoint history for plotting/analyzing training progress."""
        history = []
        
        for checkpoint_name, info in self.checkpoints.items():
            if info.get('step', 0) > 0:
                history.append({
                    'step': info['step'],
                    'epoch': info['epoch'],
                    'timestamp': info['timestamp'],
                    'metric_value': info.get('metric_value'),
                    'size_gb': info['size_gb'],
                    'is_best': self.best_checkpoint_path == info['filepath']
                })
        
        # Sort by step
        history.sort(key=lambda x: x['step'])
        
        return history
    
    def export_checkpoint(self, checkpoint_name: str, export_path: str, format: str = 'onnx') -> bool:
        """Export a checkpoint to a different format."""
        if checkpoint_name not in self.checkpoints:
            return False
        
        checkpoint_info = self.checkpoints[checkpoint_name]
        checkpoint_path = checkpoint_info['filepath']
        
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            if format.lower() == 'onnx':
                # Export to ONNX format
                self._export_to_onnx(checkpoint, export_path)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            # Clean up
            del checkpoint
            
            return True
            
        except Exception as e:
            print(f"Error exporting checkpoint: {e}")
            return False
    
    def _export_to_onnx(self, checkpoint: Dict[str, Any], export_path: str):
        """Export checkpoint to ONNX format (placeholder implementation)."""
        # This is a simplified placeholder
        # In practice, you would need to:
        # 1. Recreate the model architecture
        # 2. Load weights
        # 3. Create a dummy input
        # 4. Export using torch.onnx.export
        
        # For now, just save the state dict
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            torch.save(model_state, export_path.replace('.onnx', '_state_dict.pth'))
    
    def cleanup_all_checkpoints(self):
        """Remove all checkpoints and reset manager."""
        # Remove all checkpoint files
        for checkpoint_info in self.checkpoints.values():
            filepath = Path(checkpoint_info['filepath'])
            if filepath.exists():
                filepath.unlink()
        
        # Clear registry
        self.checkpoints.clear()
        self.best_checkpoint_path = None
        self.best_metric_value = float('inf')
        self.current_checkpoint = 0
        
        # Reset metadata
        self.metadata['checkpoints'].clear()
        self.metadata['best_checkpoints'].clear()
        self._save_metadata()
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics for checkpoints."""
        total_size = sum(info['size_gb'] for info in self.checkpoints.values())
        
        return {
            'total_checkpoints': len(self.checkpoints),
            'total_size_gb': total_size,
            'avg_checkpoint_size_gb': total_size / len(self.checkpoints) if self.checkpoints else 0,
            'keep_n_checkpoints': self.keep_n_checkpoints,
            'max_checkpoint_size_gb': self.max_checkpoint_size_gb,
            'compression_enabled': self.compression,
            'save_directory': str(self.save_dir)
        }


class DistributedCheckpointManager(CheckpointManager):
    """
    Checkpoint manager for distributed training scenarios.
    
    Handles checkpoint synchronization across multiple processes/nodes.
    """
    
    def __init__(
        self,
        save_dir: str,
        rank: int = 0,
        world_size: int = 1,
        **kwargs
    ):
        super().__init__(save_dir, **kwargs)
        self.rank = rank
        self.world_size = world_size
        
        # Create rank-specific subdirectory
        if world_size > 1:
            self.rank_dir = self.save_dir / f"rank_{rank}"
            self.rank_dir.mkdir(parents=True, exist_ok=True)
            self.save_dir = self.rank_dir
    
    def save_distributed_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        **kwargs
    ) -> str:
        """Save checkpoint in distributed setting."""
        if self.world_size > 1:
            # Add distributed info to checkpoint
            additional_state = {
                'distributed': {
                    'rank': self.rank,
                    'world_size': self.world_size
                }
            }
            kwargs.setdefault('additional_state', {})
            kwargs['additional_state'].update(additional_state)
        
        return super().save_checkpoint(model, optimizer, **kwargs)
    
    def load_distributed_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Load checkpoint in distributed setting."""
        # In distributed settings, typically load the same checkpoint on all ranks
        checkpoint_info = super().load_checkpoint(model=model, optimizer=optimizer, **kwargs)
        
        # Verify distributed consistency
        if 'distributed' in checkpoint_info.get('training_state', {}):
            dist_info = checkpoint_info['training_state']['distributed']
            if dist_info['rank'] != self.rank:
                warnings.warn(f"Loaded checkpoint from rank {dist_info['rank']} "
                            f"but current rank is {self.rank}")
        
        return checkpoint_info


class CheckpointValidator:
    """
    Utility for validating checkpoint integrity and compatibility.
    """
    
    @staticmethod
    def validate_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
        """Validate a checkpoint file."""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Check required fields
            required_fields = ['model_state_dict']
            for field in required_fields:
                if field not in checkpoint:
                    validation_result['errors'].append(f"Missing required field: {field}")
                    validation_result['valid'] = False
            
            # Check model state dict
            if 'model_state_dict' in checkpoint:
                model_state = checkpoint['model_state_dict']
                if not isinstance(model_state, dict):
                    validation_result['errors'].append("model_state_dict is not a dictionary")
                    validation_result['valid'] = False
                else:
                    # Check for common issues
                    for key, value in model_state.items():
                        if not torch.is_tensor(value):
                            validation_result['warnings'].append(f"Non-tensor value in state dict: {key}")
            
            # Extract info
            validation_result['info'] = {
                'checkpoint_size_gb': Path(checkpoint_path).stat().st_size / (1024**3),
                'step': checkpoint.get('step', 'unknown'),
                'epoch': checkpoint.get('epoch', 'unknown'),
                'timestamp': checkpoint.get('timestamp', 'unknown'),
                'model_name': checkpoint.get('model_name', 'unknown'),
                'has_optimizer': 'optimizer_state_dict' in checkpoint,
                'has_scheduler': 'lr_scheduler_state_dict' in checkpoint,
                'has_metrics': 'metrics' in checkpoint
            }
            
            del checkpoint
            
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Failed to load checkpoint: {str(e)}")
        
        return validation_result
    
    @staticmethod
    def compare_checkpoints(checkpoint1_path: str, checkpoint2_path: str) -> Dict[str, Any]:
        """Compare two checkpoints and report differences."""
        checkpoint1 = torch.load(checkpoint1_path, map_location='cpu')
        checkpoint2 = torch.load(checkpoint2_path, map_location='cpu')
        
        comparison = {
            'model_size_difference': 0,
            'step_difference': 0,
            'parameter_differences': {},
            'missing_in_checkpoint1': [],
            'missing_in_checkpoint2': []
        }
        
        # Compare model state dicts
        if 'model_state_dict' in checkpoint1 and 'model_state_dict' in checkpoint2:
            state1 = checkpoint1['model_state_dict']
            state2 = checkpoint2['model_state_dict']
            
            # Find parameter differences
            all_keys = set(state1.keys()) | set(state2.keys())
            
            for key in all_keys:
                if key in state1 and key in state2:
                    if not torch.equal(state1[key], state2[key]):
                        comparison['parameter_differences'][key] = {
                            'shape': state1[key].shape,
                            'diff_norm': (state1[key] - state2[key]).norm().item()
                        }
                elif key in state1:
                    comparison['missing_in_checkpoint2'].append(key)
                else:
                    comparison['missing_in_checkpoint1'].append(key)
        
        # Compare steps
        comparison['step_difference'] = checkpoint2.get('step', 0) - checkpoint1.get('step', 0)
        
        # Compare file sizes
        size1 = Path(checkpoint1_path).stat().st_size
        size2 = Path(checkpoint2_path).stat().st_size
        comparison['model_size_difference'] = (size2 - size1) / (1024**3)  # GB
        
        del checkpoint1, checkpoint2
        
        return comparison