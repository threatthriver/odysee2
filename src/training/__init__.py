"""
Training Infrastructure Package

This package contains the core training infrastructure components for
large-scale language model training, including gradient accumulation,
mixed precision training, and advanced optimization strategies.
"""

from .training_loop import TrainingLoop
from .gradient_accumulation import GradientAccumulator, DynamicGradientAccumulator
from .mixed_precision import MixedPrecisionTrainer, MemoryEfficientMixedPrecision
from .optimizers import AdamW, LAMB, AdaFactor, RMSprop
from .lr_schedulers import (
    get_linear_schedule_with_warmup, 
    get_cosine_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    OneCycleLRWithWarmup,
    CyclicLRWithWarmup
)
from .training_utils import TrainingMetrics, ModelProfiler, EarlyStopping, ModelValidator
from .checkpoint_manager import CheckpointManager, DistributedCheckpointManager

__all__ = [
    "TrainingLoop",
    "GradientAccumulator", 
    "DynamicGradientAccumulator",
    "MixedPrecisionTrainer",
    "MemoryEfficientMixedPrecision",
    "AdamW",
    "LAMB",
    "AdaFactor",
    "RMSprop",
    "get_linear_schedule_with_warmup",
    "get_cosine_schedule_with_warmup",
    "get_polynomial_decay_schedule_with_warmup",
    "OneCycleLRWithWarmup",
    "CyclicLRWithWarmup", 
    "TrainingMetrics",
    "ModelProfiler",
    "EarlyStopping",
    "ModelValidator",
    "CheckpointManager",
    "DistributedCheckpointManager"
]