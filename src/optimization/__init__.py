"""
Advanced Optimization Framework for Language Models

This module provides cutting-edge optimization techniques for efficient training
of large language models with advanced memory management and hardware optimizations.
"""

from .attention_optimizations import (
    FlashAttention,
    LinearAttention,
    SparseAttention,
    MultiQueryAttention,
    create_attention_mechanism,
    compare_attention_complexity
)

from .advanced_optimizers import (
    LAMBOptimizer,
    AdaFactorOptimizer,
    AdamW8Bit,
    AdaptiveLRScheduler,
    create_optimizer,
    get_memory_efficient_optimizer
)

from .gradient_strategies import (
    GradientClipper,
    GradientAccumulator,
    GradientCheckpointing,
    MemoryEfficientBackprop,
    create_gradient_optimizer,
    benchmark_gradient_strategies
)

from .hardware_optimization import (
    CUDAMemoryManager,
    MixedPrecisionTrainer,
    CustomCudaKernels,
    MemoryPool,
    get_optimal_config,
    benchmark_hardware_optimizations
)

from .parallel_training import (
    TensorParallel,
    PipelineParallel,
    ExpertParallel,
    ModelParallelManager,
    auto_select_parallel_strategy,
    benchmark_parallel_strategies
)

from .memory_management import (
    MemoryProfiler,
    GradientCheckpointing as MemGradientCheckpointing,
    DynamicMemoryAllocator,
    EfficientTrainingOrchestrator,
    auto_configure_memory_management
)

from .dynamic_training import (
    CurriculumLearning,
    ProgressiveModelExpansion,
    AdaptiveBatchSizing,
    DynamicTrainingOrchestrator,
    DifficultyLevel,
    CurriculumStage,
    TrainingMetrics,
    create_curriculum_schedule
)

from .integration_testing import (
    UnifiedOptimizationSystem,
    OptimizationConfig,
    TrainingResult,
    OptimizationTestSuite,
    OptimizationExample,
    main as run_optimization_demo
)

__all__ = [
    # Attention optimizations
    'FlashAttention',
    'LinearAttention', 
    'SparseAttention',
    'MultiQueryAttention',
    'create_attention_mechanism',
    'compare_attention_complexity',
    
    # Advanced optimizers
    'LAMBOptimizer',
    'AdaFactorOptimizer',
    'AdamW8Bit',
    'AdaptiveLRScheduler',
    'create_optimizer',
    'get_memory_efficient_optimizer',
    
    # Gradient strategies
    'GradientClipper',
    'GradientAccumulator',
    'GradientCheckpointing',
    'MemoryEfficientBackprop',
    'create_gradient_optimizer',
    'benchmark_gradient_strategies',
    
    # Hardware optimizations
    'CUDAMemoryManager',
    'MixedPrecisionTrainer',
    'CustomCudaKernels',
    'MemoryPool',
    'get_optimal_config',
    'benchmark_hardware_optimizations',
    
    # Parallel training
    'TensorParallel',
    'PipelineParallel', 
    'ExpertParallel',
    'ModelParallelManager',
    'auto_select_parallel_strategy',
    'benchmark_parallel_strategies',
    
    # Memory management
    'MemoryProfiler',
    'MemGradientCheckpointing',
    'DynamicMemoryAllocator',
    'EfficientTrainingOrchestrator',
    'auto_configure_memory_management',
    
    # Dynamic training
    'CurriculumLearning',
    'ProgressiveModelExpansion',
    'AdaptiveBatchSizing',
    'DynamicTrainingOrchestrator',
    'DifficultyLevel',
    'CurriculumStage',
    'TrainingMetrics',
    'create_curriculum_schedule',
    
    # Integration
    'UnifiedOptimizationSystem',
    'OptimizationConfig',
    'TrainingResult',
    'OptimizationTestSuite',
    'OptimizationExample',
    'run_optimization_demo'
]

# Version information
__version__ = "1.0.0"
__author__ = "Kilo Code Optimization Framework"