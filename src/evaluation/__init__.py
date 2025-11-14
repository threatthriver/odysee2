"""
Comprehensive Evaluation Framework for Language Model Training

This module provides a comprehensive evaluation framework for language model training,
including reasoning benchmarks, SOTA model comparisons, performance testing,
robustness validation, and responsible AI testing.
"""

from .benchmark_registry import BenchmarkRegistry
from .evaluation_manager import EvaluationManager
from .metrics_calculator import MetricsCalculator
from .report_generator import ReportGenerator

__version__ = "1.0.0"

__all__ = [
    "BenchmarkRegistry",
    "EvaluationManager", 
    "MetricsCalculator",
    "ReportGenerator"
]