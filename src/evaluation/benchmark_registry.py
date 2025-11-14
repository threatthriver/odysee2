"""
Benchmark Registry for managing standardized evaluation benchmarks
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import json
import os

logger = logging.getLogger(__name__)


class BenchmarkType(Enum):
    """Types of benchmarks available"""
    REASONING = "reasoning"
    PERFORMANCE = "performance"
    SOTA_COMPARISON = "sota_comparison"
    ROBUSTNESS = "robustness"
    RESPONSIBLE_AI = "responsible_ai"
    MULTIMODAL = "multimodal"


class BenchmarkCategory(Enum):
    """Categories within each benchmark type"""
    CHAIN_OF_THOUGHT = "chain_of_thought"
    LOGICAL_REASONING = "logical_reasoning"
    MATHEMATICAL = "mathematical"
    COMMONSENSE = "commonsense"
    CAUSAL_REASONING = "causal_reasoning"
    MEMORY_EFFICIENCY = "memory_efficiency"
    TRAINING_SPEED = "training_speed"
    CONVERGENCE = "convergence"
    ADVERSARIAL = "adversarial"
    BIAS_DETECTION = "bias_detection"
    TOXICITY = "toxicity"
    FAIRNESS = "fairness"


@dataclass
class BenchmarkMetadata:
    """Metadata for a benchmark"""
    name: str
    description: str
    category: BenchmarkCategory
    benchmark_type: BenchmarkType
    dataset_path: str
    metrics: List[str]
    expected_score_range: tuple
    citation: Optional[str] = None
    paper_url: Optional[str] = None
    huggingface_id: Optional[str] = None


class BaseBenchmark(ABC):
    """Abstract base class for all benchmarks"""
    
    def __init__(self, metadata: BenchmarkMetadata):
        self.metadata = metadata
        self.is_loaded = False
        self.dataset = None
    
    @abstractmethod
    def load_dataset(self) -> bool:
        """Load the benchmark dataset"""
        pass
    
    @abstractmethod
    def evaluate_model(self, model, **kwargs) -> Dict[str, float]:
        """Evaluate a model on this benchmark"""
        pass
    
    @abstractmethod
    def validate_setup(self) -> bool:
        """Validate that the benchmark is properly set up"""
        pass


class BenchmarkRegistry:
    """Registry for managing and accessing all available benchmarks"""
    
    def __init__(self, cache_dir: str = "./benchmark_cache"):
        self.benchmarks: Dict[str, BaseBenchmark] = {}
        self.metadata: Dict[str, BenchmarkMetadata] = {}
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._initialize_builtin_benchmarks()
    
    def _initialize_builtin_benchmarks(self):
        """Initialize built-in benchmark definitions"""
        
        # Reasoning benchmarks
        reasoning_benchmarks = [
            BenchmarkMetadata(
                name="MMLU",
                description="Massive Multitask Language Understanding",
                category=BenchmarkCategory.LOGICAL_REASONING,
                benchmark_type=BenchmarkType.REASONING,
                dataset_path="huggingface://hails/mmlu_no_dev",
                metrics=["accuracy", "macro_f1"],
                expected_score_range=(0.25, 0.95),
                citation="@article{hendryckstest2021measuring,",
                paper_url="https://arxiv.org/abs/2009.03300",
                huggingface_id="hails/mmlu_no_dev"
            ),
            BenchmarkMetadata(
                name="GSM8K",
                description="Grade School Math 8K",
                category=BenchmarkCategory.MATHEMATICAL,
                benchmark_type=BenchmarkType.REASONING,
                dataset_path="huggingface://gsm8k",
                metrics=["exact_match", "accuracy"],
                expected_score_range=(0.0, 1.0),
                citation="@article{cobbe2021training,",
                paper_url="https://arxiv.org/abs/2110.14168",
                huggingface_id="gsm8k"
            ),
            BenchmarkMetadata(
                name="HellaSwag",
                description="Harder and Better Language Understanding Test",
                category=BenchmarkCategory.LOGICAL_REASONING,
                benchmark_type=BenchmarkType.REASONING,
                dataset_path="huggingface://hellaswag",
                metrics=["accuracy"],
                expected_score_range=(0.25, 0.95),
                citation="@article{zellers2019hellaswag,",
                paper_url="https://arxiv.org/abs/1905.07830",
                huggingface_id="hellaswag"
            ),
            BenchmarkMetadata(
                name="ARC-Challenge",
                description="AI2 Reasoning Challenge - Challenge Set",
                category=BenchmarkCategory.LOGICAL_REASONING,
                benchmark_type=BenchmarkType.REASONING,
                dataset_path="huggingface://ai2_arc",
                metrics=["accuracy"],
                expected_score_range=(0.0, 1.0),
                citation="@article{clark2019think,",
                paper_url="https://arxiv.org/abs/1903.11100",
                huggingface_id="ai2_arc"
            ),
            BenchmarkMetadata(
                name="CommonsenseQA",
                description="Commonsense Question Answering",
                category=BenchmarkCategory.COMMONSENSE,
                benchmark_type=BenchmarkType.REASONING,
                dataset_path="huggingface://commonsense_qa",
                metrics=["accuracy"],
                expected_score_range=(0.5, 0.95),
                citation="@article{talmor2019commonsenseqa,",
                paper_url="https://arxiv.org/abs/1811.00937",
                huggingface_id="commonsense_qa"
            )
        ]
        
        # Performance benchmarks
        performance_benchmarks = [
            BenchmarkMetadata(
                name="MemoryEfficiency",
                description="Memory usage during training",
                category=BenchmarkCategory.MEMORY_EFFICIENCY,
                benchmark_type=BenchmarkType.PERFORMANCE,
                dataset_path="synthetic",
                metrics=["peak_memory_gb", "memory_reduction_percent"],
                expected_score_range=(0.1, 10.0)
            ),
            BenchmarkMetadata(
                name="TrainingSpeed",
                description="Training throughput and speed",
                category=BenchmarkCategory.TRAINING_SPEED,
                benchmark_type=BenchmarkType.PERFORMANCE,
                dataset_path="synthetic",
                metrics=["tokens_per_second", "steps_per_second"],
                expected_score_range=(1.0, 10000.0)
            ),
            BenchmarkMetadata(
                name="ConvergenceRate",
                description="Model convergence characteristics",
                category=BenchmarkCategory.CONVERGENCE,
                benchmark_type=BenchmarkType.PERFORMANCE,
                dataset_path="synthetic",
                metrics=["convergence_steps", "final_loss", "convergence_rate"],
                expected_score_range=(0.0, 100000.0)
            )
        ]
        
        # Robustness benchmarks
        robustness_benchmarks = [
            BenchmarkMetadata(
                name="AdversarialRobustness",
                description="Model performance under adversarial attacks",
                category=BenchmarkCategory.ADVERSARIAL,
                benchmark_type=BenchmarkType.ROBUSTNESS,
                dataset_path="huggingface://adversarial_qa",
                metrics=["robustness_score", "accuracy_drop"],
                expected_score_range=(0.0, 1.0)
            ),
            BenchmarkMetadata(
                name="DistributionalShift",
                description="Performance under distributional shift",
                category=BenchmarkCategory.ADVERSARIAL,
                benchmark_type=BenchmarkType.ROBUSTNESS,
                dataset_path="synthetic",
                metrics=["shift_resilience", "performance_retention"],
                expected_score_range=(0.0, 1.0)
            )
        ]
        
        # Responsible AI benchmarks
        responsible_ai_benchmarks = [
            BenchmarkMetadata(
                name="BiasDetection",
                description="Bias detection and measurement",
                category=BenchmarkCategory.BIAS_DETECTION,
                benchmark_type=BenchmarkType.RESPONSIBLE_AI,
                dataset_path="huggingface://bias_benchmarks",
                metrics=["bias_score", "demographic_parity", "equalized_odds"],
                expected_score_range=(0.0, 1.0)
            ),
            BenchmarkMetadata(
                name="ToxicityAssessment",
                description="Toxicity and harmful content detection",
                category=BenchmarkCategory.TOXICITY,
                benchmark_type=BenchmarkType.RESPONSIBLE_AI,
                dataset_path="huggingface://toxicity_benchmarks",
                metrics=["toxicity_score", "false_positive_rate", "false_negative_rate"],
                expected_score_range=(0.0, 1.0)
            ),
            BenchmarkMetadata(
                name="FairnessEvaluation",
                description="Fairness across different groups",
                category=BenchmarkCategory.FAIRNESS,
                benchmark_type=BenchmarkType.RESPONSIBLE_AI,
                dataset_path="huggingface://fairness_benchmarks",
                metrics=["fairness_score", "group_accuracy_gap", "calibration_gap"],
                expected_score_range=(0.0, 1.0)
            )
        ]
        
        # Register all benchmarks
        for benchmark in reasoning_benchmarks + performance_benchmarks + robustness_benchmarks + responsible_ai_benchmarks:
            self.register_benchmark(benchmark)
    
    def register_benchmark(self, metadata: BenchmarkMetadata):
        """Register a new benchmark"""
        self.metadata[metadata.name] = metadata
        logger.info(f"Registered benchmark: {metadata.name}")
    
    def get_benchmark_metadata(self, name: str) -> Optional[BenchmarkMetadata]:
        """Get metadata for a specific benchmark"""
        return self.metadata.get(name)
    
    def list_benchmarks_by_type(self, benchmark_type: BenchmarkType) -> List[BenchmarkMetadata]:
        """List all benchmarks of a specific type"""
        return [meta for meta in self.metadata.values() if meta.benchmark_type == benchmark_type]
    
    def list_benchmarks_by_category(self, category: BenchmarkCategory) -> List[BenchmarkMetadata]:
        """List all benchmarks in a specific category"""
        return [meta for meta in self.metadata.values() if meta.category == category]
    
    def get_available_metrics(self, name: str) -> List[str]:
        """Get available metrics for a benchmark"""
        metadata = self.get_benchmark_metadata(name)
        return metadata.metrics if metadata else []
    
    def save_registry(self, filepath: str):
        """Save the registry to a JSON file"""
        registry_data = {
            name: {
                'name': meta.name,
                'description': meta.description,
                'category': meta.category.value,
                'benchmark_type': meta.benchmark_type.value,
                'dataset_path': meta.dataset_path,
                'metrics': meta.metrics,
                'expected_score_range': meta.expected_score_range,
                'citation': meta.citation,
                'paper_url': meta.paper_url,
                'huggingface_id': meta.huggingface_id
            }
            for name, meta in self.metadata.items()
        }
        
        with open(filepath, 'w') as f:
            json.dump(registry_data, f, indent=2)
        
        logger.info(f"Registry saved to {filepath}")
    
    def load_registry(self, filepath: str):
        """Load the registry from a JSON file"""
        if not os.path.exists(filepath):
            logger.warning(f"Registry file {filepath} not found")
            return
        
        with open(filepath, 'r') as f:
            registry_data = json.load(f)
        
        for name, data in registry_data.items():
            metadata = BenchmarkMetadata(
                name=data['name'],
                description=data['description'],
                category=BenchmarkCategory(data['category']),
                benchmark_type=BenchmarkType(data['benchmark_type']),
                dataset_path=data['dataset_path'],
                metrics=data['metrics'],
                expected_score_range=tuple(data['expected_score_range']),
                citation=data.get('citation'),
                paper_url=data.get('paper_url'),
                huggingface_id=data.get('huggingface_id')
            )
            self.metadata[name] = metadata
        
        logger.info(f"Loaded {len(self.metadata)} benchmarks from {filepath}")
    
    def validate_benchmark_setup(self, name: str) -> bool:
        """Validate that a benchmark is properly set up"""
        metadata = self.get_benchmark_metadata(name)
        if not metadata:
            logger.error(f"Benchmark {name} not found")
            return False
        
        # Check dataset path
        if metadata.dataset_path.startswith('huggingface://'):
            huggingface_id = metadata.dataset_path.replace('huggingface://', '')
            # In a real implementation, this would check if the dataset exists
            logger.info(f"Validating HuggingFace dataset: {huggingface_id}")
        elif metadata.dataset_path == "synthetic":
            logger.info(f"Using synthetic data for benchmark: {name}")
        else:
            if not os.path.exists(metadata.dataset_path):
                logger.error(f"Dataset path {metadata.dataset_path} does not exist")
                return False
        
        # Validate metrics
        if not metadata.metrics:
            logger.error(f"No metrics defined for benchmark {name}")
            return False
        
        logger.info(f"Benchmark {name} validation passed")
        return True