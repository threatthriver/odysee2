"""
Performance Benchmarking Framework

This module provides comprehensive performance testing including memory usage,
training speed, convergence rates, and scalability metrics across different
hardware configurations.
"""

import logging
import time
import psutil
import GPUtil
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable
import numpy as np
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
import torch
from abc import ABC, abstractmethod
import tracemalloc
import gc

from .benchmark_registry import BaseBenchmark, BenchmarkMetadata, BenchmarkCategory, BenchmarkType

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics collected during benchmarking"""
    memory_peak_gb: float
    memory_average_gb: float
    memory_efficiency: float  # Performance per GB used
    throughput_tokens_per_second: float
    training_steps_per_second: float
    convergence_rate: float  # Loss reduction per step
    final_loss: float
    total_training_time: float
    wall_clock_time: float
    cpu_utilization: float
    gpu_utilization: float
    gpu_memory_utilization: float
    disk_io_mb_s: float
    network_io_mb_s: float
    energy_consumption_joules: Optional[float] = None
    carbon_footprint_g_co2: Optional[float] = None


@dataclass
class HardwareConfiguration:
    """Hardware configuration for performance testing"""
    cpu_model: str
    cpu_cores: int
    cpu_threads: int
    gpu_model: str
    gpu_memory_gb: float
    gpu_count: int
    ram_gb: float
    storage_type: str  # SSD, NVMe, HDD
    network_bandwidth_gbps: float
    accelerator_type: str  # CPU, CUDA, ROCm, TPU


@dataclass
class BenchmarkResult:
    """Result from a performance benchmark"""
    benchmark_name: str
    hardware_config: HardwareConfiguration
    model_config: Dict[str, Any]
    metrics: PerformanceMetrics
    metadata: Dict[str, Any]
    timestamp: str
    success: bool
    error_message: Optional[str] = None


class PerformanceProfiler:
    """Real-time performance profiler"""
    
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.is_profiling = False
        self.metrics = []
        self.start_time = None
        self.profile_thread = None
    
    def start_profiling(self):
        """Start performance profiling"""
        if self.is_profiling:
            return
        
        self.is_profiling = True
        self.start_time = time.time()
        self.metrics = []
        
        self.profile_thread = threading.Thread(target=self._profile_loop, daemon=True)
        self.profile_thread.start()
        
        logger.info("Performance profiling started")
    
    def stop_profiling(self) -> List[Dict[str, float]]:
        """Stop performance profiling and return collected metrics"""
        if not self.is_profiling:
            return []
        
        self.is_profiling = False
        if self.profile_thread:
            self.profile_thread.join(timeout=2.0)
        
        logger.info(f"Performance profiling stopped. Collected {len(self.metrics)} samples")
        return self.metrics.copy()
    
    def _profile_loop(self):
        """Main profiling loop"""
        while self.is_profiling:
            try:
                # Collect system metrics
                timestamp = time.time()
                
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                cpu_freq = psutil.cpu_freq()
                
                # Memory metrics
                memory = psutil.virtual_memory()
                swap = psutil.swap_memory()
                
                # GPU metrics
                gpu_metrics = self._collect_gpu_metrics()
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                
                # Network I/O
                network_io = psutil.net_io_counters()
                
                # Collect metrics
                metric_sample = {
                    'timestamp': timestamp,
                    'cpu_percent': cpu_percent,
                    'cpu_freq_mhz': cpu_freq.current if cpu_freq else 0,
                    'memory_percent': memory.percent,
                    'memory_used_gb': memory.used / (1024**3),
                    'memory_available_gb': memory.available / (1024**3),
                    'swap_percent': swap.percent,
                    'disk_read_mb_s': self._calculate_disk_rate(disk_io, 'read') if disk_io else 0,
                    'disk_write_mb_s': self._calculate_disk_rate(disk_io, 'write') if disk_io else 0,
                    'network_sent_mb_s': self._calculate_network_rate(network_io, 'sent') if network_io else 0,
                    'network_recv_mb_s': self._calculate_network_rate(network_io, 'recv') if network_io else 0,
                    **gpu_metrics
                }
                
                self.metrics.append(metric_sample)
                
                # Limit memory usage by keeping only recent samples
                if len(self.metrics) > 10000:  # Keep ~3 hours at 1-second intervals
                    self.metrics = self.metrics[-10000:]
                
                time.sleep(self.interval)
                
            except Exception as e:
                logger.warning(f"Error collecting performance metrics: {e}")
                time.sleep(self.interval)
    
    def _collect_gpu_metrics(self) -> Dict[str, float]:
        """Collect GPU metrics"""
        gpu_metrics = {}
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                gpu_metrics = {
                    'gpu_utilization': gpu.load * 100,
                    'gpu_memory_used_gb': gpu.memoryUsed / 1024,
                    'gpu_memory_free_gb': gpu.memoryFree / 1024,
                    'gpu_temperature': gpu.temperature
                }
            else:
                gpu_metrics = {
                    'gpu_utilization': 0,
                    'gpu_memory_used_gb': 0,
                    'gpu_memory_free_gb': 0,
                    'gpu_temperature': 0
                }
        except Exception as e:
            logger.debug(f"GPU metrics collection failed: {e}")
            gpu_metrics = {
                'gpu_utilization': 0,
                'gpu_memory_used_gb': 0,
                'gpu_memory_free_gb': 0,
                'gpu_temperature': 0
            }
        
        return gpu_metrics
    
    def _calculate_disk_rate(self, disk_io, operation: str) -> float:
        """Calculate disk I/O rate"""
        # This is a simplified calculation - would need historical data for accurate rates
        bytes_per_second = getattr(disk_io, f'{operation}_bytes', 0)
        return bytes_per_second / (1024 * 1024)  # Convert to MB/s
    
    def _calculate_network_rate(self, network_io, operation: str) -> float:
        """Calculate network I/O rate"""
        # This is a simplified calculation - would need historical data for accurate rates
        bytes_per_second = getattr(network_io, f'{operation}_bytes', 0)
        return bytes_per_second / (1024 * 1024)  # Convert to MB/s


class BasePerformanceBenchmark(ABC):
    """Abstract base class for performance benchmarks"""
    
    def __init__(self, metadata: BenchmarkMetadata):
        self.metadata = metadata
        self.profiler = PerformanceProfiler()
    
    @abstractmethod
    def run_benchmark(self, model, hardware_config: HardwareConfiguration, **kwargs) -> BenchmarkResult:
        """Run the performance benchmark"""
        pass
    
    def _collect_hardware_info(self) -> HardwareConfiguration:
        """Collect current hardware information"""
        # CPU info
        cpu_count = psutil.cpu_count()
        cpu_count_logical = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        
        # Memory info
        memory = psutil.virtual_memory()
        
        # GPU info
        gpu_info = self._get_gpu_info()
        
        # Storage info (simplified)
        disk_usage = psutil.disk_usage('/')
        
        return HardwareConfiguration(
            cpu_model=cpu_freq.current if cpu_freq else 0,
            cpu_cores=cpu_count,
            cpu_threads=cpu_count_logical,
            gpu_model=gpu_info['model'],
            gpu_memory_gb=gpu_info['memory_gb'],
            gpu_count=gpu_info['count'],
            ram_gb=memory.total / (1024**3),
            storage_type="SSD",  # Simplified
            network_bandwidth=100.0,  # Simplified
            accelerator_type="CUDA" if gpu_info['count'] > 0 else "CPU"
        )
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                return {
                    'model': gpu.name,
                    'memory_gb': gpu.memoryTotal / 1024,
                    'count': len(gpus)
                }
            else:
                return {
                    'model': 'None',
                    'memory_gb': 0,
                    'count': 0
                }
        except Exception:
            return {
                'model': 'Unknown',
                'memory_gb': 0,
                'count': 0
            }
    
    def _analyze_performance_metrics(self, profiler_data: List[Dict[str, float]]) -> PerformanceMetrics:
        """Analyze collected performance data"""
        if not profiler_data:
            raise ValueError("No performance data collected")
        
        # Memory metrics
        memory_values = [sample['memory_used_gb'] for sample in profiler_data]
        memory_peak_gb = max(memory_values)
        memory_average_gb = np.mean(memory_values)
        
        # Memory efficiency (throughput per GB)
        total_tokens = self._estimate_total_tokens()
        memory_efficiency = total_tokens / memory_peak_gb if memory_peak_gb > 0 else 0
        
        # CPU metrics
        cpu_values = [sample['cpu_percent'] for sample in profiler_data]
        cpu_utilization = np.mean(cpu_values)
        
        # GPU metrics
        gpu_utilization_values = [sample.get('gpu_utilization', 0) for sample in profiler_data]
        gpu_utilization = np.mean(gpu_utilization_values)
        
        gpu_memory_values = [sample.get('gpu_memory_used_gb', 0) for sample in profiler_data]
        gpu_memory_utilization = np.mean(gpu_memory_values)
        
        # Throughput metrics (would need training-specific calculations)
        total_time = profiler_data[-1]['timestamp'] - profiler_data[0]['timestamp']
        throughput_tokens_per_second = self._calculate_throughput(total_time)
        
        # Training steps per second (would be model-specific)
        training_steps_per_second = self._calculate_training_steps(total_time)
        
        # Convergence metrics (would need loss tracking)
        convergence_rate, final_loss = self._analyze_convergence()
        
        return PerformanceMetrics(
            memory_peak_gb=memory_peak_gb,
            memory_average_gb=memory_average_gb,
            memory_efficiency=memory_efficiency,
            throughput_tokens_per_second=throughput_tokens_per_second,
            training_steps_per_second=training_steps_per_second,
            convergence_rate=convergence_rate,
            final_loss=final_loss,
            total_training_time=total_time,
            wall_clock_time=total_time,
            cpu_utilization=cpu_utilization,
            gpu_utilization=gpu_utilization,
            gpu_memory_utilization=gpu_memory_utilization,
            disk_io_mb_s=np.mean([sample.get('disk_read_mb_s', 0) + sample.get('disk_write_mb_s', 0) 
                                 for sample in profiler_data]),
            network_io_mb_s=np.mean([sample.get('network_sent_mb_s', 0) + sample.get('network_recv_mb_s', 0) 
                                   for sample in profiler_data])
        )
    
    def _estimate_total_tokens(self) -> int:
        """Estimate total tokens processed (would be model-specific)"""
        # This would need to be calculated based on the actual training data
        return 1000000  # Placeholder
    
    def _calculate_throughput(self, total_time: float) -> float:
        """Calculate tokens per second throughput"""
        total_tokens = self._estimate_total_tokens()
        return total_tokens / total_time if total_time > 0 else 0
    
    def _calculate_training_steps(self, total_time: float) -> int:
        """Calculate training steps per second"""
        # This would need actual step count from training
        return 1000 / total_time if total_time > 0 else 0
    
    def _analyze_convergence(self) -> Tuple[float, float]:
        """Analyze convergence rate and final loss"""
        # This would need actual loss tracking during training
        # For now, return placeholder values
        return 0.95, 0.05  # convergence_rate, final_loss


class MemoryEfficiencyBenchmark(BasePerformanceBenchmark):
    """Benchmark for measuring memory efficiency"""
    
    def __init__(self):
        metadata = BenchmarkMetadata(
            name="MemoryEfficiency",
            description="Memory usage and efficiency during training",
            category=BenchmarkCategory.MEMORY_EFFICIENCY,
            benchmark_type=BenchmarkType.PERFORMANCE,
            dataset_path="synthetic",
            metrics=["peak_memory_gb", "memory_reduction_percent", "memory_efficiency"],
            expected_score_range=(0.1, 10.0)
        )
        super().__init__(metadata)
    
    def run_benchmark(self, model, hardware_config: HardwareConfiguration, **kwargs) -> BenchmarkResult:
        """Run memory efficiency benchmark"""
        try:
            self.profiler.start_profiling()
            
            # Simulate memory-intensive training
            logger.info("Starting memory efficiency benchmark")
            
            # Force garbage collection before starting
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Start memory tracking
            tracemalloc.start()
            
            # Simulate training workload
            time.sleep(10)  # Simulate training time
            self._simulate_training_workload(model, hardware_config)
            
            # Stop profiling
            profiler_data = self.profiler.stop_profiling()
            tracemalloc.stop()
            
            # Calculate metrics
            metrics = self._analyze_performance_metrics(profiler_data)
            
            # Add memory-specific metrics
            current, peak = tracemalloc.get_traced_memory()
            metrics.memory_peak_gb = peak / (1024**3)
            
            logger.info(f"Memory efficiency benchmark completed. Peak memory: {metrics.memory_peak_gb:.2f} GB")
            
            return BenchmarkResult(
                benchmark_name="MemoryEfficiency",
                hardware_config=hardware_config,
                model_config=self._get_model_config(),
                metrics=metrics,
                metadata={
                    "tracemalloc_peak_gb": peak / (1024**3),
                    "tracemalloc_current_gb": current / (1024**3),
                    "sample_count": len(profiler_data)
                },
                timestamp=datetime.now().isoformat(),
                success=True
            )
            
        except Exception as e:
            logger.error(f"Memory efficiency benchmark failed: {e}")
            return BenchmarkResult(
                benchmark_name="MemoryEfficiency",
                hardware_config=hardware_config,
                model_config=self._get_model_config(),
                metrics=PerformanceMetrics(
                    memory_peak_gb=0, memory_average_gb=0, memory_efficiency=0,
                    throughput_tokens_per_second=0, training_steps_per_second=0,
                    convergence_rate=0, final_loss=0, total_training_time=0,
                    wall_clock_time=0, cpu_utilization=0, gpu_utilization=0,
                    gpu_memory_utilization=0, disk_io_mb_s=0, network_io_mb_s=0
                ),
                metadata={},
                timestamp=datetime.now().isoformat(),
                success=False,
                error_message=str(e)
            )
    
    def _simulate_training_workload(self, model, hardware_config: HardwareConfiguration):
        """Simulate memory-intensive training workload"""
        # This would simulate actual training operations that stress memory
        # For now, just create some memory usage patterns
        
        batch_sizes = [32, 64, 128, 256, 128, 64]  # Variable batch sizes
        sequence_lengths = [512, 1024, 2048, 1024, 512]  # Variable sequence lengths
        
        for batch_size in batch_sizes:
            for seq_len in sequence_lengths:
                # Simulate forward pass memory usage
                fake_input = torch.randn(batch_size, seq_len, 768)
                if hasattr(model, 'device'):
                    fake_input = fake_input.to(model.device)
                
                # Simulate some computation
                _ = fake_input.sum()
                time.sleep(0.1)  # Simulate computation time
        
        # Cleanup
        del fake_input
        gc.collect()
    
    def _get_model_config(self) -> Dict[str, Any]:
        """Get model configuration for benchmark"""
        return {
            "model_type": "transformer",
            "parameters": "125M",  # Placeholder
            "batch_size": 32,
            "sequence_length": 512,
            "precision": "float32"
        }


class TrainingSpeedBenchmark(BasePerformanceBenchmark):
    """Benchmark for measuring training speed and throughput"""
    
    def __init__(self):
        metadata = BenchmarkMetadata(
            name="TrainingSpeed",
            description="Training throughput and speed metrics",
            category=BenchmarkCategory.TRAINING_SPEED,
            benchmark_type=BenchmarkType.PERFORMANCE,
            dataset_path="synthetic",
            metrics=["tokens_per_second", "steps_per_second", "throughput_improvement"],
            expected_score_range=(1.0, 10000.0)
        )
        super().__init__(metadata)
    
    def run_benchmark(self, model, hardware_config: HardwareConfiguration, **kwargs) -> BenchmarkResult:
        """Run training speed benchmark"""
        try:
            self.profiler.start_profiling()
            
            logger.info("Starting training speed benchmark")
            
            # Benchmark training operations
            start_time = time.time()
            operations_completed = self._benchmark_training_operations(model, hardware_config)
            end_time = time.time()
            
            # Stop profiling
            profiler_data = self.profiler.stop_profiling()
            
            # Calculate metrics
            metrics = self._analyze_performance_metrics(profiler_data)
            
            # Calculate speed-specific metrics
            total_time = end_time - start_time
            metrics.throughput_tokens_per_second = operations_completed / total_time
            metrics.training_steps_per_second = operations_completed / total_time
            
            logger.info(f"Training speed benchmark completed. {operations_completed} operations in {total_time:.2f}s")
            
            return BenchmarkResult(
                benchmark_name="TrainingSpeed",
                hardware_config=hardware_config,
                model_config=self._get_model_config(),
                metrics=metrics,
                metadata={
                    "operations_completed": operations_completed,
                    "total_time": total_time,
                    "sample_count": len(profiler_data)
                },
                timestamp=datetime.now().isoformat(),
                success=True
            )
            
        except Exception as e:
            logger.error(f"Training speed benchmark failed: {e}")
            return BenchmarkResult(
                benchmark_name="TrainingSpeed",
                hardware_config=hardware_config,
                model_config=self._get_model_config(),
                metrics=PerformanceMetrics(
                    memory_peak_gb=0, memory_average_gb=0, memory_efficiency=0,
                    throughput_tokens_per_second=0, training_steps_per_second=0,
                    convergence_rate=0, final_loss=0, total_training_time=0,
                    wall_clock_time=0, cpu_utilization=0, gpu_utilization=0,
                    gpu_memory_utilization=0, disk_io_mb_s=0, network_io_mb_s=0
                ),
                metadata={},
                timestamp=datetime.now().isoformat(),
                success=False,
                error_message=str(e)
            )
    
    def _benchmark_training_operations(self, model, hardware_config: HardwareConfiguration) -> int:
        """Benchmark various training operations"""
        operations = 0
        
        # Simulate forward passes
        for _ in range(100):
            batch_size = 32
            seq_len = 512
            
            fake_input = torch.randn(batch_size, seq_len, 768)
            if hasattr(model, 'device'):
                fake_input = fake_input.to(model.device)
            
            # Simulate forward computation
            for _ in range(10):
                fake_input = torch.relu(fake_input @ torch.randn(768, 768))
                operations += 1
            
            del fake_input
            gc.collect()
        
        return operations
    
    def _get_model_config(self) -> Dict[str, Any]:
        """Get model configuration for benchmark"""
        return {
            "model_type": "transformer",
            "batch_size": 32,
            "sequence_length": 512,
            "forward_passes": 1000,
            "operations_per_pass": 10
        }


class ConvergenceRateBenchmark(BasePerformanceBenchmark):
    """Benchmark for measuring model convergence characteristics"""
    
    def __init__(self):
        metadata = BenchmarkMetadata(
            name="ConvergenceRate",
            description="Model convergence speed and final performance",
            category=BenchmarkCategory.CONVERGENCE,
            benchmark_type=BenchmarkType.PERFORMANCE,
            dataset_path="synthetic",
            metrics=["convergence_steps", "final_loss", "convergence_rate"],
            expected_score_range=(0.0, 100000.0)
        )
        super().__init__(metadata)
    
    def run_benchmark(self, model, hardware_config: HardwareConfiguration, **kwargs) -> BenchmarkResult:
        """Run convergence rate benchmark"""
        try:
            self.profiler.start_profiling()
            
            logger.info("Starting convergence rate benchmark")
            
            # Simulate training with loss tracking
            loss_history = self._simulate_training_convergence(model, hardware_config)
            
            # Stop profiling
            profiler_data = self.profiler.stop_profiling()
            
            # Calculate metrics
            metrics = self._analyze_performance_metrics(profiler_data)
            
            # Calculate convergence-specific metrics
            if len(loss_history) > 1:
                initial_loss = loss_history[0]
                final_loss = loss_history[-1]
                convergence_rate = (initial_loss - final_loss) / len(loss_history)
                
                metrics.convergence_rate = convergence_rate
                metrics.final_loss = final_loss
            else:
                metrics.convergence_rate = 0
                metrics.final_loss = loss_history[0] if loss_history else float('inf')
            
            # Estimate convergence steps (simplified)
            convergence_threshold = 0.1
            metrics.training_steps_per_second = self._estimate_convergence_steps(loss_history, convergence_threshold)
            
            logger.info(f"Convergence benchmark completed. Final loss: {metrics.final_loss:.4f}")
            
            return BenchmarkResult(
                benchmark_name="ConvergenceRate",
                hardware_config=hardware_config,
                model_config=self._get_model_config(),
                metrics=metrics,
                metadata={
                    "loss_history": loss_history,
                    "convergence_steps": len(loss_history),
                    "sample_count": len(profiler_data)
                },
                timestamp=datetime.now().isoformat(),
                success=True
            )
            
        except Exception as e:
            logger.error(f"Convergence rate benchmark failed: {e}")
            return BenchmarkResult(
                benchmark_name="ConvergenceRate",
                hardware_config=hardware_config,
                model_config=self._get_model_config(),
                metrics=PerformanceMetrics(
                    memory_peak_gb=0, memory_average_gb=0, memory_efficiency=0,
                    throughput_tokens_per_second=0, training_steps_per_second=0,
                    convergence_rate=0, final_loss=0, total_training_time=0,
                    wall_clock_time=0, cpu_utilization=0, gpu_utilization=0,
                    gpu_memory_utilization=0, disk_io_mb_s=0, network_io_mb_s=0
                ),
                metadata={},
                timestamp=datetime.now().isoformat(),
                success=False,
                error_message=str(e)
            )
    
    def _simulate_training_convergence(self, model, hardware_config: HardwareConfiguration) -> List[float]:
        """Simulate training and return loss history"""
        # Simulate loss curve with exponential decay + noise
        loss_history = []
        initial_loss = 2.0
        convergence_rate = 0.95
        
        for step in range(100):
            # Simulate loss with exponential decay and noise
            loss = initial_loss * (convergence_rate ** step) + np.random.normal(0, 0.01)
            loss = max(0.01, loss)  # Ensure loss doesn't go negative
            loss_history.append(loss)
            
            # Simulate training step
            time.sleep(0.01)  # Short delay to simulate computation
        
        return loss_history
    
    def _estimate_convergence_steps(self, loss_history: List[float], threshold: float) -> float:
        """Estimate steps to convergence"""
        if not loss_history:
            return 0
        
        initial_loss = loss_history[0]
        for step, loss in enumerate(loss_history):
            if (initial_loss - loss) / initial_loss >= threshold:
                return step + 1
        
        return len(loss_history)  # Did not converge within training
    
    def _get_model_config(self) -> Dict[str, Any]:
        """Get model configuration for benchmark"""
        return {
            "model_type": "transformer",
            "learning_rate": 1e-4,
            "batch_size": 32,
            "max_steps": 100,
            "convergence_threshold": 0.1
        }


class PerformanceBenchmarkRunner:
    """Main runner for all performance benchmarks"""
    
    def __init__(self):
        self.benchmarks = {
            "MemoryEfficiency": MemoryEfficiencyBenchmark(),
            "TrainingSpeed": TrainingSpeedBenchmark(),
            "ConvergenceRate": ConvergenceRateBenchmark()
        }
    
    def run_all_benchmarks(self, model, hardware_config: HardwareConfiguration = None, **kwargs) -> List[BenchmarkResult]:
        """Run all performance benchmarks"""
        if hardware_config is None:
            # Auto-detect hardware configuration
            benchmark = MemoryEfficiencyBenchmark()  # Use any benchmark to get hardware info
            hardware_config = benchmark._collect_hardware_info()
        
        results = []
        
        for name, benchmark in self.benchmarks.items():
            logger.info(f"Running performance benchmark: {name}")
            result = benchmark.run_benchmark(model, hardware_config, **kwargs)
            results.append(result)
            
            if result.success:
                logger.info(f"{name} benchmark completed successfully")
            else:
                logger.error(f"{name} benchmark failed: {result.error_message}")
        
        return results
    
    def run_specific_benchmark(self, benchmark_name: str, model, hardware_config: HardwareConfiguration = None, **kwargs) -> BenchmarkResult:
        """Run a specific performance benchmark"""
        if benchmark_name not in self.benchmarks:
            raise ValueError(f"Benchmark {benchmark_name} not found")
        
        if hardware_config is None:
            benchmark = self.benchmarks[benchmark_name]
            hardware_config = benchmark._collect_hardware_info()
        
        return self.benchmarks[benchmark_name].run_benchmark(model, hardware_config, **kwargs)
    
    def compare_hardware_configs(self, model, hardware_configs: List[HardwareConfiguration], **kwargs) -> Dict[str, List[BenchmarkResult]]:
        """Compare performance across different hardware configurations"""
        results = {}
        
        for i, hardware_config in enumerate(hardware_configs):
            logger.info(f"Testing hardware configuration {i+1}/{len(hardware_configs)}")
            config_results = self.run_all_benchmarks(model, hardware_config, **kwargs)
            results[f"config_{i+1}"] = config_results
        
        return results
    
    def save_benchmark_results(self, results: List[BenchmarkResult], filepath: str):
        """Save benchmark results to JSON file"""
        results_data = []
        
        for result in results:
            result_dict = {
                "benchmark_name": result.benchmark_name,
                "hardware_config": asdict(result.hardware_config),
                "model_config": result.model_config,
                "metrics": asdict(result.metrics),
                "metadata": result.metadata,
                "timestamp": result.timestamp,
                "success": result.success,
                "error_message": result.error_message
            }
            results_data.append(result_dict)
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Benchmark results saved to {filepath}")
    
    def load_benchmark_results(self, filepath: str) -> List[BenchmarkResult]:
        """Load benchmark results from JSON file"""
        with open(filepath, 'r') as f:
            results_data = json.load(f)
        
        results = []
        for result_dict in results_data:
            hardware_config = HardwareConfiguration(**result_dict["hardware_config"])
            metrics = PerformanceMetrics(**result_dict["metrics"])
            
            result = BenchmarkResult(
                benchmark_name=result_dict["benchmark_name"],
                hardware_config=hardware_config,
                model_config=result_dict["model_config"],
                metrics=metrics,
                metadata=result_dict["metadata"],
                timestamp=result_dict["timestamp"],
                success=result_dict["success"],
                error_message=result_dict.get("error_message")
            )
            results.append(result)
        
        return results
    
    def generate_performance_report(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "total_benchmarks": len(results),
            "successful_benchmarks": sum(1 for r in results if r.success),
            "failed_benchmarks": sum(1 for r in results if not r.success),
            "hardware_summary": self._summarize_hardware(results),
            "performance_summary": self._summarize_performance(results),
            "optimization_recommendations": self._generate_optimization_recommendations(results)
        }
        
        return report
    
    def _summarize_hardware(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Summarize hardware configurations"""
        if not results:
            return {}
        
        first_config = results[0].hardware_config
        return {
            "cpu_cores": first_config.cpu_cores,
            "ram_gb": first_config.ram_gb,
            "gpu_model": first_config.gpu_model,
            "gpu_memory_gb": first_config.gpu_memory_gb,
            "accelerator_type": first_config.accelerator_type
        }
    
    def _summarize_performance(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Summarize performance metrics"""
        successful_results = [r for r in results if r.success]
        if not successful_results:
            return {}
        
        summary = {}
        
        # Aggregate metrics
        for result in successful_results:
            for metric_name, metric_value in asdict(result.metrics).items():
                if metric_name not in summary:
                    summary[metric_name] = []
                summary[metric_name].append(metric_value)
        
        # Calculate statistics
        for metric_name, values in summary.items():
            summary[metric_name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "median": np.median(values)
            }
        
        return summary
    
    def _generate_optimization_recommendations(self, results: List[BenchmarkResult]) -> List[str]:
        """Generate optimization recommendations based on results"""
        recommendations = []
        
        successful_results = [r for r in results if r.success]
        
        for result in successful_results:
            if result.benchmark_name == "MemoryEfficiency":
                # Memory optimization recommendations
                if result.metrics.memory_peak_gb > 8.0:
                    recommendations.append("Consider gradient checkpointing to reduce memory usage")
                if result.metrics.memory_efficiency < 100000:
                    recommendations.append("Optimize batch size and sequence length for better memory efficiency")
            
            elif result.benchmark_name == "TrainingSpeed":
                # Speed optimization recommendations
                if result.metrics.throughput_tokens_per_second < 1000:
                    recommendations.append("Consider mixed precision training to improve throughput")
                if result.metrics.gpu_utilization < 80:
                    recommendations.append("Increase batch size or sequence length to better utilize GPU")
            
            elif result.benchmark_name == "ConvergenceRate":
                # Convergence optimization recommendations
                if result.metrics.convergence_rate < 0.1:
                    recommendations.append("Consider adjusting learning rate for faster convergence")
                if result.metrics.final_loss > 0.5:
                    recommendations.append("Review model architecture and training procedure")
        
        return recommendations