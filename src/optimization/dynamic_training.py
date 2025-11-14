"""
Dynamic Training Strategies

This module implements dynamic training strategies including curriculum learning,
progressive model expansion, adaptive batch sizing, and intelligent learning
rate scheduling for efficient training of large language models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Union, Callable
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import time
import random
import json
import math
from contextlib import contextmanager


class DifficultyLevel(Enum):
    """Difficulty levels for curriculum learning."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    MIXED = "mixed"


@dataclass
class CurriculumStage:
    """Stage configuration for curriculum learning."""
    name: str
    difficulty_levels: List[DifficultyLevel]
    sample_proportions: Dict[DifficultyLevel, float]
    stage_duration: int  # Number of steps or epochs
    learning_rate_modifier: float
    batch_size_modifier: float
    temperature: float = 1.0  # For sampling-based curriculum


@dataclass
class TrainingMetrics:
    """Training metrics for dynamic optimization."""
    loss: float
    accuracy: float
    perplexity: float
    gradient_norm: float
    learning_rate: float
    step_time: float
    memory_usage: float
    throughput: float  # tokens per second


class CurriculumLearning:
    """
    Advanced curriculum learning system with multiple strategies.
    
    Supports difficulty-based, sampling-based, and progressive curriculum
    learning with automatic difficulty adjustment.
    """
    
    def __init__(
        self,
        curriculum_stages: List[CurriculumStage],
        strategy: str = 'progressive',
        difficulty_predictor: Optional[Callable] = None,
        auto_adjust: bool = True
    ):
        """
        Initialize curriculum learning.
        
        Args:
            curriculum_stages: List of curriculum stages
            strategy: Curriculum strategy ('progressive', 'sampling', 'adaptive')
            difficulty_predictor: Function to predict sample difficulty
            auto_adjust: Whether to automatically adjust difficulty
        """
        self.curriculum_stages = curriculum_stages
        self.strategy = strategy
        self.difficulty_predictor = difficulty_predictor
        self.auto_adjust = auto_adjust
        
        # Current state
        self.current_stage_idx = 0
        self.current_step = 0
        self.stage_step = 0
        
        # Difficulty tracking
        self.sample_difficulties = {}
        self.performance_history = defaultdict(list)
        self.difficulty_adjustments = 0
        
        # Metrics
        self.metrics_history = []
        self.curriculum_efficiency = 0.0
        
        # Statistics
        self.stats = {
            'total_samples_processed': 0,
            'curriculum_stages_completed': 0,
            'difficulty_adjustments': 0,
            'learning_acceleration': 0.0,
            'convergence_improvement': 0.0
        }
    
    def get_current_stage(self) -> CurriculumStage:
        """Get current curriculum stage."""
        if self.current_stage_idx < len(self.curriculum_stages):
            return self.curriculum_stages[self.current_stage_idx]
        else:
            # Return last stage if we've completed all
            return self.curriculum_stages[-1]
    
    def should_advance_stage(self, metrics: TrainingMetrics) -> bool:
        """
        Determine if we should advance to the next curriculum stage.
        
        Args:
            metrics: Current training metrics
            
        Returns:
            Whether to advance to next stage
        """
        current_stage = self.get_current_stage()
        
        # Check if stage duration is met
        if self.stage_step >= current_stage.stage_duration:
            return True
        
        # Performance-based advancement
        if self.auto_adjust and len(self.performance_history['loss']) >= 10:
            recent_loss = np.mean(self.performance_history['loss'][-10:])
            target_loss = self._get_stage_target_loss()
            
            if recent_loss <= target_loss:
                return True
        
        return False
    
    def _get_stage_target_loss(self) -> float:
        """Get target loss for current stage."""
        # Simple target loss progression
        stage_idx = self.current_stage_idx
        if stage_idx == 0:
            return 2.0  # Easy target
        elif stage_idx == 1:
            return 1.5  # Medium target
        elif stage_idx == 2:
            return 1.0  # Hard target
        else:
            return 0.5  # Advanced target
    
    def advance_stage(self):
        """Advance to next curriculum stage."""
        if self.current_stage_idx < len(self.curriculum_stages) - 1:
            self.current_stage_idx += 1
            self.stage_step = 0
            self.stats['curriculum_stages_completed'] += 1
            
            print(f"Advanced to curriculum stage: {self.get_current_stage().name}")
        else:
            print("All curriculum stages completed")
    
    def get_training_parameters(self) -> Dict[str, Any]:
        """
        Get current training parameters based on curriculum stage.
        
        Returns:
            Dictionary of training parameters
        """
        current_stage = self.get_current_stage()
        
        return {
            'learning_rate_modifier': current_stage.learning_rate_modifier,
            'batch_size_modifier': current_stage.batch_size_modifier,
            'difficulty_levels': current_stage.difficulty_levels,
            'sample_proportions': current_stage.sample_proportions,
            'temperature': current_stage.temperature,
            'stage_name': current_stage.name
        }
    
    def sample_difficulty_distribution(
        self,
        available_samples: List[Any],
        num_samples: int
    ) -> List[Any]:
        """
        Sample a batch based on current curriculum difficulty distribution.
        
        Args:
            available_samples: List of available samples
            num_samples: Number of samples to select
            
        Returns:
            Selected samples for training
        """
        current_stage = self.get_current_stage()
        
        if self.strategy == 'progressive':
            return self._progressive_sampling(available_samples, num_samples)
        elif self.strategy == 'sampling':
            return self._sampling_based_selection(available_samples, num_samples)
        elif self.strategy == 'adaptive':
            return self._adaptive_sampling(available_samples, num_samples)
        else:
            # Default to random sampling
            return random.sample(available_samples, min(num_samples, len(available_samples)))
    
    def _progressive_sampling(
        self,
        available_samples: List[Any],
        num_samples: int
    ) -> List[Any]:
        """Progressive difficulty sampling."""
        current_stage = self.get_current_stage()
        difficulty_levels = current_stage.difficulty_levels
        
        # Categorize samples by difficulty
        categorized_samples = defaultdict(list)
        for sample in available_samples:
            if self.difficulty_predictor:
                difficulty = self.difficulty_predictor(sample)
                categorized_samples[difficulty].append(sample)
            else:
                # Default categorization (random assignment)
                difficulty = random.choice(difficulty_levels)
                categorized_samples[difficulty].append(sample)
        
        # Sample according to proportions
        selected_samples = []
        remaining_samples = num_samples
        
        for difficulty, proportion in current_stage.sample_proportions.items():
            num_for_difficulty = int(remaining_samples * proportion)
            num_for_difficulty = min(num_for_difficulty, len(categorized_samples[difficulty]))
            
            if num_for_difficulty > 0:
                samples = random.sample(categorized_samples[difficulty], num_for_difficulty)
                selected_samples.extend(samples)
                remaining_samples -= num_for_difficulty
        
        # Fill remaining slots with any available samples
        if remaining_samples > 0:
            remaining_available = [s for s in available_samples if s not in selected_samples]
            additional_samples = random.sample(remaining_available, min(remaining_samples, len(remaining_available)))
            selected_samples.extend(additional_samples)
        
        return selected_samples[:num_samples]
    
    def _sampling_based_selection(
        self,
        available_samples: List[Any],
        num_samples: int
    ) -> List[Any]:
        """Sampling-based difficulty selection with temperature."""
        current_stage = self.get_current_stage()
        temperature = current_stage.temperature
        
        # Calculate difficulty scores for all samples
        if self.difficulty_predictor:
            difficulty_scores = []
            for sample in available_samples:
                difficulty = self.difficulty_predictor(sample)
                # Convert difficulty to score (higher difficulty = higher score)
                score = self._difficulty_to_score(difficulty)
                difficulty_scores.append(score)
            
            # Apply temperature and convert to probabilities
            difficulty_scores = np.array(difficulty_scores)
            difficulty_scores = difficulty_scores / temperature
            probabilities = F.softmax(torch.tensor(difficulty_scores), dim=0).numpy()
            
            # Sample based on probabilities
            selected_indices = np.random.choice(
                len(available_samples), 
                size=num_samples, 
                replace=False, 
                p=probabilities
            )
            return [available_samples[i] for i in selected_indices]
        else:
            # Fallback to random sampling
            return random.sample(available_samples, min(num_samples, len(available_samples)))
    
    def _adaptive_sampling(
        self,
        available_samples: List[Any],
        num_samples: int
    ) -> List[Any]:
        """Adaptive sampling based on performance."""
        current_stage = self.get_current_stage()
        
        # Analyze recent performance
        if len(self.performance_history['loss']) < 5:
            # Not enough data, fall back to progressive
            return self._progressive_sampling(available_samples, num_samples)
        
        recent_loss = np.mean(self.performance_history['loss'][-5:])
        
        # Adjust sampling strategy based on performance
        if recent_loss > 2.0:
            # Poor performance, focus on easier samples
            difficulty_focus = DifficultyLevel.EASY
            proportion = 0.7
        elif recent_loss > 1.0:
            # Moderate performance, balanced approach
            difficulty_focus = DifficultyLevel.MEDIUM
            proportion = 0.5
        else:
            # Good performance, challenge with harder samples
            difficulty_focus = DifficultyLevel.HARD
            proportion = 0.7
        
        # Create adaptive proportions
        adaptive_proportions = current_stage.sample_proportions.copy()
        adaptive_proportions[difficulty_focus] = max(
            adaptive_proportions.get(difficulty_focus, 0.0), 
            proportion
        )
        
        # Normalize proportions
        total = sum(adaptive_proportions.values())
        adaptive_proportions = {k: v/total for k, v in adaptive_proportions.items()}
        
        # Temporarily update stage proportions
        old_proportions = current_stage.sample_proportions
        current_stage.sample_proportions = adaptive_proportions
        
        selected_samples = self._progressive_sampling(available_samples, num_samples)
        
        # Restore original proportions
        current_stage.sample_proportions = old_proportions
        
        return selected_samples
    
    def _difficulty_to_score(self, difficulty: DifficultyLevel) -> float:
        """Convert difficulty to numerical score."""
        score_map = {
            DifficultyLevel.EASY: 1.0,
            DifficultyLevel.MEDIUM: 2.0,
            DifficultyLevel.HARD: 3.0,
            DifficultyLevel.MIXED: 2.5
        }
        return score_map.get(difficulty, 2.0)
    
    def update_performance(self, metrics: TrainingMetrics):
        """Update performance tracking."""
        self.performance_history['loss'].append(metrics.loss)
        self.performance_history['accuracy'].append(metrics.accuracy)
        self.performance_history['perplexity'].append(metrics.perplexity)
        self.performance_history['gradient_norm'].append(metrics.gradient_norm)
        self.performance_history['learning_rate'].append(metrics.learning_rate)
        self.performance_history['step_time'].append(metrics.step_time)
        
        # Keep only recent history
        for key in self.performance_history:
            if len(self.performance_history[key]) > 1000:
                self.performance_history[key] = self.performance_history[key][-500:]
        
        # Update metrics history
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > 10000:
            self.metrics_history = self.metrics_history[-5000:]
    
    def step(self, metrics: TrainingMetrics) -> Dict[str, Any]:
        """
        Take a curriculum learning step.
        
        Args:
            metrics: Current training metrics
            
        Returns:
            Step information
        """
        self.current_step += 1
        self.stage_step += 1
        
        # Update performance tracking
        self.update_performance(metrics)
        
        # Check if we should advance stage
        should_advance = self.should_advance_stage(metrics)
        
        step_info = {
            'advanced_stage': False,
            'current_stage': self.get_current_stage().name,
            'stage_progress': self.stage_step / self.get_current_stage().stage_duration,
            'training_parameters': self.get_training_parameters()
        }
        
        if should_advance:
            self.advance_stage()
            step_info['advanced_stage'] = True
        
        # Update statistics
        self.stats['total_samples_processed'] += 1
        
        return step_info
    
    def calculate_curriculum_efficiency(self) -> float:
        """Calculate curriculum learning efficiency."""
        if len(self.metrics_history) < 100:
            return 0.0
        
        # Calculate learning acceleration
        recent_performance = np.mean([m.loss for m in self.metrics_history[-100:]])
        baseline_performance = np.mean([m.loss for m in self.metrics_history[:100]]) if len(self.metrics_history) > 100 else recent_performance
        
        # Improvement rate
        if baseline_performance > recent_performance:
            acceleration = (baseline_performance - recent_performance) / baseline_performance
        else:
            acceleration = 0.0
        
        # Curriculum efficiency combines acceleration with curriculum progression
        curriculum_progress = self.current_stage_idx / max(len(self.curriculum_stages), 1)
        efficiency = (acceleration + curriculum_progress) / 2
        
        self.curriculum_efficiency = efficiency
        return efficiency
    
    def get_curriculum_statistics(self) -> Dict[str, Any]:
        """Get comprehensive curriculum learning statistics."""
        efficiency = self.calculate_curriculum_efficiency()
        
        return {
            **self.stats,
            'current_stage': self.get_current_stage().name,
            'curriculum_progress': self.current_stage_idx / max(len(self.curriculum_stages), 1),
            'stage_completion': self.stage_step / self.get_current_stage().stage_duration,
            'curriculum_efficiency': efficiency,
            'performance_trends': {
                'loss_trend': np.polyfit(range(len(self.performance_history['loss'][-100:])), self.performance_history['loss'][-100:], 1)[0] if len(self.performance_history['loss']) >= 100 else 0,
                'accuracy_trend': np.polyfit(range(len(self.performance_history['accuracy'][-100:])), self.performance_history['accuracy'][-100:], 1)[0] if len(self.performance_history['accuracy']) >= 100 else 0
            },
            'difficulty_distribution': self.get_current_stage().sample_proportions
        }


class ProgressiveModelExpansion:
    """
    Progressive model expansion system that gradually increases model capacity.
    
    Implements layer-wise, width-wise, and depth-wise model expansion
    strategies for efficient training of large models.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        expansion_schedule: List[Dict[str, Any]],
        expansion_strategy: str = 'layer_wise',
        expansion_threshold: float = 0.1
    ):
        """
        Initialize progressive model expansion.
        
        Args:
            base_model: Base model to expand
            expansion_schedule: Schedule for model expansions
            expansion_strategy: Strategy for expansion ('layer_wise', 'width_wise', 'depth_wise')
            expansion_threshold: Performance threshold for expansion
        """
        self.base_model = base_model
        self.expansion_schedule = expansion_schedule
        self.expansion_strategy = expansion_strategy
        self.expansion_threshold = expansion_threshold
        
        # Expansion state
        self.current_expansion_idx = 0
        self.expanded_models = []
        self.performance_history = []
        
        # Model architecture tracking
        self.model_configs = []
        self.original_config = self._extract_model_config(base_model)
        
        # Statistics
        self.stats = {
            'total_expansions': 0,
            'expansion_efficiency': 0.0,
            'parameter_growth_rate': 0.0,
            'performance_improvement': 0.0
        }
        
        # Initialize
        self.expanded_models.append(base_model)
        self.model_configs.append(self.original_config)
    
    def _extract_model_config(self, model: nn.Module) -> Dict[str, Any]:
        """Extract model configuration."""
        config = {}
        
        # Extract transformer-specific config if available
        if hasattr(model, 'config'):
            config['hidden_size'] = getattr(model.config, 'hidden_size', 768)
            config['num_layers'] = getattr(model.config, 'num_layers', 12)
            config['num_attention_heads'] = getattr(model.config, 'num_attention_heads', 12)
            config['intermediate_size'] = getattr(model.config, 'intermediate_size', 3072)
            config['vocab_size'] = getattr(model.config, 'vocab_size', 30000)
        else:
            # Extract from model parameters
            total_params = sum(p.numel() for p in model.parameters())
            config['total_parameters'] = total_params
            
            # Estimate dimensions for transformer
            if isinstance(model, nn.Transformer):
                config['hidden_size'] = getattr(model, 'd_model', 512)
                config['num_layers'] = getattr(model, 'nhead', 8) * 6  # Rough estimate
                config['num_attention_heads'] = getattr(model, 'nhead', 8)
    
        return config
    
    def should_expand(self, performance_metrics: TrainingMetrics) -> bool:
        """
        Determine if model should be expanded based on performance.
        
        Args:
            performance_metrics: Current performance metrics
            
        Returns:
            Whether to expand the model
        """
        if self.current_expansion_idx >= len(self.expansion_schedule):
            return False  # No more expansions scheduled
        
        # Performance-based expansion
        if len(self.performance_history) >= 10:
            recent_performance = np.mean([m.loss for m in self.performance_history[-10:]])
            performance_stability = np.std([m.loss for m in self.performance_history[-10:]])
            
            # Expand if performance has plateaued
            if performance_stability < self.expansion_threshold:
                return True
        
        # Scheduled expansion
        current_schedule = self.expansion_schedule[self.current_expansion_idx]
        if 'step_threshold' in current_schedule:
            # This would be checked by the training loop
            return False
        
        return False
    
    def expand_model(self) -> nn.Module:
        """
        Expand the current model according to the schedule.
        
        Returns:
            Expanded model
        """
        if self.current_expansion_idx >= len(self.expansion_schedule):
            return self.expanded_models[-1]
        
        current_schedule = self.expansion_schedule[self.current_expansion_idx]
        expansion_type = current_schedule.get('type', 'layer_wise')
        
        # Create expanded model
        if expansion_type == 'layer_wise':
            expanded_model = self._expand_layers(current_schedule)
        elif expansion_type == 'width_wise':
            expanded_model = self._expand_width(current_schedule)
        elif expansion_type == 'depth_wise':
            expanded_model = self._expand_depth(current_schedule)
        else:
            expanded_model = self._expand_custom(current_schedule)
        
        # Update state
        self.expanded_models.append(expanded_model)
        self.current_expansion_idx += 1
        
        # Extract new config
        new_config = self._extract_model_config(expanded_model)
        self.model_configs.append(new_config)
        
        # Update statistics
        self.stats['total_expansions'] += 1
        
        print(f"Model expanded to {expansion_type}: {self.model_configs[-1]}")
        
        return expanded_model
    
    def _expand_layers(self, schedule: Dict[str, Any]) -> nn.Module:
        """Expand model by adding layers."""
        expansion_factor = schedule.get('expansion_factor', 1.5)
        layer_type = schedule.get('layer_type', 'transformer')
        
        # This is a simplified implementation
        # In reality, you would clone and modify the model architecture
        
        new_model = self.expanded_models[-1]
        
        if hasattr(new_model, 'config'):
            # Expand transformer config
            original_layers = getattr(new_model.config, 'num_layers', 12)
            new_layers = int(original_layers * expansion_factor)
            new_model.config.num_layers = new_layers
            
            # Expand intermediate size
            original_intermediate = getattr(new_model.config, 'intermediate_size', 3072)
            new_model.config.intermediate_size = int(original_intermediate * expansion_factor)
        
        return new_model
    
    def _expand_width(self, schedule: Dict[str, Any]) -> nn.Module:
        """Expand model width (hidden size, attention heads)."""
        width_factor = schedule.get('width_factor', 1.25)
        
        new_model = self.expanded_models[-1]
        
        if hasattr(new_model, 'config'):
            # Expand hidden size
            original_hidden = getattr(new_model.config, 'hidden_size', 768)
            new_model.config.hidden_size = int(original_hidden * width_factor)
            
            # Expand attention heads (must divide hidden_size)
            original_heads = getattr(new_model.config, 'num_attention_heads', 12)
            new_model.config.num_attention_heads = int(original_heads * width_factor)
            
            # Ensure heads divide hidden_size
            while new_model.config.hidden_size % new_model.config.num_attention_heads != 0:
                new_model.config.num_attention_heads -= 1
        
        return new_model
    
    def _expand_depth(self, schedule: Dict[str, Any]) -> nn.Module:
        """Expand model depth (number of layers)."""
        depth_factor = schedule.get('depth_factor', 1.3)
        
        new_model = self.expanded_models[-1]
        
        if hasattr(new_model, 'config'):
            original_layers = getattr(new_model.config, 'num_layers', 12)
            new_model.config.num_layers = int(original_layers * depth_factor)
        
        return new_model
    
    def _expand_custom(self, schedule: Dict[str, Any]) -> nn.Module:
        """Custom expansion strategy."""
        # Implementation depends on specific requirements
        return self._expand_width(schedule)
    
    def get_expansion_info(self) -> Dict[str, Any]:
        """Get current expansion information."""
        current_config = self.model_configs[-1] if self.model_configs else {}
        
        return {
            'current_expansion_idx': self.current_expansion_idx,
            'total_expansions': len(self.expansion_schedule),
            'expansion_progress': self.current_expansion_idx / max(len(self.expansion_schedule), 1),
            'current_config': current_config,
            'original_config': self.original_config,
            'expansion_strategy': self.expansion_strategy,
            'parameter_growth': self._calculate_parameter_growth()
        }
    
    def _calculate_parameter_growth(self) -> float:
        """Calculate parameter growth rate."""
        if len(self.model_configs) < 2:
            return 0.0
        
        original_params = self.original_config.get('total_parameters', 0)
        current_params = self.model_configs[-1].get('total_parameters', 0)
        
        if original_params > 0:
            growth_rate = (current_params - original_params) / original_params
            self.stats['parameter_growth_rate'] = growth_rate
            return growth_rate
        
        return 0.0
    
    def update_performance(self, metrics: TrainingMetrics):
        """Update performance tracking."""
        self.performance_history.append(metrics)
        
        # Keep only recent history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-500:]
    
    def get_expansion_statistics(self) -> Dict[str, Any]:
        """Get comprehensive expansion statistics."""
        efficiency = 0.0
        
        if len(self.performance_history) >= 10:
            # Calculate performance improvement
            recent_performance = np.mean([m.loss for m in self.performance_history[-10:]])
            baseline_performance = np.mean([m.loss for m in self.performance_history[:10]]) if len(self.performance_history) >= 20 else recent_performance
            
            if baseline_performance > recent_performance:
                improvement = (baseline_performance - recent_performance) / baseline_performance
            else:
                improvement = 0.0
            
            # Calculate efficiency as improvement per expansion
            if self.stats['total_expansions'] > 0:
                efficiency = improvement / self.stats['total_expansions']
            
            self.stats['performance_improvement'] = improvement
            self.stats['expansion_efficiency'] = efficiency
        
        return {
            **self.stats,
            **self.get_expansion_info(),
            'performance_history_length': len(self.performance_history)
        }


class AdaptiveBatchSizing:
    """
    Adaptive batch sizing system that dynamically adjusts batch size
    based on memory constraints, performance metrics, and training efficiency.
    """
    
    def __init__(
        self,
        initial_batch_size: int,
        min_batch_size: int = 1,
        max_batch_size: int = 32,
        memory_budget_mb: int = 8000,
        adjustment_frequency: int = 100,
        adaptation_strategy: str = 'performance_based'
    ):
        """
        Initialize adaptive batch sizing.
        
        Args:
            initial_batch_size: Starting batch size
            min_batch_size: Minimum allowed batch size
            max_batch_size: Maximum allowed batch size
            memory_budget_mb: Memory budget in MB
            adjustment_frequency: Steps between batch size adjustments
            adaptation_strategy: Strategy for adaptation ('performance_based', 'memory_based', 'hybrid')
        """
        self.initial_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.memory_budget_mb = memory_budget_mb
        self.adjustment_frequency = adjustment_frequency
        self.adaptation_strategy = adaptation_strategy
        
        # Current state
        self.current_batch_size = initial_batch_size
        self.adjustment_history = []
        self.performance_history = deque(maxlen=1000)
        
        # Adaptation parameters
        self.performance_window = 50
        self.memory_margin = 0.1  # 10% memory margin
        self.performance_threshold = 0.05  # 5% performance change threshold
        
        # Statistics
        self.stats = {
            'total_adjustments': 0,
            'upward_adjustments': 0,
            'downward_adjustments': 0,
            'memory_efficiency': 0.0,
            'optimal_batch_size': initial_batch_size
        }
        
        # Performance tracking
        self.step_count = 0
        self.last_adjustment_step = 0
    
    def should_adjust_batch_size(self, metrics: TrainingMetrics) -> bool:
        """
        Determine if batch size should be adjusted.
        
        Args:
            metrics: Current training metrics
            
        Returns:
            Whether to adjust batch size
        """
        # Check adjustment frequency
        if self.step_count - self.last_adjustment_step < self.adjustment_frequency:
            return False
        
        # Check if we have enough performance history
        if len(self.performance_history) < self.performance_window:
            return False
        
        # Performance-based adjustment
        if self.adaptation_strategy in ['performance_based', 'hybrid']:
            if self._performance_warrants_adjustment():
                return True
        
        # Memory-based adjustment
        if self.adaptation_strategy in ['memory_based', 'hybrid']:
            if self._memory_warrants_adjustment(metrics):
                return True
        
        return False
    
    def _performance_warrants_adjustment(self) -> bool:
        """Check if performance metrics warrant batch size adjustment."""
        recent_performance = list(self.performance_history)[-self.performance_window:]
        
        # Calculate performance trend
        if len(recent_performance) < 10:
            return False
        
        losses = [m.loss for m in recent_performance]
        gradient_norms = [m.gradient_norm for m in recent_performance]
        
        # Check for performance plateau
        loss_trend = np.polyfit(range(len(losses[-20:])), losses[-20:], 1)[0]
        loss_stability = np.std(losses[-20:])
        
        # Adjust if performance has plateaued or degraded
        if abs(loss_trend) < 0.001 and loss_stability < self.performance_threshold:
            # Performance plateau - try increasing batch size for better gradient estimates
            return True
        
        # Check gradient norm for instability
        if np.mean(gradient_norms[-10:]) > 10.0:
            # High gradient norm - try decreasing batch size
            return True
        
        return False
    
    def _memory_warrants_adjustment(self, metrics: TrainingMetrics) -> bool:
        """Check if memory usage warrants batch size adjustment."""
        # This would check actual GPU memory usage
        # For now, we'll use metrics.memory_usage
        
        if metrics.memory_usage is None:
            return False
        
        memory_usage_percent = (metrics.memory_usage / self.memory_budget_mb) * 100
        
        # Reduce batch size if memory usage is too high
        if memory_usage_percent > (1 - self.memory_margin) * 100:
            return True
        
        # Increase batch size if memory usage is very low
        if memory_usage_percent < 50.0 and self.current_batch_size < self.max_batch_size:
            return True
        
        return False
    
    def adjust_batch_size(self, metrics: TrainingMetrics) -> Dict[str, Any]:
        """
        Adjust batch size based on current conditions.
        
        Args:
            metrics: Current training metrics
            
        Returns:
            Adjustment information
        """
        if not self.should_adjust_batch_size(metrics):
            return {
                'adjusted': False,
                'current_batch_size': self.current_batch_size,
                'reason': 'No adjustment warranted'
            }
        
        # Determine adjustment direction and magnitude
        if self.adaptation_strategy in ['performance_based', 'hybrid']:
            adjustment_info = self._performance_based_adjustment()
        else:
            adjustment_info = self._memory_based_adjustment(metrics)
        
        # Apply adjustment
        new_batch_size = adjustment_info['new_batch_size']
        
        # Ensure bounds
        new_batch_size = max(self.min_batch_size, min(self.max_batch_size, new_batch_size))
        
        if new_batch_size != self.current_batch_size:
            # Update state
            old_batch_size = self.current_batch_size
            self.current_batch_size = new_batch_size
            self.last_adjustment_step = self.step_count
            
            # Update statistics
            self.stats['total_adjustments'] += 1
            if new_batch_size > old_batch_size:
                self.stats['upward_adjustments'] += 1
            else:
                self.stats['downward_adjustments'] += 1
            
            # Record adjustment
            self.adjustment_history.append({
                'step': self.step_count,
                'old_batch_size': old_batch_size,
                'new_batch_size': new_batch_size,
                'reason': adjustment_info['reason'],
                'metrics': {
                    'loss': metrics.loss,
                    'gradient_norm': metrics.gradient_norm,
                    'memory_usage': metrics.memory_usage
                }
            })
            
            print(f"Batch size adjusted: {old_batch_size} → {new_batch_size} ({adjustment_info['reason']})")
        
        return {
            'adjusted': new_batch_size != adjustment_info.get('old_batch_size', self.current_batch_size),
            'old_batch_size': self.current_batch_size,
            'new_batch_size': new_batch_size,
            'reason': adjustment_info['reason'],
            'adjustment_factor': new_batch_size / max(adjustment_info.get('old_batch_size', self.current_batch_size), 1)
        }
    
    def _performance_based_adjustment(self) -> Dict[str, Any]:
        """Performance-based batch size adjustment."""
        recent_performance = list(self.performance_history)[-self.performance_window:]
        
        losses = [m.loss for m in recent_performance]
        gradient_norms = [m.gradient_norm for m in recent_performance]
        
        # Analyze recent trends
        loss_trend = np.polyfit(range(len(losses[-20:])), losses[-20:], 1)[0]
        avg_gradient_norm = np.mean(gradient_norms[-10:])
        
        # Determine adjustment
        if abs(loss_trend) < 0.001 and avg_gradient_norm < 5.0:
            # Performance plateau with stable gradients - increase batch size
            new_size = min(int(self.current_batch_size * 1.2), self.max_batch_size)
            reason = "performance_plateau"
        elif avg_gradient_norm > 10.0:
            # High gradient norm - decrease batch size
            new_size = max(int(self.current_batch_size * 0.8), self.min_batch_size)
            reason = "high_gradient_norm"
        else:
            # No clear adjustment needed
            new_size = self.current_batch_size
            reason = "no_adjustment_needed"
        
        return {
            'new_batch_size': new_size,
            'reason': reason,
            'old_batch_size': self.current_batch_size
        }
    
    def _memory_based_adjustment(self, metrics: TrainingMetrics) -> Dict[str, Any]:
        """Memory-based batch size adjustment."""
        if metrics.memory_usage is None:
            return {
                'new_batch_size': self.current_batch_size,
                'reason': 'no_memory_data',
                'old_batch_size': self.current_batch_size
            }
        
        memory_usage_percent = (metrics.memory_usage / self.memory_budget_mb) * 100
        
        if memory_usage_percent > (1 - self.memory_margin) * 100:
            # High memory usage - decrease batch size
            new_size = max(int(self.current_batch_size * 0.8), self.min_batch_size)
            reason = "high_memory_usage"
        elif memory_usage_percent < 50.0:
            # Low memory usage - increase batch size
            new_size = min(int(self.current_batch_size * 1.3), self.max_batch_size)
            reason = "low_memory_usage"
        else:
            # Memory usage is within acceptable range
            new_size = self.current_batch_size
            reason = "optimal_memory_usage"
        
        return {
            'new_batch_size': new_size,
            'reason': reason,
            'old_batch_size': self.current_batch_size
        }
    
    def step(self, metrics: TrainingMetrics) -> Dict[str, Any]:
        """
        Take a step in adaptive batch sizing.
        
        Args:
            metrics: Current training metrics
            
        Returns:
            Step information
        """
        self.step_count += 1
        
        # Update performance history
        self.performance_history.append(metrics)
        
        # Adjust batch size if needed
        adjustment_info = self.adjust_batch_size(metrics)
        
        return {
            'current_batch_size': self.current_batch_size,
            'adjustment_info': adjustment_info,
            'memory_efficiency': self._calculate_memory_efficiency(metrics),
            'performance_stability': self._calculate_performance_stability()
        }
    
    def _calculate_memory_efficiency(self, metrics: TrainingMetrics) -> float:
        """Calculate memory efficiency."""
        if metrics.memory_usage is None:
            return 0.0
        
        efficiency = 1.0 - (metrics.memory_usage / self.memory_budget_mb)
        self.stats['memory_efficiency'] = max(0, min(1, efficiency))
        
        return efficiency
    
    def _calculate_performance_stability(self) -> float:
        """Calculate performance stability."""
        if len(self.performance_history) < 10:
            return 0.0
        
        recent_losses = [m.loss for m in list(self.performance_history)[-20:]]
        stability = 1.0 / (1.0 + np.std(recent_losses))
        
        return stability
    
    def get_batch_size_statistics(self) -> Dict[str, Any]:
        """Get comprehensive batch size statistics."""
        return {
            **self.stats,
            'current_batch_size': self.current_batch_size,
            'batch_size_range': (self.min_batch_size, self.max_batch_size),
            'adjustment_frequency': self.adjustment_frequency,
            'adaptation_strategy': self.adaptation_strategy,
            'performance_history_length': len(self.performance_history),
            'recent_adjustments': self.adjustment_history[-10:] if self.adjustment_history else []
        }


class DynamicTrainingOrchestrator:
    """
    Unified orchestrator for all dynamic training strategies.
    
    Coordinates curriculum learning, progressive expansion, and adaptive
    batch sizing for optimal training efficiency.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        curriculum_learning: Optional[CurriculumLearning] = None,
        progressive_expansion: Optional[ProgressiveModelExpansion] = None,
        adaptive_batch_sizing: Optional[AdaptiveBatchSizing] = None
    ):
        """
        Initialize dynamic training orchestrator.
        
        Args:
            model: Model to train
            device: Training device
            curriculum_learning: Curriculum learning instance
            progressive_expansion: Progressive expansion instance
            adaptive_batch_sizing: Adaptive batch sizing instance
        """
        self.model = model
        self.device = device
        
        # Component instances
        self.curriculum_learning = curriculum_learning
        self.progressive_expansion = progressive_expansion
        self.adaptive_batch_sizing = adaptive_batch_sizing
        
        # Training state
        self.current_step = 0
        self.current_model = model
        
        # Performance tracking
        self.training_history = []
        self.orchestrator_efficiency = 0.0
        
        # Statistics
        self.stats = {
            'total_steps': 0,
            'curriculum_stage_changes': 0,
            'model_expansions': 0,
            'batch_size_adjustments': 0,
            'overall_efficiency': 0.0
        }
        
        # Auto-initialize if components are None
        if self.curriculum_learning is None:
            self._initialize_default_curriculum()
        
        if self.progressive_expansion is None:
            self._initialize_default_expansion()
        
        if self.adaptive_batch_sizing is None:
            self._initialize_default_batch_sizing()
    
    def _initialize_default_curriculum(self):
        """Initialize default curriculum learning."""
        stages = [
            CurriculumStage(
                name="easy",
                difficulty_levels=[DifficultyLevel.EASY],
                sample_proportions={DifficultyLevel.EASY: 1.0},
                stage_duration=1000,
                learning_rate_modifier=1.0,
                batch_size_modifier=1.0
            ),
            CurriculumStage(
                name="medium",
                difficulty_levels=[DifficultyLevel.EASY, DifficultyLevel.MEDIUM],
                sample_proportions={DifficultyLevel.EASY: 0.3, DifficultyLevel.MEDIUM: 0.7},
                stage_duration=1500,
                learning_rate_modifier=0.8,
                batch_size_modifier=1.2
            ),
            CurriculumStage(
                name="hard",
                difficulty_levels=[DifficultyLevel.MEDIUM, DifficultyLevel.HARD],
                sample_proportions={DifficultyLevel.MEDIUM: 0.2, DifficultyLevel.HARD: 0.8},
                stage_duration=2000,
                learning_rate_modifier=0.6,
                batch_size_modifier=1.5
            )
        ]
        
        self.curriculum_learning = CurriculumLearning(stages)
    
    def _initialize_default_expansion(self):
        """Initialize default progressive expansion."""
        expansion_schedule = [
            {'type': 'width_wise', 'width_factor': 1.25},
            {'type': 'depth_wise', 'depth_factor': 1.3},
            {'type': 'layer_wise', 'expansion_factor': 1.2}
        ]
        
        self.progressive_expansion = ProgressiveModelExpansion(
            self.model, expansion_schedule
        )
    
    def _initialize_default_batch_sizing(self):
        """Initialize default adaptive batch sizing."""
        self.adaptive_batch_sizing = AdaptiveBatchSizing(
            initial_batch_size=16,
            min_batch_size=4,
            max_batch_size=64,
            adjustment_frequency=200
        )
    
    @contextmanager
    def dynamic_training_step(self, metrics: TrainingMetrics):
        """
        Context manager for dynamic training step.
        
        Args:
            metrics: Current training metrics
        """
        # Pre-step adjustments
        self.current_step += 1
        
        # Update components
        if self.curriculum_learning:
            curriculum_info = self.curriculum_learning.step(metrics)
            if curriculum_info['advanced_stage']:
                self.stats['curriculum_stage_changes'] += 1
        
        if self.progressive_expansion:
            self.progressive_expansion.update_performance(metrics)
            if self.progressive_expansion.should_expand(metrics):
                self.current_model = self.progressive_expansion.expand_model()
                self.stats['model_expansions'] += 1
        
        if self.adaptive_batch_sizing:
            batch_info = self.adaptive_batch_sizing.step(metrics)
            if batch_info['adjustment_info']['adjusted']:
                self.stats['batch_size_adjustments'] += 1
        
        # Get current training parameters
        training_parameters = self._get_current_training_parameters()
        
        yield training_parameters
        
        # Post-step updates
        self.stats['total_steps'] += 1
        self.training_history.append(metrics)
        
        # Update overall efficiency
        self.orchestrator_efficiency = self._calculate_orchestrator_efficiency()
    
    def _get_current_training_parameters(self) -> Dict[str, Any]:
        """Get current training parameters from all components."""
        parameters = {
            'batch_size': 16,  # Default
            'learning_rate': 1e-3,  # Default
            'model': self.current_model
        }
        
        # Get curriculum parameters
        if self.curriculum_learning:
            curriculum_params = self.curriculum_learning.get_training_parameters()
            parameters.update({
                'learning_rate_modifier': curriculum_params['learning_rate_modifier'],
                'batch_size_modifier': curriculum_params['batch_size_modifier'],
                'curriculum_stage': curriculum_params['stage_name']
            })
        
        # Get batch size from adaptive sizing
        if self.adaptive_batch_sizing:
            parameters['batch_size'] = self.adaptive_batch_sizing.current_batch_size
        
        return parameters
    
    def _calculate_orchestrator_efficiency(self) -> float:
        """Calculate overall orchestrator efficiency."""
        if len(self.training_history) < 100:
            return 0.0
        
        # Combine efficiency metrics from all components
        curriculum_eff = (
            self.curriculum_learning.calculate_curriculum_efficiency() 
            if self.curriculum_learning else 0.0
        )
        
        expansion_eff = (
            self.progressive_expansion.get_expansion_statistics()['expansion_efficiency']
            if self.progressive_expansion else 0.0
        )
        
        # Weighted combination
        weights = [0.4, 0.3, 0.3]  # Curriculum, Expansion, Batch sizing
        efficiencies = [curriculum_eff, expansion_eff, 0.7]  # Batch sizing efficiency estimate
        
        overall_efficiency = sum(w * e for w, e in zip(weights, efficiencies))
        self.stats['overall_efficiency'] = overall_efficiency
        
        return overall_efficiency
    
    def get_dynamic_training_info(self) -> Dict[str, Any]:
        """Get comprehensive dynamic training information."""
        info = {
            'current_step': self.current_step,
            'current_model': self.current_model,
            'orchestrator_efficiency': self.orchestrator_efficiency,
            'component_status': {
                'curriculum_learning': self.curriculum_learning is not None,
                'progressive_expansion': self.progressive_expansion is not None,
                'adaptive_batch_sizing': self.adaptive_batch_sizing is not None
            }
        }
        
        # Add component-specific info
        if self.curriculum_learning:
            info['curriculum_info'] = self.curriculum_learning.get_curriculum_statistics()
        
        if self.progressive_expansion:
            info['expansion_info'] = self.progressive_expansion.get_expansion_statistics()
        
        if self.adaptive_batch_sizing:
            info['batch_size_info'] = self.adaptive_batch_sizing.get_batch_size_statistics()
        
        return info


# Utility functions for dynamic training

def create_curriculum_schedule(
    difficulty_levels: List[DifficultyLevel],
    total_steps: int,
    progression_type: str = 'linear'
) -> List[CurriculumStage]:
    """
    Create a curriculum learning schedule.
    
    Args:
        difficulty_levels: Difficulty levels to include
        total_steps: Total training steps
        progression_type: Type of progression ('linear', 'exponential', 'adaptive')
        
    Returns:
        List of curriculum stages
    """
    if progression_type == 'linear':
        steps_per_stage = total_steps // len(difficulty_levels)
        stages = []
        
        for i, difficulty in enumerate(difficulty_levels):
            stage = CurriculumStage(
                name=f"stage_{i}_{difficulty.value}",
                difficulty_levels=[difficulty],
                sample_proportions={difficulty: 1.0},
                stage_duration=steps_per_stage,
                learning_rate_modifier=1.0 - (i * 0.2),
                batch_size_modifier=1.0 + (i * 0.2)
            )
            stages.append(stage)
    
    return stages


def auto_configure_dynamic_training(
    model_size: int,
    available_memory_gb: float,
    training_steps: int,
    performance_requirements: Dict[str, float]
) -> Dict[str, Any]:
    """
    Automatically configure dynamic training strategies.
    
    Args:
        model_size: Model size in millions of parameters
        available_memory_gb: Available GPU memory in GB
        training_steps: Total training steps
        performance_requirements: Performance requirements
        
    Returns:
        Configuration for dynamic training
    """
    # Memory-based configuration
    if available_memory_gb < 8.0:
        # Limited memory - conservative approach
        config = {
            'curriculum_strategy': 'conservative',
            'expansion_schedule': [{'type': 'width_wise', 'width_factor': 1.1}],
            'batch_size_range': (4, 16),
            'adjustment_frequency': 500
        }
    elif available_memory_gb < 16.0:
        # Medium memory - moderate approach
        config = {
            'curriculum_strategy': 'progressive',
            'expansion_schedule': [
                {'type': 'width_wise', 'width_factor': 1.2},
                {'type': 'depth_wise', 'depth_factor': 1.25}
            ],
            'batch_size_range': (8, 32),
            'adjustment_frequency': 200
        }
    else:
        # High memory - aggressive approach
        config = {
            'curriculum_strategy': 'adaptive',
            'expansion_schedule': [
                {'type': 'width_wise', 'width_factor': 1.3},
                {'type': 'depth_wise', 'depth_factor': 1.4},
                {'type': 'layer_wise', 'expansion_factor': 1.5}
            ],
            'batch_size_range': (16, 64),
            'adjustment_frequency': 100
        }
    
    # Model size adjustments
    if model_size > 1000:  # 1B+ parameters
        config['expansion_schedule'].insert(0, {'type': 'width_wise', 'width_factor': 1.1})
        config['adjustment_frequency'] = max(config['adjustment_frequency'] // 2, 50)
    
    return config


def benchmark_dynamic_strategies(
    model: nn.Module,
    training_data: List[Any],
    strategies: Dict[str, Any],
    num_iterations: int = 100
) -> Dict[str, Dict]:
    """
    Benchmark different dynamic training strategies.
    
    Args:
        model: Model to benchmark
        training_data: Training data
        strategies: Strategy configurations
        num_iterations: Number of benchmark iterations
        
    Returns:
        Benchmark results
    """
    results = {}
    
    for strategy_name, config in strategies.items():
        try:
            # Create orchestrator
            orchestrator = DynamicTrainingOrchestrator(model, torch.device('cpu'), **config)
            
            # Simulate training
            start_time = time.time()
            training_metrics = []
            
            for i in range(num_iterations):
                # Simulate metrics
                metrics = TrainingMetrics(
                    loss=2.0 * math.exp(-i / 100) + random.uniform(0.1, 0.5),
                    accuracy=0.8 - 0.3 * math.exp(-i / 50) + random.uniform(-0.05, 0.05),
                    perplexity=math.exp(2.0 * math.exp(-i / 100)),
                    gradient_norm=random.uniform(0.5, 5.0),
                    learning_rate=1e-3 * math.exp(-i / 200),
                    step_time=random.uniform(0.1, 0.5),
                    memory_usage=random.uniform(1000, 4000),
                    throughput=random.uniform(100, 1000)
                )
                
                training_metrics.append(metrics)
            
            end_time = time.time()
            
            # Get final statistics
            final_info = orchestrator.get_dynamic_training_info()
            
            results[strategy_name] = {
                'training_time': end_time - start_time,
                'final_loss': training_metrics[-1].loss,
                'final_accuracy': training_metrics[-1].accuracy,
                'convergence_rate': (training_metrics[0].loss - training_metrics[-1].loss) / num_iterations,
                'strategy_efficiency': final_info.get('orchestrator_efficiency', 0),
                'configuration': config
            }
            
        except Exception as e:
            results[strategy_name] = {
                'error': str(e),
                'training_time': float('inf'),
                'final_loss': float('inf'),
                'final_accuracy': 0.0,
                'convergence_rate': 0.0,
                'strategy_efficiency': 0.0
            }
    
    return results