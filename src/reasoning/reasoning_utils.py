"""
Reasoning Utilities and Validation System

This module provides utilities for reasoning validation, consistency checking,
and quality assessment including:
- Logical consistency validation
- Reasoning step coherence checking  
- Mathematical reasoning validation
- Argument validity assessment
- Reasoning quality scoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np
from dataclasses import dataclass
from collections import defaultdict, deque
import re
import math
from .chain_of_thought import ReasoningStep, ReasoningChain


@dataclass
class ValidationResult:
    """Result of reasoning step validation."""
    is_valid: bool
    confidence_score: float
    validation_type: str
    issues: List[str]
    suggestions: List[str]
    metadata: Dict[str, Any]


@dataclass
class ConsistencyReport:
    """Report on reasoning chain consistency."""
    overall_consistency: float
    logical_consistency: float
    temporal_consistency: float
    factual_consistency: float
    structural_consistency: float
    inconsistencies: List[Dict[str, Any]]
    recommendations: List[str]


class LogicalConsistencyValidator(nn.Module):
    """
    Validates logical consistency of reasoning steps using neural networks.
    
    Checks for logical fallacies, contradictions, and validity of reasoning patterns.
    """
    
    def __init__(self, embedding_dim: int = 256, **kwargs):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Logical pattern recognition
        self.logical_pattern_classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Contradiction detection
        self.contradiction_detector = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Logical fallacy detection
        self.fallacy_classifier = nn.Linear(embedding_dim, 10)  # 10 types of fallacies
        
        # Rule-based logical patterns
        self.logical_operators = {
            'implies': self._check_implication,
            'and': self._check_conjunction,
            'or': self._check_disjunction,
            'not': self._check_negation,
            'forall': self._check_universal_quantification,
            'exists': self._check_existential_quantification
        }
        
    def validate_logical_consistency(
        self,
        step1: ReasoningStep,
        step2: ReasoningStep,
        context: torch.Tensor
    ) -> Tuple[bool, float, List[str]]:
        """
        Validate logical consistency between two reasoning steps.
        
        Args:
            step1: First reasoning step
            step2: Second reasoning step
            context: Context tensor
            
        Returns:
            Tuple of (is_consistent, confidence, issues)
        """
        issues = []
        
        # Neural consistency check
        combined_embedding = torch.cat([step1.content, step2.content], dim=-1)
        consistency_score = self.logical_pattern_classifier(combined_embedding).item()
        
        # Contradiction detection
        contradiction_score = self.contradiction_detector(combined_embedding).item()
        
        # Rule-based logical checks
        rule_based_issues = self._apply_logical_rules(step1, step2)
        issues.extend(rule_based_issues)
        
        # Determine consistency
        is_consistent = (
            consistency_score > 0.7 and 
            contradiction_score < 0.3 and 
            len(rule_based_issues) == 0
        )
        
        confidence = (consistency_score + (1 - contradiction_score)) / 2
        
        return is_consistent, confidence, issues
        
    def _check_implication(self, premise: str, conclusion: str) -> bool:
        """Check implication validity (if P then Q)."""
        # Simple heuristic: if conclusion mentions premise, it might be valid
        premise_words = set(premise.lower().split())
        conclusion_words = set(conclusion.lower().split())
        
        # Check for logical connective words
        if any(word in conclusion.lower() for word in ['therefore', 'thus', 'hence', 'so']):
            return True
        return True  # Default to valid for now
        
    def _check_conjunction(self, statements: List[str]) -> bool:
        """Check conjunction validity (P and Q)."""
        # Ensure all statements are compatible
        return len(statements) > 0
        
    def _check_disjunction(self, statements: List[str]) -> bool:
        """Check disjunction validity (P or Q)."""
        # Ensure at least one statement is true
        return len(statements) > 0
        
    def _check_negation(self, statement: str) -> bool:
        """Check negation consistency."""
        # Check for double negatives or contradictory negations
        return 'not not' not in statement.lower()
        
    def _check_universal_quantification(self, condition: str, domain: str) -> bool:
        """Check universal quantification validity."""
        # Ensure domain is specified
        return len(domain.strip()) > 0
        
    def _check_existential_quantification(self, condition: str, domain: str) -> bool:
        """Check existential quantification validity."""
        # Ensure domain is non-empty
        return len(domain.strip()) > 0
        
    def _apply_logical_rules(self, step1: ReasoningStep, step2: ReasoningStep) -> List[str]:
        """Apply rule-based logical consistency checks."""
        issues = []
        
        # Check for temporal inconsistencies
        if hasattr(step1, 'timestamp') and hasattr(step2, 'timestamp'):
            if step1.timestamp > step2.timestamp and step1.step_id < step2.step_id:
                issues.append("Temporal inconsistency: later step depends on earlier step")
                
        # Check for circular dependencies
        if step2.step_id in step1.dependencies:
            issues.append("Circular dependency detected")
            
        # Check confidence consistency
        if abs(step1.confidence - step2.confidence) > 0.5:
            issues.append("Sudden confidence drop might indicate inconsistency")
            
        return issues


class MathematicalReasoningValidator(nn.Module):
    """
    Validates mathematical reasoning steps including arithmetic, algebra, and logic.
    
    Provides numerical validation and mathematical consistency checking.
    """
    
    def __init__(self, embedding_dim: int = 256, **kwargs):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Mathematical pattern recognition
        self.math_pattern_classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim // 4),
            nn.ReLU(),
            nn.Linear(embedding_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Equation validation
        self.equation_validator = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
        
    def validate_mathematical_step(
        self,
        step: ReasoningStep,
        previous_steps: List[ReasoningStep],
        context: torch.Tensor
    ) -> Tuple[bool, float, List[str]]:
        """
        Validate a mathematical reasoning step.
        
        Args:
            step: Mathematical reasoning step to validate
            previous_steps: Previous steps for dependency checking
            context: Context tensor
            
        Returns:
            Tuple of (is_valid, confidence, issues)
        """
        issues = []
        
        # Extract mathematical content (placeholder - would need actual math parsing)
        math_content = self._extract_mathematical_content(step.content)
        
        # Validate mathematical consistency
        consistency_score = self._validate_math_consistency(step, previous_steps)
        
        # Check for common mathematical errors
        error_patterns = self._check_math_errors(step, previous_steps)
        issues.extend(error_patterns)
        
        # Validate equations if present
        if math_content.get('equations'):
            equation_valid = self._validate_equations(math_content['equations'])
            if not equation_valid:
                issues.append("Invalid mathematical equation detected")
                
        is_valid = consistency_score > 0.8 and len(issues) == 0
        confidence = consistency_score
        
        return is_valid, confidence, issues
        
    def _extract_mathematical_content(self, content: torch.Tensor) -> Dict[str, Any]:
        """Extract mathematical elements from content tensor."""
        # This would implement actual mathematical content extraction
        # For now, return placeholder structure
        return {
            'equations': [],
            'operations': [],
            'numbers': [],
            'variables': [],
            'functions': []
        }
        
    def _validate_math_consistency(
        self,
        step: ReasoningStep,
        previous_steps: List[ReasoningStep]
    ) -> float:
        """Validate mathematical consistency with previous steps."""
        if not previous_steps:
            return 0.8  # Default confidence for first step
            
        # Check if mathematical operations are consistent
        math_scores = []
        for prev_step in previous_steps:
            if prev_step.reasoning_type == 'mathematical':
                # Compute similarity between mathematical expressions
                similarity = F.cosine_similarity(
                    step.content.unsqueeze(0), 
                    prev_step.content.unsqueeze(0)
                ).item()
                math_scores.append(similarity)
                
        return np.mean(math_scores) if math_scores else 0.8
        
    def _check_math_errors(
        self,
        step: ReasoningStep,
        previous_steps: List[ReasoningStep]
    ) -> List[str]:
        """Check for common mathematical reasoning errors."""
        issues = []
        
        # Check for division by zero (would need actual content parsing)
        # Check for invalid operations
        # Check for unit inconsistencies
        # Check for calculation errors
        
        # Placeholder checks
        if step.confidence < 0.3:
            issues.append("Low confidence in mathematical step")
            
        return issues
        
    def _validate_equations(self, equations: List[str]) -> bool:
        """Validate mathematical equations."""
        # This would implement actual equation validation
        return True  # Placeholder


class FactualConsistencyValidator(nn.Module):
    """
    Validates factual consistency and real-world knowledge accuracy.
    
    Checks against factual knowledge bases and common sense reasoning.
    """
    
    def __init__(self, embedding_dim: int = 256, **kwargs):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Factual consistency classifier
        self.factual_classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Knowledge grounding
        self.knowledge_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Common sense reasoning
        self.common_sense_checker = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def validate_factual_consistency(
        self,
        step: ReasoningStep,
        knowledge_base: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None
    ) -> Tuple[bool, float, List[str]]:
        """
        Validate factual consistency of reasoning step.
        
        Args:
            step: Reasoning step to validate
            knowledge_base: External knowledge base tensor
            context: Context tensor
            
        Returns:
            Tuple of (is_consistent, confidence, issues)
        """
        issues = []
        
        # Check against knowledge base
        if knowledge_base is not None:
            factual_score = self._check_against_knowledge(step, knowledge_base)
        else:
            factual_score = 0.8  # Default score without knowledge base
            
        # Common sense checks
        common_sense_score = self._check_common_sense(step)
        
        # Overall factual consistency
        factual_consistency = (factual_score + common_sense_score) / 2
        
        # Determine validity
        is_consistent = factual_consistency > 0.7
        
        if not is_consistent:
            issues.append("Potential factual inconsistency detected")
            
        confidence = factual_consistency
        
        return is_consistent, confidence, issues
        
    def _check_against_knowledge(
        self,
        step: ReasoningStep,
        knowledge_base: torch.Tensor
    ) -> float:
        """Check reasoning step against knowledge base."""
        # Compute similarity with knowledge base
        similarities = F.cosine_similarity(
            step.content.unsqueeze(0),
            knowledge_base
        )
        
        return torch.mean(similarities).item()
        
    def _check_common_sense(self, step: ReasoningStep) -> float:
        """Apply common sense reasoning checks."""
        # Use common sense checker
        consistency_score = self.common_sense_checker(step.content).item()
        return consistency_score


class ReasoningValidator:
    """
    Main reasoning validator that combines multiple validation approaches.
    
    Integrates logical, mathematical, and factual validation methods
    to provide comprehensive reasoning quality assessment.
    """
    
    def __init__(self, reasoning_dim: int = 256, **kwargs):
        self.reasoning_dim = reasoning_dim
        
        # Initialize validators
        self.logical_validator = LogicalConsistencyValidator(reasoning_dim)
        self.mathematical_validator = MathematicalReasoningValidator(reasoning_dim)
        self.factual_validator = FactualConsistencyValidator(reasoning_dim)
        
        # Validation weights
        self.validation_weights = {
            'logical': 0.4,
            'mathematical': 0.3,
            'factual': 0.3
        }
        
    def validate_step(
        self,
        step: ReasoningStep,
        context: torch.Tensor,
        previous_steps: List[ReasoningStep],
        knowledge_base: Optional[torch.Tensor] = None
    ) -> Tuple[bool, float]:
        """
        Validate a reasoning step using all validation methods.
        
        Args:
            step: ReasoningStep to validate
            context: Context tensor
            previous_steps: Previous reasoning steps
            knowledge_base: Optional knowledge base for factual validation
            
        Returns:
            Tuple of (is_valid, confidence_score)
        """
        validation_results = {}
        
        # Logical validation
        if previous_steps:
            logical_valid, logical_confidence, logical_issues = self.logical_validator.validate_logical_consistency(
                step, previous_steps[-1], context
            )
            validation_results['logical'] = {
                'valid': logical_valid,
                'confidence': logical_confidence,
                'issues': logical_issues
            }
        else:
            validation_results['logical'] = {
                'valid': True,
                'confidence': 0.8,
                'issues': []
            }
            
        # Mathematical validation
        if step.reasoning_type == 'mathematical':
            math_valid, math_confidence, math_issues = self.mathematical_validator.validate_mathematical_step(
                step, previous_steps, context
            )
            validation_results['mathematical'] = {
                'valid': math_valid,
                'confidence': math_confidence,
                'issues': math_issues
            }
        else:
            validation_results['mathematical'] = {
                'valid': True,
                'confidence': 0.8,
                'issues': []
            }
            
        # Factual validation
        factual_valid, factual_confidence, factual_issues = self.factual_validator.validate_factual_consistency(
            step, knowledge_base, context
        )
        validation_results['factual'] = {
            'valid': factual_valid,
            'confidence': factual_confidence,
            'issues': factual_issues
        }
        
        # Combine validation results
        overall_confidence = sum(
            self.validation_weights[key] * results['confidence']
            for key, results in validation_results.items()
        )
        
        # Consider step valid if all validations pass and confidence is high
        all_valid = all(results['valid'] for results in validation_results.values())
        is_valid = all_valid and overall_confidence > 0.6
        
        # Collect all issues
        all_issues = []
        for results in validation_results.values():
            all_issues.extend(results['issues'])
            
        return is_valid, overall_confidence
        
    def validate_reasoning_chain(
        self,
        chain: ReasoningChain,
        knowledge_base: Optional[torch.Tensor] = None
    ) -> ConsistencyReport:
        """
        Validate entire reasoning chain for consistency.
        
        Args:
            chain: ReasoningChain to validate
            knowledge_base: Optional knowledge base
            
        Returns:
            ConsistencyReport with detailed analysis
        """
        if not chain.steps:
            return ConsistencyReport(
                overall_consistency=1.0,
                logical_consistency=1.0,
                temporal_consistency=1.0,
                factual_consistency=1.0,
                structural_consistency=1.0,
                inconsistencies=[],
                recommendations=[]
            )
            
        # Validate individual steps
        step_validations = []
        for i, step in enumerate(chain.steps):
            context = chain.question.unsqueeze(0) if chain.question is not None else torch.zeros(1, self.reasoning_dim)
            previous_steps = chain.steps[:i] if i > 0 else []
            
            is_valid, confidence = self.validate_step(
                step, context, previous_steps, knowledge_base
            )
            
            step_validations.append({
                'step_id': step.step_id,
                'is_valid': is_valid,
                'confidence': confidence,
                'step': step
            })
            
        # Calculate consistency metrics
        logical_consistency = np.mean([v['confidence'] for v in step_validations])
        temporal_consistency = self._calculate_temporal_consistency(chain.steps)
        factual_consistency = np.mean([v['confidence'] for v in step_validations])  # Simplified
        structural_consistency = self._calculate_structural_consistency(chain.steps)
        
        overall_consistency = (
            logical_consistency + temporal_consistency + 
            factual_consistency + structural_consistency
        ) / 4
        
        # Find inconsistencies
        inconsistencies = []
        for validation in step_validations:
            if not validation['is_valid']:
                inconsistencies.append({
                    'step_id': validation['step_id'],
                    'type': 'validation_failure',
                    'confidence': validation['confidence'],
                    'details': 'Step failed validation checks'
                })
                
        # Generate recommendations
        recommendations = self._generate_recommendations(step_validations, inconsistencies)
        
        return ConsistencyReport(
            overall_consistency=overall_consistency,
            logical_consistency=logical_consistency,
            temporal_consistency=temporal_consistency,
            factual_consistency=factual_consistency,
            structural_consistency=structural_consistency,
            inconsistencies=inconsistencies,
            recommendations=recommendations
        )
        
    def _calculate_temporal_consistency(self, steps: List[ReasoningStep]) -> float:
        """Calculate temporal consistency of reasoning steps."""
        if len(steps) < 2:
            return 1.0
            
        # Check for logical temporal ordering
        temporal_scores = []
        for i in range(1, len(steps)):
            current_step = steps[i]
            previous_step = steps[i-1]
            
            # Check if current step logically follows from previous
            similarity = F.cosine_similarity(
                current_step.content.unsqueeze(0),
                previous_step.content.unsqueeze(0)
            ).item()
            temporal_scores.append(similarity)
            
        return np.mean(temporal_scores) if temporal_scores else 1.0
        
    def _calculate_structural_consistency(self, steps: List[ReasoningStep]) -> float:
        """Calculate structural consistency of reasoning chain."""
        if len(steps) < 2:
            return 1.0
            
        # Check dependency graph consistency
        structural_scores = []
        
        for step in steps:
            # Check if dependencies are valid
            for dep_id in step.dependencies:
                if dep_id >= step.step_id:
                    structural_scores.append(0.0)  # Invalid dependency
                else:
                    structural_scores.append(1.0)  # Valid dependency
                    
            # Check if next_steps are reciprocated
            for next_id in step.next_steps:
                if step.step_id in steps[next_id].dependencies:
                    structural_scores.append(1.0)  # Reciprocated
                else:
                    structural_scores.append(0.5)  # Not reciprocated
                    
        return np.mean(structural_scores) if structural_scores else 1.0
        
    def _generate_recommendations(
        self,
        step_validations: List[Dict],
        inconsistencies: List[Dict]
    ) -> List[str]:
        """Generate recommendations for improving reasoning chain."""
        recommendations = []
        
        # Check for low confidence steps
        low_confidence_steps = [
            v for v in step_validations 
            if v['confidence'] < 0.6
        ]
        
        if low_confidence_steps:
            recommendations.append(
                f"Consider revisiting {len(low_confidence_steps)} steps with low confidence"
            )
            
        # Check for validation failures
        if inconsistencies:
            recommendations.append(
                f"Address {len(inconsistencies)} validation failures in the reasoning chain"
            )
            
        # Check for reasoning type diversity
        reasoning_types = set(v['step'].reasoning_type for v in step_validations)
        if len(reasoning_types) < 2:
            recommendations.append("Consider using multiple reasoning types for robustness")
            
        # Check for step count
        if len(step_validations) < 3:
            recommendations.append("Reasoning chain might be too short for complex problems")
        elif len(step_validations) > 15:
            recommendations.append("Consider consolidating reasoning steps to improve clarity")
            
        return recommendations