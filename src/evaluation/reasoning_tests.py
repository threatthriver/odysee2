"""
Reasoning Capability Tests

This module provides comprehensive testing of various reasoning capabilities including
analogical reasoning, causal reasoning, counterfactual reasoning, and multi-step reasoning.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .benchmark_registry import BaseBenchmark, BenchmarkMetadata, BenchmarkCategory, BenchmarkType

logger = logging.getLogger(__name__)


@dataclass
class ReasoningTestExample:
    """Example for reasoning capability testing"""
    stimulus: str  # Initial information/scenario
    query: str     # What we want to know
    correct_reasoning: str
    expected_answer: str
    reasoning_type: str    # analogical, causal, counterfactual, etc.
    difficulty: str        # simple, medium, complex
    cognitive_load: str    # low, medium, high
    domain: str            # abstract, concrete, scientific, etc.


class BaseReasoningTest(ABC):
    """Abstract base class for reasoning capability tests"""
    
    def __init__(self, name: str, metadata: BenchmarkMetadata):
        self.name = name
        self.metadata = metadata
        self.examples: List[ReasoningTestExample] = []
    
    @abstractmethod
    def evaluate_reasoning_quality(self, generated_reasoning: str, correct_reasoning: str, query: str) -> Dict[str, float]:
        """Evaluate the quality of generated reasoning"""
        pass
    
    @abstractmethod
    def generate_reasoning_prompt(self, example: ReasoningTestExample) -> str:
        """Generate appropriate prompt for the reasoning type"""
        pass


class AnalogicalReasoningTest(BaseReasoningTest):
    """Test for analogical reasoning capabilities"""
    
    def __init__(self):
        metadata = BenchmarkMetadata(
            name="AnalogicalReasoning",
            description="Test analogical reasoning and pattern recognition",
            category=BenchmarkCategory.LOGICAL_REASONING,
            benchmark_type=BenchmarkType.REASONING,
            dataset_path="synthetic",
            metrics=["analogical_accuracy", "pattern_recognition", "structural_similarity"],
            expected_score_range=(0.0, 1.0)
        )
        super().__init__("AnalogicalReasoning", metadata)
        self.examples = self._generate_analogical_examples()
    
    def _generate_analogical_examples(self) -> List[ReasoningTestExample]:
        """Generate analogical reasoning examples"""
        examples = []
        
        # Simple analogical reasoning
        examples.extend([
            ReasoningTestExample(
                stimulus="A doctor is to a patient as a teacher is to a ____",
                query="Fill in the blank to complete the analogy",
                correct_reasoning="The relationship is helper-to-recipient. A doctor helps patients, a teacher helps students.",
                expected_answer="student",
                reasoning_type="analogical",
                difficulty="simple",
                cognitive_load="low",
                domain="social"
            ),
            ReasoningTestExample(
                stimulus="Bird is to wing as fish is to ____",
                query="Complete the analogy based on functional similarity",
                correct_reasoning="Wings are for flying, which is the primary function. Fins are for swimming, which is the primary function of fish.",
                expected_answer="fin",
                reasoning_type="analogical",
                difficulty="simple",
                cognitive_load="low",
                domain="biological"
            )
        ])
        
        # Complex analogical reasoning
        examples.extend([
            ReasoningTestExample(
                stimulus="The way a thermostat regulates temperature by turning heating on when cold and off when warm is analogous to how the body regulates blood sugar by releasing insulin when high and glucagon when low.",
                query="What is the deep structural similarity between these two systems?",
                correct_reasoning="Both are feedback control systems that maintain homeostasis through opposite regulatory actions when values deviate from a set point.",
                expected_answer="Homeostatic feedback control systems",
                reasoning_type="analogical",
                difficulty="complex",
                cognitive_load="high",
                domain="abstract"
            ),
            ReasoningTestExample(
                stimulus="The relationship between a government and its citizens is like the relationship between a company and its customers.",
                query="Explain the structural mapping in this analogy",
                correct_reasoning="Government provides services/protection to citizens (like a company provides products/services to customers), and citizens/t customers provide support/taxes/revenue in return. Both involve provider-recipient relationships with mutual obligations.",
                expected_answer="Provider-recipient relationship with mutual obligations",
                reasoning_type="analogical",
                difficulty="medium",
                cognitive_load="medium",
                domain="social"
            )
        ])
        
        return examples
    
    def evaluate_model(self, model, **kwargs) -> Dict[str, float]:
        """Evaluate model's analogical reasoning capabilities"""
        results = []
        
        for example in self.examples:
            prompt = self.generate_reasoning_prompt(example)
            response = model.generate(prompt) if hasattr(model, 'generate') else "Placeholder response"
            
            # Extract reasoning and answer
            reasoning, answer = self._parse_response(response)
            
            # Evaluate quality
            quality_scores = self.evaluate_reasoning_quality(reasoning, example.correct_reasoning, example.query)
            quality_scores['analogical_accuracy'] = self._evaluate_analogical_accuracy(answer, example.expected_answer)
            quality_scores['pattern_recognition'] = self._evaluate_pattern_recognition(reasoning, example.correct_reasoning)
            quality_scores['structural_similarity'] = self._evaluate_structural_similarity(reasoning, example.correct_reasoning)
            
            results.append(quality_scores)
        
        # Aggregate results
        aggregated = {}
        for metric in ['analogical_accuracy', 'pattern_recognition', 'structural_similarity']:
            aggregated[metric] = np.mean([r[metric] for r in results])
        
        aggregated['overall_analogical_reasoning'] = np.mean(list(aggregated.values()))
        
        return aggregated
    
    def evaluate_reasoning_quality(self, generated_reasoning: str, correct_reasoning: str, query: str) -> Dict[str, float]:
        """Evaluate the quality of analogical reasoning"""
        return {
            'structural_mapping': self._evaluate_structural_mapping(generated_reasoning, correct_reasoning),
            'analogical_consistency': self._evaluate_analogical_consistency(generated_reasoning, correct_reasoning),
            'logical_coherence': self._evaluate_logical_coherence(generated_reasoning)
        }
    
    def _evaluate_structural_mapping(self, generated: str, correct: str) -> float:
        """Evaluate structural mapping accuracy"""
        # Look for key structural terms
        structural_terms = ['relationship', 'similar', 'function', 'structure', 'pattern', 'correspondence']
        generated_terms = sum(1 for term in structural_terms if term.lower() in generated.lower())
        correct_terms = sum(1 for term in structural_terms if term.lower() in correct.lower())
        
        if correct_terms == 0:
            return 0.0
        
        return min(generated_terms / correct_terms, 1.0)
    
    def _evaluate_analogical_consistency(self, generated: str, correct: str) -> float:
        """Evaluate consistency with analogical reasoning principles"""
        # Check for analogical reasoning indicators
        analogy_indicators = ['like', 'similar to', 'analogous', 'corresponds', 'parallel']
        indicator_count = sum(1 for indicator in analogy_indicators if indicator.lower() in generated.lower())
        
        # Check for comparison structure
        comparison_words = ['both', 'each', 'respectively', 'respectively']
        comparison_count = sum(1 for word in comparison_words if word.lower() in generated.lower())
        
        return min((indicator_count + comparison_count) / 4, 1.0)
    
    def _evaluate_logical_coherence(self, reasoning: str) -> float:
        """Evaluate logical coherence of reasoning"""
        # Look for logical connectors
        logical_connectors = ['therefore', 'because', 'since', 'thus', 'hence', 'consequently']
        connector_count = sum(1 for connector in logical_connectors if connector.lower() in reasoning.lower())
        
        # Evaluate reasoning length and structure
        sentences = reasoning.split('.')
        if len(sentences) < 2:
            return 0.3  # Needs more structure
        
        return min(connector_count / len(sentences) * 2, 1.0)
    
    def _evaluate_analogical_accuracy(self, predicted_answer: str, correct_answer: str) -> float:
        """Evaluate accuracy of analogical answer"""
        predicted = predicted_answer.lower().strip()
        correct = correct_answer.lower().strip()
        
        # Exact match
        if predicted == correct:
            return 1.0
        
        # Partial match for related concepts
        predicted_words = set(predicted.split())
        correct_words = set(correct.split())
        
        overlap = len(predicted_words.intersection(correct_words))
        if len(correct_words) == 0:
            return 0.0
        
        return overlap / len(correct_words)
    
    def _evaluate_pattern_recognition(self, generated_reasoning: str, correct_reasoning: str) -> float:
        """Evaluate pattern recognition in reasoning"""
        # Check for pattern-related terms
        pattern_terms = ['pattern', 'similar', 'structure', 'correspondence', 'mapping']
        pattern_score = sum(1 for term in pattern_terms if term.lower() in generated_reasoning.lower())
        
        return min(pattern_score / len(pattern_terms), 1.0)
    
    def _evaluate_structural_similarity(self, generated_reasoning: str, correct_reasoning: str) -> float:
        """Evaluate structural similarity between reasoning"""
        # Analyze reasoning structure
        generated_structure = self._extract_reasoning_structure(generated_reasoning)
        correct_structure = self._extract_reasoning_structure(correct_reasoning)
        
        # Compare structures
        structure_overlap = len(set(generated_structure).intersection(set(correct_structure)))
        structure_union = len(set(generated_structure).union(set(correct_structure)))
        
        if structure_union == 0:
            return 0.0
        
        return structure_overlap / structure_union
    
    def _extract_reasoning_structure(self, reasoning: str) -> List[str]:
        """Extract structural elements from reasoning"""
        # This would extract logical structure, relationships, etc.
        # For now, return key concept words
        import re
        words = re.findall(r'\b\w+\b', reasoning.lower())
        # Filter for structural words
        structural_words = [w for w in words if w in ['because', 'therefore', 'thus', 'like', 'similar', 'corresponds']]
        return structural_words
    
    def generate_reasoning_prompt(self, example: ReasoningTestExample) -> str:
        """Generate prompt for analogical reasoning"""
        return f"""
        Analyze the following analogy and provide detailed reasoning:
        
        Stimulus: {example.stimulus}
        Query: {example.query}
        
        Please provide step-by-step reasoning explaining the analogy structure and your answer.
        Format your response as: REASONING: [your reasoning] ANSWER: [your answer]
        """


class CausalReasoningTest(BaseReasoningTest):
    """Test for causal reasoning capabilities"""
    
    def __init__(self):
        metadata = BenchmarkMetadata(
            name="CausalReasoning",
            description="Test causal reasoning and cause-effect relationships",
            category=BenchmarkCategory.LOGICAL_REASONING,
            benchmark_type=BenchmarkType.REASONING,
            dataset_path="synthetic",
            metrics=["causal_accuracy", "causal_chain_length", "mechanism_understanding"],
            expected_score_range=(0.0, 1.0)
        )
        super().__init__("CausalReasoning", metadata)
        self.examples = self._generate_causal_examples()
    
    def _generate_causal_examples(self) -> List[ReasoningTestExample]:
        """Generate causal reasoning examples"""
        examples = []
        
        # Direct causation
        examples.extend([
            ReasoningTestExample(
                stimulus="When you touch a hot stove, you get burned.",
                query="What causes the burning sensation when touching a hot stove?",
                correct_reasoning="Heat from the stove transfers to your skin through conduction, causing tissue damage which activates pain receptors, resulting in the burning sensation.",
                expected_answer="Heat transfer causing tissue damage and pain receptor activation",
                reasoning_type="causal",
                difficulty="simple",
                cognitive_load="low",
                domain="physical"
            ),
            ReasoningTestExample(
                stimulus="Plant leaves turn yellow when they don't receive enough sunlight.",
                query="Why do plants need sunlight to maintain green leaves?",
                correct_reasoning="Chlorophyll in leaves requires sunlight for photosynthesis. Without sufficient light, chlorophyll breaks down and other pigments become visible, causing yellowing.",
                expected_answer="Chlorophyll breakdown due to insufficient photosynthesis",
                reasoning_type="causal",
                difficulty="medium",
                cognitive_load="medium",
                domain="biological"
            )
        ])
        
        # Complex causal chains
        examples.extend([
            ReasoningTestExample(
                stimulus="Urbanization leads to more cars, which increases air pollution, which can cause respiratory problems in residents.",
                query="Trace the complete causal chain from urbanization to health problems",
                correct_reasoning="Urbanization → increased vehicle use → higher emissions → air pollution → respiratory irritation/inflammation → health problems",
                expected_answer="Urbanization → vehicles → emissions → pollution → health issues",
                reasoning_type="causal",
                difficulty="complex",
                cognitive_load="high",
                domain="social"
            )
        ])
        
        return examples
    
    def evaluate_model(self, model, **kwargs) -> Dict[str, float]:
        """Evaluate model's causal reasoning capabilities"""
        results = []
        
        for example in self.examples:
            prompt = self.generate_reasoning_prompt(example)
            response = model.generate(prompt) if hasattr(model, 'generate') else "Placeholder response"
            
            reasoning, answer = self._parse_response(response)
            
            quality_scores = self.evaluate_reasoning_quality(reasoning, example.correct_reasoning, example.query)
            quality_scores['causal_accuracy'] = self._evaluate_causal_accuracy(answer, example.expected_answer)
            quality_scores['causal_chain_length'] = self._evaluate_causal_chain_length(reasoning, example.correct_reasoning)
            quality_scores['mechanism_understanding'] = self._evaluate_mechanism_understanding(reasoning, example.correct_reasoning)
            
            results.append(quality_scores)
        
        # Aggregate results
        aggregated = {}
        for metric in ['causal_accuracy', 'causal_chain_length', 'mechanism_understanding']:
            aggregated[metric] = np.mean([r[metric] for r in results])
        
        aggregated['overall_causal_reasoning'] = np.mean(list(aggregated.values()))
        
        return aggregated
    
    def evaluate_reasoning_quality(self, generated_reasoning: str, correct_reasoning: str, query: str) -> Dict[str, float]:
        """Evaluate the quality of causal reasoning"""
        return {
            'causal_clarity': self._evaluate_causal_clarity(generated_reasoning),
            'temporal_sequence': self._evaluate_temporal_sequence(generated_reasoning),
            'mechanism_explanation': self._evaluate_mechanism_explanation(generated_reasoning, correct_reasoning)
        }
    
    def _evaluate_causal_accuracy(self, predicted_answer: str, correct_answer: str) -> float:
        """Evaluate accuracy of causal explanation"""
        predicted = predicted_answer.lower().strip()
        correct = correct_answer.lower().strip()
        
        # Check for causal relationship terms
        causal_terms = ['cause', 'because', 'due to', 'leads to', 'results in', 'causes']
        predicted_causal = sum(1 for term in causal_terms if term in predicted)
        correct_causal = sum(1 for term in causal_terms if term in correct)
        
        if correct_causal == 0:
            return 0.0
        
        causal_accuracy = predicted_causal / correct_causal
        
        # Also check semantic similarity
        predicted_words = set(predicted.split())
        correct_words = set(correct.split())
        semantic_overlap = len(predicted_words.intersection(correct_words))
        semantic_accuracy = semantic_overlap / len(correct_words) if correct_words else 0
        
        return (causal_accuracy + semantic_accuracy) / 2
    
    def _evaluate_causal_chain_length(self, generated_reasoning: str, correct_reasoning: str) -> float:
        """Evaluate completeness of causal chain"""
        # Count causal connectors
        causal_connectors = ['because', 'leads to', 'results in', 'causes', 'therefore', 'so', 'thus']
        connector_count = sum(1 for connector in causal_connectors if connector in generated_reasoning.lower())
        
        # Reward multiple steps in reasoning
        sentences = generated_reasoning.split('.')
        step_diversity = min(len(sentences) / 3, 1.0)  # Normalize to 3 steps
        
        return min((connector_count + step_diversity) / 3, 1.0)
    
    def _evaluate_mechanism_understanding(self, generated_reasoning: str, correct_reasoning: str) -> float:
        """Evaluate understanding of causal mechanisms"""
        # Look for mechanistic explanations
        mechanism_terms = ['process', 'mechanism', 'system', 'interaction', 'reaction', 'function']
        mechanism_score = sum(1 for term in mechanism_terms if term.lower() in generated_reasoning.lower())
        
        return min(mechanism_score / 4, 1.0)
    
    def _evaluate_causal_clarity(self, reasoning: str) -> float:
        """Evaluate clarity of causal explanation"""
        # Check for clear cause-effect language
        clear_indicators = ['because', 'due to', 'causes', 'leads to', 'results in']
        clarity_score = sum(1 for indicator in clear_indicators if indicator in reasoning.lower())
        
        return min(clarity_score / 3, 1.0)
    
    def _evaluate_temporal_sequence(self, reasoning: str) -> float:
        """Evaluate temporal ordering in causal reasoning"""
        # Look for temporal indicators
        temporal_terms = ['first', 'then', 'next', 'after', 'before', 'subsequently', 'consequently']
        temporal_score = sum(1 for term in temporal_terms if term.lower() in reasoning.lower())
        
        return min(temporal_score / 3, 1.0)
    
    def _evaluate_mechanism_explanation(self, generated_reasoning: str, correct_reasoning: str) -> float:
        """Evaluate explanation of causal mechanisms"""
        # Look for mechanistic language
        mechanism_words = ['how', 'process', 'function', 'system', 'interaction']
        mechanism_score = sum(1 for word in mechanism_words if word.lower() in generated_reasoning.lower())
        
        return min(mechanism_score / 3, 1.0)
    
    def generate_reasoning_prompt(self, example: ReasoningTestExample) -> str:
        """Generate prompt for causal reasoning"""
        return f"""
        Analyze the causal relationships in the following scenario:
        
        Scenario: {example.stimulus}
        Question: {example.query}
        
        Provide a detailed causal explanation, identifying causes, effects, and mechanisms.
        Format your response as: REASONING: [causal explanation] ANSWER: [final answer]
        """


class CounterfactualReasoningTest(BaseReasoningTest):
    """Test for counterfactual reasoning capabilities"""
    
    def __init__(self):
        metadata = BenchmarkMetadata(
            name="CounterfactualReasoning",
            description="Test counterfactual and hypothetical reasoning",
            category=BenchmarkCategory.LOGICAL_REASONING,
            benchmark_type=BenchmarkType.REASONING,
            dataset_path="synthetic",
            metrics=["counterfactual_accuracy", "hypothetical_consistency", "world_modeling"],
            expected_score_range=(0.0, 1.0)
        )
        super().__init__("CounterfactualReasoning", metadata)
        self.examples = self._generate_counterfactual_examples()
    
    def _generate_counterfactual_examples(self) -> List[ReasoningTestExample]:
        """Generate counterfactual reasoning examples"""
        examples = []
        
        examples.extend([
            ReasoningTestExample(
                stimulus="If the student had studied harder, they would have passed the exam.",
                query="What would likely have happened if the student had studied harder?",
                correct_reasoning="If the student studied harder, they would have understood the material better, leading to better test preparation and higher confidence, which would result in passing the exam.",
                expected_answer="They would have passed due to better preparation and understanding",
                reasoning_type="counterfactual",
                difficulty="simple",
                cognitive_load="medium",
                domain="educational"
            ),
            ReasoningTestExample(
                stimulus="The company went bankrupt because they couldn't compete with larger rivals.",
                query="What could have prevented the company's bankruptcy?",
                correct_reasoning="The company could have differentiated their products, found a niche market, merged with competitors, or innovated to create competitive advantages that would have allowed them to compete successfully.",
                expected_answer="Differentiation, niche markets, mergers, or innovation to gain competitive advantages",
                reasoning_type="counterfactual",
                difficulty="medium",
                cognitive_load="high",
                domain="business"
            )
        ])
        
        return examples
    
    def evaluate_model(self, model, **kwargs) -> Dict[str, float]:
        """Evaluate model's counterfactual reasoning capabilities"""
        results = []
        
        for example in self.examples:
            prompt = self.generate_reasoning_prompt(example)
            response = model.generate(prompt) if hasattr(model, 'generate') else "Placeholder response"
            
            reasoning, answer = self._parse_response(response)
            
            quality_scores = self.evaluate_reasoning_quality(reasoning, example.correct_reasoning, example.query)
            quality_scores['counterfactual_accuracy'] = self._evaluate_counterfactual_accuracy(answer, example.expected_answer)
            quality_scores['hypothetical_consistency'] = self._evaluate_hypothetical_consistency(reasoning, example.correct_reasoning)
            quality_scores['world_modeling'] = self._evaluate_world_modeling(reasoning)
            
            results.append(quality_scores)
        
        # Aggregate results
        aggregated = {}
        for metric in ['counterfactual_accuracy', 'hypothetical_consistency', 'world_modeling']:
            aggregated[metric] = np.mean([r[metric] for r in results])
        
        aggregated['overall_counterfactual_reasoning'] = np.mean(list(aggregated.values()))
        
        return aggregated
    
    def evaluate_reasoning_quality(self, generated_reasoning: str, correct_reasoning: str, query: str) -> Dict[str, float]:
        """Evaluate the quality of counterfactual reasoning"""
        return {
            'hypothetical_clarity': self._evaluate_hypothetical_clarity(generated_reasoning),
            'alternative_scenarios': self._evaluate_alternative_scenarios(generated_reasoning, correct_reasoning),
            'causal_consistency': self._evaluate_causal_consistency(generated_reasoning, correct_reasoning)
        }
    
    def _evaluate_counterfactual_accuracy(self, predicted_answer: str, correct_answer: str) -> float:
        """Evaluate accuracy of counterfactual reasoning"""
        predicted = predicted_answer.lower().strip()
        correct = correct_answer.lower().strip()
        
        # Check for hypothetical language
        hypothetical_terms = ['would', 'could', 'might', 'if', 'had', 'were to']
        predicted_hypothetical = sum(1 for term in hypothetical_terms if term in predicted)
        
        # Semantic similarity
        predicted_words = set(predicted.split())
        correct_words = set(correct.split())
        semantic_overlap = len(predicted_words.intersection(correct_words))
        semantic_accuracy = semantic_overlap / len(correct_words) if correct_words else 0
        
        return semantic_accuracy
    
    def _evaluate_hypothetical_consistency(self, generated_reasoning: str, correct_reasoning: str) -> float:
        """Evaluate consistency of hypothetical scenarios"""
        # Check for conditional structure
        conditional_terms = ['if', 'were', 'would', 'could', 'might']
        conditional_score = sum(1 for term in conditional_terms if term.lower() in generated_reasoning.lower())
        
        return min(conditional_score / 4, 1.0)
    
    def _evaluate_world_modeling(self, reasoning: str) -> float:
        """Evaluate ability to model alternative worlds"""
        # Check for world-modeling indicators
        modeling_terms = ['scenario', 'situation', 'outcome', 'result', 'consequence']
        modeling_score = sum(1 for term in modeling_terms if term.lower() in reasoning.lower())
        
        return min(modeling_score / 3, 1.0)
    
    def _evaluate_hypothetical_clarity(self, reasoning: str) -> float:
        """Evaluate clarity of hypothetical statements"""
        hypothetical_words = ['if', 'were', 'would', 'could', 'might', 'suppose']
        clarity_score = sum(1 for word in hypothetical_words if word.lower() in reasoning.lower())
        
        return min(clarity_score / 3, 1.0)
    
    def _evaluate_alternative_scenarios(self, generated_reasoning: str, correct_reasoning: str) -> float:
        """Evaluate consideration of alternative scenarios"""
        # Look for multiple possibilities or alternatives
        alternative_indicators = ['alternatively', 'instead', 'other', 'another', 'different']
        alternative_score = sum(1 for indicator in alternative_indicators if indicator.lower() in generated_reasoning.lower())
        
        return min(alternative_score / 3, 1.0)
    
    def _evaluate_causal_consistency(self, generated_reasoning: str, correct_reasoning: str) -> float:
        """Evaluate causal consistency in counterfactual reasoning"""
        # Check for causal relationships in hypothetical scenarios
        causal_terms = ['lead to', 'result in', 'cause', 'because', 'due to']
        causal_score = sum(1 for term in causal_terms if term.lower() in generated_reasoning.lower())
        
        return min(causal_score / 3, 1.0)
    
    def generate_reasoning_prompt(self, example: ReasoningTestExample) -> str:
        """Generate prompt for counterfactual reasoning"""
        return f"""
        Consider this scenario and think about what could have happened differently:
        
        Scenario: {example.stimulus}
        Question: {example.query}
        
        Provide detailed counterfactual reasoning, explaining alternative possibilities and their likely outcomes.
        Format your response as: REASONING: [counterfactual analysis] ANSWER: [alternative scenario]
        """


class MultiStepReasoningTest(BaseReasoningTest):
    """Test for multi-step reasoning capabilities"""
    
    def __init__(self):
        metadata = BenchmarkMetadata(
            name="MultiStepReasoning",
            description="Test multi-step and compound reasoning",
            category=BenchmarkCategory.LOGICAL_REASONING,
            benchmark_type=BenchmarkType.REASONING,
            dataset_path="synthetic",
            metrics=["step_accuracy", "reasoning_depth", "logical_coherence"],
            expected_score_range=(0.0, 1.0)
        )
        super().__init__("MultiStepReasoning", metadata)
        self.examples = self._generate_multistep_examples()
    
    def _generate_multistep_examples(self) -> List[ReasoningTestExample]:
        """Generate multi-step reasoning examples"""
        examples = []
        
        examples.extend([
            ReasoningTestExample(
                stimulus="A company has 1000 employees. 60% work in the main office. Of those, 40% are in management. The main office has 5 floors. Each floor has equal number of management employees.",
                query="How many management employees are on each floor of the main office?",
                correct_reasoning="Step 1: Main office employees = 1000 × 0.60 = 600. Step 2: Management employees = 600 × 0.40 = 240. Step 3: Management per floor = 240 ÷ 5 = 48. Therefore, there are 48 management employees on each floor.",
                expected_answer="48 management employees per floor",
                reasoning_type="multistep",
                difficulty="medium",
                cognitive_load="medium",
                domain="mathematical"
            ),
            ReasoningTestExample(
                stimulus="A detective finds clues: 1) The victim was found in the garden. 2) Muddy footprints led from the house to the garden. 3) The back door was unlocked. 4) There were signs of forced entry at the front door. 5) Valuable items are missing.",
                query="What is the most likely sequence of events that occurred?",
                correct_reasoning="Step 1: Someone entered through the front door forcibly. Step 2: They went through the house to the back. Step 3: They went out the back door to the garden. Step 4: The confrontation or crime occurred in the garden. Step 5: They returned through the garden and left via the back door. The muddy footprints suggest they used the back route.",
                expected_answer="Front entry → house → back exit → garden crime → back exit",
                reasoning_type="multistep",
                difficulty="complex",
                cognitive_load="high",
                domain="logical"
            )
        ])
        
        return examples
    
    def evaluate_model(self, model, **kwargs) -> Dict[str, float]:
        """Evaluate model's multi-step reasoning capabilities"""
        results = []
        
        for example in self.examples:
            prompt = self.generate_reasoning_prompt(example)
            response = model.generate(prompt) if hasattr(model, 'generate') else "Placeholder response"
            
            reasoning, answer = self._parse_response(response)
            
            quality_scores = self.evaluate_reasoning_quality(reasoning, example.correct_reasoning, example.query)
            quality_scores['step_accuracy'] = self._evaluate_step_accuracy(reasoning, example.correct_reasoning)
            quality_scores['reasoning_depth'] = self._evaluate_reasoning_depth(reasoning)
            quality_scores['logical_coherence'] = self._evaluate_logical_coherence(reasoning)
            
            results.append(quality_scores)
        
        # Aggregate results
        aggregated = {}
        for metric in ['step_accuracy', 'reasoning_depth', 'logical_coherence']:
            aggregated[metric] = np.mean([r[metric] for r in results])
        
        aggregated['overall_multistep_reasoning'] = np.mean(list(aggregated.values()))
        
        return aggregated
    
    def evaluate_reasoning_quality(self, generated_reasoning: str, correct_reasoning: str, query: str) -> Dict[str, float]:
        """Evaluate the quality of multi-step reasoning"""
        return {
            'step_clarity': self._evaluate_step_clarity(generated_reasoning),
            'progression_logic': self._evaluate_progression_logic(generated_reasoning),
            'intermediate_conclusions': self._evaluate_intermediate_conclusions(generated_reasoning)
        }
    
    def _evaluate_step_accuracy(self, generated_reasoning: str, correct_reasoning: str) -> float:
        """Evaluate accuracy of reasoning steps"""
        # Count step indicators
        step_indicators = ['step', 'first', 'then', 'next', 'finally', 'therefore']
        step_score = sum(1 for indicator in step_indicators if indicator.lower() in generated_reasoning.lower())
        
        # Evaluate numerical accuracy for math problems
        if any(char.isdigit() for char in generated_reasoning):
            # Look for calculation steps
            calculation_terms = ['= ', '×', '÷', '+', '-']
            calculation_score = sum(1 for term in calculation_terms if term in generated_reasoning)
            return min((step_score + calculation_score) / 6, 1.0)
        
        return min(step_score / 4, 1.0)
    
    def _evaluate_reasoning_depth(self, reasoning: str) -> float:
        """Evaluate depth of reasoning"""
        # Count reasoning steps by sentence structure
        sentences = reasoning.split('.')
        complex_sentences = sum(1 for sentence in sentences if len(sentence.split()) > 10)
        
        # Count logical connectors
        connectors = ['therefore', 'because', 'since', 'thus', 'hence', 'consequently']
        connector_score = sum(1 for connector in connectors if connector.lower() in reasoning.lower())
        
        depth_score = (len(sentences) + complex_sentences + connector_score) / 5
        return min(depth_score, 1.0)
    
    def _evaluate_logical_coherence(self, reasoning: str) -> float:
        """Evaluate logical coherence of multi-step reasoning"""
        # Check for logical flow indicators
        flow_indicators = ['therefore', 'thus', 'hence', 'consequently', 'as a result']
        flow_score = sum(1 for indicator in flow_indicators if indicator.lower() in reasoning.lower())
        
        # Check for intermediate conclusions
        conclusion_indicators = ['therefore', 'this means', 'so', 'which shows']
        conclusion_score = sum(1 for indicator in conclusion_indicators if indicator.lower() in reasoning.lower())
        
        coherence_score = (flow_score + conclusion_score) / 4
        return min(coherence_score, 1.0)
    
    def _evaluate_step_clarity(self, reasoning: str) -> float:
        """Evaluate clarity of individual steps"""
        step_words = ['step', 'first', 'second', 'then', 'next', 'finally']
        clarity_score = sum(1 for word in step_words if word.lower() in reasoning.lower())
        
        return min(clarity_score / 4, 1.0)
    
    def _evaluate_progression_logic(self, reasoning: str) -> float:
        """Evaluate logical progression between steps"""
        progression_words = ['leads to', 'results in', 'therefore', 'thus', 'so']
        progression_score = sum(1 for word in progression_words if word.lower() in reasoning.lower())
        
        return min(progression_score / 3, 1.0)
    
    def _evaluate_intermediate_conclusions(self, reasoning: str) -> float:
        """Evaluate use of intermediate conclusions"""
        conclusion_indicators = ['therefore', 'this means', 'which shows', 'indicates']
        conclusion_score = sum(1 for indicator in conclusion_indicators if indicator.lower() in reasoning.lower())
        
        return min(conclusion_score / 3, 1.0)
    
    def generate_reasoning_prompt(self, example: ReasoningTestExample) -> str:
        """Generate prompt for multi-step reasoning"""
        return f"""
        Solve this problem using step-by-step reasoning:
        
        Problem: {example.stimulus}
        Question: {example.query}
        
        Break down your reasoning into clear steps and show your work at each step.
        Format your response as: REASONING: [step-by-step solution] ANSWER: [final answer]
        """


class ReasoningCapabilityTestSuite:
    """Suite of reasoning capability tests"""
    
    def __init__(self):
        self.tests = {
            "analogical": AnalogicalReasoningTest(),
            "causal": CausalReasoningTest(),
            "counterfactual": CounterfactualReasoningTest(),
            "multistep": MultiStepReasoningTest()
        }
    
    def run_all_tests(self, model, **kwargs) -> Dict[str, Dict[str, float]]:
        """Run all reasoning capability tests"""
        results = {}
        
        for test_name, test in self.tests.items():
            logger.info(f"Running {test_name} reasoning test")
            try:
                test_results = test.evaluate_model(model, **kwargs)
                results[test_name] = test_results
            except Exception as e:
                logger.error(f"Error running {test_name} test: {e}")
                results[test_name] = {"error": str(e)}
        
        return results
    
    def run_specific_test(self, test_name: str, model, **kwargs) -> Dict[str, float]:
        """Run a specific reasoning test"""
        if test_name not in self.tests:
            raise ValueError(f"Test {test_name} not found")
        
        return self.tests[test_name].evaluate_model(model, **kwargs)
    
    def _parse_response(self, response: str) -> Tuple[str, str]:
        """Parse model response to extract reasoning and answer"""
        try:
            # Look for REASONING and ANSWER markers
            if "REASONING:" in response and "ANSWER:" in response:
                parts = response.split("ANSWER:", 1)
                reasoning = parts[0].replace("REASONING:", "").strip()
                answer = parts[1].strip()
                return reasoning, answer
            else:
                # Fallback: try to extract the last meaningful sentence as answer
                sentences = [s.strip() for s in response.split('.') if s.strip()]
                if sentences:
                    answer = sentences[-1]
                    reasoning = '. '.join(sentences[:-1]) if len(sentences) > 1 else response
                    return reasoning, answer
        except:
            pass
        
        return response, ""