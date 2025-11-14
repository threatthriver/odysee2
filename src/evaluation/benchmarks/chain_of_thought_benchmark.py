"""
Chain-of-Thought Reasoning Benchmark

This benchmark evaluates a model's ability to perform step-by-step reasoning
and generate coherent reasoning chains.
"""

import logging
import torch
from typing import Dict, List, Any, Optional, Tuple
import json
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..benchmark_registry import BaseBenchmark, BenchmarkMetadata, BenchmarkCategory, BenchmarkType

logger = logging.getLogger(__name__)


@dataclass
class CoTExample:
    """Example for chain-of-thought reasoning"""
    question: str
    correct_answer: str
    reasoning_chain: List[str]
    difficulty: str  # easy, medium, hard
    subject: str
    ground_truth_chain: List[str]


class ChainOfThoughtBenchmark(BaseBenchmark):
    """Benchmark for evaluating chain-of-thought reasoning capabilities"""
    
    def __init__(self, metadata: BenchmarkMetadata):
        super().__init__(metadata)
        self.examples: List[CoTExample] = []
        self.evaluation_prompts = self._initialize_prompts()
    
    def _initialize_prompts(self) -> Dict[str, str]:
        """Initialize evaluation prompts for different reasoning types"""
        return {
            "step_by_step": """
            Please solve this problem step by step, showing your reasoning at each step.
            Return your answer in the format: REASONING: [your step-by-step reasoning] ANSWER: [final answer]
            
            Question: {question}
            """,
            
            "justify_each_step": """
            Solve this problem by justifying each step of your reasoning.
            For each step, explain why you made that particular choice.
            Return your answer in the format: REASONING: [step-by-step with justifications] ANSWER: [final answer]
            
            Question: {question}
            """,
            
            "explain_reasoning": """
            Think through this problem carefully and explain your complete reasoning process.
            Show all intermediate steps and logical connections.
            Return your answer in the format: REASONING: [complete reasoning process] ANSWER: [final answer]
            
            Question: {question}
            """,
            
            "verify_steps": """
            Solve this problem step by step, then verify each step is logically sound.
            If you find an error, correct it and continue.
            Return your answer in the format: REASONING: [step-by-step with verification] ANSWER: [final answer]
            
            Question: {question}
            """
        }
    
    def load_dataset(self) -> bool:
        """Load chain-of-thought reasoning examples"""
        try:
            # In a real implementation, this would load from actual datasets
            # For now, we'll create synthetic examples
            self.examples = self._generate_synthetic_examples()
            self.is_loaded = True
            
            logger.info(f"Loaded {len(self.examples)} chain-of-thought examples")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load CoT dataset: {e}")
            return False
    
    def _generate_synthetic_examples(self) -> List[CoTExample]:
        """Generate synthetic CoT examples for testing"""
        examples = []
        
        # Mathematical reasoning examples
        math_examples = [
            CoTExample(
                question="If a store sells pencils for $0.50 each and erasers for $0.25 each, and Sarah bought 3 pencils and 2 erasers, how much did she spend?",
                correct_answer="$2.00",
                reasoning_chain=[
                    "3 pencils at $0.50 each = $1.50",
                    "2 erasers at $0.25 each = $0.50", 
                    "Total = $1.50 + $0.50 = $2.00"
                ],
                difficulty="easy",
                subject="mathematics",
                ground_truth_chain=["$1.50 for pencils", "$0.50 for erasers", "$2.00 total"]
            ),
            CoTExample(
                question="A rectangle has a length of 12 cm and a width of 8 cm. What is its area and perimeter?",
                correct_answer="Area: 96 square cm, Perimeter: 40 cm",
                reasoning_chain=[
                    "Area = length × width = 12 cm × 8 cm = 96 square cm",
                    "Perimeter = 2 × (length + width) = 2 × (12 + 8) = 2 × 20 = 40 cm"
                ],
                difficulty="medium",
                subject="geometry",
                ground_truth_chain=["96 sq cm area", "40 cm perimeter"]
            )
        ]
        
        # Logical reasoning examples
        logic_examples = [
            CoTExample(
                question="All birds can fly. Penguins are birds. Can penguins fly?",
                correct_answer="No, penguins cannot fly",
                reasoning_chain=[
                    "All birds can fly (major premise)",
                    "Penguins are birds (minor premise)",
                    "Therefore, penguins can fly (conclusion) - BUT this contradicts real knowledge",
                    "Correction: While penguins are birds, they cannot fly due to their adaptations"
                ],
                difficulty="hard",
                subject="logic",
                ground_truth_chain=["penguins are flightless birds", "answer is no"]
            )
        ]
        
        # Commonsense reasoning examples
        commonsense_examples = [
            CoTExample(
                question="Why do people use umbrellas when it rains?",
                correct_answer="To stay dry and protected from rain",
                reasoning_chain=[
                    "Umbrellas provide shelter from rain",
                    "Rain makes people wet and uncomfortable",
                    "People want to stay dry",
                    "Therefore, people use umbrellas to stay dry in rain"
                ],
                difficulty="easy",
                subject="commonsense",
                ground_truth_chain=["protection from rain", "staying dry"]
            )
        ]
        
        return math_examples + logic_examples + commonsense_examples
    
    def evaluate_model(self, model, **kwargs) -> Dict[str, float]:
        """Evaluate a model's chain-of-thought reasoning capabilities"""
        if not self.is_loaded:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        # Get evaluation configuration
        prompt_type = kwargs.get('prompt_type', 'step_by_step')
        batch_size = kwargs.get('batch_size', 8)
        max_length = kwargs.get('max_length', 512)
        
        # Evaluate on different reasoning types
        results = {}
        
        for difficulty in ['easy', 'medium', 'hard']:
            difficulty_results = self._evaluate_by_difficulty(model, difficulty, prompt_type, batch_size, max_length)
            results[f'{difficulty}_difficulty'] = difficulty_results
        
        # Evaluate overall performance
        all_results = []
        for example in self.examples:
            score = self._evaluate_single_example(model, example, prompt_type, max_length)
            all_results.append(score)
        
        results['overall_accuracy'] = np.mean(all_results)
        results['coherence_score'] = np.mean([r['coherence'] for r in all_results])
        results['reasoning_quality'] = np.mean([r['reasoning_quality'] for r in all_results])
        
        logger.info(f"CoT evaluation complete. Overall accuracy: {results['overall_accuracy']:.3f}")
        return results
    
    def _evaluate_by_difficulty(self, model, difficulty: str, prompt_type: str, batch_size: int, max_length: int) -> Dict[str, float]:
        """Evaluate performance on examples of a specific difficulty"""
        filtered_examples = [ex for ex in self.examples if ex.difficulty == difficulty]
        
        if not filtered_examples:
            return {'accuracy': 0.0, 'coherence': 0.0, 'reasoning_quality': 0.0}
        
        results = []
        for example in filtered_examples:
            score = self._evaluate_single_example(model, example, prompt_type, max_length)
            results.append(score)
        
        return {
            'accuracy': np.mean([r['accuracy'] for r in results]),
            'coherence': np.mean([r['coherence'] for r in results]),
            'reasoning_quality': np.mean([r['reasoning_quality'] for r in results])
        }
    
    def _evaluate_single_example(self, model, example: CoTExample, prompt_type: str, max_length: int) -> Dict[str, float]:
        """Evaluate a single CoT example"""
        try:
            # Generate response using the model
            prompt = self.evaluation_prompts[prompt_type].format(question=example.question)
            response = self._generate_model_response(model, prompt, max_length)
            
            # Extract reasoning and answer from response
            reasoning, answer = self._parse_response(response)
            
            # Calculate metrics
            accuracy = self._calculate_answer_accuracy(answer, example.correct_answer)
            coherence = self._calculate_reasoning_coherence(reasoning, example.ground_truth_chain)
            reasoning_quality = self._assess_reasoning_quality(reasoning, example.question)
            
            return {
                'accuracy': accuracy,
                'coherence': coherence,
                'reasoning_quality': reasoning_quality,
                'generated_reasoning': reasoning,
                'generated_answer': answer,
                'correct_answer': example.correct_answer
            }
            
        except Exception as e:
            logger.warning(f"Error evaluating example: {e}")
            return {'accuracy': 0.0, 'coherence': 0.0, 'reasoning_quality': 0.0}
    
    def _generate_model_response(self, model, prompt: str, max_length: int) -> str:
        """Generate model response (placeholder implementation)"""
        # In a real implementation, this would call the actual model
        # For demonstration purposes, we'll simulate responses
        
        if "pencils" in prompt and "erasers" in prompt:
            return "REASONING: I need to calculate the cost of pencils and erasers separately. First, 3 pencils at $0.50 each equals $1.50. Next, 2 erasers at $0.25 each equals $0.50. Finally, adding these together gives $2.00. ANSWER: $2.00"
        
        elif "rectangle" in prompt:
            return "REASONING: To find the area of a rectangle, I multiply length by width: 12 cm × 8 cm = 96 square cm. For the perimeter, I add length and width then multiply by 2: (12 + 8) × 2 = 40 cm. ANSWER: Area: 96 square cm, Perimeter: 40 cm"
        
        elif "birds" in prompt and "penguins" in prompt:
            return "REASONING: The statement says all birds can fly and penguins are birds. However, this creates a logical contradiction with real-world knowledge. Penguins are indeed birds but cannot fly due to their aquatic adaptations. ANSWER: No, penguins cannot fly"
        
        else:
            return "REASONING: This is a commonsense question about umbrellas and rain. ANSWER: To stay dry"
    
    def _parse_response(self, response: str) -> Tuple[str, str]:
        """Parse model response to extract reasoning and answer"""
        try:
            parts = response.split("ANSWER:")
            if len(parts) == 2:
                reasoning = parts[0].replace("REASONING:", "").strip()
                answer = parts[1].strip()
                return reasoning, answer
            else:
                return response, ""
        except:
            return response, ""
    
    def _calculate_answer_accuracy(self, predicted_answer: str, correct_answer: str) -> float:
        """Calculate accuracy of predicted answer"""
        # Simple exact match for now - could be enhanced with semantic similarity
        predicted = predicted_answer.lower().strip()
        correct = correct_answer.lower().strip()
        
        # Handle numerical answers
        if any(char.isdigit() for char in predicted) and any(char.isdigit() for char in correct):
            return 1.0 if predicted in correct or correct in predicted else 0.0
        
        # Handle yes/no questions
        if predicted in ['yes', 'no'] and correct in ['yes', 'no']:
            return 1.0 if predicted == correct else 0.0
        
        # Text similarity (simplified)
        words_predicted = set(predicted.split())
        words_correct = set(correct.split())
        
        if not words_correct:
            return 0.0
        
        intersection = words_predicted.intersection(words_correct)
        return len(intersection) / len(words_correct)
    
    def _calculate_reasoning_coherence(self, generated_reasoning: str, ground_truth_steps: List[str]) -> float:
        """Calculate coherence score of generated reasoning"""
        if not generated_reasoning or not ground_truth_steps:
            return 0.0
        
        # Count logical connectors and step indicators
        connectors = ['first', 'next', 'then', 'therefore', 'because', 'since', 'thus', 'hence']
        step_indicators = ['step', 'calculate', 'add', 'multiply', 'divide', 'subtract']
        
        connector_count = sum(1 for connector in connectors if connector.lower() in generated_reasoning.lower())
        step_count = sum(1 for indicator in step_indicators if indicator.lower() in generated_reasoning.lower())
        
        # Normalize by reasoning length
        words = generated_reasoning.split()
        if len(words) < 3:
            return 0.0
        
        coherence_score = (connector_count + step_count) / len(words)
        return min(coherence_score * 10, 1.0)  # Scale to [0, 1]
    
    def _assess_reasoning_quality(self, reasoning: str, question: str) -> float:
        """Assess the overall quality of reasoning"""
        if not reasoning:
            return 0.0
        
        # Criteria for good reasoning
        criteria_scores = {
            'completeness': self._assess_completeness(reasoning, question),
            'logical_flow': self._assess_logical_flow(reasoning),
            'step_clarity': self._assess_step_clarity(reasoning),
            'relevance': self._assess_relevance(reasoning, question)
        }
        
        return np.mean(list(criteria_scores.values()))
    
    def _assess_completeness(self, reasoning: str, question: str) -> float:
        """Assess if reasoning addresses all aspects of the question"""
        question_words = set(question.lower().split())
        reasoning_words = set(reasoning.lower().split())
        
        # Check if key question elements are addressed
        coverage = len(question_words.intersection(reasoning_words)) / len(question_words)
        return min(coverage * 2, 1.0)  # Weight coverage higher
    
    def _assess_logical_flow(self, reasoning: str) -> float:
        """Assess logical flow in reasoning"""
        # Look for logical progression indicators
        flow_indicators = ['then', 'next', 'after', 'finally', 'therefore', 'so', 'thus']
        indicator_count = sum(1 for indicator in flow_indicators if indicator in reasoning.lower())
        
        # Penalize lack of flow indicators but don't over-weight
        return min(indicator_count / 3, 1.0)
    
    def _assess_step_clarity(self, reasoning: str) -> float:
        """Assess clarity of reasoning steps"""
        # Look for step indicators and clear intermediate results
        step_indicators = ['=', 'equals', 'result', 'step', 'calculate']
        clarity_count = sum(1 for indicator in step_indicators if indicator in reasoning.lower())
        
        return min(clarity_count / 4, 1.0)
    
    def _assess_relevance(self, reasoning: str, question: str) -> float:
        """Assess relevance of reasoning to the question"""
        # Simple relevance check based on shared domain keywords
        reasoning_lower = reasoning.lower()
        question_lower = question.lower()
        
        domain_keywords = {
            'math': ['add', 'subtract', 'multiply', 'divide', 'calculate', 'equals'],
            'logic': ['therefore', 'because', 'since', 'if', 'then'],
            'commonsense': ['people', 'use', 'why', 'because']
        }
        
        question_domain = None
        for domain, keywords in domain_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                question_domain = domain
                break
        
        if question_domain:
            domain_keywords_in_reasoning = sum(1 for keyword in domain_keywords[question_domain] 
                                              if keyword in reasoning_lower)
            return min(domain_keywords_in_reasoning / 3, 1.0)
        
        return 0.5  # Neutral score if no clear domain
    
    def validate_setup(self) -> bool:
        """Validate the CoT benchmark setup"""
        try:
            # Check if evaluation prompts are defined
            if not self.evaluation_prompts:
                logger.error("No evaluation prompts defined")
                return False
            
            # Check prompt format consistency
            for prompt_type, prompt in self.evaluation_prompts.items():
                if 'REASONING:' not in prompt or 'ANSWER:' not in prompt:
                    logger.error(f"Invalid prompt format for {prompt_type}")
                    return False
            
            logger.info("CoT benchmark validation passed")
            return True
            
        except Exception as e:
            logger.error(f"CoT benchmark validation failed: {e}")
            return False