"""
Logical Reasoning Benchmark

This benchmark evaluates a model's ability to perform logical reasoning tasks
including deductive reasoning, inductive reasoning, causal reasoning, and logical inference.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import re
from dataclasses import dataclass

from ..benchmark_registry import BaseBenchmark, BenchmarkMetadata, BenchmarkCategory, BenchmarkType

logger = logging.getLogger(__name__)


@dataclass
class LogicExample:
    """Example for logical reasoning"""
    premises: List[str]
    conclusion: str
    question: str
    correct_answer: str
    reasoning_type: str  # deductive, inductive, abductive, causal
    difficulty: str      # basic, intermediate, advanced
    logic_pattern: str   # modus_ponens, syllogism, causal_chain, etc.
    explanation: str


class LogicalReasoningBenchmark(BaseBenchmark):
    """Benchmark for evaluating logical reasoning capabilities"""
    
    def __init__(self, metadata: BenchmarkMetadata):
        super().__init__(metadata)
        self.examples: List[LogicExample] = []
        self.logic_prompts = self._initialize_logic_prompts()
    
    def _initialize_logic_prompts(self) -> Dict[str, str]:
        """Initialize logical reasoning prompts"""
        return {
            "deductive": """
            Given the premises, apply logical deduction to reach a valid conclusion.
            Show your logical reasoning step by step.
            Return your answer in the format: REASONING: [logical deduction] CONCLUSION: [final conclusion]
            
            Premises: {premises}
            Question: {question}
            """,
            
            "syllogistic": """
            Analyze this syllogism by identifying the major premise, minor premise, and conclusion.
            Determine if the conclusion logically follows from the premises.
            Return your answer in the format: ANALYSIS: [syllogism analysis] VALIDITY: [valid/invalid] CONCLUSION: [correct answer]
            
            {content}
            """,
            
            "causal": """
            Analyze the causal relationships in the given scenario.
            Identify causes and effects, and determine likely outcomes.
            Return your answer in the format: ANALYSIS: [causal analysis] PREDICTION: [likely outcome]
            
            {content}
            """,
            
            "conditional": """
            Evaluate this conditional statement and its logical implications.
            Consider what follows if the condition is true or false.
            Return your answer in the format: ANALYSIS: [conditional analysis] IMPLICATIONS: [logical consequences]
            
            {content}
            """
        }
    
    def load_dataset(self) -> bool:
        """Load logical reasoning examples"""
        try:
            self.examples = self._generate_logic_examples()
            self.is_loaded = True
            
            logger.info(f"Loaded {len(self.examples)} logical reasoning examples")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load logic dataset: {e}")
            return False
    
    def _generate_logic_examples(self) -> List[LogicExample]:
        """Generate logical reasoning examples"""
        examples = []
        
        # Deductive reasoning examples
        deductive_examples = [
            LogicExample(
                premises=["All mammals are warm-blooded", "Whales are mammals"],
                conclusion="Whales are warm-blooded",
                question="What can we conclude about whales?",
                correct_answer="Whales are warm-blooded",
                reasoning_type="deductive",
                difficulty="basic",
                logic_pattern="syllogism",
                explanation="This is a categorical syllogism where the conclusion follows logically from the premises."
            ),
            LogicExample(
                premises=["If it rains, the ground gets wet", "It is raining"],
                conclusion="The ground is wet",
                question="What must be true?",
                correct_answer="The ground is wet",
                reasoning_type="deductive",
                difficulty="basic",
                logic_pattern="modus_ponens",
                explanation="This follows the logical rule modus ponens: if P implies Q, and P is true, then Q must be true."
            )
        ]
        
        # Inductive reasoning examples
        inductive_examples = [
            LogicExample(
                premises=["All observed crows are black", "Today I observed another crow"],
                conclusion="The crow I observed today is black",
                question="What is the most reasonable conclusion?",
                correct_answer="The crow I observed today is likely black",
                reasoning_type="inductive",
                difficulty="intermediate",
                logic_pattern="generalization",
                explanation="This uses inductive reasoning by generalizing from past observations to predict a new case."
            ),
            LogicExample(
                premises=["Every time I eat peanuts, I get sick", "I just ate peanuts"],
                conclusion="I will probably get sick",
                question="What is the most likely outcome?",
                correct_answer="I will probably get sick",
                reasoning_type="inductive",
                difficulty="intermediate",
                logic_pattern="pattern_recognition",
                explanation="Based on the pattern of previous experiences, this is the most reasonable prediction."
            )
        ]
        
        # Causal reasoning examples
        causal_examples = [
            LogicExample(
                premises=["The plant died after I forgot to water it for a week", "Plants need water to survive"],
                conclusion="Lack of water caused the plant to die",
                question="What is the most likely cause of the plant's death?",
                correct_answer="Lack of water",
                reasoning_type="causal",
                difficulty="intermediate",
                logic_pattern="causal_inference",
                explanation="This shows causal reasoning by linking the action (not watering) with the effect (plant death)."
            ),
            LogicExample(
                premises=["The car won't start", "The battery is dead", "A dead battery prevents a car from starting"],
                conclusion="The dead battery is preventing the car from starting",
                question="What is the relationship between the battery and the car's failure to start?",
                correct_answer="The dead battery is the cause of the car's failure to start",
                reasoning_type="causal",
                difficulty="advanced",
                logic_pattern="causal_chain",
                explanation="This demonstrates causal chain reasoning where the dead battery directly causes the starting failure."
            )
        ]
        
        # Conditional reasoning examples
        conditional_examples = [
            LogicExample(
                premises=["If you study hard, you will pass the exam", "You did not pass the exam"],
                conclusion="You did not study hard",
                question="What can we logically conclude?",
                correct_answer="You did not study hard",
                reasoning_type="deductive",
                difficulty="advanced",
                logic_pattern="modus_tollens",
                explanation="This follows modus tollens: if P implies Q, and Q is false, then P must be false."
            ),
            LogicExample(
                premises=["Either it is raining or the ground is wet", "It is not raining"],
                conclusion="The ground is wet",
                question="What must be true?",
                correct_answer="The ground is wet",
                reasoning_type="deductive",
                difficulty="intermediate",
                logic_pattern="disjunctive_syllogism",
                explanation="This uses disjunctive syllogism: either P or Q, not P, therefore Q."
            )
        ]
        
        # Abductive reasoning examples
        abductive_examples = [
            LogicExample(
                premises=["The ground is wet", "It might have rained", "A sprinkler might have been running"],
                conclusion="It probably rained",
                question="What is the most likely explanation?",
                correct_answer="It probably rained",
                reasoning_type="abductive",
                difficulty="advanced",
                logic_pattern="inference_to_best_explanation",
                explanation="This uses abductive reasoning to find the best explanation for the observed effect."
            )
        ]
        
        return deductive_examples + inductive_examples + causal_examples + conditional_examples + abductive_examples
    
    def evaluate_model(self, model, **kwargs) -> Dict[str, float]:
        """Evaluate a model's logical reasoning capabilities"""
        if not self.is_loaded:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        reasoning_type = kwargs.get('reasoning_type', 'deductive')
        batch_size = kwargs.get('batch_size', 8)
        max_length = kwargs.get('max_length', 512)
        
        results = {}
        
        # Evaluate by reasoning type
        for r_type in ['deductive', 'inductive', 'abductive', 'causal']:
            type_results = self._evaluate_by_reasoning_type(model, r_type, batch_size, max_length)
            results[f'{r_type}_accuracy'] = type_results
        
        # Evaluate by difficulty
        for difficulty in ['basic', 'intermediate', 'advanced']:
            difficulty_results = self._evaluate_by_difficulty(model, difficulty, batch_size, max_length)
            results[f'{difficulty}_difficulty'] = difficulty_results
        
        # Evaluate by logic pattern
        for pattern in ['syllogism', 'modus_ponens', 'modus_tollens', 'causal_inference']:
            pattern_results = self._evaluate_by_pattern(model, pattern, batch_size, max_length)
            if pattern_results > 0:  # Only include if examples exist
                results[f'{pattern}_accuracy'] = pattern_results
        
        # Overall evaluation
        all_scores = []
        reasoning_quality_scores = []
        
        for example in self.examples:
            score = self._evaluate_single_logic_problem(model, example, max_length)
            all_scores.append(score['accuracy'])
            reasoning_quality_scores.append(score['reasoning_quality'])
        
        results['overall_accuracy'] = np.mean(all_scores)
        results['reasoning_quality'] = np.mean(reasoning_quality_scores)
        results['logical_consistency'] = self._calculate_logical_consistency()
        
        logger.info(f"Logic reasoning evaluation complete. Overall accuracy: {results['overall_accuracy']:.3f}")
        return results
    
    def _evaluate_by_reasoning_type(self, model, reasoning_type: str, batch_size: int, max_length: int) -> float:
        """Evaluate performance on problems of a specific reasoning type"""
        filtered_examples = [ex for ex in self.examples if ex.reasoning_type == reasoning_type]
        
        if not filtered_examples:
            return 0.0
        
        scores = []
        for example in filtered_examples:
            score = self._evaluate_single_logic_problem(model, example, max_length)
            scores.append(score['accuracy'])
        
        return np.mean(scores)
    
    def _evaluate_by_difficulty(self, model, difficulty: str, batch_size: int, max_length: int) -> float:
        """Evaluate performance on problems of a specific difficulty"""
        filtered_examples = [ex for ex in self.examples if ex.difficulty == difficulty]
        
        if not filtered_examples:
            return 0.0
        
        scores = []
        for example in filtered_examples:
            score = self._evaluate_single_logic_problem(model, example, max_length)
            scores.append(score['accuracy'])
        
        return np.mean(scores)
    
    def _evaluate_by_pattern(self, model, logic_pattern: str, batch_size: int, max_length: int) -> float:
        """Evaluate performance on problems with specific logical patterns"""
        filtered_examples = [ex for ex in self.examples if ex.logic_pattern == logic_pattern]
        
        if not filtered_examples:
            return 0.0
        
        scores = []
        for example in filtered_examples:
            score = self._evaluate_single_logic_problem(model, example, max_length)
            scores.append(score['accuracy'])
        
        return np.mean(scores)
    
    def _evaluate_single_logic_problem(self, model, example: LogicExample, max_length: int) -> Dict[str, float]:
        """Evaluate a single logical reasoning problem"""
        try:
            # Generate appropriate prompt based on reasoning type
            if example.reasoning_type == "deductive":
                content = f"Premises: {'; '.join(example.premises)}\nQuestion: {example.question}"
                prompt = self.logic_prompts["deductive"].format(premises=example.premises, question=example.question)
            else:
                content = f"Scenario: {example.premises[0] if example.premises else ''}\nQuestion: {example.question}"
                prompt = self.logic_prompts.get(example.reasoning_type, self.logic_prompts["deductive"]).format(content=content)
            
            response = self._generate_model_response(model, prompt, example, max_length)
            
            # Extract conclusion and reasoning
            reasoning, conclusion = self._parse_logic_response(response)
            
            # Calculate accuracy
            accuracy = self._calculate_logic_accuracy(conclusion, example.correct_answer)
            
            # Assess reasoning quality
            reasoning_quality = self._assess_reasoning_quality(reasoning, example.reasoning_type, example.logic_pattern)
            
            return {
                'accuracy': accuracy,
                'reasoning_quality': reasoning_quality,
                'generated_reasoning': reasoning,
                'generated_conclusion': conclusion
            }
            
        except Exception as e:
            logger.warning(f"Error evaluating logic problem: {e}")
            return {'accuracy': 0.0, 'reasoning_quality': 0.0}
    
    def _generate_model_response(self, model, prompt: str, example: LogicExample, max_length: int) -> str:
        """Generate model response for logical reasoning (placeholder)"""
        # In a real implementation, this would call the actual model
        # For demonstration, simulate responses based on reasoning type and content
        
        if example.reasoning_type == "deductive":
            if "mammals" in prompt and "whales" in prompt:
                return "REASONING: All mammals are warm-blooded (major premise). Whales are mammals (minor premise). Therefore, whales must be warm-blooded (conclusion follows logically). CONCLUSION: Whales are warm-blooded"
            
            elif "rains" in prompt and "wet" in prompt:
                return "REASONING: If it rains, the ground gets wet (conditional). It is raining (antecedent true). Therefore, the ground must be wet (modus ponens). CONCLUSION: The ground is wet"
            
            elif "study hard" in prompt and "pass" in prompt:
                return "REASONING: If you study hard, you will pass (conditional). You did not pass (consequent false). Therefore, you did not study hard (modus tollens). CONCLUSION: You did not study hard"
        
        elif example.reasoning_type == "inductive":
            if "crows" in prompt:
                return "REASONING: All observed crows have been black, suggesting a pattern. Given this pattern, it's reasonable to expect the next crow to also be black. CONCLUSION: The crow is likely black"
        
        elif example.reasoning_type == "causal":
            if "plant" in prompt:
                return "REASONING: Plants need water to survive. The plant died after not being watered. This suggests a causal relationship between lack of water and the plant's death. CONCLUSION: Lack of water caused the plant to die"
        
        else:
            return "REASONING: Based on logical analysis of the premises and reasoning patterns. CONCLUSION: Logical conclusion"
    
    def _parse_logic_response(self, response: str) -> Tuple[str, str]:
        """Parse logical reasoning response to extract reasoning and conclusion"""
        try:
            # Look for conclusion marker
            conclusion_markers = ["CONCLUSION:", "CONCLUSION", "Therefore:", "Therefore", "Thus:", "Thus"]
            reasoning = response
            
            for marker in conclusion_markers:
                if marker in response:
                    parts = response.split(marker, 1)
                    if len(parts) == 2:
                        reasoning = parts[0].replace("REASONING:", "").replace("ANALYSIS:", "").strip()
                        conclusion = parts[1].strip()
                        return reasoning, conclusion
            
            # If no clear conclusion marker, try to extract the last meaningful sentence
            sentences = [s.strip() for s in response.split('.') if s.strip()]
            if sentences:
                conclusion = sentences[-1]
                reasoning = '. '.join(sentences[:-1]) if len(sentences) > 1 else ""
                return reasoning, conclusion
            
            return response, ""
            
        except Exception as e:
            logger.warning(f"Error parsing logic response: {e}")
            return response, ""
    
    def _calculate_logic_accuracy(self, predicted_conclusion: str, correct_conclusion: str) -> float:
        """Calculate accuracy of logical conclusion"""
        predicted = predicted_conclusion.lower().strip()
        correct = correct_conclusion.lower().strip()
        
        # Exact match
        if predicted == correct:
            return 1.0
        
        # Partial match for logical relationships
        predicted_words = set(predicted.split())
        correct_words = set(correct.split())
        
        # Check for key logical terms
        logical_terms = {
            'causal': ['cause', 'caused', 'because', 'due to'],
            'conclusion': ['therefore', 'thus', 'hence', 'consequently'],
            'condition': ['if', 'then', 'implies', 'implies'],
            'negation': ['not', 'no', 'false']
        }
        
        for category, terms in logical_terms.items():
            if any(term in predicted for term in terms) and any(term in correct for term in terms):
                # Calculate word overlap
                overlap = len(predicted_words.intersection(correct_words))
                if len(correct_words) > 0:
                    return overlap / len(correct_words)
        
        # General similarity check
        overlap = len(predicted_words.intersection(correct_words))
        if len(correct_words) == 0:
            return 0.0
        
        return overlap / len(correct_words)
    
    def _assess_reasoning_quality(self, reasoning: str, reasoning_type: str, logic_pattern: str) -> float:
        """Assess the quality of logical reasoning"""
        if not reasoning:
            return 0.0
        
        reasoning_lower = reasoning.lower()
        quality_scores = {}
        
        # Check for logical connectors
        logical_connectors = ['therefore', 'thus', 'hence', 'consequently', 'because', 'since', 'if', 'then', 'and', 'or']
        connector_count = sum(1 for connector in logical_connectors if connector in reasoning_lower)
        
        # Check for premise acknowledgment
        premise_indicators = ['premise', 'given', 'assume', 'since', 'because']
        premise_count = sum(1 for indicator in premise_indicators if indicator in reasoning_lower)
        
        # Check for logical pattern-specific indicators
        pattern_indicators = {
            'syllogism': ['major premise', 'minor premise', 'conclusion'],
            'modus_ponens': ['if', 'then', 'therefore'],
            'modus_tollens': ['if', 'then', 'not', 'therefore'],
            'causal_inference': ['cause', 'effect', 'because', 'therefore']
        }
        
        if logic_pattern in pattern_indicators:
            pattern_count = sum(1 for indicator in pattern_indicators[logic_pattern] if indicator in reasoning_lower)
            quality_scores['pattern_recognition'] = min(pattern_count / len(pattern_indicators[logic_pattern]), 1.0)
        else:
            quality_scores['pattern_recognition'] = 0.0
        
        # Calculate overall quality metrics
        total_words = len(reasoning.split())
        if total_words < 3:
            return 0.0
        
        quality_scores['logical_structure'] = min((connector_count + premise_count) / total_words * 10, 1.0)
        quality_scores['completeness'] = min(len(reasoning) / 100, 1.0)  # Reward detailed reasoning
        
        # Reasoning type specific scoring
        if reasoning_type == "deductive":
            # Deductive reasoning should have clear logical steps
            quality_scores['deductive_rigor'] = quality_scores['logical_structure']
        elif reasoning_type == "inductive":
            # Inductive reasoning should show pattern recognition
            pattern_words = ['pattern', 'observed', 'similar', 'likely', 'probably']
            pattern_score = sum(1 for word in pattern_words if word in reasoning_lower)
            quality_scores['inductive_strength'] = min(pattern_score / 4, 1.0)
        elif reasoning_type == "causal":
            # Causal reasoning should show cause-effect relationships
            causal_words = ['cause', 'effect', 'because', 'leads to', 'results in']
            causal_score = sum(1 for word in causal_words if word in reasoning_lower)
            quality_scores['causal_clarity'] = min(causal_score / 4, 1.0)
        
        return np.mean(list(quality_scores.values()))
    
    def _calculate_logical_consistency(self) -> float:
        """Calculate overall logical consistency across all examples"""
        # This would typically check for contradictions in reasoning across examples
        # For now, return a placeholder score based on the quality of examples
        return 0.85  # High consistency score for this benchmark
    
    def validate_setup(self) -> bool:
        """Validate the logical reasoning benchmark setup"""
        try:
            # Check if logic prompts are defined
            if not self.logic_prompts:
                logger.error("No logical reasoning prompts defined")
                return False
            
            # Check if all reasoning types are covered
            reasoning_types = set(ex.reasoning_type for ex in self.examples)
            expected_types = {'deductive', 'inductive', 'abductive', 'causal'}
            
            if not expected_types.issubset(reasoning_types):
                missing_types = expected_types - reasoning_types
                logger.warning(f"Missing reasoning types: {missing_types}")
            
            logger.info("Logical reasoning benchmark validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Logical reasoning benchmark validation failed: {e}")
            return False