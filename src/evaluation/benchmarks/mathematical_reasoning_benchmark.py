"""
Mathematical Reasoning Benchmark

This benchmark evaluates a model's ability to solve mathematical problems
of varying complexity, from arithmetic to advanced mathematical concepts.
"""

import logging
import torch
import sympy as sp
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from fractions import Fraction
import re
from dataclasses import dataclass

from ..benchmark_registry import BaseBenchmark, BenchmarkMetadata, BenchmarkCategory, BenchmarkType

logger = logging.getLogger(__name__)


@dataclass
class MathExample:
    """Example for mathematical reasoning"""
    question: str
    correct_answer: Union[str, float, Fraction]
    answer_type: str  # numerical, symbolic, verbal
    difficulty: str   # elementary, intermediate, advanced
    topic: str        # arithmetic, algebra, geometry, calculus, etc.
    solution_steps: List[str]
    mathematical_concepts: List[str]


class MathematicalReasoningBenchmark(BaseBenchmark):
    """Benchmark for evaluating mathematical reasoning capabilities"""
    
    def __init__(self, metadata: BenchmarkMetadata):
        super().__init__(metadata)
        self.examples: List[MathExample] = []
        self.math_prompts = self._initialize_math_prompts()
        self.symbolic_solver = self._initialize_symbolic_solver()
    
    def _initialize_math_prompts(self) -> Dict[str, str]:
        """Initialize mathematical problem-solving prompts"""
        return {
            "step_by_step": """
            Solve this mathematical problem step by step, showing all calculations.
            Return your answer in the format: SOLUTION: [step-by-step solution] ANSWER: [final answer]
            
            Problem: {question}
            """,
            
            "show_work": """
            Solve this problem showing all intermediate work and reasoning.
            Clearly indicate each mathematical operation and concept used.
            Return your answer in the format: SOLUTION: [detailed solution with work] ANSWER: [final answer]
            
            Problem: {question}
            """,
            
            "verify_result": """
            Solve this problem, then verify your answer by working backwards or using an alternative method.
            Return your answer in the format: SOLUTION: [solution with verification] ANSWER: [final answer]
            
            Problem: {question}
            """,
            
            "concept_focused": """
            Solve this problem, identifying and applying the relevant mathematical concepts.
            Explain which concepts you're using and why.
            Return your answer in the format: SOLUTION: [concept-focused solution] ANSWER: [final answer]
            
            Problem: {question}
            """
        }
    
    def _initialize_symbolic_solver(self):
        """Initialize symbolic math solver for verification"""
        # Initialize SymPy for symbolic mathematics
        try:
            x, y, z = sp.symbols('x y z')
            return {'symbols': [x, y, z], 'available': True}
        except:
            logger.warning("SymPy not available for symbolic verification")
            return {'symbols': [], 'available': False}
    
    def load_dataset(self) -> bool:
        """Load mathematical reasoning examples"""
        try:
            self.examples = self._generate_math_examples()
            self.is_loaded = True
            
            logger.info(f"Loaded {len(self.examples)} mathematical reasoning examples")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load math dataset: {e}")
            return False
    
    def _generate_math_examples(self) -> List[MathExample]:
        """Generate mathematical reasoning examples"""
        examples = []
        
        # Arithmetic examples
        arithmetic_examples = [
            MathExample(
                question="Calculate: 127 + 89 - 34 × 2",
                correct_answer=148,
                answer_type="numerical",
                difficulty="elementary",
                topic="arithmetic",
                solution_steps=[
                    "Apply order of operations: 34 × 2 = 68",
                    "Then: 127 + 89 = 216", 
                    "Finally: 216 - 68 = 148"
                ],
                mathematical_concepts=["order_of_operations", "arithmetic"]
            ),
            MathExample(
                question="What is 25% of 240?",
                correct_answer=60,
                answer_type="numerical", 
                difficulty="elementary",
                topic="percentages",
                solution_steps=[
                    "25% = 25/100 = 1/4",
                    "25% of 240 = (1/4) × 240 = 60"
                ],
                mathematical_concepts=["percentages", "fractions"]
            )
        ]
        
        # Algebra examples
        algebra_examples = [
            MathExample(
                question="Solve for x: 2x + 5 = 17",
                correct_answer=6,
                answer_type="numerical",
                difficulty="intermediate",
                topic="algebra",
                solution_steps=[
                    "Subtract 5 from both sides: 2x = 12",
                    "Divide both sides by 2: x = 6",
                    "Verify: 2(6) + 5 = 17 ✓"
                ],
                mathematical_concepts=["linear_equations", "algebraic_manipulation"]
            ),
            MathExample(
                question="If y = 3x² + 2x - 1, find dy/dx",
                correct_answer="6x + 2",
                answer_type="symbolic",
                difficulty="advanced",
                topic="calculus",
                solution_steps=[
                    "Apply power rule to 3x²: d/dx(3x²) = 6x",
                    "Apply power rule to 2x: d/dx(2x) = 2", 
                    "Derivative of constant -1 is 0",
                    "Therefore: dy/dx = 6x + 2"
                ],
                mathematical_concepts=["calculus", "derivatives", "power_rule"]
            )
        ]
        
        # Geometry examples
        geometry_examples = [
            MathExample(
                question="A circle has radius 5 cm. Find its area and circumference.",
                correct_answer="Area: 25π cm², Circumference: 10π cm",
                answer_type="symbolic",
                difficulty="intermediate", 
                topic="geometry",
                solution_steps=[
                    "Area formula: A = πr² = π(5)² = 25π cm²",
                    "Circumference formula: C = 2πr = 2π(5) = 10π cm"
                ],
                mathematical_concepts=["geometry", "circle_formulas"]
            ),
            MathExample(
                question="Find the volume of a sphere with radius 3 cm.",
                correct_answer="36π cm³",
                answer_type="symbolic",
                difficulty="intermediate",
                topic="geometry", 
                solution_steps=[
                    "Volume formula: V = (4/3)πr³",
                    "V = (4/3)π(3)³ = (4/3)π(27) = 36π cm³"
                ],
                mathematical_concepts=["geometry", "sphere_volume"]
            )
        ]
        
        # Word problem examples
        word_problem_examples = [
            MathExample(
                question="A car travels 240 miles in 4 hours. What is its average speed?",
                correct_answer="60 miles per hour",
                answer_type="numerical",
                difficulty="intermediate",
                topic="word_problems",
                solution_steps=[
                    "Average speed = distance ÷ time",
                    "Speed = 240 miles ÷ 4 hours = 60 mph"
                ],
                mathematical_concepts=["word_problems", "rate_problems"]
            ),
            MathExample(
                question="A rectangle's length is 3 times its width. If the perimeter is 32 cm, find the dimensions.",
                correct_answer="Length: 12 cm, Width: 4 cm",
                answer_type="numerical",
                difficulty="intermediate",
                topic="algebra",
                solution_steps=[
                    "Let width = w, then length = 3w",
                    "Perimeter = 2(length + width) = 2(3w + w) = 8w",
                    "8w = 32, so w = 4 cm",
                    "Length = 3(4) = 12 cm"
                ],
                mathematical_concepts=["algebra", "perimeter", "word_problems"]
            )
        ]
        
        return arithmetic_examples + algebra_examples + geometry_examples + word_problem_examples
    
    def evaluate_model(self, model, **kwargs) -> Dict[str, float]:
        """Evaluate a model's mathematical reasoning capabilities"""
        if not self.is_loaded:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        prompt_type = kwargs.get('prompt_type', 'step_by_step')
        batch_size = kwargs.get('batch_size', 8)
        max_length = kwargs.get('max_length', 512)
        
        results = {}
        
        # Evaluate by topic
        for topic in ['arithmetic', 'algebra', 'geometry', 'calculus', 'word_problems']:
            topic_results = self._evaluate_by_topic(model, topic, prompt_type, batch_size, max_length)
            results[f'{topic}_score'] = topic_results
        
        # Evaluate by difficulty
        for difficulty in ['elementary', 'intermediate', 'advanced']:
            difficulty_results = self._evaluate_by_difficulty(model, difficulty, prompt_type, batch_size, max_length)
            results[f'{difficulty}_difficulty'] = difficulty_results
        
        # Overall evaluation
        all_scores = []
        step_accuracy_scores = []
        concept_usage_scores = []
        
        for example in self.examples:
            score = self._evaluate_single_math_problem(model, example, prompt_type, max_length)
            all_scores.append(score['correctness'])
            step_accuracy_scores.append(score['step_accuracy'])
            concept_usage_scores.append(score['concept_usage'])
        
        results['overall_accuracy'] = np.mean(all_scores)
        results['step_accuracy'] = np.mean(step_accuracy_scores)
        results['concept_usage'] = np.mean(concept_usage_scores)
        results['mathematical_rigor'] = np.mean([
            (correctness + step_accuracy + concept_usage) / 3 
            for correctness, step_accuracy, concept_usage in 
            zip(all_scores, step_accuracy_scores, concept_usage_scores)
        ])
        
        logger.info(f"Math reasoning evaluation complete. Overall accuracy: {results['overall_accuracy']:.3f}")
        return results
    
    def _evaluate_by_topic(self, model, topic: str, prompt_type: str, batch_size: int, max_length: int) -> float:
        """Evaluate performance on problems of a specific topic"""
        filtered_examples = [ex for ex in self.examples if ex.topic == topic]
        
        if not filtered_examples:
            return 0.0
        
        scores = []
        for example in filtered_examples:
            score = self._evaluate_single_math_problem(model, example, prompt_type, max_length)
            scores.append(score['correctness'])
        
        return np.mean(scores)
    
    def _evaluate_by_difficulty(self, model, difficulty: str, prompt_type: str, batch_size: int, max_length: int) -> float:
        """Evaluate performance on problems of a specific difficulty"""
        filtered_examples = [ex for ex in self.examples if ex.difficulty == difficulty]
        
        if not filtered_examples:
            return 0.0
        
        scores = []
        for example in filtered_examples:
            score = self._evaluate_single_math_problem(model, example, prompt_type, max_length)
            scores.append(score['correctness'])
        
        return np.mean(scores)
    
    def _evaluate_single_math_problem(self, model, example: MathExample, prompt_type: str, max_length: int) -> Dict[str, float]:
        """Evaluate a single mathematical problem"""
        try:
            prompt = self.math_prompts[prompt_type].format(question=example.question)
            response = self._generate_model_response(model, prompt, max_length)
            
            # Parse solution and answer
            solution, answer = self._parse_math_response(response)
            
            # Calculate correctness
            correctness = self._calculate_math_correctness(answer, example.correct_answer, example.answer_type)
            
            # Calculate step accuracy
            step_accuracy = self._assess_step_accuracy(solution, example.solution_steps)
            
            # Calculate concept usage
            concept_usage = self._assess_concept_usage(solution, example.mathematical_concepts)
            
            return {
                'correctness': correctness,
                'step_accuracy': step_accuracy,
                'concept_usage': concept_usage,
                'generated_solution': solution,
                'generated_answer': answer
            }
            
        except Exception as e:
            logger.warning(f"Error evaluating math problem: {e}")
            return {'correctness': 0.0, 'step_accuracy': 0.0, 'concept_usage': 0.0}
    
    def _generate_model_response(self, model, prompt: str, max_length: int) -> str:
        """Generate model response for mathematical problems (placeholder)"""
        # In a real implementation, this would call the actual model
        # For demonstration, simulate responses based on problem content
        
        if "127 + 89 - 34 × 2" in prompt:
            return "SOLUTION: Following order of operations: First 34 × 2 = 68. Then 127 + 89 = 216. Finally 216 - 68 = 148. ANSWER: 148"
        
        elif "25% of 240" in prompt:
            return "SOLUTION: 25% = 25/100 = 1/4. Therefore 25% of 240 = (1/4) × 240 = 60. ANSWER: 60"
        
        elif "2x + 5 = 17" in prompt:
            return "SOLUTION: Subtract 5 from both sides: 2x = 12. Divide both sides by 2: x = 6. ANSWER: 6"
        
        elif "y = 3x² + 2x - 1" in prompt:
            return "SOLUTION: Apply power rule to 3x² to get 6x. Apply power rule to 2x to get 2. Constant term -1 has derivative 0. ANSWER: 6x + 2"
        
        elif "radius 5 cm" in prompt and "area" in prompt:
            return "SOLUTION: Area = πr² = π(5)² = 25π cm². Circumference = 2πr = 2π(5) = 10π cm. ANSWER: Area: 25π cm², Circumference: 10π cm"
        
        elif "240 miles" in prompt and "4 hours" in prompt:
            return "SOLUTION: Average speed = distance ÷ time = 240 miles ÷ 4 hours = 60 mph. ANSWER: 60 miles per hour"
        
        else:
            return "SOLUTION: This requires basic mathematical reasoning. ANSWER: [numerical result]"
    
    def _parse_math_response(self, response: str) -> Tuple[str, str]:
        """Parse mathematical response to extract solution and answer"""
        try:
            parts = response.split("ANSWER:")
            if len(parts) == 2:
                solution = parts[0].replace("SOLUTION:", "").strip()
                answer = parts[1].strip()
                return solution, answer
            else:
                # Try to extract numerical answer if no clear separation
                numbers = re.findall(r'-?\d+\.?\d*', response)
                answer = numbers[-1] if numbers else ""
                return response, answer
        except:
            return response, ""
    
    def _calculate_math_correctness(self, predicted_answer: str, correct_answer: Union[str, float, Fraction], answer_type: str) -> float:
        """Calculate correctness of mathematical answer"""
        try:
            # Normalize predicted answer
            predicted = predicted_answer.strip().lower()
            
            if answer_type == "numerical":
                # Extract numerical value from predicted answer
                predicted_nums = re.findall(r'-?\d+\.?\d*', predicted)
                if predicted_nums:
                    predicted_val = float(predicted_nums[0])
                    
                    # Compare with correct answer
                    if isinstance(correct_answer, (int, float)):
                        return 1.0 if abs(predicted_val - correct_answer) < 1e-6 else 0.0
                    elif isinstance(correct_answer, Fraction):
                        correct_float = float(correct_answer)
                        return 1.0 if abs(predicted_val - correct_float) < 1e-6 else 0.0
                
                # String comparison for exact matches
                correct_str = str(correct_answer).lower()
                return 1.0 if predicted in correct_str or correct_str in predicted else 0.0
                
            elif answer_type == "symbolic":
                # For symbolic answers, use string matching with some flexibility
                correct_str = str(correct_answer).lower()
                
                # Handle π expressions
                if "π" in correct_str:
                    return 1.0 if "pi" in predicted or "π" in predicted else 0.0
                
                # Handle basic symbolic expressions
                return 1.0 if predicted == correct_str else 0.0
                
            else:  # verbal
                correct_str = str(correct_answer).lower()
                return 1.0 if predicted == correct_str else 0.0
                
        except Exception as e:
            logger.warning(f"Error calculating math correctness: {e}")
            return 0.0
    
    def _assess_step_accuracy(self, solution: str, expected_steps: List[str]) -> float:
        """Assess accuracy of mathematical solution steps"""
        if not solution or not expected_steps:
            return 0.0
        
        solution_lower = solution.lower()
        step_indicators = ['step', 'first', 'then', 'next', 'finally', 'therefore']
        
        # Check for step indicators
        indicator_count = sum(1 for indicator in step_indicators if indicator in solution_lower)
        
        # Check for mathematical operations
        math_operations = ['+', '-', '×', '*', '÷', '/', '=', 'solve', 'calculate', 'apply']
        operation_count = sum(1 for op in math_operations if op in solution_lower)
        
        # Calculate step accuracy based on structure and completeness
        total_expected_operations = len(expected_steps) * 2  # Expect structure + calculation per step
        actual_math_content = indicator_count + operation_count
        
        if total_expected_operations == 0:
            return 0.0
        
        step_score = min(actual_math_content / total_expected_operations, 1.0)
        return step_score
    
    def _assess_concept_usage(self, solution: str, expected_concepts: List[str]) -> float:
        """Assess usage of mathematical concepts in solution"""
        if not solution or not expected_concepts:
            return 0.0
        
        solution_lower = solution.lower()
        
        # Concept indicators
        concept_indicators = {
            'order_of_operations': ['order of operations', 'pemdas', 'bodmas'],
            'arithmetic': ['add', 'subtract', 'multiply', 'divide', 'plus', 'minus'],
            'fractions': ['fraction', 'numerator', 'denominator'],
            'percentages': ['percent', 'percentage', '%'],
            'linear_equations': ['solve for', 'equation', 'variable'],
            'algebraic_manipulation': ['subtract', 'add', 'divide both sides', 'multiply both sides'],
            'calculus': ['derivative', 'd/dx', 'power rule', 'integration'],
            'derivatives': ['derivative', 'd/dx', 'differentiate'],
            'power_rule': ['power rule', 'x^n'],
            'geometry': ['area', 'circumference', 'radius', 'diameter'],
            'circle_formulas': ['πr²', '2πr'],
            'sphere_volume': ['volume', 'sphere', '(4/3)πr³'],
            'word_problems': ['distance', 'time', 'speed', 'rate'],
            'rate_problems': ['miles per hour', 'per', 'rate'],
            'perimeter': ['perimeter', '2(length + width)'],
            'word_problems': ['let', 'define', 'equation', 'formula']
        }
        
        concept_scores = []
        for concept in expected_concepts:
            if concept in concept_indicators:
                indicators = concept_indicators[concept]
                concept_present = any(indicator in solution_lower for indicator in indicators)
                concept_scores.append(1.0 if concept_present else 0.0)
            else:
                # Generic concept matching
                concept_words = concept.split('_')
                concept_match = sum(1 for word in concept_words if word in solution_lower)
                concept_scores.append(concept_match / len(concept_words))
        
        return np.mean(concept_scores) if concept_scores else 0.0
    
    def validate_setup(self) -> bool:
        """Validate the mathematical reasoning benchmark setup"""
        try:
            # Check if math prompts are defined
            if not self.math_prompts:
                logger.error("No mathematical prompts defined")
                return False
            
            # Check prompt format
            for prompt_type, prompt in self.math_prompts.items():
                if 'SOLUTION:' not in prompt or 'ANSWER:' not in prompt:
                    logger.error(f"Invalid prompt format for {prompt_type}")
                    return False
            
            logger.info("Mathematical reasoning benchmark validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Mathematical reasoning benchmark validation failed: {e}")
            return False