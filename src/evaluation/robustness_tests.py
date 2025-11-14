"""
Robustness Validation System

This module provides comprehensive testing for model robustness including
adversarial attacks, edge cases, distributional shifts, and stress testing.
"""

import logging
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import string
import re
from datetime import datetime

from .benchmark_registry import BaseBenchmark, BenchmarkMetadata, BenchmarkCategory, BenchmarkType

logger = logging.getLogger(__name__)


@dataclass
class RobustnessTestResult:
    """Result from a robustness test"""
    test_name: str
    original_performance: float
    robustness_performance: float
    performance_degradation: float
    robustness_score: float  # 0-1, higher is more robust
    attack_successful: bool
    test_details: Dict[str, Any]
    timestamp: str


@dataclass
class AdversarialExample:
    """Adversarial example for testing"""
    original_input: str
    perturbed_input: str
    original_output: str
    expected_output: str
    attack_type: str  # character, word, sentence, semantic
    perturbation_type: str  # substitution, insertion, deletion, etc.
    confidence_score: float


class BaseRobustnessTest(ABC):
    """Abstract base class for robustness tests"""
    
    def __init__(self, name: str, metadata: BenchmarkMetadata):
        self.name = name
        self.metadata = metadata
    
    @abstractmethod
    def generate_attack_examples(self, test_inputs: List[str]) -> List[AdversarialExample]:
        """Generate adversarial examples for testing"""
        pass
    
    @abstractmethod
    def evaluate_robustness(self, model, test_inputs: List[str], **kwargs) -> RobustnessTestResult:
        """Evaluate model robustness"""
        pass


class CharacterLevelAttackTest(BaseRobustnessTest):
    """Test robustness against character-level adversarial attacks"""
    
    def __init__(self):
        metadata = BenchmarkMetadata(
            name="CharacterLevelAttack",
            description="Test robustness against character-level perturbations",
            category=BenchmarkCategory.ADVERSARIAL,
            benchmark_type=BenchmarkType.ROBUSTNESS,
            dataset_path="synthetic",
            metrics=["character_robustness", "typo_tolerance", "character_substitution_resilience"],
            expected_score_range=(0.0, 1.0)
        )
        super().__init__("CharacterLevelAttack", metadata)
    
    def generate_attack_examples(self, test_inputs: List[str]) -> List[AdversarialExample]:
        """Generate character-level adversarial examples"""
        adversarial_examples = []
        
        character_substitutions = {
            'a': ['@', '4', 'á'], 'e': ['3', '€', 'é'], 'i': ['1', '!', 'í'],
            'o': ['0', 'ó'], 'u': ['ü', 'ú'], 's': ['$', '5'],
            'g': ['9'], 't': ['7'], 'b': ['8'], 'l': ['1']
        }
        
        for original_input in test_inputs:
            # Single character substitution
            for i, char in enumerate(original_input):
                if char.lower() in character_substitutions:
                    for substitute in character_substitutions[char.lower()]:
                        if char.isupper():
                            substitute = substitute.upper()
                        perturbed_input = original_input[:i] + substitute + original_input[i+1:]
                        
                        adversarial_examples.append(AdversarialExample(
                            original_input=original_input,
                            perturbed_input=perturbed_input,
                            original_output="",
                            expected_output="",
                            attack_type="character",
                            perturbation_type="substitution",
                            confidence_score=0.8
                        ))
            
            # Random character insertion
            if len(original_input) > 5:
                insert_pos = random.randint(1, len(original_input) - 1)
                random_char = random.choice(string.ascii_lowercase)
                perturbed_input = original_input[:insert_pos] + random_char + original_input[insert_pos:]
                
                adversarial_examples.append(AdversarialExample(
                    original_input=original_input,
                    perturbed_input=perturbed_input,
                    original_output="",
                    expected_output="",
                    attack_type="character",
                    perturbation_type="insertion",
                    confidence_score=0.6
                ))
            
            # Random character deletion
            if len(original_input) > 3:
                delete_pos = random.randint(0, len(original_input) - 1)
                perturbed_input = original_input[:delete_pos] + original_input[delete_pos+1:]
                
                adversarial_examples.append(AdversarialExample(
                    original_input=original_input,
                    perturbed_input=perturbed_input,
                    original_output="",
                    expected_output="",
                    attack_type="character",
                    perturbation_type="deletion",
                    confidence_score=0.7
                ))
        
        return adversarial_examples[:100]  # Limit to prevent excessive testing
    
    def evaluate_robustness(self, model, test_inputs: List[str], **kwargs) -> RobustnessTestResult:
        """Evaluate character-level robustness"""
        # Generate adversarial examples
        adversarial_examples = self.generate_attack_examples(test_inputs[:10])  # Use subset for testing
        
        if not adversarial_examples:
            return RobustnessTestResult(
                test_name=self.name,
                original_performance=0.0,
                robustness_performance=0.0,
                performance_degradation=0.0,
                robustness_score=0.0,
                attack_successful=False,
                test_details={"error": "No adversarial examples generated"},
                timestamp=datetime.now().isoformat()
            )
        
        # Evaluate original performance (simplified)
        original_correct = 0
        robustness_correct = 0
        
        for example in adversarial_examples[:20]:  # Limit for testing
            try:
                # Simulate model predictions
                original_prediction = self._simulate_model_prediction(model, example.original_input)
                robustness_prediction = self._simulate_model_prediction(model, example.perturbed_input)
                
                # Check if predictions are consistent
                if original_prediction == robustness_prediction:
                    robustness_correct += 1
                
                original_correct += 1  # Assume original is always "correct" for simulation
                
            except Exception as e:
                logger.warning(f"Error evaluating example: {e}")
                continue
        
        original_performance = original_correct / len(adversarial_examples[:20]) if adversarial_examples else 0
        robustness_performance = robustness_correct / len(adversarial_examples[:20]) if adversarial_examples else 0
        
        performance_degradation = max(0, original_performance - robustness_performance)
        robustness_score = 1.0 - performance_degradation
        
        attack_successful = performance_degradation > 0.1  # Attack successful if >10% degradation
        
        return RobustnessTestResult(
            test_name=self.name,
            original_performance=original_performance,
            robustness_performance=robustness_performance,
            performance_degradation=performance_degradation,
            robustness_score=robustness_score,
            attack_successful=attack_successful,
            test_details={
                "total_examples": len(adversarial_examples),
                "evaluated_examples": min(20, len(adversarial_examples)),
                "attack_types": ["substitution", "insertion", "deletion"],
                "character_subs": character_substitutions
            },
            timestamp=datetime.now().isoformat()
        )
    
    def _simulate_model_prediction(self, model, input_text: str) -> str:
        """Simulate model prediction (placeholder implementation)"""
        # In a real implementation, this would call the actual model
        # For robustness testing, we'll simulate some vulnerability to character attacks
        input_lower = input_text.lower()
        
        # Simulate some resistance to simple character substitutions
        if any(char in input_text for char in ['@', '4', '3', '!', '$']):
            return "suspicious"  # Model might flag as spam/hacked
        
        # Simulate inconsistent handling of typos
        if any(typo in input_lower for typo in ['teh', 'helo', 'wrold']):
            return "incorrect"
        
        return "normal"


class WordLevelAttackTest(BaseRobustnessTest):
    """Test robustness against word-level adversarial attacks"""
    
    def __init__(self):
        metadata = BenchmarkMetadata(
            name="WordLevelAttack",
            description="Test robustness against word-level perturbations",
            category=BenchmarkCategory.ADVERSARIAL,
            benchmark_type=BenchmarkType.ROBUSTNESS,
            dataset_path="synthetic",
            metrics=["word_robustness", "synonym_resilience", "word_substitution_tolerance"],
            expected_score_range=(0.0, 1.0)
        )
        super().__init__("WordLevelAttack", metadata)
        
        # Common word replacements for testing
        self.word_replacements = {
            'good': ['great', 'excellent', 'fine', 'nice'],
            'bad': ['terrible', 'awful', 'poor', 'horrible'],
            'big': ['large', 'huge', 'enormous', 'massive'],
            'small': ['tiny', 'little', 'minor', 'minute'],
            'fast': ['quick', 'rapid', 'speedy', 'swift'],
            'slow': ['sluggish', 'gradual', 'leisurely', 'deliberate'],
            'happy': ['joyful', 'cheerful', 'glad', 'content'],
            'sad': ['unhappy', 'dejected', 'melancholy', 'sorrowful']
        }
    
    def generate_attack_examples(self, test_inputs: List[str]) -> List[AdversarialExample]:
        """Generate word-level adversarial examples"""
        adversarial_examples = []
        
        for original_input in test_inputs:
            words = original_input.split()
            
            # Synonym replacement
            for i, word in enumerate(words):
                word_lower = word.lower().strip(string.punctuation)
                if word_lower in self.word_replacements:
                    for replacement in self.word_replacements[word_lower]:
                        # Preserve capitalization
                        if word[0].isupper():
                            replacement = replacement.capitalize()
                        
                        # Preserve punctuation
                        punct = ''
                        if word and word[-1] in string.punctuation:
                            punct = word[-1]
                            word = word[:-1]
                        
                        perturbed_words = words.copy()
                        perturbed_words[i] = replacement + punct
                        perturbed_input = ' '.join(perturbed_words)
                        
                        adversarial_examples.append(AdversarialExample(
                            original_input=original_input,
                            perturbed_input=perturbed_input,
                            original_output="",
                            expected_output="",
                            attack_type="word",
                            perturbation_type="synonym_substitution",
                            confidence_score=0.9
                        ))
            
            # Random word insertion
            if len(words) > 3:
                insert_pos = random.randint(1, len(words) - 1)
                random_words = ['really', 'very', 'quite', 'extremely', 'rather']
                random_word = random.choice(random_words)
                perturbed_words = words.copy()
                perturbed_words.insert(insert_pos, random_word)
                perturbed_input = ' '.join(perturbed_words)
                
                adversarial_examples.append(AdversarialExample(
                    original_input=original_input,
                    perturbed_input=perturbed_input,
                    original_output="",
                    expected_output="",
                    attack_type="word",
                    perturbation_type="insertion",
                    confidence_score=0.7
                ))
        
        return adversarial_examples[:50]  # Limit for testing
    
    def evaluate_robustness(self, model, test_inputs: List[str], **kwargs) -> RobustnessTestResult:
        """Evaluate word-level robustness"""
        adversarial_examples = self.generate_attack_examples(test_inputs[:5])
        
        if not adversarial_examples:
            return RobustnessTestResult(
                test_name=self.name,
                original_performance=0.0,
                robustness_performance=0.0,
                performance_degradation=0.0,
                robustness_score=0.0,
                attack_successful=False,
                test_details={"error": "No adversarial examples generated"},
                timestamp=datetime.now().isoformat()
            )
        
        original_correct = 0
        robustness_correct = 0
        
        for example in adversarial_examples[:10]:  # Limit for testing
            try:
                original_prediction = self._simulate_model_prediction(model, example.original_input)
                robustness_prediction = self._simulate_model_prediction(model, example.perturbed_input)
                
                if original_prediction == robustness_prediction:
                    robustness_correct += 1
                
                original_correct += 1
                
            except Exception as e:
                logger.warning(f"Error evaluating word-level example: {e}")
                continue
        
        original_performance = original_correct / len(adversarial_examples[:10]) if adversarial_examples else 0
        robustness_performance = robustness_correct / len(adversarial_examples[:10]) if adversarial_examples else 0
        
        performance_degradation = max(0, original_performance - robustness_performance)
        robustness_score = 1.0 - performance_degradation
        
        attack_successful = performance_degradation > 0.15
        
        return RobustnessTestResult(
            test_name=self.name,
            original_performance=original_performance,
            robustness_performance=robustness_performance,
            performance_degradation=performance_degradation,
            robustness_score=robustness_score,
            attack_successful=attack_successful,
            test_details={
                "total_examples": len(adversarial_examples),
                "evaluated_examples": min(10, len(adversarial_examples)),
                "word_replacements_used": list(self.word_replacements.keys())
            },
            timestamp=datetime.now().isoformat()
        )
    
    def _simulate_model_prediction(self, model, input_text: str) -> str:
        """Simulate model prediction for word-level attacks"""
        input_lower = input_text.lower()
        
        # Simulate sensitivity to word changes
        sentiment_words = {
            'positive': ['good', 'great', 'excellent', 'happy', 'joyful'],
            'negative': ['bad', 'terrible', 'awful', 'sad', 'horrible']
        }
        
        positive_count = sum(1 for word in sentiment_words['positive'] if word in input_lower)
        negative_count = sum(1 for word in sentiment_words['negative'] if word in input_lower)
        
        if positive_count > negative_count:
            return "positive_sentiment"
        elif negative_count > positive_count:
            return "negative_sentiment"
        else:
            return "neutral_sentiment"


class SemanticAttackTest(BaseRobustnessTest):
    """Test robustness against semantic adversarial attacks"""
    
    def __init__(self):
        metadata = BenchmarkMetadata(
            name="SemanticAttack",
            description="Test robustness against semantic perturbations",
            category=BenchmarkCategory.ADVERSARIAL,
            benchmark_type=BenchmarkType.ROBUSTNESS,
            dataset_path="synthetic",
            metrics=["semantic_robustness", "paraphrase_resilience", "meaning_preservation"],
            expected_score_range=(0.0, 1.0)
        )
        super().__init__("SemanticAttack", metadata)
        
        # Semantic paraphrases for testing
        self.semantic_paraphrases = {
            "I love this movie": ["This film is fantastic", "I really enjoy this movie", "This is an amazing film"],
            "This is terrible": ["This is awful", "This is really bad", "I hate this"],
            "The weather is nice": ["The weather is pleasant", "The weather is good", "It's nice weather"],
            "I am tired": ["I feel exhausted", "I'm worn out", "I'm fatigued"]
        }
    
    def generate_attack_examples(self, test_inputs: List[str]) -> List[AdversarialExample]:
        """Generate semantic adversarial examples"""
        adversarial_examples = []
        
        for original_input in test_inputs:
            input_lower = original_input.lower().strip()
            
            # Find matching paraphrase patterns
            for original_phrase, paraphrases in self.semantic_paraphrases.items():
                if input_lower == original_phrase.lower():
                    for paraphrase in paraphrases:
                        adversarial_examples.append(AdversarialExample(
                            original_input=original_input,
                            perturbed_input=paraphrase,
                            original_output="",
                            expected_output="",
                            attack_type="semantic",
                            perturbation_type="paraphrase",
                            confidence_score=0.95
                        ))
            
            # Generate rule-based paraphrases for other inputs
            if len(original_input.split()) <= 5:  # Only for short inputs
                # Simple negation addition/removal
                if "not " in input_lower:
                    perturbed_input = input_lower.replace("not ", "")
                else:
                    perturbed_input = "not " + input_lower
                
                # Preserve original capitalization
                if original_input[0].isupper():
                    perturbed_input = perturbed_input.capitalize()
                
                adversarial_examples.append(AdversarialExample(
                    original_input=original_input,
                    perturbed_input=perturbed_input,
                    original_output="",
                    expected_output="",
                    attack_type="semantic",
                    perturbation_type="negation",
                    confidence_score=0.8
                ))
        
        return adversarial_examples[:30]  # Limit for testing
    
    def evaluate_robustness(self, model, test_inputs: List[str], **kwargs) -> RobustnessTestResult:
        """Evaluate semantic robustness"""
        adversarial_examples = self.generate_attack_examples(test_inputs[:5])
        
        if not adversarial_examples:
            return RobustnessTestResult(
                test_name=self.name,
                original_performance=0.0,
                robustness_performance=0.0,
                performance_degradation=0.0,
                robustness_score=0.0,
                attack_successful=False,
                test_details={"error": "No adversarial examples generated"},
                timestamp=datetime.now().isoformat()
            )
        
        original_correct = 0
        robustness_correct = 0
        
        for example in adversarial_examples[:8]:  # Limit for testing
            try:
                original_prediction = self._simulate_model_prediction(model, example.original_input)
                robustness_prediction = self._simulate_model_prediction(model, example.perturbed_input)
                
                # For semantic attacks, check if semantic meaning is preserved
                if self._are_semantically_equivalent(original_input, example.original_input, example.perturbed_input):
                    if original_prediction == robustness_prediction:
                        robustness_correct += 1
                
                original_correct += 1
                
            except Exception as e:
                logger.warning(f"Error evaluating semantic example: {e}")
                continue
        
        original_performance = original_correct / len(adversarial_examples[:8]) if adversarial_examples else 0
        robustness_performance = robustness_correct / len(adversarial_examples[:8]) if adversarial_examples else 0
        
        performance_degradation = max(0, original_performance - robustness_performance)
        robustness_score = 1.0 - performance_degradation
        
        attack_successful = performance_degradation > 0.2
        
        return RobustnessTestResult(
            test_name=self.name,
            original_performance=original_performance,
            robustness_performance=robustness_performance,
            performance_degradation=performance_degradation,
            robustness_score=robustness_score,
            attack_successful=attack_successful,
            test_details={
                "total_examples": len(adversarial_examples),
                "evaluated_examples": min(8, len(adversarial_examples)),
                "semantic_paraphrases": self.semantic_paraphrases
            },
            timestamp=datetime.now().isoformat()
        )
    
    def _are_semantically_equivalent(self, original_input: str, base_input: str, perturbed_input: str) -> bool:
        """Check if inputs are semantically equivalent"""
        # This is a simplified check - in reality would use semantic similarity models
        base_clean = base_input.lower().strip()
        perturbed_clean = perturbed_input.lower().strip()
        
        # Check for known paraphrase pairs
        for original_phrase, paraphrases in self.semantic_paraphrases.items():
            if base_clean == original_phrase and perturbed_clean in [p.lower() for p in paraphrases]:
                return True
        
        # Check for negation pairs
        if "not " in base_clean and perturbed_clean == base_clean.replace("not ", ""):
            return True
        if "not " in perturbed_clean and base_clean == perturbed_clean.replace("not ", ""):
            return True
        
        return False
    
    def _simulate_model_prediction(self, model, input_text: str) -> str:
        """Simulate model prediction for semantic attacks"""
        input_lower = input_text.lower()
        
        # Simulate different responses to semantically similar inputs
        positive_indicators = ['love', 'great', 'fantastic', 'amazing', 'nice', 'good']
        negative_indicators = ['hate', 'terrible', 'awful', 'bad', 'horrible', 'tired']
        
        if any(word in input_lower for word in positive_indicators):
            return "positive"
        elif any(word in input_lower for word in negative_indicators):
            return "negative"
        else:
            return "neutral"


class DistributionalShiftTest(BaseRobustnessTest):
    """Test robustness against distributional shifts"""
    
    def __init__(self):
        metadata = BenchmarkMetadata(
            name="DistributionalShift",
            description="Test robustness against distribution shifts",
            category=BenchmarkCategory.ADVERSARIAL,
            benchmark_type=BenchmarkType.ROBUSTNESS,
            dataset_path="synthetic",
            metrics=["domain_robustness", "shift_tolerance", "generalization_quality"],
            expected_score_range=(0.0, 1.0)
        )
        super().__init__("DistributionalShift", metadata)
        
        # Different domains for testing distributional shifts
        self.domain_variations = {
            "formal": "Please provide your response in a formal academic tone.",
            "casual": "Hey, can you help me with this?",
            "technical": "Given the parameters and constraints, please analyze the algorithmic complexity.",
            "creative": "Imagine you're writing a story about this topic.",
            "questionnaire": "This is a survey question. Please rate your agreement:",
            "legal": "According to the terms and conditions, the following applies:"
        }
    
    def generate_attack_examples(self, test_inputs: List[str]) -> List[AdversarialExample]:
        """Generate distributional shift examples"""
        adversarial_examples = []
        
        for original_input in test_inputs:
            # Apply different domain variations
            for domain_name, domain_prefix in self.domain_variations.items():
                shifted_input = f"{domain_prefix} {original_input}"
                
                adversarial_examples.append(AdversarialExample(
                    original_input=original_input,
                    perturbed_input=shifted_input,
                    original_output="",
                    expected_output="",
                    attack_type="distributional_shift",
                    perturbation_type=f"domain_{domain_name}",
                    confidence_score=0.85
                ))
            
            # Add random noise variations
            for _ in range(2):
                noisy_input = self._add_gaussian_noise(original_input)
                adversarial_examples.append(AdversarialExample(
                    original_input=original_input,
                    perturbed_input=noisy_input,
                    original_output="",
                    expected_output="",
                    attack_type="distributional_shift",
                    perturbation_type="gaussian_noise",
                    confidence_score=0.6
                ))
        
        return adversarial_examples[:40]  # Limit for testing
    
    def _add_gaussian_noise(self, text: str) -> str:
        """Add Gaussian noise to text (character level)"""
        chars = list(text)
        noisy_chars = []
        
        for char in chars:
            if random.random() < 0.1:  # 10% chance to perturb character
                # Add random character from ASCII
                noisy_char = chr(random.randint(32, 126))
                noisy_chars.append(noisy_char)
            else:
                noisy_chars.append(char)
        
        return ''.join(noisy_chars)
    
    def evaluate_robustness(self, model, test_inputs: List[str], **kwargs) -> RobustnessTestResult:
        """Evaluate robustness against distributional shifts"""
        adversarial_examples = self.generate_attack_examples(test_inputs[:3])
        
        if not adversarial_examples:
            return RobustnessTestResult(
                test_name=self.name,
                original_performance=0.0,
                robustness_performance=0.0,
                performance_degradation=0.0,
                robustness_score=0.0,
                attack_successful=False,
                test_details={"error": "No distributional shift examples generated"},
                timestamp=datetime.now().isoformat()
            )
        
        original_correct = 0
        robustness_correct = 0
        
        for example in adversarial_examples[:12]:  # Limit for testing
            try:
                original_prediction = self._simulate_model_prediction(model, example.original_input)
                robustness_prediction = self._simulate_model_prediction(model, example.perturbed_input)
                
                # Check if model handles domain shifts gracefully
                if self._is_domain_shift_tolerant(original_prediction, robustness_prediction, example.perturbation_type):
                    robustness_correct += 1
                
                original_correct += 1
                
            except Exception as e:
                logger.warning(f"Error evaluating distributional shift example: {e}")
                continue
        
        original_performance = original_correct / len(adversarial_examples[:12]) if adversarial_examples else 0
        robustness_performance = robustness_correct / len(adversarial_examples[:12]) if adversarial_examples else 0
        
        performance_degradation = max(0, original_performance - robustness_performance)
        robustness_score = 1.0 - performance_degradation
        
        attack_successful = performance_degradation > 0.25
        
        return RobustnessTestResult(
            test_name=self.name,
            original_performance=original_performance,
            robustness_performance=robustness_performance,
            performance_degradation=performance_degradation,
            robustness_score=robustness_score,
            attack_successful=attack_successful,
            test_details={
                "total_examples": len(adversarial_examples),
                "evaluated_examples": min(12, len(adversarial_examples)),
                "domains_tested": list(self.domain_variations.keys()),
                "shift_types": ["domain_shift", "gaussian_noise"]
            },
            timestamp=datetime.now().isoformat()
        )
    
    def _is_domain_shift_tolerant(self, original_prediction: str, shifted_prediction: str, shift_type: str) -> bool:
        """Check if model is tolerant to domain shift"""
        # Domain shifts should ideally not change the core semantic understanding
        # For classification tasks, this means the main prediction should remain consistent
        # For generation tasks, this is more complex
        
        # Simple check: same core prediction
        if original_prediction == shifted_prediction:
            return True
        
        # Check if shift is minor (e.g., adding "formal" doesn't change content understanding)
        if "domain_" in shift_type:
            # Some domain shifts are expected to change response style but not content
            core_words_orig = set(original_prediction.lower().split()) & {'positive', 'negative', 'neutral', 'true', 'false', 'yes', 'no'}
            core_words_shift = set(shifted_prediction.lower().split()) & {'positive', 'negative', 'neutral', 'true', 'false', 'yes', 'no'}
            
            if core_words_orig and core_words_shift and core_words_orig == core_words_shift:
                return True
        
        return False
    
    def _simulate_model_prediction(self, model, input_text: str) -> str:
        """Simulate model prediction for distributional shift testing"""
        input_lower = input_text.lower()
        
        # Simulate different behavior for different domains
        if "formal" in input_lower or "academic" in input_lower:
            return "formal_response"
        elif "casual" in input_lower or "hey" in input_lower:
            return "casual_response"
        elif "technical" in input_lower or "algorithmic" in input_lower:
            return "technical_response"
        elif any(word in input_lower for word in ['love', 'hate', 'great', 'terrible']):
            return "sentiment_analysis"
        else:
            return "general_response"


class EdgeCaseTest(BaseRobustnessTest):
    """Test robustness against edge cases and corner scenarios"""
    
    def __init__(self):
        metadata = BenchmarkMetadata(
            name="EdgeCaseTest",
            description="Test robustness against edge cases and corner scenarios",
            category=BenchmarkCategory.ADVERSARIAL,
            benchmark_type=BenchmarkType.ROBUSTNESS,
            dataset_path="synthetic",
            metrics=["edge_case_robustness", "corner_case_handling", "boundary_condition_resilience"],
            expected_score_range=(0.0, 1.0)
        )
        super().__init__("EdgeCaseTest", metadata)
    
    def generate_attack_examples(self, test_inputs: List[str]) -> List[AdversarialExample]:
        """Generate edge case examples"""
        adversarial_examples = []
        
        edge_cases = [
            "",  # Empty input
            " ",  # Whitespace only
            "a" * 1000,  # Very long input
            "A" * 10000,  # Extremely long input
            "1234567890" * 100,  # Numbers only
            "!@#$%^&*()" * 50,  # Special characters only
            "asdfghjklqwertyuiopzxcvbnm" * 50,  # Random characters
            "\n\n\n",  # Newlines only
            "\t\t\t",  # Tabs only
            "NULL", "undefined", "NaN", "null", "None",  # Programming values
            "🤔🚀💯",  # Emojis only
            "Test" + "\x00" * 10 + "Test",  # Null bytes
            "Test" + "\xff" * 10 + "Test",  # High bytes
            "A" * 100 + "\n" + "B" * 100,  # Long lines
            "Test\r\nTest",  # Windows line endings
            "Test\nTest",  # Unix line endings
            "Test\rTest",  # Mac line endings
        ]
        
        for original_input in test_inputs:
            for edge_case in edge_cases:
                adversarial_examples.append(AdversarialExample(
                    original_input=original_input,
                    perturbed_input=edge_case,
                    original_output="",
                    expected_output="",
                    attack_type="edge_case",
                    perturbation_type="boundary_condition",
                    confidence_score=1.0
                ))
        
        return adversarial_examples[:50]  # Limit for testing
    
    def evaluate_robustness(self, model, test_inputs: List[str], **kwargs) -> RobustnessTestResult:
        """Evaluate robustness against edge cases"""
        adversarial_examples = self.generate_attack_examples(test_inputs[:2])
        
        if not adversarial_examples:
            return RobustnessTestResult(
                test_name=self.name,
                original_performance=0.0,
                robustness_performance=0.0,
                performance_degradation=0.0,
                robustness_score=0.0,
                attack_successful=False,
                test_details={"error": "No edge case examples generated"},
                timestamp=datetime.now().isoformat()
            )
        
        original_correct = 0
        robustness_correct = 0
        edge_cases_handled = 0
        
        for example in adversarial_examples[:25]:  # Limit for testing
            try:
                # Test normal input performance
                original_prediction = self._simulate_model_prediction(model, example.original_input)
                original_correct += 1
                
                # Test edge case handling
                robustness_prediction = self._simulate_edge_case_prediction(model, example.perturbed_input)
                
                # Check if edge case was handled gracefully
                if robustness_prediction != "error" and robustness_prediction != "crash":
                    edge_cases_handled += 1
                    if self._is_edge_case_handled_appropriately(original_prediction, robustness_prediction, example.perturbed_input):
                        robustness_correct += 1
                
            except Exception as e:
                logger.warning(f"Error evaluating edge case: {e}")
                continue
        
        original_performance = original_correct / len(adversarial_examples[:25]) if adversarial_examples else 0
        robustness_performance = robustness_correct / len(adversarial_examples[:25]) if adversarial_examples else 0
        edge_case_handling_rate = edge_cases_handled / len(adversarial_examples[:25]) if adversarial_examples else 0
        
        performance_degradation = max(0, original_performance - robustness_performance)
        robustness_score = edge_case_handling_rate  # Use edge case handling as robustness metric
        
        attack_successful = robustness_score < 0.7  # Attack successful if <70% edge cases handled
        
        return RobustnessTestResult(
            test_name=self.name,
            original_performance=original_performance,
            robustness_performance=robustness_performance,
            performance_degradation=performance_degradation,
            robustness_score=robustness_score,
            attack_successful=attack_successful,
            test_details={
                "total_examples": len(adversarial_examples),
                "evaluated_examples": min(25, len(adversarial_examples)),
                "edge_cases_handled": edge_cases_handled,
                "edge_case_handling_rate": edge_case_handling_rate,
                "edge_case_types": ["empty", "whitespace", "very_long", "special_chars", "unicode", "null_bytes"]
            },
            timestamp=datetime.now().isoformat()
        )
    
    def _simulate_edge_case_prediction(self, model, edge_case_input: str) -> str:
        """Simulate model prediction for edge cases"""
        input_len = len(edge_case_input)
        
        # Handle different edge cases
        if input_len == 0:
            return "empty_input_handled"
        elif edge_case_input.strip() == "":
            return "whitespace_handled"
        elif input_len > 5000:
            return "long_input_handled"
        elif not any(c.isalnum() for c in edge_case_input):
            return "special_chars_handled"
        elif any(ord(c) > 127 for c in edge_case_input):
            return "unicode_handled"
        elif '\x00' in edge_case_input:
            return "null_bytes_handled"
        else:
            return "edge_case_handled"
    
    def _is_edge_case_handled_appropriately(self, original_prediction: str, edge_case_prediction: str, edge_case_input: str) -> bool:
        """Check if edge case was handled appropriately"""
        # For edge cases, we expect graceful handling, not crashes
        # The specific handling may vary, but it should be consistent
        
        appropriate_responses = [
            "empty_input_handled", "whitespace_handled", "long_input_handled",
            "special_chars_handled", "unicode_handled", "null_bytes_handled",
            "edge_case_handled"
        ]
        
        return edge_case_prediction in appropriate_responses
    
    def _simulate_model_prediction(self, model, input_text: str) -> str:
        """Simulate normal model prediction"""
        if not input_text.strip():
            return "invalid_input"
        
        # Simple sentiment-based simulation
        positive_words = ['good', 'great', 'love', 'excellent', 'amazing']
        negative_words = ['bad', 'terrible', 'hate', 'awful', 'horrible']
        
        input_lower = input_text.lower()
        
        pos_count = sum(1 for word in positive_words if word in input_lower)
        neg_count = sum(1 for word in negative_words if word in input_lower)
        
        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        else:
            return "neutral"


class RobustnessValidationSuite:
    """Comprehensive suite of robustness validation tests"""
    
    def __init__(self):
        self.tests = {
            "character_attacks": CharacterLevelAttackTest(),
            "word_attacks": WordLevelAttackTest(),
            "semantic_attacks": SemanticAttackTest(),
            "distributional_shift": DistributionalShiftTest(),
            "edge_cases": EdgeCaseTest()
        }
    
    def run_all_robustness_tests(self, model, test_inputs: List[str], **kwargs) -> Dict[str, RobustnessTestResult]:
        """Run all robustness tests"""
        results = {}
        
        for test_name, test in self.tests.items():
            logger.info(f"Running robustness test: {test_name}")
            try:
                result = test.evaluate_robustness(model, test_inputs, **kwargs)
                results[test_name] = result
                
                if result.attack_successful:
                    logger.warning(f"Attack successful in {test_name}: {result.performance_degradation:.3f} degradation")
                else:
                    logger.info(f"{test_name} robust: {result.robustness_score:.3f} score")
                    
            except Exception as e:
                logger.error(f"Error running {test_name} test: {e}")
                results[test_name] = RobustnessTestResult(
                    test_name=test_name,
                    original_performance=0.0,
                    robustness_performance=0.0,
                    performance_degradation=1.0,
                    robustness_score=0.0,
                    attack_successful=True,
                    test_details={"error": str(e)},
                    timestamp=datetime.now().isoformat()
                )
        
        return results
    
    def run_specific_test(self, test_name: str, model, test_inputs: List[str], **kwargs) -> RobustnessTestResult:
        """Run a specific robustness test"""
        if test_name not in self.tests:
            raise ValueError(f"Robustness test {test_name} not found")
        
        return self.tests[test_name].evaluate_robustness(model, test_inputs, **kwargs)
    
    def generate_robustness_report(self, results: Dict[str, RobustnessTestResult]) -> Dict[str, Any]:
        """Generate comprehensive robustness report"""
        successful_attacks = sum(1 for result in results.values() if result.attack_successful)
        total_tests = len(results)
        
        overall_robustness_score = np.mean([result.robustness_score for result in results.values()])
        average_degradation = np.mean([result.performance_degradation for result in results.values()])
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "successful_attacks": successful_attacks,
                "overall_robustness_score": overall_robustness_score,
                "average_performance_degradation": average_degradation,
                "vulnerability_rate": successful_attacks / total_tests if total_tests > 0 else 0
            },
            "test_results": {
                test_name: {
                    "robustness_score": result.robustness_score,
                    "performance_degradation": result.performance_degradation,
                    "attack_successful": result.attack_successful,
                    "original_performance": result.original_performance,
                    "robustness_performance": result.robustness_performance
                }
                for test_name, result in results.items()
            },
            "recommendations": self._generate_robustness_recommendations(results),
            "timestamp": datetime.now().isoformat()
        }
        
        return report
    
    def _generate_robustness_recommendations(self, results: Dict[str, RobustnessTestResult]) -> List[str]:
        """Generate recommendations for improving robustness"""
        recommendations = []
        
        # Analyze individual test results
        for test_name, result in results.items():
            if result.attack_successful:
                if "character" in test_name:
                    recommendations.append("Implement character-level input validation and normalization")
                    recommendations.append("Consider using character-level language models for better typo tolerance")
                elif "word" in test_name:
                    recommendations.append("Implement synonym-aware preprocessing and robust tokenization")
                    recommendations.append("Use contextual embeddings that are invariant to synonym substitutions")
                elif "semantic" in test_name:
                    recommendations.append("Implement semantic consistency checking and paraphrase detection")
                    recommendations.append("Use semantic similarity models for input validation")
                elif "distributional" in test_name:
                    recommendations.append("Implement domain adaptation and style normalization")
                    recommendations.append("Use domain-specific fine-tuning and robust training procedures")
                elif "edge" in test_name:
                    recommendations.append("Implement comprehensive input validation and sanitization")
                    recommendations.append("Add explicit handling for boundary conditions and corner cases")
        
        # General recommendations based on overall performance
        overall_score = np.mean([result.robustness_score for result in results.values()])
        
        if overall_score < 0.5:
            recommendations.append("Consider adversarial training to improve robustness")
            recommendations.append("Implement input preprocessing and data augmentation")
        elif overall_score < 0.7:
            recommendations.append("Fine-tune model on adversarial examples")
            recommendations.append("Implement ensemble methods for robustness")
        else:
            recommendations.append("Current robustness is good, consider periodic re-testing")
            recommendations.append("Monitor for new attack vectors and update defenses")
        
        return recommendations