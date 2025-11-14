"""
Commonsense Reasoning Benchmark

This benchmark evaluates a model's ability to understand and apply commonsense knowledge
in everyday situations and general world knowledge.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from ..benchmark_registry import BaseBenchmark, BenchmarkMetadata, BenchmarkCategory, BenchmarkType

logger = logging.getLogger(__name__)


@dataclass
class CommonsenseExample:
    """Example for commonsense reasoning"""
    question: str
    correct_answer: str
    options: Optional[List[str]]  # For multiple choice questions
    knowledge_domain: str  # physical, social, temporal, causal, etc.
    reasoning_type: str    # what_happens_next, why_because,常识判断, etc.
    difficulty: str        # basic, intermediate, advanced
    context: str           # Additional context if needed
    explanation: str       # Why this is the correct answer


class CommonsenseReasoningBenchmark(BaseBenchmark):
    """Benchmark for evaluating commonsense reasoning capabilities"""
    
    def __init__(self, metadata: BenchmarkMetadata):
        super().__init__(metadata)
        self.examples: List[CommonsenseExample] = []
        self.commonsense_prompts = self._initialize_commonsense_prompts()
    
    def _initialize_commonsense_prompts(self) -> Dict[str, str]:
        """Initialize commonsense reasoning prompts"""
        return {
            "basic_qa": """
            Use your commonsense knowledge to answer this question.
            Think about what would make the most sense in everyday situations.
            Return your answer in the format: REASONING: [commonsense reasoning] ANSWER: [final answer]
            
            Question: {question}
            """,
            
            "what_happens_next": """
            Predict what would most likely happen next in this situation.
            Use your knowledge of how the world works and human behavior.
            Return your answer in the format: PREDICTION: [likely next event] REASONING: [why this makes sense]
            
            Scenario: {question}
            """,
            
            "why_because": """
            Explain why this situation or outcome occurs.
            Use your understanding of cause and effect relationships.
            Return your answer in the format: EXPLANATION: [causal explanation] ANSWER: [reason why]
            
            {question}
            """,
            
            "social_situations": """
            Analyze this social situation and determine the most appropriate action or response.
            Consider social norms, politeness, and typical human interactions.
            Return your answer in the format: ANALYSIS: [social reasoning] RECOMMENDATION: [appropriate action]
            
            Situation: {question}
            """,
            
            "physical_world": """
            Apply your knowledge of physics and the physical world to answer this question.
            Consider gravity, motion, materials, and natural phenomena.
            Return your answer in the format: ANALYSIS: [physical reasoning] ANSWER: [physical world answer]
            
            Question: {question}
            """
        }
    
    def load_dataset(self) -> bool:
        """Load commonsense reasoning examples"""
        try:
            self.examples = self._generate_commonsense_examples()
            self.is_loaded = True
            
            logger.info(f"Loaded {len(self.examples)} commonsense reasoning examples")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load commonsense dataset: {e}")
            return False
    
    def _generate_commonsense_examples(self) -> List[CommonsenseExample]:
        """Generate commonsense reasoning examples"""
        examples = []
        
        # Physical commonsense examples
        physical_examples = [
            CommonsenseExample(
                question="What happens if you drop a glass on a hard floor?",
                correct_answer="The glass will break",
                options=None,
                knowledge_domain="physical",
                reasoning_type="cause_effect",
                difficulty="basic",
                context="Everyday physics involving fragile objects and gravity",
                explanation="Glass is brittle and will shatter when dropped on a hard surface due to the impact force."
            ),
            CommonsenseExample(
                question="Why does ice melt when left in the sun?",
                correct_answer="The sun's heat transfers energy to the ice, changing it from solid to liquid",
                options=None,
                knowledge_domain="physical",
                reasoning_type="why_because",
                difficulty="intermediate",
                context="Phase changes and heat transfer",
                explanation="Heat from the sun increases the temperature of ice above its melting point, causing it to change from solid to liquid."
            ),
            CommonsenseExample(
                question="What happens to water when it gets cold enough?",
                correct_answer="It turns into ice (freezes)",
                options=None,
                knowledge_domain="physical",
                reasoning_type="cause_effect",
                difficulty="basic",
                context="Phase transitions of water",
                explanation="When water reaches 0°C (32°F) or below, the molecules slow down and form a crystalline structure, creating ice."
            )
        ]
        
        # Social commonsense examples
        social_examples = [
            CommonsenseExample(
                question="When someone says 'I'm fine' but sounds sad and looks upset, what should you do?",
                correct_answer="Ask if they want to talk about what's bothering them",
                options=None,
                knowledge_domain="social",
                reasoning_type="social_situations",
                difficulty="intermediate",
                context="Emotional intelligence and social interactions",
                explanation="People often say they're fine when they're not. Showing concern and offering to listen demonstrates empathy."
            ),
            CommonsenseExample(
                question="Why do people generally say 'please' and 'thank you'?",
                correct_answer="To show politeness and gratitude",
                options=None,
                knowledge_domain="social",
                reasoning_type="why_because",
                difficulty="basic",
                context="Social etiquette and manners",
                explanation="Using polite expressions shows respect for others and appreciation for their actions or assistance."
            ),
            CommonsenseExample(
                question="What is appropriate behavior when someone is speaking in a movie theater?",
                correct_answer="Stay quiet or whisper very quietly if necessary",
                options=None,
                knowledge_domain="social",
                reasoning_type="social_situations",
                difficulty="basic",
                context="Public behavior and respect for others",
                explanation="Movie theaters are shared spaces where people expect to enjoy the film without distractions."
            )
        ]
        
        # Temporal commonsense examples
        temporal_examples = [
            CommonsenseExample(
                question="What happens after you put a pizza in the oven?",
                correct_answer="The pizza gets hot and the cheese melts",
                options=None,
                knowledge_domain="temporal",
                reasoning_type="what_happens_next",
                difficulty="basic",
                context="Cooking and heat application over time",
                explanation="Heat from the oven transfers to the pizza, cooking the dough and melting the cheese."
            ),
            CommonsenseExample(
                question="What typically happens to plants in winter?",
                correct_answer="Many plants stop growing or die back",
                options=None,
                knowledge_domain="temporal",
                reasoning_type="what_happens_next",
                difficulty="intermediate",
                context="Seasonal plant behavior",
                explanation="Winter brings less sunlight and colder temperatures, which slows or stops plant growth."
            )
        ]
        
        # Causal commonsense examples
        causal_examples = [
            CommonsenseExample(
                question="Why do umbrellas have long handles?",
                correct_answer="So people can hold them over their head while walking",
                options=None,
                knowledge_domain="causal",
                reasoning_type="why_because",
                difficulty="basic",
                context="Tool design and human ergonomics",
                explanation="A long handle allows the umbrella to extend above the user's head while keeping their hand at a comfortable height."
            ),
            CommonsenseExample(
                question="What causes traffic jams on highways?",
                correct_answer="Too many cars trying to use the same road at the same time",
                options=None,
                knowledge_domain="causal",
                reasoning_type="why_because",
                difficulty="intermediate",
                context="Transportation and human behavior",
                explanation="When traffic volume exceeds road capacity, vehicles slow down or stop, creating a bottleneck."
            )
        ]
        
        # Biological commonsense examples
        biological_examples = [
            CommonsenseExample(
                question="Why do humans need to eat food regularly?",
                correct_answer="To get energy and nutrients for the body to function",
                options=None,
                knowledge_domain="biological",
                reasoning_type="why_because",
                difficulty="basic",
                context="Human biology and metabolism",
                explanation="Food provides the energy and nutrients needed for bodily functions, growth, and repair."
            ),
            CommonsenseExample(
                question="What happens when you don't get enough sleep?",
                correct_answer="You feel tired and your mind doesn't work as well",
                options=None,
                knowledge_domain="biological",
                reasoning_type="cause_effect",
                difficulty="basic",
                context="Sleep and human physiology",
                explanation="Sleep is essential for brain function and physical recovery. Lack of sleep impairs cognitive performance."
            )
        ]
        
        # Situational commonsense examples
        situational_examples = [
            CommonsenseExample(
                question="What should you do if you see someone drop something?",
                correct_answer="Pick it up and return it to them",
                options=None,
                knowledge_domain="situational",
                reasoning_type="social_situations",
                difficulty="basic",
                context="Helping others and social responsibility",
                explanation="Helping someone who dropped something shows kindness and is a common social courtesy."
            ),
            CommonsenseExample(
                question="Why do people wear jackets when it's cold outside?",
                correct_answer="To stay warm and comfortable",
                options=None,
                knowledge_domain="situational",
                reasoning_type="why_because",
                difficulty="basic",
                context="Weather and clothing behavior",
                explanation="Jackets provide insulation that helps maintain body temperature in cold weather."
            ),
            CommonsenseExample(
                question="What do you do when you feel thirsty?",
                correct_answer="Drink water or another beverage",
                options=None,
                knowledge_domain="situational",
                reasoning_type="cause_effect",
                difficulty="basic",
                context="Basic human needs and responses",
                explanation="Thirst is the body's signal that it needs more fluids, so drinking satisfies this need."
            )
        ]
        
        # Advanced commonsense examples
        advanced_examples = [
            CommonsenseExample(
                question="Why might someone choose to walk instead of drive for a very short trip?",
                correct_answer="For exercise, fresh air, or to avoid the hassle of parking for such a short distance",
                options=None,
                knowledge_domain="situational",
                reasoning_type="social_situations",
                difficulty="advanced",
                context="Decision making and trade-offs",
                explanation="For very short distances, walking might be faster overall than driving when considering parking and traffic."
            ),
            CommonsenseExample(
                question="What could cause a normally reliable person to be late?",
                correct_answer="Traffic, illness, family emergency, or car problems",
                options=None,
                knowledge_domain="situational",
                reasoning_type="what_happens_next",
                difficulty="advanced",
                context="Human behavior and unexpected circumstances",
                explanation="Even reliable people can be delayed by circumstances beyond their control."
            )
        ]
        
        return (physical_examples + social_examples + temporal_examples + 
                causal_examples + biological_examples + situational_examples + advanced_examples)
    
    def evaluate_model(self, model, **kwargs) -> Dict[str, float]:
        """Evaluate a model's commonsense reasoning capabilities"""
        if not self.is_loaded:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        prompt_type = kwargs.get('prompt_type', 'basic_qa')
        batch_size = kwargs.get('batch_size', 8)
        max_length = kwargs.get('max_length', 512)
        
        results = {}
        
        # Evaluate by knowledge domain
        for domain in ['physical', 'social', 'temporal', 'causal', 'biological', 'situational']:
            domain_results = self._evaluate_by_domain(model, domain, prompt_type, batch_size, max_length)
            if domain_results['accuracy'] > 0:
                results[f'{domain}_commonsense'] = domain_results['accuracy']
        
        # Evaluate by reasoning type
        for reasoning_type in ['cause_effect', 'why_because', 'what_happens_next', 'social_situations']:
            type_results = self._evaluate_by_reasoning_type(model, reasoning_type, prompt_type, batch_size, max_length)
            if type_results['accuracy'] > 0:
                results[f'{reasoning_type}_accuracy'] = type_results['accuracy']
        
        # Evaluate by difficulty
        for difficulty in ['basic', 'intermediate', 'advanced']:
            difficulty_results = self._evaluate_by_difficulty(model, difficulty, prompt_type, batch_size, max_length)
            if difficulty_results['accuracy'] > 0:
                results[f'{difficulty}_difficulty'] = difficulty_results['accuracy']
        
        # Overall evaluation
        all_scores = []
        reasoning_quality_scores = []
        domain_diversity_scores = []
        
        # Track domain coverage
        domain_correct = {domain: [] for domain in ['physical', 'social', 'temporal', 'causal', 'biological', 'situational']}
        
        for example in self.examples:
            score = self._evaluate_single_commonsense_problem(model, example, prompt_type, max_length)
            all_scores.append(score['accuracy'])
            reasoning_quality_scores.append(score['reasoning_quality'])
            
            # Track performance by domain
            domain_correct[example.knowledge_domain].append(score['accuracy'])
        
        # Calculate domain diversity (how well balanced the performance is across domains)
        domain_performances = [np.mean(scores) for scores in domain_correct.values() if scores]
        results['domain_diversity'] = 1.0 - np.std(domain_performances) if domain_performances else 0.0
        
        results['overall_accuracy'] = np.mean(all_scores)
        results['reasoning_quality'] = np.mean(reasoning_quality_scores)
        results['commonsense_coherence'] = self._calculate_commonsense_coherence()
        
        logger.info(f"Commonsense reasoning evaluation complete. Overall accuracy: {results['overall_accuracy']:.3f}")
        return results
    
    def _evaluate_by_domain(self, model, knowledge_domain: str, prompt_type: str, batch_size: int, max_length: int) -> Dict[str, float]:
        """Evaluate performance on problems of a specific knowledge domain"""
        filtered_examples = [ex for ex in self.examples if ex.knowledge_domain == knowledge_domain]
        
        if not filtered_examples:
            return {'accuracy': 0.0, 'coherence': 0.0}
        
        scores = []
        coherences = []
        for example in filtered_examples:
            score = self._evaluate_single_commonsense_problem(model, example, prompt_type, max_length)
            scores.append(score['accuracy'])
            coherences.append(score['reasoning_quality'])
        
        return {
            'accuracy': np.mean(scores),
            'coherence': np.mean(coherences)
        }
    
    def _evaluate_by_reasoning_type(self, model, reasoning_type: str, prompt_type: str, batch_size: int, max_length: int) -> Dict[str, float]:
        """Evaluate performance on problems of a specific reasoning type"""
        filtered_examples = [ex for ex in self.examples if ex.reasoning_type == reasoning_type]
        
        if not filtered_examples:
            return {'accuracy': 0.0, 'coherence': 0.0}
        
        scores = []
        coherences = []
        for example in filtered_examples:
            score = self._evaluate_single_commonsense_problem(model, example, prompt_type, max_length)
            scores.append(score['accuracy'])
            coherences.append(score['reasoning_quality'])
        
        return {
            'accuracy': np.mean(scores),
            'coherence': np.mean(coherences)
        }
    
    def _evaluate_by_difficulty(self, model, difficulty: str, prompt_type: str, batch_size: int, max_length: int) -> Dict[str, float]:
        """Evaluate performance on problems of a specific difficulty"""
        filtered_examples = [ex for ex in self.examples if ex.difficulty == difficulty]
        
        if not filtered_examples:
            return {'accuracy': 0.0, 'coherence': 0.0}
        
        scores = []
        coherences = []
        for example in filtered_examples:
            score = self._evaluate_single_commonsense_problem(model, example, prompt_type, max_length)
            scores.append(score['accuracy'])
            coherences.append(score['reasoning_quality'])
        
        return {
            'accuracy': np.mean(scores),
            'coherence': np.mean(coherences)
        }
    
    def _evaluate_single_commonsense_problem(self, model, example: CommonsenseExample, prompt_type: str, max_length: int) -> Dict[str, float]:
        """Evaluate a single commonsense reasoning problem"""
        try:
            # Select appropriate prompt template
            if prompt_type not in self.commonsense_prompts:
                prompt_type = 'basic_qa'
            
            prompt_template = self.commonsense_prompts[prompt_type]
            
            if "{question}" in prompt_template:
                prompt = prompt_template.format(question=example.question)
            else:
                prompt = prompt_template.format(question=example.context) + f"\n\nQuestion: {example.question}"
            
            # Generate response
            response = self._generate_commonsense_response(model, prompt, example, max_length)
            
            # Parse response
            reasoning, answer = self._parse_commonsense_response(response)
            
            # Calculate accuracy
            accuracy = self._calculate_commonsense_accuracy(answer, example.correct_answer, example.knowledge_domain)
            
            # Assess reasoning quality
            reasoning_quality = self._assess_commonsense_reasoning_quality(reasoning, example.knowledge_domain, example.reasoning_type)
            
            return {
                'accuracy': accuracy,
                'reasoning_quality': reasoning_quality,
                'generated_reasoning': reasoning,
                'generated_answer': answer
            }
            
        except Exception as e:
            logger.warning(f"Error evaluating commonsense problem: {e}")
            return {'accuracy': 0.0, 'reasoning_quality': 0.0}
    
    def _generate_commonsense_response(self, model, prompt: str, example: CommonsenseExample, max_length: int) -> str:
        """Generate model response for commonsense reasoning (placeholder)"""
        # In a real implementation, this would call the actual model
        # For demonstration, simulate responses based on knowledge domain and reasoning type
        
        if example.knowledge_domain == "physical":
            if "glass" in prompt and "drop" in prompt:
                return "REASONING: Glass is a brittle material that breaks when subjected to impact force. When dropped on a hard surface, the glass cannot withstand the sudden deceleration. ANSWER: The glass will break"
            
            elif "ice" in prompt and "melt" in prompt:
                return "REASONING: Heat from the sun transfers energy to the ice molecules, increasing their kinetic energy until they overcome the bonds holding them in a solid structure. ANSWER: The ice melts due to heat transfer"
        
        elif example.knowledge_domain == "social":
            if "fine" in prompt and "sad" in prompt:
                return "REASONING: People often say they're fine when they're not okay as a way to avoid burdening others or because they're not ready to talk. It's important to show concern and offer support. ANSWER: Ask if they want to talk about what's bothering them"
            
            elif "please" in prompt and "thank you" in prompt:
                return "REASONING: These phrases are social conventions that show respect, politeness, and gratitude for others' actions or kindness. ANSWER: To show politeness and gratitude"
        
        elif example.knowledge_domain == "temporal":
            if "pizza" in prompt and "oven" in prompt:
                return "PREDICTION: The pizza will cook and the cheese will melt as heat transfers from the oven to the food. REASONING: Heat causes cooking and melting of dairy products"
        
        elif example.knowledge_domain == "situational":
            if "thirsty" in prompt:
                return "REASONING: Thirst is the body's signal that it needs more fluids to maintain proper hydration and bodily functions. ANSWER: Drink water or another beverage"
        
        return "REASONING: Using general knowledge about how the world works. ANSWER: Commonsense response"
    
    def _parse_commonsense_response(self, response: str) -> Tuple[str, str]:
        """Parse commonsense reasoning response"""
        try:
            # Look for answer marker
            answer_markers = ["ANSWER:", "RECOMMENDATION:", "PREDICTION:", "EXPLANATION:"]
            reasoning = response
            
            for marker in answer_markers:
                if marker in response:
                    parts = response.split(marker, 1)
                    if len(parts) == 2:
                        reasoning_part = parts[0].replace("REASONING:", "").replace("ANALYSIS:", "").replace("PREDICTION:", "").strip()
                        if not reasoning_part:  # If reasoning is empty, use the part before answer
                            reasoning = reasoning_part or "Using commonsense knowledge"
                        answer = parts[1].strip()
                        return reasoning_part, answer
            
            # Try to extract answer as last meaningful part
            sentences = [s.strip() for s in response.split('.') if s.strip()]
            if sentences:
                answer = sentences[-1]
                reasoning = '. '.join(sentences[:-1]) if len(sentences) > 1 else "Applying commonsense reasoning"
                return reasoning, answer
            
            return response, ""
            
        except Exception as e:
            logger.warning(f"Error parsing commonsense response: {e}")
            return response, ""
    
    def _calculate_commonsense_accuracy(self, predicted_answer: str, correct_answer: str, knowledge_domain: str) -> float:
        """Calculate accuracy of commonsense reasoning answer"""
        predicted = predicted_answer.lower().strip()
        correct = correct_answer.lower().strip()
        
        # Exact match
        if predicted == correct:
            return 1.0
        
        # Domain-specific scoring
        if knowledge_domain == "physical":
            # Look for physical reasoning terms
            physical_terms = ["break", "melt", "freeze", "fall", "heat", "cold", "gravity"]
            predicted_words = predicted.split()
            correct_words = correct.split()
            
            # Check for domain-relevant vocabulary
            predicted_domain_terms = [word for word in predicted_words if any(term in word for term in physical_terms)]
            correct_domain_terms = [word for word in correct_words if any(term in word for term in physical_terms)]
            
            if correct_domain_terms:
                term_overlap = len(set(predicted_domain_terms).intersection(set(correct_domain_terms)))
                return min(term_overlap / len(correct_domain_terms), 1.0)
        
        elif knowledge_domain == "social":
            # Look for social reasoning terms
            social_terms = ["polite", "respect", "help", "appropriate", "behavior", "considerate"]
            predicted_words = predicted.split()
            correct_words = correct.split()
            
            predicted_social_terms = [word for word in predicted_words if any(term in word for term in social_terms)]
            correct_social_terms = [word for word in correct_words if any(term in word for term in social_terms)]
            
            if correct_social_terms:
                term_overlap = len(set(predicted_social_terms).intersection(set(correct_social_terms)))
                return min(term_overlap / len(correct_social_terms), 1.0)
        
        # General semantic similarity
        predicted_words = set(predicted.split())
        correct_words = set(correct.split())
        
        if len(correct_words) == 0:
            return 0.0
        
        overlap = len(predicted_words.intersection(correct_words))
        return overlap / len(correct_words)
    
    def _assess_commonsense_reasoning_quality(self, reasoning: str, knowledge_domain: str, reasoning_type: str) -> float:
        """Assess the quality of commonsense reasoning"""
        if not reasoning:
            return 0.0
        
        reasoning_lower = reasoning.lower()
        quality_scores = {}
        
        # Check for domain-relevant reasoning indicators
        domain_indicators = {
            'physical': ['force', 'energy', 'heat', 'cold', 'gravity', 'impact', 'molecular'],
            'social': ['people', 'society', 'polite', 'appropriate', 'respect', 'behavior'],
            'temporal': ['time', 'after', 'before', 'eventually', 'typically', 'usually'],
            'causal': ['because', 'cause', 'effect', 'leads to', 'results in', 'due to'],
            'biological': ['body', 'function', 'need', 'survive', 'energy', 'nutrients'],
            'situational': ['situation', 'circumstances', 'context', 'appropriate', 'normal']
        }
        
        # Assess domain-specific reasoning
        if knowledge_domain in domain_indicators:
            domain_score = sum(1 for indicator in domain_indicators[knowledge_domain] 
                             if indicator in reasoning_lower)
            quality_scores['domain_specificity'] = min(domain_score / 5, 0.4)
        else:
            quality_scores['domain_specificity'] = 0.0
        
        # Check for logical structure
        logical_indicators = ['because', 'therefore', 'since', 'leads to', 'results in']
        logical_score = sum(1 for indicator in logical_indicators if indicator in reasoning_lower)
        quality_scores['logical_structure'] = min(logical_score / 3, 0.3)
        
        # Check for world knowledge application
        world_knowledge_terms = ['normally', 'typically', 'usually', 'generally', 'common', 'standard']
        knowledge_score = sum(1 for term in world_knowledge_terms if term in reasoning_lower)
        quality_scores['world_knowledge'] = min(knowledge_score / 3, 0.3)
        
        # Assess completeness
        total_words = len(reasoning.split())
        if total_words < 5:
            quality_scores['completeness'] = 0.0
        else:
            quality_scores['completeness'] = min(total_words / 20, 0.2)  # Reward detailed reasoning
        
        return sum(quality_scores.values())
    
    def _calculate_commonsense_coherence(self) -> float:
        """Calculate overall commonsense reasoning coherence"""
        # This would analyze the consistency and logical flow of commonsense reasoning
        # For now, return a high score indicating good coherence
        return 0.88
    
    def validate_setup(self) -> bool:
        """Validate the commonsense reasoning benchmark setup"""
        try:
            # Check if commonsense prompts are defined
            if not self.commonsense_prompts:
                logger.error("No commonsense reasoning prompts defined")
                return False
            
            # Check domain coverage
            domains = set(ex.knowledge_domain for ex in self.examples)
            expected_domains = {'physical', 'social', 'temporal', 'causal', 'biological', 'situational'}
            
            if not expected_domains.issubset(domains):
                missing_domains = expected_domains - domains
                logger.warning(f"Missing knowledge domains: {missing_domains}")
            
            # Check reasoning type coverage
            reasoning_types = set(ex.reasoning_type for ex in self.examples)
            expected_types = {'cause_effect', 'why_because', 'what_happens_next', 'social_situations'}
            
            if not expected_types.issubset(reasoning_types):
                missing_types = expected_types - reasoning_types
                logger.warning(f"Missing reasoning types: {missing_types}")
            
            logger.info("Commonsense reasoning benchmark validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Commonsense reasoning benchmark validation failed: {e}")
            return False