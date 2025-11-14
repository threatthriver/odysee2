"""
Advanced Prompt Engineering for Reasoning

This module implements sophisticated prompt engineering capabilities including:
- Dynamic prompt generation based on task requirements
- Few-shot learning with reasoning examples
- Template-based reasoning instruction
- Prompt optimization and adaptation
- Reasoning instruction templates for different reasoning types
- Context-aware prompt construction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import json
import random
import re
from collections import defaultdict, Counter
import numpy as np

from .chain_of_thought import ReasoningStep, ReasoningChain
from .symbolic_reasoning import SymbolicExpression, SymbolicOperationType


class PromptTemplateType(Enum):
    """Types of prompt templates for different reasoning tasks."""
    CHAIN_OF_THOUGHT = "chain_of_thought"
    MATHEMATICAL_REASONING = "mathematical_reasoning"
    LOGICAL_REASONING = "logical_reasoning"
    CAUSAL_REASONING = "causal_reasoning"
    ANALOGICAL_REASONING = "analogical_reasoning"
    MULTI_MODAL_REASONING = "multi_modal_reasoning"
    STEP_BY_STEP = "step_by_step"
    QUESTION_ANSWERING = "question_answering"
    PROBLEM_SOLVING = "problem_solving"
    CREATIVE_THINKING = "creative_thinking"


class ReasoningInstruction(Enum):
    """Types of reasoning instructions to include in prompts."""
    ANALYZE = "analyze"
    CALCULATE = "calculate"
    COMPARE = "compare"
    CONTRAST = "contrast"
    DEDUCE = "deduce"
    EVALUATE = "evaluate"
    EXPLAIN = "explain"
    GENERALIZE = "generalize"
    HYPOTHESIZE = "hypothesize"
    INFER = "infer"
    JUSTIFY = "justify"
    PREDICT = "predict"
    PROVE = "prove"
    REASON = "reason"
    SOLVE = "solve"
    SUMMARIZE = "summarize"
    SYNTHESIZE = "synthesize"


@dataclass
class PromptTemplate:
    """Template for generating reasoning prompts."""
    template_type: PromptTemplateType
    instruction: str
    reasoning_structure: List[str]
    example_format: str
    output_format: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class FewShotExample:
    """Example for few-shot learning in reasoning."""
    problem: str
    reasoning_steps: List[str]
    final_answer: str
    reasoning_type: str
    complexity_score: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DynamicPromptConfig:
    """Configuration for dynamic prompt generation."""
    max_context_length: int = 2048
    template_variations: int = 3
    few_shot_count: int = 2
    reasoning_step_guidance: bool = True
    include_intermediate_steps: bool = True
    adapt_to_difficulty: bool = True
    include_confidence_guidance: bool = True
    multi_modal_instructions: bool = False


class ReasoningPromptTemplate:
    """
    Collection of reasoning prompt templates for different types of tasks.
    
    Provides standardized templates that can be dynamically adapted and
    combined to create effective reasoning prompts.
    """
    
    @staticmethod
    def get_chain_of_thought_template() -> PromptTemplate:
        """Get chain-of-thought reasoning template."""
        return PromptTemplate(
            template_type=PromptTemplateType.CHAIN_OF_THOUGHT,
            instruction="Let's think step by step to solve this problem.",
            reasoning_structure=[
                "First, I need to understand what is being asked.",
                "Next, I should identify the key information.",
                "Then, I will break down the problem into smaller parts.",
                "After that, I'll work through each part systematically.",
                "Finally, I'll combine the results to get the answer."
            ],
            example_format="Problem: {problem}\nLet's think step by step:\n{reasoning_steps}\nTherefore, the answer is: {answer}",
            output_format="Please provide your reasoning step by step, clearly explaining each step of your thought process.",
            metadata={
                'steps_required': 5,
                'difficulty_progression': 'increasing',
                'confidence_indicators': True
            }
        )
    
    @staticmethod
    def get_mathematical_reasoning_template() -> PromptTemplate:
        """Get mathematical reasoning template."""
        return PromptTemplate(
            template_type=PromptTemplateType.MATHEMATICAL_REASONING,
            instruction="Let's solve this mathematical problem step by step using logical reasoning.",
            reasoning_structure=[
                "Identify the mathematical concept or formula needed.",
                "Extract the given values and constraints.",
                "Set up the mathematical equation or expression.",
                "Solve step by step, showing all calculations.",
                "Verify the solution by checking the result."
            ],
            example_format="Math Problem: {problem}\nMathematical reasoning:\n{reasoning_steps}\nFinal answer: {answer}",
            output_format="Show your mathematical work step by step, including all calculations and reasoning.",
            metadata={
                'mathematical_domains': ['algebra', 'geometry', 'calculus', 'statistics'],
                'requires_symbolic_computation': True
            }
        )
    
    @staticmethod
    def get_logical_reasoning_template() -> PromptTemplate:
        """Get logical reasoning template."""
        return PromptTemplate(
            template_type=PromptTemplateType.LOGICAL_REASONING,
            instruction="Let's analyze this logically by examining the premises and drawing valid conclusions.",
            reasoning_structure=[
                "Identify the premises and statements given.",
                "Check for logical consistency and validity.",
                "Apply logical rules and inference patterns.",
                "Consider alternative interpretations.",
                "Draw a well-supported conclusion."
            ],
            example_format="Logical Problem: {problem}\nLogical analysis:\n{reasoning_steps}\nConclusion: {answer}",
            output_format="Analyze the logical structure and provide a well-reasoned conclusion.",
            metadata={
                'logical_operations': ['deduction', 'induction', 'abduction'],
                'fallacy_detection': True
            }
        )
    
    @staticmethod
    def get_multi_modal_reasoning_template() -> PromptTemplate:
        """Get multi-modal reasoning template."""
        return PromptTemplate(
            template_type=PromptTemplateType.MULTI_MODAL_REASONING,
            instruction="Let's integrate information from multiple sources (text, images, data) to reason about this problem.",
            reasoning_structure=[
                "Identify all available modalities and their content.",
                "Extract relevant information from each modality.",
                "Find connections and relationships between modalities.",
                "Synthesize information across modalities.",
                "Formulate a comprehensive conclusion."
            ],
            example_format="Multi-modal Problem: {problem}\nModality integration:\n{reasoning_steps}\nIntegrated answer: {answer}",
            output_format="Reason across multiple modalities and provide an integrated analysis.",
            metadata={
                'modalities': ['text', 'image', 'table', 'audio'],
                'cross_modal_consistency_required': True
            }
        )
    
    @staticmethod
    def get_all_templates() -> Dict[PromptTemplateType, PromptTemplate]:
        """Get all available prompt templates."""
        return {
            PromptTemplateType.CHAIN_OF_THOUGHT: ReasoningPromptTemplate.get_chain_of_thought_template(),
            PromptTemplateType.MATHEMATICAL_REASONING: ReasoningPromptTemplate.get_mathematical_reasoning_template(),
            PromptTemplateType.LOGICAL_REASONING: ReasoningPromptTemplate.get_logical_reasoning_template(),
            PromptTemplateType.MULTI_MODAL_REASONING: ReasoningPromptTemplate.get_multi_modal_reasoning_template(),
            PromptTemplateType.STEP_BY_STEP: ReasoningPromptTemplate._get_step_by_step_template(),
            PromptTemplateType.QUESTION_ANSWERING: ReasoningPromptTemplate._get_qa_template(),
            PromptTemplateType.PROBLEM_SOLVING: ReasoningPromptTemplate._get_problem_solving_template(),
        }
    
    @staticmethod
    def _get_step_by_step_template() -> PromptTemplate:
        """Get step-by-step reasoning template."""
        return PromptTemplate(
            template_type=PromptTemplateType.STEP_BY_STEP,
            instruction="Break this down into clear, manageable steps.",
            reasoning_structure=[
                "Define the problem clearly.",
                "List what you know and what you need to find.",
                "Identify the approach or method to use.",
                "Execute the approach step by step.",
                "Review and confirm the answer."
            ],
            example_format="Problem: {problem}\nStep-by-step solution:\n{reasoning_steps}\nAnswer: {answer}",
            output_format="Provide a clear step-by-step breakdown of your reasoning process.",
            metadata={'clarity_emphasis': True}
        )
    
    @staticmethod
    def _get_qa_template() -> PromptTemplate:
        """Get question-answering template."""
        return PromptTemplate(
            template_type=PromptTemplateType.QUESTION_ANSWERING,
            instruction="Carefully analyze the question and provide a comprehensive answer.",
            reasoning_structure=[
                "Understand what the question is asking.",
                "Identify key concepts and terms.",
                "Analyze the context and constraints.",
                "Formulate a complete answer.",
                "Check if the answer fully addresses the question."
            ],
            example_format="Question: {problem}\nAnalysis:\n{reasoning_steps}\nAnswer: {answer}",
            output_format="Provide a comprehensive and accurate answer with supporting reasoning.",
            metadata={'completeness_check': True}
        )
    
    @staticmethod
    def _get_problem_solving_template() -> PromptTemplate:
        """Get problem-solving template."""
        return PromptTemplate(
            template_type=PromptTemplateType.PROBLEM_SOLVING,
            instruction="Apply systematic problem-solving techniques to tackle this challenge.",
            reasoning_structure=[
                "Clearly define the problem and success criteria.",
                "Generate potential solution approaches.",
                "Evaluate and select the best approach.",
                "Implement the solution systematically.",
                "Assess the results and iterate if necessary."
            ],
            example_format="Problem: {problem}\nProblem-solving process:\n{reasoning_steps}\nSolution: {answer}",
            output_format="Apply systematic problem-solving methodology with clear rationale.",
            metadata={'methodology_focus': True}
        )


class FewShotReasoningExample:
    """
    Manager for few-shot learning examples in reasoning tasks.
    
    Maintains a collection of diverse reasoning examples that can be
    selected and adapted for different tasks and difficulty levels.
    """
    
    def __init__(self, embedding_dim: int = 256):
        self.embedding_dim = embedding_dim
        self.examples: List[FewShotExample] = []
        self.example_embeddings = {}
        self.difficulty_distribution = defaultdict(list)
        
        # Initialize with some default examples
        self._initialize_default_examples()
        
    def _initialize_default_examples(self):
        """Initialize with some default reasoning examples."""
        default_examples = [
            FewShotExample(
                problem="If all roses are flowers, and some flowers are red, can we conclude that some roses are red?",
                reasoning_steps=[
                    "We know that all roses are flowers (universal affirmative).",
                    "We know that some flowers are red (particular affirmative).",
                    "This doesn't necessarily mean that the red flowers include roses.",
                    "The red flowers could be other types of flowers like tulips or carnations.",
                    "Therefore, we cannot conclude that some roses are red."
                ],
                final_answer="No, we cannot conclude that some roses are red.",
                reasoning_type="logical_reasoning",
                complexity_score=0.6
            ),
            FewShotExample(
                problem="A train travels 120 miles in 2 hours. What's its average speed?",
                reasoning_steps=[
                    "The problem asks for average speed.",
                    "Average speed is total distance divided by total time.",
                    "Total distance = 120 miles.",
                    "Total time = 2 hours.",
                    "Average speed = 120 miles ÷ 2 hours = 60 miles per hour."
                ],
                final_answer="60 miles per hour",
                reasoning_type="mathematical_reasoning",
                complexity_score=0.3
            ),
            FewShotExample(
                problem="Compare the causes of World War I and World War II.",
                reasoning_steps=[
                    "For WWI: Militarism, alliances, imperialism, and nationalism were key factors.",
                    "For WWII: Aggressive nationalism, failure of appeasement, and economic factors played major roles.",
                    "Both involved complex international tensions and alliance systems.",
                    "However, WWII also had the specific context of unresolved WWI issues and economic depression.",
                    "The causes were different but interconnected through historical context."
                ],
                final_answer="While both wars involved complex international tensions, WWI was caused by militarism, alliances, and nationalism, while WWII built on unresolved WWI issues, aggressive nationalism, and economic factors.",
                reasoning_type="comparative_reasoning",
                complexity_score=0.8
            )
        ]
        
        for example in default_examples:
            self.add_example(example)
            
    def add_example(self, example: FewShotExample):
        """Add a new few-shot example."""
        self.examples.append(example)
        
        # Categorize by reasoning type
        self.difficulty_distribution[example.reasoning_type].append(example)
        
        # Compute embedding (placeholder - would use actual embedding model)
        embedding = torch.randn(self.embedding_dim)
        self.example_embeddings[example] = embedding
        
    def select_examples(
        self,
        task_type: str,
        count: int = 2,
        difficulty_range: Tuple[float, float] = (0.0, 1.0),
        exclude_examples: Optional[List[FewShotExample]] = None
    ) -> List[FewShotExample]:
        """
        Select relevant few-shot examples for a task.
        
        Args:
            task_type: Type of reasoning task
            count: Number of examples to select
            difficulty_range: Preferred difficulty range (0.0 to 1.0)
            exclude_examples: Examples to exclude from selection
            
        Returns:
            List of selected few-shot examples
        """
        exclude_examples = exclude_examples or []
        
        # Get examples for the task type
        relevant_examples = [
            ex for ex in self.examples 
            if ex.reasoning_type == task_type and ex not in exclude_examples
        ]
        
        # Filter by difficulty
        filtered_examples = [
            ex for ex in relevant_examples
            if difficulty_range[0] <= ex.complexity_score <= difficulty_range[1]
        ]
        
        if not filtered_examples:
            # Fall back to all examples of this type
            filtered_examples = relevant_examples
            
        if not filtered_examples:
            # Fall back to any examples
            filtered_examples = [ex for ex in self.examples if ex not in exclude_examples]
        
        # Select diverse examples (by complexity)
        selected = self._select_diverse_examples(filtered_examples, count)
        
        return selected
        
    def _select_diverse_examples(
        self,
        candidates: List[FewShotExample],
        count: int
    ) -> List[FewShotExample]:
        """Select diverse examples to maximize learning value."""
        if len(candidates) <= count:
            return candidates
            
        # Sort by complexity for diversity
        candidates_sorted = sorted(candidates, key=lambda x: x.complexity_score)
        
        # Select evenly spaced examples
        step = max(1, len(candidates_sorted) // count)
        selected = candidates_sorted[::step][:count]
        
        return selected
        
    def adapt_example(
        self,
        example: FewShotExample,
        new_problem: str,
        preserve_structure: bool = True
    ) -> FewShotExample:
        """
        Adapt an existing example to a new problem while preserving reasoning structure.
        
        Args:
            example: Original example to adapt
            new_problem: New problem to adapt to
            preserve_structure: Whether to preserve the reasoning structure
            
        Returns:
            Adapted few-shot example
        """
        adapted_example = FewShotExample(
            problem=new_problem,
            reasoning_steps=example.reasoning_steps.copy() if preserve_structure else [],
            final_answer="",  # To be filled by reasoning process
            reasoning_type=example.reasoning_type,
            complexity_score=example.complexity_score,
            metadata=example.metadata.copy()
        )
        
        return adapted_example
        
    def get_example_statistics(self) -> Dict[str, Any]:
        """Get statistics about the example collection."""
        return {
            'total_examples': len(self.examples),
            'by_reasoning_type': {
                reason_type: len(examples)
                for reason_type, examples in self.difficulty_distribution.items()
            },
            'difficulty_distribution': {
                'min': min(ex.complexity_score for ex in self.examples),
                'max': max(ex.complexity_score for ex in self.examples),
                'mean': np.mean([ex.complexity_score for ex in self.examples]),
                'std': np.std([ex.complexity_score for ex in self.examples])
            }
        }


class DynamicPromptGenerator(nn.Module):
    """
    Dynamic prompt generator that creates tailored prompts for reasoning tasks.
    
    Combines prompt templates, few-shot examples, and context-aware generation
    to create effective reasoning prompts.
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        config: Optional[DynamicPromptConfig] = None,
        **kwargs
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.config = config or DynamicPromptConfig()
        
        # Prompt templates
        self.templates = ReasoningPromptTemplate.get_all_templates()
        
        # Few-shot examples
        self.few_shot_manager = FewShotReasoningExample(embedding_dim)
        
        # Context encoder for prompt adaptation
        self.context_encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
        # Template selector network
        self.template_selector = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, len(self.templates)),
            nn.Softmax(dim=-1)
        )
        
        # Instruction generator
        self.instruction_generator = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, len(ReasoningInstruction)),
            nn.Softmax(dim=-1)
        )
        
    def generate_prompt(
        self,
        problem: str,
        context: Optional[torch.Tensor] = None,
        task_type: str = "general",
        difficulty_level: float = 0.5,
        available_modalities: Optional[List[str]] = None,
        reasoning_history: Optional[List[ReasoningStep]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a dynamic prompt for a reasoning task.
        
        Args:
            problem: The problem to solve
            context: Context embedding for the task
            task_type: Type of reasoning task
            difficulty_level: Difficulty level (0.0 to 1.0)
            available_modalities: Available modalities (text, image, etc.)
            reasoning_history: Previous reasoning steps
            
        Returns:
            Dictionary containing generated prompt components
        """
        # Encode context
        if context is not None:
            encoded_context = self.context_encoder(context)
        else:
            encoded_context = torch.zeros(self.embedding_dim)
            
        # Select appropriate template
        template = self._select_template(encoded_context, task_type, available_modalities)
        
        # Select few-shot examples
        few_shot_examples = self._select_few_shot_examples(
            task_type, difficulty_level, reasoning_history
        )
        
        # Generate instructions
        instructions = self._generate_instructions(encoded_context, task_type)
        
        # Construct full prompt
        prompt_components = self._construct_prompt(
            problem, template, few_shot_examples, instructions, context
        )
        
        # Adapt prompt based on context and history
        adapted_prompt = self._adapt_prompt(
            prompt_components, context, reasoning_history, difficulty_level
        )
        
        return adapted_prompt
        
    def _select_template(
        self,
        context: torch.Tensor,
        task_type: str,
        available_modalities: Optional[List[str]] = None
    ) -> PromptTemplate:
        """Select the most appropriate prompt template."""
        available_modalities = available_modalities or []
        
        # Map task types to template types
        task_to_template = {
            'mathematical': PromptTemplateType.MATHEMATICAL_REASONING,
            'logical': PromptTemplateType.LOGICAL_REASONING,
            'multi_modal': PromptTemplateType.MULTI_MODAL_REASONING,
            'step_by_step': PromptTemplateType.STEP_BY_STEP,
            'qa': PromptTemplateType.QUESTION_ANSWERING,
            'problem_solving': PromptTemplateType.PROBLEM_SOLVING,
            'general': PromptTemplateType.CHAIN_OF_THOUGHT
        }
        
        # Use task-specific template if available
        if task_type in task_to_template:
            preferred_template_type = task_to_template[task_type]
            if preferred_template_type in self.templates:
                return self.templates[preferred_template_type]
        
        # Use multi-modal template if multiple modalities available
        if len(available_modalities) > 1:
            return self.templates[PromptTemplateType.MULTI_MODAL_REASONING]
        
        # Use mathematical template for math problems
        if 'math' in task_type.lower() or 'calculate' in task_type.lower():
            return self.templates[PromptTemplateType.MATHEMATICAL_REASONING]
        
        # Default to chain-of-thought
        return self.templates[PromptTemplateType.CHAIN_OF_THOUGHT]
        
    def _select_few_shot_examples(
        self,
        task_type: str,
        difficulty_level: float,
        reasoning_history: Optional[List[ReasoningStep]] = None
    ) -> List[FewShotExample]:
        """Select appropriate few-shot examples."""
        difficulty_range = (
            max(0.0, difficulty_level - 0.3),
            min(1.0, difficulty_level + 0.3)
        )
        
        count = min(self.config.few_shot_count, 3)  # Limit to prevent prompt bloat
        
        # Exclude examples similar to reasoning history
        exclude = []
        if reasoning_history:
            # Simple exclusion based on reasoning type
            history_types = set(step.reasoning_type for step in reasoning_history)
            exclude = [ex for ex in self.few_shot_manager.examples 
                      if ex.reasoning_type in history_types]
        
        return self.few_shot_manager.select_examples(
            task_type, count, difficulty_range, exclude
        )
        
    def _generate_instructions(
        self,
        context: torch.Tensor,
        task_type: str
    ) -> List[ReasoningInstruction]:
        """Generate specific instructions for the reasoning task."""
        # Map task types to relevant instructions
        task_instructions = {
            'mathematical': [
                ReasoningInstruction.ANALYZE,
                ReasoningInstruction.CALCULATE,
                ReasoningInstruction.SOLVE
            ],
            'logical': [
                ReasoningInstruction.ANALYZE,
                ReasoningInstruction.DEDUCE,
                ReasoningInstruction.JUSTIFY
            ],
            'comparative': [
                ReasoningInstruction.COMPARE,
                ReasoningInstruction.CONTRAST,
                ReasoningInstruction.EVALUATE
            ],
            'creative': [
                ReasoningInstruction.HYPOTHESIZE,
                ReasoningInstruction.SYNTHESIZE,
                ReasoningInstruction.EVALUATE
            ]
        }
        
        # Get instructions for task type
        relevant_instructions = task_instructions.get(
            task_type,
            [ReasoningInstruction.ANALYZE, ReasoningInstruction.REASON]
        )
        
        # Use neural instruction generation if context available
        if context.numel() > 0:
            context_expanded = context.unsqueeze(0)
            instruction_probs = self.instruction_generator(context_expanded)
            
            # Select top instructions
            top_indices = torch.topk(instruction_probs, k=3).indices[0]
            neural_instructions = [
                list(ReasoningInstruction)[idx.item()]
                for idx in top_indices
            ]
            
            # Combine neural and task-specific instructions
            combined = list(set(relevant_instructions + neural_instructions))
            return combined[:5]  # Limit to prevent prompt bloat
        
        return relevant_instructions
        
    def _construct_prompt(
        self,
        problem: str,
        template: PromptTemplate,
        few_shot_examples: List[FewShotExample],
        instructions: List[ReasoningInstruction],
        context: Optional[torch.Tensor]
    ) -> Dict[str, Any]:
        """Construct the basic prompt structure."""
        # Base instruction
        instruction_text = template.instruction
        
        # Add specific instructions
        if instructions:
            instruction_details = [
                f"- {inst.value.replace('_', ' ').title()}"
                for inst in instructions
            ]
            instruction_text += " Instructions:\n" + "\n".join(instruction_details)
        
        # Format few-shot examples
        formatted_examples = []
        for example in few_shot_examples:
            formatted_example = template.example_format.format(
                problem=example.problem,
                reasoning_steps="\n".join(f"  {i+1}. {step}" for i, step in enumerate(example.reasoning_steps)),
                answer=example.final_answer
            )
            formatted_examples.append(formatted_example)
        
        # Construct full prompt
        prompt_parts = {
            'instruction': instruction_text,
            'few_shot_examples': formatted_examples,
            'problem': problem,
            'output_format': template.output_format,
            'reasoning_structure': template.reasoning_structure,
            'template_type': template.template_type
        }
        
        return prompt_parts
        
    def _adapt_prompt(
        self,
        prompt_components: Dict[str, Any],
        context: Optional[torch.Tensor],
        reasoning_history: Optional[List[ReasoningStep]],
        difficulty_level: float
    ) -> Dict[str, Any]:
        """Adapt the prompt based on context and history."""
        adapted_components = prompt_components.copy()
        
        # Add context-specific adaptations
        if context is not None and self.config.adapt_to_difficulty:
            if difficulty_level > 0.7:
                # High difficulty: add more structure and guidance
                adapted_components['guidance_level'] = 'high'
                adapted_components['structure_emphasis'] = True
                adapted_components['confidence_check'] = True
            elif difficulty_level < 0.3:
                # Low difficulty: simplify and provide more support
                adapted_components['guidance_level'] = 'low'
                adapted_components['simplified_language'] = True
                adapted_components['examples_enhanced'] = True
            else:
                # Medium difficulty: balanced approach
                adapted_components['guidance_level'] = 'medium'
        
        # Incorporate reasoning history
        if reasoning_history:
            adapted_components['reasoning_history'] = [
                {
                    'step_id': step.step_id,
                    'type': step.reasoning_type,
                    'confidence': step.confidence,
                    'dependencies': step.dependencies
                }
                for step in reasoning_history
            ]
            
            # Suggest next reasoning steps
            suggested_structure = self._suggest_next_steps(reasoning_history)
            adapted_components['suggested_structure'] = suggested_structure
        
        # Add confidence guidance if enabled
        if self.config.include_confidence_guidance:
            adapted_components['confidence_guidance'] = [
                "Rate your confidence in each reasoning step (High/Medium/Low).",
                "Explain any uncertainties or alternative interpretations.",
                "Verify your final answer before concluding."
            ]
        
        # Format final prompt
        final_prompt = self._format_final_prompt(adapted_components)
        adapted_components['final_prompt'] = final_prompt
        
        return adapted_components
        
    def _suggest_next_steps(
        self,
        reasoning_history: List[ReasoningStep]
    ) -> List[str]:
        """Suggest next reasoning steps based on history."""
        if not reasoning_history:
            return []
            
        # Analyze reasoning patterns
        reasoning_types = [step.reasoning_type for step in reasoning_history]
        type_counts = Counter(reasoning_types)
        
        # Get current reasoning structure
        current_structure = reasoning_types
        
        # Suggest next steps based on patterns
        if 'logical' in current_structure and 'mathematical' not in current_structure:
            return ["Consider mathematical verification of the logical conclusion."]
        elif 'mathematical' in current_structure and 'verification' not in current_structure:
            return ["Verify the calculation steps", "Check for alternative solution methods"]
        elif len(current_structure) < 3:
            return ["Consider alternative approaches", "Evaluate the reasoning completeness"]
        else:
            return ["Synthesize conclusions", "Consider broader implications"]
            
    def _format_final_prompt(self, components: Dict[str, Any]) -> str:
        """Format all components into a final prompt string."""
        prompt_parts = []
        
        # Add main instruction
        prompt_parts.append(components['instruction'])
        prompt_parts.append("")
        
        # Add few-shot examples if present
        if components.get('few_shot_examples'):
            prompt_parts.append("Here are some examples:")
            for example in components['few_shot_examples']:
                prompt_parts.append(example)
                prompt_parts.append("")
        
        # Add the problem
        prompt_parts.append(f"Problem: {components['problem']}")
        prompt_parts.append("")
        
        # Add reasoning structure guidance
        if 'suggested_structure' in components:
            prompt_parts.append("Suggested next steps:")
            for step in components['suggested_structure']:
                prompt_parts.append(f"- {step}")
            prompt_parts.append("")
        
        # Add output format requirement
        prompt_parts.append(components['output_format'])
        
        # Add confidence guidance if present
        if components.get('confidence_guidance'):
            prompt_parts.append("\nAdditional guidance:")
            for guidance in components['confidence_guidance']:
                prompt_parts.append(f"- {guidance}")
        
        return "\n".join(prompt_parts)
        
    def get_prompt_statistics(self) -> Dict[str, Any]:
        """Get statistics about the prompt generation system."""
        return {
            'available_templates': list(self.templates.keys()),
            'template_count': len(self.templates),
            'few_shot_examples': self.few_shot_manager.get_example_statistics(),
            'configuration': {
                'max_context_length': self.config.max_context_length,
                'few_shot_count': self.config.few_shot_count,
                'template_variations': self.config.template_variations
            }
        }


class ReasoningPromptOptimizer(nn.Module):
    """
    Optimizer for reasoning prompts that learns to improve prompt effectiveness
    based on reasoning performance and outcomes.
    """
    
    def __init__(self, embedding_dim: int = 256, **kwargs):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Prompt effectiveness predictor
        self.effectiveness_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
        
        # Template adaptation network
        self.template_adapter = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
    def optimize_prompt(
        self,
        prompt: str,
        context: torch.Tensor,
        performance_feedback: Optional[torch.Tensor] = None,
        **kwargs
    ) -> str:
        """
        Optimize a prompt based on performance feedback.
        
        Args:
            prompt: Original prompt string
            context: Context embedding
            performance_feedback: Feedback on prompt effectiveness
            
        Returns:
            Optimized prompt string
        """
        # This would implement actual prompt optimization using feedback
        # For now, return the original prompt with potential minor adaptations
        
        adapted_prompt = self._apply_optimization_heuristics(prompt, context)
        
        return adapted_prompt
        
    def _apply_optimization_heuristics(self, prompt: str, context: torch.Tensor) -> str:
        """Apply heuristic optimizations to the prompt."""
        # Simple heuristics for prompt optimization
        
        # Check prompt length and adapt if too long/short
        prompt_words = prompt.split()
        if len(prompt_words) > 500:
            # Truncate if too long
            prompt = " ".join(prompt_words[:500]) + "..."
        elif len(prompt_words) < 50:
            # Add clarifications if too short
            prompt += "\n\nPlease provide a detailed and comprehensive response."
        
        # Ensure clarity and structure
        if not prompt.strip().endswith(('.', '?', '!')):
            prompt += "."
            
        return prompt