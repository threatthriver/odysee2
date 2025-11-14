"""
Chain-of-Thought (CoT) Reasoning Processing System

This module implements sophisticated chain-of-thought reasoning capabilities including:
- Explicit reasoning step generation and tracking
- Intermediate reasoning storage and management  
- Reasoning-aware loss functions
- Integration with transformer architectures for step-by-step reasoning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import copy
from dataclasses import dataclass
from collections import defaultdict
import math
import warnings

from .reasoning_memory import ReasoningMemory
from .reasoning_utils import ReasoningValidator


@dataclass
class ReasoningStep:
    """Represents a single reasoning step in the chain."""
    step_id: int
    content: torch.Tensor
    confidence: float
    reasoning_type: str  # 'logical', 'mathematical', 'inductive', 'deductive'
    dependencies: List[int]  # Step IDs this step depends on
    next_steps: List[int]   # Step IDs that depend on this step
    metadata: Dict[str, Any]


@dataclass 
class ReasoningChain:
    """Represents a complete chain of reasoning steps."""
    chain_id: str
    question: torch.Tensor
    final_answer: torch.Tensor
    steps: List[ReasoningStep]
    reasoning_metadata: Dict[str, Any]
    quality_score: float
    is_valid: bool


class ReasoningStateTracker(nn.Module):
    """
    Tracks the current state of reasoning process including:
    - Active reasoning steps
    - Step dependencies and relationships
    - Reasoning progress and completion status
    - Confidence scores for each step
    """
    
    def __init__(
        self,
        hidden_size: int,
        max_reasoning_steps: int = 20,
        reasoning_dim: int = 256,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_reasoning_steps = max_reasoning_steps
        self.reasoning_dim = reasoning_dim
        
        # State tracking components
        self.step_tracker = nn.LSTM(
            input_size=hidden_size,
            hidden_size=reasoning_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Step quality prediction
        self.step_quality_predictor = nn.Sequential(
            nn.Linear(reasoning_dim, reasoning_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(reasoning_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Reasoning type classification
        self.reasoning_type_classifier = nn.Linear(
            reasoning_dim, 
            len(['logical', 'mathematical', 'inductive', 'deductive', 'analogic'])
        )
        
        # Attention for step dependencies
        self.step_attention = nn.MultiheadAttention(
            embed_dim=reasoning_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Initialize tracking state
        self.reset_state()
        
    def reset_state(self):
        """Reset the reasoning state tracker."""
        self.active_steps: List[ReasoningStep] = []
        self.completed_steps: List[ReasoningStep] = []
        self.step_history: List[Dict] = []
        self.current_step_id = 0
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        reasoning_context: Optional[torch.Tensor] = None,
        step_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the reasoning state tracker.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            reasoning_context: [batch_size, max_steps, reasoning_dim]
            step_mask: [batch_size, max_steps] - mask for valid steps
            
        Returns:
            Dict containing step states, quality scores, and reasoning type predictions
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Aggregate sequence information for step processing
        if reasoning_context is None:
            # Create initial reasoning context
            reasoning_context = torch.zeros(
                batch_size, self.max_reasoning_steps, self.reasoning_dim,
                device=hidden_states.device
            )
        
        # Process reasoning steps
        step_outputs, (hidden, cell) = self.step_tracker(reasoning_context)
        
        # Predict step quality
        quality_scores = self.step_quality_predictor(step_outputs)
        
        # Classify reasoning types
        reasoning_types = F.softmax(
            self.reasoning_type_classifier(step_outputs), dim=-1
        )
        
        # Update step attention weights
        attended_steps, attention_weights = self.step_attention(
            step_outputs, step_outputs, step_outputs,
            key_padding_mask=~step_mask.bool() if step_mask is not None else None
        )
        
        return {
            'step_outputs': step_outputs,
            'step_quality': quality_scores,
            'reasoning_types': reasoning_types,
            'attention_weights': attention_weights,
            'hidden_state': hidden,
            'cell_state': cell,
            'attended_steps': attended_steps
        }


class ChainOfThoughtProcessor(nn.Module):
    """
    Core Chain-of-Thought reasoning processor that manages the generation,
    validation, and coordination of reasoning steps.
    """
    
    def __init__(
        self,
        hidden_size: int,
        max_reasoning_steps: int = 20,
        reasoning_dim: int = 256,
        enable_step_validation: bool = True,
        use_reasoning_memory: bool = True,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_reasoning_steps = max_reasoning_steps
        self.reasoning_dim = reasoning_dim
        self.enable_step_validation = enable_step_validation
        self.use_reasoning_memory = use_reasoning_memory
        
        # State tracking
        self.state_tracker = ReasoningStateTracker(
            hidden_size=hidden_size,
            max_reasoning_steps=max_reasoning_steps,
            reasoning_dim=reasoning_dim
        )
        
        # Step generation components
        self.step_generator = nn.Sequential(
            nn.Linear(hidden_size + reasoning_dim, reasoning_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(reasoning_dim, reasoning_dim),
            nn.LayerNorm(reasoning_dim)
        )
        
        # Final answer synthesis
        self.answer_synthesizer = nn.Sequential(
            nn.Linear(hidden_size + reasoning_dim, reasoning_dim),
            nn.ReLU(),
            nn.Linear(reasoning_dim, hidden_size)
        )
        
        # Step coherence checker
        self.coherence_checker = nn.Sequential(
            nn.Linear(reasoning_dim * 2, reasoning_dim),
            nn.ReLU(),
            nn.Linear(reasoning_dim, 1),
            nn.Sigmoid()
        )
        
        # Memory components
        if use_reasoning_memory:
            self.reasoning_memory = ReasoningMemory(
                embedding_dim=reasoning_dim,
                max_memory_size=1000
            )
        
        # Validation components
        if enable_step_validation:
            self.validator = ReasoningValidator(reasoning_dim=reasoning_dim)
        
        # Initialize
        self.reset_processing_state()
        
    def reset_processing_state(self):
        """Reset the processing state for new reasoning chains."""
        self.state_tracker.reset_state()
        self.current_chains: Dict[str, ReasoningChain] = {}
        
    def generate_reasoning_step(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        previous_steps: List[ReasoningStep],
        **kwargs
    ) -> torch.Tensor:
        """
        Generate a new reasoning step based on query, context, and previous steps.
        
        Args:
            query: [batch_size, hidden_size] - current query/question
            context: [batch_size, seq_len, hidden_size] - contextual information
            previous_steps: List of previous reasoning steps
            
        Returns:
            New reasoning step embeddings: [batch_size, reasoning_dim]
        """
        batch_size = query.size(0)
        
        # Encode previous steps if they exist
        if previous_steps:
            step_embeddings = torch.stack([
                step.content for step in previous_steps
            ], dim=1)  # [batch_size, num_steps, hidden_size]
            
            # Aggregate step information
            step_summary = torch.mean(step_embeddings, dim=1)  # [batch_size, hidden_size]
        else:
            step_summary = torch.zeros(batch_size, self.hidden_size, device=query.device)
        
        # Combine query, context, and step summary
        combined_input = torch.cat([query, step_summary, context.mean(dim=1)], dim=-1)
        
        # Generate new step
        new_step = self.step_generator(combined_input)
        
        return new_step
        
    def validate_reasoning_step(
        self,
        step: ReasoningStep,
        context: torch.Tensor,
        previous_steps: List[ReasoningStep]
    ) -> Tuple[bool, float]:
        """
        Validate a reasoning step for coherence and logical consistency.
        
        Args:
            step: ReasoningStep to validate
            context: Context tensor
            previous_steps: Previous steps for dependency checking
            
        Returns:
            Tuple of (is_valid, confidence_score)
        """
        if not self.enable_step_validation:
            return True, 0.8  # Default confidence if validation disabled
            
        # Use the validator to check step consistency
        is_valid, confidence = self.validator.validate_step(
            step, context, previous_steps
        )
        
        return is_valid, confidence
        
    def synthesize_final_answer(
        self,
        reasoning_steps: List[ReasoningStep],
        original_query: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """
        Synthesize final answer from reasoning steps.
        
        Args:
            reasoning_steps: Complete chain of reasoning steps
            original_query: Original query tensor
            context: Context information tensor
            
        Returns:
            Final answer tensor: [batch_size, hidden_size]
        """
        if not reasoning_steps:
            # No reasoning steps, use direct approach
            return original_query
            
        # Aggregate reasoning information
        step_embeddings = torch.stack([
            step.content for step in reasoning_steps
        ], dim=1)  # [batch_size, num_steps, reasoning_dim]
        
        # Attention-based aggregation
        query_expanded = original_query.unsqueeze(1).expand(-1, step_embeddings.size(1), -1)
        
        # Compute attention weights
        attention_scores = torch.sum(query_expanded * step_embeddings, dim=-1)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Weighted aggregation
        aggregated_reasoning = torch.sum(
            step_embeddings * attention_weights.unsqueeze(-1), dim=1
        )
        
        # Synthesize final answer
        combined_input = torch.cat([original_query, aggregated_reasoning, context.mean(dim=1)], dim=-1)
        final_answer = self.answer_synthesizer(combined_input)
        
        return final_answer
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        reasoning_prompts: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the CoT processor.
        
        Args:
            input_ids: [batch_size, seq_len] - input token IDs
            attention_mask: [batch_size, seq_len] - attention mask
            labels: [batch_size, seq_len] - target labels for supervised learning
            reasoning_prompts: Optional reasoning prompts to guide CoT
            
        Returns:
            Dict containing outputs, reasoning chains, and auxiliary information
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # Initialize reasoning chains
        reasoning_chains = []
        
        # Process each example in the batch
        for batch_idx in range(batch_size):
            chain = self._process_single_example(
                input_ids[batch_idx],
                attention_mask[batch_idx],
                labels[batch_idx] if labels is not None else None,
                reasoning_prompts[batch_idx] if reasoning_prompts else None,
                **kwargs
            )
            reasoning_chains.append(chain)
            
        # Aggregate results
        outputs = self._aggregate_outputs(reasoning_chains, input_ids, attention_mask, labels)
        
        return outputs
        
    def _process_single_example(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        reasoning_prompt: Optional[str] = None,
        **kwargs
    ) -> ReasoningChain:
        """Process a single example through the CoT pipeline."""
        # This would integrate with the actual model embeddings
        # For now, create placeholder embeddings
        embeddings = torch.randn(1, input_ids.size(0), self.hidden_size, device=input_ids.device)
        
        chain_id = f"chain_{hash(str(input_ids.cpu().numpy()))}"
        reasoning_steps = []
        
        # Generate reasoning steps iteratively
        for step_idx in range(self.max_reasoning_steps):
            # Generate step
            step_content = self.generate_reasoning_step(
                query=embeddings.mean(dim=1),
                context=embeddings,
                previous_steps=reasoning_steps
            )
            
            # Get step quality and type
            step_state = self.state_tracker(
                hidden_states=embeddings,
                reasoning_context=step_content.unsqueeze(0)
            )
            
            quality = step_state['step_quality'][0, 0].item()
            reasoning_type_idx = torch.argmax(step_state['reasoning_types'][0, 0])
            reasoning_types = ['logical', 'mathematical', 'inductive', 'deductive', 'analogic']
            reasoning_type = reasoning_types[reasoning_type_idx]
            
            # Create reasoning step
            step = ReasoningStep(
                step_id=step_idx,
                content=step_content,
                confidence=quality,
                reasoning_type=reasoning_type,
                dependencies=[],  # Would be determined based on actual logic
                next_steps=[],
                metadata={}
            )
            
            # Validate step
            is_valid, validation_confidence = self.validate_reasoning_step(
                step, embeddings, reasoning_steps
            )
            
            if not is_valid and validation_confidence < 0.5:
                break  # Stop if step is invalid
                
            reasoning_steps.append(step)
            
            # Early termination condition (e.g., high confidence, reaching answer)
            if quality > 0.9 or step_idx >= self.max_reasoning_steps - 1:
                break
                
        # Synthesize final answer
        final_answer = self.synthesize_final_answer(
            reasoning_steps, 
            embeddings.mean(dim=1),
            embeddings
        )
        
        # Create reasoning chain
        chain = ReasoningChain(
            chain_id=chain_id,
            question=embeddings.mean(dim=1),
            final_answer=final_answer,
            steps=reasoning_steps,
            reasoning_metadata={},
            quality_score=sum(step.confidence for step in reasoning_steps) / len(reasoning_steps),
            is_valid=True
        )
        
        return chain
        
    def _aggregate_outputs(
        self,
        chains: List[ReasoningChain],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Aggregate outputs from multiple reasoning chains."""
        batch_size = input_ids.size(0)
        
        # Extract final answers
        final_answers = torch.stack([chain.final_answer for chain in chains], dim=0)
        
        # Extract step information
        all_steps = []
        all_step_qualities = []
        all_step_types = []
        
        for chain in chains:
            chain_steps = []
            chain_qualities = []
            chain_types = []
            
            for step in chain.steps:
                chain_steps.append(step.content)
                chain_qualities.append(step.confidence)
                chain_types.append(step.reasoning_type)
                
            # Pad to max steps
            while len(chain_steps) < self.max_reasoning_steps:
                chain_steps.append(torch.zeros(self.reasoning_dim, device=input_ids.device))
                chain_qualities.append(0.0)
                chain_types.append('none')
                
            all_steps.append(torch.stack(chain_steps, dim=1))
            all_step_qualities.append(torch.tensor(chain_qualities[:self.max_reasoning_steps]))
            all_step_types.append(chain_types[:self.max_reasoning_steps])
            
        # Stack step information
        step_embeddings = torch.stack(all_steps, dim=0)  # [batch_size, max_steps, reasoning_dim]
        step_qualities = torch.stack(all_step_qualities, dim=0)  # [batch_size, max_steps]
        
        outputs = {
            'logits': final_answers,  # Final answer predictions
            'reasoning_steps': step_embeddings,  # Step embeddings
            'step_qualities': step_qualities,  # Quality scores
            'reasoning_chains': chains,  # Full reasoning chains
            'num_steps': torch.tensor([len(chain.steps) for chain in chains]),
            'chain_quality': torch.tensor([chain.quality_score for chain in chains])
        }
        
        return outputs


class ReasoningAwareLoss(nn.Module):
    """
    Loss function that considers reasoning chain quality, step coherence,
    and final answer accuracy.
    """
    
    def __init__(
        self,
        reasoning_weight: float = 0.3,
        step_coherence_weight: float = 0.2,
        answer_weight: float = 0.5,
        use_reasoning_supervision: bool = True,
        **kwargs
    ):
        super().__init__()
        self.reasoning_weight = reasoning_weight
        self.step_coherence_weight = step_coherence_weight
        self.answer_weight = answer_weight
        self.use_reasoning_supervision = use_reasoning_supervision
        
        # Component loss functions
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute reasoning-aware loss.
        
        Args:
            outputs: Model outputs from ChainOfThoughtProcessor
            targets: Target information including final answers and reasoning chains
            
        Returns:
            Dict containing total loss and component losses
        """
        losses = {}
        
        # Final answer loss
        if 'final_answer_targets' in targets:
            answer_loss = self.cross_entropy_loss(
                outputs['logits'], targets['final_answer_targets']
            )
            losses['answer_loss'] = answer_loss
        else:
            answer_loss = torch.tensor(0.0, device=outputs['logits'].device)
            
        # Reasoning step quality loss
        if 'step_quality_targets' in targets and 'step_qualities' in outputs:
            step_quality_loss = self.mse_loss(
                outputs['step_qualities'], targets['step_quality_targets']
            )
            losses['step_quality_loss'] = step_quality_loss
        else:
            step_quality_loss = torch.tensor(0.0, device=outputs['logits'].device)
            
        # Step coherence loss (if reasoning chains are provided)
        coherence_loss = torch.tensor(0.0, device=outputs['logits'].device)
        if 'reasoning_chains' in outputs and 'reasoning_chain_targets' in targets:
            for chain_output, chain_target in zip(outputs['reasoning_chains'], targets['reasoning_chain_targets']):
                chain_coherence = self._compute_chain_coherence(chain_output, chain_target)
                coherence_loss += chain_coherence
                
        losses['coherence_loss'] = coherence_loss
        
        # Reasoning diversity loss (encourage diverse reasoning approaches)
        diversity_loss = self._compute_diversity_loss(outputs['reasoning_steps'])
        losses['diversity_loss'] = diversity_loss
        
        # Weighted total loss
        total_loss = (
            self.answer_weight * answer_loss +
            self.reasoning_weight * step_quality_loss +
            self.step_coherence_weight * coherence_loss +
            0.1 * diversity_loss  # Small weight for diversity
        )
        
        losses['total_loss'] = total_loss
        
        return losses
        
    def _compute_chain_coherence(
        self,
        chain_output: ReasoningChain,
        chain_target: ReasoningChain
    ) -> torch.Tensor:
        """Compute coherence loss for a reasoning chain."""
        if len(chain_output.steps) < 2 or len(chain_target.steps) < 2:
            return torch.tensor(0.0, device=chain_output.final_answer.device)
            
        # Extract step embeddings
        output_steps = torch.stack([step.content for step in chain_output.steps])
        target_steps = torch.stack([step.content for step in chain_target.steps])
        
        # Compute similarity between consecutive steps
        coherence_scores = []
        for i in range(min(len(output_steps), len(target_steps)) - 1):
            # Cosine similarity between consecutive steps
            similarity = F.cosine_similarity(
                output_steps[i:i+2].unsqueeze(0),
                target_steps[i:i+2].unsqueeze(0)
            )
            coherence_scores.append(similarity)
            
        if coherence_scores:
            # Loss is 1 - average coherence
            avg_coherence = torch.stack(coherence_scores).mean()
            coherence_loss = 1.0 - avg_coherence
        else:
            coherence_loss = torch.tensor(0.0, device=chain_output.final_answer.device)
            
        return coherence_loss
        
    def _compute_diversity_loss(self, reasoning_steps: torch.Tensor) -> torch.Tensor:
        """Compute diversity loss to encourage varied reasoning approaches."""
        # Compute pairwise similarities between reasoning steps
        batch_size, max_steps, dim = reasoning_steps.size()
        
        # Reshape for computation
        steps_flat = reasoning_steps.view(batch_size * max_steps, dim)
        
        # Compute pairwise cosine similarities
        similarities = torch.mm(steps_flat, steps_flat.t())
        
        # Create mask for valid steps (non-zero embeddings)
        step_norms = torch.norm(steps_flat, dim=1, keepdim=True)
        step_norms = torch.clamp(step_norms, min=1e-8)
        normalized_steps = steps_flat / step_norms
        
        similarities = torch.mm(normalized_steps, normalized_steps.t())
        
        # Remove diagonal (self-similarities)
        mask = torch.eye(max_steps * batch_size, device=similarities.device).bool()
        off_diagonal_similarities = similarities[~mask]
        
        # Diversity loss encourages low similarity between different steps
        diversity_loss = torch.mean(off_diagonal_similarities)
        
        return diversity_loss