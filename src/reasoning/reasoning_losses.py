"""
Specialized Reasoning Loss Functions

This module implements sophisticated loss functions specifically designed for
training reasoning models, including:
- Reasoning-aware loss functions that reward correct reasoning chains
- Multi-task loss functions for different reasoning components
- Consistency losses that ensure logical coherence
- Quality-aware losses that assess reasoning step quality
- Hierarchical losses for nested reasoning structures
- Contrastive losses for reasoning diversity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import math
import numpy as np
from collections import defaultdict

from .chain_of_thought import ReasoningChain, ReasoningStep, ReasoningAwareLoss
from .symbolic_reasoning import SymbolicExpression
from .multimodal_reasoning import ModalityRepresentation


class ReasoningQualityLoss(nn.Module):
    """
    Loss function that evaluates the quality of reasoning chains.
    
    Assesses reasoning based on:
    - Logical consistency
    - Step coherence
    - Answer accuracy
    - Reasoning completeness
    """
    
    def __init__(
        self,
        coherence_weight: float = 0.3,
        consistency_weight: float = 0.3,
        accuracy_weight: float = 0.4,
        use_hard_negative_mining: bool = True,
        temperature: float = 1.0,
        **kwargs
    ):
        super().__init__()
        self.coherence_weight = coherence_weight
        self.consistency_weight = consistency_weight
        self.accuracy_weight = accuracy_weight
        self.use_hard_negative_mining = use_hard_negative_mining
        self.temperature = temperature
        
        # Quality assessment components
        self.coherence_scorer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.consistency_scorer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Hard negative mining
        if use_hard_negative_mining:
            self.negative_sampler = nn.Linear(256, 128)
        
    def forward(
        self,
        reasoning_chains: List[ReasoningChain],
        predictions: torch.Tensor,
        targets: torch.Tensor,
        step_representations: Optional[List[torch.Tensor]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute reasoning quality loss.
        
        Args:
            reasoning_chains: List of reasoning chains
            predictions: Final answer predictions
            targets: Ground truth answers
            step_representations: Representations of reasoning steps
            
        Returns:
            Dict containing loss components
        """
        batch_size = len(reasoning_chains)
        device = predictions.device
        
        losses = {}
        
        # 1. Answer accuracy loss
        accuracy_loss = self._compute_accuracy_loss(predictions, targets)
        losses['accuracy_loss'] = accuracy_loss
        
        # 2. Reasoning coherence loss
        coherence_loss = self._compute_coherence_loss(reasoning_chains, step_representations)
        losses['coherence_loss'] = coherence_loss
        
        # 3. Reasoning consistency loss
        consistency_loss = self._compute_consistency_loss(reasoning_chains)
        losses['consistency_loss'] = consistency_loss
        
        # 4. Overall reasoning quality loss
        total_quality_loss = (
            self.accuracy_weight * accuracy_loss +
            self.coherence_weight * coherence_loss +
            self.consistency_weight * consistency_loss
        )
        
        losses['total_quality_loss'] = total_quality_loss
        
        return losses
        
    def _compute_accuracy_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute final answer accuracy loss."""
        if predictions.shape == targets.shape:
            # Classification loss
            return F.cross_entropy(predictions, targets, reduction='mean')
        else:
            # Regression loss
            return F.mse_loss(predictions.squeeze(), targets.float(), reduction='mean')
            
    def _compute_coherence_loss(
        self,
        reasoning_chains: List[ReasoningChain],
        step_representations: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """Compute coherence loss for reasoning steps."""
        coherence_losses = []
        
        for chain in reasoning_chains:
            if len(chain.steps) < 2:
                continue
                
            chain_coherence = 0.0
            num_pairs = 0
            
            # Compute coherence between consecutive steps
            for i in range(len(chain.steps) - 1):
                step1 = chain.steps[i]
                step2 = chain.steps[i + 1]
                
                # Compute step similarity
                if step_representations is not None:
                    # Use provided representations
                    rep1 = step_representations[i]
                    rep2 = step_representations[i + 1]
                else:
                    # Use step content
                    rep1 = step1.content
                    rep2 = step2.content
                
                # Cosine similarity
                similarity = F.cosine_similarity(rep1, rep2, dim=0)
                
                # Coherence score (higher similarity = better coherence)
                coherence_score = torch.clamp(similarity, min=0.0)
                chain_coherence += (1.0 - coherence_score)  # Lower is better
                num_pairs += 1
            
            if num_pairs > 0:
                chain_coherence = chain_coherence / num_pairs
                coherence_losses.append(chain_coherence)
        
        return torch.mean(torch.stack(coherence_losses)) if coherence_losses else torch.tensor(0.0)
        
    def _compute_consistency_loss(
        self,
        reasoning_chains: List[ReasoningChain]
    ) -> torch.Tensor:
        """Compute logical consistency loss."""
        consistency_losses = []
        
        for chain in reasoning_chains:
            # Check for logical contradictions
            inconsistencies = 0
            total_checks = 0
            
            for i, step1 in enumerate(chain.steps):
                for j, step2 in enumerate(chain.steps[i+1:], i+1):
                    # Simple consistency check based on confidence and type
                    if (step1.reasoning_type == step2.reasoning_type and
                        abs(step1.confidence - step2.confidence) > 0.5):
                        inconsistencies += 1
                    total_checks += 1
            
            if total_checks > 0:
                inconsistency_ratio = inconsistencies / total_checks
                consistency_losses.append(inconsistency_ratio)
        
        return torch.mean(torch.stack(consistency_losses)) if consistency_losses else torch.tensor(0.0)


class ReasoningDiversityLoss(nn.Module):
    """
    Loss function that encourages diverse reasoning approaches.
    
    Uses contrastive learning to promote reasoning diversity
    while maintaining quality.
    """
    
    def __init__(
        self,
        diversity_weight: float = 0.1,
        margin: float = 1.0,
        use_hard_negative_mining: bool = True,
        **kwargs
    ):
        super().__init__()
        self.diversity_weight = diversity_weight
        self.margin = margin
        self.use_hard_negative_mining = use_hard_negative_mining
        
        # Diversity assessment network
        self.diversity_scorer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(
        self,
        reasoning_chains: List[ReasoningChain],
        step_representations: List[torch.Tensor],
        hard_negatives: Optional[List[torch.Tensor]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute reasoning diversity loss.
        
        Args:
            reasoning_chains: List of reasoning chains
            step_representations: Representations of reasoning steps
            hard_negatives: Hard negative examples
            
        Returns:
            Dict containing diversity loss components
        """
        batch_size = len(reasoning_chains)
        
        losses = {}
        
        # 1. Intra-chain diversity loss (diversity within a single reasoning chain)
        intra_diversity_loss = self._compute_intra_chain_diversity(
            reasoning_chains, step_representations
        )
        losses['intra_diversity_loss'] = intra_diversity_loss
        
        # 2. Inter-chain diversity loss (diversity between different chains)
        inter_diversity_loss = self._compute_inter_chain_diversity(
            reasoning_chains, step_representations
        )
        losses['inter_diversity_loss'] = inter_diversity_loss
        
        # 3. Hard negative mining loss if enabled
        if self.use_hard_negative_mining and hard_negatives is not None:
            hard_negative_loss = self._compute_hard_negative_loss(
                step_representations, hard_negatives
            )
            losses['hard_negative_loss'] = hard_negative_loss
        else:
            losses['hard_negative_loss'] = torch.tensor(0.0)
        
        # 4. Total diversity loss
        total_diversity_loss = (
            intra_diversity_loss + 
            inter_diversity_loss + 
            self.diversity_weight * losses['hard_negative_loss']
        )
        
        losses['total_diversity_loss'] = total_diversity_loss
        
        return losses
        
    def _compute_intra_chain_diversity(
        self,
        reasoning_chains: List[ReasoningChain],
        step_representations: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute diversity within individual reasoning chains."""
        diversity_losses = []
        
        for chain_idx, chain in enumerate(reasoning_chains):
            if len(chain.steps) < 2:
                continue
                
            chain_steps = step_representations[chain_idx]
            
            # Compute pairwise similarities within the chain
            similarities = []
            for i in range(len(chain_steps)):
                for j in range(i + 1, len(chain_steps)):
                    similarity = F.cosine_similarity(chain_steps[i], chain_steps[j], dim=0)
                    similarities.append(similarity)
            
            if similarities:
                # Diversity loss: encourage lower similarities
                avg_similarity = torch.mean(torch.stack(similarities))
                diversity_loss = torch.clamp(avg_similarity, max=self.margin)
                diversity_losses.append(diversity_loss)
        
        return torch.mean(torch.stack(diversity_losses)) if diversity_losses else torch.tensor(0.0)
        
    def _compute_inter_chain_diversity(
        self,
        reasoning_chains: List[ReasoningChain],
        step_representations: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute diversity between different reasoning chains."""
        if len(reasoning_chains) < 2:
            return torch.tensor(0.0)
        
        # Sample pairs of chains for comparison
        chain_pairs = []
        for i in range(len(reasoning_chains)):
            for j in range(i + 1, len(reasoning_chains)):
                chain_pairs.append((i, j))
        
        diversity_scores = []
        for chain1_idx, chain2_idx in chain_pairs:
            # Get representative steps from each chain
            chain1_steps = step_representations[chain1_idx]
            chain2_steps = step_representations[chain2_idx]
            
            # Compute cross-chain similarities
            cross_similarities = []
            for step1 in chain1_steps:
                for step2 in chain2_steps:
                    similarity = F.cosine_similarity(step1, step2, dim=0)
                    cross_similarities.append(similarity)
            
            if cross_similarities:
                # Diversity score: encourage lower cross-chain similarity
                avg_cross_similarity = torch.mean(torch.stack(cross_similarities))
                diversity_score = torch.clamp(avg_cross_similarity, max=self.margin)
                diversity_scores.append(diversity_score)
        
        return torch.mean(torch.stack(diversity_scores)) if diversity_scores else torch.tensor(0.0)
        
    def _compute_hard_negative_loss(
        self,
        positive_representations: List[torch.Tensor],
        hard_negatives: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute loss using hard negative mining."""
        if len(hard_negatives) == 0:
            return torch.tensor(0.0)
        
        triplet_losses = []
        
        for pos_repr in positive_representations:
            for neg_repr in hard_negatives:
                # Compute triplet-like loss
                similarity_pos = torch.mean(pos_repr)  # Simplified
                similarity_neg = torch.mean(neg_repr)  # Simplified
                
                loss = F.relu(similarity_neg - similarity_pos + self.margin)
                triplet_losses.append(loss)
        
        return torch.mean(torch.stack(triplet_losses)) if triplet_losses else torch.tensor(0.0)


class SymbolicReasoningLoss(nn.Module):
    """
    Loss function for symbolic reasoning operations.
    
    Specifically designed to train symbolic reasoning components
    and ensure mathematical/logical correctness.
    """
    
    def __init__(
        self,
        symbolic_weight: float = 0.2,
        mathematical_accuracy_weight: float = 0.5,
        logical_consistency_weight: float = 0.3,
        **kwargs
    ):
        super().__init__()
        self.symbolic_weight = symbolic_weight
        self.mathematical_accuracy_weight = mathematical_accuracy_weight
        self.logical_consistency_weight = logical_consistency_weight
        
        # Symbolic operation accuracy assessor
        self.symbolic_evaluator = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Mathematical consistency checker
        self.math_consistency_checker = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        symbolic_expressions: List[SymbolicExpression],
        symbolic_outputs: Dict[str, torch.Tensor],
        ground_truth_expressions: Optional[List[SymbolicExpression]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute symbolic reasoning loss.
        
        Args:
            symbolic_expressions: List of symbolic expressions
            symbolic_outputs: Outputs from symbolic reasoning engine
            ground_truth_expressions: Ground truth symbolic expressions
            
        Returns:
            Dict containing symbolic reasoning loss components
        """
        losses = {}
        
        # 1. Symbolic operation accuracy loss
        symbolic_accuracy_loss = self._compute_symbolic_accuracy_loss(
            symbolic_expressions, symbolic_outputs
        )
        losses['symbolic_accuracy_loss'] = symbolic_accuracy_loss
        
        # 2. Mathematical consistency loss
        math_consistency_loss = self._compute_math_consistency_loss(
            symbolic_expressions, symbolic_outputs
        )
        losses['math_consistency_loss'] = math_consistency_loss
        
        # 3. Expression correctness loss (if ground truth available)
        if ground_truth_expressions is not None:
            expression_correctness_loss = self._compute_expression_correctness_loss(
                symbolic_expressions, ground_truth_expressions, symbolic_outputs
            )
            losses['expression_correctness_loss'] = expression_correctness_loss
        else:
            losses['expression_correctness_loss'] = torch.tensor(0.0)
        
        # 4. Total symbolic reasoning loss
        total_symbolic_loss = (
            self.mathematical_accuracy_weight * symbolic_accuracy_loss +
            self.logical_consistency_weight * math_consistency_loss +
            self.symbolic_weight * losses['expression_correctness_loss']
        )
        
        losses['total_symbolic_loss'] = total_symbolic_loss
        
        return losses
        
    def _compute_symbolic_accuracy_loss(
        self,
        symbolic_expressions: List[SymbolicExpression],
        symbolic_outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute accuracy loss for symbolic operations."""
        accuracy_scores = []
        
        for expr in symbolic_expressions:
            # Use symbolic evaluator to assess expression quality
            expr_embedding = self._expression_to_embedding(expr)
            accuracy_score = self.symbolic_evaluator(expr_embedding)
            accuracy_scores.append(accuracy_score)
        
        # Loss is 1 - accuracy (encourage high accuracy)
        losses = [1.0 - score for score in accuracy_scores]
        return torch.mean(torch.stack(losses))
        
    def _compute_math_consistency_loss(
        self,
        symbolic_expressions: List[SymbolicExpression],
        symbolic_outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute mathematical consistency loss."""
        consistency_scores = []
        
        for i, expr1 in enumerate(symbolic_expressions):
            for j, expr2 in enumerate(symbolic_expressions[i+1:], i+1):
                # Check consistency between expressions
                consistency_score = self._check_expression_consistency(expr1, expr2)
                consistency_scores.append(consistency_score)
        
        if consistency_scores:
            # Loss is 1 - consistency (encourage consistency)
            losses = [1.0 - score for score in consistency_scores]
            return torch.mean(torch.stack(losses))
        else:
            return torch.tensor(0.0)
        
    def _compute_expression_correctness_loss(
        self,
        predicted_expressions: List[SymbolicExpression],
        ground_truth_expressions: List[SymbolicExpression],
        symbolic_outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute correctness loss comparing to ground truth."""
        correctness_losses = []
        
        for pred_expr, gt_expr in zip(predicted_expressions, ground_truth_expressions):
            # Compare expressions structurally and semantically
            structural_similarity = self._compute_structural_similarity(pred_expr, gt_expr)
            semantic_similarity = self._compute_semantic_similarity(pred_expr, gt_expr)
            
            # Combined correctness score
            correctness_score = (structural_similarity + semantic_similarity) / 2
            loss = 1.0 - correctness_score
            correctness_losses.append(loss)
        
        return torch.mean(torch.stack(correctness_losses))
        
    def _expression_to_embedding(self, expression: SymbolicExpression) -> torch.Tensor:
        """Convert symbolic expression to embedding."""
        # Simple embedding based on operation type and structure
        embedding = torch.zeros(256)
        
        # Operation type encoding
        if hasattr(expression, 'operation'):
            op_hash = hash(str(expression.operation))
            embedding[op_hash % 256] = 1.0
        
        # Structure encoding
        embedding[128 + len(expression.operands) % 128] = 1.0
        
        return embedding
        
    def _check_expression_consistency(
        self,
        expr1: SymbolicExpression,
        expr2: SymbolicExpression
    ) -> torch.Tensor:
        """Check consistency between two expressions."""
        # Simplified consistency check
        embeddings1 = self._expression_to_embedding(expr1)
        embeddings2 = self._expression_to_embedding(expr2)
        
        similarity = F.cosine_similarity(embeddings1, embeddings2, dim=0)
        return torch.clamp(similarity, min=0.0, max=1.0)
        
    def _compute_structural_similarity(
        self,
        expr1: SymbolicExpression,
        expr2: SymbolicExpression
    ) -> float:
        """Compute structural similarity between expressions."""
        # Simplified structural similarity
        if expr1.operation == expr2.operation and len(expr1.operands) == len(expr2.operands):
            return 1.0
        else:
            return 0.0
        
    def _compute_semantic_similarity(
        self,
        expr1: SymbolicExpression,
        expr2: SymbolicExpression
    ) -> float:
        """Compute semantic similarity between expressions."""
        # Simplified semantic similarity using embeddings
        emb1 = self._expression_to_embedding(expr1)
        emb2 = self._expression_to_embedding(expr2)
        
        similarity = F.cosine_similarity(emb1, emb2, dim=0)
        return float(torch.clamp(similarity, min=0.0, max=1.0))


class MultiModalReasoningLoss(nn.Module):
    """
    Loss function for multi-modal reasoning tasks.
    
    Ensures consistency and effectiveness across different modalities.
    """
    
    def __init__(
        self,
        cross_modal_consistency_weight: float = 0.4,
        modality_specific_weight: float = 0.3,
        fusion_quality_weight: float = 0.3,
        **kwargs
    ):
        super().__init__()
        self.cross_modal_consistency_weight = cross_modal_consistency_weight
        self.modality_specific_weight = modality_specific_weight
        self.fusion_quality_weight = fusion_quality_weight
        
        # Cross-modal consistency assessor
        self.consistency_scorer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Modality-specific quality assessor
        self.modality_scorer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        modality_representations: Dict[str, ModalityRepresentation],
        cross_modal_attention_weights: Dict[Tuple[str, str], torch.Tensor],
        final_representation: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-modal reasoning loss.
        
        Args:
            modality_representations: Representations from different modalities
            cross_modal_attention_weights: Attention weights between modalities
            final_representation: Fused representation
            
        Returns:
            Dict containing multi-modal loss components
        """
        losses = {}
        
        # 1. Cross-modal consistency loss
        consistency_loss = self._compute_cross_modal_consistency_loss(
            modality_representations, cross_modal_attention_weights
        )
        losses['consistency_loss'] = consistency_loss
        
        # 2. Modality-specific quality loss
        modality_quality_loss = self._compute_modality_quality_loss(
            modality_representations
        )
        losses['modality_quality_loss'] = modality_quality_loss
        
        # 3. Fusion quality loss
        fusion_quality_loss = self._compute_fusion_quality_loss(
            modality_representations, final_representation
        )
        losses['fusion_quality_loss'] = fusion_quality_loss
        
        # 4. Total multi-modal loss
        total_multimodal_loss = (
            self.cross_modal_consistency_weight * consistency_loss +
            self.modality_specific_weight * modality_quality_loss +
            self.fusion_quality_weight * fusion_quality_loss
        )
        
        losses['total_multimodal_loss'] = total_multimodal_loss
        
        return losses
        
    def _compute_cross_modal_consistency_loss(
        self,
        modality_representations: Dict[str, ModalityRepresentation],
        cross_modal_attention_weights: Dict[Tuple[str, str], torch.Tensor]
    ) -> torch.Tensor:
        """Compute cross-modal consistency loss."""
        consistency_scores = []
        
        modalities = list(modality_representations.keys())
        
        for i, mod1 in enumerate(modalities):
            for j, mod2 in enumerate(modalities[i+1:], i+1):
                # Get modality representations
                repr1 = modality_representations[mod1].embedding
                repr2 = modality_representations[mod2].embedding
                
                # Compute consistency score
                consistency_score = self.consistency_scorer(
                    torch.cat([repr1, repr2], dim=0)
                )
                consistency_scores.append(consistency_score)
        
        if consistency_scores:
            # Loss is 1 - consistency (encourage high consistency)
            losses = [1.0 - score for score in consistency_scores]
            return torch.mean(torch.stack(losses))
        else:
            return torch.tensor(0.0)
        
    def _compute_modality_quality_loss(
        self,
        modality_representations: Dict[str, ModalityRepresentation]
    ) -> torch.Tensor:
        """Compute quality loss for individual modalities."""
        quality_scores = []
        
        for modality, representation in modality_representations.items():
            quality_score = self.modality_scorer(representation.embedding)
            quality_scores.append(quality_score)
        
        if quality_scores:
            # Loss is 1 - quality (encourage high quality)
            losses = [1.0 - score for score in quality_scores]
            return torch.mean(torch.stack(losses))
        else:
            return torch.tensor(0.0)
        
    def _compute_fusion_quality_loss(
        self,
        modality_representations: Dict[str, ModalityRepresentation],
        final_representation: torch.Tensor
    ) -> torch.Tensor:
        """Compute fusion quality loss."""
        # Simple quality assessment of the fused representation
        quality_score = self.modality_scorer(final_representation)
        
        # Loss is 1 - quality (encourage high quality fusion)
        return 1.0 - quality_score


class ComprehensiveReasoningLoss(nn.Module):
    """
    Comprehensive loss function that combines all reasoning components.
    
    Integrates multiple reasoning loss functions to provide
    a unified training objective for reasoning models.
    """
    
    def __init__(
        self,
        quality_weight: float = 0.3,
        diversity_weight: float = 0.1,
        symbolic_weight: float = 0.2,
        multimodal_weight: float = 0.2,
        main_task_weight: float = 0.2,
        use_adaptive_weights: bool = True,
        **kwargs
    ):
        super().__init__()
        self.quality_weight = quality_weight
        self.diversity_weight = diversity_weight
        self.symbolic_weight = symbolic_weight
        self.multimodal_weight = multimodal_weight
        self.main_task_weight = main_task_weight
        self.use_adaptive_weights = use_adaptive_weights
        
        # Initialize component losses
        self.quality_loss = ReasoningQualityLoss()
        self.diversity_loss = ReasoningDiversityLoss()
        self.symbolic_loss = SymbolicReasoningLoss()
        self.multimodal_loss = MultiModalReasoningLoss()
        
        # Adaptive weight controller
        if use_adaptive_weights:
            self.weight_controller = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 4)  # 4 component losses
            )
        
    def forward(
        self,
        reasoning_chains: List[ReasoningChain],
        predictions: torch.Tensor,
        targets: torch.Tensor,
        step_representations: Optional[List[torch.Tensor]] = None,
        symbolic_expressions: Optional[List[SymbolicExpression]] = None,
        symbolic_outputs: Optional[Dict[str, torch.Tensor]] = None,
        modality_representations: Optional[Dict[str, ModalityRepresentation]] = None,
        cross_modal_attention: Optional[Dict] = None,
        final_representation: Optional[torch.Tensor] = None,
        context_embedding: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive reasoning loss.
        
        Args:
            reasoning_chains: List of reasoning chains
            predictions: Final predictions
            targets: Ground truth targets
            step_representations: Reasoning step representations
            symbolic_expressions: Symbolic expressions
            symbolic_outputs: Symbolic reasoning outputs
            modality_representations: Multi-modal representations
            cross_modal_attention: Cross-modal attention weights
            final_representation: Final fused representation
            context_embedding: Context embedding for adaptive weights
            
        Returns:
            Dict containing all loss components
        """
        all_losses = {}
        
        # 1. Main task loss (primary objective)
        if predictions.shape == targets.shape:
            main_task_loss = F.cross_entropy(predictions, targets, reduction='mean')
        else:
            main_task_loss = F.mse_loss(predictions.squeeze(), targets.float(), reduction='mean')
        all_losses['main_task_loss'] = main_task_loss
        
        # 2. Reasoning quality loss
        if reasoning_chains and step_representations:
            quality_losses = self.quality_loss(
                reasoning_chains, predictions, targets, step_representations
            )
            all_losses.update({f'quality_{k}': v for k, v in quality_losses.items()})
        
        # 3. Reasoning diversity loss
        if reasoning_chains and step_representations:
            diversity_losses = self.diversity_loss(
                reasoning_chains, step_representations
            )
            all_losses.update({f'diversity_{k}': v for k, v in diversity_losses.items()})
        
        # 4. Symbolic reasoning loss
        if symbolic_expressions and symbolic_outputs:
            symbolic_losses = self.symbolic_loss(
                symbolic_expressions, symbolic_outputs
            )
            all_losses.update({f'symbolic_{k}': v for k, v in symbolic_losses.items()})
        
        # 5. Multi-modal reasoning loss
        if (modality_representations and cross_modal_attention and 
            final_representation is not None):
            multimodal_losses = self.multimodal_loss(
                modality_representations, cross_modal_attention, final_representation
            )
            all_losses.update({f'multimodal_{k}': v for k, v in multimodal_losses.items()})
        
        # 6. Adaptive weight computation
        if self.use_adaptive_weights and context_embedding is not None:
            adaptive_weights = self._compute_adaptive_weights(context_embedding)
            all_losses['adaptive_weights'] = adaptive_weights
        else:
            adaptive_weights = torch.tensor([self.quality_weight, self.diversity_weight, 
                                          self.symbolic_weight, self.multimodal_weight])
        
        # 7. Weighted combination
        total_loss = main_task_loss * self.main_task_weight
        
        # Add weighted reasoning losses
        if 'quality_total_quality_loss' in all_losses:
            total_loss += adaptive_weights[0] * all_losses['quality_total_quality_loss']
        if 'diversity_total_diversity_loss' in all_losses:
            total_loss += adaptive_weights[1] * all_losses['diversity_total_diversity_loss']
        if 'symbolic_total_symbolic_loss' in all_losses:
            total_loss += adaptive_weights[2] * all_losses['symbolic_total_symbolic_loss']
        if 'multimodal_total_multimodal_loss' in all_losses:
            total_loss += adaptive_weights[3] * all_losses['multimodal_total_multimodal_loss']
        
        all_losses['total_reasoning_loss'] = total_loss
        
        return all_losses
        
    def _compute_adaptive_weights(self, context_embedding: torch.Tensor) -> torch.Tensor:
        """Compute adaptive weights based on context."""
        # Ensure context embedding has correct shape
        if context_embedding.dim() == 1:
            context_embedding = context_embedding.unsqueeze(0)
        
        # Compute adaptive weights
        raw_weights = self.weight_controller(context_embedding)
        adaptive_weights = F.softmax(raw_weights, dim=-1)
        
        return adaptive_weights.squeeze(0)