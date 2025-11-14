"""
Reasoning Memory Management System

This module provides sophisticated memory management for reasoning processes including:
- Long-term memory storage for reasoning patterns
- Working memory for current reasoning chains
- Episodic memory for reasoning experiences
- Memory retrieval and attention mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
import numpy as np
from dataclasses import dataclass
from collections import defaultdict, deque
import math


class MemoryEntry(NamedTuple):
    """Represents a single memory entry in the reasoning system."""
    key: torch.Tensor
    value: torch.Tensor
    metadata: Dict[str, Any]
    timestamp: float
    access_count: int = 0
    last_accessed: float = 0.0


@dataclass
class ReasoningMemoryConfig:
    """Configuration for reasoning memory system."""
    embedding_dim: int = 256
    max_memory_size: int = 1000
    working_memory_size: int = 50
    attention_heads: int = 8
    memory_dropout: float = 0.1
    retrieval_threshold: float = 0.7
    update_strategy: str = "attention_based"  # attention_based, recency_based, importance_based


class AttentionBasedMemory(nn.Module):
    """
    Attention-based memory retrieval and storage system.
    
    Uses attention mechanisms to store, retrieve, and update memory entries
    based on relevance to current reasoning context.
    """
    
    def __init__(self, config: ReasoningMemoryConfig):
        super().__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim
        self.max_memory_size = config.max_memory_size
        self.working_memory_size = config.working_memory_size
        
        # Memory storage
        self.memory_keys = nn.Parameter(
            torch.randn(self.max_memory_size, self.embedding_dim)
        )
        self.memory_values = nn.Parameter(
            torch.randn(self.max_memory_size, self.embedding_dim)
        )
        
        # Memory importance weights
        self.importance_weights = nn.Parameter(
            torch.ones(self.max_memory_size) / self.max_memory_size
        )
        
        # Memory access patterns tracking
        self.access_counts = torch.zeros(self.max_memory_size)
        self.timestamps = torch.zeros(self.max_memory_size)
        
        # Attention mechanisms for retrieval
        self.key_attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=config.attention_heads,
            dropout=config.memory_dropout,
            batch_first=True
        )
        
        self.value_attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=config.attention_heads,
            dropout=config.memory_dropout,
            batch_first=True
        )
        
        # Memory update gate
        self.update_gate = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.Sigmoid()
        )
        
        # Initialize memory entries
        self.reset_memory()
        
    def reset_memory(self):
        """Reset all memory entries."""
        self.memory_entries = [None] * self.max_memory_size
        self.working_memory = deque(maxlen=self.working_memory_size)
        self.lru_order = deque()
        
    def store_memory(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
        current_time: float = 0.0
    ) -> int:
        """
        Store a new memory entry using attention-based selection.
        
        Args:
            key: Memory key tensor [embedding_dim]
            value: Memory value tensor [embedding_dim]
            metadata: Additional metadata for the memory
            current_time: Current timestamp
            
        Returns:
            Index where memory was stored
        """
        batch_size = key.size(0) if key.dim() > 1 else 1
        if key.dim() == 1:
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
            
        # Compute memory importance using attention
        with torch.no_grad():
            # Score existing memories based on similarity
            similarities = F.cosine_similarity(
                key.unsqueeze(1), self.memory_keys.unsqueeze(0), dim=-1
            )
            
            # Find least important memory slot
            importance_scores = self.importance_weights * (1 + self.access_counts)
            replace_idx = torch.argmin(importance_scores).item()
            
        # Update memory
        self.memory_keys[replace_idx] = key.squeeze(0)
        self.memory_values[replace_idx] = value.squeeze(0)
        self.importance_weights.data[replace_idx] = 1.0
        self.access_counts[replace_idx] = 0
        self.timestamps[replace_idx] = current_time
        
        # Store metadata
        self.memory_entries[replace_idx] = MemoryEntry(
            key=key.squeeze(0),
            value=value.squeeze(0),
            metadata=metadata or {},
            timestamp=current_time
        )
        
        return replace_idx
        
    def retrieve_memory(
        self,
        query: torch.Tensor,
        top_k: int = 5,
        threshold: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve relevant memories using attention-based retrieval.
        
        Args:
            query: Query tensor [batch_size, embedding_dim]
            top_k: Number of memories to retrieve
            threshold: Similarity threshold for retrieval
            
        Returns:
            Tuple of (retrieved_keys, retrieved_values, attention_weights)
        """
        batch_size = query.size(0) if query.dim() > 1 else 1
        if query.dim() == 1:
            query = query.unsqueeze(0)
            
        threshold = threshold or self.config.retrieval_threshold
        
        # Compute attention scores
        attention_weights = F.softmax(
            torch.matmul(query, self.memory_keys.t()) / math.sqrt(self.embedding_dim),
            dim=-1
        )
        
        # Apply threshold
        if threshold > 0:
            mask = attention_weights > threshold
            attention_weights = attention_weights * mask.float()
            attention_weights = attention_weights / (attention_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Retrieve top-k memories
        top_k = min(top_k, self.max_memory_size)
        top_weights, top_indices = torch.topk(attention_weights, top_k, dim=-1)
        
        # Gather retrieved memories
        retrieved_keys = torch.gather(
            self.memory_keys.unsqueeze(0).expand(batch_size, -1, -1),
            1,
            top_indices.unsqueeze(-1).expand(-1, -1, self.embedding_dim)
        )
        
        retrieved_values = torch.gather(
            self.memory_values.unsqueeze(0).expand(batch_size, -1, -1),
            1,
            top_indices.unsqueeze(-1).expand(-1, -1, self.embedding_dim)
        )
        
        # Update access counts
        for i in range(batch_size):
            for idx in top_indices[i]:
                self.access_counts[idx.item()] += 1
                
        return retrieved_keys, retrieved_values, top_weights
        
    def update_memory(
        self,
        index: int,
        new_key: torch.Tensor,
        new_value: torch.Tensor,
        update_strength: float = 0.1
    ) -> None:
        """
        Update an existing memory entry.
        
        Args:
            index: Index of memory to update
            new_key: New key tensor
            new_value: New value tensor
            update_strength: Strength of the update (0-1)
        """
        # Interpolate between old and new values
        self.memory_keys[index] = (
            (1 - update_strength) * self.memory_keys[index] + 
            update_strength * new_key
        )
        
        self.memory_values[index] = (
            (1 - update_strength) * self.memory_values[index] + 
            update_strength * new_value
        )
        
        # Update importance
        self.importance_weights.data[index] += update_strength
        
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the memory system."""
        return {
            'total_entries': len([e for e in self.memory_entries if e is not None]),
            'average_importance': torch.mean(self.importance_weights).item(),
            'max_access_count': torch.max(self.access_counts).item(),
            'memory_utilization': len([e for e in self.memory_entries if e is not None]) / self.max_memory_size
        }


class WorkingMemory(nn.Module):
    """
    Working memory system for current reasoning operations.
    
    Maintains active reasoning steps, intermediate results, and
    current focus of attention during reasoning processes.
    """
    
    def __init__(self, config: ReasoningMemoryConfig):
        super().__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim
        self.capacity = config.working_memory_size
        
        # Working memory components
        self.working_buffer = nn.Parameter(
            torch.randn(self.capacity, self.embedding_dim)
        )
        
        # Attention mechanism for working memory management
        self.working_attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=config.attention_heads,
            dropout=config.memory_dropout,
            batch_first=True
        )
        
        # Memory consolidation mechanism
        self.consolidation_gate = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.Sigmoid()
        )
        
        self.current_size = 0
        self.active_indices = []
        
    def add_to_working_memory(
        self,
        item: torch.Tensor,
        priority: Optional[torch.Tensor] = None
    ) -> int:
        """
        Add item to working memory.
        
        Args:
            item: Item to add [embedding_dim]
            priority: Optional priority scores
            
        Returns:
            Index in working memory
        """
        if item.dim() == 1:
            item = item.unsqueeze(0)
            
        batch_size = item.size(0)
        if batch_size == 1:
            # Single item addition
            if self.current_size < self.capacity:
                # Add to next available slot
                index = self.current_size
                self.working_buffer[index] = item.squeeze(0)
                self.active_indices.append(index)
                self.current_size += 1
                return index
            else:
                # Replace least important item
                if priority is not None:
                    # Use priority to determine replacement
                    min_priority_idx = torch.argmin(priority).item()
                    index = self.active_indices[min_priority_idx]
                    self.working_buffer[index] = item.squeeze(0)
                    return index
                else:
                    # Replace oldest item (simple FIFO)
                    index = self.active_indices.pop(0)
                    self.working_buffer[index] = item.squeeze(0)
                    self.active_indices.append(index)
                    return index
        else:
            # Batch addition
            indices = []
            for i, single_item in enumerate(item):
                idx = self.add_to_working_memory(single_item, priority[i] if priority is not None else None)
                indices.append(idx)
            return indices
            
    def update_working_memory(
        self,
        index: int,
        new_item: torch.Tensor,
        update_strength: float = 0.5
    ) -> None:
        """Update item in working memory."""
        if new_item.dim() == 1:
            new_item = new_item.unsqueeze(0)
            
        self.working_buffer[index] = (
            (1 - update_strength) * self.working_buffer[index] + 
            update_strength * new_item.squeeze(0)
        )
        
    def get_attended_working_memory(
        self,
        query: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get attention-weighted view of working memory.
        
        Args:
            query: Query tensor for attention
            attention_mask: Optional mask for working memory items
            
        Returns:
            Tuple of (attended_memory, attention_weights)
        """
        if self.current_size == 0:
            return torch.zeros_like(query), torch.tensor([]).to(query.device)
            
        # Get active working memory
        active_memory = self.working_buffer[:self.current_size]
        
        # Compute attention
        attended_memory, attention_weights = self.working_attention(
            query.unsqueeze(1), active_memory.unsqueeze(0), active_memory.unsqueeze(0),
            key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
        )
        
        return attended_memory.squeeze(1), attention_weights
        
    def clear_working_memory(self):
        """Clear working memory."""
        self.working_buffer.data.zero_()
        self.current_size = 0
        self.active_indices.clear()


class EpisodicMemory(nn.Module):
    """
    Episodic memory system for storing and retrieving reasoning episodes.
    
    Manages complete reasoning episodes and their outcomes for
    future reference and pattern learning.
    """
    
    def __init__(self, config: ReasoningMemoryConfig):
        super().__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim
        self.max_episodes = config.max_memory_size // 2
        
        # Episode storage
        self.episode_keys = nn.Parameter(
            torch.randn(self.max_episodes, self.embedding_dim)
        )
        self.episode_values = nn.Parameter(
            torch.randn(self.max_episodes, self.embedding_dim * 2)  # Include outcome
        )
        
        # Episode metadata
        self.episode_metadata = []
        self.episode_success_rates = torch.zeros(self.max_episodes)
        self.episode_usage_counts = torch.zeros(self.max_episodes)
        
        self.current_episode_count = 0
        
    def store_episode(
        self,
        episode_key: torch.Tensor,
        episode_value: torch.Tensor,
        outcome: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
        success: bool = True
    ) -> int:
        """
        Store a reasoning episode.
        
        Args:
            episode_key: Key representing the episode [embedding_dim]
            episode_value: Episode content [embedding_dim]
            outcome: Episode outcome [embedding_dim]
            metadata: Episode metadata
            success: Whether episode was successful
            
        Returns:
            Index where episode was stored
        """
        if episode_key.dim() == 1:
            episode_key = episode_key.unsqueeze(0)
            episode_value = episode_value.unsqueeze(0)
            outcome = outcome.unsqueeze(0)
            
        # Combine episode and outcome
        combined_value = torch.cat([episode_value, outcome], dim=-1)
        
        if self.current_episode_count < self.max_episodes:
            # Add new episode
            index = self.current_episode_count
            self.episode_keys[index] = episode_key.squeeze(0)
            self.episode_values[index] = combined_value.squeeze(0)
            self.episode_success_rates[index] = 1.0 if success else 0.0
            self.episode_usage_counts[index] = 0
            self.episode_metadata.append(metadata or {})
            self.current_episode_count += 1
        else:
            # Replace episode with lowest success rate
            min_success_idx = torch.argmin(self.episode_success_rates).item()
            index = min_success_idx
            self.episode_keys[index] = episode_key.squeeze(0)
            self.episode_values[index] = combined_value.squeeze(0)
            self.episode_success_rates[index] = 1.0 if success else 0.0
            self.episode_metadata[index] = metadata or {}
            self.episode_usage_counts[index] = 0
            
        return index
        
    def retrieve_similar_episodes(
        self,
        query: torch.Tensor,
        top_k: int = 5,
        success_threshold: float = 0.7
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve episodes similar to query.
        
        Args:
            query: Query tensor
            top_k: Number of episodes to retrieve
            success_threshold: Minimum success rate to consider
            
        Returns:
            Tuple of (episode_keys, episode_values, success_rates)
        """
        if query.dim() == 1:
            query = query.unsqueeze(0)
            
        # Compute similarities
        similarities = F.cosine_similarity(
            query.unsqueeze(1), self.episode_keys[:self.current_episode_count].unsqueeze(0), dim=-1
        )
        
        # Filter by success rate
        valid_mask = self.episode_success_rates[:self.current_episode_count] >= success_threshold
        if valid_mask.sum() == 0:
            # If no high-success episodes, use all episodes
            valid_mask = torch.ones_like(valid_mask)
            
        # Apply mask and get top-k
        masked_similarities = similarities * valid_mask.float()
        top_weights, top_indices = torch.topk(masked_similarities, min(top_k, len(similarities)), dim=-1)
        
        # Update usage counts
        self.episode_usage_counts[top_indices] += 1
        
        # Retrieve episodes
        retrieved_keys = torch.gather(
            self.episode_keys[:self.current_episode_count].unsqueeze(0).expand(query.size(0), -1, -1),
            1,
            top_indices.unsqueeze(-1).expand(-1, -1, self.embedding_dim)
        )
        
        retrieved_values = torch.gather(
            self.episode_values[:self.current_episode_count].unsqueeze(0).expand(query.size(0), -1, -1),
            1,
            top_indices.unsqueeze(-1).expand(-1, -1, self.embedding_dim * 2)
        )
        
        retrieved_success = torch.gather(
            self.episode_success_rates[:self.current_episode_count].unsqueeze(0).expand(query.size(0), -1),
            1,
            top_indices
        )
        
        return retrieved_keys, retrieved_values, retrieved_success


class ReasoningMemory:
    """
    Main reasoning memory system that integrates long-term, working, and episodic memory.
    
    Provides unified interface for memory operations during reasoning processes.
    """
    
    def __init__(self, embedding_dim: int, max_memory_size: int = 1000):
        config = ReasoningMemoryConfig(
            embedding_dim=embedding_dim,
            max_memory_size=max_memory_size
        )
        
        self.embedding_dim = embedding_dim
        self.long_term_memory = AttentionBasedMemory(config)
        self.working_memory = WorkingMemory(config)
        self.episodic_memory = EpisodicMemory(config)
        
        # Memory integration layers
        self.memory_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def store_reasoning_step(
        self,
        step_key: torch.Tensor,
        step_value: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, int]:
        """
        Store reasoning step in all relevant memory systems.
        
        Args:
            step_key: Key for the reasoning step
            step_value: Value/content of the reasoning step
            metadata: Additional metadata
            
        Returns:
            Dict containing storage indices for each memory system
        """
        indices = {}
        
        # Store in long-term memory
        indices['long_term'] = self.long_term_memory.store_memory(step_key, step_value, metadata)
        
        # Store in working memory
        indices['working'] = self.working_memory.add_to_working_memory(step_key)
        
        return indices
        
    def retrieve_reasoning_context(
        self,
        query: torch.Tensor,
        retrieval_type: str = "all"  # "all", "long_term", "working", "episodic"
    ) -> Dict[str, torch.Tensor]:
        """
        Retrieve relevant context from memory systems.
        
        Args:
            query: Query tensor for retrieval
            retrieval_type: Type of memory to retrieve from
            
        Returns:
            Dict containing retrieved tensors from different memory systems
        """
        retrieved = {}
        
        if retrieval_type in ["all", "long_term"]:
            # Retrieve from long-term memory
            lt_keys, lt_values, lt_weights = self.long_term_memory.retrieve_memory(query)
            retrieved['long_term_keys'] = lt_keys
            retrieved['long_term_values'] = lt_values
            retrieved['long_term_weights'] = lt_weights
            
        if retrieval_type in ["all", "working"]:
            # Retrieve from working memory
            wm_content, wm_attention = self.working_memory.get_attended_working_memory(query)
            retrieved['working_memory'] = wm_content
            retrieved['working_attention'] = wm_attention
            
        if retrieval_type in ["all", "episodic"]:
            # Retrieve from episodic memory
            ep_keys, ep_values, ep_success = self.episodic_memory.retrieve_similar_episodes(query)
            retrieved['episodic_keys'] = ep_keys
            retrieved['episodic_values'] = ep_values
            retrieved['episodic_success'] = ep_success
            
        # Integrate retrieved memories
        if len(retrieved) > 1:
            integrated_memory = self._integrate_memories(retrieved)
            retrieved['integrated_memory'] = integrated_memory
            
        return retrieved
        
    def _integrate_memories(self, memories: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Integrate memories from different systems."""
        memory_tensors = []
        
        # Collect memory tensors
        if 'long_term_values' in memories:
            memory_tensors.append(memories['long_term_values'].mean(dim=1))
        if 'working_memory' in memories:
            memory_tensors.append(memories['working_memory'])
        if 'episodic_values' in memories:
            # Separate episode content and outcome
            ep_values = memories['episodic_values']
            episode_content = ep_values[:, :, :self.embedding_dim]
            memory_tensors.append(episode_content.mean(dim=1))
            
        if not memory_tensors:
            return torch.zeros(1, self.embedding_dim)
            
        # Concatenate and integrate
        concatenated = torch.cat(memory_tensors, dim=-1)
        integrated = self.memory_fusion(concatenated)
        
        return integrated
        
    def store_reasoning_episode(
        self,
        episode_key: torch.Tensor,
        episode_content: torch.Tensor,
        outcome: torch.Tensor,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store a complete reasoning episode."""
        self.episodic_memory.store_episode(
            episode_key, episode_content, outcome, metadata, success
        )
        
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics."""
        return {
            'long_term': self.long_term_memory.get_memory_summary(),
            'working_memory_size': self.working_memory.current_size,
            'episodic_episodes': self.episodic_memory.current_episode_count,
            'total_memory_utilization': (
                self.long_term_memory.get_memory_summary()['memory_utilization'] +
                (self.working_memory.current_size / self.working_memory.capacity) +
                (self.episodic_memory.current_episode_count / self.episodic_memory.max_episodes)
            ) / 3
        }
        
    def reset_all_memory(self):
        """Reset all memory systems."""
        self.long_term_memory.reset_memory()
        self.working_memory.clear_working_memory()
        # Note: Episodic memory doesn't have reset, as episodes are valuable