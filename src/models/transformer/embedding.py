"""
Embedding Layers for Transformer Models

This module implements various embedding layers used in transformer architectures,
including token embeddings, position embeddings, segment embeddings, and combinations.
"""

import torch
import torch.nn as nn
from typing import Optional


class TokenEmbedding(nn.Module):
    """
    Token embedding layer for transformer models.
    
    Converts input token IDs to dense embeddings.
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        padding_idx: Optional[int] = None,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        
        # Token embedding
        self.token_embeddings = nn.Embedding(
            vocab_size, 
            hidden_size, 
            padding_idx=padding_idx
        )
        
        self.dropout = nn.Dropout(dropout)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize token embeddings."""
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=0.1)
        
        # Initialize padding tokens to zero
        if self.padding_idx is not None:
            with torch.no_grad():
                self.token_embeddings.weight[self.padding_idx].zero_()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Convert token IDs to embeddings.
        
        Args:
            input_ids: [batch_size, seq_len] - token IDs
            
        Returns:
            embeddings: [batch_size, seq_len, hidden_size]
        """
        token_embeddings = self.token_embeddings(input_ids)
        embeddings = self.dropout(token_embeddings)
        return embeddings
    
    def extra_repr(self) -> str:
        return (f'vocab_size={self.vocab_size}, hidden_size={self.hidden_size}, '
                f'padding_idx={self.padding_idx}')


class PositionEmbedding(nn.Module):
    """
    Position embedding layer for transformers.
    
    Learns position embeddings during training.
    """
    
    def __init__(
        self,
        max_position_embeddings: int,
        hidden_size: int,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        
        # Position embedding
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize position embeddings."""
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.1)
    
    def forward(
        self,
        position_ids: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Convert position IDs to embeddings.
        
        Args:
            position_ids: [batch_size, seq_len] - position indices
            
        Returns:
            embeddings: [batch_size, seq_len, hidden_size]
        """
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = self.dropout(position_embeddings)
        return embeddings
    
    def extra_repr(self) -> str:
        return (f'max_position_embeddings={self.max_position_embeddings}, '
                f'hidden_size={self.hidden_size}')


class SegmentEmbedding(nn.Module):
    """
    Segment embedding layer for models that use segment information.
    
    Used in BERT-style models to distinguish between different sentences/segments.
    """
    
    def __init__(
        self,
        num_segments: int,
        hidden_size: int,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.num_segments = num_segments
        self.hidden_size = hidden_size
        
        # Segment embedding
        self.segment_embeddings = nn.Embedding(num_segments, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize segment embeddings."""
        nn.init.normal_(self.segment_embeddings.weight, mean=0.0, std=0.1)
    
    def forward(
        self,
        segment_ids: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Convert segment IDs to embeddings.
        
        Args:
            segment_ids: [batch_size, seq_len] - segment IDs
            
        Returns:
            embeddings: [batch_size, seq_len, hidden_size]
        """
        segment_embeddings = self.segment_embeddings(segment_ids)
        embeddings = self.dropout(segment_embeddings)
        return embeddings
    
    def extra_repr(self) -> str:
        return (f'num_segments={self.num_segments}, hidden_size={self.hidden_size}')


class CombinedEmbedding(nn.Module):
    """
    Combined embedding layer that concatenates multiple embedding types.
    
    Combines token, position, and segment embeddings for models like BERT.
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_embeddings: Optional[int] = None,
        num_segments: Optional[int] = None,
        padding_idx: Optional[int] = None,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        
        # Token embedding (always required)
        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            padding_idx=padding_idx,
            dropout=0.0  # Dropout applied after combination
        )
        
        # Position embedding (optional)
        self.position_embedding = None
        if max_position_embeddings is not None:
            self.position_embedding = PositionEmbedding(
                max_position_embeddings=max_position_embeddings,
                hidden_size=hidden_size,
                dropout=0.0  # Dropout applied after combination
            )
        
        # Segment embedding (optional)
        self.segment_embedding = None
        if num_segments is not None:
            self.segment_embedding = SegmentEmbedding(
                num_segments=num_segments,
                hidden_size=hidden_size,
                dropout=0.0  # Dropout applied after combination
            )
        
        # Final projection to ensure consistent hidden size
        if self.position_embedding or self.segment_embedding:
            # Project combined embeddings to hidden_size
            combined_size = self._get_combined_size()
            if combined_size != hidden_size:
                self.embedding_proj = nn.Linear(combined_size, hidden_size)
            else:
                self.embedding_proj = nn.Identity()
        else:
            self.embedding_proj = nn.Identity()
    
    def _get_combined_size(self) -> int:
        """Get the size of combined embeddings."""
        size = self.hidden_size  # token embeddings
        
        if self.position_embedding:
            size += self.hidden_size
        
        if self.segment_embedding:
            size += self.hidden_size
        
        return size
    
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Combine multiple embedding types.
        
        Args:
            input_ids: [batch_size, seq_len] - token IDs
            position_ids: [batch_size, seq_len] - optional position indices
            segment_ids: [batch_size, seq_len] - optional segment IDs
            
        Returns:
            embeddings: [batch_size, seq_len, hidden_size]
        """
        # Get token embeddings
        embeddings = self.token_embedding(input_ids)
        
        # Combine with position embeddings
        if self.position_embedding:
            if position_ids is None:
                seq_len = input_ids.size(1)
                position_ids = torch.arange(
                    seq_len, device=input_ids.device
                ).unsqueeze(0).expand_as(input_ids)
            
            position_embeddings = self.position_embedding(position_ids)
            embeddings = embeddings + position_embeddings
        
        # Combine with segment embeddings
        if self.segment_embedding:
            if segment_ids is None:
                segment_ids = torch.zeros_like(input_ids)
            
            segment_embeddings = self.segment_embedding(segment_ids)
            embeddings = embeddings + segment_embeddings
        
        # Project to final hidden size if needed
        embeddings = self.embedding_proj(embeddings)
        
        # Apply dropout
        embeddings = self.dropout(embeddings)
        
        return embeddings


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) layer.
    
    Applies rotary positional embedding to token embeddings using
    the approach described in "RoFormer: Enhanced Transformer with Rotary Position Embedding".
    """
    
    def __init__(
        self,
        hidden_size: int,
        base: int = 10000,
        max_position_embeddings: int = 2048,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        
        # Create rotation matrices
        inv_freq = 1.0 / (base ** (torch.arange(0, hidden_size, 2).float() / hidden_size))
        self.register_buffer('inv_freq', inv_freq)
    
    def _get_cos_sin_cache(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get cosine and sine cache for given sequence length."""
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_cached = emb.cos()[None, None, :, :]
        sin_cached = emb.sin()[None, None, :, :]
        
        return cos_cached, sin_cached
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dimensions."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(
        self,
        token_embeddings: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply rotary positional embedding to token embeddings.
        
        Args:
            token_embeddings: [batch_size, seq_len, hidden_size] - token embeddings
            position_ids: [batch_size, seq_len] - optional position indices
            
        Returns:
            rotated_embeddings: [batch_size, seq_len, hidden_size]
        """
        seq_len = token_embeddings.size(1)
        
        # Get cos and sin cache
        cos_cached, sin_cached = self._get_cos_sin_cache(seq_len, token_embeddings.device)
        
        # Apply rotary embedding
        cos = cos_cached[:, :, :seq_len, :]
        sin = sin_cached[:, :, :seq_len, :]
        
        rotated_embeddings = (
            token_embeddings * cos + self._rotate_half(token_embeddings) * sin
        )
        
        return rotated_embeddings
    
    def extra_repr(self) -> str:
        return (f'hidden_size={self.hidden_size}, base={self.base}, '
                f'max_position_embeddings={self.max_position_embeddings}')


class ALiBiEmbedding(nn.Module):
    """
    Attention with Linear Biases (ALiBi) embedding layer.
    
    Applies linear biases to attention scores based on relative positions,
    eliminating the need for explicit positional embeddings.
    """
    
    def __init__(
        self,
        num_attention_heads: int,
        max_position_embeddings: int = 2048,
        slope_range: tuple = (1e-2, 1e-1),
        **kwargs
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.slope_range = slope_range
        
        # Create bias matrix (will be applied in attention mechanism)
        slopes = torch.linspace(slope_range[0], slope_range[1], num_attention_heads)
        
        # Create position differences
        position_diff = torch.arange(max_position_embeddings).unsqueeze(0) - \
                       torch.arange(max_position_embeddings).unsqueeze(1)
        
        # Create bias matrix
        bias_matrix = -slopes.view(1, num_attention_heads, 1, 1) * position_diff.abs().unsqueeze(0).unsqueeze(0)
        
        # Apply upper triangular mask
        mask = torch.triu(torch.ones(max_position_embeddings, max_position_embeddings), diagonal=1)
        bias_matrix = bias_matrix.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 1, float('-inf'))
        
        self.register_buffer('bias_matrix', bias_matrix)
    
    def forward(
        self,
        token_embeddings: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Return token embeddings unchanged - ALiBi bias is applied in attention.
        
        Args:
            token_embeddings: [batch_size, seq_len, hidden_size] - token embeddings
            
        Returns:
            embeddings: [batch_size, seq_len, hidden_size] (unchanged)
        """
        return token_embeddings
    
    def get_bias_matrix(self, seq_len: int) -> torch.Tensor:
        """Get bias matrix for current sequence length."""
        if seq_len > self.max_position_embeddings:
            # Extend bias matrix (simplified version)
            return self.bias_matrix[:, :, :seq_len, :seq_len]
        return self.bias_matrix[:, :, :seq_len, :seq_len]