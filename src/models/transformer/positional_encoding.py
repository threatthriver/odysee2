"""
Positional Encoding Modules

This module implements various positional encoding strategies for transformer models,
including traditional sinusoidal encoding and modern alternatives like RoPE and ALiBi.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
import numpy as np


class PositionalEncoding(nn.Module):
    """
    Traditional sinusoidal positional encoding for transformers.
    
    This implementation follows the original Transformer paper's approach
    using sine and cosine functions at different frequencies.
    """
    
    def __init__(
        self,
        hidden_size: int,
        max_position_embeddings: int = 512,
        dropout: float = 0.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        
        # Create positional encoding matrix
        pe = torch.zeros(max_position_embeddings, hidden_size)
        position = torch.arange(0, max_position_embeddings).unsqueeze(1).float()
        
        # Create the angle for each dimension
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2).float() * 
            -(math.log(10000.0) / hidden_size)
        )
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension
        pe = pe.unsqueeze(0)  # Shape: [1, max_pos, hidden_size]
        
        # Register as buffer (not trainable but saved in state dict)
        self.register_buffer('pe', pe)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            input_ids: [batch_size, seq_len, hidden_size] - input embeddings
            position_ids: [batch_size, seq_len] - optional position indices
            
        Returns:
            embeddings_with_pe: [batch_size, seq_len, hidden_size]
        """
        seq_len = input_ids.size(1)
        
        if position_ids is None:
            # Use default position encoding
            if seq_len <= self.max_position_embeddings:
                position_embeddings = self.pe[:, :seq_len]
            else:
                # Extend positional encoding if sequence is longer than max_position_embeddings
                position_embeddings = self._extend_position_embeddings(seq_len)
        else:
            position_embeddings = self._create_custom_position_embeddings(position_ids)
        
        # Add positional encoding to input
        embeddings_with_pe = input_ids + position_embeddings
        embeddings_with_pe = self.dropout(embeddings_with_pe)
        
        return embeddings_with_pe
    
    def _extend_position_embeddings(self, seq_len: int) -> torch.Tensor:
        """Extend positional embeddings for longer sequences."""
        # This is a simplified extension - for production use, consider
        # learning-based approaches or other methods for longer sequences
        extended_pe = torch.zeros(1, seq_len, self.hidden_size, device=self.pe.device)
        
        # Repeat existing embeddings and extrapolate for longer sequences
        repeated_pe = self.pe.repeat_interleave(
            math.ceil(seq_len / self.max_position_embeddings), dim=1
        )
        
        return repeated_pe[:, :seq_len]
    
    def _create_custom_position_embeddings(self, position_ids: torch.Tensor) -> torch.Tensor:
        """Create positional embeddings for custom position indices."""
        batch_size, seq_len = position_ids.size()
        
        # Gather embeddings at specified positions
        position_embeddings = torch.gather(
            self.pe.expand(batch_size, -1, -1),
            dim=1,
            index=position_ids.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        )
        
        return position_embeddings


class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Positional Encoding (RoPE) implementation.
    
    RoPE rotary multiplies the query and key vectors by rotation matrices
    that depend on the relative positions, providing better extrapolation
    capabilities and improved performance on long sequences.
    """
    
    def __init__(
        self,
        hidden_size: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
        dropout: float = 0.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Compute inverse frequency
        inv_freq = 1.0 / (base ** (torch.arange(0, hidden_size, 2).float() / hidden_size))
        self.register_buffer('inv_freq', inv_freq)
        
        self.dropout = nn.Dropout(dropout)
        
        # Precompute rotation matrices for all positions
        self._compute_rotation_matrices()
    
    def _compute_rotation_matrices(self):
        """Precompute rotation matrices for positions 0 to max_position_embeddings."""
        seq_len = self.max_position_embeddings
        
        # Create position array
        t = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        
        # Compute frequencies for all positions
        freqs = torch.outer(t.squeeze(), self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)  # Repeat for even/odd indices
        
        # Compute rotation angles
        self.register_buffer('cos_cached', freqs.cos())
        self.register_buffer('sin_cached', freqs.sin())
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dimensions of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary positional embedding to query and key tensors."""
        # Extract rotation matrices for current sequence length
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        
        # Apply rotation to queries and keys
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_embed, k_embed
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply rotary positional encoding to hidden states.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            position_ids: [batch_size, seq_len]
            
        Returns:
            encoded_states: [batch_size, seq_len, hidden_size]
        """
        # For RoPE, we typically apply the rotation in the attention mechanism
        # This method is kept for compatibility but the actual application
        # happens in MultiHeadAttention when using RoPE
        return hidden_states


class ALiBiPositionalEncoding(nn.Module):
    """
    Attention with Linear Biases (ALiBi) positional encoding.
    
    ALiBi applies a linear bias to attention scores based on relative positions,
    eliminating the need for explicit positional embeddings and providing
    better extrapolation to longer sequences.
    """
    
    def __init__(
        self,
        num_attention_heads: int,
        max_position_embeddings: int = 2048,
        slope_range: Tuple[float, float] = (1e-2, 1e-1),
        **kwargs
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.slope_range = slope_range
        
        # Compute slopes for each attention head
        slopes = torch.linspace(slope_range[0], slope_range[1], num_attention_heads)
        
        # Create position differences matrix
        position_diff = torch.arange(max_position_embeddings).unsqueeze(0) - \
                       torch.arange(max_position_embeddings).unsqueeze(1)
        
        # Create bias matrix: -slope * position_diff
        bias_matrix = -slopes.view(1, num_attention_heads, 1, 1) * position_diff.abs().unsqueeze(0).unsqueeze(0)
        
        # Apply upper triangular mask (only future positions get bias)
        mask = torch.triu(torch.ones(max_position_embeddings, max_position_embeddings), diagonal=1)
        bias_matrix = bias_matrix.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 1, float('-inf'))
        
        self.register_buffer('bias_matrix', bias_matrix)
    
    def get_bias_matrix(self, seq_len: int) -> torch.Tensor:
        """Get bias matrix for current sequence length."""
        if seq_len > self.max_position_embeddings:
            # Extend bias matrix for longer sequences
            extended_bias = self._extend_bias_matrix(seq_len)
            return extended_bias
        
        return self.bias_matrix[:, :, :seq_len, :seq_len]
    
    def _extend_bias_matrix(self, seq_len: int) -> torch.Tensor:
        """Extend bias matrix for longer sequences."""
        # For production use, implement proper extension logic
        # This is a simplified version
        current_len = self.max_position_embeddings
        
        # Pad with negative infinity for positions beyond current max
        extended_bias = torch.full(
            (1, self.num_attention_heads, seq_len, seq_len),
            float('-inf'),
            device=self.bias_matrix.device
        )
        
        # Copy existing bias matrix
        extended_bias[:, :, :current_len, :current_len] = self.bias_matrix
        
        return extended_bias


class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding for transformers.
    
    This implementation allows the model to attend to relative positions
    rather than absolute positions, often leading to better generalization.
    """
    
    def __init__(
        self,
        hidden_size: int,
        max_relative_position: int = 128,
        dropout: float = 0.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_relative_position = max_relative_position
        
        # Create relative position embeddings
        vocab_size = max_relative_position * 2 + 1
        self.relative_embeddings = nn.Embedding(vocab_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def get_relative_position_table(
        self,
        seq_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """Get relative position table for given sequence length."""
        # Create relative position indices
        range_vec = torch.arange(seq_len)
        relative_indices = range_vec.unsqueeze(0) - range_vec.unsqueeze(1)
        # Clip to valid range
        relative_indices = torch.clamp(
            relative_indices, 
            -self.max_relative_position, 
            self.max_relative_position
        )
        # Shift to positive indices
        relative_indices += self.max_relative_position
        
        return self.relative_embeddings(relative_indices)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Apply relative positional encoding."""
        # In practice, relative positional encoding is applied within the attention mechanism
        # This method maintains compatibility with the interface
        return hidden_states


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional embeddings for transformers.
    
    Instead of using fixed positional encodings, this approach learns
    positional embeddings during training, allowing the model to discover
    its own positional patterns.
    """
    
    def __init__(
        self,
        max_position_embeddings: int,
        hidden_size: int,
        dropout: float = 0.0
    ):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize position embeddings
        self._init_position_embeddings()
    
    def _init_position_embeddings(self):
        """Initialize position embeddings with small random values."""
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.1)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Add learned positional embeddings to input embeddings.
        
        Args:
            input_ids: [batch_size, seq_len, hidden_size] - input embeddings
            position_ids: [batch_size, seq_len] - optional position indices
            
        Returns:
            embeddings_with_pos: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len = input_ids.size(0), input_ids.size(1)
        
        if position_ids is None:
            # Create default position indices
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            position_ids = position_ids.expand(batch_size, -1)
        
        # Get positional embeddings
        position_embeddings = self.position_embeddings(position_ids)
        
        # Add to input embeddings
        embeddings_with_pos = input_ids + position_embeddings
        embeddings_with_pos = self.dropout(embeddings_with_pos)
        
        return embeddings_with_pos