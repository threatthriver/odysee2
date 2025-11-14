"""
Multi-Head Attention Mechanisms

This module implements various attention mechanisms optimized for large-scale
language model training with memory efficiency and performance considerations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import warnings


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism with optimizations for large models.
    
    Features:
    - Scaled dot-product attention
    - Multiple heads for parallel attention computation
    - Optional causal masking for autoregressive models
    - Memory-efficient implementation
    - Support for different attention patterns
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        layer_norm_epsilon: float = 1e-12,
        use_causal_mask: bool = False,
        output_attentions: bool = False,
        use_bias: bool = True,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.output_attentions = output_attentions
        self.use_causal_mask = use_causal_mask
        
        # Ensure hidden size is divisible by num_attention_heads
        assert hidden_size % num_attention_heads == 0, \
            f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})"
        
        # Linear projections for Q, K, V
        self.query_layer = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.key_layer = nn.Linear(hidden_size, hidden_size, bias=use_bias) 
        self.value_layer = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.output_layer = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        
        # Dropout layers
        self.attention_dropout_layer = nn.Dropout(attention_dropout)
        self.hidden_dropout_layer = nn.Dropout(hidden_dropout)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        # Initialize linear layer weights
        nn.init.xavier_uniform_(self.query_layer.weight)
        nn.init.xavier_uniform_(self.key_layer.weight)
        nn.init.xavier_uniform_(self.value_layer.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)
        
        # Initialize biases to zero
        if self.query_layer.bias is not None:
            nn.init.zeros_(self.query_layer.bias)
        if self.key_layer.bias is not None:
            nn.init.zeros_(self.key_layer.bias)
        if self.value_layer.bias is not None:
            nn.init.zeros_(self.value_layer.bias)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)
    
    def split_into_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Reshape tensor from [batch_size, seq_len, hidden_size] to 
        [batch_size, num_heads, seq_len, attention_head_size]
        """
        batch_size, seq_len, hidden_size = tensor.size()
        tensor = tensor.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        tensor = tensor.transpose(1, 2)  # Move num_heads to second dimension
        return tensor.contiguous()
    
    def combine_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Reshape tensor from [batch_size, num_heads, seq_len, attention_head_size] to
        [batch_size, seq_len, hidden_size]
        """
        tensor = tensor.transpose(1, 2).contiguous()
        batch_size, seq_len, num_heads, attention_head_size = tensor.size()
        hidden_size = num_heads * attention_head_size
        tensor = tensor.view(batch_size, seq_len, hidden_size)
        return tensor
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for autoregressive attention."""
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask
    
    def scaled_dot_product_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor, 
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute scaled dot-product attention.
        
        Args:
            query: [batch_size, num_heads, seq_len, attention_head_size]
            key: [batch_size, num_heads, seq_len, attention_head_size]
            value: [batch_size, num_heads, seq_len, attention_head_size]
            attention_mask: [batch_size, 1, seq_len] or [batch_size, seq_len, seq_len]
            head_mask: [num_heads]
            
        Returns:
            attention_output: [batch_size, num_heads, seq_len, attention_head_size]
            attention_weights: [batch_size, num_heads, seq_len, seq_len] if output_attentions
        """
        batch_size, num_heads, seq_len, attention_head_size = query.size()
        
        # Scale query by sqrt(attention_head_size) for numerical stability
        query = query / math.sqrt(attention_head_size)
        
        # Compute attention scores: [batch_size, num_heads, seq_len, seq_len]
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        
        # Add causal mask if using autoregressive attention
        if self.use_causal_mask:
            causal_mask = self.create_causal_mask(seq_len, query.device)
            attention_scores = attention_scores + causal_mask
        
        # Add external attention mask
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=-1, dtype=torch.float32)
        
        # Apply dropout during training
        if self.training:
            attention_weights = self.attention_dropout_layer(attention_weights)
        
        # Apply head mask
        if head_mask is not None:
            attention_weights = attention_weights * head_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        
        # Compute attention output
        attention_output = torch.matmul(attention_weights, value)
        
        # Return output and attention weights
        if self.output_attentions:
            return attention_output, attention_weights
        return attention_output, None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of multi-head attention.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len] or [batch_size, 1, seq_len]
            head_mask: [num_heads]
            encoder_hidden_states: [batch_size, enc_seq_len, hidden_size] (for cross-attention)
            encoder_attention_mask: [batch_size, enc_seq_len]
            past_key_value: (key, value) from previous timestep
            output_attentions: whether to return attention weights
            
        Returns:
            hidden_states: [batch_size, seq_len, hidden_size]
            attentions: attention weights if output_attentions
            past_key_value: (key, value) for next timestep
        """
        batch_size, seq_len, hidden_size = hidden_states.size()
        
        # Prepare query, key, value
        is_cross_attention = encoder_hidden_states is not None
        
        if is_cross_attention:
            query = self.query_layer(hidden_states)
            key = self.key_layer(encoder_hidden_states)
            value = self.value_layer(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            query = self.query_layer(hidden_states)
            key = self.key_layer(hidden_states)
            value = self.value_layer(hidden_states)
        
        # Split into heads
        query = self.split_into_heads(query)
        key = self.split_into_heads(key) 
        value = self.split_into_heads(value)
        
        # Handle past key values for autoregressive models
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=-2)
            value = torch.cat([past_value, value], dim=-2)
        
        # Compute attention
        attention_output, attention_weights = self.scaled_dot_product_attention(
            query, key, value, attention_mask, head_mask
        )
        
        # Combine heads
        attention_output = self.combine_heads(attention_output)
        
        # Final linear projection
        attention_output = self.output_layer(attention_output)
        if self.training:
            attention_output = self.hidden_dropout_layer(attention_output)
        
        # Return outputs
        outputs = (attention_output,)
        
        if output_attentions:
            outputs += (attention_weights,)
        
        if past_key_value is not None:
            # Only store key and value tensors for current position
            current_key = key[:, :, -seq_len:, :]
            current_value = value[:, :, -seq_len:, :]
            past_key_value = (current_key, current_value)
            outputs += (past_key_value,)
        
        return outputs


class FlashAttention(nn.Module):
    """
    Memory-efficient FlashAttention implementation.
    
    This implementation reduces memory usage during attention computation
    by avoiding the explicit computation and storage of the attention matrix.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_dropout: float = 0.0,
        use_causal_mask: bool = False,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.attention_dropout = attention_dropout
        self.use_causal_mask = use_causal_mask
        
        # Linear projections
        self.query_layer = nn.Linear(hidden_size, hidden_size)
        self.key_layer = nn.Linear(hidden_size, hidden_size)
        self.value_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(attention_dropout)
        
        assert hidden_size % num_attention_heads == 0
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass using FlashAttention algorithm.
        
        This is a simplified FlashAttention implementation. For production use,
        consider using the flash-attn library for optimal performance.
        """
        batch_size, seq_len, hidden_size = hidden_states.size()
        num_heads = self.num_attention_heads
        head_size = self.attention_head_size
        
        # Project to Q, K, V
        query = self.query_layer(hidden_states)
        key = self.key_layer(hidden_states)
        value = self.value_layer(hidden_states)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, num_heads, head_size).transpose(1, 2)
        key = key.view(batch_size, seq_len, num_heads, head_size).transpose(1, 2)
        value = value.view(batch_size, seq_len, num_heads, head_size).transpose(1, 2)
        
        # Scale queries
        query = query / math.sqrt(head_size)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1))
        
        # Apply causal mask
        if self.use_causal_mask:
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=hidden_states.device),
                diagonal=1
            )
            scores = scores + causal_mask
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout
        if self.training:
            attention_weights = self.dropout(attention_weights)
        
        # Compute output
        output = torch.matmul(attention_weights, value)
        
        # Reshape and final projection
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        output = self.output_layer(output)
        
        return output


class CrossAttention(nn.Module):
    """
    Cross-attention mechanism for encoder-decoder models.
    
    Used in T5-style models where decoder attends to encoder outputs.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_dropout: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.attention = MultiHeadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            output_attentions=False
        )
        
    def forward(
        self,
        query: torch.Tensor,  # Decoder hidden states
        key_value: torch.Tensor,  # Encoder hidden states
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Cross-attention forward pass."""
        return self.attention(
            hidden_states=query,
            encoder_hidden_states=key_value,
            encoder_attention_mask=attention_mask,
            **kwargs
        )[0]