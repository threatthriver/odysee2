"""
Transformer Layer and Block Components

This module implements the core transformer layer and block components that
combine attention, feed-forward networks, and normalization layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

from .attention import MultiHeadAttention, FlashAttention, CrossAttention
from .feed_forward import FeedForward, GatedLinearUnit
from .layer_norm import LayerNorm, RMSLayerNorm, PreNormLayer, PostNormLayer


class TransformerLayer(nn.Module):
    """
    Basic transformer layer with attention and feed-forward networks.
    
    This layer implements the standard transformer architecture with:
    - Multi-head self-attention
    - Position-wise feed-forward network
    - Residual connections
    - Layer normalization (configurable pre/post norm)
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: Optional[int] = None,
        hidden_act: str = 'gelu',
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        layer_norm_epsilon: float = 1e-12,
        pre_norm: bool = True,
        add_cross_attention: bool = False,
        cross_attention_intermediate_size: Optional[int] = None,
        use_flash_attention: bool = False,
        use_gated_ffn: bool = False,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size or 4 * hidden_size
        self.pre_norm = pre_norm
        self.add_cross_attention = add_cross_attention
        self.use_flash_attention = use_flash_attention
        self.use_gated_ffn = use_gated_ffn
        
        # Attention mechanism
        if use_flash_attention:
            self.attention = FlashAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                attention_dropout=attention_dropout,
                **kwargs
            )
        else:
            self.attention = MultiHeadAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                attention_dropout=attention_dropout,
                **kwargs
            )
        
        # Cross-attention (for encoder-decoder models)
        if add_cross_attention:
            cross_attn_intermediate_size = cross_attention_intermediate_size or self.intermediate_size
            self.cross_attention = CrossAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                attention_dropout=attention_dropout,
                **kwargs
            )
            
            # Cross-attention layer norm
            self.cross_attention_layer_norm = LayerNorm(hidden_size, layer_norm_epsilon)
        
        # Feed-forward network
        if use_gated_ffn:
            self.feed_forward = GatedLinearUnit(
                hidden_size=hidden_size,
                intermediate_size=self.intermediate_size,
                activation=hidden_act,
                dropout=hidden_dropout,
                **kwargs
            )
        else:
            self.feed_forward = FeedForward(
                hidden_size=hidden_size,
                intermediate_size=self.intermediate_size,
                activation=hidden_act,
                dropout=hidden_dropout,
                **kwargs
            )
        
        # Layer normalization
        self.attention_layer_norm = LayerNorm(hidden_size, layer_norm_epsilon)
        self.feed_forward_layer_norm = LayerNorm(hidden_size, layer_norm_epsilon)
        
        # DropPath for stochastic depth (optional)
        self.drop_path = None
    
    def set_drop_path(self, drop_path_rate: float):
        """Set drop path rate for stochastic depth."""
        if drop_path_rate > 0:
            from .dropout import DropPath
            self.drop_path = DropPath(drop_path_rate)
    
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
        Forward pass of transformer layer.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len] or [batch_size, 1, seq_len]
            head_mask: [num_heads]
            encoder_hidden_states: [batch_size, enc_seq_len, hidden_size]
            encoder_attention_mask: [batch_size, enc_seq_len]
            past_key_value: (key, value) from previous timestep
            output_attentions: whether to return attention weights
            
        Returns:
            hidden_states: [batch_size, seq_len, hidden_size]
            attentions: attention weights if output_attentions
            cross_attentions: cross-attention weights if add_cross_attention
            past_key_value: (key, value) for next timestep
        """
        # Store outputs for concatenation
        outputs = ()
        
        # Self-attention
        attention_outputs = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            **kwargs
        )
        
        attention_output = attention_outputs[0]
        attention_weights = attention_outputs[1] if output_attentions else None
        past_key_value = attention_outputs[-1] if past_key_value else None
        
        # Apply residual connection
        if self.pre_norm:
            # Pre-norm: attention + residual
            attention_output = self.attention_layer_norm(attention_output, hidden_states)
        else:
            # Post-norm: residual + normalization
            residual = hidden_states
            hidden_states = attention_output
            hidden_states = self.attention_layer_norm(hidden_states)
            hidden_states = hidden_states + residual
        
        # Apply drop path
        if self.drop_path is not None and self.training:
            hidden_states = self.drop_path(hidden_states)
        
        # Cross-attention (for encoder-decoder models)
        if self.add_cross_attention and encoder_hidden_states is not None:
            cross_attention_outputs = self.cross_attention(
                query=hidden_states,
                key_value=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **kwargs
            )
            
            cross_attention_output = cross_attention_outputs[0]
            cross_attention_weights = cross_attention_outputs[1] if output_attentions else None
            
            # Apply residual connection and layer norm
            if self.pre_norm:
                cross_attention_output = self.cross_attention_layer_norm(
                    cross_attention_output, hidden_states
                )
            else:
                residual = hidden_states
                hidden_states = cross_attention_output
                hidden_states = self.cross_attention_layer_norm(hidden_states)
                hidden_states = hidden_states + residual
            
            # Add to outputs
            if output_attentions:
                outputs = outputs + (cross_attention_weights,)
        
        # Feed-forward network
        ffn_outputs = self.feed_forward(hidden_states)
        
        # Apply residual connection
        if self.pre_norm:
            # Pre-norm: ffn + residual
            ffn_outputs = self.feed_forward_layer_norm(ffn_outputs, hidden_states)
        else:
            # Post-norm: residual + normalization
            residual = hidden_states
            hidden_states = ffn_outputs
            hidden_states = self.feed_forward_layer_norm(hidden_states)
            hidden_states = hidden_states + residual
        
        # Apply drop path
        if self.drop_path is not None and self.training:
            hidden_states = self.drop_path(hidden_states)
        
        # Prepare outputs
        outputs = (hidden_states,) + outputs
        
        if output_attentions:
            outputs = outputs + (attention_weights,)
        
        if past_key_value is not None:
            outputs = outputs + (past_key_value,)
        
        return outputs
    
    def extra_repr(self) -> str:
        return (f'hidden_size={self.hidden_size}, num_attention_heads={self.num_attention_heads}, '
                f'intermediate_size={self.intermediate_size}, pre_norm={self.pre_norm}, '
                f'use_flash_attention={self.use_flash_attention}, use_gated_ffn={self.use_gated_ffn}')


class TransformerBlock(nn.Module):
    """
    Transformer block with additional components for advanced architectures.
    
    This block includes additional features like:
    - Gradient checkpointing support
    - Memory-efficient attention
    - Advanced normalization strategies
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: Optional[int] = None,
        hidden_act: str = 'gelu',
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        layer_norm_epsilon: float = 1e-12,
        pre_norm: bool = True,
        use_rms_norm: bool = False,
        use_memory_efficient_attention: bool = False,
        add_cross_attention: bool = False,
        use_gated_ffn: bool = False,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.pre_norm = pre_norm
        self.use_rms_norm = use_rms_norm
        self.use_memory_efficient_attention = use_memory_efficient_attention
        
        # Choose normalization layer
        norm_class = RMSLayerNorm if use_rms_norm else LayerNorm
        
        # Attention component
        if use_memory_efficient_attention:
            self.attention = FlashAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                attention_dropout=attention_dropout,
                **kwargs
            )
        else:
            self.attention = MultiHeadAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                attention_dropout=attention_dropout,
                **kwargs
            )
        
        # Feed-forward component
        if use_gated_ffn:
            self.feed_forward = GatedLinearUnit(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size or 4 * hidden_size,
                activation=hidden_act,
                dropout=hidden_dropout,
                **kwargs
            )
        else:
            self.feed_forward = FeedForward(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size or 4 * hidden_size,
                activation=hidden_act,
                dropout=hidden_dropout,
                **kwargs
            )
        
        # Normalization layers
        self.norm1 = norm_class(hidden_size, layer_norm_epsilon)
        self.norm2 = norm_class(hidden_size, layer_norm_epsilon)
        
        # Cross-attention (optional)
        if add_cross_attention:
            self.cross_attention = CrossAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                attention_dropout=attention_dropout,
                **kwargs
            )
            self.norm_cross = norm_class(hidden_size, layer_norm_epsilon)
    
    def _gradient_checkpointing_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass with gradient checkpointing."""
        # This function will be called during gradient checkpointing
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        # Use PyTorch's checkpointing utilities
        if self.pre_norm:
            # Pre-norm: norm -> attention -> residual
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.norm1), hidden_states
            )
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.attention), hidden_states, attention_mask, **kwargs
            )
            
            # Residual connection
            hidden_states = hidden_states + kwargs.get('hidden_states', hidden_states)
            
            # Cross-attention (if present)
            if hasattr(self, 'cross_attention') and 'encoder_hidden_states' in kwargs:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.norm_cross), hidden_states
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.cross_attention), 
                    hidden_states, kwargs['encoder_hidden_states'], 
                    kwargs.get('encoder_attention_mask', None), **kwargs
                )
                hidden_states = hidden_states + hidden_states  # Simple residual
            
            # Feed-forward
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.norm2), hidden_states
            )
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.feed_forward), hidden_states, **kwargs
            )
            
            # Final residual
            hidden_states = hidden_states + kwargs.get('hidden_states', hidden_states)
        else:
            # Post-norm: attention -> norm -> residual
            # (simplified for post-norm)
            pass
        
        return hidden_states
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass of transformer block with gradient checkpointing support.
        """
        # Determine if gradient checkpointing should be used
        use_checkpointing = (
            self.training and 
            torch.utils.checkpoint.checkpoint_activations and
            hidden_states.requires_grad
        )
        
        if use_checkpointing:
            return self._gradient_checkpointing_forward(
                hidden_states, attention_mask, **kwargs
            )
        else:
            # Use standard transformer layer implementation
            layer = TransformerLayer(
                hidden_size=self.hidden_size,
                num_attention_heads=self.num_attention_heads,
                pre_norm=self.pre_norm,
                use_flash_attention=self.use_memory_efficient_attention,
                **kwargs
            )
            outputs = layer(hidden_states, attention_mask, **kwargs)
            return outputs[0]  # Return hidden states
    
    def extra_repr(self) -> str:
        return (f'hidden_size={self.hidden_size}, num_attention_heads={self.num_attention_heads}, '
                f'pre_norm={self.pre_norm}, use_rms_norm={self.use_rms_norm}, '
                f'use_memory_efficient_attention={self.use_memory_efficient_attention}')


class TransformerEncoderLayer(TransformerLayer):
    """
    Encoder layer for transformer models.
    
    This is essentially the same as TransformerLayer but with encoder-specific
    configurations and optimizations.
    """
    
    def __init__(self, *args, **kwargs):
        # Ensure no cross-attention for encoder-only models
        kwargs.setdefault('add_cross_attention', False)
        super().__init__(*args, **kwargs)


class TransformerDecoderLayer(TransformerLayer):
    """
    Decoder layer for transformer models.
    
    This layer includes cross-attention capabilities for encoder-decoder models
    and is optimized for autoregressive generation.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: Optional[int] = None,
        hidden_act: str = 'gelu',
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        layer_norm_epsilon: float = 1e-12,
        pre_norm: bool = True,
        add_cross_attention: bool = True,
        **kwargs
    ):
        # Always enable cross-attention for decoder
        kwargs.update({
            'add_cross_attention': add_cross_attention,
            'use_causal_mask': True  # Decoder always uses causal attention
        })
        
        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            layer_norm_epsilon=layer_norm_epsilon,
            pre_norm=pre_norm,
            **kwargs
        )