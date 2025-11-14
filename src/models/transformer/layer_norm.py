"""
Layer Normalization and RMS Layer Normalization

This module implements various normalization techniques commonly used in transformer
architectures, with optimizations for large-scale language model training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LayerNorm(nn.Module):
    """
    Layer Normalization with optional elementwise affine transformation.
    
    This implementation follows the standard transformer layer norm
    with optional bias and scale parameters.
    """
    
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-12,
        elementwise_affine: bool = True,
        bias: bool = True,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
            if bias:
                self.bias = nn.Parameter(torch.zeros(hidden_size))
            else:
                self.register_parameter('bias', None)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Apply layer normalization.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            residual: Optional residual connection [batch_size, seq_len, hidden_size]
            
        Returns:
            normalized_states: [batch_size, seq_len, hidden_size]
        """
        # Compute mean and variance along the last dimension
        mean = hidden_states.mean(dim=-1, keepdim=True)
        variance = hidden_states.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize
        normalized = (hidden_states - mean) / torch.sqrt(variance + self.eps)
        
        # Apply affine transformation if enabled
        if self.elementwise_affine and self.weight is not None:
            normalized = normalized * self.weight
            if self.bias is not None:
                normalized = normalized + self.bias
        
        # Apply residual connection if provided
        if residual is not None:
            normalized = normalized + residual
        
        return normalized
    
    def extra_repr(self) -> str:
        return (f'hidden_size={self.hidden_size}, eps={self.eps}, '
                f'elementwise_affine={self.elementwise_affine}')


class RMSLayerNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    RMSLayerNorm is a simpler variant of LayerNorm that only scales the inputs
    without centering them. It's computationally more efficient and has shown
    to perform as well as or better than LayerNorm in some cases.
    """
    
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter('weight', None)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Apply RMS layer normalization.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            residual: Optional residual connection [batch_size, seq_len, hidden_size]
            
        Returns:
            normalized_states: [batch_size, seq_len, hidden_size]
        """
        # Compute RMS along the last dimension
        rms = torch.rsqrt(hidden_states.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        
        # Normalize by RMS
        normalized = hidden_states * rms
        
        # Apply affine transformation if enabled
        if self.elementwise_affine and self.weight is not None:
            normalized = normalized * self.weight
        
        # Apply residual connection if provided
        if residual is not None:
            normalized = normalized + residual
        
        return normalized
    
    def extra_repr(self) -> str:
        return (f'hidden_size={self.hidden_size}, eps={self.eps}, '
                f'elementwise_affine={self.elementwise_affine}')


class PreNormLayer(nn.Module):
    """
    Pre-normalization wrapper for transformer layers.
    
    This module applies layer normalization before sublayer computation,
    which is the approach used in many modern transformer architectures.
    """
    
    def __init__(
        self,
        sublayer: nn.Module,
        norm_type: str = 'layer_norm',
        hidden_size: int = 768,
        eps: float = 1e-12,
        **kwargs
    ):
        super().__init__()
        self.sublayer = sublayer
        self.norm_type = norm_type
        
        if norm_type == 'layer_norm':
            self.norm = LayerNorm(hidden_size, eps)
        elif norm_type == 'rms_layer_norm':
            self.norm = RMSLayerNorm(hidden_size, eps)
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Apply pre-normalization and sublayer computation.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        # Apply normalization before sublayer
        normalized = self.norm(hidden_states)
        
        # Apply sublayer (attention or feed-forward)
        output = self.sublayer(normalized, **kwargs)
        
        # Return output directly - residual connection is handled in transformer layer
        return output


class PostNormLayer(nn.Module):
    """
    Post-normalization wrapper for transformer layers.
    
    This module applies layer normalization after sublayer computation,
    which was the original approach in the Transformer paper.
    """
    
    def __init__(
        self,
        sublayer: nn.Module,
        norm_type: str = 'layer_norm',
        hidden_size: int = 768,
        eps: float = 1e-12,
        **kwargs
    ):
        super().__init__()
        self.sublayer = sublayer
        self.norm_type = norm_type
        
        if norm_type == 'layer_norm':
            self.norm = LayerNorm(hidden_size, eps)
        elif norm_type == 'rms_layer_norm':
            self.norm = RMSLayerNorm(hidden_size, eps)
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Apply sublayer computation and post-normalization.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        # Apply sublayer (attention or feed-forward) with residual
        output = self.sublayer(hidden_states, **kwargs)
        
        # Apply normalization after sublayer
        normalized = self.norm(output)
        
        return normalized