"""
Dropout and Stochastic Depth Utilities

This module implements dropout variations and stochastic depth mechanisms
commonly used in transformer architectures for regularization and training
stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DropPath(nn.Module):
    """
    Stochastic Depth (DropPath) for transformer models.
    
    DropPath randomly drops entire paths (layers) during training,
    which can improve model performance and training stability.
    """
    
    def __init__(
        self,
        drop_prob: float = 0.0,
        scale_by_keep: bool = True
    ):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
        
        if drop_prob < 0 or drop_prob > 1:
            raise ValueError(f"drop_prob should be in [0, 1], but got {drop_prob}")
    
    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply stochastic depth to hidden states.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size] - input tensor
            
        Returns:
            output: [batch_size, seq_len, hidden_size] - with stochastic depth applied
        """
        if not self.training or self.drop_prob == 0:
            return hidden_states
        
        # Create binary mask
        keep_prob = 1 - self.drop_prob
        shape = (hidden_states.shape[0], 1, 1)  # Match batch_size, seq_len=1, hidden_size=1
        if hidden_states.device.type == "cpu":
            # CPU implementation
            mask = torch.bernoulli(torch.empty(shape).fill_(keep_prob))
        else:
            # GPU/CUDA implementation
            mask = torch.bernoulli(torch.empty(shape, device=hidden_states.device).fill_(keep_prob))
        
        # Scale the output to maintain expected value
        if self.scale_by_keep:
            output = hidden_states / keep_prob * mask
        else:
            output = hidden_states * mask
        
        return output
    
    def extra_repr(self) -> str:
        return f'drop_prob={self.drop_prob}, scale_by_keep={self.scale_by_keep}'


class Dropout(nn.Module):
    """
    Standard dropout with various scheduling strategies.
    
    Supports different dropout rates for different parts of the model
    and can handle various dropout scheduling strategies.
    """
    
    def __init__(
        self,
        p: float = 0.5,
        inplace: bool = False
    ):
        super().__init__()
        self.p = p
        self.inplace = inplace
    
    def forward(
        self,
        input: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply dropout to input tensor.
        
        Args:
            input: input tensor
            
        Returns:
            output: tensor with dropout applied (during training) or unchanged (during inference)
        """
        return F.dropout(input, self.p, self.training, self.inplace)


class EmbeddingDropout(nn.Module):
    """
    Dropout applied to embeddings before they are passed through the model.
    
    This is different from standard dropout as it's applied to the embedding
    vectors rather than individual positions.
    """
    
    def __init__(
        self,
        p: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.p = p
    
    def forward(
        self,
        embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply dropout to embeddings.
        
        Args:
            embedding: [batch_size, seq_len, hidden_size] - embedding tensor
            
        Returns:
            output: [batch_size, seq_len, hidden_size] - with embedding dropout applied
        """
        if not self.training or self.p == 0:
            return embedding
        
        # Create dropout mask for the entire embedding vectors
        batch_size, seq_len, hidden_size = embedding.size()
        
        # Sample binary mask for each embedding vector
        if embedding.device.type == "cpu":
            mask = torch.bernoulli(torch.empty(batch_size, 1, hidden_size).fill_(1 - self.p))
        else:
            mask = torch.bernoulli(torch.empty(batch_size, 1, hidden_size, device=embedding.device).fill_(1 - self.p))
        
        # Apply mask - drop entire embedding vectors
        output = embedding * mask / (1 - self.p)  # Scale to maintain expected value
        
        return output
    
    def extra_repr(self) -> str:
        return f'p={self.p}'


class StructuredDropout(nn.Module):
    """
    Structured dropout that drops entire channels or features.
    
    More effective than standard dropout for high-dimensional inputs
    and can provide better regularization.
    """
    
    def __init__(
        self,
        p: float = 0.1,
        structured_type: str = 'feature',  # 'feature', 'spatial', 'channel'
        **kwargs
    ):
        super().__init__()
        self.p = p
        self.structured_type = structured_type
        
        if structured_type not in ['feature', 'spatial', 'channel']:
            raise ValueError(f"structured_type should be one of ['feature', 'spatial', 'channel'], got {structured_type}")
    
    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply structured dropout.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size] - input tensor
            
        Returns:
            output: [batch_size, seq_len, hidden_size] - with structured dropout applied
        """
        if not self.training or self.p == 0:
            return hidden_states
        
        if self.structured_type == 'feature':
            # Drop entire feature dimensions
            batch_size, seq_len, hidden_size = hidden_states.size()
            
            if hidden_states.device.type == "cpu":
                mask = torch.bernoulli(torch.empty(1, 1, hidden_size).fill_(1 - self.p))
            else:
                mask = torch.bernoulli(torch.empty(1, 1, hidden_size, device=hidden_states.device).fill_(1 - self.p))
            
            output = hidden_states * mask / (1 - self.p)
            
        elif self.structured_type == 'spatial':
            # Drop entire spatial positions
            batch_size, seq_len, hidden_size = hidden_states.size()
            
            if hidden_states.device.type == "cpu":
                mask = torch.bernoulli(torch.empty(1, seq_len, 1).fill_(1 - self.p))
            else:
                mask = torch.bernoulli(torch.empty(1, seq_len, 1, device=hidden_states.device).fill_(1 - self.p))
            
            output = hidden_states * mask / (1 - self.p)
            
        elif self.structured_type == 'channel':
            # Drop entire channels (last dimension for embedding)
            batch_size, seq_len, hidden_size = hidden_states.size()
            
            if hidden_states.device.type == "cpu":
                mask = torch.bernoulli(torch.empty(batch_size, 1, 1).fill_(1 - self.p))
            else:
                mask = torch.bernoulli(torch.empty(batch_size, 1, 1, device=hidden_states.device).fill_(1 - self.p))
            
            output = hidden_states * mask / (1 - self.p)
        
        return output
    
    def extra_repr(self) -> str:
        return f'p={self.p}, structured_type={self.structured_type}'


class VariationalDropout(nn.Module):
    """
    Variational dropout that maintains consistent dropout across time steps.
    
    This is particularly useful for RNNs and can be applied to transformers
    for temporal consistency.
    """
    
    def __init__(
        self,
        p: float = 0.5,
        **kwargs
    ):
        super().__init__()
        self.p = p
    
    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply variational dropout (consistent across time steps).
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size] - input tensor
            
        Returns:
            output: [batch_size, seq_len, hidden_size] - with variational dropout applied
        """
        if not self.training or self.p == 0:
            return hidden_states
        
        batch_size, seq_len, hidden_size = hidden_states.size()
        
        # Create dropout mask that is consistent across the sequence dimension
        # This ensures the same features are dropped across all time steps
        if hidden_states.device.type == "cpu":
            mask = torch.bernoulli(torch.empty(batch_size, 1, hidden_size).fill_(1 - self.p))
        else:
            mask = torch.bernoulli(torch.empty(batch_size, 1, hidden_size, device=hidden_states.device).fill_(1 - self.p))
        
        # Apply mask and scale
        output = hidden_states * mask / (1 - self.p)
        
        return output
    
    def extra_repr(self) -> str:
        return f'p={self.p}'


class DropoutScheduler:
    """
    Scheduler for dynamic dropout rates during training.
    
    Can implement various scheduling strategies like linear warmup,
    cosine annealing, or step decay for dropout rates.
    """
    
    def __init__(
        self,
        initial_dropout: float = 0.0,
        final_dropout: float = 0.0,
        schedule_type: str = 'linear',
        **kwargs
    ):
        self.initial_dropout = initial_dropout
        self.final_dropout = final_dropout
        self.schedule_type = schedule_type
        self.current_dropout = initial_dropout
        self.steps = 0
    
    def step(self):
        """Update dropout rate based on schedule."""
        self.steps += 1
        
        if self.schedule_type == 'linear':
            # Linear interpolation between initial and final dropout
            progress = min(self.steps / 1000.0, 1.0)  # Assume 1000 steps for schedule
            self.current_dropout = self.initial_dropout + progress * (self.final_dropout - self.initial_dropout)
        
        elif self.schedule_type == 'cosine':
            # Cosine annealing
            progress = min(self.steps / 1000.0, 1.0)
            self.current_dropout = self.final_dropout + 0.5 * (self.initial_dropout - self.final_dropout) * (1 + torch.cos(torch.pi * progress))
        
        elif self.schedule_type == 'step':
            # Step decay
            if self.steps == 500:
                self.current_dropout = (self.initial_dropout + self.final_dropout) / 2
            elif self.steps >= 1000:
                self.current_dropout = self.final_dropout
    
    def get_current_dropout(self) -> float:
        """Get current dropout rate."""
        return self.current_dropout
    
    def reset(self):
        """Reset scheduler to initial state."""
        self.steps = 0
        self.current_dropout = self.initial_dropout