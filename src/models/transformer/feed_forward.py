"""
Feed-Forward Networks and Gated Linear Units

This module implements various feed-forward network architectures commonly used
in transformer models, including Gated Linear Units (GLU) variants and memory-efficient
implementations for large-scale training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN) with dropout.
    
    This implementation follows the standard transformer architecture with:
    - Two linear transformations with activation in between
    - Dropout for regularization
    - Residual connection support
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = 'gelu',
        dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        bias: bool = True,
        layer_norm_epsilon: float = 1e-12,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation = activation
        self.dropout = dropout
        self.hidden_dropout = hidden_dropout
        
        # First linear transformation
        self.dense_h_to_4h = nn.Linear(hidden_size, intermediate_size, bias=bias)
        
        # Second linear transformation
        self.dense_4h_to_h = nn.Linear(intermediate_size, hidden_size, bias=bias)
        
        # Dropout layers
        self.dropout_layer = nn.Dropout(dropout)
        self.hidden_dropout_layer = nn.Dropout(hidden_dropout)
        
        # Activation function
        self.activation_fn = self._get_activation_function(activation)
        
        self._reset_parameters()
    
    def _get_activation_function(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
            'swish': nn.SiLU(),
            'gelu_new': nn.GELU(approximate='tanh'),
            'linear': nn.Identity()
        }
        
        if activation not in activations:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        return activations[activation]
    
    def _reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        # Initialize linear layers
        nn.init.xavier_uniform_(self.dense_h_to_4h.weight)
        nn.init.xavier_uniform_(self.dense_4h_to_h.weight)
        
        # Initialize biases to zero
        if self.dense_h_to_4h.bias is not None:
            nn.init.zeros_(self.dense_h_to_4h.bias)
        if self.dense_4h_to_h.bias is not None:
            nn.init.zeros_(self.dense_4h_to_h.bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Apply feed-forward transformation.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        # First linear transformation
        intermediate_output = self.dense_h_to_4h(hidden_states)
        
        # Apply activation
        intermediate_output = self.activation_fn(intermediate_output)
        
        # Apply hidden dropout during training
        if self.training:
            intermediate_output = self.hidden_dropout_layer(intermediate_output)
        
        # Second linear transformation
        output = self.dense_4h_to_h(intermediate_output)
        
        # Apply dropout
        if self.training:
            output = self.dropout_layer(output)
        
        return output
    
    def extra_repr(self) -> str:
        return (f'hidden_size={self.hidden_size}, intermediate_size={self.intermediate_size}, '
                f'activation={self.activation}, dropout={self.dropout}')


class GatedLinearUnit(nn.Module):
    """
    Gated Linear Unit (GLU) feed-forward network.
    
    GLU introduces a gating mechanism that controls information flow,
    often leading to better performance than standard feed-forward networks.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = 'gelu',
        gating_method: str = 'gated',
        dropout: float = 0.0,
        bias: bool = True,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation = activation
        self.gating_method = gating_method
        self.dropout = dropout
        
        # Linear transformations for gating
        if gating_method == 'gated':
            # Standard GLU with separate gate and value projections
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
            self.value_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
            self.out_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        elif gating_method == 'parallel':
            # Parallel GLU where gate and value are computed in parallel
            self.linear = nn.Linear(hidden_size, 2 * intermediate_size, bias=bias)
            self.out_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        else:
            raise ValueError(f"Unsupported gating method: {gating_method}")
        
        self.dropout_layer = nn.Dropout(dropout)
        self.activation_fn = self._get_activation_function(activation)
        
        self._reset_parameters()
    
    def _get_activation_function(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
            'swish': nn.SiLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh()
        }
        
        if activation not in activations:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        return activations[activation]
    
    def _reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        if self.gating_method == 'gated':
            nn.init.xavier_uniform_(self.gate_proj.weight)
            nn.init.xavier_uniform_(self.value_proj.weight)
            nn.init.xavier_uniform_(self.out_proj.weight)
            
            if self.gate_proj.bias is not None:
                nn.init.zeros_(self.gate_proj.bias)
            if self.value_proj.bias is not None:
                nn.init.zeros_(self.value_proj.bias)
            if self.out_proj.bias is not None:
                nn.init.zeros_(self.out_proj.bias)
        else:  # parallel
            nn.init.xavier_uniform_(self.linear.weight)
            nn.init.xavier_uniform_(self.out_proj.weight)
            
            if self.linear.bias is not None:
                nn.init.zeros_(self.linear.bias)
            if self.out_proj.bias is not None:
                nn.init.zeros_(self.out_proj.bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Apply gated linear unit transformation.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        if self.gating_method == 'gated':
            # Compute gate and value
            gate = self.gate_proj(hidden_states)
            value = self.value_proj(hidden_states)
            
            # Apply activation to gate
            gate = self.activation_fn(gate)
            
            # Element-wise multiplication
            gated_output = gate * value
            
        else:  # parallel
            # Compute gate and value in parallel
            gate_value = self.linear(hidden_states)
            gate, value = gate_value.chunk(2, dim=-1)
            
            # Apply activation to gate
            gate = self.activation_fn(gate)
            
            # Element-wise multiplication
            gated_output = gate * value
        
        # Output projection
        output = self.out_proj(gated_output)
        
        # Apply dropout
        if self.training:
            output = self.dropout_layer(output)
        
        return output
    
    def extra_repr(self) -> str:
        return (f'hidden_size={self.hidden_size}, intermediate_size={self.intermediate_size}, '
                f'activation={self.activation}, gating_method={self.gating_method}')


class ParallelFeedForward(nn.Module):
    """
    Memory-efficient parallel feed-forward network.
    
    This implementation computes multiple feed-forward transformations
    in parallel to improve throughput on modern hardware.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_parallel: int = 4,
        activation: str = 'gelu',
        dropout: float = 0.0,
        bias: bool = True,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_parallel = num_parallel
        self.activation = activation
        self.dropout = dropout
        
        # Combine all linear transformations into one large linear layer
        combined_size = intermediate_size * num_parallel
        self.combined_proj = nn.Linear(hidden_size, combined_size, bias=bias)
        self.out_proj = nn.Linear(combined_size, hidden_size, bias=bias)
        
        self.dropout_layer = nn.Dropout(dropout)
        self.activation_fn = self._get_activation_function(activation)
        
        self._reset_parameters()
    
    def _get_activation_function(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
            'swish': nn.SiLU()
        }
        
        if activation not in activations:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        return activations[activation]
    
    def _reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.combined_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.combined_proj.bias is not None:
            nn.init.zeros_(self.combined_proj.bias)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Apply parallel feed-forward transformation.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.size()
        
        # Combined projection
        combined_output = self.combined_proj(hidden_states)
        
        # Reshape for parallel computation
        # [batch_size, seq_len, num_parallel, intermediate_size]
        combined_output = combined_output.view(
            batch_size, seq_len, self.num_parallel, self.intermediate_size
        )
        
        # Apply activation to all parallel paths
        combined_output = self.activation_fn(combined_output)
        
        # Combine parallel outputs through averaging or concatenation
        if self.num_parallel > 1:
            # Average pooling across parallel paths
            combined_output = combined_output.mean(dim=2)
        else:
            # Single path - no averaging needed
            combined_output = combined_output.squeeze(2)
        
        # Final projection
        output = self.out_proj(combined_output)
        
        # Apply dropout
        if self.training:
            output = self.dropout_layer(output)
        
        return output
    
    def extra_repr(self) -> str:
        return (f'hidden_size={self.hidden_size}, intermediate_size={self.intermediate_size}, '
                f'num_parallel={self.num_parallel}, activation={self.activation}')


class mixture_of_experts(nn.Module):
    """
    Mixture of Experts (MoE) feed-forward network.
    
    MoE allows the model to have many more parameters while maintaining
    computational efficiency by only activating a subset of experts per token.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        top_k: int = 2,
        activation: str = 'gelu',
        dropout: float = 0.0,
        bias: bool = True,
        load_balancing_coef: float = 0.01,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.activation = activation
        self.dropout = dropout
        self.load_balancing_coef = load_balancing_coef
        
        # Create experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, intermediate_size, bias=bias),
                self._get_activation_function(activation),
                nn.Linear(intermediate_size, hidden_size, bias=bias)
            ) for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Linear(hidden_size, num_experts, bias=bias)
        
        self.dropout_layer = nn.Dropout(dropout)
        
        self._reset_parameters()
    
    def _get_activation_function(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
            'swish': nn.SiLU()
        }
        
        if activation not in activations:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        return activations[activation]
    
    def _reset_parameters(self):
        """Initialize parameters."""
        for expert in self.experts:
            for layer in expert:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        
        nn.init.xavier_uniform_(self.gate.weight)
        if self.gate.bias is not None:
            nn.init.zeros_(self.gate.bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Apply mixture of experts transformation.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            
        Returns:
            output: [batch_size, seq_len, hidden_size]
            loss: Load balancing loss
            aux_loss: Additional auxiliary losses
        """
        batch_size, seq_len, hidden_size = hidden_states.size()
        
        # Flatten tokens for easier processing
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        
        # Compute gating weights
        gate_logits = self.gate(hidden_states_flat)
        
        # Select top-k experts
        gate_weights, selected_experts = torch.topk(
            gate_logits, self.top_k, dim=-1
        )
        
        # Apply softmax to gate weights
        gate_weights = F.softmax(gate_weights, dim=-1, dtype=torch.float32)
        
        # Initialize output
        output = torch.zeros_like(hidden_states_flat)
        
        # Initialize load balancing loss
        gate_mean = gate_logits.mean(dim=0)
        gate_var = gate_logits.var(dim=0)
        load_balancing_loss = self.load_balancing_coef * (
            (gate_var / (gate_mean ** 2 + 1e-8)).mean()
        )
        
        # Compute expert outputs
        for i, expert in enumerate(self.experts):
            # Find tokens assigned to this expert
            expert_mask = (selected_experts == i).any(dim=-1)
            if expert_mask.any():
                # Get tokens for this expert
                tokens = hidden_states_flat[expert_mask]
                expert_gate_weights = gate_weights[expert_mask]
                
                # Compute expert output
                expert_output = expert(tokens)
                
                # Weighted combination
                if self.top_k > 1:
                    # Average weights across selected experts for this token
                    token_mask = selected_experts[expert_mask] == i
                    weight_sum = token_mask.float().sum(dim=-1, keepdim=True)
                    expert_gate_weights = token_mask.float() / weight_sum
                
                # Add to final output
                output[expert_mask] += expert_gate_weights * expert_output
        
        # Apply dropout
        if self.training:
            output = self.dropout_layer(output)
        
        # Reshape back to original shape
        output = output.view(batch_size, seq_len, hidden_size)
        
        return output, load_balancing_loss
    
    def extra_repr(self) -> str:
        return (f'hidden_size={self.hidden_size}, intermediate_size={self.intermediate_size}, '
                f'num_experts={self.num_experts}, top_k={self.top_k}')