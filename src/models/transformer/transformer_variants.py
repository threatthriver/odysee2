"""
Transformer Variants

This module implements various transformer model architectures including
GPT-style, BERT-style, and T5-style models with different configurations
optimized for their respective use cases.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Union

from .transformer_layer import (
    TransformerLayer, TransformerBlock, 
    TransformerEncoderLayer, TransformerDecoderLayer
)
from .embedding import CombinedEmbedding, TokenEmbedding, PositionEmbedding, RotaryEmbedding
from .positional_encoding import ALiBiPositionalEncoding


class GPTModel(nn.Module):
    """
    GPT-style language model (autoregressive decoder-only).
    
    This model uses:
    - Causal self-attention (autoregressive)
    - Single embedding type
    - Position embeddings (learned or RoPE/ALiBi)
    - No encoder-decoder structure
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: Optional[int] = None,
        max_position_embeddings: int = 2048,
        hidden_act: str = 'gelu',
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        layer_norm_epsilon: float = 1e-5,
        pre_norm: bool = True,
        use_rms_norm: bool = False,
        use_flash_attention: bool = False,
        use_rotary_embeddings: bool = False,
        use_alibi: bool = False,
        embedding_dropout: float = 0.0,
        num_label_classes: Optional[int] = None,  # For classification head
        **kwargs
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.use_rotary_embeddings = use_rotary_embeddings
        self.use_alibi = use_alibi
        self.num_label_classes = num_label_classes
        
        # Embedding configuration
        self.use_position_embeddings = not (use_rotary_embeddings or use_alibi)
        
        # Token embedding
        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            dropout=embedding_dropout
        )
        
        # Position embedding (if not using RoPE or ALiBi)
        if self.use_position_embeddings:
            self.position_embedding = PositionEmbedding(
                max_position_embeddings=max_position_embeddings,
                hidden_size=hidden_size,
                dropout=embedding_dropout
            )
        
        # Rotary embeddings (if using RoPE)
        if use_rotary_embeddings:
            self.rotary_embeddings = RotaryEmbedding(
                hidden_size=hidden_size,
                max_position_embeddings=max_position_embeddings
            )
        
        # ALiBi positional encoding (if using ALiBi)
        if use_alibi:
            self.alibi_embeddings = ALiBiPositionalEncoding(
                num_attention_heads=num_attention_heads,
                max_position_embeddings=max_position_embeddings
            )
        
        # Dropout
        self.dropout = nn.Dropout(embedding_dropout)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerDecoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size or 4 * hidden_size,
                hidden_act=hidden_act,
                attention_dropout=attention_dropout,
                hidden_dropout=hidden_dropout,
                layer_norm_epsilon=layer_norm_epsilon,
                pre_norm=pre_norm,
                use_rms_norm=use_rms_norm,
                use_flash_attention=use_flash_attention,
                use_causal_mask=True,
                add_cross_attention=False,  # No cross-attention in GPT
                **kwargs
            ) for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        
        # Classification head (optional)
        if num_label_classes is not None:
            self.classifier = nn.Linear(hidden_size, num_label_classes)
        else:
            self.classifier = None
    
    def create_position_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create position IDs from input IDs."""
        batch_size, seq_len = input_ids.size()
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, -1)
        return position_ids
    
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs
    ) -> Union[torch.Tensor, dict]:
        """
        Forward pass of GPT model.
        
        Args:
            input_ids: [batch_size, seq_len] - input token IDs
            position_ids: [batch_size, seq_len] - optional position indices
            past_key_values: List of (key, value) tuples for each layer
            attention_mask: [batch_size, seq_len] - attention mask
            head_mask: [num_layers, num_heads] - mask for attention heads
            labels: [batch_size, seq_len] - optional labels for loss computation
            use_cache: whether to use past key values for faster generation
            output_attentions: whether to return attention weights
            output_hidden_states: whether to return hidden states
            return_dict: whether to return a dictionary or tensor
            
        Returns:
            outputs with various components based on configuration
        """
        batch_size, seq_len = input_ids.size()
        
        # Create position IDs if not provided
        if position_ids is None:
            position_ids = self.create_position_ids(input_ids)
        
        # Embedding
        # Token embeddings
        hidden_states = self.token_embedding(input_ids)
        
        # Add position embeddings or apply rotary/ALiBi
        if self.use_position_embeddings:
            position_embeddings = self.position_embedding(position_ids)
            hidden_states = hidden_states + position_embeddings
        elif self.use_rotary_embeddings:
            hidden_states = self.rotary_embeddings(hidden_states, position_ids)
        # ALiBi is applied in attention mechanism
        
        # Apply dropout
        hidden_states = self.dropout(hidden_states)
        
        # Initialize containers for outputs
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        present_key_values = () if use_cache else None
        
        # Process attention mask
        if attention_mask is not None:
            # Convert to attention mask format
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Process head mask
        if head_mask is not None:
            head_mask = head_mask.view(-1, self.num_attention_heads)
            head_mask = head_mask[:, None, None, :]
        else:
            head_mask = [None] * self.num_layers
        
        # Forward pass through transformer layers
        for i, layer in enumerate(self.transformer_layers):
            # Collect hidden states
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            # Forward pass through layer
            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[i] if head_mask else None,
                past_key_value=past_key_values[i] if past_key_values else None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **kwargs
            )
            
            hidden_states = layer_outputs[0]
            
            # Collect attention weights
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
            
            # Collect present key values
            if use_cache:
                present_key_values = present_key_values + (layer_outputs[-1],)
        
        # Apply final layer norm
        hidden_states = self.final_layer_norm(hidden_states)
        
        # Collect final hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # Classification head
        logits = None
        loss = None
        if self.classifier is not None:
            # Use final token for classification
            pooled_output = hidden_states[:, -1, :]  # Take last token
            logits = self.classifier(pooled_output)
            
            if labels is not None:
                loss = nn.CrossEntropyLoss()(logits, labels)
        
        # Language modeling head (if no classification head)
        if self.classifier is None:
            logits = torch.matmul(hidden_states, self.token_embedding.token_embeddings.weight.t())
            
            if labels is not None:
                # Shift labels for causal language modeling
                shift_labels = labels[..., 1:].contiguous()
                shift_logits = logits[..., :-1, :].contiguous()
                
                # Flatten for loss computation
                loss = nn.CrossEntropyLoss()(
                    shift_logits.view(-1, self.vocab_size), 
                    shift_labels.view(-1)
                )
        
        if return_dict:
            return {
                'last_hidden_state': hidden_states,
                'hidden_states': all_hidden_states,
                'attentions': all_self_attentions,
                'past_key_values': present_key_values,
                'logits': logits,
                'loss': loss
            }
        else:
            return hidden_states, loss, logits


class BERTModel(nn.Module):
    """
    BERT-style encoder-only model (bidirectional attention).
    
    This model uses:
    - Bidirectional self-attention (no causality constraint)
    - Combined embeddings (token, position, segment)
    - Pre-training objectives (MLM, NSP)
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: Optional[int] = None,
        max_position_embeddings: int = 512,
        num_segments: int = 2,
        hidden_act: str = 'gelu',
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-12,
        pre_norm: bool = True,
        use_rms_norm: bool = False,
        use_flash_attention: bool = False,
        embedding_dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.num_segments = num_segments
        
        # Combined embedding layer
        self.embeddings = CombinedEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            num_segments=num_segments,
            dropout=embedding_dropout
        )
        
        # Dropout
        self.dropout = nn.Dropout(embedding_dropout)
        
        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size or 4 * hidden_size,
                hidden_act=hidden_act,
                attention_dropout=attention_dropout,
                hidden_dropout=hidden_dropout,
                layer_norm_epsilon=layer_norm_epsilon,
                pre_norm=pre_norm,
                use_rms_norm=use_rms_norm,
                use_flash_attention=use_flash_attention,
                add_cross_attention=False,  # No cross-attention in BERT
                **kwargs
            ) for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs
    ) -> Union[torch.Tensor, dict]:
        """
        Forward pass of BERT model.
        
        Args:
            input_ids: [batch_size, seq_len] - input token IDs
            attention_mask: [batch_size, seq_len] - attention mask (1 for real, 0 for padding)
            token_type_ids: [batch_size, seq_len] - segment IDs (for NSP)
            position_ids: [batch_size, seq_len] - optional position indices
            head_mask: [num_layers, num_heads] - mask for attention heads
            inputs_embeds: [batch_size, seq_len, hidden_size] - optional input embeddings
            output_attentions: whether to return attention weights
            output_hidden_states: whether to return hidden states
            return_dict: whether to return a dictionary or tensor
            
        Returns:
            outputs with various components
        """
        batch_size, seq_len = input_ids.size() if input_ids is not None else inputs_embeds.size()[:2]
        
        # Create default position IDs
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device if input_ids is not None else inputs_embeds.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Create default token type IDs
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # Process attention mask
        if attention_mask is not None:
            # Convert to attention mask format: [batch_size, 1, 1, seq_len]
            attention_mask = attention_mask.view(batch_size, 1, 1, seq_len)
            attention_mask = attention_mask.to(dtype=torch.float32)
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Process head mask
        if head_mask is not None:
            head_mask = head_mask.view(-1, self.num_attention_heads)
            head_mask = head_mask[:, None, None, :]
        else:
            head_mask = [None] * self.num_layers
        
        # Embedding
        if inputs_embeds is None:
            hidden_states = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                segment_ids=token_type_ids
            )
        else:
            hidden_states = inputs_embeds
        
        # Apply dropout
        hidden_states = self.dropout(hidden_states)
        
        # Initialize containers for outputs
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        
        # Forward pass through transformer layers
        for i, layer in enumerate(self.transformer_layers):
            # Collect hidden states
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            # Forward pass through layer
            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[i] if head_mask else None,
                output_attentions=output_attentions,
                **kwargs
            )
            
            hidden_states = layer_outputs[0]
            
            # Collect attention weights
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        
        # Apply final layer norm
        hidden_states = self.final_layer_norm(hidden_states)
        
        # Collect final hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        if return_dict:
            return {
                'last_hidden_state': hidden_states,
                'hidden_states': all_hidden_states,
                'attentions': all_self_attentions
            }
        else:
            return hidden_states


class T5Model(nn.Module):
    """
    T5-style encoder-decoder model.
    
    This model uses:
    - Encoder with bidirectional attention
    - Decoder with causal self-attention and cross-attention
    - Relative positional encoding
    - Encoder-decoder attention mechanism
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        num_encoder_layers: int = 12,
        num_decoder_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: Optional[int] = None,
        max_position_embeddings: int = 512,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        hidden_act: str = 'gelu',
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-6,
        pre_norm: bool = False,  # T5 uses post-norm
        use_rms_norm: bool = False,
        use_flash_attention: bool = False,
        embedding_dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        
        # Token embedding
        self.shared_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            dropout=embedding_dropout
        )
        
        # Position embedding
        self.position_embedding = PositionEmbedding(
            max_position_embeddings=max_position_embeddings,
            hidden_size=hidden_size,
            dropout=embedding_dropout
        )
        
        # Dropout
        self.dropout = nn.Dropout(embedding_dropout)
        
        # Encoder
        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size or 4 * hidden_size,
                hidden_act=hidden_act,
                attention_dropout=attention_dropout,
                hidden_dropout=hidden_dropout,
                layer_norm_epsilon=layer_norm_epsilon,
                pre_norm=pre_norm,
                use_rms_norm=use_rms_norm,
                use_flash_attention=use_flash_attention,
                add_cross_attention=False,
                **kwargs
            ) for _ in range(num_encoder_layers)
        ])
        
        # Decoder
        self.decoder = nn.ModuleList([
            TransformerDecoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size or 4 * hidden_size,
                hidden_act=hidden_act,
                attention_dropout=attention_dropout,
                hidden_dropout=hidden_dropout,
                layer_norm_epsilon=layer_norm_epsilon,
                pre_norm=pre_norm,
                use_rms_norm=use_rms_norm,
                use_flash_attention=use_flash_attention,
                add_cross_attention=True,  # Cross-attention for encoder-decoder
                **kwargs
            ) for _ in range(num_decoder_layers)
        ])
        
        # Final layer norms
        self.encoder_final_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.decoder_final_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs
    ) -> Union[torch.Tensor, dict]:
        """
        Forward pass of T5 model.
        
        Args:
            input_ids: [batch_size, enc_seq_len] - encoder input IDs
            attention_mask: [batch_size, enc_seq_len] - encoder attention mask
            decoder_input_ids: [batch_size, dec_seq_len] - decoder input IDs
            decoder_attention_mask: [batch_size, dec_seq_len] - decoder attention mask
            encoder_outputs: [batch_size, enc_seq_len, hidden_size] - encoder outputs
            past_key_values: decoder past key values
            head_mask: [num_encoder_layers, num_heads] - encoder head mask
            decoder_head_mask: [num_decoder_layers, num_heads] - decoder head mask
            cross_attn_head_mask: [num_decoder_layers, num_heads] - cross-attention head mask
            inputs_embeds: [batch_size, enc_seq_len, hidden_size] - encoder embeddings
            decoder_inputs_embeds: [batch_size, dec_seq_len, hidden_size] - decoder embeddings
            use_cache: whether to use past key values
            output_attentions: whether to return attention weights
            output_hidden_states: whether to return hidden states
            return_dict: whether to return a dictionary or tensor
            
        Returns:
            outputs with encoder-decoder components
        """
        # Forward pass through encoder
        encoder_outputs = self.encoder_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs
        )
        
        # Forward pass through decoder
        decoder_outputs = self.decoder_forward(
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs['last_hidden_state'],
            past_key_values=past_key_values,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs
        )
        
        # Combine outputs
        if return_dict:
            return {
                'encoder_last_hidden_state': encoder_outputs['last_hidden_state'],
                'decoder_last_hidden_state': decoder_outputs['last_hidden_state'],
                'encoder_hidden_states': encoder_outputs.get('hidden_states'),
                'decoder_hidden_states': decoder_outputs.get('hidden_states'),
                'encoder_attentions': encoder_outputs.get('attentions'),
                'decoder_attentions': decoder_outputs.get('attentions'),
                'cross_attentions': decoder_outputs.get('cross_attentions'),
                'past_key_values': decoder_outputs.get('past_key_values'),
            }
        else:
            return decoder_outputs
    
    def encoder_forward(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        inputs_embeds: Optional[torch.Tensor],
        head_mask: Optional[torch.Tensor],
        output_attentions: bool,
        output_hidden_states: bool,
        **kwargs
    ) -> dict:
        """Forward pass through encoder."""
        batch_size, seq_len = (
            input_ids.size() if input_ids is not None else inputs_embeds.size()[:2]
        )
        
        # Process attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, 1, 1, seq_len)
            attention_mask = attention_mask.to(dtype=torch.float32)
            attention_mask = (1.0 - attention_mask) * -10000.0
        else:
            attention_mask = None
        
        # Process head mask
        if head_mask is not None:
            head_mask = head_mask.view(-1, self.num_attention_heads)
            head_mask = head_mask[:, None, None, :]
        else:
            head_mask = [None] * self.num_encoder_layers
        
        # Embedding
        if inputs_embeds is None:
            # Use shared embedding
            hidden_states = self.shared_embedding(input_ids)
        else:
            hidden_states = inputs_embeds
        
        # Add positional embeddings
        position_ids = torch.arange(seq_len, device=input_ids.device if input_ids is not None else inputs_embeds.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embedding(position_ids)
        hidden_states = hidden_states + position_embeddings
        
        # Apply dropout
        hidden_states = self.dropout(hidden_states)
        
        # Initialize containers
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        
        # Forward pass through encoder layers
        for i, layer in enumerate(self.encoder):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[i] if head_mask else None,
                output_attentions=output_attentions,
                **kwargs
            )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        
        # Apply final layer norm
        hidden_states = self.encoder_final_layer_norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        return {
            'last_hidden_state': hidden_states,
            'hidden_states': all_hidden_states,
            'attentions': all_self_attentions
        }
    
    def decoder_forward(
        self,
        decoder_input_ids: Optional[torch.Tensor],
        decoder_attention_mask: Optional[torch.Tensor],
        encoder_hidden_states: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
        decoder_head_mask: Optional[torch.Tensor],
        cross_attn_head_mask: Optional[torch.Tensor],
        decoder_inputs_embeds: Optional[torch.Tensor],
        use_cache: bool,
        output_attentions: bool,
        output_hidden_states: bool,
        **kwargs
    ) -> dict:
        """Forward pass through decoder."""
        batch_size, seq_len = (
            decoder_input_ids.size() if decoder_input_ids is not None else decoder_inputs_embeds.size()[:2]
        )
        
        # Process attention mask
        if decoder_attention_mask is not None:
            decoder_attention_mask = decoder_attention_mask.view(batch_size, 1, 1, seq_len)
            decoder_attention_mask = decoder_attention_mask.to(dtype=torch.float32)
            decoder_attention_mask = (1.0 - decoder_attention_mask) * -10000.0
        else:
            decoder_attention_mask = None
        
        # Process head masks
        if decoder_head_mask is not None:
            decoder_head_mask = decoder_head_mask.view(-1, self.num_attention_heads)
            decoder_head_mask = decoder_head_mask[:, None, None, :]
        else:
            decoder_head_mask = [None] * self.num_decoder_layers
        
        if cross_attn_head_mask is not None:
            cross_attn_head_mask = cross_attn_head_mask.view(-1, self.num_attention_heads)
            cross_attn_head_mask = cross_attn_head_mask[:, None, None, :]
        else:
            cross_attn_head_mask = [None] * self.num_decoder_layers
        
        # Embedding
        if decoder_inputs_embeds is None:
            # Use shared embedding
            hidden_states = self.shared_embedding(decoder_input_ids)
        else:
            hidden_states = decoder_inputs_embeds
        
        # Add positional embeddings
        position_ids = torch.arange(seq_len, device=decoder_input_ids.device if decoder_input_ids is not None else decoder_inputs_embeds.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embedding(position_ids)
        hidden_states = hidden_states + position_embeddings
        
        # Apply dropout
        hidden_states = self.dropout(hidden_states)
        
        # Initialize containers
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None
        present_key_values = () if use_cache else None
        
        # Forward pass through decoder layers
        for i, layer in enumerate(self.decoder):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=decoder_attention_mask,
                head_mask=decoder_head_mask[i] if decoder_head_mask else None,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=None,  # T5 handles encoder mask differently
                cross_attention_head_mask=cross_attn_head_mask[i] if cross_attn_head_mask else None,
                past_key_value=past_key_values[i] if past_key_values else None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **kwargs
            )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if layer_outputs[2] is not None:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
            
            if use_cache:
                present_key_values = present_key_values + (layer_outputs[-1],)
        
        # Apply final layer norm
        hidden_states = self.decoder_final_layer_norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        return {
            'last_hidden_state': hidden_states,
            'hidden_states': all_hidden_states,
            'attentions': all_self_attentions,
            'cross_attentions': all_cross_attentions,
            'past_key_values': present_key_values
        }