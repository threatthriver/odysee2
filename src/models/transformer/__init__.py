"""
Transformer Architecture Package

This package contains implementations of various transformer architectures
for large-scale language model training, including memory-efficient optimizations.
"""

from .attention import MultiHeadAttention, FlashAttention
from .transformer_layer import TransformerLayer, TransformerBlock
from .positional_encoding import PositionalEncoding, RotaryPositionalEncoding, ALiBiPositionalEncoding
from .layer_norm import LayerNorm, RMSLayerNorm
from .feed_forward import FeedForward, GatedLinearUnit
from .transformer_variants import GPTModel, BERTModel, T5Model
from .embedding import TokenEmbedding, SegmentEmbedding, PositionEmbedding
from .dropout import DropPath

__all__ = [
    "MultiHeadAttention",
    "FlashAttention", 
    "TransformerLayer",
    "TransformerBlock",
    "PositionalEncoding",
    "RotaryPositionalEncoding", 
    "ALiBiPositionalEncoding",
    "LayerNorm",
    "RMSLayerNorm",
    "FeedForward",
    "GatedLinearUnit",
    "GPTModel",
    "BERTModel", 
    "T5Model",
    "TokenEmbedding",
    "SegmentEmbedding",
    "PositionEmbedding",
    "DropPath"
]