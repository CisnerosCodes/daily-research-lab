"""Fractional Key Normalization (FKN) - drop-in attention normalization from daily-research-lab."""
from .fkn import FKNCausalSelfAttention, FractionalKeyNorm, QKNorm, patch_nanogpt

__all__ = ["FractionalKeyNorm", "FKNCausalSelfAttention", "QKNorm", "patch_nanogpt"]
__version__ = "0.1.0"
