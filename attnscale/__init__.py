"""attnscale: diagnose and fix the attention logit scale at initialization.

See the paper in ../paper/ (daily-research-lab, 2026-09-02).
"""
from .attnscale import (FractionalKeyNorm, KeyScale, QKNorm, ScaledAttention,
                        initial_logit_std, patch_nanogpt, scale_key_projection_,
                        suggest_key_scale)

__all__ = ["KeyScale", "scale_key_projection_", "initial_logit_std", "suggest_key_scale",
           "FractionalKeyNorm", "QKNorm", "ScaledAttention", "patch_nanogpt"]
__version__ = "0.2.0"
