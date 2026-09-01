"""Fractional Key Normalization (FKN): a drop-in attention normalization.

Standard QK-norm RMS-normalizes both queries and keys per head. Key-only normalization
(QUEST, ICLR 2026) normalizes only the keys. FKN normalizes only the keys, but only
*fractionally*: the per-token key magnitude re-enters through a learnable per-head
exponent alpha, measured relative to a running (EMA) per-head scale.

    r_t     = RMS_head(k_t)                                   (per token, per head; keeps grad)
    s_h     = EMA over training batches of mean_t r_t         (no grad, per head)
    k_hat_t = g * (k_t / r_t) * clamp(r_t / s_h, 1/c, c) ** alpha_h

Unclamped this is  k_hat_t = g * k_t / (r_t ** (1 - alpha_h) * s_h ** alpha_h):
    alpha = 0  ->  key-only RMSNorm (the magnitude is deleted; QUEST-like endpoint)
    alpha = 1  ->  keys divided by a per-head running scale only (no per-token norm)
    learned    ->  the model chooses how much key magnitude to keep, per head.

Empirically (daily-research-lab, Aug-Sep 2026, ~0.42M-param char LMs): at tiny heads
(head_dim=4) full QK-norm and key-only norm both pay a ~+0.13 bpc cliff vs no norm, while
FKN with learnable alpha lands ~0.07 bpc BELOW the unnormalized baseline (3/3 seeds); the
gradient path through r_t is load-bearing (restoring the magnitude *value* with the gradient
severed pays the full cliff). See paper/ in the repo for the head-width phase diagram.

Usage (any PyTorch attention):

    fkn = FractionalKeyNorm(n_head=8, head_dim=64)
    k = fkn(k)          # k: (B, T, n_head, head_dim)  or (B, n_head, T, head_dim) with layout="bhtd"
    # queries untouched; then your usual softmax(q k^T / sqrt(d)) v

nanoGPT drop-in: replace `CausalSelfAttention` with `FKNCausalSelfAttention` (same config
fields: n_embd, n_head, block_size, dropout, bias).

Notes
- Apply FKN *before* RoPE if you use rotary embeddings (RoPE is norm-preserving, so the
  order does not change r_t, but the gain g is per channel and should see un-rotated keys).
- The EMA buffer is updated only in train() mode; in eval() the last value is used.
- The EMA is a per-process buffer; under data-parallel training each rank keeps its own
  estimate (they agree to within batch noise). Sync it like BatchNorm stats if you care.
- r_t is computed in float32 for numerical safety under bf16/fp16 autocast.
"""
from __future__ import annotations

import math
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FractionalKeyNorm", "FKNCausalSelfAttention", "QKNorm"]


class FractionalKeyNorm(nn.Module):
    """Fractional per-head key normalization with a learnable magnitude exponent.

    Args:
        n_head:        number of key heads (use n_kv_head for GQA/MQA).
        head_dim:      per-head dimension.
        alpha_init:    initial exponent (1.0 = start at "running-scale only", the setting
                       that produced the reported results; 0.0 = start at key-only RMSNorm).
        alpha_learnable: if False, alpha is frozen at alpha_init.
        ratio_clamp:   c in clamp(r_t / s_h, 1/c, c); bounds the per-token rescale.
        ema_momentum:  momentum of the running per-head scale s_h.
        gain:          learnable per-channel gain (length n_head*head_dim), init 1.
        eps:           RMS epsilon.
        layout:        "bthd" for (B, T, H, D) [default] or "bhtd" for (B, H, T, D).
    """

    def __init__(self, n_head: int, head_dim: int, *, alpha_init: float = 1.0,
                 alpha_learnable: bool = True, ratio_clamp: float = 8.0,
                 ema_momentum: float = 0.99, gain: bool = True, eps: float = 1e-6,
                 layout: Literal["bthd", "bhtd"] = "bthd"):
        super().__init__()
        assert layout in ("bthd", "bhtd")
        self.n_head, self.head_dim = n_head, head_dim
        self.ratio_clamp, self.ema_momentum, self.eps, self.layout = ratio_clamp, ema_momentum, eps, layout
        self.alpha = nn.Parameter(torch.full((n_head,), float(alpha_init)), requires_grad=alpha_learnable)
        self.gain = nn.Parameter(torch.ones(n_head * head_dim)) if gain else None
        self.register_buffer("rms_ema", torch.ones(n_head))
        self.register_buffer("ema_ready", torch.zeros(1))

    def extra_repr(self) -> str:
        return (f"n_head={self.n_head}, head_dim={self.head_dim}, alpha_learnable={self.alpha.requires_grad}, "
                f"ratio_clamp={self.ratio_clamp}, ema_momentum={self.ema_momentum}, layout={self.layout}")

    @torch.no_grad()
    def _update_ema(self, r_det: torch.Tensor):
        # r_det: (B, T, H) per-token per-head RMS, detached
        batch_mean = r_det.float().mean(dim=(0, 1))
        if float(self.ema_ready) == 0.0:
            self.rms_ema.copy_(batch_mean)
            self.ema_ready.fill_(1.0)
        else:
            self.rms_ema.mul_(self.ema_momentum).add_(batch_mean, alpha=1 - self.ema_momentum)

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        if self.layout == "bhtd":
            k = k.transpose(1, 2)                      # -> (B, T, H, D)
        B, T, H, D = k.shape
        assert H == self.n_head and D == self.head_dim, (k.shape, self.n_head, self.head_dim)
        k32 = k.float()
        r = k32.pow(2).mean(-1).add(1e-12).sqrt()      # (B, T, H), gradient flows through r
        k_dir = k32 * torch.rsqrt(k32.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.training:
            self._update_ema(r.detach())
        c = self.ratio_clamp
        ratio = (r / self.rms_ema.view(1, 1, -1)).clamp(1.0 / c, c)
        tau = torch.exp(self.alpha.view(1, 1, -1) * ratio.log())    # (B, T, H)
        out = k_dir * tau.unsqueeze(-1)
        if self.gain is not None:
            out = out.reshape(B, T, H * D) * self.gain
            out = out.view(B, T, H, D)
        out = out.to(k.dtype)
        if self.layout == "bhtd":
            out = out.transpose(1, 2)
        return out

    @torch.no_grad()
    def effective_exponent(self) -> torch.Tensor:
        """Per-head exponent on the raw key magnitude: k_hat ~ k * r^(alpha-1)."""
        return self.alpha.detach() - 1.0


class QKNorm(nn.Module):
    """Reference per-head RMSNorm + per-channel gain (the QK-norm / key-only-norm baseline)."""

    def __init__(self, n_head: int, head_dim: int, eps: float = 1e-6, layout: str = "bthd"):
        super().__init__()
        self.n_head, self.head_dim, self.eps, self.layout = n_head, head_dim, eps, layout
        self.gain = nn.Parameter(torch.ones(n_head * head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layout == "bhtd":
            x = x.transpose(1, 2)
        B, T, H, D = x.shape
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = (x.reshape(B, T, H * D) * self.gain).view(B, T, H, D)
        return x.transpose(1, 2) if self.layout == "bhtd" else x


class FKNCausalSelfAttention(nn.Module):
    """nanoGPT-compatible causal self-attention with a selectable q/k normalization.

    `config` needs: n_embd, n_head, block_size, dropout, bias  (exactly nanoGPT's GPTConfig).
    norm: "none" | "qknorm" | "knorm" | "fkn"   (fkn = FractionalKeyNorm on keys, queries untouched)
    """

    def __init__(self, config, norm: str = "fkn", fkn_kwargs: Optional[dict] = None):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert norm in ("none", "qknorm", "knorm", "fkn")
        self.n_head, self.n_embd = config.n_head, config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.norm = norm
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        if norm == "qknorm":
            self.q_norm = QKNorm(self.n_head, self.head_dim)
            self.k_norm = QKNorm(self.n_head, self.head_dim)
        elif norm == "knorm":
            self.k_norm = QKNorm(self.n_head, self.head_dim)
        elif norm == "fkn":
            self.k_norm = FractionalKeyNorm(self.n_head, self.head_dim, **(fkn_kwargs or {}))
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                 .view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        if self.norm == "qknorm":
            q, k = self.q_norm(q), self.k_norm(k)
        elif self.norm in ("knorm", "fkn"):
            k = self.k_norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)   # (B, nh, T, hd)
        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                               dropout_p=self.dropout if self.training else 0,
                                               is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = self.attn_dropout(F.softmax(att, dim=-1))
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


def patch_nanogpt(model_module, norm: str = "fkn", fkn_kwargs: Optional[dict] = None):
    """Monkey-patch nanoGPT's `model` module so `GPT(config)` builds FKN attention.

        import model as nanogpt_model
        from fkn import patch_nanogpt
        patch_nanogpt(nanogpt_model, norm="fkn")
        gpt = nanogpt_model.GPT(config)
    """
    class _Patched(FKNCausalSelfAttention):
        def __init__(self, config):
            super().__init__(config, norm=norm, fkn_kwargs=fkn_kwargs)
    model_module.CausalSelfAttention = _Patched
    return model_module
