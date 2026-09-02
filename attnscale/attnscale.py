"""attnscale: diagnose and fix the attention logit scale at initialization.

The finding this package exists for (daily-research-lab, Jul-Sep 2026, ~0.42M-param char LMs):

  At small head dimension the standard 1/sqrt(head_dim) leaves attention logits far too small at
  initialization. A single fixed per-head multiplier on the keys fixes it and beats every
  normalization scheme we tried, including QK-norm and key-only norm. The best multiplier moves
  with head width. The SAME dial made learnable from its default does NOT travel to the optimum
  within a normal budget, which is why a learnable temperature looks useless when you start it at 1.

So the recommended workflow is: MEASURE your initial logit scale, SWEEP the constant once, and
then either multiply your keys by it or fold it into the key projection's initialization.

    from attnscale import initial_logit_std, KeyScale

    print(initial_logit_std(q, k, head_dim))   # diagnose: ours was 0.05 at head_dim 4
    kscale = KeyScale(n_head=32, c=8.0)        # fix: one constant, zero parameters
    k = kscale(k)                              # then softmax(q k^T / sqrt(d)) v as usual

Measure the constant for YOUR configuration rather than copying ours; it depends on the
initialization scheme, the head width, and the corpus. `python -m attnscale.bench --sweep-c`
does it in a few CPU-minutes.

Also here, for reference and for reproducing the paper's arms:
  QKNorm             per-head RMSNorm with a learnable gain (the standard baseline)
  FractionalKeyNorm  the family we tried before finding the constant: keys RMS-normalized then
                     rescaled by a learnable power of their own relative magnitude. alpha=0 is
                     key-only RMSNorm, alpha=1 is a per-head running scale. It works, but a
                     constant does better and costs less; kept because the paper reports it.
"""
from __future__ import annotations

import math
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["KeyScale", "scale_key_projection_", "initial_logit_std", "suggest_key_scale",
           "FractionalKeyNorm", "QKNorm", "ScaledAttention", "patch_nanogpt"]


# --------------------------------------------------------------------------- the fix
class KeyScale(nn.Module):
    """Multiply keys by a fixed per-head constant: the whole intervention.

    k_hat = c_h * k.  With the usual 1/sqrt(head_dim), this scales the attention logits by c_h,
    i.e. it is an inverse temperature set once rather than learned.

    Args:
        n_head:    number of key heads (use n_kv_head for GQA/MQA).
        c:         the multiplier. A float applies to every head; a sequence sets it per head.
        learnable: keep it fixed (default) or let the optimizer move it. Learnable is only
                   worth it when you START near the right value; from c=1 it does not travel.
        layout:    "bthd" for (B, T, H, D) [default] or "bhtd" for (B, H, T, D).

    Zero parameters when learnable=False, one scalar per head when True. The forward is one
    broadcast multiply.
    """

    def __init__(self, n_head: int, c: float | list[float] = 1.0, *, learnable: bool = False,
                 layout: Literal["bthd", "bhtd"] = "bthd"):
        super().__init__()
        assert layout in ("bthd", "bhtd")
        self.n_head, self.layout, self.learnable = n_head, layout, learnable
        c_t = torch.as_tensor(c, dtype=torch.float32)
        if c_t.ndim == 0:
            c_t = c_t.repeat(n_head)
        assert c_t.shape == (n_head,), f"c must be a scalar or length {n_head}, got {tuple(c_t.shape)}"
        assert torch.all(c_t > 0), "c must be positive"
        log_c = c_t.log()
        if learnable:
            self.log_c = nn.Parameter(log_c)
        else:
            self.register_buffer("log_c", log_c)

    @property
    def c(self) -> torch.Tensor:
        return self.log_c.detach().exp()

    def extra_repr(self) -> str:
        c = self.c
        cs = f"{float(c[0]):g}" if bool((c == c[0]).all()) else f"{[round(float(v), 3) for v in c]}"
        return f"n_head={self.n_head}, c={cs}, learnable={self.learnable}, layout={self.layout}"

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        h_axis = 2 if self.layout == "bthd" else 1
        shape = [1, 1, 1, 1]
        shape[h_axis] = self.n_head
        return k * self.log_c.exp().view(*shape).to(k.dtype)


@torch.no_grad()
def scale_key_projection_(qkv_or_k_proj: nn.Linear, factor: float, *, d_model: Optional[int] = None,
                          which: Literal["fused_qkv", "k_only"] = "fused_qkv") -> nn.Linear:
    """Fold the multiplier into the weights, in place, so there is no runtime op at all.

    which="fused_qkv": the layer is a single Linear(d, 3d) laid out [q; k; v] (nanoGPT style);
                       only the key third is scaled. Pass d_model, or it is inferred.
    which="k_only":    the layer is a dedicated key projection; the whole weight is scaled.

    NOTE this is not exactly equivalent to KeyScale during training: enlarging the weights changes
    how weight decay and Adam's per-parameter normalization act on them. In our sweep it recovered
    most, but not all, of the runtime multiplier's benefit. Prefer KeyScale unless you need the
    forward pass untouched.
    """
    W = qkv_or_k_proj.weight
    if which == "k_only":
        W.mul_(factor)
        return qkv_or_k_proj
    d = d_model if d_model is not None else W.shape[0] // 3
    assert W.shape[0] == 3 * d, f"expected a fused [q;k;v] weight of {3 * d} rows, got {W.shape[0]}"
    W[d:2 * d].mul_(factor)
    return qkv_or_k_proj


# --------------------------------------------------------------------------- diagnostics
@torch.no_grad()
def initial_logit_std(q: torch.Tensor, k: torch.Tensor, head_dim: Optional[int] = None,
                      layout: Literal["bthd", "bhtd"] = "bthd") -> float:
    """Standard deviation of the pre-softmax attention logits, the number to look at.

    Feed one batch of queries and keys from an UNTRAINED model. In our runs this was about 0.05
    at head_dim 4 and rose with head width; a value far below the trained value (2 to 3 here)
    means attention starts almost uniform and has to travel a long way.
    """
    if layout == "bthd":
        q, k = q.transpose(1, 2), k.transpose(1, 2)
    d = head_dim or q.shape[-1]
    logits = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(d)
    return float(logits.std())


@torch.no_grad()
def suggest_key_scale(k: torch.Tensor, layout: Literal["bthd", "bhtd"] = "bthd") -> torch.Tensor:
    """A starting point for c: the reciprocal of the per-head key RMS at initialization.

    This is the multiplier that puts keys at unit RMS, i.e. the same average scale an RMSNorm
    would impose, WITHOUT destroying per-token magnitude. It is a starting point for a sweep,
    not an answer: in our runs the empirical optimum was near this value at some head widths
    and up to 2x it at others.
    """
    if layout == "bhtd":
        k = k.transpose(1, 2)
    r = k.float().pow(2).mean(-1).sqrt()          # (B, T, H)
    return 1.0 / r.mean(dim=(0, 1)).clamp(min=1e-9)


# --------------------------------------------------------------------------- reference arms
class QKNorm(nn.Module):
    """Per-head RMSNorm + per-channel gain: the standard QK-norm / key-only-norm operator."""

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


class FractionalKeyNorm(nn.Module):
    """Keys RMS-normalized, then rescaled by a learnable power of their relative magnitude.

        r_t     = RMS_head(k_t)                       (per token, per head; gradient attached)
        s_h     = EMA over training batches of mean_t r_t
        k_hat_t = g * (k_t / r_t) * clamp(r_t / s_h, 1/c, c) ** alpha_h

    alpha = 0 recovers key-only RMSNorm; alpha = 1 leaves a per-head running scale only.
    Reported in the paper for completeness: it beats the unnormalized baseline at every head
    width we tested, but so does a plain constant, which is cheaper and does slightly better.
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
        batch_mean = r_det.float().mean(dim=(0, 1))
        if float(self.ema_ready) == 0.0:
            self.rms_ema.copy_(batch_mean)
            self.ema_ready.fill_(1.0)
        else:
            self.rms_ema.mul_(self.ema_momentum).add_(batch_mean, alpha=1 - self.ema_momentum)

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        if self.layout == "bhtd":
            k = k.transpose(1, 2)
        B, T, H, D = k.shape
        assert H == self.n_head and D == self.head_dim, (k.shape, self.n_head, self.head_dim)
        k32 = k.float()
        r = k32.pow(2).mean(-1).add(1e-12).sqrt()
        k_dir = k32 * torch.rsqrt(k32.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.training:
            self._update_ema(r.detach())
        c = self.ratio_clamp
        ratio = (r / self.rms_ema.view(1, 1, -1)).clamp(1.0 / c, c)
        tau = torch.exp(self.alpha.view(1, 1, -1) * ratio.log())
        out = k_dir * tau.unsqueeze(-1)
        if self.gain is not None:
            out = (out.reshape(B, T, H * D) * self.gain).view(B, T, H, D)
        out = out.to(k.dtype)
        return out.transpose(1, 2) if self.layout == "bhtd" else out


# --------------------------------------------------------------------------- nanoGPT drop-in
class ScaledAttention(nn.Module):
    """nanoGPT-compatible causal self-attention with a selectable key treatment.

    `config` needs n_embd, n_head, block_size, dropout, bias (nanoGPT's GPTConfig).
    key: "none" | "kscale" | "qknorm" | "knorm" | "fkn"
      kscale  the recommended fix: a fixed per-head multiplier (pass c=... in key_kwargs)
      qknorm  RMSNorm both sides;  knorm  RMSNorm keys only;  fkn  FractionalKeyNorm
    """

    def __init__(self, config, key: str = "kscale", key_kwargs: Optional[dict] = None):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert key in ("none", "kscale", "qknorm", "knorm", "fkn")
        self.n_head, self.n_embd = config.n_head, config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout, self.key = config.dropout, key
        kw = dict(key_kwargs or {})
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        if key == "kscale":
            self.k_op = KeyScale(self.n_head, **kw)
        elif key == "qknorm":
            self.q_op = QKNorm(self.n_head, self.head_dim)
            self.k_op = QKNorm(self.n_head, self.head_dim)
        elif key == "knorm":
            self.k_op = QKNorm(self.n_head, self.head_dim)
        elif key == "fkn":
            self.k_op = FractionalKeyNorm(self.n_head, self.head_dim, **kw)
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
        if self.key == "qknorm":
            q = self.q_op(q)
        if self.key != "none":
            k = self.k_op(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
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


def patch_nanogpt(model_module, key: str = "kscale", key_kwargs: Optional[dict] = None):
    """Monkey-patch nanoGPT's `model` module so GPT(config) builds this attention.

        import model as nanogpt_model
        from attnscale import patch_nanogpt
        patch_nanogpt(nanogpt_model, key="kscale", key_kwargs={"c": 8.0})
        gpt = nanogpt_model.GPT(config)
    """
    class _Patched(ScaledAttention):
        def __init__(self, config):
            super().__init__(config, key=key, key_kwargs=key_kwargs)
    model_module.CausalSelfAttention = _Patched
    return model_module
