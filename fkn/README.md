# fkn: Fractional Key Normalization

A drop-in attention normalization for nanoGPT-style transformers, distilled from the
daily-research-lab attention-norm thread (2026-07-26 to 2026-09-01). Paper: `../paper/`.

## What it is

QK-norm RMS-normalizes queries and keys per head. Key-only norm (QUEST, ICLR 2026) normalizes
only the keys. FKN normalizes only the keys, *fractionally*: after the RMS-norm, each key is
rescaled by a learnable per-head power of its own magnitude relative to a running per-head scale.

```
r_t   = RMS_head(k_t)                                # per token, per head, gradient attached
s_h   = EMA over batches of mean_t r_t               # per head, no gradient
k_hat = g * (k_t / r_t) * clamp(r_t / s_h, 1/c, c) ** alpha_h        (c = 8, alpha init 1, learnable)
```

Unclamped: `k_hat = g * k_t / (r_t ** (1 - alpha) * s_h ** alpha)`.
`alpha = 0` is key-only RMSNorm, `alpha = 1` is "keys divided by a per-head running scale",
and the model learns where to sit. Queries are untouched.

## Why

At tiny heads (head_dim 4, 32 heads at d_model 128) both QK-norm and key-only norm pay a
+0.13 bpc cliff versus no normalization. The lab showed the cliff is a severed *gradient*
channel through the per-token key magnitude (restoring the magnitude value with the gradient
projected out pays the full cliff; a gain without a norm is free). FKN reopens that channel and
lands 0.070 bpc *below* the unnormalized baseline at head_dim 4 (3/3 paired seeds). See the
paper for the full head-width phase diagram, the second-corpus and longer-training checks, and
the ablation that freezes the exponent.

## Install / use

No packaging needed; copy `fkn/` into your project (PyTorch >= 2.0).

```python
from fkn import FractionalKeyNorm, FKNCausalSelfAttention, patch_nanogpt

# 1) inside your own attention
knorm = FractionalKeyNorm(n_head=8, head_dim=64)        # layout="bthd" (B,T,H,D) default; "bhtd" also supported
k = knorm(k)                                            # then softmax(q k^T / sqrt(d)) v as usual

# 2) nanoGPT: same config fields (n_embd, n_head, block_size, dropout, bias)
attn = FKNCausalSelfAttention(config, norm="fkn")       # norm in {"none", "qknorm", "knorm", "fkn"}

# 3) or patch nanoGPT's model module before building GPT(config)
import model as nanogpt_model
patch_nanogpt(nanogpt_model, norm="fkn")
```

Notes: apply before RoPE; the EMA buffer updates only in `train()` mode and is per process
under data parallelism; `r_t` is computed in float32 under autocast.

## Check it

```
python fkn/test_fkn.py                     # 9 tests: harness math, endpoints, gradient flow, EMA, layouts, autocast, determinism
python -m fkn.bench --text fkn/data/tinyshakespeare.txt --norm none --head-dim 4 --seed 0 --warmup 60   # 3.09304 bpc
python -m fkn.bench --text fkn/data/tinyshakespeare.txt --norm fkn  --head-dim 4 --seed 0 --warmup 60   # 3.0167  bpc
```

Both numbers match the lab's archived runs (registry 2026-08-30 and 2026-08-31) to five decimals,
so `bench` puts your corpus on the same scale as every number in the paper:

```
python -m fkn.bench --text my_corpus.txt --norm fkn --head-dim 32 --d-model 256 --steps 2000 --json out.json
```

## Recommended settings

| head_dim (at d_model 128) | recommendation | evidence (registry) |
|---|---|---|
| 4 | FKN (alpha init 1, learnable); never plain QK-norm or key-only norm | 2026-08-31, 2026-09-01 |
| 16 to 32 | see the paper's Section 7 table (kill test) | 2026-09-01 |
| 64 to 128 | key-only norm family (FKN or plain) beats QK-norm | 2026-08-30, 2026-09-01 |

Defaults: `alpha_init=1.0`, `alpha_learnable=True`, `ratio_clamp=8.0`, `ema_momentum=0.99`, per-channel gain on.
