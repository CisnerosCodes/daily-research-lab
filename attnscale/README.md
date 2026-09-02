# attnscale

Diagnose and fix the **attention logit scale at initialization**. Distilled from the
daily-research-lab attention-normalization thread (2026-07-26 to 2026-09-02). Paper: [`../paper/`](../paper/).

## The finding

At small head dimension the standard `1/sqrt(head_dim)` leaves attention logits far too small at
initialization: in our 0.42M-parameter character models the initial logit standard deviation was
**0.05** at head_dim 4, against a trained value around 3. Attention starts almost uniform and has a
long way to travel.

A **single fixed per-head multiplier on the keys** fixes this, and it beats every normalization
scheme we tried, including QK-norm and key-only norm. Three results make the case:

- The best multiplier **moves with head width**, and 1 (the default) is far from it at small head_dim.
- The same dial made **learnable from 1 does not travel there** within a normal budget. This is why
  a learnable attention temperature looks useless if you initialize it at 1 — a result we ourselves
  reported before understanding it.
- At a **matched initial logit scale**, a constant beats an RMSNorm by a wide margin, because the
  normalizer also destroys per-token key magnitude, which carries information.

So the normalizer was doing two things with opposite signs: a helpful mean-scale correction and a
harmful deletion. Do the first without the second.

## Use it

Copy `attnscale/` into your project (PyTorch >= 2.0). No dependencies beyond torch.

**1. Diagnose.** On an untrained model, look at the scale you start from:

```python
from attnscale import initial_logit_std, suggest_key_scale
print(initial_logit_std(q, k, head_dim))   # q, k: (B, T, n_head, head_dim)
print(suggest_key_scale(k))                 # a per-head starting point for the sweep
```

**2. Sweep the constant for your configuration.** Do not copy our numbers; the right value depends
on your initialization scheme, head width and corpus. A few CPU-minutes:

```
python -m attnscale.bench --text my_corpus.txt --head-dim 16 --sweep-c 1 2 4 8 16
```

**3. Apply it.** Zero parameters, one broadcast multiply:

```python
from attnscale import KeyScale
kscale = KeyScale(n_head=8, c=8.0)          # or c=[...] per head
k = kscale(k)                                # then softmax(q k^T / sqrt(d)) v as usual
```

Or fold it into the weights so the forward pass is untouched (slightly weaker in our sweep, because
enlarging the weights changes how weight decay and Adam act on them):

```python
from attnscale import scale_key_projection_
scale_key_projection_(block.attn.c_attn, 8.0, d_model=768)   # fused [q;k;v] Linear
```

nanoGPT drop-in, same config fields:

```python
from attnscale import ScaledAttention, patch_nanogpt
attn = ScaledAttention(config, key="kscale", key_kwargs={"c": 8.0})
# or, before building the model:
import model as nanogpt_model
patch_nanogpt(nanogpt_model, key="kscale", key_kwargs={"c": 8.0})
```

`key` accepts `none`, `kscale`, `qknorm`, `knorm`, `fkn`, so the paper's arms are all reachable.

## Also included

`FractionalKeyNorm` is the architecture we built before finding the constant: keys RMS-normalized,
then rescaled by a learnable per-head power of their own relative magnitude. `alpha = 0` recovers
key-only RMSNorm and `alpha = 1` leaves a per-head running scale. It does beat the unnormalized
baseline at every head width we tested — but a constant does better, costs less, and needs no
running statistic. It is kept because the paper reports it and because the family is a clean way to
interpolate between the two endpoints.

`QKNorm` is the standard operator, for reference and for reproducing baselines.

## Check it

```
python attnscale/test_attnscale.py     # 13 tests: constant multiplier, layouts, init folding,
                                       # diagnostics, the fractional family's two endpoints,
                                       # gradient flow, EMA behaviour, autocast, determinism
python -m attnscale.bench --text attnscale/data/tinyshakespeare.txt --key none --head-dim 4 --seed 0 --warmup 60
```

The last command reproduces the lab's archived unnormalized baseline (3.09304 bpc) to five decimals,
so any number you measure with this tool is on the same scale as every number in the paper.

## Caveats worth reading

- Measured at d_model 128, two layers, 600 steps, character-level text, three paired seeds. The
  effects shrink by a factor of two to five at 3x the training budget, though the sign holds at
  small head_dim.
- The argument is about initialization scale, which is a property of the parameterization rather
  than the corpus, so we expect it to carry to larger models — but we have not tested that, and the
  sweep is cheap enough that you should just run it.
- If your model already uses QK-norm and is training stably at scale, this result does not say to
  remove it. It says the scale correction and the normalization are separable, and that at small
  head dimension they pull in opposite directions.
