# Fractional Key Normalization: What QK-Norm Deletes at Tiny Heads, and How to Give It Back

**A head-geometry study of attention normalization from 300+ paired CPU-scale runs, with a drop-in module.**

*daily-research-lab, 2026-09-01. Author: Adrian Cisneros (CisnerosCodes), with Claude as research assistant. Code, data hashes, and every number in this paper live in the repository; every figure is regenerated from `results.json` files by `paper/make_figures.py`.*

## Abstract

QK-normalization (RMS-normalizing queries and keys per head before the dot product) is now a default in production language models, and key-only normalization was recently proposed as a gentler alternative (QUEST, ICLR 2026). We ask a question neither line of work answers: **how does the right normalization depend on head geometry?** Holding parameters, FLOPs, initialization and data stream exactly fixed, we sweep the iso-parameter head split of a 0.42M-parameter character-level transformer (head_dim 4 to 128 at d_model 128) across four normalization arms (none, query-only, key-only, both) with three paired seeds. The answer is a phase diagram, not a rule: at many tiny heads every norm hurts and the damage rides the key side (+0.13 bpc at head_dim 4); at mid splits full QK-norm wins; at wide heads dropping the query norm strictly helps. We then dissect the tiny-head cliff with nine controlled ablations and find that it is a **severed gradient channel**: restoring the per-token key magnitude as a forward value with the gradient projected out pays the full cliff, while a learnable gain without a norm is free. Reopening the channel gives **Fractional Key Normalization (FKN)**: keys are RMS-normalized and then rescaled by a learnable per-head power of their own relative magnitude, so the model chooses how much per-token key scale to keep. Key-only norm is FKN's alpha = 0 endpoint; a per-head running-scale normalization is its alpha = 1 endpoint. At head_dim 4, FKN lands 0.070 bpc *below* the unnormalized baseline (3/3 seeds), where both QK-norm and key-only norm sit 0.13 bpc above it. We report kill tests across the full head split, on a second corpus (character-level Penn Treebank), and under 3x longer training, plus an ablation that freezes the exponent to test whether the win is the running scale alone. We ship FKN as a tested nanoGPT drop-in whose benchmark reproduces the lab's archived numbers to five decimals. Two companion ledgers from the same lab, on plateau escape in gated linear attention and on when weight-tied recursion pays, are summarized with their actionable recipes. All results are early-training, small-scale, CPU-only, and we say so wherever it matters.

## 1. Introduction

Attention normalization has become a default rather than a decision. Gemma 2/3, OLMo 2 and Qwen3 apply an RMSNorm to queries and keys per head; the motivation is stability at scale, and the evidence is mostly "loss spikes went away". Two things are missing from that picture. First, nobody has asked whether the answer depends on the shape of the heads: at a fixed model width, do 32 heads of dimension 4 want the same normalization as 1 head of dimension 128? Second, the field has started to notice that normalization deletes something. QUEST (ICLR 2026) argues for normalizing only the keys so that each query keeps control of its own softmax sharpness; NaLaFormer (2025) re-injects the query norm into linear attention for the same reason. Both state the mechanism; neither tests it causally against the alternatives.

This paper is built from a lab notebook: 61 small experiments run one per night on a CPU, each with a pre-registered hypothesis, a novelty check, paired seeds, and a results file that a later night can replicate bit for bit. Nine of those nights form a single thread on attention normalization, and the thread ends in a place its first night did not predict. We report that thread as a paper, add the kill tests a paper needs, and package the surviving architecture as something you can use this afternoon.

**Contributions.**

1. **A head-geometry phase diagram of attention normalization** (Section 4). At exactly iso-parameter head splits with byte-identical paired initializations and data streams, the best of {no norm, query-only, key-only, QK-norm} changes with head width, and the interaction between the two one-sided norms flips sign from additive at tiny heads to destructive at wide heads.
2. **A causal anatomy of the tiny-head cliff** (Section 5). Static temperature refunds 2% of the cliff, the optimizer leaves the dial at 1, and the loss is not a sharpness cap. The cliff follows the key norm, not the query norm. Restoring the key magnitude *value* with its gradient severed pays the full cliff; a learnable gain with no norm is free. The casualty is the gradient path through the per-token key magnitude.
3. **Fractional Key Normalization** (Section 6), a two-parameter-per-head family that contains key-only RMSNorm and per-head running-scale normalization as endpoints and lets the model pick the exponent. At head_dim 4 it beats the unnormalized baseline by 0.070 bpc in 3 of 3 paired seeds; the kill tests in Section 7 report where else it holds.
4. **A drop-in implementation** (Section 10) for nanoGPT-style attention with unit tests and a benchmark that reproduces the archived anchors to five decimal places, so the claims here are checkable on any laptop.
5. **Two companion ledgers** (Section 9): a training recipe that makes gated linear attention escape the MQAR plateau at step 300 instead of 1100, derived from a chain of ten causal experiments; and twenty evidence-backed rules for when weight-tied recursion pays.

We are explicit about scale: d_model 128, two layers, 600 training steps (1800 in the long-training check), character-level text, three seeds. This is a small-scale proxy study in the lineage of Wortsman et al. (2023). Its value is the control, not the scale.

## 2. Background and related work

**QK-norm.** Henry et al. (2020) L2-normalize queries and keys along the head dimension and replace 1/sqrt(d) with a learnable scalar. Dehghani et al. (2023) used a LayerNorm variant to stabilize a 22B vision transformer; Wortsman et al. (2023) showed at small scale that QK-LayerNorm removes the attention-logit-growth instability and widens the learning-rate basin. Production models (Gemma 2/3, OLMo 2, Qwen3) apply an RMSNorm with a learnable gain to both sides. Our `qknorm` arm is exactly that: per-head RMS normalization with eps 1e-6 and a learnable per-channel gain, followed by the usual 1/sqrt(head_dim).

**Key-only normalization.** QUEST (arXiv:2604.00199, ICLR 2026) proposes constraining keys to a hypersphere while leaving queries free, arguing that this "allows each token to individually control the sharpness of its softmax distribution and prevents large key norms from stealing attention globally". QUEST's own text notes that one-sided normalization had not been proposed before it. We arrived at key-only normalization independently (registry 2026-08-23) and confirm its advantage at wide heads; we do not claim it as new. What we add is the regime map (it loses at tiny and mid splits), the mechanism (what it deletes), and the fractional generalization. Our operator differs from QUEST's: RMSNorm with a learnable per-channel gain rather than a projection to the unit sphere. We could not read QUEST in full from the sandbox (the arXiv host was blocked); its exact operator and ablation table should be checked before this section is finalized for submission.

**Per-token temperature.** Selective Attention (SSA, NeurIPS 2024) learns a data-dependent inverse temperature on the query; Veličković et al. (2024) sharpen softmax at inference with an entropy-derived adaptive temperature; Scalable-Softmax scales logits by a learnable per-head multiple of log n; Gated Attention (Qiu et al., NeurIPS 2025) applies a query-dependent sigmoid gate to the attention output. Our magnitude channel is in this family functionally but differs in three ways that matter for the mechanism claim: it is a parameter-free statistic of the pre-norm vector (its RMS relative to a running per-head average) with a single learned exponent per head, it is applied on the key side, and the paper's contribution is the discriminating experiment (value versus gradient) rather than the capability.

**Head dimension.** Bhojanapalli et al. (2020) explain why loss degrades as head_dim = d/h shrinks at fixed d_model (a low-rank bottleneck on the attention matrix). Multi-Head Attention Residuals (2026) report a U-shaped loss in head count for a different query mechanism. An existing QK-norm x head-count ablation (arXiv:2606.03825) reports stability down to head_dim 16, consistent with our finding that the damage only appears at head_dim 4. We find no prior report that QK-norm converts a monotone head-dim curve into a U-shaped one.

**Evidence standards.** A 2026 update to the Narang et al. transformer-modification study (arXiv:2605.20798) found that most modifications do not transfer at 1 to 3B and insists on multi-seed noise floors. We adopt that standard at our scale: every comparison is paired (same init, same batches), and we report per-seed wins, not just means.

## 3. Setup

**Model.** A two-layer pre-norm decoder-only transformer, d_model 128, FFN 512 (GELU), learned absolute positions, context 96, character vocabulary 65, no biases, untied output head: 423,424 parameters. Attention uses `F.scaled_dot_product_attention` with the default 1/sqrt(head_dim) scale in every arm, so the temperature at initialization varies 5.7x across the head split; this is the confound the thread's second night was built to control.

**Iso-parameter head split.** At fixed d_model the QKV and output projections do not depend on the split, so every (n_head, head_dim) pair with n_head x head_dim = 128 has identical parameters and identical FLOPs. The only differences are how the same vectors are reshaped before the softmax and the 1/sqrt(head_dim) that follows. We use head_dim in {4, 8, 16, 32, 64, 128}.

**Arms.** Let q_t, k_t be the per-head query and key of token t, r(x) the RMS over the head dimension, and g a learnable per-channel gain (length d_model, initialized to 1, excluded from weight decay).

| arm | query | key |
|---|---|---|
| baseline | q | k |
| qknorm | g_q * q / r(q) | g_k * k / r(k) |
| qnorm_only | g_q * q / r(q) | k |
| knorm_only | q | g_k * k / r(k) |
| **knorm_dynk (FKN)** | q | g_k * (k / r(k)) * clamp(r(k) / s_h, 1/8, 8)^alpha_h |
| k_emascale | q | same as FKN with alpha_h frozen at 1 |
| qknorm_dynq | g_q * (q / r(q)) * clamp(r(q) / s_h, 1/8, 8)^alpha_h | g_k * k / r(k) |

Here s_h is a per-head exponential moving average (momentum 0.99) of the batch mean of r, updated only in training mode, and alpha_h is a learnable per-head exponent initialized to 1. In every dynamic arm the gradient flows through r (the "undetached" form); Section 5 shows why that matters.

**Training.** AdamW (betas 0.9/0.95), peak learning rate 3e-3, 60 warmup steps then cosine to 10%, 600 steps of 16 x 96 tokens (0.92M tokens), weight decay 0.1 on matrices only, gradient clip 1.0. Validation is bits per character on 480 contiguous held-out blocks (46,080 characters).

**Paired inits and determinism.** All shared weights are drawn from the same seed before any arm-specific parameter is created; arm-specific extras are initialized to constants and consume no random numbers. The batch stream is a per-seed NumPy generator replayed identically by every arm. A per-seed init signature (sum of absolute weights) is asserted equal across arms in every run. Every night's harness re-runs its parents' anchor cells and demands agreement to 0.0005 bpc; across the thread these agree to five decimals, and the runs in this paper reproduce the 2026-08-30 and 2026-08-31 anchors exactly (Table A1).

**What "better" means.** Differences are reported in bits per character and as paired-seed win counts. The lab's tolerance for "within noise" is 0.015 bpc, chosen as roughly the baseline's seed spread at the mid splits.

## 4. The head-geometry phase diagram

<!-- FIG1 -->

<!-- E1_TABLE -->

<!-- FIG2 -->

Three facts organize the table.

**The unnormalized curve is monotone; QK-norm makes it a U.** Without normalization, validation loss falls monotonically with head_dim (Spearman −1.00 in the 2026-07-26 sweep), a plateau above 64 and a steep tax below 32. With QK-norm, the mid-range tax disappears (−0.14 bpc at head_dim 16, −0.10 at 32) and an interior optimum appears at head_dim 32 that beats every unnormalized split. The head-dim tax is two taxes: a temperature tax in the mid range that normalization removes, and a tiny-head tax that normalization makes worse.

**At tiny heads the damage rides the key side.** At head_dim 4, key-only norm pays essentially the full QK-norm cliff (+0.127 vs +0.130 bpc) while query-only norm is nearly free (+0.024). This is the reversal that redirected the thread: the earlier rescue via a query-side magnitude channel (Section 5) was a compensation route through the surviving side, not the causal side.

**The two one-sided norms interact, and the sign of the interaction flips with width.** Writing the interaction as delta(qknorm) − delta(q-only) − delta(k-only), it is −0.021 and −0.001 at head_dim 4 and 16 (the norms are additive or mildly synergistic), then +0.021, +0.081, +0.074 at 32, 64, 128 (stacking both norms on one wide head is destructive). This is why "norm one side and stop" is the wide-head rule and QK-norm keeps its crown at mid splits.

## 5. Anatomy of the tiny-head cliff

The cliff at head_dim 4 is +0.130 bpc (3 seeds, per-seed +0.130/+0.156/+0.104). Five controlled experiments, all on byte-identical paired inits, locate its cause.

**It is not an average temperature cap (2026-07-31).** Unit-RMS 4-dimensional heads have bounded logits (|logit| ≤ sqrt(4) = 2 at unit gain), and the trained QK-norm heads are indeed flatter (normalized entropy 0.90 vs 0.74). But a free per-head learnable temperature on top of QK-norm refunds 2.1% of the cliff, and the optimizer leaves it at tau ≈ 1.04 (range 0.73 to 1.38) when matching the baseline's logit scale would need tau ≈ 2.6. Low sharpness is a symptom both arms share, not the disease.

**Per-token modulation is half of it (2026-08-02).** A per-token temperature on the query, tau_t = clamp(r_t / s_h, 1/8, 8)^alpha with r_t detached, refunds 54% of the cliff, with a rescue that is nearly identical across the three paired seeds (0.073/0.067/0.070) even though the cliff itself varies 1.5x. Mean sharpness is unchanged (entropy 0.896 vs 0.902); what changes is the per-token spread of the applied temperature (std 0.33), which tracks the pre-norm magnitude spread the baseline had.

**The gradient path is the other half (2026-08-06).** Letting the gradient flow through r_t raises the rescue to 98%: the arm lands within the baseline's own seed spread and beats the paired baseline in 2 of 3 seeds. The control that pins the mechanism is tau_t = r_t undetached with no running average, clamp, or exponent: it refunds 4%. The channel has to be restored in relative, clamped, learnable-exponent form *and* be differentiable.

**The side that matters is the key side (2026-08-30).** The one-sided sweep of Section 4 showed the cliff following the key norm. So the query-side rescue above was compensation: with keys pinned to unit RMS, a learnable per-token query scale can re-encode what the key norm destroyed.

**Value versus gradient (2026-08-31).** Six key-side arms at head_dim 4 (Figure 4). A learnable per-channel gain with no norm is free (+0.002). Freezing the gain inside the norm changes nothing (nogain − knorm_only = −0.001), so the gain is not the cliff. The decisive arm multiplies the normalized key back by its own detached RMS: the forward values are those of the gain-only arm (identical up to a 1e-6 epsilon), yet it pays the full cliff (+0.130), because RMS normalization projects the radial component out of the key gradient and the QKV weights never receive the signal to shape per-token key magnitudes. Reopening that channel with the undetached exponent form on top of key-only norm recovers 155% of the cliff: −0.070 bpc below the unnormalized baseline in 3 of 3 seeds, with a seed spread (0.010) at the baseline's level.

<!-- FIG4 -->

The mechanism statement, then: at tiny heads, per-token key magnitude is how a head expresses token selectivity, and what a norm destroys is not the magnitude's value but the *learnability* of that magnitude through the QKV projection.

<!-- FIG5 -->

## 6. Fractional Key Normalization

**Definition.** For each head h and token t, with r_t = RMS(k_t) computed with the gradient attached and s_h a per-head running mean of r,

k_hat_t = g ⊙ (k_t / r_t) · clamp(r_t / s_h, 1/c, c)^alpha_h,  with c = 8, alpha_h learnable, initialized to 1.

Away from the clamp this is

k_hat_t = g ⊙ k_t / (r_t^(1 − alpha_h) · s_h^alpha_h),

a *fractional* RMS normalization: the key's own magnitude enters with exponent alpha − 1.

- alpha = 0: key-only RMSNorm with gain (the QUEST-like endpoint; the magnitude is deleted).
- alpha = 1: k / s_h, no per-token normalization at all, only a per-head running scale (a batch-statistics normalization of the key scale, like a scale-only BatchNorm on keys).
- alpha in between: the model keeps part of the magnitude.

Queries are untouched. The 1/sqrt(head_dim) scale is unchanged. Cost: one exponent per head, one gain per channel, and a per-head buffer; the forward adds one RMS, one division and one power per key. The clamp bounds the per-token rescale to [1/8, 8]; the theory of normalized attention (Mudarisov et al., 2025) gives the reason a bound is needed, since aggressive per-token scaling trades separability for gradient instability.

**What the model chooses.** The learned exponent is the diagnostic (Figure 3). At head_dim 4 the key-side alpha sits at 0.95 (three seeds: 0.954/0.946/0.956): the model wants almost all of the magnitude back. On the query side of the 2026-08-11 composite, alpha fell monotonically from 0.98 at head_dim 4 to 0.66 at head_dim 128: the channel disengages where wide heads no longer need it. Section 7 reports the key-side curve.

<!-- FIG3 -->

**Why the running scale matters.** With alpha near 1 the forward pass is close to k · g / s_h. Two things distinguish that from the unnormalized baseline: the per-head running scale s_h (an adaptive, non-learned temperature that tracks the key statistics during training) and the tiny residual exponent. The `k_emascale` arm in Section 7 freezes alpha at 1 to ask which of the two carries the win.

**Ordering with rotary embeddings.** RoPE is norm-preserving, so r_t is the same before and after rotation; apply FKN before RoPE so the per-channel gain sees unrotated keys.

## 7. Kill tests

Every earlier "strictly better" candidate in this thread died one head width away from where it was found: the QK-norm + query-channel composite inherited the single-head tax at head_dim 128 (2026-08-11), and key-only norm inherited the cliff at head_dim 4 (2026-08-30). So the test for FKN is the whole curve, a second corpus, and longer training.

### 7.1 The full head split

<!-- E1_VERDICT -->

### 7.2 Is the exponent doing the work, or the running scale?

<!-- E1_EMASCALE -->

### 7.3 A second corpus: character-level Penn Treebank

<!-- E2_SECTION -->

### 7.4 Three times longer training

<!-- E3_SECTION -->

## 8. What we recommend

<!-- RECIPE_TABLE -->

The rule of thumb that survives all the tests we ran: **normalize keys, leave queries alone, and let the key magnitude back in through a learnable exponent.** Where the exponent would sit at zero the model will put it there; where a norm would hurt, the exponent opens.

## 9. Two companion ledgers from the same lab

The attention thread is the deepest in the notebook, but two other threads produced recipes worth acting on. Both are summarized from their registry rows; the full tables are in the repository.

### 9.1 Plateau escape in gated linear attention (MQAR)

Multi-query associative recall (MQAR) is the litmus test that separates softmax attention from sub-quadratic mixers. Ten nights of paired experiments on a 94k-parameter gated linear-attention model at 8 key-value pairs produced a causal chain rather than a leaderboard:

- **A fixed-step "capacity frontier" measures escape time, not capacity** (2026-07-27). The elu+1 model that sits at 0.17 accuracy for 15,000 steps breaks out at 17,500; the same cell at d_model 128 flips from 0.18 to 0.98 on an init change.
- **Only a dense per-channel gate escapes reliably** (2026-07-28). Static, scalar and rank-4 gates are exact no-ops on the plateau; rank-1 is an init coin flip; the dense gate escapes 10 to 20x earlier at identical state size.
- **The gate's win is content routing** (2026-07-29). The identical gate fed another sequence's content is a no-op; noise on the gate logits is a no-op; freezing the gate after breakout costs nothing.
- **The gate is rate-limiting from below, never pacing from above** (2026-08-01, 2026-08-03); **the backbone sets the clock** (2026-08-07); **joint learning-rate scaling escapes at step 400 in every seed** (2026-08-13); **weight decay is a seed-exact no-op on timing** (2026-08-26).
- **Gradient noise is the residual, and it hurts** (2026-09-01). Escape time measured in drift units (step x lr multiplier) collapses onto lr/B within 10%; raising batch size with the learning rate restores the inverse scaling, and batch 256 at 4x learning rate escapes at step 300 in all three seeds. The sign is the opposite of the diffusion-escape folklore: on this plateau, less gradient noise means earlier escape.

<!-- FIG8 -->

**Recipe.** Dense per-channel gate (zero-initialized weight, bias 3.0), AdamW with no warmup or clipping, learning rate 4e-3 on every parameter group, batch 256, weight decay 0.01. Escape at step 300/300/300 versus 1100 for the standard recipe. If your budget is wall-clock on a CPU rather than steps, batch 16 at 4x learning rate escapes in fewer seconds; the lr/B law says the two knobs trade off exactly. And the honest comparison outside the gate family: a Taylor-exp (BASED-style) feature map with no gate at all solves the same cell by step 500 at the ordinary learning rate.

### 9.2 When weight-tied recursion pays

The lab's flagship idea ("Shadow": a tiny model whose every choice is earned through an ablation) began with a weight-tied looped block and a falsification target. Twelve experiments later the ledger reads:

- **On language-model loss at iso-FLOP the loop loses, and depth itself is nearly flat** at 0.06 to 0.2M parameters (+0.079 bpc for the k=4 loop vs untied depth; untied k=4 vs k=1: −0.004).
- **Entropy-based early exit is indistinguishable from a coin flip** at matched compute, because the fixed-depth quality curve it would exploit is flat.
- **The loop earns test-time compute only under a stochastic depth schedule** (train with k ~ U{1..K}): frontier accuracy 0.85 at 2.7x the trained depth versus 0.55 for fixed-K training, and an 11x gain in full-sequence exact match; untied depth cannot be extended by any hack (cycling blocks leaves the outputs bit-identical).
- **Supervising the intermediate state beats every unsupervised extra-compute mechanism**: routing the state through the answer space scales with depth (0.22 to 0.51 solve rate on 4x4 Sudoku) while a full-width latent is flat (0.22 to 0.25); three tokens of discrete supervision reach 1.000 at fewer FLOPs than four pause tokens.
- **Trained halting learns real difficulty** (+0.15 over a compute-matched random exit) but under-spends out of distribution (allocation slope 0.68 where 1.0 is required); uniform stochastic depth holds the ceiling at 1.000 exact match at 5x the training length.
- **The gains are serial, not hierarchical**: with sequence length controlled, no arm's optimal loop count grows with parse depth on ListOps.

<!-- FIG9 -->

**Rule.** Tie the block, train the depth stochastically with per-iteration input injection, or do not loop at all; test a halting rule only on a task whose fixed-depth curve is steep; and spend your supervision on the intermediate state before you spend it on latent compute.

## 10. What you can do today

Three things are possible now that were not possible before this work, in the sense that each was a claim in a notebook rather than a tool.

**1. Drop FKN into nanoGPT with one line.** The `fkn/` package provides `FractionalKeyNorm` (works on (B, T, H, D) or (B, H, T, D) keys), `FKNCausalSelfAttention` (a nanoGPT `CausalSelfAttention` replacement that takes the same config and a `norm` argument in {none, qknorm, knorm, fkn}), and `patch_nanogpt(model_module)`. Nine unit tests check the math against the lab harness, the alpha = 0 and alpha = 1 endpoints, gradient flow through the magnitude, EMA behaviour in train and eval modes, both tensor layouts, autocast, and determinism.

```python
from fkn import FractionalKeyNorm
knorm = FractionalKeyNorm(n_head=8, head_dim=64)   # alpha init 1, learnable; clamp 8; EMA 0.99
k = knorm(k)                                       # k: (B, T, n_head, head_dim); queries untouched
```

**2. Run the head-split benchmark on your own corpus.** `python -m fkn.bench --text your.txt --norm fkn --head-dim 32` trains the lab's two-layer recipe through the drop-in module and prints validation bits per character and the learned exponents. On tiny-shakespeare it reproduces the archived anchors exactly (baseline 3.09304, FKN 3.0167 at head_dim 4, seed 0), so any number you get is on the same scale as every number in this paper.

**3. Use the MQAR recipe.** If you train a gated linear-attention model on recall and it sits on a plateau, the fix is not a new gate but a joint learning-rate and batch-size raise (Section 9.1). The harness for that recipe is `experiments/2026-09-01_mqar-escape-noise-vs-batch`.

Beyond those, the thread leaves a scaling question that a GPU can answer in an afternoon and this lab cannot: does the FKN margin at head_dim 4 to 8 survive at d_model 512 to 1024 with 16 to 32 heads of dimension 32 to 64, the regime production models actually use? The bench script takes `--d-model` and `--head-dim` and the module is autocast-safe; that experiment is the natural next step.

## 11. Limitations and threats to validity

- **Scale.** One architecture (two layers, d_model 128), 0.42M parameters, 600 steps (all arms sit at 2.8 to 3.2 bpc where a converged character model reaches about 1.5), one learning rate shared by all arms. A mechanism that is merely slower to optimize looks exactly like a loser here; the 1800-step check narrows but does not close this gap.
- **Head splits are iso-parameter but not iso-temperature.** The 1/sqrt(head_dim) scale is inherited from the baseline in every arm. No arm removes or learns it.
- **Three seeds.** Every headline claim is a 3-of-3 paired win with the paired difference outside the baseline's seed spread; effects smaller than 0.015 bpc are reported as ties. Eight to ten seeds would be needed to resolve the variance-inflation observations at the single-head split.
- **Corpora.** Character-level text only; two corpora. Tokenized language models, vision and long-context regimes are untested.
- **Prior art.** The arXiv host was unreachable from the sandbox, so QUEST and several 2026 papers are cited from search summaries; their exact operators and ablation tables should be verified before submission.
- **The running scale is a per-process buffer.** Under data-parallel training each rank keeps its own estimate; the module documents this.

## 12. Ideas the results generate

Each item is an observation from the ledger, an inference, and the experiment that would settle it. They are ordered by expected information per CPU-minute.

1. **Observation:** at head_dim 4 the learned key exponent is 0.95. **Inference:** the win may be the per-head running scale, not per-token magnitude. **Experiment:** the `k_emascale` arm in Section 7.2 is this test; if it matches FKN, the architecture simplifies to "divide keys by a per-head EMA of their RMS" and the story changes from selectivity to adaptive temperature.
2. **Observation:** the interaction between the two one-sided norms turns destructive above head_dim 32. **Inference:** two full-width gains stack too many sharpness dials on one softmax. **Experiment:** logit-scale-matched arms (hold the trained pre-softmax logit std fixed across arms) would turn the over-sharpening account from correlational to causal.
3. **Observation:** the q-side channel disengages with width (alpha 0.98 to 0.66). **Inference:** alpha is a learned estimate of how much magnitude a head needs. **Experiment:** initialize alpha at 0.5 and at 0 and check whether it converges to the same width-dependent curve; if it does, alpha is a property of the head geometry, not of the init.
4. **Observation:** a static per-head temperature refunds 2% of the cliff and the optimizer refuses to use it. **Inference:** the loss landscape near tau = 1 is flat in tau but not in per-token scale. **Experiment:** measure the Hessian diagonal for tau versus alpha at init.
5. **Observation:** in MQAR, escape drift-time collapses onto lr/B and noise hurts. **Inference:** the plateau is a saddle whose escape direction is a low-signal, high-noise gradient. **Experiment:** the backlog's exponent fit (eval every 25 steps, grid over lr and B) distinguishes lr/B from lr^2/B and sizes the curvature residual.
6. **Observation:** the Taylor feature map escapes with no gate at all. **Inference:** the gate is compensating for a feature-map deficiency. **Experiment:** dense gate on top of the Taylor map; if escape does not move, gating and feature-map expressivity are substitutes, not complements.
7. **Observation:** stochastic depth extrapolates on serial tasks and hurts on hierarchical ones. **Inference:** the useful inductive bias is depth-invariance, and hierarchy needs supervision of subtree values. **Experiment:** answer-space refinement (the Sudoku routing) on ListOps with intermediate subtree supervision; the prediction is that iteration t then locks to nesting level t.
8. **Observation:** the never-crossed cell in the recursion ledger is stochastic depth x iso-FLOP language-model loss. **Experiment:** rerun the first Shadow experiment with k ~ U{1..4}, input injection and a per-loop norm; this is the cheapest way to learn whether the loop's language-model loss was a training-schedule artifact.
9. **Observation:** FKN is defined on keys; the symmetric query-side version on top of query-only norm is untested. **Experiment:** `qnorm_dynq` at head_dim 4; the 155% key-side overshoot predicts more than 100% if the mechanism is side-symmetric, and less if the key side is special.
10. **Observation:** every attention-norm result is at d_model 128. **Experiment:** the width-scaling check named in the backlog (d_model 64 and 256): does the QK-norm optimum stay at head_dim 32, or track sqrt(d)?

## References

- Henry, A., Dachapally, P. R., Pawar, S., Chen, Y. Query-Key Normalization for Transformers. Findings of EMNLP 2020. arXiv:2010.04245.
- Dehghani, M. et al. Scaling Vision Transformers to 22 Billion Parameters. 2023.
- Wortsman, M. et al. Small-scale proxies for large-scale Transformer training instabilities. ICLR 2024. arXiv:2309.14322.
- QUEST: A robust attention formulation using query-modulated spherical attention. ICLR 2026. arXiv:2604.00199.
- Meng, Z. et al. Norm x Direction: Restoring the Missing Query Norm in Vision Linear Attention (NaLaFormer). arXiv:2506.21137.
- Selective Attention: Enhancing Transformer through Principled Context Control. NeurIPS 2024. arXiv:2411.12892.
- Veličković, P., Perivolaropoulos, C., Barbero, F., Pascanu, R. Softmax is not Enough (for Sharp Size Generalisation). arXiv:2410.01104.
- Nakanishi, K. M. Scalable-Softmax Is Superior for Attention. arXiv:2501.19399.
- Qiu, Z. et al. Gated Attention for Large Language Models. NeurIPS 2025. arXiv:2505.06708.
- Mudarisov, T. et al. Limitations of Normalization in Attention Mechanism. arXiv:2508.17821.
- Bhojanapalli, S. et al. Low-Rank Bottleneck in Multi-head Attention Models. ICML 2020. arXiv:2002.07028.
- Luo, C., Cai, Z., Hu, J. Multi-Head Attention Residuals. arXiv:2607.27230.
- Most Transformer Modifications Still Do Not Transfer at 1-3B. arXiv:2605.20798.
- Loshchilov, I. et al. nGPT: Normalized Transformer with Representation Learning on the Hypersphere. arXiv:2410.01131.
- Kimi Team. Kimi K2: Open Agentic Intelligence (MuonClip / QK-Clip). arXiv:2507.20534.
- Arora, S. et al. Zoology: Measuring and Improving Recall in Efficient Language Models; and BASED (Taylor-exp linear attention). 2023-2024.
- Geiping, J. et al. Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach. arXiv:2502.05171.
- Hao, S. et al. Training Large Language Models to Reason in a Continuous Latent Space (Coconut). 2024.
- Jolicoeur-Martineau, A. Less is More: Recursive Reasoning with Tiny Networks (TRM). arXiv:2510.04871.
- Xie, Z., Sato, I., Sugiyama, M. A Diffusion Theory for Deep Learning Dynamics. arXiv:2002.03495.

## Appendix A. Replication anchors

<!-- ANCHOR_TABLE -->

## Appendix B. The attention-norm ledger

| night | registry id | claim it established | key number |
|---|---|---|---|
| 07-26 | head-dim-vs-count-isoparam | no U-curve without norm; monotone in head_dim | Spearman −1.00; +0.217 bpc at hd 4 |
| 07-30 | qknorm-head-dim | QK-norm makes a U with optimum hd 32 and deepens the hd 4 cliff | −0.099 at 32; +0.143 at 4 |
| 07-31 | qknorm-hd4-temperature-rescue | static per-head temperature refunds 2%; optimizer leaves tau at 1 | rescue 0.021 |
| 08-02 | qknorm-hd4-dynamic-temperature | detached per-token query temperature refunds 54% | rescue 0.536 |
| 08-06 | qknorm-hd4-undetached-magnitude | gradient through r_t refunds 98%; raw r_t refunds 4% | rescue 0.977 / 0.035 |
| 08-11 | qknorm-dyntemp-composite-sweep | composite dominates QK-norm but inherits the hd 128 tax; alpha disengages with width | alpha 0.98 to 0.66 |
| 08-23 | qknorm-nh1-tax-mechanism | at one head, each one-sided norm wins and both together lose | k-only −0.041; interaction +0.074 |
| 08-30 | knorm-only-head-sweep | the cliff follows the key norm; interaction flips sign with width | k-only +0.127 at hd 4, −0.069 at hd 64 |
| 08-31 | hd4-kside-cliff-mechanism | the cliff is a severed magnitude gradient; FKN lands below baseline | magrestore +0.130; FKN −0.070 |
| 09-01 | knorm-dynk-head-sweep | full head split kill test, plus the alpha-fixed ablation | see Section 7 |
| 09-01 | knorm-dynk-ptb-transfer | second corpus | see Section 7 |
| 09-01 | knorm-dynk-longer-training | 3x training | see Section 7 |

## Appendix C. Reproduction

```
pip install torch==2.13.0 numpy matplotlib pyyaml      # the lab's runs used torch 2.13.0+cu130 on CPU
python experiments/2026-09-01_knorm-dynk-head-sweep/run.py       # ~2 h on 2 CPU threads (or shard: --seeds / --head-dims / --tag, then --merge)
python paper/make_figures.py                                       # regenerates every figure from results.json
python fkn/test_fkn.py                                             # 9 unit tests
python -m fkn.bench --text fkn/data/tinyshakespeare.txt --norm fkn --head-dim 4 --seed 0 --warmup 60   # 3.0167
```
