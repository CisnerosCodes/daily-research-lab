# Set the Temperature, Not the Norm

**What a nine-night attention-normalization thread was actually measuring: the logit scale you start from.**

*daily-research-lab, 2026-09-02. Adrian Cisneros (CisnerosCodes), with Claude as research assistant. Every number here is read out of a `results.json` in this repository by `paper/fill_sections.py`; every figure is regenerated from those files by `paper/make_figures.py`.*

## Abstract

QK-normalization is a default in production language models, and key-only normalization was recently proposed as a gentler alternative. We asked how the right choice depends on head geometry, and swept the iso-parameter head split of a 0.42M-parameter character-level transformer at exactly matched parameters, FLOPs, initializations and data streams. Normalization's effect turns out to be strongly head-width dependent: at 32 heads of dimension 4 every normalizer costs about +0.13 bits per character, and the damage rides the key side, while at mid widths it pays −0.14. Nine controlled ablations then locate the tiny-head cost in a severed gradient: restoring the per-token key magnitude as a forward value with its gradient projected out pays the full cost, while a learnable gain with no normalizer is free. That result stands. The architecture we built on it does not. Freezing the repair's learnable exponent changes nothing, slowing its running statistic *helps* monotonically, and applying it to queries instead of keys works equally well — three signs that the mechanism is not adaptive and not key-specific. Following them, we find the whole effect is a **constant**: multiplying keys by a fixed per-head scalar, chosen once, beats every normalization arm in the thread. The best constant moves with head width, the default of 1 is badly wrong at small head dimension, and the same dial made *learnable from 1* does not travel there, which reconciles a two-percent rescue reported earlier with a large one now. The intervention folds into the key projection's initialization, so it costs no parameters and no runtime. We verify the head-width map on a second corpus and under 3x longer training, and report where it fails: the wide-head ordering is corpus- and budget-specific and dissolves with more training. We ship the one-line fix, a benchmark that reproduces our archived numbers to five decimals, and two companion recipes from the same lab. All results are small-scale, early-training and CPU-only.

## 1. Introduction

Attention normalization has become a default rather than a decision. Gemma 2/3, OLMo 2 and Qwen3 apply an RMSNorm to queries and keys per head; the stated motivation is stability, and the usual evidence is that loss spikes went away. Two questions are missing from that picture. Does the right answer depend on the shape of the heads, and what exactly does a normalizer take away when it helps or hurts?

This paper comes out of a lab notebook: sixty-odd small experiments, one per night on a CPU, each with a hypothesis registered before the run, paired seeds, and a results file that a later night replicates bit for bit. Nine of those nights form one thread on attention normalization. The thread proposed an architecture, and then five kill tests refuted it and pointed somewhere simpler and better. We report both halves, because the refutation is the result.

**What we found, in order.**

1. **Normalization's effect depends on head geometry** (Section 4). At exactly iso-parameter head splits, the best of no-norm, query-only, key-only and both changes with head width, and the interaction between the two one-sided norms flips sign from roughly additive at many tiny heads to destructive at one wide head.
2. **The tiny-head cost is a severed gradient, not a lost value** (Section 5). Restoring the per-token key magnitude in the forward pass with its gradient projected out pays the full cost; a learnable gain without a normalizer is free.
3. **The repair we built on that is over-engineered** (Section 6). Its learnable exponent buys nothing, its running statistic is better the less it adapts, and it works the same on queries. Each of these is a paired, three-seed result.
4. **The mechanism is a constant, and it is a temperature** (Section 7). A fixed per-head multiplier on the keys beats every normalization arm in the thread. The best multiplier moves with head width; at small head dimension the default is several times too small. The same dial made learnable from the default does not travel to the optimum within the budget, which is why an earlier experiment concluded that temperature was not the problem.
5. **It folds into the initialization** (Section 7.3). Scaling the key projection's initial weights reproduces the runtime multiplier, so the fix costs no parameters, no runtime, and one line.
6. **Where it holds and where it does not** (Section 8). The tiny-head result transfers to a second corpus and survives 3x training with its sign intact and its magnitude halved. The wide-head ordering does neither.

We are explicit about scale throughout: d_model 128, two layers, 600 steps for the main grid and 1800 for the training check, character-level text, three paired seeds. This is a small-scale proxy study in the lineage of Wortsman et al. (2023). Its value is the control, not the scale, and Section 11 says plainly which claims we expect to survive scaling and which we do not.

## 2. Background and related work

**QK-norm.** Henry et al. (2020) L2-normalize queries and keys along the head dimension and replace 1/sqrt(d) with a learnable scalar. Dehghani et al. (2023) used a LayerNorm variant to stabilize a 22B vision transformer; Wortsman et al. (2023) showed at small scale that QK-LayerNorm removes attention-logit growth and widens the usable learning-rate range. Production models apply an RMSNorm with a learnable gain to both sides, which is exactly our `qknorm` arm.

**Key-only normalization.** QUEST (arXiv:2604.00199, ICLR 2026) constrains keys to a hypersphere while leaving queries free, arguing that this lets each token control the sharpness of its own softmax and stops large-norm keys from taking attention globally. Its own text states that one-sided normalization had not been proposed before it. We reached key-only normalization independently (registry 2026-08-23) and do not claim it as new; what we add is the head-width map, the mechanism, and the finding that a constant does better than either. Our operator also differs, being an RMSNorm with a learnable per-channel gain rather than a projection to the sphere. We could not read QUEST from this sandbox because the arXiv host is blocked here, so its exact operator and ablation table should be checked before this section is submitted anywhere.

**Normalization that keeps the magnitude.** SeeDNorm (Cai et al., 2025) starts from the observation this paper ends on: RMSNorm discards the input norm in the forward pass and a static gain cannot recover it. Its fix is a data-dependent self-rescaling coefficient. NaLaFormer (Meng et al., 2025) re-injects the query norm into linear attention for the same reason. ScaleNorm (Nguyen and Salazar, 2019) goes the other way, replacing LayerNorm with a single learned length per layer. Our result sits underneath all three: before asking what a normalizer should preserve, check what scale the logits started at.

**Attention temperature.** Selective Attention (NeurIPS 2024) learns a data-dependent inverse temperature on the query. Veličković et al. (2024) sharpen softmax at inference from its entropy. Scalable-Softmax scales logits by a learnable per-head multiple of log n. Gated Attention (Qiu et al., NeurIPS 2025) applies a query-dependent sigmoid gate to the attention output. Our finding is not a new temperature mechanism; it is that the *initial value* of the simplest possible temperature is the binding constraint at small head dimension, and that learnability does not substitute for setting it.

**Initialization and scale.** The 1/sqrt(head_dim) factor is the standard fix for logit growth with head dimension, derived for queries and keys with unit-variance independent entries. Real queries and keys are neither unit-variance nor independent, since both are linear images of the same normalized residual stream. Our Section 7 measures what the scale actually is. This connects to muP (Yang et al.) and to the lab's own muP replication (registry 2026-07-26), which found that most of the practical benefit of muP at these sizes is tunability rather than loss.

**Evidence standards.** A 2026 update to the Narang et al. transformer-modification study (arXiv:2605.20798) found that most modifications do not transfer at 1 to 3B and requires multi-seed noise floors. We adopt that standard at our scale: every comparison is paired on byte-identical initializations and data streams, and we report per-seed win counts alongside means, since a paired difference is the correct noise test for this design.

## 3. Setup

**Model.** Two-layer pre-norm decoder-only transformer, d_model 128, FFN 512 with GELU, learned absolute positions, context 96, character vocabulary 65, no biases, untied output head: 423,424 parameters. Attention uses `F.scaled_dot_product_attention` with its default 1/sqrt(head_dim) scale in every arm.

**Iso-parameter head splits.** At fixed d_model the QKV and output projections do not depend on the split, so every (n_head, head_dim) pair with n_head x head_dim = 128 has identical parameters and identical FLOPs. The only differences are how the same vectors are reshaped before the softmax and the 1/sqrt(head_dim) that follows. We use head_dim in {4, 8, 16, 32, 64, 128}; head_dim 8 had never been run with any normalization arm before this paper.

**Arms.** Let q_t and k_t be a token's per-head query and key, r(x) the root-mean-square over the head dimension, and g a learnable per-channel gain of length d_model, initialized to one and excluded from weight decay.

| arm | query | key |
|---|---|---|
| baseline | q | k |
| qknorm | g_q · q / r(q) | g_k · k / r(k) |
| qnorm_only | g_q · q / r(q) | k |
| knorm_only | q | g_k · k / r(k) |
| magnitude channel (key) | q | g_k · (k / r(k)) · clamp(r(k) / s_h, 1/8, 8)^alpha_h |
| frozen exponent | q | the same with alpha_h fixed at 1 |
| fixed multiplier | q | c_h · k, with c_h a constant |

Here s_h is a per-head running mean of r updated only in training mode, and alpha_h is a learnable per-head exponent initialized to 1. Section 6 dismantles this family and Section 7 replaces it.

**Training.** AdamW with betas 0.9 and 0.95, peak learning rate 3e-3, 60 warmup steps then cosine to a tenth, 600 steps of 16 sequences by 96 tokens, weight decay 0.1 on matrices only, gradient clipping at 1.0. Validation is bits per character over 480 contiguous held-out blocks, 46,080 characters.

**Paired initializations.** All shared weights are drawn from one seed before any arm-specific parameter exists; arm-specific extras are constants that consume no random numbers. The batch stream is a per-seed generator replayed identically by every arm. A per-seed initialization signature is asserted equal across arms in every run. Every night re-runs its parents' anchor cells and requires agreement to 0.0005 bpc.

**What "better" means.** Differences are in bits per character. Because the design is paired, the noise test we report is the sign of the per-seed difference, not the overlap of per-arm spreads.

## 4. Normalization depends on head geometry

<!-- FIG1 -->

<!-- E1_TABLE -->

<!-- FIG2 -->

Three facts organize the table.

**Without normalization the curve is monotone in head width; with it, it is not.** Validation loss falls monotonically as heads get wider, a plateau above 64 and a steep tax below 32. Adding QK-norm removes most of the mid-range tax and introduces an interior optimum, while making the smallest split distinctly worse.

**At tiny heads the damage rides the key side.** At head_dim 4, key-only normalization pays essentially the whole two-sided cost while query-only normalization is nearly free. This is the reversal that redirected the thread: an earlier repair through a query-side magnitude channel was compensation through the surviving side, not the causal side.

**The two one-sided norms interact, and the sign of the interaction flips with width.** Writing the interaction as the two-sided delta minus the two one-sided deltas, it is near zero or slightly favourable at head_dim 4 and 16 and clearly destructive at 64 and 128. Stacking two normalizers on one wide head is the pathology; at many narrow heads they are close to additive.

## 5. The tiny-head cost is a severed gradient

Five controlled experiments, all on byte-identical paired initializations, locate the cause of the cost at head_dim 4. This section is unchanged by the later results and we still believe it.

**It is not an average sharpness cap** (registry 2026-07-31). Unit-RMS four-dimensional heads have bounded logits, and the normalized heads are measurably flatter. But a free per-head learnable temperature on top of QK-norm refunds two percent of the cost, and the optimizer leaves the dial essentially at one when matching the baseline's logit scale would need roughly 2.6. Section 7 explains why, and the explanation is the paper's main result.

**Per-token modulation is half of it** (2026-08-02). A per-token temperature on the query, computed from the query's own magnitude relative to a running average, refunds 54 percent, with a rescue nearly identical across paired seeds even though the underlying cost varies by half. Mean sharpness barely moves; what moves is the per-token spread.

**The gradient path is the other half** (2026-08-06). Letting the gradient flow through the magnitude raises the rescue to 98 percent. The control that pins it is the raw undetached magnitude with no running average, clamp or exponent, which refunds four percent.

**The causal side is the key side** (2026-08-30). The one-sided sweep of Section 4 shows the cost following the key norm, so the query-side rescue was compensation.

**Value versus gradient** (2026-08-31). Six key-side arms at head_dim 4. A learnable per-channel gain with no normalizer is free. Freezing the gain inside the normalizer changes nothing, so the gain is not the cause. The decisive arm multiplies the normalized key back by its own detached magnitude: its forward values equal the gain-only arm's to within a 1e-6 epsilon, yet it pays the full cost, because RMS normalization projects the radial component out of the key gradient and the projection weights never receive the signal to shape per-token key magnitudes.

<!-- FIG4 -->

The mechanism statement is therefore: at tiny heads what a normalizer destroys is not the magnitude's value but the *learnability* of that magnitude through the projection.

## 6. Three signs the repair was over-engineered

Reopening that gradient channel gave an architecture: normalize keys, then rescale each key by a learnable per-head power of its own magnitude relative to a running per-head scale. At head_dim 4 it landed 0.070 bpc below the unnormalized baseline in three of three seeds, where both QK-norm and key-only normalization sit 0.13 above. We then ran the tests that would kill it. All three came back against the design, and each is a paired three-seed comparison.

**The learnable exponent buys nothing.** Freezing the exponent at one changes the result by between +0.001 and −0.007 bpc across six head widths, inside the seed spread everywhere. The model does move the exponent when free to, and it moves it further at wider heads, but the movement does not pay.

<!-- E1_ALPHA -->

<!-- FIG3 -->

**Adaptation makes it worse, monotonically.** If the win came from a running statistic tracking the key distribution, tracking faster should help. It does the opposite. At head_dim 4 the scale *frozen at its first batch* is nearly twice as good as the standard running average, and the ordering is monotone from frozen through slow to fast.

<!-- FIG10 -->

<!-- E4_TABLE -->

**It is not key-specific.** The identical construction on the query side performs the same at every width. Applying it to both sides is best at narrow heads and fails at one wide head, reproducing the interaction of Section 4 rather than escaping it.

<!-- FIG11 -->

<!-- E5_TABLE -->

Three independent signals all say the same thing: whatever is helping is a *constant*, it does not need to be learned, and it does not care which side of the dot product it sits on.

## 7. The mechanism is a temperature set at initialization

A per-head scale frozen at its first batch is, algebraically, a constant. Writing s_h for that constant, the arm computes

k_hat = (k / r_t) · (r_t / s_h) = k / s_h,

so the per-token normalization cancels exactly and what remains is a fixed per-head multiplier on the keys, which is a fixed multiplier on the attention logits. That is a temperature. So we swept it.

<!-- E6_TABLE -->

<!-- FIG12 -->

### 7.1 The best constant moves with head width, and the default is wrong

<!-- E6_CURVE -->

### 7.2 The same dial, learnable, does not get there

<!-- E6_LEARN -->

This reconciles the two results that could not both be true. A learnable per-head temperature exists in both experiments. Started at the optimum it stays there and keeps the win; started at the default it barely moves, and at some widths it moves the wrong way. The binding constraint is not whether the model *can* express the right temperature. It is that gradient descent will not travel there from the default within the budget, because the direction is nearly flat in loss and the parameter is one scalar per head competing with 423,424 others.

### 7.3 Folding it into the weights

Because the multiplier is a constant, it does not have to exist at runtime at all: scaling the key
projection's initial weights by the same factor produces the same initial logit scale. That version
is free in every sense, and it is the one to reach for if the forward pass must stay untouched.

<!-- E6_KINIT -->

It is not exactly equivalent, and we should say why. Enlarging the weights changes how weight decay
and Adam's per-parameter normalization act on them for the rest of training, so the two arms share
an initialization but not a trajectory. In our sweep the folded version recovered most, not all, of
the runtime multiplier's benefit.

### 7.4 What the normalizer was doing all along

The comparison the sweep makes available is between a normalizer and a constant *at the same initial logit scale*. RMS-normalizing keys raises their magnitude by roughly the reciprocal of their initial RMS, which is about 4.4 at this width and initialization scheme. That is the same rescale the constant applies. The two arms therefore start from nearly the same attention sharpness and differ only in whether per-token magnitude survives.

<!-- E6_DECOMP -->

Read that way, the normalizer was doing two things at once and they point in opposite directions. Its mean-scale effect is a large help. Everything it does beyond setting the mean scale is a larger harm. The net is the cost we spent nine nights explaining.

## 8. Where the map holds, and where it does not

Every earlier "strictly better" candidate in this thread died one head width away from where it was found. So the tests that matter are the ones that try to kill the current one.

### 8.1 A second corpus

<!-- E2_TABLE -->

<!-- FIG6 -->

The tiny-head half of the map transfers and gets larger: the normalization cost at head_dim 4 grows on character-level Penn Treebank, and the magnitude repair beats the baseline in three of three seeds at every width tested. The wide-head half does not transfer. On this corpus QK-norm does not lose at head_dim 64, and the composite arm wins at both mid and wide splits. The claim "drop the query norm at wide heads" is a tiny-shakespeare claim, not a general one.

### 8.2 Three times longer training

<!-- E3_TABLE -->

<!-- FIG7 -->

Every effect shrinks with training, by a factor of two to five. The sign survives where it matters: at head_dim 4 the normalization cost is still positive in three of three paired seeds and the repair still negative in three of three, both roughly halved. At head_dim 64 the arms become indistinguishable from each other, all landing within about 0.005 bpc while beating the baseline by about 0.02. So the wide-head *ordering* is a property of the early-training regime and should not be quoted as an architecture recommendation. The tiny-head result is the one that survives every stress test we applied.

## 9. What to actually do

<!-- RECIPE -->

The rule that survives all six experiments: **check the scale of your attention logits at initialization before you reach for a normalizer.** At small head dimension the standard 1/sqrt(head_dim) leaves them far too small, a normalizer fixes that as a side effect while destroying something else, and a constant fixes it without the side effect. Where a normalizer is genuinely wanted for stability at scale, this result does not argue against it; it argues that its scale effect and its normalization effect should be set separately, because at small head dimension they have opposite signs.

## 10. Two companion recipes from the same lab

The attention thread is the deepest in the notebook, and two others produced results worth acting on. Both are summarized from their registry rows; the full tables are in the repository.

### 10.1 Plateau escape in gated linear attention

Multi-query associative recall separates softmax attention from sub-quadratic mixers. Ten nights of paired experiments on a 94k-parameter gated linear-attention model produced a causal chain rather than a leaderboard. A fixed-step "capacity frontier" turns out to measure escape time, not capacity: the model that sits at chance for 15,000 steps breaks out at 17,500. Only a dense per-channel gate escapes reliably, and its advantage is content routing, since the same gate fed another sequence's content is an exact no-op. The gate is rate-limiting from below but never paces from above; the backbone sets the clock; weight decay is a seed-exact no-op on timing. The last suspect was gradient noise, and it is convicted with the opposite sign to the folklore: escape time measured in drift units collapses onto the ratio of learning rate to batch size, and *less* noise escapes earlier.

<!-- FIG8 -->

**Recipe.** Dense per-channel gate, AdamW with no warmup or clipping, learning rate 4e-3 on every parameter group, batch 256. Escape at step 300 in all three seeds, against 1100 for the standard recipe. If the budget is wall-clock on a CPU rather than steps, batch 16 at the same 4x learning rate escapes in fewer seconds; the law says the two knobs trade off exactly. The honest comparison outside the gate family: a Taylor-expansion feature map with no gate at all solves the same cell by step 500 at the ordinary learning rate.

### 10.2 When weight-tied recursion pays

The lab's flagship idea began with a weight-tied looped block and a falsification target. Twelve experiments later: on language-model loss at matched FLOPs the loop loses, and depth itself is nearly flat at this size. Entropy-based early exit is indistinguishable from a coin flip at matched compute, because the fixed-depth quality curve it would exploit is flat. The loop earns test-time compute only under a stochastic depth schedule, where it reaches 0.85 frontier accuracy at 2.7x its trained depth against 0.55 for fixed-depth training; untied depth cannot be extended by any trick. Supervising the intermediate state beats every unsupervised extra-compute mechanism, and trained halting learns real difficulty but under-spends out of distribution.

<!-- FIG9 -->

**Rule.** Tie the block and train the depth stochastically with per-iteration input injection, or do not loop at all. Test a halting rule only on a task whose fixed-depth curve is steep. Spend supervision on the intermediate state before spending it on latent compute.

## 11. Limitations

- **Scale.** One architecture, two layers, d_model 128, 0.42M parameters, 600 steps for the main grid. All arms sit far from convergence. A mechanism that is merely slower to optimize is indistinguishable from a worse one here, and Section 8.2 shows this matters: the wide-head ordering does not survive 3x training.
- **The temperature result is the one we expect to scale, and it is untested above this size.** The argument is about initialization scale, which is a property of the parameterization rather than of the corpus, and it should be checked directly at d_model 512 to 1024 with 16 to 32 heads.
- **Head splits are iso-parameter but not iso-temperature by construction.** That is the point of Section 7, but it also means our "no norm" baseline is not a tuned baseline: part of what every normalizer earns here is a scale correction the baseline never got.
- **Three seeds.** Every headline claim is a three-of-three paired win with all per-seed differences the same sign. Effects below about 0.015 bpc are reported as ties.
- **Two corpora, character level.** Tokenized models, vision, and long context are untested.
- **Prior art was searched from a sandbox where arXiv is blocked**, so several 2026 papers including QUEST are cited from search summaries rather than full text and should be verified before submission.

## 12. Ideas this generates

Ordered by expected information per CPU-minute. Each is an observation, an inference, and the experiment that would settle it.

1. **Observation:** the optimal constant differs by head width. **Inference:** the right parameterization sets the key projection's initialization scale as a function of head_dim, not a fixed 0.02. **Experiment:** fit the optimal constant against head width across two or three d_model values and check whether it follows a clean power law; if it does, it is a one-line change to every transformer initializer.
2. **Observation:** a learnable temperature does not travel from 1 to the optimum. **Inference:** the loss surface is nearly flat in that direction at the default. **Experiment:** measure the gradient and curvature of the loss with respect to log c at initialization across head widths; a flat gradient at small head_dim and a steep one at large would explain the whole thing analytically.
3. **Observation:** the baseline was never given the scale correction. **Inference:** part of every published normalization win at small head dimension may be a scale correction in disguise. **Experiment:** re-run a standard QK-norm ablation against a *temperature-tuned* baseline rather than a default one, at a scale where QK-norm is known to help.
4. **Observation:** per-token magnitude costs 0.26 bpc to destroy at the same mean scale. **Inference:** the information is in the magnitude, so a normalizer that preserves it should recover it. **Experiment:** SeeDNorm on the key side against our constant, at head_dim 4, same harness.
5. **Observation:** stacking both sides fails only at one wide head. **Inference:** the failure is two independent sharpness dials on one softmax. **Experiment:** hold the trained logit standard deviation fixed across arms and see whether the interaction disappears, which turns a correlational account into a causal one.
6. **Observation:** at head_dim 64 after 1800 steps every normalizer ties. **Inference:** these are optimization-speed effects, not capacity effects. **Experiment:** train one wide-head cell to convergence and check whether any gap remains.
7. **Observation, from the recall thread:** the Taylor feature map escapes with no gate. **Inference:** gating compensates for a feature-map deficiency rather than adding capacity. **Experiment:** dense gate on top of the Taylor map; if escape time does not move, they are substitutes.
8. **Observation, from the recursion thread:** the never-crossed cell is stochastic depth against matched-FLOP language-model loss. **Experiment:** re-run the first loop experiment with a stochastic depth schedule and input injection, the cheapest way to learn whether the loop's loss verdict was a schedule artifact.

## References

- Henry, A., Dachapally, P. R., Pawar, S., Chen, Y. Query-Key Normalization for Transformers. Findings of EMNLP 2020. arXiv:2010.04245.
- Dehghani, M. et al. Scaling Vision Transformers to 22 Billion Parameters. 2023.
- Wortsman, M. et al. Small-scale proxies for large-scale Transformer training instabilities. ICLR 2024. arXiv:2309.14322.
- QUEST: A robust attention formulation using query-modulated spherical attention. ICLR 2026. arXiv:2604.00199.
- Cai, W., Zhu, D., Liu, Q., Min, Q. SeeDNorm: Self-Rescaled Dynamic Normalization. arXiv:2510.22777.
- Meng, Z. et al. Norm x Direction: Restoring the Missing Query Norm in Vision Linear Attention. arXiv:2506.21137.
- Nguyen, T. Q., Salazar, J. Transformers without Tears (ScaleNorm). IWSLT 2019. arXiv:1910.05895.
- Selective Attention: Enhancing Transformer through Principled Context Control. NeurIPS 2024. arXiv:2411.12892.
- Veličković, P. et al. Softmax is not Enough (for Sharp Size Generalisation). arXiv:2410.01104.
- Nakanishi, K. M. Scalable-Softmax Is Superior for Attention. arXiv:2501.19399.
- Qiu, Z. et al. Gated Attention for Large Language Models. NeurIPS 2025. arXiv:2505.06708.
- Mudarisov, T. et al. Limitations of Normalization in Attention Mechanism. arXiv:2508.17821.
- Bhojanapalli, S. et al. Low-Rank Bottleneck in Multi-head Attention Models. ICML 2020. arXiv:2002.07028.
- Most Transformer Modifications Still Do Not Transfer at 1-3B. arXiv:2605.20798.
- Loshchilov, I. et al. nGPT: Normalized Transformer on the Hypersphere. arXiv:2410.01131.
- Kimi Team. Kimi K2 (MuonClip / QK-Clip). arXiv:2507.20534.
- Arora, S. et al. Zoology / BASED. 2023-2024.
- Geiping, J. et al. Scaling up Test-Time Compute with Latent Reasoning. arXiv:2502.05171.
- Xie, Z., Sato, I., Sugiyama, M. A Diffusion Theory for Deep Learning Dynamics. arXiv:2002.03495.

## Appendix A. Replication anchors

Every experiment in this paper re-runs cells from its parents and requires agreement to 0.0005 bpc before its own numbers are trusted.

<!-- ANCHOR_TABLE -->

## Appendix B. The attention-norm ledger

| night | registry id | what it established |
|---|---|---|
| 07-26 | head-dim-vs-count-isoparam | loss is monotone in head width without normalization |
| 07-30 | qknorm-head-dim | QK-norm removes the mid-range tax and deepens the tiny-head cost |
| 07-31 | qknorm-hd4-temperature-rescue | a learnable static temperature refunds two percent and is left at one |
| 08-02 | qknorm-hd4-dynamic-temperature | a detached per-token query temperature refunds 54 percent |
| 08-06 | qknorm-hd4-undetached-magnitude | opening the gradient path refunds 98 percent; the raw form refunds four |
| 08-11 | qknorm-dyntemp-composite-sweep | the composite dominates QK-norm but inherits its one-head cost |
| 08-23 | qknorm-nh1-tax-mechanism | at one head each one-sided norm wins and both together lose |
| 08-30 | knorm-only-head-sweep | the tiny-head cost follows the key norm; the interaction flips sign with width |
| 08-31 | hd4-kside-cliff-mechanism | the cost is a severed magnitude gradient, not a lost value |
| 09-01 | knorm-dynk-head-sweep | the repair beats baseline at every width, and the frozen exponent matches it |
| 09-01 | knorm-dynk-ptb-transfer | the tiny-head half transfers to a second corpus; the wide-head half does not |
| 09-01 | knorm-dynk-longer-training | every effect halves at 3x training; the wide-head ordering dissolves |
| 09-01 | kscale-adaptive-vs-static | less adaptation is better; a frozen scale wins |
| 09-01 | fractional-norm-both-sides | the channel is not key-specific; both sides fails at one wide head |
| 09-01 | logit-scale-sweep | it is a temperature, the default is wrong, and learning it from the default does not work |

## Appendix C. Reproduction

```
pip install torch==2.13.0 numpy matplotlib pyyaml     # the lab's runs used torch 2.13.0 on CPU
python experiments/2026-09-01_logit-scale-sweep/run.py        # shard with --head-dims/--seeds/--tag, then --merge
python paper/fill_sections.py --apply                          # regenerate every table from results.json
python paper/make_figures.py                                   # regenerate every figure
python attnscale/test_attnscale.py                             # unit tests for the shipped module
```
