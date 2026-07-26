# Experiment backlog

Curated from a multi-agent research sweep (July 2026). Every item is CPU-only (minutes to ~1 hour), has a novelty angle, and lists the closest prior art so you can **search before you build**. Difficulty is 1 (trivial) to 5 (hard). Pull one per day; when you run it, move it into the registry and check it off here.

Legend: ⭐ = strong first picks · 🟢 trains on CPU as-is · 🟡 CPU-OK if you avoid an optional CUDA path.

---

## Track A — Latent / recursive / non-token reasoning (Coconut · TRM · HRM · CTM lineage)

| id | idea | arch | tiny task | ~CPU | novel angle | closest prior art | diff |
|---|---|---|---|---|---|---|---|
| gol-depth-recursion ⭐ | weight-tied cell learns Game of Life; test-time iterations = compute | tied recurrent CNN <10k | Conway 16×16, generated | 5–15 min | does iterating MORE at test time generalize to MORE steps than trained? tied vs untied | difflogic-ca, Neural CA | 2 |
| ~~trm-nano-sudoku~~ ✅ 2026-07-25 | DONE: outer answer-space refinement climbs 0.22→0.51 solve rate at T=1..8 matched compute while inner latent recursion stays flat (0.22→0.25); latent depth is spent on memorization; "94% at step 1" holds only for the latent arm | 2-layer MLP-recursion ~0.12M | 4×4 Sudoku, split by solution grid | ~7 min | reproduce ARC-Prize "refinement-loop-not-hierarchy" at nano scale | [trm_sudoku](https://github.com/allthingssecurity/trm_sudoku) | 2 |
| ~~coconut-toy-graph~~ ✅ 2026-07-25 | DONE: continuous thoughts +0.27 acc over no-CoT, +0.19 over pause tokens, and IMPROVE with hop depth where discrete CoT degrades (0.93 vs 0.77 at 3 hops) — but the curriculum is load-bearing (+0.33) and the thought is NOT a linearly-readable BFS frontier (probe barely beats random-init control) | 2-layer GPT ~0.1M | synthetic DAG reachability, adjacency-slot encoding | ~9 min | probe the latent for the BFS frontier; continuous vs discrete vs no-CoT | [coconut](https://github.com/facebookresearch/coconut), [lucidrains](https://github.com/lucidrains/coconut-pytorch) | 3 |
| looped-halt-nrasp | looped transformer + PonderNet halting for length generalization | 1 tied block ~0.1–0.4M | n-RASP-L copy/parity/add, train≤20 test≤40 | 15–40 min | adaptive depth vs fixed loops for extrapolation | [looped-tf](https://github.com/UW-Madison-Lee-Lab/looped-tf) | 3 |
| ctm-parity-sync | Continuous Thought Machine, ablate the synchronization readout | CTM ~0.05–0.3M | 32/40-bit parity, generated | 15–45 min | is timing-sync the signal, or just recurrence depth? | [CTM](https://github.com/SakanaAI/continuous-thought-machines) | 3 |
| filler-vs-recur | pause/filler tokens vs true recurrence at equal compute | 2-layer tf ~0.2–0.5M | modular-arithmetic chains | 15–30 min | which extra-compute mechanism buys more accuracy per FLOP | [pause tokens](https://arxiv.org/abs/2310.02226) | 2 |
| quantized-coconut | VQ bottleneck on the continuous thought → readable thoughts | Coconut + tiny codebook | graph reachability / mod-arith | 20–40 min | how discrete can a "thought" be before accuracy drops? | none direct | 3 |
| listops-recursion | latent recursion on nested-expression evaluation | looped block ~0.2–0.6M | ListOps-mini, depth 1–4 → test 5–6 | 20–40 min | scale loops with parse depth; probe subtree values | [looped-tf](https://github.com/UW-Madison-Lee-Lab/looped-tf) | 3 |

---

## Track B — Interpretability / grokking / toy world-models (very CPU-friendly, very shareable)

| id | idea | arch | tiny task | ~CPU | novel angle | closest prior art | diff |
|---|---|---|---|---|---|---|---|
| ~~grokking-modular-addition~~ ✅ 2026-07-25 | DONE: grok reproduced (p=59, delay 642 steps, test 1.0, 6 Fourier freqs = 90.5% power) but spectral entropy LAGS the jump by 600-900 steps while restricted/excluded loss LEADS (+351/+562) — the "build once, reuse" spectral measure idea is refuted; grokked model.pt + memorized model_mid.pt saved locally (gitignored) for the SAE follow-up | 1-layer tf no LN, 57k params | (a+b) mod 59, frac sweep 0.5/0.4/0.3 | ~10 min | benchmark spectral-entropy-collapse vs restricted/excluded loss; does Grokfast distort it? | [progress-measures](https://github.com/mechanistic-interpretability-grokking/progress-measures-paper) | 2 |
| ~~dyck-probe-can-lie~~ ✅ 2026-07-25 | DONE: confirmed — depth_parity probes at 0.995 but LEACE-erasing it costs −0.16 nats (less than a random direction) while top_type at equal probe acc costs +5.31 nats; probe accuracy carries no information about causal use. Verdict: replication of 2604.22128 with certified-erasure upgrades | 2-layer 1-head d=32, 27k params | Dyck-2 brackets, LEACE | ~6 min | probe-accuracy vs causal-effect per variable (decodable ≠ used) | [2604.22128](https://arxiv.org/html/2604.22128v1) | 3 |
| ~~superposition-phase-diagram~~ ✅ 2026-07-23 | DONE as `superposition-correlation-phase`: (density × correlation) phase diagram; pairs MERGE, they do not orthogonalize | Anthropic toy model, tiny | synthetic sparse features | seconds–2 min | correlated features / alt activations deform the geometry | [toy-models-of-superposition](https://github.com/anthropics/toy-models-of-superposition) | 1 |
| superposition-merge-breakpoint | when does pair-merging become too lossy? unequal importance or anticorrelated values within a pair should force local orthogonality; find the breakpoint | Anthropic toy model, tiny | synthetic correlated pairs | minutes | follow-up to superposition-correlation-phase (merging won everywhere at equal importance) | [toy-models-of-superposition](https://github.com/anthropics/toy-models-of-superposition) | 2 |
| sae-on-merged-pairs | train a tiny SAE on the merged-pair toy model: does it recover 8 pair-features or 16 true features? | toy model + small SAE | activations from superposition-correlation-phase | minutes | SAE faithfulness when ground truth is a KNOWN merged representation | ARENA 3.1 SAE tutorials | 2 |
| othello-probe-vs-cause | emergent board world-model, probed vs ablated | probe pretrained Othello-GPT | ~1–4k Othello games (probe only) | 20–60 min | which board cells are decodable but NOT causal | [othello_world](https://github.com/likenneth/othello_world) | 3 |
| sae-on-grokked-model | do SAEs recover the KNOWN Fourier features? | grok tf + small SAE | mod-addition activations | 30–55 min | SAE faithfulness vs a provable ground truth | ARENA 3.1, [canonical-units critique](https://proceedings.iclr.cc/paper_files/paper/2025/file/84ca3f2d9d9bfca13f69b48ea63eb4a5-Paper-Conference.pdf) | 3 |
| induction-heads-emergence | watch an induction head form as a phase transition | 2-layer attn-only, d≈64–128 | repeated random-token seqs | 10–30 min | predict the onset with a progress measure | [in-context-learning-and-induction-heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) | 3 |
| clock-vs-pizza | two algorithms for one task; map the phase boundary | 1-layer ReLU tf, width sweep | (a+b) mod p | 15–45 min/run | quantitative phase diagram of what tips clock↔pizza | [2306.17844](https://arxiv.org/abs/2306.17844) | 3 |
| tracr-ground-truth-lab | interpretability with the answer key | Tracr-compiled tf (no training) | histogram/sort/parens | minutes | use as a benchmark to score your probes/SAEs | [tracr](https://github.com/google-deepmind/tracr) | 2 |
| group-grokking | grokking beyond +: abelian vs non-abelian groups | 1-layer tf, d=128 | ℤ/p vs S₅ multiplication | 15–90 min | does the progress measure fire the same across the divide? | [2302.03025](https://arxiv.org/abs/2302.03025) | 4 |

---

## Track C — Non-transformer architectures (genuinely different families)

| id | idea | arch | tiny task | ~CPU | novel angle | closest prior art | diff |
|---|---|---|---|---|---|---|---|
| ~~tsetlin-logic~~ ✅ 2026-07-24 | DONE as `tsetlin-dnf-recovery`: noise buries rules (precision 0.49 at eps=0.2), it does not erase them (recall 1.0) | pure-numpy TM (pyTsetlinMachine C ext does not build here) | planted 3-clause DNF | ~4 min | exact-rule recovery + interpretability tradeoff | [pyTsetlinMachine](https://github.com/cair/pyTsetlinMachine) | 2 |
| tsetlin-clause-pruning | validation-based clause pruning/weighting restores precision 1.0 on the noise-buried rule set from tsetlin-dnf-recovery | pure-numpy TM + prune step | planted 3-clause DNF, eps 0.2-0.3 | minutes | follow-up: recover a CLEAN rule set from a cluttered TM | own registry 2026-07-24 | 2 |
| zoology-mqar-recall ⭐🟡 | why sub-quadratic models fail at recall | swap mixers | MQAR synthetic | minutes | add your own pure-PyTorch mixer to the harness | [zoology](https://github.com/HazyResearch/zoology) | 3 |
| ~~kan-feynman-symbolic~~ ✅ 2026-07-25 | DONE: split — symbolic recovery EXACT (Coulomb's law, exponents within 0.006, snapping cuts RMSE 32x) but sample-efficiency REFUTED (KAN 2.8x worse than swept iso-param MLP at n=25; wins only at large n vs single unswept MLP) | pykan 0.2.8, 200 params vs best-of-4 MLP | Feynman I.12.2, n sweep 25-1000 | ~8 min | sample-efficiency vs MLP + PySR | [pykan](https://github.com/KindXiaoming/pykan) | 2 |
| minrnn-selcopy 🟢 | "Were RNNs All We Needed?" minGRU/minLSTM | minRNN parallel scan | selective-copy | min–30 min | ablate the paper's central "no hidden-state gate" claim | [minRNNs](https://github.com/BorealisAI/minRNNs) | 2 |
| esn-reservoir 🟢 | fixed random reservoir, train only a linear readout | Echo State Network | Mackey-Glass chaos | seconds | edge-of-chaos spectral-radius ridge | [reservoirpy](https://github.com/reservoirpy/reservoirpy) | 1 |
| mamba-mini-induction 🟢 | tiny selective SSM grows induction on copy | mamba.py ~0.25–0.5M | selective-copy | 15–45 min | is Δ-selectivity the load-bearing part? | [mamba.py](https://github.com/alxndrTL/mamba.py) | 3 |
| cfc-liquid-timeseries 🟢 | closed-form continuous-time (liquid) net | CfC | irregular sine / Mackey-Glass | sec–min | robustness under irregular sampling vs LSTM | [CfC](https://github.com/raminmh/CfC) | 3 |
| retnet-nano-char 🟢 | retention (decaying linear attention) char LM | RetNet ~0.5–1M | tiny-shakespeare | 20–40 min | prove parallel≈recurrent; does decay schedule matter? | [yet-another-retnet](https://github.com/fkodom/yet-another-retnet) | 3 |
| snn-spiking 🟢 | spiking net + surrogate gradients | SNN 0.05–0.5M | rate/latency-coded MNIST | 10–30 min | time-step vs accuracy frontier; rate vs latency coding | [snntorch](https://github.com/jeshraghian/snntorch) | 3 |
| modern-hopfield-recall 🟢 | associative memory / pattern completion | modern Hopfield | corrupted→recovered images | minutes | capacity vs corruption curve | [hopfield-layers](https://github.com/ml-jku/hopfield-layers) | 2 |

> ⚠️ Do NOT fork directly (they secretly need a GPU/CUDA kernels): `state-spaces/mamba` (use mamba.py), `nanoRWKV` both forks (use pure-PyTorch WKV), `flash-linear-attention`, `NX-AI/xlstm` (use myscience/x-lstm). `zoology`/`safari` are CPU-OK only if you skip the optional `mamba_ssm`/`fla`/`causal-conv1d`/`fftconv` extras.

---

## Track D — Tiny controlled ablations (one plottable answer per afternoon)

| id | hypothesis | setup | ~CPU | novel angle | closest prior art | diff |
|---|---|---|---|---|---|---|
| ~~nope-vs-rope-vs-alibi~~ ✅ 2026-07-25 | DONE as `pe-length-gen-tiny`: ranking INVERTS at 0.1M/iso-compute — ALiBi 0.82 > RoPE 0.56 > NoPE 0.19 > APE 0.08 (OOD token acc); follow-up idea: give NoPE 10-20x steps to separate slow-learning from cannot-extrapolate | 2-layer decoder, copy, train≤16 test≤32 | ~11 min (4 PEs × 2 seeds) | first tiny-CPU replication of the 107M result | [2305.19466](https://arxiv.org/pdf/2305.19466v2) | 2 |
| grokking-weight-decay-phase ⭐ | weight decay controls the memorize→grok delay | 1–2 layer tf, mod-97, WD sweep | 2–3 hr | steps-to-grok vs weight-decay curve | [teddykoker/grokking](https://github.com/teddykoker/grokking) | 2 |
| loop-test-time-compute ⭐ | looped block trades test-time compute for accuracy | shared block, parity/add, train K≤3 test K≤8 | ~1.5 hr | CPU parity demo of recurrent-depth extrapolation | [2502.05171](https://arxiv.org/pdf/2502.05171v1) | 3 |
| muon-vs-adamw-vs-soap | Muon's win survives per-step but shrinks per-wall-clock on CPU | nanoGPT-char, 3 optimizers | 1–3 hr | "seconds not steps" normalization | [Muon](https://github.com/KellerJordan/Muon) | 3 |
| swiglu-vs-gelu-isoparam | SwiGLU's edge shrinks once you equalize params | nanoGPT-char, iso-param FFN | ~1.5 hr | the iso-parameter control most demos skip | [2002.05202](https://arxiv.org/pdf/2002.05202) | 2 |
| byte-vs-bpe-isoflops | BPE wins bits/byte at fixed compute; bytes win spelling | nanoGPT-char, enwik8-1M | ~1.5 hr | strict bits-per-byte + spelling probe | ByT5/BLT | 2 |
| weight-tying-tiny | tying helps only when vocab/param ratio is high | char vs BPE vocab | ~1.5 hr | benefit-of-tying vs embedding-fraction curve | [1608.05859](https://arxiv.org/abs/1608.05859) | 1 |
| head-dim-vs-count-isoparam | moderate head_dim beats 1-giant and many-tiny heads | d_model=128, vary (n_head, head_dim) | ~1.5 hr | clean iso-param U-curve isolating accuracy | QK-norm/GQA discussions | 2 |
| mup-lr-transfer | optimal LR is width-invariant under µP, drifts under SP | nanoGPT-char, widths 64/128/256 | 2–3 hr | does µP LR-transfer survive Muon? | [microsoft/mup](https://github.com/microsoft/mup) | 4 |
| data-repetition-epochs | the "~4 epochs ≈ fresh data" knee at ~1M params | fixed compute, vary repetition | ~1.5 hr | does the knee shift at tiny scale? | [2305.16264](https://arxiv.org/abs/2305.16264) | 2 |

---

## Track E — Shadow (ledger-built tiny model → ARC Prize 2026)

Shadow is the lab's first flagship: a tiny model whose every architecture choice is earned through a forge ablation and recorded in a one-page **architecture ledger** citing the registry experiment id that justified it. Primary target: **ARC Prize 2026** (Kaggle, deadline 2026-11-02, offline, compute-limited — the board where a 7M recursive model already proved tiny can compete). The talking-LM version (TinyStories/BabyLM-style) is the longer arc, deliberately unanchored for now. Research day 2026-07-23: novelty verdict **partial-prior-art** — the looped/recursive mechanism is colonized at ≥1.4B (Ouro 2510.25741, MoR 2507.10524, recurrent-depth 2502.05171, TRM 2510.04871) but the ≤1–50M iso-FLOP territory on natural data is open, and the TRM critique (2512.11847: 94% of performance at recursion step 1; gains from augmentation ensembling + puzzle-id conditioning) is precisely the claim this track exists to test honestly. New-architecture status is **earned, not asserted**: if E1 falsifies the loop, Shadow ships as the ledger-assembled model and the novelty lives in the method plus component-level firsts.

| id | hypothesis | setup | ~CPU | novel angle | closest prior art | diff |
|---|---|---|---|---|---|---|
| ~~shadow-loop-vs-depth-isoflop~~ ✅ 2026-07-25 | DONE: the loop LOSES — tied-k4 +0.079 bpc vs untied-k4 at iso-FLOPs, and tied-k4 is worse than tied-k1 at 4x FLOPs; depth axis nearly flat. Shadow's loop must earn its place on test-time-compute/extrapolation axes, not raw LM loss. Caveat: undertrained, 0.06-0.2M params | nanoGPT-char ~0.06-0.2M, k∈{1,2,4} loops vs matched-depth baselines, 2 seeds, tiny-shakespeare | ~11 min | first iso-FLOP loop-vs-depth control at ≤1M params on natural-language data; Ouro/recurrent-depth only tested ≥1.4B | [2510.25741](https://arxiv.org/abs/2510.25741), [2502.05171](https://arxiv.org/pdf/2502.05171v1) | 3 |
| ~~shadow-recursion-capacity-window~~ ✅ 2026-07-25 | DONE: does NOT transfer — tied never wins at any width (delta +0.023/+0.069/+0.036/−0.005 bpc at d=16/32/64/128); tied penalty PEAKS at intermediate width, inverse of the GoL shape. Caveat: 400 steps, early-training regime | tiny looped LM, width sweep d∈{16,32,64,128}, tied vs untied k=4, tiny-shakespeare | ~8 min | direct transfer test of registry 2026-07-21_gol-depth-recursion to language; no published capacity-window sweep for looped LMs | own registry + [looped-tf](https://github.com/UW-Madison-Lee-Lab/looped-tf) | 3 |
| ~~shadow-halt-entropy-tiny~~ ✅ 2026-07-25 | DONE: entropy halting never beats fixed depth and is indistinguishable from compute-matched RANDOM exit — the entropy signal carries no depth-allocation information; collapses to MAX depth (not shallow); best point overall is the dedicated k=1 model. Provisional prior-art flag: arXiv 2607.20519 unverifiable (403) | inference-time entropy exit on deep-supervised k=4 looped char-LM, 0.062M, tiny-shakespeare, + random-exit control | ~8 min | adaptive-depth halting has never been ablated below 1B; the TRM critique ([2512.11847](https://arxiv.org/abs/2512.11847)) predicts collapse to shallow | [2507.10524](https://arxiv.org/abs/2507.10524) | 4 |

**Track E bridge to ARC:** the ARC-facing complement is already in Track A — `trm-nano-sudoku`, `coconut-toy-graph`, `looped-halt-nrasp` are the solver-side ablations. Winners from E and A merge into the Shadow ARC entry. Ledger rule: no knob enters the ARC submission without a registry id behind it.

---

### Notes
- Several 2026-dated arXiv IDs surfaced in research were abstract-only; verify before leaning on their exact numbers.
- The **spectral-entropy progress measure** is high-leverage: build it once, reuse across grokking / induction / group-grokking / emergence experiments.
- The heaviest / most likely to exceed an hour on CPU: anything maze/energy-based, µP grids, GPT-2-small LoRA, from-scratch board world-models. Shrink first.
