# daily-research-lab

A personal lab for **one small AI/ML research experiment per day** — every experiment trains on a normal CPU (no GPU), is recorded so nothing is ever redone, and is checked against prior work before it is built. Standout experiments graduate into their own repositories.

## Why this exists
Small, sharp experiments compound. A tiny model that reveals something (a grokking curve, a probe that lies, a recursion that generalizes) is worth more than a big model that reveals nothing. This repo is the engine: a low-friction loop that turns one hour a day into a growing, searchable body of work.

## The loop (do this each day)
1. Pick an idea (from `docs/BACKLOG.md` or a new one).
2. **Search before creating** — check your own registry, then arXiv / Semantic Scholar / OpenAlex / GitHub / Hugging Face (see `docs/SYSTEM.md`). Record what you found.
3. Scaffold: `python scripts/new_experiment.py "<slug>"`.
4. Implement `run.py` (seeded, CPU, writes `results.json`).
5. Run it. Look at the result. Write the one-paragraph takeaway.
6. `python scripts/validate_registry.py && python scripts/build_index.py`
7. Commit and push. If it is a standout, spin it into its own repo.

## Layout
```
registry.jsonl              append-only master log (source of truth)
registry.schema.json        schema the registry is validated against
docs/BACKLOG.md             curated experiment ideas (the runway)
docs/DATASETS.md            tiny CPU-friendly datasets ("fuel")
docs/SYSTEM.md              the daily workflow + search-before-create
templates/                  the per-experiment skeleton
scripts/                    scaffold / novelty-check / validate / index
experiments/                one dated folder per experiment
```

## Index
<!-- INDEX:START -->
**25 experiments logged.**

| Date | Experiment | Status | Result |
|---|---|---|---|
| 2026-07-26 | [MQAR at nano scale: the linear-attention recall cliff is pinned at N~8 across a 4x width sweep (state-size story refuted at fixed budget), and a scalar forget gate is a no-op](experiments/2026-07-26_mqar-state-capacity) | ✅ done | Frontier (max N with acc>=0.9, d=32/64/128): attn 8/16/16, linattn 2/4/4, gla 2/4/4. The l |
| 2026-07-25 | [NoPE vs RoPE vs ALiBi vs APE: does the length-generalization ranking survive at 0.1M params?](experiments/2026-07-25_pe-length-gen-tiny) | ✅ done | Hypothesis refuted: at 0.1M params and iso-compute the length-generalization ranking is AL |
| 2026-07-25 | [Shadow E1: at iso-FLOPs on a 0.06-0.21M-param char LM, a weight-tied looped block loses to plain depth](experiments/2026-07-25_shadow-loop-vs-depth-isoflop) | ✅ done | Hypothesis supported. At iso-FLOPs the weight-tied loop never beats untied depth and the p |
| 2026-07-25 | [Does the Game-of-Life capacity window transfer to language? Tied vs untied recursion across width](experiments/2026-07-25_shadow-recursion-capacity-window) | ✅ done | Hypothesis refuted - the Game-of-Life capacity window does not transfer to language. Weigh |
| 2026-07-25 | [Shadow E3: entropy-based adaptive exit on a tiny looped char LM - no better than a coin flip at matched compute](experiments/2026-07-25_shadow-halt-entropy-tiny) | ✅ done | Hypothesis refuted, honest negative. Every entropy operating point sits on or above the fi |
| 2026-07-25 | [TRM at nano scale: the outer refinement loop is the whole story, the inner latent recursion is not](experiments/2026-07-25_trm-nano-sudoku) | ✅ done | Hypothesis confirmed on the mechanism, refuted on the saturation claim - and the two resul |
| 2026-07-25 | [nano-Coconut on DAG reachability: continuous thoughts need the curriculum, still lose to discrete CoT, and are not a BFS frontier](experiments/2026-07-25_coconut-toy-graph) | ✅ done | Half confirmed, half refuted, and the refuted half is the interesting one. On accuracy the |
| 2026-07-25 | [Grokking (a+b) mod 59: spectral-entropy collapse LAGS the test-accuracy jump; only restricted loss leads](experiments/2026-07-25_grokking-modular-addition) | ✅ done | Hypothesis refuted, with a mechanism for why the opposite is reported. The classic grokkin |
| 2026-07-25 | [The probe can lie: at matched 99%+ linear-probe accuracy, top-of-stack type costs 5.31 excess nats to erase and depth parity costs less than a random direction](experiments/2026-07-25_dyck-probe-can-lie) | ✅ done | Hypothesis confirmed for the headline feature, with two honest misses. The 27k-param model |
| 2026-07-25 | [KAN on Feynman I.12.2: Coulomb's law recovered EXACTLY (exponent error 0.006), but the sample-efficiency claim is refuted - the KAN is 2.8x worse than an iso-param MLP at n=25 and its relative standing IMPROVES with data](experiments/2026-07-25_kan-feynman-symbolic) | ✅ done | The two halves of the KAN pitch come apart on the same equation. Symbolic recovery is EXAC |
| 2026-07-25 | [MQAR at ~0.1M params: attention holds 1.00 recall at every KV load to 24 while a decay-gated linear-attention mixer and a routing-free gated conv both sit ON the no-recall guessing baseline - the 'graceful capacity slide' is chance, not capacity](experiments/2026-07-25_zoology-mqar-recall) | ✅ done | The zoology ranking reproduces at nano scale and the fresh question dissolves. Under a str |
| 2026-07-25 | [Weight decay and the grokking delay on (a+b) mod 59: monotone where it groks (1825 -> 750 steps for WD 1 -> 3) but the window is only ~1 decade wide, and WD never touches memorization speed](experiments/2026-07-25_grokking-weight-decay-phase) | ✅ done | A 7-point weight-decay sweep spanning three orders of magnitude (0.03 to 30) on a 57k-para |
| 2026-07-25 | [Test-time compute on a 0.06M-param weight-tied loop: trained at a FIXED K=3 the loop degrades past its trained depth (frontier acc 1.00 -> 0.55, easy instances 1.00 -> 0.71); trained with a STOCHASTIC depth schedule the identical model extrapolates to 2.67x the trained depth (0.85 at K=8, 11x full-sequence exact match)](experiments/2026-07-25_loop-test-time-compute) | ✅ done | The direct test of the 'test-time-compute axis' that 2026-07-25_shadow-loop-vs-depth-isofl |
| 2026-07-25 | [SAEs on a grokked (a+b) mod 59 transformer: against a provable Fourier ground truth the raw neuron basis is 0.75 frequency-pure and the best usable SAE only 0.32 - the sparsity prior fights the density of the true features](experiments/2026-07-25_sae-on-grokked-model) | ✅ done | The hypothesis is REFUTED, and the sharpest number is the baseline nobody runs. Against a  |
| 2026-07-25 | [The merge breakpoint: correlated feature pairs never choose local orthogonality - unequal importance kills the weak member by amplitude decay along the SAME direction, and anticorrelated values only bend the merge to |cos| 0.64-0.75](experiments/2026-07-25_superposition-merge-breakpoint) | ✅ done | The hypothesis is REFUTED on the importance axis and only partially met on the value axis; |
| 2026-07-25 | [Echo State Network on Mackey-Glass-17: there is no edge-of-chaos ridge just below sr=1 - the free-running optimum sits ABOVE 1 in 13 of 16 reservoirs (median sr 1.10), the sr=0.9 default costs 7.3x the error, and the memory-capacity peak is too flat to localise it](experiments/2026-07-25_esn-reservoir) | ✅ done | The hypothesis is REFUTED on location and only half-confirmed on sharpness. (1) THE OPTIMU |
| 2026-07-25 | [SAEs on the merged-pair toy model: the dictionary recovers all 8 MERGED pair-directions one-to-one and 0 of the 16 true features (max member-selectivity 0.007/1.0 across 27 SAEs) - but an MLP probe reads the within-pair difference at R2 0.47 from the same 4 numbers, so the SAE's ceiling is its linear hypothesis class, not lost information](experiments/2026-07-25_sae-on-merged-pairs) | ✅ done | The hypothesis is CONFIRMED on the SAE half and REFUTED on the mechanism half, and the ref |
| 2026-07-25 | [Validation-based clause pruning un-buries the planted DNF: at 20% label noise a 500-sample NOISY validation set restores Tsetlin clause precision 0.49 -> 1.00 with zero recall loss, and the binding constraint is the unknown noise rate, not the missing clean labels](experiments/2026-07-25_tsetlin-clause-pruning) | ✅ done | Hypothesis confirmed at eps=0.2, partially at eps=0.3, refuted at eps=0.4 - and the real o |
| 2026-07-25 | [Modern Hopfield capacity at d=64 is exponential in d with the textbook constant (0.244 bits/dim, R2=0.993) but stops responding to beta above 1 - the update is just a nearest-neighbour decoder, and corruption, not temperature, is the binding constraint](experiments/2026-07-25_modern-hopfield-recall) | ✅ done | Split verdict. CONFIRMED: capacity is exponential in d with the textbook constant - log2(c |
| 2026-07-25 | [Pause/filler tokens and true recurrence buy the SAME accuracy per FLOP on serial mod-5 arithmetic chains (tie within 1-2 eval SE), both saturate after the FIRST extra application, the whole-sequence loop is 3.1-3.6x WORSE per FLOP than spending nothing, and 3 tokens of discrete intermediate supervision hit 1.000 at fewer FLOPs than 4 pause tokens](experiments/2026-07-25_filler-vs-recur) | ✅ done | Main hypothesis refuted by a tie, with three sharper findings underneath. (1) Put on the s |
| 2026-07-25 | [Removing the hidden-state dependence of the gates (minGRU/minLSTM) costs 0.36-0.64 exact match vs a standard GRU on selective copy at matched params and steps - and the cost is ORDERING older items (hard recency profile: last 2 slots perfect, rest near chance), not remembering distant ones; depth (2 layers: 0.39->0.82) and 3x steps (0.39->0.96) substitute for the missing gate at k=4](experiments/2026-07-25_minrnn-selcopy) | ✅ done | Hypothesis confirmed, with the mechanism localised and two honest qualifiers. (1) At ~60k  |
| 2026-07-25 | [Induction heads form as a phase transition and weight-space K-composition LEADS it by 601 steps (133 steps under a strictly online criterion) where the attention-pattern scores LAG - but the rise is shared by all 16 head pairs, overshoots 1.68x and decays, and fires a late false positive on a model where induction is impossible](experiments/2026-07-25_induction-heads-emergence) | ✅ done | Hypothesis confirmed on TIMING, refuted on SPECIFICITY. (1) The phase transition reproduce |
| 2026-07-24 | [Tsetlin Machine: does exact rule recovery break before accuracy under label noise?](experiments/2026-07-24_tsetlin-dnf-recovery) | ✅ done | Hypothesis partially refuted: exact recall of the planted clauses survives to 20% label no |
| 2026-07-23 | [Superposition phase diagram: does feature correlation resist or reshape superposition?](experiments/2026-07-23_superposition-correlation-phase) | ✅ done | Hypothesis refuted: across the grid, raising within-pair correlation makes the model colla |
| 2026-07-21 | [Game of Life: when does test-time recursion still generalize?](experiments/2026-07-21_gol-depth-recursion) | ✅ done | Capacity window: at H=4 tied recursion learns the exact reusable rule and extrapolates to  |
<!-- INDEX:END -->

## Ground rules
- Every experiment must fit on a CPU in roughly minutes to ~1 hour.
- Every experiment records a novelty check. Replication is allowed — but labelled honestly.
- Results are written by code, never by hand.
- License: MIT (code), CC-BY-4.0 (notes/figures).
