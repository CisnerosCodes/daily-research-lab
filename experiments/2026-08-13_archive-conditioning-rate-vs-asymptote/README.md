# Archive-conditioned generation: does in-context negative feedback raise the idea ceiling, or only walk the same support faster?

**Status: PRE-REGISTERED before any generation run.** This section (Hypothesis through Kill conditions) was written and committed before the first generation call. Results appear below the marked line and were written by `analyze.py`, never by hand.

## Question

Agentic "reflect and diversify" loops routinely feed a model its own previous outputs with an instruction to produce something different. The E-series measured that a single aligned model's idea generation has a finite support (Chao1-estimated ~54 functional classes on the reference problem) and that prompt-level interventions do not enlarge it, while component swaps (models, knowledge, framings) do. Archive-conditioning is the ambiguous case: it is prompt-level machinery, but it injects information (the history of what has already been found) that changes round over round — a minimal self-improvement loop with the archive as the accumulating state. Is it a rate intervention or a ceiling intervention? And does an *operator* on the archive (extract the assumptions all prior ideas share, force their negation) behave differently from the archive alone?

## Hypotheses and predicted directions

- **H1 (rate).** Archive-conditioning (arm B) raises distinct functional classes per fixed idea budget over independent identically-prompted batches (arm A). Predicted direction: positive, at or above 2 sigma of measured replicate noise.
- **H2 (asymptote).** Archive-conditioning does **not** raise the estimated support: bias-corrected Chao1 for arm B falls within 2 sigma of arm A. Prediction: archive feedback is a walks-faster intervention that only reduces within-support duplication. This is the load-bearing prediction: if it holds, "ask it to avoid what it already said" — the default diversity move in agentic loops — buys sampling efficiency, not new territory.
- **H3 (operator).** Assumption-negation (arm C) produces more arm-unique classes (classes reached by C and by no other arm within the same replicate) than arm B, at a higher incoherence cost. Predicted direction: positive on arm-unique classes, positive on incoherent count.

## Falsification conditions

- H1 falsified if the B-minus-A paired difference in distinct classes is below 2 sigma of replicate noise. This would be a directly publishable negative: in-context negative feedback does not even accelerate coverage.
- H2 falsified if Chao1(B) exceeds Chao1(A) by more than 2 sigma — archive-conditioning would then be a genuine ceiling intervention and the E-series "prompt axis is exhausted" conclusion needs a carve-out.
- H3 falsified if C's arm-unique class count is at or below B's.

## Design

One problem, fixed across all arms: *public library systems permanently lose circulating items that are borrowed and never returned; generate intervention mechanisms that reduce permanent non-return.* The problem is deliberately not the E-series clinic no-show problem, so no prior codebook or session output can contaminate generation or judging.

Three arms at **iso-idea-budget** (named control): 4 rounds x 6 ideas = 24 ideas per replicate, identical generator model (Claude Haiku via subagent), identical output schema. Arms differ only in the conditioning text of rounds 2-4; round 1 is identical across arms and serves as a manipulation check.

- **A — iid control.** Four independent batches, no history.
- **B — archive feedback.** Sequential; round k sees the titles of all ideas from rounds 1..k-1 with the instruction that new ideas must be functionally different from all of them.
- **C — assumption negation.** Sequential; round k sees the same archive, must first state 2-3 unstated assumptions shared by all prior ideas, then generate 6 ideas each violating at least one shared assumption.

Five replicates per arm ("seeds": independent re-runs; the generator exposes no RNG seed, so replicate dispersion is the measured noise floor, as in every experiment in this repo). Total 360 ideas.

**Judging.** Per replicate, all 72 ideas from the three arms are pooled, shuffled by a fixed-seed deterministic PRNG, assigned blind IDs, and clustered by a Claude Opus judge into functional-equivalence classes, with an explicit incoherent/off-problem flag. The judge never sees arm identity, round, or replicate. **Judge noise is measured the same way seed noise is** — replicate 0's pool is clustered by three independent judge instances; between-judge dispersion of per-arm class counts and mean pairwise adjusted Rand index are reported next to every headline number. This is the first experiment in the programme to carry a judge-noise bar, per the lessons document.

**Metrics** (computed by `analyze.py` from raw JSON):
1. Distinct coherent classes per arm per replicate (primary, judge 0).
2. Accumulation curve: distinct classes after rounds 1, 2, 3, 4.
3. Bias-corrected Chao1 per arm per replicate: S_obs + f1(f1-1)/(2(f2+1)).
4. Arm-unique classes within replicate.
5. Incoherent idea count per arm (validity gate).
6. All arm contrasts as paired per-replicate differences, reported as multiples of the SD of the paired differences and of the control arm's replicate SD.

## Kill conditions

If round-1 class counts differ between arms by more than 2 sigma, the manipulation check has failed (arms are not exchangeable at baseline) and the run is discarded, reported as failed, and redesigned. If the judge returns partitions that drop or duplicate more than 2 percent of idea IDs after one retry, the replicate is excluded and the exclusion reported.

## Novelty check

Completed before the first generation call; full report in `docs/self-improvement-loops/03_novelty_audit.md` (claims 5 and 6). **Verdict: partial-prior-art, and the differentiator is confirmed live.** Archive-conditioned generation is a claimed *mechanism* — NoveltyBench's in-context regeneration (arXiv 2504.05228), Denial Prompting (arXiv 2407.09007), Nova (arXiv 2410.14255), and "Anchorless Diversification" (arXiv 2605.30150) names it "population-referential divergence" and calls it a strong baseline. But every published result reports gains at fixed budget — a rate claim; the audit found **no work that measures whether archive conditioning moves the accumulation-curve asymptote versus the rate**, which is exactly what this experiment tests. Likewise no found work operationalizes negation of the assumptions *invariant across the whole generated set* (Denial Prompting bans the last solution's techniques; arm C bans what every idea shares). This experiment therefore claims no mechanism: it is the first application of the asymptote classifier — the programme's strongest surviving claim per the audit — to a mechanism the literature already owns.

---

## RESULTS (written by analyze.py after the pre-registration commit)

Analyzed 360 ideas across 5 replicates. Manipulation check (round-1 exchangeability): means {'A': 6, 'B': 5.6, 'C': 6}, max pairwise gap 0.4 vs 2 sigma = 0.0 — **indeterminate**.

| Arm | Distinct classes /24 (mean ± SD) | Chao1-bc (mean ± SD) | Arm-unique classes | Incoherent (total) | Accumulation r1→r4 |
|---|---|---|---|---|---|
| A iid control | 12.6 ± 0.89 | 17.12 ± 5.84 | 1.4 | 0 | 6 → 9.6 → 11 → 12.6 |
| B archive feedback | 15 ± 1.0 | 26.52 ± 9.93 | 2.8 | 1 | 5.6 → 9.8 → 13.2 → 15 |
| C assumption negation | 14.2 ± 2.17 | 33.54 ± 21.46 | 3.4 | 3 | 6 → 10.6 → 12.6 → 14.2 |

**Contrasts (paired per-replicate differences, as multiples of noise):**

| Contrast | Mean diff | x SD(paired) | x SD(control arm) |
|---|---|---|---|
| H1: B − A, distinct classes | 2.4 | 1.32 | 2.68 |
| H2: B − A, Chao1-bc | 9.396 | 0.69 | 1.61 |
| C − A, distinct classes | 1.6 | 0.77 | 1.79 |
| C − A, Chao1-bc | 16.418 | 0.74 | 2.81 |
| H3: C − B, arm-unique classes | 0.6 | 0.29 | 0.34 |
| C − B, incoherent count | 0.4 | 0.73 | 0.89 |

**Judge noise** (replicate 0, three independent judges): between-judge SD of per-arm distinct counts {'A': 0.0, 'B': 0.0, 'C': 0.58}; pairwise adjusted Rand index [0.916, 0.861, 0.877] (mean 0.885). Every count above is conditional on a judge partition; this is the measured width of that conditionality.

**Verdicts against the pre-registered conditions:**

- H1 (rate): not supported (below 2 sigma)
- H2 (asymptote): supported (Chao1 within 2 sigma: walks faster, not higher ceiling)
- H3 (operator): supported


## Takeaway (hand-written; every number above is from analyze.py)

**Archive feedback accelerates coverage and shows no detectable ceiling move — the pattern the pre-registration predicted — but the honest verdict is more guarded than the code-written lines above, in three specific ways.**

1. **H1 is marginal, not clean.** The B-minus-A effect on distinct classes is +2.4 per 24 ideas, which is 2.68x the control arm's replicate SD but only 1.32x the SD of the paired differences. The pre-registration said "2 sigma of measured replicate noise" without fixing the denominator, and the verdict function took the stricter of the two, so H1 records as not supported. The direction is consistent (B > A in 5 of 5 replicates on the accumulation endpoint: paired diffs 2, 4, 2, 1, 3), and the accumulation curves separate cleanly from round 3 on (B 5.6 → 9.8 → 13.2 → 15.0 versus A 6 → 9.6 → 11 → 12.6). A replication at n = 10 replicates would settle it; at n = 5 the strict bar is not met and that is what the registry records.

2. **H2 "supported" means "not falsified," and the power is low.** Chao1-bc at 24 ideas per arm is extremely noisy (per-arm SDs of 5.8, 9.9, 21.5 — the estimator is singleton-driven and 24 items is a small sample). The observed Chao1 gap (B − A = +9.4) is inside 2 sigma on both denominators, so the pre-registered "walks faster, not higher ceiling" reading stands, but this design could not have detected a ceiling move smaller than roughly a doubling. The next iteration needs more ideas per arm (48+) and multiple estimators (Chao-Bunge, jackknife) per the roadmap's estimator-robustness rule. Note also the point estimates run in the *opposite* direction to a strong version of the claim (Chao1 means 17.1 / 26.5 / 33.5 for A / B / C): if that ordering survived a powered replication, archive conditioning would be moving the ceiling and the E-series "prompt axis is exhausted" conclusion would need the carve-out that H2's falsification clause anticipated. Flagging this against ourselves now, so a powered follow-up is committed either way.

3. **The round-1 manipulation check was mis-specified by us, not failed by the data.** With 6 ideas per round, round-1 distinct classes are capped at 6, and arm A hit exactly 6 in all five replicates, making the pre-registered 2-sigma threshold degenerate (sigma = 0). Round-1 means were 6.0 / 5.6 / 6.0 — the 0.4 gap is two within-round duplicates in arm B under a prompt that is byte-identical across arms at round 1 by construction. We proceed, and record the design lesson: a baseline-exchangeability check must be specified on a measure that is not at ceiling.

**H3 (assumption negation) is directionally as predicted and statistically weak.** C produced the most arm-unique classes (3.4 versus B's 2.8 and A's 1.4) and the highest incoherence (3 versus 1 versus 0 — the predicted coherence tax), but the C-minus-B contrast is 0.29x paired noise. The operator is not refuted and not established; per the roadmap, if a powered replication does not separate it from plain archive feedback, it gets dropped (IDEAFix, arXiv 2606.00875, already reports homogenization persisting across 25 defixation strategies, so the burden of proof is on the operator).

**What is solid regardless of the marginal verdicts:** the judge-noise bar — the programme's first — is tight (between-judge SD at most 0.58 classes on per-arm counts, mean pairwise ARI 0.885, and the arm ordering B ≥ C > A reproduced by all three independent judges), so the measured effects are not judge artifacts; and partition integrity was perfect (0 dropped, 0 duplicated IDs across all 7 judge calls). The instrument works. The effects it measured here are a rate-side acceleration that just misses the strict pre-registered bar, and an asymptote question that needs a bigger sample — both of which are exactly what the rate-versus-asymptote classifier is for.
