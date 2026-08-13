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

A dedicated prior-art audit (to be recorded in `docs/self-improvement-loops/03_novelty_audit.md`, claims 5 and 6) is running before the first generation call; the generation run is gated on its verdict, and the verdict will be recorded here and in the registry row before results are added. Expected nearest neighbours: iterative "avoid what you already said" prompting in fragments; the differentiator under test is the rate-versus-asymptote decomposition of the intervention's effect.

---

## RESULTS (written by analyze.py after the pre-registration commit)

*Pending run.*
