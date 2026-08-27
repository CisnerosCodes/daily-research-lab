# Budget extension: does archive conditioning ever cross the control arm's estimated ceiling?

**Status: PRE-REGISTERED before any generation run.** This section (Question through Metrics) was written and committed before the first generation call; the commit timestamp is the evidence. Results appear below the marked line and were written by `analyze.py`, never by hand.

## Question

The powered replication (`experiments/2026-08-20_archive-conditioning-powered-replication/`) settled the rate question and left one residual. Archive feedback beat the iid control on distinct functional classes in 10 of 10 replicates, +5.3 per 48 ideas at 3.0x the paired SD, and the effect grew with budget (+2.4 per 24 in the pilot). No estimator cleared the asymptote bar, so "walks the same support faster" stood. But all three point estimates ran positive, and its own takeaway named the decisive next test rather than more replicates:

> arm B's observed 24.5 classes at 48 ideas has not yet crossed the control arm's estimated asymptote (Chao1-bc 27.2); a 16-round run in which B's *observed* count crosses A's *estimated* ceiling would falsify walks-faster directly, with no estimator noise in the way.

This is that run. It doubles the horizon again — 16 rounds, 96 ideas per arm-replicate — and asks a question that does not depend on comparing two noisy estimators to each other.

## The logic of the crossing test, stated honestly before the run

Under walks-faster, arms A and B draw from one common support of size `S`. B spends its budget more efficiently, so B's observed class count rises faster, but it can only ever approach `S` — never pass it. So if B's **observed** count exceeds a defensible estimate of A's support, walks-faster is in trouble.

The asymmetry that must be stated up front, because it decides how much any outcome is worth:

- **Chao1, ACE, and jackknife-2 are downward-biased estimators of richness** under detection heterogeneity. Each is closer to a lower bound on `S` than to `S` itself. Therefore *crossing an estimate is a necessary but not sufficient condition for support extension*: B could pass A's Chao1 and still be inside A's true support. A crossing result is evidence, not proof, and this document will not be allowed to call it proof afterwards.
- **Failing to cross is the stronger direction.** If B's observed count at 96 ideas sits below even a lower-bound estimate of A's support, B has not exhausted the territory the control arm already commands, and there is no case at all for extension.

To keep the bar conservative and to foreclose estimator shopping, the primary criterion uses **the maximum of the three estimators of arm A** — the least downward-biased of them per replicate — as the ceiling B must clear. Individual per-estimator crossings are reported as secondary descriptives only.

## Design

Same problem (public library permanent non-return), same generator (Claude Haiku subagents), same blinding scheme (fixed-seed deterministic PRNG, pooled blind IDs per replicate), same judge model (Claude Opus, high effort), **iso-idea-budget** as the named control. Changes from the powered replication, each with its reason:

1. **Rounds 8 → 16 (96 ideas per arm-replicate).** The whole point: the crossing test needs B far enough along the accumulation curve to have a chance of passing A's estimated ceiling, and needs the curve long enough for a rate effect to show its turnover.
2. **Arm C dropped.** The powered replication's H3 came in at 1.62x the paired SD, below the pre-registered bar, and the pilot had committed to dropping it in that event. It is gone, not quietly rested — reviving it would need its own pre-registered run.
3. **Replicates 10 → 8.** Two arms instead of three at double the rounds keeps the run at roughly the previous agent count (256 generation calls plus 12 judge calls, against 254 total last time). The primary test here is a crossing, not a small-effect contrast, so the loss from 10 to 8 replicates is accepted deliberately and recorded here rather than discovered later.
4. **Judge pools 144 → 192 ideas.** Two arms x 96. Larger single-call clustering load; the in-harness integrity check with one retry is unchanged and the kill condition below covers the added risk.
5. **Judge noise still measured on two replicates** (0 and 1, three independent judges each).

## Pre-registered hypotheses, criteria, and falsification conditions

- **H4 (crossing — primary).** Per replicate, compute `B_observed_distinct(96) − max(Chao1-bc(A), ACE(A), jackknife-2(A))`. **Walks-faster is falsified** if the mean of these paired differences is positive and at least 2x the SD of the paired differences. Below that bar, walks-faster survives its most direct available test. Predicted direction: negative or near zero (the programme's own claim is that it should *not* cross).
- **H5 (gap curvature).** The B − A gap in distinct classes measured at rounds 4, 8, 12, and 16. A rate mechanism operating on a common support predicts the gap rises and then **turns over** as the control arm catches up near the shared ceiling; a support extension predicts continued growth. Two criteria, both computed:
  - **Turnover** declared if the round-16 gap is below the maximum observed gap by at least 1x the SD of the paired (max-gap minus final-gap) differences.
  - **Monotone growth** declared if `gap(r16) − gap(r8)` is at least 2x the SD of those paired differences.
  Prior points for context: +2.4 at 24 ideas (pilot), +5.3 at 48 (powered replication). Both are in the rising phase; this run covers the region where the turnover must appear if walks-faster is right.
- **H6 (estimator stability — self-audit).** For arm A only, each estimator computed on the first 48 ideas and again on all 96, paired within replicate. **Stable** if the mean paired increase is below 2x SD(paired). Honest prediction: it will *not* be stable — these estimators are known to climb with sample size — in which case every "estimated ceiling" number this lab has published, including the E-series ~54, carries a budget-dependence caveat that must be stated wherever those numbers appear. Predicting against ourselves here is the point.

## Kill conditions

1. Manipulation check: max pairwise arm gap in round-1 mean distinct classes greater than 2x the SD pooled over all 16 arm-replicate round-1 counts — run discarded, reported as failed, redesigned.
2. Judge integrity: a replicate whose primary judge drops or duplicates more than 2 percent of the 192 IDs after the one in-harness retry is excluded and the exclusion reported. More than 2 of 8 replicates excluded fails the run.
3. Any generation chain returning fewer than 16 complete rounds makes its replicate unusable; the replicate is excluded and reported.

## Novelty check

Unchanged from the pilot and the powered replication (`docs/self-improvement-loops/03_novelty_audit.md`, claims 5 and 6; verdict partial-prior-art). Archive conditioning is a mechanism the literature already owns (NoveltyBench arXiv 2504.05228, Denial Prompting arXiv 2407.09007, Nova arXiv 2410.14255, Anchorless Diversification arXiv 2605.30150). The rate-versus-asymptote classification of it is the lab's own frame, and this run is the third instalment of the lab's own thread, labelled as such. KnowSum (arXiv 2506.02058) remains the must-cite nearest neighbour for unseen-species estimation on LLM outputs.

## Metrics

Computed by `analyze.py` from `raw_data.json`: distinct coherent classes per arm per replicate (primary, judge 0); accumulation curves over rounds 1-16; Chao1-bc, ACE, and jackknife-2 per arm per replicate at 48 and at 96 ideas; the H4 crossing margin against A's maximum estimator and against each estimator separately; the H5 gap trajectory; incoherent counts (validity gate) reported per arm and per half of the run; judge-noise dispersion and pairwise adjusted Rand index on replicates 0 and 1; partition integrity per judge call.

---

## RESULTS (written by analyze.py after the pre-registration commit)

Not yet run.
