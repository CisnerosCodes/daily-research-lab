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

Analyzed 1536 ideas; 8 replicates analyzed, 0 excluded. Run failed by exclusion rule: False.

Manipulation check (round-1 exchangeability, pooled sigma): means {'A': 5.88, 'B': 5.88}, max pairwise gap 0.0 vs 2 x pooled SD = 0.684 — **PASS**.

| Arm | Distinct /96 (mean ± SD) | Distinct /48 | Chao1-bc | ACE | Jackknife-2 | Incoherent (1st half / 2nd half) |
|---|---|---|---|---|---|---|
| A iid control | 24.62 ± 1.85 | 20 | 27.01 ± 2.9 | 28.32 ± 2.56 | 30.5 ± 4.97 | 1 (0 / 1) |
| B archive feedback | 33.38 ± 2.92 | 26.38 | 36.94 ± 5.31 | 38.46 ± 6.39 | 39.92 ± 9.78 | 9 (4 / 5) |

**Accumulation curves** (mean distinct classes after each round):

| Round | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A iid | 5.88 | 9.5 | 12.75 | 14.75 | 16.12 | 17.25 | 18.62 | 20 | 20.75 | 22 | 22.38 | 22.75 | 23.38 | 23.88 | 24.38 | 24.62 |
| B archive | 5.88 | 10.75 | 15.25 | 19 | 21.88 | 23.62 | 25.62 | 26.38 | 27.5 | 28.62 | 29.5 | 30.62 | 31 | 32.12 | 32.62 | 33.38 |
| gap B−A | 0 | 1.25 | 2.5 | 4.25 | 5.75 | 6.38 | 7 | 6.38 | 6.75 | 6.62 | 7.12 | 7.88 | 7.62 | 8.25 | 8.25 | 8.75 |

**H4 (primary, crossing test):** B's observed distinct classes at 96 ideas minus the maximum of arm A's three estimators, paired per replicate.

| Quantity | n | Mean diff | x SD(paired) | replicates crossed |
|---|---|---|---|---|
| **vs max estimator (pre-registered bar)** | 8 | 2.668 | 0.48 | 6/8 |
| vs Chao1-bc (secondary) | 8 | 6.361 | 1.51 | 7/8 |
| vs ACE (secondary) | 8 | 5.051 | 1.37 | 7/8 |
| vs jackknife-2 (secondary) | 8 | 2.88 | 0.49 | 6/8 |

**H5 (gap curvature):** mean B−A gap at rounds 4/8/12/16 = 4.25 / 6.38 / 7.88 / 8.75. Turnover (peak minus final): 1 (0.84x paired SD, 1x bar). Growth (r16 − r8): 2.375 (0.74x paired SD, 2x bar).

**H6 (estimator stability on arm A, self-audit):** each estimator at 48 vs 96 ideas.

| Estimator | mean at 48 | mean at 96 | increase | x SD(paired) |
|---|---|---|---|---|
| Chao1-bc | 29.03 | 27.01 | -2.019 | -0.37 |
| ACE | 27.9 | 28.32 | 0.423 | 0.15 |
| Jackknife-2 | 33.79 | 30.5 | -3.292 | -0.69 |

**Rate contrast carried forward** (B − A distinct classes at 96 ideas): 8.75 at 2.56x paired SD (4.74x the control arm's SD), paired diffs [15, 6, 10, 10, 10, 9, 4, 6].

**Judge noise** (replicate 0, 3 independent judges): between-judge SD of per-arm distinct counts {'A': 0.0, 'B': 3.0}; pairwise ARI [0.871, 0.869, 0.855] (mean 0.865).

**Judge noise** (replicate 1, 3 independent judges): between-judge SD of per-arm distinct counts {'A': 1.53, 'B': 0.58}; pairwise ARI [0.751, 0.829, 0.787] (mean 0.789).

**Verdicts against the pre-registered criteria:**

- H4 (crossing, primary): walks-faster survives: B's observed count minus A's maximum estimator is 2.668 (0.48x paired SD, below the 2x bar); crossed in 6/8 replicates
- H5 (gap curvature): neither criterion fires (turnover 0.84x vs 1x bar; growth 0.74x vs 2x bar) — gap shape indeterminate at this n
- H6 (estimator stability): stable: no estimator's 48-to-96 increase clears 2x paired SD


## Takeaway (hand-written; every number above is from analyze.py)

**Read the pre-registered verdicts and then read this paragraph, because the two do not say the same thing.** Under the rules fixed before the run, walks-faster survives: the crossing margin against arm A's maximum estimator is +2.67 classes at 0.48x the paired SD, nowhere near the 2x bar. But the direction has inverted since the powered replication, and three separate signals now point the same way. Arm B's mean observed count (33.4) sits above **all three** of arm A's mean estimates (Chao1-bc 27.0, ACE 28.3, jackknife-2 30.5). B crossed A's estimate in 6 of 8 replicates against the maximum estimator and in 7 of 8 against Chao1-bc and ACE. And the B−A gap never turned over: it ends at 8.75, the largest value on the whole 16-round curve. The claim that archive conditioning walks a fixed support faster is the weakest it has been since the programme started making it, and the lab should say that out loud rather than bank the technical pass.

1. **What kept walks-faster alive was dispersion, not a comfortable margin.** The eight per-replicate crossing margins were +9.0, −2.9, +6.9, +8.2, +0.1, +5.9, −5.9, +0.0. Six positive, two negative, mean +2.67 against a paired SD of 5.59. Almost all of that spread comes from arm A's jackknife-2, which ranged from 22.1 to 35.9 across replicates — the estimator carrying the "maximum" role is also the noisiest, so the conservative choice made in the pre-registration is also the choice that made the test hardest to pass. That was the right trade to fix in advance and it is the reason the verdict reads as it does; it is not evidence that B stayed inside A's support.

2. **The escape route the pre-registration reserved is narrower than expected, but it is not closed.** The pre-registration said crossing is necessary but not sufficient, because Chao1, ACE, and jackknife-2 are downward-biased. H6 was written to test whether that bias shrinks with budget, with the honest prediction that the estimators would climb from 48 to 96 ideas. **That prediction was wrong.** Arm A's estimates were stable or slightly lower at 96 than at 48 (Chao1-bc −2.0, ACE +0.4, jackknife-2 −3.3; none clears the 2x bar). Being wrong here matters in a specific direction: a ceiling estimate that stops moving while the other arm's observed count keeps climbing past it is what support extension looks like. The counter-argument that remains, and it is a real one, is that **a stable estimate is not necessarily an accurate one** — all three estimators are lower bounds, and a lower bound can be stably wrong. Stability rules out one explanation for the near-crossing; it does not establish that A's true support is 27 to 30 classes.

3. **Neither arm saturated, so nothing here is an observed plateau.** Over the final five rounds arm A was still adding 0.37, 0.63, 0.50, 0.50, and 0.24 classes per round, and arm B 1.12, 0.38, 1.12, 0.50, and 0.76. Both accumulation curves are decelerating and neither is flat. Every ceiling number in this experiment is an estimate, not a measured asymptote, and the same is true of the E-series ~54. That distinction should be stated wherever those numbers appear.

4. **The rate effect is now measured at three budgets and it keeps growing.** B − A on distinct classes: +2.4 per 24 ideas (pilot), +5.3 per 48 (powered replication), **+8.75 per 96 here, at 2.56x the paired SD and 4.74x the control arm's SD**, positive in 8 of 8 replicates. A rate mechanism operating inside a shared support must eventually give this back as the control arm catches up. Sixteen rounds produced no sign of that: the gap peaked at round 16 in the pooled curve, and in 4 of 8 replicates the final gap was the individual peak. H5 fired neither criterion at n = 8, so this is a shape observation and not a verdict, but the shape is the wrong one for the claim being defended.

5. **The incoherence tax replicates and did not accelerate.** Arm B produced 9 incoherent items to arm A's 1, split 4 in the first half and 5 in the second. The powered replication saw 6 in B over 8 rounds; doubling the horizon did not double the rate. Conditioning pressure costs validity at a roughly constant per-round rate rather than a compounding one, which is a more benign trade than the earlier run suggested.

6. **Judge noise rose and needs watching.** Between-judge SD on arm B reached 3.0 classes on replicate 0 — the largest judge dispersion the programme has measured — and mean pairwise ARI fell to 0.865 and 0.789, the lowest recorded. The 8.75-class effect is still roughly three times the judge-noise width, so the headline is not a partition artifact, but the margin has shrunk from about five times in the previous run. Clustering 192 items in one call is plausibly the cause; all 12 judges passed integrity on the first attempt, so this is dispersion in judgement, not in bookkeeping.

**What this changes, and the next run.** The practical rule for loop builders is unaffected: archive conditioning buys a large and growing coverage advantage at a small validity cost, and any loop that generates without it is paying a duplicate tax that compounds with horizon. What is now in question is the *explanation* the programme has attached to that advantage. The decisive test is no longer another budget doubling on both arms — it is a test with no estimator in the loop at all: **run arm A alone to roughly four times the budget (384 ideas) and ask whether A's observed count ever reaches the 33.4 classes B reached at 96.** If A saturates below B's 96-idea count, walks-faster is dead on observed data and archive conditioning becomes a support-extending intervention, which would put a carve-out into the E-series conclusion exactly where its falsification clause anticipated one. If A climbs past B, the estimators were biased low, walks-faster is vindicated on the strongest possible evidence, and the programme learns that its own instrument understates ceilings at these budgets. That run is single-arm and therefore cheap. It should be next, and the programme should not publish the rate-versus-asymptote classifier as settled until it has been done.
