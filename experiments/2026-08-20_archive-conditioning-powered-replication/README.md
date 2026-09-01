# Archive conditioning, powered: rate effect on trial at n = 10, asymptote question decided by an estimator battery

**Status: PRE-REGISTERED before any generation run.** This section (Question through Kill conditions) was written and committed before the first generation call; the commit timestamp is the evidence. Results appear below the marked line and were written by `analyze.py`, never by hand.

## Question

The pilot (`experiments/2026-08-13_archive-conditioning-rate-vs-asymptote/`) left three open verdicts. Archive feedback raised distinct-class yield in 5 of 5 replicates (+2.4 classes per 24 ideas) but landed at 1.32x the paired-difference SD — below the lab's 2-sigma bar. The asymptote question was underpowered: Chao1 at 24 ideas per arm carried SDs of 5.8 to 21.5, and the point estimates ran in the *opposite* direction to the walks-faster prediction (A 17.1 < B 26.5 < C 33.5). The assumption-negation operator was directional and weak. This replication doubles both replicates and per-arm sample, fixes the two specification defects the pilot exposed (denominator ambiguity, degenerate manipulation check), and decides all three questions under rules fixed in advance.

## Changes from the pilot, each with its reason

1. **Replicates 5 → 10.** Halves the standard error of the paired mean difference and stabilizes the SD estimate the criterion divides by.
2. **Rounds 4 → 8 (48 ideas per arm-replicate, still 6 per round).** Doubles the abundance data the richness estimators consume and extends the accumulation curve into the region where the pilot showed arm B still climbing.
3. **Estimator battery.** Bias-corrected Chao1, ACE, and second-order jackknife, computed per arm per replicate, replacing single-estimator dependence (roadmap estimator-robustness rule).
4. **H1 denominator fixed.** The criterion is the SD of the paired per-replicate differences; the control arm's replicate SD is reported as a secondary descriptive only. The pilot's verdict function took the stricter of two denominators because the pre-registration had not chosen; this one chooses.
5. **Manipulation check re-specified.** Round-1 distinct-class counts compared across arms against 2x the SD pooled over all 30 arm-replicate round-1 counts. Round-1 prompts are byte-identical across arms by construction, so a zero pooled SD with zero gap passes trivially; the pilot's version (sigma from arm A alone, at the 6-class ceiling) was degenerate and is documented there as a design lesson.
6. **Judge noise on two replicates.** Three independent judges on replicates 0 and 1 (pilot: replicate 0 only).
7. **Judge integrity retry made explicit in the harness.** Each judge call is checked in-workflow for every blind ID appearing exactly once; a failure above 2 percent triggers exactly one retry with an emphatic integrity addendum, per the pilot's kill-condition language.

Everything else is held identical to the pilot: same problem (public library permanent non-return), same three arms (A iid control, B archive feedback, C assumption negation), byte-identical generation prompts, same generator (Claude Haiku subagents), same blinding scheme (fixed-seed deterministic PRNG, pooled 3-arm replicate pools), same judge model (Claude Opus, high effort), **iso-idea-budget** as the named control.

## Pre-registered hypotheses, criteria, and falsification conditions

- **H1 (rate).** B − A on distinct coherent classes per 48 ideas, paired per replicate. Supported if mean paired difference >= 2x SD(paired differences). Predicted direction: positive (the pilot saw +2.4 at 24 ideas with curves still diverging). Falsified below that bar — and published as the negative "in-context negative feedback does not clear the lab's evidence bar even at n = 10."
- **H2 (asymptote).** Decided two-sided by the battery: a **ceiling carve-out** is declared if and only if at least 2 of the 3 estimators (Chao1-bc, ACE, jackknife-2) show B − A with |mean paired difference| >= 2x SD(paired) *in the positive direction*. Fewer than 2: **walks-faster stands at doubled power**. A negative-direction 2-of-3 result would be reported as anomalous and unpredicted. This rule exists to prevent post-hoc estimator shopping in either direction.
- **H3 (operator).** C − B on arm-unique classes, paired. Supported if >= 2x SD(paired). Below the bar, **arm C is dropped from the programme**, per the pilot's commitment and the IDEAFix finding (arXiv 2606.00875) that homogenization persists across 25 defixation strategies.

## Kill conditions

1. Manipulation check: max pairwise arm gap in round-1 mean distinct classes > 2x pooled round-1 SD (over all 30 cells) — run discarded, reported as failed, redesigned.
2. Judge integrity: a replicate whose primary judge drops or duplicates more than 2 percent of IDs after the one in-harness retry is excluded and the exclusion reported. If more than 2 of 10 replicates are excluded, the run is reported as failed.
3. Any generation chain that returns fewer than 8 complete rounds makes its replicate unusable for the accumulation analysis; the replicate is excluded and reported.

## Novelty check

Unchanged from the pilot (`docs/self-improvement-loops/03_novelty_audit.md`, claims 5 and 6; verdict partial-prior-art): archive conditioning is a claimed mechanism, the rate-versus-asymptote classification of it is not claimed anywhere found, and this run is a powered replication of the lab's own pilot — labelled as such.

## Metrics

Computed by `analyze.py` from `raw_data.json`: distinct coherent classes per arm per replicate (primary, judge 0); accumulation curves over rounds 1-8; Chao1-bc, ACE, and jackknife-2 per arm per replicate; arm-unique classes; incoherent counts (validity gate); all contrasts as paired differences with SD(paired) multiples (control-arm SD secondary); judge-noise dispersion and pairwise adjusted Rand index on replicates 0 and 1; partition integrity per judge.

---

## RESULTS (written by analyze.py after the pre-registration commit)

Analyzed 1440 ideas; 10 replicates analyzed, 0 excluded. Run failed by exclusion rule: False.

Manipulation check (round-1 exchangeability, pooled sigma): means {'A': 5.7, 'B': 5.7, 'C': 5.9}, max pairwise gap 0.2 vs 2 x pooled SD = 0.86 — **PASS**.

| Arm | Distinct /48 (mean ± SD) | Chao1-bc | ACE | Jackknife-2 | Arm-unique | Incoherent | Accumulation r1→r8 |
|---|---|---|---|---|---|---|---|
| A iid control | 19.2 ± 1.87 | 27.21 ± 9.12 | 26.51 ± 5.69 | 30.18 ± 7.13 | 0.8 | 0 | 5.7 → 9.7 → 12.4 → 14.1 → 15.5 → 17.2 → 18.6 → 19.2 |
| B archive feedback | 24.5 ± 2.59 | 32.22 ± 6.61 | 34.65 ± 5.65 | 39.46 ± 7.79 | 2.5 | 6 | 5.7 → 10.7 → 14.5 → 17.7 → 19.7 → 21.6 → 23.2 → 24.5 |
| C assumption negation | 23.9 ± 3.14 | 34.55 ± 9.27 | 34.32 ± 8.67 | 40.36 ± 10.01 | 5.4 | 8 | 5.9 → 10.7 → 14.6 → 18 → 19.7 → 22 → 23 → 23.9 |

**Contrasts (paired per-replicate differences; the pre-registered criterion is x SD(paired) >= 2):**

| Contrast | n pairs | Mean diff | x SD(paired) | x SD(control) |
|---|---|---|---|---|
| H1: B − A, distinct classes | 10 | 5.3 | 3.0 | 2.83 |
| H2: B − A, Chao1-bc | 10 | 5.014 | 0.52 | 0.55 |
| H2: B − A, ACE | 10 | 8.14 | 1.47 | 1.43 |
| H2: B − A, jackknife-2 | 10 | 9.277 | 1.14 | 1.3 |
| C − A, distinct classes | 10 | 4.7 | 1.95 | 2.51 |
| C − A, Chao1-bc | 10 | 7.343 | 0.67 | 0.81 |
| H3: C − B, arm-unique classes | 10 | 2.9 | 1.62 | 2.98 |
| C − B, incoherent count | 10 | 0.2 | 0.18 | 0.24 |

**Judge noise** (replicate 0, 3 independent judges): between-judge SD of per-arm distinct counts {'A': 1.15, 'B': 1.0, 'C': 1.0}; pairwise ARI [0.84, 0.898, 0.868] (mean 0.869).

**Judge noise** (replicate 1, 3 independent judges): between-judge SD of per-arm distinct counts {'A': 1.0, 'B': 0.58, 'C': 0.58}; pairwise ARI [0.838, 0.847, 0.788] (mean 0.824).

**Verdicts against the pre-registered criteria:**

- H1 (rate): supported (3.0x paired SD)
- H2 (asymptote): walks-faster stands at doubled power (0/3 estimators clear the positive bar)
- H3 (operator): not supported (1.62x paired SD) — ARM C IS DROPPED from the programme per the pilot's commitment


## Takeaway (hand-written; every number above is from analyze.py)

**The classifier has sorted its first mechanism with real power: archive conditioning is a rate intervention.** All three pre-registered questions resolved under rules fixed before the run, with zero excluded replicates, a passing manipulation check, and all 14 judge partitions passing integrity on the first attempt.

1. **H1 is now clean where the pilot was marginal.** B beat A on distinct classes in 10 of 10 replicates (paired diffs 6, 8, 5, 5, 2, 5, 6, 6, 7, 3), mean +5.3 per 48 ideas at 3.0x the paired SD. The effect also *grew* with budget — +2.4 per 24 ideas in the pilot, +5.3 per 48 here — which is exactly the signature of a rate mechanism: the iid control wastes an increasing share of its budget on duplicates as the easy classes fill up, while the archive arm keeps dodging them. In-context negative feedback works, and works harder the longer the run.

2. **H2: walks-faster stands, with one honest residual.** Zero of three estimators cleared the 2x bar (Chao1-bc 0.52x, ACE 1.47x, jackknife-2 1.14x), so under the pre-registered 2-of-3 rule there is no ceiling carve-out and the E-series "prompt-level machinery does not enlarge the support" conclusion survives its sharpest test yet. The residual to state plainly: all three point estimates run positive (+5.0, +8.1, +9.3), the same direction the pilot hinted. A small support extension below our detection threshold is not excluded. The decisive follow-up is a budget-extension test rather than more replicates: arm B's observed 24.5 classes at 48 ideas has not yet crossed the control arm's estimated asymptote (Chao1-bc 27.2); a 16-round run in which B's *observed* count crosses A's *estimated* ceiling would falsify walks-faster directly, with no estimator noise in the way. That test is now the cheapest way to close the question and should be the next run on this thread.

3. **H3: arm C is dropped, and the denominator discipline is the story.** C's arm-unique advantage over B came in at 1.62x the paired SD — below the bar, so per the pilot's commitment the assumption-negation operator leaves the programme. The instructive detail: against the control arm's SD it would have scored 2.98x, which under the pilot's ambiguous dual criterion could have been called a win. Fixing the denominator in advance is precisely what prevented that. For the record, C's descriptive profile was genuinely distinctive — 5.4 arm-unique classes per replicate versus B's 2.5, positive in 9 of 10 replicates (diffs 5, 3, 1, 3, 2, 2, 6, 3, 0, 4) — so the operator is shelved as "suggestive, failed the fixed bar," not "refuted." Reviving it requires a dedicated pre-registered test at roughly double the replicates, and nothing in the current roadmap justifies that spend while the budget-extension test and the EvoTrace instrumentation are open.

4. **A new cost signal at longer horizons.** Both conditioned arms began paying an incoherence tax that the pilot barely showed at 4 rounds: 6 incoherent items in B and 8 in C over 8 rounds, against 0 in the control. Conditioning pressure buys coverage and starts spending validity as the archive grows. The coherence gate caught it, which is what it is for; any production use of archive conditioning should expect this trade to steepen with horizon length.

5. **Judge noise remains far below the effects.** Between-judge SD of per-arm counts is about 1 class, mean pairwise ARI 0.869 and 0.824 on the two triple-judged replicates, and every judge reproduced the B > A ordering. The H1 effect is roughly five times the judge-noise width; nothing here is a partition artifact.

**Programme consequence.** The rate-versus-asymptote classifier now has its first powered, pre-registered sorting: the default diversity move of agentic loops — feed the model its own history and ask for different — accelerates coverage of a fixed support and shows no detectable support extension. Combined with the E9 finding that component swaps do extend the support, the practical rule for loop builders reads: archive conditioning to stop paying the duplicate tax, component swaps to move the ceiling. The budget-extension test above and the EvoTrace instrumentation (roadmap Wedge 1) are the two next runs on this thread.
