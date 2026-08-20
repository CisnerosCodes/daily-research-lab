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

*Pending run.*
