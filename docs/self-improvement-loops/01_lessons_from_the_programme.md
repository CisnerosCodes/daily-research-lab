# Lessons from the programme so far: the ideation E-series and Ratchet

Written 2026-08-13 as part of the self-improvement-loops research update. This document reviews the two bodies of work that live *outside* this repo's dated experiment folders: the ideation-diversity E-series (E1 through E9, run in interactive sessions, summarized in `E9_RESULT.md` and the Kaggle handoff brief) and the Ratchet artifact (`ratchet.jsx`, a browser-based verified self-improvement lab). The companion documents in this folder cover the external literature (`02_survey_self_improvement_loops.md`), the prior-art audit of the programme's own claims (`03_novelty_audit.md`), and the review of this repo's 56 training experiments (`04_repo_review.md`).

## What the E-series established, in one paragraph

A prompt-chained idea generator on a single aligned model mode-collapses, measurably: 120 unconstrained domain slots collapse to 37 concepts, six major sectors of human activity receive 1.7 percent of hits, and a random-string "entropy seed" makes collapse worse. Seeding from an external enumerated taxonomy beats free choice (41.0 versus 35.0 distinct functional classes per 48 ideas, three blinded judges, arms nearly disjoint, union 2.14x coverage). Chao1 species-richness estimation put the single-system ceiling at ~54 functional idea classes, and E9 then showed the ceiling belongs to the *system*, not the problem: at equal budget, a union of models created 9 new idea classes outside a fixed 35-class codebook, external knowledge injection created 7, problem reframings created 4, and the best single-model prompt configuration created 1. Prompt engineering is the exhausted axis; component swapping is the live one.

## The good — patterns worth keeping

**1. Measurement preceded intervention.** The programme spent its first experiments establishing that collapse is real, quantified, and a sampling failure rather than a knowledge failure (the ban-clause result: 1.7 percent sector coverage jumping to 38.3 percent under one banned-cluster clause is the single cleanest demonstration in the series). Most public work on "LLM creativity" starts with an intervention and a vibe; this started with an instrument.

**2. The escape metric.** E9's scoring — new classes created *outside* a fixed reference codebook of already-known classes — is a genuinely better measure than any within-batch diversity number, because it is immune to the failure mode E8 caught: stacked prompt modifications gamed the diversity metric while producing 21 percent incoherent output. Diversity metrics reward difference; the escape metric rewards *new territory*. Keep it.

**3. Self-flagged traps.** The "raises the ceiling versus walks the same support faster" distinction, and the requirement that every proposal say which it is and how to tell, is the programme's sharpest intellectual asset. It is a falsifiability discipline that most published diversity work does not have; the novelty audit (companion document) indicates nobody in the literature is making this distinction explicitly.

**4. Prior-art search applied to the outputs, not just the method.** E9 did not stop at "21 new idea classes"; it ran a real prior-art search on the eight most distinctive mechanisms and found five already claimed. That killed the temptation to advertise idea counts as if they were inventions, and it surfaced the one genuinely actionable survivor (the third-party logistics piggyback with the USPS Carrier Alert precedent). The willingness to let the search kill the headline is the credibility of the whole lab.

**5. Kill conditions stated in advance.** The Kaggle handoff names a programme-level kill condition (if nothing beats a Verbalized-Sampling API baseline at iso-token-budget by 2 sigma, abandon the local stack and publish the negative). Very few solo research programmes write down the condition under which they stop.

## The bad — weaknesses that must be fixed before any of this turns heads

**1. Judge-conditional numbers presented as facts.** Every richness count in the series — 37, 41 versus 35, 54, the 9/7/4/1 escape scores — is conditional on one judging protocol, and mostly on *one judge instance*. E9's own limitations section concedes this, and one of its proposals ("redefine the equivalence relation") makes the deeper point: the partition, not the ideas, defines the count. Nothing in the series measures judge-repartition noise the way the repo's training experiments measure seed noise. Until distinct-class counts carry a judge-noise bar the way losses carry a seed-noise bar, a skeptical reviewer can attribute every effect to the judge. This is the programme's largest single methodological hole, and it is cheap to fix: same ideas, k independent judge instances, report between-judge dispersion of every headline count, and adjusted agreement (e.g., ARI) between partitions.

**2. n = 1 problem, n = 1 session.** The entire ceiling story rests on one problem (clinic no-shows) and one codebook derived from one E8 judge. The 9-to-1 model-union-versus-control gap is too large to be judge noise, but it could still be problem-specific. The Kaggle handoff already prescribes the fix (5 problem spaces x 5 seeds); it has not happened yet.

**3. Confounded arms, acknowledged but unresolved.** E9's model-union arm varied model size along with model identity, so "different distribution" and "more capable model" are not separated. The handoff proposes the clean version (three open-weight families at fixed 7-8B). Until then, the honest claim is "swapping generators moved the ceiling," not "distributional diversity moved the ceiling."

**4. Chao1 used outside its assumptions, knowingly.** Chao1 assumes a closed population and roughly homogeneous detection; the measured yield decay (4.80 to 2.60 new classes per run) is itself evidence of a long-tailed heterogeneous regime where Chao1 is a biased lower bound. The proposal list already contains the remedy (Chao-Bunge, jackknife, Hill-number extrapolation, larger n); the writeups should stop printing "~54" without an estimator-uncertainty qualifier.

**5. No seed-noise floor on the E9 escape scores.** The repo's non-negotiable rule — five seeds minimum, effects as multiples of measured seed noise — was not applied to the E9 arms (one run per arm, one judge per arm). The 9-versus-1 gap is presumably far above any plausible floor, but "presumably" is exactly what the rule exists to eliminate.

## Ratchet, reviewed as a self-improvement loop

Ratchet is a FunSearch-descendant compressed into a browser artifact: three proposer roles (refiner, hybridizer, explorer at temperatures 0.3/0.7/1.0) mutate a champion program; a mechanical evaluator on seeded instances is the pawl; guidance blocks inside the proposer prompts carry win-rate statistics, are selected by a softly-greedy bandit, and are rewritten by a meta agent every three generations. It is honest in the ways that matter — measured live baselines, a held-out and a distribution-shifted verdict set the agents never optimize against, an explicit statement that goals without mechanical verifiers are out of scope.

Its limitations, read against the literature the survey document covers:

- **No diversity maintenance.** The population is a top-8 truncation by training fitness with exact-normalized-string dedup. This is precisely the architecture the E-series showed has a ceiling: a single aligned generator, selection pressure toward one attractor, and no mechanism that rewards *different* over *better*. FunSearch used an island model for exactly this reason; MAP-Elites-style archives (AlphaEvolve lineage) keep stepping-stones alive that greedy truncation kills. Ratchet's stagnation prompt ("abandon the current paradigm entirely") is a plea to the generator to be diverse; the E-series result predicts pleading does not work — the population structure must enforce it.
- **The guidance-block bandit is confounded.** Blocks are credited with the fitness delta of whatever candidate they happened to accompany, but role and temperature are assigned independently, so a block's "win rate" mixes its own effect with the role mix it was drawn into. With uses in the single digits, the bandit is selecting on noise. The E8 lesson (12 prompt modifications, 2 apparent winners, none replicated) applies verbatim to guidance-block selection.
- **Single-model proposers.** All three roles call the same model. E9's 9-to-1 result says the cheapest available ceiling-raise for Ratchet is proposer-model union, and nothing in the architecture prevents it.
- **The champion advances on training fitness alone.** Held-out and shift sets exist but only at verdict time. A candidate that overfits the 24 training seeds is crowned and steers all subsequent refinement. Cheap fix: advance on train, but log held-out at every crowning so overfitting-drift is visible in the strip.

None of these are fatal; all four are upgrade paths, and the first and third connect Ratchet directly to the E-series findings. The synthesis document (`05_roadmap.md`) develops this: the E-series built an instrument for measuring generator-support exhaustion, Ratchet is a loop whose plateau is plausibly *caused* by generator-support exhaustion, and nobody in the surveyed literature instruments an evolutionary code-search loop with support-richness estimators to show the plateau is a diversity event. That is the bridge between the two halves of this programme.

## Rules the E-series should inherit from this repo

The repo's 56 training experiments run under discipline the E-series only partially adopted. The union of both rulebooks, applied to all future ideation work:

1. Five seeds (independent replicate runs) minimum; effects reported as multiples of measured replicate noise.
2. Judge-noise measured the same way seed noise is: k independent judge partitions of the same idea pool, dispersion reported next to every count.
3. Named control for every comparison (iso-idea-budget, iso-token-budget, iso-wall-clock), never two axes varied silently.
4. Novelty check before code, recorded in the registry with verdict.
5. Escape-style metrics (new territory versus fixed codebook) preferred over within-batch diversity metrics wherever a codebook exists.
6. A validity/coherence gate on every arm, scored blind, so that territory bought with incoherence (E9 arm D: 8 of 48 incoherent) is visibly taxed.
7. Kill conditions written before the run.
