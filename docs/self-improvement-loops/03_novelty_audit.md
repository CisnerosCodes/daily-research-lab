# Prior-art audit: the programme's seven core claims

Written 2026-08-13/14 by a dedicated audit agent (Claude Opus) running live web searches, as part of the self-improvement-loops research update. The verdicts below gate the archive-conditioning experiment in `experiments/2026-08-13_archive-conditioning-rate-vs-asymptote/`.

## Verification caveat (read first)

**WebFetch was blocked for every domain attempted** in the audit session (`arxiv.org`, `api.semanticscholar.org`, `api.openalex.org`, `openreview.net`, and others were denied by the egress proxy policy). Consequence: **no primary PDF was read.** Every citation below rests on search-engine retrieval of abstract text and paper pages. Titles, arXiv identifiers, author names, and abstract-level claims are reliable at that level. Method-level details are **not** independently verified. Papers where the method-level detail is load-bearing for a verdict are flagged with **[VERIFY BY HAND]**.

A second caveat: several decisive papers carry 2026 arXiv identifiers (2506.02058 is 2025; 2604.12015, 2602.23413, 2604.24372, 2605.30150, 2606.10302 are 2026). These are recent enough that a solo researcher may not have encountered them, and they are the ones that do the most damage.

---

## Claim 1 — Species-richness estimators for LLM output diversity

**Verdict: PARTIAL, with a small and uncomfortable delta. The core method is claimed.**

The nearest neighbour is not close — it is nearly on top of the claim.

**KNOWSUM — "Evaluating the Unseen Capabilities: How Many Theorems Do LLMs Know?"** (Xiang Li, Jiayi Xin, Qi Long, Weijie J. Su), arXiv [2506.02058](https://arxiv.org/abs/2506.02058), June 2025. KnowSum is a statistical framework that estimates the unobserved portion of an LLM's knowledge by extrapolating from the appearance frequencies of observed instances, using the smoothed Good-Turing estimator. It is demonstrated on **three** applications, and the third is explicitly **"measuring output diversity"** — retrieved descriptions state it estimates how many semantically distinct outputs a model could generate for open-ended prompts, by prompting repeatedly (order of 100 times), sorting results into common and rare responses, and extrapolating the unseen portion. Reported headline: 50 to 80 percent of LLM-encoded knowledge remains unobserved under standard prompting. **[VERIFY BY HAND — the exact framing and scale of the diversity application is the single most important thing to check in this entire audit.]**

**UCS — "Estimating Unseen Coverage for Improved In-Context Learning"** (Jiayi Xin, Xiang Li, Evan Qiang, Weiqing He, Tianqi Shang, Weijie J. Su, Qi Long), arXiv [2604.12015](https://arxiv.org/abs/2604.12015), ACL 2026 Findings. Same group. Induces discrete latent clusters from embeddings and estimates the number of **unrevealed clusters** via a smoothed Good-Turing estimator on the empirical frequency spectrum. This is the "functional idea classes plus unseen-class estimation" construction, applied to demonstration selection rather than ideation.

Supporting lineage, all pointing the same way:
- **"Blind-Spot Mass: A Good-Turing Framework for Quantifying Deployment Coverage Risk in Machine Learning Systems"**, arXiv [2604.05057](https://arxiv.org/abs/2604.05057), April 2026. Good-Turing unseen-species estimation of under-supported regions of an operational distribution.
- **"Knowing when to stop: insights from ecology for building catalogues, collections, and corpora"** (Jan Hajič jr., Fabian Moss), arXiv [2507.14614](https://arxiv.org/abs/2507.14614), DLfM 2025. Applies **Chao1 by name** to estimate repertoire coverage of a corpus, reporting upper bounds of 50 to 80 percent. Not LLMs, but it establishes that "apply Chao1 to a generated or collected artifact population and read the observed-versus-estimated gap as a coverage diagnostic" is a published move.
- **"The Rest is Silence: Leveraging Unseen Species Models for Computational Musicology"**, arXiv [2507.14638](https://arxiv.org/abs/2507.14638).
- **"On the Role of Unobserved Sequences on Sample-based Uncertainty Quantification for LLMs"**, arXiv [2510.04439](https://arxiv.org/abs/2510.04439), UncertaiNLP at EMNLP 2025. Argues the probability of unobserved sequences is the missing term in LLM entropy estimation.
- **"Benchmarking Linguistic Diversity of Large Language Models"** (Guo, Shang, Clavel), arXiv [2412.10271](https://arxiv.org/abs/2412.10271), TACL 2025. Imports **Hill numbers** — the ecological diversity family Chao1 sits inside — into LLM output evaluation as "effective vocabulary size." Lexical and syntactic, not idea-class, and no asymptotic estimation.

**Exact remaining delta.** Three things are the programme's and nobody else's, and they are all narrow:
1. **Chao1 specifically** rather than smoothed Good-Turing. This is a near-worthless delta — they are the same estimator family answering the same question, and reviewers will treat the substitution as a variant, not a contribution.
2. The unit of "species" being a **blinded-judge-assigned functional idea class in an open-ended ideation task**, rather than theorem names, latent embedding clusters, or vocabulary types.
3. Framing the observed-versus-asymptote gap as a **mode-collapse diagnostic** rather than an evaluation-completeness correction. KnowSum's motivation is "our benchmarks undercount what the model knows"; ours is "the model cannot reach past this ceiling." Same statistic, opposite rhetorical direction.

**Do not claim species-richness estimation for LLM output as a novel contribution.** It is claimed. The programme may claim its instantiation on judged idea classes, and it should cite KnowSum in the first paragraph rather than the related work, because a reviewer who finds it later will assume it was hidden.

---

## Claim 2 — Rate-versus-asymptote distinction for diversity interventions

**Verdict: PARTIAL, and this is the strongest surviving claim.**

Nobody found fits an accumulation curve, extrapolates its asymptote, and uses the change in that asymptote to classify interventions. Several papers circle it.

**Closest neighbour on the curve itself: "Homogenizing effect of large language models (LLMs) on creative diversity: An empirical comparison of human and ChatGPT writing"**, [ScienceDirect S294988212500091X](https://www.sciencedirect.com/science/article/pii/S294988212500091X). This paper introduces a **"diversity growth rate"** measure that "dynamically captures the extent to which each additional creative output contributes to collective diversity," across 2,200 college admissions essays in three preregistered studies, and reports that the human-versus-GPT-4 gap **widens as more essays are included**, and persists "despite efforts to enhance AI-generated content through both prompt and parameter modifications." This is an accumulation curve, and the finding that prompt and parameter tweaks fail to close the gap is a rate-versus-support result in everything but name. **What they measure is the slope. They do not fit or extrapolate the asymptote, and they do not use it as a classifier for interventions.** **[VERIFY BY HAND — this is the single paper most likely to have quietly done the analysis. Read the methods section for how the growth rate is estimated and whether any extrapolation appears.]**

**Closest neighbour on the taxonomy: "Where You Inject Diversity Matters: A Unified Framework for Diverse Generation"** (Cheng Zhang, Rui Xin, Chudi Zhong), arXiv [2606.10302](https://arxiv.org/abs/2606.10302). Its axis is **where** the diversity source enters — Level 0 no injection, Level 1 surface-level (explicitly including "a random seed string, a nonce token, or an arbitrary identifier," which is precisely the hex entropy seed tested in E1, and which they classify as carrying no semantic content about the output), Level 2 specification-level. Their diagnostic is a **transmission score**: how effectively variation in the source reaches the final output. That is an axis about **signal propagation**, not about support size. A Level 2 method that walks a fixed support faster and one that enlarges the support score identically under transmission.

**Closest neighbour on support language: "Geometry of Knowledge Allows Extending Diversity Boundaries of Large Language Models"** (Bystroński, Han, Chawla, Kajdanowicz), arXiv [2507.13874](https://arxiv.org/abs/2507.13874). Explicitly claims to "systematically expand the model's reachable semantic range" via manifold traversal conditioning. The vocabulary of support expansion is present; the measurement is diversity scores, not an asymptote. **[VERIFY BY HAND]**

Also relevant and pointing at the distinction without naming it:
- **"Multi-LLM Systems Exhibit Robust Semantic Collapse"**, arXiv [2605.17193](https://arxiv.org/abs/2605.17193). Lexical diversity measures rise under interventions such as higher temperature, yet semantic trajectories remain stable or decrease, with no evidence any intervention attenuates semantic convergence across 62 baseline comparisons. This is the distinction, observed, in a different metric pair (lexical versus semantic rather than rate versus asymptote).
- **"Sampling More, Getting Less: Calibration is the Diversity Bottleneck in LLMs"**, arXiv [2605.11128](https://arxiv.org/abs/2605.11128). Attributes collapse to order and shape miscalibration — "valid alternatives are present in the model distribution" but are not exposed. This is a mechanistic argument that most decoding interventions are rate interventions, made at the token level.
- **"More Is Not More: What Matters for Diversity in LLM Opinions?"**, arXiv [2607.20429](https://arxiv.org/abs/2607.20429). Explicitly notes the landscape is fragmented with methods evaluated in isolation under incomparable metrics — that is the gap being filled, stated by someone else without filling it.
- **"Evaluating the Diversity and Quality of LLM Generated Content"**, arXiv [2504.12522](https://arxiv.org/abs/2504.12522). Offers a rival explanatory axis that must be addressed: RL-tuned models show lower lexical diversity but greater **effective semantic diversity**, "not from increasing diversity among high-quality outputs, but from generating more high-quality outputs overall." A reviewer will ask whether asymptote shifts are quality-threshold artifacts.

**Exact remaining delta:** fitting or extrapolating the **asymptote** of a species-accumulation curve over judged idea classes, and using the estimated asymptote — not the slope, not a single-batch distinct count — as the classifier that sorts interventions into support-walking versus support-enlarging. No paper found does this. **This is defensible as a novel methodological contribution.** It is also the contribution that makes claims 1, 4, and 5 coherent, and the paper should be built around it rather than around Chao1.

---

## Claim 3 — Taxonomy-conditioned seeding

**Verdict: PARTIAL. The mechanism is claimed. Only the specific comparison survives.**

**LAB: Large-Scale Alignment for ChatBots**, arXiv [2403.01081](https://arxiv.org/abs/2403.01081), MIT-IBM Watson AI Lab, is the neighbour not previously on the list and the one that hurts most. LAB is driven by a manually curated taxonomy (knowledge, foundational skills, compositional skills, split into granular levels with tasks at leaf nodes), and a retrieved description states the taxonomy-driven approach "enabl[es] targeted coverage of the support of the teacher model distribution around individual leaf nodes," explicitly motivated by the observation that random seed selection makes teacher models "generate synthetic data from dominant modes and ignore the long tail of interesting tasks." That is the same mechanism, motivation, and vocabulary — for alignment data rather than ideation, with a hand-built taxonomy rather than a public standard. Shipped as InstructLab.

The neighbours previously listed, with their exact mechanisms:
- **SimpleStrat**, arXiv [2410.09038](https://arxiv.org/abs/2410.09038). Confirmed: **auto-stratification** — the LLM itself is prompted to identify useful partitions of the solution space. Three stages: auto-stratification, heuristic estimation, probabilistic prompting. Evaluated on CoverageQA by KL divergence from uniform over valid ground-truth answers. **The strata are internal, not external.** The delta against SimpleStrat is real and is exactly the free-choice-versus-external-enumeration contrast.
- **PersonaHub / "Scaling Synthetic Data Creation with 1,000,000,000 Personas"**, arXiv [2406.20094](https://arxiv.org/abs/2406.20094). One billion personas curated from web data acting as "distributed carriers of world knowledge." External enumerated seed set at extreme scale, though a mined set rather than a maintained standard taxonomy.
- **Verbalized Sampling**, arXiv [2510.01171](https://arxiv.org/abs/2510.01171). Not taxonomy conditioning at all — it asks the model to verbalize a distribution over responses with probabilities. Reports 1.6 to 2.1 times diversity gains in creative writing, and attributes mode collapse to **typicality bias in preference data**. It is a competing explanation for the collapse observation, not a competing mechanism for the intervention.
- **"Where You Inject Diversity Matters"** Level 2, arXiv [2606.10302](https://arxiv.org/abs/2606.10302). Their proposed methods are "fully automated specification-level generation" — the specifications are **model-generated**, not externally sourced. Same internal-versus-external split as SimpleStrat.

Two more worth knowing about:
- **"Anchorless Diversification for Parallel LLM Ideation"**, arXiv [2605.30150](https://arxiv.org/abs/2605.30150), May 2026. This paper poses the same question — can anchorless methods rival methods depending on observed seed ideas — and finds **semantic direction stratification** (one planning call in which the model proposes broad semantic directions, budget split evenly across them) gives the best diversity-quality-compute frontier. It is the strongest published version of the arm the E-series beat, and it concludes in favor of self-generated structure. It must be engaged directly.
- **"The Alien Space of Science: Sampling Coherent but Cognitively Unavailable Research Directions"** (Artiles, Weiss, Brinkmann, Rahwan, Schölkopf, Pal, Larochelle, Goyal, Rahaman), arXiv [2603.01092](https://arxiv.org/abs/2603.01092). Decomposes literature into "idea atoms," builds a shared external vocabulary, then learns a coherence model and an **availability** model, and samples combinations that are coherent but improbable under what researchers would propose. External enumerated structure driving ideation into unoccupied regions.

**Exact remaining delta:** conditioning on a **standardized, externally maintained, publicly enumerated taxonomy** (NAICS, O\*NET, OpenAlex topics) as the seed set for open-ended ideation, plus the head-to-head against free choice showing **near-disjoint idea classes** (union at 2.14 times coverage). The near-disjointness result is the valuable part and nothing was found reporting it. The mechanism is not novel; the empirical contrast is publishable as a finding, not as a method.

---

## Claim 4 — Model-union for support expansion

**Verdict: PARTIAL. "Ensembles are more diverse" is thoroughly claimed. "Union richness exceeds single-model asymptote" is not — and part of the literature disputes the premise.**

Prior art establishing that heterogeneous model pooling raises observed diversity:
- **"How Diversely Can Language Models Solve Problems? Exploring the Algorithmic Diversity of Model-Generated Code"**, arXiv [2503.00691](https://arxiv.org/abs/2503.00691), EMNLP 2025 Findings. The closest neighbour. It clusters solutions into algorithmic equivalence classes via LLM reasoning, defines metrics (DA@K, EA) that **quantify the number of unique algorithms**, finds model-generated solutions have low algorithmic diversity, and then finds that "code diversity can be enhanced with the help of **heterogeneous models** and setting temperature beyond 1.0." That is the same experiment — equivalence-class counting, model-union intervention, observed gain — one domain over and **without any asymptote**. **[VERIFY BY HAND — check whether EA involves any unseen-class extrapolation. If it does, the claim 4 delta shrinks toward zero.]**
- **"Diversity is the Strength of the AI Crowd"** (Aitchison, Jeen, Shevlane, Day), arXiv [2606.29661](https://arxiv.org/abs/2606.29661), ICML 2026 workshop. Frontier LLMs make highly correlated predictions; the strength of the crowd comes from complementary errors, not more sampling.
- **"Quantifying Diversity of Thought: A Predictive Law of Weighted LLM Ensemble Lift"** (Junade Ali), arXiv [2607.17384](https://arxiv.org/abs/2607.17384). Formal decomposition of ensemble lift into rescue and damage masses, 767,520 inferences over ten open-weight models.
- **"Are Diversity Metrics Measuring Diversity? A Capability-Controlled Audit of Majority-Vote Gain in LLM Ensembles"**, arXiv [2607.20768](https://arxiv.org/abs/2607.20768). A methodological audit worth reading before claiming an ensemble diversity effect.
- **"Epistemic diversity across language models mitigates knowledge collapse"** (Hodel, West), arXiv [2512.15011](https://arxiv.org/abs/2512.15011), and the earlier arXiv [2510.04226](https://arxiv.org/abs/2510.04226). Explicitly ecology-inspired; finds ecosystem diversity mitigates collapse **only up to an optimal level**, and that a few diverse models "fail to express the rich mixture of the full, true distribution."
- **Mixture-of-Agents** and successors, including **"Mixture-of-Models: Unifying Heterogeneous Agents via N-Way Self-Evaluating Deliberation"**, arXiv [2601.16863](https://arxiv.org/abs/2601.16863).

Prior art **against** the premise, which must be confronted:
- **"We're Different, We're the Same: Creative Homogeneity Across LLMs"** (Emily Wenger, Yoed Kenett), arXiv [2501.19361](https://arxiv.org/abs/2501.19361). LLM responses are far more similar to other LLM responses than human responses are to each other, **after controlling for response structure**, across a broad set of models. The paper's whole point is that switching models does not help.
- **"Large language models are homogeneously creative"**, [PNAS Nexus 5(3) pgag042](https://academic.oup.com/pnasnexus/article/5/3/pgag042/8529001).
- **"Human diversity fuels collective creativity that large language models cannot simulate or sustain"** (Mengchen Dong, Hiromu Yakura), arXiv [2607.26899](https://arxiv.org/abs/2607.26899), July 2026. Preregistered. They simulated a full writer pool using personas from real backgrounds, **three model families**, native-language prompting, and elevated temperature — and the simulation could not reproduce the human collective diversity. This is a direct, well-powered negative result on model-union support expansion. **[VERIFY BY HAND — read what exactly failed to be reproduced, since their target is human diversity rather than absolute class count.]**
- **"Diversity Collapse in Multi-Agent LLM Systems: Structural Coupling and Collective Failure in Open-Ended Idea Generation"**, arXiv [2604.18005](https://arxiv.org/abs/2604.18005), ACL 2026 Findings, 10,000-plus research proposals. Reports a compute-efficiency paradox where stronger aligned models give diminishing marginal diversity, and group-size scaling gives diminishing returns.

**Exact remaining delta:** measuring **union richness against a single model's estimated asymptote** rather than against its observed count. Nobody found doing this. Every ensemble-diversity paper compares observed to observed. The E9 test — 9 new classes outside a fixed 35-class codebook from model union versus 1 from the best single-model prompt configuration — is the right shape and is unclaimed.

Be aware that 2501.19361 and 2607.26899 are strong contrary evidence and the E9 result cuts against them. That is not fatal, but "cross-model class overlap is low" is an empirical claim that two well-executed papers dispute, so the defence has to rest on the difference in unit of analysis (blinded-judge functional idea classes over enumerated domains, versus standardized creativity-test items and metaphor tasks). Say so explicitly.

---

## Claim 5 — Archive-conditioned generation for ideation

**Verdict: EXISTS. The mechanism is thoroughly claimed. Dead as a novel mechanism.**

- **NoveltyBench** (arXiv [2504.05228](https://arxiv.org/abs/2504.05228)) includes **in-context regeneration**: after each generation the model is explicitly asked for a different answer while all previous answers stay in context. Outputs are grouped into equivalence classes by a fine-tuned DeBERTa model, reported as `distinct_k`. That is the archive-conditioning mechanism and the equivalence-class counting, published in April 2025.
- **Denial Prompting**, in **"Benchmarking Language Model Creativity: A Case Study on Code Generation"** (Yining Lu et al.), arXiv [2407.09007](https://arxiv.org/abs/2407.09007), NAACL 2025. Incrementally bans the techniques used in the model's own previous solutions, scored by **NeoGauge**, with the NeoCoder dataset released. Finding: even GPT-4 falls short of human-like creativity, and advanced reasoning strategies (MCTS, self-correction) gave no significant creativity improvement.
- **QDAIF: Quality-Diversity through AI Feedback** (Bradley, Dai, Lehman, Clune et al.), arXiv [2310.13032](https://arxiv.org/abs/2310.13032). Evolutionary archive with LM-generated variation and LM-evaluated quality and diversity; covers more of a specified search space than non-QD controls.
- **Intelligent Go-Explore**, arXiv [2405.15143](https://arxiv.org/abs/2405.15143), ICLR 2025. Foundation model selects archived states to return to, chooses actions, and judges whether new states are interesting enough to archive.
- **Nova: An Iterative Planning and Search Approach to Enhance Novelty and Diversity of LLM Generated Ideas**, arXiv [2410.14255](https://arxiv.org/abs/2410.14255). Iterated planning against the current idea set; reports **3.4 times more unique novel ideas**. Note carefully: that is a pure **rate** claim at fixed budget.
- **"Anchorless Diversification for Parallel LLM Ideation"**, arXiv [2605.30150](https://arxiv.org/abs/2605.30150), names the mechanism **"population-referential divergence"** and calls it "a strong low-cost baseline."
- **Multi-Novelty**, arXiv [2502.12700](https://arxiv.org/abs/2502.12700); **G2: Guided Generation for Enhanced Output Diversity in LLMs**, arXiv [2511.00432](https://arxiv.org/abs/2511.00432), EMNLP 2025; **Avoidance Decoding for Diverse Multi-Branch Story Generation**, arXiv [2509.02170](https://arxiv.org/abs/2509.02170).

**Exact remaining delta:** whether archive conditioning moves the **asymptote** or only the rate. Nobody measured it. Every paper above reports gains at fixed budget — Nova's 3.4 times, NoveltyBench's `distinct_k` at k equals 10, QDAIF's coverage of a **specified** (bounded) search space. Not one fits or extrapolates a ceiling.

But notice what this means: **the surviving delta on claim 5 is not a mechanism, it is claim 2 applied to a mechanism somebody else published.** Do not present archive conditioning as a contribution. Present it as one of the interventions the asymptote classifier sorts.

---

## Claim 6 — Assumption-negation operator

**Verdict: PARTIAL, narrow delta.**

**Human technique — cite it, it is old.** Assumption reversal, also called reverse brainstorming or flipped assumptions: write down the assumptions a solution is presumed to satisfy, then invert each and ideate from the inversion. It descends from Osborn's brainstorming checklist lineage and is standard in design-thinking practice. There is no single canonical academic citation; cite it as an established practitioner technique and move on. Claiming it as novel would be embarrassing.

**LLM operationalizations already published:**
- **Denial Prompting**, arXiv [2407.09007](https://arxiv.org/abs/2407.09007). The closest neighbour by a wide margin. It negates **against the model's own prior output**: solve, then prohibit the techniques the model just used, then repeat with accumulating constraints.
- **IDEAFix: Evaluation Framework for Creative Defixation Prompting in LLMs**, arXiv [2606.00875](https://arxiv.org/abs/2606.00875), June 2026. 81 design briefs expanded to 567 task variations, paired with **25 prompting strategies inspired by creativity and defixation methods**, including SCAMPER and TRIZ variants. Finding: simple prompting strategies boost originality, but **output homogenization persists across models**, confirming inherent limits. This is a benchmark that almost certainly already contains a strategy close to this one, and it reports a negative result on the class.
- **AutoTRIZ**, arXiv [2403.13002](https://arxiv.org/abs/2403.13002) (journal version in *Advanced Engineering Informatics*). Automates TRIZ contradiction resolution using the 39 engineering parameters, the contradiction matrix, and 40 inventive principles.
- **TRIZ Agents**, arXiv [2506.18783](https://arxiv.org/abs/2506.18783); **TRIZ-GPT**, arXiv [2408.05897](https://arxiv.org/abs/2408.05897); **Supermind Ideator**, arXiv [2311.01937](https://arxiv.org/abs/2311.01937).
- **AssumptionMiner: Extracting, Tracing, and Revising Implicit Assumptions in LLM Code Generation**, arXiv [2607.22898](https://arxiv.org/abs/2607.22898), July 2026. Makes implicit assumptions "a first-class output." This is the extraction half of the operator, built for correctness rather than creativity. Nobody has bolted its extraction step onto Denial Prompting's negation step, which is essentially this proposal.

**Exact remaining delta:** extracting the assumptions **invariant across the whole set of ideas generated so far** — the intersection rather than the per-solution technique list — and negating those as hard constraints. Denial Prompting bans what the last solution used; this bans what every solution shares. That is a genuine but small distinction, and it is an operator-level refinement, not a new idea. IDEAFix's negative result on defixation prompting generally is the bigger problem: if a 567-variation benchmark with 25 defixation strategies found persistent homogenization, the burden is to show this variant is qualitatively different rather than the 26th strategy.

---

## Claim 7 — Dual-loop evolution (programs plus prompt-guidance co-evolution)

**Verdict: EXISTS. Dead. Do not claim this.**

**EvoX: Meta-Evolution for Automated Discovery**, arXiv [2602.23413](https://arxiv.org/abs/2602.23413), February 2026. Retrieved descriptions call it, in these words, **"a dual-loop meta-evolution system that dynamically evolves search strategies"** — an inner loop that evolves solutions and an outer loop that evolves the search strategy governing generation, jointly evolving candidates and the strategies used to produce them. It maintains an evolved program database that constructs each next-generation input by choosing a parent program, a variation operator, and an optional inspiration set. Evaluated on roughly 200 real-world optimization tasks, **outperforming AlphaEvolve, OpenEvolve, GEPA, and ShinkaEvolve on the majority**. Every structural element of Ratchet's dual loop appears in that description. **[VERIFY BY HAND — but the framing match is close enough to treat as dispositive until proven otherwise.]**

**SeaEvo: Advancing Algorithm Discovery with Strategy Space Evolution**, arXiv [2604.24372](https://arxiv.org/abs/2604.24372), April 2026. Elevates natural-language strategy to a **first-class population-level representation** in LLM-driven evolutionary program search. Its stated motivation is that per-program fitness gives "little visibility into strategy-family dynamics, making it difficult to detect when an entire class of approaches has **saturated**," and that scalar-fitness selection discards lower-scoring candidates encoding strategically useful directions. That is the guidance-blocks-with-win-rate-statistics loop, plus a saturation diagnostic that also rhymes with claim 2. Improves both OpenEvolve and ShinkaEvolve backbones; 20.6 percent average relative improvement across four systems benchmarks.

**EvoPH — "Experience-Guided Reflective Co-Evolution of Prompts and Heuristics for Automatic Algorithm Design"** (Yihong Liu, Junyi Li, Wayne Xin Zhao, Hongyu Lu, Ji-Rong Wen), arXiv [2509.24509](https://arxiv.org/abs/2509.24509). **Prompts co-evolved with heuristic algorithms, guided by performance feedback**, with island migration and elite selection, in a closed loop of generation, evaluation, experience storage, and reflection — on combinatorial optimization problems with mechanical scoring.

**ShinkaEvolve**, arXiv [2509.19349](https://arxiv.org/abs/2509.19349) (Lange, Imajuku, Cetin, Sakana AI). Supplies the **bandit-based ensemble selection** component: adaptive performance-based selection prioritizing contributors by fitness improvement, plus code-novelty rejection sampling. New state-of-the-art circle packing in 150 samples.

**Promptbreeder**, arXiv [2309.16797](https://arxiv.org/abs/2309.16797). The **meta-rewriting** component: mutation-prompts that are themselves generated and improved by the LLM, self-referentially.

**GEPA**, arXiv [2507.19457](https://arxiv.org/abs/2507.19457). Reflective prompt evolution with Pareto-frontier sampling across a pool of top performers.

Also in the space: **CodeEvolve** ([2510.14150](https://arxiv.org/abs/2510.14150), [2605.04677](https://arxiv.org/abs/2605.04677)), **CORAL** ([2604.01658](https://arxiv.org/abs/2604.01658)), **GEAR** ([2605.13874](https://arxiv.org/abs/2605.13874)), **MadEvolve** ([2605.23007](https://arxiv.org/abs/2605.23007), maintains an ideas tree with all past mutations, scores, and lineage in every prompt), **LEVI** ([2605.09764](https://arxiv.org/abs/2605.09764)), **"Compute Allocation in Evolutionary Search: From Depth-Breadth to Multi-Armed Bandits"** ([2605.29268](https://arxiv.org/abs/2605.29268)).

**Exact remaining delta: none identified.** Mechanically-verified code evolution is FunSearch and AlphaEvolve. Bandit selection over proposers is ShinkaEvolve. Meta-rewritten guidance is Promptbreeder and AlphaEvolve's meta prompt evolution. Prompt-plus-program co-evolution under a mechanical evaluator is EvoPH. Strategy blocks as first-class population state with family-level fitness statistics is SeaEvo. The dual-loop framing by name is EvoX. **This claim is dead. Build on these systems, cite them, and do not present the architecture as a contribution.**

---

## Ranked summary

| Rank | Claim | Verdict | Nearest prior art | What survives | Defensible? |
|---|---|---|---|---|---|
| 1 | **2. Rate versus asymptote as intervention classifier** | PARTIAL | Diversity growth rate (ScienceDirect S294988212500091X); "Where You Inject Diversity Matters" [2606.10302](https://arxiv.org/abs/2606.10302); "Geometry of Knowledge" [2507.13874](https://arxiv.org/abs/2507.13874) | Fitting and extrapolating the **asymptote** of an accumulation curve over judged idea classes, and using the asymptote shift to sort interventions into support-walking versus support-enlarging | **Yes — strongest. Build the paper on this.** |
| 2 | **4. Union richness versus single-model asymptote** | PARTIAL | Algorithmic diversity [2503.00691](https://arxiv.org/abs/2503.00691); AI Crowd [2606.29661](https://arxiv.org/abs/2606.29661); Epistemic diversity [2512.15011](https://arxiv.org/abs/2512.15011) | Comparing union richness to a single model's **estimated asymptote** rather than its observed count | **Yes on the measurement.** Substance contested by [2501.19361](https://arxiv.org/abs/2501.19361) and [2607.26899](https://arxiv.org/abs/2607.26899) — engage directly |
| 3 | **3. External-taxonomy seeding versus free choice** | PARTIAL | LAB / InstructLab [2403.01081](https://arxiv.org/abs/2403.01081); SimpleStrat [2410.09038](https://arxiv.org/abs/2410.09038); Anchorless Diversification [2605.30150](https://arxiv.org/abs/2605.30150); Alien Space [2603.01092](https://arxiv.org/abs/2603.01092) | The **near-disjointness** of the two arms' idea classes (union 2.14 times) under a standardized public taxonomy | **As a finding, not a mechanism.** Anchorless Diversification concludes the opposite and must be rebutted |
| 4 | **1. Species-richness estimation of LLM output support** | PARTIAL (method essentially EXISTS) | **KnowSum [2506.02058](https://arxiv.org/abs/2506.02058)** — Good-Turing unseen-species on LLM outputs, one application named "measuring output diversity"; UCS [2604.12015](https://arxiv.org/abs/2604.12015) | Only the application to blinded-judge functional idea classes, and the mode-collapse-diagnostic framing | **Weak.** Cite KnowSum prominently; do not claim the method |
| 5 | **6. Assumption-negation operator** | PARTIAL, narrow | **Denial Prompting [2407.09007](https://arxiv.org/abs/2407.09007)**; IDEAFix [2606.00875](https://arxiv.org/abs/2606.00875); AssumptionMiner [2607.22898](https://arxiv.org/abs/2607.22898); AutoTRIZ [2403.13002](https://arxiv.org/abs/2403.13002) | Negating the assumptions **invariant across the whole idea set** rather than the last solution's techniques | **Marginal.** An operator refinement. IDEAFix already reports a null for the class |
| 6 | **5. Archive-conditioned generation** | **EXISTS** | NoveltyBench in-context regeneration [2504.05228](https://arxiv.org/abs/2504.05228); Denial Prompting [2407.09007](https://arxiv.org/abs/2407.09007); QDAIF [2310.13032](https://arxiv.org/abs/2310.13032); IGE [2405.15143](https://arxiv.org/abs/2405.15143); Nova [2410.14255](https://arxiv.org/abs/2410.14255) | Nothing as a mechanism. Only the asymptote question, which is claim 2 | **No.** Demote to an intervention the classifier evaluates |
| 7 | **7. Dual-loop evolution** | **EXISTS** | **EvoX [2602.23413](https://arxiv.org/abs/2602.23413)** ("dual-loop meta-evolution"); SeaEvo [2604.24372](https://arxiv.org/abs/2604.24372); EvoPH [2509.24509](https://arxiv.org/abs/2509.24509); ShinkaEvolve [2509.19349](https://arxiv.org/abs/2509.19349); Promptbreeder [2309.16797](https://arxiv.org/abs/2309.16797) | Nothing found | **No. Dead.** |

## Structural observation

The surviving deltas in claims 1, 2, 4, and 5 all reduce to a **single** contribution: *asymptotic richness estimation as the evaluation frame for diversity interventions*. Claim 2 is that contribution stated directly. Claim 4 is that contribution applied to model union. Claim 5's remainder is that contribution applied to archive conditioning. Claim 1 is the statistical machinery, and KnowSum already owns it.

That is one paper, not four. Its thesis is the measurement frame, not any individual mechanism, and every mechanism tested — taxonomy seeding, model union, archive conditioning, assumption negation — becomes evidence rather than contribution. Claims 3 and 6 are secondary empirical results inside that paper. Claim 7 belongs to a different literature that has already closed.

## Things the audit could not verify

1. **All of it, at the primary-source level.** WebFetch was blocked for every domain attempted. No PDF was read.
2. **KnowSum's output-diversity application** ([2506.02058](https://arxiv.org/abs/2506.02058)) — the exact construction and scale. This determines whether claim 1 is PARTIAL or EXISTS.
3. **The diversity growth rate methods** (ScienceDirect S294988212500091X) — whether any asymptote fitting or extrapolation occurs. This determines whether claim 2 survives at all.
4. **EvoX's dual-loop architecture** ([2602.23413](https://arxiv.org/abs/2602.23413)) — whether the outer loop carries per-strategy win-rate statistics and bandit selection specifically. Claim 7 is judged dead regardless, given SeaEvo and EvoPH.
5. **The EA metric in [2503.00691](https://arxiv.org/abs/2503.00691)** — whether it involves unseen-class extrapolation. If it does, claim 4's delta narrows sharply.
6. **[2607.26899](https://arxiv.org/abs/2607.26899)** — exactly what the three-model-family simulation failed to reproduce, since it is the strongest published counter-evidence to claim 4.
7. **IDEAFix's 25 strategies** ([2606.00875](https://arxiv.org/abs/2606.00875)) — whether one is already assumption negation over an idea set. If so, claim 6 drops to EXISTS.
8. Google Scholar was not queryable as a distinct source; coverage came through general web search. Non-arXiv venues (ACM CHI, CSCW, Design Studies, *Creativity Research Journal*) are under-sampled relative to arXiv, and claims 3 and 6 are the ones most likely to have unfound neighbours in the design-research literature.
