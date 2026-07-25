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
**4 experiments logged.**

| Date | Experiment | Status | Result |
|---|---|---|---|
| 2026-07-25 | [NoPE vs RoPE vs ALiBi vs APE: does the length-generalization ranking survive at 0.1M params?](experiments/2026-07-25_pe-length-gen-tiny) | ✅ done | Hypothesis refuted: at 0.1M params and iso-compute the length-generalization ranking is AL |
| 2026-07-24 | [Tsetlin Machine: does exact rule recovery break before accuracy under label noise?](experiments/2026-07-24_tsetlin-dnf-recovery) | ✅ done | Hypothesis partially refuted: exact recall of the planted clauses survives to 20% label no |
| 2026-07-23 | [Superposition phase diagram: does feature correlation resist or reshape superposition?](experiments/2026-07-23_superposition-correlation-phase) | ✅ done | Hypothesis refuted: across the grid, raising within-pair correlation makes the model colla |
| 2026-07-21 | [Game of Life: when does test-time recursion still generalize?](experiments/2026-07-21_gol-depth-recursion) | ✅ done | Capacity window: at H=4 tied recursion learns the exact reusable rule and extrapolates to  |
<!-- INDEX:END -->

## Ground rules
- Every experiment must fit on a CPU in roughly minutes to ~1 hour.
- Every experiment records a novelty check. Replication is allowed — but labelled honestly.
- Results are written by code, never by hand.
- License: MIT (code), CC-BY-4.0 (notes/figures).
