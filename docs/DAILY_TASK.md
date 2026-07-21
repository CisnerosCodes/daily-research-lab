# Daily task — the autonomous nightly procedure

This is the exact procedure the scheduled cloud session follows each night. It is written so a fresh
agent with no memory can run one complete, non-duplicated experiment and leave a clean record.

**Hard constraints**
- CPU-only. No GPU. Keep every model tiny and every dataset small.
- Time-box the whole run; a single experiment should train in roughly minutes to ~30 min.
- Exactly ONE experiment per night.
- Never duplicate: check the registry AND the world before building.
- Be honest: a null or negative result is a real result — record it, do not fake success.

**Procedure**
1. **Sync.** Clone/pull the repo from GitHub so the registry and backlog are current.
2. **Pick.** Choose the next experiment: prefer an unstarted ⭐ item in `docs/BACKLOG.md`, or rotate
   through the tracks (A latent-reasoning → B interpretability → C non-transformer → D ablation), or
   propose a fresh small idea that builds on a recent result.
3. **Search before creating.** Run `python scripts/novelty_check.py "<query>"` and `grep` the
   registry. If the idea already exists (ours or the world's) with no new angle, pick another. Record
   the sources and a verdict.
4. **Scaffold.** `python scripts/new_experiment.py "<slug>" "<Title>"`.
5. **Implement** `run.py`: seeded, CPU-only, writes `results.json` and a `chart.png`.
6. **Run** it. If it errors or the result is null, still finalize it with `status: failed` or an honest
   negative writeup — that is valuable and prevents re-attempting the same dead end.
7. **Record.** Fill the experiment `README.md` (hypothesis → method → result → takeaway) and complete
   the registry row (metrics, links, novelty_check, status `done`/`failed`).
8. **Validate + index.** `python scripts/validate_registry.py && python scripts/build_index.py`.
9. **Commit & push** to GitHub with a message summarizing the finding.
10. **Report.** Print a short summary of the day's result and deliver the chart to the user.

**Promotion.** If a result is strong or a thread is deepening, note it in the registry (`spun_out`)
and flag it for graduation into its own repository.

**Idea hygiene.** When you invent a new idea (not from the backlog), append it to `docs/BACKLOG.md`
so future runs can see it was considered.
