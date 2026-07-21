# The system: how a day works

The goal is a loop so low-friction that the only real effort is the idea and the code. Everything else (scaffolding, validation, the index) is scripted.

## Daily checklist
1. **One-line hypothesis.** Write it before anything else.
2. **Dedup against your own history.** `grep` the registry or query it:
   ```bash
   grep -i "grokking" registry.jsonl
   # or, for SQL: duckdb -c "SELECT id,title FROM read_json_auto('registry.jsonl') WHERE title ILIKE '%grok%'"
   ```
   Also check `related_ids` on any close matches. This is the cheapest way to never recreate your own work.
3. **World novelty check (~10 min).** Run `python scripts/novelty_check.py "<query>"` (queries arXiv + Semantic Scholar + OpenAlex + GitHub + Hugging Face) and skim the hits. Record each `{name, query, url, hits}` into the experiment's registry row. Assign a verdict:
   - `novel` — nothing close.
   - `partial-prior-art` — related work exists; state your differentiator.
   - `replication` — this reproduces known work (fine, just labelled).
   If it is pure prior art and you add nothing, pick another idea.
4. **Scaffold.** `python scripts/new_experiment.py "<slug>"` copies the template into `experiments/<date>_<slug>/` and appends a draft registry row (`status: in_progress`).
5. **Implement.** All config + the seed live in `experiment.yaml`; `run.py` reads it. Keep the model tiny and the data small.
6. **Run.** `python experiments/<date>_<slug>/run.py` — it writes `results.json` (metrics, seed, git commit, environment).
7. **Finalize.** Fill the experiment README (hypothesis → method → result → takeaway). Complete the registry row (`status: done|failed`, metrics, links, novelty_check).
8. **Validate + index.** `python scripts/validate_registry.py && python scripts/build_index.py`
9. **Commit & push.** The daily commit is also what keeps any scheduled automation alive.
10. **Promote if notable.** Spin standouts into their own repo; set `status: spun_out` and `links.spinout_repo`.

## Search-before-create: the sources
Papers with Code was shut down by Meta in July 2025 — do not rely on it. The current stack:

| Source | What it answers | How |
|---|---|---|
| **Your `registry.jsonl`** | "Did I already do this?" | grep / DuckDB |
| **arXiv API** | "Is there a paper?" | `http://export.arxiv.org/api/query?search_query=abs:%22...%22` (3s between calls) |
| **Semantic Scholar** | Papers + citations + "similar work" | `https://api.semanticscholar.org/graph/v1/paper/search?query=...` (get a free key for automation) |
| **OpenAlex** | Open Google-Scholar replacement (abstracts) | `https://api.openalex.org/works?search=...&mailto=you@email` |
| **GitHub** | "Is there code already?" | `gh search repos "..."` / `gh search code "..."` |
| **Hugging Face** | Existing models/datasets | `HfApi().list_models(search=...)`, `list_datasets(...)` |

Google Scholar has no official API and blocks scraping — use OpenAlex + Semantic Scholar instead, and reserve Scholar for occasional manual eyeballing.

Optional standing radar: self-host Karpathy's `arxiv-sanity-lite` for tagged daily recommendations.

## Reproducibility hygiene (baked into the template)
- Set and record ONE seed everywhere (`random`, `numpy`, `torch`); set `PYTHONHASHSEED`; for bitwise determinism set `OMP_NUM_THREADS=1` and `torch.use_deterministic_algorithms(True)`.
- Pin dependencies per experiment (`requirements.txt` via `pip freeze`, or `uv`).
- Every hyperparameter lives in `experiment.yaml` — no magic constants in code.
- `run.py` writes the git commit SHA into `results.json`, binding each row to exact source.
- Record each dataset's name, source, license, and a content hash.

## GitHub strategy
Monorepo-first (this repo) for the daily habit and the single dedup registry. When an experiment becomes substantial, spin it into its own repo with its own README + topics + license, backlink it, and mark the registry row `spun_out`. That gives you "lots of projects" without paying repo-creation tax every day. A profile README can showcase the highlights and the streak.
