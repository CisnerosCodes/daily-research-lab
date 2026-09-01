"""Append registry rows for tonight's experiments, built from their results.json (numbers never typed).

Usage:  python paper/register_rows.py            # dry run: prints the rows
        python paper/register_rows.py --apply    # appends to registry.jsonl (skips ids already present)
The prose fields (hypothesis / result_summary / conclusion) are read from SUMMARIES below.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"

DATASETS = {
    "shakespeare": {"name": "tiny-shakespeare (character level)",
                    "source": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
                    "license": "public domain text; char-rnn repo MIT", "hash": "md5:6fb458f1232090904fb40fe944165e91"},
    "ptb": {"name": "Penn Treebank (Mikolov preprocessing), character level, train/valid files",
            "source": "https://github.com/wojzaremba/lstm (data/ptb.train.txt, data/ptb.valid.txt)",
            "license": "PTB text via the widely redistributed Mikolov preprocessing (LDC origin); wojzaremba/lstm Apache-2.0",
            "hash": "md5:f26c4b92c5fdc7b3f8c7cdcb991d8420 (train), md5:aa0affc06ff7c36e977d7cd49e3839bf (valid)"},
}

# filled in by hand after reading results (see paper/paper.md Section 7)
SUMMARIES = json.loads((ROOT / "paper" / "summaries.json").read_text()) if (ROOT / "paper" / "summaries.json").exists() else {}


def row_for(exp_id, dataset_key, tags, related):
    res = json.load(open(EXP / exp_id / "results.json"))
    cfg = res["config"]
    p = cfg["params"]
    m = res["metrics"]
    by = m["by_arm_hd"]
    per = m["per_head_dim"]
    metrics = {"headline": m["headline"], "n_runs": len(res["runs"]),
               "duration_sec": res["duration_sec"],
               "val_bpc_mean_by_arm_hd": {a: {hd: by[a][hd]["val_bpc_mean"] for hd in by[a]} for a in by},
               "delta_vs_baseline_by_arm_hd": {hd: {a: per[hd]["arms"][a]["delta_vs_baseline"] for a in per[hd]["arms"]} for hd in per},
               "best_arm_by_hd": {hd: per[hd]["best_arm"] for hd in per},
               "replication_ok": m["replication_vs_parents"]["ok"],
               "replication_checked": m["replication_vs_parents"]["n_checked"]}
    if "verdicts" in m:
        metrics["strictly_better_than_both_parents"] = {a: v["strictly_better_than_both_parents"] for a, v in m["verdicts"].items()}
    s = SUMMARIES.get(exp_id, {})
    slug = exp_id.split("_", 1)[1]
    return {
        "id": exp_id, "date": exp_id[:10], "title": cfg["title"], "slug": slug,
        "hypothesis": cfg["hypothesis"].strip(),
        "question": s.get("question", ""),
        "method": s.get("method", f"{len(p['arm_order'])} arms x head_dim {p['head_dims']} x seeds {p['seeds']}, byte-identical paired inits; "
                                  f"d_model {p['d_model']}, {p['n_layer']} layers, {p['steps']} steps, AdamW {p['lr']} cosine; val bpc on {p['max_eval_blocks']} blocks."),
        "architecture": f"2-layer pre-norm char-GPT, d_model {p['d_model']}, ~423k params, arms: {', '.join(p['arm_order'])}",
        "dataset": DATASETS[dataset_key], "task": "character-level language modelling (val bits per character)",
        "framework": f"torch {res['env']['torch']} (CPU, {res['env']['torch_threads']} threads)", "seed": 0,
        "compute": f"CPU, {res['duration_sec'] / 60:.0f} min total across shards ({len(res['runs'])} runs)",
        "metrics": metrics, "result_summary": s.get("result_summary", ""), "status": "done",
        "novelty_check": {"checked_on": exp_id[:10], "verdict": s.get("verdict", "novel"),
                          "sources": [{"name": "web search", "query": q, "hits": None} for q in s.get("queries", [])],
                          "conclusion": s.get("conclusion", "")},
        "prior_art": s.get("prior_art", []), "related_ids": related,
        "links": {"folder": f"experiments/{exp_id}", "results_json": f"experiments/{exp_id}/results.json",
                  "chart": f"experiments/{exp_id}/chart.png", "paper": "paper/paper.md"},
        "tags": tags,
    }


ROWS = [
    ("2026-09-01_knorm-dynk-head-sweep", "shakespeare",
     ["attention-norm", "qk-norm", "head-dim", "fractional-key-norm", "kill-test", "cpu-only"],
     ["2026-08-31_hd4-kside-cliff-mechanism", "2026-08-30_knorm-only-head-sweep", "2026-08-11_qknorm-dyntemp-composite-sweep"]),
    ("2026-09-01_knorm-dynk-ptb-transfer", "ptb",
     ["attention-norm", "fractional-key-norm", "transfer", "penn-treebank", "cpu-only"],
     ["2026-09-01_knorm-dynk-head-sweep"]),
    ("2026-09-01_knorm-dynk-longer-training", "shakespeare",
     ["attention-norm", "fractional-key-norm", "longer-training", "cpu-only"],
     ["2026-09-01_knorm-dynk-head-sweep", "2026-08-30_knorm-only-head-sweep"]),
    ("2026-09-01_kscale-adaptive-vs-static", "shakespeare",
     ["attention-norm", "running-scale", "mechanism", "cpu-only"],
     ["2026-09-01_knorm-dynk-head-sweep", "2026-08-31_hd4-kside-cliff-mechanism"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    existing = {json.loads(l)["id"] for l in open(ROOT / "registry.jsonl") if l.strip()}
    out = []
    for exp_id, ds, tags, related in ROWS:
        if not (EXP / exp_id / "results.json").exists():
            print("skip (no results yet):", exp_id)
            continue
        if exp_id in existing:
            print("skip (already registered):", exp_id)
            continue
        out.append(row_for(exp_id, ds, tags, related))
    for r in out:
        print(json.dumps(r)[:400], "...")
    if args.apply and out:
        with open(ROOT / "registry.jsonl", "a") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
        print(f"appended {len(out)} rows")


if __name__ == "__main__":
    main()
