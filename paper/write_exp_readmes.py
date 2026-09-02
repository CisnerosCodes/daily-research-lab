"""Write each new experiment's README.md from its results.json and paper/summaries.json.

Usage:  python paper/write_exp_readmes.py [--apply]
Keeps the lab's hypothesis -> method -> result -> takeaway shape, and takes every number from
results.json so nothing is transcribed by hand.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
SUM = json.loads((ROOT / "paper" / "summaries.json").read_text())

TAKEAWAY = {
    "2026-09-01_knorm-dynk-head-sweep":
        "The magnitude channel is the first arm in the thread to beat the unnormalised baseline at "
        "every head width, but its learnable exponent is inert: freezing it at 1 matches it everywhere. "
        "That control is what redirected the thread from normalisation to scale.",
    "2026-09-01_knorm-dynk-ptb-transfer":
        "The phase diagram splits in two. The tiny-head half (norms hurt, the magnitude channel repairs) "
        "transfers to a second corpus and gets larger. The wide-head half (drop the query norm) does not "
        "transfer at all and should not be quoted as a recommendation.",
    "2026-09-01_knorm-dynk-longer-training":
        "Budget matters more than the thread assumed. Everything halves at 3x training and the wide-head "
        "ordering dissolves into a 0.005-bpc band. Only the tiny-head result keeps its sign in every "
        "paired seed, so it is the only part of the map that survives every stress test.",
    "2026-09-01_kscale-adaptive-vs-static":
        "The win is a constant, not a running statistic: the less the per-head scale adapts, the better it "
        "does, and a scale frozen at the first batch is best. Since a frozen per-head scale is algebraically "
        "a fixed multiplier on the attention logits, this turns the whole thread into a temperature question.",
    "2026-09-01_fractional-norm-both-sides":
        "The magnitude channel is not key-specific - the query mirror matches it at all six widths - which "
        "kills the 'repair the causal side' reading. Applying it to both sides is best at narrow heads and "
        "worse than baseline at one wide head, reproducing the thread's q x k interaction rather than escaping it.",
    "2026-09-01_logit-scale-sweep":
        "The tiny-head normalisation cliff is largely a mis-set attention temperature. A fixed per-head key "
        "multiplier beats every normalisation arm in the thread, the best value moves with head width, and "
        "the same dial made learnable from the default does not travel there - which is why 2026-07-31 "
        "concluded that temperature was not the problem.",
}


def table(res, arms=None):
    per, by = res["metrics"]["per_head_dim"], res["metrics"]["by_arm_hd"]
    hds = [int(h) for h in res["config"]["params"]["head_dims"] if str(h) in per]
    arms = arms or res["config"]["params"]["arm_order"]
    arms = [a for a in arms if a in by]
    out = ["| arm | " + " | ".join(f"hd {hd}" for hd in hds) + " |", "|---|" + "---|" * len(hds)]
    for a in arms:
        cells = []
        for hd in hds:
            r = by[a].get(str(hd))
            if not r:
                cells.append("—")
                continue
            d = per[str(hd)]["arms"][a].get("delta_vs_baseline")
            best = per[str(hd)]["best_arm"] == a
            v = f"**{r['val_bpc_mean']:.4f}**" if best else f"{r['val_bpc_mean']:.4f}"
            cells.append(v if d is None or a == "baseline" else f"{v} ({d:+.3f})")
        out.append(f"| `{a}` | " + " | ".join(cells) + " |")
    return "\n".join(out)


def readme_for(exp_id):
    res = json.load(open(EXP / exp_id / "results.json"))
    cfg, p, m = res["config"], res["config"]["params"], res["metrics"]
    s = SUM.get(exp_id, {})
    rep = m["replication_vs_parents"]
    ds = cfg["dataset"]
    body = f"""# {cfg['title']}

**Date:** {exp_id[:10]} · **Status:** done · **Part of** [the attention-scale paper](../../paper/paper.md)

## Hypothesis
{cfg['hypothesis'].strip()}

## Method
- {len(p['arm_order'])} arms x head_dim {p['head_dims']} x seeds {p['seeds']}, byte-identical paired
  initialisations and a shared batch stream, asserted by an init-signature check at run time.
- 2-layer pre-norm char-GPT, d_model {p['d_model']}, ~423k params, block {p['block_size']},
  AdamW lr {p['lr']} with {p['warmup']} warmup steps then cosine, {p['steps']} steps,
  weight decay {p['weight_decay']} on matrices only, grad clip {p['grad_clip']}.
- Data: {ds['name']} ({ds.get('hash', 'n/a')}).
- Metric: validation bits per character over {p['max_eval_blocks']} contiguous held-out blocks.
- {len(res['runs'])} runs, {res['duration_sec'] / 60:.0f} min CPU total across shards.
- Replication: {rep['n_ok']}/{rep['n_checked']} archived cells from parent nights reproduced within
  {rep['tol']} bpc{' (all exact)' if rep['n_checked'] and rep['n_ok'] == rep['n_checked'] else ''}.

## Result
{s.get('result_summary', '').strip()}

{table(res)}

*Mean val bpc over {len(p['seeds'])} paired seeds, with the change against the unnormalised baseline
in brackets. Bold marks the best arm at that head width.*

![result](chart.png)

## Takeaway
{TAKEAWAY[exp_id]}

## Novelty check
- Verdict: {s.get('verdict', 'unchecked')}
- Queries: {'; '.join(s.get('queries', [])) or 'n/a'}
- Conclusion: {s.get('conclusion', '')}
- Closest prior art: {', '.join(s.get('prior_art', [])) or 'n/a'}
- Note: arXiv and OpenAlex return 403 from this sandbox, so novelty checks use web search plus a
  registry grep, as on every night since 2026-08-06.
"""
    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    for exp_id in TAKEAWAY:
        if not (EXP / exp_id / "results.json").exists():
            print("skip (no results):", exp_id)
            continue
        body = readme_for(exp_id)
        if args.apply:
            (EXP / exp_id / "README.md").write_text(body)
            print("wrote", exp_id + "/README.md")
        else:
            print(f"===== {exp_id}\n{body[:600]}\n...")


if __name__ == "__main__":
    main()
