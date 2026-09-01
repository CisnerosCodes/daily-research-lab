"""Generate the data-driven tables for paper.md from results.json files.

Usage:  python paper/fill_sections.py            # prints markdown blocks for each placeholder key
        python paper/fill_sections.py --apply    # replaces <!-- KEY --> placeholders in paper.md in place
Prose verdicts are written by hand; this script only produces the tables so numbers are never typed.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
LABEL = {"baseline": "no norm", "qknorm": "QK-norm", "qnorm_only": "query-only norm",
         "knorm_only": "key-only norm", "knorm_dynk": "**FKN**", "k_emascale": "key / running scale (alpha = 1 fixed)",
         "qknorm_dynq": "QK-norm + query channel",
         "k_static_init_scale": "key / first-batch scale (static)", "k_emascale_m09": "key / running scale, m 0.9",
         "k_emascale_m0999": "key / running scale, m 0.999", "q_emascale": "query / running scale"}


def load(exp_id):
    p = EXP / exp_id / "results.json"
    return json.load(open(p)) if p.exists() else None


def fmt(x, nd=3, sign=False):
    if x is None:
        return "—"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def master_table(res, arms=None, caption=""):
    by = res["metrics"]["by_arm_hd"]
    per = res["metrics"]["per_head_dim"]
    hds = [int(h) for h in res["config"]["params"]["head_dims"] if str(h) in per]
    arms = arms or [a for a in LABEL if a in by]
    head = "| arm | " + " | ".join(f"hd {hd} ({128 // hd} heads)" for hd in hds) + " |"
    sep = "|---|" + "---|" * len(hds)
    rows = [head, sep]
    for a in arms:
        cells = []
        for hd in hds:
            r = by[a].get(str(hd))
            if r is None:
                cells.append("—")
                continue
            d = per[str(hd)]["arms"][a]["delta_vs_baseline"]
            best = per[str(hd)]["best_arm"] == a
            v = f"{r['val_bpc_mean']:.4f}"
            v = f"**{v}**" if best else v
            cells.append(f"{v} ({d:+.3f}; ±{r['seed_spread_bpc'] / 2:.3f})" if a != "baseline" else f"{v} (±{r['seed_spread_bpc'] / 2:.3f})")
        rows.append(f"| {LABEL[a]} | " + " | ".join(cells) + " |")
    rows.append("")
    rows.append(f"*{caption} Cells: mean val bpc (delta vs no norm; ± half the seed spread). Bold = best arm at that width.*")
    # paired wins
    rows.append("")
    rows.append("| arm | " + " | ".join(f"hd {hd}" for hd in hds) + " |")
    rows.append(sep)
    for a in arms:
        if a == "baseline":
            continue
        cells = []
        for hd in hds:
            x = per[str(hd)]["arms"].get(a)
            cells.append(f"{x['beats_baseline_seeds']} / {x['beats_qknorm_seeds']}" if x else "—")
        rows.append(f"| {LABEL[a]} | " + " | ".join(cells) + " |")
    rows.append("")
    rows.append("*Paired-seed wins: seeds where the arm beats no norm / seeds where it beats QK-norm, at identical inits and batches.*")
    return "\n".join(rows)


def alpha_table(res):
    P2 = res["metrics"]["P2_alpha_vs_head_dim"]
    hds = [int(h) for h in res["config"]["params"]["head_dims"]]
    ak, aq = P2["k_alpha_knorm_dynk"], P2["q_alpha_qknorm_dynq"]
    rows = ["| exponent | " + " | ".join(f"hd {hd}" for hd in hds) + " |", "|---|" + "---|" * len(hds)]
    rows.append("| key alpha (FKN) | " + " | ".join(fmt(ak.get(str(hd)), 2) for hd in hds) + " |")
    rows.append("| query alpha (QK-norm + query channel) | " + " | ".join(fmt(aq.get(str(hd)), 2) for hd in hds) + " |")
    return "\n".join(rows)


def verdict_lines(res):
    v = res["metrics"]["verdicts"]
    per = res["metrics"]["per_head_dim"]
    out = []
    for a, d in v.items():
        out.append(f"- {LABEL[a]}: within tolerance of the best arm everywhere = {d['within_tol_of_best_everywhere']}; "
                   f"beats no norm everywhere = {d['beats_baseline_mean_everywhere']}; beats QK-norm everywhere = {d['beats_qknorm_mean_everywhere']}.")
    gb = res["metrics"]["global_best"]
    out.append(f"- Global best: {LABEL[gb['arm']]} at hd {gb['head_dim']} = {gb['bpc']:.4f} bpc.")
    p3 = res["metrics"].get("P3_dynk_vs_dynq", {})
    if p3:
        out.append("- FKN minus (QK-norm + query channel): " + ", ".join(
            f"hd {hd}: {d['dynk_minus_dynq']:+.3f} ({d['dynk_beats_dynq_seeds']})" for hd, d in p3.items()) + ".")
    return "\n".join(out)


def anchor_table(res_list):
    rows = ["| source night | arm | hd | seed | archived | this run | delta |", "|---|---|---|---|---|---|---|"]
    n_ok = n = 0
    for res in res_list:
        if res is None:
            continue
        rep = res["metrics"]["replication_vs_parents"]
        for p in rep["pairs"]:
            rows.append(f"| {p['source']} | {p['arm']} | {p['head_dim']} | {p['seed']} | {p['expected']:.5f} | {p['tonight']:.5f} | {p['delta']:+.5f} |")
            n += 1
            n_ok += int(p["ok"])
    rows.append("")
    rows.append(f"*{n_ok} of {n} archived cells reproduced within 0.0005 bpc (torch 2.13.0+cu130, CPU, 2 threads).*")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    e1 = load("2026-09-01_knorm-dynk-head-sweep")
    e2 = load("2026-09-01_knorm-dynk-ptb-transfer")
    e3 = load("2026-09-01_knorm-dynk-longer-training")
    e4 = load("2026-09-01_kscale-adaptive-vs-static")
    blocks = {}
    if e1:
        core = ["baseline", "qknorm", "qnorm_only", "knorm_only", "knorm_dynk", "k_emascale", "qknorm_dynq"]
        blocks["E1_TABLE"] = master_table(e1, core, "Table 1. Head-split sweep on tiny-shakespeare, 600 steps, three paired seeds (registry 2026-09-01_knorm-dynk-head-sweep).")
        blocks["E1_ALPHA"] = alpha_table(e1)
        blocks["E1_VERDICT_AUTO"] = verdict_lines(e1)
    if e2:
        blocks["E2_TABLE"] = master_table(e2, None, "Table 2. Character-level Penn Treebank, 600 steps, three paired seeds (registry 2026-09-01_knorm-dynk-ptb-transfer).")
    if e3:
        blocks["E3_TABLE"] = master_table(e3, None, "Table 3. 1800 steps on tiny-shakespeare, three paired seeds (registry 2026-09-01_knorm-dynk-longer-training).")
    if e4:
        blocks["E4_TABLE"] = master_table(e4, None, "Table 4. Adaptive vs static key scale, 600 steps, three paired seeds (registry 2026-09-01_kscale-adaptive-vs-static).")
    blocks["ANCHOR_TABLE"] = anchor_table([e1, e2, e3, e4])
    for k, v in blocks.items():
        print(f"\n<!-- {k} -->\n{v}\n")
    if args.apply:
        p = ROOT / "paper" / "paper.md"
        s = p.read_text()
        for k, v in blocks.items():
            s = s.replace(f"<!-- {k} -->", v)
        p.write_text(s)
        print("applied to paper.md")


if __name__ == "__main__":
    main()
