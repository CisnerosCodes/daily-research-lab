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
         "knorm_only": "key-only norm", "knorm_dynk": "**magnitude channel** (learnable exponent)", "k_emascale": "magnitude channel (exponent frozen at 1)",
         "qknorm_dynq": "QK-norm + query channel",
         "k_static_init_scale": "key / first-batch scale (static)", "k_emascale_m09": "key / running scale, m 0.9",
         "k_emascale_m0999": "key / running scale, m 0.999", "q_emascale": "query / running scale",
         "fqn": "magnitude channel, queries", "fqkn": "magnitude channel, both sides",
         "c2": "fixed multiplier c = 2", "c4": "fixed multiplier c = 4", "c8": "fixed multiplier c = 8",
         "c16": "fixed multiplier c = 16", "c_learn1": "learnable c, starts at 1",
         "c_learn4": "learnable c, starts at 4", "kinit_x4": "key init scaled x4 (no runtime op)"}


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
    head = "| arm | " + " | ".join(f"hd {hd} ({128 // hd} head{'s' if 128 // hd > 1 else ''})" for hd in hds) + " |"
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
            if not x:
                cells.append("—")
            elif a == "qknorm":
                cells.append(f"{x['beats_baseline_seeds']} / —")
            elif x.get("beats_qknorm_seeds") is not None:
                cells.append(f"{x['beats_baseline_seeds']} / {x['beats_qknorm_seeds']}")
            else:
                cells.append(f"{x['beats_baseline_seeds']}")
        rows.append(f"| {LABEL[a]} | " + " | ".join(cells) + " |")
    rows.append("")
    has_qk = any(per[str(hd)]["arms"].get(a, {}).get("beats_qknorm_seeds") is not None
                 for a in arms for hd in hds)
    rows.append("*Paired-seed wins: seeds where the arm beats no norm"
                + (" / seeds where it beats QK-norm" if has_qk else "")
                + ", at identical initializations and batches.*")
    return "\n".join(rows)


def alpha_table(res):
    P2 = res["metrics"]["P2_alpha_vs_head_dim"]
    hds = [int(h) for h in res["config"]["params"]["head_dims"]]
    ak, aq = P2["k_alpha_knorm_dynk"], P2["q_alpha_qknorm_dynq"]
    rows = ["| exponent | " + " | ".join(f"hd {hd}" for hd in hds) + " |", "|---|" + "---|" * len(hds)]
    rows.append("| key-side exponent, learnable arm | " + " | ".join(fmt(ak.get(str(hd)), 2) for hd in hds) + " |")
    rows.append("| query-side exponent, QK-norm + query channel | " + " | ".join(fmt(aq.get(str(hd)), 2) for hd in hds) + " |")
    rows.append("")
    rows.append("*The model does move the exponent, and further at wider heads. Freezing it at 1 costs nothing "
                "(Table 1), so the movement does not pay.*")
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
        out.append("- Magnitude channel minus (QK-norm + query channel): " + ", ".join(
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



def e6_curve(res):
    """The fixed-c curve per head width, plus where the optimum sits."""
    per = res["metrics"]["per_head_dim"]
    hds = [int(h) for h in res["config"]["params"]["head_dims"] if str(h) in per]
    cs = sorted({float(c) for hd in hds for c in per[str(hd)].get("fixed_c_curve", {})})
    rows = ["| head width | " + " | ".join(f"c = {c:g}" for c in cs) + " | best c | gain at best c |",
            "|---|" + "---|" * (len(cs) + 2)]
    for hd in hds:
        cur = per[str(hd)].get("fixed_c_curve", {})
        base = per[str(hd)]["arms"]["baseline"]["bpc"]
        cells = []
        for c in cs:
            v = cur.get(str(c))
            if v is None:
                cells.append("—")
            else:
                d = v - base
                cells.append(f"**{d:+.3f}**" if abs(v - per[str(hd)]["best_fixed_c_bpc"]) < 1e-9 else f"{d:+.3f}")
        rows.append(f"| head_dim {hd} ({128 // hd} heads) | " + " | ".join(cells) +
                    f" | **{per[str(hd)]['best_fixed_c']:g}** | {per[str(hd)]['best_fixed_c_bpc'] - base:+.3f} |")
    rows.append("")
    rows.append("*Change in val bpc against the same arm at c = 1 (the default), three paired seeds. "
                "Bold marks the best constant at each width.*")
    return "\n".join(rows)


def e6_learn(res):
    per = res["metrics"]["per_head_dim"]
    hds = [int(h) for h in res["config"]["params"]["head_dims"] if str(h) in per]
    rows = ["| head width | learnable from 1 | learnable from 4 | best fixed c | c reached from 1 | c reached from 4 |",
            "|---|---|---|---|---|---|"]
    by = res["metrics"]["by_arm_hd"]
    for hd in hds:
        A = per[str(hd)]["arms"]
        c_reached = by.get("c_learn1", {}).get(str(hd), {}).get("c_mean")
        c_reached4 = by.get("c_learn4", {}).get(str(hd), {}).get("c_mean")
        rows.append(f"| head_dim {hd} | {A['c_learn1']['delta_vs_baseline']:+.3f} "
                    f"({A['c_learn1']['beats_baseline_seeds']}) | {A['c_learn4']['delta_vs_baseline']:+.3f} "
                    f"({A['c_learn4']['beats_baseline_seeds']}) | {per[str(hd)]['best_fixed_c_bpc'] - A['baseline']['bpc']:+.3f} "
                    f"at c = {per[str(hd)]['best_fixed_c']:g} | {c_reached:.2f} | {c_reached4:.2f} |"
                    if c_reached is not None and c_reached4 is not None else "| — |")
    rows.append("")
    rows.append("*Change in val bpc vs the default, with paired-seed wins in brackets. The same one-scalar-per-head "
                "dial, started in two places. The last two columns are where the dial actually ended up.*")
    return "\n".join(rows)


def e6_kinit(res):
    per = res["metrics"]["per_head_dim"]
    hds = [int(h) for h in res["config"]["params"]["head_dims"] if str(h) in per]
    rows = ["| head width | multiplier at runtime (c = 4) | same factor folded into the key init | difference |",
            "|---|---|---|---|"]
    for hd in hds:
        A = per[str(hd)]["arms"]
        if "kinit_x4" not in A or "c4" not in A:
            continue
        rows.append(f"| head_dim {hd} | {A['c4']['delta_vs_baseline']:+.3f} | "
                    f"{A['kinit_x4']['delta_vs_baseline']:+.3f} | "
                    f"{A['kinit_x4']['delta_vs_baseline'] - A['c4']['delta_vs_baseline']:+.3f} |")
    rows.append("")
    rows.append("*Change in val bpc vs the default, three paired seeds. Both arms start from the same "
                "attention logit scale; only the runtime version keeps the factor out of the weights.*")
    return "\n".join(rows)


def e6_decomp(res):
    """Normalizer vs constant at a comparable initial logit scale."""
    per, by = res["metrics"]["per_head_dim"], res["metrics"]["by_arm_hd"]
    hds = [int(h) for h in res["config"]["params"]["head_dims"] if str(h) in per]
    rows = ["| head width | arm | initial logit std | Δ val bpc vs default |", "|---|---|---|---|"]
    for hd in hds:
        A = per[str(hd)]["arms"]
        # the fixed-c arm whose initial logit scale is closest to the normalizer's
        kn = by["knorm_only"][str(hd)]["init_logit_std_mean"]
        cands = [a for a in ("c2", "c4", "c8", "c16") if a in A]
        near = min(cands, key=lambda a: abs(by[a][str(hd)]["init_logit_std_mean"] - kn))
        for a, lab in ((near, f"constant, {LABEL[near].split(' ')[-1]}"), ("knorm_only", "key-only RMS-norm")):
            rows.append(f"| head_dim {hd} | {lab} | {by[a][str(hd)]['init_logit_std_mean']:.3f} | "
                        f"{A[a]['delta_vs_baseline']:+.3f} |")
        gap = A["knorm_only"]["delta_vs_baseline"] - A[near]["delta_vs_baseline"]
        rows.append(f"| head_dim {hd} | **cost of the normalizer at matched scale** | | **{gap:+.3f}** |")
    rows.append("")
    rows.append("*At a matched initial logit scale the only remaining difference is whether per-token key "
                "magnitude survives. The last row of each block is what destroying it costs.*")
    return "\n".join(rows)


def recipe(res6, res1):
    per = res6["metrics"]["per_head_dim"]
    hds = [int(h) for h in res6["config"]["params"]["head_dims"] if str(h) in per]
    rows = ["| if your head_dim is | do this | measured here |", "|---|---|---|"]
    for hd in hds:
        c = per[str(hd)]["best_fixed_c"]
        g = per[str(hd)]["best_fixed_c_bpc"] - per[str(hd)]["arms"]["baseline"]["bpc"]
        act = ("no change needed" if c == 1 else f"multiply keys by {c:g}")
        rows.append(f"| {hd} ({128 // hd} heads at d_model 128) | {act} | {g:+.3f} bpc |")
    rows.append("")
    rows.append("*The constant is not universal: it depends on the initialization scheme and the head width. "
                "Measure it once for your configuration with `python -m attnscale.bench --sweep-c`, which takes a few "
                "minutes on a CPU, rather than copying these values.*")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    e1 = load("2026-09-01_knorm-dynk-head-sweep")
    e2 = load("2026-09-01_knorm-dynk-ptb-transfer")
    e3 = load("2026-09-01_knorm-dynk-longer-training")
    e4 = load("2026-09-01_kscale-adaptive-vs-static")
    e5 = load("2026-09-01_fractional-norm-both-sides")
    e6 = load("2026-09-01_logit-scale-sweep")
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
    if e5:
        blocks["E5_TABLE"] = master_table(
            e5, ["baseline", "knorm_dynk", "fqn", "fqkn", "qknorm_dynq"],
            "Table 5. Which side carries the magnitude channel (registry 2026-09-01_fractional-norm-both-sides); "
            "comparison arms imported from the same-night head sweep.")
    if e6:
        blocks["E6_TABLE"] = master_table(
            e6, None, "Table 6. Fixed key multiplier against learnable and against the weight init "
                      "(registry 2026-09-01_logit-scale-sweep).")
        blocks["E6_CURVE"] = e6_curve(e6)
        blocks["E6_LEARN"] = e6_learn(e6)
        blocks["E6_DECOMP"] = e6_decomp(e6)
        blocks["E6_KINIT"] = e6_kinit(e6)
        if e1:
            blocks["RECIPE"] = recipe(e6, e1)
    blocks["ANCHOR_TABLE"] = anchor_table([e1, e2, e3, e4, e6])
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
