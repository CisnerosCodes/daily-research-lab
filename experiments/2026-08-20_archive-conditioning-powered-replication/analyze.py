#!/usr/bin/env python3
"""Analysis for 2026-08-20_archive-conditioning-powered-replication.

Reads raw_data.json (written by the generation/judging workflow), computes every
pre-registered metric under the criteria fixed in the README, writes results.json,
regenerates the RESULTS section of README.md, and draws chart.png. No number in
the README results section is written by hand.

Pre-registered criteria implemented here:
- H1: B-A distinct classes, supported iff mean paired diff >= 2 * SD(paired).
- H2: ceiling carve-out iff >= 2 of 3 estimators (chao1_bc, ace, jackknife2)
  show B-A mean paired diff >= 2 * SD(paired) in the POSITIVE direction;
  a negative-direction 2-of-3 is reported as anomalous; otherwise walks-faster.
- H3: C-B arm-unique classes, supported iff >= 2 * SD(paired); else drop arm C.
- Manipulation check: max pairwise arm gap in round-1 means vs 2 * SD pooled
  over all arm-replicate round-1 counts.
- Exclusions: replicate excluded if primary judge integrity error > 2 percent
  of IDs, or any arm chain has fewer than 8 complete rounds; > 2 exclusions
  fails the run.
"""
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).parent
ARMS = ["A", "B", "C"]
ROUNDS = list(range(1, 9))
IDEAS_PER_ROUND = 6


def load():
    return json.loads((HERE / "raw_data.json").read_text())


def build_partition(judge):
    part = {}
    for ci, cl in enumerate(judge["classes"]):
        for bid in cl["member_ids"]:
            part[bid] = f"c{ci}"
    for bid in judge.get("incoherent_ids", []):
        part[bid] = "INCOHERENT"
    return part


def partition_integrity(judge, expected_ids):
    assigned = []
    for cl in judge["classes"]:
        assigned.extend(cl["member_ids"])
    assigned.extend(judge.get("incoherent_ids", []))
    dupes = [k for k, v in Counter(assigned).items() if v > 1]
    missing = sorted(set(expected_ids) - set(assigned))
    extra = sorted(set(assigned) - set(expected_ids))
    n_bad = len(dupes) + len(missing)
    return {
        "n_dupes": len(dupes),
        "n_missing": len(missing),
        "n_extra": len(extra),
        "bad_fraction": round(n_bad / max(1, len(expected_ids)), 4),
    }


# ---------------------------------------------------------------- estimators
def chao1_bc(abund):
    s = len(abund)
    f1 = sum(1 for a in abund if a == 1)
    f2 = sum(1 for a in abund if a == 2)
    return s + (f1 * (f1 - 1)) / (2 * (f2 + 1))


def ace(abund, rare_cutoff=10):
    rare = [a for a in abund if a <= rare_cutoff]
    s_abund = sum(1 for a in abund if a > rare_cutoff)
    s_rare = len(rare)
    n_rare = sum(rare)
    f1 = sum(1 for a in rare if a == 1)
    if n_rare == 0:
        return float(len(abund))
    c_ace = 1 - f1 / n_rare
    if c_ace == 0:
        # All rare classes are singletons; ACE undefined. Fall back to Chao1-bc
        # and mark via NaN so the caller can count availability.
        return float("nan")
    fsum = sum(i * (i - 1) * sum(1 for a in rare if a == i) for i in range(1, rare_cutoff + 1))
    gamma2 = max((s_rare / c_ace) * fsum / (n_rare * (n_rare - 1)) - 1, 0) if n_rare > 1 else 0
    return s_abund + s_rare / c_ace + (f1 / c_ace) * gamma2


def jackknife2(abund, m):
    s = len(abund)
    f1 = sum(1 for a in abund if a == 1)
    f2 = sum(1 for a in abund if a == 2)
    if m < 2:
        return float(s)
    return s + f1 * (2 * m - 3) / m - f2 * ((m - 2) ** 2) / (m * (m - 1))


ESTIMATORS = ["chao1_bc", "ace", "jackknife2"]


def ari(labels_a, labels_b):
    ids = sorted(set(labels_a) & set(labels_b))
    if not ids:
        return float("nan")
    ct = defaultdict(int)
    for i in ids:
        ct[(labels_a[i], labels_b[i])] += 1
    a_marg = Counter(labels_a[i] for i in ids)
    b_marg = Counter(labels_b[i] for i in ids)
    n = len(ids)

    def c2(x):
        return x * (x - 1) / 2

    sum_ij = sum(c2(v) for v in ct.values())
    sum_a = sum(c2(v) for v in a_marg.values())
    sum_b = sum(c2(v) for v in b_marg.values())
    exp = sum_a * sum_b / c2(n) if c2(n) else 0.0
    mx = (sum_a + sum_b) / 2
    return (sum_ij - exp) / (mx - exp) if mx != exp else 1.0


def sd(xs):
    xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    return statistics.stdev(xs) if len(xs) > 1 else float("nan")


def mean(xs):
    xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    return statistics.mean(xs) if xs else float("nan")


def analyze():
    raw = load()
    ideas = raw["ideas"]
    all_seeds = sorted({i["seed"] for i in ideas})
    by_seed = defaultdict(list)
    for i in ideas:
        by_seed[i["seed"]].append(i)
    judges = defaultdict(dict)
    for j in raw["judges"]:
        judges[j["seed"]][j["judge"]] = j

    # ---------------- exclusions (pre-registered kill conditions 2 and 3)
    exclusions = {}
    for s in all_seeds:
        pool = by_seed[s]
        reasons = []
        for arm in ARMS:
            rounds_present = {i["round"] for i in pool if i["arm"] == arm}
            if rounds_present != set(ROUNDS):
                reasons.append(f"arm {arm} has rounds {sorted(rounds_present)} (needs all 8)")
        if 0 not in judges[s]:
            reasons.append("no primary judge")
        else:
            integ = partition_integrity(judges[s][0], [i["blind_id"] for i in pool])
            if integ["bad_fraction"] > 0.02:
                reasons.append(f"judge integrity bad_fraction {integ['bad_fraction']} > 0.02")
        if reasons:
            exclusions[s] = reasons
    seeds = [s for s in all_seeds if s not in exclusions]
    run_failed = len(exclusions) > 2

    integrity = {
        s: partition_integrity(judges[s][0], [i["blind_id"] for i in by_seed[s]])
        for s in all_seeds
        if 0 in judges[s]
    }

    # ---------------- per-seed, per-arm metrics (primary judge)
    per_seed = {}
    for s in seeds:
        pool = by_seed[s]
        part = build_partition(judges[s][0])
        seed_res = {}
        arm_classes = {}
        for arm in ARMS:
            arm_ideas = [i for i in pool if i["arm"] == arm]
            labels = [part.get(i["blind_id"], "MISSING") for i in arm_ideas]
            coherent = [l for l in labels if l not in ("INCOHERENT", "MISSING")]
            classes = set(coherent)
            arm_classes[arm] = classes
            acc = {}
            for r in ROUNDS:
                upto = [
                    part.get(i["blind_id"])
                    for i in arm_ideas
                    if i["round"] <= r
                    and part.get(i["blind_id"]) not in ("INCOHERENT", "MISSING", None)
                ]
                acc[r] = len(set(upto))
            abund = list(Counter(coherent).values())
            m = len(coherent)
            seed_res[arm] = {
                "distinct": len(classes),
                "accumulation": acc,
                "chao1_bc": round(chao1_bc(abund), 3),
                "ace": round(ace(abund), 3) if not math.isnan(ace(abund)) else None,
                "jackknife2": round(jackknife2(abund, m), 3),
                "incoherent": sum(1 for l in labels if l == "INCOHERENT"),
                "missing": sum(1 for l in labels if l == "MISSING"),
                "round1_distinct": acc[1],
            }
        for arm in ARMS:
            others = set().union(*(arm_classes[o] for o in ARMS if o != arm))
            seed_res[arm]["arm_unique"] = len(arm_classes[arm] - others)
        per_seed[s] = seed_res

    def series(arm, key):
        out = []
        for s in seeds:
            v = per_seed[s][arm][key]
            out.append(float("nan") if v is None else v)
        return out

    def contrast(key, hi, lo):
        diffs = []
        for s in seeds:
            a, b = per_seed[s][hi][key], per_seed[s][lo][key]
            if a is None or b is None:
                continue
            diffs.append(a - b)
        m_ = mean(diffs)
        s_paired = sd(diffs)
        s_ctrl = sd(series(lo, key))
        return {
            "n_pairs": len(diffs),
            "paired_diffs": [round(d, 3) for d in diffs],
            "mean_diff": round(m_, 3),
            "sd_paired": round(s_paired, 3) if not math.isnan(s_paired) else None,
            "sd_control_arm": round(s_ctrl, 3) if not math.isnan(s_ctrl) else None,
            "x_sd_paired": round(m_ / s_paired, 2) if s_paired and not math.isnan(s_paired) else None,
            "x_sd_control": round(m_ / s_ctrl, 2) if s_ctrl and not math.isnan(s_ctrl) else None,
        }

    arms_summary = {}
    for arm in ARMS:
        acc_mean = {
            r: round(mean([per_seed[s][arm]["accumulation"][r] for s in seeds]), 2)
            for r in ROUNDS
        }
        arms_summary[arm] = {
            "distinct_mean": round(mean(series(arm, "distinct")), 2),
            "distinct_sd": round(sd(series(arm, "distinct")), 2),
            "chao1_mean": round(mean(series(arm, "chao1_bc")), 2),
            "chao1_sd": round(sd(series(arm, "chao1_bc")), 2),
            "ace_mean": round(mean(series(arm, "ace")), 2),
            "ace_sd": round(sd(series(arm, "ace")), 2),
            "jk2_mean": round(mean(series(arm, "jackknife2")), 2),
            "jk2_sd": round(sd(series(arm, "jackknife2")), 2),
            "arm_unique_mean": round(mean(series(arm, "arm_unique")), 2),
            "incoherent_total": int(sum(series(arm, "incoherent"))),
            "round1_distinct_mean": round(mean(series(arm, "round1_distinct")), 2),
            "accumulation_mean": acc_mean,
        }

    contrasts = {
        "H1_B_minus_A_distinct": contrast("distinct", "B", "A"),
        "H2_B_minus_A_chao1_bc": contrast("chao1_bc", "B", "A"),
        "H2_B_minus_A_ace": contrast("ace", "B", "A"),
        "H2_B_minus_A_jackknife2": contrast("jackknife2", "B", "A"),
        "H3_C_minus_B_arm_unique": contrast("arm_unique", "C", "B"),
        "C_minus_A_distinct": contrast("distinct", "C", "A"),
        "C_minus_A_chao1_bc": contrast("chao1_bc", "C", "A"),
        "C_minus_B_incoherent": contrast("incoherent", "C", "B"),
    }

    # ---------------- manipulation check (pooled sigma over all cells)
    r1_all = []
    r1_by_arm = {}
    for arm in ARMS:
        vals = series(arm, "round1_distinct")
        r1_by_arm[arm] = vals
        r1_all.extend(vals)
    pooled_sd = sd(r1_all)
    arm_means = {a: mean(r1_by_arm[a]) for a in ARMS}
    max_gap = max(abs(arm_means[x] - arm_means[y]) for x, y in combinations(ARMS, 2))
    if math.isnan(pooled_sd):
        manip_pass = None
    elif pooled_sd == 0:
        manip_pass = max_gap == 0
    else:
        manip_pass = max_gap <= 2 * pooled_sd
    manip = {
        "round1_means": {a: round(arm_means[a], 2) for a in ARMS},
        "pooled_sd_all_cells": round(pooled_sd, 3) if not math.isnan(pooled_sd) else None,
        "max_pairwise_gap": round(max_gap, 3),
        "pass": manip_pass,
    }

    # ---------------- judge noise (all multi-judged seeds)
    judge_noise = []
    for s in seeds:
        if len(judges[s]) <= 1:
            continue
        parts = {ji: build_partition(j) for ji, j in judges[s].items()}
        pool = by_seed[s]
        per_judge_counts = {
            ji: {
                arm: len(
                    {
                        part.get(i["blind_id"])
                        for i in pool
                        if i["arm"] == arm
                        and part.get(i["blind_id"]) not in ("INCOHERENT", None)
                    }
                )
                for arm in ARMS
            }
            for ji, part in parts.items()
        }
        aris = [round(ari(parts[x], parts[y]), 3) for x, y in combinations(sorted(parts), 2)]
        judge_noise.append(
            {
                "seed": s,
                "per_judge_distinct": per_judge_counts,
                "between_judge_sd": {
                    arm: round(sd([per_judge_counts[ji][arm] for ji in sorted(parts)]), 2)
                    for arm in ARMS
                },
                "pairwise_ari": aris,
                "mean_ari": round(mean(aris), 3),
            }
        )

    # ---------------- pre-registered verdicts
    def clears(c):
        return (
            c["x_sd_paired"] is not None
            and c["mean_diff"] > 0
            and c["x_sd_paired"] >= 2
        )

    h1 = contrasts["H1_B_minus_A_distinct"]
    v_h1 = (
        f"supported ({h1['x_sd_paired']}x paired SD)"
        if clears(h1)
        else f"not supported ({h1['x_sd_paired']}x paired SD, below 2)"
    )

    est_keys = {
        "chao1_bc": "H2_B_minus_A_chao1_bc",
        "ace": "H2_B_minus_A_ace",
        "jackknife2": "H2_B_minus_A_jackknife2",
    }
    pos_clear = [e for e, k in est_keys.items() if clears(contrasts[k])]
    neg_clear = [
        e
        for e, k in est_keys.items()
        if contrasts[k]["x_sd_paired"] is not None
        and contrasts[k]["mean_diff"] < 0
        and abs(contrasts[k]["x_sd_paired"]) >= 2
    ]
    if len(pos_clear) >= 2:
        v_h2 = f"CEILING CARVE-OUT: {len(pos_clear)}/3 estimators >= 2x paired SD positive ({', '.join(pos_clear)})"
    elif len(neg_clear) >= 2:
        v_h2 = f"anomalous negative direction: {len(neg_clear)}/3 estimators >= 2x paired SD negative ({', '.join(neg_clear)}) — unpredicted, reported as such"
    else:
        v_h2 = f"walks-faster stands at doubled power ({len(pos_clear)}/3 estimators clear the positive bar)"

    h3 = contrasts["H3_C_minus_B_arm_unique"]
    v_h3 = (
        f"supported ({h3['x_sd_paired']}x paired SD) — arm C stays"
        if clears(h3)
        else f"not supported ({h3['x_sd_paired']}x paired SD) — ARM C IS DROPPED from the programme per the pilot's commitment"
    )

    results = {
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "n_seeds_analyzed": len(seeds),
        "n_seeds_excluded": len(exclusions),
        "exclusions": {str(k): v for k, v in exclusions.items()},
        "run_failed_by_exclusion_rule": run_failed,
        "n_ideas": len(ideas),
        "partition_integrity": {str(k): v for k, v in integrity.items()},
        "per_seed": {str(k): v for k, v in per_seed.items()},
        "arms": arms_summary,
        "contrasts": contrasts,
        "manipulation_check_round1": manip,
        "judge_noise": judge_noise,
        "verdicts": {"H1_rate": v_h1, "H2_asymptote": v_h2, "H3_operator": v_h3},
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=2))
    write_readme(results)
    try:
        chart(results)
    except Exception as e:
        print(f"chart skipped: {e}")
    print(json.dumps(results["verdicts"], indent=2))
    return results


def write_readme(res):
    readme = HERE / "README.md"
    text = readme.read_text()
    marker = "## RESULTS"
    head = text.split(marker)[0]
    a = res["arms"]
    c = res["contrasts"]
    m = res["manipulation_check_round1"]
    lines = [
        marker + " (written by analyze.py after the pre-registration commit)",
        "",
        f"Analyzed {res['n_ideas']} ideas; {res['n_seeds_analyzed']} replicates analyzed, "
        f"{res['n_seeds_excluded']} excluded"
        + (f" ({res['exclusions']})" if res["exclusions"] else "")
        + f". Run failed by exclusion rule: {res['run_failed_by_exclusion_rule']}.",
        "",
        f"Manipulation check (round-1 exchangeability, pooled sigma): means {m['round1_means']}, "
        f"max pairwise gap {m['max_pairwise_gap']} vs 2 x pooled SD = "
        f"{None if m['pooled_sd_all_cells'] is None else round(2 * m['pooled_sd_all_cells'], 3)} — "
        f"**{'PASS' if m['pass'] else 'FAIL' if m['pass'] is not None else 'indeterminate'}**.",
        "",
        "| Arm | Distinct /48 (mean ± SD) | Chao1-bc | ACE | Jackknife-2 | Arm-unique | Incoherent | Accumulation r1→r8 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm, label in [("A", "A iid control"), ("B", "B archive feedback"), ("C", "C assumption negation")]:
        acc = a[arm]["accumulation_mean"]
        acc_str = " → ".join(str(acc[r]) for r in ROUNDS)
        lines.append(
            f"| {label} | {a[arm]['distinct_mean']} ± {a[arm]['distinct_sd']} | "
            f"{a[arm]['chao1_mean']} ± {a[arm]['chao1_sd']} | "
            f"{a[arm]['ace_mean']} ± {a[arm]['ace_sd']} | "
            f"{a[arm]['jk2_mean']} ± {a[arm]['jk2_sd']} | "
            f"{a[arm]['arm_unique_mean']} | {a[arm]['incoherent_total']} | {acc_str} |"
        )
    lines += [
        "",
        "**Contrasts (paired per-replicate differences; the pre-registered criterion is x SD(paired) >= 2):**",
        "",
        "| Contrast | n pairs | Mean diff | x SD(paired) | x SD(control) |",
        "|---|---|---|---|---|",
    ]
    for key, label in [
        ("H1_B_minus_A_distinct", "H1: B − A, distinct classes"),
        ("H2_B_minus_A_chao1_bc", "H2: B − A, Chao1-bc"),
        ("H2_B_minus_A_ace", "H2: B − A, ACE"),
        ("H2_B_minus_A_jackknife2", "H2: B − A, jackknife-2"),
        ("C_minus_A_distinct", "C − A, distinct classes"),
        ("C_minus_A_chao1_bc", "C − A, Chao1-bc"),
        ("H3_C_minus_B_arm_unique", "H3: C − B, arm-unique classes"),
        ("C_minus_B_incoherent", "C − B, incoherent count"),
    ]:
        cc = c[key]
        lines.append(
            f"| {label} | {cc['n_pairs']} | {cc['mean_diff']} | {cc['x_sd_paired']} | {cc['x_sd_control']} |"
        )
    for jn in res["judge_noise"]:
        lines += [
            "",
            f"**Judge noise** (replicate {jn['seed']}, {len(jn['per_judge_distinct'])} independent judges): "
            f"between-judge SD of per-arm distinct counts {jn['between_judge_sd']}; "
            f"pairwise ARI {jn['pairwise_ari']} (mean {jn['mean_ari']}).",
        ]
    lines += [
        "",
        "**Verdicts against the pre-registered criteria:**",
        "",
        f"- H1 (rate): {res['verdicts']['H1_rate']}",
        f"- H2 (asymptote): {res['verdicts']['H2_asymptote']}",
        f"- H3 (operator): {res['verdicts']['H3_operator']}",
        "",
    ]
    readme.write_text(head + "\n".join(lines) + "\n")


def chart(res):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = res["arms"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = {"A": "#888888", "B": "#2b8cbe", "C": "#d95f0e"}
    names = {"A": "A iid", "B": "B archive", "C": "C negation"}
    for arm in ARMS:
        acc = a[arm]["accumulation_mean"]
        axes[0].plot(ROUNDS, [acc[r] for r in ROUNDS], marker="o", color=colors[arm], label=names[arm])
    axes[0].set_xlabel("round")
    axes[0].set_ylabel("distinct classes (mean over replicates)")
    axes[0].set_title("Accumulation (8 rounds)")
    axes[0].legend()
    est_labels = ["chao1", "ace", "jk2"]
    est_means = {
        "A": [a["A"]["chao1_mean"], a["A"]["ace_mean"], a["A"]["jk2_mean"]],
        "B": [a["B"]["chao1_mean"], a["B"]["ace_mean"], a["B"]["jk2_mean"]],
        "C": [a["C"]["chao1_mean"], a["C"]["ace_mean"], a["C"]["jk2_mean"]],
    }
    est_sds = {
        "A": [a["A"]["chao1_sd"], a["A"]["ace_sd"], a["A"]["jk2_sd"]],
        "B": [a["B"]["chao1_sd"], a["B"]["ace_sd"], a["B"]["jk2_sd"]],
        "C": [a["C"]["chao1_sd"], a["C"]["ace_sd"], a["C"]["jk2_sd"]],
    }
    w = 0.25
    xs = range(len(est_labels))
    for k, arm in enumerate(ARMS):
        axes[1].bar(
            [x + (k - 1) * w for x in xs],
            est_means[arm],
            width=w,
            yerr=est_sds[arm],
            color=colors[arm],
            capsize=3,
            label=names[arm],
        )
    axes[1].set_xticks(list(xs), est_labels)
    axes[1].set_title("Estimated support (battery)")
    axes[1].legend()
    fig.suptitle("Archive conditioning, powered replication")
    fig.tight_layout()
    fig.savefig(HERE / "chart.png", dpi=120)


if __name__ == "__main__":
    analyze()
