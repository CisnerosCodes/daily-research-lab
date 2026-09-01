#!/usr/bin/env python3
"""Analysis for 2026-08-13_archive-conditioning-rate-vs-asymptote.

Reads raw_data.json (written by the generation/judging workflow), computes every
pre-registered metric, writes results.json, regenerates the RESULTS section of
README.md, and draws chart.png. No number in the README results section is
written by hand.
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
ROUNDS = [1, 2, 3, 4]


def load():
    return json.loads((HERE / "raw_data.json").read_text())


def build_partition(judge):
    """blind_id -> class label; incoherent ids get their own 'INCOHERENT' label."""
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
    return {"n_dupes": len(dupes), "n_missing": len(missing), "n_extra": len(extra)}


def chao1_bc(abundances):
    s_obs = len(abundances)
    f1 = sum(1 for a in abundances if a == 1)
    f2 = sum(1 for a in abundances if a == 2)
    return s_obs + (f1 * (f1 - 1)) / (2 * (f2 + 1))


def ari(labels_a, labels_b):
    """Adjusted Rand index between two labelings given as dicts id->label."""
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


def analyze():
    raw = load()
    ideas = raw["ideas"]
    seeds = sorted({i["seed"] for i in ideas})
    by_seed = defaultdict(list)
    for i in ideas:
        by_seed[i["seed"]].append(i)

    judges = defaultdict(dict)  # seed -> judge_idx -> judge obj
    for j in raw["judges"]:
        judges[j["seed"]][j["judge"]] = j

    integrity = {}
    per_seed = {}
    for s in seeds:
        pool = by_seed[s]
        expected = [i["blind_id"] for i in pool]
        j0 = judges[s][0]
        integrity[s] = partition_integrity(j0, expected)
        part = build_partition(j0)
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
            abund = Counter(coherent)
            seed_res[arm] = {
                "distinct": len(classes),
                "accumulation": acc,
                "chao1_bc": round(chao1_bc(list(abund.values())), 3),
                "incoherent": sum(1 for l in labels if l == "INCOHERENT"),
                "missing": sum(1 for l in labels if l == "MISSING"),
                "round1_distinct": acc[1],
            }
        for arm in ARMS:
            others = set().union(*(arm_classes[o] for o in ARMS if o != arm))
            seed_res[arm]["arm_unique"] = len(arm_classes[arm] - others)
        per_seed[s] = seed_res

    def series(arm, key):
        return [per_seed[s][arm][key] for s in seeds]

    def sd(xs):
        return statistics.stdev(xs) if len(xs) > 1 else float("nan")

    def contrast(key, hi, lo):
        diffs = [per_seed[s][hi][key] - per_seed[s][lo][key] for s in seeds]
        m = statistics.mean(diffs)
        s_paired = sd(diffs)
        s_ctrl = sd(series(lo, key))
        return {
            "paired_diffs": diffs,
            "mean_diff": round(m, 3),
            "sd_paired": round(s_paired, 3),
            "sd_control_arm": round(s_ctrl, 3),
            "x_sd_paired": round(m / s_paired, 2) if s_paired else None,
            "x_sd_control": round(m / s_ctrl, 2) if s_ctrl else None,
        }

    arms_summary = {
        arm: {
            "distinct_mean": round(statistics.mean(series(arm, "distinct")), 2),
            "distinct_sd": round(sd(series(arm, "distinct")), 2),
            "chao1_mean": round(statistics.mean(series(arm, "chao1_bc")), 2),
            "chao1_sd": round(sd(series(arm, "chao1_bc")), 2),
            "arm_unique_mean": round(statistics.mean(series(arm, "arm_unique")), 2),
            "incoherent_total": sum(series(arm, "incoherent")),
            "round1_distinct_mean": round(
                statistics.mean(series(arm, "round1_distinct")), 2
            ),
            "accumulation_mean": {
                r: round(
                    statistics.mean(per_seed[s][arm]["accumulation"][r] for s in seeds), 2
                )
                for r in ROUNDS
            },
        }
        for arm in ARMS
    }

    contrasts = {
        "H1_B_minus_A_distinct": contrast("distinct", "B", "A"),
        "H2_B_minus_A_chao1": contrast("chao1_bc", "B", "A"),
        "H3_C_minus_B_arm_unique": contrast("arm_unique", "C", "B"),
        "C_minus_A_distinct": contrast("distinct", "C", "A"),
        "C_minus_A_chao1": contrast("chao1_bc", "C", "A"),
        "C_minus_B_incoherent": contrast("incoherent", "C", "B"),
    }

    # Manipulation check: round-1 exchangeability.
    r1 = {arm: series(arm, "round1_distinct") for arm in ARMS}
    r1_sd = sd(r1["A"])
    r1_maxgap = max(
        abs(statistics.mean(r1[x]) - statistics.mean(r1[y]))
        for x, y in combinations(ARMS, 2)
    )
    manip = {
        "round1_means": {a: round(statistics.mean(r1[a]), 2) for a in ARMS},
        "round1_sd_A": round(r1_sd, 3) if not math.isnan(r1_sd) else None,
        "max_pairwise_gap": round(r1_maxgap, 3),
        "pass": bool(r1_maxgap <= 2 * r1_sd) if r1_sd and not math.isnan(r1_sd) else None,
    }

    # Judge noise on the multi-judged seed.
    jn = None
    multi = [s for s in seeds if len(judges[s]) > 1]
    if multi:
        s = multi[0]
        parts = {ji: build_partition(j) for ji, j in judges[s].items()}
        pool = by_seed[s]
        per_judge_counts = {}
        for ji, part in parts.items():
            per_judge_counts[ji] = {
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
        aris = [
            round(ari(parts[x], parts[y]), 3) for x, y in combinations(sorted(parts), 2)
        ]
        jn = {
            "seed": s,
            "per_judge_distinct": per_judge_counts,
            "between_judge_sd": {
                arm: round(sd([per_judge_counts[ji][arm] for ji in sorted(parts)]), 2)
                for arm in ARMS
            },
            "pairwise_ari": aris,
            "mean_ari": round(statistics.mean(aris), 3),
        }

    def verdict_h1():
        c = contrasts["H1_B_minus_A_distinct"]
        ref = min(x for x in (c["x_sd_paired"], c["x_sd_control"]) if x is not None)
        return "supported" if ref >= 2 else "not supported (below 2 sigma)"

    def verdict_h2():
        c = contrasts["H2_B_minus_A_chao1"]
        xs = [abs(x) for x in (c["x_sd_paired"], c["x_sd_control"]) if x is not None]
        return (
            "supported (Chao1 within 2 sigma: walks faster, not higher ceiling)"
            if xs and max(xs) < 2
            else "falsified (Chao1 moved by >= 2 sigma)"
        )

    def verdict_h3():
        c = contrasts["H3_C_minus_B_arm_unique"]
        return "supported" if c["mean_diff"] > 0 else "not supported"

    results = {
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "n_seeds": len(seeds),
        "n_ideas": len(ideas),
        "partition_integrity": {str(k): v for k, v in integrity.items()},
        "per_seed": {str(k): v for k, v in per_seed.items()},
        "arms": arms_summary,
        "contrasts": contrasts,
        "manipulation_check_round1": manip,
        "judge_noise": jn,
        "verdicts": {"H1_rate": verdict_h1(), "H2_asymptote": verdict_h2(), "H3_operator": verdict_h3()},
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=2))
    write_readme(results)
    try:
        chart(results)
    except Exception as e:  # chart is decorative; never block results on it
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
    jn = res["judge_noise"]
    lines = [
        marker + " (written by analyze.py after the pre-registration commit)",
        "",
        f"Analyzed {res['n_ideas']} ideas across {res['n_seeds']} replicates. "
        f"Manipulation check (round-1 exchangeability): means {m['round1_means']}, max pairwise gap {m['max_pairwise_gap']} vs 2 sigma = {None if m['round1_sd_A'] is None else round(2 * m['round1_sd_A'], 3)} — **{'PASS' if m['pass'] else 'FAIL' if m['pass'] is not None else 'indeterminate'}**.",
        "",
        "| Arm | Distinct classes /24 (mean ± SD) | Chao1-bc (mean ± SD) | Arm-unique classes | Incoherent (total) | Accumulation r1→r4 |",
        "|---|---|---|---|---|---|",
    ]
    for arm, label in [("A", "A iid control"), ("B", "B archive feedback"), ("C", "C assumption negation")]:
        acc = a[arm]["accumulation_mean"]
        lines.append(
            f"| {label} | {a[arm]['distinct_mean']} ± {a[arm]['distinct_sd']} | "
            f"{a[arm]['chao1_mean']} ± {a[arm]['chao1_sd']} | {a[arm]['arm_unique_mean']} | "
            f"{a[arm]['incoherent_total']} | {acc[1]} → {acc[2]} → {acc[3]} → {acc[4]} |"
        )
    lines += [
        "",
        "**Contrasts (paired per-replicate differences, as multiples of noise):**",
        "",
        "| Contrast | Mean diff | x SD(paired) | x SD(control arm) |",
        "|---|---|---|---|",
    ]
    for key, label in [
        ("H1_B_minus_A_distinct", "H1: B − A, distinct classes"),
        ("H2_B_minus_A_chao1", "H2: B − A, Chao1-bc"),
        ("C_minus_A_distinct", "C − A, distinct classes"),
        ("C_minus_A_chao1", "C − A, Chao1-bc"),
        ("H3_C_minus_B_arm_unique", "H3: C − B, arm-unique classes"),
        ("C_minus_B_incoherent", "C − B, incoherent count"),
    ]:
        cc = c[key]
        lines.append(
            f"| {label} | {cc['mean_diff']} | {cc['x_sd_paired']} | {cc['x_sd_control']} |"
        )
    if jn:
        lines += [
            "",
            f"**Judge noise** (replicate {jn['seed']}, three independent judges): between-judge SD of per-arm distinct counts "
            f"{jn['between_judge_sd']}; pairwise adjusted Rand index {jn['pairwise_ari']} (mean {jn['mean_ari']}). "
            "Every count above is conditional on a judge partition; this is the measured width of that conditionality.",
        ]
    lines += [
        "",
        "**Verdicts against the pre-registered conditions:**",
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
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = {"A": "#888888", "B": "#2b8cbe", "C": "#d95f0e"}
    names = {"A": "A iid", "B": "B archive", "C": "C negation"}
    for arm in ARMS:
        acc = a[arm]["accumulation_mean"]
        axes[0].plot(ROUNDS, [acc[r] for r in ROUNDS], marker="o", color=colors[arm], label=names[arm])
    axes[0].set_xlabel("round")
    axes[0].set_ylabel("distinct classes (mean over replicates)")
    axes[0].set_title("Accumulation")
    axes[0].legend()
    xs = range(len(ARMS))
    axes[1].bar(
        xs,
        [a[arm]["chao1_mean"] for arm in ARMS],
        yerr=[a[arm]["chao1_sd"] for arm in ARMS],
        color=[colors[arm] for arm in ARMS],
        capsize=4,
    )
    axes[1].set_xticks(list(xs), [names[arm] for arm in ARMS])
    axes[1].set_title("Chao1-bc estimated support")
    fig.suptitle("Archive conditioning: rate vs asymptote")
    fig.tight_layout()
    fig.savefig(HERE / "chart.png", dpi=120)


if __name__ == "__main__":
    analyze()
