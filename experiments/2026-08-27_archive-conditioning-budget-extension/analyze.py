#!/usr/bin/env python3
"""Analysis for 2026-08-27_archive-conditioning-budget-extension.

Reads raw_data.json (written by the generation/judging workflow), computes every
pre-registered metric under the criteria fixed in the README, writes results.json,
regenerates the RESULTS section of README.md, and draws chart.png. No number in
the README results section is written by hand.

Pre-registered criteria implemented here:
- H4 (primary, crossing): per replicate, B_observed_distinct(96) minus
  max(chao1_bc(A), ace(A), jackknife2(A)). Walks-faster is FALSIFIED iff the
  mean paired difference is positive and >= 2 * SD(paired). Per-estimator
  crossings are secondary descriptives only.
- H5 (gap curvature): B-A distinct-class gap at rounds 4, 8, 12, 16.
  Turnover iff (max gap - final gap) >= 1 * SD(paired of that quantity).
  Monotone growth iff (gap_r16 - gap_r8) >= 2 * SD(paired).
  Both are computed and reported; they are not mutually exclusive by
  construction, so the verdict names whichever fire.
- H6 (estimator stability, self-audit): arm A only, each estimator at 48 vs 96
  ideas, paired. Stable iff mean paired increase < 2 * SD(paired).
- Manipulation check: max pairwise arm gap in round-1 means vs 2 * SD pooled
  over all arm-replicate round-1 counts.
- Exclusions: replicate excluded if primary judge integrity error > 2 percent
  of IDs, or any arm chain has fewer than 16 complete rounds; > 2 exclusions
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
ARMS = ["A", "B"]
ROUNDS = list(range(1, 17))
IDEAS_PER_ROUND = 6
HALF_ROUND = 8  # rounds 1..8 == the first 48 ideas, matching the prior run
GAP_ROUNDS = [4, 8, 12, 16]


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


def estimate_all(coherent_labels):
    """All three estimators over a list of per-idea class labels."""
    abund = list(Counter(coherent_labels).values())
    m = len(coherent_labels)
    a = ace(abund)
    return {
        "chao1_bc": round(chao1_bc(abund), 3),
        "ace": None if math.isnan(a) else round(a, 3),
        "jackknife2": round(jackknife2(abund, m), 3),
    }


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


def paired(diffs):
    """Summarize a list of paired per-replicate differences."""
    diffs = [d for d in diffs if not (isinstance(d, float) and math.isnan(d))]
    m_ = mean(diffs)
    s_ = sd(diffs)
    return {
        "n_pairs": len(diffs),
        "paired_diffs": [round(d, 3) for d in diffs],
        "mean_diff": round(m_, 3) if not math.isnan(m_) else None,
        "sd_paired": round(s_, 3) if not math.isnan(s_) else None,
        "x_sd_paired": round(m_ / s_, 2) if s_ and not math.isnan(s_) else None,
    }


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
                reasons.append(f"arm {arm} has {len(rounds_present)} rounds (needs all 16)")
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
        for arm in ARMS:
            arm_ideas = sorted(
                [i for i in pool if i["arm"] == arm], key=lambda x: (x["round"], x["idx"])
            )
            labels = [part.get(i["blind_id"], "MISSING") for i in arm_ideas]
            coherent_full = [
                part[i["blind_id"]]
                for i in arm_ideas
                if part.get(i["blind_id"]) not in ("INCOHERENT", "MISSING", None)
            ]
            coherent_half = [
                part[i["blind_id"]]
                for i in arm_ideas
                if i["round"] <= HALF_ROUND
                and part.get(i["blind_id"]) not in ("INCOHERENT", "MISSING", None)
            ]
            acc = {}
            for r in ROUNDS:
                upto = [
                    part.get(i["blind_id"])
                    for i in arm_ideas
                    if i["round"] <= r
                    and part.get(i["blind_id"]) not in ("INCOHERENT", "MISSING", None)
                ]
                acc[r] = len(set(upto))
            est96 = estimate_all(coherent_full)
            est48 = estimate_all(coherent_half)
            seed_res[arm] = {
                "distinct": len(set(coherent_full)),
                "distinct_at_48": len(set(coherent_half)),
                "accumulation": acc,
                "est_96": est96,
                "est_48": est48,
                "incoherent": sum(1 for l in labels if l == "INCOHERENT"),
                "incoherent_first_half": sum(
                    1
                    for i in arm_ideas
                    if i["round"] <= HALF_ROUND and part.get(i["blind_id"]) == "INCOHERENT"
                ),
                "incoherent_second_half": sum(
                    1
                    for i in arm_ideas
                    if i["round"] > HALF_ROUND and part.get(i["blind_id"]) == "INCOHERENT"
                ),
                "missing": sum(1 for l in labels if l == "MISSING"),
                "round1_distinct": acc[1],
            }
            for k in ESTIMATORS:
                seed_res[arm][k] = est96[k]
        # A's ceiling = the maximum (least downward-biased) of its three estimators
        a_vals = [v for v in seed_res["A"]["est_96"].values() if v is not None]
        seed_res["A"]["ceiling_max_estimator"] = round(max(a_vals), 3) if a_vals else None
        per_seed[s] = seed_res

    # ---------------- H4: crossing test (primary)
    h4_diffs, h4_per_est = [], {k: [] for k in ESTIMATORS}
    for s in seeds:
        ceil_ = per_seed[s]["A"]["ceiling_max_estimator"]
        bobs = per_seed[s]["B"]["distinct"]
        if ceil_ is not None:
            h4_diffs.append(bobs - ceil_)
        for k in ESTIMATORS:
            v = per_seed[s]["A"]["est_96"][k]
            if v is not None:
                h4_per_est[k].append(bobs - v)
    h4 = paired(h4_diffs)
    h4["n_replicates_crossed"] = sum(1 for d in h4_diffs if d > 0)
    h4_secondary = {k: paired(v) for k, v in h4_per_est.items()}
    for k in ESTIMATORS:
        h4_secondary[k]["n_replicates_crossed"] = sum(1 for d in h4_per_est[k] if d > 0)

    # ---------------- H5: gap curvature
    gaps = {r: [] for r in GAP_ROUNDS}
    per_seed_gap = {}
    for s in seeds:
        g = {r: per_seed[s]["B"]["accumulation"][r] - per_seed[s]["A"]["accumulation"][r]
             for r in ROUNDS}
        per_seed_gap[s] = g
        for r in GAP_ROUNDS:
            gaps[r].append(g[r])
    turnover_diffs = [max(per_seed_gap[s].values()) - per_seed_gap[s][16] for s in seeds]
    growth_diffs = [per_seed_gap[s][16] - per_seed_gap[s][8] for s in seeds]
    h5 = {
        "gap_by_round": {r: round(mean(gaps[r]), 2) for r in GAP_ROUNDS},
        "gap_full_curve": {r: round(mean([per_seed_gap[s][r] for s in seeds]), 2) for r in ROUNDS},
        "turnover": paired(turnover_diffs),
        "growth_r16_minus_r8": paired(growth_diffs),
        "argmax_gap_round_per_seed": {
            str(s): max(per_seed_gap[s], key=lambda r: per_seed_gap[s][r]) for s in seeds
        },
    }

    # ---------------- H6: estimator stability on arm A (self-audit)
    h6 = {}
    for k in ESTIMATORS:
        d = []
        for s in seeds:
            lo, hi = per_seed[s]["A"]["est_48"][k], per_seed[s]["A"]["est_96"][k]
            if lo is not None and hi is not None:
                d.append(hi - lo)
        h6[k] = paired(d)
        h6[k]["mean_at_48"] = round(
            mean([per_seed[s]["A"]["est_48"][k] for s in seeds
                  if per_seed[s]["A"]["est_48"][k] is not None]), 2)
        h6[k]["mean_at_96"] = round(
            mean([per_seed[s]["A"]["est_96"][k] for s in seeds
                  if per_seed[s]["A"]["est_96"][k] is not None]), 2)

    # ---------------- descriptive arm summary + the rate contrast, for continuity
    def series(arm, key):
        return [per_seed[s][arm][key] for s in seeds]

    arms_summary = {}
    for arm in ARMS:
        arms_summary[arm] = {
            "distinct_mean": round(mean(series(arm, "distinct")), 2),
            "distinct_sd": round(sd(series(arm, "distinct")), 2),
            "distinct_at_48_mean": round(mean(series(arm, "distinct_at_48")), 2),
            "incoherent_total": int(sum(series(arm, "incoherent"))),
            "incoherent_first_half": int(sum(series(arm, "incoherent_first_half"))),
            "incoherent_second_half": int(sum(series(arm, "incoherent_second_half"))),
            "round1_distinct_mean": round(mean(series(arm, "round1_distinct")), 2),
            "accumulation_mean": {
                r: round(mean([per_seed[s][arm]["accumulation"][r] for s in seeds]), 2)
                for r in ROUNDS
            },
        }
        for k in ESTIMATORS:
            vals = [v for v in series(arm, k) if v is not None]
            arms_summary[arm][k + "_mean"] = round(mean(vals), 2)
            arms_summary[arm][k + "_sd"] = round(sd(vals), 2) if len(vals) > 1 else None

    rate_contrast = paired(
        [per_seed[s]["B"]["distinct"] - per_seed[s]["A"]["distinct"] for s in seeds]
    )
    rate_contrast["sd_control_arm"] = round(sd(series("A", "distinct")), 3)
    rate_contrast["x_sd_control"] = (
        round(rate_contrast["mean_diff"] / rate_contrast["sd_control_arm"], 2)
        if rate_contrast["sd_control_arm"] else None
    )

    # ---------------- manipulation check (pooled sigma over all cells)
    r1_all, r1_by_arm = [], {}
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

    # ---------------- judge noise
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
    def clears(p, k=2):
        return (
            p["x_sd_paired"] is not None
            and p["mean_diff"] is not None
            and p["mean_diff"] > 0
            and p["x_sd_paired"] >= k
        )

    if clears(h4):
        v_h4 = (
            f"WALKS-FASTER FALSIFIED: B's observed count exceeds A's maximum estimator by "
            f"{h4['mean_diff']} ({h4['x_sd_paired']}x paired SD), crossing in "
            f"{h4['n_replicates_crossed']}/{len(seeds)} replicates. Necessary but not sufficient "
            f"for support extension — the estimators are downward-biased, as pre-registered."
        )
    else:
        v_h4 = (
            f"walks-faster survives: B's observed count minus A's maximum estimator is "
            f"{h4['mean_diff']} ({h4['x_sd_paired']}x paired SD, below the 2x bar); crossed in "
            f"{h4['n_replicates_crossed']}/{len(seeds)} replicates"
        )

    turn = h5["turnover"]
    grow = h5["growth_r16_minus_r8"]
    turn_fired = clears(turn, k=1)
    grow_fired = clears(grow, k=2)
    parts_v = []
    if turn_fired:
        parts_v.append(f"TURNOVER detected ({turn['mean_diff']} below peak, {turn['x_sd_paired']}x paired SD)")
    if grow_fired:
        parts_v.append(f"MONOTONE GROWTH detected (r16-r8 = {grow['mean_diff']}, {grow['x_sd_paired']}x paired SD)")
    if not parts_v:
        parts_v.append(
            f"neither criterion fires (turnover {turn['x_sd_paired']}x vs 1x bar; "
            f"growth {grow['x_sd_paired']}x vs 2x bar) — gap shape indeterminate at this n"
        )
    v_h5 = "; ".join(parts_v)

    unstable = [k for k in ESTIMATORS if clears(h6[k])]
    if unstable:
        v_h6 = (
            f"NOT stable: {', '.join(unstable)} climb with budget at >= 2x paired SD "
            f"(48 -> 96 ideas). Every estimated ceiling this lab has published carries a "
            f"budget-dependence caveat, as predicted against ourselves in the pre-registration."
        )
    else:
        v_h6 = "stable: no estimator's 48-to-96 increase clears 2x paired SD"

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
        "H4_crossing_vs_max_estimator": h4,
        "H4_secondary_per_estimator": h4_secondary,
        "H5_gap_curvature": h5,
        "H6_estimator_stability_armA": h6,
        "rate_contrast_B_minus_A_distinct": rate_contrast,
        "manipulation_check_round1": manip,
        "judge_noise": judge_noise,
        "verdicts": {"H4_crossing": v_h4, "H5_gap_curvature": v_h5, "H6_estimator_stability": v_h6},
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
    m = res["manipulation_check_round1"]
    h4 = res["H4_crossing_vs_max_estimator"]
    h5 = res["H5_gap_curvature"]
    h6 = res["H6_estimator_stability_armA"]
    rc = res["rate_contrast_B_minus_A_distinct"]
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
        "| Arm | Distinct /96 (mean ± SD) | Distinct /48 | Chao1-bc | ACE | Jackknife-2 | Incoherent (1st half / 2nd half) |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm, label in [("A", "A iid control"), ("B", "B archive feedback")]:
        d = a[arm]
        lines.append(
            f"| {label} | {d['distinct_mean']} ± {d['distinct_sd']} | {d['distinct_at_48_mean']} | "
            f"{d['chao1_bc_mean']} ± {d['chao1_bc_sd']} | {d['ace_mean']} ± {d['ace_sd']} | "
            f"{d['jackknife2_mean']} ± {d['jackknife2_sd']} | "
            f"{d['incoherent_total']} ({d['incoherent_first_half']} / {d['incoherent_second_half']}) |"
        )
    acc_a = a["A"]["accumulation_mean"]
    acc_b = a["B"]["accumulation_mean"]
    lines += [
        "",
        "**Accumulation curves** (mean distinct classes after each round):",
        "",
        "| Round | " + " | ".join(str(r) for r in ROUNDS) + " |",
        "|---" * (len(ROUNDS) + 1) + "|",
        "| A iid | " + " | ".join(str(acc_a[r]) for r in ROUNDS) + " |",
        "| B archive | " + " | ".join(str(acc_b[r]) for r in ROUNDS) + " |",
        "| gap B−A | " + " | ".join(str(h5["gap_full_curve"][r]) for r in ROUNDS) + " |",
        "",
        "**H4 (primary, crossing test):** B's observed distinct classes at 96 ideas minus the "
        "maximum of arm A's three estimators, paired per replicate.",
        "",
        "| Quantity | n | Mean diff | x SD(paired) | replicates crossed |",
        "|---|---|---|---|---|",
        f"| **vs max estimator (pre-registered bar)** | {h4['n_pairs']} | {h4['mean_diff']} | "
        f"{h4['x_sd_paired']} | {h4['n_replicates_crossed']}/{res['n_seeds_analyzed']} |",
    ]
    for k, label in [("chao1_bc", "vs Chao1-bc (secondary)"), ("ace", "vs ACE (secondary)"),
                     ("jackknife2", "vs jackknife-2 (secondary)")]:
        s_ = res["H4_secondary_per_estimator"][k]
        lines.append(
            f"| {label} | {s_['n_pairs']} | {s_['mean_diff']} | {s_['x_sd_paired']} | "
            f"{s_['n_replicates_crossed']}/{res['n_seeds_analyzed']} |"
        )
    lines += [
        "",
        f"**H5 (gap curvature):** mean B−A gap at rounds 4/8/12/16 = "
        + " / ".join(str(h5["gap_by_round"][r]) for r in GAP_ROUNDS)
        + f". Turnover (peak minus final): {h5['turnover']['mean_diff']} "
        f"({h5['turnover']['x_sd_paired']}x paired SD, 1x bar). "
        f"Growth (r16 − r8): {h5['growth_r16_minus_r8']['mean_diff']} "
        f"({h5['growth_r16_minus_r8']['x_sd_paired']}x paired SD, 2x bar).",
        "",
        "**H6 (estimator stability on arm A, self-audit):** each estimator at 48 vs 96 ideas.",
        "",
        "| Estimator | mean at 48 | mean at 96 | increase | x SD(paired) |",
        "|---|---|---|---|---|",
    ]
    for k, label in [("chao1_bc", "Chao1-bc"), ("ace", "ACE"), ("jackknife2", "Jackknife-2")]:
        e = h6[k]
        lines.append(
            f"| {label} | {e['mean_at_48']} | {e['mean_at_96']} | {e['mean_diff']} | {e['x_sd_paired']} |"
        )
    lines += [
        "",
        f"**Rate contrast carried forward** (B − A distinct classes at 96 ideas): "
        f"{rc['mean_diff']} at {rc['x_sd_paired']}x paired SD "
        f"({rc['x_sd_control']}x the control arm's SD), paired diffs {rc['paired_diffs']}.",
    ]
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
        f"- H4 (crossing, primary): {res['verdicts']['H4_crossing']}",
        f"- H5 (gap curvature): {res['verdicts']['H5_gap_curvature']}",
        f"- H6 (estimator stability): {res['verdicts']['H6_estimator_stability']}",
        "",
    ]
    readme.write_text(head + "\n".join(lines) + "\n")


def chart(res):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = res["arms"]
    h5 = res["H5_gap_curvature"]
    h6 = res["H6_estimator_stability_armA"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    colors = {"A": "#888888", "B": "#2b8cbe"}

    for arm, name in [("A", "A iid"), ("B", "B archive")]:
        acc = a[arm]["accumulation_mean"]
        axes[0].plot(ROUNDS, [acc[r] for r in ROUNDS], marker="o", ms=3,
                     color=colors[arm], label=name)
    ceil_ = max(a["A"]["chao1_bc_mean"], a["A"]["ace_mean"], a["A"]["jackknife2_mean"])
    axes[0].axhline(ceil_, ls="--", color="#c0392b",
                    label=f"A max estimator ({ceil_:.1f})")
    axes[0].set_xlabel("round (6 ideas each)")
    axes[0].set_ylabel("distinct classes")
    axes[0].set_title("Accumulation to 96 ideas vs A's ceiling")
    axes[0].legend(fontsize=8)

    axes[1].plot(ROUNDS, [h5["gap_full_curve"][r] for r in ROUNDS], marker="o", ms=3,
                 color="#2b8cbe")
    axes[1].axhline(0, color="#bbbbbb", lw=0.8)
    axes[1].set_xlabel("round")
    axes[1].set_ylabel("B − A distinct classes")
    axes[1].set_title("Gap curvature (H5)")

    labels = ["chao1", "ace", "jk2"]
    keys = ESTIMATORS
    w = 0.35
    xs = range(len(labels))
    axes[2].bar([x - w / 2 for x in xs], [h6[k]["mean_at_48"] for k in keys], width=w,
                color="#bbbbbb", label="A at 48 ideas")
    axes[2].bar([x + w / 2 for x in xs], [h6[k]["mean_at_96"] for k in keys], width=w,
                color="#555555", label="A at 96 ideas")
    axes[2].set_xticks(list(xs), labels)
    axes[2].set_title("Estimator stability, arm A (H6)")
    axes[2].legend(fontsize=8)

    fig.suptitle("Archive conditioning, budget extension to 16 rounds")
    fig.tight_layout()
    fig.savefig(HERE / "chart.png", dpi=120)


if __name__ == "__main__":
    analyze()
