"""Render every figure in the paper from the experiments' results.json files.

Usage:  python paper/make_figures.py            # writes paper/figures/*_{light,dark}.png
Each figure is rendered twice (light / dark theme tokens) on a transparent background so the
HTML paper can swap them with the viewer's theme. Colors follow a validated categorical palette
(fixed assignment per arm, never cycled); the reference arm is drawn in a neutral text tone.
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

THEMES = {
    "light": {"text": "#0b0b0b", "text2": "#52514e", "muted": "#8a8984", "grid": "#e6e5e1",
              "neutral": "#52514e",
              "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
              "magenta": "#e87ba4", "green": "#008300", "violet": "#4a3aa7", "red": "#e34948"},
    "dark": {"text": "#ffffff", "text2": "#c3c2b7", "muted": "#8f8e86", "grid": "#33332f",
             "neutral": "#c3c2b7",
             "blue": "#3987e5", "orange": "#d95926", "aqua": "#199e70", "yellow": "#c98500",
             "magenta": "#d55181", "green": "#008300", "violet": "#9085e9", "red": "#e66767"},
}
# fixed arm -> (color slot, linestyle, label)
ARMS = {
    "baseline":    ("neutral", "--", "no norm (baseline)"),
    "qknorm":      ("blue",    "-",  "QK-norm (both sides)"),
    "knorm_only":  ("aqua",    "-",  "key-only norm"),
    "knorm_dynk":  ("orange",  "-",  "FKN (key-only norm + magnitude channel)"),
    "k_emascale":  ("violet",  ":",  "key / running scale (alpha fixed = 1)"),
    "qnorm_only":  ("yellow",  ":",  "query-only norm"),
    "qknorm_dynq": ("magenta", "-.", "QK-norm + query channel (08-11 composite)"),
}


def style(theme):
    t = THEMES[theme]
    plt.rcParams.update({
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
        "savefig.transparent": True,
        "text.color": t["text"], "axes.labelcolor": t["text2"], "axes.edgecolor": t["grid"],
        "xtick.color": t["text2"], "ytick.color": t["text2"], "axes.titlecolor": t["text"],
        "grid.color": t["grid"], "grid.linestyle": "-", "grid.linewidth": 0.8,
        "axes.grid": True, "axes.axisbelow": True, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8.5,
        "font.size": 10, "font.family": "DejaVu Sans", "axes.titlesize": 10.5,
        "axes.titleweight": "bold", "lines.linewidth": 1.8, "lines.markersize": 5.5,
    })
    return t


def save(fig, name, theme):
    fig.savefig(OUT / f"{name}_{theme}.png", dpi=160, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def load(exp_id):
    p = EXP / exp_id / "results.json"
    return json.load(open(p)) if p.exists() else None


# ----------------------------------------------------------------------------- E1 figures
def fig_phase_diagram(theme, res, name="fig1_phase_diagram", arms=None,
                      title="Validation loss across the iso-parameter head split (600 steps, 3 paired seeds)"):
    t = style(theme)
    by = res["metrics"]["by_arm_hd"]
    hds = [int(h) for h in res["config"]["params"]["head_dims"]]
    arms = arms or [a for a in ARMS if a in by]
    x = np.arange(len(hds))
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for a in arms:
        col, ls, lab = ARMS[a]
        ys = [by[a][str(hd)]["val_bpc_mean"] if str(hd) in by[a] else np.nan for hd in hds]
        lw = 2.6 if a == "knorm_dynk" else 1.7
        ax.plot(x, ys, ls, color=t[col], lw=lw, marker="o", ms=6 if a == "knorm_dynk" else 5,
                mec="none" if theme == "dark" else "none", label=lab, zorder=3 if a == "knorm_dynk" else 2)
        for i, hd in enumerate(hds):
            if str(hd) in by[a]:
                ss = list(by[a][str(hd)]["val_bpc_per_seed"].values())
                ax.plot([i] * len(ss), ss, ".", color=t[col], ms=3.5, alpha=0.55, zorder=1)
    # selective direct label: FKN at its extremes
    if "knorm_dynk" in by:
        for i, hd in enumerate(hds):
            if str(hd) in by["knorm_dynk"] and hd in (hds[0], hds[-1]):
                v = by["knorm_dynk"][str(hd)]["val_bpc_mean"]
                ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points", xytext=(0, -14 if hd == hds[0] else 9),
                            ha="center", fontsize=8.5, color=t["text2"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"hd {hd}\n({128 // hd} heads)" for hd in hds])
    ax.set_ylabel("val bits per character  (lower is better)")
    ax.set_title(title)
    ax.legend(loc="upper right", ncol=1)
    save(fig, name, theme)


def fig_delta_bars(theme, res, name="fig2_delta_vs_baseline", arms=None,
                   title="Change in val bpc relative to the unnormalized baseline"):
    t = style(theme)
    per = res["metrics"]["per_head_dim"]
    hds = [int(h) for h in res["config"]["params"]["head_dims"] if str(h) in per]
    arms = arms or [a for a in ARMS if a != "baseline" and all(a in per[str(hd)]["arms"] for hd in hds)]
    x = np.arange(len(hds))
    w = 0.8 / len(arms)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for j, a in enumerate(arms):
        col, _, lab = ARMS[a]
        ys = [per[str(hd)]["arms"][a]["delta_vs_baseline"] for hd in hds]
        bars = ax.bar(x + (j - (len(arms) - 1) / 2) * w, ys, w * 0.9, color=t[col], label=lab,
                      edgecolor="none", zorder=2)
        if a == "knorm_dynk":
            for b, v in zip(bars, ys):
                ax.annotate(f"{v:+.3f}", (b.get_x() + b.get_width() / 2, v),
                            textcoords="offset points", xytext=(0, -11 if v < 0 else 3),
                            ha="center", fontsize=7.5, color=t["text2"])
    ax.axhline(0, color=t["text2"], lw=0.9, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"hd {hd}" for hd in hds])
    ax.set_ylabel("Δ val bpc vs baseline  (negative = better)")
    ax.set_title(title)
    ax.legend(loc="lower right", ncol=2)
    save(fig, name, theme)


def fig_alpha(theme, res, name="fig3_alpha_vs_headdim"):
    t = style(theme)
    P2 = res["metrics"]["P2_alpha_vs_head_dim"]
    hds = [int(h) for h in res["config"]["params"]["head_dims"]]
    x = np.arange(len(hds))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ak = {int(k): v for k, v in P2["k_alpha_knorm_dynk"].items() if v is not None}
    aq = {int(k): v for k, v in P2["q_alpha_qknorm_dynq"].items() if v is not None}
    if ak:
        ax.plot([hds.index(h) for h in ak], [ak[h] for h in ak], "-o", color=t["orange"], lw=2.4,
                label="key exponent α (FKN)")
    if aq:
        ax.plot([hds.index(h) for h in aq], [aq[h] for h in aq], "-.s", color=t["magenta"], lw=1.6,
                label="query exponent α (QK-norm + query channel)")
    ax.axhline(1.0, color=t["muted"], lw=0.9, ls="--")
    ax.text(len(hds) - 0.6, 1.005, "α = 1: magnitude fully restored", ha="right", va="bottom",
            fontsize=8, color=t["muted"])
    ax.axhline(0.0, color=t["muted"], lw=0.9, ls="--")
    ax.text(len(hds) - 0.6, 0.02, "α = 0: pure per-token norm", ha="right", va="bottom",
            fontsize=8, color=t["muted"])
    ax.set_ylim(-0.05, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"hd {hd}" for hd in hds])
    ax.set_ylabel("learned exponent α (mean over heads, layers, seeds)")
    ax.set_title("How much key magnitude the model chooses to keep")
    ax.legend(loc="lower left")
    save(fig, name, theme)


def fig_cliff_decomp(theme, name="fig4_cliff_decomposition"):
    """2026-08-31: six key-side arms at hd=4 - value vs gradient."""
    res = load("2026-08-31_hd4-kside-cliff-mechanism")
    if res is None:
        return
    t = style(theme)
    by = res["metrics"]["by_arm"]
    order = ["baseline", "kgain_only", "knorm_only", "knorm_nogain", "knorm_magrestore", "knorm_dynk"]
    labels = {"baseline": "no norm", "kgain_only": "gain only\n(no norm)", "knorm_only": "key norm\n+ gain",
              "knorm_nogain": "key norm\n(gain frozen)", "knorm_magrestore": "key norm ×\nmagnitude value\n(grad severed)",
              "knorm_dynk": "FKN: key norm ×\nmagnitude channel\n(grad open)"}
    cols = {"baseline": "neutral", "kgain_only": "muted", "knorm_only": "aqua", "knorm_nogain": "aqua",
            "knorm_magrestore": "blue", "knorm_dynk": "orange"}
    base = by["baseline"]["val_bpc_mean"]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    x = np.arange(len(order))
    for i, a in enumerate(order):
        v = by[a]["val_bpc_mean"] - base
        ax.bar(i, v, 0.58, color=t[cols[a]], edgecolor="none", zorder=2)
        ss = [s - base for s in by[a]["val_bpc_per_seed"].values()]
        ax.plot([i] * len(ss), ss, ".", color=t["text"], ms=4, alpha=0.6, zorder=3)
        ax.annotate(f"{v:+.3f}", (i, v), textcoords="offset points", xytext=(0, 4 if v >= 0 else -12),
                    ha="center", fontsize=8.5, color=t["text2"])
    ax.axhline(0, color=t["text2"], lw=0.9, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[a] for a in order], fontsize=8)
    ax.set_ylabel("Δ val bpc vs no norm, hd = 4 (32 heads)")
    ax.set_title("The tiny-head cliff is a severed magnitude gradient, not a lost value (2026-08-31)")
    save(fig, name, theme)


def fig_timeline(theme, name="fig5_thread_timeline"):
    """Best hd=4 configuration found on each night of the thread."""
    t = style(theme)
    nights = [
        ("07-26", "no norm", 3.09291, "neutral"),
        ("07-30", "QK-norm", 3.23590, "blue"),
        ("07-31", "+ static τ", 3.21780, "blue"),
        ("08-02", "+ per-token τ (detached)", 3.15076, "magenta"),
        ("08-06", "+ per-token τ (grad open)", 3.09340, "magenta"),
        ("08-30", "key-only norm", 3.21778, "aqua"),
        ("08-31", "FKN", 3.02007, "orange"),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    xs = np.arange(len(nights))
    base = nights[0][2]
    for i, (d, lab, v, col) in enumerate(nights):
        ax.bar(i, v - base, 0.6, color=t[col], edgecolor="none", zorder=2)
        ax.annotate(f"{v:.3f}", (i, v - base), textcoords="offset points",
                    xytext=(0, 4 if v >= base else -12), ha="center", fontsize=8.5, color=t["text2"])
    ax.axhline(0, color=t["text2"], lw=0.9, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{d}\n{lab}" for d, lab, _, _ in nights], fontsize=8)
    ax.set_ylabel("Δ val bpc vs no norm at hd = 4")
    ax.set_title("Nine nights at the tiny-head cliff (hd = 4, d_model = 128)")
    save(fig, name, theme)


# ----------------------------------------------------------------------------- E2 / E3
def fig_grouped_by_hd(theme, res, name, title, ylabel="val bits per character", arms=None):
    t = style(theme)
    by = res["metrics"]["by_arm_hd"]
    hds = [int(h) for h in res["config"]["params"]["head_dims"]]
    arms = arms or [a for a in ARMS if a in by]
    x = np.arange(len(hds))
    w = 0.8 / len(arms)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    lo, hi = 1e9, -1e9
    for j, a in enumerate(arms):
        col, _, lab = ARMS[a]
        ys = [by[a][str(hd)]["val_bpc_mean"] if str(hd) in by[a] else np.nan for hd in hds]
        xs = x + (j - (len(arms) - 1) / 2) * w
        ax.bar(xs, ys, w * 0.9, color=t[col], label=lab, edgecolor="none", zorder=2)
        for xi, hd in zip(xs, hds):
            if str(hd) in by[a]:
                ss = list(by[a][str(hd)]["val_bpc_per_seed"].values())
                ax.plot([xi] * len(ss), ss, ".", color=t["text"], ms=3.5, alpha=0.6, zorder=3)
                lo, hi = min(lo, min(ss)), max(hi, max(ss))
    ax.set_ylim(lo - 0.02, hi + 0.03)
    ax.set_xticks(x)
    ax.set_xticklabels([f"hd {hd} ({128 // hd} heads)" for hd in hds])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right", ncol=2)
    save(fig, name, theme)


# ----------------------------------------------------------------------------- MQAR
def fig_mqar(theme, name="fig8_mqar_recipe"):
    r28 = load("2026-07-28_mqar-min-selectivity")
    r91 = load("2026-09-01_mqar-escape-noise-vs-batch")
    if r28 is None or r91 is None:
        return
    t = style(theme)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    # left: gate-rank cliff at N=8
    m = r28["metrics"]["mean_acc"]
    order = ["none", "static", "scalar", "rank1", "rank4", "dense"]
    labels = {"none": "no gate", "static": "static", "scalar": "scalar", "rank1": "rank-1", "rank4": "rank-4", "dense": "dense\nper-channel"}
    ys = [m[f"{a}_N8"] for a in order]
    per_seed = {}
    for run in r28["metrics"]["runs"]:
        if run["num_pairs"] == 8:
            per_seed.setdefault(run["arm"], []).append(run["acc"])
    for i, a in enumerate(order):
        ax1.bar(i, ys[i], 0.6, color=t["orange"] if a == "dense" else t["blue"], edgecolor="none", zorder=2)
        ss = per_seed.get(a, [])
        ax1.plot([i] * len(ss), ss, ".", color=t["text"], ms=4, alpha=0.6, zorder=3)
    ax1.axhline(0.175, color=t["muted"], lw=0.9, ls="--")
    ax1.text(0.05, 0.19, "guessing plateau", fontsize=8, color=t["muted"])
    ax1.set_xticks(range(len(order)))
    ax1.set_xticklabels([labels[a] for a in order], fontsize=8.5)
    ax1.set_ylabel("MQAR accuracy at 2000 steps (8 key-value pairs)")
    ax1.set_title("Only a dense per-channel gate escapes (07-28)")
    ax1.set_ylim(0, 1.0)
    # right: escape step vs noise scale lr/B
    es = r91["metrics"]["escape_steps_N8"]
    cfg = {"b16_1x": (16, 1), "b64_1x": (64, 1), "b16_4x": (16, 4), "b64_4x": (64, 4), "b256_4x": (256, 4)}
    for arm, (B, m_) in cfg.items():
        steps = [s for s in es[arm] if s is not None]
        noise = m_ / B
        colr = t["orange"] if m_ == 4 else t["blue"]
        ax2.plot([noise] * len(steps), steps, "o", color=colr, ms=6, alpha=0.85, zorder=3)
        ax2.annotate(f"B={B}, lr×{m_}" + (" (1 seed censored)" if len(steps) < 3 else ""),
                     (noise, max(steps)), textcoords="offset points", xytext=(6, 4), fontsize=7.5, color=t["text2"])
    ax2.set_xscale("log")
    ax2.set_xlabel("gradient-noise scale  lr multiplier / batch size")
    ax2.set_ylabel("escape step (first eval with accuracy ≥ 0.5)")
    ax2.set_title("More gradient noise, later escape (09-01)")
    from matplotlib.lines import Line2D
    ax2.legend(handles=[Line2D([], [], marker="o", ls="", color=t["blue"], label="lr × 1"),
                        Line2D([], [], marker="o", ls="", color=t["orange"], label="lr × 4 (all groups)")],
               loc="upper left")
    fig.tight_layout(w_pad=3)
    save(fig, name, theme)


# ----------------------------------------------------------------------------- looped depth
def fig_loop(theme, name="fig9_loop_test_time_compute"):
    r = load("2026-07-25_loop-test-time-compute")
    if r is None:
        return
    t = style(theme)
    fr = r["metrics"]["frontier_acc_by_arm"]
    ks = [int(k) for k in fr["tied_randK_inj"].keys()]
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    series = [("tied_fixK3_inj", "blue", "-", "weight-tied loop, trained at fixed K=3"),
              ("tied_randK_inj", "orange", "-", "weight-tied loop, trained with K ~ U{1..3}"),
              ("untied_d3", "neutral", "--", "untied 3-layer stack (3× params)")]
    for arm, col, ls, lab in series:
        ys = [fr[arm][str(k)] for k in ks]
        ax.plot(ks, ys, ls, color=t[col], marker="o", lw=2.4 if arm == "tied_randK_inj" else 1.7, label=lab)
    ax.axvline(3, color=t["muted"], lw=0.9, ls="--")
    ax.text(3.08, 0.52, "trained depth", fontsize=8, color=t["muted"])
    ax.axhline(0.5, color=t["muted"], lw=0.9, ls="--")
    ax.set_xlabel("test-time loop count K")
    ax.set_ylabel("frontier accuracy (hardest solvable prefix-parity length)")
    ax.set_title("Test-time depth extrapolation needs a stochastic-depth schedule (07-25)")
    ax.set_ylim(0.4, 1.03)
    ax.legend(loc="lower left")
    save(fig, name, theme)


# ----------------------------------------------------------------------------- main
def main():
    e1 = load("2026-09-01_knorm-dynk-head-sweep")
    e2 = load("2026-09-01_knorm-dynk-ptb-transfer")
    e3 = load("2026-09-01_knorm-dynk-longer-training")
    core = ["baseline", "qknorm", "knorm_only", "knorm_dynk", "k_emascale"]
    for theme in ("light", "dark"):
        if e1:
            fig_phase_diagram(theme, e1, arms=core)
            fig_phase_diagram(theme, e1, name="figS1_phase_diagram_all_arms",
                              title="All seven arms across the head split")
            fig_delta_bars(theme, e1, arms=[a for a in core if a != "baseline"])
            fig_delta_bars(theme, e1, name="figS2_delta_all_arms", title="Δ vs baseline, all arms")
            fig_alpha(theme, e1)
        fig_cliff_decomp(theme)
        fig_timeline(theme)
        if e2:
            fig_grouped_by_hd(theme, e2, "fig6_ptb_transfer",
                              "Second corpus: character-level Penn Treebank (600 steps, 3 paired seeds)",
                              ylabel="val bits per character (char-PTB)")
        if e3:
            fig_grouped_by_hd(theme, e3, "fig7_longer_training",
                              "3× longer training (1800 steps, 3 paired seeds)")
        fig_mqar(theme)
        fig_loop(theme)
    print("figures written to", OUT, sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
