"""When does pair-merging in a toy superposition model become too lossy?

Follow-up to 2026-07-23_superposition-correlation-phase, where correlated feature pairs
always MERGED onto one shared direction (within-pair |cos| -> 1.0) at EQUAL importance and
independent values. Here we stress the merge along two axes and look for the breakpoint at
which the pair abandons the shared direction for a locally orthogonal basis.

Model (identical family to the sibling run):
    x_hat = ReLU(W^T W x + b),  W: (m=4, n=16), b: (16,)   ~ 80 params
    loss  = mean_i  I_i * (x_i - x_hat_i)^2                (I = per-feature importance)

Data: n=16 features in 8 pairs. Within a pair,
  * INDICATOR correlation rho, exactly as in the sibling: per sample per pair, with prob rho
    both members share ONE Bernoulli(p) on/off coin, else they draw independent coins.
  * VALUE correlation c in [-1, 1], by the same mixture trick: with prob |c| the second
    member's value is copied (c > 0) or mirrored, v2 = 1 - v1 (c < 0), from the first
    member's Uniform[0,1] value; otherwise it is an independent Uniform[0,1] draw. Both
    marginals stay Uniform[0,1] and the Pearson correlation of the ACTIVE values is exactly c.
  * IMPORTANCE ratio r: importances within a pair are (2/(1+r), 2r/(1+r)) so the pair's TOTAL
    importance is 2 regardless of r -- r moves the split, not the pair's share of the loss.

Three sweeps (cells are cached by (r, rho, c, seed), so overlaps are trained once):
  A  importance ratio r in {1..256} x rho in {0, 0.75, 1}   (rho=0 = unpaired control)
  B  value correlation c in {+1..-1} x rho in {0.75, 1}
  C  joint corner: r in {1,4,16} x c in {+1, 0, -1} at rho=1

Per cell we classify each of the 8 pairs as MERGED (both members represented, |cos| >= 0.7),
LOCAL-ORTHOGONAL (both represented, |cos| <= 0.3), AMBIGUOUS (both represented, in between),
or DROPPED (at least one member has ||W_i|| < 0.5), and we measure per-feature held-out
reconstruction loss normalised by the variance of that feature (1.0 = no better than the best
constant predictor, 0.0 = perfect).

Deterministic, CPU-only, single-threaded. Writes results.json and chart.png.
Usage:  python run.py
"""
import json, os, random, subprocess, sys, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Mean of empty slice")

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent


def set_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE).decode().strip()
    except Exception:
        return "nogit"


def load_config():
    import yaml
    with open(HERE / "experiment.yaml") as f:
        return yaml.safe_load(f)


def env_info():
    info = {"python": sys.version.split()[0]}
    for mod in ("numpy", "torch"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:
            pass
    return info


def sanitize(o):
    """NaN/inf -> None so results.json is strict-valid JSON (NaN means 'no such pair exists')."""
    import math
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, dict):
        return {k: sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    return o


# ----------------------------------------------------------------------------- data
def sample_batch(gen, batch, n, p, rho, vcorr):
    """Sparse paired features: indicator correlation exactly rho, value correlation exactly vcorr."""
    import torch
    n_pairs = n // 2
    # --- indicators: shared-coin mixture -> within-pair indicator correlation = rho
    share = torch.rand(batch, n_pairs, generator=gen) < rho
    shared_coin = torch.rand(batch, n_pairs, generator=gen) < p
    own = torch.rand(batch, n_pairs, 2, generator=gen) < p
    ind = torch.where(share.unsqueeze(-1), shared_coin.unsqueeze(-1).expand(-1, -1, 2), own).float()
    # --- values: copy / mirror mixture -> within-pair value correlation = vcorr
    v0 = torch.rand(batch, n_pairs, generator=gen)
    v_indep = torch.rand(batch, n_pairs, generator=gen)
    link = torch.rand(batch, n_pairs, generator=gen) < abs(vcorr)
    v_dep = v0 if vcorr >= 0 else (1.0 - v0)
    v1 = torch.where(link, v_dep, v_indep)
    vals = torch.stack([v0, v1], dim=-1)
    return (ind * vals).reshape(batch, n)


def importances(n, r):
    """Within each pair: (low, high) = (2/(1+r), 2r/(1+r)); pair total is 2 for every r."""
    import torch
    lo, hi = 2.0 / (1.0 + r), 2.0 * r / (1.0 + r)
    imp = torch.empty(n)
    imp[0::2] = lo
    imp[1::2] = hi
    return imp


def cell_seed(r, rho, vcorr, seed):
    key = (int(r), round(float(rho), 4), round(float(vcorr), 4), int(seed))
    h = 0
    for v in key:
        h = (h * 1000003 + int(round(v * 10000))) % (2 ** 31 - 1)
    return h


# ----------------------------------------------------------------------------- train
def train_cell(seed, n, m, p, rho, vcorr, r, steps, batch, lr):
    import torch
    gen = torch.Generator().manual_seed(cell_seed(r, rho, vcorr, seed))
    W = torch.empty(m, n)
    torch.nn.init.xavier_normal_(W, generator=gen)
    W = W.requires_grad_(True)
    b = torch.zeros(n, requires_grad=True)
    imp = importances(n, r)
    opt = torch.optim.Adam([W, b], lr=lr)
    final = float("nan")
    for _ in range(steps):
        x = sample_batch(gen, batch, n, p, rho, vcorr)
        x_hat = torch.relu(x @ W.T @ W + b)
        loss = (imp * (x - x_hat) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        final = loss.item()
    return W.detach(), b.detach(), imp, final


# ----------------------------------------------------------------------------- analysis
def analyze(W, b, imp, n, m, p, rho, vcorr, P):
    import torch
    thresh = P["represented_norm_threshold"]
    merge_t, ortho_t = P["merge_cos_threshold"], P["ortho_cos_threshold"]
    n_pairs = n // 2

    norms = W.norm(dim=0)
    rep = norms >= thresh
    Wn = W / norms.clamp_min(1e-8)
    C = Wn.T @ Wn  # signed cosine matrix

    within_abs, within_signed, cross_abs = [], [], []
    n_merged = n_ortho = n_ambig = n_dropped = 0
    n_dropped_lo = n_dropped_hi = 0
    for k in range(n_pairs):
        i, j = 2 * k, 2 * k + 1  # i = LOW importance, j = HIGH importance
        if rep[i] and rep[j]:
            c = float(C[i, j])
            within_abs.append(abs(c))
            within_signed.append(c)
            if abs(c) >= merge_t:
                n_merged += 1
            elif abs(c) <= ortho_t:
                n_ortho += 1
            else:
                n_ambig += 1
        else:
            n_dropped += 1
            if not rep[i]:
                n_dropped_lo += 1
            if not rep[j]:
                n_dropped_hi += 1
    for i in range(n):
        for j in range(i + 1, n):
            if rep[i] and rep[j] and not (j == i + 1 and i % 2 == 0):
                cross_abs.append(abs(float(C[i, j])))

    mean = lambda v: float(sum(v) / len(v)) if v else float("nan")

    # --- held-out per-feature reconstruction loss, normalised by feature variance
    egen = torch.Generator().manual_seed(cell_seed(0, rho, vcorr, 777))
    xe = sample_batch(egen, P["eval_batch"], n, p, rho, vcorr)
    with torch.no_grad():
        xe_hat = torch.relu(xe @ W.T @ W + b)
    mse_f = ((xe - xe_hat) ** 2).mean(dim=0)
    var_f = xe.var(dim=0, unbiased=False).clamp_min(1e-12)
    nmse_f = (mse_f / var_f)
    # sanity: does the generator produce the correlations it claims?
    act = (xe[:, 0::2] > 0) & (xe[:, 1::2] > 0)
    if int(act.sum()) > 10:
        va, vb = xe[:, 0::2][act], xe[:, 1::2][act]
        emp_vcorr = float(((va - va.mean()) * (vb - vb.mean())).mean() / (va.std(unbiased=False) * vb.std(unbiased=False) + 1e-12))
    else:
        emp_vcorr = float("nan")
    ia = (xe[:, 0::2] > 0).float().reshape(-1)
    ib = (xe[:, 1::2] > 0).float().reshape(-1)
    emp_rho = float(((ia - ia.mean()) * (ib - ib.mean())).mean() / (ia.std(unbiased=False) * ib.std(unbiased=False) + 1e-12))

    return {
        "n_represented": int(rep.sum()),
        "frac_represented": float(rep.float().mean()),
        "features_per_dim": int(rep.sum()) / m,
        "within_pair_abs_cos": mean(within_abs),
        "within_pair_signed_cos": mean(within_signed),
        "cross_pair_abs_cos": mean(cross_abs),
        "frac_pairs_both_repr": (n_pairs - n_dropped) / n_pairs,
        "frac_pairs_merged": n_merged / n_pairs,
        "frac_pairs_ortho": n_ortho / n_pairs,
        "frac_pairs_ambiguous": n_ambig / n_pairs,
        "frac_pairs_dropped": n_dropped / n_pairs,
        "frac_lo_dropped": n_dropped_lo / n_pairs,
        "frac_hi_dropped": n_dropped_hi / n_pairs,
        "nmse_lo": float(nmse_f[0::2].mean()),
        "nmse_hi": float(nmse_f[1::2].mean()),
        "nmse_all": float(nmse_f.mean()),
        "norm_lo": float(norms[0::2].mean()),
        "norm_hi": float(norms[1::2].mean()),
        "emp_value_corr": emp_vcorr,
        "emp_indicator_corr": emp_rho,
    }


# ----------------------------------------------------------------------------- driver
def main():
    import numpy as np
    cfg = load_config()
    P = cfg["params"]
    seed = int(cfg.get("seed", 0))
    set_seeds(seed)
    t0 = time.time()

    n, m, p = P["n_features"], P["m_hidden"], P["density"]
    seeds = P["seeds"]
    cache = {}

    def cell(r, rho, vcorr):
        """Mean-over-seeds analysis for one (r, rho, vcorr) cell; trained once, cached."""
        key = (int(r), float(rho), float(vcorr))
        if key in cache:
            return cache[key]
        runs = []
        for s in seeds:
            W, b, imp, tl = train_cell(s, n, m, p, rho, vcorr, r, P["steps"], P["batch"], P["lr"])
            a = analyze(W, b, imp, n, m, p, rho, vcorr, P)
            a["train_loss"] = tl
            runs.append(a)
        agg = {k: float(np.nanmean([r_[k] for r_ in runs])) for k in runs[0]}
        agg["_runs"] = runs
        cache[key] = agg
        print(f"  r={r:<4} rho={rho:<5} c={vcorr:<5} |cos|={agg['within_pair_abs_cos']:.3f} "
              f"signed={agg['within_pair_signed_cos']:+.3f} both_repr={agg['frac_pairs_both_repr']:.2f} "
              f"merged={agg['frac_pairs_merged']:.2f} ortho={agg['frac_pairs_ortho']:.2f} "
              f"nmse lo/hi={agg['nmse_lo']:.3f}/{agg['nmse_hi']:.3f}", flush=True)
        return agg

    # ---------------- sweep A: importance ratio ----------------
    A_r, A_rho = P["sweep_a_ratios"], P["sweep_a_rhos"]
    print("== sweep A: importance ratio (value corr = 0) ==", flush=True)
    A = {rho: [cell(r, rho, 0.0) for r in A_r] for rho in A_rho}

    # ---------------- sweep B: value correlation ----------------
    B_c, B_rho = P["sweep_b_vcorrs"], P["sweep_b_rhos"]
    print("== sweep B: value correlation (importance ratio = 1) ==", flush=True)
    B = {rho: [cell(1, rho, c) for c in B_c] for rho in B_rho}

    # ---------------- sweep C: joint corner ----------------
    C_r, C_c, C_rho = P["sweep_c_ratios"], P["sweep_c_vcorrs"], P["sweep_c_rho"]
    print("== sweep C: joint (rho = %g) ==" % C_rho, flush=True)
    C = [[cell(r, C_rho, c) for c in C_c] for r in C_r]

    get = lambda cells, k: [c[k] for c in cells]

    # ---------------- breakpoint detection ----------------
    def breakpoint_r(cells, rs, key="within_pair_abs_cos", t=P["merge_cos_threshold"]):
        for r_, c_ in zip(rs, cells):
            v = c_[key]
            if not np.isnan(v) and v < t:
                return r_
        return None

    def breakpoint_x(cells, xs, key="within_pair_abs_cos", t=P["merge_cos_threshold"]):
        for x_, c_ in zip(xs, cells):
            v = c_[key]
            if not np.isnan(v) and v < t:
                return x_
        return None

    # Did ANY cell in the whole experiment put a pair into a locally-orthogonal basis?
    max_frac_ortho = max(c["frac_pairs_ortho"] for c in cache.values())
    max_frac_ortho_corr = max(c["frac_pairs_ortho"] for k, c in cache.items() if k[1] > 0)
    min_abs_cos_alive = min([c["within_pair_abs_cos"] for c in cache.values()
                             if not np.isnan(c["within_pair_abs_cos"])])

    headline = {
        "merge_cos_threshold": P["merge_cos_threshold"],
        "ortho_cos_threshold": P["ortho_cos_threshold"],
        "any_pair_locally_orthogonal": bool(max_frac_ortho > 0),
        "max_frac_pairs_ortho_over_all_cells": round(float(max_frac_ortho), 4),
        "max_frac_pairs_ortho_over_correlated_cells": round(float(max_frac_ortho_corr), 4),
        "any_correlated_pair_locally_orthogonal": bool(max_frac_ortho_corr > 0),
        "min_within_abs_cos_over_surviving_pairs": round(float(min_abs_cos_alive), 4),
        "breakpoint_importance_ratio": {str(rho): breakpoint_r(A[rho], A_r) for rho in A_rho},
        "breakpoint_value_corr": {str(rho): breakpoint_x(B[rho], B_c) for rho in B_rho},
        "A_norm_lo": {str(rho): [round(v, 4) for v in get(A[rho], "norm_lo")] for rho in A_rho},
        "A_norm_hi": {str(rho): [round(v, 4) for v in get(A[rho], "norm_hi")] for rho in A_rho},
        "A_features_per_dim": {str(rho): [round(v, 3) for v in get(A[rho], "features_per_dim")] for rho in A_rho},
        "B_frac_pairs_merged": {str(rho): [round(v, 3) for v in get(B[rho], "frac_pairs_merged")] for rho in B_rho},
        "B_frac_pairs_ambiguous": {str(rho): [round(v, 3) for v in get(B[rho], "frac_pairs_ambiguous")] for rho in B_rho},
        "B_frac_pairs_ortho": {str(rho): [round(v, 3) for v in get(B[rho], "frac_pairs_ortho")] for rho in B_rho},
        "B_cross_pair_abs_cos": {str(rho): [round(v, 4) for v in get(B[rho], "cross_pair_abs_cos")] for rho in B_rho},
        "C_frac_pairs_both_repr": [[round(C[i][j]["frac_pairs_both_repr"], 3) for j in range(len(C_c))]
                                   for i in range(len(C_r))],
        "A_within_abs_cos": {str(rho): [round(v, 4) for v in get(A[rho], "within_pair_abs_cos")] for rho in A_rho},
        "A_frac_pairs_both_repr": {str(rho): [round(v, 3) for v in get(A[rho], "frac_pairs_both_repr")] for rho in A_rho},
        "A_frac_pairs_merged": {str(rho): [round(v, 3) for v in get(A[rho], "frac_pairs_merged")] for rho in A_rho},
        "A_frac_lo_dropped": {str(rho): [round(v, 3) for v in get(A[rho], "frac_lo_dropped")] for rho in A_rho},
        "A_nmse_lo": {str(rho): [round(v, 4) for v in get(A[rho], "nmse_lo")] for rho in A_rho},
        "A_nmse_hi": {str(rho): [round(v, 4) for v in get(A[rho], "nmse_hi")] for rho in A_rho},
        "B_within_abs_cos": {str(rho): [round(v, 4) for v in get(B[rho], "within_pair_abs_cos")] for rho in B_rho},
        "B_within_signed_cos": {str(rho): [round(v, 4) for v in get(B[rho], "within_pair_signed_cos")] for rho in B_rho},
        "B_frac_pairs_both_repr": {str(rho): [round(v, 3) for v in get(B[rho], "frac_pairs_both_repr")] for rho in B_rho},
        "B_nmse_all": {str(rho): [round(v, 4) for v in get(B[rho], "nmse_all")] for rho in B_rho},
        "C_within_abs_cos": [[round(C[i][j]["within_pair_abs_cos"], 4) for j in range(len(C_c))] for i in range(len(C_r))],
        "C_within_signed_cos": [[round(C[i][j]["within_pair_signed_cos"], 4) for j in range(len(C_c))] for i in range(len(C_r))],
        "axes": {"A_ratios": A_r, "A_rhos": A_rho, "B_vcorrs": B_c, "B_rhos": B_rho,
                 "C_ratios": C_r, "C_vcorrs": C_c, "C_rho": C_rho},
    }

    # ---------------- chart ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18.5, 9.6))
    fig.suptitle("Where does a merged feature pair break? Anthropic-style toy model ReLU(WᵀWx+b), "
                 f"n={n} features in 8 pairs → m={m} dims, density p={p}, {len(seeds)} seeds, "
                 f"{len(cache)} cells", fontsize=13)
    cols = {0.0: "#888888", 0.75: "#4477aa", 1.0: "#bb5566"}
    mt, ot = P["merge_cos_threshold"], P["ortho_cos_threshold"]

    def logr(ax):
        ax.set_xscale("log", base=2)
        ax.set_xticks(A_r, [str(r_) for r_ in A_r])
        ax.set_xlabel("within-pair importance ratio r  (log₂)")
        ax.grid(alpha=0.25, lw=0.5)

    # (0,0) sweep A: |cos| among surviving pairs + survival
    ax = axes[0, 0]
    for rho in A_rho:
        ax.plot(A_r, get(A[rho], "within_pair_abs_cos"), "-o", ms=6, lw=2.2,
                color=cols[rho], label=f"within |cos|, ρ={rho:g}")
        ax.plot(A_r, get(A[rho], "frac_pairs_both_repr"), "--s", ms=4, lw=1.3, alpha=0.6,
                color=cols[rho], label=f"frac pairs alive, ρ={rho:g}")
    ax.axhspan(0, ot, color="#2ca02c", alpha=0.10)
    ax.axhline(mt, color="k", ls=":", lw=1.2)
    ax.text(A_r[0], mt + 0.02, f"merge threshold {mt}", fontsize=8)
    ax.text(A_r[0], ot / 2, "locally-orthogonal zone |cos| ≤ 0.3\n(no CORRELATED pair ever enters it)",
            fontsize=8, color="#1a6b1a")
    ax.set_ylabel("within-pair |cos|  /  fraction of pairs")
    ax.set_ylim(-0.03, 1.10)
    ax.set_title("A1. unequal importance: for ρ>0, |cos| never falls —\nthe pair vanishes instead (line stops = no pair survives)",
                 fontsize=10)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="lower left")
    logr(ax)

    # (0,1) sweep A: the actual mechanism = amplitude decay along the SAME direction
    ax = axes[0, 1]
    for rho in A_rho:
        ax.plot(A_r, get(A[rho], "norm_lo"), "-o", ms=6, lw=2.2, color=cols[rho],
                label=f"weak member ‖W‖, ρ={rho:g}")
        ax.plot(A_r, get(A[rho], "norm_hi"), "--^", ms=4, lw=1.4, alpha=0.65, color=cols[rho],
                label=f"strong member ‖W‖, ρ={rho:g}")
    ax.axhline(P["represented_norm_threshold"], color="k", ls=":", lw=1.2)
    ax.text(A_r[0], P["represented_norm_threshold"] + 0.03, "‘represented’ threshold", fontsize=8)
    ax.set_ylabel("mean column norm ‖W_i‖")
    ax.set_title("A2. the mechanism: the weak member SHRINKS along the\nshared direction, it does not rotate off it",
                 fontsize=10)
    ax.legend(fontsize=6.5, frameon=False, ncol=1, loc="center left", bbox_to_anchor=(0.30, 0.62))
    logr(ax)

    # (0,2) sweep A: per-feature held-out loss = the correlation subsidy
    ax = axes[0, 2]
    for rho in A_rho:
        ax.plot(A_r, get(A[rho], "nmse_lo"), "-o", ms=6, lw=2.2, color=cols[rho],
                label=f"weak member, ρ={rho:g}")
        ax.plot(A_r, get(A[rho], "nmse_hi"), "--^", ms=4, lw=1.4, alpha=0.65, color=cols[rho],
                label=f"strong member, ρ={rho:g}")
    ax.axhline(1.0, color="k", ls=":", lw=1.2)
    ax.text(A_r[0], 1.02, "= best constant predictor (feature carries nothing)", fontsize=8)
    ax.set_ylabel("held-out MSE / feature variance")
    ax.set_ylim(-0.03, 1.15)
    ax.set_title("A3. merging is a SUBSIDY: at matched importance the weak\nmember survives only if it co-occurs (gap ρ=0 vs ρ=1)",
                 fontsize=10)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="lower right")
    logr(ax)

    # (1,0) sweep B: signed and |cos| vs value correlation
    ax = axes[1, 0]
    for rho in B_rho:
        ax.plot(B_c, get(B[rho], "within_pair_abs_cos"), "-o", ms=6, lw=2.2, color=cols[rho],
                label=f"|cos|, ρ={rho:g}")
        ax.plot(B_c, get(B[rho], "within_pair_signed_cos"), "--d", ms=4, lw=1.4, alpha=0.6,
                color=cols[rho], label=f"signed cos, ρ={rho:g}")
    ax.axhspan(-ot, ot, color="#2ca02c", alpha=0.10)
    ax.axhline(mt, color="k", ls=":", lw=1.2)
    ax.axhline(0.0, color="#aaaaaa", lw=0.8)
    bp = breakpoint_x(B[B_rho[0]], B_c)
    if bp is not None:
        ax.axvline(bp, color="#cc7700", lw=1.4, ls="-.")
        ax.text(bp, -0.55, f" breakpoint\n c={bp:g} (ρ={B_rho[0]:g})", fontsize=8, color="#aa5500")
    ax.invert_xaxis()
    ax.set_xlabel("within-pair VALUE correlation c   (+1 identical → −1 mirrored)")
    ax.set_ylabel("mean within-pair cosine")
    ax.set_ylim(-1.05, 1.10)
    ax.set_title("B1. anticorrelated VALUES do bend the merge —\nbut it stays positive, never antipodal, never orthogonal",
                 fontsize=10)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="lower left")
    ax.grid(alpha=0.25, lw=0.5)

    # (1,1) sweep B: pair classification + reconstruction cost
    ax = axes[1, 1]
    w = 0.13
    xs = np.arange(len(B_c), dtype=float)
    for t_, rho in enumerate(B_rho):
        off = (t_ - 0.5) * (3 * w + 0.03)
        for s_, (key, cc, lab) in enumerate([("frac_pairs_merged", "#bb5566", "merged"),
                                             ("frac_pairs_ambiguous", "#ddaa33", "ambiguous"),
                                             ("frac_pairs_ortho", "#2ca02c", "orthogonal")]):
            ax.bar(xs + off + (s_ - 1) * w, get(B[rho], key), width=w, color=cc,
                   alpha=0.55 if rho == B_rho[0] else 1.0, edgecolor="white", lw=0.5,
                   label=f"{lab}, ρ={rho:g}")
    ax2 = ax.twinx()
    for rho in B_rho:
        ax2.plot(xs, get(B[rho], "nmse_all"), "-o", ms=5, lw=2, color="#222255",
                 alpha=0.55 if rho == B_rho[0] else 1.0, label=f"held-out NMSE, ρ={rho:g}")
    ax2.set_ylabel("held-out MSE / variance (line)")
    ax2.set_ylim(0, 0.5)
    ax.set_xticks(xs, [f"{c:g}" for c in B_c])
    ax.set_xlabel("within-pair VALUE correlation c")
    ax.set_ylabel("fraction of pairs (bars)")
    ax.set_ylim(0, 1.15)
    ax.set_title("B2. pairs go merged → AMBIGUOUS, never orthogonal,\nwhile the reconstruction cost keeps climbing", fontsize=10)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.2, frameon=True, framealpha=0.85, edgecolor="none",
              ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02))

    # (1,2) sweep C heatmap: joint corner
    ax = axes[1, 2]
    M = np.array(headline["C_within_abs_cos"], dtype=float)
    S = np.array(headline["C_frac_pairs_both_repr"], dtype=float)
    cmap = matplotlib.cm.get_cmap("RdBu_r").copy()
    cmap.set_bad("#dcdcdc")
    im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0, origin="upper")
    ax.set_xticks(range(len(C_c)), [f"{c:g}" for c in C_c])
    ax.set_yticks(range(len(C_r)), [f"{r_:g}" for r_ in C_r])
    ax.set_xlabel("within-pair value correlation c")
    ax.set_ylabel("within-pair importance ratio r")
    ax.set_title(f"C. both stressors at once (ρ={C_rho:g}): within-pair |cos|\n(and fraction of pairs still alive)",
                 fontsize=10)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"|cos| {M[i, j]:.2f}\nalive {S[i, j]:.2f}", ha="center", va="center",
                        fontsize=8.5, color="white" if (M[i, j] > 0.75 or M[i, j] < 0.25) else "#111111")
            else:
                ax.text(j, i, "pair gone\nalive 0.00", ha="center", va="center", fontsize=8.5, color="#555555")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(HERE / "chart.png", dpi=150)

    metrics = {
        "headline": headline,
        "cells": [
            {"r": k[0], "rho": k[1], "value_corr": k[2],
             **{kk: (round(vv, 5) if isinstance(vv, float) else vv)
                for kk, vv in v.items() if kk != "_runs"}}
            for k, v in cache.items()
        ],
        "n_cells_trained": len(cache),
        "n_runs_trained": len(cache) * len(seeds),
    }

    results = {
        "id": cfg.get("id", "unknown"),
        "git_commit": git_sha(),
        "seed": seed,
        "duration_sec": round(time.time() - t0, 2),
        "metrics": metrics,
        "env": env_info(),
        "status": "done",
    }
    results = sanitize(results)
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2, allow_nan=False)
    print(json.dumps({k: results[k] for k in ("id", "duration_sec", "status")}, indent=2))
    print("headline:", json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()
