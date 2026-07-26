"""Modern Hopfield network as associative memory: capacity vs corruption, beta, and dimension.

Hand-rolled Ramsauer et al. (arXiv:2008.02217) update
    xi_new = X^T softmax(beta * X xi)
over a bank of stored bipolar patterns X (N x d). No training, pure numpy matmuls.
Classical Hebbian outer-product Hopfield (capacity ~0.14 d) is the reference line.

Deterministic, CPU-only, single-threaded. Writes results.json + chart.png.

Usage:  python run.py
"""
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

# single-threaded BLAS: this box has 2 cores shared with other agents.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402
import matplotlib   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent


# ----------------------------------------------------------------------------- infra
def set_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(seed)
    except Exception:
        pass


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
    for mod in ("numpy", "torch", "sklearn", "matplotlib"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:
            pass
    return info


# ----------------------------------------------------------------------------- core
def softmax(logits, axis=-1):
    z = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def _hopfield_chunk(X, Q, beta, max_iters, tol):
    S = Q.astype(np.float64, copy=True)
    S1 = None
    diag = {}
    steps = max_iters
    for t in range(max_iters):
        P = softmax(beta * (S @ X.T), axis=1)          # (q, N)
        S_new = P @ X                                  # (q, d)
        if t == 0:
            S1 = S_new.copy()
            diag["pr"] = 1.0 / np.sum(P * P, axis=1)
            diag["top1"] = P.max(axis=1)
            diag["argmax_idx"] = P.argmax(axis=1)
        del P
        if np.max(np.abs(S_new - S)) < tol:
            S = S_new
            steps = t + 1
            break
        S = S_new
    diag["iters_used"] = steps
    return S1, S, diag


def hopfield_modern(X, Q, beta, max_iters, tol, max_elems=4_000_000):
    """Ramsauer update xi <- X^T softmax(beta X xi).

    X: (N,d) stored patterns. Q: (q,d) query states. Queries are processed in
    chunks so the dense (chunk x N) softmax matrix stays small.
    Returns the state after 1 step, the converged state, and diagnostics of the
    FIRST step's softmax distribution.
    """
    N = X.shape[0]
    chunk = max(1, min(len(Q), int(max_elems // max(N, 1))))
    S1s, Ss, prs, tops, args, iters = [], [], [], [], [], 0
    for lo in range(0, len(Q), chunk):
        s1, s, d = _hopfield_chunk(X, Q[lo:lo + chunk], beta, max_iters, tol)
        S1s.append(s1)
        Ss.append(s)
        prs.append(d["pr"])
        tops.append(d["top1"])
        args.append(d["argmax_idx"])
        iters = max(iters, d["iters_used"])
    diag = {"participation_ratio": float(np.mean(np.concatenate(prs))),
            "top1_weight": float(np.mean(np.concatenate(tops))),
            "argmax_idx": np.concatenate(args),
            "iters_used": iters}
    return np.concatenate(S1s, 0), np.concatenate(Ss, 0), diag


def hopfield_classical(X, Q, sweeps, rng):
    """Classical Hebbian outer-product Hopfield, asynchronous sign updates."""
    N, d = X.shape
    W = (X.T @ X) / d
    np.fill_diagonal(W, 0.0)
    S = Q.astype(np.float64, copy=True)
    for _ in range(sweeps):
        S_prev = S.copy()
        for i in rng.permutation(d):
            h = S @ W[:, i]
            S[:, i] = np.where(h >= 0, 1.0, -1.0)
        if np.array_equal(S, S_prev):
            break
    return S


def bipolar_sign(A):
    return np.where(A >= 0, 1.0, -1.0)


def make_queries(X, n_queries, flip_frac, rng):
    """Pick target patterns (with replacement) and flip exactly round(f*d) components."""
    N, d = X.shape
    n_flip = int(round(flip_frac * d))
    targets = rng.integers(0, N, size=n_queries)
    Q = X[targets].copy()
    for q in range(n_queries):
        idx = rng.choice(d, size=n_flip, replace=False)
        Q[q, idx] *= -1.0
    return Q, targets, n_flip


def cosine(A, B):
    num = np.sum(A * B, axis=1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12
    return num / den


def score(state, X, targets):
    """Exact retrieval (sign of state equals the true pattern) + cosine overlap."""
    true = X[targets]
    hard = bipolar_sign(state)
    exact = float(np.mean(np.all(hard == true, axis=1)))
    cos = float(np.mean(cosine(state, true)))
    hamm = float(np.mean(np.mean(hard != true, axis=1)))
    return exact, cos, hamm


def capacity_from_curve(ns, rates, thr):
    """Empirical capacity.

    Primary ('capacity'): first-crossing -- the largest N such that EVERY swept
    N' <= N still has exact-retrieval rate >= thr. This is the conservative
    'how many can I store and still be safe' reading.
    Secondary ('capacity_last'): the largest N anywhere on the curve that clears
    thr, which is more generous when the curve is noisy near the threshold.
    'censored' means the largest N we swept still cleared the bar.
    """
    order = sorted(zip(ns, rates))
    cap_first = 0
    for n, r in order:
        if r >= thr:
            cap_first = n
        else:
            break
    ok = [n for n, r in order if r >= thr]
    cap_last = max(ok) if ok else 0
    nmax = max(ns)
    return {"capacity": cap_first, "capacity_last": cap_last,
            "censored": bool(ok and cap_last == nmax)}


def n_grid(kmin, kmax):
    return sorted({int(round(2 ** (k / 2.0))) for k in range(kmin, kmax + 1)})


# ----------------------------------------------------------------------------- sweeps
def separation(X, Q, targets, max_elems=4_000_000):
    """Mean gap between the query's overlap with its own pattern and with the best
    competitor, plus the fraction of queries whose nearest stored pattern is correct."""
    N = X.shape[0]
    if N < 2:
        return float("inf"), 1.0
    chunk = max(1, int(max_elems // N))
    gaps, corr = [], []
    for lo in range(0, len(Q), chunk):
        sl = slice(lo, min(lo + chunk, len(Q)))
        sims = Q[sl] @ X.T
        tg = targets[sl]
        rows_ = np.arange(sims.shape[0])
        own = sims[rows_, tg].copy()
        sims[rows_, tg] = -np.inf
        best_other = sims.max(axis=1)
        gaps.append(own - best_other)
        corr.append(own > best_other)
    return float(np.mean(np.concatenate(gaps))), float(np.mean(np.concatenate(corr)))


def sweep_modern(X_full, ns, betas, flips, n_queries, max_iters, tol, seed_base, label,
                 max_elems=4_000_000, extra_ns=None, extra_flip=None):
    """Nested design: X = X_full[:N], and queries for a given (flip, N) come from a
    fixed RNG stream so every beta sees byte-identical inputs.
    extra_ns/extra_flip add a deeper N range for one corruption level only."""
    rows = []
    for f in flips:
        n_list = list(ns)
        if extra_ns and extra_flip is not None and f == extra_flip:
            n_list += [n for n in extra_ns if n <= len(X_full)]
        for N in n_list:
            X = X_full[:N]
            d = X.shape[1]
            rng = np.random.default_rng([seed_base, int(f * 1000), N])
            Q, targets, n_flip = make_queries(X, n_queries, f, rng)
            sep, nn_correct = separation(X, Q, targets, max_elems)
            for beta in betas:
                S1, Sinf, diag = hopfield_modern(X, Q, beta, max_iters, tol, max_elems)
                e1, c1, h1 = score(S1, X, targets)
                ei, ci, hi = score(Sinf, X, targets)
                rows.append({
                    "label": label, "d": int(d), "N": int(N), "flip": float(f),
                    "beta": float(beta), "n_flip": int(n_flip),
                    "exact_1step": e1, "cos_1step": c1, "hamming_1step": h1,
                    "exact_iter": ei, "cos_iter": ci, "hamming_iter": hi,
                    "argmax_correct": float(np.mean(diag["argmax_idx"] == targets)),
                    "participation_ratio": diag["participation_ratio"],
                    "top1_weight": diag["top1_weight"],
                    "norm_ratio_1step": float(np.mean(np.linalg.norm(S1, axis=1)) / np.sqrt(d)),
                    "norm_ratio_iter": float(np.mean(np.linalg.norm(Sinf, axis=1)) / np.sqrt(d)),
                    "iters_used": diag["iters_used"],
                    "mean_separation": sep,
                    "nn_correct": nn_correct,
                })
    return rows


def sweep_classical(X_full, ns, flips, n_queries, sweeps, seed_base):
    rows = []
    for f in flips:
        for N in ns:
            X = X_full[:N]
            d = X.shape[1]
            rng = np.random.default_rng([seed_base, int(f * 1000), N])
            Q, targets, _ = make_queries(X, n_queries, f, rng)
            S = hopfield_classical(X, Q, sweeps, np.random.default_rng([seed_base + 1, N]))
            e, c, h = score(S, X, targets)
            rows.append({"label": "classical", "d": int(d), "N": int(N), "flip": float(f),
                         "beta": None, "exact_iter": e, "cos_iter": c, "hamming_iter": h,
                         "exact_1step": e, "cos_1step": c, "hamming_1step": h})
    return rows


def curve(rows, **filt):
    sel = [r for r in rows if all(r[k] == v for k, v in filt.items())]
    sel.sort(key=lambda r: r["N"])
    return [r["N"] for r in sel], sel


# ----------------------------------------------------------------------------- main
def main():
    cfg = load_config()
    p = cfg["params"]
    seed = int(cfg.get("seed", 0))
    set_seeds(seed)
    t0 = time.time()

    ns = n_grid(p["n_grid_exponent_min"], p["n_grid_exponent_max"])
    deep = list(p["deep_n"])
    d = p["d_main"]
    nq = p["n_queries"]
    thr = p["capacity_threshold"]
    me = p["max_matrix_elems"]
    rng0 = np.random.default_rng(seed)

    # ---- 1. main sweep: random bipolar patterns at d = 64 -------------------
    # the headline slice (f = deep_flip) is swept an extra 4 octaves deeper,
    # because at low corruption the boundary sits far above N = 4096.
    X_main = bipolar_sign(rng0.random((max(max(ns), max(deep)), d)) - 0.5)
    main_rows = sweep_modern(X_main, ns, p["betas"], p["flip_fracs"], nq,
                             p["max_iters"], p["converge_tol"], seed, "random_d64",
                             max_elems=me, extra_ns=deep, extra_flip=p["deep_flip"])
    print(f"[{time.time()-t0:.0f}s] main sweep done ({len(main_rows)} cells)")

    # ---- 2. classical Hebbian reference ------------------------------------
    ns_cl = [n for n in ns if n <= p["classical_max_n"]]
    cl_rows = sweep_classical(X_main, ns_cl, p["flip_fracs"], nq, p["classical_sweeps"], seed)
    print(f"[{time.time()-t0:.0f}s] classical sweep done ({len(cl_rows)} cells)")

    # ---- 3. dimension sweep: is capacity exponential in d? ------------------
    dim_rows = []
    for dd in p["d_sweep"]:
        rng_d = np.random.default_rng([seed, 777, dd])
        Xd = bipolar_sign(rng_d.random((max(ns), dd)) - 0.5)
        dim_rows += sweep_modern(Xd, ns, p["d_sweep_betas"], [p["d_sweep_flip"]], nq,
                                 p["max_iters"], p["converge_tol"], seed, f"random_d{dd}",
                                 max_elems=me)
    print(f"[{time.time()-t0:.0f}s] dimension sweep done ({len(dim_rows)} cells)")

    # ---- 4. correlated real patterns: binarized 8x8 digits ------------------
    digit_rows, digits_note = [], ""
    try:
        from sklearn.datasets import load_digits
        dig = load_digits()
        imgs = dig.images.reshape(len(dig.images), -1)          # (1797, 64)
        Xdig = bipolar_sign(imgs - p["digits_threshold"] - 0.5)
        perm = np.random.default_rng([seed, 999]).permutation(len(Xdig))
        Xdig = Xdig[perm]
        # drop exact duplicates: identical stored patterns make "retrieved the right
        # index" ill-posed
        _, uniq = np.unique(Xdig, axis=0, return_index=True)
        Xdig = Xdig[np.sort(uniq)]
        k = min(300, len(Xdig))
        cos_off = (Xdig[:k] @ Xdig[:k].T / Xdig.shape[1])[np.triu_indices(k, 1)]
        ns_dig = [n for n in ns if n <= min(p["digits_max_n"], len(Xdig))]
        digit_rows = sweep_modern(Xdig, ns_dig, p["digits_betas"], p["digits_flips"], nq,
                                  p["max_iters"], p["converge_tol"], seed, "digits_d64",
                                  max_elems=me)
        digits_note = (f"{len(Xdig)} unique binarized digits after dedup (of "
                       f"{len(dig.images)}); mean |cos| between stored patterns "
                       f"{float(np.mean(np.abs(cos_off))):.3f} vs "
                       f"{np.sqrt(2/(np.pi*Xdig.shape[1])):.3f} expected for random bipolar")
    except Exception as exc:                                    # pragma: no cover
        digits_note = f"skipped: {exc}"
    print(f"[{time.time()-t0:.0f}s] digits sweep done ({len(digit_rows)} cells)")

    # ---- capacities --------------------------------------------------------
    caps_modern, caps_classical, caps_dim, caps_dig = {}, {}, {}, {}
    for f in p["flip_fracs"]:
        for beta in p["betas"]:
            _, sel = curve(main_rows, flip=f, beta=beta)
            for mode in ("iter", "1step"):
                caps_modern[f"f{f}_beta{beta}_{mode}"] = capacity_from_curve(
                    [r["N"] for r in sel], [r[f"exact_{mode}"] for r in sel], thr)
        _, selc = curve(cl_rows, flip=f)
        caps_classical[f"f{f}"] = capacity_from_curve(
            [r["N"] for r in selc], [r["exact_iter"] for r in selc], thr)
    dims_all = list(p["d_sweep"]) + [d]
    for dd in p["d_sweep"]:
        for beta in p["d_sweep_betas"]:
            _, sel = curve(dim_rows, label=f"random_d{dd}", beta=beta, flip=p["d_sweep_flip"])
            caps_dim[f"d{dd}_beta{beta}"] = capacity_from_curve(
                [r["N"] for r in sel], [r["exact_iter"] for r in sel], thr)
    # d = d_main comes from the main sweep (same f, same betas, same protocol,
    # different pattern bank) -- that is the only slice swept deep enough to be
    # uncensored at d=64.
    for beta in p["d_sweep_betas"]:
        caps_dim[f"d{d}_beta{beta}"] = dict(
            caps_modern[f"f{p['d_sweep_flip']}_beta{beta}_iter"], from_main_sweep=True)
    for f in p["digits_flips"]:
        for beta in p["digits_betas"]:
            _, sel = curve(digit_rows, flip=f, beta=beta)
            if sel:
                caps_dig[f"f{f}_beta{beta}"] = capacity_from_curve(
                    [r["N"] for r in sel], [r["exact_iter"] for r in sel], thr)

    # exponential test: log2(capacity) vs d should be LINEAR if capacity ~ exp(c d)
    exp_fit = {}
    for beta in p["d_sweep_betas"]:
        ds, lc = [], []
        for dd in dims_all:
            cell = caps_dim[f"d{dd}_beta{beta}"]
            if cell["capacity"] > 0 and not cell["censored"]:
                ds.append(dd)
                lc.append(float(np.log2(cell["capacity"])))
        if len(ds) >= 3:
            slope, intercept = np.polyfit(ds, lc, 1)
            pred = np.polyval([slope, intercept], ds)
            r2 = 1 - np.sum((np.array(lc) - pred) ** 2) / max(np.var(lc) * len(lc), 1e-12)
            exp_fit[f"beta{beta}"] = {"log2cap_per_dim": float(slope),
                                      "intercept": float(intercept),
                                      "r2": float(r2), "d_used": ds,
                                      "log2_capacity": lc}

    # ---- metastability: low beta -> averages of many patterns --------------
    meta = {}
    fh = p["deep_flip"]
    for beta in p["betas"]:
        _, sel = curve(main_rows, flip=fh, beta=beta)
        meta[f"beta{beta}"] = {
            "N": [r["N"] for r in sel],
            "participation_ratio": [round(r["participation_ratio"], 3) for r in sel],
            "norm_ratio_1step": [round(r["norm_ratio_1step"], 4) for r in sel],
            "norm_ratio_iter": [round(r["norm_ratio_iter"], 4) for r in sel],
            "cos_iter": [round(r["cos_iter"], 4) for r in sel],
            "exact_iter": [round(r["exact_iter"], 4) for r in sel],
            "argmax_correct": [round(r["argmax_correct"], 4) for r in sel],
        }

    hl = {f"beta{b}": caps_modern[f"f{fh}_beta{b}_iter"]["capacity"] for b in p["betas"]}
    headline = (
        f"d=64, f={fh}, iterated: empirical capacity (largest N with >=95% exact retrieval) = "
        + ", ".join(f"beta {b}: {hl[f'beta{b}']}" for b in p["betas"])
        + f"; classical Hebbian: {caps_classical[f'f{fh}']['capacity']} (0.14d = {0.14 * d:.1f}). "
        + "capacity at f=0.3: "
        + ", ".join(f"beta {b}: {caps_modern[f'f0.3_beta{b}_iter']['capacity']}" for b in p["betas"])
        + "; at f=0.4: "
        + ", ".join(f"beta {b}: {caps_modern[f'f0.4_beta{b}_iter']['capacity']}" for b in p["betas"])
    )

    metrics = {
        "headline": headline,
        "capacity_modern": caps_modern,
        "capacity_classical": caps_classical,
        "capacity_by_dim": caps_dim,
        "capacity_digits": caps_dig,
        "exponential_fit": exp_fit,
        "metastability_headline_flip": meta,
        "digits_note": digits_note,
        "n_grid": ns,
        "n_grid_deep": deep,
        "classical_theory_014d": 0.14 * d,
        "rows_main": main_rows,
        "rows_classical": cl_rows,
        "rows_dim": dim_rows,
        "rows_digits": digit_rows,
    }

    make_chart(cfg, p, main_rows, cl_rows, dim_rows, digit_rows,
               caps_modern, caps_classical, caps_dim, caps_dig, exp_fit, dims_all)

    results = {
        "id": cfg.get("id", "unknown"),
        "git_commit": git_sha(),
        "seed": seed,
        "duration_sec": round(time.time() - t0, 2),
        "metrics": metrics,
        "env": env_info(),
        "status": "done",
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if not k.startswith("rows_")},
                     indent=2)[:6000])
    print(f"\nTOTAL {results['duration_sec']}s")


def make_chart(cfg, p, main_rows, cl_rows, dim_rows, digit_rows,
               caps_modern, caps_classical, caps_dim, caps_dig, exp_fit, dims_all):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    colors = {0.25: "#4c72b0", 1.0: "#55a868", 4.0: "#c44e52", 16.0: "#8172b2"}
    fh = p["deep_flip"]

    # (a) retrieval rate vs N at the headline corruption level
    ax = axes[0, 0]
    for beta in p["betas"]:
        _, sel = curve(main_rows, flip=fh, beta=beta)
        ax.plot([r["N"] for r in sel], [r["exact_iter"] for r in sel], "o-",
                color=colors[beta], label=f"modern beta={beta}")
        ax.plot([r["N"] for r in sel], [r["exact_1step"] for r in sel], "--",
                color=colors[beta], alpha=0.45, lw=1)
    _, selnn = curve(main_rows, flip=fh, beta=p["betas"][-1])
    ax.plot([r["N"] for r in selnn], [r["nn_correct"] for r in selnn], ":", color="k", lw=2,
            label="nearest-neighbour decoder")
    _, selc = curve(cl_rows, flip=fh)
    ax.plot([r["N"] for r in selc], [r["exact_iter"] for r in selc], "s-k",
            label="classical Hebbian")
    ax.axhline(0.95, color="gray", ls=":", lw=1)
    ax.axvline(0.14 * p["d_main"], color="k", ls=":", lw=1, alpha=0.6)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N stored patterns")
    ax.set_ylabel("exact retrieval rate")
    ax.set_title(f"(a) d=64, f={fh}: exact retrieval vs N\nsolid=iterated, dashed=1-step, vline=0.14d")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) empirical capacity vs beta for each f  -- HEADLINE
    ax = axes[0, 1]
    width = 0.2
    xs = np.arange(len(p["betas"]))
    for i, f in enumerate(p["flip_fracs"]):
        vals = [max(caps_modern[f"f{f}_beta{b}_iter"]["capacity"], 0.5) for b in p["betas"]]
        cens = [caps_modern[f"f{f}_beta{b}_iter"]["censored"] for b in p["betas"]]
        bars = ax.bar(xs + (i - 1.5) * width, vals, width, label=f"f={f}", color=f"C{i}")
        for b, v, ce in zip(bars, vals, cens):
            lab = "<2" if v < 1 else ("$\\geq$" if ce else "") + str(int(v))
            ax.annotate(lab, (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=6)
    for i, f in enumerate(p["flip_fracs"]):
        c = caps_classical[f"f{f}"]["capacity"]
        ax.axhline(max(c, 0.5), color=f"C{i}", ls="--", lw=1, alpha=0.7)
    ax.set_yscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(b) for b in p["betas"]])
    ax.set_xlabel("inverse temperature beta")
    ax.set_ylabel("empirical capacity (largest N with >=95% exact)")
    ax.set_title("(b) HEADLINE: capacity vs beta at d=64\n(dashed = classical Hebbian at the same f)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # (c) capacity vs dimension (exponential test)
    ax = axes[0, 2]
    for beta in p["d_sweep_betas"]:
        ds = dims_all
        cs = [max(caps_dim[f"d{d_}_beta{beta}"]["capacity"], 0.5) for d_ in ds]
        cens = [caps_dim[f"d{d_}_beta{beta}"]["censored"] for d_ in ds]
        ax.plot(ds, cs, "o-", label=f"beta={beta}")
        for d_, c_, ce in zip(ds, cs, cens):
            if ce:
                ax.annotate("censored", (d_, c_), fontsize=7, ha="center", va="bottom")
        key = f"beta{beta}"
        if key in exp_fit:
            fit = exp_fit[key]
            xx = np.array(ds, dtype=float)
            ax.plot(xx, 2 ** (fit["log2cap_per_dim"] * xx + fit["intercept"]), ":",
                    label=f"fit {fit['log2cap_per_dim']:.3f} bits/dim (R2={fit['r2']:.2f})")
    ax.plot(dims_all, [0.14 * d_ for d_ in dims_all], "s--k", label="classical 0.14d")
    ax.set_yscale("log", base=2)
    ax.set_xlabel("pattern dimension d")
    ax.set_ylabel("empirical capacity")
    ax.set_title(f"(c) is capacity exponential in d? (f={fh})\nstraight line on log-y = exponential")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (d) metastability: participation ratio
    ax = axes[1, 0]
    for beta in p["betas"]:
        _, sel = curve(main_rows, flip=fh, beta=beta)
        ax.plot([r["N"] for r in sel], [r["participation_ratio"] for r in sel], "o-",
                color=colors[beta], label=f"beta={beta}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N stored patterns")
    ax.set_ylabel("participation ratio 1/sum(p^2)")
    ax.set_title("(d) metastability: effective # of patterns in the\nsoftmax mixture (f=" + str(fh) + "); 1 = clean retrieval")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (e) fixed-point norm collapse
    ax = axes[1, 1]
    for beta in p["betas"]:
        _, sel = curve(main_rows, flip=fh, beta=beta)
        ax.plot([r["N"] for r in sel], [r["norm_ratio_iter"] for r in sel], "o-",
                color=colors[beta], label=f"beta={beta}")
        ax.plot([r["N"] for r in sel], [r["cos_iter"] for r in sel], "--",
                color=colors[beta], alpha=0.5, lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N stored patterns")
    ax.set_ylabel("||xi||/sqrt(d) (solid) · cosine to true (dashed)")
    ax.set_title("(e) fixed-point norm collapse: <<1 means the state\nis an average of many patterns, not a memory")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (f) digits (correlated) vs random
    ax = axes[1, 2]
    if digit_rows:
        for i, beta in enumerate(p["digits_betas"]):
            _, sel = curve(digit_rows, flip=fh, beta=beta)
            ax.plot([r["N"] for r in sel], [r["exact_iter"] for r in sel], "o-",
                    color=f"C{i}", label=f"digits beta={beta}")
            _, selr = curve(main_rows, flip=fh, beta=beta)
            if selr:
                ax.plot([r["N"] for r in selr], [r["exact_iter"] for r in selr], "--",
                        color=f"C{i}", alpha=0.5, label=f"random beta={beta}")
        ax.axhline(0.95, color="gray", ls=":", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("N stored patterns")
        ax.set_ylabel("exact retrieval rate")
        ax.set_title(f"(f) correlated patterns (binarized 8x8 digits)\nvs random bipolar, d=64, f={fh}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "digits sweep unavailable", ha="center")

    fig.suptitle(cfg["title"], fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(HERE / "chart.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
