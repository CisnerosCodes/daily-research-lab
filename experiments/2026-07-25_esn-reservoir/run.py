"""Echo State Network on Mackey-Glass-17: is there an edge-of-chaos ridge?

Hand-rolled ESN in numpy (fixed random reservoir, ridge-regression readout only).
Sweeps the recurrent matrix's spectral radius sr from 0.30 to 1.60 in 30 steps, at three
reservoir sizes and several seeds, and measures
  (a) 1-step-ahead NRMSE (teacher forced),
  (b) free-running NRMSE at the classic MG-17 horizon of 84 steps,
  (c) valid prediction time,
  (d) linear memory capacity (delayed-input reconstruction from an i.i.d. drive),
so we can ask whether the NRMSE optimum is a sharp ridge just below sr=1 and whether it
coincides with the memory-capacity peak.

Deterministic, CPU-only, single threaded.  Usage:  python run.py
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------------------
# boilerplate
# --------------------------------------------------------------------------------------
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
    for mod in ("numpy", "torch", "matplotlib"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:
            pass
    return info


# --------------------------------------------------------------------------------------
# Mackey-Glass tau=17
# --------------------------------------------------------------------------------------
def mackey_glass(n, tau, dt, beta, gamma, power, x0, discard):
    """RK4 integration of dx/dt = beta*x(t-tau)/(1+x(t-tau)^p) - gamma*x(t), sampled at unit dt.

    Deterministic: constant history x(t)=x0 on [-tau, 0].  The delayed term is held fixed
    across each RK4 step (standard for delay-DEs at small dt).
    """
    steps_per_unit = int(round(1.0 / dt))
    total = (n + discard) * steps_per_unit
    d = int(round(tau / dt))
    buf = np.empty(total + d + 1)
    buf[: d + 1] = x0

    def f(x, xd):
        return beta * xd / (1.0 + xd ** power) - gamma * x

    for i in range(d, total + d):
        x = buf[i]
        xd = buf[i - d]
        k1 = f(x, xd)
        k2 = f(x + 0.5 * dt * k1, xd)
        k3 = f(x + 0.5 * dt * k2, xd)
        k4 = f(x + dt * k3, xd)
        buf[i + 1] = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    series = buf[d + 1 :][::steps_per_unit]
    return series[discard : discard + n].copy()


# --------------------------------------------------------------------------------------
# ESN
# --------------------------------------------------------------------------------------
def make_base_reservoir(N, seed, density):
    """Topology and weights are drawn ONCE per (N, seed); sr only rescales W.

    This makes the spectral-radius sweep a within-reservoir comparison, and lets the
    input-scaling arms reuse identical topologies.
    """
    rng = np.random.default_rng(seed)
    mask = rng.random((N, N)) < density
    W = np.zeros((N, N))
    W[mask] = rng.uniform(-1.0, 1.0, int(mask.sum()))
    rho = float(np.max(np.abs(np.linalg.eigvals(W))))
    W_unit = W / rho  # spectral radius exactly 1
    Win_unit = rng.uniform(-1.0, 1.0, N)
    bias_unit = rng.uniform(-1.0, 1.0, N)
    return W_unit, Win_unit, bias_unit, rho


def collect_states(W, Win, bias, u, leak):
    """Teacher-forced reservoir run.  X[t] is the state after consuming u[t]."""
    N = W.shape[0]
    T = len(u)
    X = np.empty((T, N))
    x = np.zeros(N)
    if leak >= 1.0:
        for t in range(T):
            x = np.tanh(W @ x + Win * u[t] + bias)
            X[t] = x
    else:
        for t in range(T):
            x = (1.0 - leak) * x + leak * np.tanh(W @ x + Win * u[t] + bias)
            X[t] = x
    return X


def ridge_fit_multi(Phi, Y, alphas):
    """Return {alpha: Wout} for a shared design matrix.  Gram is built once."""
    G = Phi.T @ Phi
    C = Phi.T @ Y
    eye = np.eye(G.shape[0])
    out = {}
    for a in alphas:
        try:
            out[a] = np.linalg.solve(G + a * eye, C)
        except np.linalg.LinAlgError:
            out[a] = np.linalg.lstsq(G + a * eye, C, rcond=None)[0]
    return out


def nrmse(pred, true, ref_std):
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    if not np.all(np.isfinite(pred)):
        return float("inf")
    return float(np.sqrt(np.mean((pred - true) ** 2)) / ref_std)


def free_run(W, Win, bias, leak, Wout, x_start, u_start, horizon, clip):
    """Generative mode: the readout's own output is fed back as the next input."""
    x = x_start.copy()
    Wout = np.asarray(Wout).ravel()
    u_cur = float(u_start)
    preds = np.empty(horizon)
    clipped = 0
    for h in range(horizon):
        phi = np.empty(2 + x.shape[0])
        phi[0] = 1.0
        phi[1] = u_cur
        phi[2:] = x
        p = float(phi @ Wout)
        if not np.isfinite(p):
            preds[h:] = np.nan
            return preds, clipped, True
        if abs(p) > clip:
            p = clip if p > 0 else -clip
            clipped += 1
        preds[h] = p
        z = np.tanh(W @ x + Win * p + bias)
        x = z if leak >= 1.0 else (1.0 - leak) * x + leak * z
        u_cur = p
    return preds, clipped, False


# --------------------------------------------------------------------------------------
# memory capacity
# --------------------------------------------------------------------------------------
def memory_capacity(W, Win, bias, leak, N, p, rng, sig_std):
    kmax = int(round(p["mc_kmax_frac"] * N))
    n_wash, n_tr, n_va, n_te = p["mc_washout"], p["mc_train"], p["mc_val"], p["mc_test"]
    T = n_wash + n_tr + n_va + n_te
    amp = (np.sqrt(3.0) * sig_std) if p["mc_match_input_std"] else 1.0
    u = rng.uniform(-amp, amp, T + kmax)
    u_drive = u[kmax:]  # u_drive[t] is the input at time t; u[kmax + t - k] is its k-lag
    X = collect_states(W, Win, bias, u_drive, leak)

    Phi = np.concatenate([np.ones((T, 1)), X], axis=1)
    Y = np.empty((T, kmax))
    for k in range(1, kmax + 1):
        Y[:, k - 1] = u[kmax - k : kmax - k + T]

    s_tr = slice(n_wash, n_wash + n_tr)
    s_va = slice(n_wash + n_tr, n_wash + n_tr + n_va)
    s_te = slice(n_wash + n_tr + n_va, T)

    sols = ridge_fit_multi(Phi[s_tr], Y[s_tr], p["ridge_alphas"])

    def total_mc(sl, Wout):
        P = Phi[sl] @ Wout
        Yt = Y[sl]
        tot = 0.0
        curve = np.empty(kmax)
        for k in range(kmax):
            a, b = P[:, k], Yt[:, k]
            va, vb = a.var(), b.var()
            if va <= 1e-14 or vb <= 1e-14 or not np.isfinite(va):
                curve[k] = 0.0
                continue
            r = np.mean((a - a.mean()) * (b - b.mean())) / np.sqrt(va * vb)
            curve[k] = float(min(max(r * r, 0.0), 1.0))
            tot += curve[k]
        return tot, curve

    best_a, best_v = None, -1.0
    for a, Wo in sols.items():
        v, _ = total_mc(s_va, Wo)
        if v > best_v:
            best_v, best_a = v, a
    mc, curve = total_mc(s_te, sols[best_a])
    return float(mc), float(best_a), curve


# --------------------------------------------------------------------------------------
# one (N, sr, seed) cell
# --------------------------------------------------------------------------------------
def run_cell(N, sr, seed, base, y, idx, p, in_scale, arm_label):
    W_unit, Win_unit, bias_unit, _ = base
    W = W_unit * sr
    Win = Win_unit * in_scale
    bias = bias_unit * in_scale
    leak = p["leak_rate"]
    H = p["horizon"]
    clip = p["freerun_clip"]
    alphas = p["ridge_alphas"]

    X = collect_states(W, Win, bias, y, leak)
    T = len(y)
    Phi = np.concatenate([np.ones((T, 1)), y.reshape(-1, 1), X], axis=1)
    target = np.empty(T)
    target[:-1] = y[1:]
    target[-1] = np.nan  # unusable

    tr, va, te = idx["train"], idx["val"], idx["test"]
    sols = ridge_fit_multi(Phi[tr], target[tr].reshape(-1, 1), alphas)

    sig_std = float(y[te].std())

    # ---- 1-step-ahead: alpha selected on validation ----
    best_a1, best_v1 = None, float("inf")
    for a, Wo in sols.items():
        v = nrmse((Phi[va] @ Wo).ravel(), target[va], float(y[va].std()))
        if v < best_v1:
            best_v1, best_a1 = v, a
    nrmse1_test = nrmse((Phi[te] @ sols[best_a1]).ravel(), target[te], sig_std)
    nrmse1_train = nrmse((Phi[tr] @ sols[best_a1]).ravel(), target[tr], float(y[tr].std()))

    # ---- free running: alpha selected on validation NRMSE@H ----
    def freerun_eval(Wout, starts):
        errs_end, traj_sq, vts, div, clipped_tot = [], np.zeros(H), [], 0, 0
        for t0 in starts:
            preds, nclip, diverged = free_run(
                W, Win, bias, leak, Wout, X[t0], y[t0], H, clip
            )
            truth = y[t0 + 1 : t0 + 1 + H]
            clipped_tot += nclip
            if diverged or not np.all(np.isfinite(preds)):
                div += 1
                errs_end.append(np.nan)
                vts.append(0)
                continue
            e = preds - truth
            errs_end.append(e[-1] ** 2)
            traj_sq += e ** 2
            bad = np.nonzero(np.abs(e) > p["valid_time_thresh"] * sig_std)[0]
            vts.append(int(bad[0]) if len(bad) else H)
        n_ok = int(np.sum(~np.isnan(errs_end)))
        if n_ok == 0:
            return float("inf"), float("inf"), 0.0, div, clipped_tot
        end = float(np.sqrt(np.nanmean(errs_end)) / sig_std)
        traj = float(np.sqrt(np.mean(traj_sq / max(n_ok, 1))) / sig_std)
        return end, traj, float(np.mean(vts)), div, clipped_tot

    va_starts = np.linspace(va.start, va.stop - H - 2, p["n_freerun_val"]).astype(int)
    te_starts = np.linspace(te.start, te.stop - H - 2, p["n_freerun_test"]).astype(int)

    best_af, best_vf = None, float("inf")
    for a, Wo in sols.items():
        v, _, _, _, _ = freerun_eval(Wo, va_starts)
        if v < best_vf:
            best_vf, best_af = v, a
    nrmse_h, nrmse_traj, vt, ndiv, nclip = freerun_eval(sols[best_af], te_starts)

    wnorm = float(np.linalg.norm(sols[best_af]))

    mc, mc_alpha, mc_curve = (float("nan"), float("nan"), None)
    if p["mc_enabled"]:
        rng = np.random.default_rng(100000 + 1000 * seed + N)
        mc, mc_alpha, mc_curve = memory_capacity(
            W, Win, bias, leak, N, p, rng, float(y.std())
        )

    return {
        "arm": arm_label,
        "N": N,
        "input_scaling": float(in_scale),
        "sr": float(sr),
        "seed": int(seed),
        "nrmse_1step_test": nrmse1_test,
        "nrmse_1step_train": nrmse1_train,
        "alpha_1step": float(best_a1),
        "nrmse_h84_test": nrmse_h,
        "nrmse_traj84_test": nrmse_traj,
        "valid_time": vt,
        "alpha_freerun": float(best_af),
        "n_diverged": int(ndiv),
        "n_clipped_steps": int(nclip),
        "readout_norm": wnorm,
        "memory_capacity": mc,
        "mc_alpha": mc_alpha,
        "mc_curve_head": [round(float(v), 4) for v in mc_curve[:30]] if mc_curve is not None else None,
    }


# --------------------------------------------------------------------------------------
# aggregation helpers
# --------------------------------------------------------------------------------------
def band(srs, vals, factor, mode="min"):
    """Hull of spectral radii within `factor` of the optimum."""
    v = np.asarray(vals, dtype=float)
    s = np.asarray(srs, dtype=float)
    ok = np.isfinite(v)
    if not ok.any():
        return None
    if mode == "min":
        best = np.nanmin(v[ok])
        sel = ok & (v <= factor * best)
        opt = float(s[ok][np.nanargmin(v[ok])])
    else:
        best = np.nanmax(v[ok])
        sel = ok & (v >= best / factor)
        opt = float(s[ok][np.nanargmax(v[ok])])
    ssel = s[sel]
    return {
        "opt_sr": opt,
        "opt_val": float(best),
        "band_lo": float(ssel.min()),
        "band_hi": float(ssel.max()),
        "band_width": float(ssel.max() - ssel.min()),
        "band_n_points": int(sel.sum()),
        "n_points_total": int(ok.sum()),
        "band_frac_of_sweep": round(float(sel.sum()) / float(ok.sum()), 4),
    }


def main():
    cfg = load_config()
    p = cfg["params"]
    seed = int(cfg.get("seed", 0))
    set_seeds(seed)
    t0 = time.time()

    # ---------------- data ----------------
    n_need = p["n_washout"] + p["n_train"] + p["n_val"] + p["n_test"] + p["horizon"] + 5
    raw = mackey_glass(
        n_need, p["mg_tau"], p["mg_dt"], p["mg_beta"], p["mg_gamma"],
        p["mg_power"], p["mg_x0"], p["mg_discard"],
    )
    y = np.tanh(raw - 1.0) if p["mg_squash"] else raw.copy()
    a = p["n_washout"]
    b = a + p["n_train"]
    c = b + p["n_val"]
    d = c + p["n_test"]
    idx = {"train": slice(a, b), "val": slice(b, c), "test": slice(c, d)}
    data_info = {
        "n_generated": int(len(raw)),
        "raw_min": float(raw.min()),
        "raw_max": float(raw.max()),
        "raw_std": float(raw.std()),
        "squashed_std": float(y.std()),
        "test_std": float(y[idx["test"]].std()),
        "splits": {k: [v.start, v.stop] for k, v in idx.items()},
    }
    print(f"[data] MG-17 n={len(raw)} raw range [{raw.min():.3f},{raw.max():.3f}] "
          f"squashed std {y.std():.4f}  ({time.time()-t0:.1f}s)")

    # ---------------- sweep ----------------
    srs = [float(s) for s in p["spectral_radii"]]
    arms = [dict(a) for a in p["arms"]]
    cells = []
    skipped = []
    cap = float(p["time_cap_s"])
    base_cache = {}

    for arm in arms:
        N, in_scale, lbl = int(arm["N"]), float(arm["input_scaling"]), arm["label"]
        for s_i in range(int(arm["n_seeds"])):
            key = (N, s_i)
            if key not in base_cache:
                base_cache[key] = make_base_reservoir(N, 1000 * s_i + N, p["density"])
            base = base_cache[key]
            for sr in srs:
                if time.time() - t0 > cap:
                    skipped.append({"arm": lbl, "seed": s_i, "sr": sr})
                    continue
                cells.append(run_cell(N, sr, s_i, base, y, idx, p, in_scale, lbl))
            done = [c for c in cells if c["arm"] == lbl and c["seed"] == s_i]
            if done:
                bst = min(done, key=lambda r: r["nrmse_h84_test"])
                print(f"[sweep] {lbl} seed={s_i}  argmin-sr(84)={bst['sr']:.3f} "
                      f"nrmse84={bst['nrmse_h84_test']:.4g}  "
                      f"nrmse1_best={min(r['nrmse_1step_test'] for r in done):.2e}  "
                      f"({time.time()-t0:.0f}s)", flush=True)

    # ---------------- aggregate ----------------
    agg = {}
    for arm in arms:
        lbl = arm["label"]
        rows = []
        for sr in srs:
            sel = [c for c in cells if c["arm"] == lbl and abs(c["sr"] - sr) < 1e-9]
            if not sel:
                continue

            def m(key, _sel=sel):
                v = np.array([x[key] for x in _sel], dtype=float)
                if not np.isfinite(v).any():
                    return (float("nan"),) * 4
                return (float(np.nanmean(v)), float(np.nanstd(v)),
                        float(np.nanmin(v)), float(np.nanmax(v)))

            r1 = m("nrmse_1step_test"); r84 = m("nrmse_h84_test")
            rt = m("nrmse_traj84_test"); vt = m("valid_time"); mc = m("memory_capacity")
            rows.append({
                "sr": sr, "n_seeds": len(sel),
                "nrmse1_mean": r1[0], "nrmse1_std": r1[1], "nrmse1_min": r1[2], "nrmse1_max": r1[3],
                "nrmse84_mean": r84[0], "nrmse84_std": r84[1], "nrmse84_min": r84[2], "nrmse84_max": r84[3],
                "nrmse_traj_mean": rt[0], "valid_time_mean": vt[0], "valid_time_std": vt[1],
                "mc_mean": mc[0], "mc_std": mc[1],
                "n_diverged": int(sum(x["n_diverged"] for x in sel)),
            })
        if rows:
            agg[lbl] = rows

    factor = float(p["ridge_width_factor"])
    analysis = {}
    for arm in arms:
        lbl = arm["label"]
        rows = agg.get(lbl)
        if not rows:
            continue
        S = [r["sr"] for r in rows]
        a1 = band(S, [r["nrmse1_mean"] for r in rows], factor, "min")
        a84 = band(S, [r["nrmse84_mean"] for r in rows], factor, "min")
        amc = band(S, [r["mc_mean"] for r in rows], factor, "max")
        avt = band(S, [r["valid_time_mean"] for r in rows], factor, "max")
        per_seed_opt = {}
        for key, mk in (("nrmse_1step_test", "min"), ("nrmse_h84_test", "min"),
                        ("memory_capacity", "max")):
            opts = []
            for s_i in range(int(arm["n_seeds"])):
                sel = sorted([c for c in cells if c["arm"] == lbl and c["seed"] == s_i],
                             key=lambda r: r["sr"])
                if not sel:
                    continue
                v = np.array([x[key] for x in sel], dtype=float)
                if not np.isfinite(v).any():
                    continue
                j = int(np.nanargmin(v)) if mk == "min" else int(np.nanargmax(v))
                opts.append(float(sel[j]["sr"]))
            per_seed_opt[key] = opts
        vals84 = np.array([r["nrmse84_mean"] for r in rows], dtype=float)
        vals1 = np.array([r["nrmse1_mean"] for r in rows], dtype=float)
        valsmc = np.array([r["mc_mean"] for r in rows], dtype=float)
        Sarr = np.array(S)

        def at(sr_q, arr, _S=Sarr):
            j = int(np.argmin(np.abs(_S - sr_q)))
            return float(arr[j])

        probe = (0.3, 0.5, 0.7, 0.9, 0.95, 1.0, 1.05, 1.2, 1.4, 1.6)
        analysis[lbl] = {
            "N": int(arm["N"]), "input_scaling": float(arm["input_scaling"]),
            "n_seeds": int(arm["n_seeds"]),
            "nrmse1": a1, "nrmse84": a84, "memory_capacity": amc, "valid_time": avt,
            "per_seed_argopt": per_seed_opt,
            "nrmse84_at": {f"{q:.3f}": at(q, vals84) for q in probe},
            "nrmse1_at": {f"{q:.3f}": at(q, vals1) for q in probe},
            "mc_at": {f"{q:.3f}": at(q, valsmc) for q in probe},
            "ratio_worst_over_best_nrmse84_subcritical": float(
                np.nanmax(vals84[Sarr <= 1.0]) / np.nanmin(vals84)),
            "ratio_sr1p6_over_best_nrmse84": float(at(1.6, vals84) / np.nanmin(vals84)),
            "ratio_sr0p3_over_best_nrmse84": float(at(0.3, vals84) / np.nanmin(vals84)),
            "ratio_sr1p6_over_best_nrmse1": float(at(1.6, vals1) / np.nanmin(vals1)),
            "ratio_sr0p3_over_best_nrmse1": float(at(0.3, vals1) / np.nanmin(vals1)),
            "mc_opt_minus_nrmse84_opt_sr": (
                round(amc["opt_sr"] - a84["opt_sr"], 4) if (amc and a84) else None),
            "mc_opt_minus_nrmse1_opt_sr": (
                round(amc["opt_sr"] - a1["opt_sr"], 4) if (amc and a1) else None),
            "frac_of_sweep_within_1p5x_nrmse84": (a84 or {}).get("band_frac_of_sweep"),
            "frac_of_sweep_within_1p5x_nrmse1": (a1 or {}).get("band_frac_of_sweep"),
            "n_diverged_total": int(sum(r["n_diverged"] for r in rows)),
        }

    head_lbl = p["headline_arm"]
    hA = analysis.get(head_lbl, {})
    headline = {
        "headline_arm": head_lbl,
        "best_sr_nrmse1": (hA.get("nrmse1") or {}).get("opt_sr"),
        "best_nrmse1": (hA.get("nrmse1") or {}).get("opt_val"),
        "best_sr_nrmse84": (hA.get("nrmse84") or {}).get("opt_sr"),
        "best_nrmse84": (hA.get("nrmse84") or {}).get("opt_val"),
        "best_sr_mc": (hA.get("memory_capacity") or {}).get("opt_sr"),
        "best_mc": (hA.get("memory_capacity") or {}).get("opt_val"),
        "nrmse84_band_1p5x": [(hA.get("nrmse84") or {}).get("band_lo"),
                              (hA.get("nrmse84") or {}).get("band_hi")],
        "nrmse84_band_frac_of_sweep": (hA.get("nrmse84") or {}).get("band_frac_of_sweep"),
        "nrmse1_band_1p5x": [(hA.get("nrmse1") or {}).get("band_lo"),
                             (hA.get("nrmse1") or {}).get("band_hi")],
        "nrmse1_band_frac_of_sweep": (hA.get("nrmse1") or {}).get("band_frac_of_sweep"),
        "argmin_sr_nrmse84_by_arm": {k: (v.get("nrmse84") or {}).get("opt_sr")
                                     for k, v in analysis.items()},
        "argmin_sr_nrmse1_by_arm": {k: (v.get("nrmse1") or {}).get("opt_sr")
                                    for k, v in analysis.items()},
        "argmax_sr_mc_by_arm": {k: (v.get("memory_capacity") or {}).get("opt_sr")
                                for k, v in analysis.items()},
        "best_nrmse84_by_arm": {k: (v.get("nrmse84") or {}).get("opt_val")
                                for k, v in analysis.items()},
    }

    metrics = {
        "data": data_info,
        "sweep": {"spectral_radii": srs, "arms": arms, "n_cells": len(cells),
                  "n_skipped_cells": len(skipped), "skipped": skipped[:20]},
        "headline": headline,
        "analysis": analysis,
        "by_arm": agg,
        "per_cell": cells,
    }

    make_chart(agg, analysis, p)

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
    print(json.dumps({"headline": headline, "duration_sec": results["duration_sec"]}, indent=2))


# --------------------------------------------------------------------------------------
# chart
# --------------------------------------------------------------------------------------
def make_chart(agg, analysis, p):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    fig, ax = plt.subplots(2, 3, figsize=(17.0, 9.0))

    def plot_group(a, labels, key_mean, key_std, title, ylabel, logy, legend_fmt,
                   band_keys=None):
        """band_keys = (lo_key, hi_key) uses the seed min/max envelope (needed on log
        axes, where mean-std goes negative); otherwise mean +/- std."""
        for i, lbl in enumerate(labels):
            rows = agg.get(lbl)
            if not rows:
                continue
            S = np.array([r["sr"] for r in rows])
            M = np.array([r[key_mean] for r in rows], dtype=float)
            if band_keys is not None:
                LO = np.array([r[band_keys[0]] for r in rows], dtype=float)
                HI = np.array([r[band_keys[1]] for r in rows], dtype=float)
            else:
                E = np.array([r[key_std] for r in rows], dtype=float)
                LO, HI = M - E, M + E
            col = palette[i % len(palette)]
            A = analysis.get(lbl, {})
            a.plot(S, M, "o-", ms=3.5, lw=1.5, color=col,
                   label=legend_fmt(lbl, A))
            a.fill_between(S, LO, HI, color=col, alpha=0.18, lw=0)
        a.axvline(1.0, color="k", ls="--", lw=1, alpha=0.6)
        a.text(1.01, 0.03, "sr = 1", transform=a.get_xaxis_transform(), fontsize=8, alpha=0.7)
        if logy:
            a.set_yscale("log")
        a.set_xlabel("spectral radius  sr")
        a.set_ylabel(ylabel)
        a.set_title(title, fontsize=10)
        a.grid(alpha=0.25)
        a.legend(fontsize=7.5)

    size_arms = list(p["size_arms"])
    ins_arms = list(p["inscale_arms"])

    def fmt_size(lbl, A):
        return f"N={A.get('N','?')} ({A.get('n_seeds','?')} seeds)"

    def fmt_ins(lbl, A):
        return f"input scaling {A.get('input_scaling','?')}"

    plot_group(ax[0, 0], size_arms, "nrmse1_mean", "nrmse1_std",
               "1-step-ahead (teacher forced), input scaling 0.5",
               "test NRMSE (log), band = seed min-max", True, fmt_size,
               band_keys=("nrmse1_min", "nrmse1_max"))
    plot_group(ax[0, 1], size_arms, "nrmse84_mean", "nrmse84_std",
               "free-running error at horizon 84 (MG-17 benchmark)",
               "test NRMSE@84 (log), band = seed min-max", True, fmt_size,
               band_keys=("nrmse84_min", "nrmse84_max"))
    plot_group(ax[0, 2], size_arms, "mc_mean", "mc_std",
               "linear memory capacity (i.i.d. drive)",
               "MC = sum_k r^2(k)", False, fmt_size)
    plot_group(ax[1, 0], ins_arms, "nrmse84_mean", "nrmse84_std",
               "N=200: does input drive move the optimum? (NRMSE@84)",
               "test NRMSE@84 (log), band = seed min-max", True, fmt_ins,
               band_keys=("nrmse84_min", "nrmse84_max"))
    plot_group(ax[1, 1], ins_arms, "mc_mean", "mc_std",
               "N=200: memory capacity vs input drive",
               "MC = sum_k r^2(k)", False, fmt_ins)

    a = ax[1, 2]
    head_lbl = p["headline_arm"]
    rows = agg.get(head_lbl, [])
    if rows:
        S = np.array([r["sr"] for r in rows])

        def norm01(v, invert=False):
            v = np.array(v, dtype=float)
            if invert:
                v = -np.log10(np.maximum(v, 1e-12))
            lo, hi = np.nanmin(v), np.nanmax(v)
            return (v - lo) / (hi - lo + 1e-12)

        a.plot(S, norm01([r["nrmse1_mean"] for r in rows], invert=True), "o-", ms=3.5,
               label="1-step skill (-log NRMSE, scaled)", color="#4C72B0")
        a.plot(S, norm01([r["nrmse84_mean"] for r in rows], invert=True), "s-", ms=3.5,
               label="84-step skill (-log NRMSE, scaled)", color="#C44E52")
        a.plot(S, norm01([r["mc_mean"] for r in rows]), "^-", ms=3.5,
               label="memory capacity (scaled)", color="#55A868")
        A = analysis.get(head_lbl, {})
        for key, col in (("nrmse1", "#4C72B0"), ("nrmse84", "#C44E52"),
                         ("memory_capacity", "#55A868")):
            dct = A.get(key)
            if dct:
                a.axvline(dct["opt_sr"], color=col, ls=":", lw=1.8)
        b84 = A.get("nrmse84")
        if b84:
            a.axvspan(b84["band_lo"], b84["band_hi"], color="#C44E52", alpha=0.10, lw=0)
        a.axvline(1.0, color="k", ls="--", lw=1, alpha=0.6)
        a.set_xlabel("spectral radius  sr")
        a.set_ylabel("min-max scaled score (higher = better)")
        a.set_title(f"{head_lbl}: do the optima coincide?\n"
                    f"shaded = within 1.5x of best NRMSE@84; dotted = optima",
                    fontsize=10)
        a.grid(alpha=0.25)
        a.legend(fontsize=7.5, loc="lower left")

    fig.suptitle("Echo State Network on Mackey-Glass-17: spectral-radius sweep "
                 "(fixed random reservoir, ridge readout only)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(HERE / "chart.png", dpi=135)
    plt.close(fig)


def rebuild_chart_only():
    """Redraw chart.png from an existing results.json without recomputing the sweep."""
    cfg = load_config()
    with open(HERE / "results.json") as f:
        m = json.load(f)["metrics"]
    make_chart(m["by_arm"], m["analysis"], cfg["params"])
    print("chart.png rebuilt from results.json")


if __name__ == "__main__":
    if "--chart-only" in sys.argv:
        rebuild_chart_only()
    else:
        main()
