"""KAN vs iso-param MLP on Feynman I.12.2 + symbolic extraction.

Deterministic, CPU-only, single-threaded. Writes results.json and chart.png.

Usage:  python run.py
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import contextlib
import io
import json
import math
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CKPT = Path(tempfile.mkdtemp(prefix="kanckpt_"))


# ----------------------------------------------------------------------------- utils
def set_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    for mod in ("numpy", "torch", "kan", "sympy", "matplotlib"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:
            pass
    return info


@contextlib.contextmanager
def quiet():
    """pykan prints a tqdm bar + checkpoint chatter to both streams."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        yield buf


# ----------------------------------------------------------------------------- data
def make_data(n, seed, ranges):
    """Feynman I.12.2: F = q1*q2 / (4*pi*eps*r^2), all vars ~ U[1,5]."""
    g = torch.Generator().manual_seed(int(seed))
    lo = torch.tensor([r[0] for r in ranges])
    hi = torch.tensor([r[1] for r in ranges])
    X = torch.rand(n, len(ranges), generator=g) * (hi - lo) + lo
    q1, q2, eps, r = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    y = q1 * q2 / (4 * math.pi * eps * r**2)
    return X, y[:, None]


# ----------------------------------------------------------------------------- models
def build_mlp(hidden, seed, d_in=4):
    torch.manual_seed(seed)
    layers, d = [], d_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.Tanh()]
        d = h
    layers += [nn.Linear(d, 1)]
    return nn.Sequential(*layers)


def fit_mlp(Z, Y, hidden, opt_name, P, seed=0):
    m = build_mlp(hidden, seed)
    if opt_name == "lbfgs":
        opt = torch.optim.LBFGS(
            m.parameters(),
            lr=P["mlp_lbfgs_lr"],
            max_iter=P["mlp_lbfgs_max_iter"],
            history_size=100,
            line_search_fn="strong_wolfe",
        )
        for _ in range(P["mlp_lbfgs_steps"]):

            def closure():
                opt.zero_grad()
                loss = ((m(Z) - Y) ** 2).mean()
                loss.backward()
                return loss

            opt.step(closure)
    else:
        opt = torch.optim.Adam(m.parameters(), lr=P["mlp_adam_lr"])
        for _ in range(P["mlp_adam_steps"]):
            opt.zero_grad()
            loss = ((m(Z) - Y) ** 2).mean()
            loss.backward()
            opt.step()
    return m


def fit_kan_sweep(Z, Y, Zte_small, Yte_small, P):
    from kan import KAN

    ds = {
        "train_input": Z,
        "train_label": Y,
        "test_input": Zte_small,
        "test_label": Yte_small,
    }
    with quiet():
        m = KAN(
            width=list(P["kan_width"]),  # pykan rewrites the list in place
            grid=P["kan_grid"],
            k=P["kan_k"],
            seed=P["kan_init_seed"],
            device="cpu",
            ckpt_path=str(CKPT),
        )
        m.fit(
            ds,
            opt="LBFGS",
            steps=P["kan_steps"],
            lamb=P["kan_lamb"],
            stop_grid_update_step=P["kan_stop_grid_update_step"],
        )
    return m


def raw_rmse(model, Xte, yte, stats):
    mu, sd, ym, ys = stats
    with torch.no_grad():
        pred = model((Xte - mu) / sd) * ys + ym
    return float(((pred - yte) ** 2).mean().sqrt())


# ----------------------------------------------------------------------------- analysis
PRIMS = {
    "log(x)": lambda x: np.log(x),
    "x": lambda x: x,
    "x^2": lambda x: x**2,
    "1/x": lambda x: 1.0 / x,
    "1/x^2": lambda x: 1.0 / x**2,
    "sqrt(x)": lambda x: np.sqrt(x),
    "exp(x)": lambda x: np.exp(np.clip(x, -30, 30)),
}


def _r2(gx, y):
    A = np.stack([gx, np.ones_like(gx)], 1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1 - (resid**2).sum() / ss_tot) if ss_tot > 0 else 0.0


def best_primitive(x, y):
    """Fit y ~ a*g(x)+b by OLS for each candidate g; return ranked (name, R2).

    log/sqrt/powers absorb an inner scale into the outer affine params, but exp does
    NOT (exp(b*x) != a*exp(x)+c), so the exponential family is scanned over b.
    """
    out = {}
    with np.errstate(all="ignore"):
        for name, g in PRIMS.items():
            try:
                gx = g(x)
                if not np.all(np.isfinite(gx)):
                    continue
                out[name] = _r2(gx, y)
            except Exception:
                continue
        # exponential family with an inner scale
        bs = np.concatenate([np.geomspace(0.02, 5.0, 30), -np.geomspace(0.02, 5.0, 30)])
        best_b, best_r = None, -np.inf
        for b in bs:
            gx = np.exp(np.clip(b * x, -30, 30))
            if not np.all(np.isfinite(gx)) or gx.std() == 0:
                continue
            r = _r2(gx, y)
            if r > best_r:
                best_r, best_b = r, b
        if best_b is not None:
            out.pop("exp(x)", None)
            out[f"exp({best_b:.3g}*x)"] = best_r
    return sorted(out.items(), key=lambda kv: -kv[1])


def loglog_exponents(X, pred):
    """OLS of log|pred| on log(x_i); recovers the power-law exponents if pred is a monomial."""
    Xn = X.numpy() if torch.is_tensor(X) else X
    p = pred.detach().numpy().ravel() if torch.is_tensor(pred) else np.asarray(pred).ravel()
    ok = p > 0
    if ok.sum() < 20:
        return None, None, float(ok.mean())
    A = np.concatenate([np.log(Xn[ok]), np.ones((int(ok.sum()), 1))], 1)
    b = np.log(p[ok])
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = b - A @ coef
    r2 = float(1 - (resid**2).sum() / ((b - b.mean()) ** 2).sum())
    return [float(c) for c in coef[:-1]], r2, float(ok.mean())


# ----------------------------------------------------------------------------- main
def main():
    cfg = load_config()
    P = cfg["params"]
    EQ = cfg["equation"]
    seed = int(cfg.get("seed", 0))
    set_seeds(seed)
    t0 = time.time()

    ranges = EQ["ranges"]
    var_names = EQ["variables"]
    kan_width_label = "[" + ",".join(str(w) for w in P["kan_width"]) + "]"
    sym_width_label = "[" + ",".join(str(w) for w in P["sym_width"]) + "]"
    Xte, yte = make_data(P["n_test"], P["test_seed"], ranges)
    y_test_std = float(yte.std())

    # =============================================================== 1. sample-efficiency sweep
    arms = ["kan", "mlp_deep_lbfgs", "mlp_deep_adam", "mlp_wide_lbfgs", "mlp_wide_adam"]
    sweep = {a: {str(n): [] for n in P["train_sizes"]} for a in arms}
    n_params = {}
    t_sweep = time.time()

    for n in P["train_sizes"]:
        for ds_seed in P["data_seeds"]:
            Xtr, ytr = make_data(n, 1000 + ds_seed, ranges)
            mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-8)
            ym, ys = ytr.mean(), ytr.std().clamp_min(1e-8)
            stats = (mu, sd, ym, ys)
            Z, Y = (Xtr - mu) / sd, (ytr - ym) / ys
            Zte = (Xte - mu) / sd
            Yte = (yte - ym) / ys

            # KAN (single fixed protocol)
            set_seeds(seed)
            k = fit_kan_sweep(Z, Y, Zte[:200], Yte[:200], P)
            with quiet():
                sweep["kan"][str(n)].append(raw_rmse(k, Xte, yte, stats))
            n_params["kan"] = int(
                sum(p.numel() for nm, p in k.named_parameters() if p.requires_grad and "act_fun" in nm)
            )

            # iso-param MLPs, two optimizers each (baseline gets the best-of advantage)
            for tag, hid in (("deep", P["mlp_deep_hidden"]), ("wide", P["mlp_wide_hidden"])):
                for opt_name in ("lbfgs", "adam"):
                    set_seeds(seed)
                    m = fit_mlp(Z, Y, hid, opt_name, P, seed=seed)
                    sweep[f"mlp_{tag}_{opt_name}"][str(n)].append(raw_rmse(m, Xte, yte, stats))
                    n_params[f"mlp_{tag}"] = int(sum(p.numel() for p in m.parameters()))
        print(f"[sweep] n={n} done  t={time.time() - t_sweep:.0f}s", flush=True)

    mean = {a: {n: float(np.mean(v)) for n, v in d.items()} for a, d in sweep.items()}
    # "best MLP" = oracle-selected best MLP arm per n (conservative w.r.t. the KAN hypothesis)
    mlp_arms = [a for a in arms if a.startswith("mlp")]
    best_mlp = {str(n): min(mean[a][str(n)] for a in mlp_arms) for n in P["train_sizes"]}
    best_mlp_arm = {
        str(n): min(mlp_arms, key=lambda a: mean[a][str(n)]) for n in P["train_sizes"]
    }
    ratio = {str(n): mean["kan"][str(n)] / best_mlp[str(n)] for n in P["train_sizes"]}
    ratio_vs_single = {
        str(n): mean["kan"][str(n)] / mean["mlp_deep_lbfgs"][str(n)] for n in P["train_sizes"]
    }
    kan_wins = sum(1 for n in P["train_sizes"] if ratio[str(n)] < 1.0)
    wins_at = [n for n in P["train_sizes"] if ratio[str(n)] < 1.0]
    n_first_win = min(wins_at) if wins_at else None
    n_lo, n_hi = str(P["train_sizes"][0]), str(P["train_sizes"][-1])
    # >1 means the KAN's relative position IMPROVES with more data, i.e. the opposite of
    # the "KAN is more sample-efficient (wins most at small n)" claim.
    ratio_trend_lo_over_hi = ratio[n_lo] / ratio[n_hi]
    sweep_sec = time.time() - t_sweep
    print(f"[sweep] total {sweep_sec:.0f}s; kan wins {kan_wins}/{len(P['train_sizes'])}", flush=True)

    # =============================================================== 2. symbolic extraction
    from kan import KAN

    t_sym = time.time()
    Xs, ys_ = make_data(P["sym_n_train"], 1000 + P["data_seeds"][0], ranges)  # raw, un-standardized
    ds_sym = {
        "train_input": Xs,
        "train_label": ys_,
        "test_input": Xte[:200],
        "test_label": yte[:200],
    }

    best = None
    for isd in P["sym_init_seeds"]:
        set_seeds(seed)
        with quiet():
            m = KAN(
                width=list(P["sym_width"]), grid=P["sym_grid"], k=P["kan_k"],
                seed=isd, device="cpu", ckpt_path=str(CKPT),
            )
            m.fit(ds_sym, opt="LBFGS", steps=P["sym_steps_stage1"], lamb=P["sym_lamb"],
                  stop_grid_update_step=20)
            m = m.refine(P["sym_refine_grid"])
            r = m.fit(ds_sym, opt="LBFGS", steps=P["sym_steps_stage2"], lamb=0.0,
                      stop_grid_update_step=10)
            tr = float(r["train_loss"][-1])
        # model selection on TRAIN loss only (no test peeking)
        if best is None or tr < best[0]:
            best = (tr, m, isd)
    sym_train_loss, kmodel, sym_seed = best
    with torch.no_grad():
        pred_spline = kmodel(Xte)
    rmse_spline = float(((pred_spline - yte) ** 2).mean().sqrt())
    print(f"[sym] spline KAN seed={sym_seed} test RMSE {rmse_spline:.5f}", flush=True)

    # --- 2a. per-edge learned univariate functions vs candidate primitives
    edge_report = {}
    edge_curves = {}
    try:
        with torch.no_grad(), quiet():
            kmodel(Xs)
            pre0 = kmodel.spline_preacts[0].detach().numpy()   # (B, out, in)
            post0 = kmodel.spline_postacts[0].detach().numpy()
            pre1 = kmodel.spline_preacts[1].detach().numpy()
            post1 = kmodel.spline_postacts[1].detach().numpy()
        for i, vname in enumerate(var_names):
            x = pre0[:, 0, i]
            y = post0[:, 0, i]
            o = np.argsort(x)
            edge_curves[vname] = (x[o].tolist(), y[o].tolist())
            ranked = best_primitive(x, y)
            edge_report[vname] = {
                "best_primitive": ranked[0][0],
                "best_r2": round(ranked[0][1], 4),
                "r2_vs_log": round(dict(ranked).get("log(x)", float("nan")), 4),
                "amplitude": round(float(y.max() - y.min()), 4),
                "top3": [(k_, round(v_, 4)) for k_, v_ in ranked[:3]],
            }
        xh, yh = pre1[:, 0, 0], post1[:, 0, 0]
        o = np.argsort(xh)
        edge_curves["hidden->out"] = (xh[o].tolist(), yh[o].tolist())
        ranked = best_primitive(xh, yh)
        exp_hits = [(k_, v_) for k_, v_ in ranked if k_.startswith("exp(")]
        edge_report["hidden->out"] = {
            "best_primitive": ranked[0][0],
            "best_r2": round(ranked[0][1], 4),
            "best_exp_family": exp_hits[0][0] if exp_hits else None,
            "r2_vs_exp": round(exp_hits[0][1], 4) if exp_hits else None,
            "top3": [(k_, round(v_, 4)) for k_, v_ in ranked[:3]],
        }
    except Exception as e:  # pragma: no cover
        edge_report = {"error": repr(e)}

    # --- 2b. pykan auto_symbolic -> formula
    sym_info = {}
    pred_sym = None
    try:
        with quiet():
            kmodel(Xs)
            kmodel.auto_symbolic(lib=P["sym_lib"])
            kmodel.fit(ds_sym, opt="LBFGS", steps=P["sym_refit_steps"], lamb=0.0,
                       update_grid=False)
            from kan.utils import ex_round

            formula = kmodel.symbolic_formula()[0][0]
            formula_r = ex_round(formula, 4)
        with torch.no_grad():
            pred_sym = kmodel(Xte)
        sym_info["formula"] = str(formula_r)
        sym_info["rmse_symbolic"] = float(((pred_sym - yte) ** 2).mean().sqrt())
        sym_info["nrmse_symbolic"] = sym_info["rmse_symbolic"] / y_test_std
        exps, r2, frac_pos = loglog_exponents(Xte, pred_sym)
        sym_info["recovered_exponents"] = None if exps is None else [round(e, 3) for e in exps]
        sym_info["loglog_r2"] = None if r2 is None else round(r2, 5)
        sym_info["frac_positive_pred"] = round(frac_pos, 3)
        if exps is not None:
            sym_info["max_abs_exponent_error"] = round(
                max(abs(a - b) for a, b in zip(exps, EQ["true_log_exponents"])), 3
            )
    except Exception as e:
        sym_info["error"] = repr(e)

    # exponents of the pre-symbolic spline model (structure test independent of the library)
    exps_sp, r2_sp, fp_sp = loglog_exponents(Xte, pred_spline)
    spline_struct = {
        "recovered_exponents": None if exps_sp is None else [round(e, 3) for e in exps_sp],
        "loglog_r2": None if r2_sp is None else round(r2_sp, 5),
        "frac_positive_pred": round(fp_sp, 3),
    }
    if exps_sp is not None:
        spline_struct["max_abs_exponent_error"] = round(
            max(abs(a - b) for a, b in zip(exps_sp, EQ["true_log_exponents"])), 3
        )

    # verdict (thresholds declared in experiment.yaml)
    V = cfg["verdict_thresholds"]

    def verdict():
        if "error" in sym_info:
            return "not recovered (symbolic stage errored)"
        e = sym_info.get("max_abs_exponent_error")
        nr = sym_info.get("nrmse_symbolic", 9e9)
        r2 = sym_info.get("loglog_r2") or 0.0
        if e is None:
            return "not recovered"
        if e <= V["exact_max_exp_err"] and r2 >= V["exact_min_loglog_r2"] and nr <= V["exact_max_nrmse"]:
            return "exact (monomial recovered up to a constant factor)"
        if e <= V["approx_max_exp_err"] and nr <= V["approx_max_nrmse"]:
            return "approximate (correct power-law exponents, non-canonical primitives)"
        return "not recovered"

    sym_info["verdict"] = verdict()
    sym_info["verdict_thresholds"] = V
    sym_sec = time.time() - t_sym
    print(f"[sym] {sym_sec:.0f}s verdict={sym_info['verdict']}", flush=True)

    # =============================================================== 3. chart
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ns = P["train_sizes"]

    ax = axes[0, 0]
    style = {
        "kan": (f"KAN {kan_width_label} g={P['kan_grid']} ({n_params.get('kan')}p)", "tab:red", "o", "-", 2.2),
        "mlp_deep_lbfgs": ("MLP 4-11-11-1 LBFGS (199p)", "tab:blue", "s", "-", 1.4),
        "mlp_wide_lbfgs": ("MLP 4-33-1 LBFGS (199p)", "tab:green", "^", "-", 1.4),
        "mlp_deep_adam": ("MLP 4-11-11-1 Adam", "tab:blue", "s", "--", 1.0),
        "mlp_wide_adam": ("MLP 4-33-1 Adam", "tab:green", "^", "--", 1.0),
    }
    for a in arms:
        lab, c, mk, ls, lw = style[a]
        vals = [mean[a][str(n)] for n in ns]
        ax.plot(ns, vals, ls, color=c, marker=mk, label=lab, lw=lw, ms=5)
        lo = [min(sweep[a][str(n)]) for n in ns]
        hi = [max(sweep[a][str(n)]) for n in ns]
        ax.fill_between(ns, lo, hi, color=c, alpha=0.10, lw=0)
    ax.axhline(y_test_std, color="gray", ls=":", lw=1)
    ax.text(ns[-1], y_test_std * 1.08, "predict-the-mean baseline", color="gray",
            fontsize=8, ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training-set size n")
    ax.set_ylabel("test RMSE (original units)")
    ax.set_title("A. Sample efficiency, Feynman I.12.2\n(mean of 3 data seeds, band = min-max)")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3, which="both")

    ax = axes[0, 1]
    ax.plot(ns, [ratio[str(n)] for n in ns], "-o", color="tab:red",
            label="KAN / best-of-4 MLP (oracle)")
    ax.plot(ns, [ratio_vs_single[str(n)] for n in ns], "--s", color="tab:purple",
            label="KAN / MLP 4-11-11-1 LBFGS")
    ax.axhline(1.0, color="k", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ylo, yhi = ax.get_ylim()
    ax.fill_between(ns, ylo, 1.0, color="tab:red", alpha=0.07)
    ax.set_ylim(ylo, yhi)
    ax.text(ns[0] * 1.05, 1.0 / 1.3, "KAN better", fontsize=8, color="tab:red")
    ax.text(ns[0] * 1.05, 1.3, "MLP better", fontsize=8)
    ax.set_xlabel("training-set size n")
    ax.set_ylabel("test-RMSE ratio")
    ax.set_title("B. Is the KAN more sample-efficient?")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1, 0]
    if edge_curves:
        cols = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
        for i, vname in enumerate(var_names):
            if vname not in edge_curves:
                continue
            x, y = edge_curves[vname]
            x = np.array(x)
            y = np.array(y)
            ax.plot(x, y, color=cols[i], lw=1.6,
                    label=f"$\\phi_{{{vname}}}$  best={edge_report[vname]['best_primitive']}"
                          f" $R^2$={edge_report[vname]['best_r2']:.3f}"
                          f" (log: {edge_report[vname]['r2_vs_log']:.3f})")
            A = np.stack([np.log(x), np.ones_like(x)], 1)
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            ax.plot(x, A @ coef, color=cols[i], lw=1.0, ls=":")
        ax.set_xlabel("input value")
        ax.set_ylabel("edge output $\\phi(x)$")
        ax.set_title(f"C. Layer-0 edge functions of the {sym_width_label} KAN\n"
                     "(dotted = best-fit $a\\log x+b$)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.axis("off")
    tv = EQ["true_log_exponents"]
    lines = [
        f"Feynman {EQ['feynman_id']}:  {EQ['formula']}",
        f"vars ~ U[1,5]; std(y_test) = {y_test_std:.4f}",
        "",
        f"KAN params {n_params.get('kan')} | MLP params "
        f"{n_params.get('mlp_deep')}/{n_params.get('mlp_wide')}",
        f"KAN beats best MLP at {kan_wins}/{len(ns)} training sizes"
        + (f" (first at n={n_first_win})" if n_first_win else ""),
        "RMSE ratio (KAN/bestMLP):",
        "  " + ", ".join(f"n={n}:{ratio[str(n)]:.2f}" for n in ns),
        f"ratio(n={n_lo})/ratio(n={n_hi}) = {ratio_trend_lo_over_hi:.2f}"
        + ("  -> KAN gains WITH data" if ratio_trend_lo_over_hi > 1 else "  -> KAN gains at small n"),
        "",
        f"--- symbolic extraction ({sym_width_label} KAN, raw data) ---",
        f"spline test RMSE     {rmse_spline:.5f} (NRMSE {rmse_spline / y_test_std:.4f})",
        "symbolic test RMSE   "
        + (f"{sym_info['rmse_symbolic']:.5f} (NRMSE {sym_info['nrmse_symbolic']:.4f})"
           if "rmse_symbolic" in sym_info else "n/a"),
        f"true     log-exponents {tv}",
        f"spline   log-exponents {spline_struct['recovered_exponents']} R2={spline_struct['loglog_r2']}",
        f"symbolic log-exponents {sym_info.get('recovered_exponents')} R2={sym_info.get('loglog_r2')}",
        "",
        f"VERDICT: {sym_info['verdict']}",
        "",
        "formula:",
    ]
    f = sym_info.get("formula", sym_info.get("error", "n/a"))
    for j in range(0, min(len(f), 400), 60):
        lines.append("  " + f[j:j + 60])
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=8.0, family="monospace")
    ax.set_title("D. Recovered formula and fidelity", loc="left")

    fig.suptitle(
        "KAN vs iso-param MLP on Feynman I.12.2 - sample efficiency and symbolic recovery",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(HERE / "chart.png", dpi=140)
    plt.close(fig)

    # =============================================================== 4. results.json
    metrics = {
        "equation": EQ["formula"],
        "feynman_id": EQ["feynman_id"],
        "n_test": P["n_test"],
        "y_test_std": round(y_test_std, 6),
        "n_params": n_params,
        "train_sizes": ns,
        "data_seeds": P["data_seeds"],
        "test_rmse_mean": {a: {n: round(v, 6) for n, v in mean[a].items()} for a in arms},
        "test_rmse_all_seeds": {
            a: {n: [round(x, 6) for x in v] for n, v in sweep[a].items()} for a in arms
        },
        "best_mlp_rmse_mean": {n: round(v, 6) for n, v in best_mlp.items()},
        "best_mlp_arm": best_mlp_arm,
        "rmse_ratio_kan_over_best_mlp": {n: round(v, 4) for n, v in ratio.items()},
        "rmse_ratio_kan_over_mlp_deep_lbfgs": {n: round(v, 4) for n, v in ratio_vs_single.items()},
        "kan_wins_vs_best_mlp": kan_wins,
        "kan_wins_at_n": wins_at,
        "n_train_sizes": len(ns),
        "smallest_n_where_kan_wins": n_first_win,
        "ratio_trend_smallest_over_largest_n": round(ratio_trend_lo_over_hi, 4),
        "kan_advantage_grows_with_data": bool(ratio_trend_lo_over_hi > 1.0),
        "sample_efficiency_claim_supported": bool(kan_wins == len(ns)),
        "kan_better_at_smallest_n": bool(ratio[n_lo] < 1.0),
        "symbolic": {
            "arch": P["sym_width"],
            "chosen_init_seed": sym_seed,
            "train_mse_selected": round(sym_train_loss, 8),
            "spline_test_rmse": round(rmse_spline, 6),
            "spline_test_nrmse": round(rmse_spline / y_test_std, 5),
            "spline_structure": spline_struct,
            "edge_functions": edge_report,
            **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in sym_info.items()},
        },
        "sweep_sec": round(sweep_sec, 1),
        "symbolic_sec": round(sym_sec, 1),
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
    with open(HERE / "results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    shutil.rmtree(CKPT, ignore_errors=True)
    print(json.dumps({k: v for k, v in metrics.items() if k != "test_rmse_all_seeds"}, indent=2))


if __name__ == "__main__":
    main()
