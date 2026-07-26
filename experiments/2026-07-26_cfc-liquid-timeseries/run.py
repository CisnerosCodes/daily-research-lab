"""CfC (closed-form continuous-time / "liquid") cell vs dt-blind, dt-as-input and decay GRUs
under UNSEEN sampling irregularity.

Deterministic, CPU-only, single-threaded.  Writes results.json + chart.png.

Usage:
    python run.py                 # full experiment
    python run.py --chart-only    # redraw chart.png from an existing results.json
    python run.py --probe         # tiny timing probe, writes nothing
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json, math, random, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------------------
# boilerplate
# --------------------------------------------------------------------------------------
def set_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True)
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


def stable_int(s: str) -> int:
    """Deterministic replacement for hash() (PYTHONHASHSEED is randomised)."""
    v = 0
    for ch in s:
        v = (v * 131 + ord(ch)) % 99991
    return v


# --------------------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------------------
def mackey_glass_dense(span, tau, dt, beta, gamma, power, x0, discard):
    """RK4 integration of dx/dt = beta*x(t-tau)/(1+x(t-tau)^p) - gamma*x(t).

    Returns the DENSE trajectory on a grid of spacing `dt` (no subsampling), covering
    `span` time units after discarding `discard` time units of transient.
    Deterministic: constant history x(t)=x0 on [-tau, 0].  Same integrator as
    experiments/2026-07-25_esn-reservoir/run.py, kept dense so we can sample it at
    arbitrary continuous times.
    """
    total = int(round((span + discard) / dt))
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

    series = buf[d + 1:]
    n_discard = int(round(discard / dt))
    return series[n_discard: n_discard + int(round(span / dt))].copy()


def sample_intervals(n, T, cv, mu, gen):
    """T-1 positive intervals per row, mean EXACTLY mu, coefficient of variation EXACTLY cv.

    Gamma(shape=k, scale=mu/k) with k = 1/cv^2  =>  mean mu, var mu^2/k = (cv*mu)^2.
    cv == 0 gives the perfectly regular grid.  Nothing is clipped (beyond a 1e-4 floor),
    so the heavy tail of the high-cv regimes is preserved.
    """
    if cv <= 0.0:
        return torch.full((n, T - 1), float(mu))
    k = 1.0 / (cv * cv)
    seed = int(torch.randint(0, 2 ** 31 - 1, (1,), generator=gen).item())
    rng = np.random.default_rng(seed)
    dt = rng.gamma(shape=k, scale=mu / k, size=(n, T - 1))
    dt = np.maximum(dt, 1e-4)
    return torch.tensor(dt, dtype=torch.float32)


def make_sine_params(n, p, gen):
    K = p["sine_k"]
    a = torch.rand(n, K, generator=gen) * (p["sine_a_max"] - p["sine_a_min"]) + p["sine_a_min"]
    f = torch.rand(n, K, generator=gen) * (p["sine_f_max"] - p["sine_f_min"]) + p["sine_f_min"]
    ph = torch.rand(n, K, generator=gen) * (2 * math.pi)
    return a, f, ph


def sine_at(times, a, f, ph):
    """times [n,T] -> x [n,T]; per-sequence variance normalised to ~1."""
    x = (a[:, None, :] * torch.sin(2 * math.pi * f[:, None, :] * times[:, :, None]
                                   + ph[:, None, :])).sum(-1)
    norm = torch.sqrt((a ** 2).sum(-1) / 2.0)[:, None]
    return x / norm


def mg_at(times, series_t, grid_dt):
    """Linear interpolation of the dense MG grid at arbitrary continuous times."""
    pos = times / grid_dt
    i0 = pos.floor().long().clamp(0, series_t.numel() - 2)
    frac = (pos - i0.to(pos.dtype)).clamp(0, 1)
    return series_t[i0] * (1 - frac) + series_t[i0 + 1] * frac


class Batcher:
    """Produces (x, dt) batches.  Signal identity and sampling grid are drawn from
    SEPARATE generators, so an eval set can hold the underlying continuous-time signals
    fixed while only the sampling irregularity changes."""

    def __init__(self, dataset, p, mg_series=None):
        self.ds = dataset
        self.p = p
        self.mg = mg_series
        self.T = p["seq_len"]
        self.mu = p["mean_interval"]

    def signals(self, n, gen):
        """Draw the underlying continuous-time functions (not the sampling grid)."""
        if self.ds == "sine":
            return make_sine_params(n, self.p, gen)
        hi = self.p["mg_span"] - self.p["mg_max_window"]
        return (torch.rand(n, generator=gen) * hi,)

    def sample(self, sig, cv, gen):
        n = sig[0].shape[0]
        dt = sample_intervals(n, self.T, cv, self.mu, gen)
        t = torch.cat([torch.zeros(n, 1), dt.cumsum(-1)], dim=-1)  # [n,T], t_0 = 0
        if self.ds == "sine":
            a, f, ph = sig
            x = sine_at(t, a, f, ph)
        else:
            t0 = sig[0][:, None]
            tt = (t0 + t).clamp(0.0, self.p["mg_span"] - 2 * self.p["mg_dt"])
            x = mg_at(tt, self.mg, self.p["mg_dt"])
        return x, dt

    def batch(self, n, cv, gen):
        return self.sample(self.signals(n, gen), cv, gen)


# --------------------------------------------------------------------------------------
# Cells / arms
# --------------------------------------------------------------------------------------
class CfCCell(nn.Module):
    """Closed-form continuous-time cell, Hasani et al. arXiv:2106.13898, Eq. (10):

        x(t) = sigma(-f(x,I;theta_f) * t) (.) g(x,I;theta_g)
               + (1 - sigma(-[f(x,I;theta_f)] * t)) (.) h(x,I;theta_h)

    Implemented in the reference parametrisation of the official CfCCell
    (raminmh/CfC, ncps): a shared backbone over z = [input ; state] feeding four linear
    heads -- ff1 (= h), ff2 (= g), time_a (= -f) and time_b -- with

        c      = sigmoid(time_a(z) * dt + time_b(z))
        state' = (1 - c) (.) ff1(z) + c (.) ff2(z)

    `time_b` is an elapsed-time-independent bias present in the reference implementation
    but not in Eq. (10) as printed; without it the mixing coefficient is pinned to 0.5 at
    dt = 0.  Everything else is Eq. (10) verbatim.  The mixing coefficient is the ONLY
    place the elapsed time enters.
    """

    def __init__(self, in_dim, hidden, backbone, fixed_dt=None):
        super().__init__()
        self.hidden = hidden
        self.backbone = nn.Linear(in_dim + hidden, backbone)
        self.ff1 = nn.Linear(backbone, hidden)
        self.ff2 = nn.Linear(backbone, hidden)
        self.time_a = nn.Linear(backbone, hidden)
        self.time_b = nn.Linear(backbone, hidden)
        self.fixed_dt = fixed_dt  # control arm: gate sees a constant instead of the true dt

    def forward(self, u, h, dt):
        z = torch.tanh(self.backbone(torch.cat([u, h], dim=-1)))
        g = torch.tanh(self.ff2(z))
        q = torch.tanh(self.ff1(z))
        ts = torch.full_like(dt, self.fixed_dt) if self.fixed_dt is not None else dt
        c = torch.sigmoid(self.time_a(z) * ts + self.time_b(z))
        return c * g + (1 - c) * q


class Arm(nn.Module):
    """One sequence model.  All arms share the same interface, readout and recurrence
    schedule; they differ ONLY in how (or whether) the elapsed time dt reaches the state.

    Recurrence (identical for every arm):
        h_{i+1}    = Cell(x_i, h_i, dt_i),   dt_i = t_{i+1} - t_i
        yhat_{i+1} = W h_{i+1} + b           ->  target x_{i+1}

    dt_i is simultaneously the elapsed time of the state update and the forecast horizon,
    so the dt-blind arm genuinely does not know how far ahead it is predicting -- that is
    the property under test.
    """

    def __init__(self, kind, hidden, backbone=None, mean_interval=1.0):
        super().__init__()
        self.kind = kind
        self.hidden = hidden
        if kind in ("cfc", "cfc_fixed_dt"):
            self.cell = CfCCell(1, hidden, backbone,
                                fixed_dt=(mean_interval if kind == "cfc_fixed_dt" else None))
        elif kind == "decay_gru":
            self.cell = nn.GRUCell(1, hidden)
            # GRU-D-style exponential state decay by the elapsed time, per unit.
            self.decay_w = nn.Parameter(torch.full((hidden,), -1.0))
        else:
            # gru_blind / gru_dt have no state-dependent time term, so the whole sequence
            # can go through the fused nn.GRU.  Identical recurrence and identical
            # parameter shapes as nn.GRUCell -- purely a speed choice.
            self.cell = nn.GRU(2 if kind == "gru_dt" else 1, hidden, batch_first=True)
        self.readout = nn.Linear(hidden, 1)

    def forward(self, x, dt):
        n, T = x.shape
        if self.kind in ("gru_blind", "gru_dt"):
            u = x[:, :-1].unsqueeze(-1)
            if self.kind == "gru_dt":
                u = torch.cat([u, dt.unsqueeze(-1)], dim=-1)
            hs, _ = self.cell(u)
            return self.readout(hs).squeeze(-1)  # [n, T-1]
        h = x.new_zeros(n, self.hidden)
        out = []
        rate = nn.functional.softplus(self.decay_w) if self.kind == "decay_gru" else None
        for i in range(T - 1):
            u = x[:, i: i + 1]
            d = dt[:, i: i + 1]
            if self.kind == "decay_gru":
                h = self.cell(u, h * torch.exp(-d * rate))
            else:  # cfc / cfc_fixed_dt
                h = self.cell(u, h, d)
            out.append(self.readout(h))
        return torch.cat(out, dim=-1)  # [n, T-1], predicting x[:,1:]


def n_params_for(kind, hidden):
    if kind in ("cfc", "cfc_fixed_dt"):
        b = hidden
        return (1 + hidden) * b + b + 4 * (b * hidden + hidden) + hidden + 1
    in_dim = 2 if kind == "gru_dt" else 1
    n = 3 * hidden * (in_dim + hidden) + 6 * hidden + hidden + 1
    if kind == "decay_gru":
        n += hidden
    return n


def fit_hidden(kind, target):
    best, best_h = None, None
    for h in range(8, 400):
        d = abs(n_params_for(kind, h) - target)
        if best is None or d < best:
            best, best_h = d, h
    return best_h


def build_arm(kind, p):
    h = fit_hidden(kind, p["target_params"])
    m = Arm(kind, h, backbone=h, mean_interval=p["mean_interval"])
    n = sum(q.numel() for q in m.parameters())
    assert n == n_params_for(kind, h), (kind, h, n, n_params_for(kind, h))
    return m, h, n


# --------------------------------------------------------------------------------------
# Train / eval
# --------------------------------------------------------------------------------------
def evaluate(model, batcher, eval_sig, cvs, p, gen_seed):
    model.eval()
    burn = p["burn_in"]
    out = {}
    with torch.no_grad():
        for cv in cvs:
            g = torch.Generator().manual_seed(gen_seed + int(round(cv * 10000)))
            x, dt = batcher.sample(eval_sig, cv, g)
            pred = model(x, dt)
            tgt = x[:, 1:]
            out[f"{cv:.4f}"] = float(((pred[:, burn:] - tgt[:, burn:]) ** 2).mean())
    model.train()
    return out


def baselines(batcher, eval_sig, cvs, p, gen_seed):
    """Reference scales on the IDENTICAL eval batches: persistence and target variance."""
    burn = p["burn_in"]
    pers, var = {}, {}
    for cv in cvs:
        g = torch.Generator().manual_seed(gen_seed + int(round(cv * 10000)))
        x, _ = batcher.sample(eval_sig, cv, g)
        tgt, prev = x[:, 1:], x[:, :-1]
        pers[f"{cv:.4f}"] = float(((prev[:, burn:] - tgt[:, burn:]) ** 2).mean())
        var[f"{cv:.4f}"] = float((tgt[:, burn:] ** 2).mean())
    return pers, var


def train_one(kind, dataset, seed, lr, steps, p, batcher, eval_sig, cvs, verbose=True):
    torch.manual_seed(1000 * seed + stable_int(kind))
    model, hid, npar = build_arm(kind, p)
    decay, nodecay = [], []
    for _, q in model.named_parameters():
        (decay if q.ndim >= 2 else nodecay).append(q)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": p["weight_decay"]},
         {"params": nodecay, "weight_decay": 0.0}], lr=lr)
    warm = max(1, int(p["warmup_frac"] * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    # the data generator depends on (dataset, seed) ONLY -> every arm sees identical batches
    gdata = torch.Generator().manual_seed(50000 + 137 * seed + (0 if dataset == "sine" else 1))
    burn = p["burn_in"]
    t0 = time.time()
    losses = []
    for s in range(steps):
        x, dt = batcher.batch(p["batch_size"], p["train_cv"], gdata)
        pred = model(x, dt)
        loss = ((pred[:, burn:] - x[:, 1:][:, burn:]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), p["grad_clip"])
        opt.step()
        sched.step()
        if s % 100 == 0 or s == steps - 1:
            losses.append([s, round(float(loss.detach()), 5)])
    mse = evaluate(model, batcher, eval_sig, cvs, p, gen_seed=900000)
    dur = time.time() - t0
    if verbose:
        tkey = "%.4f" % p["train_cv"]
        at_train = mse.get(tkey, float("nan"))
        print("  [%s %s seed%d lr%g] h=%d params=%d train_loss=%.4f mse@train_cv=%.4f (%.0fs)"
              % (dataset, kind, seed, lr, hid, npar, losses[-1][1], at_train, dur), flush=True)
    return {"kind": kind, "dataset": dataset, "seed": seed, "lr": lr, "hidden": hid,
            "n_params": npar, "steps": steps, "mse": mse, "loss_curve": losses,
            "train_seconds": round(dur, 1)}


# --------------------------------------------------------------------------------------
# Chart
# --------------------------------------------------------------------------------------
COLORS = {"cfc": "#c0392b", "gru_dt": "#2471a3", "decay_gru": "#1e8449",
          "gru_blind": "#7f8c8d", "cfc_fixed_dt": "#e67e22"}
LABEL = {"cfc": "CfC (closed-form, dt in gate)", "gru_dt": "GRU + dt as input",
         "decay_gru": "decay-GRU (GRU-D style)", "gru_blind": "GRU, dt-blind",
         "cfc_fixed_dt": "CfC control: gate dt frozen at mu"}


def make_chart(res, p):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cvs = p["eval_cvs"]
    arms = p["arms"]
    agg = res["metrics"]["mse_by_arm"]
    base = res["metrics"]["baselines"]
    train_cv = p["train_cv"]
    dsets = p["datasets"]
    nice = {"sine": "sine mixtures", "mg": "Mackey-Glass tau=17"}

    fig, axes = plt.subplots(2, len(dsets), figsize=(6.3 * len(dsets), 8.4), squeeze=False)
    for col, ds in enumerate(dsets):
        ax = axes[0][col]
        for k in arms:
            ys = [agg[ds][k]["%.4f" % cv]["mean"] for cv in cvs]
            lo = [agg[ds][k]["%.4f" % cv]["min"] for cv in cvs]
            hi = [agg[ds][k]["%.4f" % cv]["max"] for cv in cvs]
            ax.plot(cvs, ys, "-o", ms=4, color=COLORS[k], label=LABEL[k],
                    lw=2.2 if k in ("cfc", "gru_dt") else 1.4,
                    ls="--" if k == "cfc_fixed_dt" else "-")
            ax.fill_between(cvs, lo, hi, color=COLORS[k], alpha=0.13, lw=0)
        ax.plot(cvs, [base[ds]["persistence"]["%.4f" % cv] for cv in cvs], ":",
                color="k", lw=1.2, label="persistence (predict x_i)")
        ax.axvline(train_cv, color="0.5", lw=1, ls="-.")
        ax.set_yscale("log")
        ax.set_xlabel("test interval coefficient of variation (mean interval fixed at mu=1)")
        ax.set_ylabel("next-value MSE")
        ax.set_title("%s  (%d seeds, band = min-max)" % (nice[ds], len(p["seeds"])), fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.annotate("trained here", xy=(train_cv, ax.get_ylim()[1]), fontsize=8,
                    color="0.35", ha="center", va="top")
        if col == 0:
            ax.legend(frameon=False, fontsize=8, loc="lower right")

    head = [0.0, 0.8485, 1.2]
    for col, ds in enumerate(dsets):
        ax = axes[1][col]
        w = 0.16
        for j, k in enumerate(arms):
            ref = agg[ds][k]["%.4f" % train_cv]["mean"]
            vals = [agg[ds][k]["%.4f" % cv]["mean"] / ref for cv in head]
            ax.bar([i + (j - 2) * w for i in range(len(head))], vals, w,
                   color=COLORS[k], label=LABEL[k])
            for i, v in enumerate(vals):
                ax.text(i + (j - 2) * w, v * 1.05, "%.2f" % v, ha="center", fontsize=7,
                        rotation=90)
        ax.axhline(1.0, color="0.4", lw=1)
        ax.set_xticks(range(len(head)))
        ax.set_xticklabels(["regular\n(cv=0)", "2x variance\n(cv=0.85)",
                            "4x variance\n(cv=1.2)"], fontsize=9)
        ax.set_yscale("log")
        ax.set_ylabel("MSE / MSE at trained irregularity")
        ax.set_title("%s: degradation under sampling shift (1.0 = no change)" % nice[ds],
                     fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)

    npar = res["metrics"]["n_params_by_arm"]
    fig.suptitle("Closed-form continuous-time (CfC) cell vs matched-parameter GRUs under "
                 "UNSEEN sampling irregularity  (matched params %d-%d, trained only at cv=%g)"
                 % (min(npar.values()), max(npar.values()), train_cv),
                 fontsize=12, y=1.005)
    fig.tight_layout()
    fig.savefig(HERE / "chart.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main():
    cfg = load_config()
    p = cfg["params"]
    seed = int(cfg.get("seed", 0))
    set_seeds(seed)
    t0 = time.time()

    if "--chart-only" in sys.argv:
        with open(HERE / "results.json") as f:
            res = json.load(f)
        make_chart(res, p)
        print("chart redrawn")
        return

    cvs = p["eval_cvs"]
    probe = "--probe" in sys.argv

    # ---------------- data ----------------
    mg_raw = mackey_glass_dense(p["mg_span"] + p["mg_max_window"] + 5, p["mg_tau"], p["mg_dt"],
                                p["mg_beta"], p["mg_gamma"], p["mg_power"], p["mg_x0"],
                                p["mg_discard"])
    mg_sq = np.tanh(mg_raw - 1.0)
    mg_norm = (mg_sq - mg_sq.mean()) / mg_sq.std()
    mg_t = torch.tensor(mg_norm, dtype=torch.float32)
    print("[data] MG dense n=%d raw[%.3f,%.3f] (%.1fs)"
          % (len(mg_raw), mg_raw.min(), mg_raw.max(), time.time() - t0), flush=True)

    batchers = {"sine": Batcher("sine", p), "mg": Batcher("mg", p, mg_series=mg_t)}
    eval_sigs, base = {}, {}
    for ds in p["datasets"]:
        g = torch.Generator().manual_seed(777001 + (0 if ds == "sine" else 1))
        eval_sigs[ds] = batchers[ds].signals(p["eval_n"], g)
        pers, var = baselines(batchers[ds], eval_sigs[ds], cvs, p, gen_seed=900000)
        base[ds] = {"persistence": pers, "target_variance": var}
    print("[data] baselines:", json.dumps(base, indent=None), flush=True)

    if probe:
        for k in p["arms"]:
            r = train_one(k, "sine", 0, p["lr"], 20, p, batchers["sine"], eval_sigs["sine"], cvs)
            print(k, r["hidden"], r["n_params"], "%.0f ms/step" % (r["train_seconds"] / 20 * 1000))
        return

    # ---------------- lr pre-sweep (fairness: every arm gets its own best lr) ----------
    print("[lr pre-sweep]", flush=True)
    sweep = {}
    for k in p["arms"]:
        sweep[k] = {}
        for lr in p["lr_sweep"]:
            r = train_one(k, "sine", 0, lr, p["lr_sweep_steps"], p, batchers["sine"],
                          eval_sigs["sine"], [p["train_cv"]])
            sweep[k][str(lr)] = r["mse"]["%.4f" % p["train_cv"]]
    best_lr = {k: float(min(sweep[k], key=lambda z: sweep[k][z])) for k in p["arms"]}
    print("[lr pre-sweep] best per arm:", best_lr, "(%.0fs)" % (time.time() - t0), flush=True)

    # ---------------- main runs ----------------
    runs = []
    for ds in p["datasets"]:
        for k in p["arms"]:
            for sd in p["seeds"]:
                runs.append(train_one(k, ds, sd, best_lr[k], p["steps"], p,
                                      batchers[ds], eval_sigs[ds], cvs))
    print("[runs] %d done (%.0fs)" % (len(runs), time.time() - t0), flush=True)

    # ---------------- aggregate ----------------
    agg = {}
    for ds in p["datasets"]:
        agg[ds] = {}
        for k in p["arms"]:
            agg[ds][k] = {}
            for cv in cvs:
                key = "%.4f" % cv
                vals = [r["mse"][key] for r in runs if r["dataset"] == ds and r["kind"] == k]
                agg[ds][k][key] = {"mean": float(np.mean(vals)), "min": float(np.min(vals)),
                                   "max": float(np.max(vals)), "per_seed": vals}

    tcv = "%.4f" % p["train_cv"]
    headline, degr = {}, {}
    for ds in p["datasets"]:
        headline[ds], degr[ds] = {}, {}
        for k in p["arms"]:
            headline[ds][k] = {
                "regular_cv0.0": round(agg[ds][k]["0.0000"]["mean"], 5),
                "trained_cv0.6": round(agg[ds][k][tcv]["mean"], 5),
                "var2x_cv0.8485": round(agg[ds][k]["0.8485"]["mean"], 5),
                "var4x_cv1.2": round(agg[ds][k]["1.2000"]["mean"], 5),
            }
            ref = agg[ds][k][tcv]["mean"]
            degr[ds][k] = {
                "regular_over_trained": round(agg[ds][k]["0.0000"]["mean"] / ref, 4),
                "var2x_over_trained": round(agg[ds][k]["0.8485"]["mean"] / ref, 4),
                "var4x_over_trained": round(agg[ds][k]["1.2000"]["mean"] / ref, 4),
            }

    seed_spread = {ds: {k: round(agg[ds][k][tcv]["max"] - agg[ds][k][tcv]["min"], 5)
                        for k in p["arms"]} for ds in p["datasets"]}
    cfc_vs_grudt = {ds: {reg: round(headline[ds]["cfc"][reg] - headline[ds]["gru_dt"][reg], 5)
                         for reg in headline[ds]["cfc"]} for ds in p["datasets"]}

    metrics = {
        "headline": "next-value MSE per arm at four sampling-irregularity regimes "
                    "(regular / trained / 2x interval variance / 4x interval variance) "
                    "at matched ~50k params; mean interval identical in every regime",
        "n_params_by_arm": {k: n_params_for(k, fit_hidden(k, p["target_params"]))
                            for k in p["arms"]},
        "hidden_by_arm": {k: fit_hidden(k, p["target_params"]) for k in p["arms"]},
        "n_runs": len(runs),
        "steps": p["steps"],
        "batch_size": p["batch_size"],
        "seq_len": p["seq_len"],
        "burn_in": p["burn_in"],
        "eval_n": p["eval_n"],
        "train_cv": p["train_cv"],
        "eval_cvs": cvs,
        "best_lr_by_arm": best_lr,
        "lr_pre_sweep_mse_at_train_cv": sweep,
        "mse_headline": headline,
        "degradation_ratio_vs_trained": degr,
        "cfc_minus_gru_dt_mse": cfc_vs_grudt,
        "seed_spread_at_train_cv": seed_spread,
        "mse_by_arm": agg,
        "baselines": base,
        "wall_clock_s": round(time.time() - t0, 1),
    }

    results = {
        "id": cfg.get("id", "unknown"),
        "git_commit": git_sha(),
        "seed": seed,
        "duration_sec": round(time.time() - t0, 2),
        "metrics": metrics,
        "runs": runs,
        "env": env_info(),
        "status": "done",
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    make_chart(results, p)
    print(json.dumps({k: results[k] for k in ("id", "duration_sec", "status")}, indent=2))
    print("headline (sine):", json.dumps(headline["sine"], indent=2))
    print("headline (mg):", json.dumps(headline["mg"], indent=2))


if __name__ == "__main__":
    main()
