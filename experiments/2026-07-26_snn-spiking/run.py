"""Rate vs latency coding in a hand-rolled surrogate-gradient SNN.

The time-step (T) vs accuracy frontier, against a matched-parameter non-spiking
ReLU MLP reference, with total spike counts as an energy proxy.

Deterministic, CPU-only, single-threaded. Writes results.json + chart.png.

Usage:  python run.py
"""
import json, os, random, subprocess, sys, time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# boilerplate
# --------------------------------------------------------------------------- #
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
    for mod in ("numpy", "torch", "sklearn", "matplotlib"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:
            pass
    return info


# --------------------------------------------------------------------------- #
# surrogate-gradient spike function
# --------------------------------------------------------------------------- #
class FastSigmoidSpike(torch.autograd.Function):
    """Heaviside forward; fast-sigmoid surrogate backward  1/(1+slope*|x|)^2."""

    @staticmethod
    def forward(ctx, x, slope):
        ctx.save_for_backward(x)
        ctx.slope = slope
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        sg = 1.0 / (1.0 + ctx.slope * x.abs()) ** 2
        return grad_out * sg, None


def spike_fn(x, slope):
    return FastSigmoidSpike.apply(x, slope)


# --------------------------------------------------------------------------- #
# encodings.  All return a float tensor of shape (T, B, 64) in [0,1].
# --------------------------------------------------------------------------- #
def encode(x01, T, coding, gen):
    """x01: (B, 64) float in [0,1] (pixel/16)."""
    B, D = x01.shape
    if coding == "constant":
        # analog "direct" input current, repeated every step (NOT a spike train)
        return x01.unsqueeze(0).expand(T, B, D).contiguous()
    if coding == "rate":
        p = x01.unsqueeze(0).expand(T, B, D)
        u = torch.rand(T, B, D, generator=gen)
        return (u < p).to(x01.dtype)
    if coding == "latency":
        # brighter pixel -> earlier single spike; zero pixels never spike
        t_idx = torch.clamp(((1.0 - x01) * T).floor().long(), 0, T - 1)   # (B, D)
        active = (x01 > 0)
        out = torch.zeros(T, B, D, dtype=x01.dtype)
        out.scatter_(0, t_idx.unsqueeze(0), active.to(x01.dtype).unsqueeze(0))
        return out
    raise ValueError(coding)


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
class SNN(nn.Module):
    """2 hidden layers of LIF neurons, feed-forward (no lateral/recurrent weights).

    Because the network is strictly feed-forward, every affine layer can be applied
    to the whole (T, B, ...) tensor in ONE matmul; only the cheap elementwise LIF
    state update needs a python loop over T.  This is what makes the sweep fit on
    one CPU thread.
    """

    def __init__(self, cfg, readout="mean_current"):
        super().__init__()
        p = cfg["params"]
        self.fc1 = nn.Linear(p["n_in"], p["n_hidden"])
        self.fc2 = nn.Linear(p["n_hidden"], p["n_hidden"])
        self.fc3 = nn.Linear(p["n_hidden"], p["n_classes"])
        self.beta = float(p["beta"])
        self.thr = float(p["threshold"])
        self.slope = float(p["surrogate_slope"])
        self.readout = readout
        self.readout_beta = float(p["readout_leaky_beta"])

    def _lif(self, cur):
        """cur: (T, B, H) input current -> spikes (T, B, H)."""
        T = cur.shape[0]
        u = torch.zeros_like(cur[0])
        s = torch.zeros_like(cur[0])
        out = []
        for t in range(T):
            u = self.beta * u + cur[t] - self.thr * s.detach()
            s = spike_fn(u - self.thr, self.slope)
            out.append(s)
        return torch.stack(out, 0)

    def forward(self, xin):
        """xin: (T, B, 64) -> logits (B, C), spike counts per layer (scalars)."""
        s1 = self._lif(self.fc1(xin))
        s2 = self._lif(self.fc2(s1))
        out_cur = self.fc3(s2)                       # (T, B, C)
        if self.readout == "mean_current":
            logits = out_cur.mean(0)
        elif self.readout == "leaky":
            v = torch.zeros_like(out_cur[0])
            for t in range(out_cur.shape[0]):
                v = self.readout_beta * v + out_cur[t]
            logits = v
        else:
            raise ValueError(self.readout)
        counts = (s1.detach().sum().item(), s2.detach().sum().item())
        return logits, counts


class MLP(nn.Module):
    """Non-spiking reference with EXACTLY the same parameter shapes."""

    def __init__(self, cfg):
        super().__init__()
        p = cfg["params"]
        self.net = nn.Sequential(
            nn.Linear(p["n_in"], p["n_hidden"]), nn.ReLU(),
            nn.Linear(p["n_hidden"], p["n_hidden"]), nn.ReLU(),
            nn.Linear(p["n_hidden"], p["n_classes"]),
        )

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def get_data(cfg):
    from sklearn.datasets import load_digits
    p = cfg["params"]
    d = load_digits()
    X = torch.tensor(d.data, dtype=torch.float32) / 16.0     # (1797, 64) in [0,1]
    y = torch.tensor(d.target, dtype=torch.long)
    n = X.shape[0]
    # FIXED stratified split, identical for every arm and every seed
    rng = np.random.RandomState(p["split_seed"])
    tr, va, te = [], [], []
    for c in range(p["n_classes"]):
        idx = np.where(d.target == c)[0]
        rng.shuffle(idx)
        n_tr = int(round(p["split_train"] * len(idx)))
        n_va = int(round(p["split_val"] * len(idx)))
        tr += list(idx[:n_tr]); va += list(idx[n_tr:n_tr + n_va]); te += list(idx[n_tr + n_va:])
    tr, va, te = (torch.tensor(sorted(v)) for v in (tr, va, te))
    assert len(set(tr.tolist()) | set(va.tolist()) | set(te.tolist())) == n
    return X, y, tr, va, te


# --------------------------------------------------------------------------- #
# train / eval
# --------------------------------------------------------------------------- #
def evaluate_snn(model, X, y, idx, T, coding, cfg, eval_seed=999):
    p = cfg["params"]
    gen = torch.Generator().manual_seed(eval_seed)
    correct = 0
    s1_tot = s2_tot = in_tot = 0.0
    with torch.no_grad():
        for i in range(0, len(idx), p["eval_batch"]):
            b = idx[i:i + p["eval_batch"]]
            xin = encode(X[b], T, coding, gen)
            logits, (c1, c2) = model(xin)
            correct += (logits.argmax(1) == y[b]).sum().item()
            s1_tot += c1; s2_tot += c2
            in_tot += xin.sum().item() if coding != "constant" else 0.0
    n = len(idx)
    return {
        "acc": correct / n,
        "n_correct": correct,
        "n": n,
        "in_spikes_per_sample": in_tot / n,
        "h1_spikes_per_sample": s1_tot / n,
        "h2_spikes_per_sample": s2_tot / n,
    }


def evaluate_mlp(model, X, y, idx, cfg):
    p = cfg["params"]
    correct = 0
    with torch.no_grad():
        for i in range(0, len(idx), p["eval_batch"]):
            b = idx[i:i + p["eval_batch"]]
            correct += (model(X[b]).argmax(1) == y[b]).sum().item()
    return {"acc": correct / len(idx), "n_correct": correct, "n": len(idx)}


def train_snn(cfg, T, coding, lr, seed, steps, X, y, tr_idx, readout="mean_current"):
    p = cfg["params"]
    set_seeds(seed)
    model = SNN(cfg, readout=readout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=p["weight_decay"])
    lossf = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed + 7777)          # Bernoulli stream
    bgen = torch.Generator().manual_seed(seed + 4242)         # batch stream
    n = len(tr_idx)
    for _ in range(steps):
        b = tr_idx[torch.randint(0, n, (p["batch_size"],), generator=bgen)]
        xin = encode(X[b], T, coding, gen)
        logits, _ = model(xin)
        loss = lossf(logits, y[b])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), p["grad_clip"])
        opt.step()
        if not torch.isfinite(loss):
            return model, False
    return model, True


def train_mlp(cfg, lr, seed, steps, X, y, tr_idx):
    p = cfg["params"]
    set_seeds(seed)
    model = MLP(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=p["weight_decay"])
    lossf = nn.CrossEntropyLoss()
    bgen = torch.Generator().manual_seed(seed + 4242)
    n = len(tr_idx)
    for _ in range(steps):
        b = tr_idx[torch.randint(0, n, (p["batch_size"],), generator=bgen)]
        loss = lossf(model(X[b]), y[b])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), p["grad_clip"])
        opt.step()
    return model


# --------------------------------------------------------------------------- #
def main():
    cfg = load_config()
    p = cfg["params"]
    t0 = time.time()
    set_seeds(int(cfg.get("seed", 0)))

    X, y, tr, va, te = get_data(cfg)
    print(f"data: train {len(tr)}  val {len(va)}  test {len(te)}")

    n_params = sum(v.numel() for v in SNN(cfg).parameters())
    macs_mlp = p["n_in"] * p["n_hidden"] + p["n_hidden"] ** 2 + p["n_hidden"] * p["n_classes"]
    print(f"params (identical for SNN and MLP): {n_params}")

    Ts = p["time_steps"]
    codings = p["codings"]

    # ---------------- stage 1: lr sweep on the VAL split, seed 0 -------------
    # Budget shrink: the lr grid is swept only at lr_sweep_time_steps; every other T
    # inherits the lr of the largest swept T <= it (falling back to the smallest).
    sweep_Ts = p["lr_sweep_time_steps"]

    def sweep_T_for(T):
        le = [s for s in sweep_Ts if s <= T]
        return max(le) if le else min(sweep_Ts)

    print("\n== stage 1: lr sweep (val split, seed 0, reduced budget) ==")
    best_lr = {}
    lr_sweep = {}
    for coding in codings:
        for T in sweep_Ts:
            accs = {}
            for lr in p["lr_grid"]:
                m, ok = train_snn(cfg, T, coding, lr, 0, p["lr_sweep_steps"], X, y, tr)
                a = evaluate_snn(m, X, y, va, T, coding, cfg)["acc"] if ok else 0.0
                accs[str(lr)] = round(a, 4)
            bl = max(p["lr_grid"], key=lambda l: accs[str(l)])
            best_lr[(coding, T)] = bl
            lr_sweep[f"{coding}_T{T}"] = {"val_acc": accs, "best_lr": bl,
                                          "at_grid_edge": bl in (p["lr_grid"][0], p["lr_grid"][-1])}
            print(f"  {coding:9s} T={T:2d}  {accs}  -> lr={bl}  ({time.time()-t0:.0f}s)")
    for coding in codings:
        for T in Ts:
            best_lr[(coding, T)] = best_lr[(coding, sweep_T_for(T))]
    mlp_lr_acc = {}
    for lr in p["lr_grid"]:
        m = train_mlp(cfg, lr, 0, p["lr_sweep_steps"], X, y, tr)
        mlp_lr_acc[str(lr)] = round(evaluate_mlp(m, X, y, va, cfg)["acc"], 4)
    mlp_best_lr = max(p["lr_grid"], key=lambda l: mlp_lr_acc[str(l)])
    print(f"  mlp            {mlp_lr_acc} -> lr={mlp_best_lr}")

    # ---------------- stage 2: final runs, full budget, all seeds -----------
    print(f"\n== stage 2: final runs ({p['steps']} steps, seeds {p['seeds']}) ==")
    results = {}     # (arm, T) -> list of per-seed dicts
    arms = [(c, "mean_current") for c in codings] + [("latency", "leaky")]
    for coding, readout in arms:
        arm = coding if readout == "mean_current" else f"{coding}_leaky"
        seeds = p["seeds"] if readout == "mean_current" else p["seeds_control_arm"]
        for T in Ts:
            lr = best_lr[(coding, T)]
            per_seed = []
            for seed in seeds:
                m, ok = train_snn(cfg, T, coding, lr, seed, p["steps"], X, y, tr, readout=readout)
                r = evaluate_snn(m, X, y, te, T, coding, cfg)
                r["diverged"] = not ok
                r["lr"] = lr
                r["seed"] = seed
                per_seed.append(r)
            results[(arm, T)] = per_seed
            accs = [r["acc"] for r in per_seed]
            tot = np.mean([r["in_spikes_per_sample"] + r["h1_spikes_per_sample"]
                           + r["h2_spikes_per_sample"] for r in per_seed])
            print(f"  {arm:16s} T={T:2d} lr={lr:<6} acc={np.mean(accs):.4f} "
                  f"+-{np.std(accs):.4f}  spikes/sample={tot:.0f}  ({time.time()-t0:.0f}s)")

    mlp_accs = []
    for seed in p["seeds"]:
        m = train_mlp(cfg, mlp_best_lr, seed, p["steps"], X, y, tr)
        mlp_accs.append(evaluate_mlp(m, X, y, te, cfg)["acc"])
    mlp_mean, mlp_std = float(np.mean(mlp_accs)), float(np.std(mlp_accs))
    print(f"  {'relu_mlp':16s}         lr={mlp_best_lr:<6} acc={mlp_mean:.4f} +-{mlp_std:.4f}")

    # ---------------- metrics ------------------------------------------------
    arm_names = sorted({a for a, _ in results})
    M = {}
    for arm in arm_names:
        acc_m, acc_s, spk, spk_corr, synops, per_layer, per_seed_acc = {}, {}, {}, {}, {}, {}, {}
        for T in Ts:
            rs = results[(arm, T)]
            a = [r["acc"] for r in rs]
            acc_m[str(T)] = round(float(np.mean(a)), 4)
            acc_s[str(T)] = round(float(np.std(a)), 4)
            per_seed_acc[str(T)] = [round(v, 4) for v in a]
            i_s = float(np.mean([r["in_spikes_per_sample"] for r in rs]))
            h1 = float(np.mean([r["h1_spikes_per_sample"] for r in rs]))
            h2 = float(np.mean([r["h2_spikes_per_sample"] for r in rs]))
            tot = i_s + h1 + h2
            spk[str(T)] = round(tot, 1)
            per_layer[str(T)] = {"input": round(i_s, 1), "hidden1": round(h1, 1), "hidden2": round(h2, 1)}
            acc = float(np.mean(a))
            spk_corr[str(T)] = round(tot / acc, 1) if acc > 0 else None
            # accumulate-only synaptic operations: each presynaptic spike drives fanout accumulates
            so = i_s * p["n_hidden"] + h1 * p["n_hidden"] + h2 * p["n_classes"]
            if arm.startswith("constant"):
                # analog input layer is dense MACs, not event-driven accumulates
                so = h1 * p["n_hidden"] + h2 * p["n_classes"]
            synops[str(T)] = round(so, 0)
        M[arm] = {"n_seeds": len(results[(arm, Ts[0])]),
                  "test_acc_mean": acc_m, "test_acc_std": acc_s, "test_acc_per_seed": per_seed_acc,
                  "spikes_per_sample": spk, "spikes_per_correct": spk_corr,
                  "spikes_by_layer": per_layer, "synops_per_sample": synops,
                  "lr_by_T": {str(T): results[(arm, T)][0]["lr"] for T in Ts},
                  "diverged_runs": sum(r["diverged"] for T in Ts for r in results[(arm, T)])}

    def first_T_matching(arm, target):
        for T in Ts:
            if M[arm]["test_acc_mean"][str(T)] >= target:
                return T
        return None

    match_full = {a: first_T_matching(a, mlp_mean) for a in arm_names}
    match_1std = {a: first_T_matching(a, mlp_mean - mlp_std) for a in arm_names}

    head2head = {str(T): round(M["latency"]["test_acc_mean"][str(T)]
                               - M["rate"]["test_acc_mean"][str(T)], 4) for T in Ts}
    energy_ratio = {str(T): round(M["rate"]["spikes_per_correct"][str(T)]
                                  / M["latency"]["spikes_per_correct"][str(T)], 3) for T in Ts}
    best = {a: max(Ts, key=lambda T: M[a]["test_acc_mean"][str(T)]) for a in arm_names}

    metrics = {
        "headline": ("test accuracy vs simulation time steps T for rate vs latency (vs constant-current) "
                     "input coding in a matched-parameter LIF SNN, against a non-spiking ReLU MLP at "
                     "identical params/steps, plus spikes per correct classification"),
        "n_params": n_params,
        "n_params_mlp": sum(v.numel() for v in MLP(cfg).parameters()),
        "mlp_macs_per_inference": macs_mlp,
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "time_steps": Ts,
        "arms": arm_names,
        "seeds": p["seeds"],
        "steps": p["steps"],
        "batch_size": p["batch_size"],
        "seeds_control_arm": p["seeds_control_arm"],
        "lr_grid": p["lr_grid"],
        "lr_sweep_time_steps": sweep_Ts,
        "lr_sweep_steps": p["lr_sweep_steps"],
        "lr_sweep_val": lr_sweep,
        "mlp_lr_sweep_val_acc": mlp_lr_acc,
        "mlp_best_lr": mlp_best_lr,
        "relu_mlp_test_acc_mean": round(mlp_mean, 4),
        "relu_mlp_test_acc_std": round(mlp_std, 4),
        "relu_mlp_test_acc_per_seed": [round(a, 4) for a in mlp_accs],
        "by_arm": M,
        "latency_minus_rate_acc_by_T": head2head,
        "rate_over_latency_spikes_per_correct": energy_ratio,
        "first_T_matching_mlp_mean": match_full,
        "first_T_matching_mlp_mean_minus_1std": match_1std,
        "best_T_by_arm": best,
        "best_acc_by_arm": {a: M[a]["test_acc_mean"][str(best[a])] for a in arm_names},
        "wall_clock_s": round(time.time() - t0, 1),
    }

    # ---------------- chart --------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"rate": "#d62728", "latency": "#1f77b4",
              "constant": "#2ca02c", "latency_leaky": "#9467bd"}
    styles = {"latency_leaky": "--"}
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 9))

    a0 = ax[0, 0]
    for arm in arm_names:
        a0.errorbar(Ts, [M[arm]["test_acc_mean"][str(T)] for T in Ts],
                    yerr=[M[arm]["test_acc_std"][str(T)] for T in Ts],
                    marker="o", capsize=3, color=colors.get(arm), ls=styles.get(arm, "-"), label=arm)
    a0.axhline(mlp_mean, color="k", ls=":", lw=2, label=f"ReLU MLP ({mlp_mean:.3f})")
    a0.fill_between([min(Ts), max(Ts)], mlp_mean - mlp_std, mlp_mean + mlp_std, color="k", alpha=0.08)
    a0.set_xscale("log", base=2); a0.set_xticks(Ts); a0.set_xticklabels(Ts)
    a0.set_xlabel("simulation time steps T"); a0.set_ylabel("test accuracy")
    a0.set_title("(a) accuracy vs T  (3 seeds, matched params & steps)")
    a0.legend(fontsize=8); a0.grid(alpha=0.3)

    a1 = ax[0, 1]
    for arm in arm_names:
        a1.plot(Ts, [M[arm]["spikes_per_correct"][str(T)] for T in Ts], marker="o",
                color=colors.get(arm), ls=styles.get(arm, "-"), label=arm)
    a1.set_xscale("log", base=2); a1.set_yscale("log")
    a1.set_xticks(Ts); a1.set_xticklabels(Ts)
    a1.set_xlabel("simulation time steps T"); a1.set_ylabel("spikes per CORRECT classification")
    a1.set_title("(b) energy proxy: total spikes (in+h1+h2) / correct")
    a1.legend(fontsize=8); a1.grid(alpha=0.3, which="both")

    a2 = ax[1, 0]
    for arm in arm_names:
        xs = [M[arm]["spikes_per_correct"][str(T)] for T in Ts]
        ys = [M[arm]["test_acc_mean"][str(T)] for T in Ts]
        a2.plot(xs, ys, marker="o", color=colors.get(arm), ls=styles.get(arm, "-"), label=arm)
        for T, xx, yy in zip(Ts, xs, ys):
            a2.annotate(str(T), (xx, yy), fontsize=6, xytext=(2, 3), textcoords="offset points")
    a2.axhline(mlp_mean, color="k", ls=":", lw=2, label="ReLU MLP")
    a2.set_xscale("log")
    a2.set_xlabel("spikes per correct classification (log)"); a2.set_ylabel("test accuracy")
    a2.set_title("(c) accuracy-energy Pareto frontier (labels = T)")
    a2.legend(fontsize=8); a2.grid(alpha=0.3, which="both")

    a3 = ax[1, 1]
    w = 0.35
    xpos = np.arange(len(Ts))
    a3.bar(xpos - w / 2, [M["rate"]["spikes_by_layer"][str(T)]["input"] for T in Ts], w,
           label="rate: input", color="#d62728", alpha=0.55)
    a3.bar(xpos - w / 2, [M["rate"]["spikes_by_layer"][str(T)]["hidden1"]
                          + M["rate"]["spikes_by_layer"][str(T)]["hidden2"] for T in Ts], w,
           bottom=[M["rate"]["spikes_by_layer"][str(T)]["input"] for T in Ts],
           label="rate: hidden", color="#d62728")
    a3.bar(xpos + w / 2, [M["latency"]["spikes_by_layer"][str(T)]["input"] for T in Ts], w,
           label="latency: input", color="#1f77b4", alpha=0.55)
    a3.bar(xpos + w / 2, [M["latency"]["spikes_by_layer"][str(T)]["hidden1"]
                          + M["latency"]["spikes_by_layer"][str(T)]["hidden2"] for T in Ts], w,
           bottom=[M["latency"]["spikes_by_layer"][str(T)]["input"] for T in Ts],
           label="latency: hidden", color="#1f77b4")
    a3.set_xticks(xpos); a3.set_xticklabels(Ts)
    a3.set_yscale("log")
    a3.set_xlabel("simulation time steps T"); a3.set_ylabel("spikes per sample (log)")
    a3.set_title("(d) where the spikes are: input vs hidden")
    a3.legend(fontsize=8); a3.grid(alpha=0.3, axis="y", which="both")

    fig.suptitle("Rate vs latency coding in a 0.085M-param surrogate-gradient LIF SNN "
                 "(sklearn digits, CPU)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(HERE / "chart.png", dpi=140)

    out = {
        "id": cfg.get("id", "unknown"),
        "git_commit": git_sha(),
        "seed": int(cfg.get("seed", 0)),
        "duration_sec": round(time.time() - t0, 2),
        "metrics": metrics,
        "env": env_info(),
        "status": "done",
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(out, f, indent=2)
    if out["duration_sec"] > p["time_budget_s"]:
        print(f"WARNING: exceeded the declared time box "
              f"({out['duration_sec']}s > {p['time_budget_s']}s)")
    print(f"\nwall clock {out['duration_sec']}s  (budget {p['time_budget_s']}s)")
    print("first T matching ReLU MLP mean:", json.dumps(metrics["first_T_matching_mlp_mean"]))


if __name__ == "__main__":
    main()
