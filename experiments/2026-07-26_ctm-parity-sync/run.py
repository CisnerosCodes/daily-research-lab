"""Nano Continuous Thought Machine: is the synchronization readout the signal, or just recurrence depth?

Five arms share an IDENTICAL recurrent core and are matched on parameter count, internal
ticks T, optimizer, step budget and batch stream.  Only the READOUT (the representation
handed to the output projection) differs:

  (a) sync      -- CTM neuron-pair SYNCHRONIZATION matrix (decayed time-average of
                   post-activation products over the tick history)          [paper]
  (b) last      -- last-tick post-activation vector  (no sync, no history)
  (c) mean      -- mean post-activation over ticks   (history, no pairing)
  (e) pairlast  -- neuron-pair products at the LAST TICK ONLY
                   (same quadratic feature and same dimensionality as sync,
                    but zero time-integration -> isolates "sync" from "pairs")
  (d) gru       -- plain GRU core, last-state readout (no per-neuron history MLPs)

Task: cumulative / prefix parity of an L-bit binary string, presented statically; the
model reads it through attention across T internal ticks.

Usage:  python run.py
"""
import json, math, os, random, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- utils
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
    for mod in ("numpy", "torch", "matplotlib"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:
            pass
    return info


# --------------------------------------------------------------------------- data
def make_batch(gen, batch, L):
    """Random L-bit strings; targets are the cumulative (prefix) parity at every index."""
    x = torch.randint(0, 2, (batch, L), generator=gen)
    y = torch.cumsum(x, dim=1) % 2
    return x, y


# --------------------------------------------------------------------------- model
class NanoCTM(nn.Module):
    """Shared core + swappable readout.

    mode in {sync, last, mean, pairlast}:  CTM core (synapse MLP + per-neuron history MLPs)
    mode == 'gru':                         GRUCell core (no per-neuron history)
    """

    def __init__(self, mode, L, d, N, M, h_syn, h_neuron, pair_idx, out_pos, seed=0):
        super().__init__()
        self.mode, self.L, self.d, self.N, self.M = mode, L, d, N, M
        self.out_pos = out_pos
        g = torch.Generator().manual_seed(seed)

        # ---- shared input pathway (identical structure for every arm) -----------
        self.tok_emb = nn.Embedding(2, d)
        self.pos_emb = nn.Parameter(torch.randn(L, d, generator=g) * 0.02)
        self.Wk = nn.Linear(d, d)
        self.Wv = nn.Linear(d, d)
        self.Wq = nn.Linear(N, d)

        if mode == "gru":
            self.cell = nn.GRUCell(d, N)
            self.z0 = nn.Parameter(torch.zeros(N))
            feat_dim = N
        else:
            # synapse model: [z_{t-1}, o_t] -> pre-activations a_t
            self.syn = nn.Sequential(nn.Linear(N + d, h_syn), nn.ReLU(), nn.Linear(h_syn, N))
            # per-neuron private history MLP: last M pre-activations -> post-activation
            self.W1 = nn.Parameter(torch.randn(N, M, h_neuron, generator=g) / math.sqrt(M))
            self.b1 = nn.Parameter(torch.zeros(N, h_neuron))
            self.W2 = nn.Parameter(torch.randn(N, h_neuron, generator=g) / math.sqrt(h_neuron))
            self.b2 = nn.Parameter(torch.zeros(N))
            self.z0 = nn.Parameter(torch.zeros(N))
            self.a0 = nn.Parameter(torch.randn(N, M, generator=g) * 0.02)
            if mode in ("sync", "pairlast"):
                self.register_buffer("pair_i", pair_idx[0])
                self.register_buffer("pair_j", pair_idx[1])
                feat_dim = pair_idx[0].numel()
                if mode == "sync":
                    self.decay = nn.Parameter(torch.zeros(feat_dim))
            else:
                feat_dim = N
        self.feat_dim = feat_dim
        self.readout = nn.Linear(feat_dim, 2 * out_pos)
        nn.init.normal_(self.readout.weight, std=0.02)
        nn.init.zeros_(self.readout.bias)

    def forward(self, x, T):
        B = x.shape[0]
        e = self.tok_emb(x) + self.pos_emb  # [B,L,d]
        k, v = self.Wk(e), self.Wv(e)
        z = self.z0.expand(B, -1)
        if self.mode != "gru":
            A = self.a0.expand(B, -1, -1)
        if self.mode == "sync":
            r = torch.exp(-torch.abs(self.decay))  # in (0,1]
            numer = torch.zeros(B, self.feat_dim)
            denom = torch.zeros(B, self.feat_dim)
        if self.mode == "mean":
            run = torch.zeros(B, self.N)

        logits = []
        for t in range(T):
            q = self.Wq(z)                                              # [B,d]
            att = torch.softmax(torch.einsum("bd,bld->bl", q, k) / math.sqrt(self.d), -1)
            o = torch.einsum("bl,bld->bd", att, v)                      # [B,d]
            if self.mode == "gru":
                z = self.cell(o, z)
                feat = z
            else:
                a = self.syn(torch.cat([z, o], -1))                     # [B,N]
                A = torch.cat([A[:, :, 1:], a.unsqueeze(-1)], -1)       # [B,N,M]
                hh = F.relu(torch.einsum("bnm,nmh->bnh", A, self.W1) + self.b1)
                z = torch.einsum("bnh,nh->bn", hh, self.W2) + self.b2   # [B,N]
                if self.mode == "sync":
                    prod = z[:, self.pair_i] * z[:, self.pair_j]
                    numer = numer * r + prod
                    denom = denom * r + 1.0
                    feat = numer / torch.sqrt(denom)
                elif self.mode == "pairlast":
                    feat = z[:, self.pair_i] * z[:, self.pair_j]
                elif self.mode == "mean":
                    run = run + z
                    feat = run / (t + 1)
                else:  # last
                    feat = z
            logits.append(self.readout(feat).view(B, self.out_pos, 2))
        return torch.stack(logits, 1)  # [B,T,out_pos,2]


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def build(mode, cfg, h_syn, h_gru, pair_idx, seed):
    N = h_gru if mode == "gru" else cfg["N"]
    return NanoCTM(mode, cfg["L"], cfg["d"], N, cfg["M"], h_syn, cfg["h_neuron"],
                   pair_idx, cfg["L"], seed=seed)


# --------------------------------------------------------------------------- loss
def ctm_loss_and_stats(logits, y):
    """CTM two-tick loss: mean of (loss at min-loss tick, loss at max-certainty tick)."""
    B, T, P, _ = logits.shape
    ce = F.cross_entropy(logits.reshape(-1, 2), y.unsqueeze(1).expand(B, T, P).reshape(-1),
                         reduction="none").view(B, T, P).mean(-1)          # [B,T]
    logp = F.log_softmax(logits, -1)
    ent = -(logp.exp() * logp).sum(-1) / math.log(2.0)                     # [B,T,P]
    cert = (1.0 - ent).mean(-1)                                            # [B,T]
    t1 = ce.argmin(1)
    t2 = cert.argmax(1)
    idx = torch.arange(B)
    loss = 0.5 * (ce[idx, t1] + ce[idx, t2]).mean()
    return loss, cert


@torch.no_grad()
def evaluate(model, X, Y, T, chunk=500):
    accs_cert, accs_final, exact_cert, per_pos = [], [], [], []
    for s in range(0, X.shape[0], chunk):
        x, y = X[s:s + chunk], Y[s:s + chunk]
        logits = model(x, T)
        _, cert = ctm_loss_and_stats(logits, y)
        B = x.shape[0]
        idx = torch.arange(B)
        tb = cert.argmax(1)
        pc = logits[idx, tb].argmax(-1)          # [B,P] max-certainty tick
        pf = logits[:, -1].argmax(-1)            # [B,P] final tick
        accs_cert.append((pc == y).float().mean(-1))
        accs_final.append((pf == y).float().mean(-1))
        exact_cert.append((pc == y).all(-1).float())
        per_pos.append((pc == y).float())
    return (torch.cat(accs_cert).mean().item(),
            torch.cat(accs_final).mean().item(),
            torch.cat(exact_cert).mean().item(),
            torch.cat(per_pos).mean(0).tolist())


# --------------------------------------------------------------------------- train
def train_one(mode, T, seed, cfg, h_syn, h_gru, pair_idx, Xte, Yte, budget_s, steps):
    set_seeds(seed)
    model = build(mode, cfg, h_syn, h_gru, pair_idx, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    warm = min(cfg["warmup"], max(1, steps // 8))
    gen = torch.Generator().manual_seed(1000 + seed)   # identical batch stream per seed
    t0 = time.time()
    losses = []
    done = 0
    for step in range(steps):
        lr_scale = (step + 1) / warm if step < warm else \
            0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))
        for gparam in opt.param_groups:
            gparam["lr"] = cfg["lr"] * lr_scale
        x, y = make_batch(gen, cfg["batch"], cfg["L"])
        logits = model(x, T)
        loss, _ = ctm_loss_and_stats(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        done = step + 1
        if time.time() - t0 > budget_s:
            break
    train_s = time.time() - t0
    ac, af, ex, pp = evaluate(model, Xte, Yte, T)
    return {
        "arm": mode, "T": T, "seed": seed, "n_params": n_params(model),
        "feat_dim": model.feat_dim, "steps_run": done, "time_capped": done < steps,
        "train_seconds": round(train_s, 1),
        "final_loss": round(float(np.mean(losses[-25:])), 4),
        "bit_acc_maxcert": round(ac, 4), "bit_acc_finaltick": round(af, 4),
        "exact_match": round(ex, 4),
        "acc_by_prefix_pos": [round(p, 4) for p in pp],
    }


# --------------------------------------------------------------------------- main
def main():
    cfg_all = load_config()
    cfg = cfg_all["params"]
    seed = int(cfg_all.get("seed", 0))
    set_seeds(seed)
    t_start = time.time()

    # fixed neuron-pair subset, shared by every sync/pairlast run
    gpair = torch.Generator().manual_seed(cfg["pair_seed"])
    iu = torch.triu_indices(cfg["N"], cfg["N"], offset=0)
    sel = torch.randperm(iu.shape[1], generator=gpair)[: cfg["n_pairs"]]
    pair_idx = (iu[0][sel].contiguous(), iu[1][sel].contiguous())

    # ---- iso-parameter matching: sync arm is the reference -------------------
    ref = build("sync", cfg, cfg["h_syn_ref"], None, pair_idx, 0)
    target = n_params(ref)

    def search(mode, lo, hi):
        best, bestd = lo, 10 ** 9
        for w in range(lo, hi + 1):
            n = n_params(build(mode, cfg, w if mode != "gru" else cfg["h_syn_ref"],
                               w if mode == "gru" else None, pair_idx, 0))
            dd = abs(n - target)
            if dd < bestd:
                best, bestd = w, dd
        return best

    h_syn_small = search("last", 100, 500)      # arms with an N-dim readout feature
    h_gru = search("gru", 40, 220)
    widths = {"sync": cfg["h_syn_ref"], "pairlast": cfg["h_syn_ref"],
              "last": h_syn_small, "mean": h_syn_small, "gru": h_gru}
    print(f"[param match] target={target}  h_syn(last/mean)={h_syn_small}  gru_hidden={h_gru}")

    # ---- fixed held-out eval set (identical for every run) -------------------
    gte = torch.Generator().manual_seed(999)
    Xte, Yte = make_batch(gte, cfg["n_eval"], cfg["L"])

    arms = cfg["arms"]
    Ts = cfg["ticks"]
    seeds = cfg["seeds"]
    total_runs = len(arms) * len(Ts) * len(seeds)
    budget_total = cfg["budget_seconds"]

    def hw(a):
        return (widths[a] if a != "gru" else cfg["h_syn_ref"],
                widths["gru"] if a == "gru" else None)

    # ---- timing calibration -> ONE global step count for every run ----------
    # (all arms must get identical steps; a per-run wall-clock cap would silently
    #  hand the cheap arms more optimisation than the expensive ones.)
    gcal = torch.Generator().manual_seed(7)
    per_step = {}
    for T in Ts:
        for a in arms:
            hs, hg = hw(a)
            m = build(a, cfg, hs, hg, pair_idx, 0)
            o = torch.optim.AdamW(m.parameters(), lr=1e-4)
            for i in range(cfg["calib_steps"] + 1):
                if i == 1:
                    tc = time.time()
                x, y = make_batch(gcal, cfg["batch"], cfg["L"])
                lo = m(x, T)
                ll, _ = ctm_loss_and_stats(lo, y)
                o.zero_grad(set_to_none=True); ll.backward(); o.step()
            per_step[(a, T)] = (time.time() - tc) / cfg["calib_steps"]
    cost_per_step = sum(per_step.values()) * len(seeds) * cfg["safety_factor"]
    avail = budget_total - (time.time() - t_start) - cfg["eval_reserve_s"]
    gran = cfg["step_granularity"]
    steps = int(min(cfg["max_steps"], max(gran, (avail / cost_per_step) // gran * gran)))
    print(f"[plan] {total_runs} runs, budget {budget_total}s, "
          f"{cost_per_step:.3f}s per step across all runs -> steps={steps} "
          f"(projected {steps * cost_per_step:.0f}s)")

    runs = []
    for T in Ts:
        for a in arms:
            for s in seeds:
                # emergency runaway guard ONLY. Deliberately not clamped to the remaining
                # budget: every arm must get the same number of steps, so a small overrun
                # is preferable to silently under-training the cheap arms' late cells.
                per = 3.0 * steps * per_step[(a, T)]
                hs, hg = hw(a)
                r = train_one(a, T, s, cfg, hs, hg, pair_idx, Xte, Yte, per, steps)
                runs.append(r)
                print(f"  {a:9s} T={T:<3d} seed={s}  params={r['n_params']:6d} "
                      f"steps={r['steps_run']:<5d} {r['train_seconds']:5.1f}s  "
                      f"bit_acc={r['bit_acc_maxcert']:.4f} exact={r['exact_match']:.3f}"
                      f"{'  [CAPPED]' if r['time_capped'] else ''}")

    # ---- aggregate ----------------------------------------------------------
    def agg(a, T, key="bit_acc_maxcert"):
        v = [r[key] for r in runs if r["arm"] == a and r["T"] == T]
        return (round(float(np.mean(v)), 4), [round(x, 4) for x in v]) if v else (None, [])

    table = {a: {str(T): agg(a, T)[0] for T in Ts} for a in arms}
    table_per_seed = {a: {str(T): agg(a, T)[1] for T in Ts} for a in arms}
    exact_table = {a: {str(T): agg(a, T, "exact_match")[0] for T in Ts} for a in arms}

    Tmax = max(Ts)
    best_T_per_arm = {a: max(Ts, key=lambda T: table[a][str(T)]) for a in arms}
    seed_spread = max(
        (max(table_per_seed[a][str(T)]) - min(table_per_seed[a][str(T)]))
        for a in arms for T in Ts if len(table_per_seed[a][str(T)]) > 1
    ) if len(seeds) > 1 else None

    deltas_at_Tmax = {f"sync_minus_{a}": round(table["sync"][str(Tmax)] - table[a][str(Tmax)], 4)
                      for a in arms if a != "sync"}
    ranking = sorted(arms, key=lambda a: -table[a][str(Tmax)])

    metrics = {
        "task": f"cumulative (prefix) parity of {cfg['L']}-bit strings",
        "target_params": target,
        "param_spread_pct": round(100 * (max(r["n_params"] for r in runs) -
                                         min(r["n_params"] for r in runs)) / target, 4),
        "widths": widths,
        "n_pairs": cfg["n_pairs"], "N_neurons": cfg["N"], "memory_M": cfg["M"],
        "ticks": Ts, "seeds": seeds, "steps": steps, "batch": cfg["batch"],
        "lr": cfg["lr"], "L_bits": cfg["L"], "n_eval": cfg["n_eval"],
        "any_run_time_capped": any(r["time_capped"] for r in runs),
        "seconds_per_step_by_cell": {f"{a}_T{T}": round(v, 4) for (a, T), v in per_step.items()},
        "bit_acc_maxcert_mean": table,
        "bit_acc_maxcert_per_seed": table_per_seed,
        "exact_match_mean": exact_table,
        "max_seed_spread_bit_acc": round(seed_spread, 4) if seed_spread is not None else None,
        "ranking_at_Tmax": ranking,
        "sync_minus_others_at_Tmax": deltas_at_Tmax,
        "best_T_per_arm": {a: best_T_per_arm[a] for a in arms},
        "acc_by_prefix_pos_at_Tmax": {
            a: [round(float(x), 4) for x in np.mean(
                [r["acc_by_prefix_pos"] for r in runs if r["arm"] == a and r["T"] == Tmax], axis=0)]
            for a in arms},
        "per_run": runs,
    }
    metrics["headline"] = (
        f"at T={Tmax}: " + ", ".join(f"{a} {table[a][str(Tmax)]:.3f}" for a in ranking) +
        f" (bit acc, chance 0.5; max seed spread {metrics['max_seed_spread_bit_acc']})"
    )

    # ---- chart --------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"sync": "#d62728", "pairlast": "#ff7f0e", "last": "#1f77b4",
              "mean": "#2ca02c", "gru": "#7f7f7f"}
    labels = {"sync": "(a) sync matrix [CTM]", "last": "(b) last-tick hidden",
              "mean": "(c) mean over ticks", "pairlast": "(e) pair-products, last tick",
              "gru": "(d) GRU + last state"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    w = 0.8 / len(arms)
    xs = np.arange(len(Ts))
    for i, a in enumerate(arms):
        vals = [table[a][str(T)] for T in Ts]
        ax.bar(xs + i * w - 0.4 + w / 2, vals, w, label=labels[a], color=colors[a])
        for j, T in enumerate(Ts):
            for v in table_per_seed[a][str(T)]:
                ax.plot(xs[j] + i * w - 0.4 + w / 2, v, "k.", ms=3)
    ax.axhline(0.5, color="k", ls="--", lw=1, label="chance")
    ax.set_xticks(xs); ax.set_xticklabels([f"T={T}" for T in Ts])
    ax.set_ylabel("prefix-parity bit accuracy (max-certainty tick)")
    ax.set_title(f"Readout ablation at matched params ({target/1000:.1f}k) & ticks")
    ax.legend(fontsize=7, loc="upper left"); ax.set_ylim(0.4, 1.02)

    ax = axes[1]
    for a in arms:
        ax.plot(Ts, [table[a][str(T)] for T in Ts], "o-", color=colors[a], label=labels[a])
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_xlabel("internal ticks T"); ax.set_ylabel("bit accuracy")
    ax.set_xscale("log", base=2); ax.set_xticks(Ts); ax.set_xticklabels(Ts)
    ax.set_title("accuracy vs recurrence depth"); ax.legend(fontsize=7)

    ax = axes[2]
    for a in arms:
        ax.plot(range(1, cfg["L"] + 1), metrics["acc_by_prefix_pos_at_Tmax"][a],
                "-", color=colors[a], label=labels[a])
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_xlabel("prefix length (bit index)"); ax.set_ylabel("accuracy")
    ax.set_title(f"per-prefix accuracy at T={Tmax}"); ax.legend(fontsize=7)

    fig.suptitle("Nano CTM: does the synchronization readout add anything beyond recurrence depth?",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(HERE / "chart.png", dpi=130)

    results = {
        "id": cfg_all.get("id", "unknown"),
        "git_commit": git_sha(),
        "seed": seed,
        "duration_sec": round(time.time() - t_start, 2),
        "metrics": metrics,
        "env": env_info(),
        "status": "done",
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + metrics["headline"])
    print(f"total {results['duration_sec']:.1f}s")


if __name__ == "__main__":
    main()
