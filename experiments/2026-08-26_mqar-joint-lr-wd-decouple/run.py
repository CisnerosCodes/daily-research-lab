"""MQAR joint-LR x weight-decay decoupling. Deterministic, CPU-only, writes results.json + chart.png.

Question (the named next step of 2026-08-13_mqar-joint-lr-scaling): uniform x4 LR (j4)
escapes at 400/400/400 where exact inverse scaling of j1's ~1100 predicts ~275 - a
~1.45x residual sublinearity that survives matching the gate's speed. The named suspect
is AdamW's decoupled weight decay: the per-step shrink is lr * wd * p, so a 4x LR also
decays weights 4x harder per step - the LR multiplier is not a pure time-rescaling of
the update. (Counting argument the run adjudicates: over a fixed AMOUNT of progress the
x4 run takes 1/4 the steps at 4x decay per step, so TOTAL decay per unit progress is
invariant - if that idealization held, wd could not explain the residual. Noise/geometry
does not rescale: per unit drift, diffusion is 4x larger at 4x LR.)

Arms (all N=8, d=64, byte-identical paired inits per seed, identical data streams):
  j1   (1x, 1x, wd .01)    in-run replication anchor of 08-13 j1 [1200/1000/1100]
  j1w0 (1x, 1x, wd 0)      does wd matter at base speed at all?
  j4   (4x, 4x, wd .01)    in-run replication anchor of 08-13 j4 [400/400/400]
  j4w0 (4x, 4x, wd 0)      decisive: wd-free uniform x4
  j4wc (4x, 4x, wd .0025)  per-step decay lr*wd matched to j1's

Interpretation matrix (escape step at N=8, eval grid 100):
  j4w0 ~ j1w0/4 (~300 or less)     -> wd IS the residual: Adam's time-reparametrization
                                       is broken by the LR-coupled decay term
  j4w0 ~ 400 (= j4)                -> wd exonerated: residual is gradient noise /
                                       plateau geometry / curvature
  j1w0 far from j1 (~1100)         -> wd shapes the 1x plateau itself (grokking-style);
                                       read ratios within matched wd only
  j4wc splits j4/j4w0              -> partial: per-step decay magnitude is the active dial
Secondary: does 400/400/400 step-determinism survive wd changes; travel-at-escape
constancy readouts continued from 08-07/08-13.

Usage:  python run.py            (full grid)
        MQAR_PILOT=1 python run.py   (tiny grid, timing sanity check)
"""
import json, math, os, random, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def set_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


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


# ----------------------------------------------------------------------------- data
def make_batch(B, N, key_vocab, val_vocab, gen):
    import torch
    keys = torch.argsort(torch.rand(B, key_vocab, generator=gen), dim=1)[:, :N]
    vals = torch.randint(0, val_vocab, (B, N), generator=gen) + key_vocab
    ctx = torch.stack([keys, vals], dim=2).reshape(B, 2 * N)
    perm = torch.argsort(torch.rand(B, N, generator=gen), dim=1)
    qkeys = torch.gather(keys, 1, perm)
    qvals = torch.gather(vals, 1, perm)
    x = torch.cat([ctx, qkeys], dim=1)                       # (B, 3N)
    y = torch.full((B, 3 * N), -100, dtype=torch.long)
    y[:, 2 * N:] = qvals
    return x, y


# ----------------------------------------------------------------------------- model
def build_model(gated, d_model, n_layers, n_heads, mlp_exp, vocab, max_len, gate_bias):
    """gated=False -> vanilla elu+1 linear attention; gated=True -> dense per-channel decay.
    Byte-identical code path to 2026-07-28/29/08-01/08-03/08-07/08-13 (their 'dense' arms)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    h, dh = n_heads, d_model // n_heads

    class Gate(nn.Module):
        """Returns per-channel log-decay logg of shape (B, h, T, dh), or None when not gated."""
        def __init__(self):
            super().__init__()
            if gated:
                self.lin = nn.Linear(d_model, h * dh, bias=True)
                nn.init.zeros_(self.lin.weight)
                nn.init.constant_(self.lin.bias, gate_bias)

        def forward(self, x):  # x: (B, T, d_model)
            if not gated:
                return None
            B, T, _ = x.shape
            logit = self.lin(x)
            return F.logsigmoid(logit).view(B, T, h, dh).permute(0, 2, 1, 3)  # (B,h,T,dh)

    class GatedLinAttn(nn.Module):
        """elu+1 linear attention with optional per-channel decay, exact closed form."""
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
            self.out = nn.Linear(d_model, d_model, bias=False)
            self.gate = Gate()

        def forward(self, x):
            B, T, _ = x.shape
            q, k, v = self.qkv(x).split(d_model, dim=2)
            q, k, v = (t.view(B, T, h, dh).transpose(1, 2) for t in (q, k, v))
            q, k = F.elu(q) + 1, F.elu(k) + 1
            logg = self.gate(x)
            if logg is None:
                scores = q @ k.transpose(2, 3)                       # (B,h,T,T)
            else:
                A = torch.cumsum(logg, dim=2)                        # (B,h,T,dh)
                D = (A.unsqueeze(3) - A.unsqueeze(2)).clamp(max=0.0).exp()  # (B,h,T,T,dh)
                scores = torch.einsum("bhtc,bhtsc,bhsc->bhts", q, D, k)
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            scores = scores.masked_fill(~mask, 0.0)
            z = scores.sum(dim=3, keepdim=True) + 1e-6
            return self.out(((scores / z) @ v).transpose(1, 2).reshape(B, T, d_model))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1 = nn.LayerNorm(d_model)
            self.mix = GatedLinAttn()
            self.n2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, mlp_exp * d_model), nn.GELU(),
                nn.Linear(mlp_exp * d_model, d_model))

        def forward(self, x):
            x = x + self.mix(self.n1(x))
            return x + self.mlp(self.n2(x))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab, d_model)
            self.pos = nn.Embedding(max_len, d_model)
            self.blocks = nn.ModuleList(Block() for _ in range(n_layers))
            self.lnf = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab, bias=False)

        def forward(self, idx):
            import torch as _t
            x = self.emb(idx) + self.pos(_t.arange(idx.shape[1], device=idx.device))
            for b in self.blocks:
                x = b(x)
            return self.head(self.lnf(x))

    return Model()


# ----------------------------------------------------------------------------- train/eval
def evaluate(model, xe, ye, chunk):
    import torch
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i in range(0, xe.shape[0], chunk):
            logits = model(xe[i:i + chunk])
            m = ye[i:i + chunk] != -100
            pred = logits.argmax(dim=2)
            correct += (pred[m] == ye[i:i + chunk][m]).sum().item()
            total += m.sum().item()
    model.train()
    return correct / total


def gate_travel(model, gate_bias):
    """Distance of gate params from their init (W starts at 0, b at gate_bias)."""
    import torch
    w_norm = b_drift = 0.0
    for n, p in model.named_parameters():
        if ".gate." not in n:
            continue
        with torch.no_grad():
            if "weight" in n:
                w_norm += float(p.norm())
            else:
                b_drift += float((p - gate_bias).norm())
    return round(w_norm, 4), round(b_drift, 4)


def snapshot_backbone(model):
    """Clone every non-gate parameter at init (for travel-from-init readouts)."""
    import torch
    with torch.no_grad():
        return {n: p.detach().clone() for n, p in model.named_parameters()
                if ".gate." not in n}


def backbone_travel(model, snap):
    """(attn qkv/out travel, whole-backbone travel): sum of ||p - p0|| per group."""
    import torch
    attn = full = 0.0
    with torch.no_grad():
        for n, p in model.named_parameters():
            if ".gate." in n:
                continue
            d = float((p - snap[n]).norm())
            full += d
            if ".qkv." in n or ".out." in n:
                attn += d
    return round(attn, 4), round(full, 4)


def param_norm(model):
    import torch
    with torch.no_grad():
        return round(float(torch.sqrt(sum(p.pow(2).sum() for p in model.parameters()))), 4)


def run_one(arm, N, seed, P):
    import torch
    d = P["d_model"]
    name, gmult, bmult, wd, init_key = (arm["name"], arm["gate_lr_mult"],
                                        arm["backbone_lr_mult"], arm["weight_decay"],
                                        arm["init_key"])
    gated = init_key == "dense"
    # deterministic init seed; sum(ord) not hash() (2026-07-27 fix). CRUCIAL: the seed uses
    # init_key, not the arm name, so all gated arms are byte-identical per seed (and the
    # j1/j4 anchors reproduce the 2026-08-13 j1/j4 rows exactly - wd lives in the optimizer).
    torch.manual_seed(1_000_003 * seed + 101 * d + 13 * N + sum(ord(c) for c in init_key) % 997)
    model = build_model(gated, d, P["n_layers"], P["n_heads"], P["mlp_expansion"],
                        P["key_vocab"] + P["val_vocab"], 3 * 16, P["gate_bias_init"])
    n_params = sum(p.numel() for p in model.parameters())
    snap = snapshot_backbone(model)
    # per-parameter-group LR exactly as 08-13; weight decay is the arm's wd everywhere.
    gate_ps = [p for n_, p in model.named_parameters() if ".gate." in n_]
    rest_ps = [p for n_, p in model.named_parameters() if ".gate." not in n_]
    opt = torch.optim.AdamW(
        [{"params": rest_ps, "lr": P["lr"] * bmult},
         {"params": gate_ps, "lr": P["lr"] * gmult}],
        lr=P["lr"], weight_decay=wd)
    # same train stream + eval set across arms for a given (N, seed) - identical to parents
    gtrain = torch.Generator().manual_seed(500_000 + 100 * N + seed)
    geval = torch.Generator().manual_seed(900_000 + 100 * N + seed)
    xe, ye = make_batch(P["eval_sequences"], N, P["key_vocab"], P["val_vocab"], geval)

    t0, acc, step, traj, travel_traj = time.time(), 0.0, 0, [], []
    saw_nan = False
    lossfn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    for step in range(1, P["train_steps"] + 1):
        x, y = make_batch(P["batch_size"], N, P["key_vocab"], P["val_vocab"], gtrain)
        logits = model(x)
        loss = lossfn(logits.reshape(-1, logits.shape[2]), y.reshape(-1))
        if not torch.isfinite(loss):
            saw_nan = True
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % P["eval_every"] == 0:
            acc = evaluate(model, xe, ye, P["eval_chunk"])
            traj.append([step, round(acc, 4)])
            gw, gb = gate_travel(model, P["gate_bias_init"]) if gated else (None, None)
            ba, bf = backbone_travel(model, snap)
            travel_traj.append([step, gw, gb, ba, bf, param_norm(model)])
            if acc >= P["early_stop_acc"]:
                break
    if not saw_nan and step % P["eval_every"] != 0 and acc < P["early_stop_acc"]:
        acc = evaluate(model, xe, ye, P["eval_chunk"])
        traj.append([step, round(acc, 4)])
        gw, gb = gate_travel(model, P["gate_bias_init"]) if gated else (None, None)
        ba, bf = backbone_travel(model, snap)
        travel_traj.append([step, gw, gb, ba, bf, param_norm(model)])
    escape = next((s for s, a in traj if a >= P["escape_threshold"]), None)
    # travel at the first eval at/after escape (the clock readout)
    travel_at_escape = None
    if escape is not None and travel_traj:
        travel_at_escape = next(
            ({"gate_W": gw, "gate_b": gb, "attn": ba, "backbone": bf, "pnorm": pn}
             for s, gw, gb, ba, bf, pn in travel_traj if s >= escape), None)
    ba_end, bf_end = backbone_travel(model, snap)
    return {"arm": name, "gate_lr_mult": gmult, "backbone_lr_mult": bmult,
            "weight_decay": wd, "num_pairs": N, "seed": seed,
            "acc": round(acc, 4), "steps": step, "escape_step": escape, "params": n_params,
            "saw_nan": saw_nan, "travel_at_escape": travel_at_escape,
            "backbone_travel_end": [ba_end, bf_end],
            "gate_travel_end": gate_travel(model, P["gate_bias_init"]) if gated else None,
            "param_norm_end": param_norm(model),
            "traj": traj, "travel_traj": travel_traj,
            "secs": round(time.time() - t0, 1)}


# ----------------------------------------------------------------------------- chart
def make_chart(runs, P, headline):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    Nm = P["num_pairs_main"]
    fam = [a["name"] for a in P["arms"]]
    cmap = {"j1": "#6a3d9a", "j1w0": "#b294d6", "j4": "#e31a1c",
            "j4w0": "#ff9a9a", "j4wc": "#fdbf6f"}
    mstyle = {0: "-", 1: "--", 2: ":"}
    seeds_all = sorted({r["seed"] for r in runs})
    cens = P["train_steps"] + 200  # plotting position for "never escaped"
    xpos = {a: i for i, a in enumerate(fam)}

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 4.2))

    # panel 1: escape step per arm, paired lines per seed + reference levels
    for s in seeds_all:
        xs, ys = [], []
        for a in fam:
            rr = [r for r in runs if r["arm"] == a and r["seed"] == s and r["num_pairs"] == Nm]
            if rr:
                e = rr[0]["escape_step"]
                xs.append(xpos[a])
                ys.append(e if e is not None else cens)
        ax1.plot(xs, ys, marker="o", ms=5, lw=1.5, ls=mstyle.get(s, "-"),
                 color="#444444", alpha=0.85, label=f"seed {s}")
    for anchor_arm, ls, col in (("j1", "-.", "#33a02c"), ("j1w0", ":", "#1f78b4")):
        e1 = [r["escape_step"] for r in runs if r["arm"] == anchor_arm and r["escape_step"]]
        if e1:
            anchor = float(np.mean(e1))
            ax1.axhline(anchor / 4.0, color=col, lw=1.2, ls=ls, alpha=0.8,
                        label=f"exact 1/4 of {anchor_arm} ({anchor/4:.0f})")
    ax1.set_xticks(list(xpos.values()), fam)
    ax1.set_xlabel("arm (lr mult, wd): j1=(1,.01) j1w0=(1,0) j4=(4,.01) j4w0=(4,0) j4wc=(4,.0025)")
    ax1.set_ylabel(f"escape step (first eval >= {P['escape_threshold']}, grid {P['eval_every']})")
    ax1.legend(frameon=False, fontsize=8)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_title("does removing/compensating wd restore exact inverse scaling?", fontsize=10)

    # panel 2: trajectories, color = arm, linestyle = seed
    for a in fam:
        rs = [r for r in runs if r["arm"] == a and r["num_pairs"] == Nm]
        first = True
        for r in rs:
            xs = [s for s, _ in r["traj"]]
            ys = [v for _, v in r["traj"]]
            ax2.plot(xs, ys, ls=mstyle.get(r["seed"], "-"), color=cmap.get(a, "#1f78b4"),
                     label=a if first else None, lw=1.3, alpha=0.9)
            first = False
    ax2.axhline(P["escape_threshold"], color="#999999", lw=0.8, ls="--")
    ax2.set_ylim(-0.03, 1.05)
    ax2.set_xlabel("train step")
    ax2.set_ylabel("recall accuracy")
    ax2.legend(frameon=False, fontsize=7.5, ncol=2)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_title("trajectories (solid s0, dashed s1, dotted s2)", fontsize=10)

    # panel 3: parameter norm over training, color = arm (what wd actually does to scale)
    for a in fam:
        rs = [r for r in runs if r["arm"] == a and r["num_pairs"] == Nm]
        first = True
        for r in rs:
            xs = [t[0] for t in r["travel_traj"]]
            ys = [t[5] for t in r["travel_traj"]]
            ax3.plot(xs, ys, ls=mstyle.get(r["seed"], "-"), color=cmap.get(a, "#1f78b4"),
                     label=a if first else None, lw=1.2, alpha=0.9)
            first = False
    ax3.set_xlabel("train step")
    ax3.set_ylabel("global parameter L2 norm")
    ax3.legend(frameon=False, fontsize=7.5, ncol=2)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.set_title("weight-norm trajectories: how different are the wd regimes?", fontsize=10)

    fig.suptitle(headline, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(HERE / "chart.png", dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    cfg = load_config()
    P = dict(cfg["params"])
    seed = int(cfg.get("seed", 0))
    set_seeds(seed)
    if os.environ.get("MQAR_PILOT"):
        P.update(train_steps=100, eval_every=50,
                 arms=[{"name": "j4w0", "gate_lr_mult": 4.0, "backbone_lr_mult": 4.0,
                        "weight_decay": 0.0, "init_key": "dense", "seeds": [0]}])
    t0 = time.time()
    runs = []

    for arm in P["arms"]:
        for s in arm["seeds"]:
            r = run_one(arm, P["num_pairs_main"], s, P)
            runs.append(r)
            print(f"[{time.time()-t0:7.1f}s] {r['arm']:5s} seed={s} wd={r['weight_decay']} "
                  f"acc={r['acc']:.3f} esc={r['escape_step']} steps={r['steps']} "
                  f"nan={r['saw_nan']} travel@esc={r['travel_at_escape']} "
                  f"pnorm_end={r['param_norm_end']} ({r['secs']}s)", flush=True)

    # aggregates
    import numpy as np
    Nm = P["num_pairs_main"]
    mean_acc, escapes, n_esc, attn_travel_esc, gate_travel_esc, pnorm_esc = {}, {}, {}, {}, {}, {}
    for a in P["arms"]:
        name = a["name"]
        rs = [r for r in runs if r["arm"] == name and r["num_pairs"] == Nm]
        if rs:
            mean_acc[name] = round(float(np.mean([r["acc"] for r in rs])), 4)
            escapes[name] = [r["escape_step"] for r in rs]
            n_esc[name] = sum(e is not None for e in escapes[name])
            tw = [r["travel_at_escape"]["attn"] for r in rs if r["travel_at_escape"]]
            attn_travel_esc[name] = round(float(np.mean(tw)), 2) if tw else None
            gw = [r["travel_at_escape"]["gate_W"] for r in rs
                  if r["travel_at_escape"] and r["travel_at_escape"]["gate_W"] is not None]
            gate_travel_esc[name] = round(float(np.mean(gw)), 2) if gw else None
            pn = [r["travel_at_escape"]["pnorm"] for r in rs if r["travel_at_escape"]]
            pnorm_esc[name] = round(float(np.mean(pn)), 2) if pn else None

    # the decision ratios (censored escapes excluded, count reported)
    def mean_esc(name):
        es = [e for e in escapes.get(name, []) if e is not None]
        return float(np.mean(es)) if es else None
    ratios = {}
    for hi, lo in (("j1", "j4"), ("j1w0", "j4w0"), ("j1w0", "j4wc"), ("j1", "j4wc")):
        a, b = mean_esc(hi), mean_esc(lo)
        ratios[f"{hi}/{lo}"] = round(a / b, 3) if a and b else None

    headline = "escape steps at N=8 (paired seeds): " + ", ".join(
        f"{a['name']} {escapes.get(a['name'], '?')}" for a in P["arms"])
    make_chart(runs, P, headline)

    results = {
        "id": cfg.get("id", "unknown"),
        "git_commit": git_sha(),
        "seed": seed,
        "duration_sec": round(time.time() - t0, 2),
        "metrics": {"runs": runs, "mean_acc_N8": mean_acc,
                    "escape_steps_N8": escapes, "n_escaped": n_esc,
                    "escape_ratio_means": ratios,
                    "attn_travel_at_escape_mean": attn_travel_esc,
                    "gate_W_at_escape_mean": gate_travel_esc,
                    "param_norm_at_escape_mean": pnorm_esc},
        "env": env_info(),
        "status": "done",
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(headline)
    print("escape ratio means:", ratios)


if __name__ == "__main__":
    main()
