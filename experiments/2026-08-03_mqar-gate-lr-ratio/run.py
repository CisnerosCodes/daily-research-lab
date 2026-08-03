"""MQAR gate LR ratio. Deterministic, CPU-only, writes results.json + chart.png.

Question (follow-up to 2026-08-01_mqar-gate-freeze-before-breakout): freeze-timing showed
that WITHHOLDING gate learning delays or prevents breakout, monotonically in the freeze
step, on paired inits. This is the causal test in the opposite direction: if gate learning
is the rate-limiting clock, then giving ONLY the gate more learning (a per-parameter-group
LR multiplier on the gate's Linear) should pull breakout EARLIER, monotonically in the
multiplier - and starving it (x0.25) should push breakout later or off the 2000-step budget.

Design (inherits the 2026-08-01 pairing upgrade): the LR multiplier lives entirely in the
optimizer, so every gated arm shares the same init key ("dense") and the same data stream.
Within a seed all glr* runs are byte-identical at step 0 and see identical batches; any
divergence is caused by the gate LR alone. glr1 is an exact replication of 2026-08-01's
"dense" arm (identical inits, identical data; only the eval cadence differs, which does not
touch the training trajectory).

Extra readout: gate travel (||W||_F from 0, ||b - b0||) is logged at EVERY eval, so we can
ask whether escape happens at a roughly constant gate travel across multipliers (the gate's
travel is the clock) or at a constant STEP (the clock is elsewhere, e.g. the qkv circuit).

Interpretation matrix:
  escape step monotone DOWN in mult          -> gate learning is rate-limiting (confirmed both directions)
  glr16 destabilizes (acc collapse/NaN)      -> gate can outrun the circuit it serves
  escape flat in mult                        -> the wait is NOT gate-side; revises 2026-08-01's story
  gate ||W|| at escape ~constant across mult -> travel-as-clock (sufficient statistic)

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
    Byte-identical code path to 2026-07-28/29/08-01 (their 'none' and 'dense' arms)."""
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


def run_one(arm, N, seed, P):
    import torch
    d = P["d_model"]
    name, mult, init_key = arm["name"], arm["gate_lr_mult"], arm["init_key"]
    gated = init_key == "dense"
    # deterministic init seed; sum(ord) not hash() (2026-07-27 fix). CRUCIAL: the seed uses
    # init_key, not the arm name, so all gated arms are byte-identical per seed (and 'glr1'/
    # 'none' reproduce the 2026-07-28/29/08-01 'dense'/'none' rows exactly).
    torch.manual_seed(1_000_003 * seed + 101 * d + 13 * N + sum(ord(c) for c in init_key) % 997)
    model = build_model(gated, d, P["n_layers"], P["n_heads"], P["mlp_expansion"],
                        P["key_vocab"] + P["val_vocab"], 3 * 16, P["gate_bias_init"])
    n_params = sum(p.numel() for p in model.parameters())
    # per-parameter-group LR: gate params at mult x base, everything else at base.
    if gated:
        gate_ps = [p for n_, p in model.named_parameters() if ".gate." in n_]
        rest_ps = [p for n_, p in model.named_parameters() if ".gate." not in n_]
        opt = torch.optim.AdamW(
            [{"params": rest_ps, "lr": P["lr"]},
             {"params": gate_ps, "lr": P["lr"] * mult}],
            lr=P["lr"], weight_decay=P["weight_decay"])
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=P["lr"], weight_decay=P["weight_decay"])
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
            if gated:
                travel_traj.append([step, *gate_travel(model, P["gate_bias_init"])])
            if acc >= P["early_stop_acc"]:
                break
    if not saw_nan and step % P["eval_every"] != 0 and acc < P["early_stop_acc"]:
        acc = evaluate(model, xe, ye, P["eval_chunk"])
        traj.append([step, round(acc, 4)])
        if gated:
            travel_traj.append([step, *gate_travel(model, P["gate_bias_init"])])
    escape = next((s for s, a in traj if a >= P["escape_threshold"]), None)
    # gate travel at the first eval at/after escape (the clock readout)
    travel_at_escape = None
    if escape is not None and travel_traj:
        travel_at_escape = next(([w, b] for s, w, b in travel_traj if s >= escape), None)
    travel_end = gate_travel(model, P["gate_bias_init"]) if gated else None
    return {"arm": name, "gate_lr_mult": mult, "num_pairs": N, "seed": seed,
            "acc": round(acc, 4), "steps": step, "escape_step": escape, "params": n_params,
            "saw_nan": saw_nan, "gate_travel_at_escape": travel_at_escape,
            "gate_travel_end": travel_end, "traj": traj, "travel_traj": travel_traj,
            "secs": round(time.time() - t0, 1)}


# ----------------------------------------------------------------------------- chart
def make_chart(runs, P, headline):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    Nm = P["num_pairs_main"]
    fam = [a["name"] for a in P["arms"] if a["init_key"] == "dense"]
    mult_of = {a["name"]: a["gate_lr_mult"] for a in P["arms"] if a["init_key"] == "dense"}
    cmap = {"glr025": "#e31a1c", "glr1": "#6a3d9a", "glr4": "#1f78b4",
            "glr16": "#33a02c", "none": "#888888"}
    mstyle = {0: "-", 1: "--", 2: ":"}
    seeds_all = sorted({r["seed"] for r in runs if r["arm"] in fam})
    cens = P["train_steps"] + 200  # plotting position for "never escaped"

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 4.2))

    # panel 1: escape step vs gate LR mult, one line per seed (paired inits)
    for s in seeds_all:
        xs, ys, censored = [], [], []
        for a in fam:
            rr = [r for r in runs if r["arm"] == a and r["seed"] == s and r["num_pairs"] == Nm]
            if rr:
                xs.append(mult_of[a])
                e = rr[0]["escape_step"]
                ys.append(e if e is not None else cens)
                censored.append(e is None)
        order = np.argsort(xs)
        xs, ys = np.array(xs, dtype=float)[order], np.array(ys, dtype=float)[order]
        censored = np.array(censored)[order]
        ax1.plot(xs, ys, marker="o", ms=5, lw=1.5, ls=mstyle.get(s, "-"),
                 color="#444444", alpha=0.85, label=f"seed {s}")
        if censored.any():
            ax1.scatter(xs[censored], ys[censored], marker="^", s=70, color="#e31a1c",
                        zorder=5, label="censored (never)" if s == seeds_all[0] else None)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks([0.25, 1, 4, 16], ["0.25x", "1x", "4x", "16x"])
    ax1.axhline(cens, color="#e31a1c", lw=0.6, ls=":", alpha=0.5)
    ax1.set_xlabel("gate LR multiplier (backbone LR fixed 1e-3)")
    ax1.set_ylabel(f"escape step (first eval >= {P['escape_threshold']}, grid {P['eval_every']})")
    ax1.legend(frameon=False, fontsize=8)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_title("escape step vs gate LR (paired inits per seed)", fontsize=10)

    # panel 2: trajectories, color = arm, linestyle = seed
    for a in fam + ["none"]:
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

    # panel 3: gate ||W|| travel at escape, per arm (the clock readout)
    for a in fam:
        rs = [r for r in runs if r["arm"] == a and r["num_pairs"] == Nm
              and r["gate_travel_at_escape"] is not None]
        for r in rs:
            ax3.scatter(mult_of[a], r["gate_travel_at_escape"][0], color=cmap.get(a),
                        s=45, alpha=0.85)
    ax3.set_xscale("log", base=2)
    ax3.set_xticks([0.25, 1, 4, 16], ["0.25x", "1x", "4x", "16x"])
    ax3.set_xlabel("gate LR multiplier")
    ax3.set_ylabel("gate ||W||_F at escape eval")
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.set_title("is escape at constant gate travel?", fontsize=10)

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
                 arms=[{"name": "none", "gate_lr_mult": None, "init_key": "none", "seeds": [0]},
                       {"name": "glr4", "gate_lr_mult": 4.0, "init_key": "dense", "seeds": [0]}])
    t0 = time.time()
    runs = []

    for arm in P["arms"]:
        for s in arm["seeds"]:
            r = run_one(arm, P["num_pairs_main"], s, P)
            runs.append(r)
            print(f"[{time.time()-t0:7.1f}s] {r['arm']:7s} seed={s} acc={r['acc']:.3f} "
                  f"esc={r['escape_step']} steps={r['steps']} nan={r['saw_nan']} "
                  f"travel@esc={r['gate_travel_at_escape']} travel@end={r['gate_travel_end']} "
                  f"({r['secs']}s)", flush=True)

    # aggregates
    import numpy as np
    Nm = P["num_pairs_main"]
    mean_acc, escapes, n_esc, travel_esc = {}, {}, {}, {}
    for a in P["arms"]:
        name = a["name"]
        rs = [r for r in runs if r["arm"] == name and r["num_pairs"] == Nm]
        if rs:
            mean_acc[name] = round(float(np.mean([r["acc"] for r in rs])), 4)
            escapes[name] = [r["escape_step"] for r in rs]
            n_esc[name] = sum(e is not None for e in escapes[name])
            tw = [r["gate_travel_at_escape"][0] for r in rs if r["gate_travel_at_escape"]]
            travel_esc[name] = round(float(np.mean(tw)), 2) if tw else None
    headline = "escape step at N=8 (paired seeds): " + ", ".join(
        f"{a['name']} {escapes.get(a['name'], '?')}" for a in P["arms"] if a["init_key"] == "dense")
    make_chart(runs, P, headline)

    results = {
        "id": cfg.get("id", "unknown"),
        "git_commit": git_sha(),
        "seed": seed,
        "duration_sec": round(time.time() - t0, 2),
        "metrics": {"runs": runs, "mean_acc_N8": mean_acc,
                    "escape_steps_N8": escapes, "n_escaped": n_esc,
                    "gate_W_travel_at_escape_mean": travel_esc},
        "env": env_info(),
        "status": "done",
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(headline)


if __name__ == "__main__":
    main()
