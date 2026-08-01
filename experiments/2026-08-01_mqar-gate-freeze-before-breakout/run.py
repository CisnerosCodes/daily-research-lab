"""MQAR gate freeze-before-breakout. Deterministic, CPU-only, writes results.json + chart.png.

Question (follow-up to 2026-07-29_mqar-gate-noise-control): freezing the dense per-channel
forget gate at step 1000 (post-breakout) cost nothing, so the gate's causal work is done by
step 1000. But is it done EARLIER? Freeze the gate at steps {0, 100, 250, 500, 750} - before
the breakout window (~750-1250 across seeds) - and see whether an early, partially-trained
gate suffices to open the door to the recall circuit, or whether gate learning must span the
whole pre-escape window.

Design upgrade over 2026-07-29: all dense-family arms (dense + every frozenK) share the SAME
init seed ("dense" enters the seed formula for all of them) and the same data stream, so within
a seed the runs are byte-identical up to the freeze step and any later divergence is caused by
the freeze alone. This is a paired comparison with zero init confound - the fix for the
2026-07-28 lesson that escape events are init-sensitive.

Also logged: the gate's distance from init (||W||_F, ||b - b0||) at freeze time and at the end,
so "how much gate learning had happened when we froze" is a measured quantity, not a guess.

Interpretation matrix:
  frozen750 ~ dense, frozen100 ~ vanilla  -> the gate's work happens in a mid-training window
  all frozenK >= 100 escape               -> a barely-trained gate already suffices (door is cheap)
  only dense escapes                      -> gate learning must span the whole pre-escape window
  frozenK escapes but LATE                -> the gate is rate-limiting: less gate -> slower door

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
    Byte-identical code path to 2026-07-28/29 (their 'none' and 'dense' arms)."""
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
    name, freeze, init_key = arm["name"], arm["freeze_step"], arm["init_key"]
    gated = init_key == "dense"
    # deterministic init seed; sum(ord) not hash() (2026-07-27 fix). CRUCIAL: the seed uses
    # init_key, not the arm name, so all dense-family arms are byte-identical per seed
    # (and 'dense'/'none' reproduce the 2026-07-28/29 rows exactly).
    torch.manual_seed(1_000_003 * seed + 101 * d + 13 * N + sum(ord(c) for c in init_key) % 997)
    model = build_model(gated, d, P["n_layers"], P["n_heads"], P["mlp_expansion"],
                        P["key_vocab"] + P["val_vocab"], 3 * 16, P["gate_bias_init"])
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=P["lr"], weight_decay=P["weight_decay"])
    # same train stream + eval set across arms for a given (N, seed) - identical to prior harness
    gtrain = torch.Generator().manual_seed(500_000 + 100 * N + seed)
    geval = torch.Generator().manual_seed(900_000 + 100 * N + seed)
    xe, ye = make_batch(P["eval_sequences"], N, P["key_vocab"], P["val_vocab"], geval)

    t0, acc, step, traj = time.time(), 0.0, 0, []
    travel_at_freeze = None
    lossfn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    for step in range(1, P["train_steps"] + 1):
        if freeze is not None and step == freeze + 1:
            for n_, p_ in model.named_parameters():
                if ".gate." in n_:
                    p_.requires_grad_(False)
            travel_at_freeze = gate_travel(model, P["gate_bias_init"])
            # grads are zeroed (set_to_none) every step below, so AdamW skips frozen params
        x, y = make_batch(P["batch_size"], N, P["key_vocab"], P["val_vocab"], gtrain)
        logits = model(x)
        loss = lossfn(logits.reshape(-1, logits.shape[2]), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % P["eval_every"] == 0:
            acc = evaluate(model, xe, ye, P["eval_chunk"])
            traj.append([step, round(acc, 4)])
            if acc >= P["early_stop_acc"]:
                break
    if step % P["eval_every"] != 0 and acc < P["early_stop_acc"]:
        acc = evaluate(model, xe, ye, P["eval_chunk"])
        traj.append([step, round(acc, 4)])
    escape = next((s for s, a in traj if a >= P["escape_threshold"]), None)
    travel_end = gate_travel(model, P["gate_bias_init"]) if gated else None
    return {"arm": name, "freeze_step": freeze, "num_pairs": N, "seed": seed,
            "acc": round(acc, 4), "steps": step, "escape_step": escape, "params": n_params,
            "gate_travel_at_freeze": travel_at_freeze, "gate_travel_end": travel_end,
            "traj": traj, "secs": round(time.time() - t0, 1)}


# ----------------------------------------------------------------------------- chart
def make_chart(runs, P, headline):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    Nm = P["num_pairs_main"]
    fam = [a["name"] for a in P["arms"] if a["init_key"] == "dense"]
    # x position for each dense-family arm: its freeze step; dense plotted at train_steps
    xof = {}
    for a in P["arms"]:
        if a["init_key"] != "dense":
            continue
        xof[a["name"]] = P["train_steps"] if a["freeze_step"] is None else a["freeze_step"]
    cmap = {"frozen0": "#bbbbbb", "frozen100": "#e31a1c", "frozen250": "#ff7f00",
            "frozen500": "#f2c020", "frozen750": "#33a02c", "dense": "#6a3d9a",
            "none": "#888888"}
    mstyle = {0: "-", 1: "--", 2: ":"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))

    # panel 1: endpoint acc vs freeze step, one line per seed (paired: same init per seed)
    seeds_all = sorted({r["seed"] for r in runs if r["arm"] in fam})
    for s in seeds_all:
        xs, ys = [], []
        for a in fam:
            rr = [r for r in runs if r["arm"] == a and r["seed"] == s and r["num_pairs"] == Nm]
            if rr:
                xs.append(xof[a]); ys.append(rr[0]["acc"])
        order = np.argsort(xs)
        ax1.plot(np.array(xs)[order], np.array(ys)[order], marker="o", ms=5,
                 lw=1.5, ls=mstyle.get(s, "-"), color="#444444", alpha=0.85,
                 label=f"seed {s}")
    van = [r["acc"] for r in runs if r["arm"] == "none" and r["num_pairs"] == Nm]
    if van:
        ax1.axhline(float(np.mean(van)), color="#888888", lw=1.0, ls="--")
        ax1.text(0.02, float(np.mean(van)) + 0.02, "vanilla plateau", fontsize=7.5,
                 color="#666666", transform=ax1.get_yaxis_transform())
    ax1.axhline(P["escape_threshold"], color="#bbbbbb", lw=0.8, ls=":")
    ax1.set_xlabel("gate frozen after this many steps (dense = never, plotted at 2000)")
    ax1.set_ylabel(f"recall accuracy at N={Nm}, {P['train_steps']} steps")
    ax1.set_ylim(0, 1.05)
    ax1.legend(frameon=False, fontsize=8)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_title("endpoint vs freeze step (paired inits per seed)", fontsize=10)

    # panel 2: trajectories at N=8, color = arm, linestyle = seed
    for a in fam + ["none"]:
        rs = [r for r in runs if r["arm"] == a and r["num_pairs"] == Nm]
        for r in rs:
            xs = [s for s, _ in r["traj"]]
            ys = [v for _, v in r["traj"]]
            ax2.plot(xs, ys, ls=mstyle.get(r["seed"], "-"), color=cmap.get(a, "#1f78b4"),
                     label=a if r["seed"] == seeds_all[0] else None, lw=1.5)
    for a in fam:
        if xof[a] < P["train_steps"]:
            ax2.axvline(xof[a], color=cmap.get(a, "#1f78b4"), lw=0.7, ls=":", alpha=0.6)
    ax2.axhline(P["escape_threshold"], color="#999999", lw=0.8, ls="--")
    ax2.set_ylim(-0.03, 1.05)
    ax2.set_xlabel("train step (dotted verticals = freeze points)")
    ax2.set_ylabel("recall accuracy")
    ax2.legend(frameon=False, fontsize=7.5, ncol=2)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_title("trajectories (solid s0, dashed s1, dotted s2)", fontsize=10)

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
                 arms=[{"name": "none", "freeze_step": None, "init_key": "none", "seeds": [0]},
                       {"name": "frozen50", "freeze_step": 50, "init_key": "dense", "seeds": [0]},
                       {"name": "dense", "freeze_step": None, "init_key": "dense", "seeds": [0]}])
    t0 = time.time()
    runs = []

    for arm in P["arms"]:
        for s in arm["seeds"]:
            r = run_one(arm, P["num_pairs_main"], s, P)
            runs.append(r)
            print(f"[{time.time()-t0:7.1f}s] {r['arm']:10s} seed={s} acc={r['acc']:.3f} "
                  f"esc={r['escape_step']} steps={r['steps']} "
                  f"travel@freeze={r['gate_travel_at_freeze']} travel@end={r['gate_travel_end']} "
                  f"({r['secs']}s)", flush=True)

    # aggregates
    import numpy as np
    Nm = P["num_pairs_main"]
    mean_acc, escapes, n_esc = {}, {}, {}
    for a in P["arms"]:
        name = a["name"]
        rs = [r for r in runs if r["arm"] == name and r["num_pairs"] == Nm]
        if rs:
            mean_acc[name] = round(float(np.mean([r["acc"] for r in rs])), 4)
            escapes[name] = [r["escape_step"] for r in rs]
            n_esc[name] = sum(e is not None for e in escapes[name])
    headline = "escapes/seeds at N=8: " + ", ".join(
        f"{a['name']} {n_esc.get(a['name'], 0)}/{len(a['seeds'])}"
        for a in P["arms"] if a["name"] != "frozen0") + \
        (f", frozen0 {n_esc.get('frozen0', 0)}/1" if "frozen0" in mean_acc else "")
    make_chart(runs, P, headline)

    results = {
        "id": cfg.get("id", "unknown"),
        "git_commit": git_sha(),
        "seed": seed,
        "duration_sec": round(time.time() - t0, 2),
        "metrics": {"runs": runs, "mean_acc_N8": mean_acc,
                    "escape_steps_N8": escapes, "n_escaped": n_esc},
        "env": env_info(),
        "status": "done",
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(headline)


if __name__ == "__main__":
    main()
