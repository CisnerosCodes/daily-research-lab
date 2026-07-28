"""MQAR minimum-selectivity gate-rank sweep. Deterministic, CPU-only, writes results.json + chart.png.

Task (byte-identical generators to 2026-07-26_mqar-state-capacity):
  sequence = [k1 v1 k2 v2 ... kN vN | q1 q2 ... qN]  (queries are the N keys, permuted)
  loss/accuracy only on the N query positions, predicting the paired value.

Fixed skeleton (2-block pre-norm transformer, 2 heads, 2x MLP, d=64) around ONE varying part:
the forget gate of an elu+1 linear-attention mixer. The selectivity axis, cheap to expensive:

  none    g = 1                               (vanilla linear attention)
  static  g_c = sigmoid(b_c)                  (learned per-channel decay, input-INDEPENDENT)
  scalar  g = sigmoid(w.x + b) per head       (2026-07-26 'gla' arm, replicated; broadcast over channels)
  rank1   g_c = sigmoid(b_c + [U(Vx)]_c)      (input-dependent per-channel, rank-1 bottleneck)
  rank4   same, rank-4 bottleneck
  dense   g_c = sigmoid(b_c + [Wx]_c)         (full-rank input-dependent per-channel; GLA-style)

All gated arms share one code path: a module produces per-channel log-decay logg (B,h,T,dh),
A = cumsum(logg), and the mixer computes exact decay-masked linear attention
  scores[t,s] = sum_c q[t,c] * exp(A_t[c] - A_s[c]) * k[s,c]   (clamped at 0, causal, row-normalized)
which is the closed form of S_t = diag(g_t) S_{t-1} + phi(k_t) v_t^T.
Gate init: weights zero (U zero for low-rank), bias +3 => every arm starts at g ~ 0.953, near-vanilla.

Cells: N=8 x seeds{0,1} x all arms (decisive cell: vanilla scores ~0.17 there, twice replicated);
N=4 x seed 0 (sanity); N=16 x seed 0 only for arms that solved N=8, plus the 'none' anchor.
Accuracy trajectory recorded at every eval for plateau analysis (lesson of 2026-07-27).

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
def build_model(arm, d_model, n_layers, n_heads, mlp_exp, vocab, max_len, gate_bias):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    h, dh = n_heads, d_model // n_heads

    class Gate(nn.Module):
        """Returns per-channel log-decay logg of shape (B, h, T, dh), or None for arm 'none'."""
        def __init__(self):
            super().__init__()
            self.arm = arm
            if arm == "none":
                pass
            elif arm == "static":
                self.bias = nn.Parameter(torch.full((h * dh,), gate_bias))
            elif arm == "scalar":
                self.lin = nn.Linear(d_model, h, bias=True)
                nn.init.zeros_(self.lin.weight)
                nn.init.constant_(self.lin.bias, gate_bias)
            elif arm == "dense":
                self.lin = nn.Linear(d_model, h * dh, bias=True)
                nn.init.zeros_(self.lin.weight)
                nn.init.constant_(self.lin.bias, gate_bias)
            elif arm.startswith("rank"):
                r = int(arm[4:])
                self.V = nn.Linear(d_model, r, bias=False)          # random init: carries input signal
                self.U = nn.Linear(r, h * dh, bias=True)
                nn.init.zeros_(self.U.weight)                        # zero init: starts input-independent
                nn.init.constant_(self.U.bias, gate_bias)
            else:
                raise ValueError(arm)

        def forward(self, x):  # x: (B, T, d_model)
            B, T, _ = x.shape
            if self.arm == "none":
                return None
            if self.arm == "static":
                logit = self.bias.view(1, 1, h * dh).expand(B, T, h * dh)
            elif self.arm == "scalar":
                logit = self.lin(x).unsqueeze(3).expand(B, T, h, dh).reshape(B, T, h * dh)
            else:  # dense / rankN
                logit = self.U(self.V(x)) if hasattr(self, "V") else self.lin(x)
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


def gate_params(model):
    return sum(p.numel() for n, p in model.named_parameters() if ".gate." in n)


def run_one(arm, N, seed, P):
    import torch
    d = P["d_model"]
    # deterministic init seed; sum(ord) not hash() (2026-07-27 fix: hash() varies across interpreters)
    torch.manual_seed(1_000_003 * seed + 101 * d + 13 * N + sum(ord(c) for c in arm) % 997)
    model = build_model(arm, d, P["n_layers"], P["n_heads"], P["mlp_expansion"],
                        P["key_vocab"] + P["val_vocab"], 3 * 16, P["gate_bias_init"])
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=P["lr"], weight_decay=P["weight_decay"])
    # same train stream + eval set across arms for a given (N, seed) - identical to prior harness
    gtrain = torch.Generator().manual_seed(500_000 + 100 * N + seed)
    geval = torch.Generator().manual_seed(900_000 + 100 * N + seed)
    xe, ye = make_batch(P["eval_sequences"], N, P["key_vocab"], P["val_vocab"], geval)

    t0, acc, step, traj = time.time(), 0.0, 0, []
    lossfn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    for step in range(1, P["train_steps"] + 1):
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
    return {"arm": arm, "num_pairs": N, "seed": seed, "acc": round(acc, 4), "steps": step,
            "params": n_params, "gate_params": gate_params(model), "traj": traj,
            "secs": round(time.time() - t0, 1)}


# ----------------------------------------------------------------------------- chart
def make_chart(runs, P, headline):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    arms = P["arms"]
    labels = {"none": "none\n(vanilla)", "static": "static\nper-chan", "scalar": "scalar\ninput-dep",
              "rank1": "rank-1\nper-chan", "rank4": "rank-4\nper-chan", "dense": "dense\nper-chan"}
    xpos = np.arange(len(arms))
    colors = {4: "#9ecae1", 8: "#3182bd", 16: "#08306b"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.0))
    # panel 1: acc vs selectivity arm, one line per N
    for N in sorted({r["num_pairs"] for r in runs}):
        means, have = [], []
        for i, a in enumerate(arms):
            accs = [r["acc"] for r in runs if r["arm"] == a and r["num_pairs"] == N]
            if accs:
                means.append(float(np.mean(accs))); have.append(i)
                for r in runs:
                    if r["arm"] == a and r["num_pairs"] == N:
                        ax1.plot(i, r["acc"], "o", ms=3.5, color=colors[N], alpha=0.45)
        ax1.plot(have, means, "o-", color=colors[N], label=f"N={N}")
    ax1.axhline(P["solve_threshold"], color="#999999", lw=0.8, ls="--")
    ax1.set_xticks(xpos, [labels[a] for a in arms], fontsize=8)
    ax1.set_ylim(-0.03, 1.05)
    ax1.set_xlabel("forget-gate parametrization (cheap → expressive)")
    ax1.set_ylabel("recall accuracy (query positions)")
    ax1.legend(frameon=False, fontsize=8, loc="center left")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_title(f"MQAR d={P['d_model']}, {P['train_steps']}-step budget", fontsize=10)

    # panel 2: N=8 trajectories (mean over seeds)
    cmap = {"none": "#888888", "static": "#b15928", "scalar": "#e31a1c",
            "rank1": "#33a02c", "rank4": "#1f78b4", "dense": "#6a3d9a"}
    for a in arms:
        trs = [r["traj"] for r in runs if r["arm"] == a and r["num_pairs"] == P["num_pairs_main"]]
        if not trs:
            continue
        L = min(len(t) for t in trs)
        xs = [trs[0][i][0] for i in range(L)]
        ys = [float(np.mean([t[i][1] for t in trs])) for i in range(L)]
        ax2.plot(xs, ys, "-", color=cmap[a], label=a, lw=1.6)
    ax2.axhline(P["solve_threshold"], color="#999999", lw=0.8, ls="--")
    ax2.set_ylim(-0.03, 1.05)
    ax2.set_xlabel("train step")
    ax2.set_ylabel("recall accuracy")
    ax2.legend(frameon=False, fontsize=8, ncol=2)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_title(f"training trajectories at N={P['num_pairs_main']} (mean of "
                  f"{len(P['seeds_main'])} seeds)", fontsize=10)

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
        P.update(seeds_main=[0], train_steps=100, eval_every=50, arms=["none", "rank1", "dense"])
    t0 = time.time()
    runs = []

    def do(arm, N, s):
        r = run_one(arm, N, s, P)
        runs.append(r)
        print(f"[{time.time()-t0:7.1f}s] {arm:7s} N={N:3d} seed={s} acc={r['acc']:.3f} "
              f"steps={r['steps']} gate_params={r['gate_params']} ({r['secs']}s)", flush=True)
        return r

    # phase 1: decisive cell N=8, all arms x seeds_main
    for arm in P["arms"]:
        for s in P["seeds_main"]:
            do(arm, P["num_pairs_main"], s)

    # phase 2: sanity N=4, all arms, seed 0
    for arm in P["arms"]:
        for s in P["seeds_extra"]:
            do(arm, P["num_pairs_sanity"], s)

    # phase 3: stretch N=16, seed 0, only arms that solved N=8 (mean >= threshold) + 'none' anchor
    import numpy as np
    solved = [a for a in P["arms"] if np.mean(
        [r["acc"] for r in runs if r["arm"] == a and r["num_pairs"] == P["num_pairs_main"]])
        >= P["solve_threshold"]]
    stretch_arms = (["none"] if "none" not in solved else []) + solved
    for arm in stretch_arms:
        for s in P["seeds_extra"]:
            do(arm, P["num_pairs_stretch"], s)

    # aggregates
    mean_acc = {}
    for a in P["arms"]:
        for N in (P["num_pairs_sanity"], P["num_pairs_main"], P["num_pairs_stretch"]):
            accs = [r["acc"] for r in runs if r["arm"] == a and r["num_pairs"] == N]
            if accs:
                mean_acc[f"{a}_N{N}"] = round(float(np.mean(accs)), 4)
    beats_vanilla = {a: round(mean_acc[f"{a}_N{P['num_pairs_main']}"]
                              - mean_acc[f"none_N{P['num_pairs_main']}"], 4)
                     for a in P["arms"] if f"{a}_N{P['num_pairs_main']}" in mean_acc}
    min_arm = next((a for a in P["arms"]
                    if mean_acc.get(f"{a}_N{P['num_pairs_main']}", 0) >= P["solve_threshold"]), None)
    headline = (f"Minimum selectivity solving N={P['num_pairs_main']} at d={P['d_model']} "
                f"within {P['train_steps']} steps: {min_arm or 'NONE (no arm solves it)'}")
    make_chart(runs, P, headline)

    results = {
        "id": cfg.get("id", "unknown"),
        "git_commit": git_sha(),
        "seed": seed,
        "duration_sec": round(time.time() - t0, 2),
        "metrics": {"runs": runs, "mean_acc": mean_acc,
                    "delta_vs_vanilla_N8": beats_vanilla,
                    "min_selectivity_arm_N8": min_arm,
                    "stretch_arms_run": stretch_arms},
        "env": env_info(),
        "status": "done",
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(headline)


if __name__ == "__main__":
    main()
