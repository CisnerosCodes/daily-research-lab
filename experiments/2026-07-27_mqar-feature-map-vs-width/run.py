"""MQAR: feature map vs width. Deterministic, CPU-only, writes results.json + chart.png.

Follow-up to 2026-07-26_mqar-state-capacity, which found the elu+1 linear-attention
recall cliff PINNED at N~8 across a 4x d_head sweep at a fixed 2000-step budget.
That left three confounded suspects: state size, feature map, optimization budget.
This experiment separates the last two:

  Phase 1 (feature map, identical budget): same 2-block skeleton, three mixers
    attn    - causal softmax attention (reference)
    linattn - linear attention, elu+1 feature map (exact rerun arm, extended to N=32)
    taylor  - linear attention, BASED-style 2nd-order Taylor-exp kernel:
              score(q,k) = 1 + s + s^2/2,  s = (q.k)/sqrt(d_head)
              (kernel-trick form of phi(x) = [1, x, x xotimes x/sqrt(2)]; the implied
              per-head state is (1 + d_head + d_head^2) x d_head, so if the cliff is a
              state/feature-map limit it should now MOVE with width)
    All linear scores are non-negative (min of 1+s+s^2/2 is 0.5) and normalized by
    their causal sum, exactly like the elu+1 arm.

  Phase 2 (budget): the failing elu+1 cells (N=8 at d=32/64/128) get a 10x step
    budget (20000 steps) with an accuracy trajectory, to separate cannot-represent
    from slow-to-learn.

Usage:  python run.py            (full grid)
        MQAR_PILOT=1 python run.py   (tiny grid, timing sanity check)
"""
import json, os, random, subprocess, sys, time
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
def build_model(mixer, d_model, n_layers, n_heads, mlp_exp, vocab, max_len):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SoftmaxAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
            self.out = nn.Linear(d_model, d_model, bias=False)

        def forward(self, x):
            B, T, _ = x.shape
            h, dh = n_heads, d_model // n_heads
            q, k, v = self.qkv(x).split(d_model, dim=2)
            q, k, v = (t.view(B, T, h, dh).transpose(1, 2) for t in (q, k, v))
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.out(o.transpose(1, 2).reshape(B, T, d_model))

    class KernelAttn(nn.Module):
        """Causal linear attention with a swappable positive kernel.

        fmap='elu'    : scores = (elu(q)+1) @ (elu(k)+1)^T          (mqar-state-capacity arm)
        fmap='taylor' : s = (q @ k^T)/sqrt(d_head); scores = 1 + s + s^2/2
                        == phi(q).phi(k) for phi(x)=[1, x, x xotimes x / sqrt2] (BASED)
        Both are masked causally and normalized by the causal row-sum.
        """
        def __init__(self, fmap):
            super().__init__()
            self.fmap = fmap
            self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
            self.out = nn.Linear(d_model, d_model, bias=False)

        def forward(self, x):
            B, T, _ = x.shape
            h, dh = n_heads, d_model // n_heads
            q, k, v = self.qkv(x).split(d_model, dim=2)
            q, k, v = (t.view(B, T, h, dh).transpose(1, 2) for t in (q, k, v))
            if self.fmap == "elu":
                q, k = F.elu(q) + 1, F.elu(k) + 1
                scores = q @ k.transpose(2, 3)
            else:  # taylor
                s = (q @ k.transpose(2, 3)) / (dh ** 0.5)
                scores = 1.0 + s + 0.5 * s * s
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            scores = scores.masked_fill(~mask, 0.0)
            z = scores.sum(dim=3, keepdim=True) + 1e-6
            return self.out(((scores / z) @ v).transpose(1, 2).reshape(B, T, d_model))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1 = nn.LayerNorm(d_model)
            self.mix = (SoftmaxAttn() if mixer == "attn"
                        else KernelAttn("elu" if mixer == "linattn" else "taylor"))
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


def run_one(mixer, N, d, seed, P, train_steps, eval_every, track_curve=False):
    import torch
    # sum(ord) instead of hash(): PYTHONHASHSEED set in-process does not fix str hashing
    torch.manual_seed(1_000_003 * seed + 101 * d + 13 * N + sum(map(ord, mixer)) % 997)
    model = build_model(mixer, d, P["n_layers"], P["n_heads"], P["mlp_expansion"],
                        P["key_vocab"] + P["val_vocab"], 3 * max(P["num_pairs"]))
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=P["lr"], weight_decay=P["weight_decay"])
    # same train stream + eval set across mixers/widths for a given (N, seed)
    gtrain = torch.Generator().manual_seed(500_000 + 100 * N + seed)
    geval = torch.Generator().manual_seed(900_000 + 100 * N + seed)
    xe, ye = make_batch(P["eval_sequences"], N, P["key_vocab"], P["val_vocab"], geval)

    t0, acc, step = time.time(), 0.0, 0
    curve = []
    lossfn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    for step in range(1, train_steps + 1):
        x, y = make_batch(P["batch_size"], N, P["key_vocab"], P["val_vocab"], gtrain)
        logits = model(x)
        loss = lossfn(logits.reshape(-1, logits.shape[2]), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % eval_every == 0:
            acc = evaluate(model, xe, ye, P["eval_chunk"])
            if track_curve:
                curve.append([step, round(acc, 4)])
            if acc >= P["early_stop_acc"]:
                break
    if step % eval_every != 0 and acc < P["early_stop_acc"]:
        acc = evaluate(model, xe, ye, P["eval_chunk"])
        if track_curve:
            curve.append([step, round(acc, 4)])
    out = {"mixer": mixer, "num_pairs": N, "d_model": d, "seed": seed,
           "acc": round(acc, 4), "steps": step, "params": n_params,
           "secs": round(time.time() - t0, 1)}
    if track_curve:
        out["curve"] = curve
    return out


# ----------------------------------------------------------------------------- chart
def make_chart(runs, long_runs, P, headline):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    dims = P["d_model"]
    pairs = P["num_pairs"]
    names = {"attn": "softmax attention (reference, N=32 only)",
             "linattn": "linear attn (elu+1, anchor cells)",
             "taylor": "linear attn (Taylor-exp / BASED)"}
    colors = {32: "#7bb0df", 64: "#3d7dbf", 128: "#0f3f6e"}

    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.6))
    for ax, mixer in zip(axes[:3], ["linattn", "taylor", "attn"]):
        for d in dims:
            grid_N = P["grid"][mixer]
            ys = []
            for N in grid_N:
                accs = [r["acc"] for r in runs
                        if r["mixer"] == mixer and r["d_model"] == d and r["num_pairs"] == N]
                ys.append(np.mean(accs) if accs else np.nan)
            ax.plot(grid_N, ys, "o-", color=colors[d], label=f"d={d}")
        ax.axhline(P["solve_threshold"], color="#999999", lw=0.8, ls="--")
        ax.set_xscale("log", base=2)
        ax.set_xticks(pairs, [str(p) for p in pairs])
        ax.set_ylim(-0.03, 1.05)
        ax.set_title(names[mixer], fontsize=10)
        ax.set_xlabel("key-value pairs N")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("recall accuracy (query positions)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")

    ax = axes[3]
    for r in long_runs:
        d = r["d_model"]
        xs = [c[0] for c in r["curve"]]
        ys = [c[1] for c in r["curve"]]
        ax.plot(xs, ys, "-", color=colors[d], label=f"d={d}")
    ax.axhline(P["solve_threshold"], color="#999999", lw=0.8, ls="--")
    ax.axvline(P["train_steps"], color="#bb4444", lw=0.8, ls=":")
    ax.text(P["train_steps"] * 1.1, 0.02, "phase-1 budget", fontsize=7, color="#bb4444")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title(f"elu+1 at N={P['long_cells'][0][1]}, 10x budget", fontsize=10)
    ax.set_xlabel("train step")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(headline, fontsize=10.5)
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
        P.update(num_pairs=[8, 32], d_model=[64], seeds=[0], train_steps=100, eval_every=50,
                 grid={"taylor": [8, 32], "linattn": [8], "attn": [32]},
                 long_cells=[[64, 8]], long_train_steps=200, long_eval_every=50)
    t0 = time.time()

    runs = []
    for mixer in ["linattn", "taylor", "attn"]:
        for d in P["d_model"]:
            for N in P["grid"][mixer]:
                for s in P["seeds"]:
                    r = run_one(mixer, N, d, s, P, P["train_steps"], P["eval_every"])
                    runs.append(r)
                    print(f"[{time.time()-t0:7.1f}s] {mixer:8s} d={d:4d} N={N:3d} "
                          f"seed={s} acc={r['acc']:.3f} steps={r['steps']} "
                          f"({r['secs']}s, {r['params']/1e3:.0f}k params)", flush=True)

    long_runs = []
    for d, N in P["long_cells"]:
        r = run_one(P["long_mixer"], N, d, P["seeds"][0], P,
                    P["long_train_steps"], P["long_eval_every"], track_curve=True)
        long_runs.append(r)
        print(f"[{time.time()-t0:7.1f}s] LONG {P['long_mixer']:8s} d={d:4d} N={N:3d} "
              f"acc={r['acc']:.3f} steps={r['steps']} ({r['secs']}s)", flush=True)

    import numpy as np
    mean_acc, frontier = {}, {}
    for mixer in ["linattn", "taylor", "attn"]:
        for d in P["d_model"]:
            best_N = 0
            for N in P["grid"][mixer]:
                accs = [r["acc"] for r in runs
                        if r["mixer"] == mixer and r["d_model"] == d and r["num_pairs"] == N]
                m = float(np.mean(accs)) if accs else float("nan")
                mean_acc[f"{mixer}_d{d}_N{N}"] = round(m, 4)
                if m >= P["solve_threshold"]:
                    best_N = max(best_N, N)
            frontier[f"{mixer}_d{d}"] = best_N

    long_summary = {f"{P['long_mixer']}_d{r['d_model']}_N{r['num_pairs']}_long":
                    {"acc": r["acc"], "steps": r["steps"]} for r in long_runs}

    headline = ("Taylor frontier (max N, acc>=0.9) d=" + "/".join(str(d) for d in P["d_model"])
                + ": " + "/".join(str(frontier[f"taylor_d{d}"]) for d in P["d_model"])
                + " = d_head/4, moves with width (2026-07-26 elu+1: pinned 2/4/4)"
                + "   |   elu+1 N=8 at 10x budget: "
                + " ".join(f"d{r['d_model']}={r['acc']:.2f}" for r in long_runs))
    make_chart(runs, long_runs, P, headline)

    results = {
        "id": cfg.get("id", "unknown"),
        "git_commit": git_sha(),
        "seed": seed,
        "duration_sec": round(time.time() - t0, 2),
        "metrics": {"runs": runs, "long_runs": long_runs, "mean_acc": mean_acc,
                    "capacity_frontier_maxN_at_0.9": frontier,
                    "long_budget_summary": long_summary},
        "env": env_info(),
        "status": "done",
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(headline)


if __name__ == "__main__":
    main()
