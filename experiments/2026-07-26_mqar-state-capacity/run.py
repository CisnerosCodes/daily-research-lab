"""MQAR state-capacity sweep. Deterministic, CPU-only, writes results.json + chart.png.

Task (zoology-style multi-query associative recall):
  sequence = [k1 v1 k2 v2 ... kN vN | q1 q2 ... qN]  (queries are the N keys, permuted)
  loss/accuracy only on the N query positions, predicting the paired value.

Three sequence mixers, identical 2-block pre-norm transformer skeleton around them:
  attn    - causal softmax multi-head attention (quadratic, no fixed state)
  linattn - causal linear attention, feature map elu+1 (per-head state = d_head x d_head)
  gla     - same linear attention + ONE input-dependent scalar forget gate per head
            (the cheapest possible "selectivity"; gate bias init +3 so it starts ~vanilla)

Both linear mixers are computed in closed form as decay-masked attention:
  S_t = g_t * S_{t-1} + phi(k_t) v_t^T  =>  out_t = sum_s exp(A_t - A_s) (phi(q_t).phi(k_s)) v_s / Z
with A_t = cumsum(log g_t) (A=0 for linattn), which is exact and fully vectorized.

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
    # N distinct keys per sequence
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

    class LinAttn(nn.Module):
        """Linear attention; gated=True adds one input-dependent scalar forget gate/head."""
        def __init__(self, gated):
            super().__init__()
            self.gated = gated
            self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
            self.out = nn.Linear(d_model, d_model, bias=False)
            if gated:
                self.gate = nn.Linear(d_model, n_heads, bias=True)
                nn.init.zeros_(self.gate.weight)
                nn.init.constant_(self.gate.bias, 3.0)  # start ~vanilla (g~0.95)

        def forward(self, x):
            B, T, _ = x.shape
            h, dh = n_heads, d_model // n_heads
            q, k, v = self.qkv(x).split(d_model, dim=2)
            q, k, v = (t.view(B, T, h, dh).transpose(1, 2) for t in (q, k, v))
            q, k = F.elu(q) + 1, F.elu(k) + 1
            if self.gated:
                logg = F.logsigmoid(self.gate(x)).transpose(1, 2)   # (B,h,T)
                A = torch.cumsum(logg, dim=2)                        # (B,h,T)
                D = (A.unsqueeze(3) - A.unsqueeze(2)).clamp(max=0.0).exp()  # (B,h,T,T)
            else:
                D = x.new_ones(1, 1, T, T)
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            scores = (q @ k.transpose(2, 3)) * D
            scores = scores.masked_fill(~mask, 0.0)
            z = scores.sum(dim=3, keepdim=True) + 1e-6
            return self.out(((scores / z) @ v).transpose(1, 2).reshape(B, T, d_model))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1 = nn.LayerNorm(d_model)
            self.mix = SoftmaxAttn() if mixer == "attn" else LinAttn(gated=(mixer == "gla"))
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

    import torch.nn.functional as F  # noqa: F811  (needed in closures above)
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


def run_one(mixer, N, d, seed, P):
    import torch
    torch.manual_seed(1_000_003 * seed + 101 * d + 13 * N + hash(mixer) % 997)
    model = build_model(mixer, d, P["n_layers"], P["n_heads"], P["mlp_expansion"],
                        P["key_vocab"] + P["val_vocab"], 3 * max(P["num_pairs"]))
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=P["lr"], weight_decay=P["weight_decay"])
    # same train stream + eval set across mixers/widths for a given (N, seed)
    gtrain = torch.Generator().manual_seed(500_000 + 100 * N + seed)
    geval = torch.Generator().manual_seed(900_000 + 100 * N + seed)
    xe, ye = make_batch(P["eval_sequences"], N, P["key_vocab"], P["val_vocab"], geval)

    t0, acc, step = time.time(), 0.0, 0
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
            if acc >= P["early_stop_acc"]:
                break
    if step % P["eval_every"] != 0 and acc < P["early_stop_acc"]:
        acc = evaluate(model, xe, ye, P["eval_chunk"])
    return {"mixer": mixer, "num_pairs": N, "d_model": d, "seed": seed,
            "acc": round(acc, 4), "steps": step, "params": n_params,
            "secs": round(time.time() - t0, 1)}


# ----------------------------------------------------------------------------- chart
def make_chart(runs, P, headline):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    mixers = P["mixers"]
    dims = P["d_model"]
    pairs = P["num_pairs"]
    names = {"attn": "softmax attention", "linattn": "linear attention",
             "gla": "linear attn + scalar forget gate"}
    colors = {32: "#7bb0df", 64: "#3d7dbf", 128: "#0f3f6e"}

    fig, axes = plt.subplots(1, len(mixers), figsize=(11.5, 3.6), sharey=True)
    for ax, mixer in zip(axes, mixers):
        for d in dims:
            ys = []
            for N in pairs:
                accs = [r["acc"] for r in runs
                        if r["mixer"] == mixer and r["d_model"] == d and r["num_pairs"] == N]
                ys.append(np.mean(accs))
            ax.plot(pairs, ys, "o-", color=colors[d], label=f"d={d}")
        ax.axhline(P["solve_threshold"], color="#999999", lw=0.8, ls="--")
        ax.set_xscale("log", base=2)
        ax.set_xticks(pairs, [str(p) for p in pairs])
        ax.set_ylim(-0.03, 1.05)
        ax.set_title(names[mixer], fontsize=10)
        ax.set_xlabel("key-value pairs N")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("recall accuracy (query positions)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(headline, fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(HERE / "chart.png", dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    cfg = load_config()
    P = dict(cfg["params"])
    seed = int(cfg.get("seed", 0))
    set_seeds(seed)
    if os.environ.get("MQAR_PILOT"):
        P.update(num_pairs=[8], d_model=[64], seeds=[0], train_steps=100, eval_every=50)
    t0 = time.time()

    runs = []
    for mixer in P["mixers"]:
        for d in P["d_model"]:
            for N in P["num_pairs"]:
                for s in P["seeds"]:
                    r = run_one(mixer, N, d, s, P)
                    runs.append(r)
                    print(f"[{time.time()-t0:7.1f}s] {mixer:8s} d={d:4d} N={N:3d} "
                          f"seed={s} acc={r['acc']:.3f} steps={r['steps']} "
                          f"({r['secs']}s, {r['params']/1e3:.0f}k params)", flush=True)

    # aggregates: mean acc per cell, capacity frontier per (mixer, d)
    import numpy as np
    mean_acc, frontier = {}, {}
    for mixer in P["mixers"]:
        for d in P["d_model"]:
            best_N = 0
            for N in P["num_pairs"]:
                accs = [r["acc"] for r in runs
                        if r["mixer"] == mixer and r["d_model"] == d and r["num_pairs"] == N]
                m = float(np.mean(accs)) if accs else float("nan")
                mean_acc[f"{mixer}_d{d}_N{N}"] = round(m, 4)
                if m >= P["solve_threshold"]:
                    best_N = max(best_N, N)
            frontier[f"{mixer}_d{d}"] = best_N

    headline = ("Capacity frontier (max N with acc>=0.9): "
                + "  ".join(f"{mx}: " + "/".join(str(frontier[f'{mx}_d{d}']) for d in P["d_model"])
                            for mx in P["mixers"])
                + f"   (d={'/'.join(str(d) for d in P['d_model'])})")
    make_chart(runs, P, headline)

    results = {
        "id": cfg.get("id", "unknown"),
        "git_commit": git_sha(),
        "seed": seed,
        "duration_sec": round(time.time() - t0, 2),
        "metrics": {"runs": runs, "mean_acc": mean_acc,
                    "capacity_frontier_maxN_at_0.9": frontier},
        "env": env_info(),
        "status": "done",
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(headline)


if __name__ == "__main__":
    main()
