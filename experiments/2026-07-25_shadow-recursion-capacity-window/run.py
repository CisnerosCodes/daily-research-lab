"""Does the Game-of-Life capacity window transfer to language modelling?

Transfer test of registry id 2026-07-21_gol-depth-recursion, which found that a weight-tied
recurrent cell beat an untied per-step stack at hidden width H=4 but not at H=2 or H=24 --
a "capacity window" for weight tying.

Here: a tiny char-level LM on tiny-shakespeare. Fixed loop count k=4.
  tied   = ONE transformer block applied 4 times
  untied = FOUR distinct transformer blocks
Both have identical compute (same number of block applications); untied has ~3-4x the parameters.
Sweep model width d and measure val loss in bits/char; the claim under test is that
delta = bits/char(tied) - bits/char(untied) is negative (tied wins) only at intermediate d.

Deterministic, CPU-only, single-threaded. Writes results.json + chart.png.
Usage:  python run.py
"""
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent


def set_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    import torch
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


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)  # 2 shared cores on this box; keep runs coexisting

LN2 = math.log(2.0)


# ----------------------------- data ----------------------------------------
def load_data(cfg):
    """Character-level tiny-shakespeare. Returns train/val uint16 arrays + vocab size."""
    name = cfg["dataset"]["local_file"]
    path = HERE / "data" / name          # corpora live in data/ (gitignored, keeps the repo tiny)
    if not path.exists() and (HERE / name).exists():
        path = HERE / name
    if not path.exists():                # fetch once on a fresh clone
        import urllib.request
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(cfg["dataset"]["source"], path)
    text = path.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in text], dtype=np.uint16)
    n_val = int(len(ids) * cfg["params"]["val_frac"])
    return ids[: len(ids) - n_val], ids[len(ids) - n_val:], len(chars)


def make_batches(data, n_batches, batch_size, block_size, gen):
    """Deterministic list of (x, y) batches drawn with the supplied torch.Generator."""
    out = []
    hi = len(data) - block_size - 1
    for _ in range(n_batches):
        ix = torch.randint(hi, (batch_size,), generator=gen)
        x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack(
            [torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix]
        )
        out.append((x, y))
    return out


# ----------------------------- model ---------------------------------------
class Block(nn.Module):
    """Standard pre-norm decoder block."""

    def __init__(self, d, n_heads, ffn_mult):
        super().__init__()
        self.h, self.dh = n_heads, d // n_heads
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.fc1 = nn.Linear(d, ffn_mult * d)
        self.fc2 = nn.Linear(ffn_mult * d, d)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class LoopedLM(nn.Module):
    """k block applications. tied=True -> one shared block; tied=False -> k distinct blocks."""

    def __init__(self, vocab, d, n_heads, ffn_mult, k, block_size, tied):
        super().__init__()
        self.k, self.tied = k, tied
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(block_size, d)
        self.blocks = nn.ModuleList(
            [Block(d, n_heads, ffn_mult) for _ in range(1 if tied else k)]
        )
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for i in range(self.k):
            x = self.blocks[0 if self.tied else i](x)
        return self.head(self.lnf(x))


# ----------------------------- train / eval --------------------------------
@torch.no_grad()
def evaluate(model, batches):
    model.eval()
    tot, n = 0.0, 0
    for x, y in batches:
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        tot += loss.item() * y.numel()
        n += y.numel()
    model.train()
    return (tot / n) / LN2  # bits per character


def lr_at(step, peak, p):
    warm = p["warmup_steps"]
    if step < warm:
        return peak * (step + 1) / warm
    prog = (step - warm) / max(1, p["steps"] - warm)
    return peak * (p["lr_final_frac"] + (1 - p["lr_final_frac"]) * 0.5 * (1 + math.cos(math.pi * prog)))


def run_one(d, tied, seed, cfg, train_data, vocab, val_curve, val_final):
    p = cfg["params"]
    set_seeds(seed * 1000 + d + (7 if tied else 0))  # distinct init per config, reproducible

    n_heads = max(p["n_heads_min"], d // 32)
    model = LoopedLM(vocab, d, n_heads, p["ffn_mult"], p["loops"], p["block_size"], tied)
    n_params = sum(q.numel() for q in model.parameters())
    n_block_params = sum(q.numel() for q in model.blocks.parameters())

    peak_lr = min(p["lr_cap"], p["lr_base"] * math.sqrt(64.0 / d))
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=p["weight_decay"])

    # IDENTICAL training batch stream for every (width, variant) at a given seed -> paired comparison
    gen = torch.Generator().manual_seed(seed)
    batches = make_batches(train_data, p["steps"], p["batch_size"], p["block_size"], gen)

    curve = []
    t0 = time.time()
    for step, (x, y) in enumerate(batches):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, peak_lr, p)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), p["grad_clip"])
        opt.step()
        if (step + 1) % p["eval_every"] == 0 and (step + 1) < p["steps"]:
            curve.append({"step": step + 1, "val_bpc": evaluate(model, val_curve)})

    final_bpc = evaluate(model, val_final)
    curve.append({"step": p["steps"], "val_bpc": final_bpc})
    train_bpc = evaluate(model, batches[-p["val_batches_curve"]:])
    dt = time.time() - t0

    return {
        "d": d,
        "tied": tied,
        "seed": seed,
        "n_params": n_params,
        "n_block_params": n_block_params,
        "peak_lr": peak_lr,
        "val_bpc": final_bpc,
        "train_bpc_last": train_bpc,
        "curve": curve,
        "seconds": round(dt, 1),
    }


# ----------------------------- main ----------------------------------------
def main():
    cfg = load_config()
    p = cfg["params"]
    t_start = time.time()

    train_data, val_data, vocab = load_data(cfg)

    # one fixed val set, shared by every run
    vgen = torch.Generator().manual_seed(1234)
    val_final = make_batches(val_data, p["val_batches_final"], p["batch_size"], p["block_size"], vgen)
    val_curve = val_final[: p["val_batches_curve"]]

    runs = []
    for d in p["widths"]:
        for tied in (True, False):
            for seed in p["seeds"]:
                r = run_one(d, tied, seed, cfg, train_data, vocab, val_curve, val_final)
                runs.append(r)
                print(
                    f"d={d:4d} {'tied  ' if tied else 'untied'} seed={seed} "
                    f"params={r['n_params']:>7d} val={r['val_bpc']:.4f} bpc  ({r['seconds']}s)",
                    flush=True,
                )

    # ---- aggregate: delta = tied - untied (negative means tied wins) ----
    by_width = {}
    for d in p["widths"]:
        t = [r["val_bpc"] for r in runs if r["d"] == d and r["tied"]]
        u = [r["val_bpc"] for r in runs if r["d"] == d and not r["tied"]]
        per_seed = [ti - ui for ti, ui in zip(t, u)]  # seeds are in the same order
        by_width[d] = {
            "tied_bpc_mean": float(np.mean(t)),
            "tied_bpc_per_seed": t,
            "untied_bpc_mean": float(np.mean(u)),
            "untied_bpc_per_seed": u,
            "delta_mean": float(np.mean(per_seed)),
            "delta_per_seed": per_seed,
            "delta_sign_consistent": bool(all(x < 0 for x in per_seed) or all(x > 0 for x in per_seed)),
            "tied_params": [r["n_params"] for r in runs if r["d"] == d and r["tied"]][0],
            "untied_params": [r["n_params"] for r in runs if r["d"] == d and not r["tied"]][0],
        }

    widths = list(p["widths"])
    deltas = [by_width[d]["delta_mean"] for d in widths]
    tied_wins = [d for d in widths if by_width[d]["delta_mean"] < 0]
    # "capacity window" = tied wins at some intermediate width but NOT at the narrowest and
    # NOT at the widest width tested.
    window = len(tied_wins) > 0 and widths[0] not in tied_wins and widths[-1] not in tied_wins
    monotone_in_width = all(deltas[i] <= deltas[i + 1] for i in range(len(deltas) - 1)) or all(
        deltas[i] >= deltas[i + 1] for i in range(len(deltas) - 1)
    )

    results = {
        "id": cfg["id"],
        "title": cfg["title"],
        "git_sha": git_sha(),
        "env": env_info(),
        "config": cfg,
        "vocab_size": vocab,
        "n_train_chars": int(len(train_data)),
        "n_val_chars": int(len(val_data)),
        "train_chars_seen_per_run": p["steps"] * p["batch_size"] * p["block_size"],
        "runs": runs,
        "by_width": {str(k): v for k, v in by_width.items()},
        "analysis": {
            "delta_by_width": {str(d): by_width[d]["delta_mean"] for d in widths},
            "widths_where_tied_wins": tied_wins,
            "capacity_window_found": bool(window),
            "delta_monotone_in_width": bool(monotone_in_width),
            "hypothesis_supported": bool(window),
        },
        "total_seconds": round(time.time() - t_start, 1),
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ----------------------------- chart -----------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    xs = np.arange(len(widths))

    a = ax[0]
    a.plot(xs, [by_width[d]["tied_bpc_mean"] for d in widths], "o-", color="#1f77b4",
           label="tied (1 block x4)")
    a.plot(xs, [by_width[d]["untied_bpc_mean"] for d in widths], "s-", color="#d62728",
           label="untied (4 blocks)")
    for i, d in enumerate(widths):
        a.scatter([i] * len(by_width[d]["tied_bpc_per_seed"]), by_width[d]["tied_bpc_per_seed"],
                  color="#1f77b4", alpha=0.35, s=18, zorder=3)
        a.scatter([i] * len(by_width[d]["untied_bpc_per_seed"]), by_width[d]["untied_bpc_per_seed"],
                  color="#d62728", alpha=0.35, s=18, zorder=3)
    a.set_xticks(xs)
    a.set_xticklabels([str(d) for d in widths])
    a.set_xlabel("model width d")
    a.set_ylabel("val loss (bits/char)")
    a.set_title("Validation loss vs width")
    a.legend()
    a.grid(alpha=0.3)

    b = ax[1]
    cols = ["#1f77b4" if v < 0 else "#d62728" for v in deltas]
    b.bar(xs, deltas, color=cols, alpha=0.85)
    for i, d in enumerate(widths):
        b.scatter([i] * len(by_width[d]["delta_per_seed"]), by_width[d]["delta_per_seed"],
                  color="k", s=20, zorder=3)
    b.axhline(0, color="k", lw=1)
    b.set_xticks(xs)
    b.set_xticklabels([str(d) for d in widths])
    b.set_xlabel("model width d")
    b.set_ylabel("bits/char  (tied - untied)")
    b.set_title("tied minus untied\n(below 0 = tied wins)")
    b.grid(alpha=0.3, axis="y")

    c = ax[2]
    cmap = plt.get_cmap("viridis")
    for i, d in enumerate(widths):
        col = cmap(i / max(1, len(widths) - 1))
        for tied, ls in ((True, "-"), (False, "--")):
            cs = [r["curve"] for r in runs if r["d"] == d and r["tied"] == tied]
            steps = [pt["step"] for pt in cs[0]]
            mean = np.mean([[pt["val_bpc"] for pt in cv] for cv in cs], axis=0)
            c.plot(steps, mean, ls, color=col, label=f"d={d} {'tied' if tied else 'untied'}")
    c.set_xlabel("training step")
    c.set_ylabel("val loss (bits/char)")
    c.set_title("Training curves (solid=tied, dashed=untied)")
    c.legend(fontsize=6, ncol=2)
    c.grid(alpha=0.3)

    fig.suptitle("Tied vs untied k=4 recursion across width, tiny-shakespeare char LM", y=1.02)
    fig.tight_layout()
    fig.savefig(HERE / "chart.png", dpi=140, bbox_inches="tight")

    print("\n--- summary ---")
    for d in widths:
        w = by_width[d]
        print(f"d={d:4d}  tied {w['tied_bpc_mean']:.4f}  untied {w['untied_bpc_mean']:.4f}  "
              f"delta {w['delta_mean']:+.4f}  sign-consistent={w['delta_sign_consistent']}")
    print("capacity_window_found:", window)
    print("total seconds:", results["total_seconds"])


if __name__ == "__main__":
    main()
