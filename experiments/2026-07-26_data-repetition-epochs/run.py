"""Data repetition at fixed compute: where is the "~4 epochs is free" knee at 0.17M params?

Fixed TOTAL TOKEN budget (steps x batch x block, identical for every arm). What varies is how much
UNIQUE data those tokens are drawn from: a contiguous prefix of the train split of fraction
u in {1, 1/2, 1/4, 1/8, 1/16, 1/32}. The repetition count is R = budget_tokens / unique_tokens,
so R sweeps ~1 -> ~34 epochs. Val split is fixed and disjoint from every unique subset.

Deterministic, CPU-only, single-threaded. Writes results.json + chart.png.

Usage:  python run.py
"""
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
LN2 = math.log(2.0)


# ----------------------------------------------------------------------------- boilerplate
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
    info = {"python": sys.version.split()[0], "threads": torch.get_num_threads()}
    for mod in ("numpy", "torch", "matplotlib"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:
            pass
    return info


# ----------------------------------------------------------------------------- data
def get_text(source_url: str):
    data_dir = HERE / "data"
    data_dir.mkdir(exist_ok=True)
    path = data_dir / "tinyshakespeare.txt"
    if not path.exists():
        # the brief says curl; urlretrieve is the same fetch without shelling out
        try:
            urllib.request.urlretrieve(source_url, path)
        except Exception:
            # offline fallback: another experiment in this repo already cached the file
            for cand in sorted(HERE.parent.glob("*/data/tinyshakespeare.txt")):
                path.write_bytes(cand.read_bytes())
                break
            else:
                raise
    raw = path.read_bytes()
    return raw.decode("utf-8"), hashlib.md5(raw).hexdigest()


# ----------------------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, d, n_head, d_ff, block_size):
        super().__init__()
        self.n_head = n_head
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.fc1 = nn.Linear(d, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d, bias=False)
        self.register_buffer(
            "mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(C, dim=2)
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.proj(y)
        h = self.ln2(x)
        x = x + self.fc2(F.gelu(self.fc1(h)))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab, d, n_layer, n_head, d_ff, block_size):
        super().__init__()
        self.block_size = block_size
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(block_size, d)
        self.blocks = nn.ModuleList([Block(d, n_head, d_ff, block_size) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        p = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(p)[None]
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.lnf(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


# ----------------------------------------------------------------------------- eval
@torch.no_grad()
def eval_bpc(model, windows, batch=32):
    """windows: (N, block+1) int64. Returns bits per character."""
    model.eval()
    tot_nats, tot_tok = 0.0, 0
    for i in range(0, windows.size(0), batch):
        w = windows[i : i + batch]
        x, y = w[:, :-1], w[:, 1:]
        _, loss = model(x, y)
        n = y.numel()
        tot_nats += loss.item() * n
        tot_tok += n
    model.train()
    return tot_nats / tot_tok / LN2


def make_windows(arr, block, n_windows=None):
    """Non-overlapping windows of length block+1; evenly spaced subsample if n_windows is given."""
    stride = block + 1
    total = arr.shape[0] // stride
    idx = np.arange(total)
    if n_windows is not None and n_windows < total:
        idx = np.linspace(0, total - 1, n_windows).astype(np.int64)
    out = np.stack([arr[i * stride : i * stride + stride] for i in idx])
    return torch.from_numpy(out.astype(np.int64))


# ----------------------------------------------------------------------------- one run
def train_one(cp, vocab_size, train_pool, val_eval_w, val_curve_w, u, seed, log, mode="prefix"):
    block, batch, steps = cp["block_size"], cp["batch_size"], cp["steps"]
    stride = block + 1
    n_win_all = train_pool.shape[0] // stride

    if mode == "prefix":
        # the main sweep: a CONTIGUOUS prefix of the train split
        n_win = max(batch, int(round(u * n_win_all)))
        sel_idx = np.arange(n_win)
    elif mode == "spread":
        # control: same number of unique windows, but EVENLY SPREAD over the whole train split,
        # so "less unique data" is decoupled from "a narrower, more distant slice of the corpus"
        n_win = max(batch, int(round(u * n_win_all)))
        sel_idx = np.linspace(0, n_win_all - 1, n_win).astype(np.int64)
    else:
        raise ValueError(mode)

    win = np.stack([train_pool[i * stride : i * stride + stride] for i in sel_idx])
    win_t = torch.from_numpy(win.astype(np.int64))
    n_unique = n_win * stride
    assert n_win >= batch, f"unique subset too small: {n_win} windows"

    k = min(cp["eval_train_windows"], n_win)
    train_eval_w = win_t[np.linspace(0, n_win - 1, k).astype(np.int64)]

    set_seeds(seed)                                       # identical init across arms at equal seed
    model = TinyGPT(vocab_size, cp["d_model"], cp["n_layer"], cp["n_head"], cp["d_ff"], block)
    n_params = sum(p.numel() for p in model.parameters())

    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cp["weight_decay"]},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=cp["lr"], betas=(0.9, 0.95),
    )

    # epoch-based ordering: shuffle the unique windows, consume, reshuffle -> exact epoch semantics
    rng = np.random.default_rng(1000 + seed)
    order, cursor, epochs_done = rng.permutation(n_win), 0, 0
    curve = []
    t0 = time.time()
    for step in range(steps):
        if step < cp["warmup_steps"]:
            lr = cp["lr"] * (step + 1) / cp["warmup_steps"]
        else:
            prog = (step - cp["warmup_steps"]) / max(1, steps - cp["warmup_steps"])
            lr = cp["lr"] * (cp["min_lr_frac"]
                             + (1 - cp["min_lr_frac"]) * 0.5 * (1 + math.cos(math.pi * prog)))
        for g in opt.param_groups:
            g["lr"] = lr

        if cursor + batch > n_win:
            order = rng.permutation(n_win)
            cursor = 0
            epochs_done += 1
        sel = order[cursor : cursor + batch]
        cursor += batch
        b = win_t[torch.from_numpy(sel)]
        x, y = b[:, :-1], b[:, 1:]

        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cp["grad_clip"])
        opt.step()

        if (step + 1) % cp["eval_every"] == 0 or step == steps - 1:
            vb = eval_bpc(model, val_curve_w)
            curve.append({"step": step + 1, "train_loss_nats": round(loss.item(), 5),
                          "val_bpc_curve": round(vb, 5)})

    val_bpc = eval_bpc(model, val_eval_w)
    train_bpc = eval_bpc(model, train_eval_w)
    secs = time.time() - t0

    tokens_seen = steps * batch * block
    rec = {
        "unique_frac": u,
        "mode": mode,
        "seed": seed,
        "n_unique_chars": int(n_unique),
        "n_unique_windows": int(n_win),
        "tokens_seen": int(tokens_seen),
        "R_repetition": round(tokens_seen / n_unique, 4),
        "steps_per_epoch": round(n_win / batch, 3),
        "full_epochs_completed": int(epochs_done),
        "val_bpc": round(val_bpc, 5),
        "train_bpc": round(train_bpc, 5),
        "gap_bpc": round(val_bpc - train_bpc, 5),
        "val_bpc_best_ckpt": round(min(c["val_bpc_curve"] for c in curve), 5),
        "val_bpc_final_ckpt": curve[-1]["val_bpc_curve"],
        "n_params": n_params,
        "seconds": round(secs, 1),
        "curve": curve,
    }
    log(f"  [{mode:<6}] u={u:<8g} R={rec['R_repetition']:>6.2f}  unique={n_unique:>7d}ch  seed={seed}  "
        f"val={val_bpc:.4f}  train={train_bpc:.4f}  gap={rec['gap_bpc']:+.4f}  [{secs:.0f}s]")
    return rec


# ----------------------------------------------------------------------------- analysis
def analyse(runs, thresh_frac=0.02):
    by_u = {}
    for r in runs:
        by_u.setdefault(r["unique_frac"], []).append(r)
    us = sorted(by_u.keys(), reverse=True)  # u=1 first -> R ascending

    agg = []
    for u in us:
        rs = by_u[u]
        v = [r["val_bpc"] for r in rs]
        agg.append({
            "unique_frac": u,
            "R": round(float(np.mean([r["R_repetition"] for r in rs])), 4),
            "n_unique_chars": rs[0]["n_unique_chars"],
            "val_bpc_mean": round(float(np.mean(v)), 5),
            "val_bpc_per_seed": [round(x, 5) for x in v],
            "val_bpc_spread": round(float(max(v) - min(v)), 5),
            "train_bpc_mean": round(float(np.mean([r["train_bpc"] for r in rs])), 5),
            "gap_bpc_mean": round(float(np.mean([r["gap_bpc"] for r in rs])), 5),
            "val_bpc_best_ckpt_mean": round(float(np.mean([r["val_bpc_best_ckpt"] for r in rs])), 5),
        })

    ref = agg[0]["val_bpc_mean"]          # R ~ 1, the all-fresh reference
    thresh = ref * (1.0 + thresh_frac)
    for a in agg:
        a["rel_excess_vs_fresh"] = round((a["val_bpc_mean"] - ref) / ref, 5)
        a["within_2pct"] = bool(a["val_bpc_mean"] <= thresh)

    within = [a for a in agg if a["within_2pct"]]
    last_within_R = within[-1]["R"] if within else None

    cross_R = None
    for i in range(len(agg) - 1):
        a, b = agg[i], agg[i + 1]
        if a["within_2pct"] and not b["within_2pct"]:
            fa, fb = a["val_bpc_mean"], b["val_bpc_mean"]
            if fb > fa:
                w = (thresh - fa) / (fb - fa)
                cross_R = float(2 ** (math.log2(a["R"]) + w * (math.log2(b["R"]) - math.log2(a["R"]))))
            break

    doublings = []
    for i in range(len(agg) - 1):
        d = agg[i + 1]["val_bpc_mean"] - agg[i]["val_bpc_mean"]
        doublings.append({
            "from_R": round(agg[i]["R"], 3),
            "to_R": round(agg[i + 1]["R"], 3),
            "delta_val_bpc": round(d, 5),
            "delta_rel_to_fresh": round(d / ref, 5),
        })
    steepest = max(doublings, key=lambda x: x["delta_val_bpc"]) if doublings else None

    mss = float(np.mean([a["val_bpc_spread"] for a in agg]))
    return agg, {
        "val_bpc_fresh_reference": ref,
        "threshold_2pct_bpc": round(thresh, 5),
        "last_R_within_2pct": last_within_R,
        "first_R_beyond_2pct": next((a["R"] for a in agg if not a["within_2pct"]), None),
        "knee_R_2pct_interpolated": round(cross_R, 3) if cross_R else None,
        "paper_knee_R": 4.0,
        "all_R_within_2pct": all(a["within_2pct"] for a in agg),
        "degradation_per_doubling": doublings,
        "steepest_doubling": steepest,
        "mean_seed_spread_bpc": round(mss, 5),
        "total_val_bpc_range": round(agg[-1]["val_bpc_mean"] - ref, 5),
        "range_over_seed_spread": round((agg[-1]["val_bpc_mean"] - ref) / mss, 2) if mss > 0 else None,
        "gap_bpc_at_min_R": agg[0]["gap_bpc_mean"],
        "gap_bpc_at_max_R": agg[-1]["gap_bpc_mean"],
    }


def control_analysis(agg, ctrl_runs):
    """Compare the spread control against the prefix arm at the same unique fraction."""
    by_u = {}
    for r in ctrl_runs:
        by_u.setdefault(r["unique_frac"], []).append(r)
    prefix = {a["unique_frac"]: a for a in agg}
    out = []
    for u in sorted(by_u, reverse=True):
        rs = by_u[u]
        v = [r["val_bpc"] for r in rs]
        pa = prefix[u]
        out.append({
            "unique_frac": u,
            "R": round(float(np.mean([r["R_repetition"] for r in rs])), 3),
            "val_bpc_spread_mode": round(float(np.mean(v)), 5),
            "val_bpc_spread_mode_per_seed": [round(x, 5) for x in v],
            "val_bpc_prefix_mode": pa["val_bpc_mean"],
            "delta_spread_minus_prefix": round(float(np.mean(v)) - pa["val_bpc_mean"], 5),
            "gap_bpc_spread_mode": round(float(np.mean([r["gap_bpc"] for r in rs])), 5),
            "gap_bpc_prefix_mode": pa["gap_bpc_mean"],
        })
    return out


# ----------------------------------------------------------------------------- chart
def make_chart(agg, head, control, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R = [a["R"] for a in agg]
    v = [a["val_bpc_mean"] for a in agg]
    tr = [a["train_bpc_mean"] for a in agg]
    gap = [a["gap_bpc_mean"] for a in agg]
    ref = head["val_bpc_fresh_reference"]
    ticks = [f"{r:.1f}" for r in R]

    fig, ax = plt.subplots(2, 2, figsize=(12.5, 8.8))

    a0 = ax[0][0]
    a0.axhspan(ref, head["threshold_2pct_bpc"], color="#bfe3c6", alpha=0.65,
               label="within 2% of all-fresh")
    a0.axhline(ref, color="#2e7d32", ls="--", lw=1)
    for a in agg:
        for s in a["val_bpc_per_seed"]:
            a0.plot(a["R"], s, "o", color="#9aa0a6", ms=4, zorder=2)
    a0.plot(R, v, "-o", color="#1a237e", lw=2, ms=7, label="val bpc (mean, 2 seeds)", zorder=3)
    if control:
        a0.plot([c["R"] for c in control], [c["val_bpc_spread_mode"] for c in control],
                "D", color="#00838f", ms=8, mfc="none", mew=2, zorder=4,
                label="control: same unique budget, spread over corpus")
    a0.axvline(4.0, color="#c62828", ls=":", lw=2, label="paper knee R=4 (up to 9B params)")
    if head["knee_R_2pct_interpolated"]:
        a0.axvline(head["knee_R_2pct_interpolated"], color="#ef6c00", ls="-.", lw=2,
                   label=f"our 2% crossing R={head['knee_R_2pct_interpolated']:.2f}")
    a0.set_xscale("log", base=2)
    a0.set_xticks(R); a0.set_xticklabels(ticks)
    a0.set_xlabel("repetition count R = budget tokens / unique tokens (epochs)")
    a0.set_ylabel("val bits/char")
    a0.set_title("Val bpc vs repetition at FIXED compute")
    a0.legend(fontsize=8); a0.grid(alpha=0.3)

    a1 = ax[0][1]
    a1.plot(R, v, "-o", color="#1a237e", lw=2, label="val bpc (held-out)")
    a1.plot(R, tr, "-s", color="#ad1457", lw=2, label="train bpc (the unique subset)")
    a1.set_xscale("log", base=2)
    a1.set_xticks(R); a1.set_xticklabels(ticks)
    a1.set_xlabel("repetition count R"); a1.set_ylabel("bits/char")
    a1.set_title("Train vs val: the memorization signature")
    a1.legend(fontsize=8); a1.grid(alpha=0.3)

    a2 = ax[1][0]
    a2.plot(R, gap, "-o", color="#00695c", lw=2)
    a2.axhline(0, color="k", lw=0.8)
    a2.set_xscale("log", base=2)
    a2.set_xticks(R); a2.set_xticklabels(ticks)
    a2.set_xlabel("repetition count R"); a2.set_ylabel("val bpc - train bpc")
    a2.set_title("Train-val gap vs R")
    a2.grid(alpha=0.3)

    a3 = ax[1][1]
    d = head["degradation_per_doubling"]
    vals = [x["delta_val_bpc"] for x in d]
    labels = [f"{x['from_R']:.1f}→{x['to_R']:.1f}" for x in d]
    mx = max(vals)
    cols = ["#c62828" if x == mx else "#5c6bc0" for x in vals]
    a3.bar(range(len(vals)), vals, color=cols)
    a3.set_xticks(range(len(vals))); a3.set_xticklabels(labels, fontsize=8)
    a3.axhline(head["mean_seed_spread_bpc"], color="#757575", ls="--", lw=1,
               label=f"mean seed spread {head['mean_seed_spread_bpc']:.3f}")
    a3.axhline(0, color="k", lw=0.8)
    a3.set_ylabel("delta val bpc"); a3.set_xlabel("each step doubles R")
    a3.set_title("Marginal cost of each doubling of repetition")
    a3.legend(fontsize=8); a3.grid(alpha=0.3, axis="y")

    fig.suptitle("Data repetition at fixed compute: 0.17M-param char LM, tiny-shakespeare, "
                 "1.075M tokens every arm", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    cfg = load_config()
    p = cfg["params"]
    seed = int(cfg.get("seed", 0))
    set_seeds(seed)
    t_start = time.time()
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    text, md5 = get_text(cfg["dataset"]["source"])
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.uint16)
    n_val = int(len(data) * p["val_frac"])
    train_pool = data[: len(data) - n_val]
    val_pool = data[len(data) - n_val :]

    block = p["block_size"]
    val_eval_w = make_windows(val_pool[: p["eval_val_chars"]], block)
    val_curve_w = make_windows(val_pool[: p["eval_val_chars"]], block, p["eval_val_windows_curve"])

    budget = p["steps"] * p["batch_size"] * block
    log(f"corpus {len(data)} chars, V={len(chars)}, md5={md5}")
    log(f"train pool {len(train_pool)} chars | val {len(val_pool)} chars "
        f"(headline eval = first {p['eval_val_chars']} chars = {val_eval_w.size(0)} windows)")
    log(f"FIXED compute budget: {p['steps']} steps x {p['batch_size']} x {block} = {budget} tokens")

    runs = []
    for u in p["unique_fractions"]:
        for s in p["seeds"]:
            runs.append(train_one(p, len(chars), train_pool, val_eval_w, val_curve_w, u, s, log))

    # CONTROL: same unique-data budget, spread over the whole train split instead of a prefix.
    # Separates "less unique data" from "a narrower slice of the corpus, further from val".
    log("\ncontrol: spread (same unique char count, evenly sampled across the whole train split)")
    ctrl = []
    for u in p["control_fractions"]:
        for s in p["seeds"]:
            ctrl.append(train_one(p, len(chars), train_pool, val_eval_w, val_curve_w, u, s, log,
                                  mode="spread"))

    agg, head = analyse(runs)
    control = control_analysis(agg, ctrl)
    head["control_spread_vs_prefix"] = control
    make_chart(agg, head, control, HERE / "chart.png")

    log("")
    log(f"all-fresh (R={agg[0]['R']:.2f}) val bpc = {head['val_bpc_fresh_reference']:.4f}; "
        f"2% threshold = {head['threshold_2pct_bpc']:.4f}")
    log(f"last R within 2%    : {head['last_R_within_2pct']}")
    log(f"first R beyond 2%   : {head['first_R_beyond_2pct']}")
    log(f"2% crossing (interp): {head['knee_R_2pct_interpolated']}   (paper: 4.0)")
    log(f"steepest doubling   : {head['steepest_doubling']}")
    log(f"seed spread {head['mean_seed_spread_bpc']:.4f} bpc; total range "
        f"{head['total_val_bpc_range']:.4f} ({head['range_over_seed_spread']}x seed spread)")
    log(f"train-val gap: {head['gap_bpc_at_min_R']:.4f} at R={agg[0]['R']:.1f} -> "
        f"{head['gap_bpc_at_max_R']:.4f} at R={agg[-1]['R']:.1f}")
    for c in control:
        log(f"control u={c['unique_frac']:<8g} R={c['R']:>6.2f}: spread {c['val_bpc_spread_mode']:.4f} "
            f"vs prefix {c['val_bpc_prefix_mode']:.4f}  (delta {c['delta_spread_minus_prefix']:+.4f})")

    metrics = {
        "headline": ("val bpc vs repetition count R at fixed 1.075M-token compute; the R at which "
                     "val bpc leaves a 2% band around the all-fresh (R~1) value"),
        **head,
        "per_unique_fraction": agg,
        "n_params": runs[0]["n_params"],
        "budget_tokens": budget,
        "n_runs": len(runs),
        "seeds": p["seeds"],
        "unique_fractions": p["unique_fractions"],
        "control_fractions": p["control_fractions"],
        "runs": runs,
        "control_runs": ctrl,
    }
    results = {
        "id": cfg["id"],
        "git_commit": git_sha(),
        "seed": seed,
        "duration_sec": round(time.time() - t_start, 2),
        "dataset_md5": md5,
        "metrics": metrics,
        "env": env_info(),
        "status": "done",
        "log": lines,
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nwrote results.json + chart.png in {results['duration_sec']:.0f}s")


if __name__ == "__main__":
    main()
