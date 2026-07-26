"""Head dimension vs head count at fixed d_model: an exactly iso-parameter ablation.

At fixed d_model the attention block's parameter count does not depend on how the
residual width is split into heads: q/k/v are still d -> 3d and the output projection is
still d -> d for every (n_head, head_dim) with n_head * head_dim = d_model. So a sweep
over

    (n_head, head_dim) in {(1,128), (2,64), (4,32), (8,16), (16,8), (32,4)}

at d_model = 128 is iso-parameter BY CONSTRUCTION -- no matching heuristics needed. It is
also iso-FLOP up to the attention-score term, which is identical too because the T x T
score matrix is computed n_head times over head_dim channels: n_head * head_dim = d.

Stronger still: because every arm has the *same module shapes*, seeding the constructor
identically gives all six arms BIT-IDENTICAL initial weights. The only difference between
two arms is how the same q/k/v vectors are reshaped into heads before the softmax (and
the 1/sqrt(head_dim) score scale that follows from it). Combined with a shared batch
stream per seed, this is a tightly paired design.

Question: U-curve (interior optimum), monotone trend, or flat within seed noise?

Mechanistic side-probe: per-head attention entropy at the end of training, normalised by
the entropy of the uniform distribution over the causal prefix. Do tiny heads collapse to
sharp single-token attention?

Deterministic, CPU-only, single-threaded. Usage:  python run.py
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
LN2 = math.log(2.0)


# ----------------------------------------------------------------------------- utils
def set_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


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
            info[mod] = getattr(__import__(mod), "__version__", "?")
        except Exception:
            pass
    return info


# ----------------------------------------------------------------------------- data
def load_data(cfg):
    p = cfg["params"]
    txt_path = HERE / "data" / "tinyshakespeare.txt"
    if not txt_path.exists():   # data/ is gitignored; fetch on a fresh clone
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(cfg["dataset"]["source"], txt_path)
    text = txt_path.read_text()
    chars = sorted(set(text))
    vocab = len(chars)
    assert vocab == p["char_vocab_size"], f"expected {p['char_vocab_size']} chars, got {vocab}"
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in text], dtype=np.int64)
    n = int(p["train_frac"] * len(ids))
    return ids[:n], ids[n:], vocab


# ---------------------------------------------------------------------------- model
class Block(nn.Module):
    """Pre-norm transformer block. n_head only changes the reshape before the softmax."""

    def __init__(self, d, n_head, d_ff):
        super().__init__()
        assert d % n_head == 0
        self.n_head = n_head
        self.head_dim = d // n_head
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.fc = nn.Linear(d, d_ff, bias=False)
        self.out = nn.Linear(d_ff, d, bias=False)

    def _qkv(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        shape = (B, T, self.n_head, self.head_dim)
        return (q.view(*shape).transpose(1, 2),
                k.view(*shape).transpose(1, 2),
                v.view(*shape).transpose(1, 2))

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self._qkv(x)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, D))
        x = x + self.out(F.gelu(self.fc(self.ln2(x))))
        return x

    def attn_probs(self, x):
        """Explicit causal attention probabilities, (B, n_head, T, T). Probe only."""
        q, k, _ = self._qkv(x)
        T = x.shape[1]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.ones(T, T, dtype=torch.bool).tril()
        att = att.masked_fill(~mask, float("-inf"))
        return torch.softmax(att, dim=-1)


class GPT(nn.Module):
    def __init__(self, vocab, d, n_layer, n_head, d_ff, block_size):
        super().__init__()
        self.vocab, self.block_size = vocab, block_size
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(block_size, d)
        self.blocks = nn.ModuleList([Block(d, n_head, d_ff) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.apply(self._init)
        for name, prm in self.named_parameters():   # nanoGPT residual-projection scaling
            if name.endswith("proj.weight") or name.endswith("out.weight"):
                nn.init.normal_(prm, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def embed(self, idx):
        T = idx.shape[1]
        return self.tok(idx) + self.pos(torch.arange(T))

    def forward(self, idx):
        x = self.embed(idx)
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x))


def n_params(m):
    return sum(q.numel() for q in m.parameters())


def make_model(vocab, n_head, p):
    d = p["d_model"]
    return GPT(vocab, d, p["n_layer"], n_head, p["d_ff_mult"] * d, p["block_size"])


# ------------------------------------------------------------------------ train/eval
def get_batch(data, rng, batch_size, block_size):
    ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i:i + block_size] for i in ix])
    y = np.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return torch.from_numpy(x), torch.from_numpy(y)


@torch.no_grad()
def eval_bpc(model, val_ids, p):
    """Char-level bits per character on held-out text (chars == tokens here)."""
    model.eval()
    bs = p["block_size"]
    n_blocks = min((len(val_ids) - 1) // bs, p["max_eval_blocks"])
    xs = np.stack([val_ids[i * bs:(i + 1) * bs] for i in range(n_blocks)])
    ys = np.stack([val_ids[i * bs + 1:(i + 1) * bs + 1] for i in range(n_blocks)])
    tot_nats, tot_tokens = 0.0, 0
    for s in range(0, n_blocks, p["eval_batch"]):
        xb = torch.from_numpy(xs[s:s + p["eval_batch"]])
        yb = torch.from_numpy(ys[s:s + p["eval_batch"]])
        loss = F.cross_entropy(model(xb).reshape(-1, model.vocab), yb.reshape(-1), reduction="sum")
        tot_nats += float(loss)
        tot_tokens += int(yb.numel())
    model.train()
    return {"bpc": tot_nats / (LN2 * tot_tokens), "eval_chars": tot_tokens}


@torch.no_grad()
def attention_entropy(model, val_ids, p):
    """Per-head mean attention entropy at end of training, on fixed val batches.

    For each query position t (0-indexed) the causal prefix has t+1 keys, so the maximum
    possible entropy is ln(t+1). We report the NORMALISED entropy H_t / ln(t+1) averaged
    over t >= entropy_min_pos, plus the mean top-1 attention weight (peakiness) and the
    mean perplexity of the attention distribution (== effective number of keys attended).
    """
    model.eval()
    bs, B = p["block_size"], p["eval_batch"]
    n_need = p["entropy_batches"] * B
    n_blocks = min((len(val_ids) - 1) // bs, n_need)
    xs = np.stack([val_ids[i * bs:(i + 1) * bs] for i in range(n_blocks)])
    t_idx = torch.arange(bs)
    norm = torch.log(t_idx.float() + 1.0).clamp(min=1e-9)
    keep = t_idx >= p["entropy_min_pos"]

    per_layer = []
    for li, blk in enumerate(model.blocks):
        H_sum = torch.zeros(blk.n_head)
        top_sum = torch.zeros(blk.n_head)
        eff_sum = torch.zeros(blk.n_head)
        nb = 0
        for s in range(0, n_blocks, B):
            xb = torch.from_numpy(xs[s:s + B])
            h = model.embed(xb)
            for j in range(li):                      # run the earlier blocks forward
                h = model.blocks[j](h)
            probs = blk.attn_probs(h)                # (B, nh, T, T)
            ent = -(probs * torch.log(probs.clamp(min=1e-12))).sum(-1)   # (B, nh, T)
            ent_n = ent / norm                       # normalised by ln(t+1)
            top = probs.max(-1).values               # (B, nh, T)
            eff = torch.exp(ent)                     # effective #keys attended
            H_sum += ent_n[:, :, keep].mean(dim=(0, 2))
            top_sum += top[:, :, keep].mean(dim=(0, 2))
            eff_sum += eff[:, :, keep].mean(dim=(0, 2))
            nb += 1
        per_layer.append({
            "layer": li,
            "entropy_norm_per_head": [round(float(u), 4) for u in (H_sum / nb)],
            "top1_weight_per_head": [round(float(u), 4) for u in (top_sum / nb)],
            "eff_keys_per_head": [round(float(u), 3) for u in (eff_sum / nb)],
        })
    model.train()
    allH = [u for L in per_layer for u in L["entropy_norm_per_head"]]
    allT = [u for L in per_layer for u in L["top1_weight_per_head"]]
    allE = [u for L in per_layer for u in L["eff_keys_per_head"]]
    return {
        "per_layer": per_layer,
        "entropy_norm_mean": round(float(np.mean(allH)), 4),
        "entropy_norm_min": round(float(np.min(allH)), 4),
        "entropy_norm_max": round(float(np.max(allH)), 4),
        "top1_weight_mean": round(float(np.mean(allT)), 4),
        "top1_weight_max": round(float(np.max(allT)), 4),
        "eff_keys_mean": round(float(np.mean(allE)), 3),
        "eff_keys_min": round(float(np.min(allE)), 3),
    }


def train_one(vocab, n_head, seed, p, train_ids, val_ids):
    head_dim = p["d_model"] // n_head
    set_seeds(seed)                       # identical init across ALL n_head at a given seed
    model = make_model(vocab, n_head, p)
    init_sig = float(sum(float(q.detach().double().abs().sum()) for q in model.parameters()))
    decay = [q for q in model.parameters() if q.dim() >= 2]
    nodecay = [q for q in model.parameters() if q.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": p["weight_decay"]},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=p["lr"], betas=(0.9, 0.95),
    )
    rng = np.random.default_rng(seed)     # identical batch stream across ALL n_head
    steps, warm = p["steps"], p["warmup"]
    losses = []
    t0 = time.time()
    for it in range(steps):
        if it < warm:
            lr = p["lr"] * (it + 1) / warm
        else:
            prog = (it - warm) / max(1, steps - warm)
            lr = p["lr"] * (p["lr_min_frac"] + (1 - p["lr_min_frac"]) * 0.5 * (1 + math.cos(math.pi * prog)))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch(train_ids, rng, p["batch_size"], p["block_size"])
        loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), p["grad_clip"])
        opt.step()
        losses.append(float(loss.detach()))
    train_s = time.time() - t0
    ev = eval_bpc(model, val_ids, p)
    ent = attention_entropy(model, val_ids, p)
    curve = [round(float(np.mean(losses[i:i + 25])), 4) for i in range(0, steps, 25)]
    rec = {
        "n_head": int(n_head), "head_dim": int(head_dim), "seed": int(seed),
        "n_params": n_params(model),
        "attn_params_per_layer": 4 * p["d_model"] * p["d_model"],
        "init_signature": round(init_sig, 6),
        "train_seconds": round(train_s, 1),
        "final_train_loss_ma50": round(float(np.mean(losses[-50:])), 4),
        "train_loss_curve_ma25": curve,
        "val_bpc": round(ev["bpc"], 5),
        "eval_chars": ev["eval_chars"],
        "attention": ent,
    }
    print(f"  [n_head={n_head:2d} head_dim={head_dim:3d} seed={seed}] P={rec['n_params']} "
          f"bpc={rec['val_bpc']:.4f} entN={ent['entropy_norm_mean']:.3f} "
          f"top1={ent['top1_weight_mean']:.3f} ({rec['train_seconds']}s)", flush=True)
    return rec


# -------------------------------------------------------------------------- analysis
def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    return 0.0 if den == 0 else float((ra * rb).sum() / den)


def make_chart(summary, runs, p, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hd = [s["head_dim"] for s in summary]
    mean = np.array([s["val_bpc_mean"] for s in summary])
    seeds = sorted({r["seed"] for r in runs})

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    ax = axes[0]
    for sd in seeds:
        ys = [r["val_bpc"] for h in hd for r in runs if r["head_dim"] == h and r["seed"] == sd]
        ax.plot(hd, ys, "o--", alpha=0.45, lw=1, ms=4, label=f"seed {sd}")
    lo = np.array([s["val_bpc_min"] for s in summary])
    hi = np.array([s["val_bpc_max"] for s in summary])
    ax.errorbar(hd, mean, yerr=[mean - lo, hi - mean], fmt="o-", color="k", lw=2, ms=7,
                capsize=4, label="mean (bar = seed range)")
    best = summary[int(np.argmin(mean))]
    ax.scatter([best["head_dim"]], [best["val_bpc_mean"]], s=200, facecolors="none",
               edgecolors="crimson", lw=2, zorder=5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(hd); ax.set_xticklabels([str(h) for h in hd])
    ax.set_xlabel("head_dim  (n_head = 128 / head_dim)")
    ax.set_ylabel("val bits per character")
    ax.set_title(f"Iso-param head split, d_model=128\nall arms {runs[0]['n_params']:,} params")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1]
    for sd in seeds:
        ys = [r["attention"]["entropy_norm_mean"] for h in hd for r in runs
              if r["head_dim"] == h and r["seed"] == sd]
        ax.plot(hd, ys, "s--", alpha=0.45, lw=1, ms=4)
    em = [s["entropy_norm_mean"] for s in summary]
    elo = [s["entropy_norm_min_head"] for s in summary]
    ehi = [s["entropy_norm_max_head"] for s in summary]
    ax.plot(hd, em, "s-", color="darkgreen", lw=2, ms=7, label="mean over heads/layers/seeds")
    ax.fill_between(hd, elo, ehi, color="darkgreen", alpha=0.15, label="min-max over heads")
    ax2 = ax.twinx()
    ax2.plot(hd, [s["top1_weight_mean"] for s in summary], "^-", color="darkorange", lw=1.5,
             ms=6, label="mean top-1 attn weight")
    ax2.set_ylabel("mean top-1 attention weight", color="darkorange")
    ax2.tick_params(axis="y", colors="darkorange")
    ax.set_xscale("log", base=2)
    ax.set_xticks(hd); ax.set_xticklabels([str(h) for h in hd])
    ax.set_xlabel("head_dim")
    ax.set_ylabel("normalised attention entropy  H / ln(t+1)", color="darkgreen")
    ax.set_title("Mechanistic probe: do tiny heads sharpen?")
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower right")

    ax = axes[2]
    xs = np.arange(len(runs[0]["train_loss_curve_ma25"])) * 25
    cmap = plt.get_cmap("viridis")
    for i, s in enumerate(summary):
        cs = np.array([r["train_loss_curve_ma25"] for r in runs if r["head_dim"] == s["head_dim"]])
        ax.plot(xs, cs.mean(0) / LN2, lw=1.6, color=cmap(i / max(1, len(summary) - 1)),
                label=f"hd={s['head_dim']} (nh={s['n_head']})")
    ax.set_xlabel("step"); ax.set_ylabel("train loss (bits/char, MA25)")
    ax.set_title(f"Training curves ({p['steps']} steps, batch {p['batch_size']} x {p['block_size']})")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    fig.suptitle("head_dim vs head count at fixed d_model - exactly iso-parameter by construction",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    cfg = load_config()
    p = cfg["params"]
    t_start = time.time()

    train_ids, val_ids, vocab = load_data(cfg)
    print(f"data: {len(train_ids)} train chars, {len(val_ids)} val chars, vocab {vocab}", flush=True)

    for nh, hd in p["configs"]:
        assert nh * hd == p["d_model"], f"({nh},{hd}) is not iso-param at d_model={p['d_model']}"

    runs = []
    for nh, hd in p["configs"]:
        for seed in p["seeds"]:
            runs.append(train_one(vocab, nh, seed, p, train_ids, val_ids))

    # ---- all arms must be exactly iso-parameter, and share their init at a given seed
    param_counts = sorted({r["n_params"] for r in runs})
    init_sigs = {sd: sorted({r["init_signature"] for r in runs if r["seed"] == sd})
                 for sd in p["seeds"]}
    iso_param_ok = len(param_counts) == 1
    same_init_ok = all(len(v) == 1 for v in init_sigs.values())

    summary = []
    for nh, hd in p["configs"]:
        rs = [r for r in runs if r["n_head"] == nh]
        b = np.array([r["val_bpc"] for r in rs])
        summary.append({
            "n_head": nh, "head_dim": hd,
            "val_bpc_per_seed": [round(float(u), 5) for u in b],
            "val_bpc_mean": round(float(b.mean()), 5),
            "val_bpc_std": round(float(b.std(ddof=1)) if len(b) > 1 else 0.0, 5),
            "val_bpc_min": round(float(b.min()), 5),
            "val_bpc_max": round(float(b.max()), 5),
            "train_loss_mean": round(float(np.mean([r["final_train_loss_ma50"] for r in rs])), 5),
            "entropy_norm_mean": round(float(np.mean([r["attention"]["entropy_norm_mean"] for r in rs])), 4),
            "entropy_norm_min_head": round(float(np.min([r["attention"]["entropy_norm_min"] for r in rs])), 4),
            "entropy_norm_max_head": round(float(np.max([r["attention"]["entropy_norm_max"] for r in rs])), 4),
            "top1_weight_mean": round(float(np.mean([r["attention"]["top1_weight_mean"] for r in rs])), 4),
            "top1_weight_max_head": round(float(np.max([r["attention"]["top1_weight_max"] for r in rs])), 4),
            "eff_keys_mean": round(float(np.mean([r["attention"]["eff_keys_mean"] for r in rs])), 3),
            "train_seconds_mean": round(float(np.mean([r["train_seconds"] for r in rs])), 1),
        })

    means = np.array([s["val_bpc_mean"] for s in summary])
    hds = np.array([s["head_dim"] for s in summary])
    argmin = int(np.argmin(means))
    interior = 0 < argmin < len(means) - 1

    # seed noise: mean within-config spread; config spread: range of config means
    within = float(np.mean([s["val_bpc_max"] - s["val_bpc_min"] for s in summary]))
    across = float(means.max() - means.min())

    # paired ranking: does the same config win in every seed?
    per_seed_best = {}
    for sd in p["seeds"]:
        rs = sorted([r for r in runs if r["seed"] == sd], key=lambda r: r["val_bpc"])
        per_seed_best[str(sd)] = {"best_head_dim": rs[0]["head_dim"],
                                  "worst_head_dim": rs[-1]["head_dim"],
                                  "order_head_dim_best_first": [r["head_dim"] for r in rs]}
    seeds_agree_on_best = len({v["best_head_dim"] for v in per_seed_best.values()}) == 1

    rho_bpc_vs_logdim = spearman(np.log2(hds), means)
    rho_ent_vs_logdim = spearman(np.log2(hds), np.array([s["entropy_norm_mean"] for s in summary]))

    # U-verdict: interior optimum AND both ends worse by more than the mean seed spread
    edge_gap_lo = float(means[0] - means[argmin])    # head_dim = 128 end (1 giant head)
    edge_gap_hi = float(means[-1] - means[argmin])   # head_dim = 4 end (32 tiny heads)
    if interior and min(edge_gap_lo, edge_gap_hi) > within:
        shape = "U-curve (interior optimum, both edges outside seed spread)"
    elif across <= within:
        shape = "flat within seed noise"
    elif abs(rho_bpc_vs_logdim) >= 0.83:
        shape = "monotone in head_dim"
    else:
        shape = "structured but not a clean U (interior optimum inside seed noise)"

    metrics = {
        "headline": "val bpc vs head_dim at fixed d_model=128 (exactly iso-parameter)",
        "shape_verdict": shape,
        "n_params_all_arms": param_counts[0] if iso_param_ok else param_counts,
        "iso_param_exact": iso_param_ok,
        "identical_init_across_arms_per_seed": same_init_ok,
        "shared_batch_stream_per_seed": True,
        "best_head_dim": int(hds[argmin]),
        "best_n_head": int(p["d_model"] // hds[argmin]),
        "best_val_bpc_mean": round(float(means[argmin]), 5),
        "worst_head_dim": int(hds[int(np.argmax(means))]),
        "worst_val_bpc_mean": round(float(means.max()), 5),
        "interior_optimum": bool(interior),
        "config_spread_bpc": round(across, 5),
        "mean_seed_spread_bpc": round(within, 5),
        "spread_ratio_config_over_seed": round(across / within, 2) if within > 0 else None,
        "edge_gap_head_dim_128_end": round(edge_gap_lo, 5),
        "edge_gap_head_dim_4_end": round(edge_gap_hi, 5),
        "spearman_bpc_vs_log2_head_dim": round(rho_bpc_vs_logdim, 4),
        "spearman_attn_entropy_vs_log2_head_dim": round(rho_ent_vs_logdim, 4),
        "seeds_agree_on_best": seeds_agree_on_best,
        "per_seed_ranking": per_seed_best,
        "by_config": summary,
        "train_steps": p["steps"],
        "tokens_per_run": p["steps"] * p["batch_size"] * p["block_size"],
        "seeds": p["seeds"],
        "n_runs": len(runs),
        "runs": runs,
    }

    make_chart(summary, runs, p, HERE / "chart.png")

    results = {
        "id": cfg.get("id", "unknown"),
        "git_commit": git_sha(),
        "seed": int(cfg.get("seed", 0)),
        "duration_sec": round(time.time() - t_start, 2),
        "metrics": metrics,
        "env": env_info(),
        "status": "done",
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== summary ===")
    print(f"iso-param exact: {iso_param_ok} ({param_counts}), identical init per seed: {same_init_ok}")
    for s in summary:
        print(f"  hd={s['head_dim']:3d} nh={s['n_head']:2d}  bpc={s['val_bpc_mean']:.4f} "
              f"+-{s['val_bpc_std']:.4f}  entN={s['entropy_norm_mean']:.3f}  "
              f"top1={s['top1_weight_mean']:.3f}  eff_keys={s['eff_keys_mean']:.2f}")
    print(f"verdict: {shape}")
    print(f"config spread {across:.4f} bpc vs mean seed spread {within:.4f} bpc")
    print(f"total {results['duration_sec']:.0f}s")


if __name__ == "__main__":
    main()
