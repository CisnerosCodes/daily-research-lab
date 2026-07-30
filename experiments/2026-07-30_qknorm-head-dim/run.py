"""Does the head-dim tax survive QK-norm?

2026-07-26_head-dim-vs-count-isoparam: at fixed d_model=128 every (n_head, head_dim)
split with n_head * head_dim = 128 is exactly iso-parameter, and val bpc came out
perfectly MONOTONE in head_dim (Spearman -1.00): one giant head tied best, 32 tiny heads
were a 0.217-bpc tax. But that run flagged its own confound: the pre-softmax logit scale
1/sqrt(head_dim) only equalises the attention temperature at init. Trained q/k norms can
drift, and they drift differently per split, so the arms were effectively trained at
different softmax temperatures (estimated ~5.7x spread).

This experiment reruns the identical sweep with a second arm that pins the temperature:
per-head RMS-norm on q and k with learnable per-channel gains (QK-norm, ViT-22B style).
The gain vectors have length d_model regardless of the split, so the qknorm arm is ALSO
exactly iso-parameter across configs. If the tiny-head tax was mis-set temperature,
QK-norm should flatten the curve; if attention with rank-head_dim score maps is
intrinsically weaker, the monotone ordering survives.

Mechanistic probes, measured on the trained models over fixed val batches:
  - pre-softmax logit std per head (the confound variable itself: is it actually spread
    out in the baseline arm and pinned in the qknorm arm?)
  - normalised attention entropy (as in the 2026-07-26 run)

Design controls carried over: identical init across all configs at a given seed (the
qknorm gains are init to ones and consume no RNG, so the shared weights are bit-identical
across ALL arm x config cells at a given seed), shared batch stream per seed, exact
iso-param within each arm.

Deterministic, CPU-only. Usage:  python run.py   (SMOKE=1 for a 40-step smoke test)
"""
import json, math, os, random, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(2)

HERE = Path(__file__).resolve().parent
LN2 = math.log(2.0)
SMOKE = os.environ.get("SMOKE") == "1"


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
    info = {"python": sys.version.split()[0], "torch_threads": torch.get_num_threads()}
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
    """Pre-norm transformer block; n_head only changes the reshape before the softmax.

    qknorm=True adds per-head RMS-norm on q and k with learnable per-channel gains of
    length d (same shape for every split -> iso-param across configs), then the usual
    1/sqrt(head_dim) scale. With unit-RMS channels the logit std is ~1 at init and stays
    pinned during training for EVERY head_dim.
    """

    def __init__(self, d, n_head, d_ff, qknorm: bool):
        super().__init__()
        assert d % n_head == 0
        self.n_head = n_head
        self.head_dim = d // n_head
        self.qknorm = qknorm
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.fc = nn.Linear(d, d_ff, bias=False)
        self.out = nn.Linear(d_ff, d, bias=False)
        if qknorm:  # init to ones: consumes no RNG, keeps shared weights bit-identical
            self.q_gain = nn.Parameter(torch.ones(d))
            self.k_gain = nn.Parameter(torch.ones(d))

    def _rms(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

    def _qkv(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        if self.qknorm:
            shape4 = (B, T, self.n_head, self.head_dim)
            q = self._rms(q.view(*shape4)).reshape(B, T, D) * self.q_gain
            k = self._rms(k.view(*shape4)).reshape(B, T, D) * self.k_gain
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

    def attn_logits(self, x):
        """Explicit pre-softmax causal logits, (B, n_head, T, T). Probe only."""
        q, k, _ = self._qkv(x)
        return (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)


class GPT(nn.Module):
    def __init__(self, vocab, d, n_layer, n_head, d_ff, block_size, qknorm):
        super().__init__()
        self.vocab, self.block_size = vocab, block_size
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(block_size, d)
        self.blocks = nn.ModuleList([Block(d, n_head, d_ff, qknorm) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.apply(self._init)
        for name, prm in self.named_parameters():   # nanoGPT residual-projection scaling
            if name.endswith("proj.weight") or name.endswith("out.weight"):
                nn.init.normal_(prm, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))
        if qknorm:  # _init must not leave the gains at anything but ones
            for blk in self.blocks:
                with torch.no_grad():
                    blk.q_gain.fill_(1.0)
                    blk.k_gain.fill_(1.0)

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


# ------------------------------------------------------------------------ train/eval
def get_batch(data, rng, batch_size, block_size):
    ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i:i + block_size] for i in ix])
    y = np.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return torch.from_numpy(x), torch.from_numpy(y)


@torch.no_grad()
def eval_bpc(model, val_ids, p):
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
def attention_probe(model, val_ids, p):
    """Trained-model attention statistics on fixed val batches.

    Per head: normalised entropy H_t/ln(t+1) (t >= entropy_min_pos), top-1 weight, and
    the std of the pre-softmax logits over the causal-valid entries -- the effective
    temperature the model actually trained itself into.
    """
    model.eval()
    bs, B = p["block_size"], p["eval_batch"]
    n_need = p["entropy_batches"] * B
    n_blocks = min((len(val_ids) - 1) // bs, n_need)
    xs = np.stack([val_ids[i * bs:(i + 1) * bs] for i in range(n_blocks)])
    t_idx = torch.arange(bs)
    norm = torch.log(t_idx.float() + 1.0).clamp(min=1e-9)
    keep = t_idx >= p["entropy_min_pos"]
    causal = torch.ones(bs, bs, dtype=torch.bool).tril()

    per_layer = []
    for li, blk in enumerate(model.blocks):
        H_sum = torch.zeros(blk.n_head)
        top_sum = torch.zeros(blk.n_head)
        lg_mu = torch.zeros(blk.n_head)
        lg_sq = torch.zeros(blk.n_head)
        lg_n = 0
        nb = 0
        for s in range(0, n_blocks, B):
            xb = torch.from_numpy(xs[s:s + B])
            h = model.embed(xb)
            for j in range(li):
                h = model.blocks[j](h)
            logits = blk.attn_logits(h)              # (B, nh, T, T)
            probs = logits.masked_fill(~causal, float("-inf")).softmax(-1)
            ent = -(probs * torch.log(probs.clamp(min=1e-12))).sum(-1)   # (B, nh, T)
            H_sum += (ent / norm)[:, :, keep].mean(dim=(0, 2))
            top_sum += probs.max(-1).values[:, :, keep].mean(dim=(0, 2))
            lv = logits.masked_select(causal.view(1, 1, bs, bs)).view(logits.shape[0], blk.n_head, -1)
            lg_mu += lv.mean(dim=(0, 2)) * lv.shape[0]
            lg_sq += lv.pow(2).mean(dim=(0, 2)) * lv.shape[0]
            lg_n += lv.shape[0]
            nb += 1
        mu = lg_mu / lg_n
        std = (lg_sq / lg_n - mu ** 2).clamp(min=0).sqrt()
        per_layer.append({
            "layer": li,
            "entropy_norm_per_head": [round(float(u), 4) for u in (H_sum / nb)],
            "top1_weight_per_head": [round(float(u), 4) for u in (top_sum / nb)],
            "logit_std_per_head": [round(float(u), 4) for u in std],
        })
    model.train()
    allH = [u for L in per_layer for u in L["entropy_norm_per_head"]]
    allT = [u for L in per_layer for u in L["top1_weight_per_head"]]
    allS = [u for L in per_layer for u in L["logit_std_per_head"]]
    return {
        "per_layer": per_layer,
        "entropy_norm_mean": round(float(np.mean(allH)), 4),
        "top1_weight_mean": round(float(np.mean(allT)), 4),
        "logit_std_mean": round(float(np.mean(allS)), 4),
        "logit_std_min_head": round(float(np.min(allS)), 4),
        "logit_std_max_head": round(float(np.max(allS)), 4),
    }


def train_one(vocab, arm, n_head, seed, p, train_ids, val_ids):
    head_dim = p["d_model"] // n_head
    qknorm = arm == "qknorm"
    set_seeds(seed)                    # identical init across ALL (arm, n_head) at a seed
    model = GPT(vocab, p["d_model"], p["n_layer"], n_head,
                p["d_ff_mult"] * p["d_model"], p["block_size"], qknorm)
    shared_sig = float(sum(float(q.detach().double().abs().sum())
                           for name, q in model.named_parameters() if "gain" not in name))
    decay = [q for q in model.parameters() if q.dim() >= 2]
    nodecay = [q for q in model.parameters() if q.dim() < 2]   # gains land here: no WD
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": p["weight_decay"]},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=p["lr"], betas=(0.9, 0.95),
    )
    rng = np.random.default_rng(seed)  # identical batch stream across ALL arms/configs
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
    probe = attention_probe(model, val_ids, p)
    curve = [round(float(np.mean(losses[i:i + 25])), 4) for i in range(0, steps, 25)]
    rec = {
        "arm": arm, "n_head": int(n_head), "head_dim": int(head_dim), "seed": int(seed),
        "n_params": n_params(model),
        "shared_init_signature": round(shared_sig, 6),
        "train_seconds": round(train_s, 1),
        "final_train_loss_ma50": round(float(np.mean(losses[-50:])), 4),
        "train_loss_curve_ma25": curve,
        "val_bpc": round(ev["bpc"], 5),
        "eval_chars": ev["eval_chars"],
        "attention": probe,
    }
    print(f"  [{arm:8s} hd={head_dim:3d} nh={n_head:2d} seed={seed}] P={rec['n_params']} "
          f"bpc={rec['val_bpc']:.4f} logit_std={probe['logit_std_mean']:.3f} "
          f"entN={probe['entropy_norm_mean']:.3f} ({rec['train_seconds']}s)", flush=True)
    return rec


# -------------------------------------------------------------------------- analysis
def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    return 0.0 if den == 0 else float((ra * rb).sum() / den)


def summarize_arm(arm, runs, configs):
    out = []
    for nh, hd in configs:
        rs = [r for r in runs if r["arm"] == arm and r["n_head"] == nh]
        b = np.array([r["val_bpc"] for r in rs])
        out.append({
            "arm": arm, "n_head": nh, "head_dim": hd,
            "val_bpc_per_seed": [round(float(u), 5) for u in b],
            "val_bpc_mean": round(float(b.mean()), 5),
            "val_bpc_min": round(float(b.min()), 5),
            "val_bpc_max": round(float(b.max()), 5),
            "logit_std_mean": round(float(np.mean([r["attention"]["logit_std_mean"] for r in rs])), 4),
            "logit_std_min_head": round(float(np.min([r["attention"]["logit_std_min_head"] for r in rs])), 4),
            "logit_std_max_head": round(float(np.max([r["attention"]["logit_std_max_head"] for r in rs])), 4),
            "entropy_norm_mean": round(float(np.mean([r["attention"]["entropy_norm_mean"] for r in rs])), 4),
            "top1_weight_mean": round(float(np.mean([r["attention"]["top1_weight_mean"] for r in rs])), 4),
            "train_seconds_mean": round(float(np.mean([r["train_seconds"] for r in rs])), 1),
        })
    return out


def arm_shape(summary):
    means = np.array([s["val_bpc_mean"] for s in summary])
    hds = np.array([s["head_dim"] for s in summary])
    argmin = int(np.argmin(means))
    interior = 0 < argmin < len(means) - 1
    within = float(np.mean([s["val_bpc_max"] - s["val_bpc_min"] for s in summary]))
    across = float(means.max() - means.min())
    rho = spearman(np.log2(hds), means)
    edge_lo = float(means[0] - means[argmin])
    edge_hi = float(means[-1] - means[argmin])
    if interior and min(edge_lo, edge_hi) > within:
        shape = "U-curve (interior optimum, both edges outside seed spread)"
    elif across <= within:
        shape = "flat within seed noise"
    elif abs(rho) >= 0.9:
        shape = "monotone in head_dim"
    else:
        shape = "structured but not a clean U"
    return {
        "shape_verdict": shape,
        "best_head_dim": int(hds[argmin]),
        "best_val_bpc_mean": round(float(means[argmin]), 5),
        "worst_head_dim": int(hds[int(np.argmax(means))]),
        "worst_val_bpc_mean": round(float(means.max()), 5),
        "config_spread_bpc": round(across, 5),
        "mean_seed_spread_bpc": round(within, 5),
        "spread_ratio_config_over_seed": round(across / within, 2) if within > 0 else None,
        "spearman_bpc_vs_log2_head_dim": round(rho, 4),
        "interior_optimum": bool(interior),
    }


def make_chart(sums, runs, p, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = p["arms"]
    colors = {"baseline": "#444444", "qknorm": "#c0392b"}
    hd = [s["head_dim"] for s in sums[arms[0]]]
    seeds = sorted({r["seed"] for r in runs})

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    ax = axes[0]
    for arm in arms:
        S = sums[arm]
        mean = np.array([s["val_bpc_mean"] for s in S])
        lo = np.array([s["val_bpc_min"] for s in S])
        hi = np.array([s["val_bpc_max"] for s in S])
        ax.errorbar(hd, mean, yerr=[mean - lo, hi - mean], fmt="o-", lw=2, ms=6, capsize=4,
                    color=colors[arm], label=f"{arm} (bar = seed range)")
        for sd in seeds:
            ys = [r["val_bpc"] for h in hd for r in runs
                  if r["arm"] == arm and r["head_dim"] == h and r["seed"] == sd]
            ax.plot(hd, ys, "--", alpha=0.3, lw=0.9, color=colors[arm])
    ax.set_xscale("log", base=2)
    ax.set_xticks(hd); ax.set_xticklabels([str(h) for h in hd])
    ax.set_xlabel("head_dim  (n_head = 128 / head_dim)")
    ax.set_ylabel("val bits per character")
    ax.set_title("Iso-param head split, with vs without QK-norm")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1]
    for arm in arms:
        S = sums[arm]
        m = [s["logit_std_mean"] for s in S]
        lo = [s["logit_std_min_head"] for s in S]
        hi = [s["logit_std_max_head"] for s in S]
        ax.plot(hd, m, "s-", lw=2, ms=6, color=colors[arm], label=f"{arm} mean")
        ax.fill_between(hd, lo, hi, color=colors[arm], alpha=0.15)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(hd); ax.set_xticklabels([str(h) for h in hd])
    ax.set_xlabel("head_dim")
    ax.set_ylabel("pre-softmax logit std (trained)")
    ax.set_title("The confound variable itself\n(shade = min..max over heads)")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)

    ax = axes[2]
    base = np.array([s["val_bpc_mean"] for s in sums["baseline"]])
    qk = np.array([s["val_bpc_mean"] for s in sums["qknorm"]])
    x = np.arange(len(hd))
    ax.bar(x, qk - base, 0.55, color=["#2c7fb8" if v < 0 else "#d95f0e" for v in (qk - base)])
    for sd in seeds:
        db = []
        for h in hd:
            b = [r["val_bpc"] for r in runs if r["arm"] == "baseline" and r["head_dim"] == h and r["seed"] == sd][0]
            q = [r["val_bpc"] for r in runs if r["arm"] == "qknorm" and r["head_dim"] == h and r["seed"] == sd][0]
            db.append(q - b)
        ax.plot(x, db, "ko", ms=4, alpha=0.5)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([str(h) for h in hd])
    ax.set_xlabel("head_dim")
    ax.set_ylabel("val bpc:  qknorm - baseline")
    ax.set_title("Paired effect of QK-norm per split\n(dots = per-seed paired deltas)")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Does the head-dim tax survive QK-norm? (d_model=128, exactly iso-param per arm)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    cfg = load_config()
    p = cfg["params"]
    if SMOKE:
        p["steps"], p["warmup"], p["configs"], p["seeds"] = 40, 5, [[1, 128], [32, 4]], [0]
    t_start = time.time()

    train_ids, val_ids, vocab = load_data(cfg)
    print(f"data: {len(train_ids)} train chars, {len(val_ids)} val chars, vocab {vocab}", flush=True)
    for nh, hd in p["configs"]:
        assert nh * hd == p["d_model"], f"({nh},{hd}) is not iso-param at d_model={p['d_model']}"

    runs = []
    for arm in p["arms"]:
        for nh, hd in p["configs"]:
            for seed in p["seeds"]:
                runs.append(train_one(vocab, arm, nh, seed, p, train_ids, val_ids))

    # invariants: iso-param within each arm; shared weights identical across everything per seed
    iso = {arm: sorted({r["n_params"] for r in runs if r["arm"] == arm}) for arm in p["arms"]}
    iso_ok = all(len(v) == 1 for v in iso.values())
    sig_ok = all(
        len({r["shared_init_signature"] for r in runs if r["seed"] == sd}) == 1
        for sd in p["seeds"])

    sums = {arm: summarize_arm(arm, runs, p["configs"]) for arm in p["arms"]}
    shapes = {arm: arm_shape(sums[arm]) for arm in p["arms"]}

    base_spread = shapes["baseline"]["config_spread_bpc"]
    qk_spread = shapes["qknorm"]["config_spread_bpc"]
    tax_removed = (base_spread - qk_spread) / base_spread if base_spread > 0 else None

    paired = []
    for i, (nh, hd) in enumerate(p["configs"]):
        b = sums["baseline"][i]["val_bpc_mean"]
        q = sums["qknorm"][i]["val_bpc_mean"]
        paired.append({"head_dim": hd, "qknorm_minus_baseline_bpc": round(q - b, 5)})

    def std_spread(arm):
        v = [s["logit_std_mean"] for s in sums[arm]]
        return round(max(v) / max(min(v), 1e-9), 2)

    metrics = {
        "headline": "val bpc vs head_dim at fixed d_model=128, baseline vs QK-norm",
        "iso_param_exact_within_arm": iso_ok,
        "n_params_by_arm": {a: v[0] if len(v) == 1 else v for a, v in iso.items()},
        "identical_shared_init_per_seed": sig_ok,
        "shared_batch_stream": True,
        "baseline": shapes["baseline"],
        "qknorm": shapes["qknorm"],
        "logit_std_spread_ratio_baseline": std_spread("baseline"),
        "logit_std_spread_ratio_qknorm": std_spread("qknorm"),
        "fraction_of_head_dim_tax_removed_by_qknorm": round(tax_removed, 4) if tax_removed is not None else None,
        "paired_effect_of_qknorm": paired,
        "by_config": {a: sums[a] for a in p["arms"]},
        "train_steps": p["steps"],
        "tokens_per_run": p["steps"] * p["batch_size"] * p["block_size"],
        "seeds": p["seeds"],
        "n_runs": len(runs),
        "runs": runs,
    }

    make_chart(sums, runs, p, HERE / "chart.png")

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
    print(f"iso-param within arm: {iso_ok} {iso}; shared init identical per seed: {sig_ok}")
    for arm in p["arms"]:
        sh = shapes[arm]
        print(f"[{arm}] {sh['shape_verdict']}  spread={sh['config_spread_bpc']:.4f} bpc "
              f"(seed spread {sh['mean_seed_spread_bpc']:.4f})  rho={sh['spearman_bpc_vs_log2_head_dim']:.2f}  "
              f"best hd={sh['best_head_dim']}")
    print(f"trained logit-std spread across configs: baseline {std_spread('baseline')}x, "
          f"qknorm {std_spread('qknorm')}x")
    print(f"fraction of head-dim tax removed by qknorm: {tax_removed}")
    print(f"total {results['duration_sec']:.0f}s")


if __name__ == "__main__":
    main()
