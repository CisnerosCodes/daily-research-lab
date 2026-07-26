"""SwiGLU vs GELU at EQUAL parameters -- the iso-parameter control most demos skip.

A GLU-style FFN uses three weight matrices (gate W, up V, down W2) where a vanilla FFN
uses two (up W, down W2). At the same d_ff a SwiGLU block therefore carries 1.5x the FFN
parameters and 1.5x the FFN FLOPs. Shazeer (arXiv:2002.05202) handles this by shrinking
d_ff by 2/3, i.e. d_ff = (8/3) * d_model instead of 4 * d_model, and reports SwiGLU still
winning. Most tutorials and blog demos do NOT apply the correction.

Four arms at d_model = 128, everything else identical:

    gelu_4x     mlp,    d_ff = 512   -> 2 * d * d_ff = 131,072 FFN weights / layer
    swiglu_iso  swiglu, d_ff = 341   -> 3 * d * d_ff = 130,944 FFN weights / layer  (iso-param)
    swiglu_4x   swiglu, d_ff = 512   -> 3 * d * d_ff = 196,608 FFN weights / layer  (UNFAIR)
    relu_4x     mlp,    d_ff = 512   -> 2 * d * d_ff = 131,072 FFN weights / layer  (reference)

Headline: val bits-per-character, gelu_4x vs swiglu_iso, with the per-seed spread next to
it. Is the iso-param gap seed-separated, or is it noise? And how much of the swiglu_4x
margin is bought with the extra parameters rather than with the gating nonlinearity?

Every seed replays an IDENTICAL batch stream across all four arms, so the comparison is
paired. Init cannot be shared (the FFN shapes differ by construction), so the seed spread
is the honest error bar.

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


# ------------------------------------------------------------------------------- FFN
ACTS = {"gelu": F.gelu, "relu": F.relu, "silu": F.silu}


class MLP(nn.Module):
    """Vanilla two-matrix FFN: out( act( fc(x) ) ).  2 * d * d_ff weights."""

    def __init__(self, d, d_ff, act):
        super().__init__()
        self.fc = nn.Linear(d, d_ff, bias=False)
        self.out = nn.Linear(d_ff, d, bias=False)
        self.act = ACTS[act]

    def forward(self, x):
        return self.out(self.act(self.fc(x)))


class GLU(nn.Module):
    """GLU-style three-matrix FFN: out( act(fc(x)) * gate(x) ).  3 * d * d_ff weights.

    act='silu' gives SwiGLU (Shazeer 2020). The gate branch is linear, as in the paper.
    """

    def __init__(self, d, d_ff, act):
        super().__init__()
        self.fc = nn.Linear(d, d_ff, bias=False)      # W  (activated branch)
        self.gate = nn.Linear(d, d_ff, bias=False)    # V  (linear branch)
        self.out = nn.Linear(d_ff, d, bias=False)     # W2 (down projection)
        self.act = ACTS[act]

    def forward(self, x):
        return self.out(self.act(self.fc(x)) * self.gate(x))


def make_ffn(kind, d, d_ff, act):
    if kind == "mlp":
        return MLP(d, d_ff, act)
    if kind == "swiglu":
        return GLU(d, d_ff, act)
    raise ValueError(kind)


def ffn_weights(kind, d, d_ff):
    return (2 if kind == "mlp" else 3) * d * d_ff


def ffn_macs_per_token(kind, d, d_ff):
    """Multiply-accumulates in one forward pass of the FFN, per token."""
    return (2 if kind == "mlp" else 3) * d * d_ff


# ----------------------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, d, n_head, kind, d_ff, act):
        super().__init__()
        assert d % n_head == 0
        self.n_head, self.head_dim = n_head, d // n_head
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ffn = make_ffn(kind, d, d_ff, act)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        shape = (B, T, self.n_head, self.head_dim)
        q = q.view(*shape).transpose(1, 2)
        k = k.view(*shape).transpose(1, 2)
        v = v.view(*shape).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, D))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab, d, n_layer, n_head, kind, d_ff, act, block_size):
        super().__init__()
        self.vocab, self.block_size = vocab, block_size
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(block_size, d)
        self.blocks = nn.ModuleList(
            [Block(d, n_head, kind, d_ff, act) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.apply(self._init)
        for name, prm in self.named_parameters():   # nanoGPT residual-projection scaling
            if name.endswith("proj.weight") or name.endswith("ffn.out.weight"):
                nn.init.normal_(prm, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T))
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


def train_one(vocab, arm, seed, p, train_ids, val_ids):
    d = p["d_model"]
    set_seeds(seed)
    model = GPT(vocab, d, p["n_layer"], p["n_head"],
                arm["kind"], arm["d_ff"], arm["act"], p["block_size"])
    decay = [q for q in model.parameters() if q.dim() >= 2]
    nodecay = [q for q in model.parameters() if q.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": p["weight_decay"]},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=p["lr"], betas=(0.9, 0.95),
    )
    rng = np.random.default_rng(seed)     # IDENTICAL batch stream across all arms
    steps, warm = p["steps"], p["warmup"]
    losses = []
    t0 = time.time()
    for it in range(steps):
        if it < warm:
            lr = p["lr"] * (it + 1) / warm
        else:
            prog = (it - warm) / max(1, steps - warm)
            lr = p["lr"] * (p["lr_min_frac"]
                            + (1 - p["lr_min_frac"]) * 0.5 * (1 + math.cos(math.pi * prog)))
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
    curve = [round(float(np.mean(losses[i:i + 25])), 4) for i in range(0, steps, 25)]
    ffn_w = ffn_weights(arm["kind"], d, arm["d_ff"]) * p["n_layer"]
    rec = {
        "arm": arm["name"], "kind": arm["kind"], "act": arm["act"], "d_ff": arm["d_ff"],
        "d_ff_over_d_model": round(arm["d_ff"] / d, 4),
        "seed": int(seed),
        "n_params": n_params(model),
        "ffn_params": ffn_w,
        "ffn_macs_per_token_fwd": ffn_macs_per_token(arm["kind"], d, arm["d_ff"]) * p["n_layer"],
        "train_seconds": round(train_s, 1),
        "final_train_loss_ma50": round(float(np.mean(losses[-50:])), 4),
        "final_train_bpc_ma50": round(float(np.mean(losses[-50:])) / LN2, 4),
        "train_loss_curve_ma25": curve,
        "val_bpc": round(ev["bpc"], 5),
        "eval_chars": ev["eval_chars"],
    }
    print(f"  [{arm['name']:<10s} d_ff={arm['d_ff']:4d} seed={seed}] P={rec['n_params']:,} "
          f"ffnP={ffn_w:,} bpc={rec['val_bpc']:.4f} ({rec['train_seconds']}s)", flush=True)
    return rec


# -------------------------------------------------------------------------- analysis
def paired_delta(runs, a, b, seeds):
    """b minus a, per seed. Negative => b is better (lower bpc)."""
    out = {}
    for sd in seeds:
        va = next(r["val_bpc"] for r in runs if r["arm"] == a and r["seed"] == sd)
        vb = next(r["val_bpc"] for r in runs if r["arm"] == b and r["seed"] == sd)
        out[str(sd)] = round(vb - va, 5)
    vals = np.array(list(out.values()))
    same_sign = bool(np.all(vals > 0) or np.all(vals < 0))
    return {
        "per_seed": out,
        "mean": round(float(vals.mean()), 5),
        "std": round(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, 5),
        "min": round(float(vals.min()), 5),
        "max": round(float(vals.max()), 5),
        "all_seeds_same_sign": same_sign,
    }


def wall_clock_matched(runs, summary, p, ref="gelu_4x"):
    """Approximate 'who is ahead at equal CPU SECONDS', from the stored training curves.

    The arms are iso-parameter and (for the iso pair) iso-MAC, but they are NOT iso-wall-
    clock: three skinnier GEMMs plus an elementwise product cost more per step than two
    fatter GEMMs. So we ask: within the reference arm's training-time budget, how many
    steps does each arm complete, and what is its smoothed TRAIN bits-per-char there?

    Caveats, on purpose: this is train loss (not val bpc), and the cosine LR schedule was
    written for the full step count, so truncating an arm mid-schedule leaves its LR
    undecayed and handicaps it slightly. Indicative, not a substitute for a real
    time-budgeted rerun.
    """
    sec = {s["arm"]: s["train_seconds_mean"] for s in summary}
    steps = p["steps"]
    budget = sec[ref]
    out = {"reference_arm": ref, "budget_seconds": round(budget, 1),
           "seconds_per_step": {k: round(v / steps, 5) for k, v in sec.items()},
           "by_arm": {}}
    for s in summary:
        arm = s["arm"]
        cs = np.array([r["train_loss_curve_ma25"] for r in runs if r["arm"] == arm])
        curve = cs.mean(0) / LN2
        n_steps = int(budget / (sec[arm] / steps))
        idx = min(len(curve) - 1, max(0, n_steps // 25 - 1))
        out["by_arm"][arm] = {
            "steps_within_budget": n_steps,
            "train_bpc_ma25_at_budget": round(float(curve[idx]), 4),
            "train_bpc_ma25_at_full_steps": round(float(curve[-1]), 4),
            "slowdown_vs_reference": round(sec[arm] / budget, 4),
        }
    ref_bpc = out["by_arm"][ref]["train_bpc_ma25_at_budget"]
    out["delta_vs_reference_at_budget"] = {
        s["arm"]: round(out["by_arm"][s["arm"]]["train_bpc_ma25_at_budget"] - ref_bpc, 4)
        for s in summary if s["arm"] != ref}
    return out


def make_chart(summary, runs, deltas, p, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [s["arm"] for s in summary]
    means = np.array([s["val_bpc_mean"] for s in summary])
    lo = np.array([s["val_bpc_min"] for s in summary])
    hi = np.array([s["val_bpc_max"] for s in summary])
    seeds = sorted({r["seed"] for r in runs})
    colors = {"gelu_4x": "#1f77b4", "swiglu_iso": "#2ca02c",
              "swiglu_4x": "#d62728", "relu_4x": "#7f7f7f"}
    cols = [colors.get(n, "#333333") for n in names]
    xs = np.arange(len(names))

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    # --- panel 1: the headline, val bpc per arm with every seed shown
    ax = axes[0]
    ax.bar(xs, means, color=cols, alpha=0.35, width=0.6)
    ax.errorbar(xs, means, yerr=[means - lo, hi - means], fmt="none", ecolor="k",
                capsize=5, lw=1.6)
    for sd in seeds:
        ys = [next(r["val_bpc"] for r in runs if r["arm"] == n and r["seed"] == sd)
              for n in names]
        ax.plot(xs, ys, "o", ms=6, mfc="none", mec="k", mew=1.2, alpha=0.8,
                label="individual seeds" if sd == seeds[0] else None)
    span = max(hi.max() - lo.min(), 1e-6)
    for i, s in enumerate(summary):
        ax.text(xs[i], hi[i] + 0.06 * span, f"{s['val_bpc_mean']:.4f}", ha="center", fontsize=8)
        ax.text(xs[i], lo[i] - 0.22 * span, f"{s['n_params']/1000:.1f}k params",
                ha="center", fontsize=7, color="dimgray")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{s['arm']}\nd_ff={s['d_ff']}" for s in summary], fontsize=8)
    ax.set_ylabel("val bits per character")
    ax.set_ylim(lo.min() - 0.42 * span, hi.max() + 0.22 * span)
    ax.set_title("Headline: val bpc per FFN arm\n(circles = individual seeds, bar = seed range)",
                 fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="upper right")

    # --- panel 2: paired per-seed deltas vs the GELU-4x baseline
    ax = axes[1]
    keys = [k for k in deltas if k.startswith("vs_gelu_4x__")]
    labs = [k.split("__")[1] for k in keys]
    xs2 = np.arange(len(keys))
    ax.axhline(0, color="k", lw=1.2)
    for i, k in enumerate(keys):
        dd = deltas[k]
        vals = list(dd["per_seed"].values())
        ax.scatter([i] * len(vals), vals, s=55, facecolors="none",
                   edgecolors=colors.get(labs[i], "k"), lw=1.6, zorder=3)
        ax.plot([i - 0.24, i + 0.24], [dd["mean"]] * 2, lw=3,
                color=colors.get(labs[i], "k"), zorder=2)
        ax.text(i + 0.27, dd["mean"], f"{dd['mean']:+.4f}", va="center", fontsize=8)
    ax.set_xticks(xs2)
    ax.set_xticklabels(labs, fontsize=9)
    ax.set_xlim(-0.6, len(keys) - 0.25)
    ax.set_ylabel("delta val bpc vs gelu_4x  (negative = beats GELU)")
    ax.set_title("Paired per-seed deltas vs GELU-4x\n(same batch stream per seed)", fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    # --- panel 3: bpc vs parameter count, the confound made visible
    ax = axes[2]
    for i, s in enumerate(summary):
        ax.errorbar([s["n_params"]], [s["val_bpc_mean"]],
                    yerr=[[s["val_bpc_mean"] - s["val_bpc_min"]],
                          [s["val_bpc_max"] - s["val_bpc_mean"]]],
                    fmt="o", ms=10, capsize=4, color=cols[i], label=s["arm"])
    ax.set_xlabel("total parameters")
    ax.set_ylabel("val bits per character")
    ax.set_title("The confound, made visible:\nbpc vs parameter count", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle(
        "SwiGLU vs GELU at equal parameters - tiny char LM, "
        f"{p['steps']} steps x batch {p['batch_size']} x {p['block_size']}, "
        f"{len(seeds)} seeds", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    cfg = load_config()
    p = cfg["params"]
    t_start = time.time()

    train_ids, val_ids, vocab = load_data(cfg)
    print(f"data: {len(train_ids)} train chars, {len(val_ids)} val chars, vocab {vocab}",
          flush=True)

    arms = p["arms"]
    d = p["d_model"]
    print("planned FFN weights per layer:")
    for a in arms:
        print(f"  {a['name']:<10s} kind={a['kind']:<6s} d_ff={a['d_ff']:4d} "
              f"({a['d_ff']/d:.3f} x d_model) -> {ffn_weights(a['kind'], d, a['d_ff']):,}")

    runs = []
    for a in arms:
        for seed in p["seeds"]:
            runs.append(train_one(vocab, a, seed, p, train_ids, val_ids))

    summary = []
    for a in arms:
        rs = [r for r in runs if r["arm"] == a["name"]]
        b = np.array([r["val_bpc"] for r in rs])
        summary.append({
            "arm": a["name"], "kind": a["kind"], "act": a["act"], "d_ff": a["d_ff"],
            "d_ff_over_d_model": round(a["d_ff"] / d, 4),
            "n_params": rs[0]["n_params"],
            "ffn_params": rs[0]["ffn_params"],
            "ffn_macs_per_token_fwd": rs[0]["ffn_macs_per_token_fwd"],
            "val_bpc_per_seed": [round(float(u), 5) for u in b],
            "val_bpc_mean": round(float(b.mean()), 5),
            "val_bpc_std": round(float(b.std(ddof=1)) if len(b) > 1 else 0.0, 5),
            "val_bpc_min": round(float(b.min()), 5),
            "val_bpc_max": round(float(b.max()), 5),
            "val_bpc_seed_spread": round(float(b.max() - b.min()), 5),
            "train_loss_mean_nats": round(float(np.mean([r["final_train_loss_ma50"] for r in rs])), 5),
            "train_seconds_mean": round(float(np.mean([r["train_seconds"] for r in rs])), 1),
        })
    by_arm = {s["arm"]: s for s in summary}

    # ---- iso-parameter verification for the designated pair
    a_name, b_name = p["iso_pair"]
    pa, pb = by_arm[a_name]["n_params"], by_arm[b_name]["n_params"]
    iso_rel_diff = abs(pa - pb) / pa
    iso_ok = iso_rel_diff < p["iso_param_tol_frac"]
    assert iso_ok, (f"iso-param pair {a_name}/{b_name} differ by {iso_rel_diff:.4%} "
                    f"(> {p['iso_param_tol_frac']:.1%}): {pa} vs {pb}")

    seeds = p["seeds"]
    deltas = {f"vs_{a_name}__{s['arm']}": paired_delta(runs, a_name, s["arm"], seeds)
              for s in summary if s["arm"] != a_name}
    # also the direct iso-param-vs-param-advantaged SwiGLU comparison
    deltas["swiglu_iso__to__swiglu_4x"] = paired_delta(runs, "swiglu_iso", "swiglu_4x", seeds)

    d_iso = deltas[f"vs_{a_name}__{b_name}"]                     # swiglu_iso - gelu_4x
    d_unfair = deltas[f"vs_{a_name}__swiglu_4x"]                 # swiglu_4x  - gelu_4x
    d_relu = deltas[f"vs_{a_name}__relu_4x"]                     # relu_4x    - gelu_4x

    mean_seed_spread = float(np.mean([s["val_bpc_seed_spread"] for s in summary]))

    # verdict for the headline pair: seed-separated only if every seed agrees on the sign
    # AND the mean gap exceeds the typical within-arm seed spread.
    if d_iso["all_seeds_same_sign"] and abs(d_iso["mean"]) > mean_seed_spread:
        iso_verdict = ("seed-separated: SwiGLU wins at iso-param" if d_iso["mean"] < 0
                       else "seed-separated: GELU wins at iso-param")
    elif d_iso["all_seeds_same_sign"]:
        iso_verdict = ("consistent sign but gap smaller than the mean seed spread -- "
                       "suggestive, not separated")
    else:
        iso_verdict = "tie / noise: seeds disagree on the sign"

    # how much of the swiglu_4x margin survives the parameter correction?
    if d_unfair["mean"] < 0:
        frac_surviving = d_iso["mean"] / d_unfair["mean"]
        frac_from_params = 1.0 - frac_surviving
    else:
        frac_surviving, frac_from_params = None, None

    param_excess = by_arm["swiglu_4x"]["n_params"] / by_arm["gelu_4x"]["n_params"] - 1.0
    ffn_excess = by_arm["swiglu_4x"]["ffn_params"] / by_arm["gelu_4x"]["ffn_params"] - 1.0

    ranking = sorted(summary, key=lambda s: s["val_bpc_mean"])
    per_seed_rank = {}
    for sd in seeds:
        rs = sorted([r for r in runs if r["seed"] == sd], key=lambda r: r["val_bpc"])
        per_seed_rank[str(sd)] = [r["arm"] for r in rs]
    seeds_agree_on_best = len({v[0] for v in per_seed_rank.values()}) == 1

    metrics = {
        "headline": ("val bpc, GELU-4x vs SwiGLU-8/3x at matched parameters, with the "
                     "param-advantaged SwiGLU-4x and ReLU-4x alongside"),
        "iso_param_verdict": iso_verdict,
        "iso_param_pair": [a_name, b_name],
        "iso_param_counts": {a_name: pa, b_name: pb},
        "iso_param_rel_diff": round(iso_rel_diff, 6),
        "iso_param_within_1pct": bool(iso_ok),
        "n_params_by_arm": {s["arm"]: s["n_params"] for s in summary},
        "ffn_params_by_arm": {s["arm"]: s["ffn_params"] for s in summary},
        "ffn_macs_per_token_by_arm": {s["arm"]: s["ffn_macs_per_token_fwd"] for s in summary},
        "val_bpc_mean_by_arm": {s["arm"]: s["val_bpc_mean"] for s in summary},
        "val_bpc_seed_spread_by_arm": {s["arm"]: s["val_bpc_seed_spread"] for s in summary},
        "mean_seed_spread_bpc": round(mean_seed_spread, 5),
        "delta_swiglu_iso_minus_gelu4x": d_iso,
        "delta_swiglu_4x_minus_gelu4x": d_unfair,
        "delta_relu4x_minus_gelu4x": d_relu,
        "delta_swiglu_4x_minus_swiglu_iso": deltas["swiglu_iso__to__swiglu_4x"],
        "swiglu_4x_param_excess_frac": round(param_excess, 5),
        "swiglu_4x_ffn_param_excess_frac": round(ffn_excess, 5),
        "frac_of_swiglu4x_margin_surviving_isoparam": (None if frac_surviving is None
                                                       else round(frac_surviving, 4)),
        "frac_of_swiglu4x_margin_attributable_to_params": (None if frac_from_params is None
                                                           else round(frac_from_params, 4)),
        "ranking_best_first": [s["arm"] for s in ranking],
        "best_arm": ranking[0]["arm"],
        "best_val_bpc_mean": ranking[0]["val_bpc_mean"],
        "per_seed_ranking_best_first": per_seed_rank,
        "seeds_agree_on_best": seeds_agree_on_best,
        "arm_spread_bpc": round(float(max(s["val_bpc_mean"] for s in summary)
                                      - min(s["val_bpc_mean"] for s in summary)), 5),
        "by_arm": summary,
        "all_paired_deltas": deltas,
        "train_steps": p["steps"],
        "tokens_per_run": p["steps"] * p["batch_size"] * p["block_size"],
        "seeds": seeds,
        "n_runs": len(runs),
        "eval_chars": runs[0]["eval_chars"],
        "shared_batch_stream_per_seed": True,
        "wall_clock_matched_approx": wall_clock_matched(runs, summary, p, ref=a_name),
        "runs": runs,
    }

    make_chart(summary, runs, deltas, p, HERE / "chart.png")

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
    print(f"iso-param pair {a_name}={pa:,} vs {b_name}={pb:,} "
          f"(rel diff {iso_rel_diff:.4%}, within 1%: {iso_ok})")
    for s in summary:
        print(f"  {s['arm']:<10s} d_ff={s['d_ff']:4d}  P={s['n_params']:,}  "
              f"bpc={s['val_bpc_mean']:.4f} +-{s['val_bpc_std']:.4f} "
              f"(range {s['val_bpc_seed_spread']:.4f})  {s['train_seconds_mean']}s")
    print(f"iso-param delta (swiglu_iso - gelu_4x): {d_iso['mean']:+.4f} "
          f"per-seed {list(d_iso['per_seed'].values())}")
    print(f"unfair delta   (swiglu_4x  - gelu_4x): {d_unfair['mean']:+.4f} "
          f"per-seed {list(d_unfair['per_seed'].values())}")
    print(f"relu delta     (relu_4x    - gelu_4x): {d_relu['mean']:+.4f}")
    print(f"mean seed spread: {mean_seed_spread:.4f} bpc")
    print(f"VERDICT: {iso_verdict}")
    print(f"total {results['duration_sec']:.0f}s")


if __name__ == "__main__":
    main()
