"""muP learning-rate transfer at char-LM scale.

CLAIM UNDER TEST (Yang & Hu, "Tensor Programs V: Tuning Large Neural Networks via
Zero-Shot Hyperparameter Transfer", arXiv:2203.03466): under the Maximal Update
Parametrization (muP) the loss-optimal learning rate is width-invariant, so it can be
tuned on a narrow proxy model and transferred; under standard parametrization (SP) it
drifts toward smaller values as width grows.

--------------------------------------------------------------------------------------
EXACTLY WHAT IS IMPLEMENTED (hand-rolled; microsoft/mup is NOT installed)
--------------------------------------------------------------------------------------
base_width n0 = 32, width multiplier m = d_model / n0 in {1, 2, 4}.

Rules follow TP5 Table 3 (the abc-parametrization table; muP column, **Adam** rows) and
Table 8 (the same rules restated in terms of a width multiplier -- the form the
microsoft/mup package ships), plus the "1/d attention" rule of TP5 Sec. 4 / Table 8:

  layer type          | SP (baseline)                  | muP (this file)
  --------------------+--------------------------------+---------------------------------
  input/embedding     | std sigma0, Adam lr eta        | std sigma0, Adam lr eta   (same)
  hidden matrices     | std 1/sqrt(fan_in), lr eta     | std 1/sqrt(fan_in), lr eta / m
  readout (d -> V)    | std 1/sqrt(d), mult 1, lr eta  | std 1/sqrt(n0), mult 1/m, lr eta
  LayerNorm gains/bias| lr eta                         | lr eta                    (same)
  attention logits    | q.k / sqrt(d_head)             | q.k * sqrt(d_head0) / d_head

Two deliberate deviations from a naive reading of the paper:
 (1) The readout row is TP5's "output weights" row (multiplier 1/fan_in, init var
     1/fan_in^2, Adam lr 1/fan_in) put through the abc-symmetry of TP5 Sec. 3.3 --
     (mult, std, lr) -> (mult/theta, theta*std, theta*lr) leaves Adam training exactly
     invariant because Adam's per-entry step size does not depend on parameter scale --
     with theta = m. That yields the (mult 1/m, width-independent init, width-independent
     lr) form the mup package actually ships: MuReadout applies output_mult/width_mult in
     the forward pass and MuAdam leaves the "vector-like" readout lr alone. Both forms are
     the same parametrization; this one makes muP == SP at m = 1.
 (2) The attention rule is written as sqrt(d_head0)/d_head rather than bare 1/d_head. The
     constant sqrt(d_head0) is the mup package's tunable `attn_mult`, fixed here to the
     value that makes muP coincide with SP at the base width. This is standard base-width
     HP alignment; without it the two parametrizations would already differ at m = 1 by a
     constant factor and the comparison would be confounded.

CONSEQUENCE, AND THE DESIGN'S MAIN CONTROL: at d = 32 = base_width the muP and SP arms
are the SAME parametrization, so their loss curves must coincide EXACTLY. run.py checks
this numerically (metrics.base_width_identity_max_abs_bpc_diff, expected 0.0). Every
difference at d = 64 and d = 128 is therefore attributable to width scaling alone.

--------------------------------------------------------------------------------------
DESIGN
--------------------------------------------------------------------------------------
6 base LRs (2^-10 .. 2^-5, integer octaves) x 3 widths (32/64/128) x {SP, muP} = 36 runs,
500 steps each, batch 16, ctx 64, 1 seed, tiny-shakespeare chars. Every run sees the
IDENTICAL batch stream and the identical seeded init draw, so the comparison is tightly
paired. This is a SHAPE experiment -- where the lr-vs-loss minimum sits -- not a quality
experiment; 500 steps at 0.03-0.42M params does not train these models out.

Headline: argmin base-LR per width under SP vs muP, both as the discrete grid argmin and
as a 3-point parabolic interpolation in log2(lr) space, plus the drift slope in octaves
per width doubling.

Plus muP's own diagnostic, the COORDINATE CHECK: activation / logit-update RMS after a
few steps, which muP is designed to keep width-invariant and SP is not.

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
    if not txt_path.exists():           # data/ is gitignored; fetch on a fresh clone
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
    """Pre-norm transformer block. `attn_scale` is the ONLY architectural muP knob."""

    def __init__(self, d, n_head, d_ff, attn_scale):
        super().__init__()
        assert d % n_head == 0
        self.n_head, self.head_dim = n_head, d // n_head
        self.attn_scale = attn_scale
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.fc = nn.Linear(d, d_ff, bias=False)
        self.out = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        sh = (B, T, self.n_head, self.head_dim)
        q = q.view(*sh).transpose(1, 2)
        k = k.view(*sh).transpose(1, 2)
        v = v.view(*sh).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * self.attn_scale     # muP ~1/d_head, SP 1/sqrt(d_head)
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        att = att.masked_fill(~mask, float("-inf")).softmax(dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(y)
        x = x + self.out(F.gelu(self.fc(self.ln2(x))))
        return x


class GPT(nn.Module):
    """nanoGPT-style char LM whose init / multipliers / lrs depend on `param` and m."""

    def __init__(self, vocab, d, p, param):
        super().__init__()
        n0 = p["base_width"]
        self.vocab, self.d, self.m, self.param = vocab, d, d / n0, param
        n_head, d_ff = p["n_head"], p["d_ff_mult"] * d
        head_dim, head_dim0 = d // n_head, n0 // n_head

        # --- RULE 1: attention logit scale --------------------------------------------
        self.attn_scale = (math.sqrt(head_dim0) / head_dim) if param == "mup" \
            else (1.0 / math.sqrt(head_dim))
        # --- RULE 3a: readout forward multiplier --------------------------------------
        self.readout_mult = (1.0 / self.m) if param == "mup" else 1.0

        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(p["block_size"], d)
        self.blocks = nn.ModuleList(
            [Block(d, n_head, d_ff, self.attn_scale) for _ in range(p["n_layer"])])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

        # --- RULE 2: init ---------------------------------------------------------------
        sigma0 = p["emb_init_std"]
        nn.init.normal_(self.tok.weight, 0.0, sigma0)         # input: width-independent
        nn.init.normal_(self.pos.weight, 0.0, sigma0)
        for b in self.blocks:                                 # hidden: 1/sqrt(fan_in), both
            for lin in (b.qkv, b.proj, b.fc, b.out):
                nn.init.normal_(lin.weight, 0.0, 1.0 / math.sqrt(lin.weight.shape[1]))
        # --- RULE 3b: readout init; SP tracks fan_in, muP freezes it at the base width --
        self.readout_init_std = (1.0 / math.sqrt(n0)) if param == "mup" else (1.0 / math.sqrt(d))
        nn.init.normal_(self.head.weight, 0.0, self.readout_init_std)

    def embed(self, idx):
        return self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))

    def forward(self, idx, return_acts=False):
        x = self.embed(idx)
        acts = {"emb": x} if return_acts else None
        for i, b in enumerate(self.blocks):
            x = b(x)
            if return_acts:
                acts["block%d" % i] = x
        logits = self.head(self.lnf(x)) * self.readout_mult   # RULE 3a
        if return_acts:
            acts["logits"] = logits
            return logits, acts
        return logits

    # --- RULE 4: per-group Adam learning rates ----------------------------------------
    def param_groups(self, base_lr):
        """input / vector / readout -> eta ; hidden matrices -> eta/m (muP) or eta (SP)."""
        hidden, other = [], []
        for name, prm in self.named_parameters():
            is_hidden = name.startswith("blocks.") and prm.dim() >= 2
            (hidden if is_hidden else other).append(prm)
        hidden_lr = base_lr / self.m if self.param == "mup" else base_lr
        groups = [{"params": other, "lr": base_lr},
                  {"params": hidden, "lr": hidden_lr}]
        return groups, hidden_lr


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
    tot, ntok = 0.0, 0
    for s in range(0, n_blocks, p["eval_batch"]):
        xb = torch.from_numpy(xs[s:s + p["eval_batch"]])
        yb = torch.from_numpy(ys[s:s + p["eval_batch"]])
        tot += float(F.cross_entropy(model(xb).reshape(-1, model.vocab), yb.reshape(-1),
                                     reduction="sum"))
        ntok += int(yb.numel())
    model.train()
    return tot / (LN2 * ntok), ntok


def train_one(vocab, d, param, lr_exp, seed, p, train_ids, val_ids):
    base_lr = 2.0 ** lr_exp
    set_seeds(seed)                                # identical init draw for every arm
    model = GPT(vocab, d, p, param)
    groups, hidden_lr = model.param_groups(base_lr)
    opt = torch.optim.AdamW(groups, lr=base_lr, betas=tuple(p["betas"]),
                            weight_decay=p["weight_decay"])
    group_base = [g["lr"] for g in opt.param_groups]
    rng = np.random.default_rng(seed)              # identical batch stream for every arm
    losses, diverged = [], False
    t0 = time.time()
    for it in range(p["steps"]):
        mult = min(1.0, (it + 1) / p["warmup"])    # linear warmup, then CONSTANT lr
        for g, bl in zip(opt.param_groups, group_base):
            g["lr"] = bl * mult
        x, y = get_batch(train_ids, rng, p["batch_size"], p["block_size"])
        loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
        lv = float(loss.detach())
        if (not math.isfinite(lv)) or lv > p["divergence_loss"]:
            diverged = True
            losses.append(lv if math.isfinite(lv) else float("nan"))
            break
        losses.append(lv)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), p["grad_clip"])
        opt.step()
    train_s = time.time() - t0
    if diverged:
        bpc, ntok = float("nan"), 0
    else:
        bpc, ntok = eval_bpc(model, val_ids, p)
        if not math.isfinite(bpc):
            diverged, bpc = True, float("nan")
    finite = [v for v in losses if math.isfinite(v)]
    rec = {
        "param": param, "d_model": int(d), "m": d / p["base_width"],
        "lr_exp": int(lr_exp), "base_lr": base_lr, "hidden_lr": hidden_lr,
        "readout_mult": model.readout_mult, "attn_scale": round(model.attn_scale, 8),
        "readout_init_std": round(model.readout_init_std, 6),
        "seed": int(seed), "n_params": n_params(model),
        "steps_completed": len(losses), "diverged": bool(diverged),
        "val_bpc": None if diverged else round(bpc, 5),
        "final_train_bpc_ma50": (None if diverged or len(finite) < 1 else
                                 round(float(np.mean(finite[-50:])) / LN2, 5)),
        "train_bpc_curve_ma25": [round(float(np.mean(finite[i:i + 25])) / LN2, 4)
                                 for i in range(0, len(finite), 25)],
        "eval_chars": ntok, "train_seconds": round(train_s, 1),
    }
    shown = "DIVERGED" if diverged else ("%.4f" % bpc)
    print("  [%-4s d=%3d lr=2^%-3d] P=%6d val_bpc=%-8s (%.1fs)"
          % (param, d, lr_exp, rec["n_params"], shown, train_s), flush=True)
    return rec


# ------------------------------------------------------------------- coordinate check
def coord_check(vocab, d, param, p, train_ids):
    """muP's own diagnostic: are activation / logit-update coordinates width-invariant?

    Records RMS of activations at init and after `coord_check_steps` Adam steps, plus the
    RMS of the CHANGE in logits on a fixed probe batch -- the quantity muP is designed to
    keep Theta(1) in width and SP is not.
    """
    base_lr = 2.0 ** p["coord_check_lr_exponent"]
    set_seeds(0)
    model = GPT(vocab, d, p, param)
    groups, _ = model.param_groups(base_lr)
    opt = torch.optim.AdamW(groups, lr=base_lr, betas=tuple(p["betas"]), weight_decay=0.0)
    rng = np.random.default_rng(0)
    probe_x, _ = get_batch(train_ids, rng, p["coord_check_batch"], p["block_size"])

    def snap():
        with torch.no_grad():
            _, acts = model(probe_x, return_acts=True)
        return {k: v.detach().clone() for k, v in acts.items()}

    a0 = snap()
    for _ in range(p["coord_check_steps"]):
        x, y = get_batch(train_ids, rng, p["coord_check_batch"], p["block_size"])
        loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), p["grad_clip"])
        opt.step()
    a1 = snap()

    def rms(t):
        return float(t.double().pow(2).mean().sqrt())

    return {
        "param": param, "d_model": int(d), "steps": p["coord_check_steps"],
        "init_rms": {k: round(rms(v), 5) for k, v in a0.items()},
        "trained_rms": {k: round(rms(v), 5) for k, v in a1.items()},
        "delta_rms": {k: round(rms(a1[k] - a0[k]), 5) for k in a0},
    }


# -------------------------------------------------------------------------- analysis
def parabolic_argmin(xs, ys):
    """Vertex of the parabola through the discrete argmin and its two neighbours.

    xs are equally spaced (integer octaves). Returns (x*, kind) where kind is
    'interior' (usable), 'edge' (minimum at a grid endpoint -> argmin is only a bound)
    or 'nonconvex' (the local parabola opens downward).
    """
    i = int(np.argmin(ys))
    if i == 0 or i == len(xs) - 1:
        return float(xs[i]), "edge"
    x0, x1, x2 = float(xs[i - 1]), float(xs[i]), float(xs[i + 1])
    y0, y1, y2 = float(ys[i - 1]), float(ys[i]), float(ys[i + 1])
    denom = y0 - 2.0 * y1 + y2
    if denom <= 0:
        return x1, "nonconvex"
    h = 0.5 * (x2 - x0)                       # grid spacing
    xstar = x1 + 0.5 * h * (y0 - y2) / denom  # standard 3-point vertex formula
    return float(np.clip(xstar, x0, x2)), "interior"


def linfit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.allclose(x, x[0]):
        return 0.0, float(np.mean(y))
    A = np.stack([x, np.ones_like(x)], 1)
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(sol[0]), float(sol[1])


def analyse(runs, p):
    out = {}
    for param in p["parametrizations"]:
        per_width = []
        for d in p["widths"]:
            rs = sorted([r for r in runs if r["param"] == param and r["d_model"] == d],
                        key=lambda r: r["lr_exp"])
            xs = [r["lr_exp"] for r in rs]
            ys = [(r["val_bpc"] if r["val_bpc"] is not None else float("inf")) for r in rs]
            fin = [i for i, v in enumerate(ys) if math.isfinite(v)]
            disc = xs[int(np.argmin(ys))]
            if len(fin) == len(ys):
                xstar, kind = parabolic_argmin(xs, ys)
            else:                                # interpolate over the finite points only
                xstar, kind = parabolic_argmin([xs[i] for i in fin], [ys[i] for i in fin])
                kind += "+diverged_tail"
            per_width.append({
                "d_model": d, "log2_width": math.log2(d),
                "lr_exps": xs,
                "val_bpc": [None if not math.isfinite(v) else v for v in ys],
                "n_diverged": len(ys) - len(fin),
                "argmin_lr_exp_discrete": int(disc),
                "argmin_lr_exp_interp": round(float(xstar), 4),
                "argmin_kind": kind,
                "best_val_bpc": None if not fin else round(min(ys[i] for i in fin), 5),
                "curve_depth_bpc": (None if len(fin) < 2 else
                                    round(max(ys[i] for i in fin) - min(ys[i] for i in fin), 5)),
            })
        # ---- the practical question muTransfer exists to answer: tune the LR on the
        # ---- narrow proxy (base width), apply it zero-shot at every larger width, and
        # ---- pay the difference against that width's own tuned optimum.
        base_exp = per_width[0]["argmin_lr_exp_discrete"]
        for w in per_width:
            j = w["lr_exps"].index(base_exp)
            at_base = w["val_bpc"][j]
            w["bpc_at_base_width_lr"] = at_base
            w["transfer_penalty_bpc"] = (None if at_base is None or w["best_val_bpc"] is None
                                         else round(at_base - w["best_val_bpc"], 5))

        # ---- does the WHOLE curve transfer, not just its argmin? mean |dbpc| between the
        # ---- two widest models over all LRs where both are finite.
        a, b = per_width[-2]["val_bpc"], per_width[-1]["val_bpc"]
        both = [(u, v) for u, v in zip(a, b) if u is not None and v is not None]
        curve_gap = round(float(np.mean([abs(u - v) for u, v in both])), 5) if both else None

        lw = [w["log2_width"] for w in per_width]
        sl_d, _ = linfit(lw, [w["argmin_lr_exp_discrete"] for w in per_width])
        sl_i, _ = linfit(lw, [w["argmin_lr_exp_interp"] for w in per_width])
        out[param] = {
            "transferred_lr_exp_from_base_width": int(base_exp),
            "transfer_penalty_bpc_by_width": {str(w["d_model"]): w["transfer_penalty_bpc"]
                                              for w in per_width},
            "transfer_penalty_bpc_at_max_width": per_width[-1]["transfer_penalty_bpc"],
            "curve_gap_mean_abs_bpc_two_widest": curve_gap,
            "per_width": per_width,
            "argmin_octaves_per_width_doubling_discrete": round(sl_d, 4),
            "argmin_octaves_per_width_doubling_interp": round(sl_i, 4),
            "argmin_spread_octaves_discrete": int(
                max(w["argmin_lr_exp_discrete"] for w in per_width)
                - min(w["argmin_lr_exp_discrete"] for w in per_width)),
            "argmin_spread_octaves_interp": round(
                max(w["argmin_lr_exp_interp"] for w in per_width)
                - min(w["argmin_lr_exp_interp"] for w in per_width), 4),
        }
    return out


def make_chart(summary, coords, p, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    widths = p["widths"]
    palette = ["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c", "#9467bd"]
    colors = {d: palette[i % len(palette)] for i, d in enumerate(widths)}
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))

    for ax, param, nice in zip(axes[:2], ["sp", "mup"], ["SP (standard param.)", "muP"]):
        for d in widths:
            w = next(x for x in summary[param]["per_width"] if x["d_model"] == d)
            xs = w["lr_exps"]
            ys = [np.nan if v is None else v for v in w["val_bpc"]]
            ax.plot(xs, ys, "o-", color=colors[d], lw=2, ms=6,
                    label="d=%d (m=%g)" % (d, d / p["base_width"]))
            ax.axvline(w["argmin_lr_exp_interp"], color=colors[d], ls=":", lw=1.6, alpha=0.85)
        base_exp = summary[param]["transferred_lr_exp_from_base_width"]
        ax.axvline(base_exp, color="k", lw=1.2, alpha=0.35)
        pen = summary[param]["transfer_penalty_bpc_at_max_width"]
        ax.annotate("transfer base LR $2^{%d}$\nto d=%d: %s bpc worse"
                    % (base_exp, widths[-1], "n/a" if pen is None else "%+.3f" % pen),
                    xy=(base_exp, 0.62), xycoords=("data", "axes fraction"),
                    fontsize=7.5, ha="left", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.25", fc="lightyellow", ec="0.6", lw=0.6))
        sl = summary[param]["argmin_octaves_per_width_doubling_interp"]
        sp = summary[param]["argmin_spread_octaves_interp"]
        ax.set_title("%s\nargmin drift %+.2f oct / width doubling (spread %.2f oct)"
                     % (nice, sl, sp))
        ax.set_xlabel("base learning rate (log2)")
        ax.set_ylabel("val bits/char after %d steps" % p["steps"])
        ax.set_xticks(p["lr_exponents"])
        ax.set_xticklabels(["$2^{%d}$" % e for e in p["lr_exponents"]])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    lo = min(a.get_ylim()[0] for a in axes[:2])
    hi = max(a.get_ylim()[1] for a in axes[:2])
    n_div = sum(w["n_diverged"] for param in ["sp", "mup"]
                for w in summary[param]["per_width"])
    for a in axes[:2]:
        a.set_ylim(lo, hi)
        if n_div:
            a.text(0.02, 0.03, "%d diverged LR(s) omitted (broken line)" % n_div,
                   transform=a.transAxes, fontsize=7, alpha=0.6)

    ax = axes[2]
    for param, mk, nice in [("sp", "s--", "SP"), ("mup", "o-", "muP")]:
        w = summary[param]["per_width"]
        ax.plot([x["log2_width"] for x in w], [x["argmin_lr_exp_interp"] for x in w], mk,
                lw=2, ms=9,
                label="%s (%+.2f oct/doubling)"
                      % (nice, summary[param]["argmin_octaves_per_width_doubling_interp"]))
        ax.plot([x["log2_width"] for x in w], [x["argmin_lr_exp_discrete"] for x in w],
                mk[0], ms=5, alpha=0.35)
    ax.set_xticks([math.log2(d) for d in widths])
    ax.set_xticklabels([str(d) for d in widths])
    ax.set_xlabel("width d_model")
    ax.set_ylabel("argmin base lr (log2)")
    ax.set_title("HEADLINE: does the optimum move?\nlarge = parabolic interp, small = grid argmin")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[3]
    last_block = "block%d" % (p["n_layer"] - 1)
    for param, mk, nice in [("sp", "s--", "SP"), ("mup", "o-", "muP")]:
        cs = sorted([c for c in coords if c["param"] == param], key=lambda c: c["d_model"])
        ax.plot([c["d_model"] for c in cs], [c["delta_rms"]["logits"] for c in cs], mk,
                lw=2, ms=9, label="%s: logit change" % nice)
        ax.plot([c["d_model"] for c in cs], [c["delta_rms"][last_block] for c in cs], mk,
                lw=1.2, ms=5, alpha=0.45, label="%s: last-block act change" % nice)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(widths)
    ax.set_xticklabels([str(d) for d in widths])
    ax.set_xlabel("width d_model")
    ax.set_ylabel("RMS change after %d steps" % p["coord_check_steps"])
    ax.set_title("Coordinate check (muP's own diagnostic)\nflat = width-invariant updates")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)

    fig.suptitle("muP LR transfer at char-LM scale: %d LRs x widths %s x {SP, muP}, "
                 "%d steps, tiny-shakespeare (base_width %d)"
                 % (len(p["lr_exponents"]), widths, p["steps"], p["base_width"]), fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------------------ main
def main():
    cfg = load_config()
    p = cfg["params"]
    seed = int(cfg.get("seed", 0))
    set_seeds(seed)
    t0 = time.time()
    train_ids, val_ids, vocab = load_data(cfg)
    print("data: %d train chars, %d val chars, vocab %d" % (len(train_ids), len(val_ids), vocab),
          flush=True)

    last_block = "block%d" % (p["n_layer"] - 1)
    print("coordinate check (muP diagnostic)...", flush=True)
    coords = [coord_check(vocab, d, param, p, train_ids)
              for param in p["parametrizations"] for d in p["widths"]]
    for c in coords:
        print("  [%-4s d=%3d] logit dRMS=%.4f  last-block dRMS=%.4f  logit RMS@init=%.4f"
              % (c["param"], c["d_model"], c["delta_rms"]["logits"],
                 c["delta_rms"][last_block], c["init_rms"]["logits"]), flush=True)

    runs, stop = [], False
    for param in p["parametrizations"]:
        for d in p["widths"]:
            for e in p["lr_exponents"]:
                for sd in p["seeds"]:
                    runs.append(train_one(vocab, d, param, e, sd, p, train_ids, val_ids))
                    if time.time() - t0 > p["time_cap_s"]:
                        print("TIME CAP HIT - stopping sweep early", flush=True)
                        stop = True
                if stop:
                    break
            if stop:
                break
        if stop:
            break

    # ---- control: at the base width muP and SP ARE the same parametrization ----------
    base_d = p["base_width"]
    diffs = []
    for e in p["lr_exponents"]:
        a = next((r for r in runs if r["param"] == "sp" and r["d_model"] == base_d
                  and r["lr_exp"] == e), None)
        b = next((r for r in runs if r["param"] == "mup" and r["d_model"] == base_d
                  and r["lr_exp"] == e), None)
        if a and b and a["val_bpc"] is not None and b["val_bpc"] is not None:
            diffs.append(abs(a["val_bpc"] - b["val_bpc"]))
    base_identity = round(max(diffs), 8) if diffs else None

    summary = analyse(runs, p)
    cc = {}
    for k in ("logits", last_block):
        for param in p["parametrizations"]:
            cs = sorted([c for c in coords if c["param"] == param], key=lambda c: c["d_model"])
            v = [c["delta_rms"][k] for c in cs]
            cc["%s_delta_rms_%s" % (param, k)] = v
            cc["%s_delta_rms_%s_ratio_wide_over_base" % (param, k)] = (
                round(v[-1] / v[0], 4) if v and v[0] > 0 else None)
    for param in p["parametrizations"]:
        cs = sorted([c for c in coords if c["param"] == param], key=lambda c: c["d_model"])
        cc["%s_init_rms_logits" % param] = [c["init_rms"]["logits"] for c in cs]

    metrics = {
        "headline": (
            "argmin base-lr drift: SP %+.2f octaves per width doubling, muP %+.2f; "
            "argmin spread over d=%d..%d: SP %.2f oct, muP %.2f oct; "
            "zero-shot transfer penalty at d=%d (LR tuned at base width %d): "
            "SP %s bpc vs muP %s bpc"
            % (summary["sp"]["argmin_octaves_per_width_doubling_interp"],
               summary["mup"]["argmin_octaves_per_width_doubling_interp"],
               p["widths"][0], p["widths"][-1],
               summary["sp"]["argmin_spread_octaves_interp"],
               summary["mup"]["argmin_spread_octaves_interp"],
               p["widths"][-1], p["base_width"],
               summary["sp"]["transfer_penalty_bpc_at_max_width"],
               summary["mup"]["transfer_penalty_bpc_at_max_width"])),
        "base_width": p["base_width"],
        "widths": p["widths"],
        "lr_exponents": p["lr_exponents"],
        "steps": p["steps"],
        "n_runs": len(runs),
        "n_diverged": sum(1 for r in runs if r["diverged"]),
        "base_width_identity_max_abs_bpc_diff": base_identity,
        "params_by_width": {str(d): next((r["n_params"] for r in runs if r["d_model"] == d), None)
                            for d in p["widths"]},
        "sp": {k: v for k, v in summary["sp"].items() if k != "per_width"},
        "mup": {k: v for k, v in summary["mup"].items() if k != "per_width"},
        "argmin_table": {
            param: {str(w["d_model"]): {"discrete_log2": w["argmin_lr_exp_discrete"],
                                        "interp_log2": w["argmin_lr_exp_interp"],
                                        "kind": w["argmin_kind"],
                                        "best_val_bpc": w["best_val_bpc"],
                                        "bpc_at_base_width_lr": w["bpc_at_base_width_lr"],
                                        "transfer_penalty_bpc": w["transfer_penalty_bpc"],
                                        "curve_depth_bpc": w["curve_depth_bpc"],
                                        "n_diverged": w["n_diverged"]}
                    for w in summary[param]["per_width"]}
            for param in p["parametrizations"]},
        "coord_check": cc,
        "per_width_curves": {param: summary[param]["per_width"] for param in p["parametrizations"]},
    }

    make_chart(summary, coords, p, HERE / "chart.png")

    results = {
        "id": cfg.get("id", "unknown"),
        "git_commit": git_sha(),
        "seed": seed,
        "duration_sec": round(time.time() - t0, 2),
        "metrics": metrics,
        "runs": runs,
        "coord_check_raw": coords,
        "env": env_info(),
        "status": "done",
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(metrics["headline"])
    print("base-width identity control (SP vs muP at d=%d, max |dbpc|): %s"
          % (base_d, base_identity))
    print("total %.1fs" % results["duration_sec"])


if __name__ == "__main__":
    main()
