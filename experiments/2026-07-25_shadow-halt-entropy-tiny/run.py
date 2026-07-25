"""Shadow E3 - entropy-based adaptive exit (Ouro-style) on a tiny looped char LM.

Question: on a weight-tied block looped up to k=4 times, does a per-token
entropy exit trace a BETTER val-bits/char vs mean-loops/token frontier than
simply fixing the loop count? Or does it collapse to a fixed depth (the
prediction of the TRM critique, arXiv:2512.11847)?

Design
------
Four trained models (2 seeds each), all the SAME weight-tied block:
  fixed_k1 / fixed_k2 / fixed_k4  - CE only on the readout after the last loop
  deepsup_k4                      - CE averaged over readouts after loops 1..4
Only `deepsup_k4` has calibrated intermediate readouts, which is what makes an
early exit legal at all. (The backlog allowed an inference-time exit on a plain
fixed-k4 model as a shrink; we run that too, as an ABLATION, and it is
catastrophic - see `naive_exit_on_fixed_k4` in results.json.)

Then, with no extra training, we sweep three inference-time policies on the
deepsup model and plot them all on one bits/char vs mean-loops/token axis:
  * fixed        - run every token exactly k in {1,2,3,4} loops
  * entropy exit - after loop i, a token whose predictive entropy <= tau stops;
                   its hidden state freezes (it still serves as a key/value for
                   other positions) and its loop-i distribution is its answer
  * random exit  - matched-compute control: each active token exits with prob p,
                   p solved so the mean loops/token matches an entropy point.
                   This separates "entropy is informative" from "any early exit
                   at this average depth is fine".

Compute accounting: mean loops/token is the headline compute proxy (as the
backlog asks). We also report a FLOP-honest x-axis that charges the entropy
policy for the extra LM-head evaluation it needs at every loop. Nothing here
claims wall-clock savings - dense batched inference cannot realise them.

Data: tiny-shakespeare, char level (substituted for TinyStories; see README).
Deterministic, CPU-only, single-thread. Writes results.json + chart.png.

Usage:  python run.py            (full run, ~8-10 min on 1 CPU thread)
        python run.py --smoke    (tiny step budget, for a plumbing check)
"""
import json, math, os, random, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "tinyshakespeare.txt"
DATA_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
            "master/data/tinyshakespeare/input.txt")
LOG2 = math.log(2.0)


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


# ----------------------------- model ---------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)   # 2 shared cores on this box - be a good neighbour


class Block(nn.Module):
    """Pre-norm transformer block (nanoGPT style, no biases in attn)."""

    def __init__(self, d, h, dff):
        super().__init__()
        self.h, self.dh = h, d // h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.fc1 = nn.Linear(d, dff)
        self.fc2 = nn.Linear(dff, d)

    def forward(self, x, mask):
        B, T, D = x.shape
        y = self.ln1(x)
        q, k, v = self.qkv(y).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        att = att.masked_fill(mask, float("-inf")).softmax(-1)
        x = x + self.proj((att @ v).transpose(1, 2).reshape(B, T, D))
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class LoopedCharLM(nn.Module):
    """One weight-tied block applied up to k_max times, with a readout usable
    after ANY loop (that is what an early exit needs)."""

    def __init__(self, vocab, d, h, dff, k_max, block_size):
        super().__init__()
        self.k_max, self.vocab = k_max, vocab
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(block_size, d)
        self.block = Block(d, h, dff)
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.register_buffer(
            "mask", torch.triu(torch.ones(block_size, block_size, dtype=torch.bool), 1))
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def embed(self, idx):
        T = idx.shape[1]
        return self.tok(idx) + self.pos(torch.arange(T))[None]

    def readout(self, x):
        return self.head(self.ln_f(x))

    def forward_all_loops(self, idx, k):
        """Return the list of readouts after loops 1..k (training path)."""
        x = self.embed(idx)
        m = self.mask[:idx.shape[1], :idx.shape[1]]
        outs = []
        for _ in range(k):
            x = self.block(x, m)
            outs.append(self.readout(x))
        return outs


def flops_per_token(d, dff, T, vocab):
    """Analytic forward FLOPs per token (mult+add = 2 flops), causal attention."""
    block = (
        2 * (3 * d * d)        # qkv
        + 2 * (d * d)          # out proj
        + 2 * (d * dff) * 2    # ffn up + down
        + 2 * 2 * d * (T / 2)  # QK^T and AV, causal -> avg T/2 keys
    )
    head = 2 * d * vocab       # LM head (the halting probe's extra cost)
    return block, head


# ----------------------------- data ----------------------------------------
def load_data(train_frac):
    import numpy as np
    if not DATA.exists():
        DATA.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(DATA_URL, DATA)
    text = DATA.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in text], dtype=np.uint16)
    n = int(len(ids) * train_frac)
    return ids[:n], ids[n:], len(chars), len(text)


def get_batch(data, rng, B, T):
    import numpy as np
    ix = rng.integers(0, len(data) - T - 1, size=B)
    x = np.stack([data[i:i + T] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1:i + 1 + T] for i in ix]).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


def fixed_val_batches(val, B, T, n):
    """The SAME val batches for every run and every eval mode."""
    import numpy as np
    rng = np.random.default_rng(987654321)
    return [get_batch(val, rng, B, T) for _ in range(n)]


# ------------------------- evaluation policies ------------------------------
@torch.no_grad()
def eval_fixed(model, batches, k):
    """Every token gets exactly k loops. Returns (bpc, mean_loops=k)."""
    model.eval()
    tot, ntok = 0.0, 0
    for x, y in batches:
        logits = model.forward_all_loops(x, k)[-1]
        tot += float(F.cross_entropy(logits.reshape(-1, model.vocab),
                                     y.reshape(-1), reduction="sum"))
        ntok += y.numel()
    model.train()
    return (tot / ntok) / LOG2, float(k)


@torch.no_grad()
def eval_exit(model, batches, k_max, tau=None, p_random=None, gen=None):
    """Per-token early exit.

    tau      : entropy threshold in bits; a token with H <= tau stops.
    p_random : matched-compute control; each active token stops with prob p.
    Exactly one of the two must be given.

    A halted token's hidden state is frozen but still acts as a key/value for
    other positions (standard for early-exit transformers). Its recorded
    prediction is its readout at the loop it halted on.
    """
    assert (tau is None) != (p_random is None)
    model.eval()
    V = model.vocab
    tot, ntok = 0.0, 0
    loops_sum = 0.0
    exit_hist = [0] * (k_max + 1)     # index i = tokens whose last loop was i
    for x, y in batches:
        B, T = x.shape
        m = model.mask[:T, :T]
        h = model.embed(x)
        final = torch.zeros(B, T, V)
        loops = torch.zeros(B, T, dtype=torch.long)
        active = torch.ones(B, T, dtype=torch.bool)
        for i in range(1, k_max + 1):
            a3 = active.unsqueeze(-1)
            h = torch.where(a3, model.block(h, m), h)
            loops = loops + active.long()
            logits = model.readout(h)
            final = torch.where(a3, logits, final)
            if i == k_max:
                break
            if tau is not None:
                logp = F.log_softmax(logits, dim=-1)
                H = -(logp.exp() * logp).sum(-1) / LOG2      # entropy in bits
                stop = active & (H <= tau)
            else:
                r = torch.rand(B, T, generator=gen)
                stop = active & (r < p_random)
            active = active & ~stop
            if not bool(active.any()):
                break
        tot += float(F.cross_entropy(final.reshape(-1, V), y.reshape(-1),
                                     reduction="sum"))
        ntok += y.numel()
        loops_sum += float(loops.sum())
        for i in range(1, k_max + 1):
            exit_hist[i] += int((loops == i).sum())
    model.train()
    return (tot / ntok) / LOG2, loops_sum / ntok, [c / ntok for c in exit_hist[1:]]


def p_for_mean_loops(target, k_max):
    """Solve 1 + (1-p) + ... + (1-p)^(k_max-1) = target for p in [0,1]."""
    if target <= 1.0:
        return 1.0
    if target >= k_max:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        p = 0.5 * (lo + hi)
        m = sum((1.0 - p) ** j for j in range(k_max))
        if m > target:
            lo = p
        else:
            hi = p
    return 0.5 * (lo + hi)


def interp_at(xs, ys, x):
    """Piecewise-linear interpolation of the fixed-k frontier at mean-loops x."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i])
    return ys[-1]


# ----------------------------- training -------------------------------------
def train_one(cond, seed, P, train, val_batches, vocab, log):
    import numpy as np
    set_seeds(seed)
    rng = np.random.default_rng(seed * 7919 + 13)
    T, B, k = P["block_size"], P["batch_size"], cond["k"]
    deep = cond["supervision"] == "deep"

    model = LoopedCharLM(vocab, P["d_model"], P["n_heads"], P["d_ff"],
                         P["k_max"], T)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=P["lr"],
                            betas=tuple(P["betas"]), weight_decay=P["weight_decay"])

    bf, hf = flops_per_token(P["d_model"], P["d_ff"], T, vocab)
    n_heads_fwd = k if deep else 1
    train_flops_per_step = 3.0 * (k * bf + n_heads_fwd * hf) * B * T

    curve, t0, capped = [], time.time(), False
    loss = torch.tensor(float("nan"))
    step = -1
    for step in range(P["steps"]):
        frac = (step + 1) / P["warmup"]
        if frac < 1.0:
            lr = P["lr"] * frac
        else:
            prog = (step + 1 - P["warmup"]) / max(1, P["steps"] - P["warmup"])
            lr = P["lr"] * (P["min_lr_frac"] + (1 - P["min_lr_frac"])
                            * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch(train, rng, B, T)
        outs = model.forward_all_loops(x, k)
        yf = y.reshape(-1)
        if deep:
            loss = sum(F.cross_entropy(o.reshape(-1, vocab), yf) for o in outs) / len(outs)
        else:
            loss = F.cross_entropy(outs[-1].reshape(-1, vocab), yf)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), P["grad_clip"])
        opt.step()
        if (step + 1) % P["eval_every"] == 0 or step == 0:
            bpc, _ = eval_fixed(model, val_batches, k)
            curve.append({"step": step + 1, "val_bpc": round(bpc, 4)})
        if time.time() - t0 > P["time_cap_s_per_run"]:
            capped = True
            break
    if not curve or curve[-1]["step"] != step + 1:
        bpc, _ = eval_fixed(model, val_batches, k)
        curve.append({"step": step + 1, "val_bpc": round(bpc, 4)})
    secs = time.time() - t0

    log(f"  {cond['name']:11s} seed{seed}: params={n_params} steps={step+1} "
        f"{secs:5.0f}s{' CAPPED' if capped else ''} val_bpc(k={k})={curve[-1]['val_bpc']:.4f}")
    rec = {"cond": cond["name"], "k": k, "supervision": cond["supervision"],
           "seed": seed, "n_params": n_params, "steps_run": step + 1,
           "time_capped": capped, "train_seconds": round(secs, 1),
           "total_train_flops": (step + 1) * train_flops_per_step,
           "final_train_loss_nats": round(float(loss.detach()), 4),
           "val_bpc_at_train_k": curve[-1]["val_bpc"], "curve": curve}
    return model, rec


# ----------------------------- chart ---------------------------------------
def make_chart(P, agg, out):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kmax = P["k_max"]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.0, 4.5),
                                        width_ratios=[2.2, 1.9, 1.6])

    # ---- panel 1: the frontier -------------------------------------------
    fx = agg["deep_fixed"]["loops"]
    fy = agg["deep_fixed"]["bpc"]
    fe = agg["deep_fixed"]["bpc_std"]
    ax1.errorbar(fx, fy, yerr=fe, fmt="o-", color="#1a4e8a", lw=2.4, ms=8, capsize=4,
                 zorder=4, label="fixed k (deep-sup model)")
    for xx, yy, kk in zip(fx, fy, range(1, kmax + 1)):
        ax1.annotate(f"k={kk}", (xx, yy), textcoords="offset points",
                     xytext=(4, 8), fontsize=8, color="#1a4e8a")

    ex = agg["entropy"]["loops"]
    ey = agg["entropy"]["bpc"]
    ax1.plot(ex, ey, "s--", color="#c95d3c", lw=2.2, ms=5.5, zorder=5,
             label="entropy exit (tau sweep)")

    rx = agg["random"]["loops"]
    ry = agg["random"]["bpc"]
    ax1.plot(rx, ry, "^:", color="#7a7a7a", lw=1.8, ms=5, zorder=3,
             label="random exit (matched compute)")

    sx = agg["sep_fixed"]["loops"]
    sy = agg["sep_fixed"]["bpc"]
    ax1.plot(sx, sy, "D-", color="#1a7f64", lw=1.8, ms=7, alpha=0.9, zorder=4,
             label="separately trained fixed-k")

    ax1.scatter([sx[0]], [sy[0]], s=210, facecolors="none", edgecolors="#1a7f64",
                lw=2.0, zorder=6)
    ax1.annotate("best point overall:\ndedicated k=1", (sx[0], sy[0]),
                 textcoords="offset points", xytext=(14, 6), fontsize=8,
                 color="#1a7f64")
    ax1.set_xlabel("mean loops per token  (compute proxy)")
    ax1.set_ylabel("val bits/char")
    ax1.set_title("Quality-compute frontier: adaptive vs fixed depth", fontsize=10.5)
    ax1.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(alpha=0.15)

    # ---- panel 2: where do tokens exit? ----------------------------------
    taus = agg["entropy"]["tau"]
    hist = np.array(agg["entropy"]["exit_hist"])       # (n_tau, kmax)
    colors = ["#f2c14e", "#e08a3c", "#a8563c", "#3d5a80"]
    bottom = np.zeros(len(taus))
    xs = np.arange(len(taus))
    for i in range(kmax):
        ax2.bar(xs, hist[:, i], bottom=bottom, color=colors[i], width=0.82,
                label=f"exit at loop {i+1}")
        bottom = bottom + hist[:, i]
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"{t:g}" for t in taus], fontsize=7, rotation=90)
    ax2.set_xlabel("entropy threshold tau (bits)")
    ax2.set_ylabel("fraction of val tokens")
    ax2.set_title("Exit-loop distribution: loops 2-3 are never used", fontsize=10.5)
    ax2.legend(frameon=False, fontsize=7.5, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.44), columnspacing=1.0, handlelength=1.2)
    ax2.set_ylim(0, 1.0)
    ax2.spines[["top", "right"]].set_visible(False)

    # ---- panel 3: adaptive minus fixed at matched compute -----------------
    d_ent = agg["entropy"]["delta_vs_fixed"]
    d_rnd = agg["random"]["delta_vs_fixed"]
    w = 0.4
    xs = np.arange(len(ex))
    ax3.bar(xs - w / 2, d_ent, width=w, color="#c95d3c", label="entropy exit")
    ax3.bar(xs + w / 2, d_rnd, width=w, color="#7a7a7a", label="random exit")
    ax3.axhline(0, color="0.2", lw=1)
    ax3.set_xticks(xs)
    ax3.set_xticklabels([f"{l:.2f}" for l in ex], fontsize=7, rotation=90)
    ax3.set_xlabel("mean loops/token at that operating point")
    ax3.set_ylabel("bpc minus fixed-k frontier")
    ax3.set_title("below 0 = adaptive WINS at matched compute", fontsize=10.5)
    ax3.legend(frameon=False, fontsize=8)
    ax3.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Shadow E3 - per-token entropy halting vs fixed depth on a tied looped char LM "
        f"(d={P['d_model']}, k_max={kmax}, ~{agg['n_params']/1e3:.0f}k params, "
        f"tiny-shakespeare, {P['steps']} steps, {len(P['seeds'])} seeds)",
        fontsize=11, y=1.04)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")


# ----------------------------- main ----------------------------------------
def main():
    smoke = "--smoke" in sys.argv
    cfg = load_config()
    P = cfg["params"]
    if "--chart-only" in sys.argv:
        # redraw chart.png from an existing results.json (no retraining)
        m = json.load(open(HERE / "results.json"))["metrics"]
        make_chart(P, {"deep_fixed": m["deepsup_fixed_k"],
                       "entropy": m["entropy_exit"],
                       "random": m["random_exit_control"],
                       "sep_fixed": m["separately_trained_fixed_k"],
                       "n_params": m["n_params"]}, HERE / "chart.png")
        print("chart.png rebuilt from results.json")
        return
    if smoke:
        P["steps"], P["eval_every"], P["eval_batches"] = 30, 30, 4
        P["seeds"] = [0]
        P["entropy_taus"] = [0.0, 1.0, 6.1]
    set_seeds(int(cfg.get("seed", 0)))
    t0 = time.time()
    log = lambda s: print(s, flush=True)

    import numpy as np
    train, val, vocab, n_chars = load_data(P["train_frac"])
    T, B, kmax = P["block_size"], P["batch_size"], P["k_max"]
    val_batches = fixed_val_batches(val, B, T, P["eval_batches"])
    log(f"tiny-shakespeare: {n_chars} chars, vocab={vocab}, train={len(train)} "
        f"val={len(val)} | val tokens per eval = {P['eval_batches']*B*T}")

    bf, hf = flops_per_token(P["d_model"], P["d_ff"], T, vocab)
    log(f"flops/token: block={bf:.0f} head={hf:.0f} (head = {hf/bf*100:.1f}% of a loop)")

    # ---------------- train ------------------------------------------------
    runs, models = [], {}
    for cond in P["conditions"]:
        for seed in P["seeds"]:
            model, rec = train_one(cond, seed, P, train, val_batches, vocab, log)
            runs.append(rec)
            models[(cond["name"], seed)] = model

    # ---------------- evaluate policies on the deep-sup model --------------
    gen = torch.Generator().manual_seed(P["random_exit_seed"])
    per_seed = {}
    for seed in P["seeds"]:
        m = models[("deepsup_k4", seed)]
        d = {"fixed": [], "entropy": [], "random": [], "naive": []}
        for k in range(1, kmax + 1):
            bpc, loops = eval_fixed(m, val_batches, k)
            d["fixed"].append({"k": k, "bpc": bpc, "loops": loops})
        for tau in P["entropy_taus"]:
            bpc, loops, hist = eval_exit(m, val_batches, kmax, tau=tau)
            d["entropy"].append({"tau": tau, "bpc": bpc, "loops": loops,
                                 "exit_hist": hist})
        for e in d["entropy"]:
            p = p_for_mean_loops(e["loops"], kmax)
            bpc, loops, hist = eval_exit(m, val_batches, kmax, p_random=p, gen=gen)
            d["random"].append({"p": p, "target_loops": e["loops"], "bpc": bpc,
                                "loops": loops, "exit_hist": hist})
        # ablation: the backlog's fallback - entropy exit on a model whose
        # intermediate readouts were never supervised
        mn = models[("fixed_k4", seed)]
        for tau in P["entropy_taus"]:
            bpc, loops, hist = eval_exit(mn, val_batches, kmax, tau=tau)
            d["naive"].append({"tau": tau, "bpc": bpc, "loops": loops})
        per_seed[seed] = d
        log(f"  deepsup seed{seed} fixed-k bpc: "
            + ", ".join(f"k{e['k']}={e['bpc']:.4f}" for e in d["fixed"]))

    # separately trained fixed-k models (their own trained depth)
    sep = {}
    for cond in P["conditions"]:
        if cond["supervision"] != "final":
            continue
        vals = [r["val_bpc_at_train_k"] for r in runs if r["cond"] == cond["name"]]
        sep[cond["k"]] = (float(np.mean(vals)), vals)

    # ---------------- aggregate over seeds ---------------------------------
    S = P["seeds"]

    def mean_over_seeds(policy, field, idx):
        return float(np.mean([per_seed[s][policy][idx][field] for s in S]))

    def std_over_seeds(policy, field, idx):
        return float(np.std([per_seed[s][policy][idx][field] for s in S]))

    deep_fixed = {
        "k": list(range(1, kmax + 1)),
        "loops": [float(k) for k in range(1, kmax + 1)],
        "bpc": [round(mean_over_seeds("fixed", "bpc", i), 4) for i in range(kmax)],
        "bpc_std": [round(std_over_seeds("fixed", "bpc", i), 4) for i in range(kmax)],
        "bpc_per_seed": [[round(per_seed[s]["fixed"][i]["bpc"], 4) for s in S]
                         for i in range(kmax)],
    }
    fx, fy = deep_fixed["loops"], deep_fixed["bpc"]

    n_tau = len(P["entropy_taus"])
    ent = {"tau": P["entropy_taus"],
           "bpc": [round(mean_over_seeds("entropy", "bpc", i), 4) for i in range(n_tau)],
           "bpc_std": [round(std_over_seeds("entropy", "bpc", i), 4) for i in range(n_tau)],
           "loops": [round(mean_over_seeds("entropy", "loops", i), 4) for i in range(n_tau)],
           "exit_hist": [[round(float(np.mean([per_seed[s]["entropy"][i]["exit_hist"][j]
                                               for s in S])), 4) for j in range(kmax)]
                         for i in range(n_tau)]}
    ent["delta_vs_fixed"] = [round(ent["bpc"][i] - interp_at(fx, fy, ent["loops"][i]), 4)
                             for i in range(n_tau)]
    # FLOP-honest x-axis: the entropy policy pays for a head eval every loop it runs
    ent["equiv_loops_flops"] = [round(l * (bf + hf) / bf, 4) for l in ent["loops"]]
    ent["delta_vs_fixed_flopadj"] = [
        round(ent["bpc"][i] - interp_at(fx, fy, ent["equiv_loops_flops"][i]), 4)
        for i in range(n_tau)]

    rnd = {"p": [round(mean_over_seeds("random", "p", i), 4) for i in range(n_tau)],
           "bpc": [round(mean_over_seeds("random", "bpc", i), 4) for i in range(n_tau)],
           "loops": [round(mean_over_seeds("random", "loops", i), 4) for i in range(n_tau)]}
    rnd["delta_vs_fixed"] = [round(rnd["bpc"][i] - interp_at(fx, fy, rnd["loops"][i]), 4)
                             for i in range(n_tau)]

    naive = {"tau": P["entropy_taus"],
             "bpc": [round(mean_over_seeds("naive", "bpc", i), 4) for i in range(n_tau)],
             "loops": [round(mean_over_seeds("naive", "loops", i), 4) for i in range(n_tau)]}

    sep_ks = sorted(sep)
    sep_fixed = {"k": sep_ks, "loops": [float(k) for k in sep_ks],
                 "bpc": [round(sep[k][0], 4) for k in sep_ks],
                 "bpc_per_seed": [[round(v, 4) for v in sep[k][1]] for k in sep_ks]}
    ent["delta_vs_separately_trained_fixed"] = [
        round(ent["bpc"][i] - interp_at(sep_fixed["loops"], sep_fixed["bpc"],
                                        ent["loops"][i]), 4) for i in range(n_tau)]

    # ---------------- verdict ----------------------------------------------
    # interior operating points only (tau=0 and tau>log2(V) are the fixed
    # endpoints by construction and must sit on the frontier)
    interior = [i for i in range(n_tau) if 1.02 < ent["loops"][i] < kmax - 0.02]
    best_i = min(interior, key=lambda i: ent["delta_vs_fixed"][i]) if interior else 0
    best_delta = ent["delta_vs_fixed"][best_i]
    ent_beats_random = [round(ent["bpc"][i] - rnd["bpc"][i], 4) for i in range(n_tau)]
    mean_ent_vs_rnd = float(np.mean([ent_beats_random[i] for i in interior])) if interior else 0.0

    loops_gain_1_to_4 = round(deep_fixed["bpc"][0] - deep_fixed["bpc"][-1], 4)
    endpoint_gap = round(deep_fixed["bpc"][0] - sep[1][0], 4)  # deep-sup k1 vs its own trained k1

    adaptive_wins = best_delta < -1e-4
    verdict = (
        f"hypothesis {'SUPPORTED' if adaptive_wins else 'REFUTED'}: at matched mean "
        f"loops/token the best entropy operating point is {best_delta:+.4f} bits/char "
        f"vs the fixed-k frontier (negative = adaptive wins)")

    metrics = {
        "n_params": runs[0]["n_params"],
        "flops_per_token": {"block": bf, "lm_head": hf,
                            "head_frac_of_block": round(hf / bf, 4)},
        "deepsup_fixed_k": deep_fixed,
        "separately_trained_fixed_k": sep_fixed,
        "entropy_exit": ent,
        "random_exit_control": rnd,
        "naive_exit_on_fixed_k4": naive,
        "entropy_minus_random_bpc": ent_beats_random,
        "mean_entropy_minus_random_interior": round(mean_ent_vs_rnd, 4),
        "best_interior_tau": P["entropy_taus"][best_i],
        "best_interior_mean_loops": ent["loops"][best_i],
        "best_interior_delta_vs_fixed_bpc": best_delta,
        "best_interior_delta_flopadj_bpc": ent["delta_vs_fixed_flopadj"][best_i],
        "deepsup_bpc_gain_k1_to_k4": loops_gain_1_to_4,
        "deepsup_k1_minus_dedicated_k1_bpc": endpoint_gap,
        "deepsup_k4_minus_dedicated_k4_bpc": round(deep_fixed["bpc"][-1] - sep[4][0], 4),
        "per_run": runs,
        "per_seed_raw": {str(s): per_seed[s] for s in S},
        "verdict": verdict,
        "headline": (
            f"deep-sup model fixed-k bpc: " +
            ", ".join(f"k{k}={deep_fixed['bpc'][k-1]:.3f}" for k in deep_fixed["k"]) +
            f" | best entropy point tau={P['entropy_taus'][best_i]:g} at "
            f"{ent['loops'][best_i]:.2f} loops/token: {ent['bpc'][best_i]:.3f} bpc "
            f"({best_delta:+.4f} vs fixed frontier)"),
    }

    agg = {"deep_fixed": deep_fixed, "entropy": ent, "random": rnd,
           "sep_fixed": sep_fixed, "n_params": runs[0]["n_params"]}
    make_chart(P, agg, HERE / ("chart_smoke.png" if smoke else "chart.png"))

    results = {
        "id": cfg.get("id", "unknown"),
        "git_commit": git_sha(),
        "seed": int(cfg.get("seed", 0)),
        "duration_sec": round(time.time() - t0, 2),
        "metrics": metrics,
        "env": env_info(),
        "status": "done",
    }
    if not smoke:
        with open(HERE / "results.json", "w") as f:
            json.dump(results, f, indent=2)
    print(json.dumps({kk: results[kk] for kk in ("id", "duration_sec", "status")}, indent=2))
    print("headline:", metrics["headline"])
    print("entropy delta vs fixed frontier:", ent["delta_vs_fixed"])
    print("random  delta vs fixed frontier:", rnd["delta_vs_fixed"])
    print("entropy - random (neg = entropy informative):", ent_beats_random)
    print("exit hist at best tau:", ent["exit_hist"][best_i])
    print(verdict)


if __name__ == "__main__":
    main()
