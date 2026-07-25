"""Shadow E1 - weight-tied looped block vs plain depth at ISO-FLOPs, tiny char LM.

Falsification target for the lab's Shadow model. A block applied k times (weight-tied,
1/k the block params) does exactly the same forward/backward FLOPs as a k-layer untied
stack. So for each k in {1,2,4} we train `tied` and `untied` for the SAME number of steps
at the SAME per-step FLOPs and compare validation bits/char.

If the loop does not beat depth here, Shadow's core mechanism dies cheaply.

Data: tiny-shakespeare, char level (substitute for TinyStories-1M; see README).
Deterministic, CPU-only, single-thread. Writes results.json + chart.png.

Usage:  python run.py
"""
import json, math, os, random, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "tinyshakespeare.txt"
DATA_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
            "master/data/tinyshakespeare/input.txt")


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
    for mod in ("numpy", "torch"):
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


class CharLM(nn.Module):
    """mode='tied'   -> 1 block applied k times  (k * F FLOPs, 1 block of params)
       mode='untied' -> k distinct blocks         (k * F FLOPs, k blocks of params)
    Both do the SAME number of block applications, hence iso-FLOP."""

    def __init__(self, vocab, d, h, dff, k, mode, block_size):
        super().__init__()
        assert mode in ("tied", "untied")
        self.k, self.mode = k, mode
        n_blocks = 1 if mode == "tied" else k
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(block_size, d)
        self.blocks = nn.ModuleList([Block(d, h, dff) for _ in range(n_blocks)])
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

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T))[None]
        m = self.mask[:T, :T]
        if self.mode == "tied":
            blk = self.blocks[0]
            for _ in range(self.k):
                x = blk(x, m)
        else:
            for blk in self.blocks:
                x = blk(x, m)
        return self.head(self.ln_f(x))


def fwd_flops_per_token(d, dff, h, T, k):
    """Analytic forward FLOPs per token (mult+add = 2 flops), causal attention.
    Identical for tied and untied at the same k - that is the iso-FLOP claim."""
    per_block = (
        2 * (3 * d * d)        # qkv
        + 2 * (d * d)          # out proj
        + 2 * (d * dff) * 2    # ffn up + down
        + 2 * 2 * d * (T / 2)  # QK^T and AV, causal -> avg T/2 keys
    )
    return k * per_block


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


@torch.no_grad()
def eval_bpc(model, val, B, T, n_batches, vocab):
    """Fixed val batches (same rng every call/run) -> bits per char."""
    import numpy as np
    model.eval()
    rng = np.random.default_rng(987654321)
    tot, ntok = 0.0, 0
    for _ in range(n_batches):
        x, y = get_batch(val, rng, B, T)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1), reduction="sum")
        tot += float(loss)
        ntok += y.numel()
    model.train()
    return (tot / ntok) / math.log(2.0)


# ----------------------------- one run -------------------------------------
def train_one(mode, k, seed, P, train, val, vocab, log):
    import numpy as np
    set_seeds(seed)
    rng = np.random.default_rng(seed * 7919 + 13)
    T, B = P["block_size"], P["batch_size"]

    model = CharLM(vocab, P["d_model"], P["n_heads"], P["d_ff"], k, mode, T)
    n_params = sum(p.numel() for p in model.parameters())
    n_block_params = sum(p.numel() for p in model.blocks.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=P["lr"],
                            betas=tuple(P["betas"]), weight_decay=P["weight_decay"])

    fpt = fwd_flops_per_token(P["d_model"], P["d_ff"], P["n_heads"], T, k)
    train_flops_per_step = 3.0 * fpt * B * T   # fwd + bwd ~= 3x fwd

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
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), P["grad_clip"])
        opt.step()
        if (step + 1) % P["eval_every"] == 0 or step == 0:
            bpc = eval_bpc(model, val, B, T, P["eval_batches"], vocab)
            curve.append({"step": step + 1, "val_bpc": round(bpc, 4),
                          "train_flops": (step + 1) * train_flops_per_step})
        if time.time() - t0 > P["time_cap_s_per_run"]:
            capped = True
            break
    if not curve or curve[-1]["step"] != step + 1:
        bpc = eval_bpc(model, val, B, T, P["eval_batches"], vocab)
        curve.append({"step": step + 1, "val_bpc": round(bpc, 4),
                      "train_flops": (step + 1) * train_flops_per_step})
    secs = time.time() - t0

    final_bpc = curve[-1]["val_bpc"]
    best_bpc = min(c["val_bpc"] for c in curve)
    log(f"  {mode:7s} k={k} seed{seed}: params={n_params:6d} (block {n_block_params:6d}) "
        f"steps={step+1} {secs:5.0f}s{' CAPPED' if capped else ''} "
        f"val_bpc={final_bpc:.4f} best={best_bpc:.4f}")
    return {"mode": mode, "k": k, "seed": seed, "n_params": n_params,
            "n_block_params": n_block_params, "steps_run": step + 1,
            "time_capped": capped, "train_seconds": round(secs, 1),
            "fwd_flops_per_token": fpt,
            "total_train_flops": (step + 1) * train_flops_per_step,
            "final_train_loss_nats": round(float(loss.detach()), 4),
            "val_bpc": final_bpc, "best_val_bpc": best_bpc, "curve": curve}


# ----------------------------- chart ---------------------------------------
def make_chart(P, runs, table, deltas):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kcol = {1: "#8a817c", 2: "#3d5a80", 4: "#c95d3c"}
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.5, 4.3),
                                        width_ratios=[2.1, 2.0, 1.3])

    # panel 1: learning curves (drop the step-1 point, it compresses the axis)
    for k in P["ks"]:
        for mode in P["modes"]:
            rs = [r for r in runs if r["mode"] == mode and r["k"] == k]
            pts = [c for c in rs[0]["curve"] if c["step"] > 1]
            steps = [c["step"] for c in pts]
            ys = np.mean([[c["val_bpc"] for c in r["curve"] if c["step"] > 1]
                          for r in rs], axis=0)
            ax1.plot(steps, ys, "-" if mode == "tied" else "--",
                     color=kcol[k], lw=2, marker="o" if mode == "tied" else "s", ms=3.5,
                     alpha=1.0 if mode == "tied" else 0.75,
                     label=f"{mode} k={k}")
    ax1.set_xlabel("training step (iso-FLOP within each k)")
    ax1.set_ylabel("val bits/char")
    ax1.set_title("Learning curves (mean of 2 seeds); solid = tied loop", fontsize=10)
    ax1.legend(frameon=False, fontsize=8, ncol=2, loc="lower left")
    ax1.spines[["top", "right"]].set_visible(False)

    # inset: zoom on the tail where the loop tax opens up
    axi = ax1.inset_axes([0.46, 0.42, 0.5, 0.5])
    for k in P["ks"]:
        for mode in P["modes"]:
            rs = [r for r in runs if r["mode"] == mode and r["k"] == k]
            pts = [c for c in rs[0]["curve"] if c["step"] >= P["steps"] * 0.6]
            steps = [c["step"] for c in pts]
            ys = np.mean([[c["val_bpc"] for c in r["curve"]
                           if c["step"] >= P["steps"] * 0.6] for r in rs], axis=0)
            axi.plot(steps, ys, "-" if mode == "tied" else "--", color=kcol[k],
                     lw=1.6, alpha=1.0 if mode == "tied" else 0.75)
    axi.tick_params(labelsize=7)
    axi.set_title("tail zoom", fontsize=7)
    axi.spines[["top", "right"]].set_visible(False)

    # panel 2: final bpc vs training FLOPs
    for mode, ls, mk in (("tied", "-", "o"), ("untied", "--", "s")):
        xs = [table[f"{mode}_k{k}"]["total_train_flops"] for k in P["ks"]]
        ys = [table[f"{mode}_k{k}"]["val_bpc_mean"] for k in P["ks"]]
        errs = [table[f"{mode}_k{k}"]["val_bpc_std"] for k in P["ks"]]
        ax2.errorbar(xs, ys, yerr=errs, fmt=mk + ls, lw=2, ms=7, capsize=4,
                     color="#1a7f64" if mode == "tied" else "#7b2d43",
                     label=f"{mode} block")
        for k, x, y in zip(P["ks"], xs, ys):
            ax2.annotate(f"k={k}, {table[f'{mode}_k{k}']['n_params']/1e3:.0f}k params",
                         (x, y), textcoords="offset points",
                         xytext=(-16, 10 if mode == "tied" else -18), fontsize=7)
    ax2.set_xscale("log")
    ax2.set_xlabel("total training FLOPs (matched between tied/untied at each k)")
    ax2.set_ylabel("final val bits/char")
    ax2.set_title("Iso-FLOP frontier: loop vs depth (err = seed spread)", fontsize=10)
    ax2.legend(frameon=False, fontsize=9, loc="center left")
    ax2.spines[["top", "right"]].set_visible(False)

    # panel 3: delta bars
    xs = [f"k={k}" for k in P["ks"]]
    ys = [deltas[f"k{k}"] for k in P["ks"]]
    ax3.bar(xs, ys, color=["#c95d3c" if y > 1e-9 else "#8a817c" for y in ys])
    ax3.axhline(0, color="0.3", lw=1)
    span = max(max(ys) - min(min(ys), 0), 1e-3)
    for i, y in enumerate(ys):
        ax3.text(i, y + 0.03 * span, f"{y:+.3f}", ha="center", fontsize=9)
    ax3.set_ylim(min(min(ys), 0) - 0.05 * span, max(ys) + 0.18 * span)
    ax3.set_ylabel("tied - untied  (val bpc)")
    ax3.set_title("above 0 = loop LOSES", fontsize=10)
    ax3.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Shadow E1 - weight-tied looped block vs untied depth at iso-FLOPs "
                 f"(char LM, d=64, tiny-shakespeare, {P['steps']} steps, 2 seeds)",
                 fontsize=11, y=1.03)
    fig.tight_layout()
    fig.savefig(HERE / "chart.png", dpi=160, bbox_inches="tight")


# ----------------------------- main ----------------------------------------
def main():
    cfg = load_config()
    P = cfg["params"]
    set_seeds(int(cfg.get("seed", 0)))
    t0 = time.time()
    log = lambda s: print(s, flush=True)

    train, val, vocab, n_chars = load_data(P["train_frac"])
    log(f"tiny-shakespeare: {n_chars} chars, vocab={vocab}, "
        f"train={len(train)} val={len(val)}")

    runs = []
    for k in P["ks"]:
        for mode in P["modes"]:
            for seed in P["seeds"]:
                runs.append(train_one(mode, k, seed, P, train, val, vocab, log))

    import numpy as np

    def agg(mode, k, key="val_bpc"):
        rs = [r for r in runs if r["mode"] == mode and r["k"] == k]
        return float(np.mean([r[key] for r in rs])), [r[key] for r in rs]

    table, deltas = {}, {}
    for k in P["ks"]:
        for mode in P["modes"]:
            m, vals = agg(mode, k)
            mb, _ = agg(mode, k, "best_val_bpc")
            rs = [r for r in runs if r["mode"] == mode and r["k"] == k]
            table[f"{mode}_k{k}"] = {
                "val_bpc_mean": round(m, 4),
                "val_bpc_per_seed": [round(v, 4) for v in vals],
                "val_bpc_std": round(float(np.std(vals)), 4),
                "best_val_bpc_mean": round(mb, 4),
                "n_params": rs[0]["n_params"],
                "n_block_params": rs[0]["n_block_params"],
                "fwd_flops_per_token": rs[0]["fwd_flops_per_token"],
                "total_train_flops": rs[0]["total_train_flops"],
            }
        tm, _ = agg("tied", k)
        um, _ = agg("untied", k)
        deltas[f"k{k}"] = round(tm - um, 4)   # >0 means the LOOP IS WORSE

    # k=1 is the same architecture in both modes -> code-path sanity check
    sanity = round(abs(table["tied_k1"]["val_bpc_mean"] - table["untied_k1"]["val_bpc_mean"]), 6)

    # does looping buy anything at all over its own iso-param shallow self?
    loop_gain = {f"k{k}": round(table["tied_k1"]["val_bpc_mean"] - table[f"tied_k{k}"]["val_bpc_mean"], 4)
                 for k in P["ks"]}
    depth_gain = {f"k{k}": round(table["untied_k1"]["val_bpc_mean"] - table[f"untied_k{k}"]["val_bpc_mean"], 4)
                  for k in P["ks"]}

    ks_gt1 = [k for k in P["ks"] if k > 1]
    loop_loses = all(deltas[f"k{k}"] > 0 for k in ks_gt1)
    verdict = ("hypothesis SUPPORTED: tied loop never beats untied depth at iso-FLOPs"
               if loop_loses else
               "hypothesis REFUTED at >=1 k: tied loop beats untied depth somewhere")

    metrics = {
        "per_run": runs,
        "table": table,
        "tied_minus_untied_bpc": deltas,
        "loop_gain_over_k1_bpc": loop_gain,
        "depth_gain_over_k1_bpc": depth_gain,
        "k1_sanity_gap_bpc": sanity,
        "verdict": verdict,
        "headline": ("val bits/char (mean of %d seeds): " % len(P["seeds"]) +
                     ", ".join(f"{m}-k{k}={table[f'{m}_k{k}']['val_bpc_mean']:.3f}"
                               for k in P["ks"] for m in P["modes"])),
    }

    make_chart(P, runs, table, deltas)

    results = {
        "id": cfg.get("id", "unknown"),
        "git_commit": git_sha(),
        "seed": int(cfg.get("seed", 0)),
        "duration_sec": round(time.time() - t0, 2),
        "metrics": metrics,
        "env": env_info(),
        "status": "done",
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({kk: results[kk] for kk in ("id", "duration_sec", "status")}, indent=2))
    print("headline:", metrics["headline"])
    print("deltas (tied - untied, >0 = loop loses):", metrics["tied_minus_untied_bpc"])
    print("k=1 sanity gap (should be 0.0):", sanity)
    print(metrics["verdict"])


if __name__ == "__main__":
    main()
