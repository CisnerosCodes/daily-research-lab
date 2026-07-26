"""Weight tying vs embedding fraction, on a nanoGPT-style char/BPE LM over tiny-shakespeare.

Question: is the benefit of tying the input embedding to the output unembedding a
function of the *embedding fraction* (how much of the model's parameter budget lives in
the two vocab-facing matrices)?

We move the embedding fraction two ways:
  (a) vocab size at fixed d_model=64:  char (65) -> BPE 512 -> BPE 2000
  (b) d_model at fixed char vocab:     32 -> 64 -> 128
and at every point train a tied and an untied model.

Matching policy (PRIMARY): matched-architecture. Tied and untied share d_model, n_layer,
n_head, d_ff, block_size and vocab; the ONLY difference is whether the output projection
reuses the token-embedding matrix. The untied model therefore has V*d MORE parameters --
that is the point: tying is being asked "is dropping those params free or better?".
A secondary MATCHED-TOTAL-PARAMS control is run at the two extremes of the embedding-
fraction axis: there the untied model's d_model is shrunk until its total parameter count
matches the tied model's.

Headline metric: delta val bits-per-CHARACTER (tied - untied) as a function of embedding
fraction. bpc (not bits-per-token) so numbers are comparable across vocabularies.

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


# ------------------------------------------------------------------------ tokenizer
def fit_bpe(train_ids, base_vocab, n_merges, snapshots):
    """Greedy BPE fitted on the TRAIN stream only.

    Returns (merges, token_char_lengths, {V: encoded_train_stream}).
    Deterministic: ties in pair frequency are broken by np.argmax (lowest key wins).
    """
    a = train_ids.astype(np.int32).copy()
    V = base_vocab
    merges = []
    tok_len = [1] * base_vocab
    out = {}
    want = set(snapshots)
    if base_vocab in want:
        out[base_vocab] = a.copy()
    for _ in range(n_merges):
        keys = a[:-1].astype(np.int64) * V + a[1:]
        cnt = np.bincount(keys, minlength=V * V)
        best = int(cnt.argmax())
        if cnt[best] < 2:
            break
        x, y = best // V, best % V
        idx = np.flatnonzero((a[:-1] == x) & (a[1:] == y))
        if x == y:  # forbid overlapping merges of a repeated symbol
            keep = np.ones(len(idx), dtype=bool)
            prev = -10
            for j, i in enumerate(idx):
                if i == prev + 1:
                    keep[j] = False
                else:
                    prev = i
            idx = idx[keep]
        nxt = a.copy()
        nxt[idx] = V
        dele = np.zeros(len(a), dtype=bool)
        dele[idx + 1] = True
        a = nxt[~dele]
        merges.append((int(x), int(y)))
        tok_len.append(tok_len[x] + tok_len[y])
        V += 1
        if V in want:
            out[V] = a.copy()
    return merges, np.array(tok_len, dtype=np.int64), out


def apply_merges(ids, merges, base_vocab, upto):
    a = ids.astype(np.int32).copy()
    for k in range(upto):
        x, y = merges[k]
        idx = np.flatnonzero((a[:-1] == x) & (a[1:] == y))
        if len(idx) == 0:
            continue
        if x == y:
            keep = np.ones(len(idx), dtype=bool)
            prev = -10
            for j, i in enumerate(idx):
                if i == prev + 1:
                    keep[j] = False
                else:
                    prev = i
            idx = idx[keep]
        nxt = a.copy()
        nxt[idx] = base_vocab + k
        dele = np.zeros(len(a), dtype=bool)
        dele[idx + 1] = True
        a = nxt[~dele]
    return a


def build_tokenizers(cfg):
    """Returns ({V: dict(train, val, tok_len, ...)}, base_vocab, n_train_chars, n_val_chars)."""
    p = cfg["params"]
    txt_path = HERE / "data" / "tinyshakespeare.txt"
    if not txt_path.exists():   # data/ is gitignored; fetch on a fresh clone
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(cfg["dataset"]["source"], txt_path)
    text = txt_path.read_text()
    chars = sorted(set(text))
    base_vocab = len(chars)
    assert base_vocab == p["char_vocab_size"], f"expected {p['char_vocab_size']} chars, got {base_vocab}"
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in text], dtype=np.int32)
    n = int(p["train_frac"] * len(ids))
    tr_c, va_c = ids[:n], ids[n:]

    targets = sorted({int(c["vocab"]) for c in p["configs"]})
    bpe_targets = [v for v in targets if v > base_vocab]
    max_v = max(bpe_targets) if bpe_targets else base_vocab

    cache = HERE / "data" / f"bpe_cache_{max_v}_{p['train_frac']}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        merges = [tuple(int(u) for u in t) for t in z["merges"]]
        tok_len_all = z["tok_len"]
        snaps = {int(k): z[f"tr_{int(k)}"] for k in z["snap_keys"]}
    else:
        merges, tok_len_all, snaps = fit_bpe(tr_c, base_vocab, max_v - base_vocab, targets)
        np.savez_compressed(
            cache, merges=np.array(merges, dtype=np.int64), tok_len=tok_len_all,
            snap_keys=np.array(sorted(snaps.keys())),
            **{f"tr_{k}": v for k, v in snaps.items()},
        )

    data = {}
    for V in targets:
        upto = V - base_vocab
        tr = snaps[V] if V in snaps else apply_merges(tr_c, merges, base_vocab, upto)
        va = va_c.copy() if upto == 0 else apply_merges(va_c, merges, base_vocab, upto)
        data[V] = {"train": tr.astype(np.int64), "val": va.astype(np.int64),
                   "tok_len": tok_len_all[:V].astype(np.int64),
                   "compression": float(len(tr_c) / len(tr))}
    return data, base_vocab, int(len(tr_c)), int(len(va_c))


# ---------------------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, d, n_head, d_ff):
        super().__init__()
        self.n_head = n_head
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.fc = nn.Linear(d, d_ff, bias=False)
        self.out = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(D, dim=2)
        q = q.view(B, T, self.n_head, D // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, D // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, D // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, D))
        x = x + self.out(F.gelu(self.fc(self.ln2(x))))
        return x


class GPT(nn.Module):
    def __init__(self, vocab, d, n_layer, n_head, d_ff, block_size, tied):
        super().__init__()
        self.vocab, self.block_size, self.tied = vocab, block_size, tied
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(block_size, d)
        self.blocks = nn.ModuleList([Block(d, n_head, d_ff) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.apply(self._init)
        for name, prm in self.named_parameters():   # nanoGPT residual-projection scaling
            if name.endswith("proj.weight") or name.endswith("out.weight"):
                nn.init.normal_(prm, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))
        if tied:
            self.head.weight = self.tok.weight

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x))


def n_params(m):
    return sum(q.numel() for q in m.parameters())   # shared tensors counted once


def make_model(vocab, d, p, tied):
    return GPT(vocab, d, p["n_layer"], p["n_head"], p["d_ff_mult"] * d, p["block_size"], tied)


def param_count(vocab, d, p, tied):
    return n_params(make_model(vocab, d, p, tied))


# ------------------------------------------------------------------------ train/eval
def get_batch(data, rng, batch_size, block_size):
    ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i:i + block_size] for i in ix])
    y = np.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return torch.from_numpy(x), torch.from_numpy(y)


@torch.no_grad()
def eval_bpc(model, val_ids, tok_len, block_size, eval_batch):
    """Sum token NLL (nats) over all predicted tokens / (ln2 * chars those tokens cover)."""
    model.eval()
    n_blocks = (len(val_ids) - 1) // block_size
    xs = np.stack([val_ids[i * block_size:(i + 1) * block_size] for i in range(n_blocks)])
    ys = np.stack([val_ids[i * block_size + 1:(i + 1) * block_size + 1] for i in range(n_blocks)])
    tot_nats, tot_chars, tot_tokens = 0.0, 0, 0
    for s in range(0, n_blocks, eval_batch):
        xb = torch.from_numpy(xs[s:s + eval_batch])
        yb = torch.from_numpy(ys[s:s + eval_batch])
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, model.vocab), yb.reshape(-1), reduction="sum")
        tot_nats += float(loss)
        tot_chars += int(tok_len[yb.numpy()].sum())
        tot_tokens += int(yb.numel())
    model.train()
    return {"bpc": tot_nats / (LN2 * tot_chars), "bpt": tot_nats / (LN2 * tot_tokens),
            "eval_tokens": tot_tokens, "eval_chars": tot_chars}


def train_one(vocab, d, tied, seed, p, dat, tag):
    set_seeds(seed)
    model = make_model(vocab, d, p, tied)
    decay = [q for q in model.parameters() if q.dim() >= 2]
    nodecay = [q for q in model.parameters() if q.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": p["weight_decay"]},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=p["lr"], betas=(0.9, 0.95),
    )
    rng = np.random.default_rng(seed)   # tied and untied see the identical batch stream
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
        x, y = get_batch(dat["train"], rng, p["batch_size"], p["block_size"])
        loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), p["grad_clip"])
        opt.step()
        losses.append(float(loss.detach()))
    train_s = time.time() - t0
    ev = eval_bpc(model, dat["val"], dat["tok_len"], p["block_size"], p["eval_batch"])
    rec = {"tag": tag, "vocab": vocab, "d_model": d, "tied": bool(tied), "seed": int(seed),
           "n_params": n_params(model), "n_params_vocab_matrices": vocab * d * (1 if tied else 2),
           "train_seconds": round(train_s, 1),
           "final_train_loss_ma50": round(float(np.mean(losses[-50:])), 4),
           "val_bpc": round(ev["bpc"], 5), "val_bits_per_token": round(ev["bpt"], 5),
           "eval_chars": ev["eval_chars"]}
    print(f"  [{tag:15s}] V={vocab:5d} d={d:3d} tied={str(tied):5s} seed={seed} "
          f"P={rec['n_params']:6d} bpc={rec['val_bpc']:.4f} ({rec['train_seconds']}s)", flush=True)
    return rec


def bpc_by_seed(rs, tied):
    return [r["val_bpc"] for r in sorted([x for x in rs if x["tied"] == tied], key=lambda x: x["seed"])]


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    return 0.0 if den == 0 else float((ra * rb).sum() / den)


# ----------------------------------------------------------------------------- chart
def make_chart(per_config, control, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    xs = [c["emb_frac_untied"] for c in per_config]
    ds = [c["delta_bpc_mean"] for c in per_config]
    es = [c["delta_bpc_std"] for c in per_config]

    ax = axes[0]
    ax.axhline(0, color="k", lw=1.2, ls="--", zorder=1)
    ax.errorbar(xs, ds, yerr=es, marker="o", ms=8, lw=2, capsize=4, color="#c0392b", zorder=3,
                label="mean over seeds")
    for c in per_config:
        for dd in c["delta_bpc_per_seed"]:
            ax.plot(c["emb_frac_untied"], dd, marker="_", ms=14, color="#7f8c8d", zorder=2)
        ax.annotate(c["name"], (c["emb_frac_untied"], c["delta_bpc_mean"]),
                    textcoords="offset points", xytext=(0, 13), ha="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("embedding fraction   2·V·d / P(untied)")
    ax.set_ylabel("Δ val bpc   (tied − untied)")
    ax.set_title("Headline: benefit of tying vs embedding fraction\n(below the dashed line = tying wins)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    lo, hi = min(ds + [0.0]), max(ds + [0.0])
    span = max(hi - lo, 1e-3)
    ax.set_ylim(lo - 0.12 * span, hi + 0.28 * span)

    ax = axes[1]
    idx = np.arange(len(per_config))
    w = 0.38
    ax.bar(idx - w / 2, [c["bpc_tied_mean"] for c in per_config], w, label="tied", color="#2980b9")
    ax.bar(idx + w / 2, [c["bpc_untied_mean"] for c in per_config], w, label="untied", color="#e67e22")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{c['name']}\nef={c['emb_frac_untied']:.2f}" for c in per_config], fontsize=8)
    ax.set_ylabel("val bits per character")
    ax.set_title("Absolute val bpc (matched architecture)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    if control:
        idx = np.arange(len(control))
        ax.bar(idx - w / 2, [c["bpc_tied"] for c in control], w, label="tied (full d_model)",
               color="#2980b9")
        ax.bar(idx + w / 2, [c["bpc_untied_matched"] for c in control], w,
               label="untied (d_model shrunk to match P)", color="#8e44ad")
        for i, c in enumerate(control):
            ax.annotate(f"Δ={c['delta_bpc']:+.4f}", (i, max(c["bpc_tied"], c["bpc_untied_matched"])),
                        textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9)
        ax.set_xticks(idx)
        ax.set_xticklabels([f"{c['config']}\nd {c['d_model_tied']}→{c['d_model_untied_matched']}, "
                            f"ef={c['emb_frac_untied']:.2f}" for c in control], fontsize=8)
        ax.set_ylabel("val bits per character")
        ax.set_ylim(0, 1.35 * max(max(c["bpc_tied"], c["bpc_untied_matched"]) for c in control))
        ax.legend(fontsize=8, loc="upper right")
    else:
        ax.text(0.5, 0.5, "control runs skipped (time budget)", ha="center", va="center")
    ax.set_title("Control: matched TOTAL params\n(untied d_model shrunk to equal tied param count)")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Weight tying vs embedding fraction — 1-layer nanoGPT-style LM, tiny-shakespeare, CPU",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    cfg = load_config()
    p = cfg["params"]
    seed0 = int(cfg.get("seed", 0))
    set_seeds(seed0)
    t_start = time.time()

    print("building tokenizers ...", flush=True)
    t_tok = time.time()
    data, base_vocab, n_train_chars, n_val_chars = build_tokenizers(cfg)
    tok_s = time.time() - t_tok
    for V in sorted(data):
        print(f"  V={V:5d} train_tokens={len(data[V]['train']):7d} val_tokens={len(data[V]['val']):6d} "
              f"chars/token={data[V]['compression']:.3f}", flush=True)

    # ---- embedding fractions (defined on the untied reference architecture) ----
    grid = []
    for c in p["configs"]:
        V, d = int(c["vocab"]), int(c["d_model"])
        P_un = param_count(V, d, p, tied=False)
        P_ti = param_count(V, d, p, tied=True)
        grid.append({"name": c["name"], "vocab": V, "d_model": d,
                     "n_params_untied": P_un, "n_params_tied": P_ti,
                     "emb_frac_untied": round(2 * V * d / P_un, 4),
                     "emb_frac_tied": round(V * d / P_ti, 4)})
    grid.sort(key=lambda g: g["emb_frac_untied"])
    print("\ngrid (sorted by embedding fraction):", flush=True)
    for g in grid:
        print(f"  {g['name']:11s} V={g['vocab']:5d} d={g['d_model']:3d} "
              f"P_untied={g['n_params_untied']:6d} P_tied={g['n_params_tied']:6d} "
              f"emb_frac={g['emb_frac_untied']:.3f}", flush=True)

    # ---- primary sweep: matched architecture, tied vs untied ----
    runs = []
    print("\n--- primary sweep (matched architecture) ---", flush=True)
    for g in grid:
        for seed in p["seeds"]:
            for tied in (True, False):
                runs.append(train_one(g["vocab"], g["d_model"], tied, seed, p, data[g["vocab"]], g["name"]))

    per_config = []
    for g in grid:
        rs = [r for r in runs if r["tag"] == g["name"]]
        ti, un = bpc_by_seed(rs, True), bpc_by_seed(rs, False)
        deltas = [t - u for t, u in zip(ti, un)]
        per_config.append({
            **g,
            "bpc_tied_mean": round(float(np.mean(ti)), 5),
            "bpc_untied_mean": round(float(np.mean(un)), 5),
            "bpc_tied_per_seed": ti, "bpc_untied_per_seed": un,
            "delta_bpc_per_seed": [round(x, 5) for x in deltas],
            "delta_bpc_mean": round(float(np.mean(deltas)), 5),
            "delta_bpc_std": round(float(np.std(deltas)), 5),
            "delta_sign_consistent": bool(len(set(np.sign(deltas))) == 1),
            "tying_wins": bool(np.mean(deltas) < 0),
        })

    # ---- secondary: matched-total-params control at the extremes ----
    control, controls_skipped = [], []
    print("\n--- control (matched total params) ---", flush=True)
    for name in p["control_configs"]:
        g = next(x for x in grid if x["name"] == name)
        if time.time() - t_start > p["time_budget_s"]:
            controls_skipped.append(name)
            print(f"  SKIPPED {name} (time budget)", flush=True)
            continue
        target = g["n_params_tied"]
        best_d, best_gap = None, None
        for d in range(p["n_head"], g["d_model"] + 1, p["n_head"]):
            gap = abs(param_count(g["vocab"], d, p, tied=False) - target)
            if best_gap is None or gap < best_gap:
                best_d, best_gap = d, gap
        r = train_one(g["vocab"], best_d, False, p["control_seed"], p, data[g["vocab"]], name + "_ctrl")
        tied_ref = next(x for x in runs if x["tag"] == name and x["tied"] and x["seed"] == p["control_seed"])
        control.append({
            "config": name, "vocab": g["vocab"], "d_model_tied": g["d_model"],
            "d_model_untied_matched": best_d,
            "n_params_tied": tied_ref["n_params"], "n_params_untied_matched": r["n_params"],
            "param_gap_pct": round(100 * (r["n_params"] - tied_ref["n_params"]) / tied_ref["n_params"], 2),
            "emb_frac_untied": g["emb_frac_untied"],
            "bpc_tied": tied_ref["val_bpc"], "bpc_untied_matched": r["val_bpc"],
            "delta_bpc": round(tied_ref["val_bpc"] - r["val_bpc"], 5),
        })
        runs.append(r)

    # ---- crossover on the primary curve ----
    xs = [c["emb_frac_untied"] for c in per_config]
    ds = [c["delta_bpc_mean"] for c in per_config]
    crossover = None
    for i in range(len(xs) - 1):
        if (ds[i] > 0) != (ds[i + 1] > 0) and ds[i] != ds[i + 1]:
            f = ds[i] / (ds[i] - ds[i + 1])
            crossover = round(float(xs[i] + f * (xs[i + 1] - xs[i])), 4)
            break
    if all(d < 0 for d in ds):
        pattern = "tying always helps at this scale (delta < 0 at every embedding fraction)"
    elif all(d > 0 for d in ds):
        pattern = "tying never helps at this scale (delta > 0 at every embedding fraction)"
    elif crossover is not None:
        pattern = f"crossover at embedding fraction ~{crossover}"
    else:
        pattern = "non-monotone / no single crossover"

    metrics = {
        "headline": "delta val bpc (tied - untied) vs embedding fraction",
        "matching_policy": p["matching_policy"],
        "grid": grid,
        "per_config": per_config,
        "delta_bpc_vs_emb_frac": [{"name": c["name"], "emb_frac": c["emb_frac_untied"],
                                   "delta_bpc": c["delta_bpc_mean"], "std": c["delta_bpc_std"]}
                                  for c in per_config],
        "crossover_emb_frac": crossover,
        "pattern": pattern,
        "spearman_delta_vs_embfrac": round(float(spearman(xs, ds)), 4),
        "matched_total_params_control": control,
        "controls_skipped_for_time": controls_skipped,
        "per_run": runs,
        "tokenizer": {str(V): {"train_tokens": int(len(data[V]["train"])),
                               "val_tokens": int(len(data[V]["val"])),
                               "chars_per_token": round(data[V]["compression"], 4)}
                      for V in sorted(data)},
        "n_train_chars": n_train_chars, "n_val_chars": n_val_chars,
        "tokenizer_build_seconds": round(tok_s, 1),
    }

    make_chart(per_config, control, HERE / "chart.png")

    results = {"id": cfg["id"], "git_commit": git_sha(), "seed": seed0,
               "duration_sec": round(time.time() - t_start, 2),
               "metrics": metrics, "env": env_info(), "status": "done"}
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== summary ===")
    print(f"pattern: {pattern}")
    for c in per_config:
        print(f"  {c['name']:11s} emb_frac={c['emb_frac_untied']:.3f}  "
              f"tied {c['bpc_tied_mean']:.4f}  untied {c['bpc_untied_mean']:.4f}  "
              f"delta {c['delta_bpc_mean']:+.4f} (sd {c['delta_bpc_std']:.4f}, "
              f"sign-consistent={c['delta_sign_consistent']})")
    for c in control:
        print(f"  CTRL {c['config']:11s} tied d={c['d_model_tied']} (P={c['n_params_tied']}) vs "
              f"untied d={c['d_model_untied_matched']} (P={c['n_params_untied_matched']}, "
              f"{c['param_gap_pct']:+.1f}%): delta {c['delta_bpc']:+.4f}")
    print(f"total wall clock: {results['duration_sec']}s")


if __name__ == "__main__":
    main()
