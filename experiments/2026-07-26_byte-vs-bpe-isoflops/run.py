"""Byte/char-level vs BPE at ISO-FLOPs on tiny-shakespeare.

Two arms of an otherwise identical nanoGPT-style LM (2 layers, d_model 80, ctx 128 TOKENS):
  (a) char  -- 65-symbol vocab (tiny-shakespeare is pure ASCII, so 1 char == 1 byte)
  (b) bpe   -- 512-symbol vocab from a greedy merge loop fitted on the TRAIN split only

THE CONTROL THAT MATTERS. A BPE token covers ~3 bytes, so at the same context length in
TOKENS the BPE arm sees ~3x more TEXT per forward pass. Comparing at iso-STEPS would be
meaningless. We instead equalise total training FLOPs:

    fwd_flops_per_token = 2 * [ n_layer*(4*d^2 + 2*d*d_ff) + d*V ]  +  n_layer * 2 * ctx * d
    train_flops         = 3 * fwd_flops_per_token * tokens_processed

The char arm's vocab-facing head is smaller (80x65 vs 80x512), so at equal FLOPs it gets
proportionally MORE optimisation steps. The full accounting is written to results.json.

Headline metric: validation BITS PER BYTE -- total NLL in bits over the val text divided by
the number of val BYTES actually predicted. Not bits-per-token: that would be incomparable.
Evaluation is STRIDED (50% overlap, only the second half of each window is scored) so that
neither arm is penalised by cold-start block boundaries.

Second metric: a character-composition (spelling) probe. For each of 144 words drawn from
val (72 rare + 72 common), the model must pick the true spelling out of {true word, 3
adjacent-interior-character TRANSPOSITIONS}. Distractors have the identical character
multiset and length, so the task is pure orthographic composition. Both arms score the
COMPLETE string context+candidate under their own tokenisation, so the comparison is exact
and free of the mid-word prompt-boundary problem. We also report incremental bits-per-byte
on the true word span, and split accuracy by how many BPE tokens the word costs.

Deterministic, CPU-only, single-threaded.  Usage:  python run.py
"""
import json, math, os, random, re, subprocess, sys, time
from collections import Counter
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
def fit_bpe(train_ids, base_vocab, n_merges):
    """Greedy BPE fitted on the TRAIN stream only (approach reused from
    experiments/2026-07-26_weight-tying-tiny/run.py).

    Returns (merges, token_byte_lengths, encoded_train_stream).
    Deterministic: pair-frequency ties broken by np.argmax (lowest key wins).
    """
    a = train_ids.astype(np.int32).copy()
    V = base_vocab
    merges = []
    tok_len = [1] * base_vocab
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
    return merges, np.array(tok_len, dtype=np.int64), a


def apply_merges(ids, merges, base_vocab):
    a = ids.astype(np.int32).copy()
    for k, (x, y) in enumerate(merges):
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


class Tok:
    """Minimal encoder shared by both arms (char arm has an empty merge list)."""

    def __init__(self, stoi, merges, base_vocab, tok_len, vocab):
        self.stoi, self.merges, self.base = stoi, merges, base_vocab
        self.tok_len, self.vocab = tok_len, vocab
        self._cache = {}

    def encode(self, s):
        hit = self._cache.get(s)
        if hit is None:
            ids = np.array([self.stoi[c] for c in s], dtype=np.int32)
            hit = (apply_merges(ids, self.merges, self.base) if self.merges else ids).astype(np.int64)
            self._cache[s] = hit
        return hit


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
    def __init__(self, vocab, p):
        super().__init__()
        d, nl = p["d_model"], p["n_layer"]
        self.vocab, self.block_size = vocab, p["block_size"]
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(p["block_size"], d)
        self.blocks = nn.ModuleList([Block(d, p["n_head"], p["d_ff"]) for _ in range(nl)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.apply(self._init)
        for name, prm in self.named_parameters():   # nanoGPT residual-projection scaling
            if name.endswith("proj.weight") or name.endswith("out.weight"):
                nn.init.normal_(prm, mean=0.0, std=0.02 / math.sqrt(2 * nl))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1]))
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x))


def n_params(m):
    return sum(q.numel() for q in m.parameters())


# ------------------------------------------------------------------- FLOP accounting
def flop_model(vocab, p):
    """Explicit, auditable per-token FLOP model. Embedding LOOKUP is free (a gather);
    the output head is a real matmul and is counted. Attention uses the causal average
    over positions (each query attends to ctx/2 keys on average, QK^T and AV each
    2 FLOPs per multiply-add pair)."""
    d, nl, ctx = p["d_model"], p["n_layer"], p["block_size"]
    mm = nl * (4 * d * d + 2 * d * p["d_ff"]) + d * vocab
    attn = nl * 2 * ctx * d
    fwd = 2 * mm + attn
    return {"matmul_params_per_token": int(mm), "attn_flops_per_token": int(attn),
            "fwd_flops_per_token": int(fwd), "train_flops_per_token": int(3 * fwd)}


# ------------------------------------------------------------------------ train/eval
def get_batch(data, rng, batch_size, block_size):
    ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i:i + block_size] for i in ix])
    y = np.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return torch.from_numpy(x), torch.from_numpy(y)


def strided_plan(n, ctx):
    """Windows tiling targets 1..N with >= ctx/2 tokens of context each, no overlap.

    Window at start s uses inputs ids[s:s+ctx] and targets ids[s+1:s+ctx+1].
    s=0 scores all ctx positions; every later window (stride ctx/2) scores only its
    last ctx/2 positions, which exactly continues the previous window's coverage.
    """
    half = ctx // 2
    plan = []
    s = 0
    while s + ctx + 1 <= n:
        plan.append((s, 0 if s == 0 else half))
        s += half
    return plan


@torch.no_grad()
def eval_bpb(model, ids, tok_len, ctx, eval_batch, max_windows=None):
    """Total bits over predicted val tokens / bytes those tokens cover."""
    model.eval()
    plan = strided_plan(len(ids), ctx)
    if max_windows is not None:
        plan = plan[:max_windows]
    tot_nats, tot_bytes, tot_tokens = 0.0, 0, 0
    for s0 in range(0, len(plan), eval_batch):
        chunk = plan[s0:s0 + eval_batch]
        xb = torch.from_numpy(np.stack([ids[s:s + ctx] for s, _ in chunk]))
        yb = np.stack([ids[s + 1:s + ctx + 1] for s, _ in chunk])
        logits = model(xb)
        lp = F.log_softmax(logits.float(), dim=-1)
        ybt = torch.from_numpy(yb)
        tokll = lp.gather(-1, ybt.unsqueeze(-1)).squeeze(-1)      # (B, ctx)
        mask = np.zeros(yb.shape, dtype=bool)
        for j, (_, off) in enumerate(chunk):
            mask[j, off:] = True
        tot_nats += float(-(tokll * torch.from_numpy(mask)).sum())
        tot_bytes += int(tok_len[yb[mask]].sum())
        tot_tokens += int(mask.sum())
    model.train()
    return {"bits_per_byte": tot_nats / (LN2 * tot_bytes),
            "bits_per_token": tot_nats / (LN2 * tot_tokens),
            "eval_bytes": tot_bytes, "eval_tokens": tot_tokens, "n_windows": len(plan)}


def train_one(arm, steps, lr, seed, p, dat, fm, curve_windows, tag, checkpoints=True):
    set_seeds(seed)
    vocab = dat["vocab"]
    model = GPT(vocab, p)
    decay = [q for q in model.parameters() if q.dim() >= 2]
    nodecay = [q for q in model.parameters() if q.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": p["weight_decay"]},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=lr, betas=tuple(p["betas"]),
    )
    rng = np.random.default_rng(seed)
    warm, ctx, bs = p["warmup"], p["block_size"], p["batch_size"]
    tps = bs * ctx                                    # tokens per step
    fl_per_step = fm["train_flops_per_token"] * tps
    bytes_per_token = dat["bytes_per_token"]
    ckpts = set()
    if checkpoints:
        nck = p["n_checkpoints"]
        ckpts = {max(1, round(steps * (i + 1) / nck)) for i in range(nck)}
    curve, losses = [], []
    t0 = time.time()
    for it in range(steps):
        if it < warm:
            cur = lr * (it + 1) / warm
        else:
            prog = (it - warm) / max(1, steps - warm)
            cur = lr * (p["lr_min_frac"] + (1 - p["lr_min_frac"]) * 0.5 * (1 + math.cos(math.pi * prog)))
        for g in opt.param_groups:
            g["lr"] = cur
        x, y = get_batch(dat["train"], rng, bs, ctx)
        loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), p["grad_clip"])
        opt.step()
        losses.append(float(loss.detach()))
        if (it + 1) in ckpts:
            ev = eval_bpb(model, dat["val"], dat["tok_len"], ctx, p["eval_batch"], curve_windows)
            curve.append({"step": it + 1, "tokens_seen": (it + 1) * tps,
                          "bytes_seen": (it + 1) * tps * bytes_per_token,
                          "train_flops": (it + 1) * fl_per_step,
                          "sub_bits_per_byte": round(ev["bits_per_byte"], 5)})
    train_s = time.time() - t0
    ev = eval_bpb(model, dat["val"], dat["tok_len"], ctx, p["eval_batch"])
    rec = {"tag": tag, "arm": arm, "vocab": vocab, "seed": int(seed), "lr": lr, "steps": steps,
           "n_params": n_params(model), "train_seconds": round(train_s, 1),
           "tokens_processed": steps * tps, "bytes_processed": steps * tps * bytes_per_token,
           "train_flops": steps * fl_per_step,
           "epochs_over_corpus": steps * tps * bytes_per_token / dat["n_train_bytes"],
           "final_train_loss_ma50": round(float(np.mean(losses[-50:])), 4),
           "val_bits_per_byte": round(ev["bits_per_byte"], 5),
           "val_bits_per_token": round(ev["bits_per_token"], 5),
           "val_eval_bytes": ev["eval_bytes"], "val_eval_tokens": ev["eval_tokens"],
           "curve": curve}
    print(f"  [{tag:22s}] arm={arm:4s} V={vocab:4d} lr={lr:.4f} seed={seed} steps={steps:4d} "
          f"P={rec['n_params']:6d} bpb={rec['val_bits_per_byte']:.4f} ({rec['train_seconds']}s)",
          flush=True)
    return model, rec


# ---------------------------------------------------------------- spelling probe
@torch.no_grad()
def seq_logprob(model, ids, ctx):
    """Total log-prob (nats) of tokens 1..end of a single sequence, canonical tokenisation."""
    ids = ids[:ctx + 1]
    x = torch.from_numpy(ids[:-1]).unsqueeze(0)
    y = torch.from_numpy(ids[1:]).unsqueeze(0)
    lp = F.log_softmax(model(x).float(), dim=-1)
    return float(lp.gather(-1, y.unsqueeze(-1)).sum())


def build_probe(text, n_train_chars, p, rng):
    """Word items: (context, true_word, [distractors]) drawn from the VAL split only."""
    train_txt, val_txt = text[:n_train_chars], text[n_train_chars:]
    train_counts = Counter(w.lower() for w in re.findall(r"[A-Za-z]+", train_txt))
    minlen, nd = p["probe_min_word_len"], p["probe_n_distractors"]
    nctx, want = p["probe_context_chars"], p["probe_words_per_pool"]
    pools = {"rare": [], "common": []}
    seen = set()
    for m in re.finditer(r"[A-Za-z]+", val_txt):
        w = m.group(0)
        if len(w) < minlen or m.start() < nctx:
            continue
        lw = w.lower()
        if lw in seen:
            continue
        c = train_counts.get(lw, 0)
        pool = ("rare" if c <= p["probe_rare_max_train_count"]
                else "common" if c >= p["probe_common_min_train_count"] else None)
        if pool is None or len(pools[pool]) >= want:
            continue
        # adjacent INTERIOR transpositions: first and last character stay fixed
        cands = []
        for i in range(1, len(w) - 2):
            if w[i] != w[i + 1]:
                v = w[:i] + w[i + 1] + w[i] + w[i + 2:]
                if v != w and v not in cands:
                    cands.append(v)
        if len(cands) < nd:
            continue
        pick = sorted(rng.choice(len(cands), size=nd, replace=False).tolist())
        seen.add(lw)
        pools[pool].append({"word": w, "pool": pool, "train_count": int(c),
                            "context": val_txt[m.start() - nctx:m.start()],
                            "distractors": [cands[i] for i in pick]})
    return pools["rare"] + pools["common"]


@torch.no_grad()
def run_probe(model, tokk, items, ctx, bpe_tok):
    """Returns per-item records: which candidate wins, and incremental bits/byte of the word.

    `n_bpe_tokens` is ALWAYS measured with the BPE tokenizer (even for the char arm) so the
    same word falls in the same bin in both arms and the bins are comparable.
    """
    model.eval()
    out = []
    for it in items:
        c = it["context"]
        lp_ctx = seq_logprob(model, tokk.encode(c), ctx)
        cands = [it["word"]] + it["distractors"]
        lps = [seq_logprob(model, tokk.encode(c + w), ctx) for w in cands]
        best = int(np.argmax(lps))
        nb = len(it["word"].encode("utf-8"))
        out.append({"word": it["word"], "pool": it["pool"], "correct": best == 0,
                    "margin_bits": (lps[0] - max(lps[1:])) / LN2,
                    "word_bits_per_byte": (lp_ctx - lps[0]) / (LN2 * nb),
                    "n_bpe_tokens": int(len(bpe_tok.encode(it["word"])))})
    model.train()
    return out


def agg_probe(recs, key=None):
    sel = [r for r in recs if key is None or key(r)]
    if not sel:
        return None
    return {"n": len(sel),
            "accuracy": round(float(np.mean([r["correct"] for r in sel])), 4),
            "mean_margin_bits": round(float(np.mean([r["margin_bits"] for r in sel])), 4),
            "word_bits_per_byte": round(float(np.mean([r["word_bits_per_byte"] for r in sel])), 4)}


# ------------------------------------------------------------------------------ misc
def interp_at(curve, xkey, xval, ykey="sub_bits_per_byte"):
    """Linear interpolation of a training curve at a given x (returns None if out of range)."""
    xs = [c[xkey] for c in curve]
    ys = [c[ykey] for c in curve]
    if xval < xs[0] or xval > xs[-1]:
        return None
    return float(np.interp(xval, xs, ys))


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


# ------------------------------------------------------------------------------ main
def main():
    cfg = load_config()
    p = cfg["params"]
    t_start = time.time()
    set_seeds(int(cfg.get("seed", 0)))

    # ---------------------------------------------------------------- data + tokenizers
    txt_path = HERE / "data" / "tinyshakespeare.txt"
    if not txt_path.exists():
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(cfg["dataset"]["source"], txt_path)
    text = txt_path.read_text(encoding="utf-8")
    assert text.isascii(), "corpus must be ASCII for 1 char == 1 byte"
    chars = sorted(set(text))
    base_vocab = len(chars)
    assert base_vocab == p["char_vocab_size"], f"expected {p['char_vocab_size']} chars, got {base_vocab}"
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in text], dtype=np.int32)
    n_train = int(p["train_frac"] * len(ids))
    tr_c, va_c = ids[:n_train], ids[n_train:]

    Vb = p["bpe_vocab_size"]
    cache = HERE / "data" / f"bpe_{Vb}_{p['train_frac']}.npz"
    t_bpe = time.time()
    if cache.exists():
        z = np.load(cache)
        merges = [(int(a), int(b)) for a, b in z["merges"]]
        tok_len_b = z["tok_len"]
        tr_b = z["train"]
    else:
        merges, tok_len_b, tr_b = fit_bpe(tr_c, base_vocab, Vb - base_vocab)
        np.savez_compressed(cache, merges=np.array(merges, dtype=np.int64),
                            tok_len=tok_len_b, train=tr_b)
    va_b = apply_merges(va_c, merges, base_vocab)
    bpe_fit_s = round(time.time() - t_bpe, 1)
    assert len(merges) == Vb - base_vocab

    tok_len_c = np.ones(base_vocab, dtype=np.int64)
    arms = {
        "char": {"vocab": base_vocab, "train": tr_c.astype(np.int64), "val": va_c.astype(np.int64),
                 "tok_len": tok_len_c, "merges": []},
        "bpe": {"vocab": Vb, "train": tr_b.astype(np.int64), "val": va_b.astype(np.int64),
                "tok_len": tok_len_b, "merges": merges},
    }
    for a, d in arms.items():
        d["n_train_bytes"] = int(len(tr_c))
        d["n_val_bytes"] = int(len(va_c))
        d["bytes_per_token"] = float(len(tr_c) / len(d["train"]))
        d["tokenizer"] = Tok(stoi, d["merges"], base_vocab, d["tok_len"], d["vocab"])
    print(f"corpus: {len(text)} bytes | train {len(tr_c)} bytes | val {len(va_c)} bytes")
    print(f"tokens: char train={len(tr_c)} val={len(va_c)} | "
          f"bpe train={len(tr_b)} val={len(va_b)} "
          f"(compression {arms['bpe']['bytes_per_token']:.3f} bytes/token, fit {bpe_fit_s}s)")

    # ------------------------------------------------------------------ iso-FLOP budget
    fm = {a: flop_model(d["vocab"], p) for a, d in arms.items()}
    tps = p["batch_size"] * p["block_size"]
    steps = {"bpe": int(p["steps_bpe_anchor"])}
    ratio = fm["bpe"]["train_flops_per_token"] / fm["char"]["train_flops_per_token"]
    steps["char"] = int(round(steps["bpe"] * ratio))
    budget = {a: fm[a]["train_flops_per_token"] * tps * steps[a] for a in arms}
    flop_mismatch = abs(budget["char"] - budget["bpe"]) / budget["bpe"]
    print(f"iso-FLOP: char/bpe flops-per-token ratio {1/ratio:.4f} -> "
          f"steps char={steps['char']} bpe={steps['bpe']}; "
          f"budget {budget['char']:.3e} vs {budget['bpe']:.3e} (mismatch {flop_mismatch*100:.3f}%)")

    curve_windows = {a: max(4, int(p["curve_eval_chars"] / d["bytes_per_token"] / (p["block_size"] // 2)))
                     for a, d in arms.items()}

    # ------------------------------------------------------------------- lr probe (seed 0)
    probe_steps = {"bpe": int(p["lr_probe_steps_bpe"])}
    probe_steps["char"] = int(round(probe_steps["bpe"] * ratio))
    lr_sweep, best_lr, grids, edge_extended = {}, {}, {}, {}
    print("lr probe (short runs, seed %d):" % p["lr_probe_seed"])
    for a in ("char", "bpe"):
        grid = list(p["lr_grid"])
        res = {}
        for lr in grid:
            _, r = train_one(a, probe_steps[a], lr, p["lr_probe_seed"], p, arms[a], fm[a],
                             curve_windows[a], f"lrprobe-{a}", checkpoints=False)
            res[lr] = r["val_bits_per_byte"]
        ext = False
        if p["lr_probe_edge_extend"]:
            b = min(res, key=res.get)
            if b == grid[0] or b == grid[-1]:
                lr2 = round(b / 2, 8) if b == grid[0] else round(b * 2, 8)
                _, r = train_one(a, probe_steps[a], lr2, p["lr_probe_seed"], p, arms[a], fm[a],
                                 curve_windows[a], f"lrprobe-{a}", checkpoints=False)
                res[lr2] = r["val_bits_per_byte"]
                grid = sorted(grid + [lr2])
                ext = True
        best = min(res, key=res.get)
        lr_sweep[a] = {str(k): v for k, v in sorted(res.items())}
        best_lr[a], grids[a], edge_extended[a] = best, grid, ext
        print(f"  -> {a}: best lr {best} (grid {grid}, extended={ext}, "
              f"still at edge={best in (grid[0], grid[-1])})")

    # ------------------------------------------------------------------------ main runs
    runs, models = [], {}
    print("main iso-FLOP runs:")
    for a in ("char", "bpe"):
        for sd in p["seeds"]:
            m, r = train_one(a, steps[a], best_lr[a], sd, p, arms[a], fm[a],
                             curve_windows[a], f"main-{a}-s{sd}")
            runs.append(r)
            models[(a, sd)] = m

    by_arm = {a: [r for r in runs if r["arm"] == a] for a in ("char", "bpe")}
    bpb = {a: [r["val_bits_per_byte"] for r in by_arm[a]] for a in by_arm}
    mean_bpb = {a: float(np.mean(v)) for a, v in bpb.items()}
    delta = mean_bpb["bpe"] - mean_bpb["char"]
    seed_spread = {a: float(max(v) - min(v)) for a, v in bpb.items()}

    # ------------------------------------------------------- secondary: iso-BYTES reading
    # The char arm sees far fewer BYTES at iso-FLOPs. Read both arms' curves at the
    # number of training bytes the char arm actually consumed. (Not a compute-matched
    # comparison -- the BPE arm reaches that point with fewer FLOPs -- it isolates
    # "who learns more per byte of text".)
    iso_bytes = by_arm["char"][0]["bytes_processed"]
    iso_b = {}
    for a in ("char", "bpe"):
        vals = [interp_at(r["curve"], "bytes_seen", iso_bytes) for r in by_arm[a]]
        vals = [v for v in vals if v is not None]
        iso_b[a] = float(np.mean(vals)) if vals else None
    iso_bytes_delta = (iso_b["bpe"] - iso_b["char"]) if None not in iso_b.values() else None

    # ------------------------------------------------------------------- spelling probe
    print("spelling / character-composition probe:")
    items = build_probe(text, n_train, p, np.random.default_rng(1234))
    bpe_tok = arms["bpe"]["tokenizer"]
    probe_raw, probe_agg = {}, {}
    for a in ("char", "bpe"):
        per_seed = []
        for sd in p["seeds"]:
            per_seed.append(run_probe(models[(a, sd)], arms[a]["tokenizer"], items,
                                      p["block_size"], bpe_tok))
        flat = [r for s in per_seed for r in s]
        probe_raw[a] = per_seed
        probe_agg[a] = {
            "overall": agg_probe(flat),
            "rare": agg_probe(flat, lambda r: r["pool"] == "rare"),
            "common": agg_probe(flat, lambda r: r["pool"] == "common"),
            "bpe_tok_1": agg_probe(flat, lambda r: r["n_bpe_tokens"] == 1),
            "bpe_tok_2_3": agg_probe(flat, lambda r: 2 <= r["n_bpe_tokens"] <= 3),
            "bpe_tok_4plus": agg_probe(flat, lambda r: r["n_bpe_tokens"] >= 4),
            "per_seed_accuracy": [round(float(np.mean([r["correct"] for r in s])), 4) for s in per_seed],
        }
        k = int(sum(r["correct"] for r in flat))
        probe_agg[a]["overall"]["wilson95"] = wilson(k, len(flat))
        print(f"  {a:4s}: acc={probe_agg[a]['overall']['accuracy']:.3f} "
              f"(rare {probe_agg[a]['rare']['accuracy']:.3f} / "
              f"common {probe_agg[a]['common']['accuracy']:.3f}) "
              f"word_bpb={probe_agg[a]['overall']['word_bits_per_byte']:.3f}")
    # paired McNemar-style counts on matched items (same item, same seed index)
    def mcnemar(sel=lambda r: True):
        c = b = 0
        for si in range(len(p["seeds"])):
            for rc, rb in zip(probe_raw["char"][si], probe_raw["bpe"][si]):
                if not sel(rc):
                    continue
                if rc["correct"] and not rb["correct"]:
                    c += 1
                elif rb["correct"] and not rc["correct"]:
                    b += 1
        n = c + b
        pv = None
        if n:
            k = min(c, b)
            pv = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))
        return {"char_only_correct": c, "bpe_only_correct": b, "n_discordant": n,
                "two_sided_p": (round(pv, 6) if pv is not None else None)}

    mc_all = mcnemar()
    mc_multi = mcnemar(lambda r: r["n_bpe_tokens"] >= 4)
    mc_few = mcnemar(lambda r: r["n_bpe_tokens"] <= 3)
    mcnemar_p = mc_all["two_sided_p"]
    pair_c, pair_b, n_disc = mc_all["char_only_correct"], mc_all["bpe_only_correct"], mc_all["n_discordant"]
    chance = 1.0 / (1 + p["probe_n_distractors"])
    tok_hist = Counter(int(len(arms["bpe"]["tokenizer"].encode(it["word"]))) for it in items)
    # per-item table (seed-averaged), so the split can be re-derived from results.json
    item_table = []
    for j, it in enumerate(items):
        item_table.append({
            "word": it["word"], "pool": it["pool"], "train_count": it["train_count"],
            "n_bpe_tokens": probe_raw["bpe"][0][j]["n_bpe_tokens"],
            "char_correct": int(sum(probe_raw["char"][s][j]["correct"] for s in range(len(p["seeds"])))),
            "bpe_correct": int(sum(probe_raw["bpe"][s][j]["correct"] for s in range(len(p["seeds"])))),
        })

    # -------------------------------------------------------------------------- metrics
    metrics = {
        "headline": ("validation BITS PER BYTE at matched training FLOPs, char(65) vs BPE(512), "
                     "plus a transposition spelling probe"),
        "substitution_note": cfg["dataset"]["substitution_note"],
        "n_params": {a: by_arm[a][0]["n_params"] for a in by_arm},
        "vocab": {a: arms[a]["vocab"] for a in arms},
        "corpus_bytes": len(text), "train_bytes": int(len(tr_c)), "val_bytes": int(len(va_c)),
        "train_tokens": {a: int(len(arms[a]["train"])) for a in arms},
        "val_tokens": {a: int(len(arms[a]["val"])) for a in arms},
        "bytes_per_token": {a: round(arms[a]["bytes_per_token"], 4) for a in arms},
        "bpe_compression_ratio": round(arms["bpe"]["bytes_per_token"], 4),
        "bpe_fit_seconds": bpe_fit_s,
        "context_bytes_effective": {a: round(p["block_size"] * arms[a]["bytes_per_token"], 1) for a in arms},

        "ISO_FLOP_ACCOUNTING": {
            "formula": ("fwd_flops_per_token = 2*[n_layer*(4*d^2 + 2*d*d_ff) + d*V] + n_layer*2*ctx*d ; "
                        "train_flops = 3 * fwd_flops_per_token * tokens_processed"),
            "flop_model": fm,
            "tokens_per_step": tps,
            "steps": steps,
            "step_ratio_char_over_bpe": round(steps["char"] / steps["bpe"], 4),
            "total_train_flops": {a: float(budget[a]) for a in budget},
            "flop_mismatch_frac": round(flop_mismatch, 6),
            "tokens_processed": {a: by_arm[a][0]["tokens_processed"] for a in by_arm},
            "bytes_processed": {a: round(by_arm[a][0]["bytes_processed"], 1) for a in by_arm},
            "bytes_advantage_bpe_over_char": round(
                by_arm["bpe"][0]["bytes_processed"] / by_arm["char"][0]["bytes_processed"], 4),
            "epochs_over_corpus": {a: round(by_arm[a][0]["epochs_over_corpus"], 3) for a in by_arm},
            "note": ("the char arm gets MORE steps because its output head is 80x65 not 80x512; "
                     "the BPE arm still sees ~2.4x more BYTES of text -- that IS the mechanism "
                     "under test, not a confound"),
        },

        "lr_probe": {"grid_run": grids, "val_bpb": lr_sweep, "best_lr": best_lr,
                     "edge_extended": edge_extended,
                     "probe_steps": probe_steps, "probe_seed": p["lr_probe_seed"],
                     "caveat": "lr chosen at 25% of the final budget; may not be optimal at full length"},

        "PRIMARY_val_bits_per_byte": {a: round(mean_bpb[a], 5) for a in mean_bpb},
        "PRIMARY_per_seed": {a: bpb[a] for a in bpb},
        "PRIMARY_delta_bpe_minus_char": round(delta, 5),
        "PRIMARY_winner_on_mean": "bpe" if delta < 0 else "char",
        "PRIMARY_per_seed_delta": [round(b - c, 5) for b, c in zip(bpb["bpe"], bpb["char"])],
        "PRIMARY_all_seeds_same_sign": bool(all(
            (b - c < 0) == (delta < 0) for b, c in zip(bpb["bpe"], bpb["char"]))),
        "seed_spread_bpb": {a: round(v, 5) for a, v in seed_spread.items()},
        "seed_sd_bpb": {a: round(float(np.std(v, ddof=1)), 5) for a, v in bpb.items()},
        "delta_in_seed_spreads": (round(abs(delta) / max(1e-9, float(np.mean(list(seed_spread.values())))), 2)),
        "val_bits_per_token": {a: round(float(np.mean([r["val_bits_per_token"] for r in by_arm[a]])), 5)
                               for a in by_arm},
        "val_eval_bytes": {a: by_arm[a][0]["val_eval_bytes"] for a in by_arm},
        "val_eval_byte_coverage": {a: round(by_arm[a][0]["val_eval_bytes"] / len(va_c), 4) for a in by_arm},

        "SECONDARY_iso_bytes": {
            "at_train_bytes": round(iso_bytes, 1),
            "bits_per_byte_subsampled_val": {a: (round(v, 5) if v is not None else None)
                                             for a, v in iso_b.items()},
            "delta_bpe_minus_char": (round(iso_bytes_delta, 5) if iso_bytes_delta is not None else None),
            "note": ("read off the training curves at equal TRAINING BYTES, not equal FLOPs; "
                     "the BPE arm reaches this point with ~2.4x fewer FLOPs, so this is a "
                     "per-byte data-efficiency reading, not a compute-matched one. Uses the "
                     "subsampled curve eval, so it is not directly comparable to the headline number."),
        },

        "SPELLING_PROBE": {
            "design": ("144 words from the VAL split (72 rare: train count <= 3; 72 common: train "
                       "count >= 100), each >= 6 chars. 3 distractors per word made by transposing "
                       "an adjacent INTERIOR character pair (first/last char fixed), so every "
                       "distractor has the identical character multiset and length. Each model "
                       "scores the COMPLETE string context(96 bytes)+candidate under its own "
                       "canonical tokenisation and must rank the true spelling first. 4-way choice, "
                       "chance = 0.25."),
            "chance": chance,
            "n_items": len(items),
            "n_rare": sum(1 for i in items if i["pool"] == "rare"),
            "n_common": sum(1 for i in items if i["pool"] == "common"),
            "bpe_tokens_per_word_hist": {str(k): v for k, v in sorted(tok_hist.items())},
            "results": probe_agg,
            "delta_accuracy_char_minus_bpe": round(
                probe_agg["char"]["overall"]["accuracy"] - probe_agg["bpe"]["overall"]["accuracy"], 4),
            "delta_accuracy_char_minus_bpe_by_bin": {
                b: (round(probe_agg["char"][b]["accuracy"] - probe_agg["bpe"][b]["accuracy"], 4)
                    if probe_agg["char"].get(b) and probe_agg["bpe"].get(b) else None)
                for b in ("rare", "common", "bpe_tok_1", "bpe_tok_2_3", "bpe_tok_4plus")},
            "paired_mcnemar": mc_all,
            "paired_mcnemar_words_of_4plus_bpe_tokens": mc_multi,
            "paired_mcnemar_words_of_le3_bpe_tokens": mc_few,
            "item_table": item_table,
            "caveat": ("a BPE model assigns probability only to its canonical tokenisation, which "
                       "is a LOWER bound on the true string probability and is looser for the odd "
                       "distractor strings -- this biases the probe slightly IN FAVOUR of the BPE arm"),
        },

        "runs": [{k: v for k, v in r.items() if k != "curve"} for r in runs],
        "curves": {r["tag"]: r["curve"] for r in runs},
        "wall_clock_s": None,
    }

    # honest verdict: a mean difference smaller than the worse arm's seed spread, or with
    # disagreeing per-seed signs, is reported as INCONCLUSIVE rather than as a win.
    worst_spread = max(seed_spread.values())
    if not metrics["PRIMARY_all_seeds_same_sign"]:
        verdict = "inconclusive (per-seed signs disagree)"
    elif abs(delta) < worst_spread:
        verdict = "inconclusive (mean gap smaller than the worse arm's seed spread)"
    else:
        verdict = ("bpe wins bits-per-byte" if delta < 0 else "char wins bits-per-byte")
    metrics["PRIMARY_verdict"] = verdict

    pa = probe_agg["char"]["overall"]["accuracy"] - probe_agg["bpe"]["overall"]["accuracy"]
    metrics["SPELLING_PROBE"]["verdict"] = (
        ("char (byte-level) wins" if pa > 0 else "bpe wins") +
        (" (p<0.05 paired McNemar)" if (mcnemar_p is not None and mcnemar_p < 0.05)
         else " on the point estimate, NOT significant (paired McNemar p="
              f"{mcnemar_p})"))

    # ---------------------------------------------------------------------------- chart
    make_chart(metrics)

    metrics["wall_clock_s"] = round(time.time() - t_start, 1)
    results = {
        "id": cfg["id"], "git_commit": git_sha(), "seed": int(cfg.get("seed", 0)),
        "duration_sec": metrics["wall_clock_s"], "metrics": metrics,
        "env": env_info(), "status": "done",
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n=== SUMMARY ===")
    print(f"val bits/byte @ iso-FLOPs: char {mean_bpb['char']:.4f} | bpe {mean_bpb['bpe']:.4f} "
          f"| delta(bpe-char) {delta:+.4f} -> {metrics['PRIMARY_verdict']}")
    print(f"spelling probe: char {probe_agg['char']['overall']['accuracy']:.3f} | "
          f"bpe {probe_agg['bpe']['overall']['accuracy']:.3f} | chance {chance:.2f} "
          f"| McNemar p={mcnemar_p}")
    print(f"wall clock {metrics['wall_clock_s']}s")


def make_chart(m):
    """Redraws chart.png from the metrics dict alone (so it can be regenerated from results.json)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_arm = {a: [dict(r, curve=m["curves"][r["tag"]]) for r in m["runs"] if r["arm"] == a]
              for a in ("char", "bpe")}
    probe_agg = m["SPELLING_PROBE"]["results"]
    chance = m["SPELLING_PROBE"]["chance"]
    n_seeds = len(m["PRIMARY_per_seed"]["char"])
    col = {"char": "#1f77b4", "bpe": "#d62728"}
    lbl = {"char": "char / byte-level (V=65)", "bpe": "BPE-512"}
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 9.5))

    a0 = ax[0, 0]
    for a in ("char", "bpe"):
        for i, r in enumerate(by_arm[a]):
            a0.plot([c["train_flops"] for c in r["curve"]], [c["sub_bits_per_byte"] for c in r["curve"]],
                    color=col[a], alpha=0.85, lw=1.6,
                    label=f"{lbl[a]}, {r['steps']} steps" if i == 0 else None)
        a0.scatter([m["ISO_FLOP_ACCOUNTING"]["total_train_flops"][a]] * len(by_arm[a]),
                   [r["val_bits_per_byte"] for r in by_arm[a]], color=col[a], marker="*", s=190,
                   zorder=5, edgecolor="k", linewidth=0.5)
    a0.axvline(m["ISO_FLOP_ACCOUNTING"]["total_train_flops"]["bpe"], color="k", ls=":", lw=1)
    a0.set_xscale("log")
    a0.set_xlabel("training FLOPs (3 x fwd, cumulative)")
    a0.set_ylabel("val bits per byte")
    a0.set_title("(a) ISO-FLOP: bits-per-byte vs compute\nlines = subsampled val, stars = full val at the budget")
    a0.legend(fontsize=8)
    a0.grid(alpha=0.3)

    a1 = ax[0, 1]
    for a in ("char", "bpe"):
        for i, r in enumerate(by_arm[a]):
            a1.plot([c["bytes_seen"] / 1e6 for c in r["curve"]], [c["sub_bits_per_byte"] for c in r["curve"]],
                    color=col[a], alpha=0.85, lw=1.6, label=lbl[a] if i == 0 else None)
    ib = m["SECONDARY_iso_bytes"]["at_train_bytes"] / 1e6
    a1.axvline(ib, color="k", ls="--", lw=1)
    a1.text(ib * 0.98, a1.get_ylim()[1], "iso-BYTES ", fontsize=8, va="top", ha="right")
    a1.set_xscale("log")
    a1.set_xlabel("training text seen (millions of bytes)")
    a1.set_ylabel("val bits per byte (subsampled val)")
    sb = m["SECONDARY_iso_bytes"]["bits_per_byte_subsampled_val"]
    a1.set_title(f"(b) same curves vs TEXT seen -- at equal BYTES the char arm\n"
                 f"is AHEAD ({sb['char']:.3f} vs {sb['bpe']:.3f}); BPE gets there for ~2.4x fewer FLOPs")
    a1.legend(fontsize=8)
    a1.grid(alpha=0.3)

    a2 = ax[1, 0]
    lo = min(min(m["PRIMARY_per_seed"][a]) for a in ("char", "bpe"))
    hi = max(max(m["PRIMARY_per_seed"][a]) for a in ("char", "bpe"))
    pad = 0.25 * (hi - lo)
    for i, a in enumerate(("char", "bpe")):
        vs = m["PRIMARY_per_seed"][a]
        mu = m["PRIMARY_val_bits_per_byte"][a]
        a2.scatter([i] * len(vs), vs, color=col[a], s=90, zorder=4, edgecolor="k", linewidth=0.5)
        a2.hlines(mu, i - 0.22, i + 0.22, color=col[a], lw=3, zorder=3)
        a2.text(i + 0.28, mu, f"mean {mu:.4f}", va="center", fontsize=10, fontweight="bold")
        for s, v in enumerate(vs):
            a2.text(i - 0.28, v, f"s{s}", va="center", ha="right", fontsize=8, color="0.3")
    a2.set_xlim(-0.75, 2.0)
    a2.set_ylim(lo - pad, hi + pad)
    a2.set_xticks([0, 1])
    a2.set_xticklabels([f"char (V=65)\n{m['n_params']['char']:,}p, {m['ISO_FLOP_ACCOUNTING']['steps']['char']} steps, "
                        f"{m['ISO_FLOP_ACCOUNTING']['bytes_processed']['char']/1e6:.2f}M bytes",
                        f"BPE-512\n{m['n_params']['bpe']:,}p, {m['ISO_FLOP_ACCOUNTING']['steps']['bpe']} steps, "
                        f"{m['ISO_FLOP_ACCOUNTING']['bytes_processed']['bpe']/1e6:.2f}M bytes"], fontsize=9)
    a2.set_ylabel("val bits per byte (full val, strided)")
    a2.set_title(f"(c) HEADLINE at matched FLOPs ({m['ISO_FLOP_ACCOUNTING']['total_train_flops']['bpe']:.2e}, "
                 f"mismatch {m['ISO_FLOP_ACCOUNTING']['flop_mismatch_frac']*100:.2f}%)\n"
                 f"delta {m['PRIMARY_delta_bpe_minus_char']:+.4f} -- {m['PRIMARY_verdict']}", fontsize=10)
    a2.grid(alpha=0.3, axis="y")

    a3 = ax[1, 1]
    groups = [("overall", "overall"), ("rare", "rare word"), ("common", "common word"),
              ("bpe_tok_2_3", "word = 2-3\nBPE tokens"), ("bpe_tok_4plus", "word = 4+\nBPE tokens")]
    groups = [g for g in groups if probe_agg["char"].get(g[0])]
    w = 0.36
    xs = np.arange(len(groups))
    for j, a in enumerate(("char", "bpe")):
        v = [probe_agg[a][g]["accuracy"] for g, _ in groups]
        a3.bar(xs + (j - 0.5) * w, v, width=w, color=col[a], label=lbl[a])
        for x, y in zip(xs + (j - 0.5) * w, v):
            a3.text(x, y + 0.012, f"{y:.3f}", ha="center", fontsize=8)
    a3.axhline(chance, color="k", ls="--", lw=1)
    a3.text(-0.6, chance + 0.015, "chance", fontsize=8)
    a3.set_xticks(xs)
    a3.set_xticklabels([f"{lab}\n(n={probe_agg['char'][g]['n']})" for g, lab in groups], fontsize=8)
    a3.set_ylabel("spelling-probe accuracy")
    a3.set_ylim(0, 1.18)
    mcp = m["SPELLING_PROBE"]["paired_mcnemar"]["two_sided_p"]
    a3.set_title(f"(d) character-composition probe: true word vs 3 interior\n"
                 f"transpositions (4-way). char-bpe = "
                 f"{m['SPELLING_PROBE']['delta_accuracy_char_minus_bpe']:+.4f}, McNemar p={mcp:.2f}",
                 fontsize=10)
    a3.legend(fontsize=8, loc="lower left", ncol=2)
    a3.grid(alpha=0.3, axis="y")

    fig.suptitle(f"Byte/char vs BPE-512 at iso-FLOPs -- tiny-shakespeare, 2-layer LM (~0.2M params), "
                 f"{n_seeds} seeds", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(HERE / "chart.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
