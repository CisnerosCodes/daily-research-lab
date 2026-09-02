"""Find the right attention logit scale for YOUR model and corpus, on a CPU.

This reproduces the daily-research-lab char-LM recipe *through the drop-in module*
(2-layer pre-norm GPT, d_model 128, block 96, batch 16, AdamW 3e-3 cosine, 600 steps),
so the anchor numbers are checkable:

    python -m attnscale.bench --text data/tinyshakespeare.txt --key none --head-dim 4 --seed 0   # 3.0930 bpc

THE MAIN THING: sweep the constant for your configuration. This is the measurement the paper
says to make rather than copying our numbers.

    python -m attnscale.bench --text my_corpus.txt --head-dim 16 --sweep-c 1 2 4 8 16

It trains one short run per value, prints a table, and names the best c. Then use it:

    from attnscale import KeyScale
    k = KeyScale(n_head, c=BEST).cuda()(k)

Single runs still work for any arm: --key none|kscale|qknorm|knorm|fkn (with --c for kscale).
"""
import argparse, json, math, os, random, sys, time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from attnscale import ScaledAttention, initial_logit_std  # noqa: E402

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
LN2 = math.log(2.0)


def set_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


class Block(nn.Module):
    """Same module registration order as the lab harness (ln1, ln2, attn[c_attn, c_proj], fc, out)."""

    def __init__(self, cfg, key, key_kwargs):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.attn = ScaledAttention(cfg, key=key, key_kwargs=key_kwargs)
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.out = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, vocab, cfg, n_layer, key, key_kwargs):
        super().__init__()
        self.vocab, self.block_size = vocab, cfg.block_size
        self.tok = nn.Embedding(vocab, cfg.n_embd)
        self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg, key, key_kwargs) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, vocab, bias=False)
        self.apply(self._init)
        for name, prm in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("out.weight"):
                nn.init.normal_(prm, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))
        with torch.no_grad():
            for m in self.modules():
                if hasattr(m, "gain") and isinstance(getattr(m, "gain"), torch.Tensor):
                    m.gain.fill_(1.0)
                if hasattr(m, "alpha") and isinstance(getattr(m, "alpha"), torch.Tensor):
                    pass   # keep alpha_init

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


def get_batch(data, rng, bsz, bs):
    ix = rng.integers(0, len(data) - bs - 1, size=bsz)
    x = np.stack([data[i:i + bs] for i in ix]); y = np.stack([data[i + 1:i + 1 + bs] for i in ix])
    return torch.from_numpy(x), torch.from_numpy(y)


@torch.no_grad()
def eval_bpc(model, val_ids, bs, eval_batch=32, max_blocks=480):
    model.eval()
    n_blocks = min((len(val_ids) - 1) // bs, max_blocks)
    xs = np.stack([val_ids[i * bs:(i + 1) * bs] for i in range(n_blocks)])
    ys = np.stack([val_ids[i * bs + 1:(i + 1) * bs + 1] for i in range(n_blocks)])
    nats, toks = 0.0, 0
    for s in range(0, n_blocks, eval_batch):
        xb, yb = torch.from_numpy(xs[s:s + eval_batch]), torch.from_numpy(ys[s:s + eval_batch])
        nats += float(F.cross_entropy(model(xb).reshape(-1, model.vocab), yb.reshape(-1), reduction="sum"))
        toks += yb.numel()
    model.train()
    return nats / (LN2 * toks)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", required=True, help="path to a UTF-8 text file (char-level LM)")
    ap.add_argument("--val-text", default=None, help="optional separate validation file")
    ap.add_argument("--key", default="kscale", choices=["none", "kscale", "qknorm", "knorm", "fkn"])
    ap.add_argument("--c", type=float, default=1.0, help="the constant for --key kscale")
    ap.add_argument("--sweep-c", type=float, nargs="*", default=None,
                    help="train one run per value and report the best (the recommended workflow)")
    ap.add_argument("--head-dim", type=int, default=32)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--block", type=int, default=96)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=None, help="default: 10%% of steps")
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-frac", type=float, default=0.9)
    ap.add_argument("--alpha-init", type=float, default=1.0)
    ap.add_argument("--alpha-frozen", action="store_true")
    ap.add_argument("--json", default=None, help="write results to this JSON path")
    a = ap.parse_args()

    text = Path(a.text).read_text()
    if a.val_text:
        vtext = Path(a.val_text).read_text()
        chars = sorted(set(text) | set(vtext))
        stoi = {c: i for i, c in enumerate(chars)}
        train_ids = np.array([stoi[c] for c in text], dtype=np.int64)
        val_ids = np.array([stoi[c] for c in vtext], dtype=np.int64)
    else:
        chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
        ids = np.array([stoi[c] for c in text], dtype=np.int64)
        n = int(a.train_frac * len(ids)); train_ids, val_ids = ids[:n], ids[n:]
    vocab = len(chars)
    assert a.d_model % a.head_dim == 0
    cfg = SimpleNamespace(n_embd=a.d_model, n_head=a.d_model // a.head_dim, block_size=a.block,
                          dropout=0.0, bias=False)
    def build_and_train(key, c, seed):
        set_seeds(seed)
        kw = {"c": c} if key == "kscale" else ({"alpha_init": a.alpha_init,
                                                "alpha_learnable": not a.alpha_frozen} if key == "fkn" else {})
        model = GPT(vocab, cfg, a.n_layer, key, kw)
        n_params = sum(p.numel() for p in model.parameters())
        # diagnostic: the logit scale this model STARTS from
        with torch.no_grad():
            xb, _ = get_batch(train_ids, np.random.default_rng(0), 8, a.block)
            h = model.tok(xb) + model.pos(torch.arange(xb.shape[1]))
            blk = model.blocks[0]
            qq, kk, _ = blk.attn.c_attn(blk.ln1(h)).split(a.d_model, dim=2)
            qq = qq.view(*qq.shape[:2], cfg.n_head, a.head_dim)
            kk = kk.view(*kk.shape[:2], cfg.n_head, a.head_dim)
            if key != "none":
                kk = blk.attn.k_op(kk)
            init_std = initial_logit_std(qq, kk, a.head_dim)
        decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
        nodecay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd},
                                 {"params": nodecay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.95))
        warm = a.warmup if a.warmup is not None else max(1, a.steps // 10)
        rng = np.random.default_rng(seed)
        t0 = time.time()
        for it in range(a.steps):
            if it < warm:
                lr = a.lr * (it + 1) / warm
            else:
                prog = (it - warm) / max(1, a.steps - warm)
                lr = a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))
            for g in opt.param_groups:
                g["lr"] = lr
            x, y = get_batch(train_ids, rng, a.batch, a.block)
            loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        return {"val_bpc": round(eval_bpc(model, val_ids, a.block), 5), "n_params": n_params,
                "init_logit_std": round(init_std, 4), "train_seconds": round(time.time() - t0, 1),
                "model": model}

    if a.sweep_c:
        print(f"sweeping c over {a.sweep_c} at head_dim {a.head_dim} "
              f"({cfg.n_head} heads, d_model {a.d_model}), {a.steps} steps, seed {a.seed}\n")
        rows = []
        for c in a.sweep_c:
            r = build_and_train("kscale", c, a.seed)
            r.pop("model")
            r["c"] = c
            rows.append(r)
            print(f"  c = {c:<6g} val bpc {r['val_bpc']:.4f}   initial logit std {r['init_logit_std']:.3f}"
                  f"   ({r['train_seconds']:.0f}s)", flush=True)
        best = min(rows, key=lambda r: r["val_bpc"])
        ref = next((r for r in rows if r["c"] == 1.0), None)
        print(f"\n  best c = {best['c']:g}  at {best['val_bpc']:.4f} bpc" +
              (f", {ref['val_bpc'] - best['val_bpc']:+.4f} better than the default c = 1" if ref else ""))
        print(f"  use it:  from attnscale import KeyScale;  k = KeyScale({cfg.n_head}, c={best['c']:g})(k)")
        if len(a.sweep_c) > 2 and best["c"] in (min(a.sweep_c), max(a.sweep_c)):
            print("  NOTE the best value is at the edge of the sweep; widen the range.")
        out = {"sweep": rows, "best_c": best["c"], "head_dim": a.head_dim, "n_head": cfg.n_head,
               "d_model": a.d_model, "steps": a.steps, "seed": a.seed, "vocab": vocab}
        if a.json:
            Path(a.json).write_text(json.dumps(out, indent=1))
        return

    r = build_and_train(a.key, a.c, a.seed)
    model = r.pop("model")
    res = {"key": a.key, "c": a.c if a.key == "kscale" else None, "head_dim": a.head_dim,
           "n_head": cfg.n_head, "d_model": a.d_model, "n_layer": a.n_layer, "steps": a.steps,
           "seed": a.seed, "vocab": vocab, **r}
    if a.key == "fkn":
        res["alpha_per_layer"] = [[round(float(v), 4) for v in b.attn.k_op.alpha] for b in model.blocks]
    print(json.dumps(res, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
