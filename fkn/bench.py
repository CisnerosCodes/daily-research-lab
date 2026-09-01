"""Benchmark FKN against no-norm / QK-norm / key-only norm on any text file, on a CPU.

This reproduces the daily-research-lab char-LM recipe *through the drop-in module*
(2-layer pre-norm GPT, d_model 128, block 96, batch 16, AdamW 3e-3 cosine, 600 steps),
so the anchor numbers are checkable:

    python -m fkn.bench --text data/tinyshakespeare.txt --norm none --head-dim 4 --seed 0   # 3.0930 bpc
    python -m fkn.bench --text data/tinyshakespeare.txt --norm fkn  --head-dim 4 --seed 0   # 3.0167 bpc

Then point it at YOUR corpus and YOUR head split:

    python -m fkn.bench --text my_corpus.txt --norm fkn --head-dim 32 --steps 2000 --json out.json

Prints val bits-per-character, seed, wall time and (for fkn) the learned per-head exponents.
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
from fkn import FKNCausalSelfAttention  # noqa: E402

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

    def __init__(self, cfg, norm, fkn_kwargs):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.attn = FKNCausalSelfAttention(cfg, norm=norm, fkn_kwargs=fkn_kwargs)
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.out = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, vocab, cfg, n_layer, norm, fkn_kwargs):
        super().__init__()
        self.vocab, self.block_size = vocab, cfg.block_size
        self.tok = nn.Embedding(vocab, cfg.n_embd)
        self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg, norm, fkn_kwargs) for _ in range(n_layer)])
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
    ap.add_argument("--norm", default="fkn", choices=["none", "qknorm", "knorm", "fkn"])
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
    set_seeds(a.seed)
    fkn_kwargs = {"alpha_init": a.alpha_init, "alpha_learnable": not a.alpha_frozen}
    model = GPT(vocab, cfg, a.n_layer, a.norm, fkn_kwargs)
    n_params = sum(p.numel() for p in model.parameters())
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd},
                             {"params": nodecay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.95))
    warm = a.warmup if a.warmup is not None else max(1, a.steps // 10)
    rng = np.random.default_rng(a.seed)
    t0 = time.time(); losses = []
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
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        losses.append(float(loss))
        if (it + 1) % max(1, a.steps // 5) == 0:
            print(f"  step {it + 1:5d}  train loss {np.mean(losses[-50:]):.4f}  ({time.time() - t0:.0f}s)", flush=True)
    bpc = eval_bpc(model, val_ids, a.block)
    res = {"norm": a.norm, "head_dim": a.head_dim, "n_head": cfg.n_head, "d_model": a.d_model,
           "n_layer": a.n_layer, "steps": a.steps, "seed": a.seed, "n_params": n_params,
           "vocab": vocab, "val_bpc": round(bpc, 5), "train_seconds": round(time.time() - t0, 1)}
    if a.norm == "fkn":
        res["alpha_per_layer"] = [[round(float(v), 4) for v in b.attn.k_norm.alpha] for b in model.blocks]
        res["rms_ema_per_layer"] = [[round(float(v), 4) for v in b.attn.k_norm.rms_ema] for b in model.blocks]
    print(json.dumps(res, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
