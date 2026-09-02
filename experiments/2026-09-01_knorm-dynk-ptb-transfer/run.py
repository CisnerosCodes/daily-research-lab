"""Does the key-side magnitude channel transfer to a second corpus (character-level Penn Treebank)?

Lineage:
  2026-07-30_qknorm-head-dim      QK-norm flips the head-split curve to a U, +0.143 cliff at hd=4.
  2026-08-11_composite-sweep      qknorm + undetached dynq closes the cliff but inherits the
                                  nh=1 tax (+0.037 at hd=128): not strictly better.
  2026-08-30_knorm-only-head-sweep  the cliff follows the KEY norm; k-only wins hd>=64,
                                  qknorm wins hd 16-32: no strict winner.
  2026-08-31_hd4-kside-cliff      undetached dynk on k-norm-only recovers 155% of the hd=4
                                  cliff, -0.070 bpc BELOW baseline (3/3 seeds).

Tonight: six arms x head_dim {4,16,32,64,128} x 3 byte-identical paired seeds:

  baseline      no norm                                     (anchor 08-30)
  qknorm        RMS-norm + gain on q and k                  (anchor 08-30)
  qnorm_only    RMS-norm + gain on q                        (anchor 08-30)
  knorm_only    RMS-norm + gain on k                        (anchor 08-30)
  knorm_dynk    knorm_only + undetached per-token key-magnitude channel   (anchor 08-31 @hd4)
  qknorm_dynq   qknorm + undetached per-token query-magnitude channel     (anchor 08-11, 2 seeds)

Predictions registered up front (see experiment.yaml): P1 knorm_dynk within tolerance of
the best arm at every head width; P2 k-alpha disengages with width; P3 the two channels are
not interchangeable (dynk > dynq at hd=4 and hd=128).

Deterministic, CPU-only.  Usage:
  python run.py                                  # full grid, single process
  python run.py --seeds 0 --tag s0               # shard: writes results_part_s0.json
  python run.py --merge                          # merge results_part_*.json -> results.json + chart.png
  SMOKE=1 python run.py                          # quick smoke test
"""
import argparse, glob, json, math, os, random, subprocess, sys, time
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
    ds = cfg["dataset"]
    paths = {}
    for split in ("train", "valid"):
        fp = HERE / "data" / f"ptb.{split}.txt"
        if not fp.exists():   # data/ is gitignored; fetch on a fresh clone
            fp.parent.mkdir(parents=True, exist_ok=True)
            import urllib.request
            urllib.request.urlretrieve(ds[f"source_{split}"], fp)
        paths[split] = fp.read_text()
    chars = sorted(set(paths["train"]) | set(paths["valid"]))
    vocab = len(chars)
    assert vocab == p["char_vocab_size"], f"expected {p['char_vocab_size']} chars, got {vocab}"
    stoi = {c: i for i, c in enumerate(chars)}
    train_ids = np.array([stoi[c] for c in paths["train"]], dtype=np.int64)
    val_ids = np.array([stoi[c] for c in paths["valid"]], dtype=np.int64)
    return train_ids, val_ids, vocab


# ---------------------------------------------------------------------------- model
class Block(nn.Module):
    """Pre-norm transformer block, byte-compatible with 2026-07-30 ... 2026-08-31.

    Per side s in {q, k}:
      norm_s : per-head RMS-norm (eps 1e-6) + learnable per-channel gain of length d
      dyn_s  : "undetached" -> multiply the normalised vector by
               tau_s = clamp(r_s / EMA(r_s), 1/c, c) ** alpha_s  (per token, per head)
               with r_s the PRE-norm per-token per-head RMS keeping its gradient,
               alpha_s a learnable per-head exponent (init 1), EMA a no-grad running mean.
               "alpha_fixed1" -> same channel with alpha frozen at 1: the forward pass is
               (k / r) * (r / EMA) * gain = k * gain / EMA_head(r) whenever the clamp is
               inactive, i.e. the per-token norm is undone and only a per-head running
               SCALE remains (the ablation that asks whether the learnable exponent matters).
    Ordering (to replicate the parents bit-exactly):
      q side (08-11): q = rms(q) * gain ; q = q * tau_q
      k side (08-31): k = rms(k) * tau_k ; k = k * gain
    All extra parameters/buffers init deterministically (ones): paired inits by construction.
    """

    def __init__(self, d, n_head, d_ff, norm_q, norm_k, dyn_q, dyn_k,
                 ema_momentum: float, ratio_clamp: float):
        super().__init__()
        assert d % n_head == 0
        assert dyn_q in (None, "undetached", "alpha_fixed1")
        assert dyn_k in (None, "undetached", "alpha_fixed1")
        assert (dyn_q is None or norm_q) and (dyn_k is None or norm_k), "dyn sits on a norm"
        self.n_head, self.head_dim = n_head, d // n_head
        self.norm_q, self.norm_k, self.dyn_q, self.dyn_k = norm_q, norm_k, dyn_q, dyn_k
        self.ema_momentum, self.ratio_clamp = ema_momentum, ratio_clamp
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.fc = nn.Linear(d, d_ff, bias=False)
        self.out = nn.Linear(d_ff, d, bias=False)
        # init to ones: consumes no RNG, keeps shared weights bit-identical across arms
        if norm_q:
            self.q_gain = nn.Parameter(torch.ones(d))
        if norm_k:
            self.k_gain = nn.Parameter(torch.ones(d))
        if dyn_q:
            self.q_alpha = nn.Parameter(torch.ones(n_head), requires_grad=(dyn_q == "undetached"))
            self.register_buffer("q_rms_ema", torch.ones(n_head))
            self.register_buffer("q_ema_ready", torch.zeros(1))
        if dyn_k:
            self.k_alpha = nn.Parameter(torch.ones(n_head), requires_grad=(dyn_k == "undetached"))
            self.register_buffer("k_rms_ema", torch.ones(n_head))
            self.register_buffer("k_ema_ready", torch.zeros(1))

    def _rms(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

    def _update_ema(self, ema, ready, r_det):
        if self.training:
            with torch.no_grad():
                batch_mean = r_det.mean(dim=(0, 1))
                if float(ready) == 0.0:
                    ema.copy_(batch_mean)
                    ready.fill_(1.0)
                else:
                    ema.mul_(self.ema_momentum).add_(batch_mean, alpha=1 - self.ema_momentum)

    def _tau(self, r_grad, ema, alpha):
        c = self.ratio_clamp
        ratio = (r_grad / ema.view(1, 1, -1)).clamp(1.0 / c, c)
        return torch.exp(alpha.view(1, 1, -1) * ratio.log())   # undetached: grad through r

    def _qkv(self, x, return_stats=False):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        shape4 = (B, T, self.n_head, self.head_dim)
        q4, k4 = q.view(*shape4), k.view(*shape4)
        rq_grad = q4.pow(2).mean(-1).add(1e-12).sqrt()   # (B, T, nh) pre-norm RMS, WITH grad
        rk_grad = k4.pow(2).mean(-1).add(1e-12).sqrt()
        rq_det, rk_det = rq_grad.detach(), rk_grad.detach()
        tau_q = tau_k = None
        # ---- query side (08-11 ordering: gain, then tau)
        if self.norm_q:
            q = self._rms(q4).reshape(B, T, D) * self.q_gain
        q = q.view(*shape4)
        if self.dyn_q:
            self._update_ema(self.q_rms_ema, self.q_ema_ready, rq_det)
            tau_q = self._tau(rq_grad, self.q_rms_ema, self.q_alpha)
            q = q * tau_q.unsqueeze(-1)
        # ---- key side (08-31 ordering: tau, then gain)
        if self.norm_k:
            k4 = self._rms(k4)
            if self.dyn_k:
                self._update_ema(self.k_rms_ema, self.k_ema_ready, rk_det)
                tau_k = self._tau(rk_grad, self.k_rms_ema, self.k_alpha)
                k4 = k4 * tau_k.unsqueeze(-1)
            k = k4.reshape(B, T, D) * self.k_gain
        k = k.view(*shape4)
        out = (q.transpose(1, 2), k.transpose(1, 2), v.view(*shape4).transpose(1, 2))
        if return_stats:
            return out, {"r_q": rq_det, "r_k": rk_det,
                         "tau_q": None if tau_q is None else tau_q.detach(),
                         "tau_k": None if tau_k is None else tau_k.detach()}
        return out

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self._qkv(x)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, D))
        x = x + self.out(F.gelu(self.fc(self.ln2(x))))
        return x

    def attn_logits(self, x, with_stats=False):
        """Explicit pre-softmax causal logits, (B, n_head, T, T). Probe only."""
        (q, k, _), stats = self._qkv(x, return_stats=True)
        logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if with_stats:
            return logits, stats
        return logits


class GPT(nn.Module):
    def __init__(self, vocab, d, n_layer, n_head, d_ff, block_size, arm_flags,
                 ema_momentum, ratio_clamp):
        super().__init__()
        norm_q, norm_k, dyn_q, dyn_k = arm_flags
        self.vocab, self.block_size = vocab, block_size
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(block_size, d)
        self.blocks = nn.ModuleList([
            Block(d, n_head, d_ff, norm_q, norm_k, dyn_q, dyn_k, ema_momentum, ratio_clamp)
            for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.apply(self._init)
        for name, prm in self.named_parameters():   # nanoGPT residual-projection scaling
            if name.endswith("proj.weight") or name.endswith("out.weight"):
                nn.init.normal_(prm, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))
        with torch.no_grad():   # _init must not disturb the deterministic extras
            for blk in self.blocks:
                for nm in ("q_gain", "k_gain", "q_alpha", "k_alpha"):
                    if hasattr(blk, nm):
                        getattr(blk, nm).fill_(1.0)

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
    """Trained-model attention statistics on fixed val batches (as in the parents)."""
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
        nh = blk.n_head
        H_sum, top_sum = torch.zeros(nh), torch.zeros(nh)
        lg_mu, lg_sq, lg_n = torch.zeros(nh), torch.zeros(nh), 0
        acc = {s: [torch.zeros(nh), torch.zeros(nh)] for s in ("r_q", "r_k")}
        tau = {s: [torch.zeros(nh), torch.zeros(nh), 0] for s in ("tau_q", "tau_k")}
        r_n, nb = 0, 0
        for s0 in range(0, n_blocks, B):
            xb = torch.from_numpy(xs[s0:s0 + B])
            h = model.embed(xb)
            for j in range(li):
                h = model.blocks[j](h)
            logits, stats = blk.attn_logits(h, with_stats=True)   # (B, nh, T, T)
            probs = logits.masked_fill(~causal, float("-inf")).softmax(-1)
            ent = -(probs * torch.log(probs.clamp(min=1e-12))).sum(-1)   # (B, nh, T)
            H_sum += (ent / norm)[:, :, keep].mean(dim=(0, 2))
            top_sum += probs.max(-1).values[:, :, keep].mean(dim=(0, 2))
            lv = logits.masked_select(causal.view(1, 1, bs, bs)).view(logits.shape[0], nh, -1)
            lg_mu += lv.mean(dim=(0, 2)) * lv.shape[0]
            lg_sq += lv.pow(2).mean(dim=(0, 2)) * lv.shape[0]
            lg_n += lv.shape[0]
            for key in acc:
                v = stats[key]
                acc[key][0] += v.mean(dim=(0, 1)) * v.shape[0]
                acc[key][1] += v.pow(2).mean(dim=(0, 1)) * v.shape[0]
            for key in tau:
                if stats.get(key) is not None:
                    tk = stats[key]
                    tau[key][0] += tk.mean(dim=(0, 1)) * tk.shape[0]
                    tau[key][1] += tk.pow(2).mean(dim=(0, 1)) * tk.shape[0]
                    tau[key][2] += tk.shape[0]
            r_n += stats["r_q"].shape[0]
            nb += 1
        mu = lg_mu / lg_n
        std = (lg_sq / lg_n - mu ** 2).clamp(min=0).sqrt()

        def cv(key):
            m_ = acc[key][0] / r_n
            s_ = (acc[key][1] / r_n - m_ ** 2).clamp(min=0).sqrt()
            return s_ / m_.clamp(min=1e-9)

        row = {
            "layer": li,
            "entropy_norm_per_head": [round(float(u), 4) for u in (H_sum / nb)],
            "top1_weight_per_head": [round(float(u), 4) for u in (top_sum / nb)],
            "logit_std_per_head": [round(float(u), 4) for u in std],
            "q_rms_token_cv_per_head": [round(float(u), 4) for u in cv("r_q")],
            "k_rms_token_cv_per_head": [round(float(u), 4) for u in cv("r_k")],
        }
        for key, side in (("tau_q", "q"), ("tau_k", "k")):
            if tau[key][2] > 0:
                tm = tau[key][0] / tau[key][2]
                ts = (tau[key][1] / tau[key][2] - tm ** 2).clamp(min=0).sqrt()
                row[f"tau_{side}_mean_per_head"] = [round(float(u), 4) for u in tm]
                row[f"tau_{side}_std_per_head"] = [round(float(u), 4) for u in ts]
                row[f"{side}_alpha_per_head"] = [round(float(u), 4)
                                                 for u in getattr(blk, f"{side}_alpha")]
        per_layer.append(row)
    model.train()

    def flat(key):
        return [u for L in per_layer for u in L.get(key, [])]

    summ = {
        "per_layer": per_layer,
        "entropy_norm_mean": round(float(np.mean(flat("entropy_norm_per_head"))), 4),
        "top1_weight_mean": round(float(np.mean(flat("top1_weight_per_head"))), 4),
        "logit_std_mean": round(float(np.mean(flat("logit_std_per_head"))), 4),
        "q_rms_token_cv_mean": round(float(np.mean(flat("q_rms_token_cv_per_head"))), 4),
        "k_rms_token_cv_mean": round(float(np.mean(flat("k_rms_token_cv_per_head"))), 4),
    }
    for side in ("q", "k"):
        if flat(f"tau_{side}_mean_per_head"):
            summ[f"tau_{side}_mean"] = round(float(np.mean(flat(f"tau_{side}_mean_per_head"))), 4)
            summ[f"tau_{side}_std_mean"] = round(float(np.mean(flat(f"tau_{side}_std_per_head"))), 4)
            summ[f"{side}_alpha_mean"] = round(float(np.mean(flat(f"{side}_alpha_per_head"))), 4)
    return summ


def train_one(vocab, arm, arm_flags, head_dim, seed, p, train_ids, val_ids):
    set_seeds(seed)                    # identical init across ALL arms at a seed
    n_head = p["d_model"] // head_dim
    model = GPT(vocab, p["d_model"], p["n_layer"], n_head,
                p["d_ff_mult"] * p["d_model"], p["block_size"], arm_flags,
                p["ema_momentum"], p["ratio_clamp"])
    shared_sig = float(sum(float(q.detach().double().abs().sum())
                           for name, q in model.named_parameters()
                           if "gain" not in name and "alpha" not in name))
    decay = [q for q in model.parameters() if q.requires_grad and q.dim() >= 2]
    nodecay = [q for q in model.parameters() if q.requires_grad and q.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": p["weight_decay"]},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=p["lr"], betas=(0.9, 0.95),
    )
    rng = np.random.default_rng(seed)  # identical batch stream across ALL arms
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
    rec = {
        "arm": arm, "seed": int(seed), "n_head": int(n_head), "head_dim": int(head_dim),
        "n_params": n_params(model),
        "shared_init_signature": round(shared_sig, 6),
        "train_seconds": round(train_s, 1),
        "final_train_loss_ma50": round(float(np.mean(losses[-50:])), 4),
        "val_bpc": round(ev["bpc"], 5),
        "eval_chars": ev["eval_chars"],
        "attention": probe,
    }
    extra = ""
    for side in ("q", "k"):
        if f"tau_{side}_mean" in probe:
            extra += (f" tau{side.upper()}={probe[f'tau_{side}_mean']:.3f}"
                      f"±{probe[f'tau_{side}_std_mean']:.3f} a{side.upper()}={probe[f'{side}_alpha_mean']:.3f}")
    print(f"  [hd={head_dim:3d} {arm:12s} seed={seed}] P={rec['n_params']} bpc={rec['val_bpc']:.4f} "
          f"logit_std={probe['logit_std_mean']:.3f} entN={probe['entropy_norm_mean']:.3f} "
          f"top1={probe['top1_weight_mean']:.3f}{extra} ({rec['train_seconds']}s)", flush=True)
    return rec


# -------------------------------------------------------------------------- analysis
def summarize(runs, p):
    arms, hds, tol = p["arm_order"], p["head_dims"], p["tolerance_bpc"]
    by = {}
    for arm in arms:
        for hd in hds:
            rs = [r for r in runs if r["arm"] == arm and r["head_dim"] == hd]
            if not rs:
                continue
            per_seed = {int(r["seed"]): r["val_bpc"] for r in rs}
            b = np.array(list(per_seed.values()))
            probes = {k: round(float(np.mean([r["attention"][k] for r in rs])), 4)
                      for k in ("entropy_norm_mean", "top1_weight_mean", "logit_std_mean",
                                "q_rms_token_cv_mean", "k_rms_token_cv_mean")}
            row = {"val_bpc_per_seed": per_seed, "val_bpc_mean": round(float(b.mean()), 5),
                   "seed_spread_bpc": round(float(b.max() - b.min()), 5),
                   "n_params": rs[0]["n_params"], **probes}
            for side in ("q", "k"):
                if all(f"{side}_alpha_mean" in r["attention"] for r in rs):
                    row[f"{side}_alpha_mean"] = round(float(np.mean(
                        [r["attention"][f"{side}_alpha_mean"] for r in rs])), 4)
                    row[f"tau_{side}_std_mean"] = round(float(np.mean(
                        [r["attention"][f"tau_{side}_std_mean"] for r in rs])), 4)
            by.setdefault(arm, {})[hd] = row

    def mean(arm, hd):
        return by.get(arm, {}).get(hd, {}).get("val_bpc_mean")

    def seeds(arm, hd):
        return by.get(arm, {}).get(hd, {}).get("val_bpc_per_seed", {})

    # per-hd table of deltas and paired-seed wins
    per_hd = {}
    for hd in hds:
        present = [a for a in arms if hd in by.get(a, {})]
        if not present:
            continue
        best_arm = min(present, key=lambda a: mean(a, hd))
        row = {"best_arm": best_arm, "best_bpc": mean(best_arm, hd), "arms": {}}
        for a in present:
            d_base = round(mean(a, hd) - mean("baseline", hd), 5) if "baseline" in present else None
            d_qk = round(mean(a, hd) - mean("qknorm", hd), 5) if "qknorm" in present else None
            d_best = round(mean(a, hd) - mean(best_arm, hd), 5)
            sb, sq = seeds(a, hd), seeds("baseline", hd)
            wins_base = sum(sb[s] < sq[s] for s in sb if s in sq)
            sk = seeds("qknorm", hd)
            wins_qk = sum(sb[s] < sk[s] for s in sb if s in sk)
            sko = seeds("knorm_only", hd)
            wins_ko = sum(sb[s] < sko[s] for s in sb if s in sko)
            row["arms"][a] = {"bpc": mean(a, hd), "delta_vs_baseline": d_base,
                              "delta_vs_qknorm": d_qk, "delta_vs_best": d_best,
                              "within_tol_of_best": bool(d_best <= tol),
                              "beats_baseline_seeds": f"{wins_base}/{len(sq)}" if sq else None,
                              "beats_qknorm_seeds": f"{wins_qk}/{len(sk)}" if sk else None,
                              "beats_knorm_only_seeds": f"{wins_ko}/{len(sko)}" if sko else None}
        per_hd[hd] = row

    # strictly-better verdicts: arm within tol of best at every hd, and beats both parents
    verdicts = {}
    for a in arms:
        if not all(hd in by.get(a, {}) for hd in hds):
            continue
        within_all = all(per_hd[hd]["arms"][a]["within_tol_of_best"] for hd in hds)
        beats_base_all = all(mean(a, hd) < mean("baseline", hd) for hd in hds
                             if "baseline" in by and hd in by["baseline"])
        beats_qk_all = all(mean(a, hd) < mean("qknorm", hd) for hd in hds
                           if "qknorm" in by and hd in by["qknorm"])
        verdicts[a] = {"within_tol_of_best_everywhere": within_all,
                       "beats_baseline_mean_everywhere": beats_base_all,
                       "beats_qknorm_mean_everywhere": beats_qk_all,
                       "strictly_better_than_both_parents": bool(within_all and beats_base_all
                                                                 and beats_qk_all)}

    # P2: k-alpha vs head width
    alpha_k = {hd: by["knorm_dynk"][hd].get("k_alpha_mean") for hd in hds
               if "knorm_dynk" in by and hd in by["knorm_dynk"]}
    alpha_q = {hd: by["qknorm_dynq"][hd].get("q_alpha_mean") for hd in hds
               if "qknorm_dynq" in by and hd in by["qknorm_dynq"]}
    # P3: dynk vs dynq head-to-head per hd
    p3 = {}
    for hd in hds:
        if mean("knorm_dynk", hd) is not None and mean("qknorm_dynq", hd) is not None:
            sk_, sq_ = seeds("knorm_dynk", hd), seeds("qknorm_dynq", hd)
            p3[hd] = {"dynk_minus_dynq": round(mean("knorm_dynk", hd) - mean("qknorm_dynq", hd), 5),
                      "dynk_beats_dynq_seeds": f"{sum(sk_[s] < sq_[s] for s in sk_ if s in sq_)}/{len(sq_)}"}

    # replication vs parents (only cells run tonight are compared)
    rep = {"tol": p["rep_tol_bpc"], "pairs": [], "n_ok": 0, "n_checked": 0, "ok": True}
    for src, table in p["anchors"].items():
        for arm, per_hd_tab in table.items():
            for hd, per_seed in per_hd_tab.items():
                for s, expect in per_seed.items():
                    got = seeds(arm, int(hd)).get(int(s))
                    if got is None:
                        continue
                    d = round(got - expect, 5)
                    ok = abs(d) <= p["rep_tol_bpc"]
                    rep["pairs"].append({"source": src, "arm": arm, "head_dim": int(hd),
                                         "seed": int(s), "tonight": got, "expected": expect,
                                         "delta": d, "ok": ok})
                    rep["n_checked"] += 1
                    rep["n_ok"] += int(ok)
                    rep["ok"] = rep["ok"] and ok

    global_best = min(((a, hd) for a in by for hd in by[a]), key=lambda t: mean(*t))
    return {"by_arm_hd": by, "per_head_dim": per_hd, "verdicts": verdicts,
            "P2_alpha_vs_head_dim": {"k_alpha_knorm_dynk": alpha_k, "q_alpha_qknorm_dynq": alpha_q},
            "P3_dynk_vs_dynq": p3,
            "global_best": {"arm": global_best[0], "head_dim": global_best[1],
                            "bpc": mean(*global_best)},
            "tolerance_bpc": tol, "replication_vs_parents": rep}


def make_chart(m, p, headline, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms, hds = p["arm_order"], p["head_dims"]
    labels = {"baseline": "baseline (no norm)", "qknorm": "QK-norm", "qnorm_only": "q-norm only",
              "knorm_only": "k-norm only", "knorm_dynk": "k-norm + dynk (proposed)",
              "qknorm_dynq": "QK-norm + dynq (08-11)", "k_emascale": "k / EMA-scale (alpha fixed 1)"}
    colors = {"baseline": "#444444", "qknorm": "#8e44ad", "qnorm_only": "#3498db",
              "knorm_only": "#27ae60", "knorm_dynk": "#c0392b", "qknorm_dynq": "#e67e22",
              "k_emascale": "#7f8c8d"}
    styles = {"baseline": "--", "qknorm": "-", "qnorm_only": ":", "knorm_only": "-",
              "knorm_dynk": "-", "qknorm_dynq": "-.", "k_emascale": ":"}
    by = m["by_arm_hd"]
    x = np.arange(len(hds))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    ax = axes[0]
    for a in arms:
        if a not in by:
            continue
        ys = [by[a][hd]["val_bpc_mean"] if hd in by[a] else np.nan for hd in hds]
        ax.plot(x, ys, styles[a], color=colors[a], lw=2.2 if a == "knorm_dynk" else 1.5,
                marker="o", ms=5, label=labels[a])
        for i, hd in enumerate(hds):
            if hd in by[a]:
                ss = list(by[a][hd]["val_bpc_per_seed"].values())
                ax.plot([i] * len(ss), ss, ".", color=colors[a], ms=4, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"hd={hd}\nnh={p['d_model'] // hd}" for hd in hds])
    ax.set_ylabel("val bits per character, char-PTB (600 steps)")
    ax.set_title("six norm arms across the iso-parameter head split", fontsize=10)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25)

    ax = axes[1]
    w = 0.13
    darms = [a for a in arms if a != "baseline" and a in by]
    for j, a in enumerate(darms):
        ys = [m["per_head_dim"][hd]["arms"][a]["delta_vs_baseline"]
              if hd in m["per_head_dim"] and a in m["per_head_dim"][hd]["arms"] else np.nan
              for hd in hds]
        ax.bar(x + (j - (len(darms) - 1) / 2) * w, ys, w, color=colors[a], label=labels[a])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"hd={hd}" for hd in hds])
    ax.set_ylabel("Δ val bpc vs baseline (negative = better)")
    ax.set_title("where each norm helps or hurts", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[2]
    ak = m["P2_alpha_vs_head_dim"]["k_alpha_knorm_dynk"]
    aq = m["P2_alpha_vs_head_dim"]["q_alpha_qknorm_dynq"]
    if ak:
        ax.plot([hds.index(hd) for hd in ak], [ak[hd] for hd in ak], "-o", color=colors["knorm_dynk"],
                label="k-alpha (k-norm + dynk)")
    if aq:
        ax.plot([hds.index(hd) for hd in aq], [aq[hd] for hd in aq], "-.s", color=colors["qknorm_dynq"],
                label="q-alpha (QK-norm + dynq)")
    ax.axhline(1.0, color="k", lw=0.8, ls="--", label="init (alpha = 1)")
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"hd={hd}" for hd in hds])
    ax.set_ylabel("learned exponent alpha (mean over heads, layers, seeds)")
    ax.set_title("does the magnitude channel disengage with head width?", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.suptitle(headline, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def headline_from(m, p):
    v = m["verdicts"].get("knorm_dynk", {})
    best = m["global_best"]
    parts = []
    for hd in p["head_dims"]:
        r = m["per_head_dim"].get(hd)
        if r and "knorm_dynk" in r["arms"]:
            parts.append(f"hd{hd}:{r['arms']['knorm_dynk']['delta_vs_baseline']:+.3f}")
    rep = m["replication_vs_parents"]
    return (f"knorm_dynk vs baseline [{' '.join(parts)}]; strictly better than both parents: "
            f"{v.get('strictly_better_than_both_parents')}; global best {best['arm']}@hd{best['head_dim']} "
            f"= {best['bpc']:.4f}; replication {rep['n_ok']}/{rep['n_checked']} within {rep['tol']}")


# ------------------------------------------------------------------------------ main
def merge(cfg, p):
    runs, seen = [], set()
    for part in sorted(glob.glob(str(HERE / "results_part_*.json"))):
        for r in json.load(open(part))["runs"]:
            key = (r["arm"], r["head_dim"], r["seed"])
            if key not in seen:
                seen.add(key)
                runs.append(r)
    print(f"merged {len(runs)} runs from parts")
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--head-dims", type=int, nargs="*", default=None)
    ap.add_argument("--arms", type=str, nargs="*", default=None)
    ap.add_argument("--tag", type=str, default=None, help="write results_part_<tag>.json only")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    cfg = load_config()
    p = dict(cfg["params"])
    if SMOKE:
        p["steps"], p["warmup"], p["max_eval_blocks"], p["entropy_batches"] = 40, 4, 32, 1
        p["seeds"], p["head_dims"] = [0], [4, 64]
    seeds = args.seeds if args.seeds is not None else p["seeds"]
    hds = args.head_dims if args.head_dims is not None else p["head_dims"]
    arms = args.arms if args.arms is not None else p["arm_order"]

    if args.merge:
        runs = merge(cfg, p)
    else:
        train_ids, val_ids, vocab = load_data(cfg)
        print(f"data: {len(train_ids)} train / {len(val_ids)} val chars, vocab={vocab}")
        runs, sigs = [], {}
        for hd in hds:
            for seed in seeds:
                for arm in arms:
                    rec = train_one(vocab, arm, p["arms"][arm], hd, seed, p, train_ids, val_ids)
                    runs.append(rec)
                    sigs.setdefault((hd, seed), set()).add(rec["shared_init_signature"])
        for key, ss in sigs.items():
            assert len(ss) == 1, f"shared-init mismatch at (hd, seed)={key}: {ss}"
        print("paired-init check passed: identical shared weights across all arms at every (hd, seed)")
        if args.tag:
            out = HERE / f"results_part_{args.tag}.json"
            json.dump({"runs": runs, "env": env_info(), "seconds": round(time.time() - t0, 1)},
                      open(out, "w"), indent=1)
            print(f"wrote {out} ({len(runs)} runs, {time.time() - t0:.0f}s)")
            return

    m = summarize(runs, p)
    headline = headline_from(m, p)
    print("\n" + headline)
    for hd in p["head_dims"]:
        r = m["per_head_dim"].get(hd)
        if not r:
            continue
        print(f"  hd={hd:3d} best={r['best_arm']:12s} " + "  ".join(
            f"{a}={r['arms'][a]['bpc']:.4f}({r['arms'][a]['delta_vs_baseline']:+.3f})"
            for a in p["arm_order"] if a in r["arms"]))
    rep = m["replication_vs_parents"]
    bad = [q for q in rep["pairs"] if not q["ok"]]
    print(f"replication vs parents: {rep['n_ok']}/{rep['n_checked']} ok" +
          (f"; MISMATCHES: {bad[:6]}" if bad else ""))

    results = {
        "id": cfg["id"], "git_commit": git_sha(), "seed": cfg["seed"],
        "duration_sec": round(time.time() - t0, 2), "smoke": SMOKE,
        "metrics": {"headline": headline, **m},
        "runs": runs, "env": env_info(), "config": cfg,
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=1)
    make_chart(m, p, headline, HERE / "chart.png")
    print(f"\nwrote results.json + chart.png in {results['duration_sec']}s")


if __name__ == "__main__":
    main()
