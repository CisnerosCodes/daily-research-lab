"""What does QK-norm break at a single head? Decomposing the nh=1 tax.

Lineage:
  2026-07-30_qknorm-head-dim: QK-norm flips the head-split curve into a U, but at nh=1
  (hd=128) it is a small reproducible TAX: +0.033 bpc over the no-norm baseline.
  2026-08-02/06: the hd=4 cliff is per-token QUERY magnitude (54% detached, 98% undetached).
  2026-08-11: the composite (qknorm + undetached dynq) fixes the small-head cliff but
  INHERITS the nh=1 tax (+0.037 vs baseline) and its alpha disengages (0.66) — the model
  itself says the query channel is not what is missing at a single head.

Tonight, seven arms on byte-identical paired inits and a shared batch stream, all at nh=1:

  baseline            no norm (replication anchor, 08-11 seeds 0/1)
  qknorm              per-head RMS-norm q,k + learnable per-channel gains (anchor)
  qnorm_only          norm + gain on q only                (which SIDE carries the tax?)
  knorm_only          norm + gain on k only
  qknorm_frozen_gain  q,k norm, gains frozen at 1          (is it gain geometry?)
  qknorm_dynk         qknorm + UNDETACHED per-token key temperature (08-06 machinery, k side)
  qknorm_dynqk        qknorm + undetached dynq AND dynk    (do both channels recover it?)

Readouts: which single component recovers the baseline's bpc; q-vs-k additivity;
rescue fraction of the key channel; probes (logit std, entropy, per-token q/k RMS CV,
learned alphas / applied tau spread).

Deterministic, CPU-only. Usage:  python run.py   (SMOKE=1 for a quick smoke test)
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
    """Pre-norm transformer block, byte-compatible with 2026-07-30/31/08-02/08-06/08-11.

    norm_q / norm_k: per-head RMS-norm on that side + learnable per-channel gain of
    length d (frozen at 1 when gains_learnable=False — same forward, no gradient).
    dyn_q / dyn_k ("undetached"): the normalised side is scaled per token/head by
    tau = clamp(r/EMA, 1/c, c)^alpha with r the PRE-norm RMS of that side KEEPING its
    gradient (EMA update no_grad) — the exact 08-06 winner machinery, applied per side.
    All extra parameters/buffers init deterministically (ones), so the RNG stream for
    the shared weights is identical across arms: paired inits by construction.
    """

    def __init__(self, d, n_head, d_ff, norm_q, norm_k, gains_learnable, dyn_q, dyn_k,
                 ema_momentum: float, ratio_clamp: float):
        super().__init__()
        assert d % n_head == 0
        assert dyn_q in (None, "undetached") and dyn_k in (None, "undetached")
        assert (dyn_q is None or norm_q) and (dyn_k is None or norm_k), \
            "dynamic arms are defined on top of the corresponding norm"
        self.n_head = n_head
        self.head_dim = d // n_head
        self.norm_q, self.norm_k = norm_q, norm_k
        self.dyn_q, self.dyn_k = dyn_q, dyn_k
        self.ema_momentum = ema_momentum
        self.ratio_clamp = ratio_clamp
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.fc = nn.Linear(d, d_ff, bias=False)
        self.out = nn.Linear(d_ff, d, bias=False)
        # init to ones: consumes no RNG, keeps shared weights bit-identical across arms
        if norm_q:
            self.q_gain = nn.Parameter(torch.ones(d), requires_grad=gains_learnable)
        if norm_k:
            self.k_gain = nn.Parameter(torch.ones(d), requires_grad=gains_learnable)
        if dyn_q is not None:
            self.q_alpha = nn.Parameter(torch.ones(n_head))
            self.register_buffer("q_rms_ema", torch.ones(n_head))
            self.register_buffer("q_ema_ready", torch.zeros(1))
        if dyn_k is not None:
            self.k_alpha = nn.Parameter(torch.ones(n_head))
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

    def _dyn_tau(self, r_grad, r_det, ema, alpha):
        c = self.ratio_clamp
        ratio = (r_grad / ema.view(1, 1, -1)).clamp(1.0 / c, c)
        return torch.exp(alpha.view(1, 1, -1) * ratio.log())

    def _qkv(self, x, return_stats=False):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        shape4 = (B, T, self.n_head, self.head_dim)
        q4, k4 = q.view(*shape4), k.view(*shape4)
        rq_grad = q4.pow(2).mean(-1).add(1e-12).sqrt()          # (B, T, nh) pre-norm RMS, WITH grad
        rk_grad = k4.pow(2).mean(-1).add(1e-12).sqrt()
        rq_det, rk_det = rq_grad.detach(), rk_grad.detach()
        if self.norm_q:
            q = self._rms(q4).reshape(B, T, D) * self.q_gain
        if self.norm_k:
            k = self._rms(k4).reshape(B, T, D) * self.k_gain
        q = q.view(*shape4)
        k = k.view(*shape4)
        tau_q = tau_k = None
        if self.dyn_q is not None:
            self._update_ema(self.q_rms_ema, self.q_ema_ready, rq_det)
            tau_q = self._dyn_tau(rq_grad, rq_det, self.q_rms_ema, self.q_alpha)
            q = q * tau_q.unsqueeze(-1)
        if self.dyn_k is not None:
            self._update_ema(self.k_rms_ema, self.k_ema_ready, rk_det)
            tau_k = self._dyn_tau(rk_grad, rk_det, self.k_rms_ema, self.k_alpha)
            k = k * tau_k.unsqueeze(-1)
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
        norm_q, norm_k, gains_learnable, dyn_q, dyn_k = arm_flags
        self.vocab, self.block_size = vocab, block_size
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(block_size, d)
        self.blocks = nn.ModuleList([
            Block(d, n_head, d_ff, norm_q, norm_k, gains_learnable, dyn_q, dyn_k,
                  ema_momentum, ratio_clamp)
            for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.apply(self._init)
        for name, prm in self.named_parameters():   # nanoGPT residual-projection scaling
            if name.endswith("proj.weight") or name.endswith("out.weight"):
                nn.init.normal_(prm, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))
        with torch.no_grad():   # _init must not disturb the deterministic extras
            for blk in self.blocks:
                if norm_q:
                    blk.q_gain.fill_(1.0)
                if norm_k:
                    blk.k_gain.fill_(1.0)
                if dyn_q is not None:
                    blk.q_alpha.fill_(1.0)
                if dyn_k is not None:
                    blk.k_alpha.fill_(1.0)

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
    """Trained-model attention statistics on fixed val batches (as in the parents), plus
    per-token q AND k modulation stats: RMS CVs, realised tau spreads, learned alphas."""
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
        acc = {s: [torch.zeros(nh), torch.zeros(nh)] for s in ("r_q", "r_k", "tau_q", "tau_k")}
        seen = {s: False for s in acc}
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
                if v is not None:
                    acc[key][0] += v.mean(dim=(0, 1)) * v.shape[0]
                    acc[key][1] += v.pow(2).mean(dim=(0, 1)) * v.shape[0]
                    seen[key] = True
            r_n += stats["r_q"].shape[0]
            nb += 1
        mu = lg_mu / lg_n
        std = (lg_sq / lg_n - mu ** 2).clamp(min=0).sqrt()

        def cv(key):
            m_ = acc[key][0] / r_n
            s_ = (acc[key][1] / r_n - m_ ** 2).clamp(min=0).sqrt()
            return m_, s_, s_ / m_.clamp(min=1e-9)

        rq_m, _, rq_cv = cv("r_q")
        rk_m, _, rk_cv = cv("r_k")
        entry = {
            "layer": li,
            "entropy_norm_per_head": [round(float(u), 4) for u in (H_sum / nb)],
            "top1_weight_per_head": [round(float(u), 4) for u in (top_sum / nb)],
            "logit_std_per_head": [round(float(u), 4) for u in std],
            "q_rms_token_cv_per_head": [round(float(u), 4) for u in rq_cv],
            "k_rms_token_cv_per_head": [round(float(u), 4) for u in rk_cv],
        }
        for side, key, alpha_name in (("q", "tau_q", "q_alpha"), ("k", "tau_k", "k_alpha")):
            if seen[key]:
                tm, ts, _ = cv(key)
                entry[f"tau_{side}_token_mean_per_head"] = [round(float(u), 4) for u in tm]
                entry[f"tau_{side}_token_std_per_head"] = [round(float(u), 4) for u in ts]
                entry[f"alpha_{side}_per_head"] = [round(float(u), 4)
                                                  for u in getattr(blk, alpha_name)]
        per_layer.append(entry)
    model.train()

    def flat(key):
        return [u for L in per_layer for u in L.get(key, [])]

    out = {
        "per_layer": per_layer,
        "entropy_norm_mean": round(float(np.mean(flat("entropy_norm_per_head"))), 4),
        "top1_weight_mean": round(float(np.mean(flat("top1_weight_per_head"))), 4),
        "logit_std_mean": round(float(np.mean(flat("logit_std_per_head"))), 4),
        "q_rms_token_cv_mean": round(float(np.mean(flat("q_rms_token_cv_per_head"))), 4),
        "k_rms_token_cv_mean": round(float(np.mean(flat("k_rms_token_cv_per_head"))), 4),
    }
    for side in ("q", "k"):
        aa = flat(f"alpha_{side}_per_head")
        if aa:
            out[f"alpha_{side}_mean"] = round(float(np.mean(aa)), 4)
            out[f"alpha_{side}_min"] = round(float(np.min(aa)), 4)
            out[f"alpha_{side}_max"] = round(float(np.max(aa)), 4)
        ts = flat(f"tau_{side}_token_std_per_head")
        if ts:
            out[f"tau_{side}_token_std_mean"] = round(float(np.mean(ts)), 4)
    return out


def train_one(vocab, arm, arm_flags, seed, p, train_ids, val_ids):
    n_head = p["n_head"]
    set_seeds(seed)                    # identical init across ALL arms at a seed
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
    curve = [round(float(np.mean(losses[i:i + 25])), 4) for i in range(0, steps, 25)]
    rec = {
        "arm": arm, "seed": int(seed), "n_head": int(n_head),
        "head_dim": int(p["d_model"] // n_head),
        "n_params": n_params(model),
        "shared_init_signature": round(shared_sig, 6),
        "train_seconds": round(train_s, 1),
        "final_train_loss_ma50": round(float(np.mean(losses[-50:])), 4),
        "train_loss_curve_ma25": curve,
        "val_bpc": round(ev["bpc"], 5),
        "eval_chars": ev["eval_chars"],
        "attention": probe,
    }
    extra = ""
    for side in ("q", "k"):
        if f"alpha_{side}_mean" in probe:
            extra += (f" a_{side}={probe[f'alpha_{side}_mean']:.2f}"
                      f" tauStd_{side}={probe.get(f'tau_{side}_token_std_mean', float('nan')):.3f}")
    print(f"  [{arm:18s} seed={seed}] P={rec['n_params']} bpc={rec['val_bpc']:.4f} "
          f"logit_std={probe['logit_std_mean']:.3f} entN={probe['entropy_norm_mean']:.3f} "
          f"top1={probe['top1_weight_mean']:.3f} qCV={probe['q_rms_token_cv_mean']:.3f} "
          f"kCV={probe['k_rms_token_cv_mean']:.3f}{extra} ({rec['train_seconds']}s)", flush=True)
    return rec


# -------------------------------------------------------------------------- analysis
def summarize(runs, p):
    arms = p["arm_order"]
    by_arm = {}
    for arm in arms:
        rs = [r for r in runs if r["arm"] == arm]
        b = np.array([r["val_bpc"] for r in rs])
        row = {
            "val_bpc_per_seed": {int(r["seed"]): r["val_bpc"] for r in rs},
            "val_bpc_mean": round(float(b.mean()), 5),
            "seed_spread_bpc": round(float(b.max() - b.min()), 5),
            "entropy_norm_mean": round(float(np.mean([r["attention"]["entropy_norm_mean"] for r in rs])), 4),
            "top1_weight_mean": round(float(np.mean([r["attention"]["top1_weight_mean"] for r in rs])), 4),
            "logit_std_mean": round(float(np.mean([r["attention"]["logit_std_mean"] for r in rs])), 4),
            "q_rms_token_cv_mean": round(float(np.mean([r["attention"]["q_rms_token_cv_mean"] for r in rs])), 4),
            "k_rms_token_cv_mean": round(float(np.mean([r["attention"]["k_rms_token_cv_mean"] for r in rs])), 4),
            "train_seconds_mean": round(float(np.mean([r["train_seconds"] for r in rs])), 1),
        }
        for key in ("alpha_q_mean", "alpha_k_mean", "alpha_q_min", "alpha_k_min",
                    "alpha_q_max", "alpha_k_max", "tau_q_token_std_mean", "tau_k_token_std_mean"):
            vals = [r["attention"].get(key) for r in rs if key in r["attention"]]
            if vals:
                agg = np.min if key.endswith("_min") else (np.max if key.endswith("_max") else np.mean)
                row[key] = round(float(agg(vals)), 4)
        by_arm[arm] = row

    base = by_arm["baseline"]["val_bpc_mean"]
    qk = by_arm["qknorm"]["val_bpc_mean"]
    tax = qk - base
    tol = p["tolerance_bpc"]
    decomposition = []
    for arm in arms:
        if arm == "baseline":
            continue
        d = by_arm[arm]["val_bpc_mean"] - base
        entry = {
            "arm": arm,
            "delta_vs_baseline": round(d, 5),
            "share_of_tax": round(d / tax, 3) if abs(tax) > 1e-9 else None,
            "recovers_baseline": bool(d <= tol),
        }
        if arm.startswith("qknorm_dyn"):
            entry["rescue_fraction_of_tax"] = round((qk - by_arm[arm]["val_bpc_mean"]) / tax, 3) \
                if abs(tax) > 1e-9 else None
        decomposition.append(entry)

    additivity = {
        "qnorm_only_plus_knorm_only": round(
            (by_arm["qnorm_only"]["val_bpc_mean"] - base)
            + (by_arm["knorm_only"]["val_bpc_mean"] - base), 5),
        "qknorm_delta": round(tax, 5),
    }
    frozen_gain_effect = round(by_arm["qknorm_frozen_gain"]["val_bpc_mean"] - qk, 5)

    # replication vs 2026-08-11 per-seed values (deterministic harness, same seeds)
    rep = []
    for arm in ("baseline", "qknorm"):
        parent = p["parent_nh1_per_seed"][arm]
        for seed, val in parent.items():
            tonight = by_arm[arm]["val_bpc_per_seed"].get(int(seed))
            if tonight is not None:
                rep.append({"arm": arm, "seed": int(seed), "tonight": tonight,
                            "parent_2026_08_11": val, "delta": round(tonight - val, 5)})

    recovering = [e["arm"] for e in decomposition if e["recovers_baseline"]]
    return {
        "by_arm": by_arm,
        "baseline_bpc": base,
        "qknorm_bpc": qk,
        "nh1_tax_bpc_tonight": round(tax, 5),
        "nh1_tax_bpc_parent": p["parent_nh1_tax_bpc"],
        "tolerance_bpc": tol,
        "decomposition": decomposition,
        "additivity_check": additivity,
        "frozen_gain_minus_qknorm": frozen_gain_effect,
        "arms_recovering_baseline": recovering,
        "replication_vs_2026_08_11": rep,
    }


def make_chart(m, p, headline, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = p["arm_order"]
    labels = {
        "baseline": "baseline\n(no norm)", "qknorm": "qknorm\n(q+k, gains)",
        "qnorm_only": "q-norm\nonly", "knorm_only": "k-norm\nonly",
        "qknorm_frozen_gain": "qknorm\nfrozen gains", "qknorm_dynk": "qknorm\n+dyn k",
        "qknorm_dynqk": "qknorm\n+dyn q+k",
    }
    colors = {
        "baseline": "#444444", "qknorm": "#c0392b", "qnorm_only": "#e67e22",
        "knorm_only": "#8e44ad", "qknorm_frozen_gain": "#2980b9",
        "qknorm_dynk": "#16a085", "qknorm_dynqk": "#27ae60",
    }
    base = m["baseline_bpc"]
    tol = p["tolerance_bpc"]
    xi = np.arange(len(arms))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))

    # panel 1: absolute bpc per arm with seed dots
    ax = axes[0]
    for i, arm in enumerate(arms):
        row = m["by_arm"][arm]
        ax.bar(i, row["val_bpc_mean"], 0.65, color=colors[arm])
        seeds = list(row["val_bpc_per_seed"].values())
        ax.plot([i] * len(seeds), seeds, "k.", ms=5, alpha=0.6, zorder=3)
    ax.axhline(base, color="#444444", lw=1.0, ls="--", label="baseline mean")
    ax.axhspan(base - tol, base + tol, color="gray", alpha=0.15, label=f"±{tol} tol")
    ax.axhline(m["qknorm_bpc"], color="#c0392b", lw=1.0, ls=":", label="qknorm mean")
    lo = min(r["val_bpc_mean"] for r in m["by_arm"].values())
    hi = max(max(r["val_bpc_per_seed"].values()) for r in m["by_arm"].values())
    ax.set_ylim(lo - 0.02, hi + 0.02)
    ax.set_xticks(xi)
    ax.set_xticklabels([labels[a] for a in arms], fontsize=7)
    ax.set_ylabel("val bits per character")
    ax.set_title("nh=1 (hd=128): val bpc by arm, 3 seeds, paired inits", fontsize=10)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25, axis="y")

    # panel 2: the tax decomposition
    ax = axes[1]
    d_arms = [a for a in arms if a != "baseline"]
    ds = [m["by_arm"][a]["val_bpc_mean"] - base for a in d_arms]
    ax.bar(np.arange(len(d_arms)), ds, 0.6, color=[colors[a] for a in d_arms])
    for i, (a, d) in enumerate(zip(d_arms, ds)):
        share = next(e["share_of_tax"] for e in m["decomposition"] if e["arm"] == a)
        ax.text(i, d + (0.001 if d >= 0 else -0.003), f"{share:+.0%}\nof tax",
                ha="center", fontsize=7)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhspan(-tol, tol, color="gray", alpha=0.12)
    ax.axhline(m["nh1_tax_bpc_tonight"], color="#c0392b", lw=1.0, ls=":",
               label=f"full qknorm tax ({m['nh1_tax_bpc_tonight']:+.4f})")
    ax.set_xticks(np.arange(len(d_arms)))
    ax.set_xticklabels([labels[a] for a in d_arms], fontsize=7)
    ax.set_ylabel("Δ val bpc vs paired baseline")
    ax.set_title("which component carries the single-head tax?", fontsize=10)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25, axis="y")

    # panel 3: mechanism probes per arm
    ax = axes[2]
    w = 0.27
    ax.bar(xi - w, [m["by_arm"][a]["logit_std_mean"] for a in arms], w,
           color="#2c7fb8", label="attn logit std")
    ax.bar(xi, [10 * m["by_arm"][a]["q_rms_token_cv_mean"] for a in arms], w,
           color="#e67e22", label="q-RMS token CV ×10")
    ax.bar(xi + w, [10 * m["by_arm"][a]["k_rms_token_cv_mean"] for a in arms], w,
           color="#8e44ad", label="k-RMS token CV ×10")
    for i, a in enumerate(arms):
        row = m["by_arm"][a]
        note = []
        if "alpha_k_mean" in row:
            note.append(f"αk={row['alpha_k_mean']:.2f}")
        if "alpha_q_mean" in row:
            note.append(f"αq={row['alpha_q_mean']:.2f}")
        if note:
            ax.text(i, ax.get_ylim()[1] * 0.02, "\n".join(note), ha="center", fontsize=6.5)
    ax.set_xticks(xi)
    ax.set_xticklabels([labels[a] for a in arms], fontsize=7)
    ax.set_title("probes: sharpness + per-token magnitude structure", fontsize=10)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle(headline, fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------------ main
def main():
    t0 = time.time()
    cfg = load_config()
    p = dict(cfg["params"])
    if SMOKE:
        p["steps"], p["warmup"], p["max_eval_blocks"], p["entropy_batches"] = 40, 4, 32, 1
        p["seeds"] = [0]
    train_ids, val_ids, vocab = load_data(cfg)
    print(f"data: {len(train_ids)} train / {len(val_ids)} val chars, vocab={vocab}")

    runs = []
    sigs = {}
    for seed in p["seeds"]:
        for arm in p["arm_order"]:
            flags = p["arms"][arm]
            rec = train_one(vocab, arm, flags, seed, p, train_ids, val_ids)
            runs.append(rec)
            sigs.setdefault(seed, set()).add(rec["shared_init_signature"])
    for seed, ss in sigs.items():
        assert len(ss) == 1, f"shared-init mismatch at seed={seed}: {ss}"
    print("paired-init check passed: identical shared weights across all arms at every seed")

    m = summarize(runs, p)
    rec_arms = m["arms_recovering_baseline"]
    headline = (f"QK-norm nh=1 tax {m['nh1_tax_bpc_tonight']:+.4f} bpc — "
                f"recovered by: {', '.join(rec_arms) if rec_arms else 'NO single component'}; "
                f"q-only {next(e['delta_vs_baseline'] for e in m['decomposition'] if e['arm']=='qnorm_only'):+.4f}, "
                f"k-only {next(e['delta_vs_baseline'] for e in m['decomposition'] if e['arm']=='knorm_only'):+.4f}, "
                f"frozen-gain {m['frozen_gain_minus_qknorm']:+.4f} vs qknorm, "
                f"dynk rescue {next(e.get('rescue_fraction_of_tax') for e in m['decomposition'] if e['arm']=='qknorm_dynk')}")
    print("\n" + headline)
    for e in m["decomposition"]:
        print(f"  {e['arm']:18s} Δ={e['delta_vs_baseline']:+.5f} "
              f"share={e['share_of_tax']} recovers={e['recovers_baseline']}")
    print(f"  additivity: q-only + k-only = {m['additivity_check']['qnorm_only_plus_knorm_only']:+.5f} "
          f"vs qknorm Δ {m['additivity_check']['qknorm_delta']:+.5f}")
    for r in m["replication_vs_2026_08_11"]:
        print(f"  replication {r['arm']} seed {r['seed']}: {r['tonight']} vs {r['parent_2026_08_11']} "
              f"(Δ {r['delta']})")

    results = {
        "id": cfg["id"],
        "git_commit": git_sha(),
        "seed": cfg["seed"],
        "duration_sec": round(time.time() - t0, 2),
        "smoke": SMOKE,
        "metrics": {"headline": headline, **m},
        "runs": runs,
        "env": env_info(),
        "config": cfg,
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=1)
    make_chart(m, p, headline, HERE / "chart.png")
    print(f"\nwrote results.json + chart.png in {results['duration_sec']}s")


if __name__ == "__main__":
    main()
