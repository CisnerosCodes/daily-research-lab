"""The last untested component of the QK-norm hd=4 cliff: the GRADIENT PATH through ||q_t||.

Lineage:
  2026-07-30_qknorm-head-dim: QK-norm helps every iso-param head split of d_model=128 except
  head_dim=4, where it COSTS +0.143 bpc.
  2026-07-31_qknorm-hd4-temperature-rescue: static per-head learnable temperature refunds 2%
  of the cliff — eliminated causally.
  2026-08-02_qknorm-hd4-dynamic-temperature: restoring the pre-norm query RMS as a DETACHED
  per-token feature, tau = (r_t/EMA)^alpha, refunds 54%. But detaching severs the gradient
  path the baseline has: the baseline can LEARN where to place large ||q_t||.

Tonight, three arms on top of qknorm at the cliff config (hd=4, nh=32), all on the shared-init
shared-batch-stream harness:

  dynq_detached    exact 2026-08-02 arm (replication anchor): tau = clamp(r_det/EMA)^alpha
  dynq_undetached  byte-identical EXCEPT r is not detached — the ONLY change is the gradient
                   path through the per-token magnitude (EMA update still no_grad)
  dynq_raw         tau = r_t undetached, no EMA, no clamp, no alpha. Algebraically
                   _rms(q)*gain*r_t ~ q*gain — the baseline with per-channel gains — so this
                   kills the exact-form confound of the (r/EMA)^alpha parametrisation.

Readouts:
  - rescue fraction of the cliff per arm: (qknorm - arm) / (qknorm - baseline);
    references: static 0.02 (07-31), detached 0.54 (08-02).
  - mechanistic signature of the open path: pre-norm q-RMS token CV per arm — does the model
    place MORE per-token magnitude variance when the gradient can shape it?
  - learned alphas, realised tau spread, sharpness probes (entropy/top-1/logit std).

Deterministic, CPU-only. Usage:  python run.py   (SMOKE=1 for a 40-step smoke test)
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
# arm -> (qknorm, dyn_mode); dyn_mode in {None, "detached", "undetached", "raw"}
ARM_FLAGS = {
    "baseline": (False, None),
    "qknorm": (True, None),
    "dynq_detached": (True, "detached"),
    "dynq_undetached": (True, "undetached"),
    "dynq_raw": (True, "raw"),
}


class Block(nn.Module):
    """Pre-norm transformer block, identical to 2026-07-30/31/08-02, with the dynamic
    query-temperature machinery parameterised by dyn_mode.

    qknorm:     per-head RMS-norm on q and k with learnable per-channel gains of length d,
                then 1/sqrt(head_dim) inside attention.
    detached:   normalised q scaled per token/head by tau = clamp(r_det/EMA, 1/c, c)^alpha,
                r_det the DETACHED pre-norm q-RMS (exact 2026-08-02 dynq arm).
    undetached: identical formula, but r keeps its gradient — the model can learn where to
                place large ||q_t||. EMA update is still under no_grad (train-time only).
    raw:        tau = r_t undetached, no EMA, no clamp, no alpha.
    Because scaling happens before the dot product, SDPA and the probes see identical
    effective logits.
    """

    def __init__(self, d, n_head, d_ff, qknorm: bool, dyn_mode,
                 ema_momentum: float, ratio_clamp: float):
        super().__init__()
        assert d % n_head == 0
        assert qknorm or dyn_mode is None, "dynamic arms are defined on top of qknorm"
        self.n_head = n_head
        self.head_dim = d // n_head
        self.qknorm = qknorm
        self.dyn_mode = dyn_mode
        self.ema_momentum = ema_momentum
        self.ratio_clamp = ratio_clamp
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.fc = nn.Linear(d, d_ff, bias=False)
        self.out = nn.Linear(d_ff, d, bias=False)
        if qknorm:  # init to ones: consumes no RNG, keeps shared weights bit-identical
            self.q_gain = nn.Parameter(torch.ones(d))
            self.k_gain = nn.Parameter(torch.ones(d))
        if dyn_mode in ("detached", "undetached"):  # deterministic init: consumes no RNG
            self.q_alpha = nn.Parameter(torch.ones(n_head))
            self.register_buffer("q_rms_ema", torch.ones(n_head))
            self.register_buffer("q_ema_ready", torch.zeros(1))

    def _rms(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

    def _update_ema(self, r_det):
        ema, ready = self.q_rms_ema, self.q_ema_ready
        if self.training:
            with torch.no_grad():
                batch_mean = r_det.mean(dim=(0, 1))
                if float(ready) == 0.0:
                    ema.copy_(batch_mean)
                    ready.fill_(1.0)
                else:
                    ema.mul_(self.ema_momentum).add_(batch_mean, alpha=1 - self.ema_momentum)

    def _qkv(self, x, return_stats=False):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        shape4 = (B, T, self.n_head, self.head_dim)
        q4, k4 = q.view(*shape4), k.view(*shape4)
        r_grad = q4.pow(2).mean(-1).add(1e-12).sqrt()          # (B, T, nh) pre-norm q-RMS, WITH grad
        r_det = r_grad.detach()
        if self.qknorm:
            q = self._rms(q4).reshape(B, T, D) * self.q_gain
            k = self._rms(k4).reshape(B, T, D) * self.k_gain
        q = q.view(*shape4)
        k = k.view(*shape4)
        tau = None
        if self.dyn_mode in ("detached", "undetached"):
            self._update_ema(r_det)
            r_use = r_det if self.dyn_mode == "detached" else r_grad
            c = self.ratio_clamp
            ratio = (r_use / self.q_rms_ema.view(1, 1, -1)).clamp(1.0 / c, c)
            tau = torch.exp(self.q_alpha.view(1, 1, -1) * ratio.log())
            q = q * tau.unsqueeze(-1)
        elif self.dyn_mode == "raw":
            tau = r_grad                                        # undetached, no EMA/clamp/alpha
            q = q * tau.unsqueeze(-1)
        out = (q.transpose(1, 2), k.transpose(1, 2), v.view(*shape4).transpose(1, 2))
        if return_stats:
            return out, {"r_q": r_det, "tau_q": None if tau is None else tau.detach()}
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
    def __init__(self, vocab, d, n_layer, n_head, d_ff, block_size, qknorm, dyn_mode,
                 ema_momentum, ratio_clamp):
        super().__init__()
        self.vocab, self.block_size = vocab, block_size
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(block_size, d)
        self.blocks = nn.ModuleList([
            Block(d, n_head, d_ff, qknorm, dyn_mode, ema_momentum, ratio_clamp)
            for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.apply(self._init)
        for name, prm in self.named_parameters():   # nanoGPT residual-projection scaling
            if name.endswith("proj.weight") or name.endswith("out.weight"):
                nn.init.normal_(prm, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))
        with torch.no_grad():   # _init must not disturb the deterministic extras
            for blk in self.blocks:
                if qknorm:
                    blk.q_gain.fill_(1.0)
                    blk.k_gain.fill_(1.0)
                if dyn_mode in ("detached", "undetached"):
                    blk.q_alpha.fill_(1.0)

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
    per-token modulation stats: CV of pre-norm q-RMS (the open channel's signature) and the
    realised tau spread (what the dynamic arms actually apply)."""
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
        H_sum = torch.zeros(blk.n_head)
        top_sum = torch.zeros(blk.n_head)
        lg_mu = torch.zeros(blk.n_head)
        lg_sq = torch.zeros(blk.n_head)
        lg_n = 0
        rq_mu = torch.zeros(blk.n_head)
        rq_sq = torch.zeros(blk.n_head)
        tau_mu = torch.zeros(blk.n_head)
        tau_sq = torch.zeros(blk.n_head)
        r_n = 0
        nb = 0
        for s in range(0, n_blocks, B):
            xb = torch.from_numpy(xs[s:s + B])
            h = model.embed(xb)
            for j in range(li):
                h = model.blocks[j](h)
            logits, stats = blk.attn_logits(h, with_stats=True)   # (B, nh, T, T)
            probs = logits.masked_fill(~causal, float("-inf")).softmax(-1)
            ent = -(probs * torch.log(probs.clamp(min=1e-12))).sum(-1)   # (B, nh, T)
            H_sum += (ent / norm)[:, :, keep].mean(dim=(0, 2))
            top_sum += probs.max(-1).values[:, :, keep].mean(dim=(0, 2))
            lv = logits.masked_select(causal.view(1, 1, bs, bs)).view(logits.shape[0], blk.n_head, -1)
            lg_mu += lv.mean(dim=(0, 2)) * lv.shape[0]
            lg_sq += lv.pow(2).mean(dim=(0, 2)) * lv.shape[0]
            lg_n += lv.shape[0]
            rq = stats["r_q"]                                     # (B, T, nh)
            rq_mu += rq.mean(dim=(0, 1)) * rq.shape[0]
            rq_sq += rq.pow(2).mean(dim=(0, 1)) * rq.shape[0]
            if stats["tau_q"] is not None:
                tq = stats["tau_q"]
                tau_mu += tq.mean(dim=(0, 1)) * tq.shape[0]
                tau_sq += tq.pow(2).mean(dim=(0, 1)) * tq.shape[0]
            r_n += rq.shape[0]
            nb += 1
        mu = lg_mu / lg_n
        std = (lg_sq / lg_n - mu ** 2).clamp(min=0).sqrt()
        rqm = rq_mu / r_n
        rqs = (rq_sq / r_n - rqm ** 2).clamp(min=0).sqrt()
        entry = {
            "layer": li,
            "entropy_norm_per_head": [round(float(u), 4) for u in (H_sum / nb)],
            "top1_weight_per_head": [round(float(u), 4) for u in (top_sum / nb)],
            "logit_std_per_head": [round(float(u), 4) for u in std],
            "q_rms_token_cv_per_head": [round(float(u), 4) for u in (rqs / rqm.clamp(min=1e-9))],
            "q_rms_token_mean_per_head": [round(float(u), 4) for u in rqm],
        }
        if blk.dyn_mode is not None:
            tm = tau_mu / r_n
            ts = (tau_sq / r_n - tm ** 2).clamp(min=0).sqrt()
            entry["tau_token_mean_per_head"] = [round(float(u), 4) for u in tm]
            entry["tau_token_std_per_head"] = [round(float(u), 4) for u in ts]
        if blk.dyn_mode in ("detached", "undetached"):
            entry["alpha_per_head"] = [round(float(u), 4) for u in blk.q_alpha]
        per_layer.append(entry)
    model.train()
    allH = [u for L in per_layer for u in L["entropy_norm_per_head"]]
    allT = [u for L in per_layer for u in L["top1_weight_per_head"]]
    allS = [u for L in per_layer for u in L["logit_std_per_head"]]
    allCV = [u for L in per_layer for u in L["q_rms_token_cv_per_head"]]
    out = {
        "per_layer": per_layer,
        "entropy_norm_mean": round(float(np.mean(allH)), 4),
        "top1_weight_mean": round(float(np.mean(allT)), 4),
        "logit_std_mean": round(float(np.mean(allS)), 4),
        "logit_std_min_head": round(float(np.min(allS)), 4),
        "logit_std_max_head": round(float(np.max(allS)), 4),
        "q_rms_token_cv_mean": round(float(np.mean(allCV)), 4),
    }
    allA = [u for L in per_layer for u in L.get("alpha_per_head", [])]
    if allA:
        out["alpha_mean"] = round(float(np.mean(allA)), 4)
        out["alpha_min"] = round(float(np.min(allA)), 4)
        out["alpha_max"] = round(float(np.max(allA)), 4)
    allTS = [u for L in per_layer for u in L.get("tau_token_std_per_head", [])]
    if allTS:
        out["tau_token_std_mean"] = round(float(np.mean(allTS)), 4)
    return out


def train_one(vocab, arm, n_head, seed, p, train_ids, val_ids):
    head_dim = p["d_model"] // n_head
    qknorm, dyn_mode = ARM_FLAGS[arm]
    set_seeds(seed)                    # identical init across ALL arms at a seed
    model = GPT(vocab, p["d_model"], p["n_layer"], n_head,
                p["d_ff_mult"] * p["d_model"], p["block_size"], qknorm, dyn_mode,
                p["ema_momentum"], p["ratio_clamp"])
    shared_sig = float(sum(float(q.detach().double().abs().sum())
                           for name, q in model.named_parameters()
                           if "gain" not in name and "alpha" not in name))
    decay = [q for q in model.parameters() if q.dim() >= 2]
    nodecay = [q for q in model.parameters() if q.dim() < 2]   # gains + alphas: no WD
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
        "arm": arm, "n_head": int(n_head), "head_dim": int(head_dim), "seed": int(seed),
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
    if dyn_mode in ("detached", "undetached"):
        extra += f" alpha=[{probe['alpha_min']},{probe['alpha_max']}]"
    if dyn_mode is not None and "tau_token_std_mean" in probe:
        extra += f" tauStd={probe['tau_token_std_mean']:.3f}"
    print(f"  [{arm:16s} hd={head_dim:3d} seed={seed}] P={rec['n_params']} "
          f"bpc={rec['val_bpc']:.4f} logit_std={probe['logit_std_mean']:.3f} "
          f"entN={probe['entropy_norm_mean']:.3f} top1={probe['top1_weight_mean']:.3f} "
          f"qCV={probe['q_rms_token_cv_mean']:.3f}{extra} ({rec['train_seconds']}s)", flush=True)
    return rec


# -------------------------------------------------------------------------- analysis
def summarize(runs, p):
    out = {}
    for arm in p["arms"]:
        rs = [r for r in runs if r["arm"] == arm]
        b = np.array([r["val_bpc"] for r in rs])
        out[arm] = {
            "arm": arm,
            "val_bpc_per_seed": [round(float(u), 5) for u in b],
            "val_bpc_mean": round(float(b.mean()), 5),
            "val_bpc_min": round(float(b.min()), 5),
            "val_bpc_max": round(float(b.max()), 5),
            "seed_spread_bpc": round(float(b.max() - b.min()), 5),
            "entropy_norm_mean": round(float(np.mean([r["attention"]["entropy_norm_mean"] for r in rs])), 4),
            "top1_weight_mean": round(float(np.mean([r["attention"]["top1_weight_mean"] for r in rs])), 4),
            "logit_std_mean": round(float(np.mean([r["attention"]["logit_std_mean"] for r in rs])), 4),
            "q_rms_token_cv_mean": round(float(np.mean([r["attention"]["q_rms_token_cv_mean"] for r in rs])), 4),
            "train_seconds_mean": round(float(np.mean([r["train_seconds"] for r in rs])), 1),
        }
        for key in ("alpha_mean", "alpha_min", "alpha_max", "tau_token_std_mean"):
            vals = [r["attention"].get(key) for r in rs if key in r["attention"]]
            if vals:
                agg = np.min if key.endswith("_min") else (np.max if key.endswith("_max") else np.mean)
                out[arm][key] = round(float(agg(vals)), 4)
    return out


def make_chart(sums, runs, p, headline, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"baseline": "baseline\n(no norm)", "qknorm": "QK-norm",
              "dynq_detached": "+dyn q-temp\n(detached, 08-02)",
              "dynq_undetached": "+dyn q-temp\n(UNdetached)",
              "dynq_raw": "+raw $r_t$\n(undetached)"}
    colors = {"baseline": "#444444", "qknorm": "#c0392b", "dynq_detached": "#2c7fb8",
              "dynq_undetached": "#41ab5d", "dynq_raw": "#8856a7"}
    arms = p["arms"]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9))

    # panel 1: bpc by arm at hd=4
    ax = axes[0]
    x = np.arange(len(arms))
    for i, arm in enumerate(arms):
        s = sums[arm]
        ax.bar(i, s["val_bpc_mean"], 0.6, color=colors[arm])
        ys = s["val_bpc_per_seed"]
        ax.plot([i] * len(ys), ys, "ko", ms=4, alpha=0.6, zorder=3)
    ax.axhline(sums["baseline"]["val_bpc_mean"], color="#444444", lw=0.9, ls="--", alpha=0.7)
    ax.axhline(sums["qknorm"]["val_bpc_mean"], color="#c0392b", lw=0.9, ls="--", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[a] for a in arms], fontsize=8)
    ax.set_ylabel("val bits per character")
    lo = min(s["val_bpc_min"] for s in sums.values()) - 0.02
    hi = max(s["val_bpc_max"] for s in sums.values()) + 0.02
    ax.set_ylim(lo, hi)
    ax.set_title("hd=4 cliff: does opening the gradient path\nthrough $||q_t||$ close it? (dots = seeds)")
    ax.grid(alpha=0.3, axis="y")

    # panel 2: rescue fraction of the cliff
    ax = axes[1]
    cliff = sums["qknorm"]["val_bpc_mean"] - sums["baseline"]["val_bpc_mean"]
    names = ["static tau\n(07-31 ref)", "detached\n(08-02 ref)"]
    vals = [p["parent_static_tau_rescue_fraction_2026_07_31"],
            p["parent_detached_rescue_fraction_2026_08_02"]]
    cols = ["#bbbbbb", "#9ecae1"]
    for arm in ("dynq_detached", "dynq_undetached", "dynq_raw"):
        names.append(labels[arm].replace("\n", " "))
        vals.append((sums["qknorm"]["val_bpc_mean"] - sums[arm]["val_bpc_mean"]) / cliff)
        cols.append(colors[arm])
    ax.bar(np.arange(len(names)), vals, 0.6, color=cols)
    ax.axhline(1.0, color="k", lw=0.8, ls="--", label="full closure (= baseline)")
    ax.axhline(0.0, color="#888888", lw=0.8)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("rescue fraction of the qknorm cliff")
    ax.set_title("How much of the +%.3f-bpc cliff each dial refunds" % cliff)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    # panel 3: the open channel's signature — per-token q-RMS CV by arm
    ax = axes[2]
    for i, arm in enumerate(arms):
        cvs = [r["attention"]["q_rms_token_cv_mean"] for r in runs if r["arm"] == arm]
        ax.bar(i, np.mean(cvs), 0.6, color=colors[arm])
        ax.plot([i] * len(cvs), cvs, "ko", ms=4, alpha=0.6, zorder=3)
    ax.set_xticks(np.arange(len(arms)))
    ax.set_xticklabels([labels[a] for a in arms], fontsize=8)
    ax.set_ylabel("pre-norm q-RMS token CV (mean over heads)")
    ax.set_title("Does the model place MORE per-token magnitude\nvariance when the gradient can shape it?")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(f"Gradient path through per-token query magnitude, hd=4 nh=32 d=128, tiny-shakespeare — {headline}",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    cfg = load_config()
    p = cfg["params"]
    if SMOKE:
        p["steps"], p["warmup"], p["seeds"] = 40, 5, [0]
    t_start = time.time()

    train_ids, val_ids, vocab = load_data(cfg)
    print(f"data: {len(train_ids)} train chars, {len(val_ids)} val chars, vocab {vocab}", flush=True)
    (nh, hd), = p["configs"]
    assert nh * hd == p["d_model"], f"({nh},{hd}) does not tile d_model={p['d_model']}"

    runs = []
    for arm in p["arms"]:
        for seed in p["seeds"]:
            runs.append(train_one(vocab, arm, nh, seed, p, train_ids, val_ids))

    sig_ok = all(
        len({r["shared_init_signature"] for r in runs if r["seed"] == sd}) == 1
        for sd in p["seeds"])

    sums = summarize(runs, p)
    cliff = sums["qknorm"]["val_bpc_mean"] - sums["baseline"]["val_bpc_mean"]

    rescue = {}
    for arm in ("dynq_detached", "dynq_undetached", "dynq_raw"):
        rescued = sums["qknorm"]["val_bpc_mean"] - sums[arm]["val_bpc_mean"]
        rescue[arm] = {
            "rescue_bpc_qknorm_minus_arm": round(rescued, 5),
            "rescue_fraction_of_cliff": round(rescued / cliff, 4) if abs(cliff) > 1e-9 else None,
            "arm_minus_baseline_bpc": round(sums[arm]["val_bpc_mean"] - sums["baseline"]["val_bpc_mean"], 5),
        }

    # per-seed paired deltas (identical init + batch stream make these true pairs)
    paired = []
    for sd in p["seeds"]:
        def bpc_of(arm):
            return [r["val_bpc"] for r in runs if r["arm"] == arm and r["seed"] == sd][0]
        b, q = bpc_of("baseline"), bpc_of("qknorm")
        row = {"seed": sd, "cliff_qknorm_minus_baseline": round(q - b, 5)}
        for arm in ("dynq_detached", "dynq_undetached", "dynq_raw"):
            row[f"rescue_{arm}"] = round(q - bpc_of(arm), 5)
            row[f"{arm}_minus_baseline"] = round(bpc_of(arm) - b, 5)
        paired.append(row)

    # replication check vs 2026-08-02 (same harness, same seeds -> detached arm should land close)
    replication = {
        "parent_cliff_bpc": p["parent_cliff_bpc_2026_08_02"],
        "tonight_cliff_bpc": round(cliff, 5),
        "parent_detached_rescue_fraction": p["parent_detached_rescue_fraction_2026_08_02"],
        "tonight_detached_rescue_fraction": rescue["dynq_detached"]["rescue_fraction_of_cliff"],
    }

    fr_det = rescue["dynq_detached"]["rescue_fraction_of_cliff"]
    fr_und = rescue["dynq_undetached"]["rescue_fraction_of_cliff"]
    fr_raw = rescue["dynq_raw"]["rescue_fraction_of_cliff"]
    headline_short = (f"rescue: detached {fr_det:.2f}, undetached {fr_und:.2f}, raw {fr_raw:.2f}")

    metrics = {
        "headline": "does opening the gradient path through per-token ||q_t|| close the rest of the hd=4 qknorm cliff?",
        "identical_shared_init_per_seed": sig_ok,
        "shared_batch_stream": True,
        "n_params_by_arm": {r["arm"]: r["n_params"] for r in runs if r["seed"] == p["seeds"][0]},
        "cliff_bpc_qknorm_minus_baseline": round(cliff, 5),
        "replication_vs_2026_08_02": replication,
        "parent_static_tau_rescue_fraction_2026_07_31": p["parent_static_tau_rescue_fraction_2026_07_31"],
        "rescue": rescue,
        "seed_spread_bpc_by_arm": {a: sums[a]["seed_spread_bpc"] for a in p["arms"]},
        "paired_per_seed": paired,
        "bpc_by_arm": {a: sums[a]["val_bpc_mean"] for a in p["arms"]},
        "sharpness": {
            "entropy_norm": {a: sums[a]["entropy_norm_mean"] for a in p["arms"]},
            "top1_weight": {a: sums[a]["top1_weight_mean"] for a in p["arms"]},
            "logit_std_mean": {a: sums[a]["logit_std_mean"] for a in p["arms"]},
        },
        "per_token_magnitude_channel": {
            "q_rms_token_cv": {a: sums[a]["q_rms_token_cv_mean"] for a in p["arms"]},
            "applied_tau_token_std": {a: sums[a].get("tau_token_std_mean") for a in p["arms"]},
        },
        "learned_exponents": {
            "dynq_detached_alpha": {k: sums["dynq_detached"].get(f"alpha_{k}") for k in ("mean", "min", "max")},
            "dynq_undetached_alpha": {k: sums["dynq_undetached"].get(f"alpha_{k}") for k in ("mean", "min", "max")},
        },
        "by_arm": sums,
        "train_steps": p["steps"],
        "tokens_per_run": p["steps"] * p["batch_size"] * p["block_size"],
        "seeds": p["seeds"],
        "n_runs": len(runs),
        "runs": runs,
    }

    make_chart(sums, runs, p, headline_short, HERE / "chart.png")

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
    print(f"shared init identical per seed: {sig_ok}")
    print("bpc: " + " | ".join(f"{a} {sums[a]['val_bpc_mean']:.4f}" for a in p["arms"]))
    print(f"cliff {cliff:+.4f} bpc | {headline_short}")
    print("q-RMS token CV: " + " | ".join(f"{a} {sums[a]['q_rms_token_cv_mean']:.3f}" for a in p["arms"]))
    print(f"total {results['duration_sec']:.0f}s")


if __name__ == "__main__":
    main()
