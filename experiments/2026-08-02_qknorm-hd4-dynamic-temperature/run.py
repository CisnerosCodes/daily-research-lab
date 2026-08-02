"""Query-dependent (per-token) temperature on top of QK-norm: the dynamic test of the hd=4 cliff.

Lineage:
  2026-07-30_qknorm-head-dim: QK-norm helps every iso-param head split of d_model=128 except
  head_dim=4, where it COSTS +0.143 bpc. Probes suggested a sharpness cap.
  2026-07-31_qknorm-hd4-temperature-rescue: the causal test of that story FAILED — a free
  per-head static learnable temperature refunds 2% of the cliff and the optimizer leaves it
  at tau~1. Named next suspect: per-token (query-dependent) logit magnitude, which the
  unnormalised baseline has via ||q_t|| and which RMS-norm destroys and no static scalar can
  restore.

This experiment restores exactly that quantity. On top of QK-norm, each head h scales its
normalised query at token t by

    tau_{t,h} = (r_{t,h} / EMA_h) ** alpha_h        (ratio clamped to [1/8, 8])

where r_{t,h} is the DETACHED RMS of the PRE-norm query of token t in head h, EMA_h is a
train-time running mean of r (frozen at eval, batchnorm-style), and alpha_h is a per-head
learnable exponent with init 1.0 — the dial starts fully ON (per-token magnitude restored in
relative terms) and alpha -> 0 recovers plain qknorm, so gradient descent can keep it, tune
it, or discard it. A second arm (qknorm_dynqk) restores key-side per-token magnitude the same
way (its own beta_h, EMA and clamp), since the baseline's logits carry ||q_t||*||k_s||.

Arms (all on the shared-init, shared-batch-stream harness of 2026-07-30/31):
  baseline      no norm
  qknorm        per-head RMS on q,k + length-d learnable gains
  qknorm_dynq   qknorm + query-side dynamic temperature (both configs: cliff + do-no-harm)
  qknorm_dynqk  qknorm + query- AND key-side dynamic temperature (cliff config only)

Readouts:
  - rescue fraction at hd=4: (qknorm - arm) / (qknorm - baseline); static-tau reference from
    2026-07-31 is 0.021.
  - do-no-harm at hd=32 for dynq.
  - learned alphas (kept at ~1? driven to 0 = dial discarded? amplified?) and realised
    per-token tau spread, vs the baseline's own per-token q-RMS modulation (CV of r_{t,h}).
  - sharpness probes (attn entropy, top-1 weight, logit std) as in the parents.

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
class Block(nn.Module):
    """Pre-norm transformer block, identical to 2026-07-30/31, plus optional per-token
    (query-dependent and/or key-dependent) dynamic temperature on top of QK-norm.

    qknorm: per-head RMS-norm on q and k with learnable per-channel gains of length d
    (same shape for every split -> iso-param across configs), then 1/sqrt(head_dim).
    dynq:   normalised q additionally scaled per token/head by tau_{t,h} =
            clamp(r_{t,h}/EMA_h, 1/c, c) ** alpha_h with r the detached pre-norm q-RMS.
            alpha init 1 (dial ON), EMA updated in training mode only (frozen at eval).
    dynk:   same machinery on the key side (beta_h, its own EMA).
    Because scaling happens before the dot product, SDPA and the probes see identical
    effective logits.
    """

    def __init__(self, d, n_head, d_ff, qknorm: bool, dynq: bool, dynk: bool,
                 ema_momentum: float, ratio_clamp: float):
        super().__init__()
        assert d % n_head == 0
        assert qknorm or not (dynq or dynk), "dynamic arms are defined on top of qknorm"
        self.n_head = n_head
        self.head_dim = d // n_head
        self.qknorm = qknorm
        self.dynq = dynq
        self.dynk = dynk
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
        if dynq:    # init deterministic: consumes no RNG
            self.q_alpha = nn.Parameter(torch.ones(n_head))
            self.register_buffer("q_rms_ema", torch.ones(n_head))
            self.register_buffer("q_ema_ready", torch.zeros(1))
        if dynk:
            self.k_alpha = nn.Parameter(torch.ones(n_head))
            self.register_buffer("k_rms_ema", torch.ones(n_head))
            self.register_buffer("k_ema_ready", torch.zeros(1))

    def _rms(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

    def _dyn_tau(self, r, ema_name, ready_name, alpha):
        """r: detached pre-norm RMS (B, T, nh). Returns tau (B, T, nh)."""
        ema = getattr(self, ema_name)
        ready = getattr(self, ready_name)
        if self.training:
            with torch.no_grad():
                batch_mean = r.mean(dim=(0, 1))
                if float(ready) == 0.0:
                    ema.copy_(batch_mean)
                    ready.fill_(1.0)
                else:
                    ema.mul_(self.ema_momentum).add_(batch_mean, alpha=1 - self.ema_momentum)
        c = self.ratio_clamp
        ratio = (r / ema.view(1, 1, -1)).clamp(1.0 / c, c)
        return torch.exp(alpha.view(1, 1, -1) * ratio.log())

    def _qkv(self, x, return_stats=False):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        shape4 = (B, T, self.n_head, self.head_dim)
        q4, k4 = q.view(*shape4), k.view(*shape4)
        r_q = q4.pow(2).mean(-1).add(1e-12).sqrt().detach()   # (B, T, nh) pre-norm q-RMS
        r_k = k4.pow(2).mean(-1).add(1e-12).sqrt().detach()
        if self.qknorm:
            q = self._rms(q4).reshape(B, T, D) * self.q_gain
            k = self._rms(k4).reshape(B, T, D) * self.k_gain
        q = q.view(*shape4)
        k = k.view(*shape4)
        tau_q = tau_k = None
        if self.dynq:
            tau_q = self._dyn_tau(r_q, "q_rms_ema", "q_ema_ready", self.q_alpha)
            q = q * tau_q.unsqueeze(-1)
        if self.dynk:
            tau_k = self._dyn_tau(r_k, "k_rms_ema", "k_ema_ready", self.k_alpha)
            k = k * tau_k.unsqueeze(-1)
        out = (q.transpose(1, 2), k.transpose(1, 2), v.view(*shape4).transpose(1, 2))
        if return_stats:
            return out, {"r_q": r_q, "r_k": r_k, "tau_q": tau_q, "tau_k": tau_k}
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
    def __init__(self, vocab, d, n_layer, n_head, d_ff, block_size, qknorm, dynq, dynk,
                 ema_momentum, ratio_clamp):
        super().__init__()
        self.vocab, self.block_size = vocab, block_size
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(block_size, d)
        self.blocks = nn.ModuleList([
            Block(d, n_head, d_ff, qknorm, dynq, dynk, ema_momentum, ratio_clamp)
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
                if dynq:
                    blk.q_alpha.fill_(1.0)
                if dynk:
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
    per-token modulation stats: CV of pre-norm q-RMS (what the baseline modulates with) and
    the realised tau spread (what the dynamic arms actually apply)."""
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
        }
        if blk.dynq:
            tm = tau_mu / r_n
            ts = (tau_sq / r_n - tm ** 2).clamp(min=0).sqrt()
            entry["alpha_per_head"] = [round(float(u), 4) for u in blk.q_alpha]
            entry["tau_token_mean_per_head"] = [round(float(u), 4) for u in tm]
            entry["tau_token_std_per_head"] = [round(float(u), 4) for u in ts]
        if blk.dynk:
            entry["beta_per_head"] = [round(float(u), 4) for u in blk.k_alpha]
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
        out["tau_token_std_mean"] = round(float(np.mean(allTS)), 4)
    allB = [u for L in per_layer for u in L.get("beta_per_head", [])]
    if allB:
        out["beta_mean"] = round(float(np.mean(allB)), 4)
        out["beta_min"] = round(float(np.min(allB)), 4)
        out["beta_max"] = round(float(np.max(allB)), 4)
    return out


ARM_FLAGS = {   # arm -> (qknorm, dynq, dynk)
    "baseline": (False, False, False),
    "qknorm": (True, False, False),
    "qknorm_dynq": (True, True, False),
    "qknorm_dynqk": (True, True, True),
}


def train_one(vocab, arm, n_head, seed, p, train_ids, val_ids):
    head_dim = p["d_model"] // n_head
    qknorm, dynq, dynk = ARM_FLAGS[arm]
    set_seeds(seed)                    # identical init across ALL (arm, n_head) at a seed
    model = GPT(vocab, p["d_model"], p["n_layer"], n_head,
                p["d_ff_mult"] * p["d_model"], p["block_size"], qknorm, dynq, dynk,
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
    rng = np.random.default_rng(seed)  # identical batch stream across ALL arms/configs
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
    if dynq:
        extra += f" alpha=[{probe['alpha_min']},{probe['alpha_max']}] tauStd={probe['tau_token_std_mean']:.3f}"
    if dynk:
        extra += f" beta=[{probe['beta_min']},{probe['beta_max']}]"
    print(f"  [{arm:12s} hd={head_dim:3d} nh={n_head:2d} seed={seed}] P={rec['n_params']} "
          f"bpc={rec['val_bpc']:.4f} logit_std={probe['logit_std_mean']:.3f} "
          f"entN={probe['entropy_norm_mean']:.3f} top1={probe['top1_weight_mean']:.3f}"
          f"{extra} ({rec['train_seconds']}s)", flush=True)
    return rec


# -------------------------------------------------------------------------- analysis
def summarize(runs, p):
    out = {}
    for arm in p["arms"]:
        rows = []
        for nh, hd in p["configs"]:
            if hd not in p["arm_head_dims"][arm]:
                continue
            rs = [r for r in runs if r["arm"] == arm and r["n_head"] == nh]
            b = np.array([r["val_bpc"] for r in rs])
            row = {
                "arm": arm, "n_head": nh, "head_dim": hd,
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
            for key in ("alpha_mean", "alpha_min", "alpha_max", "tau_token_std_mean",
                        "beta_mean", "beta_min", "beta_max"):
                vals = [r["attention"].get(key) for r in rs if key in r["attention"]]
                if vals:
                    agg = np.min if key.endswith("_min") else (np.max if key.endswith("_max") else np.mean)
                    row[key] = round(float(agg(vals)), 4)
            rows.append(row)
        out[arm] = rows
    return out


def make_chart(sums, runs, p, headline, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"baseline": "baseline (no norm)", "qknorm": "QK-norm",
              "qknorm_dynq": "QK-norm + dyn q-temp", "qknorm_dynqk": "QK-norm + dyn qk-temp"}
    colors = {"baseline": "#444444", "qknorm": "#c0392b",
              "qknorm_dynq": "#2c7fb8", "qknorm_dynqk": "#41ab5d"}
    arms = p["arms"]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9))

    # panel 1: bpc by arm at each config
    ax = axes[0]
    configs = p["configs"]
    width = 0.2
    for ci, (nh, hd) in enumerate(configs):
        present = [a for a in arms if hd in p["arm_head_dims"][a]]
        for ai, arm in enumerate(present):
            s = [row for row in sums[arm] if row["head_dim"] == hd][0]
            xpos = ci + (ai - (len(present) - 1) / 2) * width
            ax.bar(xpos, s["val_bpc_mean"], width * 0.92, color=colors[arm],
                   label=labels[arm] if ci == max(range(len(configs))) else None)
            ys = s["val_bpc_per_seed"]
            ax.plot([xpos] * len(ys), ys, "ko", ms=4, alpha=0.6, zorder=3)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([f"hd={hd}\n(nh={nh})" for nh, hd in configs])
    ax.set_ylabel("val bits per character")
    lo = min(s["val_bpc_min"] for a in arms for s in sums[a]) - 0.02
    hi = max(s["val_bpc_max"] for a in arms for s in sums[a]) + 0.02
    ax.set_ylim(lo, hi)
    ax.set_title("Does a per-token (dynamic) temperature rescue the hd=4 cliff?\n(dots = seeds)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    # panel 2: per-token modulation restored? (baseline's own CV vs realised tau spread)
    ax = axes[1]
    hd4 = [(a, [row for row in sums[a] if row["head_dim"] == 4][0]) for a in arms]
    names, vals, cols = [], [], []
    for a, s in hd4:
        names.append(labels[a].replace("QK-norm + ", "+"))
        # baseline/qknorm: the modulation the model APPLIES is 0 (no per-token scale);
        # dynamic arms: realised tau std. Overlay everyone's pre-norm q-RMS CV as dots.
        vals.append(s.get("tau_token_std_mean", 0.0))
        cols.append(colors[a])
    x = np.arange(len(names))
    ax.bar(x, vals, 0.55, color=cols, label="applied per-token tau std")
    ax.plot(x, [s["q_rms_token_cv_mean"] for _, s in hd4], "kD", ms=7,
            label="available pre-norm q-RMS CV")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_title("hd=4: per-token modulation — available vs applied")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    # panel 3: learned exponents alpha (q side) and beta (k side) at hd=4
    ax = axes[2]
    groups = [("qknorm_dynq", "alpha_per_head", "alpha (q), dynq arm"),
              ("qknorm_dynqk", "alpha_per_head", "alpha (q), dynqk arm"),
              ("qknorm_dynqk", "beta_per_head", "beta (k), dynqk arm")]
    for gi, (arm, key, lab) in enumerate(groups):
        es = []
        for r in runs:
            if r["arm"] == arm and r["head_dim"] == 4:
                for L in r["attention"]["per_layer"]:
                    es.extend(L.get(key, []))
        xs = np.full(len(es), gi) + np.linspace(-0.22, 0.22, max(len(es), 2))[:len(es)]
        ax.plot(xs, es, "o", ms=3.5, alpha=0.5, color=colors[arm])
    ax.axhline(1.0, color="k", lw=0.8, ls="--", label="init (dial fully on)")
    ax.axhline(0.0, color="#888888", lw=0.8, ls=":", label="0 = plain qknorm")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[2] for g in groups], fontsize=8)
    ax.set_ylabel("learned exponent")
    ax.set_title("Did the heads keep the dial?\n(all heads x layers x seeds, hd=4)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    fig.suptitle(f"Query-dependent temperature on top of QK-norm (d_model=128, tiny-shakespeare) — {headline}",
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
    for nh, hd in p["configs"]:
        assert nh * hd == p["d_model"], f"({nh},{hd}) does not tile d_model={p['d_model']}"

    runs = []
    for arm in p["arms"]:
        for nh, hd in p["configs"]:
            if hd not in p["arm_head_dims"][arm]:
                continue
            for seed in p["seeds"]:
                runs.append(train_one(vocab, arm, nh, seed, p, train_ids, val_ids))

    sig_ok = all(
        len({r["shared_init_signature"] for r in runs if r["seed"] == sd}) == 1
        for sd in p["seeds"])

    sums = summarize(runs, p)

    def mean_bpc(arm, hd):
        return [s["val_bpc_mean"] for s in sums[arm] if s["head_dim"] == hd][0]

    def seed_spread(arm, hd):
        return [s["seed_spread_bpc"] for s in sums[arm] if s["head_dim"] == hd][0]

    # headline: rescue fractions at the cliff
    cliff = mean_bpc("qknorm", 4) - mean_bpc("baseline", 4)          # parents: +0.143 / +0.130
    rescue = {}
    for arm in ("qknorm_dynq", "qknorm_dynqk"):
        rescued = mean_bpc("qknorm", 4) - mean_bpc(arm, 4)
        rescue[arm] = {
            "rescue_bpc_qknorm_minus_arm_hd4": round(rescued, 5),
            "rescue_fraction_of_cliff": round(rescued / cliff, 4) if abs(cliff) > 1e-9 else None,
            "arm_minus_baseline_bpc_hd4": round(mean_bpc(arm, 4) - mean_bpc("baseline", 4), 5),
        }

    # per-seed paired deltas (identical init + batch stream make these true pairs)
    paired = []
    for sd in p["seeds"]:
        def bpc_of(arm):
            return [r["val_bpc"] for r in runs
                    if r["arm"] == arm and r["head_dim"] == 4 and r["seed"] == sd][0]
        b, q = bpc_of("baseline"), bpc_of("qknorm")
        dq, dqk = bpc_of("qknorm_dynq"), bpc_of("qknorm_dynqk")
        paired.append({"seed": sd,
                       "cliff_qknorm_minus_baseline": round(q - b, 5),
                       "rescue_dynq": round(q - dq, 5),
                       "rescue_dynqk": round(q - dqk, 5),
                       "dynq_minus_baseline": round(dq - b, 5),
                       "dynqk_minus_baseline": round(dqk - b, 5)})

    # do-no-harm at the qknorm optimum
    harm_hd32 = mean_bpc("qknorm_dynq", 32) - mean_bpc("qknorm", 32)

    s4 = {a: [s for s in sums[a] if s["head_dim"] == 4][0] for a in p["arms"]}

    fr_q = rescue["qknorm_dynq"]["rescue_fraction_of_cliff"]
    fr_qk = rescue["qknorm_dynqk"]["rescue_fraction_of_cliff"]
    headline_short = f"rescue fraction: dynq {fr_q:.2f}, dynqk {fr_qk:.2f} (static-tau ref 0.02)"

    metrics = {
        "headline": "rescue fraction of the hd=4 qknorm cliff by a per-token (query-dependent) temperature",
        "identical_shared_init_per_seed": sig_ok,
        "shared_batch_stream": True,
        "n_params_by_arm_config": {f"{r['arm']}_hd{r['head_dim']}": r["n_params"]
                                   for r in runs if r["seed"] == p["seeds"][0]},
        "cliff_bpc_qknorm_minus_baseline_hd4": round(cliff, 5),
        "parent_cliff_bpc_2026_07_30": 0.14299,
        "parent_cliff_bpc_2026_07_31": 0.13031,
        "parent_static_tau_rescue_fraction_2026_07_31": 0.021,
        "rescue": rescue,
        "seed_spread_bpc_hd4_by_arm": {a: seed_spread(a, 4) for a in p["arms"]},
        "paired_per_seed_hd4": paired,
        "do_no_harm_hd32_dynq_minus_qknorm_bpc": round(harm_hd32, 5),
        "bpc_by_arm_config": {a: {f"hd{s['head_dim']}": s["val_bpc_mean"] for s in sums[a]}
                              for a in p["arms"]},
        "sharpness_at_hd4": {
            "entropy_norm": {a: s4[a]["entropy_norm_mean"] for a in p["arms"]},
            "top1_weight": {a: s4[a]["top1_weight_mean"] for a in p["arms"]},
            "logit_std_mean": {a: s4[a]["logit_std_mean"] for a in p["arms"]},
        },
        "per_token_modulation_at_hd4": {
            "available_q_rms_token_cv": {a: s4[a]["q_rms_token_cv_mean"] for a in p["arms"]},
            "applied_tau_token_std": {a: s4[a].get("tau_token_std_mean") for a in p["arms"]},
        },
        "learned_exponents_hd4": {
            "dynq_alpha": {k: s4["qknorm_dynq"].get(f"alpha_{k}") for k in ("mean", "min", "max")},
            "dynqk_alpha": {k: s4["qknorm_dynqk"].get(f"alpha_{k}") for k in ("mean", "min", "max")},
            "dynqk_beta": {k: s4["qknorm_dynqk"].get(f"beta_{k}") for k in ("mean", "min", "max")},
        },
        "by_config": sums,
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
    print(f"hd=4  baseline {mean_bpc('baseline', 4):.4f} | qknorm {mean_bpc('qknorm', 4):.4f} "
          f"| +dynq {mean_bpc('qknorm_dynq', 4):.4f} | +dynqk {mean_bpc('qknorm_dynqk', 4):.4f}")
    print(f"hd=32 baseline {mean_bpc('baseline', 32):.4f} | qknorm {mean_bpc('qknorm', 32):.4f} "
          f"| +dynq {mean_bpc('qknorm_dynq', 32):.4f}")
    print(f"cliff {cliff:+.4f} bpc | {headline_short}")
    print(f"do-no-harm at hd=32 (dynq - qknorm): {harm_hd32:+.4f} bpc")
    print(f"total {results['duration_sec']:.0f}s")


if __name__ == "__main__":
    main()
