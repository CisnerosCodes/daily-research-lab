"""Is QK-norm + undetached dynamic query temperature a STRICTLY BETTER parametrisation?

Lineage:
  2026-07-30_qknorm-head-dim: at d_model=128, QK-norm helps hd=16/32/64, small tax at hd=128,
  big cliff at hd=4 (+0.143 bpc); flips the baseline's monotone curve into a U with interior
  optimum hd=32.
  2026-07-31 / 08-02 / 08-06: the hd=4 cliff decomposes into 2% static temperature, 54%
  detached per-token magnitude feature, 98% once the gradient path through ||q_t|| is open
  (undetached tau = clamp(r_t/EMA)^alpha). But 08-06 tested ONLY the cliff config, and the
  detached arm cost +0.011 at hd=32 (08-02), so composite do-no-harm is genuinely open.

Tonight: the full 2026-07-30 head-split sweep (hd 4/16/32/64/128), three arms on paired inits
and a shared batch stream:

  baseline    no norm (07-30 arm, replication anchor)
  qknorm      per-head RMS-norm q,k + learnable per-channel gains (07-30 arm, replication anchor)
  composite   qknorm + UNDETACHED dynq (the 08-06 winner) at every split

Readouts:
  - strictly-better verdict: is composite <= min(baseline, qknorm) + tol at EVERY split?
  - does the composite keep the hd=4 closure AND qknorm's mid-range win simultaneously?
  - does the U-curve optimum move? does the hd=128 qknorm tax disappear?
  - mechanistic: alpha engagement, applied tau spread, and per-token q-RMS CV vs head_dim —
    is the restored channel used LESS where qknorm alone already wins?

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
# arm -> (qknorm, dyn_mode); composite = qknorm + undetached dynq (the 08-06 winner)
ARM_FLAGS = {
    "baseline": (False, None),
    "qknorm": (True, None),
    "composite": (True, "undetached"),
}


class Block(nn.Module):
    """Pre-norm transformer block, byte-compatible with 2026-07-30/31/08-02/08-06.

    qknorm:     per-head RMS-norm on q and k with learnable per-channel gains of length d,
                then 1/sqrt(head_dim) inside attention.
    undetached: normalised q scaled per token/head by tau = clamp(r/EMA, 1/c, c)^alpha with
                r the pre-norm q-RMS KEEPING its gradient (EMA update still no_grad).
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
    per-token modulation stats: CV of pre-norm q-RMS and the realised tau spread."""
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
        if blk.dyn_mode is not None:
            tm = tau_mu / r_n
            ts = (tau_sq / r_n - tm ** 2).clamp(min=0).sqrt()
            entry["tau_token_mean_per_head"] = [round(float(u), 4) for u in tm]
            entry["tau_token_std_per_head"] = [round(float(u), 4) for u in ts]
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
    set_seeds(seed)                    # identical init across ALL arms at a (config, seed)
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
    if dyn_mode is not None:
        extra = (f" alpha=[{probe['alpha_min']},{probe['alpha_max']}]"
                 f" tauStd={probe.get('tau_token_std_mean', float('nan')):.3f}")
    print(f"  [{arm:9s} hd={head_dim:3d} seed={seed}] P={rec['n_params']} "
          f"bpc={rec['val_bpc']:.4f} logit_std={probe['logit_std_mean']:.3f} "
          f"entN={probe['entropy_norm_mean']:.3f} top1={probe['top1_weight_mean']:.3f} "
          f"qCV={probe['q_rms_token_cv_mean']:.3f}{extra} ({rec['train_seconds']}s)", flush=True)
    return rec


# -------------------------------------------------------------------------- analysis
def summarize(runs, p):
    """Per (arm, head_dim) aggregates + the paired deltas and the strictly-better verdict."""
    hds = sorted({r["head_dim"] for r in runs})
    by_arm = {}
    for arm in p["arms"]:
        rows = []
        for hd in hds:
            rs = [r for r in runs if r["arm"] == arm and r["head_dim"] == hd]
            b = np.array([r["val_bpc"] for r in rs])
            row = {
                "head_dim": hd,
                "n_head": rs[0]["n_head"],
                "val_bpc_per_seed": [round(float(u), 5) for u in b],
                "val_bpc_mean": round(float(b.mean()), 5),
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
                    row[key] = round(float(agg(vals)), 4)
            rows.append(row)
        means = [r["val_bpc_mean"] for r in rows]
        best_i = int(np.argmin(means))
        by_arm[arm] = {
            "by_head_dim": rows,
            "best_head_dim": rows[best_i]["head_dim"],
            "best_val_bpc_mean": rows[best_i]["val_bpc_mean"],
            "interior_optimum": bool(0 < best_i < len(rows) - 1),
            "mean_seed_spread_bpc": round(float(np.mean([r["seed_spread_bpc"] for r in rows])), 5),
        }

    def mean_of(arm, hd):
        return next(r["val_bpc_mean"] for r in by_arm[arm]["by_head_dim"] if r["head_dim"] == hd)

    tol = p["tolerance_bpc"]
    paired, strict_ok = [], True
    for hd in hds:
        b, q, c = mean_of("baseline", hd), mean_of("qknorm", hd), mean_of("composite", hd)
        best_parent = min(b, q)
        ok = c <= best_parent + tol
        strict_ok = strict_ok and ok
        paired.append({
            "head_dim": hd,
            "baseline": b, "qknorm": q, "composite": c,
            "qknorm_minus_baseline": round(q - b, 5),
            "composite_minus_baseline": round(c - b, 5),
            "composite_minus_qknorm": round(c - q, 5),
            "composite_minus_best_parent": round(c - best_parent, 5),
            "beats_or_matches_both_parents": bool(ok),
        })
    # replication anchors vs 2026-07-30 (same recipe, same seeds)
    rep = []
    for arm, key in (("baseline", "parent_2026_07_30_baseline_bpc"),
                     ("qknorm", "parent_2026_07_30_qknorm_bpc")):
        for hd in hds:
            parent = p[key].get(hd)
            if parent is not None:
                rep.append({"arm": arm, "head_dim": hd, "tonight": mean_of(arm, hd),
                            "parent_2026_07_30": parent,
                            "delta": round(mean_of(arm, hd) - parent, 5)})
    hd_min = min(hds)
    cliff = mean_of("qknorm", hd_min) - mean_of("baseline", hd_min)
    rescue = (mean_of("qknorm", hd_min) - mean_of("composite", hd_min)) / cliff if abs(cliff) > 1e-9 else None
    return {
        "by_arm": by_arm,
        "paired_by_head_dim": paired,
        "strictly_better_verdict": bool(strict_ok),
        "tolerance_bpc": tol,
        "hd4_cliff_bpc_tonight": round(float(cliff), 5),
        "hd4_composite_rescue_fraction": None if rescue is None else round(float(rescue), 4),
        "replication_vs_2026_07_30": rep,
    }


def make_chart(m, p, headline, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"baseline": "#444444", "qknorm": "#c0392b", "composite": "#41ab5d"}
    labels = {"baseline": "baseline (no norm)", "qknorm": "QK-norm",
              "composite": "QK-norm + undetached dyn q-temp"}
    hds = [r["head_dim"] for r in m["by_arm"]["baseline"]["by_head_dim"]]
    xs = np.log2(hds)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9))

    # panel 1: the three curves
    ax = axes[0]
    for arm in p["arms"]:
        rows = m["by_arm"][arm]["by_head_dim"]
        ys = [r["val_bpc_mean"] for r in rows]
        ax.plot(xs, ys, "o-", color=colors[arm], label=labels[arm], lw=2, ms=5)
        for r, x in zip(rows, xs):
            ax.plot([x] * len(r["val_bpc_per_seed"]), r["val_bpc_per_seed"],
                    "k.", ms=3.5, alpha=0.45, zorder=3)
    # parent overlay (2-seed means, 2026-07-30)
    for key, c in (("parent_2026_07_30_baseline_bpc", "#444444"),
                   ("parent_2026_07_30_qknorm_bpc", "#c0392b")):
        ph = sorted(p[key]);  ax.plot(np.log2(ph), [p[key][h] for h in ph],
                                      "--", color=c, lw=0.9, alpha=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(h) for h in hds])
    ax.set_xlabel("head_dim (n_head = 128/head_dim)")
    ax.set_ylabel("val bits per character")
    ax.set_title("val bpc vs head split (dashed = 2026-07-30)", fontsize=10)
    ax.legend(fontsize=8, loc="upper center")
    ax.grid(alpha=0.25)

    # panel 2: paired deltas vs baseline
    ax = axes[1]
    w = 0.35
    xi = np.arange(len(hds))
    for off, arm in ((-w / 2, "qknorm"), (w / 2, "composite")):
        ds = [next(r for r in m["paired_by_head_dim"] if r["head_dim"] == h)[f"{arm}_minus_baseline"]
              for h in hds]
        ax.bar(xi + off, ds, w, color=colors[arm], label=f"{labels[arm]} − baseline")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhspan(-p["tolerance_bpc"], p["tolerance_bpc"], color="gray", alpha=0.12)
    ax.set_xticks(xi)
    ax.set_xticklabels([str(h) for h in hds])
    ax.set_xlabel("head_dim")
    ax.set_ylabel("Δ val bpc vs paired baseline")
    verdict = "STRICTLY BETTER" if m["strictly_better_verdict"] else "NOT strictly better"
    ax.set_title(f"paired effect — composite is {verdict} (tol ±{p['tolerance_bpc']})", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")

    # panel 3: the channel's engagement vs head_dim (composite arm)
    ax = axes[2]
    rows = m["by_arm"]["composite"]["by_head_dim"]
    ax.plot(xs, [r.get("tau_token_std_mean", np.nan) for r in rows], "o-",
            color="#2c7fb8", lw=2, ms=5, label="applied tau token-std")
    ax.plot(xs, [r.get("alpha_mean", np.nan) for r in rows], "s-",
            color="#8856a7", lw=2, ms=5, label="learned alpha (mean)")
    ax.plot(xs, [r["q_rms_token_cv_mean"] for r in rows], "^-",
            color="#41ab5d", lw=2, ms=5, label="pre-norm q-RMS token CV")
    for arm in ("baseline", "qknorm"):
        ax.plot(xs, [r["q_rms_token_cv_mean"] for r in m["by_arm"][arm]["by_head_dim"]],
                "--", color=colors[arm], lw=1.0, alpha=0.6,
                label=f"q-RMS CV ({arm})")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(h) for h in hds])
    ax.set_xlabel("head_dim")
    ax.set_title("how hard the restored channel works, by split", fontsize=10)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25)

    fig.suptitle(headline, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------------ main
def main():
    t0 = time.time()
    cfg = load_config()
    p = dict(cfg["params"])
    if SMOKE:
        p["steps"], p["warmup"], p["max_eval_blocks"], p["entropy_batches"] = 40, 4, 32, 1
        p["configs"] = [[32, 4], [1, 128]]
        p["seeds"] = [0]
    train_ids, val_ids, vocab = load_data(cfg)
    print(f"data: {len(train_ids)} train / {len(val_ids)} val chars, vocab={vocab}")

    runs = []
    for n_head, head_dim in p["configs"]:
        assert n_head * head_dim == p["d_model"]
        sigs = {}
        for seed in p["seeds"]:
            for arm in p["arms"]:
                rec = train_one(vocab, arm, n_head, seed, p, train_ids, val_ids)
                runs.append(rec)
                sigs.setdefault(seed, set()).add(rec["shared_init_signature"])
        for seed, ss in sigs.items():
            assert len(ss) == 1, f"shared-init mismatch at nh={n_head} seed={seed}: {ss}"
    print("paired-init check passed: identical shared weights across arms at every (config, seed)")

    m = summarize(runs, p)
    verdict = "STRICTLY BETTER (within tol) at every split" if m["strictly_better_verdict"] \
        else "NOT strictly better"
    headline = (f"QK-norm + undetached dyn q-temp vs both parents across head splits: {verdict}; "
                f"hd4 rescue {m['hd4_composite_rescue_fraction']}")
    print("\n" + headline)
    for row in m["paired_by_head_dim"]:
        print(f"  hd={row['head_dim']:3d}: base {row['baseline']:.4f}  qknorm {row['qknorm']:.4f} "
              f"({row['qknorm_minus_baseline']:+.4f})  composite {row['composite']:.4f} "
              f"({row['composite_minus_baseline']:+.4f} vs base, "
              f"{row['composite_minus_qknorm']:+.4f} vs qknorm) "
              f"{'OK' if row['beats_or_matches_both_parents'] else 'FAILS'}")

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
