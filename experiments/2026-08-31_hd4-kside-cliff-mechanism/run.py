"""What does the key norm destroy at hd=4 - per-token magnitude or direction?

Lineage:
  2026-07-30_qknorm-head-dim: QK-norm pays a +0.143 cliff at hd=4.
  2026-08-02/06: cliff attributed to per-token QUERY magnitude destruction (undetached
  q-magnitude channel closes 98%) - but only ever tested on top of FULL qknorm.
  2026-08-30_knorm-only-head-sweep: side-swap - at hd=4 k-norm-only pays essentially the
  full cliff (+0.127 vs qknorm's +0.130) while q-norm-only is nearly free (+0.024).

Tonight: six k-side arms at hd=4 (nh=32), 3 seeds, byte-identical paired inits, that
factor the key norm into gain / magnitude-value / magnitude-gradient components:

  baseline          no k-touch                      (anchor, replicates 08-30)
  knorm_only        RMS-norm + learnable gain       (anchor, the cliff arm)
  knorm_nogain      RMS-norm, gain frozen at 1
  kgain_only        learnable gain, NO norm
  knorm_magrestore  RMS-norm + gain, then multiply back the DETACHED per-token k-RMS:
                    forward VALUE == kgain_only, but grad wrt k flows through the
                    normalized (radial-projected) geometry
  knorm_dynk        knorm_only + undetached per-head tau = clamp(r/EMA,1/c,c)^alpha
                    (the 08-06 magnitude-gradient channel, k-side, learnable alpha)

Predictions registered up front:
  P1  magnitude-VALUE account: magrestore recovers most of the cliff; kgain_only ~ base.
  P2  the gain is not the cliff: knorm_nogain ~ knorm_only (per 08-23 frozen-gain).
  P3  if 08-02's "key-side restoration hurts" was a full-qknorm artifact, undetached dynk
      HELPS on top of k-norm-only.

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
K_MODES = ("none", "norm_gain", "norm_nogain", "gain_only", "norm_gain_magrestore",
           "norm_gain_dynk")


class Block(nn.Module):
    """Pre-norm transformer block, byte-compatible with 2026-07-30/.../08-30.

    Only the KEY side is touched, parameterised by k_mode (see module docstring).
    All extra parameters/buffers init deterministically (ones): paired inits by
    construction; for k_mode in {none, norm_gain} the forward math is bit-identical to
    the 08-30 arms of the same name.
    """

    def __init__(self, d, n_head, d_ff, k_mode: str, ema_momentum: float, ratio_clamp: float):
        super().__init__()
        assert d % n_head == 0 and k_mode in K_MODES
        self.n_head = n_head
        self.head_dim = d // n_head
        self.k_mode = k_mode
        self.ema_momentum = ema_momentum
        self.ratio_clamp = ratio_clamp
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.fc = nn.Linear(d, d_ff, bias=False)
        self.out = nn.Linear(d_ff, d, bias=False)
        # init to ones: consumes no RNG, keeps shared weights bit-identical across arms
        if k_mode != "none":
            self.k_gain = nn.Parameter(torch.ones(d), requires_grad=(k_mode != "norm_nogain"))
        if k_mode == "norm_gain_dynk":
            self.k_alpha = nn.Parameter(torch.ones(n_head))
            self.register_buffer("k_rms_ema", torch.ones(n_head))
            self.register_buffer("k_ema_ready", torch.zeros(1))

    def _rms(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

    def _update_ema(self, r_det):
        ema, ready = self.k_rms_ema, self.k_ema_ready
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
        rq_det = q4.pow(2).mean(-1).add(1e-12).sqrt().detach()   # (B, T, nh) pre-norm RMS
        rk_grad = k4.pow(2).mean(-1).add(1e-12).sqrt()           # WITH grad (dynk uses it)
        rk_det = rk_grad.detach()
        tau = None
        m = self.k_mode
        if m in ("norm_gain", "norm_nogain", "norm_gain_magrestore", "norm_gain_dynk"):
            k4 = self._rms(k4)
            if m == "norm_gain_magrestore":
                k4 = k4 * rk_det.unsqueeze(-1)   # restore the VALUE, not the gradient
            elif m == "norm_gain_dynk":
                self._update_ema(rk_det)
                c = self.ratio_clamp
                ratio = (rk_grad / self.k_rms_ema.view(1, 1, -1)).clamp(1.0 / c, c)
                tau = torch.exp(self.k_alpha.view(1, 1, -1) * ratio.log())   # undetached
                k4 = k4 * tau.unsqueeze(-1)
            k = k4.reshape(B, T, D) * self.k_gain
        elif m == "gain_only":
            k = k4.reshape(B, T, D) * self.k_gain
        k = k.view(*shape4)
        out = (q4.transpose(1, 2), k.transpose(1, 2), v.view(*shape4).transpose(1, 2))
        if return_stats:
            return out, {"r_q": rq_det, "r_k": rk_det,
                         "tau_k": None if tau is None else tau.detach()}
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
    def __init__(self, vocab, d, n_layer, n_head, d_ff, block_size, k_mode,
                 ema_momentum, ratio_clamp):
        super().__init__()
        self.vocab, self.block_size = vocab, block_size
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(block_size, d)
        self.blocks = nn.ModuleList([
            Block(d, n_head, d_ff, k_mode, ema_momentum, ratio_clamp)
            for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.apply(self._init)
        for name, prm in self.named_parameters():   # nanoGPT residual-projection scaling
            if name.endswith("proj.weight") or name.endswith("out.weight"):
                nn.init.normal_(prm, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))
        with torch.no_grad():   # _init must not disturb the deterministic extras
            for blk in self.blocks:
                if k_mode != "none":
                    blk.k_gain.fill_(1.0)
                if k_mode == "norm_gain_dynk":
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
    """Trained-model attention statistics on fixed val batches (as in the parents):
    entropy, top-1, logit std, per-token q/k RMS CV, realised tau spread (dynk)."""
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
        tau_mu, tau_sq, tau_n = torch.zeros(nh), torch.zeros(nh), 0
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
            if stats.get("tau_k") is not None:
                tk = stats["tau_k"]
                tau_mu += tk.mean(dim=(0, 1)) * tk.shape[0]
                tau_sq += tk.pow(2).mean(dim=(0, 1)) * tk.shape[0]
                tau_n += tk.shape[0]
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
        if tau_n > 0:
            tm = tau_mu / tau_n
            ts = (tau_sq / tau_n - tm ** 2).clamp(min=0).sqrt()
            row["tau_k_mean_per_head"] = [round(float(u), 4) for u in tm]
            row["tau_k_std_per_head"] = [round(float(u), 4) for u in ts]
            row["k_alpha_per_head"] = [round(float(u), 4) for u in blk.k_alpha]
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
    if flat("tau_k_mean_per_head"):
        summ["tau_k_mean"] = round(float(np.mean(flat("tau_k_mean_per_head"))), 4)
        summ["tau_k_std_mean"] = round(float(np.mean(flat("tau_k_std_per_head"))), 4)
        summ["k_alpha_mean"] = round(float(np.mean(flat("k_alpha_per_head"))), 4)
    return summ


def train_one(vocab, arm, k_mode, seed, p, train_ids, val_ids):
    set_seeds(seed)                    # identical init across ALL arms at a seed
    n_head = p["d_model"] // p["head_dim"]
    model = GPT(vocab, p["d_model"], p["n_layer"], n_head,
                p["d_ff_mult"] * p["d_model"], p["block_size"], k_mode,
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
        "arm": arm, "seed": int(seed), "n_head": int(n_head),
        "head_dim": int(p["head_dim"]),
        "n_params": n_params(model),
        "shared_init_signature": round(shared_sig, 6),
        "train_seconds": round(train_s, 1),
        "final_train_loss_ma50": round(float(np.mean(losses[-50:])), 4),
        "val_bpc": round(ev["bpc"], 5),
        "eval_chars": ev["eval_chars"],
        "attention": probe,
    }
    extra = ""
    if "tau_k_mean" in probe:
        extra = f" tauK={probe['tau_k_mean']:.3f}±{probe['tau_k_std_mean']:.3f} aK={probe['k_alpha_mean']:.3f}"
    print(f"  [{arm:16s} seed={seed}] P={rec['n_params']} bpc={rec['val_bpc']:.4f} "
          f"logit_std={probe['logit_std_mean']:.3f} entN={probe['entropy_norm_mean']:.3f} "
          f"top1={probe['top1_weight_mean']:.3f} kCV={probe['k_rms_token_cv_mean']:.3f}"
          f"{extra} ({rec['train_seconds']}s)", flush=True)
    return rec


# -------------------------------------------------------------------------- analysis
def summarize(runs, p):
    arms = p["arm_order"]
    tol = p["tolerance_bpc"]

    by = {}
    for arm in arms:
        rs = [r for r in runs if r["arm"] == arm]
        if not rs:
            continue
        per_seed = {int(r["seed"]): r["val_bpc"] for r in rs}
        b = np.array(list(per_seed.values()))
        probes = {k: round(float(np.mean([r["attention"][k] for r in rs])), 4)
                  for k in ("entropy_norm_mean", "top1_weight_mean", "logit_std_mean",
                            "q_rms_token_cv_mean", "k_rms_token_cv_mean")}
        by[arm] = {"val_bpc_per_seed": per_seed,
                   "val_bpc_mean": round(float(b.mean()), 5),
                   "seed_spread_bpc": round(float(b.max() - b.min()), 5), **probes}
        if any("tau_k_mean" in r["attention"] for r in rs):
            by[arm]["tau_k_mean"] = round(float(np.mean(
                [r["attention"]["tau_k_mean"] for r in rs])), 4)
            by[arm]["k_alpha_mean"] = round(float(np.mean(
                [r["attention"]["k_alpha_mean"] for r in rs])), 4)

    base, cliff_arm = by["baseline"], by["knorm_only"]
    cliff = round(cliff_arm["val_bpc_mean"] - base["val_bpc_mean"], 5)

    def delta_and_recovery(arm):
        d = round(by[arm]["val_bpc_mean"] - base["val_bpc_mean"], 5)
        rec_frac = round(1.0 - d / cliff, 4) if cliff != 0 else None
        beats_knorm = sum(by[arm]["val_bpc_per_seed"][s] < cliff_arm["val_bpc_per_seed"][s]
                          for s in by[arm]["val_bpc_per_seed"])
        worse_than_base = sum(by[arm]["val_bpc_per_seed"][s] > base["val_bpc_per_seed"][s]
                              for s in by[arm]["val_bpc_per_seed"])
        n = len(by[arm]["val_bpc_per_seed"])
        return {"delta_vs_baseline": d, "cliff_recovery_frac": rec_frac,
                "beats_knorm_only_seeds": f"{beats_knorm}/{n}",
                "worse_than_baseline_seeds": f"{worse_than_base}/{n}"}

    decomp = {arm: delta_and_recovery(arm) for arm in arms if arm != "baseline"}

    # replication vs 2026-08-30 hd=4 (baseline + knorm_only per seed)
    rep = {"tol": p["rep_tol_bpc"], "pairs": [], "ok": True}
    for arm, table in p["expect_0830_hd4"].items():
        for s, expect in table.items():
            got = by[arm]["val_bpc_per_seed"].get(int(s))
            if got is None:
                continue
            d = round(got - expect, 5)
            rep["pairs"].append({"arm": arm, "seed": int(s), "tonight": got,
                                 "expected_2026_08_30": expect, "delta": d})
            if abs(d) > p["rep_tol_bpc"]:
                rep["ok"] = False

    mg, go, ng, dk = (decomp[a] for a in
                      ("knorm_magrestore", "kgain_only", "knorm_nogain", "knorm_dynk"))
    verdicts = {
        "P1_magnitude_value_account": {
            "magrestore_recovery_frac": mg["cliff_recovery_frac"],
            "kgain_only_delta": go["delta_vs_baseline"],
            "holds": bool(mg["cliff_recovery_frac"] is not None
                          and mg["cliff_recovery_frac"] > 0.5
                          and abs(go["delta_vs_baseline"]) <= tol),
        },
        "P2_gain_not_the_cliff": {
            "nogain_minus_knorm_only": round(
                by["knorm_nogain"]["val_bpc_mean"] - cliff_arm["val_bpc_mean"], 5),
            "holds": bool(abs(by["knorm_nogain"]["val_bpc_mean"]
                              - cliff_arm["val_bpc_mean"]) <= tol),
        },
        "P3_dynk_helps_without_qside": {
            "dynk_recovery_frac": dk["cliff_recovery_frac"],
            "dynk_delta_vs_knorm_only": round(
                by["knorm_dynk"]["val_bpc_mean"] - cliff_arm["val_bpc_mean"], 5),
            "holds": bool(by["knorm_dynk"]["val_bpc_mean"]
                          < cliff_arm["val_bpc_mean"] - tol),
        },
    }
    return {"by_arm": by, "cliff_bpc": cliff, "decomposition": decomp,
            "tolerance_bpc": tol, "replication_vs_2026_08_30": rep,
            "predictions": verdicts}


def make_chart(m, p, headline, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = p["arm_order"]
    labels = {"baseline": "baseline\n(no k-touch)", "knorm_only": "k-norm\n+gain",
              "knorm_nogain": "k-norm\nno gain", "kgain_only": "k-gain\nno norm",
              "knorm_magrestore": "k-norm ×\nmag-restore", "knorm_dynk": "k-norm\n+dynk"}
    colors = {"baseline": "#444444", "knorm_only": "#8e44ad", "knorm_nogain": "#b085c9",
              "kgain_only": "#27ae60", "knorm_magrestore": "#2980b9", "knorm_dynk": "#c0392b"}
    by = m["by_arm"]
    xi = np.arange(len(arms))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))

    ax = axes[0]
    for i, arm in enumerate(arms):
        ax.bar(i, by[arm]["val_bpc_mean"], 0.62, color=colors[arm])
        seeds = list(by[arm]["val_bpc_per_seed"].values())
        ax.plot([i] * len(seeds), seeds, "k.", ms=5, alpha=0.6)
    ax.axhline(by["baseline"]["val_bpc_mean"], color="#444444", lw=0.9, ls="--",
               label="baseline mean")
    ax.axhline(by["knorm_only"]["val_bpc_mean"], color="#8e44ad", lw=0.9, ls="--",
               label="k-norm-only mean (the cliff)")
    ax.set_xticks(xi)
    ax.set_xticklabels([labels[a] for a in arms], fontsize=8)
    ax.set_ylim(min(min(by[a]["val_bpc_per_seed"].values()) for a in arms) - 0.01,
                max(max(by[a]["val_bpc_per_seed"].values()) for a in arms) + 0.01)
    ax.set_ylabel("val bits per character (hd=4, nh=32)")
    ax.set_title("six k-side arms at hd=4 (3 seeds, paired inits)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    darms = [a for a in arms if a != "baseline"]
    fr = [m["decomposition"][a]["cliff_recovery_frac"] for a in darms]
    ax.bar(np.arange(len(darms)), fr, 0.55, color=[colors[a] for a in darms])
    for i, v in enumerate(fr):
        ax.text(i, v + (0.03 if v >= 0 else -0.08), f"{v:+.2f}", ha="center", fontsize=9)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(1, color="#27ae60", lw=0.8, ls="--", label="full recovery")
    ax.set_xticks(np.arange(len(darms)))
    ax.set_xticklabels([labels[a] for a in darms], fontsize=8)
    ax.set_ylabel("cliff recovery fraction  (1 − Δ/cliff)")
    ax.set_title(f"who gives the cliff back? (cliff = {m['cliff_bpc']:+.3f} bpc)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[2]
    w = 0.38
    kcv = [by[a]["k_rms_token_cv_mean"] for a in arms]
    lstd = [by[a]["logit_std_mean"] for a in arms]
    ax.bar(xi - w / 2, kcv, w, color="#2980b9", label="post-hoc k-RMS token CV (pre-norm)")
    ax2 = ax.twinx()
    ax2.bar(xi + w / 2, lstd, w, color="#e67e22", label="attn logit std")
    ax.set_xticks(xi)
    ax.set_xticklabels([labels[a] for a in arms], fontsize=8)
    ax.set_ylabel("k-RMS token CV", color="#2980b9")
    ax2.set_ylabel("attn logit std", color="#e67e22")
    ax.set_title("probes: key-magnitude spread vs logit sharpness", fontsize=10)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle(headline, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
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
            rec = train_one(vocab, arm, p["arms"][arm], seed, p, train_ids, val_ids)
            runs.append(rec)
            sigs.setdefault(seed, set()).add(rec["shared_init_signature"])
    for seed, ss in sigs.items():
        assert len(ss) == 1, f"shared-init mismatch at seed={seed}: {ss}"
    print("paired-init check passed: identical shared weights across all arms at every seed")

    m = summarize(runs, p)
    P = m["predictions"]
    headline = (f"hd=4 key-norm cliff {m['cliff_bpc']:+.3f} bpc decomposed: "
                f"mag-restore recovers {P['P1_magnitude_value_account']['magrestore_recovery_frac']:+.2f}, "
                f"gain-only Δ {m['decomposition']['kgain_only']['delta_vs_baseline']:+.3f}, "
                f"no-gain Δvs-knorm {P['P2_gain_not_the_cliff']['nogain_minus_knorm_only']:+.3f}, "
                f"dynk recovers {P['P3_dynk_helps_without_qside']['dynk_recovery_frac']:+.2f} "
                f"(P1 {P['P1_magnitude_value_account']['holds']} / "
                f"P2 {P['P2_gain_not_the_cliff']['holds']} / "
                f"P3 {P['P3_dynk_helps_without_qside']['holds']}); "
                f"replication vs 08-30: {m['replication_vs_2026_08_30']['ok']}")
    print("\n" + headline)
    for arm in p["arm_order"]:
        b = m["by_arm"][arm]
        d = m["decomposition"].get(arm, {})
        print(f"  {arm:16s} bpc={b['val_bpc_mean']:.4f} spread={b['seed_spread_bpc']:.4f} "
              f"Δ={d.get('delta_vs_baseline', 0):+.4f} rec={d.get('cliff_recovery_frac', '-')}")

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
