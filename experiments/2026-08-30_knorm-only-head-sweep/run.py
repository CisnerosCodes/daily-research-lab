"""Is one-sided key-norm the strictly-better attention norm? Head-split sweep.

Lineage:
  2026-07-30_qknorm-head-dim: QK-norm turns the monotone head-dim tax into a U with
  optimum hd=32, but pays a cliff at hd=4 (+0.143) and a tax at nh=1 (+0.033).
  2026-08-02/06: the hd=4 cliff is per-token QUERY magnitude destruction (98% closed by
  restoring the undetached q-magnitude channel).
  2026-08-11: the composite fixes hd=4 but inherits the nh=1 tax -> NOT strictly better.
  2026-08-23: at nh=1 the decomposition INVERTS - k-norm-only beats baseline AND qknorm
  (-0.041, 3/3 seeds, 4-8x lower variance); the qknorm cost is a q x k interaction.

Tonight: the four norm arms (baseline / qknorm / qnorm_only / knorm_only) across the full
iso-parameter head-split family n_head x head_dim = 128, 3 seeds, byte-identical paired
inits. hd in {4,16,32,64} runs fresh; hd=128 (nh=1) is imported from the bit-exact 08-23
run (identical harness + env) after LIVE verification reruns.

Predictions registered up front:
  P1  k-norm-only shows NO hd=4 cliff (the cliff is q-side); qnorm_only inherits it.
  P2  one-sided arms converge toward full qknorm as heads multiply (interaction term
      shrinks with head_dim).
  P3  open verdict: does knorm_only beat qknorm at qknorm's own U-optimum hd=32 -
      and both parents at EVERY split (the "strictly better" claim)?

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
    """Pre-norm transformer block, byte-compatible with 2026-07-30/31/08-02/06/11/23.

    norm_q / norm_k: per-head RMS-norm on that side + learnable per-channel gain of
    length d. The dyn machinery of the 08-23 Block is kept structurally (asserts, flag
    layout) but every arm tonight runs with dyn_q = dyn_k = None, so the forward math for
    each (norm_q, norm_k) combination is bit-identical to the 08-23 arms of the same name.
    All extra parameters init deterministically (ones): paired inits by construction.
    """

    def __init__(self, d, n_head, d_ff, norm_q, norm_k, gains_learnable, dyn_q, dyn_k,
                 ema_momentum: float, ratio_clamp: float):
        super().__init__()
        assert d % n_head == 0
        assert dyn_q is None and dyn_k is None, "tonight's arms are norm-only"
        self.n_head = n_head
        self.head_dim = d // n_head
        self.norm_q, self.norm_k = norm_q, norm_k
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

    def _rms(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

    def _qkv(self, x, return_stats=False):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        shape4 = (B, T, self.n_head, self.head_dim)
        q4, k4 = q.view(*shape4), k.view(*shape4)
        rq_det = q4.pow(2).mean(-1).add(1e-12).sqrt().detach()   # (B, T, nh) pre-norm RMS
        rk_det = k4.pow(2).mean(-1).add(1e-12).sqrt().detach()
        if self.norm_q:
            q = self._rms(q4).reshape(B, T, D) * self.q_gain
        if self.norm_k:
            k = self._rms(k4).reshape(B, T, D) * self.k_gain
        q = q.view(*shape4)
        k = k.view(*shape4)
        out = (q.transpose(1, 2), k.transpose(1, 2), v.view(*shape4).transpose(1, 2))
        if return_stats:
            return out, {"r_q": rq_det, "r_k": rk_det}
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
    entropy, top-1, logit std, per-token q/k RMS CV."""
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
            r_n += stats["r_q"].shape[0]
            nb += 1
        mu = lg_mu / lg_n
        std = (lg_sq / lg_n - mu ** 2).clamp(min=0).sqrt()

        def cv(key):
            m_ = acc[key][0] / r_n
            s_ = (acc[key][1] / r_n - m_ ** 2).clamp(min=0).sqrt()
            return s_ / m_.clamp(min=1e-9)

        per_layer.append({
            "layer": li,
            "entropy_norm_per_head": [round(float(u), 4) for u in (H_sum / nb)],
            "top1_weight_per_head": [round(float(u), 4) for u in (top_sum / nb)],
            "logit_std_per_head": [round(float(u), 4) for u in std],
            "q_rms_token_cv_per_head": [round(float(u), 4) for u in cv("r_q")],
            "k_rms_token_cv_per_head": [round(float(u), 4) for u in cv("r_k")],
        })
    model.train()

    def flat(key):
        return [u for L in per_layer for u in L.get(key, [])]

    return {
        "per_layer": per_layer,
        "entropy_norm_mean": round(float(np.mean(flat("entropy_norm_per_head"))), 4),
        "top1_weight_mean": round(float(np.mean(flat("top1_weight_per_head"))), 4),
        "logit_std_mean": round(float(np.mean(flat("logit_std_per_head"))), 4),
        "q_rms_token_cv_mean": round(float(np.mean(flat("q_rms_token_cv_per_head"))), 4),
        "k_rms_token_cv_mean": round(float(np.mean(flat("k_rms_token_cv_per_head"))), 4),
    }


def train_one(vocab, arm, arm_flags, n_head, seed, p, train_ids, val_ids):
    set_seeds(seed)                    # identical init across ALL arms at a (n_head, seed)
    model = GPT(vocab, p["d_model"], p["n_layer"], n_head,
                p["d_ff_mult"] * p["d_model"], p["block_size"], arm_flags,
                p["ema_momentum"], p["ratio_clamp"])
    shared_sig = float(sum(float(q.detach().double().abs().sum())
                           for name, q in model.named_parameters()
                           if "gain" not in name))
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
        "head_dim": int(p["d_model"] // n_head),
        "n_params": n_params(model),
        "shared_init_signature": round(shared_sig, 6),
        "train_seconds": round(train_s, 1),
        "final_train_loss_ma50": round(float(np.mean(losses[-50:])), 4),
        "val_bpc": round(ev["bpc"], 5),
        "eval_chars": ev["eval_chars"],
        "attention": probe,
    }
    print(f"  [hd={rec['head_dim']:3d} {arm:10s} seed={seed}] P={rec['n_params']} "
          f"bpc={rec['val_bpc']:.4f} logit_std={probe['logit_std_mean']:.3f} "
          f"entN={probe['entropy_norm_mean']:.3f} top1={probe['top1_weight_mean']:.3f} "
          f"qCV={probe['q_rms_token_cv_mean']:.3f} kCV={probe['k_rms_token_cv_mean']:.3f} "
          f"({rec['train_seconds']}s)", flush=True)
    return rec


# -------------------------------------------------------------------------- analysis
def summarize(runs, imported, p, hd128_source):
    arms = p["arm_order"]
    head_dims = sorted({r["head_dim"] for r in runs} | {128})
    tol = p["tolerance_bpc"]

    by = {}   # (arm, hd) -> row
    for arm in arms:
        for hd in head_dims:
            rs = [r for r in runs if r["arm"] == arm and r["head_dim"] == hd]
            if rs:
                per_seed = {int(r["seed"]): r["val_bpc"] for r in rs}
                probes = {k: round(float(np.mean([r["attention"][k] for r in rs])), 4)
                          for k in ("entropy_norm_mean", "top1_weight_mean", "logit_std_mean",
                                    "q_rms_token_cv_mean", "k_rms_token_cv_mean")}
                src = "fresh"
            elif hd == 128 and imported is not None:
                per_seed = {int(s): v for s, v in imported[arm].items()}
                probes = {}
                src = hd128_source
            else:
                continue
            b = np.array(list(per_seed.values()))
            by[(arm, hd)] = {
                "val_bpc_per_seed": per_seed,
                "val_bpc_mean": round(float(b.mean()), 5),
                "seed_spread_bpc": round(float(b.max() - b.min()), 5),
                "source": src, **probes,
            }

    def mean(arm, hd):
        return by[(arm, hd)]["val_bpc_mean"]

    per_hd = []
    for hd in head_dims:
        base, qk = mean("baseline", hd), mean("qknorm", hd)
        qo, ko = mean("qnorm_only", hd), mean("knorm_only", hd)
        # paired per-seed comparisons
        k_beats_base = sum(by[("knorm_only", hd)]["val_bpc_per_seed"][s]
                           < by[("baseline", hd)]["val_bpc_per_seed"][s]
                           for s in by[("knorm_only", hd)]["val_bpc_per_seed"])
        k_beats_qk = sum(by[("knorm_only", hd)]["val_bpc_per_seed"][s]
                         < by[("qknorm", hd)]["val_bpc_per_seed"][s]
                         for s in by[("knorm_only", hd)]["val_bpc_per_seed"])
        per_hd.append({
            "head_dim": hd, "n_head": p["d_model"] // hd,
            "baseline": base, "qknorm": qk, "qnorm_only": qo, "knorm_only": ko,
            "qknorm_minus_baseline": round(qk - base, 5),
            "qnorm_only_minus_baseline": round(qo - base, 5),
            "knorm_only_minus_baseline": round(ko - base, 5),
            "knorm_only_minus_qknorm": round(ko - qk, 5),
            "interaction_bpc": round((qk - base) - (qo - base) - (ko - base), 5),
            "knorm_beats_baseline_seeds": f"{k_beats_base}/{len(by[('knorm_only', hd)]['val_bpc_per_seed'])}",
            "knorm_beats_qknorm_seeds": f"{k_beats_qk}/{len(by[('knorm_only', hd)]['val_bpc_per_seed'])}",
            "knorm_strictly_better_here": bool(ko <= base + tol and ko <= qk + tol),
        })

    # headline verdicts
    strictly = all(row["knorm_strictly_better_here"] for row in per_hd)
    row4 = next((r for r in per_hd if r["head_dim"] == 4), per_hd[0])
    row32 = next((r for r in per_hd if r["head_dim"] == 32), per_hd[0])
    curves = {arm: {r["head_dim"]: r[arm] for r in per_hd} for arm in arms}
    optima = {arm: min(curves[arm], key=curves[arm].get) for arm in arms}
    best = min(((arm, hd, v) for arm in arms for hd, v in curves[arm].items()),
               key=lambda t: t[2])

    return {
        "by_arm_hd": {f"{arm}@hd{hd}": v for (arm, hd), v in by.items()},
        "per_head_dim": per_hd,
        "tolerance_bpc": tol,
        "hd128_source": hd128_source,
        "P1_hd4_cliff": {
            "qknorm_cliff": row4["qknorm_minus_baseline"],
            "qnorm_only_cliff": row4["qnorm_only_minus_baseline"],
            "knorm_only_cliff": row4["knorm_only_minus_baseline"],
            "knorm_avoids_cliff": bool(row4["knorm_only_minus_baseline"] < row4["qknorm_minus_baseline"] / 2),
            "qside_prediction_holds": bool(
                row4["qnorm_only_minus_baseline"] > row4["knorm_only_minus_baseline"]),
        },
        "P2_interaction_by_hd": {r["head_dim"]: r["interaction_bpc"] for r in per_hd},
        "P3_u_optimum_head_to_head": {
            "qknorm_at_hd32": row32["qknorm"], "knorm_only_at_hd32": row32["knorm_only"],
            "delta": row32["knorm_only_minus_qknorm"],
            "knorm_only_optimum": {"head_dim": optima["knorm_only"],
                                   "bpc": curves["knorm_only"][optima["knorm_only"]]},
        },
        "optimum_per_arm": optima,
        "global_best": {"arm": best[0], "head_dim": best[1], "bpc": best[2]},
        "strictly_better_verdict": strictly,
    }


def make_chart(m, p, headline, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = p["arm_order"]
    labels = {"baseline": "baseline (no norm)", "qknorm": "QK-norm (both sides)",
              "qnorm_only": "q-norm only", "knorm_only": "k-norm only"}
    colors = {"baseline": "#444444", "qknorm": "#c0392b",
              "qnorm_only": "#e67e22", "knorm_only": "#8e44ad"}
    per_hd = m["per_head_dim"]
    hds = [r["head_dim"] for r in per_hd]
    xi = np.arange(len(hds))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))

    # panel 1: the four head-split curves with per-seed dots
    ax = axes[0]
    for arm in arms:
        ys = [r[arm] for r in per_hd]
        ax.plot(xi, ys, "-o", color=colors[arm], label=labels[arm], ms=4)
        for i, hd in enumerate(hds):
            seeds = list(m["by_arm_hd"][f"{arm}@hd{hd}"]["val_bpc_per_seed"].values())
            ax.plot([i] * len(seeds), seeds, ".", color=colors[arm], ms=3, alpha=0.45)
    ax.set_xticks(xi)
    ax.set_xticklabels([f"hd={h}\nnh={p['d_model']//h}" for h in hds], fontsize=8)
    ax.set_ylabel("val bits per character")
    ax.set_title("head-split curve per norm arm (3 seeds, paired inits)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # panel 2: deltas vs paired baseline
    ax = axes[1]
    w = 0.26
    for j, arm in enumerate([a for a in arms if a != "baseline"]):
        ds = [r[f"{arm}_minus_baseline"] for r in per_hd]
        ax.bar(xi + (j - 1) * w, ds, w, color=colors[arm], label=labels[arm])
    ax.axhline(0, color="k", lw=0.8)
    ax.axhspan(-p["tolerance_bpc"], p["tolerance_bpc"], color="gray", alpha=0.12,
               label=f"±{p['tolerance_bpc']} tol")
    ax.set_xticks(xi)
    ax.set_xticklabels([f"hd={h}" for h in hds], fontsize=8)
    ax.set_ylabel("Δ val bpc vs paired baseline")
    ax.set_title("P1: who pays the hd=4 cliff? (cliff was q-side)", fontsize=10)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25, axis="y")

    # panel 3: the q x k interaction term across the split
    ax = axes[2]
    inter = [r["interaction_bpc"] for r in per_hd]
    ax.bar(xi, inter, 0.5, color="#2980b9")
    for i, v in enumerate(inter):
        ax.text(i, v + (0.002 if v >= 0 else -0.006), f"{v:+.3f}", ha="center", fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xi)
    ax.set_xticklabels([f"hd={h}" for h in hds], fontsize=8)
    ax.set_ylabel("interaction bpc  (qk − q-only − k-only, vs base)")
    ax.set_title("P2: the q×k interaction term across the split", fontsize=10)
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle(headline, fontsize=10)
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
        p["head_dims_fresh"] = [32]
    train_ids, val_ids, vocab = load_data(cfg)
    print(f"data: {len(train_ids)} train / {len(val_ids)} val chars, vocab={vocab}")

    runs = []
    sigs = {}
    for hd in p["head_dims_fresh"]:
        n_head = p["d_model"] // hd
        for seed in p["seeds"]:
            for arm in p["arm_order"]:
                rec = train_one(vocab, arm, p["arms"][arm], n_head, seed, p, train_ids, val_ids)
                runs.append(rec)
                sigs.setdefault((hd, seed), set()).add(rec["shared_init_signature"])
    for key, ss in sigs.items():
        assert len(ss) == 1, f"shared-init mismatch at (hd, seed)={key}: {ss}"
    print("paired-init check passed: identical shared weights across all arms at every (hd, seed)")

    # --- hd=128 (nh=1): live verification, then import from 2026-08-23 ---
    imported = p["import_hd128_per_seed"]
    verification = []
    hd128_source = "imported_from_2026-08-23"
    if not SMOKE:
        print("\nverifying bit-exact replication of 2026-08-23 at hd=128 ...")
        ok = True
        for arm, seed in p["verify_pairs"]:
            rec = train_one(vocab, arm, p["arms"][arm], 1, seed, p, train_ids, val_ids)
            expect = imported[arm][seed]
            delta = rec["val_bpc"] - expect
            verification.append({"arm": arm, "seed": seed, "tonight": rec["val_bpc"],
                                 "expected_2026_08_23": expect, "delta": round(delta, 5)})
            print(f"    verify {arm} seed {seed}: {rec['val_bpc']} vs {expect} (Δ {delta:+.5f})")
            if abs(delta) > p["rep_tol_bpc"]:
                ok = False
        if not ok:
            print("  REPLICATION FAILED -> dropping the import, running hd=128 fresh")
            hd128_source = "fresh_after_failed_verification"
            for seed in p["seeds"]:
                for arm in p["arm_order"]:
                    runs.append(train_one(vocab, arm, p["arms"][arm], 1, seed, p, train_ids, val_ids))
            imported = None
    else:
        hd128_source = "imported_unverified_smoke"

    m = summarize(runs, imported, p, hd128_source)
    m["hd128_verification"] = verification

    p1, p3 = m["P1_hd4_cliff"], m["P3_u_optimum_head_to_head"]
    headline = (f"k-norm-only strictly better than baseline AND qknorm at every split: "
                f"{m['strictly_better_verdict']} — hd4 cliffs qk {p1['qknorm_cliff']:+.3f} / "
                f"q-only {p1['qnorm_only_cliff']:+.3f} / k-only {p1['knorm_only_cliff']:+.3f}; "
                f"at qknorm's U-optimum hd32: k-only {p3['delta']:+.4f} vs qknorm; "
                f"global best: {m['global_best']['arm']}@hd{m['global_best']['head_dim']} "
                f"{m['global_best']['bpc']:.4f}")
    print("\n" + headline)
    for r in m["per_head_dim"]:
        print(f"  hd={r['head_dim']:3d}: base {r['baseline']:.4f}  qk {r['qknorm']:.4f} "
              f"({r['qknorm_minus_baseline']:+.4f})  q-only {r['qnorm_only']:.4f} "
              f"({r['qnorm_only_minus_baseline']:+.4f})  k-only {r['knorm_only']:.4f} "
              f"({r['knorm_only_minus_baseline']:+.4f})  inter {r['interaction_bpc']:+.4f} "
              f"k>base {r['knorm_beats_baseline_seeds']} k>qk {r['knorm_beats_qknorm_seeds']}")

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
