"""Is the tiny-head cliff a mis-set attention temperature after all?

2026-09-01_kscale-adaptive-vs-static found that dividing keys by a per-head scale FROZEN at its
first-batch value beats the same scale tracked by an EMA (-0.134 vs -0.070 bpc at hd=4), and that
slower tracking is monotonically better. A frozen per-head scale is algebraically just a constant:

    k_hat = (k / r_t) * (r_t / s_h) = k / s_h        (s_h constant)

i.e. a per-head multiplier on the attention logits - a TEMPERATURE. But 2026-07-31 tested a learnable
per-head temperature on top of QK-norm and it refunded 2% of the cliff, with the optimizer leaving the
dial at tau ~ 1.04 when matching the baseline logit scale needed ~2.6.

Both cannot be right unless the binding constraint is WHERE THE CONSTANT STARTS, not whether it exists.
Tonight sweeps the constant directly and separates four things:

  baseline      c = 1                                     (anchor; replicates 08-30)
  c2/c4/c8/c16  keys multiplied by a fixed constant c      (the temperature curve)
  c_learn1      per-head learnable log c, init log(1)      (07-31's design, in this harness)
  c_learn4      per-head learnable log c, init log(4)      (same dial, better starting point)
  kinit_x4      key projection weight scaled x4 AT INIT    (folded into the weights: zero runtime cost)
  knorm_only    RMS-norm + gain on keys                    (anchor; the cliff arm)

Predictions registered up front:
  P1 the fixed-c curve has an interior optimum that MOVES with head width: large c at hd=4,
     c ~ 1 at hd=64 (if the cliff is a temperature effect, the default is only wrong at tiny heads).
  P2 c_learn1 refunds far less than the best fixed c (the dial is unreachable from 1 in 600 steps),
     while c_learn4 keeps most of c4's win - the constraint is the starting point, not learnability.
  P3 kinit_x4 matches c4 within tolerance: the whole intervention can be folded into the weight init.

Deterministic, CPU-only.  Usage:
  python run.py                      # full grid
  python run.py --head-dims 4 --tag x
  python run.py --merge
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


def load_data(cfg):
    p = cfg["params"]
    txt_path = HERE / "data" / "tinyshakespeare.txt"
    if not txt_path.exists():
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
    """Pre-norm block, byte-compatible with 2026-07-30 ... 2026-09-01.

    Arm spec (a dict from experiment.yaml):
      norm_k     : RMS-norm + per-channel gain on keys (the cliff arm)
      k_mult     : keys multiplied by a per-head constant c (logits scaled by c)
      c_learnable: make log(c) a learnable per-head parameter (init log(k_mult))
      k_init_mult: multiply the KEY THIRD of the qkv weight by this at init, no runtime op
    Queries are never touched. Extra parameters init deterministically: paired inits by construction.
    """

    def __init__(self, d, n_head, d_ff, spec):
        super().__init__()
        assert d % n_head == 0
        self.n_head, self.head_dim, self.d = n_head, d // n_head, d
        self.norm_k = bool(spec.get("norm_k", False))
        self.k_mult = float(spec.get("k_mult", 1.0))
        self.c_learnable = bool(spec.get("c_learnable", False))
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.fc = nn.Linear(d, d_ff, bias=False)
        self.out = nn.Linear(d_ff, d, bias=False)
        if self.norm_k:
            self.k_gain = nn.Parameter(torch.ones(d))
        # log c per head; a plain buffer when frozen so it never enters the optimizer
        if self.c_learnable:
            self.log_c = nn.Parameter(torch.full((n_head,), math.log(self.k_mult)))
        else:
            self.register_buffer("log_c", torch.full((n_head,), math.log(self.k_mult)))

    def _rms(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

    def _qkv(self, x, return_stats=False):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        shape4 = (B, T, self.n_head, self.head_dim)
        q4, k4 = q.view(*shape4), k.view(*shape4)
        rq_det = q4.pow(2).mean(-1).add(1e-12).sqrt().detach()
        rk_det = k4.pow(2).mean(-1).add(1e-12).sqrt().detach()
        if self.norm_k:
            k4 = self._rms(k4).reshape(B, T, D) * self.k_gain
            k4 = k4.view(*shape4)
        c = torch.exp(self.log_c).view(1, 1, self.n_head, 1)
        k4 = k4 * c
        out = (q4.transpose(1, 2), k4.transpose(1, 2), v.view(*shape4).transpose(1, 2))
        if return_stats:
            return out, {"r_q": rq_det, "r_k": rk_det,
                         "c": torch.exp(self.log_c).detach()}
        return out

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self._qkv(x)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, D))
        x = x + self.out(F.gelu(self.fc(self.ln2(x))))
        return x

    def attn_logits(self, x, with_stats=False):
        (q, k, _), stats = self._qkv(x, return_stats=True)
        logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if with_stats:
            return logits, stats
        return logits


class GPT(nn.Module):
    def __init__(self, vocab, d, n_layer, n_head, d_ff, block_size, spec):
        super().__init__()
        self.vocab, self.block_size = vocab, block_size
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(block_size, d)
        self.blocks = nn.ModuleList([Block(d, n_head, d_ff, spec) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.apply(self._init)
        for name, prm in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("out.weight"):
                nn.init.normal_(prm, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))
        with torch.no_grad():
            for blk in self.blocks:
                if hasattr(blk, "k_gain"):
                    blk.k_gain.fill_(1.0)
            # the folded-into-weights arm: scale the KEY third of qkv AFTER all other init,
            # so every arm still shares byte-identical draws from the RNG stream
            kim = float(spec.get("k_init_mult", 1.0))
            if kim != 1.0:
                for blk in self.blocks:
                    blk.qkv.weight[d:2 * d].mul_(kim)

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
def attention_probe(model, val_ids, p, at_init=False):
    model.eval()
    bs, B = p["block_size"], p["eval_batch"]
    n_blocks = min((len(val_ids) - 1) // bs, p["entropy_batches"] * B)
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
            logits, stats = blk.attn_logits(h, with_stats=True)
            probs = logits.masked_fill(~causal, float("-inf")).softmax(-1)
            ent = -(probs * torch.log(probs.clamp(min=1e-12))).sum(-1)
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
            "k_rms_mean_per_head": [round(float(u), 4) for u in (acc["r_k"][0] / r_n)],
            "c_per_head": [round(float(u), 4) for u in torch.exp(blk.log_c)],
        })
    model.train()

    def flat(key):
        return [u for L in per_layer for u in L.get(key, [])]

    return {"per_layer": per_layer,
            "entropy_norm_mean": round(float(np.mean(flat("entropy_norm_per_head"))), 4),
            "top1_weight_mean": round(float(np.mean(flat("top1_weight_per_head"))), 4),
            "logit_std_mean": round(float(np.mean(flat("logit_std_per_head"))), 4),
            "q_rms_token_cv_mean": round(float(np.mean(flat("q_rms_token_cv_per_head"))), 4),
            "k_rms_token_cv_mean": round(float(np.mean(flat("k_rms_token_cv_per_head"))), 4),
            "k_rms_mean": round(float(np.mean(flat("k_rms_mean_per_head"))), 4),
            "c_mean": round(float(np.mean(flat("c_per_head"))), 4)}


def train_one(vocab, arm, spec, head_dim, seed, p, train_ids, val_ids):
    set_seeds(seed)
    n_head = p["d_model"] // head_dim
    model = GPT(vocab, p["d_model"], p["n_layer"], n_head,
                p["d_ff_mult"] * p["d_model"], p["block_size"], spec)
    # signature over weights that must be shared: excludes gains, log_c, and the key third
    # (the kinit arm deliberately rescales that block, so it is reported separately)
    shared_sig = float(sum(float(q.detach().double().abs().sum())
                           for name, q in model.named_parameters()
                           if "gain" not in name and "log_c" not in name and "qkv" not in name))
    init_probe = attention_probe(model, val_ids, p)
    decay = [q for q in model.parameters() if q.requires_grad and q.dim() >= 2]
    nodecay = [q for q in model.parameters() if q.requires_grad and q.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": p["weight_decay"]},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=p["lr"], betas=(0.9, 0.95))
    rng = np.random.default_rng(seed)
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
    rec = {"arm": arm, "seed": int(seed), "n_head": int(n_head), "head_dim": int(head_dim),
           "n_params": n_params(model), "shared_init_signature": round(shared_sig, 6),
           "train_seconds": round(train_s, 1),
           "final_train_loss_ma50": round(float(np.mean(losses[-50:])), 4),
           "val_bpc": round(ev["bpc"], 5), "eval_chars": ev["eval_chars"],
           "attention": probe, "attention_at_init": init_probe}
    print(f"  [hd={head_dim:3d} {arm:12s} seed={seed}] bpc={rec['val_bpc']:.4f} "
          f"c={probe['c_mean']:.3f} init_logit_std={init_probe['logit_std_mean']:.3f} "
          f"logit_std={probe['logit_std_mean']:.3f} entN={probe['entropy_norm_mean']:.3f} "
          f"top1={probe['top1_weight_mean']:.3f} ({rec['train_seconds']}s)", flush=True)
    return rec


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
            row = {"val_bpc_per_seed": per_seed, "val_bpc_mean": round(float(b.mean()), 5),
                   "seed_spread_bpc": round(float(b.max() - b.min()), 5),
                   "n_params": rs[0]["n_params"]}
            for k in ("entropy_norm_mean", "top1_weight_mean", "logit_std_mean",
                      "k_rms_token_cv_mean", "k_rms_mean", "c_mean"):
                row[k] = round(float(np.mean([r["attention"][k] for r in rs])), 4)
            row["init_logit_std_mean"] = round(float(np.mean([r["attention_at_init"]["logit_std_mean"] for r in rs])), 4)
            row["init_entropy_norm_mean"] = round(float(np.mean([r["attention_at_init"]["entropy_norm_mean"] for r in rs])), 4)
            by.setdefault(arm, {})[hd] = row

    def mean(a, hd):
        return by.get(a, {}).get(hd, {}).get("val_bpc_mean")

    def seeds(a, hd):
        return by.get(a, {}).get(hd, {}).get("val_bpc_per_seed", {})

    per_hd = {}
    for hd in hds:
        present = [a for a in arms if hd in by.get(a, {})]
        if not present:
            continue
        best = min(present, key=lambda a: mean(a, hd))
        row = {"best_arm": best, "best_bpc": mean(best, hd), "arms": {}}
        for a in present:
            sb, sq = seeds(a, hd), seeds("baseline", hd)
            row["arms"][a] = {
                "bpc": mean(a, hd),
                "delta_vs_baseline": round(mean(a, hd) - mean("baseline", hd), 5) if sq else None,
                "delta_vs_best": round(mean(a, hd) - mean(best, hd), 5),
                "beats_baseline_seeds": f"{sum(sb[s] < sq[s] for s in sb if s in sq)}/{len(sq)}" if sq else None}
        # the fixed-c curve and its argmin
        cs = {a: p["arms"][a].get("k_mult", 1.0) for a in present
              if not p["arms"][a].get("c_learnable") and not p["arms"][a].get("norm_k")
              and p["arms"][a].get("k_init_mult", 1.0) == 1.0}
        if cs:
            best_c_arm = min(cs, key=lambda a: mean(a, hd))
            row["fixed_c_curve"] = {str(cs[a]): mean(a, hd) for a in sorted(cs, key=lambda a: cs[a])}
            row["best_fixed_c"] = cs[best_c_arm]
            row["best_fixed_c_bpc"] = mean(best_c_arm, hd)
        per_hd[hd] = row

    verdicts = {}
    for hd in hds:
        r = per_hd.get(hd)
        if not r:
            continue
        A = r["arms"]
        v = {}
        if "best_fixed_c" in r:
            v["P1_best_fixed_c"] = r["best_fixed_c"]
            v["P1_best_fixed_c_gain"] = round(r["best_fixed_c_bpc"] - A["baseline"]["bpc"], 5)
        if "c_learn1" in A and "c_learn4" in A and "best_fixed_c_bpc" in r:
            v["P2_learn_from_1_gain"] = A["c_learn1"]["delta_vs_baseline"]
            v["P2_learn_from_4_gain"] = A["c_learn4"]["delta_vs_baseline"]
            v["P2_learn1_recovers_frac_of_best_fixed"] = (
                round(A["c_learn1"]["delta_vs_baseline"] / v["P1_best_fixed_c_gain"], 3)
                if v.get("P1_best_fixed_c_gain") else None)
            v["P2_holds"] = bool(A["c_learn4"]["bpc"] < A["c_learn1"]["bpc"] - tol)
        if "kinit_x4" in A and "c4" in A:
            v["P3_kinit_minus_c4"] = round(A["kinit_x4"]["bpc"] - A["c4"]["bpc"], 5)
            v["P3_holds"] = bool(abs(v["P3_kinit_minus_c4"]) <= tol)
        verdicts[hd] = v

    rep = {"tol": p["rep_tol_bpc"], "pairs": [], "n_ok": 0, "n_checked": 0, "ok": True}
    for src, table in p.get("anchors", {}).items():
        for arm, per_hd_tab in table.items():
            for hd, per_seed in per_hd_tab.items():
                for s, expect in per_seed.items():
                    got = seeds(arm, int(hd)).get(int(s))
                    if got is None:
                        continue
                    d = round(got - expect, 5)
                    ok = abs(d) <= p["rep_tol_bpc"]
                    rep["pairs"].append({"source": src, "arm": arm, "head_dim": int(hd), "seed": int(s),
                                         "expected": expect, "tonight": got, "delta": d, "ok": ok})
                    rep["n_checked"] += 1
                    rep["n_ok"] += int(ok)
                    rep["ok"] = rep["ok"] and ok

    gb = min(((a, hd) for a in by for hd in by[a]), key=lambda t: mean(*t))
    return {"by_arm_hd": by, "per_head_dim": per_hd, "verdicts": verdicts,
            "global_best": {"arm": gb[0], "head_dim": gb[1], "bpc": mean(*gb)},
            "tolerance_bpc": tol, "replication_vs_parents": rep}


def make_chart(m, p, headline, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hds = [hd for hd in p["head_dims"] if hd in m["per_head_dim"]]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    colors = {4: "#c0392b", 8: "#e67e22", 16: "#8e44ad", 32: "#2980b9", 64: "#27ae60", 128: "#444444"}

    ax = axes[0]
    for hd in hds:
        curve = m["per_head_dim"][hd].get("fixed_c_curve", {})
        if not curve:
            continue
        cs = sorted(float(c) for c in curve)
        ax.plot(cs, [curve[str(c) if str(c) in curve else f"{c}"] for c in cs], "-o",
                color=colors.get(hd, "#666"), label=f"hd {hd}")
        b = m["per_head_dim"][hd]["best_fixed_c"]
        ax.plot([b], [m["per_head_dim"][hd]["best_fixed_c_bpc"]], "*", ms=16,
                color=colors.get(hd, "#666"))
    ax.set_xscale("log", base=2)
    ax.set_xlabel("fixed key multiplier c  (c = 1 is the default)")
    ax.set_ylabel("val bits per character")
    ax.set_title("the temperature curve moves with head width", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    order = [a for a in p["arm_order"] if a in m["by_arm_hd"]]
    x = np.arange(len(order))
    w = 0.8 / max(1, len(hds))
    for j, hd in enumerate(hds):
        ys = [m["per_head_dim"][hd]["arms"].get(a, {}).get("delta_vs_baseline", np.nan) for a in order]
        ax.bar(x + (j - (len(hds) - 1) / 2) * w, ys, w * 0.9, color=colors.get(hd, "#666"), label=f"hd {hd}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Δ val bpc vs baseline")
    ax.set_title("learnable-from-1 vs learnable-from-4 vs folded into init", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[2]
    for hd in hds:
        arms_ = [a for a in order if a in m["by_arm_hd"] and hd in m["by_arm_hd"][a]]
        xs = [m["by_arm_hd"][a][hd]["init_logit_std_mean"] for a in arms_]
        ys = [m["by_arm_hd"][a][hd]["val_bpc_mean"] for a in arms_]
        ax.plot(xs, ys, "o", color=colors.get(hd, "#666"), label=f"hd {hd}")
    ax.set_xscale("log")
    ax.set_xlabel("attention logit std AT INITIALIZATION")
    ax.set_ylabel("val bits per character")
    ax.set_title("loss vs the logit scale the model starts from", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.suptitle(headline, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def merge(p):
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
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    cfg = load_config()
    p = dict(cfg["params"])
    if SMOKE:
        p["steps"], p["warmup"], p["max_eval_blocks"], p["entropy_batches"] = 40, 4, 32, 1
        p["seeds"], p["head_dims"] = [0], [4]
    seeds = args.seeds if args.seeds is not None else p["seeds"]
    hds = args.head_dims if args.head_dims is not None else p["head_dims"]
    arms = args.arms if args.arms is not None else p["arm_order"]

    if args.merge:
        runs = merge(p)
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
        print("paired-init check passed (non-qkv weights identical across arms at every (hd, seed))")
        if args.tag:
            out = HERE / f"results_part_{args.tag}.json"
            json.dump({"runs": runs, "env": env_info(), "seconds": round(time.time() - t0, 1)},
                      open(out, "w"), indent=1)
            print(f"wrote {out} ({len(runs)} runs, {time.time() - t0:.0f}s)")
            return

    m = summarize(runs, p)
    parts = []
    for hd in p["head_dims"]:
        v = m["verdicts"].get(hd, {})
        if "P1_best_fixed_c" in v:
            parts.append(f"hd{hd}: best c={v['P1_best_fixed_c']:g} ({v['P1_best_fixed_c_gain']:+.3f})")
    headline = ("optimal fixed key multiplier by head width [" + "; ".join(parts) + "]; "
                f"global best {m['global_best']['arm']}@hd{m['global_best']['head_dim']} = {m['global_best']['bpc']:.4f}; "
                f"replication {m['replication_vs_parents']['n_ok']}/{m['replication_vs_parents']['n_checked']}")
    print("\n" + headline)
    for hd in p["head_dims"]:
        r = m["per_head_dim"].get(hd)
        if not r:
            continue
        print(f"  hd={hd:3d} best={r['best_arm']:12s} " + "  ".join(
            f"{a}={r['arms'][a]['bpc']:.4f}({r['arms'][a]['delta_vs_baseline']:+.3f})"
            for a in p["arm_order"] if a in r["arms"]))
        print(f"        verdicts: {json.dumps(m['verdicts'].get(hd, {}))}")

    results = {"id": cfg["id"], "git_commit": git_sha(), "seed": cfg["seed"],
               "duration_sec": round(time.time() - t0, 2), "smoke": SMOKE,
               "metrics": {"headline": headline, **m},
               "runs": runs, "env": env_info(), "config": cfg}
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=1)
    make_chart(m, p, headline, HERE / "chart.png")
    print(f"\nwrote results.json + chart.png in {results['duration_sec']}s")


if __name__ == "__main__":
    main()
