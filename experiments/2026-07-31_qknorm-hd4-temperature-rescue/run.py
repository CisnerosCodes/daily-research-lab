"""Per-head learnable temperature on top of QK-norm: the causal test of the hd=4 cliff.

2026-07-30_qknorm-head-dim: QK-norm helps every iso-param head split of d_model=128
except head_dim=4, where it costs +0.143 bpc. The probes suggested a mechanism: with
unit-RMS q/k channels a head's pre-softmax logits are bounded by ~sqrt(head_dim) x gain,
so 4-dim qknorm heads cannot sharpen (entropy 0.90 vs 0.74 baseline, top-1 weight 0.107
vs 0.232, trained logit std pinned at 1.27 vs a 0.22-9.24 per-head spread in the
baseline). But that was an inference from correlational probes, not an intervention.

This experiment does the intervention. Third arm "qknorm_temp" = QK-norm exactly as
before PLUS a per-head, per-layer learnable log-temperature: logits are scaled by
exp(log_tau_h) (log_tau init 0 -> tau init 1, RNG-free, excluded from weight decay,
same parametrisation the QK-norm literature itself uses as a learnable scale g). This
restores the sharpness dial that RMS-normalisation removed while keeping q/k drift
bounded — the confound the baseline suffers from stays controlled.

Readouts, at the cliff (32 heads x 4 dims) and at the qknorm optimum (4 heads x 32 dims):
  - rescue fraction at hd=4: (qknorm - qknorm_temp) / (qknorm - baseline). 1.0 = tau
    fully refunds the +0.143 cliff; 0 = the cliff is not about temperature.
  - do-no-harm at hd=32: qknorm_temp must keep (or beat) qknorm's -0.099 win.
  - mechanism probes: did tau actually restore sharpness (entropy/top-1/logit std back
    toward baseline), and what taus did the heads learn (spread = per-head temperatures
    are real; all ~1 = the dial was not even used).

Controls carried over from the parent harness: identical shared-weight init across ALL
arm x config cells at a given seed (gains and log_taus init deterministically, consume
no RNG), shared batch stream per seed, exact iso-param across configs within each arm
(gain vectors are length d_model; log_tau has n_head*head_dim/head_dim... no: length
n_head per layer, so the temp arm differs across configs by |n_head difference| params —
reported, and 56 params out of 424k cannot carry a 0.1-bpc effect; the paired
comparisons that matter (arm vs arm at fixed config) are exactly iso-architecture).

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
    """Pre-norm transformer block, identical to 2026-07-30_qknorm-head-dim, plus an
    optional per-head learnable log-temperature on top of QK-norm.

    qknorm: per-head RMS-norm on q and k with learnable per-channel gains of length d
    (same shape for every split -> iso-param across configs), then 1/sqrt(head_dim).
    temp:   q additionally scaled per head by exp(log_tau_h), log_tau init 0. Because q
    is scaled before the dot product, both the SDPA fast path and the probe see the
    identical effective logits exp(log_tau_h) * <q_norm, k_norm> / sqrt(head_dim).
    """

    def __init__(self, d, n_head, d_ff, qknorm: bool, temp: bool):
        super().__init__()
        assert d % n_head == 0
        assert qknorm or not temp, "temp arm is defined on top of qknorm"
        self.n_head = n_head
        self.head_dim = d // n_head
        self.qknorm = qknorm
        self.temp = temp
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.fc = nn.Linear(d, d_ff, bias=False)
        self.out = nn.Linear(d_ff, d, bias=False)
        if qknorm:  # init to ones: consumes no RNG, keeps shared weights bit-identical
            self.q_gain = nn.Parameter(torch.ones(d))
            self.k_gain = nn.Parameter(torch.ones(d))
        if temp:    # init to zeros (tau=1): consumes no RNG
            self.log_tau = nn.Parameter(torch.zeros(n_head))

    def _rms(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

    def _qkv(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        shape4 = (B, T, self.n_head, self.head_dim)
        if self.qknorm:
            q = self._rms(q.view(*shape4)).reshape(B, T, D) * self.q_gain
            k = self._rms(k.view(*shape4)).reshape(B, T, D) * self.k_gain
        q = q.view(*shape4)
        if self.temp:
            q = q * torch.exp(self.log_tau).view(1, 1, self.n_head, 1)
        return (q.transpose(1, 2),
                k.view(*shape4).transpose(1, 2),
                v.view(*shape4).transpose(1, 2))

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self._qkv(x)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, D))
        x = x + self.out(F.gelu(self.fc(self.ln2(x))))
        return x

    def attn_logits(self, x):
        """Explicit pre-softmax causal logits, (B, n_head, T, T). Probe only."""
        q, k, _ = self._qkv(x)
        return (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)


class GPT(nn.Module):
    def __init__(self, vocab, d, n_layer, n_head, d_ff, block_size, qknorm, temp):
        super().__init__()
        self.vocab, self.block_size = vocab, block_size
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(block_size, d)
        self.blocks = nn.ModuleList([Block(d, n_head, d_ff, qknorm, temp) for _ in range(n_layer)])
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
                if temp:
                    blk.log_tau.zero_()

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
    """Trained-model attention statistics on fixed val batches (as in the parent run)."""
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
        nb = 0
        for s in range(0, n_blocks, B):
            xb = torch.from_numpy(xs[s:s + B])
            h = model.embed(xb)
            for j in range(li):
                h = model.blocks[j](h)
            logits = blk.attn_logits(h)              # (B, nh, T, T)
            probs = logits.masked_fill(~causal, float("-inf")).softmax(-1)
            ent = -(probs * torch.log(probs.clamp(min=1e-12))).sum(-1)   # (B, nh, T)
            H_sum += (ent / norm)[:, :, keep].mean(dim=(0, 2))
            top_sum += probs.max(-1).values[:, :, keep].mean(dim=(0, 2))
            lv = logits.masked_select(causal.view(1, 1, bs, bs)).view(logits.shape[0], blk.n_head, -1)
            lg_mu += lv.mean(dim=(0, 2)) * lv.shape[0]
            lg_sq += lv.pow(2).mean(dim=(0, 2)) * lv.shape[0]
            lg_n += lv.shape[0]
            nb += 1
        mu = lg_mu / lg_n
        std = (lg_sq / lg_n - mu ** 2).clamp(min=0).sqrt()
        entry = {
            "layer": li,
            "entropy_norm_per_head": [round(float(u), 4) for u in (H_sum / nb)],
            "top1_weight_per_head": [round(float(u), 4) for u in (top_sum / nb)],
            "logit_std_per_head": [round(float(u), 4) for u in std],
        }
        if blk.temp:
            entry["tau_per_head"] = [round(float(u), 4) for u in torch.exp(blk.log_tau)]
        per_layer.append(entry)
    model.train()
    allH = [u for L in per_layer for u in L["entropy_norm_per_head"]]
    allT = [u for L in per_layer for u in L["top1_weight_per_head"]]
    allS = [u for L in per_layer for u in L["logit_std_per_head"]]
    out = {
        "per_layer": per_layer,
        "entropy_norm_mean": round(float(np.mean(allH)), 4),
        "top1_weight_mean": round(float(np.mean(allT)), 4),
        "logit_std_mean": round(float(np.mean(allS)), 4),
        "logit_std_min_head": round(float(np.min(allS)), 4),
        "logit_std_max_head": round(float(np.max(allS)), 4),
    }
    allTau = [u for L in per_layer for u in L.get("tau_per_head", [])]
    if allTau:
        out["tau_mean"] = round(float(np.mean(allTau)), 4)
        out["tau_min"] = round(float(np.min(allTau)), 4)
        out["tau_max"] = round(float(np.max(allTau)), 4)
    return out


def train_one(vocab, arm, n_head, seed, p, train_ids, val_ids):
    head_dim = p["d_model"] // n_head
    qknorm = arm in ("qknorm", "qknorm_temp")
    temp = arm == "qknorm_temp"
    set_seeds(seed)                    # identical init across ALL (arm, n_head) at a seed
    model = GPT(vocab, p["d_model"], p["n_layer"], n_head,
                p["d_ff_mult"] * p["d_model"], p["block_size"], qknorm, temp)
    shared_sig = float(sum(float(q.detach().double().abs().sum())
                           for name, q in model.named_parameters()
                           if "gain" not in name and "tau" not in name))
    decay = [q for q in model.parameters() if q.dim() >= 2]
    nodecay = [q for q in model.parameters() if q.dim() < 2]   # gains + log_tau: no WD
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
    tau_info = f" tau=[{probe.get('tau_min')},{probe.get('tau_max')}]" if temp else ""
    print(f"  [{arm:11s} hd={head_dim:3d} nh={n_head:2d} seed={seed}] P={rec['n_params']} "
          f"bpc={rec['val_bpc']:.4f} logit_std={probe['logit_std_mean']:.3f} "
          f"entN={probe['entropy_norm_mean']:.3f} top1={probe['top1_weight_mean']:.3f}"
          f"{tau_info} ({rec['train_seconds']}s)", flush=True)
    return rec


# -------------------------------------------------------------------------- analysis
def summarize(runs, arms, configs):
    out = {}
    for arm in arms:
        rows = []
        for nh, hd in configs:
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
                "logit_std_min_head": round(float(np.min([r["attention"]["logit_std_min_head"] for r in rs])), 4),
                "logit_std_max_head": round(float(np.max([r["attention"]["logit_std_max_head"] for r in rs])), 4),
                "train_seconds_mean": round(float(np.mean([r["train_seconds"] for r in rs])), 1),
            }
            taus = [r["attention"].get("tau_mean") for r in rs if "tau_mean" in r["attention"]]
            if taus:
                row["tau_mean"] = round(float(np.mean(taus)), 4)
                row["tau_min"] = round(float(np.min([r["attention"]["tau_min"] for r in rs])), 4)
                row["tau_max"] = round(float(np.max([r["attention"]["tau_max"] for r in rs])), 4)
            rows.append(row)
        out[arm] = rows
    return out


def make_chart(sums, runs, p, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = p["arms"]
    labels = {"baseline": "baseline (no norm)", "qknorm": "QK-norm", "qknorm_temp": "QK-norm + per-head tau"}
    colors = {"baseline": "#444444", "qknorm": "#c0392b", "qknorm_temp": "#2c7fb8"}
    configs = p["configs"]
    seeds = p["seeds"]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9))

    # panel 1: bpc by arm at each config
    ax = axes[0]
    width = 0.26
    for ci, (nh, hd) in enumerate(configs):
        for ai, arm in enumerate(arms):
            s = sums[arm][ci]
            xpos = ci + (ai - 1) * width
            ax.bar(xpos, s["val_bpc_mean"], width * 0.92, color=colors[arm],
                   label=labels[arm] if ci == 0 else None)
            ys = s["val_bpc_per_seed"]
            ax.plot([xpos] * len(ys), ys, "ko", ms=4, alpha=0.6, zorder=3)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([f"hd={hd}\n(nh={nh})" for nh, hd in configs])
    ax.set_ylabel("val bits per character")
    lo = min(s["val_bpc_min"] for a in arms for s in sums[a]) - 0.02
    hi = max(s["val_bpc_max"] for a in arms for s in sums[a]) + 0.02
    ax.set_ylim(lo, hi)
    ax.set_title("Does a learnable per-head tau rescue the hd=4 cliff?\n(dots = seeds)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    # panel 2: sharpness probes at hd=4
    ax = axes[1]
    ci4 = [i for i, (nh, hd) in enumerate(configs) if hd == 4][0]
    probes = ["entropy_norm_mean", "top1_weight_mean"]
    pl = {"entropy_norm_mean": "attn entropy (norm.)", "top1_weight_mean": "top-1 weight"}
    x = np.arange(len(probes))
    for ai, arm in enumerate(arms):
        s = sums[arm][ci4]
        vals = [s[pr] for pr in probes]
        ax.bar(x + (ai - 1) * width, vals, width * 0.92, color=colors[arm], label=labels[arm])
    ax.set_xticks(x)
    ax.set_xticklabels([pl[pr] for pr in probes])
    ax.set_title("Sharpness at hd=4: did tau restore the dial?")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    # panel 3: learned taus per head (temp arm), both configs
    ax = axes[2]
    for ci, (nh, hd) in enumerate(configs):
        taus = []
        for r in runs:
            if r["arm"] == "qknorm_temp" and r["n_head"] == nh:
                for L in r["attention"]["per_layer"]:
                    taus.extend(L.get("tau_per_head", []))
        xs = np.full(len(taus), ci) + np.linspace(-0.18, 0.18, max(len(taus), 2))[:len(taus)]
        ax.plot(xs, taus, "o", ms=4, alpha=0.55, color="#2c7fb8")
    ax.axhline(1.0, color="k", lw=0.8, ls="--", label="init tau=1")
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([f"hd={hd}" for nh, hd in configs])
    ax.set_ylabel("learned tau = exp(log_tau)")
    ax.set_yscale("log")
    ax.set_title("What temperatures did the heads learn?\n(all heads x layers x seeds)")
    ax.grid(alpha=0.3, which="both", axis="y")
    ax.legend(fontsize=8)

    fig.suptitle("Per-head learnable temperature on top of QK-norm (d_model=128, tiny-shakespeare)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
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
            for seed in p["seeds"]:
                runs.append(train_one(vocab, arm, nh, seed, p, train_ids, val_ids))

    sig_ok = all(
        len({r["shared_init_signature"] for r in runs if r["seed"] == sd}) == 1
        for sd in p["seeds"])

    sums = summarize(runs, p["arms"], p["configs"])

    def mean_bpc(arm, hd):
        return [s["val_bpc_mean"] for s in sums[arm] if s["head_dim"] == hd][0]

    def seed_spread(arm, hd):
        return [s["seed_spread_bpc"] for s in sums[arm] if s["head_dim"] == hd][0]

    # headline: the rescue fraction at the cliff
    cliff = mean_bpc("qknorm", 4) - mean_bpc("baseline", 4)          # parent: +0.143
    rescued = mean_bpc("qknorm", 4) - mean_bpc("qknorm_temp", 4)     # how much tau refunds
    rescue_fraction = rescued / cliff if abs(cliff) > 1e-9 else None
    temp_vs_baseline_hd4 = mean_bpc("qknorm_temp", 4) - mean_bpc("baseline", 4)

    # per-seed paired rescue (identical init + batch stream make these true pairs)
    paired = []
    for sd in p["seeds"]:
        b = [r["val_bpc"] for r in runs if r["arm"] == "baseline" and r["head_dim"] == 4 and r["seed"] == sd][0]
        q = [r["val_bpc"] for r in runs if r["arm"] == "qknorm" and r["head_dim"] == 4 and r["seed"] == sd][0]
        t = [r["val_bpc"] for r in runs if r["arm"] == "qknorm_temp" and r["head_dim"] == 4 and r["seed"] == sd][0]
        paired.append({"seed": sd, "cliff_qknorm_minus_baseline": round(q - b, 5),
                       "rescue_qknorm_minus_temp": round(q - t, 5),
                       "temp_minus_baseline": round(t - b, 5)})

    # do-no-harm at the qknorm optimum
    harm_hd32 = mean_bpc("qknorm_temp", 32) - mean_bpc("qknorm", 32)

    s4t = [s for s in sums["qknorm_temp"] if s["head_dim"] == 4][0]
    s4q = [s for s in sums["qknorm"] if s["head_dim"] == 4][0]
    s4b = [s for s in sums["baseline"] if s["head_dim"] == 4][0]

    metrics = {
        "headline": "rescue fraction of the hd=4 qknorm cliff by a per-head learnable temperature",
        "identical_shared_init_per_seed": sig_ok,
        "shared_batch_stream": True,
        "n_params_by_arm_config": {f"{r['arm']}_hd{r['head_dim']}": r["n_params"]
                                   for r in runs if r["seed"] == p["seeds"][0]},
        "cliff_bpc_qknorm_minus_baseline_hd4": round(cliff, 5),
        "parent_cliff_bpc_2026_07_30": 0.14299,
        "rescue_bpc_qknorm_minus_temp_hd4": round(rescued, 5),
        "rescue_fraction_of_cliff": round(rescue_fraction, 4) if rescue_fraction is not None else None,
        "temp_minus_baseline_bpc_hd4": round(temp_vs_baseline_hd4, 5),
        "seed_spread_bpc_hd4_by_arm": {a: seed_spread(a, 4) for a in p["arms"]},
        "paired_per_seed_hd4": paired,
        "do_no_harm_hd32_temp_minus_qknorm_bpc": round(harm_hd32, 5),
        "bpc_by_arm_config": {a: {f"hd{s['head_dim']}": s["val_bpc_mean"] for s in sums[a]} for a in p["arms"]},
        "sharpness_at_hd4": {
            "entropy_norm": {"baseline": s4b["entropy_norm_mean"], "qknorm": s4q["entropy_norm_mean"],
                             "qknorm_temp": s4t["entropy_norm_mean"]},
            "top1_weight": {"baseline": s4b["top1_weight_mean"], "qknorm": s4q["top1_weight_mean"],
                            "qknorm_temp": s4t["top1_weight_mean"]},
            "logit_std_mean": {"baseline": s4b["logit_std_mean"], "qknorm": s4q["logit_std_mean"],
                               "qknorm_temp": s4t["logit_std_mean"]},
            "logit_std_per_head_range": {"baseline": [s4b["logit_std_min_head"], s4b["logit_std_max_head"]],
                                         "qknorm": [s4q["logit_std_min_head"], s4q["logit_std_max_head"]],
                                         "qknorm_temp": [s4t["logit_std_min_head"], s4t["logit_std_max_head"]]},
        },
        "learned_tau": {f"hd{s['head_dim']}": {"mean": s.get("tau_mean"), "min": s.get("tau_min"),
                                               "max": s.get("tau_max")}
                        for s in sums["qknorm_temp"]},
        "by_config": sums,
        "train_steps": p["steps"],
        "tokens_per_run": p["steps"] * p["batch_size"] * p["block_size"],
        "seeds": p["seeds"],
        "n_runs": len(runs),
        "runs": runs,
    }

    make_chart(sums, runs, p, HERE / "chart.png")

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
          f"| qknorm+tau {mean_bpc('qknorm_temp', 4):.4f}")
    print(f"hd=32 baseline {mean_bpc('baseline', 32):.4f} | qknorm {mean_bpc('qknorm', 32):.4f} "
          f"| qknorm+tau {mean_bpc('qknorm_temp', 32):.4f}")
    print(f"cliff {cliff:+.4f} bpc, rescued {rescued:+.4f} -> rescue fraction {rescue_fraction:.3f}"
          if rescue_fraction is not None else "cliff ~0, rescue fraction undefined")
    print(f"temp - baseline at hd=4: {temp_vs_baseline_hd4:+.4f} bpc; "
          f"do-no-harm at hd=32 (temp - qknorm): {harm_hd32:+.4f} bpc")
    print(f"total {results['duration_sec']:.0f}s")


if __name__ == "__main__":
    main()
