"""RetNet retention at nano scale on a tiny-shakespeare char LM.

Three questions, one run:

1. DUALITY. Hand-rolled multi-scale retention implemented TWICE - the parallel form
   (masked QK^T Hadamard a geometric decay matrix D[n,m] = gamma^(n-m)) and the recurrent
   form (state recurrence S_t = gamma * S_{t-1} + k_t^T v_t, y_t = q_t S_t). Same weights,
   same batch -> max abs logit deviation. The paper asserts these are the same function;
   we measure how same.
2. RETENTION vs ATTENTION. Iso-parameter (d_ff of the attention arm binary-searched to
   match), iso-step, iso-data-stream. Metric: val bits/char.
3. DECAY SCHEDULE ABLATION at matched everything: (i) the paper's multi-scale geometric
   gammas 1 - 2^-(5+h), (ii) all heads the same gamma = 0.97, (iii) gammas all 1.0
   (no decay = pure causal linear attention), (iv) learned gammas (sigmoid-parameterised,
   initialised at the multi-scale values).

Deterministic, CPU-only, single thread. Writes results.json + chart.png.

Usage:  python run.py
"""
import json, math, os, random, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "tinyshakespeare.txt"
DATA_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
            "master/data/tinyshakespeare/input.txt")

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)   # 2 shared cores on this box - be a good neighbour


# ----------------------------- plumbing ------------------------------------
def set_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    info = {"python": sys.version.split()[0]}
    for mod in ("numpy", "torch", "matplotlib"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:
            pass
    return info


def get_data():
    if not DATA.exists():
        DATA.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(DATA_URL, DATA)
    return DATA.read_text()


# ----------------------------- mixers --------------------------------------
class MultiScaleRetention(nn.Module):
    """RetNet multi-scale retention (arXiv 2307.08621), hand-rolled.

    Both forms are exposed and are mathematically identical:
      parallel:  Y = (Q K^T . D) V,  D[n,m] = gamma^(n-m) for n>=m else 0
      recurrent: S_t = gamma S_{t-1} + k_t^T v_t,  y_t = q_t S_t

    Deviation from the paper: no xPos rotation on Q/K (the model uses learned absolute
    position embeddings instead, shared with the attention arm so the arms stay matched).
    Everything else is the paper's block: headwise GroupNorm on the retention output,
    a swish gate, then the output projection.
    """

    def __init__(self, d, h, gammas, learn_gamma=False):
        super().__init__()
        self.h, self.dh, self.d = h, d // h, d
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wg = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        self.gn = nn.GroupNorm(h, d)          # headwise normalisation
        self.scale = 1.0 / math.sqrt(self.dh)
        g = torch.as_tensor(gammas, dtype=torch.float32)
        self.learn_gamma = learn_gamma
        if learn_gamma:
            # sigmoid-parameterised, initialised at the given gammas
            self.theta = nn.Parameter(torch.log(g / (1.0 - g)))
        else:
            self.register_buffer("gamma_buf", g)

    def gammas(self):
        return torch.sigmoid(self.theta) if self.learn_gamma else self.gamma_buf

    def _qkv(self, x):
        B, T, D = x.shape
        q = self.wq(x).view(B, T, self.h, self.dh).transpose(1, 2) * self.scale
        k = self.wk(x).view(B, T, self.h, self.dh).transpose(1, 2)
        v = self.wv(x).view(B, T, self.h, self.dh).transpose(1, 2)
        return q, k, v

    def _post(self, x, y):
        """headwise GroupNorm -> swish gate -> output projection. Shared by both forms."""
        B, T, D = x.shape
        y = y.transpose(1, 2).reshape(B, T, D)
        y = self.gn(y.reshape(B * T, D)).reshape(B, T, D)
        return self.wo(F.silu(self.wg(x)) * y)

    def forward_parallel(self, x):
        B, T, D = x.shape
        q, k, v = self._qkv(x)
        gam = self.gammas().clamp(min=1e-6, max=1.0)
        idx = torch.arange(T, device=x.device)
        delta = (idx[:, None] - idx[None, :]).float()          # n - m
        causal = (delta >= 0).float()
        # D[h,n,m] = gamma_h^(n-m) on the causal triangle, 0 above it
        Dm = torch.exp(delta.clamp(min=0.0)[None] * torch.log(gam)[:, None, None]) * causal[None]
        y = ((q @ k.transpose(-2, -1)) * Dm[None]) @ v
        return self._post(x, y)

    def forward_recurrent(self, x):
        B, T, D = x.shape
        q, k, v = self._qkv(x)
        gam = self.gammas().clamp(min=1e-6, max=1.0).view(1, self.h, 1, 1)
        S = torch.zeros(B, self.h, self.dh, self.dh, dtype=x.dtype, device=x.device)
        outs = []
        for t in range(T):
            kt = k[:, :, t, :]                                   # (B,h,dh)
            vt = v[:, :, t, :]
            S = gam * S + kt.unsqueeze(-1) * vt.unsqueeze(-2)     # (B,h,dh,dh) outer product
            outs.append(torch.einsum("bhd,bhde->bhe", q[:, :, t, :], S))
        y = torch.stack(outs, dim=2)                              # (B,h,T,dh)
        return self._post(x, y)

    def forward(self, x, mode="parallel"):
        return self.forward_parallel(x) if mode == "parallel" else self.forward_recurrent(x)


class SoftmaxAttention(nn.Module):
    """Standard causal multi-head softmax attention (nanoGPT style), the reference arm."""

    def __init__(self, d, h):
        super().__init__()
        self.h, self.dh = h, d // h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x, mode="parallel"):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1)
        att = att.masked_fill(mask, float("-inf")).softmax(-1)
        return self.proj((att @ v).transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, mixer, d, d_ff):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.mix = mixer
        self.fc1, self.fc2 = nn.Linear(d, d_ff), nn.Linear(d_ff, d)

    def forward(self, x, mode="parallel"):
        x = x + self.mix(self.ln1(x), mode=mode)
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class CharLM(nn.Module):
    def __init__(self, vocab, d, h, n_layers, d_ff, ctx, arm, gammas):
        super().__init__()
        self.ctx = ctx
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        blocks = []
        for _ in range(n_layers):
            if arm == "attn":
                mix = SoftmaxAttention(d, h)
            else:
                mix = MultiScaleRetention(d, h, gammas, learn_gamma=(arm == "ret_learned"))
            blocks.append(Block(mix, d, d_ff))
        self.blocks = nn.ModuleList(blocks)
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, mode="parallel"):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x, mode=mode)
        return self.head(self.lnf(x))


# ----------------------------- training ------------------------------------
def n_params(m):
    return sum(p.numel() for p in m.parameters())


def make_model(arm, cfg, vocab, d_ff, gammas):
    p = cfg["params"]
    return CharLM(vocab, p["d_model"], p["n_heads"], p["n_layers"], d_ff, p["ctx"], arm, gammas)


def batch_stream(data, ctx, bs, gen):
    """Deterministic batch sampler - every arm sees the identical stream."""
    hi = len(data) - ctx - 1
    while True:
        ix = torch.randint(hi, (bs,), generator=gen)
        x = torch.stack([data[i:i + ctx] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + ctx] for i in ix])
        yield x, y


@torch.no_grad()
def evaluate(model, val_batches):
    model.eval()
    tot, n = 0.0, 0
    for x, y in val_batches:
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        tot += loss.item() * y.numel()
        n += y.numel()
    model.train()
    return (tot / n) / math.log(2.0)      # bits per char


def lr_at(step, p):
    if step < p["warmup"]:
        return p["lr"] * (step + 1) / p["warmup"]
    t = (step - p["warmup"]) / max(1, p["train_steps"] - p["warmup"])
    return p["lr"] * (p["min_lr_frac"] + (1 - p["min_lr_frac"]) * 0.5 * (1 + math.cos(math.pi * t)))


def train_one(arm, seed, cfg, vocab, d_ff, gammas, train_data, val_batches, val_batches_curve):
    p = cfg["params"]
    set_seeds(seed)
    model = make_model(arm, cfg, vocab, d_ff, gammas)
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"],
                            betas=(0.9, 0.95))
    gen = torch.Generator().manual_seed(1234)      # same data stream for every arm/seed
    stream = batch_stream(train_data, p["ctx"], p["batch_size"], gen)
    curve = []
    t0 = time.time()
    for step in range(p["train_steps"]):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, p)
        x, y = next(stream)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), p["grad_clip"])
        opt.step()
        if (step + 1) % p["eval_every"] == 0 and (step + 1) < p["train_steps"]:
            curve.append((step + 1, evaluate(model, val_batches_curve)))
    final = evaluate(model, val_batches)
    curve.append((p["train_steps"], final))
    return model, final, curve, round(time.time() - t0, 1)


# --------------------- parallel vs recurrent equivalence --------------------
@torch.no_grad()
def equivalence(model, lengths, bs, vocab, seed=7):
    """Max abs deviation between the two forms' logits, per sequence length."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    model.eval()
    for T in lengths:
        idx = torch.randint(vocab, (bs, T), generator=g)
        lp = model(idx, mode="parallel")
        lr = model(idx, mode="recurrent")
        out[str(T)] = {
            "max_abs_dev": float((lp - lr).abs().max()),
            "mean_abs_dev": float((lp - lr).abs().mean()),
            "max_abs_logit": float(lp.abs().max()),
            "max_rel_dev": float((lp - lr).abs().max() / lp.abs().max()),
        }
    model.train()
    return out


# ----------------------------- main ----------------------------------------
def main():
    cfg = load_config()
    p = cfg["params"]
    set_seeds(int(cfg.get("seed", 0)))
    t_start = time.time()

    # ---- data
    text = get_data()
    chars = sorted(set(text))
    vocab = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n_val = int(len(data) * p["val_frac"])
    train_data, val_data = data[:-n_val], data[-n_val:]

    vgen = torch.Generator().manual_seed(999)
    vstream = batch_stream(val_data, p["ctx"], p["batch_size"], vgen)
    val_batches = [next(vstream) for _ in range(p["eval_batches_final"])]
    val_batches_curve = val_batches[:p["eval_batches_curve"]]

    # ---- decay schedules
    H = p["n_heads"]
    multiscale = [1.0 - 2.0 ** (-(p["multiscale_base_exp"] + h)) for h in range(H)]
    schedules = {
        "ret_multiscale": multiscale,
        "ret_uniform": [p["uniform_gamma"]] * H,
        "ret_nodecay": [p["nodecay_gamma"]] * H,
        "ret_learned": multiscale,           # init; trained
        "attn": multiscale,                  # unused
    }

    # ---- iso-parameter: binary-search the attention arm's d_ff to match retention params
    ref = make_model("ret_multiscale", cfg, vocab, p["d_ff"], multiscale)
    target = n_params(ref)
    lo, hi, best, best_gap = 32, 1024, p["d_ff"], 10 ** 9
    while lo <= hi:
        mid = (lo + hi) // 2
        m = make_model("attn", cfg, vocab, mid, multiscale)
        gap = n_params(m) - target
        if abs(gap) < best_gap:
            best, best_gap = mid, abs(gap)
        if gap < 0:
            lo = mid + 1
        else:
            hi = mid - 1
    d_ff_attn = best
    attn_params = n_params(make_model("attn", cfg, vocab, d_ff_attn, multiscale))
    print(f"[iso-param] retention {target} params (d_ff={p['d_ff']}), "
          f"attention {attn_params} params (d_ff={d_ff_attn}), "
          f"spread {100*abs(attn_params-target)/target:.3f}%", flush=True)

    # ---- (1) equivalence at init, for every decay schedule
    equiv_init = {}
    for arm in ["ret_multiscale", "ret_uniform", "ret_nodecay", "ret_learned"]:
        set_seeds(0)
        m = make_model(arm, cfg, vocab, p["d_ff"], schedules[arm])
        equiv_init[arm] = equivalence(m, p["equiv_lengths"], p["equiv_batch"], vocab)
    print("[equiv @init]", json.dumps({k: v[str(p["ctx"])]["max_abs_dev"]
                                       for k, v in equiv_init.items()}), flush=True)

    # ---- (2)(3) train every arm, seed-major so a partial run still has a full sweep
    results_runs, curves, learned_gammas, seeds_done = {}, {}, {}, []
    equiv_trained = {}
    for seed in p["seeds"]:
        if time.time() - t_start > p["time_budget_sec"]:
            print(f"[budget] skipping seed {seed} (elapsed {time.time()-t_start:.0f}s)", flush=True)
            break
        for arm in p["arms"]:
            d_ff = d_ff_attn if arm == "attn" else p["d_ff"]
            model, bpc, curve, secs = train_one(arm, seed, cfg, vocab, d_ff, schedules[arm],
                                                train_data, val_batches, val_batches_curve)
            results_runs.setdefault(arm, {})[str(seed)] = bpc
            curves.setdefault(arm, {})[str(seed)] = curve
            print(f"[train] arm={arm:15s} seed={seed} val_bpc={bpc:.4f} ({secs}s)", flush=True)
            if arm == "ret_learned":
                learned_gammas[str(seed)] = [
                    [round(float(g), 5) for g in b.mix.gammas().detach()] for b in model.blocks
                ]
            if seed == p["seeds"][0] and arm != "attn":
                # (1b) equivalence on TRAINED weights - the harder test
                equiv_trained[arm] = equivalence(model, p["equiv_lengths"],
                                                 p["equiv_batch"], vocab)
        seeds_done.append(seed)

    # ---- timing of the two forms (honest note: python-loop recurrence is slow in torch)
    set_seeds(0)
    m = make_model("ret_multiscale", cfg, vocab, p["d_ff"], multiscale)
    xb = torch.randint(vocab, (p["equiv_batch"], p["ctx"]))
    with torch.no_grad():
        t0 = time.time(); [m(xb, mode="parallel") for _ in range(3)]; t_par = (time.time() - t0) / 3
        t0 = time.time(); [m(xb, mode="recurrent") for _ in range(3)]; t_rec = (time.time() - t0) / 3

    # ---- aggregate
    def agg(arm):
        vals = list(results_runs.get(arm, {}).values())
        if not vals:
            return None
        return {"mean": round(sum(vals) / len(vals), 4),
                "min": round(min(vals), 4), "max": round(max(vals), 4),
                "n_seeds": len(vals)}

    summary = {a: agg(a) for a in p["arms"]}
    seed_spread = (max((summary[a]["max"] - summary[a]["min"]) for a in p["arms"] if summary[a])
                   if len(seeds_done) > 1 else None)
    ret_arms = [a for a in p["arms"] if a.startswith("ret_") and summary[a]]
    best_ret = min(ret_arms, key=lambda a: summary[a]["mean"])
    worst_ret = max(ret_arms, key=lambda a: summary[a]["mean"])
    decay_spread = round(summary[worst_ret]["mean"] - summary[best_ret]["mean"], 4)
    ret_minus_attn = round(summary["ret_multiscale"]["mean"] - summary["attn"]["mean"], 4)
    max_dev_ctx = max(v[str(p["ctx"])]["max_abs_dev"] for v in equiv_init.values())
    max_dev_trained = (max(v[str(p["ctx"])]["max_abs_dev"] for v in equiv_trained.values())
                       if equiv_trained else None)

    metrics = {
        "n_params": {"retention": target, "attention": attn_params},
        "d_ff": {"retention": p["d_ff"], "attention": d_ff_attn},
        "param_spread_pct": round(100 * abs(attn_params - target) / target, 4),
        "train_steps": p["train_steps"],
        "ctx": p["ctx"],
        "batch_size": p["batch_size"],
        "tokens_per_run": p["train_steps"] * p["batch_size"] * p["ctx"],
        "seeds_completed": seeds_done,
        "multiscale_gammas": [round(g, 6) for g in multiscale],
        "val_bpc": summary,
        "val_bpc_per_seed": results_runs,
        "ret_multiscale_minus_attn_bpc": ret_minus_attn,
        "best_retention_arm": best_ret,
        "worst_retention_arm": worst_ret,
        "decay_schedule_ranking": sorted([(a, summary[a]["mean"]) for a in ret_arms],
                                         key=lambda kv: kv[1]),
        "decay_schedule_spread_bpc": decay_spread,
        "seed_spread_max_bpc": round(seed_spread, 4) if seed_spread is not None else None,
        "equivalence_init": equiv_init,
        "equivalence_trained": equiv_trained,
        "max_abs_logit_dev_at_ctx_init": max_dev_ctx,
        "max_abs_logit_dev_at_ctx_trained": max_dev_trained,
        "learned_gammas_final": learned_gammas,
        "forward_sec_parallel": round(t_par, 4),
        "forward_sec_recurrent_pyloop": round(t_rec, 4),
        "curves": curves,
    }
    metrics["headline"] = (
        f"parallel vs recurrent max abs logit deviation {max_dev_ctx:.2e} at T={p['ctx']} (init)"
        + (f" / {max_dev_trained:.2e} (trained)" if max_dev_trained else "")
        + f"; retention(multiscale) {summary['ret_multiscale']['mean']:.4f} vs attention "
          f"{summary['attn']['mean']:.4f} val bpc (delta {ret_minus_attn:+.4f}); "
          f"decay-schedule spread {decay_spread:.4f} bpc vs seed spread "
          f"{metrics['seed_spread_max_bpc']}"
    )

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

    make_chart(cfg, metrics)
    skip = ("curves", "equivalence_init", "equivalence_trained", "val_bpc_per_seed")
    print(json.dumps({k: v for k, v in metrics.items() if k not in skip}, indent=2))
    print("duration_sec", results["duration_sec"])


def make_chart(cfg, m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    p = cfg["params"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))

    label = {"attn": "softmax attn", "ret_multiscale": "ret multi-scale",
             "ret_uniform": "ret uniform .97", "ret_nodecay": "ret no decay (g=1)",
             "ret_learned": "ret learned g"}
    colors = {"attn": "#444444", "ret_multiscale": "#1f77b4", "ret_uniform": "#ff7f0e",
              "ret_nodecay": "#2ca02c", "ret_learned": "#d62728"}

    # (a) final val bpc
    a = ax[0, 0]
    arms = [x for x in p["arms"] if m["val_bpc"].get(x)]
    means = [m["val_bpc"][x]["mean"] for x in arms]
    errs = [[m["val_bpc"][x]["mean"] - m["val_bpc"][x]["min"] for x in arms],
            [m["val_bpc"][x]["max"] - m["val_bpc"][x]["mean"] for x in arms]]
    a.bar(range(len(arms)), means, yerr=errs, capsize=4, color=[colors[x] for x in arms])
    a.set_xticks(range(len(arms)))
    a.set_xticklabels([label[x] for x in arms], rotation=20, ha="right", fontsize=8)
    a.set_ylim(min(means) - 0.10, max(means) + 0.06)
    a.set_ylabel("val bits/char (lower better)")
    a.set_title(f"(a) iso-param ({m['n_params']['retention']} params), "
                f"{m['train_steps']} steps, {len(m['seeds_completed'])} seed(s)")
    for i, v in enumerate(means):
        a.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)

    # (b) curves
    b = ax[0, 1]
    for arm in arms:
        c = m["curves"][arm][str(m["seeds_completed"][0])]
        b.plot([s for s, _ in c], [v for _, v in c], marker="o", ms=3,
               color=colors[arm], label=label[arm])
    b.set_xlabel("step"); b.set_ylabel("val bits/char")
    b.set_title("(b) training curves (seed %s)" % m["seeds_completed"][0])
    b.legend(fontsize=7); b.grid(alpha=0.3)

    # (c) parallel vs recurrent deviation
    c = ax[1, 0]
    Ls = p["equiv_lengths"]
    for arm, d in m["equivalence_init"].items():
        c.plot(Ls, [d[str(L)]["max_abs_dev"] for L in Ls], marker="o", ms=4,
               color=colors[arm], label=label[arm] + " (init)")
    for arm, d in m.get("equivalence_trained", {}).items():
        c.plot(Ls, [d[str(L)]["max_abs_dev"] for L in Ls], marker="s", ms=4, ls="--",
               color=colors[arm], alpha=0.6, label=label[arm] + " (trained)")
    c.set_yscale("log"); c.set_xlabel("sequence length T")
    c.set_ylabel("max |logit_parallel - logit_recurrent|")
    c.set_title("(c) duality check: parallel vs recurrent")
    c.axhline(1e-5, color="k", ls=":", lw=1)
    c.text(Ls[0], 1.2e-5, "1e-5 target", fontsize=7)
    c.legend(fontsize=6, ncol=2); c.grid(alpha=0.3, which="both")

    # (d) decay profiles
    d_ax = ax[1, 1]
    delta = np.arange(0, p["ctx"] + 1)
    for g in m["multiscale_gammas"]:
        d_ax.plot(delta, np.asarray(g, dtype=float) ** delta,
                  color=colors["ret_multiscale"], alpha=0.7, lw=1)
    d_ax.plot([], [], color=colors["ret_multiscale"], label="multi-scale (4 heads)")
    d_ax.plot(delta, np.full(delta.shape, 1.0), color=colors["ret_nodecay"],
              label="no decay g=1.0")
    d_ax.plot(delta, float(p["uniform_gamma"]) ** delta, color=colors["ret_uniform"],
              label=f"uniform g={p['uniform_gamma']}")
    lg = m.get("learned_gammas_final", {})
    if lg:
        first = lg[sorted(lg.keys())[0]]
        for layer in first:
            for g in layer:
                d_ax.plot(delta, float(g) ** delta, color=colors["ret_learned"],
                          alpha=0.5, lw=1, ls="--")
        d_ax.plot([], [], color=colors["ret_learned"], ls="--", label="learned (final)")
    d_ax.set_xlabel("distance n - m"); d_ax.set_ylabel("retention weight gamma^(n-m)")
    d_ax.set_title("(d) decay schedules compared")
    d_ax.legend(fontsize=7); d_ax.grid(alpha=0.3)

    fig.suptitle("RetNet retention at nano scale: duality, retention vs attention, decay schedule",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(HERE / "chart.png", dpi=130)


if __name__ == "__main__":
    main()
