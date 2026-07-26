"""MQAR (multi-query associative recall) at nano scale: attention vs decay-gated
linear attention vs gated long-convolution, at matched params and matched steps.

Hand-rolled after zoology (Arora et al., arXiv:2312.04927). The zoology repo is NOT
cloned - its optional extras (mamba_ssm / causal-conv1d / fla) want CUDA.

Task
----
Fixed context of `seq_len` tokens:

    [ k1 v1 k2 v2 ... kN vN ]  [ PAD ... PAD ]  [ q1 q2 ... qM ]

Keys are tokens 0..n_keys-1, values are tokens n_keys..n_keys+n_values-1, PAD is last.
Keys within a sequence are distinct; values are drawn i.i.d.; the M query keys are a
distinct subset of the N keys present. Loss/accuracy live ONLY at the M query positions,
where the model must emit the value that was paired with that key earlier in the SAME
sequence (so there is nothing to memorise across sequences - it is pure in-context recall).
Context length T and the number of queries M are held FIXED; only N (the memory load)
varies, and the slack is PAD.

ONE model per mixer is trained on a uniform MIXTURE of loads N ~ U{1..max_kv_train} and
then evaluated separately at each load. This is a deliberate deviation from a per-load
training sweep, forced by the CPU time-box: in pilots, a model trained on a single fixed
load N >= 8 never leaves chance in 10k steps for ANY of the three mixers, while N = 4
is solved in ~750 steps. Mixing loads is what makes the recall circuit form at all, and
it turns the sweep into 3 training runs instead of 3 x n_loads.

Three mixers drop into an identical pre-LN residual block skeleton:
  attn  - causal multi-head softmax attention           (quadratic reference)
  gla   - decay-gated linear attention, RetNet/GLA flavoured, in its parallel form
          (this is the "add your own pure-PyTorch mixer" arm)
  gconv - gated long depthwise causal convolution        (no content-based routing at all)

d_ff is binary-searched per mixer so all three land on the same total parameter count.

Deterministic, CPU-only, single-threaded. Writes results.json + chart.png.
Usage:  python run.py
"""
import json, math, os, random, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)


# ----------------------------- housekeeping --------------------------------
def set_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
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
    for mod in ("numpy", "torch"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:
            pass
    return info


# ----------------------------- data ----------------------------------------
def make_mqar(rng, n, loads, P):
    """`loads` is either an int (every sequence has that many KV pairs) or an array of
    per-sequence loads. Returns (x, qpos, tgt): x (n,T) tokens, qpos (n,M) query
    positions, tgt (n,M) the value each query must produce."""
    T, M = P["seq_len"], P["n_queries"]
    NK, NV = P["n_keys"], P["n_values"]
    PAD = NK + NV
    if np.isscalar(loads):
        loads = np.full(n, int(loads))
    x = np.full((n, T), PAD, dtype=np.int64)
    tgt = np.zeros((n, M), dtype=np.int64)
    for i in range(n):
        N = int(loads[i])
        keys = rng.choice(NK, size=N, replace=False)              # distinct keys
        vals = rng.integers(0, NV, size=N) + NK                   # values, with repeats
        x[i, 0:2 * N:2] = keys
        x[i, 1:2 * N:2] = vals
        # M queries; if the sequence holds fewer than M pairs, keys repeat as queries
        pick = rng.choice(N, size=M, replace=(N < M))
        x[i, T - M:] = keys[pick]
        tgt[i] = vals[pick]
    qpos = np.tile(np.arange(T - M, T), (n, 1))
    return torch.from_numpy(x), torch.from_numpy(qpos), torch.from_numpy(tgt)


# ----------------------------- mixers --------------------------------------
class SoftmaxAttn(nn.Module):
    """Causal multi-head softmax attention: the quadratic, content-routed reference."""
    def __init__(self, d, h, T):
        super().__init__()
        self.h, self.dh = h, d // h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.register_buffer("mask", torch.triu(torch.ones(T, T, dtype=torch.bool), 1))

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        a = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        a = a.masked_fill(self.mask[:T, :T], float("-inf")).softmax(-1)
        return self.proj((a @ v).transpose(1, 2).reshape(B, T, D))


class GatedLinearAttn(nn.Module):
    """Decay-gated linear attention (RetNet / GLA flavoured), parallel form.

    state_t = g_t * state_{t-1} + phi(k_t) v_t^T   with a per-head, input-dependent
    scalar gate g_t = sigmoid(w_g . x_t) in (0,1). Written in parallel form via the
    cumulative log-decay c_t = sum_{r<=t} log g_r:

        A[t,s] = <phi(q_t), phi(k_s)> * exp(c_t - c_s)   for s <= t
        out_t  = (A @ v)_t / (sum_s A[t,s] + eps)

    phi = elu+1 keeps the scores non-negative so the denominator is a real normaliser.
    Followed by a SiLU output gate. Algebraically identical to the recurrence, but a
    matmul. Decay biases are initialised RetNet-style to 1 - 2^-(3+head) so head
    memories span ~8..64 tokens at init; the gate is free to learn otherwise.
    """
    def __init__(self, d, h, T):
        super().__init__()
        self.h, self.dh = h, d // h
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.g = nn.Linear(d, h)               # per-head scalar decay gate
        self.og = nn.Linear(d, d, bias=False)  # output gate
        self.proj = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            self.g.weight.mul_(0.1)
            decay = torch.tensor([1.0 - 2.0 ** (-(3.0 + i)) for i in range(h)])
            self.g.bias.copy_(torch.log(decay / (1 - decay)))
        self.register_buffer("mask", torch.triu(torch.ones(T, T, dtype=torch.bool), 1))

    def forward(self, x):
        B, T, D = x.shape
        q = F.elu(self.q(x)) + 1.0
        k = F.elu(self.k(x)) + 1.0
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = self.v(x).view(B, T, self.h, self.dh).transpose(1, 2)
        c = F.logsigmoid(self.g(x)).transpose(1, 2).cumsum(-1)          # (B,H,T), <= 0
        d = c.unsqueeze(-1) - c.unsqueeze(-2)                           # log decay, <= 0 for s<=t
        # mask BEFORE the exp: for s > t the difference is positive and exponentiating it
        # first overflows and puts inf into the backward pass (observed as NaN in a pilot).
        d = d.masked_fill(self.mask[:T, :T], -1e9)
        a = (q @ k.transpose(-2, -1)) * d.exp()
        out = (a @ v) / (a.sum(-1, keepdim=True) + 1e-6)
        out = out.transpose(1, 2).reshape(B, T, D) * F.silu(self.og(x))
        return self.proj(out)


class GatedConv(nn.Module):
    """Gated long depthwise causal convolution (H3 / Hyena-lite control).

    out = proj( u * conv(v) ) with u, v linear in x and a LEARNED per-channel kernel of
    full context length. Multiplicative gating is present, but the mixing weights are
    input-INDEPENDENT: no content-based routing anywhere.
    """
    def __init__(self, d, h, T):
        super().__init__()
        self.inp = nn.Linear(d, 2 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.filt = nn.Parameter(torch.randn(d, T) / math.sqrt(T))
        idx = (torch.arange(T)[:, None] - torch.arange(T)[None, :]).clamp(min=0)
        self.register_buffer("idx", idx)
        self.register_buffer("causal",
                             (torch.arange(T)[:, None] >= torch.arange(T)[None, :]).float())

    def forward(self, x):
        B, T, D = x.shape
        u, v = self.inp(x).chunk(2, dim=-1)
        kern = self.filt[:, self.idx[:T, :T]] * self.causal[:T, :T]     # (D,T,T) Toeplitz
        y = torch.einsum("cts,bsc->btc", kern, v)
        return self.proj(u * y)


MIXERS = {"attn": SoftmaxAttn, "gla": GatedLinearAttn, "gconv": GatedConv}


# ----------------------------- model ---------------------------------------
class Block(nn.Module):
    def __init__(self, mixer, d, h, dff, T):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.mix = MIXERS[mixer](d, h, T)
        self.ff = nn.Sequential(nn.Linear(d, dff), nn.GELU(), nn.Linear(dff, d))

    def forward(self, x):
        x = x + self.mix(self.ln1(x))
        return x + self.ff(self.ln2(x))


class TinyMixerLM(nn.Module):
    def __init__(self, mixer, vocab, d, h, dff, n_layers, T, emb_std=1.0):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(T, d)
        # Embedding scale is load-bearing here (pilot-verified): the first half of the
        # recall circuit is a previous-token head, which needs the position signal to be
        # comparable in size to the token signal. Both tables get the SAME std.
        nn.init.normal_(self.emb.weight, std=emb_std)
        nn.init.normal_(self.pos.weight, std=emb_std)
        self.blocks = nn.ModuleList([Block(mixer, d, h, dff, T) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx):
        x = self.emb(idx) + self.pos(torch.arange(idx.shape[1]))[None]
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))


def n_params(mixer, dff, P, vocab):
    m = TinyMixerLM(mixer, vocab, P["d_model"], P["n_heads"], dff,
                    P["n_layers"], P["seq_len"], P["emb_std"])
    return sum(p.numel() for p in m.parameters())


def pick_dff(mixer, P, vocab):
    """Binary-search the FFN width so every mixer lands on target_params (iso-param)."""
    lo, hi = 4, 1024
    while lo < hi:
        mid = (lo + hi) // 2
        if n_params(mixer, mid, P, vocab) < P["target_params"]:
            lo = mid + 1
        else:
            hi = mid
    best = min([lo - 1, lo], key=lambda d: abs(n_params(mixer, d, P, vocab) - P["target_params"]))
    return max(best, 4)


# ----------------------------- train / eval --------------------------------
def gather_q(logits, qpos):
    """logits (B,T,V), qpos (B,M) -> (B,M,V) logits at the query positions."""
    B, T, V = logits.shape
    return logits.gather(1, qpos.unsqueeze(-1).expand(-1, -1, V))


@torch.no_grad()
def evaluate(model, ev, P):
    model.eval()
    x, qpos, tgt = ev
    correct, exact, tot = 0, 0, 0
    bs = 128
    for i in range(0, x.shape[0], bs):
        lg = gather_q(model(x[i:i + bs]), qpos[i:i + bs])
        hit = (lg.argmax(-1) == tgt[i:i + bs])
        correct += int(hit.sum())
        exact += int(hit.all(-1).sum())
        tot += hit.numel()
    model.train()
    return correct / tot, exact / x.shape[0]


def eval_sets(P):
    """One held-out set per eval load, built from a fixed seed so every mixer sees
    byte-identical evaluation data."""
    return {l: make_mqar(np.random.default_rng(202607 + l), P["eval_n"], l, P)
            for l in P["eval_loads"]}


def no_recall_baselines(evs, P):
    """Accuracy of strategies that IGNORE the query key, on the same eval sets.

    A mixer that has learned nothing about key-value matching but has learned "the
    answer is one of the value tokens in this context" will land on one of these. They
    are the floor the recall curves have to be read against - in particular every model
    scores 1.0 at load 1 for free, because there is only one value to emit.
    """
    out = {}
    for l, (x, _, tgt) in evs.items():
        xv = x.numpy()[:, 1:2 * l:2]                       # (n, l) the context values
        t = tgt.numpy()                                    # (n, M) the answers
        uni = float((xv[:, None, :] == t[:, :, None]).mean(-1).mean())
        mode = np.array([np.bincount(row).argmax() for row in xv])
        out[str(l)] = {
            "uniform_random_present_value": round(uni, 4),
            "mode_of_present_values": round(float((mode[:, None] == t).mean()), 4),
            "most_recent_value": round(float((xv[:, -1][:, None] == t).mean()), 4),
        }
        out[str(l)]["best"] = round(max(v for k, v in out[str(l)].items()), 4)
    return out


def train_one(mixer, dff, P, vocab, seed, evs, log):
    """One model per mixer, trained on a MIXTURE of KV loads (uniform over
    1..max_kv_train), then evaluated separately at each load. Training on the mixture
    is what makes the recall circuit form at all inside a CPU time-box: trained on a
    single high load, none of the three mixers leaves chance in 10k steps (pilot)."""
    set_seeds(seed)
    rng = np.random.default_rng(seed * 7919 + 13)
    model = TinyMixerLM(mixer, vocab, P["d_model"], P["n_heads"], dff,
                        P["n_layers"], P["seq_len"], P["emb_std"])
    npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=P["lr"], weight_decay=P["weight_decay"])
    hardest = max(P["eval_loads"])
    curve, t0, capped, loss, step = [], time.time(), False, torch.tensor(float("nan")), 0
    for step in range(P["steps"]):
        for g in opt.param_groups:
            g["lr"] = P["lr"] * min(1.0, (step + 1) / P["warmup"])
        loads = rng.integers(1, P["max_kv_train"] + 1, size=P["batch_size"])
        x, qpos, tgt = make_mqar(rng, P["batch_size"], loads, P)
        lg = gather_q(model(x), qpos)
        loss = F.cross_entropy(lg.reshape(-1, vocab), tgt.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), P["grad_clip"])
        opt.step()
        if (step + 1) % P["eval_every"] == 0:
            curve.append([step + 1, round(evaluate(model, evs[hardest], P)[0], 4)])
            log(f"    {mixer:6s} step {step + 1:5d} loss {float(loss.detach()):.3f} "
                f"acc@{hardest} {curve[-1][1]:.3f}")
        if time.time() - t0 > P["time_cap_s_per_run"]:
            capped = True
            break
    res = {l: evaluate(model, evs[l], P) for l in P["eval_loads"]}
    acc = {l: round(res[l][0], 4) for l in P["eval_loads"]}
    exact = {l: round(res[l][1], 4) for l in P["eval_loads"]}
    secs = time.time() - t0
    log(f"  {mixer:6s} dff={dff:3d} params={npar} steps={step + 1}"
        f" ({secs:.0f}s{' CAPPED' if capped else ''}) "
        + " ".join(f"N{l}={acc[l]:.2f}" for l in P["eval_loads"])
        + f" loss={float(loss.detach()):.3f}")
    return {"mixer": mixer, "d_ff": dff, "n_params": npar,
            "steps_run": step + 1, "time_capped": capped, "seconds": round(secs, 1),
            "final_train_loss": round(float(loss.detach()), 4),
            "recall_acc_by_load": {str(l): acc[l] for l in P["eval_loads"]},
            "seq_exact_by_load": {str(l): exact[l] for l in P["eval_loads"]},
            "acc_curve_hardest_load": curve}


# ----------------------------- main ----------------------------------------
def main():
    cfg = load_config()
    P = cfg["params"]
    seed = int(cfg.get("seed", 0))
    set_seeds(seed)
    t0 = time.time()
    log = lambda s: print(s, flush=True)
    vocab = P["n_keys"] + P["n_values"] + 1          # + PAD
    chance = 1.0 / P["n_values"]

    dffs = {m: pick_dff(m, P, vocab) for m in P["mixers"]}
    pcounts = {m: n_params(m, dffs[m], P, vocab) for m in P["mixers"]}
    log(f"iso-param FFN widths: {dffs}  ->  params {pcounts}")

    evs = eval_sets(P)
    base = no_recall_baselines(evs, P)
    log("no-recall baselines (best): " +
        ", ".join(f"N{l}={base[str(l)]['best']:.3f}" for l in P["eval_loads"]))
    runs = [train_one(m, dffs[m], P, vocab, seed, evs, log) for m in P["mixers"]]
    loads = P["eval_loads"]

    # ---- aggregate ----
    curves = {r["mixer"]: r["recall_acc_by_load"] for r in runs}

    def threshold(m, thr=0.9):
        for l in loads:
            if curves[m][str(l)] < thr:
                return l
        return None

    thr90 = {m: threshold(m, 0.9) for m in P["mixers"]}
    # cliff vs graceful: is the loss concentrated in one adjacent-load step (cliff)
    # or spread across the sweep (graceful)?
    shape = {}
    for m in P["mixers"]:
        ys = [curves[m][str(l)] for l in loads]
        drops = [round(ys[i] - ys[i + 1], 4) for i in range(len(ys) - 1)]
        total = max(ys[0] - ys[-1], 1e-9)
        j = int(np.argmax(drops))
        shape[m] = {
            "acc_by_load": curves[m],
            "adjacent_drops": drops,
            "total_drop": round(ys[0] - ys[-1], 4),
            "max_drop": round(max(drops), 4),
            "max_drop_between": f"{loads[j]}->{loads[j + 1]}",
            "cliffiness": round(max(drops) / total, 4),   # 1.0 = one single cliff
            "load_below_90pct": thr90[m],
        }

    metrics = {
        "task": (f"MQAR: fixed {P['seq_len']}-token context, {P['n_queries']} queries, "
                 f"trained on a uniform mixture of KV loads 1..{P['max_kv_train']}, "
                 f"evaluated at loads {loads}"),
        "chance_acc": round(chance, 4),
        "iso_param": {"d_ff": dffs, "n_params": pcounts,
                      "param_spread_pct": round(100 * (max(pcounts.values()) - min(pcounts.values()))
                                                / max(pcounts.values()), 3)},
        "recall_acc_by_mixer_load": curves,
        "seq_exact_by_mixer_load": {r["mixer"]: r["seq_exact_by_load"] for r in runs},
        "load_below_90pct": thr90,
        "no_recall_baselines": base,
        "margin_over_no_recall_baseline": {
            m: {str(l): round(curves[m][str(l)] - base[str(l)]["best"], 4) for l in loads}
            for m in P["mixers"]},
        "failure_shape": shape,
        "per_run": runs,
        "any_time_capped": any(r["time_capped"] for r in runs),
    }
    hl = "; ".join(f"{m} " + "/".join(f"{curves[m][str(l)]:.2f}" for l in loads)
                   for m in P["mixers"])
    metrics["headline"] = (f"recall acc at KV load {loads} -> {hl}. "
                           "First load <90%: " +
                           ", ".join(f"{m}={thr90[m] if thr90[m] else 'never'}"
                                     for m in P["mixers"]))

    # ----------------------------- chart ------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"attn": "#1a7f64", "gla": "#c95d3c", "gconv": "#8a817c"}
    names = {"attn": "softmax attention (quadratic)",
             "gla": "decay-gated linear attention",
             "gconv": "gated long conv (no routing)"}
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.3), width_ratios=[2.2, 2, 1.4])

    for m in P["mixers"]:
        ax1.plot(loads, [curves[m][str(l)] for l in loads], "o-",
                 color=colors[m], lw=2, ms=5, label=names[m])
    ax1.axhline(0.9, color="0.5", ls="--", lw=1)
    ax1.text(loads[0], 0.915, "90% threshold", fontsize=8, color="0.4")
    ax1.plot(loads, [base[str(l)]["best"] for l in loads], "s--", color="0.6", lw=1.4,
             ms=4, label="best no-recall baseline (ignores the query)")
    ax1.axhline(chance, color="0.7", ls=":", lw=1)
    ax1.text(loads[0], chance + 0.015, "chance (1/32)", fontsize=8, color="0.55")
    ax1.set_xscale("log", base=2); ax1.set_xticks(loads)
    ax1.set_xticklabels([str(l) for l in loads])
    ax1.set_xlabel("KV pairs in context (memory load)")
    ax1.set_ylabel("recall accuracy at query positions")
    ax1.set_ylim(-0.05, 1.08); ax1.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax1.set_title("Recall vs memory load (iso-param, iso-steps)", fontsize=10)

    hardest = max(loads)
    for m in P["mixers"]:
        r = next(r for r in runs if r["mixer"] == m)
        if r["acc_curve_hardest_load"]:
            ax2.plot([c[0] for c in r["acc_curve_hardest_load"]],
                     [c[1] for c in r["acc_curve_hardest_load"]],
                     "-", color=colors[m], lw=2, label=names[m])
    ax2.axhline(0.9, color="0.5", ls="--", lw=1)
    ax2.set_xlabel("training step"); ax2.set_ylabel("held-out recall accuracy")
    ax2.set_ylim(-0.05, 1.08); ax2.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax2.set_title(f"Learning curve at the hardest load (N={hardest})", fontsize=10)

    ax3.bar(list(P["mixers"]), [shape[m]["total_drop"] for m in P["mixers"]],
            color=[colors[m] for m in P["mixers"]])
    for i, m in enumerate(P["mixers"]):
        ax3.text(i, shape[m]["total_drop"] + 0.015,
                 f"cliffiness\n{shape[m]['cliffiness']:.2f}", ha="center", fontsize=8)
    ax3.set_ylabel(f"accuracy lost, load {loads[0]} -> {loads[-1]}")
    ax3.set_ylim(0, 1.15)
    ax3.set_title("How much is lost, and how sharply", fontsize=10)
    for ax in (ax1, ax2, ax3):
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("MQAR at ~0.1M params: attention vs decay-gated linear attention vs gated conv",
                 fontsize=11.5, y=1.02)
    fig.tight_layout()
    fig.savefig(HERE / "chart.png", dpi=160, bbox_inches="tight")

    results = {
        "id": cfg.get("id", "unknown"),
        "git_commit": git_sha(),
        "seed": seed,
        "duration_sec": round(time.time() - t0, 2),
        "metrics": metrics,
        "env": env_info(),
        "status": "done",
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: results[k] for k in ("id", "duration_sec", "status")}, indent=2))
    print("headline:", metrics["headline"])


if __name__ == "__main__":
    main()
