"""Pause/filler tokens vs true recurrence at MATCHED inference FLOPs.

Task: left-nested modular-arithmetic chains, ((a op1 b) op2 c) op3 d ... mod p,
ops from {+,-,*}; difficulty = chain length n (number of binary reductions).

One nested architecture, seven arms that differ ONLY in how extra inference
compute is spent between the expression and the answer readout:

  direct            1 pass over the 11-token prefix                    (no extra compute)
  pause  k          k learnable <pause> tokens appended before readout (arXiv:2310.02226)
  recur_full  k     the SAME 2-layer block looped k+1 times over the whole sequence
  recur_tail  k     1 full pass + k extra 2-layer applications at the ANSWER POSITION ONLY,
                    attending to the frozen pass-1 KV cache.  This costs exactly the same
                    FLOPs as `pause k` -> the clean matched-FLOP head-to-head.
  cot               discrete supervision on the intermediate values (reference upper line)

Both recurrent styles are trained twice: at FIXED depth and with STOCHASTIC depth
(k ~ U{1..k_max}), which is the fix that made the loop extrapolate in registry row
2026-07-25_loop-test-time-compute (prefix parity). This tests whether it transfers.

Deterministic, CPU-only, single-threaded. Writes results.json + chart.png.
Usage:  python run.py
"""
import json, math, os, random, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent


def set_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    import torch
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


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)

# ------------------------------- vocab --------------------------------------
# 0 PAD, 1 BOS, 2 EQ, 3 PAUSE, 4..4+p-1 digits, then the three ops
PAD, BOS, EQ, PAUSE = 0, 1, 2, 3
DIG0 = 4
OPS = ["+", "-", "*"]


def vocab_size(p):
    return DIG0 + p + len(OPS)


def op_id(p, j):
    return DIG0 + p + j


# ------------------------------- data ---------------------------------------
def make_examples(rng, n, P):
    """Returns prefix token ids (n, T_prefix), answer ids (n,), trace ids (n, S+1), chain lens (n,).

    Layout (fixed length): [BOS] [PAD ...] a op b op c ... [EQ]
    left-padded so that the expression is right-aligned and EQ is always last.
    The trace is the running value after each reduction, right-aligned into S+1 slots
    (S = cot_slots), earlier slots filled with PAD (ignored in the loss).
    """
    p, maxn, S = P["p"], P["max_chain"], P["cot_slots"]
    expr_slots = 2 * maxn + 1
    T = 1 + expr_slots + 1
    lens = rng.choice(np.asarray(P["chain_lens"]), size=n)
    seq = np.full((n, T), PAD, dtype=np.int64)
    seq[:, 0] = BOS
    seq[:, -1] = EQ
    trace = np.full((n, S + 1), PAD, dtype=np.int64)
    ans = np.zeros(n, dtype=np.int64)
    for i in range(n):
        nn_ = int(lens[i])
        operands = rng.integers(0, p, size=nn_ + 1)
        opsel = rng.integers(0, len(OPS), size=nn_)
        toks = [DIG0 + int(operands[0])]
        v = int(operands[0])
        vals = []
        for j in range(nn_):
            o, b = int(opsel[j]), int(operands[j + 1])
            toks += [op_id(p, o), DIG0 + b]
            v = (v + b) % p if o == 0 else ((v - b) % p if o == 1 else (v * b) % p)
            vals.append(v)
        # right-align expression into slots 1..expr_slots
        seq[i, 1 + expr_slots - len(toks):1 + expr_slots] = np.asarray(toks)
        ans[i] = DIG0 + v
        # running values including the leading operand, left-padded by repeating it,
        # so every trace slot holds a real digit and the answer is always the last slot
        full = [int(operands[0])] + vals
        if len(full) < S + 1:
            full = [full[0]] * (S + 1 - len(full)) + full
        trace[i] = np.asarray([DIG0 + x for x in full[-(S + 1):]])
    return (torch.from_numpy(seq), torch.from_numpy(ans),
            torch.from_numpy(trace), torch.from_numpy(lens.astype(np.int64)))


# ------------------------------- model --------------------------------------
class Block(nn.Module):
    def __init__(self, d, h, dff):
        super().__init__()
        self.h, self.dh = h, d // h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ff = nn.Sequential(nn.Linear(d, dff), nn.GELU(), nn.Linear(dff, d))

    def forward(self, x, past_kv=None, causal=True):
        B, T, D = x.shape
        y = self.ln1(x)
        q, k, v = self.qkv(y).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        kv_new = (k, v)
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        if causal and past_kv is None:
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
            att = att.masked_fill(mask, float("-inf"))
        att = att.softmax(-1)
        x = x + self.proj((att @ v).transpose(1, 2).reshape(B, T, D))
        x = x + self.ff(self.ln2(x))
        return x, kv_new


class TinyLM(nn.Module):
    """One architecture for every arm. `inject` is applied once per loop iteration,
    so `direct` is exactly `recur_full` with zero extra loops -> arms are nested."""

    def __init__(self, vocab, d, h, dff, n_layers, max_pos):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_pos, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.inject = nn.Linear(2 * d, d)
        self.blocks = nn.ModuleList([Block(d, h, dff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def embed(self, idx):
        return self.emb(idx) + self.pos(torch.arange(idx.shape[1]))[None]

    def forward_full(self, idx, loops=1):
        """Whole-sequence loop. Returns logits for every position."""
        e = self.embed(idx)
        h = e
        for _ in range(loops):
            z = self.inject(torch.cat([h, e], dim=-1))
            for b in self.blocks:
                z, _ = b(z, None, True)
            h = z
        return self.head(self.ln_f(h))

    def forward_tail(self, idx, k_extra):
        """1 full pass, then k_extra 2-layer applications at the LAST position only,
        attending to the frozen pass-1 KV cache. Returns logits at the last position."""
        e = self.embed(idx)
        z = self.inject(torch.cat([e, e], dim=-1))
        caches = []
        for b in self.blocks:
            z, kv = b(z, None, True)
            caches.append((kv[0][:, :, :-1].contiguous(), kv[1][:, :, :-1].contiguous()))
        h = z[:, -1:, :]
        e_last = e[:, -1:, :]
        for _ in range(k_extra):
            u = self.inject(torch.cat([h, e_last], dim=-1))
            for b, c in zip(self.blocks, caches):
                u, _ = b(u, c, False)
            h = u
        return self.head(self.ln_f(h))[:, -1]


# ------------------------------- FLOP model ---------------------------------
def flop_pieces(P):
    d, dff, L = P["d_model"], P["d_ff"], P["n_layers"]
    core = 2 * (4 * d * d + 2 * d * dff)      # qkv+proj+ffn, per layer per position
    per_ctx = 2 * (2 * d)                     # scores + weighted values, per attended key
    inj = 2 * (2 * d * d)                     # injection adapter, per position
    return core, per_ctx, inj, L


def flops_pass(T, P):
    """One full forward pass over a causal sequence of T positions."""
    core, per_ctx, inj, L = flop_pieces(P)
    ctx_sum = T * (T + 1) // 2
    return T * inj + L * (T * core + per_ctx * ctx_sum)


def flops_tail_step(T, P):
    """One extra 2-layer application at a single position attending to T keys."""
    core, per_ctx, inj, L = flop_pieces(P)
    return inj + L * (core + per_ctx * T)


def arm_flops(arm, k, P):
    T = 1 + (2 * P["max_chain"] + 1) + 1
    if arm == "direct":
        return flops_pass(T, P)
    if arm == "pause":
        return flops_pass(T + k, P)
    if arm == "cot":
        # with a KV cache, emitting S extra tokens costs the same as processing S extra positions
        return flops_pass(T + P["cot_slots"], P)
    if arm.startswith("recur_full"):
        return (k + 1) * flops_pass(T, P)
    if arm.startswith("recur_tail"):
        return flops_pass(T, P) + k * flops_tail_step(T, P)
    raise ValueError(arm)


# ------------------------------- train / eval -------------------------------
@torch.no_grad()
def evaluate(model, arm, k, P, ev, dev_loops=None):
    """Accuracy overall and per chain length on the fixed eval set."""
    seq, ans, trace, lens = ev
    correct = {}
    B = seq.shape[0]
    if arm == "cot":
        cur = seq
        outs = []
        for _ in range(P["cot_slots"] + 1):
            logits = model.forward_full(cur, loops=1)[:, -1]
            nxt = logits.argmax(-1, keepdim=True)
            outs.append(nxt)
            cur = torch.cat([cur, nxt], dim=1)
        pred = outs[-1][:, 0]
    elif arm == "pause":
        pad = torch.full((B, k), PAUSE, dtype=torch.long)
        pred = model.forward_full(torch.cat([seq, pad], dim=1), loops=1)[:, -1].argmax(-1)
    elif arm == "direct":
        pred = model.forward_full(seq, loops=1)[:, -1].argmax(-1)
    elif arm.startswith("recur_full"):
        loops = (dev_loops if dev_loops is not None else k) + 1
        pred = model.forward_full(seq, loops=loops)[:, -1].argmax(-1)
    else:  # recur_tail
        ke = dev_loops if dev_loops is not None else k
        pred = model.forward_tail(seq, ke).argmax(-1)
    ok = (pred == ans).float()
    out = {"overall": float(ok.mean())}
    for n in P["chain_lens"]:
        m = lens == n
        out[str(n)] = float(ok[m].mean())
    return out


def train_one(arm, k, seed, P, ev, log):
    set_seeds(seed)
    rng = np.random.default_rng(seed * 9176 + 13)
    V = vocab_size(P["p"])
    model = TinyLM(V, P["d_model"], P["n_heads"], P["d_ff"], P["n_layers"], P["max_positions"])
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=P["lr"], weight_decay=P["weight_decay"])
    gen = torch.Generator().manual_seed(seed * 7717 + 5)
    t0, capped, loss = time.time(), False, torch.tensor(float("nan"))
    step = -1
    for step in range(P["steps"]):
        for g in opt.param_groups:
            g["lr"] = P["lr"] * min(1.0, (step + 1) / P["warmup"])
        seq, ans, trace, lens = make_examples(rng, P["batch_size"], P)
        if arm == "cot":
            # teacher-forced trace: inputs = prefix + trace[:-1], targets = trace
            idx = torch.cat([seq, trace[:, :-1]], dim=1)
            logits = model.forward_full(idx, loops=1)
            T0 = seq.shape[1] - 1               # position that predicts trace[0]
            lg = logits[:, T0:T0 + trace.shape[1]]
            tgt = trace.clone()
            tgt[tgt == PAD] = -100
            loss = F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), ignore_index=-100)
        elif arm == "pause":
            pad = torch.full((seq.shape[0], k), PAUSE, dtype=torch.long)
            logits = model.forward_full(torch.cat([seq, pad], dim=1), loops=1)[:, -1]
            loss = F.cross_entropy(logits, ans)
        elif arm == "direct":
            loss = F.cross_entropy(model.forward_full(seq, loops=1)[:, -1], ans)
        elif arm == "recur_full_fix":
            loss = F.cross_entropy(model.forward_full(seq, loops=k + 1)[:, -1], ans)
        elif arm == "recur_full_stoch":
            lp = int(torch.randint(1, k + 2, (1,), generator=gen))
            loss = F.cross_entropy(model.forward_full(seq, loops=lp)[:, -1], ans)
        elif arm == "recur_tail_fix":
            loss = F.cross_entropy(model.forward_tail(seq, k), ans)
        elif arm == "recur_tail_stoch":
            ke = int(torch.randint(0, k + 1, (1,), generator=gen))
            loss = F.cross_entropy(model.forward_tail(seq, ke), ans)
        else:
            raise ValueError(arm)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), P["grad_clip"])
        opt.step()
        if time.time() - t0 > P["time_cap_s_per_run"]:
            capped = True
            break
    train_s = time.time() - t0
    lossv = float(loss.detach())
    model.eval()
    acc = evaluate(model, arm, k, P, ev)
    extrap = {}
    is_top = ((arm.startswith("recur_tail") and k == max(P["k_values"]))
              or (arm.startswith("recur_full") and k == max(P["full_k_values"])))
    if is_top:
        for kk in P["extrapolation_k"]:
            extrap[str(kk)] = evaluate(model, arm, k, P, ev, dev_loops=kk)["overall"]
    fl = arm_flops(arm, k, P)
    bylen = " ".join(f"n{n} {acc[str(n)]:.2f}" for n in P["chain_lens"])
    log(f"  {arm:17s} k={k} seed{seed}: steps={step+1}{' CAPPED' if capped else ''} "
        f"{train_s:5.1f}s loss={lossv:.3f} acc={acc['overall']:.3f} "
        f"({bylen}) MFLOPs={fl/1e6:.2f}")
    return {"arm": arm, "k": k, "seed": seed, "n_params": n_params, "steps_run": step + 1,
            "time_capped": capped, "train_seconds": round(train_s, 1),
            "final_loss": round(lossv, 4), "flops_per_example": fl,
            "acc": {kk: round(v, 4) for kk, v in acc.items()},
            "extrapolation_acc": {kk: round(v, 4) for kk, v in extrap.items()}}


# ------------------------------- main ---------------------------------------
def main():
    cfg = load_config()
    P = cfg["params"]
    set_seeds(int(cfg.get("seed", 0)))
    t0 = time.time()
    log = lambda s: print(s, flush=True)

    erng = np.random.default_rng(P["eval_seed"])
    parts = []
    for n in P["chain_lens"]:
        Pn = dict(P); Pn["chain_lens"] = [n]
        parts.append(make_examples(erng, P["eval_n_per_len"], Pn))
    ev = tuple(torch.cat([p[i] for p in parts], dim=0) for i in range(4))
    # empirical majority-class baseline (the answer marginal is skewed by the * operator)
    _ans, _len = ev[1], ev[3]
    majority = {"overall": float(torch.bincount(_ans).max()) / len(_ans)}
    for n in P["chain_lens"]:
        a = _ans[_len == n]
        majority[str(n)] = float(torch.bincount(a).max()) / len(a)
    log(f"eval set: {ev[0].shape[0]} examples, prefix len {ev[0].shape[1]}, "
        f"uniform chance = {1.0/P['p']:.4f}, majority-class baseline = "
        + ", ".join(f"{k}:{v:.3f}" for k, v in majority.items()))

    configs = []
    for arm in P["arms"]:
        if arm == "direct":
            configs.append((arm, 0))
        elif arm == "cot":
            configs.append((arm, P["cot_slots"]))
        elif arm.startswith("recur_full"):
            configs += [(arm, k) for k in P["full_k_values"]]
        else:
            configs += [(arm, k) for k in P["k_values"]]

    runs = []
    for arm, k in configs:
        for seed in P["seeds"]:
            runs.append(train_one(arm, k, seed, P, ev, log))

    # ---------------- aggregate ----------------
    agg = {}
    for arm, k in configs:
        rs = [r for r in runs if r["arm"] == arm and r["k"] == k]
        key = f"{arm}_k{k}"
        agg[key] = {
            "arm": arm, "k": k,
            "flops_per_example": rs[0]["flops_per_example"],
            "mflops": round(rs[0]["flops_per_example"] / 1e6, 3),
            "acc": {kk: round(float(np.mean([r["acc"][kk] for r in rs])), 4)
                    for kk in rs[0]["acc"]},
            "acc_std": round(float(np.std([r["acc"]["overall"] for r in rs])), 4),
            "acc_per_seed": [r["acc"]["overall"] for r in rs],
            "acc_per_mflop": round(float(np.mean([r["acc"]["overall"] for r in rs]))
                                   / (rs[0]["flops_per_example"] / 1e6), 4),
            "extrapolation_acc": ({kk: round(float(np.mean([r["extrapolation_acc"][kk] for r in rs])), 4)
                                   for kk in rs[0]["extrapolation_acc"]}
                                  if rs[0]["extrapolation_acc"] else {}),
        }

    base = agg["direct_k0"]["acc"]["overall"]
    # matched-FLOP head-to-head: pause k vs recur_tail_* k (identical FLOPs by construction)
    matched = {}
    for k in P["k_values"]:
        row = {"mflops": agg[f"pause_k{k}"]["mflops"],
               "pause": agg[f"pause_k{k}"]["acc"]["overall"],
               "recur_tail_fix": agg[f"recur_tail_fix_k{k}"]["acc"]["overall"],
               "recur_tail_stoch": agg[f"recur_tail_stoch_k{k}"]["acc"]["overall"]}
        fp, fr = agg[f"pause_k{k}"]["flops_per_example"], agg[f"recur_tail_fix_k{k}"]["flops_per_example"]
        row["flop_rel_mismatch"] = round(abs(fp - fr) / fp, 6)
        row["best"] = max(("pause", "recur_tail_fix", "recur_tail_stoch"), key=lambda a: row[a])
        row["delta_recur_best_minus_pause"] = round(
            max(row["recur_tail_fix"], row["recur_tail_stoch"]) - row["pause"], 4)
        matched[str(k)] = {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in row.items()}

    stoch_vs_fix = {}
    for style, kk_list in (("recur_full", P["full_k_values"]), ("recur_tail", P["k_values"])):
        for k in kk_list:
            stoch_vs_fix[f"{style}_k{k}"] = round(
                agg[f"{style}_stoch_k{k}"]["acc"]["overall"]
                - agg[f"{style}_fix_k{k}"]["acc"]["overall"], 4)

    ranking = sorted(agg, key=lambda kk: -agg[kk]["acc_per_mflop"])
    acc_ranking = sorted(agg, key=lambda kk: -agg[kk]["acc"]["overall"])

    kmax = max(P["k_values"])
    kfull = max(P["full_k_values"])
    metrics = {
        "uniform_chance_acc": round(1.0 / P["p"], 4),
        "majority_class_baseline": {k: round(v, 4) for k, v in majority.items()},
        "n_params": runs[0]["n_params"],
        "prefix_len": int(ev[0].shape[1]),
        "eval_n": int(ev[0].shape[0]),
        "per_run": runs,
        "aggregate": agg,
        "direct_baseline_acc": base,
        "matched_flop_head_to_head": matched,
        "stochastic_minus_fixed_depth": stoch_vs_fix,
        "acc_per_mflop_ranking": ranking,
        "acc_ranking": acc_ranking,
        "best_acc_per_mflop": {"config": ranking[0], "value": agg[ranking[0]]["acc_per_mflop"],
                               "acc": agg[ranking[0]]["acc"]["overall"]},
        "best_acc": {"config": acc_ranking[0], "value": agg[acc_ranking[0]]["acc"]["overall"],
                     "mflops": agg[acc_ranking[0]]["mflops"]},
        "cot_reference_acc": agg[f"cot_k{P['cot_slots']}"]["acc"]["overall"],
        "extrapolation_kmax": {kk: agg[kk]["extrapolation_acc"] for kk in agg
                               if agg[kk]["extrapolation_acc"]},
        "acc_by_chain_len": {kk: agg[kk]["acc"] for kk in agg},
        "whole_sequence_loop_vs_matched_flop_pause": {
            "recur_full_fix": agg[f"recur_full_fix_k{kfull}"]["acc"]["overall"],
            "recur_full_stoch": agg[f"recur_full_stoch_k{kfull}"]["acc"]["overall"],
            "mflops_full": agg[f"recur_full_fix_k{kfull}"]["mflops"],
            "mflops_pause_kmax": agg[f"pause_k{kmax}"]["mflops"],
            "flop_ratio_full_over_pause_kmax": round(
                agg[f"recur_full_fix_k{kfull}"]["flops_per_example"]
                / agg[f"pause_k{kmax}"]["flops_per_example"], 3),
        },
        "headline": "",
    }
    metrics["headline"] = (
        f"at IDENTICAL inference FLOPs (pause k vs tail-recurrence k), the winner is "
        + ", ".join(f"k={k}: {matched[str(k)]['best']} "
                    f"({max(matched[str(k)]['pause'], matched[str(k)]['recur_tail_fix'], matched[str(k)]['recur_tail_stoch']):.3f} "
                    f"vs pause {matched[str(k)]['pause']:.3f})" for k in P["k_values"])
        + f"; best accuracy-per-MFLOP overall = {ranking[0]}; direct baseline {base:.3f}, "
          f"CoT reference {metrics['cot_reference_acc']:.3f}, "
          f"majority-class baseline {majority['overall']:.3f}")

    make_chart(P, agg, majority, metrics, base, kmax, kfull)

    results = {
        "id": cfg.get("id", "unknown"),
        "git_commit": git_sha(),
        "seed": int(cfg.get("seed", 0)),
        "duration_sec": round(time.time() - t0, 2),
        "metrics": metrics,
        "env": env_info(),
        "status": "done",
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: results[k] for k in ("id", "duration_sec", "status")}, indent=2))
    print("headline:", metrics["headline"])


def make_chart(P, agg, majority, metrics, base, kmax, kfull):
    """Three-panel figure. Split out of main() so it can be regenerated from results.json."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = {"direct": "#8a817c", "pause": "#c95d3c", "recur_tail_fix": "#3d5a80",
         "recur_tail_stoch": "#1a7f64", "recur_full_fix": "#7b6ca8",
         "recur_full_stoch": "#2b8ca8", "cot": "#b0a160"}
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

    mbase = majority["overall"]
    ck = f"cot_k{P['cot_slots']}"
    n_seeds = len(P["seeds"])

    # (1) accuracy vs FLOPs
    ax = axes[0]
    for a, kl in (("pause", P["k_values"]), ("recur_tail_fix", P["k_values"]),
                  ("recur_tail_stoch", P["k_values"]),
                  ("recur_full_fix", P["full_k_values"]), ("recur_full_stoch", P["full_k_values"])):
        xs = [agg[f"{a}_k{k}"]["mflops"] for k in kl]
        ys = [agg[f"{a}_k{k}"]["acc"]["overall"] for k in kl]
        ax.plot(xs, ys, "o-" if len(xs) > 1 else "D", color=C[a], label=a, lw=2, ms=6)
    ax.plot([agg["direct_k0"]["mflops"]], [base], "s", color=C["direct"], ms=8, label="direct (k=0)")
    ax.plot([agg[ck]["mflops"]], [agg[ck]["acc"]["overall"]], "*", color=C["cot"], ms=14,
            label="CoT supervision (ref)")
    ax.axhline(mbase, color="0.6", ls=":", lw=1)
    ax.text(agg["direct_k0"]["mflops"], mbase + 0.012, "majority-class baseline",
            fontsize=8, color="0.5")
    ax.set_xlabel("inference MFLOPs per example"); ax.set_ylabel("answer accuracy")
    ax.set_ylim(0.2, 1.30)
    ax.set_title("Accuracy vs inference FLOPs\n(pause k and tail-recurrence k coincide in x)",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2)
    ax.spines[["top", "right"]].set_visible(False)

    # (2) matched-FLOP head-to-head bars
    ax = axes[1]
    w, ks = 0.26, P["k_values"]
    xpos = np.arange(len(ks))
    # +-1 binomial standard error on the eval set (only error bar available at one seed)
    se = math.sqrt(0.4 * 0.6 / metrics["eval_n"])
    for i, a in enumerate(("pause", "recur_tail_fix", "recur_tail_stoch")):
        vals = [agg[f"{a}_k{k}"]["acc"]["overall"] for k in ks]
        errs = ([agg[f"{a}_k{k}"]["acc_std"] for k in ks] if n_seeds > 1 else [se] * len(ks))
        ax.bar(xpos + (i - 1) * w, vals, w, yerr=errs, capsize=2, color=C[a], label=a)
        for j, v in enumerate(vals):
            ax.text(xpos[j] + (i - 1) * w, v + 0.035, f"{v:.3f}", ha="center", fontsize=7)
    ax.axhline(base, color=C["direct"], ls="--", lw=1.5)
    ax.text(-0.45, base + 0.012, f"direct {base:.3f}", fontsize=8, color=C["direct"])
    ax.axhline(mbase, color="0.6", ls=":", lw=1)
    ax.text(-0.45, mbase - 0.045, f"majority {mbase:.3f}", fontsize=8, color="0.5")
    ax.set_xticks(xpos); ax.set_xticklabels([f"k={k}\n{agg[f'pause_k{k}']['mflops']:.2f} MFLOP" for k in ks])
    ax.set_ylabel("answer accuracy"); ax.set_ylim(0, 0.62)
    ax.set_title("Matched-FLOP head-to-head (identical FLOPs within each group)\n"
                 + ("error bars = seed std" if n_seeds > 1
                    else f"error bars = +-1 eval binomial SE ({se:.3f}); ONE seed"), fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left"); ax.spines[["top", "right"]].set_visible(False)

    # (3) accuracy vs chain length at k = kmax
    ax = axes[2]
    for a, kk_ in (("pause", kmax), ("recur_tail_fix", kmax), ("recur_tail_stoch", kmax),
                   ("recur_full_fix", kfull), ("recur_full_stoch", kfull)):
        ys = [agg[f"{a}_k{kk_}"]["acc"][str(n)] for n in P["chain_lens"]]
        ax.plot(P["chain_lens"], ys, "o-", color=C[a], label=f"{a} k={kk_}", lw=2, ms=5)
    ax.plot(P["chain_lens"], [agg["direct_k0"]["acc"][str(n)] for n in P["chain_lens"]],
            "s--", color=C["direct"], label="direct", lw=1.5, ms=5)
    ax.plot(P["chain_lens"], [agg[ck]["acc"][str(n)] for n in P["chain_lens"]],
            "*--", color=C["cot"], label="CoT (ref)", lw=1.5, ms=10)
    ax.plot(P["chain_lens"], [majority[str(n)] for n in P["chain_lens"]],
            ":", color="0.6", lw=1.2, label="majority class")
    ax.set_xticks(P["chain_lens"]); ax.set_xlabel("chain length (number of ops)")
    ax.set_ylabel("answer accuracy"); ax.set_ylim(0, 1.05)
    ax.set_title(f"Difficulty scaling (tail/pause at k={kmax}, loop at k={kfull})", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5); ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Pause/filler tokens vs true recurrence at matched inference FLOPs - "
                 f"mod-{P['p']} arithmetic chains, {metrics['n_params']/1000:.0f}k params, "
                 f"{P['steps']} steps, "
                 + (f"{n_seeds} seeds" if n_seeds > 1 else "single seed"), fontsize=11, y=1.03)
    fig.tight_layout()
    fig.savefig(HERE / "chart.png", dpi=155, bbox_inches="tight")


if __name__ == "__main__":
    main()
