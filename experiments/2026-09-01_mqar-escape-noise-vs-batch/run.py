"""MQAR escape sublinearity: batch size as the gradient-noise dial. Deterministic, CPU-only,
writes results.json + chart.png.

Question (the named next step of 2026-08-26_mqar-joint-lr-wd-decouple): uniform x4 LR
escapes at 400/400/400 where exact inverse scaling of 1x's ~1100 predicts ~275 - a
~1.45x residual sublinearity that survives matching the gate's speed (08-13) and is
seed-exactly invariant to weight decay (08-26). Last suspects standing: stochastic-
gradient diffusion and plateau curvature. Batch size is the clean noise dial: under
Adam, drift per step ~ lr while diffusion per step ~ lr^2/B, so noise accumulated per
unit drift ~ lr/B. Varying B at fixed LR changes noise with drift fixed.

Arms (all N=8, d=64, wd .01, byte-identical paired inits per seed):
  b64_1x  (B=64,  1x)  lr/B=1/64  anchor = 08-26 j1 [1200/1000/1100]
  b64_4x  (B=64,  4x)  lr/B=1/16  anchor = 08-26 j4 [400/400/400]
  b16_1x  (B=16,  1x)  lr/B=1/16  noise-matched to b64_4x
  b256_4x (B=256, 4x)  lr/B=1/64  noise-matched to b64_1x
  b16_4x  (B=16,  4x)  lr/B=1/4   most noise; sharpest determinism-break probe

Interpretation matrix (escape step at N=8, eval grid 100; drift units = step * lr_mult):
  noise IS the residual   -> drift-units-at-escape collapse onto lr/B: b16_1x ~ 4x
                             b64_4x's step (~1600); b256_4x ~ b64_1x/4 (~275-300,
                             restoring exact inverse); x4 determinism degrades as B drops
  noise exonerated        -> escape steps B-invariant at fixed LR (b16_1x ~ b64_1x,
                             b256_4x ~ b16_4x ~ 400), the wd result all over again;
                             curvature becomes the last suspect
  small-B censoring       -> admissible: 2026-07-25_zoology-mqar-recall saw batch 16
                             kill recall-circuit formation in the mixed-load regime
Secondary: does 400/400/400 step-determinism at x4 survive B=16 and B=256; travel-at-
escape constancy readouts continued from 08-07/08-13/08-26.

Cost control (NEW): runs stop post_escape_evals evals after escape - the primary metric
is escape timing; acc is reported as acc_at_stop and is NOT comparable to prior rows.

Usage:  python run.py            (full grid)
        MQAR_PILOT=1 python run.py   (tiny grid, timing sanity check)
"""
import json, math, os, random, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def set_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


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


# ----------------------------------------------------------------------------- data
def make_batch(B, N, key_vocab, val_vocab, gen):
    import torch
    keys = torch.argsort(torch.rand(B, key_vocab, generator=gen), dim=1)[:, :N]
    vals = torch.randint(0, val_vocab, (B, N), generator=gen) + key_vocab
    ctx = torch.stack([keys, vals], dim=2).reshape(B, 2 * N)
    perm = torch.argsort(torch.rand(B, N, generator=gen), dim=1)
    qkeys = torch.gather(keys, 1, perm)
    qvals = torch.gather(vals, 1, perm)
    x = torch.cat([ctx, qkeys], dim=1)                       # (B, 3N)
    y = torch.full((B, 3 * N), -100, dtype=torch.long)
    y[:, 2 * N:] = qvals
    return x, y


# ----------------------------------------------------------------------------- model
def build_model(gated, d_model, n_layers, n_heads, mlp_exp, vocab, max_len, gate_bias):
    """gated=False -> vanilla elu+1 linear attention; gated=True -> dense per-channel decay.
    Byte-identical code path to 2026-07-28/29/08-01/08-03/08-07/08-13 (their 'dense' arms)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    h, dh = n_heads, d_model // n_heads

    class Gate(nn.Module):
        """Returns per-channel log-decay logg of shape (B, h, T, dh), or None when not gated."""
        def __init__(self):
            super().__init__()
            if gated:
                self.lin = nn.Linear(d_model, h * dh, bias=True)
                nn.init.zeros_(self.lin.weight)
                nn.init.constant_(self.lin.bias, gate_bias)

        def forward(self, x):  # x: (B, T, d_model)
            if not gated:
                return None
            B, T, _ = x.shape
            logit = self.lin(x)
            return F.logsigmoid(logit).view(B, T, h, dh).permute(0, 2, 1, 3)  # (B,h,T,dh)

    class GatedLinAttn(nn.Module):
        """elu+1 linear attention with optional per-channel decay, exact closed form."""
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
            self.out = nn.Linear(d_model, d_model, bias=False)
            self.gate = Gate()

        def forward(self, x):
            B, T, _ = x.shape
            q, k, v = self.qkv(x).split(d_model, dim=2)
            q, k, v = (t.view(B, T, h, dh).transpose(1, 2) for t in (q, k, v))
            q, k = F.elu(q) + 1, F.elu(k) + 1
            logg = self.gate(x)
            if logg is None:
                scores = q @ k.transpose(2, 3)                       # (B,h,T,T)
            else:
                A = torch.cumsum(logg, dim=2)                        # (B,h,T,dh)
                D = (A.unsqueeze(3) - A.unsqueeze(2)).clamp(max=0.0).exp()  # (B,h,T,T,dh)
                scores = torch.einsum("bhtc,bhtsc,bhsc->bhts", q, D, k)
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            scores = scores.masked_fill(~mask, 0.0)
            z = scores.sum(dim=3, keepdim=True) + 1e-6
            return self.out(((scores / z) @ v).transpose(1, 2).reshape(B, T, d_model))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1 = nn.LayerNorm(d_model)
            self.mix = GatedLinAttn()
            self.n2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, mlp_exp * d_model), nn.GELU(),
                nn.Linear(mlp_exp * d_model, d_model))

        def forward(self, x):
            x = x + self.mix(self.n1(x))
            return x + self.mlp(self.n2(x))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab, d_model)
            self.pos = nn.Embedding(max_len, d_model)
            self.blocks = nn.ModuleList(Block() for _ in range(n_layers))
            self.lnf = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab, bias=False)

        def forward(self, idx):
            import torch as _t
            x = self.emb(idx) + self.pos(_t.arange(idx.shape[1], device=idx.device))
            for b in self.blocks:
                x = b(x)
            return self.head(self.lnf(x))

    return Model()


# ----------------------------------------------------------------------------- train/eval
def evaluate(model, xe, ye, chunk):
    import torch
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i in range(0, xe.shape[0], chunk):
            logits = model(xe[i:i + chunk])
            m = ye[i:i + chunk] != -100
            pred = logits.argmax(dim=2)
            correct += (pred[m] == ye[i:i + chunk][m]).sum().item()
            total += m.sum().item()
    model.train()
    return correct / total


def gate_travel(model, gate_bias):
    """Distance of gate params from their init (W starts at 0, b at gate_bias)."""
    import torch
    w_norm = b_drift = 0.0
    for n, p in model.named_parameters():
        if ".gate." not in n:
            continue
        with torch.no_grad():
            if "weight" in n:
                w_norm += float(p.norm())
            else:
                b_drift += float((p - gate_bias).norm())
    return round(w_norm, 4), round(b_drift, 4)


def snapshot_backbone(model):
    """Clone every non-gate parameter at init (for travel-from-init readouts)."""
    import torch
    with torch.no_grad():
        return {n: p.detach().clone() for n, p in model.named_parameters()
                if ".gate." not in n}


def backbone_travel(model, snap):
    """(attn qkv/out travel, whole-backbone travel): sum of ||p - p0|| per group."""
    import torch
    attn = full = 0.0
    with torch.no_grad():
        for n, p in model.named_parameters():
            if ".gate." in n:
                continue
            d = float((p - snap[n]).norm())
            full += d
            if ".qkv." in n or ".out." in n:
                attn += d
    return round(attn, 4), round(full, 4)


def param_norm(model):
    import torch
    with torch.no_grad():
        return round(float(torch.sqrt(sum(p.pow(2).sum() for p in model.parameters()))), 4)


def run_one(arm, N, seed, P):
    import torch
    d = P["d_model"]
    name, gmult, bmult, wd, init_key = (arm["name"], arm["gate_lr_mult"],
                                        arm["backbone_lr_mult"], arm["weight_decay"],
                                        arm["init_key"])
    batch_size = arm["batch_size"]
    gated = init_key == "dense"
    # deterministic init seed; sum(ord) not hash() (2026-07-27 fix). CRUCIAL: the seed uses
    # init_key, not the arm name, so all gated arms are byte-identical per seed (and the
    # j1/j4 anchors reproduce the 2026-08-13 j1/j4 rows exactly - wd lives in the optimizer).
    torch.manual_seed(1_000_003 * seed + 101 * d + 13 * N + sum(ord(c) for c in init_key) % 997)
    model = build_model(gated, d, P["n_layers"], P["n_heads"], P["mlp_expansion"],
                        P["key_vocab"] + P["val_vocab"], 3 * 16, P["gate_bias_init"])
    n_params = sum(p.numel() for p in model.parameters())
    snap = snapshot_backbone(model)
    # per-parameter-group LR exactly as 08-13; weight decay is the arm's wd everywhere.
    gate_ps = [p for n_, p in model.named_parameters() if ".gate." in n_]
    rest_ps = [p for n_, p in model.named_parameters() if ".gate." not in n_]
    opt = torch.optim.AdamW(
        [{"params": rest_ps, "lr": P["lr"] * bmult},
         {"params": gate_ps, "lr": P["lr"] * gmult}],
        lr=P["lr"], weight_decay=wd)
    # same train stream + eval set across arms for a given (N, seed) - identical to parents
    gtrain = torch.Generator().manual_seed(500_000 + 100 * N + seed)
    geval = torch.Generator().manual_seed(900_000 + 100 * N + seed)
    xe, ye = make_batch(P["eval_sequences"], N, P["key_vocab"], P["val_vocab"], geval)

    t0, acc, step, traj, travel_traj = time.time(), 0.0, 0, [], []
    saw_nan = False
    evals_after_escape, escaped_at_eval = 0, None
    max_steps = arm.get("train_steps", P["train_steps"])  # per-arm cap (b256_4x cost control)
    lossfn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    for step in range(1, max_steps + 1):
        x, y = make_batch(batch_size, N, P["key_vocab"], P["val_vocab"], gtrain)
        logits = model(x)
        loss = lossfn(logits.reshape(-1, logits.shape[2]), y.reshape(-1))
        if not torch.isfinite(loss):
            saw_nan = True
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % P["eval_every"] == 0:
            acc = evaluate(model, xe, ye, P["eval_chunk"])
            traj.append([step, round(acc, 4)])
            gw, gb = gate_travel(model, P["gate_bias_init"]) if gated else (None, None)
            ba, bf = backbone_travel(model, snap)
            travel_traj.append([step, gw, gb, ba, bf, param_norm(model)])
            if acc >= P["escape_threshold"] and escaped_at_eval is None:
                escaped_at_eval = step
            if escaped_at_eval is not None and step > escaped_at_eval:
                evals_after_escape += 1
            if acc >= P["early_stop_acc"] or evals_after_escape >= P["post_escape_evals"]:
                break
    if not saw_nan and step % P["eval_every"] != 0 and acc < P["early_stop_acc"]:
        acc = evaluate(model, xe, ye, P["eval_chunk"])
        traj.append([step, round(acc, 4)])
        gw, gb = gate_travel(model, P["gate_bias_init"]) if gated else (None, None)
        ba, bf = backbone_travel(model, snap)
        travel_traj.append([step, gw, gb, ba, bf, param_norm(model)])
    escape = next((s for s, a in traj if a >= P["escape_threshold"]), None)
    # travel at the first eval at/after escape (the clock readout)
    travel_at_escape = None
    if escape is not None and travel_traj:
        travel_at_escape = next(
            ({"gate_W": gw, "gate_b": gb, "attn": ba, "backbone": bf, "pnorm": pn}
             for s, gw, gb, ba, bf, pn in travel_traj if s >= escape), None)
    ba_end, bf_end = backbone_travel(model, snap)
    return {"arm": name, "batch_size": batch_size, "gate_lr_mult": gmult,
            "backbone_lr_mult": bmult, "noise_per_drift": round(bmult / batch_size, 5),
            "weight_decay": wd, "num_pairs": N, "seed": seed,
            "acc_at_stop": round(acc, 4), "steps": step, "max_steps": max_steps,
            "escape_step": escape,
            "escape_drift_units": (escape * bmult) if escape is not None else None,
            "params": n_params,
            "saw_nan": saw_nan, "travel_at_escape": travel_at_escape,
            "backbone_travel_end": [ba_end, bf_end],
            "gate_travel_end": gate_travel(model, P["gate_bias_init"]) if gated else None,
            "param_norm_end": param_norm(model),
            "traj": traj, "travel_traj": travel_traj,
            "secs": round(time.time() - t0, 1)}


# ----------------------------------------------------------------------------- chart
def make_chart(runs, P, headline):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    Nm = P["num_pairs_main"]
    fam = [a["name"] for a in P["arms"]]
    cmap = {"b64_1x": "#6a3d9a", "b64_4x": "#e31a1c", "b16_1x": "#1f78b4",
            "b256_4x": "#ff7f00", "b16_4x": "#33a02c"}
    mstyle = {0: "-", 1: "--", 2: ":"}
    seeds_all = sorted({r["seed"] for r in runs})
    cens = P["train_steps"] + 200  # plotting position for "never escaped"
    xpos = {a: i for i, a in enumerate(fam)}

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 4.2))

    # panel 1: escape step per arm, paired lines per seed + noise-account predictions
    for s in seeds_all:
        xs, ys = [], []
        for a in fam:
            rr = [r for r in runs if r["arm"] == a and r["seed"] == s and r["num_pairs"] == Nm]
            if rr:
                e = rr[0]["escape_step"]
                xs.append(xpos[a])
                ys.append(e if e is not None else cens)
        ax1.plot(xs, ys, marker="o", ms=5, lw=1.5, ls=mstyle.get(s, "-"),
                 color="#444444", alpha=0.85, label=f"seed {s}")
    e1 = [r["escape_step"] for r in runs if r["arm"] == "b64_1x" and r["escape_step"]]
    e4 = [r["escape_step"] for r in runs if r["arm"] == "b64_4x" and r["escape_step"]]
    if e1:
        ax1.axhline(float(np.mean(e1)) / 4.0, color="#ff7f00", lw=1.2, ls="-.", alpha=0.8,
                    label=f"noise acct: b256_4x ~ b64_1x/4 ({np.mean(e1)/4:.0f})")
    if e4:
        ax1.axhline(float(np.mean(e4)) * 4.0, color="#1f78b4", lw=1.2, ls=":", alpha=0.8,
                    label=f"noise acct: b16_1x ~ 4x b64_4x ({np.mean(e4)*4:.0f})")
    ax1.axhline(cens, color="#cccccc", lw=0.8)
    ax1.text(0.02, cens - 60, "censored (never escaped)", fontsize=7, color="#888888")
    ax1.set_xticks(list(xpos.values()), fam)
    ax1.set_xlabel("arm (batch, lr mult); lr/B: 1/64, 1/16, 1/16, 1/64, 1/4")
    ax1.set_ylabel(f"escape step (first eval >= {P['escape_threshold']}, grid {P['eval_every']})")
    ax1.legend(frameon=False, fontsize=8)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_title("does escape timing track batch size (noise) at fixed LR?", fontsize=10)

    # panel 2: drift units at escape vs noise-per-drift lr/B - the collapse test.
    # noise account: points at equal lr/B coincide regardless of (lr, B) split.
    for a in fam:
        rs = [r for r in runs if r["arm"] == a and r["num_pairs"] == Nm]
        for r in rs:
            npd = r["noise_per_drift"]
            du = r["escape_drift_units"] if r["escape_drift_units"] is not None \
                else cens * r["backbone_lr_mult"]
            mk = "o" if r["escape_drift_units"] is not None else "x"
            ax2.plot([npd], [du], mk, ms=7, color=cmap.get(a, "#444444"), alpha=0.85,
                     label=a if r["seed"] == 0 else None)
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("noise per unit drift  lr_mult / B  (log2)")
    ax2.set_ylabel("drift units at escape  (step x lr_mult; x = censored)")
    ax2.legend(frameon=False, fontsize=8)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_title("collapse test: is escape a function of lr/B alone?", fontsize=10)

    # panel 3: trajectories, color = arm, linestyle = seed
    for a in fam:
        rs = [r for r in runs if r["arm"] == a and r["num_pairs"] == Nm]
        first = True
        for r in rs:
            xs = [s for s, _ in r["traj"]]
            ys = [v for _, v in r["traj"]]
            ax3.plot(xs, ys, ls=mstyle.get(r["seed"], "-"), color=cmap.get(a, "#1f78b4"),
                     label=a if first else None, lw=1.3, alpha=0.9)
            first = False
    ax3.axhline(P["escape_threshold"], color="#999999", lw=0.8, ls="--")
    ax3.set_ylim(-0.03, 1.05)
    ax3.set_xlabel("train step (runs stop 200 steps post-escape by design)")
    ax3.set_ylabel("recall accuracy")
    ax3.legend(frameon=False, fontsize=7.5, ncol=2)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.set_title("trajectories (solid s0, dashed s1, dotted s2)", fontsize=10)

    fig.suptitle(headline, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(HERE / "chart.png", dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    cfg = load_config()
    P = dict(cfg["params"])
    seed = int(cfg.get("seed", 0))
    set_seeds(seed)
    if os.environ.get("MQAR_PILOT"):
        P.update(train_steps=100, eval_every=50,
                 arms=[{"name": "b256_4x", "batch_size": 256, "gate_lr_mult": 4.0,
                        "backbone_lr_mult": 4.0, "weight_decay": 0.01,
                        "init_key": "dense", "seeds": [0]}])
    t0 = time.time()
    runs = []

    for arm in P["arms"]:
        for s in arm["seeds"]:
            r = run_one(arm, P["num_pairs_main"], s, P)
            runs.append(r)
            print(f"[{time.time()-t0:7.1f}s] {r['arm']:8s} seed={s} B={r['batch_size']} "
                  f"acc@stop={r['acc_at_stop']:.3f} esc={r['escape_step']} "
                  f"steps={r['steps']} nan={r['saw_nan']} "
                  f"travel@esc={r['travel_at_escape']} ({r['secs']}s)", flush=True)

    # aggregates
    import numpy as np
    Nm = P["num_pairs_main"]
    mean_acc, escapes, n_esc, attn_travel_esc, gate_travel_esc, pnorm_esc = {}, {}, {}, {}, {}, {}
    for a in P["arms"]:
        name = a["name"]
        rs = [r for r in runs if r["arm"] == name and r["num_pairs"] == Nm]
        if rs:
            mean_acc[name] = round(float(np.mean([r["acc_at_stop"] for r in rs])), 4)
            escapes[name] = [r["escape_step"] for r in rs]
            n_esc[name] = sum(e is not None for e in escapes[name])
            tw = [r["travel_at_escape"]["attn"] for r in rs if r["travel_at_escape"]]
            attn_travel_esc[name] = round(float(np.mean(tw)), 2) if tw else None
            gw = [r["travel_at_escape"]["gate_W"] for r in rs
                  if r["travel_at_escape"] and r["travel_at_escape"]["gate_W"] is not None]
            gate_travel_esc[name] = round(float(np.mean(gw)), 2) if gw else None
            pn = [r["travel_at_escape"]["pnorm"] for r in rs if r["travel_at_escape"]]
            pnorm_esc[name] = round(float(np.mean(pn)), 2) if pn else None

    def mean_esc(name):
        es = [e for e in escapes.get(name, []) if e is not None]
        return float(np.mean(es)) if es else None

    # decision quantities
    # (1) B-effect at fixed LR: noise account predicts b16_1x < b64_1x and b256_4x < b64_4x is
    #     FALSE - careful with sign: MORE noise (smaller B) has cost the runs so far (x4 needs
    #     more drift units than 1x), so noise-hurts predicts b16_1x LATER than b64_1x and
    #     b256_4x EARLIER than b64_4x. Invariance (like wd) exonerates noise.
    ratios = {}
    for hi, lo in (("b16_1x", "b64_1x"), ("b64_4x", "b256_4x"), ("b16_4x", "b64_4x"),
                   ("b64_1x", "b64_4x"), ("b64_1x", "b256_4x"), ("b16_1x", "b64_4x")):
        a_, b_ = mean_esc(hi), mean_esc(lo)
        ratios[f"{hi}/{lo}"] = round(a_ / b_, 3) if a_ and b_ else None
    # (2) matched-noise pairs in drift units (noise account: ratio ~ 1.0)
    drift = {}
    for name in escapes:
        arm = next(a for a in P["arms"] if a["name"] == name)
        es = [e * arm["backbone_lr_mult"] for e in escapes[name] if e is not None]
        drift[name] = round(float(np.mean(es)), 1) if es else None
    matched = {}
    for a_, b_ in (("b16_1x", "b64_4x"), ("b64_1x", "b256_4x")):
        if drift.get(a_) and drift.get(b_):
            matched[f"{a_}/{b_}_drift_ratio"] = round(drift[a_] / drift[b_], 3)
        else:
            matched[f"{a_}/{b_}_drift_ratio"] = None
    # (3) determinism at x4: per-arm escape-step spread (max-min), seeds escaped only
    spread = {}
    for name in escapes:
        es = [e for e in escapes[name] if e is not None]
        spread[name] = (max(es) - min(es)) if len(es) == len(escapes[name]) and es else None

    headline = "escape steps at N=8 (paired seeds): " + ", ".join(
        f"{a['name']} {escapes.get(a['name'], '?')}" for a in P["arms"])
    make_chart(runs, P, headline)

    results = {
        "id": cfg.get("id", "unknown"),
        "git_commit": git_sha(),
        "seed": seed,
        "duration_sec": round(time.time() - t0, 2),
        "metrics": {"runs": runs, "mean_acc_at_stop_N8": mean_acc,
                    "escape_steps_N8": escapes, "n_escaped": n_esc,
                    "escape_step_spread": spread,
                    "escape_ratio_means": ratios,
                    "mean_drift_units_at_escape": drift,
                    "matched_noise_pairs": matched,
                    "attn_travel_at_escape_mean": attn_travel_esc,
                    "gate_W_at_escape_mean": gate_travel_esc,
                    "param_norm_at_escape_mean": pnorm_esc},
        "env": env_info(),
        "status": "done",
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(headline)
    print("escape ratio means:", ratios)
    print("drift units at escape:", drift)
    print("matched-noise pairs (noise account -> ~1.0):", matched)
    print("escape-step spread (determinism):", spread)


if __name__ == "__main__":
    main()
