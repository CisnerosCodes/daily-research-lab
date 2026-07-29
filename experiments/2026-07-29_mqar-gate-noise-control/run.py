"""MQAR gate noise control. Deterministic, CPU-only, writes results.json + chart.png.

Question (follow-up to 2026-07-28_mqar-min-selectivity): the dense input-dependent per-channel
forget gate reliably escapes the vanilla-linear-attention plateau at d=64 N=8 (~750-1000 steps,
both seeds) at IDENTICAL state size. Is that win input-dependent ROUTING per se, or just extra
gradient dimensionality / stochastic perturbation flowing through the decay path?

Controls, all sharing the exact decay-masked linear-attention code path of 2026-07-28:

  none        g = 1                      vanilla plateau anchor              [replication]
  noisegate   g_c = sigmoid(b_c + s*eps) train-time-only Gaussian logit noise, eps fresh per
                                         (step, pos, channel); eval uses bias alone
                                         -> pure decay-path noise, no input, no extra grad dims
  shufgate    g_c = sigmoid(b_c+[Wx']_c) x' = batch-SHUFFLED activations (detached): identical
                                         parametrization, gradient dimensionality and input
                                         statistics to dense, but the WRONG sequence's content
  frozen1000  dense, gate params frozen (requires_grad=False) at step 1000, post-breakout
  dense       g_c = sigmoid(b_c + [Wx]_c) the 2026-07-28 winner                [replication]

Readouts (2026-07-28 lesson: fixed-budget endpoints near a plateau are init-noise-ordered):
endpoint accuracy AND escape step (first eval crossing 0.5), full trajectories.

Interpretation matrix:
  shufgate ~ dense              -> the win is optimization conditioning, not routing
  shufgate ~ vanilla            -> input-dependence on the RIGHT input is load-bearing
  noisegate > vanilla           -> part of the effect is noise-assisted plateau escape
  frozen1000 ~ dense            -> the gate's work is done by step 1000 (escape trigger)

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
def build_model(arm, d_model, n_layers, n_heads, mlp_exp, vocab, max_len, gate_bias,
                noise_sigma, aux_seed):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    h, dh = n_heads, d_model // n_heads

    class Gate(nn.Module):
        """Returns per-channel log-decay logg of shape (B, h, T, dh), or None for arm 'none'."""
        def __init__(self, layer_idx):
            super().__init__()
            self.arm = arm
            # dedicated generator per gate module -> deterministic noise / shuffles
            self.gen = torch.Generator().manual_seed(aux_seed + 7919 * layer_idx)
            if arm == "none":
                pass
            elif arm == "noisegate":
                self.bias = nn.Parameter(torch.full((h * dh,), gate_bias))
            else:  # shufgate / frozen1000 / dense: full d_model -> h*dh linear
                self.lin = nn.Linear(d_model, h * dh, bias=True)
                nn.init.zeros_(self.lin.weight)
                nn.init.constant_(self.lin.bias, gate_bias)

        def forward(self, x):  # x: (B, T, d_model)
            B, T, _ = x.shape
            if self.arm == "none":
                return None
            if self.arm == "noisegate":
                logit = self.bias.view(1, 1, h * dh).expand(B, T, h * dh)
                if self.training:
                    eps = torch.randn(B, T, h * dh, generator=self.gen)
                    logit = logit + noise_sigma * eps
            elif self.arm == "shufgate":
                perm = torch.randperm(B, generator=self.gen)
                logit = self.lin(x[perm].detach())
            else:  # dense / frozen1000
                logit = self.lin(x)
            return F.logsigmoid(logit).view(B, T, h, dh).permute(0, 2, 1, 3)  # (B,h,T,dh)

    class GatedLinAttn(nn.Module):
        """elu+1 linear attention with optional per-channel decay, exact closed form."""
        def __init__(self, layer_idx):
            super().__init__()
            self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
            self.out = nn.Linear(d_model, d_model, bias=False)
            self.gate = Gate(layer_idx)

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
        def __init__(self, layer_idx):
            super().__init__()
            self.n1 = nn.LayerNorm(d_model)
            self.mix = GatedLinAttn(layer_idx)
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
            self.blocks = nn.ModuleList(Block(i) for i in range(n_layers))
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


def gate_params(model):
    return sum(p.numel() for n, p in model.named_parameters() if ".gate." in n)


def run_one(arm, N, seed, P):
    import torch
    d = P["d_model"]
    # deterministic init seed; sum(ord) not hash() (2026-07-27 fix: hash() varies across interpreters)
    # NOTE: 'dense' and 'none' use the same formula as 2026-07-28 -> byte-identical inits there.
    torch.manual_seed(1_000_003 * seed + 101 * d + 13 * N + sum(ord(c) for c in arm) % 997)
    aux_seed = 42_000 + 1000 * seed + 10 * N + sum(ord(c) for c in arm) % 991
    model = build_model(arm, d, P["n_layers"], P["n_heads"], P["mlp_expansion"],
                        P["key_vocab"] + P["val_vocab"], 3 * 16, P["gate_bias_init"],
                        P["noise_sigma"], aux_seed)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=P["lr"], weight_decay=P["weight_decay"])
    # same train stream + eval set across arms for a given (N, seed) - identical to prior harness
    gtrain = torch.Generator().manual_seed(500_000 + 100 * N + seed)
    geval = torch.Generator().manual_seed(900_000 + 100 * N + seed)
    xe, ye = make_batch(P["eval_sequences"], N, P["key_vocab"], P["val_vocab"], geval)

    t0, acc, step, traj = time.time(), 0.0, 0, []
    lossfn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    for step in range(1, P["train_steps"] + 1):
        if arm == "frozen1000" and step == P["freeze_step"] + 1:
            for n_, p_ in model.named_parameters():
                if ".gate." in n_:
                    p_.requires_grad_(False)
            # grads are zeroed (set_to_none) every step below, so AdamW skips frozen params
        x, y = make_batch(P["batch_size"], N, P["key_vocab"], P["val_vocab"], gtrain)
        logits = model(x)
        loss = lossfn(logits.reshape(-1, logits.shape[2]), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % P["eval_every"] == 0:
            acc = evaluate(model, xe, ye, P["eval_chunk"])
            traj.append([step, round(acc, 4)])
            if acc >= P["early_stop_acc"]:
                break
    if step % P["eval_every"] != 0 and acc < P["early_stop_acc"]:
        acc = evaluate(model, xe, ye, P["eval_chunk"])
        traj.append([step, round(acc, 4)])
    escape = next((s for s, a in traj if a >= P["escape_threshold"]), None)
    return {"arm": arm, "num_pairs": N, "seed": seed, "acc": round(acc, 4), "steps": step,
            "escape_step": escape, "params": n_params, "gate_params": gate_params(model),
            "traj": traj, "secs": round(time.time() - t0, 1)}


# ----------------------------------------------------------------------------- chart
def make_chart(runs, P, headline):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    arms = P["arms"]
    Nm = P["num_pairs_main"]
    labels = {"none": "none\n(vanilla)", "noisegate": "noise\n(no input)",
              "shufgate": "shuffled\n(wrong input)", "frozen1000": "dense\nfrozen@1k",
              "dense": "dense\n(right input)"}
    cmap = {"none": "#888888", "noisegate": "#e31a1c", "shufgate": "#ff7f00",
            "frozen1000": "#33a02c", "dense": "#6a3d9a"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.0))
    # panel 1: endpoint acc per arm at N=8, seeds as dots, mean as bar
    xpos = np.arange(len(arms))
    for i, a in enumerate(arms):
        accs = [r["acc"] for r in runs if r["arm"] == a and r["num_pairs"] == Nm]
        if not accs:
            continue
        ax1.bar(i, float(np.mean(accs)), width=0.62, color=cmap[a], alpha=0.75)
        for r in runs:
            if r["arm"] == a and r["num_pairs"] == Nm:
                ax1.plot(i, r["acc"], "o", ms=4.5, color="black", alpha=0.6)
                if r["escape_step"]:
                    ax1.annotate(f"esc {r['escape_step']}", (i, r["acc"]),
                                 textcoords="offset points", xytext=(6, -3), fontsize=6.5)
    ax1.axhline(P["escape_threshold"], color="#999999", lw=0.8, ls="--")
    ax1.set_xticks(xpos, [labels[a] for a in arms], fontsize=8)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel("gate control (what information reaches the decay path)")
    ax1.set_ylabel(f"recall accuracy at N={Nm}, {P['train_steps']} steps")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_title(f"endpoints (dots = seeds {P['seeds_main']})", fontsize=10)

    # panel 2: N=8 trajectories, one line per (arm, seed)
    for a in arms:
        rs = [r for r in runs if r["arm"] == a and r["num_pairs"] == Nm]
        for j, r in enumerate(rs):
            xs = [s for s, _ in r["traj"]]
            ys = [v for _, v in r["traj"]]
            ax2.plot(xs, ys, "-" if j == 0 else "--", color=cmap[a],
                     label=a if j == 0 else None, lw=1.6)
    ax2.axhline(P["escape_threshold"], color="#999999", lw=0.8, ls="--")
    ax2.set_ylim(-0.03, 1.05)
    ax2.set_xlabel("train step")
    ax2.set_ylabel("recall accuracy")
    ax2.legend(frameon=False, fontsize=8, ncol=2)
    ax2.spines[["top", "right"]].set_visible(False)
    seed_note = (f"solid seed {P['seeds_main'][0]}, dashed seed {P['seeds_main'][1]}"
                 if len(P["seeds_main"]) > 1 else f"seed {P['seeds_main'][0]}")
    ax2.set_title(f"trajectories at N={Nm} ({seed_note})", fontsize=10)

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
        P.update(seeds_main=[0], train_steps=100, eval_every=50, freeze_step=50,
                 arms=["none", "noisegate", "shufgate", "frozen1000", "dense"],
                 sanity_arms=[])
    t0 = time.time()
    runs = []

    def do(arm, N, s):
        r = run_one(arm, N, s, P)
        runs.append(r)
        print(f"[{time.time()-t0:7.1f}s] {arm:10s} N={N:3d} seed={s} acc={r['acc']:.3f} "
              f"esc={r['escape_step']} steps={r['steps']} gate_params={r['gate_params']} "
              f"({r['secs']}s)", flush=True)
        return r

    # phase 1: decisive cell N=8, all arms x seeds_main
    for arm in P["arms"]:
        for s in P["seeds_main"]:
            do(arm, P["num_pairs_main"], s)

    # phase 2: sanity N=4 for the two NEW mechanisms (can they learn the task at all?)
    for arm in P["sanity_arms"]:
        for s in P["seeds_extra"]:
            do(arm, P["num_pairs_sanity"], s)

    # aggregates
    import numpy as np
    Nm = P["num_pairs_main"]
    mean_acc, escapes = {}, {}
    for a in P["arms"]:
        accs = [r["acc"] for r in runs if r["arm"] == a and r["num_pairs"] == Nm]
        if accs:
            mean_acc[a] = round(float(np.mean(accs)), 4)
            escapes[a] = [r["escape_step"] for r in runs
                          if r["arm"] == a and r["num_pairs"] == Nm]
    delta_vs_vanilla = {a: round(mean_acc[a] - mean_acc["none"], 4) for a in mean_acc}
    delta_vs_dense = {a: round(mean_acc[a] - mean_acc["dense"], 4) for a in mean_acc}
    n_esc = {a: sum(e is not None for e in escapes[a]) for a in escapes}
    routing_arms = [a for a in ("dense", "frozen1000") if n_esc.get(a, 0) == 2]
    control_escaped = [a for a in ("noisegate", "shufgate") if n_esc.get(a, 0) > 0]
    headline = (f"N={Nm} escapes (of 2 seeds): dense {n_esc.get('dense', 0)}, frozen@1k "
                f"{n_esc.get('frozen1000', 0)}, shuffled-input {n_esc.get('shufgate', 0)}, "
                f"noise {n_esc.get('noisegate', 0)}, vanilla {n_esc.get('none', 0)}")
    make_chart(runs, P, headline)

    results = {
        "id": cfg.get("id", "unknown"),
        "git_commit": git_sha(),
        "seed": seed,
        "duration_sec": round(time.time() - t0, 2),
        "metrics": {"runs": runs, "mean_acc_N8": mean_acc,
                    "delta_vs_vanilla_N8": delta_vs_vanilla,
                    "delta_vs_dense_N8": delta_vs_dense,
                    "escape_steps_N8": escapes, "n_escaped_of_2": n_esc,
                    "reliable_escape_arms": routing_arms,
                    "controls_that_escaped": control_escaped},
        "env": env_info(),
        "status": "done",
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(headline)


if __name__ == "__main__":
    main()
