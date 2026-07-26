"""minGRU (input-only gates) vs a standard GRU (input+hidden-state gates) on SELECTIVE COPY.

Ablates the central claim of "Were RNNs All We Needed?" (arXiv:2410.01201): that stripping the
hidden-state dependence out of the gates - which is what turns the recurrence into a parallel scan -
costs nothing. The Mamba paper (arXiv:2312.00752) motivates selective state updates with exactly the
selective-copy task. For minGRU the two claims conflict: its gate sees the current input (so it CAN
decide "this token is data, integrate it") but never the state (so it CANNOT decide "this is the 3rd
data token, put it in slot 3").

Task: a length-L sequence of a single blank/noise token with k data tokens at uniformly random
positions; the model must emit the k data-token values in order of appearance.
Readout is identical for every arm: k linear heads on the final hidden state (via a shared
bottleneck), so the comparison is purely about the recurrence, not about a decoder.

Arms (hidden width fitted per arm so TOTAL params ~= target for every arm):
  gru          1-layer standard GRU (nn.GRU; gates see x AND h)
  mingru_mp    1-layer minGRU, matched params (much wider state, no hidden-state gate)
  mingru_ms    1-layer minGRU, matched hidden-state SIZE to the GRU (fewer params) - capacity control
  minlstm_mp   1-layer minLSTM, matched params
  gru_2l       2-layer GRU        (probe cells only)
  mingru_2l    2-layer minGRU     (probe cells only) - does depth recover state-dependent gating?
  mingru_long  1-layer minGRU, matched params, 3x steps (probe cells only) - slow or incapable?

Deterministic, CPU-only, single-threaded. Writes results.json + chart.png.
Usage:  python run.py
"""
import json, os, random, subprocess, sys, time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

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
    for mod in ("numpy", "torch", "matplotlib"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:
            pass
    return info


# --------------------------------------------------------------------------- torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)


# ----------------------------------------------------------------- data
def make_batch(B, L, k, v_data, gen):
    """Selective copy: k data tokens (ids 1..v_data) at random positions in L blanks (id 0)."""
    r = torch.rand(B, L, generator=gen)
    pos = r.argsort(dim=1)[:, :k].sort(dim=1).values            # (B,k) sorted random positions
    vals = torch.randint(1, 1 + v_data, (B, k), generator=gen)  # (B,k) data values
    x = torch.zeros(B, L, dtype=torch.long)
    x.scatter_(1, pos, vals)
    return x, vals - 1                                          # targets in 0..v_data-1


# ----------------------------------------------------------------- cells
class MinGRU(nn.Module):
    """minGRU (arXiv:2410.01201 eq. 'minGRU'): h_t = (1-z_t) h_{t-1} + z_t h~_t,
    z_t = sigmoid(Linear(x_t)), h~_t = Linear(x_t).  Gates depend on the INPUT ONLY, so the
    recurrence is a linear scan and could be run with an associative parallel scan; we use the
    mathematically identical sequential form (CPU, short sequences)."""

    def __init__(self, d_in, d_h):
        super().__init__()
        self.d_h = d_h
        self.lin = nn.Linear(d_in, 2 * d_h)

    def gates(self, e):
        z, hc = self.lin(e).chunk(2, dim=-1)
        return torch.sigmoid(z), hc

    def forward(self, e, all_steps):
        z, hc = self.gates(e)
        zl, hl = z.unbind(1), hc.unbind(1)
        h = e.new_zeros(e.shape[0], self.d_h)
        out = []
        for t in range(len(zl)):
            h = (1.0 - zl[t]) * h + zl[t] * hl[t]
            if all_steps:
                out.append(h)
        return torch.stack(out, 1) if all_steps else h


class MinLSTM(nn.Module):
    """minLSTM (arXiv:2410.01201): f,i = sigmoid(Linear(x)), normalised to f',i' = f/(f+i), i/(f+i);
    h_t = f' h_{t-1} + i' h~_t with h~_t = Linear(x_t). Input-only gates."""

    def __init__(self, d_in, d_h):
        super().__init__()
        self.d_h = d_h
        self.lin = nn.Linear(d_in, 3 * d_h)

    def forward(self, e, all_steps):
        f, i, hc = self.lin(e).chunk(3, dim=-1)
        f, i = torch.sigmoid(f), torch.sigmoid(i)
        den = f + i
        f, i = f / den, i / den
        fl, il, hl = f.unbind(1), i.unbind(1), hc.unbind(1)
        h = e.new_zeros(e.shape[0], self.d_h)
        out = []
        for t in range(len(fl)):
            h = fl[t] * h + il[t] * hl[t]
            if all_steps:
                out.append(h)
        return torch.stack(out, 1) if all_steps else h


class StdGRU(nn.Module):
    """Standard GRU (nn.GRU): r_t, z_t and the candidate all depend on x_t AND h_{t-1}."""

    def __init__(self, d_in, d_h):
        super().__init__()
        self.g = nn.GRU(d_in, d_h, batch_first=True)

    def forward(self, e, all_steps):
        o, hn = self.g(e)
        return o if all_steps else hn[-1]


CELLS = {"gru": StdGRU, "mingru": MinGRU, "minlstm": MinLSTM}


class Model(nn.Module):
    """Embedding -> n_layers x (cell + LayerNorm) -> shared bottleneck -> k slot heads.
    Everything except the cell is identical across arms."""

    def __init__(self, cell, d_h, k, n_layers, v_data, d_in, d_read):
        super().__init__()
        self.k, self.v_data, self.n_layers = k, v_data, n_layers
        self.emb = nn.Embedding(1 + v_data, d_in)
        dims = [d_in] + [d_h] * n_layers
        self.cells = nn.ModuleList([CELLS[cell](dims[i], d_h) for i in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_h) for _ in range(n_layers)])
        self.proj = nn.Linear(d_h, d_read)
        self.heads = nn.Linear(d_read, k * v_data)

    def forward(self, x):
        e = self.emb(x)
        for i, (c, n) in enumerate(zip(self.cells, self.norms)):
            e = n(c(e, all_steps=(i < self.n_layers - 1)))
        return self.heads(F.gelu(self.proj(e))).view(-1, self.k, self.v_data)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def fit_width(cell, k, n_layers, target, P):
    """Pick the hidden width whose TOTAL parameter count is closest to `target`."""
    best = None
    for d_h in range(8, 1200, 2):
        n = n_params(Model(cell, d_h, k, n_layers, P["v_data"], P["d_in"], P["d_read"]))
        d = abs(n - target)
        if best is None or d < best[1]:
            best = (d_h, d, n)
        if n > target * 1.8:
            break
    return best[0], best[2]


# ----------------------------------------------------------------- train / eval
def evaluate(model, L, k, P, n_eval):
    """Fixed held-out set (disjoint eval seed). Returns per-token, exact-match, per-slot acc."""
    gen = torch.Generator().manual_seed(999999)
    B = 128
    tok = ex = n = 0
    slot = torch.zeros(k)
    with torch.no_grad():
        for _ in range(max(1, n_eval // B)):
            x, y = make_batch(B, L, k, P["v_data"], gen)
            pred = model(x).argmax(-1)
            corr = (pred == y).float()
            tok += corr.sum().item()
            slot += corr.sum(0)
            ex += corr.all(1).float().sum().item()
            n += B
    return tok / (n * k), ex / n, (slot / n).tolist(), n


def gate_stats(model, L, k, P):
    """For minGRU arms: mean input gate z on noise tokens vs on data tokens (layer 0)."""
    cell = model.cells[0]
    if not isinstance(cell, MinGRU):
        return None
    gen = torch.Generator().manual_seed(555555)
    with torch.no_grad():
        x, _ = make_batch(256, L, k, P["v_data"], gen)
        z, _ = cell.gates(model.emb(x))
        zm = z.mean(-1)                       # (B,T) mean gate over channels
        is_data = (x > 0)
        return {"z_noise": zm[~is_data].mean().item(), "z_data": zm[is_data].mean().item()}


def train_one(cell, d_h, n_layers, L, k, steps, seed, P):
    set_seeds(seed)
    model = Model(cell, d_h, k, n_layers, P["v_data"], P["d_in"], P["d_read"])
    opt = torch.optim.AdamW(model.parameters(), lr=P["lr"], weight_decay=P["weight_decay"])
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=P["lr"], total_steps=steps, pct_start=P["warmup_frac"])
    gen = torch.Generator().manual_seed(1000 + seed)
    t0 = time.time()
    losses = []
    for i in range(steps):
        x, y = make_batch(P["batch_size"], L, k, P["v_data"], gen)
        loss = F.cross_entropy(model(x).reshape(-1, P["v_data"]), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), P["grad_clip"])
        opt.step()
        sch.step()
        if i % 50 == 0 or i == steps - 1:
            losses.append(round(loss.item(), 4))
    train_s = time.time() - t0
    tok, exact, slot, n_ev = evaluate(model, L, k, P, P["eval_n"])
    return {
        "cell": cell, "n_layers": n_layers, "d_h": d_h, "n_params": n_params(model),
        "L": L, "k": k, "steps": steps, "seed": seed,
        "token_acc": round(tok, 4), "exact_acc": round(exact, 4),
        "slot_acc": [round(s, 4) for s in slot], "eval_n": n_ev,
        "train_sec": round(train_s, 1), "loss_curve": losses,
        "gate_stats": gate_stats(model, L, k, P),
    }


# ----------------------------------------------------------------- main
def main():
    cfg = load_config()
    P = cfg["params"]
    seed0 = int(cfg.get("seed", 0))
    set_seeds(seed0)
    t_start = time.time()

    cells = [tuple(c) for c in P["cells"]]
    probe_cells = [tuple(c) for c in P["probe_cells"]]
    steps = P["steps"]
    target = P["target_params"]

    # ---- fit widths so every arm has ~the same total parameter count -------------
    # (k changes the head size, so widths are fitted per k)
    widths = {}
    for (L, k) in cells:
        for cell, nl in [("gru", 1), ("mingru", 1), ("minlstm", 1), ("gru", 2), ("mingru", 2)]:
            key = (cell, nl, k)
            if key not in widths:
                widths[key] = fit_width(cell, k, nl, target, P)

    # ---- build the run queue, highest priority first -----------------------------
    queue = []   # (arm_name, cell, d_h, n_layers, L, k, steps, seed)
    for (L, k) in cells:                                    # 1. core, seed 0
        for arm, cell, nl in [("gru", "gru", 1), ("mingru_mp", "mingru", 1),
                              ("minlstm_mp", "minlstm", 1)]:
            d_h, _ = widths[(cell, nl, k)]
            queue.append((arm, (L, k), d_h, nl, steps, seed0))
        # capacity control: minGRU with the SAME hidden-state size as the GRU
        d_gru, _ = widths[("gru", 1, k)]
        queue.append(("mingru_ms", (L, k), d_gru, 1, steps, seed0))
    for (L, k) in probe_cells:                              # 2. extra-budget probe
        d_h, _ = widths[("mingru", 1, k)]
        queue.append(("mingru_long", (L, k), d_h, 1, steps * P["long_budget_mult"], seed0))
    for (L, k) in probe_cells:                              # 3. depth probe
        for arm, cell in [("gru_2l", "gru"), ("mingru_2l", "mingru")]:
            d_h, _ = widths[(cell, 2, k)]
            queue.append((arm, (L, k), d_h, 2, steps, seed0))
    for s in P["seeds"][1:]:                                # 4. seed robustness (headline pair)
        for (L, k) in cells:
            for arm, cell in [("gru", "gru"), ("mingru_mp", "mingru")]:
                d_h, _ = widths[(cell, 1, k)]
                queue.append((arm, (L, k), d_h, 1, steps, s))

    ARM_CELL = {"gru": "gru", "gru_2l": "gru", "mingru_mp": "mingru", "mingru_ms": "mingru",
                "mingru_long": "mingru", "mingru_2l": "mingru", "minlstm_mp": "minlstm"}

    runs, skipped = [], []
    for (arm, (L, k), d_h, nl, st, sd) in queue:
        if time.time() - t_start > P["time_cap_s"]:
            skipped.append({"arm": arm, "L": L, "k": k, "steps": st, "seed": sd})
            continue
        r = train_one(ARM_CELL[arm], d_h, nl, L, k, st, sd, P)
        r["arm"] = arm
        runs.append(r)
        print(f"[{time.time()-t_start:6.1f}s] {arm:12s} L={L:3d} k={k} seed={sd} "
              f"p={r['n_params']:6d} d={d_h:4d} steps={st:4d} "
              f"tok={r['token_acc']:.3f} exact={r['exact_acc']:.3f} ({r['train_sec']}s)", flush=True)

    # ---- aggregate ---------------------------------------------------------------
    def pick(arm, L, k, seed=None, steps_=None):
        for r in runs:
            if r["arm"] == arm and r["L"] == L and r["k"] == k \
               and (seed is None or r["seed"] == seed) and (steps_ is None or r["steps"] == steps_):
                return r
        return None

    def mean_over_seeds(arm, L, k, field):
        v = [r[field] for r in runs if r["arm"] == arm and r["L"] == L and r["k"] == k]
        return round(float(np.mean(v)), 4) if v else None

    chance_tok = 1.0 / P["v_data"]
    by_cell = {}
    for (L, k) in cells:
        cellkey = f"L{L}_k{k}"
        entry = {"chance_token_acc": round(chance_tok, 4),
                 "chance_exact_acc": float(f"{chance_tok ** k:.3g}")}
        for arm in ["gru", "mingru_mp", "mingru_ms", "minlstm_mp"]:
            r = pick(arm, L, k, seed=seed0)
            if r:
                entry[arm] = {"exact": r["exact_acc"], "token": r["token_acc"],
                              "n_params": r["n_params"], "d_h": r["d_h"],
                              "slot_acc": r["slot_acc"], "gate_stats": r["gate_stats"]}
        if "gru" in entry and "mingru_mp" in entry:
            entry["gap_exact_gru_minus_mingru"] = round(
                entry["gru"]["exact"] - entry["mingru_mp"]["exact"], 4)
            entry["gap_token_gru_minus_mingru"] = round(
                entry["gru"]["token"] - entry["mingru_mp"]["token"], 4)
        entry["seed_mean"] = {a: {"exact": mean_over_seeds(a, L, k, "exact_acc"),
                                  "token": mean_over_seeds(a, L, k, "token_acc")}
                              for a in ["gru", "mingru_mp"]}
        by_cell[cellkey] = entry

    probes = {}
    for (L, k) in probe_cells:
        ck = f"L{L}_k{k}"
        d = {}
        base = pick("mingru_mp", L, k, seed=seed0, steps_=steps)
        lng = pick("mingru_long", L, k, seed=seed0)
        if base:
            d["mingru_1x_steps"] = {"steps": base["steps"], "exact": base["exact_acc"],
                                    "token": base["token_acc"]}
        if lng:
            d["mingru_3x_steps"] = {"steps": lng["steps"], "exact": lng["exact_acc"],
                                    "token": lng["token_acc"]}
        for arm in ["gru_2l", "mingru_2l"]:
            r = pick(arm, L, k, seed=seed0)
            if r:
                d[arm] = {"exact": r["exact_acc"], "token": r["token_acc"],
                          "n_params": r["n_params"], "d_h": r["d_h"]}
        for arm, ref in [("gru", "gru_2l"), ("mingru_mp", "mingru_2l")]:
            r = pick(arm, L, k, seed=seed0, steps_=steps)
            if r:
                d[arm + "_1l"] = {"exact": r["exact_acc"], "token": r["token_acc"]}
        probes[ck] = d

    # headline
    headline = {}
    for (L, k) in cells:
        ck = f"L{L}_k{k}"
        e = by_cell[ck]
        if "gru" in e and "mingru_mp" in e:
            headline[ck] = {"gru_exact": e["gru"]["exact"],
                            "mingru_exact": e["mingru_mp"]["exact"],
                            "gap": e["gap_exact_gru_minus_mingru"]}

    metrics = {
        "task": "selective copy (single blank/noise token, k data tokens at random positions, "
                "targets = data values in order of appearance)",
        "v_data": P["v_data"], "steps": steps, "batch_size": P["batch_size"], "lr": P["lr"],
        "eval_n": P["eval_n"], "target_params": target,
        "eval_binomial_se_at_half": round(0.5 / np.sqrt(P["eval_n"]), 4),
        "widths": {f"{c}_{nl}l_k{k}": {"d_h": v[0], "n_params": v[1]}
                   for (c, nl, k), v in sorted(widths.items())},
        "headline_exact_match": headline,
        "by_cell": by_cell,
        "probes": probes,
        "runs": runs,
        "skipped_runs": skipped,
        "wall_clock_s": round(time.time() - t_start, 1),
    }

    make_chart(by_cell, probes, cells, probe_cells, P)

    results = {
        "id": cfg.get("id", "unknown"),
        "git_commit": git_sha(),
        "seed": seed0,
        "duration_sec": round(time.time() - t_start, 2),
        "metrics": metrics,
        "env": env_info(),
        "status": "done" if runs else "failed",
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({"headline": headline, "probes": probes,
                      "wall_clock_s": metrics["wall_clock_s"],
                      "skipped": skipped}, indent=2))


# ----------------------------------------------------------------- chart
def make_chart(by_cell, probes, cells, probe_cells, P):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = ["gru", "mingru_mp", "mingru_ms", "minlstm_mp"]
    labels = {"gru": "GRU (x+h gates)", "mingru_mp": "minGRU (x-only, matched params)",
              "mingru_ms": "minGRU (x-only, matched state size)",
              "minlstm_mp": "minLSTM (x-only, matched params)"}
    colors = {"gru": "#1f77b4", "mingru_mp": "#d62728", "mingru_ms": "#ff9896",
              "minlstm_mp": "#9467bd"}
    ck = [f"L{L}_k{k}" for (L, k) in cells]
    xs = np.arange(len(ck))
    w = 0.2

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))

    for i, (ax, field, title) in enumerate([
            (axes[0], "exact", "Exact-sequence accuracy"),
            (axes[1], "token", "Per-token accuracy")]):
        for j, a in enumerate(arms):
            vals = [by_cell[c].get(a, {}).get(field, np.nan) for c in ck]
            ax.bar(xs + (j - 1.5) * w, vals, w, label=labels[a], color=colors[a])
        if field == "token":
            ax.axhline(1.0 / P["v_data"], ls="--", c="k", lw=1, label="chance (1/8)")
        ax.set_xticks(xs)
        ax.set_xticklabels([c.replace("_", "  ") for c in ck])
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.set_ylabel("accuracy")
        ax.grid(axis="y", alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")

    # panel 3: per-slot accuracy on the hardest cell - the mechanism
    ax = axes[2]
    hard = f"L{cells[-1][0]}_k{cells[-1][1]}"
    for a in arms:
        sl = by_cell[hard].get(a, {}).get("slot_acc")
        if sl:
            ax.plot(np.arange(1, len(sl) + 1), sl, "o-", color=colors[a], label=labels[a])
    ax.axhline(1.0 / P["v_data"], ls="--", c="k", lw=1, label="chance (1/8)")
    ax.set_xlabel("output slot (1 = first data token seen)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Per-slot accuracy, {hard.replace('_', '  ')}")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[3]
    pk = [f"L{L}_k{k}" for (L, k) in probe_cells]
    probe_arms = [("mingru_1x_steps", "minGRU 1x steps", "#d62728"),
                  ("mingru_3x_steps", "minGRU 3x steps", "#ff7f0e"),
                  ("mingru_2l", "minGRU 2 layers", "#2ca02c"),
                  ("gru_2l", "GRU 2 layers", "#1f77b4")]
    xs2 = np.arange(len(pk))
    w2 = 0.2
    for j, (key, lab, col) in enumerate(probe_arms):
        vals = [probes.get(c, {}).get(key, {}).get("exact", np.nan) for c in pk]
        ax.bar(xs2 + (j - 1.5) * w2, vals, w2, label=lab, color=col)
    ax.set_xticks(xs2)
    ax.set_xticklabels([c.replace("_", "  ") for c in pk])
    ax.set_ylim(0, 1.05)
    ax.set_title("Probes: more steps vs more depth (exact match)")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Selective copy: do input-only gates (minGRU/minLSTM) cost anything vs a full GRU?",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(HERE / "chart.png", dpi=130)


if __name__ == "__main__":
    main()
