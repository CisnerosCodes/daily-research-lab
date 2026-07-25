"""TRM-style recursion at nano scale: outer refinement loop vs inner latent recursion.

Backlog item `trm-nano-sudoku` (Track A). Tests, at ~0.12M params on 4x4 Sudoku, which
half of the Tiny Recursive Model (arXiv:2510.04871) recursion actually buys accuracy:

  (a) outer_deepsup   -- model repeatedly refines a full candidate SOLUTION; all state
                         between steps passes through the 4-way answer distribution.
                         Loss at every outer step (TRM-style deep supervision).
  (a') outer_finalsup -- same loop, loss only at the final step (deep-supervision control).
  (b) inner_latent    -- model iterates a full-width latent z for the same number of core
                         applications, then decodes ONCE.
  (c) single-pass baseline == depth T=1, where (a), (a') and (b) all collapse to one
                         core application followed by one decode.

Matched compute: every arm performs exactly T applications of the SAME 2-layer MLP core and
has an identical parameter count. Only the path information takes between steps differs.

Deterministic, CPU-only (1 thread), writes results.json + chart.png.
Usage:  python run.py
"""
import os

# must be set before torch is imported -- this box has 2 cores shared with other agents
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json, random, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
N_CELLS, N_VALS, SIDE, BOX = 16, 4, 4, 2


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
def enumerate_solutions():
    """All valid 4x4 Sudoku grids (rows/cols/2x2 boxes each contain 1..4). Should be 288."""
    grid = [0] * N_CELLS
    out = []

    def ok(i, v):
        r, c = divmod(i, SIDE)
        br, bc = (r // BOX) * BOX, (c // BOX) * BOX
        for j in range(SIDE):
            if grid[r * SIDE + j] == v or grid[j * SIDE + c] == v:
                return False
        for dr in range(BOX):
            for dc in range(BOX):
                if grid[(br + dr) * SIDE + (bc + dc)] == v:
                    return False
        return True

    def rec(i):
        if i == N_CELLS:
            out.append(list(grid))
            return
        for v in range(1, N_VALS + 1):
            if ok(i, v):
                grid[i] = v
                rec(i + 1)
                grid[i] = 0

    rec(0)
    return np.array(out, dtype=np.int64)  # (288, 16)


def make_puzzles(sols_idx, all_sols, n_want, rng, gmin, gmax, forbidden):
    """Rejection-sample uniquely-solvable puzzles from the given solution indices.

    A puzzle is the givens vector (0 = blank). Uniqueness is checked against ALL 288
    grids, so the target is the one and only completion consistent with the givens.
    `forbidden` is a set of puzzle tuples to avoid (prevents train/test leakage).
    """
    puzzles, targets, seen = [], [], set()
    attempts, max_attempts = 0, n_want * 400
    while len(puzzles) < n_want and attempts < max_attempts:
        attempts += 1
        si = int(sols_idx[rng.integers(len(sols_idx))])
        sol = all_sols[si]
        k = int(rng.integers(gmin, gmax + 1))
        pos = rng.choice(N_CELLS, size=k, replace=False)
        g = np.zeros(N_CELLS, dtype=np.int64)
        g[pos] = sol[pos]
        key = tuple(g.tolist())
        if key in seen or key in forbidden:
            continue
        # unique-solution check: how many of the 288 grids agree on the given cells?
        if int((all_sols[:, pos] == sol[pos]).all(axis=1).sum()) != 1:
            continue
        seen.add(key)
        puzzles.append(g)
        targets.append(sol)
    return (np.stack(puzzles), np.stack(targets), seen)


def build_dataset(P, seed):
    rng = np.random.default_rng(seed)
    sols = enumerate_solutions()
    order = rng.permutation(len(sols))
    n_tr_grid = int(round(P["train_grid_frac"] * len(sols)))
    tr_g, te_g = order[:n_tr_grid], order[n_tr_grid:]

    xtr, ytr, tr_keys = make_puzzles(tr_g, sols, P["n_train"], rng,
                                     P["givens_min"], P["givens_max"], set())
    # held-out GRIDS: a genuine generalization test (solutions never seen in training)
    xun, yun, _ = make_puzzles(te_g, sols, P["n_test_unseen"], rng,
                               P["givens_min"], P["givens_max"], set())
    # seen GRIDS, unseen MASKS: the easier, memorization-friendly split
    xse, yse, _ = make_puzzles(tr_g, sols, P["n_test_seen"], rng,
                               P["givens_min"], P["givens_max"], tr_keys)
    meta = {"n_solution_grids": int(len(sols)), "n_train_grids": int(len(tr_g)),
            "n_test_grids": int(len(te_g)), "n_train_puzzles": int(len(xtr)),
            "n_test_unseen": int(len(xun)), "n_test_seen": int(len(xse)),
            "mean_blanks_train": float((xtr == 0).sum(1).mean())}
    to = lambda a: torch.from_numpy(a)
    return (to(xtr), to(ytr)), (to(xun), to(yun)), (to(xse), to(yse)), meta


# ----------------------------- model ---------------------------------------
class TRMNano(nn.Module):
    """One shared 2-layer MLP core, applied T times. Identical params for every arm."""

    def __init__(self, d_cell, d_hidden):
        super().__init__()
        D = N_CELLS * d_cell
        self.D, self.d_cell = D, d_cell
        self.emb_x = nn.Embedding(N_VALS + 1, d_cell)      # 0 = blank, 1..4 = given
        self.emb_y = nn.Parameter(torch.randn(N_VALS, d_cell) * 0.02)  # soft answer re-embed
        self.cell_bias = nn.Parameter(torch.zeros(N_CELLS, d_cell))
        self.ln_in = nn.LayerNorm(D)
        self.fc1 = nn.Linear(D, d_hidden)
        self.fc2 = nn.Linear(d_hidden, D)
        self.ln_out = nn.LayerNorm(D)
        self.head = nn.Linear(D, N_CELLS * N_VALS)

    def core(self, u):                       # ONE unit of compute
        return self.fc2(F.gelu(self.fc1(self.ln_in(u))))

    def enc_x(self, x):
        return (self.emb_x(x) + self.cell_bias).flatten(1)          # (B, D)

    def enc_y(self, probs):
        return (probs @ self.emb_y).flatten(1)                      # (B, D)

    def decode(self, h):
        return self.head(self.ln_out(h)).view(-1, N_CELLS, N_VALS)

    # ---- arm (a)/(a'): OUTER refinement loop --------------------------------
    # state between steps passes ONLY through the 4-way answer distribution
    def forward_outer(self, x, T):
        ex = self.enc_x(x)
        probs = x.new_full((x.shape[0], N_CELLS, N_VALS), 1.0 / N_VALS, dtype=ex.dtype)
        outs = []
        for _ in range(T):
            logits = self.decode(self.core(ex + self.enc_y(probs)))
            outs.append(logits)
            probs = logits.softmax(-1)
        return outs

    # ---- arm (b): INNER latent recursion ------------------------------------
    # state between steps is the full-width latent z; decode once at the end
    def forward_inner(self, x, T):
        ex = self.enc_x(x)
        z = torch.zeros_like(ex)
        for _ in range(T):
            z = z + self.core(ex + z)
        return [self.decode(z)]


# ----------------------------- train / eval --------------------------------
@torch.no_grad()
def evaluate(model, arm, T, x, y, chunk=400):
    model.eval()
    cell_hit, cell_tot, solved = 0, 0, 0
    for i in range(0, len(x), chunk):
        xb, yb = x[i:i + chunk], y[i:i + chunk]
        outs = model.forward_inner(xb, T) if arm == "inner_latent" else model.forward_outer(xb, T)
        pred = outs[-1].argmax(-1) + 1                     # classes 0..3 -> values 1..4
        blank = xb == 0
        corr = (pred == yb) & blank
        cell_hit += int(corr.sum())
        cell_tot += int(blank.sum())
        solved += int((corr.sum(1) == blank.sum(1)).sum())
    model.train()
    return cell_hit / cell_tot, solved / len(x)


def train_one(arm, T, seed, P, data, log):
    (xtr, ytr), (xun, yun), (xse, yse) = data
    set_seeds(seed)
    model = TRMNano(P["d_cell"], P["d_hidden"])
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=P["lr"], weight_decay=P["weight_decay"])
    gen = torch.Generator().manual_seed(seed * 7919 + 13)
    tgt_all = ytr - 1                                       # values 1..4 -> classes 0..3

    t0, capped, step, last_loss = time.time(), False, 0, float("nan")
    for step in range(P["steps"]):
        frac = (step + 1) / P["warmup"] if step < P["warmup"] else 1.0
        prog = max(0.0, (step - P["warmup"]) / max(1, P["steps"] - P["warmup"]))
        lr = P["lr"] * frac * (0.5 * (1 + np.cos(np.pi * prog)) if step >= P["warmup"] else 1.0)
        for g in opt.param_groups:
            g["lr"] = lr
        idx = torch.randint(0, len(xtr), (P["batch_size"],), generator=gen)
        xb, tb = xtr[idx], tgt_all[idx]
        outs = model.forward_inner(xb, T) if arm == "inner_latent" else model.forward_outer(xb, T)
        use = outs if arm == "outer_deepsup" else outs[-1:]
        loss = sum(F.cross_entropy(o.reshape(-1, N_VALS), tb.reshape(-1)) for o in use) / len(use)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), P["grad_clip"])
        opt.step()
        last_loss = float(loss.detach())
        if time.time() - t0 > P["time_cap_s_per_run"]:
            capped = True
            break
    train_s = time.time() - t0

    un_cell, un_solve = evaluate(model, arm, T, xun, yun)
    se_cell, se_solve = evaluate(model, arm, T, xse, yse)
    tr_cell, tr_solve = evaluate(model, arm, T, xtr[:800], ytr[:800])
    log(f"  {arm:15s} T={T} seed{seed}: params={n_params} steps={step + 1}"
        f"{' CAPPED' if capped else ''} ({train_s:5.1f}s) loss={last_loss:.3f} "
        f"| unseen cell={un_cell:.3f} solve={un_solve:.3f} "
        f"| seen cell={se_cell:.3f} solve={se_solve:.3f}")
    return {"arm": arm, "depth": T, "seed": seed, "n_params": n_params,
            "steps_run": step + 1, "time_capped": capped,
            "train_seconds": round(train_s, 1), "final_loss": round(last_loss, 4),
            "core_applications_per_forward": T,
            "train_core_applications": (step + 1) * T,
            "unseen_cell_acc": round(un_cell, 4), "unseen_solve_rate": round(un_solve, 4),
            "seen_cell_acc": round(se_cell, 4), "seen_solve_rate": round(se_solve, 4),
            "trainset_cell_acc": round(tr_cell, 4), "trainset_solve_rate": round(tr_solve, 4)}


# ----------------------------- main ----------------------------------------
def main():
    cfg = load_config()
    P = cfg["params"]
    set_seeds(int(cfg.get("seed", 0)))
    t0 = time.time()
    log = lambda s: print(s, flush=True)

    tr, un, se, meta = build_dataset(P, int(cfg.get("seed", 0)))
    log(f"data: {meta}")
    data = (tr, un, se)

    runs = []
    for T in P["depths"]:
        for arm in P["arms"]:
            for seed in P["seeds"]:
                runs.append(train_one(arm, T, seed, P, data, log))

    # ---- aggregate: mean over seeds per (arm, depth) ------------------------
    keys = ["unseen_cell_acc", "unseen_solve_rate", "seen_cell_acc",
            "seen_solve_rate", "trainset_solve_rate"]
    agg = {}
    for arm in P["arms"]:
        agg[arm] = {}
        for T in P["depths"]:
            rs = [r for r in runs if r["arm"] == arm and r["depth"] == T]
            agg[arm][str(T)] = {k: round(float(np.mean([r[k] for r in rs])), 4) for k in keys}
            agg[arm][str(T)]["unseen_solve_rate_std"] = round(
                float(np.std([r["unseen_solve_rate"] for r in rs])), 4)

    baseline = {k: round(float(np.mean([r[k] for r in runs if r["depth"] == 1])), 4) for k in keys}

    # the TRM-critique metric: how much of the best-depth score is already at depth 1?
    depth1_fraction = {}
    for arm in P["arms"]:
        best_T = max(P["depths"], key=lambda T: agg[arm][str(T)]["unseen_solve_rate"])
        best = agg[arm][str(best_T)]["unseen_solve_rate"]
        d1 = agg[arm]["1"]["unseen_solve_rate"]
        depth1_fraction[arm] = {
            "best_depth": best_T,
            "best_unseen_solve_rate": best,
            "depth1_unseen_solve_rate": d1,
            "frac_of_best_at_depth1": round(d1 / best, 4) if best > 0 else None,
            "abs_gain_over_depth1": round(best - d1, 4)}

    # head-to-head at matched compute (same T, same param count)
    outer_vs_inner = {
        str(T): {
            "delta_unseen_solve_rate_outer_minus_inner": round(
                agg["outer_deepsup"][str(T)]["unseen_solve_rate"]
                - agg["inner_latent"][str(T)]["unseen_solve_rate"], 4),
            "delta_unseen_cell_acc_outer_minus_inner": round(
                agg["outer_deepsup"][str(T)]["unseen_cell_acc"]
                - agg["inner_latent"][str(T)]["unseen_cell_acc"], 4),
            "delta_deepsup_minus_finalsup": round(
                agg["outer_deepsup"][str(T)]["unseen_solve_rate"]
                - agg["outer_finalsup"][str(T)]["unseen_solve_rate"], 4),
        } for T in P["depths"]}

    best_overall = max(runs, key=lambda r: r["unseen_solve_rate"])
    metrics = {
        "data": meta,
        "per_run": runs,
        "aggregate": agg,
        "single_pass_baseline_depth1": baseline,
        "depth1_fraction": depth1_fraction,
        "outer_vs_inner_at_matched_compute": outer_vs_inner,
        "best_config": {k: best_overall[k] for k in ("arm", "depth", "seed",
                                                    "unseen_solve_rate", "unseen_cell_acc")},
        "n_params": runs[0]["n_params"],
        "headline": ("unseen-grid solve rate @T=1/2/4/8 -- " + " | ".join(
            f"{arm}: " + "/".join(f"{agg[arm][str(T)]['unseen_solve_rate']:.3f}"
                                  for T in P["depths"]) for arm in P["arms"])),
    }

    # ----------------------------- chart ------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"outer_deepsup": "#1a7f64", "outer_finalsup": "#7fb069", "inner_latent": "#c95d3c"}
    labels = {"outer_deepsup": "(a) outer refinement + deep sup.",
              "outer_finalsup": "(a') outer refinement, final sup.",
              "inner_latent": "(b) inner latent recursion"}
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.5, 4.3), width_ratios=[2, 2, 1.5])
    xs = P["depths"]
    for ax, key, name in ((ax1, "unseen_solve_rate", "full-puzzle solve rate"),
                          (ax2, "unseen_cell_acc", "cell accuracy (blank cells)")):
        for arm in P["arms"]:
            ys = [agg[arm][str(T)][key] for T in xs]
            ax.plot(xs, ys, "o-", color=colors[arm], label=labels[arm], lw=2, ms=5)
        ax.axhline(baseline[key], color="0.6", ls="--", lw=1)
        ax.text(xs[-1], baseline[key] + 0.012, "single-pass baseline (T=1)",
                fontsize=7.5, color="0.45", ha="right")
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs); ax.set_xticklabels([str(t) for t in xs])
        ax.set_xlabel("recursion depth T (= core-block applications, matched across arms)")
        ax.set_ylabel(name)
        ax.set_ylim(-0.03, 1.03)
        ax.legend(frameon=False, fontsize=8, loc="best")
        ax.spines[["top", "right"]].set_visible(False)
    ax1.set_title("Held-out GRIDS: solve rate", fontsize=10)
    ax2.set_title("Held-out GRIDS: cell accuracy", fontsize=10)

    order = P["arms"]
    fr = [depth1_fraction[a]["frac_of_best_at_depth1"] or 0.0 for a in order]
    ax3.bar(range(len(order)), fr, color=[colors[a] for a in order])
    ax3.axhline(0.94, color="#b03a2e", ls="--", lw=1.2)
    ax3.text(len(order) - 0.5, 0.955, "2512.11847 claim: 94%", fontsize=8,
             color="#b03a2e", ha="right")
    for i, a in enumerate(order):
        ax3.text(i, fr[i] + 0.02, f"{fr[i]:.2f}", ha="center", fontsize=9)
    ax3.set_xticks(range(len(order)))
    ax3.set_xticklabels(["(a) outer\ndeep sup", "(a') outer\nfinal sup", "(b) inner\nlatent"],
                        fontsize=8)
    ax3.set_ylabel("solve rate at T=1 / best depth")
    ax3.set_ylim(0, 1.25)
    ax3.set_title("How much is already there at depth 1?", fontsize=10)
    ax3.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"TRM recursion at nano scale ({metrics['n_params'] / 1e3:.0f}k params, "
                 f"4x4 Sudoku, held-out solution grids, mean of {len(P['seeds'])} seeds)",
                 fontsize=11, y=1.03)
    fig.tight_layout()
    fig.savefig(HERE / "chart.png", dpi=160, bbox_inches="tight")

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
    print("depth1_fraction:", json.dumps(depth1_fraction, indent=2))
    print("outer_vs_inner:", json.dumps(outer_vs_inner, indent=2))


if __name__ == "__main__":
    main()
