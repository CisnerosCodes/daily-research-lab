"""Day 1 - Game of Life: when does test-time recursion stop generalizing?

A weight-tied recurrent cell learns Conway's Game of Life, trained to roll out up to K_train
steps (feeding its own soft prediction back in). We then test whether iterating the SAME cell
MORE times (extrapolation to depth 6) still predicts well. We sweep the cell's capacity: a big
cell learns the rule exactly (recursion is free), a starved cell cannot, so error compounds with
depth - and there we ask whether tied recursion extrapolates better than an untied per-step stack.

Deterministic, CPU-only. Writes results.json + chart.png.
"""
import json, os, random, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
HERE = Path(__file__).resolve().parent


def set_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE).decode().strip()
    except Exception:
        return "nogit"


def load_cfg():
    import yaml
    with open(HERE / "experiment.yaml") as f:
        return yaml.safe_load(f)


def life_step(g):
    import numpy as np
    nb = sum(np.roll(np.roll(g, i, 1), j, 2)
             for i in (-1, 0, 1) for j in (-1, 0, 1) if not (i == 0 and j == 0))
    return ((nb == 3) | ((g == 1) & (nb == 2))).astype("float32")


def make_data(n, size, density, k, rng):
    import numpy as np
    x0 = (rng.random((n, 1, size, size)) < density).astype("float32")
    frames = [x0]
    for _ in range(k):
        frames.append(life_step(frames[-1]))
    return x0, np.stack(frames[1:], axis=1)


def main():
    import numpy as np
    import torch
    import torch.nn as nn
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_cfg()
    p = cfg["params"]
    seed = int(cfg.get("seed", 0))
    set_seeds(seed)
    torch.set_num_threads(2)
    rng = np.random.default_rng(seed)
    t0 = time.time()

    size, dens = p["grid"], p["density"]
    k_train, k_eval = p["k_train"], p["k_eval"]
    Hs = p["hidden_sweep"]

    x_tr, y_tr = make_data(p["n_train"], size, dens, k_eval, rng)
    x_te, y_te = make_data(p["n_test"], size, dens, k_eval, rng)
    x_tr, y_tr = torch.tensor(x_tr), torch.tensor(y_tr)
    x_te, y_te = torch.tensor(x_te), torch.tensor(y_te)

    def cell(H):
        return nn.Sequential(
            nn.Conv2d(1, H, 3, padding=1, padding_mode="circular"), nn.ReLU(),
            nn.Conv2d(H, H, 1), nn.ReLU(),
            nn.Conv2d(H, 1, 1))

    class Tied(nn.Module):
        def __init__(s, H): super().__init__(); s.c = cell(H)
        def get(s, i): return s.c

    class Untied(nn.Module):
        def __init__(s, H): super().__init__(); s.cs = nn.ModuleList([cell(H) for _ in range(k_train)])
        def get(s, i): return s.cs[min(i, k_train - 1)]

    def rollout(model, x0, k, hard):
        state, logits = x0, []
        for i in range(k):
            lg = model.get(i)(state)
            logits.append(lg)
            prob = torch.sigmoid(lg)
            state = (prob > 0.5).float() if hard else prob
        return logits

    def train(model):
        opt = torch.optim.Adam(model.parameters(), lr=p["lr"])
        lossf = nn.BCEWithLogitsLoss()
        n = x_tr.shape[0]
        for _ in range(p["steps"]):
            idx = torch.randint(0, n, (p["batch"],))
            xb, yb = x_tr[idx], y_tr[idx]
            logits = rollout(model, xb, k_train, hard=False)
            loss = sum(lossf(logits[i], yb[:, i]) for i in range(k_train))
            opt.zero_grad(); loss.backward(); opt.step()
        return model

    @torch.no_grad()
    def eval_acc(model):
        logits = rollout(model, x_te, k_eval, hard=True)
        return [float(((torch.sigmoid(logits[i]) > 0.5).float() == y_te[:, i]).float().mean())
                for i in range(k_eval)]

    def nparm(m): return sum(v.numel() for v in m.parameters())

    steps_axis = list(range(1, k_eval + 1))
    dead_acc = [float((y_te[:, i] == 0).float().mean()) for i in range(k_eval)]
    results_by_H = {}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for ax, H in zip(axes.ravel(), Hs):
        set_seeds(seed)  # same init seed per capacity for fairness
        tied = train(Tied(H)); ta = eval_acc(tied)
        set_seeds(seed)
        unt = train(Untied(H)); ua = eval_acc(unt)
        results_by_H[str(H)] = {
            "tied_acc_per_step": [round(a, 4) for a in ta],
            "untied_acc_per_step": [round(a, 4) for a in ua],
            "tied_params": nparm(tied), "untied_params": nparm(unt),
            "tied_minus_untied_step6": round(ta[-1] - ua[-1], 4),
        }
        ax.plot(steps_axis, ta, "o-", label="tied recurrent")
        ax.plot(steps_axis, ua, "s--", label="untied stack")
        ax.plot(steps_axis, dead_acc, ":", color="gray", label="always-dead")
        ax.axvspan(0.5, k_train + 0.5, color="green", alpha=0.07)
        ax.set_title(f"hidden H={H}  (tied {nparm(tied)} params)", fontsize=9)
        ax.grid(alpha=0.3); ax.set_ylim(0.5, 1.01)
    for ax in axes[-1]:
        ax.set_xlabel("rollout step (test-time iterations)")
    for ax in axes[:, 0]:
        ax.set_ylabel("per-cell accuracy")
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Game of Life: when does test-time recursion still generalize?\n"
                 "(green band = depths seen in training; steps 4-6 = extrapolation)", fontsize=11)
    fig.tight_layout()
    fig.savefig(HERE / "chart.png", dpi=120)

    metrics = {"by_hidden": results_by_H,
               "always_dead_baseline_per_step": [round(a, 4) for a in dead_acc],
               "k_train": k_train, "k_eval": k_eval, "hidden_sweep": Hs}
    results = {"id": cfg["id"], "git_commit": git_sha(), "seed": seed,
               "duration_sec": round(time.time() - t0, 2), "metrics": metrics,
               "env": {"python": sys.version.split()[0], "torch": torch.__version__,
                       "numpy": np.__version__}, "status": "done"}
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"\ndone in {results['duration_sec']}s -> results.json + chart.png")


if __name__ == "__main__":
    main()
