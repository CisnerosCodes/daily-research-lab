"""Superposition phase diagram over (sparsity x within-pair correlation).

Anthropic-style toy model  x_hat = ReLU(W^T W x + b)  with n=16 features in 8 pairs,
m=4 hidden dims. Within a pair, the on/off indicators are correlated: per sample per
pair, with probability rho both members share ONE Bernoulli(p) coin (identical on/off),
else they draw independent coins -> indicator correlation is exactly rho. Active values
are independent Uniform[0,1], so even perfectly co-occurring features carry distinct
information and merging directions is lossy.

For each (density p, correlation rho, seed) we train the toy model online (fresh batch
each step) and measure:
  - frac_represented: fraction of features with ||W_i|| >= threshold
  - features_per_dim: represented count / m  (>1 -> superposition)
  - within-pair vs cross-pair mean |cos(W_i, W_j)|  (merge vs local-orthogonality)
  - mean interference sum_{j!=i} (W_i_hat . W_j)^2

Deterministic, CPU-only, writes results.json and chart.png.
Usage:  python run.py
"""
import json, os, random, subprocess, sys, time
from pathlib import Path

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


def sample_batch(gen, batch, n, p, rho, device):
    """Sparse features in pairs with within-pair indicator correlation exactly rho."""
    import torch
    n_pairs = n // 2
    # per-sample per-pair: share one coin with prob rho, else independent coins
    share = (torch.rand(batch, n_pairs, generator=gen, device=device) < rho)
    shared_coin = (torch.rand(batch, n_pairs, generator=gen, device=device) < p)
    own = (torch.rand(batch, n_pairs, 2, generator=gen, device=device) < p)
    ind = torch.where(share.unsqueeze(-1), shared_coin.unsqueeze(-1).expand(-1, -1, 2), own)
    ind = ind.reshape(batch, n).float()
    vals = torch.rand(batch, n, generator=gen, device=device)
    return ind * vals


def train_cell(seed, n, m, p, rho, steps, batch, lr):
    import torch
    device = "cpu"
    gen = torch.Generator(device).manual_seed(seed * 100003 + int(p * 1e4) * 31 + int(rho * 100))
    W = torch.empty(m, n)
    torch.nn.init.xavier_normal_(W, generator=gen)
    W = W.requires_grad_(True)
    b = torch.zeros(n, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)
    for _ in range(steps):
        x = sample_batch(gen, batch, n, p, rho, device)
        x_hat = torch.relu(x @ W.T @ W + b)
        loss = ((x - x_hat) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return W.detach(), b.detach(), loss.item()


def analyze(W, n, m, thresh):
    import torch
    norms = W.norm(dim=0)  # ||W_i|| per feature
    rep = (norms >= thresh)
    n_rep = int(rep.sum())
    Wn = W / norms.clamp_min(1e-8)  # unit columns
    C = (Wn.T @ Wn).abs()  # |cos| matrix, n x n
    within, cross = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if not (rep[i] and rep[j]):
                continue
            (within if j == i + 1 and i % 2 == 0 else cross).append(C[i, j].item())
    mean = lambda v: float(sum(v) / len(v)) if v else float("nan")
    # interference: for represented features, sum_j!=i (W_i_hat . W_j)^2
    interf = ((Wn.T @ W) ** 2).sum(dim=1) - norms ** 2  # subtract self term (Wn_i.W_i)^2 = ||W_i||^2
    interf_rep = float(interf[rep].mean()) if n_rep else float("nan")
    return {
        "n_represented": n_rep,
        "frac_represented": n_rep / n,
        "features_per_dim": n_rep / m,
        "within_pair_abs_cos": mean(within),
        "cross_pair_abs_cos": mean(cross),
        "mean_interference": interf_rep,
        "norms": [round(float(x), 4) for x in norms],
    }


def main():
    import numpy as np
    cfg = load_config()
    P = cfg["params"]
    seed = int(cfg.get("seed", 0))
    set_seeds(seed)
    t0 = time.time()

    n, m = P["n_features"], P["m_hidden"]
    densities, corrs, seeds = P["densities"], P["correlations"], P["seeds"]
    thresh = P["represented_norm_threshold"]

    grid = {}  # (p, rho) -> list of analyses over seeds
    for p in densities:
        for rho in corrs:
            runs = []
            for s in seeds:
                W, b, loss = train_cell(s, n, m, p, rho, P["steps"], P["batch"], P["lr"])
                a = analyze(W, n, m, thresh)
                a["final_loss"] = loss
                runs.append(a)
            grid[(p, rho)] = runs
            print(f"p={p:<5} rho={rho:<5} feats/dim={np.mean([r['features_per_dim'] for r in runs]):.2f} "
                  f"within|cos|={np.nanmean([r['within_pair_abs_cos'] for r in runs]):.3f} "
                  f"cross|cos|={np.nanmean([r['cross_pair_abs_cos'] for r in runs]):.3f}", flush=True)

    def cellmean(key):
        return np.array([[np.nanmean([r[key] for r in grid[(p, rho)]]) for rho in corrs] for p in densities])

    fpd = cellmean("features_per_dim")
    win = cellmean("within_pair_abs_cos")
    crs = cellmean("cross_pair_abs_cos")
    gap = win - crs  # >0: pairs merge; <0: pairs locally orthogonal vs background

    # ---- headline numbers at the sparsest density with full superposition signal
    i_sparse = 1  # p = densities[1] = 0.05, deep sparse regime
    headline = {
        "p_sparse": densities[i_sparse],
        "within_cos_rho0": round(float(win[i_sparse, 0]), 4),
        "within_cos_rho1": round(float(win[i_sparse, -1]), 4),
        "cross_cos_rho0": round(float(crs[i_sparse, 0]), 4),
        "cross_cos_rho1": round(float(crs[i_sparse, -1]), 4),
        "fpd_rho0": round(float(fpd[i_sparse, 0]), 4),
        "fpd_rho1": round(float(fpd[i_sparse, -1]), 4),
    }

    # ---------------- chart ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.suptitle("Toy-model superposition vs feature density and within-pair correlation "
                 f"(n={n} features in pairs, m={m} dims, {len(seeds)} seeds)", fontsize=11)

    def heat(ax, M, title, cmap, vmin=None, vmax=None, fmt="{:.2f}"):
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
        ax.set_xticks(range(len(corrs)), [f"{r:g}" for r in corrs])
        ax.set_yticks(range(len(densities)), [f"{d:g}" for d in densities])
        ax.set_xlabel("within-pair correlation ρ")
        ax.set_ylabel("feature density p (sparse → dense)")
        ax.set_title(title, fontsize=10)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if np.isfinite(v):
                    lo, hi = im.norm.vmin, im.norm.vmax
                    frac = (v - lo) / (hi - lo + 1e-9)
                    ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7.5,
                            color="white" if frac > 0.55 else "#1a1a2e")
        fig.colorbar(im, ax=ax, shrink=0.85)

    heat(axes[0], fpd, "features per dimension\n(>1 = superposition)", "Blues", vmin=1.0, vmax=4.0)
    lim = max(0.05, float(np.nanmax(np.abs(gap))))
    heat(axes[1], gap, "within-pair − cross-pair |cos|\n(+merge / −local orthogonality)", "RdBu_r",
         vmin=-lim, vmax=lim)
    ax = axes[2]
    colors = ["#4477aa", "#66a3d2", "#997700", "#bb5566", "#222255"]
    for j, rho in enumerate(corrs):
        ax.plot(densities, win[:, j], "-o", color=colors[j], ms=4, lw=2, label=f"ρ={rho:g}")
    ax.plot(densities, crs.mean(axis=1), "--", color="#888888", lw=1.5, label="cross-pair (mean)")
    ax.set_xscale("log")
    ax.set_xlabel("feature density p (log)")
    ax.set_ylabel("mean within-pair |cos|")
    ax.set_title("do correlated pairs merge or orthogonalize?", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(HERE / "chart.png", dpi=160)

    metrics = {
        "headline": headline,
        "features_per_dim_grid": [[round(float(v), 3) for v in row] for row in fpd],
        "within_pair_abs_cos_grid": [[round(float(v), 4) for v in row] for row in win],
        "cross_pair_abs_cos_grid": [[round(float(v), 4) for v in row] for row in crs],
        "grid_axes": {"rows_density": densities, "cols_correlation": corrs},
    }

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
    print("headline:", json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()
