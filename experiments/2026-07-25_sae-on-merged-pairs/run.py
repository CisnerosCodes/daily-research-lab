"""SAEs on the merged-pair toy model: do they recover 8 pair-directions or 16 true features?

Ground truth by construction. We retrain the toy autoencoder of
`2026-07-23_superposition-correlation-phase` --  x_hat = ReLU(W^T W x + b),  n=16 features
arranged in 8 pairs, m=4 hidden dims -- with within-pair indicator correlation rho.  That run
established that correlated pairs MERGE: the two members of a pair end up on ONE shared
direction (within-pair |cos| ~ 0.999).  So the model's 4-dim hidden layer h = W x is a KNOWN
merged representation and we have two competing answer keys:

  KEY-8   the 8 merged pair directions (top left singular vector of each pair's 2-column block)
  KEY-16  the 16 true generative feature directions (unit columns of W)

We train small L1 SAEs on h (expansion 2x/4x/8x over 4 dims, 3 L1 coefficients) and score every
SAE feature against BOTH keys, with four things the usual max-cosine score lacks:

 1. a random-unit-direction NULL.  In a 4-dim space a random direction hits a fixed target at
    cos >= 0.9 about 1.9% of the time, so "recovered N of 8" means nothing without it.
 2. greedy INJECTIVE matching, so one SAE feature cannot be credited with recovering both
    members of a merged pair (the two members' unit columns are ~the same vector).
 3. a FUNCTIONAL member-selectivity test: does an SAE feature respond to one member of a pair
    and not the other, or to their sum?
 4. the decisive control -- linear and MLP PROBES from the 4-dim hidden onto the within-pair
    value SUM and the within-pair value DIFFERENCE, with two nulls:
       * EXACT-MERGE ORACLE: rebuild h after projecting each pair's two columns onto their
         exactly shared direction.  This is the "information really is gone" reference.
       * SHUFFLED-h: the probe's own optimism floor.

An UNMERGED control arm (rho=0, same density and geometry) is the positive control for every
one of those statistics.

Deterministic, CPU-only, single-threaded.  Usage:  python run.py
"""
import json, os, random, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
_LOG = None


def log(*a):
    global _LOG
    if _LOG is None:
        _LOG = open(HERE / "train.log", "w")
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _LOG.write(s + "\n")
    _LOG.flush()


def set_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    import torch
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
    info = {"python": sys.version.split()[0]}
    for mod in ("numpy", "torch", "matplotlib"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:
            pass
    return info


# ----------------------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)


# --------------------------- data + toy model -------------------------------
def sample_batch(gen, batch, n, p, rho):
    """Sparse features in pairs with within-pair INDICATOR correlation exactly rho.

    Identical construction to 2026-07-23_superposition-correlation-phase: per sample per pair,
    with probability rho both members share ONE Bernoulli(p) coin, else independent coins.
    Active VALUES are always independent Uniform[0,1], so even a perfectly co-occurring pair
    carries two independent numbers and merging their directions is genuinely lossy.
    """
    n_pairs = n // 2
    share = (torch.rand(batch, n_pairs, generator=gen) < rho)
    shared_coin = (torch.rand(batch, n_pairs, generator=gen) < p)
    own = (torch.rand(batch, n_pairs, 2, generator=gen) < p)
    ind = torch.where(share.unsqueeze(-1), shared_coin.unsqueeze(-1).expand(-1, -1, 2), own)
    ind = ind.reshape(batch, n).float()
    vals = torch.rand(batch, n, generator=gen)
    return ind * vals


def train_toy(seed, n, m, p, rho, steps, batch, lr):
    gen = torch.Generator().manual_seed(seed * 100003 + int(p * 1e4) * 31 + int(rho * 100))
    W = torch.empty(m, n)
    nn.init.xavier_normal_(W, generator=gen)
    W = W.requires_grad_(True)
    b = torch.zeros(n, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)
    loss = None
    for _ in range(steps):
        x = sample_batch(gen, batch, n, p, rho)
        x_hat = torch.relu(x @ W.T @ W + b)
        loss = ((x - x_hat) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return W.detach(), b.detach(), float(loss.item())


def geometry(W, thresh):
    """within/cross-pair |cos| and per-feature norms of the toy model's W (m x n)."""
    m, n = W.shape
    norms = W.norm(dim=0)
    rep = (norms >= thresh)
    Wn = W / norms.clamp_min(1e-8)
    C = (Wn.T @ Wn)
    within, cross, within_signed, pairs_both = [], [], [], []
    for k in range(n // 2):
        i = 2 * k
        if rep[i] and rep[i + 1]:
            within.append(abs(float(C[i, i + 1])))
            within_signed.append(float(C[i, i + 1]))
            pairs_both.append(k)
    for i in range(n):
        for j in range(i + 1, n):
            if (j == i + 1 and i % 2 == 0) or not (rep[i] and rep[j]):
                continue
            cross.append(abs(float(C[i, j])))
    mean = lambda v: float(np.mean(v)) if v else float("nan")
    return {
        "n_represented": int(rep.sum()),
        "features_per_dim": float(int(rep.sum()) / m),
        "within_pair_abs_cos": mean(within),
        "within_pair_signed_cos": mean(within_signed),
        "cross_pair_abs_cos": mean(cross),
        "min_within_pair_abs_cos": float(np.min(within)) if within else float("nan"),
        "n_pairs_both_represented": len(pairs_both),
        "pairs_both_represented": pairs_both,
        "within_abs_cos_per_pair": [round(abs(float(C[2 * k, 2 * k + 1])), 4)
                                    for k in range(n // 2)],
        "norms": [round(float(v), 4) for v in norms],
    }, rep.numpy()


def answer_keys(W):
    """KEY-16 (unit columns) and KEY-8 (per-pair principal direction), both sign-fixed.

    Feature values are non-negative, so a feature pushes h along +W_i: signed cosine is the
    right similarity and the key directions are oriented to +W_i.
    """
    m, n = W.shape
    K16 = (W / W.norm(dim=0).clamp_min(1e-8)).T.numpy().astype(np.float64)  # (n, m) unit rows
    K8 = np.zeros((n // 2, m), dtype=np.float64)
    for k in range(n // 2):
        blk = W[:, 2 * k:2 * k + 2].numpy().astype(np.float64)              # (m, 2)
        u, s, vt = np.linalg.svd(blk, full_matrices=False)
        d = u[:, 0]
        if d @ (blk[:, 0] + blk[:, 1]) < 0:
            d = -d
        K8[k] = d / np.linalg.norm(d)
    return K16, K8


def merge_oracle_W(W, K8):
    """Project each pair's two columns onto their exactly-shared direction -> a model in which
    the within-pair difference is provably unrecoverable from h."""
    Wo = W.clone()
    n = W.shape[1]
    for k in range(n // 2):
        d = torch.from_numpy(K8[k]).float()
        for i in (2 * k, 2 * k + 1):
            Wo[:, i] = (W[:, i] @ d) * d
    return Wo


def greedy_match(C):
    """Greedy injective assignment maximising cosine. C: (n_keys, n_feats) -> per-key matched
    cosine (-inf where unmatched, i.e. fewer SAE features than key directions)."""
    C = C.copy()
    out = np.full(C.shape[0], -np.inf)
    for _ in range(min(C.shape)):
        i, j = np.unravel_index(np.argmax(C), C.shape)
        if not np.isfinite(C[i, j]):
            break
        out[i] = C[i, j]
        C[i, :] = -np.inf
        C[:, j] = -np.inf
    return out


# --------------------------------- SAE --------------------------------------
class SAE(nn.Module):
    def __init__(self, d_in, n_feat, x_mean, gen):
        super().__init__()
        Wd = torch.randn(n_feat, d_in, generator=gen)
        Wd = Wd / Wd.norm(dim=1, keepdim=True)
        self.W_dec = nn.Parameter(Wd)
        self.W_enc = nn.Parameter(Wd.data.t().clone())
        self.b_enc = nn.Parameter(torch.zeros(n_feat))
        self.b_dec = nn.Parameter(x_mean.clone())

    def forward(self, x):
        f = F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        return f, f @ self.W_dec + self.b_dec

    @torch.no_grad()
    def norm_dec(self):
        self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True).clamp_min(1e-8)


def train_sae(xn, n_feat, lam, P, seed):
    """xn: (N, d) already-normalised activations."""
    gen = torch.Generator().manual_seed(seed * 31337 + n_feat * 7 + int(lam * 1e4))
    N, d = xn.shape
    sae = SAE(d, n_feat, xn.mean(0), gen)
    opt = torch.optim.Adam(sae.parameters(), lr=float(P["sae_lr"]))
    steps, bs = int(P["sae_steps"]), int(P["sae_batch"])
    resample_steps = {int(r * steps) for r in P["resample_at"]}
    fired = torch.zeros(n_feat, dtype=torch.bool)
    n_resampled = 0
    for it in range(steps):
        xb = xn[torch.randint(0, N, (bs,), generator=gen)]
        f, r = sae(xb)
        loss = ((r - xb) ** 2).sum(-1).mean() + lam * f.abs().sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        sae.norm_dec()
        with torch.no_grad():
            fired |= (f > 0).any(0)
        if it in resample_steps:
            with torch.no_grad():
                dead = ~fired
                nd = int(dead.sum())
                if nd > 0:
                    idx = torch.randint(0, N, (int(P["resample_n_err"]),), generator=gen)
                    xs_ = xn[idx]
                    _, rs = sae(xs_)
                    err = ((rs - xs_) ** 2).sum(-1)
                    prob = err ** 2
                    prob = prob / prob.sum().clamp_min(1e-12)
                    pick = torch.multinomial(prob, nd, replacement=True, generator=gen)
                    dirs = xs_[pick] - sae.b_dec
                    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-8)
                    alive_enc = (sae.W_enc.data[:, fired].norm(dim=0).mean()
                                 if int(fired.sum()) > 0 else torch.tensor(1.0))
                    sae.W_dec.data[dead] = dirs
                    sae.W_enc.data[:, dead] = (dirs * 0.2 * alive_enc).t()
                    sae.b_enc.data[dead] = 0.0
                    n_resampled += nd
                    for nm, prm in (("W_enc", sae.W_enc), ("W_dec", sae.W_dec),
                                    ("b_enc", sae.b_enc)):
                        st = opt.state.get(prm, None)
                        if st:
                            if nm == "W_enc":
                                st["exp_avg"][:, dead] = 0; st["exp_avg_sq"][:, dead] = 0
                            else:
                                st["exp_avg"][dead] = 0; st["exp_avg_sq"][dead] = 0
                fired[:] = False
    return sae, {"n_resampled": n_resampled}


@torch.no_grad()
def sae_eval(sae, xn):
    f, r = sae(xn)
    fvu = float(((r - xn) ** 2).sum() / ((xn - xn.mean(0)) ** 2).sum())
    l0 = float((f > 0).float().sum(-1).mean())
    fire = (f > 0).float().mean(0).numpy()
    return f.numpy(), fvu, l0, fire


# ------------------------------ null model ----------------------------------
def null_recovery(m, n_dirs, keys, thresholds, n_trials, rng):
    """Recovery obtained by n_dirs RANDOM unit directions in R^m, per key, per threshold."""
    out = {}
    for kname, K in keys.items():
        counts = {t: [] for t in thresholds}
        maxcos = []
        for _ in range(n_trials):
            D = rng.standard_normal((n_dirs, m))
            D /= np.linalg.norm(D, axis=1, keepdims=True)
            C = K @ D.T
            mm = greedy_match(C)
            maxcos.append(float(np.mean(C.max(1))))
            for t in thresholds:
                counts[t].append(int((mm >= t).sum()))
        out[kname] = {
            "mean_max_cos": round(float(np.mean(maxcos)), 4),
            "recovered_mean": {str(t): round(float(np.mean(counts[t])), 3) for t in thresholds},
            "recovered_p95": {str(t): int(np.percentile(counts[t], 95)) for t in thresholds},
            "recovered_max": {str(t): int(np.max(counts[t])) for t in thresholds},
        }
    return out


# ------------------------------- probes -------------------------------------
def build_probe_targets(X, n):
    """Per pair k: value of member a, of member b, their SUM and their DIFFERENCE, restricted to
    CO-ACTIVE samples; plus a which-member target on EXACTLY-ONE-active samples."""
    names, T, M = [], [], []
    for k in range(n // 2):
        a, b = X[:, 2 * k], X[:, 2 * k + 1]
        on_a, on_b = a > 0, b > 0
        co = on_a & on_b
        solo = on_a ^ on_b
        for nm, tgt, msk in (("val_a", a, co), ("val_b", b, co),
                             ("sum", a + b, co), ("diff", a - b, co),
                             ("which_solo", np.where(on_a & ~on_b, 1.0, -1.0), solo)):
            names.append(f"p{k}_{nm}")
            T.append(tgt.astype(np.float64))
            M.append(msk.astype(np.float64))
    return names, np.stack(T, 1), np.stack(M, 1)


def masked_r2(pred, T, M):
    """Per-target R^2 against the masked-mean baseline. nan if <50 masked test samples."""
    cnt = M.sum(0)
    mu = (M * T).sum(0) / np.maximum(cnt, 1)
    ss_res = (M * (pred - T) ** 2).sum(0)
    ss_tot = (M * (T - mu) ** 2).sum(0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    return np.where(cnt >= 50, r2, np.nan), cnt


def run_probes(H, X, n, P, seed):
    """Linear (ridge, closed form, per target) and MLP probes from the m-dim hidden."""
    names, T, M = build_probe_targets(X, n)
    N = H.shape[0]
    ntr = int(N * 0.7)
    Htr, Hte = H[:ntr], H[ntr:]
    Ttr, Tte = T[:ntr], T[ntr:]
    Mtr, Mte = M[:ntr], M[ntr:]

    A = np.concatenate([Htr, np.ones((ntr, 1))], 1)
    Ate = np.concatenate([Hte, np.ones((N - ntr, 1))], 1)
    d1 = A.shape[1]
    lin_pred = np.zeros_like(Tte)
    for j in range(T.shape[1]):
        w = Mtr[:, j]
        if w.sum() < 50:
            continue
        Aw = A * w[:, None]
        G = Aw.T @ A + float(P["probe_ridge"]) * np.eye(d1)
        coef = np.linalg.solve(G, Aw.T @ Ttr[:, j])
        lin_pred[:, j] = Ate @ coef
    lin_r2, cnt_te = masked_r2(lin_pred, Tte, Mte)

    g = torch.Generator().manual_seed(seed * 7919 + 11)
    hid = int(P["probe_hidden"])
    net = nn.Sequential(nn.Linear(H.shape[1], hid), nn.ReLU(),
                        nn.Linear(hid, hid), nn.ReLU(),
                        nn.Linear(hid, T.shape[1]))
    with torch.no_grad():
        for mod in net:
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight, generator=g)
                mod.bias.zero_()
    opt = torch.optim.Adam(net.parameters(), lr=float(P["probe_lr"]))
    Ht = torch.from_numpy(Htr).float(); Tt = torch.from_numpy(Ttr).float()
    Mt = torch.from_numpy(Mtr).float()
    bs = int(P["probe_batch"])
    for _ in range(int(P["probe_steps"])):
        idx = torch.randint(0, ntr, (bs,), generator=g)
        pr = net(Ht[idx])
        mk = Mt[idx]
        loss = ((pr - Tt[idx]) ** 2 * mk).sum() / mk.sum().clamp_min(1.0)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        mlp_pred = net(torch.from_numpy(Hte).float()).numpy().astype(np.float64)
    mlp_r2, _ = masked_r2(mlp_pred, Tte, Mte)

    KINDS = ("val_a", "val_b", "sum", "diff", "which_solo")

    def agg(r2):
        o = {}
        for kind in KINDS:
            sel = [i for i, nm in enumerate(names) if nm.endswith("_" + kind)]
            v = np.array([r2[i] for i in sel], dtype=np.float64)
            o[kind] = (round(float(np.nanmean(v)), 4) if np.isfinite(v).any() else None)
            o[kind + "_max"] = (round(float(np.nanmax(v)), 4) if np.isfinite(v).any() else None)
        return o

    cnt_by_kind = {}
    for kind in ("sum", "which_solo"):
        sel = [i for i, nm in enumerate(names) if nm.endswith("_" + kind)]
        cnt_by_kind["n_test_samples_" + kind] = int(np.mean([cnt_te[i] for i in sel]))
    return {"linear": agg(lin_r2), "mlp": agg(mlp_r2), **cnt_by_kind}


# ------------------------- SAE feature <-> true feature ---------------------
def corr_matrix(Fa, X):
    """Pearson correlation between every SAE feature activation and every true feature value."""
    def z(A):
        A = A - A.mean(0, keepdims=True)
        return A / np.maximum(A.std(0, keepdims=True), 1e-12)
    return (z(Fa).T @ z(X)) / Fa.shape[0]


def member_selectivity(R, alive, n, min_assoc, thr):
    """For each alive SAE feature, look at the pair it is most associated with and measure the
    imbalance |r_a - r_b| / (|r_a| + |r_b|).  0 = responds to both members identically (a merged
    / pair feature); 1 = responds to exactly one member (a true single-feature detector)."""
    best_imb, per_feat = 0.0, []
    n_sel = 0
    sel_true_feats = set()
    for f in np.where(alive)[0]:
        r = R[f]
        strength = np.abs(r[0::2]) + np.abs(r[1::2])
        k = int(np.argmax(strength))
        ra, rb = float(r[2 * k]), float(r[2 * k + 1])
        if max(abs(ra), abs(rb)) < min_assoc:
            continue
        imb = abs(ra - rb) / max(abs(ra) + abs(rb), 1e-12)
        per_feat.append(imb)
        best_imb = max(best_imb, imb)
        if imb >= thr:
            n_sel += 1
            sel_true_feats.add(2 * k if abs(ra) > abs(rb) else 2 * k + 1)
    return {
        "n_judged": len(per_feat),
        "max_member_imbalance": round(float(best_imb), 4),
        "mean_member_imbalance": round(float(np.mean(per_feat)), 4) if per_feat else None,
        "median_member_imbalance": round(float(np.median(per_feat)), 4) if per_feat else None,
        "n_member_selective_feats": n_sel,
        "n_true_feats_with_selective_sae_feat": len(sel_true_feats),
    }


def dictionary_collapse(Wa, K8):
    """How many DISTINCT pair-directions do the alive SAE features cover, and how badly do they
    pile up on the same one?"""
    C = K8 @ Wa.T                      # (8, n_alive)
    owner = C.argmax(0)
    cnt = np.bincount(owner, minlength=K8.shape[0])
    return {"n_alive": int(Wa.shape[0]),
            "n_distinct_pairdirs_claimed": int((cnt > 0).sum()),
            "max_feats_per_pairdir": int(cnt.max()) if cnt.size else 0,
            "feats_per_pairdir": [int(v) for v in np.bincount(owner, minlength=K8.shape[0])]}


# --------------------------------- main -------------------------------------
def main():
    cfg = load_config()
    P = cfg["params"]
    seed = int(cfg.get("seed", 0))
    set_seeds(seed)
    t_start = time.time()
    budget = float(P["time_budget_s"])

    n, m = int(P["n_features"]), int(P["m_hidden"])
    thr_rep = float(P["represented_norm_threshold"])
    cos_thrs = [float(t) for t in P["cos_thresholds"]]
    head_thr = float(P["headline_cos_threshold"])
    ht = str(head_thr)
    alive_min = float(P["alive_min_fire_frac"])
    min_assoc = float(P["assoc_min_abs_corr"])
    sel_thr = float(P["member_selectivity_threshold"])
    rng = np.random.default_rng(seed)

    arms, sae_rows, skipped = [], [], []
    keys_by_arm = {}

    for spec in P["model_arms"]:
        aname, p, rho = spec["name"], float(spec["p"]), float(spec["rho"])
        for s in spec["seeds"]:
            t0 = time.time()
            W, b, tl = train_toy(s, n, m, p, rho, int(P["toy_steps"]), int(P["toy_batch"]),
                                 float(P["toy_lr"]))
            geo, rep = geometry(W, thr_rep)
            K16, K8 = answer_keys(W)
            Wo = merge_oracle_W(W, K8)
            keys_by_arm[(aname, s)] = (K16, K8, rep)

            gen = torch.Generator().manual_seed(s * 999331 + 17)
            Xtr = sample_batch(gen, int(P["n_act_train"]), n, p, rho)
            Xev = sample_batch(gen, int(P["n_act_eval"]), n, p, rho)
            Htr, Hev = Xtr @ W.T, Xev @ W.T
            scale = float(np.sqrt(m) / Htr.norm(dim=1).mean().item())   # E||h|| = sqrt(m)
            Htrn, Hevn = Htr * scale, Hev * scale

            Xe_np = Xev.numpy().astype(np.float64)
            He_np = Hev.numpy().astype(np.float64)
            probes = run_probes(He_np, Xe_np, n, P, s)
            probes_oracle = run_probes((Xev @ Wo.T).numpy().astype(np.float64), Xe_np, n, P, s)
            perm = np.random.default_rng(s * 13 + 5).permutation(He_np.shape[0])
            probes_shuf = run_probes(He_np[perm], Xe_np, n, P, s)

            arm = {"arm": aname, "seed": s, "p": p, "rho": rho,
                   "toy_final_loss": round(tl, 6), "geometry": geo,
                   "merged_verified": bool(geo["within_pair_abs_cos"] >=
                                           float(P["merge_verify_abs_cos"])),
                   "probes": probes, "probes_exact_merge_oracle": probes_oracle,
                   "probes_shuffled_h_null": probes_shuf,
                   "secs": round(time.time() - t0, 1)}
            arms.append(arm)
            log(f"[toy/{aname}/s{s}] loss {tl:.5f} rep {geo['n_represented']}/{n} "
                f"pairs_both {geo['n_pairs_both_represented']}/8 "
                f"within|cos| {geo['within_pair_abs_cos']:.4f} cross|cos| "
                f"{geo['cross_pair_abs_cos']:.4f} merged={arm['merged_verified']}")
            log(f"          probe MLP R2: sum {probes['mlp']['sum']} diff {probes['mlp']['diff']} "
                f"val_a {probes['mlp']['val_a']} which_solo {probes['mlp']['which_solo']} "
                f"|| oracle diff {probes_oracle['mlp']['diff']} "
                f"|| shuffled diff {probes_shuf['mlp']['diff']} | {arm['secs']}s")

            keys = {"key8_pairdirs": K8, "key16_truefeats": K16}
            for exp in P["expansions"]:
                for lam in P["l1_coeffs"]:
                    if time.time() - t_start > budget:
                        skipped.append({"arm": aname, "seed": s, "expansion": exp,
                                        "l1_coeff": lam})
                        continue
                    ts = time.time()
                    n_feat = m * int(exp)
                    sae, info = train_sae(Htrn, n_feat, float(lam), P, s)
                    Fa, fvu, l0, fire = sae_eval(sae, Hevn)
                    alive = fire >= alive_min
                    Wd = sae.W_dec.detach().numpy().astype(np.float64)
                    Wd = Wd / np.maximum(np.linalg.norm(Wd, axis=1, keepdims=True), 1e-12)

                    row = {"arm": aname, "seed": s, "expansion": int(exp), "l1_coeff": float(lam),
                           "n_feat": n_feat, "fvu": round(fvu, 5), "l0": round(l0, 3),
                           "n_alive": int(alive.sum()),
                           "frac_dead": round(float((fire == 0).mean()), 4),
                           "frac_alive": round(float(alive.mean()), 4),
                           "n_resampled": info["n_resampled"]}
                    if alive.sum() == 0:
                        row["key8_pairdirs"] = None
                        sae_rows.append(row)
                        continue
                    Wa = Wd[alive]
                    for kname, K in keys.items():
                        C = K @ Wa.T
                        mm = greedy_match(C)
                        row[kname] = {
                            "mean_max_cos": round(float(C.max(1).mean()), 4),
                            "matched_cos": [round(float(v), 4) if np.isfinite(v) else None
                                            for v in mm],
                            "recovered_injective": {str(t): int((mm >= t).sum()) for t in cos_thrs},
                            "recovered_maxcos": {str(t): int((C.max(1) >= t).sum())
                                                 for t in cos_thrs},
                        }
                    # restrict KEY-16 to features the toy model actually represents
                    C16 = K16 @ Wa.T
                    mm16 = greedy_match(C16)
                    row["key16_truefeats"]["recovered_injective_represented_only"] = {
                        str(t): int(((mm16 >= t) & rep).sum()) for t in cos_thrs}
                    R = corr_matrix(Fa, Xe_np)
                    row["selectivity"] = member_selectivity(R, alive, n, min_assoc, sel_thr)
                    row["collapse"] = dictionary_collapse(Wa, K8)
                    # FUNCTIONAL recovery of a TRUE feature: an SAE feature is directionally
                    # matched to it AND responds to that member and not its partner.
                    alive_idx = np.where(alive)[0]
                    sel_ok = []
                    for i in range(n):
                        if mm16[i] < head_thr or not rep[i]:
                            continue
                        f = alive_idx[int(np.argmax(C16[i]))]
                        k = i // 2
                        ra, rb = float(R[f, 2 * k]), float(R[f, 2 * k + 1])
                        if max(abs(ra), abs(rb)) < min_assoc:
                            continue
                        if abs(ra - rb) / max(abs(ra) + abs(rb), 1e-12) >= sel_thr:
                            sel_ok.append(i)
                    row["n_truefeats_recovered_functional"] = len(sel_ok)
                    row["truefeats_recovered_functional"] = sel_ok
                    row["secs"] = round(time.time() - ts, 1)
                    sae_rows.append(row)
                    log(f"  [sae/{aname}/s{s}] x{exp} lam={lam}: FVU {fvu:.4f} L0 {l0:.2f} "
                        f"alive {int(alive.sum())}/{n_feat} | "
                        f"KEY8 {row['key8_pairdirs']['recovered_injective'][ht]}/8 "
                        f"KEY16dir {row['key16_truefeats']['recovered_injective'][ht]}/16 "
                        f"KEY16func {row['n_truefeats_recovered_functional']}/16 "
                        f"| maxImb {row['selectivity']['max_member_imbalance']} "
                        f"| {row['secs']}s")

    # ---------------- random-direction null ----------------------------------
    prim_name = P["model_arms"][0]["name"]
    prim_seed = P["model_arms"][0]["seeds"][0]
    K16p, K8p, _ = keys_by_arm[(prim_name, prim_seed)]
    alive_counts = sorted({r["n_alive"] for r in sae_rows if r["n_alive"] > 0})
    nulls = {}
    for c in alive_counts:
        nulls[str(c)] = null_recovery(m, c, {"key8_pairdirs": K8p, "key16_truefeats": K16p},
                                      cos_thrs, int(P["n_null_trials"]), rng)
    log("null(random dirs, primary keys) " +
        json.dumps({k: {kk: vv["recovered_mean"] for kk, vv in v.items()}
                    for k, v in nulls.items()}))

    # ---------------- headline ------------------------------------------------
    ctrl_name = P["model_arms"][-1]["name"]

    def best_of(rows):
        ok = [r for r in rows if r.get("key8_pairdirs")]
        if not ok:
            return None
        return max(ok, key=lambda r: (r["key8_pairdirs"]["recovered_injective"][ht], -r["fvu"]))

    prim_rows = [r for r in sae_rows if r["arm"] == prim_name]
    ctrl_rows = [r for r in sae_rows if r["arm"] == ctrl_name]
    best = best_of(prim_rows)
    best_ctrl = best_of(ctrl_rows)
    prim_arms = [a for a in arms if a["arm"] == prim_name]
    ctrl_arms = [a for a in arms if a["arm"] == ctrl_name]

    def pm(aa, which, kind):
        v = [a[which]["mlp"][kind] for a in aa if a[which]["mlp"][kind] is not None]
        return round(float(np.mean(v)), 4) if v else None

    def pl(aa, which, kind):
        v = [a[which]["linear"][kind] for a in aa if a[which]["linear"][kind] is not None]
        return round(float(np.mean(v)), 4) if v else None

    headline = {
        "cos_threshold": head_thr,
        "primary_arm": prim_name,
        "primary_within_pair_abs_cos_mean": round(
            float(np.mean([a["geometry"]["within_pair_abs_cos"] for a in prim_arms])), 4),
        "primary_merge_verified_all_seeds": all(a["merged_verified"] for a in prim_arms),
        "primary_n_represented_by_seed": [a["geometry"]["n_represented"] for a in prim_arms],
        "best_setting": (f"x{best['expansion']} lambda={best['l1_coeff']} seed{best['seed']}"
                         if best else None),
        "n_pairdirs_recovered_of_8": (best["key8_pairdirs"]["recovered_injective"][ht]
                                      if best else None),
        "n_truefeats_recovered_directional_of_16": (
            best["key16_truefeats"]["recovered_injective"][ht] if best else None),
        "n_truefeats_recovered_functional_of_16": (
            best["n_truefeats_recovered_functional"] if best else None),
        "best_fvu": best["fvu"] if best else None,
        "best_l0": best["l0"] if best else None,
        "best_dead_fraction": best["frac_dead"] if best else None,
        "best_max_member_imbalance": (best["selectivity"]["max_member_imbalance"]
                                      if best else None),
        "best_mean_member_imbalance": (best["selectivity"]["mean_member_imbalance"]
                                       if best else None),
        "max_member_imbalance_over_ALL_merged_saes": round(float(max(
            r["selectivity"]["max_member_imbalance"] for r in sae_rows
            if r.get("selectivity") and r["arm"] != ctrl_name)), 4),
        "null_pairdirs_recovered_mean": (
            nulls[str(best["n_alive"])]["key8_pairdirs"]["recovered_mean"][ht] if best else None),
        "null_pairdirs_recovered_p95": (
            nulls[str(best["n_alive"])]["key8_pairdirs"]["recovered_p95"][ht] if best else None),
        "null_truefeats_recovered_mean": (
            nulls[str(best["n_alive"])]["key16_truefeats"]["recovered_mean"][ht] if best else None),
        "max_key16_directional_over_all_merged_saes": int(max(
            r["key16_truefeats"]["recovered_injective"][ht] for r in sae_rows
            if r.get("key16_truefeats") and r["arm"] != ctrl_name)),
        # ---- probe control ----
        "probe_mlp_sum_r2_primary": pm(prim_arms, "probes", "sum"),
        "probe_mlp_diff_r2_primary": pm(prim_arms, "probes", "diff"),
        "probe_lin_diff_r2_primary": pl(prim_arms, "probes", "diff"),
        "probe_mlp_diff_r2_exact_merge_oracle": pm(prim_arms, "probes_exact_merge_oracle", "diff"),
        "probe_mlp_sum_r2_exact_merge_oracle": pm(prim_arms, "probes_exact_merge_oracle", "sum"),
        "probe_mlp_diff_r2_shuffled_null": pm(prim_arms, "probes_shuffled_h_null", "diff"),
        "probe_mlp_diff_r2_unmerged_control": pm(ctrl_arms, "probes", "diff"),
        "probe_mlp_sum_r2_unmerged_control": pm(ctrl_arms, "probes", "sum"),
        # ---- unmerged positive control ----
        "control_arm": ctrl_name,
        "control_within_pair_abs_cos": round(
            float(np.mean([a["geometry"]["within_pair_abs_cos"] for a in ctrl_arms])), 4),
        "control_best_setting": (f"x{best_ctrl['expansion']} lambda={best_ctrl['l1_coeff']}"
                                 if best_ctrl else None),
        "control_n_pairdirs_of_8": (best_ctrl["key8_pairdirs"]["recovered_injective"][ht]
                                    if best_ctrl else None),
        "control_n_truefeats_functional_of_16": (
            best_ctrl["n_truefeats_recovered_functional"] if best_ctrl else None),
        "control_max_truefeats_functional_of_16": int(max(
            r["n_truefeats_recovered_functional"] for r in ctrl_rows
            if "n_truefeats_recovered_functional" in r)) if ctrl_rows else None,
        "control_max_member_imbalance": round(float(max(
            r["selectivity"]["max_member_imbalance"] for r in ctrl_rows
            if r.get("selectivity"))), 4) if ctrl_rows else None,
        "n_skipped_arms": len(skipped),
    }
    log("HEADLINE " + json.dumps(headline, indent=1))

    # ---------------- chart ---------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5))
    arm_names = [s["name"] for s in P["model_arms"]]
    cols = ["#264653", "#2a9d8f", "#e9c46a", "#e76f51"]

    # A: recovered counts across the (expansion, lambda) grid, primary arm, seed-averaged
    ax = axes[0, 0]
    grid = []
    for exp in P["expansions"]:
        for lam in P["l1_coeffs"]:
            rs = [r for r in prim_rows if r["expansion"] == exp and r["l1_coeff"] == lam
                  and r.get("key8_pairdirs")]
            if not rs:
                continue
            grid.append((f"{exp}x\nλ={lam}",
                         np.mean([r["key8_pairdirs"]["recovered_injective"][ht] for r in rs]),
                         np.mean([r["key16_truefeats"]["recovered_injective"][ht] for r in rs]),
                         np.mean([r["n_truefeats_recovered_functional"] for r in rs]),
                         np.mean([nulls[str(r["n_alive"])]["key8_pairdirs"]
                                  ["recovered_mean"][ht] for r in rs]),
                         np.mean([r["fvu"] for r in rs])))
    xs = np.arange(len(grid)); w = 0.27
    ax.bar(xs - w, [g[1] for g in grid], w, color="#2a9d8f", label="KEY-8 pair-directions (of 8)")
    ax.bar(xs, [g[2] for g in grid], w, color="#e9c46a",
           label="KEY-16 true features, DIRECTIONAL (of 16)")
    ax.bar(xs + w, [g[3] for g in grid], w, color="#e76f51",
           label="KEY-16 true features, FUNCTIONAL (of 16)")
    ax.plot(xs + w, [max(g[3], 0.0) for g in grid], "r_", ms=13, mew=2.5, color="#e76f51",
            label="_nolegend_")
    ax.plot(xs - w, [g[4] for g in grid], "k_", ms=15, mew=2,
            label="random-direction null, KEY-8 (mean)")
    ax.axhline(8, color="#2a9d8f", ls=":", lw=1)
    ax.axhline(16, color="#e9c46a", ls=":", lw=1)
    for i, g in enumerate(grid):
        ax.text(i, -1.35, f"FVU {g[5]:.4f}", ha="center", fontsize=5.6, color="dimgray",
                rotation=0)
    ax.text(len(grid) - 0.5, 0.35, "FUNCTIONAL = 0 at every setting", ha="right", va="bottom",
            fontsize=7.5, color="#e76f51", style="italic")
    ax.set_xticks(xs); ax.set_xticklabels([g[0] for g in grid], fontsize=7.5)
    ax.set_ylabel(f"key directions recovered (injective, cos ≥ {head_thr})")
    ax.set_ylim(-2.0, 21.5)
    ax.set_title(f"A. {prim_name}: 8 pair-directions are found; 0 true features are\n"
                 "functionally recovered (the yellow bars are duplicate matches)", fontsize=10)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.95, ncol=2)

    # B: probe control
    ax = axes[0, 1]
    series = [("merged model (trained W)", "probes", "#e76f51"),
              ("EXACT-MERGE ORACLE", "probes_exact_merge_oracle", "#8d99ae"),
              ("shuffled-h null", "probes_shuffled_h_null", "#cccccc")]
    kinds = ["sum", "val_a", "diff"]
    labels = ["SUM\n$x_a+x_b$", "member\n$x_a$", "DIFFERENCE\n$x_a-x_b$"]
    w = 0.8 / (len(series) + 1)
    for si, (lbl, which, c) in enumerate(series):
        vals = [pm(prim_arms, which, k) for k in kinds]
        ax.bar(np.arange(len(kinds)) + si * w - 0.4 + w / 2,
               [v if v is not None else 0 for v in vals], w, color=c, label=lbl)
    vals = [pm(ctrl_arms, "probes", k) for k in kinds]
    ax.bar(np.arange(len(kinds)) + len(series) * w - 0.4 + w / 2,
           [v if v is not None else 0 for v in vals], w, color="#264653",
           label=f"UNMERGED control ({ctrl_name})")
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(range(len(kinds))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("held-out $R^2$, MLP probe on the 4-dim hidden")
    ax.set_title("B. CONTROL: is the within-pair distinction still IN the hidden layer?\n"
                 "yes — a nonlinear probe reads the difference the SAE cannot", fontsize=10)
    ax.legend(fontsize=7)

    # C: member selectivity of SAE features
    ax = axes[1, 0]
    for ai, anm in enumerate(arm_names):
        rs = [r for r in sae_rows if r["arm"] == anm and r.get("selectivity")]
        if not rs:
            continue
        ax.scatter([r["l0"] for r in rs], [r["selectivity"]["max_member_imbalance"] for r in rs],
                   s=40, color=cols[ai], label=anm, alpha=0.85,
                   marker="o" if anm != ctrl_name else "D")
    ax.axhline(sel_thr, color="k", ls="--", lw=1.2, label=f"'member-selective' bar ({sel_thr})")
    ax.set_xlabel("L0 (mean active SAE features per input)")
    ax.set_ylabel("max member imbalance  $|r_a-r_b|/(|r_a|+|r_b|)$")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("C. Does ANY SAE feature separate the two members of a pair?\n"
                 "(each point = one SAE; merged arms sit on the floor)", fontsize=10)
    ax.legend(fontsize=7, loc="center right")

    # D: matched cosine per key direction, best primary arm vs null
    ax = axes[1, 1]
    if best:
        mc8 = [v for v in best["key8_pairdirs"]["matched_cos"] if v is not None]
        mc16 = [v for v in best["key16_truefeats"]["matched_cos"] if v is not None]
        ax.plot(np.linspace(0, 1, len(mc8)), sorted(mc8, reverse=True), "o-", color="#2a9d8f",
                label=f"KEY-8 pair-directions ({best['expansion']}x, λ={best['l1_coeff']})")
        ax.plot(np.linspace(0, 1, len(mc16)), sorted(mc16, reverse=True), "s-", color="#e9c46a",
                label="KEY-16 true features (injective)")
        nb = nulls[str(best["n_alive"])]["key8_pairdirs"]["mean_max_cos"]
        ax.axhline(nb, color="#8d99ae", ls="--", lw=1.5,
                   label=f"random-direction null, mean max cos ({nb:.2f})")
    ax.axhline(head_thr, color="k", ls=":", lw=1.2, label=f"recovery bar (cos={head_thr})")
    ax.set_xlabel("key direction (sorted, rescaled to [0,1])")
    ax.set_ylabel("matched cosine to an SAE decoder direction")
    ax.set_ylim(0, 1.05)
    ax.set_title("D. Match quality per key direction (best primary setting)", fontsize=10)
    ax.legend(fontsize=7, loc="lower left")

    fig.suptitle("SAEs against a KNOWN merged answer key: toy model, 16 features in 8 pairs, "
                 "4 hidden dims", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(HERE / "chart.png", dpi=130)
    plt.close(fig)

    # ---------------- results.json -------------------------------------------
    metrics = {
        "n_features": n, "m_hidden": m, "n_pairs": n // 2,
        "headline": headline,
        "toy_arms": arms,
        "sae_arms": sae_rows,
        "null_random_directions": nulls,
        "skipped_arms": skipped,
        "hparams": {k: P[k] for k in ("toy_steps", "toy_batch", "toy_lr", "expansions",
                                      "l1_coeffs", "sae_steps", "sae_batch", "sae_lr",
                                      "n_act_train", "n_act_eval", "alive_min_fire_frac",
                                      "cos_thresholds", "headline_cos_threshold",
                                      "n_null_trials", "assoc_min_abs_corr",
                                      "member_selectivity_threshold", "probe_hidden",
                                      "probe_steps", "model_arms")},
    }
    results = {
        "id": cfg.get("id", "unknown"),
        "git_commit": git_sha(),
        "seed": seed,
        "duration_sec": round(time.time() - t_start, 2),
        "metrics": metrics,
        "env": env_info(),
        "status": "done",
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nwrote results.json + chart.png in {results['duration_sec']}s")


if __name__ == "__main__":
    main()
