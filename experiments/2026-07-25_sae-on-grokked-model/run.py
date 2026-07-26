"""SAEs on a grokked modular-addition transformer: do they recover the KNOWN Fourier features?

Ground truth: a 1-layer LayerNorm-free transformer that has fully grokked (a+b) mod 59
provably implements the Fourier-multiplication algorithm -- its MLP neurons are (close to)
single-frequency cos/sin features. That gives us a *provable* answer key against which SAE
faithfulness can be scored, unlike the usual language-model setting.

We train small overcomplete L1 SAEs on the post-ReLU MLP hidden activations for all p*p
inputs and score every feature by the concentration of its 2D Fourier power on a single
frequency, comparing against (a) the raw MLP neuron basis, (b) SAEs on a memorized-only
checkpoint, (c) SAEs on a random-init transformer, and (d) random-direction nulls.

Deterministic, CPU-only, single-threaded.  Usage:  python run.py
"""
import json, os, random, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
_LOG = None          # opened lazily so that merely importing this module never
                     # truncates the log of a completed run


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


class GrokTransformer(nn.Module):
    """Byte-for-byte the architecture of 2026-07-25_grokking-modular-addition.

    1-layer attention + MLP, NO LayerNorm, no biases. Input is [a, b, '='];
    logits over the p residues are read from the last position only.
    """

    def __init__(self, p, d_model, n_heads, d_mlp, n_ctx, init_std_scale):
        super().__init__()
        self.p, self.d_model, self.n_heads = p, d_model, n_heads
        self.d_head = d_model // n_heads
        std = init_std_scale / (d_model ** 0.5)
        self.W_E = nn.Parameter(torch.randn(p + 1, d_model) * std)
        self.W_pos = nn.Parameter(torch.randn(n_ctx, d_model) * std)
        self.W_Q = nn.Parameter(torch.randn(d_model, d_model) * std)
        self.W_K = nn.Parameter(torch.randn(d_model, d_model) * std)
        self.W_V = nn.Parameter(torch.randn(d_model, d_model) * std)
        self.W_O = nn.Parameter(torch.randn(d_model, d_model) * std)
        self.W_in = nn.Parameter(torch.randn(d_model, d_mlp) * std)
        self.W_out = nn.Parameter(torch.randn(d_mlp, d_model) * std)
        self.W_U = nn.Parameter(torch.randn(d_model, p) * std)

    @torch.no_grad()
    def acts(self, idx):
        """-> (resid_post_attn (N,d_model), mlp_hidden_post_relu (N,d_mlp), logits (N,p))"""
        N, T = idx.shape
        H, Dh = self.n_heads, self.d_head
        x = self.W_E[idx] + self.W_pos[None, :T, :]
        last = x[:, -1:, :]
        q = (last @ self.W_Q).view(N, 1, H, Dh).transpose(1, 2)
        k = (x @ self.W_K).view(N, T, H, Dh).transpose(1, 2)
        v = (x @ self.W_V).view(N, T, H, Dh).transpose(1, 2)
        att = (q @ k.transpose(-2, -1) / (Dh ** 0.5)).softmax(-1)
        z = (att @ v).transpose(1, 2).reshape(N, 1, self.d_model) @ self.W_O
        h = (last + z).view(N, self.d_model)
        mlp = F.relu(h @ self.W_in)
        logits = (h + mlp @ self.W_out) @ self.W_U
        return h, mlp, logits


# --------------------------- Fourier ground truth ---------------------------
class FourierScorer:
    """2D real-Fourier power decomposition of maps over the (a, b) grid of Z_p x Z_p.

    Basis rows: [1/sqrt(p)] + for k=1..(p-1)/2: sqrt(2/p) cos(2pi k x/p), sqrt(2/p) sin(...).
    Orthonormal, so Parseval holds exactly.

    A "frequency-k" component of a map f(a,b) is any basis product (i,j) whose row
    frequencies both lie in {0, k} -- i.e. the 3x3 block spanned by
    {1, cos w_k a, sin w_k a} x {1, cos w_k b, sin w_k b}, minus the DC term.
    These blocks are disjoint for different k once DC is removed, so the shares are
    a genuine partition of a subset of the total power.  This is exactly the structure
    a grokked mod-add model is proven to use (Nanda et al. 2301.05217).

    purity(f) = max_k power_k(f) / (total power - DC power)
    """

    def __init__(self, p):
        self.p = p
        self.m = (p - 1) // 2
        xs = np.arange(p)
        B = np.zeros((p, p), dtype=np.float32)
        fr = [0]
        B[0] = 1.0 / np.sqrt(p)
        for k in range(1, self.m + 1):
            B[2 * k - 1] = np.sqrt(2.0 / p) * np.cos(2 * np.pi * k * xs / p)
            B[2 * k] = np.sqrt(2.0 / p) * np.sin(2 * np.pi * k * xs / p)
            fr += [k, k]
        self.B = B
        self.freq_of_row = np.array(fr)
        # group id per (i, j) basis-product cell
        G = np.full((p, p), -1, dtype=np.int64)
        for k in range(1, self.m + 1):
            ri = np.where((self.freq_of_row == 0) | (self.freq_of_row == k))[0]
            G[np.ix_(ri, ri)] = k
        G[0, 0] = -1                       # DC belongs to no frequency
        self.G = G
        gflat = G.reshape(-1)
        M = np.zeros((self.m + 1, p * p), dtype=np.float32)
        for k in range(1, self.m + 1):
            M[k] = (gflat == k)
        self.M = M                          # (m+1, p*p) group indicator
        # sum-frequency (a+b) templates, per k, orthonormalised
        A, Bb = np.meshgrid(xs, xs, indexing="ij")
        S = np.zeros((p * p, 2 * self.m), dtype=np.float32)
        for k in range(1, self.m + 1):
            c = np.cos(2 * np.pi * k * (A + Bb) / p).reshape(-1)
            s = np.sin(2 * np.pi * k * (A + Bb) / p).reshape(-1)
            S[:, 2 * (k - 1)] = c / np.linalg.norm(c)
            S[:, 2 * (k - 1) + 1] = s / np.linalg.norm(s)
        self.S = S

    def score(self, acts):
        """acts: (p*p, n_feat) float32 -> dict of per-feature arrays."""
        p = self.p
        A = np.ascontiguousarray(acts, dtype=np.float32).reshape(p, p, -1)
        Fm = np.einsum("ia,abf->ibf", self.B, A, optimize=True)
        Fm = np.einsum("jb,ibf->ijf", self.B, Fm, optimize=True)
        P2 = (Fm ** 2).reshape(p * p, -1)
        tot = P2.sum(0) - P2[0]                       # non-DC power
        Pk = self.M @ P2                              # (m+1, n_feat)
        denom = np.maximum(tot, 1e-20)
        pur = Pk[1:].max(0) / denom
        best = Pk[1:].argmax(0) + 1
        # share of the map that is a pure cos/sin of w_k (a+b), at the best k
        proj = (acts.T.astype(np.float32) @ self.S)   # (n_feat, 2m)
        sumpow = proj[:, 0::2] ** 2 + proj[:, 1::2] ** 2   # (n_feat, m)
        sum_at_best = sumpow[np.arange(sumpow.shape[0]), best - 1]
        pur = np.where(tot > 1e-12, pur, 0.0)
        return {"purity": pur.astype(np.float64),
                "best_freq": best.astype(int),
                "power_by_freq": Pk[1:].astype(np.float64),
                "total_power": tot.astype(np.float64),
                "sumfreq_share": (sum_at_best / denom).astype(np.float64)}


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


def train_sae(x_raw, n_feat, lam, P, seed):
    """x_raw: (N, d) tensor of raw activations. Returns (sae, xn, scale, info)."""
    gen = torch.Generator().manual_seed(seed)
    N, d = x_raw.shape
    scale = float(np.sqrt(d) / x_raw.norm(dim=1).mean().item())   # E||x|| = sqrt(d)
    xn = x_raw * scale
    sae = SAE(d, n_feat, xn.mean(0), gen)
    opt = torch.optim.Adam(sae.parameters(), lr=P["sae_lr"])
    steps, bs = int(P["sae_steps"]), int(P["sae_batch"])
    resample_steps = {int(r * steps) for r in P["resample_at"]}
    fired = torch.zeros(n_feat, dtype=torch.bool)
    n_resampled = 0
    for it in range(steps):
        xb = xn[torch.randint(0, N, (bs,), generator=gen)]
        f, r = sae(xb)
        recon = ((r - xb) ** 2).sum(-1).mean()
        l1 = f.abs().sum(-1).mean()
        loss = recon + lam * l1
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
                    prob = (err ** 2)
                    prob = prob / prob.sum().clamp_min(1e-12)
                    pick = torch.multinomial(prob, nd, replacement=True, generator=gen)
                    dirs = (xs_[pick] - sae.b_dec)
                    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-8)
                    alive_enc_norm = (sae.W_enc.data[:, fired].norm(dim=0).mean()
                                      if int(fired.sum()) > 0 else torch.tensor(1.0))
                    sae.W_dec.data[dead] = dirs
                    sae.W_enc.data[:, dead] = (dirs * 0.2 * alive_enc_norm).t()
                    sae.b_enc.data[dead] = 0.0
                    n_resampled += nd
                    # reset Adam state for the resampled features
                    for pgroup, prm in (("W_enc", sae.W_enc), ("W_dec", sae.W_dec),
                                        ("b_enc", sae.b_enc)):
                        st = opt.state.get(prm, None)
                        if st:
                            if pgroup == "W_enc":
                                st["exp_avg"][:, dead] = 0; st["exp_avg_sq"][:, dead] = 0
                            else:
                                st["exp_avg"][dead] = 0; st["exp_avg_sq"][dead] = 0
                fired[:] = False
    return sae, xn, scale, {"n_resampled": n_resampled}


@torch.no_grad()
def sae_eval(sae, xn):
    f, r = sae(xn)
    fvu = float(((r - xn) ** 2).sum() / ((xn - xn.mean(0)) ** 2).sum())
    l0 = float((f > 0).float().sum(-1).mean())
    fire_frac = (f > 0).float().mean(0).numpy()
    return f.numpy(), fvu, l0, fire_frac


# ------------------------------- analysis -----------------------------------
def summarise(pur, best, alive, thr, thr_loose, key_freqs):
    out = {"n_alive": int(alive.sum())}
    if alive.sum() == 0:
        out.update({"mean_purity": None, "median_purity": None, "frac_pure": None,
                    "frac_pure_loose": None, "n_pure": 0, "freqs_claimed": [],
                    "freqs_claimed_loose": [], "key_freqs_recovered": False,
                    "n_key_freqs_recovered": 0})
        return out
    pa, ba = pur[alive], best[alive]
    pure = pa >= thr
    loose = pa >= thr_loose
    claimed = sorted({int(f) for f in ba[pure]})
    claimed_loose = sorted({int(f) for f in ba[loose]})
    out.update({
        "mean_purity": round(float(pa.mean()), 4),
        "median_purity": round(float(np.median(pa)), 4),
        "p90_purity": round(float(np.percentile(pa, 90)), 4),
        "max_purity": round(float(pa.max()), 4),
        "frac_pure": round(float(pure.mean()), 4),
        "frac_pure_loose": round(float(loose.mean()), 4),
        "n_pure": int(pure.sum()),
        "freqs_claimed": claimed,
        "freqs_claimed_loose": claimed_loose,
        "n_key_freqs_recovered": len(set(claimed) & set(key_freqs)),
        "key_freqs_recovered": set(key_freqs).issubset(set(claimed)),
        "freqs_claimed_exact_match": sorted(claimed) == sorted(key_freqs),
    })
    return out


def main():
    cfg = load_config()
    P = cfg["params"]
    seed = int(cfg.get("seed", 0))
    set_seeds(seed)
    t_start = time.time()
    budget = float(P["time_budget_s"])

    # ---------------- load the three activation sources ----------------------
    def build_from_ckpt(path):
        ck = torch.load(HERE / path, map_location="cpu", weights_only=False)
        A = ck["arch"]
        m = GrokTransformer(ck["p"], A["d_model"], A["n_heads"], A["d_mlp"],
                            A["n_ctx"], A["init_std_scale"])
        m.load_state_dict(ck["state_dict"])
        m.eval()
        return m, ck

    gm, gck = build_from_ckpt(P["grokked_ckpt"])
    mm, mck = build_from_ckpt(P["memorized_ckpt"])
    p = gck["p"]
    A = gck["arch"]
    torch.manual_seed(int(P["random_seed_model"]))
    rm = GrokTransformer(p, A["d_model"], A["n_heads"], A["d_mlp"], A["n_ctx"],
                         A["init_std_scale"])
    rm.eval()
    key_freqs = [int(k) for k in gck["key_freqs"]]

    aa, bb = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    a_all, b_all = aa.reshape(-1), bb.reshape(-1)
    y_all = (a_all + b_all) % p
    X = torch.from_numpy(np.stack([a_all, b_all, np.full_like(a_all, p)], 1)).long()

    models = {"grokked": gm, "memorized": mm, "random_init": rm}
    acts, model_acc = {}, {}
    for name, mdl in models.items():
        h, mlp, logits = mdl.acts(X)
        acts[name] = mlp.contiguous()
        model_acc[name] = float((logits.argmax(-1).numpy() == y_all).mean())
    log(f"p={p}  d_mlp={A['d_mlp']}  N={p*p} inputs")
    log(f"all-pair accuracy: grokked {model_acc['grokked']:.4f}  "
        f"memorized {model_acc['memorized']:.4f}  random {model_acc['random_init']:.4f}")
    log(f"grokked ckpt: step {gck['step']}, train {gck['train_acc']:.3f}, "
        f"test {gck['test_acc']:.3f}, key_freqs {key_freqs}")
    log(f"memorized ckpt: step {mck['step']}, train {mck['train_acc']:.3f}, "
        f"test {mck['test_acc']:.3f} (train_frac {mck['train_frac']})")

    scorer = FourierScorer(p)
    thr, thr_l = float(P["purity_threshold"]), float(P["purity_threshold_loose"])
    alive_min = float(P["alive_min_fire_frac"])

    # ---------------- baseline 1: the raw MLP neuron basis --------------------
    neuron = {}
    for name in models:
        a_np = acts[name].numpy()
        sc = scorer.score(a_np)
        fire = (a_np > 0).mean(0)
        alive = fire >= alive_min
        s = summarise(sc["purity"], sc["best_freq"], alive, thr, thr_l, key_freqs)
        s["n_units"] = int(a_np.shape[1])
        s["frac_dead"] = round(float((fire == 0).mean()), 4)
        s["mean_fire_frac_alive"] = round(float(fire[alive].mean()) if alive.sum() else 0.0, 4)
        s["mean_sumfreq_share_alive"] = round(
            float(sc["sumfreq_share"][alive].mean()) if alive.sum() else 0.0, 4)
        # total activation power per frequency, over all neurons
        pw = sc["power_by_freq"].sum(1)
        s["power_share_by_freq_top6"] = {int(k + 1): round(float(pw[k] / pw.sum()), 4)
                                         for k in np.argsort(-pw)[:6]}
        neuron[name] = s
        log(f"[neurons/{name}] alive {s['n_alive']}/{s['n_units']}  "
            f"mean purity {s['mean_purity']}  frac_pure {s['frac_pure']}  "
            f"freqs {s['freqs_claimed']}")

    # ---------------- baseline 2: random-direction null -----------------------
    rng = np.random.default_rng(seed)
    nulls = {}
    for name in models:
        a_np = acts[name].numpy()
        D = rng.standard_normal((a_np.shape[1], int(P["n_null_dirs"]))).astype(np.float32)
        D /= np.linalg.norm(D, axis=0, keepdims=True)
        sc = scorer.score(a_np @ D)
        nulls[name] = {
            "mean_purity": round(float(sc["purity"].mean()), 4),
            "p99_purity": round(float(np.percentile(sc["purity"], 99)), 4),
            "frac_pure": round(float((sc["purity"] >= thr).mean()), 4),
        }
    # null for *sparse localised indicator* maps, matched to a typical SAE feature
    sparse_null = {}
    for ff in (0.02, 0.05, 0.15):
        n_on = max(1, int(round(ff * p * p)))
        Msp = np.zeros((p * p, 256), dtype=np.float32)
        for j in range(256):
            Msp[rng.choice(p * p, n_on, replace=False), j] = 1.0
        scn = scorer.score(Msp)
        sparse_null[str(ff)] = {"mean_purity": round(float(scn["purity"].mean()), 4),
                                "p99_purity": round(float(np.percentile(scn["purity"], 99)), 4)}
    log(f"null (random directions): {nulls}")
    log(f"null (random sparse indicator maps): {sparse_null}")

    # ---------------- SAE arms ------------------------------------------------
    d_in = int(A["d_mlp"])
    lams = [float(l) for l in P["l1_coeffs"]]
    arms = [(name, lam, int(P["expansion"])) for name in models for lam in lams]
    mid_lam = lams[len(lams) // 2]
    arms.append(("grokked", mid_lam, int(P["expansion_extra"])))

    sae_rows, skipped = [], []
    for (name, lam, exp) in arms:
        if time.time() - t_start > budget:
            skipped.append({"model": name, "l1": lam, "expansion": exp})
            log(f"SKIP {name} lam={lam} x{exp} (budget {budget}s exhausted)")
            continue
        t0 = time.time()
        n_feat = d_in * exp
        sae, xn, scale, info = train_sae(acts[name], n_feat, lam, P, seed)
        f, fvu, l0, fire = sae_eval(sae, xn)
        sc = scorer.score(f)
        alive = fire >= alive_min
        row = summarise(sc["purity"], sc["best_freq"], alive, thr, thr_l, key_freqs)
        row.update({
            "model": name, "l1_coeff": lam, "expansion": exp, "n_feat": n_feat,
            "fvu": round(fvu, 5), "l0": round(l0, 3),
            "frac_dead": round(float((fire == 0).mean()), 4),
            "frac_rare": round(float(((fire > 0) & (fire < alive_min)).mean()), 4),
            "frac_alive": round(float(alive.mean()), 4),
            "mean_fire_frac_alive": round(float(fire[alive].mean()) if alive.sum() else 0.0, 4),
            "mean_sumfreq_share_alive": round(
                float(sc["sumfreq_share"][alive].mean()) if alive.sum() else 0.0, 4),
            "n_resampled": info["n_resampled"],
            "secs": round(time.time() - t0, 1),
        })
        # decoder-direction purity: Fourier purity of the map (a,b) -> w_f . x(a,b)
        Wd = sae.W_dec.detach().numpy().T.astype(np.float32)      # (d_in, n_feat)
        proj = acts[name].numpy() @ Wd
        scd = scorer.score(proj)
        dec = summarise(scd["purity"], scd["best_freq"], alive, thr, thr_l, key_freqs)
        row["dec_mean_purity"] = dec["mean_purity"]
        row["dec_frac_pure"] = dec["frac_pure"]
        row["dec_freqs_claimed"] = dec["freqs_claimed"]
        row["_purity"] = sc["purity"][alive].tolist()
        row["_best"] = sc["best_freq"][alive].tolist()
        row["_dec_purity"] = scd["purity"][alive].tolist()
        sae_rows.append(row)
        log(f"[sae/{name}] lam={lam} x{exp}: FVU {row['fvu']:.4f} L0 {row['l0']:.1f} "
            f"dead {row['frac_dead']:.2f} alive {row['n_alive']}/{n_feat} "
            f"| act purity mean {row['mean_purity']} frac_pure {row['frac_pure']} "
            f"| dec purity mean {row['dec_mean_purity']} frac_pure {row['dec_frac_pure']} "
            f"| {row['secs']}s")

    # ---------------- headline ------------------------------------------------
    # An arm that has collapsed (reconstructs nothing / keeps a handful of features)
    # can score a trivially perfect purity, so the headline is taken over USABLE arms
    # only: FVU <= 0.25 and at least 10 alive features. Collapsed arms are still
    # reported, flagged, and discussed.
    FVU_MAX, ALIVE_MIN_N = 0.25, 10
    for r in sae_rows:
        r["usable"] = bool(r["fvu"] <= FVU_MAX and r["n_alive"] >= ALIVE_MIN_N)
    grok_rows = [r for r in sae_rows if r["model"] == "grokked"
                 and r["expansion"] == int(P["expansion"])]
    grok_usable = [r for r in sae_rows if r["model"] == "grokked" and r["usable"]]
    grok_usable_4x = [r for r in grok_usable if r["expansion"] == int(P["expansion"])]

    def pick(rows):
        return max(rows, key=lambda r: (r["frac_pure"] or 0.0)) if rows else None

    best_row, best4 = pick(grok_usable), pick(grok_usable_4x)
    collapsed = [{"model": r["model"], "l1_coeff": r["l1_coeff"], "expansion": r["expansion"],
                  "fvu": r["fvu"], "n_alive": r["n_alive"], "frac_pure": r["frac_pure"]}
                 for r in sae_rows if not r["usable"]]
    headline = {
        "usable_arm_criteria": f"fvu<={FVU_MAX} and n_alive>={ALIVE_MIN_N}",
        "neuron_frac_pure_grokked": neuron["grokked"]["frac_pure"],
        "neuron_mean_purity_grokked": neuron["grokked"]["mean_purity"],
        "neuron_freqs_claimed_grokked": neuron["grokked"]["freqs_claimed"],
        "sae_best_frac_pure_grokked": best_row["frac_pure"] if best_row else None,
        "sae_best_arm": (f"lambda={best_row['l1_coeff']} x{best_row['expansion']}"
                         if best_row else None),
        "sae_best_mean_purity_grokked": best_row["mean_purity"] if best_row else None,
        "sae_best_dec_frac_pure_grokked": best_row["dec_frac_pure"] if best_row else None,
        "sae_best_freqs_claimed": best_row["freqs_claimed"] if best_row else None,
        "sae_best_4x_frac_pure": best4["frac_pure"] if best4 else None,
        "sae_best_4x_arm": (f"lambda={best4['l1_coeff']} x{best4['expansion']}"
                            if best4 else None),
        "sae_best_dead_fraction": best_row["frac_dead"] if best_row else None,
        "key_freqs": key_freqs,
        "key_freqs_recovered_by_sae": bool(best_row["key_freqs_recovered"]) if best_row else None,
        "key_freqs_recovered_by_neurons": bool(neuron["grokked"]["key_freqs_recovered"]),
        "null_frac_pure_random_dirs_grokked": nulls["grokked"]["frac_pure"],
        "null_mean_purity_random_dirs_grokked": nulls["grokked"]["mean_purity"],
        "n_collapsed_arms": len(collapsed),
    }
    log("HEADLINE " + json.dumps(headline))

    # ---------------- chart ---------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    ax = axes[0, 0]
    a_np = acts["grokked"].numpy()
    nsc = scorer.score(a_np)
    nal = (a_np > 0).mean(0) >= alive_min
    bins = np.linspace(0, 1, 26)
    ax.hist(nsc["purity"][nal], bins=bins, alpha=0.65, density=True,
            label=f"MLP neurons, grokked (n={int(nal.sum())})", color="#2a9d8f")
    mid = next((r for r in grok_rows if r["l1_coeff"] == mid_lam), None)
    if mid:
        ax.hist(np.array(mid["_purity"]), bins=bins, alpha=0.6, density=True,
                label=f"SAE feats, grokked $\\lambda$={mid_lam} (n={mid['n_alive']})",
                color="#e76f51")
        ax.hist(np.array(mid["_dec_purity"]), bins=bins, alpha=0.45, density=True,
                label="SAE decoder directions, grokked", color="#264653", histtype="step", lw=2)
    rnd = next((r for r in sae_rows if r["model"] == "random_init"
                and r["l1_coeff"] == mid_lam), None)
    if rnd and rnd["n_alive"]:
        ax.hist(np.array(rnd["_purity"]), bins=bins, alpha=0.5, density=True,
                label=f"SAE feats, RANDOM-init model (n={rnd['n_alive']})",
                color="#8d99ae", histtype="step", lw=2, ls="--")
    ax.axvline(thr, color="k", ls=":", lw=1.5)
    ax.text(thr + 0.01, ax.get_ylim()[1] * 0.92, "pure\nthreshold", fontsize=8)
    ax.set_xlabel("single-frequency Fourier purity")
    ax.set_ylabel("density")
    ax.set_title("A. The neuron basis is frequency-pure; SAE features are not", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper center")

    ax = axes[0, 1]
    cols = {"grokked": "#e76f51", "memorized": "#457b9d", "random_init": "#8d99ae"}
    for name in models:
        rs = sorted([r for r in sae_rows if r["model"] == name
                     and r["expansion"] == int(P["expansion"])], key=lambda r: r["l0"])
        if not rs:
            continue
        ax.plot([r["l0"] for r in rs], [r["fvu"] for r in rs], "o-", color=cols[name],
                label=name)
        for r in rs:
            ax.annotate(f"$\\lambda$={r['l1_coeff']}", (r["l0"], r["fvu"]), fontsize=7,
                        xytext=(3, 4), textcoords="offset points")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("L0 (mean active features / input)")
    ax.set_ylabel("FVU (fraction of variance unexplained)")
    ax.set_title("B. Reconstruction-vs-sparsity tradeoff", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    ax = axes[1, 0]
    w = 0.35
    xs_ = np.arange(len(sae_rows))
    ax.bar(xs_ - w / 2, [r["frac_pure"] or 0 for r in sae_rows], w,
           label="activation-map purity", color="#e76f51",
           hatch=["" if r["usable"] else "//" for r in sae_rows])
    ax.bar(xs_ + w / 2, [r["dec_frac_pure"] or 0 for r in sae_rows], w,
           label="decoder-direction purity", color="#264653",
           hatch=["" if r["usable"] else "//" for r in sae_rows])
    ax.axhline(neuron["grokked"]["frac_pure"], color="#2a9d8f", ls="--", lw=2,
               label=f"MLP neurons, grokked ({neuron['grokked']['frac_pure']:.2f})")
    ax.axhline(nulls["grokked"]["frac_pure"], color="#8d99ae", ls=":", lw=2,
               label=f"random-direction null ({nulls['grokked']['frac_pure']:.3f})")
    for i, r in enumerate(sae_rows):
        top = max(r["frac_pure"] or 0, r["dec_frac_pure"] or 0)
        ax.text(i, top + 0.02, f"FVU {r['fvu']:.2f}", ha="center", va="bottom", fontsize=6,
                color="dimgray")
    ax.plot([], [], color="w", label="hatched = COLLAPSED arm (FVU>0.25 or <10 alive)")
    ax.set_xticks(xs_)
    ax.set_xticklabels([f"{r['model'][:4]}\n$\\lambda$={r['l1_coeff']}\nx{r['expansion']}"
                        for r in sae_rows], fontsize=7)
    ax.set_ylabel(f"fraction of ALIVE units with purity $\\geq$ {thr}")
    ax.set_ylim(0, 1.28)
    ax.set_title("C. Headline: fraction of alive units that are frequency-pure", fontsize=10)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.95)

    ax = axes[1, 1]
    ks = np.arange(1, scorer.m + 1)
    npure = np.array([int(((nsc["purity"][nal] >= thr) &
                           (nsc["best_freq"][nal] == k)).sum()) for k in ks])
    ax.bar(ks - 0.2, npure, 0.4, color="#2a9d8f", label="pure MLP neurons")
    if best_row:
        pb = np.array(best_row["_best"])[np.array(best_row["_purity"]) >= thr]
        sp = np.array([int((pb == k).sum()) for k in ks]) if pb.size else np.zeros_like(ks)
        ax.bar(ks + 0.2, sp, 0.4, color="#e76f51",
               label=(f"pure SAE feats (best usable arm:\n$\\lambda$="
                      f"{best_row['l1_coeff']}, x{best_row['expansion']})"))
    for k in key_freqs:
        ax.axvline(k, color="#e9c46a", lw=6, alpha=0.35, zorder=0)
    ax.plot([], [], color="#e9c46a", lw=6, alpha=0.35,
            label=f"model key freqs {key_freqs}")
    ax.set_xlabel("frequency k")
    ax.set_ylabel("# frequency-pure units")
    ax.set_title("D. Which frequencies get claimed?", fontsize=10)
    ax.legend(fontsize=8)

    fig.suptitle("SAEs vs a provable ground truth: grokked (a+b) mod 59, MLP hidden layer",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(HERE / "chart.png", dpi=130)
    plt.close(fig)

    # ---------------- results.json -------------------------------------------
    public_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in sae_rows]
    metrics = {
        "p": p, "n_inputs": p * p, "site": P["site"], "d_in": d_in,
        "key_freqs": key_freqs,
        "purity_threshold": thr, "purity_threshold_loose": thr_l,
        "alive_min_fire_frac": alive_min,
        "model_all_pair_accuracy": {k: round(v, 4) for k, v in model_acc.items()},
        "grokked_ckpt": {"step": int(gck["step"]), "train_acc": round(float(gck["train_acc"]), 4),
                         "test_acc": round(float(gck["test_acc"]), 4),
                         "train_frac": float(gck["train_frac"])},
        "memorized_ckpt": {"step": int(mck["step"]), "train_acc": round(float(mck["train_acc"]), 4),
                           "test_acc": round(float(mck["test_acc"]), 4),
                           "train_frac": float(mck["train_frac"])},
        "headline": headline,
        "collapsed_arms": collapsed,
        "neuron_baseline": neuron,
        "null_random_directions": nulls,
        "null_random_sparse_maps": sparse_null,
        "sae_arms": public_rows,
        "skipped_arms": skipped,
        "sae_hparams": {k: P[k] for k in ("expansion", "expansion_extra", "l1_coeffs",
                                          "sae_steps", "sae_batch", "sae_lr",
                                          "resample_at", "n_null_dirs")},
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
