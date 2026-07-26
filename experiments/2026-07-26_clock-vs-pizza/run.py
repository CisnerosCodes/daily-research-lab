"""Clock vs Pizza: mapping the algorithmic phase boundary with an attention-rate knob.

Zhong et al. (arXiv:2306.17844, "The Clock and the Pizza") show that a 1-layer transformer
trained on (a+b) mod p can implement one of two DIFFERENT algorithms:

  CLOCK  - attention is used to multiply rotations; the answer logit does not depend on a-b.
  PIZZA  - the model AVERAGES the two operand embeddings and then applies a nonlinearity;
           the averaged vector picks up an amplitude factor cos(pi*w*(a-b)/p), so the answer
           logit DOES depend on a-b.

They report that the choice is controlled by an "attention rate" alpha which interpolates the
attention matrix toward a constant matrix, with clock at high alpha and pizza at low alpha.

This run sweeps that knob and asks where the boundary is, using the paper's own two
discriminators (their Definitions 4.1 and 4.2, implemented verbatim below).

Knob (note the sign convention):  att = (1 - r) * softmax(QK^T / sqrt(d_h))  +  r * (1/T)
  r = 0 -> standard attention         = the paper's alpha = 1 -> predicted CLOCK
  r = 1 -> constant/uniform attention = the paper's alpha = 0 -> predicted PIZZA (their Model A)
We sweep r and report alpha = 1 - r alongside.

SHRUNK to fit a ~12 minute 1-thread CPU box: d_model 64 (the paper fixes 128 in its
attention-rate figure and sweeps 32-512 elsewhere), d_mlp 128, 1200 training steps per arm,
5 r-values x 2 seeds plus up to 4 boundary-refinement arms. p=59 is NOT shrunk - it is the
paper's own modulus.

Everything else follows the lab's proven grokking recipe (rows 2026-07-25_grokking-modular-addition
and 2026-07-25_grokking-weight-decay-phase): full-batch AdamW, lr 1e-3, wd 1.0, betas (0.9, 0.98),
train_frac 0.5, no LayerNorm, no biases.

Fairness: at equal seed every arm gets the IDENTICAL initialisation, the IDENTICAL data split and
the IDENTICAL number of optimiser steps. Only r differs.

Usage:  python run.py
"""
import json, os, random, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent


def set_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
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


# ----------------------------- model ---------------------------------------
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)


class GrokTransformer(nn.Module):
    """1-layer attention + MLP transformer, NO LayerNorm, no biases (Nanda-et-al style).

    Vocab is p+1 (numbers 0..p-1 plus the "=" token). Input is always [a, b, =]; logits over
    the p residues are read from the last position only.

    `attn_uniform_rate` r mixes the attention distribution toward uniform:
        att = (1 - r) * softmax(.) + r * (1/T)
    This keeps attention rows stochastic. Zhong et al. write M' = M*alpha + J*(1-alpha) with J
    the ALL-ONE matrix, which is the same family up to a constant row scale that the learnable
    W_O can absorb; we keep rows normalised so that r is exactly "the fraction of the attention
    average that is forced to be uniform".
    """

    def __init__(self, p, d_model, n_heads, d_mlp, n_ctx, init_std_scale, attn_uniform_rate):
        super().__init__()
        self.p, self.d_model, self.n_heads = p, d_model, n_heads
        self.d_head = d_model // n_heads
        self.n_ctx = n_ctx
        self.r = float(attn_uniform_rate)
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

    def _core(self, x, return_att=False):
        """x: (N, T, d) residual stream AFTER embedding + position -> (N, d) at the last pos."""
        N, T = x.shape[0], x.shape[1]
        H, Dh = self.n_heads, self.d_head
        last = x[:, -1:, :]                                        # (N, 1, d)
        q = (last @ self.W_Q).view(N, 1, H, Dh).transpose(1, 2)
        k = (x @ self.W_K).view(N, T, H, Dh).transpose(1, 2)
        v = (x @ self.W_V).view(N, T, H, Dh).transpose(1, 2)
        att = (q @ k.transpose(-2, -1) / (Dh ** 0.5)).softmax(-1)   # (N, H, 1, T)
        if self.r > 0.0:
            att = (1.0 - self.r) * att + self.r * (1.0 / T)
        z = (att @ v).transpose(1, 2).reshape(N, 1, self.d_model) @ self.W_O
        h = last + z
        h = h + F.relu(h @ self.W_in) @ self.W_out
        h = h.view(N, self.d_model)
        return (h, att) if return_att else h

    def hidden(self, idx):
        x = self.W_E[idx] + self.W_pos[None, : idx.shape[1], :]
        return self._core(x)

    def forward(self, idx):
        return self.hidden(idx) @ self.W_U

    def logits_from_embeddings(self, e_a, e_b):
        """e_a, e_b: (N, d) RAW token embedding vectors (leaf tensors) -> logits (N, p).

        Needed for gradient symmetricity: dQ/dE_a and dQ/dE_b must be taken with respect to the
        two operand embeddings SEPARATELY, so they have to be distinct differentiable inputs.
        """
        e_eq = self.W_E[self.p].expand(e_a.shape[0], -1)
        x = torch.stack([e_a, e_b, e_eq], dim=1) + self.W_pos[None, :, :]
        return self._core(x) @ self.W_U

    def attention_weights(self, idx):
        x = self.W_E[idx] + self.W_pos[None, : idx.shape[1], :]
        return self._core(x, return_att=True)[1]


# --------------------- discriminators (Zhong et al. 2306.17844) ---------------------
def correct_logit_matrix(model, p):
    """L[i, j] = Q_{ij, i+j}: the logit assigned to the CORRECT answer for input (i, j).

    This is exactly the "correct logit matrix" of the paper's Definition 4.2.
    """
    a, b = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    a, b = a.reshape(-1), b.reshape(-1)
    X = torch.from_numpy(np.stack([a, b, np.full_like(a, p)], 1)).long()
    with torch.no_grad():
        lg = model(X).numpy()
    L = lg[np.arange(len(a)), (a + b) % p].reshape(p, p)
    return L, lg.reshape(p, p, p)


def distance_irrelevance(L, p):
    """Zhong et al. Definition 4.2:

        q = [ (1/p) * sum_{d in Z_p} std( L[i, i+d] : i in Z_p ) ]  /  std( L[i,j] : i,j in Z_p^2 )

    Read it as: hold the DISTANCE d = b - a fixed and see how much the correct logit still
    varies as the SUM sweeps. If the correct logit is essentially a function of d (pizza), each
    inner std is small and q -> 0. If the correct logit ignores d (clock), each inner std equals
    the global std and q -> 1. So HIGH q = clock, LOW q = pizza.
    The paper reports q = 0.85 for its clock model and q = 0.17 for its pizza model, and states
    that pizza typically lands in [0, 0.4] and clock in [0.4, 1].
    """
    idx = np.arange(p)
    rows = np.stack([L[idx, (idx + d) % p] for d in range(p)])      # (p_d, p_i)
    per_d_std = rows.std(axis=1)
    return float(per_d_std.mean() / (L.std() + 1e-30)), per_d_std


def gradient_symmetricity(model, p, n_samples, seed):
    """Zhong et al. Definition 4.1:

        s_g = (1/|S|) * sum_{(a,b,c) in S} cos( dQ_abc/dE_a , dQ_abc/dE_b )

    S is a random set of (input a, input b, output logit index c) triples; c is sampled
    uniformly and is NOT required to be the correct answer, exactly as in the paper.

    Why this discriminates: with fully uniform attention the last-position residual stream
    depends on the operands only through (E_a + E_b) W_V / T, so the network is an exactly
    SYMMETRIC function of the two embedding vectors and the two gradients coincide -> s_g = 1.
    Real attention makes the mixing weights depend on the query, so the two operands enter
    asymmetrically -> s_g well below 1. Paper: 0.9937 for its pizza model, 0.3336 for its clock.

    NOTE (repeated in the README): s_g = 1 at r = 1 is FORCED by the architecture, so that
    endpoint is not evidence. The informative content is where the curve crosses in between.
    """
    rng = np.random.default_rng(seed)
    a = rng.integers(0, p, n_samples)
    b = rng.integers(0, p, n_samples)
    c = rng.integers(0, p, n_samples)
    e_a = model.W_E[torch.from_numpy(a).long()].detach().clone().requires_grad_(True)
    e_b = model.W_E[torch.from_numpy(b).long()].detach().clone().requires_grad_(True)
    lg = model.logits_from_embeddings(e_a, e_b)
    sel = lg[torch.arange(n_samples), torch.from_numpy(c).long()].sum()
    g_a, g_b = torch.autograd.grad(sel, [e_a, e_b])
    cs = F.cosine_similarity(g_a, g_b, dim=1).detach().numpy()
    return float(cs.mean()), float(cs.std())


def distance_profile(L, p):
    """f(d) = mean over i of L[i, i+d]: the correct logit as a function of the distance b-a.

    Pizza's amplitude factor cos(pi*w*(a-b)/p) makes this a strongly modulated curve; clock
    predicts it flat. `modulation` = std(f) / |mean(L)| is a scale-free amplitude of that
    modulation - a third, cruder discriminator reported as a cross-check on q.
    """
    idx = np.arange(p)
    f = np.array([L[idx, (idx + d) % p].mean() for d in range(p)])
    return f, float(f.std() / (abs(L.mean()) + 1e-30))


def embedding_fourier(W_E_num, share, max_k):
    """DFT down the token axis of the p number-token embedding rows; DC dropped.

    Returns (key frequencies, the power share they hold, effective number of frequencies,
    normalised spectrum). Both clock and pizza are Fourier-multiplication algorithms, so this
    is NOT a discriminator - it is a check that every arm learned a Fourier solution rather
    than something degenerate.
    """
    spec = np.fft.rfft(W_E_num, axis=0)
    pw = (np.abs(spec) ** 2)[1:].sum(axis=1)
    pw = pw / max(pw.sum(), 1e-30)
    order = np.argsort(-pw)
    keys, tot = [], 0.0
    for i in order:
        keys.append(int(i) + 1)
        tot += pw[i]
        if tot >= share or len(keys) >= max_k:
            break
    eff = float(1.0 / max((pw ** 2).sum(), 1e-30))
    return keys, float(tot), eff, pw


def distance_spectrum(L, p):
    """Power spectrum over the DISTANCE axis of the correct-logit surface.

    Row i of the reindexed surface is d -> L[i, i+d]; we remove each row's mean (stripping the
    sum-dependence) and take the DFT along d. A pizza model concentrates power in a few
    distance frequencies; a clock model leaves this spectrum close to flat noise.
    """
    idx = np.arange(p)
    A = np.stack([L[idx, (idx + d) % p] for d in range(p)], axis=1)   # A[i, d]
    A = A - A.mean(axis=1, keepdims=True)
    sp = np.fft.rfft(A, axis=1)
    pw = (np.abs(sp) ** 2)[:, 1:].sum(axis=0)
    return pw / max(pw.sum(), 1e-30)


# ----------------------------- training ---------------------------------------
def make_data(p, train_frac, seed):
    a, b = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    a, b = a.reshape(-1), b.reshape(-1)
    y = (a + b) % p
    X = torch.from_numpy(np.stack([a, b, np.full_like(a, p)], 1)).long()
    Y = torch.from_numpy(y).long()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(p * p)
    n_train = int(round(train_frac * p * p))
    tr, te = np.sort(perm[:n_train]), np.sort(perm[n_train:])
    return X[tr], Y[tr], X[te], Y[te]


def build_arm(P, r, seed):
    """Construct the model and its data split. At equal seed the initialisation and the split
    are IDENTICAL for every r, so any difference between arms is attributable to r alone."""
    p = int(P["p"])
    set_seeds(seed)
    Xtr, Ytr, Xte, Yte = make_data(p, P["train_frac"], seed)
    model = GrokTransformer(p, P["d_model"], P["n_heads"], P["d_mlp"], P["n_ctx"],
                            P["init_std_scale"], r)
    return model, (Xtr, Ytr, Xte, Yte)


def train_arm(P, model, data, r, seed, log):
    Xtr, Ytr, Xte, Yte = data
    opt = torch.optim.AdamW(model.parameters(), lr=P["lr"], weight_decay=P["weight_decay"],
                            betas=(P["beta1"], P["beta2"]))
    t0 = time.time()
    curve, step_grok = [], None
    for step in range(1, int(P["train_steps"]) + 1):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(Xtr), Ytr)
        loss.backward()
        opt.step()
        if step % int(P["eval_every"]) == 0 or step == 1:
            with torch.no_grad():
                tr_acc = (model(Xtr).argmax(-1) == Ytr).float().mean().item()
                te_out = model(Xte)
                te_acc = (te_out.argmax(-1) == Yte).float().mean().item()
                te_loss = F.cross_entropy(te_out, Yte).item()
            curve.append({"step": step, "train_acc": round(tr_acc, 4),
                          "test_acc": round(te_acc, 4), "test_loss": round(te_loss, 5)})
            if step_grok is None and te_acc >= P["grok_threshold"]:
                step_grok = step
    with torch.no_grad():
        final_tr = (model(Xtr).argmax(-1) == Ytr).float().mean().item()
        final_te = (model(Xte).argmax(-1) == Yte).float().mean().item()
    secs = time.time() - t0
    log(f"    r={r:<6} seed={seed}  train_acc={final_tr:.4f} test_acc={final_te:.4f} "
        f"grok@{step_grok}  {secs:.0f}s")
    return {"final_train_acc": round(final_tr, 4), "final_test_acc": round(final_te, 4),
            "step_to_grok": step_grok, "grokked": bool(final_te >= P["grok_threshold"]),
            "train_seconds": round(secs, 1), "curve": curve}


def analyse_arm(model, P):
    p = int(P["p"])
    L, _ = correct_logit_matrix(model, p)
    q, _ = distance_irrelevance(L, p)
    sg, sg_sd = gradient_symmetricity(model, p, int(P["grad_sym_samples"]), int(P["grad_sym_seed"]))
    f, mod = distance_profile(L, p)
    keys, share, eff, _ = embedding_fourier(model.W_E.detach().numpy()[:p],
                                            P["emb_fourier_share"], int(P["emb_fourier_max_k"]))
    dsp = distance_spectrum(L, p)
    top_d = [int(i) + 1 for i in np.argsort(-dsp)[:4]]
    rng = np.random.default_rng(0)                       # attention sanity probe
    a = rng.integers(0, p, 300); b = rng.integers(0, p, 300)
    Xs = torch.from_numpy(np.stack([a, b, np.full(300, p)], 1)).long()
    with torch.no_grad():
        att = model.attention_weights(Xs).squeeze(2).numpy()          # (N, H, T)
    return {
        "distance_irrelevance": round(q, 4),
        "grad_symmetricity": round(sg, 4),
        "grad_symmetricity_sd": round(sg_sd, 4),
        "dist_profile_modulation": round(mod, 4),
        "correct_logit_mean": round(float(L.mean()), 3),
        "correct_logit_std": round(float(L.std()), 3),
        "emb_key_freqs": keys,
        "emb_key_freq_power_share": round(share, 4),
        "emb_effective_n_freqs": round(eff, 3),
        "top_distance_freqs": top_d,
        "top_distance_freq_power": round(float(dsp.max()), 4),
        "attn_mean_per_position": [round(float(v), 4) for v in att.mean((0, 1))],
        "attn_l1_dev_from_uniform": round(float(np.abs(att - 1.0 / P["n_ctx"]).sum(-1).mean()), 4),
        "_L": L, "_f": f, "_dsp": dsp,
    }


# ----------------------------- boundary location ---------------------------------
def crossing(xs, ys, thresh):
    """First x at which the piecewise-linear curve y(x) crosses `thresh`. None if never."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    o = np.argsort(xs)
    xs, ys = xs[o], ys[o]
    for i in range(len(xs) - 1):
        y0, y1 = ys[i], ys[i + 1]
        if (y0 - thresh) * (y1 - thresh) <= 0 and y0 != y1:
            return float(xs[i] + (thresh - y0) * (xs[i + 1] - xs[i]) / (y1 - y0))
    return None


# ----------------------------- main ---------------------------------------
def main():
    cfg = load_config()
    P = cfg["params"]
    seed = int(cfg.get("seed", 0))
    t_start = time.time()
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    p = int(P["p"])
    arms = []                     # priority order: 5 headline r at both seeds, then refinements
    for s in P["seeds"]:
        for r in P["attention_rates_r"]:
            arms.append((float(r), int(s), "main"))
    for s in P["seeds"]:
        for r in P["refine_rates_r"]:
            arms.append((float(r), int(s), "refine"))

    log(f"clock-vs-pizza: p={p}, d_model={P['d_model']}, d_mlp={P['d_mlp']}, "
        f"train_frac={P['train_frac']}, {P['train_steps']} steps/arm, {len(arms)} arms queued")

    results, skipped = [], []
    n_params = None
    for (r, s, kind) in arms:
        if time.time() - t_start > P["total_train_budget_s"]:
            skipped.append({"r": r, "seed": s, "kind": kind})
            continue
        model, data = build_arm(P, r, s)
        if n_params is None:
            n_params = int(sum(w.numel() for w in model.parameters()))
        # ---- VALIDITY CONTROL: measure both discriminators on the UNTRAINED model.
        # Mixing attention toward uniform makes the network a more symmetric function of the
        # two operand embeddings whether or not it has learned anything, so part of any
        # s_g-vs-r ramp is architectural, not algorithmic. Subtracting this baseline is what
        # separates "the model LEARNED pizza" from "the knob mechanically symmetrised it".
        init_q, _ = distance_irrelevance(correct_logit_matrix(model, p)[0], p)
        init_sg, _ = gradient_symmetricity(model, p, int(P["grad_sym_samples"]),
                                           int(P["grad_sym_seed"]))
        tinfo = train_arm(P, model, data, r, s, log)
        ana = analyse_arm(model, P)
        row = {"r": r, "attention_rate_alpha": round(1.0 - r, 4), "seed": s, "kind": kind,
               "init_distance_irrelevance": round(init_q, 4),
               "init_grad_symmetricity": round(init_sg, 4)}
        row.update(tinfo)
        row.update(ana)
        log(f"       distance_irrelevance q={ana['distance_irrelevance']:.4f} (init {init_q:.4f})   "
            f"grad_symmetricity s_g={ana['grad_symmetricity']:.4f} (init {init_sg:.4f})   "
            f"dist-modulation={ana['dist_profile_modulation']:.4f}")
        results.append(row)
    for sk in skipped:
        log(f"  SKIPPED (budget): r={sk['r']} seed={sk['seed']} ({sk['kind']})")

    # ---- aggregate over seeds ----
    rs = sorted({row["r"] for row in results})
    agg = {}
    for r in rs:
        g = [row for row in results if row["r"] == r]
        agg[r] = {
            "n_seeds": len(g),
            "seeds": [row["seed"] for row in g],
            "distance_irrelevance_mean": round(float(np.mean([x["distance_irrelevance"] for x in g])), 4),
            "distance_irrelevance_per_seed": [x["distance_irrelevance"] for x in g],
            "grad_symmetricity_mean": round(float(np.mean([x["grad_symmetricity"] for x in g])), 4),
            "grad_symmetricity_per_seed": [x["grad_symmetricity"] for x in g],
            "dist_modulation_mean": round(float(np.mean([x["dist_profile_modulation"] for x in g])), 4),
            "test_acc_mean": round(float(np.mean([x["final_test_acc"] for x in g])), 4),
            "test_acc_per_seed": [x["final_test_acc"] for x in g],
            "train_acc_mean": round(float(np.mean([x["final_train_acc"] for x in g])), 4),
            "all_grokked": all(x["grokked"] for x in g),
            "step_to_grok_per_seed": [x["step_to_grok"] for x in g],
            "emb_effective_n_freqs_mean": round(float(np.mean([x["emb_effective_n_freqs"] for x in g])), 3),
            "attn_l1_dev_from_uniform_mean": round(float(np.mean([x["attn_l1_dev_from_uniform"] for x in g])), 4),
            "init_grad_symmetricity_mean": round(float(np.mean([x["init_grad_symmetricity"] for x in g])), 4),
            "init_distance_irrelevance_mean": round(float(np.mean([x["init_distance_irrelevance"] for x in g])), 4),
        }

    q_curve = [agg[r]["distance_irrelevance_mean"] for r in rs]
    s_curve = [agg[r]["grad_symmetricity_mean"] for r in rs]
    s_init_curve = [agg[r]["init_grad_symmetricity_mean"] for r in rs]
    q_init_curve = [agg[r]["init_distance_irrelevance_mean"] for r in rs]

    # Two boundary definitions:
    #  (a) internal midpoint - halfway between THIS sweep's own r=0 and r=1 endpoint values;
    #  (b) paper-referenced  - the threshold implied by Zhong et al.'s reported model values.
    sg_mid_internal = 0.5 * (s_curve[0] + s_curve[-1])
    q_mid_internal = 0.5 * (q_curve[0] + q_curve[-1])
    sg_paper_thresh = 0.5 * (P["ref_clock_grad_sym"] + P["ref_pizza_grad_sym"])
    q_paper_thresh = P["dist_irrel_pizza_ceiling"]

    boundary = {
        "r_star_grad_sym_internal_midpoint": crossing(rs, s_curve, sg_mid_internal),
        "grad_sym_internal_midpoint_value": round(float(sg_mid_internal), 4),
        "r_star_dist_irrel_internal_midpoint": crossing(rs, q_curve, q_mid_internal),
        "dist_irrel_internal_midpoint_value": round(float(q_mid_internal), 4),
        "r_star_grad_sym_paper_threshold": crossing(rs, s_curve, sg_paper_thresh),
        "grad_sym_paper_threshold": round(float(sg_paper_thresh), 4),
        "r_star_dist_irrel_paper_threshold": crossing(rs, q_curve, q_paper_thresh),
        "dist_irrel_paper_threshold": q_paper_thresh,
    }
    for k in [k for k in boundary if k.startswith("r_star_")]:
        if boundary[k] is not None:
            boundary[k] = round(boundary[k], 4)
    for k in [k for k in list(boundary) if k.startswith("r_star_")]:
        boundary[k.replace("r_star_", "alpha_star_")] = (
            None if boundary[k] is None else round(1.0 - boundary[k], 4))

    mono_q = all(q_curve[i] >= q_curve[i + 1] for i in range(len(q_curve) - 1))
    mono_s = all(s_curve[i] <= s_curve[i + 1] for i in range(len(s_curve) - 1))

    def sharpness(curve):
        """max single-interval change / mean interval change. 1.0 = perfectly linear ramp;
        large = the whole change is concentrated in one interval, i.e. a sharp transition."""
        d = np.abs(np.diff(curve))
        return None if d.sum() <= 0 else round(float(d.max() / d.mean()), 3)

    seed_spread_q = [float(np.ptp(agg[r]["distance_irrelevance_per_seed"])) for r in rs if agg[r]["n_seeds"] > 1]
    seed_spread_s = [float(np.ptp(agg[r]["grad_symmetricity_per_seed"])) for r in rs if agg[r]["n_seeds"] > 1]
    msq = max(seed_spread_q) if seed_spread_q else None
    mss = max(seed_spread_s) if seed_spread_s else None

    metrics = {
        "headline": ("distance irrelevance q and gradient symmetricity s_g (Zhong et al. Defs 4.2 / 4.1) "
                     "as a function of the uniform-attention mixing rate r, with the clock->pizza "
                     "phase boundary located by linear interpolation"),
        "n_params": n_params,
        "p": p, "d_model": P["d_model"], "d_mlp": P["d_mlp"], "n_heads": P["n_heads"],
        "train_frac": P["train_frac"], "train_steps": P["train_steps"],
        "n_arms_run": len(results), "n_arms_skipped": len(skipped), "skipped_arms": skipped,
        "r_values": rs,
        "alpha_values_paper_convention": [round(1 - r, 4) for r in rs],
        "distance_irrelevance_by_r": {str(r): agg[r]["distance_irrelevance_mean"] for r in rs},
        "distance_irrelevance_per_seed_by_r": {str(r): agg[r]["distance_irrelevance_per_seed"] for r in rs},
        "grad_symmetricity_by_r": {str(r): agg[r]["grad_symmetricity_mean"] for r in rs},
        "grad_symmetricity_per_seed_by_r": {str(r): agg[r]["grad_symmetricity_per_seed"] for r in rs},
        "dist_modulation_by_r": {str(r): agg[r]["dist_modulation_mean"] for r in rs},
        "test_acc_by_r": {str(r): agg[r]["test_acc_mean"] for r in rs},
        "test_acc_per_seed_by_r": {str(r): agg[r]["test_acc_per_seed"] for r in rs},
        "train_acc_by_r": {str(r): agg[r]["train_acc_mean"] for r in rs},
        "all_grokked_by_r": {str(r): agg[r]["all_grokked"] for r in rs},
        "step_to_grok_by_r": {str(r): agg[r]["step_to_grok_per_seed"] for r in rs},
        "censored_arms": [f"r={row['r']},seed={row['seed']}" for row in results if not row["grokked"]],
        "n_censored": sum(1 for row in results if not row["grokked"]),
        "attn_l1_dev_from_uniform_by_r": {str(r): agg[r]["attn_l1_dev_from_uniform_mean"] for r in rs},
        "emb_effective_n_freqs_by_r": {str(r): agg[r]["emb_effective_n_freqs_mean"] for r in rs},
        "init_grad_symmetricity_by_r": {str(r): agg[r]["init_grad_symmetricity_mean"] for r in rs},
        "init_distance_irrelevance_by_r": {str(r): agg[r]["init_distance_irrelevance_mean"] for r in rs},
        "learned_minus_init_grad_sym_by_r": {
            str(r): round(agg[r]["grad_symmetricity_mean"] - agg[r]["init_grad_symmetricity_mean"], 4) for r in rs},
        "learned_minus_init_dist_irrel_by_r": {
            str(r): round(agg[r]["distance_irrelevance_mean"] - agg[r]["init_distance_irrelevance_mean"], 4) for r in rs},
        "init_control_note": ("s_g and q measured on the UNTRAINED model at the same r and seed. "
                              "The architectural ramp is what the knob produces mechanically; the "
                              "trained-minus-init difference is what training actually contributed."),
        "r_star_grad_sym_init_baseline": crossing(rs, s_init_curve, 0.5 * (s_init_curve[0] + s_init_curve[-1])),
        "init_swing_grad_sym": round(float(s_init_curve[-1] - s_init_curve[0]), 4),
        "init_swing_dist_irrel": round(float(q_init_curve[0] - q_init_curve[-1]), 4),
        "trained_swing_over_init_swing_grad_sym": (
            round(float((s_curve[-1] - s_curve[0]) / (s_init_curve[-1] - s_init_curve[0])), 3)
            if abs(s_init_curve[-1] - s_init_curve[0]) > 1e-9 else None),
        "phase_boundary": boundary,
        "monotone_dist_irrelevance_decreasing_in_r": bool(mono_q),
        "monotone_grad_symmetricity_increasing_in_r": bool(mono_s),
        "sharpness_ratio_grad_sym": sharpness(s_curve),
        "sharpness_ratio_dist_irrel": sharpness(q_curve),
        "total_swing_grad_sym": round(float(s_curve[-1] - s_curve[0]), 4),
        "total_swing_dist_irrel": round(float(q_curve[0] - q_curve[-1]), 4),
        "max_seed_spread_dist_irrel": None if msq is None else round(msq, 4),
        "max_seed_spread_grad_sym": None if mss is None else round(mss, 4),
        "swing_over_seed_spread_dist_irrel": (
            round(float((q_curve[0] - q_curve[-1]) / msq), 2) if msq else None),
        "swing_over_seed_spread_grad_sym": (
            round(float((s_curve[-1] - s_curve[0]) / mss), 2) if mss else None),
        "paper_reference_values": {
            "pizza_grad_symmetricity": P["ref_pizza_grad_sym"],
            "clock_grad_symmetricity": P["ref_clock_grad_sym"],
            "pizza_distance_irrelevance": P["ref_pizza_dist_irrel"],
            "clock_distance_irrelevance": P["ref_clock_dist_irrel"],
            "pizza_dist_irrel_band": [0.0, P["dist_irrel_pizza_ceiling"]],
        },
        "endpoint_r0_vs_paper_clock": {
            "our_dist_irrel": q_curve[0], "paper_clock_dist_irrel": P["ref_clock_dist_irrel"],
            "our_grad_sym": s_curve[0], "paper_clock_grad_sym": P["ref_clock_grad_sym"],
        },
        "endpoint_r1_vs_paper_pizza": {
            "our_dist_irrel": q_curve[-1], "paper_pizza_dist_irrel": P["ref_pizza_dist_irrel"],
            "our_grad_sym": s_curve[-1], "paper_pizza_grad_sym": P["ref_pizza_grad_sym"],
            "reaches_paper_pizza_dist_irrel_band": bool(q_curve[-1] <= P["dist_irrel_pizza_ceiling"]),
        },
        "per_arm": [{k: v for k, v in row.items() if not k.startswith("_") and k != "curve"}
                    for row in results],
        "curves": {f"r={row['r']}_seed={row['seed']}": row["curve"] for row in results},
    }

    make_chart(P, rs, agg, results, boundary, metrics)

    out = {
        "id": cfg.get("id", "unknown"),
        "git_commit": git_sha(),
        "seed": seed,
        "duration_sec": round(time.time() - t_start, 2),
        "metrics": metrics,
        "env": env_info(),
        "log": lines,
        "status": "done",
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(out, f, indent=2)
    log(f"\nwrote results.json and chart.png in {out['duration_sec']:.0f}s")
    log(f"HEADLINE   q(r): {q_curve}   (high = clock)")
    log(f"HEADLINE s_g(r): {s_curve}   (high = pizza)")
    log(f"boundary: {json.dumps(boundary)}")


def make_chart(P, rs, agg, results, boundary, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = int(P["p"])
    fig, ax = plt.subplots(2, 3, figsize=(16.5, 9.5))
    CLOCK, PIZZA = "#1f77b4", "#d62728"
    _pal = plt.get_cmap("plasma")
    rcol = {r: _pal(0.05 + 0.8 * i / max(len(rs) - 1, 1)) for i, r in enumerate(rs)}
    s0 = results[0]["seed"]

    # (0,0) HEADLINE: the two discriminators vs r
    a0 = ax[0, 0]
    q_m = [agg[r]["distance_irrelevance_mean"] for r in rs]
    s_m = [agg[r]["grad_symmetricity_mean"] for r in rs]
    for r in rs:
        for v in agg[r]["distance_irrelevance_per_seed"]:
            a0.plot(r, v, ".", color=CLOCK, alpha=0.45, ms=9)
        for v in agg[r]["grad_symmetricity_per_seed"]:
            a0.plot(r, v, ".", color=PIZZA, alpha=0.45, ms=9)
    a0.axhspan(0, P["dist_irrel_pizza_ceiling"], color="0.86", alpha=0.6, zorder=0)
    a0.plot(rs, [agg[r]["init_distance_irrelevance_mean"] for r in rs], "--", color=CLOCK,
            alpha=0.55, lw=1.3, label="$q$ at INIT (untrained control)")
    a0.plot(rs, [agg[r]["init_grad_symmetricity_mean"] for r in rs], "--", color=PIZZA,
            alpha=0.55, lw=1.3, label="$s_g$ at INIT (architectural baseline)")
    a0.plot(rs, q_m, "-o", color=CLOCK, lw=2, label="distance irrelevance $q$  (high = clock)")
    a0.plot(rs, s_m, "-s", color=PIZZA, lw=2, label="gradient symmetricity $s_g$  (high = pizza)")
    a0.axhline(P["ref_clock_dist_irrel"], color=CLOCK, ls=":", lw=1)
    a0.axhline(P["ref_pizza_grad_sym"], color=PIZZA, ls=":", lw=1)
    a0.axhline(P["ref_clock_grad_sym"], color=PIZZA, ls=":", lw=1)
    a0.text(0.02, 0.05, "paper's pizza band for $q$ ($q\\leq0.4$)", fontsize=7, color="0.3")
    a0.text(0.02, P["ref_clock_dist_irrel"] + 0.02, "paper clock $q$=0.85", fontsize=7, color=CLOCK)
    a0.text(0.02, P["ref_clock_grad_sym"] + 0.02, "paper clock $s_g$=0.33", fontsize=7, color=PIZZA)
    rb = boundary.get("r_star_grad_sym_internal_midpoint")
    if rb is not None:
        a0.axvline(rb, color="k", ls="--", lw=1.5)
        a0.annotate(f"$r^*$={rb:.2f} ($s_g$)", (rb, 1.20), fontsize=8, ha="center")
    rb2 = boundary.get("r_star_dist_irrel_internal_midpoint")
    if rb2 is not None:
        a0.axvline(rb2, color="0.35", ls="-.", lw=1.3)
        a0.annotate(f"$r^*$={rb2:.2f} ($q$)", (rb2, 1.09), fontsize=8, ha="center", color="0.3")
    a0.set_xlabel("uniform-attention mixing rate $r$")
    a0.set_ylabel("discriminator value")
    a0.set_title("HEADLINE: clock $\\rightarrow$ pizza as attention\nis replaced by a uniform average")
    a0.set_ylim(-0.35, 1.32)
    a0.legend(fontsize=8, loc="center left")
    a0.grid(alpha=0.3)
    sec = a0.secondary_xaxis("top", functions=(lambda x: 1 - x, lambda x: 1 - x))
    sec.set_xlabel("attention rate $\\alpha = 1-r$  (Zhong et al. convention)", fontsize=8)

    # (0,1) control: did every arm still solve the task?
    a1 = ax[0, 1]
    a1.plot(rs, [agg[r]["train_acc_mean"] for r in rs], "-o", color="0.5", label="train acc")
    a1.plot(rs, [agg[r]["test_acc_mean"] for r in rs], "-o", color="darkgreen", label="test acc")
    a1.axhline(P["grok_threshold"], color="r", ls="--", lw=1, label=f"grok threshold {P['grok_threshold']}")
    a1.set_ylim(0.0, 1.06)
    a1.set_xlabel("$r$")
    a1.set_ylabel(f"accuracy after {P['train_steps']} steps")
    _nc, _na = metrics["n_censored"], metrics["n_arms_run"]
    a1.set_title("Control: every arm still solves the task\n"
                 "(nothing censored $\\Rightarrow$ we are comparing\nalgorithms, not competence)"
                 if _nc == 0 else
                 "Control: task performance by arm\n"
                 f"({_nc}/{_na} arms below the {P['grok_threshold']} grok threshold:\n"
                 + ", ".join(metrics["censored_arms"]) + ")")
    for _row in results:                       # ring the censored arms
        if not _row["grokked"]:
            a1.plot(_row["r"], _row["final_test_acc"], "o", mfc="none", mec="red", ms=13, mew=1.8)
    a1.legend(fontsize=8, loc="lower left")
    a1.grid(alpha=0.3)
    a1b = a1.twinx()
    stg = [np.mean([x for x in agg[r]["step_to_grok_per_seed"] if x is not None])
           if any(x is not None for x in agg[r]["step_to_grok_per_seed"]) else np.nan for r in rs]
    a1b.plot(rs, stg, "--^", color="purple", alpha=0.75, label="steps to grok")
    a1b.set_ylabel("steps to grok", color="purple", fontsize=9)
    a1b.legend(fontsize=8, loc="center right")

    # (0,2) grokking curves
    a2 = ax[0, 2]
    for row in results:
        if row["seed"] != s0 or row["kind"] != "main":
            continue
        cur = row["curve"]
        a2.plot([e["step"] for e in cur], [e["test_acc"] for e in cur],
                color=rcol[row["r"]], lw=1.8, label=f"r={row['r']}")
    a2.set_xlabel("step")
    a2.set_ylabel("test accuracy")
    a2.set_title(f"Grokking curves (seed {s0})\ndark = clock end ($r$=0), light = pizza end ($r$=1)")
    a2.legend(fontsize=7)
    a2.grid(alpha=0.3)

    # (1,0) the pizza signature: distance profile
    a3 = ax[1, 0]
    for row in results:
        if row["seed"] != s0 or row["kind"] != "main":
            continue
        f = row["_f"]
        a3.plot(np.arange(p), (f - f.mean()) / abs(row["correct_logit_mean"]),
                color=rcol[row["r"]], lw=1.7, label=f"r={row['r']}")
    a3.set_xlabel("distance $d = b - a \\ (\\mathrm{mod}\\ p)$")
    a3.set_ylabel("centred mean correct logit / |mean logit|")
    a3.set_title("The pizza signature: the correct logit acquires\na dependence on the distance $b-a$ as $r\\rightarrow1$")
    a3.legend(fontsize=7)
    a3.grid(alpha=0.3)

    # (1,1) / (1,2) correct-logit surface in (a, distance) coordinates: clock end vs pizza end
    main0 = [row for row in results if row["seed"] == s0 and row["kind"] == "main"]
    for j, row in enumerate([main0[0], main0[-1]]):
        axx = ax[1, 1 + j]
        L = row["_L"]
        idx = np.arange(p)
        A = np.stack([L[idx, (idx + d) % p] for d in range(p)], axis=1)   # A[a, d]
        im = axx.imshow(A, aspect="auto", cmap="viridis", origin="lower")
        axx.set_xlabel("distance $d = b-a$")
        axx.set_ylabel("$a$")
        axx.set_title(f"correct-logit surface, $r$={row['r']}  "
                      f"({'CLOCK end' if j == 0 else 'PIZZA end'})\n"
                      f"$q$={row['distance_irrelevance']:.3f},  $s_g$={row['grad_symmetricity']:.3f}\n"
                      "vertical banding = distance dependence")
        plt.colorbar(im, ax=axx, fraction=0.046)

    fig.suptitle("Clock vs Pizza on (a+b) mod 59: the algorithmic phase boundary vs attention rate\n"
                 f"1-layer transformer, d_model={P['d_model']}, d_mlp={P['d_mlp']}, "
                 f"{metrics['n_arms_run']} arms x {P['train_steps']} steps  "
                 "(discriminators: Zhong et al. arXiv:2306.17844, Defs 4.1 / 4.2)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(HERE / "chart.png", dpi=125)
    plt.close(fig)


if __name__ == "__main__":
    main()
