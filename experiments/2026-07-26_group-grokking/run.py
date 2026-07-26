"""Grokking beyond addition: abelian vs non-abelian group multiplication.

The lab has grokked (a+b) mod 59 twice (2026-07-25_grokking-modular-addition,
2026-07-25_grokking-weight-decay-phase) and mapped its algorithmic phase diagram
(2026-07-26_clock-vs-pizza). Every one of those rows lives on a COMMUTATIVE group,
where the learned algorithm is a Fourier multiplication and every irreducible
representation is one-dimensional. This run asks whether anything survives the
abelian / non-abelian divide.

Task: (a, b) -> a*b in a finite group G, sequence [a, b, "="], logits read from the
last position only, full-batch AdamW with weight decay 1.0 on a random fraction of the
|G|^2 table cells. Exactly the recipe of the sibling rows, with d_mlp shrunk 256 -> 128
as in clock-vs-pizza.

GROUPS (matched-order pairs, so table size / vocab / parameter count are identical
within a pair and the ONLY difference is commutativity):

  order 48, 2304 cells:  Z/48   (abelian, 48 one-dim irreps)
                         D_24   (NON-abelian, symmetries of the 24-gon;
                                 irreps 1,1,1,1 and eleven 2-dim)
  order 24,  576 cells:  Z/24   (abelian, 24 one-dim irreps)
                         S_4    (NON-abelian; irreps 1,1,2,3,3 -- the group used by
                                 the prior art, Chughtai et al. 2302.03025)

D_24 is the sharpest possible partner for Z/48: D_24 = Z/24 semidirect Z/2, so it is
the SAME order and differs from a cyclic group by exactly one semidirect twist.
S_5 (order 120, 14400 cells) from the backlog is far outside a 12-minute CPU box.

--------------------------------------------------------------------------------
THE TWO PROGRESS MEASURES, GENERALISED

Both measures the lab used on modular addition are stated in terms of "Fourier
frequency". The group-theoretic object that a frequency IS, is an irreducible
representation of Z/n. So both generalise verbatim once you replace "frequency" by
"irrep", and both reduce EXACTLY to the abelian versions when G is cyclic. That is what
makes this an apples-to-apples comparison across the divide rather than two different
experiments.

(1) IRREP POWER CONCENTRATION of the token embeddings (generalises "Fourier power
    concentration"). C[G] decomposes as a direct sum of isotypic components, one per
    irrep rho, of dimension d_rho^2. The orthogonal projector onto the rho-component of
    the regular representation is

        P_rho[x, y] = (d_rho / |G|) * chi_rho( y x^{-1} )

    (chi = character). Power of the embedding matrix in rho is tr(W^T P_rho W). For
    G = Z/n every d_rho = 1 and this is precisely the squared DFT magnitude at
    frequency rho. Reported as: the power distribution p_rho, its Shannon entropy, and
    -- the cross-group-comparable headline -- the excess concentration

        KL_bits = sum_rho p_rho * log2( p_rho / q_rho ),   q_rho = d_rho^2 / |G|

    against the NULL q_rho that a random embedding produces (verified in-code). Raw
    entropy is not comparable across groups because the groups have different numbers of
    irreps of different dimensions; KL against the group's own null is.

(2) RESTRICTED / EXCLUDED LOSS (Nanda et al. 2301.05217), generalised. The correct
    algorithm for a*b = c must produce logits of the form

        L(a, b, c) = phi( a b c^{-1} ),   phi a CLASS function

    because the ideal target delta(a b c^{-1} = e) equals (1/|G|) sum_rho d_rho *
    chi_rho(a b c^{-1}). The functions f_rho(a,b,c) = chi_rho(a b c^{-1}) are mutually
    orthogonal with ||f_rho||^2 = |G|^3, so the logit tensor splits cleanly into the part
    supported on the model's own KEY IRREPS (RESTRICTED) and the rest (EXCLUDED). Both
    facts are asserted numerically at startup: the ideal delta-logits reconstruct to
    ~1e-14 from the full irrep set, and random logits leave >99.9% of their energy
    outside it. For G = Z/n this is the translation-invariant subspace, i.e. Nanda's
    measure restricted to the diagonal frequency triples (k, k, k) -- the canonical
    version, slightly tighter than his product mask.

    Key irreps come from the FINAL model (Nanda's own protocol), so this measure is
    computed by replaying in-RAM checkpoints after training.

Lead time is measured from the MEMORISATION step (train acc >= 0.99), not step 0, and
each measure is normalised by its own total post-memorisation movement -- identical
machinery to 2026-07-25_grokking-modular-addition so the numbers are comparable to it.

Runs are attempted in a fixed priority order under a hard global wall-clock budget;
anything not reached is reported as skipped. Censored (never-grokked) arms are reported
honestly, not dropped.

Deterministic, CPU-only, single-threaded. Writes results.json and chart.png.

Usage:  python run.py
"""
import itertools
import json
import os
import random
import sys
import time
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
    """Read the current commit from .git WITHOUT invoking git (this run must not touch git)."""
    try:
        d = HERE
        for _ in range(6):
            g = d / ".git"
            if g.is_dir():
                head = (g / "HEAD").read_text().strip()
                if head.startswith("ref:"):
                    ref = head.split(" ", 1)[1].strip()
                    f = g / ref
                    if f.exists():
                        return f.read_text().strip()
                    packed = (g / "packed-refs")
                    if packed.exists():
                        for line in packed.read_text().splitlines():
                            if line.endswith(" " + ref):
                                return line.split(" ")[0]
                    return "nogit"
                return head
            d = d.parent
    except Exception:
        pass
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


# ============================ groups + characters ============================
def _cyclic(n):
    a = np.arange(n)
    mul = (a[:, None] + a[None, :]) % n
    chars = np.exp(2j * np.pi * np.outer(a, a) / n)          # chi_k(g) = e^{2 pi i k g / n}
    return dict(name=f"Z/{n}", n=n, mul=mul, chars=chars,
                dims=np.ones(n, dtype=int),
                irrep_labels=[f"k={i}" for i in range(n)], abelian=True)


def _dihedral(m):
    """D_m = symmetries of the m-gon, order 2m. element a^r b^s -> index r + m*s."""
    assert m % 2 == 0, "character-table branch below assumes m even"
    n = 2 * m
    ix = lambda r, s: (r % m) + m * (s % 2)
    mul = np.zeros((n, n), dtype=int)
    for r1 in range(m):
        for s1 in range(2):
            for r2 in range(m):
                for s2 in range(2):
                    r = (r1 + (r2 if s1 == 0 else -r2)) % m
                    mul[ix(r1, s1), ix(r2, s2)] = ix(r, s1 + s2)
    r = np.arange(m)
    rows, dims, labels = [], [], []
    for er, es, lab in [(1, 1, "triv"), (1, -1, "sgn_s"), (-1, 1, "sgn_r"), (-1, -1, "sgn_rs")]:
        base = np.power(float(er), r)
        rows.append(np.concatenate([base, base * float(es)]).astype(complex))
        dims.append(1)
        labels.append(f"1d-{lab}")
    for h in range(1, m // 2):
        rows.append(np.concatenate([2 * np.cos(2 * np.pi * h * r / m),
                                    np.zeros(m)]).astype(complex))
        dims.append(2)
        labels.append(f"2d-h={h}")
    return dict(name=f"D_{m}", n=n, mul=mul, chars=np.array(rows),
                dims=np.array(dims), irrep_labels=labels, abelian=False)


def _symmetric4():
    perms = list(itertools.permutations(range(4)))
    n, index = 24, {p: i for i, p in enumerate(perms)}
    mul = np.zeros((n, n), dtype=int)
    for i, p in enumerate(perms):
        for j, q in enumerate(perms):
            mul[i, j] = index[tuple(p[q[x]] for x in range(4))]   # (p q)(x) = p(q(x))

    def ctype(p):
        seen, cyc = set(), []
        for x in range(4):
            if x in seen:
                continue
            l, y = 0, x
            while y not in seen:
                seen.add(y)
                y = p[y]
                l += 1
            cyc.append(l)
        return tuple(sorted(cyc, reverse=True))

    order = [(1, 1, 1, 1), (2, 1, 1), (2, 2), (3, 1), (4,)]
    cls = np.array([order.index(ctype(p)) for p in perms])
    tab = np.array([[1, 1, 1, 1, 1],
                    [1, -1, 1, 1, -1],
                    [2, 0, 2, -1, 0],
                    [3, 1, -1, 0, -1],
                    [3, -1, -1, 0, 1]], dtype=float)
    return dict(name="S_4", n=n, mul=mul, chars=tab[:, cls].astype(complex),
                dims=np.array([1, 1, 2, 3, 3]),
                irrep_labels=["triv", "sgn", "2d", "std", "std*sgn"], abelian=False)


GROUP_BUILDERS = {"Z24": lambda: _cyclic(24), "Z48": lambda: _cyclic(48),
                  "D24": lambda: _dihedral(24), "S4": _symmetric4}


def build_group(key):
    G = GROUP_BUILDERS[key]()
    G["key"] = key
    n, mul = G["n"], G["mul"]
    e = int(np.where((mul == np.arange(n)[None, :]).all(1))[0][0])
    inv = np.argmax(mul == e, axis=1)
    G["e"], G["inv"] = e, inv
    # T[a, b, c] = a * b * c^{-1}
    G["T"] = mul[mul[:, :, None], inv[None, None, :]]
    # index matrix for the isotypic projector: PIX[x, y] = y * x^{-1}
    G["PIX"] = mul[np.arange(n)[None, :].repeat(n, 0),
                   inv[np.arange(n)[:, None].repeat(n, 1)]]
    return G


def verify_group(G, tol=1e-9):
    """Assert the table is a group and the character table is the real one. Returns a dict."""
    n, mul, ch, d = G["n"], G["mul"], G["chars"], G["dims"]
    rng = np.random.default_rng(0)
    for i in range(n):                       # Latin square
        assert sorted(mul[i].tolist()) == list(range(n))
        assert sorted(mul[:, i].tolist()) == list(range(n))
    a = np.repeat(np.arange(n), n * n)        # associativity, all n^3 triples
    b = np.tile(np.repeat(np.arange(n), n), n)
    c = np.tile(np.arange(n), n * n)
    assert (mul[mul[a, b], c] == mul[a, mul[b, c]]).all(), "not associative"
    assert int((d ** 2).sum()) == n, "sum of squared irrep dims != |G|"
    assert np.abs(ch[:, G["e"]] - d).max() < tol, "chi(e) != dim"
    assert np.abs(ch @ ch.conj().T / n - np.eye(len(d))).max() < tol, "characters not orthonormal"
    for r in range(len(d)):                   # class functions
        for h in rng.integers(0, n, 4):
            assert np.abs(ch[r][mul[mul[int(h)], G["inv"][int(h)]]] - ch[r]).max() < tol
    P = np.stack([d[r] / n * ch[r][G["PIX"]] for r in range(len(d))])
    assert np.abs(P.sum(0) - np.eye(n)).max() < tol, "isotypic projectors do not sum to I"
    assert np.abs(P[0] @ P[0] - P[0]).max() < tol, "projector not idempotent"
    # the ideal delta-logits must lie EXACTLY in the class-function-of-abc^{-1} subspace
    L = np.zeros((n, n, n))
    aa, bb = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    L[aa, bb, mul] = 1.0
    L = L - L.mean(-1, keepdims=True)
    rec = class_project(L, G, np.arange(len(d)))
    err_ideal = float(np.abs(rec - L).max())
    R = rng.normal(size=(n, n, n))
    R = R - R.mean(-1, keepdims=True)
    out_frac = float(((R - class_project(R, G, np.arange(len(d)))) ** 2).sum() / (R ** 2).sum())
    assert err_ideal < 1e-9, "ideal logits not spanned by the irrep basis"
    assert out_frac > 0.99, "random logits suspiciously inside the restricted subspace"
    return {"group": G["name"], "order": int(n), "abelian": bool(G["abelian"]),
            "n_irreps": int(len(d)), "irrep_dims": [int(x) for x in d],
            "ideal_logit_reconstruction_maxerr": err_ideal,
            "random_logit_energy_outside_restricted_subspace": round(out_frac, 6)}


def class_project(L, G, keep):
    """Project L (n,n,n) onto span{ (a,b,c) -> chi_rho(a b c^{-1}) : rho in keep }.

    The f_rho are orthogonal with ||f_rho||^2 = n^3, so the projection collapses to a
    single scatter-add over the n^3 index tensor followed by a band-limit in irrep space.
    """
    n, T, ch = G["n"], G["T"], G["chars"]
    S = np.bincount(T.reshape(-1), weights=np.ascontiguousarray(L).reshape(-1), minlength=n)
    coef = (ch[keep].conj() @ S) / n ** 3
    return np.real(coef @ ch[keep])[T]


def irrep_power(W, G):
    """Isotypic power of the columns of W (n, d): power_rho = tr(W^T P_rho W).

    Uses power_rho = (d_rho/n) * sum_g chi_rho(g) * S(g) with
    S(g) = sum_{x,y: y x^{-1} = g} <W_x, W_y>, i.e. one Gram + one scatter-add.
    """
    n, ch, dims = G["n"], G["chars"], G["dims"]
    C = np.asarray(W, dtype=np.float64) @ np.asarray(W, dtype=np.float64).T
    S = np.bincount(G["PIX"].reshape(-1), weights=C.reshape(-1), minlength=n)
    pw = np.real(ch @ S) * dims / n
    return np.clip(pw, 0.0, None)


def conj_closure(G, idxs):
    """Close a set of irreps under complex conjugation (matters for cyclic groups)."""
    ch = G["chars"]
    out = set(int(i) for i in idxs)
    for i in list(out):
        d = np.abs(ch - ch[i].conj()[None, :]).max(axis=1)
        out.add(int(np.argmin(d)))
    return sorted(out)


# ============================ model ============================
class GrokTransformer(nn.Module):
    """1-layer attention + MLP transformer, NO LayerNorm, no biases (Nanda-style).

    Vocab is |G|+1 (group elements plus "="). Input is [a, b, =]; logits over the |G|
    elements are read from the last position only, so the MLP/unembed run at that one
    position (mathematically identical to the full pass, ~3x cheaper).
    """

    def __init__(self, n, d_model, n_heads, d_mlp, n_ctx, init_std_scale):
        super().__init__()
        self.n, self.d_model, self.n_heads = n, d_model, n_heads
        self.d_head = d_model // n_heads
        std = init_std_scale / (d_model ** 0.5)
        self.W_E = nn.Parameter(torch.randn(n + 1, d_model) * std)
        self.W_pos = nn.Parameter(torch.randn(n_ctx, d_model) * std)
        self.W_Q = nn.Parameter(torch.randn(d_model, d_model) * std)
        self.W_K = nn.Parameter(torch.randn(d_model, d_model) * std)
        self.W_V = nn.Parameter(torch.randn(d_model, d_model) * std)
        self.W_O = nn.Parameter(torch.randn(d_model, d_model) * std)
        self.W_in = nn.Parameter(torch.randn(d_model, d_mlp) * std)
        self.W_out = nn.Parameter(torch.randn(d_mlp, d_model) * std)
        self.W_U = nn.Parameter(torch.randn(d_model, n) * std)

    def hidden(self, idx):
        N, T = idx.shape
        H, Dh = self.n_heads, self.d_head
        x = self.W_E[idx] + self.W_pos[None, :T, :]
        last = x[:, -1:, :]
        q = (last @ self.W_Q).view(N, 1, H, Dh).transpose(1, 2)
        k = (x @ self.W_K).view(N, T, H, Dh).transpose(1, 2)
        v = (x @ self.W_V).view(N, T, H, Dh).transpose(1, 2)
        att = (q @ k.transpose(-2, -1) / (Dh ** 0.5)).softmax(-1)
        z = (att @ v).transpose(1, 2).reshape(N, 1, self.d_model) @ self.W_O
        h = last + z
        h = h + F.relu(h @ self.W_in) @ self.W_out
        return h.view(N, self.d_model)

    def forward(self, idx):
        return self.hidden(idx) @ self.W_U


# ============================ helpers ============================
def entropy_bits(pk):
    pk = np.asarray(pk, dtype=np.float64)
    pk = pk / max(pk.sum(), 1e-30)
    return float(-(pk * np.log2(pk + 1e-30)).sum())


def kl_bits(p, q):
    p = np.asarray(p, np.float64) / max(np.sum(p), 1e-30)
    q = np.asarray(q, np.float64) / max(np.sum(q), 1e-30)
    return float((p * np.log2((p + 1e-30) / (q + 1e-30))).sum())


def svd_entropy(W):
    s = np.linalg.svd(np.asarray(W, dtype=np.float64), compute_uv=False)
    return entropy_bits(s ** 2)


def first_crossing(steps, values, thresh, rising=True, start_step=None):
    idx0 = 0
    if start_step is not None:
        while idx0 < len(steps) and steps[idx0] < start_step:
            idx0 += 1
        if idx0 >= len(steps):
            return None
    if not values:
        return None
    if rising and values[idx0] >= thresh:
        return float(steps[idx0])
    if (not rising) and values[idx0] <= thresh:
        return float(steps[idx0])
    for i in range(idx0 + 1, len(values)):
        a, b = values[i - 1], values[i]
        hit = (a < thresh <= b) if rising else (a > thresh >= b)
        if hit:
            f = (thresh - a) / (b - a) if b != a else 0.0
            return float(steps[i - 1] + f * (steps[i] - steps[i - 1]))
    return None


def interp_at(steps, values, x):
    if x is None or not steps:
        return None
    return float(np.interp(x, steps, values))


def progress_fraction(steps, values, t_mem):
    """Fraction of total POST-MEMORISATION movement completed. Sign-agnostic."""
    if t_mem is None:
        return None
    v_mem = interp_at(steps, values, t_mem)
    post = [v for s, v in zip(steps, values) if s >= t_mem]
    if not post:
        return None
    denom = post[-1] - v_mem
    if abs(denom) < 1e-12:
        return None
    return [float((v - v_mem) / denom) for v in values]


# ============================ one run ============================
def train_one(P, G, train_frac, max_steps, time_cap_s, seed, log):
    n = G["n"]
    set_seeds(seed)
    t0 = time.time()

    a_all, b_all = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    a_all, b_all = a_all.reshape(-1), b_all.reshape(-1)
    y_all = G["mul"][a_all, b_all]
    X = torch.from_numpy(np.stack([a_all, b_all, np.full_like(a_all, n)], 1)).long()
    Y = torch.from_numpy(y_all).long()

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n * n)
    n_train = int(round(train_frac * n * n))
    tr_idx, te_idx = np.sort(perm[:n_train]), np.sort(perm[n_train:])
    Xtr, Ytr, Xte, Yte = X[tr_idx], Y[tr_idx], X[te_idx], Y[te_idx]
    tr_a, tr_b, tr_y = a_all[tr_idx], b_all[tr_idx], y_all[tr_idx]
    te_a, te_b, te_y = a_all[te_idx], b_all[te_idx], y_all[te_idx]
    log(f"  {G['name']:5s} frac={train_frac}  {len(tr_idx)} train / {len(te_idx)} test  "
        f"({'abelian' if G['abelian'] else 'NON-abelian'}, {len(G['dims'])} irreps)")

    model = GrokTransformer(n, P["d_model"], P["n_heads"], P["d_mlp"],
                            P["n_ctx"], P["init_std_scale"])
    n_params = sum(q.numel() for q in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=P["lr"], weight_decay=P["weight_decay"],
                            betas=(P["beta1"], P["beta2"]))
    null_p = (G["dims"].astype(np.float64) ** 2) / n            # random-embedding null

    @torch.no_grad()
    def evaluate():
        model.eval()
        out = {}
        ltr = model(Xtr)
        out["train_loss"] = float(F.cross_entropy(ltr, Ytr))
        out["train_acc"] = float((ltr.argmax(-1) == Ytr).float().mean())
        lte = model(Xte)
        out["test_loss"] = float(F.cross_entropy(lte, Yte))
        out["test_acc"] = float((lte.argmax(-1) == Yte).float().mean())
        we = model.W_E.detach().numpy()[:n]
        pw = irrep_power(we, G)
        p = pw / max(pw.sum(), 1e-30)
        out["_pw"] = p
        out["H_irrep"] = entropy_bits(p)
        out["KL_irrep"] = kl_bits(p, null_p)
        out["H_svd"] = svd_entropy(we)
        model.train()
        return out

    keys = ("step", "train_acc", "test_acc", "train_loss", "test_loss",
            "H_irrep", "KL_irrep", "H_svd", "wall_s")
    hist = {k: [] for k in keys}
    ckpts = []
    init_pw = final_pw = None
    ev = int(P["eval_every"])
    ck_every = int(P["ckpt_every_evals"]) * ev
    capped, step, n_ev = False, 0, 0

    for step in range(int(max_steps) + 1):
        if step % ev == 0:
            e = evaluate()
            if init_pw is None:
                init_pw = e["_pw"]
            final_pw = e["_pw"]
            hist["step"].append(step)
            hist["wall_s"].append(round(time.time() - t0, 2))
            for k in ("train_acc", "test_acc", "train_loss", "test_loss",
                      "H_irrep", "KL_irrep", "H_svd"):
                hist[k].append(e[k])
            if step % ck_every == 0:
                ckpts.append((step, {k: v.detach().clone()
                                     for k, v in model.state_dict().items()}))
            n_ev += 1
            if n_ev % 25 == 1:
                log(f"    step {step:6d}  tr {e['train_acc']:.4f}  te {e['test_acc']:.4f}  "
                    f"KL {e['KL_irrep']:.3f}  H {e['H_irrep']:.3f}  ({time.time()-t0:.0f}s)")
        if step == int(max_steps):
            break
        if time.time() - t0 > time_cap_s:
            capped = True
            log(f"    TIME CAP at step {step} ({time.time()-t0:.0f}s)")
            break
        loss = F.cross_entropy(model(Xtr), Ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()

    steps_run = step
    final = evaluate()
    final_pw = final["_pw"]

    # ---- key irreps of the FINAL model, then replay restricted/excluded ----
    order = np.argsort(final_pw)[::-1]
    cum = np.cumsum(final_pw[order])
    n_key = int(min(np.searchsorted(cum, float(P["key_irrep_power"])) + 1,
                    int(P["max_key_irreps"]), len(order)))
    key_irreps = conj_closure(G, [int(order[i]) for i in range(n_key)])
    key_share = float(final_pw[key_irreps].sum())
    def ce_acc(arr, ia, ib, y):
        z = torch.from_numpy(np.ascontiguousarray(arr[ia, ib])).double()
        yy = torch.from_numpy(y).long()
        return float(F.cross_entropy(z, yy)), float((z.argmax(-1) == yy).double().mean())

    rep = {k: [] for k in ("step", "restricted", "excluded", "full",
                           "restricted_test_ce", "restricted_train_acc",
                           "restricted_test_acc", "excluded_test_acc",
                           "restricted_energy_frac")}
    probe = GrokTransformer(n, P["d_model"], P["n_heads"], P["d_mlp"],
                            P["n_ctx"], P["init_std_scale"])
    all_irreps = np.arange(len(G["dims"]))
    R_all_final_acc = None
    with torch.no_grad():
        for st, sd in ckpts:
            probe.load_state_dict(sd)
            L = probe(X).numpy().reshape(n, n, n).astype(np.float64)
            L = L - L.mean(-1, keepdims=True)
            R = class_project(L, G, np.array(key_irreps))
            Ex = L - R
            r_tr_ce, r_tr_ac = ce_acc(R, tr_a, tr_b, tr_y)
            r_te_ce, r_te_ac = ce_acc(R, te_a, te_b, te_y)
            e_tr_ce, _ = ce_acc(Ex, tr_a, tr_b, tr_y)
            _, e_te_ac = ce_acc(Ex, te_a, te_b, te_y)
            f_tr_ce, _ = ce_acc(L, tr_a, tr_b, tr_y)
            rep["step"].append(st)
            rep["restricted"].append(r_tr_ce)
            rep["restricted_test_ce"].append(r_te_ce)
            rep["restricted_train_acc"].append(r_tr_ac)
            rep["restricted_test_acc"].append(r_te_ac)
            rep["excluded"].append(e_tr_ce)
            rep["excluded_test_acc"].append(e_te_ac)
            rep["full"].append(f_tr_ce)
            # THE quantitative version: what SHARE of the centred logit energy actually sits
            # in the correct-algorithm subspace? Restricted-only accuracy is only a 1-bit
            # (sign) statement -- it is 1.0 whenever the class-function part peaks at the
            # identity, however small that part is. This is the magnitude.
            rep["restricted_energy_frac"].append(float((R ** 2).sum() / max((L ** 2).sum(), 1e-30)))
        # robustness: same readout with EVERY irrep kept, on the final checkpoint, so the
        # "correct circuit is present" claim does not depend on the key-irrep selection.
        if ckpts:
            probe.load_state_dict(ckpts[-1][1])
            L = probe(X).numpy().reshape(n, n, n).astype(np.float64)
            L = L - L.mean(-1, keepdims=True)
            R_all_final_acc = ce_acc(class_project(L, G, all_irreps), te_a, te_b, te_y)[1]
    rep_steps = rep["step"]
    rep_restricted, rep_excluded, rep_full = rep["restricted"], rep["excluded"], rep["full"]
    train_s = time.time() - t0
    log(f"    key irreps {[G['irrep_labels'][i] for i in key_irreps]} "
        f"({100*key_share:.1f}% of W_E isotypic power); replayed {len(ckpts)} ckpts; "
        f"{train_s:.0f}s")

    return dict(group=G["key"], group_name=G["name"], abelian=bool(G["abelian"]),
                order=n, train_frac=train_frac, n_params=n_params,
                n_train=int(len(tr_idx)), n_test=int(len(te_idx)),
                steps_run=steps_run, time_capped=capped, max_steps=int(max_steps),
                train_seconds=round(train_s, 1),
                sec_per_step=round(train_s / max(steps_run, 1), 4),
                hist=hist, final=final, init_pw=init_pw, final_pw=final_pw,
                null_pw=null_p, key_irreps=key_irreps, key_irrep_share=key_share,
                irrep_labels=G["irrep_labels"], dims=G["dims"],
                rep_steps=rep_steps, rep_restricted=rep_restricted,
                rep_excluded=rep_excluded, rep_full=rep_full, rep=rep,
                restricted_all_irreps_test_acc=R_all_final_acc)


# ============================ lead-time analysis ============================
def analyse(run, P):
    hist = run["hist"]
    S, tr_a, te_a = hist["step"], hist["train_acc"], hist["test_acc"]
    t_mem = first_crossing(S, tr_a, float(P["train_acc_memorized_threshold"]))
    chance = 1.0 / run["order"]
    t_test = {str(t): first_crossing(S, te_a, t, start_step=t_mem)
              for t in P["test_acc_jump_thresholds"]}
    t_plateau_exit = first_crossing(S, te_a, chance + float(P["plateau_exit_abs"]),
                                    start_step=t_mem)
    t_test50 = t_test.get("0.5")

    series = {"KL_irrep": (S, hist["KL_irrep"]),
              "H_irrep": (S, hist["H_irrep"]),
              "H_svd": (S, hist["H_svd"]),
              "restricted_loss": (run["rep_steps"], run["rep_restricted"]),
              "excluded_loss": (run["rep_steps"], run["rep_excluded"]),
              "restricted_energy_frac": (run["rep_steps"],
                                         run["rep"]["restricted_energy_frac"])}
    measures = {}
    for name, (st, val) in series.items():
        frac = progress_fraction(st, val, t_mem)
        m = {"value_at_memorization": interp_at(st, val, t_mem),
             "value_final": float(val[-1]) if val else None}
        if frac is None:
            m.update({"t_10pct": None, "t_50pct": None, "lead_vs_test50_steps": None,
                      "lead_vs_plateau_exit_steps": None, "frac_done_at_test50": None,
                      "frac_overshoot": None, "monotone_post_mem": None})
        else:
            t10 = first_crossing(st, frac, 0.10, start_step=t_mem)
            t50 = first_crossing(st, frac, 0.50, start_step=t_mem)
            m["t_10pct"], m["t_50pct"] = t10, t50
            m["lead_vs_test50_steps"] = (None if (t50 is None or t_test50 is None)
                                         else round(t_test50 - t50, 1))
            m["lead_vs_plateau_exit_steps"] = (None if (t10 is None or t_plateau_exit is None)
                                               else round(t_plateau_exit - t10, 1))
            m["frac_done_at_test50"] = (None if t_test50 is None
                                        else round(float(interp_at(st, frac, t_test50)), 4))
            post = [f for s, f in zip(st, frac) if t_mem is None or s >= t_mem]
            m["frac_overshoot"] = round(float(max(post)), 3) if post else None
            m["monotone_post_mem"] = bool(post and max(post) <= 1.02)
            m["_frac_series"] = [round(float(x), 5) for x in frac]
            m["_frac_steps"] = [int(x) for x in st]
        measures[name] = m
    return dict(t_mem=t_mem, t_test=t_test, t_test50=t_test50,
                t_plateau_exit=t_plateau_exit, chance=chance, measures=measures)


# ============================ main ============================
def main():
    cfg = load_config()
    P = cfg["params"]
    seed = int(cfg.get("seed", 0))
    t0 = time.time()
    log = lambda s: print(s, flush=True)

    # --- build + verify every group before spending any compute ---
    groups, group_checks = {}, []
    for key in sorted({r["group"] for r in P["runs"]}):
        G = build_group(key)
        chk = verify_group(G)
        groups[key] = G
        group_checks.append(chk)
        log(f"group OK: {chk['group']:5s} order {chk['order']:3d} "
            f"{'abelian    ' if chk['abelian'] else 'NON-abelian'} "
            f"irreps {chk['irrep_dims'] if len(chk['irrep_dims']) <= 8 else str(chk['irrep_dims'][:6])+'...'} "
            f"| ideal-logit reconstruction err {chk['ideal_logit_reconstruction_maxerr']:.1e}, "
            f"random-logit energy outside restricted subspace "
            f"{chk['random_logit_energy_outside_restricted_subspace']:.4f}")

    budget = float(P["total_train_budget_s"])
    runs, analyses, skipped = [], [], []
    for spec in P["runs"]:
        spent = time.time() - t0
        if spent + 12.0 > budget:
            skipped.append({"group": spec["group"], "train_frac": spec["train_frac"],
                            "reason": f"global budget ({budget}s) exhausted at {spent:.0f}s"})
            log(f"SKIP {spec['group']} frac={spec['train_frac']} (budget)")
            continue
        cap = min(float(spec["time_cap_s"]), budget - spent - 5.0)
        log(f"run: {spec['group']} train_frac={spec['train_frac']} "
            f"cap={cap:.0f}s max_steps={spec['max_steps']}")
        r = train_one(P, groups[spec["group"]], float(spec["train_frac"]),
                      int(spec["max_steps"]), cap, seed, log)
        a = analyse(r, P)
        delay = (None if (a["t_mem"] is None or a["t_test50"] is None)
                 else a["t_test50"] - a["t_mem"])
        log(f"    -> final test_acc {r['final']['test_acc']:.4f} after {r['steps_run']} steps; "
            f"mem@{a['t_mem']}, test50@{a['t_test50']}, delay="
            f"{'CENSORED' if delay is None else f'{delay:.0f}'}")
        runs.append(r)
        analyses.append(a)

    # ------------------------------ metrics ------------------------------
    def run_metrics(r, a):
        ms = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
              for k, v in a["measures"].items()}
        pw, npw = r["final_pw"], r["null_pw"]
        ordv = np.argsort(pw)[::-1]
        rp = r["rep"]
        grokked = bool(r["final"]["test_acc"] >= 0.9)
        assert abs(rp["restricted"][-1] - rp["restricted_test_ce"][-1]) < 1e-6, \
            "restricted CE must be identical on train and test by construction"
        return {
            "group": r["group"], "group_name": r["group_name"], "abelian": r["abelian"],
            "order": r["order"], "n_irreps": int(len(pw)),
            "train_frac": r["train_frac"], "n_train": r["n_train"], "n_test": r["n_test"],
            "n_params": r["n_params"],
            "steps_run": r["steps_run"], "max_steps": r["max_steps"],
            "time_capped": r["time_capped"], "train_seconds": r["train_seconds"],
            "sec_per_step": r["sec_per_step"],
            "final_train_acc": round(r["final"]["train_acc"], 4),
            "final_test_acc": round(r["final"]["test_acc"], 4),
            "final_train_loss": round(r["final"]["train_loss"], 5),
            "final_test_loss": round(r["final"]["test_loss"], 5),
            "best_test_acc": round(float(max(r["hist"]["test_acc"])), 4),
            "memorized": bool(max(r["hist"]["train_acc"])
                              >= P["train_acc_memorized_threshold"]),
            "left_plateau": a["t_plateau_exit"] is not None,
            "grokked_full": grokked,
            "step_memorized": a["t_mem"],
            "step_test_acc_thresholds": a["t_test"],
            "step_test_plateau_exit": a["t_plateau_exit"],
            "grok_delay_steps": (None if (a["t_mem"] is None or a["t_test50"] is None)
                                 else round(a["t_test50"] - a["t_mem"], 1)),
            "grok_delay_censored": bool(a["t_test50"] is None),
            "KL_irrep_init": round(r["hist"]["KL_irrep"][0], 4),
            "KL_irrep_final": round(r["hist"]["KL_irrep"][-1], 4),
            "H_irrep_init": round(r["hist"]["H_irrep"][0], 4),
            "H_irrep_final": round(r["hist"]["H_irrep"][-1], 4),
            "H_irrep_null_bits": round(entropy_bits(npw), 4),
            "H_svd_init": round(r["hist"]["H_svd"][0], 4),
            "H_svd_final": round(r["hist"]["H_svd"][-1], 4),
            "effective_n_irreps_init": round(float(2 ** r["hist"]["H_irrep"][0]), 2),
            "effective_n_irreps_final": round(float(2 ** r["hist"]["H_irrep"][-1]), 2),
            "top1_irrep_share_final": round(float(pw[ordv[0]]), 4),
            "top3_irrep_share_final": round(float(pw[ordv[:3]].sum()), 4),
            "top3_irrep_null_share": round(float(npw[ordv[:3]].sum()), 4),
            "top3_irrep_labels": [r["irrep_labels"][int(i)] for i in ordv[:3]],
            "key_irreps": [r["irrep_labels"][i] for i in r["key_irreps"]],
            "n_key_irreps": len(r["key_irreps"]),
            "key_irrep_power_share": round(r["key_irrep_share"], 4),
            # --- is the CORRECT group-composition circuit present in the logits even when
            #     the model does not generalise? A restricted logit is phi(a b c^{-1}), whose
            #     softmax cross-entropy is IDENTICAL on every (a, b) cell (asserted below), so
            #     its accuracy is the same on train and test and is in fact a ONE-BIT quantity:
            #     1.0 iff phi peaks at the identity, however tiny phi is. The magnitude that
            #     actually matters is therefore the ENERGY FRACTION, reported alongside.
            "RESTRICTED_energy_frac_final": round(float(rp["restricted_energy_frac"][-1]), 6),
            "RESTRICTED_energy_frac_at_memorization": (
                None if a["t_mem"] is None else
                round(float(interp_at(rp["step"], rp["restricted_energy_frac"], a["t_mem"])), 6)),
            "RESTRICTED_ONLY_test_acc_final": round(float(rp["restricted_test_acc"][-1]), 4),
            "RESTRICTED_ONLY_train_acc_final": round(float(rp["restricted_train_acc"][-1]), 4),
            "RESTRICTED_ONLY_test_acc_all_irreps_final": (
                None if r["restricted_all_irreps_test_acc"] is None
                else round(float(r["restricted_all_irreps_test_acc"]), 4)),
            "restricted_train_ce_final": round(float(rp["restricted"][-1]), 6),
            "restricted_test_ce_final": round(float(rp["restricted_test_ce"][-1]), 6),
            "restricted_train_minus_test_ce": round(float(rp["restricted"][-1]
                                                          - rp["restricted_test_ce"][-1]), 9),
            "excluded_only_test_acc_final": round(float(rp["excluded_test_acc"][-1]), 4),
            "full_minus_restricted_test_acc": round(float(r["final"]["test_acc"]
                                                          - rp["restricted_test_acc"][-1]), 4),
            "measures": ms,
            "history": {k: ([int(x) for x in v] if k == "step"
                            else [round(float(x), 6) for x in v])
                        for k, v in r["hist"].items()},
            "replay_history": {k: ([int(x) for x in v] if k == "step"
                                   else [round(float(x), 6) for x in v])
                               for k, v in rp.items()},
            "final_irrep_power": [round(float(x), 6) for x in pw],
            "init_irrep_power": [round(float(x), 6) for x in r["init_pw"]],
            "null_irrep_power": [round(float(x), 6) for x in npw],
            "irrep_labels": r["irrep_labels"],
        }

    per_run = [run_metrics(r, a) for r, a in zip(runs, analyses)]
    by_key = {(m["group"], m["train_frac"]): m for m in per_run}

    # --- matched-order abelian vs non-abelian comparisons ---
    pairs_spec = [("Z48", "D24", 48), ("Z24", "S4", 24)]
    comparisons = []
    for ab, nab, order in pairs_spec:
        for frac in sorted({m["train_frac"] for m in per_run}):
            A, B = by_key.get((ab, frac)), by_key.get((nab, frac))
            if A is None or B is None:
                continue
            comparisons.append({
                "order": order, "train_frac": frac,
                "abelian_group": A["group_name"], "nonabelian_group": B["group_name"],
                "n_train": A["n_train"],
                "abelian_grokked": A["grokked_full"], "nonabelian_grokked": B["grokked_full"],
                "abelian_final_test_acc": A["final_test_acc"],
                "nonabelian_final_test_acc": B["final_test_acc"],
                "abelian_step_memorized": A["step_memorized"],
                "nonabelian_step_memorized": B["step_memorized"],
                "abelian_grok_delay": A["grok_delay_steps"],
                "nonabelian_grok_delay": B["grok_delay_steps"],
                "nonabelian_delay_censored_at_step": (B["steps_run"]
                                                      if B["grok_delay_censored"] else None),
                "delay_ratio_nonabelian_over_abelian": (
                    None if (A["grok_delay_steps"] in (None, 0)
                             or B["grok_delay_steps"] is None)
                    else round(B["grok_delay_steps"] / A["grok_delay_steps"], 3)),
                "abelian_KL_irrep_final": A["KL_irrep_final"],
                "nonabelian_KL_irrep_final": B["KL_irrep_final"],
                "abelian_restricted_lead": A["measures"]["restricted_loss"]["lead_vs_test50_steps"],
                "nonabelian_restricted_lead": B["measures"]["restricted_loss"]["lead_vs_test50_steps"],
                "abelian_excluded_lead": A["measures"]["excluded_loss"]["lead_vs_test50_steps"],
                "nonabelian_excluded_lead": B["measures"]["excluded_loss"]["lead_vs_test50_steps"],
                "abelian_KL_lead": A["measures"]["KL_irrep"]["lead_vs_test50_steps"],
                "nonabelian_KL_lead": B["measures"]["KL_irrep"]["lead_vs_test50_steps"],
            })

    # --- the correct-circuit-vs-readout dissociation ---
    circuit = [{"run": f"{m['group_name']}@{m['train_frac']}", "abelian": m["abelian"],
                "full_model_test_acc": m["final_test_acc"],
                "restricted_energy_frac_at_memorization": m["RESTRICTED_energy_frac_at_memorization"],
                "restricted_energy_frac_final": m["RESTRICTED_energy_frac_final"],
                "restricted_only_test_acc": m["RESTRICTED_ONLY_test_acc_final"],
                "restricted_only_test_acc_all_irreps": m["RESTRICTED_ONLY_test_acc_all_irreps_final"],
                "excluded_only_test_acc": m["excluded_only_test_acc_final"],
                "gap_full_minus_restricted": m["full_minus_restricted_test_acc"],
                "restricted_train_minus_test_ce": m["restricted_train_minus_test_ce"]}
               for m in per_run]
    drowned = [c for c in circuit
               if c["restricted_only_test_acc"] >= 0.9 and c["full_model_test_acc"] < 0.5]
    ef_grok = [m["RESTRICTED_energy_frac_final"] for m in per_run if m["grokked_full"]]
    ef_cens = [m["RESTRICTED_energy_frac_final"] for m in per_run if not m["grokked_full"]]
    energy_summary = {
        "definition": "share of the centred logit tensor's squared energy lying in the "
                      "correct-algorithm subspace {phi(a b c^-1) on the key irreps}",
        "grokked_runs_mean": (None if not ef_grok else round(float(np.mean(ef_grok)), 6)),
        "not_grokked_runs_mean": (None if not ef_cens else round(float(np.mean(ef_cens)), 6)),
        "grokked_min": (None if not ef_grok else round(float(min(ef_grok)), 6)),
        "not_grokked_max": (None if not ef_cens else round(float(max(ef_cens)), 6)),
        "separates_grokked_from_censored": bool(ef_grok and ef_cens
                                                and min(ef_grok) > max(ef_cens)),
    }

    grokked_runs = [m for m in per_run if m["grokked_full"]]
    lead_summary = {}
    for name in ("restricted_loss", "excluded_loss", "restricted_energy_frac",
                 "KL_irrep", "H_irrep", "H_svd"):
        rows = [(m["group_name"], m["abelian"], m["train_frac"],
                 m["measures"][name]["lead_vs_test50_steps"])
                for m in per_run if m["measures"][name]["lead_vs_test50_steps"] is not None]
        ab_leads = [v for _, isab, _, v in rows if isab]
        na_leads = [v for _, isab, _, v in rows if not isab]
        lead_summary[name] = {
            "per_run": [{"group": g, "abelian": isab, "train_frac": f, "lead_steps": v}
                        for g, isab, f, v in rows],
            "abelian_mean_lead": (None if not ab_leads
                                  else round(float(np.mean(ab_leads)), 1)),
            "nonabelian_mean_lead": (None if not na_leads
                                     else round(float(np.mean(na_leads)), 1)),
            "leads_on_all_abelian": bool(ab_leads) and all(v > 0 for v in ab_leads),
            "leads_on_all_nonabelian": bool(na_leads) and all(v > 0 for v in na_leads),
        }

    # false-positive control: runs that memorised but never left the plateau
    fp = [{"group": m["group_name"], "abelian": m["abelian"], "train_frac": m["train_frac"],
           "final_test_acc": m["final_test_acc"], "chance": round(1.0 / m["order"], 4),
           "steps_run": m["steps_run"],
           "KL_irrep_rise_bits": round(m["KL_irrep_final"] - m["KL_irrep_init"], 4),
           "H_irrep_drop_bits": round(m["H_irrep_init"] - m["H_irrep_final"], 4),
           "top3_irrep_share_final": m["top3_irrep_share_final"],
           "top3_irrep_null_share": m["top3_irrep_null_share"]}
          for m in per_run if m["memorized"] and not m["left_plateau"]]

    metrics = {
        "n_runs": len(runs), "n_skipped": len(skipped), "skipped_runs": skipped,
        "group_verification": group_checks,
        "arch": {k: P[k] for k in ("d_model", "n_heads", "d_mlp", "n_ctx", "init_std_scale")},
        "optimizer": {"lr": P["lr"], "weight_decay": P["weight_decay"],
                      "betas": [P["beta1"], P["beta2"]], "full_batch": True},
        "per_run": per_run,
        "matched_order_comparisons": comparisons,
        "restricted_circuit_vs_full_model": circuit,
        "runs_with_correct_circuit_but_no_generalization": drowned,
        "restricted_energy_fraction_summary": energy_summary,
        "n_runs_with_correct_circuit_but_no_generalization": len(drowned),
        "progress_measure_lead_summary": lead_summary,
        "false_positive_control_memorized_never_grokked": fp,
        "n_grokked": len(grokked_runs),
        "grokked": [f"{m['group_name']}@{m['train_frac']}" for m in grokked_runs],
        "not_grokked": [f"{m['group_name']}@{m['train_frac']}" for m in per_run
                        if not m["grokked_full"]],
        "total_compute_seconds": round(sum(r["train_seconds"] for r in runs), 1),
        "wall_clock_seconds": round(time.time() - t0, 1),
    }

    # ------------------------------ headline ------------------------------
    hl = []
    for c in comparisons:
        na = ("CENSORED at step %d" % c["nonabelian_delay_censored_at_step"]
              if c["nonabelian_grok_delay"] is None else "%.0f" % c["nonabelian_grok_delay"])
        ab = "CENSORED" if c["abelian_grok_delay"] is None else "%.0f" % c["abelian_grok_delay"]
        hl.append(f"order {c['order']} frac {c['train_frac']} ({c['n_train']} train): "
                  f"{c['abelian_group']} delay {ab} vs {c['nonabelian_group']} delay {na}"
                  + (f" (ratio {c['delay_ratio_nonabelian_over_abelian']:.2f}x)"
                     if c["delay_ratio_nonabelian_over_abelian"] else ""))
    rl = lead_summary["restricted_loss"]
    hl.append("restricted-loss lead over test-acc-50%: abelian mean "
              f"{rl['abelian_mean_lead']}, non-abelian mean {rl['nonabelian_mean_lead']} steps")
    kl = lead_summary["KL_irrep"]
    hl.append("irrep-power-concentration (KL vs null) lead: abelian mean "
              f"{kl['abelian_mean_lead']}, non-abelian mean {kl['nonabelian_mean_lead']} steps")
    if drowned:
        hl.append(f"{len(drowned)}/{len(circuit)} run(s) have a correctly-SIGNED irrep circuit "
                  "(restricted-only test acc 1.0, a 1-bit statement) while the full model is "
                  "<0.50; the MAGNITUDE separates them: restricted logit-energy fraction "
                  f"{energy_summary['grokked_runs_mean']} on grokked runs vs "
                  f"{energy_summary['not_grokked_runs_mean']} on censored runs "
                  f"(clean separation: {energy_summary['separates_grokked_from_censored']})")
    metrics["headline"] = " | ".join(hl)

    # ------------------------------ chart ------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C_AB, C_NA = "#2b6cb0", "#c0392b"
    CM = {"KL_irrep": "#1a7f64", "H_irrep": "#7b4fa3", "H_svd": "#c9a227",
          "restricted_loss": "#2b7bba", "excluded_loss": "#b03a48",
          "restricted_energy_frac": "#e08a1e"}
    fig, axes = plt.subplots(3, 3, figsize=(18.5, 13.6))

    def acc_panel(ax, keys, title):
        any_plot = False
        for k, frac in keys:
            m = by_key.get((k, frac))
            if m is None:
                continue
            any_plot = True
            col = C_AB if m["abelian"] else C_NA
            S = [max(s, 1) for s in m["history"]["step"]]
            ax.plot(S, m["history"]["train_acc"], color=col, lw=1.3, ls="--", alpha=.75)
            ax.plot(S, m["history"]["test_acc"], color=col, lw=2.2,
                    label=f"{m['group_name']} test (final {m['final_test_acc']:.2f})")
            rh = m["replay_history"]
            ax.plot([max(s, 1) for s in rh["step"]], rh["restricted_energy_frac"],
                    color=col, lw=1.6, ls=(0, (1, 1.2)), alpha=.95,
                    label=f"{m['group_name']} restricted energy frac "
                          f"({m['RESTRICTED_energy_frac_final']:.2f})")
            if m["step_memorized"]:
                ax.axvline(max(m["step_memorized"], 1), color=col, ls=":", lw=1, alpha=.7)
            t50 = m["step_test_acc_thresholds"].get("0.5")
            if t50:
                ax.axvline(t50, color=col, ls="-.", lw=1.1, alpha=.9)
        if not any_plot:
            ax.text(.5, .5, "not run (budget)", ha="center", va="center", fontsize=11)
        ax.set_xscale("log")
        ax.set_ylim(-.03, 1.05)
        ax.set_xlabel("optimizer step (full batch)")
        ax.set_ylabel("accuracy")
        ax.legend(frameon=False, fontsize=7.2, loc="center left")
        ax.set_title(title + "\nthin dashed = train acc, thick = test acc,"
                             "\nfine dotted = share of logit energy in the correct-algorithm"
                             " subspace", fontsize=9, pad=7)
        ax.spines[["top", "right"]].set_visible(False)

    acc_panel(axes[0][0], [("Z48", 0.5), ("D24", 0.5)],
              "ORDER 48, frac 0.5: Z/48 vs D_24")
    acc_panel(axes[0][1], [("Z24", 0.6), ("S4", 0.6)],
              "ORDER 24, frac 0.6: Z/24 vs S_4")

    # grok delay vs train fraction
    ax = axes[0][2]
    for gk, col, mk in [("Z48", C_AB, "o"), ("D24", C_NA, "s"),
                        ("Z24", "#4a90d9", "^"), ("S4", "#e07b39", "v")]:
        ms = sorted([m for m in per_run if m["group"] == gk], key=lambda z: z["train_frac"])
        if not ms:
            continue
        xs = [m["train_frac"] for m in ms]
        ys = [m["grok_delay_steps"] if m["grok_delay_steps"] else np.nan for m in ms]
        ax.plot(xs, ys, mk + "-", color=col, lw=1.8, ms=7, label=ms[0]["group_name"])
        for m in ms:
            if m["grok_delay_steps"] is None:
                ax.plot([m["train_frac"]], [m["steps_run"]], "x", color=col, ms=11, mew=2.4)
                ax.annotate("censored", (m["train_frac"], m["steps_run"]),
                            textcoords="offset points", xytext=(4, 5),
                            fontsize=7.5, color=col)
    ax.set_yscale("log")
    ax.set_xlabel("train fraction of the multiplication table")
    ax.set_ylabel("grok delay (steps, memorization -> test acc 50%)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Grok delay vs data\n(x = never reached 50% inside the box, plotted at the cap)",
                 fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)

    # irrep power spectra: init vs final vs null
    spec_targets = [("Z48", 0.5), ("D24", 0.5), ("S4", 0.6)]
    for j, (gk, frac) in enumerate(spec_targets):
        ax = axes[1][j]
        m = by_key.get((gk, frac))
        if m is None:
            ax.text(.5, .5, f"{gk}@{frac} not run", ha="center", va="center")
            ax.set_axis_off()
            continue
        x = np.arange(m["n_irreps"])
        ax.bar(x, m["init_irrep_power"], color="0.78", label="init")
        ax.bar(x, m["final_irrep_power"], color=(C_AB if m["abelian"] else C_NA),
               alpha=.85, label="final")
        ax.plot(x, m["null_irrep_power"], "k_", ms=6, mew=1.4,
                label="random-embedding null $d_\\rho^2/|G|$")
        ax.set_xlabel("irreducible representation")
        ax.set_ylabel("share of $W_E$ isotypic power")
        ax.legend(frameon=False, fontsize=7.5)
        if m["n_irreps"] <= 16:
            ax.set_xticks(x)
            ax.set_xticklabels(m["irrep_labels"], rotation=60, fontsize=6, ha="right")
        ax.set_title(f"{m['group_name']} @ frac {frac}: irrep power concentration\n"
                     f"KL vs null {m['KL_irrep_init']:.2f} -> {m['KL_irrep_final']:.2f} bits, "
                     f"eff. #irreps {m['effective_n_irreps_init']:.1f} -> "
                     f"{m['effective_n_irreps_final']:.1f}"
                     + ("" if m["grokked_full"] else "  [NOT GROKKED]"), fontsize=9.5)
        ax.spines[["top", "right"]].set_visible(False)

    # progress measures for the two headline arms + lead bar chart
    def prog_panel(ax, gk, frac):
        m = by_key.get((gk, frac))
        idx = next((i for i, r in enumerate(runs)
                    if r["group"] == gk and r["train_frac"] == frac), None)
        if m is None or idx is None:
            ax.text(.5, .5, f"{gk}@{frac} not run", ha="center", va="center")
            ax.set_axis_off()
            return
        a = analyses[idx]
        S = [max(s, 1) for s in m["history"]["step"]]
        ax.plot(S, m["history"]["test_acc"], color="#c95d3c", lw=2.4, label="test acc", zorder=5)
        for name, mm in a["measures"].items():
            if "_frac_series" not in mm:
                continue
            lab = (f"{name} ({mm['lead_vs_test50_steps']:+} st)"
                   if mm["lead_vs_test50_steps"] is not None else name)
            ax.plot([max(s, 1) for s in mm["_frac_steps"]],
                    np.clip(mm["_frac_series"], -.05, 1.05),
                    color=CM[name], lw=1.5, alpha=.9, label=lab)
            if mm["t_50pct"]:
                ax.plot([max(mm["t_50pct"], 1)], [.5], "o", color=CM[name], ms=5, zorder=6)
        ax.axhline(.5, color="0.8", ls=":", lw=1)
        if a["t_mem"]:
            ax.axvline(max(a["t_mem"], 1), color="#3d5a80", ls="--", lw=1)
        if a["t_test50"]:
            ax.axvline(a["t_test50"], color="#c95d3c", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_ylim(-.06, 1.08)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("fraction of post-memorization movement")
        ax.legend(frameon=False, fontsize=7, loc="upper left")
        ax.set_title(f"{m['group_name']} @ frac {frac}"
                     f" ({'abelian' if m['abelian'] else 'NON-abelian'}): progress measures\n"
                     "dots = 50% crossing; + lead = fires BEFORE test acc", fontsize=9.5)
        ax.spines[["top", "right"]].set_visible(False)

    prog_panel(axes[2][0], "Z48", 0.5)
    prog_panel(axes[2][1], "D24", 0.5)

    ax = axes[2][2]
    names = ["restricted_loss", "excluded_loss", "restricted_energy_frac",
             "KL_irrep", "H_svd"]
    labs = [f"{m['group_name']}@{m['train_frac']}" for m in per_run
            if m["measures"]["restricted_loss"]["lead_vs_test50_steps"] is not None]
    sel = [m for m in per_run
           if m["measures"]["restricted_loss"]["lead_vs_test50_steps"] is not None]
    if sel:
        w = 0.8 / len(names)
        xs = np.arange(len(sel))
        for i, nm in enumerate(names):
            vals = [m["measures"][nm]["lead_vs_test50_steps"] or 0 for m in sel]
            ax.bar(xs + i * w - 0.4 + w / 2, vals, w, color=CM[nm], label=nm)
        ax.axhline(0, color="k", lw=1)
        ax.set_xticks(xs)
        ax.set_xticklabels(labs, rotation=20, fontsize=8)
        ax.set_ylabel("lead over test acc 50% (steps; + = fires first)")
        ax.legend(frameon=False, fontsize=7.5)
    else:
        ax.text(.5, .5, "no run reached test acc 50%", ha="center", va="center")
    ax.set_title("Does the leading indicator fire the same way\nacross the abelian / non-abelian divide?",
                 fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Grokking beyond addition: abelian vs non-abelian group multiplication - "
                 f"1-layer transformer, no LayerNorm, d={P['d_model']}, d_mlp={P['d_mlp']}, "
                 f"full-batch AdamW (lr {P['lr']}, wd {P['weight_decay']}), seed {seed}",
                 fontsize=12.5, y=1.003)
    fig.tight_layout(h_pad=2.4, w_pad=2.2)
    fig.savefig(HERE / "chart.png", dpi=140, bbox_inches="tight")

    results = {"id": cfg.get("id", "unknown"), "git_commit": git_sha(), "seed": seed,
               "duration_sec": round(time.time() - t0, 2), "metrics": metrics,
               "env": env_info(), "status": "done"}
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    log("headline: " + metrics["headline"])
    log(json.dumps({k: results[k] for k in ("id", "duration_sec", "status")}, indent=2))


if __name__ == "__main__":
    main()
