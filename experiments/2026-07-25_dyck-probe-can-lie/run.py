"""The probe can lie: probe accuracy vs causal effect per stack-feature in a tiny Dyck-2 transformer.

Deterministic, CPU-only, single-threaded. Writes results.json + chart.png.

Pipeline
  1. generate seeded Dyck-2 bracket sequences (2 types, max depth 10, 10-20 pairs)
  2. train a 2-layer / 1-head / d_model=32 decoder on next-token prediction
  3. for 7 candidate state features x 2 residual-stream sites, fit linear probes
     (+ a shuffled-label probe control)
  4. ERASE each feature from the residual stream with a LEACE eraser (closed-form perfect
     linear concept erasure, Belrose et al. 2023) - which guarantees that no linear probe can
     beat the majority baseline afterwards, and is verified by refitting a probe - then measure
     the behavioural damage against RANDOM erasers of identical form and rank
  5. deliverable: per-feature (probe accuracy, causal effect) scatter

Why LEACE rather than plain directional ablation: a null causal effect only means something if
the feature really is gone. LEACE certifies that (rank <= n_classes-1, and because it operates in
whitened space a random rank-matched eraser is automatically variance-matched).

Usage:  python run.py
"""
import json, math, os, random, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- vocab
PAD, BOS, EOS = 0, 1, 2
OPEN = [3, 5]        # '(', '['
CLOSE = [4, 6]       # ')', ']'
VOCAB = 7


def set_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
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
            info[mod] = getattr(__import__(mod), "__version__", "?")
        except Exception:
            pass
    return info


# ---------------------------------------------------------------- data
def gen_seq(rng, min_pairs, max_pairs, max_depth, p_open):
    """One Dyck-2 string as a token list (no BOS/EOS)."""
    n = int(rng.integers(min_pairs, max_pairs + 1))
    total = 2 * n
    toks, stack, opens_used = [], [], 0
    for i in range(total):
        remaining = total - i
        can_open = opens_used < n and len(stack) < max_depth
        can_close = len(stack) > 0
        if not can_close:
            do_open = True
        elif not can_open or remaining <= len(stack):
            do_open = False
        else:
            do_open = rng.random() < p_open
        if do_open:
            t = int(rng.integers(0, 2))
            toks.append(OPEN[t]); stack.append(t); opens_used += 1
        else:
            t = stack.pop()
            toks.append(CLOSE[t])
    return toks


def build_tensors(seqs, T, max_depth):
    """Pack sequences into padded tensors + per-position ground-truth stack state."""
    N = len(seqs)
    idx = np.zeros((N, T), dtype=np.int64)
    depth = np.zeros((N, T), dtype=np.int64)
    top_t = np.full((N, T), -1, dtype=np.int64)
    sec_t = np.full((N, T), -1, dtype=np.int64)
    top_pos = np.full((N, T), -1, dtype=np.int64)
    cur_open = np.full((N, T), -1, dtype=np.int64)
    valid = np.zeros((N, T), dtype=bool)          # positions that make a prediction

    for i, s in enumerate(seqs):
        full = [BOS] + s + [EOS]
        L = len(full)
        idx[i, :L] = full
        stack, stack_pos = [], []
        for t, tok in enumerate(full):
            if tok in OPEN:
                bt = OPEN.index(tok); stack.append(bt); stack_pos.append(t); cur_open[i, t] = 1
            elif tok in CLOSE:
                stack.pop(); stack_pos.pop(); cur_open[i, t] = 0
            depth[i, t] = len(stack)
            if stack:
                top_t[i, t] = stack[-1]; top_pos[i, t] = stack_pos[-1]
            if len(stack) >= 2:
                sec_t[i, t] = stack[-2]
        valid[i, :L - 1] = True                   # predict tokens 1..L-1
    return dict(idx=idx, depth=depth, top_t=top_t, sec_t=sec_t,
                top_pos=top_pos, cur_open=cur_open, valid=valid)


def make_features(D, max_depth):
    """feature name -> (labels, valid mask, n_classes)."""
    depth, valid = D["depth"], D["valid"]
    N, T = D["idx"].shape
    pos = np.arange(T)[None, :].repeat(N, 0)
    recency = np.where(D["top_pos"] >= 0, pos - D["top_pos"], -1)
    return {
        "top_type":        (D["top_t"],                        valid & (depth > 0),  2),
        "depth_gt0":       ((depth > 0).astype(np.int64),      valid,                2),
        "depth_exact":     (np.clip(depth, 0, max_depth),      valid,   max_depth + 1),
        "depth_parity":    (depth % 2,                         valid,                2),
        "second_type":     (D["sec_t"],                        valid & (depth >= 2), 2),
        # restricted to CLOSER positions so it cannot be a proxy for cur_tok_is_open
        # (after an opener the top of stack is the current token, i.e. recency is always 0)
        "top_recency":     ((recency > 3).astype(np.int64),
                            valid & (depth > 0) & (D["cur_open"] == 0), 2),
        "cur_tok_is_open": (D["cur_open"],                     valid & (D["cur_open"] >= 0), 2),
    }


FEATURE_ORDER = ["top_type", "depth_gt0", "depth_exact", "depth_parity",
                 "second_type", "top_recency", "cur_tok_is_open"]

# Does the next-token decision AT THIS POSITION provably need the variable?
#   top_type        yes - picks ')' vs ']'
#   depth_gt0       yes - decides whether a closer is legal at all, and whether EOS is legal
#   cur_tok_is_open yes - the conditional distribution over open/close/EOS differs after an
#                         opener vs a closer, and it is what sets the new top-of-stack
#   depth_exact     no  - only the depth==0 predicate matters, never the magnitude
#   depth_parity    no  - and it is FREE: in any Dyck prefix depth(t) = t (mod 2)
#   second_type     no  - needed several tokens later, never for the immediate next token
#   top_recency     no  - how far back the matching opener sits does not change the answer
EXPECTED_CAUSAL = {"top_type": True, "depth_gt0": True, "depth_exact": False,
                   "depth_parity": False, "second_type": False,
                   "top_recency": False, "cur_tok_is_open": True}


# ---------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d_ff)
        self.fc2 = nn.Linear(d_ff, d)
        self.d = d

    def forward(self, x, attn_bias):
        h = self.ln1(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        att = (q @ k.transpose(1, 2)) / math.sqrt(self.d) + attn_bias
        x = x + self.o(torch.softmax(att, dim=-1) @ v)
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


def apply_eraser(x, er):
    """er = (A, mu): the affine map  x -> mu + (x - mu) A^T."""
    A, mu = er
    return (x - mu) @ A.T + mu


class TinyTF(nn.Module):
    def __init__(self, d, d_ff, n_layers, T):
        super().__init__()
        assert n_layers == 2
        self.tok = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(T, d)
        self.blocks = nn.ModuleList([Block(d, d_ff) for _ in range(n_layers)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB, bias=False)

    def forward(self, idx, er=None, er_site=None, want_acts=False):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T))
        causal = torch.tril(torch.ones(T, T)) == 0
        keypad = (idx == PAD)[:, None, :]
        bias = torch.zeros(B, T, T)
        bias.masked_fill_(causal[None], float("-inf"))
        bias = bias.masked_fill(keypad, float("-inf"))
        # column 0 is BOS (never PAD) and visible to every causal row, so no row is all -inf
        acts = {}
        x = self.blocks[0](x, bias)
        if er_site == "resid_mid":
            x = apply_eraser(x, er)
        if want_acts:
            acts["resid_mid"] = x.detach().clone()
        x = self.blocks[1](x, bias)
        if er_site == "resid_final":
            x = apply_eraser(x, er)
        if want_acts:
            acts["resid_final"] = x.detach().clone()
        return self.head(self.lnf(x)), acts


# ---------------------------------------------------------------- behavioural eval
class EvalSet:
    def __init__(self, D):
        self.idx = torch.from_numpy(D["idx"])
        tgt = torch.full_like(self.idx, PAD)
        tgt[:, :-1] = self.idx[:, 1:]
        self.tgt = tgt
        self.valid = torch.from_numpy(D["valid"])
        depth = torch.from_numpy(D["depth"])
        self.is_close = self.valid & ((tgt == CLOSE[0]) | (tgt == CLOSE[1]))
        self.close_correct = torch.where(tgt == CLOSE[0], CLOSE[0], CLOSE[1])
        self.close_other = torch.where(tgt == CLOSE[0], CLOSE[1], CLOSE[0])
        self.d0 = self.valid & (depth == 0)


@torch.no_grad()
def behav_eval(model, ev, er=None, site=None):
    logits, _ = model(ev.idx, er=er, er_site=site)
    v = ev.valid
    ce = F.cross_entropy(logits[v], ev.tgt[v]).item()
    acc = (logits[v].argmax(-1) == ev.tgt[v]).float().mean().item()
    lc = logits[ev.is_close]
    ctype = (lc.gather(-1, ev.close_correct[ev.is_close][:, None]).squeeze(-1) >
             lc.gather(-1, ev.close_other[ev.is_close][:, None]).squeeze(-1)).float().mean().item()
    p0 = torch.softmax(logits[ev.d0], dim=-1)
    illegal = p0[:, CLOSE].sum(-1).mean().item()
    # decision-relevant CE slices: depth-0 positions (where depth>0 matters) and
    # positions whose target is a closer (where top-of-stack type matters)
    ce_d0 = F.cross_entropy(logits[ev.d0], ev.tgt[ev.d0]).item()
    ce_close = F.cross_entropy(logits[ev.is_close], ev.tgt[ev.is_close]).item()
    return {"ce": ce, "tok_acc": acc, "close_type_acc": ctype, "illegal_close_mass": illegal,
            "ce_d0": ce_d0, "ce_close": ce_close}


# ---------------------------------------------------------------- probes
def train_probe(Xtr, ytr, Xte, yte, n_classes, steps, lr, wd, seed):
    g = torch.Generator().manual_seed(int(seed) % (2 ** 31))
    W = torch.zeros(Xtr.shape[1], n_classes)
    with torch.no_grad():
        W.normal_(0, 0.02, generator=g)
    W.requires_grad_(True)
    b = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(Xtr @ W + b, ytr).backward()
        opt.step()
    with torch.no_grad():
        pred = (Xte @ W + b).argmax(-1)
        return acc_of(pred, yte), pred


def acc_of(pred, y):
    return (pred == y).float().mean().item()


# ---------------------------------------------------------------- LEACE erasure
def sym_powers(S, eps):
    ev, U = torch.linalg.eigh(S)
    ev = ev.clamp_min(eps)
    return U @ torch.diag(ev.sqrt()) @ U.T, U @ torch.diag(ev.rsqrt()) @ U.T


def cramers_v(a, b, ka, kb):
    """Association between two categorical label vectors (0 = independent, 1 = deterministic)."""
    tab = np.zeros((ka, kb))
    np.add.at(tab, (a, b), 1.0)
    n = tab.sum()
    if n == 0:
        return 0.0
    exp = tab.sum(1, keepdims=True) @ tab.sum(0, keepdims=True) / n
    chi2 = float(((tab - exp) ** 2 / np.maximum(exp, 1e-9)).sum())
    denom = n * (min(ka, kb) - 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else 0.0


def leace(X, y, n_classes, eps_frac=1e-4, tol=1e-6):
    """Closed-form perfect linear concept erasure (Belrose et al. 2023).

    Returns (A, mu, rank): the affine eraser  h -> mu + (h - mu) A^T  plus the whitening
    factors so a rank-matched RANDOM eraser of identical form can be built.
    """
    N, d = X.shape
    mu = X.mean(0)
    Xc = X - mu
    S = Xc.T @ Xc / N
    eps = eps_frac * float(S.diagonal().sum()) / d
    S_half, S_ihalf = sym_powers(S, eps)
    Z = F.one_hot(y, n_classes).float()
    Zc = Z - Z.mean(0)
    Sxz = Xc.T @ Zc / N                      # (d, k)
    M = S_ihalf @ Sxz                        # whitened cross-covariance
    U, sv, _ = torch.linalg.svd(M, full_matrices=False)
    r = int((sv > tol * max(float(sv[0]), 1e-12)).sum())
    r = max(r, 1)
    Uk = U[:, :r]
    A = S_half @ (torch.eye(d) - Uk @ Uk.T) @ S_ihalf
    return A, mu, r, (S_half, S_ihalf, S), Uk


def random_eraser(whiten, mu, r, gen):
    S_half, S_ihalf, _ = whiten
    d = S_half.shape[0]
    Q = torch.linalg.qr(torch.randn(d, r, generator=gen))[0]
    return (S_half @ (torch.eye(d) - Q @ Q.T) @ S_ihalf, mu)


def var_removed(A, S):
    tot = float(S.diagonal().sum())
    return 1.0 - float((A @ S @ A.T).diagonal().sum()) / tot


# ---------------------------------------------------------------- one full run
def run_one(C, seed):
    set_seeds(seed)
    t0 = time.time()
    T, d = C["max_positions"], C["d_model"]
    rng = np.random.default_rng(seed)

    # ---- data ------------------------------------------------------
    gargs = (C["min_pairs"], C["max_pairs"], C["max_depth"], C["p_open"])
    train_seqs = [gen_seq(rng, *gargs) for _ in range(C["n_train_seqs"])]
    eval_seqs = [gen_seq(rng, *gargs) for _ in range(C["n_eval_seqs"])]
    probe_seqs = [gen_seq(rng, *gargs) for _ in range(C["n_probe_seqs"])]
    Dtr = build_tensors(train_seqs, T, C["max_depth"])
    Dev = build_tensors(eval_seqs, T, C["max_depth"])
    Dpr = build_tensors(probe_seqs, T, C["max_depth"])
    ev = EvalSet(Dev)
    tr_idx = torch.from_numpy(Dtr["idx"])
    tr_tgt = torch.full_like(tr_idx, PAD); tr_tgt[:, :-1] = tr_idx[:, 1:]
    print(f"[data] {len(train_seqs)} train seqs, mean depth "
          f"{Dtr['depth'][Dtr['valid']].mean():.2f}, max len "
          f"{max(len(s) for s in train_seqs) + 2}  ({time.time()-t0:.1f}s)", flush=True)

    # ---- train -----------------------------------------------------
    model = TinyTF(d, C["d_ff"], C["n_layers"], T)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=C["lr"], weight_decay=C["weight_decay"])
    gen = torch.Generator().manual_seed(seed + 1)
    curve, steps_done = [], 0
    t_train = time.time()
    for step in range(C["steps"]):
        for gp in opt.param_groups:
            gp["lr"] = C["lr"] * min(1.0, (step + 1) / C["warmup"])
        b = torch.randint(0, len(train_seqs), (C["batch_size"],), generator=gen)
        x, y = tr_idx[b], tr_tgt[b]
        logits, _ = model(x)
        m = y != PAD
        F.cross_entropy(logits[m], y[m]).backward()
        opt.step(); opt.zero_grad()
        steps_done = step + 1
        if (step + 1) % C["eval_every"] == 0 or step == 0:
            model.eval(); mm = behav_eval(model, ev); model.train()
            curve.append({"step": step + 1, **{k: round(v, 5) for k, v in mm.items()}})
            print(f"  step {step+1:5d} ce {mm['ce']:.4f} close_type {mm['close_type_acc']:.4f} "
                  f"illegal {mm['illegal_close_mass']:.4f} ({time.time()-t_train:.0f}s)", flush=True)
        if time.time() - t_train > C["time_cap_s_train"]:
            print(f"  [time cap] stopping at step {steps_done}", flush=True)
            break
    model.eval()
    clean = behav_eval(model, ev)
    train_sec = time.time() - t_train
    print(f"[train] {steps_done} steps, {train_sec:.0f}s, params {n_params}, clean {clean}", flush=True)

    # ---- collect activations --------------------------------------
    t_act = time.time()
    pr_idx = torch.from_numpy(Dpr["idx"])
    acts = {s: [] for s in C["sites"]}
    with torch.no_grad():
        for i in range(0, len(pr_idx), 256):
            _, a = model(pr_idx[i:i + 256], want_acts=True)
            for s in C["sites"]:
                acts[s].append(a[s])
    acts = {s: torch.cat(v, 0) for s, v in acts.items()}          # (N, T, d)
    feats = make_features(Dpr, C["max_depth"])
    N = len(pr_idx)
    seq_is_test = np.zeros(N, dtype=bool); seq_is_test[N // 2:] = True   # split BY SEQUENCE
    valid_flat = torch.from_numpy(Dpr["valid"].reshape(-1))
    scale = {}
    for s in C["sites"]:
        flat = acts[s].reshape(-1, d)[valid_flat]
        scale[s] = (flat.mean(0), flat.std().item())   # centre + SCALAR scale (probe input only)
    print(f"[acts] collected ({time.time()-t_act:.1f}s)", flush=True)

    def subset(site, fname, sub_rng):
        """Raw + standardised activations and labels for one feature at one site."""
        y_all, mask_all, ncls = feats[fname]
        y_t = torch.from_numpy(y_all)
        out = {"n_classes": ncls}
        for tag, mk, cap in (("tr", mask_all & (~seq_is_test)[:, None], C["probe_train_n"]),
                             ("te", mask_all & seq_is_test[:, None], C["probe_test_n"])):
            mt = torch.from_numpy(mk)
            R, y = acts[site][mt], y_t[mt]
            if len(R) > cap:
                sel = torch.from_numpy(sub_rng.choice(len(R), cap, replace=False))
                R, y = R[sel], y[sel]
            out["R" + tag] = R.contiguous()
            out["y" + tag] = y.contiguous()
        return out

    # ---- probes + causal ------------------------------------------
    t_probe = time.time()
    rand_cache = {}
    per_feature = []
    erased_dirs = {}
    for site in C["sites"]:
        amu, asc = scale[site]
        std = lambda R: (R - amu) / asc                                   # noqa: E731
        for fi, fname in enumerate(FEATURE_ORDER):
            sub_rng = np.random.default_rng(seed + 100 * fi + (1 if site == "resid_final" else 0))
            S = subset(site, fname, sub_rng)
            Rtr, ytr, Rte, yte, ncls = S["Rtr"], S["ytr"], S["Rte"], S["yte"], S["n_classes"]
            Xtr, Xte = std(Rtr), std(Rte)
            maj = float(torch.bincount(yte, minlength=ncls).max()) / len(yte)

            # (i) probe
            probe_acc, pred0 = train_probe(Xtr, ytr, Xte, yte, ncls, C["probe_steps"],
                                           C["probe_lr"], C["probe_wd"], seed + 7 * fi)
            pear = float(np.corrcoef(pred0.numpy(), yte.numpy())[0, 1]) if ncls > 2 else None

            # (ii) shuffled-label probe control
            perm = torch.from_numpy(sub_rng.permutation(len(ytr)))
            perm_te = torch.from_numpy(sub_rng.permutation(len(yte)))
            shuf_acc, _ = train_probe(Xtr, ytr[perm], Xte, yte[perm_te], ncls, C["probe_steps"],
                                      C["probe_lr"], C["probe_wd"], seed + 7 * fi + 3)

            # (iii) LEACE eraser fitted on RAW activations at this site, + guardedness check
            A_e, mu_e, rank, whiten, Uk = leace(Rtr, ytr, ncls, C["leace_eps_frac"])
            er = (A_e, mu_e)
            erased_dirs[(site, fname)] = Uk
            guard_acc, _ = train_probe(std(apply_eraser(Rtr, er)), ytr,
                                       std(apply_eraser(Rte, er)), yte, ncls, C["probe_steps"],
                                       C["probe_lr"], C["probe_wd"], seed + 7 * fi + 11)

            # (iv) shuffled-label eraser control (same form and fit procedure, no real signal)
            A_sh, mu_sh, _, _, _ = leace(Rtr, ytr[perm], ncls, C["leace_eps_frac"])
            er_shuf = (A_sh, mu_sh)

            # (v) behaviour under each eraser
            m_real = behav_eval(model, ev, er=er, site=site)
            m_shuf = behav_eval(model, ev, er=er_shuf, site=site)
            key = (site, rank)
            if key not in rand_cache:
                g2 = torch.Generator().manual_seed(seed + 999 + rank)
                runs = []
                for _ in range(C["n_random_controls"]):
                    er_r = random_eraser(whiten, mu_e, rank, g2)
                    runs.append((behav_eval(model, ev, er=er_r, site=site),
                                 var_removed(er_r[0], whiten[2])))
                rand_cache[key] = runs
            rnd = [r[0] for r in rand_cache[key]]
            rnd_var = float(np.mean([r[1] for r in rand_cache[key]]))

            rec = {
                "site": site, "feature": fname, "n_classes": ncls,
                "majority_baseline": round(maj, 4),
                "probe_acc": round(probe_acc, 4),
                "probe_selectivity": round(probe_acc - maj, 4),
                "shuffled_probe_acc": round(shuf_acc, 4),
                "probe_pearson_r": None if pear is None else round(pear, 4),
                "erasure_rank": rank,
                "refit_probe_acc_after_erasure": round(guard_acc, 4),
                "linearly_guarded": bool(guard_acc <= maj + C["guard_tol"]),
                "var_frac_removed": round(var_removed(A_e, whiten[2]), 4),
                "var_frac_removed_random": round(rnd_var, 4),
                "delta_ce": round(m_real["ce"] - clean["ce"], 4),
                "delta_close_type_acc": round(m_real["close_type_acc"] - clean["close_type_acc"], 4),
                "delta_illegal_close_mass": round(
                    m_real["illegal_close_mass"] - clean["illegal_close_mass"], 5),
                "rand_delta_close_mean": round(float(np.mean([x["close_type_acc"] for x in rnd]))
                                               - clean["close_type_acc"], 4),
                "rand_delta_close_std": round(float(np.std([x["close_type_acc"] for x in rnd])), 4),
                "rand_delta_illegal_mean": round(float(np.mean([x["illegal_close_mass"] for x in rnd]))
                                                 - clean["illegal_close_mass"], 5),
                "shuf_delta_ce": round(m_shuf["ce"] - clean["ce"], 4),
                "shuf_delta_close": round(m_shuf["close_type_acc"] - clean["close_type_acc"], 4),
                "expected_causal": EXPECTED_CAUSAL[fname],
            }
            # cross-entropy excess on ALL positions and on the two decision-relevant slices:
            #   ce_d0    = positions at depth 0 (the only place depth>0 changes the answer)
            #   ce_close = positions whose target is a closer (where top-of-stack type decides)
            for sk in ("ce", "ce_d0", "ce_close"):
                rm = float(np.mean([x[sk] for x in rnd])) - clean[sk]
                rs = max(float(np.std([x[sk] for x in rnd])), 1e-4)
                rec[f"rand_delta_{sk}_mean"] = round(rm, 4)
                rec[f"rand_delta_{sk}_std"] = round(rs, 4)
                rec[f"causal_{sk}_excess"] = round((m_real[sk] - clean[sk]) - rm, 4)
                rec[f"causal_{sk}_z"] = round(rec[f"causal_{sk}_excess"] / rs, 2)
            best = max(("ce", "ce_d0", "ce_close"), key=lambda s: rec[f"causal_{s}_excess"])
            rec["causal_effect_slice"] = best
            rec["causal_effect_nats"] = rec[f"causal_{best}_excess"]
            rec["causal_effect_z"] = rec[f"causal_{best}_z"]
            rec["causal_close_excess"] = round(
                rec["rand_delta_close_mean"] - rec["delta_close_type_acc"], 4)
            rec["causal_illegal_excess"] = round(
                rec["delta_illegal_close_mass"] - rec["rand_delta_illegal_mean"], 5)
            rec["shuf_ce_excess"] = round(rec["shuf_delta_ce"] - rec["rand_delta_ce_mean"], 4)
            # verdict: erasing must cost more than an absolute floor AND stand clear of the
            # spread of the rank-matched random erasers, on at least one behavioural slice
            rec["causally_used"] = bool(
                rec["causal_effect_nats"] >= C["causal_ce_threshold"]
                and rec["causal_effect_z"] >= C["causal_z_threshold"])
            rec["decodable"] = bool(rec["probe_selectivity"] >= C["decodable_threshold"])
            per_feature.append(rec)
            print(f"  [{site:11s}] {fname:15s} probe {probe_acc:.3f} (maj {maj:.3f} shuf {shuf_acc:.3f})"
                  f" r={rank} guard {guard_acc:.3f} | excess nats all {rec['causal_ce_excess']:+.3f}"
                  f" d0 {rec['causal_ce_d0_excess']:+.3f} close {rec['causal_ce_close_excess']:+.3f}"
                  f" -> {rec['causal_effect_nats']:+.3f} ({rec['causal_effect_slice']}, z="
                  f"{rec['causal_effect_z']:+.1f}) | dCloseAcc {rec['delta_close_type_acc']:+.3f}"
                  f" | used={rec['causally_used']}", flush=True)
    print(f"[probe+causal] {time.time()-t_probe:.0f}s", flush=True)

    # ---- entanglement: are the "unused" features even separable from the used ones? ----
    # (i) ground-truth label association between features, on positions where both are defined
    label_assoc = {}
    for i, fa in enumerate(FEATURE_ORDER):
        ya, ma, ka = feats[fa]
        for fb in FEATURE_ORDER[i + 1:]:
            yb, mb, kb = feats[fb]
            both = ma & mb
            label_assoc[f"{fa}|{fb}"] = round(
                cramers_v(ya[both].astype(int), yb[both].astype(int), ka, kb), 3)
    # (ii) geometric overlap of the erased subspaces at resid_final (cos of smallest princ. angle)
    dir_overlap = {}
    for i, fa in enumerate(FEATURE_ORDER):
        Ua = erased_dirs.get(("resid_final", fa))
        for fb in FEATURE_ORDER[i + 1:]:
            Ub = erased_dirs.get(("resid_final", fb))
            if Ua is None or Ub is None:
                continue
            dir_overlap[f"{fa}|{fb}"] = round(
                float(torch.linalg.svdvals(Ua.T @ Ub)[0].clamp(0, 1)), 3)

    return {
        "seed": seed,
        "n_params": n_params,
        "steps_trained": steps_done,
        "train_sec": round(train_sec, 1),
        "clean": {k: round(v, 5) for k, v in clean.items()},
        "mean_depth": round(float(Dtr["depth"][Dtr["valid"]].mean()), 3),
        "n_eval_positions": int(ev.valid.sum()),
        "label_association_cramers_v": label_assoc,
        "erased_subspace_overlap_resid_final": dir_overlap,
        "training_curve": curve,
        "per_feature": per_feature,
        "wall_sec": round(time.time() - t0, 1),
    }


# ---------------------------------------------------------------- main
AGG_FIELDS = ["probe_acc", "probe_selectivity", "shuffled_probe_acc", "majority_baseline",
              "refit_probe_acc_after_erasure", "var_frac_removed", "var_frac_removed_random",
              "delta_ce", "rand_delta_ce_mean", "rand_delta_ce_std",
              "causal_effect_nats", "causal_effect_z",
              "causal_ce_excess", "causal_ce_z", "causal_ce_d0_excess", "causal_ce_d0_z",
              "causal_ce_close_excess", "causal_ce_close_z",
              "shuf_ce_excess", "causal_close_excess", "causal_illegal_excess",
              "delta_close_type_acc", "rand_delta_close_mean", "rand_delta_close_std",
              "shuf_delta_close"]


def main():
    cfg_all = load_config()
    C = cfg_all["params"]
    t0 = time.time()
    runs = []
    for s in C["seeds"]:
        print(f"\n================ seed {s} ================", flush=True)
        runs.append(run_one(C, int(s)))

    # ---- aggregate across seeds ------------------------------------
    per_feature = []
    for i, ref in enumerate(runs[0]["per_feature"]):
        rows = [r["per_feature"][i] for r in runs]
        assert all(x["feature"] == ref["feature"] and x["site"] == ref["site"] for x in rows)
        agg = {"site": ref["site"], "feature": ref["feature"], "n_classes": ref["n_classes"],
               "expected_causal": ref["expected_causal"],
               "erasure_rank": ref["erasure_rank"],
               "linearly_guarded_all_seeds": bool(all(x["linearly_guarded"] for x in rows)),
               "decodable_all_seeds": bool(all(x["decodable"] for x in rows)),
               "causally_used_all_seeds": bool(all(x["causally_used"] for x in rows)),
               "causally_used_any_seed": bool(any(x["causally_used"] for x in rows))}
        for f in AGG_FIELDS:
            vals = [x[f] for x in rows]
            agg[f] = round(float(np.mean(vals)), 4)
            agg[f + "_by_seed"] = vals
        per_feature.append(agg)

    fin = [r for r in per_feature if r["site"] == "resid_final"]
    decodable = [r for r in fin if r["decodable_all_seeds"]]
    # "the probe can lie": decodable in every seed, never causally used in any seed. The headline
    # goes to the strongest such case - erasure that costs STRICTLY LESS than a random eraser of
    # the same rank - ranked by probe selectivity; otherwise to the most selective probe.
    lying = sorted([r for r in decodable if not r["causally_used_any_seed"]],
                   key=lambda r: (max(r["causal_effect_nats_by_seed"]) < 0, r["probe_selectivity"]),
                   reverse=True)
    used = sorted([r for r in decodable if r["causally_used_all_seeds"]],
                  key=lambda r: -r["causal_effect_nats"])
    mism = [r["feature"] for r in fin if r["causally_used_all_seeds"] != r["expected_causal"]]

    metrics = {
        "n_seeds": len(runs),
        "seeds": [r["seed"] for r in runs],
        "n_params": runs[0]["n_params"],
        "steps_trained": [r["steps_trained"] for r in runs],
        "train_sec": [r["train_sec"] for r in runs],
        "mean_depth": runs[0]["mean_depth"],
        "n_eval_positions": runs[0]["n_eval_positions"],
        "clean_ce": round(float(np.mean([r["clean"]["ce"] for r in runs])), 4),
        "clean_close_type_acc": round(float(np.mean([r["clean"]["close_type_acc"] for r in runs])), 4),
        "clean_tok_acc": round(float(np.mean([r["clean"]["tok_acc"] for r in runs])), 4),
        "clean_illegal_close_mass": round(
            float(np.mean([r["clean"]["illegal_close_mass"] for r in runs])), 5),
        "clean_by_seed": [r["clean"] for r in runs],
        "headline_decodable_but_unused": lying[0]["feature"] if lying else None,
        "decodable_but_unused": [r["feature"] for r in lying],
        "decodable_and_used": [r["feature"] for r in used],
        "predictions_matched": bool(not mism),
        "mismatched_features": mism,
        "n_features_linearly_guarded_final_all_seeds":
            sum(1 for r in fin if r["linearly_guarded_all_seeds"]),
        "label_association_cramers_v": runs[0]["label_association_cramers_v"],
        "erased_subspace_overlap_resid_final": runs[0]["erased_subspace_overlap_resid_final"],
        "training_curve": runs[0]["training_curve"],
        "per_feature": per_feature,
        "per_seed_raw": [{k: v for k, v in r.items() if k != "training_curve"} for r in runs],
    }
    if lying:
        h = lying[0]
        ref = (f" - versus {used[0]['causal_effect_nats']:+.3f} nats for {used[0]['feature']}."
               if used else ".")
        metrics["headline"] = (
            f"{h['feature']} probes at {h['probe_acc']:.3f} (majority {h['majority_baseline']:.3f}, "
            f"shuffled-label control {h['shuffled_probe_acc']:.3f}) at resid_final and is "
            f"{'linearly guarded' if h['linearly_guarded_all_seeds'] else 'only partly erased'} "
            f"after rank-{h['erasure_rank']} LEACE erasure (refit probe "
            f"{h['refit_probe_acc_after_erasure']:.3f} vs majority {h['majority_baseline']:.3f}), "
            f"yet the best-case erasure damage over rank-matched random erasers is only "
            f"{h['causal_effect_nats']:+.3f} nats (z={h['causal_effect_z']:+.1f}) and "
            f"{h['causal_close_excess']:+.3f} close-type accuracy" + ref)
    else:
        metrics["headline"] = ("every decodable feature at resid_final also carried a causal "
                               "effect above threshold - no 'probe can lie' quadrant at this scale.")

    results = {
        "id": cfg_all.get("id", "unknown"),
        "git_commit": git_sha(),
        "seed": int(cfg_all.get("seed", 0)),
        "duration_sec": round(time.time() - t0, 2),
        "metrics": metrics,
        "config": C,
        "env": env_info(),
        "status": "done",
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    make_chart(metrics)
    print(json.dumps({k: v for k, v in metrics.items()
                      if k not in ("training_curve", "per_feature", "per_seed_raw",
                                   "label_association_cramers_v",
                                   "erased_subspace_overlap_resid_final")}, indent=2))
    print(f"[total] {time.time()-t0:.0f}s")


# ---------------------------------------------------------------- chart
def make_chart(m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(13.5, 10))
    pf = m["per_feature"]
    fin = [r for r in pf if r["site"] == "resid_final"]
    mid = [r for r in pf if r["site"] == "resid_mid"]
    names = [r["feature"] for r in fin]

    # (a) training curve
    c = m["training_curve"]
    a = ax[0, 0]
    a.plot([p["step"] for p in c], [p["ce"] for p in c], color="tab:red", label="next-token CE (nats)")
    a.set_xlabel("step"); a.set_ylabel("CE (nats)", color="tab:red")
    a2 = a.twinx()
    a2.plot([p["step"] for p in c], [p["close_type_acc"] for p in c], color="tab:blue",
            label="close-type acc")
    a2.plot([p["step"] for p in c], [p["illegal_close_mass"] for p in c], color="tab:green",
            ls="--", label="illegal-close mass @ depth 0")
    a2.set_ylabel("accuracy / prob mass"); a2.set_ylim(0, 1.02)
    a.set_title(f"(a) training: CE {m['clean_ce']:.3f}, close-type acc {m['clean_close_type_acc']:.3f}")
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")

    # (b) the deliverable scatter
    a = ax[0, 1]
    for rows, mk, lab, filled in ((fin, "o", "resid_final", True), (mid, "^", "resid_mid", False)):
        cols = ["tab:red" if r["expected_causal"] else "tab:blue" for r in rows]
        for r, cc in zip(rows, cols):          # seed-to-seed spread
            v = r["causal_effect_nats_by_seed"]
            a.plot([r["probe_selectivity"]] * 2, [min(v), max(v)], color=cc, lw=1.2,
                   alpha=0.5, zorder=2)
        a.scatter([r["probe_selectivity"] for r in rows], [r["causal_effect_nats"] for r in rows],
                  marker=mk, s=110, c=(cols if filled else "none"), edgecolors=cols,
                  linewidths=1.8, label=lab, zorder=4)
        sig = [r for r in rows if r["causally_used_all_seeds"]]
        if sig:
            a.scatter([r["probe_selectivity"] for r in sig], [r["causal_effect_nats"] for r in sig],
                      marker=mk, s=230, facecolors="none", edgecolors="k", linewidths=1.0,
                      zorder=5, label=("causally used (every seed)" if filled else None))
    offs = {"depth_gt0": (10, 6), "top_recency": (10, -14), "depth_parity": (-14, -16),
            "second_type": (10, -12), "cur_tok_is_open": (10, -13)}
    for r in fin:
        off = offs.get(r["feature"], (10, -4))
        a.annotate(r["feature"], (r["probe_selectivity"], r["causal_effect_nats"]),
                   fontsize=8.5, xytext=off, textcoords="offset points",
                   ha=("right" if off[0] < 0 else "left"))
    a.scatter([r["shuffled_probe_acc"] - r["majority_baseline"] for r in fin],
              [r["shuf_ce_excess"] for r in fin],
              marker="x", c="grey", s=45, label="shuffled-label eraser control", zorder=3)
    a.axhline(0, color="k", lw=0.9)
    a.axhline(0.05, color="k", lw=0.7, ls="--")
    a.axvline(0.05, color="k", lw=0.7, ls=":")
    a.set_xlabel("probe accuracy above majority baseline (selectivity)")
    a.set_ylabel("causal effect: excess CE (nats) over rank-matched random erasers\n"
                 "(best of: all positions / depth-0 positions / closer targets)")
    a.set_yscale("symlog", linthresh=0.1)
    a.set_xlim(-0.06, 0.92)
    a.set_title("(b) DELIVERABLE: decodable != used   (red = task provably needs it)")
    a.legend(fontsize=7.5, loc="lower right")
    a.text(0.02, 0.36, "high probe accuracy,\ncausal effect at or below\na random direction\n"
                        "= 'the probe can lie'", ha="left", va="top",
           transform=a.transAxes, fontsize=8.5, style="italic", color="tab:blue")

    # (c) probe accuracy vs controls
    a = ax[1, 0]
    xs = np.arange(len(names)); w = 0.27
    a.bar(xs - w, [r["probe_acc"] for r in fin], w, label="probe acc", color="tab:blue")
    a.bar(xs, [r["majority_baseline"] for r in fin], w, label="majority baseline", color="lightgrey")
    a.bar(xs + w, [r["shuffled_probe_acc"] for r in fin], w, label="shuffled-label probe",
          color="tab:orange")
    for i, r in enumerate(fin):
        a.text(i, 1.03, f"r={r['erasure_rank']}\n{r['refit_probe_acc_after_erasure']:.2f}",
               ha="center", fontsize=7)
    a.set_xticks(xs); a.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    a.set_ylim(0, 1.2); a.set_ylabel("accuracy")
    a.set_title("(c) probe accuracy at resid_final\n(r = LEACE rank; below it, post-erasure refit acc)")
    a.legend(fontsize=8, loc="lower left")

    # (d) which sub-behaviour each erasure damages (role specialisation)
    a = ax[1, 1]
    a.bar(xs - w, [r["causal_ce_close_excess"] for r in fin], w,
          label="CE on closer targets\n(the ')' vs ']' decision - needs top-of-stack type)",
          color="tab:purple")
    a.bar(xs, [r["causal_ce_d0_excess"] for r in fin], w,
          label="CE at depth-0 positions\n(the legality/EOS decision - needs depth>0)",
          color="tab:green")
    a.bar(xs + w, [r["causal_ce_excess"] for r in fin], w,
          label="CE on all positions", color="tab:grey")
    a.set_xticks(xs); a.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    a.set_ylabel("excess nats over rank-matched random erasers")
    a.set_yscale("symlog", linthresh=0.1)
    a.set_title("(d) role specialisation: which decision each erasure breaks")
    a.legend(fontsize=7.5); a.axhline(0, color="k", lw=0.8)

    fig.suptitle("The probe can lie: decodability vs causal use of stack features in a "
                 f"{m['n_params']/1000:.0f}k-param Dyck-2 transformer", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(HERE / "chart.png", dpi=130)


if __name__ == "__main__":
    main()
