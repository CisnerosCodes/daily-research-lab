"""Grokking on (a+b) mod p, and a head-to-head benchmark of four progress measures.

Setup follows the standard grokking recipe (Power et al. 2201.02177; Nanda et al.
2301.05217): a 1-layer, LayerNorm-free, bias-free transformer over the sequence
[a, b, "="], logits read from the last position only, FULL-BATCH AdamW with weight
decay 1.0, betas (0.9, 0.98), on a fixed random fraction of the p^2 pairs.

SHRUNK to fit a ~10 minute CPU box: p=59 (not 97/113), d_model=64 (not 128), and a
per-run wall-clock cap. A train-fraction sweep replaces a single run, because the
memorisation->generalisation DELAY is controlled by the train fraction and we want to
show the delay appearing and lengthening as data is removed.

--------------------------------------------------------------------------------
The question: several 2026 papers propose "spectral entropy collapse" as an early
warning signal for grokking. Does it actually LEAD the test-accuracy jump, and does
it beat the older restricted/excluded-loss measures at the same scale?

Four progress measures are tracked at every eval:

  1. H_fourier  - Shannon entropy of the DFT power spectrum of the token-embedding
                  matrix W_E[0:p], taken down the TOKEN axis, DC dropped, summed over
                  embedding columns. Random embeddings are spectrally flat
                  (-> log2(p//2) bits); the grokked Fourier-multiplication algorithm
                  needs a handful of frequencies (-> ~2 bits).
  2. H_svd      - Shannon entropy of the normalised singular-value ENERGY spectrum
                  (s_i^2 / sum s_j^2) of the same W_E[0:p]. Basis-free rank measure.
  3. H_cov      - Shannon entropy of the eigenvalue spectrum of the covariance of the
                  PENULTIMATE-layer representations over the train set. This is the
                  measure of arXiv 2604.13123, reimplemented here for comparison.
  4. restricted / excluded loss (Nanda et al. 2301.05217). Logits are computed over
                  the full p x p grid, centred over the output axis, 2D-DFT'd over
                  (a, b), and split into the part supported on the model's own key
                  frequencies (RESTRICTED) and the remainder (EXCLUDED). Restricted
                  loss falling / excluded loss rising = the Fourier circuit forming
                  and the memorising circuit being cleaned up.

Measures 1-3 are online and cheap. Measure 4 needs the key frequencies of the FINAL
model, so training stores periodic in-RAM checkpoints and replays them afterwards -
that is Nanda's own protocol, and it is what makes 4 a fair comparison rather than a
causally-impossible one.

Lead time is measured from the MEMORISATION step (train acc >= 0.99), not from step 0,
so the large transient every measure shows while the network is memorising cannot be
mistaken for an early warning. For each measure we compute the fraction of its total
post-memorisation movement completed at step t, and compare when that fraction crosses
0.5 (and 0.1) with when test accuracy crosses 0.5 (and leaves its plateau).

Efficiency note: only the final sequence position is read out, so the MLP and the
unembedding are applied at that one position only (keys/values still come from all 3
positions). Mathematically identical to the full forward pass and ~3x cheaper.

Deterministic, CPU-only, single-threaded. Writes results.json, chart.png, model.pt
(the final model of whichever run grokked BEST) and model_mid.pt (a memorised-but-not-
yet-grokked checkpoint of the same run, for the SAE follow-up experiment).

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


# ----------------------------- model ---------------------------------------
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)


class GrokTransformer(nn.Module):
    """1-layer attention + MLP transformer, NO LayerNorm, no biases.

    Vocab is p+1 (numbers 0..p-1 plus the "=" token). Input is always [a, b, =];
    the logits over the p residues are read from the last position only.
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

    def hidden(self, idx):
        """idx: (N, n_ctx) long -> penultimate residual stream (N, d) at last pos."""
        N, T = idx.shape
        H, Dh = self.n_heads, self.d_head
        x = self.W_E[idx] + self.W_pos[None, :T, :]          # (N, T, d)
        last = x[:, -1:, :]                                   # (N, 1, d)
        q = (last @ self.W_Q).view(N, 1, H, Dh).transpose(1, 2)
        k = (x @ self.W_K).view(N, T, H, Dh).transpose(1, 2)
        v = (x @ self.W_V).view(N, T, H, Dh).transpose(1, 2)
        att = (q @ k.transpose(-2, -1) / (Dh ** 0.5)).softmax(-1)   # (N,H,1,T)
        z = (att @ v).transpose(1, 2).reshape(N, 1, self.d_model) @ self.W_O
        h = last + z
        h = h + F.relu(h @ self.W_in) @ self.W_out
        return h.view(N, self.d_model)

    def forward(self, idx):
        return self.hidden(idx) @ self.W_U


# ------------------------- progress measures --------------------------------
def entropy_bits(pk):
    pk = np.asarray(pk, dtype=np.float64)
    pk = pk / max(pk.sum(), 1e-30)
    return float(-(pk * np.log2(pk + 1e-30)).sum())


def fourier_entropy(W):
    """W: (p, d) embedding rows of the p NUMBER tokens.

    DFT down the token axis per column, drop DC, sum power over columns, normalise
    to a distribution over the p//2 non-DC frequencies, Shannon entropy in bits.
    Returns (entropy_bits, normalised aggregate power spectrum).
    """
    spec = np.fft.rfft(W, axis=0)                    # (p//2+1, d)
    pw = (np.abs(spec) ** 2)[1:]                     # drop DC -> (p//2, d)
    agg = pw.sum(axis=1)
    agg = agg / max(agg.sum(), 1e-30)
    return entropy_bits(agg), agg


def svd_entropy(W):
    """Entropy (bits) of the normalised singular-value ENERGY spectrum of W."""
    s = np.linalg.svd(np.asarray(W, dtype=np.float64), compute_uv=False)
    return entropy_bits(s ** 2)


def cov_eig_entropy(Hrep):
    """Entropy (bits) of the eigenvalue spectrum of cov(Hrep). Hrep: (N, d).

    This is the measure of arXiv 2604.13123 (spectral entropy of the representation
    covariance), reimplemented for a head-to-head comparison.
    """
    X = np.asarray(Hrep, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    ev = np.linalg.svd(X, compute_uv=False) ** 2      # eigenvalues of X^T X (prop. to cov)
    return entropy_bits(np.clip(ev, 0, None))


# ----------------------- restricted / excluded loss -------------------------
def freq_index(p):
    """Map DFT bin k (0..p-1) to its frequency magnitude min(k, p-k)."""
    k = np.arange(p)
    return np.minimum(k, p - k)


def restricted_excluded_loss(logits_grid, key_freqs, tr_a, tr_b, tr_y, p):
    """logits_grid: (p, p, p) numpy, [a, b, c].

    Centre over the output axis c, 2D-DFT over (a, b), split into the part supported
    on frequency pairs drawn from key_freqs U {0} (RESTRICTED) and the rest
    (EXCLUDED), and return the train cross-entropy of each part.
    """
    L = logits_grid - logits_grid.mean(axis=-1, keepdims=True)
    Ff = np.fft.fft2(L, axes=(0, 1))
    fi = freq_index(p)
    allowed = np.zeros(p, dtype=bool)
    allowed[np.isin(fi, list(key_freqs) + [0])] = True
    mask = np.outer(allowed, allowed)[:, :, None]
    restricted = np.real(np.fft.ifft2(Ff * mask, axes=(0, 1)))
    excluded = L - restricted

    def ce(arr):
        z = torch.from_numpy(np.ascontiguousarray(arr[tr_a, tr_b])).double()
        return float(F.cross_entropy(z, torch.from_numpy(tr_y).long()))

    return ce(restricted), ce(excluded), ce(L)


# ----------------------------- helpers -------------------------------------
def first_crossing(steps, values, thresh, rising=True, start_step=None):
    """First step at which `values` crosses `thresh` (linear interp). None if never.

    If start_step is given, only crossings at or after that step count.
    """
    idx0 = 0
    if start_step is not None:
        while idx0 < len(steps) and steps[idx0] < start_step:
            idx0 += 1
        if idx0 >= len(steps):
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
    """Fraction of total POST-MEMORISATION movement completed, as a list.

    0 at the memorisation step, 1 at the run's extreme value after it. Sign-agnostic:
    works for measures that fall (entropies, restricted loss) and rise (excluded loss).
    """
    if t_mem is None:
        return None
    v_mem = interp_at(steps, values, t_mem)
    post = [v for s, v in zip(steps, values) if s >= t_mem]
    if not post:
        return None
    v_end = post[-1]
    denom = v_end - v_mem
    if abs(denom) < 1e-12:
        return None
    return [float((v - v_mem) / denom) for v in values]


# ----------------------------- one run --------------------------------------
def train_one(P, train_frac, time_cap_s, seed, log):
    p = int(P["p"])
    set_seeds(seed)
    t0 = time.time()

    a_all, b_all = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    a_all, b_all = a_all.reshape(-1), b_all.reshape(-1)
    y_all = (a_all + b_all) % p
    X = torch.from_numpy(np.stack([a_all, b_all, np.full_like(a_all, p)], 1)).long()
    Y = torch.from_numpy(y_all).long()

    rng = np.random.default_rng(seed)
    perm = rng.permutation(p * p)
    n_train = int(round(train_frac * p * p))
    tr_idx, te_idx = np.sort(perm[:n_train]), np.sort(perm[n_train:])
    Xtr, Ytr, Xte, Yte = X[tr_idx], Y[tr_idx], X[te_idx], Y[te_idx]
    tr_a, tr_b, tr_y = a_all[tr_idx], b_all[tr_idx], y_all[tr_idx]
    log(f"  data: p={p}, frac={train_frac}, {len(tr_idx)} train / {len(te_idx)} test")

    model = GrokTransformer(p, P["d_model"], P["n_heads"], P["d_mlp"],
                            P["n_ctx"], P["init_std_scale"])
    n_params = sum(q.numel() for q in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=P["lr"], weight_decay=P["weight_decay"],
                            betas=(P["beta1"], P["beta2"]))

    @torch.no_grad()
    def evaluate():
        model.eval()
        out = {}
        htr = model.hidden(Xtr)
        ltr = htr @ model.W_U
        out["train_loss"] = float(F.cross_entropy(ltr, Ytr))
        out["train_acc"] = float((ltr.argmax(-1) == Ytr).float().mean())
        lte = model(Xte)
        out["test_loss"] = float(F.cross_entropy(lte, Yte))
        out["test_acc"] = float((lte.argmax(-1) == Yte).float().mean())
        we = model.W_E.detach().numpy()[:p]
        out["H_fourier"], out["_spec"] = fourier_entropy(we)
        out["H_svd"] = svd_entropy(we)
        out["H_cov"] = cov_eig_entropy(htr.numpy())
        model.train()
        return out

    keys = ("step", "train_acc", "test_acc", "train_loss", "test_loss",
            "H_fourier", "H_svd", "H_cov", "wall_s")
    hist = {k: [] for k in keys}
    ckpts = []                    # (step, state_dict clone) for the replay measure
    init_spec = final_spec = None
    mid_saved = None
    memo_thr = float(P["train_acc_memorized_threshold"])
    ev = int(P["eval_every"])
    ck_every = int(P["ckpt_every_evals"]) * ev
    capped, step, n_ev = False, 0, 0

    for step in range(int(P["max_steps"]) + 1):
        if step % ev == 0:
            e = evaluate()
            if init_spec is None:
                init_spec = e["_spec"]
            final_spec = e["_spec"]
            hist["step"].append(step)
            hist["wall_s"].append(round(time.time() - t0, 2))
            for k in ("train_acc", "test_acc", "train_loss", "test_loss",
                      "H_fourier", "H_svd", "H_cov"):
                hist[k].append(e[k])
            if step % ck_every == 0:
                ckpts.append((step, {k: v.detach().clone()
                                     for k, v in model.state_dict().items()}))
            if (mid_saved is None and e["train_acc"] >= memo_thr
                    and e["test_acc"] < 0.3):
                mid_saved = (step, {k: v.detach().clone()
                                    for k, v in model.state_dict().items()},
                             e["train_acc"], e["test_acc"])
            n_ev += 1
            if n_ev % 20 == 1:
                log(f"    step {step:6d}  tr {e['train_acc']:.4f}  te {e['test_acc']:.4f}  "
                    f"Hf {e['H_fourier']:.3f}  Hs {e['H_svd']:.3f}  Hc {e['H_cov']:.3f}  "
                    f"({time.time()-t0:.0f}s)")
        if step == int(P["max_steps"]):
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
    train_s = time.time() - t0
    final = evaluate()
    final_spec = final["_spec"]

    # ---- key frequencies of the FINAL model, then replay restricted/excluded ----
    order = np.argsort(final_spec)[::-1]
    cum = np.cumsum(final_spec[order])
    n_key = int(min(np.searchsorted(cum, float(P["key_freq_power"])) + 1,
                    int(P["max_key_freqs"])))
    key_freqs = [int(order[i]) + 1 for i in range(n_key)]      # +1: DC was dropped
    Xgrid = X                                                   # all p^2 pairs, ordered a-major
    rep_steps, rep_restricted, rep_excluded, rep_full = [], [], [], []
    probe = GrokTransformer(p, P["d_model"], P["n_heads"], P["d_mlp"],
                            P["n_ctx"], P["init_std_scale"])
    with torch.no_grad():
        for st, sd in ckpts:
            probe.load_state_dict(sd)
            lg = probe(Xgrid).numpy().reshape(p, p, p)
            r, x, f = restricted_excluded_loss(lg, key_freqs, tr_a, tr_b, tr_y, p)
            rep_steps.append(st)
            rep_restricted.append(r)
            rep_excluded.append(x)
            rep_full.append(f)
    log(f"    key freqs {key_freqs} ({100*cum[n_key-1]:.1f}% of embedding power); "
        f"replayed {len(ckpts)} checkpoints")

    return dict(
        train_frac=train_frac, p=p, n_params=n_params,
        n_train=int(len(tr_idx)), n_test=int(len(te_idx)),
        steps_run=steps_run, time_capped=capped, train_seconds=round(train_s, 1),
        sec_per_step=round(train_s / max(steps_run, 1), 4),
        hist=hist, final=final, init_spec=init_spec, final_spec=final_spec,
        key_freqs=key_freqs, key_freq_power=float(cum[n_key - 1]),
        rep_steps=rep_steps, rep_restricted=rep_restricted,
        rep_excluded=rep_excluded, rep_full=rep_full,
        mid_saved=mid_saved, model=model, tr_idx=tr_idx, te_idx=te_idx,
    )


# --------------------------- lead-time analysis ------------------------------
def analyse(run, P, cut=None):
    """Lead-time analysis. If `cut` is given, every series is truncated to steps <= cut,
    which simulates having STOPPED training at that point (used to expose the
    normalisation bias that truncation introduces)."""
    def cutf(st, val):
        if cut is None:
            return list(st), list(val)
        pr = [(s, v) for s, v in zip(st, val) if s <= cut]
        return [x[0] for x in pr], [x[1] for x in pr]

    hist = run["hist"]
    S, tr_a = cutf(hist["step"], hist["train_acc"])
    _, te_a = cutf(hist["step"], hist["test_acc"])
    memo_thr = float(P["train_acc_memorized_threshold"])
    t_mem = first_crossing(S, tr_a, memo_thr)

    # test-accuracy reference events
    chance = 1.0 / run["p"]
    plateau_exit_thr = float(P["plateau_exit_abs"])
    t_test = {str(t): first_crossing(S, te_a, t, start_step=t_mem)
              for t in P["test_acc_jump_thresholds"]}
    t_plateau_exit = first_crossing(S, te_a, chance + plateau_exit_thr, start_step=t_mem)
    t_test50 = t_test.get("0.5")

    # every measure, on a common footing
    series = {
        "H_fourier": cutf(hist["step"], hist["H_fourier"]),
        "H_svd": cutf(hist["step"], hist["H_svd"]),
        "H_cov": cutf(hist["step"], hist["H_cov"]),
        "restricted_loss": cutf(run["rep_steps"], run["rep_restricted"]),
        "excluded_loss": cutf(run["rep_steps"], run["rep_excluded"]),
    }
    measures = {}
    for name, (st, val) in series.items():
        frac = progress_fraction(st, val, t_mem)
        m = {
            "value_at_memorization": interp_at(st, val, t_mem),
            "value_final": float(val[-1]) if val else None,
        }
        if frac is None:
            m.update({"t_10pct": None, "t_50pct": None,
                      "lead_vs_test50_steps": None, "lead_vs_plateau_exit_steps": None,
                      "frac_done_at_test50": None})
        else:
            t10 = first_crossing(st, frac, 0.10, start_step=t_mem)
            t50 = first_crossing(st, frac, 0.50, start_step=t_mem)
            m["t_10pct"] = t10
            m["t_50pct"] = t50
            m["lead_vs_test50_steps"] = (None if (t50 is None or t_test50 is None)
                                         else round(t_test50 - t50, 1))
            m["lead_vs_plateau_exit_steps"] = (None if (t10 is None or t_plateau_exit is None)
                                               else round(t_plateau_exit - t10, 1))
            m["frac_done_at_test50"] = (None if t_test50 is None
                                        else round(float(interp_at(st, frac, t_test50)), 4))
            # >1 means the measure overshot and came back: it is NOT monotone after
            # memorisation, so its "50% of net movement" crossing is not a clean onset.
            post = [f for s, f in zip(st, frac) if t_mem is None or s >= t_mem]
            m["frac_overshoot"] = round(float(max(post)), 3) if post else None
            m["monotone_post_mem"] = bool(post and max(post) <= 1.02)
            m["_frac_series"] = [round(float(x), 5) for x in frac]
            m["_frac_steps"] = [int(x) for x in st]
        measures[name] = m

    return dict(t_mem=t_mem, t_test=t_test, t_test50=t_test50,
                t_plateau_exit=t_plateau_exit, chance=chance, measures=measures)


# ----------------------------- main ----------------------------------------
def main():
    cfg = load_config()
    P = cfg["params"]
    seed = int(cfg.get("seed", 0))
    t0 = time.time()
    log = lambda s: print(s, flush=True)

    runs, analyses = [], []
    for spec in P["runs"]:
        log(f"run: train_frac={spec['train_frac']} cap={spec['time_cap_s']}s")
        r = train_one(P, float(spec["train_frac"]), float(spec["time_cap_s"]), seed, log)
        a = analyse(r, P)
        log(f"    -> final test_acc {r['final']['test_acc']:.4f} after {r['steps_run']} steps; "
            f"memorized at {a['t_mem']}, test50 at {a['t_test50']}")
        runs.append(r)
        analyses.append(a)

    # ---- primary run for the lead-time analysis. A lead time is undefined without a
    #      test-accuracy jump, and it is BIASED if training stopped at the jump (the
    #      "fraction of total movement" denominator is then still growing). So prefer a
    #      run that grokked FULLY inside its box; fall back to any run that hit 0.5. ----
    prim_i = next((i for i, (r, a) in enumerate(zip(runs, analyses))
                   if a["t_test50"] is not None and r["final"]["test_acc"] >= 0.9),
                  next((i for i, a in enumerate(analyses)
                        if a["t_test50"] is not None), 0))
    prim, prim_a = runs[prim_i], analyses[prim_i]
    # ---- checkpoints for the downstream SAE experiment come from the run that
    #      actually grokked BEST, so model.pt is always the most-grokked model. ----
    ck = max(runs, key=lambda r: r["final"]["test_acc"])
    torch.save({"config": dict(P), "p": ck["p"], "seed": seed, "step": ck["steps_run"],
                "train_frac": ck["train_frac"],
                "train_idx": ck["tr_idx"], "test_idx": ck["te_idx"],
                "train_acc": ck["final"]["train_acc"], "test_acc": ck["final"]["test_acc"],
                "key_freqs": ck["key_freqs"],
                "arch": {k: P[k] for k in ("d_model", "n_heads", "d_mlp", "n_ctx",
                                           "init_std_scale")},
                "state_dict": ck["model"].state_dict()}, HERE / "model.pt")
    # mid checkpoint: prefer the same run as model.pt; if that run's plateau was too
    # short to contain one, fall back to the next-best-grokked run that has one.
    mid_run = ck if ck["mid_saved"] is not None else next(
        (r for r in sorted(runs, key=lambda r: -r["final"]["test_acc"])
         if r["mid_saved"] is not None), None)
    mid_step = None
    if mid_run is not None:
        mid_step, sd, mtr, mte = mid_run["mid_saved"]
        torch.save({"config": dict(P), "p": mid_run["p"], "seed": seed, "step": mid_step,
                    "train_frac": mid_run["train_frac"],
                    "train_idx": mid_run["tr_idx"], "test_idx": mid_run["te_idx"],
                    "train_acc": mtr, "test_acc": mte,
                    "arch": {k: P[k] for k in ("d_model", "n_heads", "d_mlp", "n_ctx",
                                               "init_std_scale")},
                    "state_dict": sd}, HERE / "model_mid.pt")

    # ----------------------------- metrics -----------------------------------
    def run_metrics(r, a):
        ms = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
              for k, v in a["measures"].items()}
        return {
            "train_frac": r["train_frac"], "n_train": r["n_train"], "n_test": r["n_test"],
            "steps_run": r["steps_run"], "time_capped": r["time_capped"],
            "train_seconds": r["train_seconds"], "sec_per_step": r["sec_per_step"],
            "final_train_acc": round(r["final"]["train_acc"], 4),
            "final_test_acc": round(r["final"]["test_acc"], 4),
            "final_train_loss": round(r["final"]["train_loss"], 5),
            "final_test_loss": round(r["final"]["test_loss"], 5),
            "memorized": bool(max(r["hist"]["train_acc"]) >= P["train_acc_memorized_threshold"]),
            "grok_started": bool(max(r["hist"]["test_acc"]) >= 0.3),
            "grokked_full": bool(r["final"]["test_acc"] >= 0.9),
            "step_memorized": a["t_mem"],
            "step_test_acc_thresholds": a["t_test"],
            "step_test_plateau_exit": a["t_plateau_exit"],
            "grok_delay_steps": (None if (a["t_mem"] is None or a["t_test50"] is None)
                                 else round(a["t_test50"] - a["t_mem"], 1)),
            "H_fourier_init": round(r["hist"]["H_fourier"][0], 4),
            "H_fourier_final": round(r["hist"]["H_fourier"][-1], 4),
            "H_svd_init": round(r["hist"]["H_svd"][0], 4),
            "H_svd_final": round(r["hist"]["H_svd"][-1], 4),
            "H_cov_init": round(r["hist"]["H_cov"][0], 4),
            "H_cov_final": round(r["hist"]["H_cov"][-1], 4),
            "effective_n_frequencies_init": round(float(2 ** r["hist"]["H_fourier"][0]), 2),
            "effective_n_frequencies_final": round(float(2 ** r["hist"]["H_fourier"][-1]), 2),
            "key_frequencies": r["key_freqs"],
            "key_freq_power_share": round(r["key_freq_power"], 4),
            "measures": ms,
            "history": {k: ([int(x) for x in v] if k == "step"
                            else [round(float(x), 6) for x in v])
                        for k, v in r["hist"].items()},
            "replay_history": {
                "step": [int(x) for x in r["rep_steps"]],
                "restricted_loss": [round(float(x), 6) for x in r["rep_restricted"]],
                "excluded_loss": [round(float(x), 6) for x in r["rep_excluded"]],
                "full_centered_loss": [round(float(x), 6) for x in r["rep_full"]],
            },
            "final_agg_power_spectrum": [round(float(x), 6) for x in r["final_spec"]],
            "init_agg_power_spectrum": [round(float(x), 6) for x in r["init_spec"]],
        }

    per_run = [run_metrics(r, a) for r, a in zip(runs, analyses)]

    def lead_table_for(a):
        return {name: {"lead_vs_test50_steps": m["lead_vs_test50_steps"],
                       "lead_vs_plateau_exit_steps": m["lead_vs_plateau_exit_steps"],
                       "frac_done_at_test50": m["frac_done_at_test50"],
                       "frac_overshoot": m.get("frac_overshoot"),
                       "monotone_post_mem": m.get("monotone_post_mem")}
                for name, m in a["measures"].items()}

    # leaderboard on the primary (fully grokked) run
    lead_table = lead_table_for(prim_a)

    # --- control 1 (deterministic): re-analyse the SAME primary run as if training had
    #     been STOPPED at the test-accuracy jump. The "fraction of total movement"
    #     denominator is then still growing, so measures that really lag can be made to
    #     look like they lead. This isolates the normalisation artifact within one run. ---
    cut_step = next((s for s, t in zip(prim["hist"]["step"], prim["hist"]["test_acc"])
                     if t >= 0.5), None)
    prim_cut_a = analyse(prim, P, cut=cut_step) if cut_step is not None else None

    # --- control 2: a run that memorised but NEVER left the memorisation plateau
    #     (test acc never exceeded chance + plateau_exit_abs). Any measure that moves
    #     here is producing a FALSE POSITIVE. Keyed on plateau exit, not on the 0.5
    #     crossing, so it is robust to where the wall-clock cap happens to land. ---
    null_i = next((i for i, (r, a) in enumerate(zip(runs, analyses))
                   if a["t_plateau_exit"] is None and a["t_mem"] is not None), None)
    # --- a run that left the plateau but did not finish grokking inside its box ---
    part_i = next((i for i, (r, a) in enumerate(zip(runs, analyses))
                   if a["t_plateau_exit"] is not None and r["final"]["test_acc"] < 0.9), None)
    null_moved = None
    if null_i is not None:
        na, nr = analyses[null_i], runs[null_i]
        null_moved = {
            "train_frac": nr["train_frac"],
            "final_test_acc": round(nr["final"]["test_acc"], 4),
            "chance": round(na["chance"], 4),
            "steps_run": nr["steps_run"],
            "step_at_50pct_of_measure_movement": {
                name: m["t_50pct"] for name, m in na["measures"].items()},
            "H_fourier_drop_bits": round(nr["hist"]["H_fourier"][0]
                                         - nr["hist"]["H_fourier"][-1], 4),
            "H_svd_drop_bits": round(nr["hist"]["H_svd"][0] - nr["hist"]["H_svd"][-1], 4),
            "H_cov_drop_bits": round(nr["hist"]["H_cov"][0] - nr["hist"]["H_cov"][-1], 4),
            "key_freq_power_share": round(nr["key_freq_power"], 4),
        }
    leads = {k: v["lead_vs_test50_steps"] for k, v in lead_table.items()
             if v["lead_vs_test50_steps"] is not None}
    best = max(leads, key=leads.get) if leads else None
    any_leads = {k: v for k, v in leads.items() if v > 0}

    metrics = {
        "n_params": prim["n_params"], "p": prim["p"],
        "primary_train_frac": prim["train_frac"],
        "primary_run_index": prim_i,
        "n_runs": len(runs),
        "per_run": per_run,
        "lead_table_primary_run": lead_table,
        "lead_table_primary_truncated_at_jump": (None if prim_cut_a is None
                                                 else lead_table_for(prim_cut_a)),
        "truncation_cut_step": cut_step,
        "partial_run_train_frac": (None if part_i is None else runs[part_i]["train_frac"]),
        "partial_run_final_test_acc": (None if part_i is None
                                       else round(runs[part_i]["final"]["test_acc"], 4)),
        "false_positive_control_never_grokked": null_moved,
        "best_measure_primary_run": best,
        "measures_that_lead": sorted(any_leads.keys()),
        "n_measures_that_lead": len(any_leads),
        "spectral_entropies_lead_on_completed_run": bool(
            all((lead_table.get(k, {}).get("lead_vs_test50_steps") or 0) > 0
                for k in ("H_fourier", "H_svd", "H_cov"))),
        "prior_art_measure_H_cov_lead_steps": lead_table.get("H_cov", {}).get("lead_vs_test50_steps"),
        "mid_checkpoint_step": mid_step,
        "mid_checkpoint_from_train_frac": (None if mid_run is None else mid_run["train_frac"]),
        "checkpoint_from_train_frac": ck["train_frac"],
        "checkpoint_test_acc": round(ck["final"]["test_acc"], 4),
        "total_compute_seconds": round(sum(r["train_seconds"] for r in runs), 1),
    }
    if leads:
        hl = (f"p={prim['p']}, primary run train_frac={prim['train_frac']} "
              f"(grokked fully: final test acc {prim['final']['test_acc']:.3f}): "
              f"memorized at step {prim_a['t_mem']:.0f}, test acc 50% at "
              f"{prim_a['t_test50']:.0f} -> grok delay "
              f"{per_run[prim_i]['grok_delay_steps']:.0f} steps. Lead over test-acc-50% "
              "(+ = fires first): "
              + ", ".join(f"{k} {v:+.0f}" for k, v in leads.items()))
        if prim_cut_a is not None:
            ct = lead_table_for(prim_cut_a)
            hl += (". Same run re-analysed as if STOPPED at the jump flips them to "
                   + ", ".join(f"{k} {ct[k]['lead_vs_test50_steps']:+.0f}"
                               for k in ("H_fourier", "H_svd", "H_cov")
                               if ct.get(k, {}).get("lead_vs_test50_steps") is not None)
                   + " (normalization artifact)")
        if null_moved is not None:
            hl += (f". FALSE-POSITIVE control (train_frac={null_moved['train_frac']}, "
                   f"test acc {null_moved['final_test_acc']:.3f} vs chance "
                   f"{null_moved['chance']:.3f} after {null_moved['steps_run']} steps): "
                   f"H_fourier still fell {null_moved['H_fourier_drop_bits']:.2f} bits and "
                   f"H_svd {null_moved['H_svd_drop_bits']:.2f} bits with zero generalization")
    else:
        hl = f"p={prim['p']}: no test-acc-50% crossing in any run inside the time box"
    metrics["headline"] = hl

    # ----------------------------- chart -------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C_TR, C_TE = "#3d5a80", "#c95d3c"
    CM = {"H_fourier": "#1a7f64", "H_svd": "#7b4fa3", "H_cov": "#c9a227",
          "restricted_loss": "#2b7bba", "excluded_loss": "#b03a48"}
    nrow = len(runs)
    fig, axes = plt.subplots(nrow, 3, figsize=(16.5, 4.3 * nrow), squeeze=False)

    for i, (r, a) in enumerate(zip(runs, analyses)):
        S = r["hist"]["step"]
        xs = [max(s, 1) for s in S]
        ax1, ax2, ax3 = axes[i]

        # --- panel 1: the grokking curve
        ax1.plot(xs, r["hist"]["train_acc"], color=C_TR, lw=2, label="train acc")
        ax1.plot(xs, r["hist"]["test_acc"], color=C_TE, lw=2, label="test acc")
        ax1.axhline(a["chance"], color="0.6", ls=":", lw=1)
        ax1.text(xs[1], a["chance"] + .02, "chance", fontsize=7, color="0.45")
        if a["t_mem"]:
            ax1.axvline(max(a["t_mem"], 1), color=C_TR, ls="--", lw=1)
            ax1.text(max(a["t_mem"], 1) * 1.1, .5, "memorized", rotation=90,
                     fontsize=8, color=C_TR)
        if a["t_test50"]:
            ax1.axvline(a["t_test50"], color=C_TE, ls="--", lw=1)
            ax1.text(a["t_test50"] * 1.1, .5, "test acc 50%", rotation=90,
                     fontsize=8, color=C_TE)
        d = per_run[i]["grok_delay_steps"]
        if d and r["final"]["test_acc"] >= 0.9:
            sub = f"GROKKED, delay {d:.0f} steps"
        elif d:
            sub = f"delay {d:.0f} steps, TRUNCATED at test acc {r['final']['test_acc']:.2f}"
        else:
            sub = f"NEVER left the plateau (test acc {r['final']['test_acc']:.3f} = chance)"
        ax1.set_xscale("log")
        ax1.set_xlabel("optimizer step (full batch)")
        ax1.set_ylabel("accuracy")
        ax1.set_ylim(-.03, 1.05)
        ax1.legend(frameon=False, fontsize=8, loc="center left")
        ax1.set_title(f"train_frac={r['train_frac']} ({r['n_train']} pairs): {sub}",
                      fontsize=10)
        ax1.spines[["top", "right"]].set_visible(False)

        # --- panel 2: normalised progress of every measure vs test acc
        ax2.plot(xs, r["hist"]["test_acc"], color=C_TE, lw=2.4, label="test acc", zorder=5)
        for name, m in a["measures"].items():
            if "_frac_series" not in m:
                continue
            fs, fv = m["_frac_steps"], m["_frac_series"]
            ax2.plot([max(s, 1) for s in fs], np.clip(fv, -0.05, 1.05),
                     color=CM[name], lw=1.5, alpha=.9,
                     label=f"{name} ({m['lead_vs_test50_steps']:+} st)"
                     if m["lead_vs_test50_steps"] is not None else name)
            if m["t_50pct"]:
                ax2.plot([max(m["t_50pct"], 1)], [0.5], "o", color=CM[name], ms=5, zorder=6)
        ax2.axhline(0.5, color="0.75", ls=":", lw=1)
        if a["t_mem"]:
            ax2.axvline(max(a["t_mem"], 1), color=C_TR, ls="--", lw=1)
        if a["t_test50"]:
            ax2.axvline(a["t_test50"], color=C_TE, ls="--", lw=1)
        ax2.set_xscale("log")
        ax2.set_ylim(-.06, 1.08)
        ax2.set_xlabel("optimizer step")
        ax2.set_ylabel("fraction of post-memorization movement")
        ax2.legend(frameon=False, fontsize=7.5, loc="upper left")
        ax2.set_title("Progress measures vs generalization\n"
                      "(dots = 50% crossing; + lead = fires BEFORE test acc)", fontsize=10)
        ax2.spines[["top", "right"]].set_visible(False)

        # --- panel 3: raw restricted/excluded loss, plus the Fourier spectrum inset
        rs = [max(s, 1) for s in r["rep_steps"]]
        ax3.plot(rs, r["rep_full"], color="0.5", lw=1.4, label="full (centered) train loss")
        ax3.plot(rs, r["rep_restricted"], color=CM["restricted_loss"], lw=2,
                 label="restricted loss")
        ax3.plot(rs, r["rep_excluded"], color=CM["excluded_loss"], lw=2, label="excluded loss")
        ax3.set_xscale("log")
        ax3.set_yscale("log")
        ax3.set_xlabel("optimizer step")
        ax3.set_ylabel("train cross-entropy")
        ax3.legend(frameon=False, fontsize=8, loc="lower left")
        ax3.set_title(f"Nanda restricted/excluded loss\nkey freqs {r['key_freqs']} "
                      f"= {100*r['key_freq_power']:.0f}% of W_E power", fontsize=10)
        ax3.spines[["top", "right"]].set_visible(False)
        axi = ax3.inset_axes([0.56, 0.60, 0.42, 0.36])
        fr = np.arange(1, len(r["final_spec"]) + 1)
        axi.bar(fr, r["init_spec"], color="0.75")
        axi.bar(fr, r["final_spec"], color=CM["H_fourier"], alpha=.85)
        axi.set_title("W_E Fourier power (grey=init)", fontsize=6.5)
        axi.tick_params(labelsize=6)

    fig.suptitle(f"Grokking (a+b) mod {prim['p']} - 1-layer transformer, no LayerNorm, "
                 f"{prim['n_params']/1000:.0f}k params, full-batch AdamW "
                 f"(lr {P['lr']}, wd {P['weight_decay']}), seed {seed} - "
                 f"four progress measures benchmarked head-to-head",
                 fontsize=12, y=1.005)
    fig.tight_layout()
    fig.savefig(HERE / "chart.png", dpi=145, bbox_inches="tight")

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
    log("headline: " + metrics["headline"])
    log(json.dumps({k: results[k] for k in ("id", "duration_sec", "status")}, indent=2))


if __name__ == "__main__":
    main()
