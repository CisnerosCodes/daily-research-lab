"""Induction-head emergence as a phase transition, and whether K-composition strength
predicts it.

Setting (Olsson et al. 2022, transformer-circuits.pub): a 2-layer ATTENTION-ONLY
transformer (no MLPs, no LayerNorm) on periodically-repeating random-token sequences.
Per sample a period p ~ U{p_min..p_max} is drawn, the first p tokens are sampled WITHOUT
replacement from the vocab (so every token is unique inside a period and the induction
target is unambiguous), and the sequence is tiled to length T. Tokens at target index
j <= p-1 are first occurrences (irreducible loss = log V); tokens at j >= p+1 are only
predictable through an induction circuit (prev-token head in layer 0 K-composing with an
induction head in layer 1). The per-sample random period makes a purely positional
"attend to i-p" shortcut impossible.

Tracked every `eval_every` steps on a fixed held-out eval batch:
  * in-context loss delta   = mean loss(repeat targets) - mean loss(first-occurrence targets)
  * prev-token score        = layer-0 head attention mass on position i-1
  * induction score         = layer-1 head attention mass on (prev occurrence of token i) + 1
  * K/Q/V composition       = ||W_QK^(1) W_OV^(0)||_F / (||W_QK||_F ||W_OV||_F) for all 16
                              head pairs, against a random-orthogonal-rotation NULL that
                              preserves both singular-value spectra

Claim under test: the in-context loss drops sharply, and composition strength starts
rising BEFORE it (a leading indicator). Framed against 2026-07-25_grokking-modular-addition,
where every spectral progress measure LAGGED and the published lead turned out to be a
truncation artifact -- so leads are reported under BOTH a 50%-of-total-movement criterion
(needs the final model; artifact-prone) and an absolute online z-threshold criterion.

Deterministic, CPU-only, writes results.json + chart.png.
Usage:  python run.py
"""
import json, math, os, random, subprocess, sys, time
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


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)


# ----------------------------- model ---------------------------------------
class Attn(nn.Module):
    """Multi-head causal attention, no bias. Column convention for the analysis:
    q = W_Q x with W_Q = self.q.weight[head rows] (dh, d)."""

    def __init__(self, d, h):
        super().__init__()
        self.h, self.dh = h, d // h
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x, need_att=False):
        B, T, D = x.shape
        q = self.q(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.k(x).view(B, T, self.h, self.dh).transpose(1, 2)
        v = self.v(x).view(B, T, self.h, self.dh).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        att = att.masked_fill(mask, float("-inf")).softmax(-1)
        y = self.o((att @ v).transpose(1, 2).reshape(B, T, D))
        return (y, att) if need_att else (y, None)

    # --- circuit matrices (Anthropic "Mathematical Framework" convention) ---
    def W_QK(self, h):
        s = slice(h * self.dh, (h + 1) * self.dh)
        return self.q.weight[s].T @ self.k.weight[s]          # (d, d)

    def W_OV(self, h):
        s = slice(h * self.dh, (h + 1) * self.dh)
        return self.o.weight[:, s] @ self.v.weight[s]          # (d, d)


class AttnOnlyLM(nn.Module):
    """2-layer attention-only transformer. No MLP, no LayerNorm."""

    def __init__(self, vocab, d, h, n_layers, max_pos):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_pos, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.layers = nn.ModuleList([Attn(d, h) for _ in range(n_layers)])
        self.unemb = nn.Linear(d, vocab, bias=False)

    def forward(self, idx, need_att=False):
        x = self.emb(idx) + self.pos(torch.arange(idx.shape[1]))[None]
        atts = []
        for lyr in self.layers:
            y, a = lyr(x, need_att)
            atts.append(a)
            x = x + y
        return self.unemb(x), atts


# ----------------------------- data ----------------------------------------
def make_batch(rng, n, P, repeating=True):
    """Periodic sequences x[i] = x[i % p], first p tokens distinct.
    repeating=False -> i.i.d. tokens (control: induction is impossible)."""
    T, V = P["seq_len"], P["vocab"]
    if not repeating:
        ps = rng.integers(P["period_min"], P["period_max"] + 1, size=n)  # unused, keeps shape
        return torch.from_numpy(rng.integers(0, V, size=(n, T)).astype(np.int64)), ps
    ps = rng.integers(P["period_min"], P["period_max"] + 1, size=n)
    seq = np.zeros((n, T), dtype=np.int64)
    for i, p in enumerate(ps):
        base = rng.choice(V, size=p, replace=False)
        seq[i] = np.tile(base, int(np.ceil(T / p)))[:T]
    return torch.from_numpy(seq), ps


def eval_masks(ps, T):
    """Target index j = t+1 (predicted from query position t).
    first-occurrence: j <= p-1.  induction-solvable repeat: j >= p+1 (query t >= p).
    j == p is excluded: it is a repeat token but has no earlier occurrence of x[t] to
    induct from, so neither mask claims it."""
    j = np.arange(1, T)[None, :]                    # target indices, aligned to t = j-1
    first = j <= (ps[:, None] - 1)
    rep = j >= (ps[:, None] + 1)
    return torch.from_numpy(rep), torch.from_numpy(first)


# --------------------------- diagnostics ------------------------------------
def attention_scores(atts, ps, P):
    """prev-token score per layer-0 head; induction score per layer-1 head."""
    T = P["seq_len"]
    a0, a1 = atts[0], atts[1]                       # (B, H, T, T)
    B, H = a0.shape[0], a0.shape[1]
    t = torch.arange(T)

    # prev-token: mass at source t-1, for query t >= 1
    prev = a0[:, :, 1:, :].gather(-1, (t[1:] - 1).view(1, 1, -1, 1).expand(B, H, T - 1, 1))
    prev_score = prev.squeeze(-1).mean(dim=(0, 2))  # (H,)

    # induction: query t >= p attends to source t - p + 1
    pt = torch.from_numpy(ps)
    src = (t[None, :] - pt[:, None] + 1)            # (B, T)
    valid = (t[None, :] >= pt[:, None])             # (B, T)
    src_c = src.clamp(min=0)
    got = a1.gather(-1, src_c.view(B, 1, T, 1).expand(B, H, T, 1)).squeeze(-1)   # (B,H,T)
    vm = valid[:, None, :].expand(B, H, T)
    ind_score = (got * vm).sum(dim=(0, 2)) / vm.sum(dim=(0, 2))

    # uniform-attention reference values (causal, mass 1/(t+1) on any single source)
    unif_prev = float(np.mean([1.0 / (i + 1) for i in range(1, T)]))
    v_np = valid.numpy()
    denom = np.tile((t.numpy() + 1.0)[None, :], (B, 1))
    unif_ind = float((v_np / denom).sum() / v_np.sum())
    return prev_score.numpy(), ind_score.numpy(), unif_prev, unif_ind


def composition_scores(model, rots):
    """Normalized composition scores for every (layer0 head, layer1 head) pair, plus a
    null from random orthogonal rotations that preserve both singular-value spectra."""
    H = model.layers[0].h
    with torch.no_grad():
        QK1 = [model.layers[1].W_QK(h) for h in range(H)]
        OV1 = [model.layers[1].W_OV(h) for h in range(H)]
        OV0 = [model.layers[0].W_OV(h) for h in range(H)]
        out = {k: np.zeros((H, H)) for k in ("K", "Q", "V")}
        null = {k: np.zeros((H, H, len(rots))) for k in ("K", "Q", "V")}
        for h1 in range(H):
            A = {"K": QK1[h1], "Q": QK1[h1].T, "V": OV1[h1]}
            nA = {k: float(torch.linalg.norm(m)) for k, m in A.items()}
            for h0 in range(H):
                B = OV0[h0]
                nB = float(torch.linalg.norm(B))
                for k in ("K", "Q", "V"):
                    out[k][h0, h1] = float(torch.linalg.norm(A[k] @ B)) / (nA[k] * nB + 1e-12)
                    for r, R in enumerate(rots):
                        null[k][h0, h1, r] = float(torch.linalg.norm(A[k] @ (R @ B))) / (nA[k] * nB + 1e-12)
    return out, null


def make_rotations(n, d, seed):
    g = torch.Generator().manual_seed(seed)
    rots = []
    for _ in range(n):
        q, r = torch.linalg.qr(torch.randn(d, d, generator=g))
        rots.append(q * torch.sign(torch.diagonal(r))[None, :])
    return rots


# --------------------------- onset analysis ---------------------------------
def _interp_cross(steps, y, level, rising):
    """First step (linearly interpolated) at which y crosses `level`."""
    for i in range(1, len(y)):
        a, b = y[i - 1], y[i]
        if (rising and b >= level > a) or ((not rising) and b <= level < a):
            if b == a:
                return float(steps[i])
            f = (level - a) / (b - a)
            return float(steps[i - 1] + f * (steps[i] - steps[i - 1]))
    return None


def movement_onset(steps, y, frac, n_edge=3, rising=True):
    """Step at which y has completed `frac` of its total init->final movement."""
    init = float(np.mean(y[:n_edge]))
    fin = float(np.mean(y[-n_edge:]))
    if abs(fin - init) < 1e-9:
        return None, init, fin
    lvl = init + frac * (fin - init)
    return _interp_cross(steps, y, lvl, rising=(fin > init)), init, fin


def threshold_onset(steps, y, thr, sustain, rising=True):
    """ONLINE onset: first step where y crosses a fixed, pre-declared threshold and stays
    across `sustain` consecutive evals. Uses no information from the future."""
    y = np.asarray(y, dtype=float)
    run = 0
    for i, val in enumerate(y):
        ok = (val > thr) if rising else (val < thr)
        if ok:
            run += 1
            if run >= sustain:
                return float(steps[i - sustain + 1])
        else:
            run = 0
    return None


# ----------------------------- one run -------------------------------------
def train_one(seed, P, repeating, log, tag):
    set_seeds(seed)
    rng = np.random.default_rng(seed * 1000 + 7)
    erng = np.random.default_rng(P["eval_seed"])
    V, T = P["vocab"], P["seq_len"]

    # fixed eval batch: always the REPEATING distribution, even for the control model
    xe, pse = make_batch(erng, P["eval_n"], P, repeating=True)
    rep_mask, first_mask = eval_masks(pse, T)

    model = AttnOnlyLM(V, P["d_model"], P["n_heads"], P["n_layers"], T)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=P["lr"],
                            weight_decay=P["weight_decay"], betas=tuple(P["betas"]))
    rots = make_rotations(P["n_null_rotations"], P["d_model"], seed=1234)

    series = {k: [] for k in ("step", "loss_rep", "loss_first", "delta",
                              "prev_max", "ind_max", "compK_max", "train_loss")}
    series["prev_all"], series["ind_all"] = [], []
    series["compK_mat"], series["compQ_mat"], series["compV_mat"] = [], [], []
    for k in ("K", "Q", "V"):
        series[f"comp{k}_null_mu_mat"], series[f"comp{k}_null_sd_mat"] = [], []

    def do_eval(step, train_loss):
        model.eval()
        with torch.no_grad():
            logits, atts = model(xe, need_att=True)
            ls = F.cross_entropy(logits[:, :-1].reshape(-1, V), xe[:, 1:].reshape(-1),
                                 reduction="none").view(xe.shape[0], T - 1)
            lr_ = float(ls[rep_mask].mean())
            lf_ = float(ls[first_mask].mean())
            prev, ind, up, ui = attention_scores(atts, pse, P)
        comp, null = composition_scores(model, rots)
        model.train()
        series["unif_prev"], series["unif_ind"] = up, ui
        series["step"].append(step)
        series["train_loss"].append(train_loss)
        series["loss_rep"].append(lr_); series["loss_first"].append(lf_)
        series["delta"].append(lr_ - lf_)
        series["prev_all"].append(prev.tolist()); series["ind_all"].append(ind.tolist())
        series["prev_max"].append(float(prev.max())); series["ind_max"].append(float(ind.max()))
        series["compK_mat"].append(comp["K"].tolist())
        series["compQ_mat"].append(comp["Q"].tolist())
        series["compV_mat"].append(comp["V"].tolist())
        series["compK_max"].append(float(comp["K"].max()))
        for k in ("K", "Q", "V"):
            series[f"comp{k}_null_mu_mat"].append(null[k].mean(axis=-1).tolist())
            series[f"comp{k}_null_sd_mat"].append(null[k].std(axis=-1).tolist())

    eval_at = sorted(set(list(range(0, min(P["dense_eval_until"], P["steps"]) + 1,
                                    P["dense_eval_every"]))
                         + list(range(0, P["steps"] + 1, P["eval_every"]))
                         + [P["steps"]]))
    eval_at = set(eval_at)
    t0, capped, loss = time.time(), False, torch.tensor(float("nan"))
    for step in range(P["steps"] + 1):
        if step in eval_at:
            do_eval(step, float(loss.detach()) if step else float("nan"))
        if step == P["steps"]:
            break
        for g in opt.param_groups:
            g["lr"] = P["lr"] * min(1.0, (step + 1) / P["warmup"])
        x, _ = make_batch(rng, P["batch_size"], P, repeating=repeating)
        logits, _ = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if time.time() - t0 > P["time_cap_s_per_run"]:
            capped = True
            do_eval(step + 1, float(loss.detach()))
            break
    wall = time.time() - t0
    log(f"  [{tag}] params={n_params} steps={series['step'][-1]} ({wall:.0f}s"
        f"{' CAPPED' if capped else ''}) delta={series['delta'][-1]:+.3f} "
        f"prev={series['prev_max'][-1]:.3f} ind={series['ind_max'][-1]:.3f} "
        f"compK_max={series['compK_max'][-1]:.4f} "
        f"(null {np.mean(series['compK_null_mu_mat'][-1]):.4f})")
    return {"seed": seed, "repeating": repeating, "tag": tag, "n_params": n_params,
            "wall_s": round(wall, 1), "time_capped": capped, "series": series}


# --------------------------- per-run analysis -------------------------------
def analyse(run, P):
    S = run["series"]
    steps = np.array(S["step"], dtype=float)
    delta = np.array(S["delta"], dtype=float)
    H = P["n_heads"]

    # winning head pair, chosen from the FINAL model (not online-available; flagged)
    prev_fin = np.array(S["prev_all"][-1]); ind_fin = np.array(S["ind_all"][-1])
    h0 = int(prev_fin.argmax()); h1 = int(ind_fin.argmax())

    compK = np.array(S["compK_mat"])                      # (t, H, H)
    win = compK[:, h0, h1]
    others = np.array([[compK[t, a, b] for a in range(H) for b in range(H)
                        if not (a == h0 and b == h1)] for t in range(len(steps))])
    comp_max = np.array(S["compK_max"])
    prev_win = np.array([S["prev_all"][t][h0] for t in range(len(steps))])
    ind_win = np.array([S["ind_all"][t][h1] for t in range(len(steps))])

    # phase transition of the in-context loss delta (delta falls)
    t50, d_init, d_fin = movement_onset(steps, delta, 0.5)
    t10, _, _ = movement_onset(steps, delta, 0.1)
    t90, _, _ = movement_onset(steps, delta, 0.9)

    null_mu_m = np.array(S["compK_null_mu_mat"]); null_sd_m = np.array(S["compK_null_sd_mat"])
    null_mu, null_sd = null_mu_m[:, h0, h1], null_sd_m[:, h0, h1]
    z_win = (win - null_mu) / (null_sd + 1e-12)
    z_all = (compK - null_mu_m) / (null_sd_m + 1e-12)
    z_max = z_all.reshape(len(steps), -1).max(axis=1)

    meas = {"compK_winpair": win, "compK_maxpair": comp_max,
            "prev_token_score": prev_win, "induction_score": ind_win}

    # --- criteria A/B: fraction of total init->final movement (needs the final model) ---
    onset_mov = {}
    for frac in P["movement_fracs"]:
        onset_mov[frac] = {k: movement_onset(steps, y, frac)[0] for k, y in meas.items()}
    t_loss = {frac: movement_onset(steps, delta, frac)[0] for frac in P["movement_fracs"]}
    lead_mov = {frac: {k: (None if (onset_mov[frac][k] is None or t_loss[frac] is None)
                           else round(t_loss[frac] - onset_mov[frac][k], 1)) for k in meas}
                for frac in P["movement_fracs"]}

    # --- criterion C: ONLINE absolute thresholds, pre-declared, no future info ---
    su = P["onset_sustain"]
    thr_delta = -P["delta_frac_thresh"] * math.log(P["vocab"])
    thr_prev = P["attn_ratio_thresh"] * S["unif_prev"]
    thr_ind = P["attn_ratio_thresh"] * S["unif_ind"]
    onset_abs_ = {
        "compK_winpair": threshold_onset(steps, z_win, P["z_null_thresh"], su),
        "compK_maxpair": threshold_onset(steps, z_max, P["z_null_thresh"], su),
        "prev_token_score": threshold_onset(steps, prev_win, thr_prev, su),
        "induction_score": threshold_onset(steps, ind_win, thr_ind, su),
    }
    o_delta_abs = threshold_onset(steps, delta, thr_delta, su, rising=False)
    lead_abs = {k: (None if (onset_abs_[k] is None or o_delta_abs is None)
                    else round(o_delta_abs - onset_abs_[k], 1)) for k in meas}

    # truncation-artifact control: redo the 50%-of-movement criterion as if training had
    # STOPPED at the transition (this is what manufactured fake leads in the grokking run)
    if t50 is not None:
        cut = int(np.searchsorted(steps, t50) + 1)
        cut = max(cut, 6)
        st_c, dl_c = steps[:cut], delta[:cut]
        t50_c = movement_onset(st_c, dl_c, 0.5)[0]
        lead_mov_trunc = {}
        for k, y in meas.items():
            oc = movement_onset(st_c, y[:cut], 0.5)[0]
            lead_mov_trunc[k] = (None if (oc is None or t50_c is None) else round(t50_c - oc, 1))
    else:
        lead_mov_trunc = {k: None for k in meas}

    # Q- and V-composition of the same pair, each against its OWN rotation null
    # (specificity: only the KEY side should be reading the layer-0 head)
    zqv = {}
    for kind in ("Q", "V"):
        v = float(np.array(S[f"comp{kind}_mat"])[-1, h0, h1])
        mu = float(np.array(S[f"comp{kind}_null_mu_mat"])[-1, h0, h1])
        sd = float(np.array(S[f"comp{kind}_null_sd_mat"])[-1, h0, h1])
        zqv[kind] = (v, (v - mu) / (sd + 1e-12), mu)

    # specificity over time: is the early composition rise SPECIFIC to the winning pair,
    # or is it a broad rise shared by all 16 pairs?
    z_flat = z_all.reshape(len(steps), -1)
    z_mean_all = z_flat.mean(axis=1)
    others_z = np.array([[z_all[t, a, b] for a in range(H) for b in range(H)
                          if not (a == h0 and b == h1)] for t in range(len(steps))])
    at = (lambda y: None if t50 is None else round(float(np.interp(t50, steps, y)), 4))
    spec = {
        "compK_z_final_winpair": round(float(z_win[-1]), 2),
        "compK_z_final_max_over_pairs": round(float(z_flat[-1].max()), 2),
        "compK_z_final_mean_over_pairs": round(float(z_mean_all[-1]), 2),
        "compK_z_final_mean_other_pairs": round(float(others_z[-1].mean()), 2),
        "compK_z_at_transition_winpair": at(z_win),
        "compK_z_at_transition_mean_over_pairs": at(z_mean_all),
        "compK_winpair_minus_others_at_transition": (
            None if t50 is None else round(float(np.interp(t50, steps, win)
                                                 - np.interp(t50, steps, others.mean(axis=1))), 4)),
        "compK_winpair_minus_others_final": round(float(win[-1] - others[-1].mean()), 4),
    }

    # how far each measure has travelled at the moment the loss transition happens
    # (the grokking experiment's "frac_collapse_done_at_test50" number)
    frac_done = {}
    for k, y in meas.items():
        y = np.asarray(y, dtype=float)
        lo, hi = float(np.mean(y[:3])), float(np.mean(y[-3:]))
        if t50 is None or abs(hi - lo) < 1e-12:
            frac_done[k] = None
        else:
            frac_done[k] = round(float((np.interp(t50, steps, y) - lo) / (hi - lo)), 4)

    return {
        "tag": run["tag"], "seed": run["seed"], "repeating": run["repeating"],
        "winning_pair": {"layer0_head": h0, "layer1_head": h1},
        "final_prev_token_scores": [round(float(v), 4) for v in prev_fin],
        "final_induction_scores": [round(float(v), 4) for v in ind_fin],
        "delta_init": round(d_init, 4), "delta_final": round(d_fin, 4),
        "loss_rep_final": round(S["loss_rep"][-1], 4),
        "loss_first_final": round(S["loss_first"][-1], 4),
        "transition_step_50": None if t50 is None else round(t50, 1),
        "transition_step_10": None if t10 is None else round(t10, 1),
        "transition_step_90": None if t90 is None else round(t90, 1),
        "transition_width_10_90": (None if (t10 is None or t90 is None) else round(t90 - t10, 1)),
        "delta_abs_onset_step": o_delta_abs,
        "abs_thresholds": {"delta_nats": round(thr_delta, 4), "prev_token": round(thr_prev, 4),
                           "induction": round(thr_ind, 4), "comp_z_vs_null": P["z_null_thresh"],
                           "uniform_prev_baseline": round(S["unif_prev"], 4),
                           "uniform_induction_baseline": round(S["unif_ind"], 4)},
        "onset_movement": {str(f): {k: (None if v is None else round(v, 1))
                                    for k, v in onset_mov[f].items()} for f in P["movement_fracs"]},
        "loss_onset_movement": {str(f): (None if t_loss[f] is None else round(t_loss[f], 1))
                                for f in P["movement_fracs"]},
        "onset_absolute": {k: (None if v is None else round(v, 1)) for k, v in onset_abs_.items()},
        "lead_steps_movement": {str(f): lead_mov[f] for f in P["movement_fracs"]},
        "lead_steps_absolute": lead_abs,
        "lead_steps_movement50_truncated_at_transition": lead_mov_trunc,
        "compK_winpair_init": round(float(win[0]), 4),
        "compK_winpair_final": round(float(win[-1]), 4),
        "compK_null_mean_final": round(float(null_mu[-1]), 4),
        "compK_winpair_z_vs_null_final": round(float(z_win[-1]), 2),
        "compK_winpair_z_vs_null_init": round(float(z_win[0]), 2),
        "compK_other_pairs_mean_final": round(float(others[-1].mean()), 4),
        "specificity": spec,
        "compQ_winpair_final": round(float(zqv["Q"][0]), 4),
        "compV_winpair_final": round(float(zqv["V"][0]), 4),
        "compQ_winpair_z_vs_null_final": round(float(zqv["Q"][1]), 2),
        "compV_winpair_z_vs_null_final": round(float(zqv["V"][1]), 2),
        "frac_of_movement_done_at_transition": frac_done,
        "_arrays": {"steps": steps, "delta": delta, "win": win, "others": others,
                    "comp_max": comp_max, "prev": prev_win, "ind": ind_win,
                    "null_mu": null_mu, "null_sd": null_sd, "z_win": z_win,
                    "t50": t50, "t10": t10, "t90": t90,
                    "onset_mov": onset_mov[0.5], "onset_abs": onset_abs_,
                    "o_delta_abs": o_delta_abs},
    }


# ----------------------------- main ----------------------------------------
def main():
    cfg = load_config()
    P = cfg["params"]
    set_seeds(int(cfg.get("seed", 0)))
    t0 = time.time()
    log = lambda s: print(s, flush=True)
    log(f"induction-heads-emergence: {len(P['seeds'])} seeds"
        f"{' + i.i.d. control' if P['run_control'] else ''}")

    runs = [train_one(s, P, True, log, f"seed{s}") for s in P["seeds"]]
    if P["run_control"]:
        runs.append(train_one(P["seeds"][0], P, False, log, "control_iid"))

    an = [analyse(r, P) for r in runs]
    main_an = [a for a in an if a["repeating"]]
    ctrl = next((a for a in an if not a["repeating"]), None)

    def agg(getter):
        vals = [getter(a) for a in main_an]
        vals = [v for v in vals if v is not None]
        return (round(float(np.mean(vals)), 1), [v for v in vals]) if vals else (None, [])

    trans_mean, trans_all = agg(lambda a: a["transition_step_50"])
    width_mean, width_all = agg(lambda a: a["transition_width_10_90"])
    keys = ["compK_winpair", "compK_maxpair", "prev_token_score", "induction_score"]
    lead_mov_mean = {k: agg(lambda a, k=k: a["lead_steps_movement"]["0.5"][k]) for k in keys}
    lead_mov10_mean = {k: agg(lambda a, k=k: a["lead_steps_movement"]["0.1"][k]) for k in keys}
    lead_abs_mean = {k: agg(lambda a, k=k: a["lead_steps_absolute"][k]) for k in keys}
    lead_trunc_mean = {k: agg(lambda a, k=k: a["lead_steps_movement50_truncated_at_transition"][k])
                       for k in keys}

    comp_leads_mov = lead_mov_mean["compK_winpair"][0] is not None and lead_mov_mean["compK_winpair"][0] > 0
    comp_leads_abs = lead_abs_mean["compK_winpair"][0] is not None and lead_abs_mean["compK_winpair"][0] > 0

    metrics = {
        "n_params": runs[0]["n_params"],
        "steps": P["steps"], "eval_every": P["eval_every"], "eval_n": P["eval_n"],
        "seeds": P["seeds"], "vocab": P["vocab"], "seq_len": P["seq_len"],
        "period_range": [P["period_min"], P["period_max"]],
        "log_vocab_nats": round(math.log(P["vocab"]), 4),
        "phase_transition_step_mean": trans_mean,
        "phase_transition_step_per_seed": trans_all,
        "transition_width_10_90_mean": width_mean,
        "transition_width_10_90_per_seed": width_all,
        "transition_width_frac_of_training": (None if width_mean is None
                                              else round(width_mean / P["steps"], 4)),
        "delta_init_mean": round(float(np.mean([a["delta_init"] for a in main_an])), 4),
        "delta_final_mean": round(float(np.mean([a["delta_final"] for a in main_an])), 4),
        "loss_rep_final_mean": round(float(np.mean([a["loss_rep_final"] for a in main_an])), 4),
        "loss_first_final_mean": round(float(np.mean([a["loss_first_final"] for a in main_an])), 4),
        "lead_steps_movement50": {k: {"mean": v[0], "per_seed": v[1]} for k, v in lead_mov_mean.items()},
        "lead_steps_movement10": {k: {"mean": v[0], "per_seed": v[1]} for k, v in lead_mov10_mean.items()},
        "lead_steps_absolute_online": {k: {"mean": v[0], "per_seed": v[1]} for k, v in lead_abs_mean.items()},
        "lead_steps_movement50_truncated_control": {k: {"mean": v[0], "per_seed": v[1]}
                                                    for k, v in lead_trunc_mean.items()},
        "onset_steps_absolute_online": {k: [a["onset_absolute"][k] for a in main_an] for k in keys},
        "loss_onset_step_absolute_online": [a["delta_abs_onset_step"] for a in main_an],
        "abs_thresholds": main_an[0]["abs_thresholds"],
        "composition_leads_under_movement50": bool(comp_leads_mov),
        "composition_leads_under_absolute_online": bool(comp_leads_abs),
        "measures_that_lead_movement50": [k for k in keys
                                          if lead_mov_mean[k][0] is not None and lead_mov_mean[k][0] > 0],
        "measures_that_lead_absolute_online": [k for k in keys
                                               if lead_abs_mean[k][0] is not None and lead_abs_mean[k][0] > 0],
        "compK_winpair_final_mean": round(float(np.mean([a["compK_winpair_final"] for a in main_an])), 4),
        "compK_winpair_init_mean": round(float(np.mean([a["compK_winpair_init"] for a in main_an])), 4),
        "compK_null_mean_final": round(float(np.mean([a["compK_null_mean_final"] for a in main_an])), 4),
        "compK_winpair_z_vs_null_final_mean": round(float(np.mean(
            [a["compK_winpair_z_vs_null_final"] for a in main_an])), 2),
        "specificity_mean": {k: round(float(np.mean([a["specificity"][k] for a in main_an])), 3)
                             for k in main_an[0]["specificity"]},
        "compQ_winpair_final_mean": round(float(np.mean([a["compQ_winpair_final"] for a in main_an])), 4),
        "compV_winpair_final_mean": round(float(np.mean([a["compV_winpair_final"] for a in main_an])), 4),
        "compQ_winpair_z_vs_null_final_mean": round(float(np.mean(
            [a["compQ_winpair_z_vs_null_final"] for a in main_an])), 2),
        "compV_winpair_z_vs_null_final_mean": round(float(np.mean(
            [a["compV_winpair_z_vs_null_final"] for a in main_an])), 2),
        "frac_of_movement_done_at_transition": {
            k: round(float(np.mean([a["frac_of_movement_done_at_transition"][k] for a in main_an])), 4)
            for k in keys},
        "per_run": [{k: v for k, v in a.items() if k != "_arrays"} for a in an],
        "wall_seconds_per_run": {r["tag"]: r["wall_s"] for r in runs},
    }
    if ctrl is not None:
        c = ctrl["_arrays"]
        metrics["control_iid"] = {
            "delta_final": ctrl["delta_final"],
            "compK_winpair_final": ctrl["compK_winpair_final"],
            "compK_null_mean_final": ctrl["compK_null_mean_final"],
            "compK_winpair_z_vs_null_final": ctrl["compK_winpair_z_vs_null_final"],
            "compK_max_final": round(float(c["comp_max"][-1]), 4),
            "compK_z_final_max_over_pairs": ctrl["specificity"]["compK_z_final_max_over_pairs"],
            "compK_z_final_mean_over_pairs": ctrl["specificity"]["compK_z_final_mean_over_pairs"],
            "prev_token_score_final": round(float(c["prev"][-1]), 4),
            "induction_score_final": round(float(c["ind"][-1]), 4),
            "compK_abs_onset_step": ctrl["onset_absolute"]["compK_winpair"],
            "compK_maxpair_abs_onset_step": ctrl["onset_absolute"]["compK_maxpair"],
            "prev_token_abs_onset_step": ctrl["onset_absolute"]["prev_token_score"],
            "induction_abs_onset_step": ctrl["onset_absolute"]["induction_score"],
            "delta_abs_onset_step": ctrl["delta_abs_onset_step"],
            "false_positive_composition": ctrl["onset_absolute"]["compK_maxpair"] is not None,
        }

    lm = metrics["lead_steps_movement50"]; la = metrics["lead_steps_absolute_online"]
    metrics["headline"] = (
        f"in-context loss delta {metrics['delta_init_mean']:+.2f} -> {metrics['delta_final_mean']:+.2f} nats, "
        f"phase transition at step {trans_mean} (10-90% width {width_mean} steps = "
        f"{metrics['transition_width_frac_of_training']:.0%} of training); "
        f"K-composition of the winning pair leads by {lm['compK_winpair']['mean']} steps under the "
        f"50%-of-movement criterion and {la['compK_winpair']['mean']} steps under the online "
        f"absolute-threshold criterion; prev-token score {lm['prev_token_score']['mean']} / "
        f"{la['prev_token_score']['mean']}, induction score {lm['induction_score']['mean']} / "
        f"{la['induction_score']['mean']} (positive = leads)")

    # ----------------------------- chart ------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = {"delta": "#1f2d3d", "comp": "#c95d3c", "prev": "#1a7f64", "ind": "#3d5a80",
         "null": "#b0b0b0", "ctrl": "#8a817c"}
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.6))

    # (1) the phase transition
    ax = axes[0][0]
    for i, a in enumerate(main_an):
        A = a["_arrays"]
        ax.plot(A["steps"], A["delta"], lw=1.8, alpha=0.9,
                color=C["delta"], ls=["-", "--", ":"][i % 3], label=f"seed {a['seed']}")
        if A["t50"]:
            ax.axvline(A["t50"], color=C["comp"], lw=0.8, alpha=0.5)
    if ctrl is not None:
        ax.plot(ctrl["_arrays"]["steps"], ctrl["_arrays"]["delta"], color=C["ctrl"], lw=1.6,
                label="control (i.i.d. data)")
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xlabel("training step"); ax.set_ylabel("in-context loss delta (nats)")
    ax.set_title(f"1. Induction bump: loss(repeat) - loss(first occurrence)\n"
                 f"transition at step {trans_mean}, 10-90% width {width_mean}", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)

    # (2) all measures normalized, seed 0
    a0 = main_an[0]; A = a0["_arrays"]
    ax = axes[0][1]

    def nrm(y):
        y = np.asarray(y, dtype=float)
        lo, hi = np.mean(y[:3]), np.mean(y[-3:])
        return (y - lo) / (hi - lo) if abs(hi - lo) > 1e-12 else y * 0

    ax.plot(A["steps"], nrm(-A["delta"]), color=C["delta"], lw=2.4, label="in-context loss drop")
    ax.plot(A["steps"], nrm(A["win"]), color=C["comp"], lw=1.8, label="K-composition (win pair)")
    ax.plot(A["steps"], nrm(A["prev"]), color=C["prev"], lw=1.8, label="prev-token score (L0)")
    ax.plot(A["steps"], nrm(A["ind"]), color=C["ind"], lw=1.8, label="induction score (L1)")
    for nm, key, col in (("loss", None, C["delta"]), ("comp", "compK_winpair", C["comp"]),
                         ("prev", "prev_token_score", C["prev"]), ("ind", "induction_score", C["ind"])):
        v = A["t50"] if key is None else A["onset_mov"][key]
        if v:
            ax.axvline(v, color=col, lw=1.0, ls="--", alpha=0.7)
    ax.axhline(0.5, color="0.85", lw=0.8, zorder=0)
    ax.set_xlabel("training step"); ax.set_ylabel("fraction of total movement")
    ax.set_ylim(-0.15, 1.25)
    ax.set_title("2. Who moves first? (seed %d, dashed = 50%%-of-movement onset)" % a0["seed"],
                 fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left"); ax.spines[["top", "right"]].set_visible(False)

    # (3) composition in absolute units against its null
    ax = axes[0][2]
    ax.plot(A["steps"], A["win"], color=C["comp"], lw=2.0, label="winning pair (h0*,h1*)")
    ax.plot(A["steps"], A["others"].mean(axis=1), color="#e0a080", lw=1.4, label="other 15 pairs (mean)")
    ax.plot(A["steps"], A["null_mu"], color=C["null"], lw=1.4, label="rotation null (mean)")
    ax.fill_between(A["steps"], A["null_mu"] - 3 * A["null_sd"], A["null_mu"] + 3 * A["null_sd"],
                    color=C["null"], alpha=0.3, label="null +/- 3 sd")
    if A["t50"]:
        ax.axvline(A["t50"], color=C["delta"], lw=1.2, ls="--", label="loss transition")
    ax.set_xlabel("training step"); ax.set_ylabel(r"$\|W_{QK}^{(1)}W_{OV}^{(0)}\|_F$ (normalized)")
    ax.set_title("3. K-composition strength vs spectrum-preserving null", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)

    # (4) lead bars, movement criterion
    ax = axes[1][0]
    labs = ["K-comp\n(win pair)", "K-comp\n(max pair)", "prev-token\nscore", "induction\nscore"]
    xs = np.arange(len(keys))
    ax.bar(xs, [lead_mov_mean[k][0] or 0 for k in keys],
           color=[C["comp"], "#e0a080", C["prev"], C["ind"]])
    for i, k in enumerate(keys):
        for v in lead_mov_mean[k][1]:
            ax.plot(i, v, "k.", ms=6, alpha=0.6)
        if lead_mov_mean[k][0] is not None:
            ax.text(i, lead_mov_mean[k][0], f"{lead_mov_mean[k][0]:+.0f}", ha="center",
                    va="bottom" if lead_mov_mean[k][0] >= 0 else "top", fontsize=9)
    ax.axhline(0, color="0.3", lw=1)
    ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel("lead over loss transition (steps)")
    ax.set_title("4. Lead, 50%-of-movement criterion (>0 = leads)", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)

    # (5) lead bars, online absolute criterion
    ax = axes[1][1]
    ax.bar(xs, [lead_abs_mean[k][0] or 0 for k in keys],
           color=[C["comp"], "#e0a080", C["prev"], C["ind"]])
    for i, k in enumerate(keys):
        for v in lead_abs_mean[k][1]:
            ax.plot(i, v, "k.", ms=6, alpha=0.6)
        if lead_abs_mean[k][0] is not None:
            ax.text(i, lead_abs_mean[k][0], f"{lead_abs_mean[k][0]:+.0f}", ha="center",
                    va="bottom" if lead_abs_mean[k][0] >= 0 else "top", fontsize=9)
    ax.axhline(0, color="0.3", lw=1)
    ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel("lead over loss onset (steps)")
    ax.set_title(f"5. Lead, ONLINE absolute thresholds (comp z>{P['z_null_thresh']:.0f} vs null;\n"
                 f"attn > {P['attn_ratio_thresh']:.0f}x uniform; loss < -{P['delta_frac_thresh']:.0%} log V)",
                 fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)

    # (6) head-pair composition heatmap, final
    ax = axes[1][2]
    Hh = P["n_heads"]
    final_mat = np.array(runs[0]["series"]["compK_mat"][-1])
    im = ax.imshow(final_mat, cmap="OrRd", vmin=final_mat.min(), vmax=final_mat.max())
    ax.set_xticks(range(Hh)); ax.set_yticks(range(Hh))
    ax.set_xlabel("layer-1 head (induction side)"); ax.set_ylabel("layer-0 head (prev-token side)")
    h0w, h1w = a0["winning_pair"]["layer0_head"], a0["winning_pair"]["layer1_head"]
    ax.add_patch(plt.Rectangle((h1w - 0.5, h0w - 0.5), 1, 1, fill=False, edgecolor="#1a7f64", lw=2.5))
    for i in range(Hh):
        for j in range(Hh):
            ax.text(j, i, f"{final_mat[i, j]:.3f}", ha="center", va="center", fontsize=7.5,
                    color="black")
    ax.set_title(f"6. Final K-composition per head pair (seed {a0['seed']});\n"
                 f"green box = (prev-token h{h0w}, induction h{h1w})", fontsize=9.5)
    fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Induction-head emergence in a 2-layer attention-only transformer "
                 f"({runs[0]['n_params']:,} params): does K-composition lead the phase transition?",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(HERE / "chart.png", dpi=150, bbox_inches="tight")

    # ----------------------------- write ------------------------------------
    for a in an:
        a.pop("_arrays", None)

    def shrink(o, nd=5):
        """Round the logged series so results.json stays a sane size (no information
        loss at these magnitudes: losses ~5, attention ~0.5, composition ~0.2)."""
        if isinstance(o, float):
            return round(o, nd)
        if isinstance(o, list):
            return [shrink(v, nd) for v in o]
        if isinstance(o, dict):
            return {k: shrink(v, nd) for k, v in o.items()}
        return o
    results = {
        "id": cfg.get("id", "unknown"),
        "git_commit": git_sha(),
        "seed": int(cfg.get("seed", 0)),
        "duration_sec": round(time.time() - t0, 2),
        "metrics": metrics,
        "series": {r["tag"]: shrink({k: v for k, v in r["series"].items()
                                     if k not in ("compQ_mat", "compV_mat", "compQ_null_mu_mat",
                                                  "compQ_null_sd_mat", "compV_null_mu_mat",
                                                  "compV_null_sd_mat")}) for r in runs},
        "env": env_info(),
        "status": "done",
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: results[k] for k in ("id", "duration_sec", "status")}, indent=2))
    print("headline:", metrics["headline"])


if __name__ == "__main__":
    main()
