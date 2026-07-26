"""Trained (PonderNet-style) halting vs stochastic depth for LENGTH generalization.

Convergence point of four earlier registry rows:
  2026-07-25_loop-test-time-compute   -- stochastic depth k~U{1..K} was THE fix for DEPTH
                                         extrapolation; fixed-K training degrades past K_train.
  2026-07-25_shadow-halt-entropy-tiny -- INFERENCE-TIME entropy halting was worthless
                                         (indistinguishable from a compute-matched coin flip),
                                         but the halting was never TRAINED.
  2026-07-25_pe-length-gen-tiny       -- at 0.1M params ALiBi > RoPE > NoPE > APE for length gen.
  2026-07-25_filler-vs-recur          -- loops saturate on serial tasks without intermediate
                                         supervision.

Question: does TRAINED halting (a learned per-step, per-position halt probability with the
PonderNet expected-loss weighting and a KL to a geometric prior) succeed where inference-time
entropy halting failed -- and does it match or beat the stochastic-depth recipe on LENGTH
(not just depth) generalization?

Task (n-RASP-L style): prefix-parity / cumulative XOR. Input [BOS, x_1..x_L], target
y_i = XOR(x_1..x_i). Attention is banded causal with WINDOW 1: every position attends only to
itself and its immediate predecessor. Therefore after K applications of the tied block position i
has only seen bits [i-K, i], so

    position i is solvable IFF K >= i - 1,

i.e. the number of required sequential steps is EXACTLY the position index, and the required
loop count for a length-L input is exactly L-1. Length generalization and depth generalization
are the same axis by construction, which is what makes "does the halting head learn to loop MORE
on longer inputs?" a question with an analytic ground truth.

Arms (matched architecture, ~0.0586M params; the ponder arms add a 65-param halt head):
  (a) fixed_K7      -- trained at a FIXED K=7
  (b) stoch_depth   -- k ~ U{1..7} per batch with L = k+1 (the proven recipe)
  (c) ponder        -- PonderNet halting head, N=7 unrolled steps, expected-loss + KL(p||Geom)
  (d) ponder_stoch  -- (c) but with a stochastic unroll horizon N ~ U{L-1..7}

Deterministic, CPU-only, single-threaded. Writes results.json + chart.png.
Usage:  python run.py
"""
import json, math, os, random, subprocess, sys, time
from pathlib import Path

# --- must be set before torch spins up its thread pools (2 shared cores) ---
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)


# ----------------------------- housekeeping --------------------------------
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
    for mod in ("numpy", "torch", "matplotlib"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:
            pass
    return info


# ----------------------------- data ----------------------------------------
def make_batch(n, L, gen, bos=2):
    """tokens (n, L+1) = [BOS, x_1..x_L];  targets (n, L) = cumulative XOR."""
    bits = torch.randint(0, 2, (n, L), generator=gen)
    tgt = torch.cumsum(bits, dim=1) % 2
    tok = torch.cat([torch.full((n, 1), bos, dtype=torch.long), bits], dim=1)
    return tok, tgt


# ----------------------------- model ---------------------------------------
def band_mask(T, window):
    """(T, T) bool, True = allowed. Position i attends to [i-window, i]."""
    idx = torch.arange(T)
    d = idx[:, None] - idx[None, :]
    return (d >= 0) & (d <= window)


class Block(nn.Module):
    """Pre-LN transformer block with a banded causal attention window, NoPE."""

    def __init__(self, d, h, dff):
        super().__init__()
        self.h, self.dh = h, d // h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.fc1, self.fc2 = nn.Linear(d, dff), nn.Linear(dff, d)

    def forward(self, x, mask):
        B, T, D = x.shape
        z = self.ln1(x)
        q, k, v = self.qkv(z).split(D, dim=2)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (self.dh ** 0.5)
        att = att.masked_fill(~mask[None, None, :T, :T], float("-inf"))
        att = att.softmax(dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(y)
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class LoopModel(nn.Module):
    """One weight-tied block, applied K times with per-iteration input injection.
    Optional PonderNet halt head reading the hidden state after every iteration."""

    def __init__(self, p, with_halt, lambda_prior):
        super().__init__()
        d, h, dff = p["d_model"], p["n_heads"], p["d_ff"]
        self.window = p["attn_window"]
        self.emb = nn.Embedding(p["vocab"], d)
        self.block = Block(d, h, dff)
        self.adapter = nn.Linear(2 * d, d, bias=False) if p["inject"] else None
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, 2)
        self.halt = nn.Linear(d, 1) if with_halt else None
        if self.halt is not None:
            # start at the prior halt rate so the model does not halt at step 1 from init
            nn.init.zeros_(self.halt.weight)
            with torch.no_grad():
                self.halt.bias.fill_(math.log(lambda_prior / (1.0 - lambda_prior)))

    def unroll(self, tok, K):
        """Returns (logits_per_step, lam_per_step) for steps 1..K.
        logits_per_step[n]: (B, T, 2);  lam_per_step[n]: (B, T) or None."""
        T = tok.shape[1]
        mask = band_mask(T, self.window)
        e = self.emb(tok)
        hh = e
        logits, lams = [], []
        for _ in range(K):
            inp = self.adapter(torch.cat([hh, e], dim=-1)) if self.adapter is not None else hh
            hh = self.block(inp, mask)
            logits.append(self.head(self.ln_f(hh)))
            if self.halt is not None:
                lams.append(torch.sigmoid(self.halt(hh)).squeeze(-1).clamp(1e-6, 1 - 1e-6))
            else:
                lams.append(None)
        return logits, lams


# ----------------------------- ponder machinery ----------------------------
def halt_distribution(lams):
    """lams: list of length N of (B, L) halt probs. Returns p: (N, B, L) summing to 1 over N,
    with the last step absorbing the remaining mass (PonderNet's truncated unroll)."""
    N = len(lams)
    ps, notyet = [], torch.ones_like(lams[0])
    for n in range(N):
        if n < N - 1:
            ps.append(lams[n] * notyet)
            notyet = notyet * (1.0 - lams[n])
        else:
            ps.append(notyet)
    return torch.stack(ps, dim=0)


def geometric_prior(N, lam_p, device):
    """Truncated + renormalised Geom(lam_p) over steps 1..N. Returns (N,)."""
    n = torch.arange(N, dtype=torch.float32, device=device)
    pr = lam_p * (1.0 - lam_p) ** n
    return pr / pr.sum()


# ----------------------------- training ------------------------------------
def train_arm(kind, p, seed, log):
    set_seeds(seed)
    with_halt = kind.startswith("ponder")
    model = LoopModel(p, with_halt, p["ponder_lambda_prior"])
    n_params = sum(t.numel() for t in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    gen = torch.Generator().manual_seed(seed * 1000 + 7)
    steps, K_tr = p["steps"], p["k_train"]
    Lmin, Lmax = p["train_len_min"], p["train_len_max"]
    lam_p, beta = p["ponder_lambda_prior"], p["ponder_beta"]

    t0 = time.time()
    losses, done_steps = [], 0
    for step in range(steps):
        if step < p["warmup"]:
            lr = p["lr"] * (step + 1) / p["warmup"]
        else:
            prog = (step - p["warmup"]) / max(1, steps - p["warmup"])
            lr = p["lr"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))
        for g in opt.param_groups:
            g["lr"] = lr

        if kind == "stochastic":
            k = int(torch.randint(1, K_tr + 1, (1,), generator=gen).item())
            L, N = k + 1, k
        elif kind in ("fixed", "ponder"):
            L = int(torch.randint(Lmin, Lmax + 1, (1,), generator=gen).item())
            N = K_tr
        elif kind == "ponder_stochastic":
            L = int(torch.randint(Lmin, Lmax + 1, (1,), generator=gen).item())
            N = max(1, int(torch.randint(L - 1, K_tr + 1, (1,), generator=gen).item()))
        else:
            raise ValueError(kind)

        tok, tgt = make_batch(p["batch_size"], L, gen)
        logits, lams = model.unroll(tok, N)

        if with_halt:
            ce = torch.stack([F.cross_entropy(lg[:, 1:, :].reshape(-1, 2), tgt.reshape(-1),
                                              reduction="none").view(tgt.shape) for lg in logits], 0)
            pd = halt_distribution([l[:, 1:] for l in lams])          # (N, B, L)
            l_rec = (pd * ce).sum(0).mean()
            prior = geometric_prior(N, lam_p, pd.device).view(N, 1, 1)
            l_reg = (pd * (torch.log(pd.clamp_min(1e-9)) - torch.log(prior))).sum(0).mean()
            loss = l_rec + beta * l_reg
        else:
            loss = F.cross_entropy(logits[-1][:, 1:, :].reshape(-1, 2), tgt.reshape(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), p["grad_clip"])
        opt.step()
        losses.append(float(loss.item()))
        done_steps = step + 1
        if time.time() - t0 > p["time_cap_s_per_run"]:
            log(f"      TIME CAP hit at step {done_steps}/{steps}")
            break

    return model, {
        "n_params": n_params,
        "steps_done": done_steps,
        "train_seconds": round(time.time() - t0, 1),
        "final_loss": round(float(np.mean(losses[-50:])), 6),
        "hit_time_cap": done_steps < steps,
    }


# ----------------------------- evaluation ----------------------------------
def first_true_step(flag):
    """flag: (K, B, L) bool. Returns (B, L) long = 1-indexed first True step, else K."""
    K = flag.shape[0]
    idx = torch.arange(K, dtype=torch.long).view(-1, 1, 1).expand_as(flag)
    picked = torch.where(flag, idx, torch.full_like(idx, K - 1))
    return picked.min(dim=0).values + 1


@torch.no_grad()
def evaluate(model, p, kind):
    """One length-L_max eval batch gives every shorter length EXACTLY: attention is causal, so
    the prediction at position i depends only on tokens 1..i and is identical for any total
    length >= i (verified by prefix_consistency_check)."""
    model.eval()
    Lm, Kmax = p["test_len_max"], p["k_max_test"]
    gen = torch.Generator().manual_seed(p["eval_seed"])
    tok, tgt = make_batch(p["eval_n"], Lm, gen)
    logits, lams = model.unroll(tok, Kmax)
    preds = torch.stack([lg[:, 1:, :].argmax(-1) for lg in logits], 0)      # (Kmax, B, Lm)
    correct = (preds == tgt[None]).float()                                  # (Kmax, B, Lm)

    out = {}
    tok_acc, em = {}, {}
    for K in p["test_k_values"]:
        c = correct[K - 1]
        tok_acc[K] = {L: round(float(c[:, :L].mean()), 4) for L in p["test_lengths"]}
        em[K] = {L: round(float((c[:, :L].min(dim=1).values > 0.5).float().mean()), 4)
                 for L in p["test_lengths"]}
    out["fixed_k_token_acc"] = tok_acc
    out["fixed_k_exact_match"] = em
    out["per_position_acc_at_kmax"] = [round(float(correct[Kmax - 1, :, i].mean()), 4) for i in range(Lm)]
    # accuracy at the analytically REQUIRED depth for position i+1 (i.e. K = i, min 1)
    out["per_position_acc_at_required_k"] = [
        round(float(correct[max(1, i) - 1, :, i].mean()), 4) for i in range(Lm)]

    if kind.startswith("ponder"):
        lam = torch.stack([l[:, 1:] for l in lams], 0)                      # (Kmax, B, Lm)
        thr = p["halt_threshold"]
        pdist = halt_distribution([lam[n] for n in range(Kmax)])
        cum = 1.0 - torch.cumprod(1.0 - lam, dim=0)
        gen_s = torch.Generator().manual_seed(p["eval_seed"] + 1)
        u = torch.rand(lam.shape, generator=gen_s)
        pol = {"cum_thresh": first_true_step(cum >= thr),                   # median of p(halt)
               "lambda_thresh": first_true_step(lam >= thr),
               "argmax_p": pdist.argmax(0) + 1,
               "sampled": first_true_step(u < lam)}
        nn_ = torch.arange(1, Kmax + 1, dtype=torch.float32).view(-1, 1, 1)
        exp_steps = (pdist * nn_).sum(0)                                    # (B, Lm)

        out["mean_lambda_per_step"] = [round(float(lam[n].mean()), 4) for n in range(Kmax)]
        out["mean_lambda_step1_by_position"] = [round(float(lam[0, :, i].mean()), 4) for i in range(Lm)]
        out["expected_steps_by_position"] = [round(float(exp_steps[:, i].mean()), 3) for i in range(Lm)]
        out["policies"] = {n: policy_stats(hs, correct, p) for n, hs in pol.items()}

        # ---- the decisive control (mirrors 2026-07-25_shadow-halt-entropy-tiny) ----
        # (i) sweep the halting threshold to trace an accuracy-vs-compute frontier for the
        #     LEARNED signal; (ii) trace the same frontier for a compute-matched RANDOM exit
        #     (geometric halting, no information about difficulty at all).
        learned, rand = [], []
        for t in p["threshold_sweep"]:
            hs = first_true_step(cum >= t)
            learned.append(frontier_point(hs, correct, p, {"threshold": t}))
        gen_r = torch.Generator().manual_seed(p["eval_seed"] + 2)
        for q in p["random_exit_q"]:
            ur = torch.rand(lam.shape, generator=gen_r)
            hs = first_true_step(ur < q)
            rand.append(frontier_point(hs, correct, p, {"q": q}))
        out["frontier_learned"] = learned
        out["frontier_random"] = rand
    return out


def policy_stats(hs, correct, p):
    Kmax = p["k_max_test"]
    hs_f = hs.float()
    idx = (hs - 1).clamp(0, Kmax - 1).long()
    c = torch.gather(correct, 0, idx.unsqueeze(0))[0]
    return {
        "token_acc": {L: round(float(c[:, :L].mean()), 4) for L in p["test_lengths"]},
        "exact_match": {L: round(float((c[:, :L].min(dim=1).values > 0.5).float().mean()), 4)
                        for L in p["test_lengths"]},
        "mean_loops_by_length": {L: round(float(hs_f[:, :L].mean()), 3) for L in p["test_lengths"]},
        "mean_halt_step_by_position": [round(float(hs_f[:, i].mean()), 3) for i in range(p["test_len_max"])],
        "frac_at_kmax": round(float((hs == Kmax).float().mean()), 4),
        "halt_step_std": round(float(hs_f.std()), 3),
    }


def frontier_point(hs, correct, p, tag):
    Kmax = p["k_max_test"]
    idx = (hs - 1).clamp(0, Kmax - 1).long()
    c = torch.gather(correct, 0, idx.unsqueeze(0))[0]
    pt = dict(tag)
    pt["mean_loops_by_length"] = {L: round(float(hs.float()[:, :L].mean()), 3) for L in p["test_lengths"]}
    pt["token_acc"] = {L: round(float(c[:, :L].mean()), 4) for L in p["test_lengths"]}
    pt["exact_match"] = {L: round(float((c[:, :L].min(dim=1).values > 0.5).float().mean()), 4)
                         for L in p["test_lengths"]}
    return pt


@torch.no_grad()
def prefix_consistency_check(model, p):
    """Verify that evaluating a length-L_max batch really reproduces a genuine short batch."""
    gen = torch.Generator().manual_seed(p["eval_seed"])
    tok, _ = make_batch(64, p["test_len_max"], gen)
    K = p["k_train"]
    long_logits, _ = model.unroll(tok, K)
    short_logits, _ = model.unroll(tok[:, : p["train_len_max"] + 1], K)
    a = long_logits[-1][:, 1: p["train_len_max"] + 1, :].argmax(-1)
    b = short_logits[-1][:, 1:, :].argmax(-1)
    return bool((a == b).all())


# ----------------------------- main ----------------------------------------
PRIMARY_POLICY = "cum_thresh"


def main():
    global PRIMARY_POLICY
    t_start = time.time()
    cfg = load_config()
    p = cfg["params"]
    PRIMARY_POLICY = p["primary_policy"]
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    log(f"=== {cfg['id']} ===")
    log(f"task: prefix-parity, window {p['attn_window']} -> position i needs K >= i-1 loops")
    log(f"train lengths U{{{p['train_len_min']}..{p['train_len_max']}}} (K_train={p['k_train']}), "
        f"test lengths up to {p['test_len_max']} (needs K={p['test_len_max']-1}), "
        f"K_max_test={p['k_max_test']}")

    runs = {}
    for arm in p["arms"]:
        name, kind = arm["name"], arm["kind"]
        runs[name] = {"kind": kind, "note": arm["note"], "seeds": {}}
        for seed in p["seeds"]:
            log(f"  [{name}] seed {seed} ...")
            model, tr = train_arm(kind, p, seed, log)
            ev = evaluate(model, p, kind)
            ev["prefix_consistency_ok"] = prefix_consistency_check(model, p)
            runs[name]["seeds"][seed] = {"train": tr, "eval": ev}
            best_in = max(ev["fixed_k_token_acc"][K][p["train_len_max"]] for K in p["test_k_values"])
            LN = p["test_len_max"]
            ln = (f"      {tr['train_seconds']}s loss={tr['final_loss']:.4f} "
                  f"in-dist(L={p['train_len_max']}) bestK acc={best_in:.3f} "
                  f"| L={LN}: K7={ev['fixed_k_token_acc'][p['k_train']][LN]:.3f} "
                  f"Kmax={ev['fixed_k_token_acc'][p['k_max_test']][LN]:.3f}")
            if kind.startswith("ponder"):
                pl = ev["policies"][PRIMARY_POLICY]
                ln += f" ADAPT={pl['token_acc'][LN]:.3f} (loops {pl['mean_loops_by_length'][LN]:.2f})"
            log(ln)

    # ----------------------- aggregate across seeds -------------------------
    agg = {}
    Ls, Ks = p["test_lengths"], p["test_k_values"]
    Lm, Kmax, Ktr = p["test_len_max"], p["k_max_test"], p["k_train"]
    for name, r in runs.items():
        sd = list(r["seeds"].values())
        a = {"kind": r["kind"], "note": r["note"],
             "n_params": sd[0]["train"]["n_params"],
             "steps_done": sd[0]["train"]["steps_done"],
             "train_seconds": round(float(np.mean([s["train"]["train_seconds"] for s in sd])), 1),
             "final_loss": round(float(np.mean([s["train"]["final_loss"] for s in sd])), 6),
             "prefix_consistency_ok": all(s["eval"]["prefix_consistency_ok"] for s in sd)}
        for metric in ("fixed_k_token_acc", "fixed_k_exact_match"):
            a[metric] = {K: {L: round(float(np.mean([s["eval"][metric][K][L] for s in sd])), 4)
                             for L in Ls} for K in Ks}
        a["oracle_bestK_token_acc"] = {L: max(a["fixed_k_token_acc"][K][L] for K in Ks) for L in Ls}
        a["oracle_bestK"] = {L: int(max(Ks, key=lambda K: a["fixed_k_token_acc"][K][L])) for L in Ls}
        a["oracle_bestK_exact_match"] = {L: max(a["fixed_k_exact_match"][K][L] for K in Ks) for L in Ls}
        a["token_acc_at_Ktrain"] = {L: a["fixed_k_token_acc"][Ktr][L] for L in Ls}
        a["token_acc_at_Kmax"] = {L: a["fixed_k_token_acc"][Kmax][L] for L in Ls}
        a["exact_match_at_Kmax"] = {L: a["fixed_k_exact_match"][Kmax][L] for L in Ls}
        a["per_position_acc_at_kmax"] = [round(float(np.mean(
            [s["eval"]["per_position_acc_at_kmax"][i] for s in sd])), 4) for i in range(Lm)]
        a["per_position_acc_at_required_k"] = [round(float(np.mean(
            [s["eval"]["per_position_acc_at_required_k"][i] for s in sd])), 4) for i in range(Lm)]
        a["token_acc_per_seed_at_longest_Kmax"] = [s["eval"]["fixed_k_token_acc"][Kmax][Ls[-1]] for s in sd]
        if r["kind"].startswith("ponder"):
            a["mean_lambda_per_step"] = [round(float(np.mean(
                [s["eval"]["mean_lambda_per_step"][n] for s in sd])), 4) for n in range(Kmax)]
            a["expected_steps_by_position"] = [round(float(np.mean(
                [s["eval"]["expected_steps_by_position"][i] for s in sd])), 3) for i in range(Lm)]
            a["policies"] = {}
            for pol in sd[0]["eval"]["policies"]:
                pa = {}
                for m in ("token_acc", "exact_match", "mean_loops_by_length"):
                    pa[m] = {L: round(float(np.mean([s["eval"]["policies"][pol][m][L] for s in sd])), 4)
                             for L in Ls}
                pa["mean_halt_step_by_position"] = [round(float(np.mean(
                    [s["eval"]["policies"][pol]["mean_halt_step_by_position"][i] for s in sd])), 3)
                    for i in range(Lm)]
                pa["frac_at_kmax"] = round(float(np.mean(
                    [s["eval"]["policies"][pol]["frac_at_kmax"] for s in sd])), 4)
                pa["halt_step_std"] = round(float(np.mean(
                    [s["eval"]["policies"][pol]["halt_step_std"] for s in sd])), 3)
                y = np.array(pa["mean_halt_step_by_position"], dtype=float)
                x = np.arange(1, Lm + 1, dtype=float)
                pa["halt_step_vs_position_slope"] = round(float(np.polyfit(x, y, 1)[0]), 4)
                pa["halt_step_pos1"] = float(y[0])
                pa["halt_step_pos_max"] = float(y[-1])
                pa["token_acc_per_seed_at_longest"] = [
                    s["eval"]["policies"][pol]["token_acc"][Ls[-1]] for s in sd]
                a["policies"][pol] = pa
            # --- frontiers, seed-averaged ---
            for fk in ("frontier_learned", "frontier_random"):
                npts = len(sd[0]["eval"][fk])
                a[fk] = []
                for j in range(npts):
                    pt = {k: v for k, v in sd[0]["eval"][fk][j].items()
                          if k in ("threshold", "q")}
                    for m in ("mean_loops_by_length", "token_acc", "exact_match"):
                        pt[m] = {L: round(float(np.mean([s["eval"][fk][j][m][L] for s in sd])), 4)
                                 for L in Ls}
                    a[fk].append(pt)
            # --- decisive: learned vs compute-matched random exit at the longest length ---
            Lz = Ls[-1]
            rx = np.array([q["mean_loops_by_length"][Lz] for q in a["frontier_random"]])
            ry = np.array([q["token_acc"][Lz] for q in a["frontier_random"]])
            o = np.argsort(rx)
            rx, ry = rx[o], ry[o]
            cmp_pts = []
            for q in a["frontier_learned"]:
                mx = q["mean_loops_by_length"][Lz]
                if rx.min() <= mx <= rx.max():
                    cmp_pts.append({"threshold": q["threshold"], "mean_loops": mx,
                                    "learned_acc": q["token_acc"][Lz],
                                    "random_acc_matched": round(float(np.interp(mx, rx, ry)), 4),
                                    "delta": round(float(q["token_acc"][Lz] - np.interp(mx, rx, ry)), 4)})
            a["learned_vs_random_at_longest"] = cmp_pts
            a["max_learned_minus_random_at_longest"] = (
                round(max(c["delta"] for c in cmp_pts), 4) if cmp_pts else None)
            a["mean_learned_minus_random_at_longest"] = (
                round(float(np.mean([c["delta"] for c in cmp_pts])), 4) if cmp_pts else None)
        agg[name] = a

    # ----------------------- headline ---------------------------------------
    Lmax = Ls[-1]
    head = {"longest_test_length": Lmax, "required_loops_at_longest": Lmax - 1,
            "train_len_max": p["train_len_max"], "k_train": Ktr,
            "token_acc_at_longest": {}, "exact_match_at_longest": {}}
    for name, a in agg.items():
        head["token_acc_at_longest"][name] = {
            "at_Ktrain": a["token_acc_at_Ktrain"][Lmax],
            "at_Kmax": a["token_acc_at_Kmax"][Lmax],
            "oracle_bestK": a["oracle_bestK_token_acc"][Lmax],
            "oracle_bestK_value": a["oracle_bestK"][Lmax]}
        head["exact_match_at_longest"][name] = {
            "at_Kmax": a["exact_match_at_Kmax"][Lmax],
            "oracle_bestK": a["oracle_bestK_exact_match"][Lmax]}
        if "policies" in a:
            pl = a["policies"][PRIMARY_POLICY]
            head["token_acc_at_longest"][name]["adaptive"] = pl["token_acc"][Lmax]
            head["token_acc_at_longest"][name]["adaptive_mean_loops"] = pl["mean_loops_by_length"][Lmax]
            head["token_acc_at_longest"][name]["adaptive_all_policies"] = {
                q: a["policies"][q]["token_acc"][Lmax] for q in a["policies"]}
            head["exact_match_at_longest"][name]["adaptive"] = pl["exact_match"][Lmax]

    # --- ISO-COMPUTE comparison: is adaptive halting worth it at MATCHED mean loops? ---
    # The adaptive arms spend fewer loops than K_max. Compare each adaptive operating point
    # against EVERY arm's fixed-K frontier interpolated to the same mean loops per position.
    iso = {}
    for name, a in agg.items():
        if "policies" not in a:
            continue
        iso[name] = {}
        for L in Ls:
            m = a["policies"][PRIMARY_POLICY]["mean_loops_by_length"][L]
            ad = a["policies"][PRIMARY_POLICY]["token_acc"][L]
            row = {"adaptive_mean_loops": m, "adaptive_token_acc": ad, "vs_fixed_K": {}}
            for other, b in agg.items():
                y = [b["fixed_k_token_acc"][K][L] for K in Ks]
                v = float(np.interp(m, Ks, y))
                row["vs_fixed_K"][other] = {"fixedK_acc_at_matched_compute": round(v, 4),
                                            "delta": round(ad - v, 4)}
            iso[name][L] = row
    for name in iso:
        agg[name]["iso_compute_vs_fixed_K"] = iso[name]

    ponders = [n for n in agg if agg[n]["kind"].startswith("ponder")]
    head["primary_adaptive_policy"] = PRIMARY_POLICY
    best_ponder = max(ponders, key=lambda n: agg[n]["policies"][PRIMARY_POLICY]["token_acc"][Lmax])
    sd_acc = agg["stoch_depth"]["oracle_bestK_token_acc"][Lmax]
    ad_acc = agg[best_ponder]["policies"][PRIMARY_POLICY]["token_acc"][Lmax]
    head["best_adaptive_arm"] = best_ponder
    head["stoch_depth_oracle_token_acc_at_longest"] = sd_acc
    head["best_adaptive_token_acc_at_longest"] = ad_acc
    head["adaptive_minus_stochdepth_token_acc_at_longest"] = round(ad_acc - sd_acc, 4)
    head["halt_step_vs_position_slope"] = {
        n: agg[n]["policies"][PRIMARY_POLICY]["halt_step_vs_position_slope"] for n in ponders}
    head["required_slope"] = 1.0
    head["adaptive_learns_to_loop_more_on_longer_inputs"] = {
        n: bool(v > 0.05) for n, v in head["halt_step_vs_position_slope"].items()}
    head["learned_halt_vs_compute_matched_random_at_longest"] = {
        n: {"max_delta": agg[n]["max_learned_minus_random_at_longest"],
            "mean_delta": agg[n]["mean_learned_minus_random_at_longest"]} for n in ponders}
    head["frac_positions_running_to_kmax"] = {
        n: agg[n]["policies"][PRIMARY_POLICY]["frac_at_kmax"] for n in ponders}
    # the two-sided verdict: adaptive loses on absolute ceiling, wins at matched compute
    head["adaptive_minus_stochdepth_fixedK_at_MATCHED_compute_at_longest"] = {
        n: agg[n]["iso_compute_vs_fixed_K"][Lmax]["vs_fixed_K"]["stoch_depth"]["delta"]
        for n in ponders}
    head["adaptive_minus_own_fixedK_at_MATCHED_compute_at_longest"] = {
        n: agg[n]["iso_compute_vs_fixed_K"][Lmax]["vs_fixed_K"][n]["delta"] for n in ponders}

    log("")
    log(f"HEADLINE @ L={Lmax} (needs {Lmax-1} loops);  adaptive policy = {PRIMARY_POLICY}")
    for n, v in head["token_acc_at_longest"].items():
        extra = (f"  ADAPTIVE={v['adaptive']:.3f} (mean loops {v['adaptive_mean_loops']:.2f})"
                 if "adaptive" in v else "")
        log(f"  {n:14s} K_train={v['at_Ktrain']:.3f}  K_max={v['at_Kmax']:.3f}  "
            f"oracle-bestK={v['oracle_bestK']:.3f} (K={v['oracle_bestK_value']}){extra}")
    log(f"  adaptive({best_ponder}) - stoch_depth(oracle) = "
        f"{head['adaptive_minus_stochdepth_token_acc_at_longest']:+.4f}")
    log(f"  halt-step-vs-position slope: {head['halt_step_vs_position_slope']} (required 1.0)")
    log(f"  learned halt vs compute-matched RANDOM exit (max/mean delta): "
        f"{head['learned_halt_vs_compute_matched_random_at_longest']}")
    log(f"  ADAPTIVE vs stoch_depth fixed-K at MATCHED mean compute: "
        f"{head['adaptive_minus_stochdepth_fixedK_at_MATCHED_compute_at_longest']}")

    # write results FIRST so a plotting bug can never destroy a completed run
    results = {"id": cfg["id"], "title": cfg["title"], "date": "2026-07-26",
               "git_sha": git_sha(), "env": env_info(), "config": cfg,
               "wall_clock_seconds": round(time.time() - t_start, 1),
               "headline": head, "arms": agg, "per_seed": runs, "log": log_lines}
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    make_chart(agg, p, head)
    log(f"\nwrote results.json + chart.png in {results['wall_clock_seconds']}s")


def make_chart(agg, p, head):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Ls, Lm, Kmax, Ktr = p["test_lengths"], p["test_len_max"], p["k_max_test"], p["k_train"]
    colors = {"fixed_K7": "#c0392b", "stoch_depth": "#2980b9",
              "ponder": "#27ae60", "ponder_stoch": "#8e44ad"}
    fig, ax = plt.subplots(2, 4, figsize=(22, 9))
    pos = np.arange(1, Lm + 1)
    Lz = Ls[-1]

    a0 = ax[0, 0]
    for n, a in agg.items():
        a0.plot(Ls, [a["oracle_bestK_token_acc"][L] for L in Ls], "-o", ms=4,
                color=colors[n], label=f"{n} (oracle best-K)")
        if "policies" in a:
            a0.plot(Ls, [a["policies"][PRIMARY_POLICY]["token_acc"][L] for L in Ls], "--s", ms=4,
                    color=colors[n], alpha=.85, label=f"{n} ADAPTIVE")
    a0.axvline(p["train_len_max"], color="k", ls=":", lw=1)
    a0.text(p["train_len_max"] + .25, .52, "train max len", fontsize=8, rotation=90)
    a0.axhline(.5, color="gray", lw=.8, ls="--")
    a0.set_xlabel("test length L"); a0.set_ylabel("token accuracy")
    a0.set_title("Token accuracy vs test LENGTH\n(solid = best fixed K per length, oracle)")
    a0.legend(fontsize=7); a0.grid(alpha=.3); a0.set_ylim(.45, 1.02)

    a1 = ax[0, 1]
    for n, a in agg.items():
        a1.plot(Ls, [a["oracle_bestK_exact_match"][L] for L in Ls], "-o", ms=4,
                color=colors[n], label=f"{n} (oracle best-K)")
        if "policies" in a:
            a1.plot(Ls, [a["policies"][PRIMARY_POLICY]["exact_match"][L] for L in Ls], "--s", ms=4,
                    color=colors[n], alpha=.85, label=f"{n} ADAPTIVE")
    a1.axvline(p["train_len_max"], color="k", ls=":", lw=1)
    a1.set_xlabel("test length L"); a1.set_ylabel("whole-sequence exact match")
    a1.set_title("Exact match vs test LENGTH")
    a1.legend(fontsize=7); a1.grid(alpha=.3)

    a2 = ax[0, 2]
    a2.plot(Ls, [L - 1 for L in Ls], "k-", lw=2, label="required loops (L-1)")
    for n, a in agg.items():
        if "policies" in a:
            a2.plot(Ls, [a["policies"][PRIMARY_POLICY]["mean_loops_by_length"][L] for L in Ls],
                    "--s", ms=4, color=colors[n], label=f"{n} learned halt")
        else:
            a2.plot(Ls, [Ktr] * len(Ls), ":", color=colors[n], label=f"{n} (trained K={Ktr})")
    a2.axvline(p["train_len_max"], color="k", ls=":", lw=1)
    a2.set_xlabel("test length L"); a2.set_ylabel("mean loops used per position")
    a2.set_title("Does trained halting spend MORE loops\non longer inputs?")
    a2.legend(fontsize=7); a2.grid(alpha=.3)

    # frontier: accuracy vs compute at the longest test length -- learned halting vs a
    # compute-matched RANDOM exit vs the fixed-K frontier of the same model
    af = ax[0, 3]
    for n, a in agg.items():
        if "policies" not in a:
            af.plot(p["test_k_values"], [a["fixed_k_token_acc"][K][Lz] for K in p["test_k_values"]],
                    ":", lw=1.4, color=colors[n], alpha=.9, label=f"{n} fixed-K frontier")
            continue
        lx = [q["mean_loops_by_length"][Lz] for q in a["frontier_learned"]]
        ly = [q["token_acc"][Lz] for q in a["frontier_learned"]]
        rx = [q["mean_loops_by_length"][Lz] for q in a["frontier_random"]]
        ry = [q["token_acc"][Lz] for q in a["frontier_random"]]
        af.plot(lx, ly, "-o", ms=4, color=colors[n], label=f"{n} LEARNED halt (thr sweep)")
        af.plot(rx, ry, "--x", ms=5, color=colors[n], alpha=.55,
                label=f"{n} compute-matched RANDOM exit")
        af.plot(p["test_k_values"], [a["fixed_k_token_acc"][K][Lz] for K in p["test_k_values"]],
                ":", lw=1.2, color=colors[n], alpha=.8, label=f"{n} fixed-K frontier")
    af.axhline(.5, color="gray", lw=.8, ls="--")
    af.set_xlabel("mean loops used per position"); af.set_ylabel(f"token accuracy at L={Lz}")
    af.set_title(f"DECISIVE CONTROL: does the LEARNED halt signal\nbeat a coin flip at matched compute? (L={Lz})")
    af.legend(fontsize=6); af.grid(alpha=.3); af.set_ylim(.45, 1.02)

    a3 = ax[1, 0]
    a3.plot(pos, pos - 1, "k-", lw=2, label="required loops (i-1)")
    for n, a in agg.items():
        if "policies" in a:
            a3.plot(pos, a["policies"][PRIMARY_POLICY]["mean_halt_step_by_position"], "--s", ms=4,
                    color=colors[n], label=f"{n} halt step ({PRIMARY_POLICY})")
            a3.plot(pos, a["expected_steps_by_position"], ":", lw=1.2, color=colors[n], alpha=.7,
                    label=f"{n} E[steps]")
    a3.axvline(p["train_len_max"], color="k", ls=":", lw=1)
    a3.set_xlabel("position index i (= difficulty)"); a3.set_ylabel("loops")
    a3.set_title("Halt step vs position (difficulty)")
    a3.legend(fontsize=7); a3.grid(alpha=.3)

    a4 = ax[1, 1]
    for n, a in agg.items():
        a4.plot(pos, a["per_position_acc_at_kmax"], "-o", ms=3, color=colors[n],
                label=f"{n} @K={Kmax}")
    a4.axvline(p["train_len_max"], color="k", ls=":", lw=1)
    a4.axhline(.5, color="gray", lw=.8, ls="--")
    a4.set_xlabel("position index i"); a4.set_ylabel("accuracy")
    a4.set_title(f"Per-position accuracy at max test compute (K={Kmax})")
    a4.legend(fontsize=7); a4.grid(alpha=.3); a4.set_ylim(.4, 1.02)

    al = ax[1, 2]
    for n, a in agg.items():
        if "policies" in a:
            al.plot(range(1, Kmax + 1), a["mean_lambda_per_step"], "-o", ms=4, color=colors[n],
                    label=f"{n} mean lambda_n")
    al.axhline(p["ponder_lambda_prior"], color="k", ls="--", lw=1,
               label=f"geometric prior lambda={p['ponder_lambda_prior']}")
    al.axhline(p["halt_threshold"], color="gray", ls=":", lw=1, label="halt threshold 0.5")
    al.axvline(Ktr, color="k", ls=":", lw=1)
    al.set_xlabel("loop step n"); al.set_ylabel("mean learned halt prob lambda_n")
    al.set_title("Learned per-step halting rate\n(collapse to max depth = lambda stays low)")
    al.legend(fontsize=7); al.grid(alpha=.3)

    a5 = ax[1, 3]
    Lmax = Ls[-1]
    names, vals, cols = [], [], []
    for n, a in agg.items():
        names.append(f"{n}\noracle K"); vals.append(a["oracle_bestK_token_acc"][Lmax]); cols.append(colors[n])
        if "policies" in a:
            names.append(f"{n}\nADAPTIVE")
            vals.append(a["policies"][PRIMARY_POLICY]["token_acc"][Lmax]); cols.append(colors[n])
    a5.bar(range(len(vals)), vals, color=cols, alpha=.85)
    for i, v in enumerate(vals):
        a5.text(i, v + .005, f"{v:.3f}", ha="center", fontsize=8)
    a5.set_xticks(range(len(names))); a5.set_xticklabels(names, fontsize=7)
    a5.axhline(.5, color="gray", lw=.8, ls="--")
    a5.set_ylabel("token accuracy"); a5.set_ylim(.45, 1.02)
    a5.set_title(f"HEADLINE: token accuracy at L={Lmax} ({Lmax-1} loops required)")
    a5.grid(alpha=.3, axis="y")

    fig.suptitle("Trained (PonderNet) halting vs stochastic depth for LENGTH generalization -- "
                 "prefix-parity, window-1 attention, one weight-tied block (0.059M params)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(HERE / "chart.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
