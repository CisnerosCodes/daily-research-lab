"""Nano Continuous Thought Machine: is the synchronization readout the signal,
or is it just recurrence depth?

Ablation ladder at matched params / matched internal ticks / matched steps, on
cumulative parity of an L-bit string presented ALL AT ONCE (the CTM paper's own
showcase task; the model must iterate internally, there is no input sequence):

  (a) ctm_sync   -- neuron-level models (a private MLP per neuron over a short
                    history of that neuron's pre-activations) + synchronization
                    readout (outputs from pairwise products of neuron activation
                    histories across internal ticks, with a learned decay)
  (b) ctm_lastz  -- same recurrence + same neuron-level models, but the readout
                    is a plain linear map of the LAST-TICK post-activations
                    (kills synchronization, keeps everything else)
  (c) gru_lastz  -- plain GRU with last-tick readout at matched param count
                    (kills both neuron-level models and synchronization)
  (d) gru_sync   -- synchronization readout bolted onto the plain GRU
                    (adds sync without neuron-level models)

Plus a depth control: arms (a) and (b) re-run at T=2 internal ticks, to check
whether the ladder is about internal *time* at all.

Deterministic, CPU-only (1 thread), writes results.json + chart.png.
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
    for mod in ("numpy", "torch", "matplotlib"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:
            pass
    return info


# ----------------------------- setup ---------------------------------------
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)


# ----------------------------- data ----------------------------------------
def parity_batch(rng, n, L):
    """x in {-1,+1}^L presented all at once; target = CUMULATIVE parity at every
    index (y_k = parity of bits 0..k), as in the CTM paper's parity task.
    Headline metric is y_{L-1} = parity of the whole string."""
    bits = rng.integers(0, 2, size=(n, L)).astype(np.int64)      # 0/1
    y = np.cumsum(bits, axis=1) % 2                               # (n, L) in {0,1}
    x = (bits * 2 - 1).astype(np.float32)                         # +-1
    return torch.from_numpy(x), torch.from_numpy(y)


# ----------------------------- readouts ------------------------------------
class SyncReadout(nn.Module):
    """Synchronization-matrix readout. For P sampled neuron pairs (i,j) track the
    decayed running inner product of their activation histories over internal
    ticks and read the output off that vector:

        num_t = r_ij * num_{t-1} + z_i^t * z_j^t
        den_t = r_ij * den_{t-1} + 1
        S_t   = num_t / sqrt(den_t)          -> Linear(P, n_out)

    r_ij = exp(-clamp(alpha_ij, 0, 15)); alpha init 0 => r = 1 (plain running mean
    over ticks) and learnable, matching the CTM formulation.
    """

    def __init__(self, n_units, n_pairs, n_out, gen):
        super().__init__()
        self.register_buffer("idx_i", torch.randint(0, n_units, (n_pairs,), generator=gen))
        self.register_buffer("idx_j", torch.randint(0, n_units, (n_pairs,), generator=gen))
        self.alpha = nn.Parameter(torch.zeros(n_pairs))
        self.head = nn.Linear(n_pairs, n_out)
        self.n_pairs = n_pairs
        self.num = self.den = None

    def reset(self, B, device):
        self.num = torch.zeros(B, self.n_pairs, device=device)
        self.den = torch.zeros(B, self.n_pairs, device=device)

    def forward(self, z):
        r = torch.exp(-self.alpha.clamp(0.0, 15.0))
        self.num = r * self.num + z[:, self.idx_i] * z[:, self.idx_j]
        self.den = r * self.den + 1.0
        return self.head(self.num / torch.sqrt(self.den))


class SyncNowReadout(nn.Module):
    """Mechanism control: the same P pairwise PRODUCTS z_i^t * z_j^t, but read off
    the CURRENT tick only -- no accumulation over internal time, no decay. This
    keeps the sync readout's quadratic (bilinear) feature expansion while removing
    the cross-tick 'neuron timing synchronization' part entirely. It has n_pairs
    fewer parameters than SyncReadout (no decay vector, 0.2% of the model)."""

    def __init__(self, n_units, n_pairs, n_out, gen):
        super().__init__()
        self.register_buffer("idx_i", torch.randint(0, n_units, (n_pairs,), generator=gen))
        self.register_buffer("idx_j", torch.randint(0, n_units, (n_pairs,), generator=gen))
        self.head = nn.Linear(n_pairs, n_out)

    def reset(self, B, device):
        pass

    def forward(self, z):
        return self.head(z[:, self.idx_i] * z[:, self.idx_j])


class LastReadout(nn.Module):
    """Plain linear readout of the current hidden state (no cross-tick term)."""

    def __init__(self, n_units, n_out):
        super().__init__()
        self.head = nn.Linear(n_units, n_out)

    def reset(self, B, device):
        pass

    def forward(self, z):
        return self.head(z)


# ----------------------------- models --------------------------------------
class NanoCTM(nn.Module):
    """CTM ingredients: (1) a synapse MLP producing pre-activations from the
    previous post-activations + the input, (2) per-neuron MLPs over a length-M
    history of that neuron's pre-activations, (3) a readout (sync or last-tick)."""

    def __init__(self, L, D, H_syn, M, H_nlm, readout):
        super().__init__()
        self.D, self.M = D, M
        self.syn = nn.Sequential(
            nn.Linear(D + L, H_syn), nn.LayerNorm(H_syn), nn.GELU(),
            nn.Linear(H_syn, D))
        # neuron-level models: one private 2-layer MLP (M -> H_nlm -> 1) per neuron
        self.w1 = nn.Parameter(torch.randn(D, M, H_nlm) / math.sqrt(M))
        self.b1 = nn.Parameter(torch.zeros(D, H_nlm))
        self.w2 = nn.Parameter(torch.randn(D, H_nlm) / math.sqrt(H_nlm))
        self.b2 = nn.Parameter(torch.zeros(D))
        self.start_trace = nn.Parameter(torch.zeros(D, M))
        self.start_z = nn.Parameter(torch.zeros(D))
        self.readout = readout

    def nlm(self, A):                       # A: (B, D, M) -> (B, D)
        h = torch.einsum("bdm,dmh->bdh", A, self.w1) + self.b1
        h = F.gelu(h)
        return torch.einsum("bdh,dh->bd", h, self.w2) + self.b2

    def forward(self, x, T):
        B = x.shape[0]
        A = self.start_trace.unsqueeze(0).expand(B, -1, -1)
        z = self.start_z.unsqueeze(0).expand(B, -1)
        self.readout.reset(B, x.device)
        outs = []
        for _ in range(T):
            a = self.syn(torch.cat([z, x], dim=1))            # (B, D) pre-activations
            A = torch.cat([A[:, :, 1:], a.unsqueeze(-1)], dim=2)
            z = self.nlm(A)                                   # (B, D) post-activations
            outs.append(self.readout(z))
        return torch.stack(outs, dim=1)                       # (B, T, n_out)


class NanoGRU(nn.Module):
    """Matched-size plain GRU fed the same static input at every internal tick."""

    def __init__(self, L, Hh, readout):
        super().__init__()
        self.cell = nn.GRUCell(L, Hh)
        self.h0 = nn.Parameter(torch.zeros(Hh))
        self.readout = readout

    def forward(self, x, T):
        B = x.shape[0]
        h = self.h0.unsqueeze(0).expand(B, -1)
        self.readout.reset(B, x.device)
        outs = []
        for _ in range(T):
            h = self.cell(x, h)
            outs.append(self.readout(h))
        return torch.stack(outs, dim=1)


# --------------------- analytic param counts (for matching) -----------------
def ctm_param_count(L, P):
    D, Hs = P["ctm_neurons"], P["ctm_synapse_hidden"]
    M, Hn = P["ctm_memory"], P["ctm_nlm_hidden"]
    syn = (D + L) * Hs + Hs + 2 * Hs + Hs * D + D     # linear + LayerNorm + linear
    nlm = D * (M * Hn + Hn + Hn + 1)
    start = D * M + D
    head = D * (2 * L) + 2 * L                        # identical shape for both readouts
    return syn + nlm + start + head


def gru_param_count(L, h):
    gru = 3 * (h * h + L * h + 2 * h)                 # GRUCell w_ih, w_hh, b_ih, b_hh
    head = h * (2 * L) + 2 * L
    return gru + h + head                             # + learnable h0


def build(arm, L, P, seed):
    """Returns (model, n_params). Widths chosen so every arm at a given L has
    (near-)identical parameter count; sync arms carry only +n_pairs extra decay
    parameters, since n_pairs == n_units keeps the readout matrix the same shape."""
    gen = torch.Generator().manual_seed(seed * 7919 + 13)
    n_out = 2 * L
    def mk(n_units):
        if arm.endswith("syncnow"):
            return SyncNowReadout(n_units, n_units, n_out, gen)
        if arm.endswith("sync"):
            return SyncReadout(n_units, n_units, n_out, gen)
        return LastReadout(n_units, n_out)

    D = P["ctm_neurons"]
    if arm.startswith("ctm"):
        m = NanoCTM(L, D, P["ctm_synapse_hidden"], P["ctm_memory"],
                    P["ctm_nlm_hidden"], mk(D))
    else:
        target = ctm_param_count(L, P)
        h = min(range(8, 400), key=lambda k: abs(gru_param_count(L, k) - target))
        m = NanoGRU(L, h, mk(h))
    return m, sum(p.numel() for p in m.parameters())


# ----------------------------- train / eval --------------------------------
@torch.no_grad()
def evaluate(model, T, L, n, seed_eval):
    rng = np.random.default_rng(seed_eval)
    x, y = parity_batch(rng, n, L)
    logits = model(x, T).view(x.shape[0], T, L, 2)           # (B, T, L, 2)
    pred = logits.argmax(-1)                                  # (B, T, L)
    correct = (pred == y.unsqueeze(1))                        # (B, T, L)
    full_by_tick = correct[:, :, -1].float().mean(0)          # parity of ALL L bits
    allpos_by_tick = correct.float().mean(dim=(0, 2))
    seq_by_tick = correct.all(-1).float().mean(0)             # every prefix right
    # CTM-style certainty selection: per sample, take the tick with lowest entropy
    probs = logits.softmax(-1)
    ent = -(probs * probs.clamp_min(1e-9).log()).sum(-1).mean(-1)    # (B, T)
    best_t = ent.argmin(dim=1)
    b = torch.arange(x.shape[0])
    cert_full = correct[b, best_t, -1].float().mean()
    return {
        "full_parity_last_tick": float(full_by_tick[-1]),
        "full_parity_best_tick": float(full_by_tick.max()),
        "full_parity_argmax_tick": int(full_by_tick.argmax()) + 1,
        "full_parity_certainty_tick": float(cert_full),
        "allpos_last_tick": float(allpos_by_tick[-1]),
        "allpos_best_tick": float(allpos_by_tick.max()),
        "seq_exact_last_tick": float(seq_by_tick[-1]),
        "full_parity_by_tick": [round(float(v), 4) for v in full_by_tick],
        "allpos_by_tick": [round(float(v), 4) for v in allpos_by_tick],
    }


def train_one(arm, L, T, seed, P, deadline, log):
    set_seeds(seed)
    model, n_params = build(arm, L, P, seed)
    rng = np.random.default_rng(seed * 100003 + L * 17 + 5)
    opt = torch.optim.AdamW(model.parameters(), lr=P["lr"],
                            weight_decay=P["weight_decay"])
    t0, capped, step = time.time(), False, 0
    losses = []
    for step in range(P["steps"]):
        lr = P["lr"] * min(1.0, (step + 1) / P["warmup"])
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = parity_batch(rng, P["batch_size"], L)
        logits = model(x, T).view(x.shape[0], T, L, 2)
        loss = F.cross_entropy(logits.reshape(-1, 2),
                               y.unsqueeze(1).expand(-1, T, -1).reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            losses.append(round(float(loss.detach()), 4))
        if time.time() > deadline or time.time() - t0 > P["time_cap_s_per_run"]:
            capped = True
            break
    train_s = time.time() - t0
    model.eval()
    ev = evaluate(model, T, L, P["eval_n"], P["eval_seed"])
    rec = {"arm": arm, "L": L, "T": T, "seed": seed, "n_params": n_params,
           "steps_run": step + 1, "time_capped": capped,
           "train_seconds": round(train_s, 1),
           "final_loss": round(float(loss.detach()), 4),
           "loss_trace": losses, **ev}
    log(f"  {arm:10s} L={L:2d} T={T:2d} s{seed} p={n_params} "
        f"steps={step + 1}{' CAP' if capped else ''} {train_s:5.1f}s  "
        f"full={ev['full_parity_last_tick']:.3f} "
        f"(best-tick {ev['full_parity_best_tick']:.3f}) "
        f"allpos={ev['allpos_last_tick']:.3f}")
    return rec


# ----------------------------- main ----------------------------------------
def main():
    cfg = load_config()
    P = cfg["params"]
    set_seeds(int(cfg.get("seed", 0)))
    t0 = time.time()
    log = lambda s: print(s, flush=True)
    global_deadline = t0 + P["total_time_budget_s"]

    runs = []
    log("== main ladder (T=%d) ==" % P["ticks"])
    for L in P["lengths"]:
        for arm in P["arms"]:
            for seed in P["seeds"]:
                runs.append(train_one(arm, L, P["ticks"], seed, P, global_deadline, log))

    depth_runs = []
    if time.time() < global_deadline - P["depth_control_reserve_s"]:
        log("== depth control (T=%d) ==" % P["ticks_shallow"])
        for arm in P["depth_control_arms"]:
            depth_runs.append(train_one(arm, P["depth_control_length"],
                                        P["ticks_shallow"], P["seeds"][0], P,
                                        global_deadline, log))
    else:
        log("== depth control SKIPPED (out of time budget) ==")

    mech_runs = []
    if time.time() < global_deadline - P["mech_control_reserve_s"]:
        log("== mechanism control: instantaneous pairwise products, no cross-tick "
            "accumulation (T=%d) ==" % P["ticks"])
        for arm in P["mech_control_arms"]:
            for seed in P["seeds"]:
                mech_runs.append(train_one(arm, P["mech_control_length"], P["ticks"],
                                           seed, P, global_deadline, log))
    else:
        log("== mechanism control SKIPPED (out of time budget) ==")

    # ----------------------------- aggregate --------------------------------
    def m(rs, key):
        return float(np.mean([r[key] for r in rs])) if rs else float("nan")

    aggregate = {}
    for L in P["lengths"]:
        for arm in P["arms"]:
            rs = [r for r in runs if r["L"] == L and r["arm"] == arm]
            aggregate[f"L{L}_{arm}"] = {
                "n_params": rs[0]["n_params"] if rs else None,
                "seeds": [r["seed"] for r in rs],
                "full_parity_last_tick": round(m(rs, "full_parity_last_tick"), 4),
                "full_parity_best_tick": round(m(rs, "full_parity_best_tick"), 4),
                "full_parity_certainty_tick": round(m(rs, "full_parity_certainty_tick"), 4),
                "allpos_last_tick": round(m(rs, "allpos_last_tick"), 4),
                "allpos_best_tick": round(m(rs, "allpos_best_tick"), 4),
                "seq_exact_last_tick": round(m(rs, "seq_exact_last_tick"), 4),
                "final_loss": round(m(rs, "final_loss"), 4),
                "per_seed_full_parity": [round(r["full_parity_last_tick"], 4) for r in rs],
                "per_seed_allpos": [round(r["allpos_last_tick"], 4) for r in rs],
            }

    contrasts = {}
    for L in P["lengths"]:
        g = lambda a, k: aggregate[f"L{L}_{a}"][k]
        contrasts[f"L{L}"] = {
            "sync_effect_in_ctm_allpos": round(g("ctm_sync", "allpos_last_tick") - g("ctm_lastz", "allpos_last_tick"), 4),
            "sync_effect_in_gru_allpos": round(g("gru_sync", "allpos_last_tick") - g("gru_lastz", "allpos_last_tick"), 4),
            "nlm_effect_allpos": round(g("ctm_lastz", "allpos_last_tick") - g("gru_lastz", "allpos_last_tick"), 4),
            "full_ctm_vs_plain_gru_allpos": round(g("ctm_sync", "allpos_last_tick") - g("gru_lastz", "allpos_last_tick"), 4),
            "sync_effect_in_ctm_fullparity": round(g("ctm_sync", "full_parity_last_tick") - g("ctm_lastz", "full_parity_last_tick"), 4),
            "sync_effect_in_gru_fullparity": round(g("gru_sync", "full_parity_last_tick") - g("gru_lastz", "full_parity_last_tick"), 4),
            "ladder_a_gt_b_gt_c_allpos": bool(
                g("ctm_sync", "allpos_last_tick") > g("ctm_lastz", "allpos_last_tick") >
                g("gru_lastz", "allpos_last_tick")),
            "per_seed_sync_effect_in_ctm_allpos": [
                round(a - b, 4) for a, b in
                zip(aggregate[f"L{L}_ctm_sync"]["per_seed_allpos"],
                    aggregate[f"L{L}_ctm_lastz"]["per_seed_allpos"])],
            "ranking_allpos": sorted(P["arms"], key=lambda a: -g(a, "allpos_last_tick")),
        }

    depth = {}
    for r in depth_runs:
        base = [x for x in runs if x["arm"] == r["arm"] and x["L"] == r["L"]
                and x["seed"] == r["seed"]]
        depth[r["arm"]] = {
            "T_shallow": r["T"],
            "allpos_shallow": round(r["allpos_last_tick"], 4),
            "full_parity_shallow": round(r["full_parity_last_tick"], 4),
            "T_deep": P["ticks"],
            "allpos_deep": round(base[0]["allpos_last_tick"], 4) if base else None,
            "full_parity_deep": round(base[0]["full_parity_last_tick"], 4) if base else None,
            "n_params": r["n_params"],
        }

    mech = {}
    for arm in P["mech_control_arms"]:
        rs = [r for r in mech_runs if r["arm"] == arm]
        if not rs:
            continue
        Lm = P["mech_control_length"]
        base_sync = aggregate[f"L{Lm}_ctm_sync"]["allpos_last_tick"]
        base_last = aggregate[f"L{Lm}_ctm_lastz"]["allpos_last_tick"]
        v = round(m(rs, "allpos_last_tick"), 4)
        mech[arm] = {
            "L": Lm, "T": P["ticks"], "n_params": rs[0]["n_params"],
            "allpos_last_tick": v,
            "per_seed_allpos": [round(r["allpos_last_tick"], 4) for r in rs],
            "full_parity_last_tick": round(m(rs, "full_parity_last_tick"), 4),
            "ctm_sync_allpos": base_sync,
            "ctm_lastz_allpos": base_last,
            "vs_ctm_lastz": round(v - base_last, 4),
            "vs_ctm_sync": round(v - base_sync, 4),
            "frac_of_sync_gain_recovered_without_timing":
                round((v - base_last) / (base_sync - base_last), 3)
                if abs(base_sync - base_last) > 1e-6 else None,
        }

    Lmax = max(P["lengths"])
    headline = (
        f"allpos acc @L={Lmax}: " +
        ", ".join(f"{a}={aggregate[f'L{Lmax}_{a}']['allpos_last_tick']:.3f}" for a in P["arms"]) +
        f" | full {Lmax}-bit parity: " +
        ", ".join(f"{a}={aggregate[f'L{Lmax}_{a}']['full_parity_last_tick']:.3f}" for a in P["arms"]) +
        f" | sync effect inside CTM (allpos) = "
        f"{contrasts[f'L{Lmax}']['sync_effect_in_ctm_allpos']:+.3f}")

    metrics = {
        "arms": P["arms"], "lengths": P["lengths"], "ticks": P["ticks"],
        "seeds": P["seeds"], "steps": P["steps"], "batch_size": P["batch_size"],
        "lr": P["lr"], "chance_full_parity": 0.5, "chance_allpos": 0.5,
        "param_counts": {f"L{L}_{a}": aggregate[f"L{L}_{a}"]["n_params"]
                         for L in P["lengths"] for a in P["arms"]},
        "aggregate": aggregate,
        "contrasts": contrasts,
        "depth_control": depth,
        "mechanism_control": mech,
        "per_run": runs + depth_runs + mech_runs,
        "any_run_time_capped": any(r["time_capped"] for r in runs + depth_runs + mech_runs),
        "headline": headline,
    }

    # ----------------------------- chart ------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"ctm_sync": "#1a7f64", "ctm_lastz": "#3d5a80",
              "gru_lastz": "#8a817c", "gru_sync": "#c95d3c"}
    labels = {"ctm_sync": "(a) CTM + sync readout",
              "ctm_lastz": "(b) CTM, last-tick readout",
              "gru_lastz": "(c) GRU, last-tick readout",
              "gru_sync": "(d) GRU + sync readout"}
    fig, axes = plt.subplots(1, 4, figsize=(19.5, 4.4))

    ax = axes[0]
    w, xs = 0.2, np.arange(len(P["lengths"]))
    for k, arm in enumerate(P["arms"]):
        vals = [aggregate[f"L{L}_{arm}"]["allpos_last_tick"] for L in P["lengths"]]
        ax.bar(xs + (k - 1.5) * w, vals, w, color=colors[arm], label=labels[arm])
        for xi, v in zip(xs + (k - 1.5) * w, vals):
            ax.text(xi, v + 0.008, f"{v:.2f}", ha="center", fontsize=7)
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.text(len(P["lengths"]) - 0.55, 0.512, "chance", fontsize=7, color="0.3")
    ax.set_xticks(xs); ax.set_xticklabels([f"{L} bits" for L in P["lengths"]])
    ax.set_ylabel("cumulative-parity accuracy (all positions)")
    ax.set_ylim(0.4, 1.06); ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.set_title("Ablation ladder, matched params / ticks / steps", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    for arm in P["arms"]:
        vals = [aggregate[f"L{L}_{arm}"]["full_parity_last_tick"] for L in P["lengths"]]
        ax.plot(P["lengths"], vals, "o-", color=colors[arm], label=labels[arm], lw=2, ms=5)
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_xlabel("input length (bits)"); ax.set_ylabel("full-string parity accuracy")
    ax.set_xticks(P["lengths"]); ax.set_ylim(0.4, 1.06)
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_title("Parity of the WHOLE string", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    Lp = P["lengths"][0]
    for arm in P["arms"]:
        rs = [r for r in runs if r["L"] == Lp and r["arm"] == arm]
        if not rs:
            continue
        curve = np.mean([r["allpos_by_tick"] for r in rs], axis=0)
        ax.plot(range(1, len(curve) + 1), curve, "o-", color=colors[arm],
                label=labels[arm], lw=2, ms=4)
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_xlabel("internal tick"); ax.set_ylabel(f"accuracy (all positions), L={Lp}")
    ax.set_ylim(0.4, 1.06); ax.legend(frameon=False, fontsize=7.5)
    ax.set_title("Does accuracy accrue over internal time?", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    # --- panel 4: what is the sync readout actually buying? (controls, L=16) ---
    ax = axes[3]
    Lm = P["mech_control_length"]
    bars = [("CTM last-tick\nT=12", aggregate[f"L{Lm}_ctm_lastz"]["allpos_last_tick"], "#3d5a80"),
            ("CTM sync\nT=12", aggregate[f"L{Lm}_ctm_sync"]["allpos_last_tick"], "#1a7f64")]
    if "ctm_syncnow" in mech:
        bars.append(("CTM pairwise products\nCURRENT TICK ONLY\nT=12",
                     mech["ctm_syncnow"]["allpos_last_tick"], "#7fbf9a"))
    if "ctm_lastz" in depth:
        bars.append(("CTM last-tick\nT=2", depth["ctm_lastz"]["allpos_shallow"], "#9db4cc"))
    if "ctm_sync" in depth:
        bars.append(("CTM sync\nT=2", depth["ctm_sync"]["allpos_shallow"], "#8fd0b8"))
    ax.bar(range(len(bars)), [b[1] for b in bars], color=[b[2] for b in bars], width=0.65)
    for i, b in enumerate(bars):
        ax.text(i, b[1] + 0.004, f"{b[1]:.3f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] for b in bars], fontsize=6.5)
    ax.set_ylim(0.70, 0.81)
    ax.set_ylabel(f"accuracy (all positions), L={Lm}")
    ax.set_title("Controls: it is the pairwise PRODUCT,\nnot the cross-tick timing", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    nparam = aggregate["L%d_ctm_sync" % P["lengths"][0]]["n_params"]
    fig.suptitle("Nano CTM: ablating the synchronization readout on parity "
                 "(~%.0fk params, T=%d internal ticks, %d seeds)"
                 % (nparam / 1000, P["ticks"], len(P["seeds"])), fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(HERE / "chart.png", dpi=160, bbox_inches="tight")

    results = {
        "id": cfg.get("id", "unknown"),
        "git_commit": git_sha(),
        "seed": int(cfg.get("seed", 0)),
        "duration_sec": round(time.time() - t0, 2),
        "metrics": metrics,
        "env": env_info(),
        "status": "done",
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: results[k] for k in ("id", "duration_sec", "status")}, indent=2))
    print("headline:", headline)


if __name__ == "__main__":
    main()
