"""Weight decay and the grokking delay on (a+b) mod 59.

Sweeps AdamW decoupled weight decay over three orders of magnitude on a 1-layer,
LayerNorm-free transformer trained full-batch on 50% of the (a+b) mod 59 table, and
measures, per arm:

  * step_memorized  = first step with train acc >= memorize_threshold (0.99)
  * steps_to_grok   = first step with test  acc >= grok_threshold     (0.95)
  * grok_delay      = steps_to_grok - step_memorized
  * final / best test accuracy, and test accuracy at a COMMON step (1200)
  * the L2 norm of all parameters at memorization, at grok, and at the end

Arms that never reach the grok threshold inside the shared step cap are reported as
CENSORED (steps_to_grok > cap) rather than dropped. Deterministic, CPU-only, 1 thread.

Usage:  python run.py
"""
import json, os, random, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
LOG = open(HERE / "train.log", "a" if "--chart-only" in sys.argv else "w")


def log(msg):
    print(msg, flush=True)
    LOG.write(msg + "\n")
    LOG.flush()


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
    Identical to the architecture of 2026-07-25_grokking-modular-addition.
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

    def forward(self, idx):
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
        return h.view(N, self.d_model) @ self.W_U


# ----------------------------- helpers -------------------------------------
def param_norm(model):
    with torch.no_grad():
        return float(torch.sqrt(sum((q.detach() ** 2).sum() for q in model.parameters())))


def first_crossing(steps, values, thresh):
    """First step at which values >= thresh. None if never. Eval-grid resolution."""
    for s, v in zip(steps, values):
        if v >= thresh:
            return int(s)
    return None


def value_at_step(steps, values, target):
    """Value at the last eval with step <= target (None if the run never got there)."""
    out = None
    for s, v in zip(steps, values):
        if s <= target:
            out = v
        else:
            break
    return out


def spearman(x, y):
    """Spearman rank correlation with average ranks for ties. No scipy dependency."""
    def ranks(a):
        a = np.asarray(a, dtype=np.float64)
        order = np.argsort(a, kind="mergesort")
        r = np.empty(len(a), dtype=np.float64)
        r[order] = np.arange(1, len(a) + 1, dtype=np.float64)
        for v in np.unique(a):
            m = a == v
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    rx, ry = ranks(x), ranks(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


# ----------------------------- one arm -------------------------------------
def train_arm(P, wd, seed, budget_s):
    """Train one weight-decay arm. Returns history + summary."""
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
    n_train = int(round(float(P["train_frac"]) * p * p))
    tr_idx, te_idx = np.sort(perm[:n_train]), np.sort(perm[n_train:])
    Xtr, Ytr, Xte, Yte = X[tr_idx], Y[tr_idx], X[te_idx], Y[te_idx]

    model = GrokTransformer(p, P["d_model"], P["n_heads"], P["d_mlp"],
                            P["n_ctx"], P["init_std_scale"])
    n_params = sum(q.numel() for q in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=P["lr"], weight_decay=wd,
                            betas=(P["beta1"], P["beta2"]))

    @torch.no_grad()
    def evaluate():
        model.eval()
        ltr, lte = model(Xtr), model(Xte)
        out = {
            "train_loss": float(F.cross_entropy(ltr, Ytr)),
            "train_acc": float((ltr.argmax(-1) == Ytr).float().mean()),
            "test_loss": float(F.cross_entropy(lte, Yte)),
            "test_acc": float((lte.argmax(-1) == Yte).float().mean()),
            "w_norm": param_norm(model),
        }
        model.train()
        return out

    keys = ("step", "train_acc", "test_acc", "train_loss", "test_loss", "w_norm")
    hist = {k: [] for k in keys}
    ev = int(P["eval_every"])
    max_steps, min_steps = int(P["max_steps"]), int(P["min_steps"])
    grok_thr, memo_thr = float(P["grok_threshold"]), float(P["memorize_threshold"])
    confirm = int(P["confirm_steps"])
    ab_below = float(P["abort_train_acc_below"])
    ab_win, ab_delta = int(P["abort_no_improve_window"]), float(P["abort_no_improve_delta"])

    grok_seen = None
    stop_reason, step = "step_cap", 0
    best_sd = None

    for step in range(max_steps + 1):
        if step % ev == 0:
            e = evaluate()
            hist["step"].append(step)
            for k in ("train_acc", "test_acc", "train_loss", "test_loss", "w_norm"):
                hist[k].append(e[k])
            if grok_seen is None and e["test_acc"] >= grok_thr:
                grok_seen = step
                best_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
                log(f"    GROK at step {step} (test {e['test_acc']:.4f})")
            if step % (ev * 16) == 0:
                log(f"    step {step:5d}  tr {e['train_acc']:.4f}  te {e['test_acc']:.4f}  "
                    f"|w| {e['w_norm']:.2f}  ({time.time()-t0:.0f}s)")
            if step >= min_steps:
                # 1. confirmed grok -> stop
                if grok_seen is not None and step >= grok_seen + confirm:
                    stop_reason = "grokked_and_confirmed"
                    break
                # 2. cannot fit the training set and is not improving -> stop
                if e["train_acc"] < ab_below:
                    past = value_at_step(hist["step"], hist["train_acc"], step - ab_win)
                    if past is not None and e["train_acc"] - past < ab_delta:
                        stop_reason = "aborted_cannot_fit_train"
                        log(f"    ABORT (cannot fit train) at step {step}, "
                            f"train acc {e['train_acc']:.4f}")
                        break
        if step == max_steps:
            stop_reason = "step_cap"
            break
        if time.time() - t0 > budget_s:
            stop_reason = "wall_clock_cap"
            log(f"    WALL-CLOCK CAP at step {step} ({time.time()-t0:.0f}s)")
            break
        loss = F.cross_entropy(model(Xtr), Ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()

    final = evaluate()
    wall = time.time() - t0
    steps_run = int(hist["step"][-1])

    memo_step = first_crossing(hist["step"], hist["train_acc"], memo_thr)
    grok_step = first_crossing(hist["step"], hist["test_acc"], grok_thr)
    censored = grok_step is None
    common = int(P["common_compare_step"])

    def r4(v):
        return None if v is None else round(float(v), 4)

    summary = {
        "weight_decay": wd,
        "seed": seed,
        "n_params": n_params,
        "steps_run": steps_run,
        "stop_reason": stop_reason,
        "wall_s": round(wall, 1),
        "step_memorized": memo_step,
        "memorized": memo_step is not None,
        "steps_to_grok": grok_step,
        "grok_censored": censored,
        "grok_delay": (None if (grok_step is None or memo_step is None)
                       else int(grok_step - memo_step)),
        "final_train_acc": r4(final["train_acc"]),
        "final_test_acc": r4(final["test_acc"]),
        "final_test_loss": r4(final["test_loss"]),
        "best_test_acc": r4(max(hist["test_acc"])),
        "best_train_acc": r4(max(hist["train_acc"])),
        "test_acc_at_common_step": r4(value_at_step(hist["step"], hist["test_acc"], common)),
        "train_acc_at_common_step": r4(value_at_step(hist["step"], hist["train_acc"], common)),
        "w_norm_init": round(hist["w_norm"][0], 3),
        "w_norm_max": round(float(max(hist["w_norm"])), 3),
        "w_norm_min": round(float(min(hist["w_norm"])), 3),
        "w_norm_final": round(final["w_norm"], 3),
        "w_norm_at_memorize": (None if memo_step is None else
                               round(value_at_step(hist["step"], hist["w_norm"], memo_step), 3)),
        "w_norm_at_grok": (None if grok_step is None else
                           round(value_at_step(hist["step"], hist["w_norm"], grok_step), 3)),
    }
    hist = {k: ([int(x) for x in v] if k == "step" else [round(float(x), 6) for x in v])
            for k, v in hist.items()}
    return {"summary": summary, "history": hist, "state_dict": best_sd}


# ----------------------------- chart ---------------------------------------
def make_chart(runs, P, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cap = int(P["max_steps"])
    grok_thr = float(P["grok_threshold"])
    s0runs = sorted([r for r in runs if r["summary"]["seed"] == 0],
                    key=lambda r: r["summary"]["weight_decay"])
    wds = [r["summary"]["weight_decay"] for r in s0runs]
    cmap = plt.get_cmap("viridis")
    cols = {wd: cmap(i / max(len(wds) - 1, 1)) for i, wd in enumerate(wds)}

    fig, ax = plt.subplots(2, 3, figsize=(18, 10))

    # (a) test acc curves
    a = ax[0, 0]
    for r in s0runs:
        s = r["summary"]
        a.plot(r["history"]["step"], r["history"]["test_acc"],
               color=cols[s["weight_decay"]], lw=1.8, label=f"wd={s['weight_decay']:g}")
    a.axhline(grok_thr, color="k", ls="--", lw=1, alpha=0.6)
    a.text(cap * 0.02, grok_thr + 0.02, f"grok threshold {grok_thr}", fontsize=8)
    a.set_xlabel("step"); a.set_ylabel("test accuracy")
    a.set_title("(a) Test accuracy (held-out pairs), seed 0")
    a.set_ylim(-0.03, 1.03); a.legend(fontsize=8, loc="center right")

    # (b) train acc curves
    a = ax[0, 1]
    for r in s0runs:
        s = r["summary"]
        a.plot(r["history"]["step"], r["history"]["train_acc"],
               color=cols[s["weight_decay"]], lw=1.8, label=f"wd={s['weight_decay']:g}")
    a.axhline(float(P["memorize_threshold"]), color="k", ls="--", lw=1, alpha=0.6)
    a.set_xlabel("step"); a.set_ylabel("train accuracy")
    a.set_title("(b) Train accuracy (memorization), seed 0")
    a.set_ylim(-0.03, 1.03); a.legend(fontsize=8, loc="lower right")

    # (c) steps-to-grok vs weight decay
    a = ax[0, 2]
    for seed, mk, alpha in ((0, "o", 1.0), (1, "s", 0.5)):
        rs = sorted([r["summary"] for r in runs if r["summary"]["seed"] == seed],
                    key=lambda s: s["weight_decay"])
        if not rs:
            continue
        gx = [s["weight_decay"] for s in rs if not s["grok_censored"]]
        gy = [s["steps_to_grok"] for s in rs if not s["grok_censored"]]
        dx = [s["weight_decay"] for s in rs if s["grok_delay"] is not None]
        dy = [s["grok_delay"] for s in rs if s["grok_delay"] is not None]
        cx = [s["weight_decay"] for s in rs if s["grok_censored"]]
        if gx:
            a.plot(gx, gy, mk + "-", color="C0", alpha=alpha, ms=8,
                   label=f"steps-to-grok (seed {seed})")
        if dx:
            a.plot(dx, dy, mk + "--", color="C1", alpha=alpha, ms=6,
                   label=f"delay = grok - memorize (seed {seed})")
        for x in cx:
            a.annotate("", xy=(x, cap * 2.6), xytext=(x, cap),
                       arrowprops=dict(arrowstyle="-|>", color="C3", lw=2, alpha=alpha))
            a.plot([x], [cap], mk, color="C3", ms=8, alpha=alpha)
    a.axhline(cap, color="C3", ls=":", lw=1.5)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_ylim(top=cap * 4.0)
    a.text(0.02, 0.90, f"step cap {cap}: arrows = CENSORED (> {cap})",
           fontsize=8, color="C3", transform=a.transAxes, va="top")
    a.set_xlabel("AdamW weight decay"); a.set_ylabel("steps")
    a.set_title("(c) Steps-to-grok vs weight decay")
    a.legend(fontsize=7, loc="lower left")
    a.grid(alpha=0.25, which="both")

    # (d) final / common-step accuracy vs weight decay
    a = ax[1, 0]
    rs = [r["summary"] for r in s0runs]
    xs = [s["weight_decay"] for s in rs]

    def nz(vals):   # an arm stopped before the common step has no value there
        return [np.nan if v is None else float(v) for v in vals]

    a.plot(xs, nz([s["best_test_acc"] for s in rs]), "o-", color="C0", label="best test acc")
    a.plot(xs, nz([s["test_acc_at_common_step"] for s in rs]), "s--", color="C2",
           label=f"test acc @ step {P['common_compare_step']}")
    a.plot(xs, nz([s["train_acc_at_common_step"] for s in rs]), "^:", color="C4",
           label=f"train acc @ step {P['common_compare_step']}")
    a.axhline(grok_thr, color="k", ls="--", lw=1, alpha=0.5)
    a.set_xscale("log"); a.set_xlabel("AdamW weight decay"); a.set_ylabel("accuracy")
    a.set_ylim(-0.03, 1.03)
    a.set_title("(d) Generalization vs weight decay (the destabilization end)")
    a.legend(fontsize=8); a.grid(alpha=0.25)

    # (e) parameter norm trajectories
    a = ax[1, 1]
    for r in s0runs:
        s = r["summary"]
        a.plot(r["history"]["step"], r["history"]["w_norm"],
               color=cols[s["weight_decay"]], lw=1.8, label=f"wd={s['weight_decay']:g}")
        if s["steps_to_grok"] is not None:
            a.plot([s["steps_to_grok"]], [s["w_norm_at_grok"]], "*", color="C3", ms=16)
    a.set_xlabel("step"); a.set_ylabel(r"$\|\theta\|_2$ (all parameters)")
    a.set_title("(e) Weight norm; red stars = the grok step")
    a.legend(fontsize=8); a.grid(alpha=0.25)

    # (f) memorize step vs grok delay
    a = ax[1, 2]
    for i, s in enumerate(rs):
        m = s["step_memorized"]
        g = s["steps_to_grok"] if s["steps_to_grok"] is not None else cap
        if m is not None:
            a.barh(i, m, color="C4", alpha=0.85)
            a.barh(i, max(g - m, 0), left=m,
                   color=("C3" if s["grok_censored"] else "C2"), alpha=0.85)
            if s["grok_censored"]:
                a.annotate("", xy=(cap * 1.14, i), xytext=(cap, i),
                           arrowprops=dict(arrowstyle="-|>", color="C3", lw=2))
        else:
            a.barh(i, s["steps_run"], color="0.7", alpha=0.85)
            a.text(cap * 0.03, i, f"never memorized (ran {s['steps_run']} steps)",
                   va="center", fontsize=8)
    a.set_yticks(np.arange(len(rs)))
    a.set_yticklabels([f"wd={s['weight_decay']:g}" for s in rs])
    a.set_xlabel("steps")
    a.set_title("(f) purple = to memorize, green = grok delay, red = censored")
    a.grid(alpha=0.25, axis="x")

    fig.suptitle("Weight decay and the grokking delay - 1-layer transformer, (a+b) mod 59, "
                 "50% train, full-batch AdamW", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ----------------------------- main ----------------------------------------
def rebuild_chart_from_results():
    """`python run.py --chart-only` re-renders chart.png from results.json, no retraining."""
    cfg = load_config()
    P = cfg["params"]
    res = json.load(open(HERE / "results.json"))
    summaries = res["metrics"]["table_seed0"] + res["metrics"]["table_seed1"]
    runs = []
    for h in res["histories"]:
        s = [q for q in summaries
             if q["weight_decay"] == h["weight_decay"] and q["seed"] == h["seed"]][0]
        runs.append({"summary": s,
                     "history": {k: v for k, v in h.items()
                                 if k not in ("weight_decay", "seed")},
                     "state_dict": None})
    make_chart(runs, P, HERE / "chart.png")
    log("chart.png re-rendered from results.json")


def main():
    cfg = load_config()
    P = cfg["params"]
    t_start = time.time()
    budget = float(P["total_train_budget_s"])
    wds = [float(w) for w in P["weight_decays"]]
    cap = int(P["max_steps"])

    runs = []
    log(f"=== seed 0: full sweep over weight decay {wds} ===")
    for wd in wds:
        remaining = budget - (time.time() - t_start)
        log(f"  [seed 0] wd={wd:g}  (budget left {remaining:.0f}s)")
        r = train_arm(P, wd, 0, remaining)
        s = r["summary"]
        log(f"    -> steps_to_grok={s['steps_to_grok']} censored={s['grok_censored']} "
            f"memo={s['step_memorized']} final_test={s['final_test_acc']} "
            f"({s['wall_s']}s, {s['stop_reason']})")
        runs.append(r)

    # --- seed 1: partial replication with whatever budget is left ---
    log("=== seed 1: partial replication (only arms that fit the remaining budget) ===")
    by_wd = {r["summary"]["weight_decay"]: r["summary"] for r in runs}
    for wd in [float(w) for w in P["seed1_priority"]]:
        remaining = budget - (time.time() - t_start)
        est = by_wd[wd]["wall_s"] * 1.4 + 5      # seed-0 cost + headroom for CPU contention
        if est > remaining:
            log(f"  [seed 1] SKIP wd={wd:g}: est {est:.0f}s > remaining {remaining:.0f}s")
            continue
        log(f"  [seed 1] wd={wd:g}  (est {est:.0f}s, budget left {remaining:.0f}s)")
        r = train_arm(P, wd, 1, remaining)
        s = r["summary"]
        log(f"    -> steps_to_grok={s['steps_to_grok']} censored={s['grok_censored']} "
            f"memo={s['step_memorized']} final_test={s['final_test_acc']} "
            f"({s['wall_s']}s, {s['stop_reason']})")
        runs.append(r)

    train_s = time.time() - t_start
    log(f"=== training done: {train_s:.0f}s over {len(runs)} arms ===")

    # ---------------- analysis ----------------
    S = [r["summary"] for r in runs]
    s0 = sorted([s for s in S if s["seed"] == 0], key=lambda s: s["weight_decay"])
    s1 = sorted([s for s in S if s["seed"] == 1], key=lambda s: s["weight_decay"])

    grokked = sorted([s for s in s0 if not s["grok_censored"]],
                     key=lambda s: s["weight_decay"])
    # monotone-decreasing check over ALL seed-0 arms; censored arms scored at cap+1,
    # which is a LOWER bound on their true steps-to-grok, so the rank statistic stays valid.
    y_all = [s["steps_to_grok"] if s["steps_to_grok"] is not None else cap + 1 for s in s0]
    rho_all = spearman([s["weight_decay"] for s in s0], y_all)

    # the "rising branch": arms at or below the weight decay with the best test accuracy
    best_wd = max(s0, key=lambda s: (s["best_test_acc"], -s["weight_decay"]))["weight_decay"]
    low = [s for s in s0 if s["weight_decay"] <= best_wd]
    rho_low = spearman([s["weight_decay"] for s in low],
                       [s["steps_to_grok"] if s["steps_to_grok"] is not None else cap + 1
                        for s in low])

    gy = [s["steps_to_grok"] for s in grokked]
    strictly_decreasing_among_grokked = all(gy[i] > gy[i + 1] for i in range(len(gy) - 1))
    grok_wds = [s["weight_decay"] for s in grokked]
    window = [min(grok_wds), max(grok_wds)] if grok_wds else [None, None]

    agree = []
    for a in s1:
        b = [s for s in s0 if s["weight_decay"] == a["weight_decay"]][0]
        agree.append({
            "weight_decay": a["weight_decay"],
            "seed0_steps_to_grok": b["steps_to_grok"],
            "seed1_steps_to_grok": a["steps_to_grok"],
            "seed0_censored": b["grok_censored"],
            "seed1_censored": a["grok_censored"],
            "same_censoring_verdict": b["grok_censored"] == a["grok_censored"],
            "seed0_final_test_acc": b["final_test_acc"],
            "seed1_final_test_acc": a["final_test_acc"],
        })

    def tbl(s):
        return {k: s[k] for k in (
            "weight_decay", "seed", "step_memorized", "steps_to_grok", "grok_delay",
            "grok_censored", "steps_run", "final_train_acc", "final_test_acc",
            "best_test_acc", "test_acc_at_common_step", "train_acc_at_common_step",
            "w_norm_at_memorize", "w_norm_at_grok", "w_norm_max", "w_norm_final",
            "stop_reason", "wall_s")}

    fastest = min(grokked, key=lambda s: s["steps_to_grok"]) if grokked else None
    destab = [s for s in s0 if s["final_train_acc"] < 0.99]

    metrics = {
        "p": int(P["p"]),
        "n_params": s0[0]["n_params"],
        "train_frac": float(P["train_frac"]),
        "n_train": int(round(float(P["train_frac"]) * int(P["p"]) ** 2)),
        "chance_acc": float(P["chance_acc"]),
        "weight_decays": wds,
        "step_cap": cap,
        "grok_threshold": float(P["grok_threshold"]),
        "memorize_threshold": float(P["memorize_threshold"]),
        "common_compare_step": int(P["common_compare_step"]),
        "n_arms": len(runs),
        "n_arms_seed0": len(s0),
        "n_arms_seed1": len(s1),
        "seed1_weight_decays": [s["weight_decay"] for s in s1],
        "table_seed0": [tbl(s) for s in s0],
        "table_seed1": [tbl(s) for s in s1],
        "n_grokked_seed0": len(grokked),
        "n_censored_seed0": len(s0) - len(grokked),
        "grokking_window_wd": window,
        "fastest_wd": (None if fastest is None else fastest["weight_decay"]),
        "fastest_steps_to_grok": (None if fastest is None else fastest["steps_to_grok"]),
        "fastest_grok_delay": (None if fastest is None else fastest["grok_delay"]),
        "spearman_wd_vs_steps_to_grok_all": round(rho_all, 4),
        "spearman_wd_vs_steps_to_grok_below_optimum": round(rho_low, 4),
        "strictly_decreasing_among_grokked": bool(strictly_decreasing_among_grokked),
        "monotone_decreasing_overall": bool(rho_all <= -0.9),
        "hypothesis_monotone_supported": bool(strictly_decreasing_among_grokked
                                              and len(grokked) >= 2),
        "hypothesis_destabilization_supported": bool(len(destab) > 0),
        "destabilized_wds": [s["weight_decay"] for s in destab],
        "steps_to_grok_by_wd": {f"{s['weight_decay']:g}": s["steps_to_grok"] for s in s0},
        "grok_delay_by_wd": {f"{s['weight_decay']:g}": s["grok_delay"] for s in s0},
        "step_memorized_by_wd": {f"{s['weight_decay']:g}": s["step_memorized"] for s in s0},
        "best_test_acc_by_wd": {f"{s['weight_decay']:g}": s["best_test_acc"] for s in s0},
        "final_train_acc_by_wd": {f"{s['weight_decay']:g}": s["final_train_acc"] for s in s0},
        "w_norm_at_grok_by_wd": {f"{s['weight_decay']:g}": s["w_norm_at_grok"]
                                 for s in s0 if s["w_norm_at_grok"] is not None},
        "w_norm_max_by_wd": {f"{s['weight_decay']:g}": s["w_norm_max"] for s in s0},
        "seed_agreement": agree,
        "train_sec": round(train_s, 1),
        "sibling_baseline": {
            "id": "2026-07-25_grokking-modular-addition",
            "weight_decay": 1.0, "train_frac": 0.5, "p": 59, "d_model": 64,
            "grok_delay_steps_at_test_acc_0.5": 642,
            "note": "same architecture and split protocol; its delay used a test-acc-0.5 "
                    "crossing, ours uses 0.95, so 642 is a lower bound on the delay measured here",
        },
    }

    bits = []
    if grokked:
        bits.append("grokked at wd " + ", ".join(
            f"{s['weight_decay']:g} ({s['steps_to_grok']} steps, delay {s['grok_delay']})"
            for s in grokked))
    cens = [s for s in s0 if s["grok_censored"]]
    if cens:
        bits.append(f"censored (>{cap} steps) at wd " + ", ".join(
            f"{s['weight_decay']:g} (best test {s['best_test_acc']:.3f})" for s in cens))
    metrics["headline"] = "; ".join(bits)

    results = {
        "id": cfg.get("id", "unknown"),
        "git_commit": git_sha(),
        "seed": int(cfg.get("seed", 0)),
        "duration_sec": round(time.time() - t_start, 2),
        "metrics": metrics,
        "histories": [{"weight_decay": r["summary"]["weight_decay"],
                       "seed": r["summary"]["seed"], **r["history"]} for r in runs],
        "env": env_info(),
        "status": "done",
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    make_chart(runs, P, HERE / "chart.png")

    if fastest is not None:
        fr = [r for r in runs if r["summary"]["seed"] == 0
              and r["summary"]["weight_decay"] == fastest["weight_decay"]][0]
        if fr["state_dict"] is not None:
            torch.save({"state_dict": fr["state_dict"],
                        "arch": {k: P[k] for k in ("p", "d_model", "n_heads", "d_mlp",
                                                   "n_ctx", "init_std_scale")},
                        "weight_decay": fastest["weight_decay"],
                        "step": fastest["steps_to_grok"],
                        "train_frac": float(P["train_frac"]), "seed": 0},
                       HERE / "model.pt")

    log("HEADLINE: " + metrics["headline"])
    log(f"TOTAL {results['duration_sec']}s")
    print(json.dumps({k: v for k, v in metrics.items()
                      if k not in ("table_seed0", "table_seed1", "seed_agreement")}, indent=2))


if __name__ == "__main__":
    if "--chart-only" in sys.argv:
        rebuild_chart_from_results()
    else:
        main()
