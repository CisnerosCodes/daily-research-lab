"""Which part of Mamba's selectivity is load-bearing? A minimal selective SSM on SELECTIVE COPY.

Hand-rolled minimal Mamba-style (S6) block in pure PyTorch (sequential scan; CPU). No mamba.py,
no state-spaces/mamba, no CUDA kernels.

    h_t = exp(dt_t * A) . h_{t-1}  +  (dt_t * B_t) . x_t          per (channel, state) pair
    y_t = sum_N (h_t . C_t) + D . x_t

Arms differ ONLY in which of {dt, B, C} is a function of the current input; everything else
(embedding, depthwise causal conv, SiLU gating, readout, optimiser, data) is identical:

  full   dt, B, C all input-dependent                 (Mamba S6)
  delta  dt input-dependent; B, C learned constants   <- the paper's claimed core mechanism
  bc     B, C input-dependent; dt a learned constant
  lti    nothing input-dependent                      (S4-lite / LTI)
  gru    nn.GRU reference at matched params           (gates see x AND h)

probe: full_noconv = full with the depthwise conv removed (is the win the SSM or the conv?)

Task and generator are REUSED VERBATIM from experiments/2026-07-25_minrnn-selcopy so the numbers
are directly comparable to the minGRU/GRU row: length-L sequence of a blank token (id 0) with k
data tokens (ids 1..v_data) at uniformly random positions; target = the k values in order of
appearance; readout = k linear slot heads on the LAST position.

CONTEXT / the question this answers: minrnn-selcopy found that a gate which cannot see the hidden
state (minGRU) fails at ORDERING - it recovers only the last ~2 items. Mamba's dt gate is ALSO
input-only. Does dt-selectivity hit the same ordering wall, or does the structured N-dim state
with its spectrum of per-channel decay rates avoid it? Per-slot accuracy is the readout.

Deterministic, CPU-only, single-threaded. Writes results.json + chart.png.
Usage:  python run.py
"""
import json, os, random, subprocess, sys, time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

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


# --------------------------------------------------------------------------- torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)


# ------------------------------------------------- data (verbatim from minrnn-selcopy)
def make_batch(B, L, k, v_data, gen):
    """Selective copy: k data tokens (ids 1..v_data) at random positions in L blanks (id 0)."""
    r = torch.rand(B, L, generator=gen)
    pos = r.argsort(dim=1)[:, :k].sort(dim=1).values            # (B,k) sorted random positions
    vals = torch.randint(1, 1 + v_data, (B, k), generator=gen)  # (B,k) data values
    x = torch.zeros(B, L, dtype=torch.long)
    x.scatter_(1, pos, vals)
    return x, vals - 1                                          # targets in 0..v_data-1


# ------------------------------------------------- the selective SSM block
class MambaBlock(nn.Module):
    """Minimal Mamba-style block. `delta_sel` / `bc_sel` switch selectivity on and off.

    d_inner = expand * d_model, state size N = d_state.
    A is diagonal, real, per (channel, state): A[i,n] = -exp(A_log[i,n]), S4D-real init A = -(n+1).
    dt is per-channel (B,T,d_inner); B_t, C_t are per-state (B,T,N) shared across channels when
    selective, and per-(channel,state) learned constants when not.
    """

    def __init__(self, d_model, d_state, expand, d_conv, dt_rank, delta_sel, bc_sel, use_conv):
        super().__init__()
        d_inner = expand * d_model
        self.d_inner, self.d_state = d_inner, d_state
        self.delta_sel, self.bc_sel, self.use_conv = delta_sel, bc_sel, use_conv
        self.d_conv = d_conv

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        if use_conv:
            self.conv = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1)

        # Mamba's dt bias init: softplus^-1(dt) with dt log-uniform in [dt_min, dt_max]
        dt = torch.exp(torch.rand(d_inner) * (np.log(0.1) - np.log(0.001)) + np.log(0.001))
        dt_inv = dt + torch.log(-torch.expm1(-dt))
        if delta_sel:
            self.x_proj_dt = nn.Linear(d_inner, dt_rank, bias=False)
            self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
            with torch.no_grad():
                self.dt_proj.bias.copy_(dt_inv)
        else:
            self.dt_bias = nn.Parameter(dt_inv)

        if bc_sel:
            self.x_proj_bc = nn.Linear(d_inner, 2 * d_state, bias=False)
        else:
            self.B_par = nn.Parameter(torch.randn(d_inner, d_state) / np.sqrt(d_state))
            self.C_par = nn.Parameter(torch.randn(d_inner, d_state) / np.sqrt(d_state))

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def deltas(self, xs):
        if self.delta_sel:
            return F.softplus(self.dt_proj(self.x_proj_dt(xs)))          # (B,T,d_inner)
        return F.softplus(self.dt_bias).expand(xs.shape[0], xs.shape[1], self.d_inner)

    def pre(self, u):
        """Everything before the scan: in_proj, depthwise causal conv, SiLU. Returns (xs, z)."""
        xz = self.in_proj(u)
        xs, z = xz.chunk(2, dim=-1)
        if self.use_conv:
            T = xs.shape[1]
            xs = self.conv(xs.transpose(1, 2))[:, :, :T].transpose(1, 2)
        return F.silu(xs), z

    def forward(self, u, all_steps=True):
        xs, z = self.pre(u)                                              # (B,T,d_inner)
        Bsz, T, _ = xs.shape
        dt = self.deltas(xs)
        A = -torch.exp(self.A_log)                                       # (d_inner,N)

        # (dt * x) is per-channel and small; fusing it in before the (B,T,d_inner,N) broadcast
        # keeps only ONE big elementwise op for Bx instead of two.
        dtx = (dt * xs).unsqueeze(-1)                                    # (B,T,d_inner,1)
        if self.bc_sel:
            Bt, Ct = self.x_proj_bc(xs).chunk(2, dim=-1)                 # (B,T,N)
            Bx = dtx * Bt.unsqueeze(2)                                   # (B,T,d_inner,N)
        else:
            Bx = dtx * self.B_par                                        # (B,T,d_inner,N)
        dA = torch.exp(dt.unsqueeze(-1) * A)                             # (B,T,d_inner,N)

        h = xs.new_zeros(Bsz, self.d_inner, self.d_state)
        dAl, Bxl = dA.unbind(1), Bx.unbind(1)
        Cl = Ct.unbind(1) if self.bc_sel else None
        ys = []
        for t in range(T):
            h = dAl[t] * h + Bxl[t]
            if all_steps or t == T - 1:
                c = Cl[t].unsqueeze(1) if self.bc_sel else self.C_par    # (B,1,N) or (d_inner,N)
                ys.append((h * c).sum(-1))                               # (B,d_inner)
        y = torch.stack(ys, 1)
        xr, zr = (xs, z) if all_steps else (xs[:, -1:], z[:, -1:])
        y = (y + self.D * xr) * F.silu(zr)
        return self.out_proj(y)


class GRUBlock(nn.Module):
    """Reference: standard GRU, gates see x AND h. Same residual-block interface; the hidden
    width is expand*d_model so the arm sits on the same width sweep as the SSM arms."""

    def __init__(self, d_model, expand):
        super().__init__()
        self.g = nn.GRU(d_model, expand * d_model, batch_first=True)
        self.proj = nn.Linear(expand * d_model, d_model, bias=False)

    def forward(self, u, all_steps=True):
        o, hn = self.g(u)
        return self.proj(o if all_steps else hn[-1].unsqueeze(1))


class Model(nn.Module):
    """Embedding -> n_blocks x (pre-LN residual block) -> LN -> GELU bottleneck -> k slot heads
    on the LAST position. Everything except the block internals is identical across arms."""

    def __init__(self, spec, d_model, k, n_blocks, P):
        super().__init__()
        self.k, self.v_data, self.n_blocks = k, P["v_data"], n_blocks
        self.emb = nn.Embedding(1 + P["v_data"], d_model)
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_blocks)])
        blocks = []
        for _ in range(n_blocks):
            if spec["kind"] == "gru":
                blocks.append(GRUBlock(d_model, P["expand"]))
            else:
                dt_rank = max(1, int(np.ceil(d_model / 16)))
                blocks.append(MambaBlock(
                    d_model, P["d_state"], P["expand"], P["d_conv"], dt_rank,
                    spec["delta_sel"], spec["bc_sel"], spec["use_conv"]))
        self.blocks = nn.ModuleList(blocks)
        self.norm_f = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, P["d_read"])
        self.heads = nn.Linear(P["d_read"], k * P["v_data"])

    def forward(self, x):
        e = self.emb(x)
        for i, (b, n) in enumerate(zip(self.blocks, self.norms)):
            last = (i == self.n_blocks - 1)
            out = b(n(e), all_steps=not last)
            e = (e[:, -1:] if last else e) + out
        h = self.norm_f(e[:, -1])
        return self.heads(F.gelu(self.proj(h))).view(-1, self.k, self.v_data)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ------------------------------------------------- arm table
ARM_TABLE = {                    # delta_sel, bc_sel, use_conv
    "full":        (True,  True,  True),
    "delta":       (True,  False, True),
    "bc":          (False, True,  True),
    "lti":         (False, False, True),
    "full_noconv": (True,  True,  False),
}


def arm_spec(arm):
    if arm == "gru":
        return {"kind": "gru"}
    t = ARM_TABLE[arm]
    return {"kind": "ssm", "delta_sel": t[0], "bc_sel": t[1], "use_conv": t[2]}


def build(arm, d_model, k, n_blocks, P):
    return Model(arm_spec(arm), d_model, k, n_blocks, P)


def fit_width(arm, k, n_blocks, target, P):
    """Pick d_model so the TOTAL parameter count is as close as possible to `target`."""
    best = None
    for w in range(8, 600, 2):
        n = n_params(build(arm, w, k, n_blocks, P))
        d = abs(n - target)
        if best is None or d < best[1]:
            best = (w, d, n)
        if n > target * 1.8:
            break
    return best[0], best[2]


# ------------------------------------------------- train / eval
def evaluate(model, L, k, P, n_eval):
    """Fixed held-out set (disjoint eval seed 999999, same as minrnn-selcopy)."""
    gen = torch.Generator().manual_seed(999999)
    B = 128
    tok = ex = n = 0
    slot = torch.zeros(k)
    with torch.no_grad():
        for _ in range(max(1, n_eval // B)):
            x, y = make_batch(B, L, k, P["v_data"], gen)
            pred = model(x).argmax(-1)
            corr = (pred == y).float()
            tok += corr.sum().item()
            slot += corr.sum(0)
            ex += corr.all(1).float().sum().item()
            n += B
    return tok / (n * k), ex / n, (slot / n).tolist(), n


def delta_stats(model, L, k, P):
    """For dt-selective arms: mean softplus(dt) on noise tokens vs data tokens, and the implied
    per-step retention exp(dt*A) at the slowest/fastest state. Mamba analogue of minGRU's gate stat."""
    blk = model.blocks[0]
    if not isinstance(blk, MambaBlock) or not blk.delta_sel:
        return None
    gen = torch.Generator().manual_seed(555555)
    with torch.no_grad():
        x, _ = make_batch(256, L, k, P["v_data"], gen)
        xs, _ = blk.pre(model.norms[0](model.emb(x)))
        dt = blk.deltas(xs).mean(-1)                       # (B,T) mean dt over channels
        is_data = (x > 0)
        dn, dd = dt[~is_data].mean().item(), dt[is_data].mean().item()
        A = -torch.exp(blk.A_log)
        a_slow, a_fast = A.max().item(), A.min().item()    # A<0: max = slowest decay
        return {"dt_noise": round(dn, 4), "dt_data": round(dd, 4),
                "dt_ratio_data_over_noise": round(dd / max(dn, 1e-9), 3),
                "A_slowest": round(a_slow, 4), "A_fastest": round(a_fast, 4),
                "retention_noise_slowest_state": round(float(np.exp(dn * a_slow)), 4),
                "retention_data_slowest_state": round(float(np.exp(dd * a_slow)), 4),
                "retention_noise_fastest_state": round(float(np.exp(dn * a_fast)), 6)}


def state_spectrum(model, L, k, P):
    """How wide is the retention spectrum the state actually uses? Reports the spread of
    per-(channel,state) retention exp(dt*A) at the mean dt on NOISE tokens - i.e. how many
    distinguishable timescales the block holds while it is idling between data tokens."""
    blk = model.blocks[0]
    if not isinstance(blk, MambaBlock):
        return None
    gen = torch.Generator().manual_seed(555555)
    with torch.no_grad():
        x, _ = make_batch(128, L, k, P["v_data"], gen)
        xs, _ = blk.pre(model.norms[0](model.emb(x)))
        dt = blk.deltas(xs)                                # (B,T,d_inner)
        is_noise = (x == 0).unsqueeze(-1)
        dt_noise = (dt * is_noise).sum((0, 1)) / is_noise.sum((0, 1)).clamp(min=1)  # (d_inner,)
        A = -torch.exp(blk.A_log)                          # (d_inner,N)
        ret = torch.exp(dt_noise.unsqueeze(-1) * A).flatten()
        # half-life in tokens for each (channel,state), clipped for readability
        hl = torch.log(torch.tensor(0.5)) / torch.log(ret.clamp(1e-6, 1 - 1e-6))
        q = torch.quantile(hl, torch.tensor([0.1, 0.5, 0.9]))
        return {"retention_noise_p10": round(float(torch.quantile(ret, 0.10)), 4),
                "retention_noise_p50": round(float(torch.quantile(ret, 0.50)), 4),
                "retention_noise_p90": round(float(torch.quantile(ret, 0.90)), 4),
                "halflife_tokens_p10": round(float(q[0]), 2),
                "halflife_tokens_p50": round(float(q[1]), 2),
                "halflife_tokens_p90": round(float(q[2]), 2),
                "frac_states_halflife_gt_L": round(float((hl > L).float().mean()), 4)}


def recency_index(slot):
    """+1 = only the last two items are readable (minGRU's signature); 0 = flat across slots."""
    if len(slot) < 4:
        return None
    return round(float(np.mean(slot[-2:]) - np.mean(slot[:-2])), 4)


def param_groups(model, wd):
    """No weight decay on 1-D params (biases, norms, D, dt_bias) or on A_log - decaying A_log
    pulls the SSM's decay rates toward A=-1 (fast forgetting), which would be an ablation in
    itself. Standard Mamba practice; applied identically to every arm."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if (p.ndim < 2 or n.endswith("A_log")) else decay).append(p)
    return [{"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0}]


def train_one(arm, width, n_blocks, L, k, steps, seed, P):
    set_seeds(seed)
    model = build(arm, width, k, n_blocks, P)
    opt = torch.optim.AdamW(param_groups(model, P["weight_decay"]), lr=P["lr"])
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=P["lr"], total_steps=steps, pct_start=P["warmup_frac"])
    gen = torch.Generator().manual_seed(1000 + seed)
    t0 = time.time()
    losses = []
    for i in range(steps):
        x, y = make_batch(P["batch_size"], L, k, P["v_data"], gen)
        loss = F.cross_entropy(model(x).reshape(-1, P["v_data"]), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), P["grad_clip"])
        opt.step()
        sch.step()
        if i % 50 == 0 or i == steps - 1:
            losses.append(round(loss.item(), 4))
    train_s = time.time() - t0
    tok, exact, slot, n_ev = evaluate(model, L, k, P, P["eval_n"])
    return {
        "arm": arm, "n_blocks": n_blocks, "width": width, "n_params": n_params(model),
        "L": L, "k": k, "steps": steps, "seed": seed,
        "token_acc": round(tok, 4), "exact_acc": round(exact, 4),
        "slot_acc": [round(s, 4) for s in slot], "recency_index": recency_index(slot),
        "eval_n": n_ev, "train_sec": round(train_s, 1), "loss_curve": losses,
        "delta_stats": delta_stats(model, L, k, P),
        "state_spectrum": state_spectrum(model, L, k, P),
    }


# ------------------------------------------------- main
def main():
    cfg = load_config()
    P = cfg["params"]
    seed0 = int(cfg.get("seed", 0))
    set_seeds(seed0)
    t_start = time.time()

    steps = int(os.environ.get("SMOKE_STEPS", P["steps"]))
    cl = lambda key: [tuple(c) for c in P[key]]
    cells, probe_cells = cl("cells"), cl("probe_cells")
    seed_cells, span_cells, depth_cells = cl("seed_cells"), cl("span_cells"), cl("depth_cells")
    target = P["target_params"]
    main_arms = list(P["arms"])
    nb2 = P["depth_probe_blocks"]

    # ---- run queue, highest priority first -------------------------------------------
    queue = []                                                  # (arm, cell, n_blocks, seed)
    for c in cells:                                             # 1. core sweep, seed 0
        for arm in main_arms:
            queue.append((arm, c, 1, seed0))
    for s in P["seeds"][1:]:                                    # 2. seed robustness
        for c in seed_cells:
            for arm in P["seed_arms"]:
                queue.append((arm, c, 1, s))
    for c in probe_cells:                                       # 3. conv control
        queue.append(("full_noconv", c, 1, seed0))
    for c in span_cells:                                        # 4. span control (longer L)
        for arm in P["span_arms"]:
            queue.append((arm, c, 1, seed0))
    for c in depth_cells:                                       # 5. depth probe (2 blocks)
        for arm in P["depth_arms"]:
            queue.append((arm, c, nb2, seed0))

    # ---- fit widths so every arm has ~the same TOTAL parameter count -----------------
    widths = {}
    for (arm, (L, k), nb, _s) in queue:
        if (arm, nb, k) not in widths:
            widths[(arm, nb, k)] = fit_width(arm, k, nb, target, P)

    # RESUME=1 reuses the runs already stored in results.json and only trains the queue entries
    # that are missing (used to top up runs that the wall-clock cap skipped on a first pass).
    # Every run is seeded from its own (arm, cell, seed) config, so a resumed run is bit-identical
    # to the same run in a single uninterrupted pass. `python run.py` alone still does everything.
    runs, skipped, prev_wall = [], [], 0.0
    if os.environ.get("RESUME") == "1" and (HERE / "results.json").exists():
        with open(HERE / "results.json") as f:
            prev = json.load(f)["metrics"]
        runs, prev_wall = list(prev["runs"]), prev.get("wall_clock_s", 0.0)
    done = {(r["arm"], r["L"], r["k"], r["n_blocks"], r["seed"]) for r in runs}

    for (arm, (L, k), nb, sd) in queue:
        st = steps
        if (arm, L, k, nb, sd) in done:
            continue
        if time.time() - t_start > P["time_cap_s"]:
            skipped.append({"arm": arm, "L": L, "k": k, "n_blocks": nb, "steps": st, "seed": sd})
            continue
        w, _ = widths[(arm, nb, k)]
        r = train_one(arm, w, nb, L, k, st, sd, P)
        runs.append(r)
        print(f"[{time.time()-t_start:6.1f}s] {arm:12s} nb={nb} L={L:3d} k={k} seed={sd} "
              f"p={r['n_params']:6d} w={w:4d} tok={r['token_acc']:.3f} "
              f"exact={r['exact_acc']:.3f} rec={r['recency_index']} ({r['train_sec']}s)", flush=True)

    # ---- aggregate --------------------------------------------------------------------
    def pick(arm, L, k, seed=None, nb=1):
        for r in runs:
            if r["arm"] == arm and r["L"] == L and r["k"] == k and r["n_blocks"] == nb \
               and (seed is None or r["seed"] == seed):
                return r
        return None

    def mean_over_seeds(arm, L, k, field):
        v = [r[field] for r in runs if r["arm"] == arm and r["L"] == L and r["k"] == k
             and r["n_blocks"] == 1]
        return round(float(np.mean(v)), 4) if v else None

    chance_tok = 1.0 / P["v_data"]
    by_cell = {}
    for (L, k) in cells:
        ck = f"L{L}_k{k}"
        entry = {"chance_token_acc": round(chance_tok, 4),
                 "chance_exact_acc": float(f"{chance_tok ** k:.3g}")}
        for arm in main_arms + ["full_noconv"]:
            r = pick(arm, L, k, seed=seed0)
            if r:
                entry[arm] = {"exact": r["exact_acc"], "token": r["token_acc"],
                              "n_params": r["n_params"], "width": r["width"],
                              "slot_acc": r["slot_acc"], "recency_index": r["recency_index"],
                              "delta_stats": r["delta_stats"],
                              "state_spectrum": r["state_spectrum"]}

        # Selectivity decomposition, reported on BOTH metrics: exact match saturates at the floor
        # once k=8 (no arm gets a full 8-token sequence right at this budget), so per-token
        # accuracy is the metric that still discriminates there.
        for fld, sfx in [("exact", ""), ("token", "_token")]:
            def g(a, _f=fld):
                return entry.get(a, {}).get(_f)
            if g("lti") is not None:
                for a, name in [("delta", "gain_from_delta_only"), ("bc", "gain_from_bc_only"),
                                ("full", "gain_from_full")]:
                    if g(a) is not None:
                        entry[name + sfx] = round(g(a) - g("lti"), 4)
            if g("full") is not None:
                if g("delta") is not None:
                    entry["full_minus_delta" + sfx] = round(g("full") - g("delta"), 4)
                if g("gru") is not None:
                    entry["full_minus_gru" + sfx] = round(g("full") - g("gru"), 4)
        # what the depthwise conv contributes, on the same scale as the selectivity gains
        if "full" in entry and "full_noconv" in entry:
            for fld, sfx in [("exact", ""), ("token", "_token")]:
                entry["conv_contribution" + sfx] = round(
                    entry["full"][fld] - entry["full_noconv"][fld], 4)
        # ordering readout: recency index per arm (+1 = only the last two slots readable)
        entry["recency_index_by_arm"] = {a: entry[a]["recency_index"]
                                         for a in main_arms if a in entry}
        entry["seed_mean_exact"] = {a: mean_over_seeds(a, L, k, "exact_acc") for a in main_arms
                                    if mean_over_seeds(a, L, k, "exact_acc") is not None}
        entry["seed_mean_token"] = {a: mean_over_seeds(a, L, k, "token_acc") for a in main_arms
                                    if mean_over_seeds(a, L, k, "token_acc") is not None}
        by_cell[ck] = entry

    def summarise(r, extra=()):
        d = {"exact": r["exact_acc"], "token": r["token_acc"], "n_params": r["n_params"],
             "width": r["width"], "slot_acc": r["slot_acc"],
             "recency_index": r["recency_index"]}
        for f in extra:
            d[f] = r[f]
        return d

    probes = {"conv_control": {}, "span_control": {}, "depth_control": {}}
    for (L, k) in probe_cells:                                  # is the win the SSM or the conv?
        ck, d = f"L{L}_k{k}", {}
        for arm in ["full", "full_noconv"]:
            r = pick(arm, L, k, seed=seed0)
            if r:
                d[arm] = summarise(r)
        if "full" in d and "full_noconv" in d:
            d["conv_contribution_exact"] = round(d["full"]["exact"] - d["full_noconv"]["exact"], 4)
        probes["conv_control"][ck] = d
    for (L, k) in span_cells:                                   # does a longer L change the ranking?
        ck, d = f"L{L}_k{k}", {}
        for arm in P["span_arms"]:
            r = pick(arm, L, k, seed=seed0)
            if r:
                d[arm] = summarise(r, extra=["delta_stats"])
        probes["span_control"][ck] = d
    for (L, k) in depth_cells:                                  # does depth substitute for gating?
        ck, d = f"L{L}_k{k}", {}
        for arm in P["depth_arms"]:
            for nb, tag in [(1, "1blk"), (nb2, f"{nb2}blk")]:
                r = pick(arm, L, k, seed=seed0, nb=nb)
                if r:
                    d[f"{arm}_{tag}"] = summarise(r)
            a1, a2 = d.get(f"{arm}_1blk"), d.get(f"{arm}_{nb2}blk")
            if a1 and a2:
                d[f"{arm}_depth_gain_exact"] = round(a2["exact"] - a1["exact"], 4)
        probes["depth_control"][ck] = d

    headline = {}
    for (L, k) in cells:
        ck = f"L{L}_k{k}"
        e = by_cell[ck]
        headline[ck] = {a: e.get(a, {}).get("exact") for a in main_arms}
        headline[ck]["token_acc"] = {a: e.get(a, {}).get("token") for a in main_arms if a in e}
        headline[ck]["recency_index"] = e.get("recency_index_by_arm")
        for base in ["gain_from_delta_only", "gain_from_bc_only", "gain_from_full",
                     "full_minus_delta", "full_minus_gru"]:
            for f in (base, base + "_token"):
                if f in e:
                    headline[ck][f] = e[f]

    metrics = {
        "task": "selective copy (blank token id 0, k data tokens at random positions, targets = "
                "data values in order of appearance); generator identical to 2026-07-25_minrnn-selcopy",
        "arm_definitions": {
            "full": "dt,B,C input-dependent (Mamba S6)",
            "delta": "dt input-dependent; B,C learned constants",
            "bc": "B,C input-dependent; dt a learned constant",
            "lti": "nothing input-dependent (S4-lite / LTI)",
            "gru": "nn.GRU reference (gates see x AND h)",
            "full_noconv": "full, depthwise causal conv removed (probe)"},
        "v_data": P["v_data"], "d_state": P["d_state"], "expand": P["expand"],
        "d_conv": P["d_conv"], "steps": steps, "batch_size": P["batch_size"], "lr": P["lr"],
        "eval_n": P["eval_n"], "target_params": target,
        "eval_binomial_se_at_half": round(0.5 / np.sqrt(P["eval_n"]), 4),
        "widths": {f"{a}_{nb}blk_k{k}": {"d_model": v[0], "n_params": v[1]}
                   for (a, nb, k), v in sorted(widths.items())},
        "headline_exact_match": headline,
        "by_cell": by_cell,
        "probes": probes,
        "lr_pre_sweep": P["lr_pre_sweep"],
        "minrnn_selcopy_reference": P["reference_row"],
        "runs": runs,
        "skipped_runs": skipped,
        "n_runs": len(runs),
        "wall_clock_s": round(prev_wall + time.time() - t_start, 1),
        "wall_clock_note": ("total training+eval seconds across all runs; a first pass under a "
                            "660 s cap left 2 runs, topped up with RESUME=1" if prev_wall else
                            "single pass"),
    }

    make_chart(by_cell, probes, cells, probe_cells, main_arms, P)

    results = {
        "id": cfg.get("id", "unknown"),
        "git_commit": git_sha(),
        "seed": seed0,
        "duration_sec": round(prev_wall + time.time() - t_start, 2),
        "metrics": metrics,
        "env": env_info(),
        "status": "done" if runs else "failed",
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({"headline": headline, "probes": probes,
                      "wall_clock_s": metrics["wall_clock_s"], "skipped": skipped}, indent=2))


# ------------------------------------------------- chart
LAB = {"full": "full S6 (dt,B,C)", "delta": "dt-only selective", "bc": "B,C-only selective",
       "lti": "LTI (nothing selective)", "gru": "GRU (x+h gates)",
       "full_noconv": "full, no conv"}
COL = {"full": "#d62728", "delta": "#ff7f0e", "bc": "#2ca02c", "lti": "#7f7f7f",
       "gru": "#1f77b4", "full_noconv": "#9467bd"}


def make_chart(by_cell, probes, cells, probe_cells, main_arms, P):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ck = [f"L{L}_k{k}" for (L, k) in cells]
    xs = np.arange(len(ck))
    w = 0.82 / max(1, len(main_arms))
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.8))

    for i, (ax, field, title) in enumerate([
            (axes[0], "exact", "Exact-sequence accuracy"),
            (axes[1], "token", "Per-token accuracy")]):
        for j, a in enumerate(main_arms):
            vals = [by_cell[c].get(a, {}).get(field, np.nan) for c in ck]
            ax.bar(xs + (j - (len(main_arms) - 1) / 2) * w, vals, w, label=LAB[a], color=COL[a])
        if field == "token":
            ax.axhline(1.0 / P["v_data"], ls="--", c="k", lw=1, label=f"chance (1/{P['v_data']})")
        ax.set_xticks(xs)
        ax.set_xticklabels([c.replace("_", "  ") for c in ck])
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.set_ylabel("accuracy")
        ax.grid(axis="y", alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")

    # panel 3: per-slot accuracy on the hardest cell - the ORDERING readout
    ax = axes[2]
    hard = f"L{cells[-1][0]}_k{cells[-1][1]}"
    for a in main_arms:
        sl = by_cell[hard].get(a, {}).get("slot_acc")
        if sl:
            ax.plot(np.arange(1, len(sl) + 1), sl, "o-", color=COL[a], label=LAB[a])
    ax.axhline(1.0 / P["v_data"], ls="--", c="k", lw=1, label="chance")
    ax.set_xlabel("output slot (1 = first data token seen)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Ordering readout: per-slot accuracy, {hard.replace('_', '  ')}")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)

    # panel 4: selectivity decomposition - per-token gain over the LTI baseline at matched params.
    # (Per-token, not exact match: exact match is at the floor for every SSM arm once k=8.)
    ax = axes[3]
    keys = [("gain_from_delta_only_token", "dt-only selectivity", "#ff7f0e"),
            ("gain_from_bc_only_token", "B,C-only selectivity", "#2ca02c"),
            ("gain_from_full_token", "full selectivity (dt+B,C)", "#d62728"),
            ("conv_contribution_token", "the depthwise conv\n(full - full_noconv)", "#9467bd")]
    w2 = 0.82 / len(keys)
    for j, (key, lab, col) in enumerate(keys):
        vals = [by_cell[c].get(key, np.nan) for c in ck]
        bars = ax.bar(xs + (j - 1.5) * w2, vals, w2, label=lab, color=col)
        for b, v in zip(bars, vals):
            if v == v:
                ax.text(b.get_x() + b.get_width() / 2, v + (0.004 if v >= 0 else -0.012),
                        f"{v:+.3f}", ha="center", fontsize=6.5)
    ax.axhline(0, c="k", lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([c.replace("_", "  ") for c in ck])
    ax.set_ylabel("per-token accuracy gain over LTI")
    ax.set_title("Which selectivity component is load-bearing?\n(matched total params, vs LTI baseline)")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Minimal selective SSM on selective copy: is $\\Delta$-selectivity the "
                 "load-bearing part of Mamba?", fontsize=13)
    fig.tight_layout()
    fig.savefig(HERE / "chart.png", dpi=130)


if __name__ == "__main__":
    main()
