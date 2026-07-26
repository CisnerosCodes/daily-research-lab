"""Muon vs AdamW on a 0.42M-param char LM: does the per-step win survive per WALL-CLOCK SECOND?

Muon (Keller Jordan) = SGD-Nesterov momentum whose 2D-matrix update is orthogonalized by a
5-iteration Newton-Schulz quintic before being applied.  Embeddings, the LM head, and all
norms/scalars stay on AdamW, exactly as the recipe prescribes.

The claim under test is NOT "does Muon reduce loss per step" (well replicated) but the honest
accounting question: the Newton-Schulz cost is a FIXED per-parameter cost that does not shrink
with the token count of a step.  Keller's own rule of thumb puts the FLOP overhead at

        overhead <= T * d_model / tokens_per_step

which is ~0.7% at nanoGPT-speedrun scale (T=5, d=768, 524k tokens/step) but ~63% here
(T=5, d=128, 1024 tokens/step) -- a ~100x larger tax.  This is therefore the regime where the
"seconds, not steps" correction should bite hardest.

Design
------
* lr swept over 3 values PER OPTIMIZER, best picked per optimizer (neither is handicapped).
* 2 seeds at the best lr; identical init and identical batch stream per seed across arms.
* Three headline arms:
    adamw          -- 600 steps at its best lr
    muon           -- 600 steps at its best lr           (MATCHED STEPS vs adamw)
    muon_isotime   -- K < 600 steps, K chosen from the measured per-step overhead so that
                      its wall clock matches the adamw arm's, with the cosine schedule
                      rewritten over its own K   (MATCHED SECONDS vs adamw)
  The third arm is what makes the seconds comparison fair: simply truncating the 600-step
  Muon run would leave its LR schedule undecayed and handicap it.  We report BOTH the
  retrained iso-time arm (primary) and the truncated-curve reading (secondary).
* Wall clock is measured with the eval passes EXCLUDED from the training clock.
* An interleaved A/B micro-benchmark measures the per-step overhead factor at four batch
  sizes, so the result can be extrapolated off this box.

SOAP is deliberately SKIPPED (a correct Shampoo-eigenbasis + Adam implementation does not fit
the CPU budget); the comparison is honestly renamed "Muon vs AdamW" in the README.  The slug
keeps its backlog id.

Deterministic, CPU-only, single-threaded.  Usage:  python run.py
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
LN2 = math.log(2.0)


# ----------------------------------------------------------------------------- utils
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
    info = {"python": sys.version.split()[0], "torch_threads": torch.get_num_threads()}
    for mod in ("numpy", "torch", "matplotlib"):
        try:
            info[mod] = getattr(__import__(mod), "__version__", "?")
        except Exception:
            pass
    try:
        import platform
        info["platform"] = platform.platform()
        info["cpu_count"] = os.cpu_count()
    except Exception:
        pass
    return info


# ------------------------------------------------------------------------------ data
def load_data(cfg):
    p = cfg["params"]
    txt_path = HERE / "data" / "tinyshakespeare.txt"
    if not txt_path.exists():                      # data/ is gitignored; fetch on a fresh clone
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(cfg["dataset"]["source"], txt_path)
    text = txt_path.read_text()
    chars = sorted(set(text))
    vocab = len(chars)
    assert vocab == p["char_vocab_size"], f"expected {p['char_vocab_size']} chars, got {vocab}"
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in text], dtype=np.int64)
    n = int(p["train_frac"] * len(ids))
    return ids[:n], ids[n:], vocab


# ----------------------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, d, n_head, d_ff):
        super().__init__()
        assert d % n_head == 0
        self.n_head, self.head_dim = n_head, d // n_head
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.fc = nn.Linear(d, d_ff, bias=False)
        self.out = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        shape = (B, T, self.n_head, self.head_dim)
        q = q.view(*shape).transpose(1, 2)
        k = k.view(*shape).transpose(1, 2)
        v = v.view(*shape).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, D))
        x = x + self.out(F.gelu(self.fc(self.ln2(x))))
        return x


class GPT(nn.Module):
    def __init__(self, vocab, d, n_layer, n_head, d_ff, block_size):
        super().__init__()
        self.vocab, self.block_size, self.d = vocab, block_size, d
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(block_size, d)
        self.blocks = nn.ModuleList([Block(d, n_head, d_ff) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.apply(self._init)
        for name, prm in self.named_parameters():      # nanoGPT residual-projection scaling
            if name.endswith("proj.weight") or name.endswith("out.weight"):
                nn.init.normal_(prm, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T))
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x))


def n_params(m):
    return sum(q.numel() for q in m.parameters())


# ------------------------------------------------------------------------------ Muon
def zeropower_via_newtonschulz5(G, steps=5):
    """Orthogonalize G via the tuned quintic Newton-Schulz iteration (Keller Jordan).

    phi(x) = a*x + b*x^3 + c*x^5 applied to the singular values, a,b,c = (3.4445,-4.7750,2.0315).
    The reference implementation runs this in bfloat16; on this CPU bfloat16 matmul is NOT
    faster than float32 (measured by the precision probe below), so we run float32, which is
    also the more accurate choice.  Deviation documented in the README.
    """
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Momentum SGD whose 2D update is orthogonalized by Newton-Schulz, then rescaled.

    Hand-rolled from the algorithm description in https://kellerjordan.github.io/posts/muon/
    (no dependency on the reference repo, which assumes CUDA + distributed).
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 weight_decay=0.0, ns_steps=5):
        super().__init__(list(params), dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                            weight_decay=weight_decay, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, mom, nest = group["lr"], group["momentum"], group["nesterov"]
            wd, ns = group["weight_decay"], group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(g)
                buf = st["buf"]
                buf.lerp_(g, 1 - mom)
                u = g.lerp(buf, mom) if nest else buf
                u = zeropower_via_newtonschulz5(u, ns)
                if wd:
                    p.mul_(1 - lr * wd)
                # rescale so the RMS of the update is comparable across shapes
                p.add_(u, alpha=-lr * max(1.0, p.size(0) / p.size(1)) ** 0.5)


def split_params(model):
    """Muon gets the 2D weight matrices of the hidden BLOCKS only.

    Embeddings, positional table, LM head, and all LayerNorm gains/biases go to AdamW --
    the split the Muon post explicitly prescribes.
    """
    muon, adam_decay, adam_nodecay = [], [], []
    for name, prm in model.named_parameters():
        if name.startswith("blocks") and prm.dim() == 2:
            muon.append(prm)
        elif prm.dim() >= 2:
            adam_decay.append(prm)
        else:
            adam_nodecay.append(prm)
    return muon, adam_decay, adam_nodecay


def ns_flops_for(shapes, ns_steps):
    """FLOPs of the Newton-Schulz iteration for a list of (rows, cols) matrices.

    After the internal transpose X is (m, n) with m = min(rows, cols) <= n.  Per iteration:
      A   = X @ X.T   -> 2*m*m*n
      A@A             -> 2*m^3
      B @ X           -> 2*m*m*n
    """
    tot = 0
    for r, c in shapes:
        m, n = min(r, c), max(r, c)
        tot += ns_steps * (4 * m * m * n + 2 * m ** 3)
    return tot


# ------------------------------------------------------------------------ train/eval
def get_batch(data, rng, batch_size, block_size):
    ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i:i + block_size] for i in ix])
    y = np.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return torch.from_numpy(x), torch.from_numpy(y)


@torch.no_grad()
def eval_bpc(model, val_ids, p, n_blocks_max):
    model.eval()
    bs = p["block_size"]
    n_blocks = min((len(val_ids) - 1) // bs, n_blocks_max)
    xs = np.stack([val_ids[i * bs:(i + 1) * bs] for i in range(n_blocks)])
    ys = np.stack([val_ids[i * bs + 1:(i + 1) * bs + 1] for i in range(n_blocks)])
    tot_nats, tot_tokens = 0.0, 0
    for s in range(0, n_blocks, p["eval_batch"]):
        xb = torch.from_numpy(xs[s:s + p["eval_batch"]])
        yb = torch.from_numpy(ys[s:s + p["eval_batch"]])
        loss = F.cross_entropy(model(xb).reshape(-1, model.vocab), yb.reshape(-1), reduction="sum")
        tot_nats += float(loss)
        tot_tokens += int(yb.numel())
    model.train()
    return tot_nats / (LN2 * tot_tokens), tot_tokens


def build_model(vocab, p, seed):
    set_seeds(seed)
    return GPT(vocab, p["d_model"], p["n_layer"], p["n_head"], p["d_ff"], p["block_size"])


def make_optimizers(model, opt_name, lr, aux_lr, p):
    muon_p, adam_dec, adam_nodec = split_params(model)
    if opt_name == "adamw":
        groups = [{"params": muon_p + adam_dec, "weight_decay": p["weight_decay"]},
                  {"params": adam_nodec, "weight_decay": 0.0}]
        return [torch.optim.AdamW(groups, lr=lr, betas=tuple(p["adamw_betas"]))], muon_p
    if opt_name == "muon":
        opts = [Muon(muon_p, lr=lr, momentum=p["muon_momentum"],
                     nesterov=p["muon_nesterov"], weight_decay=p["weight_decay"],
                     ns_steps=p["ns_steps"]),
                torch.optim.AdamW([{"params": adam_dec, "weight_decay": p["weight_decay"]},
                                   {"params": adam_nodec, "weight_decay": 0.0}],
                                  lr=aux_lr, betas=tuple(p["adamw_betas"]))]
        return opts, muon_p
    raise ValueError(opt_name)


def lr_at(it, base_lr, steps, warm, min_frac):
    if it < warm:
        return base_lr * (it + 1) / warm
    prog = (it - warm) / max(1, steps - warm)
    return base_lr * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * prog)))


def train_one(vocab, arm, opt_name, lr, aux_lr, seed, steps, p, train_ids, val_ids,
              final_eval=True):
    """One training run.  Wall clock EXCLUDES every eval pass."""
    model = build_model(vocab, p, seed)
    opts, muon_p = make_optimizers(model, opt_name, lr, aux_lr, p)
    all_p = list(model.parameters())
    rng = np.random.default_rng(seed)              # IDENTICAL batch stream across arms
    warm = min(p["warmup"], max(1, steps // 10))

    bpc0, _ = eval_bpc(model, val_ids, p, p["eval_blocks_interim"])
    ckpts = [{"step": 0, "train_seconds": 0.0, "val_bpc": round(bpc0, 5)}]
    losses, train_s, opt_s = [], 0.0, 0.0

    for it in range(steps):
        t0 = time.perf_counter()
        cur = lr_at(it, lr, steps, warm, p["lr_min_frac"])
        aux_cur = lr_at(it, aux_lr, steps, warm, p["lr_min_frac"])
        opts[0].param_groups[0]["lr"] = cur
        if opt_name == "adamw":
            opts[0].param_groups[1]["lr"] = cur
        else:
            for g in opts[1].param_groups:
                g["lr"] = aux_cur
        x, y = get_batch(train_ids, rng, p["batch_size"], p["block_size"])
        loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_p, p["grad_clip"])
        t_opt = time.perf_counter()
        for o in opts:
            o.step()
        opt_s += time.perf_counter() - t_opt
        train_s += time.perf_counter() - t0
        losses.append(float(loss.detach()))
        if (it + 1) % p["eval_every"] == 0 or (it + 1) == steps:
            b, _ = eval_bpc(model, val_ids, p, p["eval_blocks_interim"])
            ckpts.append({"step": it + 1, "train_seconds": round(train_s, 3),
                          "val_bpc": round(b, 5)})

    if final_eval:
        vb, n_eval = eval_bpc(model, val_ids, p, p["eval_blocks_final"])
    else:
        vb, n_eval = ckpts[-1]["val_bpc"], 0

    rec = {
        "arm": arm, "optimizer": opt_name, "lr": lr, "aux_adamw_lr": aux_lr,
        "seed": int(seed), "steps": steps,
        "n_params": n_params(model),
        "muon_params": int(sum(q.numel() for q in muon_p)),
        "train_seconds": round(train_s, 3),
        "opt_step_seconds": round(opt_s, 3),
        "sec_per_step": round(train_s / steps, 6),
        "opt_sec_per_step": round(opt_s / steps, 6),
        "final_train_loss_ma50": round(float(np.mean(losses[-50:])), 4),
        "val_bpc": round(vb, 5),
        "eval_chars": n_eval,
        "curve": ckpts,
    }
    print(f"  [{arm:<13s} lr={lr:<6g} seed={seed} steps={steps:4d}] "
          f"bpc={rec['val_bpc']:.4f}  {rec['train_seconds']:.1f}s "
          f"({rec['sec_per_step']*1000:.1f} ms/step, opt {rec['opt_sec_per_step']*1000:.1f} ms)",
          flush=True)
    return rec


# ------------------------------------------------------------- timing micro-benchmark
def micro_benchmark(vocab, p, train_ids):
    """Interleaved A/B per-step timing at several batch sizes.

    AdamW and Muon steps are alternated within each repetition so that any machine-load
    drift (this box has 2 shared cores) hits both arms nearly equally, and we report the
    MEDIAN over repetitions rather than the mean.
    """
    out = {}
    rng = np.random.default_rng(12345)
    for bs in p["bench_batch_sizes"]:
        models, optss = {}, {}
        for name in ("adamw", "muon"):
            m = build_model(vocab, p, 0)
            o, _ = make_optimizers(m, name, 0.003 if name == "adamw" else 0.02, 0.003, p)
            models[name], optss[name] = m, o
        x, y = get_batch(train_ids, rng, bs, p["block_size"])

        def one(name):
            m, opts = models[name], optss[name]
            t0 = time.perf_counter()
            loss = F.cross_entropy(m(x).reshape(-1, vocab), y.reshape(-1))
            for o in opts:
                o.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(m.parameters()), p["grad_clip"])
            t1 = time.perf_counter()
            for o in opts:
                o.step()
            t2 = time.perf_counter()
            return t2 - t0, t2 - t1

        for _ in range(p["bench_warmup"]):
            one("adamw"); one("muon")
        recs = {"adamw": [], "muon": []}
        opts_only = {"adamw": [], "muon": []}
        for _ in range(p["bench_reps"]):
            for name in ("adamw", "muon"):
                tot, opt = one(name)
                recs[name].append(tot)
                opts_only[name].append(opt)
        med = {k: float(np.median(v)) for k, v in recs.items()}
        medo = {k: float(np.median(v)) for k, v in opts_only.items()}
        out[str(bs)] = {
            "tokens_per_step": bs * p["block_size"],
            "adamw_sec_per_step": round(med["adamw"], 6),
            "muon_sec_per_step": round(med["muon"], 6),
            "adamw_opt_sec": round(medo["adamw"], 6),
            "muon_opt_sec": round(medo["muon"], 6),
            "overhead_factor": round(med["muon"] / med["adamw"], 4),
            "overhead_pct": round(100 * (med["muon"] / med["adamw"] - 1), 2),
        }
        print(f"  bench bs={bs:3d} ({bs*p['block_size']:5d} tok/step): "
              f"adamw {med['adamw']*1000:6.1f} ms | muon {med['muon']*1000:6.1f} ms | "
              f"x{out[str(bs)]['overhead_factor']:.3f}", flush=True)
    return out


def bench_ns_precision(p):
    """Is the reference implementation's bfloat16 Newton-Schulz actually faster on this CPU?"""
    shapes = [(3 * p["d_model"], p["d_model"]), (p["d_model"], p["d_model"]),
              (p["d_ff"], p["d_model"]), (p["d_model"], p["d_ff"])]
    out = {}

    def ns_bf16(m, steps):
        a, b, c = 3.4445, -4.7750, 2.0315
        X = m
        tr = X.size(0) > X.size(1)
        if tr:
            X = X.T
        X = X / (X.float().norm().to(X.dtype) + 1e-3)
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * (A @ A)
            X = a * X + B @ X
        return X

    for tag in ("float32", "bfloat16"):
        try:
            mats = [torch.randn(r, c) for r, c in shapes]
            if tag == "bfloat16":
                mats = [m.bfloat16() for m in mats]
            fn = (lambda m: zeropower_via_newtonschulz5(m, p["ns_steps"])) if tag == "float32" \
                else (lambda m: ns_bf16(m, p["ns_steps"]))
            for _ in range(3):
                for m in mats:
                    fn(m)
            ts = []
            for _ in range(8):
                t0 = time.perf_counter()
                for m in mats:
                    fn(m)
                ts.append(time.perf_counter() - t0)
            out[tag] = round(float(np.median(ts)), 6)
        except Exception as e:                                    # bf16 matmul may be missing
            out[tag] = None
            out[tag + "_error"] = str(e)[:160]
    if out.get("float32") and out.get("bfloat16"):
        out["bf16_speedup_vs_fp32"] = round(out["float32"] / out["bfloat16"], 3)
    out["note"] = ("seconds for one layer's worth of Muon matrices; multiply by n_layer "
                   f"({p['n_layer']}) for a full optimizer step")
    return out


# --------------------------------------------------------------------------- analysis
def bpc_at_seconds(curve, budget):
    """Linear interpolation of a (seconds -> val bpc) curve; None if the run never got there."""
    xs = [c["train_seconds"] for c in curve]
    ys = [c["val_bpc"] for c in curve]
    if budget < xs[0] or budget > xs[-1]:
        return None
    return round(float(np.interp(budget, xs, ys)), 5)


def mean_curve(runs, key_x="step"):
    n = min(len(r["curve"]) for r in runs)
    return [{key_x: runs[0]["curve"][i][key_x],
             "train_seconds": round(float(np.mean([r["curve"][i]["train_seconds"] for r in runs])), 3),
             "val_bpc": round(float(np.mean([r["curve"][i]["val_bpc"] for r in runs])), 5)}
            for i in range(n)]


def make_chart(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = {"adamw": "#1f77b4", "muon": "#d62728", "muon_isotime": "#ff7f0e"}
    fig, axes = plt.subplots(1, 4, figsize=(21, 5.0))

    # --- panel 1: lr sweep, both optimizers
    ax = axes[0]
    for opt in ("adamw", "muon"):
        sw = res["lr_sweep"][opt]
        lrs = [s["lr"] for s in sw]
        bp = [s["val_bpc"] for s in sw]
        ax.plot(lrs, bp, "o-", color=C[opt], ms=8, lw=2, label=opt)
        for l, b in zip(lrs, bp):
            ax.annotate(f"{b:.3f}", (l, b), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=7)
        best = res["best_lr"][opt]
        bb = min(s["val_bpc"] for s in sw)
        ax.plot([best], [bb], "*", color=C[opt], ms=18, zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("learning rate (log)")
    ax.set_ylabel("val bits per character")
    ax.set_title("lr swept per optimizer, seed 0\n(star = best; neither arm handicapped)",
                 fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # --- panel 2: val bpc vs STEPS
    ax = axes[1]
    for arm in ("adamw", "muon"):
        for r in res["runs"]:
            if r["arm"] != arm:
                continue
            ax.plot([c["step"] for c in r["curve"]], [c["val_bpc"] for c in r["curve"]],
                    color=C[arm], alpha=0.35, lw=1)
        mc = res["mean_curves"][arm]
        ax.plot([c["step"] for c in mc], [c["val_bpc"] for c in mc], color=C[arm], lw=2.5,
                label=f"{arm} (lr {res['best_lr'][arm]:g})")
    d = res["headline"]["delta_bpc_matched_steps"]
    ax.set_xlabel("optimizer steps")
    ax.set_ylabel("val bits per character")
    ax.set_title(f"MATCHED STEPS: muon - adamw = {d:+.4f} bpc\n"
                 f"(thin = seeds, thick = mean)", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # --- panel 3: val bpc vs WALL-CLOCK SECONDS
    ax = axes[2]
    for arm in ("adamw", "muon", "muon_isotime"):
        rs = [r for r in res["runs"] if r["arm"] == arm]
        if not rs:
            continue
        for r in rs:
            ax.plot([c["train_seconds"] for c in r["curve"]], [c["val_bpc"] for c in r["curve"]],
                    color=C[arm], alpha=0.3, lw=1,
                    ls="--" if arm == "muon_isotime" else "-")
        mc = res["mean_curves"][arm]
        ax.plot([c["train_seconds"] for c in mc], [c["val_bpc"] for c in mc], color=C[arm],
                lw=2.5, ls="--" if arm == "muon_isotime" else "-",
                label=f"{arm} ({rs[0]['steps']} steps)")
    b = res["headline"]["seconds_budget"]
    ax.axvline(b, color="k", ls=":", lw=1.5)
    ax.annotate(f"budget {b:.0f}s", (b, ax.get_ylim()[1]), textcoords="offset points",
                xytext=(-4, -12), ha="right", fontsize=8)
    d2 = res["headline"]["delta_bpc_matched_seconds_retrained"]
    ax.set_xlabel("training wall-clock seconds (eval excluded)")
    ax.set_ylabel("val bits per character")
    ax.set_title(f"MATCHED SECONDS: muon_isotime - adamw = {d2:+.4f} bpc\n"
                 f"(1 CPU thread; iso-time arm retrained with its own schedule)", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # --- panel 4: measured overhead factor vs tokens/step, with Keller's rule
    ax = axes[3]
    bench = res["micro_benchmark"]["per_step"]
    toks = [v["tokens_per_step"] for v in bench.values()]
    ov = [100 * (v["overhead_factor"] - 1) for v in bench.values()]
    ax.plot(toks, ov, "s-", color="#d62728", ms=8, lw=2, label="measured wall-clock overhead")
    pred = [100 * res["params"]["ns_steps"] * res["params"]["d_model"] / t for t in toks]
    ax.plot(toks, pred, "^--", color="#555555", ms=7, lw=1.6,
            label="Keller rule  T*d/tokens_per_step")
    here = res["params"]["batch_size"] * res["params"]["block_size"]
    ax.axvline(here, color="k", ls=":", lw=1.5)
    ax.annotate("this experiment", (here, min(ov)), textcoords="offset points",
                xytext=(5, 2), fontsize=8, rotation=90, va="bottom", color="dimgray")
    sp = res["micro_benchmark"]["speedrun_reference"]
    ax.annotate(f"nanoGPT speedrun regime:\nT*d/B = {sp['keller_rule_pct']:.2f}%\n"
                f"(d={sp['d_model']}, B={sp['tokens_per_step']:,})",
                xy=(0.03, 0.06), xycoords="axes fraction", fontsize=7.5, color="dimgray")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("tokens per optimizer step (log)")
    ax.set_ylabel("Muon per-step overhead, % (log)")
    ax.set_title("The Newton-Schulz tax is a FIXED cost:\nit shrinks only as tokens/step grows",
                 fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5)

    fig.suptitle(
        f"Muon vs AdamW at {res['n_params']:,} params on tiny-shakespeare, CPU 1 thread "
        f"(SOAP skipped - see README): steps vs seconds", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------------------ main
def main():
    cfg = load_config()
    p = cfg["params"]
    t_start = time.time()

    train_ids, val_ids, vocab = load_data(cfg)
    print(f"data: {len(train_ids):,} train chars, {len(val_ids):,} val chars, vocab {vocab}",
          flush=True)

    probe = build_model(vocab, p, 0)
    muon_p, adam_dec, adam_nodec = split_params(probe)
    N = n_params(probe)
    n_muon = int(sum(q.numel() for q in muon_p))
    muon_shapes = [tuple(q.shape) for q in muon_p]
    tokens_per_step = p["batch_size"] * p["block_size"]
    n_nonemb = N - probe.tok.weight.numel() - probe.pos.weight.numel()
    model_flops = 6 * n_nonemb * tokens_per_step          # fwd+bwd, standard approximation
    ns_fl = ns_flops_for(muon_shapes, p["ns_steps"])
    print(f"model: {N:,} params ({n_muon:,} on Muon, {N - n_muon:,} on AdamW), "
          f"{tokens_per_step} tokens/step")
    print(f"predicted NS FLOP overhead: {100*ns_fl/model_flops:.1f}%  |  "
          f"Keller rule T*d/B = {100*p['ns_steps']*p['d_model']/tokens_per_step:.1f}%", flush=True)

    print("\n--- timing micro-benchmark (interleaved A/B) ---", flush=True)
    bench = micro_benchmark(vocab, p, train_ids)
    ns_prec = bench_ns_precision(p)
    print(f"  NS precision probe: {ns_prec}", flush=True)

    runs = []

    def sweep(opt_name, grid, aux_lr_fn):
        """lr sweep with an EDGE GUARD.

        If the best lr lands on an endpoint of the grid, the grid is extended one point
        further in that direction (x3 / /3), up to `lr_grid_edge_extensions` times.  Without
        this, "we swept 3 values" can silently mean "the baseline was capped", which would
        inflate the other optimizer's margin -- the single most common way an optimizer
        comparison lies.
        """
        sw, ext = [], 0
        todo = list(grid)
        while todo:
            lr = todo.pop(0)
            aux = aux_lr_fn(lr)
            r = train_one(vocab, f"sweep_{opt_name}_lr{lr:g}", opt_name, lr, aux,
                          p["seeds"][0], p["steps"], p, train_ids, val_ids)
            sw.append({"lr": lr, "val_bpc": r["val_bpc"], "sec_per_step": r["sec_per_step"],
                       "extension": bool(lr not in grid)})
            runs.append(r)
            if not todo and ext < p["lr_grid_edge_extensions"]:
                sw_sorted = sorted(sw, key=lambda s: s["lr"])
                best = min(sw, key=lambda s: s["val_bpc"])["lr"]
                if best == sw_sorted[-1]["lr"]:
                    todo.append(best * 3.0); ext += 1
                    print(f"  ! best lr {best:g} is the TOP of the grid -> extending to "
                          f"{best*3:g}", flush=True)
                elif best == sw_sorted[0]["lr"]:
                    todo.append(best / 3.0); ext += 1
                    print(f"  ! best lr {best:g} is the BOTTOM of the grid -> extending to "
                          f"{best/3:g}", flush=True)
        sw = sorted(sw, key=lambda s: s["lr"])
        best = min(sw, key=lambda s: s["val_bpc"])["lr"]
        sw_lrs = [s["lr"] for s in sw]
        edge = best in (sw_lrs[0], sw_lrs[-1])
        print(f"  best {opt_name} lr = {best:g}  (grid {sw_lrs}, still at edge: {edge})",
              flush=True)
        return sw, best, edge

    # ---------------- phase A: AdamW lr sweep (seed 0)
    print("\n--- phase A: AdamW lr sweep (seed 0) ---", flush=True)
    sweep_adamw, best_adamw_lr, adamw_edge = sweep("adamw", p["adamw_lrs"], lambda lr: lr)

    # ---------------- phase B: Muon lr sweep (seed 0), aux AdamW at AdamW's best lr
    print(f"\n--- phase B: Muon lr sweep (seed 0), aux AdamW lr fixed at {best_adamw_lr:g} ---",
          flush=True)
    sweep_muon, best_muon_lr, muon_edge = sweep("muon", p["muon_lrs"],
                                                lambda lr: best_adamw_lr)

    # ---------------- phase C: the two headline arms, all seeds, at their best lr
    print("\n--- phase C: headline arms at best lr, all seeds ---", flush=True)
    for seed in p["seeds"]:
        runs.append(train_one(vocab, "adamw", "adamw", best_adamw_lr, best_adamw_lr, seed,
                              p["steps"], p, train_ids, val_ids))
    for seed in p["seeds"]:
        runs.append(train_one(vocab, "muon", "muon", best_muon_lr, best_adamw_lr, seed,
                              p["steps"], p, train_ids, val_ids))

    adamw_runs = [r for r in runs if r["arm"] == "adamw"]
    muon_runs = [r for r in runs if r["arm"] == "muon"]
    a_sps = float(np.mean([r["sec_per_step"] for r in adamw_runs]))
    m_sps = float(np.mean([r["sec_per_step"] for r in muon_runs]))
    overhead_train = m_sps / a_sps
    seconds_budget = float(np.mean([r["train_seconds"] for r in adamw_runs]))
    iso_steps = max(1, int(round(p["steps"] / overhead_train)))
    print(f"\n  measured in-training overhead factor: x{overhead_train:.4f}  "
          f"-> iso-time Muon gets {iso_steps} steps for the same ~{seconds_budget:.1f}s",
          flush=True)

    # ---------------- phase D: MATCHED SECONDS -- Muon retrained with a shorter schedule
    print("\n--- phase D: muon_isotime (matched wall clock, own cosine schedule) ---", flush=True)
    for seed in p["seeds"]:
        runs.append(train_one(vocab, "muon_isotime", "muon", best_muon_lr, best_adamw_lr, seed,
                              iso_steps, p, train_ids, val_ids))
    iso_runs = [r for r in runs if r["arm"] == "muon_isotime"]

    # ---------------- analysis
    def stat(rs):
        b = np.array([r["val_bpc"] for r in rs])
        return {"mean": round(float(b.mean()), 5),
                "per_seed": [round(float(u), 5) for u in b],
                "spread": round(float(b.max() - b.min()), 5),
                "seconds_mean": round(float(np.mean([r["train_seconds"] for r in rs])), 2),
                "steps": rs[0]["steps"]}

    S = {"adamw": stat(adamw_runs), "muon": stat(muon_runs), "muon_isotime": stat(iso_runs)}
    seed_spread = float(np.mean([S[k]["spread"] for k in S]))

    d_steps = round(S["muon"]["mean"] - S["adamw"]["mean"], 5)
    d_sec_retrained = round(S["muon_isotime"]["mean"] - S["adamw"]["mean"], 5)
    per_seed_steps = [round(m - a, 5) for m, a in zip(S["muon"]["per_seed"], S["adamw"]["per_seed"])]
    per_seed_sec = [round(m - a, 5) for m, a in zip(S["muon_isotime"]["per_seed"],
                                                    S["adamw"]["per_seed"])]

    # secondary, schedule-unfair reading: interpolate the FULL 600-step Muon curve at the budget
    trunc = [bpc_at_seconds(r["curve"], seconds_budget) for r in muon_runs]
    trunc = [t for t in trunc if t is not None]
    d_sec_trunc = (round(float(np.mean(trunc)) - S["adamw"]["mean"], 5) if trunc else None)

    def _reach(curve, target, key):
        for i in range(1, len(curve)):
            if curve[i]["val_bpc"] <= target:
                x0, x1 = curve[i - 1][key], curve[i][key]
                y0, y1 = curve[i - 1]["val_bpc"], curve[i]["val_bpc"]
                if y0 == y1:
                    return x1
                return x0 + (target - y0) * (x1 - x0) / (y1 - y0)
        return None

    def _mean_reach(rs, target, key):
        v = [_reach(r["curve"], target, key) for r in rs]
        v = [u for u in v if u is not None]
        return round(float(np.mean(v)), 2) if v else None

    tgt = S["adamw"]["mean"]
    m_reach_s = _mean_reach(muon_runs, tgt, "train_seconds")
    a_reach_s = _mean_reach(adamw_runs, tgt, "train_seconds")
    m_reach_st = _mean_reach(muon_runs, tgt, "step")
    a_reach_st = _mean_reach(adamw_runs, tgt, "step")
    time_speedup = round(a_reach_s / m_reach_s, 4) if (m_reach_s and a_reach_s) else None
    step_speedup = round(a_reach_st / m_reach_st, 4) if (m_reach_st and a_reach_st) else None

    def verdict(delta, spread):
        if abs(delta) <= spread:
            return "tie (gap within the mean seed spread)"
        return "Muon wins" if delta < 0 else "AdamW wins"

    res = {
        "n_params": N,
        "params": p,
        "best_lr": {"adamw": best_adamw_lr, "muon": best_muon_lr},
        "lr_sweep": {"adamw": sweep_adamw, "muon": sweep_muon},
        "runs": runs,
        "mean_curves": {k: mean_curve([r for r in runs if r["arm"] == k])
                        for k in ("adamw", "muon", "muon_isotime")},
        "micro_benchmark": {
            "per_step": bench,
            "ns_precision_probe_seconds": ns_prec,
            "speedrun_reference": {"d_model": 768, "tokens_per_step": 524288,
                                   "keller_rule_pct": round(100 * p["ns_steps"] * 768 / 524288, 4)},
        },
        "headline": {
            "delta_bpc_matched_steps": d_steps,
            "delta_bpc_matched_seconds_retrained": d_sec_retrained,
            "seconds_budget": round(seconds_budget, 2),
        },
    }

    metrics = {
        "headline": ("val bpc of Muon vs AdamW at MATCHED STEPS and at MATCHED WALL-CLOCK "
                     "SECONDS on 1 CPU thread, each at its own best lr, plus the measured "
                     "per-step Newton-Schulz overhead factor"),
        "soap_included": False,
        "soap_skip_reason": ("SOAP (Shampoo eigenbasis + Adam) needs a periodic eigendecomposition "
                             "of per-layer preconditioners; a correct implementation does not fit "
                             "the 12-minute CPU budget. The backlog sanctions this shrink; the "
                             "comparison is honestly renamed Muon vs AdamW."),
        "n_params": N,
        "n_params_on_muon": n_muon,
        "n_params_on_adamw_in_muon_arm": int(N - n_muon),
        "muon_matrix_shapes": [list(s) for s in muon_shapes],
        "tokens_per_step": tokens_per_step,
        "steps": p["steps"],
        "iso_time_steps": iso_steps,
        "seeds": p["seeds"],
        "n_runs": len(runs),

        "lr_sweep_adamw_val_bpc": {str(s["lr"]): s["val_bpc"] for s in sweep_adamw},
        "lr_sweep_muon_val_bpc": {str(s["lr"]): s["val_bpc"] for s in sweep_muon},
        "best_lr": {"adamw": best_adamw_lr, "muon": best_muon_lr},
        "lr_grids_actually_run": {"adamw": [s["lr"] for s in sweep_adamw],
                                  "muon": [s["lr"] for s in sweep_muon]},
        "lr_grid_was_extended": {"adamw": any(s["extension"] for s in sweep_adamw),
                                 "muon": any(s["extension"] for s in sweep_muon)},
        "best_lr_still_at_grid_edge_after_extension": {"adamw": bool(adamw_edge),
                                                       "muon": bool(muon_edge)},
        "aux_adamw_lr_in_muon_arm": best_adamw_lr,

        "val_bpc_by_arm": {k: S[k]["mean"] for k in S},
        "val_bpc_per_seed_by_arm": {k: S[k]["per_seed"] for k in S},
        "seed_spread_by_arm": {k: S[k]["spread"] for k in S},
        "mean_seed_spread_bpc": round(seed_spread, 5),
        "train_seconds_by_arm": {k: S[k]["seconds_mean"] for k in S},
        "steps_by_arm": {k: S[k]["steps"] for k in S},

        "MATCHED_STEPS_delta_muon_minus_adamw": d_steps,
        "MATCHED_STEPS_per_seed_delta": per_seed_steps,
        "MATCHED_STEPS_verdict": verdict(d_steps, seed_spread),
        "MATCHED_STEPS_all_seeds_same_sign": bool(len(set(np.sign(per_seed_steps))) == 1),

        "MATCHED_SECONDS_delta_muon_isotime_minus_adamw": d_sec_retrained,
        "MATCHED_SECONDS_per_seed_delta": per_seed_sec,
        "MATCHED_SECONDS_verdict": verdict(d_sec_retrained, seed_spread),
        "MATCHED_SECONDS_all_seeds_same_sign": bool(len(set(np.sign(per_seed_sec))) == 1),
        "MATCHED_SECONDS_budget_s": round(seconds_budget, 2),
        "MATCHED_SECONDS_isotime_achieved_s": S["muon_isotime"]["seconds_mean"],
        "MATCHED_SECONDS_budget_match_rel_error": round(
            abs(S["muon_isotime"]["seconds_mean"] - seconds_budget) / seconds_budget, 4),
        "MATCHED_SECONDS_delta_truncated_curve_reading": d_sec_trunc,

        "overhead_factor_in_training": round(overhead_train, 4),
        "overhead_pct_in_training": round(100 * (overhead_train - 1), 2),
        "sec_per_step_adamw": round(a_sps, 6),
        "sec_per_step_muon": round(m_sps, 6),
        "opt_sec_per_step_adamw": round(float(np.mean([r["opt_sec_per_step"] for r in adamw_runs])), 6),
        "opt_sec_per_step_muon": round(float(np.mean([r["opt_sec_per_step"] for r in muon_runs])), 6),
        "optimizer_step_share_of_wallclock": {
            "adamw": round(float(np.mean([r["opt_step_seconds"] / r["train_seconds"]
                                          for r in adamw_runs])), 4),
            "muon": round(float(np.mean([r["opt_step_seconds"] / r["train_seconds"]
                                         for r in muon_runs])), 4),
        },
        "overhead_factor_by_batch_size": {k: v["overhead_factor"] for k, v in bench.items()},
        "overhead_pct_by_tokens_per_step": {str(v["tokens_per_step"]): v["overhead_pct"]
                                            for v in bench.values()},
        "keller_rule_pct_here": round(100 * p["ns_steps"] * p["d_model"] / tokens_per_step, 3),
        "keller_rule_pct_nanogpt_speedrun": round(100 * p["ns_steps"] * 768 / 524288, 4),
        "ns_flops_per_step": int(ns_fl),
        "model_flops_per_step_fwd_bwd_approx": int(model_flops),
        "ns_flop_overhead_pct_measured_shapes": round(100 * ns_fl / model_flops, 3),
        "ns_precision_probe_seconds": ns_prec,

        "steps_for_muon_to_reach_adamw_final_bpc": m_reach_st,
        "steps_for_adamw_to_reach_its_final_bpc": a_reach_st,
        "step_domain_speedup_of_muon": step_speedup,
        "seconds_for_muon_to_reach_adamw_final_bpc": m_reach_s,
        "seconds_for_adamw_to_reach_its_final_bpc": a_reach_s,
        "wallclock_speedup_of_muon": time_speedup,

        "runs": runs,
        "mean_curves": res["mean_curves"],
        "micro_benchmark": res["micro_benchmark"],
        "wall_clock_s": round(time.time() - t_start, 1),
    }

    make_chart(res, HERE / "chart.png")

    results = {
        "id": cfg.get("id", "unknown"),
        "git_commit": git_sha(),
        "seed": int(cfg.get("seed", 0)),
        "duration_sec": round(time.time() - t_start, 2),
        "metrics": metrics,
        "env": env_info(),
        "status": "done",
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== summary ===")
    print(f"best lr: adamw {best_adamw_lr:g}  muon {best_muon_lr:g}")
    for k in ("adamw", "muon", "muon_isotime"):
        print(f"  {k:<13s} steps={S[k]['steps']:4d}  {S[k]['seconds_mean']:6.1f}s  "
              f"bpc={S[k]['mean']:.4f}  per-seed {S[k]['per_seed']}")
    print(f"mean seed spread: {seed_spread:.4f} bpc")
    print(f"MATCHED STEPS   muon - adamw = {d_steps:+.4f}  -> {verdict(d_steps, seed_spread)}")
    print(f"MATCHED SECONDS iso  - adamw = {d_sec_retrained:+.4f}  -> "
          f"{verdict(d_sec_retrained, seed_spread)}")
    print(f"overhead factor in training: x{overhead_train:.3f} "
          f"({100*(overhead_train-1):.1f}%);  Keller rule predicts "
          f"{100*p['ns_steps']*p['d_model']/tokens_per_step:.1f}%")
    print(f"step-domain speedup {step_speedup}  |  wall-clock speedup {time_speedup}")
    print(f"total {results['duration_sec']:.0f}s")


if __name__ == "__main__":
    main()
