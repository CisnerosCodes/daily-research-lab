"""Does a weight-tied looped block learn to scale its recursion with PARSE DEPTH?

Registry context this experiment builds on:
  2026-07-25_loop-test-time-compute  -- stochastic depth k~U{1..K} at train time is the
      recipe that turns a tied loop into a real test-time-compute axis; fixed-K degrades
      past its trained depth.
  2026-07-26_looped-halt-nrasp       -- ... but that was on prefix-parity, where train
      LENGTH was locked to train DEPTH. The correction recorded there is that fixed-K
      fails only under that locking.
  2026-07-25_filler-vs-recur         -- loops saturate without intermediate supervision.
  2026-07-25_trm-nano-sudoku         -- answer-space refinement beats latent recursion.

All four of those tasks are SERIAL (a chain). This one is HIERARCHICAL: ListOps-mini,
nested prefix expressions [MAX 2 [MIN 4 7] 9 [MED 1 5 8]] over digits 0-9. A depth-d
expression needs d rounds of bottom-up reduction, so "iterations = parse depth" is the
natural hypothesis. Train on nesting depth 1-4, test on depth 5-6.

CRITICAL CONTROL: depth and length are DECOUPLED by construction. Every expression at
every depth is grown (by adding extra arguments / shallow siblings, never raising the
height) to a target token length drawn from the SAME band, so a depth-6 test expression
is length-in-distribution w.r.t. depth 2-4 training expressions. Without this the
headline would be confounded with plain length extrapolation.

Arms (matched params, matched steps, matched batch stream per seed):
  (a) fixed_k4     -- always 4 loops, loss on the last one
  (b) stoch_k1to4  -- k ~ U{1..4} per batch, loss on the last one
  (c) deepsup_k4   -- always 4 loops, loss at EVERY iteration (the filler-vs-recur probe)
Arm (c) of the backlog spec -- "K scaled to parse depth at TEST time only" -- costs no
extra training: it is read straight off the accuracy(depth, K) table, at K=d and K=d+2.

Bonus probe: linear probe on the latent at the ']' token of every subtree, per loop
iteration t and per subtree height h. Does iteration t resolve nesting level t?

Deterministic, CPU-only, single-threaded. Writes results.json + chart.png.
Usage:  python run.py
"""
import json, os, random, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)

import listops as LO

PILOT = os.environ.get("PILOT", "") == "1"


def set_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def git_sha():
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
    for mod in ("numpy", "torch", "matplotlib", "sklearn"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:
            pass
    return info


# ------------------------------ model --------------------------------------
class Block(nn.Module):
    """Pre-LN transformer block, BIDIRECTIONAL attention (this is a classifier, the
    whole expression is available); PAD positions are masked out of every attention."""

    def __init__(self, d, h, dff):
        super().__init__()
        self.h, self.dh = h, d // h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.fc1, self.fc2 = nn.Linear(d, dff), nn.Linear(dff, d)

    def forward(self, x, keep):
        B, T, D = x.shape
        z = self.ln1(x)
        q, k, v = self.qkv(z).split(D, dim=2)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (self.dh ** 0.5)
        att = att.masked_fill(~keep[:, None, None, :], float("-inf")).softmax(dim=-1)
        x = x + self.proj((att @ v).transpose(1, 2).reshape(B, T, D))
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class LoopModel(nn.Module):
    """One weight-tied block applied K times with recurrent-depth-style per-iteration
    input injection (adapter on [h ; e]). Answer read off the CLS position (index 0)."""

    def __init__(self, p):
        super().__init__()
        d, h, dff = p["d_model"], p["n_heads"], p["d_ff"]
        self.emb = nn.Embedding(LO.VOCAB, d)
        self.pos = nn.Parameter(torch.zeros(p["max_len"], d))
        nn.init.normal_(self.pos, std=0.02)
        self.block = Block(d, h, dff)
        self.adapter = nn.Linear(2 * d, d, bias=False)
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, 10)

    def run(self, tok, K, collect=False):
        """Returns list of CLS logits, one per iteration 1..K (and optionally the full
        hidden states per iteration, for the probe)."""
        keep = tok != LO.PAD
        e = self.emb(tok) + self.pos[None, : tok.shape[1]]
        x = e
        logits, hiddens = [], []
        for _ in range(K):
            x = self.block(self.adapter(torch.cat([x, e], dim=-1)), keep)
            logits.append(self.head(self.ln_f(x[:, 0])))
            if collect:
                hiddens.append(x.detach())
        return (logits, hiddens) if collect else (logits, None)


# ------------------------------ data ---------------------------------------
def build_data(cfg):
    p = cfg["params"]
    lo, hi, ml = p["len_lo"], p["len_hi"], p["max_len"]
    tr_depths = p["train_depths"]
    n_train = p["n_train"] if not PILOT else 8000

    rng = random.Random(int(cfg["seed"]) * 1000 + 11)
    tr_ids, tr_ans, tr_d = [], [], []
    per = n_train // len(tr_depths)
    for d in tr_depths:
        seen = set()
        while len(seen) < per:
            node = LO.gen_expr(rng, d, lo, hi)
            ids, a, _, L = LO.encode(node, ml)
            key = tuple(ids[:L])
            if key in seen:
                continue
            seen.add(key)
            tr_ids.append(ids); tr_ans.append(a); tr_d.append(d)
    train_keys = set(tuple(i) for i in tr_ids)

    n_eval = p["n_eval"] if not PILOT else 192
    ev = {}
    lens_by_depth, arity_by_depth, bag_by_depth = {}, {}, {}
    for d in p["eval_depths"]:
        ids, ans, spans, lens, ars, bag = [], [], [], [], [], []
        seen = set()
        while len(ids) < n_eval:
            node = LO.gen_expr(rng, d, lo, hi)
            i, a, s, L = LO.encode(node, ml)
            key = tuple(i)
            if key in seen or key in train_keys:
                continue
            seen.add(key)
            ids.append(i); ans.append(a); spans.append(s); lens.append(L - 1)
            ars.append(max(len(n.kids) for n in LO._internal_nodes(node, [])))
            bag.append(int(LO.bag_heuristic(node) == a))
        ev[d] = {"ids": torch.tensor(ids), "ans": torch.tensor(ans), "spans": spans}
        lens_by_depth[d] = float(np.mean(lens))
        arity_by_depth[d] = float(np.mean(ars))
        bag_by_depth[d] = float(np.mean(bag))
    stats = {
        "mean_expr_tokens_by_depth": {str(k): round(v, 2) for k, v in lens_by_depth.items()},
        "mean_max_arity_by_depth": {str(k): round(v, 2) for k, v in arity_by_depth.items()},
        "majority_baseline_by_depth": {
            str(d): round(float(np.bincount(ev[d]["ans"].numpy(), minlength=10).max())
                          / len(ev[d]["ans"]), 4) for d in ev},
        # THE control that makes every number below interpretable: apply the ROOT operator
        # to the flat multiset of ALL digits in the expression, ignoring all structure.
        # A model that only learns this shortcut cannot beat this line.
        "bag_of_digits_shortcut_by_depth": {str(k): round(v, 4) for k, v in bag_by_depth.items()},
        "chance": 0.1,
        "n_train": len(tr_ids), "n_eval_per_depth": n_eval,
        "eval_examples_also_in_train": 0,
        "token_len_band": [lo, hi],
    }
    return (torch.tensor(tr_ids), torch.tensor(tr_ans), torch.tensor(tr_d)), ev, stats


# ------------------------------ train --------------------------------------
def train_arm(arm, seed, train, cfg, log):
    p = cfg["params"]
    steps = p["steps"] if not PILOT else 300
    K = p["k_train"]
    tr_ids, tr_ans, _ = train

    set_seeds(seed)
    model = LoopModel(p)
    nparam = sum(q.numel() for q in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["wd"])
    gen = torch.Generator().manual_seed(seed + 777)   # SAME batch stream for every arm
    kgen = torch.Generator().manual_seed(seed + 999)

    warm, B = p["warmup"], p["batch"]
    loops_used = 0
    t0 = time.time()
    loss = torch.tensor(0.0)
    for step in range(steps):
        lr = p["lr"] * (step + 1) / warm if step < warm else p["lr"] * (
            0.1 + 0.45 * (1 + np.cos(np.pi * (step - warm) / max(1, steps - warm))))
        for g in opt.param_groups:
            g["lr"] = lr
        idx = torch.randint(0, len(tr_ids), (B,), generator=gen)
        tok, y = tr_ids[idx], tr_ans[idx]

        if arm == "stoch_k1to4":
            k = int(torch.randint(1, K + 1, (1,), generator=kgen).item())
        else:
            k = K
        logits, _ = model.run(tok, k)
        if arm == "deepsup_k4":
            loss = sum(F.cross_entropy(lg, y) for lg in logits) / len(logits)
        else:
            loss = F.cross_entropy(logits[-1], y)
        loops_used += k
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0 or step == steps - 1:
            acc = (logits[-1].argmax(-1) == y).float().mean().item()
            log.append(f"{arm} s{seed} step {step:4d} loss {loss.item():.4f} bacc {acc:.3f} lr {lr:.2e}")
            print(log[-1], flush=True)
    return model, {
        "n_params": nparam,
        "train_seconds": round(time.time() - t0, 1),
        "mean_loops_per_step": round(loops_used / steps, 3),
        "block_applications_x_batch": loops_used * B,
        "final_train_loss": round(float(loss.item()), 4),
    }


# ------------------------------ eval ---------------------------------------
@torch.no_grad()
def eval_model(model, ev, k_max, chunk=128):
    """One forward pass to k_max, reading the answer out at EVERY iteration.
    Reading out at iteration t is bit-identical to having run the model with K=t."""
    out = {}
    for d, data in ev.items():
        ids, ans = data["ids"], data["ans"]
        correct = np.zeros(k_max)
        for i in range(0, len(ids), chunk):
            tok, y = ids[i:i + chunk], ans[i:i + chunk]
            logits, _ = model.run(tok, k_max)
            for t in range(k_max):
                correct[t] += (logits[t].argmax(-1) == y).sum().item()
        out[d] = (correct / len(ids)).round(4).tolist()
    return out


# ------------------------------ probe --------------------------------------
@torch.no_grad()
def collect_probe_data(model, ev, k_max, max_per_cell=1200):
    """Hidden state at every subtree's ']' token, per loop iteration, keyed by that
    subtree's HEIGHT. Question: at which iteration t does height-h become decodable?"""
    feats = {t: {} for t in range(k_max)}
    labels = {t: {} for t in range(k_max)}
    for d, data in ev.items():
        ids, spans = data["ids"], data["spans"]
        for i in range(0, len(ids), 128):
            tok = ids[i:i + 128]
            _, hid = model.run(tok, k_max, collect=True)
            for j, sp in enumerate(spans[i:i + 128]):
                for (close_idx, h, val) in sp:
                    if len(feats[0].setdefault(h, [])) >= max_per_cell:
                        continue
                    for t in range(k_max):
                        feats[t].setdefault(h, []).append(hid[t][j, close_idx].numpy())
                        labels[t].setdefault(h, []).append(val)
    return feats, labels


def run_probes(feats, labels, k_max, seed=0):
    from sklearn.linear_model import LogisticRegression
    heights = sorted(feats[0].keys())
    acc, maj = {}, {}
    for h in heights:
        for t in range(k_max):
            X = np.asarray(feats[t][h], dtype=np.float64)
            y = np.asarray(labels[t][h])
            if len(y) < 200 or len(set(y.tolist())) < 2:
                continue
            rs = np.random.RandomState(seed)
            perm = rs.permutation(len(y))
            X, y = X[perm], y[perm]
            ntr = int(0.7 * len(y))
            clf = LogisticRegression(max_iter=400, C=1.0)
            clf.fit(X[:ntr], y[:ntr])
            acc[f"h{h}_t{t+1}"] = round(float((clf.predict(X[ntr:]) == y[ntr:]).mean()), 4)
            maj[f"h{h}"] = round(float(np.bincount(y[ntr:], minlength=10).max() / len(y[ntr:])), 4)
    return acc, maj, heights


# ------------------------------ chart --------------------------------------
def make_chart(res, cfg, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = cfg["params"]
    k_max = p["k_eval_max"]
    arms = res["arms"]
    depths = p["eval_depths"]
    ks = list(range(1, k_max + 1))
    colors = {"fixed_k4": "#c1442e", "stoch_k1to4": "#1f6fb4", "deepsup_k4": "#2e8b57"}

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.8))

    # (1,2) accuracy vs test-time K at the two EXTRAPOLATION depths
    for col, dd in enumerate([5, 6]):
        ax = axes[0, col]
        for arm in arms:
            m = np.array(res["acc_mean"][arm][str(dd)])
            s = np.array(res["acc_std"][arm][str(dd)])
            ax.plot(ks, m, "o-", color=colors[arm], label=arm, lw=2)
            ax.fill_between(ks, m - s, m + s, color=colors[arm], alpha=0.16)
        ax.axhline(res["data_stats"]["majority_baseline_by_depth"][str(dd)], ls=":", c="gray",
                   label="majority baseline")
        ax.axhline(res["data_stats"]["bag_of_digits_shortcut_by_depth"][str(dd)], ls="--",
                   c="darkgreen", lw=1.2, label="bag-of-digits shortcut")
        ax.axvline(p["k_train"], ls="--", c="k", lw=1, alpha=.6)
        ax.axvline(dd, ls="-.", c="purple", lw=1, alpha=.6)
        lo_y = ax.get_ylim()[0]
        ax.text(p["k_train"] + .06, lo_y, " K_train", fontsize=8, color="k")
        ax.text(dd + .06, lo_y, " K=d", fontsize=8, color="purple")
        ax.set_title(f"HEADLINE: depth {dd} (beyond training)\naccuracy vs test-time loops")
        ax.set_xlabel("test-time loops K"); ax.set_ylabel("accuracy")
        ax.legend(fontsize=8); ax.grid(alpha=.25)

    # (3) accuracy vs nesting depth
    ax = axes[0, 2]
    for arm in arms:
        at_ktrain = [res["acc_mean"][arm][str(d)][p["k_train"] - 1] for d in depths]
        best = [max(res["acc_mean"][arm][str(d)]) for d in depths]
        ax.plot(depths, at_ktrain, "o-", color=colors[arm], lw=2, label=f"{arm} @K=4")
        ax.plot(depths, best, "s--", color=colors[arm], lw=1.4, alpha=.6, label=f"{arm} @best K")
    ax.plot(depths, [res["data_stats"]["majority_baseline_by_depth"][str(d)] for d in depths],
            ":", c="gray", label="majority")
    ax.plot(depths, [res["data_stats"]["bag_of_digits_shortcut_by_depth"][str(d)] for d in depths],
            "--", c="darkgreen", lw=1.4, label="bag-of-digits shortcut")
    ax.axvspan(4.5, 6.5, color="orange", alpha=.10)
    ax.text(5.5, ax.get_ylim()[1] * .95, "unseen depth", ha="center", fontsize=8, color="darkorange")
    ax.set_title("accuracy vs nesting depth"); ax.set_xlabel("nesting depth d")
    ax.set_ylabel("accuracy"); ax.legend(fontsize=7); ax.grid(alpha=.25)

    # (4) K-scaled-to-depth policies
    ax = axes[1, 0]
    w = 0.82 / (len(arms) * 3)
    xs = np.arange(len(depths))
    for ai, arm in enumerate(arms):
        for pi, (name, hatch) in enumerate([("K=4", ""), ("K=d", "//"), ("K=d+2", "xx")]):
            vals = []
            for d in depths:
                curve = res["acc_mean"][arm][str(d)]
                kk = {0: p["k_train"], 1: d, 2: d + 2}[pi]
                vals.append(curve[min(kk, k_max) - 1])
            ax.bar(xs + (ai * 3 + pi) * w - 0.41 + w / 2, vals, w, hatch=hatch,
                   color=colors[arm], alpha=[1.0, .70, .45][pi],
                   label=f"{arm} {name}", edgecolor="w", lw=.4)
    ax.set_xticks(xs); ax.set_xticklabels([f"d{d}" for d in depths])
    ax.set_title('arm (c): "scale K with parse depth", TEST time only')
    ax.set_ylabel("accuracy"); ax.legend(fontsize=6, ncol=2); ax.grid(alpha=.25, axis="y")

    # (5) gain from extra test-time compute, per depth
    ax = axes[1, 1]
    for arm in arms:
        dl = [max(res["acc_mean"][arm][str(d)][p["k_train"]:])
              - res["acc_mean"][arm][str(d)][p["k_train"] - 1] for d in depths]
        ax.plot(depths, dl, "o-", color=colors[arm], lw=2, label=arm)
    ax.axhline(0, c="k", lw=1)
    ax.axvspan(4.5, 6.5, color="orange", alpha=.10)
    ax.set_title("gain from loops BEYOND K_train\n(best K>4) - (K=4)")
    ax.set_xlabel("nesting depth d"); ax.set_ylabel("delta accuracy")
    ax.legend(fontsize=8); ax.grid(alpha=.25)

    # (6) probe heat map
    ax = axes[1, 2]
    pr = res.get("probe", {})
    if pr.get("acc"):
        hs = pr["heights"]
        M = np.full((len(hs), k_max), np.nan)
        for hi, h in enumerate(hs):
            for t in range(k_max):
                v = pr["acc"].get(f"h{h}_t{t+1}")
                if v is not None:
                    M[hi, t] = v
        im = ax.imshow(M, cmap="viridis", aspect="auto", origin="lower",
                       extent=[0.5, k_max + .5, hs[0] - .5, hs[-1] + .5])
        plt.colorbar(im, ax=ax, fraction=.046)
        ax.plot(hs, hs, "w--", lw=1.4, label="t = h (hypothesis)")
        for hi, h in enumerate(hs):
            row = M[hi]
            if not np.all(np.isnan(row)):
                ax.plot(int(np.nanargmax(row)) + 1, h, "r*", ms=12)
        ax.set_xlabel("loop iteration t"); ax.set_ylabel("subtree height h")
        ax.set_title(f"probe: subtree value from its ']' latent\n({pr['probe_arm']}; red star = argmax t)")
        ax.legend(fontsize=7, loc="lower right")
    else:
        ax.text(.5, .5, "probe skipped", ha="center"); ax.axis("off")

    fig.suptitle("ListOps-mini: does a looped tied block scale its recursion with PARSE DEPTH?"
                 "   (train depth 1-4, test 5-6, length-matched)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=125)


# ------------------------------ main ---------------------------------------
def main():
    cfg = load_config()
    p = cfg["params"]
    seed = int(cfg["seed"])
    set_seeds(seed)
    t0 = time.time()
    log = []

    train, ev, dstats = build_data(cfg)
    print("data:", json.dumps(dstats), flush=True)

    arms = p["arms"]
    seeds = p["seeds"] if not PILOT else [0]
    k_max = p["k_eval_max"]

    per_run, models = {}, {}
    acc_runs = {a: [] for a in arms}
    for arm in arms:
        for s in seeds:
            model, info = train_arm(arm, s, train, cfg, log)
            a = eval_model(model, ev, k_max)
            acc_runs[arm].append(a)
            per_run[f"{arm}_seed{s}"] = {**info,
                                         "acc_by_depth_by_K": {str(k): v for k, v in a.items()}}
            models[(arm, s)] = model
            print(f"  {arm} s{s}: d5 {a[5]}  d6 {a[6]}  ({info['train_seconds']}s)", flush=True)

    acc_mean = {a: {str(d): np.mean([r[d] for r in acc_runs[a]], axis=0).round(4).tolist()
                    for d in ev} for a in arms}
    acc_std = {a: {str(d): np.std([r[d] for r in acc_runs[a]], axis=0).round(4).tolist()
                   for d in ev} for a in arms}

    probe = {}
    probe_arm = p.get("probe_arm", "stoch_k1to4")
    probe_arm = probe_arm if probe_arm in arms else arms[0]
    if p.get("run_probe", True):
        tp = time.time()
        feats, labels = collect_probe_data(models[(probe_arm, seeds[0])], ev, k_max)
        pacc, pmaj, heights = run_probes(feats, labels, k_max, seed=seed)
        probe = {
            "probe_arm": probe_arm, "probe_seed": seeds[0], "acc": pacc,
            "majority_by_height": pmaj, "heights": heights,
            "argmax_iteration_by_height": {
                f"h{h}": int(np.nanargmax([pacc.get(f"h{h}_t{t+1}", np.nan) for t in range(k_max)])) + 1
                for h in heights if any(f"h{h}_t{t+1}" in pacc for t in range(k_max))},
            "seconds": round(time.time() - tp, 1)}
        print("probe argmax t by height:",
              json.dumps(probe["argmax_iteration_by_height"]), flush=True)

    # ---------------- derived headline numbers ----------------
    kt, deep = p["k_train"], [5, 6]
    headline = {}
    for arm in arms:
        at_kt = {d: acc_mean[arm][str(d)][kt - 1] for d in ev}
        best_k = {d: int(np.argmax(acc_mean[arm][str(d)])) + 1 for d in ev}
        best_v = {d: max(acc_mean[arm][str(d)]) for d in ev}
        gain = {d: round(max(acc_mean[arm][str(d)][kt:]) - at_kt[d], 4) for d in ev}
        h = {
            "acc_at_Ktrain_by_depth": {str(d): round(at_kt[d], 4) for d in ev},
            "acc_at_bestK_by_depth": {str(d): round(best_v[d], 4) for d in ev},
            "bestK_by_depth": {str(d): best_k[d] for d in ev},
            "gain_from_K_beyond_Ktrain_by_depth": {str(d): gain[d] for d in ev},
            "acc_at_K_eq_d_by_depth": {str(d): acc_mean[arm][str(d)][min(d, k_max) - 1] for d in ev},
            "acc_at_K_eq_d_plus_2_by_depth": {str(d): acc_mean[arm][str(d)][min(d + 2, k_max) - 1] for d in ev},
            "HEADLINE_acc_deep_5_6_at_Ktrain": round(float(np.mean([at_kt[d] for d in deep])), 4),
            "HEADLINE_acc_deep_5_6_at_bestK": round(float(np.mean([best_v[d] for d in deep])), 4),
            "HEADLINE_gain_deep_5_6_from_extra_K": round(float(np.mean([gain[d] for d in deep])), 4),
            "acc_indist_1_4_at_Ktrain": round(float(np.mean([at_kt[d] for d in [1, 2, 3, 4]])), 4),
            "degradation_at_Kmax_vs_Ktrain_indist": round(float(np.mean(
                [acc_mean[arm][str(d)][k_max - 1] - at_kt[d] for d in [1, 2, 3, 4]])), 4),
            "margin_over_bag_shortcut_at_bestK_by_depth": {
                str(d): round(best_v[d] - dstats["bag_of_digits_shortcut_by_depth"][str(d)], 4)
                for d in ev},
        }
        h["policy_deep_5_6"] = {
            "K_fixed_4": round(float(np.mean([h["acc_at_Ktrain_by_depth"][str(d)] for d in deep])), 4),
            "K_eq_d": round(float(np.mean([h["acc_at_K_eq_d_by_depth"][str(d)] for d in deep])), 4),
            "K_eq_d_plus_2": round(float(np.mean([h["acc_at_K_eq_d_plus_2_by_depth"][str(d)] for d in deep])), 4),
        }
        headline[arm] = h

    metrics = {
        "headline": "accuracy at nesting depth 5-6 (beyond training) per arm, and whether "
                    "test-time loops K>K_train buy anything on deeper expressions",
        "arms": arms, "seeds": seeds, "k_train": kt, "k_eval_max": k_max,
        "steps": p["steps"], "batch": p["batch"],
        "data_stats": dstats,
        "acc_mean": acc_mean, "acc_std": acc_std,
        "per_run": per_run, "by_arm": headline, "probe": probe,
    }
    make_chart(metrics, cfg, HERE / "chart.png")

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
    with open(HERE / "train.log", "w") as f:
        f.write("\n".join(log) + "\n")
    print(json.dumps(headline, indent=2))
    print("TOTAL", results["duration_sec"], "s")


if __name__ == "__main__":
    main()
