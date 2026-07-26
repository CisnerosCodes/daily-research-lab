"""Does a weight-tied looped block trade TEST-TIME compute for accuracy?

Train a tiny (0.06M param) weight-tied transformer block at K=3 loop iterations, then at
test time run K in {1..8} and ask whether accuracy on HARDER instances improves past the
trained K -- or degrades, which is the common failure.

Task: prefix-parity (cumulative XOR). Input  BOS x_1 .. x_L  with x_i in {0,1};
target y_i = XOR(x_1..x_i). The attention window is 1 (every position attends only to
itself and its immediate predecessor), so after K applications of the block position i
can only have seen bits [i-K, i]. Therefore

    difficulty d (= position index) is SOLVABLE iff  K >= d - 1

i.e. the number of required sequential steps IS the difficulty, and extra test-time loops
are the only way to solve deeper instances. Train on L <= 4 (exactly what K=3 covers),
test on L = 9 with K up to 8 (exactly what K=8 covers).

Arms: tied loop w/ input injection, tied loop w/o injection, tied loop trained with a
stochastic depth schedule, and an untied depth-3 control (2.6x params) that structurally
cannot extend.

Deterministic, CPU-only, single-threaded. Writes results.json + chart.png.
Usage:  python run.py
"""
import json, os, random, subprocess, sys, time
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
def make_batch(n, L, gen, vocab_bos=2):
    """tokens (n, L+1) = [BOS, x_1..x_L];  targets (n, L) = cumulative XOR."""
    bits = torch.randint(0, 2, (n, L), generator=gen)
    tgt = torch.cumsum(bits, dim=1) % 2
    tok = torch.cat([torch.full((n, 1), vocab_bos, dtype=torch.long), bits], dim=1)
    return tok, tgt


# ----------------------------- model ---------------------------------------
def band_mask(T, window):
    """(T, T) bool, True = allowed. Position i attends to [i-window, i]."""
    idx = torch.arange(T)
    d = idx[:, None] - idx[None, :]
    return (d >= 0) & (d <= window)


class Block(nn.Module):
    """Pre-LN transformer block with a banded causal attention window."""

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
    def __init__(self, cfg, tied, inject):
        super().__init__()
        p = cfg["params"]
        d, h, dff = p["d_model"], p["n_heads"], p["d_ff"]
        self.tied, self.inject = tied, inject
        self.window = p["attn_window"]
        self.emb = nn.Embedding(p["vocab"], d)
        n_blocks = 1 if tied else p["k_train"]
        self.blocks = nn.ModuleList([Block(d, h, dff) for _ in range(n_blocks)])
        self.adapter = nn.Linear(2 * d, d, bias=False) if inject else None
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, 2)

    def forward(self, tok, K, mode="normal", return_norms=False):
        """mode: 'normal' (tied: repeat the single block; untied: clamp at the last block)
                 'cycle'  (untied only, K>depth: cycle blocks 0,1,2,0,1,2,...)
                 'last'   (untied only, K>depth: run 0,1,2 then repeat block 2)"""
        T = tok.shape[1]
        mask = band_mask(T, self.window)
        e = self.emb(tok)
        hh = e
        norms = []
        nb = len(self.blocks)
        for t in range(K):
            if self.tied:
                blk = self.blocks[0]
            elif mode == "cycle":
                blk = self.blocks[t % nb]
            else:  # "normal" and "last" agree: clamp at the final block
                blk = self.blocks[min(t, nb - 1)]
            if self.inject:
                hh = self.adapter(torch.cat([hh, e], dim=-1))
            hh = blk(hh, mask)
            if return_norms:
                norms.append(float(hh.norm(dim=-1).mean()))
        logits = self.head(self.ln_f(hh))[:, 1:, :]  # drop BOS position
        return (logits, norms) if return_norms else logits


# ----------------------------- train / eval --------------------------------
def train_arm(cfg, arm, seed, log):
    p = cfg["params"]
    set_seeds(seed)
    model = LoopModel(cfg, arm["tied"], arm["inject"])
    n_params = sum(q.numel() for q in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    gen = torch.Generator().manual_seed(seed * 1000 + 7)
    rng = random.Random(seed * 1000 + 13)

    t0 = time.time()
    step, losses, hit_cap = 0, [], False
    for step in range(1, p["steps"] + 1):
        if arm["k_schedule"] == "random":
            k = rng.randint(1, p["k_train"])
            L = k + 1
        else:
            k = p["k_train"]
            L = rng.randint(p["train_len_min"], p["train_len_max"])
        tok, tgt = make_batch(p["batch_size"], L, gen)
        logits = model(tok, k)
        loss = F.cross_entropy(logits.reshape(-1, 2), tgt.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), p["grad_clip"])
        lr = p["lr"] * min(1.0, step / max(1, p["warmup"]))   # linear warmup
        for g in opt.param_groups:
            g["lr"] = lr
        opt.step()
        losses.append(float(loss.detach()))
        if step % 500 == 0:
            log(f"    {arm['name']} seed{seed} step {step:5d} loss {np.mean(losses[-200:]):.4f} "
                f"({time.time()-t0:.0f}s)")
        if time.time() - t0 > p["time_cap_s_per_run"]:
            hit_cap = True
            break
    return model, {
        "n_params": n_params,
        "steps_run": step,
        "final_train_loss": round(float(np.mean(losses[-200:])), 5),
        "hit_time_cap": hit_cap,
        "train_sec": round(time.time() - t0, 1),
    }


@torch.no_grad()
def eval_arm(cfg, model, mode="normal"):
    """Returns acc[K][d] over difficulties d = 1..test_len, plus exact-match and norms."""
    p = cfg["params"]
    gen = torch.Generator().manual_seed(p["eval_seed"])
    tok, tgt = make_batch(p["eval_n"], p["test_len"], gen)
    out = {}
    for K in p["test_k_values"]:
        logits, norms = model(tok, K, mode=mode, return_norms=True)
        pred = logits.argmax(-1)
        correct = (pred == tgt).float()                        # (N, L)
        per_pos = correct.mean(0).tolist()                     # difficulty d = index+1
        prefix_em = correct.cumprod(dim=1).mean(0).tolist()    # exact match on first d positions
        out[K] = {
            "acc_by_difficulty": [round(a, 4) for a in per_pos],
            "prefix_exact_match": [round(a, 4) for a in prefix_em],
            "exact_match_full": round(float(correct.prod(dim=1).mean()), 4),
            "resid_norm_by_iter": [round(n, 3) for n in norms],
        }
    return out


def deepest_solved(acc_by_difficulty, thr):
    """Largest d such that every difficulty <= d is at >= thr accuracy."""
    d = 0
    for i, a in enumerate(acc_by_difficulty):
        if a >= thr:
            d = i + 1
        else:
            break
    return d


# ----------------------------- main ----------------------------------------
def main():
    cfg = load_config()
    p = cfg["params"]
    seed = int(cfg.get("seed", 0))
    t_start = time.time()
    logf = open(HERE / "train.log", "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(f"=== {cfg['id']} ===")
    log(f"arms={[a['name'] for a in p['arms']]} seeds={p['seeds']}")

    runs = {}   # arm -> seed -> {"train": ..., "eval": {mode: {K: {...}}}}
    for arm in p["arms"]:
        runs[arm["name"]] = {}
        for s in p["seeds"]:
            log(f"  [{arm['name']}] seed {s} ...")
            model, tr = train_arm(cfg, arm, s, log)
            model.eval()
            ev = {"normal": eval_arm(cfg, model, "normal")}
            if not arm["tied"]:
                ev["cycle"] = eval_arm(cfg, model, "cycle")
            runs[arm["name"]][s] = {"train": tr, "eval": ev}
            k3 = ev["normal"][p["k_train"]]["acc_by_difficulty"]
            k8 = ev["normal"][max(p["test_k_values"])]["acc_by_difficulty"]
            log(f"    params={tr['n_params']} loss={tr['final_train_loss']:.4f} "
                f"steps={tr['steps_run']} {tr['train_sec']}s")
            log(f"    K=3 acc/difficulty {['%.2f' % a for a in k3]}")
            log(f"    K=8 acc/difficulty {['%.2f' % a for a in k8]}")

    # ---------------- aggregate across seeds ----------------
    K_list = p["test_k_values"]
    D = p["test_len"]
    THR = p["solve_threshold"]
    agg = {}
    for name, per_seed in runs.items():
        modes = list(next(iter(per_seed.values()))["eval"].keys())
        agg[name] = {}
        for mode in modes:
            m = {}
            for K in K_list:
                A = np.array([per_seed[s]["eval"][mode][K]["acc_by_difficulty"] for s in p["seeds"]])
                E = np.array([per_seed[s]["eval"][mode][K]["exact_match_full"] for s in p["seeds"]])
                m[K] = {
                    "acc_mean": [round(x, 4) for x in A.mean(0).tolist()],
                    "acc_std": [round(x, 4) for x in A.std(0).tolist()],
                    "exact_match_full_mean": round(float(E.mean()), 4),
                    "deepest_solved_mean": round(float(np.mean([deepest_solved(a, THR) for a in A])), 3),
                    "deepest_solved_per_seed": [deepest_solved(a, THR) for a in A],
                }
            agg[name][mode] = m

    # ---------------- headline metrics ----------------
    K_TR = p["k_train"]
    K_MAX = max(K_list)
    HEAD_ARM = "tied_fixK3_inj"
    head = agg[HEAD_ARM]["normal"]

    def acc(name, mode, K, d):
        return agg[name][mode][K]["acc_mean"][d - 1]

    # (1) does EXTRA test-time compute help on HARDER instances?
    hard_gain = {}
    for name in agg:
        g = {}
        m = agg[name]["normal"]
        for d in range(K_TR + 2, D + 1):          # difficulties unreachable at K_train
            best_k = max(K_list, key=lambda K: m[K]["acc_mean"][d - 1])
            g[f"d{d}"] = {
                "acc_at_Ktrain": round(m[K_TR]["acc_mean"][d - 1], 4),
                "acc_at_Kmax": round(m[K_MAX]["acc_mean"][d - 1], 4),
                "acc_at_required_K": round(m[min(d - 1, K_MAX)]["acc_mean"][d - 1], 4),
                "best_K": best_k,
                "best_acc": round(m[best_k]["acc_mean"][d - 1], 4),
            }
        hard_gain[name] = g

    # (2) does accuracy on EASY (trained-depth) instances degrade past K_train?
    degrade = {}
    for name in agg:
        m = agg[name]["normal"]
        base = np.mean([m[K_TR]["acc_mean"][d - 1] for d in range(1, K_TR + 2)])
        far = np.mean([m[K_MAX]["acc_mean"][d - 1] for d in range(1, K_TR + 2)])
        degrade[name] = {
            "easy_acc_at_Ktrain": round(float(base), 4),
            "easy_acc_at_Kmax": round(float(far), 4),
            "delta": round(float(far - base), 4),
        }

    # (3) monotonicity of accuracy in K for each difficulty (over K = 1..min(d-1, Kmax))
    mono = {}
    for name in agg:
        mm = {}
        m = agg[name]["normal"]
        for d in range(1, D + 1):
            ks = [K for K in K_list if K <= max(1, d - 1)]
            vals = [m[K]["acc_mean"][d - 1] for K in ks]
            mm[f"d{d}"] = {
                "K_range": ks,
                "acc": [round(v, 4) for v in vals],
                "nondecreasing": all(vals[i + 1] >= vals[i] - 1e-9 for i in range(len(vals) - 1)),
            }
        mono[name] = mm

    # (4) FRONTIER accuracy: accuracy at difficulty d = K+1, the deepest instance that K
    #     loop iterations can solve at all. This is the single cleanest "did extra test-time
    #     compute buy new capability" curve: an ideal extrapolator sits at 1.0 for every K.
    frontier = {}
    frontier_seed = {}
    for name in agg:
        m = agg[name]["normal"]
        frontier[name] = {str(K): m[K]["acc_mean"][K] for K in K_list if K + 1 <= D}
        frontier_seed[name] = {
            str(K): [round(runs[name][s]["eval"]["normal"][K]["acc_by_difficulty"][K], 4)
                     for s in p["seeds"]]
            for K in K_list if K + 1 <= D
        }
    # easy-instance (trained-depth, d <= K_train+1) accuracy as a function of test K
    easy_by_K = {
        name: {str(K): round(float(np.mean(agg[name]["normal"][K]["acc_mean"][:K_TR + 1])), 4)
               for K in K_list}
        for name in agg
    }

    # (5) ideal-staircase agreement: deepest solved should equal K+1
    stair = {}
    for name in agg:
        m = agg[name]["normal"]
        stair[name] = {str(K): m[K]["deepest_solved_mean"] for K in K_list}
    stair["_ideal"] = {str(K): min(K + 1, D) for K in K_list}

    metrics = {
        "task": "prefix-parity (cumulative XOR), attention window 1 => difficulty d needs K >= d-1",
        "k_train": K_TR,
        "test_k_values": K_list,
        "test_len": D,
        "train_len_range": [p["train_len_min"], p["train_len_max"]],
        "chance_acc": 0.5,
        "eval_n": p["eval_n"],
        "n_params_by_arm": {n: runs[n][p["seeds"][0]]["train"]["n_params"] for n in runs},
        "final_train_loss_by_arm": {
            n: {str(s): runs[n][s]["train"]["final_train_loss"] for s in p["seeds"]} for n in runs
        },
        "steps_run_by_arm": {
            n: {str(s): runs[n][s]["train"]["steps_run"] for s in p["seeds"]} for n in runs
        },
        "hit_time_cap_by_arm": {
            n: {str(s): runs[n][s]["train"]["hit_time_cap"] for s in p["seeds"]} for n in runs
        },
        "train_sec_by_arm": {
            n: {str(s): runs[n][s]["train"]["train_sec"] for s in p["seeds"]} for n in runs
        },
        "acc_matrix_by_arm": {
            n: {mode: {str(K): agg[n][mode][K]["acc_mean"] for K in K_list} for mode in agg[n]}
            for n in agg
        },
        "acc_std_by_arm": {
            n: {str(K): agg[n]["normal"][K]["acc_std"] for K in K_list} for n in agg
        },
        "exact_match_full_by_arm": {
            n: {mode: {str(K): agg[n][mode][K]["exact_match_full_mean"] for K in K_list}
                for mode in agg[n]} for n in agg
        },
        "solve_threshold": THR,
        "frontier_acc_by_arm": frontier,
        "frontier_acc_per_seed_by_arm": frontier_seed,
        "easy_acc_by_test_K_by_arm": easy_by_K,
        "acc_matrix_per_seed_by_arm": {
            n: {str(s): {str(K): runs[n][s]["eval"]["normal"][K]["acc_by_difficulty"]
                         for K in K_list} for s in p["seeds"]} for n in runs
        },
        "deepest_solved_by_arm": stair,
        "deepest_solved_per_seed": {
            n: {str(K): agg[n]["normal"][K]["deepest_solved_per_seed"] for K in K_list} for n in agg
        },
        "hard_instance_gain_by_arm": hard_gain,
        "easy_instance_degradation_by_arm": degrade,
        "monotonicity_by_arm": mono,
        "resid_norm_by_iter_at_Kmax": {
            n: runs[n][p["seeds"][0]]["eval"]["normal"][K_MAX]["resid_norm_by_iter"] for n in runs
        },
        "seeds": p["seeds"],
    }

    # verdict
    hard_ds = [f"d{d}" for d in range(K_TR + 2, D + 1)]
    verdict = {"n_hard_difficulties": len(hard_ds)}
    beyond = [K for K in K_list if K > K_TR and K + 1 <= D]
    for name in agg:
        verdict[name] = {
            "frontier_acc_at_Ktrain": frontier[name][str(K_TR)],
            "frontier_acc_beyond_Ktrain_mean": round(
                float(np.mean([frontier[name][str(K)] for K in beyond])), 4),
            "frontier_acc_at_Kmax": frontier[name][str(K_MAX)],
            "n_hard_difficulties_monotone_in_K": sum(
                1 for k in hard_ds if mono[name][k]["nondecreasing"]),
            "extra_test_compute_helps_hard": bool(all(
                hard_gain[name][k]["acc_at_Kmax"] > hard_gain[name][k]["acc_at_Ktrain"] + 0.02
                for k in hard_ds)),
            "easy_degrades_past_Ktrain": bool(degrade[name]["delta"] < -0.02),
            "stair_matches_ideal_n_of_%d" % len(K_list): sum(
                1 for K in K_list if abs(stair[name][str(K)] - min(K + 1, D)) < 1e-9),
        }
    metrics["verdict"] = verdict

    results = {
        "id": cfg.get("id", "unknown"),
        "git_commit": git_sha(),
        "seed": seed,
        "duration_sec": round(time.time() - t_start, 2),
        "metrics": metrics,
        "env": env_info(),
        "status": "done",
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    make_chart(cfg, metrics)
    log(f"TOTAL {results['duration_sec']}s")
    log(json.dumps(verdict, indent=2))
    logf.close()


# ----------------------------- chart ---------------------------------------
def make_chart(cfg, metrics):
    """Rebuilt purely from `metrics`, so `python run.py --chart-only` can redraw it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = cfg["params"]
    K_list, D, K_TR = p["test_k_values"], p["test_len"], p["k_train"]
    A = metrics["acc_matrix_by_arm"]
    arms = list(A.keys())
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    def heat(axis, name, tag):
        M = np.array([A[name]["normal"][str(K)] for K in K_list])
        im = axis.imshow(M, vmin=0.5, vmax=1.0, cmap="viridis", aspect="auto", origin="lower")
        axis.set_xticks(range(D), [str(d) for d in range(1, D + 1)])
        axis.set_yticks(range(len(K_list)), [str(k) for k in K_list])
        axis.set_xlabel("difficulty d  (= position; requires K >= d-1 sequential steps)")
        axis.set_ylabel("test-time loop iterations K")
        axis.set_title(f"{tag} {name}\naccuracy(test K, difficulty); white dashed = ideal frontier d = K+1")
        axis.plot(range(len(K_list)), [min(K_list[i] + 1, D) - 1 for i in range(len(K_list))],
                  "w--", lw=1.6)
        if K_TR in K_list:
            axis.axhline(K_list.index(K_TR), color="red", lw=1.2, ls=":")
            axis.text(-0.42, K_list.index(K_TR) + 0.18, "K_train", color="red", fontsize=8)
        for i in range(len(K_list)):
            for j in range(D):
                axis.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                          color="w" if M[i, j] < 0.8 else "k")
        fig.colorbar(im, ax=axis, label="bit accuracy (mean over seeds)")

    heat(ax[0, 0], "tied_fixK3_inj", "(a) FIXED K=3 training -")
    heat(ax[0, 1], "tied_randK_inj", "(b) STOCHASTIC K~U{1,2,3} training -")

    # (c) frontier accuracy: accuracy at the deepest difficulty K loops could solve (d = K+1)
    F = metrics["frontier_acc_by_arm"]
    ks = [K for K in K_list if K + 1 <= D]
    for name in arms:
        ax[1, 0].plot(ks, [F[name][str(K)] for K in ks], marker="o", ms=5, label=name)
    ax[1, 0].axhline(1.0, color="k", ls=":", lw=2, label="ideal extrapolator")
    ax[1, 0].axhline(0.5, color="gray", ls="--", lw=0.8, label="chance")
    ax[1, 0].axvline(K_TR, color="red", ls=":", lw=1.2)
    ax[1, 0].text(K_TR + 0.07, 0.44, "K_train=3", color="red", fontsize=8)
    ax[1, 0].set_xlabel("test-time loop iterations K")
    ax[1, 0].set_ylabel("accuracy at the frontier difficulty d = K+1")
    ax[1, 0].set_ylim(0.38, 1.04)
    ax[1, 0].set_title("(c) HEADLINE: does one more loop buy one more step of reasoning?")
    ax[1, 0].legend(fontsize=8)
    ax[1, 0].grid(alpha=0.25)

    # (d) does easy (trained-depth) accuracy survive extra loops?
    E = metrics["easy_acc_by_test_K_by_arm"]
    for name in arms:
        ax[1, 1].plot(K_list, [E[name][str(K)] for K in K_list], marker="s", ms=5, label=name)
    ax[1, 1].axhline(0.5, color="gray", ls="--", lw=0.8)
    ax[1, 1].axvline(K_TR, color="red", ls=":", lw=1.2)
    ax[1, 1].set_xlabel("test-time loop iterations K")
    ax[1, 1].set_ylabel(f"mean accuracy on EASY difficulties d <= {K_TR + 1}")
    ax[1, 1].set_ylim(0.38, 1.04)
    ax[1, 1].set_title("(d) the tax: what extra loops do to instances already solved at K_train")
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].grid(alpha=0.25)

    fig.suptitle("Test-time compute on a 0.06M-param weight-tied loop: TRAIN at K=3, TEST at K up to 8\n"
                 "prefix-parity with attention window 1, so difficulty d needs exactly d-1 sequential steps "
                 "(3 seeds, 1024 eval seqs, CPU)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(HERE / "chart.png", dpi=130)


if __name__ == "__main__":
    if "--chart-only" in sys.argv:
        make_chart(load_config(), json.load(open(HERE / "results.json"))["metrics"])
    else:
        main()
