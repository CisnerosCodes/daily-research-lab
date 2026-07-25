"""NoPE vs RoPE vs ALiBi vs APE length generalization at ~0.1M params.

Tiny CPU replication of Kazemnejad et al. 2023 (arXiv:2305.19466, 107M params):
decoder-only ranking NoPE >= ALiBi > RoPE >= APE for length extrapolation.

Task: autoregressive copy. Sequence = BOS x1..xL SEP x1..xL EOS.
Train L in [4,16]; evaluate greedy-decoded exact match at L up to 32.

Deterministic, CPU-only, writes results.json + chart.png.
Usage:  python run.py
"""
import json, math, os, random, subprocess, sys, time
from pathlib import Path

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
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(os.cpu_count() or 4)


def rope_rotate(x, pos, theta=10000.0):
    """x: (B, H, T, Dh); pos: (T,) positions. Standard RoPE on q/k."""
    B, H, T, Dh = x.shape
    half = Dh // 2
    freqs = theta ** (-torch.arange(0, half, dtype=torch.float32) / half)  # (half,)
    ang = pos[:, None].float() * freqs[None, :]                            # (T, half)
    cos, sin = ang.cos(), ang.sin()                                        # (T, half)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def alibi_slopes(n_heads):
    # standard geometric slopes: 2^(-8i/n) for i=1..n
    return torch.tensor([2 ** (-8.0 * (i + 1) / n_heads) for i in range(n_heads)])


class Block(nn.Module):
    def __init__(self, d, h, dff, pe):
        super().__init__()
        self.h, self.dh, self.pe = h, d // h, pe
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ff = nn.Sequential(nn.Linear(d, dff), nn.GELU(), nn.Linear(dff, d))
        if pe == "alibi":
            self.register_buffer("slopes", alibi_slopes(h))

    def forward(self, x):
        B, T, D = x.shape
        y = self.ln1(x)
        q, k, v = self.qkv(y).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        pos = torch.arange(T)
        if self.pe == "rope":
            q, k = rope_rotate(q, pos), rope_rotate(k, pos)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)  # (B,H,T,T)
        if self.pe == "alibi":
            rel = pos[None, :] - pos[:, None]                 # j - i
            bias = self.slopes[:, None, None] * rel[None].clamp(max=0)
            att = att + bias
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        att = att.masked_fill(mask, float("-inf")).softmax(-1)
        x = x + self.proj((att @ v).transpose(1, 2).reshape(B, T, D))
        x = x + self.ff(self.ln2(x))
        return x


class TinyLM(nn.Module):
    def __init__(self, vocab, d, h, dff, n_layers, pe, max_pos):
        super().__init__()
        self.pe = pe
        self.emb = nn.Embedding(vocab, d)
        if pe == "ape":
            self.pos_emb = nn.Embedding(max_pos, d)
            nn.init.normal_(self.pos_emb.weight, std=0.02)
        self.blocks = nn.ModuleList([Block(d, h, dff, pe) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx):
        x = self.emb(idx)
        if self.pe == "ape":
            x = x + self.pos_emb(torch.arange(idx.shape[1]))[None]
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))


# ----------------------------- data ----------------------------------------
def make_batch(rng, n, lmin, lmax, V, BOS, SEP, EOS, PAD):
    """Returns padded (input, target) with target=-100 outside the answer span."""
    import numpy as np
    lens = rng.integers(lmin, lmax + 1, size=n)
    T = int(2 * lens.max() + 3)
    seq = np.full((n, T), PAD, dtype=np.int64)
    tgt = np.full((n, T), -100, dtype=np.int64)
    for i, L in enumerate(lens):
        x = rng.integers(0, V, size=L)
        row = np.concatenate([[BOS], x, [SEP], x, [EOS]])
        seq[i, : len(row)] = row
        # predict answer tokens x1..xL and EOS from positions SEP..(SEP+L)
        a = L + 1  # index of SEP
        tgt[i, a : a + L + 1] = row[a + 1 : a + L + 2]
    return torch.from_numpy(seq), torch.from_numpy(tgt), lens


@torch.no_grad()
def exact_match(model, rng, n, L, V, BOS, SEP, EOS):
    """Greedy autoregressive decode of the answer; exact match incl. EOS."""
    import numpy as np
    x = rng.integers(0, V, size=(n, L))
    prompt = np.concatenate(
        [np.full((n, 1), BOS), x, np.full((n, 1), SEP)], axis=1)
    cur = torch.from_numpy(prompt)
    outs = []
    for _ in range(L + 1):
        logits = model(cur)[:, -1]
        nxt = logits.argmax(-1, keepdim=True)
        outs.append(nxt)
        cur = torch.cat([cur, nxt], dim=1)
    out = torch.cat(outs, dim=1).numpy()          # (n, L+1)
    want = np.concatenate([x, np.full((n, 1), EOS)], axis=1)
    em = float((out == want).all(axis=1).mean())
    tok = float((out == want).mean())
    return em, tok


# ----------------------------- one run -------------------------------------
def train_one(pe, seed, P, log):
    import numpy as np
    set_seeds(seed)
    rng = np.random.default_rng(seed * 1000 + 7)
    V, BOS, SEP, EOS, PAD = P["vocab_data"], 16, 17, 18, 19
    model = TinyLM(20, P["d_model"], P["n_heads"], P["d_ff"],
                   P["n_layers"], pe, P["max_positions"])
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=P["lr"],
                            weight_decay=P["weight_decay"])
    t0, capped = time.time(), False
    for step in range(P["steps"]):
        lr = P["lr"] * min(1.0, (step + 1) / P["warmup"])
        for g in opt.param_groups:
            g["lr"] = lr
        seq, tgt, _ = make_batch(rng, P["batch_size"], P["train_len_min"],
                                 P["train_len_max"], V, BOS, SEP, EOS, PAD)
        logits = model(seq[:, :-1])
        # logits[:, t] predicts seq[:, t+1]; tgt[:, t] already holds seq[:, t+1]
        loss = F.cross_entropy(logits.reshape(-1, 20), tgt[:, :-1].reshape(-1),
                               ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
        if time.time() - t0 > P["time_cap_s_per_run"]:
            capped = True
            break
    train_s = time.time() - t0
    model.eval()
    acc, tokacc = {}, {}
    for L in P["test_lengths"]:
        erng = np.random.default_rng(12345 + L)  # same eval data for every run
        acc[L], tokacc[L] = exact_match(model, erng, P["eval_n_per_len"], L, V, BOS, SEP, EOS)
    log(f"  {pe} seed{seed}: params={n_params} steps={step+1} "
        f"({train_s:.0f}s{' CAPPED' if capped else ''}) "
        f"em16={acc[16]:.2f} em24={acc[24]:.2f} tok24={tokacc[24]:.2f} tok32={tokacc[32]:.2f}")
    return {"pe": pe, "seed": seed, "n_params": n_params, "steps_run": step + 1,
            "train_seconds": round(train_s, 1), "time_capped": capped,
            "final_loss": round(float(loss.detach()), 4),
            "exact_match_by_len": {str(k): v for k, v in acc.items()},
            "token_acc_by_len": {str(k): v for k, v in tokacc.items()}}


# ----------------------------- main ----------------------------------------
def main():
    cfg = load_config()
    P = cfg["params"]
    set_seeds(int(cfg.get("seed", 0)))
    t0 = time.time()
    log = lambda s: print(s, flush=True)

    runs = []
    for pe in P["pes"]:
        for seed in P["seeds"]:
            runs.append(train_one(pe, seed, P, log))

    # aggregate: mean over seeds per (pe, length)
    import numpy as np
    ood_lens = [L for L in P["test_lengths"] if L > P["train_len_max"]]
    id_lens = [L for L in P["test_lengths"] if L <= P["train_len_max"]]
    agg = {}
    for pe in P["pes"]:
        rs = [r for r in runs if r["pe"] == pe]
        em = {L: float(np.mean([r["exact_match_by_len"][str(L)] for r in rs]))
              for L in P["test_lengths"]}
        tok = {L: float(np.mean([r["token_acc_by_len"][str(L)] for r in rs]))
               for L in P["test_lengths"]}
        agg[pe] = {
            "em_curve": {str(k): round(v, 4) for k, v in em.items()},
            "tok_curve": {str(k): round(v, 4) for k, v in tok.items()},
            "em_id_mean": round(float(np.mean([em[L] for L in id_lens])), 4),
            "em_ood_mean": round(float(np.mean([em[L] for L in ood_lens])), 4),
            "tok_id_mean": round(float(np.mean([tok[L] for L in id_lens])), 4),
            "tok_ood_mean": round(float(np.mean([tok[L] for L in ood_lens])), 4),
        }
    ranking = sorted(P["pes"], key=lambda pe: -agg[pe]["tok_ood_mean"])

    metrics = {
        "per_run": runs,
        "aggregate": agg,
        "ood_ranking_by_token_acc": ranking,
        "tok_ood_mean_by_pe": {pe: agg[pe]["tok_ood_mean"] for pe in P["pes"]},
        "em_ood_mean_by_pe": {pe: agg[pe]["em_ood_mean"] for pe in P["pes"]},
        "em_id_mean_by_pe": {pe: agg[pe]["em_id_mean"] for pe in P["pes"]},
        "headline": ("OOD token-acc (len 18-32): " +
                     ", ".join(f"{pe}={agg[pe]['tok_ood_mean']:.3f}" for pe in ranking)),
    }

    # ----------------------------- chart ------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"nope": "#1a7f64", "alibi": "#3d5a80", "rope": "#c95d3c", "ape": "#8a817c"}
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.2), width_ratios=[2, 2, 1])
    for ax, key, name in ((ax1, "em_curve", "greedy exact match"),
                          (ax2, "tok_curve", "greedy token accuracy")):
        for pe in P["pes"]:
            xs = P["test_lengths"]
            ys = [agg[pe][key][str(L)] for L in xs]
            ax.plot(xs, ys, "o-", color=colors[pe], label=pe.upper(), lw=2, ms=4)
        ax.axvspan(P["train_len_min"], P["train_len_max"], color="0.92", zorder=0)
        ax.text(P["train_len_min"] + 0.5, 1.04, "train lengths", fontsize=8, color="0.4")
        ax.set_xlabel("copy length L"); ax.set_ylabel(name)
        ax.set_ylim(-0.05, 1.12); ax.legend(frameon=False, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    ax1.set_title("Exact match (whole answer)", fontsize=10)
    ax2.set_title("Token accuracy (graded view)", fontsize=10)
    order = ranking
    ax3.bar([pe.upper() for pe in order],
            [agg[pe]["tok_ood_mean"] for pe in order],
            color=[colors[pe] for pe in order])
    for i, pe in enumerate(order):
        ax3.text(i, agg[pe]["tok_ood_mean"] + 0.02, f"{agg[pe]['tok_ood_mean']:.2f}",
                 ha="center", fontsize=9)
    ax3.set_ylabel("mean OOD token acc (L=18-32)")
    ax3.set_ylim(0, 1.12); ax3.set_title("OOD ranking", fontsize=10)
    ax3.tick_params(axis="x", labelsize=8)
    ax3.spines[["top", "right"]].set_visible(False)
    fig.suptitle("NoPE vs APE vs RoPE vs ALiBi at ~0.1M params - copy task, "
                 "train L in [4,16], mean of 2 seeds", fontsize=11, y=1.02)
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
    print("headline:", metrics["headline"])


if __name__ == "__main__":
    main()
