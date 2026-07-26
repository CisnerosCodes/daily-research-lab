"""Quantized Coconut: a VQ bottleneck on the continuous thought.

How discrete can a "thought" be before accuracy drops?

Sibling experiment 2026-07-25_coconut-toy-graph established, on this exact task and
model, that Coconut-style continuous thoughts reach 0.873 held-out accuracy (vs 0.937
discrete CoT, 0.687 pause tokens, 0.600 no-CoT) and that the staged curriculum is
load-bearing (+0.33).  Its task generator, model skeleton, forward pass and curriculum
are REUSED VERBATIM here; the only change is a vector-quantization bottleneck.

Arms (all `coconut`, all with the same staged curriculum, params, steps, optimiser):

  K=inf : unquantized continuous thoughts               (replication of the sibling arm)
  K=256 / 64 / 16 / 4 : each fed-back thought vector is replaced by its nearest
          codebook entry, VQ-VAE style (arXiv:1711.00937): hard argmin forward,
          straight-through estimator backward, codebook + commitment (beta) loss,
          and dead-code restarts every `vq_restart_every` steps.

A "continuous thought" is the model's own last hidden state fed back as the next input
embedding (Hao et al. 2024, arXiv:2412.06769).  With H=3 thought slots there are 3 such
fed-back vectors per example; ALL 3 are quantized.  The answer is read from the hidden
state at the last thought position (as in Coconut), which is not itself quantized.

Bonus ("readable thoughts"): per-slot codebook usage perplexity and normalized mutual
information between the emitted code and (a) the yes/no label, (b) the hop count, (c) the
node id on the ground-truth BFS path at each hop -- each against a label-shuffled control
that measures the finite-sample MI bias.

Deterministic, CPU-only, single-threaded.  Writes results.json + chart.png.
Usage:  python run.py
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import math
import random
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent


# ----------------------------- plumbing ------------------------------------
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
    for mod in ("numpy", "torch"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:
            pass
    return info


# ----------------------------- vocab ---------------------------------------
def make_vocab(N):
    """node tokens are 0..N-1; specials follow."""
    return {
        "N": N,
        "BOS": N, "SEP": N + 1, "QM": N + 2, "NONE": N + 3,
        "YES": N + 4, "NO": N + 5, "PAUSE": N + 6,
        "size": N + 7,
    }


# ----------------------------- data ----------------------------------------
# (verbatim from experiments/2026-07-25_coconut-toy-graph/run.py -- same seeds,
#  so the train/test pools are bit-identical to the sibling's)
def gen_example(rng, N, E, H, V, K):
    """One random DAG + one balanced reachability query.

    The graph is written as a FIXED-WIDTH adjacency-slot block: K token slots per
    node, in node-id order, holding that node's sorted successors padded with
    NONE.  Node ids are then randomly permuted, so token identity carries no
    topological information and the model must do genuine relational lookup.
    """
    for _ in range(400):
        # sample exactly E forward edges subject to out-degree <= K
        adj = [[] for _ in range(N)]
        cand = [(i, j) for i in range(N) for j in range(i + 1, N)]
        order = rng.permutation(len(cand))
        ne = 0
        for oi in order:
            u, v = cand[oi]
            if len(adj[u]) < K:
                adj[u].append(v)
                ne += 1
                if ne == E:
                    break
        if ne < E:
            continue

        # all-pairs BFS (N is tiny)
        dist = np.full((N, N), -1, dtype=np.int64)
        for s in range(N):
            dist[s, s] = 0
            dq = deque([s])
            while dq:
                u = dq.popleft()
                for w in adj[u]:
                    if dist[s, w] < 0:
                        dist[s, w] = dist[s, u] + 1
                        dq.append(w)

        # positives = reachable within H hops; negatives = unreachable OR further than H
        pos_pool = {h: [] for h in range(1, H + 1)}
        neg_pool = []
        for s in range(N):
            for t in range(N):
                if s == t:
                    continue
                d = dist[s, t]
                if d < 0 or d > H:
                    neg_pool.append((s, t))
                else:
                    pos_pool[d].append((s, t))
        if not neg_pool or any(len(pos_pool[h]) == 0 for h in range(1, H + 1)):
            continue

        # balanced label, hop-stratified within the positives
        if rng.random() < 0.5:
            hop = int(rng.integers(1, H + 1))
            pool, label = pos_pool[hop], 1
        else:
            hop, pool, label = 0, neg_pool, 0
        src, dst = pool[int(rng.integers(len(pool)))]

        # canonical CoT trace: shortest path if reachable, else a greedy dead-end walk
        cot = []
        if label == 1:
            path, cur = [dst], dst
            while cur != src:
                preds = [u for u in range(N)
                         if cur in adj[u] and dist[src, u] == dist[src, cur] - 1]
                cur = min(preds)
                path.append(cur)
            path.reverse()
            cot = path[1:]
        else:
            cur = src
            for _ in range(H):
                succ = sorted(adj[cur])
                if not succ:
                    break
                cur = succ[0]
                cot.append(cur)

        # relabel nodes so token identity carries no topological-order information
        perm = rng.permutation(N)                    # perm[old] = new
        relab = lambda x: int(perm[x])
        succ_r = [[] for _ in range(N)]
        for u in range(N):
            succ_r[relab(u)] = sorted(relab(v) for v in adj[u])

        prefix = [V["BOS"]]
        for u in range(N):
            slots = succ_r[u] + [V["NONE"]] * K
            prefix += slots[:K]
        src_r, dst_r = relab(src), relab(dst)
        prefix += [V["SEP"], src_r, dst_r, V["QM"]]

        cot_r = ([relab(c) for c in cot] + [V["NONE"]] * H)[:H]
        dist_src = np.full(N, -1, dtype=np.int64)
        for v in range(N):
            dist_src[relab(v)] = dist[src, v]
        return {
            "prefix": np.array(prefix, dtype=np.int64),
            "cot": np.array(cot_r, dtype=np.int64),
            "ans": V["YES"] if label == 1 else V["NO"],
            "label": label,
            "hop": hop,
            "dist": dist_src,
        }
    raise RuntimeError("could not sample a usable DAG")


def build_pool(seed, n, N, E, H, V, K):
    rng = np.random.default_rng(seed)
    ex = [gen_example(rng, N, E, H, V, K) for _ in range(n)]
    return {
        "prefix": np.stack([e["prefix"] for e in ex]),
        "cot": np.stack([e["cot"] for e in ex]),
        "ans": np.array([e["ans"] for e in ex]),
        "label": np.array([e["label"] for e in ex]),
        "hop": np.array([e["hop"] for e in ex]),
        "dist": np.stack([e["dist"] for e in ex]),
    }


# ----------------------------- model ---------------------------------------
# (verbatim from the sibling)
class Block(nn.Module):
    def __init__(self, d, h, dff):
        super().__init__()
        self.h, self.dh = h, d // h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ff = nn.Sequential(nn.Linear(d, dff), nn.GELU(), nn.Linear(dff, d))

    def forward(self, x, cache=None):
        """x: (B,T,D) new positions. cache: dict holding past k/v."""
        B, T, D = x.shape
        y = self.ln1(x)
        q, k, v = self.qkv(y).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        if cache is not None and cache.get("k") is not None:
            k = torch.cat([cache["k"], k], dim=2)
            v = torch.cat([cache["v"], v], dim=2)
        if cache is not None:
            cache["k"], cache["v"] = k, v
        S = k.shape[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)      # (B,H,T,S)
        qi = torch.arange(S - T, S)[:, None]
        ki = torch.arange(S)[None, :]
        att = att.masked_fill((ki > qi)[None, None], float("-inf")).softmax(-1)
        x = x + self.proj((att @ v).transpose(1, 2).reshape(B, T, D))
        x = x + self.ff(self.ln2(x))
        return x


class TinyLM(nn.Module):
    def __init__(self, vocab, d, h, dff, n_layers, max_pos):
        super().__init__()
        self.d = d
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_pos, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.blocks = nn.ModuleList([Block(d, h, dff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def new_cache(self):
        return [{"k": None, "v": None} for _ in self.blocks]

    def step(self, x_emb, start_pos, cache):
        """x_emb: (B,T,D) token/thought embeddings WITHOUT the positional term."""
        T = x_emb.shape[1]
        x = x_emb + self.pos(torch.arange(start_pos, start_pos + T))[None]
        for b, c in zip(self.blocks, cache):
            x = b(x, c)
        return self.ln_f(x)

    def logits(self, hid):
        return self.head(hid)


# ----------------------------- the VQ bottleneck ---------------------------
class VQ(nn.Module):
    """VQ-VAE bottleneck (arXiv:1711.00937) on a single vector per call.

    forward: hard nearest-neighbour lookup, straight-through gradient to the input,
    codebook loss ||sg[h]-e||^2 + beta*||h-sg[e]||^2 returned separately so the caller
    can weight it.  `usage` counts code hits since the last restart.
    """

    def __init__(self, K, d, beta, init_std):
        super().__init__()
        self.K, self.beta = K, beta
        self.codebook = nn.Parameter(torch.randn(K, d) * init_std)
        self.register_buffer("usage", torch.zeros(K))
        self.last_h = None

    def forward(self, h, track=True):
        """h: (B,D) -> (z (B,D) straight-through, loss scalar, idx (B,))"""
        with torch.no_grad():
            d2 = ((h * h).sum(1, keepdim=True)
                  - 2.0 * h @ self.codebook.t()
                  + (self.codebook * self.codebook).sum(1)[None])
            idx = d2.argmin(1)
        zq = self.codebook[idx]
        loss = F.mse_loss(zq, h.detach()) + self.beta * F.mse_loss(h, zq.detach())
        z = h + (zq - h).detach()
        if track:
            with torch.no_grad():
                self.usage.index_add_(0, idx, torch.ones(idx.shape[0]))
                self.last_h = h.detach()
        return z, loss, idx

    @torch.no_grad()
    def restart_dead(self, gen):
        """Reset codes unused since the last call to random thought vectors."""
        dead = (self.usage == 0).nonzero().flatten()
        n_dead = int(dead.numel())
        if n_dead and self.last_h is not None and self.last_h.shape[0] > 0:
            sel = torch.randint(0, self.last_h.shape[0], (n_dead,), generator=gen)
            noise = torch.randn(n_dead, self.codebook.shape[1], generator=gen) * 0.01
            self.codebook.data[dead] = self.last_h[sel] + noise
        self.usage.zero_()
        return n_dead


# ----------------------------- forward -------------------------------------
def forward_coconut(model, vq, prefix, cot, V, H, k_thoughts=None, collect_codes=False):
    """Coconut forward with an optional VQ bottleneck on every fed-back thought.

    Returns (aux_terms, ans_logits, vq_loss, codes).
      aux_terms: [(logits (B,vocab), target (B,))] CE on the CoT tokens still in discrete
                 form during the curriculum stages (identical to the sibling).
      codes:     (B,k) long tensor of emitted code indices, or None.
    """
    B, P = prefix.shape
    cache = model.new_cache()
    hpre = model.step(model.emb(prefix), 0, cache)          # (B,P,D)
    h_last = hpre[:, -1]                                    # hidden at the QM position
    terms = []
    vq_loss = torch.zeros(())
    codes = [] if collect_codes else None

    k = H if k_thoughts is None else k_thoughts
    if k == 0:
        terms.append((model.logits(h_last), cot[:, 0]))
    cur = h_last
    for j in range(k):
        thought = cur
        if vq is not None:
            thought, l, idx = vq(thought, track=model.training)
            vq_loss = vq_loss + l
            if collect_codes:
                codes.append(idx)
        cur = model.step(thought[:, None, :], P + j, cache)[:, 0]
    if k > 0 and vq is not None:
        vq_loss = vq_loss / k
    if k < H:
        rest = cot[:, k:]
        if k > 0:
            terms.append((model.logits(cur), rest[:, 0]))
        hs = model.step(model.emb(rest), P + k, cache)
        for j in range(H - k - 1):
            terms.append((model.logits(hs[:, j]), rest[:, j + 1]))
        cur = hs[:, -1]
    return terms, model.logits(cur), vq_loss, (torch.stack(codes, 1) if codes else None)


# ----------------------------- eval ----------------------------------------
@torch.no_grad()
def evaluate(model, vq, pool, V, H, bs=500, collect_codes=False):
    """Binary accuracy (argmax restricted to YES/NO), overall and by hop class."""
    model.eval()
    n = pool["prefix"].shape[0]
    correct = np.zeros(n, dtype=bool)
    all_codes = [] if collect_codes else None
    for s in range(0, n, bs):
        prefix = torch.from_numpy(pool["prefix"][s:s + bs])
        cot = torch.from_numpy(pool["cot"][s:s + bs])
        _, ans_logits, _, codes = forward_coconut(model, vq, prefix, cot, V, H,
                                                  collect_codes=collect_codes)
        if collect_codes and codes is not None:
            all_codes.append(codes.numpy())
        pred_yes = ans_logits[:, V["YES"]] > ans_logits[:, V["NO"]]
        want = torch.from_numpy(pool["label"][s:s + bs]).bool()
        correct[s:s + bs] = (pred_yes == want).numpy()
    model.train()
    hop = pool["hop"]
    out = {
        "acc": float(correct.mean()),
        "acc_by_hop": {str(h): float(correct[hop == h].mean()) for h in sorted(set(hop.tolist()))},
        "acc_yes": float(correct[pool["label"] == 1].mean()),
        "acc_no": float(correct[pool["label"] == 0].mean()),
    }
    if collect_codes:
        out["_codes"] = np.concatenate(all_codes, 0) if all_codes else None
    return out


# ----------------------------- code statistics ------------------------------
def entropy_bits(counts):
    c = np.asarray(counts, dtype=np.float64)
    c = c[c > 0]
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def mutual_info_bits(a, b):
    """MI(a;b) in bits for two integer arrays (plug-in estimator)."""
    au, ai = np.unique(a, return_inverse=True)
    bu, bi = np.unique(b, return_inverse=True)
    joint = np.zeros((len(au), len(bu)), dtype=np.float64)
    np.add.at(joint, (ai, bi), 1.0)
    return float(entropy_bits(joint.sum(1)) + entropy_bits(joint.sum(0)) - entropy_bits(joint.ravel()))


def code_stats(codes, pool, H, K, shuffle_seed):
    """Usage perplexity + normalized MI between code and label / hop / path node.

    Every MI is reported alongside a label-shuffled control computed the same way,
    which measures the finite-sample positive bias (large with K=256 codes).
    """
    n, S = codes.shape
    rng = np.random.default_rng(shuffle_seed)
    perm = rng.permutation(n)
    label, hop, cot = pool["label"], pool["hop"], pool["cot"]
    out = {"n_slots": S, "per_slot": [], "used_total": int(len(np.unique(codes)))}
    for s in range(S):
        c = codes[:, s]
        cnt = np.bincount(c, minlength=K)
        ent = entropy_bits(cnt)
        slot = {
            "perplexity": round(float(2.0 ** ent), 3),
            "entropy_bits": round(ent, 4),
            "n_used": int((cnt > 0).sum()),
            "top1_share": round(float(cnt.max() / cnt.sum()), 4),
        }
        for name, tgt in (("label", label), ("hop", hop)):
            h_t = entropy_bits(np.bincount(tgt))
            mi = mutual_info_bits(c, tgt)
            mi0 = mutual_info_bits(c, tgt[perm])
            slot[f"nmi_{name}"] = round(mi / h_t, 4)
            slot[f"nmi_{name}_shuffled"] = round(mi0 / h_t, 4)
            slot[f"nmi_{name}_corrected"] = round((mi - mi0) / h_t, 4)
        # node on the ground-truth BFS path at each hop j (NONE-padded for negatives)
        best, best_j = -1.0, None
        for j in range(H):
            h_t = entropy_bits(np.bincount(cot[:, j]))
            mi = (mutual_info_bits(c, cot[:, j]) - mutual_info_bits(c, cot[perm, j])) / h_t
            slot[f"nmi_pathnode_h{j + 1}_corrected"] = round(mi, 4)
            if mi > best:
                best, best_j = mi, j + 1
        slot["nmi_pathnode_best"] = round(best, 4)
        slot["nmi_pathnode_best_hop"] = best_j
        out["per_slot"].append(slot)
    out["perplexity_mean"] = round(float(np.mean([s["perplexity"] for s in out["per_slot"]])), 3)
    out["nmi_label_corrected_max"] = round(
        float(max(s["nmi_label_corrected"] for s in out["per_slot"])), 4)
    out["nmi_hop_corrected_max"] = round(
        float(max(s["nmi_hop_corrected"] for s in out["per_slot"])), 4)
    out["nmi_pathnode_corrected_max"] = round(
        float(max(s["nmi_pathnode_best"] for s in out["per_slot"])), 4)
    return out


# ----------------------------- training ------------------------------------
def curriculum_k(step, P, H):
    """Number of leading CoT tokens replaced by (quantized) continuous thoughts."""
    stage_len = max(1, int(P["steps"] * P["curriculum_frac"] / H))   # stages k=0,1,..,H-1
    return int(min(H, step // stage_len))


def train_one(K, seed, P, V, train_pool, test_pool, log):
    set_seeds(seed)
    H = P["n_thoughts"]
    rng = np.random.default_rng(10_000 + seed)
    gen = torch.Generator().manual_seed(20_000 + seed)
    maxpos = train_pool["prefix"].shape[1] + H + 2
    model = TinyLM(V["size"], P["d_model"], P["n_heads"], P["d_ff"], P["n_layers"], maxpos)
    vq = None if K is None else VQ(K, P["d_model"], P["vq_beta"], P["vq_init_std"])
    n_params = sum(p.numel() for p in model.parameters())
    n_vq_params = 0 if vq is None else sum(p.numel() for p in vq.parameters())
    groups = [{"params": list(model.parameters()), "weight_decay": P["weight_decay"]}]
    if vq is not None:
        groups.append({"params": list(vq.parameters()), "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, lr=P["lr"])
    npool = train_pool["prefix"].shape[0]
    t0, capped, step, dead_total = time.time(), False, 0, 0
    loss = torch.tensor(0.0)
    vq_l = torch.tensor(0.0)
    for step in range(P["steps"]):
        lr = P["lr"] * min(1.0, (step + 1) / P["warmup"])
        for g in opt.param_groups:
            g["lr"] = lr
        idx = rng.integers(0, npool, size=P["batch_size"])
        prefix = torch.from_numpy(train_pool["prefix"][idx])
        cot = torch.from_numpy(train_pool["cot"][idx])
        ans = torch.from_numpy(train_pool["ans"][idx])

        k = curriculum_k(step, P, H)
        terms, ans_logits, vq_l, _ = forward_coconut(model, vq, prefix, cot, V, H, k_thoughts=k)
        loss = F.cross_entropy(ans_logits, ans)
        for lg, tg in terms:
            loss = loss + F.cross_entropy(lg, tg)
        loss = loss + P["vq_weight"] * vq_l
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(model.parameters())
                                 + ([] if vq is None else list(vq.parameters())), P["grad_clip"])
        opt.step()
        if vq is not None and P["vq_restart_every"] and (step + 1) % P["vq_restart_every"] == 0:
            dead_total += vq.restart_dead(gen)
        if time.time() - t0 > P["time_cap_s_per_run"]:
            capped = True
            break
    train_s = time.time() - t0
    ev = evaluate(model, vq, test_pool, V, H, collect_codes=vq is not None)
    codes = ev.pop("_codes", None)
    stats = None if codes is None else code_stats(codes, test_pool, H, K, P["mi_shuffle_seed"])
    name = "cont(K=inf)" if K is None else f"K={K}"
    log(f"  {name:11s} seed{seed}: params={n_params}(+{n_vq_params} vq) steps={step+1}"
        f"{' CAPPED' if capped else ''} ({train_s:.0f}s) acc={ev['acc']:.3f} "
        f"by_hop={ {k2: round(v, 2) for k2, v in ev['acc_by_hop'].items()} }"
        + ("" if stats is None else
           f" ppl={[s['perplexity'] for s in stats['per_slot']]} "
           f"nmi(hop)={[s['nmi_hop_corrected'] for s in stats['per_slot']]}"))
    return {"K": K, "seed": seed, "n_params": n_params, "n_vq_params": n_vq_params,
            "steps_run": step + 1, "train_seconds": round(train_s, 1), "time_capped": capped,
            "final_loss": round(float(loss.detach()), 4),
            "final_vq_loss": round(float(vq_l.detach()), 4) if vq is not None else None,
            "dead_code_restarts": dead_total, **ev, "code_stats": stats}


# ----------------------------- main ----------------------------------------
def main():
    cfg = load_config()
    P = cfg["params"]
    set_seeds(int(cfg.get("seed", 0)))
    t0 = time.time()
    log = lambda s: print(s, flush=True)
    N, E, H, KD = P["n_nodes"], P["n_edges"], P["n_thoughts"], P["max_out_degree"]
    V = make_vocab(N)

    log("building data pools ...")
    td = time.time()
    train_pool = build_pool(1234, P["n_train"], N, E, H, V, KD)
    test_pool = build_pool(777_001, P["n_test"], N, E, H, V, KD)
    log(f"  pools built in {time.time()-td:.1f}s; prefix len={train_pool['prefix'].shape[1]}, "
        f"train label balance={train_pool['label'].mean():.3f}")

    runs = []
    for K in P["codebook_sizes"]:
        for seed in P["seeds"]:
            runs.append(train_one(K, seed, P, V, train_pool, test_pool, log))

    # ----------------------------- aggregate ---------------------------------
    def key(K):
        return "inf" if K is None else str(K)

    agg = {}
    for K in P["codebook_sizes"]:
        rs = [r for r in runs if r["K"] == K]
        hops = sorted({h for r in rs for h in r["acc_by_hop"]}, key=int)
        a = {
            "K": K,
            "bits_per_thought": None if K is None else round(math.log2(K), 3),
            "acc_mean": round(float(np.mean([r["acc"] for r in rs])), 4),
            "acc_std": round(float(np.std([r["acc"] for r in rs])), 4),
            "acc_by_hop_mean": {h: round(float(np.mean([r["acc_by_hop"][h] for r in rs])), 4)
                                for h in hops},
            "acc_yes_mean": round(float(np.mean([r["acc_yes"] for r in rs])), 4),
            "acc_no_mean": round(float(np.mean([r["acc_no"] for r in rs])), 4),
            "steps_run_min": int(min(r["steps_run"] for r in rs)),
            "any_time_capped": bool(any(r["time_capped"] for r in rs)),
            "train_seconds_mean": round(float(np.mean([r["train_seconds"] for r in rs])), 1),
        }
        if K is not None:
            a["perplexity_mean"] = round(
                float(np.mean([r["code_stats"]["perplexity_mean"] for r in rs])), 3)
            a["perplexity_by_slot_mean"] = [
                round(float(np.mean([r["code_stats"]["per_slot"][s]["perplexity"] for r in rs])), 3)
                for s in range(H)]
            a["codes_used_mean"] = round(
                float(np.mean([r["code_stats"]["used_total"] for r in rs])), 2)
            for nm in ("nmi_label_corrected_max", "nmi_hop_corrected_max",
                       "nmi_pathnode_corrected_max"):
                a[nm] = round(float(np.mean([r["code_stats"][nm] for r in rs])), 4)
            a["dead_code_restarts_mean"] = round(
                float(np.mean([r["dead_code_restarts"] for r in rs])), 1)
        agg[key(K)] = a

    base = agg["inf"]["acc_mean"]
    ks_q = [K for K in P["codebook_sizes"] if K is not None]
    deltas = {key(K): round(agg[key(K)]["acc_mean"] - base, 4) for K in ks_q}
    # "matches the continuous baseline" = within one baseline seed-std of it
    tol = max(agg["inf"]["acc_std"], 0.01)
    matching = [K for K in sorted(ks_q) if agg[key(K)]["acc_mean"] >= base - tol]
    k_star = min(matching) if matching else None
    ref = P["sibling_reference"]

    metrics = {
        "per_run": runs,
        "aggregate": agg,
        "acc_mean_by_K": {key(K): agg[key(K)]["acc_mean"] for K in P["codebook_sizes"]},
        "delta_vs_continuous": deltas,
        "K_star_matches_continuous": k_star,
        "match_tolerance": round(float(tol), 4),
        "continuous_baseline_acc": base,
        "worst_quantized_acc": min(agg[key(K)]["acc_mean"] for K in ks_q),
        "max_drop_from_quantization": round(
            base - min(agg[key(K)]["acc_mean"] for K in ks_q), 4),
        "seed_std_max": round(float(max(agg[k]["acc_std"] for k in agg)), 4),
        "codebook_perplexity_by_K": {key(K): agg[key(K)]["perplexity_mean"] for K in ks_q},
        "nmi_hop_corrected_by_K": {key(K): agg[key(K)]["nmi_hop_corrected_max"] for K in ks_q},
        "nmi_label_corrected_by_K": {key(K): agg[key(K)]["nmi_label_corrected_max"] for K in ks_q},
        "nmi_pathnode_corrected_by_K": {key(K): agg[key(K)]["nmi_pathnode_corrected_max"]
                                        for K in ks_q},
        "sibling_reference": ref,
        "n_params": runs[0]["n_params"],
        "headline": (
            "held-out acc vs codebook size: "
            + ", ".join(f"K={key(K)}:{agg[key(K)]['acc_mean']:.3f}" for K in P["codebook_sizes"])
            + f"; smallest K matching the continuous baseline ({base:.3f} +- {tol:.3f}) = {k_star}"
            + f"; max drop from quantization = {base - min(agg[key(K)]['acc_mean'] for K in ks_q):+.3f}"
            + f"; code usage perplexity = "
            + ", ".join(f"K={K}:{agg[key(K)]['perplexity_mean']:.1f}" for K in ks_q)),
    }

    # ----------------------------- chart -------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(18.5, 4.4), width_ratios=[1.35, 1.25, 1.1, 1.25])
    ax1, ax2, ax3, ax4 = axes
    xs = np.array([math.log2(K) for K in ks_q])
    ys = np.array([agg[key(K)]["acc_mean"] for K in ks_q])
    es = np.array([agg[key(K)]["acc_std"] for K in ks_q])

    ax1.errorbar(xs, ys, yerr=es, fmt="o-", lw=2, ms=6, capsize=3, color="#1a7f64",
                 label="VQ thought (this run)")
    ax1.axhline(base, color="#1a7f64", ls="--", lw=1.6,
                label=f"continuous K=inf (this run, {base:.3f})")
    ax1.fill_between([xs.min() - 0.5, xs.max() + 0.5], base - tol, base + tol,
                     color="#1a7f64", alpha=0.12, lw=0)
    for nm, col, sty in (("cot", "#3d5a80", ":"), ("pause", "#c9a13c", ":"),
                         ("nocot", "#8a817c", ":")):
        ax1.axhline(ref[nm], color=col, ls=sty, lw=1.4, label=f"{nm} (sibling, {ref[nm]:.3f})")
    ax1.axhline(0.5, color="0.6", ls="-", lw=0.9)
    ax1.text(xs.max() + 0.35, 0.507, "chance", fontsize=7, color="0.5", ha="right")
    ax1.set_xticks(xs); ax1.set_xticklabels([str(K) for K in ks_q])
    ax1.set_xlim(xs.min() - 0.5, xs.max() + 0.5)
    ax1.set_xlabel("codebook size K  (log scale)")
    ax1.set_ylabel("held-out accuracy")
    ax1.set_ylim(0.45, 1.0)
    ax1.legend(frameon=False, fontsize=6.8, loc="lower right")
    ax1.set_title("Headline: accuracy vs discreteness of the thought", fontsize=10)

    hop_names = {"1": "1-hop", "2": "2-hop", "3": "3-hop", "0": "no path"}
    allK = list(P["codebook_sizes"])
    xh = np.arange(4); w = 0.16
    cmap = plt.get_cmap("viridis")
    for i, K in enumerate(allK):
        cols = "#333333" if K is None else cmap(0.15 + 0.7 * i / max(1, len(allK) - 1))
        vals = [agg[key(K)]["acc_by_hop_mean"].get(h, np.nan) for h in ["1", "2", "3", "0"]]
        ax2.bar(xh + (i - (len(allK) - 1) / 2) * w, vals, width=w, color=cols,
                label=("K=inf" if K is None else f"K={K}"))
    ax2.set_xticks(xh); ax2.set_xticklabels([hop_names[h] for h in ["1", "2", "3", "0"]], fontsize=8)
    ax2.axhline(0.5, color="0.5", ls="--", lw=1)
    ax2.set_ylim(0, 1.08); ax2.set_ylabel("accuracy")
    ax2.legend(frameon=False, fontsize=7, ncol=2)
    ax2.set_title("Stratified by shortest-path length", fontsize=10)

    ppl = [agg[key(K)]["perplexity_mean"] for K in ks_q]
    ax3.plot(xs, np.log2(ppl), "o-", lw=2, ms=6, color="#b23a48", label="used (perplexity)")
    ax3.plot(xs, xs, "--", lw=1.4, color="0.5", label="all K codes used")
    for i, K in enumerate(ks_q):
        ax3.annotate(f"{ppl[i]:.1f}", (xs[i], np.log2(ppl[i])), textcoords="offset points",
                     xytext=(6, -10), fontsize=7)
    ax3.set_xticks(xs); ax3.set_xticklabels([str(K) for K in ks_q])
    ax3.set_yticks(xs); ax3.set_yticklabels([str(K) for K in ks_q])
    ax3.set_xlabel("codebook size K"); ax3.set_ylabel("effective codes used (perplexity)")
    ax3.legend(frameon=False, fontsize=7)
    ax3.set_title("Codebook collapse?", fontsize=10)

    for nm, col, lab in (("nmi_hop_corrected_by_K", "#3d5a80", "code vs hop count"),
                         ("nmi_label_corrected_by_K", "#1a7f64", "code vs YES/NO"),
                         ("nmi_pathnode_corrected_by_K", "#c9a13c", "code vs BFS path node")):
        ax4.plot(xs, [metrics[nm][key(K)] for K in ks_q], "o-", lw=2, ms=5, color=col, label=lab)
    ax4.axhline(0.0, color="0.5", ls="--", lw=1)
    ax4.set_xticks(xs); ax4.set_xticklabels([str(K) for K in ks_q])
    ax4.set_xlabel("codebook size K")
    ax4.set_ylabel("normalized MI (bits/bit), shuffle-corrected")
    ax4.legend(frameon=False, fontsize=7)
    ax4.set_title("'Readable thoughts': what does the code say?", fontsize=10)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Quantized Coconut: VQ bottleneck on the continuous thought "
                 f"(DAG reachability, N={N} nodes, H={H} thoughts, "
                 f"{metrics['n_params']/1e3:.0f}k params, {P['steps']} steps, "
                 f"{len(P['seeds'])} seeds)", fontsize=11, y=1.03)
    fig.tight_layout()
    fig.savefig(HERE / "chart.png", dpi=155, bbox_inches="tight")

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
