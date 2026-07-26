"""nano-Coconut on toy DAG reachability.

Five matched training regimes on a ~0.10M-param 2-layer decoder, plus a BFS-frontier probe:

  nocot          : prefix -> YES/NO directly
  pause          : prefix + 3 learned <pause> tokens -> YES/NO       (filler-token control)
  cot            : prefix + 3 supervised hop tokens  -> YES/NO       (discrete chain-of-thought)
  coconut        : prefix + 3 CONTINUOUS thoughts    -> YES/NO       (staged Coconut curriculum)
  coconut_nocur  : same, but no curriculum (3 thoughts from step 0)

Continuous thought = the model's own last hidden state fed back as the next input
embedding (Hao et al. 2024, arXiv:2412.06769), one thought per reasoning hop.

`pause`, `cot`, `coconut*` all use exactly the same number of token positions (P + H)
and the same number of transformer-block applications, so they are compute-matched;
`nocot` is the cheaper reference point (P positions).  All arms share the identical
architecture, parameter count, optimiser, batch size and number of training steps.

Novel angle: a per-node logistic probe on the hidden state at each thought slot,
predicting the BFS frontier F_k = {v : dist(src,v) == k} and the reach set
R_k = {v : 1 <= dist(src,v) <= k}, reported as macro AUC against TWO controls:
(1) a shuffled-label control (labels permuted across examples) and (2) an
untrained randomly-initialised model.

Deterministic, CPU-only, single-threaded. Writes results.json + chart.png.
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
def gen_example(rng, N, E, H, V, K):
    """One random DAG + one balanced reachability query.

    The graph is written as a FIXED-WIDTH adjacency-slot block: K token slots per
    node, in node-id order, holding that node's sorted successors padded with
    NONE.  Node ids are then randomly permuted, so token identity carries no
    topological information and the model must do genuine relational lookup --
    but the *position* of a node's block is a deterministic function of its id,
    which is what makes the task learnable inside the CPU time-box (a free-order
    edge list needs an induction-head-like circuit and did not leave chance in
    ~2k steps; see README).
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


# ----------------------------- forward regimes ------------------------------
def forward_arm(model, arm, prefix, cot, V, H, k_thoughts=None, collect_hidden=False):
    """Returns (aux_terms, ans_logits, hiddens).

    aux_terms: [(logits (B,vocab), target (B,))] extra CoT-token supervision.
    hiddens:   (B, S, D) hidden at slot positions P-1 .. P+H-1 (S = H+1), or None.
               slot 0 is the hidden at the query position (before any thought);
               slot j>0 is the hidden AT thought/CoT/pause token j.
    """
    B, P = prefix.shape
    cache = model.new_cache()
    hpre = model.step(model.emb(prefix), 0, cache)          # (B,P,D)
    h_last = hpre[:, -1]                                    # hidden at the QM position
    terms = []
    hid_list = [h_last] if collect_hidden else None
    pack = lambda: (torch.stack(hid_list, 1) if collect_hidden else None)

    if arm == "nocot":
        return terms, model.logits(h_last), pack()

    if arm == "pause":
        pause = torch.full((B, H), V["PAUSE"], dtype=torch.long)
        hs = model.step(model.emb(pause), P, cache)          # (B,H,D)
        if collect_hidden:
            hid_list += [hs[:, j] for j in range(H)]
        return terms, model.logits(hs[:, -1]), pack()

    if arm == "cot":
        hs = model.step(model.emb(cot), P, cache)            # (B,H,D)
        if collect_hidden:
            hid_list += [hs[:, j] for j in range(H)]
        terms.append((model.logits(h_last), cot[:, 0]))
        for j in range(H - 1):
            terms.append((model.logits(hs[:, j]), cot[:, j + 1]))
        return terms, model.logits(hs[:, -1]), pack()

    # coconut: first k slots are continuous thoughts, the rest discrete CoT tokens
    k = H if k_thoughts is None else k_thoughts
    if k == 0:
        terms.append((model.logits(h_last), cot[:, 0]))
    cur = h_last
    for j in range(k):
        cur = model.step(cur[:, None, :], P + j, cache)[:, 0]   # thought emb = previous hidden
        if collect_hidden:
            hid_list.append(cur)
    if k < H:
        rest = cot[:, k:]
        if k > 0:
            terms.append((model.logits(cur), rest[:, 0]))
        hs = model.step(model.emb(rest), P + k, cache)
        if collect_hidden:
            hid_list += [hs[:, j] for j in range(H - k)]
        for j in range(H - k - 1):
            terms.append((model.logits(hs[:, j]), rest[:, j + 1]))
        cur = hs[:, -1]
    return terms, model.logits(cur), pack()


# ----------------------------- eval ----------------------------------------
@torch.no_grad()
def evaluate(model, arm, pool, V, H, bs=500):
    """Binary accuracy (argmax restricted to YES/NO), overall and by hop class.

    The `cot` arm decodes its H hop tokens greedily (no teacher forcing), so its
    number is an honest end-to-end number, not a teacher-forced one.
    """
    model.eval()
    n = pool["prefix"].shape[0]
    correct = np.zeros(n, dtype=bool)
    trace_ok = np.zeros(n, dtype=bool)
    for s in range(0, n, bs):
        prefix = torch.from_numpy(pool["prefix"][s:s + bs])
        cot = torch.from_numpy(pool["cot"][s:s + bs])
        B, P = prefix.shape
        if arm == "cot":
            cache = model.new_cache()
            cur = model.step(model.emb(prefix), 0, cache)[:, -1]
            gen = []
            for j in range(H):
                nxt = model.logits(cur).argmax(-1)
                gen.append(nxt)
                cur = model.step(model.emb(nxt[:, None]), P + j, cache)[:, 0]
            ans_logits = model.logits(cur)
            trace_ok[s:s + bs] = (torch.stack(gen, 1) == cot).all(1).numpy()
        else:
            _, ans_logits, _ = forward_arm(model, arm, prefix, cot, V, H)
        pred_yes = ans_logits[:, V["YES"]] > ans_logits[:, V["NO"]]
        want = torch.from_numpy(pool["label"][s:s + bs]).bool()
        correct[s:s + bs] = (pred_yes == want).numpy()
    model.train()
    hop = pool["hop"]
    by_hop = {str(h): float(correct[hop == h].mean()) for h in sorted(set(hop.tolist()))}
    return {
        "acc": float(correct.mean()),
        "acc_by_hop": by_hop,
        "acc_yes": float(correct[pool["label"] == 1].mean()),
        "acc_no": float(correct[pool["label"] == 0].mean()),
        "trace_exact": float(trace_ok.mean()) if arm == "cot" else None,
    }


# ----------------------------- training ------------------------------------
def curriculum_k(arm, step, P, H):
    """Number of leading CoT tokens replaced by continuous thoughts at this step."""
    if arm == "coconut_nocur":
        return H
    if arm != "coconut":
        return None
    frac = P["curriculum_frac"]
    stage_len = max(1, int(P["steps"] * frac / H))       # H stages: k=0,1,..,H-1
    return int(min(H, step // stage_len))


def train_one(arm, seed, P, V, train_pool, test_pool, log):
    set_seeds(seed)
    H = P["n_thoughts"]
    rng = np.random.default_rng(10_000 + seed)
    maxpos = train_pool["prefix"].shape[1] + H + 2
    model = TinyLM(V["size"], P["d_model"], P["n_heads"], P["d_ff"], P["n_layers"], maxpos)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=P["lr"], weight_decay=P["weight_decay"])
    npool = train_pool["prefix"].shape[0]
    t0, capped, step, loss = time.time(), False, 0, torch.tensor(0.0)
    for step in range(P["steps"]):
        lr = P["lr"] * min(1.0, (step + 1) / P["warmup"])
        for g in opt.param_groups:
            g["lr"] = lr
        idx = rng.integers(0, npool, size=P["batch_size"])
        prefix = torch.from_numpy(train_pool["prefix"][idx])
        cot = torch.from_numpy(train_pool["cot"][idx])
        ans = torch.from_numpy(train_pool["ans"][idx])

        k = curriculum_k(arm, step, P, H)
        terms, ans_logits, _ = forward_arm(model, arm, prefix, cot, V, H, k_thoughts=k)
        loss = F.cross_entropy(ans_logits, ans)
        for lg, tg in terms:
            loss = loss + F.cross_entropy(lg, tg)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), P["grad_clip"])
        opt.step()
        if time.time() - t0 > P["time_cap_s_per_run"]:
            capped = True
            break
    train_s = time.time() - t0
    ev = evaluate(model, arm, test_pool, V, H)
    log(f"  {arm:14s} seed{seed}: params={n_params} steps={step+1}"
        f"{' CAPPED' if capped else ''} ({train_s:.0f}s) acc={ev['acc']:.3f} "
        f"by_hop={ {k2: round(v, 2) for k2, v in ev['acc_by_hop'].items()} }")
    return model, {"arm": arm, "seed": seed, "n_params": n_params, "steps_run": step + 1,
                   "train_seconds": round(train_s, 1), "time_capped": capped,
                   "final_loss": round(float(loss.detach()), 4), **ev}


# ----------------------------- probe ---------------------------------------
@torch.no_grad()
def collect_hiddens(model, arm, pool, V, H, bs=500):
    model.eval()
    outs = []
    for s in range(0, pool["prefix"].shape[0], bs):
        prefix = torch.from_numpy(pool["prefix"][s:s + bs])
        cot = torch.from_numpy(pool["cot"][s:s + bs])
        _, _, hid = forward_arm(model, arm, prefix, cot, V, H, collect_hidden=True)
        outs.append(hid)
    model.train()
    return torch.cat(outs, 0)            # (n, S, D)


def auc(scores, labels):
    """Rank-based AUC with tie handling; nan if a class is missing."""
    pos, neg = labels == 1, labels == 0
    n1, n0 = int(pos.sum()), int(neg.sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def fit_logreg(Xtr, Ytr, Xte, P):
    """Independent logistic regressions, one per column of Y (no cross-terms)."""
    W = torch.zeros(Xtr.shape[1], Ytr.shape[1], requires_grad=True)
    b = torch.zeros(Ytr.shape[1], requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=P["probe_lr"], weight_decay=P["probe_wd"])
    xt, yt = torch.from_numpy(Xtr).float(), torch.from_numpy(Ytr).float()
    for _ in range(P["probe_steps"]):
        loss = F.binary_cross_entropy_with_logits(xt @ W + b, yt)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return (torch.from_numpy(Xte).float() @ W + b).numpy()


def macro_auc(scores, labels):
    a = [auc(scores[:, v], labels[:, v]) for v in range(labels.shape[1])]
    a = [x for x in a if not math.isnan(x)]
    return round(float(np.mean(a)), 4) if a else float("nan")


def probe_targets(dist, H):
    """(keys, blocks) for frontier F_k and reach set R_k, k = 1..H."""
    keys, blocks = [], []
    for target in ("frontier", "reach"):
        for k in range(1, H + 1):
            y = (dist == k) if target == "frontier" else ((dist >= 1) & (dist <= k))
            keys.append((target, k))
            blocks.append(y.astype(np.float32))
    return keys, blocks


def run_probes(model, arm, ptrain, ptest, V, H, P, shuffle_seed=None):
    """macro AUC[target][slot][k] for frontier F_k and reach set R_k.

    All (target, k, node) columns are fit in ONE multi-output logistic regression
    per slot; the columns do not interact, so this is identical to fitting each
    per-node probe separately but ~6x cheaper.

    shuffle_seed: if given, the label rows are permuted across examples (the same
    permutation for every target column, so label structure is preserved and only
    the X-Y correspondence is destroyed) -> the chance-level control.
    """
    Htr = collect_hiddens(model, arm, ptrain, V, H).numpy()
    Hte = collect_hiddens(model, arm, ptest, V, H).numpy()
    dtr, dte = ptrain["dist"], ptest["dist"]
    if shuffle_seed is not None:
        srng = np.random.default_rng(shuffle_seed)
        dtr = dtr[srng.permutation(dtr.shape[0])]
        dte = dte[srng.permutation(dte.shape[0])]
    keys, Ytr_blocks = probe_targets(dtr, H)
    _, Yte_blocks = probe_targets(dte, H)
    Ytr = np.concatenate(Ytr_blocks, axis=1)
    N = V["N"]
    out = {"frontier": {}, "reach": {}}
    for slot in range(Htr.shape[1]):
        Xtr, Xte = Htr[:, slot], Hte[:, slot]
        mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
        sc = fit_logreg((Xtr - mu) / sd, Ytr, (Xte - mu) / sd, P)
        for i, (target, k) in enumerate(keys):
            out[target].setdefault(str(slot), {})[str(k)] = macro_auc(
                sc[:, i * N:(i + 1) * N], Yte_blocks[i])
    return out


def matched_diag(pr, H):
    """AUC of F_k read off the hidden AT thought slot k (slot 0 = before any thought)."""
    return [pr["frontier"].get(str(k), {}).get(str(k), float("nan")) for k in range(1, H + 1)]


def slot0_row(pr, H, target="frontier"):
    """AUC of F_k / R_k read off the query-position hidden, before any thought."""
    return [pr[target]["0"][str(k)] for k in range(1, H + 1)]


# ----------------------------- main ----------------------------------------
def main():
    cfg = load_config()
    P = cfg["params"]
    set_seeds(int(cfg.get("seed", 0)))
    t0 = time.time()
    log = lambda s: print(s, flush=True)
    N, E, H, K = P["n_nodes"], P["n_edges"], P["n_thoughts"], P["max_out_degree"]
    V = make_vocab(N)

    log("building data pools ...")
    td = time.time()
    train_pool = build_pool(1234, P["n_train"], N, E, H, V, K)
    test_pool = build_pool(777_001, P["n_test"], N, E, H, V, K)
    probe_tr = build_pool(777_002, P["n_probe_train"], N, E, H, V, K)
    probe_te = build_pool(777_003, P["n_probe_test"], N, E, H, V, K)
    frontier_sizes = {str(k): round(float((train_pool["dist"] == k).sum(1).mean()), 3)
                      for k in range(1, H + 1)}
    log(f"  pools built in {time.time()-td:.1f}s; prefix len={train_pool['prefix'].shape[1]}, "
        f"train label balance={train_pool['label'].mean():.3f}, "
        f"hop counts={ {h: int((train_pool['hop']==h).sum()) for h in range(0, H+1)} }, "
        f"mean |F_k|={frontier_sizes}")

    runs, models = [], {}
    for arm in P["arms"]:
        for seed in P["seeds"]:
            model, r = train_one(arm, seed, P, V, train_pool, test_pool, log)
            runs.append(r)
            if seed == P["seeds"][0]:
                models[arm] = model

    log("probing for the BFS frontier ...")
    probes = {}
    for arm in P["probe_arms"]:
        if arm not in models:
            continue
        tp = time.time()
        probes[arm] = run_probes(models[arm], arm, probe_tr, probe_te, V, H, P)
        log(f"  probe {arm:14s} {time.time()-tp:4.0f}s  slot0={slot0_row(probes[arm], H)}"
            f"  matched-slot={matched_diag(probes[arm], H)}")
    # control 1: shuffled labels on the coconut model (chance level)
    probes["shuffled_control"] = run_probes(models["coconut"], "coconut", probe_tr, probe_te,
                                            V, H, P, shuffle_seed=4242)
    log(f"  probe shuffled_ctrl    slot0={slot0_row(probes['shuffled_control'], H)}"
        f"  matched-slot={matched_diag(probes['shuffled_control'], H)}")
    # control 2: untrained randomly initialised model
    set_seeds(999)
    ctrl = TinyLM(V["size"], P["d_model"], P["n_heads"], P["d_ff"], P["n_layers"],
                  train_pool["prefix"].shape[1] + H + 2)
    probes["untrained_control"] = run_probes(ctrl, "coconut", probe_tr, probe_te, V, H, P)
    log(f"  probe untrained_ctrl   slot0={slot0_row(probes['untrained_control'], H)}"
        f"  matched-slot={matched_diag(probes['untrained_control'], H)}")

    # ----------------------------- aggregate ---------------------------------
    agg = {}
    for arm in P["arms"]:
        rs = [r for r in runs if r["arm"] == arm]
        hops = sorted({h for r in rs for h in r["acc_by_hop"]}, key=int)
        agg[arm] = {
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
        if arm == "cot":
            agg[arm]["trace_exact_mean"] = round(
                float(np.mean([r["trace_exact"] for r in rs])), 4)
    ranking = sorted(P["arms"], key=lambda a: -agg[a]["acc_mean"])
    d_coco_cot = round(agg["coconut"]["acc_mean"] - agg["cot"]["acc_mean"], 4)
    d_coco_pause = round(agg["coconut"]["acc_mean"] - agg["pause"]["acc_mean"], 4)
    d_coco_nocot = round(agg["coconut"]["acc_mean"] - agg["nocot"]["acc_mean"], 4)
    d_cur = round(agg["coconut"]["acc_mean"] - agg["coconut_nocur"]["acc_mean"], 4)

    diag_co = matched_diag(probes["coconut"], H)
    diag_sh = matched_diag(probes["shuffled_control"], H)
    diag_un = matched_diag(probes["untrained_control"], H)
    s0_co = slot0_row(probes["coconut"], H)
    s0_nocot = slot0_row(probes["nocot"], H) if "nocot" in probes else None

    metrics = {
        "per_run": runs,
        "aggregate": agg,
        "ranking_by_acc": ranking,
        "acc_mean_by_arm": {a: agg[a]["acc_mean"] for a in P["arms"]},
        "delta_coconut_minus_cot": d_coco_cot,
        "delta_coconut_minus_pause": d_coco_pause,
        "delta_coconut_minus_nocot": d_coco_nocot,
        "delta_curriculum": d_cur,
        "seed_std_max": round(float(max(agg[a]["acc_std"] for a in P["arms"])), 4),
        "mean_frontier_size": frontier_sizes,
        "probes": probes,
        "probe_frontier_matched_slot_coconut": diag_co,
        "probe_frontier_matched_slot_shuffled": diag_sh,
        "probe_frontier_matched_slot_untrained": diag_un,
        "probe_frontier_lift_over_shuffled": [round(a - b, 4) for a, b in zip(diag_co, diag_sh)],
        "probe_frontier_lift_over_untrained": [round(a - b, 4) for a, b in zip(diag_co, diag_un)],
        "probe_frontier_slot0_coconut": s0_co,
        "probe_frontier_slot0_nocot": s0_nocot,
        "probe_thought_gain_over_slot0": [round(a - b, 4) for a, b in zip(diag_co, s0_co)],
        "n_params": runs[0]["n_params"],
        "headline": (
            "held-out accuracy: " + ", ".join(f"{a}={agg[a]['acc_mean']:.3f}" for a in ranking)
            + f"; best={ranking[0]}; coconut-cot={d_coco_cot:+.3f}, "
              f"coconut-pause={d_coco_pause:+.3f}, coconut-nocot={d_coco_nocot:+.3f}; "
              f"frontier-probe AUC at matched slot (F1/F2/F3) {diag_co} "
              f"vs shuffled {diag_sh} vs slot0 {s0_co}"),
    }

    # ----------------------------- chart -------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"nocot": "#8a817c", "pause": "#c9a13c", "cot": "#3d5a80",
              "coconut": "#1a7f64", "coconut_nocur": "#7fbfa5",
              "shuffled_control": "#b23a48", "untrained_control": "#555555"}
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4), width_ratios=[1.1, 1.5, 1.5, 1.2])
    ax1, ax2, ax3, ax4 = axes

    ax1.bar([a.replace("_", "\n") for a in ranking], [agg[a]["acc_mean"] for a in ranking],
            yerr=[agg[a]["acc_std"] for a in ranking],
            color=[colors[a] for a in ranking], capsize=3)
    for i, a in enumerate(ranking):
        ax1.text(i, agg[a]["acc_mean"] + 0.025, f"{agg[a]['acc_mean']:.3f}",
                 ha="center", fontsize=8)
    ax1.axhline(0.5, color="0.5", ls="--", lw=1)
    ax1.text(len(ranking) - 0.4, 0.508, "chance", fontsize=7, color="0.5", ha="right")
    ax1.set_ylim(0.4, 1.08); ax1.set_ylabel("held-out accuracy")
    ax1.set_title("Overall (mean of 2 seeds)", fontsize=10)
    ax1.tick_params(axis="x", labelsize=7)

    hop_names = {"0": "no path\n(<=3 hops)", "1": "1-hop", "2": "2-hop", "3": "3-hop"}
    hops = ["1", "2", "3", "0"]
    xs = np.arange(len(hops)); w = 0.16
    for i, a in enumerate(P["arms"]):
        ys = [agg[a]["acc_by_hop_mean"].get(h, np.nan) for h in hops]
        ax2.bar(xs + (i - 2) * w, ys, width=w, color=colors[a], label=a)
    ax2.set_xticks(xs); ax2.set_xticklabels([hop_names[h] for h in hops], fontsize=8)
    ax2.axhline(0.5, color="0.5", ls="--", lw=1)
    ax2.set_ylim(0, 1.08); ax2.set_ylabel("accuracy")
    ax2.legend(frameon=False, fontsize=7, ncol=2)
    ax2.set_title("Stratified by shortest-path length", fontsize=10)

    ks = list(range(1, H + 1))
    for a, sty in (("coconut", "o-"), ("coconut_nocur", "d-"), ("cot", "s-"),
                   ("pause", "^-"), ("shuffled_control", "x--"), ("untrained_control", "+--")):
        if a not in probes:
            continue
        ys = matched_diag(probes[a], H)
        if any(isinstance(y, float) and math.isnan(y) for y in ys):
            continue
        ax3.plot(ks, ys, sty, lw=2, ms=5, color=colors.get(a, "0.35"),
                 label=a.replace("_control", " ctrl"))
    ax3.plot(ks, s0_co, ":", lw=2, color="#1a7f64", alpha=0.55,
             label="coconut slot0 (pre-thought)")
    if s0_nocot is not None:
        ax3.plot(ks, s0_nocot, ":", lw=2, color="#8a817c", alpha=0.8, label="nocot slot0")
    ax3.axhline(0.5, color="0.5", ls="--", lw=1)
    ax3.set_xticks(ks); ax3.set_xlabel("hop k  (probed at thought slot k)")
    ax3.set_ylabel("macro AUC, frontier $F_k$")
    ax3.set_ylim(0.4, 1.02); ax3.legend(frameon=False, fontsize=6.5, ncol=2)
    ax3.set_title("Probe: is node v at distance exactly k from src?", fontsize=10)

    mat = np.array([[probes["coconut"]["reach"][str(s)][str(k)] for k in ks]
                    for s in range(H + 1)])
    im = ax4.imshow(mat, cmap="viridis", vmin=0.5, vmax=1.0, aspect="auto")
    ax4.set_xticks(range(len(ks))); ax4.set_xticklabels([f"$R_{k}$" for k in ks])
    ax4.set_yticks(range(H + 1))
    ax4.set_yticklabels(["slot 0\n(pre)"] + [f"thought {s}" for s in range(1, H + 1)],
                        fontsize=8)
    for i in range(H + 1):
        for j in range(len(ks)):
            ax4.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8,
                     color="w" if mat[i, j] < 0.8 else "k")
    fig.colorbar(im, ax=ax4, fraction=0.046)
    ax4.set_title("coconut: reach-set $R_k$ AUC by slot", fontsize=10)

    for ax in axes[:3]:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"nano-Coconut on DAG reachability (N={N} nodes, E={E} edges, out-deg<={K}, "
                 f"H={H} thought slots, {metrics['n_params']/1e3:.0f}k params, "
                 f"{P['steps']} steps, 2 seeds)", fontsize=11, y=1.03)
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
