"""Interpretability with the answer key: probes and SAEs against a Tracr-compiled reverse circuit.

Pipeline
  0. Compile the RASP program `reverse` with Tracr (JAX) -> a 4-layer transformer whose every
     residual dimension carries a human-readable label (variable:value).
  1. Port the weights to torch/numpy and verify the port reproduces JAX to float precision.
  2. Verify the compiled model computes `reverse` EXACTLY on the exhaustive input set.
  3. Linear probes for every intermediate variable at every residual site, scored against the
     constructive answer key of WHERE each variable actually lives.
  4. Causal test (dyck-probe-can-lie port): zero the variable's OWN known dims at a site vs random
     rank-matched subspaces -> decodable vs used, with ground truth.
  5. L1 SAE on the residual stream. Answer key = the constructed axis basis (uncompressed) and the
     empirically-fitted encoding matrix of a linearly COMPRESSED tracr model (constructed
     superposition).

CPU only, single thread, deterministic. Usage: python run.py
"""
import json, os, random, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

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
    for mod in ("numpy", "torch", "jax", "tracr"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:
            pass
    return info


# ---------------------------------------------------------------------------
# torch port of a tracr-compiled transformer (no layernorm, non-causal, relu MLP)
# ---------------------------------------------------------------------------
class TorchTracr:
    def __init__(self, params, cfg, dtype):
        import numpy as np
        import torch
        self.t = torch
        self.dtype = dtype
        self.nl = cfg.num_layers
        self.ks = cfg.key_size
        self.P = {k: {kk: torch.tensor(np.asarray(vv), dtype=dtype) for kk, vv in v.items()}
                  for k, v in params.items()}
        self.tok_emb = self.P["token_embed"]["embeddings"]
        self.pos_emb = self.P["pos_embed"]["embeddings"]
        self.d_model = self.tok_emb.shape[1]

    def embed(self, tokens):
        torch = self.t
        Tt = tokens.shape[1]
        return self.tok_emb[tokens] + self.pos_emb[torch.arange(Tt)][None]

    def run(self, x, mask, hook=None):
        """x: (B,T,D) embeddings; mask: (B,T) bool key mask. hook(site_idx, resid)->resid."""
        import numpy as np
        torch = self.t
        NEG = torch.tensor(-1e30, dtype=self.dtype)
        if hook is not None:
            x = hook(0, x)
        resids = [x]
        for l in range(self.nl):
            pre = f"transformer/layer_{l}/"
            q = x @ self.P[pre + "attn/query"]["w"] + self.P[pre + "attn/query"]["b"]
            k = x @ self.P[pre + "attn/key"]["w"] + self.P[pre + "attn/key"]["b"]
            v = x @ self.P[pre + "attn/value"]["w"] + self.P[pre + "attn/value"]["b"]
            logits = torch.einsum("btd,bTd->btT", q, k) / float(np.sqrt(self.ks))
            logits = torch.where(mask[:, None, :], logits, NEG)
            w = torch.softmax(logits, dim=-1)
            a = torch.einsum("btT,bTd->btd", w, v)
            o = a @ self.P[pre + "attn/linear"]["w"] + self.P[pre + "attn/linear"]["b"]
            x = x + o
            if hook is not None:
                x = hook(2 * l + 1, x)
            resids.append(x)
            h = torch.relu(x @ self.P[pre + "mlp/linear_1"]["w"] + self.P[pre + "mlp/linear_1"]["b"])
            o2 = h @ self.P[pre + "mlp/linear_2"]["w"] + self.P[pre + "mlp/linear_2"]["b"]
            x = x + o2
            if hook is not None:
                x = hook(2 * l + 2, x)
            resids.append(x)
        return resids


# ---------------------------------------------------------------------------
# probes / SAE helpers
# ---------------------------------------------------------------------------
def fit_logreg(Xtr, ytr, Xte, yte, n_classes, steps, lr, wd, seed):
    import torch
    g = torch.Generator().manual_seed(seed)
    d = Xtr.shape[1]
    W = (0.01 * torch.randn(d, n_classes, generator=g)).requires_grad_(True)
    b = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xtr @ W + b, ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        logit = Xte @ W + b
        pred = logit.argmax(-1)
        acc = (pred == yte).double().mean().item()
    return acc, pred


class SAE:
    def __init__(self, d, n_feat, seed):
        import torch
        g = torch.Generator().manual_seed(seed)
        self.d, self.F = d, n_feat
        Wd = torch.randn(n_feat, d, generator=g)
        Wd = Wd / Wd.norm(dim=1, keepdim=True)
        self.W_dec = Wd.clone().requires_grad_(True)
        self.W_enc = Wd.t().clone().contiguous().requires_grad_(True)
        self.b_enc = torch.zeros(n_feat, requires_grad=True)
        self.b_dec = torch.zeros(d, requires_grad=True)

    def params(self):
        return [self.W_enc, self.b_enc, self.W_dec, self.b_dec]

    def encode(self, x):
        import torch
        return torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def decode(self, f):
        return f @ self.W_dec + self.b_dec


def train_sae(X, n_feat, lam, steps, batch, lr, seed):
    import torch
    sae = SAE(X.shape[1], n_feat, seed)
    opt = torch.optim.Adam(sae.params(), lr=lr)
    g = torch.Generator().manual_seed(seed + 7777)
    N = X.shape[0]
    fire_count = torch.zeros(n_feat)
    resample_at = {int(0.4 * steps), int(0.7 * steps)}
    for s in range(steps):
        idx = torch.randint(0, N, (batch,), generator=g)
        x = X[idx]
        f = sae.encode(x)
        xh = sae.decode(f)
        loss = ((x - xh) ** 2).sum(-1).mean() + lam * f.abs().sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            sae.W_dec.data /= sae.W_dec.data.norm(dim=1, keepdim=True).clamp_min(1e-8)
            fire_count += (f > 0).float().sum(0)
        if s in resample_at:
            with torch.no_grad():
                dead = (fire_count == 0).nonzero().flatten()
                if len(dead) > 0:
                    xs = X[torch.randint(0, N, (len(dead),), generator=g)]
                    dirs = xs - sae.b_dec
                    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-8)
                    sae.W_dec.data[dead] = dirs
                    sae.W_enc.data[:, dead] = dirs.t() * 0.2
                    sae.b_enc.data[dead] = 0.0
                fire_count.zero_()
            opt = torch.optim.Adam(sae.params(), lr=lr)
    with torch.no_grad():
        f = sae.encode(X)
        xh = sae.decode(f)
        fvu = (((X - xh) ** 2).sum() / ((X - X.mean(0)) ** 2).sum()).item()
        l0 = (f > 0).double().sum(-1).mean().item()
        fire_frac = (f > 0).double().mean(0)
    return sae, f.detach(), dict(fvu=fvu, l0=l0), fire_frac


def greedy_match(cos):
    """cos: (K keys, F feats). Greedy injective matching, highest cosine first."""
    import torch
    K, F = cos.shape
    c = cos.clone()
    out = [None] * K
    for _ in range(min(K, F)):
        v = float(c.max())
        if v <= -1e8:
            break
        flat = int(torch.argmax(c.reshape(-1)))
        i, j = flat // F, flat % F
        out[i] = (j, v)
        c[i, :] = -1e9
        c[:, j] = -1e9
    return out


def best_f1(f, y):
    """max F1 over thresholds of relu feature activation f predicting binary y."""
    import torch
    ys_sum = float(y.sum())
    if ys_sum == 0:
        return 0.0
    order = torch.argsort(f, descending=True)
    ys = y[order].double()
    tp = torch.cumsum(ys, 0)
    k = torch.arange(1, len(ys) + 1, dtype=torch.float64)
    prec = tp / k
    rec = tp / ys_sum
    f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-12)
    valid = f[order] > 0
    if int(valid.sum()) == 0:
        return 0.0
    return float(f1[valid].max())


def main():
    import numpy as np
    import itertools
    import torch
    torch.set_num_threads(1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_config()
    p = cfg["params"]
    seed = int(cfg.get("seed", 0))
    set_seeds(seed)
    t0 = time.time()
    M = {}
    stage_t = {}

    # ================================================================= 0. compile
    from tracr.compiler import compiling
    from tracr.rasp import rasp

    all_true = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE)
    length = rasp.SelectorWidth(all_true).named("length")
    opp_idx_sop = (length - rasp.indices).named("opp_idx")
    opp_idx_m1_sop = (opp_idx_sop - 1).named("opp_idx-1")
    rev_sel = rasp.Select(rasp.indices, opp_idx_m1_sop, rasp.Comparison.EQ).named("reverse_selector")
    program = rasp.Aggregate(rev_sel, rasp.tokens).named("reverse")

    VOCAB = set(p["vocab"])
    ts = time.time()
    model = compiling.compile_rasp_to_model(
        program, vocab=VOCAB, max_seq_len=p["max_seq_len"],
        compiler_bos=p["compiler_bos"], compiler_pad=p["compiler_pad"])
    stage_t["compile"] = round(time.time() - ts, 2)
    mcfg = model.model_config
    labels = list(model.residual_labels)
    D = len(labels)
    M["tracr_path"] = ("tracr INSTALLED from github (pip package name does not exist); "
                       "no hand-compiled fallback needed")
    M["program"] = "reverse(tokens) = Aggregate(Select(indices, opp_idx-1, EQ), tokens)"
    M["n_layers"] = mcfg.num_layers
    M["n_heads"] = mcfg.num_heads
    M["key_size"] = mcfg.key_size
    M["mlp_hidden"] = mcfg.mlp_hidden_size
    M["layer_norm"] = mcfg.layer_norm
    M["causal"] = mcfg.causal
    M["d_model"] = D
    M["residual_labels"] = labels
    M["n_params"] = int(sum(int(np.asarray(v).size) for m_ in model.params.values()
                            for v in m_.values()))

    enc = model.input_encoder
    emap = dict(enc.encoding_map)
    pad_id, bos_id = emap[p["compiler_pad"]], emap[p["compiler_bos"]]
    out_order = sorted(model.output_encoder.encoding_map,
                       key=lambda k: model.output_encoder.encoding_map[k])

    def find_block(prefix):
        bases = {(l.rsplit(":", 1)[0] if ":" in l else l) for l in labels}
        cands = [b for b in bases if (b == prefix or b.startswith(prefix + "_"))
                 and not b.endswith("_selector_width_attn_output")]
        assert len(cands) == 1, (prefix, sorted(cands))
        blk = {}
        for i, l in enumerate(labels):
            if ":" in l and l.rsplit(":", 1)[0] == cands[0]:
                blk[l.rsplit(":", 1)[1]] = i
        return cands[0], blk

    VARS = {}
    for key, prefix in [("tokens", "tokens"), ("indices", "indices"), ("length", "length"),
                        ("opp_idx", "opp_idx"), ("opp_idx_m1", "opp_idx-1"),
                        ("reverse", "reverse")]:
        node, blk = find_block(prefix)
        VARS[key] = dict(node=node, dims=blk)
    M["variable_nodes"] = {k: v["node"] for k, v in VARS.items()}
    out_dims = [VARS["reverse"]["dims"][str(v)] for v in out_order]

    # ================================================================= 1. torch port
    tt = TorchTracr(model.params, mcfg, torch.float64)

    voc = sorted(VOCAB)
    seqs = []
    for L in range(p["min_real_len"], p["max_real_len"] + 1):
        for combo in itertools.product(voc, repeat=L):
            seqs.append(list(combo))
    Nseq = len(seqs)
    T = p["max_seq_len"]
    tok_ids = np.full((Nseq, T), pad_id, dtype=np.int64)
    for i, s in enumerate(seqs):
        tok_ids[i, 0] = bos_id
        for j, c in enumerate(s):
            tok_ids[i, 1 + j] = emap[c]
    tok_ids_t = torch.tensor(tok_ids)
    mask = tok_ids_t != pad_id

    emb = tt.embed(tok_ids_t)
    resids = tt.run(emb, mask)
    SITES = ["emb"] + [f"L{l}.{k}" for l in range(mcfg.num_layers) for k in ("attn", "mlp")]
    M["sites"] = SITES

    rng = np.random.default_rng(seed)
    sample = rng.choice(Nseq, size=min(40, Nseq), replace=False)
    maxdiff = 0.0
    for i in sample:
        toks = [p["compiler_bos"]] + seqs[int(i)]
        jx = model.apply(toks)
        jr = [np.asarray(jx.input_embeddings)] + [np.asarray(z) for z in jx.residuals]
        for si, jrr in enumerate(jr):
            tr = resids[si][int(i), : len(toks)].numpy()
            maxdiff = max(maxdiff, float(np.abs(tr - jrr[0]).max()))
    M["torch_port_max_abs_diff_vs_jax"] = maxdiff
    M["torch_port_n_inputs_checked"] = int(len(sample))

    # ================================================================= 2. exact correctness
    final = resids[-1]
    decoded = final[:, :, out_dims].argmax(-1).numpy()
    ok = 0
    for i, s in enumerate(seqs):
        if [out_order[decoded[i, 1 + j]] for j in range(len(s))] == s[::-1]:
            ok += 1
    M["n_inputs_exhaustive"] = Nseq
    M["exact_sequence_accuracy"] = ok / Nseq
    M["exhaustive_over"] = (f"ALL sequences of length {p['min_real_len']}..{p['max_real_len']} "
                            f"over |V|={len(voc)}")

    # ---------------- per-position ground truth ----------------
    pos_seq, pos_idx = [], []
    for i, s in enumerate(seqs):
        for j in range(len(s)):
            pos_seq.append(i); pos_idx.append(1 + j)
    pos_seq = np.array(pos_seq); pos_idx = np.array(pos_idx)
    Npos = len(pos_seq)
    M["n_positions"] = int(Npos)
    M["positions_used"] = "all non-BOS non-PAD positions"

    lens = np.array([len(s) for s in seqs])
    gt = {}
    gt["tokens"] = np.array([emap[seqs[i][j - 1]] for i, j in zip(pos_seq, pos_idx)])
    # tracr's position embedding row 0 is all-zero, so model position i carries rasp `indices` i-1
    gt["indices"] = pos_idx - 1
    gt["length"] = lens[pos_seq]
    gt["opp_idx"] = lens[pos_seq] - gt["indices"]
    gt["opp_idx_m1"] = gt["opp_idx"] - 1
    gt["reverse"] = np.array([emap[seqs[i][len(seqs[i]) - (j - 1) - 1]]
                              for i, j in zip(pos_seq, pos_idx)])

    rasp_ok = True
    for i in rng.choice(Nseq, size=min(60, Nseq), replace=False):
        s = seqs[int(i)]
        if list(program(s)) != s[::-1]:
            rasp_ok = False
        if list(length(s)) != [len(s)] * len(s):
            rasp_ok = False
        if list(opp_idx_m1_sop(s)) != [len(s) - k - 1 for k in range(len(s))]:
            rasp_ok = False
    M["rasp_ground_truth_selfcheck"] = rasp_ok

    # ---- where does each variable LIVE? (constructive answer key) ----
    lives, readoff = {}, {}
    for v, info in VARS.items():
        vals = sorted(info["dims"])
        dims = [info["dims"][k] for k in vals]
        if v in ("tokens", "reverse"):
            valcode = np.array([emap[k] for k in vals])
        else:
            valcode = np.array([int(k) for k in vals])
        row = []
        for si in range(len(SITES)):
            blk = resids[si][pos_seq, pos_idx][:, dims].numpy()
            pred = valcode[blk.argmax(-1)]
            row.append(dict(readoff_acc=float((pred == gt[v]).mean()),
                            frac_onehot_active=float((blk.max(-1) > 0.5).mean())))
        readoff[v] = row
        lives[v] = [bool(r["readoff_acc"] > 0.9999 and r["frac_onehot_active"] > 0.9999)
                    for r in row]
    # tracr's selector-width trick writes length in a NUMERICAL encoding (1/(1+w)) one half-layer
    # before the MLP converts it to the one-hot block. Track that separately: it is the honest
    # explanation for one class of probe "false positive".
    swdim = [i for i, l in enumerate(labels) if l.endswith("_selector_width_attn_output")]
    num_len = []
    for si in range(len(SITES)):
        if not swdim:
            num_len.append(False); continue
        x = resids[si][pos_seq, pos_idx][:, swdim[0]].numpy()
        vals = {}
        ok_ = True
        for L in sorted(set(gt["length"].tolist())):
            u = np.unique(np.round(x[gt["length"] == L], 6))
            if len(u) != 1:
                ok_ = False; break
            vals[L] = float(u[0])
        num_len.append(bool(ok_ and len(set(vals.values())) == len(vals)))
    M["length_numerically_exact_at_site"] = [SITES[i] for i, b in enumerate(num_len) if b]
    M["selector_width_dim"] = labels[swdim[0]] if swdim else None

    M["readoff_from_own_dims"] = {v: [round(r["readoff_acc"], 4) for r in readoff[v]] for v in VARS}
    M["lives_at_site"] = {v: [SITES[i] for i, b in enumerate(lives[v]) if b] for v in VARS}
    M["birth_site"] = {v: (SITES[lives[v].index(True)] if any(lives[v]) else None) for v in VARS}
    stage_t["compile_verify"] = round(time.time() - t0, 2)

    # ================================================================= 3. linear probes
    ts = time.time()
    perm = rng.permutation(Nseq)
    ntr = int(p["probe_train_frac"] * Nseq)
    tr_seq = set(perm[:ntr].tolist())
    tr_mask = np.array([s in tr_seq for s in pos_seq])
    te_mask = ~tr_mask
    trm, tem = torch.tensor(tr_mask), torch.tensor(te_mask)
    M["probe_n_train_positions"] = int(tr_mask.sum())
    M["probe_n_test_positions"] = int(te_mask.sum())
    M["probe_split"] = "by sequence (no sequence appears in both splits)"

    acts = []
    for si in range(len(SITES)):
        A = resids[si][pos_seq, pos_idx].to(torch.float32)
        mu, sd = A[trm].mean(0), A[trm].std(0).clamp_min(1e-6)
        acts.append((A - mu) / sd)

    probe = {}
    for v in VARS:
        classes = sorted(set(gt[v].tolist()))
        cmap = {c: i for i, c in enumerate(classes)}
        y = torch.tensor([cmap[c] for c in gt[v].tolist()])
        ytr, yte = y[trm], y[tem]
        maj = float(torch.bincount(yte, minlength=len(classes)).max()) / len(yte)
        accs, sh_accs, preds = [], [], []
        for si in range(len(SITES)):
            Xtr, Xte = acts[si][trm], acts[si][tem]
            a, pr = fit_logreg(Xtr, ytr, Xte, yte, len(classes), p["probe_steps"],
                               p["probe_lr"], p["probe_wd"], seed + si)
            gsh = torch.Generator().manual_seed(seed * 31 + si)
            ysh = ytr[torch.randperm(len(ytr), generator=gsh)]
            ash, _ = fit_logreg(Xtr, ysh, Xte, yte, len(classes), p["probe_steps"],
                                p["probe_lr"], p["probe_wd"], seed + si)
            accs.append(a); sh_accs.append(ash); preds.append(pr)
        probe[v] = dict(acc=accs, shuffled=sh_accs, majority=maj,
                        n_classes=len(classes), preds=preds)

    M["probe_acc"] = {v: [round(a, 4) for a in probe[v]["acc"]] for v in VARS}
    M["probe_shuffled_label_acc"] = {v: [round(a, 4) for a in probe[v]["shuffled"]] for v in VARS}
    M["probe_majority"] = {v: round(probe[v]["majority"], 4) for v in VARS}
    M["probe_n_classes"] = {v: probe[v]["n_classes"] for v in VARS}

    marg = p["detect_margin"]
    TP = FP = FN = TN = 0
    conf = {}
    for v in VARS:
        row = []
        for si in range(len(SITES)):
            det = probe[v]["acc"][si] - probe[v]["majority"] >= marg
            liv = lives[v][si]
            row.append(("TP" if det else "FN") if liv else ("FP" if det else "TN"))
            if liv and det: TP += 1
            elif liv and not det: FN += 1
            elif (not liv) and det: FP += 1
            else: TN += 1
        conf[v] = row
    M["localization_confusion"] = conf
    M["localization_TP"], M["localization_FP"] = TP, FP
    M["localization_FN"], M["localization_TN"] = FN, TN
    M["localization_accuracy"] = round((TP + TN) / (TP + TN + FP + FN), 4)
    M["localization_precision"] = round(TP / max(TP + FP, 1), 4)
    M["localization_recall"] = round(TP / max(TP + FN, 1), 4)
    M["localization_false_positive_rate"] = round(FP / max(FP + TN, 1), 4)
    M["detect_rule"] = f"probe test acc - majority >= {marg}"

    sweep = {}
    for mg in [0.02, 0.05, 0.1, 0.2, 0.4]:
        tp = fp = fn = tn = 0
        for v in VARS:
            for si in range(len(SITES)):
                det = probe[v]["acc"][si] - probe[v]["majority"] >= mg
                liv = lives[v][si]
                if liv and det: tp += 1
                elif liv and not det: fn += 1
                elif (not liv) and det: fp += 1
                else: tn += 1
        sweep[str(mg)] = dict(TP=tp, FP=fp, FN=fn, TN=tn,
                              acc=round((tp + tn) / (tp + tn + fp + fn), 4))
    M["localization_margin_sweep"] = sweep

    # why does the `reverse` probe fire before `reverse` exists?
    fixed = (gt["opp_idx_m1"] == (pos_idx - 1))
    M["frac_positions_selfreverse_fixedpoint"] = round(float(fixed.mean()), 4)
    te_idx = np.where(te_mask)[0]
    classes_r = sorted(set(gt["reverse"].tolist()))
    cmap_r = {c: i for i, c in enumerate(classes_r)}
    yv = np.array([cmap_r[c] for c in gt["reverse"][te_idx].tolist()])
    fte = fixed[te_idx]
    fp_detail = {}
    for si in range(len(SITES)):
        if lives["reverse"][si]:
            continue
        pr = probe["reverse"]["preds"][si].numpy()
        fp_detail[SITES[si]] = dict(
            acc_all=round(float((pr == yv).mean()), 4),
            acc_on_fixedpoints=round(float((pr[fte] == yv[fte]).mean()), 4) if fte.sum() else None,
            acc_off_fixedpoints=round(float((pr[~fte] == yv[~fte]).mean()), 4) if (~fte).sum() else None,
            frac_fixedpoints=round(float(fte.mean()), 4))
    M["reverse_probe_before_it_exists"] = fp_detail
    M["reverse_probe_fixedpoint_prediction"] = round(
        float(fte.mean() + (1 - fte.mean()) / len(classes_r)), 4)

    M["length_probe_before_it_exists"] = {
        SITES[si]: round(probe["length"]["acc"][si], 4)
        for si in range(len(SITES)) if not lives["length"][si]}
    # bayes bound for length at the embedding: a position knows only its own index i
    cnt = np.bincount(lens, minlength=p["max_real_len"] + 1).astype(float)
    right = 0
    for i in range(1, p["max_real_len"] + 1):
        elig = cnt.copy(); elig[:i] = 0
        m_ = int(elig.argmax())
        sel = (pos_idx == i)
        right += int(((lens[pos_seq] == m_) & sel).sum())
    M["length_probe_bayes_bound_at_embedding"] = round(right / Npos, 4)
    stage_t["probes"] = round(time.time() - ts, 2)

    # ================================================================= 4. causal test
    ts = time.time()

    def run_with_projection(site, Pmat):
        def hook(si, x):
            return x @ Pmat if si == site else x
        r = tt.run(emb, mask, hook=hook)
        dec = r[-1][:, :, out_dims].argmax(-1).numpy()
        good = 0
        for i, s in enumerate(seqs):
            if [out_order[dec[i, 1 + j]] for j in range(len(s))] == s[::-1]:
                good += 1
        return good / Nseq

    I = torch.eye(D, dtype=torch.float64)
    gcau = torch.Generator().manual_seed(seed + 999)
    rand_cache, causal = {}, {}
    for v in VARS:
        dims = sorted(VARS[v]["dims"].values())
        r = len(dims)
        Pv = I.clone()
        for d_ in dims:
            Pv[d_, d_] = 0.0
        row = []
        for si in range(len(SITES)):
            acc_v = run_with_projection(si, Pv)
            if (r, si) not in rand_cache:
                accs_r = []
                for _ in range(p["n_random_erasers"]):
                    Q = torch.linalg.qr(torch.randn(D, r, generator=gcau, dtype=torch.float64))[0]
                    accs_r.append(run_with_projection(si, I - Q @ Q.t()))
                rand_cache[(r, si)] = (float(np.mean(accs_r)), float(np.std(accs_r)))
            mu_r, sd_r = rand_cache[(r, si)]
            row.append(dict(rank=r, acc_after_erasing_var=round(acc_v, 4),
                            acc_after_random_rank_matched=round(mu_r, 4),
                            random_sd=round(sd_r, 4),
                            damage=round(M["exact_sequence_accuracy"] - acc_v, 4),
                            excess_damage_over_random=round(mu_r - acc_v, 4)))
        causal[v] = row
    M["causal_erasure"] = causal
    # inert == erasing the variable's OWN dims costs nothing, while a random rank-matched
    # erasure at the same site is free to be catastrophic (reported alongside).
    dec_unused = []
    for v in VARS:
        for si in range(len(SITES)):
            if lives[v][si] and probe[v]["acc"][si] > 0.99 and causal[v][si]["damage"] <= 0.01:
                dec_unused.append(dict(var=v, site=SITES[si],
                                       probe_acc=round(probe[v]["acc"][si], 4),
                                       damage=causal[v][si]["damage"],
                                       random_rank_matched_damage=round(
                                           M["exact_sequence_accuracy"]
                                           - causal[v][si]["acc_after_random_rank_matched"], 4)))
    M["decodable_but_causally_inert_cells"] = dec_unused
    M["decodable_but_causally_inert_rule"] = ("variable LIVES at the site, probe acc > 0.99, and "
                                              "erasing its own dims costs <= 0.01 exact accuracy")
    M["n_live_cells"] = int(sum(sum(lives[v]) for v in VARS))
    stage_t["causal"] = round(time.time() - ts, 2)

    # ================================================================= 5a. SAE, clean basis
    ts = time.time()
    SAE_SITE = len(SITES) - 1
    M["sae_site"] = SITES[SAE_SITE]
    Xraw = resids[SAE_SITE][pos_seq, pos_idx].to(torch.float32)
    X = Xraw * (float(np.sqrt(D)) / Xraw.norm(dim=1).mean().item())
    alive_axes = [j for j in range(D) if float(Xraw[:, j].std()) > 1e-6]
    K = len(alive_axes)
    M["n_alive_axes_clean"] = K
    M["alive_axes_labels"] = [labels[j] for j in alive_axes]
    M["dead_axes_labels"] = [labels[j] for j in range(D) if j not in alive_axes]
    binary_axis = [bool(((Xraw[:, j] == 0) | ((Xraw[:, j] - 1).abs() < 1e-6)).all())
                   for j in alive_axes]
    M["n_binary_axes_clean"] = int(sum(binary_axis))
    M["non_binary_axes_labels"] = [labels[j] for j, b in zip(alive_axes, binary_axis) if not b]

    key_clean = torch.zeros(K, D, dtype=torch.float32)
    for a, j in enumerate(alive_axes):
        key_clean[a, j] = 1.0
    axis_vals = [(Xraw[:, j] > 0.5) if binary_axis[a] else Xraw[:, j]
                 for a, j in enumerate(alive_axes)]
    M["axis_mean_fire_frac"] = round(float(np.mean(
        [float((Xraw[:, j] > 0.5).float().mean()) for j in alive_axes])), 4)
    # how redundant is the ANSWER KEY itself?  (opp_idx-1 == opp_idx - 1 makes some axes identical)
    Yk = torch.stack([v.float() for v in axis_vals], 1)
    Yc_ = Yk - Yk.mean(0)
    C = (Yc_.t() @ Yc_) / (Yc_.norm(dim=0)[:, None] * Yc_.norm(dim=0)[None, :]).clamp_min(1e-12)
    Ca = C.abs().clone(); Ca.fill_diagonal_(0)
    dup = [(labels[alive_axes[i]], labels[alive_axes[j]], round(float(Ca[i, j]), 4))
           for i in range(K) for j in range(i + 1, K) if float(Ca[i, j]) >= 0.99]
    M["answer_key_duplicate_axis_pairs"] = dup
    M["answer_key_n_duplicate_pairs"] = len(dup)
    M["answer_key_mean_abs_offdiag_corr"] = round(float(Ca.sum() / (K * K - K)), 4)
    M["true_code_mean_L0"] = round(float((Xraw[:, alive_axes] > 0.5).float().sum(1).mean()), 3)

    def _behav(fi, a, avals, is_bin):
        """behavioural agreement between a feature and a ground-truth axis."""
        if is_bin[a]:
            return best_f1(fi, avals[a])
        fv = fi - fi.mean(); av = avals[a].float() - avals[a].float().mean()
        return abs(float((fv @ av) / (fv.norm() * av.norm()).clamp_min(1e-12)))

    def score_sae(sae, f, fire_frac, key, avals, is_bin, cos_thr, f1_thr, alive_thr):
        zero = dict(n_alive=0, n_recovered_dir=0, n_recovered_dir_and_func=0,
                    n_recovered_behavioural=0, mean_matched_cos=0.0, naive_max_cos_mean=0.0,
                    matches=[])
        alive_f = (fire_frac >= alive_thr).nonzero().flatten()
        if len(alive_f) == 0:
            return zero
        Wd = sae.W_dec.detach()[alive_f]
        Wd = Wd / Wd.norm(dim=1, keepdim=True).clamp_min(1e-8)
        F = f[:, alive_f]
        cosmat = key @ Wd.t()
        Y = torch.stack([avals[a].float() for a in range(key.shape[0])], 1)
        Fc = F - F.mean(0); Yc = Y - Y.mean(0)
        cormat = (Yc.t() @ Fc) / (Yc.norm(dim=0)[:, None] * Fc.norm(dim=0)[None, :]).clamp_min(1e-12)
        mt_dir = greedy_match(cosmat)
        mt_fun = greedy_match(cormat.abs())
        ndir = nboth = nbeh = 0
        matches, mcos = [], []
        for a, m in enumerate(mt_dir):
            if m is None:
                matches.append(None); continue
            j, c = m
            sc = _behav(F[:, j], a, avals, is_bin)
            mcos.append(float(c))
            ndir += int(c >= cos_thr); nboth += int(c >= cos_thr and sc >= f1_thr)
            matches.append(dict(cos=round(float(c), 4), behav=round(float(sc), 4)))
        for a, m in enumerate(mt_fun):
            if m is None:
                continue
            nbeh += int(_behav(F[:, m[0]], a, avals, is_bin) >= f1_thr)
        return dict(n_alive=int(len(alive_f)), n_recovered_dir=ndir,
                    n_recovered_dir_and_func=nboth, n_recovered_behavioural=nbeh,
                    mean_matched_cos=round(float(np.mean(mcos)) if mcos else 0.0, 4),
                    naive_max_cos_mean=round(float(cosmat.max(1).values.mean()), 4),
                    matches=matches)

    def null_recovery(key, n_feat, repeats, cos_thr, seed0):
        vals = []
        for t_ in range(repeats):
            g = torch.Generator().manual_seed(seed0 + t_)
            R = torch.randn(n_feat, key.shape[1], generator=g)
            R = R / R.norm(dim=1, keepdim=True)
            mt = greedy_match(key @ R.t())
            vals.append(sum(1 for m in mt if m is not None and m[1] >= cos_thr))
        return [round(float(np.mean(vals)), 3), float(np.percentile(vals, 95)), float(np.max(vals))]

    def run_sae_grid(Xd, key, avals, is_bin, tag, expansions=None, seeds=None):
        grid, best = {}, None
        for ex in (expansions or p["sae_expansions"]):
            for lam in p["sae_lambdas"]:
                for sd in (seeds or p["sae_seeds"]):
                    nf = int(ex * Xd.shape[1])
                    sae, f, st, ff = train_sae(Xd, nf, lam, p["sae_steps"], p["sae_batch"],
                                               p["sae_lr"], seed * 100 + sd)
                    sc = score_sae(sae, f, ff, key, avals, is_bin, p["sae_cos_threshold"],
                                   p["sae_f1_threshold"], p["sae_alive_fire_frac"])
                    cell = dict(expansion=ex, lam=lam, seed=sd, n_feat=nf,
                                fvu=round(st["fvu"], 5), l0=round(st["l0"], 3),
                                frac_dead=round(1 - sc["n_alive"] / nf, 4),
                                n_alive=sc["n_alive"], recovered_dir=sc["n_recovered_dir"],
                                recovered_func=sc["n_recovered_dir_and_func"],
                                recovered_behavioural=sc["n_recovered_behavioural"],
                                mean_matched_cos=sc["mean_matched_cos"],
                                naive_max_cos_mean=sc["naive_max_cos_mean"])
                    grid[f"{tag}_x{ex}_l{lam}_s{sd}"] = cell
                    kb = (sc["n_recovered_dir_and_func"], sc["n_recovered_dir"],
                          sc["n_recovered_behavioural"], -st["fvu"])
                    if best is None or kb > best[0]:
                        best = (kb, cell, sc)
        return grid, best

    grid_clean, best_clean = run_sae_grid(X, key_clean, axis_vals, binary_axis, "clean")
    M["sae_grid_clean"] = grid_clean
    M["sae_best_clean"] = best_clean[1]
    M["sae_best_clean_matches"] = best_clean[2]["matches"]
    M["sae_best_clean_recovered_func_of_K"] = f"{best_clean[1]['recovered_func']}/{K}"
    M["sae_best_clean_recovered_dir_of_K"] = f"{best_clean[1]['recovered_dir']}/{K}"
    M["sae_best_clean_recovery_rate"] = round(best_clean[1]["recovered_func"] / K, 4)
    M["sae_best_behavioural_clean_of_K"] = f"{max(c['recovered_behavioural'] for c in grid_clean.values())}/{K}"
    M["sae_max_recovered_dir_over_grid_clean"] = max(c["recovered_dir"] for c in grid_clean.values())
    M["sae_max_matched_cos_over_grid_clean"] = max(c["mean_matched_cos"] for c in grid_clean.values())
    M["sae_max_naive_max_cos_clean"] = max(c["naive_max_cos_mean"] for c in grid_clean.values())
    M["sae_null_clean_mean_p95_max"] = null_recovery(key_clean, best_clean[1]["n_feat"],
                                                     p["sae_null_repeats"],
                                                     p["sae_cos_threshold"], seed + 4242)
    M["neuron_baseline_clean_recovery"] = f"{K}/{K} (the residual axes ARE the key, by construction)"
    stage_t["sae_clean"] = round(time.time() - ts, 2)

    # ================================================================= 5b. compressed tracr
    ts = time.time()
    emb32 = emb.to(torch.float32)
    tt32 = TorchTracr(model.params, mcfg, torch.float32)
    resids32 = [r.to(torch.float32) for r in resids]
    tgt_all = {i: [out_order.index(seqs[i][len(seqs[i]) - j - 1]) for j in range(len(seqs[i]))]
               for i in range(Nseq)}

    def compressed_forward(W, x_emb, msk):
        r = x_emb @ W.t()
        outs = [r]
        NEG = torch.tensor(-1e30, dtype=torch.float32)
        for l in range(mcfg.num_layers):
            pre = f"transformer/layer_{l}/"
            x = r @ W
            q = x @ tt32.P[pre + "attn/query"]["w"] + tt32.P[pre + "attn/query"]["b"]
            k = x @ tt32.P[pre + "attn/key"]["w"] + tt32.P[pre + "attn/key"]["b"]
            v = x @ tt32.P[pre + "attn/value"]["w"] + tt32.P[pre + "attn/value"]["b"]
            lo = torch.einsum("btd,bTd->btT", q, k) / float(np.sqrt(mcfg.key_size))
            lo = torch.where(msk[:, None, :], lo, NEG)
            a = torch.einsum("btT,bTd->btd", torch.softmax(lo, -1), v)
            o = a @ tt32.P[pre + "attn/linear"]["w"] + tt32.P[pre + "attn/linear"]["b"]
            r = r + o @ W.t()
            outs.append(r)
            x = r @ W
            h = torch.relu(x @ tt32.P[pre + "mlp/linear_1"]["w"] + tt32.P[pre + "mlp/linear_1"]["b"])
            o2 = h @ tt32.P[pre + "mlp/linear_2"]["w"] + tt32.P[pre + "mlp/linear_2"]["b"]
            r = r + o2 @ W.t()
            outs.append(r)
        return outs

    def compressed_exact_acc(W):
        with torch.no_grad():
            o = compressed_forward(W, emb32, mask)
            dec = (o[-1] @ W)[:, :, out_dims].argmax(-1).numpy()
        good = 0
        for i, s in enumerate(seqs):
            if [out_order[dec[i, 1 + j]] for j in range(len(s))] == s[::-1]:
                good += 1
        return good / Nseq

    comp_results, chosen_k, chosen_W = {}, None, None
    for k in p["compress_dims"]:
        gW = torch.Generator().manual_seed(seed + 13 * k)
        W = (torch.randn(k, D, generator=gW) / float(np.sqrt(D))).requires_grad_(True)
        opt = torch.optim.Adam([W], lr=p["compress_lr"])
        gb = torch.Generator().manual_seed(seed + 77 * k)
        for _ in range(p["compress_steps"]):
            bidx = torch.randint(0, Nseq, (p["compress_batch"],), generator=gb)
            outs = compressed_forward(W, emb32[bidx], mask[bidx])
            lo = (outs[-1] @ W)[:, :, out_dims]
            si_, sj_, tgt = [], [], []
            for a, b in enumerate(bidx.tolist()):
                for j in range(len(seqs[b])):
                    si_.append(a); sj_.append(1 + j); tgt.append(tgt_all[b][j])
            ce = torch.nn.functional.cross_entropy(
                lo[torch.tensor(si_), torch.tensor(sj_)], torch.tensor(tgt))
            mb = mask[bidx]
            mse = sum(((outs[t_] @ W - resids32[t_][bidx]) ** 2)[mb].mean()
                      for t_ in range(len(outs)))
            (ce + mse).backward()
            opt.step(); opt.zero_grad()
        acc = compressed_exact_acc(W)
        comp_results[str(k)] = dict(k=k, compression_ratio=round(D / k, 3),
                                    exact_sequence_accuracy=round(acc, 4))
        if acc > 0.999:
            chosen_k, chosen_W = k, W.detach().clone()
    M["compression_sweep"] = comp_results
    M["compression_chosen_k"] = chosen_k
    M["compression_note"] = ("residual lives in R^k; read = r@W, write = y@W^T, tracr weights FROZEN; "
                             "W trained on output CE + layerwise MSE to the original residual stream")
    stage_t["compression"] = round(time.time() - ts, 2)

    # -------- 5c. INDEPENDENCE CONTROL --------------------------------------
    # Same K axes, same marginal firing rates, but drawn INDEPENDENTLY. If the SAE aces this and
    # fails on the real tracr code, the failure is caused by the code's correlation structure,
    # not by the SAE, the dictionary size, or this harness.
    ts = time.time()
    gctl = torch.Generator().manual_seed(seed + 31337)
    Xctl_raw = torch.zeros_like(Xraw)
    for a, j in enumerate(alive_axes):
        if binary_axis[a]:
            pr = float((Xraw[:, j] > 0.5).float().mean())
            Xctl_raw[:, j] = (torch.rand(Npos, generator=gctl) < pr).float()
        else:
            Xctl_raw[:, j] = Xraw[torch.randint(0, Npos, (Npos,), generator=gctl), j]
    for j in range(D):
        if j not in alive_axes:
            Xctl_raw[:, j] = Xraw[0, j]
    Xctl = Xctl_raw * (float(np.sqrt(D)) / Xctl_raw.norm(dim=1).mean().item())
    axis_vals_ctl = [(Xctl_raw[:, j] > 0.5) if binary_axis[a] else Xctl_raw[:, j]
                     for a, j in enumerate(alive_axes)]
    Yc2 = torch.stack([v.float() for v in axis_vals_ctl], 1)
    Yc2 = Yc2 - Yc2.mean(0)
    Cc = (Yc2.t() @ Yc2) / (Yc2.norm(dim=0)[:, None] * Yc2.norm(dim=0)[None, :]).clamp_min(1e-12)
    Cc = Cc.abs(); Cc.fill_diagonal_(0)
    M["control_key_mean_abs_offdiag_corr"] = round(float(Cc.sum() / (K * K - K)), 4)
    M["control_mean_L0"] = round(float((Xctl_raw[:, alive_axes] > 0.5).float().sum(1).mean()), 3)
    grid_ctl, best_ctl = run_sae_grid(Xctl, key_clean, axis_vals_ctl, binary_axis, "ctl",
                                      expansions=p["sae_control_expansions"], seeds=[0])
    M["sae_grid_independence_control"] = grid_ctl
    M["sae_control_best_recovered_dir_of_K"] = \
        f"{max(c['recovered_dir'] for c in grid_ctl.values())}/{K}"
    M["sae_control_best_recovered_behavioural_of_K"] = \
        f"{max(c['recovered_behavioural'] for c in grid_ctl.values())}/{K}"
    M["sae_control_best_mean_matched_cos"] = max(c["mean_matched_cos"] for c in grid_ctl.values())
    M["independence_control_note"] = (
        "identical K axes and marginal firing rates, sampled INDEPENDENTLY; same SAE code, "
        "same scoring, reduced grid")
    stage_t["sae_control"] = round(time.time() - ts, 2)

    # -------- 5d. CONSTRUCTED SUPERPOSITION (activation-space projection) ----
    # The trained model compression above does not reach exactness, so the superposed case is built
    # in ACTIVATION space instead: a random orthonormal-row projection P (k x D) of the SAME known
    # code. Key directions are P e_j, so the answer key stays exact while the basis stops being
    # axis-aligned and the K features no longer fit in orthogonal directions.
    ts = time.time()
    proj_grids, proj_summary = {}, {}
    for k in p["proj_dims"]:
        gP = torch.Generator().manual_seed(seed + 555 + k)
        Pm = torch.linalg.qr(torch.randn(D, k, generator=gP))[0].t()  # (k, D), orthonormal rows
        Xp = X @ Pm.t()
        Xp = Xp * (float(np.sqrt(k)) / Xp.norm(dim=1).mean().item())
        keyp = Pm[:, alive_axes].t()
        keyp = keyp / keyp.norm(dim=1, keepdim=True).clamp_min(1e-8)
        od = (keyp @ keyp.t()).abs(); od.fill_diagonal_(0)
        g_, b_ = run_sae_grid(Xp, keyp, axis_vals, binary_axis, f"proj{k}",
                              expansions=p["sae_proj_expansions"], seeds=[0])
        proj_grids[str(k)] = g_
        proj_summary[str(k)] = dict(
            k=k, superposition_ratio=round(K / k, 3),
            key_mean_abs_cos=round(float(od.sum() / (K * K - K)), 4),
            key_max_abs_cos=round(float(od.max()), 4),
            best_recovered_dir=max(c["recovered_dir"] for c in g_.values()),
            best_recovered_behavioural=max(c["recovered_behavioural"] for c in g_.values()),
            best_mean_matched_cos=max(c["mean_matched_cos"] for c in g_.values()),
            raw_dim_baseline_recovered_dir=sum(
                1 for m in greedy_match(keyp @ torch.eye(k))
                if m is not None and m[1] >= p["sae_cos_threshold"]),
            null=null_recovery(keyp, int(max(p["sae_proj_expansions"]) * k),
                               p["sae_null_repeats"], p["sae_cos_threshold"], seed + 900 + k))
    M["sae_grid_projected"] = proj_grids
    M["projected_summary"] = proj_summary
    M["projection_note"] = (
        "activation-space linear compression of the SAME constructed code (orthonormal-row P), "
        "NOT a functioning compressed transformer; k < K means the K key directions cannot be "
        "mutually orthogonal")
    stage_t["sae_projected"] = round(time.time() - ts, 2)
    grid_comp = None

    M["prior_lab_sae_scores"] = {
        "2026-07-25_sae-on-grokked-model": "best usable frac_pure 0.318 vs raw neurons 0.750",
        "2026-07-25_sae-on-merged-pairs": "8/8 merged pair-directions but 0/16 true features",
    }
    M["stage_seconds"] = stage_t
    M["wall_clock_s"] = round(time.time() - t0, 1)

    # ================================================================= chart
    fig, axm = plt.subplots(2, 2, figsize=(15, 9.5))
    varorder = ["tokens", "indices", "length", "opp_idx", "opp_idx_m1", "reverse"]

    ax = axm[0, 0]
    Z = np.array([[probe[v]["acc"][si] - probe[v]["majority"] for si in range(len(SITES))]
                  for v in varorder])
    im = ax.imshow(Z, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(SITES))); ax.set_xticklabels(SITES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(varorder))); ax.set_yticklabels(varorder, fontsize=9)
    for a_, v in enumerate(varorder):
        for si in range(len(SITES)):
            if lives[v][si]:
                ax.add_patch(plt.Rectangle((si - .5, a_ - .5), 1, 1, fill=False, ec="red", lw=2.2))
            if conf[v][si] == "FP":
                ax.text(si, a_, "FP", ha="center", va="center", color="w", fontsize=7, weight="bold")
    ax.set_title(f"probe acc - majority  (red box = variable provably LIVES here)\n"
                 f"localization acc {M['localization_accuracy']} | "
                 f"{FP} false positives over {FP + TN} non-live cells", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.03)

    ax = axm[0, 1]
    Zc = np.array([[causal[v][si]["damage"] for si in range(len(SITES))] for v in varorder])
    im2 = ax.imshow(Zc, cmap="magma", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(SITES))); ax.set_xticklabels(SITES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(varorder))); ax.set_yticklabels(varorder, fontsize=9)
    for a_, v in enumerate(varorder):
        for si in range(len(SITES)):
            if lives[v][si]:
                ax.add_patch(plt.Rectangle((si - .5, a_ - .5), 1, 1, fill=False, ec="red", lw=2.2))
            if lives[v][si] and probe[v]["acc"][si] > 0.99 and causal[v][si]["damage"] <= 0.01:
                ax.text(si, a_, "LIE", ha="center", va="center", color="cyan",
                        fontsize=7, weight="bold")
    ax.set_title("exact-seq accuracy LOST by erasing the variable's own dims\n"
                 f"red box = lives here; LIE = probe 1.00 but erasure costs nothing "
                 f"({len(dec_unused)}/{M['n_live_cells']} live cells)", fontsize=9)
    plt.colorbar(im2, ax=ax, fraction=0.03)

    ax = axm[1, 0]
    curves = [("tracr code (clean axes)", grid_clean, "tab:green", "clean",
               p["sae_expansions"], p["sae_seeds"]),
              ("independence control", grid_ctl, "tab:blue", "ctl",
               p["sae_control_expansions"], [0])]
    for k in p["proj_dims"]:
        curves.append((f"projected k={k} ({round(K / k, 2)}x superposed)", proj_grids[str(k)],
                       "tab:purple" if k == p["proj_dims"][0] else "tab:brown",
                       f"proj{k}", p["sae_proj_expansions"], [0]))
    for tag, grid, col, pre, exps, sds in curves:
        for field, ls in [("recovered_dir", "-"), ("recovered_behavioural", ":")]:
            ys_ = [max(grid[f"{pre}_x{ex}_l{l}_s{s}"][field]
                       for ex in exps for s in sds) / K for l in p["sae_lambdas"]]
            ax.plot(p["sae_lambdas"], ys_, marker="o", ls=ls, color=col, alpha=0.85, ms=4,
                    label=f"{tag}: {'direction' if field == 'recovered_dir' else 'behaviour'}")
    ax.axhline(0.318, ls="--", c="tab:orange", lw=1.2, label="lab prior: SAE on grokked (0.32 pure)")
    ax.set_xscale("log"); ax.set_xlabel("L1 coefficient (best over expansion / seed)")
    ax.set_ylabel("fraction of constructed features recovered")
    ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=5.5, loc="center left")
    ax.set_title(f"SAE recovery of the CONSTRUCTED basis (K={K} known directions)\n"
                 f"solid = decoder direction cos>={p['sae_cos_threshold']}, "
                 f"dotted = firing pattern F1>={p['sae_f1_threshold']}", fontsize=9)

    ax = axm[1, 1]
    ax.axis("off")
    lines = [
        "tracr path: INSTALLED from github (no hand-compiled fallback)",
        f"program reverse | {mcfg.num_layers} layers, d_model {D}, {M['n_params']} params, no LN",
        f"exact accuracy on ALL {Nseq} inputs: {M['exact_sequence_accuracy']:.4f}",
        f"torch port vs jax max|diff|: {maxdiff:.2e}",
        "",
        f"PROBE LOCALIZATION  acc {M['localization_accuracy']}  recall {M['localization_recall']}",
        f"  precision {M['localization_precision']}  FPR {M['localization_false_positive_rate']}",
        f"  ({TP} TP, {FP} FP, {FN} FN, {TN} TN over {len(varorder)}x{len(SITES)} cells)",
        "",
        f"SAE on the CLEAN constructed basis (axis-aligned, K={K}):",
        f"  direction cos>=0.95 : {M['sae_best_clean_recovered_dir_of_K']}"
        f"  (mean matched cos {M['sae_max_matched_cos_over_grid_clean']})",
        f"  firing pattern F1>=0.95 : {M['sae_best_behavioural_clean_of_K']}",
        f"  best cell x{M['sae_best_clean']['expansion']} lam={M['sae_best_clean']['lam']}"
        f"  FVU {M['sae_best_clean']['fvu']}  L0 {M['sae_best_clean']['l0']}"
        f" (true L0 {M['true_code_mean_L0']})",
        f"  random-direction null {M['sae_null_clean_mean_p95_max'][0]}"
        f" (p95 {M['sae_null_clean_mean_p95_max'][1]:.0f})",
    ]
    lines += [
        f"  answer key is REDUNDANT by construction: {M['answer_key_n_duplicate_pairs']} axis pairs",
        f"  at |corr|>=0.99, mean |offdiag corr| {M['answer_key_mean_abs_offdiag_corr']}",
        "",
        f"INDEPENDENCE CONTROL (same K axes, independent draws):",
        f"  direction {M['sae_control_best_recovered_dir_of_K']}, behaviour "
        f"{M['sae_control_best_recovered_behavioural_of_K']},"
        f" mean cos {M['sae_control_best_mean_matched_cos']}",
    ]
    for k in p["proj_dims"]:
        s_ = proj_summary[str(k)]
        lines.append(f"PROJECTED k={k} ({s_['superposition_ratio']}x, key |cos| "
                     f"{s_['key_mean_abs_cos']}): dir {s_['best_recovered_dir']}/{K},"
                     f" behav {s_['best_recovered_behavioural']}/{K}")
    lines += [
        "",
        f"model compression to R^k (frozen weights) NOT exact: " +
        ", ".join(f"k={c['k']}->{c['exact_sequence_accuracy']}"
                  for c in M["compression_sweep"].values()),
        "", f"wall clock {M['wall_clock_s']}s (CPU, 1 thread)"]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=9, family="monospace")

    fig.suptitle("Interpretability with the answer key: a Tracr-compiled `reverse` circuit",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(HERE / "chart.png", dpi=140)

    results = {
        "id": cfg.get("id", "unknown"),
        "git_commit": git_sha(),
        "seed": seed,
        "duration_sec": round(time.time() - t0, 2),
        "metrics": M,
        "env": env_info(),
        "status": "done",
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(json.dumps({k: M[k] for k in (
        "exact_sequence_accuracy", "torch_port_max_abs_diff_vs_jax", "birth_site",
        "localization_accuracy", "localization_TP", "localization_FP", "localization_FN",
        "n_alive_axes_clean", "sae_best_clean_recovered_dir_of_K",
        "sae_best_behavioural_clean_of_K", "sae_control_best_recovered_dir_of_K",
        "sae_control_best_recovered_behavioural_of_K", "projected_summary",
        "compression_sweep", "wall_clock_s", "stage_seconds") if k in M}, indent=2, default=str))


if __name__ == "__main__":
    main()
