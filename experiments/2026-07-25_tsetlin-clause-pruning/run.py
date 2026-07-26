"""Can validation-based clause pruning dig the planted DNF back out from under label noise?

Follow-up to experiments/2026-07-24_tsetlin-dnf-recovery, which found that label noise
BURIES the planted rules rather than erasing them: at eps=0.2 all 3 planted clauses are
still present verbatim (exact recall 1.0) but only ~49% of the distinct learned
positive-polarity clauses are planted clauses (clause precision 0.49); at eps=0.4
precision is 0.07.

Here we reuse that experiment's pure-numpy Tsetlin Machine and planted-DNF generator
verbatim (the TM class, the DNF label function and the clause metrics are copied so this
folder is self-contained and reproduces the prior run's numbers exactly), train the same
models, and then try to CLEAN the clause list post hoc:

  * hold out a validation set drawn from the SAME NOISY DISTRIBUTION (labels flipped at
    the same rate eps) -- we never assume access to clean labels;
  * score every learned clause by its individual validation precision
    P(y_val = polarity | clause fires) and its coverage;
  * sweep a pruning threshold tau (drop clauses below it) and, separately, a soft
    clause weighting; and
  * re-measure clause precision, exact clause recall, mean best Jaccard, clean test
    accuracy, functional accuracy over all 1024 inputs, and the accuracy of reading the
    surviving positive clauses off as a plain DNF.

Deterministic, CPU-only, writes results.json and chart.png.

Usage:  python run.py
"""
import json, os, random, subprocess, sys, time
from itertools import product
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "2026-07-24_tsetlin-dnf-recovery" / "results.json"


def set_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


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
    for mod in ("numpy", "matplotlib"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:
            pass
    return info


# ----------------------------------------------------------------- data (from 2026-07-24)
def dnf_label(X: np.ndarray, dnf) -> np.ndarray:
    """X: (n, o) in {0,1}. dnf: list of clauses, clause = list of (var, sign)."""
    y = np.zeros(len(X), dtype=np.int64)
    for clause in dnf:
        sat = np.ones(len(X), dtype=bool)
        for var, sign in clause:
            sat &= (X[:, var] == sign)
        y |= sat
    return y


def make_data(rng: np.random.Generator, n: int, o: int, dnf, noise: float):
    X = rng.integers(0, 2, size=(n, o)).astype(np.uint8)
    y = dnf_label(X, dnf)
    if noise > 0:
        flip = rng.random(n) < noise
        y = np.where(flip, 1 - y, y)
    return X, y


# ------------------------------------------- vanilla Tsetlin Machine (from 2026-07-24)
class TsetlinMachine:
    """Vanilla two-class TM (Granmo 2018), pure numpy, per-sample updates.

    m clauses: even indices positive polarity (vote +1), odd negative (vote -1).
    2o literals per clause: [x_0..x_{o-1}, NOT x_0..NOT x_{o-1}].
    Automaton state in [1, 2N]; action = include literal iff state > N.
    """

    def __init__(self, o, n_clauses, T, s, n_states, rng):
        self.o, self.m, self.T, self.s, self.N = o, n_clauses, T, s, n_states
        self.rng = rng
        self.state = np.full((self.m, 2 * o), self.N, dtype=np.int32)
        self.pos = (np.arange(self.m) % 2 == 0)  # polarity mask

    def _include(self):
        return self.state > self.N

    def _clause_out(self, lits, training):
        inc = self._include()
        violated = (inc & (lits == 0)).any(axis=1)
        out = ~violated
        if not training:  # empty clause votes 0 at inference
            out &= inc.any(axis=1)
        return out

    def predict(self, X):
        preds = np.empty(len(X), dtype=np.int64)
        for i, x in enumerate(X):
            lits = np.concatenate([x, 1 - x])
            out = self._clause_out(lits, training=False)
            v = int(out[self.pos].sum()) - int(out[~self.pos].sum())
            preds[i] = 1 if v > 0 else 0
        return preds

    def fit_epoch(self, X, y):
        N2 = 2 * self.N
        for x, yi in zip(X, y):
            lits = np.concatenate([x, 1 - x])            # (2o,)
            out = self._clause_out(lits, training=True)  # (m,)
            v = int(out[self.pos].sum()) - int(out[~self.pos].sum())
            vc = max(-self.T, min(self.T, v))
            if yi == 1:
                p = (self.T - vc) / (2 * self.T)
                type1, type2 = self.pos, ~self.pos
            else:
                p = (self.T + vc) / (2 * self.T)
                type1, type2 = ~self.pos, self.pos
            sel = self.rng.random(self.m) < p

            inc = self._include()
            lit1 = lits == 1

            # ---- Type I feedback
            t1 = type1 & sel
            if t1.any():
                r = self.rng.random((self.m, 2 * self.o))
                match = t1 & out
                nomatch = t1 & ~out
                self.state[np.ix_(match, lit1)] += 1
                m0 = match[:, None] & (~lit1)[None, :] & (r < 1.0 / self.s)
                self.state[m0] -= 1
                mN = nomatch[:, None] & (r < 1.0 / self.s)
                self.state[mN] -= 1

            # ---- Type II feedback
            t2 = type2 & sel & out
            if t2.any():
                m2 = t2[:, None] & (~lit1)[None, :] & (~inc)
                self.state[m2] += 1

            np.clip(self.state, 1, N2, out=self.state)

    # ---- symbolic extraction
    def clause_sets(self):
        """List (len m) of frozensets of (var, sign) -- ALL clauses, both polarities."""
        inc = self._include()
        out = []
        for j in range(self.m):
            lits = set()
            for k in np.where(inc[j])[0]:
                var, sign = (int(k), 1) if k < self.o else (int(k) - self.o, 0)
                lits.add((var, sign))
            out.append(frozenset(lits))
        return out

    def fire_matrix(self, X):
        """(m, n) bool: clause j fires on sample i (inference semantics: empty -> never)."""
        lits = np.concatenate([X, 1 - X], axis=1).astype(np.int16)  # (n, 2o)
        inc = self._include().astype(np.int16)                      # (m, 2o)
        viol = inc @ (1 - lits).T                                   # (m, n) # violated literals
        nonempty = inc.any(axis=1)
        return (viol == 0) & nonempty[:, None]


# ----------------------------------------------------------------- metrics
def clause_metrics(learned, planted_sets):
    """learned: list of frozensets (positive-polarity clauses that SURVIVED).
    Returns exact recall, clause precision, mean best Jaccard, distinct sorted list."""
    learned_nonempty = [c for c in learned if c]
    distinct = set(learned_nonempty)
    recovered = sum(1 for c in planted_sets if c in distinct)
    exact_recovery = recovered / len(planted_sets)
    precision = (sum(1 for c in distinct if c in planted_sets) / len(distinct)) if distinct else 0.0
    jac = []
    for c in planted_sets:
        best = 0.0
        for l in learned_nonempty:
            j = len(c & l) / len(c | l)
            best = max(best, j)
        jac.append(best)
    return exact_recovery, precision, float(np.mean(jac)), sorted(distinct, key=sorted)


def wilson(k, n, z):
    """Vectorised Wilson score interval for a binomial proportion. n==0 -> (0, 1)."""
    k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
    safe = np.maximum(n, 1.0)
    p = k / safe
    d = 1.0 + z * z / safe
    centre = (p + z * z / (2 * safe)) / d
    half = z * np.sqrt(p * (1 - p) / safe + z * z / (4 * safe * safe)) / d
    lo, hi = centre - half, centre + half
    return np.where(n > 0, lo, 0.0), np.where(n > 0, hi, 1.0)


def vote_acc(fire, pos_mask, keep, y, weights=None):
    """Accuracy of the (possibly pruned / weighted) TM vote. fire: (m, n) bool."""
    w = np.ones(len(pos_mask)) if weights is None else np.asarray(weights, dtype=float)
    w = w * keep
    sgn = np.where(pos_mask, 1.0, -1.0) * w
    v = sgn @ fire
    return float(((v > 0).astype(np.int64) == y).mean())


def dnf_readoff_acc(fire_all, pos_mask, keep, y_all):
    """Accuracy of reading the surviving POSITIVE clauses off as a plain DNF."""
    sel = keep & pos_mask
    if not sel.any():
        pred = np.zeros(fire_all.shape[1], dtype=np.int64)
    else:
        pred = fire_all[sel].any(axis=0).astype(np.int64)
    return float((pred == y_all).mean())


def main():
    cfg = load_config()
    P = cfg["params"]
    set_seeds(int(cfg.get("seed", 0)))
    t0 = time.time()

    o = P["n_vars"]
    dnf = [[tuple(lit) for lit in clause] for clause in P["planted_dnf"]]
    planted_sets = [frozenset(c) for c in dnf]
    all_X = np.array(list(product([0, 1], repeat=o)), dtype=np.uint8)
    all_y = dnf_label(all_X, dnf)

    taus = np.linspace(0.0, 1.0, int(P["tau_grid_points"]))
    min_fires = list(P["min_fire_grid"])
    val_sizes = list(P["val_sizes"])
    VS_H, MF_H = int(P["val_size_headline"]), int(P["min_fire_headline"])
    alpha = float(P["auto_alpha"])
    zval, mcf, mcfloor = float(P["wilson_z"]), float(P["min_cov_frac"]), int(P["min_cov_floor"])

    rows = []
    for noise in P["noise_levels"]:
        for seed in P["seeds"]:
            # --- identical rng stream to 2026-07-24 so the baselines reproduce exactly
            rng = np.random.default_rng(1000 * seed + int(noise * 1000))
            Xtr, ytr = make_data(rng, P["n_train"], o, dnf, noise)
            Xte, yte = make_data(rng, P["n_test"], o, dnf, 0.0)  # CLEAN test labels (measurement only)
            tm = TsetlinMachine(o, P["n_clauses"], P["T"], P["s"], P["n_states"], rng)
            for _ in range(P["epochs"]):
                perm = rng.permutation(len(Xtr))
                tm.fit_epoch(Xtr[perm], ytr[perm])

            # --- validation set: separate rng stream, SAME noisy distribution as training
            rng_val = np.random.default_rng(900000 + 1000 * seed + int(noise * 1000))
            Xva_big, yva_big = make_data(rng_val, max(val_sizes), o, dnf, noise)

            sets = tm.clause_sets()
            pos_mask = tm.pos.copy()
            f_te, f_all = tm.fire_matrix(Xte), tm.fire_matrix(all_X)
            f_va_big = tm.fire_matrix(Xva_big)
            nonempty = np.array([len(c) > 0 for c in sets])

            keep_all = nonempty.copy()  # baseline: every non-empty clause kept
            base_rec, base_prec, base_jac, base_distinct = clause_metrics(
                [c for j, c in enumerate(sets) if pos_mask[j]], planted_sets)
            base = {
                "exact_recall": base_rec, "clause_precision": base_prec,
                "mean_best_jaccard": base_jac,
                "n_distinct_pos": len(base_distinct),
                "test_acc": vote_acc(f_te, pos_mask, keep_all, yte),
                "functional_acc": vote_acc(f_all, pos_mask, keep_all, all_y),
                "dnf_readoff_acc": dnf_readoff_acc(f_all, pos_mask, keep_all, all_y),
                "distinct_pos_clauses": [sorted(c) for c in base_distinct],
            }

            # a clause is SOUND if every input it fires on truly has y = its polarity
            sound = np.array([
                bool((all_y[f_all[j]] == (1 if pos_mask[j] else 0)).all()) if f_all[j].any() else False
                for j in range(len(sets))])

            per_val = {}
            for nv in val_sizes:
                yva = yva_big[:nv]
                f_va = f_va_big[:, :nv]
                fires = f_va.sum(axis=1)                       # per-clause validation firings
                hit = np.where(pos_mask[:, None], yva[None, :] == 1, yva[None, :] == 0)
                prec = np.where(fires > 0, (f_va & hit).sum(axis=1) / np.maximum(fires, 1), 0.0)
                p0 = float((yva == 1).mean())

                # the full threshold sweep is only stored/analysed at the headline validation size
                sweep = {"tau": taus.tolist()}
                for mf in (min_fires if nv == VS_H else []):
                    scoreable = nonempty & (fires >= mf)
                    curves = {k: [] for k in ("clause_precision", "exact_recall", "mean_best_jaccard",
                                              "n_distinct_pos", "n_kept_pos", "test_acc",
                                              "functional_acc", "dnf_readoff_acc", "frac_sound_kept")}
                    for tau in taus:
                        keep = scoreable & (prec >= tau)
                        kp = keep & pos_mask
                        r, pr, jc, dis = clause_metrics(
                            [c for j, c in enumerate(sets) if kp[j]], planted_sets)
                        curves["clause_precision"].append(pr)
                        curves["exact_recall"].append(r)
                        curves["mean_best_jaccard"].append(jc)
                        curves["n_distinct_pos"].append(len(dis))
                        curves["n_kept_pos"].append(int(kp.sum()))
                        curves["test_acc"].append(vote_acc(f_te, pos_mask, keep, yte))
                        curves["functional_acc"].append(vote_acc(f_all, pos_mask, keep, all_y))
                        curves["dnf_readoff_acc"].append(dnf_readoff_acc(f_all, pos_mask, keep, all_y))
                        curves["frac_sound_kept"].append(float(sound[kp].mean()) if kp.any() else 0.0)
                    sweep[f"min_fire_{mf}"] = {k: [round(float(v), 6) for v in vs]
                                               for k, vs in curves.items()}

                # ---- oracle-free rule: tau = alpha * (max scoreable precision), per polarity
                scoreable = nonempty & (fires >= MF_H)
                keep = np.zeros(len(sets), dtype=bool)
                for polarity in (True, False):
                    sc = scoreable & (pos_mask == polarity)
                    if sc.any():
                        keep |= sc & (prec >= alpha * prec[sc].max())
                kp = keep & pos_mask
                r, pr, jc, dis = clause_metrics([c for j, c in enumerate(sets) if kp[j]], planted_sets)
                auto = {"exact_recall": r, "clause_precision": pr, "mean_best_jaccard": jc,
                        "n_distinct_pos": len(dis), "n_kept_pos": int(kp.sum()),
                        "test_acc": vote_acc(f_te, pos_mask, keep, yte),
                        "functional_acc": vote_acc(f_all, pos_mask, keep, all_y),
                        "dnf_readoff_acc": dnf_readoff_acc(f_all, pos_mask, keep, all_y),
                        "frac_sound_kept": float(sound[kp].mean()) if kp.any() else 0.0,
                        "distinct_pos_clauses": [sorted(c) for c in dis]}

                # ---- statistically corrected oracle-free rule (Wilson):
                # the achievable precision ceiling is ~1-eps and is UNKNOWN; estimate it as the
                # largest Wilson lower bound over well-covered clauses, then keep every clause
                # whose Wilson upper bound still reaches that ceiling.
                min_cov_n = max(mcfloor, int(round(mcf * nv)))
                lo, hi = wilson((f_va & hit).sum(axis=1), fires, zval)
                covered = nonempty & (fires >= min_cov_n)
                keep_w = np.zeros(len(sets), dtype=bool)
                ceilings = {}
                for polarity in (True, False):
                    sc = covered & (pos_mask == polarity)
                    if sc.any():
                        ceil_p = float(lo[sc].max())
                        ceilings["pos" if polarity else "neg"] = ceil_p
                        keep_w |= sc & (hi >= ceil_p)
                kw = keep_w & pos_mask
                r2, pr2, jc2, dis2 = clause_metrics(
                    [c for j, c in enumerate(sets) if kw[j]], planted_sets)
                wilson_rule = {
                    "exact_recall": r2, "clause_precision": pr2, "mean_best_jaccard": jc2,
                    "n_distinct_pos": len(dis2), "n_kept_pos": int(kw.sum()),
                    "test_acc": vote_acc(f_te, pos_mask, keep_w, yte),
                    "functional_acc": vote_acc(f_all, pos_mask, keep_w, all_y),
                    "dnf_readoff_acc": dnf_readoff_acc(f_all, pos_mask, keep_w, all_y),
                    "frac_sound_kept": float(sound[kw].mean()) if kw.any() else 0.0,
                    "min_cov_n": min_cov_n,
                    "estimated_precision_ceiling": ceilings,
                    "implied_eps_hat": (1.0 - ceilings["pos"]) if "pos" in ceilings else None,
                    "distinct_pos_clauses": [sorted(c) for c in dis2],
                }

                # ---- soft weighting instead of pruning
                base_rate = np.where(pos_mask, p0, 1.0 - p0)
                w = np.clip((prec - base_rate) / np.maximum(1.0 - base_rate, 1e-9), 0.0, 1.0)
                w = w * (nonempty & (fires >= MF_H))
                is_planted = np.array([c in planted_sets for c in sets])
                wp = w * pos_mask
                pn = pos_mask & nonempty
                weighting = {
                    "test_acc": vote_acc(f_te, pos_mask, np.ones(len(sets), dtype=bool), yte, weights=w),
                    "functional_acc": vote_acc(f_all, pos_mask, np.ones(len(sets), dtype=bool),
                                               all_y, weights=w),
                    "weight_mass_on_planted": float(wp[is_planted].sum() / wp.sum()) if wp.sum() > 0 else 0.0,
                    "uniform_mass_on_planted": float((pn & is_planted).sum() / max(pn.sum(), 1)),
                    "mean_weight_planted": float(w[pn & is_planted].mean()) if (pn & is_planted).any() else None,
                    "mean_weight_spurious": float(w[pn & ~is_planted].mean()) if (pn & ~is_planted).any() else None,
                }

                per_val[str(nv)] = {"sweep": sweep, "auto_rule": auto, "wilson_rule": wilson_rule,
                                    "weighting": weighting, "val_pos_rate": p0}

            # per-clause diagnostic table at the headline validation size
            yva = yva_big[:VS_H]
            f_va = f_va_big[:, :VS_H]
            fires = f_va.sum(axis=1)
            hit = np.where(pos_mask[:, None], yva[None, :] == 1, yva[None, :] == 0)
            prec_h = np.where(fires > 0, (f_va & hit).sum(axis=1) / np.maximum(fires, 1), 0.0)
            rows.append({
                "noise": noise, "seed": seed, "baseline": base, "per_val": per_val,
                "clause_table": {
                    "polarity": ["+" if b else "-" for b in pos_mask.tolist()],
                    "clause": [sorted(c) for c in sets],
                    "is_planted": [bool(c in planted_sets) for c in sets],
                    "sound": sound.tolist(),
                    "val_fires": fires.tolist(),
                    "val_precision": [round(float(v), 4) for v in prec_h],
                },
            })
            a = per_val[str(VS_H)]["auto_rule"]
            wr = per_val[str(VS_H)]["wilson_rule"]
            print(f"eps={noise:.2f} seed={seed} | base prec={base['clause_precision']:.2f} "
                  f"rec={base['exact_recall']:.2f} acc={base['test_acc']:.3f} | naive-max "
                  f"prec={a['clause_precision']:.2f} rec={a['exact_recall']:.2f} | wilson "
                  f"prec={wr['clause_precision']:.2f} rec={wr['exact_recall']:.2f} "
                  f"acc={wr['test_acc']:.3f} kept={wr['n_kept_pos']} "
                  f"eps_hat={wr['implied_eps_hat']} ({time.time()-t0:.0f}s)", flush=True)

    # ---------------------------------------------------------------- aggregate
    noises = P["noise_levels"]

    def agg_over_seeds(getter):
        out = {"mean": [], "min": [], "max": []}
        for nz in noises:
            vals = [getter(r) for r in rows if r["noise"] == nz]
            vals = [v for v in vals if v is not None]
            if not vals:
                out["mean"].append(None); out["min"].append(None); out["max"].append(None)
                continue
            out["mean"].append(float(np.mean(vals)))
            out["min"].append(float(np.min(vals)))
            out["max"].append(float(np.max(vals)))
        return out

    aggregate = {"baseline": {}, "auto_rule": {}, "wilson_rule": {}, "weighting": {}}
    for k in ("exact_recall", "clause_precision", "mean_best_jaccard", "test_acc",
              "functional_acc", "dnf_readoff_acc", "n_distinct_pos"):
        aggregate["baseline"][k] = agg_over_seeds(lambda r, k=k: r["baseline"][k])
    for k in ("exact_recall", "clause_precision", "mean_best_jaccard", "test_acc",
              "functional_acc", "dnf_readoff_acc", "n_distinct_pos", "n_kept_pos", "frac_sound_kept"):
        aggregate["auto_rule"][k] = agg_over_seeds(
            lambda r, k=k: r["per_val"][str(VS_H)]["auto_rule"][k])
    for k in ("exact_recall", "clause_precision", "mean_best_jaccard", "test_acc",
              "functional_acc", "dnf_readoff_acc", "n_distinct_pos", "n_kept_pos",
              "frac_sound_kept", "implied_eps_hat"):
        aggregate["wilson_rule"][k] = agg_over_seeds(
            lambda r, k=k: r["per_val"][str(VS_H)]["wilson_rule"][k])
    for k in ("test_acc", "functional_acc", "weight_mass_on_planted", "uniform_mass_on_planted"):
        aggregate["weighting"][k] = agg_over_seeds(
            lambda r, k=k: r["per_val"][str(VS_H)]["weighting"][k])

    # threshold sweep, averaged over seeds, at the headline val size / min_fire
    sweep_agg = {}
    for nz in noises:
        sel = [r for r in rows if r["noise"] == nz]
        d = {}
        for k in ("clause_precision", "exact_recall", "mean_best_jaccard", "test_acc",
                  "functional_acc", "dnf_readoff_acc", "n_distinct_pos", "frac_sound_kept"):
            arr = np.array([r["per_val"][str(VS_H)]["sweep"][f"min_fire_{MF_H}"][k] for r in sel],
                           dtype=float)
            d[k] = arr.mean(axis=0).tolist()
        cp, er = np.array(d["clause_precision"]), np.array(d["exact_recall"])
        d["best_clause_precision"] = float(cp.max())
        d["tau_at_best_clause_precision"] = float(taus[int(cp.argmax())])
        ok = np.where(er >= er[0] - 1e-9)[0]  # thresholds that lose no planted clause
        d["best_clause_precision_no_recall_loss"] = float(cp[ok].max()) if len(ok) else 0.0
        d["tau_no_recall_loss"] = float(taus[ok[int(cp[ok].argmax())]]) if len(ok) else None
        sweep_agg[str(nz)] = d

    # val-size sensitivity of the auto rule
    val_size_sens = {}
    for nv in val_sizes:
        val_size_sens[str(nv)] = {
            rule: {k: [float(np.mean([r["per_val"][str(nv)][rule][k]
                                      for r in rows if r["noise"] == nz])) for nz in noises]
                   for k in ("clause_precision", "exact_recall", "test_acc", "n_kept_pos")}
            for rule in ("auto_rule", "wilson_rule")}

    # ---------------------------------------------------------------- reproduction check
    repro = {"available": False}
    if PRIOR.exists():
        try:
            prev = json.load(open(PRIOR))["metrics"]["per_run"]
            key = {(round(p["noise"], 4), p["seed"]): p for p in prev}
            diffs = []
            for r in rows:
                p = key.get((round(r["noise"], 4), r["seed"]))
                if p is None:
                    continue
                diffs.append(max(abs(p["clause_precision"] - r["baseline"]["clause_precision"]),
                                 abs(p["exact_recovery"] - r["baseline"]["exact_recall"]),
                                 abs(p["test_acc"] - r["baseline"]["test_acc"])))
            repro = {"available": True, "n_matched": len(diffs),
                     "max_abs_diff_vs_2026_07_24": float(max(diffs)) if diffs else None,
                     "exact": bool(diffs and max(diffs) < 1e-9)}
        except Exception as e:  # pragma: no cover
            repro = {"available": False, "error": str(e)}

    idx = {str(nz): i for i, nz in enumerate(noises)}

    def at(block, key, nz):
        return round(aggregate[block][key]["mean"][idx[str(nz)]], 3)

    headline = (
        f"Wilson rule (no clean labels): eps=0.2 clause precision "
        f"{at('baseline','clause_precision',0.2)} -> {at('wilson_rule','clause_precision',0.2)} "
        f"with exact recall {at('baseline','exact_recall',0.2)} -> "
        f"{at('wilson_rule','exact_recall',0.2)} and test acc "
        f"{at('baseline','test_acc',0.2)} -> {at('wilson_rule','test_acc',0.2)}; "
        f"eps=0.3 precision {at('baseline','clause_precision',0.3)} -> "
        f"{at('wilson_rule','clause_precision',0.3)} with recall "
        f"{at('baseline','exact_recall',0.3)} -> {at('wilson_rule','exact_recall',0.3)}; "
        f"eps=0.4 precision {at('baseline','clause_precision',0.4)} -> "
        f"{at('wilson_rule','clause_precision',0.4)} (recall "
        f"{at('baseline','exact_recall',0.4)} -> {at('wilson_rule','exact_recall',0.4)}). "
        f"Naive max-relative rule over-prunes: eps=0.2 recall "
        f"{at('baseline','exact_recall',0.2)} -> {at('auto_rule','exact_recall',0.2)}. "
        f"Best swept tau with no recall loss: eps=0.2 precision "
        f"{round(sweep_agg['0.2']['best_clause_precision_no_recall_loss'], 3)} at tau="
        f"{sweep_agg['0.2']['tau_no_recall_loss']}, eps=0.3 "
        f"{round(sweep_agg['0.3']['best_clause_precision_no_recall_loss'], 3)} at tau="
        f"{sweep_agg['0.3']['tau_no_recall_loss']}")

    # ---------------------------------------------------------------- chart
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {0.0: "tab:green", 0.2: "tab:blue", 0.3: "tab:orange", 0.4: "tab:red"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    ax = axes[0]
    for nz in noises:
        d = sweep_agg[str(nz)]
        c = colors.get(nz, "gray")
        ax.plot(taus, d["clause_precision"], "-", color=c, lw=2, label=f"eps={nz}")
        ax.plot(taus, d["exact_recall"], "--", color=c, lw=1.5, alpha=0.8)
        ax.axvline(1.0 - nz, color=c, ls=":", lw=1, alpha=0.55)  # theoretical ceiling 1-eps
    ax.annotate("dotted verticals: 1-eps\n(precision a SOUND clause\ncan reach on noisy labels)",
                xy=(0.02, 0.30), xycoords="axes fraction", fontsize=7)
    ax.set_xlabel("pruning threshold tau (validation clause precision)")
    ax.set_ylabel("clause precision (solid) / exact recall (dashed)")
    ax.set_title("Pruning threshold sweep\n(noisy validation set, n=%d, min %d firings)" % (VS_H, MF_H))
    ax.set_ylim(-0.03, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    for nz in noises:
        d = sweep_agg[str(nz)]
        c = colors.get(nz, "gray")
        ax.plot(taus, d["test_acc"], "-", color=c, lw=2, label=f"eps={nz}")
        ax.plot(taus, d["dnf_readoff_acc"], ":", color=c, lw=1.8, alpha=0.9)
    ax.set_xlabel("pruning threshold tau")
    ax.set_ylabel("clean test acc (solid) / DNF read-off acc (dotted)")
    ax.set_title("Accuracy cost of pruning")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[2]
    x = np.arange(len(noises))
    w = 0.26
    ax.bar(x - w, aggregate["baseline"]["clause_precision"]["mean"], w,
           label="precision: no pruning", color="tab:gray")
    ax.bar(x, aggregate["auto_rule"]["clause_precision"]["mean"], w,
           label="precision: naive max-relative rule", color="tab:purple", alpha=0.75)
    ax.bar(x + w, aggregate["wilson_rule"]["clause_precision"]["mean"], w,
           label="precision: Wilson rule", color="tab:blue")
    ax.plot(x, aggregate["baseline"]["exact_recall"]["mean"], "kv:", alpha=0.55,
            label="exact recall (before pruning)")
    ax.plot(x + w, aggregate["wilson_rule"]["exact_recall"]["mean"], "k^--",
            label="exact recall (Wilson rule)")
    ax.plot(x, aggregate["auto_rule"]["exact_recall"]["mean"], "x", color="tab:purple", ms=9,
            label="exact recall (naive rule)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"eps={n}" for n in noises])
    ax.set_ylabel("metric (mean over 3 seeds)")
    ax.set_title("Two oracle-free pruning rules\n(no clean labels anywhere)")
    ax.set_ylim(0, 1.14)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.92)

    fig.suptitle("Validation-based clause pruning on a noise-buried Tsetlin rule set "
                 "(planted 3-clause DNF, 10 vars)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(HERE / "chart.png", dpi=150)

    results = {
        "id": cfg.get("id", "unknown"),
        "git_commit": git_sha(),
        "seed": int(cfg.get("seed", 0)),
        "duration_sec": round(time.time() - t0, 2),
        "notes": ("Validation labels are drawn from the SAME noisy distribution as training "
                  "(flip rate eps); clean labels are never used for pruning, only for measuring "
                  "test/functional accuracy afterwards."),
        "metrics": {
            "noise_levels": noises,
            "tau_grid": taus.tolist(),
            "val_size_headline": VS_H,
            "min_fire_headline": MF_H,
            "auto_alpha": alpha,
            "aggregate": aggregate,
            "sweep_aggregate": sweep_agg,
            "val_size_sensitivity": val_size_sens,
            "reproduction_check_vs_2026_07_24": repro,
            "headline": headline,
            "per_run": rows,
        },
        "env": env_info(),
        "status": "done",
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + headline)
    print("repro check vs 2026-07-24:", repro)


if __name__ == "__main__":
    main()
