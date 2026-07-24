"""Tsetlin Machine: does exact rule recovery break before accuracy under label noise?

Plants a 3-clause DNF over 10 boolean variables, trains a vanilla (Granmo 2018)
Tsetlin Machine implemented in pure numpy at increasing label-noise levels, and
measures (a) clean test accuracy, (b) functional accuracy over the full input
space, and (c) EXACT symbolic recovery of the planted clauses from the learned
positive-polarity clause set.

Deterministic, CPU-only, writes results.json and chart.png.

Usage:  python run.py
"""
import json, os, random, subprocess, sys, time
from itertools import product
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


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


# ----------------------------------------------------------------------------- data
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


# ----------------------------------------------------------------------------- vanilla Tsetlin Machine
class TsetlinMachine:
    """Vanilla two-class TM (Granmo 2018), pure numpy, per-sample updates.

    m clauses: even indices positive polarity (vote +1), odd negative (vote -1).
    2o literals per clause: [x_0..x_{o-1}, NOT x_0..NOT x_{o-1}].
    Automaton state in [1, 2N]; action = include literal iff state > N.
    """

    def __init__(self, o, n_clauses, T, s, n_states, rng):
        self.o, self.m, self.T, self.s, self.N = o, n_clauses, T, s, n_states
        self.rng = rng
        # start on the exclude boundary
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
            lits = np.concatenate([x, 1 - x])          # (2o,)
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
            lit1 = lits == 1  # (2o,) literal value 1

            # ---- Type I feedback (recognize: push clause toward matching y-class samples)
            t1 = type1 & sel
            if t1.any():
                r = self.rng.random((self.m, 2 * self.o))
                match = t1 & out          # clause fired
                nomatch = t1 & ~out       # clause silent
                # fired clause: reward 1-literals toward include (boosted true positive),
                # push 0-literals toward exclude with prob 1/s
                self.state[np.ix_(match, lit1)] += 1
                m0 = match[:, None] & (~lit1)[None, :] & (r < 1.0 / self.s)
                self.state[m0] -= 1
                # silent clause: everything drifts toward exclude with prob 1/s
                mN = nomatch[:, None] & (r < 1.0 / self.s)
                self.state[mN] -= 1

            # ---- Type II feedback (reject: make clause veto wrong-class samples)
            t2 = type2 & sel & out  # only fired clauses
            if t2.any():
                # include a currently-excluded 0-literal to break the match
                m2 = t2[:, None] & (~lit1)[None, :] & (~inc)
                self.state[m2] += 1

            np.clip(self.state, 1, N2, out=self.state)

    # ---- symbolic extraction
    def positive_clauses(self):
        """Return list of frozensets of (var, sign) for positive-polarity clauses."""
        inc = self._include()
        out = []
        for j in np.where(self.pos)[0]:
            lits = set()
            for k in np.where(inc[j])[0]:
                var, sign = (int(k), 1) if k < self.o else (int(k) - self.o, 0)
                lits.add((var, sign))
            out.append(frozenset(lits))
        return out


# ----------------------------------------------------------------------------- metrics
def clause_metrics(learned, planted):
    planted_sets = [frozenset((v, s) for v, s in c) for c in planted]
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


def main():
    cfg = load_config()
    P = cfg["params"]
    set_seeds(int(cfg.get("seed", 0)))
    t0 = time.time()

    o = P["n_vars"]
    dnf = [[tuple(lit) for lit in clause] for clause in P["planted_dnf"]]
    all_X = np.array(list(product([0, 1], repeat=o)), dtype=np.uint8)  # full input space
    all_y = dnf_label(all_X, dnf)

    rows = []
    for noise in P["noise_levels"]:
        for seed in P["seeds"]:
            rng = np.random.default_rng(1000 * seed + int(noise * 1000))
            Xtr, ytr = make_data(rng, P["n_train"], o, dnf, noise)
            Xte, yte = make_data(rng, P["n_test"], o, dnf, 0.0)  # clean test labels
            tm = TsetlinMachine(o, P["n_clauses"], P["T"], P["s"], P["n_states"], rng)
            for _ in range(P["epochs"]):
                perm = rng.permutation(len(Xtr))
                tm.fit_epoch(Xtr[perm], ytr[perm])
            test_acc = float((tm.predict(Xte) == yte).mean())
            func_acc = float((tm.predict(all_X) == all_y).mean())
            rec, prec, jac, distinct = clause_metrics(tm.positive_clauses(), dnf)
            rows.append({"noise": noise, "seed": seed, "test_acc": test_acc,
                         "functional_acc": func_acc, "exact_recovery": rec,
                         "clause_precision": prec, "mean_best_jaccard": jac,
                         "distinct_pos_clauses": [sorted(c) for c in distinct]})
            print(f"noise={noise:.2f} seed={seed}  test_acc={test_acc:.4f} "
                  f"func_acc={func_acc:.4f} recovery={rec:.2f} precision={prec:.2f} "
                  f"jaccard={jac:.2f}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- aggregate per noise level
    noises = P["noise_levels"]
    agg = {}
    for key in ("test_acc", "functional_acc", "exact_recovery", "clause_precision", "mean_best_jaccard"):
        agg[key] = {"mean": [], "min": [], "max": []}
        for nz in noises:
            vals = [r[key] for r in rows if r["noise"] == nz]
            agg[key]["mean"].append(float(np.mean(vals)))
            agg[key]["min"].append(float(np.min(vals)))
            agg[key]["max"].append(float(np.max(vals)))

    def first_noise_below(key, thresh):
        for nz, mv in zip(noises, agg[key]["mean"]):
            if mv < thresh:
                return nz
        return None

    recovery_break = first_noise_below("exact_recovery", 1.0)
    acc_break = first_noise_below("test_acc", 0.95)
    headline = (f"exact recovery first breaks at noise={recovery_break}, "
                f"test accuracy first drops below 0.95 at noise={acc_break}")

    # ---- chart
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {"exact_recovery": ("tab:red", "exact clause recovery (planted DNF)"),
              "clause_precision": ("tab:orange", "clause precision (learned = planted)"),
              "mean_best_jaccard": ("tab:purple", "mean best Jaccard (soft recovery)"),
              "test_acc": ("tab:blue", "clean test accuracy"),
              "functional_acc": ("tab:green", "functional accuracy (all 1024 inputs)")}
    for key, (color, label) in styles.items():
        ax.plot(noises, agg[key]["mean"], "-o", color=color, label=label, lw=2, ms=4)
        ax.fill_between(noises, agg[key]["min"], agg[key]["max"], color=color, alpha=0.12)
    ax.set_xlabel("training label-noise rate")
    ax.set_ylabel("metric (mean over 3 seeds, min-max band)")
    ax.set_title("Tsetlin Machine on a planted 3-clause DNF:\nsymbolic recovery vs accuracy under label noise")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "chart.png", dpi=150)

    results = {
        "id": cfg.get("id", "unknown"),
        "git_commit": git_sha(),
        "seed": int(cfg.get("seed", 0)),
        "duration_sec": round(time.time() - t0, 2),
        "metrics": {
            "noise_levels": noises,
            "aggregate": agg,
            "recovery_first_breaks_at_noise": recovery_break,
            "test_acc_first_below_095_at_noise": acc_break,
            "headline": headline,
            "per_run": rows,
        },
        "env": env_info(),
        "status": "done",
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(headline)


if __name__ == "__main__":
    main()
