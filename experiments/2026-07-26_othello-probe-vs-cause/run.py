"""Mini-Othello world model: probed vs caused, per board cell.

Pipeline
  1. Hand-written vectorised numpy Reversi engine (5x5), self-tested against hand-checked
     positions and the canonical 8x8 opening.
  2. ~150k random-legal-move games -> a tiny GPT (205k params, 4 layers, d=64) trained to
     predict the next move.  Verification: top-1 legality rate + legal-set F1.
  3. Linear probes per board cell for {empty, mine, yours} on the residual stream at every layer.
  4. Causal test: flip a cell's colour IN THE RESIDUAL along the probe's own direction and ask
     whether the model's legal-move prediction changes the way the EDITED board implies.
  5. Deliverable: per-cell probe accuracy vs per-cell causal-edit success.

CPU only, single thread, seeded.  Usage: python run.py
"""
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
FAST = os.environ.get("OTHELLO_FAST") == "1"

# ----------------------------------------------------------------------------
# 1. Vectorised mini-Othello engine
# ----------------------------------------------------------------------------
DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
OFF = 2  # off-board sentinel (never equals +1/-1/0)


def shift(x, dr, dc, fill):
    """out[..., r, c] = x[..., r-dr, c-dc]; out-of-range filled with `fill`."""
    out = np.full_like(x, fill)
    N = x.shape[-1]
    rs = slice(max(0, -dr), N - max(0, dr))
    rd = slice(max(0, dr), N - max(0, -dr))
    cs = slice(max(0, -dc), N - max(0, dc))
    cd = slice(max(0, dc), N - max(0, -dc))
    out[..., rd, cd] = x[..., rs, cs]
    return out


def legal_mask(board, player):
    """board (G,N,N) int8 in {-1,0,1}, player (G,) int8 -> bool (G,N,N)."""
    G, N, _ = board.shape
    p = player[:, None, None]
    legal = np.zeros((G, N, N), dtype=bool)
    for dr, dc in DIRS:
        nb = shift(board, dr, dc, OFF)
        run = nb == -p                       # step 1 must land on an opponent disc
        found = np.zeros((G, N, N), dtype=bool)
        for k in range(2, N):
            nb = shift(board, dr * k, dc * k, OFF)
            found |= run & (nb == p)         # run closed by one of our own discs
            run &= nb == -p
        legal |= found
    legal &= board == 0
    return legal


def legal_mask_chunked(board, player, chunk=40000):
    out = np.zeros(board.shape, dtype=bool)
    for i in range(0, board.shape[0], chunk):
        out[i:i + chunk] = legal_mask(board[i:i + chunk], player[i:i + chunk])
    return out


def apply_move(board, player, move, active):
    """In place. move (G,) flat cell index, active (G,) bool."""
    G, N, _ = board.shape
    onehot = np.zeros((G, N * N), dtype=bool)
    idx = np.where(active)[0]
    onehot[idx, move[idx]] = True
    onehot = onehot.reshape(G, N, N)
    p = player[:, None, None]
    flip = np.zeros((G, N, N), dtype=bool)
    for dr, dc in DIRS:
        run = np.zeros((G, N, N), dtype=bool)
        cur = shift(onehot, -dr, -dc, False)      # cell one step away from the move
        alive = np.ones((G, 1, 1), dtype=bool)
        for _ in range(1, N):
            v_opp = (cur & (board == -p)).any(axis=(1, 2))[:, None, None]
            v_own = (cur & (board == p)).any(axis=(1, 2))[:, None, None]
            flip |= run & v_own & alive           # opponent run closed by our disc -> flip it
            alive = alive & v_opp
            run = (run | cur) & alive
            if not alive.any():
                break
            cur = shift(cur, -dr, -dc, False)
    board[:] = np.where(onehot | flip, np.broadcast_to(p, board.shape), board)
    return board


def start_board(N):
    b = np.zeros((N, N), dtype=np.int8)
    r = c = (N - 1) // 2 if N % 2 == 0 else 1
    b[r, c] = -1
    b[r, c + 1] = 1
    b[r + 1, c] = 1
    b[r + 1, c + 1] = -1
    return b


def engine_selftest():
    """Hand-checked positions. Returns a dict of pass/fail flags."""
    out = {}
    b4 = start_board(4)
    lm = legal_mask(b4[None], np.array([1], np.int8))[0]
    out["4x4_opening_legal_moves"] = sorted(np.flatnonzero(lm.ravel()).tolist()) == [1, 4, 11, 14]
    b8 = start_board(8)
    lm = legal_mask(b8[None], np.array([1], np.int8))[0]
    out["8x8_canonical_opening_D3_C4_F5_E6"] = (
        sorted(np.flatnonzero(lm.ravel()).tolist()) == [19, 26, 37, 44])
    bb = b4[None].copy()
    apply_move(bb, np.array([1], np.int8), np.array([1]), np.array([True]))
    out["4x4_single_flip"] = bb[0].ravel().tolist() == [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, -1, 0, 0, 0, 0, 0]
    c = np.zeros((4, 4), np.int8)
    c[0, 0], c[0, 1], c[0, 2] = 1, -1, -1
    cc = c[None].copy()
    apply_move(cc, np.array([1], np.int8), np.array([3]), np.array([True]))
    out["4x4_two_disc_line_flip"] = cc[0, 0].tolist() == [1, 1, 1, 1]
    b5 = start_board(5)
    lm = legal_mask(b5[None], np.array([1], np.int8))[0]
    # start:  .....   black legal = bracket a white through a black:
    #         .WB..   (0,1)->(1,1)W->(2,1)B = 1 ; (1,0)->(1,1)W->(1,2)B = 5
    #         .BW..   (2,3)->(2,2)W->(2,1)B = 13 ; (3,2)->(2,2)W->(1,2)B = 17
    out["5x5_opening_legal_moves"] = sorted(np.flatnonzero(lm.ravel()).tolist()) == [1, 5, 13, 17]
    # no legal move on an empty board
    out["empty_board_no_moves"] = not legal_mask(np.zeros((1, 5, 5), np.int8),
                                                 np.array([1], np.int8)).any()
    return out


def gen_games(N, G, rng, max_len):
    """Uniform-random legal self play (with passes). Returns
    moves (G,T) int64 (-1 pad), boards (G,T,N,N) int8 board BEFORE move t,
    players (G,T) int8 player to move at t, lengths (G,), n_pass (G,)."""
    board = np.tile(start_board(N), (G, 1, 1))
    player = np.ones(G, np.int8)
    alive = np.ones(G, bool)
    moves = np.full((G, max_len), -1, np.int64)
    boards = np.zeros((G, max_len, N, N), np.int8)
    players = np.zeros((G, max_len), np.int8)
    n_pass = np.zeros(G, np.int64)
    lengths = np.zeros(G, np.int64)
    for t in range(max_len):
        lm = legal_mask(board, player)
        has = lm.any(axis=(1, 2))
        need_pass = alive & ~has
        if need_pass.any():                     # no move -> pass to the opponent
            player = np.where(need_pass, -player, player).astype(np.int8)
            lm2 = legal_mask(board, player)
            has2 = lm2.any(axis=(1, 2))
            lm = np.where(need_pass[:, None, None], lm2, lm)
            has = np.where(need_pass, has2, has)
            n_pass += need_pass & has2
            alive &= ~(need_pass & ~has2)       # neither side can move -> game over
        alive &= has
        boards[:, t] = board
        players[:, t] = player
        if not alive.any():
            break
        scores = lm.reshape(G, -1).astype(np.float64) * rng.random((G, N * N))
        mv = scores.argmax(axis=1)
        moves[alive, t] = mv[alive]
        lengths += alive
        apply_move(board, player, mv, alive)
        player = (-player).astype(np.int8)
    return moves, boards, players, lengths, n_pass


# ----------------------------------------------------------------------------
# 2. Tiny GPT
# ----------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        sh = lambda z: z.view(B, T, self.h, D // self.h).transpose(1, 2)
        a = F.scaled_dot_product_attention(sh(q), sh(k), sh(v), is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, vocab, seq, d, n_layer, n_head):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(seq, d)
        self.blocks = nn.ModuleList([Block(d, n_head) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, hook=None, want_acts=False):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1]))[None]
        acts = []
        for li, b in enumerate(self.blocks):
            x = b(x)
            if hook is not None:
                x = hook(li, x)
            if want_acts:
                acts.append(x)
        return self.head(self.lnf(x)), acts


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
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


def env_info():
    info = {"python": sys.version.split()[0]}
    for mod in ("numpy", "torch"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            pass
    return info


def build_sequences(moves, lengths, bos, seq_len):
    """inputs [BOS, m_0, ..., m_{T-1}] (length T+1 = seq_len); target at position p is m_p,
    so position p sees exactly the board AFTER move p-1, i.e. the board BEFORE move p."""
    G, T = moves.shape
    assert seq_len == T + 1
    inp = np.full((G, seq_len), bos, np.int64)
    inp[:, 1:] = np.where(moves >= 0, moves, bos)
    tgt = np.full((G, seq_len), -100, np.int64)
    tgt[:, :T] = moves
    valid = np.arange(seq_len)[None, :] < lengths[:, None]
    tgt = np.where(valid, tgt, -100)
    return inp, tgt, valid


def pad_time(a, T):
    """pad axis 1 up to length T with zeros."""
    pad = [(0, 0)] * a.ndim
    pad[1] = (0, T - a.shape[1])
    return np.pad(a, pad)


def cell_labels(boards, players):
    """(M,N,N) board, (M,) player -> (M, N*N) in {0 empty, 1 mine, 2 yours}."""
    b = boards.reshape(boards.shape[0], -1)
    p = players[:, None]
    return np.where(b == 0, 0, np.where(b == p, 1, 2)).astype(np.int64)


# ----------------------------------------------------------------------------
def main():
    import yaml
    with open(HERE / "experiment.yaml") as f:
        cfg = yaml.safe_load(f)
    seed = int(cfg["seed"])
    set_seeds(seed)
    t_start = time.time()
    timings = {}
    M = {}

    N = int(cfg["board"]["n"])
    NC = N * N
    MAXL = int(cfg["board"]["max_moves"])
    SEQ = int(cfg["model"]["seq_len"])
    BOS = NC          # token id 25 doubles as BOS and PAD
    VOCAB = NC + 1

    # ---- 1. engine self-test -------------------------------------------------
    st = engine_selftest()
    M["engine_selftest"] = st
    assert all(st.values()), f"engine self-test FAILED: {st}"
    print("engine self-test:", st)

    # ---- 2. data -------------------------------------------------------------
    t0 = time.time()
    n_train = 4000 if FAST else int(cfg["data"]["n_train_games"])
    n_eval = 1000 if FAST else int(cfg["data"]["n_eval_games"])
    n_pr_tr = 500 if FAST else int(cfg["data"]["n_probe_train_games"])
    n_pr_te = 500 if FAST else int(cfg["data"]["n_probe_test_games"])
    rng = np.random.default_rng(seed)
    tr_moves, _, _, tr_len, tr_pass = gen_games(N, n_train, rng, MAXL)
    ev_moves, ev_boards, ev_players, ev_len, _ = gen_games(N, n_eval, rng, MAXL)
    pr_moves, pr_boards, pr_players, pr_len, _ = gen_games(N, n_pr_tr + n_pr_te, rng, MAXL)
    timings["data_gen_s"] = round(time.time() - t0, 1)

    tr_uniq = len({tuple(m[:l]) for m, l in zip(tr_moves[:20000], tr_len[:20000])})
    M["data"] = {
        "n_train_games": n_train, "n_eval_games": n_eval,
        "mean_game_len": float(tr_len.mean()), "median_game_len": float(np.median(tr_len)),
        "min_game_len": int(tr_len.min()), "max_game_len": int(tr_len.max()),
        "frac_games_with_pass": float((tr_pass > 0).mean()),
        "mean_passes_per_game": float(tr_pass.mean()),
        "unique_games_in_first_20k": tr_uniq,
        "n_train_moves": int(tr_len.sum()),
    }
    print("data:", M["data"], f"({timings['data_gen_s']}s)")

    tr_inp, tr_tgt, _ = build_sequences(tr_moves, tr_len, BOS, SEQ)
    ev_inp, ev_tgt, ev_valid = build_sequences(ev_moves, ev_len, BOS, SEQ)
    tr_inp_t = torch.from_numpy(tr_inp)
    tr_tgt_t = torch.from_numpy(tr_tgt)
    ev_inp_t = torch.from_numpy(ev_inp)

    # ground-truth legal masks for every eval position (padded to SEQ; last column is never valid)
    ev_legal = legal_mask_chunked(ev_boards.reshape(-1, N, N),
                                  ev_players.reshape(-1)).reshape(n_eval, MAXL, NC)
    ev_legal = pad_time(ev_legal, SEQ)
    ev_legal_t = torch.from_numpy(ev_legal)
    pr_boards = pad_time(pr_boards, SEQ)
    pr_players = pad_time(pr_players, SEQ)
    M["data"]["mean_legal_moves_per_position"] = float(ev_legal.sum(-1)[ev_valid].mean())
    # entropy of the uniform-over-legal generator = irreducible CE
    M["data"]["irreducible_ce_nats"] = float(np.log(ev_legal.sum(-1)[ev_valid]).mean())

    # ---- 3. model + training -------------------------------------------------
    mc = cfg["model"]
    model = GPT(VOCAB, SEQ, int(mc["d_model"]), int(mc["n_layer"]), int(mc["n_head"]))
    n_params = sum(p.numel() for p in model.parameters())
    M["model"] = {"n_params": n_params, **{k: mc[k] for k in
                  ("d_model", "n_layer", "n_head", "seq_len", "vocab")}}
    print("params:", n_params)

    tc = cfg["train"]
    steps = 120 if FAST else int(tc["steps"])
    bs = int(tc["batch_size"])
    warm = int(tc["warmup"])
    opt = torch.optim.AdamW(model.parameters(), lr=float(tc["lr"]),
                            weight_decay=float(tc["weight_decay"]), betas=(0.9, 0.95))

    @torch.no_grad()
    def eval_model(n_games=1000):
        model.eval()
        tot_ce = tot_n = 0.0
        top1_legal = top1_n = 0
        tp = fp = fn = 0
        probs_all, legal_all_, mask_all = [], [], []
        for i in range(0, n_games, 500):
            xb = ev_inp_t[i:i + 500]
            logits, _ = model(xb)
            lg = logits[:, :, :NC]
            tg = torch.from_numpy(ev_tgt[i:i + 500])
            m = tg >= 0
            ce = F.cross_entropy(logits.reshape(-1, VOCAB), tg.reshape(-1),
                                 ignore_index=-100, reduction="sum")
            tot_ce += float(ce); tot_n += int(m.sum())
            pred = lg.argmax(-1)
            L = ev_legal_t[i:i + 500]
            hit = L.gather(2, pred.unsqueeze(-1)).squeeze(-1)
            top1_legal += int(hit[m].sum()); top1_n += int(m.sum())
            p = torch.softmax(logits, -1)[:, :, :NC]
            probs_all.append(p[m]); legal_all_.append(L[m]); mask_all.append(m)
        model.train()
        P = torch.cat(probs_all); Lm = torch.cat(legal_all_)
        return (tot_ce / tot_n, top1_legal / top1_n, P, Lm)

    curve = []
    t0 = time.time()
    g = torch.Generator().manual_seed(seed)
    for step in range(steps):
        lr = float(tc["lr"]) * min(1.0, (step + 1) / warm) * \
            (0.5 * (1 + math.cos(math.pi * step / steps)) * 0.9 + 0.1)
        for gp in opt.param_groups:
            gp["lr"] = lr
        idx = torch.randint(0, n_train, (bs,), generator=g)
        logits, _ = model(tr_inp_t[idx])
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), tr_tgt_t[idx].reshape(-1),
                               ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % int(tc["eval_every"]) == 0 or step == steps - 1:
            ce, leg, _, _ = eval_model(500)
            curve.append({"step": step, "train_loss": loss.item(), "eval_ce": ce,
                          "top1_legal": leg, "elapsed_s": round(time.time() - t0, 1)})
            print(f"  step {step:5d} loss {float(loss):.4f} eval_ce {ce:.4f} "
                  f"top1_legal {leg:.4f}  [{time.time()-t0:.0f}s]")
    timings["train_s"] = round(time.time() - t0, 1)
    M["train_curve"] = curve
    M["train_steps"] = steps
    M["train_tokens"] = steps * bs * SEQ

    ce, top1, Pev, Lev = eval_model(n_eval)
    M["eval"] = {"final_ce_nats": ce, "top1_legality_rate": top1,
                 "irreducible_ce_nats": M["data"]["irreducible_ce_nats"],
                 "excess_ce_nats": ce - M["data"]["irreducible_ce_nats"]}

    # calibrate a probability threshold for "this move is legal"
    best = (None, -1)
    for tau in np.concatenate([np.logspace(-5, -0.7, 40)]):
        pr = (Pev > tau)
        tp = float((pr & Lev).sum()); fp = float((pr & ~Lev).sum()); fn = float((~pr & Lev).sum())
        f1 = 2 * tp / max(1e-9, 2 * tp + fp + fn)
        if f1 > best[1]:
            best = (float(tau), f1)
    TAU, legal_f1 = best
    M["eval"]["legal_set_tau"] = TAU
    M["eval"]["legal_set_f1"] = legal_f1
    pr = Pev > TAU
    M["eval"]["legal_set_precision"] = float((pr & Lev).sum()) / max(1, float(pr.sum()))
    M["eval"]["legal_set_recall"] = float((pr & Lev).sum()) / float(Lev.sum())
    M["eval"]["exact_legal_set_match_rate"] = float((pr == Lev).all(-1).float().mean())
    print("eval:", M["eval"])

    # ---- 4. probes -----------------------------------------------------------
    t0 = time.time()
    pr_inp, _, pr_valid = build_sequences(pr_moves, pr_len, BOS, SEQ)
    n_layer = int(mc["n_layer"])
    acts_by_layer = [[] for _ in range(n_layer)]
    lab_list, gidx_list, pos_list = [], [], []
    with torch.no_grad():
        model.eval()
        for i in range(0, pr_inp.shape[0], 500):
            xb = torch.from_numpy(pr_inp[i:i + 500])
            _, acts = model(xb, want_acts=True)
            v = pr_valid[i:i + 500].copy()
            v[:, 0] = False                      # position 0 (BOS) sees the constant start board
            vm = torch.from_numpy(v)
            gi, pi = np.nonzero(v)
            for li in range(n_layer):
                acts_by_layer[li].append(acts[li][vm].numpy().astype(np.float32))
            lab_list.append(cell_labels(pr_boards[i:i + 500][v], pr_players[i:i + 500][v]))
            gidx_list.append(gi + i); pos_list.append(pi)
    A = [np.concatenate(a) for a in acts_by_layer]
    Y = np.concatenate(lab_list)
    GIDX = np.concatenate(gidx_list); POS = np.concatenate(pos_list)
    is_test = GIDX >= n_pr_tr                    # split BY GAME
    d = int(mc["d_model"])
    M["probe"] = {"n_train_positions": int((~is_test).sum()),
                  "n_test_positions": int(is_test.sum())}

    ytr = torch.from_numpy(Y[~is_test]); yte = torch.from_numpy(Y[is_test])
    maj = np.stack([np.bincount(Y[is_test][:, c], minlength=3) for c in range(NC)])
    maj_acc = (maj.max(1) / maj.sum(1))
    M["probe"]["majority_per_cell"] = maj_acc.round(4).tolist()

    probe_acc = {}
    probe_store = {}
    pc = cfg["probe"]
    for li in range(n_layer):
        Xtr_np, Xte_np = A[li][~is_test], A[li][is_test]
        mu, sd = Xtr_np.mean(0), Xtr_np.std(0) + 1e-6
        Xtr = torch.from_numpy((Xtr_np - mu) / sd)
        Xte = torch.from_numpy((Xte_np - mu) / sd)
        W = torch.zeros(NC, d, 3, requires_grad=True)
        b = torch.zeros(NC, 3, requires_grad=True)
        o = torch.optim.Adam([W, b], lr=float(pc["lr"]), weight_decay=float(pc["weight_decay"]))
        gg = torch.Generator().manual_seed(seed + li)
        nsteps = 60 if FAST else int(pc["steps"])
        for s in range(nsteps):
            for gp in o.param_groups:
                gp["lr"] = float(pc["lr"]) * 0.5 * (1 + math.cos(math.pi * s / nsteps))
            j = torch.randint(0, Xtr.shape[0], (8192,), generator=gg)
            lg = torch.einsum("nd,cdk->nck", Xtr[j], W) + b
            loss = F.cross_entropy(lg.reshape(-1, 3), ytr[j].reshape(-1))
            o.zero_grad(set_to_none=True); loss.backward(); o.step()
        with torch.no_grad():
            preds = []
            for i in range(0, Xte.shape[0], 20000):
                preds.append((torch.einsum("nd,cdk->nck", Xte[i:i + 20000], W) + b).argmax(-1))
            pred = torch.cat(preds)
            acc = (pred == yte).float().mean(0).numpy()
            occ = yte > 0
            acc_occ = np.array([float(((pred[:, c] == yte[:, c]) & occ[:, c]).sum()) /
                                max(1, int(occ[:, c].sum())) for c in range(NC)])
        probe_acc[li] = acc
        probe_store[li] = {"W": W.detach().clone(), "b": b.detach().clone(),
                           "mu": mu, "sd": sd, "acc": acc, "acc_occ": acc_occ}
        print(f"  probe L{li}: mean acc {acc.mean():.4f} (majority {maj_acc.mean():.4f}) "
              f"occupied-only {acc_occ.mean():.4f}")
    M["probe"]["mean_acc_per_layer"] = [float(probe_acc[li].mean()) for li in range(n_layer)]
    M["probe"]["mean_majority"] = float(maj_acc.mean())
    timings["probe_s"] = round(time.time() - t0, 1)

    # intervention layer: most decodable layer that still leaves computation downstream
    cand = list(range(n_layer - 1))
    LAYER = max(cand, key=lambda li: probe_acc[li].mean())
    M["probe"]["intervention_layer"] = LAYER
    M["probe"]["acc_per_cell_at_layer"] = probe_store[LAYER]["acc"].round(4).tolist()
    M["probe"]["acc_occupied_per_cell_at_layer"] = probe_store[LAYER]["acc_occ"].round(4).tolist()

    # ---- 5. causal test ------------------------------------------------------
    t0 = time.time()
    cc = cfg["causal"]
    MINPOS = int(cc["min_move_index"])
    MAXI = 40 if FAST else int(cc["max_instances_per_cell"])
    margins = [float(x) for x in cc["margin_grid"]]

    ps = probe_store[LAYER]
    sd_t = torch.from_numpy(ps["sd"])
    W_eff = ps["W"] / sd_t[None, :, None]          # d(logit)/d(raw residual)
    b_eff = ps["b"] - torch.einsum("d,cdk->ck", torch.from_numpy(ps["mu"]) / sd_t, ps["W"])

    def probe_pred_raw(h):                          # h (B, d) raw residual
        return (torch.einsum("nd,cdk->nck", h, W_eff) + b_eff).argmax(-1)

    # candidate positions from held-out probe games
    cand_i = np.flatnonzero((POS >= MINPOS) & is_test)
    rng2 = np.random.default_rng(seed + 77)
    rng2.shuffle(cand_i)
    cand_i = cand_i[:20000]
    cg, cp = GIDX[cand_i], POS[cand_i]
    cboards = pr_boards[cg, cp]                     # (K,N,N)
    cplayers = pr_players[cg, cp]
    clab = cell_labels(cboards, cplayers)           # (K, NC)
    clegal = legal_mask_chunked(cboards, cplayers).reshape(-1, NC)

    per_cell = []
    for c in range(NC):
        occ = np.flatnonzero(clab[:, c] > 0)
        if len(occ) == 0:
            per_cell.append({"cell": c, "n_usable": 0, "skipped": True}); continue
        eb = cboards[occ].copy()                    # edited board: flip cell c's colour
        eb[:, c // N, c % N] *= -1
        elegal = legal_mask_chunked(eb, cplayers[occ]).reshape(-1, NC)
        usable = (elegal != clegal[occ]).any(1)     # only edits the legal set can register
        keep = np.flatnonzero(usable)[:MAXI]
        if len(keep) < 10:
            per_cell.append({"cell": c, "n_usable": int(usable.sum()), "skipped": True}); continue
        sel = occ[keep]
        el = elegal[keep]
        ol = clegal[sel]
        dmask = torch.from_numpy(el != ol)          # moves whose legality the edit changes
        el_t, ol_t = torch.from_numpy(el), torch.from_numpy(ol)
        cur = torch.from_numpy(clab[sel, c])
        tgt_cls = torch.where(cur == 1, torch.tensor(2), torch.tensor(1))  # mine <-> yours

        xb = torch.from_numpy(pr_inp[cg[sel]])
        posb = torch.from_numpy(cp[sel])
        ar = torch.arange(len(sel))

        with torch.no_grad():
            captured = {}

            def cap(li, x):
                if li == LAYER:
                    captured["h"] = x[ar, posb].clone()
                return x
            logits0, _ = model(xb, hook=cap)
            p0 = torch.softmax(logits0[ar, posb], -1)[:, :NC]
            h0 = captured["h"]
            pre_probe = probe_pred_raw(h0)
            pred_before = p0 > TAU
            Wc, bc = W_eff[c], b_eff[c]                               # (d,3), (3,)

            def make_delta(gam, rounds=3):
                """Minimum-norm residual patch that makes cell c's probe read `tgt_cls` with
                logit margin >= gam over the runner-up class (Li et al. style world-model edit,
                solved in closed form instead of by gradient descent)."""
                delta = torch.zeros_like(h0)
                for _ in range(rounds):
                    lg = (h0 + delta) @ Wc + bc                       # (B,3)
                    lg_t = lg.gather(1, tgt_cls[:, None]).squeeze(1)
                    masked = lg.scatter(1, tgt_cls[:, None], -1e9)
                    oth = masked.argmax(1)
                    u = (Wc[:, tgt_cls] - Wc[:, oth]).T               # (B,d)
                    m = lg_t - masked.max(1).values
                    step = ((gam - m) / ((u * u).sum(-1) + 1e-9)).clamp(min=0)
                    delta = delta + step[:, None] * u
                return delta

            def score(delta):
                def edit(li, x):
                    if li == LAYER:
                        x = x.clone()
                        x[ar, posb] = x[ar, posb] + delta
                    return x
                logits1, _ = model(xb, hook=edit)
                p1 = torch.softmax(logits1[ar, posb], -1)[:, :NC]
                post = probe_pred_raw(h0 + delta)
                agree = ((p1 > TAU) == el_t)
                strict = float((agree & dmask).sum()) / float(dmask.sum())
                direc = float(((p1 > p0) == el_t)[dmask].float().mean())
                changed = post != pre_probe
                n_other_changed = changed.sum(1) - changed[:, c].long()
                coll = float(n_other_changed.float().mean()) / (NC - 1)
                # "clean" edits: the target cell's readout flipped and NO other cell moved
                cln = (n_other_changed == 0) & (post[:, c] == tgt_cls)
                dm_c = dmask & cln[:, None]
                strict_clean = (float((agree & dm_c).sum()) / float(dm_c.sum())
                                if int(dm_c.sum()) > 0 else None)
                return {"strict": strict, "direction": direc,
                        "probe_flip": float((post[:, c] == tgt_cls).float().mean()),
                        "collateral": coll, "frac_clean": float(cln.float().mean()),
                        "strict_clean": strict_clean, "n_clean": int(cln.sum()),
                        "delta_norm": float(delta.norm(dim=-1).mean()),
                        "rel_delta_norm": float((delta.norm(dim=-1) /
                                                 h0.norm(dim=-1)).mean())}

            rows = {}
            grand = torch.Generator().manual_seed(seed + 1000 + c)
            for gam in margins:
                delta = make_delta(gam)
                rows[str(gam)] = score(delta)
                r = torch.randn(h0.shape, generator=grand)
                r = r / r.norm(dim=-1, keepdim=True) * delta.norm(dim=-1, keepdim=True)
                rows["rand" + str(gam)] = score(r)

            no_edit = float(((pred_before == el_t) & dmask).sum()) / float(dmask.sum())
            ceiling = float(((pred_before == ol_t) & dmask).sum()) / float(dmask.sum())

        per_cell.append({
            "cell": c, "n": len(sel), "n_occupied_candidates": int(len(occ)),
            "n_usable": int(usable.sum()),
            "mean_changed_moves": float(dmask.float().sum(1).mean()),
            "no_edit_baseline": no_edit, "model_legality_ceiling": ceiling,
            "by_margin": rows,
        })

    ok = [x for x in per_cell if not x.get("skipped")]
    mean_by_margin = {str(g): float(np.mean([x["by_margin"][str(g)]["strict"] for x in ok]))
                      for g in margins}
    coll_by_margin = {str(g): float(np.mean([x["by_margin"][str(g)]["collateral"] for x in ok]))
                      for g in margins}
    # HEADLINE margin: the largest gamma meeting the PRE-STATED specificity constraint
    # (collateral <= budget), NOT the one that maximises the outcome. Both are reported.
    budget = float(cc.get("collateral_budget", 0.10))
    spec_ok = [g for g in margins if coll_by_margin[str(g)] <= budget]
    GAM = str(max(spec_ok)) if spec_ok else str(min(margins))
    GAM_MAX = max(mean_by_margin, key=mean_by_margin.get)
    M["causal"] = {
        "intervention_layer": LAYER, "tau": TAU, "n_cells_evaluated": len(ok),
        "margin_sweep_mean_strict": mean_by_margin,
        "margin_sweep_mean_direction": {str(g): float(np.mean(
            [x["by_margin"][str(g)]["direction"] for x in ok])) for g in margins},
        "margin_sweep_mean_probe_flip": {str(g): float(np.mean(
            [x["by_margin"][str(g)]["probe_flip"] for x in ok])) for g in margins},
        "margin_sweep_mean_collateral": {str(g): float(np.mean(
            [x["by_margin"][str(g)]["collateral"] for x in ok])) for g in margins},
        "margin_sweep_mean_frac_clean": {str(g): float(np.mean(
            [x["by_margin"][str(g)]["frac_clean"] for x in ok])) for g in margins},
        "margin_sweep_mean_strict_clean": {str(g): (float(np.mean(
            [x["by_margin"][str(g)]["strict_clean"] for x in ok
             if x["by_margin"][str(g)]["strict_clean"] is not None]))
            if any(x["by_margin"][str(g)]["strict_clean"] is not None for x in ok)
            else None) for g in margins},
        "chosen_margin": GAM, "collateral_budget": budget,
        "margin_maximising_strict": GAM_MAX,
        "margin_choice_rule": "largest gamma with mean collateral <= collateral_budget "
                              "(pre-stated specificity constraint, not outcome-maximising)",
        "margin_sweep_random_control_strict": {str(g): float(np.mean(
            [x["by_margin"]["rand" + str(g)]["strict"] for x in ok])) for g in margins},
        "margin_sweep_random_control_direction": {str(g): float(np.mean(
            [x["by_margin"]["rand" + str(g)]["direction"] for x in ok])) for g in margins},
        "margin_sweep_mean_rel_delta_norm": {str(g): float(np.mean(
            [x["by_margin"][str(g)]["rel_delta_norm"] for x in ok])) for g in margins},
        "random_control_mean_strict": float(np.mean(
            [x["by_margin"]["rand" + GAM]["strict"] for x in ok])),
        "random_control_mean_direction": float(np.mean(
            [x["by_margin"]["rand" + GAM]["direction"] for x in ok])),
        "no_edit_baseline_mean": float(np.mean([x["no_edit_baseline"] for x in ok])),
        "model_legality_ceiling_mean": float(np.mean([x["model_legality_ceiling"] for x in ok])),
        "per_cell": per_cell,
    }
    timings["causal_s"] = round(time.time() - t0, 1)
    print("causal margin sweep:", mean_by_margin, "chosen", GAM)

    # ---- 6. the deliverable: probe accuracy vs causal success ----------------
    cells = [x["cell"] for x in ok]
    pa = np.array([probe_store[LAYER]["acc_occ"][c] for c in cells])       # matched: occupied only
    pa3 = np.array([probe_store[LAYER]["acc"][c] for c in cells])
    cs = np.array([x["by_margin"][GAM]["strict"] for x in ok])
    cd = np.array([x["by_margin"][GAM]["direction"] for x in ok])
    ceil = np.array([x["model_legality_ceiling"] for x in ok])
    pf = np.array([x["by_margin"][GAM]["probe_flip"] for x in ok])

    def pearson(a, b):
        if a.std() < 1e-9 or b.std() < 1e-9:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    def spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        return pearson(ra, rb)

    # per-cell EXCESS over the norm-matched random-direction control = the causal quantity
    cx = np.array([x["by_margin"][GAM]["strict"] - x["by_margin"]["rand" + GAM]["strict"]
                   for x in ok])
    cdx = np.array([x["by_margin"][GAM]["direction"] - x["by_margin"]["rand" + GAM]["direction"]
                    for x in ok])
    hi_probe = pa >= np.median(pa)
    lo_cause = cx <= 0.05          # causal edit buys <=5 points over a random matched-norm patch
    M["headline"] = {
        "chosen_margin": GAM,
        "pearson_probe_vs_causal_strict": pearson(pa, cs),
        "spearman_probe_vs_causal_strict": spearman(pa, cs),
        "pearson_probe_vs_causal_excess": pearson(pa, cx),
        "spearman_probe_vs_causal_excess": spearman(pa, cx),
        "pearson_probe_vs_causal_direction": pearson(pa, cd),
        "spearman_probe_vs_causal_direction": spearman(pa, cd),
        "pearson_probe_vs_direction_excess": pearson(pa, cdx),
        "pearson_probe3way_vs_causal_strict": pearson(pa3, cs),
        "n_cells": len(cells),
        "n_high_probe_low_cause": int((hi_probe & lo_cause).sum()),
        "high_probe_low_cause_cells": [int(cells[i]) for i in np.flatnonzero(hi_probe & lo_cause)],
        "high_probe_low_cause_rule": "probe acc (occupied) >= median AND strict causal excess "
                                     "over the norm-matched random control <= 0.05",
        "mean_probe_acc_occ": float(pa.mean()),
        "mean_causal_strict": float(cs.mean()),
        "mean_causal_excess": float(cx.mean()),
        "mean_causal_direction": float(cd.mean()),
        "mean_direction_excess": float(cdx.mean()),
        "mean_probe_flip_rate": float(pf.mean()),
        "causal_strict_range": [float(cs.min()), float(cs.max())],
        "causal_excess_range": [float(cx.min()), float(cx.max())],
        "probe_acc_occ_range": [float(pa.min()), float(pa.max())],
        "pearson_causal_vs_model_ceiling": pearson(ceil, cs),
    }
    # the same correlation at EVERY intervention strength, so nothing hinges on the choice
    corr_by_margin = {}
    for g in margins:
        gs = str(g)
        s_ = np.array([x["by_margin"][gs]["strict"] for x in ok])
        e_ = np.array([x["by_margin"][gs]["strict"] - x["by_margin"]["rand" + gs]["strict"]
                       for x in ok])
        d_ = np.array([x["by_margin"][gs]["direction"] for x in ok])
        corr_by_margin[gs] = {
            "pearson_probe_vs_strict": pearson(pa, s_),
            "spearman_probe_vs_strict": spearman(pa, s_),
            "pearson_probe_vs_excess": pearson(pa, e_),
            "pearson_probe_vs_direction": pearson(pa, d_),
            "n_high_probe_low_cause": int(((pa >= np.median(pa)) & (e_ <= 0.05)).sum()),
            "mean_excess": float(e_.mean()),
        }
    M["headline"]["by_margin"] = corr_by_margin

    # ---- geometry: is "decodable but not causal" a board-position property? ---
    def kind_of(i):
        r, cl = i // N, i % N
        edge_r, edge_c = r in (0, N - 1), cl in (0, N - 1)
        return "corner" if (edge_r and edge_c) else ("edge" if (edge_r or edge_c) else "interior")

    kinds = np.array([kind_of(c) for c in cells])
    geo = {}
    for k in ("corner", "edge", "interior"):
        s = kinds == k
        if s.sum() == 0:
            continue
        geo[k] = {"n": int(s.sum()),
                  "mean_probe_acc_occ": float(pa[s].mean()),
                  "mean_probe_acc_3way": float(pa3[s].mean()),
                  "mean_causal_strict": float(cs[s].mean()),
                  "mean_causal_excess": float(cx[s].mean()),
                  "mean_direction": float(cd[s].mean()),
                  "mean_changed_moves": float(np.mean(
                      [ok[i]["mean_changed_moves"] for i in np.flatnonzero(s)])),
                  "cells": [int(cells[i]) for i in np.flatnonzero(s)]}
    interior = kinds == "interior"
    obs = float(cx[interior].mean() - cx[~interior].mean())
    rp = np.random.default_rng(seed)
    cnt = 0
    NPERM = 20000
    for _ in range(NPERM):
        pm = rp.permutation(cx)
        if pm[:int(interior.sum())].mean() - pm[int(interior.sum()):].mean() >= obs:
            cnt += 1
    M["geometry"] = {
        "by_kind": geo,
        "interior_minus_border_causal_excess": obs,
        "permutation_p_one_sided": (cnt + 1) / (NPERM + 1),
        "n_perm": NPERM,
        "interior_excess_range": [float(cx[interior].min()), float(cx[interior].max())],
        "border_excess_range": [float(cx[~interior].min()), float(cx[~interior].max())],
        "pearson_probe3way_vs_causal_excess": pearson(pa3, cx),
        "spearman_probe3way_vs_causal_excess": spearman(pa3, cx),
        "note": "3-way probe accuracy (the quantity the Othello-GPT literature reports) is "
                "highest exactly on the cells with the LEAST causal effect",
    }
    print("geometry:", json.dumps(M["geometry"], indent=1, default=float))
    print("HEADLINE:", json.dumps(M["headline"], indent=1))

    # ---- 7. chart ------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    a = ax[0, 0]
    a.plot([c["step"] for c in curve], [c["top1_legal"] for c in curve], "o-", label="top-1 legality")
    a.axhline(1.0, ls=":", c="gray")
    a.set_xlabel("step"); a.set_ylabel("rate"); a.set_title(
        f"(a) legal-move learning\nfinal top-1 legality {top1:.4f}, legal-set F1 {legal_f1:.3f}")
    a2 = a.twinx()
    a2.plot([c["step"] for c in curve], [c["eval_ce"] for c in curve], "s--", c="crimson", alpha=.6)
    a2.axhline(M["data"]["irreducible_ce_nats"], ls=":", c="crimson")
    a2.set_ylabel("eval CE (nats), dashed", color="crimson")
    a.legend(loc="lower right", fontsize=8)

    grid_p = np.full(NC, np.nan); grid_c = np.full(NC, np.nan)
    for i, c in enumerate(cells):
        grid_p[c] = pa[i]; grid_c[c] = cx[i]
    for j, (g, ttl, vmax) in enumerate(
            [(grid_p, f"(b) probe accuracy per cell\n(mine/yours, occupied, layer {LAYER})", 1.0),
             (grid_c, f"(c) CAUSAL effect per cell (gamma={GAM})\nstrict success minus "
                      f"matched-norm random control", float(np.nanmax(grid_c)))]):
        a = ax[0, 1 + j]
        im = a.imshow(g.reshape(N, N), vmin=0, vmax=vmax, cmap="viridis")
        for r in range(N):
            for cc_ in range(N):
                v = g.reshape(N, N)[r, cc_]
                a.text(cc_, r, "n/a" if np.isnan(v) else f"{v:.2f}", ha="center", va="center",
                       color="w" if (np.isnan(v) or v < .6 * vmax) else "k", fontsize=9)
        a.set_title(ttl, fontsize=10); a.set_xticks(range(N)); a.set_yticks(range(N))
        plt.colorbar(im, ax=a, fraction=0.046)

    a = ax[1, 0]
    style = {"interior": ("tab:green", "o"), "edge": ("tab:blue", "s"),
             "corner": ("tab:red", "D")}
    for k, (col, mk) in style.items():
        s = kinds == k
        if s.sum():
            a.scatter(pa[s], cx[s], c=col, marker=mk, s=95, edgecolor="k", zorder=3,
                      label=f"{k} (n={int(s.sum())}, mean excess {cx[s].mean():.3f})")
    for i, c in enumerate(cells):
        a.annotate(str(c), (pa[i], cx[i]), fontsize=6, xytext=(4, 3), textcoords="offset points")
    a.axhline(0.05, ls="--", c="gray", label="low-cause cut (excess <= 0.05)")
    a.axvline(float(np.median(pa)), ls=":", c="gray", label="median probe acc")
    a.set_xlabel("linear probe accuracy (mine/yours, occupied cells)")
    a.set_ylabel("causal excess over matched-norm random control")
    a.set_title("(d) DELIVERABLE: probed vs caused, per cell\n"
                f"gamma={GAM}: pearson r={M['headline']['pearson_probe_vs_causal_excess']:.3f}, "
                f"high-probe/low-cause = {M['headline']['n_high_probe_low_cause']}/{len(cells)} "
                f"{M['headline']['high_probe_low_cause_cells']}", fontsize=10)
    a.legend(fontsize=6.5, loc="best"); a.grid(alpha=.3)

    a = ax[1, 1]
    xs = margins
    a.plot(xs, [mean_by_margin[str(g)] for g in margins], "o-", label="causal strict")
    a.plot(xs, [M["causal"]["margin_sweep_mean_direction"][str(g)] for g in margins], "s-",
           label="causal direction")
    a.plot(xs, [M["causal"]["margin_sweep_mean_probe_flip"][str(g)] for g in margins], "^-",
           label="probe flip rate")
    a.plot(xs, [M["causal"]["margin_sweep_mean_collateral"][str(g)] for g in margins], "v-",
           label="collateral (other cells)")
    a.plot(xs, [M["causal"]["margin_sweep_random_control_strict"][str(g)] for g in margins], "--",
           c="gray", label="random ctrl (strict)")
    a.plot(xs, [M["causal"]["margin_sweep_random_control_direction"][str(g)] for g in margins],
           ":", c="dimgray", label="random ctrl (direction)")
    a.plot(xs, [M["causal"]["margin_sweep_mean_rel_delta_norm"][str(g)] for g in margins], "d-",
           c="saddlebrown", label="||edit|| / ||residual||")
    a.axhline(M["causal"]["no_edit_baseline_mean"], ls="-.", c="crimson", lw=1,
              label="no-edit baseline")
    a.axvline(float(GAM), c="k", lw=1, alpha=.5)
    a.set_xscale("log"); a.set_xlabel("probe logit margin gamma"); a.set_ylabel("rate")
    a.set_title("(e) intervention strength: effect is bought with specificity\n"
                f"vertical line = headline gamma (collateral budget {budget})")
    a.legend(fontsize=6.5); a.grid(alpha=.3)

    a = ax[1, 2]
    for li in range(n_layer):
        a.plot(range(NC), np.sort(probe_acc[li])[::-1], "-", label=f"layer {li}")
    a.plot(range(NC), np.sort(maj_acc)[::-1], "k:", label="majority")
    a.set_xlabel("cell (sorted)"); a.set_ylabel("3-way probe accuracy")
    a.set_title("(f) probe accuracy by layer"); a.legend(fontsize=7); a.grid(alpha=.3)

    fig.suptitle("Mini-Othello (5x5) world model: which cells are decodable but not causal?",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(HERE / "chart.png", dpi=110)

    # ---- write ---------------------------------------------------------------
    timings["total_s"] = round(time.time() - t_start, 1)
    M["timings"] = timings
    results = {
        "id": cfg["id"], "git_commit": git_sha(), "seed": seed,
        "duration_sec": timings["total_s"], "metrics": M, "env": env_info(),
        "status": "done",
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("wrote results.json; total", timings["total_s"], "s")


if __name__ == "__main__":
    main()
