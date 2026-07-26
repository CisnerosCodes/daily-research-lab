"""ListOps-mini: nested prefix expressions over digits 0-9 with ops MAX / MIN / MED.

Key design constraint: DEPTH and LENGTH are DECOUPLED. Every expression, whatever its
nesting depth, is grown to a target token length drawn from the SAME band. Without this
control, "does test-time depth-6 accuracy improve with more loops" would be confounded
with plain length extrapolation (deeper trees are naturally longer), and the registry
already knows (2026-07-26_looped-halt-nrasp) that fixed-K only fails when train length is
locked to train depth. Here it is not.

Vocab (17 symbols):
    0 PAD, 1 CLS, 2 '[', 3 ']', 4 MAX, 5 MIN, 6 MED, 7..16 digits 0..9
"""
import random

PAD, CLS, LBR, RBR = 0, 1, 2, 3
OPS = {"MAX": 4, "MIN": 5, "MED": 6}
OP_IDS = list(OPS.values())
DIGIT0 = 7
VOCAB = 17


def apply_op(op_id, vals):
    if op_id == OPS["MAX"]:
        return max(vals)
    if op_id == OPS["MIN"]:
        return min(vals)
    s = sorted(vals)
    return s[(len(s) - 1) // 2]  # floor median, stays in 0..9


def bag_heuristic(node):
    """The structure-blind shortcut: apply the ROOT operator to the flat multiset of all
    digits anywhere in the expression. Exact by construction at depth 1; the accuracy of
    this at depth >= 2 is the bar a genuinely recursive model has to clear."""
    digs = []

    def rec(n):
        if n.op is None:
            digs.append(n.val)
        else:
            for k in n.kids:
                rec(k)
    rec(node)
    return apply_op(node.op, digs)


class Node:
    """Either a leaf digit (op is None) or an internal node with >=2 children."""

    __slots__ = ("op", "kids", "val", "height")

    def __init__(self, op=None, kids=None, val=None):
        self.op = op
        self.kids = kids or []
        if op is None:
            self.val, self.height = val, 0
        else:
            self.val = apply_op(op, [k.val for k in self.kids])
            self.height = 1 + max(k.height for k in self.kids)

    def recompute(self):
        if self.op is not None:
            self.val = apply_op(self.op, [k.val for k in self.kids])
            self.height = 1 + max(k.height for k in self.kids)
        return self


def leaf(rng):
    return Node(val=rng.randrange(10))


def n_tokens(node):
    if node.op is None:
        return 1
    return 3 + sum(n_tokens(k) for k in node.kids)  # '[' OP kids ']'


def spine(rng, depth):
    """Minimal exact-height-`depth` tree: every node has 2 children, one of which is the
    next level down (or two leaves at height 1)."""
    if depth == 0:
        return leaf(rng)
    child = spine(rng, depth - 1)
    kids = [child, leaf(rng)]
    rng.shuffle(kids)
    return Node(op=rng.choice(OP_IDS), kids=kids)


def _internal_nodes(node, out):
    if node.op is not None:
        out.append(node)
        for k in node.kids:
            _internal_nodes(k, out)
    return out


def grow(rng, root, target_len, max_arity=32):
    """Pad the tree out to ~target_len tokens by adding extra arguments (digits, or
    shallow sub-expressions) to randomly chosen internal nodes. Never increases height."""
    for _ in range(400):
        cur = n_tokens(root)
        if cur >= target_len:
            break
        cands = [n for n in _internal_nodes(root, []) if len(n.kids) < max_arity]
        if not cands:
            break
        host = rng.choice(cands)
        room = target_len - cur
        # A sibling subtree may only be added if its height stays STRICTLY below the
        # host's own height, otherwise growing would raise the tree's height and break
        # the exact-depth contract.
        if room >= 5 and rng.random() < 0.5 and host.height >= 2:
            hmax = min(host.height - 1, 3)
            h = rng.randint(1, hmax)
            sub = spine(rng, h)
            if n_tokens(sub) > room:
                sub = leaf(rng)
            host.kids.append(sub)
        else:
            host.kids.append(leaf(rng))
        _refresh(root)
    return root


def _refresh(node):
    if node.op is not None:
        for k in node.kids:
            _refresh(k)
        node.recompute()


def gen_expr(rng, depth, len_lo, len_hi):
    root = spine(rng, depth)
    target = rng.randint(len_lo, len_hi)
    grow(rng, root, target)
    _refresh(root)
    assert root.height == depth, (root.height, depth)
    return root


def tokenize(node, out):
    if node.op is None:
        out.append(DIGIT0 + node.val)
    else:
        out.append(LBR)
        out.append(node.op)
        for k in node.kids:
            tokenize(k, out)
        out.append(RBR)
    return out


def annotate(node, toks_pos):
    """Walk the token stream in the same order as tokenize() and record, for every
    internal node, the index of its CLOSING bracket, its height and its value.
    Returns list of (close_idx, height, value)."""
    spans = []

    def rec(n, i):
        if n.op is None:
            return i + 1
        start = i
        i += 2  # '[' OP
        for k in n.kids:
            i = rec(k, i)
        spans.append((i, n.height, n.val))  # i is the ']' index
        return i + 1

    rec(node, toks_pos)
    return spans


def encode(node, max_len):
    """[CLS] + tokens, right-padded to max_len. Returns (ids, answer, spans, true_len)."""
    toks = tokenize(node, [])
    ids = [CLS] + toks
    L = len(ids)
    assert L <= max_len, (L, max_len)
    spans = annotate(node, 1)  # +1 for the CLS offset
    ids = ids + [PAD] * (max_len - L)
    return ids, node.val, spans, L


def make_dataset(seed, depths, n_per_depth, len_lo, len_hi, max_len):
    """Returns dict depth -> (ids list, answers list, spans list, lens list)."""
    rng = random.Random(seed)
    out = {}
    for d in depths:
        ids, ans, spans, lens = [], [], [], []
        seen = set()
        tries = 0
        while len(ids) < n_per_depth and tries < n_per_depth * 50:
            tries += 1
            node = gen_expr(rng, d, len_lo, len_hi)
            i, a, s, L = encode(node, max_len)
            key = tuple(i[:L])
            if key in seen:
                continue
            seen.add(key)
            ids.append(i)
            ans.append(a)
            spans.append(s)
            lens.append(L)
        out[d] = (ids, ans, spans, lens)
    return out


if __name__ == "__main__":
    rng = random.Random(0)
    import statistics as st

    for d in range(1, 7):
        lens, vals = [], []
        for _ in range(400):
            n = gen_expr(rng, d, 26, 40)
            t = tokenize(n, [])
            lens.append(len(t))
            vals.append(n.val)
        print(f"depth {d}: len min/mean/max {min(lens)}/{st.mean(lens):.1f}/{max(lens)}"
              f"  ans entropy-ish {len(set(vals))} distinct, mode frac "
              f"{max(vals.count(v) for v in set(vals))/len(vals):.3f}")
    n = gen_expr(rng, 3, 26, 40)
    print("".join(
        {2: "[", 3: "]", 4: "MAX ", 5: "MIN ", 6: "MED "}.get(t, str(t - DIGIT0) + " ")
        for t in tokenize(n, [])), "=", n.val)
    print("spans (close_idx,height,val):", annotate(n, 1))
