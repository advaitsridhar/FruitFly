"""
The genome as a wiring recipe: grow new connectomes from cell-type rules, and squeeze the rules
through a bottleneck to see how much of the fly's behaviour survives.

A genome of ~140 million letters cannot list 90 million synapses. What it can hold is rules: which
cell types connect to which, how often and how strongly. This module learns exactly those rules
from the MaleCNS wiring (per pair of *(cell type, side)* groups: how many neuron pairs connect and
the distribution of their synapse counts), grows a **synthetic fly** from them (the same 176,422
neurons, wired afresh at random within the rules), and, in the spirit of the genomic-bottleneck
experiments (Koulakov, Shuvaev & Zador 2024), compresses the rule table to a low rank so that a
"genome size" dial decides how specific the wiring can be. Running the validated experiments on
each grown fly then measures which reflexes live in the type-level rules and which need the exact
neuron-to-neuron wiring (the retinotopic optic-lobe columns, for instance).

Levels, from the most to the least specific genome::

    real                the connectome itself (nothing grown)
    type                rules per (cell type, side) pair: ~1.5 M pairs for 11,691 types
    bottleneck:K        the type rules approximated at rank K (each group gets a K-number code;
                        a connection rule is the product of two codes), K = 8 ... 512
    class               rules per (class, side) pair only: a few hundred groups

Everything is deterministic given a seed: two flies grown with the same rules and seed are
identical, two seeds are two individuals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .connectome import Connectome

MIN_SYN, MAX_SYN = 5, 65535        # the data keeps connections with >= 5 synapses


@dataclass
class Rules:
    """Type-level wiring rules learned from a connectome (see :func:`learn_rules`)."""
    level: str                      # "type", "class", or "bottleneck:K"
    groups: list[str]               # group names, e.g. "MN9/L"
    group_of: np.ndarray            # int32 per neuron
    order: np.ndarray               # neurons sorted by group
    bounds: np.ndarray              # group g's neurons are order[bounds[g]:bounds[g+1]]
    pair_pre: np.ndarray            # int32, one entry per connected group pair
    pair_post: np.ndarray
    pair_edges: np.ndarray          # how many neuron pairs of the group pair are connected
    pair_mu: np.ndarray             # mean of log(synapses) over those connections
    pair_sigma: np.ndarray          # standard deviation of log(synapses)
    n_edges_total: int
    n_syn_total: int
    rank: int | None = None
    notes: dict = field(default_factory=dict)

    @property
    def n_groups(self) -> int:
        return len(self.groups)

    @property
    def n_pairs(self) -> int:
        return int(self.pair_pre.size)

    def size_numbers(self) -> int:
        """How many numbers the genome holds at this level (the table, or the codes)."""
        if self.rank:
            return int(2 * self.n_groups * self.rank + 2 * self.n_groups)
        return int(3 * self.n_pairs)

    def summary(self) -> dict:
        return {"level": self.level, "groups": self.n_groups, "pairs": self.n_pairs, "edges": int(self.pair_edges.sum()),
                "numbers": self.size_numbers(), "rank": self.rank, **self.notes}


def _group_key(conn: Connectome, by: str) -> np.ndarray:
    if by == "type":
        names = np.where(conn.types == "", "?", conn.types)
    elif by == "class":
        names = np.array([f"{sc or '?'}:{c or '?'}" for sc, c in zip(conn.superclass, conn.cls)], dtype=object)
    else:
        raise ValueError("rules can be learned per 'type' or per 'class'")
    return np.array([f"{n}/{s or '-'}" for n, s in zip(names, conn.side)], dtype=object)


def learn_rules(conn: Connectome, by: str = "type") -> Rules:
    """Aggregate the wiring by (group, side) pairs: connection counts and the log-normal shape of
    the synapse counts. ``by="type"`` uses cell types (unannotated neurons form one group per side),
    ``by="class"`` the coarser superclass:class labels."""
    keys = _group_key(conn, by)
    groups, group_of = np.unique(keys, return_inverse=True)
    group_of = group_of.astype(np.int32)
    order = np.argsort(group_of, kind="stable")
    bounds = np.searchsorted(group_of[order], np.arange(groups.size + 1))
    pre_g = group_of[conn.pre_idx].astype(np.int64)
    post_g = group_of[conn.post_idx].astype(np.int64)
    pair_key = pre_g * groups.size + post_g
    uniq, inverse, counts = np.unique(pair_key, return_inverse=True, return_counts=True)
    logs = np.log(conn.n_syn.astype(np.float64))
    s1 = np.bincount(inverse, weights=logs, minlength=uniq.size)
    s2 = np.bincount(inverse, weights=logs * logs, minlength=uniq.size)
    mu = s1 / counts
    var = np.maximum(0.0, s2 / counts - mu * mu)
    return Rules(level=by, groups=groups.tolist(), group_of=group_of, order=order.astype(np.int64), bounds=bounds.astype(np.int64),
                 pair_pre=(uniq // groups.size).astype(np.int32), pair_post=(uniq % groups.size).astype(np.int32),
                 pair_edges=counts.astype(np.int64), pair_mu=mu.astype(np.float32), pair_sigma=np.sqrt(var).astype(np.float32),
                 n_edges_total=int(conn.n_edges), n_syn_total=int(conn.n_syn.sum()))


# ---------------------------------------------------------------- the bottleneck
def _spmat(rules: Rules):
    """The rule matrix M[g, h] = log1p(edges between groups g and h), as (rows, cols, vals)."""
    return rules.pair_pre.astype(np.int64), rules.pair_post.astype(np.int64), np.log1p(rules.pair_edges.astype(np.float64))


class _Sparse:
    """A sparse n x n matrix from coordinates, with fast M @ X and M.T @ X (segments + reduceat)."""

    def __init__(self, rows, cols, vals, n):
        self.n = n
        o = np.argsort(rows, kind="stable")
        self.r_rows, self.r_cols, self.r_vals = rows[o], cols[o], vals[o]
        self.r_starts = np.flatnonzero(np.r_[True, self.r_rows[1:] != self.r_rows[:-1]])
        self.r_ids = self.r_rows[self.r_starts]
        o = np.argsort(cols, kind="stable")
        self.c_rows, self.c_cols, self.c_vals = rows[o], cols[o], vals[o]
        self.c_starts = np.flatnonzero(np.r_[True, self.c_cols[1:] != self.c_cols[:-1]])
        self.c_ids = self.c_cols[self.c_starts]

    def _apply(self, idx_src, vals, starts, ids, X):
        out = np.zeros((self.n, X.shape[1]))
        for a in range(0, X.shape[1], 16):                   # a few columns at a time keeps memory small
            prod = vals[:, None] * X[idx_src, a:a + 16]
            out[ids, a:a + 16] = np.add.reduceat(prod, starts, axis=0)
        return out

    def dot(self, X):
        return self._apply(self.r_cols, self.r_vals, self.r_starts, self.r_ids, X)

    def tdot(self, X):
        return self._apply(self.c_rows, self.c_vals, self.c_starts, self.c_ids, X)


def _randomized_svd(rows, cols, vals, n, rank, seed, n_iter=3):
    """Top-``rank`` singular triplets of a sparse n x n matrix given as coordinates."""
    rng = np.random.default_rng(seed)
    p = min(n, rank + 12)
    M = _Sparse(rows, cols, vals, n)
    matmul, matmul_t = M.dot, M.tdot

    Q, _ = np.linalg.qr(matmul(rng.standard_normal((n, p))))
    for _ in range(n_iter):
        Q, _ = np.linalg.qr(matmul_t(Q))
        Q, _ = np.linalg.qr(matmul(Q))
    B = matmul_t(Q).T                     # p x n  (= Q.T M)
    Ub, S, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ Ub
    return U[:, :rank], S[:rank], Vt[:rank]


def bottleneck(rules: Rules, rank: int, seed: int = 0) -> Rules:
    """Approximate the type rules at ``rank``: every group gets a code of ``rank`` numbers for its
    outputs and one for its inputs, and a pair's connection strength is their dot product. The
    approximation is re-thresholded so the grown fly keeps the original number of connected pairs
    and connections; synapse counts per connection come from each presynaptic group's own
    distribution (``rank`` numbers per group are the "genome"; the pair table is gone)."""
    if rules.rank:
        raise ValueError("bottleneck() takes the full type rules")
    n = rules.n_groups
    rank = int(max(1, min(rank, n - 1)))
    rows, cols, vals = _spmat(rules)
    U, S, Vt = _randomized_svd(rows, cols, vals, n, rank, seed)
    A = (U * S).astype(np.float32)          # n x rank
    B = Vt.T.astype(np.float32)             # n x rank
    # per presynaptic group: synapse-count shape (a group-level parameter, kept outside the codes)
    g_s1 = np.bincount(rules.pair_pre, weights=rules.pair_mu * rules.pair_edges, minlength=n)
    g_w = np.bincount(rules.pair_pre, weights=rules.pair_edges.astype(np.float64), minlength=n)
    g_mu = np.where(g_w > 0, g_s1 / np.maximum(g_w, 1e-9), np.log(MIN_SYN)).astype(np.float32)
    g_s2 = np.bincount(rules.pair_pre, weights=(rules.pair_sigma ** 2 + rules.pair_mu ** 2) * rules.pair_edges, minlength=n)
    g_sigma = np.sqrt(np.maximum(0.0, np.where(g_w > 0, g_s2 / np.maximum(g_w, 1e-9), 0.0) - g_mu ** 2)).astype(np.float32)
    # reconstruct blockwise; find the threshold that keeps the original number of connected pairs
    target_pairs = rules.n_pairs
    block = max(64, min(2048, int(5e7 // max(1, n))))     # ~50 M floats per block at most
    hist_edges = np.linspace(0.0, float(np.log1p(rules.pair_edges.max())) + 1.0, 4097)
    hist = np.zeros(hist_edges.size - 1, dtype=np.int64)
    for a in range(0, n, block):
        R = A[a:a + block] @ B.T
        hist += np.histogram(R[R > hist_edges[0]], bins=hist_edges)[0]
    cum = np.cumsum(hist[::-1])[::-1]                   # pairs with value >= bin edge
    k = int(np.searchsorted(-cum, -target_pairs))       # first bin whose count drops below the target
    tau = float(hist_edges[min(k, hist_edges.size - 1)])
    tau = max(tau, float(np.log1p(0.05)))               # a blurred rule may mean "a connection now and then"
    pre_l, post_l, val_l = [], [], []
    for a in range(0, n, block):
        R = A[a:a + block] @ B.T
        r, c = np.nonzero(R >= tau)
        pre_l.append(r + a); post_l.append(c); val_l.append(R[r, c])
    pre = np.concatenate(pre_l).astype(np.int32); post = np.concatenate(post_l).astype(np.int32)
    val = np.concatenate(val_l).astype(np.float64)
    keep = pre != post                                  # a group may connect to itself only when its neurons differ
    keep |= (rules.bounds[pre + 1] - rules.bounds[pre]) > 1
    pre, post, val = pre[keep], post[keep], val[keep]
    edges = np.expm1(np.maximum(val, 0.0))
    cap = (rules.bounds[pre + 1] - rules.bounds[pre]).astype(np.float64) * (rules.bounds[post + 1] - rules.bounds[post])
    for _ in range(4):                                  # the same total number of connections, within each pair's capacity
        edges *= rules.n_edges_total / max(1.0, edges.sum())
        edges = np.minimum(edges, cap)
    out = Rules(level=f"bottleneck:{rank}", groups=rules.groups, group_of=rules.group_of, order=rules.order, bounds=rules.bounds,
                pair_pre=pre, pair_post=post, pair_edges=edges, pair_mu=g_mu[pre], pair_sigma=g_sigma[pre],
                n_edges_total=rules.n_edges_total, n_syn_total=rules.n_syn_total, rank=rank,
                notes={"threshold": round(tau, 4), "pairs_kept": int(pre.size), "singular_values": [round(float(s), 1) for s in S[:8]]})
    return out


# ---------------------------------------------------------------- growing a fly
def grow(conn: Connectome, rules: Rules, seed: int = 0) -> Connectome:
    """Wire the connectome's neurons afresh from the rules: for every connected group pair, the
    rules' number of connections is placed between random members of the two groups, each with a
    synapse count drawn from the pair's log-normal. Returns a new :class:`Connectome` sharing the
    neuron tables (same types, sides, somata) with its own wiring."""
    rng = np.random.default_rng(seed)
    n = conn.n
    edges = rules.pair_edges.astype(np.float64)
    G, H = rules.pair_pre.astype(np.int64), rules.pair_post.astype(np.int64)
    sz_g = rules.bounds[G + 1] - rules.bounds[G]
    sz_h = rules.bounds[H + 1] - rules.bounds[H]
    cap = sz_g * sz_h
    pre_l, post_l, pair_l = [], [], []
    # dense pairs (most of a small group pair is wired): decide every possible connection by a coin
    # with the pair's connection probability, so nothing is lost to repeated draws
    dense = np.flatnonzero((cap <= 4096) | (edges > 0.15 * cap))
    if dense.size:
        counts = cap[dense]
        for a in range(0, dense.size, 100000):
            d = dense[a:a + 100000]
            c = cap[d]
            slot_pair = np.repeat(d, c)
            slot = np.arange(int(c.sum())) - np.repeat(np.cumsum(c) - c, c)
            prob = (edges[d] / np.maximum(c, 1))[np.searchsorted(d, slot_pair)]
            hit = rng.random(slot.size) < prob
            slot_pair, slot = slot_pair[hit], slot[hit]
            g, h = G[slot_pair], H[slot_pair]
            pre_l.append(rules.order[rules.bounds[g] + slot // sz_h[slot_pair]])
            post_l.append(rules.order[rules.bounds[h] + slot % sz_h[slot_pair]])
            pair_l.append(slot_pair)
    # sparse pairs (a few connections among many possible): draw them with replacement
    sparse = np.setdiff1d(np.arange(rules.n_pairs), dense, assume_unique=True)
    if sparse.size:
        e = edges[sparse]
        e = np.floor(e + rng.random(e.size)).astype(np.int64)          # fractional counts: round at random
        pair = np.repeat(sparse, np.maximum(e, 0))
        g, h = G[pair], H[pair]
        pre_l.append(rules.order[rules.bounds[g] + (rng.random(pair.size) * sz_g[pair]).astype(np.int64)])
        post_l.append(rules.order[rules.bounds[h] + (rng.random(pair.size) * sz_h[pair]).astype(np.int64)])
        pair_l.append(pair)
    pre = np.concatenate(pre_l) if pre_l else np.zeros(0, np.int64)
    post = np.concatenate(post_l) if post_l else np.zeros(0, np.int64)
    pair = np.concatenate(pair_l) if pair_l else np.zeros(0, np.int64)
    keep = pre != post                                    # no neuron connects to itself
    pre, post, pair = pre[keep], post[keep], pair[keep]
    syn = np.exp(rng.normal(rules.pair_mu[pair].astype(np.float64), rules.pair_sigma[pair].astype(np.float64)))
    syn = np.clip(np.rint(syn), MIN_SYN, MAX_SYN)
    # merge duplicate (pre, post) draws by adding their synapses, then sort into CSR order
    key = pre * n + post
    uniq, inv = np.unique(key, return_inverse=True)
    syn_u = np.bincount(inv, weights=syn, minlength=uniq.size)
    pre_u = (uniq // n).astype(np.int32)
    post_u = (uniq % n).astype(np.int32)
    n_syn = np.clip(syn_u, MIN_SYN, MAX_SYN).astype(np.uint16)
    row_ptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(row_ptr, pre_u + 1, 1)
    row_ptr = np.cumsum(row_ptr)
    grown = conn.rewired(row_ptr, post_u, n_syn, label=f"grown:{rules.level}:seed{seed}")
    return grown


LEVELS = [
    ("real", "the real wiring: every neuron-to-neuron connection as reconstructed"),
    ("type", "rules per cell type and side: which types connect, how often and how strongly; the neurons are rewired at random within them"),
    ("bottleneck:256", "the type rules squeezed to a 256-number code per group"),
    ("bottleneck:64", "the type rules squeezed to a 64-number code per group"),
    ("bottleneck:16", "the type rules squeezed to a 16-number code per group"),
    ("class", "rules per class and side only: no cell-type identity at all"),
]


def valid_level(level: str) -> bool:
    level = str(level).strip().lower()
    if level in ("real", "type", "class"):
        return True
    if level.startswith("bottleneck:"):
        try:
            return 1 <= int(level.split(":", 1)[1]) <= 2048
        except ValueError:
            return False
    return False


def grow_level(conn: Connectome, level: str, seed: int = 0, rules_cache: dict | None = None) -> tuple[Connectome, Rules]:
    """``level`` is ``type``, ``class`` or ``bottleneck:K`` (``real`` returns the connectome itself)."""
    level = level.strip().lower()
    if level in ("real", "none", ""):
        return conn, None
    cache = rules_cache if rules_cache is not None else {}
    if level == "type" or level.startswith("bottleneck"):
        base = cache.get("type")
        if base is None:
            base = cache["type"] = learn_rules(conn, "type")
        rules = base
        if level.startswith("bottleneck"):
            rank = int(level.split(":")[1]) if ":" in level else 64
            rules = cache.get(f"bottleneck:{rank}")
            if rules is None:
                rules = cache[f"bottleneck:{rank}"] = bottleneck(base, rank, seed=0)
    elif level == "class":
        rules = cache.get("class")
        if rules is None:
            rules = cache["class"] = learn_rules(conn, "class")
    else:
        raise ValueError("level must be real, type, class or bottleneck:K")
    return grow(conn, rules, seed), rules


def compare(real: Connectome, grown: Connectome) -> dict:
    """How the grown wiring differs from the real one, at the neuron level."""
    a = set(zip(real.pre_idx.tolist(), real.post_idx.tolist())) if real.n_edges < 3_000_000 else None
    out = {"edges_real": int(real.n_edges), "edges_grown": int(grown.n_edges),
           "synapses_real": int(real.n_syn.sum()), "synapses_grown": int(grown.n_syn.sum())}
    if a is not None:
        b = set(zip(grown.pre_idx.tolist(), grown.post_idx.tolist()))
        out["shared_connections"] = len(a & b)
    else:                                                   # too big for sets: sample
        rng = np.random.default_rng(0)
        k = rng.choice(real.n_edges, 200_000, replace=False)
        keys_real = real.pre_idx[k].astype(np.int64) * real.n + real.post_idx[k]
        # sorted unique keys and a binary search: np.unique / np.isin on these 6 M keys took 6 s each (NumPy 2.4)
        keys_grown = np.sort(grown.pre_idx.astype(np.int64) * grown.n + grown.post_idx)
        keys_grown = keys_grown[np.concatenate(([True], keys_grown[1:] != keys_grown[:-1]))]
        if keys_grown.size:
            pos = np.minimum(np.searchsorted(keys_grown, keys_real), keys_grown.size - 1)
            out["shared_connections_fraction"] = float((keys_grown[pos] == keys_real).mean())
        else:
            out["shared_connections_fraction"] = 0.0
    return out
