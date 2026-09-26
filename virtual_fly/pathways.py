"""
Tracing pathways through the wiring diagram.

"How does a small moving object seen by the right eye end up steering the fly?" is a question
about the *graph*, not about the simulation, and it can be answered directly from the synapse
counts. :func:`trace` searches the cell-type-level graph for the strongest chains of connections
from one population to another, scoring each hop by the fraction of the target type's input that
comes from the source type (the usual "input fraction" measure used in connectomics). A path's
score is the product of its hops, so a chain of strong connections beats a chain of weak ones,
and the sign of every neuron along the way tells you whether the signal arrives excitatory or
inhibitory (an even number of inhibitory neurons flips it back).

This is a static analysis: it says which routes *exist* and how strong they are, not which ones
carry activity in a given simulation. It favours short chains of strong connections; a signal that
travels over many weak parallel routes (sugar taste to MN9 is one) shows up as several low-scoring
paths such as ``LB3c -> GNG232 (G2N-1) -> DNge080 (Roundup) -> MN9``. Use :func:`lesion_scan` in
:mod:`virtual_fly.experiments` to test which routes actually carry the signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .connectome import Connectome, TypeGraph


@dataclass
class Hop:
    src: str
    dst: str
    synapses: int
    fraction: float          # of the target's total input synapses
    sign: int                # +1 if the source type is excitatory, -1 if inhibitory
    src_size: int
    dst_size: int


@dataclass
class Path:
    nodes: list[str]
    hops: list[Hop]
    score: float             # product of hop fractions (0..1)
    net_sign: int            # +1 = the source excites the target along this route, -1 = inhibits

    def describe(self) -> str:
        parts = [self.nodes[0]]
        for h in self.hops:
            arrow = "-(+)->" if h.sign > 0 else "-(-)->"
            parts.append(f"{arrow} {h.dst} [{h.synapses} syn, {100 * h.fraction:.1f}% of input]")
        tag = "net excitatory" if self.net_sign > 0 else "net inhibitory"
        return f"{' '.join(parts)}   score {self.score:.4f}, {tag}"

    def to_dict(self) -> dict:
        return {"nodes": self.nodes, "score": self.score, "net_sign": self.net_sign,
                "hops": [h.__dict__ for h in self.hops]}


def trace(conn: Connectome, src_spec: str, dst_spec: str, max_hops: int = 4, beam: int = 500,
          top: int = 8, min_fraction: float = 0.002, excitatory_only: bool = False,
          avoid: str | None = None) -> list[Path]:
    """Find the strongest pathways from one population to another.

    ``src_spec`` / ``dst_spec`` are population specs (see :meth:`Connectome.select`); the search
    runs on the cell-type graph, so ``"LC10a/L"`` means the type LC10a on the left. Up to
    ``max_hops`` connections are followed with a beam search keeping the ``beam`` best partial
    paths; hops carrying less than ``min_fraction`` of the target's input are ignored.
    ``excitatory_only`` drops inhibitory intermediates; ``avoid`` is a spec of types to route around
    (to see what remains after a lesion).
    """
    tg: TypeGraph = conn.type_graph()
    src_nodes = tg.nodes_of(src_spec)
    dst_nodes = set(tg.nodes_of(dst_spec).tolist())
    if src_nodes.size == 0:
        raise ValueError(f"no neurons match '{src_spec}'")
    if not dst_nodes:
        raise ValueError(f"no neurons match '{dst_spec}'")
    avoid_nodes = set(tg.nodes_of(avoid).tolist()) if avoid else set()
    # partial paths: (score, nodes list, sign)
    frontier = [(1.0, [int(s)], 1) for s in src_nodes if int(s) not in avoid_nodes]
    found: list[tuple[float, list[int], int]] = []
    seen_final = set()
    for hop in range(max_hops):
        candidates = []
        for score, nodes, sign in frontier:
            last = nodes[-1]
            if excitatory_only and hop > 0 and tg.sign[last] < 0:
                continue
            dst, w = tg.out(last)
            if dst.size == 0:
                continue
            frac = w / np.maximum(tg.in_total[dst], 1.0)
            ok = frac >= min_fraction
            if not ok.any():
                continue
            s_sign = int(tg.sign[last])
            for d, f in zip(dst[ok].tolist(), frac[ok].tolist()):
                if d in nodes or d in avoid_nodes:
                    continue
                new_score = score * f
                new_nodes = nodes + [d]
                new_sign = sign * s_sign
                if d in dst_nodes:
                    key = tuple(new_nodes)
                    if key not in seen_final:
                        seen_final.add(key)
                        found.append((new_score, new_nodes, new_sign))
                else:
                    candidates.append((new_score, new_nodes, new_sign))
        candidates.sort(key=lambda c: -c[0])
        frontier = candidates[:beam]
        if not frontier:
            break
    found.sort(key=lambda c: -c[0])
    paths = []
    for score, nodes, sign in found[:top]:
        hops = []
        for a, b in zip(nodes[:-1], nodes[1:]):
            dst, w = tg.out(a)
            k = int(np.flatnonzero(dst == b)[0])
            hops.append(Hop(tg.names[a], tg.names[b], int(w[k]), float(w[k] / max(tg.in_total[b], 1.0)),
                            int(tg.sign[a]), int(tg.size[a]), int(tg.size[b])))
        paths.append(Path([tg.names[i] for i in nodes], hops, float(score), int(sign)))
    return paths


def relay_ranking(paths: list[Path]) -> list[tuple[str, float]]:
    """Intermediate types ranked by how much path score flows through them (lesion candidates)."""
    score: dict[str, float] = {}
    for p in paths:
        for node in p.nodes[1:-1]:
            score[node] = score.get(node, 0.0) + p.score
    return sorted(score.items(), key=lambda kv: -kv[1])


def strongest_partners(conn: Connectome, spec: str, direction: str = "out", top: int = 15,
                       by_side: bool = True) -> list[dict]:
    """Convenience wrapper: the strongest input or output types of a population as dicts. ``fraction`` is a
    share of the partner's own traffic: for ``direction="out"`` the share of the partner's input that comes
    from the population, for ``direction="in"`` the share of the partner's output that goes to it."""
    rows = conn.outputs_of(spec, top=top, by_side=by_side) if direction == "out" else \
        conn.inputs_of(spec, top=top, by_side=by_side)
    tg = conn.type_graph()
    for r in rows:
        if r["side"]:
            nodes = [tg.index[n] for n in (f"{r['type']}/{r['side']}",) if n in tg.index]
        else:                                    # summed over sides
            nodes = [tg.index[n] for n in tg.index if n == r["type"] or n.startswith(r["type"] + "/")]
        totals = tg.in_total if direction == "out" else tg.out_total
        total = float(sum(totals[n] for n in nodes))
        r["fraction"] = float(r["synapses"] / total) if total > 0 else None
    return rows
