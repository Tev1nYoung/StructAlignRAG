from __future__ import annotations

from collections import deque
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

from ..config import StructAlignRAGConfig
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


Adj = Dict[str, List[Tuple[str, float, str]]]


def _allowed_edge_types(config: StructAlignRAGConfig) -> Set[str]:
    allowed = {"mentions", "in_passage", "in_doc", "entails", "contradicts"}
    if bool(getattr(config, "ppr_allow_sim_edges", False)):
        allowed.add("sim")
    return allowed


def _edge_strength(config: StructAlignRAGConfig, typ: str, cost: float) -> float:
    eps = float(getattr(config, "ppr_eps", 1e-6) or 1e-6)
    typ = str(typ)
    if typ == "mentions":
        s = float(getattr(config, "ppr_strength_mentions", 1.0))
    elif typ == "in_passage":
        s = float(getattr(config, "ppr_strength_in_passage", 1.0))
    elif typ == "in_doc":
        s = float(getattr(config, "ppr_strength_in_doc", 0.7))
    elif typ == "entails":
        s = float(getattr(config, "ppr_strength_entails", 0.5))
    elif typ == "contradicts":
        s = float(getattr(config, "ppr_strength_contradicts", 0.0))
    elif typ == "sim":
        s = float(getattr(config, "ppr_strength_sim", 0.2))
    else:
        s = 0.0
    if s <= 0.0:
        return 0.0
    c = float(cost) if cost is not None else 0.0
    return float(s) * (1.0 / (eps + max(0.0, c)))


def induce_subgraph_nodes(
    *,
    adj: Adj,
    seed_nodes: Sequence[str],
    config: StructAlignRAGConfig,
) -> Set[str]:
    """
    BFS k-hop induced subgraph on selected edge types.
    """
    hops = int(getattr(config, "ppr_hops", 2) or 2)
    max_nodes = int(getattr(config, "ppr_max_nodes", 5000) or 5000)
    allowed = _allowed_edge_types(config)

    induced: Set[str] = set()
    q = deque()
    for s in seed_nodes:
        s = str(s)
        if not s:
            continue
        if s not in adj:
            continue
        if s in induced:
            continue
        induced.add(s)
        q.append((s, 0))
        if len(induced) >= max_nodes:
            return induced

    while q:
        u, d = q.popleft()
        if d >= hops:
            continue
        for v, cost, typ in adj.get(u, []):
            typ = str(typ)
            if typ not in allowed:
                continue
            if _edge_strength(config, typ, float(cost)) <= 0.0:
                continue
            v = str(v)
            if not v:
                continue
            if v in induced:
                continue
            induced.add(v)
            if len(induced) >= max_nodes:
                return induced
            q.append((v, d + 1))

    return induced


def mini_ppr(
    *,
    adj: Adj,
    nodes: Sequence[str],
    seed_weights: Dict[str, float],
    config: StructAlignRAGConfig,
) -> Dict[str, float]:
    """
    Sparse power-iteration personalized PageRank on a small induced subgraph.
    """
    steps = int(getattr(config, "ppr_steps", 10) or 10)
    alpha = float(getattr(config, "ppr_alpha", 0.85) or 0.85)
    alpha = max(0.0, min(1.0, alpha))
    allowed = _allowed_edge_types(config)

    idx = {str(n): i for i, n in enumerate(nodes)}
    n = len(nodes)
    if n == 0:
        return {}

    # Seed vector
    s = [0.0] * n
    for node, w in (seed_weights or {}).items():
        i = idx.get(str(node))
        if i is None:
            continue
        try:
            wf = float(w)
        except Exception:
            wf = 0.0
        if wf > 0.0:
            s[i] += wf
    ssum = sum(s)
    if ssum <= 0.0:
        # Uniform over all nodes as a last resort.
        s = [1.0 / n] * n
    else:
        inv = 1.0 / ssum
        s = [x * inv for x in s]

    # Precompute transitions
    nbrs: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    for u, uidx in idx.items():
        outs: List[Tuple[int, float]] = []
        strengths = []
        for v, cost, typ in adj.get(u, []):
            typ = str(typ)
            if typ not in allowed:
                continue
            v = str(v)
            vidx = idx.get(v)
            if vidx is None:
                continue
            st = _edge_strength(config, typ, float(cost))
            if st <= 0.0:
                continue
            outs.append((vidx, st))
            strengths.append(st)

        if not outs:
            continue
        norm = sum(strengths)
        if norm <= 0.0:
            continue
        inv = 1.0 / norm
        nbrs[uidx] = [(vidx, st * inv) for vidx, st in outs]

    # Power iteration
    p = list(s)
    for _t in range(max(1, steps)):
        newp = [(1.0 - alpha) * si for si in s]
        for uidx in range(n):
            pu = p[uidx]
            if pu <= 0.0:
                continue
            outs = nbrs[uidx]
            if not outs:
                # Dangling mass goes back to seed distribution.
                for i in range(n):
                    newp[i] += alpha * pu * s[i]
                continue
            for vidx, prob in outs:
                newp[vidx] += alpha * pu * prob
        p = newp

    return {str(nodes[i]): float(p[i]) for i in range(n) if p[i] > 0.0}


def rrf_from_scores(
    *,
    scores: Dict[str, float],
    prefix: str,
    rrf_k: int,
    pool: int,
) -> Dict[str, float]:
    items = [(n, float(s)) for n, s in (scores or {}).items() if str(n).startswith(prefix)]
    items.sort(key=lambda x: x[1], reverse=True)
    out: Dict[str, float] = {}
    for r, (n, _s) in enumerate(items[: max(0, int(pool))]):
        out[str(n)] = 1.0 / float(int(rrf_k) + int(r) + 1)
    return out


def run_local_propagation_rrf(
    *,
    adj: Adj,
    seed_capsules: Sequence[str],
    seed_entities: Sequence[str],
    config: StructAlignRAGConfig,
) -> Dict[str, Any]:
    """
    Query-level local propagation:
    - induce a small subgraph around seeds
    - run mini-PPR
    - convert to RRF weights for robust fusion
    """
    seed_capsules = [str(x) for x in (seed_capsules or []) if str(x)]
    seed_entities = [str(x) for x in (seed_entities or []) if str(x)]
    seed_nodes = list(dict.fromkeys(seed_capsules + seed_entities))  # stable dedup
    if not seed_nodes:
        return {
            "enabled": False,
            "num_nodes": 0,
            "seed_capsules": [],
            "seed_entities": [],
            "capsule_rrf": {},
            "doc_rrf_nodes": {},
            "passage_rrf": {},
            "top_capsules": [],
            "top_docs": [],
            "top_passage_ids": [],
        }

    induced = induce_subgraph_nodes(adj=adj, seed_nodes=seed_nodes, config=config)
    nodes = sorted(induced)

    # Seed weights: capsules slightly heavier than entities by default.
    seed_w: Dict[str, float] = {}
    for i, n in enumerate(seed_capsules):
        seed_w[n] = seed_w.get(n, 0.0) + 2.0
    for i, n in enumerate(seed_entities):
        seed_w[n] = seed_w.get(n, 0.0) + 1.0

    scores = mini_ppr(adj=adj, nodes=nodes, seed_weights=seed_w, config=config)

    rrf_k = int(getattr(config, "ppr_rrf_k", 60) or 60)
    pool = int(getattr(config, "ppr_rrf_pool", 200) or 200)
    capsule_rrf = rrf_from_scores(scores=scores, prefix="C:", rrf_k=rrf_k, pool=pool)
    doc_rrf_nodes = rrf_from_scores(scores=scores, prefix="D:", rrf_k=rrf_k, pool=pool)
    passage_rrf = rrf_from_scores(scores=scores, prefix="P:", rrf_k=rrf_k, pool=pool)

    # Top lists for debug (use raw PPR score).
    top_caps = sorted([(n, float(s)) for n, s in scores.items() if n.startswith("C:")], key=lambda x: x[1], reverse=True)[:20]
    top_docs_raw = sorted([(n, float(s)) for n, s in scores.items() if n.startswith("D:")], key=lambda x: x[1], reverse=True)[:20]
    top_pass_raw = sorted([(n, float(s)) for n, s in scores.items() if n.startswith("P:")], key=lambda x: x[1], reverse=True)[:50]

    top_doc_pairs: List[Tuple[int, float]] = []
    for n, s in top_docs_raw:
        try:
            didx = int(str(n).split(":", 1)[1])
        except Exception:
            continue
        top_doc_pairs.append((didx, float(s)))

    top_passage_ids: List[str] = []
    for n, _s in top_pass_raw:
        pid = str(n).split(":", 1)[1] if ":" in str(n) else ""
        if pid:
            top_passage_ids.append(pid)

    return {
        "enabled": True,
        "num_nodes": int(len(nodes)),
        "seed_capsules": seed_capsules[:50],
        "seed_entities": [e.split(":", 1)[1] if e.startswith("E:") else e for e in seed_entities[:50]],
        "capsule_rrf": capsule_rrf,
        "doc_rrf_nodes": doc_rrf_nodes,
        "passage_rrf": passage_rrf,
        "top_capsules": top_caps,
        "top_docs": top_doc_pairs,
        "top_passage_ids": top_passage_ids,
    }

