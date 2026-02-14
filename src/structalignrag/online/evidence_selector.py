from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..config import StructAlignRAGConfig
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def _approx_tokens(text: str) -> int:
    # Cheap budget proxy (MVP). Replace with a tokenizer if needed.
    return len((text or "").split())


@dataclass
class _PassageCand:
    pid: str
    row: int
    doc_idx: int
    toks: int
    dense: float
    hits: Dict[str, float]  # gid -> best prize covered by this passage


def greedy_select_evidence_passages(
    *,
    config: StructAlignRAGConfig,
    passages: Sequence[Dict[str, Any]],
    pid_to_row: Dict[str, int],
    candidate_pids: Set[str],
    passage_sim_max: np.ndarray,
    passage_sim_anchor: Optional[np.ndarray] = None,
    passage_to_canonical_capsules: Optional[Dict[str, List[str]]],
    group_prize_maps: Dict[str, Dict[str, float]],
    group_ids: Sequence[str],
    seed_pids: Optional[Sequence[str]] = None,
    group_weights: Optional[Dict[str, float]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Coverage-aware greedy evidence packing.

    Returns (selected_passages, debug).
    """
    if not candidate_pids:
        return [], {"pool_size": 0, "selected_pids": [], "selected_tokens": 0}

    budget = int(getattr(config, "passage_token_budget", 0) or 0)
    max_n = int(getattr(config, "qa_top_k_passages", 5) or 5)

    dense_w = float(getattr(config, "evidence_dense_w", 0.25))
    gain_w = float(getattr(config, "evidence_gain_w", 0.75))
    same_doc_pen = float(getattr(config, "evidence_same_doc_penalty", 0.10))
    len_pen = float(getattr(config, "evidence_len_penalty", 0.0))

    group_ids = [str(g) for g in (group_ids or [])]
    group_w = {str(g): float(group_weights.get(str(g), 1.0)) for g in group_ids} if isinstance(group_weights, dict) else {str(g): 1.0 for g in group_ids}
    prize_maps = {str(g): (group_prize_maps.get(str(g)) or {}) for g in group_ids}

    # Precompute candidate features once.
    cands: List[_PassageCand] = []
    for pid in sorted(candidate_pids):
        row = pid_to_row.get(pid)
        if row is None:
            continue
        if row < 0 or row >= len(passages):
            continue
        p = passages[int(row)] or {}
        doc_idx = int(p.get("doc_idx") or 0)
        toks = int(p.get("token_count") or _approx_tokens(str(p.get("text") or "")))
        if passage_sim_anchor is not None:
            try:
                dense = float(passage_sim_anchor[int(row)])
            except Exception:
                dense = 0.0
        else:
            dense = float(passage_sim_max[int(row)]) if passage_sim_max is not None else 0.0

        hits: Dict[str, float] = {}
        if passage_to_canonical_capsules:
            ccids = passage_to_canonical_capsules.get(str(pid)) or []
            if ccids:
                for ccid in ccids:
                    node_id = f"C:{ccid}"
                    for gid in group_ids:
                        pm = prize_maps.get(gid) or {}
                        if node_id in pm:
                            prev = hits.get(gid, 0.0)
                            hits[gid] = max(prev, float(pm[node_id]))

        cands.append(_PassageCand(pid=str(pid), row=int(row), doc_idx=doc_idx, toks=toks, dense=dense, hits=hits))

    if not cands:
        return [], {"pool_size": 0, "selected_pids": [], "selected_tokens": 0}

    # Greedy packing under token budget.
    best_gid: Dict[str, float] = {gid: 0.0 for gid in group_ids}
    selected: List[_PassageCand] = []
    selected_pids: Set[str] = set()
    selected_doc_counts: Dict[int, int] = {}
    total_tokens = 0
    total_gain = 0.0
    seed_pids_used: List[str] = []

    # If budget is disabled (<=0), treat as very large.
    if budget <= 0:
        budget = 10**9

    def _gain(c: _PassageCand) -> float:
        g = 0.0
        for gid, hit in c.hits.items():
            prev = best_gid.get(gid, 0.0)
            if hit > prev:
                g += float(hit - prev) * float(group_w.get(gid, 1.0))
        return g

    # Seed with forced passages (e.g., top-doc best passages) to keep evidence answer-oriented.
    pid_to_cand = {c.pid: c for c in cands}
    for pid in list(seed_pids or []):
        pid = str(pid)
        c = pid_to_cand.get(pid)
        if c is None:
            continue
        if c.pid in selected_pids:
            continue
        if total_tokens + int(c.toks) > budget:
            continue
        selected.append(c)
        selected_pids.add(c.pid)
        selected_doc_counts[int(c.doc_idx)] = int(selected_doc_counts.get(int(c.doc_idx), 0)) + 1
        total_tokens += int(c.toks)
        seed_pids_used.append(c.pid)
        for gid, hit in c.hits.items():
            if hit > best_gid.get(gid, 0.0):
                best_gid[gid] = float(hit)
        if len(selected) >= max_n:
            break

    for _step in range(max_n):
        if len(selected) >= max_n:
            break
        best = None
        best_obj = -1e18
        best_gain = 0.0
        for c in cands:
            if c.pid in selected_pids:
                continue
            if total_tokens + int(c.toks) > budget:
                continue
            g = _gain(c)
            doc_pen = same_doc_pen * float(selected_doc_counts.get(int(c.doc_idx), 0))
            obj = dense_w * float(c.dense) + gain_w * float(g) - float(doc_pen) - len_pen * float(c.toks)
            if obj > best_obj:
                best_obj = obj
                best = c
                best_gain = g

        if best is None:
            break

        # Select
        selected.append(best)
        selected_pids.add(best.pid)
        selected_doc_counts[int(best.doc_idx)] = int(selected_doc_counts.get(int(best.doc_idx), 0)) + 1
        total_tokens += int(best.toks)
        total_gain += float(best_gain)

        # Update covered-best prizes
        for gid, hit in best.hits.items():
            if hit > best_gid.get(gid, 0.0):
                best_gid[gid] = float(hit)

        if len(selected) >= max_n:
            break

    # If we didn't fill enough passages, top-up by dense similarity (max over variants).
    if len(selected) < max_n:
        remaining = [c for c in cands if c.pid not in selected_pids]
        remaining.sort(key=lambda x: x.dense, reverse=True)
        for c in remaining:
            if len(selected) >= max_n:
                break
            if total_tokens + int(c.toks) > budget:
                continue
            selected.append(c)
            selected_pids.add(c.pid)
            selected_doc_counts[int(c.doc_idx)] = int(selected_doc_counts.get(int(c.doc_idx), 0)) + 1
            total_tokens += int(c.toks)

    out_passages: List[Dict[str, Any]] = [dict(passages[int(c.row)]) for c in selected]
    dbg = {
        "pool_size": int(len(cands)),
        "seed_pids_used": list(seed_pids_used),
        "selected_pids": [c.pid for c in selected],
        "selected_doc_idxs": [int(c.doc_idx) for c in selected],
        "selected_tokens": int(total_tokens),
        "selected_gain": float(total_gain),
        "budget": int(budget),
    }
    return out_passages, dbg
