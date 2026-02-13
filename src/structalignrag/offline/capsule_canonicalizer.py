from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np

from ..config import StructAlignRAGConfig
from ..utils.union_find import UnionFind


def _dedup_provenance(prov: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for p in prov or []:
        if not isinstance(p, dict):
            continue
        key = (
            str(p.get("doc_id") or ""),
            int(p.get("doc_idx") or 0),
            str(p.get("passage_id") or ""),
            int(p.get("sent_id") or 0),
            tuple(p.get("span") or []),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    out.sort(
        key=lambda x: (
            int(x.get("doc_idx") or 0),
            str(x.get("passage_id") or ""),
            int(x.get("sent_id") or 0),
            tuple(x.get("span") or []),
        )
    )
    return out


def canonicalize_capsules(
    capsules: List[Dict[str, Any]],
    capsule_embeddings: np.ndarray,
    config: StructAlignRAGConfig,
) -> Tuple[List[Dict[str, Any]], Dict[str, str], np.ndarray, List[str]]:
    """
    Returns:
    - canonical_capsules (list)
    - capsule_id -> canonical_id mapping
    - canonical_embeddings (float32, normalized)
    - canonical_ids (list aligned with canonical_embeddings rows)
    """
    if not capsules:
        return [], {}, np.zeros((0, 1), dtype=np.float32), []

    n, d = capsule_embeddings.shape
    if n != len(capsules):
        raise ValueError("capsule_embeddings length mismatch")

    if not config.enable_capsule_canonicalization:
        canonical_ids = [f"CC_{i:07d}" for i in range(len(capsules))]
        cap2can = {capsules[i]["capsule_id"]: canonical_ids[i] for i in range(len(capsules))}
        can_emb = capsule_embeddings.copy()
        # already normalized
        canonical_capsules = []
        for i, cap in enumerate(capsules):
            ccid = canonical_ids[i]
            canonical_capsules.append(
                {
                    "canonical_id": ccid,
                    "text": cap.get("canonical_text") or cap.get("text") or "",
                    "predicate": cap.get("predicate"),
                    "polarity": cap.get("polarity"),
                    "arguments": cap.get("arguments") or [],
                    "entity_ids": cap.get("entity_ids") or [],
                    "provenance": cap.get("provenance") or [],
                    "member_capsule_ids": [cap["capsule_id"]],
                    "stats": {"freq": 1},
                }
            )
        return canonical_capsules, cap2can, can_emb, canonical_ids

    # FAISS index on normalized vectors => inner product == cosine sim.
    emb = np.ascontiguousarray(capsule_embeddings.astype(np.float32))
    index = faiss.IndexFlatIP(d)
    index.add(emb)

    uf = UnionFind(n)
    top_m = max(2, int(config.capsule_ann_top_m))
    sims, nbrs = index.search(emb, top_m)

    entity_sets = [set(c.get("entity_ids") or []) for c in capsules]
    preds = [str(c.get("predicate") or "").strip().lower() for c in capsules]
    pols = [str(c.get("polarity") or "affirm").strip().lower() for c in capsules]

    for i in range(n):
        for rank in range(1, top_m):
            j = int(nbrs[i, rank])
            if j < 0 or j == i:
                continue
            sim = float(sims[i, rank])
            if sim < float(config.capsule_sim_threshold):
                continue
            if j < i:
                continue

            # Guardrails: shared entities + polarity match.
            if entity_sets[i] and entity_sets[j]:
                if not (entity_sets[i] & entity_sets[j]):
                    continue
            else:
                # If no entities, be conservative (do not merge).
                continue

            if pols[i] != pols[j]:
                continue

            # Predicate: require match or very high sim.
            if preds[i] and preds[j] and preds[i] != preds[j] and sim < 0.93:
                continue

            uf.union(i, j)

    root_to_ids: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        root_to_ids[uf.find(i)].append(i)

    canonical_capsules: List[Dict[str, Any]] = []
    cap2can: Dict[str, str] = {}
    canonical_ids: List[str] = []
    canonical_embs: List[np.ndarray] = []

    for cluster_idx, ids in enumerate(root_to_ids.values()):
        ccid = f"CC_{cluster_idx:07d}"
        canonical_ids.append(ccid)

        # Representative capsule: choose the one with most provenance (or first).
        rep = sorted(
            ids,
            key=lambda k: (
                -len(capsules[k].get("provenance") or []),
                str(capsules[k].get("capsule_id") or f"{k:08d}"),
            ),
        )[0]

        prov = []
        ent_ids = set()
        for k in ids:
            prov.extend(capsules[k].get("provenance") or [])
            for e in capsules[k].get("entity_ids") or []:
                ent_ids.add(e)
            cap2can[capsules[k]["capsule_id"]] = ccid

        prov = _dedup_provenance(prov)

        # Mean embedding for cluster
        vec = emb[ids].mean(axis=0)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        canonical_embs.append(vec.astype(np.float32))

        canonical_capsules.append(
            {
                "canonical_id": ccid,
                "text": (capsules[rep].get("text") or capsules[rep].get("canonical_text") or "").strip(),
                "predicate": capsules[rep].get("predicate"),
                "polarity": capsules[rep].get("polarity"),
                "arguments": capsules[rep].get("arguments") or [],
                "entity_ids": sorted(ent_ids),
                "provenance": prov,
                "member_capsule_ids": sorted([capsules[k]["capsule_id"] for k in ids]),
                "stats": {"freq": len(ids), "num_provenance": len(prov)},
            }
        )

    can_emb = np.stack(canonical_embs, axis=0) if canonical_embs else np.zeros((0, d), dtype=np.float32)
    return canonical_capsules, cap2can, can_emb, canonical_ids
