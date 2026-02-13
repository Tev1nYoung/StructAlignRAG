from __future__ import annotations

import json
import os
import pickle
from typing import Any, Dict, List, Set, Tuple

import faiss
import numpy as np
from tqdm import tqdm

from ..config import StructAlignRAGConfig
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


EdgeRow = Dict[str, Any]


def _add_edge(adj: Dict[str, List[Tuple[str, float, str]]], edges: List[EdgeRow], u: str, v: str, typ: str, weight: float, meta: Dict[str, Any] | None = None) -> None:
    adj.setdefault(u, []).append((v, float(weight), typ))
    adj.setdefault(v, []).append((u, float(weight), typ))
    row: EdgeRow = {"src": u, "dst": v, "type": typ, "weight": float(weight)}
    if meta:
        row.update(meta)
    edges.append(row)


def _extract_json_object(text: str) -> Dict[str, Any]:
    import json as _json
    import re as _re

    if not text:
        raise ValueError("empty response")
    t = str(text).strip()
    if "```" in t:
        blocks = _re.findall(r"```(?:json)?\\s*(\\{.*?\\})\\s*```", t, flags=_re.DOTALL)
        if blocks:
            t = max(blocks, key=len).strip()
    start = t.find("{")
    end = t.rfind("}")
    if start >= 0 and end > start:
        t = t[start : end + 1]
    return _json.loads(t)


def _build_nli_edges(
    config: StructAlignRAGConfig,
    canonical_capsules: List[Dict[str, Any]],
    canonical_embeddings: np.ndarray,
    llm,
) -> List[EdgeRow]:
    """
    Optional (cached) LLM-based NLI edges between canonical capsules.

    We restrict candidate pairs using ANN cosine similarity and shared entities,
    and cap the total LLM calls with `nli_max_pairs`.
    """
    if (not config.enable_nli_edges) or llm is None:
        return []
    if canonical_embeddings is None or len(canonical_capsules) <= 1:
        return []

    emb = np.ascontiguousarray(canonical_embeddings.astype(np.float32))
    d = emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(emb)

    # Candidate neighbors (cheap, ANN exact IP here). We further filter by sim threshold.
    k = min(max(2, int(config.sim_edge_topk or 8)), len(canonical_capsules))
    sims, nbrs = index.search(emb, k)

    entity_sets = [set(c.get("entity_ids") or []) for c in canonical_capsules]
    texts = [str(c.get("text") or "").strip() for c in canonical_capsules]

    pairs: List[Tuple[float, int, int]] = []
    added = set()
    for i in range(len(canonical_capsules)):
        for rank in range(1, k):
            j = int(nbrs[i, rank])
            if j < 0 or j == i:
                continue
            if j < i:
                continue
            sim = float(sims[i, rank])
            if sim < float(config.nli_min_sim):
                continue
            if entity_sets[i] and entity_sets[j] and not (entity_sets[i] & entity_sets[j]):
                continue
            key = (i, j)
            if key in added:
                continue
            added.add(key)
            pairs.append((sim, i, j))

    pairs.sort(key=lambda x: -x[0])
    if config.nli_max_pairs is not None and int(config.nli_max_pairs) > 0:
        pairs = pairs[: int(config.nli_max_pairs)]

    if not pairs:
        return []

    system = (
        "You are a natural language inference (NLI) classifier. "
        "Output strict JSON only. Be conservative: if unsure, output neutral."
    )

    out: List[EdgeRow] = []
    pbar = tqdm(pairs, desc="NLI Edges", disable=False, ascii=True, dynamic_ncols=True)
    for sim, i, j in pbar:
        a = texts[i]
        b = texts[j]
        if not a or not b:
            continue
        if a == b:
            # Equivalent statements by string identity: add an entail edge without LLM.
            out.append(
                {
                    "src": f"C:{canonical_capsules[i]['canonical_id']}",
                    "dst": f"C:{canonical_capsules[j]['canonical_id']}",
                    "type": "entails",
                    "weight": 0.1,
                    "sim": float(sim),
                    "label": "equivalent",
                    "confidence": 1.0,
                    "cache_hit": True,
                }
            )
            continue

        user = (
            f"Statement A:\n{a}\n\n"
            f"Statement B:\n{b}\n\n"
            "Return JSON:\n"
            "{\n"
            '  \"label\": \"equivalent|entails|entailed_by|contradicts|neutral\",\n'
            '  \"confidence\": 0.0\n'
            "}\n\n"
            "Rules:\n"
            "- equivalent: A and B mean the same fact.\n"
            "- entails: A implies B (but not vice versa).\n"
            "- entailed_by: B implies A (but not vice versa).\n"
            "- contradicts: A and B cannot both be true.\n"
            "- neutral: none of the above.\n"
            "- If unsure, choose neutral."
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        try:
            try:
                raw, meta = llm.infer(messages=messages, temperature=0.0, response_format={"type": "json_object"})
            except Exception:
                raw, meta = llm.infer(messages=messages, temperature=0.0)
            obj = _extract_json_object(raw)
            label = str(obj.get("label") or "neutral").strip().lower()
            conf = obj.get("confidence")
            try:
                conf_f = float(conf)
            except Exception:
                conf_f = 0.0
            cache_hit = bool((meta or {}).get("cache_hit"))
        except Exception:
            label = "neutral"
            conf_f = 0.0
            cache_hit = False

        if label == "neutral":
            continue

        typ = "contradicts" if label == "contradicts" else "entails"
        w = 1.0 if typ == "contradicts" else 0.2

        out.append(
            {
                "src": f"C:{canonical_capsules[i]['canonical_id']}",
                "dst": f"C:{canonical_capsules[j]['canonical_id']}",
                "type": typ,
                "weight": float(w),
                "sim": float(sim),
                "label": label,
                "confidence": float(conf_f),
                "cache_hit": cache_hit,
            }
        )

    return out


def build_evidence_graph(
    config: StructAlignRAGConfig,
    docs: List[Dict[str, Any]],
    passages: List[Dict[str, Any]],
    entities: List[Dict[str, Any]],
    canonical_capsules: List[Dict[str, Any]],
    canonical_embeddings: np.ndarray,
    out_dir: str,
    llm=None,
) -> Dict[str, Any]:
    """
    Builds:
    - graph_edges.jsonl
    - graph_adj.pkl
    """
    os.makedirs(out_dir, exist_ok=True)
    edge_path = os.path.join(out_dir, "graph_edges.jsonl")
    adj_path = os.path.join(out_dir, "graph_adj.pkl")

    adj: Dict[str, List[Tuple[str, float, str]]] = {}
    edges: List[EdgeRow] = []

    # Node init (optional; adjacency can be implicit)
    for d in docs:
        adj.setdefault(f"D:{int(d['doc_idx'])}", [])
    for p in passages:
        adj.setdefault(f"P:{p['passage_id']}", [])
    for e in entities:
        adj.setdefault(f"E:{e['entity_id']}", [])
    for c in canonical_capsules:
        adj.setdefault(f"C:{c['canonical_id']}", [])

    # Passage -> Doc edges
    for p in passages:
        _add_edge(adj, edges, f"P:{p['passage_id']}", f"D:{int(p['doc_idx'])}", "in_doc", config.w_passage_doc)

    # Capsule -> Passage / Entity edges
    for c in canonical_capsules:
        cid = f"C:{c['canonical_id']}"
        # provenance passages
        seen_p: Set[str] = set()
        for prov in c.get("provenance") or []:
            pid = prov.get("passage_id")
            if not pid:
                continue
            if pid in seen_p:
                continue
            seen_p.add(pid)
            _add_edge(adj, edges, cid, f"P:{pid}", "in_passage", config.w_capsule_passage)

        for eid in c.get("entity_ids") or []:
            _add_edge(adj, edges, cid, f"E:{eid}", "mentions", config.w_capsule_entity)

    # Similarity edges between canonical capsules
    if canonical_embeddings is not None and len(canonical_capsules) > 1 and config.sim_edge_topk > 0:
        emb = np.ascontiguousarray(canonical_embeddings.astype(np.float32))
        d = emb.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(emb)

        k = min(int(config.sim_edge_topk), len(canonical_capsules))
        sims, nbrs = index.search(emb, k)

        added = set()
        pbar = tqdm(range(len(canonical_capsules)), desc="Sim Edges", disable=False, ascii=True, dynamic_ncols=True)
        for i in pbar:
            ci = canonical_capsules[i]["canonical_id"]
            for rank in range(1, k):
                j = int(nbrs[i, rank])
                if j < 0 or j == i:
                    continue
                sim = float(sims[i, rank])
                if sim < float(config.sim_edge_threshold):
                    continue
                cj = canonical_capsules[j]["canonical_id"]
                a = (ci, cj) if ci < cj else (cj, ci)
                if a in added:
                    continue
                added.add(a)
                # Sim edges are intentionally "costly" to avoid drift in dense entity graphs.
                # Keep a mostly-constant traversal cost; store the actual similarity in edge metadata.
                w = float(config.w_sim_base)
                _add_edge(adj, edges, f"C:{ci}", f"C:{cj}", "sim", w, meta={"sim": sim})
            pbar.set_postfix({"edges": len(added)})

    # Optional NLI edges (cached)
    nli_rows = _build_nli_edges(config=config, canonical_capsules=canonical_capsules, canonical_embeddings=canonical_embeddings, llm=llm)
    if nli_rows:
        added = set()
        for r in nli_rows:
            u = str(r["src"])
            v = str(r["dst"])
            typ = str(r["type"])
            key = (u, v, typ) if u < v else (v, u, typ)
            if key in added:
                continue
            added.add(key)
            _add_edge(adj, edges, u, v, typ, float(r.get("weight", 0.0)), meta={k: v for k, v in r.items() if k not in ("src", "dst", "type", "weight")})

    # Deterministic adjacency ordering
    for u in list(adj.keys()):
        adj[u].sort(key=lambda x: (x[0], x[2], x[1]))

    # Persist
    edges.sort(key=lambda r: (str(r.get("src")), str(r.get("dst")), str(r.get("type"))))
    with open(edge_path, "w", encoding="utf-8") as f:
        for row in edges:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(adj_path, "wb") as f:
        pickle.dump(adj, f)

    logger.info(f"[StructAlignRAG] [OFFLINE_GRAPH] written | edges={len(edges)} path={edge_path}")
    return {"edge_path": edge_path, "adj_path": adj_path, "num_edges": len(edges), "num_nodes": len(adj)}
