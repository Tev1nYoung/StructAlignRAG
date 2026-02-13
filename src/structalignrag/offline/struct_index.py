from __future__ import annotations

import json
import os
import pickle
from collections import defaultdict
from typing import Any, Dict, List, Tuple


def build_struct_index(
    docs: List[Dict[str, Any]],
    passages: List[Dict[str, Any]],
    entities: List[Dict[str, Any]],
    canonical_capsules: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build lightweight structural indices for online retrieval/search:
    - doc_idx -> passage_ids
    - passage_id -> doc_idx
    - passage_id -> canonical_ids
    - canonical_id -> passage_ids
    - entity_id -> canonical_ids

    This is intentionally minimal and dependency-free (pickle/json).
    """
    doc_to_passages: Dict[int, List[str]] = defaultdict(list)
    passage_to_doc: Dict[str, int] = {}
    for p in passages:
        doc_idx = int(p.get("doc_idx", 0))
        pid = str(p.get("passage_id"))
        if not pid:
            continue
        doc_to_passages[doc_idx].append(pid)
        passage_to_doc[pid] = doc_idx

    passage_to_canonical: Dict[str, List[str]] = defaultdict(list)
    canonical_to_passages: Dict[str, List[str]] = defaultdict(list)
    entity_to_canonical: Dict[str, List[str]] = defaultdict(list)

    for c in canonical_capsules:
        ccid = str(c.get("canonical_id"))
        if not ccid:
            continue
        for eid in c.get("entity_ids") or []:
            if eid:
                entity_to_canonical[str(eid)].append(ccid)
        seen_p = set()
        for prov in c.get("provenance") or []:
            pid = str((prov or {}).get("passage_id") or "")
            if not pid or pid in seen_p:
                continue
            seen_p.add(pid)
            passage_to_canonical[pid].append(ccid)
            canonical_to_passages[ccid].append(pid)

    # Dedup + sort deterministically
    for k in list(doc_to_passages.keys()):
        doc_to_passages[k] = sorted(set(doc_to_passages[k]))
    for k in list(passage_to_canonical.keys()):
        passage_to_canonical[k] = sorted(set(passage_to_canonical[k]))
    for k in list(canonical_to_passages.keys()):
        canonical_to_passages[k] = sorted(set(canonical_to_passages[k]))
    for k in list(entity_to_canonical.keys()):
        entity_to_canonical[k] = sorted(set(entity_to_canonical[k]))

    # Entity alias -> id map (surface forms)
    alias_to_entity_id: Dict[str, str] = {}
    for e in entities:
        eid = str(e.get("entity_id") or "")
        for a in e.get("aliases") or []:
            a = str(a or "").strip()
            if not a:
                continue
            alias_to_entity_id[a] = eid

    return {
        "doc_to_passages": dict(doc_to_passages),
        "passage_to_doc": passage_to_doc,
        "passage_to_canonical_capsules": dict(passage_to_canonical),
        "canonical_capsule_to_passages": dict(canonical_to_passages),
        "entity_to_canonical_capsules": dict(entity_to_canonical),
        "entity_alias_to_id": alias_to_entity_id,
        "stats": {
            "num_docs": len(docs),
            "num_passages": len(passages),
            "num_entities": len(entities),
            "num_canonical_capsules": len(canonical_capsules),
        },
    }


def save_struct_index(struct_index: Dict[str, Any], out_dir: str) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    pkl_path = os.path.join(out_dir, "struct_index.pkl")
    alias_path = os.path.join(out_dir, "entity_alias_to_id.json")

    with open(pkl_path, "wb") as f:
        pickle.dump(struct_index, f)

    with open(alias_path, "w", encoding="utf-8") as f:
        json.dump(struct_index.get("entity_alias_to_id", {}), f, ensure_ascii=False, indent=2)

    return pkl_path, alias_path

