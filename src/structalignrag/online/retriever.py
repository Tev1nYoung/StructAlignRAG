from __future__ import annotations

import ast
import heapq
import json
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ..config import StructAlignRAGConfig
from ..utils.logging_utils import get_logger
from ..utils.text_utils import extract_entity_mentions, normalize_entity
from ..utils.genericness_utils import title_genericness_score

logger = get_logger(__name__)


def _safe_topk(scores: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.array([], dtype=np.int64)
    k = min(k, scores.shape[0])
    if k == scores.shape[0]:
        return np.argsort(-scores)
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx


def _approx_tokens(text: str) -> int:
    # Cheap budget proxy (MVP). Replace with tiktoken if needed.
    return len((text or "").split())


def _parse_json_object(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("empty response")
    raw = str(text).strip()

    if "```" in raw:
        blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
        if blocks:
            raw = max(blocks, key=len).strip()

    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]

    variants = [
        raw,
        re.sub(r",\s*([}\]])", r"\1", raw),
        re.sub(r"(\])\s*(\"selected\"\s*:)", r"\1,\n\2", raw),
        re.sub(r"(\])\s*(\"selected_doc_indices\"\s*:)", r"\1,\n\2", raw),
    ]
    last_err: Exception | None = None
    for v in variants:
        try:
            obj = json.loads(v)
            if isinstance(obj, dict):
                return obj
        except Exception as e:
            last_err = e
        try:
            v_py = v
            v_py = re.sub(r"\bnull\b", "None", v_py, flags=re.IGNORECASE)
            v_py = re.sub(r"\btrue\b", "True", v_py, flags=re.IGNORECASE)
            v_py = re.sub(r"\bfalse\b", "False", v_py, flags=re.IGNORECASE)
            obj = ast.literal_eval(v_py)
            if isinstance(obj, dict):
                return obj
        except Exception as e:
            last_err = e

    raise last_err or ValueError("failed to parse JSON object")


def _build_parent_map(query_dag: Dict[str, Any], node_ids: Set[str]) -> Dict[str, Set[str]]:
    """
    Return child_id -> set(parent_ids).
    Prefer node.depends_on, fall back to query_dag.edges if present.
    """
    parents: Dict[str, Set[str]] = {nid: set() for nid in node_ids}

    # 1) depends_on from nodes (preferred)
    for n in query_dag.get("nodes") or []:
        cid = str(n.get("id") or "")
        if not cid or cid not in parents:
            continue
        for pid in n.get("depends_on") or []:
            pid = str(pid)
            if pid in parents and pid != cid:
                parents[cid].add(pid)

    # 2) edges fallback (only for missing depends_on)
    if any(parents[cid] for cid in parents):
        return parents

    for e in query_dag.get("edges") or []:
        src = dst = None
        if isinstance(e, dict):
            src = e.get("source") or e.get("from") or e.get("src")
            dst = e.get("target") or e.get("to") or e.get("dst")
        elif isinstance(e, (list, tuple)) and len(e) >= 2:
            src, dst = e[0], e[1]
        if src is None or dst is None:
            continue
        src = str(src)
        dst = str(dst)
        if src in parents and dst in parents and src != dst:
            parents[dst].add(src)

    return parents


def _topo_sort(node_ids: List[str], parents: Dict[str, Set[str]]) -> List[str]:
    """
    Kahn topo sort. If cycle exists, return original order (stable).
    """
    indeg = {nid: 0 for nid in node_ids}
    children: Dict[str, List[str]] = defaultdict(list)
    for child, ps in parents.items():
        for p in ps:
            if p in indeg and child in indeg:
                indeg[child] += 1
                children[p].append(child)

    q = deque([nid for nid in node_ids if indeg.get(nid, 0) == 0])
    out: List[str] = []
    while q:
        u = q.popleft()
        out.append(u)
        for v in children.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    if len(out) != len(node_ids):
        return list(node_ids)
    return out


def _entity_overlap(a: Set[str], b: Set[str]) -> int:
    if not a or not b:
        return 0
    # Use smaller set for efficiency.
    if len(a) > len(b):
        a, b = b, a
    return sum(1 for x in a if x in b)


def dijkstra_path(
    adj: Dict[str, List[Tuple[str, float, str]]],
    src: str,
    dst: str,
    allowed: Set[str],
) -> Optional[List[str]]:
    if src == dst:
        return [src]
    if src not in allowed or dst not in allowed:
        return None

    dist: Dict[str, float] = {src: 0.0}
    prev: Dict[str, str] = {}
    pq: List[Tuple[float, str]] = [(0.0, src)]
    visited: Set[str] = set()

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == dst:
            break
        for v, w, _typ in adj.get(u, []):
            if v not in allowed:
                continue
            nd = d + float(w)
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if dst not in prev and dst != src:
        return None

    # reconstruct
    path = [dst]
    cur = dst
    while cur != src:
        cur = prev.get(cur)
        if cur is None:
            return None
        path.append(cur)
    path.reverse()
    return path


@dataclass
class RetrievalResult:
    retrieved_docs: List[str]
    selected_passages: List[Dict[str, Any]]
    debug: Dict[str, Any]


class StructAlignRetriever:
    def __init__(self, config: StructAlignRAGConfig) -> None:
        self.config = config

    def retrieve(
        self,
        question: str,
        query_dag: Dict[str, Any],
        index: Dict[str, Any],
        embedder,
        llm=None,
    ) -> RetrievalResult:
        t0 = time.time()

        docs: List[Dict[str, Any]] = index["docs"]
        passages: List[Dict[str, Any]] = index["passages"]
        entities: List[Dict[str, Any]] = index["entities"]
        canonical_capsules: List[Dict[str, Any]] = index["canonical_capsules"]
        passage_emb: np.ndarray = index["passage_emb"]
        canonical_capsule_emb: np.ndarray = index["canonical_capsule_emb"]
        adj = index["adj"]

        doc_idx_to_doc = {int(d["doc_idx"]): d for d in docs}
        doc_title_norm_to_idx: Dict[str, int] = {}
        for d in docs:
            t = str(d.get("title") or "").strip()
            if not t:
                continue
            doc_title_norm_to_idx.setdefault(normalize_entity(t), int(d["doc_idx"]))

        # Title genericness stats (unsupervised, corpus-derived).
        title_token_idf: Dict[str, float] = index.get("title_token_idf") or {}
        title_idf_p10 = float(index.get("title_idf_p10") or 0.0)
        title_idf_p90 = float(index.get("title_idf_p90") or 6.0)
        doc_genericness: Dict[int, float] = index.get("doc_genericness") or {}
        title_norm_to_genericness: Dict[str, float] = index.get("title_norm_to_genericness") or {}
        generic_skip_th = float(getattr(self.config, "genericness_skip_threshold", 0.85))

        passage_id_to_passage = {p["passage_id"]: p for p in passages}
        can_id_to_cap = {c["canonical_id"]: c for c in canonical_capsules}
        cap_node_to_entset: Dict[str, Set[str]] = {}
        cap_node_to_docset: Dict[str, Set[int]] = {}
        for c in canonical_capsules:
            cid = str(c.get("canonical_id"))
            if not cid:
                continue
            cap_node_to_entset[f"C:{cid}"] = set(str(e) for e in (c.get("entity_ids") or []) if e)
            docset: Set[int] = set()
            for prov in c.get("provenance") or []:
                didx = (prov or {}).get("doc_idx")
                if didx is None:
                    continue
                docset.add(int(didx))
            cap_node_to_docset[f"C:{cid}"] = docset

        # entity alias map: alias/canonical_name -> entity_id
        ent_norm_to_id: Dict[str, str] = {}
        ent_id_to_canon: Dict[str, str] = {}
        for e in entities:
            eid = str(e.get("entity_id"))
            canon = str(e.get("canonical_name") or "")
            if canon:
                ent_id_to_canon[eid] = canon
                ent_norm_to_id.setdefault(normalize_entity(canon), eid)
            for a in e.get("aliases") or []:
                a = str(a or "").strip()
                if not a:
                    continue
                ent_norm_to_id.setdefault(normalize_entity(a), eid)

        # Always include the original question as a retrieval anchor group.
        # Query-DAG nodes are often "plan-like" and can be under-specified without executing intermediate answers.
        raw_nodes = query_dag.get("nodes") or []
        has_anchor = False
        q_norm = (question or "").strip().lower()
        for n in raw_nodes:
            nq = str((n or {}).get("question") or "").strip().lower()
            if nq and nq == q_norm:
                has_anchor = True
                break
        if has_anchor:
            nodes = list(raw_nodes)
        else:
            used_ids = {str((n or {}).get("id") or "") for n in raw_nodes}
            anchor_id = "q0" if "q0" not in used_ids else "q_orig"
            nodes = [{"id": anchor_id, "question": question, "depends_on": [], "operator": "other", "vars_in": [], "vars_out": [], "constraints": {}}] + list(raw_nodes)

        # Embed once per query (question + all subQs) for speed and to avoid repeated GPU calls.
        # This does not change embeddings (same model, same texts), only reduces overhead.
        texts_to_embed = [question] + [str((n or {}).get("question") or question) for n in nodes]
        q_subq_emb = embedder.encode(texts_to_embed, instruction=str(getattr(self.config, "embedding_query_instruction", "") or ""))
        q_emb = q_subq_emb[0]
        subq_emb_rows = q_subq_emb[1:]
        p_sims = passage_emb @ q_emb
        dense_doc_score: Dict[int, float] = {}
        doc_best_passage_row: Dict[int, int] = {}
        doc_best_passage_sim: Dict[int, float] = {}
        for i, p in enumerate(passages):
            didx = p.get("doc_idx")
            if didx is None:
                continue
            d = int(didx)
            s = float(p_sims[int(i)])
            prev = dense_doc_score.get(d)
            if prev is None or s > prev:
                dense_doc_score[d] = s
            prevp = doc_best_passage_sim.get(d)
            if prevp is None or s > prevp:
                doc_best_passage_sim[d] = s
                doc_best_passage_row[d] = int(i)
        dense_doc_rank = [d for d, _s in sorted(dense_doc_score.items(), key=lambda x: x[1], reverse=True)]

        group_candidates: List[Dict[str, Any]] = []
        group_prize_maps: Dict[str, Dict[str, float]] = {}
        group_seed_nodes: Dict[str, List[str]] = {}
        group_subq_emb: Dict[str, np.ndarray] = {}

        # Build candidate groups
        for ni, n in enumerate(nodes):
            gid = str(n.get("id") or "q")
            subq = str(n.get("question") or question)
            subq_emb = subq_emb_rows[ni] if ni < len(subq_emb_rows) else q_emb
            group_subq_emb[gid] = subq_emb
            sims = canonical_capsule_emb @ subq_emb
            top_idx = _safe_topk(sims, min(self.config.subq_top_capsule, len(canonical_capsules)))

            # entity bonus
            sub_mentions = extract_entity_mentions(subq)
            sub_ent_ids = set()
            for m in sub_mentions:
                norm = normalize_entity(m)
                if norm in ent_norm_to_id:
                    sub_ent_ids.add(ent_norm_to_id[norm])

            cands = []
            prize_map: Dict[str, float] = {}
            for ci in top_idx:
                c = canonical_capsules[int(ci)]
                node_id = f"C:{c['canonical_id']}"
                sim = float(sims[int(ci)])
                prize = sim
                if sub_ent_ids and any(eid in sub_ent_ids for eid in c.get("entity_ids", [])):
                    prize += self.config.bonus_entity_match
                prize_map[node_id] = prize
                doc_indices = sorted(set(int((prov or {}).get("doc_idx")) for prov in (c.get("provenance") or []) if (prov or {}).get("doc_idx") is not None))
                cands.append(
                    {
                        "group_id": gid,
                        "subq": subq,
                        "canonical_index": int(ci),
                        "node_id": node_id,
                        "sim": sim,
                        "prize": prize,
                        "entity_ids": sorted(set(str(e) for e in (c.get("entity_ids") or []) if e)),
                        "doc_indices": doc_indices,
                    }
                )

            # sort by prize desc
            cands.sort(key=lambda x: x["prize"], reverse=True)
            group_candidates.append({"group_id": gid, "subq": subq, "candidates": cands})
            group_prize_maps[gid] = prize_map
            group_seed_nodes[gid] = [x["node_id"] for x in cands[: self.config.seed_top_s]]

        node_ids = [str(g["group_id"]) for g in group_candidates]

        # Multi-query dense doc ranks (question + subQs) and RRF fusion.
        group_dense_doc_rank: Dict[str, List[int]] = {}
        group_doc_best_passage_row: Dict[str, Dict[int, int]] = {}
        group_doc_best_passage_sim: Dict[str, Dict[int, float]] = {}
        for gid, emb in group_subq_emb.items():
            try:
                sims_g = passage_emb @ emb
            except Exception:
                continue
            doc_score_g: Dict[int, float] = {}
            doc_best_row_g: Dict[int, int] = {}
            for i, p in enumerate(passages):
                didx = p.get("doc_idx")
                if didx is None:
                    continue
                d = int(didx)
                s = float(sims_g[int(i)])
                prev = doc_score_g.get(d)
                if prev is None or s > prev:
                    doc_score_g[d] = s
                    doc_best_row_g[d] = int(i)
            group_dense_doc_rank[gid] = [d for d, _s in sorted(doc_score_g.items(), key=lambda x: x[1], reverse=True)]
            group_doc_best_passage_row[gid] = doc_best_row_g
            group_doc_best_passage_sim[gid] = doc_score_g

        rrf_k = int(getattr(self.config, "doc_rrf_k", 60))
        rrf_pool = int(getattr(self.config, "doc_rrf_pool", 120))
        rrf_scores: Dict[int, float] = defaultdict(float)
        # Include the original question rank + each subQ rank (duplicates are fine: RRF sums evidence).
        rank_lists: List[List[int]] = [dense_doc_rank]
        for gid in node_ids:
            if gid in group_dense_doc_rank:
                rank_lists.append(group_dense_doc_rank[gid])
        for rlist in rank_lists:
            for r, d in enumerate(rlist[: max(1, rrf_pool)]):
                rrf_scores[int(d)] += 1.0 / float(rrf_k + r + 1)

        # DAG-aware binding assignment (zero-shot variable binding via entity overlap).
        parents = _build_parent_map(query_dag, set(node_ids))
        topo = _topo_sort(node_ids, parents)

        # Beam search over the DAG order.
        # State: (score, assignment_dict, used_docs, used_caps)
        beam: List[Tuple[float, Dict[str, str], Set[int], Set[str]]] = [(0.0, {}, set(), set())]
        group_to_cands: Dict[str, List[Dict[str, Any]]] = {g["group_id"]: g["candidates"] for g in group_candidates}

        for gid in topo:
            cands = group_to_cands.get(gid) or []
            if not cands:
                # No candidates: propagate as-is.
                beam = [(s, dict(a, **{gid: "C:__none__"}), set(used_docs), set(used_caps)) for s, a, used_docs, used_caps in beam]
                continue

            cands = cands[: max(1, int(self.config.binding_candidate_k))]
            new_beam: List[Tuple[float, Dict[str, str], Set[int], Set[str]]] = []
            pgids = sorted(parents.get(gid) or [])

            for score, assign, used_docs, used_caps in beam:
                parent_nodes = [assign.get(p) for p in pgids if p in assign]
                parent_entsets = [cap_node_to_entset.get(n, set()) for n in parent_nodes if n and n.startswith("C:")]

                for c in cands:
                    nid = str(c["node_id"])
                    base = float(c["prize"])

                    # Structural alignment bonus/penalty: prefer candidates sharing entities with parents.
                    overlap = 0
                    if parent_entsets:
                        cent = cap_node_to_entset.get(nid, set())
                        for pe in parent_entsets:
                            overlap += _entity_overlap(cent, pe)
                        if overlap > 0:
                            base += float(self.config.binding_overlap_bonus) * float(overlap)
                        else:
                            base -= float(self.config.binding_no_overlap_penalty)

                    # Encourage multi-doc coverage (multi-hop) and avoid degenerate "all groups -> same doc".
                    cand_docs = cap_node_to_docset.get(nid, set())
                    if cand_docs:
                        novel = len(cand_docs - used_docs)
                        repeat = len(cand_docs & used_docs)
                        base += float(getattr(self.config, "binding_doc_diversity_bonus", 0.0)) * float(novel)
                        base -= float(getattr(self.config, "binding_repeat_doc_penalty", 0.0)) * float(repeat)

                    # Soft diversity over capsules (avoid picking the exact same capsule for every group).
                    if nid in used_caps:
                        base -= float(getattr(self.config, "binding_repeat_capsule_penalty", 0.0))

                    new_assign = dict(assign)
                    new_assign[gid] = nid
                    new_used_docs = set(used_docs)
                    new_used_docs.update(cand_docs)
                    new_used_caps = set(used_caps)
                    new_used_caps.add(nid)
                    new_beam.append((score + base, new_assign, new_used_docs, new_used_caps))

            # Keep top beam states.
            new_beam.sort(key=lambda x: x[0], reverse=True)
            beam = new_beam[: max(1, int(self.config.binding_beam_size))]

        best_score, chosen_assign, _used_docs_best, _used_caps_best = max(beam, key=lambda x: x[0])

        # Induced nodes set
        induced: Set[str] = set()
        for g in group_candidates:
            # Keep induced graph compact: only the most promising candidates per group.
            keep_n = max(int(self.config.binding_candidate_k), int(self.config.seed_top_s))
            for x in g["candidates"][:keep_n]:
                induced.add(x["node_id"])
                cap = can_id_to_cap.get(x["node_id"].split(":", 1)[1])
                if not cap:
                    continue
                for ent_id in cap.get("entity_ids", []):
                    induced.add(f"E:{ent_id}")
                for prov in cap.get("provenance") or []:
                    pid = prov.get("passage_id")
                    didx = prov.get("doc_idx")
                    if pid:
                        induced.add(f"P:{pid}")
                    if didx is not None:
                        induced.add(f"D:{int(didx)}")

                # Optional: include sim neighbors to allow structure alignment to use sim edges.
                if self.config.neighbor_expand_sim_top_m and self.config.neighbor_expand_sim_top_m > 0:
                    neighbors = [t for t in adj.get(x["node_id"], []) if t[2] == "sim"]
                    neighbors = sorted(neighbors, key=lambda t: t[1])[: int(self.config.neighbor_expand_sim_top_m)]
                    for v, _w, _typ in neighbors:
                        induced.add(v)

        # Cap induced size (MVP: keep as-is; too large is unlikely with current defaults)
        if len(induced) > self.config.induced_max_nodes:
            # Keep only top candidates per group.
            keep: Set[str] = set()
            for g in group_candidates:
                for x in g["candidates"][: max(10, self.config.seed_top_s)]:
                    keep.add(x["node_id"])
                    cap = can_id_to_cap.get(x["node_id"].split(":", 1)[1])
                    if not cap:
                        continue
                    for ent_id in cap.get("entity_ids", []):
                        keep.add(f"E:{ent_id}")
                    for prov in cap.get("provenance") or []:
                        pid = prov.get("passage_id")
                        didx = prov.get("doc_idx")
                        if pid:
                            keep.add(f"P:{pid}")
                        if didx is not None:
                            keep.add(f"D:{int(didx)}")
            induced = keep

        chosen: Dict[str, str] = dict(chosen_assign)
        # Root: pick a root group (no parents) with highest per-group prize.
        roots = [gid for gid in node_ids if not (parents.get(gid) or set())]
        if not roots:
            roots = list(node_ids)

        def _group_prize(gid: str) -> float:
            nid = chosen.get(gid) or "C:__none__"
            return float(group_prize_maps.get(gid, {}).get(nid, 0.0))

        root_gid = max(roots, key=_group_prize) if roots else (node_ids[0] if node_ids else "q0")
        root = chosen.get(root_gid, "C:__none__")

        selected: Set[str] = set()
        for _gid, nid in chosen.items():
            if nid and nid != "C:__none__":
                selected.add(nid)

        # Add multiple high-prize seeds per group to avoid degenerate "single capsule" selection,
        # improving multi-hop doc coverage (critical for Recall@2/5).
        for g in group_candidates:
            for x in g.get("candidates", [])[: int(self.config.seed_top_s)]:
                nid = str(x.get("node_id") or "")
                if nid and nid != "C:__none__":
                    selected.add(nid)

        # Connect along DAG dependencies (better than a single root-star for multi-hop binding).
        any_edge = False
        for child, ps in parents.items():
            for p in ps:
                any_edge = True
                src = chosen.get(p)
                dst = chosen.get(child)
                if not src or not dst or src == "C:__none__" or dst == "C:__none__":
                    continue
                if src == dst:
                    continue
                path = dijkstra_path(adj, src, dst, induced)
                if path is None:
                    continue
                for u in path:
                    selected.add(u)

        # Fallback: if DAG has no edges (or all empty), connect to root to allow expansion.
        if not any_edge and root and root != "C:__none__":
            selected.add(root)
            for gid, nid in chosen.items():
                if not nid or nid == "C:__none__" or nid == root:
                    continue
                path = dijkstra_path(adj, root, nid, induced)
                if path is None:
                    continue
                for u in path:
                    selected.add(u)

        # Coverage: any group has at least one selected candidate
        covered = 0
        for g in group_candidates:
            gid = g["group_id"]
            ok = False
            for x in g["candidates"][: self.config.seed_top_s]:
                if x["node_id"] in selected:
                    ok = True
                    break
            if ok:
                covered += 1
        subq_coverage = covered / max(len(group_candidates), 1)

        # Score docs based on selected capsules prizes
        doc_score: Dict[int, float] = {}
        entity_score: Dict[str, float] = {}
        selected_caps = [n for n in selected if n.startswith("C:")]
        for cap_node in selected_caps:
            can_id = cap_node.split(":", 1)[1]
            cap = can_id_to_cap.get(can_id)
            if not cap:
                continue
            best_prize = 0.0
            for pm in group_prize_maps.values():
                if cap_node in pm:
                    best_prize = max(best_prize, pm[cap_node])
            for prov in cap.get("provenance") or []:
                didx = prov.get("doc_idx")
                if didx is None:
                    continue
                d = int(didx)
                doc_score[d] = max(doc_score.get(d, 0.0), best_prize)

            for eid in cap.get("entity_ids") or []:
                if eid:
                    entity_score[str(eid)] = max(entity_score.get(str(eid), 0.0), best_prize)

        # Blend in dense passage-aggregated doc signal (robustness when capsules are sparse/noisy).
        dense_w = float(getattr(self.config, "doc_dense_score_weight", 0.9))
        for d, s in dense_doc_score.items():
            ds = max(0.0, dense_w * float(s))
            doc_score[int(d)] = max(doc_score.get(int(d), 0.0), ds)

        # Multi-query RRF doc fusion (question + subQs) boosts Recall@2/5 for multi-hop.
        rrf_w = float(getattr(self.config, "doc_rrf_weight", 0.0))
        if rrf_w > 0.0 and rrf_scores:
            for d, s in rrf_scores.items():
                doc_score[int(d)] = float(doc_score.get(int(d), 0.0)) + rrf_w * float(s)

        # Entity-jump: if an entity mentioned in high-prize capsules matches a doc title in the corpus,
        # boost that doc. This is a cheap zero-shot way to recover second-hop pages.
        jump_top_m = int(getattr(self.config, "entity_jump_top_m", 0))
        jump_bonus = float(getattr(self.config, "entity_jump_bonus", 0.0))
        jump_doc_bonus: Dict[int, float] = {}
        if jump_top_m > 0 and jump_bonus > 0.0 and entity_score:
            for eid, s in sorted(entity_score.items(), key=lambda x: x[1], reverse=True)[: max(1, jump_top_m)]:
                name = ent_id_to_canon.get(eid) or ""
                if not name:
                    continue
                didx = doc_title_norm_to_idx.get(normalize_entity(name))
                if didx is None:
                    continue
                g = float(doc_genericness.get(int(didx), title_genericness_score(name, title_token_idf, title_idf_p10, title_idf_p90)))
                if g >= generic_skip_th:
                    continue
                bonus = jump_bonus * float(s)
                doc_score[int(didx)] = float(doc_score.get(int(didx), 0.0)) + bonus
                jump_doc_bonus[int(didx)] = max(jump_doc_bonus.get(int(didx), 0.0), float(bonus))

        # Penalize overly-generic titles (corpus-derived genericness), without hardcoded blocklists.
        # Important: only apply after a high threshold to avoid harming normal entity pages that happen
        # to have low title-IDF (e.g., "United Kingdom", "USB").
        if bool(getattr(self.config, "enable_genericness_penalty", True)) and doc_score and doc_genericness:
            w = float(getattr(self.config, "genericness_penalty_weight", 0.0))
            th = float(getattr(self.config, "genericness_penalty_threshold", generic_skip_th))
            th = max(0.0, min(1.0, th))
            if w > 0.0 and th < 1.0:
                denom = max(1e-6, 1.0 - th)
                for d, g in doc_genericness.items():
                    didx = int(d)
                    if didx not in doc_score:
                        continue
                    gg = float(g)
                    if gg < th:
                        continue
                    # Scale to [0,1] above threshold so the maximum penalty is exactly `w`.
                    scaled = (gg - th) / denom
                    doc_score[didx] = float(doc_score.get(didx, 0.0)) - w * float(scaled)

        # Doc-mention jump: use top-ranked docs as seeds, extract TitleCase mentions from their passages,
        # and boost docs whose titles match those mentions. This recovers second-hop entity pages like
        # "Stephen Warbeck" from a "Billy Elliot" seed without any training.
        if bool(getattr(self.config, "enable_doc_mention_jump", False)) and doc_score:
            try:
                seed_n = int(getattr(self.config, "doc_mention_jump_from_top_n", 2))
                top_m = int(getattr(self.config, "doc_mention_jump_top_m", 20))
                bonus = float(getattr(self.config, "doc_mention_jump_bonus", 0.35))
                seed_n = max(1, min(seed_n, 10))
                top_m = max(1, min(top_m, 80))
                if bonus > 0.0:
                    score_rank0 = [int(d) for d, _s in sorted(doc_score.items(), key=lambda x: x[1], reverse=True)]
                    seeds = score_rank0[:seed_n] if score_rank0 else []
                    for sd in seeds:
                        row = doc_best_passage_row.get(int(sd))
                        if row is None:
                            continue
                        seed_text = str((passages[int(row)] or {}).get("text") or "")
                        seed_score = max(0.0, float(doc_score.get(int(sd), 0.0)))
                        if not seed_text or seed_score <= 0.0:
                            continue
                        mentions = extract_entity_mentions(seed_text)[:top_m]
                        for m in mentions:
                            norm = normalize_entity(m)
                            if not norm:
                                continue
                            didx = doc_title_norm_to_idx.get(norm)
                            if didx is None:
                                continue
                            didx = int(didx)
                            if didx == int(sd):
                                continue
                            # Skip boosting generic/disambiguation/list pages.
                            g = float(doc_genericness.get(didx, title_norm_to_genericness.get(norm, 0.0)))
                            if g >= generic_skip_th:
                                continue
                            doc_score[didx] = float(doc_score.get(didx, 0.0)) + bonus * seed_score
            except Exception:
                pass

        # Optional diversify: greedy cover subQs early.
        struct_docs_rank: List[int] = []
        if self.config.doc_rank_diversify and group_candidates:
            doc_to_groups: Dict[int, Set[str]] = defaultdict(set)
            for g in group_candidates:
                gid = str(g.get("group_id") or "")
                for x in (g.get("candidates") or [])[: int(self.config.seed_top_s)]:
                    nid = str(x.get("node_id") or "")
                    if not nid or nid == "C:__none__":
                        continue
                    cap = can_id_to_cap.get(nid.split(":", 1)[1])
                    if not cap:
                        continue
                    for prov in cap.get("provenance") or []:
                        didx = (prov or {}).get("doc_idx")
                        if didx is None:
                            continue
                        doc_to_groups[int(didx)].add(gid)

            # Also use dense per-subQ ranks to drive early group coverage.
            group_topk = int(getattr(self.config, "doc_group_top_k", min(40, len(docs))))
            for gid in node_ids:
                rlist = group_dense_doc_rank.get(gid) or []
                for d in rlist[: max(1, group_topk)]:
                    doc_to_groups[int(d)].add(gid)

            # Score-first, diversify-second:
            # - Keep the best scoring doc as rank-1 (protect Recall@1)
            # - Only diversify within a high-score pool for the next few slots (targeting Recall@2/5)
            score_rank = [int(d) for d, _s in sorted(doc_score.items(), key=lambda x: x[1], reverse=True)]
            pool_size = int(getattr(self.config, "doc_diversify_pool", 200))
            diversify_n = int(getattr(self.config, "doc_diversify_n", 5))
            gain_w = float(getattr(self.config, "doc_diversify_gain_weight", 0.06))
            pool = set(score_rank[: max(1, min(pool_size, len(score_rank)))])

            selected_docs: List[int] = []
            remaining_groups = set(node_ids)

            if score_rank:
                first = int(score_rank[0])
                selected_docs.append(first)
                pool.discard(first)
                remaining_groups -= (doc_to_groups.get(first) or set())

            while len(selected_docs) < max(1, diversify_n) and remaining_groups and pool:
                best_doc = None
                best_obj = -1e18
                best_score_doc = -1e18
                for d in pool:
                    gain = len((doc_to_groups.get(d) or set()) & remaining_groups)
                    s = float(doc_score.get(d, 0.0))
                    obj = s + gain_w * float(gain)
                    if obj > best_obj or (obj == best_obj and s > best_score_doc):
                        best_doc = int(d)
                        best_obj = obj
                        best_score_doc = s
                if best_doc is None:
                    break
                selected_docs.append(best_doc)
                pool.discard(best_doc)
                remaining_groups -= (doc_to_groups.get(best_doc) or set())

            # Finish with score order (pool first, then the rest).
            rest_pool = [d for d in score_rank[: max(1, min(pool_size, len(score_rank)))] if d not in set(selected_docs)]
            rest_tail = [d for d in score_rank[max(1, min(pool_size, len(score_rank))):] if d not in set(selected_docs)]
            struct_docs_rank = selected_docs + rest_pool + rest_tail
        else:
            struct_docs_rank = [doc for doc, _ in sorted(doc_score.items(), key=lambda x: x[1], reverse=True)]

        # Optional: LLM rerank for top docs (targets Recall@2/5). Cheap because N is small and cached.
        if (
            llm is not None
            and bool(getattr(self.config, "enable_llm_doc_rerank", False))
            and len(struct_docs_rank) >= 3
        ):
            try:
                top_n = int(getattr(self.config, "llm_doc_rerank_top_n", 10))
                select_k = int(getattr(self.config, "llm_doc_rerank_select_k", 2))
                snippet_chars = int(getattr(self.config, "llm_doc_rerank_snippet_chars", 320))
                top_n = max(3, min(top_n, len(struct_docs_rank)))
                select_k = max(1, min(select_k, 5))
                cand_docs = [int(d) for d in struct_docs_rank[:top_n]]

                cand_lines: List[str] = []
                for i, d in enumerate(cand_docs):
                    doc = doc_idx_to_doc.get(int(d)) or {}
                    title = str(doc.get("title") or "").strip()
                    # Use the best passage across query variants so the reranker sees the right evidence even for 2-hop docs.
                    row = doc_best_passage_row.get(int(d))
                    best_sim = float(doc_best_passage_sim.get(int(d), -1e18))
                    for gid in node_ids:
                        row_g = (group_doc_best_passage_row.get(gid) or {}).get(int(d))
                        sim_g = (group_doc_best_passage_sim.get(gid) or {}).get(int(d))
                        if row_g is None or sim_g is None:
                            continue
                        if float(sim_g) > best_sim:
                            best_sim = float(sim_g)
                            row = int(row_g)
                    snippet = ""
                    if row is not None:
                        snippet = str((passages[int(row)] or {}).get("text") or "")
                    snippet = re.sub(r"\s+", " ", snippet).strip()
                    if snippet_chars > 0 and len(snippet) > snippet_chars:
                        snippet = snippet[:snippet_chars].rstrip() + " ..."
                    if title:
                        cand_lines.append(f"{i}) {title}: {snippet}")
                    else:
                        cand_lines.append(f"{i}) (no title): {snippet}")

                system = (
                    "You are a retrieval reranker for multi-hop QA. "
                    "Given a question and candidate Wikipedia passages, pick the smallest set of candidates "
                    "that together provide enough information to answer. "
                    "Prefer candidates that cover different hops/entities. "
                    "Output strict JSON only."
                )
                user = (
                    f"Question:\n{question}\n\n"
                    "Candidates:\n"
                    + "\n".join(cand_lines)
                    + f"\n\nReturn JSON: {{\"selected\": [i1, i2]}} with exactly {select_k} indices."
                )
                messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
                try:
                    raw, meta = llm.infer(messages=messages, response_format={"type": "json_object"}, temperature=0.0)
                except Exception:
                    raw, meta = llm.infer(messages=messages, temperature=0.0)

                obj = _parse_json_object(raw or "")
                idxs = obj.get("selected") or obj.get("selected_doc_indices") or []
                if not isinstance(idxs, list):
                    idxs = []

                sel_docs: List[int] = []
                seen = set()
                for x in idxs:
                    try:
                        j = int(x)
                    except Exception:
                        continue
                    if 0 <= j < len(cand_docs):
                        d = int(cand_docs[j])
                        if d not in seen:
                            sel_docs.append(d)
                            seen.add(d)
                    if len(sel_docs) >= select_k:
                        break

                # If LLM output is partial, fill from the top candidates.
                if sel_docs:
                    for d in cand_docs:
                        if len(sel_docs) >= select_k:
                            break
                        d = int(d)
                        if d not in seen:
                            sel_docs.append(d)
                            seen.add(d)

                    rest = [int(d) for d in struct_docs_rank if int(d) not in seen]
                    struct_docs_rank = sel_docs + rest
                    logger.debug(
                        f"[StructAlignRAG] [ONLINE_RERANK] applied | top_n={top_n} selected={sel_docs} cache_hit={bool((meta or {}).get('cache_hit'))}"
                    )
            except Exception as e:
                logger.debug(f"[StructAlignRAG] [ONLINE_RERANK] skipped | err={type(e).__name__}: {e}")

        # Build retrieved_docs list: struct-selected first, then dense fill.
        retrieved_docs: List[str] = []
        seen_docs: Set[int] = set()
        for d in struct_docs_rank:
            doc = doc_idx_to_doc.get(d)
            if doc:
                retrieved_docs.append(doc["doc_text"])
                seen_docs.add(d)
        for d in dense_doc_rank:
            if d in seen_docs:
                continue
            doc = doc_idx_to_doc.get(d)
            if doc:
                retrieved_docs.append(doc["doc_text"])
                seen_docs.add(d)
            if len(retrieved_docs) >= min(self.config.retrieval_top_k, len(docs)):
                break

        # Select passages for QA: primarily use passages from the top-ranked docs.
        # This keeps QA grounded to the same objects used in Recall@k and helps 2-hop questions
        # where the second-hop doc may have low similarity to the original question.
        chosen_passages: List[Dict[str, Any]] = []
        chosen_pids: Set[str] = set()
        total_tokens = 0

        for d in struct_docs_rank:
            if len(chosen_passages) >= self.config.qa_top_k_passages:
                break
            d = int(d)
            # Pick the best passage from this doc across *any* query variant (original question + subQs).
            row = doc_best_passage_row.get(d)
            best_sim = float(doc_best_passage_sim.get(d, -1e18))
            for gid in node_ids:
                row_g = (group_doc_best_passage_row.get(gid) or {}).get(d)
                sim_g = (group_doc_best_passage_sim.get(gid) or {}).get(d)
                if row_g is None or sim_g is None:
                    continue
                if float(sim_g) > best_sim:
                    best_sim = float(sim_g)
                    row = int(row_g)

            if row is None:
                continue
            p = passages[int(row)]
            pid = str(p.get("passage_id") or "")
            if not pid or pid in chosen_pids:
                continue
            toks = int(p.get("token_count") or _approx_tokens(p.get("text", "")))
            if total_tokens + toks > self.config.passage_token_budget:
                continue
            chosen_passages.append(p)
            chosen_pids.add(pid)
            total_tokens += toks

        # Fallback: dense top passages for the original question.
        if not chosen_passages:
            top_p_rows = _safe_topk(p_sims, min(self.config.qa_top_k_passages * 50, len(passages))).tolist()
            for row in top_p_rows:
                if len(chosen_passages) >= self.config.qa_top_k_passages:
                    break
                p = passages[int(row)]
                pid = str(p.get("passage_id") or "")
                if not pid or pid in chosen_pids:
                    continue
                toks = int(p.get("token_count") or _approx_tokens(p.get("text", "")))
                if total_tokens + toks > self.config.passage_token_budget:
                    continue
                chosen_passages.append(p)
                chosen_pids.add(pid)
                total_tokens += toks

        debug = {
            "subq_coverage": subq_coverage,
            "selected_nodes": sorted(selected),
            "selected_docs": struct_docs_rank,
            "selected_passages": [p.get("passage_id") for p in chosen_passages],
            "evidence_tokens": total_tokens,
            "elapsed_s": round(time.time() - t0, 4),
            "num_groups": len(group_candidates),
            "root": root,
            "binding_best_score": round(float(best_score), 6),
            "binding_topo": topo,
            "binding_parents": {k: sorted(list(v)) for k, v in parents.items()},
            "binding_assignment": chosen,
        }

        return RetrievalResult(
            retrieved_docs=retrieved_docs,
            selected_passages=chosen_passages,
            debug=debug,
        )
