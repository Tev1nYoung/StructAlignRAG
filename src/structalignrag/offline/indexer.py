from __future__ import annotations

import json
import os
import pickle
import time
from dataclasses import asdict
from typing import Any, Dict, List

import numpy as np

from ..config import StructAlignRAGConfig
from ..utils.genericness_utils import build_title_token_idf, title_avg_idf, title_genericness_score
from ..utils.logging_utils import get_logger
from ..utils.text_utils import clean_wiki_text, normalize_entity
from .capsule_canonicalizer import canonicalize_capsules
from .capsule_extractor import batch_extract_capsules
from .entity_canonicalizer import attach_entity_ids, canonicalize_entities
from .faiss_utils import build_ip_index, save_faiss_index, save_id_map
from .graph_builder import build_evidence_graph
from .passage_splitter import split_corpus_to_docs_and_passages
from .struct_index import build_struct_index, save_struct_index

logger = get_logger(__name__)

_PRONOUNS = {
    "he",
    "she",
    "they",
    "it",
    "him",
    "her",
    "them",
    "his",
    "their",
    "its",
}


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


class OfflineIndexer:
    def __init__(self, config: StructAlignRAGConfig, meta_dir: str) -> None:
        self.config = config
        self.meta_dir = meta_dir
        os.makedirs(self.meta_dir, exist_ok=True)

        # Core artifacts
        self.docs_path = os.path.join(self.meta_dir, "docs.jsonl")
        self.passages_path = os.path.join(self.meta_dir, "passages.jsonl")
        self.entities_path = os.path.join(self.meta_dir, "entities.jsonl")
        self.capsules_path = os.path.join(self.meta_dir, "capsules.jsonl")  # extracted capsules (each has canonical_id)
        self.canonical_capsules_path = os.path.join(self.meta_dir, "canonical_capsules.jsonl")
        self.cap2can_path = os.path.join(self.meta_dir, "capsule_to_canonical.jsonl")

        # Embeddings
        self.doc_emb_path = os.path.join(self.meta_dir, "doc_embeddings.npy")
        self.passage_emb_path = os.path.join(self.meta_dir, "passage_embeddings.npy")
        self.capsule_emb_path = os.path.join(self.meta_dir, "capsule_embeddings.npy")
        self.canonical_capsule_emb_path = os.path.join(self.meta_dir, "canonical_capsule_embeddings.npy")

        # FAISS
        self.faiss_passages_path = os.path.join(self.meta_dir, "faiss_passages.index")
        self.faiss_passage_ids_path = os.path.join(self.meta_dir, "faiss_passage_ids.json")
        self.faiss_capsules_path = os.path.join(self.meta_dir, "faiss_capsules.index")
        self.faiss_capsule_ids_path = os.path.join(self.meta_dir, "faiss_capsule_ids.json")

        # Graph
        self.graph_edges_path = os.path.join(self.meta_dir, "graph_edges.jsonl")
        self.graph_adj_path = os.path.join(self.meta_dir, "graph_adj.pkl")

        # Struct indices (for online speed + analysis)
        self.struct_index_path = os.path.join(self.meta_dir, "struct_index.pkl")
        self.entity_alias_to_id_path = os.path.join(self.meta_dir, "entity_alias_to_id.json")

        # Meta
        self.index_meta_path = os.path.join(self.meta_dir, "index_meta.json")
        self.offline_stats_path = os.path.join(self.meta_dir, "offline_stats.json")

    def _index_ready(self) -> bool:
        """
        Safety check: if index_meta exists but artifacts are missing (or legacy format),
        trigger rebuild even when force_index_from_scratch=false.
        """
        if not os.path.exists(self.index_meta_path):
            return False
        try:
            with open(self.index_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            return False

        paths = meta.get("paths") or {}
        required_keys = [
            "docs",
            "passages",
            "entities",
            "capsules",
            "canonical_capsules",
            "capsule_to_canonical",
            "doc_embeddings",
            "passage_embeddings",
            "canonical_capsule_embeddings",
            "faiss_passages",
            "faiss_passage_ids",
            "faiss_capsules",
            "faiss_capsule_ids",
            "graph_edges",
            "graph_adj",
            "struct_index",
            "entity_alias_to_id",
            "offline_stats",
        ]
        for k in required_keys:
            rel = paths.get(k)
            if not rel:
                return False
            if not os.path.exists(os.path.join(self.meta_dir, rel)):
                return False
        return True

    def build_or_load(self, corpus: List[Dict[str, Any]], embedder, llm) -> Dict[str, Any]:
        if (not self.config.force_index_from_scratch) and self._index_ready():
            logger.info(f"[StructAlignRAG] offline index exists, loading | meta={self.index_meta_path}")
            with open(self.index_meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        elif (not self.config.force_index_from_scratch) and os.path.exists(self.index_meta_path):
            logger.warning(
                f"[StructAlignRAG] offline index meta exists but artifacts are missing/legacy, rebuilding | meta={self.index_meta_path}"
            )

        t0 = time.time()
        step_times: Dict[str, float] = {}
        logger.info(f"[StructAlignRAG] [Step 0] offline indexing start | docs={len(corpus)}")

        # 0) docs + passages
        ts = time.time()
        docs, passages = split_corpus_to_docs_and_passages(corpus, self.config, tokenizer=embedder.tokenizer)
        _write_jsonl(self.docs_path, docs)
        _write_jsonl(self.passages_path, passages)
        step_times["split_docs_passages_s"] = round(time.time() - ts, 4)
        logger.info(f"[StructAlignRAG] [OFFLINE_SPLIT] docs/passages ready | docs={len(docs)} passages={len(passages)}")

        # 1) doc embeddings (for doc-level dense fill, optional but cheap)
        doc_texts_for_emb = [f"{d['title']}\n{clean_wiki_text(d.get('text',''))}" for d in docs]
        logger.info(f"[StructAlignRAG] [Step 0c] embedding docs ... | docs={len(docs)}")
        ts = time.time()
        doc_emb = embedder.encode(doc_texts_for_emb)
        np.save(self.doc_emb_path, doc_emb)
        step_times["embed_docs_s"] = round(time.time() - ts, 4)

        # 2) passage embeddings + FAISS
        passage_texts_for_emb = [f"{p['title']}\n{p['text']}" for p in passages]
        logger.info(f"[StructAlignRAG] [Step 0c] embedding passages ... | passages={len(passages)}")
        ts = time.time()
        passage_emb = embedder.encode(passage_texts_for_emb)
        np.save(self.passage_emb_path, passage_emb)
        p_index = build_ip_index(passage_emb)
        save_faiss_index(p_index, self.faiss_passages_path)
        save_id_map([p["passage_id"] for p in passages], self.faiss_passage_ids_path)
        step_times["embed_passages_s"] = round(time.time() - ts, 4)
        logger.info(f"[StructAlignRAG] [OFFLINE_INDEX] passage FAISS ready | path={self.faiss_passages_path}")

        # 3) capsule extraction (LLM, cached) per passage
        logger.info(f"[StructAlignRAG] [Step 0a] extracting capsules ... | mode={self.config.capsule_mode}")
        ts = time.time()
        raw_capsules, cap_stats = batch_extract_capsules(passages, llm=llm, config=self.config)
        step_times["extract_capsules_s"] = round(time.time() - ts, 4)
        logger.info(f"[StructAlignRAG] [OFFLINE_CAPSULE_EXTRACT] done | {cap_stats}")

        # Deterministic capsule ordering (thread completion order is non-deterministic).
        def _cap_key(cap: Dict[str, Any]):
            prov0 = (cap.get("provenance") or [{}])[0]
            return (
                int(prov0.get("doc_idx") or 0),
                str(prov0.get("passage_id") or ""),
                int(prov0.get("sent_id") or 0),
                str(cap.get("predicate") or ""),
                str(cap.get("canonical_text") or ""),
            )

        raw_capsules = sorted(raw_capsules, key=_cap_key)

        doc_idx_to_title = {int(d["doc_idx"]): str(d.get("title") or "") for d in docs}

        # Assign capsule_id and attach passage/doc/title
        capsules: List[Dict[str, Any]] = []
        for i, cap in enumerate(raw_capsules):
            cid = f"C_{i:08d}"
            # provenance already has doc_idx/passage_id/sent_id/span/quote
            prov0 = (cap.get("provenance") or [{}])[0]
            doc_idx = int(prov0.get("doc_idx"))
            doc_id = str(prov0.get("doc_id") or f"D_{doc_idx:07d}")
            passage_id = str(prov0.get("passage_id"))
            text = str(cap.get("canonical_text") or "").strip()
            if not text:
                text = str(prov0.get("quote") or "").strip()
            capsules.append(
                {
                    "capsule_id": cid,
                    "canonical_id": None,  # to be filled
                    "doc_id": doc_id,
                    "doc_idx": doc_idx,
                    "passage_id": passage_id,
                    "title": doc_idx_to_title.get(doc_idx, ""),
                    # Spec: "text" is the normalized statement used for embedding.
                    "text": text,
                    # Backward-compat (older dev iterations used canonical_text).
                    "canonical_text": text,
                    "predicate": cap.get("predicate") or "",
                    "polarity": cap.get("polarity") or "affirm",
                    "arguments": cap.get("arguments") or [],
                    "modifiers": cap.get("modifiers") or {},
                    "provenance": cap.get("provenance") or [],
                    "quality": cap.get("quality") or {"extractor": "unknown", "confidence": 0.0},
                }
            )

        # 4) entity canonicalization + attach entity_ids to capsule arguments
        logger.info(f"[StructAlignRAG] [OFFLINE_ENTITY] canonicalizing entities ...")
        ts = time.time()
        doc_titles = [d["title"] for d in docs]
        entities, surface_to_eid = canonicalize_entities(capsules, doc_titles, self.config)
        attach_entity_ids(capsules, surface_to_eid)

        # Pronoun linking: attach pronouns (arg0) to the doc-title entity_id when available.
        for cap in capsules:
            title = str(cap.get("title") or "").strip()
            title_eid = surface_to_eid.get(title)
            if not title_eid:
                continue
            updated = False
            for arg in cap.get("arguments", []) or []:
                if not isinstance(arg, dict):
                    continue
                surf = str(arg.get("surface") or "").strip()
                if arg.get("entity_id") is None and surf.lower() in _PRONOUNS:
                    arg["entity_id"] = title_eid
                    updated = True
            if updated:
                cap["entity_ids"] = sorted(
                    set([str(a.get("entity_id")) for a in cap.get("arguments", []) if a.get("entity_id")])
                )

        _write_jsonl(self.entities_path, entities)
        step_times["canonicalize_entities_s"] = round(time.time() - ts, 4)
        logger.info(f"[StructAlignRAG] [OFFLINE_ENTITY] entities ready | entities={len(entities)}")

        # 5) embed capsules (raw) on canonical_text (or fallback to quote)
        logger.info(f"[StructAlignRAG] [Step 0c] embedding capsules ... | capsules={len(capsules)}")
        ts = time.time()
        cap_texts = []
        for c in capsules:
            t = (c.get("text") or c.get("canonical_text") or "").strip()
            if not t:
                prov = (c.get("provenance") or [{}])[0]
                t = str(prov.get("quote") or "")
            cap_texts.append(t)
        cap_emb = embedder.encode(cap_texts)
        np.save(self.capsule_emb_path, cap_emb)
        step_times["embed_capsules_s"] = round(time.time() - ts, 4)

        # 6) capsule canonicalization
        logger.info(f"[StructAlignRAG] [OFFLINE_CANONICALIZE] canonicalizing capsules ...")
        ts = time.time()
        canonical_capsules, cap2can, canonical_emb, canonical_ids = canonicalize_capsules(capsules, cap_emb, self.config)
        np.save(self.canonical_capsule_emb_path, canonical_emb)
        step_times["canonicalize_capsules_s"] = round(time.time() - ts, 4)

        # update capsules canonical_id
        for c in capsules:
            c["canonical_id"] = cap2can.get(c["capsule_id"])

        _write_jsonl(self.capsules_path, capsules)
        _write_jsonl(self.canonical_capsules_path, canonical_capsules)
        _write_jsonl(self.cap2can_path, [{"capsule_id": k, "canonical_id": v} for k, v in cap2can.items()])
        logger.info(
            f"[StructAlignRAG] [OFFLINE_CANONICALIZE] canonical capsules ready | raw={len(capsules)} canonical={len(canonical_capsules)}"
        )

        # 7) capsule FAISS on canonical capsules
        c_index = build_ip_index(canonical_emb)
        save_faiss_index(c_index, self.faiss_capsules_path)
        save_id_map(canonical_ids, self.faiss_capsule_ids_path)
        logger.info(f"[StructAlignRAG] [OFFLINE_INDEX] capsule FAISS ready | path={self.faiss_capsules_path}")

        # 7.5) structural indices for fast online access
        ts = time.time()
        struct_index = build_struct_index(docs=docs, passages=passages, entities=entities, canonical_capsules=canonical_capsules)
        save_struct_index(struct_index, out_dir=self.meta_dir)
        step_times["build_struct_index_s"] = round(time.time() - ts, 4)

        # 8) evidence graph with typed edges (+ sim edges)
        logger.info(f"[StructAlignRAG] [Step 0b] building evidential graph ...")
        ts = time.time()
        gmeta = build_evidence_graph(
            config=self.config,
            docs=docs,
            passages=passages,
            entities=entities,
            canonical_capsules=canonical_capsules,
            canonical_embeddings=canonical_emb,
            out_dir=self.meta_dir,
            llm=llm,
        )
        step_times["build_graph_s"] = round(time.time() - ts, 4)

        offline_stats = {
            "step_times_s": step_times,
            "capsule_extract_stats": cap_stats,
        }
        with open(self.offline_stats_path, "w", encoding="utf-8") as f:
            json.dump(offline_stats, f, ensure_ascii=False, indent=2)

        # index_meta
        index_meta = {
            "config": asdict(self.config),
            "num_docs": len(docs),
            "num_passages": len(passages),
            "num_capsules": len(capsules),
            "num_canonical_capsules": len(canonical_capsules),
            "num_entities": len(entities),
            "graph": gmeta,
            "paths": {
                "docs": "docs.jsonl",
                "passages": "passages.jsonl",
                "entities": "entities.jsonl",
                "capsules": "capsules.jsonl",
                "canonical_capsules": "canonical_capsules.jsonl",
                "capsule_to_canonical": "capsule_to_canonical.jsonl",
                "doc_embeddings": "doc_embeddings.npy",
                "passage_embeddings": "passage_embeddings.npy",
                "capsule_embeddings": "capsule_embeddings.npy",
                "canonical_capsule_embeddings": "canonical_capsule_embeddings.npy",
                "faiss_passages": "faiss_passages.index",
                "faiss_passage_ids": "faiss_passage_ids.json",
                "faiss_capsules": "faiss_capsules.index",
                "faiss_capsule_ids": "faiss_capsule_ids.json",
                "graph_edges": "graph_edges.jsonl",
                "graph_adj": "graph_adj.pkl",
                "struct_index": "struct_index.pkl",
                "entity_alias_to_id": "entity_alias_to_id.json",
                "offline_stats": "offline_stats.json",
            },
            "elapsed_s": round(time.time() - t0, 4),
        }
        with open(self.index_meta_path, "w", encoding="utf-8") as f:
            json.dump(index_meta, f, ensure_ascii=False, indent=2)

        logger.info(f"[StructAlignRAG] [Step 0] offline indexing done | meta={self.index_meta_path} total_elapsed={time.time()-t0:.2f}s")
        return index_meta

    def load_index(self) -> Dict[str, Any]:
        if not os.path.exists(self.index_meta_path):
            raise FileNotFoundError(self.index_meta_path)

        docs = _read_jsonl(self.docs_path)
        passages = _read_jsonl(self.passages_path)
        entities = _read_jsonl(self.entities_path)
        capsules = _read_jsonl(self.capsules_path)
        canonical_capsules = _read_jsonl(self.canonical_capsules_path)

        doc_emb = np.load(self.doc_emb_path)
        passage_emb = np.load(self.passage_emb_path)
        canonical_emb = np.load(self.canonical_capsule_emb_path)

        with open(self.graph_adj_path, "rb") as f:
            adj = pickle.load(f)

        struct_index = None
        if os.path.exists(self.struct_index_path):
            with open(self.struct_index_path, "rb") as f:
                struct_index = pickle.load(f)

        offline_stats = None
        if os.path.exists(self.offline_stats_path):
            try:
                with open(self.offline_stats_path, "r", encoding="utf-8") as f:
                    offline_stats = json.load(f)
            except Exception:
                offline_stats = None

        # Unsupervised title genericness statistics (computed at load time; cheap and dataset-agnostic).
        titles = [str(d.get("title") or "") for d in docs]
        title_token_idf = build_title_token_idf(titles)
        default_idf = float(np.percentile(list(title_token_idf.values()), 90)) if title_token_idf else 6.0
        avg_idfs = [title_avg_idf(t, title_token_idf, default_idf=default_idf) for t in titles] if titles else []
        idf_p10 = float(np.percentile(avg_idfs, 10)) if avg_idfs else 0.0
        idf_p90 = float(np.percentile(avg_idfs, 90)) if avg_idfs else default_idf

        doc_genericness: Dict[int, float] = {}
        title_norm_to_genericness: Dict[str, float] = {}
        for d in docs:
            didx = int(d.get("doc_idx"))
            title = str(d.get("title") or "")
            snippet = str(d.get("text") or "")
            snip = snippet[:400]  # enough for disamb detection
            g = title_genericness_score(title, title_token_idf, idf_p10, idf_p90, text_snippet=snip)
            doc_genericness[didx] = float(g)
            if title:
                title_norm_to_genericness[normalize_entity(title)] = float(g)

        return {
            "docs": docs,
            "passages": passages,
            "entities": entities,
            "capsules": capsules,
            "canonical_capsules": canonical_capsules,
            "doc_emb": doc_emb,
            "passage_emb": passage_emb,
            "canonical_capsule_emb": canonical_emb,
            "adj": adj,
            "struct_index": struct_index,
            "offline_stats": offline_stats,
            "title_token_idf": title_token_idf,
            "title_idf_p10": idf_p10,
            "title_idf_p90": idf_p90,
            "doc_genericness": doc_genericness,
            "title_norm_to_genericness": title_norm_to_genericness,
        }
