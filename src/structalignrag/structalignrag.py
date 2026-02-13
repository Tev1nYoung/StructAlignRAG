from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Set

from tqdm import tqdm

from .config import DEFAULT_LLM_BASE_URL, StructAlignRAGConfig
from .embed.contriever import ContrieverEmbedder
from .llm.openai_compat import CacheOpenAICompat, maybe_set_llm_key_from_file
from .metrics.metrics import extra_metrics, qa_em_f1, retrieval_recall
from .offline.indexer import OfflineIndexer
from .online.generator import AnswerGenerator
from .online.query_dag import build_query_dag
from .online.retriever import StructAlignRetriever
from .utils.logging_utils import get_logger
from .utils.naming import sanitize_model_name

logger = get_logger(__name__)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _append_jsonl(path: str, row: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_hipporag_style_metrics(path: str, record: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, indent=2, ensure_ascii=False))
        f.write("\n\n")


def _maybe_rotate_legacy_metrics(path: str) -> None:
    """
    Early versions wrote a single-line JSON (flat keys). HippoRAG writes multi-line JSON objects.
    If we detect legacy format, rename it to avoid mixing formats.
    """
    if not os.path.exists(path):
        return
    try:
        first_non_empty = None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                first_non_empty = line
                break
        if first_non_empty and first_non_empty.startswith("{") and "\"dataset\"" in first_non_empty and "\"time\"" in first_non_empty:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_path = path.replace(".jsonl", f".legacy_{ts}.jsonl")
            os.replace(path, new_path)
    except OSError:
        return


class StructAlignRAG:
    def __init__(self, config: StructAlignRAGConfig) -> None:
        self.config = config

        llm_tag = sanitize_model_name(config.llm_name)
        emb_tag = sanitize_model_name(config.embedding_model_name)
        # Match HippoRAG/HARE style: <llm_tag>_<emb_tag>
        self.meta_dir = os.path.join(config.save_dir(), f"{llm_tag}_{emb_tag}")
        _ensure_dir(self.meta_dir)

        logger.info(
            f"[StructAlignRAG] initialized | dataset={config.dataset} llm={config.llm_name} emb={config.embedding_model_name}"
        )

        # Models
        self.embedder = ContrieverEmbedder(
            model_name=config.embedding_model_name,
            batch_size=config.embedding_batch_size,
            max_length=config.embedding_max_seq_len,
            normalize=config.embedding_return_normalized,
            dtype=config.embedding_dtype,
        )

        maybe_set_llm_key_from_file(config.llm_base_url, DEFAULT_LLM_BASE_URL)
        if config.llm_base_url and "localhost" in config.llm_base_url and os.getenv("OPENAI_API_KEY") is None:
            os.environ["OPENAI_API_KEY"] = "sk-"

        cache_dir = os.path.join(config.save_dir(), "llm_cache")
        self.llm = CacheOpenAICompat(
            cache_dir=cache_dir,
            llm_name=config.llm_name,
            llm_base_url=config.llm_base_url,
            temperature=config.temperature,
            max_new_tokens=config.max_new_tokens,
            seed=config.seed,
            max_retries=2,
        )

        # Pipeline parts
        self.indexer = OfflineIndexer(config=config, meta_dir=self.meta_dir)
        self.retriever = StructAlignRetriever(config=config)
        self.generator = AnswerGenerator(config=config, llm=self.llm)

        # Outputs
        self.metrics_log_path = os.path.join(self.meta_dir, "metrics_log.jsonl")
        self.pred_path = os.path.join(self.meta_dir, "qa_predictions.json")

    def index(self, corpus: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.indexer.build_or_load(corpus=corpus, embedder=self.embedder, llm=self.llm)

    def rag_qa(
        self,
        queries: Sequence[str],
        gold_docs: Optional[Sequence[Sequence[str]]],
        gold_answers: Sequence[Set[str]],
        qids: Optional[Sequence[str]] = None,
        run_timing_s: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        t_qa0 = time.time()
        index = self.indexer.load_index()

        if qids is None:
            qids = [str(i) for i in range(len(queries))]

        n = len(queries)
        preds: List[Optional[Dict[str, Any]]] = [None for _ in range(n)]
        predicted_answers: List[Optional[str]] = [None for _ in range(n)]
        retrieved_docs_all: List[Optional[List[str]]] = [None for _ in range(n)]
        subq_cov_list: List[float] = [0.0 for _ in range(n)]
        ev_tokens_list: List[int] = [0 for _ in range(n)]
        latency_list: List[float] = [0.0 for _ in range(n)]

        # Online parallelism:
        # - We parallelize per-query (I/O bound on LLM).
        # - Embedding calls are serialized inside the embedder (thread-safe lock),
        #   so this does not change retrieval logic or introduce nondeterminism from concurrent GPU calls.
        workers = int(getattr(self.config, "online_qa_workers", 0) or 0)
        if workers <= 0:
            # Conservative auto: use up to num_keys, but cap to 8 to avoid 429 storms on some providers.
            try:
                workers = min(8, max(1, int(getattr(self.llm, "num_keys")() or 1)))
            except Exception:
                workers = 1

        logger.info(f"[StructAlignRAG] [RAG_QA] start | num_queries={n} workers={workers}")

        def _run_one(i: int, qid: str, q: str) -> Dict[str, Any]:
            qt0 = time.time()
            logger.info(f"[StructAlignRAG] [Step 1-5] query start | qid={qid} | {q}")

            dag = build_query_dag(q, llm=self.llm, config=self.config)
            rr = self.retriever.retrieve(question=q, query_dag=dag, index=index, embedder=self.embedder, llm=self.llm)
            ans, gen_meta = self.generator.answer(question=q, passages=rr.selected_passages)

            return {
                "i": i,
                "qid": qid,
                "question": q,
                "answer": ans,
                "query_dag": dag,
                "selected_passages": [{"doc_idx": p.get("doc_idx"), "title": p.get("title")} for p in rr.selected_passages],
                "retrieved_docs": rr.retrieved_docs,
                "debug": rr.debug,
                "gen_meta": gen_meta,
                "latency_s": float(time.time() - qt0),
            }

        if workers <= 1 or n <= 1:
            pbar = tqdm(total=n, desc="RAG_QA", disable=False, ascii=True, dynamic_ncols=True)
            for i, (qid, q) in enumerate(zip(qids, queries)):
                out = _run_one(i=i, qid=qid, q=q)
                predicted_answers[i] = out["answer"]
                retrieved_docs_all[i] = list(out["retrieved_docs"] or [])
                subq_cov_list[i] = float((out.get("debug") or {}).get("subq_coverage", 0.0))
                ev_tokens_list[i] = int((out.get("debug") or {}).get("evidence_tokens", 0))
                latency_list[i] = float(out.get("latency_s", 0.0))
                preds[i] = {
                    "qid": out["qid"],
                    "question": out["question"],
                    "answer": out["answer"],
                    "query_dag": out["query_dag"],
                    "selected_passages": out["selected_passages"],
                    "retrieved_docs_top5": (out["retrieved_docs"] or [])[:5],
                    "debug": out.get("debug") or {},
                    "gen_meta": out.get("gen_meta") or {},
                }
                pbar.update(1)
            pbar.close()
        else:
            pbar = tqdm(total=n, desc="RAG_QA", disable=False, ascii=True, dynamic_ncols=True)
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {
                    ex.submit(_run_one, i, qid, q): i
                    for i, (qid, q) in enumerate(zip(qids, queries))
                }
                for fut in as_completed(futs):
                    out = fut.result()
                    i = int(out["i"])
                    predicted_answers[i] = out["answer"]
                    retrieved_docs_all[i] = list(out["retrieved_docs"] or [])
                    subq_cov_list[i] = float((out.get("debug") or {}).get("subq_coverage", 0.0))
                    ev_tokens_list[i] = int((out.get("debug") or {}).get("evidence_tokens", 0))
                    latency_list[i] = float(out.get("latency_s", 0.0))
                    preds[i] = {
                        "qid": out["qid"],
                        "question": out["question"],
                        "answer": out["answer"],
                        "query_dag": out["query_dag"],
                        "selected_passages": out["selected_passages"],
                        "retrieved_docs_top5": (out["retrieved_docs"] or [])[:5],
                        "debug": out.get("debug") or {},
                        "gen_meta": out.get("gen_meta") or {},
                    }
                    pbar.update(1)
            pbar.close()

        # Tight sanity: ensure ordering arrays are filled.
        predicted_answers_f: List[str] = [str(a or "") for a in predicted_answers]
        retrieved_docs_all_f: List[List[str]] = [list(x or []) for x in retrieved_docs_all]
        preds_f: List[Dict[str, Any]] = [p or {"qid": qids[i], "question": str(queries[i]), "answer": ""} for i, p in enumerate(preds)]

        with open(self.pred_path, "w", encoding="utf-8") as f:
            json.dump(preds_f, f, ensure_ascii=False, indent=2)
        logger.info(f"[StructAlignRAG] [RAG_QA] predictions written | {self.pred_path}")

        # Metrics
        qa_metrics = qa_em_f1(gold_answers=gold_answers, predicted_answers=predicted_answers_f)
        retrieval_metrics = retrieval_recall(
            gold_docs=gold_docs,
            retrieved_docs=retrieved_docs_all_f,
            k_list=[1, 2, 5, 10, 20, 30, 50, 100, 150, 200],
        )
        extra = extra_metrics(subq_coverages=subq_cov_list, evidence_tokens=ev_tokens_list, latencies_s=latency_list)

        # Attach end-to-end run timing (measured in main) into the metrics record.
        # This avoids a separate runtime log file and keeps HippoRAG-style per-run metadata together.
        run_timing: Dict[str, float] = {}
        if isinstance(run_timing_s, dict):
            for k, v in run_timing_s.items():
                try:
                    run_timing[str(k)] = float(v)
                except Exception:
                    continue
        run_timing["rag_qa"] = float(time.time() - t_qa0)
        if ("init" in run_timing) and ("index" in run_timing):
            run_timing["total"] = float(run_timing.get("init", 0.0) + run_timing.get("index", 0.0) + run_timing["rag_qa"])
        # Keep a flattened view in extra_metrics for easy scanning.
        extra = dict(extra or {})
        extra["RunQAS"] = round(run_timing["rag_qa"], 4)
        if "init" in run_timing:
            extra["RunInitS"] = round(run_timing["init"], 4)
        if "index" in run_timing:
            extra["RunIndexS"] = round(run_timing["index"], 4)
        if "total" in run_timing:
            extra["RunTotalS"] = round(run_timing["total"], 4)

        metrics: Dict[str, Any] = {}
        metrics.update(qa_metrics)
        metrics.update(retrieval_metrics)
        metrics.update(extra)

        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "run_mode": "rag_qa",
            "dataset": self.config.dataset,
            "num_queries": len(queries),
            "llm_name": self.config.llm_name,
            "llm_base_url": self.config.llm_base_url,
            "embedding_model_name": self.config.embedding_model_name,
            "embedding_base_url": None,
            "openie_mode": None,
            "graph_type": f"structalign_capsule_{self.config.capsule_mode}",
            "retrieval_top_k": self.config.retrieval_top_k,
            "qa_top_k": self.config.qa_top_k_passages,
            "rerank_dspy_file_path": None,
            "retrieval_metrics": retrieval_metrics if retrieval_metrics else None,
            "qa_metrics": qa_metrics,
            "extra_metrics": extra,
            "run_timing_s": {k: round(v, 4) for k, v in run_timing.items()} if run_timing else None,
        }

        _maybe_rotate_legacy_metrics(self.metrics_log_path)
        _append_hipporag_style_metrics(self.metrics_log_path, record)
        logger.info(f"[StructAlignRAG] [Metrics] appended metrics log | {self.metrics_log_path}")

        logger.info(
            f"[StructAlignRAG] [RAG_QA] done | EM={qa_metrics.get('ExactMatch')} F1={qa_metrics.get('F1')} "
            f"R@5={(retrieval_metrics or {}).get('Recall@5')} SubQCoverage={extra.get('SubQCoverage')}"
        )
        return metrics
