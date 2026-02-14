import argparse
import json
import os
import time

from src.structalignrag.config import DEFAULT_EMB_NAME, DEFAULT_LLM_BASE_URL, DEFAULT_LLM_NAME, StructAlignRAGConfig
from src.structalignrag.data.dataset_loader import (
    cap_samples,
    corpus_to_docs,
    get_gold_answers,
    get_gold_docs,
    load_corpus,
    load_samples,
)
from src.structalignrag.structalignrag import StructAlignRAG
from src.structalignrag.utils.logging_utils import setup_logging


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def _parse_bool(x: str) -> bool:
    return str(x).strip().lower() in ("1", "true", "yes", "y", "t")


def main() -> None:
    parser = argparse.ArgumentParser(description="StructAlignRAG (zero-shot) retrieval and QA")
    parser.add_argument("--dataset", type=str, default="sample", help="Dataset name (reproduce/dataset/*.json)")
    parser.add_argument("--llm_base_url", type=str, default=DEFAULT_LLM_BASE_URL, help="LLM base URL")
    parser.add_argument("--llm_name", type=str, default=DEFAULT_LLM_NAME, help="LLM model name")
    parser.add_argument("--embedding_name", type=str, default=DEFAULT_EMB_NAME, help="Embedding model name")
    parser.add_argument("--embedding_batch_size", type=int, default=None, help="Embedding batch size override")
    parser.add_argument("--embedding_max_seq_len", type=int, default=None, help="Embedding max sequence length override")
    parser.add_argument("--embedding_dtype", type=str, default=None, help="Embedding dtype override: auto|float16|bfloat16|float32")
    parser.add_argument(
        "--embedding_query_instruction",
        type=str,
        default=None,
        help="Optional query instruction for instruction-tuned embedders (e.g., NV-Embed-v2).",
    )
    parser.add_argument("--save_root", type=str, default="outputs", help="Save root directory")
    parser.add_argument("--force_index_from_scratch", type=str, default="false", help="Rebuild offline index")
    # Offline tuning knobs (engineering-grade reproducibility)
    parser.add_argument("--capsule_mode", type=str, default=None, help="Offline capsule mode: llm|sentence")
    parser.add_argument("--chunk_tokens", type=int, default=None, help="Passage chunk token budget (default from config)")
    parser.add_argument("--chunk_overlap", type=int, default=None, help="Chunk overlap tokens (default from config)")
    parser.add_argument("--offline_llm_workers", type=int, default=None, help="Offline LLM parallel workers")
    parser.add_argument("--online_qa_workers", type=int, default=None, help="Online QA parallel workers (per-query)")
    parser.add_argument("--enable_nli_edges", type=str, default=None, help="Enable cached NLI edges: true|false")
    parser.add_argument("--nli_llm_workers", type=int, default=None, help="Offline NLI LLM workers (if enabled)")
    parser.add_argument("--offline_store_llm_meta", type=str, default=None, help="Store per-call llm_meta into capsules: true|false")
    # Online ablations (for paper-quality experiments)
    parser.add_argument("--enable_llm_doc_rerank", type=str, default=None, help="Enable LLM doc rerank: true|false")
    parser.add_argument("--enable_doc_mention_jump", type=str, default=None, help="Enable doc-mention jump: true|false")
    parser.add_argument("--enable_genericness_penalty", type=str, default=None, help="Enable genericness penalty: true|false")
    parser.add_argument("--genericness_penalty_weight", type=float, default=None, help="Genericness penalty weight (subtract w*scaled(g))")
    parser.add_argument("--genericness_penalty_threshold", type=float, default=None, help="Only penalize docs with genericness >= threshold")
    parser.add_argument(
        "--enable_global_evidence_selection",
        type=str,
        default=None,
        help="Enable global evidence selection (coverage-aware passage packing): true|false",
    )
    parser.add_argument(
        "--enable_local_propagation",
        type=str,
        default=None,
        help="Enable local propagation (mini-PPR on induced subgraph): true|false",
    )
    parser.add_argument(
        "--subq_coverage_top_m",
        type=int,
        default=None,
        help="SubQCoverage@M: M for evidence coverage (top-M capsules per group)",
    )
    parser.add_argument("--max_queries", type=int, default=None, help="Optional cap on number of queries")
    parser.add_argument("--query_offset", type=int, default=0, help="Optional starting offset")
    parser.add_argument("--shuffle_seed", type=int, default=None, help="Optional deterministic shuffle seed")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()

    setup_logging(args.log_level)

    dataset = args.dataset
    corpus_path = os.path.join("reproduce", "dataset", f"{dataset}_corpus.json")
    samples_path = os.path.join("reproduce", "dataset", f"{dataset}.json")

    corpus = load_corpus(corpus_path)
    samples = load_samples(samples_path)
    samples = cap_samples(samples, max_queries=args.max_queries, query_offset=args.query_offset, shuffle_seed=args.shuffle_seed)

    docs = corpus_to_docs(corpus)
    queries = [s["question"] for s in samples]
    qids = [str(s.get("id", i)) for i, s in enumerate(samples)]

    gold_answers = get_gold_answers(samples)
    try:
        gold_docs = get_gold_docs(samples, dataset_name=dataset)
        assert len(queries) == len(gold_docs) == len(gold_answers)
    except Exception:
        gold_docs = None

    cfg = StructAlignRAGConfig(
        dataset=dataset,
        save_root=args.save_root,
        llm_base_url=args.llm_base_url,
        llm_name=args.llm_name,
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=_parse_bool(args.force_index_from_scratch),
    )
    if args.embedding_batch_size is not None:
        cfg.embedding_batch_size = int(args.embedding_batch_size)
    if args.embedding_max_seq_len is not None:
        cfg.embedding_max_seq_len = int(args.embedding_max_seq_len)
    if args.embedding_dtype is not None:
        cfg.embedding_dtype = str(args.embedding_dtype)
    if args.embedding_query_instruction is not None:
        cfg.embedding_query_instruction = str(args.embedding_query_instruction)
    if args.capsule_mode is not None:
        cfg.capsule_mode = str(args.capsule_mode).strip()
    if args.chunk_tokens is not None:
        cfg.chunk_tokens = int(args.chunk_tokens)
    if args.chunk_overlap is not None:
        cfg.chunk_overlap = int(args.chunk_overlap)
    if args.offline_llm_workers is not None:
        cfg.offline_llm_workers = int(args.offline_llm_workers)
    if args.online_qa_workers is not None:
        cfg.online_qa_workers = int(args.online_qa_workers)
    if args.enable_nli_edges is not None:
        cfg.enable_nli_edges = _parse_bool(args.enable_nli_edges)
    if args.nli_llm_workers is not None:
        cfg.nli_llm_workers = int(args.nli_llm_workers)
    if args.offline_store_llm_meta is not None:
        cfg.offline_store_llm_meta = _parse_bool(args.offline_store_llm_meta)
    if args.enable_llm_doc_rerank is not None:
        cfg.enable_llm_doc_rerank = _parse_bool(args.enable_llm_doc_rerank)
    if args.enable_doc_mention_jump is not None:
        cfg.enable_doc_mention_jump = _parse_bool(args.enable_doc_mention_jump)
    if args.enable_genericness_penalty is not None:
        cfg.enable_genericness_penalty = _parse_bool(args.enable_genericness_penalty)
    if args.genericness_penalty_weight is not None:
        cfg.genericness_penalty_weight = float(args.genericness_penalty_weight)
    if args.genericness_penalty_threshold is not None:
        cfg.genericness_penalty_threshold = float(args.genericness_penalty_threshold)
    if args.enable_global_evidence_selection is not None:
        cfg.enable_global_evidence_selection = _parse_bool(args.enable_global_evidence_selection)
    if args.enable_local_propagation is not None:
        cfg.enable_local_propagation = _parse_bool(args.enable_local_propagation)
    if args.subq_coverage_top_m is not None:
        cfg.subq_coverage_top_m = int(args.subq_coverage_top_m)

    t0 = time.time()
    rag = StructAlignRAG(cfg)
    t1 = time.time()
    rag.index(corpus)
    t2 = time.time()
    metrics = rag.rag_qa(
        queries=queries,
        gold_docs=gold_docs,
        gold_answers=gold_answers,
        qids=qids,
        run_timing_s={"init": float(t1 - t0), "index": float(t2 - t1)},
    )
    t3 = time.time()

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[main] total_elapsed={t3-t0:.2f}s")


if __name__ == "__main__":
    main()
