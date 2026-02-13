from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..utils.text_utils import normalize_answer


def _f1_single(gold: str, pred: str) -> float:
    gold_tokens = normalize_answer(gold).split()
    pred_tokens = normalize_answer(pred).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / max(len(pred_tokens), 1)
    recall = 1.0 * num_same / max(len(gold_tokens), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def qa_em_f1(
    gold_answers: Sequence[Set[str]],
    predicted_answers: Sequence[str],
) -> Dict[str, float]:
    assert len(gold_answers) == len(predicted_answers)
    ems = []
    f1s = []
    for gold_set, pred in zip(gold_answers, predicted_answers):
        pred = pred or ""
        gold_list = list(gold_set) if isinstance(gold_set, set) else list(gold_set)
        em = 0.0
        f1 = 0.0
        for g in gold_list:
            if normalize_answer(g) == normalize_answer(pred):
                em = 1.0
            f1 = max(f1, _f1_single(g, pred))
        ems.append(em)
        f1s.append(f1)
    return {"ExactMatch": float(np.mean(ems) if ems else 0.0), "F1": float(np.mean(f1s) if f1s else 0.0)}


def retrieval_recall(
    gold_docs: Optional[Sequence[Sequence[str]]],
    retrieved_docs: Sequence[Sequence[str]],
    k_list: Sequence[int],
) -> Dict[str, float]:
    if gold_docs is None:
        return {}
    assert len(gold_docs) == len(retrieved_docs)
    k_list = sorted(set(int(k) for k in k_list))
    pooled = {f"Recall@{k}": 0.0 for k in k_list}
    n = len(gold_docs)
    for gd, rd in zip(gold_docs, retrieved_docs):
        gd_set = set(gd)
        for k in k_list:
            topk = rd[:k]
            hit = len(set(topk) & gd_set)
            pooled[f"Recall@{k}"] += (hit / len(gd_set)) if gd_set else 0.0
    for k in k_list:
        pooled[f"Recall@{k}"] = round(pooled[f"Recall@{k}"] / max(n, 1), 4)
    return pooled


def mean_or_none(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def extra_metrics(
    subq_coverages: List[float],
    evidence_tokens: List[int],
    latencies_s: List[float],
) -> Dict[str, float]:
    return {
        "SubQCoverage": round(mean_or_none(subq_coverages), 4),
        "EvidenceTokens": int(np.mean(evidence_tokens)) if evidence_tokens else 0,
        "LatencyAvgS": round(mean_or_none(latencies_s), 4),
    }

