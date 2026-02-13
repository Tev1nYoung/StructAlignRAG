from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from ..config import StructAlignRAGConfig
from ..utils.text_utils import clean_wiki_text, split_sentences


def _count_tokens(text: str, tokenizer) -> int:
    if not text:
        return 0
    # HuggingFace tokenizers are fast enough for our corpus sizes.
    return len(tokenizer.encode(text, add_special_tokens=False))


def _count_words(text: str) -> int:
    return len((text or "").split())


def split_corpus_to_docs_and_passages(
    corpus: List[Dict[str, Any]],
    config: StructAlignRAGConfig,
    tokenizer,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Turns corpus docs into:
    - docs: one row per doc (doc_text kept raw to match gold docs for Recall@k)
    - passages: token/word budgeted sentence windows (cleaned only for chunking)

    Passage offsets are computed on a whitespace-collapsed, cleaned doc text.
    """
    docs: List[Dict[str, Any]] = []
    passages: List[Dict[str, Any]] = []

    # Ensure deterministic order independent of input file ordering.
    # Some corpora (e.g., mini_2wikimultihopqa) do not include an explicit "idx".
    items = list(enumerate(corpus))
    items.sort(key=lambda it: int(it[1].get("idx")) if it[1].get("idx") is not None else it[0])

    for i, doc in items:
        raw_idx = doc.get("idx")
        doc_idx = int(raw_idx) if raw_idx is not None else int(i)
        doc_id = f"D_{doc_idx:07d}"
        title = str(doc.get("title", ""))
        raw_text = str(doc.get("text", ""))
        doc_text = f"{title}\n{raw_text}"
        docs.append(
            {
                "doc_id": doc_id,
                "doc_idx": doc_idx,
                "title": title,
                "text": raw_text,
                "doc_text": doc_text,
            }
        )

        clean = clean_wiki_text(raw_text)
        clean = re.sub(r"\s+", " ", clean).strip()
        if not clean:
            continue

        sents = split_sentences(clean)
        if not sents:
            continue

        # Precompute sentence spans in the collapsed-clean string.
        spans: List[Tuple[int, int]] = []
        cursor = 0
        for s in sents:
            idx = clean.find(s, cursor)
            if idx < 0:
                idx = cursor
            start = idx
            end = min(len(clean), idx + len(s))
            spans.append((start, end))
            cursor = end

        def _measure(text: str) -> int:
            if config.chunk_func == "by_word":
                return _count_words(text)
            return _count_tokens(text, tokenizer)

        i = 0
        chunk_idx = 0
        while i < len(sents):
            total = 0
            j = i
            while j < len(sents):
                cand = (sents[j] if j == i else (" " + sents[j]))
                add = _measure(cand)
                if total + add > config.chunk_tokens and j > i:
                    break
                total += add
                j += 1
                if total >= config.chunk_tokens:
                    break

            # Passage sentences [i, j)
            passage_sents = sents[i:j]
            passage_text = " ".join(passage_sents).strip()
            start_char = spans[i][0]
            end_char = spans[j - 1][1] if j - 1 < len(spans) else spans[-1][1]

            passage_id = f"P_{doc_idx:07d}_{chunk_idx:03d}"
            passages.append(
                {
                    "passage_id": passage_id,
                    "doc_id": doc_id,
                    "doc_idx": doc_idx,
                    "title": title,
                    "text": passage_text,
                    "sentences": passage_sents,
                    "start_char": start_char,
                    "end_char": end_char,
                    "sent_start": i,
                    "sent_end": j - 1,
                    "num_sents": len(passage_sents),
                    "token_count": int(total),
                    "metadata": {},
                }
            )
            chunk_idx += 1

            if j >= len(sents):
                break

            # Overlap: step back from j to include ~chunk_overlap tokens.
            back = 0
            ov = 0
            k = j - 1
            while k > i and ov < config.chunk_overlap:
                ov += _measure(sents[k])
                back += 1
                k -= 1
            i = max(i + 1, j - back)

    return docs, passages
