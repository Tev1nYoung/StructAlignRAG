from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from ..config import StructAlignRAGConfig
from ..utils.logging_utils import get_logger
from ..utils.text_utils import split_sentences

logger = get_logger(__name__)

def _extract_json_object(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("empty response")
    text = text.strip()
    if "```" in text:
        blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if blocks:
            text = max(blocks, key=len).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def _sent_spans_in_passage(passage_text: str, sents: List[str]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for s in sents:
        idx = passage_text.find(s, cursor)
        if idx < 0:
            idx = cursor
        start = idx
        end = min(len(passage_text), idx + len(s))
        spans.append((start, end))
        cursor = end
    return spans


def _fallback_sentence_capsules(passage: Dict[str, Any], config: StructAlignRAGConfig) -> List[Dict[str, Any]]:
    sents = split_sentences(passage.get("text", ""))
    spans = _sent_spans_in_passage(passage.get("text", ""), sents)

    doc_idx = int(passage.get("doc_idx", 0))
    doc_id = str(passage.get("doc_id") or f"D_{doc_idx:07d}")

    out: List[Dict[str, Any]] = []
    for sid, sent in enumerate(sents[: config.max_capsules_per_passage]):
        sent = sent.strip()
        if not sent:
            continue
        prov = {
            "doc_id": doc_id,
            "doc_idx": doc_idx,
            "passage_id": passage["passage_id"],
            "sent_id": sid,
            "span": [int(spans[sid][0]), int(spans[sid][1])],
            "quote": sent[:200],
        }
        out.append(
            {
                "predicate": "sentence",
                "polarity": "affirm",
                "arguments": [],
                "modifiers": {},
                "canonical_text": sent,
                "provenance": [prov],
                "quality": {"extractor": "heuristic", "confidence": 0.0},
            }
        )
    return out


def extract_capsules_for_passage(
    passage: Dict[str, Any],
    llm,
    config: StructAlignRAGConfig,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns (capsules, stats) where capsules are WITHOUT capsule_id/canonical_id.
    """
    if config.capsule_mode != "llm":
        return _fallback_sentence_capsules(passage, config), {"llm_ok": 0, "fallback": 1}

    sents = split_sentences(passage.get("text", ""))
    if not sents:
        return [], {"llm_ok": 1, "fallback": 0}

    doc_idx = int(passage.get("doc_idx", 0))
    doc_id = str(passage.get("doc_id") or f"D_{doc_idx:07d}")

    sent_lines = "\n".join([f"{i}: {s}" for i, s in enumerate(sents)])
    system = (
        "You extract atomic evidence capsules from a passage. Output strict JSON only. "
        "Do not invent facts. Use the provided sentence ids for provenance."
    )
    user = (
        f"Title: {passage.get('title','')}\n"
        f"Passage:\n{passage.get('text','')}\n\n"
        f"Sentences:\n{sent_lines}\n\n"
        "Return JSON:\n"
        "{\n"
        '  "capsules": [\n'
        "    {\n"
        '      "predicate": \"...\",\n'
        '      "polarity": \"affirm|neg\",\n'
        '      "arguments": [ {\"role\":\"arg0\",\"surface\":\"...\"}, {\"role\":\"arg1\",\"surface\":\"...\"} ],\n'
        '      "modifiers": {\"time\":\"...\",\"location\":\"...\",\"quantity\":\"...\",\"comparative\":\"...\"},\n'
        '      "sent_id": 0,\n'
        '      "canonical_text": \"short normalized statement\"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Rules:\n- Extract up to {config.max_capsules_per_passage} capsules.\n"
        "- predicate should be a short relation phrase.\n"
        "- canonical_text should be short and faithful to the passage.\n"
        "- Prefer factual relations/attributes/events.\n"
        "- arguments.surface must appear in the sentence text if possible.\n"
        "- Output JSON only."
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _infer_once(msgs):
        try:
            try:
                raw, meta = llm.infer(
                    messages=msgs,
                    temperature=config.offline_temperature,
                    max_completion_tokens=config.offline_max_new_tokens,
                    seed=config.seed,
                    response_format={"type": "json_object"},
                )
            except Exception:
                raw, meta = llm.infer(
                    messages=msgs,
                    temperature=config.offline_temperature,
                    max_completion_tokens=config.offline_max_new_tokens,
                    seed=config.seed,
                )
            obj = _extract_json_object(raw)
            return obj, meta
        except Exception as e:
            raise e

    try:
        try:
            obj, meta = _infer_once(messages)
        except Exception:
            # One retry with an even stricter system instruction (as specified in the design).
            strict = system + " STRICT: Return JSON only. No markdown, no code fences."
            obj, meta = _infer_once([{"role": "system", "content": strict}, {"role": "user", "content": user}])

        caps = obj.get("capsules") or []
        if not isinstance(caps, list):
            raise ValueError("capsules is not a list")

        spans = _sent_spans_in_passage(passage.get("text", ""), sents)

        out: List[Dict[str, Any]] = []
        for c in caps[: config.max_capsules_per_passage]:
            if not isinstance(c, dict):
                continue
            sid = c.get("sent_id", 0)
            try:
                sid = int(sid)
            except Exception:
                sid = 0
            sid = max(0, min(sid, len(sents) - 1))
            quote = sents[sid].strip()
            prov = {
                "doc_id": doc_id,
                "doc_idx": doc_idx,
                "passage_id": passage["passage_id"],
                "sent_id": sid,
                "span": [int(spans[sid][0]), int(spans[sid][1])],
                "quote": quote[:200],
            }

            qual = {"extractor": "llm", "confidence": 1.0}
            if bool(config.offline_store_llm_meta or config.debug_trace):
                qual["llm_meta"] = meta

            out.append(
                {
                    "predicate": str(c.get("predicate") or "").strip(),
                    "polarity": str(c.get("polarity") or "affirm").strip(),
                    "arguments": list(c.get("arguments") or []),
                    "modifiers": dict(c.get("modifiers") or {}),
                    "canonical_text": str(c.get("canonical_text") or quote).strip(),
                    "provenance": [prov],
                    "quality": qual,
                }
            )

        if not out:
            return _fallback_sentence_capsules(passage, config), {"llm_ok": 0, "fallback": 1}
        return out, {"llm_ok": 1, "fallback": 0, "cache_hit": bool(meta.get("cache_hit"))}
    except Exception as e:
        logger.warning(
            f"[StructAlignRAG] [OFFLINE_CAPSULE_EXTRACT] passage fallback | pid={passage.get('passage_id')} err={type(e).__name__}: {e}"
        )
        return _fallback_sentence_capsules(passage, config), {"llm_ok": 0, "fallback": 1, "error": str(e)}


def batch_extract_capsules(
    passages: List[Dict[str, Any]],
    llm,
    config: StructAlignRAGConfig,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns (capsules, stats). Each capsule row is WITHOUT capsule_id/canonical_id.
    """
    if not passages:
        return [], {"passages": 0, "capsules": 0}

    stats = {"passages": len(passages), "capsules": 0, "llm_ok": 0, "fallback": 0, "cache_hit": 0}
    all_caps: List[Dict[str, Any]] = []

    workers = max(1, int(config.offline_llm_workers or 1))
    pbar = tqdm(total=len(passages), desc="Capsule Extraction", disable=False, ascii=True, dynamic_ncols=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(extract_capsules_for_passage, p, llm, config): p for p in passages}
        for fut in as_completed(futs):
            caps, st = fut.result()
            all_caps.extend(caps)
            stats["capsules"] += len(caps)
            stats["llm_ok"] += int(st.get("llm_ok", 0))
            stats["fallback"] += int(st.get("fallback", 0))
            stats["cache_hit"] += int(bool(st.get("cache_hit", False)))
            pbar.update(1)
            pbar.set_postfix(
                {
                    "capsules": stats["capsules"],
                    "fallback": stats["fallback"],
                    "llm_ok": stats["llm_ok"],
                    "workers": workers,
                }
            )
    pbar.close()
    return all_caps, stats
