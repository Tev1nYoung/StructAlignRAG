from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

from ..config import StructAlignRAGConfig
from ..utils.text_utils import normalize_entity
from ..utils.union_find import UnionFind


_ROLE_PREFIXES = (
    "professor of",
    "chief ",
    "co-founder",
    "founder",
    "officer",
)

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
    "this",
    "that",
    "these",
    "those",
}

_DEMONYMS = {
    "american",
    "german",
    "french",
    "british",
    "chinese",
    "japanese",
    "korean",
    "italian",
    "spanish",
    "russian",
    "indian",
    "canadian",
    "australian",
}

_MONTHS = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)


def _looks_like_person(name: str) -> bool:
    toks = [t for t in re.split(r"\s+", name.strip()) if t]
    if len(toks) < 2 or len(toks) > 5:
        return False
    # Require at least first+last in TitleCase-ish (allow "C.")
    def ok_tok(t: str) -> bool:
        if not t:
            return False
        if len(t) == 2 and t.endswith(".") and t[0].isalpha():
            return True
        return t[0].isupper()

    return ok_tok(toks[0]) and ok_tok(toks[-1])


def _person_key(name: str) -> str:
    toks = [t for t in re.split(r"\s+", name.strip()) if t]
    if len(toks) < 2:
        return ""
    first = toks[0].strip(".")
    last = toks[-1].strip(".")
    if not first or not last:
        return ""
    return normalize_entity(f"{first} {last}")


def _looks_like_date(surface: str) -> bool:
    s = (surface or "").strip()
    if not s:
        return False
    low = s.lower()
    # Rough patterns like "April 25, 1971" or "December 22 1955"
    if any(m in low for m in _MONTHS) and re.search(r"\b\d{1,2}\b", low) and re.search(r"\b\d{4}\b", low):
        return True
    # Year-only literals are too generic to serve as bridging entities.
    if re.fullmatch(r"\d{4}", s):
        return True
    return False


def _is_valid_entity_surface(surface: str) -> bool:
    s = (surface or "").strip()
    if not s:
        return False
    if len(s) <= 2:
        return False
    if len(s) > 80:
        return False
    if len(re.split(r"\s+", s)) > 12:
        return False
    if s.isnumeric():
        return False
    low = s.lower()
    if any(low.startswith(p) for p in _ROLE_PREFIXES):
        return False
    if _looks_like_date(s):
        return False

    toks = [t for t in re.split(r"\s+", s) if t]
    if toks:
        first = toks[0].lower()
        # Possessives/determiners are almost always descriptions ("his ...", "the ...").
        if first in _PRONOUNS or first in {"a", "an", "the", "my", "your", "our"}:
            return False

    if low in _PRONOUNS:
        return False

    # Drop likely descriptors like "American neuroscientist, author, ..." (highly noisy as entity nodes).
    if "," in s and len(s) > 40:
        return False

    # "American X" where X is lowercase is usually a descriptor, not an entity.
    if toks and toks[0].lower() in _DEMONYMS:
        if len(toks) == 1 or toks[1].islower():
            return False
    if toks:
        # Handle hyphenated demonyms like "German-American biochemist"
        first0 = re.split(r"[-/]", toks[0].lower())[0]
        if first0 in _DEMONYMS:
            if len(toks) == 1 or toks[1].islower():
                return False

    return True


def canonicalize_entities(
    capsules: List[Dict[str, Any]],
    doc_titles: List[str],
    config: StructAlignRAGConfig,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Returns (entities, surface_to_entity_id).

    Entities are clustered by:
    - normalized surface
    - optional person_key(first+last) for person-like mentions
    - optional merge with doc titles
    """
    counts: Counter[str] = Counter()
    surfaces: List[str] = []
    for cap in capsules:
        for arg in cap.get("arguments", []) or []:
            if not isinstance(arg, dict):
                continue
            surf = str(arg.get("surface") or "").strip()
            if not _is_valid_entity_surface(surf):
                continue
            counts[surf] += 1
            surfaces.append(surf)
    for t in doc_titles:
        t = str(t or "").strip()
        if _is_valid_entity_surface(t):
            counts[t] += 1
            surfaces.append(t)

    uniq = list(dict.fromkeys(surfaces))

    # For lowercase concept-like mentions, keep only reasonably frequent ones to reduce noise.
    def _has_upper(m: str) -> bool:
        return any(ch.isupper() for ch in (m or "") if ch.isalpha())

    filtered: List[str] = []
    title_set = set(doc_titles)
    for m in uniq:
        if m in title_set:
            filtered.append(m)
            continue
        if _has_upper(m):
            filtered.append(m)
            continue
        toks = [t for t in re.split(r"\s+", m) if t]
        if len(toks) > int(config.entity_max_tokens_lowercase or 4):
            continue
        if int(counts.get(m, 0)) < int(config.entity_min_freq_lowercase or 2):
            continue
        filtered.append(m)
    uniq = filtered
    uf = UnionFind(len(uniq))

    norm_bucket: Dict[str, List[int]] = defaultdict(list)
    person_bucket: Dict[str, List[int]] = defaultdict(list)

    for i, s in enumerate(uniq):
        norm = normalize_entity(s)
        if norm:
            norm_bucket[norm].append(i)
        if config.entity_person_merge and _looks_like_person(s):
            pk = _person_key(s)
            if pk:
                person_bucket[pk].append(i)

    for ids in norm_bucket.values():
        if len(ids) <= 1:
            continue
        base = ids[0]
        for j in ids[1:]:
            uf.union(base, j)

    for ids in person_bucket.values():
        if len(ids) <= 1:
            continue
        base = ids[0]
        for j in ids[1:]:
            uf.union(base, j)

    # Build clusters
    root_to_ids: Dict[int, List[int]] = defaultdict(list)
    for i in range(len(uniq)):
        root_to_ids[uf.find(i)].append(i)

    clusters: List[Dict[str, Any]] = []
    for ids in root_to_ids.values():
        aliases = [uniq[i] for i in ids]
        # Prefer exact doc title match as canonical name.
        canonical = None
        for a in aliases:
            if a in title_set:
                canonical = a
                break
        if canonical is None:
            # Most frequent alias
            canonical = max(aliases, key=lambda x: (counts.get(x, 0), -len(x)))
        freq = sum(int(counts.get(a, 0)) for a in aliases)
        clusters.append(
            {
                "canonical_name": canonical,
                "aliases": sorted(set(aliases)),
                "freq": freq,
            }
        )

    # Deterministic entity ids: sort clusters first, then assign ids.
    clusters.sort(key=lambda x: (-int(x["freq"]), str(x["canonical_name"]).lower(), str(x["canonical_name"])))

    entities: List[Dict[str, Any]] = []
    surface_to_eid: Dict[str, str] = {}
    for cluster_idx, c in enumerate(clusters):
        eid = f"E_{cluster_idx:07d}"
        entities.append(
            {
                "entity_id": eid,
                "canonical_name": c["canonical_name"],
                "aliases": c["aliases"],
                "stats": {"freq": int(c["freq"])},
            }
        )
        for a in c["aliases"]:
            surface_to_eid[a] = eid

    return entities, surface_to_eid


def attach_entity_ids(
    capsules: List[Dict[str, Any]],
    surface_to_eid: Dict[str, str],
) -> None:
    for cap in capsules:
        ent_ids = []
        new_args = []
        for arg in cap.get("arguments", []) or []:
            if not isinstance(arg, dict):
                continue
            role = str(arg.get("role") or "").strip()
            surf = str(arg.get("surface") or "").strip()
            if not surf:
                continue
            eid = surface_to_eid.get(surf)
            if eid:
                ent_ids.append(eid)
            new_args.append({"role": role, "surface": surf, "entity_id": eid})
        cap["arguments"] = new_args
        cap["entity_ids"] = sorted(set([e for e in ent_ids if e]))
