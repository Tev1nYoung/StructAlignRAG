from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Tuple


_STOPWORDS = {
    # Articles / prepositions / common function words
    "a",
    "an",
    "the",
    "of",
    "for",
    "and",
    "or",
    "to",
    "by",
    "from",
    "as",
    "in",
    "on",
    "at",
    "de",
    "la",
    "von",
    "van",
    "&",
}


def tokenize_title(text: str) -> List[str]:
    """
    Tokenize a Wikipedia title-ish string into lowercase tokens.
    Keeps only [a-z0-9] sequences, drops empties.
    """
    if not text:
        return []
    toks = re.findall(r"[A-Za-z0-9]+", str(text).lower())
    return [t for t in toks if t]


def build_title_token_idf(titles: Iterable[str]) -> Dict[str, float]:
    """
    Unsupervised corpus-level token IDF over titles.
    IDF is computed over title-document frequency (df).
    """
    titles_list = list(titles)
    n = len(titles_list)
    if n <= 0:
        return {}

    df: Dict[str, int] = {}
    for t in titles_list:
        seen = set(tokenize_title(t))
        for tok in seen:
            df[tok] = df.get(tok, 0) + 1

    idf: Dict[str, float] = {}
    for tok, c in df.items():
        # Smooth to avoid div-by-zero and to keep values in a reasonable range.
        # token appears in all titles -> idf ~= 1.0
        # token appears once -> idf ~= log((n+1)/2)+1
        idf[tok] = math.log((n + 1.0) / (c + 1.0)) + 1.0
    return idf


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return float(min(values))
    if q >= 100:
        return float(max(values))
    vs = sorted(float(v) for v in values)
    k = (len(vs) - 1) * (q / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(vs[int(k)])
    d0 = vs[int(f)] * (c - k)
    d1 = vs[int(c)] * (k - f)
    return float(d0 + d1)


def title_avg_idf(title: str, token_idf: Dict[str, float], default_idf: float) -> float:
    toks = [t for t in tokenize_title(title) if t and t not in _STOPWORDS]
    if not toks:
        return 0.0
    s = 0.0
    for t in toks:
        s += float(token_idf.get(t, default_idf))
    return s / float(len(toks))


def title_stopword_ratio(title: str) -> float:
    toks = tokenize_title(title)
    if not toks:
        return 1.0
    sw = sum(1 for t in toks if t in _STOPWORDS)
    return float(sw) / float(len(toks))


def title_genericness_score(
    title: str,
    token_idf: Dict[str, float],
    idf_p10: float,
    idf_p90: float,
    text_snippet: str | None = None,
) -> float:
    """
    Return a genericness score in [0, 1], where larger means "more generic / less entity-like".

    - Uses title token IDF (lower avg IDF => more generic).
    - Uses stopword ratio in title.
    - Uses light disambiguation/list detection from title and optional snippet.

    This is intentionally unsupervised and dataset-agnostic.
    """
    t = str(title or "").strip()
    if not t:
        return 1.0

    t_low = t.lower()
    snip_low = str(text_snippet or "").strip().lower()

    is_disamb = ("(disambiguation)" in t_low) or ("may refer to" in snip_low[:200]) or ("may also refer to" in snip_low[:200])
    is_list = t_low.startswith("list of ") or t_low.startswith("index of ") or t_low.endswith(" (list)")

    default_idf = float(idf_p90 if idf_p90 > 0 else 6.0)
    avg_idf = title_avg_idf(t, token_idf, default_idf=default_idf)
    if idf_p90 > idf_p10 + 1e-9:
        idf_scaled = (avg_idf - idf_p10) / (idf_p90 - idf_p10)
    else:
        idf_scaled = 0.5
    idf_scaled = max(0.0, min(1.0, float(idf_scaled)))
    generic_from_idf = 1.0 - idf_scaled

    sw_ratio = title_stopword_ratio(t)

    # Base genericness from token statistics.
    score = 0.70 * generic_from_idf + 0.30 * sw_ratio

    # Disambiguation/list pages are almost always retrieval traps.
    if is_list:
        score += 0.25
    if is_disamb:
        score += 0.40

    return max(0.0, min(1.0, float(score)))


def build_doc_genericness(
    titles_and_snippets: List[Tuple[str, str]],
) -> Tuple[Dict[str, float], float, float, Dict[int, float]]:
    """
    Convenience helper:
    - builds token idf over titles
    - computes idf p10/p90 over per-title avg idf
    - computes a per-doc genericness score
    Returns (token_idf, idf_p10, idf_p90, genericness_by_idx)
    """
    titles = [t for t, _snip in titles_and_snippets]
    token_idf = build_title_token_idf(titles)

    # Compute avg-idf distribution for robust scaling.
    if titles:
        # Use a stable default for missing tokens when computing the distribution: p90 will be computed from the list anyway.
        tmp_default = 6.0
        avg_idfs = [title_avg_idf(t, token_idf, default_idf=tmp_default) for t in titles]
    else:
        avg_idfs = []
    idf_p10 = _percentile(avg_idfs, 10.0)
    idf_p90 = _percentile(avg_idfs, 90.0)
    if idf_p90 <= 0.0:
        idf_p90 = 6.0

    genericness_by_idx: Dict[int, float] = {}
    for i, (t, snip) in enumerate(titles_and_snippets):
        genericness_by_idx[i] = title_genericness_score(t, token_idf, idf_p10, idf_p90, text_snippet=snip)
    return token_idf, float(idf_p10), float(idf_p90), genericness_by_idx

