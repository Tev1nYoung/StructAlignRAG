import re
import string
from typing import Iterable, List, Set


_PREFIX_TRIM_WORDS = {
    "introduction",
}

_CONNECTOR_WORDS = {
    "a", "an", "the",
    "of", "for", "and", "or", "to", "by", "from", "as",
    "in", "on", "at",
    "de", "la", "von", "van",
    "&",
}

_DROP_SINGLE_TOKEN_WORDS = {
    # discourse
    "introduction", "currently", "former", "present",
    "references", "external", "links", "see", "also",
    # pronouns
    "it", "he", "she", "they", "we", "i", "you", "his", "her", "their", "its",
    "this", "that", "these", "those",
    # generic roles
    "ceo", "cfo", "coo", "cto",
    "professor", "officer",
    # generic org nouns (very noisy alone)
    "center", "centre", "science", "law", "medicine", "department", "school",
    # months
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
}

_ACRONYM_BLOCKLIST = {"CEO", "CFO", "COO", "CTO"}

_DEMONYM_BLOCKLIST = {
    "american", "german", "french", "british", "chinese", "japanese", "korean", "italian", "spanish", "russian",
    "indian", "canadian", "australian",
}


def clean_wiki_text(text: str) -> str:
    """
    Minimal Wikipedia-style cleanup to improve heuristic extraction without affecting evaluation strings.
    Note: callers that need exact doc string for Recall@k should keep using raw text for 'doc_text'.
    """
    if not text:
        return ""

    s = text
    # Remove numeric citation markers like [1], [23]
    s = re.sub(r"\[\d+\]", "", s)

    # Drop leading section header "Introduction" (common in our corpora exports)
    s = re.sub(r"^\s*Introduction\s*\n+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^\s*Introduction\s+", "", s, flags=re.IGNORECASE)

    return s


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    # Simple sentence splitter (MVP). Keeps things deterministic and dependency-free.
    text = clean_wiki_text(text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    out = [p.strip() for p in parts if p and p.strip()]
    return out


def normalize_answer(answer: str) -> str:
    def remove_articles(t: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", t)

    def white_space_fix(t: str) -> str:
        return " ".join(t.split())

    def remove_punc(t: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in t if ch not in exclude)

    def lower(t: str) -> str:
        return t.lower()

    return white_space_fix(remove_articles(remove_punc(lower(answer or ""))))


def extract_entity_mentions(text: str) -> List[str]:
    """
    Heuristic entity mention extractor (MVP).
    Goal: build bridging nodes to connect evidence across docs without training.
    """
    if not text:
        return []

    text = clean_wiki_text(text)

    # Pattern A: strict TitleCase sequences (1-6 words): "Stanford University", "Thomas C. Sudhof"
    title_seq = re.findall(r"\b(?:[A-Z][a-zA-Z.]+)(?:\s+(?:[A-Z][a-zA-Z.]+)){0,5}\b", text)

    # Pattern B: allow connectors inside names: "Center for Science and Law"
    title_with_connectors = re.findall(
        r"\b[A-Z][a-zA-Z.]+(?:\s+(?:of|for|and|the|&|in|on|at|de|la|von|van)\s+[A-Z][a-zA-Z.]+){1,6}\b",
        text,
    )

    # Acronyms: "USA", "NATO"
    acronyms = re.findall(r"\b[A-Z]{2,}\b", text)

    mentions = []
    for m in title_with_connectors + title_seq + acronyms:
        m = m.strip().strip(string.punctuation)
        if not m:
            continue

        toks = [t for t in re.split(r"\s+", m) if t]
        while toks and toks[0].lower() in _PREFIX_TRIM_WORDS:
            toks = toks[1:]
        while toks and toks[-1].lower() in _PREFIX_TRIM_WORDS:
            toks = toks[:-1]
        if not toks:
            continue

        # If after trimming we start/end with connectors, drop/trim again.
        while toks and toks[0].lower() in _CONNECTOR_WORDS:
            toks = toks[1:]
        while toks and toks[-1].lower() in _CONNECTOR_WORDS:
            toks = toks[:-1]
        if not toks:
            continue

        m2 = " ".join(toks).strip()
        if not m2:
            continue

        # Demonyms are too generic and create spurious bridges.
        if m2.lower() in _DEMONYM_BLOCKLIST:
            continue

        # Too short => noise
        if len(m2) <= 2 and m2.isalpha():
            continue

        # Single-token mentions are very noisy. Keep only strong acronyms (>=3) and not blocklisted.
        if len(toks) == 1:
            tok = toks[0]
            if tok.upper() in _ACRONYM_BLOCKLIST:
                continue
            if tok.lower() in _DROP_SINGLE_TOKEN_WORDS:
                continue
            if tok.lower() in _DEMONYM_BLOCKLIST:
                continue

            # Strong acronyms or likely proper nouns (TitleCase/CamelCase) with reasonable length.
            if tok.isupper() and len(tok) >= 3:
                mentions.append(tok)
            elif len(tok) >= 4 and tok[:1].isupper():
                mentions.append(tok)
            continue

        # Multi-word mention: keep, but avoid spans that are all connectors/stopwords.
        # (We already trimmed connectors; here we just ensure there's at least one TitleCase token.)
        has_title = any(t[:1].isupper() for t in toks if t and t.lower() not in _CONNECTOR_WORDS)
        if not has_title:
            continue

        mentions.append(m2)

    # Dedup while keeping order
    seen: Set[str] = set()
    out: List[str] = []
    for m in mentions:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def normalize_entity(mention: str) -> str:
    if not mention:
        return ""
    s = mention.strip()
    s = s.strip(string.punctuation)
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def iter_unique(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out
