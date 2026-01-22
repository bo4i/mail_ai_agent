from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Iterable

from functools import lru_cache

try:
    from pymorphy3 import MorphAnalyzer
except ImportError:
    try:
        from pymorphy2 import MorphAnalyzer
    except ImportError:
        MorphAnalyzer = None

_TOKEN_SPLIT_RE = re.compile(r"[^\wа-яё]+", re.IGNORECASE)
_HAS_LETTER_RE = re.compile(r"[a-zа-яё]", re.IGNORECASE)


@dataclass
class NormalizedText:
    lemma_list: list[str]
    lemma_set: set[str]
    lemma_string: str


def tokenize(text: str) -> list[str]:
    lowered = text.lower()
    cleaned = _TOKEN_SPLIT_RE.sub(" ", lowered)
    tokens: list[str] = []
    for token in cleaned.split():
        if len(token) <= 2:
            continue
        if not _HAS_LETTER_RE.search(token):
            continue
        if len(set(token)) == 1:
            continue
        if re.fullmatch(r"[\d_]+", token):
            continue
        tokens.append(token)
    return tokens


@lru_cache(maxsize=1)
def _get_morph() -> MorphAnalyzer | None:
    if MorphAnalyzer is None:
        return None
    try:
        return MorphAnalyzer()
    except Exception:
        return None


def lemmatize_tokens(tokens: Iterable[str]) -> list[str]:
    morph = _get_morph()
    if not morph:
        return list(tokens)
    return [morph.parse(token)[0].normal_form for token in tokens]


def normalize_text(text: str) -> NormalizedText:
    tokens = tokenize(text)
    lemmas = lemmatize_tokens(tokens)
    lemma_set = set(lemmas)
    lemma_string = " ".join(lemmas)
    return NormalizedText(lemma_list=lemmas, lemma_set=lemma_set, lemma_string=lemma_string)


def top_terms(texts: Iterable[str], *, max_terms: int, stopwords: set[str]) -> list[str]:
    tokens: list[str] = []
    for text in texts:
        tokens.extend(tokenize(text))
    lemmas = lemmatize_tokens(tokens)
    filtered = [lemma for lemma in lemmas if lemma not in stopwords]
    counts = Counter(filtered)
    return [term for term, _ in counts.most_common(max_terms)]
