"""Language-neutral normalization and lexical search features."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from typing import Any


def normalize_search_text(value: Any) -> str:
    """Normalize every Unicode script without a language-specific dictionary."""
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    characters = []
    for character in normalized:
        category = unicodedata.category(character)
        characters.append(
            character if category[0] in {"L", "M", "N"} else " "
        )
    return " ".join("".join(characters).split())


def search_segments(value: Any) -> list[str]:
    return re.findall(r"[^\W_]+", normalize_search_text(value), flags=re.UNICODE)


def search_features(value: Any) -> Counter[str]:
    """Return words plus overlapping character bigrams/trigrams."""
    features: Counter[str] = Counter()
    for segment in search_segments(value):
        features[f"w:{segment}"] += 1
        for size in (2, 3):
            if len(segment) < size:
                continue
            for index in range(len(segment) - size + 1):
                features[f"c{size}:{segment[index:index + size]}"] += 1
    return features


def search_feature_set(value: Any) -> set[str]:
    return set(search_features(value))


def search_index_terms(value: Any) -> list[str]:
    """Return compact, deterministic Unicode bigram terms for FTS indexing."""
    terms = []
    for segment in search_segments(value):
        for index in range(len(segment) - 1):
            bigram = segment[index:index + 2]
            digest = hashlib.blake2s(
                bigram.encode("utf-8"),
                digest_size=8,
            ).hexdigest()
            terms.append(f"ng2{digest}")
    return terms


def search_index_text(value: Any) -> str:
    normalized = normalize_search_text(value)
    terms = search_index_terms(normalized)
    return " ".join([normalized, *terms]) if terms else normalized
