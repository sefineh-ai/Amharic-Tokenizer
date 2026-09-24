"""Text normalization for training corpora."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Iterator, Set, Union

_LATIN_OR_DIGIT = re.compile(r"[A-Za-z0-9]")
_NON_ETHIOPIC = re.compile(r"[^ሀ-፿\s]")
_WHITESPACE = re.compile(r"\s+")
_ETHIOPIC = re.compile(r"[ሀ-፿]")
# Latin letters, and ASCII symbols that never belong in Amharic prose.
_NOISE = re.compile(r"[A-Za-z@#$%^&*_=+{}\[\]|\\<>~,]+")
# Amharic and common sentence/clause delimiters.
_SENTENCE_BOUNDARY = re.compile(r"[።፤፣፥፧?!\"']+")


def normalize_for_training(text: str) -> str:
    """Keep only Ethiopic characters, collapsing all whitespace to single spaces."""
    text = _LATIN_OR_DIGIT.sub("", text)
    text = _NON_ETHIOPIC.sub("", text)
    return _WHITESPACE.sub(" ", text).strip()


def iter_clean_sentences(lines: Iterable[str], min_length: int = 15) -> Iterator[str]:
    """Yield unique sentences from raw text lines.

    Latin letters and stray ASCII symbols are dropped, lines are split on
    Amharic punctuation, and sentences shorter than ``min_length`` characters
    or without any Ethiopic character are skipped. Duplicates are yielded once.
    """
    seen: Set[str] = set()
    for line in lines:
        text = _WHITESPACE.sub(" ", _NOISE.sub(" ", line)).strip()
        if not text:
            continue
        for sentence in _SENTENCE_BOUNDARY.split(text):
            sentence = sentence.strip()
            if len(sentence) >= min_length and sentence not in seen and _ETHIOPIC.search(sentence):
                seen.add(sentence)
                yield sentence


def clean_corpus_file(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    min_length: int = 15,
) -> int:
    """Write the sentences of :func:`iter_clean_sentences` one per line; return how many."""
    count = 0
    with open(input_path, encoding="utf-8") as src, open(output_path, "w", encoding="utf-8") as dst:
        for sentence in iter_clean_sentences(src, min_length=min_length):
            dst.write(sentence + "\n")
            count += 1
    return count
