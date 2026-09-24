"""File-to-file training pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .exceptions import TrainingError
from .serialization import PathLike
from .tokenizer import (
    DEFAULT_MAX_VOCAB_SIZE,
    DEFAULT_NUM_MERGES,
    AmharicTokenizer,
    ProgressCallback,
)


@dataclass(frozen=True)
class TrainingResult:
    tokenizer: AmharicTokenizer
    model_path: Path
    num_merges_learned: int


def train_from_file(
    corpus_path: PathLike,
    output_path: PathLike,
    num_merges: int = DEFAULT_NUM_MERGES,
    max_vocab_size: int = DEFAULT_MAX_VOCAB_SIZE,
    verbose: bool = False,
    log_every: int = 1000,
    progress: Optional[ProgressCallback] = None,
) -> TrainingResult:
    """Train a tokenizer on a UTF-8 corpus file and save it to ``output_path`` (.json)."""
    corpus_file = Path(corpus_path)
    if not corpus_file.is_file():
        raise TrainingError(f"corpus file not found: {corpus_file}")
    text = corpus_file.read_text(encoding="utf-8")
    tokenizer = AmharicTokenizer(num_merges=num_merges, max_vocab_size=max_vocab_size)
    learned = tokenizer.train(text, verbose=verbose, log_every=log_every, progress=progress)
    model_path = tokenizer.save(output_path)
    return TrainingResult(tokenizer=tokenizer, model_path=model_path, num_merges_learned=learned)


def train_and_save(
    corpus_path: PathLike,
    output_prefix: PathLike,
    num_merges: int = DEFAULT_NUM_MERGES,
    verbose: bool = False,
    log_every: int = 1000,
) -> int:
    """Backward-compatible wrapper around :func:`train_from_file`; returns the merge count."""
    return train_from_file(
        corpus_path, output_prefix, num_merges=num_merges, verbose=verbose, log_every=log_every
    ).num_merges_learned
