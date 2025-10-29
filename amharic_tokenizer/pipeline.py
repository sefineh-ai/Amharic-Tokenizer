"""Pipeline module to train and save the Amharic BPE tokenizer."""

from __future__ import annotations

from pathlib import Path

from .import AmharicTokenizer

def train_and_save(
    corpus_path: str,
    output_prefix: str,
    num_merges: int = 50000,
    verbose: bool = False,
    log_every: int = 1000,
) -> int:
    """Train the tokenizer on a corpus file and save the model.

    Args:
        corpus_path: Path to UTF-8 text corpus (one big text file is fine).
        output_prefix: Path prefix for output JSON model (without extension).
        num_merges: Number of BPE merges to learn (higher → longer subwords).
    """
    path = Path(corpus_path)
    text = path.read_text(encoding="utf-8")

    tokenizer = AmharicTokenizer(num_merges=num_merges)
    learned = tokenizer.train(text, verbose=verbose, log_every=log_every)
    tokenizer.save(output_prefix)
    return learned
