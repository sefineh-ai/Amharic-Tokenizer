#!/usr/bin/env python3
"""Measure training and tokenization speed on a corpus file.

python benchmarks/benchmark.py                       # uses the test fixture corpus
python benchmarks/benchmark.py data/cleaned_data.txt --num-merges 30000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from amharic_tokenizer import AmharicTokenizer

DEFAULT_CORPUS = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "sample_corpus.txt"


def _timed(fn, *args):
    start = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("corpus", nargs="?", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--num-merges", type=int, default=5000)
    parser.add_argument("--model", default=None, help="model for tokenization (default: bundled)")
    args = parser.parse_args()

    text = args.corpus.read_text(encoding="utf-8")
    words = len(text.split())
    print(f"corpus: {args.corpus} ({words:,} words)")

    trainer = AmharicTokenizer(num_merges=args.num_merges)
    merges, seconds = _timed(trainer.train, text)
    print(f"train:    {merges:,} merges in {seconds:.2f}s")

    tok = AmharicTokenizer.load(args.model) if args.model else AmharicTokenizer.load()
    tokens, cold = _timed(tok.tokenize, text)
    _, warm = _timed(tok.tokenize, text)
    print(
        f"tokenize: {len(tokens):,} tokens, cold {cold:.3f}s ({words / cold:,.0f} words/s), "
        f"warm cache {warm:.3f}s"
    )
    ids, seconds = _timed(tok.encode, tokens)
    print(f"encode:   {seconds:.3f}s")
    _, seconds = _timed(tok.decode, ids)
    print(f"decode:   {seconds:.3f}s")


if __name__ == "__main__":
    main()
