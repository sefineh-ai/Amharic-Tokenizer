"""``amh-tokenizer`` command-line interface."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Optional, Sequence

from . import __version__
from .exceptions import AmharicTokenizerError
from .normalization import clean_corpus_file
from .serialization import list_pretrained_models
from .tokenizer import DEFAULT_MAX_VOCAB_SIZE, DEFAULT_MODEL, DEFAULT_NUM_MERGES, AmharicTokenizer
from .training import train_from_file

logger = logging.getLogger("amharic_tokenizer.cli")


def _read_text(args: argparse.Namespace) -> str:
    return args.text if args.text is not None else sys.stdin.read()


def _cmd_train(args: argparse.Namespace) -> int:
    logger.info("training on %s (num_merges=%d)", args.corpus, args.num_merges)
    result = train_from_file(
        args.corpus,
        args.output,
        num_merges=args.num_merges,
        max_vocab_size=args.max_vocab_size,
        verbose=args.verbose,
        log_every=args.log_every,
    )
    print(f"Saved model to {result.model_path} with {result.num_merges_learned} merges")
    return 0


def _cmd_clean(args: argparse.Namespace) -> int:
    count = clean_corpus_file(args.input, args.output, min_length=args.min_length)
    print(f"Wrote {count} sentences to {args.output}")
    return 0


def _cmd_tokenize(args: argparse.Namespace) -> int:
    tokens = AmharicTokenizer.load(args.model).tokenize(_read_text(args))
    print(json.dumps(tokens, ensure_ascii=False))
    return 0


def _cmd_encode(args: argparse.Namespace) -> int:
    print(json.dumps(AmharicTokenizer.load(args.model).encode(_read_text(args))))
    return 0


def _cmd_decode(args: argparse.Namespace) -> int:
    ids = [int(i) for i in args.ids]
    print(AmharicTokenizer.load(args.model).decode(ids))
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    for name in list_pretrained_models():
        print(name)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="amh-tokenizer", description="Amharic BPE tokenizer over decomposed fidel"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="show progress and debug logs")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train a tokenizer on a corpus and save it")
    p.add_argument("corpus", help="path to a cleaned UTF-8 Amharic corpus")
    p.add_argument("output", help="output model path (.json is appended when missing)")
    p.add_argument("--num-merges", type=int, default=DEFAULT_NUM_MERGES)
    p.add_argument("--max-vocab-size", type=int, default=DEFAULT_MAX_VOCAB_SIZE)
    p.add_argument("--log-every", type=int, default=1000, help="progress interval in merges")
    # Accepted after the subcommand too (legacy syntax); SUPPRESS keeps a global -v intact.
    p.add_argument(
        "--verbose", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS
    )
    p.set_defaults(handler=_cmd_train)

    p = sub.add_parser("clean", help="extract unique Amharic sentences from raw text")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--min-length", type=int, default=15, help="minimum sentence length")
    p.set_defaults(handler=_cmd_clean)

    for name, handler, help_text in (
        ("tokenize", _cmd_tokenize, "print tokens as JSON"),
        ("encode", _cmd_encode, "print token ids as JSON"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("text", nargs="?", help="text to process (default: stdin)")
        p.add_argument("-m", "--model", default=DEFAULT_MODEL, help="model path or bundled name")
        p.set_defaults(handler=handler)

    p = sub.add_parser("decode", help="decode token ids back to text")
    p.add_argument("ids", nargs="+")
    p.add_argument("-m", "--model", default=DEFAULT_MODEL, help="model path or bundled name")
    p.set_defaults(handler=_cmd_decode)

    p = sub.add_parser("models", help="list bundled pretrained models")
    p.set_defaults(handler=_cmd_models)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="[amh-tokenizer] %(levelname)s %(message)s",
    )
    try:
        return int(args.handler(args))
    except (AmharicTokenizerError, OSError) as exc:
        print(f"amh-tokenizer: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
