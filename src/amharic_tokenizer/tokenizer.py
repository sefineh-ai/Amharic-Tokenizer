"""Public tokenizer API."""

from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from ._bpe import BPEEncoder, learn_merges
from .exceptions import TrainingError
from .fidel import compose
from .normalization import normalize_for_training
from .serialization import ModelState, PathLike, read_model, write_model
from .vocabulary import EOW_TOKEN, UNK_TOKEN, Vocabulary

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "amh_bpe_v0.2.7"
DEFAULT_NUM_MERGES = 50000
DEFAULT_MAX_VOCAB_SIZE = 30000

ProgressCallback = Callable[[str], None]


def _print_progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


class AmharicTokenizer:
    """Byte-pair-encoding tokenizer over decomposed Amharic fidel.

    Text is split on whitespace, every fidel is decomposed into consonant +
    vowel symbols, each word gets an ``<eow>`` marker, and learned merges are
    applied within each word. :meth:`detokenize` reverses the process.

    Example::

        tok = AmharicTokenizer.load()          # bundled pretrained model
        ids = tok.encode("ኢትዮጵያ ጥሩ ናት")
        text = tok.decode(ids)
    """

    def __init__(
        self,
        num_merges: int = DEFAULT_NUM_MERGES,
        max_vocab_size: int = DEFAULT_MAX_VOCAB_SIZE,
    ) -> None:
        if num_merges < 0:
            raise TrainingError(f"num_merges must be >= 0, got {num_merges}")
        if max_vocab_size <= 0:
            raise TrainingError(f"max_vocab_size must be > 0, got {max_vocab_size}")
        self._num_merges = num_merges
        self._max_vocab_size = max_vocab_size
        self._vocab = Vocabulary.base()
        self._merge_ranks: Dict[str, int] = {}
        self._encoder: Optional[BPEEncoder] = None

    # ------------------------------------------------------------------ training

    def train(
        self,
        corpus: str,
        verbose: bool = False,
        log_every: int = 1000,
        progress: Optional[ProgressCallback] = None,
    ) -> int:
        """Learn merges from ``corpus`` and return the total number of learned merges.

        Only Ethiopic characters of ``corpus`` are used. Merges are learned from
        scratch on ``corpus`` (up to ``num_merges`` iterations); tokens that are
        already known keep their ids and ranks and only new tokens are appended,
        so calling this on a trained model extends it rather than replacing it.
        Progress lines go to ``progress`` if given, else to stderr when ``verbose``.
        """
        if log_every <= 0:
            raise TrainingError(f"log_every must be > 0, got {log_every}")
        if progress is None and verbose:
            progress = _print_progress
        # Counter keeps first-occurrence order, which breaks ties between equal counts.
        word_freqs = Counter(normalize_for_training(corpus).split())
        learned = learn_merges(
            word_freqs,
            self._vocab,
            self._num_merges,
            self._max_vocab_size,
            progress,
            log_every,
        )
        for token, count in learned:
            self._merge_ranks[token] = len(self._merge_ranks) + 1
            self._vocab.add(token, count)
        self._encoder = None
        logger.debug("learned %d merges (%d total)", len(learned), len(self._merge_ranks))
        return len(self._merge_ranks)

    # ------------------------------------------------------------------ inference

    def tokenize(self, text: str) -> List[str]:
        """Split ``text`` into subword tokens; each word's last token ends with ``<eow>``."""
        if self._encoder is None:
            self._encoder = BPEEncoder(self._merge_ranks)
        return self._encoder.tokenize(text)

    def encode(self, text: Union[str, Iterable[str]]) -> List[int]:
        """Encode a string, or a token list from :meth:`tokenize`, to ids."""
        tokens = self.tokenize(text) if isinstance(text, str) else text
        return self._vocab.tokens_to_ids(tokens)

    def convert_tokens_to_ids(self, tokens: Iterable[str]) -> List[int]:
        return self._vocab.tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, ids: Iterable[int]) -> List[str]:
        return self._vocab.ids_to_tokens(ids)

    def decode(self, ids: Iterable[int]) -> str:
        return self.detokenize(self._vocab.ids_to_tokens(ids))

    def detokenize(self, tokens: Sequence[str]) -> str:
        """Join tokens back into text, recomposing fidel and dropping ``<unk>``."""
        words = "".join(tokens).replace(EOW_TOKEN, " ").split()
        return compose(" ".join(words)).replace(UNK_TOKEN, "")

    # ------------------------------------------------------------------ inspection

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def num_merges(self) -> int:
        return self._num_merges

    @property
    def max_vocab_size(self) -> int:
        return self._max_vocab_size

    @property
    def merges(self) -> Mapping[str, int]:
        """Read-only view of learned tokens and their merge ranks."""
        return MappingProxyType(self._merge_ranks)

    @property
    def vocabulary(self) -> Vocabulary:
        return self._vocab

    def is_trained(self) -> bool:
        return bool(self._merge_ranks)

    def token_to_id(self, token: str) -> int:
        return self._vocab.token_to_id(token)

    def id_to_token(self, token_id: int) -> str:
        return self._vocab.id_to_token(token_id)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(vocab_size={self.vocab_size}, merges={len(self._merge_ranks)}, "
            f"num_merges={self._num_merges}, max_vocab_size={self._max_vocab_size})"
        )

    # ------------------------------------------------------------------ persistence

    def to_state(self) -> ModelState:
        return ModelState(
            num_merges=self._num_merges,
            max_vocab_size=self._max_vocab_size,
            vocabulary=dict(self._vocab.frequencies),
            merge_rank_map=dict(self._merge_ranks),
            token_to_id=dict(self._vocab.token_to_id_map),
            id_to_token=dict(self._vocab.id_to_token_map),
            next_id=self._vocab.next_id,
        )

    @classmethod
    def from_state(cls, state: ModelState) -> AmharicTokenizer:
        tokenizer = cls(num_merges=state.num_merges, max_vocab_size=state.max_vocab_size)
        tokenizer._vocab = Vocabulary.from_mappings(
            state.vocabulary, state.token_to_id, state.id_to_token, state.next_id
        )
        tokenizer._merge_ranks = dict(state.merge_rank_map)
        return tokenizer

    def save(self, path: PathLike) -> Path:
        """Save the model as JSON (``.json`` appended when missing); return the written path."""
        target = write_model(self.to_state(), path)
        logger.info("saved tokenizer to %s", target)
        return target

    @classmethod
    def load(cls, name_or_path: PathLike = DEFAULT_MODEL) -> AmharicTokenizer:
        """Load a model from a file path or by bundled model name (default: latest)."""
        state = read_model(
            name_or_path,
            defaults={"num_merges": DEFAULT_NUM_MERGES, "max_vocab_size": DEFAULT_MAX_VOCAB_SIZE},
        )
        tokenizer = cls.from_state(state)
        logger.info("loaded tokenizer from %s", name_or_path)
        return tokenizer
