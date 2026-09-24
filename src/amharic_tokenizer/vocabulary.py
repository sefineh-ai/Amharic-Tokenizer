"""Token vocabulary: token <-> id mapping plus the corpus frequency of each token."""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Mapping

from .fidel import base_symbols

#: Appended to every word; merges never cross it, and it becomes a space on decode.
EOW_TOKEN = "<eow>"
#: Id used for tokens that are not in the vocabulary.
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = (EOW_TOKEN, UNK_TOKEN)


class Vocabulary:
    """Bidirectional token/id mapping.

    Ids are assigned in insertion order and never reused, so a token's id is
    stable for the lifetime of a model.
    """

    def __init__(self) -> None:
        self._frequencies: Dict[str, int] = {}
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        self._next_id = 0

    @classmethod
    def base(cls) -> Vocabulary:
        """Vocabulary of every decomposed fidel symbol plus the special tokens, sorted."""
        vocab = cls()
        for token in sorted({*base_symbols(), *SPECIAL_TOKENS}):
            vocab.add(token, 0)
        return vocab

    @classmethod
    def from_mappings(
        cls,
        frequencies: Mapping[str, int],
        token_to_id: Mapping[str, int],
        id_to_token: Mapping[int, str],
        next_id: int,
    ) -> Vocabulary:
        vocab = cls()
        vocab._frequencies = dict(frequencies)
        vocab._token_to_id = dict(token_to_id)
        vocab._id_to_token = dict(id_to_token)
        vocab._next_id = next_id
        return vocab

    def add(self, token: str, frequency: int = 0) -> int:
        """Add ``token`` if it is new and return its id."""
        if token not in self._frequencies:
            self._frequencies[token] = frequency
        token_id = self._token_to_id.get(token)
        if token_id is None:
            token_id = self._next_id
            self._token_to_id[token] = token_id
            self._id_to_token[token_id] = token
            self._next_id += 1
        return token_id

    @property
    def unk_id(self) -> int:
        return self._token_to_id.get(UNK_TOKEN, -1)

    @property
    def next_id(self) -> int:
        return self._next_id

    @property
    def frequencies(self) -> Mapping[str, int]:
        return self._frequencies

    @property
    def token_to_id_map(self) -> Mapping[str, int]:
        return self._token_to_id

    @property
    def id_to_token_map(self) -> Mapping[int, str]:
        return self._id_to_token

    def token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self.unk_id)

    def id_to_token(self, token_id: int) -> str:
        return self._id_to_token.get(token_id, UNK_TOKEN)

    def tokens_to_ids(self, tokens: Iterable[str]) -> List[int]:
        lookup = self._token_to_id
        unk_id = self.unk_id
        return [lookup.get(token, unk_id) for token in tokens]

    def ids_to_tokens(self, ids: Iterable[int]) -> List[str]:
        lookup = self._id_to_token
        return [lookup.get(i, UNK_TOKEN) for i in ids]

    def __contains__(self, token: object) -> bool:
        return token in self._token_to_id

    def __len__(self) -> int:
        return len(self._token_to_id)

    def __iter__(self) -> Iterator[str]:
        return iter(self._token_to_id)
