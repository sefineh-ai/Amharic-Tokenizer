"""Top-level package imports for amharic_tokenizer."""

from .fidel_map import AMHARIC_FIDEL_MAP, reverse_fidel_map
from .tokenizer import AmharicTokenizer

__all__ = ["AmharicTokenizer", "AMHARIC_FIDEL_MAP", "reverse_fidel_map"]
