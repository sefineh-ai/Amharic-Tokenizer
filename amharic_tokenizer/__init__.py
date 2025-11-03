"""Top-level package imports for amharic_tokenizer."""

from .fidel_map import AMHARIC_FIDEL_MAP, REVERSE_FIDEL_MAP
from .tokenizer import AmharicTokenizer

__all__ = ["AmharicTokenizer", "AMHARIC_FIDEL_MAP", "REVERSE_FIDEL_MAP"]
