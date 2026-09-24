"""Amharic BPE tokenizer over decomposed fidel, with a Cython core."""

from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("amharic-tokenizer")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0+unknown"

from .exceptions import (
    AmharicTokenizerError,
    InvalidModelError,
    ModelNotFoundError,
    TrainingError,
)
from .fidel import AMHARIC_FIDEL_MAP, REVERSE_FIDEL_MAP, compose, decompose
from .serialization import list_pretrained_models
from .tokenizer import DEFAULT_MODEL, AmharicTokenizer
from .training import TrainingResult, train_and_save, train_from_file
from .vocabulary import EOW_TOKEN, UNK_TOKEN, Vocabulary

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "AMHARIC_FIDEL_MAP",
    "DEFAULT_MODEL",
    "EOW_TOKEN",
    "REVERSE_FIDEL_MAP",
    "UNK_TOKEN",
    "AmharicTokenizer",
    "AmharicTokenizerError",
    "InvalidModelError",
    "ModelNotFoundError",
    "TrainingError",
    "TrainingResult",
    "Vocabulary",
    "__version__",
    "compose",
    "decompose",
    "list_pretrained_models",
    "train_and_save",
    "train_from_file",
]
