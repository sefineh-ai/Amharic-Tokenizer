"""Exception hierarchy for the amharic_tokenizer package.

Every error raised deliberately by the package derives from
:class:`AmharicTokenizerError`, so callers can catch a single base class.
"""

from __future__ import annotations


class AmharicTokenizerError(Exception):
    """Base class for all errors raised by amharic_tokenizer."""


class ModelNotFoundError(AmharicTokenizerError, FileNotFoundError):
    """A model could not be found on disk or among the bundled pretrained models."""


class InvalidModelError(AmharicTokenizerError, ValueError):
    """A model file exists but its content is not a valid tokenizer state."""


class TrainingError(AmharicTokenizerError, ValueError):
    """Training was requested with invalid parameters or input."""
