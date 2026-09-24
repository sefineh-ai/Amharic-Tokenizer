"""Reading and writing tokenizer models (JSON), and locating bundled pretrained models."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Union

from .exceptions import InvalidModelError, ModelNotFoundError

if TYPE_CHECKING:
    from importlib.abc import Traversable

MODEL_SUFFIX = ".json"
FORMAT_VERSION = 1
_MODELS_PACKAGE = "amharic_tokenizer"
_MODELS_DIR = "models"

PathLike = Union[str, Path]

_REQUIRED_KEYS = ("vocabulary", "merge_rank_map", "token_to_id", "id_to_token", "next_id")


@dataclass
class ModelState:
    """Everything needed to reconstruct a trained tokenizer."""

    num_merges: int
    max_vocab_size: int
    vocabulary: Dict[str, int]
    merge_rank_map: Dict[str, int]
    token_to_id: Dict[str, int]
    id_to_token: Dict[int, str]
    next_id: int
    format_version: int = field(default=FORMAT_VERSION)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_version": self.format_version,
            "num_merges": self.num_merges,
            "max_vocab_size": self.max_vocab_size,
            "vocabulary": self.vocabulary,
            "merge_rank_map": self.merge_rank_map,
            "token_to_id": self.token_to_id,
            # JSON object keys must be strings.
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()},
            "next_id": self.next_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], defaults: Mapping[str, int]) -> ModelState:
        if not isinstance(data, Mapping):
            raise InvalidModelError("model JSON must be an object")
        missing = [key for key in _REQUIRED_KEYS if key not in data]
        if missing:
            raise InvalidModelError(f"model is missing required keys: {', '.join(missing)}")
        try:
            state = cls(
                num_merges=int(data.get("num_merges", defaults["num_merges"])),
                max_vocab_size=int(data.get("max_vocab_size", defaults["max_vocab_size"])),
                vocabulary={str(k): int(v) for k, v in data["vocabulary"].items()},
                merge_rank_map={str(k): int(v) for k, v in data["merge_rank_map"].items()},
                token_to_id={str(k): int(v) for k, v in data["token_to_id"].items()},
                id_to_token={int(k): str(v) for k, v in data["id_to_token"].items()},
                next_id=int(data["next_id"]),
                format_version=int(data.get("format_version", FORMAT_VERSION)),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise InvalidModelError(f"model has malformed fields: {exc}") from exc
        if state.format_version > FORMAT_VERSION:
            raise InvalidModelError(
                f"model format version {state.format_version} is newer than supported "
                f"version {FORMAT_VERSION}; upgrade amharic-tokenizer"
            )
        if len(state.token_to_id) != len(state.id_to_token):
            raise InvalidModelError("token_to_id and id_to_token have different sizes")
        return state


def _with_suffix(name: str) -> str:
    return name if name.endswith(MODEL_SUFFIX) else name + MODEL_SUFFIX


def _bundled_models_dir() -> Traversable:
    return resources.files(_MODELS_PACKAGE).joinpath(_MODELS_DIR)


def list_pretrained_models() -> List[str]:
    """Names of the models shipped with the package, e.g. ``["amh_bpe_v0.2.7", ...]``."""
    return sorted(
        entry.name[: -len(MODEL_SUFFIX)]
        for entry in _bundled_models_dir().iterdir()
        if entry.name.endswith(MODEL_SUFFIX)
    )


def read_model_text(name_or_path: PathLike) -> str:
    """Return the JSON text of a model.

    ``name_or_path`` is tried as a filesystem path first (``.json`` is appended
    when missing), then as the name of a bundled pretrained model.
    """
    path = Path(_with_suffix(str(name_or_path)))
    if path.is_file():
        return path.read_text(encoding="utf-8")
    bundled = _bundled_models_dir().joinpath(path.name)
    if bundled.is_file():
        return bundled.read_text(encoding="utf-8")
    available = ", ".join(list_pretrained_models()) or "none"
    raise ModelNotFoundError(
        f"model '{name_or_path}' not found at '{path}' or among bundled models ({available})"
    )


def read_model(name_or_path: PathLike, defaults: Mapping[str, int]) -> ModelState:
    text = read_model_text(name_or_path)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InvalidModelError(f"model '{name_or_path}' is not valid JSON: {exc}") from exc
    return ModelState.from_dict(data, defaults)


def write_model(state: ModelState, path: PathLike) -> Path:
    """Write ``state`` as JSON (``.json`` appended when missing) and return the final path."""
    target = Path(_with_suffix(str(path)))
    if target.parent != Path(""):
        target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        json.dump(state.to_dict(), fh, ensure_ascii=False, indent=4)
    return target
