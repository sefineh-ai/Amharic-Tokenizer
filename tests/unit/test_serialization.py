import json

import pytest

from amharic_tokenizer import InvalidModelError, ModelNotFoundError
from amharic_tokenizer.serialization import (
    FORMAT_VERSION,
    ModelState,
    list_pretrained_models,
    read_model,
    read_model_text,
    write_model,
)

DEFAULTS = {"num_merges": 7, "max_vocab_size": 11}


def _state():
    return ModelState(
        num_merges=10,
        max_vocab_size=20,
        vocabulary={"a": 0, "ab": 3},
        merge_rank_map={"ab": 1},
        token_to_id={"a": 0, "ab": 1},
        id_to_token={0: "a", 1: "ab"},
        next_id=2,
    )


def test_write_then_read_roundtrip(tmp_path):
    path = write_model(_state(), tmp_path / "model")
    assert path.name == "model.json"
    assert read_model(path, DEFAULTS) == _state()
    assert read_model(tmp_path / "model", DEFAULTS) == _state()


def test_write_creates_parent_directories(tmp_path):
    path = write_model(_state(), tmp_path / "a" / "b" / "m.json")
    assert path.is_file()


def test_written_json_layout(tmp_path):
    data = json.loads(write_model(_state(), tmp_path / "m").read_text(encoding="utf-8"))
    assert data["format_version"] == FORMAT_VERSION
    assert data["id_to_token"] == {"0": "a", "1": "ab"}


def test_legacy_models_without_optional_keys_use_defaults():
    data = _state().to_dict()
    for key in ("format_version", "num_merges", "max_vocab_size"):
        del data[key]
    state = ModelState.from_dict(data, DEFAULTS)
    assert (state.num_merges, state.max_vocab_size) == (7, 11)


def test_missing_required_keys():
    data = _state().to_dict()
    del data["token_to_id"]
    with pytest.raises(InvalidModelError, match="token_to_id"):
        ModelState.from_dict(data, DEFAULTS)


def test_malformed_fields():
    data = _state().to_dict()
    data["id_to_token"] = {"not-an-int": "a"}
    with pytest.raises(InvalidModelError, match="malformed"):
        ModelState.from_dict(data, DEFAULTS)


def test_non_object_json():
    with pytest.raises(InvalidModelError):
        ModelState.from_dict([], DEFAULTS)  # type: ignore[arg-type]


def test_inconsistent_id_maps():
    data = _state().to_dict()
    data["id_to_token"] = {"0": "a"}
    with pytest.raises(InvalidModelError, match="different sizes"):
        ModelState.from_dict(data, DEFAULTS)


def test_newer_format_version_is_rejected():
    data = _state().to_dict()
    data["format_version"] = FORMAT_VERSION + 1
    with pytest.raises(InvalidModelError, match="newer"):
        ModelState.from_dict(data, DEFAULTS)


def test_invalid_json_file(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(InvalidModelError, match="not valid JSON"):
        read_model(bad, DEFAULTS)


def test_missing_model_lists_bundled_models(tmp_path):
    with pytest.raises(ModelNotFoundError, match=r"amh_bpe_v0\.2\.7") as excinfo:
        read_model_text(tmp_path / "nope")
    assert isinstance(excinfo.value, FileNotFoundError)


def test_list_pretrained_models():
    models = list_pretrained_models()
    assert "amh_bpe_v0.2.7" in models
    assert models == sorted(models)


def test_filesystem_path_takes_precedence_over_bundled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_model(_state(), "amh_bpe_v0.2.7")
    assert read_model("amh_bpe_v0.2.7", DEFAULTS) == _state()
