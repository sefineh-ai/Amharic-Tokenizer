"""Train -> save -> load, on a real corpus sample."""

import json

import pytest

from amharic_tokenizer import (
    AmharicTokenizer,
    TrainingError,
    train_and_save,
    train_from_file,
)


@pytest.mark.parametrize("case", ["merges_300", "vocab_cap"])
def test_training_matches_golden(case, golden, sample_corpus):
    expected = golden["trained"][case]
    tok = AmharicTokenizer(**expected["kwargs"])
    assert tok.train(sample_corpus) == expected["learned"]
    state = tok.to_state()
    assert state.merge_rank_map == expected["merge_rank_map"]
    assert state.vocabulary == expected["vocabulary"]
    assert state.token_to_id == expected["token_to_id"]
    assert tok.tokenize(expected["sample"]) == expected["tokens"]
    assert tok.encode(expected["sample"]) == expected["ids"]


def test_vocab_cap_is_respected(golden, sample_corpus):
    tok = AmharicTokenizer(num_merges=5000, max_vocab_size=420)
    tok.train(sample_corpus)
    assert tok.vocab_size == 420


def test_train_from_file_save_and_reload(tmp_path, fixtures_dir, trained_tokenizer, sample_corpus):
    result = train_from_file(
        fixtures_dir / "sample_corpus.txt", tmp_path / "out" / "model", num_merges=300
    )
    assert result.model_path == tmp_path / "out" / "model.json"
    assert result.num_merges_learned == 300

    reloaded = AmharicTokenizer.load(result.model_path)
    assert reloaded.to_state() == trained_tokenizer.to_state()
    assert reloaded.encode(sample_corpus) == trained_tokenizer.encode(sample_corpus)
    assert reloaded.decode(reloaded.encode("ሰላም ዓለም")) == "ሰላም ዓለም"


def test_saved_model_is_readable_json(tmp_path, trained_tokenizer):
    path = trained_tokenizer.save(tmp_path / "m")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data["token_to_id"]) == trained_tokenizer.vocab_size


def test_train_from_missing_file(tmp_path):
    with pytest.raises(TrainingError, match="not found"):
        train_from_file(tmp_path / "missing.txt", tmp_path / "m")


def test_legacy_train_and_save(tmp_path, fixtures_dir):
    assert train_and_save(fixtures_dir / "sample_corpus.txt", tmp_path / "m", num_merges=10) == 10
    assert (tmp_path / "m.json").is_file()
