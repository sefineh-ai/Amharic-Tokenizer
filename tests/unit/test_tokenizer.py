import pytest

from amharic_tokenizer import AmharicTokenizer, TrainingError
from amharic_tokenizer.vocabulary import EOW_TOKEN, UNK_TOKEN

TOY_CORPUS = "ሰላም ሰላም ሰላማዊ ሰላምታ ሰላም ዓለም ዓለም ዓለማዊ"


@pytest.fixture
def toy():
    tok = AmharicTokenizer(num_merges=50)
    tok.train(TOY_CORPUS)
    return tok


def test_new_tokenizer_is_untrained():
    tok = AmharicTokenizer()
    assert not tok.is_trained()
    assert tok.vocab_size == 45
    assert tok.merges == {}


def test_untrained_tokenizer_emits_decomposed_symbols():
    assert AmharicTokenizer().tokenize("ሉ") == ["ለ", "ኡ", EOW_TOKEN]


@pytest.mark.parametrize(("kwargs"), [{"num_merges": -1}, {"max_vocab_size": 0}])
def test_invalid_constructor_arguments(kwargs):
    with pytest.raises(TrainingError):
        AmharicTokenizer(**kwargs)


def test_train_returns_total_merges_and_updates_vocab(toy):
    assert toy.is_trained()
    assert toy.vocab_size == 45 + len(toy.merges)
    assert list(toy.merges.values()) == list(range(1, len(toy.merges) + 1))


def test_train_rejects_bad_log_every():
    with pytest.raises(TrainingError):
        AmharicTokenizer().train(TOY_CORPUS, log_every=0)


def test_train_ignores_non_ethiopic_text():
    tok = AmharicTokenizer(num_merges=50)
    assert tok.train("hello world 123 hello world 123") == 0


def test_retraining_on_same_corpus_is_idempotent():
    tok = AmharicTokenizer(num_merges=2)
    assert tok.train(TOY_CORPUS) == 2
    state = tok.to_state()
    assert tok.train(TOY_CORPUS) == 2
    assert tok.to_state() == state


def test_retraining_keeps_existing_merges_and_appends_new_ones():
    tok = AmharicTokenizer(num_merges=2)
    tok.train(TOY_CORPUS)
    first = dict(tok.merges)
    tok.train("ቡና ቡና ቡና")
    assert dict(list(tok.merges.items())[:2]) == first
    assert len(tok.merges) > 2


def test_train_verbose_writes_progress_to_stderr(capsys):
    AmharicTokenizer(num_merges=3).train(TOY_CORPUS, verbose=True, log_every=1)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Training completed" in captured.err


def test_train_progress_callback():
    messages = []
    AmharicTokenizer(num_merges=3).train(TOY_CORPUS, progress=messages.append)
    assert any("Training completed" in m for m in messages)


def test_training_invalidates_encoder(toy):
    before = toy.tokenize("ሰላም")
    tok = AmharicTokenizer(num_merges=50)
    assert tok.tokenize("ሰላም") != before  # untrained: symbols only
    tok.train(TOY_CORPUS)
    assert tok.tokenize("ሰላም") == before


def test_merged_word_is_one_token(toy):
    assert toy.tokenize("ሰላም") == ["ሰአለኣመእ" + EOW_TOKEN]


def test_roundtrip(toy):
    text = "ሰላም ዓለም ሰላማዊ"
    assert toy.detokenize(toy.tokenize(text)) == text
    assert toy.decode(toy.encode(text)) == text


def test_encode_accepts_tokens(toy):
    tokens = toy.tokenize("ሰላም ዓለም")
    assert toy.encode(tokens) == toy.encode("ሰላም ዓለም")
    assert toy.convert_tokens_to_ids(tokens) == toy.encode(tokens)


def test_convert_ids_to_tokens(toy):
    tokens = toy.tokenize("ሰላም")
    assert toy.convert_ids_to_tokens(toy.encode(tokens)) == tokens


def test_unknown_symbols_encode_to_unk(toy):
    ids = toy.encode("A")
    assert ids == [toy.token_to_id(UNK_TOKEN), toy.token_to_id(EOW_TOKEN)]
    assert toy.id_to_token(ids[0]) == UNK_TOKEN


def test_decode_drops_unk_and_unknown_ids(toy):
    assert toy.decode([toy.token_to_id(UNK_TOKEN), 10**9]) == ""


def test_detokenize_normalizes_whitespace(toy):
    assert toy.detokenize(["ሰአ", EOW_TOKEN, EOW_TOKEN, "ለአ", EOW_TOKEN]) == "ሰ ለ"


def test_empty_text(toy):
    assert toy.tokenize("") == []
    assert toy.encode("") == []
    assert toy.decode([]) == ""


def test_merges_view_is_read_only(toy):
    with pytest.raises(TypeError):
        toy.merges["x"] = 1  # type: ignore[index]


def test_state_roundtrip(toy):
    clone = AmharicTokenizer.from_state(toy.to_state())
    assert clone.to_state() == toy.to_state()
    assert clone.tokenize(TOY_CORPUS) == toy.tokenize(TOY_CORPUS)


def test_repr(toy):
    assert repr(toy).startswith("AmharicTokenizer(vocab_size=")
