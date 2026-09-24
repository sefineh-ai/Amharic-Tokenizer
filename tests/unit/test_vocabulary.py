from amharic_tokenizer.fidel import base_symbols
from amharic_tokenizer.vocabulary import EOW_TOKEN, UNK_TOKEN, Vocabulary


def test_base_vocabulary_contents_and_order():
    vocab = Vocabulary.base()
    expected = sorted({*base_symbols(), EOW_TOKEN, UNK_TOKEN})
    assert list(vocab) == expected
    assert [vocab.token_to_id(t) for t in expected] == list(range(len(expected)))
    assert vocab.next_id == len(expected)


def test_add_assigns_sequential_ids_and_is_idempotent():
    vocab = Vocabulary()
    assert vocab.add("a", 5) == 0
    assert vocab.add("b") == 1
    assert vocab.add("a", 99) == 0
    assert vocab.frequencies["a"] == 5
    assert len(vocab) == 2


def test_unknown_tokens_map_to_unk():
    vocab = Vocabulary.base()
    assert vocab.token_to_id("not-a-token") == vocab.unk_id
    assert vocab.id_to_token(10**9) == UNK_TOKEN
    assert vocab.tokens_to_ids(["ለ", "zzz"]) == [vocab.token_to_id("ለ"), vocab.unk_id]
    assert vocab.ids_to_tokens([vocab.token_to_id("ለ"), -5]) == ["ለ", UNK_TOKEN]


def test_unk_id_without_unk_token():
    assert Vocabulary().unk_id == -1


def test_from_mappings_roundtrip():
    vocab = Vocabulary.base()
    vocab.add("ለአ", 3)
    copy = Vocabulary.from_mappings(
        vocab.frequencies, vocab.token_to_id_map, vocab.id_to_token_map, vocab.next_id
    )
    assert list(copy) == list(vocab)
    assert copy.token_to_id("ለአ") == vocab.token_to_id("ለአ")
    assert copy.add("new") == vocab.next_id


def test_contains():
    vocab = Vocabulary.base()
    assert EOW_TOKEN in vocab
    assert "missing" not in vocab
