"""Tests for the Cython kernels in amharic_tokenizer._bpe."""

from collections import Counter

from amharic_tokenizer._bpe import BPEEncoder, learn_merges
from amharic_tokenizer.vocabulary import EOW_TOKEN


def _learn(words, num_merges=100, max_vocab_size=10**6, known=()):
    return learn_merges(Counter(words), known, num_merges, max_vocab_size)


def test_learns_most_frequent_pair_first():
    # ሉ = ለ+ኡ occurs 3 times, ሳ = ሰ+ኣ once.
    merges = _learn(["ሳ", "ሉ", "ሉ", "ሉ"], num_merges=1)
    assert merges == [("ለኡ", 3)]


def test_merges_include_end_of_word_marker():
    merges = _learn(["ለ"] * 3, num_merges=2)
    assert [token for token, _ in merges] == ["ለአ", "ለአ" + EOW_TOKEN]


def test_stops_when_no_pair_occurs_twice():
    assert _learn(["ሉ", "ሳ"]) == []


def test_empty_input():
    assert _learn([]) == []


def test_respects_num_merges():
    assert len(_learn(["ለሰመ"] * 5, num_merges=2)) == 2


def test_respects_max_vocab_size():
    known = ["x", "y", "z"]
    assert len(_learn(["ለሰመ"] * 5, max_vocab_size=5, known=known)) == 2


def test_ties_go_to_first_seen_pair():
    # Every pair occurs twice; ሳ is seen first, so ሰ+ኣ is merged first.
    merges = _learn(["ሳ", "ሉ", "ሳ", "ሉ"], num_merges=1)
    assert merges == [("ሰኣ", 2)]


def test_known_tokens_are_not_relearned():
    merges = _learn(["ለ"] * 3, num_merges=2, known=["ለአ"])
    # ለአ is applied (so ለአ<eow> can be learned) but not reported as new.
    assert [token for token, _ in merges] == ["ለአ" + EOW_TOKEN]


def test_progress_callback_receives_messages():
    messages = []
    learn_merges(Counter(["ለ"] * 3), (), 2, 10**6, messages.append, 1)
    assert messages[0].startswith("Counted 3 words")
    assert messages[-1].startswith("Training completed: 2 merges")


def test_encoder_without_merges_returns_symbols():
    assert BPEEncoder({}).tokenize("ሉ ሀ") == ["ለ", "ኡ", EOW_TOKEN, "ሀ", "አ", EOW_TOKEN]


def test_encoder_applies_merges_by_rank():
    ranks = {"ለኡ": 1, "ለኡ" + EOW_TOKEN: 2}
    assert BPEEncoder(ranks).tokenize("ሉ ሉ") == ["ለኡ" + EOW_TOKEN] * 2


def test_encoder_lower_rank_wins():
    # ለአ (rank 1) must be applied before አለ (rank 2) in ለአለአ.
    enc = BPEEncoder({"ለአ": 1, "አለ": 2})
    assert enc.tokenize("ለለ") == ["ለአ", "ለአ", EOW_TOKEN]


def test_encoder_whitespace_only():
    assert BPEEncoder({"ለአ": 1}).tokenize(" \n\t ") == []


def test_encoder_cached_words_give_same_tokens():
    enc = BPEEncoder({"ለአ": 1})
    first = enc.tokenize("ለ ለ ሰ")
    assert enc.tokenize("ለ ለ ሰ") == first


def test_encoder_matches_training_segmentation():
    words = ["ሰላም", "ሰላም", "ሰላማዊ", "ሰላምታ", "ሰላም"]
    merges = _learn(words, num_merges=20)
    enc = BPEEncoder({token: rank for rank, (token, _) in enumerate(merges, 1)})
    assert enc.tokenize("ሰላም") == ["ሰአለኣመእ" + EOW_TOKEN]
