"""End-to-end checks against the bundled pretrained models."""

import pytest

from amharic_tokenizer import (
    AMHARIC_FIDEL_MAP,
    DEFAULT_MODEL,
    AmharicTokenizer,
    list_pretrained_models,
)

ROUNDTRIP_TEXT = (
    "የኮሪደር ልማት ገፀ በረከት የሆናቸው የከተማችን ሰፈሮች በነዋሪዎች አንደበት በሰዓት 209 ኪሎ ሜትር "
    "የሚጓዘው አውሎ ንፋስ ከጃማይካ ቀጥሎ ኩባ ደርሷል ጠቅላይ"
)


def test_default_model_is_bundled():
    assert DEFAULT_MODEL in list_pretrained_models()


def test_default_model_shape(pretrained):
    assert pretrained.vocab_size == 30000
    assert pretrained.is_trained()


def test_roundtrip_basic(pretrained):
    tokens = pretrained.tokenize(ROUNDTRIP_TEXT)
    assert pretrained.detokenize(tokens) == ROUNDTRIP_TEXT


def test_matches_golden_outputs(pretrained, golden):
    for case in golden["pretrained"]:
        text = case["text"]
        assert pretrained.tokenize(text) == case["tokens"], text
        assert pretrained.encode(text) == case["ids"], text
        assert pretrained.detokenize(case["tokens"]) == case["detokenized"], text
        assert pretrained.decode(case["ids"]) == case["decoded"], text


def test_encode_decode_roundtrip_on_corpus(pretrained, sample_corpus):
    # Only words made entirely of mapped fidel are guaranteed to be in the vocabulary.
    words = [w for w in sample_corpus.split() if all(ch in AMHARIC_FIDEL_MAP for ch in w)]
    text = " ".join(words)
    assert len(words) > 1000
    assert pretrained.decode(pretrained.encode(text)) == text


@pytest.mark.parametrize("name", list_pretrained_models())
def test_every_bundled_model_loads_and_roundtrips(name):
    tok = AmharicTokenizer.load(name)
    assert tok.vocab_size > 45
    assert tok.detokenize(tok.tokenize("ኢትዮጵያ ጥሩ ናት")) == "ኢትዮጵያ ጥሩ ናት"


def test_load_accepts_json_suffix():
    assert AmharicTokenizer.load(DEFAULT_MODEL + ".json").vocab_size == 30000
