import pytest

from amharic_tokenizer.fidel import (
    AMHARIC_FIDEL_MAP,
    MAX_DECOMPOSITION_LENGTH,
    REVERSE_FIDEL_MAP,
    base_symbols,
    compose,
    decompose,
)


def test_map_is_a_bijection():
    assert len(REVERSE_FIDEL_MAP) == len(AMHARIC_FIDEL_MAP)
    for fidel, decomposition in AMHARIC_FIDEL_MAP.items():
        assert REVERSE_FIDEL_MAP[decomposition] == fidel


def test_every_fidel_is_ethiopic():
    for fidel, decomposition in AMHARIC_FIDEL_MAP.items():
        assert len(fidel) == 1
        assert all("ሀ" <= ch <= "፿" for ch in fidel + decomposition)


def test_max_decomposition_length():
    assert MAX_DECOMPOSITION_LENGTH == 3


@pytest.mark.parametrize(
    ("fidel", "expected"),
    [("ሉ", "ለኡ"), ("ቋ", "ቀኡኣ"), ("ጓ", "ገኡኣ"), ("ጐ", "ገኡአ"), ("ኧ", "ኧ")],
)
def test_decompose_single_fidel(fidel, expected):
    assert decompose(fidel) == expected


def test_decompose_keeps_unknown_characters():
    assert decompose("A1 ።") == "A1 ።"


@pytest.mark.parametrize(
    "text",
    ["ኢትዮጵያ ጥሩ ናት", "ቋንቋ ጓደኛ ኳስ ጐበዝ", "ዓለም ቨርጂኒያ", "Hello ሰላም 123", ""],
)
def test_compose_inverts_decompose(text):
    assert compose(decompose(text)) == text


def test_compose_every_fidel():
    for fidel, decomposition in AMHARIC_FIDEL_MAP.items():
        assert compose(decomposition) == fidel


def test_compose_is_greedy_longest_match():
    # ለ+ኡ+ኣ is the decomposition of ሏ, which wins over ሉ followed by ኣ.
    assert compose("ለኡኣ") == "ሏ"


def test_base_symbols_sorted_and_unique():
    symbols = base_symbols()
    assert symbols == sorted(set(symbols))
    assert "አ" in symbols and "ለ" in symbols
