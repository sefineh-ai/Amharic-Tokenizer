from amharic_tokenizer.normalization import (
    clean_corpus_file,
    iter_clean_sentences,
    normalize_for_training,
)


def test_normalize_drops_latin_digits_and_punctuation():
    assert normalize_for_training("ሰላም Hello 123, ዓለም!") == "ሰላም ዓለም"


def test_normalize_keeps_ethiopic_punctuation():
    assert normalize_for_training("ሰላም። ዓለም") == "ሰላም። ዓለም"


def test_normalize_collapses_whitespace():
    assert normalize_for_training("  ሰላም\n\t\tዓለም  ") == "ሰላም ዓለም"


def test_normalize_empty():
    assert normalize_for_training("abc 123") == ""


def test_iter_clean_sentences_splits_and_deduplicates():
    lines = ["ይህ የመጀመሪያው ዓረፍተ ነገር ነው። ይህ ሁለተኛው ዓረፍተ ነገር ነው።", "ይህ የመጀመሪያው ዓረፍተ ነገር ነው።"]
    assert list(iter_clean_sentences(lines, min_length=5)) == [
        "ይህ የመጀመሪያው ዓረፍተ ነገር ነው",
        "ይህ ሁለተኛው ዓረፍተ ነገር ነው",
    ]


def test_iter_clean_sentences_filters_short_and_non_amharic():
    lines = ["አጭር።", "only english text here", "", "   "]
    assert list(iter_clean_sentences(lines, min_length=10)) == []


def test_iter_clean_sentences_strips_latin():
    assert list(iter_clean_sentences(["ሰላም Hello ዓለም እንዴት ነህ"], min_length=5)) == [
        "ሰላም ዓለም እንዴት ነህ"
    ]


def test_clean_corpus_file(tmp_path):
    src = tmp_path / "raw.txt"
    dst = tmp_path / "clean.txt"
    src.write_text("ይህ የመጀመሪያው ዓረፍተ ነገር ነው።\nabc\n", encoding="utf-8")
    assert clean_corpus_file(src, dst, min_length=5) == 1
    assert dst.read_text(encoding="utf-8") == "ይህ የመጀመሪያው ዓረፍተ ነገር ነው\n"
