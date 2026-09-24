"""The ``amh-tokenizer`` CLI, in-process and as a real subprocess."""

import json
import subprocess
import sys

import pytest

from amharic_tokenizer import DEFAULT_MODEL, __version__
from amharic_tokenizer.cli import main


def run(capsys, *argv):
    code = main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


def test_train_then_use_model(tmp_path, fixtures_dir, capsys):
    model = tmp_path / "model"
    code, out, _ = run(
        capsys, "train", str(fixtures_dir / "sample_corpus.txt"), str(model), "--num-merges", "50"
    )
    assert code == 0
    assert "with 50 merges" in out

    code, out, _ = run(capsys, "encode", "ሰላም", "--model", str(model))
    ids = json.loads(out)
    code, out, _ = run(capsys, "decode", *map(str, ids), "--model", str(model))
    assert (code, out.strip()) == (0, "ሰላም")


def test_train_legacy_verbose_flag_prints_progress(tmp_path, fixtures_dir, capsys):
    code, _, err = run(
        capsys,
        "train",
        str(fixtures_dir / "sample_corpus.txt"),
        str(tmp_path / "m"),
        "--num-merges",
        "5",
        "--verbose",
        "--log-every",
        "1",
    )
    assert code == 0
    assert "Merge 1/5" in err


def test_global_verbose_flag_is_not_overridden_by_subcommand(tmp_path, fixtures_dir, capsys):
    code, _, err = run(
        capsys,
        "-v",
        "train",
        str(fixtures_dir / "sample_corpus.txt"),
        str(tmp_path / "m"),
        "--num-merges",
        "3",
    )
    assert code == 0
    assert "Training completed" in err


def test_tokenize_default_model(capsys, pretrained):
    code, out, _ = run(capsys, "tokenize", "ኢትዮጵያ ጥሩ ናት")
    assert code == 0
    assert json.loads(out) == pretrained.tokenize("ኢትዮጵያ ጥሩ ናት")


def test_tokenize_reads_stdin(capsys, monkeypatch, pretrained):
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("ሰላም"))
    code, out, _ = run(capsys, "tokenize")
    assert code == 0
    assert json.loads(out) == pretrained.tokenize("ሰላም")


def test_clean(tmp_path, capsys):
    raw = tmp_path / "raw.txt"
    raw.write_text("ይህ የመጀመሪያው ዓረፍተ ነገር ነው። abc\n", encoding="utf-8")
    code, out, _ = run(capsys, "clean", str(raw), str(tmp_path / "clean.txt"), "--min-length", "5")
    assert code == 0
    assert "Wrote 1 sentences" in out


def test_models_lists_default(capsys):
    code, out, _ = run(capsys, "models")
    assert code == 0
    assert DEFAULT_MODEL in out.split()


def test_missing_model_is_a_clean_error(capsys, tmp_path):
    code, out, err = run(capsys, "tokenize", "ሰላም", "--model", str(tmp_path / "nope"))
    assert code == 1
    assert out == ""
    assert err.startswith("amh-tokenizer: error: model")


def test_missing_command_exits_with_usage(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


def test_module_entry_point_subprocess():
    proc = subprocess.run(
        [sys.executable, "-m", "amharic_tokenizer", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == f"amh-tokenizer {__version__}"
