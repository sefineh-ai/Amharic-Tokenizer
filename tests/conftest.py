"""Shared fixtures.

``fixtures/golden.json`` holds outputs recorded from the pre-refactor v0.2.7
implementation (bundled-model tokenization and two training runs on
``fixtures/sample_corpus.txt``); tests compare against it to guarantee the
tokenizer's behavior does not drift.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from amharic_tokenizer import AmharicTokenizer

FIXTURES = Path(__file__).parent / "fixtures"


def pytest_collection_modifyitems(items: list) -> None:
    for item in items:
        if "integration" in Path(str(item.fspath)).parts:
            item.add_marker(pytest.mark.integration)


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def sample_corpus() -> str:
    return (FIXTURES / "sample_corpus.txt").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def golden() -> Dict[str, Any]:
    return json.loads((FIXTURES / "golden.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def trained_tokenizer(sample_corpus: str) -> AmharicTokenizer:
    """A small model trained on the sample corpus. Treat as read-only."""
    tok = AmharicTokenizer(num_merges=300)
    tok.train(sample_corpus)
    return tok


@pytest.fixture(scope="session")
def pretrained() -> AmharicTokenizer:
    """The default bundled model. Treat as read-only."""
    return AmharicTokenizer.load()
