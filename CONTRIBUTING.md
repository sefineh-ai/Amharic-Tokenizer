# Contributing

Thanks for considering a contribution!

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # also compiles the Cython extension
```

Rebuild after changing `src/amharic_tokenizer/_bpe.pyx`: `pip install -e .`

## Before opening a pull request

```bash
ruff check . && ruff format --check .
mypy
pytest
```

or simply `tox`.

## Guidelines

- Keep the dependency direction described in [docs/architecture.md](docs/architecture.md).
  Only performance-critical loops belong in Cython.
- Every behavior change needs a test: unit tests in `tests/unit`, end-to-end
  tests in `tests/integration`.
- `tests/fixtures/golden.json` pins tokenization and training outputs. If you
  change them on purpose, regenerate the fixture in the same PR and explain why
  in `CHANGELOG.md`. Changing these outputs requires a new model version.
- New pretrained models go in `src/amharic_tokenizer/models/` as `amh_bpe_vX.Y.Z.json`.
  Update `DEFAULT_MODEL` in `tokenizer.py` when a new model becomes the default.
