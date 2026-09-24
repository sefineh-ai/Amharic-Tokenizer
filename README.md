# Amharic Tokenizer  🇪🇹
[![PyPI Version](https://img.shields.io/pypi/v/amharic-tokenizer.svg?logo=pypi&logoColor=white)](https://pypi.org/project/amharic-tokenizer/)
[![Python Versions](https://img.shields.io/pypi/pyversions/amharic-tokenizer.svg?logo=python&logoColor=white)](https://pypi.org/project/amharic-tokenizer/)
[![Cython](https://img.shields.io/badge/Cython-Optimized-brightgreen?logo=cython&logoColor=white)](https://cython.org/)
[![Build Status](https://github.com/sefineh-ai/Amharic-Tokenizer/actions/workflows/ci.yml/badge.svg)](https://github.com/sefineh-ai/Amharic-Tokenizer/actions)
[![License](https://img.shields.io/github/license/sefineh-ai/Amharic-Tokenizer.svg?color=yellow)](https://github.com/sefineh-ai/Amharic-Tokenizer/blob/main/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/amharic-tokenizer?label=Downloads&color=orange)](https://pypi.org/project/amharic-tokenizer/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)

[![GitHub Sponsors](https://img.shields.io/badge/Sponsor-GitHub-%23EA4AAA?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/sefineh-ai)
[![Patreon](https://img.shields.io/badge/Support-Patreon-orange.svg?logo=patreon&logoColor=white)](https://patreon.com/sefineh)
[![Open Collective](https://img.shields.io/badge/Backers-Open%20Collective-blue.svg?logo=opencollective&logoColor=white)](https://opencollective.com/sefineh-ai)


**Amharic tokenizer with a GPT-style BPE-like pipeline over decomposed fidel.**
Implements: **cleaning → fidel decomposition → BPE training/application → detokenization**, with a **Cython core for speed**.

---

## What's new in v0.2.7

**Training on a 16-million-word corpus: 12 hours → 42 seconds (~1,000× faster).**
Tokenization is also linear in text length instead of quadratic.

| Version | Training time (16M words) | Speedup |
|---------|--------------------------:|--------:|
| v0.2.6  | 12 hours                  | 1×      |
| v0.2.7  | 42 seconds                | ~1,000× |

Same corpus, same 30,000-token vocabulary, same machine.

### 1. Much faster training
- `train()` previously went through every word in the corpus and copied all pair counts for every merge.
- Each distinct word is now stored once with its frequency, an index maps each pair to the words that contain it, and a heap tracks the most frequent pair. Each merge only updates the words that contain the merged pair.
- The learned merges, counts and IDs match the v0.2.6 algorithm on the benchmark corpora. Ties still go to the pair seen first in the corpus.
- With `--verbose`, progress is printed one line at a time with elapsed time.

### 2. Much faster tokenization (linear instead of quadratic)
- `tokenize()` and `encode()` previously applied merges to the whole text at once: every merge step scanned and rebuilt every word to find a single best pair, so time grew quadratically with text length.
- Merges never cross word boundaries (each word ends with `<eow>`), so each word is now tokenized independently, and results are cached per word so repeated words are not tokenized again.
- Merge lookups use integer symbol ids in a C++ hash table instead of building a new string for every pair, so the inner loop runs without Python objects.
- `encode()` now also accepts the token list returned by `tokenize()` (`tok.encode(tokens)`), so tokenizing and then encoding no longer tokenizes twice.
- For the same model, tokens and IDs are identical to v0.2.6.

| Words | Before | After | Speedup |
|------:|-------:|------:|--------:|
| 2,000 | 1.23 s | 0.002 s | ~500x |
| 4,000 | 3.89 s | 0.005 s | ~800x |
| 8,000 | 11.08 s | 0.007 s | ~1,700x |
| 400,000 | not practical | 0.18 s | |

Measured with `amh_bpe_v0.2.6` on text from `data/cleaned_data.txt`, starting from an empty word cache each time. The first `tokenize()` call after `load()` or `train()` also builds the merge lookup table once (about 0.05 s).

### 3. Updated fidel map and new pretrained model
- v0.2.7 ships an updated fidel map and a new pretrained model, `amh_bpe_v0.2.7`. The v0.2.6 model is still included, so existing code keeps working:

  ```python
  from amharic_tokenizer import AmharicTokenizer

  tok = AmharicTokenizer.load("amh_bpe_v0.2.7")  # new model
  old = AmharicTokenizer.load("amh_bpe_v0.2.6")  # previous model
  ```

- **Upgrading:** token IDs from `amh_bpe_v0.2.6` and `amh_bpe_v0.2.7` are not interchangeable. If you stored IDs, keep decoding them with the model that produced them.

### 4. Project restructure
- `src/` layout with a small Cython core (`_bpe.pyx`) behind a pure-Python, type-checked API. See [docs/architecture.md](docs/architecture.md).
- `AmharicTokenizer.load()` now also loads models from any file path (previously only bundled models could be loaded), and defaults to the latest bundled model.
- New CLI commands: `clean`, `tokenize`, `encode`, `decode`, `models`.
- Typed exceptions (`ModelNotFoundError`, `InvalidModelError`, `TrainingError`), logging instead of `print`, `py.typed` marker.
- Unit and integration test suites (pinned against recorded v0.2.7 outputs), CI on Linux/macOS/Windows and Python 3.9–3.13.

Speed on your own machine:

```bash
python benchmarks/benchmark.py                                    # small bundled sample
python benchmarks/benchmark.py data/cleaned_data.txt --num-merges 30000
```

---

## What's new in v0.2.6
- 30,000-token vocabulary trained on a larger, more diverse Amharic corpus.
- Pretrained models load by name: `AmharicTokenizer.load("amh_bpe_v0.2.6")`.
- Full token ↔ id round trip (`tokenize`, `encode`, `decode`, `detokenize`).

---

## Installation

```bash
pip install amharic-tokenizer
amh-tokenizer --help
```

From source (compiles the Cython extension):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

---

## Quick start (Python)

```python
from amharic_tokenizer import AmharicTokenizer

tok = AmharicTokenizer.load()  # latest bundled model (amh_bpe_v0.2.7)

text = "ኢትዮጵያ ጥሩ ናት"
tokens = tok.tokenize(text)  # subword tokens; each word ends with <eow>
ids = tok.encode(text)  # or tok.encode(tokens) to skip re-tokenizing
assert tok.decode(ids) == text
assert tok.detokenize(tokens) == text
```

Characters outside the vocabulary (Latin letters, digits, most punctuation) are
encoded as `<unk>` and dropped by `decode()`.

## Training

```bash
# Optional: turn raw scraped text into unique, cleaned sentences
amh-tokenizer clean raw_amharic.txt cleaned_amharic.txt

# Train and save (.json is appended when missing)
amh-tokenizer train cleaned_amharic.txt models/amh_bpe \
  --num-merges 50000 --max-vocab-size 30000 --verbose --log-every 2000
```

```python
from amharic_tokenizer import AmharicTokenizer, train_from_file

result = train_from_file("cleaned_amharic.txt", "models/amh_bpe", num_merges=50000)
tok = AmharicTokenizer.load(result.model_path)

# or, in memory
tok = AmharicTokenizer(num_merges=2000, max_vocab_size=5000)
tok.train(corpus_text, verbose=True, log_every=100)
tok.save("amh_bpe_model")
```

## CLI

| Command | Description |
|---------|-------------|
| `amh-tokenizer train CORPUS OUTPUT [--num-merges N] [--max-vocab-size N] [--log-every N]` | Train and save a model |
| `amh-tokenizer clean INPUT OUTPUT [--min-length N]` | Extract unique Amharic sentences from raw text |
| `amh-tokenizer tokenize [TEXT] [-m MODEL]` | Print tokens as JSON (reads stdin without `TEXT`) |
| `amh-tokenizer encode [TEXT] [-m MODEL]` | Print ids as JSON |
| `amh-tokenizer decode ID... [-m MODEL]` | Print decoded text |
| `amh-tokenizer models` | List bundled pretrained models |

`-m/--model` accepts a file path or a bundled model name. Add `-v` for debug logs.

---

## API

```python
AmharicTokenizer(num_merges=50000, max_vocab_size=30000)
```

| Member | Description |
|--------|-------------|
| `train(corpus, verbose=False, log_every=1000, progress=None) -> int` | Learn merges; returns the total merge count |
| `tokenize(text) -> list[str]` | Subword tokens |
| `encode(text_or_tokens) -> list[int]` | Token ids |
| `decode(ids) -> str` / `detokenize(tokens) -> str` | Back to text |
| `convert_tokens_to_ids(tokens)` / `convert_ids_to_tokens(ids)` | Vocabulary lookups |
| `token_to_id(token)` / `id_to_token(id)` | Single lookups |
| `vocab_size`, `merges`, `num_merges`, `max_vocab_size` | Properties |
| `is_trained() -> bool` | Whether any merges are learned |
| `save(path) -> Path` / `AmharicTokenizer.load(name_or_path=DEFAULT_MODEL)` | Persistence |

Other exports: `train_from_file`, `list_pretrained_models`, `decompose`, `compose`,
`AMHARIC_FIDEL_MAP`, `REVERSE_FIDEL_MAP`, and the exceptions `AmharicTokenizerError`,
`ModelNotFoundError`, `InvalidModelError` and `TrainingError`.

**Upgrading from ≤ 0.2.6:** `amharic_tokenizer.fidel_map` is now `amharic_tokenizer.fidel`,
`amharic_tokenizer.pipeline` is now `amharic_tokenizer.training` (`train_and_save` is kept),
and the private `_vocabulary`, `_merge_rank_map`, … attributes are replaced by the
properties above. Load and save no longer print; use `logging` to see those messages.

---

## Project layout

```
src/amharic_tokenizer/   the package (see docs/architecture.md)
tests/unit/              fast, isolated module tests
tests/integration/       bundled models, training pipeline, CLI
tests/fixtures/          sample corpus + golden outputs
benchmarks/              speed measurements
tools/                   corpus crawler (not installed)
docs/                    architecture notes
```

## Development

```bash
pip install -e ".[dev]"
pytest                          # all tests
pytest tests/unit               # unit only
pytest -m integration           # integration only
ruff check . && ruff format --check .
mypy
tox                             # everything, on every installed Python
```

After editing `_bpe.pyx`, rebuild with `pip install -e .` (or `python setup.py build_ext --inplace`).
See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Notes

* Longer, more diverse corpora and higher `num_merges` produce longer subwords.
* Training and tokenization work over **decomposed fidel**; detokenization recomposes the original Amharic characters.
* Tokenized words are cached per tokenizer (up to 200,000 distinct words); the cache is reset by `train()` and `load()`.
* Token ids are model-specific: decode ids with the model that produced them.

## Troubleshooting

* **ModuleNotFoundError inside the repo:** install in editable mode (`pip install -e .`).
* **TestPyPI installs:** resolve build dependencies from PyPI:

```bash
pip install -i https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple amharic-tokenizer
```

---

## License

This project is licensed under the MIT License – see the [LICENSE](https://github.com/sefineh-ai/Amharic-Tokenizer/blob/main/LICENSE) file for details.
