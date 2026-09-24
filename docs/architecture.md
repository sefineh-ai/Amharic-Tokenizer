# Architecture

## Pipeline

```
raw text ──► normalization ──► fidel decomposition ──► BPE (train / apply) ──► ids
                                                                                │
text ◄──────────── fidel composition ◄──────────── tokens ◄─────────────────────┘
```

1. **Normalization** (`normalization.py`): for training, keep only Ethiopic
   characters. `clean_corpus_file` / `amh-tokenizer clean` turn scraped text into
   unique sentences.
2. **Decomposition** (`fidel.py`): each fidel becomes consonant + vowel symbols,
   e.g. `ሉ → ለ ኡ`. Words get a trailing `<eow>` symbol.
3. **BPE** (`_bpe.pyx`): `learn_merges` learns merges ranked by frequency;
   `BPEEncoder` applies them to each word, lowest rank first.
4. **Vocabulary** (`vocabulary.py`): maps tokens to stable integer ids.
   Unknown tokens map to `<unk>`.
5. **Composition** (`fidel.compose`): `detokenize` joins tokens, turns `<eow>`
   into spaces and recomposes the fidel with a greedy longest match.

## Package layout

```
src/amharic_tokenizer/
├── __init__.py        public API re-exports, __version__
├── tokenizer.py       AmharicTokenizer facade (pure Python)
├── _bpe.pyx / .pyi    Cython kernels: learn_merges, BPEEncoder
├── vocabulary.py      Vocabulary, special tokens
├── fidel.py           fidel map, decompose / compose
├── normalization.py   corpus cleaning
├── serialization.py   ModelState, JSON read/write, bundled model lookup
├── training.py        train_from_file pipeline
├── cli.py             amh-tokenizer command
├── exceptions.py      error hierarchy
└── models/            bundled pretrained models (amh_bpe_v*.json)
```

### Dependency direction

```
cli ──► training ──► tokenizer ──► _bpe ──► fidel
                         │
                         ├──► vocabulary ──► fidel
                         ├──► serialization ──► exceptions
                         └──► normalization
```

Modules only import "downward". `fidel`, `normalization` and `exceptions` have
no internal dependencies. The Cython module holds only the two hot loops, so
the rest of the codebase is plain, type-checked Python.

## Design decisions

- **Why decompose fidel?** Amharic fidel encode consonant and vowel together.
  Splitting them lets BPE share subwords across vowel variants of a stem.
- **Why do merges stay inside a word?** Each word ends with `<eow>`, so a merge
  can't cross words. That lets `BPEEncoder` tokenize and cache each word
  on its own. Tokenization time grows linearly with text length.
- **Training complexity.** Each distinct word is stored once with its count.
  An index maps each pair to the words that contain it, and a heap with lazy
  deletion yields the best pair. So each merge only touches the affected words.
  Ties go to the pair that appeared first in the corpus, so training is fully
  deterministic.
- **Model format.** A JSON file (`format_version: 1`) holding the vocabulary,
  frequencies, merge ranks and both id maps. Models without `format_version`
  (≤ v0.2.7) load as version 1. Newer versions are rejected with a clear error.
- **Model lookup.** `AmharicTokenizer.load(x)` checks the filesystem path `x`
  first (`.json` optional), then the bundled model named `x`.

## Tests

| Suite | Scope |
|-------|-------|
| `tests/unit` | Each module on its own, with small inputs: fidel map invariants, normalization, vocabulary, Cython kernels, tokenizer API, serialization and validation. |
| `tests/integration` | Bundled models, train → save → load on a real corpus sample, the CLI in-process and as a subprocess. |

`tests/fixtures/golden.json` stores outputs recorded from the v0.2.7
implementation before the refactor: bundled-model tokens and ids, plus the full
merge tables from two training runs. The integration suite checks against it,
so any change in tokenization or training behavior fails the tests.
