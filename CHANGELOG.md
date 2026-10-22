# Changelog

## Unreleased

### Fixed
- Ethiopic digits (`፩`-`፻`) and punctuation (`፡`, `።`, `፣`, `፤`, `፥`, `፦`, `፧`, `፨`) had no base
  vocabulary slot: an occurrence that never got merged into a larger learned token encoded to
  `<unk>` and silently disappeared on `decode`/`detokenize`. They are now part of
  `Vocabulary.base()`, and the bundled pretrained models have been patched (new ids appended,
  nothing renumbered) so existing model files benefit without retraining.

### Changed
- `normalize_for_training` dropped a redundant regex pass over the whole corpus (Latin/digit
  stripping was already a subset of the Ethiopic-only filter that ran right after it).

## 0.2.7

### Added
- Pretrained model `amh_bpe_v0.2.7` (now the default for `AmharicTokenizer.load()`).
- CLI commands `clean`, `tokenize`, `encode`, `decode`, `models`, plus `--version` and `--max-vocab-size`.
- `python -m amharic_tokenizer`.
- API: `vocab_size`, `merges`, `is_trained()`, `convert_tokens_to_ids`, `convert_ids_to_tokens`,
  `token_to_id`, `id_to_token`, `train_from_file`, `list_pretrained_models`, `decompose`, `compose`.
- Typed exceptions and a `py.typed` marker.
- Unit and integration test suites, CI (lint, mypy, tests on 3.9–3.13 and three OSes, wheel build).

### Changed
- Training ~1,000× faster (heap + pair→word index); tokenization is linear in text length.
- Fidel map: `ጓ` fixed (was `ጏ`); added `ቨ`, `ዐ`, `ቐ`, `ዸ` series and labialized forms.
- `src/` layout; the Cython extension is now `amharic_tokenizer._bpe` and only contains the hot loops.
- `load()` accepts filesystem paths (before, only bundled models could be loaded).
- `train(verbose=True)` writes progress to stderr; `save()`/`load()` log instead of printing.
- `amharic_tokenizer.fidel_map` → `amharic_tokenizer.fidel`; `amharic_tokenizer.pipeline` → `amharic_tokenizer.training`.
- Python ≥ 3.9.

### Removed
- `setup.cfg` and `tools/configure_pypirc.py` (releases now use PyPI trusted publishing via `.github/workflows/publish.yml`).
- Private `_vocabulary`, `_merge_rank_map`, `_token_to_id`, … attributes and the `preprocess` / `_clean_corpus` methods.
