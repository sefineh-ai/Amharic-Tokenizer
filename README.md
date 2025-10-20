# Amharic Tokenizer

**Amharic tokenizer with a GPT-style BPE-like pipeline over decomposed fidel.**
Implements: **cleaning → fidel decomposition → BPE training/application → detokenization**, with a **Cython core for speed**.

---

## What's new in 0.1.2

- WordPiece-style continuation prefixes: non-initial subwords are now prefixed with `##` during tokenization.
  - Example: `Going` → `['G', '##o', '##i', '##n', '##g', '</w>']`
  - Amharic example:
    Input: `የተባለ ውን የሚያደርገው ም በዚህ ምክንያት ነው`
    Tokens:
    ```
    ['የአተአ', '##በ', '##ኣለ', '##አ', '</w>', ' ', 'ወእ', '##ነ', '##እ', '</w>', ' ', 'የአመኢየኣ', '##ደ', '##አረ', '##እ', '##ገ', '##አወእ', '</w>', ' ', 'መእ', '</w>', ' ', 'በአ', '##ዘኢ', '##ሀ', '##እ', '</w>', ' ', 'መእ', '##ከ', '##እነእ', '##የኣ', '##ተእ', '</w>', ' ', 'ነ', '##አወእ', '</w>']
    ```
    Detokenization matches the input.
- Detokenization fixes:
  - Strips `##` correctly and handles embedded `</w>` markers without leaking into text.
  - Avoids extra spaces resulting from end-of-word handling.
- Developer ergonomics: `AmharicTokenizer.from_default()` returns a minimally trained instance for quick experiments.

> Note: The `</w>` token remains an internal end-of-word marker in the token stream; it is never emitted in detokenized text.

---

## Installation

### From PyPI (recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

pip install amharic-tokenizer
```

Verify the CLI:

```bash
amh-tokenizer --help
```

### From source (for development)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## Training (CLI)

```bash
# Train on a cleaned Amharic text corpus and save model
amh-tokenizer train /abs/path/to/cleaned_amharic.txt /abs/path/to/amh_bpe \
  --num-merges 50000 --verbose --log-every 2000

# Example using relative paths
amh-tokenizer train cleaned_amharic.txt amh_bpe --num-merges 50000 --verbose --log-every 2000
```

---

## Quick Usage (Python)

```python
from amharic_tokenizer import AmharicTokenizer

# Load a trained model
tok = AmharicTokenizer.load("amh_bpe")  # use full path if needed

text = "ኢትዮጵያ ጥሩ ናት።"

# Tokenize
tokens = tok.tokenize(text)
print(tokens)  # variable-length subword tokens

# Optional: remove end-of-word markers for display
display_tokens = [t.replace('</w>', '') for t in tokens if t != '</w>']
print(display_tokens)

# Detokenize back to original text
print(tok.detokenize(tokens))
```

---

## Example Script

```bash
# Test a single string
python examples/try_tokenizer.py amh_bpe --text "ኢትዮጵያ ጥሩ ናት።"

# Test a file
python examples/try_tokenizer.py amh_bpe --file cleaned_amharic.txt
```

> Tip: If running examples directly by path, ensure the package is installed (`pip install -e .`)
> or run as a module from the project root:

```bash
python -m examples.try_tokenizer amh_bpe --text "..."
```

---

## API

```python
AmharicTokenizer(num_merges=50000)
```

* `train(corpus_text, verbose=False, log_every=1000) -> int`
* `tokenize(text) -> list[str]`
* `detokenize(tokens) -> str`
* `save(path_prefix)` / `load(path_prefix)`
* `is_trained() -> bool`

---

## Notes

* Longer, more diverse corpora and higher `num_merges` produce longer subwords.
* Training and tokenization work over **decomposed fidel**; detokenization recomposes the original Amharic characters.

---

## Troubleshooting

* **ModuleNotFoundError inside the repo:** install in editable mode (`pip install -e .`)
  or run scripts from outside the repo to avoid shadowing the installed package.
* **TestPyPI installs:** resolve build dependencies from PyPI:

```bash
pip install -i https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple amharic-tokenizer
```

---

## License

This project is licensed under the MIT License – see the [LICENSE](https://github.com/sefineh-ai/AMH-Tokenizer/blob/main/LICENSE) file for details.
