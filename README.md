# Amharic Tokenizer  🇪🇹
[![PyPI Version](https://img.shields.io/pypi/v/amharic-tokenizer.svg?logo=pypi&logoColor=white)](https://pypi.org/project/amharic-tokenizer/)
[![Python Versions](https://img.shields.io/pypi/pyversions/amharic-tokenizer.svg?logo=python&logoColor=white)](https://pypi.org/project/amharic-tokenizer/)
[![Cython](https://img.shields.io/badge/Cython-Optimized-brightgreen?logo=cython&logoColor=white)](https://cython.org/)
[![Build Status](https://github.com/sefineh-ai/AMH-Tokenizer/actions/workflows/pylint.yml/badge.svg)](https://github.com/sefineh-ai/AMH-Tokenizer/actions)
[![License](https://img.shields.io/github/license/sefineh-ai/AMH-Tokenizer.svg?color=yellow)](https://github.com/sefineh-ai/AMH-Tokenizer/blob/main/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/amharic-tokenizer?label=Downloads&color=orange)](https://pypi.org/project/amharic-tokenizer/)
[![Code Style: PEP8](https://img.shields.io/badge/code%20style-pep8-lightgrey.svg)](https://www.python.org/dev/peps/pep-0008/)

[![GitHub Sponsors](https://img.shields.io/badge/Sponsor-GitHub-%23EA4AAA?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/sefineh-ai)
[![Patreon](https://img.shields.io/badge/Support-Patreon-orange.svg?logo=patreon&logoColor=white)](https://patreon.com/sefineh)
[![Open Collective](https://img.shields.io/badge/Backers-Open%20Collective-blue.svg?logo=opencollective&logoColor=white)](https://opencollective.com/sefineh-ai)


**Amharic tokenizer with a GPT-style BPE-like pipeline over decomposed fidel.**
Implements: **cleaning → fidel decomposition → BPE training/application → detokenization**, with a **Cython core for speed**.

---
## What's new in v0.2.5
- Vocab size: 10000 tokens
- Trained on a larger and more diverse Amharic corpus
- Improved tokenization quality and detokenization accuracy
- Better handling of edge cases and rare words
1. **Pretrained tokenizer loading**

  - You can now load a pretrained tokenizer directly:

   ```python
   from amharic_tokenizer import AmharicTokenizer
   tok = AmharicTokenizer.load("amh_bpe_v0.2.5")
   ```
   This version includes a pretrained model (`amh_bpe_v0.2.5`) that can be used immediately without any additional setup and training.

2. **Full token-to-ID and ID-to-token functionality**
  - Added complete round-trip processing methods:
   ```python
   tokens = tok.tokenize(text)
   ids = tok.encode(tokens)
   detokenized = tok.detokenize(tokens)
   ```
   The tokenizer now supports seamless conversion between tokens and IDs, ensuring full consistency between tokenization and detokenization.
   
---

### Test Script: test_roundtrip_basic.py

```python
from amharic_tokenizer import AmharicTokenizer
def test_roundtrip_basic():
    """Load a trained tokenizer, tokenize text, convert to IDs, and detokenize."""
    tok = AmharicTokenizer.load("amh_bpe_v0.2.4")
    text = (
        "የኮሪደር ልማት ገፀ በረከት የሆናቸው የከተማችን ሰፈሮች በነዋሪዎች አንደበት በሰዓት 209 ኪሎ ሜትር የሚጓዘው አውሎ ንፋስ ከጃማይካ ቀጥሎ ኩባ ደርሷል ጠቅላይ" )

    tokens = tok.tokenize(text)
    ids = tok.encode(text)
    detokenized = tok.detokenize(tokens)
    print("Original Text: ", text)
    print("Tokens: ", tokens)
    print("IDs: ", ids)
    print("Detokenized Text: ", detokenized)
    assert text == detokenized, "Detokenized text does not match the original."
if __name__ == "__main__":
    test_roundtrip_basic()

Output:    
    Tokenizer state loaded from amh_bpe_v0.2.4.json
    Original Text:  የኮሪደር ልማት ገፀ በረከት የሆናቸው የከተማችን ሰፈሮች በነዋሪዎች አንደበት በሰዓት 209 ኪሎ ሜትር የሚጓዘው አውሎ ንፋስ ከጃማይካ ቀጥሎ ኩባ ደርሷል ጠቅላይ
    Tokens:  ['የአከኦ', 'ረኢደአረእ<eow>', 'ለእመኣተእ<eow>', 'ገአ', 'ፀ', 'አ<eow>', 'በአረአ', 'ከአተእ<eow>', 'የአሀኦነ', 'ኣቸአወእ<eow>', 'የአ', 'ከአተአመኣ', 'ቸእነእ<eow>', 'ሰአፈአረ', 'ኦቸእ<eow>', 'በአ', 'ነአወኣረኢወኦቸእ<eow>', 'አነእደአ', 'በአተእ<eow>', 'በአሰአ', 'ዓተእ<eow>', '2', '0', '9', '<eow>', 'ከኢለኦ<eow>', 'መኤተእረእ<eow>', 'የአመኢ', 'ጓ', 'ዘ', 'አወእ<eow>', 'አወ', 'እለኦ<eow>', 'ነእ', 'ፈኣ', 'ሰእ<eow>', 'ከአ', 'ጀኣ', 'መኣየእ', 'ከኣ<eow>', 'ቀአጠእለኦ<eow>', 'ከኡ', 'በኣ<eow>', 'ደአረእሰ', 'ኡኣለእ<eow>', 'ጠአቀእለኣየእ<eow>']
    IDs:  [2794, 4229, 1136, 66, 37, 79, 711, 1556, 1480, 116, 43, 1467, 1162, 4664, 68, 45, 1618, 2182, 219, 1831, 879, 1, 1, 1, 0, 2824, 2684, 95, 1, 27, 58, 46, 4373, 67, 206, 83, 62, 1083, 4653, 230, 3916, 191, 202, 1221, 477, 496]
    Detokenized Text:  የኮሪደር ልማት ገፀ በረከት የሆናቸው የከተማችን ሰፈሮች በነዋሪዎች አንደበት በሰዓት 209 ኪሎ ሜትር የሚጓዘው አውሎ ንፋስ ከጃማይካ ቀጥሎ ኩባ ደርሷል ጠቅላይ
```
### Additional Improvements
* Added `vocab_size` property for inspecting model vocabulary.
* Added `test_roundtrip_basic.py` example script for verifying tokenizer round-trip behavior.
* Internal `<eow>` token remains an end-of-word marker and is excluded from final detokenized output.
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
## Training (Python)
```python
from amharic_tokenizer.tokenizer import AmharicTokenizer

tokenizer = AmharicTokenizer(vocab_size=5000, num_merges=2000)
tokenizer.train(corpus_text, verbose=True, log_every=100)
tokenizer.save("amh_bpe_model")
tokenizer = AmharicTokenizer.load("amh_bpe_model")

```
---

## Quick Usage (Python)

```python
from amharic_tokenizer import AmharicTokenizer

# Load a trained model
tok = AmharicTokenizer.load("amh_bpe_v0.2.4")

text = "ኢትዮጵያ ጥሩ ናት።"

# Tokenize
tokens = tok.tokenize(text)
print(tokens)  # variable-length subword tokens
# Tokens to ids
ids = tok.encode(text) # or tok.convert_tokens_to_ids(tokens)
decoded = tok.decode(ids)  # or tok.detokenize(tokens)

display_tokens = [t.replace('<eow>', '') for t in tokens if t != '<eow>']
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
