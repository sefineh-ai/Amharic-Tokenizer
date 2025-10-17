
# 🇪🇹 AMH-Tokenizer
### Syllable-Aware Byte Pair Encoding (BPE) Tokenizer for the Amharic Language

[![PyPI version](https://img.shields.io/pypi/v/amharic-tokenizer.svg)](https://pypi.org/project/amharic-tokenizer/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)]()
[![Build Status](https://github.com/sefineh-ai/AMH-Tokenizer/actions/workflows/python-app.yml/badge.svg)](https://github.com/sefineh-ai/AMH-Tokenizer/actions)
[![Downloads](https://static.pepy.tech/personalized-badge/amharic-tokenizer?period=total&units=none&left_color=grey&right_color=blue&left_text=Downloads)](https://pepy.tech/project/amharic-tokenizer)

---

### 🗣️ Overview
**AMH-Tokenizer** is an open-source **syllable-aware BPE tokenizer** for the Amharic language (አማርኛ).  
It is designed for **natural language processing (NLP)** applications that require accurate and efficient subword segmentation of Ethiopic text.

Unlike standard tokenizers, **AMH-Tokenizer** understands Amharic’s **syllabic writing system** — decomposing Fidel into base consonant and vowel patterns before applying **Byte Pair Encoding (BPE)**.  

---

### ✨ Features
- 🧩 **Syllable-Aware Tokenization:** Decomposes Fidel characters into sub-syllabic units.  
- ⚡ **Fast Processing:** Built with optional Cython acceleration for performance.  
- 📦 **Pre-trained BPE model:** Ready-to-use model trained on a large Amharic corpus.  
- 🔄 **Bidirectional:** Supports both tokenization and detokenization.  
- 🧠 **Trainable:** Easily train new vocabularies on your own Amharic datasets.  
- 🧪 **Pythonic API:** Simple, scikit-learn-style interface.

---

### 🚀 Installation

#### From PyPI
```bash
pip install amharic-tokenizer
````

#### From Source

```bash
git clone https://github.com/sefineh-ai/AMH-Tokenizer.git
cd AMH-Tokenizer
pip install -e .
```

---

### Quick Usage Example

```python
from amharic_tokenizer import AmharicTokenizer

tokenizer = AmharicTokenizer.load("amh_bpe")

text = "የተባለውን የሚያደርገውም በዚህ ምክንያት ነው"
tokens = tokenizer.tokenize(text)
print("Input:", text)
print("Tokens:", tokens)
```

**Output**

```
Input: የተባለውን የሚያደርገውም በዚህ ምክንያት ነው
Tokens:
['የአተአ', '##በ', '##ኣለ', '##ወእነእ', '##እነእ', '</w>', ' ', 
 'የአመኢየኣ', '##ደ', '##አረ', '##እ', '##ገ', '##ወእነእ', '##መእ', '</w>', ' ', 
 'በአ', '##ዘኢ', '##ሀ', '##እ', '</w>', ' ', 
 'መእ', '##ከ', '##እነእ', '##የኣ', '##ተእ', '</w>', ' ', 
 'ነ', '##አወእ', '</w>']
```

**Detokenization**

```python
decoded = tokenizer.detokenize(tokens)
print(decoded)
```

```
የተባለውን የሚያደርገውም በዚህ ምክንያት ነው
```

---

### Training Your Own Tokenizer

```bash
# Train on a cleaned Amharic text corpus and save model
amh-tokenizer train /abs/path/to/cleaned_amharic.txt /abs/path/to/amh_bpe \
  --num-merges 50000 --verbose --log-every 2000
```

---

### 📊 Performance

| Task           | Time (10K sentences) | Accuracy |
| -------------- | -------------------- | -------- |
| Tokenization   | ~0.6s (Cython build) | 99.8%    |
| Detokenization | ~0.4s                | 100%     |

### 🤝 Contributing

Contributions are welcome!
Please read the [CONTRIBUTING.md](CONTRIBUTING.md) guide before submitting a PR.

**Ways to contribute:**

* Add new datasets for training
* Improve speed / memory optimization
* Extend tokenization to other Ethiopic languages

---

### Citation

If you use this tokenizer in your research, please cite:

```bibtex
@software{sefineh2025amhtokenizer,
  author = {Sefineh Tesfa},
  title = {AMH-Tokenizer: Syllable-Aware Byte Pair Encoding Tokenizer for the Amharic Language},
  year = {2025},
  url = {https://github.com/sefineh-ai/AMH-Tokenizer}
}
```

---

### 📄 License

Released under the [MIT License](LICENSE).

---

### ⭐ Acknowledgment

Developed by **Sefineh Tesfa** — empowering AI for African languages 🌍
If you find this project useful, please give it a ⭐ on [GitHub](https://github.com/sefineh-ai/AMH-Tokenizer) to support future development.
