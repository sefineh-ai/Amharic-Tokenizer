## Amharic Tokenizer Docs

### Overview
This project implements a BPE-like tokenizer for Amharic by first decomposing fidel to base+vowel strings, learning frequent merges (BPE), and composing back on detokenization.

### Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Train (CLI)
```bash
amh-tokenizer train cleaned_amharic.txt amh_bpe --num-merges 50000 --verbose --log-every 2000
```

### Inference
```python
from amharic_tokenizer import AmharicTokenizer
tok = AmharicTokenizer.load("amh_bpe")
tokens = tok.tokenize("ኢትዮጵያ ጥሩ ናት።")
print(tokens)
print(tok.detokenize(tokens))
```

### Programmatic training
```python
from amharic_tokenizer.pipeline import train_and_save
train_and_save("cleaned_amharic.txt", "amh_bpe", num_merges=50000, verbose=True)
```

### Notes
- Longer corpora and higher merges yield longer subwords.
- Training/tokenization operate on decomposed fidel; detokenization composes back.


