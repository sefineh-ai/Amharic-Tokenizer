# Amharic Tokenizer

A fast and linguistically informed Amharic tokenizer based on base+vowel decomposition.
Built with Cython for speed and accuracy.

### Example

```python
from amharic_tokenizer import AmharicTokenizer

tokenizer = AmharicTokenizer.from_default()
text = "ሀሁሂ"
tokens = tokenizer.tokenize(text)
print(tokens)
# ['ሀአ', 'ሀኡ', 'ሀኢ']

print(tokenizer.detokenize(tokens))
# ሀሁሂ
```