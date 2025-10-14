
from amharic_tokenizer import AmharicTokenizer

def test_roundtrip_basic():
    tok = AmharicTokenizer.from_default()
    text = "ሀሁሂ"
    tokens = tok.tokenize(text)
    assert isinstance(tokens, list), f"Tokens should be a list, got {type(tokens)}"
    assert tok.detokenize(tokens) == text, f"Detokenization should return the original text, got {tok.detokenize(tokens)}"  
if __name__ == "__main__":
    test_roundtrip_basic()
    print("All tests passed")