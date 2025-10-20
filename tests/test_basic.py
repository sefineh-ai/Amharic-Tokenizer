
from amharic_tokenizer import AmharicTokenizer

def test_roundtrip_basic():
    tok = AmharicTokenizer.load('amh_bpe_v0.1.2')
    text = "ኢትዮጵያ በጊዜው ያልከፈለችውን የአንድ ቢሊዮን ዶላር ዩሮ ቦንድ በተመለከተ በገንዘብ ሚኒስቴር እና በግል አበዳሪዎች ኮሚቴ መካከል ሲደርግ የነበረው ድርድር ያለ ስምምነት ተጠናቀቀ።"
    tokens = tok.tokenize(text)
    print(tokens)
    assert isinstance(tokens, list), f"Tokens should be a list, got {type(tokens)}"
    assert tok.detokenize(tokens) == text, f"Detokenization should return the original text, got {tok.detokenize(tokens)}"  
if __name__ == "__main__":
    test_roundtrip_basic()
    print("All tests passed")