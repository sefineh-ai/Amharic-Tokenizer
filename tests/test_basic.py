"""Test script for basic roundtrip functionality of the AmharicTokenizer."""

from amharic_tokenizer import AmharicTokenizer
def test_roundtrip_basic():
    """Load a trained tokenizer, tokenize text, convert to IDs, and detokenize."""
    tok = AmharicTokenizer.load("amh_bpe_vv0.2.4")
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
