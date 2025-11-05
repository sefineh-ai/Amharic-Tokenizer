"""Test script for basic roundtrip functionality of the AmharicTokenizer."""

from amharic_tokenizer import AmharicTokenizer


def test_roundtrip_basic():
    """Load a trained tokenizer, tokenize text, convert to IDs, and detokenize."""
    tok = AmharicTokenizer.load("amh_bpe_v0.2.3")
    text = (
        "የኮሪደር ልማት ገፀ በረከት የሆናቸው የከተማችን ሰፈሮች በነዋሪዎች አንደበት በሰዓት 209 ኪሎ ሜትር የሚጓዘው አውሎ ንፋስ ከጃማይካ ቀጥሎ ኩባ ደርሷል ጠቅላይ" )

    # Tokenize text
    ids = tok.encode(text)
    print("Tokens", tok.decode(ids))
    print(ids)
    print(tok.tokenize(text))
    print(tok.detokenize(tok.tokenize(text)))
    

if __name__ == "__main__":
    test_roundtrip_basic()
