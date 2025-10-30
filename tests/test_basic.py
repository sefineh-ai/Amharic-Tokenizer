"""Test script for basic roundtrip functionality of the AmharicTokenizer."""

from amharic_tokenizer import AmharicTokenizer


def test_roundtrip_basic():
    """Load a trained tokenizer, tokenize text, convert to IDs, and detokenize."""
    tok = AmharicTokenizer.load("amh_bpe_v0.2.0")

    text = (
        "ስዊድን ከኢትዮጵያ ጋር ያላትን ግንኙነት አስመልክቶ "
        "አዲስ የትብብር ስልት መነደፉን አምባሳደሩ ገልጸዋል"
    )

    # Tokenize text
    tokens = tok.tokenize(text)
    print("Tokens:", tokens)

    # Convert tokens to IDs
    ids = tok.convert_tokens_to_ids(tokens)
    print("IDs:", ids)

    # Convert IDs back to tokens
    tokens_from_ids = tok.convert_ids_to_tokens(ids)
    print("Tokens from IDs:", tokens_from_ids)

    # Clean tokens for display
    display_tokens = [t.replace("</w>", "") for t in tokens if t != "</w>"]
    print("Display Tokens:", display_tokens)

    # Detokenize back to original text
    detokenized = tok.detokenize(tokens)
    print("Detokenized:", detokenized)
    print("vocab_size:", tok.vocab_size)

if __name__ == "__main__":
    test_roundtrip_basic()
