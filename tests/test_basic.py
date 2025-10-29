from amharic_tokenizer import AmharicTokenizer

def test_roundtrip_basic():
    tok = AmharicTokenizer.load('amh_bpe')
    text = "ስዊድን ከኢትዮጵያ ጋር ያላትን ግንኙነት አስመልክቶ አዲስ የትብብር ስልት መነደፉን አምባሳደሩ ገልጸዋል"
    tokens = tok.tokenize(text)
    print(tokens)
    ids = tok.convert_tokens_to_ids(tokens)
    print(ids)
    print(tok.convert_ids_to_tokens(ids))
    display_tokens = [t.replace('</w>', '') for t in tokens if t != '</w>']
    print("Display Tokens:", display_tokens)
    print(tok.detokenize(tokens))
if __name__ == "__main__":
    test_roundtrip_basic()