# cython: language_level=3
import json
from amharic_tokenizer.fidel_map import AMHARIC_FIDEL_MAP

cdef class AmharicTokenizer:
    cdef dict _fidel_map
    cdef dict _reverse_map
    cdef int vocab_size
    cdef int num_merges
    cdef list _merges
    cdef set _merge_lookup
    cdef dict _vocab

    def __init__(self, fidel_map=None, vocab_size=5000, num_merges=100):
        self._fidel_map = fidel_map or AMHARIC_FIDEL_MAP
        self.vocab_size = vocab_size
        self.num_merges = num_merges
        self._merges = []
        self._merge_lookup = set()
        self._vocab = {}
        self._reverse_map = {v: k for k, v in self._fidel_map.items()}

    @classmethod
    def from_default(cls):
        return cls()

    cpdef str _decompose(self, text):
        cdef list chars = []
        cdef str c
        for c in text:
            chars.append(self._fidel_map.get(c, c))
        return ''.join(chars)

    cpdef str _compose(self, text):
        cdef int i = 0
        cdef int n = len(text)
        cdef list result = []
        cdef list keys_sorted = sorted(self._reverse_map.keys(), key=len, reverse=True)
        cdef bint matched
        cdef str key
        while i < n:
            matched = False
            for key in keys_sorted:
                if text.startswith(key, i):
                    result.append(self._reverse_map[key])
                    i += len(key)
                    matched = True
                    break
            if not matched:
                result.append(text[i])
                i += 1
        return ''.join(result)

    cpdef dict _get_vocab(self, corpus):
        cdef dict vocab = {}
        cdef str word
        cdef str key
        for word in corpus.split():
            tokens = list(word)
            tokens.append("</w>")
            key = ' '.join(tokens)
            vocab[key] = vocab.get(key, 0) + 1
        return vocab

    cpdef dict _get_stats(self, dict vocab):
        cdef dict pairs = {}
        cdef str word
        cdef int freq
        cdef list symbols
        cdef int i
        cdef tuple p
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                p = (symbols[i], symbols[i+1])
                pairs[p] = pairs.get(p, 0) + freq
        return pairs

    cpdef dict _merge_vocab(self, tuple pair, dict vocab):
        cdef str merged = ''.join(pair)
        cdef dict new_vocab = {}
        cdef str pair_str = ' '.join(pair)
        cdef str word
        cdef int freq
        for word, freq in vocab.items():
            new_vocab[word.replace(pair_str, merged)] = freq
        return new_vocab

    cpdef train(self, str corpus_text):
        cdef str corpus = self._decompose(corpus_text)
        cdef dict vocab = self._get_vocab(corpus)
        self._vocab = vocab.copy()
        cdef list merges = []
        cdef dict pairs
        cdef tuple best_pair
        cdef int i

        for i in range(self.num_merges):
            pairs = self._get_stats(vocab)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            vocab = self._merge_vocab(best_pair, vocab)
            merges.append(best_pair)

        self._merges = merges
        self._merge_lookup = set(merges)
        self._vocab = vocab

    cpdef list tokenize(self, str text):
        cdef list words = text.split()
        cdef list tokenized = []
        cdef str word
        cdef list seqs
        cdef int i
        cdef tuple pair
        for word in words:
            seqs = []
            for c in word:
                seqs.append(self._fidel_map.get(c, c))
            seqs.append("</w>")
            i = 0
            while i < len(seqs)-1:
                pair = (seqs[i], seqs[i+1])
                if pair in self._merge_lookup:
                    seqs[i:i+2] = [''.join(pair)]
                    i = max(i-1, 0)
                else:
                    i += 1
            tokenized.extend(seqs)
            tokenized.append(" ")
        if tokenized:
            tokenized.pop()
        return tokenized

    cpdef str detokenize(self, list tokens):
        cdef list words = []
        cdef list current_word = []
        cdef str t
        for t in tokens:
            if t == " ":
                words.append(self._compose(''.join(current_word)))
                current_word = []
            elif t != "</w>":
                current_word.append(t)
        if current_word:
            words.append(self._compose(''.join(current_word)))
        return ' '.join(words)

    cpdef save(self, str path_prefix):
        data = {
            "merges": self._merges,
            "vocab": self._vocab,
            "reverse_map": self._reverse_map
        }
        with open(f"{path_prefix}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, str path_prefix):
        with open(f"{path_prefix}.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        tokenizer = cls()
        tokenizer._merges = data["merges"]
        tokenizer._merge_lookup = set(tokenizer._merges)
        tokenizer._vocab = data["vocab"]
        tokenizer._reverse_map = data["reverse_map"]
        return tokenizer
