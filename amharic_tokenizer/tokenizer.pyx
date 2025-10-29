# cython: language_level=3
import json
import re
from amharic_tokenizer.fidel_map import AMHARIC_FIDEL_MAP

cdef class AmharicTokenizer:
    cdef dict _fidel_map
    cdef public dict _reverse_map
    cdef int vocab_size
    cdef int num_merges
    cdef public list _merges
    cdef public set _merge_lookup
    cdef public dict _vocab
    cdef public dict _merge_ranks
    cdef public dict _token_to_id
    cdef public dict _id_to_token

    def __init__(self, fidel_map=None, vocab_size=5000, num_merges=100):
        self._fidel_map = fidel_map or AMHARIC_FIDEL_MAP
        self.vocab_size = vocab_size
        self.num_merges = num_merges
        self._merges = []
        self._merge_lookup = set()
        self._vocab = {}
        self._reverse_map = {v: k for k, v in self._fidel_map.items()}
        self._merge_ranks = {}
        self._token_to_id = {}
        self._id_to_token = {}
    
    cpdef build_token_mappings(self):
        """
        Builds token -> ID and ID -> token dictionaries, ensuring all
        tokens, including '##' prefixed continuations and the space token, are mapped.
        """
        cdef list special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>", " "] # <-- CRITICAL: Include the space token
        self._token_to_id = {}
        self._id_to_token = {}
        cdef int idx = 0
        cdef list raw_bpe_tokens
        cdef str token
        
        for t in special_tokens:
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1
            
        raw_bpe_tokens = self.get_vocab_tokens()
        for token in raw_bpe_tokens:
            if token not in self._token_to_id:
                self._token_to_id[token] = idx
                self._id_to_token[idx] = token
                idx += 1
        
            if token != "</w>":
                prefixed_token = "##" + token
                if prefixed_token not in self._token_to_id:
                     self._token_to_id[prefixed_token] = idx
                     self._id_to_token[idx] = prefixed_token
                     idx += 1

    cpdef list convert_tokens_to_ids(self, list tokens):
        """
        Converts a list of tokens to a list of IDs.
        Unknown tokens are mapped to <unk>.
        """
        unk_id = self._token_to_id.get("<unk>", 1)
        return [self._token_to_id.get(t, unk_id) for t in tokens]

    cpdef list convert_ids_to_tokens(self, list ids):
        """
        Converts a list of IDs to a list of tokens.
        Unknown IDs are mapped to <unk>.
        """
        unk_token = "<unk>"
        return [self._id_to_token.get(i, unk_token) for i in ids]

    cpdef list get_vocab_tokens(self):
        """
        Returns the list of all unique BPE tokens in the vocab.
        This includes all base characters (initial vocabulary) and all learned subwords.
        """
        cdef set tokens = set()
        cdef str word
        for word in self._vocab.keys():
            tokens.update(word.split())
        cdef tuple pair
        for pair in self._merges:
            tokens.add(pair[0])
            tokens.add(pair[1])
            tokens.add(''.join(pair))
        for t in tokens.copy():
            for char in t:
                if len(char) == 1 and char not in tokens:
                     tokens.add(char)

        tokens.add("</w>") 

        return sorted(tokens)


    cpdef bint is_trained(self):
        return bool(self._merges)

    cpdef str _clean(self, str text):
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text)
        return text.strip()

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

    cpdef int train(self, str corpus_text, bint verbose=False, int log_every=1000):
        cdef str cleaned = self._clean(corpus_text)
        cdef str corpus = self._decompose(cleaned)
        cdef dict vocab = self._get_vocab(corpus)
        self._vocab = vocab.copy()
        cdef list merges = []
        cdef dict pairs
        cdef tuple best_pair
        cdef int i

        if verbose:
            print(f"[AMH-Tokenizer] Training start: target_merges={self.num_merges}, corpus_chars={len(corpus)}")

        for i in range(self.num_merges):
            pairs = self._get_stats(vocab)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            vocab = self._merge_vocab(best_pair, vocab)
            merges.append(best_pair)
            if verbose and (i + 1) % log_every == 0:
                print(f"[AMH-Tokenizer] Merges learned: {i + 1}")

        self._merges = merges
        self._merge_lookup = set(merges)
        self._vocab = vocab
        self._merge_ranks = {pair: idx for idx, pair in enumerate(self._merges)}
        self.build_token_mappings()
        if verbose:
            print(f"[AMH-Tokenizer] Training complete: total_merges={len(self._merges)}")
        return len(self._merges)

    cpdef list encode(self, str text):
        """
        Tokenizes the text and converts the resulting tokens to a list of IDs.
        """
        cdef list tokens = self.tokenize(text)
        return self.convert_tokens_to_ids(tokens)
    cpdef str decode(self, list ids):
        """
        Converts a list of IDs back to text.
        """
        cdef list tokens = self.convert_ids_to_tokens(ids)
        return self.detokenize(tokens)

    cpdef list _apply_bpe(self, list seqs):
        cdef dict ranks = self._merge_ranks
        if not ranks:
            return seqs
        cdef list tokens = seqs[:]
        cdef int i
        cdef tuple pair
        cdef tuple best_pair
        cdef int best_rank
        cdef int idx
        while True:
            best_pair = None
            best_rank = 0x7fffffff
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                if pair in ranks:
                    idx = ranks[pair]
                    if idx < best_rank:
                        best_rank = idx
                        best_pair = pair
            if best_pair is None:
                break
            i = 0
            while i < len(tokens) - 1:
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                    tokens[i:i+2] = [''.join(best_pair)]
                else:
                    i += 1
        return tokens

    cpdef list tokenize(self, str text):
        cdef str cleaned = self._clean(text)
        if not self._merges:
            raise ValueError("Tokenizer is not trained. Call train(corpus_text) before tokenize().")
        cdef list words = cleaned.split()
        cdef list tokenized = []
        cdef str word
        cdef list seqs
        cdef list prefixed
        cdef bint is_first
        cdef str core
        for word in words:
            seqs = list(self._decompose(word))
            seqs.append("</w>")
            seqs = self._apply_bpe(seqs)
            prefixed = []
            is_first = True
            for t in seqs:
                if t == "</w>":
                    prefixed.append(t)
                    is_first = True
                else:
                    if t.endswith("</w>"):
                        core = t[:-4]
                        if is_first:
                            prefixed.append(core)
                        else:
                            prefixed.append("##" + core)
                        prefixed.append("</w>")
                        is_first = True
                    else:
                        if is_first:
                            prefixed.append(t)
                            is_first = False
                        else:
                            prefixed.append("##" + t)
            tokenized.extend(prefixed)
            tokenized.append(" ")
        if tokenized:
            tokenized.pop()
        return tokenized

    cpdef str detokenize(self, list tokens):
        cdef list words = []
        cdef list current_word = []
        cdef str t
        cdef str piece
        for t in tokens:
            if t == " ":
                if current_word:
                    words.append(self._compose(''.join(current_word)))
                    current_word = []
            elif t == "</w>":
                words.append(self._compose(''.join(current_word)))
                current_word = []
            else:
                piece = t[2:] if t.startswith("##") else t
                if piece.endswith("</w>"):
                    piece = piece[:-4]
                    if piece:
                        current_word.append(piece)
                    words.append(self._compose(''.join(current_word)))
                    current_word = []
                else:
                    current_word.append(piece)
        if current_word:
            words.append(self._compose(''.join(current_word)))
        return ' '.join(words)

    cpdef save(self, str path_prefix):
        data = {
            "merges": self._merges,
            "vocab": self._vocab,
            "reverse_map": self._reverse_map,
            "token_to_id": self._token_to_id,
            "id_to_token": self._id_to_token
        }
        with open(f"{path_prefix}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, str path_prefix):
        with open(f"{path_prefix}.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        tokenizer = cls()
        tokenizer._merges = [tuple(p) for p in data["merges"]]
        tokenizer._merge_lookup = set(tokenizer._merges)
        tokenizer._vocab = data["vocab"]
        tokenizer._reverse_map = data["reverse_map"]
        tokenizer._merge_ranks = {pair: idx for idx, pair in enumerate(tokenizer._merges)}
        tokenizer._token_to_id = data["token_to_id"]
        tokenizer._id_to_token = {v: k for k, v in tokenizer._token_to_id.items()} 
        return tokenizer
