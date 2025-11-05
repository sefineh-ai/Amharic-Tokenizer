# amharic_tokenizer.pyx

import re
import json
from collections import Counter
from typing import List
from amharic_tokenizer.fidel_map import AMHARIC_FIDEL_MAP, REVERSE_FIDEL_MAP
cdef class AmharicTokenizer:
    """
    Optimized BPE Tokenizer for Amharic Fidel using Cython.
    
    The training process is now restricted by max_vocab_size to prevent
    unintended growth of the vocabulary.
    """

    cdef public dict _vocabulary
    cdef public dict _merge_rank_map
    cdef public int _num_merges 
    cdef public int _max_vocab_size 
    cdef public dict _token_to_id
    cdef public dict _id_to_token
    cdef public int _next_id

    def __init__(self, int num_merges=50000, int max_vocab_size=5000):
        self._vocabulary = {}
        self._merge_rank_map = {}
        self._num_merges = num_merges
        self._max_vocab_size = max_vocab_size 
        self._token_to_id = {}
        self._id_to_token = {}
        self._next_id = 0
        self._initialize_base_vocabulary()

    cpdef str _clean_corpus(self, str text):
        """
        Clean the Amharic corpus by:
        - Removing English letters and numbers
        - Optionally remove unwanted punctuation
        - Keep Amharic Fidel characters and whitespace
        """
        cleaned_text = re.sub(r'[A-Za-z0-9]', '', text)
        cleaned_text = re.sub(r'[^\u1200-\u137F\s]', '', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text

    cdef void _initialize_base_vocabulary(self):
        cdef set initial_tokens = set()
        cdef str char
        for char_list in AMHARIC_FIDEL_MAP.values():
            for char in char_list:
                initial_tokens.add(char)
        initial_tokens.add('<eow>')
        initial_tokens.add('<unk>') 
        
        cdef list sorted_initial_tokens = sorted(list(initial_tokens))
        for token in sorted_initial_tokens:
            self._vocabulary[token] = 0
            self._add_to_vocab_maps(token)

    cdef void _add_to_vocab_maps(self, str token):
        if token not in self._token_to_id:
            self._token_to_id[token] = self._next_id
            self._id_to_token[self._next_id] = token
            self._next_id += 1

    @staticmethod
    def _get_pairs(tokens: List[str]):
        cdef dict pairs = {}
        cdef int i
        cdef tuple pair
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i+1])
            pairs[pair] = pairs.get(pair, 0) + 1
        return pairs

    cpdef List[List[str]] preprocess(self, str amharic_corpus):
        cdef list words = amharic_corpus.split()
        cdef list preprocessed_corpus = []
        cdef list mapped_word
        cdef str char
        for word in words:
            mapped_word = []
            for char in word:
                mapped_word.extend(list(AMHARIC_FIDEL_MAP.get(char, [char])))
            mapped_word.append('<eow>')
            preprocessed_corpus.append(mapped_word)
        return preprocessed_corpus

    cpdef int train(self, str amharic_corpus, bint verbose=False, int log_every=1000):
        cdef list tokenized_words = self.preprocess(self._clean_corpus(amharic_corpus))
        cdef pair_counts = Counter()
        cdef list word_tokens
        cdef dict word_pairs
        for word_tokens in tokenized_words:
            word_pairs = self._get_pairs(word_tokens)
            pair_counts.update(word_pairs)

        cdef int i, j
        cdef str new_token
        cdef tuple best_pair
        cdef list token_list, new_list, new_tokenized_words
        cdef object new_pair_counts
        for i in range(self._num_merges):
            if len(self._vocabulary) >= self._max_vocab_size:
                if verbose:
                    print(f"Stopping BPE training. Max vocabulary size ({self._max_vocab_size}) reached.")
                break
            if verbose and (i + 1) % log_every == 0:
                print(f"Merge {i + 1}/{self._num_merges} completed. Current vocab size: {len(self._vocabulary)}")
                
            if not pair_counts:
                break
                
            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] < 2:
                break
                
            new_token = ''.join(best_pair)
            if new_token not in self._vocabulary:
                self._merge_rank_map[new_token] = len(self._merge_rank_map) + 1
                self._vocabulary[new_token] = pair_counts[best_pair]
                self._add_to_vocab_maps(new_token)

            new_tokenized_words = []
            new_pair_counts = pair_counts.copy()
            for token_list in tokenized_words:
                if new_token in ''.join(token_list) or best_pair in self._get_pairs(token_list):
                    old_pairs = self._get_pairs(token_list)
                    new_list = []
                    j = 0
                    while j < len(token_list):
                        if j < len(token_list) - 1 and (token_list[j], token_list[j+1]) == best_pair:
                            new_list.append(new_token)
                            j += 2
                        else:
                            new_list.append(token_list[j])
                            j += 1
                    new_tokenized_words.append(new_list)
                    new_pair_counts.subtract(old_pairs)
                    new_pair_counts.update(self._get_pairs(new_list))
                else:
                    new_tokenized_words.append(token_list)
            tokenized_words = new_tokenized_words
            pair_counts = new_pair_counts
        return len(self._merge_rank_map)

    cpdef tuple _get_best_merge(self, list current_corpus, dict reversed_merge_map):
        cdef tuple best_pair = None
        cdef float highest_priority_rank = float('inf') 
        cdef list token_list
        cdef int j
        cdef str pair_str
        cdef object rank
        for token_list in current_corpus:
            for j in range(len(token_list) - 1):
                pair_str = token_list[j] + token_list[j+1]
                rank = reversed_merge_map.get(pair_str)
                if rank is not None and rank < highest_priority_rank:
                    highest_priority_rank = rank
                    best_pair = (token_list[j], token_list[j+1])
        return best_pair

    cpdef List[str] tokenize(self, str text):
        cdef list corpus = self.preprocess(text)
        cdef dict reversed_merge_map = self._merge_rank_map
        cdef tuple best_pair
        cdef int i, j
        cdef str pair_str
        cdef list updated_corpus, token_list, new_token_list
        while True:
            best_pair = self._get_best_merge(corpus, reversed_merge_map)
            if best_pair is None:
                break
            pair_str = ''.join(best_pair)
            updated_corpus = []
            for token_list in corpus:
                new_token_list = []
                i = 0
                while i < len(token_list):
                    if i < len(token_list) - 1 and (token_list[i], token_list[i+1]) == best_pair:
                        new_token_list.append(pair_str)
                        i += 2
                    else:
                        new_token_list.append(token_list[i])
                        i += 1
                updated_corpus.append(new_token_list)
            corpus = updated_corpus
        return [token for word_tokens in corpus for token in word_tokens]

    cpdef List[int] encode(self, str text):
        cdef list tokens = self.tokenize(text)
        return [self._token_to_id.get(token, self._token_to_id.get('<unk>', -1)) for token in tokens]

    cpdef str decode(self, List[int] token_ids):
        cdef list tokens = [self._id_to_token.get(i, '<unk>') for i in token_ids]
        return self.detokenize(tokens)

    cpdef str detokenize(self, List[str] tokens):
        cdef str temp_string = "".join(tokens).replace("<eow>", " ")
        cdef list word_segments = temp_string.split()
        cdef list final_text_words = []
        cdef int i, length
        cdef str sub_string
        cdef bint found_match
        cdef list chars_list, reconstructed_word
        cdef int MAX_CV_LENGTH = 3
        
        for word_string in word_segments:
            chars_list = list(word_string)
            reconstructed_word = []
            i = 0
            while i < len(chars_list):
                found_match = False
                for length in range(MAX_CV_LENGTH, 0, -1):
                    sub_string = "".join(chars_list[i:i+length])
                    if sub_string in REVERSE_FIDEL_MAP:
                        reconstructed_word.append(REVERSE_FIDEL_MAP[sub_string])
                        i += length
                        found_match = True
                        break
                if not found_match:
                    reconstructed_word.append(chars_list[i])
                    i += 1
            final_text_words.append("".join(reconstructed_word))
        return " ".join(final_text_words).replace("<unk>", "")

    cpdef void save(self, str file_path):
        if not file_path.endswith(".json"):
            file_path += ".json"

        cdef dict state = {
            'num_merges': self._num_merges,
            'max_vocab_size': self._max_vocab_size,
            'vocabulary': self._vocabulary,
            'merge_rank_map': self._merge_rank_map,
            'token_to_id': self._token_to_id,
            'id_to_token': {str(k): v for k, v in self._id_to_token.items()}, 
            'next_id': self._next_id
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=4)
        print(f"Tokenizer state saved to {file_path}")

    @classmethod
    def load(cls, file_path):
        if not file_path.endswith(".json"):
            file_path += ".json"
        with open(file_path, 'r', encoding='utf-8') as f:
            state = json.load(f)
            
        tokenizer = cls(
            num_merges=state.get('num_merges', 50000), 
            max_vocab_size=state.get('max_vocab_size', 5000)
        )
        tokenizer._vocabulary = state['vocabulary']
        tokenizer._merge_rank_map = state['merge_rank_map']
        tokenizer._token_to_id = state['token_to_id']
        tokenizer._id_to_token = {int(k): v for k, v in state['id_to_token'].items()}
        tokenizer._next_id = state['next_id']
        print(f"Tokenizer state loaded from {file_path}")
        return tokenizer