"""
Amharic BPE Tokenizer

This module implements a byte-pair encoding (BPE) tokenizer optimized
for the Amharic Fidel script. Supports training, tokenization, encoding,
decoding, and saving/loading tokenizer state.
"""

import re
import json
from collections import Counter
from typing import List
from amharic_tokenizer.fidel_map import AMHARIC_FIDEL_MAP, REVERSE_FIDEL_MAP


class AmharicTokenizer:
    """
    Optimized BPE Tokenizer for Amharic Fidel.
    Pure Python version (converted from Cython .pyx).
    """

    def __init__(self, num_merges=50000, max_vocab_size=10000):
        """Initialize tokenizer with merge settings and base vocabulary."""
        self._vocabulary = {}
        self._merge_rank_map = {}
        self._num_merges = num_merges
        self._max_vocab_size = max_vocab_size
        self._token_to_id = {}
        self._id_to_token = {}
        self._next_id = 0
        self._initialize_base_vocabulary()

    def _clean_corpus(self, text: str) -> str:
        """Remove non-Amharic characters and normalize whitespace."""
        cleaned_text = re.sub(r'[A-Za-z0-9]', '', text)
        cleaned_text = re.sub(r'[^\u1200-\u137F\s]', '', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text

    def _initialize_base_vocabulary(self):
        """Initialize vocabulary with base Fidel characters and special tokens."""
        initial_tokens = set()
        for char_list in AMHARIC_FIDEL_MAP.values():
            for char in char_list:
                initial_tokens.add(char)
        initial_tokens.add('<eow>')
        initial_tokens.add('<unk>')

        for token in sorted(initial_tokens):
            self._vocabulary[token] = 0
            self._add_to_vocab_maps(token)

    def _add_to_vocab_maps(self, token: str):
        """Add a token to tokenizer mappings for encoding/decoding."""
        if token not in self._token_to_id:
            self._token_to_id[token] = self._next_id
            self._id_to_token[self._next_id] = token
            self._next_id += 1

    @staticmethod
    def _get_pairs(tokens: List[str]):
        """Return all consecutive token pairs with their counts."""
        pairs = {}
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pairs[pair] = pairs.get(pair, 0) + 1
        return pairs

    def preprocess(self, amharic_corpus: str):
        """Convert text to a list of tokens for BPE processing."""
        words = amharic_corpus.split()
        preprocessed_corpus = []

        for word in words:
            mapped_word = []
            for char in word:
                mapped_word.extend(list(AMHARIC_FIDEL_MAP.get(char, [char])))
            mapped_word.append('<eow>')
            preprocessed_corpus.append(mapped_word)

        return preprocessed_corpus

    def _merge_best_pair(self, tokenized_words, best_pair, new_token):
        """Merge the best pair in all tokenized words and update pair counts."""
        new_tokenized_words = []
        pair_counts = Counter()

        for token_list in tokenized_words:
            new_list = []
            i = 0
            while i < len(token_list):
                if i < len(token_list) - 1 and (token_list[i], token_list[i + 1]) == best_pair:
                    new_list.append(new_token)
                    i += 2
                else:
                    new_list.append(token_list[i])
                    i += 1
            new_tokenized_words.append(new_list)
            pair_counts.update(self._get_pairs(new_list))

        return new_tokenized_words, pair_counts

    def train(self, amharic_corpus: str, verbose=False, log_every=1000):
        """Train BPE merges on the given Amharic corpus."""
        tokenized_words = self.preprocess(self._clean_corpus(amharic_corpus))
        pair_counts = Counter()
        for word_tokens in tokenized_words:
            pair_counts.update(self._get_pairs(word_tokens))

        for i in range(self._num_merges):
            if len(self._vocabulary) >= self._max_vocab_size:
                if verbose:
                    print(
                        f"Max vocabulary size ({self._max_vocab_size}) reached.")
                break

            if verbose and (i + 1) % log_every == 0:
                print(
                    f"Merge {i + 1}/{self._num_merges}. Vocab size: {len(self._vocabulary)}")

            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] < 2:
                break

            new_token = "".join(best_pair)
            if new_token not in self._vocabulary:
                self._merge_rank_map[new_token] = len(self._merge_rank_map) + 1
                self._vocabulary[new_token] = pair_counts[best_pair]
                self._add_to_vocab_maps(new_token)

            tokenized_words, pair_counts = self._merge_best_pair(
                tokenized_words, best_pair, new_token)

        return len(self._merge_rank_map)

    def _get_best_merge(self, current_corpus, reversed_merge_map):
        """Find the highest-priority pair to merge based on trained merges."""
        best_pair = None
        highest_priority_rank = float("inf")

        for token_list in current_corpus:
            for i in range(len(token_list) - 1):
                pair_str = token_list[i] + token_list[i + 1]
                rank = reversed_merge_map.get(pair_str)
                if rank is not None and rank < highest_priority_rank:
                    highest_priority_rank = rank
                    best_pair = (token_list[i], token_list[i + 1])

        return best_pair

    def tokenize(self, text: str):
        """Tokenize input text using trained BPE merges."""
        corpus = self.preprocess(text)
        reversed_merge_map = self._merge_rank_map

        while True:
            best_pair = self._get_best_merge(corpus, reversed_merge_map)
            if best_pair is None:
                break

            pair_str = "".join(best_pair)
            updated_corpus = []

            for token_list in corpus:
                new_list = []
                i = 0
                while i < len(token_list):
                    if i < len(token_list) - 1 and (token_list[i], token_list[i + 1]) == best_pair:
                        new_list.append(pair_str)
                        i += 2
                    else:
                        new_list.append(token_list[i])
                        i += 1
                updated_corpus.append(new_list)

            corpus = updated_corpus

        return [token for word in corpus for token in word]

    def encode(self, text: str):
        """Convert text to token IDs."""
        tokens = self.tokenize(text)
        return [self._token_to_id.get(t, self._token_to_id.get("<unk>", -1)) for t in tokens]

    def decode(self, token_ids: List[int]):
        """Convert token IDs back to text."""
        tokens = [self._id_to_token.get(i, "<unk>") for i in token_ids]
        return self.detokenize(tokens)

    def detokenize(self, tokens: List[str]):
        """Convert tokens back to readable Amharic text."""
        temp_string = "".join(tokens).replace("<eow>", " ")
        word_segments = temp_string.split()

        final_words = []
        max_cv_length = 3

        for word_string in word_segments:
            chars = list(word_string)
            reconstructed = []
            i = 0

            while i < len(chars):
                found = False
                for length in range(max_cv_length, 0, -1):
                    sub = "".join(chars[i: i + length])
                    if sub in REVERSE_FIDEL_MAP:
                        reconstructed.append(REVERSE_FIDEL_MAP[sub])
                        i += length
                        found = True
                        break
                if not found:
                    reconstructed.append(chars[i])
                    i += 1

            final_words.append("".join(reconstructed))

        return " ".join(final_words).replace("<unk>", "")

    def save(self, file_path: str):
        """Save tokenizer state to a JSON file."""
        if not file_path.endswith(".json"):
            file_path += ".json"

        state = {
            "num_merges": self._num_merges,
            "max_vocab_size": self._max_vocab_size,
            "vocabulary": self._vocabulary,
            "merge_rank_map": self._merge_rank_map,
            "token_to_id": self._token_to_id,
            "id_to_token": {str(k): v for k, v in self._id_to_token.items()},
            "next_id": self._next_id,
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=4)

    @classmethod
    def load(cls, file_path):
        """Load tokenizer state from a JSON file."""
        if not file_path.endswith(".json"):
            file_path += ".json"

        with open(file_path, "r", encoding="utf-8") as f:
            state = json.load(f)

        tokenizer = cls(
            num_merges=state.get("num_merges", 50000),
            max_vocab_size=state.get("max_vocab_size", 10000),
        )

        tokenizer._vocabulary = state["vocabulary"]
        tokenizer._merge_rank_map = state["merge_rank_map"]
        tokenizer._token_to_id = state["token_to_id"]
        tokenizer._id_to_token = {
            int(k): v for k, v in state["id_to_token"].items()}
        tokenizer._next_id = state["next_id"]

        return tokenizer
