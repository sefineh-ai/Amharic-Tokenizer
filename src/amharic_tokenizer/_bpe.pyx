# cython: language_level=3
"""Cython kernels for BPE over decomposed fidel.

This module holds only the two hot paths:

* :func:`learn_merges` - learn merges from word frequencies (training).
* :class:`BPEEncoder` - apply learned merges to text (inference).

Vocabulary bookkeeping, persistence and the public API live in pure Python
(see ``tokenizer.py``).
"""

import time
from heapq import heapify, heappop, heappush

from cython.operator cimport dereference as deref
from libc.limits cimport INT_MAX
from libc.stdint cimport int64_t
from libcpp.pair cimport pair as cpp_pair
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

from amharic_tokenizer.fidel import AMHARIC_FIDEL_MAP

EOW = "<eow>"
MAX_CACHED_WORDS = 200000


cdef list _word_symbols(str word):
    """Decomposed symbols of ``word`` followed by the end-of-word marker."""
    cdef list symbols = []
    cdef str ch
    for ch in word:
        symbols.extend(AMHARIC_FIDEL_MAP.get(ch, ch))
    symbols.append(EOW)
    return symbols


cdef dict _count_pairs(list tokens):
    cdef dict pairs = {}
    cdef Py_ssize_t i
    cdef tuple pair
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        pairs[pair] = pairs.get(pair, 0) + 1
    return pairs


def learn_merges(
    object word_freqs,
    object known_tokens,
    int num_merges,
    int max_vocab_size,
    object progress=None,
    int log_every=1000,
):
    """Learn BPE merges and return the new tokens as ``[(token, count), ...]`` in rank order.

    ``word_freqs`` maps each distinct word to its corpus frequency; its iteration
    order is the first-occurrence order and breaks ties between equal counts.
    ``known_tokens`` is the starting vocabulary: training stops once it plus the
    new tokens reaches ``max_vocab_size``. ``progress`` (optional) receives
    human-readable status lines.

    Each distinct word is stored once with its frequency, an index maps every
    pair to the words containing it, and a heap keeps the best pair, so each
    merge only touches the words that contain the merged pair.
    """
    cdef double t0 = time.time()
    cdef set known = set(known_tokens)
    cdef list learned = []
    cdef list words = []          # word id -> current token list
    cdef list freqs = []          # word id -> number of occurrences in the corpus
    cdef dict pair_counts = {}    # pair -> corpus count
    cdef dict pair_order = {}     # pair -> first-seen index, breaks ties between equal counts
    cdef dict pair_words = {}     # pair -> ids of words that contain (or once contained) it
    cdef list heap
    cdef dict changed, old_pairs
    cdef list token_list, new_list, affected
    cdef tuple best_pair, pair
    cdef str new_token
    cdef long long best_count, freq, count
    cdef Py_ssize_t i, j, n, wid

    for word, freq in word_freqs.items():
        wid = len(words)
        token_list = _word_symbols(word)
        words.append(token_list)
        freqs.append(freq)
        for pair, count in _count_pairs(token_list).items():
            if pair not in pair_counts:
                pair_order[pair] = len(pair_order)
                pair_counts[pair] = 0
                pair_words[pair] = set()
            pair_counts[pair] += count * freq
            pair_words[pair].add(wid)
    heap = [(-count, pair_order[pair], pair) for pair, count in pair_counts.items()]
    heapify(heap)
    if progress is not None:
        progress(f"Counted {sum(freqs)} words ({len(words)} distinct) and {len(pair_counts)} pairs "
                 f"in {time.time() - t0:.1f}s")

    for i in range(num_merges):
        if len(known) >= max_vocab_size:
            if progress is not None:
                progress(f"Stopping BPE training. Max vocabulary size ({max_vocab_size}) reached "
                         f"after {i} merges in {(time.time() - t0) / 60:.1f} minutes")
            break

        if progress is not None and (i == 0 or (i + 1) % log_every == 0 or i == num_merges - 1):
            progress(f"Merge {i + 1}/{num_merges} - vocab size {len(known)} - "
                     f"{time.time() - t0:.1f}s elapsed")

        if not pair_counts:
            break

        # Skip stale heap entries whose count has changed since they were pushed.
        while pair_counts[heap[0][2]] != -heap[0][0]:
            heappop(heap)
        best_count = -heap[0][0]
        best_pair = heap[0][2]
        if best_count < 2:
            break

        new_token = ''.join(best_pair)
        if new_token not in known:
            known.add(new_token)
            learned.append((new_token, best_count))

        # Words in first-occurrence order, so new pairs get a deterministic tie-break order.
        affected = sorted(pair_words[best_pair])
        changed = {}
        for wid in affected:
            token_list = words[wid]
            old_pairs = _count_pairs(token_list)
            if best_pair not in old_pairs:
                continue
            n = len(token_list)
            new_list = []
            j = 0
            while j < n:
                if j < n - 1 and token_list[j] == best_pair[0] and token_list[j + 1] == best_pair[1]:
                    new_list.append(new_token)
                    j += 2
                else:
                    new_list.append(token_list[j])
                    j += 1
            words[wid] = new_list
            freq = freqs[wid]
            for pair, count in old_pairs.items():
                pair_counts[pair] -= count * freq
                changed[pair] = None
            for pair, count in _count_pairs(new_list).items():
                if pair not in pair_counts:
                    pair_order[pair] = len(pair_order)
                    pair_counts[pair] = 0
                    pair_words[pair] = set()
                pair_counts[pair] += count * freq
                pair_words[pair].add(wid)
                changed[pair] = None
        for pair in changed:
            heappush(heap, (-pair_counts[pair], pair_order[pair], pair))

    if progress is not None:
        progress(f"Training completed: {len(learned)} merges in {(time.time() - t0) / 60:.1f} minutes")
    return learned


cdef class BPEEncoder:
    """Applies a fixed set of ranked merges to text.

    Merges are indexed by the integer ids of the two symbols they join:
    ``(left_id << 32 | right_id) -> (rank, merged_id)``, held in a C++ hash
    table so the inner merge loop runs without Python objects. Tokenized words
    are cached (up to ``MAX_CACHED_WORDS`` distinct words).
    """

    cdef unordered_map[int64_t, cpp_pair[int, int]] _pair_table
    cdef dict _symbol_ids   # symbol string -> id
    cdef list _symbols      # id -> symbol string
    cdef dict _char_ids     # fidel -> list of symbol ids of its decomposition
    cdef dict _word_cache
    cdef int _eow_id

    def __init__(self, object merge_ranks):
        """``merge_ranks`` maps each merged token to its rank (lower merges first)."""
        cdef str token, ch
        cdef Py_ssize_t k
        cdef object left, right
        cdef int merged_id
        self._symbol_ids = {}
        self._symbols = []
        self._char_ids = {}
        self._word_cache = {}
        self._eow_id = self._symbol_id(EOW)
        # A symbol is '<eow>', a single character, or a learned token, so every way
        # of splitting a learned token into two such symbols is a matching pair.
        for token in merge_ranks:
            self._symbol_id(token)
            for ch in token:
                self._symbol_id(ch)
        for token, rank in merge_ranks.items():
            merged_id = self._symbol_ids[token]
            for k in range(1, len(token)):
                left = self._symbol_ids.get(token[:k])
                right = self._symbol_ids.get(token[k:])
                if left is not None and right is not None:
                    self._pair_table[(<int64_t>left << 32) | <int64_t>right] = cpp_pair[int, int](rank, merged_id)

    cdef int _symbol_id(self, str symbol):
        cdef object sid = self._symbol_ids.get(symbol)
        if sid is None:
            sid = len(self._symbols)
            self._symbol_ids[symbol] = sid
            self._symbols.append(symbol)
        return sid

    cdef list _tokenize_word(self, str word):
        """Apply merges to one word, lowest rank first.

        Merges never cross word boundaries (each word ends with <eow>), so this
        gives the same result as merging over the whole text at once.
        """
        cdef vector[int] ids
        cdef list char_ids
        cdef str ch, part
        cdef int sid, left, right, merged_id, best_rank
        cdef Py_ssize_t i, w, n, best_i
        cdef unordered_map[int64_t, cpp_pair[int, int]].iterator found
        cdef unordered_map[int64_t, cpp_pair[int, int]].iterator not_found = self._pair_table.end()

        for ch in word:
            char_ids = self._char_ids.get(ch)
            if char_ids is None:
                char_ids = [self._symbol_id(part) for part in AMHARIC_FIDEL_MAP.get(ch, ch)]
                self._char_ids[ch] = char_ids
            for sid in char_ids:
                ids.push_back(sid)
        ids.push_back(self._eow_id)

        while True:
            n = ids.size()
            best_i = -1
            best_rank = INT_MAX
            merged_id = -1
            for i in range(n - 1):
                found = self._pair_table.find((<int64_t>ids[i] << 32) | <int64_t>ids[i + 1])
                if found != not_found and deref(found).second.first < best_rank:
                    best_rank = deref(found).second.first
                    merged_id = deref(found).second.second
                    best_i = i
            if best_i < 0:
                break
            left = ids[best_i]
            right = ids[best_i + 1]
            w = 0
            i = 0
            while i < n:
                if i < n - 1 and ids[i] == left and ids[i + 1] == right:
                    ids[w] = merged_id
                    i += 2
                else:
                    ids[w] = ids[i]
                    i += 1
                w += 1
            ids.resize(w)
        return [self._symbols[ids[i]] for i in range(<Py_ssize_t>ids.size())]

    cpdef list tokenize(self, str text):
        """Split ``text`` on whitespace and return the merged tokens of every word."""
        cdef list tokens = []
        cdef list word_tokens
        cdef dict cache = self._word_cache
        cdef str word
        for word in text.split():
            word_tokens = cache.get(word)
            if word_tokens is None:
                word_tokens = self._tokenize_word(word)
                if len(cache) >= MAX_CACHED_WORDS:
                    cache.clear()
                cache[word] = word_tokens
            tokens.extend(word_tokens)
        return tokens
