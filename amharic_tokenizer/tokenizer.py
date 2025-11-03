import json
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import Counter
from amharic_tokenizer.fidel_map import AMHARIC_FIDEL_MAP, REVERSE_FIDEL_MAP
class AmharicTokenizer:
    """
    Optimized BPE Tokenizer for Amharic Fidel. Retains only necessary attributes 
    and methods for production use.
    """

    def __init__(self, num_merges: int = 50000) -> None:
        """
        Initializes the tokenizer state and base vocabulary.

        :param num_merges: The maximum number of merge operations to perform.
        """
        self.__vocabulary: Dict[str, int] = {}
        self.__merge_rank_map: Dict[str, int] = {}
        self.__num_merges: int = num_merges
        
        self.__token_to_id: Dict[str, int] = {}
        self.__id_to_token: Dict[int, str] = {}
        self.__next_id: int = 0

        self.__initialize_base_vocabulary()


    def __initialize_base_vocabulary(self) -> None:
        """Populates the initial vocabulary and ID maps using base C/V units."""
        initial_tokens: Set[str] = set()
        for char_list in AMHARIC_FIDEL_MAP.values():
            for char in char_list:
                initial_tokens.add(char)
        initial_tokens.add('<eow>')
        
        sorted_initial_tokens: List[str] = sorted(list(initial_tokens))
        for token in sorted_initial_tokens:
            self.__vocabulary[token] = 0
            self.__add_to_vocab_maps(token)

    def __add_to_vocab_maps(self, token: str) -> None:
        """Adds a new token to the ID maps and increments the counter."""
        if token not in self.__token_to_id:
            self.__token_to_id[token] = self.__next_id
            self.__id_to_token[self.__next_id] = token
            self.__next_id += 1

    @staticmethod
    def __get_pairs(tokens: List[str]) -> Counter[Tuple[str, str]]:
        """Efficiently counts adjacent pairs in a list of tokens."""
        return Counter(zip(tokens[:-1], tokens[1:]))

    # --- Preprocessing ---

    def preprocess(self, amharic_corpus: str) -> List[List[str]]:
        """Decomposes Amharic text into base C/V units and adds <eow> word markers."""
        words: List[str] = amharic_corpus.split()
        preprocessed_corpus: List[List[str]] = []
        for word in words:
            mapped_word: List[str] = []
            for char in word:
                # AMHARIC_FIDEL_MAP values are strings
                mapped_word.extend(list(AMHARIC_FIDEL_MAP.get(char, [char])))
            mapped_word.append('<eow>') 
            preprocessed_corpus.append(mapped_word)
        return preprocessed_corpus
    
    # --- Training ---

    def train(self, amharic_corpus: str, verbose: bool = False, log_every: int = 1000) -> int:
        """
        Trains the BPE model by iteratively merging the most frequent token pair.
        The implementation uses an efficient pair count update strategy.
        """
        tokenized_words: List[List[str]] = self.preprocess(amharic_corpus)
        
        pair_counts: Counter[Tuple[str, str]] = Counter()
        for word_tokens in tokenized_words:
            pair_counts.update(self.__get_pairs(word_tokens))
        
        for i in range(self.__num_merges):
            if verbose and (i + 1) % log_every == 0:
                print(f"Merge {i + 1}/{self.__num_merges} completed.")

            if not pair_counts: break
            
            best_pair: Tuple[str, str] = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] < 2: break

            new_token: str = ''.join(best_pair)
            
            # Update maps
            self.__merge_rank_map[new_token] = len(self.__merge_rank_map) + 1
            self.__vocabulary[new_token] = pair_counts[best_pair]
            self.__add_to_vocab_maps(new_token)
            
            # Efficiently update corpus and pair counts
            new_tokenized_words: List[List[str]] = []
            new_pair_counts: Counter[Tuple[str, str]] = pair_counts.copy()
            
            for token_list in tokenized_words:
                
                # We need to re-scan words that contained the merged pair
                if new_token in ''.join(token_list): 
                    old_pairs = self.__get_pairs(token_list)
                    
                    # Apply merge
                    new_list: List[str] = []
                    j: int = 0
                    while j < len(token_list):
                        if j < len(token_list) - 1 and (token_list[j], token_list[j+1]) == best_pair:
                            new_list.append(new_token)
                            j += 2
                        else:
                            new_list.append(token_list[j])
                            j += 1
                            
                    new_tokenized_words.append(new_list)

                    # Update pair counts
                    new_pair_counts.subtract(old_pairs)
                    new_pair_counts.update(self.__get_pairs(new_list))
                    
                else:
                    new_tokenized_words.append(token_list)
            
            tokenized_words = new_tokenized_words
            pair_counts = new_pair_counts

        return len(self.__merge_rank_map)

    # --- tokenize ---
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes text by applying learned merges sequentially based on rank.
        """
        corpus: List[List[str]] = self.preprocess(text)
        
        # Reverse map created locally for fast rank lookup during tokenize
        reversed_merge_map: Dict[str, int] = self.__merge_rank_map

        def get_best_merge(current_corpus: List[List[str]]) -> Optional[Tuple[str, str]]:
            """Finds the highest-priority (lowest rank) merge applicable in the current corpus."""
            best_pair: Optional[Tuple[str, str]] = None
            highest_priority_rank: float = float('inf') 
            
            for token_list in current_corpus:
                for j in range(len(token_list) - 1):
                    pair_str: str = token_list[j] + token_list[j + 1]
                    
                    rank = reversed_merge_map.get(pair_str)
                    
                    if rank is not None and rank < highest_priority_rank:
                        highest_priority_rank = rank
                        best_pair = (token_list[j], token_list[j + 1])
                        
            return best_pair
        
        while True:
            best_pair = get_best_merge(corpus)
            if best_pair is None: break
            pair_str: str = ''.join(best_pair)
            updated_corpus: List[List[str]] = []
            
            for token_list in corpus:
                new_token_list: List[str] = []
                i: int = 0
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

    # --- Encoding and Decoding ---
    
    def encode(self, text: str) -> List[int]:
        """Converts the input text into a sequence of integer IDs."""
        tokens: List[str] = self.tokenize(text)
        return [self.__token_to_id.get(token, -1) for token in tokens]

    def decode(self, token_ids: List[int]) -> str:
        """Converts a sequence of integer IDs back into a detokenized string."""
        tokens: List[str] = [self.__id_to_token.get(id, '<unk>') for id in token_ids]
        return self.detokenize(tokens)
            
    def detokenize(self, tokens: List[str]) -> str:
        """Reconstructs Amharic Fidel characters from BPE tokens and C/V units."""
        temp_string: str = "".join(tokens).replace("<eow>", " ")
        word_segments: List[str] = temp_string.split()
        
        final_text_words: List[str] = []
        MAX_CV_LENGTH: int = 3
        
        for word_string in word_segments:
            chars_list: List[str] = list(word_string) 
            reconstructed_word: List[str] = []
            i: int = 0
            
            while i < len(chars_list):
                found_match: bool = False
                
                for length in range(MAX_CV_LENGTH, 0, -1): 
                    sub_string: str = "".join(chars_list[i : i + length])
                    
                    if sub_string in REVERSE_FIDEL_MAP:
                        reconstructed_word.append(REVERSE_FIDEL_MAP[sub_string])
                        i += length
                        found_match = True
                        break
                        
                if not found_match:
                    reconstructed_word.append(chars_list[i])
                    i += 1
                    
            final_text_words.append("".join(reconstructed_word))
            
        return " ".join(final_text_words)

    # --- Serialization/Deserialization ---
    
    def save(self, file_path: str) -> None:
        """Saves the trained tokenizer state to a JSON file."""
        state: Dict[str, Any] = {
            'num_merges': self.__num_merges,
            'vocabulary': self.__vocabulary,
            'merge_rank_map': self.__merge_rank_map,
            'token_to_id': self.__token_to_id,
            'id_to_token': {str(k): v for k, v in self.__id_to_token.items()}, 
            'next_id': self.__next_id
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=4)
        print(f"Tokenizer state saved to {file_path}")

    @classmethod
    def load_tokenizer(cls, file_path: str) -> 'AmharicTokenizer':
        """Loads a tokenizer state from a JSON file and creates a new instance."""
        with open(file_path, 'r', encoding='utf-8') as f:
            state: Dict[str, Any] = json.load(f)
        
        # Initialize with configuration values
        tokenizer: 'AmharicTokenizer' = cls(
            num_merges=state.get('num_merges', 50000)
        )
        
        # Overwrite private attributes
        tokenizer.__vocabulary = state['vocabulary']
        tokenizer.__merge_rank_map = state['merge_rank_map']
        tokenizer.__token_to_id = state['token_to_id']
        tokenizer.__id_to_token = {int(k): v for k, v in state['id_to_token'].items()}
        tokenizer.__next_id = state['next_id']
        
        print(f"Tokenizer state loaded from {file_path}")
        return tokenizer
