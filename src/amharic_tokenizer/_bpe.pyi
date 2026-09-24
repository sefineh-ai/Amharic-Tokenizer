from typing import Callable, Iterable, List, Mapping, Optional, Tuple

EOW: str
MAX_CACHED_WORDS: int

def learn_merges(
    word_freqs: Mapping[str, int],
    known_tokens: Iterable[str],
    num_merges: int,
    max_vocab_size: int,
    progress: Optional[Callable[[str], None]] = ...,
    log_every: int = ...,
) -> List[Tuple[str, int]]: ...

class BPEEncoder:
    def __init__(self, merge_ranks: Mapping[str, int]) -> None: ...
    def tokenize(self, text: str) -> List[str]: ...
