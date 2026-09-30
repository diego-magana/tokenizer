"""
The tokenizer GPT-2 actually uses: byte-level BPE inside regex-split chunks.

One thing changes relative to `basic.py`: text is pre-split with a regex so
merges never cross a chunk boundary — "dog." and "dog!" share the token "dog"
instead of learning two. Training then counts pair statistics over the
*distinct* chunks weighted by how often each occurs, which is the same numbers
as counting over the whole corpus at a fraction of the work. Encoding is
unchanged: replay the merges in learned order, one chunk at a time.
"""

from collections import Counter

import regex as re

from .base import Tokenizer, get_stats, merge

# GPT-2's pattern, from openai/gpt-2 src/encoder.py: contractions, then letters,
# then numbers, then punctuation, each with an optional leading space, then runs
# of whitespace that leave their last space to attach to the next word.
GPT2_SPLIT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class RegexTokenizer(Tokenizer):

    def __init__(self, pattern=GPT2_SPLIT_PATTERN):
        super().__init__()
        self.pattern = pattern
        self.compiled_pattern = re.compile(pattern)
        self._cache = {}  # chunk -> ids; encoding is chunk-local, so it memoizes

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256, "the 256 byte tokens are always present"
        num_merges = vocab_size - 256

        # A 2 MB corpus is ~400k chunks but only ~40k distinct ones. Keep one copy
        # of each with its count; every pass below is over the distinct ones.
        chunk_counts = Counter(self.compiled_pattern.findall(text))
        words = [(list(chunk.encode("utf-8")), c) for chunk, c in chunk_counts.items()]

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = {}
            for word, c in words:
                for pair in zip(word, word[1:]):
                    stats[pair] = stats.get(pair, 0) + c
            if not stats:
                break
            # Most frequent pair; ties broken by the pair itself, so the result
            # depends only on the corpus. That is what makes training to 16k and
            # keeping the first k merges identical to training to k.
            pair = max(stats, key=lambda p: (stats[p], p))
            idx = 256 + i
            words = [(merge(w, pair, idx) if pair[0] in w else w, c) for w, c in words]
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose and (i + 1) % 1000 == 0:
                print(f"merge {i + 1}/{num_merges}: {pair} -> {idx} ({vocab[idx]!r})")

        self.merges = merges
        self.vocab = vocab
        self._cache = {}

    def _encode_chunk(self, ids):
        # Identical to BasicTokenizer.encode: earliest-learned present pair first.
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text):
        ids = []
        for chunk in self.compiled_pattern.findall(text):
            if chunk not in self._cache:
                self._cache[chunk] = self._encode_chunk(list(chunk.encode("utf-8")))
            ids.extend(self._cache[chunk])
        return ids

    def load(self, model_file):
        super().load(model_file)
        self.compiled_pattern = re.compile(self.pattern)
        self._cache = {}
