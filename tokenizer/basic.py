"""
The tokenizer exactly as the lecture builds it: byte-level BPE with no
pre-splitting and no special tokens. `notebooks/01_bpe_walkthrough.ipynb`
derives every line of this file; the class is the same code plus save/load.
"""

from .base import Tokenizer, get_stats, merge


class BasicTokenizer(Tokenizer):

    def train(self, text, vocab_size, verbose=False):
        """
        Learn `vocab_size - 256` merges from `text`.

        No gradients, no GPU: a greedy, deterministic counting procedure. Each
        iteration recounts pairs from scratch, because the previous merge created
        adjacencies that did not exist before.
        """
        assert vocab_size >= 256, "the 256 byte tokens are always present"
        num_merges = vocab_size - 256

        ids = list(text.encode("utf-8"))
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:  # fewer than two ids left; nothing to merge
                break
            pair = max(stats, key=stats.get)  # most frequent; ties -> first seen
            idx = 256 + i
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merge {i + 1}/{num_merges}: {pair} -> {idx} "
                      f"({vocab[idx]!r}) had {stats[pair]} occurrences")

        self.merges = merges
        self.vocab = vocab

    def encode(self, text):
        """
        str -> ids, by replaying the merges in *learned* order.

        Not most-frequent-first, not longest-match-first: earliest-learned present
        pair first, one merge at a time, rescan. Any other rule can produce a
        token sequence the model was never trained on.
        """
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:  # fewer than two ids -> no pairs; encode("") == []
            stats = get_stats(ids)  # only the keys matter: which pairs exist
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # no present pair was ever learned; done
            ids = merge(ids, pair, self.merges[pair])
        return ids
