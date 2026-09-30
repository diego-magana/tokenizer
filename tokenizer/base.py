"""
The two functions the algorithm is made of, and the base class every tokenizer
in this package shares.

`get_stats` is the "statistics" half of BPE: count adjacent pairs. `merge` is
the "rewrite" half: replace one pair everywhere it occurs. Training is a loop
that alternates them; encoding replays the learned merges in learned order.
"""


def get_stats(ids, counts=None):
    """
    Count how often each adjacent pair appears in `ids`.

    Pass an existing `counts` dict to accumulate across several sequences —
    the regex tokenizer counts across chunks this way. Insertion order is
    first-occurrence order, which is what breaks ties in `max(stats, key=...)`:
    arbitrary but deterministic.
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """
    Replace every non-overlapping occurrence of `pair` in `ids` with `idx`.

    Left-to-right and greedy: for ids=[1, 1, 1] and pair=(1, 1) the result is
    [idx, 1], not [1, idx]. Training and encoding must make the same choice or
    they stop agreeing. Returns a new list; `ids` is not mutated.
    """
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2  # consume both — this is what makes matches non-overlapping
        else:
            newids.append(ids[i])
            i += 1
    return newids


def render_token(token_bytes):
    """Human-readable form of a token's bytes for .vocab files: invalid UTF-8
    becomes U+FFFD and whitespace/control characters are escaped, so one token
    is always one line."""
    s = token_bytes.decode("utf-8", errors="replace")
    return "".join(f"\\u{ord(ch):04x}" if ch.isspace() or ord(ch) < 32 else ch for ch in s)


class Tokenizer:
    """
    Base class. A trained tokenizer is three things:

      merges          (int, int) -> int, in learned order (dict insertion order is
                      load-bearing — `encode` and `_build_vocab` both rely on it)
      pattern         regex used to pre-split text, or "" for none
      special_tokens  str -> int, ids appended after the BPE vocabulary

    Everything else — `vocab` (id -> bytes), decode, save/load — derives from those.
    """

    def __init__(self):
        self.merges = {}
        self.pattern = ""
        self.special_tokens = {}
        self.vocab = self._build_vocab()

    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError

    def encode(self, text):
        raise NotImplementedError

    def decode(self, ids):
        """ids -> str. Any id list decodes; invalid UTF-8 becomes U+FFFD rather
        than raising, because a model can emit a lone continuation byte."""
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        return text_bytes.decode("utf-8", errors="replace")

    def truncated(self, vocab_size):
        """
        The same tokenizer with only its first `vocab_size - 256` merges.

        Merges are learned greedily and in order, so the first k merges of a
        tokenizer trained to 16k are exactly the tokenizer trained to k on the
        same corpus. One training run therefore yields every smaller vocabulary.
        """
        assert 256 <= vocab_size <= len(self.vocab)
        other = self.__class__.__new__(self.__class__)
        other.__dict__.update(self.__dict__)
        other.merges = dict(list(self.merges.items())[: vocab_size - 256])
        other.special_tokens = {}
        other.vocab = other._build_vocab()
        other._cache = {}
        return other

    def _build_vocab(self):
        """The decoding table is derived, not stored: 256 byte tokens, then every
        merge replayed in order. A merge's children always have smaller ids than
        the merge itself, so they are built by the time they are needed."""
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, file_prefix):
        """
        Write `<prefix>.model` (what `load` reads) and `<prefix>.vocab` (for
        humans; never read back). The model file has the same shape as GPT-2's
        vocab.bpe: one merge per line, in learned order, ids implied by position.
        """
        with open(file_prefix + ".model", "w", encoding="utf-8") as f:
            f.write("bpe v1\n")
            f.write(f"{self.pattern}\n")
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            for p0, p1 in self.merges:
                f.write(f"{p0} {p1}\n")

        inverted = {idx: pair for pair, idx in self.merges.items()}
        with open(file_prefix + ".vocab", "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                s = render_token(token)
                if idx in inverted:
                    p0, p1 = inverted[idx]
                    f.write(f"[{render_token(self.vocab[p0])}][{render_token(self.vocab[p1])}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        """Inverse of `save`. Rebuilds `vocab` from the merges."""
        assert model_file.endswith(".model")
        merges, special_tokens = {}, {}
        idx = 256
        with open(model_file, encoding="utf-8") as f:
            assert f.readline().strip() == "bpe v1"
            self.pattern = f.readline().strip()
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            for line in f:
                p0, p1 = map(int, line.split())
                merges[(p0, p1)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()
