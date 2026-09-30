"""
RegexTokenizer invariants: the weighted count equals the plain one, merges never
cross a chunk boundary, encode reproduces training, and a longer training run
truncates to exactly a shorter one — the property the vocabulary sweep rests on.
"""

from pathlib import Path

import pytest
import regex as re

from tokenizer import GPT2_SPLIT_PATTERN, RegexTokenizer, get_stats, merge

TEXT = (Path(__file__).parent.parent / "data" / "programmers_intro_to_unicode.txt").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tok():
    t = RegexTokenizer()
    t.train(TEXT, vocab_size=400)
    return t


def plain_train(text, vocab_size):
    """Reference: one entry per chunk occurrence, no weighting. Same tie-break."""
    words = [list(c.encode("utf-8")) for c in re.findall(GPT2_SPLIT_PATTERN, text)]
    merges = {}
    for i in range(vocab_size - 256):
        stats = {}
        for w in words:
            get_stats(w, stats)
        pair = max(stats, key=lambda p: (stats[p], p))
        words = [merge(w, pair, 256 + i) for w in words]
        merges[pair] = 256 + i
    return merges, words


def test_weighted_count_matches_plain_count(tok):
    merges, _ = plain_train(TEXT, 400)
    assert tok.merges == merges


def test_encode_reproduces_training_time_ids(tok):
    _, words = plain_train(TEXT, 400)
    assert tok.encode(TEXT) == [idx for w in words for idx in w]


def test_longer_run_truncates_to_shorter_run(tok):
    short = RegexTokenizer()
    short.train(TEXT, vocab_size=300)
    assert tok.truncated(300).merges == short.merges
    assert tok.truncated(300).encode(TEXT[:3000]) == short.encode(TEXT[:3000])


def test_no_token_crosses_a_chunk_boundary(tok):
    chunks = {c.encode("utf-8") for c in re.findall(GPT2_SPLIT_PATTERN, TEXT)}
    for idx in range(256, len(tok.vocab)):
        assert any(tok.vocab[idx] in chunk for chunk in chunks)


def test_space_attaches_to_the_following_word():
    assert re.findall(GPT2_SPLIT_PATTERN, "hello world") == ["hello", " world"]


def test_roundtrip_on_held_out_text(tok):
    held_out = "for i in range(1, 101):\n    if i % 15 == 0: print('FizzBuzz')  # 안녕 👋"
    assert tok.decode(tok.encode(held_out)) == held_out
    assert tok.encode("") == []


def test_save_load_reproduces_identical_ids(tok, tmp_path):
    tok.save(str(tmp_path / "t"))
    loaded = RegexTokenizer()
    loaded.load(str(tmp_path / "t.model"))
    assert loaded.pattern == GPT2_SPLIT_PATTERN
    assert loaded.encode(TEXT[:3000]) == tok.encode(TEXT[:3000])
