"""
Invariants the rest of the repo rests on. Every analysis notebook assumes these;
if one fails, no downstream number can be trusted.
"""

import random

import pytest

from tokenizer import BasicTokenizer, get_stats, merge


# --- the two primitives ------------------------------------------------------

def test_get_stats_counts_adjacent_pairs():
    assert get_stats([1, 2, 3, 1, 2]) == {(1, 2): 2, (2, 3): 1, (3, 1): 1}


def test_get_stats_accumulates_into_existing_counts():
    counts = get_stats([1, 2])
    get_stats([1, 2, 2], counts)
    assert counts == {(1, 2): 2, (2, 2): 1}


def test_get_stats_short_inputs():
    assert get_stats([]) == {}
    assert get_stats([7]) == {}


def test_merge_replaces_only_the_matching_pair():
    assert merge([5, 6, 6, 7, 9, 1], (6, 7), 99) == [5, 6, 99, 9, 1]


def test_merge_is_greedy_left_to_right_and_non_overlapping():
    assert merge([1, 1, 1], (1, 1), 256) == [256, 1]
    assert merge([1, 1, 1, 1], (1, 1), 256) == [256, 256]


def test_merge_does_not_mutate_input():
    ids = [1, 2, 1, 2]
    merge(ids, (1, 2), 256)
    assert ids == [1, 2, 1, 2]


# --- training ----------------------------------------------------------------

TEXT = (
    "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 😄 The very name strikes fear and awe into the hearts "
    "of programmers worldwide. We all know we ought to “support Unicode” in our "
    "software (whatever that means—like using wchar_t for all the strings, right?). "
    "But Unicode can be abstruse, and diving into the thousand-page Unicode Standard "
    "plus its dozens of supplementary annexes, reports, and notes can be more than a "
    "little intimidating."
)


@pytest.fixture(scope="module")
def tok():
    t = BasicTokenizer()
    t.train(TEXT, vocab_size=300)
    return t


def test_train_learns_the_requested_number_of_merges(tok):
    assert len(tok.merges) == 300 - 256
    assert list(tok.merges.values()) == list(range(256, 300))  # ids handed out in order
    assert len(tok.vocab) == 300


def test_merge_children_always_precede_the_merge(tok):
    for (p0, p1), idx in tok.merges.items():
        assert p0 < idx and p1 < idx


def test_vocab_bytes_are_children_concatenated(tok):
    for (p0, p1), idx in tok.merges.items():
        assert tok.vocab[idx] == tok.vocab[p0] + tok.vocab[p1]


def test_train_on_text_too_short_to_fill_vocab_stops_cleanly():
    t = BasicTokenizer()
    t.train("ab", vocab_size=300)  # one pair, then nothing left to merge
    assert t.merges == {(97, 98): 256}
    assert t.decode(t.encode("ab")) == "ab"


# --- encode / decode ---------------------------------------------------------

def test_encode_empty_and_single_byte(tok):
    assert tok.encode("") == []
    assert tok.encode("a") == [97]


def test_roundtrip_on_training_text(tok):
    assert tok.decode(tok.encode(TEXT)) == TEXT


def test_roundtrip_on_held_out_text(tok):
    held_out = "Byte-level fallback means coverage is total — 안녕하세요 👋, no OOV, ever."
    assert tok.decode(tok.encode(held_out)) == held_out


def test_roundtrip_on_random_unicode(tok):
    rng = random.Random(1337)
    for _ in range(50):
        chars = []
        for _ in range(rng.randint(0, 60)):
            cp = rng.randint(0, 0x10FFFF)
            if 0xD800 <= cp <= 0xDFFF:  # surrogates are not encodable as UTF-8
                continue
            chars.append(chr(cp))
        s = "".join(chars)
        assert tok.decode(tok.encode(s)) == s


def test_encoding_compresses_the_training_text(tok):
    assert len(tok.encode(TEXT)) < len(TEXT.encode("utf-8"))


def test_decode_replaces_invalid_utf8_instead_of_raising(tok):
    assert tok.decode([128]) == "\ufffd"  # a lone continuation byte


def test_encode_applies_merges_in_learned_order_not_frequency_order():
    # Two merges that compete for the middle byte of "abc". Whichever was learned
    # first must win, regardless of anything about the input string.
    t = BasicTokenizer()
    t.merges = {(97, 98): 256, (98, 99): 257}  # "ab" learned before "bc"
    t.vocab = t._build_vocab()
    assert t.encode("abc") == [256, 99]

    t.merges = {(98, 99): 256, (97, 98): 257}  # "bc" learned before "ab"
    t.vocab = t._build_vocab()
    assert t.encode("abc") == [97, 256]


def test_encode_reproduces_training_time_ids(tok):
    # Replaying the merges on the training text must land on exactly the ids the
    # training loop ended with. If it doesn't, encode and train disagree about
    # the algorithm and the model would see sequences it never trained on.
    ids = list(TEXT.encode("utf-8"))
    for pair, idx in tok.merges.items():
        ids = merge(ids, pair, idx)
    assert tok.encode(TEXT) == ids


# --- save / load -------------------------------------------------------------

def test_save_load_reproduces_identical_ids(tok, tmp_path):
    prefix = str(tmp_path / "tok300")
    tok.save(prefix)

    loaded = BasicTokenizer()
    loaded.load(prefix + ".model")
    assert loaded.merges == tok.merges
    assert loaded.vocab == tok.vocab
    assert loaded.encode(TEXT) == tok.encode(TEXT)
    assert (tmp_path / "tok300.vocab").exists()
