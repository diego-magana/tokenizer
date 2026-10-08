# tokenizer

![CI](https://github.com/diego-magana/tokenizer/actions/workflows/ci.yml/badge.svg)

A byte-level BPE tokenizer built from scratch — trainer, encoder, decoder — and one experiment on what vocabulary size actually buys.

Fourth in the series **[micrograd](https://github.com/diego-magana/micrograd) → [makemore](https://github.com/diego-magana/makemore) → [gpt](https://github.com/diego-magana/gpt) → tokenizer**. gpt tokenized Tiny Shakespeare one character at a time; this is the tokenizer a real GPT uses.

If you want the short version, read [`notebooks/02_vocab_size.ipynb`](notebooks/02_vocab_size.ipynb): the question, the prediction, the figure.

## What I found

A model is billed, bounded, and slowed per token, so bytes per token is a price, and vocabulary size is the one dial a tokenizer has for it. This is a look at what each doubling of that dial buys, on one tokenizer and one corpus.

One training run covers the whole sweep. BPE learns merges greedily and in order, so the first *k* merges of a 16k run are exactly the tokenizer you'd get by training to *k*: train once, truncate to every smaller vocabulary. I trained on 2 MB of FineWeb-Edu prose, measured from 512 to 16k on the training text and on 1 MB of held-out prose, and put GPT-2 and cl100k on the same held-out text for reference.

| vocab | train | held-out | gap | tokens per MB, held-out |
|---|---|---|---|---|
| 512 | 2.01 | 2.01 | 0.0% | 497k |
| 1,024 | 2.51 | 2.51 | 0.1% | 399k |
| 2,048 | 2.97 | 2.95 | 0.9% | 339k |
| 4,096 | 3.47 | 3.40 | 2.0% | 294k |
| 8,192 | 3.98 | 3.83 | 3.8% | 261k |
| 16,384 | 4.44 | 4.20 | 5.8% | 238k |
| GPT-2 (50,257) | — | 4.66 | — | 215k |
| cl100k (100,277) | — | 4.78 | — | 209k |

[![Compression vs vocabulary size](assets/vocab_size.png)](notebooks/02_vocab_size.ipynb)

**Each doubling adds about half a byte per token.** On the training text the gain is 0.46–0.51 at every step from 512 to 16k, so compression is linear in log vocabulary — the straight line in the figure. What you pay for is the reciprocal, so the same half-byte saves fewer tokens each time: 98k per megabyte for the first doubling, then 60k, 45k, 33k, and 23k.

**The held-out line falls behind as the vocabulary grows.** It sits on the training line at 512 — the two samples agree at the byte-pair level — and trails it by 0.9% at 2k, 3.8% at 8k, and 5.8% at 16k. The last doubling adds 0.46 on the training text and 0.36 held out: a growing share of the new merges are pairs frequent in these 2 MB and nowhere else, which is the ordinary reason tokenizers are trained on far more text than this. GPT-2, with 50k entries and a corpus 20,000× larger, reaches 4.66 on this held-out text, about where the held-out line points; cl100k adds 2.6% with twice the vocabulary again, most of it spent on code and other languages.

**What this doesn't show.** Bytes per token is compression, not quality: whether a larger vocabulary helps a model predict is a training experiment. Two megabytes and one held-out sample make the trend visible but not precise — the gap is one sample's number. The references were trained on other corpora, cl100k with a different pre-split regex too, so they show where production tokenizers land on this text, not what this trainer would reach at their size.

## What it builds

Text becomes UTF-8 bytes, so the starting vocabulary is 256 tokens and every string is representable. Training is a counting loop — find the most frequent adjacent pair, replace it everywhere with a new id, repeat — and encoding replays those merges in the order they were learned. `BasicTokenizer` is that algorithm exactly as the lecture builds it; `RegexTokenizer` adds GPT-2's regex pre-split, so merges never cross a word boundary. [`notebooks/01_bpe_walkthrough.ipynb`](notebooks/01_bpe_walkthrough.ipynb) derives all of it from one paragraph of text and ends by checking that the package learns byte-for-byte the same merges.

## Run it

```bash
pip install -e .                    # the package needs only `regex`
pip install -r requirements.txt     # tests + notebooks
pytest                              # 26 invariants
jupyter notebook notebooks/02_vocab_size.ipynb
```

The corpus and the trained tokenizer are committed, so the notebook reproduces the figure from `assets/prose_16k.model` in under a minute. Deleting that file retrains it (about seven minutes); `scripts/fetch_prose.py` refetches the corpus.

```python
from tokenizer import RegexTokenizer

tok = RegexTokenizer()
tok.load("assets/prose_16k.model")
ids = tok.encode("Unicode! 🅤🅝🅘🅒🅞🅓🅔‽")
assert tok.decode(ids) == "Unicode! 🅤🅝🅘🅒🅞🅓🅔‽"
tok.truncated(4096)                 # the same tokenizer at a 4k vocabulary
```

## Repository layout

```
tokenizer/
├── tokenizer/
│   ├── base.py         get_stats, merge, the Tokenizer base class, save/load, truncated
│   ├── basic.py        BasicTokenizer — the lecture's algorithm as a class
│   └── regex.py        RegexTokenizer — GPT-2's pre-split, weighted training
├── notebooks/
│   ├── 01_bpe_walkthrough.ipynb   the lecture, cell by cell, ending at the package
│   └── 02_vocab_size.ipynb        the experiment
├── assets/             prose_16k.model (loadable) + prose_16k.vocab (readable) · vocab_size.png
├── data/               the walkthrough's text · prose_train.txt (2 MB) · prose_heldout.txt (1 MB)
├── scripts/            download_gpt2.py · fetch_prose.py
└── tests/              test_basic.py · test_regex.py
```

## Implementation notes

**Encode applies merges in learned order, not frequency order.** The most frequent pair in the *input* is irrelevant; the earliest-learned pair *present* wins, one merge at a time. Any other rule can produce a token sequence the model was never trained on. `test_basic.py` pins this with a case where the orders disagree; `test_regex.py` checks that encoding the training text lands on exactly the ids training ended with.

**Training counts over distinct chunks, weighted by frequency.** Two megabytes is 402k regex chunks but only 36k distinct ones; counting each once with its multiplicity gives the same statistics for a tenth of the work. Ties break on the pair itself, so the result depends only on the corpus — which is what makes a 16k run truncate to exactly the 4k run, and what `test_regex.py` verifies against a plain unweighted count.

## Attribution

The algorithm, the walkthrough, and the split pattern follow Andrej Karpathy's [*Let's build the GPT Tokenizer*](https://www.youtube.com/watch?v=zduSFxRajkE); the class layout follows his [minbpe](https://github.com/karpathy/minbpe). What I added: the tests, save/load in the shape of GPT-2's `vocab.bpe`, `truncated`, and the vocabulary-size experiment with its prediction and held-out measurement.
