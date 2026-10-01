"""
Fetch the prose for notebooks/02_vocab_size.ipynb from FineWeb-Edu (Hugging
Face, `pip install datasets`): 2 MB of documents to train on, then the next
1 MB held out. Both files are committed, so this only needs to run once.

    python scripts/fetch_prose.py
"""

from pathlib import Path

from datasets import load_dataset

MB = 1_000_000
DATA = Path(__file__).resolve().parent.parent / "data"


def take(docs, n_bytes):
    out, size = [], 0
    for text in docs:
        out.append(text)
        size += len(text.encode("utf-8"))
        if size >= n_bytes:
            return "\n\n".join(out)


docs = (row["text"] for row in load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True))
(DATA / "prose_train.txt").write_text(take(docs, 2 * MB), encoding="utf-8")
(DATA / "prose_heldout.txt").write_text(take(docs, 1 * MB), encoding="utf-8")  # the iterator continues: disjoint
