"""
Fetch the two files GPT-2 shipped its tokenizer as, into assets/gpt2/.

  vocab.bpe     50,000 merges, one per line, in learned order
  encoder.json  token string -> id (50,257 entries: 256 bytes + 50,000 merges + <|endoftext|>)

Every GPT-2 size shares the same tokenizer; the 1558M path is just the one the
lecture uses. Skips files already present. Run from anywhere:

    python scripts/download_gpt2.py
"""

import urllib.request
from pathlib import Path

BASE = "https://openaipublic.blob.core.windows.net/gpt-2/models/1558M/"
FILES = ("vocab.bpe", "encoder.json")
DEFAULT_DEST = Path(__file__).resolve().parent.parent / "assets" / "gpt2"


def download(dest=DEFAULT_DEST):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = dest / name
        if path.exists():
            print(f"{path} already present")
            continue
        print(f"downloading {name} -> {path}")
        urllib.request.urlretrieve(BASE + name, path)
    return dest


if __name__ == "__main__":
    download()
