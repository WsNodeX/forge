"""End-to-end demo of the bundled English tokenizer.

Run from the repository root::

    python examples/demo.py
"""

import os
import sys

# Run from a fresh clone without `pip install -e .`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows consoles default to GBK under a Chinese locale, which cannot encode the
# emoji and the byte-level mapping characters this demo prints. Reconfigure
# rather than crash — a demo that dies on its own output is worse than useless.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
from forge import BPETokenizer, load


def main() -> None:
    tok = BPETokenizer(load("english"))

    samples = [
        "Tokenization is the silent hero of every language model.",
        "The historians studied the ancient libraries carefully.",
        "Hello world! 你好，世界 🌍 — byte-level fallback works.",
    ]

    for text in samples:
        ids = tok.encode(text)
        print("=" * 64)
        print("TEXT :", text)
        print("TOKENS:", len(ids))
        print("IDS   :", ids)
        print("DECODE:", tok.decode(ids))
        print(f"COMPRESSION: {tok.compression_ratio(text):.3f} tokens/char")

    # Special tokens are preserved verbatim through encode/decode.
    special_text = "begin <|endoftext|> end"
    print("=" * 64)
    print("SPECIAL:", special_text, "->", tok.encode(special_text))


if __name__ == "__main__":
    main()
