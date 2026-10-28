"""End-to-end demo of the bundled English tokenizer.

Run from the repository root::

    python examples/demo.py
"""

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
