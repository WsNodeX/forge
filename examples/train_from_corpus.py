"""Train a brand-new tokenizer on your own corpus.

This shows the full lifecycle: collect text -> train -> persist -> reload ->
use.  No external services, no GPU, no dependencies.

Run from the repository root::

    python examples/train_from_corpus.py
"""

import os

from forge import BPETokenizer, BPETrainer, Vocabulary

CORPUS = """
the cat sat on the mat and the dog ran across the field while the sun set slowly
behind the hills and the river flowed quietly through the valley toward the sea
a small model can still learn useful subwords when the corpus is representative
language is compression and compression is understanding and understanding is power
"""


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(here, "trained_vocab.json")

    trainer = BPETrainer(vocab_size=320, min_frequency=1, special=["<|pad|>"])
    vocab = trainer.train(CORPUS, name="mini")
    vocab.save(out_path)

    print(f"trained '{vocab.name}' with {len(vocab.merges)} merges")
    print("first merges:", [list(m) for m in vocab.merges[:8]])

    # Reload from disk and use it.
    reloaded: Vocabulary = Vocabulary.load(out_path)
    tok = BPETokenizer(reloaded)
    text = "the cat and the dog ran across the river"
    ids = tok.encode(text)
    print("encode :", ids)
    print("decode :", tok.decode(ids))
    print("saved vocab at:", out_path)


if __name__ == "__main__":
    main()
