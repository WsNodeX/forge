"""Command line interface for Forge.

Examples
--------
    python -m forge.cli encode "Tokenizers are fun!"
    python -m forge.cli decode "72 101 108 108 111"
    python -m forge.cli tokens "Tokenizers are fun!"
    python -m forge.cli train --text "a small corpus to learn from" --vocab-size 400 --out my_vocab.json
    python -m forge.cli info
"""

import argparse
import json
import sys
from typing import List

from forge import BPETokenizer, BPETrainer, Vocabulary, available, load


def _build_tokenizer(args) -> BPETokenizer:
    if getattr(args, "vocab", None):
        vocab = Vocabulary.load(args.vocab)
    else:
        vocab = load(args.name)
    return BPETokenizer(vocab)


def cmd_encode(args) -> int:
    tok = _build_tokenizer(args)
    ids = tok.encode(args.text)
    print(" ".join(str(i) for i in ids))
    print(f"# {len(ids)} tokens", file=sys.stderr)
    return 0


def cmd_tokens(args) -> int:
    tok = _build_tokenizer(args)
    for piece in tok.tokenize(args.text):
        print(piece)
    return 0


def cmd_decode(args) -> int:
    raw = args.ids.replace(",", " ").split()
    ids = [int(x) for x in raw if x != ""]
    vocab = load(args.name) if not args.vocab else Vocabulary.load(args.vocab)
    tok = BPETokenizer(vocab)
    print(tok.decode(ids))
    return 0


def cmd_train(args) -> int:
    corpus_parts: List[str] = []
    if args.text:
        corpus_parts.append(args.text)
    if args.corpus:
        with open(args.corpus, "r", encoding="utf-8") as handle:
            corpus_parts.append(handle.read())
    if not corpus_parts:
        print("error: provide --text or --corpus", file=sys.stderr)
        return 2
    corpus = "\n".join(corpus_parts)

    special = [s for s in (args.special or "").split() if s]
    trainer = BPETrainer(vocab_size=args.vocab_size, min_frequency=1, special=special)
    vocab = trainer.train(corpus, name=args.out.split("/")[-1].split(".")[0] or "custom")
    vocab.save(args.out)
    print(f"trained vocabulary '{vocab.name}' -> {args.out}")
    print(f"  merges : {len(vocab.merges)}")
    print(f"  special: {vocab.special}")
    print(f"  size   : {vocab.size} tokens")
    return 0


def cmd_info(args) -> int:
    print("Bundled vocabularies:")
    for name in available():
        vocab = load(name)
        print(f"  - {name:10s} size={vocab.size} merges={len(vocab.merges)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forge",
        description="Train and run byte-level BPE tokenizers with zero dependencies.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_enc = sub.add_parser("encode", help="encode text into token ids")
    p_enc.add_argument("text")
    p_enc.add_argument("--name", default="english")
    p_enc.add_argument("--vocab", default=None, help="path to a saved vocabulary JSON")
    p_enc.set_defaults(func=cmd_encode)

    p_tok = sub.add_parser("tokens", help="show the textual tokens")
    p_tok.add_argument("text")
    p_tok.add_argument("--name", default="english")
    p_tok.add_argument("--vocab", default=None)
    p_tok.set_defaults(func=cmd_tokens)

    p_dec = sub.add_parser("decode", help="decode token ids into text")
    p_dec.add_argument("ids", help="space or comma separated token ids")
    p_dec.add_argument("--name", default="english")
    p_dec.add_argument("--vocab", default=None)
    p_dec.set_defaults(func=cmd_decode)

    p_tr = sub.add_parser("train", help="train a new vocabulary")
    p_tr.add_argument("--text", default=None)
    p_tr.add_argument("--corpus", default=None, help="path to a text file")
    p_tr.add_argument("--vocab-size", type=int, default=512)
    p_tr.add_argument("--special", default="", help="space separated special tokens")
    p_tr.add_argument("--out", default="vocab.json", help="output JSON path")
    p_tr.set_defaults(func=cmd_train)

    p_info = sub.add_parser("info", help="list bundled vocabularies")
    p_info.set_defaults(func=cmd_info)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
