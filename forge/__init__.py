"""Forge -- a dependency-free Python toolkit for subword (BPE) tokenizers.

Forge lets you train a byte-level BPE tokenizer from scratch, persist it, and
run it in production with zero third-party dependencies.  A small English
vocabulary is bundled so the library is useful the moment it is imported.
"""

from forge.core.alphabet import ALPHABET, ByteAlphabet
from forge.core.bpe import BPETokenizer
from forge.core.pretokenize import pretokenize
from forge.core.registry import available, load
from forge.core.trainer import BPETrainer
from forge.core.vocab import BASE_SIZE, Vocabulary

__version__ = "1.0.0"

__all__ = [
    "ALPHABET",
    "ByteAlphabet",
    "BPETokenizer",
    "pretokenize",
    "BPETrainer",
    "BASE_SIZE",
    "Vocabulary",
    "load",
    "available",
    "__version__",
]
