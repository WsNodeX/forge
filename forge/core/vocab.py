"""Vocabulary model and (de)serialization.

A :class:`Vocabulary` is the compact, persistence-friendly description of a
trained tokenizer: an ordered list of merge rules plus an optional set of
special tokens.  Everything else (the symbol<->id tables) is derived from these
two pieces, so saving and loading a tokenizer is trivial.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from forge.core.alphabet import ALPHABET, ByteAlphabet

#: Number of base symbols (one per possible byte value).
BASE_SIZE = 256


@dataclass
class Vocabulary:
    """The learned state of a tokenizer.

    Attributes:
        name: Human readable identifier.
        merges: Ordered list of ``(left, right)`` merge rules.  Earlier rules
            have higher priority during encoding.
        special: Literal strings that map to dedicated token ids (e.g.
            ``"<|endoftext|>"``).  Their ids are assigned automatically beyond
            the merge range so they never collide with subword tokens.
    """

    name: str = "untitled"
    merges: List[Tuple[str, str]] = field(default_factory=list)
    special: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Derived tables
    # ------------------------------------------------------------------ #
    def build_index(self, alphabet: ByteAlphabet = ALPHABET):
        """Return ``(symbol->id, id->symbol)`` tables.

        Base bytes occupy ids ``0..255`` in byte order.  Each merge rule ``i``
        introduces a new symbol (the concatenation of its two sides) at id
        ``256 + i``.  Special tokens are placed after the merge range.
        """
        sym2id: Dict[str, int] = {}
        id2sym: Dict[int, str] = {}
        for byte in range(BASE_SIZE):
            ch = alphabet.encode(bytes([byte]))
            sym2id[ch] = byte
            id2sym[byte] = ch

        for i, (left, right) in enumerate(self.merges):
            sym = left + right
            tid = BASE_SIZE + i
            sym2id[sym] = tid
            id2sym[tid] = sym

        special_start = BASE_SIZE + len(self.merges)
        for offset, token in enumerate(self.special):
            tid = special_start + offset
            sym2id[token] = tid
            id2sym[tid] = token

        return sym2id, id2sym

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "merges": [list(m) for m in self.merges],
            "special": list(self.special),
        }

    def save(self, path: str) -> None:
        """Write the vocabulary to a JSON file."""
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "Vocabulary":
        return cls(
            name=data.get("name", "untitled"),
            merges=[tuple(m) for m in data.get("merges", [])],
            special=list(data.get("special", [])),
        )

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        """Load a vocabulary previously written with :meth:`save`."""
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    @property
    def size(self) -> int:
        """Total number of distinct token ids."""
        return BASE_SIZE + len(self.merges) + len(self.special)
