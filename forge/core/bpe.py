"""The BPE tokenizer.

Putting the pieces together: a :class:`BPETokenizer` turns text into a list of
integer token ids and back.  Encoding works in three stages:

1. **Pre-tokenize** the string into coarse chunks (see
   :mod:`forge.core.pretokenize`).
2. **Translate** each chunk into the byte-level alphabet and **apply BPE**
   greedily, always merging the highest-priority adjacent pair first.
3. **Look up** each resulting symbol in the vocabulary to obtain its id.

Decoding reverses the process symbol-by-symbol and then reassembles the bytes.
Because every symbol is guaranteed to exist in the vocabulary, decoding is
*always* the exact inverse of encoding -- even for text that was never seen
during training.
"""

import re
from typing import Dict, List, Optional, Tuple

from forge.core.alphabet import ALPHABET, ByteAlphabet
from forge.core.pretokenize import pretokenize
from forge.core.vocab import Vocabulary


class BPETokenizer:
    """A trainable / loadable byte-level BPE tokenizer."""

    def __init__(
        self,
        vocabulary: Vocabulary,
        alphabet: ByteAlphabet = ALPHABET,
    ) -> None:
        self.vocab = vocabulary
        self.alphabet = alphabet
        self.sym2id, self.id2sym = vocabulary.build_index(alphabet)

        # Priority table for pairs: (left, right) -> rank (lower = applied first).
        self.ranks: Dict[Tuple[str, str], int] = {
            (left, right): i for i, (left, right) in enumerate(vocabulary.merges)
        }

        self.special_set = set(vocabulary.special)
        self.special_lookup: Dict[str, int] = {
            tok: self.sym2id[tok] for tok in vocabulary.special
        }
        # Pre-compiled splitter for special tokens (only built when needed).
        self._special_split: Optional[re.Pattern] = None
        if self.special_set:
            escaped = [re.escape(t) for t in vocabulary.special]
            self._special_split = re.compile("(" + "|".join(escaped) + ")")

    # ------------------------------------------------------------------ #
    # Encoding
    # ------------------------------------------------------------------ #
    def encode(self, text: str) -> List[int]:
        """Encode ``text`` into a list of token ids."""
        if self._special_split is None:
            return self._encode_ordinary(text)
        parts = self._special_split.split(text)
        ids: List[int] = []
        for part in parts:
            if part in self.special_lookup:
                ids.append(self.special_lookup[part])
            else:
                ids.extend(self._encode_ordinary(part))
        return ids

    def _encode_ordinary(self, text: str) -> List[int]:
        ids: List[int] = []
        for chunk in pretokenize(text):
            symbols = [self.alphabet.encode(bytes([b])) for b in chunk.encode("utf-8")]
            symbols = self._bpe(symbols)
            ids.extend(self.sym2id[s] for s in symbols)
        return ids

    def _bpe(self, symbols: List[str]) -> List[str]:
        """Greedily merge the highest-priority adjacent pair until none remain."""
        if len(symbols) < 2:
            return symbols
        while True:
            best_rank = None
            best_idx = -1
            best_pair = None
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = self.ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = i
                    best_pair = pair
            if best_pair is None:
                break
            merged = best_pair[0] + best_pair[1]
            new_symbols: List[str] = []
            i = 0
            while i < len(symbols):
                if i == best_idx:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return symbols

    # ------------------------------------------------------------------ #
    # Decoding
    # ------------------------------------------------------------------ #
    def decode(self, ids: List[int]) -> str:
        """Decode a list of token ids back into a string."""
        output: List[str] = []
        buffer: List[str] = []

        def flush() -> None:
            if buffer:
                raw = self.alphabet.decode("".join(buffer))
                output.append(raw.decode("utf-8", errors="replace"))
                buffer.clear()

        for tid in ids:
            symbol = self.id2sym.get(tid, "")
            if symbol in self.special_set:
                flush()
                output.append(symbol)
            else:
                buffer.append(symbol)
        flush()
        return "".join(output)

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #
    def encode_decode(self, text: str) -> str:
        """Round-trip helper used by tests and demos."""
        return self.decode(self.encode(text))

    def tokenize(self, text: str) -> List[str]:
        """Return the textual tokens (symbols) without mapping to ids."""
        out: List[str] = []
        for chunk in pretokenize(text):
            symbols = [self.alphabet.encode(bytes([b])) for b in chunk.encode("utf-8")]
            out.extend(self._bpe(symbols))
        return out

    def compression_ratio(self, text: str) -> float:
        """Tokens-per-character ratio (lower means more compact)."""
        if not text:
            return 0.0
        return len(self.encode(text)) / len(text)
