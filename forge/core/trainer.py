"""BPE trainer.

Implements the classic frequency-based byte-pair encoding algorithm: start from
the 256 base symbols, then repeatedly merge the most frequent adjacent pair
across the whole corpus.  Each merge becomes a new symbol and a new rule; the
process stops once the target vocabulary size is reached or no pair occurs often
enough to be worth merging.

Training is fully deterministic for a given corpus, which makes the resulting
vocabulary reproducible and testable.
"""

from collections import Counter
from typing import Dict, Iterable, List, Tuple

from forge.core.alphabet import ALPHABET, ByteAlphabet
from forge.core.pretokenize import pretokenize
from forge.core.vocab import BASE_SIZE, Vocabulary


class BPETrainer:
    """Learns a BPE merge table from a corpus."""

    def __init__(
        self,
        vocab_size: int = 512,
        min_frequency: int = 1,
        alphabet: ByteAlphabet = ALPHABET,
        special: List[str] | None = None,
    ) -> None:
        if vocab_size < BASE_SIZE:
            raise ValueError("vocab_size must be at least the base alphabet size (256)")
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.alphabet = alphabet
        self.special = list(special or [])

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def train(self, corpus: str | Iterable[str], name: str = "custom") -> Vocabulary:
        """Train a :class:`Vocabulary` from ``corpus``.

        ``corpus`` may be a single string or an iterable of strings.
        """
        texts: List[str] = [corpus] if isinstance(corpus, str) else list(corpus)

        # Step 1: turn every chunk into a list of base symbols.
        words: List[List[str]] = []
        for text in texts:
            for chunk in pretokenize(text):
                symbols = [
                    self.alphabet.encode(bytes([b])) for b in chunk.encode("utf-8")
                ]
                if symbols:
                    words.append(symbols)

        # Step 2: iteratively merge the most frequent pair.
        merges: List[Tuple[str, str]] = []
        target_merges = self.vocab_size - BASE_SIZE - len(self.special)

        while len(merges) < target_merges:
            pair_counts = self._count_pairs(words)
            if not pair_counts:
                break
            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < self.min_frequency:
                break
            merges.append(best_pair)
            words = self._apply_merge(words, best_pair)

        return Vocabulary(name=name, merges=merges, special=list(self.special))

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _count_pairs(words: List[List[str]]) -> Counter:
        counts: Counter = Counter()
        for symbols in words:
            for i in range(len(symbols) - 1):
                counts[(symbols[i], symbols[i + 1])] += 1
        return counts

    @staticmethod
    def _apply_merge(
        words: List[List[str]], pair: Tuple[str, str]
    ) -> List[List[str]]:
        left, right = pair
        merged = left + right
        new_words: List[List[str]] = []
        for symbols in words:
            result: List[str] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == left and symbols[i + 1] == right:
                    result.append(merged)
                    i += 2
                else:
                    result.append(symbols[i])
                    i += 1
            new_words.append(result)
        return new_words
