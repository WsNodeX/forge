"""Byte-level alphabet.

A tokenizer must be able to encode *any* byte sequence losslessly, including
text that contains characters outside the training corpus or even invalid
Unicode.  The classic trick (popularised by GPT-2) is to map every one of the
256 possible bytes onto a dedicated *printable* Unicode symbol.  Once text has
been translated into this 256-symbol alphabet, the rest of the pipeline only
ever has to deal with clean, fixed-size tokens.

This module builds that mapping and exposes a small :class:`ByteAlphabet`
helper that converts between ``bytes`` and the symbol string.
"""

from typing import Dict, Tuple


def _build_tables() -> Tuple[Dict[int, str], Dict[str, int]]:
    """Create the byte->symbol and symbol->byte lookup tables.

    Printable ASCII and the common Latin-1 printable ranges keep their own
    code points.  The remaining bytes (control characters, some whitespace and
    a few separators) are shifted into the ``0x100``+ range so that no symbol
    is ever a control character.
    """
    # Bytes that are already pleasant to look at and safe to keep verbatim.
    safe = (
        list(range(33, 127))       # ! " # ... ~
        + list(range(161, 173))    # ¡ ¢ ... ¬
        + list(range(174, 256))    # ® ¯ ... ÿ
    )

    byte_to_char: Dict[int, str] = {}
    char_to_byte: Dict[str, int] = {}
    free = 0
    for byte in range(256):
        if byte in safe:
            ch = chr(byte)
        else:
            ch = chr(256 + free)
            free += 1
        byte_to_char[byte] = ch
        char_to_byte[ch] = byte
    return byte_to_char, char_to_byte


class ByteAlphabet:
    """Bijection between raw bytes and a fixed 256-symbol alphabet."""

    def __init__(self) -> None:
        self._b2c, self._c2b = _build_tables()

    def encode(self, data: bytes) -> str:
        """Translate raw bytes into the symbol string."""
        return "".join(self._b2c[b] for b in data)

    def decode(self, text: str) -> bytes:
        """Translate a symbol string back into raw bytes."""
        return bytes(self._c2b[ch] for ch in text)

    def base_symbols(self):
        """Return the 256 base symbols in byte order."""
        return [self._b2c[b] for b in range(256)]

    def symbol_for_byte(self, byte: int) -> str:
        return self._b2c[byte]

    def byte_for_symbol(self, symbol: str) -> int:
        return self._c2b[symbol]


#: Process-wide singleton used by the rest of the library.
ALPHABET = ByteAlphabet()
