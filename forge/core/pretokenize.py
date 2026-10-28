"""Pre-tokenization.

Before BPE can group characters into subwords, the raw string is chopped into
coarse "pre-tokens".  The splitting rules closely follow the de-facto standard
used by modern English tokenizers:

* apostrophe contractions (``'s``, ``'re`` ...) are kept together,
* a leading space is attached to the following word/punctuation so that the
  model can tell "word" apart from " word",
* runs of digits stay grouped,
* other non-whitespace characters are grouped,
* pure whitespace is preserved.

The result is fully deterministic, which is what guarantees that decoding is
always the exact inverse of encoding.
"""

import re
from typing import List

# Contractions are detected before the generic word rule so that ``don't`` is
# split into ``don`` + ``'t`` rather than ``don'`` + ``t``.
_CONTRACTION = r"'s|'t|'re|'ve|'m|'ll|'d"

# ``\w`` is Unicode-aware in Python's ``re`` (str mode), so accented and many
# non-Latin letters are treated as word characters out of the box.
_TOKEN_PATTERN = re.compile(
    rf"{_CONTRACTION}"      # contractions
    r"| ?\w+"               # optional leading space + word
    r"| ?[^\s\w]+"          # optional leading space + other symbols
    r"|\s+(?!\S)"           # whitespace inside a run (not the last space)
    r"|\s+"                 # remaining whitespace
)


def pretokenize(text: str) -> List[str]:
    """Split ``text`` into an ordered list of pre-tokens.

    The concatenation of the returned pieces equals ``text`` exactly.
    """
    if not text:
        return []
    return _TOKEN_PATTERN.findall(text)
