"""Bundled vocabulary registry.

``forge.pretrained`` re-exports :func:`available` and :func:`load` from here, and
``forge/__init__.py`` imports them as public API. This module was missing from the
package, so ``import forge`` raised::

    ModuleNotFoundError: No module named 'forge.core.registry'

which also broke ``forge.core.alphabet`` (it imports through here transitively).
Importing the library failed before any tokenizer code ran.

A vocabulary is registered as ``(module_path, attribute_name)`` so the pretrained
data is only imported when actually requested — the bundled English table is a few
hundred merge rules and there is no reason to pay for it on every import.
"""
from __future__ import annotations

import importlib
from typing import Dict, List, Tuple

from forge.core.vocab import Vocabulary

#: name -> (module that exposes the merge table, attribute holding it)
_REGISTRY: Dict[str, Tuple[str, str]] = {
    "english": ("forge.pretrained.english", "MERGES"),
}


def available() -> List[str]:
    """Return the names of every bundled vocabulary, sorted."""
    return sorted(_REGISTRY)


def load(name: str) -> Vocabulary:
    """Load a bundled vocabulary by name.

    Raises:
        KeyError: if ``name`` is not registered. The message lists what is
            available, because a bare KeyError with no alternatives is the kind
            of error message that costs the reader a source dive.
    """
    try:
        module_path, attribute = _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown vocabulary {name!r}; available: {available()}"
        ) from None

    module = importlib.import_module(module_path)
    raw = getattr(module, attribute)

    # The generated table stores lists; Vocabulary expects (left, right) tuples.
    merges = [tuple(rule) for rule in raw]
    return Vocabulary(name=name, merges=merges)
