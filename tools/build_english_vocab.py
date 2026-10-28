"""Generate the bundled English vocabulary (``forge/pretrained/english.py``).

This is a developer utility, not part of the published API.  It trains a BPE
model on a curated English corpus and writes the resulting merge table as a
Python module so that Forge ships with a working vocabulary out of the box.

Run from the repository root::

    python tools/build_english_vocab.py
"""

import os
from forge.core.trainer import BPETrainer

# A curated slice of frequent English words.  Repetition is intentional: it
# gives the trainer the frequency signal it needs to learn meaningful subwords
# (``th``, ``ing``, ``the``, ``and`` ...).
_COMMON_WORDS = [
    "the", "be", "of", "and", "a", "to", "in", "he", "have", "it", "that", "for",
    "they", "with", "as", "not", "on", "she", "at", "by", "this", "we", "you",
    "do", "but", "from", "or", "which", "one", "would", "all", "will", "there",
    "say", "who", "make", "when", "can", "more", "if", "no", "man", "out", "other",
    "so", "what", "time", "up", "go", "about", "than", "into", "could", "state",
    "only", "new", "year", "some", "take", "come", "these", "know", "see", "use",
    "get", "like", "then", "first", "any", "work", "now", "may", "such", "give",
    "over", "think", "most", "even", "find", "day", "also", "after", "way", "many",
    "must", "look", "before", "great", "back", "through", "long", "where", "much",
    "should", "well", "people", "down", "own", "just", "because", "good", "each",
    "those", "feel", "seem", "how", "high", "too", "place", "little", "world",
    "very", "still", "nation", "hand", "old", "life", "tell", "write", "become",
    "here", "show", "house", "both", "between", "need", "mean", "call", "develop",
    "under", "last", "right", "move", "thing", "general", "school", "never", "same",
    "another", "begin", "while", "number", "part", "turn", "real", "leave", "might",
    "want", "point", "form", "off", "child", "few", "small", "since", "against",
    "ask", "late", "home", "interest", "large", "person", "end", "open", "public",
    "follow", "present", "without", "again", "hold", "govern", "around", "possible",
    "head", "consider", "word", "program", "problem", "however", "lead", "system",
    "set", "order", "eye", "plan", "run", "keep", "face", "fact", "group", "play",
    "stand", "increase", "early", "course", "change", "help", "line", "city", "put",
    "close", "case", "force", "meet", "once", "water", "upon", "war", "build",
    "hear", "light", "unite", "live", "every", "country", "bring", "center", "let",
    "side", "try", "provide", "continue", "name", "certain", "power", "pay", "result",
    "question", "study", "woman", "member", "until", "far", "night", "always", "service",
    "away", "report", "something", "company", "week", "church", "toward", "start",
    "social", "room", "figure", "nature", "though", "young", "less", "enough", "almost",
    "read", "include", "president", "nothing", "yet", "better", "big", "boy", "cost",
    "business", "value", "second", "why", "clear", "expect", "family", "complete",
    "act", "sense", "mind", "experience", "art", "next", "near", "direct", "car",
    "law", "industry", "important", "girl", "god", "several", "matter", "usual",
    "rather", "per", "often", "kind", "among", "white", "reason", "action", "return",
    "foot", "care", "simple", "within", "love", "human", "along", "appear", "doctor",
    "believe", "speak", "active", "student", "month", "drive", "concern", "best",
    "door", "hope", "example", "inform", "body", "ever", "least", "probable", "understand",
    "reach", "effect", "different", "idea", "whole", "control", "condition", "field",
    "pass", "fall", "note", "special", "talk", "particular", "today", "measure", "walk",
    "teach", "low", "hour", "type", "carry", "rate", "remain", "full", "street", "easy",
    "although", "record", "sit", "determine", "level", "local", "sure", "receive", "thus",
    "moment", "spirit", "train", "college", "religion", "perhaps", "music", "grow",
    "free", "cause", "serve", "age", "book", "board", "recent", "sound", "office",
    "cut", "step", "class", "true", "history", "position", "above", "strong", "friend",
    "necessary", "add", "court", "deal", "tax", "support", "party", "whether", "either",
    "land", "material", "happen", "education", "death", "agree", "arm", "mother", "across",
    "quite", "anything", "town", "past", "view", "society", "manage", "answer", "break",
    "organize", "half", "fire", "lose", "money", "stop", "actual", "already", "effort",
    "wait", "department", "able", "political", "learn", "voice", "air", "together",
    "shall", "cover", "common", "subject", "draw", "short", "wife", "treat", "limit",
    "road", "letter", "color", "behind", "produce", "send", "term", "total", "university",
    "rise", "century", "success", "minute", "remember", "purpose", "test", "fight", "watch",
    "situation", "south", "ago", "difference", "stage", "father", "table", "rest", "bear",
    "entire", "market", "prepare", "explain", "offer", "plant", "charge", "ground", "west",
    "picture", "hard", "front", "lie", "modern", "dark", "surface", "rule", "regard",
    "dance", "peace", "observe", "future", "wall", "farm", "claim", "firm", "operation",
    "further", "pressure", "property", "morning", "amount", "top", "outside", "piece",
    "sometimes", "beauty", "trade", "fear", "demand", "build", "stay", "win", "throw",
    "choose", "develop", "involve", "spend", "sign", "repeat", "report", "role", "better",
]

_PARAGRAPH = (
    "the quick brown fox jumps over the lazy dog while a small child watches the "
    "moving shadows on the quiet wall and wonders about the nature of light and "
    "language and how words carry meaning across great distances and long periods "
    "of time because communication is what binds a society together and lets people "
    "build tools that remember the past and plan for the future in a world that is "
    "always changing yet somehow remains the same beneath the surface of daily life"
)

CORPUS = " ".join(_COMMON_WORDS * 3) + " " + " ".join([_PARAGRAPH] * 20)


def main() -> None:
    trainer = BPETrainer(vocab_size=512, min_frequency=1, special=["<|endoftext|>"])
    vocab = trainer.train(CORPUS, name="english")

    here = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(here, "..", "forge", "pretrained", "english.py")
    out_path = os.path.normpath(out_path)

    lines = [
        "# Auto-generated by tools/build_english_vocab.py -- do not edit by hand.",
        "# A small English BPE vocabulary (256 merges) bundled with Forge.",
        "MERGES = [",
    ]
    for left, right in vocab.merges:
        lines.append('    ["' + left + '", "' + right + '"],')
    lines.append("]")
    lines.append("")
    lines.append("SPECIAL = [")
    for tok in vocab.special:
        lines.append('    "' + tok + '",')
    lines.append("]")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    print(f"Wrote {out_path}")
    print(f"  name   : {vocab.name}")
    print(f"  merges : {len(vocab.merges)}")
    print(f"  special: {vocab.special}")
    print(f"  size   : {vocab.size} tokens")


if __name__ == "__main__":
    main()
