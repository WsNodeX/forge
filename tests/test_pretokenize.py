import unittest

from forge.core.pretokenize import pretokenize


class TestPretokenize(unittest.TestCase):
    def test_concat_equals_original(self):
        samples = [
            "Hello world",
            "don't",
            "  multiple   spaces  ",
            "abc123 def456",
            "hello, world!",
            "The rapid expansion of language models.",
        ]
        for s in samples:
            self.assertEqual("".join(pretokenize(s)), s)

    def test_contractions(self):
        self.assertIn("'t", pretokenize("don't"))
        self.assertIn("'re", pretokenize("they're"))
        self.assertIn("'s", pretokenize("it's"))

    def test_leading_space_attached(self):
        self.assertEqual(pretokenize("hello world"), ["hello", " world"])

    def test_empty_string(self):
        self.assertEqual(pretokenize(""), [])

    def test_unicode_identity(self):
        s = "你好世界 hello 🌍"
        self.assertEqual("".join(pretokenize(s)), s)


if __name__ == "__main__":
    unittest.main()
