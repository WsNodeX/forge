import unittest

from forge import BPETokenizer, available, load


class TestRegistry(unittest.TestCase):
    def test_available_includes_english(self):
        self.assertIn("english", available())

    def test_load_english(self):
        vocab = load("english")
        self.assertEqual(vocab.name, "english")
        self.assertTrue(len(vocab.merges) > 100)

    def test_english_roundtrip(self):
        tok = BPETokenizer(load("english"))
        s = "The rapid expansion of language models changed how we process text."
        self.assertEqual(tok.decode(tok.encode(s)), s)

    def test_unknown_vocabulary_raises(self):
        with self.assertRaises(KeyError):
            load("does-not-exist")


if __name__ == "__main__":
    unittest.main()
