import unittest

from forge.core.trainer import BPETrainer
from forge.core.vocab import Vocabulary


class TestTrainer(unittest.TestCase):
    def test_deterministic(self):
        corpus = "low low low lower lowest new newer newest"
        a = BPETrainer(vocab_size=300, min_frequency=1).train(corpus, "a")
        b = BPETrainer(vocab_size=300, min_frequency=1).train(corpus, "b")
        self.assertEqual(a.merges, b.merges)

    def test_learns_at_least_one_merge(self):
        v = BPETrainer(vocab_size=260, min_frequency=1).train("low low low low low", "c")
        self.assertTrue(len(v.merges) >= 1)

    def test_vocab_size_is_a_maximum(self):
        v = BPETrainer(vocab_size=300, min_frequency=1).train("ab ab ab cd cd", "d")
        self.assertLessEqual(len(v.merges), 300 - 256)
        self.assertEqual(v.size, 256 + len(v.merges) + len(v.special))

    def test_min_frequency_stops_early(self):
        v = BPETrainer(vocab_size=400, min_frequency=5).train("ab cd ef", "e")
        self.assertEqual(len(v.merges), 0)

    def test_special_tokens_preserved(self):
        v = BPETrainer(vocab_size=300, min_frequency=1, special=["<|pad|>"]).train("a b c", "f")
        self.assertIn("<|pad|>", v.special)

    def test_corpus_iterable(self):
        v = BPETrainer(vocab_size=300, min_frequency=1).train(["hello world", "world hello"], "g")
        self.assertIsInstance(v, Vocabulary)
        self.assertTrue(len(v.merges) >= 1)


if __name__ == "__main__":
    unittest.main()
