import random
import string
import unittest

from forge import BPETokenizer, BPETrainer, load


def fresh_tokenizer(corpus, vocab_size=320, special=None):
    vocab = BPETrainer(
        vocab_size=vocab_size, min_frequency=1, special=list(special or [])
    ).train(corpus, "t")
    return BPETokenizer(vocab)


class TestBPE(unittest.TestCase):
    def test_roundtrip_ascii(self):
        tok = fresh_tokenizer("the cat sat on the mat the dog ran")
        for s in ["the cat sat", "running dogs", "Hello, World!", "a b c d e"]:
            self.assertEqual(tok.decode(tok.encode(s)), s)

    def test_roundtrip_unicode(self):
        tok = fresh_tokenizer("hello world 你好 日本語 🔥 café")
        for s in ["你好世界", "日本語のテキスト", "emoji 🚀🔥", "café résumé naïve"]:
            self.assertEqual(tok.decode(tok.encode(s)), s)

    def test_roundtrip_random(self):
        tok = fresh_tokenizer(
            "a b c d e f g h i j k l m n o p q r s t u v w x y z " * 10
        )
        random.seed(1)
        for _ in range(200):
            r = random.random()
            if r < 0.6:
                s = "".join(
                    random.choice(string.ascii_letters + " .,!")
                    for _ in range(random.randint(0, 30))
                )
            else:
                s = "".join(
                    random.choice("汉日🔥éà") for _ in range(random.randint(0, 15))
                )
            self.assertEqual(tok.decode(tok.encode(s)), s, repr(s))

    def test_ids_in_range(self):
        tok = BPETokenizer(load("english"))
        ids = tok.encode("The quick brown fox jumps over the lazy dog.")
        for i in ids:
            self.assertTrue(0 <= i < tok.vocab.size)

    def test_special_token_roundtrip(self):
        tok = fresh_tokenizer("hello world", special=["<|eot|>"])
        text = "hello <|eot|> world"
        ids = tok.encode(text)
        self.assertIn(tok.special_lookup["<|eot|>"], ids)
        self.assertEqual(tok.decode(ids), text)

    def test_empty_inputs(self):
        tok = fresh_tokenizer("abc")
        self.assertEqual(tok.encode(""), [])
        self.assertEqual(tok.decode([]), "")

    def test_compression_ratio(self):
        tok = fresh_tokenizer("the the the the the", vocab_size=300)
        ratio = tok.compression_ratio("the")
        self.assertGreater(ratio, 0)


if __name__ == "__main__":
    unittest.main()
