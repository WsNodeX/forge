import os
import tempfile
import unittest

from forge.core.vocab import Vocabulary


class TestVocabulary(unittest.TestCase):
    def _sample(self):
        return Vocabulary(name="x", merges=[("a", "b"), ("c", "d")], special=["<|eot|>"])

    def test_base_byte_mapping(self):
        s2i, _ = self._sample().build_index()
        # The ASCII letter 'a' is byte 97 and should keep id 97.
        self.assertEqual(s2i["a"], 97)

    def test_merge_ids_follow_base(self):
        s2i, _ = self._sample().build_index()
        self.assertEqual(s2i["ab"], 256)
        self.assertEqual(s2i["cd"], 257)

    def test_special_ids_after_merges(self):
        s2i, i2s = self._sample().build_index()
        self.assertEqual(s2i["<|eot|>"], 256 + 2)
        self.assertEqual(i2s[256 + 2], "<|eot|>")

    def test_size_accounts_for_everything(self):
        v = self._sample()
        self.assertEqual(v.size, 256 + 2 + 1)

    def test_save_load_roundtrip(self):
        v = self._sample()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "v.json")
            v.save(path)
            v2 = Vocabulary.load(path)
            self.assertEqual(v2.merges, v.merges)
            self.assertEqual(v2.special, v.special)
            self.assertEqual(v2.name, v.name)
            self.assertEqual(v2.size, v.size)


if __name__ == "__main__":
    unittest.main()
