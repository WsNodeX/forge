import unittest

from forge.core.alphabet import ALPHABET, ByteAlphabet, _build_tables


class TestAlphabet(unittest.TestCase):
    def test_tables_are_bijections(self):
        b2c, c2b = _build_tables()
        self.assertEqual(len(b2c), 256)
        self.assertEqual(len(c2b), 256)
        for b in range(256):
            self.assertEqual(c2b[b2c[b]], b)

    def test_roundtrip_all_bytes(self):
        data = bytes(range(256))
        self.assertEqual(ALPHABET.decode(ALPHABET.encode(data)), data)

    def test_no_control_characters(self):
        # None of the 256 symbols may be a control character.
        forbidden = list(range(0, 32)) + [127]
        for b in range(256):
            self.assertNotIn(ord(ALPHABET.encode(bytes([b]))), forbidden)

    def test_singleton_consistency(self):
        other = ByteAlphabet()
        self.assertEqual(other.encode(b"forge"), ALPHABET.encode(b"forge"))

    def test_base_symbols_count(self):
        self.assertEqual(len(ALPHABET.base_symbols()), 256)


if __name__ == "__main__":
    unittest.main()
