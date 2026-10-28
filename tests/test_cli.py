import io
import sys
import unittest

from forge.cli import main


class TestCLI(unittest.TestCase):
    def _run(self, argv):
        out, err = io.StringIO(), io.StringIO()
        old = (sys.stdout, sys.stderr)
        sys.stdout, sys.stderr = out, err
        try:
            rc = main(argv)
        finally:
            sys.stdout, sys.stderr = old
        return rc, out.getvalue(), err.getvalue()

    def test_encode(self):
        rc, out, _ = self._run(["encode", "hello world"])
        self.assertEqual(rc, 0)
        ids = [int(x) for x in out.split()]
        self.assertTrue(all(0 <= i < 512 for i in ids))

    def test_tokens(self):
        rc, out, _ = self._run(["tokens", "hello"])
        self.assertEqual(rc, 0)
        self.assertTrue(len(out.strip().splitlines()) >= 1)

    def test_info(self):
        rc, out, _ = self._run(["info"])
        self.assertEqual(rc, 0)
        self.assertIn("english", out)

    def test_decode_is_inverse(self):
        rc, out, _ = self._run(["encode", "hi"])
        ids = out.split()
        rc2, out2, _ = self._run(["decode", " ".join(ids)])
        self.assertEqual(rc2, 0)
        self.assertEqual(out2.strip(), "hi")

    def test_train_writes_file(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            out_path = os.path.join(d, "v.json")
            rc, out, _ = self._run(
                ["train", "--text", "alpha beta gamma", "--vocab-size", "300", "--out", out_path]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(out_path))


if __name__ == "__main__":
    unittest.main()
