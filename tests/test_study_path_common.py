import contextlib
import io
import unittest

import torch

from dongxi_llms.study_path.common import Report, call, close, mine_or_ref, not_attempted


class StudyPathCommonTest(unittest.TestCase):
    def test_report_prints_and_strict_raises(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            r = Report("x", strict=False)
            r.expect(True, "fine")
            r.expect(False, "broken", "fix it")
            self.assertFalse(r.done())
        self.assertIn("✅ fine", out.getvalue())
        self.assertIn("❌ broken  — hint: fix it", out.getvalue())
        with contextlib.redirect_stdout(io.StringIO()):
            strict = Report("y", strict=True)
            strict.expect(False, "broken")
            with self.assertRaises(AssertionError):
                strict.done()

    def test_not_attempted_and_call(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(not_attempted(None))
            self.assertTrue(not_attempted(...))
            self.assertFalse(not_attempted(0))
            r = Report("z", strict=False)
            ok, value = call(r, lambda: 1 / 0)
        self.assertFalse(ok)
        self.assertIsNone(value)

    def test_close_never_raises(self):
        self.assertTrue(close(torch.tensor([1.0, 2.0]), [1.0, 2.0]))
        self.assertFalse(close(torch.tensor([1.0]), [1.0, 2.0]))
        self.assertFalse(close("not a number", 1.0))

    def test_mine_or_ref(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(mine_or_ref({"f": 1}, "f", True, 2), 1)
            self.assertEqual(mine_or_ref({"f": 1}, "f", False, 2), 2)


if __name__ == "__main__":
    unittest.main()
