import unittest
import torch
from dongxi_llms.course_foundations import ByteBPE, assess_smoke, fingerprint, visible_sources


class CourseFoundationsTests(unittest.TestCase):
    def test_identity_is_order_invariant_but_sensitive_to_controls(self):
        self.assertEqual(fingerprint({"a": 1, "b": 2}), fingerprint({"b": 2, "a": 1}))
        self.assertNotEqual(fingerprint({"seed": 1}), fingerprint({"seed": 2}))

    def test_successful_exit_does_not_override_safety_or_missing_loss(self):
        self.assertFalse(assess_smoke(0, [1.2], 20, 25)["accepted"])
        self.assertFalse(assess_smoke(0, [], 30, 25)["accepted"])
        self.assertFalse(assess_smoke(0, [float("nan")], 30, 25)["accepted"])
        self.assertTrue(assess_smoke(0, [1.2], 30, 25)["accepted"])

    def test_english_only_training_preserves_unseen_unicode_bytes(self):
        tokenizer = ByteBPE.train(["the cat liked the garden"] * 5, 12)
        for text in ["数", "数据库", "språkmodellen", "🧠", "", "café"]:
            self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        self.assertEqual(tokenizer.encode("数"), list("数".encode("utf-8")))

    def test_frequent_chinese_sequence_can_merge_without_losing_alphabet(self):
        tokenizer = ByteBPE.train(["数据库"] * 10, 8)
        self.assertEqual(len(tokenizer.encode("数据库")), 1)
        self.assertEqual(len(tokenizer.pieces), 264)
        self.assertEqual(tokenizer.decode(tokenizer.encode("不认识的字")), "不认识的字")

    def test_packing_forbids_cross_document_and_future_sources(self):
        mask = visible_sources([0, 0, 1, 1, 1])
        self.assertEqual(mask[3].tolist(), [False, False, True, True, False])
        self.assertTrue(torch.diag(mask).all())


if __name__ == "__main__":
    unittest.main()
