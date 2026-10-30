"""Study path S1 (reward models): every check passes its reference and rejects a plausible mistake."""
from __future__ import annotations

import contextlib
import io
import math
import unittest

import torch
import torch.nn.functional as F

from dongxi_llms.study_path import s1_reward_models as s1


def quiet(fn, *args, **kwargs):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = fn(*args, **kwargs)
    return result, buffer.getvalue()


class CheckContract(unittest.TestCase):
    """For each check: reference passes (strict), a wrong version returns False with ❌, strict raises."""

    def assert_contract(self, check, ref_args, bad_args):
        ok, out = quiet(check, *ref_args, strict=True)
        self.assertTrue(ok)
        self.assertNotIn("❌", out)
        ok, out = quiet(check, *bad_args)
        self.assertFalse(ok)
        self.assertIn("❌", out)
        with self.assertRaises(AssertionError):
            quiet(check, *bad_args, strict=True)

    def test_not_attempted(self):
        ok, out = quiet(s1.check_bt, None, None)
        self.assertFalse(ok)
        self.assertIn("⏳", out)
        ok, out = quiet(s1.check_aggregate, Ellipsis)
        self.assertFalse(ok)
        self.assertIn("⏳", out)

    def test_bt_unstable_log(self):
        bad_loss = lambda rc, rr: -torch.log(torch.sigmoid(rc - rr)).mean()
        self.assert_contract(s1.check_bt, (s1.ref_bt_prob, s1.ref_bt_loss), (s1.ref_bt_prob, bad_loss))

    def test_bt_sum_and_swap(self):
        summed = lambda rc, rr: -F.logsigmoid(rc - rr).sum()
        swapped = lambda rc, rr: -F.logsigmoid(rr - rc).mean()
        for bad in (summed, swapped):
            ok, out = quiet(s1.check_bt, s1.ref_bt_prob, bad)
            self.assertFalse(ok)
        _, out = quiet(s1.check_bt, s1.ref_bt_prob, summed)
        self.assertIn("sum", out)

    def test_bt_raising_learner_code(self):
        def broken(rc, rr):
            raise TypeError("unsupported operand type(s): 'ellipsis'")
        ok, out = quiet(s1.check_bt, s1.ref_bt_prob, broken)
        self.assertFalse(ok)
        self.assertIn("raised TypeError", out)

    def test_bt_grads(self):
        wrong_sign = lambda rc, rr: (1 - torch.sigmoid(rc - rr), -(1 - torch.sigmoid(rc - rr)))
        self.assert_contract(s1.check_bt_grads, (s1.ref_bt_grads,), (wrong_sign,))

    def test_fit_linear_rm(self):
        def reversed_fit(fc, fr, steps=300, lr=1.0):
            return s1.ref_fit_linear_rm(fr, fc, steps, lr)
        self.assert_contract(s1.check_fit_linear_rm, (s1.ref_fit_linear_rm,), (reversed_fit,))

    def test_noisy_label_prob(self):
        missing = lambda gap, eps: (1 - eps) * torch.sigmoid(gap)
        self.assert_contract(s1.check_noisy_label_prob, (s1.ref_noisy_label_prob,), (missing,))

    def test_reliability(self):
        def unweighted(p, y, n_bins=10):
            out = s1.ref_reliability(p, y, n_bins)
            out["ece"] = sum(abs(a - f) for a, f in zip(out["conf"], out["freq"])) / len(out["conf"])
            return out
        self.assert_contract(s1.check_reliability, (s1.ref_reliability,), (unweighted,))

    def test_last_token_index(self):
        count_based = lambda ids, pad: (ids != pad).long().sum(-1) - 1
        self.assert_contract(s1.check_last_token_index, (s1.ref_last_token_index,), (count_based,))
        _, out = quiet(s1.check_last_token_index, count_based)
        self.assertIn("left padding", out)

    def test_pair_accuracy(self):
        no_ties = lambda sc, sr: ((sc > sr).double().mean().item(),
                                  math.sqrt(0.25 / len(sc)))
        self.assert_contract(s1.check_pair_accuracy, (s1.ref_pair_accuracy,), (no_ties,))

    def test_rm_pair_loss(self):
        interleaved = lambda scores, n: s1.ref_bt_loss(scores[0::2], scores[1::2])
        self.assert_contract(s1.check_rm_pair_loss, (s1.ref_rm_pair_loss,), (interleaved,))

    def test_pearson(self):
        cosine = lambda a, b: float((a * b).sum() / (a.norm() * b.norm()))
        self.assert_contract(s1.check_pearson, (s1.ref_pearson,), (cosine,))

    def test_length_sliced_accuracy(self):
        def inclusive(s_c, s_r, len_c, len_r):
            longer, shorter = len_c >= len_r, len_c < len_r
            return {"chosen_longer": s1.ref_pair_accuracy(s_c[longer], s_r[longer]) + (int(longer.sum()),),
                    "chosen_shorter": s1.ref_pair_accuracy(s_c[shorter], s_r[shorter]) + (int(shorter.sum()),)}
        self.assert_contract(s1.check_length_sliced_accuracy, (s1.ref_length_sliced_accuracy,), (inclusive,))

    def test_labels(self):
        def arithmetic_only(numbers, steps):
            return [int(s1.OPS[op](a, b) == c) for a, op, b, c in steps]
        self.assert_contract(s1.check_labels, (s1.ref_outcome_label, s1.ref_process_labels),
                             (s1.ref_outcome_label, arithmetic_only))

    def test_aggregate(self):
        argmax = lambda p: {"min": p.min().item(), "product": p.prod().item(), "weakest_step": int(p.argmax())}
        self.assert_contract(s1.check_aggregate, (s1.ref_aggregate,), (argmax,))


class ToyFacts(unittest.TestCase):
    """Exact or tightly bounded facts the notebook states about its toy parts."""

    def test_shift_invariance_and_scale(self):
        rc, rr = torch.tensor([0.0, 1.0, 2.0]), torch.tensor([2.0, 1.0, 1.0])
        self.assertTrue(torch.allclose(s1.ref_bt_loss(rc + 10, rr + 10), s1.ref_bt_loss(rc, rr)))
        self.assertFalse(torch.allclose(s1.ref_bt_loss(2 * rc, 2 * rr), s1.ref_bt_loss(rc, rr)))

    def test_random_noise_keeps_direction_length_noise_does_not(self):
        test = s1.make_linear_pairs(20_000, seed=99)
        out = {}
        for judge in ("random", "length"):
            d = s1.make_linear_pairs(4000, seed=1, eps=0.2, judge=judge)
            w, _ = s1.ref_fit_linear_rm(d["fc"], d["fr"])
            agree = (((test["fa"] - test["fb"]) @ w > 0) == (test["gap_true"] > 0)).double().mean().item()
            out[judge] = (w, agree)
        w, agree = out["random"]
        self.assertLess(float(w.norm()), 1.0)                      # scale shrinks from ≈2.1
        self.assertAlmostEqual(float(w[1] / w[0]), 0.25, delta=0.05)
        self.assertGreater(agree, 0.99)
        w, agree = out["length"]
        self.assertGreater(float(w[1] / w[0]), 0.45)
        self.assertLess(agree, 0.95)

    def test_toy_solutions(self):
        labels = {name: (s1.ref_outcome_label(steps, s1.PROBLEMS[p]["gold"]),
                         s1.ref_process_labels(s1.PROBLEMS[p]["numbers"], steps))
                  for name, p, steps in s1.SOLUTIONS}
        self.assertEqual(labels["apples · A"], (1, [1, 1, 1]))
        self.assertEqual(labels["apples · B"], (0, [1, 0, 1]))
        self.assertEqual(labels["apples · C"], (1, [1, 0, 0]))      # right answer, wrong steps
        self.assertEqual(labels["apples · D"], (0, [1, 0, 1]))      # copied a number never computed
        self.assertEqual(labels["tickets · B"], (1, [1, 1, 1, 1]))
        self.assertEqual(labels["tickets · C"], (0, [1, 0, 1]))
        a = s1.ref_aggregate(s1.simulated_prm(labels["tickets · A"][1]))
        b = s1.ref_aggregate(s1.simulated_prm(labels["tickets · B"][1]))
        self.assertEqual(a["min"], b["min"])
        self.assertGreater(a["product"], b["product"])               # product prefers fewer steps


class TokenizerPadding(unittest.TestCase):
    def test_qwen_pad_differs_from_im_end(self):
        try:
            from dongxi_llms.study_path import qwen as Q
            tok = Q.load_tokenizer()
        except Exception as error:  # tokenizer not cached on this machine
            self.skipTest(f"Qwen3 tokenizer unavailable: {error}")
        self.assertNotEqual(tok.pad_token_id, tok.convert_tokens_to_ids("<|im_end|>"))
        enc = tok(["a b c", "a"], return_tensors="pt", padding=True, padding_side="left")
        idx = s1.ref_last_token_index(enc.input_ids, tok.pad_token_id)
        self.assertEqual(idx.tolist(), [enc.input_ids.shape[1] - 1] * 2)


if __name__ == "__main__":
    unittest.main()
