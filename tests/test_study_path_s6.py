import contextlib
import io
import math
import unittest

import torch
import torch.nn.functional as F

from dongxi_llms.study_path import s6_combinations as s6


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        result = fn(*args, **kwargs)
    return result, out.getvalue()


class StudyPathS6ChecksTest(unittest.TestCase):
    def assert_passes(self, check, *refs):
        ok, out = quiet(check, *refs)
        self.assertTrue(ok, out)
        self.assertNotIn("❌", out)
        quiet(check, *refs, strict=True)

    def assert_fails(self, check, *wrong):
        ok, out = quiet(check, *wrong)
        self.assertFalse(ok, out)
        self.assertIn("❌", out)
        with self.assertRaises(AssertionError):
            quiet(check, *wrong, strict=True)

    def test_not_attempted(self):
        ok, out = quiet(s6.check_best_of_n, None)
        self.assertFalse(ok)
        self.assertIn("⏳", out)
        ok, out = quiet(s6.check_combine, None, None)
        self.assertIn("⏳", out)

    def test_best_of_n(self):
        self.assert_passes(s6.check_best_of_n, s6.ref_best_of_n)
        self.assert_fails(s6.check_best_of_n, lambda samples, scores: scores[samples].argmax(1))
        ok, out = quiet(s6.check_best_of_n, lambda samples, scores: ...)
        self.assertFalse(ok)

    def test_bon_probs(self):
        self.assert_passes(s6.check_bon_probs, s6.ref_bon_probs)

        def cdf_only(p, scores, n):
            order = scores.argsort()
            out = torch.empty_like(p)
            out[order] = p[order].cumsum(0) ** n
            return out
        self.assert_fails(s6.check_bon_probs, cdf_only)
        ok, out = quiet(s6.check_bon_probs, cdf_only)
        self.assertIn("subtract", out)

    def test_bon_probs_matches_monte_carlo(self):
        p = torch.tensor([0.1, 0.25, 0.2, 0.25, 0.05, 0.15], dtype=torch.float64)
        scores = torch.tensor([1.5, 0.5, -1.0, 0.0, 2.5, -2.0], dtype=torch.float64)
        g = torch.Generator().manual_seed(0)
        samples = torch.multinomial(p, 200000 * 4, replacement=True, generator=g).view(-1, 4)
        picked = s6.ref_best_of_n(samples, scores)
        freq = torch.bincount(picked, minlength=6).double() / len(picked)
        self.assertLess((freq - s6.ref_bon_probs(p, scores, 4)).abs().max().item(), 0.005)
        # two rounds of best-of-4 equal one round of best-of-16
        two = s6.ref_bon_probs(s6.ref_bon_probs(p, scores, 4), scores, 4)
        torch.testing.assert_close(two, s6.ref_bon_probs(p, scores, 16))

    def test_sft_loss(self):
        self.assert_passes(s6.check_sft_loss, s6.ref_sft_loss)
        self.assert_fails(s6.check_sft_loss, lambda logits, picked: -torch.log_softmax(logits, -1)[picked].sum())
        self.assert_fails(s6.check_sft_loss, lambda logits, picked: -logits[picked].mean())

    def test_labels(self):
        self.assert_passes(s6.check_label_hard, s6.ref_label_hard)
        self.assert_fails(s6.check_label_hard, lambda a, b, m: (a > b, (a - b) >= m))
        self.assert_passes(s6.check_label_bt, s6.ref_label_bt)
        self.assert_fails(s6.check_label_bt, lambda a, b, g=None: (a > b, torch.ones_like(a, dtype=torch.bool)))
        self.assert_fails(s6.check_label_bt, lambda a, b, g=None: (
            torch.rand(a.shape, dtype=a.dtype) < torch.sigmoid(a - b), torch.ones_like(a, dtype=torch.bool)))

    def test_kl_optimum(self):
        self.assert_passes(s6.check_kl_optimum, s6.ref_kl_optimum)
        self.assert_fails(s6.check_kl_optimum, lambda p, r, beta: torch.softmax(p.log() + r * beta, -1))
        self.assert_fails(s6.check_kl_optimum, lambda p, r, beta: torch.softmax(r / beta, -1))

    def test_group_z(self):
        self.assert_passes(s6.check_group_z, s6.ref_group_z)
        self.assert_passes(s6.check_group_z, lambda s: (s - s.mean(-1, keepdim=True)) /
                           (s.std(-1, keepdim=True, correction=0) + 1e-6))
        self.assert_fails(s6.check_group_z, lambda s: (s - s.mean()) / (s.std() + 1e-6))
        self.assert_fails(s6.check_group_z, lambda s: (s - s.mean(-1, keepdim=True)) / s.std(-1, keepdim=True))

    def test_combine(self):
        self.assert_passes(s6.check_combine, s6.ref_combine_sum, s6.ref_combine_gate)
        self.assert_fails(s6.check_combine, s6.ref_combine_sum, lambda v, z, w: v + w * torch.sigmoid(z))
        self.assert_fails(s6.check_combine, lambda v, z, w: v + z, s6.ref_combine_gate)

    def test_rlvr_loss(self):
        self.assert_passes(s6.check_rlvr_loss, s6.ref_rlvr_loss)

        def no_baseline(logits, ref_logp, y, R, beta):
            logp = torch.log_softmax(logits, -1)
            return -(R * logp[y]).mean() + beta * (logp.exp() * (logp - ref_logp)).sum()

        def reverse_kl(logits, ref_logp, y, R, beta):
            logp = torch.log_softmax(logits, -1)
            A = R - R.mean(1, keepdim=True)
            return -(A * logp[y]).mean() + beta * (ref_logp.exp() * (ref_logp - logp)).sum()
        self.assert_fails(s6.check_rlvr_loss, no_baseline)
        self.assert_fails(s6.check_rlvr_loss, reverse_kl)
        ok, out = quiet(s6.check_rlvr_loss, reverse_kl)
        self.assertIn("KL(π_ref ‖ π)", out)

    def test_best_vs_worst(self):
        self.assert_passes(s6.check_best_vs_worst, s6.ref_best_vs_worst)

        def top_two(scores, margin):
            top = scores.topk(2, dim=1).values
            return scores.argmax(1), scores.argmin(1), (top[:, 0] - top[:, 1]) >= margin
        self.assert_fails(s6.check_best_vs_worst, top_two)

    def test_paired_diff(self):
        self.assert_passes(s6.check_paired_diff, s6.ref_paired_diff)
        self.assert_fails(s6.check_paired_diff, lambda b, a: (
            (a - b).mean(), math.sqrt(b.var() / len(b) + a.var() / len(a))))
        self.assert_fails(s6.check_paired_diff, lambda b, a: ((a - b).mean(), (a - b).std(correction=0) / math.sqrt(len(a))))

    def test_dpo_offline_leaves_unseen_logits_fixed(self):
        """The mechanism Part B asserts: DPO on pairs that never contain answer u leaves θ_u unchanged."""
        theta = torch.zeros(4, dtype=torch.float64, requires_grad=True)
        ref = torch.log_softmax(torch.zeros(4, dtype=torch.float64), -1)
        c, rj = torch.tensor([0, 1, 0]), torch.tensor([1, 2, 2])
        h = torch.log_softmax(theta, -1) - ref
        loss = -F.logsigmoid(h[c] - h[rj]).mean()
        loss.backward()
        self.assertEqual(theta.grad[3].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
