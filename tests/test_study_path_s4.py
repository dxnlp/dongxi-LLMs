"""Stage S4 (GRPO) checks: each passes its reference and catches a plausible mistake."""
import contextlib
import io
import math
import unittest

import torch

from dongxi_llms.study_path import s4_grpo as s4


def quiet(fn, *args, **kwargs):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        result = fn(*args, **kwargs)
    return result, out.getvalue()


class S4ChecksTest(unittest.TestCase):
    def assert_passes(self, check, *fns):
        ok, out = quiet(check, *fns)
        self.assertTrue(ok, out)
        self.assertNotIn("❌", out)
        quiet(check, *fns, strict=True)

    def assert_fails(self, check, *fns, hint=None):
        ok, out = quiet(check, *fns)
        self.assertFalse(ok, out)
        self.assertIn("❌", out)
        if hint:
            self.assertIn(hint, out)
        with self.assertRaises(AssertionError):
            quiet(check, *fns, strict=True)

    def test_not_attempted(self):
        ok, out = quiet(s4.check_group_advantages, None)
        self.assertFalse(ok)
        self.assertIn("⏳", out)
        ok, out = quiet(s4.check_aggregations, None, None, None)
        self.assertIn("⏳", out)

    def test_group_advantages(self):
        self.assert_passes(s4.check_group_advantages, s4.ref_group_advantages)
        sample_std = lambda R: (R - R.mean(-1, keepdim=True)) / (R.std(-1, keepdim=True) + 1e-8)
        self.assert_fails(s4.check_group_advantages, sample_std, hint="sample std")
        global_stats = lambda R: (R - R.mean()) / (R.std(correction=0) + 1e-8)
        self.assert_fails(s4.check_group_advantages, global_stats, hint="per group")
        no_guard = lambda R: (R - R.mean(-1, keepdim=True)) / R.std(-1, correction=0, keepdim=True)
        self.assert_fails(s4.check_group_advantages, no_guard)

    def test_zero_signal_prob(self):
        self.assert_passes(s4.check_zero_signal_prob, s4.ref_zero_signal_prob)
        self.assert_fails(s4.check_zero_signal_prob, lambda p, G: (1 - p) ** G, hint="add p**G")

    def test_aggregations(self):
        self.assert_passes(s4.check_aggregations, s4.ref_seq_mean, s4.ref_token_mean, s4.ref_constant_norm)
        padded = lambda u, m: (u * m).mean(1).mean()
        self.assert_fails(s4.check_aggregations, padded, s4.ref_token_mean, s4.ref_constant_norm, hint="own count")
        const_by_tokens = lambda u, m, L: (u * m).sum() / m.sum()
        self.assert_fails(s4.check_aggregations, s4.ref_seq_mean, s4.ref_token_mean, const_by_tokens)

    def test_centered(self):
        self.assert_passes(s4.check_centered_advantages, s4.ref_centered_advantages)
        self.assert_fails(s4.check_centered_advantages, s4.ref_group_advantages, hint="drops the division")

    def test_k1_k3(self):
        self.assert_passes(s4.check_k1_k3, s4.ref_k1, s4.ref_k3)
        flipped = lambda lp, lq: torch.exp(lp - lq) - (lp - lq) - 1
        self.assert_fails(s4.check_k1_k3, s4.ref_k1, flipped, hint="d = log π_ref − log π")
        self.assert_fails(s4.check_k1_k3, lambda lp, lq: lq - lp, s4.ref_k3)

    def test_kl_exact(self):
        self.assert_passes(s4.check_kl_exact, s4.ref_kl_exact)
        self.assert_fails(s4.check_kl_exact, lambda p, q: (q * (q.log() - p.log())).sum(-1), hint="KL(π_ref‖π)")
        self.assertAlmostEqual(s4.D35_KL, 0.0851, places=3)

    def test_k3_surrogate(self):
        self.assert_passes(s4.check_k3_sampled_surrogate, s4.ref_k3_sampled_surrogate)
        undetached = lambda z, lq: (z.softmax(-1) * s4.ref_k3(z.log_softmax(-1), lq)).sum()
        self.assert_fails(s4.check_k3_sampled_surrogate, undetached, hint="p.detach()")

    def test_k3_surrogate_gradient_is_reverse_kl(self):
        z = torch.randn(7, dtype=torch.float64, requires_grad=True)
        q = torch.randn(7, dtype=torch.float64).softmax(-1)
        (g,) = torch.autograd.grad(s4.ref_k3_sampled_surrogate(z, q.log()), z)
        zz = z.detach().clone().requires_grad_(True)
        (g_rev,) = torch.autograd.grad(s4.ref_kl_exact(q, zz.softmax(-1)), zz)
        torch.testing.assert_close(g, g_rev)

    def test_grpo_loss(self):
        self.assert_passes(s4.check_grpo_loss, s4.ref_grpo_loss)

        def no_min(logp, old, ref, A, mask, eps=0.2, beta=0.04):
            ratio = torch.exp(logp - old.detach())
            per = -ratio.clamp(1 - eps, 1 + eps) * A[:, None] + beta * s4.ref_k3(logp, ref)
            return s4.ref_seq_mean(per, mask)
        self.assert_fails(s4.check_grpo_loss, no_min)

        def kl_minus(logp, old, ref, A, mask, eps=0.2, beta=0.04):
            ratio = torch.exp(logp - old.detach())
            surr = torch.minimum(ratio * A[:, None], ratio.clamp(1 - eps, 1 + eps) * A[:, None])
            return s4.ref_seq_mean(-surr - beta * s4.ref_k3(logp, ref), mask)
        self.assert_fails(s4.check_grpo_loss, kl_minus)

    def test_reinforce(self):
        self.assert_passes(s4.check_reinforce_loss, s4.ref_reinforce_loss)
        self.assert_fails(s4.check_reinforce_loss,
                          lambda lp, R, m: s4.ref_seq_mean(-(R - R.mean())[:, None] * lp, m))

    def test_length_reward(self):
        self.assert_passes(s4.check_length_reward, s4.ref_length_reward)
        self.assert_fails(s4.check_length_reward, lambda t, tr: float(5 <= len(t.split()) <= 20))
        self.assert_fails(s4.check_length_reward, lambda t, tr: float(not tr and 5 < len(t.split()) < 20))

    def test_grpo_terms(self):
        self.assert_passes(s4.check_grpo_terms, s4.ref_grpo_terms)

        def attached_old(R, logp, ref, mask, G):
            A = s4.ref_group_advantages(R.view(-1, G)).flatten()
            return A, s4.ref_grpo_loss(logp, logp, ref, A, mask)
        # ref_grpo_loss detaches old_logp itself, so emulate a loss that doesn't
        def attached_old_raw(R, logp, ref, mask, G):
            A = s4.ref_group_advantages(R.view(-1, G)).flatten()
            ratio = torch.exp(logp - logp)
            per = -ratio * A[:, None] + 0.04 * s4.ref_k3(logp, ref)
            return A, s4.ref_seq_mean(per, mask)
        self.assert_passes(s4.check_grpo_terms, attached_old)
        self.assert_fails(s4.check_grpo_terms, attached_old_raw, hint="logp.detach()")

        def wrong_groups(R, logp, ref, mask, G):
            A = s4.ref_group_advantages(R.view(G, -1)).flatten()
            return A, s4.ref_grpo_loss(logp, logp.detach(), ref, A, mask)
        self.assert_fails(s4.check_grpo_terms, wrong_groups, hint="R.view(-1, G)")

    def test_zero_signal_numbers(self):
        self.assertAlmostEqual(s4.ref_zero_signal_prob(0.9, 8), 0.9 ** 8 + 0.1 ** 8)
        self.assertTrue(math.isclose(s4.ref_zero_signal_prob(0.1, 4), 0.6562))


if __name__ == "__main__":
    unittest.main()
