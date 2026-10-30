"""Stage S3 checks: each passes its reference, fails (without raising) on plausible mistakes."""
import contextlib
import io
import unittest

import torch

from dongxi_llms.study_path import s3_policy_gradient_to_ppo as s3


def quiet(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*args, **kwargs)
    return out, buf.getvalue()


class CheckContract(unittest.TestCase):
    def assert_contract(self, check, ref, wrongs, **kw):
        ok, text = quiet(check, ref, strict=True, **kw)
        self.assertTrue(ok, text)
        self.assertNotIn("❌", text)
        for wrong in wrongs:
            ok, text = quiet(check, wrong, **kw)
            self.assertFalse(ok, f"{getattr(wrong, '__name__', wrong)} should fail\n{text}")
            self.assertIn("❌", text)
            with self.assertRaises(AssertionError):
                quiet(check, wrong, strict=True, **kw)
        ok, text = quiet(check, None, **kw)
        self.assertFalse(ok)
        self.assertIn("⏳", text)

    def test_exact_grad(self):
        def no_baseline(z, R):
            return z.softmax(-1) * R
        self.assert_contract(s3.check_exact_grad, s3.ref_exact_grad, [no_baseline])

    def test_reinforce_loss(self):
        def no_minus(z, a, w):
            return (w.detach() * z.log_softmax(-1)[a]).mean()

        def prob_not_logprob(z, a, w):
            return -(w.detach() * z.softmax(-1)[a]).mean()

        def no_detach(z, a, w):
            return -(w * z.log_softmax(-1)[a]).mean()

        def summed(z, a, w):
            return -(w.detach() * z.log_softmax(-1)[a]).sum()
        self.assert_contract(s3.check_reinforce_loss, s3.ref_reinforce_loss,
                             [no_minus, prob_not_logprob, no_detach, summed])

    def test_rloo(self):
        def inclusive(R):
            return R - R.mean(-1, keepdim=True)

        def whole_tensor(R):
            return R - (R.sum() - R) / (R.shape[-1] - 1)
        self.assert_contract(s3.check_rloo, s3.ref_rloo_advantages, [inclusive, whole_tensor])

    def test_exact_values(self):
        def unweighted(probs, R):
            T, K = len(probs), probs[0].shape[-1]
            Q, V, nxt = [None] * T, [None] * T, R
            for d in reversed(range(T)):
                Q[d] = nxt.view(K ** d, K)
                V[d] = Q[d].mean(-1)
                nxt = V[d]
            return Q, V
        self.assert_contract(s3.check_exact_values, s3.ref_exact_values, [unweighted])

    def test_gae(self):
        def forward(rewards, values, mask, gamma=1.0, lam=0.95):
            T = rewards.shape[1]
            adv, running = torch.zeros_like(rewards), torch.zeros_like(rewards[:, 0])
            for t in range(T):
                nv = values[:, t + 1] * mask[:, t + 1] if t + 1 < T else torch.zeros_like(running)
                running = (rewards[:, t] + gamma * nv - values[:, t] + gamma * lam * running) * mask[:, t]
                adv[:, t] = running
            return adv

        def leaks_padding(rewards, values, mask, gamma=1.0, lam=0.95):
            T = rewards.shape[1]
            adv, running = torch.zeros_like(rewards), torch.zeros_like(rewards[:, 0])
            for t in reversed(range(T)):
                nv = values[:, t + 1] if t + 1 < T else torch.zeros_like(running)
                running = (rewards[:, t] + gamma * nv - values[:, t] + gamma * lam * running) * mask[:, t]
                adv[:, t] = running
            return adv

        def no_gamma_on_running(rewards, values, mask, gamma=1.0, lam=0.95):
            return s3.ref_gae(rewards, values, mask, gamma, lam / gamma if gamma else lam)
        self.assert_contract(s3.check_gae, s3.ref_gae, [forward, leaks_padding, no_gamma_on_running])

    def test_kl_shaped_rewards(self):
        def everywhere(R, logp, ref, mask, last, beta):
            kl = (logp - ref) * mask
            return -beta * kl + R[:, None] * mask, kl

        def off_by_one(R, logp, ref, mask, last, beta):
            kl = (logp - ref) * mask
            rew = -beta * kl
            idx = mask.sum(1).long().clamp(max=mask.shape[1] - 1)
            rew[torch.arange(len(R)), idx] += R
            return rew, kl

        def plus_beta(R, logp, ref, mask, last, beta):
            kl = (logp - ref) * mask
            rew = beta * kl
            rew[torch.arange(len(R)), last] += R
            return rew, kl
        self.assert_contract(s3.check_kl_shaped_rewards, s3.ref_kl_shaped_rewards,
                             [everywhere, off_by_one, plus_beta])

    def test_clipped_policy_loss(self):
        def max_not_min(logp, old, adv, mask, eps=0.2):
            ratio = torch.exp(logp - old)
            s = torch.max(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv)
            return -(s * mask).sum() / mask.sum(), (((ratio - 1).abs() > eps).float() * mask).sum() / mask.sum()

        def clamp_only(logp, old, adv, mask, eps=0.2):
            ratio = torch.exp(logp - old)
            s = ratio.clamp(1 - eps, 1 + eps) * adv
            return -(s * mask).sum() / mask.sum(), (((ratio - 1).abs() > eps).float() * mask).sum() / mask.sum()

        def unmasked(logp, old, adv, mask, eps=0.2):
            ratio = torch.exp(logp - old)
            s = torch.min(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv)
            return -s.mean(), (((ratio - 1).abs() > eps).float() * mask).sum() / mask.sum()
        self.assert_contract(s3.check_clipped_policy_loss, s3.ref_clipped_policy_loss,
                             [max_not_min, clamp_only, unmasked])

    def test_value_loss(self):
        def adv_target(nv, ov, adv, mask):
            return ((nv - adv) ** 2 * mask).sum() / mask.sum()

        def unmasked(nv, ov, adv, mask):
            return ((nv - (adv + ov).detach()) ** 2).mean()
        self.assert_contract(s3.check_value_loss, s3.ref_value_loss, [adv_target, unmasked])

    def test_rule_reward(self):
        def strict_lt(lengths, finished, target=48):
            return torch.where(finished, torch.where(lengths < target, 1.0, 0.5), 0.0)

        def ignores_truncation(lengths, finished, target=48):
            return torch.where(lengths <= target, 1.0, 0.5)
        self.assert_contract(s3.check_rule_reward, s3.ref_rule_reward, [strict_lt, ignores_truncation])

    def test_ppo_signals(self):
        def make(variant):
            def fn(R, old, ref, V, ro, beta=0.05, lam=0.95):
                mask, m = ro["mask"], ro["mask"].bool()
                rewards, kl = s3.ref_kl_shaped_rewards(R, old, ref, mask, ro["last"], beta)
                adv = s3.ref_gae(rewards, V, mask, 1.0, lam)
                if variant == "whiten_first":
                    adv = (adv - adv[m].mean()) / (adv[m].std() + 1e-8) * mask
                    returns = adv + V
                else:
                    returns = adv + V
                    if variant == "whiten_all":
                        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                    else:
                        adv = (adv - adv[m].mean()) / (adv[m].std() + 1e-8) * mask
                return rewards, adv, returns, kl
            return fn
        self.assert_contract(s3.check_ppo_signals, make("ref"), [make("whiten_first"), make("whiten_all")])

    def test_ref_ppo_signals_matches_variant(self):
        R, old, ref, V, ro = s3._toy_signals_case()
        out = s3.ref_ppo_signals(R, old, ref, V, ro["mask"], ro["last"], 0.05, 0.95)
        m = ro["mask"].bool()
        self.assertAlmostEqual(float(out[1][m].mean()), 0.0, places=6)
        self.assertAlmostEqual(float(out[1][m].std()), 1.0, places=6)

    def test_exact_moments_baseline_identity(self):
        z = torch.log(torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float64))
        R = torch.tensor([1.0, 2.0, 3.0, 6.0], dtype=torch.float64)
        exact = s3.ref_exact_grad(z, R)
        none = s3.exact_moments(z, R, 3, lambda r: r)
        loo = s3.exact_moments(z, R, 3, s3.ref_rloo_advantages)
        inc = s3.exact_moments(z, R, 3, lambda r: r - r.mean())
        torch.testing.assert_close(none["mean"], exact)
        torch.testing.assert_close(loo["mean"], exact)
        torch.testing.assert_close(inc["mean"], exact * 2 / 3)
        self.assertLess(loo["var"], none["var"])


if __name__ == "__main__":
    unittest.main()
