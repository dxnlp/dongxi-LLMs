"""Objective-level tests: signs, gauge, masks, support and exact expectations."""
import unittest
import torch
from dongxi_llms.reward_model_lab import bt_loss, reward_comparison
from dongxi_llms.dpo_lab import sequence_logps, dpo_loss, optimal_policy, tiny_sequence_dpo
from dongxi_llms.policy_gradient_lab import (exact_gradient, estimator_moments,
    group_estimator_moments, ppo_surrogate, kl_estimators)


class PreferencePolicyTests(unittest.TestCase):
    def test_bt_gradient_and_gauge(self):
        first = torch.tensor([.4, -.5], dtype=torch.float64, requires_grad=True)
        second = torch.tensor([-.1, .5], dtype=torch.float64, requires_grad=True)
        q = torch.tensor([.8, .2], dtype=torch.float64)
        loss = bt_loss(first, second, q)
        a, b = torch.autograd.grad(loss, (first, second))
        torch.testing.assert_close(a, ((first - second).sigmoid() - q) / 2)
        torch.testing.assert_close(b, -a)
        torch.testing.assert_close(loss, bt_loss(first + 90, second + 90, q))

    def test_sequence_mask_does_not_score_prompt_or_padding(self):
        logits = torch.zeros(2, 4, 5, dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([[-100, 1, 2, -100], [-100, 2, 3, 4]])
        mask = labels != -100
        summed = sequence_logps(logits, labels, mask)
        torch.testing.assert_close(summed, -mask.sum(1) * torch.log(torch.tensor(5., dtype=torch.float64)))
        summed.sum().backward()
        self.assertTrue((logits.grad[~mask] == 0).all())
        with self.assertRaises(ValueError):
            sequence_logps(logits, labels, torch.zeros_like(mask))

    def test_dpo_reference_detached_and_initial_gradient(self):
        chosen = torch.tensor([-2.], requires_grad=True)
        rejected = torch.tensor([-3.], requires_grad=True)
        ref_c = chosen.detach().clone().requires_grad_()
        ref_r = rejected.detach().clone().requires_grad_()
        loss, margin = dpo_loss(chosen, rejected, ref_c, ref_r, beta=.2)
        loss.backward()
        self.assertAlmostEqual(float(margin.detach()), 0.)
        self.assertAlmostEqual(float(chosen.grad), -.1, places=6)
        self.assertAlmostEqual(float(rejected.grad), .1, places=6)
        self.assertIsNone(ref_c.grad)
        self.assertIsNone(ref_r.grad)

    def test_optimum_satisfies_kl_stationarity(self):
        q = torch.tensor([.2, .3, .5], dtype=torch.float64)
        r = torch.tensor([1., -2., .4], dtype=torch.float64)
        p = optimal_policy(q, r, .7)
        stationarity = r - .7 * (p.log() - q.log() + 1)
        torch.testing.assert_close(stationarity, stationarity.mean().expand_as(r))

    def test_tiny_decoder_reference_and_finite_update(self):
        result = tiny_sequence_dpo(steps=4)
        self.assertFalse(result["reference_has_gradients"])
        self.assertTrue(result["reference_self_check"])
        self.assertTrue(all(torch.isfinite(torch.tensor(row["loss"])) for row in result["history"]))

    def test_nan_preference_is_rejected(self):
        value = torch.zeros(1)
        with self.assertRaises(ValueError):
            dpo_loss(value, value, value, value, preference=torch.full_like(value, float("nan")))

    def test_reinforce_and_baseline_are_exact(self):
        logits = torch.tensor([.3, -.2, .1], dtype=torch.float64, requires_grad=True)
        reward = torch.tensor([0., 1., 3.], dtype=torch.float64)
        (gradient,) = torch.autograd.grad((logits.softmax(-1) * reward).sum(), logits)
        torch.testing.assert_close(gradient, exact_gradient(logits, reward))
        for baseline in (0., 1.3, 8.):
            torch.testing.assert_close(estimator_moments(logits.detach(), reward, baseline)["mean"], gradient)

    def test_rloo_and_self_mean_scaling(self):
        logits = torch.tensor([.3, -.2, .1], dtype=torch.float64)
        reward = torch.tensor([0., 1., 3.], dtype=torch.float64)
        expected = exact_gradient(logits, reward)
        torch.testing.assert_close(group_estimator_moments(logits, reward, 3)["mean"], expected)
        torch.testing.assert_close(group_estimator_moments(logits, reward, 3, False)["mean"], expected * 2 / 3)

    def test_ppo_clips_only_improving_direction(self):
        ratio = torch.tensor([1.4, .6, 1.4, .6], dtype=torch.float64, requires_grad=True)
        advantage = torch.tensor([1., 1., -1., -1.], dtype=torch.float64)
        ppo_surrogate(ratio, advantage).sum().backward()
        torch.testing.assert_close(ratio.grad, torch.tensor([0., 1., -1., 0.], dtype=torch.float64))

    def test_kl_estimators_and_support(self):
        p = torch.tensor([.2, .3, .5], dtype=torch.float64)
        q = torch.tensor([.7, .2, .1], dtype=torch.float64)
        result = kl_estimators(p, q)
        self.assertAlmostEqual(result["k1"]["mean"], result["k3"]["mean"], places=12)
        self.assertTrue((result["k3"]["samples"] >= 0).all())
        self.assertNotAlmostEqual(result["k1"]["mean"], result["k2"]["mean"], places=3)
        with self.assertRaises(ValueError):
            kl_estimators(p, torch.tensor([0., .5, .5], dtype=torch.float64))


if __name__ == "__main__":
    unittest.main()
