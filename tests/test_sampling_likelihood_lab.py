"""Independent exact fixtures for DXI-13; no models, downloads or CUDA."""
import unittest

import torch

from dongxi_llms.sampling_likelihood_lab import (
    MissingSupportError, behavior_distribution, collect_categorical,
    exact_forward_kl, importance_expectation, importance_ratios,
    kl_gradient_audit, response_surrogate, sampling_support_experiment,
    target_log_probabilities, termination_masks,
)


class SamplingLikelihoodTests(unittest.TestCase):
    def setUp(self):
        self.p = torch.tensor([.55, .30, .15], dtype=torch.float64)

    def test_temperature_one_full_support_preserves_baseline(self):
        p, logp, keep = behavior_distribution(self.p.log())
        torch.testing.assert_close(p, self.p)
        torch.testing.assert_close(logp, self.p.log())
        self.assertTrue(bool(keep.all()))

    def test_top_k_then_top_p_retains_crossing_action(self):
        p, _, keep = behavior_distribution(self.p.log(), .5, 2, .9)
        expected = torch.tensor([.55**2, .3**2, 0.], dtype=torch.float64)
        torch.testing.assert_close(p, expected/expected.sum())
        self.assertEqual(keep.tolist(), [True, True, False])
        _, _, first = behavior_distribution(self.p.log(), 1., None, .5)
        self.assertEqual(first.tolist(), [True, False, False])
        # Threshold equality does not add a further token; crossing remains.
        _, _, crossing = behavior_distribution(torch.tensor([.6,.3,.1]).log(), top_p=.7)
        self.assertEqual(crossing.tolist(), [True, True, False])

    def test_tie_order_and_batched_normalization(self):
        logits = torch.tensor([[0., 0., 0.], [0., -2., 1.]], dtype=torch.float64)
        p, logp, keep = behavior_distribution(logits, top_k=2)
        self.assertEqual(keep[0].tolist(), [True, True, False])
        torch.testing.assert_close(p.sum(-1), torch.ones(2, dtype=torch.float64))
        self.assertTrue(bool(torch.isneginf(logp[~keep]).all()))
        shifted, _, _ = behavior_distribution(logits+1000., top_k=2)
        torch.testing.assert_close(p, shifted)

    def test_invalid_sampling_inputs_rejected(self):
        for kwargs in ({"temperature":0.}, {"temperature":float("nan")},
                       {"temperature":True}, {"top_k":0}, {"top_k":4},
                       {"top_k":True}, {"top_p":0.}, {"top_p":1.1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                behavior_distribution(self.p.log(), **kwargs)
        for logits in (torch.tensor([float("inf")]), torch.tensor([1,2]), torch.empty(0)):
            with self.assertRaises(ValueError):
                behavior_distribution(logits)
        for logits, temperature in ((torch.tensor([0.,-1000.]),1.),
                                    (torch.tensor([1e30,0.]),1e-30)):
            with self.assertRaises(ValueError):
                behavior_distribution(logits,temperature)

    def test_recorded_probabilities_detach_and_reproduce_samples(self):
        theta = self.p.log().requires_grad_(True)
        a = collect_categorical(theta, seed=55, temperature=.5, top_k=2, top_p=.9)
        b = collect_categorical(theta, seed=55, temperature=.5, top_k=2, top_p=.9)
        torch.testing.assert_close(a.actions, b.actions)
        torch.testing.assert_close(a.selected_log_probabilities, a.probabilities[a.actions].log())
        self.assertFalse(a.probabilities.requires_grad)
        self.assertFalse(a.selected_log_probabilities.requires_grad)
        self.assertEqual(a.portable()["temperature"], .5)
        with torch.no_grad():
            theta.add_(5.)
        torch.testing.assert_close(a.logits, self.p.log())

    def test_matching_distribution_ratios_one_temperature_matters(self):
        record = collect_categorical(self.p.log(), temperature=.5, top_k=2, top_p=.9)
        identical = target_log_probabilities(self.p.log(), .5, record.support).exp()
        ratio = importance_ratios(identical, record.probabilities)
        torch.testing.assert_close(ratio[record.support], torch.ones(2, dtype=torch.float64))
        wrong_temperature = target_log_probabilities(self.p.log(), 1., record.support).exp()
        self.assertFalse(torch.allclose(importance_ratios(wrong_temperature, record.probabilities)[record.support], ratio[record.support]))

    def test_exact_importance_value_gradient_matches_analytic(self):
        theta = self.p.log().requires_grad_(True)
        behavior, _, keep = behavior_distribution(theta, .5, 2, .9)
        target = target_log_probabilities(theta, 1., keep).exp()
        rewards = torch.tensor([0.,1.,4.], dtype=torch.float64, requires_grad=True)
        value = importance_expectation(target, behavior, rewards)
        gradient, = torch.autograd.grad(value, theta)
        expected_value = .3/(.55+.3)
        self.assertAlmostEqual(float(value.detach()), expected_value, places=14)
        expected_gradient = target.detach()*(rewards.detach()-expected_value)
        torch.testing.assert_close(gradient, expected_gradient)
        self.assertIsNone(rewards.grad)
        self.assertEqual(float(gradient[2]), 0.)

    def test_behavior_denominator_detaches_even_if_caller_makes_it_live(self):
        theta = self.p.log().requires_grad_(True)
        b = self.p.clone().requires_grad_(True)
        value = importance_expectation(theta.softmax(-1), b, torch.arange(3.,dtype=torch.float64))
        value.backward()
        self.assertIsNone(b.grad)
        self.assertGreater(float(theta.grad.abs().sum()), 0.)

    def test_missing_target_support_rejected_not_zeroed(self):
        behavior, _, _ = behavior_distribution(self.p.log(), top_k=2)
        with self.assertRaises(MissingSupportError):
            importance_ratios(self.p, behavior)
        with self.assertRaises(ValueError):
            importance_ratios(self.p*.9, self.p)

    def test_wrong_denominator_changes_value_and_gradient(self):
        result = sampling_support_experiment()
        direct = result["expectations"]["direct_conditional_target"]
        repair = result["expectations"]["correct_denominator"]
        wrong = result["expectations"]["wrong_raw_denominator"]
        self.assertAlmostEqual(direct["value"], repair["value"], places=14)
        torch.testing.assert_close(torch.tensor(direct["gradient"]), torch.tensor(repair["gradient"]))
        self.assertGreater(abs(direct["value"]-wrong["value"]), .05)
        self.assertFalse(torch.allclose(torch.tensor(direct["gradient"]), torch.tensor(wrong["gradient"])))
        self.assertAlmostEqual(result["missing_raw_target_mass"], .15)
        self.assertAlmostEqual(result["missing_raw_reward_contribution"], .6)

    def test_conditional_target_gradient_finite_and_mask_detached(self):
        theta = self.p.log().requires_grad_(True)
        mask = torch.tensor([True,False,True])
        logp = target_log_probabilities(theta, .5, mask)
        logp[0].backward()
        p = logp.detach().exp()
        expected = (torch.tensor([1.,0.,0.])-p)/.5
        torch.testing.assert_close(theta.grad, expected)
        with self.assertRaises(ValueError):
            target_log_probabilities(theta, support=torch.zeros_like(mask))

    def test_surrogate_gradient_old_advantage_reference_and_padding(self):
        theta = torch.zeros(1,3,3,dtype=torch.float64,requires_grad=True)
        old = torch.tensor([[-float("inf"), -1., float("nan")]],dtype=torch.float64,requires_grad=True)
        advantage = torch.tensor([[float("nan"),2.,float("nan")]],dtype=torch.float64,requires_grad=True)
        reference = torch.tensor([[[.2,.3,.4]]]*3,dtype=torch.float64).transpose(0,1).requires_grad_(True)
        mask = torch.tensor([[False,True,False]])
        loss = response_surrogate(theta, torch.tensor([[0,1,2]]), old, advantage,
                                  mask, reference_logits=reference, beta=.2)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(theta.grad).all()))
        self.assertGreater(float(theta.grad[0,1].abs().sum()), 0.)
        self.assertEqual(float(theta.grad[:,[0,2]].abs().sum()), 0.)
        self.assertIsNone(old.grad)
        self.assertIsNone(advantage.grad)
        self.assertIsNone(reference.grad)
        with self.assertRaises(ValueError):
            response_surrogate(theta, torch.tensor([[0,1,2]]), old, advantage, torch.zeros_like(mask))

    def test_exact_kl_analytic_gradient_and_detached_reference(self):
        theta = self.p.log().requires_grad_(True)
        reference = torch.tensor([.2,.5,.3],dtype=torch.float64).log().requires_grad_(True)
        value = exact_forward_kl(theta, reference)
        value.backward()
        p, logp, logq = theta.detach().softmax(-1), theta.detach().log_softmax(-1), reference.detach().log_softmax(-1)
        torch.testing.assert_close(theta.grad, p*(logp-logq-value.detach()))
        self.assertIsNone(reference.grad)
        masked = exact_forward_kl(theta, reference, support=torch.tensor([True,True,False]))
        gradient, = torch.autograd.grad(masked, theta)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertEqual(float(gradient[2]), 0.)

    def test_equal_kl_values_different_gradients_and_weighted_repair(self):
        results = kl_gradient_audit(self.p.log(),torch.tensor([.2,.5,.3],dtype=torch.float64).log())
        value = results["exact_kl"]["value"]
        for record in results.values():
            self.assertAlmostEqual(record["value"], value, places=14)
        actual = torch.tensor(results["exact_kl"]["gradient"],dtype=torch.float64)
        torch.testing.assert_close(torch.tensor(results["frozen_k1"]["gradient"]),torch.zeros(3),atol=1e-14,rtol=0.)
        torch.testing.assert_close(torch.tensor(results["frozen_k3"]["gradient"],dtype=torch.float64),self.p-torch.tensor([.2,.5,.3],dtype=torch.float64))
        self.assertFalse(torch.allclose(actual, torch.tensor(results["frozen_k3"]["gradient"],dtype=torch.float64)))
        for name in ("weighted_k1","weighted_k3"):
            torch.testing.assert_close(torch.tensor(results[name]["gradient"],dtype=torch.float64),actual)

    def test_kl_audit_rejects_underflow_in_either_full_support_distribution(self):
        ordinary = torch.zeros(2, dtype=torch.float64)
        underflow = torch.tensor([0., -1000.], dtype=torch.float64)
        for target, reference in ((underflow, ordinary), (ordinary, underflow),
                                  (underflow, underflow)):
            with self.subTest(target=target.tolist(), reference=reference.tolist()):
                with self.assertRaisesRegex(ValueError, "probability underflowed"):
                    kl_gradient_audit(target, reference)

    def test_kl_audit_rejects_unrepresentable_k3_but_keeps_tiny_matching_support(self):
        tiny = torch.tensor([0., -744.], dtype=torch.float64)
        ordinary = torch.zeros(2, dtype=torch.float64)
        self.assertGreater(float(tiny.softmax(-1)[1]), 0.)
        with self.assertRaisesRegex(ValueError, "k3 ratio overflowed"):
            kl_gradient_audit(tiny, ordinary)
        # A log-domain p/b avoids 1/b overflow when both actual distributions
        # have the same tiny but still representable full-support probability.
        matched = kl_gradient_audit(tiny, tiny)
        for record in matched.values():
            self.assertEqual(record["value"], 0.)
            self.assertTrue(bool(torch.isfinite(torch.tensor(record["gradient"])).all()))
            torch.testing.assert_close(torch.tensor(record["gradient"]), torch.zeros(2))

    def test_exact_kl_excludes_reference_overflow_before_arithmetic(self):
        target = torch.zeros(2, dtype=torch.float64, requires_grad=True)
        reference = torch.tensor([1e308, -1e308], dtype=torch.float64, requires_grad=True)
        value = exact_forward_kl(target, reference, support=torch.tensor([True, False]))
        self.assertEqual(float(value.detach()), 0.)
        value.backward()
        torch.testing.assert_close(target.grad, torch.zeros(2, dtype=torch.float64))
        self.assertIsNone(reference.grad)
        with self.assertRaisesRegex(ValueError, "Retained reference log probabilities overflowed"):
            exact_forward_kl(target, reference, support=torch.tensor([False, True]))

    def test_exact_kl_rejects_retained_target_underflow(self):
        target = torch.tensor([0., -1000.], dtype=torch.float64)
        reference = torch.zeros(2, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "Retained target probability underflowed"):
            exact_forward_kl(target, reference)
        # Explicitly changing to a one-action conditional target is not an
        # underflow repair of the original full-support objective.
        value = exact_forward_kl(target, reference, support=torch.tensor([True, False]))
        self.assertAlmostEqual(float(value), float(torch.tensor(2., dtype=torch.float64).log()))

    def test_eos_prompt_padding_and_cap_masks(self):
        # EOS=pad=2; EOS inside the prompt is not generated termination.
        tokens = torch.tensor([[2,8,4,2,2,2],[8,9,4,5,6,2]])
        result = termination_masks(tokens,torch.tensor([2,2]),torch.tensor([4,5]),torch.tensor([3,3]),[2])
        self.assertEqual(result["response_mask"].tolist(),[[False,False,True,True,False,False],[False,False,True,True,True,False]])
        self.assertEqual(result["terminal"].tolist(),[True,False])
        self.assertEqual(result["truncated"].tolist(),[False,True])
        self.assertEqual(result["bootstrap_allowed"].tolist(),[False,True])

    def test_stop_at_first_response_and_ignore_post_stop(self):
        result = termination_masks(torch.tensor([[8,2,5,6]]),torch.tensor([1]),torch.tensor([4]),torch.tensor([3]),[2])
        self.assertEqual(result["response_mask"].tolist(),[[False,True,False,False]])

    def test_unclassified_early_end_and_empty_response_rejected(self):
        for length in (1,2):
            with self.assertRaises(ValueError):
                termination_masks(torch.tensor([[8,4,0]]),torch.tensor([1]),torch.tensor([length]),torch.tensor([3]),[2])


if __name__ == "__main__":
    unittest.main()
