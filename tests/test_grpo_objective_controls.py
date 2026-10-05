"""Independent derivative and collection contracts for matched GRPO controls."""
import math
import unittest
from unittest.mock import patch

import torch

from dongxi_llms.grpo_objective_controls import (
    CAP, GROUP_SIZE, MAX_ATTEMPTS, PAIRS, SUPPORT, analytic_logp_gradient,
    clipping_fixture, collect_group, collect_with_filter, component_advantages,
    objective, reduction_weights, response_components, score_paths,
)
from dongxi_llms.reasoning_controls import EOS, NUMBER_START, make_decoder


class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.mask = torch.tensor([[True, False, False], [True, True, True]])
        self.old = torch.full((2, 3), -2., dtype=torch.float64, requires_grad=True)
        self.logp = self.old.detach().clone().requires_grad_()
        self.adv = torch.tensor([1., -1.], dtype=torch.float64, requires_grad=True)

    def test_reduction_weights_are_independent_known_numbers(self):
        self.assertTrue(torch.equal(reduction_weights(self.mask, "response"),
            torch.tensor([[.5, 0, 0], [1/6, 1/6, 1/6]], dtype=torch.float64)))
        self.assertTrue(torch.equal(reduction_weights(self.mask, "token"),
            torch.tensor([[.25, 0, 0], [.25, .25, .25]], dtype=torch.float64)))
        self.assertTrue(torch.equal(reduction_weights(self.mask, "fixed", 3),
            torch.tensor([[1/6, 0, 0], [1/6, 1/6, 1/6]], dtype=torch.float64)))

    def test_initial_ratio_analytic_gradient_and_detach(self):
        for reduction in ("response", "token", "fixed"):
            loss = objective(self.logp, self.old, self.adv, self.mask,
                             reduction=reduction, fixed_cap=3)
            gradient, old, advantage = torch.autograd.grad(loss,
                (self.logp, self.old, self.adv), allow_unused=True)
            expected = -reduction_weights(self.mask, reduction, 3) * self.adv.detach()[:, None]
            self.assertTrue(torch.allclose(gradient, expected, atol=1e-14))
            self.assertIsNone(old)
            self.assertIsNone(advantage)

    def test_response_zero_value_is_nonzero_gradient(self):
        loss = objective(self.logp, self.old, self.adv, self.mask)
        gradient, = torch.autograd.grad(loss, self.logp)
        self.assertAlmostEqual(float(loss.detach()), 0., places=14)
        self.assertGreater(float(gradient.norm()), .5)

    def test_asymmetric_clipping_known_signs(self):
        ratios = torch.tensor([.6, .9, 1.1, 1.3, 1.6], dtype=torch.float64)
        mask = torch.ones((2, 5), dtype=torch.bool)
        old = torch.full((2, 5), -2., dtype=torch.float64)
        logp = (old + ratios.log()).requires_grad_()
        adv = torch.tensor([1., -1.], dtype=torch.float64)
        loss = objective(logp, old, adv, mask, clip_high=.4)
        gradient, = torch.autograd.grad(loss, logp)
        expected = torch.stack((-ratios * torch.tensor([1, 1, 1, 1, 0]),
                                 ratios * torch.tensor([0, 1, 1, 1, 1]))) / 10
        self.assertTrue(torch.allclose(gradient, expected, atol=1e-14))
        self.assertTrue(torch.allclose(gradient,
            analytic_logp_gradient(logp, old, adv, mask, clip_high=.4), atol=1e-14))

    def test_padding_nonfinite_has_no_value_or_gradient(self):
        baseline = objective(self.logp, self.old, self.adv, self.mask)
        current = torch.cat((self.logp.detach(), torch.full((2, 2), float("nan"))), -1).requires_grad_()
        old = torch.cat((self.old.detach(), torch.full((2, 2), float("inf"))), -1)
        mask = torch.cat((self.mask, torch.zeros((2, 2), dtype=torch.bool)), -1)
        for reduction in ("response", "token", "fixed"):
            actual = objective(current, old, self.adv, mask, reduction=reduction, fixed_cap=3)
            expected = objective(self.logp, self.old, self.adv, self.mask,
                                 reduction=reduction, fixed_cap=3)
            self.assertTrue(torch.allclose(actual, expected))
            gradient, = torch.autograd.grad(actual, current)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertTrue(torch.equal(gradient[:, 3:], torch.zeros((2, 2), dtype=torch.float64)))
        self.assertTrue(torch.isfinite(baseline))

    def test_fixed_denominator_cannot_follow_padded_width(self):
        self.assertAlmostEqual(float(reduction_weights(self.mask, "fixed", 8).sum()), .25)
        with self.assertRaises(ValueError):
            reduction_weights(self.mask, "fixed", 2)
        for cap in (None, True, 0, -1, 3.5):
            with self.assertRaises(ValueError):
                reduction_weights(self.mask, "fixed", cap)

    def test_bad_masks_inputs_clip_and_reduction_rejected(self):
        for mask in (torch.tensor([[True, False, True]]), torch.zeros((1, 3), dtype=torch.bool),
                     torch.ones((1, 3)), torch.zeros((0, 3), dtype=torch.bool)):
            with self.assertRaises(ValueError):
                reduction_weights(mask, "response")
        for clip in (0, 1, True, float("nan")):
            with self.assertRaises(ValueError):
                objective(self.logp, self.old, self.adv, self.mask, clip_high=clip)
        with self.assertRaises(ValueError):
            objective(self.logp, self.old, self.adv, self.mask, reduction="padding_mean")
        bad = self.logp.detach().clone(); bad[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            objective(bad, self.old, self.adv, self.mask)
        huge = self.logp.detach().clone(); huge[0, 0] = 1000
        with self.assertRaises(ValueError):
            objective(huge, self.old, self.adv, self.mask)
        huge[0, 0] = -1000
        with self.assertRaises(ValueError):
            objective(huge, self.old, self.adv, self.mask)

    def test_fixture_explicitly_labeled_and_initial_clip_equal(self):
        result = clipping_fixture()
        self.assertEqual(result["kind"], "constructed-log-probability-fixture")
        self.assertEqual(result["symmetric"]["gradient"][0][3], 0)
        self.assertLess(result["asymmetric"]["gradient"][0][3], 0)
        self.assertTrue(torch.equal(
            objective(self.logp, self.old, self.adv, self.mask, clip_high=.2),
            objective(self.logp, self.old, self.adv, self.mask, clip_high=.4)))


class ComponentTests(unittest.TestCase):
    def test_population_moments_and_constant_component(self):
        components = torch.tensor([[[0., 5.], [0., 5.], [1., 5.], [1., 5.]]],
                                  dtype=torch.float64, requires_grad=True)
        result = component_advantages(components, (1., 2.), "component_std")
        expected = torch.tensor([[-.5, -.5, .5, .5]], dtype=torch.float64) / (.5 + 1e-8)
        self.assertTrue(torch.allclose(result, expected, atol=1e-14))
        self.assertFalse(result.requires_grad)

    def test_all_zero_and_all_one_have_no_relative_signal(self):
        for value in (0., 1.):
            components = torch.full((2, 8, 3), value, dtype=torch.float64)
            for scaling in ("center", "total_std", "component_std"):
                self.assertTrue(torch.equal(component_advantages(components, scaling=scaling),
                                            torch.zeros((2, 8), dtype=torch.float64)))

    def test_total_versus_component_normalization_and_weight_scaling(self):
        components = torch.tensor([[[1., 0.], [0., 10.], [1., 0.], [0., 0.]]], dtype=torch.float64)
        total = component_advantages(components, (1., 1.), "total_std")
        component = component_advantages(components, (1., 1.), "component_std")
        self.assertFalse(torch.allclose(total, component))
        rescaled = component_advantages(components, (10., 10.), "component_std")
        self.assertTrue(torch.allclose(rescaled, component * 10, atol=1e-13))
        centered = component_advantages(components, (1., 1.), "center")
        expected = (components.sum(-1) - components.sum(-1).mean(-1, keepdim=True))
        self.assertTrue(torch.equal(centered, expected))

    def test_reward_validation(self):
        good = torch.zeros((2, 4, 3), dtype=torch.float64)
        for weights in ((1., 2.), (1., 2., float("inf"))):
            with self.assertRaises(ValueError):
                component_advantages(good, weights)
        for scaling in ("sample_std", "unknown"):
            with self.assertRaises(ValueError):
                component_advantages(good, scaling=scaling)
        with self.assertRaises(ValueError):
            component_advantages(torch.zeros((1, 1, 3), dtype=torch.float64))

    def test_raw_component_rescaling_not_same_as_weight_rescaling(self):
        components = torch.tensor([[[1., 0.], [0., 2.], [1., 2.], [0., 0.]]], dtype=torch.float64)
        changed = components.clone(); changed[..., 1] *= 100
        # Independent means/std: both columns are balanced binaries after scaling.
        expected = torch.tensor([[-1., -1.], [-1., 1.], [1., 1.], [1., -1.]], dtype=torch.float64)
        expected[:, 0] = torch.tensor([1., -1., 1., -1.])
        expected = expected.sum(-1)[None]
        before = component_advantages(components, (1., 1.), "component_std")
        after = component_advantages(changed, (1., 1.), "component_std")
        self.assertTrue(torch.allclose(before, expected, atol=1e-7, rtol=0))
        self.assertTrue(torch.allclose(after, expected, atol=1e-7, rtol=0))
        self.assertFalse(torch.allclose(
            component_advantages(components, (1., 1.), "total_std"),
            component_advantages(changed, (1., 1.), "total_std")))


class CollectionTests(unittest.TestCase):
    def test_actual_conditional_collection_is_reproducible_and_recomputed(self):
        torch.set_num_threads(1)
        model = make_decoder(704).double().eval()
        record = collect_group(model, (0, 0), seed=705, attempt=1)
        again = collect_group(model, (0, 0), seed=705, attempt=1)
        self.assertEqual(record, again)
        rows = record["responses"]
        prompt = torch.tensor([[1, 3, 6, 6]] * GROUP_SIZE)
        responses = torch.tensor([r["tokens"] for r in rows])
        mask = torch.tensor([r["valid_mask"] for r in rows])
        rescored = score_paths(model, prompt, responses, mask)
        self.assertEqual(record["valid_tokens"], sum(len(r["active_tokens"]) for r in rows))
        for index, row in enumerate(rows):
            self.assertTrue(all(t in SUPPORT for t in row["active_tokens"]))
            self.assertTrue(torch.allclose(rescored[index, mask[index]],
                torch.tensor(row["behavior_logp"], dtype=torch.float64), atol=1e-14))
            if row["termination"] == "eos":
                self.assertEqual(row["active_tokens"][-1], EOS)
                self.assertNotIn(EOS, row["active_tokens"][:-1])
            else:
                self.assertEqual(len(row["active_tokens"]), CAP)
        bad = responses.clone(); bad[0, 0] = 9
        with self.assertRaises(ValueError):
            score_paths(model, prompt, bad, mask)

    def test_proxy_components_do_not_equal_independent_completed_quality(self):
        component, quality = response_components([NUMBER_START + 1] * 3, 1)
        self.assertEqual(component, [1., 0., 1.])
        self.assertEqual(quality, 0.)

    def test_scores_reach_shared_decoder_but_not_old_or_rewards(self):
        model = make_decoder(710).double().eval()
        prompt = torch.tensor([[1, 3, 6, 6], [1, 3, 6, 7]])
        response = torch.tensor([[6, 2], [7, 2]])
        mask = torch.ones_like(response, dtype=torch.bool)
        scores = score_paths(model, prompt, response, mask)
        old = scores.detach().clone().requires_grad_()
        rewards = torch.tensor([[[1., 1., 1/3], [0., 1., 1/3]]],
                               dtype=torch.float64, requires_grad=True)
        advantage = component_advantages(rewards).flatten()
        loss = objective(scores, old, advantage, mask)
        loss.backward()
        self.assertIsNone(old.grad)
        self.assertIsNone(rewards.grad)
        by_name = {name: parameter.grad for name, parameter in model.named_parameters()}
        self.assertTrue(any("token" in n and g is not None and bool(g.abs().sum() > 0)
                            for n, g in by_name.items()))
        self.assertTrue(any("q" in n and g is not None and bool(g.abs().sum() > 0)
                            for n, g in by_name.items()))
        self.assertTrue(any("mlp" in n and g is not None and bool(g.abs().sum() > 0)
                            for n, g in by_name.items()))
        component, quality = response_components([NUMBER_START, EOS], 1)
        self.assertEqual(component[:2], [0., 1.])
        self.assertEqual(quality, 0.)

    def test_retry_accounting_with_unresolved_and_mixed_groups(self):
        def fake(model, pair, seed, attempt):
            mixed = pair != (0, 0) and attempt == 2
            q = [0., 1.] * 4 if mixed else [0.] * 8
            return {"pair": list(pair), "attempt": attempt, "quality": q,
                    "valid_tokens": 16, "full_prefix_positions": 80,
                    "mixed_quality": mixed, "responses": [], "seed": seed}
        with patch("dongxi_llms.grpo_objective_controls.collect_group", side_effect=fake):
            records, costs = collect_with_filter(None, 1)
        self.assertEqual(len(records), 11)  # 5 + 5 + unresolved 1
        self.assertEqual(costs["responses_attempted"], 88)
        self.assertEqual(costs["groups_selected"], 4)
        self.assertEqual(costs["valid_tokens_attempted"], 176)
        self.assertEqual(costs["valid_tokens_selected"], 64)
        self.assertEqual(costs["unresolved_source_groups"], ["sum-positive-0-0"])
        self.assertTrue(all(r["attempt"] <= MAX_ATTEMPTS for r in records))

    def test_all_constant_groups_exhaust_budget_and_remain_in_denominator(self):
        def fake(model, pair, seed, attempt):
            return {"pair": list(pair), "attempt": attempt, "quality": [1.] * 8,
                    "valid_tokens": 8, "full_prefix_positions": 32,
                    "mixed_quality": False, "responses": [], "seed": seed}
        with patch("dongxi_llms.grpo_objective_controls.collect_group", side_effect=fake):
            records, costs = collect_with_filter(None, 1)
        self.assertEqual(len(records), len(PAIRS) * MAX_ATTEMPTS)
        self.assertEqual(costs["groups_selected"], 0)
        self.assertEqual(costs["valid_tokens_attempted"], 120)
        self.assertTrue(all(r["rejection_reason"] == "all-correct" for r in records))


if __name__ == "__main__":
    unittest.main()
