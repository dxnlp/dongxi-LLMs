"""Tests for the transparent Day 3 next-token distribution experiment."""

import unittest

import torch

from dongxi_llms.next_token_distribution_lab import (
    empirical_distribution,
    loss_and_gradients,
    make_targets,
    run_experiment,
)


class NextTokenDistributionLabTest(unittest.TestCase):
    def test_target_counts_define_expected_distribution(self) -> None:
        targets = make_targets()

        self.assertEqual(targets.shape, (100,))
        self.assertTrue(
            torch.allclose(
                empirical_distribution(targets, classes=2),
                torch.tensor([0.7, 0.3]),
            )
        )

    def test_autograd_matches_p_minus_r(self) -> None:
        targets = make_targets()
        _, _, autograd_gradient, analytical_gradient = loss_and_gradients(
            torch.tensor([0.4, -0.2]), targets
        )

        self.assertTrue(torch.allclose(autograd_gradient, analytical_gradient, atol=1e-6))

    def test_training_recovers_empirical_distribution(self) -> None:
        result = run_experiment(steps=500, learning_rate=0.5, seed=42)

        self.assertTrue(result["all_criteria_passed"])
        final = result["checkpoints"][-1]
        self.assertAlmostEqual(final["probabilities"][0], 0.7, places=5)
        self.assertAlmostEqual(final["probabilities"][1], 0.3, places=5)


if __name__ == "__main__":
    unittest.main()
