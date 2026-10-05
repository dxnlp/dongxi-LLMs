"""Actual CPU generator state and bounded-input controls for the book lesson."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import torch

from dongxi_llms.recovery_rng_lab import sampler_verification


class RecoveryRNGTests(unittest.TestCase):
    def test_all_comparison_draws_are_included_in_measured_total(self):
        actual=torch.randint
        with patch.object(torch,'randint',wraps=actual) as draws:
            result=sampler_verification()
        self.assertEqual(draws.call_count,36)
        self.assertEqual(result['total_scalar_draws'],draws.call_count)
        self.assertEqual(sum(result['draws'].values()),draws.call_count)
        self.assertEqual(result['draws']['independent_comparison_future'],6)

    def test_private_replay_preserves_live_future_and_global_rng(self):
        before = torch.get_rng_state().clone()
        result = sampler_verification()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        for key in ("history_matches", "private_state_matches",
                    "live_unchanged_by_private_check", "safe_next_matches"):
            self.assertTrue(result[key])
        self.assertEqual(result["draws"]["private_validation"], 6)
        self.assertEqual(result["draws"]["safe_future_training"], 6)
        self.assertEqual(result["state_before_check_sha256"],
                         result["state_after_private_check_sha256"])

    def test_broken_live_check_consumes_real_state(self):
        result = sampler_verification()
        self.assertTrue(result["broken_live_advanced"])
        self.assertFalse(result["broken_check_matches_history"])
        self.assertNotEqual(result["state_before_check_sha256"],
                            result["state_after_broken_check_sha256"])
        self.assertEqual(result["draws"]["broken_check_consumed_live"], 6)

    def test_zero_history_does_not_invent_draws_or_damage(self):
        result = sampler_verification(completed_updates=0)
        self.assertEqual(result["history"], [])
        self.assertEqual(result["draws"]["private_validation"], 0)
        self.assertFalse(result["broken_live_advanced"])
        self.assertEqual(result["broken_next_ids"], result["safe_next_ids"])

    def test_deterministic_independent_returns(self):
        original = sampler_verification()
        first = deepcopy(original)
        first["history"].append(-1)
        self.assertEqual(original, sampler_verification())
        self.assertNotEqual(first, original)

    def test_caps_and_exact_integer_types(self):
        invalid = {"seed": [-1, 2**63, True], "completed_updates": [-1, 1001, 1.0],
                   "accumulation": [0, 17, True], "source_count": [1, 4097, 2.0],
                   "future_updates": [-1, 65, False]}
        for key, values in invalid.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    sampler_verification(**{key: value})


if __name__ == "__main__":
    unittest.main()
