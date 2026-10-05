"""The original native story-clock lesson, separate from model-scale evidence."""
import unittest

import torch

from dongxi_llms.stories_work_lab import work_recovery_example


class StoryWorkLessonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.example = work_recovery_example()

    def test_successful_targets_and_admitted_targets_are_different(self):
        row = self.example
        self.assertEqual((row['successful_targets'], row['reserved_targets']), (17, 25))
        self.assertEqual([r['successful_targets'] for r in row['timeline']], [9, 9, 9, 17])
        self.assertEqual([r['reserved_targets'] for r in row['timeline']], [9, 17, 17, 25])

    def test_numerical_continuation_remains_original(self):
        row = self.example
        self.assertTrue(row['original_record_equal'])
        self.assertTrue(row['exact_numerical_recovery'])
        self.assertEqual(row['original_numerical_sha256'], row['restored_numerical_sha256'])
        self.assertEqual((row['actual_successful_updates'], row['recipe_horizon']), (2, 5))

    def test_failed_attempt_is_not_refunded(self):
        self.assertTrue(self.example['later_failed_work_retained'])
        self.assertEqual(len(self.example['failed_tickets']), 1)
        self.assertEqual(self.example['reserved']['train_updates'], 3)
        self.assertEqual(self.example['completed']['train_updates'], 2)

    def test_next_whole_update_refuses_without_calls_or_mutation(self):
        row = self.example
        self.assertEqual(row['next_refusal_forward_calls'], 0)
        self.assertTrue(row['refusal_numerical_state_unchanged'])
        self.assertTrue(row['refusal_work_prefix_unchanged'])

    def test_caller_rng_is_restored(self):
        before = torch.get_rng_state().clone()
        work_recovery_example()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_lower_capacity_refuses_the_same_retry(self):
        row = work_recovery_example(24)
        self.assertTrue(row['retry_refused'])
        self.assertEqual((row['successful_targets'], row['reserved_targets']), (9, 17))
        self.assertEqual(row['next_refusal_forward_calls'], 0)
        self.assertTrue(row['refusal_numerical_state_unchanged'])
        self.assertTrue(row['refusal_work_prefix_unchanged'])
        self.assertEqual(row['timeline'][-1]['event'], 'refuse8')


if __name__ == '__main__':
    unittest.main()
