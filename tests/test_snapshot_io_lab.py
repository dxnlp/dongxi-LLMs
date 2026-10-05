import unittest
import torch
from dongxi_llms.snapshot_io_lab import reader_admission_example


class SnapshotIOLessonTests(unittest.TestCase):
    def test_actual_failed_load_retains_spending_and_refuses_next(self):
        result = reader_admission_example()
        self.assertTrue(result['numerical_state_equal'])
        self.assertTrue(result['later_spending_retained'])
        self.assertTrue(result['refused_before_new_ticket'])
        self.assertEqual(result['reserved']['snapshot_load_operations'], 3)
        self.assertEqual(result['completed']['snapshot_load_operations'], 2)
        self.assertEqual(result['attempted_upper']['snapshot_load_operations'], 3)
        self.assertEqual(result['uncertain_upper']['snapshot_load_operations'], 1)
        self.assertEqual(result['receipt_prefix_sequence'], 0)
        self.assertEqual(len(result['failed_tickets']), 1)
        self.assertEqual(result['known_partial']['snapshot_hash_bytes'], result['payload_bytes'])
        self.assertEqual(result['completed']['snapshot_hash_bytes'], 4 * result['payload_bytes'])
        self.assertEqual(result['completed']['snapshot_clone_bytes'], 32)

    def test_lesson_does_not_consume_live_global_randomness(self):
        before = torch.get_rng_state().clone()
        reader_admission_example()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))


if __name__ == '__main__': unittest.main()
