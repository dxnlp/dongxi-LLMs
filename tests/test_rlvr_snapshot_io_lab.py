import random
import unittest

import torch

from dongxi_llms.rlvr_snapshot_io_lab import phase_recovery_example


class NativeRLVRReaderLessonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {seed:phase_recovery_example(seed) for seed in (2323, 2324)}

    def test_both_boundaries_replay_original_native_trajectory(self):
        for result in self.results.values():
            completed, pending = result['rows']
            self.assertEqual(completed['final_numerical_sha256'], pending['final_numerical_sha256'])
            for row in result['rows']:
                self.assertTrue(row['exact_recovery']); self.assertTrue(row['first_record_equal'])
                self.assertTrue(row['pending_ids_equal']); self.assertEqual(len(row['final_history']), 4)

    def test_pending_actions_apply_without_recollection(self):
        for result in self.results.values():
            completed, pending = result['rows']
            self.assertEqual((completed['source_cursor'], pending['source_cursor']), (2, 3))
            self.assertEqual((completed['new_collections_at_first_step'], pending['new_collections_at_first_step']), (1, 0))
            self.assertTrue(completed['sampling_rng_advanced_at_first_step'])
            self.assertFalse(pending['sampling_rng_advanced_at_first_step'])
            self.assertIsNone(completed['pending_ids']); self.assertIsNotNone(pending['pending_ids'])

    def test_same_two_update_cursor_keeps_later_independent_spending(self):
        for result in self.results.values():
            for row in result['rows']:
                self.assertEqual(row['completed_updates'], 2); self.assertEqual(row['restored_completed_updates'], 2)
                self.assertTrue(row['later_work_retained'])
                self.assertEqual(row['saved_io_prefix_sequence'], 0)
                self.assertGreater(row['runner_sequence_after_save_and_rejection'], row['saved_runner_prefix_sequence'])
                self.assertGreater(row['io_sequence_after_save_and_rejection'], row['saved_io_prefix_sequence'])

    def test_actual_shared_operations_include_rejected_load(self):
        for result in self.results.values():
            for row in result['rows']:
                self.assertEqual(row['io_reserved']['snapshot_save_operations'], 1)
                self.assertEqual(row['io_reserved']['snapshot_inspect_operations'], 1)
                self.assertEqual(row['io_reserved']['snapshot_load_operations'], 2)
                self.assertEqual(row['io_completed']['snapshot_load_operations'], 1)
                self.assertEqual(len(row['failed_io_tickets']), 1)
                self.assertEqual(row['work_completed']['train_updates'], 4)
                self.assertEqual(row['work_completed']['collections'], 4)

    def test_callable_preserves_live_python_and_torch_randomness(self):
        python = random.getstate(); tensor = torch.get_rng_state().clone()
        phase_recovery_example(2323)
        self.assertEqual(python, random.getstate()); self.assertTrue(torch.equal(tensor, torch.get_rng_state()))

    def test_undeclared_seed_refuses_before_randomness_changes(self):
        python = random.getstate(); tensor = torch.get_rng_state().clone()
        with self.assertRaises(ValueError): phase_recovery_example(True)
        with self.assertRaises(ValueError): phase_recovery_example(19)
        self.assertEqual(python, random.getstate()); self.assertTrue(torch.equal(tensor, torch.get_rng_state()))


if __name__ == '__main__': unittest.main()
