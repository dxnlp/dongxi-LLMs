import unittest

from dongxi_llms.systems_performance_lab import (
    MatmulLedger, attention_ledger, checkpoint_audit, decoder_phase_ledger, roofline, time_calls,
)
from dongxi_llms.decoder_lab import DecoderConfig


class SystemsPerformanceTests(unittest.TestCase):
    def test_matmul_closed_count_and_units(self):
        report = MatmulLedger(2, 3, 4, 8).report()
        self.assertEqual(report['matmul_flops'], 48)
        self.assertEqual(report['modeled_minimum_io_bytes'], (6+12+8)*8)
        self.assertEqual(report['arithmetic_intensity'], 48/208)

    def test_attention_read_policy(self):
        report = attention_ledger(1, 2, 5, 7, 3, query_tile=2, element_bytes=8)
        self.assertEqual(report['dense_score_value_matmul_flops'], 4*2*5*7*3)
        self.assertEqual(report['modeled_dense_io_bytes'], (30+84+30+280)*8)
        self.assertEqual(report['modeled_blockwise_io_bytes'], (30+252+30+10)*8)

    def test_roofline_two_constraints(self):
        self.assertEqual(roofline(2, 100, 10), 20)
        self.assertEqual(roofline(20, 100, 10), 100)
        for value in [0, -1, float('nan'), True]:
            with self.assertRaises(ValueError):
                roofline(value, 100, 10)

    def test_invalid_dimensions(self):
        for value in [0, -1, 2.5, True]:
            with self.assertRaises(ValueError):
                MatmulLedger(value, 3, 4)

    def test_cached_query_adds_attention_work_and_kv_reads_only(self):
        c = DecoderConfig(vocab=16, width=16, heads=4, kv_heads=4, head_dim=4,
                          layers=2, hidden=32, max_length=16, modern=True, qk_norm=False)
        short = decoder_phase_ledger(c, 2, 3, element_bytes=8)
        long = decoder_phase_ledger(c, 2, 7, element_bytes=8)
        self.assertEqual(long['cached_new_token_matmul_flops']-short['cached_new_token_matmul_flops'], 4*2*2*4*4*4)
        self.assertEqual(long['decode_cache_read_bytes']-short['decode_cache_read_bytes'], 2*2*2*4*4*4*8)
        self.assertEqual(short['decode_cache_append_bytes'], long['decode_cache_append_bytes'])
        with self.assertRaises(ValueError):
            decoder_phase_ledger(c, 2, 16)

    def test_synchronization_surrounds_every_timed_call(self):
        events = []
        result = time_calls(lambda: events.append('call'), warmup=2, repeats=2,
                            synchronize=lambda: events.append('sync'))
        self.assertEqual(events, ['call', 'call', 'sync', 'sync', 'call', 'sync', 'sync', 'call', 'sync'])
        self.assertEqual(len(result['seconds']), 2)
        self.assertGreaterEqual(result['median_seconds'], 0)

    def test_existing_checkpoint_recomputes_and_preserves_gradients(self):
        result = checkpoint_audit()
        off, on = result['arms']
        self.assertEqual(off['forward_block_entries'], 2)
        self.assertEqual(on['forward_block_entries'], 2)
        self.assertEqual(off['recomputed_block_entries'], 0)
        self.assertEqual(on['recomputed_block_entries'], 2)
        self.assertLessEqual(result['loss_abs_difference'], 1e-12)
        self.assertLessEqual(result['gradient_max_abs_difference'], 1e-12)
        self.assertLess(on['unique_backing_storage_bytes'], off['unique_backing_storage_bytes'])


if __name__ == '__main__':
    unittest.main()
