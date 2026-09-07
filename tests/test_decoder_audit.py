import copy
import unittest

import torch

from dongxi_llms.decoder_lab import DecoderConfig, TinyDecoder, parameter_count, teaching_batch
from dongxi_llms.decoder_audit import (
    parameter_ledger, trace_decoder, boundary_report, WrongAxisCenter,
)


class DecoderAuditTests(unittest.TestCase):
    def setUp(self):
        self.rng = torch.get_rng_state()
        torch.manual_seed(505)
        self.model = TinyDecoder(DecoderConfig(modern=True, qk_norm=True, kv_heads=2)).double().eval()
        self.ids, _ = teaching_batch()

    def tearDown(self):
        torch.set_rng_state(self.rng)

    def test_ledger_counts_unique_weights(self):
        for tied in (True, False):
            model = TinyDecoder(DecoderConfig(tied=tied))
            self.assertEqual(sum(parameter_ledger(model).values()), parameter_count(model))
        self.assertEqual(sum(parameter_ledger(self.model).values()), 4960)

    def test_trace_preserves_model_rng_and_hooks(self):
        rng = torch.get_rng_state().clone()
        state = copy.deepcopy(self.model.state_dict())
        rows = trace_decoder(self.model, self.ids)
        self.assertEqual(rows[-1]['output_shape'], [2, 6, 16])
        self.assertTrue(all(r['finite'] for r in rows))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, tensor in self.model.state_dict().items():
            torch.testing.assert_close(tensor, state[name])
        self.assertTrue(all(not m._forward_hooks for m in self.model.modules()))
        with self.assertRaises(ValueError):
            trace_decoder(self.model, self.ids[:, :0])
        self.assertTrue(all(not m._forward_hooks for m in self.model.modules()))

    def test_clean_and_broken_cases_are_distinguishable(self):
        clean = boundary_report(self.model, self.ids)
        wrong_offset = boundary_report(self.model, self.ids, rope_offset=0)
        leaky = copy.deepcopy(self.model)
        leaky.blocks[0].norm1 = WrongAxisCenter()
        wrong_axis = boundary_report(leaky, self.ids)
        self.assertTrue(all(clean[k] for k in ('finite','causal','cache')))
        self.assertTrue(wrong_offset['causal'])
        self.assertFalse(wrong_offset['cache'])
        self.assertTrue(wrong_axis['finite'])
        self.assertFalse(wrong_axis['causal'])
        self.assertFalse(wrong_axis['cache'])

    def test_audit_requires_valid_boundaries_and_eval(self):
        with self.assertRaises(ValueError):
            boundary_report(self.model, self.ids, prefix=0)
        self.model.train()
        with self.assertRaises(ValueError):
            boundary_report(self.model, self.ids)
