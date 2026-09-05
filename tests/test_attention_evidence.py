import unittest
import torch
from dongxi_llms.attention_evidence import (
    gradient_evidence, scaling_evidence, cache_evidence,
)


class AttentionEvidenceTest(unittest.TestCase):
    def test_chain_rule_and_finite_differences(self):
        r = gradient_evidence()
        self.assertLess(max(r['manual_errors'].values()), 1e-12)
        self.assertLess(r['finite_difference_error'], 1e-7)
        self.assertEqual(r['forbidden_score_gradient'], 0.)
        self.assertEqual(r['future_input_gradient'], 0.)
        self.assertGreater(r['prompt_input_gradient'], 0.)

    def test_detach_routing_or_values(self):
        routing = gradient_evidence('routing')['gradient_norms']
        self.assertIsNone(routing['W_Q'])
        self.assertIsNone(routing['W_K'])
        self.assertGreater(routing['W_V'], 0.)
        values = gradient_evidence('values')['gradient_norms']
        self.assertIsNone(values['W_V'])
        self.assertGreater(values['W_Q'], 0.)
        self.assertGreater(values['W_K'], 0.)

    def test_scaling_controls_spread_under_declared_iid_draws(self):
        rows = scaling_evidence()
        for r in rows:
            self.assertLess(abs(r['scaled_std'] - 1), .15)
            self.assertGreater(r['scaled_entropy'], r['raw_entropy'])
        self.assertGreater(rows[-1]['raw_std'], rows[0]['raw_std'] * 5)
        self.assertGreater(rows[-1]['scaled_jacobian_trace'], rows[-1]['raw_jacobian_trace'])

    def test_cache_equivalence_and_stale_prefix(self):
        for prefill in [1, 2, 4, 6]:
            r = cache_evidence(prefill)
            self.assertLess(max(r['decode_errors'], default=0.), 1e-12)
            self.assertEqual(r['final_shapes'], [[6, 4], [6, 4]])
            if r['cache_bytes']:
                self.assertEqual(r['cache_bytes'][-1], 768)
            self.assertEqual(r['first_layer_same_token_key_error'], 0.)
            self.assertGreater(r['second_layer_same_token_key_error'], 1e-6)
            self.assertGreater(r['stale_cache_logit_error'], 1e-6)

    def test_attention_weights_are_not_unique_output_explanations(self):
        v = torch.tensor([[2., 0.], [0., 2.], [1., 1.]], dtype=torch.float64)
        a = torch.tensor([.4, .4, .2], dtype=v.dtype)
        b = torch.tensor([.1, .1, .8], dtype=v.dtype)
        self.assertFalse(torch.equal(a, b))
        torch.testing.assert_close(a @ v, b @ v)


if __name__ == '__main__':
    unittest.main()
