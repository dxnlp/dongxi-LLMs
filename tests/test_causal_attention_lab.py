"""Causal invariants and counterexamples, independent of learned quality."""
import unittest

import torch
from torch.nn.functional import scaled_dot_product_attention

from dongxi_llms.causal_attention_lab import attention_trace, teaching_inputs


class CausalAttentionTest(unittest.TestCase):
    def test_matches_reference_for_different_value_widths(self):
        generator = torch.Generator().manual_seed(4)
        for t, dk, dv in [(1, 2, 3), (4, 3, 2), (7, 5, 4)]:
            q, k = [torch.randn(t, dk, generator=generator, dtype=torch.float64)
                    for _ in range(2)]
            v = torch.randn(t, dv, generator=generator, dtype=torch.float64)
            result = attention_trace(q, k, v)
            expected = scaled_dot_product_attention(
                q[None, None], k[None, None], v[None, None],
                is_causal=True, dropout_p=0.)[0, 0]
            torch.testing.assert_close(result['output'], expected, atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(result['weights'].sum(-1), torch.ones(t, dtype=q.dtype))
            self.assertEqual(result['weights'].triu(1).count_nonzero().item(), 0)

    def test_future_change_exposes_both_broken_modes(self):
        x, wq, wk, wv = teaching_inputs()
        changed = x.clone()
        changed[-1] = torch.tensor([8., -3., 4.])
        for mode in ['causal', 'post_softmax', 'zero_scores']:
            a = attention_trace(x @ wq, x @ wk, x @ wv, mode=mode)
            b = attention_trace(changed @ wq, changed @ wk, changed @ wv, mode=mode)
            difference = (a['output'][:-1] - b['output'][:-1]).abs().max().item()
            if mode == 'causal':
                self.assertLess(difference, 1e-12)
                # A prefix alone produces the same outputs as that prefix in a longer sequence.
                prefix = attention_trace((x @ wq)[:-1], (x @ wk)[:-1], (x @ wv)[:-1])
                torch.testing.assert_close(prefix['output'], a['output'][:-1])
            else:
                self.assertGreater(difference, 1e-3)

    def test_post_mask_renormalization_equivalence_and_underflow(self):
        scores = torch.tensor([0., 0., 10.], dtype=torch.float64)
        allowed = torch.tensor([True, True, False])
        post = scores.softmax(-1).masked_fill(~allowed, 0)
        expected = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
        torch.testing.assert_close(post / post.sum(), expected)
        scores[-1] = 10000.
        post = scores.softmax(-1).masked_fill(~allowed, 0)
        self.assertEqual(post.sum().item(), 0.)
        self.assertTrue(torch.isfinite(scores.masked_fill(~allowed, -torch.inf).softmax(-1)).all())


if __name__ == '__main__':
    unittest.main()
