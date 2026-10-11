import math
import unittest

import torch
from torch.nn import functional as F

from dongxi_llms.online_attention_lab import (
    blockwise_attention, dense_attention, forward_workspace, unstable_attention,
)


class OnlineAttentionTests(unittest.TestCase):
    def fixture(self, dtype=torch.float64):
        generator = torch.Generator().manual_seed(6505)
        return [torch.randn(shape, generator=generator, dtype=dtype, requires_grad=True)
                for shape in [(2, 2, 5, 3), (2, 2, 7, 3), (2, 2, 7, 4)]]

    def test_independent_sdpa_outputs_gradients_ragged_masks_and_tiles(self):
        for dtype, tolerance in [(torch.float64, 1e-11), (torch.float32, 3e-6)]:
            for qt, kt in [(1, 1), (2, 3), (8, 9)]:
                q, k, v = self.fixture(dtype)
                qi = torch.arange(5)[None, :, None]
                ki = torch.arange(7)[None, None, :]
                mask = (qi < torch.tensor([4, 5])[:, None, None]) & (ki < torch.tensor([6, 0])[:, None, None]) & (ki <= qi+2)
                expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None], dropout_p=0.)
                actual, lse, _ = blockwise_attention(q, k, v, query_tile=qt, key_tile=kt,
                                                     query_lengths=[4, 5], key_lengths=[6, 0], query_offset=2)
                self.assertTrue(torch.allclose(actual, expected, atol=tolerance, rtol=tolerance))
                expected_grad = torch.autograd.grad(expected.square().sum(), (q, k, v), retain_graph=True)
                actual_grad = torch.autograd.grad(actual.square().sum(), (q, k, v))
                for a, b in zip(actual_grad, expected_grad):
                    self.assertTrue(torch.isfinite(a).all())
                    self.assertTrue(torch.allclose(a, b, atol=tolerance, rtol=tolerance))
                self.assertTrue(torch.isneginf(lse[1]).all())
                self.assertTrue(torch.isneginf(lse[0, :, 4]).all())

    def test_cache_offset_and_future_invariance(self):
        q, k, v = self.fixture()
        before, _, _ = blockwise_attention(q, k, v)
        changed_v = v.detach().clone(); changed_v[:, :, 3:] += 100
        after, _, _ = blockwise_attention(q, k, changed_v)
        self.assertTrue(torch.equal(before[:, :, :3], after[:, :, :3]))
        q = q[:, :, :2]
        result, _, _ = blockwise_attention(q, k, v, query_offset=5)
        mask = torch.arange(7)[None, :] <= torch.arange(2)[:, None]+5
        reference = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        self.assertTrue(torch.allclose(result, reference, atol=1e-11, rtol=1e-11))

    def test_log_normalizer_against_direct_reduction(self):
        q, k, v = self.fixture()
        _, lse, _ = blockwise_attention(q, k, v, causal=False, key_tile=2)
        expected = (q@k.transpose(-1, -2)/math.sqrt(3)).logsumexp(-1)
        self.assertTrue(torch.allclose(lse, expected, atol=1e-12, rtol=1e-12))

    def test_large_late_maximum_exposes_both_broken_variants(self):
        q = torch.ones(1, 1, 1, 1, dtype=torch.float64)
        k = torch.tensor([0., 1000.], dtype=torch.float64).reshape(1, 1, 2, 1)
        v = torch.tensor([0., 1.], dtype=torch.float64).reshape(1, 1, 2, 1)
        correct, _, trace = blockwise_attention(q, k, v, causal=False, key_tile=1, trace=True)
        broken, _, _ = blockwise_attention(q, k, v, causal=False, key_tile=1, rescale_old=False)
        self.assertEqual(float(correct), 1.)
        self.assertEqual(float(broken), .5)
        self.assertFalse(torch.isfinite(unstable_attention(q, k, v)).all())
        self.assertEqual(trace[-1]['row_max'], [1000.])

    def test_fully_masked_dense_contract_and_zero_gradients(self):
        q, k, v = self.fixture()
        output, lse = dense_attention(q, k, v, key_lengths=[0, 0])
        self.assertEqual(float(output.detach().abs().sum()), 0.)
        self.assertTrue(torch.isneginf(lse).all())
        for gradient in torch.autograd.grad(output.sum(), (q, k, v)):
            self.assertEqual(float(gradient.abs().sum()), 0.)

    def test_gradcheck(self):
        generator = torch.Generator().manual_seed(3)
        inputs = tuple(torch.randn(1, 1, 2, 2, generator=generator, dtype=torch.float64, requires_grad=True) for _ in range(3))
        self.assertTrue(torch.autograd.gradcheck(lambda *args: blockwise_attention(*args, key_tile=1)[0], inputs))

    def test_workspace_and_invalid_boundaries(self):
        report = forward_workspace(1, 2, 5, 7, 3, query_tile=2, key_tile=3)
        self.assertEqual(report['dense_score_probability_bytes'], 2*2*5*7*8)
        self.assertEqual(report['block_score_weight_bytes'], 2*2*2*3*8)
        q, k, v = self.fixture()
        for kwargs in [dict(key_tile=0), dict(query_offset=-1), dict(key_lengths=[8, 7]), dict(query_lengths=[4.5, 5.])]:
            with self.assertRaises(ValueError):
                blockwise_attention(q, k, v, **kwargs)
        with self.assertRaises(ValueError):
            blockwise_attention(q, k.float(), v)


if __name__ == '__main__':
    unittest.main()
