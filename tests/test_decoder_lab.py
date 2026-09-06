import copy
from dataclasses import replace
import unittest

import torch
from torch.nn import functional as F

from dongxi_llms.decoder_lab import (
    DecoderConfig, TinyDecoder, DecoderBlock, MLP, attend, rope,
    layer_norm, rms_norm, analytical_parameters, parameter_count,
    cost_estimate, fit_one_batch, teaching_batch, next_token_loss,
)


class DecoderLabTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(505)
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.old_threads)

    def close(self, a, b):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-8)

    def test_layernorm_forward_backward(self):
        x = torch.randn(2, 6, 16, dtype=torch.float64, requires_grad=True)
        w = torch.randn(16, dtype=x.dtype, requires_grad=True)
        b = torch.randn(16, dtype=x.dtype, requires_grad=True)
        ours = layer_norm(x, w, b)
        reference = F.layer_norm(x, (16,), w, b, 1e-5)
        self.close(ours, reference)
        probe = torch.randn_like(x)
        a = torch.autograd.grad((ours*probe).sum(), (x, w, b))
        z = torch.autograd.grad((reference*probe).sum(), (x, w, b))
        for actual, expected in zip(a, z):
            self.close(actual, expected)

    def test_rmsnorm_reference_and_gradcheck(self):
        x = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
        w = torch.randn(4, dtype=x.dtype, requires_grad=True)
        self.close(rms_norm(x, w), F.rms_norm(x, (4,), w, 1e-6))
        self.assertTrue(torch.autograd.gradcheck(rms_norm, (x, w)))

    def test_rope_geometry_and_gradcheck(self):
        x = torch.randn(1, 2, 3, 4, dtype=torch.float64, requires_grad=True)
        y = torch.randn_like(x)
        p = torch.arange(3)
        self.close(rope(x, p).square().sum(-1), x.square().sum(-1))
        self.close((rope(x, p+7)*rope(y, p+11)).sum(-1),
                   (x*rope(y, torch.full((3,), 4))).sum(-1))
        self.assertTrue(torch.autograd.gradcheck(lambda t: rope(t, p), (x,)))

    def test_grouped_reference_and_gradients(self):
        q = torch.randn(2, 4, 6, 4, dtype=torch.float64, requires_grad=True)
        k = torch.randn(2, 2, 6, 4, dtype=q.dtype, requires_grad=True)
        v = torch.randn_like(k, requires_grad=True)
        actual, a = attend(q, k, v)
        expected = F.scaled_dot_product_attention(q, k.repeat_interleave(2, 1),
                    v.repeat_interleave(2, 1), is_causal=True)
        self.close(actual, expected)
        probe = torch.randn_like(actual)
        ga = torch.autograd.grad((actual*probe).sum(), (q, k, v))
        gb = torch.autograd.grad((expected*probe).sum(), (q, k, v))
        for left, right in zip(ga, gb):
            self.close(left, right)
        self.assertEqual(float(a.triu(1).abs().max().detach()), 0.)

    def test_mlp_positionwise_and_affine_collapse(self):
        mlp = MLP(16, 32).double()
        x = torch.randn(2, 6, 16, dtype=torch.float64)
        changed = x.clone()
        changed[:, 3] += 1
        self.close(mlp(x)[:, :3], mlp(changed)[:, :3])
        combined = F.linear(x, mlp.down.weight @ mlp.up.weight,
                            mlp.down.weight @ mlp.up.bias + mlp.down.bias)
        self.close(mlp.down(mlp.up(x)), combined)

    def test_zero_branch_pre_post_norm(self):
        cfg = DecoderConfig()
        pre, post = DecoderBlock(cfg).double(), DecoderBlock(cfg, post_norm=True).double()
        for block in (pre, post):
            with torch.no_grad():
                for p in block.attn.parameters():
                    p.zero_()
                for p in block.mlp.parameters():
                    p.zero_()
        x = torch.randn(2, 6, 16, dtype=torch.float64, requires_grad=True)
        out = pre(x)[0]
        self.close(out, x)
        self.close(torch.autograd.grad(out.sum(), x)[0], torch.ones_like(x))
        self.assertFalse(torch.allclose(post(x)[0], x))

    def test_counts_for_variants(self):
        for modern in (False, True):
            for tied in (False, True):
                for kv in (1, 2, 4):
                    cfg = DecoderConfig(modern=modern, tied=tied, kv_heads=kv, qk_norm=True)
                    self.assertEqual(parameter_count(TinyDecoder(cfg)), analytical_parameters(cfg))

    def test_causality_and_future_gradient(self):
        ids, labels = teaching_batch()
        for modern in (False, True):
            model = TinyDecoder(DecoderConfig(modern=modern, kv_heads=2, qk_norm=modern)).double()
            changed = ids.clone()
            changed[:, 4:] = 0
            self.close(model(ids)[:, :4], model(changed)[:, :4])
            states = model.token(ids).detach().requires_grad_()
            out = model.blocks[0](states)[0]
            g = torch.autograd.grad(out[:, 2].square().sum(), states)[0]
            self.assertEqual(float(g[:, 3:].abs().max()), 0.)
            loss = next_token_loss(model(ids), labels)
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_cache_chunks_offsets_and_bytes(self):
        ids, _ = teaching_batch()
        for modern in (False, True):
            for kv in (1, 2, 4):
                cfg = DecoderConfig(modern=modern, kv_heads=kv, qk_norm=modern)
                model = TinyDecoder(cfg).double().eval()
                with torch.no_grad():
                    expected = model(ids)
                    prefix, cache = model(ids[:, :2], return_cache=True)
                    middle, cache = model(ids[:, 2:4], caches=cache, return_cache=True)
                    last, cache = model(ids[:, 4:], caches=cache, return_cache=True)
                    self.close(torch.cat((prefix, middle, last), 1), expected)
                    payload = sum(t.numel()*t.element_size() for pair in cache for t in pair)
                    self.assertEqual(payload, cost_estimate(cfg, 2, 6, 8)['logical_kv_bytes'])
                    if modern:
                        _, first_cache = model(ids[:, :2], return_cache=True)
                        broken = model(ids[:, 2:], caches=first_cache, rope_offset=0)
                        self.assertGreater(float((broken-expected[:, 2:]).abs().max()), 1e-8)

    def test_recurrent_shared_gradient_sum(self):
        block = DecoderBlock(DecoderConfig()).double()
        first, second = copy.deepcopy(block), copy.deepcopy(block)
        x = torch.randn(2, 6, 16, dtype=torch.float64)
        shared = block(block(x)[0])[0]
        unrolled = second(first(x)[0])[0]
        self.close(shared, unrolled)
        shared.square().mean().backward()
        unrolled.square().mean().backward()
        for p, a, b in zip(block.parameters(), first.parameters(), second.parameters()):
            self.close(p.grad, a.grad+b.grad)

    def test_one_batch_learning(self):
        model, result = fit_one_batch()
        self.assertLess(result['final_loss'], .05)
        self.assertEqual(result['accuracy'], 1.)
        self.assertEqual(len(result['history']), 160)

    def test_invalid_geometry(self):
        with self.assertRaises(ValueError):
            DecoderConfig(heads=3, kv_heads=2)
        with self.assertRaises(ValueError):
            DecoderConfig(modern=True, head_dim=3)
        with self.assertRaises(ValueError):
            TinyDecoder()(torch.ones(2, 33, dtype=torch.long))


if __name__ == '__main__':
    unittest.main()
