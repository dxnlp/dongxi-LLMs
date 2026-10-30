"""S2 checks pass the references, catch plausible mistakes without raising, and raise under strict=True."""
import contextlib
import copy
import io
import math
import unittest

import torch
import torch.nn.functional as F

from dongxi_llms.study_path import s2_kl_optimum_and_dpo as s2


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        result = fn(*args, **kwargs)
    return result, out.getvalue()


class S2CheckTests(unittest.TestCase):
    def assertPasses(self, check, *objs):
        ok, out = quiet(check, *objs, strict=True)
        self.assertTrue(ok, out)
        self.assertNotIn("❌", out)

    def assertCatches(self, check, *objs, hint=None):
        ok, out = quiet(check, *objs)
        self.assertFalse(ok, out)
        self.assertIn("❌", out)
        if hint:
            self.assertIn(hint, out)
        with self.assertRaises(AssertionError):
            quiet(check, *objs, strict=True)

    def test_not_attempted_and_placeholders_never_raise(self):
        for check in (s2.check_kl, s2.check_optimal_policy, s2.check_objective, s2.check_dpo_loss,
                      s2.check_dpo_grads, s2.check_tabular_dpo_loss, s2.check_sequence_logps,
                      s2.check_dpo_step, s2.check_pair_accuracy, s2.check_kl_estimate):
            ok, out = quiet(check, None)
            self.assertFalse(ok)
            self.assertIn("not attempted", out)
        ok, out = quiet(s2.check_implicit_reward, None, None)
        self.assertIn("not attempted", out)
        ok, out = quiet(s2.check_kl, lambda p, q: (p * ...).sum())
        self.assertFalse(ok)
        self.assertIn("raised", out)

    def test_kl(self):
        self.assertPasses(s2.check_kl, s2.ref_kl)
        self.assertAlmostEqual(float(s2.ref_kl(s2.D35_PI, s2.D35_REF)), 0.0851, places=4)
        self.assertCatches(s2.check_kl, lambda p, q: s2.ref_kl(q, p), hint="swapped")
        self.assertCatches(s2.check_kl, lambda p, q: (p.log() - q.log()).sum(-1))
        self.assertCatches(s2.check_kl, lambda p, q: (p * (p / q).log()).sum(), hint="sum over the last")

    def test_optimal_policy(self):
        self.assertPasses(s2.check_optimal_policy, s2.ref_optimal_policy)
        self.assertCatches(s2.check_optimal_policy, lambda ref, R, b: torch.softmax(R / b, -1), hint="dropped the reference")
        self.assertCatches(s2.check_optimal_policy, lambda ref, R, b: ref * torch.exp(R / b), hint="normalize")
        self.assertCatches(s2.check_optimal_policy, lambda ref, R, b: torch.softmax(ref.log() + b * R, -1), hint="divide")

    def test_objective(self):
        J = lambda pi, ref, R, b: s2.ref_objective(pi, ref, R, b)
        self.assertPasses(s2.check_objective, J)
        self.assertCatches(s2.check_objective, lambda pi, ref, R, b: (pi * R).sum(-1) + b * s2.ref_kl(pi, ref), hint="cost")
        self.assertCatches(s2.check_objective, lambda pi, ref, R, b: (pi * R).sum(-1) - b * s2.ref_kl(ref, pi), hint="direction")

    def test_implicit_reward(self):
        self.assertPasses(s2.check_implicit_reward, s2.ref_implicit_reward, s2.ref_pair_prob)
        self.assertCatches(s2.check_implicit_reward, lambda pi, ref, b: pi.log() - ref.log(), s2.ref_pair_prob)
        self.assertCatches(s2.check_implicit_reward, s2.ref_implicit_reward, lambda s, i, j: torch.sigmoid(s[j] - s[i]),
                           hint="order")

    def test_dpo_loss(self):
        self.assertPasses(s2.check_dpo_loss, s2.ref_dpo_loss)

        def unstable(pol, ref, b):
            imp = b * (pol - ref)
            return -torch.log(torch.sigmoid(imp[:, 0] - imp[:, 1])).mean(), imp

        def no_ref(pol, ref, b):
            imp = b * (pol - ref)
            return -F.logsigmoid(b * (pol[:, 0] - pol[:, 1])).mean(), imp

        def flipped(pol, ref, b):
            imp = b * (pol - ref)
            return -F.logsigmoid(imp[:, 1] - imp[:, 0]).mean(), imp

        self.assertCatches(s2.check_dpo_loss, unstable, hint="logsigmoid")
        self.assertCatches(s2.check_dpo_loss, no_ref, hint="reference")
        self.assertCatches(s2.check_dpo_loss, flipped, hint="sign")

    def test_dpo_grads(self):
        self.assertPasses(s2.check_dpo_grads, s2.ref_dpo_grads)

        def wrong_weight(pol, ref, b):
            m = b * ((pol[:, 0] - ref[:, 0]) - (pol[:, 1] - ref[:, 1]))
            w = torch.sigmoid(m)
            return torch.stack([-b * w / 3, b * w / 3], 1)

        self.assertCatches(s2.check_dpo_grads, wrong_weight, hint="σ(−m)")
        self.assertCatches(s2.check_dpo_grads, lambda p, r, b: s2.ref_dpo_grads(p, r, b) * p.shape[0], hint="divide by B")

    def test_tabular(self):
        self.assertPasses(s2.check_tabular_dpo_loss, s2.ref_tabular_dpo_loss)
        swapped = lambda th, W, L, rl, b: s2.ref_tabular_dpo_loss(th, L, W, rl, b)
        self.assertCatches(s2.check_tabular_dpo_loss, swapped, hint="winner")
        detached = lambda th, W, L, rl, b: s2.ref_tabular_dpo_loss(th.detach(), W, L, rl, b)
        self.assertCatches(s2.check_tabular_dpo_loss, detached)

    def test_tabular_fit_matches_per_pair_loss_and_converges(self):
        W, L = s2.sample_pairs(200, s2.TOY_REWARD, seed=1)
        C = s2.pair_counts(W, L)
        theta = torch.randn(5, dtype=torch.float64)
        per_pair = s2.ref_tabular_dpo_loss(theta, W, L, s2.TOY_REF.log(), 0.7)
        imp = 0.7 * (theta.log_softmax(-1) - s2.TOY_REF.log())
        counts = (C * F.softplus(-(imp[:, None] - imp[None, :]))).sum() / C.sum()
        self.assertAlmostEqual(float(per_pair), float(counts), places=10)
        # infinite data (expected soft counts) → exactly π*
        q = torch.sigmoid(s2.TOY_REWARD[:, None] - s2.TOY_REWARD[None, :])
        soft = q * (1 - torch.eye(5, dtype=torch.float64))
        pi, _ = s2.fit_tabular_dpo(soft, s2.TOY_REF, s2.TOY_BETA, steps=3000)
        star = s2.ref_optimal_policy(s2.TOY_REF, s2.TOY_REWARD, s2.TOY_BETA)
        self.assertLess(float(s2.tv(pi, star)), 1e-3)

    def test_sequence_logps(self):
        self.assertPasses(s2.check_sequence_logps, s2.ref_sequence_logps)

        def no_shift(lg, ids, m):
            return (lg.log_softmax(-1).gather(-1, ids[..., None]).squeeze(-1) * m).sum(1)

        def no_mask(lg, ids, m):
            return lg[:, :-1].log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1).sum(1)

        def unshifted_mask(lg, ids, m):
            return (lg[:, :-1].log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1) * m[:, :-1]).sum(1)

        self.assertCatches(s2.check_sequence_logps, no_shift, hint="shift by one")
        self.assertCatches(s2.check_sequence_logps, no_mask, hint="prompt tokens")
        self.assertCatches(s2.check_sequence_logps, unshifted_mask, hint="answer_mask[:, 1:]")

    def test_dpo_step(self):
        self.assertPasses(s2.check_dpo_step, s2.ref_dpo_step)

        def ref_with_graph(policy, ref, batch, opt, beta, logps):
            ref_lp = logps(ref, batch)
            loss, implicit = s2.ref_dpo_loss(logps(policy, batch), ref_lp, beta)
            loss.backward(); opt.step(); opt.zero_grad()
            return loss.item(), implicit.detach()

        def swapped(policy, ref, batch, opt, beta, logps):
            with torch.no_grad():
                ref_lp = logps(ref, batch)
            loss, implicit = s2.ref_dpo_loss(ref_lp, logps(policy, batch), beta)
            loss.backward(); opt.step(); opt.zero_grad()
            return loss.item(), implicit.detach()

        self.assertCatches(s2.check_dpo_step, ref_with_graph, hint="no_grad")
        self.assertCatches(s2.check_dpo_step, swapped)

    def test_estimates(self):
        self.assertPasses(s2.check_pair_accuracy, s2.ref_pair_accuracy)
        strict_gt = lambda imp: ((imp[:, 0] > imp[:, 1]).float().mean(),
                                 ((imp[:, 0] > imp[:, 1]).float().mean() * 0.5 / imp.shape[0]).sqrt())
        self.assertCatches(s2.check_pair_accuracy, strict_gt, hint="tie")
        self.assertPasses(s2.check_kl_estimate, s2.ref_kl_estimate)
        unmasked = lambda a, b, m: ((a - b).sum(1).mean(), (a - b).sum(1).std() / math.sqrt(3), None)
        self.assertCatches(s2.check_kl_estimate, unmasked, hint="mask")

    def test_encode_pairs_matches_answer_logps_layout(self):
        try:
            from dongxi_llms.study_path import qwen as Q
            tok = Q.load_tokenizer()
        except Exception as error:  # tokenizer cache absent on this machine
            self.skipTest(f"Qwen3 tokenizer not cached: {error}")
        pairs = [{"prompt": [{"role": "user", "content": "Hi?"}], "chosen": "Hello.", "rejected": "Go away."}]
        b = s2.encode_pairs(tok, pairs)
        self.assertEqual(b["ids"].shape[0], 2)
        graded = tok.decode(b["ids"][0][b["amask"][0].bool()])
        self.assertEqual(graded, "Hello.<|im_end|>")
        # a tiny model through pair_logps: [1, 2] output, equals the masked sum
        model = s2.TinyLM(vocab=len(tok))
        out = s2.pair_logps(model, b)
        self.assertEqual(tuple(out.shape), (1, 2))


if __name__ == "__main__":
    unittest.main()
