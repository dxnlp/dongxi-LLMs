"""Study path S7: each check passes its reference and rejects a plausible mistake (CPU, seconds)."""
from __future__ import annotations

import contextlib
import io
import math
import unittest

import torch

from dongxi_llms.study_path import s7_failure_modes as s7


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        result = fn(*args, **kwargs)
    return result, out.getvalue()


class ChecksAcceptReferences(unittest.TestCase):
    def test_references_pass(self):
        cases = [
            (s7.check_tilt, s7.ref_tilt), (s7.check_kl, s7.ref_kl),
            (s7.check_bon_probs, s7.ref_bon_probs), (s7.check_bon_kl, s7.ref_bon_kl),
            (s7.check_length_controlled, s7.ref_length_controlled), (s7.check_entropy, s7.ref_entropy),
            (s7.check_pg_loss, s7.ref_pg_loss), (s7.check_aggregate, s7.ref_aggregate),
            (s7.check_diagnose, s7.ref_diagnose), (s7.check_masked_entropy, s7.ref_masked_entropy),
        ]
        for check, ref in cases:
            with self.subTest(check=check.__name__):
                ok, out = quiet(check, ref, strict=True)
                self.assertTrue(ok)
                self.assertNotIn("❌", out)
                self.assertIn("✅", out)

    def test_not_attempted(self):
        ok, out = quiet(s7.check_tilt, None)
        self.assertFalse(ok)
        self.assertIn("⏳", out)
        ok, out = quiet(s7.check_kl, Ellipsis)
        self.assertFalse(ok)


class ChecksRejectMistakes(unittest.TestCase):
    def rejects(self, check, fn, hint_fragment=None):
        ok, out = quiet(check, fn)
        self.assertFalse(ok)
        self.assertIn("❌", out)
        if hint_fragment:
            self.assertIn(hint_fragment, out)
        with self.assertRaises(AssertionError), contextlib.redirect_stdout(io.StringIO()):
            check(fn, strict=True)

    def test_tilt(self):
        self.rejects(s7.check_tilt, lambda lr, R, b: lr + R / b, "log_softmax")       # unnormalized
        self.rejects(s7.check_tilt, lambda lr, R, b: torch.log_softmax(lr + R * b, -1), "divide by β")
        self.rejects(s7.check_tilt, lambda lr, R, b: torch.softmax(lr + R / b, -1), "probabilities")

    def test_kl(self):
        self.rejects(s7.check_kl, lambda lp, lr: (lr.exp() * (lr - lp)).sum(-1), "KL(π_ref‖π)")
        self.rejects(s7.check_kl, lambda lp, lr: (lp - lr).sum(-1))                      # unweighted
        self.rejects(s7.check_kl, lambda lp, lr: (lp.exp() * (lp - lr)).sum(), ".sum(-1)")  # sums the batch

    def test_bon_probs(self):
        self.rejects(s7.check_bon_probs, lambda p, n: p ** n / (p ** n).sum())
        self.rejects(s7.check_bon_probs, lambda p, n: p.cumsum(0) ** n - (p.cumsum(0) - p) ** (n - 1))

    def test_bon_kl(self):
        self.rejects(s7.check_bon_kl, lambda n: math.log(n), "upper bound")
        self.rejects(s7.check_bon_kl, lambda n: math.log(n) + (n - 1) / n)

    def test_length_controlled(self):
        def unweighted(r, z, p):
            zc = z - z.mean()
            b = (zc * r).sum() / (zc ** 2).sum()
            return r - b * zc, b
        self.rejects(s7.check_length_controlled, unweighted, "weights p")

        def wrong_denominator(r, z, p):
            zc = z - (p * z).sum()
            rc = r - (p * r).sum()
            b = (p * zc * r).sum() / (p * rc ** 2).sum()
            return r - b * zc, b
        self.rejects(s7.check_length_controlled, wrong_denominator)

    def test_entropy(self):
        self.rejects(s7.check_entropy, lambda x: -(torch.softmax(x, -1) * torch.log(torch.softmax(x, -1))).sum(-1),
                     "log_softmax")
        self.rejects(s7.check_entropy, lambda x: (torch.softmax(x, -1) * torch.log_softmax(x, -1)).sum(-1))  # sign

    def test_pg_loss(self):
        self.rejects(s7.check_pg_loss, lambda lp, A, kl, H, b, l: -(A * lp).mean() + b * kl + l * H, "minus sign")
        self.rejects(s7.check_pg_loss, lambda lp, A, kl, H, b, l: -(A * lp).sum() + b * kl - l * H)

    def test_aggregate(self):
        def no_values(s, mode):
            return {"mean": s.mean(0), "min": s.min(0), "lcb": s.mean(0) - s.std(0, unbiased=False)}[mode]
        self.rejects(s7.check_aggregate, no_values)

        def unbiased(s, mode):
            return {"mean": s.mean(0), "min": s.min(0).values, "lcb": s.mean(0) - s.std(0)}[mode]
        self.rejects(s7.check_aggregate, unbiased, "population std")

    def test_diagnose(self):
        def kl_ignored(log, kl_limit, ent_ratio=0.5, t_crit=3.0):
            out = s7.ref_diagnose(log, kl_limit, ent_ratio, t_crit)
            out[s7.FLAGS[0]] = s7.trend_t(log["reward"]) > t_crit
            return out
        self.rejects(s7.check_diagnose, kl_ignored, "AND")

        def last_value(log, kl_limit, ent_ratio=0.5, t_crit=3.0):
            out = s7.ref_diagnose(log, kl_limit, ent_ratio, t_crit)
            out[s7.FLAGS[1]] = log["length"][-1] > log["length"][0]          # noise can flip this
            out[s7.FLAGS[2]] = log["entropy"][-1] < log["entropy"][0]
            return out
        self.rejects(s7.check_diagnose, last_value)

    def test_masked_entropy(self):
        self.rejects(s7.check_masked_entropy, lambda logits, m: s7.ref_entropy(logits).mean(), "mask")
        self.rejects(s7.check_masked_entropy,
                     lambda logits, m: ((s7.ref_entropy(logits) * m).sum(1) / m.sum(1)).mean(), "per answer")


class ToyWorlds(unittest.TestCase):
    def test_gaussian_peak_matches_theory(self):
        pool = s7.answer_pool(K=2**18)
        gold, proxy = s7.gold_reward(pool), s7.proxy_reward(pool, 1.0)
        lp = s7.ref_tilt(pool["logp_ref"], proxy, 1.0)
        self.assertAlmostEqual(float(s7.ref_kl(lp, pool["logp_ref"])), 1.0, delta=0.05)
        self.assertAlmostEqual(float(s7.expect(lp, gold)), 0.5, delta=0.06)
        far = s7.ref_tilt(pool["logp_ref"], proxy, 0.4)
        self.assertLess(float(s7.expect(far, gold)), float(s7.expect(lp, gold)))
        self.assertGreater(float(s7.expect(far, proxy)), float(s7.expect(lp, proxy)))

    def test_bon_kl_formula_on_large_pool(self):
        pool = s7.answer_pool(K=2**16)
        proxy = s7.proxy_reward(pool)
        for n in (2, 8, 64):
            lp = s7.bon_policy(s7.ref_bon_probs, pool["logp_ref"], proxy, n)
            self.assertAlmostEqual(float(s7.ref_kl(lp, pool["logp_ref"])), s7.ref_bon_kl(n), delta=2e-3)

    def test_toy_policy_collapses_without_kl(self):
        log0 = s7.train_toy(s7.ref_pg_loss, steps=120)
        log1 = s7.train_toy(s7.ref_pg_loss, beta=1.0, steps=120)
        self.assertLess(s7.late(log0["entropy"]), 0.5 * s7.early(log0["entropy"]))
        self.assertGreater(s7.late(log1["entropy"]), 0.8 * s7.early(log1["entropy"]))
        self.assertGreater(s7.late(log0["kl"]), s7.late(log1["kl"]))
        self.assertTrue(s7.ref_diagnose(log0, kl_limit=2.0)[s7.FLAGS[2]])

    def test_toy_stats_reference(self):
        ref = s7.toy_reference_logits()
        st = s7.toy_stats(ref, ref)
        self.assertAlmostEqual(float(st["kl"]), 0.0, places=10)
        self.assertGreater(float(st["length"]), 3.0)
        self.assertLess(float(st["length"]), 5.0)

    def test_ensemble_min_beats_single_with_independent_errors(self):
        pool = s7.answer_pool(K=2**18)
        scores, gold = s7.ensemble_world(pool, shared=0.0)
        betas = torch.logspace(0.5, -0.5, 15, dtype=torch.float64)
        single = s7.sweep(s7.ref_tilt, s7.ref_kl, pool["logp_ref"], scores[0], betas, {"gold": gold})
        mn = s7.sweep(s7.ref_tilt, s7.ref_kl, pool["logp_ref"], s7.ref_aggregate(scores, "min"), betas, {"gold": gold})
        self.assertGreater(float(mn["gold"].max()), float(single["gold"].max()) + 0.2)

    def test_trend_t(self):
        self.assertGreater(s7.trend_t([0.1 * i + (-1) ** i * 0.05 for i in range(30)]), 10)
        self.assertLess(abs(s7.trend_t([(-1) ** i for i in range(30)])), 1)


if __name__ == "__main__":
    unittest.main()
