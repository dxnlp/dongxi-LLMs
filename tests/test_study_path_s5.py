"""Study-path S5 (RLVR): every check passes its reference and rejects plausible mistakes without raising."""
import contextlib
import io
import math
import random
import re
import unittest
from fractions import Fraction

import torch

from dongxi_llms.study_path import s5_rlvr as s5


def quiet(fn, *args, **kwargs):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = fn(*args, **kwargs)
    return result, buffer.getvalue()


# ---------------------------------------------------------------- plausible wrong implementations
def extract_first_match(text):                       # ignores the "exactly one" rule
    found = s5.ANSWER_RE.findall(text)
    return found[0] if found else None


def extract_last_number(text):                       # no marker at all
    found = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return found[-1] if found else None


def parse_float(s):                                  # float, not exact
    s = s.strip().removesuffix(".")
    return None if s5.NUMBER_RE.fullmatch(s) is None else float(s.replace(",", ""))


def parse_keep_commas(s):                            # forgets the thousands separators
    s = s.strip().removesuffix(".")
    return None if s5.NUMBER_RE.fullmatch(s) is None else Fraction(s)


def reward_string_compare(text, gold, alpha=0.0):    # compares strings
    answer = s5.ref_extract_answer(text)
    formatted = answer is not None and s5.ref_parse_number(answer) is not None
    return 1.0 if answer == str(gold) else alpha * float(formatted)


def reward_additive(text, gold, alpha=0.0):          # 1 + alpha for a correct answer
    answer = s5.ref_extract_answer(text)
    value = None if answer is None else s5.ref_parse_number(answer)
    return float(value is not None and value == gold) + alpha * float(value is not None)


def pass_naive(n, c, k):
    return 1.0 - (1.0 - c / n) ** k


def zero_signal_one_sided(p, G):
    return (1 - p) ** G


def pg_loss_no_minus(theta, actions, adv):
    return (adv * torch.log_softmax(theta, -1).gather(1, actions)).mean()


def pg_loss_softmax(theta, actions, adv):
    return -(adv * torch.softmax(theta, -1).gather(1, actions)).mean()


def select_unsolved_only(successes, n):
    return successes > 0


def rewards_batch_normalised(texts, golds, G):
    R = torch.tensor([s5.ref_verifier_reward(t, g) for t, g in zip(texts, golds)])
    return R, (R - R.mean()) / (R.std() + 1e-6)


def paired_se_unpaired(before, after):
    d = after - before
    return float(d.mean()), math.sqrt(before.var().item() / len(before) + after.var().item() / len(after))


def decompose_over_all(formatted, correct):
    return float(formatted.float().mean()), float(correct.float().mean())


class ReferencesPass(unittest.TestCase):
    def test_every_reference_passes_strict(self):
        cases = [
            (s5.check_extract_answer, s5.ref_extract_answer, ()),
            (s5.check_parse_number, s5.ref_parse_number, ()),
            (s5.check_verifier_reward, s5.ref_verifier_reward, ()),
            (s5.check_pass_at_k, s5.ref_pass_at_k, ()),
            (s5.check_zero_signal_prob, s5.ref_zero_signal_prob, ()),
            (s5.check_tabular_pg_loss, s5.ref_tabular_pg_loss, ()),
            (s5.check_select_pool, s5.ref_select_pool, ()),
            (s5.check_rewards_and_advantages, s5.ref_rewards_and_advantages, (s5.ref_verifier_reward,)),
            (s5.check_paired_se, s5.ref_paired_se, ()),
            (s5.check_decompose, s5.ref_decompose, ()),
        ]
        for check, ref, extra in cases:
            with self.subTest(check=check.__name__):
                ok, out = quiet(check, ref, *extra, strict=True)
                self.assertTrue(ok)
                self.assertNotIn("❌", out)


class WrongImplementationsFail(unittest.TestCase):
    def assert_fails(self, check, fn, *extra):
        ok, out = quiet(check, fn, *extra)
        self.assertFalse(ok)
        self.assertIn("❌", out)
        with self.assertRaises(AssertionError):
            quiet(check, fn, *extra, strict=True)

    def test_extract(self):
        self.assert_fails(s5.check_extract_answer, extract_first_match)
        self.assert_fails(s5.check_extract_answer, extract_last_number)

    def test_parse(self):
        self.assert_fails(s5.check_parse_number, parse_float)
        self.assert_fails(s5.check_parse_number, parse_keep_commas)   # raises inside → ❌, not a crash

    def test_reward(self):
        self.assert_fails(s5.check_verifier_reward, reward_string_compare)
        self.assert_fails(s5.check_verifier_reward, reward_additive)

    def test_pass_at_k(self):
        self.assert_fails(s5.check_pass_at_k, pass_naive)

    def test_zero_signal(self):
        self.assert_fails(s5.check_zero_signal_prob, zero_signal_one_sided)

    def test_pg_loss(self):
        self.assert_fails(s5.check_tabular_pg_loss, pg_loss_no_minus)
        self.assert_fails(s5.check_tabular_pg_loss, pg_loss_softmax)

    def test_select(self):
        self.assert_fails(s5.check_select_pool, select_unsolved_only)

    def test_rewards_and_advantages(self):
        self.assert_fails(s5.check_rewards_and_advantages, rewards_batch_normalised, s5.ref_verifier_reward)

    def test_paired_se(self):
        self.assert_fails(s5.check_paired_se, paired_se_unpaired)

    def test_decompose(self):
        self.assert_fails(s5.check_decompose, decompose_over_all)


class NotAttemptedAndPlaceholders(unittest.TestCase):
    def test_none_and_ellipsis_are_pending(self):
        for check in (s5.check_extract_answer, s5.check_parse_number, s5.check_pass_at_k):
            for value in (None, Ellipsis):
                ok, out = quiet(check, value)
                self.assertFalse(ok)
                self.assertIn("⏳", out)

    def test_placeholder_function_does_not_raise(self):
        def placeholder(*args, **kwargs):
            return ...
        for check, extra in ((s5.check_verifier_reward, ()), (s5.check_tabular_pg_loss, ()),
                             (s5.check_rewards_and_advantages, (s5.ref_verifier_reward,)),
                             (s5.check_paired_se, ()), (s5.check_decompose, ()), (s5.check_select_pool, ()),
                             (s5.check_pass_at_k, ()), (s5.check_zero_signal_prob, ())):
            with self.subTest(check=check.__name__):
                ok, out = quiet(check, placeholder, *extra)
                self.assertFalse(ok)
                self.assertIn("❌", out)


class TasksAndVerifiers(unittest.TestCase):
    def test_generator_answers_are_exact_and_seeded(self):
        for steps in range(1, 7):
            a, b = s5.make_problems(5, steps, 20), s5.make_problems(5, steps, 20)
            self.assertEqual(a, b)
            for p in a:
                self.assertIsInstance(p["answer"], int)
                self.assertGreater(p["answer"], 0)
                self.assertEqual(p["question"].count("."), steps + 1)

    def test_replay_answer_from_text(self):
        """Recompute each answer from the sentences: the ground truth matches the text."""
        rng = random.Random(0)
        for _ in range(200):
            p = s5.make_problem(rng, rng.randint(1, 6))
            sentences = p["question"].split(". ")
            value = int(re.search(r"has (\d+)", sentences[0]).group(1))
            for s in sentences[1:]:
                if m := re.search(r"(?:buys|finds) (\d+) more", s):
                    value += int(m.group(1))
                elif m := re.search(r"gives away (\d+)", s):
                    value -= int(m.group(1))
                elif m := re.search(r"receives (\d+) boxes with (\d+)", s):
                    value += int(m.group(1)) * int(m.group(2))
                elif "twice" in s:
                    value *= 2
                elif "three times" in s:
                    value *= 3
                elif m := re.search(r"into (\d+) bags", s):
                    value //= int(m.group(1))
            self.assertEqual(value, p["answer"])

    def test_adversarial_table(self):
        strict = [s5.ref_verifier_reward(t, g) for t, g, _ in s5.ADVERSARIAL]
        labels = [l for _, _, l in s5.ADVERSARIAL]
        self.assertFalse(any(s == 1 and l == 0 for s, l in zip(strict, labels)))   # no false positives
        substring = [s5.substring_reward(t, g) for t, g, _ in s5.ADVERSARIAL]
        self.assertGreaterEqual(sum(s == 1 and l == 0 for s, l in zip(substring, labels)), 5)

    def test_unbiased_pass_at_k_is_unbiased(self):
        g = torch.Generator().manual_seed(0)
        p, n, k = 0.3, 8, 4
        c = (torch.rand(20000, n, generator=g) < p).sum(1)
        est = sum(s5.ref_pass_at_k(n, int(ci), k) for ci in c) / len(c)
        self.assertAlmostEqual(est, 1 - (1 - p) ** k, delta=0.01)


if __name__ == "__main__":
    unittest.main()
