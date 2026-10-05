"""Invariant and gradient tests for Chapters 13–15; no network or pretrained model."""
import unittest
from copy import deepcopy

import torch

from dongxi_llms.grpo_lab import (group_advantages, exact_kl, clipped_objective,
    verify_integer, verify_tokens, make_policy, frozen_copy, rollout,
    response_logits, prompt_tensor, TRAIN_PAIRS, DEV_PAIRS, EOS, NUMBER_START)
from dongxi_llms.optimization_diagnostics_lab import (hacking_experiment,
    RolloutIdentity, accept_rollout, length_weights)
from dongxi_llms.distillation_lab import (distillation_loss, majority_vote,
    validate_genealogy, original_card_fixture, release_gate)


class GRPOInvariantTests(unittest.TestCase):
    def test_group_advantage_population_and_zero(self):
        rewards = torch.tensor([[0., 0., 1., 1.], [1., 1., 1., 1.]], requires_grad=True)
        advantages = group_advantages(rewards)
        self.assertFalse(advantages.requires_grad)
        torch.testing.assert_close(advantages[0], torch.tensor([-1., -1., 1., 1.]))
        self.assertEqual(float(advantages[1].abs().sum()), 0.)

    def test_clipping_both_signs_and_detached_behavior(self):
        logp = torch.tensor([[1.5], [-1.5]], dtype=torch.float64, requires_grad=True)
        old = torch.zeros_like(logp, requires_grad=True)
        advantage = torch.tensor([1., -1.], requires_grad=True)
        loss = clipped_objective(logp, old, advantage, torch.ones_like(logp))
        loss.backward()
        torch.testing.assert_close(logp.grad, torch.zeros_like(logp))
        self.assertIsNone(old.grad)
        self.assertIsNone(advantage.grad)

    def test_padding_has_no_gradient_and_empty_rejected(self):
        logits = torch.tensor([[.1, 1.]], requires_grad=True)
        mask = torch.tensor([[True, False]])
        clipped_objective(logits, torch.zeros_like(logits), torch.tensor([1.]), mask).backward()
        self.assertEqual(float(logits.grad[0, 1]), 0.)
        with self.assertRaises(ValueError):
            clipped_objective(logits, logits, torch.tensor([1.]), torch.zeros_like(mask))

    def test_exact_kl_gradient(self):
        logits = torch.tensor([.2, -.4, .8], dtype=torch.float64, requires_grad=True)
        reference = torch.tensor([.7, -.1, .1], dtype=torch.float64)
        kl = exact_kl(logits, reference)
        kl.backward()
        p = logits.detach().softmax(-1)
        analytical = p*(logits.detach().log_softmax(-1)-reference.log_softmax(-1)-kl.detach())
        torch.testing.assert_close(logits.grad, analytical)

    def test_verifier_adversarial_cases(self):
        for text in ("35 0", "answer35", "35.0", "٣٥", "35 or 0", "__import__('os')"):
            self.assertFalse(verify_integer(text, 35))
        self.assertTrue(verify_integer(" +35\n", 35))
        self.assertEqual(verify_tokens([NUMBER_START+3, EOS, 0], 3), 1.)
        self.assertEqual(verify_tokens([NUMBER_START+3], 3), 0.)
        self.assertFalse(set(TRAIN_PAIRS) & set(DEV_PAIRS))

    def test_decoder_rollout_alignment_snapshot_and_gradients(self):
        torch.set_num_threads(1)
        policy = make_policy()
        old = frozen_copy(policy)
        prompts = prompt_tensor([(0, 0), (1, 1)])
        responses, mask, old_logp = rollout(old, prompts, torch.Generator().manual_seed(11))
        logits = response_logits(policy, prompts, responses)
        logp = logits.log_softmax(-1).gather(-1, responses[..., None]).squeeze(-1)
        torch.testing.assert_close(logp.detach(), old_logp)
        self.assertTrue(mask[:, 0].all())
        clipped_objective(logp, old_logp, torch.tensor([1., -1.]), mask).backward()
        self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum()) > 0
                            for p in policy.parameters()))
        self.assertTrue(all(p.grad is None for p in old.parameters()))

    def test_model_independent_hf_style_adapter(self):
        from types import SimpleNamespace
        from dongxi_llms.qwen_rlvr_lab import update_model
        class Adapter(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = make_policy(12)
            def forward(self, input_ids, use_cache=False):
                return SimpleNamespace(logits=self.model(input_ids))
        policy = Adapter()
        reference = deepcopy(policy)
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        before = [parameter.detach().clone() for parameter in reference.parameters()]
        result = update_model(policy, reference, prompt_tensor([(0, 0)]), 0,
                              lambda ids: "0", EOS,
                              torch.optim.AdamW(policy.parameters(), lr=.001),
                              torch.Generator().manual_seed(3), group_size=4,
                              max_new_tokens=3)
        self.assertEqual(len(result["texts"]), 4)
        self.assertLess(result["initial_log_ratio_max_abs"], 1e-6)
        self.assertTrue(torch.isfinite(torch.tensor(result["gradient_norm"])))
        for previous, parameter in zip(before, reference.parameters()):
            torch.testing.assert_close(previous, parameter)
        for eos, context in ((None, 12), (EOS, 5)):
            with self.assertRaises(ValueError):
                update_model(policy, reference, prompt_tensor([(0, 0)]), 0,
                             lambda ids: "0", eos,
                             torch.optim.SGD(policy.parameters(), lr=.001),
                             torch.Generator().manual_seed(3), max_new_tokens=3,
                             context_length=context)

    def test_prompt_contract_preserves_sft_template_and_stopping(self):
        import hashlib
        from dongxi_llms.qwen_rlvr_lab import prompt_contract
        class TokenizerFixture:
            eos_token_id = 2
            unk_token_id = 0
            chat_template = "original-template-v1"
            def encode(self, text, add_special_tokens=False):
                return [3] if text == "<|im_end|>" else [1, 4, 3]
            def apply_chat_template(self, messages, **kwargs):
                self.last_kwargs = kwargs
                return [1, 4, 3]
        tokenizer = TokenizerFixture()
        genealogy = {"template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()}
        encode, stop_ids, interface = prompt_contract(tokenizer, "chat", genealogy)
        self.assertEqual(stop_ids, [2, 3])
        self.assertEqual(encode("0+0").tolist(), [[1, 4, 3]])
        self.assertFalse(tokenizer.last_kwargs["enable_thinking"])
        with self.assertRaises(ValueError):
            prompt_contract(tokenizer, "raw", genealogy)
        with self.assertRaises(ValueError):
            prompt_contract(tokenizer, "chat", genealogy, template="changed")

    def test_frozen_greedy_panel_stop_and_context(self):
        from types import SimpleNamespace
        from dongxi_llms.qwen_rlvr_lab import evaluate_model
        class GreedyFixture(torch.nn.Module):
            def forward(self, input_ids, use_cache=False):
                logits = torch.zeros((*input_ids.shape, 12))
                token = 4 if input_ids.shape[1] == 4 else 3
                logits[:, -1, token] = 10
                return SimpleNamespace(logits=logits)
        encode = lambda text: torch.tensor([[1, 4, 4, 3]])
        panel = evaluate_model(GreedyFixture(), [(0, 0), (0, 1)], encode,
                               lambda ids: "0", [2, 3], 3, 12)
        self.assertEqual(panel["accuracy"], .5)
        self.assertEqual(panel["rows"][0]["stop"], "declared-stop")
        with self.assertRaises(ValueError):
            evaluate_model(GreedyFixture(), [(0, 0)], encode,
                           lambda ids: "0", [2, 3], 3, 5)

    def test_runner_failure_preserves_baseline_and_completed_updates(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from dongxi_llms.qwen_rlvr_lab import main, BUDGET_KEYS
        from snapshot_io_test_support import explicit_snapshot_io_args
        with tempfile.TemporaryDirectory(prefix="dongxi-rlvr-failure-test-") as directory:
            base = Path(directory)
            model_dir = base/"local-model-fixture"
            model_dir.mkdir()
            output = base/"new-run"
            limits=base/'work-limits.json'
            limits.write_text(json.dumps({key:100000 for key in BUDGET_KEYS}))
            cap_args=['--work-limits',str(limits),'--work-journal-max-bytes','1048576',
                      '--snapshot-max-bytes','16777216', *explicit_snapshot_io_args(base)]
            def fail_after_one_update(args, journal):
                journal.stage("initial-evaluation")
                journal.baseline_row({"prompt": "fixture", "correct": False})
                journal.stage("ready-for-updates", initial_heldout={"n": 1, "accuracy": 0.})
                journal.record({"update": 1, "loss": .1, "reward": 0.})
                journal.stage("update", active_update=2)
                raise RuntimeError("fixture memory guard failure")
            with patch("dongxi_llms.qwen_rlvr_lab.execute_run", side_effect=fail_after_one_update):
                with self.assertRaisesRegex(RuntimeError, "fixture memory"):
                    main(["--model-dir", str(model_dir), "--revision", "a"*40,
                          "--output", str(output), "--device", "cpu",*cap_args])
            report = json.loads((output/"report.json").read_text())
            status = json.loads((output/"status.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["completed_updates"], 1)
            self.assertEqual(report["failure"]["stage"], "update")
            self.assertEqual(report["active_update"], 2)
            self.assertEqual(report["initial_heldout"]["accuracy"], 0.)
            self.assertEqual(len(report["initial_heldout_partial"]), 1)
            self.assertEqual(len((output/"metrics.jsonl").read_text().splitlines()), 1)
            # A second attempt cannot overwrite the preserved failed-run evidence.
            with self.assertRaises(SystemExit):
                main(["--model-dir", str(model_dir), "--revision", "a"*40,
                      "--output", str(output), "--device", "cpu",*cap_args])


class DiagnosticsAndCapstoneTests(unittest.TestCase):
    def test_reward_hack_and_repair_actual_optimization(self):
        broken = hacking_experiment(updates=20)["history"]
        repair = hacking_experiment(updates=20, repair=True)["history"]
        self.assertGreater(broken[-1]["proxy_reward"], broken[0]["proxy_reward"])
        self.assertLess(broken[-1]["strict_accuracy"], broken[0]["strict_accuracy"])
        self.assertGreater(repair[-1]["strict_accuracy"], repair[0]["strict_accuracy"])

    def test_freshness_and_weighting(self):
        row = RolloutIdentity("p1", 2, "token-v1", "reward-v1", "g1", 3)
        self.assertTrue(accept_rollout(row, 2, "token-v1", "reward-v1"))
        with self.assertRaises(ValueError):
            accept_rollout(row, 3, "token-v1", "reward-v1")
        with self.assertRaises(ValueError):
            accept_rollout(row, 2, "token-v2", "reward-v1")
        torch.testing.assert_close(length_weights([2, 8]), torch.tensor([.5, .5], dtype=torch.float64))
        torch.testing.assert_close(length_weights([2, 8], "token"), torch.tensor([.2, .8], dtype=torch.float64))

    def test_distillation_gradient_temperature_and_teacher_detach(self):
        for temperature in (1., 2., 4.):
            student = torch.tensor([[.1, .2, -.3]], dtype=torch.float64, requires_grad=True)
            teacher = torch.tensor([[2., .2, -.4]], dtype=torch.float64, requires_grad=True)
            loss = distillation_loss(student, teacher, temperature)
            loss.backward()
            expected = temperature*((student.detach()/temperature).softmax(-1)
                                    -(teacher.detach()/temperature).softmax(-1))
            torch.testing.assert_close(student.grad, expected)
            self.assertIsNone(teacher.grad)

    def test_majority_stable_tie_and_genealogy_failures(self):
        self.assertEqual(majority_vote([3, 4, 4, 3]), 3)
        cards = original_card_fixture()
        self.assertTrue(validate_genealogy(cards))
        cards[0]["parent"] = "toy-rl"
        with self.assertRaises(ValueError):
            validate_genealogy(cards)
        cards = original_card_fixture()
        cards[1]["evaluation_sha256"] = "changed-panel"
        with self.assertRaises(ValueError):
            validate_genealogy(cards)

    def test_missing_release_evidence_is_not_ready(self):
        result = release_gate({"sources_checked": True})
        self.assertFalse(result["ready"])
        self.assertIn("notebooks_execute", result["missing"])


if __name__ == "__main__":
    unittest.main()
