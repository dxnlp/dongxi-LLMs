"""Independent arithmetic/interface tests, not assertions of task improvement."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
import unittest

import torch

from dongxi_llms import posttraining_assignment as pa
from dongxi_llms import reasoning_controls as rc
from dongxi_llms.grpo_lab import group_advantages, verify_integer
from dongxi_llms.sft_lab import token_loss_sum


class PosttrainingAssignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items, cls.protocol = pa.load_fixture()

    def test_independent_truth_table_and_partial_constant_baselines(self):
        # Enumerated even totals/equality pairs independently of fixture labels.
        for item in self.items:
            expected = int(item["a"]+item["b"] not in (0, 2, 4, 6)) if item["family"] == "parity" else int(item["a"]-item["b"] == 0)
            self.assertEqual(item["answer"], expected)
        for result in pa.baselines(self.items).values():
            self.assertEqual(result["constant0"], .5)
            self.assertEqual(result["constant1"], .5)
            self.assertEqual(result["majority_accuracy"], .5)

    def test_source_balance_and_paired_heldout_templates(self):
        for split in ("train", "development", "heldout-source", "heldout-family"):
            rows = [i for i in self.items if i["split"] == split]
            for answer in (0, 1):
                self.assertGreaterEqual(len({i["source_group"] for i in rows if i["answer"] == answer}), 2)
        heldout = [i for i in self.items if i["split"] == "heldout-source"]
        direct = {i["source_group"] for i in heldout if i["template"] == "direct"}
        alias = {i["source_group"] for i in heldout if i["template"] == "alias"}
        self.assertEqual(direct, alias)
        self.assertEqual(len(direct), 4)

    def test_problem_leak_and_corrupted_answers_rejected(self):
        copied = deepcopy(self.items)
        copied[0]["answer"] = 1
        with self.assertRaisesRegex(ValueError, "oracle"):
            pa.validate_fixture(copied)
        copied = deepcopy(self.items)
        copied[1]["split"] = "development"
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            pa.validate_fixture(copied)

    def test_exact_mask_and_shift_have_only_two_supervised_positions(self):
        batch = pa.supervised_batch(self.items[:2])
        self.assertEqual(batch["labels"].tolist(), [[-100]*4+[6, 2], [-100]*4+[6, 2]])
        logits = torch.zeros(2, 6, 10, dtype=torch.float64, requires_grad=True)
        total, count = token_loss_sum(logits, batch["labels"])
        self.assertEqual(count, 4)
        self.assertAlmostEqual(float(total.detach()), 4*torch.log(torch.tensor(10.)).item(), places=6)
        total.backward()
        self.assertEqual(torch.nonzero(logits.grad.abs().sum(-1)).tolist(), [[0, 3], [0, 4], [1, 3], [1, 4]])

    def test_coverage_and_repetition_match_target_exposure_not_sources(self):
        train = [i for i in self.items if i["split"] == "train"]
        narrow = [i for i in train if (i["a"], i["b"]) in ((0, 0), (0, 1))]*4
        self.assertEqual(len(train), len(narrow))
        self.assertEqual(len({i["source_group"] for i in train}), 8)
        self.assertEqual(len({i["source_group"] for i in narrow}), 2)
        for rows in (train, narrow):
            self.assertEqual(int((pa.supervised_batch(rows)["labels"] != -100).sum())*24, 768)

    def test_grammar_sampling_and_scoring_share_probability_support(self):
        model = rc.make_decoder(33671).double().eval()
        prompts = pa.prompt_ids(self.items[:2])
        tokens, masks = rc.generate(model, prompts, generator=torch.Generator().manual_seed(9))
        first, later = rc.path_logits(model, prompts, tokens)
        with torch.no_grad():
            a = model(prompts)[:, -1, [6, 7]].log_softmax(-1)
            b = model(torch.cat((prompts, tokens[:, :1]), -1))[:, -1, [2, 6, 7]].log_softmax(-1)
        self.assertTrue(torch.allclose(first.log_softmax(-1), a, atol=1e-12, rtol=0))
        self.assertTrue(torch.allclose(later.log_softmax(-1), b, atol=1e-12, rtol=0))
        self.assertTrue(bool(masks.all()))
        # Forcing EOS creates a different sampling contract, not a harmless mask.
        self.assertTrue(bool((b[:, 0].exp() < 1).all()))

    def test_answer_termination_and_complete_path_are_distinct(self):
        item = self.items[0]
        for path, answer, stop, success in (([6, 7], True, False, False), ([7, 2], False, True, False), ([6, 2], True, True, True)):
            result = pa.path_grade(torch.tensor(path), torch.tensor([True, True]), item)
            self.assertEqual((result["answer_correct"], result["natural_termination"], result["complete_path_success"]), (answer, stop, success))

    def test_finite_center_expectation_and_two_response_std_oracle(self):
        for row in pa.finite_group_audit(sizes=(2, 4))['rows']:
            p, size = row["success_probability"], row["group_size"]
            if row["normalization"] == "center-only":
                self.assertAlmostEqual(row["exact_expected_logit_gradient"], (size-1)/size*p*(1-p), places=12)
            elif size == 2:
                # Two mixed actions have population std1/2; allconstantgroups give0.
                self.assertAlmostEqual(row["exact_expected_logit_gradient"], p*(1-p)/(1+2e-8), places=12)
            self.assertAlmostEqual(row["mixed_probability"], 1-p**size-(1-p)**size, places=12)

    def test_mixed_filter_changes_prompt_population_and_cost(self):
        result = pa.finite_group_audit(sizes=(2,))['selection_populations'][0]
        self.assertGreater(result["mixed_selected_prompt_probabilities"][1], 1/3)
        self.assertAlmostEqual(sum(result["mixed_selected_prompt_probabilities"]), 1.)
        self.assertGreater(result["attempts_per_selected_group"], 1.)
        self.assertTrue(torch.equal(group_advantages(torch.ones(2, 8)), torch.zeros(2, 8)))

    def test_adversarial_grader_and_strict_reward_contract(self):
        self.assertFalse(verify_integer("Final answer: 5", 5))
        result = pa._instrument("Final answer: 5", "eos", 5)
        self.assertTrue(result["richer_complete_reward"])
        self.assertTrue(result["disagreement"])
        result = pa._instrument("5", "max_tokens", 5)
        self.assertTrue(result["richer_answer_correct"])
        self.assertFalse(result["richer_complete_reward"])
        unsafe = pa._instrument("5+__import__('os').system('echo unsafe')", "eos", 5)
        self.assertFalse(unsafe["richer_answer_correct"])
        self.assertFalse(pa._instrument("5 6", "eos", 5)["richer_complete_reward"])

    def test_original_native_rewards_replay_without_changing_archive(self):
        before = [pa.file_hash(pa.ROOT/p) for p in pa.ARCHIVES]
        audit = pa.native_reward_audit()
        self.assertEqual(audit["summary"]["archived_responses"], 192)
        self.assertEqual(audit["summary"]["strict_replay_positive_rewards"], 0)
        self.assertEqual(audit["summary"]["saved_positive_rewards"], 0)
        self.assertTrue(all(float(r["strict_terminated_integer_reward"]) == r["saved_reward"] for r in audit["rows"]))
        self.assertEqual(before, [pa.file_hash(pa.ROOT/p) for p in pa.ARCHIVES])

    def test_foreign_sampler_interface_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/"posttraining_items.json").write_text(json.dumps(self.items))
            protocol = deepcopy(self.protocol)
            protocol["interface"]["first_sampling_support"] = [2, 6, 7]
            (path/"posttraining_protocol.json").write_text(json.dumps(protocol))
            with self.assertRaisesRegex(ValueError, "sampler"):
                pa.load_fixture(path)

    def test_actual_small_chain_preserves_ancestry_and_first_sibling_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            result = pa.run_seed(self.items, self.protocol, 33671, sft_updates=1, sibling_updates=1,
                started=time.perf_counter(), cap=30, artifact_dir=directory)
            parent_hash = result["ancestry"]["coverage-sft"]["final_state_sha256"]
            for arm in self.protocol["sibling_arms"]:
                self.assertEqual(result["ancestry"][arm]["initial_state_sha256"], parent_hash)
                self.assertEqual(result["stages"][arm]["reference_state_sha256_before_and_after"], parent_hash)
                self.assertEqual(result["stages"][arm]["history"][0]["initial_log_ratio_max_abs"], 0)
            pools = [result["stages"][a]["rollouts"][0]["response_ids"] for a in self.protocol["sibling_arms"]]
            self.assertEqual(pools[0], pools[1])
            self.assertEqual(pools[1], pools[2])
            model, checkpoint = pa.load_checkpoint(Path(directory)/"seed-33671/coverage-sft.pt", self.protocol)
            self.assertEqual(rc.state_hash(model), parent_hash)
            self.assertEqual(checkpoint["parent"]["parent_state_sha256"], result["frozen_reference_checks"]["initial"])
            wrong = deepcopy(self.protocol)
            wrong["interface"]["tokens"]["EOS"] = 9
            with self.assertRaisesRegex(ValueError, "interface"):
                pa.load_checkpoint(Path(directory)/"seed-33671/coverage-sft.pt", wrong)


if __name__ == "__main__":
    unittest.main()
