"""Independent endpoint, text/step, gradient and frozen-interface contracts."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F
from dongxi_llms.run_identity import canonical_hash
from dongxi_llms.text_reward_lab import (
    RewardConfig, TinyTextReward, TextVocabulary, text_batch, process_batch,
    last_valid_indices, load_fixture, fixture_vocabulary, validate_fixture,
    fit_text_model, evaluate_pairs, group_mean_loss, probability_metrics,
    export_frozen, load_frozen, nuisance_pairs,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures/text-reward/records.json"


class TextRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.obj = load_fixture(FIXTURE)
        self.vocab = fixture_vocabulary(self.obj)
        with torch.random.fork_rng():
            torch.manual_seed(44)
            self.model = TinyTextReward(RewardConfig(len(self.vocab.tokens)))
        self.texts = [("the cup is red .", "It is red ."), ("the cup is blue .", "blue")]

    def export(self, path):
        groups = {split: sorted({r["source_group_id"] for rows in (self.obj["pairs"], self.obj["processes"])
                                for r in rows if r["split"] == split}) for split in ("train", "calibration", "test")}
        return export_frozen(self.model, self.vocab, path, fixture_path=FIXTURE,
                              source_groups=groups, seed=1601)

    def rewrite_payload(self, path, alter):
        value = json.loads(path.read_text())
        value.pop("payload_sha256")
        alter(value)
        value["payload_sha256"] = canonical_hash(value)
        path.write_text(json.dumps(value))  # Generated adversarial test artifact.

    def test_last_valid_is_position_not_length(self):
        mask = torch.tensor([[True, True, False, False], [False, False, True, True]])
        self.assertEqual(last_valid_indices(mask).tolist(), [1, 3])
        for bad in (torch.zeros(2, 3, dtype=torch.bool), torch.ones(2, 3), torch.zeros(2, 0, dtype=torch.bool)):
            with self.assertRaises(ValueError):
                last_valid_indices(bad)

    def test_left_right_padding_invariance_and_no_nan_queries(self):
        right, left = [text_batch(self.vocab, self.texts, padding=side) for side in ("right", "left")]
        actual_right = self.model(right["ids"], right["mask"])
        actual_left = self.model(left["ids"], left["mask"])
        torch.testing.assert_close(actual_left, actual_right, atol=1e-12, rtol=1e-12)
        self.assertTrue(torch.isfinite(self.model.hidden(left["ids"], left["mask"])).all())
        (actual_left.sum() + actual_right.sum()).backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters()))
        self.assertTrue((self.model.embedding.weight.grad[0] == 0).all())

    def test_extra_padding_does_not_change_scores(self):
        batch = text_batch(self.vocab, self.texts)
        expected = self.model(batch["ids"], batch["mask"])
        ids = F.pad(batch["ids"], (0, 7))
        mask = F.pad(batch["mask"], (0, 7))
        torch.testing.assert_close(expected, self.model(ids, mask), atol=1e-12, rtol=1e-12)

    def test_eos_inclusion_is_encoded_not_an_off_by_one(self):
        included = text_batch(self.vocab, self.texts, include_eos=True)
        excluded = text_batch(self.vocab, self.texts, include_eos=False)
        for i in range(2):
            self.assertEqual(int(included["ids"][i, included["endpoints"][i]]), 4)
            self.assertNotEqual(int(excluded["ids"][i, excluded["endpoints"][i]]), 4)
            self.assertEqual(int(included["mask"][i].sum()), int(excluded["mask"][i].sum()) + 1)
        self.assertNotEqual(self.vocab.interface(True)["interface_sha256"], self.vocab.interface(False)["interface_sha256"])

    def test_pair_swap_reverses_probability_and_loss_label(self):
        row = self.obj["pairs"][0]
        swapped = {**row, "left": row["right"], "right": row["left"], "q_left": 1 - row["q_left"]}
        a = evaluate_pairs(self.model, self.vocab, [row])["predictions"][0]
        b = evaluate_pairs(self.model, self.vocab, [swapped])["predictions"][0]
        self.assertAlmostEqual(a["margin"], -b["margin"], places=12)
        self.assertAlmostEqual(a["p_left"], 1 - b["p_left"], places=12)
        loss_a = F.binary_cross_entropy_with_logits(torch.tensor(a["margin"]), torch.tensor(float(row["q_left"])))
        loss_b = F.binary_cross_entropy_with_logits(torch.tensor(b["margin"]), torch.tensor(float(swapped["q_left"])))
        torch.testing.assert_close(loss_a, loss_b)

    def test_backbone_attention_and_head_receive_gradients(self):
        trained, fit = fit_text_model(self.obj, self.vocab, steps=1)
        self.assertGreater(fit["first_backward"]["embedding_norm"], 0)
        self.assertGreater(fit["first_backward"]["attention_qkv_norm"], 0)
        self.assertGreater(fit["first_backward"]["head_norm"], 0)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(trained.parameters(), self.model.parameters())))

    def test_empty_overlength_all_padding_and_bad_geometry_fail(self):
        for texts in ([], [("", "red")], [("red", "")]):
            with self.assertRaises(ValueError):
                text_batch(self.vocab, texts)
        with self.assertRaises(ValueError):
            text_batch(self.vocab, [("red " * 100, "red")])
        with self.assertRaises(ValueError):
            self.model(torch.zeros(1, 4, dtype=torch.long), torch.zeros(1, 4, dtype=torch.bool))
        with self.assertRaises(ValueError):
            RewardConfig(len(self.vocab.tokens), width=25)
        batch = text_batch(self.vocab, self.texts)
        batch["ids"][~batch["mask"]] = 7
        with self.assertRaises(ValueError):
            self.model(batch["ids"], batch["mask"])

    def test_unknown_tokens_are_reported_or_explicitly_rejected(self):
        ids, unknown = self.vocab.encode("unseenSwedishWord")
        self.assertEqual(ids, [1])
        self.assertEqual(unknown, ["unseenswedishword"])
        vocab = TextVocabulary(self.vocab.tokens, unknown_policy="error")
        with self.assertRaisesRegex(ValueError, "Unknown tokens"):
            vocab.encode("unseenSwedishWord")

    def test_source_groups_do_not_cross_splits(self):
        obj = deepcopy(self.obj)
        obj["pairs"][-1]["source_group_id"] = obj["pairs"][0]["source_group_id"]
        with self.assertRaisesRegex(ValueError, "split collision"):
            validate_fixture(obj)

    def test_step_targets_only_touch_explicit_boundaries(self):
        batch = process_batch(self.vocab, self.obj["processes"][:4], padding="left")
        selected = batch["step_targets"] != -100
        self.assertTrue(batch["mask"][selected].all())
        self.assertTrue((batch["ids"][selected] == 3).all())
        self.assertTrue((batch["step_targets"][~batch["mask"]] == -100).all())
        self.assertTrue((batch["step_targets"].gather(1, batch["endpoints"][:, None]) == -100).all())

    def test_final_answer_cannot_change_earlier_step_scores(self):
        row = self.obj["processes"][0]
        changed = {**row, "final": "Final : 6 ."}
        batch = process_batch(self.vocab, [row, changed])
        scores = self.model.token_scores(batch["ids"], batch["mask"])
        indices = (batch["step_targets"][0] != -100).nonzero().flatten()
        torch.testing.assert_close(scores[0, indices], scores[1, indices], atol=1e-12, rtol=1e-12)

    def test_correct_answer_and_valid_step_are_not_same_label(self):
        crossed = {(r["outcome"], r["steps"][0]["valid"]) for r in self.obj["processes"]}
        self.assertEqual(crossed, {(0, False), (0, True), (1, False), (1, True)})
        for objective in ("outcome", "process"):
            model, fit = fit_text_model(self.obj, self.vocab, objective=objective, steps=2)
            self.assertEqual(len(fit["history"]), 2)
            self.assertGreater(fit["first_backward"]["head_norm"], 0)

    def test_group_mean_and_soft_brier_have_correct_semantics(self):
        value = group_mean_loss(torch.tensor([1., 1., 1., 9.]), ["a", "a", "a", "b"])
        self.assertEqual(float(value), 5.)
        result = probability_metrics([.5], [.5])
        self.assertEqual(result["brier"], .25)
        self.assertIsNone(result["ranking_accuracy"])

    def test_complete_json_reload_is_exact_and_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reward.json"
            identity = self.export(path)
            frozen = load_frozen(path, expected_interface=self.vocab.interface())
            batch = text_batch(self.vocab, self.texts)
            with torch.no_grad():
                expected = self.model(batch["ids"], batch["mask"])
            self.assertTrue(torch.equal(expected, frozen.score_many(self.texts)))
            self.assertFalse(frozen.score_many(self.texts).requires_grad)
            self.assertFalse(any(p.requires_grad for p in frozen.model.parameters()))
            self.assertEqual(frozen.identity["payload_sha256"], identity["payload_sha256"])
            for key, value in self.model.state_dict().items():
                self.assertTrue(torch.equal(value, frozen.model.state_dict()[key]))
            with self.assertRaises(FileExistsError):
                self.export(path)

    def test_modified_state_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reward.json"
            self.export(path)
            obj = json.loads(path.read_text())
            obj["state"]["head.bias"][0] += 1
            path.write_text(json.dumps(obj))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_frozen(path)

    def test_even_rehashed_bad_shape_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reward.json"
            self.export(path)
            self.rewrite_payload(path, lambda obj: obj["state"].__setitem__("head.weight", [[1.]]))
            with self.assertRaisesRegex(ValueError, "state shape"):
                load_frozen(path)

    def test_equal_size_permuted_vocabulary_and_changed_eos_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reward.json"
            self.export(path)
            def change_tokens(obj):
                obj["tokens"][5], obj["tokens"][6] = obj["tokens"][6], obj["tokens"][5]
            self.rewrite_payload(path, change_tokens)
            with self.assertRaisesRegex(ValueError, "interface mismatch"):
                load_frozen(path)
            path.unlink()  # Validated test-only temporary target.
            self.export(path)
            self.rewrite_payload(path, lambda obj: obj["config"].__setitem__("include_eos", False))
            with self.assertRaisesRegex(ValueError, "interface mismatch"):
                load_frozen(path)

    def test_nuisance_slices_keep_same_source_and_authored_ties(self):
        rows = nuisance_pairs([r for r in self.obj["pairs"] if r["split"] == "test"])
        self.assertEqual(len(rows), 20)
        self.assertEqual(len({r["source_group_id"] for r in rows}), 4)
        self.assertEqual(sum(r["q_left"] == .5 for r in rows), 8)
        self.assertEqual({r["split"] for r in rows}, {"test"})

    def test_rehashed_source_split_types_and_empty_identities_are_rejected(self):
        invalid_splits = ["abc", [], [""], ["   "], [7], ["a", "a"], {"a": 1}]
        for invalid in invalid_splits:
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "reward.json"
                self.export(path)
                self.rewrite_payload(path, lambda obj: obj["source_groups"].__setitem__("train", invalid))
                with self.assertRaisesRegex(ValueError, "nonempty lists"):
                    load_frozen(path)


if __name__ == "__main__":
    unittest.main()
