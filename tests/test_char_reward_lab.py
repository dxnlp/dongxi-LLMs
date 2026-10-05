"""Independent CPU contracts for the additive character text-reward arm."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from dongxi_llms.char_reward_lab import (
    ASCIICharacterVocabulary, TOKENS, MAX_POSITIONS, char_config,
    char_text_batch, char_process_batch, encoding_audit, fit_char_model,
    evaluate_char_pairs, evaluate_char_traces, export_char_reward,
    load_char_reward, run_character_reference, FitFailure,
)
from dongxi_llms.text_reward_lab import TinyTextReward, load_fixture, nuisance_pairs, last_valid_indices
from dongxi_llms.run_identity import canonical_hash, file_digest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures/text-reward/records.json"
PROTOCOL = ROOT / "experiments/specs/2026-10-04-text-reward-vocabulary-intervention.md"


class CharacterRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.obj = load_fixture(FIXTURE)
        cls.vocab = ASCIICharacterVocabulary()
        with torch.random.fork_rng():
            torch.manual_seed(1611)
            cls.model = TinyTextReward(char_config()).eval()
        cls.texts = [("The cup is red .", "It is red ."), ("The bag is blue .", "blue")]

    def export(self, path, model=None):
        return export_char_reward(model or self.model, self.vocab, path, fixture_path=FIXTURE,
                                  source_groups=encoding_audit(self.obj)["source_splits"],
                                  seed=1611, protocol_path=PROTOCOL)

    @staticmethod
    def rewrite(path, change):
        body = json.loads(path.read_text())
        body.pop("payload_sha256")
        change(body)
        body["payload_sha256"] = canonical_hash(body)
        path.write_text(json.dumps(body))

    def test_fixed_alphabet_is_not_a_fitted_word_lookup(self):
        self.assertEqual(len(TOKENS), 100)
        self.assertEqual(self.vocab.tokens[5:], [chr(i) for i in range(32, 127)])
        self.assertEqual(len({self.vocab.encode(str(i))[0][0] for i in range(10)}), 10)
        self.assertNotEqual(self.vocab.encode("box")[0], self.vocab.encode("book")[0])
        with self.assertRaisesRegex(ValueError, "not trained"):
            ASCIICharacterVocabulary.fit(["arbitrary text"])
        with self.assertRaises(ValueError):
            ASCIICharacterVocabulary(unknown_policy="unk")

    def test_rejects_unsupported_original_text_before_casefolding(self):
        for text in ("数", "K", "hello\n", "hello\x00", "\x7f", "", "  ", None, 7):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.vocab.encode(text)
        self.assertEqual(self.vocab.encode("ABC")[0], self.vocab.encode("abc")[0])
        ids, unknown = self.vocab.encode("a b")
        self.assertEqual(len(ids), 3)
        self.assertEqual(unknown, [])
        self.assertNotIn(1, ids)

    def test_literal_special_text_is_characters_not_special_ids(self):
        ids, _ = self.vocab.encode("<eos>")
        self.assertEqual(len(ids), 5)
        self.assertNotIn(4, ids)
        self.assertEqual(self.vocab.interface()["tokenizer"]["vocab_size"], 100)

    def test_original_fixture_has_disjoint_whole_pairs_and_prefixes(self):
        audit = encoding_audit(self.obj)
        self.assertTrue(audit["passed"], audit["issues"])
        self.assertEqual(len(audit["records"]), 55)
        self.assertEqual(audit["unique_encoded_counts"]["calibration"]["pairs"], 4)
        self.assertEqual(audit["unique_encoded_counts"]["test"]["pairs"], 20)
        self.assertEqual(audit["maximum_pair_positions"], 111)
        self.assertEqual(audit["maximum_trace_positions"], 56)
        self.assertLess(audit["maximum_pair_positions"], MAX_POSITIONS)
        signatures = {s: {r["unordered_pair_sha256"] for r in audit["records"]
                           if r["kind"] == "pair" and r["split"] == s}
                      for s in ("train", "calibration", "test")}
        self.assertFalse(signatures["train"] & signatures["test"])
        self.assertFalse(signatures["calibration"] & signatures["test"])

    def test_raw_source_split_collision_is_retained_by_gate(self):
        obj = deepcopy(self.obj)
        obj["pairs"][-1]["source_group_id"] = obj["pairs"][0]["source_group_id"]
        audit = encoding_audit(obj)
        self.assertFalse(audit["passed"])
        self.assertEqual(audit["issues"][0]["kind"], "raw-source-contract")

    def test_same_source_correctly_labeled_swap_is_related_not_leakage(self):
        obj = deepcopy(self.obj)
        original = obj["pairs"][0]
        obj["pairs"].append({**original, "pair_id": "same-source-swapped", "left": original["right"],
                             "right": original["left"], "q_left": 1 - original["q_left"]})
        self.assertTrue(encoding_audit(obj)["passed"])

    def test_swapped_pair_across_sources_is_detected(self):
        obj = deepcopy(self.obj)
        original = obj["pairs"][0]
        obj["pairs"].append({**original, "pair_id": "new-source-swapped", "source_group_id": "invented-source",
                             "left": original["right"], "right": original["left"], "q_left": 1 - original["q_left"]})
        audit = encoding_audit(obj)
        self.assertFalse(audit["passed"])
        self.assertTrue(any(i["kind"] == "unordered-pair-source-or-split-collision" for i in audit["issues"]))

    def test_casefolded_full_prompt_across_splits_is_detected(self):
        obj = deepcopy(self.obj)
        original = obj["pairs"][0]
        obj["pairs"].append({**original, "pair_id": "calibration-duplicate", "source_group_id": "new-cal-source",
                             "split": "calibration", "prompt": original["prompt"].upper()})
        self.assertTrue(any(i["kind"] == "full-prompt-source-or-split-collision" for i in encoding_audit(obj)["issues"]))

    def test_step_prefix_source_collision_and_conflicting_labels(self):
        for cross_source in (False, True):
            obj = deepcopy(self.obj)
            original = obj["processes"][0]
            changed = deepcopy(original)
            changed["record_id"] = "duplicated-step-prefix"
            changed["final"] = "Final : 99 ."
            if cross_source:
                changed["source_group_id"], changed["split"] = "new-step-source", "calibration"
            else:
                changed["steps"][0]["valid"] = not changed["steps"][0]["valid"]
            obj["processes"].append(changed)
            kinds = {i["kind"] for i in encoding_audit(obj)["issues"]}
            self.assertIn("step-prefix-source-or-split-collision" if cross_source else "step-prefix-conflicting-label", kinds)

    def test_gate_failure_prevents_model_construction(self):
        obj = deepcopy(self.obj)
        obj["pairs"][0]["prompt"] += "数"
        with patch("dongxi_llms.char_reward_lab.TinyTextReward") as constructor:
            with self.assertRaisesRegex(ValueError, "gate failed"):
                fit_char_model(obj, steps=2)
            constructor.assert_not_called()

    def test_no_truncation_empty_and_all_padding_fail(self):
        with self.assertRaisesRegex(ValueError, "overlength"):
            char_text_batch(self.vocab, [("a" * 160, "b")])
        with self.assertRaises(ValueError):
            char_text_batch(self.vocab, [])
        with self.assertRaises(ValueError):
            self.model(torch.zeros((1, 3), dtype=torch.long), torch.zeros((1, 3), dtype=torch.bool))
        with self.assertRaises(ValueError):
            char_text_batch(self.vocab, [("valid", "   ")])

    def test_left_right_extra_padding_endpoint_and_finite_gradients(self):
        right = char_text_batch(self.vocab, self.texts, padding="right")
        left = char_text_batch(self.vocab, self.texts, padding="left")
        torch.testing.assert_close(self.model(right["ids"], right["mask"]), self.model(left["ids"], left["mask"]),
                                   atol=1e-12, rtol=1e-12)
        self.assertFalse(torch.equal(last_valid_indices(left["mask"]), left["mask"].sum(1) - 1))
        extra_ids = F.pad(left["ids"], (3, 2))
        extra_mask = F.pad(left["mask"], (3, 2), value=False)
        torch.testing.assert_close(self.model(extra_ids, extra_mask), self.model(left["ids"], left["mask"]),
                                   atol=1e-12, rtol=1e-12)
        model = deepcopy(self.model)
        model(extra_ids, extra_mask).sum().backward()
        self.assertTrue(torch.isfinite(model.hidden(extra_ids, extra_mask)).all())
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertEqual(float(model.embedding.weight.grad[0].abs().sum()), 0.)

    def test_eos_variants_are_distinct_saved_contracts(self):
        included = char_text_batch(self.vocab, self.texts)
        excluded = char_text_batch(self.vocab, self.texts, include_eos=False)
        self.assertTrue((included["ids"].gather(1, included["endpoints"][:, None]) == 4).all())
        self.assertFalse((excluded["ids"] == 4).any())
        self.assertNotEqual(self.vocab.interface()["interface_sha256"], self.vocab.interface(False)["interface_sha256"])

    def test_pair_swap_reverses_probability_and_preserves_reversed_loss(self):
        row = self.obj["pairs"][0]
        swapped = {**row, "left": row["right"], "right": row["left"], "q_left": 1 - row["q_left"]}
        result = evaluate_char_pairs(self.model, self.vocab, [row, swapped])["predictions"]
        self.assertAlmostEqual(result[0]["p_left"] + result[1]["p_left"], 1., places=12)
        self.assertAlmostEqual(result[0]["margin"], -result[1]["margin"], places=12)
        loss = [F.binary_cross_entropy_with_logits(torch.tensor(r["margin"]), torch.tensor(float(r["q_left"]))) for r in result]
        torch.testing.assert_close(*loss)

    def test_all_three_objectives_reach_backbone_and_head_in_two_updates(self):
        for objective in ("preference", "outcome", "process"):
            model, fit = fit_char_model(self.obj, seed=1611, objective=objective, steps=2)
            self.assertEqual(fit["completed_steps"], 2)
            self.assertTrue(all(v > 0 for v in fit["first_backward"].values()))
            allowed = {r.get("pair_id", r.get("record_id")) for branch in (self.obj["pairs"], self.obj["processes"])
                       for r in branch if r["split"] == "train"}
            self.assertTrue(set(fit["train_record_ids"]) <= allowed)
            self.assertTrue(all(p.dtype == torch.float64 and p.device.type == "cpu" for p in model.parameters()))

    def test_step_boundary_mask_and_future_answer_causality(self):
        original = self.obj["processes"][0]
        changed = {**original, "final": "Final : 12345 ."}
        batch = char_process_batch(self.vocab, [original, changed], padding="right")
        selected = batch["step_targets"] != -100
        self.assertTrue((batch["ids"][selected] == 3).all())
        self.assertTrue(batch["mask"][selected].all())
        self.assertTrue((batch["step_targets"][~batch["mask"]] == -100).all())
        positions = selected[0].nonzero().flatten()
        scores = self.model.token_scores(batch["ids"], batch["mask"])
        torch.testing.assert_close(scores[0, positions], scores[1, positions], atol=1e-12, rtol=1e-12)
        heldout = [r for r in self.obj["processes"] if r["split"] == "test"]
        held = char_process_batch(self.vocab, heldout)
        self.assertFalse(torch.equal(held["ids"][0], held["ids"][1]))
        self.assertNotEqual(heldout[0]["steps"][0]["valid"], heldout[1]["steps"][0]["valid"])

    def test_complete_numeric_export_exact_reload_and_detached_frozen_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reward.json"
            identity = self.export(path)
            frozen = load_char_reward(path, expected_interface=self.vocab.interface(),
                                      expected_file_sha256=identity["file_sha256"])
            batch = char_text_batch(self.vocab, self.texts)
            with torch.no_grad():
                self.assertTrue(torch.equal(self.model(batch["ids"], batch["mask"]), frozen.score_many(self.texts)))
            self.assertFalse(frozen.score_many(self.texts).requires_grad)
            self.assertFalse(any(p.requires_grad for p in frozen.model.parameters()))
            for key, value in self.model.state_dict().items():
                self.assertTrue(torch.equal(value, frozen.model.state_dict()[key]))
            with self.assertRaises(ValueError):
                frozen.score("valid", "数")
            with self.assertRaisesRegex(ValueError, "external identity"):
                load_char_reward(path, expected_file_sha256="0" * 64)

    def test_export_overwrite_and_wrong_numeric_dtype_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reward.json"
            self.export(path)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                self.export(path)
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaisesRegex(ValueError, "float64"):
                self.export(Path(directory) / "wrongdtype.json", deepcopy(self.model).float())

    def test_corrupt_hash_and_rehashed_bad_shape_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reward.json"
            self.export(path)
            raw = path.read_text()
            body = json.loads(raw)
            body["state"]["head.bias"][0] += 1
            path.write_text(json.dumps(body))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_char_reward(path)
            path.write_text(raw)
            self.rewrite(path, lambda b: b["state"].__setitem__("head.weight", [[1.]]))
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                load_char_reward(path)

    def test_rehashed_invalid_source_groups_and_seed_types_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reward.json"
            self.export(path)
            raw = path.read_text()
            for value in ("abc", [], [""], ["  "], [7], ["a", "a"], {"a": 1}):
                path.write_text(raw)
                self.rewrite(path, lambda b: b["source_groups"].__setitem__("train", value))
                with self.subTest(value=value), self.assertRaises(ValueError):
                    load_char_reward(path)
            path.write_text(raw)
            self.rewrite(path, lambda b: b.__setitem__("seed", True))
            with self.assertRaises(ValueError):
                load_char_reward(path)

    def test_rehashed_permuted_alphabet_and_changed_eos_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reward.json"
            self.export(path)
            raw = path.read_text()
            def permute(body):
                body["tokens"][5], body["tokens"][6] = body["tokens"][6], body["tokens"][5]
            self.rewrite(path, permute)
            with self.assertRaisesRegex(ValueError, "fixed alphabet"):
                load_char_reward(path)
            path.write_text(raw)
            self.rewrite(path, lambda b: b["config"].__setitem__("include_eos", False))
            with self.assertRaisesRegex(ValueError, "interface mismatch"):
                load_char_reward(path)

    def test_numeric_boolean_nonfinite_and_excessive_nesting_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reward.json"
            self.export(path)
            raw = path.read_text()
            for value in (True, "1.0", 10 ** 400):
                path.write_text(raw)
                self.rewrite(path, lambda b: b["state"]["head.bias"].__setitem__(0, value))
                with self.subTest(value=type(value)), self.assertRaises(ValueError):
                    load_char_reward(path)
            body = json.loads(raw)
            body["state"]["head.bias"][0] = float("nan")
            path.write_text(json.dumps(body))
            with self.assertRaises(ValueError):
                load_char_reward(path)
            path.write_text('{"unexpected":' + '[' * 10_000 + '0' + ']' * 10_000 + '}')
            with self.assertRaisesRegex(ValueError, "bounded"):
                load_char_reward(path)
            path.write_text('{"unexpected":' + '[' * 40 + '0' + ']' * 40 + '}')
            with self.assertRaisesRegex(ValueError, "nesting"):
                load_char_reward(path)

    def test_duplicate_json_field_and_size_limit_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reward.json"
            path.write_text('{"schema":1,"schema":2}')
            with self.assertRaises(ValueError):
                load_char_reward(path)
            path.write_text(' ' * (4 * 1024 ** 2 + 1))
            with self.assertRaisesRegex(ValueError, "bounded JSON"):
                load_char_reward(path)

    def test_failed_gate_and_nonfinite_fit_keep_diagnostics(self):
        obj = deepcopy(self.obj)
        obj["pairs"][0]["prompt"] += "数"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid-fixture.json"
            path.write_text(json.dumps(obj))
            result = run_character_reference(path, PROTOCOL)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["seeds"], [])
            self.assertEqual(result["failures"][0]["stage"], "pre-fit-encoding-gates")
        with patch.object(TinyTextReward, "forward", return_value=torch.full((24,), float("nan"), dtype=torch.float64)):
            with self.assertRaises(FitFailure) as error:
                fit_char_model(self.obj, steps=2)
            self.assertEqual(error.exception.evidence["history"], [])
            self.assertEqual(error.exception.evidence["planned_steps"], 2)

    def test_cli_retains_failed_gate_and_does_not_replace_output(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture, output = Path(directory) / "bad.json", Path(directory) / "result.json"
            obj = deepcopy(self.obj)
            obj["pairs"][0]["prompt"] += "数"
            fixture.write_text(json.dumps(obj))
            command = [sys.executable, "-m", "dongxi_llms.char_reward_lab", "--fixture", str(fixture),
                       "--protocol", str(PROTOCOL), "--output", str(output)]
            env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1")
            run = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=20)
            self.assertEqual(run.returncode, 1)
            self.assertFalse(json.loads(output.read_text())["encoding_audit"]["passed"])
            before = output.read_bytes()
            again = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=20)
            self.assertNotEqual(again.returncode, 0)
            self.assertIn("FileExistsError", again.stderr)
            self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
