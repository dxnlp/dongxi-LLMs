import copy
import json
from pathlib import Path
import unittest

from dongxi_llms.reasoning_evaluation import (MAX_DEPTH, SCHEMA_VERSION,
    canonicalize_answer, candidate_view, extract_answer, freeze_contract,
    grade_response, paired_group_bootstrap, replay_records, validate_record)
from dongxi_llms.grpo_lab import verify_integer

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/reasoning-evaluation"


def fixture():
    items = json.loads((FIXTURE / "items.json").read_text())
    settings = json.loads((FIXTURE / "settings.json").read_text())
    records = [json.loads(line) for line in (FIXTURE / "responses.jsonl").read_text().splitlines() if line.strip()]
    return items, settings, records, freeze_contract(items, settings)


class MathGrammarTests(unittest.TestCase):
    def test_reviewed_adversarial_cases(self):
        for case in json.loads((FIXTURE / "grading_cases.json").read_text()):
            with self.subTest(case=case["id"]):
                candidate = canonicalize_answer(case["input"])
                reference = canonicalize_answer(case["reference"])
                status = candidate["status"]
                if status == "SUPPORTED":
                    status = "CORRECT" if candidate["canonical"] == reference["canonical"] else "INCORRECT"
                self.assertEqual(status, case["status"])

    def test_balanced_nested_box_and_ambiguity(self):
        extracted = extract_answer(r"Work first. \boxed{\frac{3}{\frac{6}{2}}}")
        self.assertEqual(extracted["mechanism"], "boxed")
        self.assertEqual(canonicalize_answer(extracted["answer"])["canonical"]["value"], [1, 1])
        for text in (r"\boxed{1} \boxed{2}", "Final answer: 1\nFinal answer: 1", r"\boxed{1}" + "\nFinal answer: 1"):
            self.assertEqual(extract_answer(text)["status"], "AMBIGUOUS")
        self.assertEqual(extract_answer(r"\boxed{1")["status"], "INVALID")

    def test_resource_bounds_and_no_last_number_rescue(self):
        self.assertEqual(canonicalize_answer("(" * 20 + "1" + ")" * 20)["status"], "INVALID")
        self.assertEqual(canonicalize_answer("{" + ",".join(["1"]*65) + "}")["status"], "INVALID")
        self.assertEqual(canonicalize_answer("1" * 2049)["status"], "INVALID")
        self.assertEqual(extract_answer("a" * 16385)["status"], "INVALID")
        self.assertEqual(canonicalize_answer("The result may be 1 or 2")["status"], "UNSUPPORTED")
        self.assertEqual(canonicalize_answer("1 2")["status"], "INVALID")
        self.assertEqual(canonicalize_answer("0 .5")["status"], "INVALID")

    def test_units_and_interval_boundaries_are_not_erased(self):
        self.assertNotEqual(canonicalize_answer("100 cm")["canonical"], canonicalize_answer("1 m")["canonical"])
        self.assertNotEqual(canonicalize_answer("[1,2)")["canonical"], canonicalize_answer("[1,2]")["canonical"])
        self.assertEqual(canonicalize_answer(r"3\mathrm{s}")["canonical"], canonicalize_answer("3 seconds")["canonical"])

    def test_existing_strict_integer_contract_unchanged(self):
        self.assertTrue(verify_integer("  +2  ", 2))
        for text in ("2.0", r"\boxed{2}", "2 0", "2 cm", "２"):
            self.assertFalse(verify_integer(text, 2))


class FrozenReplayTests(unittest.TestCase):
    def setUp(self):
        self.items, self.settings, self.records, self.contract = fixture()

    def test_contract_is_canonical_and_changes_with_settings(self):
        reordered = dict(reversed(list(self.settings.items())))
        self.assertEqual(freeze_contract(self.items, reordered), self.contract)
        settings = copy.deepcopy(self.settings)
        settings["thinking_mode"] = "enabled"
        self.assertNotEqual(freeze_contract(self.items, settings)["identity"], self.contract["identity"])
        self.assertEqual(self.records[0]["contract_id"], self.contract["identity"])

    def test_item_identity_and_cross_split_group(self):
        items = copy.deepcopy(self.items)
        items[1]["split"] = "test"
        with self.assertRaises(ValueError):
            freeze_contract(items, self.settings)
        items = copy.deepcopy(self.items)
        items[0]["reference"] = r"\sqrt{4}"
        with self.assertRaises(ValueError):
            freeze_contract(items, self.settings)

    def test_replay_retains_errors_unknown_costs_and_regressions(self):
        report = replay_records(self.items, self.records, self.contract)
        self.assertEqual(len(report["rows"]), 30)
        a = report["checkpoints"]["authored-baseline"]
        b = report["checkpoints"]["authored-candidate"]
        self.assertGreater(b["summary"]["task_success"], a["summary"]["task_success"])
        self.assertLess(b["slices"]["task"]["sets"]["accuracy"], a["slices"]["task"]["sets"]["accuracy"])
        self.assertLess(b["slices"]["task"]["json-format"]["format_valid"], a["slices"]["task"]["json-format"]["format_valid"])
        self.assertEqual(a["summary"]["cost"]["generation_tokens"]["unknown_rows"], 15)
        statuses = a["summary"]["statuses"]
        self.assertEqual(statuses["ERROR"], 1)
        self.assertEqual(statuses["UNSUPPORTED"], 1)
        self.assertEqual(statuses["AMBIGUOUS"], 1)
        row = next(row for row in report["rows"] if row["item_id"] == "stopping-a" and row["checkpoint_id"] == "authored-baseline")
        self.assertTrue(row["correct"])
        self.assertTrue(row["truncated"])
        self.assertFalse(row["natural_termination"])

    def test_changed_frozen_bytes_and_duplicate_record_rejected(self):
        bad = copy.deepcopy(self.contract)
        bad["settings"]["max_new_tokens"] = 99
        with self.assertRaises(ValueError):
            replay_records(self.items, self.records, bad)
        with self.assertRaises(ValueError):
            replay_records(self.items, self.records + [self.records[0]], self.contract)
        bad_items = copy.deepcopy(self.items)
        bad_items[0]["prompt"] += " altered"
        with self.assertRaises(ValueError):
            replay_records(bad_items, self.records, self.contract)

    def test_response_schema_count_stop_and_cost_invariants(self):
        mutations = [dict(contract_id="wrong"), dict(task="wrong"), dict(source_group="wrong"),
                     dict(stop_reason="max_tokens", truncated=False), dict(stop_reason="eos", error="failed"),
                     dict(stop_reason="max_tokens", truncated=True, generated_tokens=2),
                     dict(generated_tokens=2, token_ids=[1]), dict(generated_tokens=True),
                     dict(token_ids=[True], generated_tokens=1),
                     dict(cost={"wall_seconds": .1, "generation_tokens": None, "scoring_tokens": 1.5}),
                     dict(cost={"wall_seconds": float("nan"), "generation_tokens": None, "scoring_tokens": None})]
        for change in mutations:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_record(dict(self.records[0], **change), self.contract, self.items[0])
        valid = dict(self.records[0], token_ids=[2, 3], generated_tokens=2,
                     cost={"wall_seconds": .1, "generation_tokens": 2, "scoring_tokens": 4})
        validate_record(valid, self.contract, self.items[0])

    def test_non_oracle_candidate_view_excludes_even_injected_gold(self):
        contaminated = dict(self.records[0], reference="1/2", correct=True, oracle=True)
        view = candidate_view(contaminated)
        self.assertNotIn("reference", view)
        self.assertNotIn("correct", view)
        self.assertNotIn("oracle", view)
        self.assertEqual(view["raw_response"], self.records[0]["raw_response"])

    def test_format_is_independent_of_unsupported_or_ambiguous_answer(self):
        report = replay_records(self.items, self.records, self.contract)
        for item_id in ("unsupported-a", "ambiguity-a"):
            row = next(row for row in report["rows"] if row["item_id"] == item_id and row["checkpoint_id"] == "authored-baseline")
            self.assertTrue(row["format_valid"])
            self.assertFalse(row["correct"])
            self.assertFalse(row["task_success"])

    def test_json_duplicate_keys_and_boolean_are_not_answer_four(self):
        item = next(item for item in self.items if item["kind"] == "json")
        record = next(record for record in self.records if record["item_id"] == item["id"])
        self.assertEqual(grade_response(item, dict(record, raw_response='{"answer":4,"answer":5}'))["status"], "INVALID")
        self.assertFalse(grade_response(item, dict(record, raw_response='{"answer":true}'))["correct"])
        self.assertEqual(grade_response(item, dict(record, raw_response='{"answer":NaN}'))["status"], "INVALID")

    def test_grouped_pairing_reproducible_and_wrong_coverage_rejected(self):
        rows = replay_records(self.items, self.records, self.contract)["rows"]
        a = [row for row in rows if row["checkpoint_id"] == "authored-baseline"]
        b = [row for row in rows if row["checkpoint_id"] == "authored-candidate"]
        comparison = paired_group_bootstrap(a, b, draws=100)
        self.assertEqual(comparison, paired_group_bootstrap(a, b, draws=100))
        self.assertEqual(comparison["n_source_groups"], 14)
        self.assertAlmostEqual(comparison["delta"], sum(row["task_success"]-base.get("task_success", False) for base, row in zip(a, b))/15)
        with self.assertRaises(ValueError):
            paired_group_bootstrap(a, b[:-1])
        with self.assertRaises(ValueError):
            paired_group_bootstrap(a + [a[0]], b)


if __name__ == "__main__":
    unittest.main()
