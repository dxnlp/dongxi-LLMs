"""Independent contract and controlled-failure tests for preference collection."""
from dataclasses import replace
import json
from pathlib import Path
import unittest

from dongxi_llms.preference_audit import (
    Candidate, JudgeIdentity, MAX_RAW_VERDICT_CHARS, RUBRIC, audit, blind_presentation, collect_judgment,
    load_fixture, perturb_fixture, ranking_to_pairs, ratings_to_pairs,
    rubric_hash, run_reference, simulated_collection, source_pair_weights,
    validate_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures/preference-audit/pairs.json"


class PreferenceAuditTests(unittest.TestCase):
    def setUp(self):
        self.pairs, self.labels = load_fixture(FIXTURE)
        self.pair = self.pairs[0]
        self.judge = JudgeIdentity("test-judge", "authored", "v1", RUBRIC["id"],
                                   rubric_hash(), {"method": "manual test literal"})

    def raw(self, verdict):
        return json.dumps({"verdict": verdict, "reason": "Original test reason"})

    def test_blind_order_maps_to_same_response(self):
        forward = collect_judgment(self.pair, self.judge, self.raw("A"))
        reversed_order = collect_judgment(self.pair, self.judge, self.raw("B"), swapped=True)
        self.assertEqual(forward.outcome, reversed_order.outcome)
        self.assertEqual(forward.selected_candidate_id, self.pair.left.candidate_id)
        shown = blind_presentation(self.pair)
        self.assertEqual(shown["candidates"]["A"], self.pair.left.text)
        self.assertNotIn(self.pair.left.checkpoint_id, json.dumps(shown))
        self.assertNotIn(self.pair.left.candidate_id, json.dumps(shown))

    def test_swapping_canonical_pair_reverses_review(self):
        swapped = self.pair.swapped()
        self.assertEqual(swapped.left, self.pair.right)
        label = self.labels[0].swapped()
        self.assertEqual(label.outcome, "right")
        self.assertEqual(self.labels[0].swapped().swapped(), self.labels[0])
        collected = collect_judgment(swapped, self.judge, self.raw("B"))
        self.assertEqual(collected.selected_candidate_id, self.pair.left.candidate_id)
        self.assertEqual(collected.outcome, label.outcome)

    def test_tie_abstention_invalid_are_separate(self):
        for outcome in ("tie", "abstain"):
            record = collect_judgment(self.pair, self.judge, self.raw(outcome))
            self.assertEqual(record.outcome, outcome)
            self.assertIsNone(record.selected_candidate_id)
            self.assertIsNone(record.error)
        for raw in ('A is best', '{"verdict":"A","reason":"ok","verdict":"B"}',
                    '{"verdict":"A","reason":""}', '{"verdict":"A","reason":"ok","extra":1}',
                    '{"verdict":[],"reason":"invalid"}'):
            record = collect_judgment(self.pair, self.judge, raw)
            self.assertEqual(record.outcome, "invalid")
            self.assertEqual(record.raw_verdict, raw)
            self.assertEqual(record.error_stage, "parse")

    def test_transport_failure_retains_partial_raw_response(self):
        raw = '{"verdict":"A"'
        record = collect_judgment(self.pair, self.judge, raw, transport_error="test timeout")
        self.assertEqual(record.raw_verdict, raw)
        self.assertEqual(record.error_stage, "transport")
        self.assertEqual(record.error, "test timeout")

    def test_deeply_nested_verdict_is_retained_instead_of_crashing(self):
        raw = '{"verdict":"A","reason":' + '[' * 10_000 + '0' + ']' * 10_000 + '}'
        self.assertLess(len(raw), MAX_RAW_VERDICT_CHARS)
        record = collect_judgment(self.pair, self.judge, raw)
        self.assertEqual(record.outcome, "invalid")
        self.assertEqual(record.raw_verdict, raw)
        self.assertEqual(record.error_stage, "parse")
        self.assertIn("nesting depth", record.error)
        self.assertIsNone(record.selected_candidate_id)
        # Raw-record replay must also retain the failure rather than crashing.
        report = audit(self.pairs, self.labels, [record])
        self.assertEqual(report["failures"][0]["raw_verdict"], raw)

    def test_oversized_verdict_is_not_decoded_or_truncated(self):
        raw = json.dumps({"verdict": "A", "reason": "x" * MAX_RAW_VERDICT_CHARS})
        record = collect_judgment(self.pair, self.judge, raw)
        self.assertEqual(record.outcome, "invalid")
        self.assertEqual(record.raw_verdict, raw)
        self.assertEqual(record.error_stage, "parse")
        self.assertIn("character parsing limit", record.error)
        self.assertIsNone(record.reason)

    def test_identity_and_repeats_are_validated(self):
        for repeat in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                collect_judgment(self.pair, self.judge, self.raw("A"), repeat=repeat)
        with self.assertRaises(ValueError):
            collect_judgment(self.pair, replace(self.judge, rubric_sha256="0" * 64), self.raw("A"))
        with self.assertRaises(ValueError):
            replace(self.judge, provenance="ai")
        with self.assertRaises(ValueError):
            replace(self.pair.left, provenance="ai")
        with self.assertRaises(ValueError):
            Candidate("id", "text", "imaginary", "checkpoint", {"mode": "test"})

    def test_ordinal_ratings_keep_tie_and_missing_separate(self):
        result = ratings_to_pairs({"a": 5, "b": 5, "c": 1, "d": None})
        self.assertEqual(len(result), 6)
        self.assertEqual(result[0]["outcome"], "tie")
        self.assertEqual(result[1]["outcome"], "left")
        self.assertEqual(result[2]["outcome"], "abstain")
        self.assertEqual(result[2]["raw_ratings"], {"a": 5, "d": None})
        for value in (float("nan"), float("inf"), True, 6, "5"):
            with self.assertRaises(ValueError):
                ratings_to_pairs({"a": value, "b": 2})

    def test_ranking_converts_tie_groups_without_losing_raw(self):
        ranking = [["c"], ["a", "b"]]
        result = ranking_to_pairs(ranking, candidate_ids=["a", "b", "c"])
        self.assertEqual([row["outcome"] for row in result], ["tie", "right", "right"])
        self.assertEqual(result[0]["raw_ranking"], ranking)
        for invalid in ([["a"], ["a"]], [["a"]], [[], ["a", "b", "c"]], ["a", "b", "c"]):
            with self.assertRaises(ValueError):
                ranking_to_pairs(invalid, candidate_ids=["a", "b", "c"])

    def test_source_and_family_split_collision_rejected(self):
        changed = list(self.pairs)
        changed[1] = replace(changed[1], split="test")
        with self.assertRaisesRegex(ValueError, "split collision"):
            validate_dataset(changed, self.labels)
        changed = list(self.pairs)
        changed[-1] = replace(changed[-1], family_id=changed[0].family_id)
        with self.assertRaisesRegex(ValueError, "split collision"):
            validate_dataset(changed, self.labels)

    def test_multiturn_siblings_cannot_hide_behind_new_source_id(self):
        changed = list(self.pairs)
        changed[1] = replace(changed[1], source_prompt_id="fake-new-source")
        with self.assertRaisesRegex(ValueError, "Multi-turn"):
            validate_dataset(changed, self.labels)

    def test_content_id_and_condition_collisions_rejected(self):
        changed = list(self.pairs)
        changed[1] = replace(changed[1], left=replace(changed[1].left, candidate_id=changed[0].left.candidate_id))
        with self.assertRaisesRegex(ValueError, "different content"):
            validate_dataset(changed, self.labels)
        changed = list(self.pairs)
        changed[1] = replace(changed[1], base_pair_id=changed[0].base_pair_id,
                             turn_id=changed[0].turn_id, prompt=changed[0].prompt)
        with self.assertRaisesRegex(ValueError, "duplicate condition"):
            validate_dataset(changed, self.labels)

    def test_source_weights_sum_to_one_with_variants_and_turns(self):
        pairs, labels = perturb_fixture(self.pairs, self.labels)
        weights = source_pair_weights(pairs)
        for source in {p.source_prompt_id for p in pairs}:
            self.assertAlmostEqual(sum(weights[p.pair_id] for p in pairs if p.source_prompt_id == source), 1.)
        self.assertAlmostEqual(weights["box-turn1"], 1 / 9)
        self.assertAlmostEqual(weights["addition"], 1 / 3)
        self.assertEqual(len(labels), 24)

    def test_perturbation_cannot_change_the_underlying_question(self):
        pairs, labels = perturb_fixture(self.pairs, self.labels)
        changed = list(pairs)
        changed[8] = replace(changed[8], prompt="An unrelated question with the same source tag")
        with self.assertRaisesRegex(ValueError, "inconsistent source identity"):
            validate_dataset(changed, labels)

    def test_duplicate_records_and_forged_evidence_rejected(self):
        judgment = collect_judgment(self.pair, self.judge, self.raw("A"))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            audit(self.pairs, self.labels, [judgment, judgment])
        forged = replace(judgment, raw_verdict=self.raw("B"))
        with self.assertRaisesRegex(ValueError, "replay"):
            audit(self.pairs, self.labels, [forged])
        forged = replace(judgment, record_id="new-but-fake-id")
        with self.assertRaisesRegex(ValueError, "replay"):
            audit(self.pairs, self.labels, [forged])

    def test_empty_denominators_are_null_not_zero(self):
        record = collect_judgment(self.pair, self.judge, self.raw("abstain"))
        row = audit(self.pairs, self.labels, [record])["judges"]["test-judge"]
        self.assertIsNone(row["decisive_agreement"])
        self.assertIsNone(row["order_consistency"])
        self.assertIsNone(row["repeat_disagreement"])
        self.assertEqual(row["all_outcome_agreement"], 0.)

    def test_judge_cannot_grade_itself_as_independent_reference(self):
        judge = replace(self.judge, judge_id=self.labels[0].reviewer_id)
        record = collect_judgment(self.pair, judge, self.raw("A"))
        with self.assertRaisesRegex(ValueError, "independent reviewed"):
            audit(self.pairs, self.labels, [record])

    def test_pair_rows_cannot_silently_dominate_source_average(self):
        rows = []
        for pair in self.pairs:
            correct = pair.source_prompt_id == "box-story"
            label = next(label for label in self.labels if label.pair_id == pair.pair_id)
            verdict = "A" if correct and label.outcome == "left" else "B" if correct else "abstain"
            rows.append(collect_judgment(pair, self.judge, self.raw(verdict)))
        result = audit(self.pairs, self.labels, rows)["judges"]["test-judge"]
        # Three correct box turns plus the correctly abstained unobservable source.
        self.assertEqual(result["all_outcome_agreement"], 4 / 8)
        self.assertAlmostEqual(result["source_balanced_all_outcome_agreement"], 2 / 6)

    def test_simulated_controls_expose_bias_and_keep_every_failure(self):
        result = run_reference(FIXTURE)
        self.assertEqual(result["judgment_count"], 484)
        self.assertEqual(len(result["raw_judgments"]), 484)
        self.assertEqual(len(result["failures"]), 4)
        self.assertEqual(result["review_provenance"], {"authored": 24})
        self.assertEqual(result["judgment_provenance"], {"simulated": 484})
        judges = result["judges"]
        self.assertEqual(judges["content-rule"]["all_outcome_agreement"], 1.)
        self.assertEqual(judges["first-slot"]["first_position_choice_rate"], 1.)
        self.assertEqual(judges["first-slot"]["order_consistency"], .25)
        self.assertEqual(judges["repeat-unstable"]["repeat_disagreement"], .75)
        self.assertGreater(judges["injection-sensitive"]["perturbations"]["injection"]["canonical_outcome_change_rate"], 0.)
        self.assertGreater(judges["longer-answer"]["perturbations"]["verbosity"]["canonical_outcome_change_rate"], 0.)

    def test_repeat_and_order_identity_are_not_independent_examples(self):
        pairs, labels = perturb_fixture(self.pairs, self.labels)
        rows = simulated_collection(pairs)
        result = audit(pairs, labels, rows)["judges"]["content-rule"]
        self.assertEqual(result["order_paired_observations"], 48)
        self.assertEqual(result["repeat_paired_observations"], 48)
        self.assertEqual(result["source_groups"], 6)


if __name__ == "__main__":
    unittest.main()
