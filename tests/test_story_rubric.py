"""Authored offline controls, not model generation or real reviewer ratings."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from dongxi_llms.run_identity import canonical_hash
from dongxi_llms.staged_campaign import evaluation_contracts
from dongxi_llms.story_rubric import (DIMENSIONS, SCHEMA, prepare_packet,
    evaluate_ratings, read_json, validate_contract, write_bundle)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT/"fixtures/story-rubric"


def fixture():
    return (read_json(FIXTURE/"contract.json"), read_json(FIXTURE/"checkpoints.json"),
            read_json(FIXTURE/"authored-records.jsonl", jsonl=True), read_json(FIXTURE/"raters.json"))


def signed(value, key):
    value[key] = canonical_hash({k: v for k, v in value.items() if k != key})
    return value


def full_records(contract, checkpoints, example):
    """Expand an explicit authored schema control, not hypothetical model data."""
    records = []
    for checkpoint in checkpoints:
        for item in contract["items"]:
            for recipe in range(4):
                record = copy.deepcopy(example)
                record.update(record_id=f"control-{checkpoint['checkpoint_id']}-{item['id']}-{recipe}",
                    checkpoint_id=checkpoint["checkpoint_id"], checkpoint_sha256=canonical_hash(checkpoint),
                    item_id=item["id"], source_group=item["source_group"], decoding_index=recipe,
                    opening_sha256=hashlib.sha256(item["prompt"].encode()).hexdigest())
                records.append(record)
    return records


def documents(packet, book, *, difference=True):
    links = {l["candidate_id"]: l["record_id"] for l in book["links"]}
    records = {r["record_id"]: r for r in book["records"]}
    out = []
    for index, rater in enumerate(book["raters"]):
        rows = []
        for candidate in packet["candidates"]:
            record = records[links[candidate["candidate_id"]]]
            arm = record["checkpoint_id"] == book["checkpoint_plan"][1]["checkpoint_id"]
            score = int(arm) if difference else 1
            values = {d: score for d in DIMENSIONS}
            if index == 1 and candidate == packet["candidates"][0]:
                values["ending"] = min(2, score+1)
            rows.append({"candidate_id": candidate["candidate_id"], "text_sha256": candidate["text_sha256"],
                         "scores": values, "abstention_reason": None,
                         "note": "Explicit authored numeric control; not a human or model assessment."})
        out.append({"schema_version": SCHEMA, "packet_sha256": packet["packet_sha256"],
                    "rubric_sha256": book["rubric_sha256"], "rater_id": rater["rater_id"], "ratings": rows})
    return out


class StoryRubricTests(unittest.TestCase):
    def setUp(self):
        self.contract, self.plan, self.records, self.raters = fixture()

    def packet(self, *, full=False, policy="two-rater-mean"):
        records = full_records(self.contract, self.plan, self.records[0]) if full else self.records
        return prepare_packet(self.contract, self.plan, records, self.raters, adjudication_policy=policy)

    def test_contract_is_exact_original_producer_not_reserialized_float_guess(self):
        self.assertEqual(self.contract, evaluation_contracts(ROOT)["story-publication-v1"])
        self.assertIs(validate_contract(self.contract), self.contract)

    def test_rehashed_changes_to_every_remaining_contract_field_rejected(self):
        for field, value in (("uncertainty_unit", "individual attempts"), ("required_raw", []),
                             ("stop_categories", ["natural-eos"]), ("contamination_audit", "passed"),
                             ("development_separation", "reuse training"), ("nll", "coherence score"),
                             ("rubric_scores", [False, True, 2]), ("raters", 1)):
            with self.subTest(field=field):
                altered = copy.deepcopy(self.contract); altered[field] = value
                signed(altered, "logical_contract_sha256")
                with self.assertRaises(ValueError): validate_contract(altered)

    def test_changed_panel_rubric_decoding_and_hash_rejected(self):
        for field in ("items", "rubric", "decoding", "predetermined_updates"):
            altered = copy.deepcopy(self.contract); altered[field] = []
            with self.assertRaises(ValueError): prepare_packet(altered, self.plan, self.records, self.raters)

    def test_packet_is_deterministic_blind_bounded_and_contains_complete_text(self):
        packet, book = self.packet()
        self.assertEqual((packet, book), self.packet())
        serialized = json.dumps(packet)
        for hidden in ("checkpoint_id", "update", "checkpoint_files", "generation_source", "record_id", "rater_id", "scores_by_record"):
            self.assertNotIn(hidden, serialized)
        self.assertEqual({c["text"] for c in packet["candidates"]}, {r["text"] for r in self.records})
        self.assertEqual(len(book["expected_cells"]), 96)
        self.assertEqual(sum(c["record_id"] is None for c in book["expected_cells"]), 92)
        self.assertEqual(book["unrepresented_predetermined_updates"], [0, 400, 4000, 8000])
        self.assertTrue(all(c["candidate_id"].startswith("C-") for c in packet["candidates"]))

    def test_preparation_empty_records_preserves_expected_cells(self):
        packet, book = prepare_packet(self.contract, self.plan, [], self.raters)
        report = evaluate_ratings(packet, book, [])
        self.assertEqual(report["status"], "awaiting-ratings")
        self.assertEqual(len(report["cells"]), 96)
        self.assertIsNone(report["paired_comparisons"][0]["equal_recipe_mean"])

    def test_partial_authored_scores_retain_disagreement_but_refuse_panel_interval(self):
        packet, book = self.packet(); docs = documents(packet, book)
        report = evaluate_ratings(packet, book, docs)
        paired = report["paired_comparisons"][0]
        self.assertEqual(paired["complete_pairs"], 2)
        self.assertEqual(paired["status"], "incomplete-coverage")
        self.assertIsNone(paired["equal_recipe_mean"])
        self.assertEqual(report["disagreement_candidates"], 1)
        self.assertEqual(report["rating_documents"], docs)
        self.assertEqual(report["provenance"], "authored-control")

    def test_complete_source_opening_bootstrap_is_reproducible_and_stratified(self):
        packet, book = self.packet(full=True); docs = documents(packet, book)
        report = evaluate_ratings(packet, book, docs, draws=100, seed=123)
        self.assertEqual(report, evaluate_ratings(packet, book, docs, draws=100, seed=123))
        paired = report["paired_comparisons"][0]
        self.assertEqual(paired["complete_pairs"], 48)
        for dimension in DIMENSIONS[:-1]:
            self.assertEqual(paired["equal_recipe_mean"][dimension]["delta"], 1)
            self.assertEqual(paired["equal_recipe_mean"][dimension]["interval"], [1, 1])
            self.assertEqual(paired["equal_recipe_mean"][dimension]["source_groups"], 12)
            self.assertTrue(all(v[dimension]["delta"] == 1 for v in paired["by_decoding"].values()))
        self.assertEqual(len(paired["per_source_opening_deltas"]), 12)
        baseline = report["checkpoint_summaries"][self.plan[0]["checkpoint_id"]]
        self.assertEqual(baseline["overall"]["expected_cells"], 48)
        self.assertEqual(len(baseline["by_source_opening"]), 12)
        self.assertEqual(len(baseline["overall"]["raw_rater_summaries"]), 2)

    def test_default_2000_draw_bound_and_invalid_draws_seeds(self):
        packet, book = self.packet()
        evaluate_ratings(packet, book, [], draws=2000)
        for kwargs in ({"draws": 0}, {"draws": True}, {"draws": 10000}, {"seed": -1}, {"seed": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                evaluate_ratings(packet, book, [], **kwargs)

    def test_pairing_is_frozen_before_scores_and_rejects_self_unknown_or_swapped(self):
        packet, book = self.packet()
        for pair in ((self.plan[1]["checkpoint_id"], self.plan[0]["checkpoint_id"]),
                     (self.plan[0]["checkpoint_id"], self.plan[0]["checkpoint_id"]), ("unknown", "other")):
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                evaluate_ratings(packet, book, [], comparisons=[pair])
        for changes in ({"update": 400}, {"interface_sha256": "0"*64}):
            plan = copy.deepcopy(self.plan); plan[1].update(changes)
            with self.assertRaises(ValueError): prepare_packet(self.contract, plan, [], self.raters)

    def test_distinct_raters_and_provenance_cannot_be_fabricated_from_controls(self):
        for edit in (lambda r:r[1].update(rater_id=r[0]["rater_id"]),
                     lambda r:r[0].update(shared_consultation=True),
                     lambda r:r[0].update(provenance="human")):
            raters = copy.deepcopy(self.raters); edit(raters)
            with self.assertRaises(ValueError): prepare_packet(self.contract, self.plan, self.records, raters)
        plan = copy.deepcopy(self.plan)
        for p in plan: p["provenance"] = "model-generated"
        with self.assertRaises(ValueError): prepare_packet(self.contract, plan, [], self.raters)

    def test_unknown_duplicate_mismatched_record_metadata_rejected(self):
        for edit in ({"item_id": "unknown"}, {"source_group": "other"}, {"opening_sha256": "0"*64},
                     {"checkpoint_sha256": "0"*64}, {"text": "changed"}, {"decoding_index": 4},
                     {"attempt_index": -1}, {"error": "failure not declared"}, {"truncated": False}):
            records = copy.deepcopy(self.records); records[0].update(edit)
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                prepare_packet(self.contract, self.plan, records, self.raters)
        with self.assertRaises(ValueError): prepare_packet(self.contract, self.plan, self.records+self.records[:1], self.raters)

    def test_unknown_rater_candidate_duplicate_and_changed_text_ratings_rejected(self):
        packet, book = self.packet(); original = documents(packet, book)
        changes = [lambda d:d[0].update(rater_id="unknown"),
            lambda d:d[0]["ratings"][0].update(candidate_id="unknown"),
            lambda d:d[0]["ratings"][0].update(text_sha256="0"*64),
            lambda d:d[0].update(rubric_sha256="0"*64),
            lambda d:d[0]["ratings"].append(d[0]["ratings"][0])]
        for edit in changes:
            docs = copy.deepcopy(original); edit(docs)
            with self.assertRaises(ValueError): evaluate_ratings(packet, book, docs)
        with self.assertRaises(ValueError): evaluate_ratings(packet, book, [original[0], original[0]])

    def test_scores_reject_invalid_ranges_bool_float_nan_missing_extra_dimensions(self):
        packet, book = self.packet(); original = documents(packet, book)
        for value in (-1, 3, True, .5, 1., float("nan"), "2"):
            docs = copy.deepcopy(original); docs[0]["ratings"][0]["scores"]["grammar"] = value
            with self.subTest(value=value), self.assertRaises(ValueError): evaluate_ratings(packet, book, docs)
        for alter in (lambda v:v.pop("grammar"), lambda v:v.update(extra=1)):
            docs = copy.deepcopy(original); alter(docs[0]["ratings"][0]["scores"])
            with self.assertRaises(ValueError): evaluate_ratings(packet, book, docs)

    def test_missing_rater_and_abstention_remain_null_not_zero(self):
        packet, book = self.packet(); docs = documents(packet, book)
        report = evaluate_ratings(packet, book, docs[:1])
        self.assertTrue(all(r["aggregate_scores"] is None for r in report["cells"]))
        docs[0]["ratings"][0].update(scores=None, abstention_reason="Cannot assess authored control")
        report = evaluate_ratings(packet, book, docs)
        cid = docs[0]["ratings"][0]["candidate_id"]
        self.assertIsNone(next(r for r in report["cells"] if r["candidate_id"] == cid)["aggregate_scores"])
        docs[0]["ratings"][0]["abstention_reason"] = ""
        with self.assertRaises(ValueError): evaluate_ratings(packet, book, docs)

    def test_packet_codebook_and_bijection_rehash_tampering_rejected(self):
        packet, book = self.packet()
        for field in ("links", "expected_cells", "raters", "records"):
            bad = copy.deepcopy(book); bad[field] = []; signed(bad, "codebook_sha256")
            with self.subTest(field=field), self.assertRaises(ValueError): evaluate_ratings(packet, bad, [])
        bad = copy.deepcopy(packet); bad["candidates"][0]["text"] += "changed"; signed(bad, "packet_sha256")
        with self.assertRaises(ValueError): evaluate_ratings(bad, book, [])

    def test_explicit_adjudication_retains_raw_scores_and_unresolved_null(self):
        packet, book = self.packet(policy="explicit-disagreement"); docs = documents(packet, book)
        before = evaluate_ratings(packet, book, docs)
        cid = packet["candidates"][0]["candidate_id"]
        original = next(r for r in before["cells"] if r["candidate_id"] == cid)
        self.assertIsNone(original["aggregate_scores"])
        scores = copy.deepcopy(docs[0]["ratings"][0]["scores"])
        decision = {"schema_version": SCHEMA, "packet_sha256": packet["packet_sha256"],
            "codebook_sha256": book["codebook_sha256"], "adjudicator": {"reviewer_id": "authored-adjudicator",
            "provenance": "authored-control", "declaration": "Authored decision control, not human review"},
            "ratings": [{"candidate_id": cid, "scores": scores, "rationale": "Explicit authored resolution"}]}
        report = evaluate_ratings(packet, book, docs, adjudication=decision)
        resolved = next(r for r in report["cells"] if r["candidate_id"] == cid)
        self.assertEqual(resolved["per_rater"], original["per_rater"])
        self.assertEqual(resolved["aggregate_scores"], scores)
        self.assertEqual(report["pending_adjudication_candidates"], 0)
        for alter in (lambda d:d["ratings"][0].update(rationale=""),
                      lambda d:d["ratings"][0]["scores"].update(grammar=2),
                      lambda d:d["ratings"][0].update(candidate_id="unknown")):
            bad = copy.deepcopy(decision); alter(bad)
            with self.assertRaises(ValueError): evaluate_ratings(packet, book, docs, adjudication=bad)

    def test_mean_policy_rejects_adjudication_and_unknown_policy(self):
        packet, book = self.packet()
        with self.assertRaises(ValueError): evaluate_ratings(packet, book, [], adjudication={})
        with self.assertRaises(ValueError): self.packet(policy="best-rater")

    def test_truncated_complete_text_is_rated_without_auto_ending_override(self):
        packet, book = self.packet(); docs = documents(packet, book)
        report = evaluate_ratings(packet, book, docs)
        capped = next(r for r in report["cells"] if r["truncated"] is True)
        self.assertEqual(capped["stop_reason"], "token-cap")
        self.assertIsNotNone(capped["aggregate_scores"])
        self.assertEqual({c["text"] for c in packet["candidates"]}, {r["text"] for r in self.records})

    def test_failed_extra_attempts_are_retained_charged_not_selected_as_replacements(self):
        extra = copy.deepcopy(self.records[0]); extra.update(record_id="failed-extra", attempt_index=1,
            stop_reason="failure", truncated=False, error="authored deliberate failure",
            cost={"generation_tokens": None, "wall_seconds": 7., "forward_positions": 42})
        packet, book = prepare_packet(self.contract, self.plan, self.records+[extra], self.raters)
        report = evaluate_ratings(packet, book, documents(packet, book))
        self.assertEqual(len(report["cells"]), 97)
        failed = next(r for r in report["cells"] if r["record_id"] == "failed-extra")
        self.assertEqual(failed["error"], "authored deliberate failure")
        summary = report["checkpoint_summaries"][extra["checkpoint_id"]]
        self.assertEqual(summary["overall"]["generated_cells"], 2)
        self.assertEqual(summary["all_attempts"], 3)
        self.assertEqual(summary["all_attempt_costs"]["wall_seconds"], {"known_total": 7., "unknown_attempts": 2})
        self.assertEqual(report["paired_comparisons"][0]["complete_pairs"], 2)

    def test_retry_without_primary_cannot_fill_missing_comparison_cell(self):
        records = copy.deepcopy(self.records); records[0]["attempt_index"] = 1
        packet, book = prepare_packet(self.contract, self.plan, records, self.raters)
        report = evaluate_ratings(packet, book, documents(packet, book))
        self.assertEqual(report["paired_comparisons"][0]["complete_pairs"], 1)

    def test_nonadditive_resource_peaks_and_fractional_cost_counts_rejected(self):
        for cost in ({"cuda_peak_allocated_bytes": 42}, {"forward_calls": True}, {"forward_positions": 1.5}):
            records = copy.deepcopy(self.records); records[0]["cost"].update(cost)
            with self.subTest(cost=cost), self.assertRaises(ValueError):
                prepare_packet(self.contract, self.plan, records, self.raters)

    def test_actual_token_stop_count_likelihood_cost_and_context_boundaries(self):
        record = copy.deepcopy(self.records[0]); record.update(token_ids=[1]*255+[50256],
            generated_tokens=256, prompt_tokens=20, selected_likelihoods=[-.1]*256,
            stop_reason="natural-eos", truncated=False, cost={"generation_tokens":256,"wall_seconds":1.})
        records = [record]; prepare_packet(self.contract, self.plan, records, self.raters)
        for edit in ({"generated_tokens": 255}, {"selected_likelihoods": [-.1]},
                     {"prompt_tokens": 1024}, {"stop_reason": "token-cap", "truncated": True},
                     {"stop_reason": "context-cap", "truncated": True}):
            bad = copy.deepcopy(record); bad.update(edit)
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                prepare_packet(self.contract, self.plan, [bad], self.raters)
        one = copy.deepcopy(record); one.update(token_ids=[50256],generated_tokens=1,selected_likelihoods=[-.1])
        for cost in (True, 1.):
            one["cost"]["generation_tokens"] = cost
            with self.assertRaises(ValueError): prepare_packet(self.contract, self.plan, [one], self.raters)

    def test_real_declared_records_cannot_have_unknown_tokens(self):
        plan = copy.deepcopy(self.plan); raters = copy.deepcopy(self.raters); records = copy.deepcopy(self.records)
        for p in plan: p["provenance"] = "model-generated"
        for r in raters: r["provenance"] = "ai"
        for r in records: r["checkpoint_sha256"] = canonical_hash(next(p for p in plan if p["checkpoint_id"] == r["checkpoint_id"]))
        with self.assertRaises(ValueError): prepare_packet(self.contract, plan, records, raters)

    def test_supplied_model_schema_control_defaults_to_private_shuffle_not_empirical_proof(self):
        plan = copy.deepcopy(self.plan); raters = copy.deepcopy(self.raters)
        for p in plan: p["provenance"] = "model-generated"
        for r in raters: r["provenance"] = "ai"
        record = copy.deepcopy(self.records[0])
        record.update(checkpoint_sha256=canonical_hash(plan[0]),token_ids=[1,50256],
                      prompt_tokens=10,generated_tokens=2,selected_likelihoods=[-1.,-.5],
                      stop_reason="natural-eos",truncated=False,cost={"generation_tokens":2,"wall_seconds":1.})
        with patch("dongxi_llms.story_rubric.secrets.randbits", return_value=998877) as private_rng:
            packet, book = prepare_packet(self.contract, plan, [record], raters)
        private_rng.assert_called_once_with(63)
        self.assertEqual(book["shuffle_seed"],998877)
        self.assertNotIn("998877",json.dumps(packet))
        self.assertEqual(evaluate_ratings(packet,book,[])["status"],"awaiting-ratings")
        self.assertIn("Not assessed",book["publication_clearance"])

    def test_json_reader_rejects_duplicate_keys_nonfinite_and_oversized(self):
        with tempfile.TemporaryDirectory() as directory:
            for i, text in enumerate(('{"a":1,"a":2}', '{"score":NaN}', '{"score":Infinity}')):
                path = Path(directory)/str(i); path.write_text(text)
                with self.assertRaises(ValueError): read_json(path)
            large = Path(directory)/"oversized"
            with large.open("wb") as handle:
                handle.truncate(16*1024*1024+1)
            with self.assertRaises(ValueError): read_json(large)

    def test_exclusive_bundle_refuses_even_empty_existing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError): write_bundle(directory, {"report.json": {}})
            output = Path(directory)/"new"; receipt = write_bundle(output, {"report.json":{"authored":True}})
            before = (output/"report.json").read_bytes()
            with self.assertRaises(FileExistsError): write_bundle(output, {"report.json":{"changed":True}})
            self.assertEqual(before, (output/"report.json").read_bytes())
            self.assertEqual(receipt["artifact_sha256"]["report.json"], hashlib.sha256(before).hexdigest())

    def test_actual_cli_prepare_evaluate_uses_empty_templates_and_no_overwrite(self):
        env = {**os.environ,"PYTHONDONTWRITEBYTECODE":"1","CUDA_VISIBLE_DEVICES":"", "HF_HUB_OFFLINE":"1"}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"packet"; result = Path(directory)/"result"
            command = [sys.executable,str(ROOT/"scripts/evaluate_story_ratings.py"),"prepare",
                "--contract",str(FIXTURE/"contract.json"),"--checkpoints",str(FIXTURE/"checkpoints.json"),
                "--records",str(FIXTURE/"authored-records.jsonl"),"--raters",str(FIXTURE/"raters.json"),"--output",str(output)]
            first = subprocess.run(command,capture_output=True,text=True,env=env,timeout=20)
            self.assertEqual(first.returncode,0,first.stderr)
            self.assertEqual(read_json(output/"ratings-1.json")["ratings"],[])
            self.assertEqual(read_json(output/"ratings-2.json")["ratings"],[])
            repeat = subprocess.run(command,capture_output=True,text=True,env=env,timeout=20)
            self.assertNotEqual(repeat.returncode,0)
            evaluate = subprocess.run([sys.executable,str(ROOT/"scripts/evaluate_story_ratings.py"),"evaluate",
                "--bundle",str(output),"--output",str(result)],capture_output=True,text=True,env=env,timeout=20)
            self.assertEqual(evaluate.returncode,0,evaluate.stderr)
            report = read_json(result/"report.json")
            self.assertEqual(report["status"],"awaiting-ratings")
            self.assertIsNone(report["paired_comparisons"][0]["equal_recipe_mean"])

    def test_import_and_helpers_do_not_import_model_libraries(self):
        completed = subprocess.run([sys.executable,"-B","-c",
            "import sys;sys.path.insert(0,'src');import dongxi_llms.story_rubric;"
            "assert not any(k=='torch' or k.startswith('torch.') or k=='transformers' or k.startswith('transformers.') for k in sys.modules)"],
            cwd=ROOT,capture_output=True,text=True,timeout=20)
        self.assertEqual(completed.returncode,0,completed.stderr)


if __name__ == "__main__":
    unittest.main()
