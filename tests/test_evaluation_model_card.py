"""Offline export controls using original authored and retained adapter evidence."""
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

from dongxi_llms.evaluation_model_card import (build_evaluation_card,
    export_model_card, render_model_card, _read)
from dongxi_llms.reasoning_evaluation import freeze_contract, replay_records, paired_group_bootstrap
from dongxi_llms.run_identity import canonical_hash

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures/reasoning-evaluation"
ADAPTER_REPORT = ROOT / "experiments/reports/2026-10-04-reasoning-generation-adapter-final.json"


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def refresh_identity(value):
    value["identity_sha256"] = canonical_hash({k: v for k, v in value.items() if k != "identity_sha256"})


class EvaluationModelCardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items = json.loads((FIXTURES / "items.json").read_text())
        cls.settings = json.loads((FIXTURES / "settings.json").read_text())
        cls.records = [json.loads(line) for line in (FIXTURES / "responses.jsonl").read_text().splitlines()]
        cls.contract = freeze_contract(cls.items, cls.settings)
        # Read actual retained October 4 observations only. No new model load,
        # tokenizer operation or artificial remeasurement occurs in these tests.
        cls.adapter = json.loads(ADAPTER_REPORT.read_text())

    def inputs(self, directory, records=None):
        items, contract, responses = (Path(directory) / name for name in ("items.json", "contract.json", "responses.jsonl"))
        write_json(items, self.items)
        write_json(contract, self.contract)
        with responses.open("x", encoding="utf-8") as handle:
            for row in self.records if records is None else records:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
        return dict(items_path=items, contract_path=contract, records_path=responses)

    def test_authored_card_deterministic_and_matches_authoritative_replay(self):
        card, replay = build_evaluation_card(self.items, self.records, self.contract)
        again, replay_again = build_evaluation_card(self.items, self.records, self.contract)
        self.assertEqual((card, replay), (again, replay_again))
        self.assertEqual(replay, replay_records(self.items, self.records, self.contract))
        self.assertEqual(card["evaluation"]["raw_record_count"], 30)
        self.assertEqual(card["evaluation"]["replayed_record_count"], 30)
        self.assertEqual(card["replay_sha256"], canonical_hash(replay))
        self.assertEqual(card["card_sha256"], canonical_hash({k: v for k, v in card.items() if k != "card_sha256"}))
        self.assertEqual(render_model_card(card), render_model_card(again))

    def test_canonical_authored_fixture_contract_matches_existing_records(self):
        frozen = json.loads((FIXTURES / "contract.json").read_text())
        self.assertEqual(frozen, self.contract)
        self.assertEqual({row["contract_id"] for row in self.records}, {frozen["identity"]})
        self.assertEqual(build_evaluation_card(self.items, self.records, frozen)[1],
                         replay_records(self.items, self.records, self.contract))

    def test_authored_cost_unknowns_and_missing_model_claims_are_explicit(self):
        card, _ = build_evaluation_card(self.items, self.records, self.contract)
        for model in card["models"].values():
            self.assertEqual(model["provenance_status"], "unverified or partially unavailable")
            self.assertEqual(model["hardware"]["status"], "unavailable")
            for field in ("training", "model_architecture", "genealogy"):
                self.assertIsNone(model[field]["value"])
            for value in model["recorded_costs"].values():
                self.assertEqual(value, {"known_total": 0, "known_rows": 0, "unknown_rows": 15})
        for field in ("approval", "independent_behavioral_review", "broad_capability_and_safety"):
            self.assertEqual(card[field]["status"], "unavailable")
            self.assertIsNone(card[field]["value"])
        self.assertEqual(card["generation"]["status"], "not executed by exporter")
        self.assertIn("Unknown costs are not zero measurements", render_model_card(card))

    def test_errors_unsupported_and_negative_slices_are_not_selected_away(self):
        card, replay = build_evaluation_card(self.items, self.records, self.contract)
        self.assertEqual([row["raw_response"] for row in replay["rows"]], [row["raw_response"] for row in self.records])
        self.assertEqual([row["error"] for row in replay["rows"]], [row["error"] for row in self.records])
        self.assertEqual(card["evaluation"]["errors"], 1)
        self.assertGreater(card["evaluation"]["unsupported"], 0)
        a, b = (card["models"][name]["results"]["slices"]["task"] for name in ("authored-baseline", "authored-candidate"))
        self.assertLess(b["sets"]["accuracy"], a["sets"]["accuracy"])
        self.assertLess(b["json-format"]["format_valid"], a["json-format"]["format_valid"])
        self.assertIn("no best-of-N", card["selection"]["rule"])

    def test_source_group_pairing_delegates_exactly_to_replay_instrument(self):
        card, replay = build_evaluation_card(self.items, self.records, self.contract,
            comparisons=[("authored-baseline", "authored-candidate")], draws=100, seed=123)
        a, b = ([row for row in replay["rows"] if row["checkpoint_id"] == name] for name in ("authored-baseline", "authored-candidate"))
        expected = paired_group_bootstrap(a, b, draws=100, seed=123)
        expected.pop("samples")
        actual = card["evaluation"]["paired_comparisons"][0]
        self.assertEqual(actual, dict(expected, baseline="authored-baseline", candidate="authored-candidate"))
        self.assertEqual(actual["n_source_groups"], 14)

    def test_pairing_refuses_incomplete_coverage_instead_of_intersection(self):
        with self.assertRaisesRegex(ValueError, "identical item/sample coverage"):
            build_evaluation_card(self.items, self.records[:-1], self.contract,
                comparisons=[("authored-baseline", "authored-candidate")], draws=10)
        card, _ = build_evaluation_card(self.items, self.records[:-1], self.contract)
        self.assertEqual(card["models"]["authored-candidate"]["results"]["summary"]["n"], 14)
        self.assertEqual(len(card["models"]["authored-candidate"]["results"]["missing_item_ids"]), 1)

    def test_frozen_contract_suite_duplicate_and_group_mismatch_refuse(self):
        contract = copy.deepcopy(self.contract)
        contract["settings"]["thinking_mode"] = "altered"
        with self.assertRaises(ValueError):
            build_evaluation_card(self.items, self.records, contract)
        items = copy.deepcopy(self.items)
        items[0]["prompt"] += " changed"
        with self.assertRaises(ValueError):
            build_evaluation_card(items, self.records, self.contract)
        with self.assertRaises(ValueError):
            build_evaluation_card(self.items, self.records + [self.records[0]], self.contract)
        items = copy.deepcopy(self.items)
        items[1]["split"] = "test"
        forged = copy.deepcopy(self.contract)
        forged["suite_sha256"] = canonical_hash(items)
        forged["identity"] = canonical_hash({k: v for k, v in forged.items() if k != "identity"})
        with self.assertRaises(ValueError):
            build_evaluation_card(items, [], forged)

    def test_no_text_execution_and_no_changes_to_original_inputs(self):
        records = copy.deepcopy(self.records)
        records[0]["raw_response"] = "__import__('os').system('not-a-command')"
        before = copy.deepcopy((self.items, records, self.contract))
        with patch("os.system", side_effect=AssertionError("No execution")), patch("subprocess.run", side_effect=AssertionError("No probes")):
            card, replay = build_evaluation_card(self.items, records, self.contract)
        self.assertEqual((self.items, records, self.contract), before)
        self.assertEqual(replay["rows"][0]["status"], "UNSUPPORTED")
        self.assertEqual(card["generation"]["status"], "not executed by exporter")

    def test_actual_retained_random_adapter_records_and_identity(self):
        invocation = self.adapter["invocations"]["raw"]
        items = self.adapter["fixture_items"]
        card, replay = build_evaluation_card(items, invocation["raw_records"], invocation["contract"],
            run_identities=[invocation["identity"]])
        self.assertEqual(replay, invocation["evaluation"])
        self.assertEqual(card["evaluation"]["replayed_record_count"], 4)
        model = next(iter(card["models"].values()))
        self.assertEqual(model["unlinked_record_count"], 0)
        self.assertEqual(model["recorded_costs"]["generation_tokens"]["known_rows"], 4)
        self.assertGreater(model["recorded_costs"]["attempted_forward_tokens"]["known_total"], 0)
        self.assertEqual(model["hardware"]["status"], "recorded metadata only")
        self.assertIsNone(model["model_architecture"]["value"])
        self.assertIsNone(model["genealogy"]["value"])

    def test_retained_scripted_adapter_error_preserves_partial_cost_and_origin(self):
        control = next(c for c in self.adapter["scripted_controls"] if c["label"] == "partial-forward-error")
        # These are timed adapter calls to an explicitly scripted fault source,
        # NOT a neural generation or a newly measured failure.
        card, replay = build_evaluation_card(self.adapter["fixture_items"], [control["record"]], control["contract"],
            evidence={"origin": control["origin"], "report": str(ADAPTER_REPORT)})
        row = replay["rows"][0]
        self.assertEqual(row["error"], control["record"]["error"])
        self.assertEqual(row["token_ids"], control["record"]["token_ids"])
        self.assertEqual(row["cost"], control["record"]["cost"])
        self.assertEqual(row["status"], "ERROR")
        model = next(iter(card["models"].values()))
        self.assertGreater(model["recorded_costs"]["attempted_forward_tokens"]["known_total"], model["recorded_costs"]["model_forward_tokens"]["known_total"])
        self.assertEqual(model["unlinked_record_count"], 1)
        self.assertIn("NOT an HF model output", card["evidence"]["origin"])

    def test_supplied_identity_hash_settings_interface_and_input_maps_must_match(self):
        invocation = self.adapter["invocations"]["raw"]
        original = invocation["identity"]
        for kind in ("hash", "settings", "interface", "input", "checkpoint"):
            with self.subTest(kind=kind):
                identity = copy.deepcopy(original)
                if kind == "hash":
                    identity["environment"]["python_version"] = "not-observed"
                elif kind == "settings":
                    identity["config"]["max_new_tokens"] += 1
                    refresh_identity(identity)
                elif kind == "interface":
                    identity["checkpoint_interface"]["tokenizer"]["vocab_size"] += 1
                    refresh_identity(identity)
                elif kind == "input":
                    identity["input_sha256"] = {"wrong": "a" * 64}
                    refresh_identity(identity)
                else:
                    identity["checkpoint_files"]["config.json"] = "a" * 64
                    refresh_identity(identity)
                records = copy.deepcopy(invocation["raw_records"])
                if kind != "hash":
                    for row in records:
                        row["input_identity_sha256"] = identity["identity_sha256"]
                with self.assertRaises(ValueError):
                    build_evaluation_card(self.adapter["fixture_items"], records, invocation["contract"], run_identities=[identity])
        with self.assertRaises(ValueError):
            build_evaluation_card(self.items, self.records, self.contract, run_identities=[original])

    def test_export_hashes_actual_inputs_sources_and_keeps_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.inputs(directory)
            output = Path(directory) / "card"
            original = {key: path.read_bytes() for key, path in paths.items()}
            card = export_model_card(**paths, output=output)
            self.assertEqual(set(p.name for p in output.iterdir()), {"model-card.json", "model-card.md", "replay.json"})
            self.assertEqual(json.loads((output / "model-card.json").read_text()), card)
            self.assertEqual(json.loads((output / "replay.json").read_text())["rows"][0]["raw_response"], self.records[0]["raw_response"])
            for role, binding in card["evidence"]["inputs"].items():
                data = Path(binding["path"]).read_bytes()
                self.assertEqual(binding["sha256"], hashlib.sha256(data).hexdigest())
                self.assertEqual(binding["bytes"], len(data))
            for binding in card["evidence"]["sources"].values():
                self.assertEqual(binding["sha256"], hashlib.sha256(Path(binding["path"]).read_bytes()).hexdigest())
            self.assertEqual({key: path.read_bytes() for key, path in paths.items()}, original)

    def test_output_existing_input_and_symlink_paths_never_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.inputs(directory)
            original = {key: path.read_bytes() for key, path in paths.items()}
            for output in (paths["items_path"], Path(directory), paths["records_path"]):
                with self.subTest(output=str(output)), self.assertRaises(FileExistsError):
                    export_model_card(**paths, output=output)
            linked = Path(directory) / "linked"
            linked.symlink_to(paths["items_path"])
            with self.assertRaises(ValueError):
                export_model_card(**paths, output=linked)
            self.assertEqual({key: path.read_bytes() for key, path in paths.items()}, original)

    def test_bad_input_or_pair_refuses_before_any_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.inputs(directory, self.records[:-1])
            output = Path(directory) / "new-card"
            with self.assertRaises(ValueError):
                export_model_card(**paths, output=output, comparisons=[("authored-baseline", "authored-candidate")], draws=10)
            self.assertFalse(output.exists())
            paths["contract_path"].write_text('{"identity":"first","identity":"second"}')
            with self.assertRaises(ValueError):
                export_model_card(**paths, output=output)
            self.assertFalse(output.exists())

    def test_bounded_regular_reads_reject_fifo_symlink_oversize_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fifo"
            os.mkfifo(path)
            with self.assertRaises(ValueError):
                _read(path, 100)
            regular = Path(directory) / "regular"
            regular.write_bytes(b"123")
            with self.assertRaises(ValueError):
                _read(regular, 2)
            link = Path(directory) / "link"
            link.symlink_to(regular)
            with self.assertRaises(ValueError):
                _read(link, 100)
            paths = self.inputs(directory)
            paths["contract_path"].write_text('{"identity": NaN}')
            with self.assertRaises(ValueError):
                export_model_card(**paths, output=Path(directory) / "new")

    def test_empty_ledger_has_no_model_scores_not_zero_accuracy(self):
        card, _ = build_evaluation_card(self.items, [], self.contract)
        self.assertEqual(card["models"], {})
        self.assertEqual(card["evaluation"]["replayed_record_count"], 0)
        self.assertIn("no model score is available", render_model_card(card))

    def test_external_labels_are_not_markdown_html_or_active_links(self):
        records = copy.deepcopy(self.records)
        label = '<script>alert(1)</script>|\n# Heading [link](javascript:1) `code`'
        for row in records:
            row["checkpoint_id"] = label if row["checkpoint_id"] == "authored-baseline" else "candidate"
        card, _ = build_evaluation_card(self.items, records, self.contract)
        text = render_model_card(card)
        self.assertNotIn("<script>", text)
        self.assertNotIn("\n# Heading", text)
        self.assertNotIn("[link](javascript:", text)
        self.assertIn("&#124;", text)

    def test_strict_draw_seed_pair_and_bootstrap_work_bounds(self):
        for kw in ({"draws": True}, {"draws": 10001}, {"seed": -1}, {"seed": True},
                   {"comparisons": [["authored-baseline"]]},
                   {"comparisons": [["authored-baseline", "absent"]]},
                   {"comparisons": [["authored-baseline", "authored-baseline"]]}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                build_evaluation_card(self.items, self.records, self.contract, **kw)
        with patch("dongxi_llms.evaluation_model_card.MAX_BOOTSTRAP_WORK", 10):
            with self.assertRaisesRegex(ValueError, "work bound"):
                build_evaluation_card(self.items, self.records, self.contract, comparisons=[("authored-baseline", "authored-candidate")], draws=10)

    def test_real_offline_cli_two_exports_byte_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.inputs(directory)
            outputs = []
            for i in range(2):
                output = Path(directory) / f"card-{i}"
                command = [sys.executable, str(ROOT / "scripts/export_evaluation_model_card.py"),
                    "--items", str(paths["items_path"]), "--contract", str(paths["contract_path"]),
                    "--records", str(paths["records_path"]), "--output", str(output),
                    "--compare", "authored-baseline", "authored-candidate", "--draws", "100", "--seed", "1010"]
                result = subprocess.run(command, text=True, capture_output=True, timeout=10,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout)["replayed_record_count"], 30)
                outputs.append({p.name: p.read_bytes() for p in output.iterdir()})
            self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
