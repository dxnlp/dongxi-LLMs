"""Independent arithmetic/refusal tests; no training or shared-reader claim."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

from dongxi_llms import snapshot_io_schedule as schedule

ROOT = Path(__file__).resolve().parents[1]
SPEC = "experiments/specs/2026-10-05-snapshot-io-schedule.md"
ADDENDUM = "experiments/specs/2026-10-05-snapshot-io-schedule-node-bound.md"
SOURCES = ("src/dongxi_llms/snapshot_io_schedule.py", "tests/test_snapshot_io_schedule.py", SPEC, ADDENDUM)
PRESERVED = ("src/dongxi_llms/dpo_stage_budget.py", "tests/test_dpo_stage_budget.py",
    "experiments/specs/2026-10-05-dpo-validation-stage-mapping.md",
    "experiments/reports/2026-10-05-dpo-validation-stage-mapping/run-03/verification.json")
ENVELOPE = dict(max_payload_bytes=16777216, max_tree_nodes=4096,
                max_tensor_elements=21646, max_tensor_bytes=56248,
                max_primitive_bytes=262144)


def requirements(**changes):
    options = dict(updates=2, checkpoint_every=2,
                   diagnostic_loads_per_attempt=1, complete_attempt_capacity=2)
    options.update(changes)
    return schedule.build_requirements(ENVELOPE, **options)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def rehash(value):
    value["requirements_sha256"] = canonical_hash({key: item for key, item in value.items()
                                                   if key != "requirements_sha256"})
    return value


def independent(events, envelope=ENVELOPE):
    """Literal event enumeration, not a call to production cost/sum functions."""
    result = {key: 0 for key in schedule.KEYS}
    for event in events:
        result[f"snapshot_{event}_operations"] += 1
        result["snapshot_hash_bytes"] += envelope["max_payload_bytes"]
        result["snapshot_tree_nodes"] += envelope["max_tree_nodes"]
        result["snapshot_primitive_bytes"] += envelope["max_primitive_bytes"]
        if event != "inspect": result["snapshot_tensor_elements"] += envelope["max_tensor_elements"]
        if event == "save":
            result["snapshot_clone_bytes"] += envelope["max_tensor_bytes"]
            result["snapshot_serialization_bytes"] += envelope["max_payload_bytes"]
    return result


class SnapshotIOScheduleTests(unittest.TestCase):
    def test_envelope_copy_and_exact_contract(self):
        observed = schedule.validate_envelope(ENVELOPE)
        self.assertEqual(observed, ENVELOPE); observed["max_tree_nodes"] = 1
        self.assertEqual(ENVELOPE["max_tree_nodes"], 4096)

    def test_missing_extra_and_old_envelope_fields_refuse(self):
        for key in ENVELOPE:
            value = deepcopy(ENVELOPE); del value[key]
            with self.subTest(key=key), self.assertRaises(ValueError): schedule.validate_envelope(value)
        for value in ({**ENVELOPE, "schema": "old"}, [], {**ENVELOPE, "max_bytes": 1}):
            with self.assertRaises(ValueError): schedule.validate_envelope(value)

    def test_bool_float_negative_and_oversized_envelopes_refuse(self):
        for key in ENVELOPE:
            for bad in (True, 1.0, -1, 2**63):
                with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                    schedule.validate_envelope({**ENVELOPE, key: bad})

    def test_payload_and_nodes_positive_other_dimensions_can_be_zero(self):
        for key in ("max_payload_bytes", "max_tree_nodes"):
            with self.assertRaises(ValueError): schedule.validate_envelope({**ENVELOPE, key: 0})
        value = dict(max_payload_bytes=1, max_tree_nodes=1, max_tensor_elements=0,
                     max_tensor_bytes=0, max_primitive_bytes=0)
        self.assertEqual(schedule.validate_envelope(value), value)

    def test_tensor_and_primitive_byte_bounds_refuse_before_cost(self):
        for key in ("max_tensor_bytes", "max_primitive_bytes"):
            with self.assertRaises(ValueError): schedule.operation_cost({**ENVELOPE, key: 2**25}, "save")

    def test_one_million_node_consumer_boundary(self):
        self.assertEqual(schedule.validate_envelope({**ENVELOPE, "max_tree_nodes": 1_000_000})["max_tree_nodes"], 1_000_000)
        with self.assertRaisesRegex(ValueError, "consumer bound"):
            schedule.validate_envelope({**ENVELOPE, "max_tree_nodes": 1_000_001})

    def test_inspect_cost_independent_vector(self):
        self.assertEqual(schedule.operation_cost(ENVELOPE, "inspect", payload_bytes=2**24), independent(["inspect"]))

    def test_load_cost_independent_vector(self):
        self.assertEqual(schedule.operation_cost(ENVELOPE, "load", payload_bytes=2**24), independent(["load"]))

    def test_save_cost_independent_vector(self):
        self.assertEqual(schedule.operation_cost(ENVELOPE, "save"), independent(["save"]))

    def test_reads_use_exact_shorter_independent_size(self):
        for operation in ("inspect", "load"):
            expected = independent([operation]); expected["snapshot_hash_bytes"] = 17
            self.assertEqual(schedule.operation_cost(ENVELOPE, operation, payload_bytes=17), expected)

    def test_read_size_missing_bool_zero_oversized_refuses(self):
        for operation in ("inspect", "load"):
            for value in (None, True, 0, -1, 2**24+1, 2**63, "17"):
                with self.subTest(operation=operation, value=value), self.assertRaises(ValueError):
                    schedule.operation_cost(ENVELOPE, operation, payload_bytes=value)

    def test_save_cannot_use_post_serialization_size_or_unknown_phase(self):
        with self.assertRaises(ValueError): schedule.operation_cost(ENVELOPE, "save", payload_bytes=17)
        for operation in ("restore", "old-load", True, None):
            with self.assertRaises(ValueError): schedule.operation_cost(ENVELOPE, operation)

    def test_primary_two_update_schedule_has_one_inspect_three_loads_four_saves(self):
        value = requirements(); self.assertEqual(value["schedule"]["commit_cursors"], [0, 2])
        self.assertEqual(value["fresh_limits"], independent(["save", "save", "load"]))
        self.assertEqual(value["required_limits"], independent(["save"]*4+["load"]*3+["inspect"]))
        self.assertEqual(value["required_limits"]["snapshot_hash_bytes"], 134217728)
        self.assertFalse(value["production_ready"]); self.assertFalse(value["launch_authorized"])

    def test_final_periodic_save_is_deduplicated(self):
        value = requirements(checkpoint_every=1)
        self.assertEqual(value["schedule"]["commit_cursors"], [0, 1, 2])
        self.assertEqual(value["required_limits"], independent(["save"]*6+["load"]*3+["inspect"]))

    def test_final_resume_still_inspects_loads_and_recommits(self):
        final = requirements()["resumed_at_boundaries"][-1]
        self.assertEqual(final["durable_completed"], 2)
        self.assertEqual(final["limits"], independent(["inspect", "load", "save", "load"]))

    def test_one_attempt_adds_no_resume_inspection(self):
        value = requirements(complete_attempt_capacity=1)
        self.assertEqual(value["required_limits"], value["fresh_limits"])
        self.assertEqual(value["required_limits"]["snapshot_inspect_operations"], 0)

    def test_diagnostic_load_count_is_explicit(self):
        first = requirements(); second = requirements(diagnostic_loads_per_attempt=2)
        difference = {key: second["required_limits"][key]-first["required_limits"][key] for key in schedule.KEYS}
        self.assertEqual(difference, independent(["load", "load"]))

    def test_all_schedule_scalar_boundaries_and_bool_aliases(self):
        cases = {"updates": (0, 1001), "checkpoint_every": (0, 1001),
                 "diagnostic_loads_per_attempt": (0, 17), "complete_attempt_capacity": (0, 5)}
        for key, values in cases.items():
            for bad in (*values, True, 1.0, "1", 2**63):
                with self.subTest(key=key, bad=bad), self.assertRaises(ValueError): requirements(**{key: bad})
        self.assertEqual(len(requirements(updates=1000, checkpoint_every=1)["schedule"]["commit_cursors"]), 1001)

    def test_exhaustive_small_schedules_match_literal_event_enumeration(self):
        for updates in range(1, 9):
            for cadence in range(1, 11):
                cursors = [0] + [n for n in range(1, updates+1) if n % cadence == 0 or n == updates]
                for diagnostics in range(1, 4):
                    fresh = independent(["save"]*len(cursors)+["load"]*diagnostics)
                    resumed = [independent(["inspect", "load"]+["save"]*sum(n >= k for n in cursors)
                                           +["load"]*diagnostics) for k in cursors]
                    upper = {key: max(row[key] for row in resumed) for key in schedule.KEYS}
                    for attempts in range(1, 5):
                        value = requirements(updates=updates, checkpoint_every=cadence,
                            diagnostic_loads_per_attempt=diagnostics, complete_attempt_capacity=attempts)
                        expected = {key: fresh[key]+(attempts-1)*upper[key] for key in schedule.KEYS}
                        self.assertEqual(value["required_limits"], expected)

    def test_overflow_refuses_not_python_unbounded_sum(self):
        envelope = dict(max_payload_bytes=2**63-1, max_tree_nodes=1, max_tensor_elements=0,
                        max_tensor_bytes=0, max_primitive_bytes=0)
        with self.assertRaises(ValueError): schedule.build_requirements(envelope, updates=1,
            checkpoint_every=1, diagnostic_loads_per_attempt=1, complete_attempt_capacity=1)

    def test_rehashed_missing_read_or_save_is_not_valid(self):
        for key in ("snapshot_load_operations", "snapshot_save_operations", "snapshot_hash_bytes"):
            value = requirements(); value["required_limits"][key] -= 1; rehash(value)
            with self.subTest(key=key), self.assertRaises(ValueError): schedule.validate_requirements(value)

    def test_digest_old_schema_authority_or_unknown_fields_refuse(self):
        for key, bad in (("requirements_sha256", "0"*64), ("schema", "dongxi-dpo-work-requirements-v2"),
                         ("production_ready", True), ("launch_authorized", True), ("scope", "production")):
            value = requirements(); value[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError): schedule.validate_requirements(value)
        value = requirements(); value["new_override"] = 1
        with self.assertRaises(ValueError): schedule.validate_requirements(value)

    def test_bool_alias_or_oversized_nested_record_refuses(self):
        value = requirements(); value["schedule"]["commit_cursors"][0] = False; rehash(value)
        with self.assertRaises(ValueError): schedule.validate_requirements(value)
        value = requirements(); value["resumed_at_boundaries"][0]["durable_completed"] = False; rehash(value)
        with self.assertRaises(ValueError): schedule.validate_requirements(value)
        value = requirements(); value["operations"]["inspect"]["snapshot_inspect_operations"] = True; rehash(value)
        with self.assertRaises(ValueError): schedule.validate_requirements(value)
        value = requirements(); value["resumed_at_boundaries"] *= 10000
        with self.assertRaises(ValueError): schedule.validate_requirements(value)

    def test_all_nine_cap_dimensions_refuse_short_zero_missing_extra_or_bool(self):
        value = requirements(); exact = value["required_limits"]
        self.assertEqual(schedule.assert_limits(value, exact), exact)
        for key in schedule.KEYS:
            for bad in (exact[key]-1, 0, True):
                with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                    schedule.assert_limits(value, {**exact, key: bad})
            missing = deepcopy(exact); del missing[key]
            with self.assertRaises(ValueError): schedule.assert_limits(value, missing)
        with self.assertRaises(ValueError): schedule.assert_limits(value, {**exact, "train_updates": 2})

    def test_enough_allowance_does_not_mutate_caps_or_requirements(self):
        value = requirements(); before = deepcopy(value)
        caps = {key: amount+1 for key, amount in value["required_limits"].items()}; previous = deepcopy(caps)
        accepted = schedule.assert_limits(value, caps); accepted["snapshot_hash_bytes"] = 0
        self.assertEqual(caps, previous); self.assertEqual(value, before)

    def test_module_import_does_not_import_torch_transformers_or_model_code(self):
        code = "import sys; from dongxi_llms import snapshot_io_schedule; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules; assert 'dongxi_llms.dpo_lab' not in sys.modules"
        child = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=os.environ.copy(),
                               capture_output=True, text=True, timeout=20)
        self.assertEqual(child.returncode, 0, child.stdout+child.stderr)


def collect(directory):
    """Exclusive raw arithmetic evidence; never claim actual reader admission."""
    directory = Path(directory); directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    def write(name, value):
        with (directory/name).open("xb") as handle:
            handle.write((json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n").encode())
    def hashes(names): return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
    before = hashes(SOURCES); old_before = hashes(PRESERVED)
    write("source-before.json", before); write("historical-before.json", old_before)
    command = [sys.executable, "-m", "unittest", "tests.test_snapshot_io_schedule", "-v"]
    started = time.monotonic(); child = subprocess.run(command, cwd=ROOT, env=os.environ.copy(),
        capture_output=True, text=True, timeout=60); duration = time.monotonic()-started
    write("tests.json", dict(command=command, exit_code=child.returncode,
        stdout=child.stdout, stderr=child.stderr, duration_seconds=duration))
    primary = requirements(); every_update = requirements(checkpoint_every=1)
    write("requirements.json", primary); write("cadence-one-requirements.json", every_update)
    after = hashes(SOURCES); old_after = hashes(PRESERVED)
    write("source-after.json", after); write("historical-after.json", old_after)
    record = dict(schema="dongxi-snapshot-io-schedule-verification-v1", actual_exit_code=child.returncode,
        tests=26, exhaustive_schedule_cases=960, envelope=ENVELOPE, primary_required=primary["required_limits"],
        cadence_one_required=every_update["required_limits"], source_before=before, source_after=after,
        source_stable=before==after, historical_before=old_before, historical_after=old_after,
        historical_stable=old_before==old_after, duration_seconds=duration,
        scope="pure arithmetic/refusal controls; no shared reader or model operation executed",
        production_ready=False, launch_authorized=False)
    write("verification.json", record)
    if child.returncode or before != after or old_before != old_after: raise RuntimeError("Verification failed; raw retained")
    return record


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--collect":
        print(json.dumps(collect(sys.argv[2]), sort_keys=True))
    else:
        unittest.main()
