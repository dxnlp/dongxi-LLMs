"""Independent adversarial checks of the shared trusted-local snapshot format.

No runner or model fit is used. Crafted files are confined to TemporaryDirectory;
all assertions use the public save/load/inspect interface except OS fault spies.
"""
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from dongxi_llms import training_snapshot as snapshots


def independent_contract_sha(contract):
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class TrainingSnapshotFailurePaths(unittest.TestCase):
    max_bytes = 100_000

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="snapshot-adversarial-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.contract = {
            "runner": "authored-envelope-test",
            "objective": {"name": "not-a-fit", "reduction": "valid-target-mean"},
            "interface": {"token_map": "a" * 64, "stop_ids": [2], "template": "b" * 64},
            "source_sha256": "c" * 64,
            "lock_sha256": "d" * 64,
            "data_sha256": "e" * 64,
            "parent_sha256": "f" * 64,
            "schedule_horizon": 4,
        }
        self.state = {
            "policy": {"weight": torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16)},
            "optimizer": {0: {"step": torch.tensor(2), "moment": torch.tensor([0.1, 0.2])}},
            "rng": torch.arange(8, dtype=torch.uint8),
            "cursor": 2,
            "history": [{"update": 1, "loss": 0.5}, {"update": 2, "loss": 0.4}],
            "optional": None,
            "tokens": (1, 2),
        }
        self.previous = self.directory / "previous.pt"
        self.previous_header = self.save(self.previous)
        self.previous_bytes = self.previous.read_bytes()
        self.previous_marker = self.marker(self.previous).read_bytes()

    @staticmethod
    def marker(path):
        return Path(str(path) + ".commit.json")

    def save(self, path, **changes):
        arguments = dict(contract=self.contract, state=self.state, completed_updates=2,
                         parent_invocation="original-authored-cpu-invocation",
                         phase="completed", max_bytes=self.max_bytes)
        arguments.update(changes)
        return snapshots.save_snapshot(path, **arguments)

    def expected(self, header=None, **changes):
        header = self.previous_header if header is None else header
        arguments = dict(expected_sha256=header["payload_sha256"],
                         expected_bytes=header["payload_bytes"],
                         expected_contract=self.contract, max_bytes=self.max_bytes)
        arguments.update(changes)
        return arguments

    def assert_previous_unchanged(self):
        self.assertEqual(self.previous.read_bytes(), self.previous_bytes)
        self.assertEqual(self.marker(self.previous).read_bytes(), self.previous_marker)
        self.assertEqual(snapshots.inspect_snapshot(self.previous, **self.expected()),
                         self.previous_header)

    def header_copy(self, name, header=None, raw=None):
        path = self.directory / name
        path.write_bytes(self.previous_bytes)
        if raw is None:
            raw = json.dumps(self.previous_header if header is None else header).encode()
        self.marker(path).write_bytes(raw)
        return path

    def reject_before_load(self, path, **expected_changes):
        with mock.patch.object(snapshots.torch, "load") as loader:
            with self.assertRaises((ValueError, OSError)):
                snapshots.load_snapshot(path, **self.expected(**expected_changes))
            loader.assert_not_called()
        with mock.patch.object(snapshots.torch, "load") as loader:
            with self.assertRaises((ValueError, OSError)):
                snapshots.inspect_snapshot(path, **self.expected(**expected_changes))
            loader.assert_not_called()

    def raw_payload(self, name, payload):
        """Independently serialize a deliberately invalid data-only payload."""
        path = self.directory / name
        torch.save(payload, path)
        raw = path.read_bytes()
        header = {
            "schema_version": 1,
            "payload_sha256": hashlib.sha256(raw).hexdigest(),
            "payload_bytes": len(raw),
            "contract_sha256": independent_contract_sha(self.contract),
            "phase": "completed", "completed_updates": 2,
        }
        self.marker(path).write_text(json.dumps(header), encoding="utf-8")
        return path, header

    def valid_payload(self):
        return {"schema_version": 1, "contract": deepcopy(self.contract),
                "state": deepcopy(self.state), "phase": "completed",
                "completed_updates": 2, "parent_invocation": "crafted-local-input"}

    def test_valid_roundtrip_is_data_only_with_explicit_cpu_mapping(self):
        real_load = torch.load
        with mock.patch.object(snapshots.torch, "load", wraps=real_load) as loader:
            payload = snapshots.load_snapshot(self.previous, **self.expected())
        self.assertEqual(loader.call_count, 1)
        self.assertEqual(loader.call_args.kwargs, {"map_location": "cpu", "weights_only": True})
        self.assertEqual(payload["contract"], self.contract)
        self.assertEqual(payload["parent_invocation"], "original-authored-cpu-invocation")
        self.assertEqual(payload["completed_updates"], 2)
        self.assertEqual(payload["phase"], "completed")
        self.assertTrue(torch.equal(payload["state"]["policy"]["weight"], self.state["policy"]["weight"]))
        self.assertEqual(payload["state"]["policy"]["weight"].dtype, torch.bfloat16)
        self.assertEqual(payload["state"]["tokens"], (1, 2))
        self.assertEqual(self.previous_header["contract_sha256"], independent_contract_sha(self.contract))

    def test_inspection_does_not_deserialize(self):
        with mock.patch.object(snapshots.torch, "load", side_effect=AssertionError("No tensor load")) as loader:
            actual = snapshots.inspect_snapshot(self.previous, **self.expected())
        self.assertEqual(actual, self.previous_header)
        loader.assert_not_called()

    def test_loader_consumes_verified_buffer_not_reopened_path(self):
        path = self.directory / "path-replacement.pt"
        header = self.save(path)
        real_load = torch.load

        def replace_path_after_verification(handle, **kwargs):
            self.assertIsInstance(handle, io.BytesIO)
            path.write_bytes(b"replaced-after-verification")
            return real_load(handle, **kwargs)

        with mock.patch.object(snapshots.torch, "load", side_effect=replace_path_after_verification):
            payload = snapshots.load_snapshot(path, **self.expected(header))
        self.assertEqual(payload["state"]["cursor"], 2)
        with self.assertRaises(ValueError):
            snapshots.inspect_snapshot(path, **self.expected(header))
        self.assert_previous_unchanged()

    def test_pending_phase_is_generic_and_callback_owns_pending_semantics(self):
        path = self.directory / "pending.pt"
        state = deepcopy(self.state)
        state["pending"] = {"version": 2, "post_collection_cursor": 3, "raw_ids": [[1, 2]]}
        header = self.save(path, phase="pending", state=state)
        seen = []
        payload = snapshots.load_snapshot(path, **self.expected(header),
                                           validate_payload=lambda item: seen.append(item))
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0], payload)
        self.assertEqual(payload["phase"], "pending")
        self.assertEqual(payload["state"]["pending"]["version"], 2)

    def test_wrong_independent_digest_and_size_reject_before_load(self):
        for changes in ({"expected_sha256": "0" * 64},
                        {"expected_bytes": self.previous_header["payload_bytes"] + 1},
                        {"expected_bytes": self.previous_header["payload_bytes"] - 1}):
            with self.subTest(changes=changes):
                self.reject_before_load(self.previous, **changes)

    def test_each_scientific_identity_change_rejects_before_load(self):
        for field in ("source_sha256", "lock_sha256", "data_sha256", "parent_sha256", "schedule_horizon"):
            changed = deepcopy(self.contract)
            changed[field] = 8 if field == "schedule_horizon" else "0" * 64
            with self.subTest(field=field):
                self.reject_before_load(self.previous, expected_contract=changed)
        for field, value in (("token_map", "0" * 64), ("template", "0" * 64), ("stop_ids", [3])):
            changed = deepcopy(self.contract)
            changed["interface"][field] = value
            with self.subTest(interface=field):
                self.reject_before_load(self.previous, expected_contract=changed)

    def test_moved_snapshot_and_new_parent_do_not_change_science(self):
        path = self.header_copy("relocated.pt")
        payload = snapshots.load_snapshot(path, **self.expected())
        self.assertEqual(payload["contract"], self.contract)
        new_path = self.directory / "new-invocation.pt"
        header = self.save(new_path, parent_invocation="new-authored-invocation")
        self.assertEqual(header["contract_sha256"], self.previous_header["contract_sha256"])
        payload = snapshots.load_snapshot(new_path, **self.expected(header))
        self.assertEqual(payload["parent_invocation"], "new-authored-invocation")

    def test_independent_expectations_and_bounds_require_strict_types(self):
        invalid = ({"max_bytes": 0}, {"max_bytes": True}, {"max_bytes": 1.0},
                   {"max_bytes": 2**63}, {"max_bytes": self.previous_header["payload_bytes"] - 1},
                   {"expected_bytes": True}, {"expected_bytes": 0}, {"expected_bytes": -1},
                   {"expected_sha256": "A" * 64}, {"expected_sha256": "0" * 63},
                   {"expected_contract": {}}, {"expected_contract": {"tensor": torch.ones(1)}})
        for changes in invalid:
            with self.subTest(changes=changes):
                self.reject_before_load(self.previous, **changes)

    def test_payload_byte_mutation_and_truncation_reject_before_load(self):
        for label, raw in (("mutated", bytes([self.previous_bytes[0] ^ 1]) + self.previous_bytes[1:]),
                           ("truncated", self.previous_bytes[:-1]),
                           ("extended", self.previous_bytes + b"x")):
            path = self.header_copy(label + ".pt")
            path.write_bytes(raw)
            with self.subTest(label=label):
                self.reject_before_load(path)

    def test_self_consistent_forged_header_cannot_replace_external_receipt(self):
        path = self.header_copy("forged-receipt.pt")
        path.write_bytes(b"authored-not-a-checkpoint")
        header = deepcopy(self.previous_header)
        header.update(payload_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      payload_bytes=path.stat().st_size)
        self.marker(path).write_text(json.dumps(header), encoding="utf-8")
        self.reject_before_load(path)

    def test_missing_marker_and_missing_payload_are_not_committed_snapshots(self):
        data_only = self.directory / "data-only.pt"
        data_only.write_bytes(self.previous_bytes)
        marker_only = self.directory / "marker-only.pt"
        self.marker(marker_only).write_bytes(self.previous_marker)
        for path in (data_only, marker_only):
            with self.subTest(path=path.name):
                self.reject_before_load(path)

    def test_symlink_payload_and_marker_are_rejected_before_load(self):
        linked_data = self.directory / "linked-data.pt"
        linked_data.symlink_to(self.previous)
        self.marker(linked_data).write_bytes(self.previous_marker)
        linked_marker = self.directory / "linked-marker.pt"
        linked_marker.write_bytes(self.previous_bytes)
        self.marker(linked_marker).symlink_to(self.marker(self.previous))
        for path in (linked_data, linked_marker):
            with self.subTest(path=path.name):
                self.reject_before_load(path)

    def test_header_unknown_missing_and_non_object_fields_reject_before_load(self):
        extra = dict(self.previous_header, unrecognized="not-accepted")
        missing = deepcopy(self.previous_header)
        missing.pop("phase")
        for index, header in enumerate((extra, missing, [], None, "header")):
            path = self.header_copy(f"header-fields-{index}.pt", raw=json.dumps(header).encode())
            with self.subTest(index=index):
                self.reject_before_load(path)

    def test_header_duplicate_keys_malformed_utf8_and_nonfinite_reject_before_load(self):
        valid = self.previous_marker.strip()
        duplicate = valid[:-1] + b',"phase":"completed"}'
        raws = (duplicate, b"{", b"\xff\xfe", b'{"schema_version":NaN}')
        for index, raw in enumerate(raws):
            path = self.header_copy(f"header-syntax-{index}.pt", raw=raw)
            with self.subTest(index=index):
                self.reject_before_load(path)

    def test_deep_and_oversized_headers_reject_before_load(self):
        raws = (b'{"reason":' + b"[" * 10_000 + b"0" + b"]" * 10_000 + b"}",
                b" " * (65_536 + 1))
        for index, raw in enumerate(raws):
            path = self.header_copy(f"header-bound-{index}.pt", raw=raw)
            with self.subTest(index=index):
                self.reject_before_load(path)

    def test_header_schema_phase_counter_digest_size_reject_before_load(self):
        mutations = (("schema_version", True), ("schema_version", 2),
                     ("phase", "half-updated"), ("phase", None),
                     ("completed_updates", True), ("completed_updates", -1),
                     ("completed_updates", 2**63), ("payload_bytes", True),
                     ("payload_bytes", 0), ("payload_sha256", "A" * 64),
                     ("contract_sha256", "0" * 64))
        for index, (key, value) in enumerate(mutations):
            header = deepcopy(self.previous_header)
            header[key] = value
            path = self.header_copy(f"header-types-{index}.pt", header=header)
            with self.subTest(key=key, value=value):
                self.reject_before_load(path)

    def test_payload_unknown_schema_fields_and_header_progress_disagree(self):
        mutations = (("unrecognized", "retained-but-rejected"), ("schema_version", True),
                     ("schema_version", 2), ("phase", "pending"), ("phase", "half-updated"),
                     ("completed_updates", 3), ("completed_updates", True),
                     ("completed_updates", -1))
        for index, (key, value) in enumerate(mutations):
            payload = self.valid_payload()
            payload[key] = value
            path, header = self.raw_payload(f"payload-schema-{index}.pt", payload)
            callback = mock.Mock()
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                snapshots.load_snapshot(path, **self.expected(header), validate_payload=callback)
            callback.assert_not_called()

    def test_payload_missing_fields_state_parent_and_contract_reject(self):
        payloads = []
        missing = self.valid_payload()
        missing.pop("state")
        payloads.append(missing)
        for key, value in (("state", []), ("parent_invocation", ""),
                           ("parent_invocation", 1), ("contract", {}),
                           ("contract", {"other": "different"})):
            payload = self.valid_payload()
            payload[key] = value
            payloads.append(payload)
        for index, payload in enumerate(payloads):
            path, header = self.raw_payload(f"payload-values-{index}.pt", payload)
            callback = mock.Mock()
            with self.subTest(index=index), self.assertRaises(ValueError):
                snapshots.load_snapshot(path, **self.expected(header), validate_payload=callback)
            callback.assert_not_called()

    def test_data_only_payload_gate_rejects_nonfinite_and_unsupported_tree(self):
        bad_states = ({"value": float("nan")}, {"value": torch.tensor(float("inf"))},
                      {"value": {1, 2}}, {"value": torch.ones(2, dtype=torch.complex64)},
                      {False: "bool-key-is-not-an-integer-ID"})
        for index, state in enumerate(bad_states):
            payload = self.valid_payload()
            payload["state"] = state
            path, header = self.raw_payload(f"payload-tree-{index}.pt", payload)
            with self.subTest(index=index), self.assertRaises(ValueError):
                snapshots.load_snapshot(path, **self.expected(header))

    def test_callback_rejects_cursor_after_generic_validation(self):
        def reject_cursor(payload):
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["state"]["cursor"], 2)
            raise ValueError("Declared runner cursor invariant failure")
        with self.assertRaisesRegex(ValueError, "cursor invariant"):
            snapshots.load_snapshot(self.previous, **self.expected(), validate_payload=reject_cursor)
        self.assert_previous_unchanged()

    def test_save_invalid_envelope_is_rejected_before_new_file(self):
        changes = ({"phase": "half-updated"}, {"phase": None},
                   {"completed_updates": True}, {"completed_updates": -1},
                   {"completed_updates": 2**63}, {"max_bytes": True}, {"max_bytes": 0},
                   {"parent_invocation": ""}, {"parent_invocation": 1},
                   {"contract": {}}, {"contract": {"tensor": torch.ones(1)}}, {"state": []})
        for index, invalid in enumerate(changes):
            path = self.directory / f"invalid-save-{index}.pt"
            with self.subTest(changes=invalid), self.assertRaises(ValueError):
                self.save(path, **invalid)
            self.assertFalse(path.exists())
            self.assertFalse(self.marker(path).exists())
        self.assert_previous_unchanged()

    def test_save_unsupported_objects_nonfinite_cycles_and_layouts_reject(self):
        cyclic = []
        cyclic.append(cyclic)
        deep = None
        for _ in range(66):
            deep = [deep]
        sparse = torch.sparse_coo_tensor(torch.tensor([[0]]), torch.tensor([1.0]), (2,))
        quantized = torch.quantize_per_tensor(torch.tensor([1.0]), 0.1, 0, torch.qint8)
        states = ({"value": object()}, {"value": Path("not-a-primitive")},
                  {"value": float("inf")}, {"value": torch.tensor(float("nan"))},
                  {"value": cyclic}, {"value": deep}, {"value": sparse},
                  {"value": quantized}, {"value": torch.tensor([1j])})
        for index, state in enumerate(states):
            path = self.directory / f"invalid-tree-{index}.pt"
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.save(path, state=state)
            self.assertFalse(path.exists())
        self.assert_previous_unchanged()

    def test_saved_state_and_contract_do_not_alias_later_caller_mutations(self):
        path = self.directory / "frozen.pt"
        contract = deepcopy(self.contract)
        state = deepcopy(self.state)
        header = self.save(path, contract=contract, state=state)
        contract["interface"]["stop_ids"].append(99)
        state["policy"]["weight"].fill_(77)
        state["history"][0]["loss"] = 88.0
        payload = snapshots.load_snapshot(path, **self.expected(header))
        self.assertEqual(payload["contract"], self.contract)
        self.assertTrue(torch.equal(payload["state"]["policy"]["weight"], self.state["policy"]["weight"]))
        self.assertEqual(payload["state"]["history"][0]["loss"], 0.5)

    def test_existing_data_or_marker_cannot_be_overwritten(self):
        with self.assertRaises(FileExistsError):
            self.save(self.previous)
        for label, data, marker in (("orphan-data", True, False),
                                    ("orphan-marker", False, True)):
            path = self.directory / (label + ".pt")
            if data:
                path.write_bytes(b"retained-orphan-data")
            if marker:
                self.marker(path).write_bytes(b"retained-orphan-marker")
            before = {item.name: item.read_bytes() for item in self.directory.iterdir()}
            with self.subTest(label=label), self.assertRaises(FileExistsError):
                self.save(path)
            self.assertEqual({item.name: item.read_bytes() for item in self.directory.iterdir()}, before)
        self.assert_previous_unchanged()

    def test_failed_writer_retains_partial_bytes_and_previous_commit(self):
        path = self.directory / "writer-failure.pt"

        def fail_write(payload, handle):
            handle.write(b"authored-partial-write")
            raise OSError("injected writer failure")

        with mock.patch.object(snapshots.torch, "save", side_effect=fail_write):
            with self.assertRaisesRegex(OSError, "writer failure"):
                self.save(path)
        self.assertEqual(path.read_bytes(), b"authored-partial-write")
        self.assertFalse(self.marker(path).exists())
        self.assert_previous_unchanged()

    def test_guard_failure_before_serialization_creates_no_file(self):
        path = self.directory / "guard-before.pt"
        with self.assertRaisesRegex(RuntimeError, "guard stopped"):
            self.save(path, guard=mock.Mock(side_effect=RuntimeError("guard stopped")))
        self.assertFalse(path.exists())
        self.assertFalse(self.marker(path).exists())
        self.assert_previous_unchanged()

    def test_guard_failure_after_header_fsync_retains_unpublished_evidence(self):
        path = self.directory / "guard-header.pt"

        def guard():
            if list(self.directory.glob(path.name + ".commit-*")):
                raise RuntimeError("guard stopped after temporary header")

        with self.assertRaisesRegex(RuntimeError, "temporary header"):
            self.save(path, guard=guard)
        self.assertGreater(path.stat().st_size, 0)
        temporaries = list(self.directory.glob(path.name + ".commit-*"))
        self.assertEqual(len(temporaries), 1)
        self.assertEqual(json.loads(temporaries[0].read_text())["completed_updates"], 2)
        self.assertFalse(self.marker(path).exists())
        self.assert_previous_unchanged()

    def test_data_and_header_fsync_failures_retain_evidence(self):
        real_fsync = os.fsync
        for fail_at in (1, 2):
            path = self.directory / f"fsync-failure-{fail_at}.pt"
            calls = []

            def fail_selected(descriptor):
                calls.append(descriptor)
                if len(calls) == fail_at:
                    raise OSError(f"injected fsync {fail_at}")
                return real_fsync(descriptor)

            with mock.patch.object(snapshots.os, "fsync", side_effect=fail_selected):
                with self.subTest(fail_at=fail_at), self.assertRaisesRegex(OSError, "injected fsync"):
                    self.save(path)
            self.assertGreater(path.stat().st_size, 0)
            self.assertFalse(self.marker(path).exists())
            self.assertEqual(len(list(self.directory.glob(path.name + ".commit-*"))), fail_at - 1)
            self.assert_previous_unchanged()

    def test_failed_atomic_header_publication_retains_full_data_and_header(self):
        path = self.directory / "publish-failure.pt"
        with mock.patch.object(snapshots.os, "link", side_effect=OSError("injected header publication")):
            with self.assertRaisesRegex(OSError, "header publication"):
                self.save(path)
        self.assertGreater(path.stat().st_size, 0)
        self.assertEqual(len(list(self.directory.glob(path.name + ".commit-*"))), 1)
        self.assertFalse(self.marker(path).exists())
        self.assert_previous_unchanged()

    def test_directory_fsync_failure_may_leave_visible_commit_but_returns_failure(self):
        real_fsync = os.fsync
        for fail_at in (3, 4):
            path = self.directory / f"directory-fsync-{fail_at}.pt"
            calls = []

            def fail_selected(descriptor):
                calls.append(descriptor)
                if len(calls) == fail_at:
                    raise OSError(f"injected directory fsync {fail_at}")
                return real_fsync(descriptor)

            with mock.patch.object(snapshots.os, "fsync", side_effect=fail_selected):
                with self.subTest(fail_at=fail_at), self.assertRaisesRegex(OSError, "directory fsync"):
                    self.save(path)
            self.assertTrue(self.marker(path).exists())
            header = json.loads(self.marker(path).read_text())
            self.assertEqual(snapshots.inspect_snapshot(path, **self.expected(header)), header)
            self.assertEqual(len(list(self.directory.glob(path.name + ".commit-*"))), 1 if fail_at == 3 else 0)
            self.assert_previous_unchanged()

    def test_serialized_size_limit_is_hard_and_failure_does_not_commit(self):
        path = self.directory / "serialized-bound.pt"
        with self.assertRaises((ValueError, RuntimeError)):
            self.save(path, state={"text": "x" * 10_000}, max_bytes=100)
        self.assertTrue(path.exists())
        self.assertLessEqual(path.stat().st_size, 100)
        self.assertFalse(self.marker(path).exists())
        self.assert_previous_unchanged()


if __name__ == "__main__":
    unittest.main()
