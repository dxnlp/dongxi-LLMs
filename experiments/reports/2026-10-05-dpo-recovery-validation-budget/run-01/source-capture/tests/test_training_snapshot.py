"""Original tensor-only checks; actual runner semantics have separate tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from dongxi_llms.training_snapshot import inspect_snapshot, load_snapshot, save_snapshot

LIMIT = 16 * 1024 * 1024
CONTRACT = {"runner": "original-fixture", "objective": {"lr": .01, "horizon": 4},
            "source": "s" * 64, "interface": {"ids": [0, 1, 2], "stop": 0}}


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="dongxi-snapshot-test-")
        self.tmp_path = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def write(self, state=None, name="step.pt", **kwargs):
        path = self.tmp_path / name
        state = {"policy": {"x": torch.arange(6.).reshape(2, 3)},
                 "optimizer": {"state": {0: {"step": torch.tensor(2.)}}},
                 "rng": torch.get_rng_state(), "metrics": [{"update": 2, "loss": .5}]} if state is None else state
        receipt = save_snapshot(path, contract=CONTRACT, state=state,
                                completed_updates=2, parent_invocation="fixture-A",
                                max_bytes=LIMIT, **kwargs)
        return path, receipt, state

    def load(self, path, receipt, **kwargs):
        return load_snapshot(path, expected_sha256=receipt["payload_sha256"],
            expected_bytes=receipt["payload_bytes"], expected_contract=CONTRACT,
            max_bytes=LIMIT, **kwargs)

    def test_dense_exact_roundtrip(self):
        tensors = [torch.tensor(1.), torch.empty(0), torch.arange(12.).reshape(3, 4).T,
                   torch.tensor([1., -0., .5], dtype=torch.bfloat16),
                   torch.tensor([True, False]), torch.arange(3, dtype=torch.int64)]
        for index, tensor in enumerate(tensors):
            with self.subTest(dtype=tensor.dtype, shape=tensor.shape):
                path, receipt, state = self.write({"tensor": tensor}, name=f"dense-{index}.pt")
                before = tensor.clone()
                result = self.load(path, receipt)
                self.assertEqual(result["phase"], "completed")
                self.assertEqual(result["completed_updates"], 2)
                self.assertEqual(result["parent_invocation"], "fixture-A")
                self.assertEqual(result["state"]["tensor"].dtype, tensor.dtype)
                self.assertTrue(torch.equal(result["state"]["tensor"], before))
                self.assertTrue(torch.equal(state["tensor"], before))

    def test_adam_rng_and_tuple_payload(self):
        path, receipt, state = self.write()
        result = self.load(path, receipt)
        self.assertTrue(torch.equal(result["state"]["rng"], state["rng"]))
        self.assertTrue(torch.equal(result["state"]["optimizer"]["state"][0]["step"], torch.tensor(2.)))
        self.assertEqual(result["state"]["metrics"], state["metrics"])

    def test_pending_phase_is_runner_owned(self):
        path, receipt, _ = self.write({"pending": {"tokens": torch.tensor([[1, 2]])}}, phase="pending")
        observed = []
        result = self.load(path, receipt, validate_payload=lambda value: observed.append(value["phase"]))
        self.assertEqual(observed, ["pending"])
        self.assertEqual(result["phase"], "pending")

    def test_inspection_has_no_deserialization(self):
        path, receipt, _ = self.write()
        with patch("dongxi_llms.training_snapshot.torch.load", side_effect=AssertionError("no load")):
            self.assertEqual(inspect_snapshot(path, expected_sha256=receipt["payload_sha256"],
                expected_bytes=receipt["payload_bytes"], expected_contract=CONTRACT,
                max_bytes=LIMIT), receipt)

    def test_contract_changes_reject_before_loading(self):
        path, receipt, _ = self.write()
        changed = copy.deepcopy(CONTRACT)
        changed["interface"]["ids"] = [1, 0, 2]
        with patch("dongxi_llms.training_snapshot.torch.load", side_effect=AssertionError("no load")):
            with self.assertRaisesRegex(ValueError, "contract"):
                load_snapshot(path, expected_sha256=receipt["payload_sha256"],
                    expected_bytes=receipt["payload_bytes"], expected_contract=changed, max_bytes=LIMIT)

    def test_byte_tamper_rejects_before_loading(self):
        path, receipt, _ = self.write()
        data = bytearray(path.read_bytes())
        data[len(data) // 2] ^= 1
        path.write_bytes(data)  # Test-only fault injection.
        with patch("dongxi_llms.training_snapshot.torch.load", side_effect=AssertionError("no load")):
            with self.assertRaisesRegex(ValueError, "byte identity"):
                self.load(path, receipt)

    def test_exclusive_target_preserves_bytes(self):
        path, receipt, _ = self.write()
        before = path.read_bytes(), Path(str(path) + ".commit.json").read_bytes()
        with self.assertRaises(FileExistsError):
            save_snapshot(path, contract=CONTRACT, state={}, completed_updates=3,
                          parent_invocation="fixture-B", max_bytes=LIMIT)
        self.assertEqual(before, (path.read_bytes(), Path(str(path) + ".commit.json").read_bytes()))
        self.assertEqual(self.load(path, receipt)["completed_updates"], 2)

    def test_guard_observes_work(self):
        calls = []
        guard = lambda: calls.append(1)
        path, receipt, _ = self.write(guard=guard)
        before = len(calls)
        self.load(path, receipt, guard=guard)
        self.assertGreater(before, 0)
        self.assertGreater(len(calls), before)
        self.assertEqual(json.loads(Path(str(path) + ".commit.json").read_text()), receipt)


if __name__ == "__main__":
    unittest.main()
