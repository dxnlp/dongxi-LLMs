"""Actual shared calls; failures permanently retain their whole reservation."""
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch

from dongxi_llms.run_identity import canonical_hash
from dongxi_llms.snapshot_io_budget import (IO_KEYS, SnapshotIOBudget, io_budget_contract,
    io_ledger_contract_sha256, read_work_receipt, read_bounded_json, validate_work_receipt)
from dongxi_llms.training_snapshot import inspect_snapshot, load_snapshot, save_snapshot
from dongxi_llms.work_budget import WorkLedger, WorkBudgetExceeded

SCIENCE = {'runner': 'original-shared-cpu', 'lr': .01, 'horizon': 3, 'source': 'a' * 64}
MAX_BYTES = 1024 * 1024
ENVELOPE = dict(max_payload_bytes=MAX_BYTES, max_tree_nodes=4096,
                max_tensor_elements=100000, max_tensor_bytes=500000, max_primitive_bytes=65536)


class SnapshotIOTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='dongxi-io-admission-')
        self.root = Path(self.directory.name); os.chmod(self.root, 0o700)
        self.ledgers = []

    def tearDown(self):
        for ledger in self.ledgers: ledger.close()
        self.directory.cleanup()

    def budget(self, *, limits=None, envelope=None, name='io.jsonl'):
        caps = {key: 100000000 for key in IO_KEYS}; caps.update(limits or {})
        contract = io_budget_contract(caps, envelope or ENVELOPE, 4 * 1024 * 1024)
        ledger = WorkLedger.create(self.root / name, limits=caps,
            contract_sha256=io_ledger_contract_sha256(contract, canonical_hash(SCIENCE)),
            max_bytes=contract['max_journal_bytes'], invocation_id='shared-A')
        self.ledgers.append(ledger)
        return SnapshotIOBudget(ledger, contract=contract, scientific_contract_sha256=canonical_hash(SCIENCE))

    def write(self, budget, *, state=None, name='state.pt', **kwargs):
        path = self.root / name
        state = {'weights': torch.arange(12.).reshape(3, 4), 'history': [{'update': 3, 'loss': .5}]} if state is None else state
        header = save_snapshot(path, contract=SCIENCE, state=state, completed_updates=3,
            parent_invocation='shared-A', max_bytes=MAX_BYTES, io_budget=budget,
            work_receipt_path=Path(str(path) + '.work.json'), **kwargs)
        budget.bind_receipt(read_work_receipt(str(path) + '.work.json'))
        return path, header

    def read(self, budget, path, header, *, inspect=False, **kwargs):
        function = inspect_snapshot if inspect else load_snapshot
        return function(path, expected_sha256=header['payload_sha256'], expected_bytes=header['payload_bytes'],
            expected_contract=SCIENCE, max_bytes=MAX_BYTES, io_budget=budget, **kwargs)

    def test_accounted_roundtrip_and_repeated_reads(self):
        budget = self.budget(); path, header = self.write(budget)
        self.assertEqual(header['schema_version'], 2)
        with patch('dongxi_llms.training_snapshot.torch.load', side_effect=AssertionError('inspect must not load')):
            self.assertEqual(self.read(budget, path, header, inspect=True), header)
        result = self.read(budget, path, header)
        self.read(budget, path, header)
        self.assertTrue(torch.equal(result['state']['weights'], torch.arange(12.).reshape(3, 4)))
        snapshot = budget.ledger.snapshot()
        self.assertEqual(snapshot['completed']['snapshot_save_operations'], 1)
        self.assertEqual(snapshot['completed']['snapshot_inspect_operations'], 1)
        self.assertEqual(snapshot['completed']['snapshot_load_operations'], 2)
        self.assertEqual(snapshot['completed']['snapshot_hash_bytes'], 4 * header['payload_bytes'])
        self.assertEqual(snapshot['completed']['snapshot_clone_bytes'], 48)
        self.assertEqual(snapshot['completed']['snapshot_tensor_elements'], 36)
        self.assertEqual(snapshot['completed']['snapshot_serialization_bytes'], header['payload_bytes'])

    def test_exhausted_inspect_before_any_shared_metadata_or_payload_read(self):
        budget = self.budget(limits={'snapshot_inspect_operations': 0})
        path, header = self.write(budget)
        before = budget.ledger.snapshot()
        with patch('dongxi_llms.training_snapshot._open_regular', side_effect=AssertionError('no read')):
            with self.assertRaises(WorkBudgetExceeded): self.read(budget, path, header, inspect=True)
        self.assertEqual(before, budget.ledger.snapshot())

    def test_exhausted_load_before_deserialize_or_read(self):
        budget = self.budget(limits={'snapshot_load_operations': 0}); path, header = self.write(budget)
        with patch('dongxi_llms.training_snapshot._open_regular', side_effect=AssertionError('no read')):
            with patch('dongxi_llms.training_snapshot.torch.load', side_effect=AssertionError('no deserialize')):
                with self.assertRaises(WorkBudgetExceeded): self.read(budget, path, header)

    def test_save_admission_before_contract_walk_clone_or_serializer(self):
        budget = self.budget(limits={'snapshot_save_operations': 0})
        with patch('dongxi_llms.training_snapshot._safe_tree', side_effect=AssertionError('no tree')):
            with patch('dongxi_llms.training_snapshot.torch.save', side_effect=AssertionError('no serialize')):
                with self.assertRaises(WorkBudgetExceeded): self.write(budget)
        self.assertFalse((self.root / 'state.pt').exists())

    def test_changed_independent_bytes_before_read(self):
        budget = self.budget(); path, header = self.write(budget)
        changed = dict(header, payload_sha256='0' * 64)
        with patch('dongxi_llms.training_snapshot._open_regular', side_effect=AssertionError('no read')):
            with self.assertRaisesRegex(ValueError, 'independent receipt'): self.read(budget, path, changed)

    def test_no_receipt_no_read_and_unbound_journal_no_hook(self):
        budget = self.budget(); path, header = self.write(budget); budget.expected_receipt = None
        with patch('dongxi_llms.training_snapshot._open_regular', side_effect=AssertionError('no read')):
            with self.assertRaisesRegex(ValueError, 'receipt required'): self.read(budget, path, header)
        budget.ledger.close()
        ledger = WorkLedger.open(budget.ledger.path, limits=budget.contract['limits'],
            contract_sha256=budget.ledger.contract_sha256, max_bytes=budget.ledger.max_bytes, invocation_id='unbound')
        self.ledgers.append(ledger)
        with self.assertRaisesRegex(ValueError, 'independently bound'):
            SnapshotIOBudget(ledger, contract=budget.contract, scientific_contract_sha256=canonical_hash(SCIENCE))

    def test_later_failed_charges_survive_reopen_from_presave_prefix(self):
        budget = self.budget(); path, header = self.write(budget); receipt = budget.last_receipt
        with patch('dongxi_llms.training_snapshot.torch.load', side_effect=RuntimeError('injected restricted-load failure')):
            with self.assertRaises(RuntimeError): self.read(budget, path, header)
        before = budget.ledger.snapshot(); budget.ledger.close()
        ledger = WorkLedger.open(budget.ledger.path, limits=budget.contract['limits'],
            contract_sha256=budget.ledger.contract_sha256, max_bytes=budget.ledger.max_bytes, invocation_id='shared-B',
            expected_snapshot=receipt['io_prefix'])
        self.ledgers.append(ledger)
        resumed = SnapshotIOBudget(ledger, contract=budget.contract, scientific_contract_sha256=canonical_hash(SCIENCE), expected_receipt=receipt)
        self.assertEqual(before, ledger.snapshot())
        self.assertEqual(len(before['failed_tickets']), 1)
        self.read(resumed, path, header)
        self.assertEqual(ledger.snapshot()['reserved']['snapshot_load_operations'], 2)

    def test_copied_journal_receipt_cannot_refill(self):
        budget = self.budget(); self.write(budget); receipt = budget.last_receipt
        destination = self.root / 'copied.jsonl'; shutil.copy2(budget.ledger.path, destination)
        with self.assertRaisesRegex(ValueError, 'Copied/replaced'):
            WorkLedger.open(destination, limits=budget.contract['limits'], contract_sha256=budget.ledger.contract_sha256,
                max_bytes=budget.ledger.max_bytes, invocation_id='copy', expected_snapshot=receipt['io_prefix'])

    def test_changed_known_receipt_prefix_refused(self):
        budget = self.budget(); self.write(budget)
        receipt = copy.deepcopy(budget.last_receipt)
        receipt['io_prefix']['reserved']['snapshot_load_operations'] += 1
        with self.assertRaisesRegex(ValueError, 'prefix counters changed'): budget.bind_receipt(receipt)

    def test_expanded_view_guard_precedes_finite_scan_and_clone(self):
        envelope = dict(ENVELOPE, max_tensor_elements=4)
        budget = self.budget(envelope=envelope)
        with patch('dongxi_llms.training_snapshot.torch.isfinite', side_effect=AssertionError('no finite scan')):
            with self.assertRaises(WorkBudgetExceeded): self.write(budget, state={'x': torch.tensor([1.]).expand(100)})
        self.assertFalse((self.root / 'state.pt').exists())
        self.assertEqual(budget.ledger.snapshot()['completed']['snapshot_tensor_elements'], 0)
        self.assertEqual(len(budget.ledger.snapshot()['failed_tickets']), 1)

    def test_small_tensor_byte_envelope_save_refuses_before_finite_scan(self):
        budget = self.budget(envelope=dict(ENVELOPE, max_tensor_bytes=4))
        with patch('dongxi_llms.training_snapshot.torch.isfinite', side_effect=AssertionError('no finite scan')):
            with self.assertRaisesRegex(WorkBudgetExceeded, 'tensor bytes'):
                self.write(budget, state={'x': torch.ones(2, dtype=torch.float64)})
        self.assertFalse((self.root / 'state.pt').exists())

    def test_small_tensor_byte_envelope_load_refuses_before_finite_scan(self):
        budget = self.budget(envelope=dict(ENVELOPE, max_tensor_bytes=4))
        path, header = self.write(budget, state={'x': torch.ones(1)})
        forged = torch.load(path, weights_only=True)  # Test-only fault preparation.
        forged['state']['x'] = torch.ones(2, dtype=torch.float64)
        with patch('dongxi_llms.training_snapshot.torch.load', return_value=forged):
            with patch('dongxi_llms.training_snapshot.torch.isfinite', side_effect=AssertionError('no finite scan')):
                with self.assertRaisesRegex(WorkBudgetExceeded, 'tensor bytes'): self.read(budget, path, header)

    def test_nodes_cumulative_across_expected_and_observed_contracts(self):
        # Save contract+state uses seven nodes; load visits expected+observed
        # contracts before state and therefore crosses the same whole-op bound.
        budget = self.budget(envelope=dict(ENVELOPE, max_tree_nodes=7))
        path, header = self.write(budget, state={'x': torch.tensor([1.])})
        called = []
        with self.assertRaises(WorkBudgetExceeded):
            self.read(budget, path, header, validate_payload=lambda _: called.append(1))
        self.assertEqual(called, [])
        self.assertEqual(budget.ledger.snapshot()['known_partial']['snapshot_tree_nodes'], 7)

    def test_receipt_payload_prefix_disagreement_before_callback(self):
        budget = self.budget(); path, header = self.write(budget)
        forged = torch.load(path, weights_only=True)  # Test-only fault preparation.
        forged['snapshot_io']['io_prefix']['sequence'] += 1
        called = []
        with patch('dongxi_llms.training_snapshot.torch.load', return_value=forged):
            with self.assertRaisesRegex(ValueError, 'I/O prefix differs'):
                self.read(budget, path, header, validate_payload=lambda _: called.append(1))
        self.assertEqual(called, [])

    def test_serialization_failure_spending_and_partial_bytes_retained(self):
        budget = self.budget()
        def fail(payload, writer):
            writer.write(b'partial-serializer-output'); raise RuntimeError('injected writer failure')
        with patch('dongxi_llms.training_snapshot.torch.save', side_effect=fail):
            with self.assertRaises(RuntimeError): self.write(budget)
        self.assertEqual((self.root / 'state.pt').read_bytes(), b'partial-serializer-output')
        self.assertFalse((self.root / 'state.pt.commit.json').exists())
        state = budget.ledger.snapshot()
        self.assertEqual(state['known_partial']['snapshot_serialization_bytes'], 25)
        self.assertEqual(state['reserved']['snapshot_serialization_bytes'], MAX_BYTES)
        self.assertEqual(state['uncertain_upper']['snapshot_save_operations'], 1)

    def test_bootstrap_duplicate_fifo_symlink_and_oversize_refused(self):
        metadata = self.root / 'metadata.json'; metadata.write_text('{"x":1,"x":2}')
        with self.assertRaisesRegex(ValueError, 'Duplicate'): read_bounded_json(metadata)
        fifo = self.root / 'fifo'; os.mkfifo(fifo)
        with self.assertRaisesRegex(ValueError, 'regular'): read_bounded_json(fifo)
        link = self.root / 'link'; link.symlink_to(metadata)
        with self.assertRaises(OSError): read_bounded_json(link)
        metadata.write_bytes(b' ' * 65537)
        with self.assertRaisesRegex(ValueError, 'bounded'): read_bounded_json(metadata)

    def test_schema_missing_caps_boolean_and_changed_envelope_refused(self):
        caps = dict.fromkeys(IO_KEYS, 100)
        for bad in (dict(list(caps.items())[:-1]), dict(caps, snapshot_load_operations=True)):
            with self.assertRaises(ValueError): io_budget_contract(bad, ENVELOPE, 1024)
        with self.assertRaises(ValueError): io_budget_contract(caps, dict(ENVELOPE, max_tensor_bytes=MAX_BYTES+1), 1024)
        budget = self.budget(); path, header = self.write(budget)
        with self.assertRaisesRegex(ValueError, 'bound differs'):
            inspect_snapshot(path, io_budget=budget, expected_sha256=header['payload_sha256'],
                expected_bytes=header['payload_bytes'], expected_contract=SCIENCE, max_bytes=MAX_BYTES+1)

    def test_hook_inspection_contract_mutation_cannot_change_envelope(self):
        budget = self.budget(); path, header = self.write(budget)
        budget.contract['envelope']['max_tree_nodes'] += 1
        with patch('dongxi_llms.training_snapshot._open_regular', side_effect=AssertionError('no read')):
            with self.assertRaisesRegex(ValueError, 'inspection map changed'): self.read(budget, path, header)

    def test_bound_receipt_mutation_refused_before_payload_read(self):
        budget = self.budget(); path, header = self.write(budget)
        budget.expected_receipt['io_prefix']['sequence'] = False
        with patch('dongxi_llms.training_snapshot._open_regular', side_effect=AssertionError('no read')):
            with self.assertRaises(ValueError): self.read(budget, path, header)

    def test_bound_scientific_identity_mutation_refused_before_tree(self):
        budget = self.budget(); budget.science_sha256 = canonical_hash(dict(SCIENCE, lr=.02))
        with patch('dongxi_llms.training_snapshot._safe_tree', side_effect=AssertionError('no tree')):
            with self.assertRaisesRegex(ValueError, 'bound journal/scientific identity'): self.write(budget)
        self.assertFalse((self.root / 'state.pt').exists())

    def test_replacing_hook_ledger_cannot_charge_a_fresh_journal(self):
        budget = self.budget(); path, header = self.write(budget)
        other = self.budget(name='fresh.jsonl'); before = other.ledger.snapshot()
        budget.ledger = other.ledger
        with patch('dongxi_llms.training_snapshot._open_regular', side_effect=AssertionError('no read')):
            with self.assertRaisesRegex(ValueError, 'bound journal/scientific identity'): self.read(budget, path, header)
        self.assertEqual(before, other.ledger.snapshot())

    def test_payload_boolean_prefix_not_equal_to_integer_zero(self):
        budget = self.budget(); path, header = self.write(budget)
        forged = torch.load(path, weights_only=True)  # Test-only fault preparation.
        forged['snapshot_io']['io_prefix']['sequence'] = False
        called = []
        with patch('dongxi_llms.training_snapshot.torch.load', return_value=forged):
            with self.assertRaises(ValueError):
                self.read(budget, path, header, validate_payload=lambda _: called.append(1))
        self.assertEqual(called, [])

    def test_legacy_reference_stays_unaccounted_not_silently_migrated(self):
        path = self.root / 'legacy.pt'
        header = save_snapshot(path, contract=SCIENCE, state={'x': torch.ones(1)},
            completed_updates=0, parent_invocation='reference', max_bytes=MAX_BYTES)
        self.assertEqual(header['schema_version'], 1)
        budget = self.budget(); receipt = dict(schema='dongxi-snapshot-work-receipt-v1', snapshot=header,
            io_contract_sha256=canonical_hash(budget.contract), io_prefix=budget.ledger.snapshot(), runner_work_prefix=None)
        with self.assertRaisesRegex(ValueError, 'snapshot v2'): validate_work_receipt(receipt)

    def test_header_version_change_refused_before_restricted_load(self):
        path = self.root / 'legacy.pt'
        header = save_snapshot(path, contract=SCIENCE, state={'x': torch.ones(1)},
            completed_updates=0, parent_invocation='reference', max_bytes=MAX_BYTES)
        changed = dict(header, schema_version=2)
        Path(str(path) + '.commit.json').write_text(json.dumps(changed))  # Test-only fault.
        with patch('dongxi_llms.training_snapshot.torch.load', side_effect=AssertionError('no load')):
            with self.assertRaisesRegex(ValueError, 'explicit accounted/reference mode'):
                load_snapshot(path, expected_sha256=header['payload_sha256'], expected_bytes=header['payload_bytes'],
                    expected_contract=SCIENCE, max_bytes=MAX_BYTES)


if __name__ == '__main__': unittest.main()
