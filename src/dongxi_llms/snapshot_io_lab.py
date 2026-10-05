"""Small actual CPU reader lesson; no model acquisition or training launcher."""
import os
from pathlib import Path
import tempfile

import torch

from .run_identity import canonical_hash
from .snapshot_io_budget import IO_KEYS, SnapshotIOBudget, io_budget_contract, io_ledger_contract_sha256, read_work_receipt
from .training_snapshot import inspect_snapshot, load_snapshot, save_snapshot
from .work_budget import WorkLedger, WorkBudgetExceeded


def reader_admission_example():
    """One save/inspect, two good loads, one failed callback, then a refused load.

    A fresh handle binds the old pre-save prefix, retains all later charges and
    refuses the fourth load. Units are cooperative declared visits, not FLOPs.
    Temporary owned files are cleaned by TemporaryDirectory after the trace.
    """
    science = dict(runner='cpu-reader-microscope', learning_rate=.01, horizon=3)
    envelope = dict(max_payload_bytes=65536, max_tree_nodes=128,
        max_tensor_elements=64, max_tensor_bytes=256, max_primitive_bytes=8192)
    limits = dict.fromkeys(IO_KEYS, 1000000); limits['snapshot_load_operations'] = 3
    contract = io_budget_contract(limits, envelope, 1024 * 1024)
    science_sha = canonical_hash(science)
    journal_sha = io_ledger_contract_sha256(contract, science_sha)
    with tempfile.TemporaryDirectory(prefix='dongxi-reader-lesson-') as directory:
        root = Path(directory); os.chmod(root, 0o700)
        ledger = WorkLedger.create(root/'io.jsonl', limits=limits, contract_sha256=journal_sha,
            max_bytes=contract['max_journal_bytes'], invocation_id='lesson-A')
        try:
            budget = SnapshotIOBudget(ledger, contract=contract, scientific_contract_sha256=science_sha)
            path = root/'tiny.pt'; receipt_path = root/'retained-work.json'
            state = {'weights': torch.arange(8, dtype=torch.float32)}
            header = save_snapshot(path, contract=science, state=state, completed_updates=3,
                parent_invocation='lesson-A', max_bytes=envelope['max_payload_bytes'],
                io_budget=budget, work_receipt_path=receipt_path)
            receipt = read_work_receipt(receipt_path); budget.bind_receipt(receipt)
            args = dict(expected_sha256=header['payload_sha256'], expected_bytes=header['payload_bytes'],
                expected_contract=science, max_bytes=envelope['max_payload_bytes'], io_budget=budget)
            inspect_snapshot(path, **args)
            first = load_snapshot(path, **args); second = load_snapshot(path, **args)
            def fail_callback(_): raise ValueError('declared deliberate semantic rejection after generic read')
            try: load_snapshot(path, validate_payload=fail_callback, **args)
            except ValueError as error: failure = str(error)
            else: raise AssertionError('Declared callback control must fail')
            before_reopen = ledger.snapshot()
        finally: ledger.close()
        resumed = WorkLedger.open(root/'io.jsonl', limits=limits, contract_sha256=journal_sha,
            max_bytes=contract['max_journal_bytes'], invocation_id='lesson-B', expected_snapshot=receipt['io_prefix'])
        try:
            after_reopen = resumed.snapshot()
            budget = SnapshotIOBudget(resumed, contract=contract, scientific_contract_sha256=science_sha, expected_receipt=receipt)
            args['io_budget'] = budget
            try: load_snapshot(path, **args)
            except WorkBudgetExceeded as error: refusal = str(error)
            else: raise AssertionError('Fourth load must be refused before payload work')
            after_refusal = resumed.snapshot()
            return dict(scope='actual trusted-local CPU reader, not pretrained/physical quota',
                payload_bytes=header['payload_bytes'], receipt_prefix_sequence=receipt['io_prefix']['sequence'],
                numerical_state_equal=bool(torch.equal(first['state']['weights'], second['state']['weights'])),
                later_spending_retained=before_reopen == after_reopen,
                refused_before_new_ticket=after_reopen == after_refusal,
                load_trace=[dict(reserved=prefix['reserved']['snapshot_load_operations'],
                                 completed=prefix['completed']['snapshot_load_operations'])
                            for prefix in (receipt['io_prefix'], before_reopen, after_reopen, after_refusal)],
                failure=failure, refusal=refusal, envelope=envelope,
                reserved=after_refusal['reserved'], completed=after_refusal['completed'],
                attempted_upper=after_refusal['attempted_upper'], known_partial=after_refusal['known_partial'],
                uncertain_upper=after_refusal['uncertain_upper'], failed_tickets=after_refusal['failed_tickets'])
        finally: resumed.close()
