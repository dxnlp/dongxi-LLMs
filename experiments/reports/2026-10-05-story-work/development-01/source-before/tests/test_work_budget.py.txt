"""Original CPU journal controls; no model or external resource acquisition."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from dongxi_llms.work_budget import WorkLedger, WorkLedgerError, WorkBudgetExceeded, validate_limits

CAPS={'calls':10,'positions':50}
DIGEST='1'*64
BOUND=1024*1024


class WorkBudgetTests(unittest.TestCase):
    def setUp(self):
        directory=tempfile.TemporaryDirectory(prefix='dongxi-work-budget-')
        self.addCleanup(directory.cleanup);self.root=Path(directory.name)
        self.path=self.root/'work.jsonl'

    def create(self,**changes):
        options=dict(limits=CAPS,contract_sha256=DIGEST,max_bytes=BOUND,invocation_id='initial')
        options.update(changes)
        result=WorkLedger.create(self.path,**options);self.addCleanup(result.close);return result

    def reopen(self,**changes):
        options=dict(limits=CAPS,contract_sha256=DIGEST,max_bytes=BOUND,invocation_id='recovery')
        options.update(changes)
        result=WorkLedger.open(self.path,**options);self.addCleanup(result.close);return result

    def test_strict_caps_and_vector_types(self):
        for value in (True,-1,1.1,2**63):
            with self.subTest(value=value),self.assertRaises(ValueError):validate_limits({'calls':value})
        for value in ({},{'Bad':1},{'unknown space':1}):
            with self.assertRaises(ValueError):validate_limits(value)
        ledger=self.create()
        for vector in ({'calls':True},{'other':1},{'calls':-1},{'calls':1.5},{}):
            with self.assertRaises(ValueError):ledger.reserve(vector,operation='invalid')

    def test_multidimensional_refusal_is_atomic(self):
        ledger=self.create();ledger.reserve({'calls':2,'positions':45},operation='first')
        before=self.path.read_bytes();state=ledger.snapshot()
        with self.assertRaises(WorkBudgetExceeded):ledger.reserve({'calls':3,'positions':6},operation='refused')
        self.assertEqual(before,self.path.read_bytes());self.assertEqual(state,ledger.snapshot())

    def test_no_refund_completed_failure_and_uncertain(self):
        ledger=self.create();one=ledger.reserve({'calls':3,'positions':15},operation='early-eos')
        ledger.complete(one,{'calls':1,'positions':3})
        two=ledger.reserve({'calls':3,'positions':15},operation='backend-failure')
        ledger.fail(two,'authored failure',known_actual={'calls':1,'positions':3},attempted={'calls':2,'positions':7})
        ledger.reserve({'calls':2,'positions':10},operation='crash-open')
        state=ledger.snapshot()
        self.assertEqual(state['reserved'],{'calls':8,'positions':40})
        self.assertEqual(state['completed'],{'calls':1,'positions':3})
        self.assertEqual(state['known_partial'],{'calls':1,'positions':3})
        self.assertEqual(state['attempted_upper'],{'calls':5,'positions':20})
        self.assertEqual(state['uncertain_upper'],{'calls':3,'positions':14})
        with self.assertRaises(WorkBudgetExceeded):ledger.reserve({'calls':3},operation='no-refill')

    def test_snapshot_prefix_retains_later_work(self):
        ledger=self.create();first=ledger.reserve({'calls':2},operation='prefix');ledger.complete(first,{'calls':2})
        prefix=ledger.snapshot();ledger.reserve({'calls':4},operation='later-crash');ledger.close()
        resumed=self.reopen()
        with self.assertRaises(WorkLedgerError):resumed.reserve({'calls':1},operation='unbound')
        with self.assertRaises(WorkLedgerError):resumed.fail(prefix['sequence']+1,'unbound finalization')
        current=resumed.validate_snapshot(prefix)
        self.assertEqual(current['reserved']['calls'],6)
        self.assertEqual(current['uncertain_upper']['calls'],4)

    def test_new_or_copied_journal_cannot_apply_old_prefix(self):
        ledger=self.create();prefix=ledger.snapshot();ledger.close()
        copied=self.root/'copy.jsonl';shutil.copyfile(self.path,copied);copied.chmod(0o600)
        with self.assertRaises(ValueError):WorkLedger.open(copied,limits=CAPS,contract_sha256=DIGEST,max_bytes=BOUND,invocation_id='copy',expected_snapshot=prefix)
        self.path=self.root/'fresh.jsonl';fresh=self.create()
        with self.assertRaises(ValueError):fresh.validate_snapshot(prefix)

    def test_changed_contract_and_caps_and_boolean_snapshot_refused(self):
        ledger=self.create();prefix=ledger.snapshot();ledger.close()
        for changes in ({'limits':{'calls':11,'positions':50}},{'contract_sha256':'2'*64},{'max_bytes':BOUND+1}):
            with self.assertRaises(ValueError):self.reopen(**changes)
        opened=self.reopen();changed=deepcopy(prefix);changed['limits']['calls']=True
        with self.assertRaises(ValueError):opened.validate_snapshot(changed)

    def test_single_writer_refuses_lock(self):
        self.create()
        with self.assertRaises(BlockingIOError):self.reopen()

    def test_snapshot_mutation_does_not_refill(self):
        ledger=self.create();ledger.reserve({'calls':9},operation='spent')
        snapshot=ledger.snapshot();snapshot['reserved']['calls']=0;snapshot['limits']['calls']=100
        self.assertEqual(ledger.snapshot()['reserved']['calls'],9)
        with self.assertRaises(WorkBudgetExceeded):ledger.reserve({'calls':2},operation='no-reset')

    def test_mutable_inspection_maps_refused(self):
        for attribute in ('limits','header','tickets','prefixes'):
            with self.subTest(attribute=attribute):
                self.path=self.root/f'{attribute}.jsonl';ledger=self.create()
                mapping=getattr(ledger,attribute);mapping['authored-mutation']=1
                with self.assertRaises(WorkLedgerError):ledger.snapshot()

    def test_same_size_raw_mutation_refused(self):
        ledger=self.create();raw=self.path.read_bytes()
        altered=raw.replace(b'111111',b'222222',1);self.assertEqual(len(raw),len(altered))
        with self.path.open('r+b') as handle:handle.write(altered)
        with self.assertRaises(WorkLedgerError):ledger.reserve({'calls':1},operation='tampered')
        self.assertEqual(self.path.read_bytes(),altered)

    def test_symlink_and_fifo_targets_refused_without_blocking(self):
        ledger=self.create();ledger.close();original=self.path
        link=self.root/'link';link.symlink_to(original)
        self.path=link
        with self.assertRaises(OSError):self.reopen()
        fifo=self.root/'fifo';os.mkfifo(fifo,0o600);self.path=fifo
        with self.assertRaises(ValueError):self.reopen()
        ancestor=self.root/'ancestor';ancestor.symlink_to(self.root,target_is_directory=True)
        self.path=ancestor/'other.jsonl'
        with self.assertRaises(OSError):self.create()

    def test_mode_owner_and_hardlink_refused(self):
        self.root.chmod(0o755)
        with self.assertRaises(ValueError):self.create()
        self.root.chmod(0o700);ledger=self.create();ledger.close();self.path.chmod(0o640)
        with self.assertRaises(ValueError):self.reopen()
        self.path.chmod(0o600)
        os.link(self.path,self.root/'hardlink')
        with self.assertRaises(ValueError):self.reopen()
        # Owner is observed from fstat, not assumed from path placement.
        with patch('dongxi_llms.work_budget.os.getuid',return_value=os.getuid()+1),self.assertRaises(ValueError):self.reopen()

    def test_live_unlink_replacement_and_ancestor_move_refused(self):
        for operation in ('unlink','replace','ancestor'):
            with self.subTest(operation=operation):
                directory=self.root/operation;directory.mkdir(mode=0o700);self.path=directory/'work.jsonl'
                ledger=self.create();raw=self.path.read_bytes()
                if operation=='unlink':self.path.unlink()
                elif operation=='replace':
                    replacement=directory/'replacement';replacement.write_bytes(raw);replacement.chmod(0o600);replacement.replace(self.path)
                else:
                    directory.rename(self.root/'moved');directory.mkdir(mode=0o700)
                with self.assertRaises(WorkLedgerError):ledger.reserve({'calls':1},operation='detached')

    def test_torn_corrupt_and_duplicate_json_retained(self):
        ledger=self.create();ledger.close();valid=self.path.read_bytes()
        for raw in (valid+b'{',valid.replace(b'"schema":',b'"schema":"duplicate","schema":',1),valid.replace(b'111111',b'222222',1)):
            self.path.write_bytes(raw);before=self.path.read_bytes()
            with self.assertRaises((WorkLedgerError,ValueError)):self.reopen(expected_snapshot=ledger.prefixes[0])
            self.assertEqual(before,self.path.read_bytes())

    def test_byte_bound_and_failed_append_are_retained(self):
        ledger=self.create(max_bytes=500);before=self.path.read_bytes()
        with self.assertRaises(WorkLedgerError):ledger.reserve({'calls':1},operation='bounded')
        self.assertEqual(before,self.path.read_bytes())
        self.path=self.root/'fsync.jsonl';other=self.create()
        with patch('dongxi_llms.work_budget.os.fsync',side_effect=OSError('authored fsync failure')),self.assertRaises(WorkLedgerError):
            other.reserve({'calls':1},operation='fsync')
        self.assertTrue(other.poisoned)
        with self.assertRaises(WorkLedgerError):other.snapshot()
        other.close();opened=self.reopen();opened.validate_snapshot(opened.prefixes[0])
        self.assertEqual(opened.snapshot()['reserved']['calls'],1)

    def test_failure_attempted_semantics_and_ticket_finality(self):
        ledger=self.create();ticket=ledger.reserve({'calls':2},operation='fail')
        with self.assertRaises(ValueError):ledger.fail(ticket,'bad',known_actual={'calls':2},attempted={'calls':1})
        ledger.fail(ticket,'unknown entered cost')
        self.assertEqual(ledger.snapshot()['uncertain_upper']['calls'],2)
        with self.assertRaises(ValueError):ledger.complete(ticket,{'calls':1})


if __name__=='__main__':unittest.main()
