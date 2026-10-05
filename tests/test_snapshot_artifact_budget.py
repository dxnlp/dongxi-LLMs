"""Actual shared snapshot reservation integration and fresh-process controls."""
from copy import deepcopy
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
from dongxi_llms.artifact_budget import ArtifactBudget,JOURNAL
from dongxi_llms import training_snapshot as snapshots

ROOT=Path(__file__).resolve().parents[1]
CONTRACT={'source':'original-six-value-fixture','recipe':{'horizon':2,'lr':.01}}
LIMIT=4096
OBSERVATIONS=[]


def save(path,budget=None):
    return snapshots.save_snapshot(path,contract=CONTRACT,state={'tensor':torch.arange(6.)},
        completed_updates=1,parent_invocation='authored-cpu-snapshot',max_bytes=LIMIT,artifact_budget=budget)


class SnapshotArtifactBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='dongxi-snapshot-budget-');self.base=Path(self.temp.name)
        self.budgets=[]

    def tearDown(self):
        for budget in self.budgets:budget.close()
        self.temp.cleanup()

    def budget(self,*,capacity=200000,entries=16,name='owned'):
        value=ArtifactBudget.create(self.base/name,campaign_id='fixed-snapshot-cpu',max_bytes=capacity,
            max_entries=entries,journal_max_bytes=32768)
        self.budgets.append(value);return value

    def test_refuse_entire_peak_bundle_before_serializer_or_file_open(self):
        required=32768+LIMIT+2*snapshots.HEADER_LIMIT
        for budget in (self.budget(capacity=required-1,name='bytes'),self.budget(entries=3,name='entries')):
            before=budget.receipt()
            with patch.object(snapshots.torch,'save',side_effect=AssertionError('No serialization')):
                with self.assertRaises(ValueError):save(budget.root/'refused.pt',budget)
            self.assertEqual(before,budget.receipt());self.assertEqual(os.listdir(budget.root),[JOURNAL])
        OBSERVATIONS.append(dict(control='snapshot-bundle-before-serializer-refusal',required_peak_bytes=required,
                                 byte_cap=required-1,required_entries=4,entry_cap=3))

    def test_exact_envelope_and_no_hook_bytes_and_semantics_match(self):
        budget=self.budget(capacity=32768+LIMIT+2*snapshots.HEADER_LIMIT,entries=4)
        bounded=budget.root/'bounded.pt';plain=self.base/'plain.pt'
        header=save(bounded,budget);plain_header=save(plain)
        self.assertEqual(header,plain_header);self.assertEqual(bounded.read_bytes(),plain.read_bytes())
        payload=snapshots.load_snapshot(bounded,expected_sha256=header['payload_sha256'],expected_bytes=header['payload_bytes'],
            expected_contract=CONTRACT,max_bytes=LIMIT)
        self.assertTrue(torch.equal(payload['state']['tensor'],torch.arange(6.)))
        self.assertEqual(budget.usage()['reserved_entries'],3)
        self.assertEqual(budget.usage()['reserved_pathname_bytes'],32768+header['payload_bytes']+
                         Path(str(bounded)+'.commit.json').stat().st_size)
        self.assertFalse(any('.commit-' in name for name in os.listdir(budget.root)))
        OBSERVATIONS.append(dict(control='actual-snapshot-exact-envelope',header=header,usage=budget.usage()))

    def test_two_snapshots_keep_old_new_and_header_staging_charged(self):
        budget=self.budget();old=budget.root/'old.pt';save(old,budget)
        retained=old.read_bytes(),Path(str(old)+'.commit.json').read_bytes();peaks=[]
        real=budget.reserve_bundle
        def observe(*args,**kwargs):
            result=real(*args,**kwargs);peaks.append(budget.usage());return result
        with patch.object(budget,'reserve_bundle',side_effect=observe):save(budget.root/'new.pt',budget)
        self.assertEqual(retained,(old.read_bytes(),Path(str(old)+'.commit.json').read_bytes()))
        self.assertEqual(peaks[0]['reserved_entries'],6)
        self.assertGreater(peaks[0]['reserved_pathname_bytes'],32768+LIMIT+2*snapshots.HEADER_LIMIT)
        OBSERVATIONS.append(dict(control='old-new-staging-snapshot-peak',peak=peaks[0],ending=budget.usage()))

    def test_serializer_failure_retains_partial_and_blocks_fresh_output_reset(self):
        budget=self.budget();old=budget.root/'old.pt';save(old,budget);before=old.read_bytes()
        retained=budget.receipt()
        def partial(payload,handle):handle.write(b'authored-partial');raise RuntimeError('authored serializer failure')
        with patch.object(snapshots.torch,'save',side_effect=partial):
            with self.assertRaises(RuntimeError):save(budget.root/'partial.pt',budget)
        self.assertEqual((budget.root/'partial.pt').read_bytes(),b'authored-partial')
        usage=budget.usage();self.assertEqual(usage['reserved_entries'],6)
        budget.close();restored=ArtifactBudget.restore(budget.root,expected_receipt=retained);self.budgets.append(restored)
        self.assertEqual(restored.usage()['reserved_pathname_bytes'],usage['reserved_pathname_bytes'])
        with patch.object(snapshots.torch,'save',side_effect=AssertionError('No reset serializer')):
            with self.assertRaises(ValueError):save(restored.root/'new-invocation.pt',restored)
        self.assertEqual(old.read_bytes(),before)
        OBSERVATIONS.append(dict(control='failed-snapshot-partial-and-whole-reservation-retained',usage=usage,
                                 partial_bytes=len(b'authored-partial'),new_output_refused=True))

    def test_payload_fsync_failure_retains_previous_and_partial_capacity(self):
        budget=self.budget();old=budget.root/'old.pt';save(old,budget);before=old.read_bytes()
        original=os.fsync;journal_inode=os.fstat(budget.journal_fd).st_ino
        def fault(fd):
            info=os.fstat(fd)
            if stat.S_ISREG(info.st_mode) and info.st_ino!=journal_inode:
                raise OSError('authored payload fsync failure')
            return original(fd)
        with patch('dongxi_llms.training_snapshot.os.fsync',side_effect=fault):
            with self.assertRaises(OSError):save(budget.root/'fsync-partial.pt',budget)
        self.assertEqual(old.read_bytes(),before)
        self.assertGreater((budget.root/'fsync-partial.pt').stat().st_size,0)
        self.assertEqual(budget.files['fsync-partial.pt']['limit'],LIMIT)
        self.assertEqual(budget.usage()['reserved_entries'],6)

    def test_header_link_failure_retains_stage_and_unpublished_marker_capacity(self):
        budget=self.budget();old=budget.root/'old.pt';save(old,budget);before=old.read_bytes()
        retained=budget.receipt()
        with patch('dongxi_llms.artifact_budget.os.link',side_effect=OSError('authored publication failure')):
            with self.assertRaises(OSError):save(budget.root/'failed-link.pt',budget)
        self.assertEqual(old.read_bytes(),before)
        self.assertFalse(Path(str(budget.root/'failed-link.pt')+'.commit.json').exists())
        stage=[name for name in os.listdir(budget.root) if name.startswith('failed-link.pt.commit-')]
        self.assertEqual(len(stage),1)
        usage=budget.usage();budget.close()
        restored=ArtifactBudget.restore(budget.root,expected_receipt=retained);self.budgets.append(restored)
        self.assertEqual(restored.usage()['reserved_pathname_bytes'],usage['reserved_pathname_bytes'])
        self.assertEqual(restored.files['failed-link.pt.commit.json']['limit'],snapshots.HEADER_LIMIT)
        OBSERVATIONS.append(dict(control='header-publication-failure-retained',usage=usage,stage_name=stage[0]))

    def test_outside_root_and_existing_snapshot_refuse(self):
        budget=self.budget();path=budget.root/'one.pt';save(path,budget);before=budget.receipt()
        with self.assertRaises(ValueError):save(self.base/'outside.pt',budget)
        with self.assertRaises(FileExistsError):save(path,budget)
        self.assertEqual(before,budget.receipt());self.assertFalse((self.base/'outside.pt').exists())

    def test_fresh_process_restores_retained_prefix_and_saves_actual_snapshot(self):
        budget=self.budget();save(budget.root/'parent.pt',budget)
        retained=budget.receipt();receipt_path=self.base/'independently-retained.json'
        receipt_path.write_text(json.dumps(retained))
        later=budget.reserve_bundle('after-checkpoint-failure',{'retained-partial':256})
        with self.assertRaises(RuntimeError):
            with later.writer('retained-partial') as handle:handle.write(b'p'*8);raise RuntimeError('authored later failure')
        budget.close()
        command=[sys.executable,str(Path(__file__).resolve()),'--restore-child',str(budget.root),'--receipt',str(receipt_path)]
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        observed=json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(observed['retained_partial_capacity'],256)
        self.assertEqual(observed['retained_partial_actual_bytes'],8)
        self.assertEqual(observed['completed_updates'],1)
        self.assertTrue(observed['tensor_exact']);self.assertEqual(observed['identity'],retained['identity'])
        OBSERVATIONS.append(dict(control='fresh-process-prefix-and-actual-snapshot',command=command,
            actual_exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr,observation=observed))

    def test_fifo_header_and_payload_refuse_without_peer_or_deserialization(self):
        ordinary=self.base/'ordinary.pt';header=save(ordinary);marker=Path(str(ordinary)+'.commit.json')
        code='''import sys
from unittest.mock import patch
from dongxi_llms.training_snapshot import inspect_snapshot
try:
    with patch('dongxi_llms.training_snapshot.torch.load',side_effect=AssertionError('No deserialization')):
        inspect_snapshot(sys.argv[1],expected_sha256=sys.argv[2],expected_bytes=int(sys.argv[3]),
            expected_contract={'source':'original-six-value-fixture','recipe':{'horizon':2,'lr':.01}},max_bytes=4096)
except ValueError as error:print(str(error))
else:raise AssertionError('FIFO must be refused')
'''
        for role in ('header','payload'):
            target=self.base/(role+'.pt');target_marker=Path(str(target)+'.commit.json')
            if role=='header':target.write_bytes(ordinary.read_bytes());os.mkfifo(target_marker,0o600)
            else:os.mkfifo(target,0o600);target_marker.write_bytes(marker.read_bytes())
            command=[sys.executable,'-c',code,str(target),header['payload_sha256'],str(header['payload_bytes'])]
            result=subprocess.run(command,capture_output=True,text=True,timeout=3)
            OBSERVATIONS.append(dict(control='fifo-snapshot-'+role+'-nonblocking-refusal',command=command,
                                     actual_exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr))
            self.assertEqual(result.returncode,0,result.stderr);self.assertIn('regular file',result.stdout)


def child(root,receipt):
    # The expectation was independently written before the retained later failure.
    expected=json.loads(receipt.read_text())
    with ArtifactBudget.restore(root,expected_receipt=expected) as budget:
        capacity=budget.files['retained-partial']['limit'];actual=(budget.root/'retained-partial').stat().st_size
        header=save(budget.root/'child-new-output.pt',budget)
        payload=snapshots.load_snapshot(budget.root/'child-new-output.pt',expected_sha256=header['payload_sha256'],
            expected_bytes=header['payload_bytes'],expected_contract=CONTRACT,max_bytes=LIMIT)
        print(json.dumps(dict(identity=budget.identity,retained_partial_capacity=capacity,
            retained_partial_actual_bytes=actual,completed_updates=payload['completed_updates'],
            tensor_exact=torch.equal(payload['state']['tensor'],torch.arange(6.)),usage=budget.usage())))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--restore-child',type=Path);parser.add_argument('--receipt',type=Path)
    parser.add_argument('--verify-tests',type=Path);parser.add_argument('--measure-tests',type=Path);args=parser.parse_args()
    if args.restore_child:return child(args.restore_child,args.receipt)
    if args.verify_tests:
        if args.verify_tests.exists():raise FileExistsError(args.verify_tests)
        observed=args.verify_tests.with_suffix('.observations.json')
        paths=['src/dongxi_llms/artifact_budget.py','src/dongxi_llms/training_snapshot.py',
               'tests/test_artifact_budget.py','tests/test_snapshot_artifact_budget.py',
               'experiments/specs/2026-10-05-cooperative-artifact-budget.md',
               'experiments/specs/2026-10-05-cooperative-artifact-budget-fifo.md','uv.lock']
        hashes=lambda:{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
        before=hashes();command=[sys.executable,str(Path(__file__).resolve()),'--measure-tests',str(observed)]
        started=time.monotonic();result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=60)
        evidence=dict(command=command,actual_exit_code=result.returncode,seconds=time.monotonic()-started,
            stdout=result.stdout,stderr=result.stderr,source_before=before,source_after=hashes(),
            environment=dict(executable=sys.executable,python=platform.python_version(),torch=str(torch.__version__),
                platform=platform.platform(),cuda_available=torch.cuda.is_available()),observations=str(observed),
            observations_sha256=hashlib.sha256(observed.read_bytes()).hexdigest() if observed.exists() else None)
        with args.verify_tests.open('x') as handle:json.dump(evidence,handle,indent=2);handle.write('\n')
        print(json.dumps(dict(evidence=str(args.verify_tests),actual_exit_code=result.returncode)))
        return result.returncode
    if args.measure_tests:
        import test_artifact_budget
        loader=unittest.TestLoader();suite=unittest.TestSuite([
            loader.loadTestsFromTestCase(test_artifact_budget.ArtifactBudgetTests),
            loader.loadTestsFromTestCase(SnapshotArtifactBudgetTests),
            loader.discover(str(ROOT/'tests'),pattern='test_training_snapshot*.py')])
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        with args.measure_tests.open('x') as handle:json.dump(test_artifact_budget.OBSERVATIONS+OBSERVATIONS,handle,indent=2);handle.write('\n')
        return 0 if result.wasSuccessful() else 1
    unittest.main(argv=[sys.argv[0]])


if __name__=='__main__':raise SystemExit(main())
