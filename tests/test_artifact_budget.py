"""Original cooperative byte/entry and retained-journal CPU failure controls."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from dongxi_llms.artifact_budget import ArtifactBudget,JOURNAL

OBSERVATIONS=[]


class ArtifactBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='dongxi-artifact-budget-')
        self.base=Path(self.temp.name);self.opened=[]

    def tearDown(self):
        for budget in self.opened:budget.close()
        self.temp.cleanup()

    def create(self,capacity=64,entries=8,name='owned'):
        budget=ArtifactBudget.create(self.base/name,campaign_id='original-cpu-artifacts',
            max_bytes=16384+capacity,max_entries=entries,journal_max_bytes=16384)
        self.opened.append(budget);return budget

    def restore(self,root,receipt):
        value=ArtifactBudget.restore(root,expected_receipt=receipt);self.opened.append(value);return value

    def test_exact_bundle_and_crossing_refusal_before_creation(self):
        budget=self.create(capacity=24);reservation=budget.reserve_bundle('exact',{'a':16,'b':8})
        before=budget.receipt()
        with self.assertRaisesRegex(ValueError,'Whole artifact'):budget.reserve_bundle('crossing',{'c':1})
        self.assertEqual(before,budget.receipt());self.assertFalse((budget.root/'c').exists())
        with reservation.writer('a') as handle:handle.write(b'a'*16)
        with reservation.writer('b') as handle:handle.write(b'b'*8)
        self.assertEqual(budget.usage()['reserved_pathname_bytes'],16408)
        OBSERVATIONS.append(dict(control='exact24-byte-bundle-and-refused25th',usage=budget.usage()))

    def test_entry_cap_counts_zero_byte_artifacts(self):
        budget=self.create(entries=3);reservation=budget.reserve_bundle('two',{'empty':0,'one':1})
        with reservation.writer('empty'):pass
        with reservation.writer('one') as handle:handle.write(b'x')
        with self.assertRaises(ValueError):budget.reserve_bundle('third',{'third':0})
        self.assertEqual(budget.usage()['reserved_entries'],3)

    def test_crossing_stream_retains_partial_and_full_reservation(self):
        budget=self.create(capacity=16);reservation=budget.reserve_bundle('partial',{'partial':16})
        with self.assertRaisesRegex(ValueError,'stream'):
            with reservation.writer('partial') as handle:
                handle.write(b'x'*8);handle.write(b'y'*9)
        self.assertEqual((budget.root/'partial').read_bytes(),b'x'*8)
        with self.assertRaises(ValueError):reservation.release_absent('partial')
        with self.assertRaises(ValueError):budget.reserve_bundle('next-output',{'next':1})
        self.assertEqual(budget.usage()['reserved_pathname_bytes'],16400)
        OBSERVATIONS.append(dict(control='stream-overflow-retained8-and-reserved16',usage=budget.usage()))

    def test_seek_refuses_before_moving_or_hole_growth(self):
        budget=self.create();reservation=budget.reserve_bundle('seek',{'data':16})
        with reservation.writer('data') as handle:
            handle.write(b'x'*8)
            for offset,whence in ((17,0),(9,1),(9,2),(-1,0),(True,0)):
                with self.assertRaises(ValueError):handle.seek(offset,whence)
                self.assertEqual(handle.tell(),8)
        self.assertEqual((budget.root/'data').stat().st_size,8)

    def test_seal_reclaims_only_successful_unused_capacity(self):
        budget=self.create(capacity=16);reservation=budget.reserve_bundle('old',{'old':16})
        with reservation.writer('old') as handle:handle.write(b'x'*8)
        budget.reserve_bundle('new',{'new':8})
        with self.assertRaises(ValueError):budget.reserve_bundle('excess',{'excess':1})
        self.assertEqual(budget.usage()['reserved_pathname_bytes'],16400)

    def test_absent_release_does_not_free_created_empty_partial(self):
        budget=self.create();reservation=budget.reserve_bundle('absence',{'absent':16,'partial':16})
        reservation.release_absent('absent')
        with self.assertRaises(RuntimeError):
            with reservation.writer('partial'):raise RuntimeError('authored early stream failure')
        with self.assertRaises(ValueError):reservation.release_absent('partial')
        with self.assertRaises(ValueError):budget.reserve_bundle('absence',{'duplicate-op':0})

    def test_header_hardlink_peak_and_exact_staging_removal(self):
        budget=self.create(capacity=32);reservation=budget.reserve_bundle('header',{'stage':16,'marker':16})
        with reservation.writer('stage') as handle:handle.write(b'header!!')
        reservation.link('stage','marker');peak=budget.usage()
        self.assertEqual(peak['reserved_entries'],3);self.assertEqual(peak['reserved_pathname_bytes'],16400)
        reservation.remove_staging('stage');ending=budget.usage()
        self.assertEqual(ending['reserved_entries'],2);self.assertEqual(ending['reserved_pathname_bytes'],16392)
        self.assertEqual((budget.root/'marker').read_bytes(),b'header!!')
        with self.assertRaises(ValueError):reservation.remove_staging('marker')
        OBSERVATIONS.append(dict(control='hardlinked-header-pathname-coexistence',peak=peak,ending=ending))

    def test_restored_prefix_retains_later_failed_partial(self):
        budget=self.create(capacity=64);reservation=budget.reserve_bundle('old',{'old':8})
        with reservation.writer('old') as handle:handle.write(b'o'*8)
        retained=budget.receipt();later=budget.reserve_bundle('later-failure',{'partial':32})
        with self.assertRaises(RuntimeError):
            with later.writer('partial') as handle:handle.write(b'p'*8);raise RuntimeError('authored retained failure')
        budget.close();restored=self.restore(budget.root,retained)
        self.assertEqual(restored.usage()['reserved_pathname_bytes'],16424)
        with self.assertRaises(ValueError):restored.reserve_bundle('new-invocation',{'new-output':32})
        restored.reserve_bundle('new-small-invocation',{'new-small-output':16})
        self.assertEqual((restored.root/'partial').read_bytes(),b'p'*8)
        OBSERVATIONS.append(dict(control='prefix-restore-retains-later32-reservation',usage=restored.usage(),retained=retained))

    def test_changed_caps_campaign_and_new_root_do_not_reset(self):
        budget=self.create();receipt=budget.receipt();budget.close()
        for key in ('max_bytes','max_entries','campaign_id'):
            changed=deepcopy(receipt)
            changed['identity'][key]=changed['identity'][key]+1 if key!='campaign_id' else 'different'
            with self.assertRaises(ValueError):self.restore(budget.root,changed)
        other=self.create(name='new-root');other.close()
        with self.assertRaises(ValueError):self.restore(other.root,receipt)
        with self.assertRaises(FileExistsError):self.create()

    def test_lock_refusal_preserves_active_owner(self):
        budget=self.create();receipt=budget.receipt()
        with self.assertRaises(BlockingIOError):self.restore(budget.root,receipt)
        self.assertEqual(receipt,budget.receipt())

    def test_symlink_ancestor_and_outside_paths_reject(self):
        budget=self.create();link=self.base/'symlink';link.symlink_to(budget.root,target_is_directory=True)
        with self.assertRaises(OSError):ArtifactBudget.restore(link,expected_receipt=budget.receipt())
        with self.assertRaises(OSError):ArtifactBudget.create(link/'child',campaign_id='x',max_bytes=32768,max_entries=2)
        for path in (self.base/'outside',budget.root/'nested'/'file',budget.root/'nested'/'..'/'file'):
            with self.assertRaises(ValueError):budget.name_for(path)
        with self.assertRaises(ValueError):budget.reserve_bundle('traversal',{'../outside':8})

    def test_root_replacement_and_journal_replacement_reject(self):
        budget=self.create();os.rename(budget.root,self.base/'retained-old-root');budget.root.mkdir(mode=0o700)
        with self.assertRaisesRegex(ValueError,'root identity'):budget.usage()
        budget=self.create(name='second');os.rename(budget.root/JOURNAL,budget.root/'retained-old-journal')
        (budget.root/JOURNAL).write_bytes(b'');os.chmod(budget.root/JOURNAL,0o600)
        with self.assertRaisesRegex(ValueError,'journal path identity'):budget.usage()

    def test_untracked_symlink_special_file_and_external_link_reject(self):
        for kind in ('unknown','symlink','fifo','hardlink'):
            budget=self.create(name=kind)
            if kind=='unknown':(budget.root/'unknown').write_bytes(b'x')
            elif kind=='symlink':(budget.root/'link').symlink_to(self.base/'outside')
            elif kind=='fifo':os.mkfifo(budget.root/'fifo',0o600)
            else:
                reservation=budget.reserve_bundle('file',{'file':8})
                with reservation.writer('file') as handle:handle.write(b'x')
                os.link(budget.root/'file',self.base/'unowned-link')
            with self.assertRaises(ValueError):budget.usage()

    def test_permissions_and_uid_gate(self):
        budget=self.create()
        with patch('dongxi_llms.artifact_budget.os.getuid',return_value=os.getuid()+1):
            with self.assertRaises(ValueError):budget.usage()
        os.chmod(budget.root,0o755)
        with self.assertRaises(ValueError):budget.usage()

    def test_sealed_content_tamper_and_growth_reject(self):
        for fault in ('same-size','growth'):
            budget=self.create(name=fault);reservation=budget.reserve_bundle('file',{'file':8})
            with reservation.writer('file') as handle:handle.write(b'original')
            (budget.root/'file').write_bytes(b'changed!' if fault=='same-size' else b'changed-and-too-long')
            with self.assertRaises(ValueError):budget.usage()

    def test_journal_rollback_hash_and_incomplete_tail_retained(self):
        for fault in ('rollback','hash','tail','duplicate'):
            budget=self.create(name=fault);old=budget.receipt();budget.reserve_bundle('one',{'one':8})
            latest=budget.receipt();path=budget.root/JOURNAL;raw=path.read_bytes();budget.close()
            if fault=='rollback':data=raw[:old['journal_bytes']]
            elif fault=='hash':data=raw.replace(b'"one":8',b'"one":9')
            elif fault=='tail':data=raw+b'{incomplete'
            else:data=raw.replace(b'"kind":"reserve"',b'"kind":"reserve","kind":"reserve"')
            path.write_bytes(data)
            with self.assertRaises(ValueError):self.restore(budget.root,latest if fault=='rollback' else old)
            self.assertEqual(path.read_bytes(),data)

    def test_journal_fsync_failure_is_retained_not_artifact_creation(self):
        budget=self.create();receipt=budget.receipt()
        with patch('dongxi_llms.artifact_budget.os.fsync',side_effect=OSError('authored journal fsync failure')):
            with self.assertRaises(OSError):budget.reserve_bundle('failed-commit',{'not-created':16})
        self.assertFalse((budget.root/'not-created').exists());budget.close()
        restored=self.restore(budget.root,receipt)
        self.assertEqual(restored.usage()['reserved_pathname_bytes'],16400)
        OBSERVATIONS.append(dict(control='visible-journal-fsync-failure-reservation-retained',usage=restored.usage(),
                                 limitation='visible bytes after failure do not prove crash durability'))

    def test_artifact_fsync_failure_keeps_conservative_partial(self):
        budget=self.create();reservation=budget.reserve_bundle('partial',{'partial':16})
        handle=reservation.writer('partial')
        with patch('dongxi_llms.artifact_budget.os.fsync',side_effect=OSError('authored artifact fsync failure')):
            with self.assertRaises(OSError):
                with handle:handle.write(b'x'*8)
        self.assertEqual((budget.root/'partial').read_bytes(),b'x'*8)
        self.assertEqual(budget.usage()['reserved_pathname_bytes'],16400)

    def test_fixed_journal_capacity_refuses_without_reset(self):
        budget=self.create(capacity=0,entries=2)
        for index in range(1000):
            try:
                reservation=budget.reserve_bundle(f'empty-{index}',{f'empty-{index}':0})
                reservation.release_absent(f'empty-{index}')
            except ValueError as error:
                self.assertIn('journal capacity',str(error));break
        else:self.fail('The fixed journal did not refuse')
        self.assertLessEqual((budget.root/JOURNAL).stat().st_size,16384)
        self.assertEqual(budget.identity['journal_max_bytes'],16384)

    def test_bad_caps_receipts_and_required_physical_backend_refuse(self):
        for cap in (True,0,-1,2**63):
            with self.assertRaises(ValueError):ArtifactBudget.create(self.base/'not-created',campaign_id='x',max_bytes=cap,max_entries=2)
        with self.assertRaises(RuntimeError):ArtifactBudget.create(self.base/'not-created',campaign_id='x',max_bytes=32768,
            max_entries=2,require_physical_backend=True)
        self.assertFalse((self.base/'not-created').exists())
        budget=self.create();receipt=budget.receipt();budget.close()
        for fault in ('unknown','bool'):
            bad=deepcopy(receipt)
            if fault=='unknown':bad['reset']=True
            else:bad['sequence']=False
            with self.assertRaises(ValueError):self.restore(budget.root,bad)

    def test_public_identity_and_state_views_cannot_change_frozen_caps(self):
        budget=self.create(capacity=16);budget.reserve_bundle('full',{'full':16})
        identity=budget.identity;identity['max_bytes']=999999
        files=budget.files;files['full']['limit']=0;files.clear()
        with self.assertRaises(AttributeError):budget.identity=identity
        with self.assertRaises(AttributeError):budget.files=files
        with self.assertRaises(ValueError):budget.reserve_bundle('no-reset',{'new':1})
        self.assertEqual(budget.identity['max_bytes'],16400)
        self.assertEqual(budget.files['full']['limit'],16)

    def test_live_same_inode_journal_tamper_refuses_before_new_reservation(self):
        budget=self.create();path=budget.root/JOURNAL;before=path.stat().st_ino
        data=path.read_bytes().replace(b'original-cpu-artifacts',b'original-cpu-artifactX')
        path.write_bytes(data);self.assertEqual(before,path.stat().st_ino)
        with self.assertRaisesRegex(ValueError,'journal bytes'):
            budget.reserve_bundle('not-after-tamper',{'not-created':8})
        self.assertFalse((budget.root/'not-created').exists());self.assertEqual(path.read_bytes(),data)

    def test_fifo_journal_and_digest_open_refuse_without_peer(self):
        budget=self.create();receipt=budget.receipt();budget.close()
        receipt_path=self.base/'independent-receipt.json';receipt_path.write_text(json.dumps(receipt))
        os.rename(budget.root/JOURNAL,self.base/'retained-regular-journal')
        os.mkfifo(budget.root/JOURNAL,0o600)
        code='''import json,sys,os
from dongxi_llms.artifact_budget import ArtifactBudget
from pathlib import Path
root=Path(sys.argv[2])
try:
    if sys.argv[1]=='journal':
        ArtifactBudget.restore(root,expected_receipt=json.loads(Path(sys.argv[3]).read_text()))
    else:
        with ArtifactBudget.create(root,campaign_id='fifo-digest-control',max_bytes=16448,max_entries=4,journal_max_bytes=16384) as b:
            b.reserve_bundle('fifo-digest',{'fifo':8});os.mkfifo(root/'fifo',0o600);b._digest_file('fifo')
except ValueError as error:print(str(error))
else:raise AssertionError('FIFO must be refused')
'''
        for mode,root in (('journal',budget.root),('digest',self.base/'digest-child')):
            command=[sys.executable,'-c',code,mode,str(root),str(receipt_path)]
            result=subprocess.run(command,capture_output=True,text=True,timeout=3)
            OBSERVATIONS.append(dict(control='fifo-'+mode+'-nonblocking-refusal',command=command,
                                     actual_exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr))
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('regular' if mode=='journal' else 'Unsafe or over-cap',result.stdout)


if __name__=='__main__':unittest.main()
