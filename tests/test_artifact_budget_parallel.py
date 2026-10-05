"""Small byte-only controls for opt-in, ordered complete-file SHA scheduling."""
from concurrent.futures import ThreadPoolExecutor
import errno
import hashlib
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from dongxi_llms import artifact_budget as artifacts
import test_artifact_budget as original_controls


class ParallelOriginalControls(original_controls.ArtifactBudgetTests):
    """Run every existing ledger failure control with the opt-in configuration."""
    def create(self,capacity=64,entries=8,name='owned'):
        value=artifacts.ArtifactBudget.create(self.base/name,
            campaign_id='original-cpu-artifacts',max_bytes=16384+capacity,
            max_entries=entries,journal_max_bytes=16384,hash_workers=4)
        self.opened.append(value);return value

    def restore(self,root,receipt):
        value=artifacts.ArtifactBudget.restore(root,expected_receipt=receipt,hash_workers=4)
        self.opened.append(value);return value


class ParallelInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='dongxi-artifact-parallel-')
        self.base=Path(self.temp.name);self.opened=[]

    def tearDown(self):
        for value in self.opened:value.close()
        self.temp.cleanup()

    def budget(self,workers=4,name='owned',capacity=64,entries=16):
        value=artifacts.ArtifactBudget.create(self.base/name,campaign_id='parallel-byte-control',
            max_bytes=16384+capacity,max_entries=entries,journal_max_bytes=16384,
            hash_workers=workers)
        self.opened.append(value);return value

    def files(self,count=4,workers=4):
        value=self.budget(workers=workers,capacity=max(64,count*8))
        names=[f'file-{index}' for index in range(count)]
        reservation=value.reserve_bundle('authored-files',{name:8 for name in names})
        for index,name in enumerate(names):
            with reservation.writer(name) as handle:handle.write(bytes([index])*8)
        return value,names

    def test_configuration_is_exact_bounded_and_precedes_filesystem_effects(self):
        for bad in (False,True,0,-1,5,2**63,1.0,'4',None):
            with self.subTest(value=bad),patch.object(artifacts.os,'mkdir',
                    side_effect=AssertionError('No directory mutation')):
                with self.assertRaisesRegex(ValueError,'hash_workers'):
                    self.budget(workers=bad)
            with self.subTest(restore=bad),patch.object(artifacts.ArtifactBudget,'_open',
                    side_effect=AssertionError('No root open or lock')):
                with self.assertRaisesRegex(ValueError,'hash_workers'):
                    artifacts.ArtifactBudget.restore(self.base/'absent',
                        expected_receipt={},hash_workers=bad)
        self.assertFalse((self.base/'owned').exists())
        for workers in (1,2,3,4):
            self.assertEqual(self.budget(workers=workers,name=str(workers)).hash_workers,workers)

    def test_default_serial_path_does_not_create_executor(self):
        with patch.object(artifacts,'ThreadPoolExecutor',
                          side_effect=AssertionError('Default must stay serial')):
            value,names=self.files(workers=1)
            value.usage();receipt=value.receipt();value.validate_receipt(receipt)
            value.close()
            restored=artifacts.ArtifactBudget.restore(value.root,expected_receipt=receipt)
            self.opened.append(restored)
            self.assertEqual(restored.hash_workers,1)
            restored.usage()
        self.assertEqual(len(names),4)

    def test_execution_setting_does_not_change_receipt_journal_or_charges(self):
        value,names=self.files(workers=1)
        receipt=value.receipt();usage=value.usage();raw=(value.root/artifacts.JOURNAL).read_bytes()
        identity=value.identity;value.close()
        for workers in (4,1):
            restored=artifacts.ArtifactBudget.restore(value.root,
                expected_receipt=receipt,hash_workers=workers)
            self.opened.append(restored)
            self.assertEqual(restored.receipt(),receipt);self.assertEqual(restored.usage(),usage)
            self.assertEqual(restored.identity,identity)
            self.assertEqual((value.root/artifacts.JOURNAL).read_bytes(),raw)
            with self.assertRaises(AttributeError):restored.hash_workers=2
            restored.close()
        self.assertNotIn('hash_workers',receipt);self.assertNotIn('hash_workers',identity)

    def test_every_inventory_rehashes_all_sealed_pathnames_without_cache(self):
        value,names=self.files()
        receipt=value.receipt();calls=[];digest=value._digest_file
        def observed(name):calls.append(name);return digest(name)
        with patch.object(value,'_digest_file',side_effect=observed):
            for operation in (value.usage,value.usage,value.receipt,
                              lambda:value.validate_receipt(receipt)):
                before=len(calls);operation()
                self.assertCountEqual(calls[before:],names)
        self.assertEqual(len(calls),4*len(names))

    def test_complete_sha_and_one_mib_read_bound(self):
        data=bytes(range(256))*4096+b'final31bytes-must-also-be-hashed!'
        value=self.budget(capacity=len(data)*4)
        names=[f'chunked-{index}' for index in range(4)]
        reservation=value.reserve_bundle('multichunk',{name:len(data) for name in names})
        for name in names:
            with reservation.writer(name) as handle:handle.write(data)
        calls=[];read=artifacts.os.read
        def observed(fd,size):calls.append(size);return read(fd,size)
        with patch.object(artifacts.os,'read',side_effect=observed):value.usage()
        self.assertTrue(calls);self.assertLessEqual(max(calls),1048576)
        self.assertGreater(sum(size==1048576 for size in calls),3)
        for name in names:self.assertEqual(value.files[name]['sha256'],hashlib.sha256(data).hexdigest())
        with (value.root/names[-1]).open('r+b') as handle:
            handle.seek(-1,2);handle.write(b'?')
        with self.assertRaisesRegex(ValueError,'Sealed artifact bytes changed'):value.usage()

    def test_at_most_four_outstanding_jobs_and_actual_overlap(self):
        value,names=self.files(count=8);release=threading.Event();four=threading.Event()
        lock=threading.Lock();counts=dict(submitted=0,outstanding=0,maximum=0,active=0,peak_active=0)
        errors=[];result=[];digest=value._digest_file
        class ObservedFuture:
            def __init__(self,future):self.future=future
            def result(self):
                try:return self.future.result()
                finally:
                    with lock:counts['outstanding']-=1
        class ObservedExecutor(ThreadPoolExecutor):
            def submit(self,*args,**kwargs):
                with lock:
                    counts['submitted']+=1;counts['outstanding']+=1
                    counts['maximum']=max(counts['maximum'],counts['outstanding'])
                return ObservedFuture(super().submit(*args,**kwargs))
        def blocked(name):
            with lock:
                counts['active']+=1;counts['peak_active']=max(counts['peak_active'],counts['active'])
                if counts['active']==4:four.set()
            try:
                if not release.wait(3):raise AssertionError('Digest test release timed out')
                return digest(name)
            finally:
                with lock:counts['active']-=1
        def inspect():
            try:result.append(value.usage())
            except BaseException as error:errors.append(error)
        with patch.object(artifacts,'ThreadPoolExecutor',ObservedExecutor),\
             patch.object(value,'_digest_file',side_effect=blocked):
            thread=threading.Thread(target=inspect);thread.start()
            try:
                self.assertTrue(four.wait(3));self.assertEqual(counts['submitted'],4)
                self.assertEqual(counts['outstanding'],4)
            finally:release.set();thread.join(3)
        self.assertFalse(thread.is_alive());self.assertEqual(errors,[]);self.assertEqual(len(result),1)
        self.assertEqual(counts['submitted'],len(names));self.assertEqual(counts['maximum'],4)
        self.assertEqual(counts['peak_active'],4);self.assertEqual(counts['outstanding'],0)
        self.assertEqual(counts['active'],0)

    def test_errors_follow_original_inventory_order_not_completion_order(self):
        value,names=self.files(count=2);later_done=threading.Event()
        def fail(name):
            if name==names[1]:later_done.set();raise OSError('later-worker-error')
            if not later_done.wait(3):raise AssertionError('Later worker did not run')
            raise OSError('earlier-worker-error')
        with patch.object(artifacts.os,'listdir',return_value=[artifacts.JOURNAL,*names]),\
             patch.object(value,'_digest_file',side_effect=fail):
            with self.assertRaisesRegex(OSError,'earlier-worker-error'):value.usage()

    def test_earlier_digest_failure_precedes_later_metadata_failure(self):
        value,names=self.files(count=2)
        (value.root/'unknown').write_bytes(b'x');os.chmod(value.root/'unknown',0o600)
        for order,expected in (([names[0],'unknown',names[1]],'Sealed artifact bytes changed'),
                               (['unknown',*names],'Untracked artifact entry')):
            with self.subTest(order=order),\
                 patch.object(artifacts.os,'listdir',return_value=[artifacts.JOURNAL,*order]),\
                 patch.object(value,'_digest_file',return_value='0'*64):
                with self.assertRaisesRegex(ValueError,expected):value.usage()

    def test_metadata_failure_precedes_same_entry_worker_failure(self):
        value,names=self.files(count=2)
        (value.root/names[0]).write_bytes(b'over-reserved-cap')
        with patch.object(artifacts.os,'listdir',return_value=[artifacts.JOURNAL,*names]),\
             patch.object(value,'_digest_file',side_effect=OSError('worker-failure')):
            with self.assertRaisesRegex(ValueError,'exceeds its reserved capacity'):value.usage()

    def test_workers_are_drained_and_digest_fds_closed_before_failure_returns(self):
        value,names=self.files();lock=threading.Lock();release=threading.Event()
        all_open=threading.Event();returned=threading.Event();fds={};visited=set();errors=[]
        maximum=[0];open_file=artifacts.os.open;close_file=artifacts.os.close;read=artifacts.os.read
        def observed_open(name,*args,**kwargs):
            fd=open_file(name,*args,**kwargs)
            if name in names:
                with lock:
                    fds[fd]=name;maximum[0]=max(maximum[0],len(fds))
            return fd
        def observed_close(fd):
            try:return close_file(fd)
            finally:
                with lock:fds.pop(fd,None)
        def blocked_read(fd,size):
            with lock:
                name=fds.get(fd);first=name is not None and name not in visited
                if first:
                    visited.add(name)
                    if len(visited)==4:all_open.set()
            if first:
                if not all_open.wait(3):raise AssertionError('Four file FDs did not open')
                if name==names[0]:raise OSError('authored read failure')
                if not release.wait(3):raise AssertionError('Worker drain release timed out')
            return read(fd,size)
        def inspect():
            try:value.usage()
            except BaseException as error:errors.append(error)
            finally:returned.set()
        with patch.object(artifacts.os,'open',side_effect=observed_open),\
             patch.object(artifacts.os,'close',side_effect=observed_close),\
             patch.object(artifacts.os,'read',side_effect=blocked_read),\
             patch.object(artifacts.os,'listdir',return_value=[artifacts.JOURNAL,*names]):
            thread=threading.Thread(target=inspect);thread.start()
            try:
                self.assertTrue(all_open.wait(3));self.assertFalse(returned.wait(.02))
            finally:release.set();thread.join(3)
        self.assertFalse(thread.is_alive());self.assertTrue(returned.is_set())
        self.assertEqual(maximum[0],4);self.assertEqual(fds,{})
        self.assertEqual(len(errors),1);self.assertIsInstance(errors[0],OSError)
        self.assertIn('authored read failure',str(errors[0]))

    def test_existing_anchored_digest_rejects_post_stat_swaps(self):
        for fault in ('symlink','fifo','permissions','growth'):
            with self.subTest(fault=fault):
                value=self.budget(name=fault);reservation=value.reserve_bundle('file',{'file':8})
                with reservation.writer('file') as handle:handle.write(b'original')
                ready=threading.Event();digest=value._digest_file;inspect=value._stat
                def blocked(name):
                    if not ready.wait(3):raise AssertionError('Post-stat mutation did not run')
                    return digest(name)
                def mutate(name):
                    info=inspect(name)
                    if name=='file' and not ready.is_set():
                        if fault in ('symlink','fifo'):
                            os.rename(value.root/'file',self.base/(fault+'-retained'))
                            if fault=='symlink':(value.root/'file').symlink_to(self.base/(fault+'-retained'))
                            else:os.mkfifo(value.root/'file',0o600)
                        elif fault=='permissions':os.chmod(value.root/'file',0o644)
                        else:(value.root/'file').write_bytes(b'over-cap-growth')
                        ready.set()
                    return info
                with patch.object(value,'_digest_file',side_effect=blocked),\
                     patch.object(value,'_stat',side_effect=mutate):
                    if fault=='symlink':
                        with self.assertRaises(OSError) as caught:value.usage()
                        self.assertEqual(caught.exception.errno,errno.ELOOP)
                    else:
                        with self.assertRaisesRegex(ValueError,'Unsafe or over-cap'):value.usage()

    def test_digest_growth_during_stream_keeps_cap_check_and_closes_fd(self):
        value,names=self.files(count=1);read=artifacts.os.read;grew=[False];calls=[]
        ready=threading.Event();inspect=value._stat
        def inspected(name):
            info=inspect(name)
            if name==names[0]:ready.set()
            return info
        def growing(fd,size):
            if not ready.wait(3):raise AssertionError('Main-thread stat did not run')
            chunk=read(fd,size);calls.append(size)
            if chunk and not grew[0]:
                grew[0]=True
                with (value.root/names[0]).open('ab') as handle:handle.write(b'x')
            return chunk
        with patch.object(artifacts.os,'read',side_effect=growing),\
             patch.object(value,'_stat',side_effect=inspected):
            with self.assertRaisesRegex(ValueError,'grew beyond its reserved capacity'):value.usage()
        self.assertTrue(grew[0]);self.assertEqual(calls,[9,1])
        self.assertEqual((value.root/names[0]).read_bytes(),b'\0'*8+b'x')

    def test_missing_sealed_and_disappeared_inventory_entries_are_still_refused(self):
        value,names=self.files();os.unlink(value.root/names[-1]);calls=[];digest=value._digest_file
        def observed(name):calls.append(name);return digest(name)
        with patch.object(value,'_digest_file',side_effect=observed):
            with self.assertRaisesRegex(ValueError,'Sealed artifact is missing'):value.usage()
        self.assertCountEqual(calls,names[:-1])
        inspect=value._stat
        def vanished(name):return None if name==names[0] else inspect(name)
        with patch.object(value,'_stat',side_effect=vanished):
            with self.assertRaisesRegex(ValueError,'inventory changed during inspection'):value.usage()

    def test_removing_files_are_rehashed_and_absent_release_retains_semantics(self):
        value,names=self.files(count=1)
        value._event(dict(kind='remove',name=names[0]));calls=[];digest=value._digest_file
        def observed(name):calls.append(name);return digest(name)
        with patch.object(value,'_digest_file',side_effect=observed):value.usage()
        self.assertEqual(calls,names)
        before=value.usage()['reserved_pathname_bytes'];os.unlink(value.root/names[0])
        self.assertEqual(value.usage()['reserved_pathname_bytes'],before)
        reservation=artifacts.Reservation(value,'authored-files',tuple(names))
        reservation.release_absent(names[0])
        self.assertEqual(value.usage()['reserved_pathname_bytes'],before-8)

    def test_transient_linked_file_keeps_source_inode_identity_gate(self):
        value=self.budget();reservation=value.reserve_bundle('linked',{'source':8,'target':8})
        with reservation.writer('source') as handle:handle.write(b'original')
        value._event(dict(kind='link',source='source',target='target'))
        (value.root/'target').write_bytes(b'original');os.chmod(value.root/'target',0o600)
        with self.assertRaisesRegex(ValueError,'Published artifact link identity changed'):value.usage()

    def test_hardlinked_pathnames_each_receive_their_original_sha_check(self):
        value=self.budget();reservation=value.reserve_bundle('link',{'stage':8,'marker':8})
        with reservation.writer('stage') as handle:handle.write(b'original')
        reservation.link('stage','marker');calls=[];digest=value._digest_file
        def observed(name):calls.append(name);return digest(name)
        with patch.object(value,'_digest_file',side_effect=observed):value.usage()
        self.assertCountEqual(calls,['stage','marker'])
        reservation.remove_staging('stage')
        with patch.object(value,'_digest_file',side_effect=observed):value.usage()
        self.assertEqual(calls.count('stage'),1);self.assertEqual(calls.count('marker'),2)


if __name__=='__main__':unittest.main()
