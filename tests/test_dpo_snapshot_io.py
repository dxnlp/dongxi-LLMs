"""Actual DPO snapshot I/O admission and unchanged checkpoint-on CPU replay."""
from copy import deepcopy
from contextlib import ExitStack
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
import test_dpo_checkpoint_mode as mode
import test_dpo_runner_recovery as base
import test_dpo_validation_budget as prior
from dongxi_llms import training_snapshot as shared
from dongxi_llms.artifact_budget import ArtifactBudget
from dongxi_llms.run_identity import canonical_hash
from dongxi_llms.snapshot_io_budget import (IO_KEYS, SnapshotIOBudget,io_budget_contract,
    io_ledger_contract_sha256,read_bounded_json,read_work_receipt)
from dongxi_llms.work_budget import WorkLedger,WorkBudgetExceeded

r=base.runner;ROOT=base.ROOT
SPEC='experiments/specs/2026-10-05-dpo-snapshot-io.md'
ENVELOPE=dict(max_payload_bytes=base.LIMIT,max_tree_nodes=10000,max_tensor_elements=50000,
    max_tensor_bytes=2000000,max_primitive_bytes=200000)
IO_CAPS=dict(snapshot_inspect_operations=12,snapshot_load_operations=24,snapshot_save_operations=20,
    snapshot_hash_bytes=64*base.LIMIT,snapshot_tree_nodes=560000,snapshot_tensor_elements=2000000,
    snapshot_primitive_bytes=11200000,snapshot_clone_bytes=40000000,
    snapshot_serialization_bytes=20*base.LIMIT)
ARTIFACT_CAPS=dict(max_bytes=64*1024**2,max_entries=64,journal_max_bytes=1024**2,payload_max_bytes=base.LIMIT)
SOURCES=tuple(dict.fromkeys((*prior.SOURCES,'src/dongxi_llms/snapshot_io_budget.py',
    'tests/test_dpo_snapshot_io.py','tests/test_snapshot_io_budget.py',SPEC,'tests/snapshot_io_test_support.py',
    'experiments/specs/2026-10-05-snapshot-io-admission.md',
    'experiments/specs/2026-10-05-snapshot-io-cli-fixture-compatibility.md')))


def fixture(limits=IO_CAPS,*,artifacts=False):
    values=list(prior.fixture());options=deepcopy(values[5])
    options['sources']={name:r.file_digest(ROOT/name) for name in SOURCES}
    options['snapshot_io_budget']=io_budget_contract(limits,ENVELOPE,prior.BOUND)
    if artifacts:options['snapshot_artifact_budget']=r.snapshot_artifact_contract(**ARTIFACT_CAPS)
    values[4]=r.make_recovery_contract(model=values[0],reference=values[1],optimizer=values[2],**options)
    values[5]=options
    return values


def create(directory,values,*,artifacts=False):
    directory=Path(directory);directory.mkdir(mode=0o700)
    science=canonical_hash(values[4]);budget=values[4]['work_budget'];io=values[4]['snapshot_io_budget']
    work=WorkLedger.create(directory/'model.jsonl',limits=budget['limits'],contract_sha256=science,
        max_bytes=budget['max_bytes'],invocation_id='authored-dpo-io')
    journal=WorkLedger.create(directory/'snapshot.jsonl',limits=io['limits'],
        contract_sha256=io_ledger_contract_sha256(io,science),max_bytes=io['max_journal_bytes'],
        invocation_id='authored-dpo-io')
    hook=SnapshotIOBudget(journal,contract=io,scientific_contract_sha256=science)
    artifact=None
    if artifacts:
        artifact=ArtifactBudget.create(directory/'artifacts',campaign_id=science,
            **{key:ARTIFACT_CAPS[key] for key in ('max_bytes','max_entries','journal_max_bytes')})
    return work,journal,hook,artifact


def train(directory,values,work,hook,artifact=None,**initial):
    Path(directory).mkdir(mode=0o700)
    return r.train_completed_updates(*values[:4],base.ENCODED,contract=values[4],output=directory,
        invocation_id='authored-dpo-io',pad_id=1,device='cpu',checkpoint_every=3,
        max_bytes=base.LIMIT,work_ledger=work,snapshot_io_budget=hook,
        snapshot_artifact_budget=artifact,**initial)


def load(receipt,values,work,hook,artifact=None):
    independent=read_work_receipt(receipt['work_receipt_path']);hook.bind_receipt(independent)
    return r.load_snapshot(receipt['path'],expected_sha256=receipt['payload_sha256'],
        expected_bytes=receipt['payload_bytes'],expected_contract=values[4],max_bytes=base.LIMIT,
        io_budget=hook,validate_payload=lambda value:r.validate_dpo_payload(value,values[4],work,artifact))


def inspect(receipt,values,hook):
    hook.bind_receipt(read_work_receipt(receipt['work_receipt_path']))
    return r.inspect_snapshot(receipt['path'],expected_sha256=receipt['payload_sha256'],
        expected_bytes=receipt['payload_bytes'],expected_contract=values[4],max_bytes=base.LIMIT,io_budget=hook)


def receipt_for(path):
    return dict(path=str(Path(path).resolve()),work_receipt_path=str(Path(str(path)+'.work.json').resolve()),
        **read_work_receipt(Path(str(path)+'.work.json'))['snapshot'])


def numeric(payload):
    return prior.numeric(payload)


def interrupt(directory,values,work,hook,artifact=None):
    original=r.append_metric
    def fail(path,row):
        if row.get('update')==3:raise OSError('authored metric failure after accounted update3')
        return original(path,row)
    try:
        with patch.object(r,'append_metric',side_effect=fail):train(directory,values,work,hook,artifact)
    except OSError as error:
        root=artifact.root if artifact is not None else Path(directory)/'checkpoints'
        paths=list(root.glob('*completed-000003.pt'))
        if len(paths)!=1:raise AssertionError('Expected unique committed-three snapshot')
        return receipt_for(paths[0]),dict(type=type(error).__name__,message=str(error),injected=True)
    raise AssertionError('Predeclared interruption absent')


def later_failures(receipt,values,work,hook,artifact=None):
    # The bounded guard fails inside an admitted shared load; no failed attempt
    # is mistaken for successful deserialization or refunded on resume.
    hook.bind_receipt(read_work_receipt(receipt['work_receipt_path']))
    before=hook.ledger.snapshot()
    def fail():raise OSError('authored later shared-load guard failure')
    try:r.load_snapshot(receipt['path'],expected_sha256=receipt['payload_sha256'],
        expected_bytes=receipt['payload_bytes'],expected_contract=values[4],max_bytes=base.LIMIT,
        io_budget=hook,guard=fail)
    except OSError as error:io_error=dict(type=type(error).__name__,message=str(error))
    else:raise AssertionError('Declared shared-load failure absent')
    retained=load(receipt,values,work,hook,artifact)
    bad=deepcopy(retained);bad['state']['sampler_rng']=bad['state']['sampler_rng'].float()
    work_before=work.snapshot()
    try:r.validate_dpo_payload(bad,values[4],work,artifact)
    except ValueError as error:model_error=dict(type=type(error).__name__,message=str(error))
    else:raise AssertionError('Declared semantic failure absent')
    if artifact is not None:
        reservation=artifact.reserve_bundle('later-authored-partial',{'later-partial.bin':256})
        try:
            with reservation.writer('later-partial.bin') as handle:
                handle.write(b'partial!');raise OSError('authored eight-byte partial failure')
        except OSError:pass
    return dict(io_error=io_error,io_before=before,io_after=hook.ledger.snapshot(),
        model_error=model_error,model_before=work_before,model_after=work.snapshot())


def fresh_child(bundle_file,output):
    bundle,_=read_bounded_json(bundle_file);contract,_=read_bounded_json(bundle['contract_path'])
    receipt=read_work_receipt(bundle['snapshot']['work_receipt_path'])
    work,journal,hook=r.open_resume_budgets(contract=contract,budget=contract['work_budget'],
        io_contract=contract['snapshot_io_budget'],receipt=receipt,work_path=bundle['work_path'],
        io_path=bundle['io_path'],invocation_id='fresh-authored-dpo-io',
        expected_sha256=bundle['snapshot']['payload_sha256'],expected_bytes=bundle['snapshot']['payload_bytes'])
    artifact=None
    try:
        # This actual inspection happens before constructing even the random
        # policy/reference. Artifact inventory remains excluded and later.
        inspected=r.inspect_snapshot(bundle['snapshot']['path'],expected_sha256=receipt['snapshot']['payload_sha256'],
            expected_bytes=receipt['snapshot']['payload_bytes'],expected_contract=contract,
            max_bytes=base.LIMIT,io_budget=hook)
        values=fixture(artifacts=bundle['artifact_root'] is not None)
        if values[4]!=contract:raise ValueError('Actual fresh model/source/recipe differs')
        if bundle['artifact_root'] is not None:
            artifact=ArtifactBudget.restore(bundle['artifact_root'],expected_receipt=bundle['artifact_receipt'])
        retained=load(bundle['snapshot'],values,work,hook,artifact)
        after_callback=dict(model=work.snapshot(),io=journal.snapshot())
        completed,counters,history=r.restore_dpo_payload(retained,contract=contract,model=values[0],reference=values[1],
            optimizer=values[2],sampler=values[3],work_ledger=work,snapshot_artifact_budget=artifact)
        after_restore=dict(model=work.snapshot(),io=journal.snapshot())
        result=train(output,values,work,hook,artifact,completed=completed,counters=counters,
            history=history,parent_checkpoint=bundle['snapshot'])
        final=load(result['checkpoint'],values,work,hook,artifact)
        answer=dict(sha256=base.digest(numeric(final)),restored_update=completed,completed_updates=result['completed'],
            checkpointing=bool(values[0].is_gradient_checkpointing),next_history_row=result['history'][3],
            inspected_before_model=inspected,after_callback=after_callback,after_restore=after_restore,
            final_model_work=work.snapshot(),final_io_work=journal.snapshot(),checkpoint=result['checkpoint'],
            partial_bytes=(artifact.root/'later-partial.bin').stat().st_size if artifact is not None else None,
            partial_charge=artifact.files['later-partial.bin']['limit'] if artifact is not None else None,
            artifact_receipt=artifact.receipt() if artifact is not None else None,
            artifact_usage=artifact.usage() if artifact is not None else None)
        r.write_exclusive_json(Path(output)/'fresh-result.json',answer)
        return answer
    finally:
        if artifact is not None:artifact.close()
        journal.close();work.close()


class DPOSnapshotIOTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='dongxi-dpo-io-')
        self.addCleanup(temporary.cleanup);self.root=Path(temporary.name)

    def resources(self,values,name='ledgers',*,artifacts=False):
        result=create(self.root/name,values,artifacts=artifacts)
        for item in (result[0],result[1],result[3]):
            if item is not None:self.addCleanup(item.close)
        return result

    def test_original19_unchanged_and_optional_schema1_still_legitimate(self):
        self.assertEqual(tuple(r.BUDGET_KEYS),tuple(prior.r.BUDGET_KEYS));self.assertEqual(len(r.BUDGET_KEYS),19)
        (self.root/'legacy').mkdir()
        values=mode.make_mode(True);result=base.train_fixture(self.root/'legacy',values)
        old=base.payload_for(result['checkpoint'],values[4]);self.assertEqual(old['schema_version'],1)
        accounted=fixture();self.assertEqual(accounted[4]['work_budget'],r.work_budget_contract(prior.CAPS,prior.BOUND))
        self.assertEqual(len(accounted[4]['snapshot_io_budget']['limits']),9)

    def test_accounted_science_cannot_omit_hook_or_change_contract(self):
        values=fixture();work,journal,hook,_=self.resources(values)
        with self.assertRaises(ValueError):train(self.root/'no-hook',values,work,None)
        wrong=deepcopy(values[4]);wrong['snapshot_io_budget']['limits']['snapshot_save_operations']+=1
        with self.assertRaises(ValueError):r.train_completed_updates(*values[:4],base.ENCODED,contract=wrong,
            output=self.root,invocation_id='refused',pad_id=1,device='cpu',checkpoint_every=3,
            max_bytes=base.LIMIT,work_ledger=work,snapshot_io_budget=hook)
        self.assertEqual(journal.snapshot()['reserved']['snapshot_save_operations'],0)

    def test_all_three_saves_actual_parity_and_receipt_prefixes(self):
        (self.root/'baseline').mkdir()
        original=mode.make_mode(True);baseline=base.train_fixture(self.root/'baseline',original)
        expected=base.payload_for(baseline['checkpoint'],original[4])
        values=fixture();work,journal,hook,_=self.resources(values)
        result=train(self.root/'accounted',values,work,hook);retained=load(result['checkpoint'],values,work,hook)
        self.assertEqual(base.digest(numeric(retained)),base.digest(prior.numeric(expected)))
        self.assertEqual(retained['schema_version'],2);self.assertEqual(result['history'],baseline['history'])
        self.assertEqual(journal.snapshot()['completed']['snapshot_save_operations'],3)
        independent=read_work_receipt(result['checkpoint']['work_receipt_path'])
        self.assertEqual(independent['runner_work_prefix'],retained['state']['work_ledger'])
        self.assertEqual(independent['io_prefix'],retained['snapshot_io']['io_prefix'])
        self.assertEqual(independent['io_prefix']['reserved']['snapshot_save_operations'],2)
        self.assertEqual(work.snapshot()['reserved']['train_updates'],6)
        self.assertEqual(work.snapshot()['reserved']['sampler_draws'],12)

    def test_repeat_inspect_load_each_charged_without_semantic_call_removal(self):
        values=fixture();work,journal,hook,_=self.resources(values)
        result=train(self.root/'trained',values,work,hook)
        for _ in range(2):inspect(result['checkpoint'],values,hook);load(result['checkpoint'],values,work,hook)
        self.assertEqual(journal.snapshot()['completed']['snapshot_inspect_operations'],2)
        self.assertEqual(journal.snapshot()['completed']['snapshot_load_operations'],2)
        self.assertEqual(work.snapshot()['reserved']['recovery_validation_operations'],5)

    def test_exhausted_save_before_shared_tree_clone_or_serializer(self):
        values=fixture(dict(IO_CAPS,snapshot_save_operations=0));work,journal,hook,_=self.resources(values)
        with patch.object(shared,'_safe_tree') as walked,patch.object(torch,'save') as serialized,self.assertRaises(WorkBudgetExceeded):
            train(self.root/'refused',values,work,hook)
        walked.assert_not_called();serialized.assert_not_called()
        self.assertEqual(work.snapshot()['reserved']['recovery_validation_operations'],1)
        self.assertEqual(journal.snapshot()['reserved']['snapshot_save_operations'],0)

    def test_exhausted_reads_before_payload_or_callback(self):
        for operation in ('inspect','load'):
            with self.subTest(operation=operation):
                values=fixture(dict(IO_CAPS,**{'snapshot_'+operation+'_operations':0}))
                work,journal,hook,_=self.resources(values,operation)
                result=train(self.root/(operation+'-train'),values,work,hook)
                hook.bind_receipt(read_work_receipt(result['checkpoint']['work_receipt_path']))
                with (patch.object(shared,'_verified_bytes') as read,patch.object(torch,'load') as loaded,
                      patch.object(r,'validate_dpo_payload') as semantic,self.assertRaises(WorkBudgetExceeded)):
                    (inspect if operation=='inspect' else load)(result['checkpoint'],values,*((hook,) if operation=='inspect' else (work,hook)))
                read.assert_not_called();loaded.assert_not_called();semantic.assert_not_called()

    def test_zero_tensor_envelope_refuses_before_saved_tensor_finite(self):
        values=fixture();io=values[5]['snapshot_io_budget'];io['envelope']['max_tensor_elements']=0
        values[4]=r.make_recovery_contract(model=values[0],reference=values[1],optimizer=values[2],**values[5])
        work,journal,hook,_=self.resources(values)
        with patch.object(torch,'isfinite',wraps=torch.isfinite) as finite,self.assertRaises(WorkBudgetExceeded):
            train(self.root/'tensor-refused',values,work,hook)
        # Runner semantic validation is separately admitted and legitimately
        # scans before save. The shared save must reject before its first scan.
        self.assertGreater(finite.call_count,0)
        self.assertTrue(journal.snapshot()['failed_tickets'])
        self.assertEqual(journal.snapshot()['completed']['snapshot_clone_bytes'],0)

    def test_independent_expectations_refuse_before_byte_verification(self):
        values=fixture();work,journal,hook,_=self.resources(values)
        result=train(self.root/'trained',values,work,hook);receipt=read_work_receipt(result['checkpoint']['work_receipt_path'])
        hook.bind_receipt(receipt)
        with patch.object(shared,'_verified_bytes') as read,self.assertRaises(ValueError):
            r.inspect_snapshot(result['checkpoint']['path'],expected_sha256='b'*64,
                expected_bytes=receipt['snapshot']['payload_bytes'],expected_contract=values[4],
                max_bytes=base.LIMIT,io_budget=hook)
        read.assert_not_called()

    def test_copied_or_changed_journal_and_receipt_before_payload(self):
        values=fixture();work,journal,hook,_=self.resources(values)
        result=train(self.root/'trained',values,work,hook);receipt=read_work_receipt(result['checkpoint']['work_receipt_path'])
        workpath,journalpath=work.path,journal.path;work.close();journal.close()
        for which in ('copy','receipt'):
            with self.subTest(which=which):
                expected=deepcopy(receipt);path=journalpath
                if which=='copy':path=self.root/'ledgers/copied.jsonl';shutil.copyfile(journalpath,path);path.chmod(0o600)
                else:expected['runner_work_prefix']['chain_sha256']='b'*64
                with patch.object(shared,'_verified_bytes') as read,self.assertRaises((ValueError,RuntimeError)):
                    r.open_resume_budgets(contract=values[4],budget=values[4]['work_budget'],
                        io_contract=values[4]['snapshot_io_budget'],receipt=expected,work_path=workpath,io_path=path,
                        invocation_id='refused',expected_sha256=receipt['snapshot']['payload_sha256'],
                        expected_bytes=receipt['snapshot']['payload_bytes'])
                read.assert_not_called()

    def test_pending_or_beyond_endpoint_receipt_before_journal_or_payload(self):
        values=fixture();work,journal,hook,_=self.resources(values)
        result=train(self.root/'trained',values,work,hook);receipt=read_work_receipt(result['checkpoint']['work_receipt_path'])
        for fields in ({'phase':'pending'},{'completed_updates':7}):
            with self.subTest(fields=fields):
                wrong=deepcopy(receipt);wrong['snapshot'].update(fields)
                with patch.object(WorkLedger,'open') as opened,patch.object(r,'inspect_snapshot') as read,self.assertRaises(ValueError):
                    r.open_resume_budgets(contract=values[4],budget=values[4]['work_budget'],
                        io_contract=values[4]['snapshot_io_budget'],receipt=wrong,work_path=work.path,io_path=journal.path,
                        invocation_id='refused',expected_sha256=receipt['snapshot']['payload_sha256'],
                        expected_bytes=receipt['snapshot']['payload_bytes'])
                opened.assert_not_called();read.assert_not_called()

    def test_old_or_changed_budget_no_resume_normalization(self):
        values=fixture();work,journal,hook,_=self.resources(values)
        result=train(self.root/'trained',values,work,hook);receipt=read_work_receipt(result['checkpoint']['work_receipt_path'])
        for budget in (dict(values[4]['work_budget'],schema='dongxi-dpo-logical-work-v1'),
                       dict(values[4]['work_budget'],scope='changed'),[],None):
            with self.subTest(budget=budget),patch.object(WorkLedger,'open') as opened,self.assertRaises(ValueError):
                r.open_resume_budgets(contract=values[4],budget=budget,io_contract=values[4]['snapshot_io_budget'],
                    receipt=receipt,work_path=work.path,io_path=journal.path,invocation_id='refused',
                    expected_sha256=receipt['snapshot']['payload_sha256'],expected_bytes=receipt['snapshot']['payload_bytes'])
            opened.assert_not_called()

    def cli_fixture(self,*,exhausted):
        """Mock platform/interface observers only; actual journals/shared inspect.

        This is a source-order spy, not a production receipt/interface proof.
        No real CUDA or parent weights are loaded.
        """
        values=fixture(dict(IO_CAPS,snapshot_inspect_operations=0) if exhausted else IO_CAPS,artifacts=True)
        work,journal,hook,artifact=self.resources(values,artifacts=True)
        result=train(self.root/'trained',values,work,hook,artifact);receipt=result['checkpoint']
        artifact_receipt=artifact.receipt();workpath,iopath,artifactroot=work.path,journal.path,artifact.root
        work.close();journal.close();artifact.close()
        parent=self.root/'placeholder-parent';parent.mkdir();(parent/'config.json').write_text('{}')
        (parent/'model.safetensors').write_bytes(b'authored non-model placeholder; never loaded')
        paths={}
        for name,word in (('train','red'),('validation','blue'),('evaluation','cup')):
            path=self.root/(name+'.jsonl');paths[name]=path
            row=dict(id=name,group=name,prompt=[{'role':'user','content':word}],chosen='good',rejected='bad',expected='yes')
            path.write_text(json.dumps(row)+'\n')
        for name,value in (('limits',prior.CAPS),('io-limits',values[4]['snapshot_io_budget']),
                           ('contract',values[4]),('artifact-receipt',artifact_receipt)):
            path=self.root/(name+'.json');r.write_exclusive_json(path,value);paths[name]=path
        output=self.root/'cli-output'
        argv=['--checkpoint',str(parent),'--tokenizer','authored-local-placeholder','--tokenizer-revision','a'*40,
            '--train',str(paths['train']),'--validation',str(paths['validation']),'--evaluation',str(paths['evaluation']),
            '--output',str(output),'--updates','6','--accumulation','2','--beta','.2','--lr','.008',
            '--max-length','32','--max-new-tokens','4','--checkpoint-every','3',
            '--snapshot-max-bytes',str(base.LIMIT),'--work-limits',str(paths['limits']),
            '--work-journal-max-bytes',str(prior.BOUND),'--work-journal',str(workpath),
            '--snapshot-artifact-max-bytes',str(ARTIFACT_CAPS['max_bytes']),
            '--snapshot-artifact-max-entries','64','--snapshot-artifact-journal-max-bytes',str(prior.BOUND),
            '--snapshot-artifact-root',str(artifactroot),'--snapshot-artifact-receipt',str(paths['artifact-receipt']),
            '--snapshot-io-limits',str(paths['io-limits']),'--snapshot-io-ledger',str(iopath),
            '--resume-io-receipt',receipt['work_receipt_path'],'--resume',receipt['path'],
            '--resume-sha256',receipt['payload_sha256'],'--resume-bytes',str(receipt['payload_bytes']),
            '--resume-contract',str(paths['contract']),'--environment-lock',str(ROOT/'uv.lock')]
        latent={'policy_shapes','optimizer_shapes','optimizer_parameter_names','optimizer_group',
            'reference_sha256','policy_model','reference_model'}
        observed={key:value for key,value in values[4].items() if key not in latent}
        generic=[]
        def identity(root,**kwargs):
            generic.extend(str(Path(path)) for path in kwargs['input_files'])
            return dict(source_sha256={str(Path(path).relative_to(ROOT)):r.file_digest(path) for path in kwargs['source_files']},
                input_sha256={str(Path(path)):r.file_digest(path) for path in kwargs['input_files']},
                checkpoint_files={},checkpoint_path=None,device=kwargs['device'],gpu_driver={'status':'mocked'},
                environment={'python_version':platform.python_version(),'platform':platform.platform(),
                    'machine':platform.machine(),'packages':{},
                    'environment_lock':{'path':str(ROOT/'uv.lock'),'sha256':r.file_digest(ROOT/'uv.lock')}})
        from transformers import AutoModelForCausalLM,AutoTokenizer
        with ExitStack() as stack:
            stack.enter_context(patch.object(r,'collect_run_identity',side_effect=identity))
            stack.enter_context(patch.object(r,'observable_recovery_contract',return_value=observed))
            stack.enter_context(patch.object(r,'verify_parent_tokenizer',return_value=({'generation_stop_ids':[1]},None)))
            stack.enter_context(patch.object(AutoTokenizer,'from_pretrained',return_value=base.tokenizer_fixture()))
            stack.enter_context(patch.object(torch.cuda,'is_available',return_value=True))
            stack.enter_context(patch.object(torch.cuda,'is_bf16_supported',return_value=True))
            stack.enter_context(patch.object(torch.cuda,'get_device_name',return_value='authored-mock-not-GPU'))
            stack.enter_context(patch.object(r,'host_available',return_value=30*1024**3))
            backend=stack.enter_context(patch.object(AutoModelForCausalLM,'from_pretrained',
                side_effect=RuntimeError('authored stop at model allocation boundary')))
            inventory=stack.enter_context(patch.object(ArtifactBudget,'restore'))
            if exhausted:
                verified=stack.enter_context(patch.object(shared,'_verified_bytes'))
                with self.assertRaises(WorkBudgetExceeded):r.main(argv)
                verified.assert_not_called();backend.assert_not_called()
            else:
                with self.assertRaisesRegex(RuntimeError,'authored stop at model allocation'):r.main(argv)
                self.assertEqual(backend.call_count,1)
            inventory.assert_not_called()
        saved=json.loads((output/'identity.json').read_text())
        self.assertNotIn(receipt['path'],generic);self.assertNotIn(receipt['path']+'.commit.json',generic)
        self.assertNotIn(str(paths['contract']),generic);self.assertNotIn(receipt['work_receipt_path'],generic)
        self.assertNotIn(str(paths['io-limits']),generic);self.assertNotIn(str(paths['limits']),generic)
        self.assertEqual('verified_resume_snapshot' in saved,not exhausted)
        self.assertIn('not actual',saved['expected_resume_snapshot']['boundary'])
        return dict(exhausted=exhausted,model_backend_calls=backend.call_count,
            generic_input_paths=generic,identity=saved,
            scope='mocked platform/interface ordering; actual journals/shared inspection; no real GPU/model allocation')

    def test_cli_zero_cap_before_payload_inventory_or_model_backend(self):
        self.cli_fixture(exhausted=True)

    def test_cli_actual_inspection_precedes_model_and_records_measured_identity(self):
        self.cli_fixture(exhausted=False)

    def test_bootstrap_bound_fifo_duplicates_and_identity_expectation(self):
        bad=self.root/'oversized';bad.write_bytes(b' '*65537)
        fifo=self.root/'fifo';os.mkfifo(fifo,0o600)
        duplicate=self.root/'duplicate';duplicate.write_text('{"a":1,"a":2}')
        for path in (bad,fifo,duplicate):
            with self.subTest(path=path),self.assertRaises(ValueError):read_bounded_json(path)
        identity=r.bind_bounded_metadata({'source_sha256':{},'input_sha256':{}},{},
            expected_snapshot={'payload_sha256':'a'*64,'payload_bytes':100})
        self.assertNotIn('verified_resume_snapshot',identity)
        self.assertIn('not actual',identity['expected_resume_snapshot']['boundary'])

    def test_joint_artifact_receipt_reserved_and_later_partial_preserved(self):
        values=fixture(artifacts=True);work,journal,hook,artifact=self.resources(values,artifacts=True)
        result=train(self.root/'joint',values,work,hook,artifact)
        retained=load(result['checkpoint'],values,work,hook,artifact)
        self.assertEqual(retained['state']['completed'],6)
        self.assertEqual(len(list(artifact.root.glob('*.work.json'))),3)
        self.assertEqual(artifact.usage()['reserved_entries'],10)
        for path in artifact.root.glob('*.work.json'):
            self.assertEqual(artifact.files[path.name]['state'],'sealed')

    def test_fresh_process_three_to_six_preserves_both_failed_journals_and_partial(self):
        answer=exercise_replay(self.root/'replay',artifacts=True)
        self.assertEqual(answer['expected_sha256'],answer['fresh']['sha256'])
        self.assertTrue(answer['fresh']['checkpointing']);self.assertEqual(answer['fresh']['restored_update'],3)
        self.assertEqual(answer['fresh']['partial_bytes'],8);self.assertEqual(answer['fresh']['partial_charge'],256)
        for name,key in (('io','final_io_work'),('model','final_model_work')):
            self.assertTrue(set(answer['failures'][name+'_after']['failed_tickets'])<=set(answer['fresh'][key]['failed_tickets']))
        self.assertEqual(answer['fresh']['final_model_work']['reserved']['train_updates'],6)


def exercise_replay(destination,*,artifacts=True):
    destination=Path(destination);destination.mkdir(mode=0o700)
    (destination/'baseline').mkdir()
    original=mode.make_mode(True);baseline=base.train_fixture(destination/'baseline',original)
    expected=base.payload_for(baseline['checkpoint'],original[4]);expected_sha=base.digest(numeric(expected))
    values=fixture(artifacts=artifacts);work,journal,hook,artifact=create(destination/'ledgers',values,artifacts=artifacts)
    try:
        receipt,error=interrupt(destination/'interrupted',values,work,hook,artifact)
        failed=later_failures(receipt,values,work,hook,artifact)
        bundle=dict(contract_path=str(destination/'contract.json'),snapshot=receipt,work_path=str(work.path),
            io_path=str(journal.path),artifact_root=str(artifact.root) if artifact is not None else None,
            artifact_receipt=artifact.receipt() if artifact is not None else None)
        r.write_exclusive_json(destination/'contract.json',values[4]);r.write_exclusive_json(destination/'bundle.json',bundle)
        r.write_exclusive_json(destination/'failures.json',failed)
    finally:
        if artifact is not None:artifact.close()
        journal.close();work.close()
    command=[sys.executable,str(Path(__file__).resolve()),'--child',str(destination/'bundle.json'),str(destination/'fresh')]
    started=time.monotonic();process=subprocess.run(command,cwd=ROOT,env=dict(os.environ,PYTHONPATH='src:tests',
        CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1'),
        capture_output=True,text=True,timeout=40)
    execution=dict(command=command,exit_code=process.returncode,seconds=time.monotonic()-started,
        deadline_seconds=40,stdout=process.stdout,stderr=process.stderr)
    r.write_exclusive_json(destination/'fresh-execution.json',execution)
    if process.returncode:raise AssertionError('Actual fresh replay failed: '+process.stderr)
    fresh=json.loads((destination/'fresh/fresh-result.json').read_text())
    if fresh['sha256']!=expected_sha or fresh['next_history_row']!=baseline['history'][3]:
        raise AssertionError('Fresh numerical state or next action row differs')
    return dict(expected_sha256=expected_sha,fresh=fresh,failures=failed,interruption=error,execution=execution)


def collect_reference(destination):
    """Exclusive CPU reference; fixed test/replay children, never real launch."""
    from datetime import datetime,timezone
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False,mode=0o700)
    before={name:r.file_digest(ROOT/name) for name in SOURCES};started=time.monotonic()
    result=dict(schema='dongxi-dpo-snapshot-io-reference-v1',status='running',
        created_utc=datetime.now(timezone.utc).isoformat(),command=list(sys.orig_argv),python_argv=list(sys.argv),
        source_sha256=before,commands=[],production_ready=False,launch_authorized=False,model_scale_jobs_started=0,
        environment=dict(python=platform.python_version(),platform=platform.platform(),machine=platform.machine(),
            executable=sys.executable,packages={name:importlib.metadata.version(name) for name in
            ('torch','transformers','tokenizers','peft')},cuda_visible_devices='',hf_offline=True,omp_num_threads=1))
    def save(name,value):r.write_exclusive_json(destination/name,value)
    def raw(name,value):
        path=destination/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as handle:handle.write(value);handle.flush();os.fsync(handle.fileno())
        return dict(path=name,bytes=len(value),sha256=hashlib.sha256(value).hexdigest())
    def directory(name):
        path=destination/name;path.mkdir(mode=0o700);return path
    env=dict(os.environ,PYTHONPATH='src:tests',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1')
    try:
        modules=('test_dpo_runner_recovery','test_dpo_checkpoint_mode','test_dpo_work_budget',
            'test_dpo_snapshot_artifacts','test_dpo_validation_budget','test_training_snapshot',
            'test_training_snapshot_failure_paths','test_snapshot_artifact_budget','test_snapshot_io_budget')
        for label,names in (('new-focused',('test_dpo_snapshot_io',)),('existing-regressions',modules)):
            command=[sys.executable,'-m','unittest',*names,'-v'];measured=time.monotonic()
            process=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,timeout=100)
            output=process.stdout+process.stderr;count=re.search(r'Ran (\d+) tests? in ([\d.]+)s',output)
            result['commands'].append(dict(command=command,exit_code=process.returncode,
                seconds=time.monotonic()-measured,deadline_seconds=100,
                tests_ran=int(count.group(1)) if count else None,
                unittest_seconds=float(count.group(2)) if count else None,raw_log=raw(label+'.log',output.encode())))
        for name in SOURCES:
            path=ROOT/name
            if path.stat().st_size>4*1024**2:raise ValueError('Captured source exceeded explicit file bound')
            value=path.read_bytes()
            if hashlib.sha256(value).hexdigest()!=before[name]:raise ValueError('Source drift before archive capture')
            raw('source-capture/'+name,value)
        save('declared-fixture.json',dict(original=base.FIXTURE,model_caps=prior.CAPS,
            snapshot_io_caps=IO_CAPS,snapshot_io_envelope=ENVELOPE,snapshot_io_journal_bytes=prior.BOUND,
            artifacts=ARTIFACT_CAPS,mode='actual policy checkpointing ON, frozen original reference, cache OFF'))
        on=mode.make_mode(True);on_training=base.train_fixture(directory('unhooked-on'),on)
        on_payload=base.payload_for(on_training['checkpoint'],on[4]);save('unhooked-on-training.json',on_training)
        off=mode.make_mode(False);off_training=base.train_fixture(directory('unhooked-off'),off)
        off_payload=base.payload_for(off_training['checkpoint'],off[4]);save('unhooked-off-training.json',off_training)
        result['checkpoint_off_on_parity']=mode.comparison(off_payload,on_payload)
        values=fixture();work,journal,hook,_=create(destination/'complete-ledgers',values)
        try:
            complete=train(destination/'accounted-complete',values,work,hook)
            before_load=dict(model=work.snapshot(),io=journal.snapshot())
            accounted=load(complete['checkpoint'],values,work,hook)
            after_load=dict(model=work.snapshot(),io=journal.snapshot())
            expected_sha=base.digest(numeric(on_payload));actual_sha=base.digest(numeric(accounted))
            if expected_sha!=actual_sha:raise AssertionError('Shared accounting changed original numerical state')
            result['hooked_unhooked_parity']=dict(exact=True,expected_sha256=expected_sha,actual_sha256=actual_sha,
                checkpointing=True,completed_updates=6)
            save('complete-training.json',complete);save('complete-before-final-load.json',before_load)
            save('complete-after-final-load.json',after_load);save('recovery-contract.json',values[4])
        finally:journal.close();work.close()
        replay=exercise_replay(destination/'fresh-replay',artifacts=True)
        if replay['expected_sha256']!=expected_sha:raise AssertionError('Replay reference changed')
        save('fresh-replay-summary.json',replay)
        result['fresh_process_replay']=dict(exact=True,expected_sha256=expected_sha,
            actual_sha256=replay['fresh']['sha256'],restored_update=3,completed_updates=6,
            next_history_row_exact=True,checkpointing=replay['fresh']['checkpointing'],
            actual_exit_code=replay['execution']['exit_code'],deadline_seconds=40,
            retained_io_failures=replay['fresh']['final_io_work']['failed_tickets'],
            retained_model_failures=replay['fresh']['final_model_work']['failed_tickets'],
            final_io_reserved=replay['fresh']['final_io_work']['reserved'],
            final_model_reserved=replay['fresh']['final_model_work']['reserved'],
            partial_bytes=replay['fresh']['partial_bytes'],partial_charge=replay['fresh']['partial_charge'])
        controls=[]
        for exhausted in (True,False):
            case=DPOSnapshotIOTests('test_cli_zero_cap_before_payload_inventory_or_model_backend')
            case.root=directory('cli-exhausted' if exhausted else 'cli-inspected')
            try:controls.append(case.cli_fixture(exhausted=exhausted))
            finally:case.doCleanups()
        save('cli-ordering-controls.json',controls)
        result['cli_ordering_controls']=[dict(exhausted=value['exhausted'],
            model_backend_calls=value['model_backend_calls'],
            measured_payload_verification='verified_resume_snapshot' in value['identity'],scope=value['scope']) for value in controls]
        result['status']='pass' if all(command['exit_code']==0 for command in result['commands']) else 'fail'
    except BaseException as error:
        result['status']='fail';result['failure']=dict(type=type(error).__name__,message=str(error))
    result['source_sha256_after']={name:r.file_digest(ROOT/name) for name in SOURCES}
    result['sources_unchanged']=before==result['source_sha256_after']
    if not result['sources_unchanged']:result['status']='fail'
    result['seconds']=time.monotonic()-started
    result['limitations']=dict(units='declared shared visits/bytes, not CPU instructions, time or physical quota',
        artifact_inventory_and_caller_capture='postadmission, explicitly excluded from shared snapshot dimensions',
        metadata_journal_source_parent_input_identity='excluded; bounded metadata/anchored journals still enforced',
        cli_ordering='mocked platform/interface observers, actual bound journals/shared inspection; not real Spark acceptance',
        production_pretrained_gpu_mac_hosted='not executed',all45_external_outcomes='null',learner_day=9)
    save('verification.json',result)
    return result


if __name__=='__main__':
    if len(sys.argv)==4 and sys.argv[1]=='--child':fresh_child(*sys.argv[2:])
    elif len(sys.argv)==3 and sys.argv[1]=='--collect':
        result=collect_reference(sys.argv[2]);print(json.dumps({key:result[key] for key in ('status','sources_unchanged','seconds')}))
        raise SystemExit(0 if result['status']=='pass' else 1)
    else:unittest.main()
