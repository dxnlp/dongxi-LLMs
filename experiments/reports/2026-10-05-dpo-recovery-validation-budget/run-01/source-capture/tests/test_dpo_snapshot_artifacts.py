"""Actual DPO snapshot integration; original random CPU fixtures, no acquisition."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
from dongxi_llms.artifact_budget import ArtifactBudget
from dongxi_llms.batched_cache_lab import digest
from dongxi_llms.run_identity import canonical_hash
from dongxi_llms.training_snapshot import load_snapshot
from dongxi_llms.work_budget import WorkLedger
import test_dpo_runner_recovery as base

r=base.runner
ROOT=base.ROOT
CAPS=dict(max_bytes=8*1024**2,max_entries=64,journal_max_bytes=1024**2,payload_max_bytes=1024**2)
SOURCES=['scripts/run_chapter11_spark_dpo.py','src/dongxi_llms/artifact_budget.py',
    'src/dongxi_llms/training_snapshot.py','src/dongxi_llms/work_budget.py',
    'src/dongxi_llms/dpo_lab.py','src/dongxi_llms/batched_cache_lab.py',
    'tests/test_dpo_runner_recovery.py','tests/test_dpo_snapshot_artifacts.py',
    'experiments/specs/2026-10-05-dpo-snapshot-artifact-integration.md',
    'experiments/specs/2026-10-05-dpo-cap-byte-binding.md']


def fixture(caps=None):
    model,reference,optimizer,sampler,_,options=base.build_fixture()
    model.gradient_checkpointing_enable()
    model.config.use_cache=False;reference.config.use_cache=False
    options=dict(options,sources={name:r.file_digest(ROOT/name) for name in SOURCES},
        snapshot_artifact_budget=r.snapshot_artifact_contract(**(caps or CAPS)))
    contract=r.make_recovery_contract(model=model,reference=reference,optimizer=optimizer,**options)
    return model,reference,optimizer,sampler,contract,options


def create(root,values):
    caps=values[4]['snapshot_artifact_budget']
    return ArtifactBudget.create(root,campaign_id=canonical_hash(values[4]),
        **{key:caps[key] for key in ('max_bytes','max_entries','journal_max_bytes')})


def train(output,values,ledger,**initial):
    Path(output).mkdir()
    return r.train_completed_updates(*values[:4],base.ENCODED,contract=values[4],output=output,
        invocation_id='original-artifact-fixture',pad_id=1,device='cpu',checkpoint_every=3,
        max_bytes=values[4]['snapshot_artifact_budget']['payload_max_bytes'],
        snapshot_artifact_budget=ledger,**initial)


def load(receipt,values,ledger,work_ledger=None):
    return load_snapshot(receipt['path'],expected_sha256=receipt['payload_sha256'],
        expected_bytes=receipt['payload_bytes'],expected_contract=values[4],max_bytes=CAPS['payload_max_bytes'],
        validate_payload=lambda value:r.validate_dpo_payload(value,values[4],work_ledger,
            snapshot_artifact_budget=ledger))


def numerical(payload):
    return {key:value for key,value in payload['state'].items()
        if key not in ('resume_parent','snapshot_artifact_ledger')}


def partial(ledger):
    ticket=ledger.reserve_bundle('later-failed-partial',{'later-partial.bin':256})
    try:
        with ticket.writer('later-partial.bin') as handle:
            handle.write(b'partial!')
            raise OSError('declared eight-byte partial writer failure')
    except OSError:pass


def resumed(values,ledger,receipt):
    payload=load(receipt,values,ledger)
    completed,counters,history=r.restore_dpo_payload(payload,contract=values[4],
        model=values[0],reference=values[1],optimizer=values[2],sampler=values[3],
        snapshot_artifact_budget=ledger)
    return dict(completed=completed,counters=counters,history=history,parent_checkpoint=receipt)


def fresh_child(bundle_file,output):
    bundle=json.loads(Path(bundle_file).read_text());values=fixture()
    if values[4]!=bundle['contract']:raise ValueError('Fresh source/science contract mismatch')
    with ArtifactBudget.restore(bundle['root'],expected_receipt=bundle['artifact_receipt']) as ledger:
        before=ledger.usage()
        result=train(output,values,ledger,**resumed(values,ledger,bundle['snapshot']))
        answer=dict(numerical_sha256=digest(numerical(load(result['checkpoint'],values,ledger))),
            restored_update=bundle['snapshot']['completed_updates'],completed=result['completed'],
            before_usage=before,after_usage=ledger.usage(),
            partial_bytes=(ledger.root/'later-partial.bin').stat().st_size,
            partial_charge=ledger.files['later-partial.bin']['limit'],receipt=ledger.receipt())
    r.write_exclusive_json(Path(output)/'child-result.json',answer)
    return answer


class DpoSnapshotArtifactTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory(prefix='dongxi-dpo-snapshot-artifacts-')
        self.addCleanup(self.directory.cleanup);self.root=Path(self.directory.name)

    def ledger(self,values):
        ledger=create(self.root/'artifacts',values);self.addCleanup(ledger.close);return ledger

    def test_actual_loop_numerical_parity_with_original_helper(self):
        original=base.build_fixture();original[0].gradient_checkpointing_enable()
        original[0].config.use_cache=False;original[1].config.use_cache=False
        (self.root/'original').mkdir()
        expected=base.train_fixture(self.root/'original',original)
        expected_payload=base.payload_for(expected['checkpoint'],original[4])
        values=fixture();ledger=self.ledger(values)
        result=train(self.root/'budgeted',values,ledger)
        actual=load(result['checkpoint'],values,ledger)
        self.assertEqual(digest(numerical(actual)),digest(base.numerical_state(expected_payload)))
        self.assertEqual(result['history'],expected['history'])
        self.assertEqual(len(list(ledger.root.glob('*.pt'))),3)
        self.assertEqual(ledger.usage()['reserved_entries'],7)
        self.assertLess(ledger.usage()['reserved_pathname_bytes'],CAPS['max_bytes'])

    def test_same_process_old_prefix_keeps_later_partial_and_parent(self):
        values=fixture();ledger=self.ledger(values)
        result=train(self.root/'first',values,ledger)
        parent=base.checkpoint_receipt(next(ledger.root.glob('*completed-000003.pt')))
        expected=digest(numerical(load(result['checkpoint'],values,ledger)))
        prefix=load(parent,values,ledger)['state']['snapshot_artifact_ledger']
        partial(ledger);before=ledger.usage();old_head=ledger.receipt()
        ledger.validate_receipt(prefix)
        self.assertEqual(ledger.receipt(),old_head);self.assertEqual(ledger.usage(),before)
        restarted=fixture()
        replay=train(self.root/'second',restarted,ledger,**resumed(restarted,ledger,parent))
        self.assertEqual(digest(numerical(load(replay['checkpoint'],restarted,ledger))),expected)
        self.assertEqual((ledger.root/'later-partial.bin').stat().st_size,8)
        self.assertEqual(ledger.files['later-partial.bin']['limit'],256)
        self.assertEqual(r.file_digest(Path(parent['path'])),parent['payload_sha256'])
        self.assertGreater(ledger.usage()['reserved_pathname_bytes'],before['reserved_pathname_bytes'])

    def test_fresh_process_replays_same_physical_root_without_refunds(self):
        values=fixture();ledger=self.ledger(values)
        result=train(self.root/'first',values,ledger)
        expected=digest(numerical(load(result['checkpoint'],values,ledger)))
        parent=base.checkpoint_receipt(next(ledger.root.glob('*completed-000003.pt')))
        prefix=load(parent,values,ledger)['state']['snapshot_artifact_ledger']
        partial(ledger)
        bundle=dict(contract=values[4],snapshot=parent,artifact_receipt=prefix,root=str(ledger.root))
        r.write_exclusive_json(self.root/'bundle.json',bundle);ledger.close()
        process=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--child',
            str(self.root/'bundle.json'),str(self.root/'child')],cwd=ROOT,capture_output=True,text=True,
            timeout=30,env=dict(os.environ,PYTHONPATH=f'{ROOT}/src:{ROOT}/tests',CUDA_VISIBLE_DEVICES='',
                HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1'))
        self.assertEqual(process.returncode,0,process.stdout+process.stderr)
        answer=json.loads((self.root/'child/child-result.json').read_text())
        self.assertEqual(answer['numerical_sha256'],expected)
        self.assertEqual((answer['restored_update'],answer['completed']), (3,6))
        self.assertEqual((answer['partial_bytes'],answer['partial_charge']),(8,256))

    def test_work_and_snapshot_ledgers_retain_independent_later_attempts(self):
        raw=fixture();options=dict(raw[5],work_budget=r.work_budget_contract(
            dict({key:100000 for key in r.BUDGET_KEYS},
                recovery_tensor_elements=2000000),1024**2))
        values=(*raw[:4],r.make_recovery_contract(model=raw[0],reference=raw[1],optimizer=raw[2],**options),options)
        ledger=self.ledger(values)
        work=WorkLedger.create(self.root/'work.jsonl',limits=options['work_budget']['limits'],
            contract_sha256=canonical_hash(values[4]),max_bytes=1024**2,invocation_id='joint-source-fixture')
        self.addCleanup(work.close)
        result=train(self.root/'joint-first',values,ledger,work_ledger=work)
        parent=base.checkpoint_receipt(next(ledger.root.glob('*completed-000003.pt')))
        payload=load(parent,values,ledger,work)
        ticket=work.reserve({'train_updates':1},operation='later-failed-update-allowance')
        work.fail(ticket,'declared later failure',known_actual={},attempted={})
        partial(ledger)
        restarted=fixture();restored_model=(*restarted[:4],values[4],options)
        completed,counters,history=r.restore_dpo_payload(payload,contract=values[4],
            model=restarted[0],reference=restarted[1],optimizer=restarted[2],sampler=restarted[3],
            work_ledger=work,snapshot_artifact_budget=ledger)
        replay=train(self.root/'joint-second',restored_model,ledger,completed=completed,counters=counters,
            history=history,parent_checkpoint=parent,work_ledger=work)
        self.assertEqual(work.snapshot()['reserved']['train_updates'],10)
        self.assertEqual(ledger.files['later-partial.bin']['limit'],256)
        original=load(result['checkpoint'],values,ledger,work)['state']
        actual=load(replay['checkpoint'],restored_model,ledger,work)['state']
        for key in ('policy','reference','optimizer','sampler_rng','torch_rng','counters','history'):
            self.assertEqual(digest(actual[key]),digest(original[key]))

    def test_whole_bundle_byte_refusal_before_serializer_or_update(self):
        caps=dict(CAPS,max_bytes=CAPS['journal_max_bytes']+CAPS['payload_max_bytes']+2*65536-1)
        values=fixture(caps);ledger=self.ledger(values)
        with patch('dongxi_llms.training_snapshot.torch.save') as serializer, \
                patch.object(r,'completed_dpo_update') as update,self.assertRaisesRegex(ValueError,'bundle'):
            train(self.root/'refused',values,ledger)
        serializer.assert_not_called();update.assert_not_called()
        self.assertEqual(ledger.usage()['actual_entries'],1)

    def test_whole_bundle_entry_refusal_before_serializer_or_update(self):
        values=fixture(dict(CAPS,max_entries=3));ledger=self.ledger(values)
        with patch('dongxi_llms.training_snapshot.torch.save') as serializer, \
                patch.object(r,'completed_dpo_update') as update,self.assertRaisesRegex(ValueError,'bundle'):
            train(self.root/'refused',values,ledger)
        serializer.assert_not_called();update.assert_not_called()

    def test_serialization_failure_retains_prior_commit_and_partial_charge(self):
        values=fixture();ledger=self.ledger(values);real_save=torch.save;calls=[]
        def broken(payload,handle):
            calls.append(payload['completed_updates'])
            if len(calls)==1:return real_save(payload,handle)
            handle.write(b'partial!');raise OSError('declared serialization failure')
        with patch('dongxi_llms.training_snapshot.torch.save',side_effect=broken),self.assertRaises(OSError):
            train(self.root/'failed',values,ledger)
        parent=base.checkpoint_receipt(next(ledger.root.glob('*completed-000000.pt')))
        self.assertEqual(load(parent,values,ledger)['state']['completed'],0)
        failed=next(name for name,row in ledger.files.items() if row['state']=='opened')
        self.assertEqual((ledger.root/failed).stat().st_size,8)
        self.assertEqual(ledger.files[failed]['limit'],CAPS['payload_max_bytes'])

    def test_publication_failure_retains_payload_staging_and_prior_commit(self):
        values=fixture();ledger=self.ledger(values);real_link=os.link;calls=[]
        def broken(*args,**kwargs):
            calls.append(args)
            if len(calls)==1:return real_link(*args,**kwargs)
            raise OSError('declared publication failure')
        with patch('dongxi_llms.artifact_budget.os.link',side_effect=broken),self.assertRaises(OSError):
            train(self.root/'failed',values,ledger)
        parent=base.checkpoint_receipt(next(ledger.root.glob('*completed-000000.pt')))
        self.assertEqual(load(parent,values,ledger)['state']['completed'],0)
        self.assertEqual(len(list(ledger.root.glob('*.pt'))),2)
        self.assertTrue(any(row['state']=='linked' for row in ledger.files.values()))
        self.assertTrue(any('.commit-' in name and row['state']=='sealed' for name,row in ledger.files.items()))

    def test_metric_failure_leaves_committed_history_authoritative(self):
        values=fixture();ledger=self.ledger(values)
        def broken(path,row):
            if row['update']==3:raise OSError('declared metric observer failure')
        with patch.object(r,'append_metric',side_effect=broken),self.assertRaises(OSError):
            train(self.root/'failed',values,ledger)
        parent=base.checkpoint_receipt(next(ledger.root.glob('*completed-000003.pt')))
        self.assertEqual(len(load(parent,values,ledger)['state']['history']),3)
        ledger.validate_receipt(load(parent,values,ledger)['state']['snapshot_artifact_ledger'])

    def test_bad_prefix_or_capacity_rejected_before_state_application(self):
        values=fixture();ledger=self.ledger(values)
        result=train(self.root/'first',values,ledger);payload=load(result['checkpoint'],values,ledger)
        for edit in ('head','bytes','capacity','root'):
            changed=deepcopy(payload);receipt=changed['state']['snapshot_artifact_ledger']
            if edit=='head':receipt['head_sha256']='0'*64
            elif edit=='bytes':receipt['journal_bytes']-=1
            elif edit=='capacity':receipt['identity']['max_bytes']+=1
            else:receipt['identity']['root_inode']+=1
            with patch.object(values[0],'load_state_dict') as apply,self.assertRaises(ValueError):
                r.restore_dpo_payload(changed,contract=values[4],model=values[0],reference=values[1],
                    optimizer=values[2],sampler=values[3],snapshot_artifact_budget=ledger)
            apply.assert_not_called()

    def test_new_output_cannot_change_ledger_or_payload_envelope(self):
        values=fixture();ledger=self.ledger(values)
        with self.assertRaisesRegex(ValueError,'envelope'):
            r.train_completed_updates(*values[:4],base.ENCODED,contract=values[4],output=self.root,
                invocation_id='refusal',pad_id=1,device='cpu',checkpoint_every=3,max_bytes=42,
                snapshot_artifact_budget=ledger)
        other=ArtifactBudget.create(self.root/'other',campaign_id='different',
            **{key:CAPS[key] for key in ('max_bytes','max_entries','journal_max_bytes')})
        self.addCleanup(other.close)
        with self.assertRaisesRegex(ValueError,'identity/capacity'):
            r.validate_snapshot_artifact_ledger(values[4],other)

    def test_physical_backend_remains_refused(self):
        with self.assertRaisesRegex(RuntimeError,'physical'):
            ArtifactBudget.create(self.root/'not-created',campaign_id='fixture',max_bytes=4096,
                max_entries=4,journal_max_bytes=4096,require_physical_backend=True)
        self.assertFalse((self.root/'not-created').exists())

    def test_initial_parsed_caps_cannot_be_relabelled_after_file_mutation(self):
        path=self.root/'limits.json'
        r.write_exclusive_json(path,{key:100000 for key in r.BUDGET_KEYS})
        expected=hashlib.sha256(r.read_work_limits_file(path)).hexdigest()
        r.verify_parsed_work_limits(path,expected)
        changed=self.root/'changed.json'
        r.write_exclusive_json(changed,{key:100001 for key in r.BUDGET_KEYS})
        os.replace(changed,path)
        with patch.object(r,'completed_dpo_update') as model_work,self.assertRaisesRegex(ValueError,'bytes changed'):
            r.verify_parsed_work_limits(path,expected)
        model_work.assert_not_called()

    def test_postparse_fifo_refuses_identity_binding_and_closure_without_read(self):
        path=self.root/'limits.json';r.write_exclusive_json(path,{key:100000 for key in r.BUDGET_KEYS})
        expected=hashlib.sha256(r.read_work_limits_file(path)).hexdigest()
        identity=r.bind_parsed_work_limits({'input_sha256':{},'source_sha256':{}},path,expected)
        self.assertEqual(identity['identity_sha256'],canonical_hash({key:value for key,value in identity.items()
            if key!='identity_sha256'}))
        fifo=self.root/'replacement-fifo';os.mkfifo(fifo,0o600);os.replace(fifo,path)
        with self.assertRaisesRegex(ValueError,'regular'):
            r.bind_parsed_work_limits({},path,expected)
        with self.assertRaisesRegex(ValueError,'regular'):
            r.verify_identity_files(identity,ROOT)


def collect(output):
    """Exclusive bounded raw proof, preserving development failures elsewhere."""
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    before={name:r.file_digest(ROOT/name) for name in SOURCES}
    command=[sys.executable,'-m','unittest','test_dpo_snapshot_artifacts','test_artifact_budget',
        'test_snapshot_artifact_budget','test_training_snapshot','test_training_snapshot_failure_paths',
        'test_dpo_runner_recovery','test_dpo_work_budget','test_dpo_checkpoint_mode']
    started=time.monotonic()
    process=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=90,
        env=dict(os.environ,PYTHONPATH=f'{ROOT}/src:{ROOT}/tests',CUDA_VISIBLE_DEVICES='',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1'))
    r.write_exclusive_json(output/'panel.json',dict(command=command,exit_code=process.returncode,
        stdout=process.stdout,stderr=process.stderr,seconds=time.monotonic()-started,
        source_before=before,source_after={name:r.file_digest(ROOT/name) for name in SOURCES},
        scope='original random CPU FP32 source integration; no pretrained/GPU/whole-output quota claim'))
    if process.returncode:raise RuntimeError('Raw failed panel retained')
    if before!={name:r.file_digest(ROOT/name) for name in SOURCES}:raise RuntimeError('Source changed during proof')
    values=fixture();ledger=create(output/'artifacts',values)
    try:
        result=train(output/'first',values,ledger)
        final=load(result['checkpoint'],values,ledger)
        parent=base.checkpoint_receipt(next(ledger.root.glob('*completed-000003.pt')))
        prefix=load(parent,values,ledger)['state']['snapshot_artifact_ledger']
        partial(ledger);retained_usage=ledger.usage()
        r.write_exclusive_json(output/'bundle.json',dict(contract=values[4],snapshot=parent,
            artifact_receipt=prefix,root=str(ledger.root)))
    finally:ledger.close()
    command=[sys.executable,str(Path(__file__).resolve()),'--child',str(output/'bundle.json'),str(output/'child')]
    started=time.monotonic();child=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=30,
        env=dict(os.environ,PYTHONPATH=f'{ROOT}/src:{ROOT}/tests',CUDA_VISIBLE_DEVICES='',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1'))
    answer=json.loads((output/'child/child-result.json').read_text()) if child.returncode==0 else None
    evidence=dict(command=command,exit_code=child.returncode,stdout=child.stdout,stderr=child.stderr,
        seconds=time.monotonic()-started,declared_caps=CAPS,retained_usage=retained_usage,
        original_numerical_sha256=digest(numerical(final)),child=answer,
        source_before=before,source_after={name:r.file_digest(ROOT/name) for name in SOURCES})
    r.write_exclusive_json(output/'fresh-replay.json',evidence)
    if child.returncode or answer['numerical_sha256']!=evidence['original_numerical_sha256']:
        raise RuntimeError('Raw failed fresh replay retained')
    if before!=evidence['source_after']:raise RuntimeError('Source changed during fresh replay')
    return evidence


if __name__=='__main__':
    if len(sys.argv)==4 and sys.argv[1]=='--child':fresh_child(*sys.argv[2:])
    elif len(sys.argv)==3 and sys.argv[1]=='--collect':collect(sys.argv[2])
    else:unittest.main()
