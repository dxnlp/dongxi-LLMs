"""Actual original RLVR completed/pending shared-I/O CPU acceptance controls."""
from copy import deepcopy
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib.util
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
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import test_rlvr_runner_recovery as base
import test_rlvr_work_budget as prior
from dongxi_llms import qwen_rlvr_lab as r
from dongxi_llms import training_snapshot as shared
from dongxi_llms.batched_cache_lab import digest
from dongxi_llms.run_identity import canonical_hash
from dongxi_llms.snapshot_io_budget import (IO_KEYS, SnapshotIOBudget,
    io_budget_contract, read_bounded_json, read_work_receipt)
from dongxi_llms.work_budget import WorkBudgetExceeded, WorkLedger

ROOT=base.ROOT
PROTOCOL='experiments/reports/2026-10-05-rlvr-snapshot-io-cpu-protocol.md'
PARENT_SPEC='experiments/specs/2026-10-05-rlvr-snapshot-io-admission.md'
ARCHIVE='experiments/reports/2026-10-05-rlvr-snapshot-io-preintegration'
ENVELOPE=dict(max_payload_bytes=base.LIMIT,max_tree_nodes=100000,
    max_tensor_elements=1000000,max_tensor_bytes=8*1024**2,max_primitive_bytes=1024**2)
IO_CAPS=dict(snapshot_inspect_operations=16,snapshot_load_operations=16,snapshot_save_operations=32,
    snapshot_hash_bytes=1024**3,snapshot_tree_nodes=6000000,snapshot_tensor_elements=48000000,
    snapshot_primitive_bytes=64*1024**2,snapshot_clone_bytes=256*1024**2,
    snapshot_serialization_bytes=512*1024**2)
IO_BOUND=2*1024**2
SOURCES=('src/dongxi_llms/qwen_rlvr_lab.py','src/dongxi_llms/grpo_lab.py',
    'src/dongxi_llms/batched_cache_lab.py','src/dongxi_llms/run_identity.py',
    'src/dongxi_llms/training_snapshot.py','src/dongxi_llms/snapshot_io_budget.py',
    'src/dongxi_llms/work_budget.py','tests/test_rlvr_snapshot_io.py',
    'src/dongxi_llms/artifact_budget.py','src/dongxi_llms/rlvr_snapshot_io_lab.py',
    'tests/test_rlvr_snapshot_io_lab.py','experiments/specs/2026-10-05-rlvr-reader-lesson.md',
    'tests/test_rlvr_runner_recovery.py','tests/test_rlvr_work_budget.py',
    'tests/test_snapshot_io_budget.py','tests/test_training_snapshot.py',
    'tests/test_training_snapshot_failure_paths.py','tests/snapshot_io_test_support.py',
    'pyproject.toml','uv.lock',PROTOCOL,PARENT_SPEC,ARCHIVE+'/manifest.json',
    ARCHIVE+'/qwen_rlvr_lab.py.txt',ARCHIVE+'/test_rlvr_runner_recovery.py.txt',
    ARCHIVE+'/test_rlvr_work_budget.py.txt')


def write(path,value):
    path=Path(path)
    with path.open('x',encoding='utf8') as handle:
        handle.write(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
        handle.flush();os.fsync(handle.fileno())


def fixture(seed=2323,limits=IO_CAPS,*,envelope=ENVELOPE):
    loop,old=prior.fixture(seed)
    observed={key:deepcopy(value) for key,value in old.items()
              if key not in ('loop_contract','reference_sha256')}
    observed['sources']={name:r.file_digest(ROOT/name) for name in SOURCES}
    observed['snapshot_io_contract']=io_budget_contract(limits,envelope,IO_BOUND)
    return loop,r.full_contract(observed,loop)


def create(directory,loop,contract):
    directory=Path(directory);directory.mkdir(mode=0o700)
    work,io,hook=r.open_snapshot_budgets(contract=contract,budget=contract['work_budget'],
        io_contract=contract['snapshot_io_contract'],receipt=None,work_path=directory/'work.jsonl',
        io_path=directory/'io.jsonl',invocation_id='authored-original-rlvr-io')
    loop.bind_work_ledger(work,contract)
    return work,io,hook


def receipt_for(path):
    external=read_work_receipt(Path(str(path)+'.work.json'))
    return dict(path=str(Path(path).resolve()),work_receipt_path=str(Path(str(path)+'.work.json').resolve()),
                **external['snapshot'])


def save(loop,path,contract,hook):
    header=loop.save(path,contract=contract,parent_invocation='authored-rlvr-io',
        max_bytes=base.LIMIT,io_budget=hook,work_receipt_path=Path(str(path)+'.work.json'))
    return dict(path=str(Path(path).resolve()),work_receipt_path=str(Path(str(path)+'.work.json').resolve()),**header)


def inspect(receipt,contract,hook):
    hook.bind_receipt(read_work_receipt(receipt['work_receipt_path']))
    return shared.inspect_snapshot(receipt['path'],expected_contract=contract,
        expected_sha256=receipt['payload_sha256'],expected_bytes=receipt['payload_bytes'],
        max_bytes=base.LIMIT,io_budget=hook)


def restore(loop,receipt,contract,hook):
    hook.bind_receipt(read_work_receipt(receipt['work_receipt_path']))
    return loop.restore(receipt['path'],contract=contract,expected_sha256=receipt['payload_sha256'],
        expected_bytes=receipt['payload_bytes'],max_bytes=base.LIMIT,io_budget=hook)


def numerical(loop):
    return prior.numerical(loop)


def lifecycle(directory,loop,contract,hook,*,on_commit=lambda value:None,metric_sink=lambda row:None):
    Path(directory).mkdir(mode=0o700)
    return r.run_loop_with_recovery(loop,contract=contract,output=directory,
        parent_invocation='authored-rlvr-io',max_bytes=base.LIMIT,io_budget=hook,
        on_commit=on_commit,metric_sink=metric_sink,
        baseline=lambda:prior.evaluation(loop,loop.work_ledger),
        final=lambda:prior.evaluation(loop,loop.work_ledger))


def original_actor():
    path=ROOT/ARCHIVE/'qwen_rlvr_lab.py.txt'
    name='dongxi_original_rlvr_snapshot_io_actor'
    module=SimpleNamespace(__name__=name,__file__=str(path))
    # Trusted original course archive, not arbitrary model text or pickle.
    space=vars(module)
    exec(compile(path.read_text(),str(path),'exec'),space)
    return module


def original_loop(seed):
    initial,_=base.make_loop(seed);old=original_actor()
    return old.RLVRLoop(initial.model,initial.reference,initial.optimizer,initial.generator,
        base.TRAIN,base.decode,eos_id=0,stop_ids=[0,7],group_size=3,max_new_tokens=4,
        updates=4,seed=seed,context_length=32,beta=.02)


class BoundaryReached(OSError):
    pass


def interrupt(directory,loop,contract,hook,phase):
    receipts=[]
    def committed(receipt):
        receipts.append(receipt)
        if receipt['completed_updates']==2 and receipt['phase']==phase:
            raise BoundaryReached('authored observer interruption after committed '+phase+'2')
    try:lifecycle(directory,loop,contract,hook,on_commit=committed)
    except BoundaryReached as error:
        return receipts[-1],dict(type=type(error).__name__,message=str(error),injected=True,receipts=receipts)
    raise AssertionError('Predeclared committed-two interruption absent')


def later_failures(loop,contract,hook,receipt):
    before=dict(model=loop.work_ledger.snapshot(),io=hook.ledger.snapshot())
    with patch.object(shared.torch,'load',side_effect=OSError('authored later restricted-load failure')):
        try:restore(loop,receipt,contract,hook)
        except OSError as error:io_error=dict(type=type(error).__name__,message=str(error))
        else:raise AssertionError('Predeclared admitted load failure absent')
    invalid=deepcopy(loop.snapshot_state());invalid['rng']['rollout']=invalid['rng']['rollout'].float()
    try:loop.validate_payload(dict(state=invalid,completed_updates=loop.completed,
                                  phase='pending' if loop.pending is not None else 'completed'))
    except ValueError as error:model_error=dict(type=type(error).__name__,message=str(error))
    else:raise AssertionError('Predeclared charged semantic failure absent')
    return dict(before=before,after=dict(model=loop.work_ledger.snapshot(),io=hook.ledger.snapshot()),
                io_error=io_error,model_error=model_error)


def fresh_child(bundle_path,output):
    bundle,_=read_bounded_json(bundle_path);contract,_=read_bounded_json(bundle['contract_path'])
    receipt=read_work_receipt(bundle['snapshot']['work_receipt_path'])
    work,io,hook=r.open_snapshot_budgets(contract=contract,budget=contract['work_budget'],
        io_contract=contract['snapshot_io_contract'],receipt=receipt,
        work_path=bundle['work_path'],io_path=bundle['io_path'],invocation_id='authored-fresh-rlvr-io',
        expected_sha256=bundle['snapshot']['payload_sha256'],expected_bytes=bundle['snapshot']['payload_bytes'])
    try:
        admitted=inspect(bundle['snapshot'],contract,hook)
        # Actual inspect precedes constructing even the original random model.
        loop,current=fixture(bundle['seed'])
        if current!=contract:raise ValueError('Actual fresh source/interface/science changed')
        loop.bind_work_ledger(work,contract)
        restored=restore(loop,bundle['snapshot'],contract,hook)
        after_restore=dict(model=work.snapshot(),io=io.snapshot())
        restored_pool=deepcopy(loop.pending);calls=[];actual_collect=loop.collect
        def checked_collect():
            if bundle['phase']=='pending' and loop.completed==2:
                raise AssertionError('Pending action cannot be recollected before application')
            calls.append(loop.completed);return actual_collect()
        tail=[];receipts=[]
        with patch.object(loop,'collect',side_effect=checked_collect):
            result=lifecycle(output,loop,contract,hook,on_commit=receipts.append,metric_sink=tail.append)
        if len(tail)!=2:raise AssertionError('Expected exactly two postrestore updates')
        if restored_pool is not None and digest(tail[0]['collection'])!=digest(restored_pool):
            raise AssertionError('First applied pool differs from durable pending pool')
        answer=dict(seed=bundle['seed'],phase=bundle['phase'],restored_update=restored['completed_updates'],
            completed_updates=loop.completed,state_sha256=digest(numerical(loop)),history=r.jsonable(loop.history),
            first_postrestore_action=r.jsonable(tail[0]),tail_sha256=digest(tail),
            collect_completed_cursors=calls,no_collection_before_pending_apply=bundle['phase']=='pending',
            original_pending_pool_sha256=digest(restored_pool) if restored_pool is not None else None,
            inspected_before_model=admitted,after_restore=after_restore,
            work=work.snapshot(),io=io.snapshot(),latest=result['latest'],receipts=receipts,
            initial_evaluation=result['initial'],final_evaluation=result['final'])
        write(Path(output)/'fresh-result.json',answer)
        return answer
    finally:io.close();work.close()


def replay_arm(directory,seed,phase):
    directory=Path(directory);directory.mkdir(mode=0o700)
    clean,clean_contract=fixture(seed)
    with ExitStack() as stack:
        work,io,hook=create(directory/'clean-journals',clean,clean_contract)
        stack.callback(work.close);stack.callback(io.close)
        clean_result=lifecycle(directory/'clean',clean,clean_contract,hook)
        expected=digest(numerical(clean));tail=digest(clean.history[2:])
        clean_work=work.snapshot();clean_io=io.snapshot()
    loop,contract=fixture(seed)
    if contract!=clean_contract:raise AssertionError('Fixed science changed between arms')
    work,io,hook=create(directory/'retained-journals',loop,contract)
    try:
        receipt,interruption=interrupt(directory/'interrupted',loop,contract,hook,phase)
        failures=later_failures(loop,contract,hook,receipt)
        carried=read_work_receipt(receipt['work_receipt_path'])
        write(directory/'contract.json',contract)
        bundle=dict(seed=seed,phase=phase,contract_path=str((directory/'contract.json').resolve()),
            snapshot=receipt,work_path=str((directory/'retained-journals/work.jsonl').resolve()),
            io_path=str((directory/'retained-journals/io.jsonl').resolve()))
        write(directory/'bundle.json',bundle)
        write(directory/'retained-failures.json',failures)
        write(directory/'interruption.json',interruption)
    finally:io.close();work.close()
    command=[sys.executable,str(Path(__file__).resolve()),'--child',str(directory/'bundle.json'),str(directory/'fresh')]
    started=time.monotonic();child=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,timeout=60)
    execution=dict(command=command,exit_code=child.returncode,seconds=time.monotonic()-started,
                   stdout=child.stdout,stderr=child.stderr,deadline_seconds=60)
    write(directory/'fresh-execution.json',execution)
    if child.returncode:raise AssertionError('Fresh RLVR replay child failed: '+child.stderr)
    answer=json.loads(child.stdout)
    if answer['state_sha256']!=expected or answer['tail_sha256']!=tail:
        raise AssertionError('Fresh replay differs from original fixed trajectory')
    if not set(failures['after']['io']['failed_tickets'])<=set(answer['io']['failed_tickets']):
        raise AssertionError('Later I/O failure was refunded on resume')
    if not set(failures['after']['model']['failed_tickets'])<=set(answer['work']['failed_tickets']):
        raise AssertionError('Later model-semantic failure was refunded on resume')
    if answer['collect_completed_cursors']!=([3] if phase=='pending' else [2,3]):
        raise AssertionError('Resumed action/source cursor geometry changed')
    arm=dict(seed=seed,phase=phase,expected_numerical_sha256=expected,expected_tail_sha256=tail,
        exact_fresh_numerical_replay=True,execution=execution,fresh=answer,
        carried_receipt=carried,retained_failures=failures,interruption=interruption,
        clean_work=clean_work,clean_io=clean_io,clean_evaluation=clean_result['final'])
    write(directory/'arm.json',arm)
    return arm


def publication_control(directory):
    """Native lifecycle: completed0 succeeds; pending0 header publication fails."""
    directory=Path(directory);directory.mkdir(mode=0o700)
    loop,contract=fixture();work,io,hook=create(directory/'journals',loop,contract)
    seen=[];prior_bytes={};previous_hook=None;actual_link=shared.os.link
    def committed(receipt):
        nonlocal previous_hook
        seen.append(receipt)
        prior_bytes.update({name:Path(path).read_bytes() for name,path in (
            ('payload',receipt['path']),('marker',receipt['path']+'.commit.json'),
            ('work_receipt',receipt['work_receipt_path']))})
        previous_hook=deepcopy(hook.last_receipt)
    def fail_pending(source,target,*args,**kwargs):
        if Path(target).name=='pending-000000.pt.commit.json':
            raise OSError('authored pending0 atomic header publication failure')
        return actual_link(source,target,*args,**kwargs)
    try:
        try:
            with patch.object(shared.os,'link',side_effect=fail_pending):
                lifecycle(directory/'native',loop,contract,hook,on_commit=committed)
        except OSError as error:failure=dict(type=type(error).__name__,message=str(error))
        else:raise AssertionError('Predeclared native publication failure absent')
        if len(seen)!=1 or seen[0]['phase']!='completed' or seen[0]['completed_updates']!=0:
            raise AssertionError('Failed pending publication delivered a durable pointer')
        retained=seen[0]
        current={name:Path(path).read_bytes() for name,path in (
            ('payload',retained['path']),('marker',retained['path']+'.commit.json'),
            ('work_receipt',retained['work_receipt_path']))}
        if current!=prior_bytes or hook.last_receipt!=previous_hook:
            raise AssertionError('Failed publication changed the prior durable boundary')
        failed=directory/'native/snapshots/pending-000000.pt'
        staging=list(failed.parent.glob(failed.name+'.commit-*'))
        if not failed.is_file() or len(staging)!=1 or Path(str(failed)+'.commit.json').exists() or Path(str(failed)+'.work.json').exists():
            raise AssertionError('Failed atomic publication must retain data/staging, not commit or receipt')
        draft=json.loads(staging[0].read_text())
        actual_sha=hashlib.sha256(failed.read_bytes()).hexdigest()
        if draft['payload_sha256']!=actual_sha or draft['payload_bytes']!=failed.stat().st_size:
            raise AssertionError('Retained staging header does not bind actual failed payload')
        actual_model=work.snapshot();actual_io=io.snapshot()
        if not actual_io['failed_tickets'] or actual_io['completed']['snapshot_save_operations']!=1:
            raise AssertionError('Publication failure must remain a failed admitted shared save')
        if actual_model['reserved']['collections']!=1 or loop.pending is None or loop.completed!=0:
            raise AssertionError('Failed publication changed the original pending collection')
        result=dict(error=failure,old_checkpoint=retained,old_checkpoint_unchanged=True,
            hook_last_receipt_unchanged=True,on_commit_deliveries=seen,
            failed_payload=str(failed.resolve()),actual_failed_payload_bytes=failed.stat().st_size,
            failed_payload_sha256=actual_sha,staging_header=str(staging[0].resolve()),draft_header=draft,
            failed_marker_published=False,failed_external_receipt_published=False,
            completed_updates=loop.completed,pending_pool_sha256=digest(loop.pending),
            work=actual_model,io=actual_io)
        write(directory/'publication-control.json',result);return result
    finally:io.close();work.close()


def cli_control(directory,*,exhausted=False,bootstrap_fields=None,alias_role=None,hardlink=False):
    """Native parsing/admission with mocked platform/interface, real journals.

    The parent is an authored config placeholder, never model weights. Mocked
    observable metadata cannot authorize a production launch. Model allocation
    is deliberately stopped at its provider boundary.
    """
    directory=Path(directory);directory.mkdir(mode=0o700)
    loop,contract=fixture(limits=dict(IO_CAPS,snapshot_inspect_operations=0) if exhausted else IO_CAPS)
    work,io,hook=create(directory/'journals',loop,contract)
    try:receipt=save(loop,directory/'initial.pt',contract,hook)
    finally:io.close();work.close()
    parent=directory/'placeholder-parent';parent.mkdir();write(parent/'config.json',{'max_position_embeddings':32})
    files={}
    for name,value in [('work-limits',prior.CAPS),('io-limits',contract['snapshot_io_contract']),('contract',contract)]:
        files[name]=directory/(name+'.json');write(files[name],value)
    output=directory/'cli-output'
    argv=['--model-dir',str(parent),'--revision','a'*40,'--output',str(output),'--updates','4',
        '--group-size','3','--max-new-tokens','4','--seed','2323','--lr','.008','--device','cpu',
        '--prompt-mode','raw','--max-seconds','60','--environment-lock',str(ROOT/'uv.lock'),
        '--snapshot-max-bytes',str(base.LIMIT),'--work-limits',str(files['work-limits']),
        '--work-journal-max-bytes',str(prior.BOUND),'--work-journal',str(directory/'journals/work.jsonl'),
        '--snapshot-io-limits',str(files['io-limits']),'--snapshot-io-ledger',str(directory/'journals/io.jsonl'),
        '--resume',receipt['path'],'--resume-sha256',receipt['payload_sha256'],
        '--resume-bytes',str(receipt['payload_bytes']),'--resume-contract',str(files['contract']),
        '--resume-io-receipt',receipt['work_receipt_path']]
    if bootstrap_fields is not None:
        altered=read_work_receipt(receipt['work_receipt_path']);altered['snapshot'].update(bootstrap_fields)
        files['altered-receipt']=directory/'altered-receipt.json';write(files['altered-receipt'],altered)
        argv[argv.index('--resume-io-receipt')+1]=str(files['altered-receipt'])
    if alias_role is not None:
        alias=Path(receipt['path'])
        if hardlink:
            alias=directory/'hardlinked-payload-role';os.link(receipt['path'],alias)
        if alias_role=='parent-artifact':
            os.link(receipt['path'],parent/'model.safetensors')
        elif alias_role=='template':argv.extend(['--template',str(alias)])
        elif alias_role=='environment-lock':argv[argv.index('--environment-lock')+1]=str(alias)
        else:raise ValueError('Declared alias control role unknown')
    observed={key:value for key,value in contract.items() if key not in ('loop_contract','reference_sha256')}
    generic=[];identity_calls=[];unique_prompts={}
    def encode(prompt):
        if prompt not in unique_prompts:unique_prompts[prompt]=len(unique_prompts)+2
        return torch.tensor([[1,unique_prompts[prompt]]])
    def identity(root,**kwargs):
        identity_calls.append(sorted(kwargs));generic.extend(str(Path(p)) for p in kwargs['input_files'])
        return dict(source_sha256={str(Path(p).relative_to(ROOT)):r.file_digest(p) for p in kwargs['source_files']},
            input_sha256={str(Path(p)):r.file_digest(p) for p in kwargs['input_files']},
            checkpoint_files=r.artifact_hashes(parent),checkpoint_path=str(parent),device=kwargs['device'],gpu_driver={'status':'mocked'},
            environment={'python_version':platform.python_version(),'platform':platform.platform(),
                'machine':platform.machine(),'packages':{},
                'environment_lock':{'path':str(ROOT/'uv.lock'),'sha256':r.file_digest(ROOT/'uv.lock')}})
    from transformers import AutoModelForCausalLM,AutoTokenizer
    with ExitStack() as stack:
        stack.enter_context(patch.object(r,'collect_run_identity',side_effect=identity))
        stack.enter_context(patch.object(r,'observable_contract',return_value=observed))
        stack.enter_context(patch.object(r,'tokenizer_interface',return_value={'generation_stop_ids':[0,7]}))
        stack.enter_context(patch.object(r,'prompt_contract',return_value=(encode,[0,7],{'mode':'authored-mock-raw'})))
        stack.enter_context(patch.object(r,'guard_memory',return_value=30.))
        tokenizer=stack.enter_context(patch.object(AutoTokenizer,'from_pretrained',return_value=SimpleNamespace(
            eos_token_id=0,pad_token_id=0,chat_template=None)))
        backend=stack.enter_context(patch.object(AutoModelForCausalLM,'from_pretrained',
            side_effect=RuntimeError('authored model-allocation sentinel; no model loaded')))
        actual_verify=shared._verified_bytes
        verified=stack.enter_context(patch.object(shared,'_verified_bytes',wraps=actual_verify))
        original_open=WorkLedger.open
        journal_open=stack.enter_context(patch.object(WorkLedger,'open',wraps=original_open))
        try:r.main(argv)
        except (ValueError,RuntimeError) as error:failure=dict(type=type(error).__name__,message=str(error))
        else:raise AssertionError('Expected admission refusal or model-allocation sentinel')
        refused_metadata=bootstrap_fields is not None or alias_role is not None
        expected_calls=0 if exhausted or refused_metadata else 1
        if backend.call_count!=expected_calls or verified.call_count!=expected_calls:
            raise AssertionError('CLI crossed payload/backend boundary before I/O admission')
        if refused_metadata:
            if failure['type']!='ValueError' or identity_calls or tokenizer.call_count or journal_open.call_count:
                raise AssertionError('Bootstrap/alias gate must precede generic identity/tokenizer/journals')
            if bootstrap_fields is not None and output.exists():
                raise AssertionError('Invalid phase/horizon receipt must precede output creation')
        elif exhausted and failure['type']!='WorkBudgetExceeded':
            raise AssertionError('Exhausted CLI did not reach the actual I/O cap gate')
        if not refused_metadata and not exhausted and 'model-allocation sentinel' not in failure['message']:
            raise AssertionError('Successful CLI inspection did not stop at the model boundary')
    saved=json.loads((output/'report.json').read_text()) if output.exists() else {}
    identity_record=saved.get('run_identity',{})
    hidden={receipt['path'],receipt['path']+'.commit.json',receipt['work_receipt_path'],
            str(files['contract']),str(files['io-limits']),str(files['work-limits'])}
    if hidden.intersection(generic):raise AssertionError('Generic identity secretly hashed bootstrap/payload path')
    if ('verified_resume_snapshot' in identity_record)!=(not exhausted and not refused_metadata):
        raise AssertionError('Declared expectation was mislabeled as actual payload inspection')
    answer=dict(exhausted=exhausted,command=['native r.main',*argv],failure=failure,
        bootstrap_fields=bootstrap_fields,alias_role=alias_role,hardlink=hardlink,
        generic_identity_input_paths=generic,identity_passes=len(identity_calls),
        payload_verifier_calls=verified.call_count,model_provider_calls=backend.call_count,
        tokenizer_provider_calls=tokenizer.call_count,journal_open_calls=journal_open.call_count,
        run_identity=identity_record,work=saved.get('work_ledger_current'),io=saved.get('snapshot_io_ledger_current'),
        scope='mocked platform/parent/interface observers, actual CLI/journal/shared-inspection source order; no model loaded')
    write(directory/'control.json',answer)
    return answer


class RLVRSnapshotIOTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='dongxi-rlvr-io-');self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)

    def make(self,seed=2323,limits=IO_CAPS,*,envelope=ENVELOPE,name='journals'):
        loop,contract=fixture(seed,limits,envelope=envelope)
        work,io,hook=create(self.root/name,loop,contract)
        self.addCleanup(io.close);self.addCleanup(work.close)
        return loop,contract,work,io,hook

    def test_original_23_dimensions_caps_and_archive_byte_identity(self):
        self.assertEqual(len(r.BUDGET_KEYS),23)
        self.assertEqual(prior.CAPS,{key:100000 for key in r.BUDGET_KEYS})
        archive=json.loads((ROOT/ARCHIVE/'manifest.json').read_text())
        self.assertEqual(r.file_digest(ROOT/ARCHIVE/'qwen_rlvr_lab.py.txt'),archive['original_runner_sha256'])

    def test_hooked_unhooked_retained_original_numerical_parity_both_seeds(self):
        for seed in (2323,2324):
            plain,_=base.make_loop(seed);original=original_loop(seed)
            loop,contract,work,io,hook=self.make(seed,name=str(seed))
            save(loop,self.root/f'initial-{seed}.pt',contract,hook)
            for number in range(4):
                actual=loop.apply_pending();expected=plain.apply_pending();legacy=original.apply_pending()
                self.assertEqual(digest(actual),digest(expected));self.assertEqual(digest(actual),digest(legacy))
            self.assertEqual(digest(numerical(loop)),digest(plain.snapshot_state()))
            self.assertEqual(digest(numerical(loop)),digest(original.snapshot_state()))
            self.assertEqual(work.snapshot()['completed']['train_updates'],4)

    def test_native_lifecycle_commits_all_nine_boundaries_and_prefixes(self):
        loop,contract,work,io,hook=self.make();seen=[]
        result=lifecycle(self.root/'course-loop',loop,contract,hook,on_commit=seen.append)
        self.assertEqual([(x['phase'],x['completed_updates']) for x in seen],
            [('completed',0)]+[(p,n) for n in range(4) for p,n in [('pending',n),('completed',n+1)]])
        self.assertEqual(io.snapshot()['completed']['snapshot_save_operations'],9)
        receipt=read_work_receipt(result['latest']['work_receipt_path'])
        hook.bind_receipt(receipt)
        payload=shared.load_snapshot(result['latest']['path'],expected_contract=contract,
            expected_sha256=receipt['snapshot']['payload_sha256'],expected_bytes=receipt['snapshot']['payload_bytes'],
            max_bytes=base.LIMIT,io_budget=hook)
        self.assertEqual(payload['state']['work_ledger'],receipt['runner_work_prefix'])
        self.assertEqual(payload['snapshot_io']['io_prefix'],receipt['io_prefix'])
        self.assertLess(receipt['runner_work_prefix']['sequence'],work.snapshot()['sequence'])
        self.assertLess(receipt['io_prefix']['sequence'],io.snapshot()['sequence'])

    def test_fresh_process_completed_pending_replay_both_frozen_seeds(self):
        for seed in (2323,2324):
            for phase in ('completed','pending'):
                with self.subTest(seed=seed,phase=phase):
                    arm=replay_arm(self.root/f'{seed}-{phase}',seed,phase)
                    self.assertTrue(arm['exact_fresh_numerical_replay'])

    def test_declared_hook_omission_precedes_capture_semantic_shared_and_mkdir(self):
        loop,contract,_,_,_=self.make()
        with (patch.object(loop,'snapshot_state') as capture,patch.object(loop,'validate_payload') as semantic,
              patch.object(r,'save_snapshot') as shared_save,patch.object(r,'load_snapshot') as shared_load):
            with self.assertRaises(ValueError):loop.save(self.root/'wrong.pt',contract=contract,parent_invocation='missing',max_bytes=base.LIMIT)
            with self.assertRaises(ValueError):loop.restore(self.root/'wrong.pt',contract=contract,
                expected_sha256='0'*64,expected_bytes=1,max_bytes=base.LIMIT)
            output=self.root/'never';output.mkdir()
            with self.assertRaises(ValueError):r.run_loop_with_recovery(loop,contract=contract,output=output,
                parent_invocation='missing',max_bytes=base.LIMIT)
        capture.assert_not_called();semantic.assert_not_called();shared_save.assert_not_called();shared_load.assert_not_called()
        self.assertFalse((output/'snapshots').exists())

    def test_explicit_unhooked_teaching_contract_remains_schema_one(self):
        loop,contract=base.make_loop();path=self.root/'reference.pt'
        header=loop.save(path,contract=contract,parent_invocation='explicit-unaccounted',max_bytes=base.LIMIT)
        self.assertEqual(header['schema_version'],1)
        self.assertFalse(Path(str(path)+'.work.json').exists())
        base.restore(loop,path,contract,header)

    def test_zero_save_admission_precedes_shared_tree_clone_serialization(self):
        loop,contract,work,io,hook=self.make(limits=dict(IO_CAPS,snapshot_save_operations=0))
        before=work.snapshot()['reserved']['recovery_validation_operations']
        with patch.object(shared,'_safe_tree') as tree,patch.object(torch,'save') as serializer:
            with self.assertRaises(WorkBudgetExceeded):save(loop,self.root/'refused.pt',contract,hook)
        tree.assert_not_called();serializer.assert_not_called();self.assertFalse((self.root/'refused.pt').exists())
        # Original caller-owned semantic validation already ran separately.
        self.assertEqual(work.snapshot()['reserved']['recovery_validation_operations'],before+1)
        self.assertEqual(io.snapshot()['sequence'],0)

    def test_zero_inspect_or_load_precedes_payload_and_backend(self):
        for dimension in ('snapshot_inspect_operations','snapshot_load_operations'):
            loop,contract,_,_,hook=self.make(limits=dict(IO_CAPS,**{dimension:0}),name=dimension)
            receipt=save(loop,self.root/(dimension+'.pt'),contract,hook)
            hook.bind_receipt(read_work_receipt(receipt['work_receipt_path']))
            with (patch.object(shared,'_verified_bytes') as bytes_read,patch.object(torch,'load') as deserialize,
                  patch.object(loop.model,'load_state_dict') as apply):
                with self.assertRaises(WorkBudgetExceeded):
                    if dimension=='snapshot_inspect_operations':inspect(receipt,contract,hook)
                    else:restore(loop,receipt,contract,hook)
            bytes_read.assert_not_called();deserialize.assert_not_called();apply.assert_not_called()

    def test_hook_science_or_envelope_mismatch_precedes_caller_work(self):
        loop,contract,work,io,hook=self.make();changed=deepcopy(contract)
        changed['snapshot_io_contract']['limits']['snapshot_save_operations']-=1
        with patch.object(loop,'snapshot_state') as capture,patch.object(loop,'validate_payload') as semantic:
            with self.assertRaises(ValueError):save(loop,self.root/'wrong.pt',changed,hook)
        capture.assert_not_called();semantic.assert_not_called();self.assertEqual(io.snapshot()['sequence'],0)

    def test_changed_expected_bytes_or_hash_precedes_actual_payload_read(self):
        loop,contract,_,_,hook=self.make();receipt=save(loop,self.root/'saved.pt',contract,hook)
        hook.bind_receipt(read_work_receipt(receipt['work_receipt_path']))
        for sha,size in [('0'*64,receipt['payload_bytes']),(receipt['payload_sha256'],receipt['payload_bytes']+1)]:
            with patch.object(shared,'_verified_bytes') as bytes_read:
                with self.assertRaises(ValueError):loop.restore(receipt['path'],contract=contract,
                    expected_sha256=sha,expected_bytes=size,max_bytes=base.LIMIT,io_budget=hook)
            bytes_read.assert_not_called()

    def test_bootstrap_phase_horizon_and_pending_terminal_before_journals(self):
        loop,contract,_,_,hook=self.make();receipt=save(loop,self.root/'initial.pt',contract,hook)
        retained=read_work_receipt(receipt['work_receipt_path'])
        for fields in [dict(phase='unknown'),dict(completed_updates=5),dict(phase='pending',completed_updates=4)]:
            bad=deepcopy(retained);bad['snapshot'].update(fields)
            with (patch.object(WorkLedger,'open') as journals,patch.object(shared,'inspect_snapshot') as bytes_read):
                with self.assertRaises(ValueError):r.open_snapshot_budgets(contract=contract,
                    budget=contract['work_budget'],io_contract=contract['snapshot_io_contract'],receipt=bad,
                    work_path=self.root/'absent-work',io_path=self.root/'absent-io',invocation_id='bad',
                    expected_sha256=bad['snapshot']['payload_sha256'],expected_bytes=bad['snapshot']['payload_bytes'])
            journals.assert_not_called();bytes_read.assert_not_called()

    def test_cap_schema_omission_boolean_and_different_budget_refuse(self):
        loop,contract,_,_,hook=self.make()
        for caps in [dict(IO_CAPS,snapshot_load_operations=True),dict(IO_CAPS,extra=1),
                     {k:v for k,v in IO_CAPS.items() if k!='snapshot_save_operations'}]:
            with self.assertRaises(ValueError):io_budget_contract(caps,ENVELOPE,IO_BOUND)
        bad=deepcopy(contract['work_budget']);bad['limits']['collections']=99999
        with patch.object(WorkLedger,'create') as journals:
            with self.assertRaises(ValueError):r.open_snapshot_budgets(contract=contract,budget=bad,
                io_contract=contract['snapshot_io_contract'],receipt=None,work_path=self.root/'work',
                io_path=self.root/'io',invocation_id='bad')
        journals.assert_not_called()

    def test_changed_or_copied_journal_cannot_refill_saved_receipt(self):
        loop,contract,work,io,hook=self.make();receipt=save(loop,self.root/'initial.pt',contract,hook)
        retained=read_work_receipt(receipt['work_receipt_path']);work.close();io.close()
        copied=self.root/'copied';copied.mkdir()
        for name in ('work.jsonl','io.jsonl'):shutil.copyfile(self.root/'journals'/name,copied/name);(copied/name).chmod(0o600)
        for work_path,io_path in [(copied/'work.jsonl',self.root/'journals/io.jsonl'),
                                 (self.root/'journals/work.jsonl',copied/'io.jsonl')]:
            with self.assertRaises(ValueError):r.open_snapshot_budgets(contract=contract,
                budget=contract['work_budget'],io_contract=contract['snapshot_io_contract'],receipt=retained,
                work_path=work_path,io_path=io_path,invocation_id='copy',
                expected_sha256=receipt['payload_sha256'],expected_bytes=receipt['payload_bytes'])

    def test_legacy_payload_refused_in_explicit_accounted_mode(self):
        loop,contract,_,_,hook=self.make();receipt=save(loop,self.root/'saved.pt',contract,hook)
        marker=Path(receipt['path']+'.commit.json');header=json.loads(marker.read_text());header['schema_version']=1
        marker.write_text(json.dumps(header));hook.bind_receipt(read_work_receipt(receipt['work_receipt_path']))
        actual_open=shared._open_regular
        with patch.object(shared,'_open_regular',wraps=actual_open) as reads,patch.object(torch,'load') as deserialize:
            with self.assertRaises(ValueError):restore(loop,receipt,contract,hook)
        self.assertNotIn(Path(receipt['path']),[Path(call.args[0]) for call in reads.call_args_list])
        deserialize.assert_not_called()

    def test_partial_serialization_retains_old_boundary_receipt_and_spending(self):
        loop,contract,work,io,hook=self.make();receipt=save(loop,self.root/'initial.pt',contract,hook)
        previous=Path(receipt['path']).read_bytes();previous_receipt=Path(receipt['work_receipt_path']).read_bytes()
        loop.collect()
        def partial(value,handle,*args,**kwargs):
            handle.write(b'partial!');raise OSError('authored eight-byte serializer failure')
        with patch.object(torch,'save',side_effect=partial),self.assertRaises(OSError):
            save(loop,self.root/'failed-pending.pt',contract,hook)
        self.assertEqual(Path(receipt['path']).read_bytes(),previous)
        self.assertEqual(Path(receipt['work_receipt_path']).read_bytes(),previous_receipt)
        self.assertTrue((self.root/'failed-pending.pt').exists())
        self.assertFalse(Path(str(self.root/'failed-pending.pt')+'.work.json').exists())
        self.assertTrue(io.snapshot()['failed_tickets'])
        self.assertEqual(work.snapshot()['reserved']['collections'],1)

    def test_native_hooked_failed_publication_retains_last_durable_pointer(self):
        answer=publication_control(self.root/'publication')
        self.assertEqual(len(answer['on_commit_deliveries']),1)
        self.assertTrue(answer['old_checkpoint_unchanged'])
        self.assertFalse(answer['failed_marker_published'])
        self.assertFalse(answer['failed_external_receipt_published'])

    def test_native_cli_exhausted_inspect_precedes_payload_and_model(self):
        answer=cli_control(self.root/'cli-exhausted',exhausted=True)
        self.assertEqual(answer['identity_passes'],2)

    def test_native_cli_successful_inspect_records_actual_identity_before_model(self):
        answer=cli_control(self.root/'cli-inspected',exhausted=False)
        self.assertIn('verified_resume_snapshot',answer['run_identity'])

    def test_actual_cli_phase_horizon_pending_terminal_precedes_output_and_backends(self):
        for index,fields in enumerate([dict(phase='unknown'),dict(completed_updates=5),
                                      dict(phase='pending',completed_updates=4)]):
            answer=cli_control(self.root/f'cli-phase-{index}',bootstrap_fields=fields)
            self.assertEqual(answer['identity_passes'],0)

    def test_actual_cli_direct_and_hardlink_identity_role_aliases_refuse(self):
        for role in ('template','environment-lock','parent-artifact'):
            for hardlink in (False,True):
                if role=='parent-artifact' and not hardlink:continue
                answer=cli_control(self.root/f'cli-alias-{role}-{hardlink}',alias_role=role,hardlink=hardlink)
                self.assertEqual(answer['identity_passes'],0)
                self.assertIn('aliases a generic identity role',answer['failure']['message'])

    def test_missing_payload_not_silently_hashed_by_metadata_alias_guard(self):
        parent=self.root/'placeholder';parent.mkdir()
        args=SimpleNamespace(resume=self.root/'missing-payload',model_dir=parent,
                             template=None,environment_lock=ROOT/'uv.lock')
        with patch.object(r,'file_digest') as reads:r.reject_payload_identity_aliases(args)
        reads.assert_not_called()


def main():
    if len(sys.argv)==4 and sys.argv[1]=='--child':
        print(json.dumps(fresh_child(sys.argv[2],sys.argv[3]),sort_keys=True,allow_nan=False));return
    unittest.main()


if __name__=='__main__':main()
