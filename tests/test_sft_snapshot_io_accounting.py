"""Actual CPU SFT checkpoint I/O consumer; original random fixtures only."""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
import test_sft_runner_recovery as base
import test_sft_validation_budget as semantic
import dongxi_llms.training_snapshot as shared
from dongxi_llms.run_identity import canonical_hash
from dongxi_llms.work_budget import WorkLedger,WorkBudgetExceeded
from dongxi_llms.snapshot_io_budget import (IO_KEYS,SnapshotIOBudget,io_budget_contract,
    io_ledger_contract_sha256,read_work_receipt,validate_work_receipt)

r=base.runner;ROOT=base.ROOT;LIMIT=base.LIMIT
SPEC='experiments/specs/2026-10-05-sft-snapshot-io-accounting.md'
ENVELOPE=dict(max_payload_bytes=LIMIT,max_tree_nodes=100000,max_tensor_elements=1000000,
    max_tensor_bytes=8*1024**2,max_primitive_bytes=1024**2)
IO_CAPS=dict(snapshot_inspect_operations=16,snapshot_load_operations=16,snapshot_save_operations=32,
    snapshot_hash_bytes=1024**3,snapshot_tree_nodes=6000000,snapshot_tensor_elements=48000000,
    snapshot_primitive_bytes=64*1024**2,snapshot_clone_bytes=256*1024**2,
    snapshot_serialization_bytes=512*1024**2)
NAMES=['scripts/run_chapter09_spark_sft.py','tests/test_sft_snapshot_io_accounting.py',SPEC,
    'src/dongxi_llms/snapshot_io_budget.py','src/dongxi_llms/training_snapshot.py',
    'src/dongxi_llms/work_budget.py','src/dongxi_llms/run_identity.py',
    'tests/test_sft_runner_recovery.py','tests/test_sft_validation_budget.py',
    'tests/test_sft_work_budget.py','tests/test_spark_sft_contract.py',
    'tests/snapshot_io_test_support.py',
    'experiments/specs/2026-10-05-snapshot-io-cli-fixture-compatibility.md','uv.lock']


def write(path,value):
    with Path(path).open('x') as handle:
        handle.write(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n');handle.flush();os.fsync(handle.fileno())


def make(seed=1212,mode='full',limits=IO_CAPS):
    loop,old=semantic.fixture(seed,mode);config=deepcopy(old['config'])
    contract=io_budget_contract(limits,ENVELOPE,2*1024**2)
    config.update(snapshot_io_contract=contract,snapshot_io_limits_sha256=canonical_hash(contract))
    for name in NAMES:config['source_sha256'][name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    return loop,r.stable_sft_contract(config)


def bind(stack,loop,contract,directory,*,receipt=None):
    directory=Path(directory);directory.mkdir(mode=0o700,exist_ok=True)
    budget=contract['config']['work_budget'];io=contract['config']['snapshot_io_contract']
    options=dict(limits=budget['limits'],contract_sha256=canonical_hash(contract),
        max_bytes=budget['max_bytes'],invocation_id='sft-io-reference')
    io_options=dict(limits=io['limits'],contract_sha256=io_ledger_contract_sha256(io,canonical_hash(contract)),
        max_bytes=io['max_journal_bytes'],invocation_id='sft-io-reference')
    if receipt is None:
        work=stack.enter_context(WorkLedger.create(directory/'work.jsonl',**options))
        journal=stack.enter_context(WorkLedger.create(directory/'io.jsonl',**io_options))
    else:
        work=stack.enter_context(WorkLedger.open(directory/'work.jsonl',
            expected_snapshot=receipt['runner_work_prefix'],**options))
        journal=stack.enter_context(WorkLedger.open(directory/'io.jsonl',expected_snapshot=receipt['io_prefix'],**io_options))
    loop.bind_work_ledger(work,contract)
    return work,journal,SnapshotIOBudget(journal,contract=io,scientific_contract_sha256=canonical_hash(contract),expected_receipt=receipt)


def save(loop,contract,io,path):
    header=loop.save(path,contract=contract,parent_invocation='original-io-cpu',max_bytes=LIMIT,
        io_budget=io,work_receipt_path=Path(str(path)+'.work.json'))
    return header,read_work_receipt(Path(str(path)+'.work.json'))


def inspect(path,receipt,contract,io):
    return r.inspect_snapshot(path,expected_sha256=receipt['snapshot']['payload_sha256'],
        expected_bytes=receipt['snapshot']['payload_bytes'],expected_contract=contract,max_bytes=LIMIT,io_budget=io)


def restore(loop,path,receipt,contract,io):
    return loop.restore(path,contract=contract,expected_sha256=receipt['snapshot']['payload_sha256'],
        expected_bytes=receipt['snapshot']['payload_bytes'],max_bytes=LIMIT,io_budget=io)


def lifecycle(loop,contract,io,output,metric=lambda row:None,baseline=None):
    output=Path(output);output.mkdir(mode=0o700)
    return r.run_loop_with_recovery(loop,contract=contract,output=output,parent_invocation='original-io-cpu',
        max_bytes=LIMIT,checkpoint_every=2,metric_sink=metric,
        baseline_observer=baseline or (lambda:semantic.previous.observer(loop)),
        final_observer=lambda:semantic.previous.observer(loop),exporter=lambda:None,io_budget=io)


def failed_load(loop,path,receipt,contract,io):
    io.bind_receipt(receipt);before=io.ledger.snapshot();original=base.state_digest(semantic.numerical(loop))
    with patch.object(shared.torch,'load',side_effect=RuntimeError('authored restricted-load failure after full hash')):
        try:restore(loop,path,receipt,contract,io)
        except RuntimeError as error:
            result=dict(type=type(error).__name__,message=str(error),deliberately_injected=True,
                before=before,after=io.ledger.snapshot(),numerical_unchanged=base.state_digest(semantic.numerical(loop))==original)
            if not result['numerical_unchanged']:raise AssertionError('Failed load applied numerical state')
            return result
    raise AssertionError('Declared restricted-load failure absent')


def child(contract_path,latest_path,journal_directory,output,seed,mode):
    expected=json.loads(Path(contract_path).read_text());latest=json.loads(Path(latest_path).read_text())
    receipt=read_work_receipt(latest['work_receipt_path']);loop,contract=make(int(seed),mode,
        expected['config']['snapshot_io_contract']['limits'])
    if contract!=expected:raise ValueError('Fresh source/scientific contract differs')
    with ExitStack() as stack:
        work,journal,io=bind(stack,loop,contract,journal_directory,receipt=receipt)
        before=dict(work=work.snapshot(),io=journal.snapshot())
        inspect(latest['path'],receipt,contract,io);after_inspect=journal.snapshot()
        restore(loop,latest['path'],receipt,contract,io);restored=loop.update
        after_load=dict(work=work.snapshot(),io=journal.snapshot())
        result=lifecycle(loop,contract,io,output)
        write(Path(output)/'replay.json',dict(restored_update=restored,
            numerical_sha256=base.state_digest(semantic.numerical(loop)),history=loop.history,
            before=before,after_inspect=after_inspect,after_load=after_load,
            after=dict(work=work.snapshot(),io=journal.snapshot()),lifecycle=result))


class CapturedScience(Exception):
    def __init__(self,contract):self.contract=contract


def cli_patches(stack,parent,model):
    """Explicit mocked hardware/parent resolution, never CUDA/pretrained work."""
    from transformers import AutoTokenizer,AutoModelForCausalLM
    for name,value in (('is_available',True),('is_bf16_supported',True),
                       ('get_device_name','authored metadata mock; no GPU allocation')):
        stack.enter_context(patch.object(r.torch.cuda,name,return_value=value))
    stack.enter_context(patch.object(r.torch.cuda,'manual_seed_all'))
    stack.enter_context(patch.object(r,'available_gib',return_value=30.))
    stack.enter_context(patch.object(r,'prepare_model_snapshot',return_value=parent))
    stack.enter_context(patch.object(r,'cached_snapshot',return_value=parent))
    token=stack.enter_context(patch.object(AutoTokenizer,'from_pretrained',side_effect=lambda *a,**k:base.tokenizer()))
    loading=stack.enter_context(patch.object(AutoModelForCausalLM,'from_pretrained',side_effect=model))
    return token,loading


def cli_setup(directory,limits=IO_CAPS):
    directory=Path(directory);directory.mkdir(mode=0o700)
    parent=directory/'authored-parent-metadata';parent.mkdir(mode=0o700)
    write(parent/'config.json',dict(authored_metadata_only=True))
    write(parent/'tokenizer_config.json',dict(authored_metadata_only=True))
    train=directory/'train.jsonl';dev=directory/'dev.jsonl';template=directory/'template.jinja'
    train.write_text('\n'.join(json.dumps(dict(row,group='authored-train')) for row in base.records())+'\n')
    dev.write_text('\n'.join(json.dumps(dict(row,group='authored-dev')) for row in semantic.previous.DEV)+'\n')
    template.write_text(base.TEMPLATE)
    work_caps=directory/'work-caps.json';write(work_caps,semantic.CAPS)
    io_caps=directory/'io-caps.json';write(io_caps,io_budget_contract(limits,ENVELOPE,2*1024**2))
    argv=['--revision','1'*40,'--tokenizer-revision','1'*40,'--template',str(template),
        '--train',str(train),'--dev',str(dev),'--output',str(directory/'capture'),
        '--environment-lock',str(ROOT/'uv.lock'),'--snapshot-max-bytes',str(LIMIT),
        '--work-limits',str(work_caps),'--work-journal-max-bytes',str(semantic.BOUND),
        '--snapshot-io-limits',str(io_caps),'--snapshot-io-ledger',str(directory/'unused-io.jsonl'),
        '--updates','4','--microbatch','1','--accumulation','2','--max-length','64',
        '--learning-rate','.003','--rank','2']
    original=r.stable_sft_contract
    def capture(config):raise CapturedScience(original(config))
    with ExitStack() as stack:
        token,model=cli_patches(stack,parent,AssertionError('Contract capture must precede model allocation'))
        stack.enter_context(patch.object(r,'stable_sft_contract',side_effect=capture))
        try:r.main(argv)
        except CapturedScience as error:contract=error.contract
        else:raise AssertionError('No-model scientific contract capture absent')
        if model.call_count:raise AssertionError('Capture allocated a model')
    write(directory/'independent-contract.json',contract)
    loop,_=base.make_loop()
    with ExitStack() as stack:
        work,journal,io=bind(stack,loop,contract,directory/'retained-journals')
        _,receipt=save(loop,contract,io,directory/'checkpoint.pt')
    resume=list(argv);resume[resume.index('--output')+1]=str(directory/'resume-output')
    resume[resume.index('--snapshot-io-ledger')+1]=str(directory/'retained-journals/io.jsonl')
    resume+=['--resume',str(directory/'checkpoint.pt'),'--resume-contract',str(directory/'independent-contract.json'),
        '--resume-sha256',receipt['snapshot']['payload_sha256'],'--resume-bytes',str(receipt['snapshot']['payload_bytes']),
        '--work-journal',str(directory/'retained-journals/work.jsonl'),
        '--resume-io-receipt',str(directory/'checkpoint.pt.work.json')]
    return dict(argv=resume,parent=parent,contract=contract,receipt=receipt,directory=directory,
        checkpoint=directory/'checkpoint.pt',io_caps=io_caps)


class SFTSnapshotIOTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='dongxi-sft-io-')
        self.addCleanup(temporary.cleanup);self.root=Path(temporary.name)
        self.stack=ExitStack();self.addCleanup(self.stack.close)

    def fixture(self,seed=1212,mode='full',limits=IO_CAPS,name='journals'):
        loop,contract=make(seed,mode,limits)
        return (loop,contract,*bind(self.stack,loop,contract,self.root/name))

    def test_separate_v1_io_does_not_change_existing_sixteen_caps(self):
        loop,contract,work,journal,io=self.fixture()
        self.assertEqual(work.limits,semantic.CAPS);self.assertEqual(len(r.WORK_KEYS),16)
        self.assertEqual(journal.limits,IO_CAPS);self.assertEqual(len(IO_KEYS),9)
        self.assertNotEqual(work.snapshot()['file_identity'],journal.snapshot()['file_identity'])
        self.assertEqual(contract['config']['snapshot_io_contract']['schema'],'dongxi-snapshot-io-work-v1')
        with self.assertRaises(ValueError):io_budget_contract(dict(IO_CAPS,unknown=1),ENVELOPE,1024**2)

    def test_actual_save_publishes_exact_independent_prefixes(self):
        loop,contract,work,journal,io=self.fixture();before=journal.snapshot()
        header,receipt=save(loop,contract,io,self.root/'state.pt')
        self.assertEqual(header['schema_version'],2);self.assertEqual(receipt['snapshot'],header)
        self.assertEqual(receipt['io_prefix'],before)
        self.assertEqual(receipt['runner_work_prefix'],work.snapshot())
        self.assertEqual(journal.snapshot()['reserved']['snapshot_save_operations'],1)
        self.assertGreater(journal.snapshot()['completed']['snapshot_tensor_elements'],0)
        self.assertGreater(journal.snapshot()['completed']['snapshot_clone_bytes'],0)
        self.assertEqual(work.snapshot()['reserved']['recovery_validation_operations'],1)

    def test_declared_io_hook_omission_and_mismatch_refuse_before_capture_or_load(self):
        loop,contract,work,journal,io=self.fixture()
        for operation in ('save','restore'):
            with self.subTest(operation=operation),patch.object(loop,'snapshot_state') as capture, \
                 patch.object(loop,'validate_payload') as semantics,patch.object(r,'load_snapshot') as load, \
                 self.assertRaisesRegex(ValueError,'matching bound hook'):
                if operation=='save':loop.save(self.root/'never.pt',contract=contract,parent_invocation='bad',max_bytes=LIMIT)
                else:loop.restore(self.root/'never.pt',contract=contract,expected_sha256='1'*64,expected_bytes=1,max_bytes=LIMIT)
            capture.assert_not_called();semantics.assert_not_called();load.assert_not_called()
        wrong=deepcopy(contract);wrong['config']['seed']=999
        with patch.object(loop,'snapshot_state') as capture,self.assertRaisesRegex(ValueError,'matching bound hook'):
            loop.save(self.root/'wrong.pt',contract=wrong,parent_invocation='bad',max_bytes=LIMIT,io_budget=io)
        capture.assert_not_called();self.assertEqual(journal.snapshot()['reserved']['snapshot_save_operations'],0)

    def test_cli_exhausted_inspect_binds_both_journals_before_model_and_never_generic_hashes_payload(self):
        case=cli_setup(self.root/'cli-exhausted',dict(IO_CAPS,snapshot_inspect_operations=0))
        actual=r.collect_run_identity;generic=[];calls=[];original_open=WorkLedger.open
        def identity(*args,**kwargs):
            generic.append([os.path.abspath(path) for path in kwargs['input_files']]);return actual(*args,**kwargs)
        def opened(*args,**kwargs):
            result=original_open(*args,**kwargs);calls.append(dict(bound=result.bound,path=str(args[0])));return result
        with ExitStack() as stack:
            token,model=cli_patches(stack,case['parent'],AssertionError('No model after refused inspection'))
            stack.enter_context(patch.object(r,'collect_run_identity',side_effect=identity))
            stack.enter_context(patch.object(WorkLedger,'open',side_effect=opened))
            read=stack.enter_context(patch.object(shared,'_open_regular'))
            stack.enter_context(self.assertRaises(WorkBudgetExceeded));r.main(case['argv'])
        model.assert_not_called();read.assert_not_called();self.assertEqual(len(generic),2)
        forbidden={os.path.abspath(case['checkpoint']),os.path.abspath(case['directory']/'independent-contract.json'),
            os.path.abspath(case['directory']/'checkpoint.pt.work.json'),os.path.abspath(case['io_caps'])}
        self.assertTrue(all(not forbidden.intersection(paths) for paths in generic))
        self.assertEqual(len(calls),2);self.assertTrue(all(row['bound'] for row in calls))

    def test_cli_successful_inspect_records_actual_header_before_model_sentinel(self):
        case=cli_setup(self.root/'cli-inspected')
        with ExitStack() as stack:
            token,model=cli_patches(stack,case['parent'],RuntimeError('authored no-model allocation sentinel'))
            stack.enter_context(self.assertRaisesRegex(RuntimeError,'no-model allocation sentinel'));r.main(case['argv'])
        self.assertEqual(model.call_count,1)
        identity=json.loads(next((case['directory']/'resume-output').glob('identity-*.json')).read_text())['identity']
        self.assertEqual(identity['verified_resume_snapshot']['header'],case['receipt']['snapshot'])
        self.assertIn('independently declared',identity['resume_payload_expectation']['evidence'])
        self.assertIn('actual accounted',identity['verified_resume_snapshot']['evidence'])

    def test_cli_over_horizon_receipt_refuses_before_journals_payload_or_tokenizer(self):
        case=cli_setup(self.root/'cli-horizon');receipt=deepcopy(case['receipt']);receipt['snapshot']['completed_updates']=5
        forged=case['directory']/'authored-over-horizon.json';write(forged,receipt)
        argv=list(case['argv']);argv[argv.index('--resume-io-receipt')+1]=str(forged)
        with ExitStack() as stack:
            token,model=cli_patches(stack,case['parent'],AssertionError('No model after metadata refusal'))
            opened=stack.enter_context(patch.object(WorkLedger,'open'))
            inspected=stack.enter_context(patch.object(r,'inspect_snapshot'))
            stack.enter_context(self.assertRaisesRegex(ValueError,'Independent SFT receipt'));r.main(argv)
        for spy in (token,model,opened,inspected):spy.assert_not_called()
        self.assertFalse((case['directory']/'resume-output').exists())

    def test_cli_postparse_io_cap_fifo_refuses_before_tokenizer(self):
        case=cli_setup(self.root/'cli-fifo');original=r.collect_run_identity;calls=[]
        def capture(*args,**kwargs):
            result=original(*args,**kwargs);calls.append(kwargs['input_files'])
            if len(calls)==1:
                case['io_caps'].rename(case['directory']/'retained-original-io-caps.json')
                os.mkfifo(case['io_caps'],0o600)
            return result
        with ExitStack() as stack:
            token,model=cli_patches(stack,case['parent'],AssertionError('No model after FIFO refusal'))
            stack.enter_context(patch.object(r,'collect_run_identity',side_effect=capture))
            opened=stack.enter_context(patch.object(WorkLedger,'open'))
            stack.enter_context(self.assertRaisesRegex(ValueError,'bounded regular'));r.main(case['argv'])
        for spy in (token,model,opened):spy.assert_not_called()
        self.assertEqual(len(calls),1)

    def test_early_load_guard_failure_retains_admission_without_invented_hash_visits(self):
        loop,contract,work,journal,io=self.fixture();path=self.root/'state.pt';_,receipt=save(loop,contract,io,path)
        io.bind_receipt(receipt);before=journal.snapshot()
        with patch.object(loop,'guard',side_effect=TimeoutError('authored guard before payload bytes')), \
             self.assertRaises(TimeoutError):restore(loop,path,receipt,contract,io)
        after=journal.snapshot();self.assertEqual(after['reserved']['snapshot_load_operations'],1)
        self.assertEqual(after['known_partial']['snapshot_hash_bytes'],0)
        self.assertEqual(after['attempted_upper']['snapshot_hash_bytes'],before['attempted_upper']['snapshot_hash_bytes'])
        self.assertEqual(after['reserved']['snapshot_hash_bytes']-before['reserved']['snapshot_hash_bytes'],receipt['snapshot']['payload_bytes'])
        self.assertEqual(after['uncertain_upper']['snapshot_load_operations'],1)

    def test_optional_unhooked_cpu_reference_remains_schema_one(self):
        loop,contract=base.make_loop();header=loop.save(self.root/'legacy.pt',contract=contract,
            parent_invocation='unbudgeted-original',max_bytes=LIMIT)
        self.assertEqual(header['schema_version'],1)
        resumed,_=base.make_loop();base.restore(resumed,self.root/'legacy.pt',contract,header)
        self.assertEqual(resumed.update,0)

    def test_zero_inspect_load_and_save_caps_refuse_shared_work(self):
        for operation in ('inspect','load','save'):
            with self.subTest(operation=operation):
                limits=dict(IO_CAPS,**{'snapshot_'+operation+'_operations':0})
                loop,contract,work,journal,io=self.fixture(limits=limits,name=operation)
                path=self.root/(operation+'.pt')
                if operation!='save':
                    _,receipt=save(loop,contract,io,path);io.bind_receipt(receipt)
                before=journal.snapshot()
                with (patch.object(shared,'_open_regular') as read,
                        patch.object(shared,'_safe_tree') as tree,patch.object(shared.torch,'load') as load,
                        patch.object(shared.torch,'save') as serialized,self.assertRaises(WorkBudgetExceeded)):
                    if operation=='save':save(loop,contract,io,path)
                    elif operation=='inspect':inspect(path,receipt,contract,io)
                    else:restore(loop,path,receipt,contract,io)
                for spy in (read,tree,load,serialized):spy.assert_not_called()
                self.assertEqual(journal.snapshot(),before)
                if operation=='save':self.assertFalse(path.exists())

    def test_repeated_real_inspection_and_load_calls_are_charged(self):
        loop,contract,work,journal,io=self.fixture();path=self.root/'state.pt'
        _,receipt=save(loop,contract,io,path);io.bind_receipt(receipt)
        for _ in range(2):inspect(path,receipt,contract,io);restore(loop,path,receipt,contract,io)
        after=journal.snapshot()
        self.assertEqual(after['completed']['snapshot_inspect_operations'],2)
        self.assertEqual(after['completed']['snapshot_load_operations'],2)
        self.assertEqual(work.snapshot()['completed']['recovery_validation_operations'],3)
        self.assertEqual(after['completed']['snapshot_hash_bytes'],5*receipt['snapshot']['payload_bytes'])

    def test_known_other_runner_prefix_rejected_before_semantics_or_application(self):
        loop,contract,work,journal,io=self.fixture();old=work.snapshot();path=self.root/'state.pt'
        _,receipt=save(loop,contract,io,path);receipt['runner_work_prefix']=old;io.bind_receipt(receipt)
        with patch.object(loop,'validate_payload') as semantic_spy,patch.object(loop.model,'load_state_dict') as applied, \
             self.assertRaisesRegex(ValueError,'runner-work prefix'):
            restore(loop,path,receipt,contract,io)
        semantic_spy.assert_not_called();applied.assert_not_called()
        self.assertEqual(journal.snapshot()['reserved']['snapshot_load_operations'],1)
        self.assertTrue(journal.snapshot()['failed_tickets'])

    def test_copied_journals_cannot_reset_physical_receipt_binding(self):
        loop,contract,work,journal,io=self.fixture();_,receipt=save(loop,contract,io,self.root/'state.pt')
        for name,ledger,prefix in (('work',work,receipt['runner_work_prefix']),('io',journal,receipt['io_prefix'])):
            copied=self.root/(name+'-copied.jsonl');shutil.copyfile(ledger.path,copied);copied.chmod(0o600)
            with self.subTest(name=name),self.assertRaises(ValueError):
                WorkLedger.open(copied,limits=ledger.limits,contract_sha256=ledger.contract_sha256,
                    max_bytes=ledger.max_bytes,invocation_id='copied',expected_snapshot=prefix)

    def test_failed_load_retains_known_hash_and_uncertain_attempt(self):
        loop,contract,work,journal,io=self.fixture();path=self.root/'state.pt';_,receipt=save(loop,contract,io,path)
        before_work=work.snapshot();failure=failed_load(loop,path,receipt,contract,io);after=failure['after']
        self.assertTrue(failure['numerical_unchanged']);self.assertEqual(work.snapshot(),before_work)
        self.assertEqual(after['reserved']['snapshot_load_operations'],1)
        self.assertEqual(after['completed']['snapshot_load_operations'],0)
        self.assertEqual(after['known_partial']['snapshot_hash_bytes'],receipt['snapshot']['payload_bytes'])
        self.assertEqual(after['uncertain_upper']['snapshot_load_operations'],1)

    def test_failed_save_retains_partial_file_and_prior_receipt(self):
        loop,contract,work,journal,io=self.fixture();path=self.root/'first.pt';_,receipt=save(loop,contract,io,path)
        prior=path.read_bytes();loop.completed_update();partial=self.root/'partial.pt'
        with patch.object(shared.torch,'save',side_effect=OSError('authored serialization failure')),self.assertRaises(OSError):
            save(loop,contract,io,partial)
        self.assertTrue(partial.exists());self.assertFalse(Path(str(partial)+'.commit.json').exists())
        self.assertEqual(path.read_bytes(),prior);self.assertEqual(read_work_receipt(Path(str(path)+'.work.json')),receipt)
        self.assertEqual(journal.snapshot()['reserved']['snapshot_save_operations'],2)
        self.assertEqual(journal.snapshot()['completed']['snapshot_save_operations'],1)

    def test_observer_failure_keeps_initial_accounted_snapshot(self):
        loop,contract,work,journal,io=self.fixture()
        def failure():raise OSError('authored baseline observer failure')
        with self.assertRaises(OSError):lifecycle(loop,contract,io,self.root/'output',baseline=failure)
        latest=json.loads((self.root/'output/latest-completed-snapshot.json').read_text())
        receipt=read_work_receipt(latest['work_receipt_path']);self.assertEqual(receipt['snapshot']['completed_updates'],0)
        self.assertEqual(journal.snapshot()['completed']['snapshot_save_operations'],1)

    def test_bootstrap_mutation_and_fifo_refuse_without_generic_hashing(self):
        for kind in ('changed','fifo'):
            with self.subTest(kind=kind):
                path=self.root/(kind+'.json');write(path,io_budget_contract(IO_CAPS,ENVELOPE,2*1024**2))
                _,metadata=r.snapshot_bootstrap(path);path.rename(self.root/(kind+'-original.json'))
                if kind=='fifo':os.mkfifo(path,0o600)
                else:write(path,dict(changed=True))
                start=time.monotonic()
                with self.assertRaises(ValueError):r.bind_snapshot_bootstrap({},dict(snapshot_io_limits=metadata))
                self.assertLess(time.monotonic()-start,1)

    def test_budgeted_full_and_lora_match_original_equations(self):
        for seed in (1212,1213):
            for mode in ('full','lora'):
                with self.subTest(seed=seed,mode=mode):
                    loop,contract,work,journal,io=self.fixture(seed,mode,name=f'{seed}-{mode}')
                    rows=[loop.completed_update() for _ in range(4)];actual=base.state_digest(semantic.numerical(loop))
                    original,_=base.make_loop(seed,mode)
                    self.assertEqual(rows,[base.original_update(original) for _ in range(4)])
                    self.assertEqual(actual,base.state_digest(original.snapshot_state()))

    def test_fresh_full_and_lora_replay_preserves_later_failed_io(self):
        for mode in ('full','lora'):
            with self.subTest(mode=mode):
                result=run_arm(self.root/mode,1212,mode)
                self.assertTrue(result['exact_numerical_replay']);self.assertEqual(result['execution']['exit_code'],0)
                self.assertEqual(result['replay']['restored_update'],2)
                self.assertEqual(result['replay']['after']['io']['reserved']['snapshot_load_operations'],2)
                self.assertEqual(result['replay']['after']['io']['completed']['snapshot_load_operations'],1)
                self.assertTrue(result['replay']['after']['io']['failed_tickets'])


def run_arm(directory,seed,mode):
    directory=Path(directory);directory.mkdir(mode=0o700)
    clean,contract=make(seed,mode);write(directory/'contract.json',contract)
    with ExitStack() as stack:
        work,journal,io=bind(stack,clean,contract,directory/'clean-journals')
        full=lifecycle(clean,contract,io,directory/'uninterrupted')
        expected=base.state_digest(semantic.numerical(clean));history=deepcopy(clean.history)
        clean_ledgers=dict(work=work.snapshot(),io=journal.snapshot())
    loop,again=make(seed,mode)
    def metric(row):
        if row['update']==2:raise OSError('authored interruption after durable update2')
    with ExitStack() as stack:
        work,journal,io=bind(stack,loop,again,directory/'retained-journals')
        try:lifecycle(loop,again,io,directory/'interrupted',metric=metric)
        except OSError as error:interruption=dict(type=type(error).__name__,message=str(error),deliberately_injected=True)
        else:raise AssertionError('Declared interruption absent')
        latest_path=directory/'interrupted/latest-completed-snapshot.json'
        latest=json.loads(latest_path.read_text());receipt=read_work_receipt(latest['work_receipt_path'])
        failure=failed_load(loop,latest['path'],receipt,again,io)
        charged=dict(work=work.snapshot(),io=journal.snapshot())
    write(directory/'retained-failures.json',dict(interruption=interruption,load=failure,charged=charged))
    command=[sys.executable,str(Path(__file__).resolve()),'--child',str(directory/'contract.json'),
        str(latest_path),str(directory/'retained-journals'),str(directory/'fresh'),str(seed),mode]
    start=time.monotonic();process=subprocess.run(command,cwd=ROOT,env=dict(os.environ,PYTHONPATH='src:tests',
        CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1'),
        capture_output=True,text=True,timeout=90)
    execution=dict(command=command,exit_code=process.returncode,seconds=time.monotonic()-start,stdout=process.stdout,stderr=process.stderr)
    write(directory/'fresh-execution.json',execution)
    if process.returncode:raise AssertionError('Fresh accounted I/O replay failed; raw evidence retained')
    replay=json.loads((directory/'fresh/replay.json').read_text())
    if replay['numerical_sha256']!=expected or replay['history']!=history:raise AssertionError('Fresh I/O replay changed numerical recipe')
    result=dict(seed=seed,mode=mode,exact_numerical_replay=True,expected_numerical_sha256=expected,
        history=history,clean_ledgers=clean_ledgers,charged=charged,replay=replay,execution=execution,full=full)
    write(directory/'arm.json',result);return result


def run_cli_control(directory,kind):
    limits=dict(IO_CAPS,snapshot_inspect_operations=0) if kind=='exhausted' else IO_CAPS
    case=cli_setup(directory,limits);argv=list(case['argv']);generic=[];opened=[];inspection=[]
    if kind=='horizon':
        receipt=deepcopy(case['receipt']);receipt['snapshot']['completed_updates']=5
        forged=case['directory']/'authored-over-horizon.json';write(forged,receipt)
        argv[argv.index('--resume-io-receipt')+1]=str(forged)
    original_identity=r.collect_run_identity;original_open=WorkLedger.open;original_inspect=r.inspect_snapshot
    def identity(*args,**kwargs):
        generic.append([os.path.abspath(path) for path in kwargs['input_files']]);result=original_identity(*args,**kwargs)
        if kind=='fifo' and len(generic)==1:
            case['io_caps'].rename(case['directory']/'retained-original-io-caps.json');os.mkfifo(case['io_caps'],0o600)
        return result
    def opening(*args,**kwargs):
        ledger=original_open(*args,**kwargs);opened.append(dict(path=str(args[0]),bound=ledger.bound));return ledger
    def inspecting(*args,**kwargs):
        inspection.append(dict(path=str(args[0]),bound=kwargs['io_budget'].ledger.bound,
            ledger_before=kwargs['io_budget'].ledger.snapshot()));return original_inspect(*args,**kwargs)
    start=time.monotonic();failure=None
    with ExitStack() as stack:
        token,model=cli_patches(stack,case['parent'],RuntimeError('authored no-model allocation sentinel'))
        stack.enter_context(patch.object(r,'collect_run_identity',side_effect=identity))
        stack.enter_context(patch.object(WorkLedger,'open',side_effect=opening))
        stack.enter_context(patch.object(r,'inspect_snapshot',side_effect=inspecting))
        try:r.main(argv)
        except (ValueError,RuntimeError,WorkBudgetExceeded) as error:
            failure=dict(type=type(error).__name__,message=str(error),deliberately_injected_or_declared_negative=True)
        else:raise AssertionError('No-model CLI control unexpectedly reached completion')
    if kind=='inspected' and model.call_count!=1:raise AssertionError('Successful inspection sentinel absent')
    if kind!='inspected' and model.call_count:raise AssertionError('Negative control reached model loader')
    expected_inspections=0 if kind in ('horizon','fifo') else 1
    if len(inspection)!=expected_inspections:raise AssertionError('CLI inspection ordering differs')
    forbidden={os.path.abspath(case['checkpoint']),os.path.abspath(case['directory']/'independent-contract.json'),
        os.path.abspath(case['directory']/'checkpoint.pt.work.json'),os.path.abspath(case['io_caps'])}
    if any(forbidden.intersection(paths) for paths in generic):raise AssertionError('Hidden generic snapshot/bootstrap hash')
    with ExitStack() as stack:
        loop,_=base.make_loop();work,journal,io=bind(stack,loop,case['contract'],case['directory']/'retained-journals',receipt=case['receipt'])
        retained=dict(work=work.snapshot(),io=journal.snapshot())
    result=dict(kind=kind,argv=argv,seconds=time.monotonic()-start,error=failure,
        generic_input_paths=generic,payload_and_bootstrap_excluded=True,
        model_loader_calls=model.call_count,tokenizer_loader_calls=token.call_count,
        journal_opens=opened,inspections=inspection,retained=retained,
        boundary='actual CLI order with mocked hardware/parent metadata and CPU-authored snapshot; no pretrained or GPU allocation')
    write(case['directory']/'control.json',result);return result


def collect(destination):
    destination=Path(destination).resolve();destination.mkdir(parents=True,mode=0o700,exist_ok=False)
    hashes=lambda:{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in NAMES}
    before=hashes();write(destination/'source-before.json',before)
    write(destination/'io-contract.json',io_budget_contract(IO_CAPS,ENVELOPE,2*1024**2))
    command=[sys.executable,'-m','unittest','test_sft_snapshot_io_accounting','test_sft_runner_recovery',
        'test_sft_validation_budget','test_sft_work_budget','test_spark_sft_contract','-v']
    start=time.monotonic();process=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=240)
    tests=dict(command=command,exit_code=process.returncode,seconds=time.monotonic()-start,stdout=process.stdout,stderr=process.stderr)
    write(destination/'tests.json',tests)
    if process.returncode:raise AssertionError('Focused regression failed; actual output retained')
    count=int(re.search(r'Ran (\d+) tests',process.stderr).group(1))
    arms=[run_arm(destination/f'{seed}-{mode}',seed,mode) for seed in (1212,1213) for mode in ('full','lora')]
    cli_controls=[run_cli_control(destination/('cli-'+kind),kind) for kind in ('exhausted','inspected','horizon','fifo')]
    after=hashes();write(destination/'source-after.json',after)
    if before!=after:raise AssertionError('Source drift; raw collection retained')
    artifacts={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(destination.rglob('*')) if path.is_file()}
    verification=dict(schema='dongxi-sft-snapshot-io-cpu-v1',status='pass',test_count=count,tests=tests,
        source_sha256=before,artifact_sha256=artifacts,arms=arms,cli_controls=cli_controls,invocation=sys.orig_argv,
        scope='actual SFT shared snapshot I/O + separate16-dimension semantic work; trusted-local CPU only',
        pending=['caller capture/inventory','metadata/journal costs','physical containment','broader output routing','pretrained/CUDA/BF16'])
    write(destination/'verification.json',verification)
    print(json.dumps(dict(status='pass',tests=count,arms=4,verification=str(destination/'verification.json'))))


if __name__=='__main__':
    if len(sys.argv)==8 and sys.argv[1]=='--child':child(*sys.argv[2:])
    elif len(sys.argv)==3 and sys.argv[1]=='--collect':collect(sys.argv[2])
    else:unittest.main()
