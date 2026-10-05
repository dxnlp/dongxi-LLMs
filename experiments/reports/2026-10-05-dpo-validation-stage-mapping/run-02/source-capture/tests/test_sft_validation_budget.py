"""Original CPU semantic-validation accounting; no pretrained acquisition."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
import test_sft_runner_recovery as base
import test_sft_work_budget as previous
from dongxi_llms.run_identity import canonical_hash
from dongxi_llms.work_budget import WorkLedger,WorkBudgetExceeded

r=base.runner;ROOT=base.ROOT;BOUND=1024**2
NEW_KEYS=('recovery_validation_operations','recovery_history_rows','recovery_tensor_elements','recovery_rng_states')
CAPS=dict(train_updates=16,sampled_examples=32,selector_steps=32,valid_targets=160,
    logical_sequence_tokens=640,policy_forward_calls=128,policy_forward_positions=2048,
    evaluation_calls=96,evaluation_positions=2048,generation_calls=32,
    generation_position_upper_bound=1024,generation_tokens=128,
    recovery_validation_operations=64,recovery_history_rows=256,
    recovery_tensor_elements=10000000,recovery_rng_states=128)
SPEC='experiments/specs/2026-10-05-sft-semantic-validation-work.md'
NAMES=['scripts/run_chapter09_spark_sft.py','tests/test_sft_validation_budget.py',SPEC,
    'tests/test_sft_runner_recovery.py','tests/test_sft_work_budget.py','tests/test_spark_sft_contract.py',
    'experiments/specs/2026-10-05-validation-budget-fixture-compatibility.md',
    'src/dongxi_llms/run_identity.py','src/dongxi_llms/work_budget.py',
    'src/dongxi_llms/training_snapshot.py','src/dongxi_llms/artifact_budget.py','uv.lock']


def write(path,value):
    with Path(path).open('x') as handle:
        handle.write(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n');handle.flush();os.fsync(handle.fileno())


def fixture(seed=1212,mode='full',caps=CAPS):
    torch.set_num_threads(1);loop,old=base.make_loop(seed,mode);config=deepcopy(old['config'])
    config.update(work_budget=r.work_budget_contract(caps,BOUND),dev_sha256=canonical_hash(previous.DEV),
        work_limits_sha256=hashlib.sha256((json.dumps(caps,indent=2,sort_keys=True)+'\n').encode()).hexdigest(),
        generation_dispatch='single-sequence greedy; use_cache=True; conservative uncached reservation',generation_max_new_tokens=4)
    for name in NAMES:config['source_sha256'][name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    return loop,r.stable_sft_contract(config)


def ledger(path,contract,existing=False):
    b=contract['config']['work_budget']
    return (WorkLedger.open if existing else WorkLedger.create)(path,limits=b['limits'],
        contract_sha256=canonical_hash(contract),max_bytes=b['max_bytes'],invocation_id='original-validation-cpu')


def payload(loop,contract):
    return dict(phase='completed',completed_updates=loop.update,state=deepcopy(loop.snapshot_state()),contract=contract)


def numerical(loop):return {key:value for key,value in loop.snapshot_state().items() if key!='work_ledger'}


def lifecycle(loop,contract,output,metric_sink=lambda row:None):
    return r.run_loop_with_recovery(loop,contract=contract,output=output,parent_invocation='cpu-semantic-validation',
        max_bytes=base.LIMIT,checkpoint_every=2,metric_sink=metric_sink,
        baseline_observer=lambda:previous.observer(loop),final_observer=lambda:previous.observer(loop),exporter=lambda:None)


def receipt(output):return json.loads((Path(output)/'latest-completed-snapshot.json').read_text())


def malformed_history(loop,contract):
    p=payload(loop,contract);p['state']['history'][0]['gradient_norm']=-1.
    try:loop.validate_payload(p)
    except ValueError as error:return dict(type=type(error).__name__,message=str(error),deliberately_injected=True)
    raise AssertionError('Authored malformed same-length history was accepted')


def child(contract_path,receipt_path,journal_path,output,seed,mode,refuse_next=False):
    retained=json.loads(Path(contract_path).read_text())
    loop,contract=fixture(int(seed),mode,caps=retained['config']['work_budget']['limits'])
    if contract!=retained:raise ValueError('Fresh source contract differs')
    output=Path(output);output.mkdir(mode=0o700)
    with ledger(journal_path,contract,existing=True) as work:
        loop.bind_work_ledger(work,contract);before=work.snapshot();record=json.loads(Path(receipt_path).read_text())
        restored=base.restore(loop,record['path'],contract,record['header']);after_restore=work.snapshot()
        completed=lifecycle(loop,contract,output);after=work.snapshot()
        refusal=None
        if refuse_next:
            with patch.object(loop,'_validate_payload_semantics') as scan,patch.object(loop.model,'load_state_dict') as apply:
                try:loop.validate_payload(payload(loop,contract))
                except WorkBudgetExceeded as error:refusal=dict(type=type(error).__name__,message=str(error),semantic_scan_calls=scan.call_count,state_application_calls=apply.call_count)
                else:raise AssertionError('Extra semantic validation unexpectedly fit exhausted allowance')
            if after!=work.snapshot():raise AssertionError('Refused operation altered retained spending')
        write(output/'result.json',dict(numerical_sha256=base.state_digest(numerical(loop)),history=loop.history,
            restored_update=restored['completed_updates'],before=before,after_restore=after_restore,after=after,lifecycle=completed,extra_refusal=refusal))


class SFTValidationBudgetTests(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory(prefix='dongxi-sft-validation-');self.addCleanup(t.cleanup);self.root=Path(t.name)

    def directory(self,name):
        p=self.root/name;p.mkdir(mode=0o700);return p

    def bind(self,loop,contract,name='work'):
        work=ledger(self.directory(name)/'work.jsonl',contract);self.addCleanup(work.close)
        loop.bind_work_ledger(work,contract);return work

    def test_v2_exact_schema_refuses_budgeted_v1_without_migration(self):
        self.assertEqual(r.work_budget_contract(CAPS,BOUND)['schema'],'dongxi-sft-logical-work-v2')
        for values in ({key:value for key,value in CAPS.items() if key not in NEW_KEYS},dict(CAPS,recovery_history_rows=True),dict(CAPS,unknown=1)):
            with self.assertRaises(ValueError):r.work_budget_contract(values,BOUND)
        loop,contract=fixture();work=self.bind(loop,contract);changed=deepcopy(contract)
        changed['config']['work_budget']['schema']='dongxi-sft-logical-work-v1'
        with self.assertRaises(ValueError):loop.bind_work_ledger(work,changed)
        self.assertEqual(work.snapshot()['reserved']['recovery_validation_operations'],0)

    def test_zero_caps_refuse_before_semantic_history_tensor_rng_or_application(self):
        for key in NEW_KEYS:
            with self.subTest(key=key):
                loop,contract=fixture(caps=dict(CAPS,**{key:0}));work=self.bind(loop,contract,key)
                loop.completed_update();loop.completed_update();p=payload(loop,contract)
                with patch.object(loop,'_validate_payload_semantics') as scan,patch.object(torch,'isfinite') as tensor_scan, \
                     patch.object(torch,'Generator') as rng,patch.object(loop.model,'load_state_dict') as apply:
                    with self.assertRaises(WorkBudgetExceeded):loop.validate_payload(p)
                    scan.assert_not_called();tensor_scan.assert_not_called();rng.assert_not_called();apply.assert_not_called()
                self.assertEqual(work.snapshot()['reserved']['recovery_validation_operations'],0)

    def test_fixed_layout_units_complete_per_actual_validation(self):
        for mode in ('full','lora'):
            loop,contract=fixture(mode=mode);work=self.bind(loop,contract,mode)
            for _ in range(2):loop.completed_update()
            expected_elements=sum(t.numel() for t in loop.model.state_dict().values())
            expected_elements+=sum(t.numel() for m in loop.optimizer.state_dict()['state'].values() for t in m.values())
            expected_elements+=torch.get_rng_state().numel()
            p=payload(loop,contract);loop.validate_payload(p);first=work.snapshot()
            self.assertEqual(first['completed']['recovery_validation_operations'],1)
            self.assertEqual(first['completed']['recovery_history_rows'],2)
            self.assertEqual(first['completed']['recovery_tensor_elements'],expected_elements)
            self.assertEqual(first['completed']['recovery_rng_states'],2)
            loop.validate_payload(p);after=work.snapshot()
            for key in NEW_KEYS:self.assertEqual(after['completed'][key],2*first['completed'][key])

    def test_actual_save_once_and_restore_once_with_refreshed_saved_prefix(self):
        loop,contract=fixture();work=self.bind(loop,contract);path=self.root/'completed.pt'
        header=loop.save(path,contract=contract,parent_invocation='save',max_bytes=base.LIMIT)
        after=work.snapshot();self.assertEqual(after['completed']['recovery_validation_operations'],1)
        with patch.object(loop,'validate_payload',wraps=loop.validate_payload) as validate:
            loaded=base.restore(loop,path,contract,header)
            self.assertEqual(validate.call_count,1)
        self.assertEqual(loaded['state']['work_ledger']['sequence'],after['sequence'])
        self.assertEqual(work.snapshot()['completed']['recovery_validation_operations'],2)

    def test_same_length_history_failure_retains_full_reservation_and_partial_rows(self):
        loop,contract=fixture();work=self.bind(loop,contract)
        loop.completed_update();loop.completed_update();before=base.state_digest(numerical(loop))
        failure=malformed_history(loop,contract);after=work.snapshot()
        self.assertIn('metric',failure['message']);self.assertEqual(after['reserved']['recovery_history_rows'],2)
        self.assertEqual(after['attempted_upper']['recovery_history_rows'],1)
        self.assertEqual(after['known_partial']['recovery_history_rows'],0)
        self.assertEqual(after['reserved']['recovery_validation_operations'],1)
        self.assertEqual(after['completed']['recovery_validation_operations'],0)
        self.assertEqual(after['uncertain_upper']['recovery_validation_operations'],1)
        self.assertEqual(base.state_digest(numerical(loop)),before)

    def test_nonfinite_tensor_and_same_layout_rng_failures_retain_charges(self):
        for kind in ('tensor','python-rng','torch-rng','adam'):
            with self.subTest(kind=kind):
                loop,contract=fixture();work=self.bind(loop,contract,kind);loop.completed_update();p=payload(loop,contract)
                if kind=='tensor':p['state']['model'][next(iter(p['state']['model']))].reshape(-1)[0]=float('nan')
                elif kind=='python-rng':
                    py=p['state']['rng']['python'];p['state']['rng']['python']=(py[0],py[1][:-1]+(625,),py[2])
                elif kind=='torch-rng':p['state']['rng']['torch']=torch.zeros_like(p['state']['rng']['torch'])
                else:next(iter(p['state']['optimizer']['state'].values()))['exp_avg_sq'].fill_(-1)
                before=base.state_digest(numerical(loop))
                with patch.object(loop.model,'load_state_dict') as apply,self.assertRaises(ValueError):loop.validate_payload(p)
                apply.assert_not_called();after=work.snapshot()
                self.assertEqual(after['reserved']['recovery_validation_operations'],1)
                self.assertEqual(after['completed']['recovery_validation_operations'],0)
                self.assertTrue(after['failed_tickets']);self.assertEqual(base.state_digest(numerical(loop)),before)
                self.assertGreater(after['attempted_upper']['recovery_tensor_elements'],0)

    def test_sparse_model_layout_refused_before_elementwise_scan(self):
        loop,contract=fixture();work=self.bind(loop,contract);p=payload(loop,contract)
        name=next(iter(p['state']['model']))
        p['state']['model'][name]=p['state']['model'][name].to_sparse()
        with patch.object(torch,'isfinite') as scan,patch.object(loop.model,'load_state_dict') as apply, \
             self.assertRaisesRegex(ValueError,'tensor layout'):
            loop.validate_payload(p)
        scan.assert_not_called();apply.assert_not_called();after=work.snapshot()
        self.assertEqual(after['reserved']['recovery_validation_operations'],1)
        self.assertEqual(after['completed']['recovery_validation_operations'],0)
        self.assertTrue(after['failed_tickets'])

    def test_restore_semantic_failure_is_charged_before_model_adam_rng_application(self):
        loop,contract=fixture();work=self.bind(loop,contract);loop.completed_update();p=payload(loop,contract)
        next(iter(p['state']['optimizer']['state'].values()))['exp_avg_sq'].fill_(-1)
        path=self.root/'authored-invalid.pt'
        # Explicit negative fixture publisher, not a successful production save.
        header=r.save_snapshot(path,contract=contract,state=p['state'],completed_updates=1,parent_invocation='authored-invalid',max_bytes=base.LIMIT)
        before=base.state_digest(numerical(loop))
        with patch.object(loop.model,'load_state_dict') as model,patch.object(loop.optimizer,'load_state_dict') as adam, \
             patch.object(torch,'set_rng_state') as rng,self.assertRaisesRegex(ValueError,'second moment'):
            base.restore(loop,path,contract,header)
        model.assert_not_called();adam.assert_not_called();rng.assert_not_called()
        self.assertEqual(work.snapshot()['reserved']['recovery_validation_operations'],1)
        self.assertEqual(base.state_digest(numerical(loop)),before)

    def test_hostile_scalar_and_container_lengths_refuse_without_scans_or_reservations(self):
        loop,contract=fixture();work=self.bind(loop,contract);p=payload(loop,contract)
        variants=[]
        for value in (True,2**100,2001,-1):
            changed=deepcopy(p);changed['completed_updates']=value;variants.append(changed)
        for key,value in (('history',[{}]*2001),('order',[0]*2001),('model',{})):
            changed=deepcopy(p);changed['state'][key]=value;variants.append(changed)
        changed=deepcopy(p);py=changed['state']['rng']['python'];changed['state']['rng']['python']=(py[0],(0,)*626,py[2]);variants.append(changed)
        changed=deepcopy(p);changed['state']['rng']['cuda']=[torch.zeros(1)]*100;variants.append(changed)
        for changed in variants:
            with patch.object(loop,'_validate_payload_semantics') as scan,self.assertRaises(ValueError):loop.validate_payload(changed)
            scan.assert_not_called()
        self.assertEqual(work.snapshot()['reserved']['recovery_validation_operations'],0)

    def test_prefix_and_contract_refuse_before_runner_semantics(self):
        loop,contract=fixture();work=self.bind(loop,contract);p=payload(loop,contract)
        p['state']['work_ledger']['chain_sha256']='0'*64
        with patch.object(loop,'_validate_payload_semantics') as scan,self.assertRaises(ValueError):loop.validate_payload(p)
        scan.assert_not_called();self.assertEqual(work.snapshot()['reserved']['recovery_validation_operations'],0)

    def test_same_length_history_oversized_flat_metadata_is_charged_without_hash_scan(self):
        for kind in ('oversized-row','nested-metric','huge-scalar','loop-metadata','optimizer-metadata'):
            with self.subTest(kind=kind):
                loop,contract=fixture();work=self.bind(loop,contract,kind)
                loop.completed_update();loop.completed_update();p=payload(loop,contract)
                if kind=='oversized-row':p['state']['history'][0]={str(number):number for number in range(5000)}
                elif kind=='nested-metric':p['state']['metric']['answer_nll']=[0.]*5000
                elif kind=='huge-scalar':p['state']['metric']['cursor']=2**100000
                elif kind=='loop-metadata':
                    name=next(iter(p['state']['loop_contract']['model_layout']))
                    p['state']['loop_contract']['model_layout'][name][0]=[1]*5000
                else:p['state']['optimizer']['param_groups'][0]['betas']=[.9]*5000
                forbidden=[p['state']['metric'],p['state']['history'][-1],p['state']['history'][0],
                    p['state']['loop_contract'],p['state']['optimizer']['param_groups']]
                original=r.canonical_hash
                def checked(value):
                    if any(value is invalid for invalid in forbidden):raise AssertionError('Unbounded semantic metadata passed to hash')
                    return original(value)
                with patch.object(r,'canonical_hash',side_effect=checked),patch.object(loop.model,'load_state_dict') as apply, \
                     self.assertRaises(ValueError):loop.validate_payload(p)
                apply.assert_not_called();after=work.snapshot()
                self.assertEqual(after['reserved']['recovery_validation_operations'],1)
                self.assertEqual(after['completed']['recovery_validation_operations'],0)
                self.assertTrue(after['failed_tickets'])

    def test_exhausted_validation_allowance_refuses_actual_save_and_restore(self):
        loop,contract=fixture(caps=dict(CAPS,recovery_validation_operations=1));work=self.bind(loop,contract)
        path=self.root/'first.pt';header=loop.save(path,contract=contract,parent_invocation='one',max_bytes=base.LIMIT)
        with patch.object(loop,'_validate_payload_semantics') as scan,patch.object(loop.model,'load_state_dict') as apply:
            with self.assertRaises(WorkBudgetExceeded):base.restore(loop,path,contract,header)
            with self.assertRaises(WorkBudgetExceeded):loop.save(self.root/'never.pt',contract=contract,parent_invocation='two',max_bytes=base.LIMIT)
            scan.assert_not_called();apply.assert_not_called()
        self.assertFalse((self.root/'never.pt').exists());self.assertEqual(work.snapshot()['reserved']['recovery_validation_operations'],1)

    def test_original_equations_and_four_update_full_lora_states_are_exact(self):
        for seed in (1212,1213):
            for mode in ('full','lora'):
                with self.subTest(seed=seed,mode=mode):
                    loop,contract=fixture(seed,mode);self.bind(loop,contract,f'{seed}-{mode}')
                    history=[loop.completed_update() for _ in range(4)];loop.validate_payload(payload(loop,contract))
                    expected=base.state_digest(numerical(loop));original,_=base.make_loop(seed,mode)
                    reference=[base.original_update(original) for _ in range(4)]
                    self.assertEqual(history,reference);self.assertEqual(expected,base.state_digest(original.snapshot_state()))

    def test_fresh_full_lora_replay_keeps_later_failed_validation_and_duplicate_actual_checks(self):
        for mode in ('full','lora'):
            with self.subTest(mode=mode):
                loop,contract=fixture(mode=mode);directory=self.directory(mode);journal=directory/'work.jsonl'
                with ledger(journal,contract) as work:
                    loop.bind_work_ledger(work,contract)
                    for _ in range(2):loop.completed_update()
                    path=directory/'completed2.pt';header=loop.save(path,contract=contract,parent_invocation='parent',max_bytes=base.LIMIT)
                    failure=malformed_history(loop,contract);spent=work.snapshot()
                    for _ in range(2):loop.completed_update()
                    expected=base.state_digest(numerical(loop));history=deepcopy(loop.history)
                    # The child's same physical journal also retains this
                    # original continuation, not only the earlier bad check.
                    spent=work.snapshot()
                write(directory/'contract.json',contract);write(directory/'receipt.json',dict(path=str(path),header=header));write(directory/'failure.json',failure)
                command=[sys.executable,str(Path(__file__).resolve()),'--child',str(directory/'contract.json'),str(directory/'receipt.json'),str(journal),str(directory/'fresh'),'1212',mode]
                process=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=90)
                self.assertEqual(process.returncode,0,process.stderr)
                result=json.loads((directory/'fresh/result.json').read_text())
                self.assertEqual(result['numerical_sha256'],expected);self.assertEqual(result['history'],history)
                self.assertEqual(result['before']['reserved'],spent['reserved'])
                self.assertEqual(result['after_restore']['reserved']['recovery_validation_operations'],3)
                self.assertEqual(result['after']['reserved']['recovery_validation_operations'],5)
                self.assertEqual(result['after']['completed']['recovery_validation_operations'],4)

    def test_fresh_completed_replay_exhausts_fixed_six_validation_allowances(self):
        result=exhausted_fresh_control(self.directory('exhausted'))
        self.assertEqual(result['execution']['exit_code'],0)
        self.assertEqual(result['replay']['after']['reserved']['recovery_validation_operations'],6)
        self.assertEqual(result['replay']['extra_refusal']['semantic_scan_calls'],0)
        self.assertEqual(result['replay']['extra_refusal']['state_application_calls'],0)


def exhausted_fresh_control(directory):
    directory=Path(directory);loop,contract=fixture(caps=dict(CAPS,recovery_validation_operations=6))
    journal=directory/'work.jsonl';partial=directory/'interrupted';partial.mkdir(mode=0o700)
    def metric(row):
        if row['update']==2:raise OSError('authored exhausted-control interruption after update2')
    with ledger(journal,contract) as work:
        loop.bind_work_ledger(work,contract)
        try:lifecycle(loop,contract,partial,metric)
        except OSError as error:interruption=dict(type=type(error).__name__,message=str(error),deliberately_injected=True)
        else:raise AssertionError('Declared exhausted-control interruption absent')
        record=receipt(partial);bad=malformed_history(loop,contract);charged=work.snapshot()
    write(directory/'contract.json',contract);write(directory/'receipt.json',record)
    write(directory/'retained-failures.json',dict(interruption=interruption,validation=bad,charged=charged))
    command=[sys.executable,str(Path(__file__).resolve()),'--child',str(directory/'contract.json'),str(directory/'receipt.json'),str(journal),str(directory/'fresh'),'1212','full','--refuse-next']
    started=time.monotonic();process=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=90)
    execution=dict(command=command,exit_code=process.returncode,seconds=time.monotonic()-started,stdout=process.stdout,stderr=process.stderr)
    write(directory/'fresh-execution.json',execution)
    if process.returncode:raise AssertionError('Fresh exhausted control failed; raw diagnostic retained')
    replay=json.loads((directory/'fresh/result.json').read_text());original,_=base.make_loop()
    for _ in range(4):base.original_update(original)
    if replay['numerical_sha256']!=base.state_digest(original.snapshot_state()):raise AssertionError('Fresh exhausted control changed original numerical endpoint')
    result=dict(execution=execution,replay=replay,exact_original_numerical_replay=True)
    write(directory/'control.json',result);return result


def collect(destination):
    destination=Path(destination).resolve();destination.mkdir(parents=True,mode=0o700,exist_ok=False)
    hashes=lambda:{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in NAMES}
    before=hashes();write(destination/'source-before.json',before);write(destination/'work-limits.json',CAPS)
    command=[sys.executable,'-m','unittest','test_sft_validation_budget','test_sft_runner_recovery','test_sft_work_budget','test_spark_sft_contract','test_work_budget','-v']
    started=time.monotonic();process=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=180)
    tests=dict(command=command,exit_code=process.returncode,seconds=time.monotonic()-started,stdout=process.stdout,stderr=process.stderr)
    write(destination/'tests.json',tests)
    if process.returncode:raise AssertionError('Focused controls failed; raw diagnostic retained')
    count=int(re.search(r'Ran (\d+) tests',process.stderr).group(1));arms=[]
    for seed in (1212,1213):
        for mode in ('full','lora'):
            directory=destination/f'{seed}-{mode}';directory.mkdir(mode=0o700)
            clean,contract=fixture(seed,mode);write(directory/'contract.json',contract)
            clean_output=directory/'uninterrupted';clean_output.mkdir(mode=0o700)
            with ledger(directory/'clean-work.jsonl',contract) as work:
                clean.bind_work_ledger(work,contract);full=lifecycle(clean,contract,clean_output)
                expected=base.state_digest(numerical(clean));history=deepcopy(clean.history);clean_summary=work.snapshot()
            interrupted,contract2=fixture(seed,mode);partial=directory/'interrupted';partial.mkdir(mode=0o700);journal=directory/'retained-work.jsonl'
            def metric(row):
                if row['update']==2:raise OSError('authored interruption after durable update2')
            with ledger(journal,contract2) as work:
                interrupted.bind_work_ledger(work,contract2)
                try:lifecycle(interrupted,contract2,partial,metric)
                except OSError as error:interruption=dict(type=type(error).__name__,message=str(error),deliberately_injected=True)
                else:raise AssertionError('Declared interruption absent')
                record=receipt(partial);bad=malformed_history(interrupted,contract2);charged=work.snapshot()
            write(directory/'receipt.json',record);write(directory/'retained-failures.json',dict(interruption=interruption,validation=bad,charged=charged))
            command=[sys.executable,str(Path(__file__).resolve()),'--child',str(directory/'contract.json'),str(directory/'receipt.json'),str(journal),str(directory/'fresh'),str(seed),mode]
            started=time.monotonic();process=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=90)
            execution=dict(command=command,exit_code=process.returncode,seconds=time.monotonic()-started,stdout=process.stdout,stderr=process.stderr)
            write(directory/'fresh-execution.json',execution)
            if process.returncode:raise AssertionError('Fresh semantic-work replay failed; raw diagnostic retained')
            result=json.loads((directory/'fresh/result.json').read_text());after=result['after']
            if result['numerical_sha256']!=expected or result['history']!=history or after['reserved']['recovery_validation_operations']!=6 or after['completed']['recovery_validation_operations']!=5:
                raise AssertionError('Fresh numerical/validation spending differs')
            arm=dict(seed=seed,mode=mode,expected_numerical_sha256=expected,exact_numerical_replay=True,
                history=history,clean_summary=clean_summary,charged=charged,replay=result,execution=execution,full=full)
            write(directory/'arm.json',arm);arms.append(arm)
    exhausted=destination/'exhausted-six-operations';exhausted.mkdir(mode=0o700)
    exhausted_control=exhausted_fresh_control(exhausted)
    after=hashes();write(destination/'source-after.json',after)
    if before!=after:raise AssertionError('Source drift; all outputs retained')
    artifacts={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(destination.rglob('*')) if path.is_file()}
    result=dict(schema='dongxi-sft-semantic-validation-cpu-v1',status='pass',test_count=count,tests=tests,
        source_sha256=before,artifact_sha256=artifacts,arms=arms,exhausted_control=exhausted_control,invocation=sys.orig_argv,
        scope='actual CPU runner semantic validation only; generic load/tree/hash/serialization/application excluded',
        pending=['shared restricted-load/tree/finite envelope','physical containment','artifact/output routing','pretrained/CUDA/BF16','model-scale overhead'])
    write(destination/'verification.json',result);print(json.dumps(dict(status='pass',tests=count,arms=4,verification=str(destination/'verification.json'))))


if __name__=='__main__':
    if len(sys.argv)==8 and sys.argv[1]=='--child':child(*sys.argv[2:])
    elif len(sys.argv)==9 and sys.argv[1]=='--child' and sys.argv[-1]=='--refuse-next':child(*sys.argv[2:-1],refuse_next=True)
    elif len(sys.argv)==3 and sys.argv[1]=='--collect':collect(sys.argv[2])
    else:unittest.main()
