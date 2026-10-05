"""Original CPU full/LoRA SFT work integration and immutable reference collector."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
import test_sft_runner_recovery as base
from dongxi_llms.run_identity import canonical_hash
from dongxi_llms.work_budget import WorkLedger,WorkBudgetExceeded,WorkLedgerError
from snapshot_io_test_support import explicit_snapshot_io_args

r=base.runner;ROOT=base.ROOT;BOUND=1024**2
CAPS=dict(train_updates=16,sampled_examples=32,selector_steps=32,valid_targets=160,
    logical_sequence_tokens=640,policy_forward_calls=128,policy_forward_positions=2048,
    evaluation_calls=96,evaluation_positions=2048,generation_calls=32,
    generation_position_upper_bound=1024,generation_tokens=128,
    recovery_validation_operations=128,recovery_history_rows=4096,
    recovery_tensor_elements=2000000,recovery_rng_states=1024)
DEV=[dict(id='original-dev-0',messages=[dict(role='user',content='copy one'),dict(role='assistant',content='one')]),
     dict(id='original-dev-1',messages=[dict(role='user',content='reverse red blue'),dict(role='assistant',content='blue red')])]


def write(path,value):
    with Path(path).open('x') as handle:
        handle.write(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');handle.flush();os.fsync(handle.fileno())


def fixture(seed=1212,mode='full',limits=CAPS):
    torch.set_num_threads(1)
    loop,old=base.make_loop(seed,mode);config=deepcopy(old['config'])
    config.update(work_budget=r.work_budget_contract(limits,BOUND),dev_sha256=canonical_hash(DEV),
        work_limits_sha256=hashlib.sha256((json.dumps(limits,indent=2,sort_keys=True)+'\n').encode()).hexdigest(),
        generation_dispatch='single-sequence greedy; use_cache=True; conservative uncached reservation',generation_max_new_tokens=4)
    for name in ('src/dongxi_llms/work_budget.py','src/dongxi_llms/artifact_budget.py',
                 'tests/test_sft_runner_recovery.py','tests/test_sft_work_budget.py'):
        config['source_sha256'][name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    return loop,r.stable_sft_contract(config)


def ledger_for(path,contract,invocation='initial',existing=False):
    options=contract['config']['work_budget']
    return (WorkLedger.open if existing else WorkLedger.create)(path,limits=options['limits'],
        contract_sha256=canonical_hash(contract),max_bytes=options['max_bytes'],invocation_id=invocation)


def attach(loop,contract,ledger):loop.bind_work_ledger(ledger,contract)


def numerical(loop):
    return {key:value for key,value in loop.snapshot_state().items() if key!='work_ledger'}


def generation(loop,row_sink=None):
    tokenizer=base.tokenizer()
    prefixes=[tokenizer.apply_chat_template(row['messages'][:-1],tokenize=True,
        add_generation_prompt=True,enable_thinking=False,return_dict=False) for row in DEV]
    return r.generate_sft_panel(loop,tokenizer,DEV,prefixes,cap=4,stop_ids=[3],end_message_id=3,row_sink=row_sink)


def observer(loop,row_sink=None):
    encoded=[r.encode_record(base.tokenizer(),row,64) for row in DEV]
    return dict(nll=r.evaluate_sft_panel(loop,encoded,row_sink=row_sink),samples=generation(loop,row_sink))


def lifecycle(loop,contract,output,*,metric_sink=lambda row:None,baseline=True,final=True,exporter=lambda:None):
    return r.run_loop_with_recovery(loop,contract=contract,output=output,parent_invocation='cpu-sft-work',
        max_bytes=base.LIMIT,checkpoint_every=2,metric_sink=metric_sink,
        baseline_observer=lambda:observer(loop) if baseline else None,
        final_observer=lambda:observer(loop) if final else None,exporter=exporter)


def receipt(output):return json.loads((Path(output)/'latest-completed-snapshot.json').read_text())


def restore(loop,contract,record):return base.restore(loop,record['path'],contract,record['header'])


def interrupted(loop,contract,output,*,baseline=True):
    def metric(row):
        if row['update']==2:raise OSError('authored metric failure after durable update2')
    try:lifecycle(loop,contract,output,metric_sink=metric,baseline=baseline)
    except OSError as error:return receipt(output),dict(type=type(error).__name__,message=str(error),deliberately_injected=True)
    raise AssertionError('Declared interruption absent')


def failed_later_forward(loop):
    with patch.object(loop.model,'forward',side_effect=OSError('authored entered SFT forward failure')):
        try:loop.completed_update()
        except OSError as error:return dict(type=type(error).__name__,message=str(error),deliberately_injected=True)
    raise AssertionError('Declared entered forward failure absent')


def child(receipt_file,contract_file,journal_file,output,mode,seed):
    loop,contract=fixture(int(seed),mode);expected=json.loads(Path(contract_file).read_text())
    if contract!=expected:raise ValueError('Fresh current-source scientific contract differs')
    output=Path(output);output.mkdir(mode=0o700)
    with ledger_for(journal_file,contract,'fresh-process',existing=True) as ledger:
        before=ledger.snapshot()
        attach(loop,contract,ledger);restore(loop,contract,json.loads(Path(receipt_file).read_text()))
        write(output/'ledger-after-restore.json',ledger.snapshot())
        result=lifecycle(loop,contract,output)
        write(output/'result.json',result);write(output/'ledger-before.json',before);write(output/'ledger-after.json',ledger.snapshot())
        write(output/'numerical-digest.json',dict(sha256=base.state_digest(numerical(loop)),history=loop.history))


def fifo_control(directory):
    directory=Path(directory);fifo=directory/'fifo';os.mkfifo(fifo,0o600)
    output=directory/'never-created'
    command=[sys.executable,str(ROOT/'scripts/run_chapter09_spark_sft.py'),'--revision','1'*40,
        '--tokenizer-revision','1'*40,'--template','unused','--train','unused','--dev','unused',
        '--output',str(output),'--environment-lock','uv.lock','--work-limits',str(fifo),
        '--work-journal-max-bytes',str(BOUND)]+explicit_snapshot_io_args(directory)
    started=time.monotonic();process=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=10)
    return dict(command=command,exit_code=process.returncode,seconds=time.monotonic()-started,
        stdout=process.stdout,stderr=process.stderr,output_created=output.exists(),
        fifo_retained_before_owned_temporary_cleanup=fifo.exists(),declared_timeout_seconds=10)


def cap_mutation_control(directory,kind,phase='after_identity'):
    from transformers import AutoTokenizer,AutoModelForCausalLM
    directory=Path(directory);limits=directory/'caps.json';write(limits,CAPS)
    initial=hashlib.sha256(limits.read_bytes()).hexdigest()
    train=directory/'train.jsonl';dev=directory/'dev.jsonl';template=directory/'template.jinja'
    train.write_text('\n'.join(json.dumps(dict(row,group='authored-train')) for row in base.records())+'\n')
    dev.write_text('\n'.join(json.dumps(dict(row,group='authored-dev')) for row in DEV)+'\n');template.write_text(base.TEMPLATE)
    argv=['--revision','1'*40,'--tokenizer-revision','1'*40,'--template',str(template),'--train',str(train),
        '--dev',str(dev),'--output',str(directory/'run'),'--environment-lock',str(ROOT/'uv.lock'),
        '--snapshot-max-bytes',str(base.LIMIT),'--work-limits',str(limits),'--work-journal-max-bytes',str(BOUND)]+explicit_snapshot_io_args(directory)
    actual=r.collect_run_identity;observations=[];generic_inputs=[]
    def mutate():
        limits.rename(directory/'retained-original-caps.json')
        if kind=='fifo':os.mkfifo(limits,0o600)
        else:write(limits,dict(CAPS,train_updates=CAPS['train_updates']+1))
    def observe_then_mutate(*args,**kwargs):
        generic_inputs.append([os.path.abspath(path) for path in kwargs['input_files']])
        if not observations and phase=='before_identity':mutate()
        result=actual(*args,**kwargs);observations.append(result)
        if len(observations)==1 and phase=='after_identity':mutate()
        return result
    started=time.monotonic();caught=None
    with patch.object(r,'collect_run_identity',side_effect=observe_then_mutate),patch.object(torch.cuda,'is_available',return_value=True), \
         patch.object(torch.cuda,'is_bf16_supported',return_value=True),patch.object(torch.cuda,'manual_seed_all'), \
         patch.object(r,'available_gib',return_value=30.),patch.object(AutoTokenizer,'from_pretrained') as tokenizer, \
         patch.object(AutoModelForCausalLM,'from_pretrained') as model:
        try:r.main(argv)
        except ValueError as error:caught=dict(type=type(error).__name__,message=str(error))
        result=dict(kind=kind,phase=phase,initial_parsed_sha256=initial,raw_exception=caught,
            tokenizer_loader_calls=tokenizer.call_count,model_loader_calls=model.call_count,
            seconds=time.monotonic()-started,identity_observations=len(observations),
            generic_input_paths=generic_inputs,cap_excluded_from_generic_inputs=all(
                os.path.abspath(limits) not in paths for paths in generic_inputs),
            changed_sha256=None if kind=='fifo' else hashlib.sha256(limits.read_bytes()).hexdigest(),
            scope='CPU mocked hardware readiness only; no actual GPU/tokenizer/model acquisition')
    if caught is None or result['tokenizer_loader_calls'] or result['model_loader_calls'] or not result['cap_excluded_from_generic_inputs']:
        raise AssertionError('Changed parsed caps reached a model/tokenizer loader')
    return result


def closing_cap_control(directory,kind):
    directory=Path(directory);limits=directory/'caps.json';write(limits,CAPS)
    initial=hashlib.sha256(limits.read_bytes()).hexdigest()
    identity=r.collect_run_identity(ROOT,source_files=[Path(r.__file__)],input_files=[],
        environment_lock=ROOT/'uv.lock',config={},device={'mode':'cpu','name':'cap-only-control'})
    r.bind_bounded_work_limits(identity,limits,initial)
    retained=deepcopy(identity);limits.rename(directory/'retained-original-caps.json')
    if kind=='fifo':os.mkfifo(limits,0o600)
    else:write(limits,dict(CAPS,train_updates=CAPS['train_updates']+1))
    caught=None;started=time.monotonic()
    try:r.verify_bounded_work_limits_identity(identity,limits,initial)
    except ValueError as error:caught=dict(type=type(error).__name__,message=str(error))
    if caught is None or identity!=retained:raise AssertionError('Closing cap drift was not rejected without changing retained identity')
    return dict(kind=kind,phase='closing_identity',raw_exception=caught,seconds=time.monotonic()-started,
        retained_identity=retained,identity_unchanged=True,scope='actual cap-only closing function; no model work')


def bounded_cap_control(directory,kind,phase):
    directory=Path(directory)
    command=[sys.executable,str(Path(__file__).resolve()),'--cap-control',str(directory),kind,phase]
    started=time.monotonic();process=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=10)
    execution=dict(command=command,exit_code=process.returncode,seconds=time.monotonic()-started,
        stdout=process.stdout,stderr=process.stderr,declared_timeout_seconds=10)
    write(directory/'cap-control-execution.json',execution)
    if process.returncode:raise AssertionError('Bounded cap control failed; raw execution retained')
    return dict(execution=execution,result=json.loads((directory/'cap-control.json').read_text()))


class SFTWorkBudgetTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='dongxi-sft-work-');self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)

    def directory(self,name):
        path=self.root/name;path.mkdir(mode=0o700);return path

    def ledger(self,loop,contract,name='ledger'):
        ledger=ledger_for(self.directory(name)/'work.jsonl',contract)
        self.addCleanup(ledger.close);attach(loop,contract,ledger);return ledger

    def test_original_equations_full_and_lora_exact(self):
        for mode in ('full','lora'):
            with self.subTest(mode=mode):
                loop,contract=fixture(mode=mode);self.ledger(loop,contract,mode)
                rng=deepcopy(numerical(loop)['rng']);actual=loop.completed_update();digest=base.state_digest(numerical(loop))
                original,_=base.make_loop(mode=mode);random.setstate(rng['python']);torch.set_rng_state(rng['torch'])
                expected=base.original_update(original)
                self.assertEqual(actual,expected);self.assertEqual(digest,base.state_digest(original.snapshot_state()))

    def test_all_seeds_modes_full_recipe_unbudgeted_parity_and_lora_freeze(self):
        for seed in (1212,1213):
            for mode in ('full','lora'):
                with self.subTest(seed=seed,mode=mode):
                    loop,contract=fixture(seed,mode);ledger=self.ledger(loop,contract,f'{seed}-{mode}')
                    frozen={name:p.detach().clone() for name,p in loop.model.named_parameters() if not p.requires_grad}
                    actual=[loop.completed_update() for _ in range(4)];digest=base.state_digest(numerical(loop))
                    old,_=base.make_loop(seed,mode);expected=[old.completed_update() for _ in range(4)]
                    self.assertEqual(actual,expected);self.assertEqual(digest,base.state_digest(old.snapshot_state()))
                    self.assertEqual(ledger.snapshot()['completed']['train_updates'],4)
                    self.assertEqual(ledger.snapshot()['reserved']['valid_targets'],loop.targets)
                    self.assertEqual(ledger.snapshot()['completed']['policy_forward_positions'],loop.positions)
                    for name,value in frozen.items():self.assertTrue(torch.equal(dict(loop.model.named_parameters())[name],value))

    def test_whole_update_refusal_before_cursor_selection_or_forward(self):
        loop,contract=fixture(limits=dict(CAPS,train_updates=0));ledger=self.ledger(loop,contract)
        before=base.state_digest(numerical(loop))
        with patch.object(r,'collate') as selection,patch.object(loop.model,'forward') as forward,self.assertRaises(WorkBudgetExceeded):loop.completed_update()
        selection.assert_not_called();forward.assert_not_called();self.assertFalse(loop.incomplete_update)
        self.assertEqual(loop.cursor,0);self.assertEqual(before,base.state_digest(numerical(loop)))
        self.assertEqual(ledger.snapshot()['sequence'],0)

    def test_exact_ragged_padded_window_is_reserved_unshortened(self):
        loop,contract=fixture();ledger=self.ledger(loop,contract)
        row=loop.completed_update();state=ledger.snapshot()
        self.assertEqual(state['reserved']['valid_targets'],row['targets'])
        self.assertEqual(state['reserved']['policy_forward_positions'],row['processed_positions'])
        self.assertEqual(state['reserved']['sampled_examples'],2);self.assertEqual(state['reserved']['selector_steps'],2)
        self.assertEqual(state['reserved']['policy_forward_calls'],2)

    def test_later_failed_forward_is_spent_and_loop_poisoned(self):
        loop,contract=fixture();ledger=self.ledger(loop,contract)
        failure=failed_later_forward(loop);state=ledger.snapshot()
        self.assertIn('entered',failure['message']);self.assertTrue(loop.incomplete_update)
        with self.assertRaises(RuntimeError):loop.snapshot_state()
        self.assertEqual(state['reserved']['train_updates'],1)
        self.assertEqual(state['known_partial']['selector_steps'],2)
        self.assertEqual(state['attempted_upper']['policy_forward_calls'],1)
        self.assertEqual(state['known_partial']['policy_forward_calls'],0)
        self.assertGreater(state['uncertain_upper']['policy_forward_positions'],0)

    def test_same_process_old_snapshot_keeps_later_spending_and_baseline_recharge(self):
        full,contract=fixture();self.ledger(full,contract,'full-ledger');lifecycle(full,contract,self.directory('full'))
        digest=base.state_digest(numerical(full))
        loop,contract=fixture();ledger=self.ledger(loop,contract)
        saved,_=interrupted(loop,contract,self.directory('first'));failed_later_forward(loop)
        resumed,new_contract=fixture();attach(resumed,new_contract,ledger);restore(resumed,new_contract,saved)
        lifecycle(resumed,new_contract,self.directory('resumed'))
        self.assertEqual(digest,base.state_digest(numerical(resumed)))
        self.assertEqual(ledger.snapshot()['reserved']['train_updates'],5)
        self.assertEqual(ledger.snapshot()['completed']['train_updates'],4)
        self.assertEqual(ledger.snapshot()['reserved']['generation_calls'],6) # two baselines plus final, two prompts each

    def test_fresh_process_full_and_lora_retains_failed_spending(self):
        for mode in ('full','lora'):
            with self.subTest(mode=mode):
                full,contract=fixture(mode=mode);self.ledger(full,contract,f'{mode}-full-ledger')
                lifecycle(full,contract,self.directory(f'{mode}-full'));digest=base.state_digest(numerical(full))
                loop,contract=fixture(mode=mode);ledger=self.ledger(loop,contract,f'{mode}-journal')
                saved,_=interrupted(loop,contract,self.directory(f'{mode}-first'));failed_later_forward(loop);charged=ledger.snapshot();ledger.close()
                write(self.root/f'{mode}-receipt.json',saved);write(self.root/f'{mode}-contract.json',contract)
                command=[sys.executable,str(Path(__file__).resolve()),'--child',str(self.root/f'{mode}-receipt.json'),
                    str(self.root/f'{mode}-contract.json'),str(ledger.path),str(self.root/f'{mode}-fresh'),mode,'1212']
                process=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=90)
                self.assertEqual(process.returncode,0,process.stdout+process.stderr)
                self.assertEqual(json.loads((self.root/f'{mode}-fresh/numerical-digest.json').read_text())['sha256'],digest)
                self.assertEqual(json.loads((self.root/f'{mode}-fresh/ledger-before.json').read_text()),charged)
                restored=json.loads((self.root/f'{mode}-fresh/ledger-after-restore.json').read_text())
                validation_keys={'recovery_validation_operations','recovery_history_rows',
                    'recovery_tensor_elements','recovery_rng_states'}
                for key in CAPS:
                    if key not in validation_keys:
                        self.assertEqual(restored['reserved'][key],charged['reserved'][key])
                self.assertEqual(restored['reserved']['recovery_validation_operations']-
                    charged['reserved']['recovery_validation_operations'],1)
                final=json.loads((self.root/f'{mode}-fresh/ledger-after.json').read_text())
                self.assertEqual(final['reserved']['train_updates'],5);self.assertEqual(final['completed']['train_updates'],4)

    def test_new_output_cannot_refill_exhausted_recovery(self):
        loop,contract=fixture(limits=dict(CAPS,train_updates=3));ledger=self.ledger(loop,contract)
        saved,_=interrupted(loop,contract,self.directory('first'),baseline=False);failed_later_forward(loop)
        resumed,new=fixture(limits=dict(CAPS,train_updates=3));attach(resumed,new,ledger);restore(resumed,new,saved)
        cursor=resumed.cursor
        with patch.object(resumed.model,'forward') as forward,self.assertRaises(WorkBudgetExceeded):
            lifecycle(resumed,new,self.directory('new-output'),baseline=False,final=False)
        forward.assert_not_called();self.assertEqual(resumed.cursor,cursor);self.assertFalse(resumed.incomplete_update)
        self.assertEqual(ledger.snapshot()['reserved']['train_updates'],3)

    def test_new_ledger_and_changed_caps_reject_before_state_application(self):
        loop,contract=fixture();ledger=self.ledger(loop,contract);loop.completed_update()
        path=self.root/'state.pt';header=loop.save(path,contract=contract,parent_invocation='first',max_bytes=base.LIMIT)
        new,new_contract=fixture();other=self.ledger(new,new_contract,'other')
        with patch.object(new.model,'load_state_dict') as apply,self.assertRaises(ValueError):base.restore(new,path,new_contract,header)
        apply.assert_not_called()
        wrong,wrong_contract=fixture(limits=dict(CAPS,train_updates=17));self.ledger(wrong,wrong_contract,'wrong')
        with patch.object(wrong.model,'load_state_dict') as apply,self.assertRaises(ValueError):base.restore(wrong,path,wrong_contract,header)
        apply.assert_not_called()

    def test_whole_nll_and_generation_panel_refusal_before_network(self):
        loop,contract=fixture(limits=dict(CAPS,evaluation_calls=1));self.ledger(loop,contract)
        encoded=[r.encode_record(base.tokenizer(),row,64) for row in DEV]
        with patch.object(loop.model,'forward') as forward,self.assertRaises(WorkBudgetExceeded):r.evaluate_sft_panel(loop,encoded)
        forward.assert_not_called()
        other,contract=fixture(limits=dict(CAPS,generation_tokens=7));self.ledger(other,contract,'other')
        with patch.object(other.model,'generate') as generate,patch.object(other.model,'forward') as forward,self.assertRaises(WorkBudgetExceeded):generation(other)
        generate.assert_not_called();forward.assert_not_called()

    def test_cached_full_lora_geometry_preserved_against_original_dispatch(self):
        for mode in ('full','lora'):
            with self.subTest(mode=mode):
                loop,contract=fixture(mode=mode);ledger=self.ledger(loop,contract,mode)
                rows=generation(loop);state=ledger.snapshot();tokenizer=base.tokenizer();expected=[]
                original,_=base.make_loop(mode=mode);original.model.eval()
                with torch.no_grad():
                    for row in DEV:
                        prefix=tokenizer.apply_chat_template(row['messages'][:-1],tokenize=True,
                            add_generation_prompt=True,enable_thinking=False,return_dict=False,return_tensors='pt')
                        result=original.model.generate(prefix,attention_mask=torch.ones_like(prefix),max_new_tokens=4,
                            do_sample=False,pad_token_id=3,eos_token_id=[3],use_cache=True)
                        expected.append(result[0,prefix.shape[1]:].tolist())
                self.assertEqual([row['generated_ids'] for row in rows],expected)
                self.assertEqual(state['completed']['policy_forward_calls'],sum(len(row['generated_ids']) for row in rows))
                self.assertEqual(state['completed']['policy_forward_positions'],sum(row['prompt_tokens']+len(row['generated_ids'])-1 for row in rows))
                self.assertLess(state['completed']['policy_forward_positions'],state['reserved']['policy_forward_positions'])

    def test_generation_raw_failure_and_returned_decode_failure(self):
        loop,contract=fixture();ledger=self.ledger(loop,contract);retained=[]
        with patch.object(loop.model,'forward',side_effect=OSError('authored SFT generation failure')),self.assertRaises(OSError):generation(loop,retained.append)
        self.assertIsNone(retained[0]['generated_ids']);self.assertEqual(retained[0]['stop_reason'],'error')
        self.assertEqual(ledger.snapshot()['attempted_upper']['policy_forward_calls'],1)
        other,contract=fixture();other_ledger=self.ledger(other,contract,'decode');tokenizer=base.tokenizer();retained=[]
        def stopped(ids,**kwargs):
            other.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=True)
            return torch.cat([ids,torch.tensor([[3]])],dim=1)
        with patch.object(other.model,'generate',side_effect=stopped),patch.object(tokenizer,'decode',side_effect=ValueError('authored SFT decode failure')),self.assertRaises(ValueError):
            r.generate_sft_panel(other,tokenizer,DEV[:1],[[1,7]],cap=4,stop_ids=[3],end_message_id=3,row_sink=retained.append)
        self.assertEqual(retained[0]['generated_ids'],[3]);self.assertEqual(retained[0]['final_stop_id'],3)
        self.assertFalse(retained[0]['truncated']);self.assertEqual(other_ledger.snapshot()['known_partial']['generation_tokens'],1)

    def test_nll_failure_retains_exact_record_and_entered_work(self):
        loop,contract=fixture();ledger=self.ledger(loop,contract);retained=[]
        encoded=[r.encode_record(base.tokenizer(),row,64) for row in DEV]
        with patch.object(loop.model,'forward',side_effect=OSError('authored exact NLL record failure')),self.assertRaises(OSError):
            r.evaluate_sft_panel(loop,encoded,row_sink=retained.append)
        self.assertEqual(retained[0]['id'],DEV[0]['id']);self.assertIsNone(retained[0]['loss_sum'])
        self.assertEqual(retained[0]['error']['type'],'OSError')
        self.assertEqual(ledger.snapshot()['attempted_upper']['evaluation_calls'],1)
        self.assertEqual(ledger.snapshot()['known_partial']['evaluation_calls'],0)

    def test_excess_generation_geometry_refused_before_backend(self):
        loop,contract=fixture();ledger=self.ledger(loop,contract);calls=[];real=loop.model.forward;retained=[]
        def backend(*args,**kwargs):calls.append(1);return real(*args,**kwargs)
        def excessive(ids,**kwargs):
            for _ in range(9):loop.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=True)
            return ids
        with patch.object(loop.model,'forward',side_effect=backend),patch.object(loop.model,'generate',side_effect=excessive),self.assertRaises(WorkBudgetExceeded):generation(loop,retained.append)
        self.assertLessEqual(len(calls),8);self.assertEqual(retained[0]['stop_reason'],'error')

    def test_save_metric_observer_export_failures_leave_durable_state_and_spending(self):
        for failure in ('save','metric','baseline','final','export'):
            with self.subTest(failure=failure):
                loop,contract=fixture();ledger=self.ledger(loop,contract,failure);output=self.directory(failure+'-output')
                real_save=r.save_snapshot
                def save(path,**kwargs):
                    if failure=='save' and kwargs['completed_updates']==2:raise OSError('authored periodic save failure')
                    return real_save(path,**kwargs)
                def metric(row):
                    if failure=='metric' and row['update']==2:raise OSError('authored metric failure')
                def observe(stage):
                    result=observer(loop)
                    if failure==stage:raise OSError('authored observer failure '+stage)
                    return result
                def export():
                    if failure=='export':raise OSError('authored export failure')
                with patch.object(r,'save_snapshot',side_effect=save),self.assertRaises(OSError):
                    r.run_loop_with_recovery(loop,contract=contract,output=output,parent_invocation='failures',max_bytes=base.LIMIT,
                        checkpoint_every=2,metric_sink=metric,baseline_observer=lambda:observe('baseline'),
                        final_observer=lambda:observe('final'),exporter=export)
                saved=receipt(output);expected=0 if failure in ('save','baseline') else 2 if failure=='metric' else 4
                new,new_contract=fixture();attach(new,new_contract,ledger);restore(new,new_contract,saved)
                self.assertEqual(new.update,expected);self.assertGreaterEqual(ledger.snapshot()['reserved']['train_updates'],expected)
                self.assertGreater(ledger.snapshot()['reserved']['evaluation_calls'],0)

    def test_reservation_fsync_failure_precedes_selection_and_forward(self):
        loop,contract=fixture();ledger=self.ledger(loop,contract)
        with patch('dongxi_llms.work_budget.os.fsync',side_effect=OSError('authored SFT reservation fsync failure')),patch.object(r,'collate') as selection,patch.object(loop.model,'forward') as forward,self.assertRaises(WorkLedgerError):loop.completed_update()
        selection.assert_not_called();forward.assert_not_called();self.assertEqual(loop.cursor,0);self.assertTrue(ledger.poisoned)

    def test_limits_reader_strict_regular_no_follow_and_no_peer_fifo(self):
        normal=self.root/'limits.json';write(normal,CAPS);self.assertEqual(json.loads(r.read_work_limits_file(normal)),CAPS)
        final=self.root/'final-link';final.symlink_to(normal)
        with self.assertRaises(OSError):r.read_work_limits_file(final)
        ancestor=self.root/'ancestor';ancestor.symlink_to(self.root,target_is_directory=True)
        with self.assertRaises(OSError):r.read_work_limits_file(ancestor/'limits.json')
        with self.assertRaises(ValueError):r.read_work_limits_file(self.root)
        oversized=self.root/'oversized';oversized.write_bytes(b' '*65537)
        with self.assertRaises(ValueError):r.read_work_limits_file(oversized)
        result=fifo_control(self.root)
        self.assertNotEqual(result['exit_code'],0);self.assertIn('regular no-follow file',result['stderr'])
        self.assertFalse(result['output_created'])

    def test_cli_caps_resume_and_global_hash_gate(self):
        limits=self.root/'caps.json';write(limits,CAPS)
        argv=['--revision','1'*40,'--tokenizer-revision','1'*40,'--template','unused','--train','unused',
            '--dev','unused','--output',str(self.root/'never'),'--environment-lock','uv.lock']
        with self.assertRaises(SystemExit):r.main(argv)
        argv += explicit_snapshot_io_args(self.root)
        with self.assertRaisesRegex(ValueError,'same physical'):
            r.main(argv+['--work-limits',str(limits),'--work-journal-max-bytes',str(BOUND),
                '--resume','unused','--resume-contract','unused','--resume-sha256','1'*64,'--resume-bytes','100'])
        # Regression for an existing local import that shadowed the already
        # imported hash function before real preparation could call it.
        self.assertNotIn('canonical_hash',r._main.__code__.co_varnames)
        self.assertEqual(set(CAPS),set(r.WORK_KEYS))
        for limits in (dict(CAPS,train_updates=True),dict(CAPS,train_updates=-1),dict(CAPS,unknown=1)):
            with self.assertRaises(ValueError):r.work_budget_contract(limits,BOUND)

    def test_changed_parsed_caps_and_postparse_fifo_refuse_before_loaders(self):
        for kind in ('regular','fifo'):
            with self.subTest(kind=kind):
                result=cap_mutation_control(self.directory(kind),kind)
                self.assertEqual(result['tokenizer_loader_calls'],0);self.assertEqual(result['model_loader_calls'],0)
                self.assertIn('changed before' if kind=='regular' else 'regular no-follow',result['raw_exception']['message'])


    def test_bounded_cap_role_binds_exact_bytes_and_enclosing_identity(self):
        limits=self.root/'caps.json';write(limits,CAPS);raw=limits.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        identity=dict(input_sha256={'original-data':'1'*64},config={'seed':1212},identity_sha256='2'*64)
        actual=r.bind_bounded_work_limits(identity,limits,digest)
        self.assertIs(actual,identity)
        self.assertEqual(identity['bounded_work_limits'],dict(path=os.path.abspath(limits),sha256=digest,
            bytes=len(raw),reader='bounded-nonblocking-no-follow-regular'))
        self.assertEqual(identity['identity_sha256'],canonical_hash({key:value for key,value in identity.items() if key!='identity_sha256'}))
        r.verify_bounded_work_limits_identity(identity,limits,digest)
        identity['bounded_work_limits']['bytes']+=1
        with self.assertRaisesRegex(ValueError,'identity changed'):r.verify_bounded_work_limits_identity(identity,limits,digest)
        with self.assertRaises(ValueError):r.bind_bounded_work_limits({},limits,'invalid')

    def test_postparse_replacement_before_generic_identity_refuses_with_deadline(self):
        for kind in ('regular','fifo'):
            with self.subTest(kind=kind):
                control=bounded_cap_control(self.directory('before-'+kind),kind,'before_identity')
                self.assertEqual(control['execution']['exit_code'],0)
                result=control['result'];self.assertTrue(result['cap_excluded_from_generic_inputs'])
                self.assertEqual(result['tokenizer_loader_calls'],0);self.assertEqual(result['model_loader_calls'],0)
                self.assertIn('changed before' if kind=='regular' else 'regular no-follow',result['raw_exception']['message'])

    def test_closing_identity_regular_and_fifo_refuse_with_deadline(self):
        for kind in ('regular','fifo'):
            with self.subTest(kind=kind):
                control=bounded_cap_control(self.directory('closing-'+kind),kind,'closing_identity')
                self.assertEqual(control['execution']['exit_code'],0);self.assertTrue(control['result']['identity_unchanged'])
                self.assertIn('changed before' if kind=='regular' else 'regular no-follow',control['result']['raw_exception']['message'])


def collect(output):
    output=Path(output).resolve();output.mkdir(parents=True,mode=0o700,exist_ok=False)
    names=['scripts/run_chapter09_spark_sft.py','tests/test_sft_runner_recovery.py','tests/test_spark_sft_contract.py','tests/test_sft_work_budget.py',
        'src/dongxi_llms/run_identity.py','src/dongxi_llms/work_budget.py','src/dongxi_llms/artifact_budget.py',
        'src/dongxi_llms/training_snapshot.py','uv.lock','experiments/specs/2026-10-05-sft-cumulative-work-budgets.md',
        'experiments/specs/2026-10-05-sft-cap-byte-binding.md','experiments/specs/2026-10-05-sft-bounded-cap-identity.md']
    hashes=lambda:{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
    before=hashes();write(output/'source-before.json',before);write(output/'work-limits.json',CAPS)
    write(output/'authored-fixture.json',dict(train=base.records(),dev=DEV,limits=CAPS,seeds=[1212,1213],modes=['full','lora']))
    command=[sys.executable,'-m','unittest','test_sft_work_budget','test_sft_runner_recovery','test_spark_sft_contract','test_work_budget','-v']
    started=time.monotonic();process=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=180)
    tests=dict(command=command,exit_code=process.returncode,seconds=time.monotonic()-started,stdout=process.stdout,stderr=process.stderr)
    write(output/'tests.json',tests)
    if process.returncode:raise RuntimeError('Focused SFT controls failed; retained diagnostic output')
    with tempfile.TemporaryDirectory(prefix='dongxi-sft-fifo-') as directory:raw_fifo=fifo_control(directory)
    write(output/'limits-fifo-cli.json',raw_fifo)
    if raw_fifo['exit_code']==0 or raw_fifo['output_created'] or 'regular no-follow file' not in raw_fifo['stderr']:
        raise AssertionError('Actual SFT FIFO control failed; raw exit retained')
    mutations=[]
    for kind in ('regular','fifo'):
        with tempfile.TemporaryDirectory(prefix='dongxi-sft-cap-mutation-') as directory:mutations.append(cap_mutation_control(directory,kind))
    write(output/'parsed-cap-mutation-controls.json',mutations)
    def directory(name):
        path=output/name;path.mkdir(mode=0o700);return path
    bounded_controls=[]
    for phase in ('before_identity','closing_identity'):
        for kind in ('regular','fifo'):
            bounded_controls.append(bounded_cap_control(directory(f'cap-{phase}-{kind}'),kind,phase))
    write(output/'bounded-cap-identity-controls.json',bounded_controls)
    arms=[]
    for seed in (1212,1213):
        for mode in ('full','lora'):
            arm=directory(f'{seed}-{mode}');loop,contract=fixture(seed,mode);write(arm/'contract.json',contract)
            full_journal=directory(f'{seed}-{mode}-full-journal')/'work.jsonl'
            with ledger_for(full_journal,contract) as ledger:
                attach(loop,contract,ledger);full=lifecycle(loop,contract,directory(f'{seed}-{mode}-full'))
                expected=base.state_digest(numerical(loop));history=deepcopy(loop.history);full_ledger=ledger.snapshot()
            original,contract2=fixture(seed,mode)
            journal=directory(f'{seed}-{mode}-recovery-journal')/'work.jsonl'
            with ledger_for(journal,contract2) as ledger:
                attach(original,contract2,ledger);saved,interruption=interrupted(original,contract2,directory(f'{seed}-{mode}-interrupted'))
                failure=failed_later_forward(original);charged=ledger.snapshot()
            write(arm/'receipt.json',saved);write(arm/'retained-failures.json',dict(interruption=interruption,later_failure=failure,charged=charged))
            command=[sys.executable,str(Path(__file__).resolve()),'--child',str(arm/'receipt.json'),str(arm/'contract.json'),
                     str(journal),str(output/f'{seed}-{mode}-fresh'),mode,str(seed)]
            started=time.monotonic();process=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=90)
            execution=dict(command=command,exit_code=process.returncode,seconds=time.monotonic()-started,stdout=process.stdout,stderr=process.stderr)
            write(arm/'fresh-execution.json',execution)
            if process.returncode:raise RuntimeError('Fresh actual SFT work replay failed; raw exit retained')
            actual=json.loads((output/f'{seed}-{mode}-fresh/numerical-digest.json').read_text())
            current=json.loads((output/f'{seed}-{mode}-fresh/ledger-after.json').read_text())
            if actual['sha256']!=expected or actual['history']!=history or current['reserved']['train_updates']!=5 or current['completed']['train_updates']!=4:
                raise AssertionError('Fresh SFT state/history/work replay differs')
            record=dict(seed=seed,mode=mode,exact_numerical_replay=True,expected_numerical_sha256=expected,
                full=full,full_history=history,full_ledger=full_ledger,after_later_failure=charged,recovered_ledger=current,fresh_execution=execution,
                contract_sha256=canonical_hash(contract))
            write(arm/'result.json',record);arms.append(record)
    after=hashes();write(output/'source-after.json',after)
    if before!=after:raise AssertionError('SFT reference source drift')
    artifacts={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output.rglob('*')) if path.is_file()}
    verification=dict(schema='dongxi-sft-cumulative-work-cpu-v1',status='pass',command=sys.orig_argv,
        tests=tests,test_count=64,source_sha256=before,artifact_sha256=artifacts,arms=arms,
        scope='actual full/LoRA CPUFP32 SFT; cooperative logical reservations; cached generation preserved',
        pending=['pretrained/CUDA/BF16','physical quota/cgroups','artifact/log/export wiring','model-scale overhead',
                 'hostile rollback/resistance','external campaign outcomes','capability/quality'])
    write(output/'verification.json',verification)
    print(json.dumps(dict(status='pass',tests=64,arms=4,verification=str(output/'verification.json'))))


if __name__=='__main__':
    if len(sys.argv)==8 and sys.argv[1]=='--child':child(*sys.argv[2:])
    elif len(sys.argv)==5 and sys.argv[1]=='--cap-control':
        directory,kind,phase=sys.argv[2:]
        result=closing_cap_control(directory,kind) if phase=='closing_identity' else cap_mutation_control(directory,kind,phase)
        write(Path(directory)/'cap-control.json',result)
    elif len(sys.argv)==3 and sys.argv[1]=='--collect':collect(sys.argv[2])
    else:unittest.main()
