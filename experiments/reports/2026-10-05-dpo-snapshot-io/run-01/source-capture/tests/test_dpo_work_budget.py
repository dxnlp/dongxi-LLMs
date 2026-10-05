"""Predeclared original CPU DPO work/recovery controls and evidence collector."""
from copy import deepcopy
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
import test_dpo_checkpoint_mode as mode
import test_dpo_runner_recovery as base
from dongxi_llms.run_identity import canonical_hash
from dongxi_llms.work_budget import WorkLedger,WorkBudgetExceeded

r=base.runner;ROOT=base.ROOT
CAPS=dict(train_updates=12,sampled_examples=24,sampler_draws=24,valid_targets=160,
    logical_sequence_tokens=512,policy_forward_calls=128,policy_forward_positions=1024,
    reference_forward_calls=96,reference_forward_positions=1024,evaluation_calls=64,
    evaluation_positions=512,generation_calls=16,generation_position_upper_bound=512,generation_tokens=64,
    recovery_validation_operations=128,recovery_history_rows=4096,
    recovery_tensor_elements=2000000,recovery_rng_states=1024,recovery_sampler_draws=4096)
BOUND=1024*1024
EVALUATION=[dict(id='original-heldout',expected='good')]
PREFIXES=[[7,8]]


def fixture(limits=CAPS):
    values=list(mode.make_mode(True));options=deepcopy(values[5])
    options['sources']['src/dongxi_llms/work_budget.py']=r.file_digest(ROOT/'src/dongxi_llms/work_budget.py')
    options['sources']['src/dongxi_llms/run_identity.py']=r.file_digest(ROOT/'src/dongxi_llms/run_identity.py')
    options['sources']['tests/test_dpo_work_budget.py']=r.file_digest(Path(__file__))
    options['work_budget']=r.work_budget_contract(limits,BOUND)
    values[4]=r.make_recovery_contract(model=values[0],reference=values[1],optimizer=values[2],**options)
    values[5]=options;return values


def create_ledger(path,values,invocation='original'):
    return WorkLedger.create(path,limits=values[4]['work_budget']['limits'],
        contract_sha256=canonical_hash(values[4]),max_bytes=BOUND,invocation_id=invocation)


def open_ledger(path,values,invocation='recovery'):
    return WorkLedger.open(path,limits=values[4]['work_budget']['limits'],
        contract_sha256=canonical_hash(values[4]),max_bytes=BOUND,invocation_id=invocation)


def train(output,values,ledger,initial=None,before_training=None):
    return r.train_completed_updates(*values[:4],base.ENCODED,contract=values[4],output=output,
        invocation_id='cpu-budget-fixture',pad_id=1,device='cpu',checkpoint_every=3,max_bytes=base.LIMIT,
        work_ledger=ledger,before_training=before_training,**(initial or {}))


def load(receipt,values,ledger):
    return r.load_snapshot(receipt['path'],expected_sha256=receipt['payload_sha256'],
        expected_bytes=receipt['payload_bytes'],expected_contract=values[4],max_bytes=base.LIMIT,
        validate_payload=lambda payload:r.validate_dpo_payload(payload,values[4],ledger))


def restore(receipt,values,ledger):
    payload=load(receipt,values,ledger)
    completed,counters,history=r.restore_dpo_payload(payload,contract=values[4],model=values[0],
        reference=values[1],optimizer=values[2],sampler=values[3],work_ledger=ledger)
    return dict(completed=completed,counters=counters,history=history,parent_checkpoint=receipt)


def numerical(payload):
    return {key:value for key,value in payload['state'].items() if key not in ('resume_parent','work_ledger')}


def interrupted(output,values,ledger,before_training=None):
    original=r.append_metric
    def fail(path,row):
        if row.get('update')==3:raise OSError('authored post-commit update3 interruption')
        return original(path,row)
    try:
        with patch.object(r,'append_metric',side_effect=fail):train(output,values,ledger,before_training=before_training)
    except OSError as error:return base.checkpoint_receipt(output/'checkpoints/completed-000003.pt'),str(error)
    raise AssertionError('Declared interruption absent')


def fail_after_checkpoint(values,ledger):
    # First policy call is entered, but its deliberately failing backend returns
    # nothing. Draw and supervision metadata are known; internal FLOPs are not.
    with patch.object(values[0],'forward',side_effect=OSError('authored entered-forward failure')):
        try:r.completed_dpo_update(*values[:4],base.ENCODED,pad_id=1,accumulation=2,beta=.2,
                                  device='cpu',update=4,work_ledger=ledger)
        except OSError as error:return dict(type=type(error).__name__,message=str(error),deliberately_injected=True)
    raise AssertionError('Declared failure absent')


def generation(values,ledger,row_sink=None):
    return r.generate_dpo_panel(values[0],base.tokenizer_fixture(),EVALUATION,PREFIXES,
        cap=4,stop_ids=[1],device='cpu',work_ledger=ledger,row_sink=row_sink)


def fifo_cli_control(directory):
    from snapshot_io_test_support import explicit_snapshot_io_args
    directory=Path(directory);fifo=directory/'limits-fifo';os.mkfifo(fifo,0o600)
    output=directory/'never-created'
    command=[sys.executable,str(ROOT/'scripts/run_chapter11_spark_dpo.py'),
        '--checkpoint',str(directory/'unused-checkpoint'),'--tokenizer','unused-local',
        '--tokenizer-revision','a'*40,'--train','unused-train','--validation','unused-valid',
        '--evaluation','unused-evaluation','--output',str(output),'--snapshot-max-bytes',str(base.LIMIT),
        '--snapshot-artifact-max-bytes','67108864','--snapshot-artifact-max-entries','64',
        '--snapshot-artifact-journal-max-bytes',str(BOUND),
        '--environment-lock','uv.lock','--work-limits',str(fifo),'--work-journal-max-bytes',str(BOUND)]+explicit_snapshot_io_args(directory)
    started=time.monotonic()
    process=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=10)
    return dict(command=command,exit_code=process.returncode,seconds=time.monotonic()-started,
                stdout=process.stdout,stderr=process.stderr,output_created=output.exists(),
                fifo_retained_before_temporary_cleanup=fifo.exists(),declared_timeout_seconds=10,
                scope='no-peer FIFO; trusted-local invalid input; no output/hardware/model work')


def child(receipt_file,contract_file,journal_file,output):
    values=fixture();expected=json.loads(Path(contract_file).read_text())
    if values[4]!=expected:raise ValueError('Fresh actual source/recipe contract differs')
    receipt=json.loads(Path(receipt_file).read_text())
    Path(output).mkdir(mode=0o700)
    with open_ledger(journal_file,values,'fresh') as ledger:
        before=ledger.snapshot()
        initial=restore(receipt,values,ledger)
        r.write_exclusive_json(Path(output)/'ledger-after-restore.json',ledger.snapshot())
        result=train(output,values,ledger,initial,before_training=lambda:generation(values,ledger))
        r.write_exclusive_json(Path(output)/'result.json',result)
        r.write_exclusive_json(Path(output)/'ledger-before.json',before)
        r.write_exclusive_json(Path(output)/'ledger-after.json',ledger.snapshot())


class DPOWorkBudgetTests(unittest.TestCase):
    def setUp(self):
        directory=tempfile.TemporaryDirectory(prefix='dongxi-dpo-work-')
        self.addCleanup(directory.cleanup);self.root=Path(directory.name)

    def directory(self,name):
        result=self.root/name;result.mkdir(mode=0o700);return result

    def ledger(self,values,name='ledger'):
        path=self.directory(name)/'work.jsonl';ledger=create_ledger(path,values)
        self.addCleanup(ledger.close);return ledger

    def test_actual_unbounded_and_budgeted_numerical_parity(self):
        ordinary=mode.make_mode(True);unbounded=base.train_fixture(self.directory('unbounded'),ordinary)
        expected=base.payload_for(unbounded['checkpoint'],ordinary[4])
        values=fixture();ledger=self.ledger(values)
        result=train(self.directory('bounded'),values,ledger);actual=load(result['checkpoint'],values,ledger)
        self.assertEqual(base.digest(numerical(expected)),base.digest(numerical(actual)))
        self.assertEqual(ledger.snapshot()['completed']['train_updates'],6)
        self.assertEqual(ledger.snapshot()['reserved']['valid_targets'],72)
        self.assertEqual(ledger.snapshot()['completed']['valid_targets'],62)
        self.assertEqual(ledger.snapshot()['reserved']['policy_forward_positions'],96)
        self.assertEqual(ledger.snapshot()['completed']['policy_forward_positions'],86)

    def test_whole_update_refused_before_draw_forward_or_objective_shortening(self):
        caps=dict(CAPS,valid_targets=11);values=fixture(caps);ledger=self.ledger(values)
        with patch.object(torch,'randint') as draw,patch.object(values[0],'forward') as forward,self.assertRaises(WorkBudgetExceeded):
            r.completed_dpo_update(*values[:4],base.ENCODED,pad_id=1,accumulation=2,beta=.2,
                device='cpu',update=1,work_ledger=ledger)
        draw.assert_not_called();forward.assert_not_called()
        self.assertEqual(ledger.snapshot()['sequence'],0)

    def test_reservation_persistence_failure_precedes_sampler_and_network(self):
        values=fixture();ledger=self.ledger(values)
        with patch('dongxi_llms.work_budget.os.fsync',side_effect=OSError('authored pre-work fsync failure')),patch.object(torch,'randint') as draw,patch.object(values[0],'forward') as forward,self.assertRaises(Exception):
            r.completed_dpo_update(*values[:4],base.ENCODED,pad_id=1,accumulation=2,beta=.2,
                device='cpu',update=1,work_ledger=ledger)
        draw.assert_not_called();forward.assert_not_called();self.assertTrue(ledger.poisoned)

    def test_entered_failed_forward_retains_partial_and_uncertainty(self):
        values=fixture();ledger=self.ledger(values);failure=fail_after_checkpoint(values,ledger)
        state=ledger.snapshot()
        self.assertIn('entered-forward',failure['message'])
        self.assertEqual(state['reserved']['policy_forward_calls'],4)
        self.assertEqual(state['attempted_upper']['policy_forward_calls'],1)
        self.assertEqual(state['known_partial']['policy_forward_calls'],0)
        self.assertEqual(state['uncertain_upper']['policy_forward_calls'],1)
        self.assertEqual(state['known_partial']['sampler_draws'],1)

    def test_fresh_process_same_journal_retains_later_attempts_and_exact_model(self):
        values=fixture();full_ledger=self.ledger(values,'full-ledger')
        full=train(self.directory('full'),values,full_ledger);expected=load(full['checkpoint'],values,full_ledger)
        original=fixture();ledger=self.ledger(original,'recovery-ledger')
        receipt,_=interrupted(self.directory('interrupted'),original,ledger)
        prefix=load(receipt,original,ledger)['state']['work_ledger']
        failure=fail_after_checkpoint(original,ledger);charged=ledger.snapshot();ledger.close()
        self.assertGreater(charged['reserved']['train_updates'],prefix['reserved']['train_updates'])
        r.write_exclusive_json(self.root/'receipt.json',receipt);r.write_exclusive_json(self.root/'contract.json',original[4])
        command=[sys.executable,str(Path(__file__).resolve()),'--child',str(self.root/'receipt.json'),
                 str(self.root/'contract.json'),str(ledger.path),str(self.root/'fresh')]
        process=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=90)
        self.assertEqual(process.returncode,0,process.stdout+process.stderr)
        resumed=open_ledger(ledger.path,original,'inspection');self.addCleanup(resumed.close)
        result=json.loads((self.root/'fresh/result.json').read_text());actual=load(result['checkpoint'],original,resumed)
        self.assertEqual(base.digest(numerical(expected)),base.digest(numerical(actual)))
        before=json.loads((self.root/'fresh/ledger-before.json').read_text())
        self.assertEqual(before,charged)
        restored=json.loads((self.root/'fresh/ledger-after-restore.json').read_text())
        validation_keys={'recovery_validation_operations','recovery_history_rows',
            'recovery_tensor_elements','recovery_rng_states','recovery_sampler_draws'}
        for key in CAPS:
            if key not in validation_keys:
                self.assertEqual(restored['reserved'][key],charged['reserved'][key])
        self.assertEqual(restored['reserved']['recovery_validation_operations']-
            charged['reserved']['recovery_validation_operations'],2)
        self.assertGreater(restored['reserved']['recovery_sampler_draws'],
            charged['reserved']['recovery_sampler_draws'])
        self.assertEqual(resumed.snapshot()['reserved']['train_updates'],7)
        self.assertEqual(resumed.snapshot()['completed']['train_updates'],6)
        self.assertEqual(full['history'][3],result['history'][3])

    def test_new_output_does_not_refill_exhausted_recovery_budget(self):
        caps=dict(CAPS,train_updates=4);values=fixture(caps);ledger=self.ledger(values)
        receipt,_=interrupted(self.directory('first'),values,ledger)
        fail_after_checkpoint(values,ledger);values2=fixture(caps)
        initial=restore(receipt,values2,ledger)
        real_draw=torch.randint;active_draws=[]
        def observe_draw(*args,**kwargs):
            if kwargs.get('generator') is values2[3]:active_draws.append(1)
            return real_draw(*args,**kwargs)
        with patch.object(torch,'randint',side_effect=observe_draw),patch.object(values2[0],'forward') as forward,self.assertRaises(WorkBudgetExceeded):
            train(self.directory('new-output'),values2,ledger,initial)
        # Semantic payload validation replays the fixed sampler for audit, but
        # this does not consume the active sampler. Refusal happened before work.
        forward.assert_not_called()
        self.assertFalse(active_draws)
        self.assertEqual(values2[3].get_state().tolist(),load(receipt,values2,ledger)['state']['sampler_rng'].tolist())
        self.assertEqual(ledger.snapshot()['reserved']['train_updates'],4)

    def test_same_process_recovery_recharges_baseline_and_preserves_math(self):
        expected_values=fixture();expected_ledger=self.ledger(expected_values,'expected-ledger')
        full=train(self.directory('full'),expected_values,expected_ledger,before_training=lambda:generation(expected_values,expected_ledger))
        expected=load(full['checkpoint'],expected_values,expected_ledger)
        values=fixture();ledger=self.ledger(values)
        receipt,_=interrupted(self.directory('first'),values,ledger,before_training=lambda:generation(values,ledger))
        fail_after_checkpoint(values,ledger);resumed=fixture();initial=restore(receipt,resumed,ledger)
        result=train(self.directory('resumed'),resumed,ledger,initial,before_training=lambda:generation(resumed,ledger))
        self.assertEqual(base.digest(numerical(expected)),base.digest(numerical(load(result['checkpoint'],resumed,ledger))))
        self.assertEqual(ledger.snapshot()['reserved']['generation_calls'],2)
        self.assertEqual(ledger.snapshot()['reserved']['train_updates'],7)

    def test_baseline_and_validation_whole_panels_refused_before_network(self):
        values=fixture(dict(CAPS,generation_tokens=3));ledger=self.ledger(values)
        with patch.object(values[0],'forward') as forward,patch.object(values[0],'generate') as generate,self.assertRaises(WorkBudgetExceeded):
            train(self.directory('baseline-refused'),values,ledger,before_training=lambda:generation(values,ledger))
        forward.assert_not_called();generate.assert_not_called()
        receipt=base.checkpoint_receipt(self.root/'baseline-refused/checkpoints/completed-000000.pt')
        self.assertEqual(load(receipt,values,ledger)['state']['completed'],0)
        self.assertEqual(ledger.snapshot()['reserved']['train_updates'],0)
        other=fixture(dict(CAPS,reference_forward_positions=5));other_ledger=self.ledger(other,'validation-ledger')
        with patch.object(other[0],'forward') as policy,patch.object(other[1],'forward') as reference,self.assertRaises(WorkBudgetExceeded):
            r.score_dpo_panel(other[0],other[1],base.VALID,pad_id=1,device='cpu',beta=.2,work_ledger=other_ledger)
        policy.assert_not_called();reference.assert_not_called()

    def test_new_ledger_and_wrong_caps_fail_before_model_application(self):
        values=fixture();ledger=self.ledger(values);receipt,_=interrupted(self.directory('first'),values,ledger)
        fresh=self.ledger(values,'different-ledger');new=fixture()
        with patch.object(new[0],'load_state_dict') as apply,self.assertRaises(ValueError):restore(receipt,new,fresh)
        apply.assert_not_called()
        wrong=fixture(dict(CAPS,train_updates=13))
        with patch.object(wrong[0],'load_state_dict') as apply,self.assertRaises(ValueError):restore(receipt,wrong,ledger)
        apply.assert_not_called()

    def test_baseline_repeated_invocation_and_final_panels_are_charged(self):
        values=fixture();ledger=self.ledger(values)
        first=generation(values,ledger);second=generation(values,ledger)
        validation=r.score_dpo_panel(values[0],values[1],base.VALID,pad_id=1,device='cpu',beta=.2,work_ledger=ledger)
        state=ledger.snapshot();self.assertEqual(state['reserved']['generation_calls'],2)
        self.assertEqual(state['reserved']['generation_tokens'],8)
        self.assertEqual(state['reserved']['evaluation_calls'],12)
        self.assertEqual(state['reserved']['reference_forward_calls'],2)
        self.assertTrue(validation);self.assertTrue(first[0]['generated_ids']);self.assertTrue(second[0]['generated_ids'])
        self.assertEqual(state['completed']['generation_tokens'],len(first[0]['generated_ids'])+len(second[0]['generated_ids']))
        self.assertLessEqual(state['completed']['policy_forward_positions'],state['reserved']['policy_forward_positions'])

    def test_generation_cap_and_eos_inclusive_geometry(self):
        values=fixture();ledger=self.ledger(values)
        rows=generation(values,ledger);state=ledger.snapshot();continuation=rows[0]['generated_ids']
        self.assertEqual(state['completed']['generation_tokens'],len(continuation))
        self.assertEqual(state['completed']['policy_forward_calls'],len(continuation))
        self.assertEqual(state['completed']['policy_forward_positions'],sum(2+i for i in range(len(continuation))))
        # Controlled decoder uses the real bound hook and deliberately returns
        # one declared stop; no unused cap reservation is refunded.
        other=fixture();other_ledger=self.ledger(other,'eos-ledger')
        def stopped(ids,**kwargs):
            other[0](input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
            return torch.cat([ids,torch.tensor([[1]])],dim=1)
        with patch.object(other[0],'generate',side_effect=stopped):eos=generation(other,other_ledger)
        self.assertFalse(eos[0]['truncated']);self.assertEqual(eos[0]['generated_ids'],[1])
        self.assertEqual(other_ledger.snapshot()['completed']['generation_tokens'],1)
        self.assertEqual(other_ledger.snapshot()['reserved']['generation_tokens'],4)

    def test_unexpected_generation_forward_refused_before_backend(self):
        values=fixture();ledger=self.ledger(values);retained=[]
        real=values[0].forward;count=[0]
        def backend(*args,**kwargs):count[0]+=1;return real(*args,**kwargs)
        def excessive(ids,**kwargs):
            for _ in range(5):values[0](input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
            return ids
        with patch.object(values[0],'forward',side_effect=backend),patch.object(values[0],'generate',side_effect=excessive),self.assertRaises(WorkBudgetExceeded):
            generation(values,ledger,row_sink=retained.append)
        self.assertEqual(count[0],4);self.assertEqual(retained[0]['stop_reason'],'error')
        self.assertEqual(ledger.snapshot()['known_partial']['policy_forward_calls'],4)

    def test_generation_backend_failure_has_raw_error_and_known_geometry(self):
        values=fixture();ledger=self.ledger(values);retained=[]
        with patch.object(values[0],'forward',side_effect=RuntimeError('authored generation failure')):
            rows=generation(values,ledger,row_sink=retained.append)
        self.assertEqual(rows,retained);self.assertIsNone(rows[0]['generated_ids'])
        self.assertEqual(rows[0]['stop_reason'],'error')
        state=ledger.snapshot();self.assertEqual(state['attempted_upper']['policy_forward_calls'],1)
        self.assertEqual(state['known_partial']['policy_forward_calls'],0)
        self.assertEqual(state['uncertain_upper']['policy_forward_positions'],2)

    def test_decode_failure_preserves_returned_tokens_and_stop(self):
        values=fixture();ledger=self.ledger(values);tokenizer=base.tokenizer_fixture()
        def stopped(ids,**kwargs):
            values[0](input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
            return torch.cat([ids,torch.tensor([[1]])],dim=1)
        with patch.object(values[0],'generate',side_effect=stopped),patch.object(tokenizer,'decode',side_effect=ValueError('authored decode failure')):
            rows=r.generate_dpo_panel(values[0],tokenizer,EVALUATION,PREFIXES,
                cap=4,stop_ids=[1],device='cpu',work_ledger=ledger)
        self.assertEqual(rows[0]['generated_ids'],[1]);self.assertEqual(rows[0]['generated_tokens'],1)
        self.assertEqual(rows[0]['generation_stop_reason'],'declared-stop');self.assertEqual(rows[0]['final_stop_id'],1)
        self.assertFalse(rows[0]['truncated']);self.assertEqual(ledger.snapshot()['known_partial']['generation_tokens'],1)

    def test_final_evaluation_failure_preserves_committed_snapshot_and_spent_work(self):
        values=fixture();ledger=self.ledger(values)
        result=train(self.directory('train'),values,ledger)
        prefix=load(result['checkpoint'],values,ledger)['state']['work_ledger']
        def evaluate():
            return r.score_dpo_panel(values[0],values[1],base.VALID,pad_id=1,device='cpu',beta=.2,work_ledger=ledger)
        with patch.object(values[0],'forward',side_effect=OSError('authored final eval failure')),patch.object(values[0],'save_pretrained') as export,self.assertRaises(OSError):
            r.finalize_after_commit(result,evaluate=evaluate,export=export)
        export.assert_not_called();self.assertGreater(ledger.snapshot()['reserved']['evaluation_calls'],prefix['reserved']['evaluation_calls'])
        load(result['checkpoint'],values,ledger)
        self.assertEqual(ledger.snapshot()['completed']['train_updates'],6)

    def test_export_failure_keeps_final_model_and_later_evaluation_charges(self):
        values=fixture();ledger=self.ledger(values);result=train(self.directory('train'),values,ledger)
        prefix=load(result['checkpoint'],values,ledger)['state']['work_ledger']
        def export():raise OSError('authored export failure after charged evaluation')
        with self.assertRaises(OSError):r.finalize_after_commit(result,evaluate=lambda:generation(values,ledger),export=export)
        self.assertEqual(load(result['checkpoint'],values,ledger)['state']['completed'],6)
        self.assertEqual(ledger.snapshot()['reserved']['generation_calls'],prefix['reserved']['generation_calls']+1)

    def test_cli_requires_explicit_caps_and_same_resume_journal(self):
        limits_file=self.root/'limits.json';r.write_exclusive_json(limits_file,CAPS)
        argv=['--checkpoint',str(self.root/'unused-checkpoint'),'--tokenizer','unused-local',
              '--tokenizer-revision','a'*40,'--train','unused-train','--validation','unused-valid',
              '--evaluation','unused-evaluation','--output',str(self.root/'unused-output'),
              '--snapshot-max-bytes',str(base.LIMIT),'--environment-lock','uv.lock',
              '--snapshot-artifact-max-bytes','67108864','--snapshot-artifact-max-entries','64',
              '--snapshot-artifact-journal-max-bytes',str(BOUND)]
        with self.assertRaises(SystemExit):r.main(argv)
        argv+=['--work-limits',str(limits_file),'--work-journal-max-bytes',str(BOUND),
               '--resume','unused-snapshot','--resume-sha256','b'*64,'--resume-bytes','100',
               '--resume-contract','unused-contract']
        with patch.object(torch.cuda,'is_available') as hardware,self.assertRaises(SystemExit):r.main(argv)
        hardware.assert_not_called();self.assertFalse((self.root/'unused-output').exists())
        self.assertEqual(set(CAPS),set(r.BUDGET_KEYS))
        for caps in ({key:0 for key in CAPS if key!='generation_tokens'},dict(CAPS,train_updates=True)):
            with self.assertRaises(ValueError):r.work_budget_contract(caps,BOUND)

    def test_limits_file_regular_bounded_and_no_follow(self):
        regular=self.root/'limits.json';r.write_exclusive_json(regular,CAPS)
        self.assertEqual(json.loads(r.read_work_limits_file(regular)),CAPS)
        oversized=self.root/'oversized';oversized.write_bytes(b' '*65537)
        with self.assertRaises(ValueError):r.read_work_limits_file(oversized)
        with self.assertRaises(ValueError):r.read_work_limits_file(self.root)
        linked=self.root/'linked.json';linked.symlink_to(regular)
        with self.assertRaises(OSError):r.read_work_limits_file(linked)
        ancestor=self.root/'ancestor';ancestor.symlink_to(self.root,target_is_directory=True)
        with self.assertRaises(OSError):r.read_work_limits_file(ancestor/'limits.json')

    def test_actual_cli_no_peer_fifo_rejects_without_hanging(self):
        result=fifo_cli_control(self.root)
        self.assertNotEqual(result['exit_code'],0)
        self.assertIn('Work limits must be a regular no-follow file',result['stderr'])
        self.assertFalse(result['output_created']);self.assertTrue(result['fifo_retained_before_temporary_cleanup'])


def collect(output):
    output=Path(output).resolve();output.mkdir(parents=True,mode=0o700,exist_ok=False)
    names=['scripts/run_chapter11_spark_dpo.py','src/dongxi_llms/work_budget.py',
        'tests/test_work_budget.py','tests/test_dpo_work_budget.py','tests/test_dpo_checkpoint_mode.py',
        'tests/test_dpo_runner_recovery.py','src/dongxi_llms/training_snapshot.py',
        'src/dongxi_llms/dpo_lab.py','src/dongxi_llms/batched_cache_lab.py','src/dongxi_llms/run_identity.py','uv.lock',
        'experiments/specs/2026-10-05-cumulative-work-budgets.md',
        'experiments/specs/2026-10-05-work-limits-input-hardening.md']
    before={name:r.file_digest(ROOT/name) for name in names}
    r.write_exclusive_json(output/'source-before.json',before)
    command=[sys.executable,'-m','unittest','test_work_budget','test_dpo_work_budget','test_dpo_runner_recovery','test_dpo_checkpoint_mode','-v']
    start=time.monotonic();process=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=120)
    tests=dict(command=command,exit_code=process.returncode,seconds=time.monotonic()-start,stdout=process.stdout,stderr=process.stderr)
    r.write_exclusive_json(output/'tests.json',tests)
    if process.returncode:raise RuntimeError('Focused controls failed; output retained diagnostic')
    with tempfile.TemporaryDirectory(prefix='dongxi-work-limits-fifo-') as directory:
        fifo_result=fifo_cli_control(directory)
    r.write_exclusive_json(output/'limits-fifo-cli.json',fifo_result)
    if fifo_result['exit_code']==0 or fifo_result['output_created'] or 'Work limits must be a regular no-follow file' not in fifo_result['stderr']:
        raise AssertionError('Actual no-peer FIFO CLI control failed; raw output retained')
    def directory(name):
        path=output/name;path.mkdir(mode=0o700);return path
    values=fixture();r.write_exclusive_json(output/'contract.json',values[4])
    with create_ledger(directory('full-ledger')/'work.jsonl',values) as ledger:
        full=train(directory('full'),values,ledger,before_training=lambda:generation(values,ledger))
        expected=load(full['checkpoint'],values,ledger)
        final=r.finalize_after_commit(full,evaluate=lambda:dict(validation=r.score_dpo_panel(values[0],values[1],base.VALID,
            pad_id=1,device='cpu',beta=.2,work_ledger=ledger),generation=generation(values,ledger)),export=lambda:None)
        full_state=ledger.snapshot();r.write_exclusive_json(output/'full/result.json',full)
        r.write_exclusive_json(output/'full/final-evaluation.json',final)
        r.write_exclusive_json(output/'full/final-ledger.json',full_state)
    original=fixture();journal=directory('recovery-ledger')/'work.jsonl'
    with create_ledger(journal,original) as ledger:
        receipt,interruption=interrupted(directory('interrupted'),original,ledger,before_training=lambda:generation(original,ledger))
        prefix=load(receipt,original,ledger)['state']['work_ledger']
        failure=fail_after_checkpoint(original,ledger);after_failure=ledger.snapshot()
    r.write_exclusive_json(output/'receipt.json',receipt)
    r.write_exclusive_json(output/'retained-failures.json',dict(interruption=interruption,later_attempt=failure,
        checkpoint_prefix=prefix,after_later_attempt=after_failure))
    command=[sys.executable,str(Path(__file__).resolve()),'--child',str(output/'receipt.json'),
             str(output/'contract.json'),str(journal),str(output/'fresh')]
    start=time.monotonic();process=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=90)
    execution=dict(command=command,exit_code=process.returncode,seconds=time.monotonic()-start,stdout=process.stdout,stderr=process.stderr)
    r.write_exclusive_json(output/'fresh-execution.json',execution)
    if process.returncode:raise RuntimeError('Fresh budgeted recovery failed; evidence retained')
    result=json.loads((output/'fresh/result.json').read_text())
    with open_ledger(journal,original,'final-inspection') as ledger:
        actual=load(result['checkpoint'],original,ledger);current=ledger.snapshot()
    # Baseline generation forks RNG in the full recipe and does not change math.
    equality=base.digest(numerical(expected))==base.digest(numerical(actual))
    if not equality:raise AssertionError('Budgeted fresh-process numerical replay differs')
    if current['reserved']['train_updates']!=7 or current['completed']['train_updates']!=6:
        raise AssertionError('Later charged failure was reset or treated as completed update')
    after={name:r.file_digest(ROOT/name) for name in names}
    r.write_exclusive_json(output/'source-after.json',after)
    if before!=after:raise AssertionError('Collection source drift')
    artifacts={str(path.relative_to(ROOT)):r.file_digest(path) for path in sorted(output.rglob('*')) if path.is_file()}
    verification=dict(schema='dongxi-dpo-cumulative-work-cpu-v1',status='pass',command=sys.orig_argv,
        tests=tests,test_count=63,source_sha256=before,artifact_sha256=artifacts,
        actual_contract_sha256=canonical_hash(values[4]),environment=values[4]['environment'],
        fresh_execution=execution,numerical_state_digest=base.digest(numerical(expected)),
        exact_fresh_process_replay=equality,full_training_counters=full['counters'],
        full_ledger=full_state,recovery_ledger=current,later_failure=failure,
        scope='actual DPO functions, random local CPUFP32/activation checkpointing; cooperative logical reservations only',
        pending=['pretrained/CUDA/BF16','other runner work integration','physical quota/cgroups','FLOPs/backward recomputation',
                 'model-scale overhead','privileged rollback/tamper resistance','language quality','external campaign outcomes'])
    r.write_exclusive_json(output/'verification.json',verification)
    print(json.dumps(dict(status='pass',tests=63,verification=str(output/'verification.json'),fresh_replay=equality)))


if __name__=='__main__':
    if len(sys.argv)==6 and sys.argv[1]=='--child':child(*sys.argv[2:])
    elif len(sys.argv)==3 and sys.argv[1]=='--collect':collect(sys.argv[2])
    else:unittest.main()
