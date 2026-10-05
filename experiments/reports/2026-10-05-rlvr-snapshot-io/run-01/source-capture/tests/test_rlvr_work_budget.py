"""Original bounded actual-loop RLVR work and durable spending controls."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import test_rlvr_runner_recovery as base
from dongxi_llms import qwen_rlvr_lab as r
from dongxi_llms.batched_cache_lab import digest
from dongxi_llms.run_identity import canonical_hash
from dongxi_llms.training_snapshot import save_snapshot
from dongxi_llms.work_budget import WorkLedger,WorkBudgetExceeded,_decode

ROOT=base.ROOT;BOUND=1024**2
CAPS={key:100000 for key in r.BUDGET_KEYS}
OBSERVATIONS=[]


def fixture(seed=2323,caps=None):
    initial,old=base.make_loop(seed);budget=r.work_budget_contract(CAPS if caps is None else caps,BOUND)
    loop=r.RLVRLoop(initial.model,initial.reference,initial.optimizer,initial.generator,base.TRAIN,base.decode,
        eos_id=0,stop_ids=[0,7],group_size=3,max_new_tokens=4,updates=4,seed=seed,
        context_length=32,beta=.02,work_budget=budget)
    observed={k:deepcopy(v) for k,v in old.items() if k not in ('loop_contract','reference_sha256')}
    observed['sources'].update({name:r.file_digest(ROOT/name) for name in
        ('src/dongxi_llms/work_budget.py','tests/test_rlvr_work_budget.py')})
    observed['work_budget']=budget
    return loop,r.full_contract(observed,loop)


def ledger(path,loop,contract,*,existing=False,invocation='original'):
    value=(WorkLedger.open if existing else WorkLedger.create)(path,limits=contract['work_budget']['limits'],
        contract_sha256=canonical_hash(contract),max_bytes=BOUND,invocation_id=invocation)
    loop.bind_work_ledger(value,contract);return value


def numerical(loop):
    state=deepcopy(loop.snapshot_state());state.pop('work_ledger',None)
    state['loop_contract'].pop('work_budget',None)
    return state


def restore(loop,path,contract,receipt):
    return loop.restore(path,contract=contract,expected_sha256=receipt['payload_sha256'],
        expected_bytes=receipt['payload_bytes'],max_bytes=base.LIMIT)


def evaluation(loop,work_ledger,on_row=None):
    return r.evaluate_model(loop.model,[(2,2),(2,3)],lambda prompt:torch.tensor([[1,5,2]]),
        base.decode,[0,7],4,32,on_row=on_row,work_ledger=work_ledger)


class RLVRWorkTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='dongxi-rlvr-work-');self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)

    def directory(self,name):
        path=self.root/name;path.mkdir(mode=0o700);return path

    def make(self,seed=2323,caps=None,name='accounting'):
        loop,contract=fixture(seed,caps);path=self.directory(name)/'journal.jsonl'
        account=ledger(path,loop,contract);self.addCleanup(account.close)
        return loop,contract,account,path

    def test_actual_budgeted_unbudgeted_and_archived_equation_parity(self):
        for seed in (2323,2324):
            ordinary,_=base.make_loop(seed);bounded,contract,account,_=self.make(seed,name=str(seed))
            original,_=base.make_loop(seed);archived=base.legacy_update();records=[]
            for index in range(4):
                source=base.TRAIN[index%len(base.TRAIN)]
                expected=archived(original.model,original.reference,torch.tensor([source['prompt_ids']]),
                    source['expected'],base.decode,0,original.optimizer,original.generator,
                    group_size=3,max_new_tokens=4,beta=.02,context_length=32,stop_ids=[0,7])
                found=bounded.apply_pending();plain=ordinary.apply_pending()
                for key,value in expected.items():self.assertEqual(found[key],value)
                self.assertEqual(digest(found),digest(plain))
                self.assertEqual(digest(bounded.model.state_dict()),digest(original.model.state_dict()))
                self.assertEqual(digest(bounded.optimizer.state_dict()),digest(original.optimizer.state_dict()))
                self.assertTrue(torch.equal(bounded.generator.get_state(),original.generator.get_state()))
                records.append(r.jsonable(found))
            self.assertEqual(digest(numerical(bounded)),digest(ordinary.snapshot_state()))
            state=account.snapshot();self.assertEqual(state['completed']['train_updates'],4)
            self.assertEqual(state['completed']['collections'],4)
            self.assertEqual(state['reserved']['generated_slots'],48)
            self.assertEqual(state['reserved']['multinomial_draws'],48)
            self.assertGreater(state['completed']['recovery_validation_calls'],0)
            OBSERVATIONS.append(dict(control='actual-unforced-parity',seed=seed,records=records,
                numerical_sha256=digest(numerical(bounded)),work=state))

    def test_collection_refused_before_forward_multinomial_or_state_change(self):
        caps=dict(CAPS,generation_forward_positions=41)
        loop,_,account,_=self.make(caps=caps)
        before=digest(numerical(loop));rng=loop.generator.get_state().clone()
        with patch.object(r,'model_logits') as forward,patch.object(torch,'multinomial') as sample:
            with self.assertRaises(WorkBudgetExceeded):loop.collect()
        forward.assert_not_called();sample.assert_not_called()
        self.assertEqual(before,digest(numerical(loop)));self.assertTrue(torch.equal(rng,loop.generator.get_state()))
        self.assertFalse(loop.poisoned);self.assertEqual(account.snapshot()['sequence'],0)

    def test_application_refused_before_pending_check_or_optimizer(self):
        loop,_,account,_=self.make(caps=dict(CAPS,train_updates=0));loop.collect()
        before=digest(numerical(loop));prefix=account.snapshot()
        with (patch.object(torch.func,'functional_call') as validation,patch.object(r,'model_logits') as forward,
                patch.object(loop.optimizer,'step') as step):
            with self.assertRaises(WorkBudgetExceeded):loop.apply_pending()
        validation.assert_not_called();forward.assert_not_called();step.assert_not_called()
        self.assertEqual(before,digest(numerical(loop)));self.assertEqual(prefix,account.snapshot())
        self.assertIsNotNone(loop.pending);self.assertFalse(loop.poisoned)

    def test_payload_and_full_evaluation_refusals_precede_model_work(self):
        loop,contract,account,_=self.make(caps=dict(CAPS,recovery_validation_operations=0));loop.collect()
        with (patch.object(torch.func,'functional_call') as forward,patch.object(torch,'multinomial') as sample,
                patch.object(r,'save_snapshot') as save):
            with self.assertRaises(WorkBudgetExceeded):loop.save(self.root/'refused.pt',contract=contract,
                parent_invocation='refused',max_bytes=base.LIMIT)
        forward.assert_not_called();sample.assert_not_called();save.assert_not_called()
        other,_,account2,_=self.make(caps=dict(CAPS,evaluation_calls=7),name='evaluation-accounting')
        with patch.object(r,'model_logits') as forward:
            with self.assertRaises(WorkBudgetExceeded):evaluation(other,account2)
        forward.assert_not_called();self.assertEqual(account2.snapshot()['sequence'],0)

    def test_reservation_fsync_failure_precedes_any_model_or_sample(self):
        loop,_,account,_=self.make()
        with (patch('dongxi_llms.work_budget.os.fsync',side_effect=OSError('authored fsync failure')),
                patch.object(r,'model_logits') as forward,patch.object(torch,'multinomial') as sample):
            with self.assertRaises(Exception):loop.collect()
        forward.assert_not_called();sample.assert_not_called();self.assertTrue(account.poisoned)
        self.assertEqual(loop.completed,0);self.assertIsNone(loop.pending)

    def test_failed_second_generation_call_preserves_known_and_unknown_work(self):
        loop,_,account,_=self.make();actual=r.model_logits;calls=0
        def failed(network,ids):
            nonlocal calls
            calls+=1
            if calls==2:raise OSError('authored entered second forward')
            return actual(network,ids)
        with patch.object(r,'model_logits',side_effect=failed),self.assertRaises(OSError):loop.collect()
        state=account.snapshot()
        self.assertEqual(state['known_partial']['generation_forward_calls'],1)
        self.assertEqual(state['attempted_upper']['generation_forward_calls'],2)
        self.assertEqual(state['uncertain_upper']['generation_forward_positions'],9)
        self.assertEqual(state['known_partial']['multinomial_draws'],3)
        self.assertEqual(state['reserved']['generated_slots'],12)
        self.assertTrue(loop.poisoned);self.assertIsNone(loop.pending)
        OBSERVATIONS.append(dict(control='authored-second-generation-failure',work=state,attempt=deepcopy(loop.attempts[-1])))

    def test_reference_and_optimizer_failures_retain_application_charge(self):
        for stage in ('reference','optimizer'):
            loop,contract,account,_=self.make(name=stage);loop.collect()
            path=self.root/(stage+'.pt');receipt=loop.save(path,contract=contract,parent_invocation='before-failure',max_bytes=base.LIMIT)
            if stage=='reference':control=patch.object(r,'model_logits',side_effect=OSError('authored reference failure'))
            else:control=patch.object(loop.optimizer,'step',side_effect=OSError('authored optimizer failure'))
            with control,self.assertRaises(OSError):loop.apply_pending()
            after=account.snapshot();self.assertTrue(loop.poisoned)
            self.assertEqual(after['reserved']['train_updates'],1)
            if stage=='reference':self.assertEqual(after['uncertain_upper']['reference_forward_calls'],1)
            else:self.assertEqual(after['uncertain_upper']['train_updates'],1)
            restore(loop,path,contract,receipt)
            self.assertFalse(loop.poisoned);self.assertEqual(loop.completed,0)
            self.assertEqual(account.snapshot()['reserved']['train_updates'],1)
            with patch.object(loop,'collect',side_effect=AssertionError('No resample')):loop.apply_pending()
            self.assertEqual(account.snapshot()['reserved']['train_updates'],2)
            OBSERVATIONS.append(dict(control='authored-application-failure',stage=stage,failed=after,recovered=account.snapshot()))

    def test_same_process_completed_pending_replay_all_declared_seeds(self):
        for seed in (2323,2324):
            for phase in ('completed','pending'):
                label=str(seed)+'-'+phase;loop,contract,account,path=self.make(seed,name=label)
                loop.apply_pending();loop.apply_pending()
                if phase=='pending':loop.collect()
                snapshot=self.root/(label+'.pt');receipt=loop.save(snapshot,contract=contract,parent_invocation='original',max_bytes=base.LIMIT)
                expected=[]
                while loop.completed<4:expected.append(loop.apply_pending())
                expected_state=digest(numerical(loop));later=account.snapshot();account.close()
                fresh,current=fixture(seed);self.assertEqual(current,contract)
                with ledger(path,fresh,contract,existing=True,invocation='resumed') as restored:
                    restore(fresh,snapshot,contract,receipt)
                    self.assertEqual(restored.snapshot()['reserved']['train_updates'],later['reserved']['train_updates'])
                    if phase=='pending':
                        with patch.object(fresh,'collect',side_effect=AssertionError('No new pending collection')):found=[fresh.apply_pending()]
                    else:found=[]
                    while fresh.completed<4:found.append(fresh.apply_pending())
                    self.assertEqual(digest(found),digest(expected));self.assertEqual(digest(numerical(fresh)),expected_state)
                    self.assertEqual(restored.snapshot()['reserved']['train_updates'],6)
                    OBSERVATIONS.append(dict(control='same-process-replay-retained-later-spending',seed=seed,
                        phase=phase,numerical_sha256=expected_state,work=restored.snapshot()))

    def test_fresh_process_completed_pending_replay_without_reset(self):
        for phase in ('completed','pending'):
            loop,contract,account,path=self.make(name='fresh-'+phase)
            loop.apply_pending();loop.apply_pending()
            if phase=='pending':loop.collect()
            snapshot=self.root/(phase+'.pt');receipt=loop.save(snapshot,contract=contract,parent_invocation='original',max_bytes=base.LIMIT)
            contract_path=self.root/(phase+'-contract.json');contract_path.write_text(json.dumps(contract))
            tail=[]
            while loop.completed<4:tail.append(loop.apply_pending())
            expected=digest(numerical(loop));account.close()
            command=[sys.executable,str(Path(__file__).resolve()),'--child',str(snapshot),'--contract',str(contract_path),
                '--sha256',receipt['payload_sha256'],'--bytes',str(receipt['payload_bytes']),'--journal',str(path)]
            process=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=60)
            self.assertEqual(process.returncode,0,process.stderr)
            result=json.loads(process.stdout.strip().splitlines()[-1])
            self.assertEqual(result['tail_sha256'],digest(tail));self.assertEqual(result['state_sha256'],expected)
            self.assertEqual(result['work']['reserved']['train_updates'],6)
            self.assertEqual(result['no_collection_first_pending'],phase=='pending')
            OBSERVATIONS.append(dict(control='fresh-process',phase=phase,command=command,
                actual_exit_code=process.returncode,stdout=process.stdout,stderr=process.stderr,result=result))

    def test_old_snapshot_keeps_later_failed_collection_and_refuses_retry(self):
        caps=dict(CAPS,collections=1);loop,contract,account,path=self.make(caps=caps)
        snapshot=self.root/'initial.pt';receipt=loop.save(snapshot,contract=contract,parent_invocation='original',max_bytes=base.LIMIT)
        with (patch.object(r,'model_logits',side_effect=OSError('authored later failed collect')),
                self.assertRaises(OSError)):loop.collect()
        after=account.snapshot();account.close()
        fresh,current=fixture(caps=caps);self.assertEqual(current,contract)
        with ledger(path,fresh,contract,existing=True,invocation='new-output') as reopened:
            restore(fresh,snapshot,contract,receipt);self.assertEqual(fresh.completed,0)
            self.assertEqual(reopened.snapshot()['reserved']['collections'],1)
            with (patch.object(r,'model_logits') as forward,patch.object(torch,'multinomial') as sample,
                    self.assertRaises(WorkBudgetExceeded)):fresh.collect()
            forward.assert_not_called();sample.assert_not_called()
            self.assertEqual(reopened.snapshot()['failed_tickets'],after['failed_tickets'])

    def test_changed_caps_copied_journal_and_unbound_new_allowance_refused(self):
        loop,contract,account,path=self.make();snapshot=self.root/'initial.pt'
        receipt=loop.save(snapshot,contract=contract,parent_invocation='original',max_bytes=base.LIMIT)
        account.close();copy=self.directory('copied')/'journal.jsonl';copy.write_bytes(path.read_bytes());copy.chmod(0o600)
        for candidate,caps in ((copy,CAPS),(path,dict(CAPS,collections=99999))):
            with self.assertRaises(ValueError):WorkLedger.open(candidate,limits=caps,contract_sha256=canonical_hash(contract),
                max_bytes=BOUND,invocation_id='wrong')
        fresh,_=fixture();newpath=self.directory('new-allowance')/'journal.jsonl'
        with ledger(newpath,fresh,contract,invocation='wrong'):
            with patch.object(fresh.model,'load_state_dict') as load,self.assertRaises(ValueError):restore(fresh,snapshot,contract,receipt)
            load.assert_not_called()

    def test_snapshot_prefix_tamper_is_rejected_before_validation_forward_or_apply(self):
        loop,contract,account,_=self.make();loop.collect();state=deepcopy(loop.snapshot_state())
        state['work_ledger']['reserved']['collections']=True
        path=self.root/'invalid.pt';receipt=save_snapshot(path,contract=contract,state=state,completed_updates=0,
            phase='pending',parent_invocation='authored-tamper',max_bytes=base.LIMIT)
        with (patch.object(torch.func,'functional_call') as forward,patch.object(loop.model,'load_state_dict') as apply,
                self.assertRaises(ValueError)):restore(loop,path,contract,receipt)
        forward.assert_not_called();apply.assert_not_called()

    def test_active_contract_and_malformed_history_refuse_before_validation_work(self):
        loop,contract,account,_=self.make();loop.apply_pending();state=deepcopy(loop.snapshot_state())
        changed=deepcopy(contract);changed['seed']+=1;prefix=account.snapshot()
        with patch.object(torch.func,'functional_call') as forward,patch.object(torch,'multinomial') as sample:
            with self.assertRaisesRegex(ValueError,'Snapshot contract'):loop.save(self.root/'wrong.pt',contract=changed,
                parent_invocation='wrong',max_bytes=base.LIMIT)
            for history in ([None],[{}]):
                invalid=deepcopy(state);invalid['history']=history
                with self.assertRaises(ValueError):loop.validate_payload(dict(state=invalid,completed_updates=1,phase='completed'))
        forward.assert_not_called();sample.assert_not_called();self.assertEqual(prefix,account.snapshot())

    def test_evaluation_retains_complete_and_partial_actual_calls(self):
        loop,_,account,_=self.make();rows=[];panel=evaluation(loop,account,on_row=rows.append)
        state=account.snapshot();calls=sum(row['successful_forward_calls'] for row in panel['rows'])
        positions=sum(row['successful_forward_positions'] for row in panel['rows'])
        self.assertEqual(state['reserved']['evaluation_calls'],8)
        self.assertEqual(state['reserved']['evaluation_positions'],36)
        self.assertEqual(state['completed']['evaluation_calls'],calls)
        self.assertEqual(state['completed']['evaluation_positions'],positions)
        self.assertEqual(state['completed']['evaluation_tokens'],sum(row['generated_tokens'] for row in rows))
        actual=r.model_logits;number=0;partial=[]
        def fail(network,ids):
            nonlocal number
            number+=1
            if number==2:raise OSError('authored evaluation second call')
            return actual(network,ids)
        # The actual fixed model may stop in one call; inject only a declared
        # second-call failure, retaining whichever row it reaches.
        with patch.object(r,'model_logits',side_effect=fail),self.assertRaises(OSError):evaluation(loop,account,on_row=partial.append)
        self.assertEqual(account.snapshot()['known_partial']['evaluation_calls'],1)
        self.assertEqual(account.snapshot()['uncertain_upper']['evaluation_calls'],1)
        self.assertEqual(partial[-1]['stop'],'error')
        OBSERVATIONS.append(dict(control='actual-evaluation-with-authored-failure',panel=panel,partial=partial,work=account.snapshot()))

    def test_eval_observer_and_lifecycle_failures_keep_last_durable_and_spending(self):
        loop,_,account,_=self.make();seen=[]
        def fail(row):seen.append(row);raise OSError('authored row observer failure')
        with self.assertRaises(OSError):evaluation(loop,account,on_row=fail)
        self.assertGreater(account.snapshot()['known_partial']['evaluation_calls'],0)
        for stage in ('baseline','metric','final','export','commit-observer'):
            loop,contract,account,_=self.make(name=stage);output=self.directory('output-'+stage);receipts=[]
            def callback(receipt):
                receipts.append(receipt)
                if stage=='commit-observer':raise OSError('authored commit observer')
            def evaluate(name):
                result=evaluation(loop,account)
                if stage==name:raise OSError('authored '+name)
                return result
            def metric(row):
                if stage=='metric':raise OSError('authored metric')
            def export():
                if stage=='export':raise OSError('authored export')
            with self.assertRaises(OSError):r.run_loop_with_recovery(loop,contract=contract,output=output,
                parent_invocation='failure',max_bytes=base.LIMIT,on_commit=callback,metric_sink=metric,
                baseline=lambda:evaluate('baseline'),final=lambda:evaluate('final'),export=export)
            last=receipts[-1];completed=0 if stage in ('baseline','commit-observer') else 1 if stage=='metric' else 4
            self.assertEqual(last['completed_updates'],completed)
            restore(loop,Path(last['path']),contract,last);self.assertEqual(loop.completed,completed)
            if stage in ('baseline','final','export'):self.assertGreater(account.snapshot()['reserved']['evaluation_calls'],0)
            OBSERVATIONS.append(dict(control='lifecycle-failure',stage=stage,last_durable_update=completed,work=account.snapshot()))

    def test_failed_pending_save_keeps_old_boundary_and_new_collection_charge(self):
        loop,contract,account,_=self.make();old=self.root/'old.pt'
        receipt=loop.save(old,contract=contract,parent_invocation='initial',max_bytes=base.LIMIT);original=old.read_bytes()
        loop.collect()
        with patch.object(r,'save_snapshot',side_effect=OSError('authored serializer refusal')),self.assertRaises(OSError):
            loop.save(self.root/'failed.pt',contract=contract,parent_invocation='pending',max_bytes=base.LIMIT)
        self.assertEqual(original,old.read_bytes());restore(loop,old,contract,receipt)
        self.assertEqual(loop.completed,0);self.assertIsNone(loop.pending)
        self.assertEqual(account.snapshot()['reserved']['collections'],1)

    def test_cap_contract_and_file_schema_bounds(self):
        for value in (dict(CAPS,collections=True),dict(CAPS,unknown=1),{k:v for k,v in CAPS.items() if k!='collections'}):
            with self.assertRaises(ValueError):r.work_budget_contract(value,BOUND)
        for size in (True,0,64*1024**2+1):
            with self.assertRaises(ValueError):r.work_budget_contract(CAPS,size)
        file=self.root/'caps.json';file.write_text(json.dumps(CAPS));self.assertEqual(_decode(r.read_work_limits_file(file)),CAPS)
        file.write_text('{"collections":1,"collections":2}')
        with self.assertRaises(ValueError):_decode(r.read_work_limits_file(file))
        file.write_text('x'*65537)
        with self.assertRaises(ValueError):r.read_work_limits_file(file)
        with self.assertRaises(ValueError):r.read_work_limits_file(self.root)
        target=self.root/'target';target.write_text('{}');link=self.root/'link';link.symlink_to(target)
        with self.assertRaises(OSError):r.read_work_limits_file(link)
        directory=self.directory('real');ancestor=self.root/'alias';ancestor.symlink_to(directory)
        (directory/'caps').write_text('{}')
        with self.assertRaises(OSError):r.read_work_limits_file(ancestor/'caps')

    def test_fifo_cli_rejects_before_output_or_model_work(self):
        from snapshot_io_test_support import explicit_snapshot_io_args
        fifo=self.root/'caps-fifo';os.mkfifo(fifo,0o600);output=self.root/'never-created'
        command=[sys.executable,'-m','dongxi_llms.qwen_rlvr_lab','--model-dir',str(self.root),
            '--revision','a'*40,'--output',str(output),'--work-limits',str(fifo),'--work-journal-max-bytes',str(BOUND),
            '--snapshot-max-bytes',str(base.LIMIT), *explicit_snapshot_io_args(self.root)]
        started=time.monotonic();process=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=10)
        self.assertEqual(process.returncode,1);self.assertIn('regular no-follow',process.stderr)
        self.assertFalse(output.exists())
        OBSERVATIONS.append(dict(control='actual-no-peer-fifo-cli',command=command,actual_exit_code=process.returncode,
            seconds=time.monotonic()-started,stdout=process.stdout,stderr=process.stderr,output_created=False))

    def test_changed_parsed_cap_bytes_refused_before_tokenizer_or_model_load(self):
        from snapshot_io_test_support import explicit_snapshot_io_args
        file=self.root/'caps.json';file.write_text(json.dumps(CAPS));raw=file.read_bytes()
        expected=hashlib.sha256(raw).hexdigest();output=self.root/'identity-evidence'
        journal=r.RunEvidence(output,{})
        journal.work_budget=r.work_budget_contract(CAPS,BOUND);journal.work_limits_sha256=expected
        io_args=explicit_snapshot_io_args(self.root)
        args=SimpleNamespace(environment_lock=ROOT/'uv.lock',work_limits=file,work_journal_max_bytes=BOUND,
            template=None,resume=None,snapshot_max_bytes=base.LIMIT,max_seconds=600,
            snapshot_io_limits=Path(io_args[1]),snapshot_io_ledger=Path(io_args[3]),resume_io_receipt=None)
        def changed(*args,**kwargs):
            file.write_text(json.dumps(dict(CAPS,collections=99999)));return {'authored_mock_identity':True}
        with (patch.object(r,'collect_run_identity',side_effect=changed),
                patch('transformers.AutoTokenizer.from_pretrained') as tokenizer,
                patch('transformers.AutoModelForCausalLM.from_pretrained') as model):
            with self.assertRaisesRegex(ValueError,'initial parse'):r.execute_run(args,journal)
        tokenizer.assert_not_called();model.assert_not_called()
        saved=json.loads((output/'report.json').read_text())
        self.assertEqual(saved['parsed_work_limits_sha256'],expected)
        self.assertEqual(saved['stage'],'before-load')
        self.assertFalse((output/'work-accounting').exists())
        OBSERVATIONS.append(dict(control='authored-change-after-cap-parse',parsed_sha256=expected,
            later_sha256=r.file_digest(file),stage=saved['stage'],tokenizer_model_calls=0))

    def test_cap_identity_and_closure_use_nonblocking_reader_not_generic_digest(self):
        file=self.root/'caps.json';file.write_text(json.dumps(CAPS));sha=r.file_digest(file)
        identity=dict(input_sha256={},source_sha256={},environment={'environment_lock':{'path':str(ROOT/'uv.lock'),
            'sha256':r.file_digest(ROOT/'uv.lock')}},checkpoint_path=str(self.root),checkpoint_files={})
        r.attach_work_limits_identity(identity,ROOT,file,sha)
        actual=r.file_digest
        def checked(path,guard=None):
            if Path(path)==file:raise AssertionError('Cap may not use generic blocking digest')
            return actual(path,guard)
        with patch.object(r,'file_digest',side_effect=checked),patch.object(r,'artifact_hashes',return_value={}):
            r.verify_identity_files(identity,ROOT)
            file.unlink();os.mkfifo(file,0o600)
            with self.assertRaisesRegex(ValueError,'regular no-follow'):r.verify_identity_files(identity,ROOT)
        OBSERVATIONS.append(dict(control='replaced-no-peer-fifo-at-identity-closure',generic_cap_reads=0,
            blocking_read=False,identity_sha256=identity['identity_sha256']))


def child(args):
    contract=json.loads(args.contract.read_text());loop,current=fixture(contract['seed'],contract['work_budget']['limits'])
    if current!=contract:raise ValueError('Fresh-process scientific source/recipe identity changed')
    with ledger(args.journal,loop,contract,existing=True,invocation='fresh-child') as account:
        payload=restore(loop,args.child,contract,dict(payload_sha256=args.sha256,payload_bytes=args.bytes));tail=[]
        if payload['phase']=='pending':
            with patch.object(loop,'collect',side_effect=AssertionError('Fresh pending must not resample')):tail.append(loop.apply_pending())
        while loop.completed<4:tail.append(loop.apply_pending())
        print(json.dumps(dict(phase=payload['phase'],tail_sha256=digest(tail),state_sha256=digest(numerical(loop)),
            no_collection_first_pending=payload['phase']=='pending',work=account.snapshot())))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--child',type=Path);parser.add_argument('--contract',type=Path)
    parser.add_argument('--sha256');parser.add_argument('--bytes',type=int);parser.add_argument('--journal',type=Path)
    parser.add_argument('--collect',type=Path);parser.add_argument('--observations',type=Path);args=parser.parse_args()
    if args.child:return child(args)
    if args.collect:
        args.collect.mkdir(mode=0o700,parents=True,exist_ok=False);raw=args.collect/'observations.json'
        paths=['src/dongxi_llms/qwen_rlvr_lab.py','src/dongxi_llms/grpo_lab.py','src/dongxi_llms/work_budget.py',
            'src/dongxi_llms/training_snapshot.py','src/dongxi_llms/batched_cache_lab.py',
            'tests/test_rlvr_work_budget.py','tests/test_rlvr_runner_recovery.py',
            'experiments/specs/2026-10-05-rlvr-cumulative-work.md',
            'experiments/specs/2026-10-05-rlvr-cap-byte-binding.md',
            'experiments/specs/2026-10-05-rlvr-bounded-cap-identity.md',
            'experiments/reports/2026-10-05-rlvr-runner-recovery-original.py.txt','uv.lock']
        hashes=lambda:{path:r.file_digest(ROOT/path) for path in paths}
        before=hashes();command=[sys.executable,str(Path(__file__).resolve()),'--observations',str(raw)]
        started=time.monotonic();process=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=120)
        report=dict(command=command,actual_exit_code=process.returncode,seconds=time.monotonic()-started,
            stdout=process.stdout,stderr=process.stderr,source_before=before,source_after=hashes(),
            environment=dict(executable=sys.executable,python=sys.version,platform=platform.platform(),
                torch=str(torch.__version__),cuda_available=torch.cuda.is_available()),
            observations=str(raw),observations_sha256=r.file_digest(raw) if raw.exists() else None,
            scope='actual random local tiny CPU RLVR; authored refusals; no pretrained/GPU/quota evidence')
        with (args.collect/'verification.json').open('x') as handle:json.dump(report,handle,indent=2);handle.write('\n')
        print(json.dumps(dict(directory=str(args.collect),actual_exit_code=process.returncode)))
        raise SystemExit(process.returncode)
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(RLVRWorkTests),
        unittest.defaultTestLoader.loadTestsFromTestCase(base.RLVRRecoveryTests)])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if args.observations:
        with args.observations.open('x') as handle:json.dump(r.jsonable(OBSERVATIONS),handle,indent=2);handle.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__=='__main__':main()
