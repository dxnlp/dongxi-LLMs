"""Actual runner-owned RLVR recovery with original local random HF CPU models."""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
from dongxi_llms import qwen_rlvr_lab as runner
from dongxi_llms.batched_cache_lab import digest
from dongxi_llms.training_snapshot import save_snapshot

ROOT=Path(__file__).resolve().parents[1]
LIMIT=16*1024**2
TRAIN=[dict(source_id='cpu-source-0',prompt='Authored integer 0',prompt_ids=[1,2],expected=0),
       dict(source_id='cpu-source-1',prompt='Authored integer 1',prompt_ids=[1,3,2],expected=1),
       dict(source_id='cpu-source-2',prompt='Authored integer 2',prompt_ids=[1,4,3,2],expected=2)]
EVALUATION=[dict(source_id='cpu-heldout-0',prompt='Authored heldout 4',prompt_ids=[1,5,2],expected=4)]
OBSERVATIONS=[]


def decode(ids):
    return ''.join(str(i-8) if 8<=i<=15 else '?' for i in ids)


def make_loop(seed=2323):
    from transformers import Qwen3Config,Qwen3ForCausalLM,__version__
    torch.set_num_threads(1);random.seed(seed);torch.manual_seed(seed)
    cfg=Qwen3Config(vocab_size=16,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=1,head_dim=8,max_position_embeddings=32,
        attention_dropout=0.,tie_word_embeddings=False,bos_token_id=1,eos_token_id=0,pad_token_id=0)
    cfg._attn_implementation='sdpa'
    model=Qwen3ForCausalLM(cfg).eval();reference=deepcopy(model).eval().requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.008,weight_decay=0.)
    loop=runner.RLVRLoop(model,reference,optimizer,torch.Generator().manual_seed(seed),TRAIN,decode,
        eos_id=0,stop_ids=[0,7],group_size=3,max_new_tokens=4,updates=4,seed=seed,context_length=32,beta=.02)
    sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
        ['src/dongxi_llms/qwen_rlvr_lab.py','src/dongxi_llms/grpo_lab.py',
         'src/dongxi_llms/batched_cache_lab.py','src/dongxi_llms/training_snapshot.py']}
    observed=runner.observable_contract(parent_files={'authored-random-initial-policy':digest(model.state_dict())},
        sources=sources,inputs={'authored-token-fixture':digest((TRAIN,EVALUATION))},
        environment={'python':platform.python_version(),'torch':str(torch.__version__),'transformers':__version__,
                     'lock_sha256':hashlib.sha256((ROOT/'uv.lock').read_bytes()).hexdigest()},
        interface={'source':{'tokenizer_id':'authored-integer-ID-alphabet','tokenizer_revision':'1'*40},
                   'encoding':'explicit original token sequences; no downloaded tokenizer',
                   'stops':[0,7]},train=TRAIN,evaluation=EVALUATION,seed=seed,updates=4,
        group_size=3,max_new_tokens=4,lr=.008,beta=.02,eos_id=0,stop_ids=[0,7],context_length=32,
        dtype='torch.float32',device={'mode':'cpu','name':'CPU'})
    return loop,runner.full_contract(observed,loop)


def restore(loop,path,contract,header):
    return loop.restore(path,contract=contract,expected_sha256=header['payload_sha256'],
                        expected_bytes=header['payload_bytes'],max_bytes=LIMIT)


def reseal(pool):
    pool['pool_sha256']=digest({k:v for k,v in pool.items() if k!='pool_sha256'})


def legacy_update():
    path=ROOT/'experiments/reports/2026-10-05-rlvr-runner-recovery-original.py.txt'
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='update_model')
    space={name:getattr(runner,name) for name in ('model_logits','group_advantages','exact_kl','clipped_objective','verify_integer')}
    space['torch']=torch
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),space)
    return space['update_model']


class RLVRRecoveryTests(unittest.TestCase):
    def test_wrapper_matches_retained_original_equations_exactly(self):
        for seed in (2323,2324):
            actual,_=make_loop(seed);original,_=make_loop(seed)
            options=dict(group_size=3,max_new_tokens=4,beta=.02,context_length=32,stop_ids=[0,7])
            records=[]
            for update in range(4):
                source=TRAIN[update%len(TRAIN)];prompt=torch.tensor([source['prompt_ids']])
                expected=legacy_update()(original.model,original.reference,prompt,source['expected'],decode,0,
                                       original.optimizer,original.generator,**options)
                found=runner.update_model(actual.model,actual.reference,prompt,source['expected'],decode,0,
                                          actual.optimizer,actual.generator,**options)
                self.assertEqual(found,expected)
                self.assertEqual(digest(actual.model.state_dict()),digest(original.model.state_dict()))
                self.assertEqual(digest(actual.optimizer.state_dict()),digest(original.optimizer.state_dict()))
                self.assertTrue(torch.equal(actual.generator.get_state(),original.generator.get_state()))
                records.append(found)
            OBSERVATIONS.append(dict(control='retained-original-equation-parity',seed=seed,records=records))

    def test_authored_stop_cap_and_zero_signal_controls(self):
        real=torch.multinomial
        for control in ('stop','cap','zero-signal'):
            loop,_=make_loop();steps=0
            def authored(probabilities,num_samples,generator):
                nonlocal steps
                real(probabilities,num_samples,generator=generator)  # Actual rectangular RNG geometry.
                steps+=1
                if control=='cap':return torch.tensor([[8],[9],[10]])
                if control=='zero-signal':return torch.zeros((3,1),dtype=torch.long)
                return torch.tensor([[0],[8],[7]]) if steps==1 else torch.tensor([[10],[0],[11]])
            with patch.object(runner.torch,'multinomial',side_effect=authored):pool=loop.collect()
            loop.validate_pool(pool,0,0,policy=loop.model.state_dict())
            if control=='stop':
                self.assertEqual(pool['mask'].tolist(),[[True,False],[True,True],[True,False]])
                self.assertEqual(pool['responses'].tolist(),[[0,0],[8,0],[7,0]])
                self.assertEqual(pool['rewards'],[0.,1.,0.])
            else:
                self.assertEqual(pool['rewards'],[0.,0.,0.])
                self.assertTrue(torch.equal(pool['advantages'],torch.zeros(3)))
                if control=='cap':self.assertEqual(pool['stops'],['token-limit']*3)
            OBSERVATIONS.append(dict(control='authored-'+control+'-not-sampled',pool=runner.jsonable(pool)))

    def test_signed_targets_and_digest_rng_lineage_gates(self):
        loop,_=make_loop();loop.collect();state=deepcopy(loop.snapshot_state())
        state['pending']['rng_after']=torch.Generator().manual_seed(999).get_state();reseal(state['pending'])
        with self.assertRaisesRegex(ValueError,'RNG after'):loop.validate_pool(state['pending'],0,0)
        loop.apply_pending();loop.apply_pending();state=deepcopy(loop.snapshot_state())
        state['history'][1]['collection']['policy_sha256']='0'*64;reseal(state['history'][1]['collection'])
        with self.assertRaisesRegex(ValueError,'lineage'):loop.validate_payload(dict(state=state,phase='completed',completed_updates=2))
        # Signed zero remains zero; an actually negative expected integer is allowed by the verifier contract.
        loop,_=make_loop();negative=deepcopy(TRAIN);negative[0]['expected']=-1
        loop=runner.RLVRLoop(loop.model,loop.reference,loop.optimizer,loop.generator,negative,decode,
            eos_id=0,stop_ids=[0,7],group_size=3,max_new_tokens=4,updates=4,seed=2323,context_length=32)
        pool=runner.collect_rollouts(loop.model,loop.reference,torch.tensor([[1,2]]),-1,decode,0,
            loop.generator,3,4,context_length=32,stop_ids=[0,7],source_id='cpu-source-0')
        loop.validate_pool(pool,0,0,policy=loop.model.state_dict())

    def test_byte_and_schema_failure_happen_before_apply(self):
        with tempfile.TemporaryDirectory() as directory:
            loop,contract=make_loop();path=Path(directory)/'initial.pt'
            header=loop.save(path,contract=contract,parent_invocation='original',max_bytes=LIMIT)
            for expected in ('0'*64,header['payload_sha256']):
                if expected==header['payload_sha256']:
                    data=bytearray(path.read_bytes());data[len(data)//2]^=1;path.write_bytes(data)
                with patch.object(loop.model,'load_state_dict',side_effect=AssertionError('No application')):
                    with self.assertRaises(ValueError):loop.restore(path,contract=contract,expected_sha256=expected,
                        expected_bytes=header['payload_bytes'],max_bytes=LIMIT)

    def test_completed_and_pending_replays_all_frozen_seeds(self):
        for seed in (2323,2324):
            for phase in ('completed','pending'):
                with self.subTest(seed=seed,phase=phase),tempfile.TemporaryDirectory() as directory:
                    loop,contract=make_loop(seed)
                    for _ in range(2):loop.apply_pending()
                    if phase=='pending':loop.collect()
                    path=Path(directory)/'boundary.pt'
                    header=loop.save(path,contract=contract,parent_invocation='original-cpu',max_bytes=LIMIT)
                    expected=[]
                    while loop.completed<4:expected.append(loop.apply_pending())
                    state=digest(loop.snapshot_state())
                    resumed,current=make_loop(seed);self.assertEqual(contract,current)
                    restore(resumed,path,contract,header)
                    if phase=='pending':
                        with patch.object(resumed,'collect',side_effect=AssertionError('No resample')):
                            actual=[resumed.apply_pending()]
                    else:actual=[]
                    while resumed.completed<4:actual.append(resumed.apply_pending())
                    self.assertEqual(digest(expected),digest(actual))
                    self.assertEqual(state,digest(resumed.snapshot_state()))
                    self.assertEqual(resumed.cursor,4);self.assertIsNone(resumed.pending)
                    OBSERVATIONS.append(dict(seed=seed,phase=phase,history=runner.jsonable(loop.history),
                        final_state_sha256=state,reference_sha256=loop.reference_sha256,
                        collection_work=loop.collection_work,application_work=loop.application_work,
                        exact_state_and_tail=True,no_collection_first_pending=(phase=='pending')))

    def test_fresh_process_completed_and_pending_actual_loop(self):
        for phase in ('completed','pending'):
            with self.subTest(phase=phase),tempfile.TemporaryDirectory() as directory:
                directory=Path(directory);loop,contract=make_loop()
                for _ in range(2):loop.apply_pending()
                if phase=='pending':loop.collect()
                path=directory/'boundary.pt';header=loop.save(path,contract=contract,parent_invocation='fresh-parent',max_bytes=LIMIT)
                contract_path=directory/'contract.json';contract_path.write_text(json.dumps(contract))
                tail=[]
                while loop.completed<4:tail.append(loop.apply_pending())
                command=[sys.executable,str(Path(__file__).resolve()),'--child',str(path),
                         '--contract',str(contract_path),'--sha256',header['payload_sha256'],
                         '--bytes',str(header['payload_bytes'])]
                child=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=60)
                self.assertEqual(child.returncode,0,child.stderr)
                result=json.loads(child.stdout.strip().splitlines()[-1])
                self.assertEqual(result['tail_sha256'],digest(tail))
                self.assertEqual(result['state_sha256'],digest(loop.snapshot_state()))
                self.assertEqual(result['phase'],phase)
                OBSERVATIONS.append(dict(control='fresh-process',phase=phase,command=command,
                    returncode=child.returncode,stdout=child.stdout,stderr=child.stderr,result=result))

    def test_authored_positive_pool_changes_policy_not_reference(self):
        loop,_=make_loop();pool=loop.collect()
        # Explicit authored application control, not claimed sampled output.
        pool['responses']=torch.tensor([[8,0],[9,0],[0,0]])
        pool['mask']=torch.tensor([[True,True],[True,True],[True,False]])
        inputs=torch.cat((pool['prompt_ids'].repeat(3,1),pool['responses']),-1)[:,:-1]
        with torch.no_grad():
            logits=runner.model_logits(loop.model,inputs)[:,pool['prompt_ids'].shape[1]-1:]
            pool['old_logp']=logits.log_softmax(-1).gather(-1,pool['responses'][...,None]).squeeze(-1)
        pool.update(texts=['0','1',''],rewards=[1.,0.,0.],stops=['eos']*3,final_stop_ids=[0]*3,
            advantages=runner.group_advantages(torch.tensor([[1.,0.,0.]])).flatten())
        pool['work']=runner.collection_work(2,2,3,5);reseal(pool)
        before=digest(loop.model.state_dict());reference=digest(loop.reference.state_dict())
        row=runner.apply_rollouts(loop.model,loop.reference,pool,loop.optimizer)
        self.assertGreater(row['gradient_norm'],0.)
        self.assertNotEqual(before,digest(loop.model.state_dict()))
        self.assertEqual(reference,digest(loop.reference.state_dict()))
        self.assertTrue(all(p.grad is None for p in loop.reference.parameters()))
        OBSERVATIONS.append(dict(control='authored-positive-application-not-sampled',result=row))

    def test_pending_semantic_tamper_rejected_before_any_state_application(self):
        loop,contract=make_loop();loop.apply_pending();loop.collect();base=deepcopy(loop.snapshot_state())
        variants=[]
        for key in ('cursor_before','cursor_after','version'):
            state=deepcopy(base);state['pending'][key]+=1;reseal(state['pending']);variants.append((key,state))
        for key in ('cursor','collection_work'):
            state=deepcopy(base)
            if key=='cursor':state[key]=True
            else:state[key]['generated_slots']+=1
            variants.append((key,state))
        state=deepcopy(base);state['pending']['old_logp']-=.5;reseal(state['pending']);variants.append(('resealed-old-likelihood',state))
        state=deepcopy(base);state['pending']['source_id']='wrong-source';reseal(state['pending']);variants.append(('source',state))
        state=deepcopy(base);state['pending']['mask'][0,0]=False;reseal(state['pending']);variants.append(('mask',state))
        state=deepcopy(base);state['pending']['advantages'].fill_(1.);reseal(state['pending']);variants.append(('advantage',state))
        state=deepcopy(base);state['pending']['rng_before']=torch.Generator().manual_seed(99).get_state();reseal(state['pending']);variants.append(('rng-chain',state))
        state=deepcopy(base);state['rng']['cuda']=None;variants.append(('cuda-type',state))
        state=deepcopy(base);state['reference']=deepcopy(state['policy']);variants.append(('reference-refresh',state))
        # A real policy change is separately authored because genuine zero-signal updates can be unchanged.
        state=deepcopy(base);name=next(iter(state['reference']));state['reference'][name].flatten()[0]+=1.;variants.append(('reference-byte-tamper',state))
        state=deepcopy(base);state['optimizer']['param_groups'][0]['params'][0]=False;variants.append(('bool-optimizer-id',state))
        state=deepcopy(base);index=next(iter(state['optimizer']['state']));state['optimizer']['state'][index]['step']=torch.tensor(True);variants.append(('bool-adam-step',state))
        with tempfile.TemporaryDirectory() as directory:
            for name,state in variants:
                if name=='reference-refresh' and digest(state['reference'])==loop.reference_sha256:
                    OBSERVATIONS.append(dict(control='reference-refresh-no-effect-when-policy-unmodified',negative=True));continue
                path=Path(directory)/(name+'.pt')
                header=save_snapshot(path,contract=contract,state=state,completed_updates=1,
                    parent_invocation='deliberate-invalid-state',phase='pending',max_bytes=LIMIT)
                fresh,_=make_loop();before=digest(fresh.snapshot_state())
                with self.subTest(name=name),patch.object(fresh.model,'load_state_dict',side_effect=AssertionError('Apply forbidden')):
                    with self.assertRaises(ValueError):restore(fresh,path,contract,header)
                self.assertEqual(before,digest(fresh.snapshot_state()))

    def test_full_contract_changes_reject_preload(self):
        loop,contract=make_loop()
        observed={k:v for k,v in contract.items() if k not in ('loop_contract','reference_sha256')}
        runner.verify_observable_contract(contract,observed)
        for key in ('sources','inputs','parent_files','interface','environment','seed','updates','group_size','max_new_tokens'):
            bad=deepcopy(observed);bad[key]='changed'
            with self.subTest(key=key),self.assertRaises(ValueError):runner.verify_observable_contract(contract,bad)

    def test_counter_history_bool_and_missing_rows_rejected(self):
        loop,_=make_loop();loop.apply_pending();base=deepcopy(loop.snapshot_state())
        for key in ('cursor','collection_work','application_work','history'):
            state=deepcopy(base)
            if key=='cursor':state[key]=True
            elif key=='history':state[key][0]['update']=True
            else:first=next(iter(state[key]));state[key][first]=True
            with self.subTest(key=key),self.assertRaises(ValueError):
                loop.validate_payload(dict(state=state,phase='completed',completed_updates=1))

    def test_failed_collection_retains_known_steps_but_is_not_pending(self):
        loop,_=make_loop();actual=runner.model_logits;calls=0
        def fail(network,ids):
            nonlocal calls
            calls+=1
            if calls==2:raise RuntimeError('injected second generation call')
            return actual(network,ids)
        with patch.object(runner,'model_logits',side_effect=fail),self.assertRaises(RuntimeError):loop.collect()
        self.assertIsNone(loop.pending);self.assertTrue(loop.poisoned);self.assertEqual(loop.cursor,0)
        self.assertEqual(loop.attempts[-1]['successful_generation_forward_calls'],1)
        self.assertEqual(len(loop.attempts[-1]['sampled_steps']),1)
        with self.assertRaises(ValueError):loop.snapshot_state()
        OBSERVATIONS.append(dict(control='mid-generation-failure-not-resumable',attempt=loop.attempts[-1]))

    def test_interrupted_optimizer_requires_durable_pending_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            loop,contract=make_loop();loop.collect();path=Path(directory)/'pending.pt'
            header=loop.save(path,contract=contract,parent_invocation='before-partial-update',max_bytes=LIMIT)
            def partial_step():
                with torch.no_grad():next(loop.model.parameters()).add_(.1)
                raise RuntimeError('injected partial optimizer')
            with patch.object(loop.optimizer,'step',side_effect=partial_step),self.assertRaises(RuntimeError):loop.apply_pending()
            self.assertTrue(loop.poisoned)
            with self.assertRaises(ValueError):loop.snapshot_state()
            restore(loop,path,contract,header)
            with patch.object(loop,'collect',side_effect=AssertionError('No resample')):row=loop.apply_pending()
            clean,_=make_loop();clean.collect();expected=clean.apply_pending()
            self.assertEqual(digest(row),digest(expected));self.assertEqual(digest(loop.snapshot_state()),digest(clean.snapshot_state()))

    def test_lifecycle_failures_keep_last_durable_boundary(self):
        for stage in ('baseline','metric','final','export'):
            with self.subTest(stage=stage),tempfile.TemporaryDirectory() as directory:
                loop,contract=make_loop();receipts=[]
                def fail(name):
                    if stage==name:raise RuntimeError('injected '+name)
                with self.assertRaises(RuntimeError):
                    runner.run_loop_with_recovery(loop,contract=contract,output=directory,parent_invocation='failure',
                        max_bytes=LIMIT,on_commit=receipts.append,baseline=lambda:fail('baseline'),
                        metric_sink=lambda row:fail('metric'),final=lambda:fail('final'),export=lambda:fail('export'))
                receipt=receipts[-1];expected=0 if stage=='baseline' else 1 if stage=='metric' else 4
                self.assertEqual(receipt['completed_updates'],expected)
                resumed,_=make_loop();restore(resumed,Path(receipt['path']),contract,receipt)
                self.assertEqual(resumed.completed,expected);self.assertEqual(len(resumed.history),expected)
                OBSERVATIONS.append(dict(control='lifecycle-failure',stage=stage,last_durable_update=expected,receipt=receipt))

    def test_new_save_failure_and_existing_output_preserve_old_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            loop,contract=make_loop();path=Path(directory)/'initial.pt'
            receipt=loop.save(path,contract=contract,parent_invocation='initial',max_bytes=LIMIT);before=path.read_bytes()
            loop.collect()
            with patch.object(runner,'save_snapshot',side_effect=OSError('injected save')),self.assertRaises(OSError):
                loop.save(Path(directory)/'new.pt',contract=contract,parent_invocation='new',max_bytes=LIMIT)
            self.assertEqual(path.read_bytes(),before)
            with self.assertRaises(FileExistsError):loop.save(path,contract=contract,parent_invocation='duplicate',max_bytes=LIMIT)
            restore(loop,path,contract,receipt);self.assertEqual(loop.completed,0)
            (Path(directory)/'snapshots').mkdir()
            with self.assertRaises(FileExistsError):runner.run_loop_with_recovery(loop,contract=contract,output=directory,parent_invocation='occupied',max_bytes=LIMIT)

    def test_config_path_is_operational_but_dropout_is_scientific(self):
        loop,_=make_loop();model=deepcopy(loop.model)
        before=runner.effective_config(model);model.config._name_or_path='/moved/local/parent'
        self.assertEqual(before,runner.effective_config(model))
        model.config.attention_dropout=.1;self.assertNotEqual(before,runner.effective_config(model))

    def test_tied_parameter_order_and_rng_layout_rejected(self):
        loop,_=make_loop();state=deepcopy(loop.snapshot_state());state['rng']['torch']=torch.zeros(3,dtype=torch.uint8)
        with self.assertRaises(ValueError):loop.validate_payload(dict(state=state,phase='completed',completed_updates=0))
        reversed_optimizer=torch.optim.AdamW(list(reversed(list(loop.model.parameters()))),lr=.008,weight_decay=0.)
        other=runner.RLVRLoop(loop.model,loop.reference,reversed_optimizer,torch.Generator().manual_seed(2323),TRAIN,decode,
            eos_id=0,stop_ids=[0,7],group_size=3,max_new_tokens=4,updates=4,seed=2323,context_length=32)
        self.assertNotEqual(loop.loop_contract['optimizer_names'],other.loop_contract['optimizer_names'])
        with self.assertRaises(ValueError):other.validate_payload(dict(state=loop.snapshot_state(),phase='completed',completed_updates=0))

    def test_evaluation_retains_raw_stop_and_known_partial_failure_work(self):
        from types import SimpleNamespace
        class AuthoredGreedy(torch.nn.Module):
            def forward(self,input_ids,use_cache=False):
                logits=torch.zeros((*input_ids.shape,16))
                logits[:,-1,8 if input_ids.shape[1]==2 else 0]=10
                return SimpleNamespace(logits=logits)
        model=AuthoredGreedy();seen=[]
        result=runner.evaluate_model(model,[(0,0)],lambda _:torch.tensor([[1,2]]),decode,[0,7],4,32,on_row=seen.append)
        row=result['rows'][0]
        self.assertEqual(row['tokens'],[8]);self.assertEqual(row['raw_tokens'],[8,0])
        self.assertEqual(row['final_stop_id'],0);self.assertFalse(row['truncated'])
        self.assertEqual(row['successful_forward_positions'],5)
        calls=0
        def fail_guard():
            nonlocal calls
            calls+=1
            if calls==2:raise RuntimeError('injected evaluation interruption')
        partial=[]
        with self.assertRaises(RuntimeError):runner.evaluate_model(model,[(0,0)],lambda _:torch.tensor([[1,2]]),
            decode,[0,7],4,32,before_stage=fail_guard,on_row=partial.append)
        self.assertEqual(len(partial),1);self.assertEqual(partial[0]['raw_tokens'],[8])
        self.assertEqual(partial[0]['successful_forward_positions'],2);self.assertEqual(partial[0]['stop'],'error')
        OBSERVATIONS.append(dict(control='authored-evaluation-stop-and-partial-failure',complete=row,partial=partial))

    def test_final_callback_binding_and_durable_status(self):
        # The static check binds the actual CLI's final observer; the following
        # journal check verifies persisted content and a restored pending count.
        tree=ast.parse((ROOT/'src/dongxi_llms/qwen_rlvr_lab.py').read_text())
        execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execute_run')
        final=next(n for n in execute.body if isinstance(n,ast.FunctionDef) and n.name=='final')
        calls=[n for n in ast.walk(final) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
               and n.func.id=='evaluate_model']
        self.assertEqual(len(calls),1)
        callback=next(k.value for k in calls[0].keywords if k.arg=='on_row')
        self.assertEqual(ast.dump(callback),ast.dump(ast.Attribute(value=ast.Name(id='journal',ctx=ast.Load()),
                                                                 attr='final_row',ctx=ast.Load())))
        with tempfile.TemporaryDirectory() as directory:
            journal=runner.RunEvidence(Path(directory)/'new-invocation',{})
            receipt=dict(phase='pending',completed_updates=2,payload_sha256='a'*64,payload_bytes=12)
            journal.stage('durable-pending',latest_durable_snapshot=receipt,restored_completed=2)
            row=dict(raw_tokens=[8],stop='error',error={'type':'RuntimeError','message':'authored final interruption'},
                     successful_forward_calls=1,successful_forward_positions=2)
            journal.final_row(row);journal.fail(RuntimeError('authored final interruption'))
            saved=json.loads((journal.output/'report.json').read_text())
            status=json.loads((journal.output/'status.json').read_text())
            self.assertEqual(saved['final_heldout_partial'],[row])
            self.assertEqual(saved['latest_durable_snapshot'],receipt)
            self.assertEqual(saved['records'],[])
            self.assertEqual(status['completed_updates'],2)
            self.assertEqual(status['invocation_completed_records'],0)
            self.assertEqual(status['durable_phase'],'pending')
            OBSERVATIONS.append(dict(control='journal-final-partial-and-durable-pending-count',status=status,
                                     latest_durable_snapshot=receipt,final_heldout_partial=saved['final_heldout_partial']))

    def test_score_guards_retain_only_successful_attempted_work(self):
        loop,_=make_loop()
        def fail_old_score():
            if loop.attempts[-1].get('stage')=='old-policy-score':
                raise RuntimeError('authored pre-old-score guard refusal')
        loop.guard=fail_old_score
        with self.assertRaisesRegex(RuntimeError,'pre-old-score'):loop.collect()
        attempt=loop.attempts[-1]
        self.assertGreater(attempt['successful_generation_forward_calls'],0)
        self.assertEqual(attempt['successful_old_policy_forward_calls'],0)
        self.assertEqual(attempt['successful_old_policy_forward_positions'],0)
        self.assertIsNone(loop.pending);self.assertTrue(loop.poisoned)
        old_attempt=deepcopy(attempt)
        loop,_=make_loop();loop.collect()
        collection=deepcopy(loop.attempts[-1])
        self.assertEqual(collection['successful_old_policy_forward_calls'],1)
        self.assertEqual(collection['successful_old_policy_forward_positions'],loop.pending['work']['old_policy_forward_positions'])
        before_policy=digest(loop.model.state_dict());before_optimizer=digest(loop.optimizer.state_dict())
        def fail_reference():
            if loop.attempts[-1].get('kind')=='application' and loop.attempts[-1].get('stage')=='reference-score':
                raise RuntimeError('authored pre-reference-score guard refusal')
        loop.guard=fail_reference
        with self.assertRaisesRegex(RuntimeError,'pre-reference-score'):loop.apply_pending()
        attempt=loop.attempts[-1]
        self.assertEqual(attempt['successful_reference_forward_calls'],0)
        self.assertEqual(attempt['successful_reference_forward_positions'],0)
        self.assertEqual(attempt['successful_policy_forward_calls'],0)
        self.assertEqual(loop.application_work,{})
        self.assertEqual(before_policy,digest(loop.model.state_dict()))
        self.assertEqual(before_optimizer,digest(loop.optimizer.state_dict()))
        self.assertTrue(loop.poisoned);self.assertIsNotNone(loop.pending)
        OBSERVATIONS.append(dict(control='score-guard-refusals-not-completed-work',old_score=old_attempt,
                                 successful_collection=collection,reference_score=deepcopy(attempt)))


def child(args):
    contract=json.loads(args.contract.read_text());loop,current=make_loop(contract['seed'])
    if contract!=current:raise ValueError('Fresh-process actual source/fixture identity changed')
    payload=restore(loop,args.child,contract,dict(payload_sha256=args.sha256,payload_bytes=args.bytes))
    tail=[]
    if payload['phase']=='pending':
        with patch.object(loop,'collect',side_effect=AssertionError('Fresh pending may not resample')):tail.append(loop.apply_pending())
    while loop.completed<4:tail.append(loop.apply_pending())
    print(json.dumps(dict(phase=payload['phase'],tail_sha256=digest(tail),state_sha256=digest(loop.snapshot_state()),
        no_collection_first_pending=payload['phase']=='pending',history=runner.jsonable(loop.history))))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--child',type=Path);parser.add_argument('--contract',type=Path)
    parser.add_argument('--sha256');parser.add_argument('--bytes',type=int)
    parser.add_argument('--verify-tests',type=Path);parser.add_argument('--observations-output',type=Path)
    args=parser.parse_args()
    if args.child:return child(args)
    if args.verify_tests:
        if args.verify_tests.exists():raise FileExistsError(args.verify_tests)
        raw=args.verify_tests.with_suffix('.observations.json')
        command=[sys.executable,str(Path(__file__).resolve()),'--observations-output',str(raw)]
        paths=['src/dongxi_llms/qwen_rlvr_lab.py','tests/test_rlvr_runner_recovery.py',
            'experiments/specs/2026-10-05-rlvr-runner-recovery.md','src/dongxi_llms/training_snapshot.py',
            'experiments/reports/2026-10-05-rlvr-runner-recovery-original.py.txt','uv.lock']
        hashes=lambda:{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in paths}
        before=hashes();start=time.perf_counter()
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=120)
        evidence=dict(command=command,actual_exit_code=result.returncode,seconds=time.perf_counter()-start,
            stdout=result.stdout,stderr=result.stderr,source_before=before,source_after=hashes(),
            environment={'python':sys.version,'executable':sys.executable,'torch':str(torch.__version__),
                         'platform':platform.platform(),'cuda_available':torch.cuda.is_available()},
            observations=str(raw),observations_sha256=hashlib.sha256(raw.read_bytes()).hexdigest() if raw.exists() else None)
        with args.verify_tests.open('x') as handle:json.dump(evidence,handle,indent=2);handle.write('\n')
        print(json.dumps(dict(evidence=str(args.verify_tests),exit=result.returncode)));raise SystemExit(result.returncode)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(RLVRRecoveryTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if args.observations_output:
        with args.observations_output.open('x') as handle:json.dump(runner.jsonable(OBSERVATIONS),handle,indent=2);handle.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__=='__main__':main()
