"""Actual runner-loop recovery using original tiny, randomly initialized HF models."""
import argparse
import copy
import hashlib
import importlib.util
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
from dongxi_llms.run_identity import canonical_hash,environment_identity,tokenizer_interface
from dongxi_llms.training_snapshot import inspect_snapshot,save_snapshot

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('sft_recovery_runner',ROOT/'scripts/run_chapter09_spark_sft.py')
runner=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(runner)
LIMIT=16*1024**2
TEMPLATE="{% for m in messages %}{{ '<|im_start|>' + m.role + '\n' + m.content + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"


def tokenizer():
    from tokenizers import Tokenizer,models,pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    words=['[UNK]','[BOS]','<|im_start|>','<|im_end|>','system','user','assistant',
           'copy','red','blue','one','two','three','reverse','keep','now']
    backend=Tokenizer(models.WordLevel({word:i for i,word in enumerate(words)},unk_token='[UNK]'))
    backend.pre_tokenizer=pre_tokenizers.WhitespaceSplit()
    value=PreTrainedTokenizerFast(tokenizer_object=backend,unk_token='[UNK]',bos_token='[BOS]',
        eos_token='<|im_end|>',pad_token='<|im_end|>',additional_special_tokens=['<|im_start|>'])
    value.chat_template=TEMPLATE
    return value


def records():
    return [dict(id='original-0',messages=[dict(role='system',content='keep'),
        dict(role='user',content='copy red'),dict(role='assistant',content='red')]),
        dict(id='original-1',messages=[dict(role='user',content='reverse one two'),
        dict(role='assistant',content='two one')]),
        dict(id='original-2',messages=[dict(role='user',content='copy blue'),
        dict(role='assistant',content='blue'),dict(role='user',content='copy red'),
        dict(role='assistant',content='red blue')])]


def make_loop(seed=1212,mode='full'):
    from transformers import Qwen3Config,Qwen3ForCausalLM
    random.seed(seed); torch.manual_seed(seed)
    config=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=2,head_dim=8,max_position_embeddings=64,
        attention_dropout=.1,bos_token_id=1,eos_token_id=3,pad_token_id=3,tie_word_embeddings=True)
    config._attn_implementation='sdpa'
    model=Qwen3ForCausalLM(config)
    model.config.use_cache=False; model.gradient_checkpointing_enable()
    if mode=='lora':
        from peft import LoraConfig,get_peft_model
        model=get_peft_model(model,LoraConfig(r=2,lora_alpha=2,lora_dropout=0.,
            target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
        model.enable_input_require_grads()
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=.003,weight_decay=0.)
    tok=tokenizer()
    encoded=[runner.encode_record(tok,row,64) for row in records()]
    loop=runner.SFTLoop(model,optimizer,encoded,tok.pad_token_id,microbatch=1,accumulation=2,
                        seed=seed,device='cpu',total_updates=4)
    interface=tokenizer_interface(tok,template=TEMPLATE,stop_ids=[3],
                                  tokenizer_id='authored-sft-recovery',tokenizer_revision='1'*40)
    cfg=dict(model='locally-constructed-random-Qwen3',revision='1'*40,tokenizer_revision='1'*40,
        updates=4,microbatch=1,accumulation=2,max_length=64,learning_rate=.003,seed=seed,mode=mode,rank=2,
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ['scripts/run_chapter09_spark_sft.py','src/dongxi_llms/training_snapshot.py',
             'src/dongxi_llms/run_identity.py']},
        environment_lock_sha256=hashlib.sha256((ROOT/'uv.lock').read_bytes()).hexdigest(),
        train_sha256=canonical_hash(records()),dev_sha256='authored-independent-unused-dev-control',
        template_sha256=hashlib.sha256(TEMPLATE.encode()).hexdigest(),checkpoint_interface=interface,
        base_checkpoint_files={'random-initial-policy':state_digest(model.state_dict())},
        dtype='CPU FP32; FP32 cross entropy',attention_backend='sdpa',torch_version=str(torch.__version__),
        numerical_environment=runner.observed_numerical_environment(environment_identity()),
        loss_policy='assistant body/end; one explicit shift',generation_stop_ids=[3])
    return loop,runner.stable_sft_contract(cfg)


def state_digest(value):
    h=hashlib.sha256()
    def frame(tag,data):h.update(tag+len(data).to_bytes(8,'big')+data)
    def visit(item):
        if isinstance(item,torch.Tensor):
            frame(b'tensor',str((str(item.dtype),tuple(item.shape))).encode())
            frame(b'bytes',item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item,dict):
            frame(b'dict',str(len(item)).encode())
            for key in sorted(item,key=lambda key:(type(key).__name__,repr(key))):visit(key);visit(item[key])
        elif isinstance(item,(list,tuple)):
            frame(type(item).__name__.encode(),str(len(item)).encode())
            for child in item:visit(child)
        else:frame(type(item).__name__.encode(),repr(item).encode())
    visit(value);return h.hexdigest()


def restore(loop,path,contract,header):
    return loop.restore(path,contract=contract,expected_sha256=header['payload_sha256'],
                        expected_bytes=header['payload_bytes'],max_bytes=LIMIT)


def original_update(loop):
    """Original SFT numerical equations, independent of the extracted method."""
    loop.model.train(); window=[]
    for _ in range(loop.accumulation):
        rows=[loop.train[loop.order[(loop.cursor+i)%len(loop.order)]] for i in range(loop.microbatch)]
        loop.cursor+=loop.microbatch;window.append(runner.collate(rows,loop.pad_id,'cpu'))
    count=sum(int((batch[1][:,1:]!=-100).sum()) for batch in window)
    loop.optimizer.zero_grad(set_to_none=True);total=0.
    for batch in window:
        loss=runner.summed_loss(loop.model,batch);(loss/count).backward();total+=float(loss.detach())
    norm=torch.nn.utils.clip_grad_norm_(loop.parameters,1.);loop.optimizer.step()
    loop.update+=1;loop.targets+=count;positions=sum(batch[0].numel() for batch in window);loop.positions+=positions
    loop.last_metric=dict(update=loop.update,answer_nll=total/count,targets=count,gradient_norm=float(norm),
        cursor=loop.cursor,cumulative_supervised_targets=loop.targets,processed_positions=positions,
        cumulative_processed_positions=loop.positions)
    loop.history.append(copy.deepcopy(loop.last_metric))
    return copy.deepcopy(loop.last_metric)


class SFTRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_actual_loop_all_seeds_modes_completed_replay(self):
        with tempfile.TemporaryDirectory() as root:
            for seed in (1212,1213):
                for mode in ('full','lora'):
                    with self.subTest(seed=seed,mode=mode):
                        actual,contract=make_loop(seed,mode)
                        first=[actual.completed_update() for _ in range(2)]
                        path=Path(root)/f'{seed}-{mode}.pt'
                        header=actual.save(path,contract=contract,parent_invocation='original',max_bytes=LIMIT)
                        expected=[actual.completed_update() for _ in range(2)]
                        expected_state=state_digest(actual.snapshot_state())
                        resumed,resume_contract=make_loop(seed,mode)
                        self.assertEqual(contract,resume_contract)
                        payload=restore(resumed,path,contract,header)
                        self.assertEqual(payload['parent_invocation'],'original')
                        self.assertEqual(payload['state']['metric'],first[-1])
                        self.assertEqual(expected,[resumed.completed_update() for _ in range(2)])
                        self.assertEqual(expected_state,state_digest(resumed.snapshot_state()))

    def test_extracted_update_matches_original_equations_with_rng(self):
        actual,contract=make_loop()
        initial_rng=copy.deepcopy(actual.snapshot_state()['rng'])
        row=actual.completed_update();expected=state_digest(actual.snapshot_state())
        original,_=make_loop()
        random.setstate(initial_rng['python']);torch.set_rng_state(initial_rng['torch'])
        self.assertEqual(row,original_update(original))
        self.assertEqual(expected,state_digest(original.snapshot_state()))

    def test_masks_single_shift_and_eos_alias_padding(self):
        loop,_=make_loop()
        self.assertEqual([sum(x!=-100 for x in row['labels'][1:]) for row in loop.train],[2,3,5])
        batch=runner.collate(loop.train,3,'cpu')
        self.assertTrue(bool((batch[0][batch[2]==0]==3).all()))
        self.assertTrue(bool((batch[1][batch[2]==0]==-100).all()))
        self.assertEqual(int((batch[1][:,1:]==3).sum()),4)
        for row in loop.train:self.assertEqual(row['labels'][-1],3)

    def test_contract_excludes_paths_deadlines_cadence_not_horizon(self):
        config=dict(updates=4,learning_rate=.003,output='first',train='one',runtime_seconds=60,
                    checkpoint_every=1,snapshot_max_bytes=100)
        changed=dict(config,output='second',train='moved',runtime_seconds=80,checkpoint_every=2,snapshot_max_bytes=200)
        self.assertEqual(runner.stable_sft_contract(config),runner.stable_sft_contract(changed))
        self.assertNotEqual(runner.stable_sft_contract(config),runner.stable_sft_contract(dict(config,updates=5)))

    def test_preload_byte_and_scientific_contract_refusals(self):
        with tempfile.TemporaryDirectory() as root:
            loop,contract=make_loop();loop.completed_update();path=Path(root)/'state.pt'
            header=loop.save(path,contract=contract,parent_invocation='parent',max_bytes=LIMIT)
            variants=[]
            for key in ('source_sha256','environment_lock_sha256','train_sha256','base_checkpoint_files',
                        'checkpoint_interface','template_sha256','generation_stop_ids','updates'):
                altered=copy.deepcopy(contract);altered['config'][key]='changed';variants.append(altered)
            with patch.object(runner.torch,'load',side_effect=AssertionError('deserialization forbidden')):
                for changed in variants:
                    with self.subTest(changed=changed),self.assertRaises(ValueError):
                        inspect_snapshot(path,expected_sha256=header['payload_sha256'],expected_bytes=header['payload_bytes'],
                            expected_contract=changed,max_bytes=LIMIT)
                with self.assertRaises(ValueError):
                    inspect_snapshot(path,expected_sha256='0'*64,expected_bytes=header['payload_bytes'],
                                     expected_contract=contract,max_bytes=LIMIT)
                with self.assertRaises(ValueError):
                    inspect_snapshot(path,expected_sha256=header['payload_sha256'],expected_bytes=header['payload_bytes']+1,
                                     expected_contract=contract,max_bytes=LIMIT)
                with path.open('ab') as handle:handle.write(b'corruption')
                with self.assertRaises(ValueError):
                    inspect_snapshot(path,expected_sha256=header['payload_sha256'],expected_bytes=header['payload_bytes'],
                                     expected_contract=contract,max_bytes=LIMIT)

    def test_observed_environment_changes_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as root:
            loop,contract=make_loop();path=Path(root)/'environment.pt'
            header=loop.save(path,contract=contract,parent_invocation='environment-bound',max_bytes=LIMIT)
            environment=contract['config']['numerical_environment']
            self.assertEqual(set(environment),{'python_version','platform','machine','packages'})
            self.assertNotIn('interpreter',environment);self.assertNotIn('environment_lock',environment)
            changes=[]
            for key in ('python_version','platform','machine'):
                changed=copy.deepcopy(contract);changed['config']['numerical_environment'][key]='different'
                changes.append((key,changed))
            for key in ('torch','transformers','tokenizers','peft'):
                changed=copy.deepcopy(contract)
                changed['config']['numerical_environment']['packages'][key]='different-version'
                changes.append((key,changed))
            with patch.object(runner.torch,'load',side_effect=AssertionError('deserialization forbidden')):
                for name,changed in changes:
                    with self.subTest(environment=name),self.assertRaises(ValueError):
                        inspect_snapshot(path,expected_sha256=header['payload_sha256'],
                            expected_bytes=header['payload_bytes'],expected_contract=changed,max_bytes=LIMIT)

    def test_counter_rng_order_layout_and_metric_rejected_before_apply(self):
        with tempfile.TemporaryDirectory() as root:
            loop,contract=make_loop();loop.completed_update();original=copy.deepcopy(loop.snapshot_state())
            variants=[]
            for key in ('cursor','targets','positions'):
                state=copy.deepcopy(original);state[key]+=1;variants.append((key,state))
            state=copy.deepcopy(original);state['order']=list(reversed(state['order']));variants.append(('order',state))
            state=copy.deepcopy(original);state['rng']['torch']=torch.ones(2,dtype=torch.uint8);variants.append(('rng',state))
            state=copy.deepcopy(original);state['metric']['targets']+=1;variants.append(('metric',state))
            state=copy.deepcopy(original);key=next(iter(state['model']));state['model'][key]=torch.zeros(1);variants.append(('policy',state))
            state=copy.deepcopy(original);key=next(iter(state['optimizer']['state']));state['optimizer']['state'][key]['exp_avg']=torch.zeros(1);variants.append(('optimizer',state))
            state=copy.deepcopy(original);state['history']=[];variants.append(('missing-history',state))
            state=copy.deepcopy(original);state['history'][0]['update']=2;variants.append(('history-sequence',state))
            for name,step in (('bool-step',torch.tensor(True)),('rank1-step',torch.tensor([1.]))):
                state=copy.deepcopy(original);key=next(iter(state['optimizer']['state']))
                state['optimizer']['state'][key]['step']=step;variants.append((name,state))
            state=copy.deepcopy(original);key=next(iter(state['optimizer']['state']))
            state['optimizer']['state'][key]['exp_avg_sq'].fill_(-1);variants.append(('negative-second-moment',state))
            for name,value in (('cuda-none',None),('cuda-bool',False),('cuda-dict',{})):
                state=copy.deepcopy(original);state['rng']['cuda']=value;variants.append((name,state))
            state=copy.deepcopy(original);state['order']=[bool(x) if x in (0,1) else x for x in state['order']]
            variants.append(('bool-order',state))
            state=copy.deepcopy(original);state['optimizer']['param_groups'][0]['params'][0]=False
            variants.append(('bool-optimizer-param-id',state))
            bad_keys=copy.deepcopy(original)
            bad_keys['optimizer']['state']={bool(k) if k in (0,1) else k:v
                                           for k,v in bad_keys['optimizer']['state'].items()}
            with self.assertRaisesRegex(ValueError,'coverage'):
                loop.validate_payload(dict(phase='completed',completed_updates=1,state=bad_keys,contract=contract))
            # Shared data-only serialization rejects bool mapping keys even
            # before a durable malformed payload can be constructed.
            with self.assertRaisesRegex(ValueError,'mapping key'):
                save_snapshot(Path(root)/'bool-optimizer-state-id.pt',contract=contract,state=bad_keys,
                    completed_updates=1,parent_invocation='refused-before-save',max_bytes=LIMIT)
            for name,state in variants:
                with self.subTest(name=name):
                    path=Path(root)/(name+'.pt')
                    header=save_snapshot(path,contract=contract,state=state,completed_updates=1,
                                         parent_invocation='deliberate-malformed-control',max_bytes=LIMIT)
                    restored,_=make_loop()
                    before=state_digest(restored.snapshot_state())
                    with patch.object(restored.model,'load_state_dict',side_effect=AssertionError('apply forbidden')), \
                            self.assertRaises(ValueError):restore(restored,path,contract,header)
                    self.assertEqual(before,state_digest(restored.snapshot_state()))

    def test_interrupted_backward_is_poisoned_and_durable_replay_recovers(self):
        with tempfile.TemporaryDirectory() as root:
            loop,contract=make_loop();path=Path(root)/'initial.pt'
            header=loop.save(path,contract=contract,parent_invocation='before-failure',max_bytes=LIMIT)
            actual=runner.summed_loss;calls=0
            def interrupt(model,batch):
                nonlocal calls
                calls+=1
                if calls==2:raise KeyboardInterrupt('injected after first backward')
                return actual(model,batch)
            with patch.object(runner,'summed_loss',side_effect=interrupt),self.assertRaises(KeyboardInterrupt):
                loop.completed_update()
            self.assertEqual(loop.update,0);self.assertTrue(loop.incomplete_update)
            with self.assertRaisesRegex(RuntimeError,'interrupted'):loop.snapshot_state()
            with self.assertRaisesRegex(RuntimeError,'requires restore'):loop.completed_update()
            restore(loop,path,contract,header);row=loop.completed_update();state=state_digest(loop.snapshot_state())
            clean,clean_contract=make_loop();self.assertEqual(contract,clean_contract)
            self.assertEqual(row,clean.completed_update());self.assertEqual(state,state_digest(clean.snapshot_state()))

    def test_lifecycle_failures_keep_committed_snapshot_and_metric(self):
        for failure in ('baseline','metric','final','export'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as root:
                loop,contract=make_loop();seen=[]
                def fail(name,value=None):
                    if name==failure:raise RuntimeError('injected '+name+' failure')
                    return value
                def metric(row):seen.append(row);fail('metric')
                with self.assertRaisesRegex(RuntimeError,'injected'):
                    runner.run_loop_with_recovery(loop,contract=contract,output=root,parent_invocation='lifecycle',
                        max_bytes=LIMIT,checkpoint_every=1,metric_sink=metric,
                        baseline_observer=lambda:fail('baseline',{}),final_observer=lambda:fail('final',{}),
                        exporter=lambda:fail('export'))
                pointer=json.loads((Path(root)/'latest-completed-snapshot.json').read_text())
                expected_update=0 if failure=='baseline' else 1 if failure=='metric' else 4
                self.assertEqual(pointer['header']['completed_updates'],expected_update)
                recovered,_=make_loop();payload=restore(recovered,Path(pointer['path']),contract,pointer['header'])
                self.assertEqual(recovered.update,expected_update)
                if expected_update:self.assertEqual(payload['state']['metric']['update'],expected_update)

    def test_failed_new_snapshot_preserves_previous_committed_state(self):
        with tempfile.TemporaryDirectory() as root:
            loop,contract=make_loop();first=Path(root)/'first.pt'
            header=loop.save(first,contract=contract,parent_invocation='first',max_bytes=LIMIT)
            previous=first.read_bytes();loop.completed_update();second=Path(root)/'second.pt'
            with patch.object(runner,'save_snapshot',side_effect=OSError('injected save failure')),self.assertRaises(OSError):
                loop.save(second,contract=contract,parent_invocation='second',max_bytes=LIMIT)
            self.assertEqual(first.read_bytes(),previous)
            restored,_=make_loop();restore(restored,first,contract,header);self.assertEqual(restored.update,0)
            with self.assertRaises(FileExistsError):loop.save(first,contract=contract,parent_invocation='duplicate',max_bytes=LIMIT)

    def test_missing_byte_receipt_refused_and_existing_output_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            limits=root/'work-limits.json'
            runner.write_json(limits,{key:10000 for key in runner.WORK_KEYS})
            args=['--revision','1'*40,'--tokenizer-revision','1'*40,'--train',str(root/'train'),
                  '--dev',str(root/'dev'),'--template',str(root/'template'),'--environment-lock',str(root/'lock'),
                  '--output',str(root/'output'),'--resume',str(root/'resume.pt'),
                  '--work-limits',str(limits),'--work-journal-max-bytes',str(1024**2)]
            from snapshot_io_test_support import explicit_snapshot_io_args
            args += explicit_snapshot_io_args(root)
            with self.assertRaisesRegex(ValueError,'independently expected'):runner.main(args)
            self.assertFalse((root/'output').exists())
            (root/'output').mkdir();(root/'output'/'preserved').write_text('original')
            with self.assertRaises(FileExistsError):runner.main(args)
            self.assertEqual((root/'output'/'preserved').read_text(),'original')
            for value in ('0',str(2**63)):
                with self.subTest(snapshot_limit=value), \
                        patch.object(runner,'collect_run_identity',side_effect=AssertionError('pre-allocation gate')), \
                        self.assertRaisesRegex(ValueError,'snapshot byte bound'):
                    runner.main(args+['--snapshot-max-bytes',value])

    def test_horizon_cannot_accept_extra_update(self):
        loop,_=make_loop()
        for _ in range(4):loop.completed_update()
        before=state_digest(loop.snapshot_state())
        with self.assertRaisesRegex(ValueError,'horizon'):loop.completed_update()
        self.assertEqual(before,state_digest(loop.snapshot_state()))

    def test_optimizer_names_and_real_objects_are_bound(self):
        loop,_=make_loop()
        swapped=torch.optim.AdamW(list(reversed(loop.parameters)),lr=.003,weight_decay=0.)
        other=runner.SFTLoop(loop.model,swapped,loop.train,loop.pad_id,microbatch=1,accumulation=2,
                             seed=loop.seed,device='cpu',total_updates=4)
        self.assertNotEqual(loop.loop_contract['optimizer_parameter_names'],
                            other.loop_contract['optimizer_parameter_names'])
        omitted=torch.optim.AdamW(loop.parameters[:-1],lr=.003,weight_decay=0.)
        with self.assertRaisesRegex(ValueError,'exactly once'):
            runner.SFTLoop(loop.model,omitted,loop.train,loop.pad_id,microbatch=1,accumulation=2,
                           seed=loop.seed,device='cpu',total_updates=4)

    def test_checkpoint_interval_two_carries_full_committed_history(self):
        with tempfile.TemporaryDirectory() as root:
            loop,contract=make_loop()
            def metric(row):
                if row['update']==2:raise RuntimeError('injected after commit before metric append')
            with self.assertRaises(RuntimeError):
                runner.run_loop_with_recovery(loop,contract=contract,output=root,parent_invocation='history',
                    max_bytes=LIMIT,checkpoint_every=2,metric_sink=metric,baseline_observer=lambda:None,
                    final_observer=lambda:None,exporter=lambda:None)
            pointer=json.loads((Path(root)/'latest-completed-snapshot.json').read_text())
            resumed,_=make_loop();payload=restore(resumed,Path(pointer['path']),contract,pointer['header'])
            self.assertEqual([row['update'] for row in payload['state']['history']],[1,2])
            self.assertEqual(payload['state']['history'][-1],payload['state']['metric'])

    def test_ragged_batch_work_prefix_matches_actual_collations(self):
        original,_=make_loop()
        loop=runner.SFTLoop(original.model,original.optimizer,original.train,original.pad_id,
                           microbatch=2,accumulation=2,seed=1212,device='cpu',total_updates=4)
        for _ in range(4):
            loop.completed_update()
            self.assertEqual(loop._expected_work(loop.update),(loop.cursor,loop.targets,loop.positions))

    def test_mocked_cuda_rng_rejected_before_any_state_application(self):
        loop,contract=make_loop();state=copy.deepcopy(loop.snapshot_state())
        # Metadata-only CUDA validation: no CUDA kernel/context/model is used.
        loop.device=torch.device('cuda')
        current=torch.zeros(16,dtype=torch.uint8)
        loop._validation_cuda_rng_shapes=[current.shape]
        real_generator=torch.Generator
        class MockCudaGenerator:
            def set_state(self,value):
                if not torch.equal(value,current):raise RuntimeError('invalid mocked CUDA RNG bytes')
        def generator(*,device):
            return real_generator(device='cpu') if device=='cpu' else MockCudaGenerator()
        candidates=[None,False,{},[],[torch.zeros(3)],[torch.zeros(3,dtype=torch.uint8)],
                    [torch.ones(16,dtype=torch.uint8)]]
        with patch.object(runner.torch.cuda,'get_rng_state_all',return_value=[current]), \
                patch.object(runner.torch,'Generator',side_effect=generator), \
                patch.object(loop.model,'load_state_dict',side_effect=AssertionError('apply forbidden')):
            for candidate in candidates:
                with self.subTest(candidate=candidate),self.assertRaises(ValueError):
                    malformed=copy.deepcopy(state);malformed['rng']['cuda']=candidate
                    loop.validate_payload(dict(phase='completed',completed_updates=0,state=malformed,contract=contract))
            valid=copy.deepcopy(state);valid['rng']['cuda']=[current]
            loop.validate_payload(dict(phase='completed',completed_updates=0,state=valid,contract=contract))

    def test_model_configuration_ignores_only_operational_parent_location(self):
        loop,_=make_loop();expected=copy.deepcopy(loop.loop_contract)
        loop.model.config._name_or_path='/first/local/cache'
        first=runner.SFTLoop(loop.model,loop.optimizer,loop.train,loop.pad_id,microbatch=1,accumulation=2,
                             seed=1212,device='cpu',total_updates=4)
        loop.model.config._name_or_path='/relocated/local/cache'
        second=runner.SFTLoop(loop.model,loop.optimizer,loop.train,loop.pad_id,microbatch=1,accumulation=2,
                              seed=1212,device='cpu',total_updates=4)
        self.assertEqual(expected,first.loop_contract);self.assertEqual(first.loop_contract,second.loop_contract)
        loop.model.config.attention_dropout=.2
        changed=runner.SFTLoop(loop.model,loop.optimizer,loop.train,loop.pad_id,microbatch=1,accumulation=2,
                               seed=1212,device='cpu',total_updates=4)
        self.assertNotEqual(expected['model_config_sha256'],changed.loop_contract['model_config_sha256'])

    def test_fresh_process_restores_actual_runner_loop(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);loop,contract=make_loop()
            for _ in range(2):loop.completed_update()
            path=root/'boundary.pt';header=loop.save(path,contract=contract,parent_invocation='fresh-parent',max_bytes=LIMIT)
            contract_path=root/'contract.json';contract_path.write_text(json.dumps(contract))
            expected=[loop.completed_update() for _ in range(2)];expected_state=state_digest(loop.snapshot_state())
            command=[sys.executable,str(Path(__file__).resolve()),'--child-replay',str(path),'--contract',str(contract_path),
                     '--sha256',header['payload_sha256'],'--bytes',str(header['payload_bytes'])]
            child=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,timeout=60)
            self.assertEqual(child.returncode,0,child.stderr)
            result=json.loads(child.stdout.strip().splitlines()[-1])
            self.assertEqual(result['metrics'],expected);self.assertEqual(result['final_state_sha256'],expected_state)


def child_replay(args):
    torch.set_num_threads(1)
    contract=json.loads(args.contract.read_text());cfg=contract['config']
    loop,actual=make_loop(cfg['seed'],cfg['mode'])
    if actual!=contract:raise ValueError('Fresh-process actual source/fixture identity changed')
    restore(loop,args.child_replay,contract,dict(payload_sha256=args.sha256,payload_bytes=args.bytes))
    metrics=[]
    while loop.update<loop.total_updates:metrics.append(loop.completed_update())
    print(json.dumps(dict(metrics=metrics,final_state_sha256=state_digest(loop.snapshot_state()))))


def collect_reference(path):
    """Fixed four-arm numerical evidence; every arm is reported, never selected."""
    if path.exists():raise FileExistsError(path)
    torch.set_num_threads(1)
    names=['scripts/run_chapter09_spark_sft.py','src/dongxi_llms/training_snapshot.py',
           'src/dongxi_llms/run_identity.py',
           'tests/test_sft_runner_recovery.py','tests/test_spark_sft_contract.py',
           'experiments/specs/2026-10-05-sft-runner-recovery.md','uv.lock']
    def hashes():return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
    before=hashes();start=time.monotonic()
    evidence_root=Path(tempfile.mkdtemp(prefix='dongxi-sft-recovery-reference.'))
    arms=[]
    for seed in (1212,1213):
        for mode in ('full','lora'):
            loop,contract=make_loop(seed,mode)
            initial_digest=state_digest(loop.snapshot_state())
            records_before=[loop.completed_update() for _ in range(2)]
            snapshot=evidence_root/f'{seed}-{mode}.pt'
            header=loop.save(snapshot,contract=contract,parent_invocation=f'fixed-cpu-{seed}-{mode}',max_bytes=LIMIT)
            records_after=[loop.completed_update() for _ in range(2)]
            expected_digest=state_digest(loop.snapshot_state())
            resumed,resume_contract=make_loop(seed,mode)
            if resume_contract!=contract:raise AssertionError('Fixed source/scientific identity drift')
            restored=restore(resumed,snapshot,contract,header)
            actual_rows=[resumed.completed_update() for _ in range(2)]
            actual_digest=state_digest(resumed.snapshot_state())
            contract_path=evidence_root/f'{seed}-{mode}-contract.json'
            with contract_path.open('x') as handle:json.dump(contract,handle,indent=2);handle.write('\n')
            command=[sys.executable,str(Path(__file__).resolve()),'--child-replay',str(snapshot),
                '--contract',str(contract_path),'--sha256',header['payload_sha256'],
                '--bytes',str(header['payload_bytes'])]
            child_start=time.monotonic()
            child=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,timeout=60)
            fresh=json.loads(child.stdout.strip().splitlines()[-1]) if child.returncode==0 else None
            checks=dict(same_process_metrics_exact=actual_rows==records_after,
                same_process_full_state_exact=actual_digest==expected_digest,
                fresh_process_exit_zero=child.returncode==0,
                fresh_process_metrics_exact=fresh is not None and fresh['metrics']==records_after,
                fresh_process_full_state_exact=fresh is not None and fresh['final_state_sha256']==expected_digest,
                committed_history_exact=restored['state']['history']==records_before,
                last_metric_exact=restored['state']['metric']==records_before[-1])
            arms.append(dict(seed=seed,mode=mode,initial_state_sha256=initial_digest,
                total_parameters=sum(p.numel() for p in loop.model.parameters()),
                trainable_parameters=sum(p.numel() for p in loop.parameters),
                snapshot=dict(path=str(snapshot),header=header,
                    contract_sha256=canonical_hash(contract),loop_contract=loop.loop_contract),
                uninterrupted_metrics=records_before+records_after,resumed_metrics=actual_rows,
                uninterrupted_final_state_sha256=expected_digest,resumed_final_state_sha256=actual_digest,
                cumulative_supervised_targets=loop.targets,cumulative_processed_positions=loop.positions,
                encoded_record_lengths=[len(row['ids']) for row in loop.train],
                shifted_target_counts=[sum(x!=-100 for x in row['labels'][1:]) for row in loop.train],
                fresh_process=dict(command=command,actual_exit_code=child.returncode,stdout=child.stdout,
                    stderr=child.stderr,seconds=time.monotonic()-child_start,result=fresh),checks=checks))
    extracted,_=make_loop();rng=copy.deepcopy(extracted.snapshot_state()['rng'])
    extracted_row=extracted.completed_update();extracted_digest=state_digest(extracted.snapshot_state())
    original,_=make_loop();random.setstate(rng['python']);torch.set_rng_state(rng['torch'])
    original_row=original_update(original);original_digest=state_digest(original.snapshot_state())
    parity=dict(extracted_metric=extracted_row,original_metric=original_row,
        extracted_state_sha256=extracted_digest,original_state_sha256=original_digest,
        metric_exact=extracted_row==original_row,full_state_exact=extracted_digest==original_digest)
    after=hashes()
    result=dict(schema_version=1,scope='actual SFT loop; locally constructed random tiny HF CPU only',
        actual_argv=sys.argv,runnable_invocation=[sys.executable,*sys.argv],
        python=sys.version,torch=str(torch.__version__),platform=platform.platform(),
        numerical_environment=runner.observed_numerical_environment(environment_identity()),
        cuda_available=torch.cuda.is_available(),torch_threads=torch.get_num_threads(),
        evidence_directory=str(evidence_root),source_sha256_before=before,source_sha256_after=after,
        sources_unchanged=before==after,seconds=time.monotonic()-start,
        fixed_recipe=dict(seeds=[1212,1213],modes=['full','lora'],updates=4,resume_after=2,
            microbatch=1,accumulation=2,learning_rate=.003,weight_decay=0.,clip_norm=1.,
            lora_rank=2,lora_alpha=2,lora_dropout=0.,snapshot_max_bytes=LIMIT,
            model=dict(vocabulary=32,width=16,intermediate_width=32,layers=1,
                attention_heads=2,kv_heads=2,head_dimension=8,context=64,attention_dropout=.1)),
        arms=arms,original_equation_parity=parity,
        observed_reference_updates=dict(unique_recipe_updates=16,
            same_process_replayed_updates=8,fresh_process_replayed_updates=8,parity_updates=2),
        limitations=['No pretrained/model-scale/CUDA/BF16/Flash replay was executed.',
            'Repeated recovery trajectories are not additional unique learning updates.',
            'CUDA malformed-state rejection is mocked metadata testing, not GPU validation.'])
    result['all_checks']=all(all(arm['checks'].values()) for arm in arms) and parity['metric_exact'] \
        and parity['full_state_exact'] and result['sources_unchanged']
    with path.open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps(dict(evidence=str(path),all_checks=result['all_checks'])))
    if not result['all_checks']:raise AssertionError('Fixed SFT reference had a failing arm; evidence retained')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child-replay',type=Path);parser.add_argument('--contract',type=Path)
    parser.add_argument('--sha256');parser.add_argument('--bytes',type=int)
    parser.add_argument('--verify-tests',type=Path,help='Non-overwriting actual focused test attempt evidence')
    parser.add_argument('--reference',type=Path,help='Fixed numerical four-arm recovery evidence; CPU fixtures only')
    args=parser.parse_args()
    if args.verify_tests:
        if args.verify_tests.exists():raise FileExistsError(args.verify_tests)
        names=['scripts/run_chapter09_spark_sft.py','src/dongxi_llms/training_snapshot.py',
               'src/dongxi_llms/run_identity.py',
               'tests/test_sft_runner_recovery.py','tests/test_spark_sft_contract.py',
               'experiments/specs/2026-10-05-sft-runner-recovery.md','uv.lock']
        def hashes():return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
        before=hashes();start=time.monotonic()
        command=[sys.executable,'-m','unittest','discover','-s','tests','-p','test_sft_runner_recovery.py','-v']
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=120)
        evidence=dict(command=command,actual_exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr,
            seconds=time.monotonic()-start,source_sha256_before=before,source_sha256_after=hashes(),
            actual_argv=sys.argv,python=sys.version,torch=str(torch.__version__),platform=platform.platform(),
            cuda_available=torch.cuda.is_available())
        with args.verify_tests.open('x') as handle:json.dump(evidence,handle,indent=2);handle.write('\n')
        print(json.dumps(dict(evidence=str(args.verify_tests),actual_exit_code=result.returncode)))
        raise SystemExit(result.returncode)
    elif args.reference:collect_reference(args.reference)
    elif args.child_replay:child_replay(args)
    else:unittest.main(argv=[sys.argv[0]],verbosity=2)


if __name__=='__main__':main()
