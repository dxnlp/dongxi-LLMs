"""Authored CPU admission/retention controls; no pretrained/GPU calls."""
from copy import deepcopy
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from dongxi_llms.reasoning_generation import generate_record
from dongxi_llms.run_identity import artifact_hashes,canonical_hash,tokenizer_interface
from test_native_reasoning_baselines import tokenizer_fixture

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('native_preference_evaluation',ROOT/'scripts/run_native_preference_evaluation.py')
lab=importlib.util.module_from_spec(spec);spec.loader.exec_module(lab)
PRODUCTION_PRODUCER_BINDINGS=lab.producer_bindings


class StopModel:
    def eval(self):return self
    def __call__(self,input_ids,**kwargs):
        logits=torch.full((1,input_ids.shape[1],12),-100.);logits[0,-1,2]=100.
        return SimpleNamespace(logits=logits)


class LikelihoodModel(torch.nn.Module):
    """Authored twelve-token CPU control, never a pretrained checkpoint."""
    def __init__(self,fail=False):
        super().__init__();self.bias=torch.nn.Parameter(torch.arange(12,dtype=torch.float32));self.fail=fail
    def cuda(self):return self
    def forward(self,input_ids,**kwargs):
        if self.fail:raise RuntimeError('authored CPU reference forward failure')
        return SimpleNamespace(logits=self.bias.view(1,1,-1).expand(input_ids.shape[0],input_ids.shape[1],-1))


class PreferenceEvaluationControls(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup);self.root=Path(temporary.name)
        (self.root/'experiments/reports').mkdir(parents=True);(self.root/'outputs').mkdir()
        self.tokenizer=tokenizer_fixture(self.root/'tokenizer')
        template=self.tokenizer.chat_template
        source=self.root/'source.py';source.write_text('authored source binding\n')
        self.sources={'source.py':lab._digest(str(source))};self.inputs={}
        for name,text in [('template',template),('protocol',json.dumps({'groups':{f'valid-{i}':f'group-{i}' for i in range(4)}})),
                ('interpreter','authored inert interpreter binding'),('environment_lock','authored lock')]:
            path=self.root/(name+'.txt');path.write_text(text)
            if name=='interpreter':path.chmod(0o755)
            self.inputs[name]=lab._digest(str(path),executable=name=='interpreter')
        self.parent=dict(path=str(self.root/'tokenizer'),files={'authored-weights':'a'*64},acceptance={'authored':True})
        self.producers={role:dict(path=self.parent['path'],files=self.parent['files'],parent=self.parent) for role in lab.ROLES}
        self.items={name:[dict(id=f'{name}-{i}',source_group=f'{name}-source-{i//3}',task='location' if name=='location' else 'copy',
            split='publication',prompt='Compute 1 + 2.',kind='text',reference='answer',extraction='whole',format_policy='any',
            **({'family':'arithmetic','template_id':'original-template','campaign_role':'heldout-controlled-slice'} if name=='reasoning' else {}))
            for i in range(count)] for name,count in lab.COUNTS.items()}
        self.validation=[dict(id=f'valid-{i}',prompt=[{'role':'user','content':'Compute 1 + 2.'}],chosen='1',rejected='2.') for i in range(4)]
        interface=tokenizer_interface(self.tokenizer,template=template,stop_ids=[2])
        self.native=SimpleNamespace(restore_saved_template=lambda path:template,
            verify_parent_tokenizer=lambda *args,**kwargs:(interface,{}),
            state_digest=canonical_hash,check_split_interfaces=lambda *args:dict(encoded_prompt_collisions='none observed'),
            encode_pair=lambda *args:[([4,5,2],[False,True,True]),([4,9,2],[False,True,True])])
        encoded=[self.native.encode_pair() for _ in self.validation]
        for role in ('chosen100','dpo100'):
            self.producers[role]['continuity']=dict(interface=interface,encoded_validation_sha256=canonical_hash(encoded))
        self.adapter=SimpleNamespace(parent_binding=lambda:self.parent,MODEL='authored-model',REVISION='a'*40)
        for patcher in (patch.object(lab,'ROOT',self.root),patch.object(lab,'source_bindings',return_value=self.sources),
                patch.object(lab,'input_bindings',side_effect=lambda:deepcopy(self.inputs)),
                patch.object(lab,'producer_bindings',return_value=self.producers),
                patch.object(lab,'native',return_value=self.native),patch.object(lab,'parent_adapter',return_value=self.adapter),
                patch.object(lab,'load_local_tokenizer',return_value=self.tokenizer),
                patch.object(lab,'panels',return_value=(self.items,{'train':self.validation,'validation':self.validation,
                    'evaluation':self.validation}))):
            patcher.start();self.addCleanup(patcher.stop)

    def prepare(self):return lab.prepare('run-01')

    def test_default_preparation_never_loads_weights_or_launches(self):
        with (patch.object(lab,'_supervise') as supervise,patch.object(lab,'run_generation') as generation,
                patch('subprocess.Popen') as spawn,patch('transformers.AutoModelForCausalLM.from_pretrained') as weights):
            self.assertEqual(lab.main(['--run-id','run-01']),0)
        for mock in (supervise,generation,spawn,weights):mock.assert_not_called()
        record=json.loads((lab.paths('run-01')['evidence']/'preparation.json').read_text());lab.verify_prepared(record)
        self.assertEqual(record['limits']['planned_responses_each'],144)
        self.assertEqual(record['limits']['maximum_emitted_tokens_each'],9216)
        self.assertEqual(record['limits']['validation_policy_calls'],8)
        self.assertFalse(lab.paths('run-01')['output'].exists())

    def test_closed_selectors_fixed_roles_and_existing_targets_refuse(self):
        for value in ('../escape','run-1','run-123','run-01/x'):
            with self.assertRaises(ValueError):lab.paths(value)
        with self.assertRaises(ValueError):lab.command('best-quality','run-01')
        self.prepare()
        with self.assertRaises(FileExistsError):self.prepare()

    def test_frozen_inputs_sources_and_rehashed_command_prevent_launch(self):
        record=self.prepare();changed=deepcopy(record);changed['commands']['unchanged'].append('--allow-download')
        changed['preparation_sha256']=canonical_hash({k:v for k,v in changed.items() if k!='preparation_sha256'})
        with self.assertRaisesRegex(ValueError,'Closed preparation'):lab.verify_prepared(changed)
        with patch.object(lab,'source_bindings',return_value={'changed':1}),self.assertRaisesRegex(ValueError,'producer/reference/source'):
            lab.verify_prepared(record)
        path=Path(record['input_bindings']['location/contract']['path']);path.write_text(path.read_text()+' ')
        with patch.object(lab,'_supervise') as launch,self.assertRaisesRegex(ValueError,'Frozen input'):
            lab.execute(record,'Authored CPU refusal; no GPU work')
        launch.assert_not_called()

    def test_actual_producer_drift_prevents_evaluation(self):
        record=self.prepare();changed=deepcopy(self.producers);changed['dpo100']['files']={'changed':'b'*64}
        with patch.object(lab,'producer_bindings',return_value=changed),self.assertRaisesRegex(ValueError,'producer/reference/source'):
            lab.verify_prepared(record)

    def test_negative_group_differences_keep_population_and_missing_nulls(self):
        left={str(i):dict(source_group=f'source-{i//3}',correct=True) for i in range(6)}
        right={str(i):dict(source_group=f'source-{i//3}',correct=False) for i in range(6)}
        result=lab.grouped_difference(left,right,draws=100)
        self.assertEqual(result['difference'],-1);self.assertEqual(result['interval95'],[-1.,-1.])
        self.assertEqual(result['source_groups'],2)
        right['0']['correct']=None
        self.assertIsNone(lab.grouped_difference(left,right)['interval95'])
        right['0']['correct']=False;right['0']['source_group']='changed'
        with self.assertRaises(ValueError):lab.grouped_difference(left,right)

    def test_partial_consumer_keeps_all_expected_rows_errors_and_costs(self):
        record=self.prepare();evidence=lab.paths('run-01')['evidence'];output=lab.paths('run-01')['output']
        panel='assistant';contract=json.loads((evidence/(panel+'-contract.json')).read_text())
        contract['settings']['generation'].update(device='cpu',dtype='float32')
        contract['identity']=canonical_hash({k:v for k,v in contract.items() if k!='identity'})
        (evidence/(panel+'-contract.json')).write_text(json.dumps(contract))
        root=output/'unchanged'/panel;root.mkdir(parents=True)
        rows=[generate_record(StopModel(),self.tokenizer,item,contract,checkpoint_id='authored-stop-control',
            identity={'identity_sha256':'a'*64,'input_sha256':{}}) for item in self.items[panel][:2]]
        (root/'responses.jsonl').write_text(json.dumps(rows[0])+'\n'+ '{"interrupted-write":')
        (root/'events.jsonl').write_text(''.join(json.dumps({'stage':'partial_response','record':row})+'\n' for row in rows))
        result=lab.consume(record);slice=result['independent_panels'][panel]
        self.assertEqual(len(slice['rows']['unchanged']),120)
        self.assertFalse(slice['rows']['unchanged']['assistant-0']['correct'])
        self.assertIsNone(slice['rows']['unchanged']['assistant-1']['correct'])
        self.assertEqual(len(slice['incomplete_attempts']['unchanged']),1)
        self.assertEqual(slice['incomplete_attempts']['unchanged'][0]['cost']['generation_tokens'],1)
        self.assertEqual(len(result['unreadable_jsonl']),1)
        self.assertIsNone(slice['differences_vs_unchanged']['dpo100']['task/copy']['difference'])
        self.assertNotIn('universal_score',result)

    def test_failed_child_retains_actual_receipt_and_no_later_launch(self):
        record=self.prepare()
        with patch.object(lab,'_supervise',return_value=dict(status='failed',actual_exit_code=-15)) as child:
            with self.assertRaisesRegex(RuntimeError,'child failed'):lab.execute(record,'Authored CPU failure control; never GPU')
        self.assertEqual(child.call_count,1)
        evidence=lab.paths('run-01')['evidence']
        self.assertEqual(json.loads((evidence/'unchanged-returned-supervision.json').read_text())['actual_exit_code'],-15)
        self.assertTrue((evidence/'partial-comparison.json').exists())
        self.assertFalse((evidence/'chosen100-launch.json').exists())

    def test_actual_four_location_prompts_not_eight_are_fixed(self):
        self.assertEqual(lab.COUNTS,{'location':4,'assistant':120,'reasoning':20})
        record=self.prepare();retained=json.loads((lab.paths('run-01')['evidence']/'reasoning-items.json').read_text())
        self.assertEqual(retained,self.items['reasoning'])
        self.assertEqual(record['limits']['seconds_each'],900)

    def production_pilots(self):
        inputs=deepcopy(self.inputs)
        for key in ('train','validation','evaluation'):
            path=self.root/(key+'.jsonl');path.write_text('authored exact original role '+key+'\n')
            inputs[key]=lab._digest(str(path))
        geometry=dict(interface=self.producers['chosen100']['continuity']['interface'],
            encoded_validation_sha256=self.producers['chosen100']['continuity']['encoded_validation_sha256'],
            tokenizer_snapshot=dict(path=str(self.root/'tokenizer'),files=artifact_hashes(self.root/'tokenizer')))
        records={}
        for kind in ('chosen','dpo'):
            directory=self.root/'experiments/reports'/f'native-{kind}-pilot-20261005-run-01';directory.mkdir()
            policy=self.root/'outputs'/f'native-{kind}-pilot-20261005-run-01-pilot'/'policy';policy.mkdir(parents=True)
            prepared=dict(mode='pilot',run_id='run-01',limits=dict(updates=100),parent_binding=self.parent,
                input_bindings=inputs,geometry=geometry)
            result=dict(completed_recovery=dict(completed_updates=100,path='authored committed100'),reference_has_gradients=False,
                cumulative_work_ledger=dict(open_tickets=[],failed_tickets=[]),snapshot_io_ledger=dict(open_tickets=[],failed_tickets=[]))
            genealogy=dict(kind='full-HF-chosen-policy' if kind=='chosen' else 'full-HF-DPO-policy',
                parent_checkpoint=self.parent['path'],parent_checkpoint_sha256=self.parent['files'],
                checkpoint_interface=geometry['interface'],completed_recovery=result['completed_recovery'])
            (policy/'model.safetensors').write_bytes(b'authored inert payload; never deserialized')
            (policy/'course-genealogy.json').write_text(json.dumps(genealogy))
            (policy.parent/'result.json').write_text(json.dumps(result))
            (policy.parent/'metrics.jsonl').write_text(''.join(json.dumps(dict(update=i))+'\n' for i in range(1,101)))
            checks=dict.fromkeys(('all_actual_children_completed','completed100','exactly100_metrics','clean_journals'),True)
            if kind=='chosen':checks['unchanged_reference']=True
            acceptance=dict(status='passed',parent_binding=self.parent,checks=checks,results=dict(pilot=result),
                invocations=[dict(role='pilot',result=dict(status='completed',actual_exit_code=0))],
                exports=dict(pilot=dict(path=str(policy),files=artifact_hashes(policy))))
            (directory/'preparation.json').write_text(json.dumps(prepared));(directory/'acceptance.json').write_text(json.dumps(acceptance))
            records[kind]=dict(directory=directory,policy=policy,prepared=prepared,acceptance=acceptance)
        verify=unittest.mock.Mock()
        adapter=SimpleNamespace(verify_prepared=verify,observed_geometry=lambda *args:deepcopy(geometry))
        return inputs,geometry,records,adapter

    def test_production_gate_reads_actual100_receipts_exports_and_current_science(self):
        inputs,geometry,records,adapter=self.production_pilots()
        with (patch.object(lab,'input_bindings',return_value=inputs),patch.object(lab,'producer_adapter',return_value=adapter),
                patch('transformers.AutoModelForCausalLM.from_pretrained') as weights,patch('subprocess.Popen') as launch):
            result=PRODUCTION_PRODUCER_BINDINGS()
        self.assertEqual(adapter.verify_prepared.call_count,2)
        self.assertEqual(set(result),set(lab.ROLES));self.assertEqual(result['chosen100']['continuity']['interface'],geometry['interface'])
        weights.assert_not_called();launch.assert_not_called()

    def test_production_gate_refuses_parent_data_geometry_or_false_receipt(self):
        inputs,geometry,records,adapter=self.production_pilots()
        with patch.object(lab,'input_bindings',return_value=inputs),patch.object(lab,'producer_adapter',return_value=adapter):
            prepared=deepcopy(records['chosen']['prepared']);prepared['input_bindings']['train']={'changed':True}
            (records['chosen']['directory']/'preparation.json').write_text(json.dumps(prepared))
            with self.assertRaisesRegex(ValueError,'source/input/interface'):PRODUCTION_PRODUCER_BINDINGS()
            (records['chosen']['directory']/'preparation.json').write_text(json.dumps(records['chosen']['prepared']))
            changed=deepcopy(geometry);changed['encoded_validation_sha256']='b'*64
            adapter.observed_geometry=lambda *args:changed
            with self.assertRaisesRegex(ValueError,'source/input/interface'):PRODUCTION_PRODUCER_BINDINGS()
            adapter.observed_geometry=lambda *args:geometry
            accepted=deepcopy(records['dpo']['acceptance']);accepted['invocations'][0]['result']['actual_exit_code']=1
            (records['dpo']['directory']/'acceptance.json').write_text(json.dumps(accepted))
            with self.assertRaisesRegex(ValueError,'exit0'):PRODUCTION_PRODUCER_BINDINGS()

    def test_production_gate_refuses_actual_incomplete_metrics_or_export_drift(self):
        inputs,geometry,records,adapter=self.production_pilots()
        with patch.object(lab,'input_bindings',return_value=inputs),patch.object(lab,'producer_adapter',return_value=adapter):
            (records['chosen']['policy'].parent/'metrics.jsonl').write_text(json.dumps(dict(update=100))+'\n')
            with self.assertRaisesRegex(ValueError,'export/genealogy'):PRODUCTION_PRODUCER_BINDINGS()
            (records['chosen']['policy'].parent/'metrics.jsonl').write_text(''.join(json.dumps(dict(update=i))+'\n' for i in range(1,101)))
            (records['dpo']['policy']/'model.safetensors').write_bytes(b'authored changed payload')
            with self.assertRaisesRegex(ValueError,'export/genealogy'):PRODUCTION_PRODUCER_BINDINGS()

    def test_rehashed_limits_contract_and_native_mask_changes_refuse(self):
        record=self.prepare()
        for change in ('limits','mask','interface'):
            changed=deepcopy(record)
            if change=='limits':changed['limits']['seconds_each']=1800
            elif change=='mask':changed['encoded_validation'][0][0][1][0]=True
            else:changed['interfaces']['dpo100']['template_sha256']='b'*64
            changed['preparation_sha256']=canonical_hash({k:v for k,v in changed.items() if k!='preparation_sha256'})
            with self.assertRaises(ValueError):lab.verify_prepared(changed)

    def test_original_supervisor_failure_survives_consumer_failure(self):
        record=self.prepare()
        with (patch.object(lab,'_supervise',return_value=dict(status='failed',actual_exit_code=-15)),
                patch.object(lab,'consume',side_effect=ValueError('authored malformed retained record'))):
            with self.assertRaisesRegex(RuntimeError,'child failed'):lab.execute(record,'Authored CPU failure control; never GPU')
        evidence=lab.paths('run-01')['evidence'];failure=json.loads((evidence/'failure.json').read_text())
        self.assertEqual(failure['type'],'RuntimeError');self.assertEqual(failure['invocations'][0]['result']['actual_exit_code'],-15)
        self.assertEqual(failure['partial_consumer_failure']['type'],'ValueError')

    def test_negative_likelihood_differences_are_separate_from_correctness(self):
        left={str(i):dict(source_group=f'source-{i//2}',chosen_logp=-4.) for i in range(4)}
        right={str(i):dict(source_group=f'source-{i//2}',chosen_logp=-7.) for i in range(4)}
        result=lab.grouped_numeric_difference(left,right,'chosen_logp',draws=100)
        self.assertEqual(result['difference'],-3.);self.assertEqual(result['interval95'],[-3.,-3.])
        right['0']['chosen_logp']=float('nan')
        with self.assertRaises(ValueError):lab.grouped_numeric_difference(left,right,'chosen_logp')

    def test_actual_tiny_cpu_single_shift_logps_margin_and_forward_costs(self):
        record=self.prepare();root=lab.paths('run-01')['output']/'unchanged';root.mkdir(parents=True)
        spec=importlib.util.spec_from_file_location('original_native_pair_cpu',ROOT/'scripts/run_chapter11_spark_dpo.py')
        native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
        dpo=SimpleNamespace(model_pair_loss=native.model_pair_loss,effective_model_contract=lambda model:{},
            collate=lambda rows,branch,pad,device:native.collate(rows,branch,pad,'cpu'))
        with (patch.object(lab,'native',return_value=dpo),
                patch('transformers.AutoModelForCausalLM.from_pretrained',side_effect=[LikelihoodModel(),LikelihoodModel()]) as loader,
                patch('torch.autocast',return_value=nullcontext()),patch('torch.cuda.synchronize'),patch('torch.cuda.empty_cache')):
            lab.pair_scores(record,root,lambda:None)
        rows=lab.jsonl(root/'pair-scores.jsonl');summary=json.loads((root/'pair-summary.json').read_text())
        self.assertEqual(len(rows),4);self.assertEqual(loader.call_count,2)
        for row in rows:
            self.assertAlmostEqual(row['chosen_logp']-row['rejected_logp'],-4.,places=5)
            self.assertEqual(row['reference_relative_margin'],0.);self.assertEqual(row['chosen_targets'],2)
        for role in ('policy','reference'):
            self.assertEqual(summary['cost'][role],dict(attempted_calls=8,forward_calls=8,attempted_positions=16,forward_positions=16))
        self.assertFalse(summary['reference_has_gradients'])
        self.assertEqual(lab.jsonl(root/'pair-cost-events.jsonl')[-1]['cost'],summary['cost'])

    def test_actual_tiny_cpu_failed_forward_keeps_entered_costs(self):
        record=self.prepare();root=lab.paths('run-01')['output']/'unchanged';root.mkdir(parents=True)
        spec=importlib.util.spec_from_file_location('original_native_pair_cpu_failure',ROOT/'scripts/run_chapter11_spark_dpo.py')
        native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
        dpo=SimpleNamespace(model_pair_loss=native.model_pair_loss,effective_model_contract=lambda model:{},
            collate=lambda rows,branch,pad,device:native.collate(rows,branch,pad,'cpu'))
        with (patch.object(lab,'native',return_value=dpo),
                patch('transformers.AutoModelForCausalLM.from_pretrained',side_effect=[LikelihoodModel(),LikelihoodModel(fail=True)]),
                patch('torch.autocast',return_value=nullcontext()),patch('torch.cuda.synchronize'),patch('torch.cuda.empty_cache')):
            with self.assertRaisesRegex(RuntimeError,'reference forward failure'):lab.pair_scores(record,root,lambda:None)
        cost=lab.jsonl(root/'pair-cost-events.jsonl')[-1]['cost']
        self.assertEqual(cost['policy'],dict(attempted_calls=2,forward_calls=2,attempted_positions=4,forward_positions=4))
        self.assertEqual(cost['reference'],dict(attempted_calls=1,forward_calls=0,attempted_positions=2,forward_positions=0))
        self.assertEqual(lab.jsonl(root/'pair-scores.jsonl'),[]);self.assertFalse((root/'pair-summary.json').exists())

    def receipt_fixture(self,record):
        role,panel='unchanged','assistant';root=lab.paths('run-01')['output']/role/panel;root.mkdir(parents=True)
        contract=json.loads((lab.paths('run-01')['evidence']/(panel+'-contract.json')).read_text())
        for key in lab.GENERATION_SOURCES:
            path=self.root/key;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('authored receipt source '+key+'\n')
            record['source_bindings'][key]=lab._digest(str(path))
        inputs={str(Path(record['input_bindings'][panel+'/'+kind]['path']).relative_to(self.root)):
            record['input_bindings'][panel+'/'+kind]['sha256'] for kind in ('items','contract')}
        identity=dict(schema_version=1,checkpoint_path=self.producers[role]['path'],checkpoint_files=self.producers[role]['files'],
            selected_tokenizer_path=self.producers[role]['path'],
            selected_tokenizer_files=artifact_hashes(self.producers[role]['path'],patterns=lab.TOKENIZER_PATTERNS),
            input_sha256=inputs,source_sha256={key:record['source_bindings'][key]['sha256'] for key in lab.GENERATION_SOURCES},
            config=contract['settings'],checkpoint_interface=contract['settings']['generation']['interface'],device=dict(mode='cuda'),
            command=record['commands'][role],environment=dict(interpreter=record['input_bindings']['interpreter']['path'],
                environment_lock=dict(status='hashed',path=record['input_bindings']['environment_lock']['path'],
                    sha256=record['input_bindings']['environment_lock']['sha256'])))
        identity['identity_sha256']=canonical_hash(identity)
        summary=dict(identity_sha256=identity['identity_sha256'])
        item=self.items[panel][0]
        checkpoint='local-hf-sha256:'+canonical_hash(dict(files=record['producers'][role]['files'],
            tokenizer=identity['selected_tokenizer_files'],interface=contract['settings']['generation']['interface']['interface_sha256']))
        rows=[dict(item_id=item['id'],source_group=item['source_group'],task=item['task'],split=item['split'],sample_index=0,
            checkpoint_id=checkpoint,contract_id=contract['identity'],settings_sha256=canonical_hash(contract['settings']),
            checkpoint_interface_sha256=contract['settings']['generation']['interface']['interface_sha256'],
            input_identity_sha256=identity['identity_sha256'],input_sha256=inputs)]
        path=root/'input-identity.json';path.write_text(json.dumps(identity))
        return role,panel,contract,path,identity,summary,rows

    def test_saved_input_receipt_missing_or_self_hash_altered_is_not_complete(self):
        record=self.prepare();role,panel,contract,path,identity,summary,rows=self.receipt_fixture(record)
        self.assertTrue(lab.identity_receipt_check(record,role,panel,contract,summary,rows)[0])
        path.unlink()
        valid,evidence=lab.identity_receipt_check(record,role,panel,contract,summary,rows)
        self.assertFalse(valid);self.assertEqual(evidence['status'],'missing')
        identity['checkpoint_files']={'unrelated-model':'b'*64};path.write_text(json.dumps(identity))
        self.assertFalse(lab.identity_receipt_check(record,role,panel,contract,summary,rows)[0])

    def test_rehashed_unrelated_receipt_cannot_join_selected_input_lock_and_config(self):
        record=self.prepare();role,panel,contract,path,identity,summary,rows=self.receipt_fixture(record)
        for field in ('checkpoint','tokenizer','input','source','lock','config'):
            altered=deepcopy(identity)
            if field=='checkpoint':altered['checkpoint_files']={'unrelated-model':'b'*64}
            elif field=='tokenizer':altered['selected_tokenizer_files']={'unrelated-tokenizer':'b'*64}
            elif field=='input':altered['input_sha256']={'unrelated-input':'b'*64}
            elif field=='source':altered['source_sha256']={'unrelated-source':'b'*64}
            elif field=='lock':altered['environment']['environment_lock']['sha256']='b'*64
            else:altered['config']['max_new_tokens']=128
            altered['identity_sha256']=canonical_hash({key:value for key,value in altered.items() if key!='identity_sha256'})
            path.write_text(json.dumps(altered));joined_summary=dict(identity_sha256=altered['identity_sha256'])
            joined_rows=[dict(rows[0],input_identity_sha256=altered['identity_sha256'],input_sha256=altered['input_sha256'])]
            self.assertFalse(lab.identity_receipt_check(record,role,panel,contract,joined_summary,joined_rows)[0],field)

    def test_consumer_complete_panel_requires_actual_saved_identity_receipt(self):
        record=self.prepare();panel='assistant';evidence=lab.paths('run-01')['evidence']
        # Actual authored CPU generations, never model-scale/CUDA evidence.
        contract=json.loads((evidence/(panel+'-contract.json')).read_text())
        contract['settings']['generation'].update(device='cpu',dtype='float32')
        contract['identity']=canonical_hash({key:value for key,value in contract.items() if key!='identity'})
        (evidence/(panel+'-contract.json')).write_text(json.dumps(contract))
        record['input_bindings'][panel+'/contract']=lab._digest(str(evidence/(panel+'-contract.json')))
        role,panel,contract,path,identity,summary,_=self.receipt_fixture(record)
        identity['device']['mode']='cpu';identity['identity_sha256']=canonical_hash({k:v for k,v in identity.items() if k!='identity_sha256'})
        path.write_text(json.dumps(identity))
        checkpoint='local-hf-sha256:'+canonical_hash(dict(files=record['producers'][role]['files'],
            tokenizer=identity['selected_tokenizer_files'],interface=contract['settings']['generation']['interface']['interface_sha256']))
        rows=[generate_record(StopModel(),self.tokenizer,item,contract,checkpoint_id=checkpoint,identity=identity)
            for item in self.items[panel]]
        (path.parent/'responses.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        (path.parent/'summary.json').write_text(json.dumps(dict(status='completed',record_count=120,
            contract_id=contract['identity'],checkpoint_id=checkpoint,identity_sha256=identity['identity_sha256'],inputs_unchanged=True)))
        result=lab.consume(record);key=role+'/'+panel+'/generation_identity_complete'
        self.assertTrue(result['checks'][key]);self.assertEqual(result['independent_panels'][panel]['summaries'][role]['completed'],120)
        path.unlink();result=lab.consume(record)
        self.assertFalse(result['checks'][key]);self.assertEqual(result['independent_panels'][panel]['summaries'][role]['completed'],120)
        self.assertEqual(result['independent_panels'][panel]['input_identity_receipts'][role]['status'],'missing')

    def test_identically_wrong_likelihood_groups_do_not_change_frozen_bootstrap(self):
        record=self.prepare()
        for role in lab.ROLES:
            root=lab.paths('run-01')['output']/role;root.mkdir(parents=True)
            rows=[dict(id=row['id'],source_group='same-wrong-group',chosen_logp=-4.,rejected_logp=-5.,
                reference_relative_margin=.1,reference_relative_logp_margin=1.) for row in record['validation_rows']]
            (root/'pair-scores.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        summary=lab.consume(record)
        for role in lab.ROLES:self.assertFalse(summary['checks'][role+'/frozen_validation_groups'])
        for role in ('chosen100','dpo100'):
            result=summary['validation_grouped_differences_vs_unchanged'][role]['chosen_logp']
            self.assertEqual(result['status'],'invalid-frozen-groups');self.assertIsNone(result['interval95'])
        self.assertEqual(summary['validation_likelihood_and_reference_margin']['unchanged'][0]['source_group'],'same-wrong-group')

    def test_partial_cost_provenance_binds_receipt_without_a_completed_summary(self):
        record=self.prepare();role,panel,contract,path,identity,_,rows=self.receipt_fixture(record)
        # Receipt-join unit control, not a completed/native generation claim.
        (path.parent/'events.jsonl').write_text(json.dumps(dict(stage='partial_response',record=dict(rows[0],cost={'generation_tokens':1})))+'\n')
        result=lab.consume(record);key=role+'/'+panel+'/partial_identity_join'
        self.assertTrue(result['checks'][key]);self.assertFalse(result['checks'][role+'/'+panel+'/generation_identity_complete'])
        retained=result['independent_panels'][panel]['incomplete_attempts'][role][0]
        self.assertEqual(retained['provenance_status'],'bound-selected-policy')
        wrong=dict(rows[0],checkpoint_id='unrelated-policy',cost={'generation_tokens':1})
        (path.parent/'events.jsonl').write_text(json.dumps(dict(stage='partial_response',record=wrong))+'\n')
        result=lab.consume(record);self.assertFalse(result['checks'][key])
        retained=result['independent_panels'][panel]['incomplete_attempts'][role][0]
        self.assertEqual(retained['provenance_status'],'unverified-retained-cost');self.assertEqual(retained['cost']['generation_tokens'],1)

    def test_valid_event_journal_above16MiB_preserves_partial_comparison(self):
        record=self.prepare();evidence=lab.paths('run-01')['evidence'];panel='assistant'
        contract=json.loads((evidence/(panel+'-contract.json')).read_text())
        contract['settings']['generation'].update(device='cpu',dtype='float32')
        contract['identity']=canonical_hash({key:value for key,value in contract.items() if key!='identity'})
        (evidence/(panel+'-contract.json')).write_text(json.dumps(contract))
        partial=generate_record(StopModel(),self.tokenizer,self.items[panel][0],contract,
            checkpoint_id='authored-large-event-reader-control',identity={'identity_sha256':'a'*64,'input_sha256':{}})
        root=lab.paths('run-01')['output']/'unchanged'/panel;root.mkdir(parents=True)
        line=json.dumps(dict(stage='partial_response',record=partial))+'\n'
        path=root/'events.jsonl';path.write_text(line*(17*1024**2//len(line)+1))
        self.assertGreater(path.stat().st_size,16*1024**2)
        summary=lab.consume(record);retained=summary['independent_panels'][panel]
        self.assertEqual(len(retained['rows']['unchanged']),120)
        self.assertEqual(len(retained['incomplete_attempts']['unchanged']),1)
        self.assertEqual(retained['incomplete_attempts']['unchanged'][0]['cost']['generation_tokens'],1)


if __name__=='__main__':unittest.main()
