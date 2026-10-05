"""Authored CPU closed-selector/common20 controls, never pretrained/CUDA runs."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from test_native_reasoning_baselines import tokenizer_fixture,ImmediateEOS
from dongxi_llms.reasoning_generation import generate_record,freeze_local_contract
from dongxi_llms.run_identity import artifact_hashes,canonical_hash,TOKENIZER_PATTERNS

ROOT=Path(__file__).resolve().parents[1]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

lab=module('native_rlvr_evaluation',ROOT/'scripts/run_native_rlvr_evaluation.py')
original_baseline=module('rlvr_eval_test_baseline',ROOT/'scripts/run_native_reasoning_baselines.py')


class NativeRLVREvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        temporary=tempfile.TemporaryDirectory(prefix='dongxi-rlvr-eval-authored-');self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        for name in lab.SOURCES:
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('authored CPU source fixture\n')
        (self.root/'outputs').mkdir();(self.root/'experiments/reports').mkdir(parents=True)
        self.cache=self.root/'cached-instruct';self.tokenizer=tokenizer_fixture(self.cache)
        self.write(self.cache/'config.json',{'authored_fixture':True});(self.cache/'model.safetensors').write_bytes(b'authored, not pretrained weights')
        self.panel=original_baseline.logical_panel()
        self.base_model=dict(path=str(self.cache),model='Qwen/Qwen3-0.6B',revision='c1899de289a04d12100db370d81485cdf75e47ca',
            files=artifact_hashes(self.cache),revision_evidence='Authored CPU fixture, not actual pilot evidence')
        self.lock=self.root/'selected.lock';self.lock.write_text('authored lock\n')
        inputs={}
        for key in ('math_items','custom_template'):
            path=self.root/(key+'.txt');path.write_text('authored fixed bytes\n');inputs[key]=lab._digest(str(path))
        inputs.update(interpreter=lab._digest(sys.executable,executable=True),environment_lock=lab._digest(str(self.lock)))
        source={name:lab._digest(str(self.root/name)) for name in lab.GENERATION_SOURCES}
        for key,value in (("ROOT",self.root),("INTERPRETER",sys.executable),("ENVIRONMENT_LOCK",str(self.lock))):
            current=patch.object(lab,key,value);current.start();self.addCleanup(current.stop)
        replacements=(patch.object(original_baseline,'ROOT',self.root),
            patch.object(original_baseline,'logical_panel',return_value=self.panel),
            patch.object(original_baseline,'source_bindings',return_value=source),
            patch.object(original_baseline,'fixed_inputs',side_effect=lambda:deepcopy(inputs)),
            patch.object(original_baseline,'local_model_binding',return_value=self.base_model),
            patch.object(original_baseline,'load_local_tokenizer',return_value=self.tokenizer),
            patch.object(lab,'baseline',return_value=original_baseline),
            patch.object(lab,'load_local_tokenizer',return_value=self.tokenizer))
        for current in replacements:current.start();self.addCleanup(current.stop)
        self.base_records={}
        for budget in lab.BUDGETS:
            record=original_baseline.prepare('instruct-thinking-off',budget,'run-01');self.base_records[budget]=record
            self.populate(record)
            self.write(Path(record['evidence'])/'acceptance.json',dict(status='passed',checks={key:True for key in lab.BASELINE_CHECKS},
                row='instruct-thinking-off',budget=budget,actual_exit_code=0,local_model_binding=self.base_model,
                evaluation=original_baseline.output_summary(record)))
        self.declarations={};self.accepted={}
        self.stage=SimpleNamespace(REVISION=self.base_model['revision'],
            paths=self.pilot_paths,verify_prepared=lambda record:None,
            fixed_recipe=lambda stage:dict(updates=16,output_cap=64,seed=2323,lr=1e-6,beta=.02),
            pilot_export_binding=self.pilot_export_binding)
        current=patch.object(lab,'stages',return_value=self.stage);current.start();self.addCleanup(current.stop)
        interface=freeze_local_contract(self.panel['items'],original_baseline.settings('instruct-thinking-off',32,self.tokenizer,'sample',1009),
            self.tokenizer)['settings']['generation']['interface']
        for group in lab.GROUPS:
            paths=self.pilot_paths('pilot',group,'run-01');policy=paths['pilot']/'policy'
            self.tokenizer.save_pretrained(policy);self.write(policy/'config.json',{'authored_fixture':True})
            (policy/'model.safetensors').write_bytes(f'authored group{group} weights'.encode())
            self.write(policy/'course-genealogy.json',dict(kind='full-HF-model',objective='course-response-mean-GRPO',
                parent_local_source_hashes=self.base_model['files'],upstream_revision_metadata=self.base_model['revision'],
                checkpoint_interface=interface,template_sha256=interface['template_sha256']))
            declaration=dict(stage='pilot',group=group,run_id='run-01',recipe=self.stage.fixed_recipe('pilot'),
                locations={'pilot':str(paths['pilot'])},local_model_binding=self.base_model,
                observed_geometry=dict(checkpoint_interface=interface,prompt_contract={'template_sha256':interface['template_sha256']}))
            report=dict(status='completed',committed_completed_updates=16,local_source_hashes=self.base_model['files'])
            accepted=dict(status='passed',stage='pilot',group=group,checks={key:True for key in lab.PILOT_CHECKS},
                invocations=[dict(role='pilot',status='completed',actual_exit_code=0)],reports={'pilot':report},local_model_binding=self.base_model)
            self.declarations[group]=declaration;self.accepted[group]=accepted
            self.write(paths['evidence']/'preparation.json',declaration);self.write(paths['evidence']/'acceptance.json',accepted)
            self.write(paths['pilot']/'report.json',report)
            accepted['exports']={'pilot':self.pilot_export_binding(declaration,report)}
            self.write(paths['evidence']/'acceptance.json',accepted)
        current=patch.object(lab,'_supervise');self.supervise=current.start();self.addCleanup(current.stop)

    def write(self,path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n');return path

    def pilot_paths(self,stage,group,run_id):
        stem=f'native-rlvr-g{group}-{stage}-20261005-{run_id}'
        return dict(evidence=self.root/'experiments/reports'/stem,pilot=self.root/'outputs'/stem)

    def pilot_export_binding(self,declaration,report):
        output=self.pilot_paths('pilot',declaration['group'],'run-01')['pilot'];policy=output/'policy'
        return dict(path=str(policy),files=artifact_hashes(policy),completed_updates=16,
            report_binding=lab._digest(str(output/'report.json')),final_snapshot={'authored_final16_identity':'a'*64})

    def populate(self,record,checkpoint_id=None):
        for cell in record['cells']:
            contract=json.loads(Path(cell['contract_path']).read_text());cpu=deepcopy(contract)
            cpu['settings']['generation'].update(device='cpu',dtype='float32')
            cpu['identity']=canonical_hash({key:value for key,value in cpu.items() if key!='identity'})
            # Authored CPU replay records exercise the renderer's binding paths,
            # not actual CUDA generation. The original closed contract identity
            # is deliberately rebound for this injected wrapper control only.
            model=record['local_model_binding'];interface=contract['settings']['generation']['interface']
            tokenizers=artifact_hashes(model['path'],patterns=TOKENIZER_PATTERNS)
            expected_checkpoint='local-hf-sha256:'+canonical_hash(dict(files=model['files'],tokenizer=tokenizers,interface=interface['interface_sha256']))
            input_map={str((Path(record['evidence'])/'items.json').relative_to(self.root)):record['input_bindings']['items']['sha256'],
                str(Path(cell['contract_path']).relative_to(self.root)):record['input_bindings']['contract/'+cell['id']]['sha256']}
            identity=dict(schema_version=1,checkpoint_path=str(Path(model['path']).resolve()),checkpoint_files=model['files'],
                selected_tokenizer_path=str(Path(model['path']).resolve()),selected_tokenizer_files=tokenizers,
                config=contract['settings'],checkpoint_interface=interface,device={'mode':contract['settings']['generation']['device']},command=record['argv'],
                source_sha256={name:record['source_bindings'][name]['sha256'] for name in lab.GENERATION_SOURCES},
                input_sha256=input_map,environment=dict(interpreter=sys.executable,environment_lock=dict(status='hashed',
                    path=str(self.lock),sha256=record['input_bindings']['environment_lock']['sha256'])))
            identity['identity_sha256']=canonical_hash(identity);rows=[]
            for item in self.panel['items']:
                row=generate_record(ImmediateEOS(),self.tokenizer,item,cpu,checkpoint_id=checkpoint_id or expected_checkpoint,identity=identity)
                row.update(contract_id=contract['identity'],settings_sha256=canonical_hash(contract['settings']))
                rows.append(row)
            output=Path(cell['output']);output.mkdir(parents=True)
            (output/'responses.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            self.write(output/'input-identity.json',identity);self.write(output/'observed-interface.json',interface)
            self.write(output/'summary.json',dict(status='completed',record_count=20,checkpoint_id=checkpoint_id or expected_checkpoint,
                contract_id=contract['identity'],identity_sha256=identity['identity_sha256'],inputs_unchanged=True))

    def actual_control(self,group,budget):
        record=lab.prepare(group,budget,'run-01')
        def supervise(argv,output,**kwargs):
            self.populate(record)
            return dict(status='completed',actual_exit_code=0,child_seconds=.1,minimum_sampled_available_bytes=30*lab.GIB)
        self.supervise.side_effect=supervise
        self.assertEqual(lab.execute(record,'Goal authorized authored CPU supervisor control'),0)
        return record

    def test_all_four_preparations_keep_original20_five_cells_and_no_model_execution(self):
        with patch('transformers.AutoModelForCausalLM.from_pretrained') as weights,patch.object(lab,'run_generation') as generate:
            for group in lab.GROUPS:
                for budget in lab.BUDGETS:
                    record=lab.prepare(group,budget,'run-01');lab.verify_prepared(record)
                    self.assertEqual(json.loads((Path(record['evidence'])/'items.json').read_text()),self.panel['items'])
                    self.assertEqual([cell['id'] for cell in record['cells']],['sample-1009','sample-1019','sample-1029','sample-1039','greedy-1009'])
                    self.assertEqual(record['limits']['planned_responses'],100);self.assertEqual(record['limits']['external_seconds'],900)
                    contract=json.loads(Path(record['cells'][0]['contract_path']).read_text())
                    self.assertEqual(contract['settings']['decoding'],dict(mode='sample',seed=1009,temperature=1.,top_k=None,top_p=1.))
                    self.assertEqual(contract['settings']['thinking_mode'],'disabled')
                    self.assertEqual(contract['identity'],self.base_records[budget]['cells'][0]['contract_id'])
        weights.assert_not_called();generate.assert_not_called();self.supervise.assert_not_called()

    def test_actual16_gate_and_export_genealogy_bindings_required(self):
        paths=self.pilot_paths('pilot',4,'run-01');accepted=deepcopy(self.accepted[4])
        accepted['reports']['pilot']['committed_completed_updates']=15
        self.write(paths['evidence']/'acceptance.json',accepted)
        with self.assertRaisesRegex(ValueError,'accepted16'):lab.prepare(4,32,'run-01')
        self.supervise.assert_not_called()

    def test_current_pilot_report_must_equal_accepted_report(self):
        self.write(self.pilot_paths('pilot',4,'run-01')['pilot']/'report.json',dict(status='failed'))
        with self.assertRaisesRegex(ValueError,'report bytes differ'):lab.prepare(4,32,'run-01')

    def test_swapped_policy_before_preparation_cannot_rebind_actual16_acceptance(self):
        (self.pilot_paths('pilot',4,'run-01')['pilot']/'policy/model.safetensors').write_bytes(b'not the accepted16 export')
        with self.assertRaisesRegex(ValueError,'accepted committed16'):lab.prepare(4,32,'run-01')
        self.supervise.assert_not_called()
        self.assertTrue((lab.paths(4,32,'run-01')['evidence']/'preparation-failure.json').exists())

    def test_changed_accepted_final16_receipt_binding_refuses_before_preparation(self):
        accepted=deepcopy(self.accepted[4]);accepted['exports']['pilot']['final_snapshot']['authored_final16_identity']='b'*64
        self.write(self.pilot_paths('pilot',4,'run-01')['evidence']/'acceptance.json',accepted)
        with self.assertRaisesRegex(ValueError,'accepted committed16'):lab.prepare(4,32,'run-01')

    def test_unmatched_native_template_or_parent_refuses(self):
        policy=lab.selected_policy(4)
        policy['checkpoint_interface']['template_sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'native tokenizer/template'):lab.native_interface(policy,self.tokenizer,32)
        policy=lab.selected_policy(4);policy['upstream_parent']=dict(policy['upstream_parent'],revision='f'*40)
        with self.assertRaisesRegex(ValueError,'exact common cached'):lab.native_interface(policy,self.tokenizer,32)

    def test_acceptance_export_and_source_drift_refuse_before_launch(self):
        record=lab.prepare(4,32,'run-01')
        (self.pilot_paths('pilot',4,'run-01')['pilot']/'policy/model.safetensors').write_bytes(b'changed actual export')
        with self.assertRaisesRegex(ValueError,'accepted committed16'):lab.execute(record,'Goal authorized fixed RLVR evaluation')
        self.supervise.assert_not_called()

    def test_source_or_rehashed_arbitrary_command_drift_refuses(self):
        record=lab.prepare(4,32,'run-01');record['argv']=['/usr/bin/true']
        record['preparation_sha256']=canonical_hash({key:value for key,value in record.items() if key!='preparation_sha256'})
        with self.assertRaisesRegex(ValueError,'Closed RLVR'):lab.execute(record,'Goal authorized fixed RLVR evaluation')
        self.supervise.assert_not_called()

    def test_rehashed_changed_predeclared_limits_refuse(self):
        record=lab.prepare(4,32,'run-01');record['limits']['external_seconds']=901
        record['preparation_sha256']=canonical_hash({key:value for key,value in record.items() if key!='preparation_sha256'})
        with self.assertRaisesRegex(ValueError,'Closed RLVR'):lab.verify_prepared(record)

    def test_closed_selectors_and_existing_outputs_refuse(self):
        for args in ((True,32,'run-01'),(4,True,'run-01'),(4,256,'run-01'),(16,32,'run-01'),(4,32,'../escape')):
            with self.assertRaises(ValueError):lab.paths(*args)
        locations=lab.paths(4,32,'run-01');locations['output'].mkdir()
        with self.assertRaises(FileExistsError):lab.prepare(4,32,'run-01')

    def test_missing_launch_receipt_never_generates(self):
        lab.prepare(4,32,'run-01')
        with patch.object(lab,'run_generation') as generate,self.assertRaises(FileNotFoundError):lab.child(4,32,'run-01')
        generate.assert_not_called()

    def test_supervised_900_execution_keeps_negative_records_costs_and_overlap(self):
        record=self.actual_control(4,32)
        self.assertEqual(self.supervise.call_args.kwargs['seconds'],900)
        accepted=json.loads((Path(record['evidence'])/'acceptance.json').read_text())
        self.assertEqual(accepted['evaluation']['all_original_rows']['records'],100)
        self.assertEqual(accepted['evaluation']['all_original_rows']['correct'],0)
        self.assertLess(accepted['evaluation']['heldout']['records'],100)
        self.assertEqual(accepted['evaluation']['all_original_rows']['costs']['generation_tokens'],100)
        self.assertTrue(any(row['panel_annotation']['rlvr_train_problem_overlap'] for row in accepted['evaluation']['rows']))

    def test_failed_supervisor_retains_partial_observed_attempts_without_retry(self):
        record=lab.prepare(4,32,'run-01')
        self.supervise.return_value=dict(status='failed',actual_exit_code=-15,child_seconds=.1,minimum_sampled_available_bytes=30*lab.GIB)
        with self.assertRaisesRegex(RuntimeError,'incomplete'):lab.execute(record,'Goal authorized fixed RLVR evaluation')
        self.assertEqual(self.supervise.call_count,1)
        evidence=Path(record['evidence']);self.assertTrue((evidence/'returned-supervision.json').exists())
        accepted=json.loads((evidence/'acceptance.json').read_text());self.assertEqual(accepted['status'],'failed')
        self.assertEqual(accepted['evaluation']['missing_responses'],100)

    def test_common20_comparison_separates_caps_seeds_and_train_overlap(self):
        for group in lab.GROUPS:self.actual_control(group,32)
        self.assertEqual(lab.compare_cap(32,'run-01'),0)
        report=json.loads((self.root/'experiments/reports/native-rlvr-common20-cap32-20261005-run-01.json').read_text())
        self.assertEqual(report['budget'],32);self.assertEqual(len(report['cells']),5)
        self.assertEqual(set(report['all_original']),{'unchanged-instruct','rlvr-g4','rlvr-g8'})
        self.assertTrue(report['known_overlap_item_ids'])
        self.assertEqual(report['all_original']['rlvr-g4']['records'],100)
        for cell in report['cells']:
            self.assertEqual(cell['all_original']['rlvr-g8']['records'],20)
            self.assertLess(cell['heldout']['rlvr-g8']['records'],20)
            self.assertEqual(cell['paired_heldout']['rlvr-g4']['delta'],0.)
        self.assertFalse((self.root/'experiments/reports/native-rlvr-common20-cap128-20261005-run-01.json').exists())

    def test_common_comparison_requires_both_policies_and_current_accepted_replay(self):
        self.actual_control(4,32)
        with self.assertRaises(FileNotFoundError):lab.compare_cap(32,'run-01')

    def test_true_checks_cannot_replace_actual_exit0_supervision_or_retained_receipt(self):
        for group in lab.GROUPS:self.actual_control(group,32)
        evidence=lab.paths(4,32,'run-01')['evidence'];path=evidence/'acceptance.json';accepted=json.loads(path.read_text())
        for changes in ({'status':'failed'},{'actual_exit_code':7}):
            changed=deepcopy(accepted);changed['supervision'].update(changes);self.write(path,changed)
            with self.assertRaisesRegex(ValueError,'complete actual'):lab.compare_cap(32,'run-01')
        self.write(path,accepted);returned=json.loads((evidence/'returned-supervision.json').read_text());returned['child_seconds']=9
        self.write(evidence/'returned-supervision.json',returned)
        with self.assertRaisesRegex(ValueError,'complete actual'):lab.compare_cap(32,'run-01')

    def test_swapped_raw_checkpoint_and_input_identity_fail_despite_valid_contract(self):
        record=lab.prepare(4,32,'run-01');self.populate(record)
        output=Path(record['cells'][0]['output']);path=output/'responses.jsonl';original=path.read_text()
        for key,value in (('checkpoint_id','authored-other-model'),('input_identity_sha256','b'*64),
                ('input_sha256',{}),('prompt_token_ids',[999])):
            rows=[json.loads(line) for line in original.splitlines()];rows[0][key]=value
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            gate=lab.generation_identity(record);self.assertFalse(gate['complete']);self.assertTrue(gate['cells'][0]['issues'])
        path.write_text(original);identity=json.loads((output/'input-identity.json').read_text());identity['checkpoint_path']='/different-export'
        identity['identity_sha256']=canonical_hash({key:value for key,value in identity.items() if key!='identity_sha256'})
        self.write(output/'input-identity.json',identity);self.assertFalse(lab.generation_identity(record)['complete'])

    def test_swapped_partial_cost_record_and_summary_identity_are_rejected(self):
        record=lab.prepare(4,32,'run-01');self.populate(record);output=Path(record['cells'][0]['output'])
        row=json.loads((output/'responses.jsonl').read_text().splitlines()[0]);row['checkpoint_id']='other-model'
        (output/'events.jsonl').write_text(json.dumps(dict(stage='partial_response',record=row))+'\n')
        self.assertFalse(lab.generation_identity(record)['complete'])
        (output/'events.jsonl').write_text('\n');summary=json.loads((output/'summary.json').read_text());summary['inputs_unchanged']=False
        self.write(output/'summary.json',summary);self.assertFalse(lab.generation_identity(record)['complete'])

    def test_baseline_copied_other_model_records_cannot_admit_preparation(self):
        cell=self.base_records[32]['cells'][0];path=Path(cell['output'])/'responses.jsonl'
        rows=[json.loads(line) for line in path.read_text().splitlines()];rows[0]['checkpoint_id']='other-model'
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        # Rebinding the replay summary alone is not enough to forge the model/input join.
        evidence=Path(self.base_records[32]['evidence']);accepted=json.loads((evidence/'acceptance.json').read_text())
        accepted['evaluation']=original_baseline.output_summary(self.base_records[32]);self.write(evidence/'acceptance.json',accepted)
        with self.assertRaisesRegex(ValueError,'retained rows/costs'):lab.prepare(4,32,'run-01')

    def test_closed_event_reader_uses256mib_not16mib_and_refuses_above_bound(self):
        record=lab.prepare(4,128,'run-01');self.populate(record);output=Path(record['cells'][0]['output'])
        (output/'events.jsonl').write_text('\n')
        real_digest=lab._digest;calls=[]
        def injected(path,**kwargs):
            if Path(path).name=='events.jsonl':
                calls.append(kwargs['maximum'])
                # Model a legitimate17MiB event journal without allocating it.
                self.assertGreater(kwargs['maximum'],17*1024**2)
            return real_digest(path,**kwargs)
        with patch.object(lab,'_digest',side_effect=injected):self.assertTrue(lab.generation_identity(record)['complete'])
        self.assertEqual(calls,[256*1024**2])
        def oversized(path,**kwargs):
            if Path(path).name=='events.jsonl':raise ValueError('Bounded input size exceeds maximum')
            return real_digest(path,**kwargs)
        with patch.object(lab,'_digest',side_effect=oversized),self.assertRaisesRegex(ValueError,'Bounded input size'):
            lab.summary(record)

    def test_original_baseline_raw_bytes_and_costs_are_frozen_inputs(self):
        record=lab.prepare(4,32,'run-01')
        self.assertIn('original_baseline_output/sample-1009/responses.jsonl',record['input_bindings'])
        path=Path(self.base_records[32]['cells'][0]['output'])/'responses.jsonl'
        path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'input/contract bytes changed'):lab.verify_prepared(record)


if __name__=='__main__':unittest.main()
