"""Bounded CPU admission/archive controls; no pretrained model/GPU child."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from dongxi_llms.run_identity import canonical_hash,artifact_hashes
from dongxi_llms.snapshot_io_budget import validate_io_contract,SnapshotIOBudget,io_ledger_contract_sha256
from dongxi_llms.training_snapshot import save_snapshot
from dongxi_llms.work_budget import WorkLedger
from test_native_reasoning_baselines import tokenizer_fixture

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('native_rlvr_stages',ROOT/'scripts/run_native_rlvr_stages.py')
runner=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(runner)


class NativeRLVRStageControls(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='dongxi-native-rlvr-controls-')
        self.addCleanup(temporary.cleanup);self.root=Path(temporary.name)
        (self.root/'experiments/reports').mkdir(parents=True);(self.root/'outputs').mkdir()
        self.model=dict(model=runner.MODEL,revision=runner.REVISION,path=str(self.root/'tiny-tokenizer'),
            files={'config.json':'a'*64,'model.safetensors':'b'*64},revision_evidence='Injected fixture, not pretrained evidence')
        self.tokenizer=tokenizer_fixture(self.root/'tiny-tokenizer')
        self.tokenizer.add_tokens([str(value) for value in range(11)])
        self.tokenizer.save_pretrained(self.root/'tiny-tokenizer')
        (self.root/'tiny-tokenizer/config.json').write_text(json.dumps({'max_position_embeddings':1024}))
        source=self.root/'authored-source.py';source.write_text('authored source fixture\n')
        self.sources={'authored-source':runner._digest(str(source))}
        self.prerequisites={'authored-baseline':dict(path='/fixture/acceptance',bytes=8,sha256='d'*64)}
        for patcher in (patch.object(runner,'ROOT',self.root),
                patch.object(runner,'source_bindings',return_value=self.sources),
                patch.object(runner,'model_binding',return_value=self.model),
                patch.object(runner,'prerequisite_bindings',return_value=self.prerequisites),
                patch.object(runner,'load_local_tokenizer',return_value=self.tokenizer)):
            patcher.start();self.addCleanup(patcher.stop)

    def prepare(self,stage='recovery',group=4,run_id='run-01'):
        return runner.prepare(stage,group,run_id)

    def fresh_module(self):
        spec=importlib.util.spec_from_file_location('native_rlvr_independent_admission',
            ROOT/'scripts/run_native_rlvr_stages.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.ROOT=self.root
        return module

    def admission_fixtures(self,admission,*,recovery=False):
        observed=runner.geometry(self.model)
        for patcher in (patch.object(admission,'source_bindings',return_value=self.sources),
                patch.object(admission,'model_binding',return_value=self.model),
                patch.object(admission,'baseline_adapter',return_value=SimpleNamespace(verify_prepared=lambda record:None))):
            patcher.start();self.addCleanup(patcher.stop)
        for budget in (32,128):
            directory=self.root/'experiments/reports'/f'native-reasoning-instruct-thinking-off-cap{budget}-20261005-run-01'
            directory.mkdir()
            (directory/'acceptance.json').write_text(json.dumps(dict(status='passed',actual_exit_code=0,
                row='instruct-thinking-off',budget=budget,checks={key:True for key in admission.BASELINE_CHECKS},
                local_model_binding=self.model,evaluation={'accuracy':0})))
            contract_path=directory/'sample-contract.json'
            contract_path.write_text(json.dumps({'settings':{'thinking_mode':'disabled',
                'generation':{'input_mode':'chat','interface':observed['checkpoint_interface']}}}))
            declaration=dict(row='instruct-thinking-off',budget=budget,run_id='run-01',
                local_model_binding=self.model,source_bindings=self.sources,
                input_bindings={'contract/sample':runner._digest(str(contract_path))},
                cells=[{'contract_path':str(contract_path)}])
            declaration['preparation_sha256']=canonical_hash(declaration)
            (directory/'preparation.json').write_text(json.dumps(declaration))
        if recovery:
            record=self.prepare()
            record['prerequisite_bindings']=admission.prerequisite_bindings('recovery',4,self.model,observed)
            record['preparation_sha256']=canonical_hash({key:value for key,value in record.items() if key!='preparation_sha256'})
            directory=Path(record['locations']['evidence'])
            (directory/'preparation.json').write_text(json.dumps(record))
            (directory/'acceptance.json').write_text(json.dumps(dict(status='passed',stage='recovery',group=4,
                checks={key:True for key in admission.RECOVERY_CHECKS},local_model_binding=self.model,
                invocations=[dict(role=role,status='completed',actual_exit_code=0) for role in ('source','completed','pending')])))
        return observed

    def test_default_prepare_never_launches_or_loads_model(self):
        with (patch.object(runner,'_supervise') as supervise,patch('subprocess.Popen') as spawn,
                patch('transformers.AutoModelForCausalLM.from_pretrained') as weights):
            self.assertEqual(runner.main(['--stage','recovery','--group-size','4','--run-id','run-01']),0)
        supervise.assert_not_called();spawn.assert_not_called();weights.assert_not_called()
        record=json.loads((runner.paths('recovery',4,'run-01')['evidence']/'preparation.json').read_text())
        self.assertEqual(Path(record['locations']['evidence']).stat().st_mode & 0o777,0o700)
        self.assertEqual(Path(record['locations']['journals']).stat().st_mode & 0o777,0o700)
        self.assertTrue(all(not Path(record['locations'][role]).exists() for role in ('source','completed','pending')))
        runner.verify_prepared(record)

    def test_original_native_thinking_off_prompt_pairs_and_encoded_shapes(self):
        record=self.prepare();geometry=record['observed_geometry']
        self.assertFalse(geometry['prompt_contract']['enable_thinking'])
        self.assertEqual([row['expected'] for row in geometry['train']],[5,9,9,8])
        self.assertEqual([row['expected'] for row in geometry['evaluation']],[8,7,10,9])
        self.assertEqual(geometry['train'][0]['prompt'],'Return only the integer answer. 2 + 3 =')
        self.assertEqual(len(geometry['train']),4);self.assertEqual(len(geometry['evaluation']),4)
        self.assertTrue(all(row['prompt_ids'] for row in geometry['train']+geometry['evaluation']))

    def test_work_caps_include_save_restore_and_whole_pool_validation(self):
        record=self.prepare();caps=record['allowances']['work'];lifecycle=record['allowances']['lifecycle']
        self.assertEqual(set(caps),set(runner.rlvr.BUDGET_KEYS));self.assertEqual(len(caps),23)
        self.assertEqual(caps['train_updates'],4);self.assertEqual(caps['collections'],3)
        self.assertEqual(caps['recovery_validation_operations'],16)
        self.assertEqual(caps['recovery_validation_calls'],9)
        self.assertEqual(caps['recovery_multinomial_draws'],22*4*16)
        self.assertEqual(caps['evaluation_examples'],24)
        self.assertEqual(lifecycle['snapshot_saves'],10)
        self.assertEqual(lifecycle['snapshot_loads'],2);self.assertEqual(lifecycle['snapshot_inspects'],2)

    def test_pilot_original16_and_unequal_group_token_costs(self):
        records=[self.prepare('pilot',group) for group in (4,8)]
        a,b=[record['allowances']['work'] for record in records]
        self.assertEqual(a['train_updates'],16);self.assertEqual(b['train_updates'],16)
        self.assertEqual(a['generated_slots'],4096);self.assertEqual(b['generated_slots'],8192)
        self.assertEqual(a['recovery_validation_operations'],49)
        self.assertEqual(a['recovery_validation_calls'],32)
        self.assertEqual(a['recovery_multinomial_draws'],288*4*64)
        for record in records:
            self.assertEqual(record['recipe']['beta'],.02)
            self.assertEqual(record['recipe']['external_seconds_each'],1800)
            self.assertEqual(record['recipe']['seed'],2323)
            self.assertEqual(record['recipe']['lr'],1e-6)
            self.assertEqual(record['allowances']['io']['limits']['snapshot_save_operations'],33)

    def test_snapshot_envelope_and_aggregate_io_caps_cover_real_bf16_state(self):
        record=self.prepare();contract=validate_io_contract(record['allowances']['io'])
        self.assertEqual(contract['envelope']['max_payload_bytes'],8*1024**3)
        self.assertEqual(contract['envelope']['max_tensor_bytes'],8*1024**3)
        self.assertEqual(contract['envelope']['max_tensor_elements'],4_000_000_000)
        self.assertEqual(contract['limits']['snapshot_hash_bytes'],14*8*1024**3)
        self.assertEqual(contract['limits']['snapshot_clone_bytes'],10*8*1024**3)
        self.assertEqual(contract['limits']['snapshot_tensor_elements'],12*4_000_000_000)

    def test_fixed_commands_share_physical_journals_and_resume_only_original1(self):
        record=self.prepare();original=record['fresh_argv']
        resume=dict(phase='pending',completed_updates=1,
            path=str(runner.paths('recovery',4,'run-01')['source']/'snapshots/pending-000001.pt'),
            payload_sha256='a'*64,payload_bytes=12345)
        pending=runner.fixed_argv(record,'pending',resume)
        self.assertEqual(original[1:3],['-m','dongxi_llms.qwen_rlvr_lab'])
        for argument in ('--work-journal','--snapshot-io-ledger'):
            self.assertEqual(original[original.index(argument)+1],pending[pending.index(argument)+1])
        self.assertEqual(pending[pending.index('--resume-bytes')+1],'12345')
        self.assertEqual(original[original.index('--updates')+1],'2')
        self.assertEqual(original[original.index('--max-new-tokens')+1],'16')
        with self.assertRaises(ValueError):runner.fixed_argv(record,'pending',dict(resume,completed_updates=0))
        with self.assertRaises(ValueError):runner.fixed_argv(record,'completed',resume)

    def test_unknown_selectors_existing_targets_and_disk_shortage_refuse(self):
        for args in [('sweep',4,'run-01'),('pilot',3,'run-01'),('recovery',True,'run-01'),('pilot',4,'../escape')]:
            with self.subTest(args=args),self.assertRaises(ValueError):runner.prepare(*args)
        record=self.prepare()
        with self.assertRaises(FileExistsError):self.prepare()
        self.assertTrue((Path(record['locations']['evidence'])/'preparation.json').exists())
        with patch.object(runner.shutil,'disk_usage',return_value=type('Usage',(),{'free':0})()),self.assertRaises(RuntimeError):
            self.prepare('pilot',4)

    def test_changed_sources_inputs_and_rehashed_arbitrary_command_refuse(self):
        record=self.prepare()
        with patch.object(runner,'source_bindings',return_value={'changed':1}),self.assertRaisesRegex(ValueError,'artifacts changed'):
            runner.verify_prepared(record)
        changed=deepcopy(record);changed['fresh_argv']=['/usr/bin/true']
        changed['preparation_sha256']=canonical_hash({key:value for key,value in changed.items() if key!='preparation_sha256'})
        with self.assertRaisesRegex(ValueError,'Closed stage'):runner.verify_prepared(changed)
        changed=deepcopy(record);changed['recipe']['seed']+=1
        changed['preparation_sha256']=canonical_hash({key:value for key,value in changed.items() if key!='preparation_sha256'})
        with self.assertRaisesRegex(ValueError,'recipe differs'):runner.verify_prepared(changed)
        path=Path(record['input_bindings']['work_caps']['path']);path.write_text(path.read_text()+' ')
        with patch.object(runner,'_supervise') as launch,self.assertRaisesRegex(ValueError,'input changed'):
            runner.execute(record,'Authored CPU refusal control; never GPU execution')
        launch.assert_not_called()

    def test_actual_baseline_same_bytes_and_negative_quality_are_admission_requirements(self):
        admission=self.fresh_module()
        observed=self.admission_fixtures(admission)
        bindings=admission.prerequisite_bindings('recovery',4,self.model,observed)
        self.assertEqual(len(bindings),3)
        changed=deepcopy(self.model);changed['files']['model.safetensors']='e'*64
        with self.assertRaisesRegex(ValueError,'same-byte'):admission.prerequisite_bindings('recovery',4,changed,observed)
        with self.assertRaises(FileNotFoundError):admission.prerequisite_bindings('pilot',4,self.model,observed)

    def test_pilot_needs_both_baseline_caps_and_matching_actual_recovery(self):
        admission=self.fresh_module()
        observed=self.admission_fixtures(admission,recovery=True)
        directory=self.root/'experiments/reports'/'native-rlvr-g4-recovery-20261005-run-01'
        path=directory/'acceptance.json'
        self.assertEqual(len(admission.prerequisite_bindings('pilot',4,self.model,observed)),8)
        acceptance=json.loads(path.read_text());acceptance['group']=8;path.write_text(json.dumps(acceptance))
        with self.assertRaisesRegex(ValueError,'group-matched'):admission.prerequisite_bindings('pilot',4,self.model,observed)

    def test_baseline_frozen_contract_and_native_interface_drift_refuse(self):
        admission=self.fresh_module();observed=self.admission_fixtures(admission)
        changed=deepcopy(observed);changed['checkpoint_interface']['template_sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'tokenizer/template'):admission.prerequisite_bindings('recovery',4,self.model,changed)
        directory=self.root/'experiments/reports'/'native-reasoning-instruct-thinking-off-cap32-20261005-run-01'
        (directory/'sample-contract.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'source/input bytes'):admission.prerequisite_bindings('recovery',4,self.model,observed)

    def test_recovery_recipe_geometry_and_input_drift_refuse_pilot(self):
        admission=self.fresh_module();observed=self.admission_fixtures(admission,recovery=True)
        changed=deepcopy(observed);changed['train'][0]['expected']+=1
        with self.assertRaisesRegex(ValueError,'differs from current pilot'):admission.prerequisite_bindings('pilot',4,self.model,changed)
        directory=self.root/'experiments/reports'/'native-rlvr-g4-recovery-20261005-run-01'
        path=directory/'preparation.json';record=json.loads(path.read_text())
        record['recipe']['seed']+=1
        record['preparation_sha256']=canonical_hash({key:value for key,value in record.items() if key!='preparation_sha256'})
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError,'recipe differs'):admission.prerequisite_bindings('pilot',4,self.model,observed)
        record['recipe']['seed']-=1
        record['preparation_sha256']=canonical_hash({key:value for key,value in record.items() if key!='preparation_sha256'})
        path.write_text(json.dumps(record));(directory/'work-caps.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'source/input bytes'):admission.prerequisite_bindings('pilot',4,self.model,observed)

    def test_recovery_missing_actual_resume_exit_cannot_admit_pilot(self):
        admission=self.fresh_module();observed=self.admission_fixtures(admission,recovery=True)
        path=self.root/'experiments/reports'/'native-rlvr-g4-recovery-20261005-run-01'/'acceptance.json'
        acceptance=json.loads(path.read_text());acceptance['invocations'][2]['actual_exit_code']=7
        path.write_text(json.dumps(acceptance))
        with self.assertRaisesRegex(ValueError,'exit receipts'):admission.prerequisite_bindings('pilot',4,self.model,observed)

    def test_symbolic_native_tensor_comparison_preserves_rng_and_excludes_physical_prefix(self):
        state=dict(policy={'w':torch.tensor([1.,2.],dtype=torch.bfloat16)},
            reference={'w':torch.tensor([1.,2.],dtype=torch.bfloat16)},
            optimizer={'state':{0:{'step':torch.tensor(2.),'exp_avg':torch.zeros(2,dtype=torch.bfloat16)}}},
            loop_contract={'updates':2},cursor=2,history=[{'update':1},{'update':2}],pending=None,
            rng={'rollout':torch.tensor([1,2,3],dtype=torch.uint8),'python':(3,(1,2),None)},
            work_ledger={'reserved':100})
        paths=[self.root/(name+'.pt') for name in ('source','fresh','changed')]
        for index,path in enumerate(paths):
            value=deepcopy(state);value['work_ledger']['reserved']+=index*10
            if index==2:value['rng']['rollout'][0]=2
            torch.save(dict(phase='completed',completed_updates=2,state=value,parent_invocation=str(index)),path)
        a,b,c=map(runner.symbolic_snapshot,paths)
        self.assertEqual(a,b);self.assertNotEqual(a,c)
        self.assertIn('optimizer',a['components']);self.assertIn('rng',a['components'])
        self.assertNotIn('work_ledger',a['components'])

    def test_symbolic_reader_rejects_unknown_pickle_globals(self):
        path=self.root/'unsupported.pt'
        torch.save(dict(state={'unsupported':Path('/never-executed')},phase='completed',completed_updates=0),path)
        with self.assertRaisesRegex(ValueError,'Unsupported pickle global'):runner.symbolic_snapshot(path)

    def final_pilot_fixture(self,cursor=16):
        record=self.prepare('pilot',4);output=Path(record['locations']['pilot']);(output/'snapshots').mkdir(parents=True)
        contract={'authored_cpu_scientific_contract':True,'updates':16}
        (output/'recovery-contract.json').write_text(json.dumps(contract))
        io_contract=record['allowances']['io'];ledger=WorkLedger.create(self.root/'final-io.jsonl',limits=io_contract['limits'],
            contract_sha256=io_ledger_contract_sha256(io_contract,canonical_hash(contract)),
            max_bytes=io_contract['max_journal_bytes'],invocation_id='authored-final16')
        self.addCleanup(ledger.close);budget=SnapshotIOBudget(ledger,contract=io_contract,scientific_contract_sha256=canonical_hash(contract))
        state=dict(policy={'w':torch.tensor([1.],dtype=torch.bfloat16)},reference={'w':torch.tensor([1.],dtype=torch.bfloat16)},
            optimizer={'state':{0:{'step':torch.tensor(16.)}}},cursor=cursor,rng={'rollout':torch.tensor([1],dtype=torch.uint8)},
            history=[{'update':16}],pending=None)
        path=output/'snapshots/completed-000016.pt'
        header=save_snapshot(path,contract=contract,state=state,completed_updates=16,phase='completed',
            parent_invocation='authored-final16',max_bytes=runner.ENVELOPE['max_payload_bytes'],io_budget=budget,
            work_receipt_path=Path(str(path)+'.work.json'))
        report=dict(status='completed',committed_completed_updates=16,
            latest_durable_snapshot=dict(path=str(path),**header,work_receipt_path=str(path)+'.work.json'))
        (output/'report.json').write_text(json.dumps(report));policy=output/'policy';self.tokenizer.save_pretrained(policy)
        (policy/'config.json').write_text('{}');(policy/'model.safetensors').write_bytes(b'authored-export-not-pretrained')
        genealogy=dict(kind='full-HF-model',objective='course-response-mean-GRPO',upstream_revision_metadata=runner.REVISION,
            parent_local_source_hashes=record['local_model_binding']['files'],checkpoint_interface=record['observed_geometry']['checkpoint_interface'],
            template_sha256=record['observed_geometry']['prompt_contract']['template_sha256'])
        (policy/'course-genealogy.json').write_text(json.dumps(genealogy));return record,report,path

    def test_pilot_acceptance_binds_actual_export_and_completed16_snapshot_receipts(self):
        record,report,path=self.final_pilot_fixture();bound=runner.pilot_export_binding(record,report)
        self.assertEqual(bound['files'],artifact_hashes(Path(record['locations']['pilot'])/'policy'))
        self.assertEqual(bound['completed_updates'],16);self.assertEqual(bound['final_snapshot']['header']['completed_updates'],16)
        self.assertIn('policy',bound['final_snapshot']['scientific_state']['components'])
        self.assertEqual(bound['report_binding'],runner._digest(str(Path(record['locations']['pilot'])/'report.json')))
        # Changed bytes cannot preserve the accepted binding, even before evaluation preparation.
        (Path(bound['path'])/'model.safetensors').write_bytes(b'swapped-after-acceptance')
        self.assertNotEqual(bound,runner.pilot_export_binding(record,report))
        path.write_bytes(path.read_bytes()+b'changed-payload')
        with self.assertRaisesRegex(ValueError,'payload bytes differ'):runner.pilot_export_binding(record,report)

    def test_pilot_binding_rejects_inconsistent_final_report_and_actual_cursor(self):
        record,report,path=self.final_pilot_fixture(cursor=15)
        with self.assertRaisesRegex(ValueError,'numerical policy'):runner.pilot_export_binding(record,report)
        report['latest_durable_snapshot']['payload_sha256']='e'*64
        with self.assertRaisesRegex(ValueError,'report/marker/receipt'):runner.pilot_export_binding(record,report)


if __name__=='__main__':unittest.main()
