"""Fixed story adapter contracts; no GPU, corpus model or native child launch."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('native_story_stages', ROOT/'scripts/run_native_story_stages.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class StoryStagesTests(unittest.TestCase):
    def test_every_stage_uses_the_same_bound_deterministic_entry(self):
        self.assertIn('scripts/run_deterministic_story_child.py',adapter.SOURCES)
        for stage in adapter.STAGES:
            label='pilot' if stage in adapter.PILOTS else ('profile' if stage=='story-profile' else 'source')
            command=adapter.command(stage,Path('/caps'),Path('/owned'),label)
            self.assertEqual(command[1],str(ROOT/'scripts/run_deterministic_story_child.py'))
            self.assertEqual(command[2],'train')

    def test_fixed_stage_and_path_boundary(self):
        for stage in ('assistant-pilot', 'unknown', None):
            with self.assertRaises(ValueError): adapter.locations(stage,'cpu01')
        for name in ('../escape', '/tmp/escape', 'UpperCase', '', 'a'*49):
            with self.assertRaises(ValueError): adapter.locations('story-profile',name)
        evidence, output = adapter.locations('story-profile','cpu01')
        self.assertEqual(evidence.parent, ROOT/'experiments/reports')
        self.assertEqual(output.parent, ROOT/'outputs')

    def test_profile_exact_recipe_and_two_observers(self):
        caps = adapter.limits('story-profile',[14,16,15])['limits']
        self.assertEqual(caps['train_updates'],40)
        self.assertEqual(caps['policy_forward_positions'],40*16*1024+4*1024)
        self.assertEqual(caps['generation_calls'],192)
        self.assertEqual(caps['save_operations'],2)
        self.assertEqual(caps['restore_operations'],0)
        command = adapter.command('story-profile',Path('/caps'),Path('/owned'),'profile')
        values = dict(zip(command[3::2],command[4::2]))
        self.assertEqual(values['--total'],'40')
        self.assertEqual(values['--microbatch'],'16')
        self.assertEqual(values['--accumulation'],'1')
        self.assertEqual(values['--max-seconds'],'600')
        self.assertEqual(values['--valid-target-budget'],'655360')
        self.assertNotIn('--resume',command)
        with self.assertRaises(ValueError): adapter.command('story-profile',Path('/caps'),Path('/owned'),'resumed')

    def test_shared_recovery_cost_and_fixed_intervention(self):
        control = adapter.limits('story-control-recovery',[14,16,15])['limits']
        half = adapter.limits('story-half-lr-recovery',[14,16,15])['limits']
        self.assertEqual(control,half)
        self.assertEqual(control['model_initializations'],2)
        self.assertEqual(control['train_updates'],5)
        self.assertEqual(control['evaluation_panels'],7)
        self.assertEqual(control['save_operations'],6)
        self.assertEqual(control['restore_operations'],1)
        for stage, rate, floor in (('story-control-recovery','0.0003','0.00003'),('story-half-lr-recovery','0.00015','0.000015')):
            source = adapter.command(stage,Path('/caps'),Path('/owned'),'source')
            resumed = adapter.command(stage,Path('/caps'),Path('/owned'),'resumed')
            value = lambda command,flag:command[command.index(flag)+1]
            self.assertEqual(value(source,'--total'),'3')
            self.assertEqual(value(source,'--peak-lr'),rate)
            self.assertEqual(value(source,'--floor-lr'),floor)
            self.assertEqual(value(source,'--work-journal'),value(resumed,'--work-journal'))
            self.assertEqual(value(resumed,'--resume'),'/owned/source/update-000001.pt')
            self.assertEqual(value(resumed,'--resume-work-receipt'),'/caps/retained-update-000001-receipt.json')

    def test_changed_preparation_refuses_before_launch(self):
        prepared = {'bindings':{'original':True},'caps_identity':{'original':True}}
        with patch.object(adapter,'bindings',return_value={'changed':True}), patch.object(adapter,'_supervise') as launch:
            with self.assertRaises(ValueError): adapter.assert_current(prepared,Path('/caps'))
            launch.assert_not_called()

    def test_changed_cap_refuses_before_launch(self):
        prepared = {'bindings':{'same':True},'caps_identity':{'original':True}}
        with patch.object(adapter,'bindings',return_value={'same':True}), patch.object(adapter,'identity',return_value={'changed':True}):
            with self.assertRaises(ValueError): adapter.assert_current(prepared,Path('/caps'))

    def test_nonregular_or_oversized_input_refuses(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); path=root/'data'; path.write_bytes(b'original')
            self.assertEqual(adapter.identity(path)['bytes'],8)
            with self.assertRaises(ValueError): adapter.identity(path,maximum=7)
            with self.assertRaises(ValueError): adapter.identity(root)
            link=root/'link';link.symlink_to(path)
            with self.assertRaises(OSError): adapter.identity(link)

    def test_receipt_is_independently_frozen_before_payload_read(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);output=root/'output';evidence=root/'evidence'
            (output/'source').mkdir(parents=True);evidence.mkdir()
            # Receipt validates metadata only; missing .pt payload is intentional.
            receipt=dict(schema='dongxi-story-work-receipt-v1',payload_sha256='a'*64,
                payload_bytes=100,contract_sha256='b'*64,completed_updates=1,work_prefix={})
            (output/'source/update-000001.pt.work.json').write_text(json.dumps(receipt))
            retained=adapter.freeze_receipt(output,evidence)
            self.assertEqual(json.loads(Path(retained['path']).read_text()),receipt)
            with self.assertRaises(FileExistsError): adapter.freeze_receipt(output,evidence)

    def test_wrong_cursor_receipt_refuses(self):
        with patch.object(adapter,'read_story_receipt',return_value=dict(completed_updates=2,payload_bytes=100)):
            with self.assertRaises(ValueError): adapter.freeze_receipt(Path('/output'),Path('/evidence'))

    def test_numerical_comparison_includes_dtype_rng_and_stream(self):
        import torch
        first=dict(model=torch.tensor([1.]),optimizer={'step':3},stream={'rng':torch.tensor([1,2],dtype=torch.uint8)})
        second=dict(model=torch.tensor([1.]),optimizer={'step':3},stream={'rng':torch.tensor([1,2],dtype=torch.uint8)})
        self.assertTrue(adapter.exact(first,second))
        second['stream']['rng'][1]=3
        self.assertFalse(adapter.exact(first,second))
        self.assertFalse(adapter.exact(torch.tensor([1.]),torch.tensor([1.],dtype=torch.float64)))
        self.assertFalse(adapter.exact(1,True))

    def test_unknown_prompt_geometry_refuses(self):
        for lengths in ([],[1,2],[1,2,True],[1,2,1009]):
            with self.assertRaises(ValueError): adapter.limits('story-profile',lengths)

    def test_execution_requires_operator_scope(self):
        with patch.object(adapter,'prepare') as prepare:
            with self.assertRaises(ValueError): adapter.run('story-profile','cpu01','')
            prepare.assert_not_called()

    def test_fixed_pilot_geometry_and_fresh_start(self):
        for stage,rate in (('story-control-pilot','0.0003'),('story-half-lr-pilot','0.00015')):
            caps=adapter.limits(stage,[14,16,15])
            self.assertEqual(caps['max_journal_bytes'],64*1024**2)
            self.assertEqual(caps['limits']['training_valid_targets'],50_000_000)
            self.assertEqual(caps['limits']['model_initializations'],1)
            self.assertEqual(caps['limits']['train_updates'],14000)
            self.assertEqual(caps['limits']['policy_forward_positions'],229_376_000+36*512*1024)
            self.assertEqual(caps['limits']['evaluation_panels'],36)
            self.assertEqual(caps['limits']['save_operations'],36)
            self.assertEqual(caps['limits']['generation_calls'],36*6*256)
            argv=adapter.command(stage,Path('/caps'),Path('/owned'),'pilot')
            value=lambda key:argv[argv.index(key)+1]
            for key,expected in (('--total','14000'),('--warmup','200'),('--checkpoint-every','400'),
                    ('--valid-windows','512'),('--sample-tokens','256'),('--max-seconds','14400'),('--peak-lr',rate),('--stop-after','400')):
                self.assertEqual(value(key),expected)
            self.assertNotIn('--resume',argv)

    def test_tranche_completion_never_claims_full_schedule(self):
        complete=dict(completed_updates=400,requested_stop_reached=True,schedule_complete=False)
        self.assertTrue(adapter.completion_matches('story-control-pilot',complete))
        for changed in (dict(complete,schedule_complete=True),dict(complete,completed_updates=399),
                dict(complete,requested_stop_reached=False)):
            self.assertFalse(adapter.completion_matches('story-control-pilot',changed))
        self.assertTrue(adapter.completion_matches('story-profile',dict(completed_updates=40,requested_stop_reached=True,schedule_complete=True)))

    def test_pilots_refuse_missing_actual_parent_receipts(self):
        for prerequisites in (None,{},dict(zip(adapter.PRECONDITION_STAGES,('missing','missing','missing')))):
            with self.assertRaises((ValueError,FileNotFoundError)):
                adapter.prerequisite_bindings('story-control-pilot',prerequisites,{})
        with self.assertRaises(ValueError):
            adapter.prerequisite_bindings('story-profile',{'story-profile':'cpu01'},{})

    def test_prerequisites_bind_passed_exits_and_detect_changed_identity(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);parents={stage:'cpu01' for stage in adapter.PRECONDITION_STAGES}
            def locations(stage,run_id):return root/stage/'evidence',root/stage/'output'
            current={'original':'binding'}
            for stage in parents:
                evidence,output=locations(stage,'cpu01');evidence.mkdir(parents=True)
                profile=stage=='story-profile';label='profile' if profile else 'resumed'
                (output/label).mkdir(parents=True)
                receipt=dict(status='passed',stage=stage,bindings_after=current,distinct_native_children=True,
                    invocations=[dict(status='completed',actual_exit_code=0,minimum_sampled_available_bytes=26*adapter.GIB)]*(1 if profile else 3))
                adapter.retain(evidence/'acceptance.json',receipt)
                adapter.retain(evidence/'preparation.json',{'fixed':stage})
                adapter.retain(output/label/'completion.json',dict(completed_updates=40 if profile else 3,schedule_complete=True))
                if not profile:
                    adapter.retain(evidence/'recovery-acceptance.json',dict(status='passed',checks={'exact':True}))
                    adapter.retain(evidence/'retained-update-000001-receipt.json',{'frozen':True})
            with patch.object(adapter,'locations',side_effect=locations):
                self.assertEqual(set(adapter.prerequisite_bindings('story-control-pilot',parents,current)),set(parents))
                with self.assertRaises(ValueError):adapter.prerequisite_bindings('story-control-pilot',parents,{'changed':'binding'})

    def test_checkpoint_validates_prefix_but_keeps_later_spent_cost(self):
        import torch
        from dongxi_llms.story_work_budget import publish_story_receipt
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);root.chmod(0o700)
            caps=adapter.limits('story-control-recovery',[14,16,15])
            science={'seed':909,'actual':'authored-CPU-no-model'}
            journal=root/'work.jsonl';payload=root/'final.pt'
            ledger=adapter.WorkLedger.create(journal,limits=caps['limits'],
                contract_sha256=adapter.canonical_hash(science),max_bytes=caps['max_journal_bytes'],invocation_id='authored-cpu')
            try:
                prefix=ledger.snapshot()
                with payload.open('xb') as handle:torch.save(dict(contract=science,story_work_prefix=prefix,model=torch.tensor([1.])),handle)
                publish_story_receipt(Path(str(payload)+'.work.json'),payload,science,prefix,3)
                ticket=ledger.reserve({'train_updates':1},operation='authored-later-training')
                ledger.complete(ticket,{'train_updates':1})
                completion={'work_ledger':ledger.snapshot()}
            finally:ledger.close()
            state,verified=adapter.checkpoint(payload,journal,caps,completion)
            self.assertEqual(state['story_work_prefix'],prefix)
            self.assertEqual(verified['later_reserved']['train_updates'],1)
            forged=dict(completion['work_ledger']);forged['sequence']=999
            with self.assertRaises(ValueError):adapter.checkpoint(payload,journal,caps,{'work_ledger':forged})


if __name__ == '__main__':
    unittest.main()
