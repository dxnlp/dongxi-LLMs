"""CPU adapter/schema controls; all test token rows are explicitly authored.

These tests do not generate from course weights or supply any quality ratings.
The model-generated schema is exercised with authored CPU data only.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('native_story_eval',ROOT/'scripts/run_native_story_evaluation.py')
adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)


def fixture():
    contract=adapter.evaluation_contracts(ROOT)['story-publication-v1']
    plans=[dict(checkpoint_id=adapter.checkpoint_id(arm,update),update=update,provenance='model-generated',
        checkpoint_files={'explicit-authored-CPU-schema-control':'a'*64},interface_sha256='b'*64,
        generation_source_sha256={'explicit-authored-CPU-schema-control':'c'*64})
        for arm in adapter.ARMS for update in adapter.UPDATES]
    return contract,plans


def producer_fixture(root,*,status='completed',exit_code=0):
    """Explicit CPU metadata controls, not actual producer/model evidence."""
    evidence=root/'producer';output=root/'producer-output'
    evidence.mkdir();(evidence/'supervision-pilot').mkdir()
    bound={'training':{'explicit':'authored-CPU-control'}}
    argv=adapter.training.command('story-control-pilot',evidence,output,'pilot')
    producer=dict(stage='story-control-pilot',run_id=adapter.PILOT_RUN_ID,
        bindings=bound['training'],commands={'pilot':argv},execution_tranche=dict(
            requested_stop=400,total_horizon=14000,full_schedule_completion=False))
    producer['preparation_sha256']=adapter.canonical_hash(producer)
    launch=dict(argv=argv,preparation_sha256=producer['preparation_sha256'])
    returned=dict(schema='dongxi-native-profile-watchdog-result-v1',status=status,
        child_command=argv,child_pid=12345,actual_exit_code=exit_code,
        stop_reason=None if status=='completed' else 'external-deadline',actual_native_profile_executed=True,
        limits=dict(external_seconds=14400,reserve_bytes=25*adapter.training.GIB),
        minimum_sampled_available_bytes=30*adapter.training.GIB)
    entry=dict(event='actual-story-deterministic-entry',torch_version='explicit-CPU-metadata-control',
        deterministic_algorithms=True,cublas_workspace=':4096:8',attention_backend='SDPA MATH only',tf32=False)
    for path,value in ((evidence/'preparation.json',producer),(evidence/'launch-pilot.json',launch),
            (evidence/'returned-supervision-pilot.json',returned),(evidence/'supervision-pilot/result.json',returned)):
        path.write_text(json.dumps(value)+'\n')
    (evidence/'supervision-pilot/stdout.txt').write_text(json.dumps(entry)+'\nExplicit authored CPU fixture.\n')
    if status=='completed':
        (output/'pilot').mkdir(parents=True)
        (output/'pilot/completion.json').write_text(json.dumps(dict(completed_updates=400,
            cumulative_targets=1000000,cumulative_processed_positions=400*16*1024,
            requested_stop_reached=True,schedule_complete=False,
            stopped_for_time_budget=False,stopped_for_target_budget=False))+'\n')
    return evidence,output,bound


class NativeStoryEvaluationTests(unittest.TestCase):
    def test_fixed_ten_checkpoints_and_original_480_cells(self):
        contract,plans=fixture();adapter.validate_contract(contract)
        self.assertEqual(len(contract['items']),12)
        self.assertEqual(len(plans),10)
        self.assertEqual(len(plans)*len(contract['items'])*len(adapter.DECODING),480)
        self.assertEqual({p['update'] for p in plans},{0,400,4000,8000,14000})
        for arm,update in (('best',14000),('control',14001),('control',True)):
            with self.assertRaises(ValueError):adapter.checkpoint_id(arm,update)
        for name in ('../escape','/tmp/new','UpperCase',''):
            with self.assertRaises(ValueError):adapter.locations(name)

    def test_complete_logical_generation_and_separate_nll_caps(self):
        caps=adapter.work_contract([15]*12)['limits']
        self.assertEqual(caps['generation_sequences'],48)
        self.assertEqual(caps['generation_tokens'],12288)
        self.assertEqual(caps['multinomial_draws'],9216)
        self.assertEqual(caps['generation_positions'],48*(256*15+256*255//2))
        self.assertEqual(caps['evaluation_panels'],2)
        self.assertEqual(caps['evaluation_windows'],128)
        self.assertEqual(caps['policy_forward_calls'],8)
        self.assertEqual(caps['training_valid_targets'],0)
        self.assertEqual(caps['train_updates'],0)
        with self.assertRaises(ValueError):adapter.work_contract([769]*12)
        with self.assertRaises(ValueError):adapter.work_contract([15]*11)

    def test_greedy_model_likelihood_and_behavior_probability(self):
        import torch
        logits=torch.zeros(50257);logits[17]=2.
        token,model,behavior=adapter.token_distribution(logits,adapter.DECODING[0])
        self.assertEqual(token,17);self.assertEqual(behavior,0.)
        self.assertAlmostEqual(model,float(torch.log_softmax(logits.double(),dim=0)[17]),12)

    def test_full_support_sampling_uses_private_fixed_rng(self):
        import torch
        logits=torch.zeros(50257);logits[17]=2.
        state=torch.get_rng_state().clone()
        first=adapter.token_distribution(logits,adapter.DECODING[1],torch.Generator().manual_seed(909))
        second=adapter.token_distribution(logits,adapter.DECODING[1],torch.Generator().manual_seed(909))
        self.assertEqual(first,second);self.assertTrue(torch.equal(state,torch.get_rng_state()))
        expected=float(torch.log_softmax(logits.double()/.8,dim=0)[first[0]])
        self.assertAlmostEqual(first[2],expected,12)

    def test_nonfinite_or_unrepresentable_sampler_refuses(self):
        import torch
        for logits in (torch.ones(5),torch.full((50257,),float('nan'))):
            with self.assertRaises(ValueError):adapter.token_distribution(logits,adapter.DECODING[0])
        logits=torch.full((50257,),-1000.);logits[0]=1000.
        with self.assertRaises(ValueError):adapter.token_distribution(logits,adapter.DECODING[1],torch.Generator().manual_seed(909))

    def test_real_record_schema_requires_actual_tokens_and_stops(self):
        contract,plans=fixture();item=contract['items'][0]
        record=adapter.make_record(contract,plans[0],item,0,15,[50256],[-1.],'<|endoftext|>',
            'natural-eos',None,dict(generation_tokens=1,wall_seconds=.1,forward_positions=15,forward_calls=1))
        self.assertEqual(record['selected_likelihoods'],[-1.]);self.assertFalse(record['truncated'])
        with self.assertRaises(ValueError):adapter.make_record(contract,plans[0],item,0,15,[4],[-1.],'CPU schema control',
            'token-cap',None,dict(generation_tokens=1,wall_seconds=.1))

    def test_consumer_retains_all_planned_cells_without_creating_ratings(self):
        from dongxi_llms.story_rubric import prepare_packet,evaluate_ratings
        contract,plans=fixture();item=contract['items'][0]
        record=adapter.make_record(contract,plans[0],item,0,15,[50256],[-1.],'<|endoftext|>',
            'natural-eos',None,dict(generation_tokens=1,wall_seconds=.1))
        raters=[dict(rater_id='declared-CPU-schema-control-'+str(i),provenance='ai',
            independence_declaration='Explicit authored test declaration, no actual quality review.',shared_consultation=False) for i in (1,2)]
        comparisons=[[adapter.checkpoint_id('control',update),adapter.checkpoint_id('half-lr',update)] for update in adapter.UPDATES]
        packet,book=prepare_packet(contract,plans,[record],raters,seed=909,comparisons=comparisons)
        report=evaluate_ratings(packet,book,[],draws=800)
        self.assertEqual(len(book['expected_cells']),480)
        self.assertEqual(sum(row['record_id'] is None for row in book['expected_cells']),479)
        self.assertEqual(report['status'],'awaiting-ratings')
        self.assertTrue(all(row['aggregate_scores'] is None for row in report['cells']))

    def test_interrupted_attempt_keeps_prefix_and_unstarted_cells_missing(self):
        contract,plans=fixture();cid=plans[0]['checkpoint_id'];item=contract['items'][0]
        rid=cid+'-'+item['id']+'-d1'
        prepared=dict(contract=contract,checkpoint_plan=plans)
        events=[dict(record_id=rid,phase='start',item_id=item['id'],decoding_index=1,prompt_tokens=15,elapsed_seconds=0.),
            dict(record_id=rid,phase='entered-forward',positions=15,elapsed_seconds=.01),
            dict(record_id=rid,phase='token',token_id=4,model_logp=-3.,behavior_logp=-2.,elapsed_seconds=.02),
            dict(record_id=rid,phase='entered-forward',positions=16,elapsed_seconds=.03)]
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            with (root/'token-trace.jsonl').open('w') as handle:
                for event in events:handle.write(json.dumps(event)+'\n')
                handle.write('{"incomplete"')
            result=adapter.interrupted_records(prepared,cid,root,dict(stop_reason='external-deadline'))
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['stop_reason'],'deadline')
        self.assertEqual(result[0]['token_ids'],[4])
        self.assertEqual(result[0]['cost']['forward_positions'],15)
        self.assertEqual(result[0]['cost']['attempted_forward_positions'],31)
        self.assertEqual(result[0]['cost']['wall_seconds'],.03)

    def test_empty_record_artifact_is_measured_missing_evidence(self):
        with tempfile.TemporaryDirectory() as name:
            path=Path(name)/'records.jsonl';path.touch()
            observed=adapter.record_file_identity(path)
            self.assertEqual(observed['bytes'],0)
            self.assertEqual(observed['sha256'],hashlib.sha256(b'').hexdigest())

    def test_changed_source_refuses_before_launch(self):
        with patch.object(adapter,'source_bindings',return_value={'new':'source'}),patch.object(adapter,'_supervise') as launch:
            with self.assertRaises(ValueError):adapter.assert_current({'bindings':{'old':'source'}},Path('/unused'))
            launch.assert_not_called()

    def test_execution_scope_required_before_preparation(self):
        with patch.object(adapter,'prepare') as prepare:
            with self.assertRaises(ValueError):adapter.run('cpu01','')
            prepare.assert_not_called()

    def test_checkpoint_declared_storage_interface_and_source_are_checked(self):
        import torch
        from dataclasses import asdict
        from dongxi_llms.stories_training import baseline
        names=('stories_training.py','stories_data.py','decoder_lab.py','pretraining_lab.py',
            'story_work_budget.py','work_budget.py','run_identity.py','snapshot_io_budget.py','train_stories.py')
        sources={name:adapter._digest(str(ROOT/('scripts' if name=='train_stories.py' else 'src/dongxi_llms')/name))['sha256'] for name in names}
        contract=dict(model=asdict(baseline(1024)),data=adapter.training.DATA_HASHES['manifest.json'],device='cuda',
            torch=str(torch.__version__),attention='pytorch-sdpa-auto-causal',valid_target_budget=50_000_000,
            recipe=dict(total_updates=14000,warmup=200,peak_lr=3e-4,floor_lr=3e-5,microbatch=16,
                accumulation=1,seed=909,bf16=True,activation_checkpointing=True),implementation=sources)
        state=dict(step=0,contract=contract,story_work_prefix={'explicit':'CPU-metadata-control'})
        declaration=dict(update=0,arm='control',receipt_identity={'path':'/unused'})
        receipt=dict(contract_sha256=adapter.canonical_hash(contract),work_prefix=state['story_work_prefix'])
        with patch.object(adapter,'read_story_receipt',return_value=receipt):
            adapter.validate_checkpoint_state(state,declaration,{})
            state['step']=400
            with self.assertRaises(ValueError):adapter.validate_checkpoint_state(state,declaration,{})
            state['step']=False
            with self.assertRaises(ValueError):adapter.validate_checkpoint_state(state,declaration,{})
            state['step']=0;contract['implementation']={}
            with self.assertRaises(ValueError):adapter.validate_checkpoint_state(state,declaration,{})

    def test_declared_checkpoint_backend_observed_training_and_evaluation_are_distinct(self):
        self.assertEqual(adapter.DECLARED_TRAINING_ATTENTION,'pytorch-sdpa-auto-causal')
        self.assertEqual(adapter.EVALUATION_EXECUTION['attention_declaration'],'pytorch-sdpa-auto-causal')
        self.assertIn('not a MATH-only override',adapter.EVALUATION_EXECUTION['kernel_selection'])
        self.assertEqual(adapter.EVALUATION_EXECUTION['checkpoint_storage'],'torch.float32')
        self.assertEqual(adapter.EVALUATION_EXECUTION['forward_autocast'],'CUDA torch.bfloat16')
        self.assertFalse(adapter.EVALUATION_EXECUTION['deterministic_entry_used'])
        with tempfile.TemporaryDirectory() as name:
            evidence,output,bound=producer_fixture(Path(name))
            observed=adapter.deterministic_producer_receipts('control',evidence,output,bound)
            self.assertEqual(observed['observed_entry']['attention_backend'],'SDPA MATH only')
            self.assertEqual(observed['declared_checkpoint_attention'],'pytorch-sdpa-auto-causal')
            self.assertEqual(set(observed['files']),{'preparation','launch','returned_supervision','terminal_supervision','stdout','completion'})
            self.assertEqual(observed['actual_completion']['completed_updates'],400)
            self.assertTrue(observed['actual_completion']['requested_stop_reached'])
            self.assertFalse(observed['actual_completion']['schedule_complete'])
            self.assertFalse(observed['declared_execution_tranche']['full_schedule_completion'])

    def test_capped_parent_does_not_invent_whole_pilot_success_or_block_completed_saves(self):
        with tempfile.TemporaryDirectory() as name:
            evidence,output,bound=producer_fixture(Path(name),status='failed',exit_code=-15)
            observed=adapter.deterministic_producer_receipts('control',evidence,output,bound)
            self.assertEqual(observed['invocation']['status'],'failed')
            self.assertEqual(observed['invocation']['actual_exit_code'],-15)
            self.assertEqual(observed['invocation']['stop_reason'],'external-deadline')
            self.assertIsNone(observed['actual_completion'])

    def test_missing_terminal_receipt_changed_launch_or_nonmath_entry_refuses(self):
        for problem in ('missing','launch','entry'):
            with tempfile.TemporaryDirectory() as name:
                evidence,output,bound=producer_fixture(Path(name))
                if problem=='missing':(evidence/'supervision-pilot/result.json').unlink()
                elif problem=='launch':
                    path=evidence/'launch-pilot.json';value=json.loads(path.read_text());value['argv'].append('--unselected')
                    path.write_text(json.dumps(value))
                else:
                    path=evidence/'supervision-pilot/stdout.txt';value=json.loads(path.read_text().splitlines()[0])
                    value['attention_backend']='SDPA auto';path.write_text(json.dumps(value)+'\n')
                with self.assertRaises((ValueError,FileNotFoundError)):
                    adapter.deterministic_producer_receipts('control',evidence,output,bound)

    def test_producer_receipts_are_rechecked_before_and_after_generation(self):
        with tempfile.TemporaryDirectory() as name:
            path=Path(name)/'producer.json';path.write_text('{"explicit":"CPU control"}\n')
            expected=adapter.training.identity(path)
            prepared=dict(bindings={'explicit':'control'},files={},checkpoints={},
                producers={'control':{'files':{'preparation':expected}},'half-lr':None})
            with patch.object(adapter,'source_bindings',return_value=prepared['bindings']):
                adapter.assert_current(prepared,Path(name))
                path.write_text('{"explicit":"changed CPU control"}\n')
                with self.assertRaisesRegex(ValueError,'producer receipt'):
                    adapter.assert_current(prepared,Path(name))

    def test_producer_aggregate_reader_preserves_regular_bounds_duplicate_and_finite_checks(self):
        with tempfile.TemporaryDirectory() as name:
            path=Path(name)/'metadata.json'
            for raw in ('{"a":1,"a":2}','{"a":NaN}','[]'):
                path.write_text(raw)
                with self.assertRaises(ValueError):adapter.producer_json(path)
            path.write_text('{"a":"bounded CPU control"}')
            with patch.object(adapter,'PRODUCER_CONTROL_BYTES',8):
                with self.assertRaises(ValueError):adapter.producer_json(path)

    def test_missing_and_unreceipted_checkpoint_cells_stay_null_without_producer_claims(self):
        contract,_=fixture()
        bound={'training':{'explicit':'CPU-only source control'},
            'evaluation':{'explicit-CPU-only-generation-source':{'sha256':'c'*64}}}
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);(root/'experiments/reports').mkdir(parents=True);(root/'outputs').mkdir()
            control=root/'unexecuted-story-control-pilot'/'pilot';control.mkdir(parents=True)
            incomplete=control/'update-000400.pt';incomplete.write_text('Explicit incomplete authored CPU payload; never loaded.')
            def parents(stage,run_id):return root/(stage+'-evidence'),root/('unexecuted-'+stage)
            with patch.object(adapter,'ROOT',root),patch.object(adapter,'source_bindings',return_value=bound), \
                    patch.object(adapter,'evaluation_contracts',return_value={'story-publication-v1':contract}), \
                    patch.object(adapter.training,'locations',side_effect=parents), \
                    patch.object(adapter,'deterministic_producer_receipts',side_effect=AssertionError('No completed payload/producer claim')):
                evidence,_,prepared=adapter.prepare('explicit-cpu-control')
                self.assertTrue((evidence/'preparation.json').is_file())
                self.assertTrue(all(value is None for value in prepared['producers'].values()))
                self.assertEqual(len(prepared['checkpoints']),10)
                self.assertTrue(all(value['status']=='missing-checkpoint' and value['payload_identity'] is None
                    and value['producer_receipts_sha256'] is None for value in prepared['checkpoints'].values()))


if __name__=='__main__':unittest.main()
