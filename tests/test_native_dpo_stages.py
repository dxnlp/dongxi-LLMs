"""Authored CPU stage controls; native model children and GPU are never called."""
from copy import deepcopy
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from dongxi_llms.run_identity import artifact_hashes,canonical_hash
from dongxi_llms.snapshot_io_budget import validate_io_contract
from dongxi_llms.snapshot_io_budget import SnapshotIOBudget,io_ledger_contract_sha256
from dongxi_llms.work_budget import WorkLedger
from dongxi_llms.artifact_budget import ArtifactBudget
from dongxi_llms.training_snapshot import save_snapshot
from dongxi_llms.batched_cache_lab import digest

ROOT = Path(__file__).resolve().parents[1]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

lab=module('native_dpo_stages',ROOT/'scripts/run_native_dpo_stages.py')
native=module('native_dpo_fixture_helpers',ROOT/'scripts/run_chapter11_spark_dpo.py')
cpu8=module('native_dpo_cpu8_entry',ROOT/'scripts/run_native_dpo_cpu8_child.py')


class NativeDPOStageTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        for directory in ('outputs','experiments/reports','fixtures/chapter11'):
            (self.root/directory).mkdir(parents=True)
        for name in lab.SOURCES:
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('authored source fixture\n')
        for name,count in (('train',8),('validation',4),('evaluation',4)):
            (self.root/'fixtures/chapter11'/(name+'.jsonl')).write_text(''.join(json.dumps({'id':f'{name}-{i}'})+'\n' for i in range(count)))
        self.parent=self.root/'outputs'/lab.PARENT_STEM/'policy';self.parent.mkdir(parents=True)
        self.write(self.parent/'config.json',dict(tie_word_embeddings=True))
        self.write(self.parent/'course-genealogy.json',dict(kind='full-HF-model',base_model=lab.MODEL,base_revision=lab.REVISION))
        header=json.dumps({'model.embed_tokens.weight':dict(dtype='BF16',shape=[2,2],data_offsets=[0,8])}).encode()
        (self.parent/'model.safetensors').write_bytes(struct.pack('<Q',len(header))+header+b'fixture!')
        self.write(self.root/'experiments/reports'/lab.PARENT_STEM/'acceptance.json',dict(status='passed',checks={'actual400':True},
            result={'updates':400},exported_policy={'path':str(self.parent),'files':artifact_hashes(self.parent)}))
        self.lock=self.root/'spark.lock';self.lock.write_text('authored selected lock\n')
        self.tokenizer=self.root/'tokenizer';self.tokenizer.mkdir();self.write(self.tokenizer/'tokenizer_config.json',{})
        zero=dict.fromkeys(native.BUDGET_KEYS,0)
        validation=deepcopy(zero);validation.update(valid_targets=10,logical_sequence_tokens=40,
            policy_forward_calls=8,reference_forward_calls=8,policy_forward_positions=36,reference_forward_positions=36,
            evaluation_calls=16,evaluation_positions=72)
        self.geometry=dict(interface={'authored':True},encoded_train_sha256='0'*64,encoded_validation_sha256='1'*64,
            prefix_sha256='2'*64,prefix_lengths=[29,30,31,32],
            one_update=dict(train_updates=1,sampled_examples=4,sampler_draws=4,valid_targets=40,
                logical_sequence_tokens=272,policy_forward_calls=8,reference_forward_calls=8,
                policy_forward_positions=264,reference_forward_positions=264),
            one_validation=validation,one_generation=native.generation_upper([[1]*n for n in (29,30,31,32)],64),
            validation_layout=dict(updates=100,accumulation=4,sample_work=[{}],
                policy_shapes={'weight':{'shape':[2,2],'dtype':'torch.float32'}},
                optimizer_shapes=[{'shape':[2,2],'dtype':'torch.float32'}],torch_rng_bytes=16,cuda_rng_count=1,cuda_rng_bytes=[16]),
            tokenizer_snapshot=dict(path=str(self.tokenizer),files=artifact_hashes(self.tokenizer)),
            weights_loaded=False,cuda_observed=False)
        replacements=dict(ROOT=self.root,INTERPRETER=sys.executable,ENVIRONMENT_LOCK=str(self.lock))
        for name,value in replacements.items():
            replacement=patch.object(lab,name,value);replacement.start();self.addCleanup(replacement.stop)
        for name,value in (('native_runner',native),('observed_geometry',self.geometry)):
            replacement=patch.object(lab,name,return_value=value);replacement.start();self.addCleanup(replacement.stop)
        disk=patch.object(lab.shutil,'disk_usage',return_value=SimpleNamespace(free=181*lab.GIB));self.disk=disk.start();self.addCleanup(disk.stop)
        receipt=patch.object(lab,'validate_work_receipt',side_effect=lambda value:value);receipt.start();self.addCleanup(receipt.stop)
        child=patch.object(lab,'_supervise');self.child=child.start();self.addCleanup(child.stop)

    def write(self,path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n');return path

    def evidence(self,record,name):return json.loads((Path(record['evidence'])/name).read_text())

    def replay_gate(self):
        record=lab.prepare('replay',lab.REPLAY_RUN_ID)
        witnesses={}
        hash_witnesses={}
        for role in ('clean','source','resumed','cpu-comparison'):
            self.fake_witness(record,role)
            witnesses[role]=lab.retained_cpu_witness(record['evidence'],role,comparison=role=='cpu-comparison')
            hash_witnesses[role]=lab.retained_hash_witness(record['evidence'],role,comparison=role=='cpu-comparison')
        self.write(Path(record['evidence'])/'acceptance.json',dict(status='passed',
            checks={key:True for key in lab.REPLAY_CHECKS},parent_binding=record['parent_binding'],
            invocations=[dict(role=role,result=dict(status='completed',actual_exit_code=0),cpu_thread_witness=witnesses[role],
                snapshot_hash_witness=hash_witnesses[role])
                for role in ('clean','source','resumed','cpu-comparison')]))
        return record

    def fake_witness(self,record,role,**changes):
        comparison=role=='cpu-comparison';name='comparison' if comparison else role
        target='scripts/run_native_dpo_stages.py' if comparison else 'scripts/run_chapter11_spark_dpo.py'
        witness=dict(schema='dongxi-fixed-native-dpo-cpu8-witness-v1',target=str(self.root/target),**lab.CPU_SETTINGS)
        witness.update(changes);path=Path(record['evidence'])/('supervision-'+name)/'stdout.txt';path.parent.mkdir(exist_ok=True)
        hashed=dict(schema='dongxi-snapshot-hash-runtime-v1',hash_workers=lab.HASH_WORKERS)
        hash_rows=[dict(hashed,role=item) for item in ('clean','source','resumed')] if comparison else [hashed]
        path.write_text(lab.CPU_WITNESS_PREFIX+json.dumps(witness)+'\n'+''.join(
            lab.HASH_WITNESS_PREFIX+json.dumps(row)+'\n' for row in hash_rows));return path

    def fake_child(self,record,*,fail_role=None,drift=False):
        places=lab.locations(record['mode'],record['run_id'])
        def child(argv,output,**kwargs):
            if '--comparison-child' in argv:
                self.fake_witness(record,'cpu-comparison')
                self.write(Path(record['evidence'])/'comparison.json',dict(status='passed',checks={'equal_numerical_components':True},results={}))
                return dict(status='completed',actual_exit_code=0,child_seconds=.1)
            role=next(role for role,path in places['outputs'].items() if str(path)==argv[argv.index('--output')+1])
            if role==fail_role:return dict(status='failed',actual_exit_code=-15,child_seconds=.1)
            self.fake_witness(record,role)
            path=places['outputs'][role];path.mkdir()
            group=('clean' if role=='clean' else 'shared') if record['mode']=='replay' else 'pilot'
            artifacts=places['artifacts'][group];artifacts.mkdir(exist_ok=True)
            namespace={'clean':'a','source':'b','resumed':'c','pilot':'d'}[role]*32
            endpoint=record['limits']['updates']
            cursors=[0,1,2] if role in ('clean','source') else [1,2] if role=='resumed' else [0,20,40,60,80,100]
            for cursor in cursors:
                payload=artifacts/f'{namespace}-completed-{cursor:06d}.pt';payload.write_bytes(b'authored payload')
                header=dict(schema_version=2,payload_sha256=hashlib.sha256(payload.read_bytes()).hexdigest(),
                    payload_bytes=payload.stat().st_size,contract_sha256='0'*64,phase='completed',completed_updates=cursor)
                self.write(str(payload)+'.commit.json',header);self.write(str(payload)+'.work.json',{'snapshot':header})
            latest=dict(path=str(artifacts/f'{namespace}-completed-{endpoint:06d}.pt'),completed_updates=endpoint)
            result=dict(completed_recovery=latest,snapshot_artifact_ledger={'authored_prefix':role},
                cumulative_work_ledger=dict(open_tickets={},failed_tickets={}),snapshot_io_ledger=dict(open_tickets={},failed_tickets={}))
            self.write(path/'result.json',result);self.write(path/'recovery-contract.json',{'authored_science':True,'updates':endpoint})
            rows=[dict(update=i,loss=.5,margin=.1,gradient_norm=.2,indices=[1,2,3,4],work={'authored':4})
                for i in (range(2,3) if role=='resumed' else range(1,endpoint+1))]
            (path/'metrics.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            policy=path/'policy';policy.mkdir();self.write(policy/'config.json',{});(policy/'model.safetensors').write_bytes(b'exported authored weights')
            if drift:(self.root/lab.SOURCES[0]).write_text('drifted source\n')
            return dict(status='completed',actual_exit_code=0,child_seconds=.1)
        self.child.side_effect=child

    def test_three_replay_commands_and_two_journal_domains(self):
        record=lab.prepare('replay','run-01')
        commands=record['commands'];self.assertEqual(set(commands),{'clean','source','resumed'})
        value=lambda role,flag:commands[role][commands[role].index(flag)+1]
        self.assertEqual(value('source','--work-journal'),value('resumed','--work-journal'))
        self.assertNotEqual(value('clean','--work-journal'),value('source','--work-journal'))
        self.assertEqual(value('source','--snapshot-artifact-root'),value('resumed','--snapshot-artifact-root'))
        self.assertEqual(value('source','--updates'),'2');self.assertEqual(value('source','--checkpoint-every'),'1')
        self.assertNotIn('--resume',commands['source']);self.assertNotIn('--allow-download',commands['source'])
        self.assertEqual(record['limits']['external_seconds'],600);self.child.assert_not_called()
        self.assertEqual(record['cpu_settings'],dict(omp_num_threads='8',torch_num_threads=8,torch_num_interop_threads=1))
        self.assertTrue(all(command[1]==str(self.root/lab.CPU_ENTRY) for command in commands.values()))
        self.assertEqual(record['comparison_command'][1],str(self.root/lab.CPU_ENTRY))
        self.assertEqual(record['snapshot_hash_workers'],4)
        self.assertTrue(all(value(role,'--snapshot-hash-workers')=='4' for role in commands))

    def test_exact19_caps_replayed_cost_and_extra_cpu_io(self):
        limits=lab.allowances('replay',self.geometry)
        self.assertEqual(set(limits['work_caps']),set(native.BUDGET_KEYS))
        self.assertEqual(limits['work_caps']['train_updates'],3)
        self.assertEqual(limits['work_caps']['sampled_examples'],12)
        self.assertEqual(limits['work_caps']['generation_calls'],16)
        self.assertEqual(limits['work_caps']['generation_tokens'],1024)
        self.assertEqual(limits['work_caps']['recovery_validation_operations'],7)
        self.assertEqual(limits['work_caps']['recovery_history_rows'],8)
        io=validate_io_contract(limits['snapshot_io_contract'])
        self.assertEqual(io['limits']['snapshot_save_operations'],5)
        self.assertEqual(io['limits']['snapshot_inspect_operations'],3)
        self.assertEqual(io['limits']['snapshot_load_operations'],3)
        self.assertEqual(limits['artifact_max_bytes'],81*lab.GIB)

    def test_actual_parent_export_drift_refuses(self):
        (self.parent/'model.safetensors').write_bytes(b'changed bytes')
        with self.assertRaisesRegex(ValueError,'full400'):lab.prepare('replay','run-01')
        self.child.assert_not_called()

    def test_pilot_requires_actual_replay_gate(self):
        with self.assertRaises(FileNotFoundError):lab.prepare('pilot','run-01')
        self.child.assert_not_called()

    def test_pilot100_caps_with_matching_actual_parent_gate(self):
        self.replay_gate()
        record=lab.prepare('pilot','run-01');self.fake_child(record)
        self.assertEqual(record['limits']['work_caps']['train_updates'],100)
        self.assertEqual(record['limits']['snapshot_io_contract']['limits']['snapshot_save_operations'],6)
        self.assertEqual(lab.execute(record,'goal authorized fixed DPO100'),0)
        self.assertEqual(self.child.call_args.kwargs['native_stage'],'preference-pilot')
        self.assertEqual(self.child.call_args.kwargs['seconds'],1800)
        self.assertIn(lab.REPLAY_RUN_ID,record['input_bindings']['actual_replay_preparation']['path'])

    def test_pilot_requires_current_replay_preparation_source_and_data(self):
        self.replay_gate()
        (self.root/'fixtures/chapter11/train.jsonl').write_text('changed original fixture bytes\n')
        with self.assertRaisesRegex(ValueError,'input role changed'):lab.prepare('pilot','run-01')
        self.child.assert_not_called()

    def test_pilot_requires_current_replay_source(self):
        self.replay_gate()
        (self.root/lab.SOURCES[0]).write_text('changed executable source\n')
        with self.assertRaisesRegex(ValueError,'source changed'):lab.prepare('pilot','run-01')
        self.child.assert_not_called()

    def test_pilot_requires_same_reencoded_interface_and_geometry(self):
        self.replay_gate()
        self.geometry['interface']={'different_template':True}
        self.geometry['encoded_train_sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'encoded geometry changed'):lab.prepare('pilot','run-01')
        self.child.assert_not_called()

    def test_pilot_binds_replay_preparation_after_gate(self):
        replay=self.replay_gate();record=lab.prepare('pilot','run-01')
        self.assertIn('actual_replay_preparation',record['input_bindings'])
        path=Path(replay['evidence'])/'preparation.json'
        path.write_text(path.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'input role changed'):lab.verify_prepared(record)

    def test_pilot_requires_fixedrun03_and_current_actual_cpu8_witnesses(self):
        old=lab.prepare('replay','run-01')
        self.write(Path(old['evidence'])/'acceptance.json',dict(status='passed',checks={key:True for key in lab.REPLAY_CHECKS}))
        with self.assertRaises(FileNotFoundError):lab.prepare('pilot','run-01')
        self.replay_gate();self.fake_witness({'evidence':str(lab.locations('replay',lab.REPLAY_RUN_ID)['evidence'])},'resumed',torch_num_threads=96)
        with self.assertRaisesRegex(ValueError,'thread witness differs'):lab.prepare('pilot','run-02')

    def test_thread_settings_and_comparison_command_cannot_be_rehashed_to_different_values(self):
        record=lab.prepare('replay','run-02')
        for field in ('cpu_settings','comparison_command','snapshot_hash_workers'):
            changed=deepcopy(record)
            if field=='cpu_settings':changed[field]['torch_num_threads']=16
            elif field=='comparison_command':changed[field][1]=str(self.root/'scripts/run_native_dpo_stages.py')
            else:changed[field]=1
            changed['preparation_sha256']=canonical_hash({k:v for k,v in changed.items() if k!='preparation_sha256'})
            with self.assertRaisesRegex(ValueError,'closed fixed'):lab.verify_prepared(changed)

    def test_successful_exit_without_actual_thread_witness_never_passes(self):
        record=lab.prepare('replay','run-02');self.fake_child(record)
        wrapped=self.child.side_effect
        def altered(*args,**kwargs):
            result=wrapped(*args,**kwargs)
            self.fake_witness(record,'clean',torch_num_threads=96)
            return result
        self.child.side_effect=altered
        with self.assertRaisesRegex(ValueError,'thread witness differs'):lab.execute(record,'Goal authorized authored CPU8 refusal control')
        self.assertEqual(self.child.call_count,1);self.assertTrue((Path(record['evidence'])/'returned-supervision-clean.json').exists())
        self.assertFalse((Path(record['evidence'])/'acceptance.json').exists())

    def test_direct_comparison_cannot_validate_state_without_actual_cpu8_entry(self):
        fake=SimpleNamespace(get_num_threads=lambda:96,get_num_interop_threads=lambda:1)
        with (patch.dict(sys.modules,{'torch':fake}),patch.dict(os.environ,{'OMP_NUM_THREADS':'8'}),
                patch.object(lab,'verify_prepared') as verify,patch.object(lab,'comparison_digest') as compare,
                self.assertRaisesRegex(ValueError,'before comparison state validation')):
            lab.comparison_child('replay','run-02')
        verify.assert_not_called();compare.assert_not_called()

    def test_successful_exit_without_actual_parallel_hash_witness_never_passes(self):
        record=lab.prepare('replay',lab.REPLAY_RUN_ID);self.fake_child(record)
        wrapped=self.child.side_effect
        def altered(*args,**kwargs):
            result=wrapped(*args,**kwargs)
            path=Path(record['evidence'])/'supervision-clean'/'stdout.txt'
            path.write_text(path.read_text().replace('"hash_workers": 4','"hash_workers": 1'))
            return result
        self.child.side_effect=altered
        with self.assertRaisesRegex(ValueError,'hash worker witness differs'):
            lab.execute(record,'Goal authorized authored parallel-hash refusal control')
        self.assertEqual(self.child.call_count,1)
        self.assertTrue((Path(record['evidence'])/'returned-supervision-clean.json').exists())
        self.assertFalse((Path(record['evidence'])/'acceptance.json').exists())

    def test_pilot_requires_retained_actual_hash_witness_unchanged(self):
        replay=self.replay_gate()
        path=Path(replay['evidence'])/'snapshot-hash-witness-resumed.json'
        value=json.loads(path.read_text());value['actual'][0]['hash_workers']=1
        self.write(path,value)
        with self.assertRaisesRegex(ValueError,'complete-file hash witnesses differ'):
            lab.prepare('pilot','run-01')
        self.child.assert_not_called()

    def test_comparison_hash_witness_requires_all_three_roles_in_order(self):
        record=lab.prepare('replay',lab.REPLAY_RUN_ID)
        path=self.fake_witness(record,'cpu-comparison')
        rows=path.read_text().splitlines();path.write_text('\n'.join([rows[0],rows[1],rows[3],rows[2]])+'\n')
        with self.assertRaisesRegex(ValueError,'hash worker witness differs'):
            lab.hash_witness_binding(record['evidence'],'cpu-comparison',comparison=True)

    def test_replay_execution_retains_three_children_independent_metadata(self):
        record=lab.prepare('replay','run-01');self.fake_child(record)
        self.assertEqual(lab.execute(record,'goal authorized fixed DPO2 replay'),0)
        self.assertEqual(self.child.call_count,4)
        resumed=self.evidence(record,'launch-resumed.json')['argv']
        self.assertIn('--resume',resumed)
        self.assertIn('--resume-contract',resumed)
        self.assertIn('--snapshot-artifact-receipt',resumed)
        self.assertIn('--resume-io-receipt',resumed)
        self.assertTrue(self.evidence(record,'acceptance.json')['checks']['exact_numerical_metric_tail'])
        self.assertEqual(Path(resumed[resumed.index('--resume-contract')+1]).parent,Path(record['evidence']))
        self.assertEqual(self.child.call_args.kwargs['seconds'],600)

    def test_failure_stops_without_refilling_or_launching_resume(self):
        record=lab.prepare('replay','run-01');self.fake_child(record,fail_role='source')
        with self.assertRaisesRegex(RuntimeError,'child failed'):lab.execute(record,'goal authorized fixed DPO2 replay')
        self.assertEqual(self.child.call_count,2)
        failed=self.evidence(record,'failure.json');self.assertEqual(len(failed['invocations']),2)
        self.assertEqual(self.evidence(record,'returned-supervision-source.json')['actual_exit_code'],-15)
        self.assertFalse((Path(record['evidence'])/'launch-resumed.json').exists())

    def test_postchild_source_drift_retains_actual_receipt(self):
        record=lab.prepare('replay','run-01');self.fake_child(record,drift=True)
        with self.assertRaisesRegex(ValueError,'source changed'):lab.execute(record,'goal authorized fixed DPO2 replay')
        self.assertTrue((Path(record['evidence'])/'returned-supervision-clean.json').is_file())
        self.assertEqual(self.child.call_count,1)

    def test_existing_targets_and_escape_identifiers_refuse(self):
        for identifier in ('../escape','run-1','run-123','run-01/escape'):
            with self.assertRaises(ValueError):lab.locations('replay',identifier)
        places=lab.locations('replay','run-01');places['outputs']['source'].mkdir()
        with self.assertRaises(FileExistsError):lab.prepare('replay','run-01')
        self.assertFalse(places['evidence'].exists())

    def test_bound_input_and_command_drift_prevent_launch(self):
        record=lab.prepare('replay','run-01');record['commands']['clean'].append('--allow-download')
        record['preparation_sha256']=canonical_hash({k:v for k,v in record.items() if k!='preparation_sha256'})
        with self.assertRaisesRegex(ValueError,'closed fixed'):lab.execute(record,'goal authorized fixed DPO2 replay')
        self.child.assert_not_called()

    def test_trusted_contract_bound_preserves_large_complete_science(self):
        path=self.write(self.root/'large-contract.json',{'policy_layout':'x'*92000})
        self.assertEqual(len(lab.bounded_contract(path)['policy_layout']),92000)
        path=self.write(self.root/'too-large-contract.json',{'layout':'x'*(1024**2)})
        with self.assertRaises(ValueError):lab.bounded_contract(path)

    def test_disk_shortfall_retained_before_any_child(self):
        self.disk.return_value=SimpleNamespace(free=179*lab.GIB)
        with self.assertRaisesRegex(RuntimeError,'planning space'):lab.prepare('replay','run-01')
        self.assertTrue((lab.locations('replay','run-01')['evidence']/'preparation-failure.json').is_file())
        self.child.assert_not_called()

    def test_real_tiny_cpu_comparison_charges_inspect_load_and_retains_prefixes(self):
        record=lab.prepare('replay','run-01');places=lab.locations('replay','run-01')
        output=places['outputs']['clean'];output.mkdir()
        limits=record['limits'];io_contract=limits['snapshot_io_contract']
        contract=dict(reference_sha256=digest({'weight':torch.tensor([1.,2.])}),
            work_budget=native.work_budget_contract(limits['work_caps'],lab.JOURNAL_BYTES),
            snapshot_io_budget=io_contract,snapshot_artifact_budget=native.snapshot_artifact_contract(
                max_bytes=limits['artifact_max_bytes'],max_entries=32,journal_max_bytes=lab.JOURNAL_BYTES,
                payload_max_bytes=lab.MAX_SNAPSHOT))
        science=canonical_hash(contract);journal=places['journals']['clean']
        work=WorkLedger.create(journal/'work.jsonl',limits=limits['work_caps'],contract_sha256=science,
            max_bytes=lab.JOURNAL_BYTES,invocation_id='authored-tiny-cpu')
        io=WorkLedger.create(journal/'io.jsonl',limits=io_contract['limits'],
            contract_sha256=io_ledger_contract_sha256(io_contract,science),max_bytes=lab.JOURNAL_BYTES,
            invocation_id='authored-tiny-cpu')
        artifact=ArtifactBudget.create(places['artifacts']['clean'],campaign_id=science,
            max_bytes=limits['artifact_max_bytes'],max_entries=32,journal_max_bytes=lab.JOURNAL_BYTES)
        try:
            state=dict(schema='authored-tiny-comparison',policy={'weight':torch.tensor([2.,3.])},
                reference={'weight':torch.tensor([1.,2.])},optimizer={'step':torch.tensor(2.)},
                sampler_rng=torch.tensor([1,2],dtype=torch.uint8),torch_rng=torch.tensor([3,4],dtype=torch.uint8),
                cuda_rng=[],completed=2,counters={'updates':2},history=[{'update':1},{'update':2}],
                resume_parent=None,work_ledger=work.snapshot(),snapshot_artifact_ledger=artifact.receipt())
            path=places['artifacts']['clean']/('a'*32+'-completed-000002.pt')
            save_snapshot(path,contract=contract,state=state,completed_updates=2,parent_invocation='authored-tiny-cpu',
                max_bytes=lab.MAX_SNAPSHOT,artifact_budget=artifact,
                io_budget=SnapshotIOBudget(io,contract=io_contract,scientific_contract_sha256=science),
                work_receipt_path=Path(str(path)+'.work.json'))
            result=dict(completed_recovery={'path':str(path)},snapshot_artifact_ledger=artifact.receipt())
            self.write(output/'recovery-contract.json',contract)
            expectation=lab.retain_expectation(record,'clean',result)
        finally:
            artifact.close();io.close();work.close()
        observed=lab.comparison_digest(record,expectation)
        self.assertTrue(observed['reference_matches_original'])
        self.assertEqual(observed['components'],lab.component_fingerprints(state))
        self.assertEqual(observed['io_ledger']['reserved']['snapshot_save_operations'],1)
        self.assertEqual(observed['io_ledger']['reserved']['snapshot_inspect_operations'],1)
        self.assertEqual(observed['io_ledger']['reserved']['snapshot_load_operations'],1)
        self.assertEqual(observed['work_ledger']['reserved']['train_updates'],0)
        self.assertFalse(observed['io_ledger']['open_tickets'])


class NumericalComponentTests(unittest.TestCase):
    def state(self):
        value={key:{'tensor':torch.tensor([1.,2.])} for key in lab.COMPONENTS}
        value.update(resume_parent=None,work_ledger={'later_cost':1},snapshot_artifact_ledger={'prefix':1})
        return value

    def test_operational_prefixes_excluded_but_every_numerical_component_compared(self):
        clean=self.state();resumed=deepcopy(clean)
        resumed.update(resume_parent={'completed':1},work_ledger={'later_cost':3},snapshot_artifact_ledger={'prefix':5})
        self.assertEqual(lab.component_fingerprints(clean),lab.component_fingerprints(resumed))
        for key in ('policy','reference','optimizer','sampler_rng','torch_rng','cuda_rng','history','counters'):
            changed=deepcopy(resumed);changed[key]['tensor'][0]+=1
            self.assertNotEqual(lab.component_fingerprints(clean),lab.component_fingerprints(changed),key)

    def test_extra_state_cannot_be_silently_excluded(self):
        state=self.state();state['unreviewed_state']=torch.tensor([1.])
        with self.assertRaisesRegex(ValueError,'state schema'):lab.component_fingerprints(state)


class FixedCPU8EntryControls(unittest.TestCase):
    def fake_torch(self,*,reported_threads=8):
        calls=[]
        def intra(value):
            self.assertEqual(os.environ.get('OMP_NUM_THREADS'),'8');calls.append(('intra',value))
        fake=SimpleNamespace(set_num_threads=intra,set_num_interop_threads=lambda value:calls.append(('interop',value)),
            get_num_threads=lambda:reported_threads,get_num_interop_threads=lambda:1,
            set_float32_matmul_precision=unittest.mock.Mock())
        return fake,calls

    def test_fixed_settings_before_unchanged_native_runpy_and_actual_stdout_witness(self):
        fake,calls=self.fake_torch();output=io.StringIO();arguments=['--checkpoint','/authored-fixed-parent','--updates','2']
        with (patch.dict(os.environ,{'OMP_NUM_THREADS':'96'}),patch.dict(sys.modules,{'torch':fake}),
                patch.object(sys,'argv',['authored-entry']),patch.object(cpu8.runpy,'run_path') as run,
                redirect_stdout(output)):
            self.assertEqual(cpu8.main(arguments),0)
            self.assertEqual(sys.argv,[str(ROOT/'scripts/run_chapter11_spark_dpo.py'),*arguments])
            self.assertEqual(os.environ['OMP_NUM_THREADS'],'8')
        self.assertEqual(calls,[('intra',8),('interop',1)]);fake.set_float32_matmul_precision.assert_not_called()
        run.assert_called_once_with(str(ROOT/'scripts/run_chapter11_spark_dpo.py'),run_name='__main__')
        witness=json.loads(output.getvalue().removeprefix(cpu8.WITNESS_PREFIX))
        self.assertEqual(witness,dict(schema='dongxi-fixed-native-dpo-cpu8-witness-v1',
            target=str(ROOT/'scripts/run_chapter11_spark_dpo.py'),**lab.CPU_SETTINGS))

    def test_cpu_comparison_entry_sets_same_threads_before_wrapper_state_validation(self):
        fake,calls=self.fake_torch();arguments=['--mode','replay','--run-id','run-02','--comparison-child']
        with (patch.dict(os.environ,{}),patch.dict(sys.modules,{'torch':fake}),patch.object(sys,'argv',['authored-entry']),
                patch.object(cpu8.runpy,'run_path') as run,redirect_stdout(io.StringIO())):
            self.assertEqual(cpu8.main(arguments),0)
            self.assertEqual(sys.argv,[str(ROOT/'scripts/run_native_dpo_stages.py'),*arguments])
        self.assertEqual(calls,[('intra',8),('interop',1)])
        run.assert_called_once_with(str(ROOT/'scripts/run_native_dpo_stages.py'),run_name='__main__')

    def test_actual_thread_mismatch_refuses_before_forwarding_or_witness(self):
        fake,_=self.fake_torch(reported_threads=96);output=io.StringIO()
        with (patch.dict(os.environ,{}),patch.dict(sys.modules,{'torch':fake}),patch.object(cpu8.runpy,'run_path') as run,
                redirect_stdout(output),self.assertRaisesRegex(RuntimeError,'thread settings differ')):
            cpu8.main(['--updates','2'])
        run.assert_not_called();self.assertEqual(output.getvalue(),'')


if __name__=='__main__':unittest.main()
