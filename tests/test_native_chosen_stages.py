"""Focused CPU controls for fixed chosen stage preparation/accounted comparison.

No pretrained weight load, CUDA, network or model child is called by these tests.
The real tiny snapshot check uses the same shared work19/I/O9 journal readers.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from dongxi_llms.run_identity import artifact_hashes,canonical_hash
from dongxi_llms.snapshot_io_budget import SnapshotIOBudget,io_ledger_contract_sha256
from dongxi_llms.work_budget import WorkLedger
from dongxi_llms.training_snapshot import save_snapshot

ROOT=Path(__file__).resolve().parents[1]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

lab=module('native_chosen_stages',ROOT/'scripts/run_native_chosen_stages.py')
native=module('chosen_native_budget_helpers',ROOT/'scripts/run_chapter11_spark_dpo.py')
observe_actual_geometry=lab.observed_geometry


class NativeChosenStageTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        for name in lab.SOURCES:
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('authored CPU source fixture\n')
        for name,count in (('train',8),('validation',4),('evaluation',4)):
            path=self.root/'fixtures/chapter11'/(name+'.jsonl');path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(''.join(json.dumps({'id':f'{name}-{i}'})+'\n' for i in range(count)))
        path=self.root/'fixtures/matched-chosen-sft/protocol.json';path.parent.mkdir(parents=True);self.write(path,{'authored':True})
        (self.root/'outputs').mkdir();(self.root/'experiments/reports').mkdir(parents=True)
        self.lock=self.root/'spark.lock';self.lock.write_text('authored lock\n')
        self.parent=self.root/'outputs'/'fixed-full400'/'policy';self.parent.mkdir(parents=True)
        self.write(self.parent/'config.json',{'authored':True});(self.parent/'model.safetensors').write_bytes(b'authored parent weights')
        self.tokenizer=self.root/'tokenizer';self.tokenizer.mkdir();self.write(self.tokenizer/'tokenizer_config.json',{})
        self.binding=dict(path=str(self.parent),files=artifact_hashes(self.parent),selected_parent='predeclared full400',acceptance={'authored':True})
        validation=dict.fromkeys(native.BUDGET_KEYS,0);validation.update(valid_targets=12,logical_sequence_tokens=40,
            policy_forward_calls=8,policy_forward_positions=36,reference_forward_calls=8,reference_forward_positions=36,
            evaluation_calls=16,evaluation_positions=72)
        self.geometry=dict(interface={'authored':True},encoded_train_sha256='a'*64,encoded_validation_sha256='b'*64,
            prefix_sha256='c'*64,prefix_lengths=[29,28,29,28],
            one_update=dict(train_updates=1,sampled_examples=4,sampler_draws=4,valid_targets=20,
                logical_sequence_tokens=144,policy_forward_calls=4,policy_forward_positions=144),
            one_validation=validation,one_generation=native.generation_upper([[1]*length for length in (29,28,29,28)],64),
            validation_layout=dict(updates=100,accumulation=4,sample_work=[{}],
                policy_shapes={'weight':dict(shape=[2,2],dtype='torch.float32')},
                optimizer_shapes=[dict(shape=[2,2],dtype='torch.float32')],cpu_rng_bytes=16,cuda_rng_bytes=[16]),
            tokenizer_snapshot=dict(path=str(self.tokenizer),files=artifact_hashes(self.tokenizer)),
            chosen_dataset=dict(encoded_sha256='d'*64,interface_sha256='e'*64,split={'groups':'disjoint'}),
            weights_loaded=False,cuda_observed=False)
        for key,value in dict(ROOT=self.root,INTERPRETER=sys.executable,ENVIRONMENT_LOCK=str(self.lock)).items():
            current=patch.object(lab,key,value);current.start();self.addCleanup(current.stop)
        for key,value in (('bridge',SimpleNamespace(native_runner=lambda:native)),('parent_binding',self.binding),('observed_geometry',self.geometry)):
            current=patch.object(lab,key,return_value=value);current.start();self.addCleanup(current.stop)
        current=patch.object(lab.shutil,'disk_usage',return_value=SimpleNamespace(free=181*lab.GIB));self.disk=current.start();self.addCleanup(current.stop)
        current=patch.object(lab,'validate_work_receipt',side_effect=lambda value:value);current.start();self.addCleanup(current.stop)
        current=patch.object(lab,'_supervise');self.child=current.start();self.addCleanup(current.stop)

    def write(self,path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n');return path

    def fake_child(self,record,*,fail=None,drift=False):
        places=lab.locations(record['mode'],record['run_id'])
        def run(argv,output,**kwargs):
            if '--comparison-child' in argv:
                self.write(Path(record['evidence'])/'comparison.json',dict(status='passed',
                    checks={key:True for key in lab.REPLAY_CHECKS if key not in ('all_actual_children_completed','exact_numerical_metric_tail')}))
                return dict(status='completed',actual_exit_code=0)
            role=argv[argv.index('--role')+1]
            if role==fail:return dict(status='failed',actual_exit_code=-15)
            path=places['outputs'][role];(path/'checkpoints').mkdir(parents=True)
            updates=record['limits']['updates']
            cursors=[0,1,2] if role in ('clean','source') else [1,2] if role=='resumed' else [0,20,40,60,80,100]
            for cursor in cursors:
                payload=path/'checkpoints'/f'completed-{cursor:06d}.pt';payload.write_bytes(b'authored payload')
                header=dict(phase='completed',completed_updates=cursor,payload_bytes=payload.stat().st_size,
                    payload_sha256=hashlib.sha256(payload.read_bytes()).hexdigest(),contract_sha256='0'*64)
                self.write(str(payload)+'.commit.json',header);self.write(str(payload)+'.work.json',{'snapshot':header})
            final=path/'checkpoints'/f'completed-{updates:06d}.pt'
            result=dict(completed_recovery=dict(path=str(final),completed_updates=updates),reference_sha256='f'*64,
                final_reference_sha256='f'*64,reference_has_gradients=False,
                cumulative_work_ledger=dict(open_tickets={},failed_tickets={}),snapshot_io_ledger=dict(open_tickets={},failed_tickets={}))
            self.write(path/'result.json',result);self.write(path/'recovery-contract.json',{'authored':True,'updates':updates})
            numbers=range(2,3) if role=='resumed' else range(1,updates+1)
            rows=[dict(update=i,loss=.5,gradient_norm=.2,indices=[1,2,3,4],work={'chosen_targets':4}) for i in numbers]
            (path/'metrics.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            (path/'policy').mkdir();(path/'policy'/'model.safetensors').write_bytes(b'authored export')
            if drift:(self.root/lab.SOURCES[0]).write_text('changed frozen source\n')
            return dict(status='completed',actual_exit_code=0)
        self.child.side_effect=run

    def gate(self):
        replay=lab.prepare('replay','run-01')
        self.write(Path(replay['evidence'])/'acceptance.json',dict(status='passed',checks={key:True for key in lab.REPLAY_CHECKS},
            parent_binding=replay['parent_binding'],invocations=[dict(role=role,result=dict(status='completed',actual_exit_code=0))
                for role in ('clean','source','resumed','cpu-comparison')]))
        return replay

    def test_fixed_recipe_and_closed_three_child_commands(self):
        record=lab.prepare('replay','run-01')
        self.assertEqual(set(record['commands']),{'clean','source','resumed'})
        self.assertEqual(record['recipe'],dict(seed=1818,updates=2,accumulation=4,learning_rate=5e-7,beta=.1,
            weight_decay=.01,clip_norm=1.,max_length=512,max_new_tokens=64,decoding='greedy-uncached'))
        for role,argv in record['commands'].items():
            self.assertEqual(argv[-2:],['--role',role]);self.assertNotIn('--checkpoint',argv);self.assertNotIn('--allow-download',argv)
        self.assertEqual(lab.group_for('replay','source'),lab.group_for('replay','resumed'))
        self.assertNotEqual(lab.group_for('replay','clean'),lab.group_for('replay','source'));self.child.assert_not_called()

    def test_preparation_reloaded_tokenizer_adopts_frozen_parent_template(self):
        base=deepcopy(self.geometry)
        base['validation_layout']['torch_rng_bytes']=base['validation_layout'].pop('cpu_rng_bytes')
        base['validation_layout']['cuda_rng_count']=len(base['validation_layout']['cuda_rng_bytes'])
        provided=SimpleNamespace(chat_template=None);saved=SimpleNamespace(chat_template='authored saved template')
        def verify(parent,actual_saved,actual_provided,**kwargs):
            self.assertIs(actual_saved,saved);self.assertIs(actual_provided,provided)
            actual_provided.chat_template=actual_saved.chat_template
            return base['interface'],None
        helper=SimpleNamespace(observed_geometry=lambda parent,inputs:base,
            native_runner=lambda:SimpleNamespace(verify_parent_tokenizer=verify))
        data={key:'authored' for key in ('rows_sha256','encoded_sha256','interface_sha256','train_ids',
            'train_groups','validation_ids','evaluation_ids','pad_id','stop_ids','split','terminal_suffix_ids')}
        data.update(train=[([1,2,3],[False,True,True]),([1,4,3],[False,True,True])])
        data['train']=[data['train']]
        def encoded(tokenizer,inputs):
            self.assertEqual(tokenizer.chat_template,'authored saved template');return data
        with patch.object(lab,'bridge',return_value=helper),patch('transformers.AutoTokenizer.from_pretrained',side_effect=[provided,saved]),\
                patch.object(lab,'encoded_dataset',side_effect=encoded):
            geometry=observe_actual_geometry(self.binding,{})
        self.assertEqual(geometry['one_update']['policy_forward_positions'],12)
        self.assertEqual(geometry['chosen_dataset']['terminal_suffix_ids'],'authored')

    def test_exact_shared_three_spent_updates_and_chosen_semantic_validation(self):
        limits=lab.allowances('replay',self.geometry);caps=limits['work_caps']
        self.assertEqual(set(caps),set(native.BUDGET_KEYS));self.assertEqual(caps['train_updates'],3)
        self.assertEqual(caps['sampled_examples'],12);self.assertEqual(caps['recovery_validation_operations'],6)
        self.assertEqual(caps['recovery_history_rows'],7);self.assertEqual(caps['recovery_sampler_draws'],28)
        self.assertEqual(limits['semantic_validation_cursors'],[0,1,2,1,1,2])
        self.assertEqual(limits['snapshot_io_contract']['limits']['snapshot_save_operations'],5)
        self.assertEqual(limits['snapshot_io_contract']['limits']['snapshot_inspect_operations'],3)
        self.assertEqual(limits['snapshot_io_contract']['limits']['snapshot_load_operations'],3)
        self.assertIn('no ArtifactBudget',limits['artifact_scope'])

    def test_replay_actual_receipts_and_fixed_independent_resume_expectation(self):
        record=lab.prepare('replay','run-01');self.fake_child(record)
        self.assertEqual(lab.execute(record,'Goal authorized fixed chosen2 replay'),0);self.assertEqual(self.child.call_count,4)
        expectation=json.loads((Path(record['evidence'])/'resume-expectation.json').read_text())
        lab.verify_expectation(record,expectation)
        self.assertEqual(expectation['header']['completed_updates'],1)
        self.assertEqual(set(expectation['metadata']),{'contract','work_receipt','header'})
        acceptance=json.loads((Path(record['evidence'])/'acceptance.json').read_text())
        self.assertEqual(set(acceptance['checks']),set(lab.REPLAY_CHECKS));self.assertTrue(all(acceptance['checks'].values()))

    def test_pilot_requires_actual_current_replay_gate(self):
        with self.assertRaises(FileNotFoundError):lab.prepare('pilot','run-01')
        self.child.assert_not_called()

    def test_pilot100_1800_seconds_and_six_commits(self):
        self.gate();record=lab.prepare('pilot','run-01');self.fake_child(record)
        self.assertEqual(record['limits']['work_caps']['train_updates'],100)
        self.assertEqual(record['limits']['snapshot_io_contract']['limits']['snapshot_save_operations'],6)
        self.assertEqual(lab.execute(record,'Goal authorized fixed chosen100 pilot'),0)
        self.assertEqual(self.child.call_args.kwargs['seconds'],1800)
        self.assertEqual(self.child.call_args.kwargs['native_stage'],'preference-pilot')

    def test_pilot_source_drift_refused(self):
        self.gate();(self.root/lab.SOURCES[0]).write_text('changed source\n')
        with self.assertRaisesRegex(ValueError,'source changed'):lab.prepare('pilot','run-01')
        self.child.assert_not_called()

    def test_pilot_original_group_sidecar_drift_refused(self):
        self.gate();(self.root/'fixtures/matched-chosen-sft/protocol.json').write_text('changed groups\n')
        with self.assertRaisesRegex(ValueError,'input role changed'):lab.prepare('pilot','run-01')
        self.child.assert_not_called()

    def test_pilot_interface_geometry_drift_refused(self):
        self.gate();self.geometry['chosen_dataset']['interface_sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'encoded/interface geometry changed'):lab.prepare('pilot','run-01')
        self.child.assert_not_called()

    def test_failed_child_retains_returned_receipt_and_never_launches_resume(self):
        record=lab.prepare('replay','run-01');self.fake_child(record,fail='source')
        with self.assertRaisesRegex(RuntimeError,'child failed'):lab.execute(record,'Goal authorized fixed chosen2 replay')
        self.assertEqual(self.child.call_count,2)
        evidence=Path(record['evidence']);self.assertTrue((evidence/'returned-supervision-source.json').exists())
        self.assertFalse((evidence/'launch-resumed.json').exists());self.assertTrue((evidence/'failure.json').exists())

    def test_postchild_source_drift_preserves_actual_supervision(self):
        record=lab.prepare('replay','run-01');self.fake_child(record,drift=True)
        with self.assertRaisesRegex(ValueError,'source changed'):lab.execute(record,'Goal authorized fixed chosen2 replay')
        self.assertEqual(self.child.call_count,1)
        self.assertTrue((Path(record['evidence'])/'returned-supervision-clean.json').exists())

    def test_command_or_recipe_drift_cannot_launch(self):
        record=lab.prepare('replay','run-01');record['commands']['clean'].append('--allow-download')
        record['preparation_sha256']=canonical_hash({key:value for key,value in record.items() if key!='preparation_sha256'})
        with self.assertRaisesRegex(ValueError,'fixed chosen commands'):lab.execute(record,'Goal authorized fixed chosen2 replay')
        self.child.assert_not_called()

    def test_existing_targets_and_escape_identifiers_refuse(self):
        for identifier in ('../escape','run-1','run-123','run-01/escape'):
            with self.assertRaises(ValueError):lab.locations('replay',identifier)
        places=lab.locations('replay','run-01');places['outputs']['source'].mkdir()
        with self.assertRaises(FileExistsError):lab.prepare('replay','run-01')
        self.assertFalse(places['evidence'].exists())

    def test_disk_shortfall_retained_before_execution(self):
        self.disk.return_value=SimpleNamespace(free=179*lab.GIB)
        with self.assertRaisesRegex(RuntimeError,'planning space'):lab.prepare('replay','run-01')
        self.assertTrue((lab.locations('replay','run-01')['evidence']/'preparation-failure.json').exists());self.child.assert_not_called()

    def test_large_trusted_contract_role_keeps_complete_science(self):
        path=self.write(self.root/'large.json',{'layout':'x'*92000})
        self.assertEqual(len(lab.bounded_contract(path)['layout']),92000)
        path=self.write(self.root/'oversize.json',{'layout':'x'*(1024**2)})
        with self.assertRaises(ValueError):lab.bounded_contract(path)

    def test_independent_expected_header_and_path_cannot_drift(self):
        record=lab.prepare('replay','run-01');self.fake_child(record);lab.execute(record,'Goal authorized fixed chosen2 replay')
        path=Path(record['evidence'])/'resume-expectation.json';expected=json.loads(path.read_text())
        altered=deepcopy(expected);altered['path']=str(self.root/'unbound.pt')
        with self.assertRaisesRegex(ValueError,'fixed independent'):lab.verify_expectation(record,altered)
        Path(expected['metadata']['header']['path']).write_text('{}\n')
        with self.assertRaisesRegex(ValueError,'metadata changed'):lab.verify_expectation(record,expected)

    def test_real_tiny_cpu_comparison_admits_inspect_load_and_preserves_prefix(self):
        record=lab.prepare('replay','run-01');places=lab.locations('replay','run-01');output=places['outputs']['clean']
        (output/'checkpoints').mkdir(parents=True);limits=record['limits'];control=lab.chosen();io_contract=limits['snapshot_io_contract']
        contract=dict(schema=control.RECOVERY_SCHEMA,loop=deepcopy(self.geometry['validation_layout']),
            parent_sha256='0'*64,parent_files={},inputs={},sources={},environment={},interface_sha256='1'*64,dataset={},
            recipe=record['recipe'],work_budget=control.chosen_work_budget_contract(limits['work_caps'],lab.JOURNAL_BYTES),
            snapshot_io_budget=io_contract)
        contract['loop']['updates']=2;science=canonical_hash(contract);journal=places['journals']['clean']
        work=WorkLedger.create(journal/'work.jsonl',limits=limits['work_caps'],contract_sha256=science,
            max_bytes=lab.JOURNAL_BYTES,invocation_id='authored-tiny-cpu')
        io=WorkLedger.create(journal/'io.jsonl',limits=io_contract['limits'],contract_sha256=io_ledger_contract_sha256(io_contract,science),
            max_bytes=lab.JOURNAL_BYTES,invocation_id='authored-tiny-cpu')
        try:
            state=dict(schema='authored-tiny-state',model={'weight':torch.tensor([1.,2.])},optimizer={'step':torch.tensor(2.)},
                sampler_rng=torch.tensor([1,2],dtype=torch.uint8),torch_rng=torch.tensor([3,4],dtype=torch.uint8),
                cuda_rng=[],completed=2,history=[{'update':1},{'update':2}],work_ledger=work.snapshot())
            path=output/'checkpoints'/'completed-000002.pt'
            save_snapshot(path,contract=contract,state=state,completed_updates=2,parent_invocation='authored-tiny-cpu',
                max_bytes=lab.MAX_SNAPSHOT,io_budget=SnapshotIOBudget(io,contract=io_contract,scientific_contract_sha256=science),
                work_receipt_path=str(path)+'.work.json')
            self.write(output/'recovery-contract.json',contract)
            expected=lab.retain_expectation(record,'clean',dict(completed_recovery={'path':str(path)}))
        finally:io.close();work.close()
        observed=lab.comparison_digest(record,expected)
        self.assertEqual(observed['components'],lab.component_fingerprints(state));self.assertEqual(observed['completed'],2)
        self.assertEqual(observed['io_ledger']['reserved']['snapshot_save_operations'],1)
        self.assertEqual(observed['io_ledger']['reserved']['snapshot_inspect_operations'],1)
        self.assertEqual(observed['io_ledger']['reserved']['snapshot_load_operations'],1)
        self.assertEqual(observed['work_ledger']['reserved']['train_updates'],0)
        self.assertFalse(observed['io_ledger']['open_tickets'])


class ChosenComponentTests(unittest.TestCase):
    def test_every_numerical_field_included_only_work_prefix_excluded(self):
        original={key:{'tensor':torch.tensor([1.,2.])} for key in lab.COMPONENTS};original['work_ledger']={'later_spending':1}
        later=deepcopy(original);later['work_ledger']={'later_spending':3}
        self.assertEqual(lab.component_fingerprints(original),lab.component_fingerprints(later))
        for key in lab.COMPONENTS:
            changed=deepcopy(later);changed[key]['tensor'][0]+=1
            self.assertNotEqual(lab.component_fingerprints(original),lab.component_fingerprints(changed),key)
        later['unreviewed_state']={}
        with self.assertRaisesRegex(ValueError,'state schema'):lab.component_fingerprints(later)


if __name__=='__main__':unittest.main()
