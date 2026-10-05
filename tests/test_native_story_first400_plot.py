"""Authored small CPU plot controls, never actual story training/quality evidence."""
from contextlib import ExitStack,redirect_stdout
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('native_story_first400_plot',ROOT/'scripts/plot_native_story_first400.py')
adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


def fixture(root,arm):
    """Entire producer/metric receipt schema is explicit authored inert CPU data."""
    stem=f'native-story-{arm}-pilot-{adapter.RUN_ID}'
    evidence=root/'experiments/reports'/stem;output=root/'outputs'/stem
    sources={}
    for name in adapter.REQUIRED_SOURCES:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():path.write_bytes(b'# Explicit authored inert CPU source binding\n')
        sources[name]=dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    bindings=dict(sources=sources,data={'explicit-authored-CPU-data-contract':'same-both-arms'})
    prepared=dict(stage='story-'+arm+'-pilot',run_id=adapter.RUN_ID,bindings=bindings,
        execution_tranche=dict(requested_stop=400,total_horizon=14000,full_schedule_completion=False),
        commands={'pilot':adapter.expected_command(arm,evidence,output)})
    prepared['preparation_sha256']=adapter.canonical_hash(prepared)
    command=prepared['commands']['pilot'];pid=101 if arm=='control' else 102
    terminal=dict(status='completed',actual_exit_code=0,child_pid=pid,child_command=command,
        stop_reason=None,actual_native_profile_executed=True,limits=dict(external_seconds=14400),cleanup_errors=[])
    accepted=dict(status='passed',stage='story-'+arm+'-pilot',completion_level='first400-tranche-not-full14000',
        full_schedule_complete=False,preparation_sha256=prepared['preparation_sha256'],bindings_after=bindings,
        invocations=[dict(label='pilot',status='completed',actual_exit_code=0,child_pid=pid)])
    for path,value in ((evidence/'acceptance.json',accepted),(evidence/'preparation.json',prepared),
            (evidence/'launch-pilot.json',dict(argv=command,preparation_sha256=prepared['preparation_sha256'])),
            (evidence/'returned-supervision-pilot.json',terminal),(evidence/'supervision-pilot/result.json',terminal)):
        save(path,value)
    entry=dict(event='actual-story-deterministic-entry',deterministic_algorithms=True,
        cublas_workspace=':4096:8',attention_backend='SDPA MATH only',tf32=False,
        torch_version='explicit-authored-CPU-metadata-control')
    (evidence/'supervision-pilot/stdout.txt').write_text(json.dumps(entry)+'\nExplicit inert fixture.\n')
    rows=[];targets=0
    for update in range(1,401):
        valid=3000+update%17;targets+=valid
        rows.append(dict(update=update,loss=11/(1+update/20),lr=adapter.scheduled_lr(update,arm),
            gradient_norm=.1,seconds=.01,valid_targets=valid,cumulative_targets=targets,
            processed_positions=16*1024,cumulative_processed_positions=update*16*1024,
            processed_positions_accounting='measured-input-numel'))
    (output/'pilot').mkdir(parents=True,exist_ok=True)
    (output/'pilot/metrics.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    completion=dict(completed_updates=400,requested_stop_reached=True,schedule_complete=False,
        valid_target_budget=50_000_000,stopped_for_target_budget=False,stopped_for_time_budget=False,
        cumulative_targets=targets,cumulative_processed_positions=400*16*1024,
        processed_positions_accounting='measured-input-numel',work_ledger=dict(completed=dict(train_updates=400,
            training_valid_targets=targets,training_windows=400*16),open_tickets=[],failed_tickets=[]))
    save(output/'pilot/completion.json',completion)
    return evidence,output,rows,completion


def rebind_preparation(evidence,mutate):
    prepared=json.loads((evidence/'preparation.json').read_text());mutate(prepared)
    prepared['preparation_sha256']=adapter.canonical_hash({key:value for key,value in prepared.items() if key!='preparation_sha256'})
    save(evidence/'preparation.json',prepared)
    accepted=json.loads((evidence/'acceptance.json').read_text());accepted['preparation_sha256']=prepared['preparation_sha256']
    accepted['bindings_after']=prepared['bindings'];save(evidence/'acceptance.json',accepted)
    launch=json.loads((evidence/'launch-pilot.json').read_text());launch['preparation_sha256']=prepared['preparation_sha256']
    launch['argv']=prepared['commands']['pilot'];save(evidence/'launch-pilot.json',launch)


class NativeStoryFirst400Plot(unittest.TestCase):
    def test_import_and_metadata_controls_do_not_import_torch_or_model_code(self):
        code="import sys,runpy;sys.modules['torch']=None;sys.modules['transformers']=None;runpy.run_path(sys.argv[1],run_name='not_main')"
        result=subprocess.run([sys.executable,'-B','-c',code,str(ROOT/'scripts/plot_native_story_first400.py')],
            capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_original_schedule_is_not_compressed_to400(self):
        self.assertAlmostEqual(adapter.scheduled_lr(1,'control'),1.5e-6)
        self.assertAlmostEqual(adapter.scheduled_lr(200,'control'),3e-4)
        self.assertGreater(adapter.scheduled_lr(400,'control'),.999*3e-4)
        self.assertAlmostEqual(adapter.scheduled_lr(14000,'control'),3e-5)
        for update in (1,199,200,201,400):
            self.assertEqual(adapter.scheduled_lr(update,'half-lr'),adapter.scheduled_lr(update,'control')/2)

    def test_actual_schema_fixture_has400_matched_label_and_padded_position_sequences(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(adapter,'ROOT',root):
                fixture(root,'control');fixture(root,'half-lr');inputs,arms=adapter.load_inputs()
                self.assertEqual(len(arms['control']['rows']),400)
                self.assertEqual(arms['control']['completion'],arms['half-lr']['completion'])
                self.assertFalse(arms['control']['completion']['schedule_complete'])
                self.assertEqual(arms['control']['completion']['cumulative_processed_positions'],6_553_600)
                self.assertEqual(inputs.verify(),inputs.bindings)

    def test_unaccepted_half_arm_refuses_before_any_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);target=root/'experiments/reports/native-story-first400-figures'
            with patch.object(adapter,'ROOT',root),patch.object(adapter,'OUTPUT',target):
                fixture(root,'control');evidence,output,rows,completion=fixture(root,'half-lr')
                value=json.loads((evidence/'acceptance.json').read_text());value['status']='failed';save(evidence/'acceptance.json',value)
                with self.assertRaisesRegex(ValueError,'accepted'):adapter.main()
                self.assertFalse(target.exists())

    def test_invalid400_rows_nonfinite_targets_positions_or_schedule_refuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(adapter,'ROOT',root):evidence,output,rows,completion=fixture(root,'control')
            cases=[lambda r:r.pop(),lambda r:r[0].update(update=True),lambda r:r[20].update(loss=float('nan')),
                lambda r:r[20].update(gradient_norm=float('inf')),lambda r:r[20].update(valid_targets=0),
                lambda r:r[20].update(cumulative_targets=r[20]['cumulative_targets']+1),
                lambda r:r[20].update(processed_positions=1000),lambda r:r[20].update(lr=3e-5)]
            for change in cases:
                altered=deepcopy(rows);change(altered)
                with self.subTest(change=change),self.assertRaises(ValueError):adapter.validate_rows(altered,completion,'control')

    def test_full_horizon_completion_or_unjoined_work_is_not_first400(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(adapter,'ROOT',root):evidence,output,rows,completion=fixture(root,'control')
            for mutate in (lambda c:c.update(schedule_complete=True),lambda c:c.update(requested_stop_reached=False),
                    lambda c:c.update(completed_updates=400.0),lambda c:c.update(valid_target_budget=1_000_000),
                    lambda c:c['work_ledger']['completed'].update(training_valid_targets=1),
                    lambda c:c['work_ledger'].update(failed_tickets=[{'authored':'failed'}])):
                altered=deepcopy(completion);mutate(altered)
                with self.assertRaises(ValueError):adapter.validate_rows(rows,altered,'control')

    def test_changed_bound_producer_source_or_false_math_entry_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(adapter,'ROOT',root):
                evidence,output,rows,completion=fixture(root,'control')
                source=root/'scripts/run_deterministic_story_child.py';source.write_bytes(b'# changed source\n')
                with self.assertRaisesRegex(ValueError,'source bytes changed'):adapter.read_arm('control',adapter.Inputs())
                fixture(root,'control')
                stdout=evidence/'supervision-pilot/stdout.txt';entry=json.loads(stdout.read_text().splitlines()[0])
                entry['attention_backend']='auto';stdout.write_text(json.dumps(entry)+'\n')
                with self.assertRaisesRegex(ValueError,'MATH'):adapter.read_arm('control',adapter.Inputs())

    def test_forged_command_launch_supervision_or_acceptance_join_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(adapter,'ROOT',root):
                evidence,output,rows,completion=fixture(root,'control')
                rebind_preparation(evidence,lambda p:p['commands']['pilot'].extend(['--resume','authored-not-fresh']))
                with self.assertRaisesRegex(ValueError,'first400 command'):adapter.read_arm('control',adapter.Inputs())
                for name,key,value in (('launch-pilot.json','preparation_sha256','f'*64),
                        ('returned-supervision-pilot.json','child_pid',999),('acceptance.json','full_schedule_complete',True)):
                    fixture(root,'control');path=evidence/name;document=json.loads(path.read_text());document[key]=value;save(path,document)
                    with self.assertRaises(ValueError):adapter.read_arm('control',adapter.Inputs())

    def test_rehashed_different_data_contract_or_target_sequence_cannot_be_compared(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(adapter,'ROOT',root):
                fixture(root,'control');evidence,output,rows,completion=fixture(root,'half-lr')
                rebind_preparation(evidence,lambda p:p['bindings']['data'].update(authored='different'))
                with self.assertRaisesRegex(ValueError,'bindings differ'):adapter.load_inputs()
                fixture(root,'half-lr')
                rows[0]['valid_targets']+=1
                for row in rows:row['cumulative_targets']+=1
                completion['cumulative_targets']+=1;completion['work_ledger']['completed']['training_valid_targets']+=1
                (output/'pilot/metrics.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
                save(output/'pilot/completion.json',completion)
                with self.assertRaisesRegex(ValueError,'sequence differs'):adapter.load_inputs()

    def test_partial_metric_tail_duplicate_keys_and_symlink_refuse(self):
        for raw in (b'{"x":1,"x":2}',b'{"x":Infinity}'):
            with self.assertRaises(ValueError):adapter.parse_json(raw)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(adapter,'ROOT',root):
                evidence,output,rows,completion=fixture(root,'control');path=output/'pilot/metrics.jsonl'
                path.write_bytes(path.read_bytes().rstrip(b'\n'))
                with self.assertRaisesRegex(ValueError,'tail'):adapter.read_arm('control',adapter.Inputs())
            link=root/'link';link.symlink_to(path)
            with self.assertRaises(OSError):adapter.read_bytes(link)

    def test_render_and_receipt_are_cpu_only_and_output_is_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);target=root/'experiments/reports/native-story-first400-figures'
            with ExitStack() as stack:
                stack.enter_context(patch.object(adapter,'ROOT',root));stack.enter_context(patch.object(adapter,'OUTPUT',target))
                stack.enter_context(patch.dict(os.environ,{'MPLCONFIGDIR':str(root/'matplotlib-cache'),'CUDA_VISIBLE_DEVICES':''}))
                stack.enter_context(redirect_stdout(io.StringIO()))
                fixture(root,'control');fixture(root,'half-lr');self.assertEqual(adapter.main(),0)
                receipt=json.loads((target/'inputs.json').read_text());image=target/'learning-curves.png'
                self.assertTrue(image.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))
                self.assertEqual(receipt['plot_sha256'],hashlib.sha256(image.read_bytes()).hexdigest())
                self.assertEqual(receipt['input_bindings_before'],receipt['input_bindings_after'])
                self.assertEqual(receipt['source_sha256'],hashlib.sha256(Path(adapter.__file__).read_bytes()).hexdigest())
                self.assertEqual(receipt['schedule']['original_total_updates'],14000)
                self.assertFalse(receipt['schedule']['full_schedule_complete']);self.assertIn('Not held-out NLL',receipt['scope'])
                preserved=image.read_bytes()
                with self.assertRaises(FileExistsError):adapter.main()
                self.assertEqual(image.read_bytes(),preserved)

    def test_raw_mutation_during_render_retains_failure_prefix_not_valid_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);target=root/'experiments/reports/native-story-first400-figures'
            with patch.object(adapter,'ROOT',root),patch.object(adapter,'OUTPUT',target):
                fixture(root,'control');evidence,output,rows,completion=fixture(root,'half-lr')
                def alter(arms,path):
                    path.write_bytes(b'authored incomplete image prefix')
                    rows[0]['loss']+=.1
                    (output/'pilot/metrics.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
                with patch.object(adapter,'render',side_effect=alter):
                    with self.assertRaisesRegex(ValueError,'changed during rendering'):adapter.main()
                self.assertTrue((target/'failure.json').exists());self.assertFalse((target/'inputs.json').exists())
                self.assertEqual((target/'learning-curves.png').read_bytes(),b'authored incomplete image prefix')


if __name__=='__main__':unittest.main()
