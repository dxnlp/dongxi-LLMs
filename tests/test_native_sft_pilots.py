"""Closed pilot preparation/failure retention; every owned child is mocked."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dongxi_llms.run_identity import artifact_hashes, canonical_hash
from dongxi_llms.snapshot_io_budget import IO_KEYS, validate_io_contract

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/run_native_sft_pilots.py'
spec = importlib.util.spec_from_file_location('native_sft_pilots', SCRIPT)
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


class NativeSFTPilotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for directory in ('outputs', 'experiments/reports', 'experiments/data'):
            (self.root/directory).mkdir(parents=True)
        for name in (*lab.SOURCES, *lab.OWN_SOURCES):
            path = self.root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('authored source fixture\n')
        data = self.root/'outputs/course-sft-interface-v1'
        data.mkdir()
        card = {'splits': {}}
        for name,count in (('train',240), ('dev',60)):
            text = ''.join(json.dumps({'id':f'{name}-{number}'})+'\n' for number in range(count))
            (data/(name+'.jsonl')).write_text(text)
            card['splits'][name] = {'sha256':hashlib.sha256(text.encode()).hexdigest()}
        self.write('experiments/data/instruction-interface-v1-data-card.json', card)
        template = self.root/'experiments/data/instruction_interface_v1.jinja'
        template.write_text('authored template fixture\n')
        self.base = self.root/'cached'/lab.REVISION
        self.base.mkdir(parents=True)
        (self.base/'config.json').write_text('{}\n')
        (self.base/'model.safetensors').write_bytes(b'authored fixture weights')
        reference = dict(model=lab.MODEL, revision=lab.REVISION, tokenizer_revision=lab.REVISION,
            base_checkpoint_files=artifact_hashes(self.base),
            template_sha256=hashlib.sha256(template.read_bytes()).hexdigest())
        for name in ('train','dev'):
            reference[name+'_sha256'] = card['splits'][name]['sha256']
            reference['encoded_'+name+'_sha256'] = lab.ENCODED[name]
        self.write('outputs/native-sft-full-replay-20261005-original/config.json', reference)
        self.write('experiments/reports/2026-10-05-native-sft-full-replay/acceptance-retry02.json',
            {'status':'passed','checks':{'actual_replay':True}})
        self.write('experiments/reports/2026-10-05-native-sft-lora-replay/acceptance.json',
            {'status':'passed','checks':{'actual_replay':True}})
        lock = self.root/'spark.lock'
        lock.write_text('authored environment lock\n')
        for target, value in (('ROOT',self.root), ('INTERPRETER',sys.executable),
                              ('ENVIRONMENT_LOCK',str(lock))):
            replacement = patch.object(lab,target,value)
            replacement.start();self.addCleanup(replacement.stop)
        cache = patch.object(lab,'cached_snapshot',return_value=self.base)
        cache.start();self.addCleanup(cache.stop)
        disk = patch.object(lab.shutil,'disk_usage',return_value=SimpleNamespace(free=31*lab.GIB))
        self.disk = disk.start();self.addCleanup(disk.stop)
        self.supervisor = patch.object(lab,'_supervise')
        self.child = self.supervisor.start();self.addCleanup(self.supervisor.stop)

    def write(self, relative, value):
        path = self.root/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value)+'\n')
        return path

    def prepared(self, mode='full', run_id='run-01'):
        return lab.prepare(mode,run_id)

    def read_evidence(self, record, filename):
        return json.loads((Path(record['evidence'])/filename).read_text())

    def successful_child(self, record, *, change_source=False, updates=400):
        def finish(*args,**kwargs):
            output = Path(record['output']);output.mkdir()
            result = dict(status='completed',updates=updates,
                cumulative_work_ledger=dict(open_tickets={},failed_tickets={}),
                cumulative_snapshot_io_ledger=dict(open_tickets={},failed_tickets={}))
            self.write(str((output/'result.json').relative_to(self.root)),result)
            self.write(str((output/'config.json').relative_to(self.root)),
                {'encoded_'+key+'_sha256':value for key,value in lab.ENCODED.items()})
            (output/'metrics.jsonl').write_text(''.join(json.dumps({'update':k})+'\n' for k in range(1,401)))
            for cursor in (0,100,200,300,400):
                (output/f'checkpoint-{cursor:06d}.pt.commit.json').write_text('{}\n')
            policy = output/'policy';policy.mkdir()
            if record['mode'] == 'full':
                (policy/'config.json').write_text('{}\n')
                (policy/'model.safetensors').write_bytes(b'authored exported weights')
            else:
                (policy/'adapter_config.json').write_text('{}\n')
                (policy/'adapter_model.safetensors').write_bytes(b'authored exported adapter')
            self.write(str((policy/'course-genealogy.json').relative_to(self.root)),
                dict(kind='full-HF-model' if record['mode']=='full' else 'PEFT-adapter-requires-pinned-base',
                    base_model=lab.MODEL,base_revision=lab.REVISION,
                    base_checkpoint_files=record['local_base_binding']['files']))
            if change_source:
                (self.root/lab.SOURCES[0]).write_text('changed executable source\n')
            return dict(status='completed',actual_exit_code=0,child_seconds=1.0)
        self.child.side_effect=finish

    def test_fixed_arms_caps_cadence_and_fresh_base(self):
        for mode in ('full','lora'):
            record = self.prepared(mode)
            argv = record['argv']
            self.assertEqual(argv[argv.index('--updates')+1],'400')
            self.assertEqual(argv[argv.index('--mode')+1],mode)
            self.assertEqual(argv[argv.index('--rank')+1],'8')
            self.assertEqual(argv[argv.index('--checkpoint-every')+1],'100')
            self.assertEqual(argv[argv.index('--runtime-seconds')+1],'3600')
            self.assertNotIn('--resume',argv)
            self.assertNotIn('--allow-download',argv)
            self.assertEqual(record['local_base_binding']['files'],artifact_hashes(self.base))
            self.assertEqual(record['limits']['work_caps'],lab.WORK_CAPS)
            self.assertEqual(len(lab.WORK_CAPS),16)
            io = validate_io_contract(record['limits']['snapshot_io_contract'])
            self.assertEqual(set(io['limits']),set(IO_KEYS))
            self.assertEqual(io['limits']['snapshot_save_operations'],5)
            self.assertEqual(io['limits']['snapshot_load_operations'],0)
            self.assertEqual(io['limits']['snapshot_inspect_operations'],0)
            journal_parent = Path(argv[argv.index('--work-journal')+1]).parent
            self.assertEqual(stat.S_IMODE(journal_parent.stat().st_mode),0o700)
        self.child.assert_not_called()

    def test_run_identifier_refuses_path_or_additional_argv(self):
        for value in ('../outside','run-1','run-001','run-01/evil',None):
            with self.assertRaises(ValueError):lab.paths('full',value)
        with self.assertRaises(SystemExit):lab.main(['--mode','full','--run-id','run-01','--updates','900'])
        self.child.assert_not_called()

    def test_existing_output_refused_before_any_evidence_creation(self):
        locations = lab.paths('full','run-01')
        locations['output'].mkdir()
        with self.assertRaises(FileExistsError):self.prepared()
        self.assertFalse(locations['evidence'].exists())
        self.child.assert_not_called()

    def test_disk_failure_retained_and_never_launches(self):
        self.disk.return_value = SimpleNamespace(free=30*lab.GIB-1)
        with self.assertRaises(RuntimeError):self.prepared()
        failure = json.loads((lab.paths('full','run-01')['evidence']/'preparation-failure.json').read_text())
        self.assertEqual(failure['status'],'failed')
        self.child.assert_not_called()

    def test_failed_actual_replay_prevents_preparation(self):
        self.write('experiments/reports/2026-10-05-native-sft-full-replay/acceptance-retry02.json',
            {'status':'failed','checks':{'actual_replay':False}})
        with self.assertRaisesRegex(ValueError,'recovery'):self.prepared()
        self.child.assert_not_called()

    def test_prelaunch_source_drift_retained_without_launch(self):
        record = self.prepared()
        (self.root/lab.SOURCES[0]).write_text('changed source\n')
        with self.assertRaisesRegex(ValueError,'source changed'):lab.execute(record,'authorized bounded pilot')
        self.assertEqual(self.read_evidence(record,'failure.json')['status'],'failed')
        self.child.assert_not_called()

    def test_prelaunch_input_and_base_drift_refuse(self):
        for index,target in enumerate(('train','base')):
            record = self.prepared(run_id=f'run-{index+1:02d}')
            path = (Path(record['input_bindings']['train']['path']) if target=='train'
                    else self.base/'model.safetensors')
            original = path.read_bytes();path.write_bytes(original+b'changed')
            with self.assertRaisesRegex(ValueError,'changed after preparation'):
                lab.execute(record,'authorized bounded pilot')
            path.write_bytes(original)
        self.child.assert_not_called()

    def test_prepared_command_tampering_refuses(self):
        record = deepcopy(self.prepared());record['argv'].append('--allow-download')
        with self.assertRaisesRegex(ValueError,'command or limits changed'):
            lab.execute(record,'authorized bounded pilot')
        self.child.assert_not_called()

    def test_rehashed_arbitrary_argv_still_refuses_closed_route(self):
        record = deepcopy(self.prepared());record['argv'].append('--allow-download')
        record['preparation_sha256'] = canonical_hash({key:value for key,value in record.items()
            if key != 'preparation_sha256'})
        with self.assertRaisesRegex(ValueError,'no extra argv'):
            lab.execute(record,'authorized bounded pilot')
        self.child.assert_not_called()

    def test_failed_supervisor_retained(self):
        record = self.prepared()
        self.child.return_value = dict(status='failed',actual_exit_code=-15,child_seconds=.5)
        with self.assertRaisesRegex(RuntimeError,'pilot failed'):lab.execute(record,'authorized bounded pilot')
        self.assertEqual(self.read_evidence(record,'returned-supervision.json')['actual_exit_code'],-15)
        self.assertTrue(self.read_evidence(record,'failure.json')['actual_supervision_retained'])

    def test_postlaunch_source_change_is_failure_with_child_receipt(self):
        record = self.prepared();self.successful_child(record,change_source=True)
        with self.assertRaisesRegex(ValueError,'source changed'):lab.execute(record,'authorized bounded pilot')
        self.assertTrue(self.read_evidence(record,'failure.json')['actual_supervision_retained'])
        self.assertFalse((Path(record['evidence'])/'acceptance.json').exists())

    def test_complete_full_and_adapter_export_bindings(self):
        for mode in ('full','lora'):
            record = self.prepared(mode);self.successful_child(record)
            self.assertEqual(lab.execute(record,'authorized bounded pilot'),0)
            accepted = self.read_evidence(record,'acceptance.json')
            self.assertEqual(accepted['status'],'passed')
            self.assertTrue(all(accepted['checks'].values()))
            self.assertEqual(self.child.call_args.kwargs['seconds'],3600)
            self.assertEqual(self.child.call_args.kwargs['native_stage'],'assistant-pilot')
            self.assertTrue(self.child.call_args.kwargs['native'])
            self.assertEqual(accepted['exported_policy']['path'],record['output']+'/policy')

    def test_incomplete400_keeps_failed_acceptance(self):
        record = self.prepared();self.successful_child(record,updates=399)
        with self.assertRaisesRegex(RuntimeError,'acceptance failed'):lab.execute(record,'authorized bounded pilot')
        self.assertFalse(self.read_evidence(record,'acceptance.json')['checks']['completed400'])
        self.assertEqual(self.read_evidence(record,'failure.json')['status'],'failed')


if __name__ == '__main__':
    unittest.main()
