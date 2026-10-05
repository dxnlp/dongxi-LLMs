"""Authored model-free fixtures; no native/model-weight evidence fabricated."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('close_observed_export', ROOT/'scripts/close_native_rlvr_export.py')
lab = importlib.util.module_from_spec(spec); spec.loader.exec_module(lab)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def digest(path):
    raw = Path(path).read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


class ObservedExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.index = 0
    def tearDown(self): self.temp.cleanup()
    def fixture(self, supervision=None):
        self.index += 1; base = self.root/str(self.index)
        paths = {key:base/key for key in ('evidence','pilot','journals')}
        for path in paths.values(): path.mkdir(parents=True)
        record = dict(stage='pilot', group=4, run_id='run-01', recipe={'updates':16},
            local_model_binding={'authored':True}, observed_geometry={'authored':True},
            source_bindings={}, input_bindings={}, prerequisite_bindings={})
        supervision = supervision or dict(status='failed', actual_exit_code=0, stop_reason=None,
            failure=None, cleanup_errors=[], final_record_retained=False,
            journal_error=dict(type='TimeoutError', message='Final logging timed out'),
            observations=[{'authored':'retained intact'}], in_memory_events=[{'event':'authored'}])
        for path,value in ((paths['evidence']/'preparation.json',record),
            (paths['evidence']/'returned-supervision-1.json',supervision),
            (paths['evidence']/'failure.json',{'status':'failed','authored':True}),
            (paths['pilot']/'report.json',{'status':'completed','committed_completed_updates':16}),
            (paths['journals']/'work.jsonl',{'authored':'no real operations'}),
            (paths['journals']/'io.jsonl',{'authored':'no real operations'})):
            path.write_text(json.dumps(value)+'\n')
        def retain(path,value):
            with path.open('x') as handle: json.dump(value,handle)
        result = dict(checks=dict.fromkeys(lab.CHECKS,True), exports={'pilot':dict(
            completed_updates=16, report_binding=digest(paths['pilot']/'report.json'),
            path=str(paths['pilot']/'policy'), files={'authored-placeholder':'not model weights'})})
        stages = SimpleNamespace(paths=lambda *args:paths, _digest=digest,
            canonical_hash=canonical, retain=retain, verify_prepared=lambda value:None,
            check_results=lambda value:result)
        return paths,stages,result

    def test_logging_failure_preserved_and_completed_route_exclusive(self):
        for normal in (False,True):
            supervision = None if not normal else dict(status='completed',actual_exit_code=0,
                stop_reason=None,failure=None,cleanup_errors=[],journal_error=None,final_record_retained=True)
            paths,stages,result = self.fixture(supervision)
            original = digest(paths['evidence']/'returned-supervision-1.json')
            with patch.object(lab,'load_stages',return_value=stages):
                receipt = lab.close_export(4)
                with self.assertRaises(FileExistsError): lab.close_export(4)
            self.assertEqual(receipt['status'],lab.STATUS)
            self.assertEqual(receipt['returned_supervision_binding'],original)
            self.assertEqual(digest(paths['evidence']/'returned-supervision-1.json'),original)
            self.assertEqual(receipt['supervision_observation']['status'],'completed' if normal else 'failed')
            self.assertFalse((paths['evidence']/'acceptance.json').exists())
            self.assertEqual(receipt['export_binding'],result['exports']['pilot'])
            self.assertEqual(receipt['before_bindings'],receipt['after_bindings'])
            self.assertEqual(receipt['observation_sha256'],canonical(
                {key:value for key,value in receipt.items() if key!='observation_sha256'}))

    def test_other_failures_missing_horizon_and_wrong_selectors_refused(self):
        for change in ({'actual_exit_code':1},{'actual_exit_code':False},{'stop_reason':'external-deadline'},
            {'cleanup_errors':['timeout']},{'failure':{'type':'OSError'}},
            {'journal_error':{'type':'TimeoutError','message':'other timeout'}},
            {'final_record_retained':True}):
            paths,stages,_ = self.fixture()
            path = paths['evidence']/'returned-supervision-1.json'
            value = json.loads(path.read_text()); value.update(change); path.write_text(json.dumps(value))
            with patch.object(lab,'load_stages',return_value=stages),self.assertRaises(ValueError):
                lab.close_export(4)
            self.assertFalse((paths['evidence']/'export-observation.json').exists())
        paths,stages,result = self.fixture(); result['checks'][lab.CHECKS[0]]=False
        with patch.object(lab,'load_stages',return_value=stages),self.assertRaises(ValueError): lab.close_export(4)
        paths,stages,_ = self.fixture()
        path = paths['evidence']/'returned-supervision-1.json'
        value = json.loads(path.read_text()); value.pop('failure'); path.write_text(json.dumps(value))
        with patch.object(lab,'load_stages',return_value=stages),self.assertRaises(ValueError): lab.close_export(4)
        for group,run in ((True,'run-01'),(16,'run-01'),(4,'run-02')):
            with self.assertRaises(ValueError): lab.close_export(group,run)

    def test_input_report_drift_refused_without_overwriting_original_failure(self):
        paths,stages,result = self.fixture()
        failure = digest(paths['evidence']/'failure.json')
        def changed(value):
            (paths['pilot']/'report.json').write_text('{"authored":"changed mid-closure"}')
            return result
        stages.check_results=changed
        with patch.object(lab,'load_stages',return_value=stages),self.assertRaisesRegex(ValueError,'changed'):
            lab.close_export(4)
        self.assertFalse((paths['evidence']/'export-observation.json').exists())
        self.assertEqual(digest(paths['evidence']/'failure.json'),failure)


if __name__ == '__main__': unittest.main()
