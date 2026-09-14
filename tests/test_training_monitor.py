import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer

from dongxi_llms.training_monitor import LogStore, handler, LIMIT


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run = self.root/'fixture'
        self.run.mkdir()
        (self.run/'run.json').write_text(json.dumps({'contract': {'recipe': {'clip': 1}}, 'data_manifest': {'subset': True}}))
        self.store = LogStore(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_real_metrics_and_partial_tail(self):
        (self.run/'metrics.jsonl').write_text('{"update":1,"loss":3.5}\n{"update":')
        snapshot = self.store.snapshot('fixture')
        self.assertEqual(snapshot['metrics'], [{'update':1,'loss':3.5}])
        self.assertTrue(any('partial' in warning for warning in snapshot['warnings']))
        self.assertIn('Stale', snapshot['state'])

    def test_recent_not_claimed_running_and_final_not_claimed_success(self):
        (self.run/'host-memory.jsonl').write_text('{"seconds":0,"available_gib":100}\n')
        self.assertIn('unverified', self.store.snapshot('fixture')['state'])
        (self.run/'final.json').write_text('{"update":3}')
        self.assertEqual(self.store.snapshot('fixture')['state'], 'Final observation saved')
        (self.run/'memory-summary.json').write_text('{"abort_reason":"memory reserve"}')
        self.assertIn('Aborted', self.store.snapshot('fixture')['state'])

    def test_nonfinite_and_reserve_warning(self):
        (self.run/'metrics.jsonl').write_text('{"loss":NaN,"gradient_norm":Infinity}\n')
        (self.run/'host-memory.jsonl').write_text('{"seconds":0,"available_gib":20}\n')
        snapshot = self.store.snapshot('fixture')
        self.assertIsNone(snapshot['metrics'][0]['loss'])
        self.assertTrue(any('Nonfinite' in warning for warning in snapshot['warnings']))
        self.assertTrue(any('reserve' in warning for warning in snapshot['warnings']))
        json.dumps(snapshot, allow_nan=False)

    def test_reject_traversal_and_symlinks(self):
        for name in ('../fixture', '/tmp', '..', 'x/y'):
            with self.assertRaises(ValueError): self.store.snapshot(name)
        (self.root/'alias').symlink_to(self.run)
        with self.assertRaises(ValueError): self.store.snapshot('alias')
        (self.run/'initial.json').symlink_to(self.run/'run.json')
        with self.assertRaises(ValueError): self.store.snapshot('fixture')

    def test_atomic_partial_observation_tolerated(self):
        (self.run/'update-000002.json').write_text('{"update":')
        snapshot = self.store.snapshot('fixture')
        self.assertEqual(snapshot['snapshots'], [])

    def test_new_updates_visible_without_restart(self):
        path = self.run/'metrics.jsonl'
        path.write_text('{"update":1,"loss":3}\n')
        self.assertEqual(len(self.store.snapshot('fixture')['metrics']),1)
        with path.open('a') as f: f.write('{"update":2,"loss":2}\n')
        self.assertEqual(self.store.snapshot('fixture')['metrics'][-1]['update'],2)

    def test_checkpoint_never_deserialized(self):
        (self.run/'update-000001.pt').write_bytes(b'not a valid checkpoint')
        self.assertEqual(self.store.snapshot('fixture')['checkpoints'][0]['bytes'],22)

    def test_tail_is_bounded(self):
        with (self.run/'metrics.jsonl').open('w') as f:
            for i in range(70000): f.write(json.dumps({'update': i, 'loss': 2.0})+'\n')
        rows, _, truncated = self.store.read('fixture', 'metrics.jsonl', True)
        self.assertTrue(truncated)
        self.assertEqual(rows[-1]['update'],69999)
        self.assertGreater(rows[0]['update'],0)

    def test_http_read_only_no_arbitrary_file_access(self):
        assets = Path(__file__).resolve().parents[1]/'visuals'/'training-monitor'
        server = ThreadingHTTPServer(('127.0.0.1',0), handler(self.store, assets))
        thread = threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        url=f'http://127.0.0.1:{server.server_port}'
        try:
            with urlopen(url+'/api/runs') as response:
                self.assertEqual(json.load(response)[0]['name'],'fixture')
            with urlopen(url+'/') as response:
                self.assertIn(b'Training observatory',response.read())
                self.assertIn("default-src 'self'",response.headers['Content-Security-Policy'])
            for request, code in ((Request(url+'/api/runs',method='POST'),501),
                                  (Request(url+'/api/runs',headers={'Host':'evil.example'}),403),
                                  (Request(url+'/api/runs',headers={'Origin':'https://evil.example'}),403),
                                  (Request(url+'/api/run?name=../fixture'),404),
                                  (Request(url+'/fixture/run.json'),404)):
                with self.assertRaises(HTTPError) as context: urlopen(request)
                self.assertEqual(context.exception.code,code)
        finally:
            server.shutdown();server.server_close();thread.join(timeout=2)


if __name__ == '__main__': unittest.main()
