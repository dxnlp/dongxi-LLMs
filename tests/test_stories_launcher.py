"""Exercise launch gates without allocating a model or running subprocesses."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

path = Path(__file__).resolve().parents[1]/'scripts/launch_stories_learning.py'
spec = importlib.util.spec_from_file_location('stories_launcher', path)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def fixture(self, root, full=True):
        data=root/'data';data.mkdir()
        (data/'manifest.json').write_text(json.dumps(dict(subset=not full,
            train=dict(documents=100001, valid_targets=20000000), valid=dict(documents=10001))))
        (root/'recovery.json').write_text('{"bitwise_identical":true}')
        config=dict(data=str(data),status_file=str(root/'status'/'status.json'),
                    data_service='fixture',data_wait_seconds=1,sources={},
                    recovery_report=str(root/'recovery.json'),output=str(root/'training'),training_args=[])
        file=root/'config.json';file.write_text(json.dumps(config));return file

    def test_subset_cannot_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            file=self.fixture(Path(directory), full=False)
            with patch.object(launcher.subprocess,'run') as run:
                with self.assertRaises(RuntimeError): launcher.launch(file)
                run.assert_not_called()

    def test_other_gpu_process_cannot_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            file=self.fixture(Path(directory))
            with patch.object(launcher.shutil,'disk_usage',return_value=SimpleNamespace(free=200*1024**3)), \
                 patch.object(launcher.subprocess,'check_output',return_value='12345'), \
                 patch.object(launcher.subprocess,'run') as run:
                with self.assertRaises(RuntimeError): launcher.launch(file)
                run.assert_not_called()

    def test_failed_regression_cannot_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            file=self.fixture(Path(directory))
            with patch.object(launcher.shutil,'disk_usage',return_value=SimpleNamespace(free=200*1024**3)), \
                 patch.object(launcher.subprocess,'check_output',return_value=''), \
                 patch.object(launcher.subprocess,'run',return_value=SimpleNamespace(returncode=1,stdout='',stderr='test failure')) as run:
                with self.assertRaises(RuntimeError): launcher.launch(file)
                self.assertEqual(run.call_count,1)
                self.assertIn('unittest',run.call_args.args[0])
