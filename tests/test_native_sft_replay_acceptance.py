"""CPU-only archive/binding controls; never calls the Spark launch route."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('fixed_native_sft_replay',
    ROOT/'scripts/run_native_sft_replay_acceptance.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class NativeSFTReplayControls(unittest.TestCase):
    def test_serialized_tensor_entries_exclude_unpickled_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'snapshot.pt'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('snapshot/data.pkl', b'not even a valid pickle')
                archive.writestr('snapshot/data/0', b'actual tensor bytes')
                archive.writestr('snapshot/data/not-an-index', b'ignored')
            self.assertEqual(runner.tensor_bytes(path), {'0':dict(bytes=19,
                sha256=hashlib.sha256(b'actual tensor bytes').hexdigest())})

    def test_missing_tensor_entries_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'snapshot.pt'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('snapshot/data.pkl', b'metadata only')
            with self.assertRaisesRegex(ValueError, 'tensor entries'):
                runner.tensor_bytes(path)

    def test_bindings_cover_fixed_driver_and_science(self):
        bindings = runner.bindings()
        self.assertIn('scripts/run_chapter09_spark_sft.py', bindings)
        self.assertIn('scripts/run_native_sft_replay_acceptance.py', bindings)
        self.assertIn('experiments/specs/2026-10-05-native-sft-replay.md', bindings)
        self.assertTrue(all(row['bytes'] > 0 and len(row['sha256']) == 64
                            for row in bindings.values()))

    def test_exclusive_retention_preserves_previous_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'evidence.json'
            runner.retain(path, {'attempt':1})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                runner.retain(path, {'attempt':2})
            self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
