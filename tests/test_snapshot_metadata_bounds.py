"""Role-specific contract sizing never expands ordinary receipt admission."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dongxi_llms.snapshot_io_budget import read_bounded_json


class SnapshotMetadataBoundsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='dongxi-metadata-bounds-')
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_default_receipt_bound_is_still_64_kib(self):
        path = self.root / 'contract.json'
        path.write_text(json.dumps({'shape_inventory': 'x' * 100_000}))
        with self.assertRaisesRegex(ValueError, 'bounded'):
            read_bounded_json(path)
        value, observed = read_bounded_json(path, maximum=1024**2)
        self.assertEqual(len(value['shape_inventory']), 100_000)
        self.assertEqual(observed['bytes'], path.stat().st_size)
        self.assertEqual(observed['reader'], 'bounded-nonblocking-no-follow-regular')

    def test_explicit_contract_still_rejects_more_than_one_mib(self):
        path = self.root / 'large.json'
        path.write_text(json.dumps({'x': 'x' * 1024**2}))
        with self.assertRaisesRegex(ValueError, 'bounded'):
            read_bounded_json(path, maximum=1024**2)

    def test_invalid_bound_rejected_before_open(self):
        for maximum in (0, -1, True, 1.0, None, 1024**2 + 1):
            with self.subTest(maximum=maximum), patch('os.open', side_effect=AssertionError('no read')):
                with self.assertRaises(ValueError):
                    read_bounded_json(self.root / 'absent.json', maximum=maximum)

    def test_explicit_bound_does_not_relax_duplicate_or_nonfinite_checks(self):
        path = self.root / 'invalid.json'
        for raw in ('{"x":1,"x":2}', '{"x":NaN}'):
            path.write_text(raw)
            with self.assertRaises(ValueError):
                read_bounded_json(path, maximum=1024**2)

    def test_explicit_bound_does_not_follow_symlink_or_block_on_fifo(self):
        path = self.root / 'regular.json'
        path.write_text('{"x":1}')
        link = self.root / 'link.json'
        link.symlink_to(path)
        with self.assertRaises(OSError):
            read_bounded_json(link, maximum=1024**2)
        fifo = self.root / 'fifo'
        os.mkfifo(fifo)
        with self.assertRaisesRegex(ValueError, 'regular'):
            read_bounded_json(fifo, maximum=1024**2)


if __name__ == '__main__':
    unittest.main()
