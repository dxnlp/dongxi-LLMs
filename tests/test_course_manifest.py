"""Course registry rejects structural mistakes without running models."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dongxi_llms.course_manifest import (CourseManifestError, load_manifest,
                                        select_notebooks, validate_manifest)

ROOT = Path(__file__).resolve().parents[1]


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.document = json.loads((ROOT/'docs/course_manifest.json').read_text())

    def test_schema_and_registered_routes(self):
        self.assertEqual(validate_manifest(self.document), [])
        self.assertEqual(len(self.document['chapters']), 15)
        self.assertEqual(len(self.document['days']), 28)
        self.assertEqual(len(load_manifest(ROOT, check_files=False)['notebooks']),
                         len(self.document['notebooks']))

    def test_unknown_fields_and_boolean_version(self):
        for change in ({'mystery': True}, {'schema_version': True}, {'scope': ''}):
            document = copy.deepcopy(self.document)
            document.update(change)
            self.assertTrue(validate_manifest(document))

    def test_missing_and_duplicate_days(self):
        self.document['days'].pop()
        self.assertTrue(validate_manifest(self.document))
        self.document['days'].append(copy.deepcopy(self.document['days'][0]))
        self.assertTrue(validate_manifest(self.document))

    def test_day_chapter_path_disagree(self):
        self.document['notebooks'][0]['chapter'] = 15
        self.assertTrue(any('mismatch' in p for p in validate_manifest(self.document)))
        self.document['notebooks'][0]['day'] = False
        self.assertTrue(validate_manifest(self.document))

    def test_unknown_and_duplicate_notebooks(self):
        self.document['notebooks'].append(copy.deepcopy(self.document['notebooks'][0]))
        self.assertTrue(any('Duplicate' in p for p in validate_manifest(self.document)))

    def test_unsafe_path(self):
        for path in ('/tmp/escape.ipynb', '../escape.ipynb', 'notebooks/../escape.ipynb',
                     'notebooks/day-01/null\x00.ipynb', 'notebooks/day-01/control\x1b.ipynb'):
            document = copy.deepcopy(self.document)
            document['notebooks'][0]['path'] = path
            self.assertTrue(any('Unsafe' in p for p in validate_manifest(document)))
            self.assertTrue(validate_manifest(document, ROOT))

    def test_dependency_cycles_and_unknown(self):
        first, second = self.document['notebooks'][:2]
        first['requires'], second['requires'] = [second['path']], [first['path']]
        self.assertTrue(any('cycle' in p for p in validate_manifest(self.document)))
        first['requires'] = ['notebooks/day-99/missing.ipynb']
        self.assertTrue(any('Unknown dependency' in p for p in validate_manifest(self.document)))

    def test_optional_does_not_count_as_core_day(self):
        for row in self.document['notebooks']:
            if row['day'] == 28:
                row['lane'] = 'optional'
        self.assertTrue(any('Day 28 has no core' in p for p in validate_manifest(self.document)))

    def test_extension_identity_is_validated(self):
        row = next(n for n in self.document['notebooks'] if n['lane']=='extension')
        row['improvement'] = 'DXI-99'
        self.assertTrue(any('known improvement' in p for p in validate_manifest(self.document)))

    def test_empty_unknown_and_intersecting_selectors(self):
        for arguments in ({'days': []}, {'days': [29]}, {'days': 1}, {'paths': [{}]}, {'paths': ['missing']},
                          {'lane': 'maybe'}, {'days': [1], 'paths': [self.document['notebooks'][-1]['path']]}):
            with self.assertRaises(CourseManifestError):
                select_notebooks(self.document, **arguments)
        rows = select_notebooks(self.document, days=[1], lane='extension')
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['improvement'], 'DXI-01')

    def test_missing_empty_unregistered_and_metadata(self):
        with tempfile.TemporaryDirectory(prefix='dongxi-manifest-test-') as directory:
            root = Path(directory)
            path = self.document['notebooks'][0]['path']
            file = root/path
            file.parent.mkdir(parents=True)
            file.write_text(json.dumps({'cells':[
                {'cell_type':'markdown','source':'A preview ![x](figure.png)'},
                {'cell_type':'code','source':'...'}], 'metadata': {'chapter':15}}))
            unknown = file.parent/'unregistered.ipynb'
            unknown.write_text('{}')
            problems = validate_manifest(self.document, root)
            self.assertTrue(any('Missing/empty' in p for p in problems))
            self.assertTrue(any('placeholder' in p for p in problems))
            self.assertTrue(any('Unregistered' in p for p in problems))
            self.assertTrue(any('metadata/chapter' in p for p in problems))

    def test_symlink_escape(self):
        with tempfile.TemporaryDirectory(prefix='dongxi-manifest-test-') as directory:
            root = Path(directory)/'repo'
            root.mkdir()
            outside = Path(directory)/'outside.ipynb'
            outside.write_text('{}')
            file = root/self.document['notebooks'][0]['path']
            file.parent.mkdir(parents=True)
            file.symlink_to(outside)
            self.assertTrue(any('escapes repository' in p for p in validate_manifest(self.document, root)))

    def test_integrity_json_retains_malformed_notebook_diagnostics(self):
        from scripts.check_course_integrity import main
        target = ROOT/self.document['notebooks'][0]['path']
        original = Path.read_text
        def broken(path, *args, **kwargs):
            return '{"cells":' if path == target else original(path,*args,**kwargs)
        output = io.StringIO()
        with patch.object(Path,'read_text',broken), patch('sys.argv',['check','--json']), \
             redirect_stdout(output), self.assertRaises(SystemExit) as result:
            main()
        self.assertEqual(result.exception.code, 1)
        self.assertTrue(any('Invalid notebook' in p for p in json.loads(output.getvalue())['problems']))


if __name__ == '__main__':
    unittest.main()
