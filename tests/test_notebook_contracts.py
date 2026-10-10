"""Static contract failures and complete editorial notebook preservation."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('check_notebook_contracts', ROOT/'scripts/check_notebook_contracts.py')
contracts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contracts)


class NotebookContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = 'notebooks/day-01/01_fixture.ipynb'
        self.prompt = dict(cell_type='markdown', id='prompt', metadata={}, source=[
            '## Exercise: predict a value\n', '### Prediction\n',
            'Will adding one increase the result?\n', '### Checkpoint\n', 'Explain the changed result.\n'])
        self.attempt = dict(cell_type='code', id='attempt', metadata={'tags':['learner-exercise']},
                            source=['prediction = ""\n'], outputs=[], execution_count=None)
        self.reference = dict(cell_type='code', id='reference', metadata={'tags':['reference']},
                              source=['x = 1\n', 'print(x + 1)\n'], outputs=[
                                  dict(output_type='stream', name='stdout', text=['2\n'])], execution_count=2)
        self.original = dict(cells=[self.prompt, self.attempt, self.reference],
                             metadata={'kernelspec': {'name':'course', 'display_name':'CPU'}},
                             nbformat=4, nbformat_minor=5)
        marker = dict(cell_type='markdown', id='marker', metadata={}, source=['### Reference solution\n'])
        explanation = dict(cell_type='markdown', id='explanation', metadata={}, source=[
            '### Reference explanation\n', 'Adding one increases this scalar; it does not establish a learned model result.\n'])
        self.notebook = copy.deepcopy(dict(self.original, cells=[self.prompt, self.attempt, marker, self.reference, explanation]))
        self.manifest = {'notebooks':[{'path':self.path}]}
        self.rebuild_baseline()

    def write(self, path, notebook):
        target = self.root/path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(notebook, ensure_ascii=False))

    def rebuild_baseline(self):
        self.write(self.path, self.original)
        self.baseline = dict(base_commit='a'*40, notebooks={self.path: dict(
            sha256=contracts.hashlib.sha256((self.root/self.path).read_bytes()).hexdigest(),
            code_cells=len(contracts.code_cells(self.original)),
            preserved_code_sha256=contracts.object_sha256(contracts.normalized_baseline_code(self.original)),
            notebook_metadata_sha256=contracts.object_sha256(self.original['metadata']))})
        self.originals = {self.path: copy.deepcopy(self.original)}
        pair = dict(exercise_id='one', original_cell_index=0, original_cell_id='prompt',
                    prompt_cell_id='prompt', reference_heading_cell_id='marker', reference_code_ordinal=1,
                    reference_code_cell_id=self.original['cells'][2].get('id'),
                    reference_code_sha256=contracts.object_sha256(self.original['cells'][2]),
                    explanation_cell_id='explanation', kind='computation',
                    prompt_source_sha256=contracts.object_sha256(self.notebook['cells'][0]['source']),
                    reference_heading_source_sha256=contracts.object_sha256(self.notebook['cells'][2]['source']),
                    explanation_source_sha256=contracts.object_sha256(self.notebook['cells'][4]['source']),
                    audited_basis='The existing complete code adds one to a fixed scalar and prints its value.',
                    source_prompt_excerpt='Will adding one increase the result?',
                    allowed_intervening_cell_ids=['attempt'],
                    allowed_intervening_cell_indices=[1],
                    allowed_intervening_reasons={'attempt':'Preserve the optional prediction attempt before revealing the answer.'})
        self.review = dict(schema_version=1, status='reviewed', base_commit='a'*40,
                           hash_recipe=contracts.HASH_RECIPE, notebooks=[dict(
                               path=self.path, baseline_code_sha256=self.baseline['notebooks'][self.path]['preserved_code_sha256'],
                               baseline_full_code_sha256=contracts.object_sha256(contracts.code_cells(self.original)),
                               baseline_metadata_sha256=self.baseline['notebooks'][self.path]['notebook_metadata_sha256'],
                               original_markdown=[dict(original_cell_index=0, original_cell_id='prompt',
                                                       source_sha256=contracts.object_sha256(self.prompt['source']),
                                                       reviewed_role='exercise/checkpoint',
                                                       review_note='The original question is retained and paired with the existing complete scalar reference.')],
                               reviewed_prompt_cell_ids=['prompt'], pairs=[pair], heading_exemptions=[])])

    def check(self):
        self.write(self.path, self.notebook)
        return contracts.check_contracts(self.root, self.manifest, self.baseline, self.review, self.originals)

    def messages(self, result):
        return '\n'.join(issue['message'] for issue in result['problems'])

    def test_reviewed_prediction_attempt_reference_and_explanation_pass(self):
        result = self.check()
        self.assertEqual(result['problems'], [])
        self.assertEqual(result['preservation_authorities']['independent-git-objects'], 1)
        self.assertEqual(result['reference_pairs'], 1)
        self.assertTrue(any('do not prove semantic correctness' in limit for limit in result['limits']))

    def test_missing_or_nonadjacent_reference_fails(self):
        self.notebook['cells'].pop(2)
        self.assertIn('Invalid or missing pairing identity', self.messages(self.check()))
        self.notebook['cells'].insert(2, dict(cell_type='markdown', id='marker', metadata={}, source=['### Reference solution\n']))
        self.notebook['cells'].insert(3, dict(cell_type='markdown', id='gap', metadata={}, source=['An unexplained gap.\n']))
        self.assertIn('immediately before', self.messages(self.check()))

    def test_empty_and_unfinished_existing_code_cannot_be_reference(self):
        for source, expected in [(['# No calculation here.\n'], 'empty'), (['answer = ...\n'], 'unfinished')]:
            with self.subTest(source=source):
                self.original['cells'][2]['source'] = source
                self.notebook['cells'][3]['source'] = source
                self.rebuild_baseline()
                self.assertIn(expected, self.messages(self.check()))

    def test_reference_answer_before_prediction_fails(self):
        self.notebook['cells'][0]['source'].insert(0, '### Reference solution\nThe answer is two.\n')
        self.assertIn('precedes the prediction', self.messages(self.check()))

    def test_moving_code_even_with_same_sources_fails_preservation(self):
        self.notebook['cells'][1], self.notebook['cells'][3] = self.notebook['cells'][3], self.notebook['cells'][1]
        result = self.check()
        self.assertIn('Code-cell content, order', self.messages(result))
        self.assertIn('Complete code objects differ', self.messages(result))

    def test_outputs_execution_counts_and_cell_metadata_are_preserved(self):
        for field, value in [('outputs', []), ('execution_count', None), ('metadata', {'tags':['changed']})]:
            with self.subTest(field=field):
                saved = copy.deepcopy(self.notebook['cells'][3][field])
                self.notebook['cells'][3][field] = value
                # A rebinding of the ledger cannot excuse changed original code objects.
                self.review['notebooks'][0]['pairs'][0]['reference_code_sha256'] = contracts.object_sha256(self.notebook['cells'][3])
                self.assertIn('Complete code objects differ', self.messages(self.check()))
                self.notebook['cells'][3][field] = saved

    def test_notebook_metadata_changes_fail(self):
        self.notebook['metadata']['kernelspec']['name'] = 'gpu'
        self.assertIn('Notebook metadata changed', self.messages(self.check()))

    def test_missing_id_and_explicit_null_are_distinct_full_code_objects(self):
        self.original['cells'][2].pop('id')
        self.notebook['cells'][3].pop('id')
        self.rebuild_baseline()
        self.assertEqual(self.check()['problems'], [])
        before = contracts.object_sha256(contracts.normalized_baseline_code(self.notebook))
        self.notebook['cells'][3]['id'] = None
        self.assertEqual(before, contracts.object_sha256(contracts.normalized_baseline_code(self.notebook)))
        self.assertIn('missing keys remain significant', self.messages(self.check()))

    def test_unknown_routes_and_uncovered_visible_prompts_fail(self):
        row = copy.deepcopy(self.review['notebooks'][0]); row['path'] = 'notebooks/day-01/unknown.ipynb'
        self.review['notebooks'].append(row)
        self.notebook['cells'].append(dict(cell_type='markdown', id='unreviewed', metadata={}, source=['## Interpretation checkpoint\nExplain a different result.\n']))
        messages = self.messages(self.check())
        self.assertIn('unknown/unregistered', messages)
        self.assertIn('Visible prompt heading has no reviewed pair', messages)

    def test_original_heading_disposition_and_classified_exercises_need_review(self):
        old_extra = dict(cell_type='markdown', id='older-prompt', metadata={}, source=['## Second checkpoint\nAn operational title naming a stored boundary.\n'])
        self.original['cells'].append(old_extra)
        self.rebuild_baseline()
        self.review['notebooks'][0]['original_markdown'].append(dict(original_cell_index=3, original_cell_id='older-prompt',
                                                                   source_sha256=contracts.object_sha256(old_extra['source']),
                                                                   reviewed_role='narrative/mechanism-context',
                                                                   review_note='This names an operational boundary and contains no learner exercise.'))
        self.assertIn('Original prompt heading has no pair', self.messages(self.check()))
        self.review['notebooks'][0]['heading_exemptions'] = [dict(original_cell_index=3, heading='Second checkpoint',
                                                                 reason='Review identifies this as an operational title, with no exercise instruction.')]
        self.assertEqual(self.check()['problems'], [])
        self.review['notebooks'][0]['original_markdown'][-1]['reviewed_role'] = 'exercise/checkpoint'
        self.assertIn('classified as an exercise/checkpoint must map', self.messages(self.check()))

    def test_ignored_checkpoint_copies_require_unchanged_entire_files(self):
        path = 'notebooks/day-01/.ipynb_checkpoints/old.ipynb'
        self.write(path, self.original)
        self.baseline['notebooks'][path] = dict(self.baseline['notebooks'][self.path],
            sha256=contracts.hashlib.sha256((self.root/path).read_bytes()).hexdigest())
        self.assertEqual(self.check()['problems'], [])
        changed = copy.deepcopy(self.original); changed['cells'][0]['source'].append('Changed Markdown.\n')
        self.write(path, changed)
        self.assertIn('Ignored checkpoint copy changed', self.messages(self.check()))

    def test_original_markdown_hashes_bind_raw_source_representation(self):
        self.review['notebooks'][0]['original_markdown'][0]['source_sha256'] = contracts.object_sha256(contracts.cell_source(self.prompt))
        self.assertIn('Original Markdown inventory', self.messages(self.check()))

    def test_current_prompt_and_explanation_are_bound_to_reviewed_sources(self):
        for index, name in [(0,'prompt'), (2,'reference_heading'), (4,'explanation')]:
            with self.subTest(name=name):
                self.notebook['cells'][index]['source'].append('Changed after the semantic review.\n')
                self.assertIn(f'Reviewed {name} source hash disagrees', self.messages(self.check()))
                self.notebook['cells'][index]['source'].pop()

    def test_missing_intervening_code_id_uses_reviewed_index_identity(self):
        self.original['cells'][1].pop('id')
        self.notebook['cells'][1].pop('id')
        self.rebuild_baseline()
        pair = self.review['notebooks'][0]['pairs'][0]
        pair['allowed_intervening_cell_ids'] = [None]
        pair['allowed_intervening_reasons'] = {'index:1':'Preserve the original optional prediction attempt with its absent cell ID.'}
        self.assertEqual(self.check()['problems'], [])

    def test_fenced_headings_and_index_ellipsis_are_not_false_scaffolds(self):
        source = '```python\n### Prediction\n```\n<!--\n### Prediction\n-->\n### Checkpoint\n'
        self.assertEqual(contracts.visible_headings(source), [(3,'Checkpoint',source.index('### Checkpoint'))])
        self.assertEqual(contracts.reference_problems(dict(cell_type='code', source='x = tensor[..., 0]\n', metadata={})), [])

    def test_cli_missing_ledger_returns_failure_with_semantic_limits(self):
        completed = subprocess.run([sys.executable, str(ROOT/'scripts/check_notebook_contracts.py'),
                                    '--root', str(self.root), '--json'], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 1)
        result = json.loads(completed.stdout)
        self.assertEqual(result['problems'][0]['check'], 'input')
        self.assertTrue(result['limits'])


if __name__ == '__main__':
    unittest.main()
