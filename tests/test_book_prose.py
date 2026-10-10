"""Meaningful scope and offset checks for the conservative book prose scanner."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('check_book_prose', ROOT/'scripts/check_book_prose.py')
prose = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prose)


class BookProseTests(unittest.TestCase):
    def test_join_directions_decimals_and_original_offsets(self):
        source = 'all20 updates; 0.45correct; in61.636 seconds; 11controlled; exit0.\nNext2.'
        rows = prose.inspect_markdown(source)
        self.assertEqual(len(rows), 6)
        repaired = source
        for row in reversed(rows):
            offset = row['insertion_offset']
            self.assertEqual(source[row['start']:row['end']], row['match'])
            repaired = repaired[:offset] + row['replacement'] + repaired[offset:]
        self.assertEqual(repaired, 'all 20 updates; 0.45 correct; in 61.636 seconds; 11 controlled; exit 0.\nNext 2.')
        self.assertEqual(rows[-1]['line'], 2)

    def test_valid_names_precision_units_ids_and_versions(self):
        source = 'Qwen3 Qwen3-0.6B GPT-2 BF16 FP32 sm_121 utf8 25GiB 16MiB 61.636s v2.1.0 ID2 G8 Full400 LoRA400. float64 int64 BF16-loaded DeepSeek-R1 SHA256 ARM64 L2 24th 50M 5.811e-7 5e-7 1e-3 c1899de 2d'
        self.assertEqual(prose.inspect_markdown(source), [])
        self.assertEqual(len(prose.inspect_markdown('rank8 and Chapter13')), 2)

    def test_unit_ordinal_and_exponent_inside_a_true_join(self):
        source = 'the25GiB same24th are5.811e-7'
        rows = prose.inspect_markdown(source)
        self.assertEqual(len(rows), 3)
        repaired = source
        for row in reversed(rows):
            offset = row['insertion_offset']
            repaired = repaired[:offset] + ' ' + repaired[offset:]
        self.assertEqual(repaired, 'the 25GiB same 24th are 5.811e-7')

    def test_markdown_mask_keeps_link_label_offsets(self):
        source = ('```python\nall20 = 1\n```\n~~~math\na20\n~~~\n'
                  '`from4.2` $x2 + y3$ $$z4$$ $`q5`$ '
                  '[all20](https://site.test/from4.2(a5)) '
                  'https://site.test/0.45correct\n'
                  '[ref2]: https://site.test/a3\n[visible4][ref2]\n'
                  '<span data-x="a3">text5</span><!-- hidden6 -->')
        mask = prose.visible_prose(source)
        self.assertEqual(len(mask), len(source))
        self.assertEqual([i for i,c in enumerate(mask) if c == '\n'],
                         [i for i,c in enumerate(source) if c == '\n'])
        rows = prose.inspect_markdown(source)
        self.assertEqual([r['token'] for r in rows], ['all20', 'visible4', 'text5'])
        self.assertEqual(source[rows[0]['insertion_offset']-3:rows[0]['insertion_offset']], 'all')

    def test_unmatched_dollar_does_not_hide_later_findings(self):
        self.assertEqual(len(prose.inspect_markdown('Costs $25.\n\nall20 updates.')), 1)

    def test_comparison_angle_does_not_hide_following_prose(self):
        source = '9 < 11 < 20; all20 updates.\n<span>text2</span>'
        self.assertEqual([r['token'] for r in prose.inspect_markdown(source)], ['all20', 'text2'])

    def test_narrative_candidates_are_chapter_only(self):
        source = 'DXI-17 measured on 2026-10-05; the learner approved /home/dongxi/run-01.\n'
        self.assertFalse(any(r['group'] == 'narrative' for r in prose.inspect_markdown(source)))
        rules = {r['rule'] for r in prose.inspect_markdown(source, chapter=True)}
        self.assertEqual(rules, {'task-or-run-id', 'policy-or-session-date', 'session-status', 'machine-path'})

    def test_scan_counts_exact_context_exemption_and_appendix_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for section in prose.BOOK_SECTIONS:
                (root/section).mkdir()
            (root/'chapters/01-example.md').write_text('all20 all20\nMeasured on 2026-10-05.\n')
            (root/'appendices/d-example.md').write_text('DXI-17 approved on 2026-10-05.\n')
            exemptions = [dict(path='book/chapters/01-example.md', rule='policy-or-session-date',
                               match='2026-10-05', context='Measured on 2026-10-05.', reason='Exact evidence caption.')]
            result = prose.scan_book(root, exemptions=exemptions)
            self.assertEqual(result['counts']['mechanical'], {'matches': 2, 'affected_lines': 1})
            self.assertEqual(result['counts']['narrative']['matches'], 0)
            self.assertEqual(result['exempted_counts']['matches'], 1)
            self.assertEqual(result['unused_exemptions'], [])
            self.assertEqual(len(result['files']), 2)

    def test_rejects_blanket_or_unexplained_exemptions(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'exemptions.json'
            for entry in [dict(path='book/chapters/*.md', rule='task-or-run-id', match='DXI-17',
                               context='DXI-17', reason=''),
                          dict(path='book/chapters/01-example.md', rule='task-or-run-id',
                               match='DXI-17', context='Other text', reason='Evidence')]:
                path.write_text(json.dumps(dict(schema_version=1, exemptions=[entry])))
                with self.assertRaises(ValueError):
                    prose.load_exemptions(path)

    def test_cli_distinguishes_review_mode_strict_mode_and_missing_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'chapters').mkdir()
            (root/'chapters/01-example.md').write_text('The learner requests approval.\n')
            command = [sys.executable, str(ROOT/'scripts/check_book_prose.py'),
                       '--book-root', str(root), '--no-exemptions', '--json']
            review = subprocess.run(command, capture_output=True, text=True)
            strict = subprocess.run([*command, '--strict-narrative'], capture_output=True, text=True)
            self.assertEqual(review.returncode, 0)
            self.assertEqual(strict.returncode, 1)
            self.assertEqual(json.loads(review.stdout)['counts']['narrative']['matches'], 2)
            (root/'chapters/01-example.md').write_text('all20 updates.\n')
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 1)
            (root/'chapters/01-example.md').unlink()
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 1)


if __name__ == '__main__':
    unittest.main()
