"""Regression checks for reader-facing GitHub math source conventions."""
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('check_book_math', ROOT / 'scripts/check_book_math.py')
math_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(math_check)


class BookMathTests(unittest.TestCase):
    def test_book_has_no_known_rendering_hazards(self):
        for path in (ROOT / 'book').rglob('*.md'):
            with self.subTest(path=path):
                self.assertEqual(math_check.inspect_markdown(path.read_text())[0], [])

    def test_reported_macro_and_legacy_delimiters(self):
        issues, _ = math_check.inspect_markdown(r'\(x\) $`\operatorname{softmax}(x)`$')
        self.assertEqual(len(issues), 3)

    def test_code_is_not_treated_as_math(self):
        text = '```bash\necho "$HOME"\n```\n`\\[` and `$value`\n$`x`$'
        issues, expressions = math_check.inspect_markdown(text)
        self.assertEqual(issues, [])
        self.assertEqual([e['tex'] for e in expressions], ['x'])

    def test_github_delimiter_styles(self):
        text = '$`x_i`$\n\n```math\nx^2\n```\n\n```math\ny^2\n```\n'
        issues, expressions = math_check.inspect_markdown(text)
        self.assertEqual(issues, [])
        self.assertEqual([e['tex'] for e in expressions], ['x_i', 'x^2', 'y^2'])

    def test_malformed_math(self):
        for text in ['$x', '$$x$', r'$\frac{x}{y$', r'$\begin{matrix}x\end{cases}$']:
            with self.subTest(text=text):
                self.assertTrue(math_check.inspect_markdown(text)[0])

    def test_escaped_dollars_and_braces(self):
        issues, expressions = math_check.inspect_markdown(r'Price \$5; $`\{x\}`$')
        self.assertEqual(issues, [])
        self.assertEqual(len(expressions), 1)

    def test_unprotected_display_reports_markdown_heading_hazard(self):
        issues, expressions = math_check.inspect_markdown('$$\nL_{mean}\n=\n\\frac{a}{b}\n$$')
        self.assertEqual(len(issues), 1)
        self.assertIn('fenced math block', issues[0]['message'])
        self.assertEqual(expressions[0]['tex'], 'L_{mean}\n=\n\\frac{a}{b}')

    def test_protected_display_preserves_operators_and_row_breaks(self):
        tex = 'L\n=\n' + r'\begin{bmatrix}1&2\\3&4\end{bmatrix}'
        issues, expressions = math_check.inspect_markdown('```math\n' + tex + '\n```')
        self.assertEqual(issues, [])
        self.assertEqual(expressions[0]['tex'], tex)

    def test_unprotected_inline_is_reported_even_if_tex_is_valid(self):
        issues, expressions = math_check.inspect_markdown(r'top-$k$; $\{1,2\}$; $q\|p$')
        self.assertEqual(len(issues), 3)
        self.assertEqual([e['tex'] for e in expressions], ['k', r'\{1,2\}', r'q\|p'])

    def test_adjacent_protected_formulas_do_not_become_code_spans(self):
        text = r'$`\|g\|`$ and $`\{1,2\}`$; top-$`k`$; $`r_\phi\,r`$.'
        issues, expressions = math_check.inspect_markdown(text)
        self.assertEqual(issues, [])
        self.assertEqual([e['tex'] for e in expressions],
                         [r'\|g\|', r'\{1,2\}', 'k', r'r_\phi\,r'])
        self.assertEqual(math_check.prose_only(text), text)

    def test_multiple_protected_formulas_across_paragraphs(self):
        text = '$`x`$\n\nText $`y`$.\n\n```math\nz=x+y\n```\n'
        issues, expressions = math_check.inspect_markdown(text)
        self.assertEqual(issues, [])
        self.assertEqual([e['tex'] for e in expressions], ['x', 'y', 'z=x+y'])

    def test_math_syntax_inside_literal_code_stays_literal(self):
        text = '``literal $`x`$`` and `$$x$$`\n\n~~~~python\n$`y`$\n~~~~\n\n$`z`$'
        issues, expressions = math_check.inspect_markdown(text)
        self.assertEqual(issues, [])
        self.assertEqual([e['tex'] for e in expressions], ['z'])

    def test_display_less_than_is_protected_from_html_reparsing(self):
        issues, _ = math_check.inspect_markdown('```math\ny_{<t}\n```')
        self.assertEqual(len(issues), 1)
        self.assertIn('reparses HTML', issues[0]['message'])
        issues, expressions = math_check.inspect_markdown('```math\ny_{\\lt t}\n```')
        self.assertEqual(issues, [])
        self.assertEqual(expressions[0]['tex'], r'y_{\lt t}')

    def test_multiline_row_comment_prevents_continuation_escape(self):
        tex = '\\begin{bmatrix}1&2\\\\\n3&4\\end{bmatrix}'
        issues, _ = math_check.inspect_markdown('```math\n' + tex + '\n```')
        self.assertEqual(len(issues), 1)
        self.assertIn('percent comment', issues[0]['message'])
        safe = tex.replace('\\\\\n', '\\\\%\n')
        issues, expressions = math_check.inspect_markdown('```math\n' + safe + '\n```')
        self.assertEqual(issues, [])
        self.assertEqual(expressions[0]['tex'], safe)

    def test_table_pipe_uses_tex_command_without_changing_other_inline_math(self):
        issues, _ = math_check.inspect_markdown(r'| KL $`D_{KL}(q\|p)`$ | Value |')
        self.assertEqual(len(issues), 1)
        self.assertIn('pipe escapes', issues[0]['message'])
        issues, expressions = math_check.inspect_markdown(r'| KL $`D_{KL}(q\Vert p)`$ | Value |')
        self.assertEqual(issues, [])
        self.assertEqual(expressions[0]['tex'], r'D_{KL}(q\Vert p)')
        self.assertEqual(math_check.inspect_markdown(r'Outside a table: $`q\|p`$.')[0], [])


if __name__ == '__main__':
    unittest.main()
