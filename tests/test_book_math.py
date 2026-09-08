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
        issues, _ = math_check.inspect_markdown(r'\(x\) $\operatorname{softmax}(x)$')
        self.assertEqual(len(issues), 3)

    def test_code_is_not_treated_as_math(self):
        text = '```bash\necho "$HOME"\n```\n`\\[` and `$value`\n$x$'
        issues, expressions = math_check.inspect_markdown(text)
        self.assertEqual(issues, [])
        self.assertEqual([e['tex'] for e in expressions], ['x'])

    def test_github_delimiter_styles(self):
        text = '$`x_i`$\n\n$$\nx^2\n$$\n\n```math\ny^2\n```\n'
        issues, expressions = math_check.inspect_markdown(text)
        self.assertEqual(issues, [])
        self.assertEqual([e['tex'] for e in expressions], ['x_i', 'x^2', 'y^2'])

    def test_malformed_math(self):
        for text in ['$x', '$$x$', r'$\frac{x}{y$', r'$\begin{matrix}x\end{cases}$']:
            with self.subTest(text=text):
                self.assertTrue(math_check.inspect_markdown(text)[0])

    def test_escaped_dollars_and_braces(self):
        issues, expressions = math_check.inspect_markdown(r'Price \$5; $\{x\}$')
        self.assertEqual(issues, [])
        self.assertEqual(len(expressions), 1)


if __name__ == '__main__':
    unittest.main()
