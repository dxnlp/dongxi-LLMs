"""Check the book's conservative GitHub math conventions, without a network.

This is a source linter, not GitHub's renderer or a complete TeX parser.
Run: python3 scripts/check_book_math.py
Export extracted expressions for an optional TeX parser: add --json.
"""
import argparse
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def prose_only(source):
    """Blank literal code while preserving offsets and math code fences."""
    lines = source.splitlines(keepends=True)
    fence = None
    math_fence = False
    for i, line in enumerate(lines):
        match = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line.rstrip('\n'))
        if match and fence is None:
            fence = match[1]
            math_fence = match[2].strip() == 'math'
            lines[i] = ('$$' if math_fence else '') + ' ' * (len(line.rstrip('\n')) - (2 if math_fence else 0)) + ('\n' if line.endswith('\n') else '')
        elif fence is not None:
            if match and match[1][0] == fence[0] and len(match[1]) >= len(fence) and not match[2].strip():
                lines[i] = ('$$' if math_fence else '') + ' ' * (len(line.rstrip('\n')) - (2 if math_fence else 0)) + ('\n' if line.endswith('\n') else '')
                fence = None
            elif not math_fence:
                lines[i] = re.sub(r'[^\n]', ' ', line)
    text = ''.join(lines)
    # GitHub's $`...`$ inline math is not a literal inline code span.
    return re.sub(r'(?<!\$)(`+)([^`]*?)\1(?!\$)',
                  lambda m: re.sub(r'[^\n]', ' ', m[0]), text)


def inspect_markdown(source):
    """Return (issues, expressions), with one-based source line numbers."""
    text = prose_only(source)
    issues, expressions = [], []

    def report(offset, message):
        issues.append({'line': text.count('\n', 0, offset) + 1, 'message': message})

    for match in re.finditer(r'(?<!\\)\\[\[\]()]', text):
        report(match.start(), 'Use dollar math delimiters, not legacy TeX delimiters')

    opened = None
    for match in re.finditer(r'(?<!\\)\$\$|(?<!\\)\$', text):
        if opened is None:
            opened = match
            continue
        if match[0] != opened[0]:
            report(match.start(), 'Mixed inline/display math delimiters')
            continue
        body = text[opened.end():match.start()]
        if body.startswith('`') and body.endswith('`'):
            body = body[1:-1]
        line = text.count('\n', 0, opened.start()) + 1
        expressions.append({'line': line, 'display': opened[0] == '$$', 'tex': body.strip()})
        if not body.strip():
            report(opened.start(), 'Empty math expression')
        if '\n\n' in body:
            report(opened.start(), 'Blank paragraph inside a math expression')
        if re.search(r'\\operatorname\b', body):
            report(opened.start(), 'Avoid operatorname: use upright names with mathrm')
        balance = 0
        for brace in re.finditer(r'(?<!\\)[{}]', body):
            balance += 1 if brace[0] == '{' else -1
            if balance < 0:
                break
        if balance != 0:
            report(opened.start(), 'Unbalanced TeX braces')
        stack = []
        for env in re.finditer(r'\\(begin|end)\{([^}]+)\}', body):
            if env[1] == 'begin':
                stack.append(env[2])
            elif not stack or stack.pop() != env[2]:
                report(opened.start(), 'Mismatched TeX environment')
        if stack:
            report(opened.start(), 'Unclosed TeX environment')
        opened = None
    if opened is not None:
        report(opened.start(), 'Unclosed math delimiter')
    return issues, expressions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    files = []
    for path in sorted((ROOT / 'book').rglob('*.md')):
        issues, expressions = inspect_markdown(path.read_text())
        files.append({'path': str(path.relative_to(ROOT)), 'issues': issues,
                      'expressions': expressions})
    errors = sum(len(f['issues']) for f in files)
    if args.json:
        print(json.dumps(files, indent=2))
    else:
        for f in files:
            for issue in f['issues']:
                print(f"{f['path']}:{issue['line']}: {issue['message']}")
        total = sum(len(f['expressions']) for f in files)
        print(f'{len(files)} Markdown files, {total} math expressions, {errors} issues')
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
