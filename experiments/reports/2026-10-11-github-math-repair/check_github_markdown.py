"""Compare book TeX with the actual GitHub Markdown API's math elements.

This records GitHub's Markdown preprocessing, not browser SVG/layout. Run with
a fresh --output directory; --source-prefix can point to a saved source tree.
No authentication or repository mutation is involved in these render requests.
"""
import argparse
from datetime import datetime, timezone
import difflib
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import urllib.request


ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
MARKER = 'DONGXI-MATH-AUDIT-FILE-'


class MathElements(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.file_index = None
        self.heading = None
        self.math = None
        self.elements = {}

    def handle_starttag(self, tag, attrs):
        if tag == 'h1':
            self.heading = []
        if tag == 'math-renderer':
            self.math = {'display': 'js-display-math' in dict(attrs).get('class', ''),
                         'body': []}

    def handle_data(self, data):
        if self.heading is not None:
            self.heading.append(data)
        if self.math is not None:
            self.math['body'].append(data)

    def handle_endtag(self, tag):
        if tag == 'h1' and self.heading is not None:
            title = ''.join(self.heading).strip()
            if title.startswith(MARKER):
                self.file_index = int(title[len(MARKER):])
                self.elements[self.file_index] = []
            self.heading = None
        if tag == 'math-renderer' and self.math is not None:
            body = ''.join(self.math['body']).strip()
            delimiter = '$$' if self.math['display'] else '$'
            if not body.startswith(delimiter) or not body.endswith(delimiter):
                raise ValueError('Unexpected GitHub math payload delimiters')
            tex = body[len(delimiter):-len(delimiter)].strip()
            self.elements[self.file_index].append((self.math['display'], tex))
            self.math = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-prefix', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    baseline = json.loads((REPORT / 'baseline.json').read_text())
    paths = list(baseline['files'])
    groups, current, chars = [], [], 0
    sources = {}
    for index, name in enumerate(paths):
        source = (args.source_prefix / name).read_text()
        sources[name] = hashlib.sha256(source.encode()).hexdigest()
        marked = f'# {MARKER}{index}\n\n{source}\n\n'
        if current and chars + len(marked) > 150_000:
            groups.append(current)
            current, chars = [], 0
        current.append((index, marked))
        chars += len(marked)
    if current:
        groups.append(current)
    requests, rows = [], []
    for group_index, group in enumerate(groups):
        payload = json.dumps({'text': ''.join(s for _, s in group),
                              'mode': 'gfm', 'context': 'dxnlp/dongxi-LLMs'}).encode()
        request = urllib.request.Request(
            'https://api.github.com/markdown', data=payload,
            headers={'Content-Type': 'application/json',
                     'Accept': 'application/vnd.github+json',
                     'User-Agent': 'Dongxi-LLMs-math-check'})
        with urllib.request.urlopen(request, timeout=45) as response:
            html = response.read()
            headers = {k: response.headers.get(k) for k in
                       ['Date', 'Content-Type', 'X-GitHub-Request-Id']}
        html_path = args.output / f'github-response-{group_index:02d}.html'
        html_path.write_bytes(html)
        requests.append({'utc': datetime.now(timezone.utc).isoformat(),
                         'endpoint': request.full_url, 'http_status': 200,
                         'request_sha256': hashlib.sha256(payload).hexdigest(),
                         'response_sha256': hashlib.sha256(html).hexdigest(),
                         'response_file': html_path.name, 'headers': headers,
                         'file_indexes': [i for i, _ in group]})
        parsed = MathElements()
        parsed.feed(html.decode())
        for index, _ in group:
            name = paths[index]
            expected = [(e['display'], e['tex'])
                        for e in baseline['files'][name]['expressions']]
            actual = parsed.elements[index]
            differences = []
            for kind, a, b, c, d in difflib.SequenceMatcher(
                    a=expected, b=actual, autojunk=False).get_opcodes():
                if kind != 'equal':
                    differences.append({'kind': kind,
                                        'expected_indexes': [a, b],
                                        'rendered_indexes': [c, d],
                                        'expected': expected[a:b],
                                        'rendered': actual[c:d]})
            rows.append({'path': name, 'source_sha256': sources[name],
                         'expected_expressions': len(expected),
                         'github_math_elements': len(actual),
                         'exact_tex_and_order_match': actual == expected,
                         'differences': differences})
    result = {'utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Actual GitHub GFM preprocessing; not browser SVG/layout or GitHub deployed MathJax version.',
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'baseline_sha256': hashlib.sha256((REPORT / 'baseline.json').read_bytes()).hexdigest(),
              'requests': requests, 'files': rows,
              'expected_expressions': sum(row['expected_expressions'] for row in rows),
              'github_math_elements': sum(row['github_math_elements'] for row in rows),
              'files_with_exact_tex_and_order': sum(row['exact_tex_and_order_match'] for row in rows),
              'files_with_mismatches': sum(not row['exact_tex_and_order_match'] for row in rows)}
    (args.output / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['expected_expressions', 'github_math_elements',
                                           'files_with_exact_tex_and_order', 'files_with_mismatches']}))
    return bool(result['files_with_mismatches'])


if __name__ == '__main__':
    raise SystemExit(main())
