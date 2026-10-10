"""Check visible book prose without changing Markdown or measured evidence.

Mechanical joins are errors. Narrative matches are chapter-only review
candidates; --strict-narrative also makes unresolved candidates fail. JSON
records retain original offsets, source hashes, matches and affected-line counts.
Use --book-root PATH to scan an immutable pre-edit book snapshot.
"""
import argparse
from bisect import bisect_right
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SCANNER_VERSION = '1.1'
BOOK_SECTIONS = ('chapters', 'labs', 'solutions', 'front-matter', 'appendices')
EXCLUSIONS = ['fenced and inline code', 'math', 'HTML comments and attributes',
              'link targets and reference identifiers', 'bare URLs',
              'documented model/precision/unit/version identifiers']
JOIN = re.compile(r'(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])')
TOKEN_CHAR = re.compile(r'[A-Za-z0-9_.-]')
# These are semantic identifiers, not a general exemption for words with digits.
UNITS = r'(?:KiB|MiB|GiB|TiB|KB|MB|GB|TB|B|kB|ns|us|ms|s|Hz|kHz|MHz|GHz|FLOPs|TFLOPs|nats|M|k)'
IDENTIFIERS = re.compile(
    r'(?:Qwen\d*(?:\.\d+)?(?:-[A-Za-z0-9.]+)*|GPT-?\d+(?:\.[\d]+)?'
    r'|DeepSeek-R\d+|(?:BF|FP|INT|UTF|utf|float|int)\d+|sm_\d+|(?:ID|id)\d+'
    r'|G[48]|Full400|full400|LoRA400|Chosen100|DPO100|SHA256|ARM64|L2|2[dD]|3[dD]'
    r'|(?:v\d+)(?:\.\d+)*(?:[a-z]\d*)?'
    r'|\d+(?:\.\d+)?' + UNITS + r')\Z')
NARRATIVE_RULES = {
    'task-or-run-id': re.compile(
        r'\b(?:DXI-\d+|ANIM-[A-Za-z0-9-]+|CAND-[A-Za-z0-9-]+'
        r'|X-[A-Za-z0-9-]+-\d+|run-\d+|recovery\d+|pilot\d+)\b'),
    'policy-or-session-date': re.compile(r'(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)'),
    'session-status': re.compile(
        r'\b(?:the learner|learner mastery|does not advance|'
        r'authoriz\w*|authoris\w*|approv\w*)\b', re.I),
    'machine-path': re.compile(r'/home/dongxi(?:/[^\s<>]*)?'),
}


def _blank(text):
    return re.sub(r'[^\n]', ' ', text)


def visible_prose(source):
    """Return an equally sized mask; every surviving offset addresses source."""
    lines = source.splitlines(keepends=True)
    fence = None
    for index, line in enumerate(lines):
        match = re.match(r'^ {0,3}(`{3,}|~{3,})(.*)$', line.rstrip('\n'))
        if fence is not None:
            lines[index] = _blank(line)
            if (match and match[1][0] == fence[0]
                    and len(match[1]) >= len(fence) and not match[2].strip()):
                fence = None
        elif match:
            fence = match[1]
            lines[index] = _blank(line)
    text = ''.join(lines)
    for pattern in [r'<!--.*?-->', r'(?<!`)(`+)(?!`)(.*?)\1(?!`)',
                    r'<(?:/?[A-Za-z][A-Za-z0-9-]*\b[^>\n]*|![^>\n]*)>']:
        text = re.sub(pattern, lambda match: _blank(match[0]), text, flags=re.S)
    # Mask paired dollar math, including GitHub's dollar/backtick form. Do not
    # hide the remainder of a document after an unmatched currency/dollar sign.
    opened = None
    spans = []
    for match in re.finditer(r'(?<!\\)\$\$|(?<!\\)\$', text):
        if opened is None:
            opened = match
        elif match[0] == opened[0] and '\n\n' not in text[opened.end():match.start()]:
            spans.append((opened.start(), match.end()))
            opened = None
        else:
            opened = match
    for start, end in spans:
        text = text[:start] + _blank(text[start:end]) + text[end:]
    # Keep link labels visible while masking destinations with nested brackets.
    chars = list(text)
    for match in re.finditer(r'\]\(', text):
        start, index, depth = match.end()-1, match.end(), 1
        while index < len(text) and depth and text[index] != '\n':
            if text[index] == '\\':
                index += 2
                continue
            depth += (text[index] == '(') - (text[index] == ')')
            index += 1
        if depth == 0:
            chars[start:index] = _blank(text[start:index])
    text = ''.join(chars)
    for pattern in [r'(?m)^ {0,3}\[[^\]\n]+\]:[^\n]*',
                    r'(?<=\])\[[^\]\n]*\]',
                    r'\b(?:https?://|mailto:)[^\s<>]+']:
        text = re.sub(pattern, lambda match: _blank(match[0]), text)
    return text


def _token(text, offset):
    start = end = offset
    while start and TOKEN_CHAR.fullmatch(text[start-1]):
        start -= 1
    while end < len(text) and TOKEN_CHAR.fullmatch(text[end]):
        end += 1
    return text[start:end].strip('.-')


def identifier_boundary(text, offset, token):
    """Protect a valid unit/exponent even inside a separately glued phrase."""
    if IDENTIFIERS.fullmatch(token):
        return True
    # Unformatted full weight/data hashes and short hexadecimal commit IDs are
    # provenance identifiers. Require both digits and letters, not a pure number.
    if re.fullmatch(r'[a-fA-F0-9]{7,64}', token) and re.search('[a-fA-F]', token) and re.search(r'\d', token):
        return True
    # Preserve BF16-loaded, unit-L2 and similar compounds without permitting
    # ordinary words such as all20 or answer19.
    segment_start = text.rfind('-', 0, offset)+1
    segment_end = text.find('-', offset)
    if segment_end < 0:
        segment_end = len(text)
    segment = re.search(r'[A-Za-z0-9_.]+$', text[segment_start:offset])
    after = re.match(r'[A-Za-z0-9_.]+', text[offset:segment_end])
    if segment and after and IDENTIFIERS.fullmatch((segment[0]+after[0]).rstrip('.')):
        return True
    if text[offset-1].isdigit():
        if re.match(UNITS + r'(?![A-Za-z0-9_])', text[offset:]):
            return True
        if re.match(r'(?:st|nd|rd|th)(?![A-Za-z0-9_])', text[offset:]):
            return True
        if re.match(r'[eE][+-]?\d+(?![A-Za-z0-9_])', text[offset:]):
            return True
    return False


def inspect_markdown(source, *, chapter=False):
    """Find visible joins/review candidates; offsets refer to original source."""
    text = visible_prose(source)
    newline_offsets = [i for i, char in enumerate(source) if char == '\n']
    findings = []

    def finding(group, rule, start, end, matched, **extra):
        line = bisect_right(newline_offsets, start) + 1
        line_start = source.rfind('\n', 0, start) + 1
        line_end = source.find('\n', start)
        if line_end < 0:
            line_end = len(source)
        findings.append(dict(group=group, rule=rule, line=line,
                             column=start-line_start+1, start=start, end=end,
                             match=matched, context=source[line_start:line_end],
                             **extra))

    for match in JOIN.finditer(text):
        offset = match.start()
        token = _token(text, offset)
        if identifier_boundary(text, offset, token):
            continue
        finding('mechanical', 'glued-number-word', offset-1, offset+1,
                source[offset-1:offset+1], token=token,
                insertion_offset=offset, replacement=' ')
    if chapter:
        for rule, pattern in NARRATIVE_RULES.items():
            for match in pattern.finditer(text):
                finding('narrative', rule, match.start(), match.end(), match[0])
    return sorted(findings, key=lambda row: (row['start'], row['group'], row['rule']))


def load_exemptions(path):
    """Exact match/context exemptions avoid broad path or regex suppressions."""
    if path is None or not path.exists():
        return []
    document = json.loads(path.read_text())
    if set(document) != {'schema_version', 'exemptions'} or document['schema_version'] != 1:
        raise ValueError('Expected exemption schema_version 1 and exemptions')
    entries = document['exemptions']
    if not isinstance(entries, list):
        raise ValueError('Exemptions must be a list')
    for entry in entries:
        if (not isinstance(entry, dict)
                or set(entry) != {'path', 'rule', 'match', 'context', 'reason'}
                or not all(isinstance(value, str) and value.strip() for value in entry.values())
                or entry['rule'] not in {*NARRATIVE_RULES, 'glued-number-word'}
                or '\n' in entry['context'] or entry['match'] not in entry['context']
                or not re.fullmatch(r'book/(?:chapters|labs|solutions|front-matter|appendices)/[^/*]+\.md', entry['path'])):
            raise ValueError('Exemption needs an exact book path, rule, match, context and reason')
    return entries


def counts(rows):
    return {'matches': len(rows), 'affected_lines': len({(r['path'], r['line']) for r in rows})}


def scan_book(book_root, *, exemptions=()):
    book_root = Path(book_root)
    if not book_root.is_dir():
        raise FileNotFoundError(f'Book scope does not exist: {book_root}')
    files, used = [], set()
    for section in BOOK_SECTIONS:
        for path in sorted((book_root/section).glob('*.md')):
            source = path.read_text()
            canonical_path = f'book/{section}/{path.name}'
            findings, exempted = [], []
            for row in inspect_markdown(source, chapter=section == 'chapters'):
                row['path'] = canonical_path
                matches = [index for index, entry in enumerate(exemptions)
                           if entry['path'] == canonical_path and entry['rule'] == row['rule']
                           and entry['match'] == row['match'] and entry['context'] in row['context']]
                if matches:
                    used.update(matches)
                    exempted.append(dict(row, reason=exemptions[matches[0]]['reason']))
                else:
                    findings.append(row)
            files.append(dict(path=canonical_path, sha256=hashlib.sha256(source.encode()).hexdigest(),
                              findings=findings, exempted=exempted,
                              counts={group: counts([r for r in findings if r['group'] == group])
                                      for group in ('mechanical', 'narrative')}))
    all_findings = [row for file in files for row in file['findings']]
    return dict(scanner_version=SCANNER_VERSION,
                scanner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                book_root=str(book_root.resolve()), sections=list(BOOK_SECTIONS),
                exclusions=EXCLUSIONS, counting_units='matches and distinct path/line pairs',
                identifier_pattern=IDENTIFIERS.pattern, unit_pattern=UNITS,
                narrative_patterns={rule: pattern.pattern for rule, pattern in NARRATIVE_RULES.items()},
                counts={group: counts([r for r in all_findings if r['group'] == group])
                        for group in ('mechanical', 'narrative')},
                exempted_counts=counts([row for file in files for row in file['exempted']]),
                unused_exemptions=[entry for index, entry in enumerate(exemptions) if index not in used],
                files=files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--book-root', type=Path, default=ROOT/'book')
    parser.add_argument('--exemptions', type=Path, default=ROOT/'scripts/book_prose_exemptions.json')
    parser.add_argument('--no-exemptions', action='store_true')
    parser.add_argument('--strict-narrative', action='store_true')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    result = scan_book(args.book_root, exemptions=[] if args.no_exemptions else load_exemptions(args.exemptions))
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for file in result['files']:
            for row in file['findings']:
                print(f"{row['path']}:{row['line']}:{row['column']}: {row['group']} {row['rule']}: {row['match']}")
        print(f"{len(result['files'])} Markdown files; {result['counts']}; "
              f"{result['exempted_counts']['matches']} reviewed exemptions")
    return bool(not result['files'] or result['counts']['mechanical']['matches']
                or args.strict_narrative and result['counts']['narrative']['matches'])


if __name__ == '__main__':
    raise SystemExit(main())
