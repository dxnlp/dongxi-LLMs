#!/usr/bin/env python3
"""Audit book navigation and all28 daily notebook routes; no model execution."""
import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from dongxi_llms.course_manifest import validate_manifest


def link_problems(root, path, text):
    text = re.sub(r'```.*?```','',text,flags=re.S)
    issues = []
    for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
        target = target.strip().strip('<>')
        if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:',target) or target.startswith('#'):
            continue
        dest = target.split('#')[0]
        if dest and not (path.parent/dest).exists():
            issues.append(f'{path.relative_to(root)}: missing link {target}')
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    problems = []
    try:
        registry = json.loads((root/'docs/course_manifest.json').read_text())
        problems.extend(validate_manifest(registry, root))
        if not isinstance(registry, dict):
            registry = {'days': [], 'notebooks': []}
    except (OSError, ValueError) as error:
        registry = {'days': [], 'notebooks': []}
        problems.append(f'Cannot read routing contract: {error}')
    chapters = sorted((root/'book/chapters').glob('*.md'))
    solutions = sorted((root/'book/solutions').glob('*.md'))
    for i in range(1,16):
        for kind, files in [('chapter',chapters),('solutions',solutions)]:
            matches = [p for p in files if p.name.startswith(f'{i:02d}-')]
            if len(matches) != 1:
                problems.append(f'Chapter{i} requires exactly one {kind} file: {matches}')
    if len(chapters) != 15:
        problems.append(f'Expected15 chapters, found{len(chapters)}')
    for name in ['preface','how-to-use','notation']:
        if not (root/f'book/front-matter/{name}.md').exists():
            problems.append(f'Missing front matter: {name}')
    if len(list((root/'book/appendices').glob('*.md'))) != 4:
        problems.append('Expected4 appendices')
    notebooks = sorted((root/'notebooks').glob('day-*/*.ipynb'))
    days = []
    for day in range(1,29):
        files = [p for p in notebooks if p.parent.name==f'day-{day:02d}']
        if not files:
            problems.append(f'Day{day} has no notebook')
        if not (root/f'notebooks/day-{day:02d}/README.md').exists():
            problems.append(f'Day{day} has no session index')
        days.append({'day':day,'notebooks':[str(p.relative_to(root)) for p in files]})
    for path in notebooks:
        try:
            nb = json.loads(path.read_text())
            code = [c for c in nb['cells'] if c['cell_type']=='code']
            text = '\n'.join(''.join(c['source']) if isinstance(c['source'],list)
                             else c['source'] for c in nb['cells'] if c['cell_type']=='markdown')
            if not code or not text.strip():
                problems.append(f'Empty notebook: {path.relative_to(root)}')
            if not re.search(r'!\[[^\]]*\]\(',text):
                problems.append(f'No saved explanatory preview: {path.relative_to(root)}')
            problems.extend(link_problems(root,path,text))
        except (OSError, ValueError, KeyError, TypeError) as error:
            problems.append(f'Invalid notebook {path.relative_to(root)}: {error}')
    prose = list((root/'book').rglob('*.md'))
    prose += [root/'docs/COURSE_SEQUENCE.md',root/'docs/COURSE_BLUEPRINT.md',
              root/'docs/NOTEBOOK_CURRICULUM.md',root/'docs/EXPERIMENT_MATRIX.md',
              root/'docs/RELEASE_CHECKLIST.md',root/'notebooks/README.md',
              root/'README.md',root/'BOOK.md',root/'PROGRESS.md',root/'LEARNING_MEMORY.md',
              root/'docs/handoffs/CURRENT.md',root/'visuals/animations/COURSE_STORYBOARDS.md']
    prose += list((root/'notebooks').glob('day-*/README.md'))
    for path in prose:
        if not path.exists():
            problems.append(f'Missing route file: {path.relative_to(root)}'); continue
        problems.extend(link_problems(root,path,path.read_text()))
    registered = registry.get('notebooks', [])
    registered = [row for row in registered if isinstance(row,dict)] if isinstance(registered,list) else []
    result = {'chapters':len(chapters),'solutions':len(solutions),
              'appendices':len(list((root/'book/appendices').glob('*.md'))),
              'registered_notebooks': len(registered),
              'notebook_lanes': {lane:sum(row.get('lane')==lane for row in registered)
                                 for lane in ['core','optional','extension']},
              'notebooks':len(notebooks),'days':days,'problems':problems,
              'chapter_words':{p.name:len(p.read_text().split()) for p in chapters}}
    print(json.dumps(result,indent=2) if args.json else
          f"{result['chapters']} chapters, {result['solutions']} solution guides, "
          f"{result['appendices']} appendices, {result['notebooks']} notebooks; "
          f"{len(problems)} navigation/coverage issues")
    if problems:
        if not args.json:
            for problem in problems: print(problem)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
