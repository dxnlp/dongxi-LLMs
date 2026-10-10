#!/usr/bin/env python3
"""Check declared notebook reference adjacency and editorial preservation.

This does not execute notebooks or prove that a reference answers its prompt.
The review ledger records that semantic judgment and its evidence boundary.
Static checks bind the judgment to actual prompt/code cells, visible headings,
and the immutable pre-edit code objects, outputs, order and metadata.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RUN = 'experiments/reports/2026-10-10-book-editorial-pass/run-01'
HASH_RECIPE = 'sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode())'
KINDS = {'computation', 'interpretation', 'perturbation', 'written-reflection'}
MARKDOWN_ROLES = {'exercise/checkpoint', 'reference-explanation', 'narrative/mechanism-context',
                  'saved-preview', 'architecture/visual-guide'}
LIMITS = [
    'Static headings, declared adjacency and preservation do not prove semantic correctness or complete discovery of questions in prose.',
    'Audited pairing bases are recorded content-review judgments; fresh execution and learner understanding require separate evidence.',
    'A runnable baseline reference does not establish execution of its requested perturbation.',
]


def object_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def cell_source(cell):
    value = cell.get('source', '')
    if isinstance(value, list) and all(isinstance(line, str) for line in value):
        return ''.join(value)
    if isinstance(value, str):
        return value
    raise ValueError('Cell source must be a string or a list of strings')


def code_cells(notebook):
    return [cell for cell in notebook['cells'] if cell.get('cell_type') == 'code']


def normalized_baseline_code(notebook):
    # The original capture inserted id:null for absent IDs. This normalization
    # is used only to check that historical hash; Git comparison uses full,
    # unmodified objects and distinguishes a missing key from a null value.
    return [dict(cell, id=cell.get('id')) for cell in code_cells(notebook)]


def visible_headings(source):
    """Return actual Markdown headings outside fenced code, with their offsets."""
    source = re.sub(r'<!--[\s\S]*?-->',
                    lambda match: ''.join('\n' if character == '\n' else ' ' for character in match.group()), source)
    headings, fence, offset = [], None, 0
    for line in source.splitlines(keepends=True):
        match = re.match(r'^ {0,3}(`{3,}|~{3,})', line)
        if match:
            marker = match.group(1)
            if fence is None:
                fence = (marker[0], len(marker))
            elif marker[0] == fence[0] and len(marker) >= fence[1]:
                fence = None
        elif fence is None:
            match = re.match(r'^ {0,3}(#{1,6})\s+(.+?)\s*#*\s*$', line)
            if match:
                headings.append((len(match.group(1)), match.group(2), offset))
        offset += len(line)
    return headings


def heading_positions(cell, label):
    return [offset for _, heading, offset in visible_headings(cell_source(cell))
            if re.match(rf'{re.escape(label)}(?:\b|\s*[:—–-])', heading, re.I)]


def prompt_headings(cell):
    return [heading for level, heading, _ in visible_headings(cell_source(cell))
            if level > 1 and re.search(r'\b(prediction|checkpoint|exercise|question)\b', heading, re.I)
            and not re.search(r'\b(reference|explanation)\b', heading, re.I)]


def reference_problems(cell):
    """Reject empty/incomplete scaffolds; semantic sufficiency is ledger review."""
    issues = []
    if 'learner-exercise' in cell.get('metadata', {}).get('tags', []):
        issues.append('Reference code is an optional learner exercise')
    try:
        tree = ast.parse(cell_source(cell))
    except SyntaxError as error:
        return issues + [f'Reference code is not complete Python: {error.msg}']
    if not tree.body or all(isinstance(node, ast.Pass) for node in tree.body):
        issues.append('Reference code is empty or contains only comments/pass')
    indexing_ellipses = {node for subscript in ast.walk(tree) if isinstance(subscript, ast.Subscript)
                        for node in ast.walk(subscript.slice)
                        if isinstance(node, ast.Constant) and node.value is Ellipsis}
    if any(isinstance(node, ast.Constant) and node.value is Ellipsis and node not in indexing_ellipses
           for node in ast.walk(tree)):
        issues.append('Reference code contains an unfinished ellipsis placeholder')
    return issues


def safe_notebook_path(path):
    return (isinstance(path, str) and path.startswith('notebooks/') and path.endswith('.ipynb')
            and '\\' not in path and not any(ord(c) < 32 for c in path)
            and '..' not in PurePosixPath(path).parts and str(PurePosixPath(path)) == path)


def load_git_originals(root, baseline):
    """Read tracked original notebooks; ignored checkpoint copies have no Git object."""
    commit = baseline.get('base_commit')
    if not isinstance(commit, str) or not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('Baseline must name a complete Git commit identity')
    inventory = subprocess.check_output(
        ['git', 'ls-tree', '-r', '--name-only', commit, 'notebooks'], cwd=root, text=True).splitlines()
    tracked = set(inventory)
    originals = {}
    for path in baseline['notebooks']:
        if not safe_notebook_path(path):
            raise ValueError(f'Unsafe baseline path: {path!r}')
        if path in tracked:
            raw = subprocess.check_output(['git', 'show', f'{commit}:{path}'], cwd=root)
            if hashlib.sha256(raw).hexdigest() != baseline['notebooks'][path]['sha256']:
                raise ValueError(f'Git source and immutable original file identity disagree: {path}')
            originals[path] = json.loads(raw)
    return originals


def check_contracts(root, manifest, baseline, review, originals):
    """Return a bounded static audit; originals contains independent Git objects."""
    root = Path(root)
    problems, records, preservation = [], [], []

    def problem(path, check, message, exercise=None):
        record = dict(path=path, check=check, message=message)
        if exercise is not None:
            record['exercise_id'] = exercise
        problems.append(record)

    routes = [row.get('path') for row in manifest.get('notebooks', []) if isinstance(row, dict)]
    if not routes or not all(safe_notebook_path(path) for path in routes) or len(set(routes)) != len(routes):
        problem(None, 'routing', 'Manifest must declare nonempty unique safe notebook routes')
        routes = [path for path in routes if safe_notebook_path(path)]
    routes = set(routes)
    original_inventory = baseline.get('notebooks', {})
    if not isinstance(original_inventory, dict) or not original_inventory:
        raise ValueError('Baseline notebook inventory is empty or malformed')
    inventory = {path.relative_to(root).as_posix() for path in (root/'notebooks').rglob('*.ipynb')}
    for path in sorted(inventory.symmetric_difference(original_inventory)):
        problem(path, 'inventory', 'Current inventory differs from immutable original inventory')
    for path in sorted(routes - set(original_inventory)):
        problem(path, 'routing', 'Registered route is absent from the original preservation baseline')
    if review.get('schema_version') != 1 or review.get('status') != 'reviewed':
        problem(None, 'review', 'Require schema_version 1 and a completed reviewed ledger')
    if review.get('base_commit') != baseline.get('base_commit'):
        problem(None, 'review', 'Review and immutable baseline name different Git commits')
    if review.get('hash_recipe') != HASH_RECIPE:
        problem(None, 'review', 'Unknown review hash recipe')
    rows = review.get('notebooks', [])
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError('Review notebooks must be a list of objects')
    reviewed_paths = [row.get('path') for row in rows]
    if len(set(reviewed_paths)) != len(reviewed_paths):
        problem(None, 'review', 'Review contains duplicate notebook routes')
    for path in sorted(set(reviewed_paths) - routes, key=str):
        problem(path, 'routing', 'Review contains an unknown/unregistered notebook route')
    for path in sorted(routes - set(reviewed_paths)):
        problem(path, 'review', 'Registered notebook has no completed contract review')
    ledger = {row.get('path'): row for row in rows}
    current = {}
    for path, captured in sorted(original_inventory.items()):
        if not safe_notebook_path(path):
            raise ValueError(f'Unsafe baseline path: {path!r}')
        try:
            raw = (root/path).read_bytes()
            notebook = json.loads(raw)
            cells = notebook['cells']
            if not isinstance(cells, list) or not all(isinstance(cell, dict) for cell in cells):
                raise ValueError('Notebook cells must be a list of objects')
            current[path] = notebook
            codes = code_cells(notebook)
            full_hash = object_sha256(codes)
            normalized_hash = object_sha256(normalized_baseline_code(notebook))
            metadata_hash = object_sha256(notebook['metadata'])
            if len(codes) != captured.get('code_cells') or normalized_hash != captured.get('preserved_code_sha256'):
                problem(path, 'preservation', 'Code-cell content, order, IDs, metadata, outputs or execution counts changed')
            if metadata_hash != captured.get('notebook_metadata_sha256'):
                problem(path, 'preservation', 'Notebook metadata changed')
            original = originals.get(path)
            authority = 'independent-git-objects' if original is not None else 'immutable-full-file-sha256'
            if original is not None:
                if codes != code_cells(original):
                    problem(path, 'preservation', 'Complete code objects differ from Git original (missing keys remain significant)')
                if {k:v for k,v in notebook.items() if k != 'cells'} != {k:v for k,v in original.items() if k != 'cells'}:
                    problem(path, 'preservation', 'Notebook metadata/format/header differs from Git original')
            elif path in routes:
                problem(path, 'preservation', 'Registered notebook lacks an independent original Git object')
            elif hashlib.sha256(raw).hexdigest() != captured.get('sha256'):
                problem(path, 'preservation', 'Ignored checkpoint copy changed from its immutable full-file identity')
            preservation.append(dict(path=path, code_cells=len(codes), normalized_baseline_code_sha256=normalized_hash,
                                     full_code_sha256=full_hash, metadata_sha256=metadata_hash, authority=authority))
        except (OSError, ValueError, KeyError, TypeError) as error:
            problem(path, 'preservation', f'Cannot inspect original-scope notebook: {error}')

    for path in sorted(routes & set(current) & set(ledger) & set(originals)):
        row, notebook, original = ledger[path], current[path], originals[path]
        cells, old_cells = notebook['cells'], original['cells']
        codes = [(i, cell) for i, cell in enumerate(cells) if cell.get('cell_type') == 'code']
        ids = [cell.get('id') for cell in cells if cell.get('id') is not None]
        if len(ids) != len(set(ids)):
            problem(path, 'identity', 'Notebook cell IDs are not unique')
        by_id = {cell['id']: (i, cell) for i, cell in enumerate(cells) if cell.get('id') is not None}
        captured = original_inventory[path]
        for key, expected in [('baseline_code_sha256', captured['preserved_code_sha256']),
                              ('baseline_metadata_sha256', captured['notebook_metadata_sha256']),
                              ('baseline_full_code_sha256', object_sha256(code_cells(original)))]:
            if row.get(key) != expected:
                problem(path, 'review', f'Review {key} disagrees with original identity')
        expected_markdown = [dict(original_cell_index=i, original_cell_id=cell.get('id'),
                                  source_sha256=object_sha256(cell.get('source', '')))
                             for i, cell in enumerate(old_cells) if cell.get('cell_type') == 'markdown']
        reviewed_markdown = row.get('original_markdown', [])
        if not isinstance(reviewed_markdown, list) or not all(isinstance(entry, dict) for entry in reviewed_markdown):
            problem(path, 'review', 'Original Markdown review must be a list of classified cells')
            reviewed_markdown = []
        identities = [{key:entry.get(key) for key in ('original_cell_index', 'original_cell_id', 'source_sha256')}
                      for entry in reviewed_markdown]
        if identities != expected_markdown:
            problem(path, 'review', 'Original Markdown inventory/hashes/IDs/indices disagree with Git original')
        for entry in reviewed_markdown:
            if (entry.get('reviewed_role') not in MARKDOWN_ROLES or not isinstance(entry.get('review_note'), str)
                    or not entry['review_note'].strip()):
                problem(path, 'review', 'Every original Markdown cell requires a known reviewed role and explicit note')
        pairs = row.get('pairs', [])
        if not isinstance(pairs, list) or not pairs or not all(isinstance(pair, dict) for pair in pairs):
            problem(path, 'pairing', 'Notebook needs a nonempty reviewed pair list')
            continue
        exercise_ids = [pair.get('exercise_id') for pair in pairs]
        if any(not isinstance(value, str) or not value.strip() for value in exercise_ids) or len(set(exercise_ids)) != len(exercise_ids):
            problem(path, 'pairing', 'Exercise identities must be nonempty and unique within the notebook')
        prompt_ids = {pair.get('prompt_cell_id') for pair in pairs}
        declared = row.get('reviewed_prompt_cell_ids', [])
        if not isinstance(declared, list) or len(set(declared)) != len(declared) or set(declared) != prompt_ids:
            problem(path, 'pairing', 'Every declared reviewed prompt must have a pair, with no extra pair prompts')
        exemptions = row.get('heading_exemptions', [])
        if not isinstance(exemptions, list) or not all(isinstance(entry, dict) for entry in exemptions):
            problem(path, 'coverage', 'Heading exemptions must be exact explained objects')
            exemptions = []
        used_exemptions = set()

        def exempted(heading, cell_id=None, original_index=None):
            for index, entry in enumerate(exemptions):
                if (entry.get('heading') == heading and isinstance(entry.get('reason'), str) and entry['reason'].strip()
                        and ((original_index is not None and entry.get('original_cell_index') == original_index)
                             or (original_index is None and cell_id is not None and entry.get('cell_id') == cell_id))):
                    used_exemptions.add(index)
                    return True
            return False

        for i, cell in enumerate(cells):
            if cell.get('cell_type') == 'markdown':
                for heading in prompt_headings(cell):
                    if cell.get('id') not in prompt_ids and not exempted(heading, cell_id=cell.get('id')):
                        problem(path, 'coverage', f'Visible prompt heading has no reviewed pair: cell {i}, {heading}')
        old_prompt_indices = {pair.get('original_cell_index') for pair in pairs}
        classified_prompt_indices = {entry.get('original_cell_index') for entry in reviewed_markdown
                                     if entry.get('reviewed_role') == 'exercise/checkpoint'}
        if classified_prompt_indices != old_prompt_indices:
            problem(path, 'coverage', 'Every original cell classified as an exercise/checkpoint must map to a reviewed pair, and every pair must have that role')
        for i, cell in enumerate(old_cells):
            if cell.get('cell_type') == 'markdown':
                for heading in prompt_headings(cell):
                    if i not in old_prompt_indices and not exempted(heading, original_index=i):
                        problem(path, 'coverage', f'Original prompt heading has no pair or explained disposition: cell {i}, {heading}')
        for index, _ in enumerate(exemptions):
            if index not in used_exemptions:
                problem(path, 'coverage', f'Unused/nonexact heading exemption {index}')

        for pair in pairs:
            exercise = pair.get('exercise_id')

            def pair_problem(message):
                problem(path, 'pairing', message, exercise)

            try:
                prompt_index, prompt = by_id[pair['prompt_cell_id']]
                heading_index, reference_heading = by_id[pair['reference_heading_cell_id']]
                explanation_index, explanation = by_id[pair['explanation_cell_id']]
                ordinal = pair['reference_code_ordinal']
                if not isinstance(ordinal, int) or isinstance(ordinal, bool) or not 0 <= ordinal < len(codes):
                    raise ValueError('Reference code ordinal is out of range')
                code_index, code = codes[ordinal]
                old_index = pair['original_cell_index']
                if not isinstance(old_index, int) or isinstance(old_index, bool) or not 0 <= old_index < len(old_cells):
                    raise ValueError('Original prompt index is out of range')
                old_prompt = old_cells[old_index]
                if old_prompt.get('cell_type') != 'markdown' or pair.get('original_cell_id') != old_prompt.get('id'):
                    pair_problem('Original prompt index/ID does not identify original Markdown')
                excerpt = pair.get('source_prompt_excerpt')
                if not isinstance(excerpt, str) or not excerpt.strip() or excerpt not in cell_source(old_prompt):
                    pair_problem('Prompt excerpt is missing or not present in original Markdown')
                if pair.get('kind') not in KINDS or not isinstance(pair.get('audited_basis'), str) or not pair['audited_basis'].strip():
                    pair_problem('Pair lacks an explicit review kind and audited semantic basis')
                if pair.get('reference_code_cell_id') != code.get('id') or pair.get('reference_code_sha256') != object_sha256(code):
                    pair_problem('Reference code ID/hash does not bind its full actual code object')
                if any(cell.get('cell_type') != 'markdown' for cell in (prompt, reference_heading, explanation)):
                    pair_problem('Prompt, reference heading and explanation must be Markdown cells')
                    continue
                for name, cell in [('prompt', prompt), ('reference_heading', reference_heading), ('explanation', explanation)]:
                    if pair.get(f'{name}_source_sha256') != object_sha256(cell.get('source', '')):
                        pair_problem(f'Reviewed {name} source hash disagrees with its current Markdown cell')
                predictions = heading_positions(prompt, 'Prediction')
                checkpoints = heading_positions(prompt, 'Checkpoint')
                if not predictions or not checkpoints:
                    pair_problem('Prompt lacks visible Prediction and Checkpoint headings')
                reveals = heading_positions(prompt, 'Reference solution') + heading_positions(prompt, 'Reference explanation')
                if predictions and reveals and min(reveals) < min(predictions):
                    pair_problem('Reference answer heading precedes the prediction in its prompt cell')
                if not heading_positions(reference_heading, 'Reference solution'):
                    pair_problem('Adjacent reference marker lacks a visible Reference solution heading')
                if not heading_positions(explanation, 'Reference explanation'):
                    pair_problem('Adjacent explanation lacks a visible Reference explanation heading')
                explanation_body = re.sub(r'^\s*#{1,6}[^\n]*', '', cell_source(explanation), flags=re.M).strip()
                if not explanation_body:
                    pair_problem('Reference explanation has no mechanism/evidence-boundary prose')
                if not prompt_index <= heading_index < code_index < explanation_index:
                    pair_problem('Prediction/prompt does not precede the reference and explanation')
                if heading_index != code_index - 1 or explanation_index != code_index + 1:
                    pair_problem('Runnable reference must have its marker immediately before and explanation immediately after')
                intervening = list(range(prompt_index + 1, heading_index))
                expected_ids = [cells[i].get('id') for i in intervening]
                if pair.get('allowed_intervening_cell_ids', []) != expected_ids:
                    pair_problem('Intervening learner/setup cells disagree with the reviewed ordered cell list')
                if 'allowed_intervening_cell_indices' in pair and pair['allowed_intervening_cell_indices'] != intervening:
                    pair_problem('Intervening cell indices disagree with the reviewed ordered cell list')
                reasons = pair.get('allowed_intervening_reasons', {})
                keys = [cell_id if cell_id is not None else f'index:{i}' for i, cell_id in zip(intervening, expected_ids)]
                if not isinstance(reasons, dict) or set(reasons) != set(keys) or any(not isinstance(reasons[key], str) or not reasons[key].strip() for key in keys):
                    pair_problem('Every intervening cell requires its exact reviewed reason')
                for i in intervening:
                    if cells[i].get('cell_type') == 'markdown' and (heading_positions(cells[i], 'Reference solution') or heading_positions(cells[i], 'Reference explanation')):
                        pair_problem('An intervening cell reveals a reference answer before its declared reference')
                for issue in reference_problems(code):
                    pair_problem(issue)
            except (KeyError, IndexError, TypeError, ValueError) as error:
                pair_problem(f'Invalid or missing pairing identity: {error}')
        records.append(dict(path=path, source_sha256=hashlib.sha256((root/path).read_bytes()).hexdigest(),
                            reviewed_prompts=len(prompt_ids), reference_pairs=len(pairs),
                            reference_groups=len({pair.get('reference_code_ordinal') for pair in pairs
                                                  if isinstance(pair.get('reference_code_ordinal'), int)}),
                            heading_exemptions=len(exemptions)))
    return dict(schema_version=1, scope='Static editorial notebook contract and preservation review', limits=LIMITS,
                hash_recipe=HASH_RECIPE,
                historical_code_hash_normalization='Original capture adds id:null only when a code-cell id key is absent; independent Git/full-file checks preserve missing keys.',
                base_commit=baseline.get('base_commit'), registered_notebooks=len(routes),
                reviewed_notebooks=len(records), preserved_inventory=len(preservation),
                preserved_code_cells=sum(row['code_cells'] for row in preservation),
                preservation_authorities={authority:sum(row['authority'] == authority for row in preservation)
                                          for authority in ['independent-git-objects', 'immutable-full-file-sha256']},
                reviewed_prompts=sum(row['reviewed_prompts'] for row in records),
                reference_pairs=sum(row['reference_pairs'] for row in records),
                reference_groups=sum(row['reference_groups'] for row in records),
                notebooks=records, preservation=preservation, problems=problems)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--baseline', type=Path, default=Path(RUN)/'baseline.json')
    parser.add_argument('--review', type=Path, default=Path(RUN)/'notebook-contract-review.json')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        def read(path):
            return json.loads((path if path.is_absolute() else root/path).read_text())
        baseline, review = read(args.baseline), read(args.review)
        expected_baseline = (args.baseline if args.baseline.is_absolute() else root/args.baseline).resolve()
        if not isinstance(review.get('baseline_path'), str) or (root/review['baseline_path']).resolve() != expected_baseline:
            raise ValueError('Ledger baseline_path does not identify the selected immutable baseline')
        sys.path.insert(0, str(ROOT/'src'))
        from dongxi_llms.course_manifest import load_manifest
        manifest = load_manifest(root, check_files=False)
        originals = load_git_originals(root, baseline)
        result = check_contracts(root, manifest, baseline, review, originals)
        result['review_sha256'] = hashlib.sha256((args.review if args.review.is_absolute() else root/args.review).read_bytes()).hexdigest()
        result['baseline_sha256'] = hashlib.sha256(expected_baseline.read_bytes()).hexdigest()
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        result = dict(limits=LIMITS, problems=[dict(path=None, check='input', message=str(error))])
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"{result.get('reviewed_notebooks', 0)} reviewed routes; "
              f"{result.get('reference_pairs', 0)} declared reference pairs; "
              f"{result.get('preserved_inventory', 0)} notebooks checked for preservation; "
              f"{len(result['problems'])} issues")
        for issue in result['problems']:
            print(f"{issue['path'] or 'inputs'}: {issue['message']}")
        print('Static checks do not prove semantic correctness or fresh execution.')
    return bool(result['problems'])


if __name__ == '__main__':
    raise SystemExit(main())
