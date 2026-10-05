#!/usr/bin/env python3
"""Final read-only story-source/book/ledger closure; writes only a new JSON."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
NAMES = [
    'README.md', 'BOOK.md', 'PROGRESS.md', 'LEARNING_MEMORY.md',
    'docs/COURSE_IMPROVEMENT_PLAN.md', 'docs/course_improvements.json',
    'docs/COURSE_EVIDENCE_MAP.md', 'docs/EXPERIMENT_MATRIX.md', 'docs/handoffs/CURRENT.md',
    'docs/PRODUCTION_RECOVERY_PLAN.md', 'docs/PRODUCTION_SUPERVISOR_PLAN.md',
    'docs/SNAPSHOT_INSPECTION_BUDGET_PLAN.md', 'docs/TINYSTORIES_PIPELINE.md', 'tests/README.md',
    'book/chapters/06-pretraining-as-a-controlled-system.md',
    'book/solutions/06-pretraining-as-a-controlled-system.md',
    'book/labs/06-reading-a-pretraining-run.md', 'notebooks/day-09/README.md',
    'notebooks/day-09/01_read_training_clocks.ipynb',
    'notebooks/figures/chapter-06/day-09-01_read_training_clocks-02.png',
    'learning_artifacts/day-09-pretraining-run-and-diagnosis/README.md',
    'learning_artifacts/day-09-pretraining-run-and-diagnosis/target-budget-is-a-boundary.md',
    'visuals/animations/PROPOSALS.md',
    'experiments/reports/2026-10-05-story-work-readiness.md',
    'experiments/reports/2026-10-05-story-work-lesson.md',
    'experiments/reports/2026-10-05-story-work.md',
    'experiments/reports/2026-10-05-story-work-implementation-notes.md',
    'experiments/reports/2026-10-05-story-work-independent-review.md',
    'experiments/reports/2026-10-05-story-work-independent-review.json',
    'experiments/reports/2026-10-05-story-work-root-acceptance.json',
    'experiments/reports/2026-10-05-story-work-root-collector-diagnostic.json',
]


def identity(name):
    raw = (ROOT / name).read_bytes()
    return dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def main():
    if len(sys.argv) != 2: raise SystemExit('Supply one exclusive new closure JSON path')
    target = Path(sys.argv[1])
    if target.exists(): raise FileExistsError(target)
    result = dict(schema='dongxi-story-work-document-closure-v1', created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Final local source/book/ledger and retained-evidence recheck; no model or notebook execution',
        files={name: identity(name) for name in NAMES}, commands=[], local_links=[],
        root_evidence_rechecked=0, root_evidence_bytes=0, accepted_explanatory_changes=[])
    cpu = json.loads((ROOT / 'experiments/reports/2026-10-05-story-work-cpu/run-01/cpu-verification.json').read_text())
    root = json.loads((ROOT / 'experiments/reports/2026-10-05-story-work-root-acceptance.json').read_text())
    if cpu['status'] != 'passed' or root['status'] != 'passed': raise AssertionError('CPU/root panel must pass')
    for name, expected in cpu['source_sha256_before'].items():
        actual = identity(name)
        if actual['sha256'] != expected: raise AssertionError('Measured executable changed: ' + name)
        result['files'][name] = actual
    for name, expected in root['file_sha256'].items():
        actual = identity(name)
        if actual != expected:
            if name != 'docs/course_improvements.json': raise AssertionError('Retained root binding changed: ' + name)
            result['accepted_explanatory_changes'].append(dict(path=name, measured_before=expected,
                explanatory_after=actual, boundary='After-acceptance evidence/scope/next-action amendments only; package states and learner position unchanged'))
        result['root_evidence_rechecked'] += 1
        result['root_evidence_bytes'] += actual['bytes']
    review = json.loads((ROOT / 'experiments/reports/2026-10-05-story-work-independent-review.json').read_text())
    if review['status'] != 'passed' or review['new_findings']: raise AssertionError('Independent source review did not pass')
    for binding in review['bindings']:
        if identity(binding['path'])['sha256'] != binding['sha256']:
            raise AssertionError('Independent reviewed binding changed: ' + binding['path'])
    result['independent_review_binding_entries_rechecked'] = len(review['bindings'])
    for name in NAMES:
        if not name.endswith('.md'): continue
        for link in re.findall(r'\[[^\]]*\]\(([^)\n]+)\)', (ROOT / name).read_text()):
            if link.startswith(('http:', 'https:', 'mailto:', '#', 'codex:')): continue
            link = link.strip('<>').split('#')[0]
            if not link: continue
            path = (ROOT / name).parent / link
            if not path.exists(): raise AssertionError('Broken local link: ' + name + ' -> ' + link)
            result['local_links'].append(dict(source=name, target=link))
    ledger = json.loads((ROOT / 'docs/course_improvements.json').read_text())
    counts = {state: sum(item['status'] == state for item in ledger['items']) for state in ('complete', 'in_progress', 'planned')}
    complete_ids = {item['id'] for item in ledger['items'] if item['status'] == 'complete'}
    if (counts != dict(complete=13, in_progress=5, planned=0) or complete_ids != {f'DXI-{i:02}' for i in range(4, 17)}
            or ledger['summary']['learner_day'] != 9 or ledger['summary']['notebooks'] != 76):
        raise AssertionError('Package or learner scope changed')
    evidence_count = 0
    for item in ledger['items']:
        for name in item['evidence']:
            if not (ROOT / name).is_file(): raise AssertionError('Missing actual ledger evidence: ' + name)
            evidence_count += 1
    result['ledger_evidence_paths_checked'] = evidence_count
    campaign_path = 'experiments/reports/2026-10-04-staged-campaign-preparation.json'
    campaign = json.loads((ROOT / campaign_path).read_text())['campaign']
    if (len(campaign['stage_rows']) != 45 or sum(len(row['actual']) for row in campaign['stage_rows']) != 945
            or campaign['jobs_started'] != 0 or campaign['actual_genealogy']
            or any(value is not None for row in campaign['stage_rows'] for value in row['actual'].values())):
        raise AssertionError('External campaign is not unchanged/unexecuted')
    result['files'][campaign_path] = identity(campaign_path)
    for name, command in [('math', [sys.executable, 'scripts/check_book_math.py']),
                           ('routes', [sys.executable, 'scripts/check_course_integrity.py', '--json']),
                           ('diff', ['git', 'diff', '--check'])]:
        child = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
        result['commands'].append(dict(name=name, command=command, actual_exit_code=child.returncode,
                                      stdout=child.stdout, stderr=child.stderr))
        if child.returncode: raise AssertionError('Final document check failed: ' + name)
    for name, expected in result['files'].items():
        if identity(name) != expected: raise AssertionError('Closure source changed in flight: ' + name)
    result.update(status='passed', package_counts=counts, learner_day=9, external_rows=45,
                  external_actual_fields_null=945, model_scale_jobs_started=0, actual_genealogy=[],
                  production_ready=False, launch_authorized=False)
    with target.open('x', encoding='utf8') as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps(dict(status=result['status'], files=len(result['files']), local_links=len(result['local_links']),
                         root_files=result['root_evidence_rechecked'], ledger_evidence_paths=evidence_count,
                         accepted_explanatory_changes=len(result['accepted_explanatory_changes']))), flush=True)


if __name__ == '__main__': main()
