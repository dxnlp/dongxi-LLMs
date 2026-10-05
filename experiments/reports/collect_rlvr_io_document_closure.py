#!/usr/bin/env python3
"""Read-only final docs/source closure; only exclusive new evidence is written."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT/'experiments/reports'
NAMES = [
    'README.md', 'BOOK.md', 'PROGRESS.md', 'LEARNING_MEMORY.md',
    'docs/COURSE_IMPROVEMENT_PLAN.md', 'docs/course_improvements.json',
    'docs/COURSE_EVIDENCE_MAP.md', 'docs/EXPERIMENT_MATRIX.md', 'docs/handoffs/CURRENT.md',
    'docs/PRODUCTION_RECOVERY_PLAN.md', 'docs/PRODUCTION_SUPERVISOR_PLAN.md',
    'docs/SNAPSHOT_INSPECTION_BUDGET_PLAN.md', 'tests/README.md',
    'book/chapters/13-group-relative-policy-optimization.md',
    'book/chapters/14-when-optimization-goes-wrong.md',
    'book/solutions/13-group-relative-policy-optimization.md',
    'book/solutions/14-when-optimization-goes-wrong.md',
    'book/labs/13-group-relative-policy-optimization.md',
    'book/labs/14-when-optimization-goes-wrong.md',
    'book/appendices/d-reproduction-and-environments.md',
    'notebooks/day-25/README.md', 'notebooks/day-25/02_ragged_kv_cache_and_exact_recovery.ipynb',
    'notebooks/figures/chapter-14/day-25-02_ragged_kv_cache_and_exact_recovery-08.png',
    'learning_artifacts/day-12-sft-mechanics/durable-training-boundaries.md',
    'learning_artifacts/day-23-grpo-rlvr/README.md',
    'learning_artifacts/day-25-rollout-systems-and-monitoring/README.md',
    'visuals/animations/PROPOSALS.md', 'experiments/specs/day-23-qwen-rlvr.md',
    'experiments/reports/2026-10-05-rlvr-io-readiness.md',
    'experiments/reports/2026-10-05-rlvr-reader-lesson.md',
    'experiments/reports/2026-10-05-rlvr-snapshot-io.md',
    'experiments/reports/2026-10-05-rlvr-snapshot-io-independent-review.md']


def identity(name):
    data = (ROOT/name).read_bytes()
    return dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def main():
    if len(sys.argv) != 2: raise SystemExit('One exclusive new closure JSON path required')
    result = dict(schema='dongxi-rlvr-io-document-closure-v1', created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Read-only final local source/docs/evidence recheck; no new model or notebook execution',
        files={name:identity(name) for name in NAMES}, commands=[], local_links=[],
        root_evidence_rechecked=0, root_evidence_bytes=0, accepted_explanatory_changes=[])
    cpu = json.loads((REPORTS/'2026-10-05-rlvr-io-cpu/run-01/cpu-verification.json').read_text())
    root = json.loads((REPORTS/'2026-10-05-rlvr-io-root-acceptance.json').read_text())
    for name, expected in cpu['source_sha256_before'].items():
        actual = identity(name)
        if actual['sha256'] != expected: raise AssertionError('Current executable differs: '+name)
        result['files'][name] = actual
    for name, expected in root['files'].items():
        actual = identity(name)
        if actual != expected:
            if name != 'docs/course_improvements.json': raise AssertionError('Retained root binding changed: '+name)
            result['accepted_explanatory_changes'].append(dict(path=name, measured_before=expected,
                explanatory_after=actual, boundary='after acceptance explanation/references only; statuses/counts unchanged'))
        result['root_evidence_rechecked'] += 1; result['root_evidence_bytes'] += actual['bytes']
    review_path = 'experiments/reports/2026-10-05-rlvr-snapshot-io-independent-review.json'
    review = json.loads((ROOT/review_path).read_text())
    if review['status'] != 'pass' or not review['all_map_checks_passed']:
        raise AssertionError('Released independent review must pass')
    for name, expected in review['reviewed_source_sha256'].items():
        if identity(name)['sha256'] != expected: raise AssertionError('Reviewed source changed: '+name)
    result['files'][review_path] = identity(review_path)
    result['files']['experiments/reports/2026-10-05-rlvr-io-root-acceptance.json'] = identity('experiments/reports/2026-10-05-rlvr-io-root-acceptance.json')
    for name in NAMES:
        if not name.endswith('.md'): continue
        for target in re.findall(r'\[[^\]]*\]\(([^)\n]+)\)', (ROOT/name).read_text()):
            if target.startswith(('http:', 'https:', 'mailto:', '#', 'codex:')): continue
            target = target.strip('<>').split('#')[0]
            if not target: continue
            path = (ROOT/name).parent/target
            if not path.exists(): raise AssertionError('Broken local link: '+name+' -> '+target)
            result['local_links'].append(dict(source=name, target=target))
    ledger = json.loads((ROOT/'docs/course_improvements.json').read_text())
    counts = {state:sum(row['status']==state for row in ledger['items']) for state in ('complete','in_progress','planned')}
    if counts != {'complete':13,'in_progress':5,'planned':0} or ledger['summary']['learner_day'] != 9:
        raise AssertionError('Package status or learner changed')
    for item in ledger['items']:
        for name in item.get('verification',[]):
            if not (ROOT/name).is_file(): raise AssertionError('Missing ledger evidence: '+name)
    for name, command in [('math',[sys.executable,'scripts/check_book_math.py']),
            ('routes',[sys.executable,'scripts/check_course_integrity.py','--json']),
            ('diff',['git','diff','--check'])]:
        process = subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=30)
        result['commands'].append(dict(name=name,command=command,actual_exit_code=process.returncode,
            stdout=process.stdout,stderr=process.stderr))
        if process.returncode: raise AssertionError('Final source/document check failed: '+name)
    for name, before in result['files'].items():
        if identity(name) != before: raise AssertionError('Closure file changed during recheck: '+name)
    result['status'] = 'passed'; result['package_counts'] = counts; result['learner_day'] = 9
    result['external_rows'] = 45; result['model_scale_jobs_started'] = 0
    target = Path(sys.argv[1])
    with target.open('x',encoding='utf8') as handle:
        handle.write(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],report=str(target),files=len(result['files']),
        local_links=len(result['local_links']),root_files=result['root_evidence_rechecked'],
        accepted_explanatory_changes=len(result['accepted_explanatory_changes']))),flush=True)


if __name__ == '__main__': main()
