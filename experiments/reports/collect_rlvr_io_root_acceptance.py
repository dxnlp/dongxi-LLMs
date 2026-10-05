#!/usr/bin/env python3
"""Read-only recheck of frozen RLVR/full-course evidence; exclusive new JSON.

No tests, kernels, models, services, installations or Git mutation are launched.
Retained failed artifacts stay files, not fabricated successful publications.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from collect_snapshot_io_root_acceptance import Recheck, footer, notebook_panel, source_pair

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from dongxi_llms.snapshot_io_budget import validate_work_receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu', type=Path, required=True)
    parser.add_argument('--rlvr', type=Path, required=True)
    parser.add_argument('--lesson', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit = Recheck()
    cpu = audit.json(args.cpu); rlvr = audit.json(args.rlvr); lesson = audit.json(args.lesson)
    if cpu['status'] != 'passed' or not cpu['inputs_unchanged'] or not cpu['sources_unchanged']:
        raise AssertionError('Full CPU source/input panel must pass at its frozen revision')
    sources = source_pair(audit, cpu, 'source_sha256_before', 'source_sha256_after', 'full current executable panel')
    audit.mapping(cpu['input_sha256'], 'full measured input revision before explanatory ledger update')
    full_footer = None; math = None; routes = None
    for command in cpu['commands']:
        if command.get('exit_code') != 0 or command.get('stopped_by') is not None:
            raise AssertionError('Every actual full-panel child must exit0 without guard termination')
        raw = audit.file(command['log'], expected=command['log_sha256'], relationship='full command retained log')
        if command['name'] == 'tests': full_footer = footer(raw.decode())
        if command['name'] == 'math': math = raw.decode().strip()
        if command['name'] == 'routes': routes = json.loads(raw)
    if full_footer is None or routes['problems'] or routes['registered_notebooks'] != 76:
        raise AssertionError('Actual footer and unchanged76 route registry required')
    selected = notebook_panel(audit, args.cpu.parent/'notebooks/manifest.json')
    selected += notebook_panel(audit, args.cpu.parent/'rng-lesson/manifest.json')
    if len(selected) != 6: raise AssertionError('Only the six declared fresh references are claimed')
    if rlvr['status'] != 'pass' or not rlvr['sources_unchanged']:
        raise AssertionError('Frozen RLVR component collection must pass')
    component_sources = source_pair(audit, rlvr, 'source_sha256', 'source_sha256_after', 'actual native RLVR components')
    audit.mapping(rlvr['artifact_sha256'], 'component current artifact byte identity')
    components = []
    for command in rlvr['commands']:
        if command['actual_exit_code'] != 0 or not command['actual_unittest_footer_ok']:
            raise AssertionError('Actual component exit/footer required')
        actual = footer(audit.file(command['raw_log']['path'], expected=command['raw_log']['sha256']).decode(),
                        command['actual_unittest_count'])
        components.append(dict(command=command['command'], **actual))
    arms = []
    for arm in rlvr['arms']:
        fresh = arm['fresh']; failure = arm['retained_failures']
        if (arm['execution']['exit_code'] != 0 or not arm['exact_fresh_numerical_replay']
                or fresh['state_sha256'] != arm['expected_numerical_sha256']
                or fresh['tail_sha256'] != arm['expected_tail_sha256']):
            raise AssertionError('Original trajectory/next-action digest must replay exactly')
        if not set(failure['after']['io']['failed_tickets']) <= set(fresh['io']['failed_tickets']):
            raise AssertionError('Later I/O failure refunded')
        if not set(failure['after']['model']['failed_tickets']) <= set(fresh['work']['failed_tickets']):
            raise AssertionError('Later semantic failure refunded')
        expected_cursors = [3] if arm['phase'] == 'pending' else [2, 3]
        if fresh['collect_completed_cursors'] != expected_cursors:
            raise AssertionError('Pending was recollected or original source order changed')
        arms.append(dict(seed=arm['seed'], phase=arm['phase'], state_sha256=fresh['state_sha256'],
            first_tail_sha256=fresh['tail_sha256'], actual_exit_code=arm['execution']['exit_code'],
            collect_completed_cursors=fresh['collect_completed_cursors'],
            later_failed_io_tickets=fresh['io']['failed_tickets'],
            later_failed_semantic_tickets=fresh['work']['failed_tickets']))
    if sorted((a['seed'], a['phase']) for a in arms) != [(2323,'completed'),(2323,'pending'),(2324,'completed'),(2324,'pending')]:
        raise AssertionError('All four original seed/phase arms required')
    if lesson['status'] != 'passed' or not lesson['sources_unchanged'] or not lesson['old_previews_unchanged']:
        raise AssertionError('Actual lesson preserves frozen source and old previews')
    source_pair(audit, lesson, 'source_sha256_before', 'source_sha256_after', 'native lesson source and inputs')
    audit.mapping(lesson['old_preview_sha256'], 'earlier seven figure identities unchanged')
    audit.file(lesson['preview']['path'], expected=lesson['preview']['sha256'], expected_bytes=lesson['preview']['bytes'])
    lesson_rows = notebook_panel(audit, args.lesson.parent/'notebook/manifest.json')
    audit.tree(args.lesson.parent)
    raw_root = ROOT/args.rlvr.parent if not args.rlvr.is_absolute() else args.rlvr.parent
    audit.tree(raw_root)
    markers = audit.snapshots(raw_root)
    receipts = []
    for path in sorted(raw_root.rglob('*.work.json')):
        receipt = validate_work_receipt(audit.json(path)); header = receipt['snapshot']
        payload = Path(str(path)[:-len('.work.json')]); marker = Path(str(payload)+'.commit.json')
        if audit.json(marker) != header: raise AssertionError('Receipt differs from published marker')
        audit.file(payload, expected=header['payload_sha256'], expected_bytes=header['payload_bytes'])
        receipts.append(str(path.relative_to(ROOT)))
    archive = 'experiments/reports/2026-10-05-rlvr-snapshot-io-preintegration'
    original = audit.json(archive+'/manifest.json')
    audit.file(archive+'/qwen_rlvr_lab.py.txt', expected=original['original_runner_sha256'])
    ledger = audit.json('docs/course_improvements.json')
    if ledger['summary']['complete'] != 13 or ledger['summary']['in_progress'] != 5 or ledger['summary']['learner_day'] != 9:
        raise AssertionError('Source verification cannot advance external packages or learner mastery')
    campaign = audit.json('experiments/reports/2026-10-04-staged-campaign-preparation.json')
    audit.file('experiments/reports/2026-10-04-staged-campaign-preparation.json',
        expected='34db95a7dbdcbfbfb6b108943d5cb172b2026d5db8f8528942032be5963816c4',
        relationship='unchanged previously accepted45-row external campaign')
    rows = campaign['campaign']['stage_rows']
    if (len(rows) != 45 or sum(len(row['actual']) for row in rows) != 945
            or campaign['campaign']['jobs_started'] != 0 or campaign['campaign']['actual_genealogy'] != []
            or any(value is not None for row in rows for value in row['actual'].values())):
        raise AssertionError('External outcomes must remain null')
    artifact_count = len(audit.files)
    result = dict(schema='dongxi-rlvr-io-root-acceptance-v1', created_utc=datetime.now(timezone.utc).isoformat(),
        actual_full_footer=full_footer, math_source_check=math,
        routes=dict(chapters=routes['chapters'], solutions=routes['solutions'], notebooks=76, problems=routes['problems']),
        current_executable_sources=len(sources), component_source_bindings=len(component_sources),
        component_footers_nonadditive=components, fresh_original_arms=arms,
        selected_fresh_references=selected, native_lesson_fresh_reference=lesson_rows,
        source_reference_cells=sum(row['source_code_cells'] for row in selected),
        executed_reference_cells=sum(row['executed_reference_code_cells'] for row in selected),
        actual_kernel_cells=sum(row['actual_kernel_code_cells'] for row in selected),
        actual_reference_images=sum(row['png_images'] for row in selected),
        snapshot_markers_checked=markers, independent_receipts_checked=receipts,
        file_count=artifact_count, relationship_count=len(audit.checks), files=audit.files, relationships=audit.checks,
        packages_complete=13, partial_packages=5, learner_day=9, external_rows=45,
        external_actual_null_fields=sum(len(row['actual']) for row in rows),
        scope='Read-only actual frozen offline CPU evidence; not pretrained, GPU, physical, Mac, hosted or cross-host proof',
        model_scale_jobs_started=0, launch_authorized=False, production_ready=False)
    target = args.output if args.output.is_absolute() else ROOT/args.output
    with target.open('x', encoding='utf8') as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps(dict(report=str(target), tests=full_footer, sources=len(sources),
                         files=artifact_count, relationships=len(audit.checks), receipts=len(receipts))), flush=True)


if __name__ == '__main__': main()
