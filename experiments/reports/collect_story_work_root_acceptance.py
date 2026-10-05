#!/usr/bin/env python3
"""Read-only recheck of retained story work and whole-course CPU evidence.

No training, inference, kernels, services or Git commands are started. The sole
write is a new exclusive acceptance JSON; existing evidence is never rewritten.
Trusted tiny CPU tensor artifacts are hashed before restricted deserialization.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))
from collect_snapshot_io_root_acceptance import Recheck, footer, notebook_panel, source_pair


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def render():
    import torch
    from dongxi_llms.story_work_budget import validate_story_receipt
    from test_stories_valid_target_budget import state_digest

    audit = Recheck()
    component_dir = ROOT / 'experiments/reports/2026-10-05-story-work/run-01'
    lesson_dir = ROOT / 'experiments/reports/2026-10-05-story-work-lesson/run-01'
    cpu_dir = ROOT / 'experiments/reports/2026-10-05-story-work-cpu/run-01'
    component = audit.json(component_dir / 'verification.json')
    lesson = audit.json(lesson_dir / 'verification.json')
    cpu = audit.json(cpu_dir / 'cpu-verification.json')
    require(all(r['status'] == 'passed' for r in (component, lesson, cpu)), 'All final panels must pass')
    require(not component['development'] and component['failure'] is None, 'Exclusive final component required')
    require(cpu['inputs_unchanged'] and cpu['sources_unchanged'], 'Full acceptance source/input drift')
    source_maps = [source_pair(audit, record, 'source_sha256_before', 'source_sha256_after', name)
                   for name, record in (('component', component), ('lesson', lesson), ('whole-course', cpu))]
    for name, identity in component['artifact_sha256'].items():
        audit.file(name, expected=identity['sha256'], expected_bytes=identity['bytes'],
                   relationship='component retained artifact to actual SHA and bytes')
    for name, identity in cpu['input_sha256'].items():
        audit.file(name, expected=identity, relationship='measured whole-course input bytes')

    archive_dir = ROOT / 'experiments/reports/2026-10-05-story-work-preintegration'
    archive = audit.json(archive_dir / 'manifest.json')
    for original, row in archive['source_files'].items():
        audit.file(archive_dir / row['archive'], expected=row['sha256'], expected_bytes=row['bytes'],
                   relationship='original byte archive: ' + original)
    for name, expected in component['source_sha256_before'].items():
        audit.file(component_dir / 'source-before' / (name + '.txt'), expected=expected,
                   relationship='exclusive measured source copy to live source identity')

    component_footers = {}
    for name, path, count in (('new_controls', 'new-controls.json', 33),
                              ('existing_story_and_work', 'existing-controls.json', 47),
                              ('reader_lesson', 'reader-lesson.json', 6)):
        raw = audit.json(component_dir / path)
        require(raw == component['panels'][name] and raw['actual_exit_code'] == 0,
                'Component actual process record mismatch')
        actual = footer(raw['stdout'] + raw['stderr'], count)
        require(actual['tests'] == raw['actual_footer']['test_count'] and
                actual['unittest_seconds'] == raw['actual_footer']['unittest_seconds'],
                'Parsed component footer mismatch')
        component_footers[name] = actual

    comparisons = []
    for row in component['original_equations']:
        arm = component_dir / 'original-equations' / f"{row['seed']}-{int(row['activation_checkpointing'])}"
        require(audit.json(arm / 'comparison.json') == row, 'Original equation raw disagreement')
        histories = audit.json(arm / 'history.json')
        digests = {}
        for mode, expected in row['state_sha256'].items():
            path = arm / (mode + '-numerical-state.pt')
            audit.file(path)
            digests[mode] = state_digest(torch.load(path, map_location='cpu', weights_only=True))
            require(digests[mode] == expected and len(histories[mode]) == 5,
                    'Archived/current/accounted state or schedule mismatch')
        require(len(set(digests.values())) == 1 and len({state_digest(v) for v in histories.values()}) == 1,
                'Original numerical equations changed')
        require(all(v == row['evaluations']['archived'] for v in row['evaluations'].values()) and
                all(v == row['activations']['archived'] for v in row['activations'].values()),
                'Native evaluation or activation observation changed')
        comparisons.append(dict(seed=row['seed'], activation_checkpointing=row['activation_checkpointing'],
                                actual_updates=5, original_state_sha256=next(iter(digests.values()))))

    replays = []
    pids = set()
    for row in component['arms']:
        arm = component_dir / f"arm-{row['seed']}-{int(row['checkpointing'])}"
        fresh = audit.json(arm / 'fresh/result.json')
        execution = audit.json(arm / 'fresh-execution.json')
        require(audit.json(arm / 'arm.json') == row and fresh == row['fresh'] and execution == row['execution'],
                'Retained fresh process records disagree')
        require(execution['actual_exit_code'] == 0 and json.loads(execution['stdout']) == fresh and
                execution['seconds'] < execution['deadline_seconds'] == 60, 'Fresh child execution failed')
        require(fresh['process_id'] not in pids and fresh['process_id'] != component['collection_invocation']['process_id'] and
                fresh['parent_process_id'] == component['collection_invocation']['process_id'], 'Not distinct actual child PIDs')
        pids.add(fresh['process_id'])
        require(fresh['actual_argv'] == execution['command'], 'Observed child invocation changed')
        require(fresh['restored_completed_updates'] == 2 and fresh['completed_updates'] == 5 and
                fresh['selected_ids'] == row['expected_selected_ids'] and
                fresh['history'] == row['full_successful_history'][2:] and
                state_digest(fresh['history']) == row['expected_history_sha256'], 'Actual tail/action replay differs')
        for name in ('clean/final-state.pt', 'fresh/final-state.pt'):
            path = arm / name
            audit.file(path)
            require(state_digest(torch.load(path, map_location='cpu', weights_only=True)) == row['expected_state_sha256'],
                    'Actual retained final tensor state differs')
        failed = audit.json(arm / 'retained/failed-suffix.json')
        require(failed == row['failed_suffix'] and failed['poisoned'] and failed['actual_loss_calls'] == 2 and
                set(failed['after']['failed_tickets']) <= set(fresh['work']['failed_tickets']), 'Failed attempt refunded')
        for key, value in failed['after']['reserved'].items():
            require(fresh['after_restore']['reserved'][key] >= value, 'Restoration rewound admitted work')
        comparisons_row = next(r for r in component['original_equations'] if r['seed'] == row['seed'] and
                               r['activation_checkpointing'] == row['checkpointing'])
        require(fresh['evaluation'] == comparisons_row['evaluations']['archived'] and
                fresh['activations'] == comparisons_row['activations']['archived'], 'Fresh native observations changed')
        require(fresh['work']['reserved']['training_valid_targets'] == 51 and
                fresh['work']['completed']['training_valid_targets'] == (43 if row['seed'] == 909 else 42),
                'Measured successful/reserved targets disagree')
        replays.append(dict(seed=row['seed'], activation_checkpointing=row['checkpointing'],
                            process_id=fresh['process_id'], exit_code=0, original_state_sha256=row['expected_state_sha256'],
                            completed_targets=fresh['work']['completed']['training_valid_targets'], reserved_targets=51,
                            known_partial_targets=fresh['work']['known_partial']['training_valid_targets'],
                            uncertain_upper_targets=fresh['work']['uncertain_upper']['training_valid_targets']))
    require(len(replays) == 4 and len(comparisons) == 4, 'Missing seed/checkpoint-mode arm')

    receipts = []
    for path in sorted(component_dir.rglob('*.work.json')):
        receipt = validate_story_receipt(audit.json(path))
        payload = Path(str(path).removesuffix('.work.json'))
        audit.file(payload, expected=receipt['payload_sha256'], expected_bytes=receipt['payload_bytes'],
                   relationship='independently retained story receipt to actual payload')
        receipts.append(str(path.relative_to(ROOT)))
    sampling = audit.json(component_dir / 'native-sampling/sampling.json')
    require(sampling == component['native_sampling'] and sampling['vocabulary'] == 50257 and sampling['eos'] == 50256 and
            all(v == sampling['outputs']['archived'] for v in sampling['outputs'].values()), 'Original native sampler changed')
    require(sampling['natural_eos_sequences'] == 0 and
            sampling['work']['completed']['generation_calls'] == 18 and
            sampling['work']['completed']['generation_positions'] == 72 and
            sampling['work']['completed']['multinomial_draws'] == 9, 'Actual sampler work differs')
    for row in component['failed_publication']:
        arm = component_dir / 'failed-publication' / row['kind']
        require(audit.json(arm / 'failure.json') == row and row['last_receipt_unchanged'] and
                not row['partial_receipt_exists'], 'Incomplete save acquired a receipt')
        audit.file(arm / 'completed0.pt', expected=row['old_payload_sha256'], relationship='failed save retained previous payload')
        audit.file(arm / 'completed0.pt.work.json', expected=row['old_receipt_sha256'], relationship='failed save retained previous receipt')
        partial = arm / ('incomplete1.pt' if row['kind'] == 'receipt' else 'incomplete1.pt.tmp')
        data = audit.file(partial, relationship='failed publication retains actual incomplete bytes')
        if row['kind'] == 'serialization': require(data == b'partial', 'Serializer failed bytes disappeared')

    default = audit.json(lesson_dir / 'native-work-example.json')
    refusal = audit.json(lesson_dir / 'lower-cap-retry-refusal.json')
    require((default['successful_targets'], default['reserved_targets']) == (17, 25) and
            default['original_record_equal'] and default['exact_numerical_recovery'] and
            default['original_numerical_sha256'] == default['restored_numerical_sha256'] and
            default['later_failed_work_retained'], 'Two-clock measured recovery changed')
    require((refusal['successful_targets'], refusal['reserved_targets']) == (9, 17) and refusal['retry_refused'] and
            refusal['next_refusal_forward_calls'] == 0 and refusal['refusal_numerical_state_unchanged'] and
            refusal['refusal_work_prefix_unchanged'], 'Cap24 refusal changed work or numerical state')
    for row in lesson['commands']:
        text = audit.file(row['log'], expected=row['log_sha256'], relationship='lesson process to actual log').decode()
        require(row['exit_code'] == 0 and not row['timed_out'], 'Lesson child did not pass')
        if row['name'] == 'tests': footer(text, 6)
    audit.file('notebooks/figures/chapter-06/day-09-01_read_training_clocks-01.png',
               expected=lesson['old_preview_sha256'], relationship='previous clock preview remains unchanged')
    preview = audit.file(lesson['preview']['path'], expected=lesson['preview']['sha256'],
                         expected_bytes=lesson['preview']['bytes'], relationship='fresh clock preview bytes')
    lesson_rows = notebook_panel(audit, lesson_dir / 'notebook/manifest.json')
    executed = audit.json(lesson_rows[0]['executed_artifact'])
    pngs = [out['data']['image/png'] for cell in executed['cells'] for out in cell.get('outputs', [])
            if 'image/png' in out.get('data', {})]
    last = pngs[-1]
    require(base64.b64decode(''.join(last) if isinstance(last, list) else last) == preview,
            'Preview is not the actual fresh-kernel output')

    logs = {}
    for row in cpu['commands']:
        require(row['exit_code'] == 0 and row['stopped_by'] is None and row['seconds'] < 180, 'Full bounded CPU child failed')
        logs[row['name']] = audit.file(row['log'], expected=row['log_sha256'], relationship='full CPU child to actual log').decode()
    require(len(logs) == 5, 'Full acceptance child missing')
    full_footer = footer(logs['tests'])
    math = re.fullmatch(r'(\d+) Markdown files, (\d+) math expressions, (\d+) issues\s*', logs['math'])
    require(math is not None and int(math[3]) == 0, 'Math acceptance failed')
    routes = json.loads(logs['routes'])
    require(routes['problems'] == [] and not routes.get('issues') and
            routes['registered_notebooks'] == routes['notebooks'] == 76, 'Route acceptance failed')
    notebook_rows = lesson_rows + notebook_panel(audit, cpu_dir / 'notebooks/manifest.json') + notebook_panel(audit, cpu_dir / 'rng-lesson/manifest.json')
    totals = {key: sum(row[key] for row in notebook_rows) for key in
              ('source_code_cells', 'executed_reference_code_cells', 'identity_preamble_code_cells', 'actual_kernel_code_cells', 'png_images')}
    totals['preserved_unfinished_exercise_cells'] = sum(len(row['preserved_unfinished_exercise_source_indices']) for row in notebook_rows)
    totals['notebooks'] = len(notebook_rows)
    campaign = audit.json('experiments/reports/2026-10-04-staged-campaign-preparation.json')['campaign']
    pending_fields = sum(len(row['actual']) for row in campaign['stage_rows'])
    require(len(campaign['stage_rows']) == 45 and pending_fields == 945 and not campaign['jobs_started'] and
            not campaign['actual_genealogy'] and all(value is None for row in campaign['stage_rows'] for value in row['actual'].values()),
            'External campaign is no longer unexecuted')
    ledger = audit.json('docs/course_improvements.json')
    require(ledger['summary']['complete'] == 13 and ledger['summary']['learner_day'] == 9, 'Scope/mastery changed')
    for directory in (component_dir, lesson_dir, cpu_dir, archive_dir): audit.tree(directory)
    merged = {}
    for values in source_maps:
        for name, expected in values.items():
            require(name not in merged or merged[name] == expected, 'Different measured executable revisions')
            merged[name] = expected
    audit.mapping(merged, 'unchanged source closure after root recheck')
    audit.file(Path(__file__), relationship='independent root collector identity')
    return dict(schema='dongxi-story-work-root-acceptance-v1', date_utc=datetime.now(timezone.utc).isoformat(),
                status='passed', scope='Retained native tiny CPU story work and selected course references; no new jobs launched',
                component_footers=component_footers, original_equation_comparisons=comparisons,
                actual_fresh_process_replays=replays, receipt_bindings=receipts,
                full_unittest_footer=full_footer, math=dict(markdown_files=int(math[1]), expressions=int(math[2]), issues=0),
                routes=routes, notebook_rows=notebook_rows, selected_notebook_totals=totals,
                executable_sources=len(cpu['source_sha256_before']), merged_source_sha256=merged,
                minimum_full_cpu_sampled_available_gib=cpu['minimum_sampled_available_gib'],
                minimum_component_prelaunch_available_gib=component['minimum_sampled_prelaunch_available_gib'],
                package_completion=13, learner_day=9, external_rows=45, external_actual_fields_null=945,
                external_jobs_started=0, actual_genealogy=[], production_ready=False, launch_authorized=False,
                unique_files_rechecked=len(audit.files), binding_relationships=len(audit.checks),
                file_sha256=audit.files, relationships=audit.checks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    result = render()
    with args.output.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in ('status', 'unique_files_rechecked', 'binding_relationships',
                     'full_unittest_footer', 'math', 'selected_notebook_totals')}, sort_keys=True))
