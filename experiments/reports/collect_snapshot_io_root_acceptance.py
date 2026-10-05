#!/usr/bin/env python3
"""Read-only acceptance rendering from completed, retained CPU evidence.

No tests, notebook kernels, model jobs, services or Git commands are launched.
The caller saves the rendered JSON/Markdown through apply_patch. This script
never overwrites a reference, source, notebook, journal or existing report.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / 'experiments/reports'
PREFIX = '/tmp/dongxi-course-reproduction.ZfVaEu/venv'


class Recheck:
    def __init__(self):
        self.files = {}
        self.checks = []

    def file(self, path, *, expected=None, expected_bytes=None, relationship=None):
        path = Path(path)
        if not path.is_absolute():
            path = ROOT / path
        name = str(path.relative_to(ROOT))
        data = path.read_bytes()
        identity = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
        if name in self.files and self.files[name] != identity:
            raise AssertionError(f'File changed during independent recheck: {name}')
        self.files[name] = identity
        if expected is not None and identity['sha256'] != expected:
            raise AssertionError(f'Hash mismatch: {name}')
        if expected_bytes is not None and len(data) != expected_bytes:
            raise AssertionError(f'Byte count mismatch: {name}')
        if relationship is not None:
            self.checks.append({'relationship': relationship, 'path': name,
                                'expected_sha256': expected,
                                'expected_bytes': expected_bytes})
        return data

    def json(self, path):
        return json.loads(self.file(path))

    def mapping(self, values, scope):
        for name, expected in values.items():
            self.file(name, expected=expected, relationship=scope)

    def tree(self, directory):
        for path in sorted(Path(directory).rglob('*')):
            if path.is_file():
                self.file(path)

    def snapshots(self, directory):
        result = []
        for marker in sorted(Path(directory).rglob('*.commit.json')):
            header = self.json(marker)
            payload = Path(str(marker)[:-len('.commit.json')])
            self.file(payload, expected=header['payload_sha256'],
                      expected_bytes=header['payload_bytes'],
                      relationship='retained snapshot marker to actual payload bytes')
            result.append(str(marker.relative_to(ROOT)))
        return result


def footer(text, expected=None):
    matches = re.findall(r'^Ran (\d+) tests? in ([0-9.]+)s$', text, re.MULTILINE)
    if len(matches) != 1 or not re.search(r'^OK(?: \(skipped=\d+\))?$', text, re.MULTILINE):
        raise AssertionError('Exactly one successful retained unittest footer required')
    count, seconds = int(matches[0][0]), float(matches[0][1])
    if expected is not None and count != expected:
        raise AssertionError('Retained unittest footer differs from expected component count')
    return {'tests': count, 'unittest_seconds': seconds, 'result': 'OK'}


def source_pair(audit, record, before_key, after_key, scope):
    before, after = record[before_key], record[after_key]
    if before != after:
        raise AssertionError(f'{scope}: before/after source identities differ')
    audit.mapping(before, scope + ': retained before/after/current source')
    return before


def notebook_panel(audit, manifest_path):
    manifest = audit.json(manifest_path)
    if manifest['failures']:
        raise AssertionError('Final fresh-kernel manifest contains failures')
    if manifest['kernel'] != 'dongxi-course-clean' or manifest['expected_kernel_prefix'] != PREFIX:
        raise AssertionError('Declared fresh-kernel identity differs')
    audit.file('docs/course_manifest.json', expected=manifest['course_manifest_sha256'],
               relationship='fresh-kernel manifest to registry bytes')
    audit.file('uv.lock', expected=manifest['lock_sha256'],
               relationship='fresh-kernel manifest to lock bytes')
    rows = []
    for row in manifest['notebooks']:
        source = json.loads(audit.file(row['path'], expected=row['sha256'],
                            relationship='fresh-kernel manifest to source notebook'))
        executed = audit.json(row['output'])
        identity = row['kernel_identity']
        if (identity['prefix'] != PREFIX or identity['executable'] != PREFIX + '/bin/python'
                or identity['cuda_available'] or identity['torch_cuda_build'] is not None):
            raise AssertionError('Observed fresh kernel is not the required CPU prefix')
        code_count = sum(cell['cell_type'] == 'code' for cell in source['cells'])
        skipped = row['skipped_unfinished_exercise_cells']
        images, image_cells, actual_kernel_cells = 0, [], 0
        for index, cell in enumerate(executed['cells']):
            if cell['cell_type'] != 'code':
                continue
            actual_kernel_cells += cell.get('execution_count') is not None
            for output in cell.get('outputs', []):
                if output.get('output_type') == 'error':
                    raise AssertionError('Executed notebook contains error output')
                data = output.get('data', {}).get('image/png')
                if data is not None:
                    if isinstance(data, list):
                        data = ''.join(data)
                    if not base64.b64decode(data, validate=True).startswith(b'\x89PNG\r\n\x1a\n'):
                        raise AssertionError('Notebook image output is not a PNG')
                    images += 1
                    image_cells.append(index - 1)
        reference_executed = code_count - len(skipped)
        if (code_count != row['code_cells'] or reference_executed != row['executed_code_cells']
                or images != row['images'] or image_cells != row['image_cells']
                or actual_kernel_cells != reference_executed + 1):
            raise AssertionError('Actual source/output cell or image counts disagree with manifest')
        rows.append({'path': row['path'], 'source_sha256': row['sha256'],
            'executed_artifact': row['output'], 'source_code_cells': code_count,
            'executed_reference_code_cells': reference_executed,
            'identity_preamble_code_cells': 1, 'actual_kernel_code_cells': actual_kernel_cells,
            'png_images': images, 'image_cells': image_cells,
            'preserved_unfinished_exercise_source_indices': skipped,
            'kernel_identity': identity})
    return rows


def render():
    audit = Recheck()
    sft_path = 'experiments/reports/2026-10-05-sft-snapshot-io-accounting/run-01/verification.json'
    dpo_path = 'experiments/reports/2026-10-05-dpo-snapshot-io/run-01/verification.json'
    schedule_path = 'experiments/reports/2026-10-05-snapshot-io-schedule/run-02/verification.json'
    core_path = 'experiments/reports/2026-10-05-snapshot-io-core-independent/run-04/verification.json'
    cpu_path = 'experiments/reports/2026-10-05-snapshot-io-cpu/run-02/cpu-verification.json'
    paths = dict(sft=sft_path, dpo=dpo_path, schedule=schedule_path, core=core_path, final_cpu=cpu_path)
    records = {name: audit.json(path) for name, path in paths.items()}
    sft, dpo, schedule, core, cpu = (records[name] for name in paths)
    if cpu['status'] != 'passed' or not cpu['inputs_unchanged'] or not cpu['sources_unchanged']:
        raise AssertionError('Final course-wide CPU acceptance must pass before rendering')
    if not all(row.get('exit_code') == 0 for row in cpu['commands']) or len(cpu['commands']) != 5:
        raise AssertionError('All five actual final acceptance children must pass')
    if sft['status'] != 'pass' or dpo['status'] != 'pass' or schedule['actual_exit_code'] or core['actual_exit_code']:
        raise AssertionError('A current component did not pass')

    sft_dir = ROOT / Path(sft_path).parent
    sft_before = audit.json(sft_dir / 'source-before.json')
    sft_after = audit.json(sft_dir / 'source-after.json')
    if not sft_before == sft_after == sft['source_sha256']:
        raise AssertionError('SFT source records disagree')
    audit.mapping(sft_before, 'SFT retained before/after/current source')
    audit.mapping(sft['artifact_sha256'], 'SFT recorded artifact to actual bytes')
    sft_tests = audit.json(sft_dir / 'tests.json')
    if sft_tests != sft['tests'] or sft_tests['exit_code']:
        raise AssertionError('SFT independently retained test transcript disagrees')
    sft_footer = footer(sft_tests['stdout'] + sft_tests['stderr'], 83)
    # A warning can intervene between a method's declaration and its final ok.
    # Count the actual declaration lines; the exact global OK footer proves
    # completion, rather than dropping methods with legitimate warning output.
    sft_modules = re.findall(r'^test_\S+ \((test_[^.]+)\.[^)]+\) \.\.\.', sft_tests['stderr'], re.MULTILINE)
    sft_new = sum(name == 'test_sft_snapshot_io_accounting' for name in sft_modules)
    if len(sft_modules) != 83 or sft_new != 19:
        raise AssertionError('SFT actual named test methods disagree with footer')
    sft_replays = []
    for arm in sft['arms']:
        if not arm['exact_numerical_replay'] or arm['execution']['exit_code']:
            raise AssertionError('A SFT fresh-process replay failed')
        sft_replays.append({'seed': arm['seed'], 'mode': arm['mode'],
            'fresh_process_exit_code': arm['execution']['exit_code'], 'exact_numerical_replay': True,
            'historical_numerical_sha256': arm['expected_numerical_sha256']})

    dpo_sources = source_pair(audit, dpo, 'source_sha256', 'source_sha256_after', 'DPO')
    dpo_dir = ROOT / Path(dpo_path).parent
    for name, expected in dpo_sources.items():
        audit.file(dpo_dir / 'source-capture' / name, expected=expected,
                   relationship='DPO archived source copy to measured source identity')
    dpo_footers = []
    for command in dpo['commands']:
        raw = command['raw_log']
        text = audit.file(dpo_dir / raw['path'], expected=raw['sha256'], expected_bytes=raw['bytes'],
                         relationship='DPO subprocess record to retained log bytes').decode()
        if command['exit_code']:
            raise AssertionError('DPO retained subprocess failed')
        dpo_footers.append(footer(text, command['tests_ran']))
    if [row['tests'] for row in dpo_footers] != [16, 150]:
        raise AssertionError('Unexpected DPO component footer counts')

    schedule_sources = source_pair(audit, schedule, 'source_before', 'source_after', 'schedule')
    schedule_dir = ROOT / Path(schedule_path).parent
    schedule_tests = audit.json(schedule_dir / 'tests.json')
    schedule_footer = footer(schedule_tests['stdout'] + schedule_tests['stderr'], 26)
    source_pair(audit, schedule, 'historical_before', 'historical_after', 'preserved DPO mapper')
    core_sources = source_pair(audit, core, 'source_before', 'source_after', 'independent core')
    core_dir = ROOT / Path(core_path).parent
    core_tests = audit.json(core_dir / 'tests.json')
    core_footer = footer(core_tests['stdout'] + core_tests['stderr'], 58)
    correction_path = 'experiments/reports/2026-10-05-snapshot-io-core-independent-count-correction.json'
    correction = audit.json(correction_path)
    audit.file(core_dir / 'tests.json', expected=correction['final_test_log_sha256'],
               relationship='explicit core count correction to actual final transcript')
    audit.file('experiments/reports/2026-10-05-snapshot-io-core-independent/run-03/tests.json',
               expected=correction['earlier_test_log_sha256'],
               relationship='explicit core count correction to retained earlier transcript')
    core_acceptance = audit.json('experiments/reports/2026-10-05-snapshot-io-core-independent-final-acceptance.json')
    audit.mapping(core_acceptance['source_sha256'], 'independent final core acceptance source')
    audit.mapping(core_acceptance['artifact_sha256'], 'independent final core acceptance artifact')

    cpu_sources = source_pair(audit, cpu, 'source_sha256_before', 'source_sha256_after', 'final whole-course CPU')
    current_inputs = {}
    for name, expected in cpu['input_sha256'].items():
        data = audit.file(name)
        observed = hashlib.sha256(data).hexdigest()
        if observed != expected and name != 'docs/course_improvements.json':
            raise AssertionError('A scientific lock/registry input changed after final collection')
        current_inputs[name] = {'measured_sha256': expected, 'currently_observed_sha256': observed,
            'currently_matches_measured': observed == expected,
            'interpretation': 'same bytes' if observed == expected else
                'canonical explanatory scope/next-safe-action amendment after the passed measurement; original panel input identity remains historical, not silently rewritten'}
    final_logs = {}
    for row in cpu['commands']:
        text = audit.file(row['log'], expected=row['log_sha256'],
                         relationship='final CPU subprocess to retained log').decode()
        final_logs[row['name']] = text
    full_footer = footer(final_logs['tests'])
    math_match = re.fullmatch(r'(\d+) Markdown files, (\d+) math expressions, (\d+) issues\s*', final_logs['math'])
    if not math_match or int(math_match[3]):
        raise AssertionError('Final math log is not a clean measured result')
    routes = json.loads(final_logs['routes'])
    if routes['registered_notebooks'] != routes['notebooks'] or routes.get('issues'):
        raise AssertionError('Final registry/integrity result failed')
    notebook_rows = []
    for path in ('notebooks/manifest.json', 'rng-lesson/manifest.json'):
        notebook_rows.extend(notebook_panel(audit, ROOT / Path(cpu_path).parent / path))
    totals = {key: sum(row[key] for row in notebook_rows) for key in
        ('source_code_cells', 'executed_reference_code_cells', 'identity_preamble_code_cells',
         'actual_kernel_code_cells', 'png_images')}
    totals['preserved_unfinished_exercise_cells'] = sum(len(row['preserved_unfinished_exercise_source_indices'])
                                                      for row in notebook_rows)
    totals['notebooks'] = len(notebook_rows)

    failed_path = 'experiments/reports/2026-10-05-snapshot-io-cpu/run-01/cpu-verification.json'
    failed = audit.json(failed_path)
    if failed['status'] != 'failed':
        raise AssertionError('Initial kernel discovery failure must remain historical')
    failed_logs = {}
    for row in failed['commands']:
        failed_logs[row['name']] = audit.file(row['log'], expected=row['log_sha256'],
                                  relationship='preserved failed CPU run to retained log').decode()
    failed_manifest = audit.json(ROOT / Path(failed_path).parent / 'notebooks/manifest.json')
    if len(failed_manifest['failures']) != 5 or any('NoSuchKernel' not in row['error']
                                                  for row in failed_manifest['failures']):
        raise AssertionError('Initial failure cause differs from preserved kernel-discovery diagnostic')

    snapshot_markers = []
    for directory in (sft_dir, dpo_dir):
        snapshot_markers.extend(audit.snapshots(directory))
    for directory in (sft_dir, dpo_dir, schedule_dir, core_dir,
                      ROOT / Path(cpu_path).parent, ROOT / Path(failed_path).parent):
        audit.tree(directory)
    for path in ('experiments/reports/2026-10-05-sft-snapshot-io-accounting.md',
                 'experiments/reports/2026-10-05-dpo-snapshot-io.md',
                 'experiments/reports/2026-10-05-dpo-snapshot-io-development-01.json',
                 'experiments/reports/2026-10-05-snapshot-io-schedule-node-bound.md'):
        audit.file(path)

    campaign_path = 'experiments/reports/2026-10-04-staged-campaign-preparation.json'
    campaign = audit.json(campaign_path)['campaign']
    rows = campaign['stage_rows']
    if (len(rows) != 45 or campaign['jobs_started'] != 0 or campaign['actual_genealogy']
            or any(value is not None for row in rows for value in row['actual'].values())):
        raise AssertionError('Frozen campaign must still have45 null rows, no jobs and no genealogy')
    improvements = audit.json('docs/course_improvements.json')
    if improvements['summary']['complete'] != 13 or improvements['summary']['learner_day'] != 9:
        raise AssertionError('Bounded package/learner scope differs from retained expected state')
    combined_sources = {}
    for mapping in (sft_before, dpo_sources, schedule_sources, core_sources, cpu_sources):
        for name, expected in mapping.items():
            if name in combined_sources and combined_sources[name] != expected:
                raise AssertionError('Components bind different current source identities')
            combined_sources[name] = expected
    source_after = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in combined_sources}
    if source_after != combined_sources:
        raise AssertionError('Bound executable/specification closure changed during independent collection')
    audit.file(Path(__file__), relationship='read-only acceptance collector identity')

    acceptance = {
        'schema': 'dongxi-snapshot-io-root-acceptance-v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'bounded_cpu_source_pass', 'collector_invocation': list(sys.orig_argv),
        'collection': 'read-only independent byte/count recheck; no test or kernel reruns in this collector',
        'retained_collector_development_diagnostic': {
            'actual_exit_code': 1,
            'error': 'AssertionError: SFT actual named test methods disagree with footer',
            'cause': 'first read-only renderer regex required method and ok on the same line; one existing method emitted an intervening warning;83 actual declarations/19 new methods and the retained OK footer are unchanged',
            'repair': 'count declaration lines and separately require the exact successful footer; no reference/source/test/recipe/cap changes or reruns'},
        'component_evidence': {
            'sft': {'record': sft_path, 'footer': sft_footer, 'new_controls': sft_new,
                    'existing_controls': 83-sft_new, 'sources': len(sft_before),
                    'recorded_artifacts_rehashed': len(sft['artifact_sha256']),
                    'source_archives': 'none supplied; retained before/after identities rehashed against current files',
                    'fresh_process_replays': sft_replays},
            'dpo': {'record': dpo_path, 'footer_components': dpo_footers, 'tests': 166,
                    'sources': len(dpo_sources), 'source_archives_rehashed': len(dpo_sources),
                    'fresh_process_replay': dpo['fresh_process_replay'],
                    'checkpoint_off_on_parity': dpo['checkpoint_off_on_parity']},
            'schedule': {'record': schedule_path, 'footer': schedule_footer,
                         'exhaustive_pure_schedule_cases': schedule['exhaustive_schedule_cases'],
                         'sources': len(schedule_sources), 'execution_scope': schedule['scope']},
            'independent_core': {'record': core_path, 'footer': core_footer, 'sources': len(core_sources),
                'raw_incorrect_focused_tests_field': core['focused_tests'],
                'explicit_count_correction': correction_path,
                'counts': {'snapshot_io_budget': 24, 'training_snapshot': 8, 'snapshot_io_schedule': 26},
                'six_independently_reproduced_controls_pass': core['all_six_controls_pass']}},
        'count_policy': 'Component panels overlap each other and the full971-test discovery;83+166+26+58 is not a unique test total. Use actual individual footers and the full discovery footer separately.',
        'final_course_cpu_panel': {'record': cpu_path, 'actual_children': cpu['commands'],
            'unittest_footer': full_footer, 'math': {'markdown_files': int(math_match[1]),
                'expressions': int(math_match[2]), 'issues': int(math_match[3])},
            'routes': {'registered_notebooks': routes['registered_notebooks'],
                       'chapters': routes['chapters'], 'solutions': routes['solutions'],
                       'issues': len(routes.get('issues', []))},
            'executable_sources_before_after': len(cpu_sources), 'inputs_unchanged': cpu['inputs_unchanged'],
            'source_before_after_equal': cpu['sources_unchanged'],
            'minimum_sampled_available_gib': cpu['minimum_sampled_available_gib'],
            'required_host_reserve_gib': cpu['required_host_reserve_gib'],
            'notebook_counts': totals, 'notebooks': notebook_rows,
            'kernel_discovery': 'Corrected invocation inherited explicit JUPYTER_PATH=/tmp/dongxi-course-reproduction.ZfVaEu/kernel/share/jupyter; each actual returned identity is checked independently.',
            'scope': 'six selected fresh CPU references, not all76 notebooks; skips preserve unfinished learner exercises; reference reproduction is not learner mastery'},
        'preserved_initial_full_panel_failure': {'record': failed_path,
            'status': failed['status'], 'actual_unittest_footer': footer(failed_logs['tests']),
            'notebook_child_exit': next(row['exit_code'] for row in failed['commands'] if row['name']=='notebooks'),
            'failed_notebooks': failed_manifest['failures'],
            'cause': 'root invocation omitted external JUPYTER_PATH;5 NoSuchKernel failures, no source/cap/recipe changes; fixed environment discovery only'},
        'independent_binding_recheck': {'unique_files_rehashed': len(audit.files),
            'recorded_relationship_checks': len(audit.checks),
            'snapshot_marker_payload_checks': len(snapshot_markers),
            'snapshot_markers': snapshot_markers, 'all_recorded_bindings_match': True,
            'combined_bound_source_count': len(combined_sources),
            'combined_source_sha256_before': combined_sources,
            'combined_source_sha256_after': source_after,
            'source_closure_stable_during_recheck': True,
            'file_identities': audit.files, 'relationships': audit.checks,
            'scope': 'trusted retained local files/metadata; hashes are byte identity, not external supplier authenticity or new numerical replay'},
        'course_state': {'bounded_packages_complete': 13, 'total_packages': 18, 'learner_day': 9,
            'external_campaign_record': campaign_path, 'external_stage_rows': 45,
            'all_external_actual_outcomes_null': True, 'model_scale_jobs_started': 0,
            'new_actual_checkpoint_genealogy': [], 'notebook_mastery_claim': False},
        'postmeasurement_input_recheck': current_inputs,
        'production_ready': False, 'launch_authorized': False,
        'pending': ['RLVR actual consumer shared snapshot-I/O integration',
            'caller-owned state capture and later state application; artifact inventory and metadata/journal/source/parent/input identity costs excluded from shared9 dimensions',
            'persistent story work and other-stage complete work adapters; broader logs/exports/scratch and output routing',
            'supplier authenticity, live production re-encoding and invocation enforcement',
            'physical containment/quota/watchdog/private boundary and independently granted production authorization',
            'actual pretrained/Spark/CUDA/BF16 profile, smoke, recovery and pilot; actual Mac/hosted/cross-machine evidence',
            'dependent external course final defense; no overall goal completion'],
        'acceptance_boundary': 'SFT16/DPO19 scientific work and separate snapshotIO9 CPU/source receipts retain original recipes/reference/Adam/cursor/RNG/checkpoint modes and failed spending. Declared shared visits/bytes are not all CPU instructions, time, disk/RAM/GPU quota or production hostile-payload isolation.'}
    narrative = f"""# Snapshot I/O root acceptance

The retained final CPU panel passes {full_footer['tests']} tests, {int(math_match[2])} math
expressions with zero issues and {routes['registered_notebooks']} registered routes.
Independent read-only rechecking found no byte-binding mismatch. This closes the
bounded SFT/DPO shared-reader source/CPU gate, not production or the overall goal.

## Actual counts, not additive component totals

| Retained panel | Actual successful unittest footer | Scope |
| --- | --- | --- |
| SFT run01 |83 tests;40.611s |19 new and64 earlier controls; four full/LoRA fresh-process replays |
| DPO run01 |16 tests;9.214s +150 tests;30.291s |166 overlapping controls; actual fresh checkpoint-ON3→6 replay |
| Schedule run02 |26 tests;0.211s |960 exhaustive pure arithmetic cases; no model/reader execution |
| Independent core run04 |58 tests;1.205s |24 I/O +8 snapshot +26 schedule methods; six independent guards |
| Final course CPU run02 |{full_footer['tests']} tests;{full_footer['unittest_seconds']}s |Current complete unittest discovery |

These panels overlap and must not be summed into a unique-test claim. The core
raw `focused_tests=60` inspector field is incorrect. Its unchanged transcript
says58; the explicit correction also fixes earlier run03 from58 to56. Both raw
records remain historical rather than being overwritten.

## Fresh references and current byte identities

Final run02 has six selected fresh references: {totals['source_code_cells']} source
code cells, {totals['executed_reference_code_cells']} executed reference cells,
{totals['png_images']} actual PNG outputs and {totals['preserved_unfinished_exercise_cells']}
preserved unfinished-exercise skips. Six additional identity preambles make
{totals['actual_kernel_code_cells']} actual executed kernel code cells. Each returned
kernel identity matches the existing isolated CPU prefix; this is not all76
notebook reproduction or learner mastery. Day25 alone has nine source/executed
cells and seven figures, including the actual pre-read admission microscope.

The final panel retains {len(cpu_sources)} unchanged executable source identities.
This acceptance independently rehashes {len(audit.files)} unique local files,
including all201 recorded SFT artifacts, all30 DPO source archives, retained
logs and {len(snapshot_markers)} snapshot marker→payload byte/size bindings.
The combined executable/specification source closure is stable across this
read-only recheck. SFT/core/schedule provide source identity records rather than
archived source copies; the acceptance does not claim absent archives exist.
Full per-file SHA256/byte identities and checked relationships are in the
adjacent machine-readable acceptance record.

The canonical improvement ledger's explanatory scope/next-safe-action text was
updated after this passed measurement. Its original measured input SHA remains
historical and its current SHA is recorded separately; this is not disguised as
an identical-byte input. The lock/registry and170 executable source identities
still match, and13of18/Day9/45null status is unchanged. The panel's original
`inputs_unchanged=true` refers only to its own measurement interval.

The initial full run01 is preserved:971 tests, math and routes passed, but five
notebook starts failed with `NoSuchKernel` because the root invocation omitted
external `JUPYTER_PATH`. No source, cap, science or tolerance changed. The final
run02 uses the explicit existing kernel-discovery path and all five acceptance
children exit zero. Minimum sampled available host memory is
{cpu['minimum_sampled_available_gib']:.6f}GiB against the declared25GiB reserve;
this observation is not physical containment.

## What the accepted evidence does and does not say

SFT16/DPO19 semantic-work caps stay separate from the strict nine-dimensional
shared snapshot-I/O contract. Actual source ordering binds independently
retained receipt prefixes before payload inspection and model allocation;
zero-cap controls retain spending/refusals. Actual CPU replay preserves the
original numerical policy/reference/Adam/cursor/RNG/history and checkpoint
modes. CLI hardware/interface providers are explicitly mocked—not real Spark
acceptance. The root receipt independently verifies retained byte/count
bindings; it does not rerun these numerical collectors or authenticate external
suppliers.

Still pending: RLVR's actual shared-I/O consumer; caller state capture and later
application, metadata/journal/identity costs and postadmission artifact inventory;
persistent story/other-stage work and broader outputs; supplier/live encoding and
invocation gates; physical containment/quota/watchdog and independent production
authority; pretrained/Spark/CUDA/BF16 and actual Mac/hosted/cross-machine evidence.
Declared shared visits/bytes are not an all-CPU/time/storage/hostile-payload quota.

The frozen campaign still has45 null external rows, zero model-scale jobs and
no new checkpoint genealogy. Course status remains13of18 bounded packages and
learner Day9. No production launch is authorized and the overall goal remains
unfinished. Historical failures, negative generation and resource journals are
preserved; this evidence does not create a capability-success claim.

Evidence: [final full CPU run02](2026-10-05-snapshot-io-cpu/run-02/cpu-verification.json),
[SFT83](2026-10-05-sft-snapshot-io-accounting/run-01/verification.json),
[DPO166](2026-10-05-dpo-snapshot-io/run-01/verification.json),
[schedule26](2026-10-05-snapshot-io-schedule/run-02/verification.json),
[core58 count correction](2026-10-05-snapshot-io-core-independent-count-correction.json),
[root byte/count receipt](2026-10-05-snapshot-io-root-acceptance.json).
"""
    return {'acceptance': acceptance, 'narrative': narrative}


if __name__ == '__main__':
    if sys.argv[1:] != ['--render']:
        raise SystemExit('Use --render: read-only JSON/Markdown rendering after final acceptance passes')
    print(json.dumps(render(), sort_keys=True, allow_nan=False))
