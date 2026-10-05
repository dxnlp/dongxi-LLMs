#!/usr/bin/env python3
"""Closed, offline assistant full/LoRA400 pilots under the declared3600s watchdog.

Every arm starts freshly from pinned Base. Preparation hashes local inputs but
does not import a tokenizer/model or start a child. The assistant-pilot3600s
envelope preserves the original profile900s route; completion is not forecast.
There is no arbitrary argv, resume, acquisition or implicit retry route.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dongxi_llms.native_profile_supervisor import (
    MODEL, REVISION, SOURCES, _digest, _native_probe, _new_target, _supervise)
from dongxi_llms.run_identity import artifact_hashes, cached_snapshot, canonical_hash
from dongxi_llms.snapshot_io_budget import io_budget_contract

GIB = 1024**3
INTERPRETER = '/home/dongxi/dgx-spark-dongxi/.venv/bin/python'
ENVIRONMENT_LOCK = '/home/dongxi/dgx-spark-dongxi/uv.lock'
UPDATES = 400
SECONDS = 3600
CHECKPOINT_EVERY = 100
WORK_CAPS = dict(train_updates=400, sampled_examples=1600, selector_steps=1600,
    valid_targets=10041, logical_sequence_tokens=69740,
    policy_forward_calls=2744, policy_forward_positions=134866,
    evaluation_calls=1144, evaluation_positions=71488, generation_calls=16,
    generation_position_upper_bound=66688, generation_tokens=1024,
    recovery_validation_operations=5, recovery_history_rows=1000,
    recovery_tensor_elements=15000000000, recovery_rng_states=15)
IO_ENVELOPE = dict(max_payload_bytes=5*GIB, max_tree_nodes=50000,
    max_tensor_elements=2000000000, max_tensor_bytes=4*GIB,
    max_primitive_bytes=1024**2)
IO_LIMITS = dict(snapshot_inspect_operations=0, snapshot_load_operations=0,
    snapshot_save_operations=5, snapshot_hash_bytes=25*GIB,
    snapshot_tree_nodes=250000, snapshot_tensor_elements=10000000000,
    snapshot_primitive_bytes=5*1024**2, snapshot_clone_bytes=20*GIB,
    snapshot_serialization_bytes=25*GIB)
ENCODED = dict(train='540106898fc8a864976228098002a30e316c465f5b64ea654e9fcbeb64fddc04',
    dev='633084d98aeb74eeca2415a7fddac757350eb14af33800b2abff392b51d7f30c')
OWN_SOURCES = ('scripts/run_native_sft_pilots.py', 'tests/test_native_sft_pilots.py',
    'experiments/specs/2026-10-04-staged-spark-campaign.md')


def retain(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def paths(mode, run_id):
    if mode not in ('full', 'lora') or type(run_id) is not str or re.fullmatch(r'run-[0-9]{2}', run_id) is None:
        raise ValueError('Only full/lora and a fixed run-NN identifier are supported')
    stem = f'native-sft-{mode}-pilot400-20261005-{run_id}'
    return dict(evidence=ROOT/'experiments/reports'/stem,
        output=ROOT/'outputs'/stem, journals=ROOT/'outputs'/(stem+'-journals'))


def source_bindings():
    return {name: _digest(str(ROOT/name)) for name in (*SOURCES, *OWN_SOURCES)}


def local_base_binding():
    snapshot = cached_snapshot(MODEL, REVISION)
    files = artifact_hashes(snapshot)
    if 'config.json' not in files or not any(name.endswith('.safetensors') for name in files):
        raise ValueError('Complete already cached pinned Base weights are required')
    return dict(path=str(snapshot), model=MODEL, revision=REVISION,
        files=files, revision_evidence='declared pinned cache directory and actual streamed artifact bytes')


def fixed_inputs(mode):
    receipt = ('2026-10-05-native-sft-full-replay/acceptance-retry02.json' if mode == 'full'
               else '2026-10-05-native-sft-lora-replay/acceptance.json')
    values = dict(train=ROOT/'outputs/course-sft-interface-v1/train.jsonl',
        dev=ROOT/'outputs/course-sft-interface-v1/dev.jsonl',
        template=ROOT/'experiments/data/instruction_interface_v1.jinja',
        data_card=ROOT/'experiments/data/instruction-interface-v1-data-card.json',
        recovery_acceptance=ROOT/'experiments/reports'/receipt,
        sizing_reference=ROOT/'outputs/native-sft-full-replay-20261005-original/config.json',
        interpreter=Path(INTERPRETER), environment_lock=Path(ENVIRONMENT_LOCK))
    return {key: _digest(str(value), executable=key == 'interpreter') for key, value in values.items()}


def verify_fixed_population(inputs, base):
    read = lambda key: json.loads(Path(inputs[key]['path']).read_text())
    card, reference, acceptance = (read(key) for key in ('data_card', 'sizing_reference', 'recovery_acceptance'))
    if acceptance.get('status') != 'passed' or not acceptance.get('checks') or not all(value is True for value in acceptance['checks'].values()):
        raise ValueError('Actual pretrained completed-update recovery must have passed')
    for key, count in (('train', 240), ('dev', 60)):
        if inputs[key]['sha256'] != card['splits'][key]['sha256']:
            raise ValueError('Original declared source population changed: '+key)
        rows = [json.loads(line) for line in Path(inputs[key]['path']).read_text().splitlines() if line.strip()]
        if len(rows) != count or len({row['id'] for row in rows}) != count:
            raise ValueError('Original fixed unique-ID population required: '+key)
        if reference.get('encoded_'+key+'_sha256') != ENCODED[key] or reference.get(key+'_sha256') != inputs[key]['sha256']:
            raise ValueError('Observed original tokenizer geometry no longer matches: '+key)
    if (reference.get('template_sha256') != inputs['template']['sha256']
            or reference.get('base_checkpoint_files') != base['files']
            or reference.get('model') != MODEL or reference.get('revision') != REVISION
            or reference.get('tokenizer_revision') != REVISION):
        raise ValueError('Original template/pinned local Base binding changed')


def fixed_command(mode, locations, inputs):
    evidence = locations['evidence']
    argv = [INTERPRETER, str(ROOT/'scripts/run_chapter09_spark_sft.py'),
        '--model', MODEL, '--revision', REVISION, '--tokenizer-revision', REVISION]
    for key in ('train', 'dev', 'template'):
        argv.extend(['--'+key, inputs[key]['path']])
    return argv + ['--output', str(locations['output']), '--mode', mode, '--rank', '8',
        '--updates', '400', '--microbatch', '1', '--accumulation', '4',
        '--max-length', '256', '--learning-rate', '2e-05', '--runtime-seconds', '3600',
        '--reserve-gib', '25.0', '--seed', '1212', '--checkpoint-every', '100',
        '--snapshot-max-bytes', str(5*GIB), '--work-limits', str(evidence/'work-caps.json'),
        '--work-journal-max-bytes', str(4*1024**2),
        '--work-journal', str(locations['journals']/'work.jsonl'),
        '--snapshot-io-limits', str(evidence/'io-caps.json'),
        '--snapshot-io-ledger', str(locations['journals']/'io.jsonl'),
        '--environment-lock', ENVIRONMENT_LOCK]


def prepare(mode, run_id):
    locations = paths(mode, run_id)
    # Resolve all targets first, so an existing output never causes partial reuse.
    for path in locations.values():
        _new_target(str(path))
    evidence = locations['evidence']
    evidence.mkdir(mode=0o700)
    try:
        if shutil.disk_usage(ROOT/'outputs').free < 30*GIB:
            raise RuntimeError('30GiB output planning allowance unavailable')
        sources, inputs, base = source_bindings(), fixed_inputs(mode), local_base_binding()
        verify_fixed_population(inputs, base)
        locations['journals'].mkdir(mode=0o700)
        for filename in ('work.jsonl', 'io.jsonl'):
            _new_target(str(locations['journals']/filename), private=True)
        io = io_budget_contract(IO_LIMITS, IO_ENVELOPE, 4*1024**2)
        retain(evidence/'work-caps.json', WORK_CAPS)
        retain(evidence/'io-caps.json', io)
        argv = fixed_command(mode, locations, inputs)
        for key, filename in (('work_caps', 'work-caps.json'), ('io_caps', 'io-caps.json')):
            inputs[key] = _digest(str(evidence/filename))
        record = dict(schema='dongxi-fixed-native-sft400-v1', mode=mode, run_id=run_id,
            utc=datetime.now(timezone.utc).isoformat(), stage_id=f'assistant-sft-{mode}-pilot',
            source_bindings=sources, input_bindings=inputs, local_base_binding=base, argv=argv,
            output=str(locations['output']), evidence=str(evidence), execution_requested=False,
            limits=dict(invocations=1, updates=400, external_seconds=3600,
                staged_seconds_ceiling=3600, reserve_bytes=25*GIB, commit_cursors=[0,100,200,300,400],
                output_planning_bytes=30*GIB, work_caps=deepcopy(WORK_CAPS), snapshot_io_contract=io),
            sizing=dict(encoded_sha256=ENCODED, training_targets=9321, training_positions=63378,
                dev_targets_per_panel=360, dev_positions_per_panel=2400,
                generation_prefix_lengths=[29,36,37,29,36,37,29,36],
                scope='original byte-bound tokenizer geometry; allowances are not measured model work'),
            science='fresh pinned Base; fixed400 horizon; final full arm remains predeclared downstream parent',
            scope='actual bounded pilot when explicitly executed; no publication/merge/DPO/quality-pass claim',
            boundaries=['sampled watchdog, not physical quota',
                'export and independent byte inventories are outside SFT16/I/O9',
                'declared3600-second stop can prevent completion; no automatic retry'])
        record['preparation_sha256'] = canonical_hash(record)
        retain(evidence/'preparation.json', record)
        return record
    except BaseException as error:
        retain(evidence/'preparation-failure.json', dict(status='failed', type=type(error).__name__, message=str(error)))
        raise


def verify_prepared(record):
    expected = {key: value for key, value in record.items() if key != 'preparation_sha256'}
    if canonical_hash(expected) != record.get('preparation_sha256'):
        raise ValueError('Prepared closed command or limits changed')
    locations = paths(record['mode'], record['run_id'])
    if record['output'] != str(locations['output']) or record['evidence'] != str(locations['evidence']):
        raise ValueError('Only the fixed pilot output/evidence paths are supported')
    if record['argv'] != fixed_command(record['mode'], locations, record['input_bindings']):
        raise ValueError('Only the fixed400 pilot command is supported; no extra argv')
    current_inputs = fixed_inputs(record['mode'])
    if any(record['input_bindings'].get(key) != value for key,value in current_inputs.items()):
        raise ValueError('Bound input changed after preparation or input role substituted')
    if source_bindings() != record['source_bindings']:
        raise ValueError('Executable source changed after preparation')
    for key, expected in record['input_bindings'].items():
        if _digest(expected['path'], executable=key == 'interpreter') != expected:
            raise ValueError('Bound input changed after preparation: '+key)
    if local_base_binding() != record['local_base_binding']:
        raise ValueError('Local pinned Base bytes changed after preparation')
    if (json.loads((locations['evidence']/'work-caps.json').read_text()) != WORK_CAPS
            or json.loads((locations['evidence']/'io-caps.json').read_text())
                != io_budget_contract(IO_LIMITS, IO_ENVELOPE, 4*1024**2)):
        raise ValueError('Only the fixed predeclared work/I/O caps are supported')


def execute(record, declaration):
    if type(declaration) is not str or not 12 <= len(declaration) <= 4096:
        raise ValueError('Explicit recorded goal-authorized pilot declaration required')
    evidence, output = Path(record['evidence']), Path(record['output'])
    supervision = None
    try:
        verify_prepared(record)
        _new_target(str(output))
        retain(evidence/'launch.json', dict(argv=record['argv'], operator_declaration=declaration,
            preparation_sha256=record['preparation_sha256'], source_bindings=record['source_bindings'],
            input_bindings=record['input_bindings'], local_base_binding=record['local_base_binding']))
        supervision = _supervise(record['argv'], evidence/'supervision', native=True,
            seconds=SECONDS, probe=_native_probe, operator_declaration=declaration,
            native_stage='assistant-pilot')
        retain(evidence/'returned-supervision.json', supervision)
        verify_prepared(record)
        retain(evidence/'closing-bindings.json', dict(status='unchanged',
            source_bindings=record['source_bindings'], input_bindings=record['input_bindings'],
            local_base_binding=record['local_base_binding']))
        if supervision['status'] != 'completed' or supervision['actual_exit_code'] != 0:
            raise RuntimeError('Actual owned pilot failed; returned supervision retained')
        result = json.loads((output/'result.json').read_text())
        config = json.loads((output/'config.json').read_text())
        metrics = [json.loads(line) for line in (output/'metrics.jsonl').read_text().splitlines()]
        checks = dict(completed400=result.get('status') == 'completed' and result.get('updates') == 400,
            exactly400_metrics=len(metrics) == 400 and [row.get('update') for row in metrics] == list(range(1,401)),
            original_encoded_geometry=all(config.get('encoded_'+key+'_sha256') == value for key,value in ENCODED.items()),
            no_open_or_failed_work=all(not result[key][field] for key in ('cumulative_work_ledger','cumulative_snapshot_io_ledger')
                for field in ('open_tickets','failed_tickets')))
        exported = artifact_hashes(output/'policy')
        genealogy = json.loads((output/'policy/course-genealogy.json').read_text())
        checks['fresh_pinned_base'] = (genealogy.get('base_model') == MODEL and genealogy.get('base_revision') == REVISION
            and genealogy.get('base_checkpoint_files') == record['local_base_binding']['files'])
        checks['five_completed_snapshots'] = all((output/f'checkpoint-{cursor:06d}.pt.commit.json').is_file()
            for cursor in (0,100,200,300,400))
        checks['export_kind'] = genealogy.get('kind') == ('full-HF-model' if record['mode'] == 'full' else 'PEFT-adapter-requires-pinned-base')
        if record['mode'] == 'full':
            checks['loadable_full_policy'] = ('config.json' in exported and 'model.safetensors' in exported
                and 'adapter_config.json' not in exported)
        verify_prepared(record)
        summary = dict(status='passed' if all(checks.values()) else 'failed', checks=checks,
            result=result, exported_policy=dict(path=str(output/'policy'), files=exported, genealogy=genealogy),
            actual_exit_code=supervision['actual_exit_code'], child_seconds=supervision['child_seconds'],
            scope='completed fixed400 pilot and byte-bound export; quality, publication, LoRA merge and preference/RLVR remain separate')
        retain(evidence/'acceptance.json', summary)
        if summary['status'] != 'passed':
            raise RuntimeError('Completed pilot acceptance failed; evidence retained')
        print(json.dumps(dict(status='passed', mode=record['mode'], evidence=str(evidence), policy=str(output/'policy'))))
        return 0
    except BaseException as error:
        retain(evidence/'failure.json', dict(status='failed', type=type(error).__name__, message=str(error),
            actual_supervision_retained=supervision is not None,
            scope='attempt remains retained; no automatic restart, cap change or quality substitution'))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('full','lora'), required=True)
    parser.add_argument('--run-id', required=True, help='Fixed run-NN identifier; all targets must be new')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--operator-declaration')
    args = parser.parse_args(argv)
    if args.execute and (not args.operator_declaration or not 12 <= len(args.operator_declaration) <= 4096):
        parser.error('Explicit recorded goal-authorized pilot declaration required')
    record = prepare(args.mode, args.run_id)
    if args.execute:
        return execute(record, args.operator_declaration)
    print(json.dumps(dict(status='prepared-not-executed', evidence=record['evidence'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
