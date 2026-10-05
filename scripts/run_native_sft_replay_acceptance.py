#!/usr/bin/env python3
"""Explicit fixed Spark SFT replay experiment; never part of CPU verification.

The only choices are full and rank-8 Q/V LoRA on the already pinned Base.
There is no download, generic argv, longer horizon, or pilot route. The owned
watchdog is reused without weakening its900-second/25GiB bounds. Preparation
alone does not execute a child. Each failed invocation remains in its directory.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dongxi_llms.native_profile_supervisor import (
    MODEL, REVISION, SOURCES, _digest, _native_probe, _supervise)

GIB = 1024**3


def retain(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def bindings():
    names = (*SOURCES, 'scripts/run_native_sft_replay_acceptance.py',
             'experiments/specs/2026-10-05-native-sft-replay.md')
    return {name: _digest(str(ROOT/name)) for name in names}


def tensor_bytes(path):
    """Hash serialized tensor entries only; no tensor or pickle deserialization.

    Names/order follow this same unchanged torch.save producer. This does not
    establish arbitrary archive semantic equivalence or Python RNG equality.
    """
    result = {}
    with zipfile.ZipFile(path) as archive:
        for entry in archive.infolist():
            pieces = entry.filename.split('/')
            if len(pieces) == 3 and pieces[1] == 'data' and pieces[2].isdigit():
                digest = hashlib.sha256()
                with archive.open(entry) as handle:
                    while block := handle.read(1024**2):
                        digest.update(block)
                result[pieces[2]] = dict(bytes=entry.file_size, sha256=digest.hexdigest())
    if not result:
        raise ValueError('Expected native torch.save tensor entries')
    return result


def prepare(mode):
    evidence = ROOT/'experiments/reports'/f'2026-10-05-native-sft-{mode}-replay'
    evidence.mkdir(mode=0o700, exist_ok=False)
    parent = ROOT/'outputs'/f'native-sft-{mode}-replay-20261005-journals'
    parent.mkdir(mode=0o700, exist_ok=False)
    # Two invocations; repeated evaluation and replayed updates are real cost.
    caps = dict(train_updates=40, sampled_examples=160, selector_steps=160,
        valid_targets=2350, logical_sequence_tokens=19032,
        policy_forward_calls=2448, policy_forward_positions=149284,
        evaluation_calls=2288, evaluation_positions=142976,
        generation_calls=32, generation_position_upper_bound=133376,
        generation_tokens=2048, recovery_validation_operations=8,
        recovery_history_rows=120, recovery_tensor_elements=16000000000,
        recovery_rng_states=24)
    old_io = json.loads((ROOT/'experiments/configs/native-base-profile-20261005-io.json').read_text())
    io = dict(old_io)
    io['limits'] = dict(snapshot_inspect_operations=1, snapshot_load_operations=1,
        snapshot_save_operations=5, snapshot_hash_bytes=35*GIB,
        snapshot_tree_nodes=350000, snapshot_tensor_elements=14000000000,
        snapshot_primitive_bytes=7*1024**2, snapshot_clone_bytes=28*GIB,
        snapshot_serialization_bytes=25*GIB)
    retain(evidence/'work-caps.json', caps)
    retain(evidence/'io-caps.json', io)
    if shutil.disk_usage(ROOT/'outputs').free < 24*GIB:
        raise RuntimeError('24GiB output planning allowance unavailable')
    command = ['/home/dongxi/dgx-spark-dongxi/.venv/bin/python',
        str(ROOT/'scripts/run_chapter09_spark_sft.py'), '--model', MODEL,
        '--revision', REVISION, '--tokenizer-revision', REVISION,
        '--template', str(ROOT/'experiments/data/instruction_interface_v1.jinja'),
        '--train', str(ROOT/'outputs/course-sft-interface-v1/train.jsonl'),
        '--dev', str(ROOT/'outputs/course-sft-interface-v1/dev.jsonl'),
        '--mode', mode, '--rank', '8', '--updates', '20', '--microbatch', '1',
        '--accumulation', '4', '--max-length', '256', '--learning-rate', '2e-05',
        '--runtime-seconds', '900', '--reserve-gib', '25.0', '--seed', '1212',
        '--checkpoint-every', '10', '--snapshot-max-bytes', str(5*GIB),
        '--work-limits', str(evidence/'work-caps.json'),
        '--work-journal-max-bytes', str(4*1024**2),
        '--work-journal', str(parent/'work.jsonl'),
        '--snapshot-io-limits', str(evidence/'io-caps.json'),
        '--snapshot-io-ledger', str(parent/'io.jsonl'),
        '--environment-lock', '/home/dongxi/dgx-spark-dongxi/uv.lock']
    inputs = {key: _digest(command[command.index('--'+key)+1], executable=key=='interpreter')
        for key in ('train','dev','template','work-limits','snapshot-io-limits','environment-lock')}
    inputs['interpreter'] = _digest(command[0], executable=True)
    record = dict(schema='dongxi-fixed-pretrained-sft-replay-v1', mode=mode,
        utc=datetime.now(timezone.utc).isoformat(), source_bindings=bindings(),
        input_bindings=inputs, base_argv=command, execution_requested=False,
        science='original20-update SFT; checkpoint10 replay in a fresh owned child',
        limits=dict(invocations=2, external_seconds_each=900, reserve_bytes=25*GIB),
        scope='recovery acceptance, not400-update pilot, merged LoRA or publication evaluation')
    retain(evidence/'preparation.json', record)
    return evidence, command, record


def run(mode, declaration):
    evidence, command, prepared = prepare(mode)
    prepared_sources = prepared['source_bindings']
    results = []
    original = ROOT/'outputs'/f'native-sft-{mode}-replay-20261005-original'
    resumed = ROOT/'outputs'/f'native-sft-{mode}-replay-20261005-resumed'
    try:
        for index, output in enumerate((original, resumed)):
            if bindings() != prepared_sources:
                raise ValueError('Executable source changed before launch')
            for row in prepared['input_bindings'].values():
                if _digest(row['path'], executable=row['path']==command[0]) != row:
                    raise ValueError('Input changed before launch')
            argv = command + ['--output', str(output)]
            if index:
                checkpoint = original/'checkpoint-000010.pt'
                header = json.loads(Path(str(checkpoint)+'.commit.json').read_text())
                # Retain expected metadata outside the producer before reading
                # payload bytes. The same physical journals retain later cost.
                for name, source in (
                    ('retained-contract.json', original/'recovery-contract.json'),
                    ('retained-receipt.json', Path(str(checkpoint)+'.work.json'))):
                    with (evidence/name).open('xb') as handle:
                        handle.write(source.read_bytes())
                argv += ['--resume', str(checkpoint), '--resume-contract',
                    str(evidence/'retained-contract.json'), '--resume-io-receipt',
                    str(evidence/'retained-receipt.json'), '--resume-sha256',
                    header['payload_sha256'], '--resume-bytes', str(header['payload_bytes'])]
            retain(evidence/f'launch-{index+1}.json', dict(argv=argv,
                operator_declaration=declaration, source_bindings=prepared_sources,
                input_bindings=prepared['input_bindings']))
            result = _supervise(argv, evidence/f'supervision-{index+1}', native=True,
                seconds=900, probe=_native_probe, operator_declaration=declaration)
            # Retain returned ACK/cleanup fields omitted at write-time by logger.
            retain(evidence/f'returned-supervision-{index+1}.json', result)
            results.append({k:result[k] for k in ('status','actual_exit_code',
                'child_seconds','minimum_sampled_available_bytes','final_record_retained')})
            if result['status'] != 'completed' or result['actual_exit_code'] != 0:
                raise RuntimeError('Actual owned invocation failed; evidence retained')
        original_rows = [json.loads(x) for x in (original/'metrics.jsonl').read_text().splitlines()]
        replay_rows = [json.loads(x) for x in (resumed/'metrics.jsonl').read_text().splitlines()]
        fields = ('update','answer_nll','targets','gradient_norm','cursor',
                  'cumulative_supervised_targets','processed_positions','cumulative_processed_positions')
        project = lambda rows: [{k:r[k] for k in fields} for r in rows]
        expected, actual = project(original_rows[10:]), project(replay_rows)
        tensor_a = tensor_bytes(original/'checkpoint-000020.pt')
        tensor_b = tensor_bytes(resumed/'checkpoint-000020.pt')
        original_result = json.loads((original/'result.json').read_text())
        resumed_result = json.loads((resumed/'result.json').read_text())
        checks = dict(original20=len(original_rows)==20, resumed10=len(replay_rows)==10,
            exact_numerical_tail=expected==actual,
            exact_tensor_entry_bytes=tensor_a==tensor_b,
            identical_final_samples=original_result['samples']==resumed_result['samples'])
        # Timing is not scientific sample content.
        sample_fields = ('id','prompt_ids','generated_ids','text','reference',
                         'exact_match','stopped_on_end','stop_reason','truncated','error')
        checks['identical_final_samples'] = (
            [{k:r[k] for k in sample_fields} for r in original_result['samples']] ==
            [{k:r[k] for k in sample_fields} for r in resumed_result['samples']])
        summary = dict(status='passed' if all(checks.values()) else 'failed',
            checks=checks, invocations=results, numerical_expected=expected,
            numerical_actual=actual, tensor_entries_original=tensor_a,
            tensor_entries_resumed=tensor_b,
            original_result=original_result, resumed_result=resumed_result,
            scope='actual pinned pretrained BF16/CUDA completed-update replay; tensor bytes and numerical tail, not arbitrary pickle metadata/Python RNG proof, cross-machine transfer, merged LoRA, publication capability or pilot')
        for label, directory in (('original',original),('resumed',resumed)):
            for name in ('result.json','metrics.jsonl','config.json','recovery-contract.json',
                         'baseline-generation-raw.jsonl','final-generation-raw.jsonl',
                         'baseline-nll-raw.jsonl','final-nll-raw.jsonl'):
                with (evidence/f'{label}-{name}').open('xb') as handle:
                    handle.write((directory/name).read_bytes())
        retain(evidence/'acceptance.json', summary)
        print(json.dumps(dict(status=summary['status'], checks=checks, evidence=str(evidence))))
        return 0 if summary['status']=='passed' else 1
    except BaseException as error:
        retain(evidence/'failure.json', dict(status='failed', invocations=results,
            type=type(error).__name__, message=str(error),
            scope='actual attempted work remains retained; no automatic retry or cap change'))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('full','lora'), required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--operator-declaration')
    args = parser.parse_args()
    if args.execute:
        if not args.operator_declaration or len(args.operator_declaration)<12:
            parser.error('Explicit recorded goal-authorized experiment scope required')
        return run(args.mode, args.operator_declaration)
    evidence, _, _ = prepare(args.mode)
    print(json.dumps(dict(status='prepared-not-executed', evidence=str(evidence))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
