#!/usr/bin/env python3
"""Fixed local Spark story profile/recovery and two gated learning arms.

Profile/recovery children have independent 600-second watchdogs; the fixed
pilots have 14400-second watchdogs. All use the sampled 25 GiB reserve.
Logical work is retained; byte I/O, disk quota and hostile-tree
containment are not supplied by this adapter. Default preparation launches none.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dongxi_llms.native_profile_supervisor import _digest, _native_probe, _supervise
from dongxi_llms.run_identity import canonical_hash
from dongxi_llms.story_work_budget import (
    story_work_contract, read_story_receipt, checked_payload)
from dongxi_llms.work_budget import WorkLedger
from dongxi_llms.snapshot_io_budget import read_bounded_json

PYTHON = '/home/dongxi/dgx-spark-dongxi/.venv/bin/python'
LOCK = '/home/dongxi/dgx-spark-dongxi/uv.lock'
DATA = ROOT / 'data/cache/day09-full-v2'
PRECONDITION_STAGES = ('story-profile', 'story-control-recovery', 'story-half-lr-recovery')
PILOTS = ('story-control-pilot', 'story-half-lr-pilot')
STAGES = PRECONDITION_STAGES + PILOTS
GIB = 1024**3
DATA_HASHES = {
    'manifest.json': '6ca5b3de3bf5b90028b999e0aff76e86d2e691df44ee4c32732934f5e95efc1c',
    'tokenizer.json': '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6',
    'train.bin': '261606c23defb1a40b324f92f8d9d3702c268c59b063727f8cb41b504ebd3ec3',
    'valid.bin': 'f9987defdc7088a2e69b039c3acc5cfccaa85294066f1a912125f380c80148f6',
    'train.windows.npy': 'd0a2a8dfae16ba5a6da9f3cae06517bdcc9bab591a5b0f7684622e51dbb749ea',
    'valid.windows.npy': 'cc9555350510359ca6bf14ad3dc3e8a6c905a7cb35c2c82d8472d9481cc63129',
}
SOURCES = ('scripts/train_stories.py', 'scripts/run_deterministic_story_child.py', 'scripts/run_native_story_stages.py',
    'src/dongxi_llms/stories_training.py', 'src/dongxi_llms/stories_data.py',
    'src/dongxi_llms/decoder_lab.py', 'src/dongxi_llms/pretraining_lab.py',
    'src/dongxi_llms/story_work_budget.py', 'src/dongxi_llms/work_budget.py',
    'src/dongxi_llms/run_identity.py', 'src/dongxi_llms/snapshot_io_budget.py',
    'src/dongxi_llms/native_profile_supervisor.py', 'src/dongxi_llms/campaign_supervisor.py',
    'src/dongxi_llms/staged_campaign.py',
    'experiments/specs/2026-10-04-staged-spark-campaign.md')
NUMERICAL_FIELDS = ('update', 'loss', 'lr', 'gradient_norm', 'valid_targets',
    'cumulative_targets', 'first_q_max_parameter_change', 'processed_positions',
    'cumulative_processed_positions', 'processed_positions_accounting')


def retain(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())


def locations(stage, run_id):
    if stage not in STAGES or type(run_id) is not str or re.fullmatch(r'[a-z0-9][a-z0-9-]{0,47}', run_id) is None:
        raise ValueError('Fixed stage and a bounded plain run ID required')
    name = 'native-' + stage + '-' + run_id
    return ROOT / 'experiments/reports' / name, ROOT / 'outputs' / name


def identity(path, maximum=2*GIB):
    path = Path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
            raise ValueError('Bounded regular local input required')
        sha = hashlib.sha256()
        while block := os.read(fd, 1024**2): sha.update(block)
        after = os.fstat(fd)
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError('Local input changed during hashing')
        return dict(path=str(path), bytes=before.st_size, sha256=sha.hexdigest())
    finally:
        os.close(fd)


def bindings():
    data = {name: identity(DATA / name) for name in DATA_HASHES}
    if any(data[name]['sha256'] != expected for name, expected in DATA_HASHES.items()):
        raise ValueError('The original pinned full TinyStories cache changed')
    return dict(sources={name: _digest(str(ROOT/name)) for name in SOURCES},
        data=data, interpreter=_digest(PYTHON, executable=True), environment_lock=_digest(LOCK))


def limits(stage, prefix_lengths):
    """Conservative entire-operation reservations, including later replay work."""
    if stage not in STAGES or len(prefix_lengths) != 3 or any(type(n) is not int or not 1 <= n <= 1008 for n in prefix_lengths):
        raise ValueError('Fixed native three-prompt tokenization required')
    profile, pilot = stage == 'story-profile', stage in PILOTS
    updates, panels = (14000, 36) if pilot else ((40, 2) if profile else (5, 7))
    valid_windows, sample_tokens = (512, 256) if pilot else (2, 16)
    calls = panels*6*sample_tokens
    return story_work_contract(dict(model_initializations=1 if profile or pilot else 2,
        train_updates=updates, training_windows=updates*16,
        training_valid_targets=50_000_000 if pilot else updates*16*1024,
        policy_forward_calls=updates+panels*((valid_windows+15)//16),
        policy_forward_positions=updates*16*1024+panels*valid_windows*1024,
        backward_calls=updates, optimizer_calls=updates,
        evaluation_panels=panels, evaluation_windows=panels*valid_windows,
        evaluation_valid_targets=panels*valid_windows*1024,
        generation_sequences=panels*6, generation_calls=calls,
        generation_positions=panels*2*sum(sample_tokens*n+sample_tokens*(sample_tokens-1)//2 for n in prefix_lengths),
        generation_tokens=calls, multinomial_draws=calls//2,
        activation_panels=panels, activation_positions=panels*1024,
        activation_block_applications=panels*12,
        save_operations=36 if pilot else (2 if profile else 6),
        restore_operations=0 if profile or pilot else 1), (64 if pilot else 8)*1024**2)


def command(stage, evidence, output, label):
    profile, pilot = stage == 'story-profile', stage in PILOTS
    if label not in (('pilot',) if pilot else (('profile',) if profile else ('clean', 'source', 'resumed'))):
        raise ValueError('No arbitrary child command or label')
    rate = '0.00015' if stage.startswith('story-half-lr-') else '0.0003'
    floor = '0.000015' if stage.startswith('story-half-lr-') else '0.00003'
    journal = output / ('pilot-work.jsonl' if pilot else ('profile-work.jsonl' if profile else ('clean-work.jsonl' if label == 'clean' else 'retained-work.jsonl')))
    argv = [PYTHON, str(ROOT/'scripts/run_deterministic_story_child.py'), 'train', '--data', str(DATA),
        '--output', str(output/label), '--device', 'cuda', '--total', '14000' if pilot else ('40' if profile else '3'),
        '--warmup', '200' if pilot else '1', '--microbatch', '16', '--accumulation', '1',
        '--valid-windows', '512' if pilot else '2', '--sample-tokens', '256' if pilot else '16',
        '--checkpoint-every', '400' if pilot else ('40' if profile else '1'),
        '--max-seconds', '14400' if pilot else '600', '--peak-lr', rate, '--floor-lr', floor,
        '--valid-target-budget', str(50_000_000 if pilot else (40 if profile else 3)*16*1024),
        '--work-limits', str(evidence/'work-caps.json'), '--work-journal', str(journal)]
    if pilot:
        # A fixed first tranche at the already declared comparison checkpoint.
        # Keep the original14k horizon/200 warmup: this is not a400-step schedule.
        argv += ['--stop-after','400']
    if label == 'resumed':
        argv += ['--resume', str(output/'source/update-000001.pt'),
                 '--resume-work-receipt', str(evidence/'retained-update-000001-receipt.json')]
    return argv


def prerequisite_bindings(stage, prerequisites, current):
    if stage not in PILOTS:
        if prerequisites: raise ValueError('Only fixed pilots consume prerequisite run IDs')
        return {}
    if type(prerequisites) is not dict or set(prerequisites) != set(PRECONDITION_STAGES):
        raise ValueError('Actual current profile and both arm recovery run IDs required')
    receipts = {}
    for parent, run_id in prerequisites.items():
        evidence, output = locations(parent,run_id)
        acceptance,_ = read_bounded_json(evidence/'acceptance.json')
        if (acceptance.get('status')!='passed' or acceptance.get('stage')!=parent or
                acceptance.get('bindings_after')!=current or acceptance.get('distinct_native_children') is not True):
            raise ValueError('Actual prerequisite is missing, failed or has changed source/input identity')
        expected = 1 if parent=='story-profile' else 3
        invocations = acceptance.get('invocations',[])
        if (len(invocations)!=expected or any(row.get('status')!='completed' or row.get('actual_exit_code')!=0 or
                type(row.get('minimum_sampled_available_bytes')) is not int or row['minimum_sampled_available_bytes']<25*GIB
                for row in invocations)):
            raise ValueError('Prerequisite actual exits/resources incomplete')
        files = ['acceptance.json','preparation.json']
        label = 'profile' if parent=='story-profile' else 'resumed'
        completion,_ = read_bounded_json(output/label/'completion.json')
        if completion.get('completed_updates')!=(40 if parent=='story-profile' else 3) or completion.get('schedule_complete') is not True:
            raise ValueError('Prerequisite fixed schedule incomplete')
        if parent!='story-profile':
            recovered,_ = read_bounded_json(evidence/'recovery-acceptance.json')
            if recovered.get('status')!='passed' or not recovered.get('checks') or any(v is not True for v in recovered['checks'].values()):
                raise ValueError('Actual numerical/persistent recovery checks incomplete')
            files += ['recovery-acceptance.json','retained-update-000001-receipt.json']
        receipts[parent] = dict(run_id=run_id,
            evidence={name:identity(evidence/name) for name in files},
            completion=identity(output/label/'completion.json'))
    return receipts


def prepare(stage, run_id, prerequisites=None):
    evidence, output = locations(stage, run_id)
    if evidence.exists() or evidence.is_symlink() or output.exists() or output.is_symlink():
        raise ValueError('Run evidence and output must both be new')
    bound = bindings()
    prerequisites_bound = prerequisite_bindings(stage,prerequisites,bound)
    from tokenizers import Tokenizer
    # This is tokenizer-only sizing; no Torch/model/CUDA import occurs here.
    tok = Tokenizer.from_file(str(DATA/'tokenizer.json'))
    prompts = ('Once upon a time, a little rabbit lived near a forest.',
        'Lily put her red ball in a box. Then she went outside.',
        'Tom wanted to fly his kite, but there was no wind.')
    prefix_lengths = [1+len(tok.encode(p, add_special_tokens=False).ids) for p in prompts]
    caps = limits(stage, prefix_lengths)
    storage = (40 if stage in PILOTS else 12)*GIB
    if shutil.disk_usage(ROOT/'outputs').free < storage:
        raise RuntimeError('Declared stage storage planning reserve unavailable')
    evidence.mkdir(mode=0o700); output.mkdir(mode=0o700)
    retain(evidence/'work-caps.json', caps)
    labels = ('pilot',) if stage in PILOTS else (('profile',) if stage == 'story-profile' else ('clean', 'source', 'resumed'))
    record = dict(schema='dongxi-fixed-native-story-stages-v1', stage=stage, run_id=run_id,
        utc=datetime.now(timezone.utc).isoformat(), bindings=bound, caps=caps,
        caps_identity=identity(evidence/'work-caps.json'), prefix_lengths=prefix_lengths,
        commands={label: command(stage, evidence, output, label) for label in labels},
        prerequisite_run_ids=prerequisites, prerequisite_bindings=prerequisites_bound,
        limits=dict(invocations=len(labels), external_seconds_each=14400 if stage in PILOTS else 600,
            reserve_bytes=25*GIB, disk_planning_bytes=storage), execution_requested=False,
        execution_tranche=(dict(requested_stop=400,total_horizon=14000,
            full_schedule_completion=False,checkpoint_selection='Original predetermined400 point, declared before either fresh pilot/publication result')
            if stage in PILOTS else None),
        scope=('Fixed fresh first400 story tranche within original14000 schedule; later predetermined checkpoints remain unrun; no full14k completion'
            if stage in PILOTS else 'Fixed fresh story profile or completed1-to3 replay; no publication panel, acquisition, I/O quota or physical containment claim'))
    record['preparation_sha256'] = canonical_hash(record)
    retain(evidence/'preparation.json', record)
    return evidence, output, record


def assert_current(prepared, evidence):
    if bindings() != prepared['bindings'] or identity(evidence/'work-caps.json') != prepared['caps_identity']:
        raise ValueError('Prepared source/data/environment/cap bytes changed')
    if 'prerequisite_bindings' in prepared and prerequisite_bindings(prepared['stage'],prepared['prerequisite_run_ids'],prepared['bindings']) != prepared['prerequisite_bindings']:
        raise ValueError('Retained actual prerequisite receipt changed')


def freeze_receipt(output, evidence):
    source = output/'source/update-000001.pt.work.json'
    value = read_story_receipt(source)
    if value['completed_updates'] != 1 or value['payload_bytes'] > 2*GIB:
        raise ValueError('Fixed completed1 receipt and bounded payload required')
    target = evidence/'retained-update-000001-receipt.json'
    retain(target, value)
    return identity(target)


def exact(first, second):
    """Compare tensors/primitives strictly without altering the legacy comparator."""
    import torch
    if type(first) is not type(second): return False
    if isinstance(first, torch.Tensor):
        return first.shape == second.shape and first.dtype == second.dtype and torch.equal(first, second)
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(exact(first[k], second[k]) for k in first)
    if isinstance(first, (tuple, list)):
        return len(first) == len(second) and all(exact(a,b) for a,b in zip(first,second))
    return first == second


def completion_matches(stage, value):
    pilot=stage in PILOTS
    expected=400 if pilot else (40 if stage=='story-profile' else 3)
    return (value.get('completed_updates')==expected
        and value.get('requested_stop_reached') is True
        and value.get('schedule_complete') is (not pilot))


def checkpoint(path, journal, caps, completion, retained=None):
    receipt = read_story_receipt(Path(str(path)+'.work.json'))
    if receipt['payload_bytes'] > 2*GIB: raise ValueError('Checkpoint exceeds fixed inspection envelope')
    with WorkLedger.open(journal, limits=caps['limits'], contract_sha256=receipt['contract_sha256'],
            max_bytes=caps['max_journal_bytes'], invocation_id='story-independent-verifier',
            expected_snapshot=receipt['work_prefix']) as ledger:
        if retained is not None: ledger.validate_snapshot(read_story_receipt(retained)['work_prefix'])
        ledger.validate_snapshot(completion['work_ledger'])
        current = ledger.snapshot()
        import torch
        with checked_payload(path, receipt) as handle:
            state = torch.load(handle, map_location='cpu', weights_only=True)
        if state['story_work_prefix'] != receipt['work_prefix'] or canonical_hash(state['contract']) != receipt['contract_sha256']:
            raise ValueError('Payload scientific/work prefix differs from independent receipt')
        return state, dict(receipt=receipt, current=current,
            later_reserved={k:current['reserved'][k]-receipt['work_prefix']['reserved'][k] for k in caps['limits']})


def inspect_recovery(stage, run_id):
    import torch
    torch.set_num_threads(1)
    evidence, output = locations(stage, run_id)
    prepared = json.loads((evidence/'preparation.json').read_text())
    caps = prepared['caps']
    completions = {label:json.loads((output/label/'completion.json').read_text()) for label in ('clean','source','resumed')}
    states, prefixes = {}, {}
    for label in completions:
        states[label], prefixes[label] = checkpoint(output/label/'update-000003.pt',
            output/('clean-work.jsonl' if label=='clean' else 'retained-work.jsonl'), caps, completions[label],
            None if label=='clean' else evidence/'retained-update-000001-receipt.json')
    rows = {label:[json.loads(line) for line in (output/label/'metrics.jsonl').read_text().splitlines()] for label in completions}
    projected = lambda values: [{key:row[key] for key in NUMERICAL_FIELDS} for row in values]
    numerical_keys = tuple(key for key in states['clean'] if key != 'story_work_prefix')
    checks = dict(all_completed3=all(c['completed_updates']==3 and c['requested_stop_reached'] and c['schedule_complete'] for c in completions.values()),
        clean_source_exact=exact({k:states['clean'][k] for k in numerical_keys},{k:states['source'][k] for k in numerical_keys}),
        clean_resumed_exact=exact({k:states['clean'][k] for k in numerical_keys},{k:states['resumed'][k] for k in numerical_keys}),
        resumed_updates2_and3=[r['update'] for r in rows['resumed']]==[2,3],
        exact_metric_tail=projected(rows['clean'][1:])==projected(rows['resumed']),
        preserved_later_cost=prefixes['resumed']['current']['reserved']['train_updates']==5,
        same_retained_journal=prefixes['source']['current']['file_identity']==prefixes['resumed']['current']['file_identity'],
        native_restore_charged=prefixes['resumed']['current']['completed']['restore_operations']==1)
    result = dict(status='passed' if all(checks.values()) else 'failed', checks=checks,
        numerical_keys=list(numerical_keys), work_prefixes=prefixes, completions=completions,
        actual_cpu_pid=os.getpid(), scope='Same-host completed-state numerical replay and separately validated persistent costs; no quality or pilot claim')
    retain(evidence/'recovery-acceptance.json', result)
    print(json.dumps(dict(status=result['status'], checks=checks)))
    return 0 if all(checks.values()) else 1


def run(stage, run_id, declaration, prerequisites=None):
    if type(declaration) is not str or not 12 <= len(declaration) <= 4096:
        raise ValueError('Explicit retained operator scope required')
    evidence, output, prepared = prepare(stage, run_id, prerequisites)
    results = []
    retained_identity = None
    try:
        for label, argv in prepared['commands'].items():
            assert_current(prepared, evidence)
            if label == 'resumed':
                retained_identity = freeze_receipt(output, evidence)
            retain(evidence/('launch-'+label+'.json'), dict(argv=argv, operator_declaration=declaration,
                preparation_sha256=prepared['preparation_sha256'], retained_receipt_identity=retained_identity))
            result = _supervise(argv, evidence/('supervision-'+label), native=True,
                seconds=prepared['limits']['external_seconds_each'], probe=_native_probe,
                operator_declaration=declaration, native_stage='story-pilot' if stage in PILOTS else 'profile')
            retain(evidence/('returned-supervision-'+label+'.json'), result)
            results.append(dict(label=label, **result))
            assert_current(prepared, evidence)
            if retained_identity is not None and identity(evidence/'retained-update-000001-receipt.json') != retained_identity:
                raise ValueError('Retained independent receipt changed')
            if result['status'] != 'completed' or result['actual_exit_code'] != 0:
                raise RuntimeError('Actual owned invocation failed; evidence retained')
            completion = json.loads((output/label/'completion.json').read_text())
            if not completion_matches(stage,completion):
                raise RuntimeError('Successful exit did not complete the fixed requested boundary with honest schedule status')
        if stage in ('story-control-recovery','story-half-lr-recovery'):
            verify_argv = [PYTHON, str(Path(__file__).resolve()), '--stage', stage, '--run-id', run_id, '--verify-child']
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            child = subprocess.run(verify_argv, env=env, capture_output=True, text=True, timeout=120)
            retain(evidence/'cpu-comparison-process.json', dict(argv=verify_argv, exit=child.returncode,
                stdout=child.stdout, stderr=child.stderr, CUDA_VISIBLE_DEVICES=''))
            if child.returncode: raise RuntimeError('Fresh CPU numerical/prefix comparison failed')
        assert_current(prepared, evidence)
        summary = dict(status='passed', stage=stage, preparation_sha256=prepared['preparation_sha256'],
            completion_level='first400-tranche-not-full14000' if stage in PILOTS else 'fixed-profile-or-recovery',
            full_schedule_complete=stage not in PILOTS,
            distinct_native_children=len({r['child_pid'] for r in results})==len(results),
            invocations=[{k:r[k] for k in ('label','status','actual_exit_code','child_pid','child_seconds','minimum_sampled_available_bytes')} for r in results],
            bindings_after=bindings(), scope=prepared['scope'])
        if not summary['distinct_native_children']: raise ValueError('Fresh native process identities were not distinct')
        retain(evidence/'acceptance.json', summary)
        print(json.dumps(dict(status='passed', evidence=str(evidence))))
        return 0
    except BaseException as error:
        retain(evidence/'failure.json', dict(status='failed', type=type(error).__name__, message=str(error),
            invocations=[r['label'] for r in results], scope='Failed attempts retained; no automatic retry or changed cap'))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=STAGES, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--operator-declaration')
    parser.add_argument('--profile-run-id')
    parser.add_argument('--control-recovery-run-id')
    parser.add_argument('--half-lr-recovery-run-id')
    parser.add_argument('--verify-child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    parents = dict(zip(PRECONDITION_STAGES,(args.profile_run_id,args.control_recovery_run_id,args.half_lr_recovery_run_id)))
    prerequisites = parents if any(parents.values()) else None
    locations(args.stage, args.run_id)
    if args.verify_child:
        if args.execute or args.operator_declaration or prerequisites or args.stage not in ('story-control-recovery','story-half-lr-recovery'):
            parser.error('Only internal recovery inspection is permitted')
        return inspect_recovery(args.stage,args.run_id)
    if args.execute:
        return run(args.stage,args.run_id,args.operator_declaration,prerequisites)
    if args.operator_declaration: parser.error('Execution text requires --execute')
    evidence, _, prepared = prepare(args.stage,args.run_id,prerequisites)
    print(json.dumps(dict(status='prepared-not-executed', evidence=str(evidence), preparation_sha256=prepared['preparation_sha256'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
