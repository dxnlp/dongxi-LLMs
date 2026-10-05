#!/usr/bin/env python3
"""Closed cached-instruct RLVR recovery and G4/G8 pilots; default prepares only.

Recovery executes a two-update source, then completed-1 and pending-1 fresh
resumes on the same permanently retained work/I/O journals. Pilot starts fresh
for sixteen updates. No arbitrary argv, acquisition, retry, new objective or
twenty-item publication evaluation is introduced by this adapter.
"""
import argparse
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import pickle
import re
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from dongxi_llms import qwen_rlvr_lab as rlvr
from dongxi_llms.native_profile_supervisor import _digest, _native_probe, _new_target, _supervise
from dongxi_llms.reasoning_generation import load_local_tokenizer
from dongxi_llms.run_identity import artifact_hashes, cached_snapshot, canonical_hash, tokenizer_interface
from dongxi_llms.snapshot_io_budget import io_budget_contract, validate_work_receipt, read_bounded_json
from dongxi_llms.snapshot_io_schedule import operation_cost

GIB = 1024**3
MODEL = 'Qwen/Qwen3-0.6B'
REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'
INTERPRETER = '/home/dongxi/dgx-spark-dongxi/.venv/bin/python'
ENVIRONMENT_LOCK = '/home/dongxi/dgx-spark-dongxi/uv.lock'
SEED = 2323
JOURNAL_BOUND = 4*1024**2
ENVELOPE = dict(max_payload_bytes=8*GIB, max_tensor_bytes=8*GIB,
    max_tensor_elements=4_000_000_000, max_tree_nodes=100_000, max_primitive_bytes=1024**2)
TRAIN_PAIRS = ((2,3), (4,5), (3,6), (1,7))
HELDOUT_PAIRS = ((2,6), (3,4), (5,5), (7,2))
SOURCES = ('src/dongxi_llms/qwen_rlvr_lab.py', 'src/dongxi_llms/grpo_lab.py',
    'src/dongxi_llms/run_identity.py', 'src/dongxi_llms/training_snapshot.py',
    'src/dongxi_llms/batched_cache_lab.py', 'src/dongxi_llms/work_budget.py',
    'src/dongxi_llms/snapshot_io_budget.py', 'src/dongxi_llms/snapshot_io_schedule.py',
    'src/dongxi_llms/artifact_budget.py', 'src/dongxi_llms/native_profile_supervisor.py',
    'src/dongxi_llms/campaign_supervisor.py', 'scripts/run_native_rlvr_stages.py',
    'tests/test_native_rlvr_stages.py', 'scripts/run_native_reasoning_baselines.py')
BASELINE_CHECKS = ('actual_exit0', 'all_five_cells_completed',
    'all_original_responses', 'no_unreadable_jsonl')
RECOVERY_CHECKS = ('source_completed_horizon', 'completed_completed_horizon',
    'pending_completed_horizon', 'original_prompt_and_source_contract',
    'exact_final_policy_reference_optimizer_rng_loop_history', 'exact_update2_metrics',
    'pending_does_not_resample', 'cumulative_work_ledger_same_physical_journal',
    'cumulative_work_ledger_retains_later_spending',
    'cumulative_snapshot_io_ledger_same_physical_journal',
    'cumulative_snapshot_io_ledger_retains_later_spending', 'no_open_or_failed_journal_tickets')


def retain(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())


def paths(stage, group, run_id):
    if (stage not in ('recovery', 'pilot') or type(group) is not int or group not in (4,8)
            or type(run_id) is not str or re.fullmatch(r'run-[0-9]{2}', run_id) is None):
        raise ValueError('Only recovery/pilot, G4/G8 and run-NN are supported')
    stem = f'native-rlvr-g{group}-{stage}-20261005-{run_id}'
    values = dict(evidence=ROOT/'experiments/reports'/stem, journals=ROOT/'outputs'/(stem+'-journals'))
    if stage == 'recovery':
        values.update(source=ROOT/'outputs'/(stem+'-source'),
            completed=ROOT/'outputs'/(stem+'-completed-resume'), pending=ROOT/'outputs'/(stem+'-pending-resume'))
    else:
        values['pilot'] = ROOT/'outputs'/stem
    return values


def source_bindings():
    return {name: _digest(str(ROOT/name)) for name in SOURCES}


def model_binding():
    snapshot = cached_snapshot(MODEL, REVISION)
    files = artifact_hashes(snapshot)
    if 'config.json' not in files or not any(name.endswith('.safetensors') for name in files):
        raise ValueError('Complete existing pinned-cache instruct model required')
    return dict(model=MODEL, revision=REVISION, path=str(snapshot), files=files,
        revision_evidence='Declared immutable local cache ID and actual streamed bytes; not remotely authenticated upstream correspondence')


def geometry(model):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    tokenizer = load_local_tokenizer(model['path'])
    if not isinstance(tokenizer.chat_template, str) or 'enable_thinking' not in tokenizer.chat_template:
        raise ValueError('Actual native instruct template must expose thinking-off')
    encode, stops, prompt = rlvr.prompt_contract(tokenizer, 'chat')
    def rows(pairs, role):
        return [dict(source_id=f'{role}-{index}', prompt=f'Return only the integer answer. {a} + {b} =',
            prompt_ids=encode(f'Return only the integer answer. {a} + {b} =').flatten().tolist(), expected=a+b)
            for index, (a,b) in enumerate(pairs)]
    train, evaluation = rows(TRAIN_PAIRS, 'train'), rows(HELDOUT_PAIRS, 'heldout')
    config = json.loads((Path(model['path'])/'config.json').read_text())
    context = config.get('max_position_embeddings')
    if type(context) is not int or any(len(row['prompt_ids']) + 64 > context for row in train+evaluation):
        raise ValueError('Declared original arithmetic prompts plus cap64 must fit actual context')
    if {tuple(row['prompt_ids']) for row in train} & {tuple(row['prompt_ids']) for row in evaluation}:
        raise ValueError('Actual encoded original train/heldout prompts collide')
    interface = tokenizer_interface(tokenizer, template=tokenizer.chat_template,
        stop_ids=stops, tokenizer_id=model['path'], tokenizer_revision=REVISION)
    return dict(train=train, evaluation=evaluation, prompt_contract=prompt,
        checkpoint_interface=interface, context_length=context,
        maximum_train_prompt=max(len(row['prompt_ids']) for row in train),
        maximum_evaluation_prompt=max(len(row['prompt_ids']) for row in evaluation),
        scope='Actual tokenizer-only geometry of unchanged four train/four heldout integer prompts')


def allowances(stage, group, observed):
    paths(stage, group, 'run-00')
    cap = 16 if stage == 'recovery' else 64
    collections, applications, evaluation_rows = (3,4,24) if stage == 'recovery' else (16,16,8)
    saves, reads = (10,2) if stage == 'recovery' else (33,0)
    validation_ops, validation_calls, validated_pools = (16,9,22) if stage == 'recovery' else (49,32,288)
    prompt = observed['maximum_train_prompt']; evaluation_prompt = observed['maximum_evaluation_prompt']
    collection_positions = group*(cap*prompt + cap*(cap-1)//2)
    score_positions = group*(prompt+cap-1)
    evaluation_positions = evaluation_rows*(cap*evaluation_prompt+cap*(cap-1)//2)
    work = dict(train_updates=applications, collections=collections, source_examples=collections,
        generated_slots=collections*group*cap, valid_response_tokens=collections*group*cap,
        multinomial_draws=collections*group*cap, response_targets=applications*group*cap,
        generation_forward_calls=collections*cap+evaluation_rows*cap,
        generation_forward_positions=collections*collection_positions+evaluation_positions,
        policy_forward_calls=collections*(cap+1)+applications+validation_calls+evaluation_rows*cap,
        policy_forward_positions=collections*(collection_positions+score_positions)
            +(applications+validation_calls)*score_positions+evaluation_positions,
        reference_forward_calls=applications, reference_forward_positions=applications*score_positions,
        old_policy_forward_calls=collections, old_policy_forward_positions=collections*score_positions,
        evaluation_examples=evaluation_rows, evaluation_calls=evaluation_rows*cap,
        evaluation_positions=evaluation_positions, evaluation_tokens=evaluation_rows*cap,
        recovery_validation_operations=validation_ops, recovery_validation_calls=validation_calls,
        recovery_validation_positions=validation_calls*score_positions,
        recovery_multinomial_draws=validated_pools*group*cap)
    rlvr.work_budget_contract(work, JOURNAL_BOUND)
    io_limits = {key: 0 for key in operation_cost(ENVELOPE, 'save')}
    for operation, count in (('save', saves), ('inspect', reads), ('load', reads)):
        cost = operation_cost(ENVELOPE, operation,
            **({} if operation == 'save' else {'payload_bytes': ENVELOPE['max_payload_bytes']}))
        for key, value in cost.items():
            io_limits[key] += count*value
    return dict(work=work, io=io_budget_contract(io_limits, ENVELOPE, JOURNAL_BOUND),
        lifecycle=dict(collections=collections, physical_applications=applications,
            snapshot_saves=saves, snapshot_inspects=reads, snapshot_loads=reads,
            validations=validation_ops, validation_forwards=validation_calls, validated_pools=validated_pools),
        scope='Permanent whole-operation upper reservations from original runner lifecycle, not measured work or retry capacity')


def baseline_adapter():
    spec = importlib.util.spec_from_file_location('rlvr_baseline_prerequisite',
        ROOT/'scripts/run_native_reasoning_baselines.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_record_bindings(record):
    if (record.get('preparation_sha256') != canonical_hash(
            {key:value for key,value in record.items() if key != 'preparation_sha256'})
            or not record.get('source_bindings') or not record.get('input_bindings')):
        raise ValueError('Prerequisite preparation must retain its complete frozen bindings')
    for group in ('source_bindings', 'input_bindings'):
        for key, binding in record[group].items():
            if _digest(binding['path'], executable=key == 'interpreter') != binding:
                raise ValueError('Accepted prerequisite source/input bytes are no longer current')


def prerequisite_bindings(stage, group, model, observed=None):
    budgets = (32,) if stage == 'recovery' else (32,128)
    observed = geometry(model) if observed is None else observed
    files = {}
    for budget in budgets:
        directory = ROOT/'experiments/reports'/f'native-reasoning-instruct-thinking-off-cap{budget}-20261005-run-01'
        acceptance_path = directory/'acceptance.json'
        acceptance = json.loads(acceptance_path.read_text())
        binding = acceptance.get('local_model_binding', {})
        if (acceptance.get('status') != 'passed' or acceptance.get('actual_exit_code') != 0
                or acceptance.get('row') != 'instruct-thinking-off' or acceptance.get('budget') != budget
                or any(acceptance.get('checks', {}).get(key) is not True for key in BASELINE_CHECKS)
                or binding.get('model') != MODEL or binding.get('revision') != REVISION
                or binding.get('path') != model['path'] or binding.get('files') != model['files']):
            raise ValueError('Actual same-byte instruct thinking-off baseline prerequisite required')
        preparation = directory/'preparation.json'
        declaration = json.loads(preparation.read_text())
        verify_record_bindings(declaration)
        baseline_adapter().verify_prepared(declaration)
        if (declaration.get('row') != 'instruct-thinking-off' or declaration.get('budget') != budget
                or declaration.get('run_id') != 'run-01' or declaration.get('local_model_binding') != binding):
            raise ValueError('Actual baseline acceptance/preparation selector or model differs')
        for cell in declaration['cells']:
            contract = json.loads(Path(cell['contract_path']).read_text())
            settings = contract['settings'];interface = settings['generation']['interface']
            if (settings['thinking_mode'] != 'disabled' or settings['generation']['input_mode'] != 'chat'
                    or interface['tokenizer'] != observed['checkpoint_interface']['tokenizer']
                    or interface['template_sha256'] != observed['checkpoint_interface']['template_sha256']):
                raise ValueError('Baseline native thinking-off tokenizer/template differs from RLVR')
        files[f'baseline-cap{budget}/acceptance'] = _digest(str(acceptance_path))
        files[f'baseline-cap{budget}/preparation'] = _digest(str(preparation))
        for key, binding in declaration['input_bindings'].items():
            files[f'baseline-cap{budget}/input/{key}'] = binding
    if stage == 'pilot':
        path = ROOT/'experiments/reports'/f'native-rlvr-g{group}-recovery-20261005-run-01'/'acceptance.json'
        acceptance = json.loads(path.read_text())
        if (acceptance.get('status') != 'passed' or acceptance.get('group') != group
                or acceptance.get('stage') != 'recovery' or acceptance.get('local_model_binding') != model
                or any(acceptance.get('checks', {}).get(key) is not True for key in RECOVERY_CHECKS)):
            raise ValueError('Actual group-matched completed and pending recovery prerequisite required')
        invocations = acceptance.get('invocations', [])
        if ([row.get('role') for row in invocations] != ['source','completed','pending']
                or any(row.get('status') != 'completed' or row.get('actual_exit_code') != 0 for row in invocations)):
            raise ValueError('Three actual completed/pending recovery exit receipts required')
        preparation = path.parent/'preparation.json'
        declaration = json.loads(preparation.read_text())
        verify_record_bindings(declaration)
        verify_prepared(declaration)
        if (declaration.get('stage') != 'recovery' or declaration.get('group') != group
                or declaration.get('run_id') != 'run-01' or declaration.get('local_model_binding') != model
                or declaration.get('source_bindings') != source_bindings()
                or declaration.get('observed_geometry') != observed):
            raise ValueError('Accepted recovery source/input/native interface differs from current pilot')
        for key, expected in dict(seed=SEED,lr=1e-6,beta=.02,reserve_bytes=25*GIB,
                prompt_mode='native-chat-thinking-off').items():
            if declaration['recipe'].get(key) != expected:
                raise ValueError('Accepted recovery fixed initialization/recipe differs from current pilot')
        files['recovery/acceptance'] = _digest(str(path))
        files['recovery/preparation'] = _digest(str(preparation))
    return files


def fixed_argv(record, role, resume=None):
    stage, group, run_id = record['stage'], record['group'], record['run_id']
    locations = paths(stage, group, run_id)
    if role not in (('source', 'completed', 'pending') if stage == 'recovery' else ('pilot',)):
        raise ValueError('Unknown closed invocation role')
    updates, cap, seconds = (2,16,600) if stage == 'recovery' else (16,64,1800)
    argv = [INTERPRETER, '-m', 'dongxi_llms.qwen_rlvr_lab', '--model-dir', record['local_model_binding']['path'],
        '--revision', REVISION, '--output', str(locations[role]), '--updates', str(updates),
        '--group-size', str(group), '--max-new-tokens', str(cap), '--seed', str(SEED), '--lr', '1e-06',
        '--device', 'cuda', '--prompt-mode', 'chat', '--max-seconds', str(seconds),
        '--environment-lock', ENVIRONMENT_LOCK, '--snapshot-max-bytes', str(ENVELOPE['max_payload_bytes']),
        '--work-limits', str(locations['evidence']/'work-caps.json'),
        '--work-journal-max-bytes', str(JOURNAL_BOUND), '--work-journal', str(locations['journals']/'work.jsonl'),
        '--snapshot-io-limits', str(locations['evidence']/'io-caps.json'),
        '--snapshot-io-ledger', str(locations['journals']/'io.jsonl')]
    if role in ('completed', 'pending'):
        if resume is None or resume['phase'] != role or resume['completed_updates'] != 1:
            raise ValueError('Independently retained completed-1 or pending-1 source receipt required')
        expected_payload = locations['source']/'snapshots'/f'{role}-000001.pt'
        if resume['path'] != str(expected_payload):
            raise ValueError('Only the declared original source snapshot can be resumed')
        argv.extend(['--resume', resume['path'], '--resume-contract', str(locations['evidence']/'retained-contract.json'),
            '--resume-sha256', resume['payload_sha256'], '--resume-bytes', str(resume['payload_bytes']),
            '--resume-io-receipt', str(locations['evidence']/f'retained-{role}-receipt.json')])
    elif resume is not None:
        raise ValueError('Fresh stage cannot adopt a resume')
    return argv


def fixed_recipe(stage):
    return dict(updates=2 if stage == 'recovery' else 16,
        output_cap=16 if stage == 'recovery' else 64, seed=SEED, lr=1e-6, beta=.02,
        external_seconds_each=600 if stage == 'recovery' else 1800,
        reserve_bytes=25*GIB, invocations=3 if stage == 'recovery' else 1,
        disk_planning_bytes=(88 if stage == 'recovery' else 272)*GIB,
        prompt_mode='native-chat-thinking-off')


def prepare(stage, group, run_id):
    locations = paths(stage, group, run_id)
    for path in locations.values(): _new_target(str(path))
    evidence = locations['evidence']; evidence.mkdir(mode=0o700)
    try:
        planned_bytes = (88 if stage == 'recovery' else 272)*GIB
        if shutil.disk_usage(ROOT/'outputs').free < planned_bytes:
            raise RuntimeError('Predeclared whole-stage output planning allowance unavailable')
        sources, model = source_bindings(), model_binding()
        observed = geometry(model)
        prerequisites = prerequisite_bindings(stage, group, model, observed)
        caps = allowances(stage, group, observed)
        retain(evidence/'work-caps.json', caps['work']); retain(evidence/'io-caps.json', caps['io'])
        retain(evidence/'tokenizer-geometry.json', observed)
        inputs = dict(interpreter=_digest(INTERPRETER, executable=True),
            environment_lock=_digest(ENVIRONMENT_LOCK), work_caps=_digest(str(evidence/'work-caps.json')),
            io_caps=_digest(str(evidence/'io-caps.json')), geometry=_digest(str(evidence/'tokenizer-geometry.json')))
        locations['journals'].mkdir(mode=0o700)
        for name in ('work.jsonl', 'io.jsonl'): _new_target(str(locations['journals']/name), private=True)
        record = dict(schema='dongxi-fixed-native-rlvr-stages-v1', stage=stage, group=group, run_id=run_id,
            utc=datetime.now(timezone.utc).isoformat(), locations={key:str(value) for key,value in locations.items()},
            local_model_binding=model, source_bindings=sources, input_bindings=inputs,
            prerequisite_bindings=prerequisites, observed_geometry=observed, allowances=caps,
            execution_requested=False, recipe=fixed_recipe(stage),
            boundaries=['Only original four training/four heldout arithmetic prompts; not the twenty-item reasoning publication panel',
                'G4/G8 pilots start fresh at the same cached instruct parent; unequal sampled token caps are retained',
                'Snapshots include policy/reference/BF16 Adam/RNG; original 8 GiB envelope is explicit',
                'Same physical journals retain later source and replay spending; no refund or automatic retry',
                'Snapshot hashing/archive comparisons and exports are outside the shared nine-dimensional I/O visits',
                'Sampled supervisor/25 GiB reserve is not an aggregate physical quota or GPU-idle authentication'])
        record['fresh_argv'] = fixed_argv(record, 'source' if stage == 'recovery' else 'pilot')
        if source_bindings() != sources or model_binding() != model or prerequisite_bindings(stage, group, model, observed) != prerequisites:
            raise ValueError('Source/model/prerequisite changed during preparation')
        record['preparation_sha256'] = canonical_hash(record)
        retain(evidence/'preparation.json', record)
        return record
    except BaseException as error:
        retain(evidence/'preparation-failure.json', dict(status='failed', type=type(error).__name__, message=str(error)))
        raise


def verify_prepared(record):
    locations = paths(record['stage'], record['group'], record['run_id'])
    if (record['locations'] != {key:str(value) for key,value in locations.items()}
            or canonical_hash({k:v for k,v in record.items() if k != 'preparation_sha256'}) != record.get('preparation_sha256')
            or record['fresh_argv'] != fixed_argv(record, 'source' if record['stage'] == 'recovery' else 'pilot')):
        raise ValueError('Closed stage command/selectors/locations changed')
    if source_bindings() != record['source_bindings'] or model_binding() != record['local_model_binding']:
        raise ValueError('Bound source/local model artifacts changed')
    if prerequisite_bindings(record['stage'], record['group'], record['local_model_binding'], record['observed_geometry']) != record['prerequisite_bindings']:
        raise ValueError('Bound actual prerequisites changed')
    for key, expected in record['input_bindings'].items():
        if _digest(expected['path'], executable=key == 'interpreter') != expected:
            raise ValueError('Bound fixed input changed: '+key)
    if allowances(record['stage'],record['group'],record['observed_geometry']) != record['allowances']:
        raise ValueError('Original whole-stage work/I/O allowances changed')
    if record['recipe'] != fixed_recipe(record['stage']):
        raise ValueError('Original fixed stage recipe differs')


def retain_resume_inputs(record):
    locations = {key:Path(value) for key,value in record['locations'].items()}
    source, evidence = locations['source'], locations['evidence']
    contract = json.loads((source/'recovery-contract.json').read_text())
    retain(evidence/'retained-contract.json', contract)
    result = {}
    for phase in ('completed', 'pending'):
        path = source/'snapshots'/f'{phase}-000001.pt'
        marker = json.loads(Path(str(path)+'.commit.json').read_text())
        receipt = validate_work_receipt(json.loads(Path(str(path)+'.work.json').read_text()))
        if (marker != receipt['snapshot'] or marker['phase'] != phase or marker['completed_updates'] != 1
                or marker['contract_sha256'] != canonical_hash(contract)):
            raise ValueError('Source marker/independent receipt/contract phase binding differs')
        retained_path = evidence/f'retained-{phase}-receipt.json'
        retain(retained_path, receipt)
        # Producer metadata is retained before independently streaming payload.
        observed = _digest(str(path), maximum=ENVELOPE['max_payload_bytes'])
        if observed['bytes'] != marker['payload_bytes'] or observed['sha256'] != marker['payload_sha256']:
            raise ValueError('Original source payload differs from independently retained expectation')
        result[phase] = dict(path=str(path), **marker,
            metadata_bindings={name:_digest(str(value)) for name,value in (
                ('contract', evidence/'retained-contract.json'), ('receipt', retained_path),
                ('marker', Path(str(path)+'.commit.json')))})
    retain(evidence/'independent-resume-bindings.json', result)
    return result


def verify_resume(binding):
    for value in binding['metadata_bindings'].values():
        if _digest(value['path']) != value: raise ValueError('Retained resume metadata changed')
    observed = _digest(binding['path'], maximum=ENVELOPE['max_payload_bytes'])
    if observed['bytes'] != binding['payload_bytes'] or observed['sha256'] != binding['payload_sha256']:
        raise ValueError('Bound source resume payload changed')


def symbolic_snapshot(path):
    """Compare native snapshot science without allocating/deserializing tensors.

The restricted pickle reader admits only ordinary ordered dictionaries and
Torch tensor/storage descriptors; storage bytes are streamed from this archive.
Physical work prefixes and invocation IDs are deliberately outside the science
projection. Unknown pickle globals fail instead of executing arbitrary code.
"""
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        pickles = [entry for entry in entries if entry.filename.endswith('/data.pkl')]
        if len(pickles) != 1 or pickles[0].file_size > 16*1024**2:
            raise ValueError('One bounded native tensor pickle metadata entry required')
        prefix = pickles[0].filename.rsplit('/',1)[0]
        by_name = {entry.filename:entry for entry in entries}
        cache = {}
        def storage(identity):
            if type(identity) is not tuple or len(identity) != 5 or identity[0] != 'storage':
                raise ValueError('Unsupported native storage identity')
            _, dtype, key, location, elements = identity
            name = prefix+'/data/'+str(key)
            if name not in by_name or by_name[name].file_size > ENVELOPE['max_payload_bytes']:
                raise ValueError('Missing/bounded native storage bytes required')
            if key not in cache:
                digest = hashlib.sha256()
                with archive.open(name) as handle:
                    while block := handle.read(1024**2): digest.update(block)
                cache[key] = dict(dtype=dtype, location=location, elements=elements,
                    bytes=by_name[name].file_size, sha256=digest.hexdigest())
            return cache[key]
        def tensor(*arguments):
            if len(arguments) < 4: raise ValueError('Unsupported native tensor descriptor')
            return dict(native_tensor_descriptor=True, storage=arguments[0], offset=arguments[1],
                shape=arguments[2], stride=arguments[3], remaining=arguments[4:])
        class Reader(pickle.Unpickler):
            def persistent_load(self, identity): return storage(identity)
            def find_class(self, module, name):
                if (module,name) == ('collections','OrderedDict'): return OrderedDict
                if module == 'torch._utils' and name in ('_rebuild_tensor_v2','_rebuild_tensor_v3'):
                    return tensor
                if module == 'torch' and (name.endswith('Storage') or name in ('bfloat16','float32','float64','int64','uint8','bool')):
                    return module+'.'+name
                raise ValueError('Unsupported pickle global: '+module+'.'+name)
        payload = Reader(io.BytesIO(archive.read(pickles[0]))).load()
        if not isinstance(payload,dict) or not isinstance(payload.get('state'),dict):
            raise ValueError('Native snapshot state required')
        state = dict(payload['state']); state.pop('work_ledger',None)
        def typed(value):
            if isinstance(value,dict):
                entries = [(typed(key),typed(item)) for key,item in value.items()]
                return ['dict', sorted(entries,key=lambda pair:json.dumps(pair[0],sort_keys=True))]
            if type(value) is tuple: return ['tuple',[typed(item) for item in value]]
            if type(value) is list: return ['list',[typed(item) for item in value]]
            if value is None or type(value) in (str,int,float,bool): return [type(value).__name__,value]
            raise ValueError('Unknown symbolic snapshot primitive')
        return dict(phase=payload['phase'], completed_updates=payload['completed_updates'],
            scientific_state_sha256=canonical_hash(typed(state)),
            components={key:canonical_hash(typed(value)) for key,value in state.items()},
            scope='Exact streamed storage bytes and typed native science descriptors; excludes only physical work prefix and producer invocation')


def pilot_export_binding(record, report):
    """Accept exported bytes only alongside the actual committed final16 state.

    These bounded parent-side payload/storage hashes are explicitly outside the
    native work23/I/O9 journals; they do not deserialize or allocate model weights.
    """
    locations=paths(record['stage'],record['group'],record['run_id'])
    if (record['stage']!='pilot' or record['recipe']['updates']!=16
            or report.get('status')!='completed' or report.get('committed_completed_updates')!=16):
        raise ValueError('Only the actual completed16 pilot exports can be accepted')
    output=locations['pilot'];path=output/'snapshots/completed-000016.pt'
    marker,marker_bytes=read_bounded_json(str(path)+'.commit.json')
    receipt,receipt_bytes=read_bounded_json(str(path)+'.work.json');validate_work_receipt(receipt)
    contract,contract_bytes=read_bounded_json(output/'recovery-contract.json',maximum=1024**2)
    latest=dict(path=str(path),**marker,work_receipt_path=str(path)+'.work.json')
    if (marker!=receipt['snapshot'] or marker['phase']!='completed' or marker['completed_updates']!=16
            or marker['contract_sha256']!=canonical_hash(contract) or report.get('latest_durable_snapshot')!=latest):
        raise ValueError('Actual final16 report/marker/receipt/scientific contract differs')
    payload=_digest(str(path),maximum=ENVELOPE['max_payload_bytes'])
    if payload['sha256']!=marker['payload_sha256'] or payload['bytes']!=marker['payload_bytes']:
        raise ValueError('Actual final16 committed payload bytes differ from marker')
    scientific=symbolic_snapshot(path)
    if (scientific['phase']!='completed' or scientific['completed_updates']!=16
            or scientific['components'].get('cursor')!=canonical_hash(['int',16])
            or not {'policy','reference','optimizer','rng','history'}<=set(scientific['components'])):
        raise ValueError('Actual committed16 numerical policy/reference/optimizer/RNG identity required')
    policy=output/'policy';files=artifact_hashes(policy)
    genealogy=json.loads((policy/'course-genealogy.json').read_text())
    if (not files or 'config.json' not in files or not any(name.endswith('.safetensors') for name in files)
            or 'adapter_config.json' in files or genealogy.get('kind')!='full-HF-model'
            or genealogy.get('objective')!='course-response-mean-GRPO'
            or genealogy.get('parent_local_source_hashes')!=record['local_model_binding']['files']
            or genealogy.get('checkpoint_interface')!=record['observed_geometry']['checkpoint_interface']
            or genealogy.get('upstream_revision_metadata')!=REVISION
            or genealogy.get('template_sha256')!=record['observed_geometry']['prompt_contract']['template_sha256']):
        raise ValueError('Actual full16 HF export genealogy/interface differs')
    if artifact_hashes(policy)!=files:raise ValueError('Actual exported policy changed while accepting bytes')
    return dict(path=str(policy),files=files,completed_updates=16,
        report_binding=_digest(str(output/'report.json')),
        final_snapshot=dict(path=str(path),header=marker,payload_binding=payload,scientific_state=scientific,
            metadata_bindings=dict(marker=marker_bytes,receipt=receipt_bytes,contract=contract_bytes)),
        scope='Accepted exported bytes tied to the completed16 report/receipt/actual streamed numerical final state; parent-side hashes outside work23/I/O9')


def check_results(record):
    locations = {key:Path(value) for key,value in record['locations'].items()}
    roles = ('source','completed','pending') if record['stage'] == 'recovery' else ('pilot',)
    reports = {role:json.loads((locations[role]/'report.json').read_text()) for role in roles}
    checks = {role+'_completed_horizon': report.get('status') == 'completed'
        and report.get('committed_completed_updates') == record['recipe']['updates'] for role,report in reports.items()}
    checks['original_prompt_and_source_contract'] = all(
        json.loads((locations[role]/'recovery-contract.json').read_text())['train'] == record['observed_geometry']['train']
        and json.loads((locations[role]/'recovery-contract.json').read_text())['evaluation'] == record['observed_geometry']['evaluation']
        and report['local_source_hashes'] == record['local_model_binding']['files'] for role,report in reports.items())
    science = {}
    if record['stage'] == 'recovery':
        for role in roles:
            science[role] = symbolic_snapshot(locations[role]/'snapshots/completed-000002.pt')
        checks['exact_final_policy_reference_optimizer_rng_loop_history'] = science['source'] == science['completed'] == science['pending']
        checks['exact_update2_metrics'] = (reports['source']['records'][1:] == reports['completed']['records'] == reports['pending']['records'])
        checks['pending_does_not_resample'] = (reports['pending']['cumulative_work_ledger']['completed']['collections']
            == reports['completed']['cumulative_work_ledger']['completed']['collections'])
        for journal_key in ('cumulative_work_ledger','cumulative_snapshot_io_ledger'):
            histories = [reports[role][journal_key] for role in roles]
            checks[journal_key+'_same_physical_journal'] = len({(row['ledger_id'],tuple(row['file_identity'])) for row in histories}) == 1
            checks[journal_key+'_retains_later_spending'] = all(
                later['reserved'][key] >= earlier['reserved'][key] for earlier,later in zip(histories,histories[1:])
                for key in earlier['reserved'])
    checks['no_open_or_failed_journal_tickets'] = all(not report[key][field] for report in reports.values()
        for key in ('cumulative_work_ledger','cumulative_snapshot_io_ledger') for field in ('open_tickets','failed_tickets'))
    exports={'pilot':pilot_export_binding(record,reports['pilot'])} if record['stage']=='pilot' else {}
    return dict(checks=checks, scientific_snapshots=science, reports=reports,exports=exports,
        scope='Original actual-loop numerical/recovery and four heldout greedy diagnostics; no twenty-item publication or positive-quality requirement')


def execute(record, declaration):
    if type(declaration) is not str or not 12 <= len(declaration) <= 4096:
        raise ValueError('Explicit recorded goal-authorized RLVR stage declaration required')
    evidence = Path(record['locations']['evidence']); invocations=[]; resumes={}
    try:
        # -m must resolve this byte-bound repository source, even when the shared
        # GPU environment does not have the teaching package installed.
        os.environ['PYTHONPATH'] = str(ROOT/'src')
        for name in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE'):
            os.environ[name] = '1'
        roles = ('source','completed','pending') if record['stage'] == 'recovery' else ('pilot',)
        for number, role in enumerate(roles,1):
            verify_prepared(record)
            binding = resumes.get(role)
            if binding is not None: verify_resume(binding)
            argv = fixed_argv(record, role, binding)
            _new_target(record['locations'][role])
            retain(evidence/f'launch-{number}.json', dict(role=role,argv=argv,operator_declaration=declaration,
                preparation_sha256=record['preparation_sha256'], resume_binding=binding))
            result = _supervise(argv, evidence/f'supervision-{number}', native=True,
                seconds=record['recipe']['external_seconds_each'], probe=_native_probe,
                operator_declaration=declaration, native_stage='reasoning-pilot' if record['stage'] == 'pilot' else 'profile')
            retain(evidence/f'returned-supervision-{number}.json',result)
            invocations.append(dict(role=role, status=result['status'],actual_exit_code=result['actual_exit_code'],
                child_seconds=result['child_seconds'],minimum_sampled_available_bytes=result['minimum_sampled_available_bytes']))
            verify_prepared(record)
            if binding is not None: verify_resume(binding)
            if result['status'] != 'completed' or result['actual_exit_code'] != 0:
                raise RuntimeError('Owned RLVR child failed; all evidence/costs retained')
            if role == 'source': resumes=retain_resume_inputs(record)
        verification = check_results(record)
        verify_prepared(record)
        for binding in resumes.values(): verify_resume(binding)
        retain(evidence/'closing-bindings.json',dict(status='unchanged',source_bindings=record['source_bindings'],
            input_bindings=record['input_bindings'], local_model_binding=record['local_model_binding'],
            prerequisite_bindings=record['prerequisite_bindings'], resume_bindings=resumes))
        status = 'passed' if all(verification['checks'].values()) else 'failed'
        retain(evidence/'acceptance.json',dict(status=status,stage=record['stage'],group=record['group'],
            local_model_binding=record['local_model_binding'],invocations=invocations, **verification,
            boundaries=record['boundaries']))
        if status != 'passed': raise RuntimeError('Actual numerical/recovery acceptance failed; evidence retained')
        print(json.dumps(dict(status=status,stage=record['stage'],group=record['group'],evidence=str(evidence))))
        return 0
    except BaseException as error:
        retain(evidence/'failure.json',dict(status='failed',type=type(error).__name__,message=str(error),
            invocations=invocations,boundary='Retained original journals/checkpoints/report contain partial and failed spending; no retry/cap change'))
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('recovery','pilot'),required=True)
    parser.add_argument('--group-size',type=int,choices=(4,8),required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--operator-declaration')
    args=parser.parse_args(argv)
    paths(args.stage,args.group_size,args.run_id)
    if args.execute and (not args.operator_declaration or not 12 <= len(args.operator_declaration) <= 4096):
        parser.error('Explicit recorded goal-authorized RLVR stage declaration required')
    record=prepare(args.stage,args.group_size,args.run_id)
    if args.execute:return execute(record,args.operator_declaration)
    print(json.dumps(dict(status='prepared-not-executed',stage=args.stage,group=args.group_size,
        evidence=record['locations']['evidence'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
