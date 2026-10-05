#!/usr/bin/env python3
"""Fixed local DPO2 replay and DPO100 from the predeclared full400 parent.

Preparation tokenizes original bytes on CPU, reads safetensors headers and binds
the actual parent; it never loads model weights or starts a child. Execution has
closed argv and retained independent snapshot expectations. The replay performs
clean2/source2/resume1->2; source and resume retain the same physical journals.
Policy/reference weights are FP32, with BF16 autocast in the native CUDA runner.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from dongxi_llms.native_profile_supervisor import _digest, _native_probe, _new_target, _supervise, MODEL, REVISION
from dongxi_llms.run_identity import artifact_hashes, cached_snapshot, canonical_hash
from dongxi_llms.snapshot_io_budget import (IO_KEYS, SnapshotIOBudget, io_budget_contract,
    io_ledger_contract_sha256, read_bounded_json, validate_work_receipt)
from dongxi_llms.snapshot_io_schedule import operation_cost
from dongxi_llms.work_budget import WorkLedger
from dongxi_llms.artifact_budget import ArtifactBudget

GIB = 1024**3
INTERPRETER = '/home/dongxi/dgx-spark-dongxi/.venv/bin/python'
ENVIRONMENT_LOCK = '/home/dongxi/dgx-spark-dongxi/uv.lock'
PARENT_STEM = 'native-sft-full-pilot400-20261005-run-01'
REPLAY_RUN_ID = 'run-03'
CPU_ENTRY = 'scripts/run_native_dpo_cpu8_child.py'
CPU_SETTINGS = dict(omp_num_threads='8',torch_num_threads=8,torch_num_interop_threads=1)
CPU_WITNESS_PREFIX = 'DONGXI_DPO_CPU8 '
HASH_WORKERS = 4
HASH_WITNESS_PREFIX = 'DONGXI_SNAPSHOT_HASH '
MAX_SNAPSHOT = 16*GIB
JOURNAL_BYTES = 4*1024**2
CONTRACT_MAX_BYTES = 1024**2
IO_ENVELOPE = dict(max_payload_bytes=MAX_SNAPSHOT,max_tree_nodes=100000,
    max_tensor_elements=4000000000,max_tensor_bytes=MAX_SNAPSHOT,max_primitive_bytes=1024**2)
SOURCES = ('scripts/run_native_dpo_stages.py','tests/test_native_dpo_stages.py',CPU_ENTRY,
    'scripts/run_chapter11_spark_dpo.py','src/dongxi_llms/dpo_lab.py',
    'src/dongxi_llms/run_identity.py','src/dongxi_llms/training_snapshot.py',
    'src/dongxi_llms/batched_cache_lab.py','src/dongxi_llms/work_budget.py',
    'src/dongxi_llms/artifact_budget.py','src/dongxi_llms/snapshot_io_budget.py',
    'src/dongxi_llms/snapshot_io_schedule.py','src/dongxi_llms/native_profile_supervisor.py',
    'experiments/specs/2026-10-04-staged-spark-campaign.md')
COMPONENTS = ('schema','policy','reference','optimizer','sampler_rng','torch_rng',
    'cuda_rng','completed','counters','history')
EXCLUDED = ('resume_parent','work_ledger','snapshot_artifact_ledger')
REPLAY_CHECKS = ('all_actual_children_completed','equal_numerical_components','unchanged_reference',
    'completed2','same_scientific_contract','clean_journals','exact_numerical_metric_tail','actual_fixed_cpu8_threads',
    'actual_bounded_complete_hash_workers')


def native_runner():
    spec = importlib.util.spec_from_file_location('native_dpo_stage_runner',ROOT/'scripts/run_chapter11_spark_dpo.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def retain(path,value,*,compact=False):
    with Path(path).open('x',encoding='utf-8') as handle:
        json.dump(value,handle,indent=None if compact else 2,
            separators=(',',':') if compact else None,allow_nan=False)
        handle.write('\n')


def bounded_contract(path):
    # Trusted producer science has311 state tensors and310 optimizer entries.
    # Receipts continue through the shared64KiB reader; this role is separately
    # bounded at1MiB and retained without dropping any scientific field.
    binding = _digest(str(path),maximum=CONTRACT_MAX_BYTES)
    value = json.loads(Path(path).read_text())
    if type(value) is not dict or _digest(str(path),maximum=CONTRACT_MAX_BYTES) != binding:
        raise ValueError('Bounded trusted contract changed while reading')
    return value


def locations(mode,run_id):
    if mode not in ('replay','pilot') or type(run_id) is not str or re.fullmatch(r'run-[0-9]{2}',run_id) is None:
        raise ValueError('Only replay/pilot and a fixed run-NN identifier are supported')
    stem = f'native-dpo-{mode}-20261005-{run_id}'
    groups = ('clean','shared') if mode=='replay' else ('pilot',)
    roles = ('clean','source','resumed') if mode=='replay' else ('pilot',)
    return dict(evidence=ROOT/'experiments/reports'/stem,
        outputs={role:ROOT/'outputs'/(stem+'-'+role) for role in roles},
        journals={group:ROOT/'outputs'/(stem+'-'+group+'-journals') for group in groups},
        artifacts={group:ROOT/'outputs'/(stem+'-'+group+'-snapshots') for group in groups})


def source_bindings():
    return {name:_digest(str(ROOT/name)) for name in SOURCES}


def parent_binding():
    parent = ROOT/'outputs'/PARENT_STEM/'policy'
    acceptance = ROOT/'experiments/reports'/PARENT_STEM/'acceptance.json'
    accepted = json.loads(acceptance.read_text())
    files = artifact_hashes(parent)
    genealogy = json.loads((parent/'course-genealogy.json').read_text())
    if (accepted.get('status')!='passed' or not accepted.get('checks')
            or not all(value is True for value in accepted['checks'].values())
            or accepted.get('result',{}).get('updates')!=400
            or accepted.get('exported_policy',{}).get('path')!=str(parent)
            or accepted['exported_policy'].get('files')!=files
            or genealogy.get('kind')!='full-HF-model' or genealogy.get('base_model')!=MODEL
            or genealogy.get('base_revision')!=REVISION or 'adapter_config.json' in files
            or 'model.safetensors' not in files):
        raise ValueError('Exact actual accepted full400 parent/export required')
    return dict(path=str(parent),files=files,genealogy=genealogy,
        acceptance=_digest(str(acceptance)),selected_parent='predeclared full400 final, never a quality-selected substitute')


def input_bindings(mode):
    values = {name:ROOT/'fixtures/chapter11'/(name+'.jsonl') for name in ('train','validation','evaluation')}
    values.update(interpreter=Path(INTERPRETER),environment_lock=Path(ENVIRONMENT_LOCK))
    if mode=='pilot':
        values['actual_replay_acceptance'] = locations('replay',REPLAY_RUN_ID)['evidence']/'acceptance.json'
        values['actual_replay_preparation'] = locations('replay',REPLAY_RUN_ID)['evidence']/'preparation.json'
        for role in ('clean','source','resumed','cpu-comparison'):
            evidence=locations('replay',REPLAY_RUN_ID)['evidence'];name='comparison' if role=='cpu-comparison' else role
            values['actual_replay_cpu_witness/'+role]=evidence/('cpu-thread-witness-'+role+'.json')
            values['actual_replay_cpu_stdout/'+role]=evidence/('supervision-'+name)/'stdout.txt'
            values['actual_replay_hash_witness/'+role]=evidence/('snapshot-hash-witness-'+role+'.json')
    return {key:_digest(str(path),executable=key=='interpreter') for key,path in values.items()}


def _vector_sum(keys,*vectors):
    return {key:sum(vector.get(key,0) for vector in vectors) for key in keys}


def _times(vector,count):
    return {key:value*count for key,value in vector.items()}


def observed_geometry(parent,inputs):
    from transformers import AutoTokenizer
    dpo = native_runner()
    saved = AutoTokenizer.from_pretrained(parent['path'],local_files_only=True)
    snapshot = cached_snapshot(MODEL,REVISION)
    provided = AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
    interface,_ = dpo.verify_parent_tokenizer(Path(parent['path']),saved,provided,
        tokenizer_id=MODEL,tokenizer_revision=REVISION)
    rows = {key:dpo.load_rows(inputs[key]['path']) for key in ('train','validation','evaluation')}
    if [len(rows[key]) for key in ('train','validation','evaluation')] != [8,4,4]:
        raise ValueError('Original8/4/4 preference populations required')
    encoded = {key:[dpo.encode_pair(provided,row,512) for row in rows[key]] for key in ('train','validation')}
    prefixes = [provided.apply_chat_template(row['prompt'],tokenize=True,
        add_generation_prompt=True,enable_thinking=False,return_dict=False) for row in rows['evaluation']]
    if any(not prefix or len(prefix)+64>512 for prefix in prefixes):
        raise ValueError('Original prompt plus64 tokens must fit without truncation')
    validation = dict.fromkeys(dpo.BUDGET_KEYS,0)
    for pair in encoded['validation']:
        work = dpo.example_work(pair)
        validation['valid_targets'] += work['chosen_targets']+work['rejected_targets']
        validation['logical_sequence_tokens'] += work['logical_sequence_tokens']
        for role in ('policy','reference'):
            validation[role+'_forward_calls'] += 2
            validation[role+'_forward_positions'] += work[role+'_forward_positions']
            validation['evaluation_calls'] += 2
            validation['evaluation_positions'] += work[role+'_forward_positions']
    weight = Path(parent['path'])/'model.safetensors'
    with weight.open('rb') as handle:
        length = struct.unpack('<Q',handle.read(8))[0]
        if not 1<=length<=CONTRACT_MAX_BYTES:raise ValueError('Bounded safetensors header required')
        header = json.loads(handle.read(length))
    tensors = {name:dict(shape=row['shape'],dtype='torch.float32')
        for name,row in header.items() if name!='__metadata__'}
    state = deepcopy(tensors)
    config = json.loads((Path(parent['path'])/'config.json').read_text())
    if config.get('tie_word_embeddings') is True and 'lm_head.weight' not in state:
        state['lm_head.weight'] = deepcopy(state['model.embed_tokens.weight'])
    layout = dict(updates=100,accumulation=4,sample_work=[dpo.example_work(pair) for pair in encoded['train']],
        policy_shapes=state,optimizer_shapes=list(tensors.values()),
        torch_rng_bytes=65536,cuda_rng_count=16,cuda_rng_bytes=[1024**2]*16)
    return dict(interface=interface,encoded_train_sha256=dpo.state_digest(encoded['train']),
        encoded_validation_sha256=dpo.state_digest(encoded['validation']),
        prefix_sha256=dpo.state_digest(prefixes),prefix_lengths=[len(prefix) for prefix in prefixes],
        one_update=dpo.update_upper(encoded['train'],4),one_validation=validation,
        one_generation=dpo.generation_upper(prefixes,64),validation_layout=layout,
        layout_provenance='actual saved tensor header shapes + tied-output alias; FP32 target dtype; bounded CPU/CUDA RNG allowances are declarations',
        tokenizer_snapshot=dict(path=str(snapshot),files=artifact_hashes(snapshot)),
        weights_loaded=False,cuda_observed=False)


def allowances(mode,geometry):
    dpo = native_runner()
    if mode=='replay':
        updates,checkpoint,seconds = 2,1,600
        # Source full2 plus replayed1; three clean/source/resumed CPU final reads
        # are split between journals. The shared group's maximum is declared.
        model = _vector_sum(dpo.BUDGET_KEYS,_times(geometry['one_update'],3),
            _times(geometry['one_validation'],2),_times(geometry['one_generation'],4))
        semantic_cursors = [0,1,2,1,1,1,2]
        saves,inspects,loads,artifact_gib,planning_gib = 5,3,3,81,180
    else:
        updates,checkpoint,seconds = 100,20,1800
        model = _vector_sum(dpo.BUDGET_KEYS,_times(geometry['one_update'],100),
            geometry['one_validation'],_times(geometry['one_generation'],2))
        semantic_cursors = [0,20,40,60,80,100]
        saves,inspects,loads,artifact_gib,planning_gib = 6,0,0,97,110
    layout = deepcopy(geometry['validation_layout']);layout['updates']=updates
    semantic = _vector_sum(dpo.BUDGET_KEYS,
        *(dpo.recovery_validation_upper(layout,cursor) for cursor in semantic_cursors))
    caps = _vector_sum(dpo.BUDGET_KEYS,model,semantic)
    operations = {key:operation_cost(IO_ENVELOPE,key,
        **({'payload_bytes':MAX_SNAPSHOT} if key!='save' else {})) for key in ('save','inspect','load')}
    io = io_budget_contract(_vector_sum(IO_KEYS,_times(operations['save'],saves),
        _times(operations['inspect'],inspects),_times(operations['load'],loads)),IO_ENVELOPE,JOURNAL_BYTES)
    return dict(updates=updates,checkpoint_every=checkpoint,external_seconds=seconds,
        native_stage='profile' if mode=='replay' else 'preference-pilot',work_caps=caps,snapshot_io_contract=io,
        artifact_max_bytes=artifact_gib*GIB,artifact_max_entries=32,artifact_journal_bytes=JOURNAL_BYTES,
        output_planning_bytes=planning_gib*GIB,semantic_validation_cursors=semantic_cursors,
        independent_comparison=dict(final_cpu_inspect_loads=3 if mode=='replay' else 0,
            maximum_tensor_bytes_per_payload=MAX_SNAPSHOT,
            scope='shared readers charged to I/O9; independent component framing/hash outside DPO19/I/O9; no CUDA generator validation'))


def fixed_command(record,role):
    places = locations(record['mode'],record['run_id']);limits = record['limits']
    group = ('clean' if role=='clean' else 'shared') if record['mode']=='replay' else 'pilot'
    if role not in places['outputs']:raise ValueError('Unknown fixed DPO child role')
    journal,evidence = places['journals'][group],places['evidence']
    command = [INTERPRETER,str(ROOT/CPU_ENTRY),
        '--checkpoint',record['parent_binding']['path'],'--tokenizer',MODEL,'--tokenizer-revision',REVISION]
    for key in ('train','validation','evaluation'):
        command.extend(['--'+key,record['input_bindings'][key]['path']])
    return command+['--output',str(places['outputs'][role]),'--updates',str(limits['updates']),
        '--accumulation','4','--beta','0.1','--lr','5e-07','--max-length','512',
        '--max-new-tokens','64','--seed','1818','--checkpoint-every',str(limits['checkpoint_every']),
        '--snapshot-max-bytes',str(MAX_SNAPSHOT),'--work-limits',str(evidence/'work-caps.json'),
        '--work-journal-max-bytes',str(JOURNAL_BYTES),'--work-journal',str(journal/'work.jsonl'),
        '--snapshot-io-limits',str(evidence/'io-caps.json'),'--snapshot-io-ledger',str(journal/'io.jsonl'),
        '--snapshot-artifact-max-bytes',str(limits['artifact_max_bytes']),
        '--snapshot-artifact-max-entries',str(limits['artifact_max_entries']),
        '--snapshot-artifact-journal-max-bytes',str(JOURNAL_BYTES),
        '--snapshot-hash-workers',str(HASH_WORKERS),
        '--snapshot-artifact-root',str(places['artifacts'][group]),'--environment-lock',ENVIRONMENT_LOCK]


def comparison_command(run_id):
    locations('replay',run_id)
    return [INTERPRETER,str(ROOT/CPU_ENTRY),'--mode','replay','--run-id',run_id,'--comparison-child']


def cpu_witness_binding(evidence,role,*,comparison=False):
    name='comparison' if comparison else role
    path=Path(evidence)/('supervision-'+name)/'stdout.txt';binding=_digest(str(path))
    lines=[line[len(CPU_WITNESS_PREFIX):] for line in path.read_text().splitlines() if line.startswith(CPU_WITNESS_PREFIX)]
    if len(lines)!=1:raise ValueError('Exactly one actual CPU8 stdout witness required')
    actual=json.loads(lines[0]);target='scripts/run_native_dpo_stages.py' if comparison else 'scripts/run_chapter11_spark_dpo.py'
    expected=dict(schema='dongxi-fixed-native-dpo-cpu8-witness-v1',target=str(ROOT/target),**CPU_SETTINGS)
    if actual!=expected or _digest(str(path))!=binding:raise ValueError('Actual native DPO CPU8 thread witness differs')
    return dict(role=role,actual=actual,stdout=binding)


def retained_cpu_witness(evidence,role,*,comparison=False):
    result=cpu_witness_binding(evidence,role,comparison=comparison)
    retain(Path(evidence)/('cpu-thread-witness-'+role+'.json'),result);return result


def hash_witness_binding(evidence,role,*,comparison=False):
    name='comparison' if comparison else role
    path=Path(evidence)/('supervision-'+name)/'stdout.txt';binding=_digest(str(path))
    actual=[json.loads(line[len(HASH_WITNESS_PREFIX):]) for line in path.read_text().splitlines()
        if line.startswith(HASH_WITNESS_PREFIX)]
    base=dict(schema='dongxi-snapshot-hash-runtime-v1',hash_workers=HASH_WORKERS)
    expected=[dict(base,role=item) for item in ('clean','source','resumed')] if comparison else [base]
    if actual!=expected or _digest(str(path))!=binding:
        raise ValueError('Actual bounded complete-file hash worker witness differs')
    return dict(role=role,actual=actual,stdout=binding)


def retained_hash_witness(evidence,role,*,comparison=False):
    result=hash_witness_binding(evidence,role,comparison=comparison)
    retain(Path(evidence)/('snapshot-hash-witness-'+role+'.json'),result);return result


def prepare(mode,run_id):
    places = locations(mode,run_id)
    for path in (places['evidence'],*places['outputs'].values(),*places['journals'].values(),*places['artifacts'].values()):
        _new_target(str(path))
    evidence = places['evidence'];evidence.mkdir(mode=0o700)
    try:
        sources,inputs,parent = source_bindings(),input_bindings(mode),parent_binding()
        replay = None
        if mode=='pilot':
            passed = json.loads(Path(inputs['actual_replay_acceptance']['path']).read_text())
            if passed.get('status')!='passed' or any(passed.get('checks',{}).get(key) is not True for key in REPLAY_CHECKS):
                raise ValueError('Actual selected-parent clean/source/fresh DPO2 replay required before100')
            launches = passed.get('invocations',[])
            if ([row.get('role') for row in launches]!=['clean','source','resumed','cpu-comparison']
                    or any(row.get('result',{}).get('status')!='completed'
                        or row.get('result',{}).get('actual_exit_code')!=0 for row in launches)):
                raise ValueError('Three actual replay children and accounted comparison receipts required')
            if passed.get('parent_binding')!=parent:raise ValueError('DPO2 replay used a different selected parent')
            for launch in launches:
                role=launch['role'];bound=cpu_witness_binding(locations('replay',REPLAY_RUN_ID)['evidence'],role,comparison=role=='cpu-comparison')
                if (launch.get('cpu_thread_witness')!=bound
                        or json.loads(Path(inputs['actual_replay_cpu_witness/'+role]['path']).read_text())!=bound):
                    raise ValueError('Actual CPU8 replay stdout/retained thread witnesses differ')
                hashed=hash_witness_binding(locations('replay',REPLAY_RUN_ID)['evidence'],role,comparison=role=='cpu-comparison')
                if (launch.get('snapshot_hash_witness')!=hashed
                        or json.loads(Path(inputs['actual_replay_hash_witness/'+role]['path']).read_text())!=hashed):
                    raise ValueError('Actual replay stdout/retained complete-file hash witnesses differ')
            replay = json.loads(Path(inputs['actual_replay_preparation']['path']).read_text())
            if replay.get('mode')!='replay' or replay.get('run_id')!=REPLAY_RUN_ID:
                raise ValueError('Only the predeclared CPU8/parallel-hash run-03 replay preparation supports100')
            verify_prepared(replay)
            if (replay['parent_binding']!=parent or replay['source_bindings']!=sources
                    or any(replay['input_bindings'].get(key)!=inputs[key]
                        for key in ('train','validation','evaluation','interpreter','environment_lock'))):
                raise ValueError('Replay-to-pilot source/input/parent science changed')
        geometry = observed_geometry(parent,inputs);limits = allowances(mode,geometry)
        if replay is not None and replay['geometry']!=geometry:
            raise ValueError('Replay-to-pilot tokenizer/interface/encoded geometry changed')
        if shutil.disk_usage(ROOT/'outputs').free<limits['output_planning_bytes']:
            raise RuntimeError('Declared DPO snapshot/export planning space unavailable')
        for journal in places['journals'].values():journal.mkdir(mode=0o700)
        retain(evidence/'work-caps.json',limits['work_caps'])
        retain(evidence/'io-caps.json',limits['snapshot_io_contract'])
        inputs.update(work_caps=_digest(str(evidence/'work-caps.json')),io_caps=_digest(str(evidence/'io-caps.json')))
        record = dict(schema='dongxi-fixed-native-dpo-stages-v1',mode=mode,run_id=run_id,
            utc=datetime.now(timezone.utc).isoformat(),evidence=str(evidence),source_bindings=sources,
            input_bindings=inputs,parent_binding=parent,geometry=geometry,limits=limits,
            cpu_settings=deepcopy(CPU_SETTINGS),comparison_command=comparison_command(run_id) if mode=='replay' else None,
            snapshot_hash_workers=HASH_WORKERS,
            science='FP32 policy/reference weights, BF16 CUDA autocast; fixed1818/5e-7/beta.1/acc4/512/64',
            scope='original location fixture, predeclared full400 parent; no broad preference/publication claim')
        record['commands'] = {role:fixed_command(record,role) for role in places['outputs']}
        record['preparation_sha256'] = canonical_hash(record)
        retain(evidence/'preparation.json',record)
        return record
    except BaseException as error:
        retain(evidence/'preparation-failure.json',dict(status='failed',type=type(error).__name__,message=str(error)))
        raise


def verify_prepared(record):
    if record.get('preparation_sha256')!=canonical_hash({k:v for k,v in record.items() if k!='preparation_sha256'}):
        raise ValueError('Prepared command or allowances changed')
    places = locations(record['mode'],record['run_id'])
    if (record['evidence']!=str(places['evidence']) or record['commands']!={role:fixed_command(record,role) for role in places['outputs']}
            or record.get('cpu_settings')!=CPU_SETTINGS
            or record.get('snapshot_hash_workers')!=HASH_WORKERS
            or record.get('comparison_command')!=(comparison_command(record['run_id']) if record['mode']=='replay' else None)):
        raise ValueError('Only the closed fixed DPO stage commands are supported')
    if record['limits']!=allowances(record['mode'],record['geometry']):raise ValueError('Fixed stage allowance mapping changed')
    if source_bindings()!=record['source_bindings']:raise ValueError('Frozen DPO executable source changed')
    current = input_bindings(record['mode'])
    if any(record['input_bindings'].get(k)!=v for k,v in current.items()):raise ValueError('Fixed input role changed')
    for key,binding in record['input_bindings'].items():
        if _digest(binding['path'],executable=key=='interpreter')!=binding:raise ValueError('Bound input bytes changed: '+key)
    if parent_binding()!=record['parent_binding']:raise ValueError('Selected full400 parent bytes changed')
    snapshot = record['geometry']['tokenizer_snapshot']
    if artifact_hashes(snapshot['path'])!=snapshot['files']:raise ValueError('Local tokenizer bytes changed')
    if json.loads((places['evidence']/'work-caps.json').read_text())!=record['limits']['work_caps']:
        raise ValueError('Frozen work caps changed')
    if json.loads((places['evidence']/'io-caps.json').read_text())!=record['limits']['snapshot_io_contract']:
        raise ValueError('Frozen I/O caps changed')


def retain_expectation(record,role,result,*,cursor=None):
    places = locations(record['mode'],record['run_id']);output = places['outputs'][role]
    latest = result['completed_recovery'];path = Path(latest['path'])
    expected_cursor = record['limits']['updates'] if cursor is None else cursor
    if cursor is not None:
        match = re.fullmatch(r'([0-9a-f]{32})-completed-[0-9]{6}\.pt',path.name)
        if match is None:raise ValueError('Expected native namespaced DPO snapshot')
        path = path.with_name(f'{match[1]}-completed-{cursor:06d}.pt')
    group = ('clean' if role=='clean' else 'shared') if record['mode']=='replay' else 'pilot'
    if path.parent!=places['artifacts'][group]:raise ValueError('Snapshot escaped the fixed artifact root')
    header,_ = read_bounded_json(Path(str(path)+'.commit.json'))
    receipt,_ = read_bounded_json(Path(str(path)+'.work.json'));validate_work_receipt(receipt)
    if header!=receipt['snapshot'] or header['phase']!='completed' or header['completed_updates']!=expected_cursor:
        raise ValueError('Independent snapshot header/receipt/cursor mismatch')
    contract = bounded_contract(output/'recovery-contract.json')
    prefix = places['evidence']/f'{role}-completed-{expected_cursor:06d}'
    retain(Path(str(prefix)+'-contract.json'),contract,compact=True)
    retain(Path(str(prefix)+'-work-receipt.json'),receipt)
    retain(Path(str(prefix)+'-artifact-receipt.json'),result['snapshot_artifact_ledger'])
    retain(Path(str(prefix)+'-header.json'),header)
    metadata = {key:_digest(str(Path(str(prefix)+'-'+name+'.json'))) for key,name in
        (('contract','contract'),('work_receipt','work-receipt'),('artifact_receipt','artifact-receipt'),('header','header'))}
    expectation = dict(path=str(path),header=header,metadata=metadata,group=group,role=role)
    retain(Path(str(prefix)+'-expectation.json'),expectation)
    return expectation


def verify_expectation(expectation):
    for binding in expectation['metadata'].values():
        if _digest(binding['path'])!=binding:raise ValueError('Independently retained snapshot metadata changed')


def resume_flags(expectation):
    verify_expectation(expectation);meta,header = expectation['metadata'],expectation['header']
    return ['--resume',expectation['path'],'--resume-contract',meta['contract']['path'],
        '--resume-sha256',header['payload_sha256'],'--resume-bytes',str(header['payload_bytes']),
        '--resume-io-receipt',meta['work_receipt']['path'],
        '--snapshot-artifact-receipt',meta['artifact_receipt']['path']]


def component_fingerprints(state):
    from dongxi_llms.batched_cache_lab import digest
    if set(state)!=set(COMPONENTS)|set(EXCLUDED):raise ValueError('Unexpected numerical DPO state schema')
    return {key:digest(state[key]) for key in COMPONENTS}


def comparison_digest(record,expectation):
    from dongxi_llms.training_snapshot import inspect_snapshot,load_snapshot
    verify_expectation(expectation)
    limits = record['limits'];contract = bounded_contract(expectation['metadata']['contract']['path'])
    receipt,_ = read_bounded_json(expectation['metadata']['work_receipt']['path'])
    artifacts,_ = read_bounded_json(expectation['metadata']['artifact_receipt']['path'])
    places = locations(record['mode'],record['run_id']);journal = places['journals'][expectation['group']]
    expected_group = 'clean' if expectation['role']=='clean' else 'shared'
    if (record['mode']!='replay' or expectation['role'] not in ('clean','source','resumed')
            or expectation['group']!=expected_group
            or Path(expectation['path']).parent!=places['artifacts'][expected_group]
            or expectation['header']['phase']!='completed' or expectation['header']['completed_updates']!=2):
        raise ValueError('Only the three fixed completed2 comparison payloads are supported')
    started = time.monotonic()
    def guard():
        memory = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
        if memory<25*GIB or time.monotonic()-started>600:raise RuntimeError('CPU comparison reserve/deadline crossed')
    science = canonical_hash(contract);io_contract = limits['snapshot_io_contract']
    work=io=artifact=None
    try:
        work = WorkLedger.open(journal/'work.jsonl',limits=limits['work_caps'],contract_sha256=science,
            max_bytes=JOURNAL_BYTES,invocation_id='independent-final-comparison',expected_snapshot=receipt['runner_work_prefix'])
        io = WorkLedger.open(journal/'io.jsonl',limits=io_contract['limits'],contract_sha256=io_ledger_contract_sha256(io_contract,science),
            max_bytes=JOURNAL_BYTES,invocation_id='independent-final-comparison',expected_snapshot=receipt['io_prefix'])
        artifact = ArtifactBudget.restore(places['artifacts'][expectation['group']],expected_receipt=artifacts,
            hash_workers=HASH_WORKERS)
        print(HASH_WITNESS_PREFIX+json.dumps(dict(schema='dongxi-snapshot-hash-runtime-v1',
            hash_workers=artifact.hash_workers,role=expectation['role']),sort_keys=True),flush=True)
        hook = SnapshotIOBudget(io,contract=io_contract,scientific_contract_sha256=science,expected_receipt=receipt)
        kwargs = dict(expected_sha256=expectation['header']['payload_sha256'],expected_bytes=expectation['header']['payload_bytes'],
            expected_contract=contract,max_bytes=MAX_SNAPSHOT,guard=guard,io_budget=hook)
        inspect_snapshot(expectation['path'],**kwargs)
        # Native semantic CUDA-generator restoration is not called here. The
        # shared CPU reader checks bounded tensor/tree/receipt visits; independent
        # typed hashes compare every numerical component without applying state.
        payload = load_snapshot(expectation['path'],**kwargs)
        state = payload['state']
        work.validate_snapshot(state['work_ledger']);artifact.validate_receipt(state['snapshot_artifact_ledger'])
        guard();components = component_fingerprints(state);guard()
        result = dict(components=components,reference_matches_original=components['reference']==contract['reference_sha256'],
            completed=state['completed'],contract_sha256=science,io_ledger=io.snapshot(),work_ledger=work.snapshot(),
            scope='accounted shared CPU inspect/load; typed numerical hashes outside DPO19/I/O9; no CUDA state application')
        del payload,state;gc.collect()
        return result
    finally:
        for ledger in (artifact,io,work):
            if ledger is not None:ledger.close()


def comparison_child(mode,run_id):
    if mode!='replay':raise ValueError('Comparison child supports only the fixed replay')
    import torch
    actual=dict(omp_num_threads=os.environ.get('OMP_NUM_THREADS'),torch_num_threads=torch.get_num_threads(),
        torch_num_interop_threads=torch.get_num_interop_threads())
    if actual!=CPU_SETTINGS:raise ValueError('Fixed CPU8 entry required before comparison state validation')
    evidence = locations(mode,run_id)['evidence'];record = json.loads((evidence/'preparation.json').read_text())
    verify_prepared(record)
    expectations = json.loads((evidence/'comparison-expectations.json').read_text())
    if set(expectations)!=set(('clean','source','resumed')):raise ValueError('All three fixed final expectations required')
    results = {role:comparison_digest(record,expectations[role]) for role in ('clean','source','resumed')}
    checks = dict(equal_numerical_components=results['clean']['components']==results['source']['components']==results['resumed']['components'],
        unchanged_reference=all(row['reference_matches_original'] for row in results.values()),
        completed2=all(row['completed']==2 for row in results.values()),
        same_scientific_contract=len({row['contract_sha256'] for row in results.values()})==1,
        clean_journals=all(not row[key][field] for row in results.values()
            for key in ('io_ledger','work_ledger') for field in ('open_tickets','failed_tickets')))
    verify_prepared(record)
    retain(evidence/'comparison.json',dict(status='passed' if all(checks.values()) else 'failed',checks=checks,
        results=results,excluded_operational_fields=EXCLUDED))
    return 0 if all(checks.values()) else 1


def execute(record,declaration):
    if type(declaration) is not str or not 12<=len(declaration)<=4096:raise ValueError('Recorded bounded goal-authorized DPO declaration required')
    places = locations(record['mode'],record['run_id']);evidence = places['evidence']
    results,expectations,invocations = {},{},[]
    try:
        for role in places['outputs']:
            verify_prepared(record);_new_target(str(places['outputs'][role]))
            argv = record['commands'][role]
            if role=='resumed':argv = argv+resume_flags(expectations['resume-parent'])
            retain(evidence/f'launch-{role}.json',dict(argv=argv,operator_declaration=declaration,
                preparation_sha256=record['preparation_sha256']))
            result = _supervise(argv,evidence/('supervision-'+role),native=True,
                seconds=record['limits']['external_seconds'],probe=_native_probe,
                operator_declaration=declaration,native_stage=record['limits']['native_stage'])
            retain(evidence/f'returned-supervision-{role}.json',result);invocations.append(dict(role=role,result=result))
            verify_prepared(record)
            if result['status']!='completed' or result['actual_exit_code']!=0:raise RuntimeError('Actual DPO child failed; retained spending and supervisor preserved')
            invocations[-1]['cpu_thread_witness']=retained_cpu_witness(evidence,role)
            invocations[-1]['snapshot_hash_witness']=retained_hash_witness(evidence,role)
            actual = json.loads((places['outputs'][role]/'result.json').read_text());results[role]=actual
            expectations[role] = retain_expectation(record,role,actual)
            if role=='source':expectations['resume-parent'] = retain_expectation(record,role,actual,cursor=1)
        checks = dict(all_actual_children_completed=True,actual_fixed_cpu8_threads=True,
            actual_bounded_complete_hash_workers=True)
        comparison = None
        if record['mode']=='replay':
            retain(evidence/'comparison-expectations.json',{role:expectations[role] for role in ('clean','source','resumed')})
            argv = record['comparison_command']
            retain(evidence/'launch-comparison.json',dict(argv=argv,operator_declaration=declaration,
                scope='owned CPU final comparison; shared inspect/load charged to unchanged journals'))
            result = _supervise(argv,evidence/'supervision-comparison',native=True,seconds=600,
                probe=_native_probe,operator_declaration=declaration)
            retain(evidence/'returned-supervision-comparison.json',result);invocations.append(dict(role='cpu-comparison',result=result))
            if result['status']!='completed' or result['actual_exit_code']!=0:raise RuntimeError('Owned CPU comparison failed; spending retained')
            invocations[-1]['cpu_thread_witness']=retained_cpu_witness(evidence,'cpu-comparison',comparison=True)
            invocations[-1]['snapshot_hash_witness']=retained_hash_witness(evidence,'cpu-comparison',comparison=True)
            comparison = json.loads((evidence/'comparison.json').read_text());checks.update(comparison['checks'])
            dpo = native_runner()
            project = lambda output:[{key:row[key] for key in ('update','loss','margin','gradient_norm','indices','work')}
                for row in (json.loads(line) for line in (output/'metrics.jsonl').read_text().splitlines())]
            clean,source,resumed = (project(places['outputs'][role]) for role in ('clean','source','resumed'))
            checks['exact_numerical_metric_tail'] = len(clean)==2 and clean==source and resumed==source[1:]
        else:
            actual = results['pilot'];metrics = [json.loads(line) for line in (places['outputs']['pilot']/'metrics.jsonl').read_text().splitlines()]
            checks.update(completed100=actual['completed_recovery']['completed_updates']==100,
                exactly100_metrics=[row['update'] for row in metrics]==list(range(1,101)),
                clean_journals=all(not actual[key][field] for key in ('cumulative_work_ledger','snapshot_io_ledger')
                    for field in ('open_tickets','failed_tickets')))
        verify_prepared(record)
        exports = {role:dict(path=str(output/'policy'),files=artifact_hashes(output/'policy')) for role,output in places['outputs'].items()}
        summary = dict(status='passed' if all(checks.values()) else 'failed',checks=checks,
            parent_binding=record['parent_binding'],results=results,comparison=comparison,
            invocations=invocations,exports=exports,scope=record['scope'],science=record['science'])
        retain(evidence/'acceptance.json',summary)
        if summary['status']!='passed':raise RuntimeError('Actual DPO stage acceptance failed; evidence retained')
        print(json.dumps(dict(status='passed',mode=record['mode'],evidence=str(evidence))))
        return 0
    except BaseException as error:
        retain(evidence/'failure.json',dict(status='failed',type=type(error).__name__,message=str(error),
            invocations=invocations,scope='all failed/later spending preserved; no implicit restart or cap refill'))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('replay','pilot'),required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--operator-declaration')
    parser.add_argument('--comparison-child',action='store_true',help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.comparison_child:
        if args.execute or args.operator_declaration:parser.error('Comparison child has a fixed separate route')
        return comparison_child(args.mode,args.run_id)
    if args.execute and (not args.operator_declaration or not 12<=len(args.operator_declaration)<=4096):
        parser.error('Explicit bounded goal-authorized DPO declaration required')
    record = prepare(args.mode,args.run_id)
    if args.execute:return execute(record,args.operator_declaration)
    print(json.dumps(dict(status='prepared-not-executed',evidence=record['evidence'])))
    return 0


if __name__=='__main__':raise SystemExit(main())
