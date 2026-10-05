#!/usr/bin/env python3
"""Fixed accounted chosen-only recovery2 and matched100 from accepted full400.

Preparation is local CPU tokenization/header inspection, never model loading.
Actual children have closed argv, FP32 parent weights and BF16 CUDA autocast.
Three clean2/source2/resume1->2 children retain later spending in the same
physical work19/I/O9 journals. Typed final CPU hashes are separately supervised;
they are not physical work quotas. ChosenSFTLoop has no artifact-budget API:
per-payload I/O bounds and explicit output-space planning are not artifact quotas.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dongxi_llms.native_profile_supervisor import _digest,_native_probe,_new_target,_supervise,MODEL,REVISION
from dongxi_llms.run_identity import artifact_hashes,canonical_hash,environment_identity,IdentityJournal
from dongxi_llms.snapshot_io_budget import (IO_KEYS,SnapshotIOBudget,io_budget_contract,
    io_ledger_contract_sha256,read_bounded_json,validate_work_receipt)
from dongxi_llms.snapshot_io_schedule import operation_cost
from dongxi_llms.work_budget import WorkLedger

GIB=1024**3
INTERPRETER='/home/dongxi/dgx-spark-dongxi/.venv/bin/python'
ENVIRONMENT_LOCK='/home/dongxi/dgx-spark-dongxi/uv.lock'
MAX_SNAPSHOT=16*GIB
JOURNAL_BYTES=4*1024**2
CONTRACT_MAX_BYTES=1024**2
IO_ENVELOPE=dict(max_payload_bytes=MAX_SNAPSHOT,max_tree_nodes=100000,
    max_tensor_elements=4000000000,max_tensor_bytes=MAX_SNAPSHOT,max_primitive_bytes=1024**2)
SOURCES=('scripts/run_native_chosen_stages.py','tests/test_native_chosen_stages.py',
    'tests/test_chosen_sft_control.py','tests/test_chosen_sft_recovery.py',
    'scripts/run_native_dpo_stages.py','tests/test_native_dpo_stages.py',
    'src/dongxi_llms/chosen_sft_control.py','scripts/run_chapter09_spark_sft.py',
    'scripts/run_chapter11_spark_dpo.py','src/dongxi_llms/dpo_lab.py',
    'src/dongxi_llms/work_budget.py','src/dongxi_llms/training_snapshot.py',
    'src/dongxi_llms/snapshot_io_budget.py','src/dongxi_llms/snapshot_io_schedule.py',
    'src/dongxi_llms/run_identity.py','src/dongxi_llms/batched_cache_lab.py',
    'src/dongxi_llms/native_profile_supervisor.py',
    'experiments/specs/2026-10-04-staged-spark-campaign.md',
    'experiments/specs/2026-10-05-chosen-sft-accounted-recovery.md')
COMPONENTS=('schema','model','optimizer','sampler_rng','torch_rng','cuda_rng','completed','history')
EXCLUDED=('work_ledger',)
REPLAY_CHECKS=('all_actual_children_completed','equal_numerical_components','unchanged_reference',
    'completed2','same_scientific_contract','clean_journals','exact_numerical_metric_tail')


def bridge():
    spec=importlib.util.spec_from_file_location('chosen_fixed_dpo_helpers',ROOT/'scripts/run_native_dpo_stages.py')
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def chosen():
    from dongxi_llms import chosen_sft_control
    return chosen_sft_control


def retain(path,value,*,compact=False):
    with Path(path).open('x',encoding='utf-8') as handle:
        json.dump(value,handle,indent=None if compact else 2,
            separators=(',',':') if compact else None,allow_nan=False)
        handle.write('\n')


def bounded_contract(path):
    value,_=read_bounded_json(path,maximum=CONTRACT_MAX_BYTES)
    if type(value) is not dict:raise ValueError('Plain bounded trusted chosen contract required')
    return value


def locations(mode,run_id):
    if mode not in ('replay','pilot') or type(run_id) is not str or re.fullmatch(r'run-[0-9]{2}',run_id) is None:
        raise ValueError('Only replay/pilot and fixed run-NN identifiers are supported')
    stem=f'native-chosen-{mode}-20261005-{run_id}'
    roles=('clean','source','resumed') if mode=='replay' else ('pilot',)
    groups=('clean','shared') if mode=='replay' else ('pilot',)
    return dict(evidence=ROOT/'experiments/reports'/stem,
        outputs={role:ROOT/'outputs'/(stem+'-'+role) for role in roles},
        journals={group:ROOT/'outputs'/(stem+'-'+group+'-journals') for group in groups})


def group_for(mode,role):
    places=locations(mode,'run-01')
    if role not in places['outputs']:raise ValueError('Unknown fixed chosen child role')
    return ('clean' if role=='clean' else 'shared') if mode=='replay' else 'pilot'


def source_bindings():return {name:_digest(str(ROOT/name)) for name in SOURCES}


def parent_binding():return bridge().parent_binding()


def input_bindings(mode):
    paths={name:ROOT/'fixtures/chapter11'/(name+'.jsonl') for name in ('train','validation','evaluation')}
    paths.update(protocol=ROOT/'fixtures/matched-chosen-sft/protocol.json',
        interpreter=Path(INTERPRETER),environment_lock=Path(ENVIRONMENT_LOCK))
    if mode=='pilot':
        evidence=locations('replay','run-01')['evidence']
        paths.update(actual_replay_acceptance=evidence/'acceptance.json',actual_replay_preparation=evidence/'preparation.json')
    return {key:_digest(str(path),executable=key=='interpreter') for key,path in paths.items()}


def settings(updates):
    return chosen().recipe(seed=1818,updates=updates,accumulation=4,learning_rate=5e-7,
        beta=.1,max_length=512,max_new_tokens=64)


def encoded_dataset(tokenizer,inputs):
    native=bridge().native_runner()
    protocol=json.loads(Path(inputs['protocol']['path']).read_text())
    rows=[native.load_rows(inputs[key]['path']) for key in ('train','validation','evaluation')]
    if [len(split) for split in rows]!=[8,4,4]:raise ValueError('Original8/4/4 chosen populations required')
    ids={row['id'] for split in rows for row in split}
    if (protocol.get('schema')!='dongxi-matched-chosen-sft-fixture-v1'
            or set(protocol.get('groups',{}))!=ids
            or protocol.get('inputs')!={key:'fixtures/chapter11/'+key+'.jsonl' for key in ('train','validation','evaluation')}):
        raise ValueError('Exact original scenario-group sidecar required, not its CPU recipe')
    grouped=[[dict(row,group=protocol['groups'][row['id']]) for row in split] for split in rows]
    # Actual original accepted template: real im_end151645 followed by newline198.
    # Both IDs/masks are retained and supervised, never trimmed or fake stops.
    data=chosen().encode_dataset(tokenizer,*grouped,max_length=512,terminal_suffix_ids=[198])
    if any(len(prefix)+64>512 for prefix in data['prefixes']):raise ValueError('Original prefix plus64 must fit without truncation')
    return data


def observed_geometry(parent,inputs):
    from transformers import AutoTokenizer
    base=bridge().observed_geometry(parent,inputs)
    tokenizer=AutoTokenizer.from_pretrained(base['tokenizer_snapshot']['path'],local_files_only=True)
    saved=AutoTokenizer.from_pretrained(parent['path'],local_files_only=True)
    interface,_=bridge().native_runner().verify_parent_tokenizer(Path(parent['path']),saved,tokenizer,
        tokenizer_id=MODEL,tokenizer_revision=REVISION)
    if interface!=base['interface']:raise ValueError('Preparation parent/tokenizer interface changed')
    data=encoded_dataset(tokenizer,inputs)
    samples=[dict(sampled_pairs=1,sampler_draws=1,chosen_targets=sum(pair[0][1][1:]),
        rejected_targets=0,logical_sequence_tokens=len(pair[0][0]),policy_forward_calls=1,
        reference_forward_calls=0,policy_forward_positions=len(pair[0][0]),reference_forward_positions=0)
        for pair in data['train']]
    base['one_update']=dict(train_updates=1,sampled_examples=4,sampler_draws=4,
        valid_targets=4*max(row['chosen_targets'] for row in samples),
        logical_sequence_tokens=4*max(row['logical_sequence_tokens'] for row in samples),
        policy_forward_calls=4,policy_forward_positions=4*max(row['policy_forward_positions'] for row in samples))
    layout=base['validation_layout']
    layout['sample_work']=samples;layout['cpu_rng_bytes']=layout.pop('torch_rng_bytes');layout.pop('cuda_rng_count')
    base['chosen_dataset']={key:deepcopy(data[key]) for key in ('rows_sha256','encoded_sha256','interface_sha256',
        'train_ids','train_groups','validation_ids','evaluation_ids','pad_id','stop_ids','split','terminal_suffix_ids')}
    base['layout_provenance']='actual saved tensor header shapes + tied-output alias; FP32 target; conservative CPU/CUDA RNG declarations, not CUDA observations'
    return base


def _sum(keys,*vectors):return {key:sum(vector.get(key,0) for vector in vectors) for key in keys}


def _times(vector,count):return {key:value*count for key,value in vector.items()}


def allowances(mode,geometry):
    native=bridge().native_runner();control=chosen()
    if mode=='replay':
        updates,checkpoint,seconds=2,1,600
        # Shared physical domain: source2 plus resumed1. Only one semantic load
        # callback occurs in chosen.restore, unlike native DPO's second check.
        model=_sum(native.BUDGET_KEYS,_times(geometry['one_update'],3),
            _times(geometry['one_validation'],2),_times(geometry['one_generation'],4))
        cursors=[0,1,2,1,1,2];saves,inspects,loads,planning=5,3,3,180*GIB
    else:
        updates,checkpoint,seconds=100,20,1800
        model=_sum(native.BUDGET_KEYS,_times(geometry['one_update'],100),
            geometry['one_validation'],_times(geometry['one_generation'],2))
        cursors=[0,20,40,60,80,100];saves,inspects,loads,planning=6,0,0,110*GIB
    operations={role:operation_cost(IO_ENVELOPE,role,
        **({'payload_bytes':MAX_SNAPSHOT} if role!='save' else {})) for role in ('save','inspect','load')}
    io=io_budget_contract(_sum(IO_KEYS,_times(operations['save'],saves),
        _times(operations['inspect'],inspects),_times(operations['load'],loads)),IO_ENVELOPE,JOURNAL_BYTES)
    # A declared metadata-only skeleton is passed to the actual helper; it is
    # never a bound recovery contract or fabricated model/CUDA observation.
    declared=dict(schema=control.RECOVERY_SCHEMA,loop=deepcopy(geometry['validation_layout']),
        parent_sha256='0'*64,parent_files={},inputs={},sources={},environment={},interface_sha256='0'*64,
        dataset={},recipe=settings(updates),work_budget=control.chosen_work_budget_contract(
            dict.fromkeys(native.BUDGET_KEYS,0),JOURNAL_BYTES),snapshot_io_budget=io)
    declared['loop']['updates']=updates
    semantics=_sum(native.BUDGET_KEYS,*(control.chosen_validation_upper(declared,cursor) for cursor in cursors))
    return dict(updates=updates,checkpoint_every=checkpoint,external_seconds=seconds,
        native_stage='profile' if mode=='replay' else 'preference-pilot',
        work_caps=_sum(native.BUDGET_KEYS,model,semantics),snapshot_io_contract=io,
        semantic_validation_cursors=cursors,output_planning_bytes=planning,
        artifact_scope='no ArtifactBudget integration in chosen API; per-payload I/O bound and disk planning are not artifact/physical quotas',
        independent_comparison=dict(final_cpu_inspect_loads=3 if mode=='replay' else 0,
            maximum_tensor_bytes_per_payload=MAX_SNAPSHOT,
            scope='shared readers charged to I/O9; typed component/reference byte hashes outside chosen19/I/O9 under external supervision'))


def fixed_command(record,role):
    if role not in locations(record['mode'],record['run_id'])['outputs']:raise ValueError('Unknown fixed chosen child role')
    return [INTERPRETER,str(ROOT/'scripts/run_native_chosen_stages.py'),'--mode',record['mode'],
        '--run-id',record['run_id'],'--training-child','--role',role]


def prepare(mode,run_id):
    places=locations(mode,run_id)
    for path in (places['evidence'],*places['outputs'].values(),*places['journals'].values()):_new_target(str(path))
    evidence=places['evidence'];evidence.mkdir(mode=0o700)
    sources=inputs=parent=None
    try:
        sources,inputs,parent=source_bindings(),input_bindings(mode),parent_binding()
        replay=None
        if mode=='pilot':
            accepted=json.loads(Path(inputs['actual_replay_acceptance']['path']).read_text())
            if accepted.get('status')!='passed' or any(accepted.get('checks',{}).get(key) is not True for key in REPLAY_CHECKS):
                raise ValueError('Actual selected-parent chosen2 replay required before100')
            invocations=accepted.get('invocations',[])
            if ([row.get('role') for row in invocations]!=['clean','source','resumed','cpu-comparison']
                    or any(row.get('result',{}).get('status')!='completed'
                        or row.get('result',{}).get('actual_exit_code')!=0 for row in invocations)):
                raise ValueError('Three actual chosen children and accounted comparison receipts required')
            replay=json.loads(Path(inputs['actual_replay_preparation']['path']).read_text())
            if replay.get('mode')!='replay' or replay.get('run_id')!='run-01':raise ValueError('Exact selected run-01 replay required')
            verify_prepared(replay)
            if (accepted.get('parent_binding')!=parent or replay['parent_binding']!=parent or replay['source_bindings']!=sources
                    or any(replay['input_bindings'].get(key)!=inputs[key]
                        for key in ('train','validation','evaluation','protocol','interpreter','environment_lock'))):
                raise ValueError('Replay-to-pilot source/input/parent science changed')
        geometry=observed_geometry(parent,inputs)
        if replay is not None and geometry!=replay['geometry']:raise ValueError('Replay-to-pilot encoded/interface geometry changed')
        limits=allowances(mode,geometry)
        if shutil.disk_usage(ROOT/'outputs').free<limits['output_planning_bytes']:raise RuntimeError('Declared chosen output planning space unavailable')
        for path in places['journals'].values():path.mkdir(mode=0o700)
        retain(evidence/'work-caps.json',limits['work_caps']);retain(evidence/'io-caps.json',limits['snapshot_io_contract'])
        inputs.update(work_caps=_digest(str(evidence/'work-caps.json')),io_caps=_digest(str(evidence/'io-caps.json')))
        record=dict(schema='dongxi-fixed-native-chosen-stages-v1',mode=mode,run_id=run_id,
            utc=datetime.now(timezone.utc).isoformat(),evidence=str(evidence),source_bindings=sources,
            input_bindings=inputs,parent_binding=parent,geometry=geometry,limits=limits,recipe=settings(limits['updates']),
            science='native summed chosen-token NLL / accumulation valid targets; FP32 policy/reference, BF16 CUDA;1818/5e-7/acc4/512/64',
            scope='original location control and predeclared full400 parent; not broad capability/publication evidence')
        record['commands']={role:fixed_command(record,role) for role in places['outputs']}
        record['preparation_sha256']=canonical_hash(record);retain(evidence/'preparation.json',record)
        return record
    except BaseException as error:
        retain(evidence/'preparation-failure.json',dict(status='failed',type=type(error).__name__,message=str(error),
            source_bindings=sources,input_bindings=inputs,parent_binding=parent,
            scope='actual CPU tokenizer/header preparation failure; no model weights loaded or CUDA child launched'));raise


def verify_prepared(record):
    if record.get('preparation_sha256')!=canonical_hash({key:value for key,value in record.items() if key!='preparation_sha256'}):
        raise ValueError('Prepared chosen configuration changed')
    places=locations(record['mode'],record['run_id'])
    if (record.get('schema')!='dongxi-fixed-native-chosen-stages-v1' or record['evidence']!=str(places['evidence'])
            or record['commands']!={role:fixed_command(record,role) for role in places['outputs']}
            or record['recipe']!=settings(record['limits']['updates'])):raise ValueError('Only fixed chosen commands/recipe supported')
    if record['limits']!=allowances(record['mode'],record['geometry']):raise ValueError('Fixed chosen allowance mapping changed')
    if source_bindings()!=record['source_bindings']:raise ValueError('Frozen chosen source changed')
    if any(record['input_bindings'].get(key)!=value for key,value in input_bindings(record['mode']).items()):
        raise ValueError('Fixed chosen input role changed')
    for key,binding in record['input_bindings'].items():
        if _digest(binding['path'],executable=key=='interpreter')!=binding:raise ValueError('Bound chosen input bytes changed: '+key)
    if parent_binding()!=record['parent_binding']:raise ValueError('Selected full400 parent bytes changed')
    snapshot=record['geometry']['tokenizer_snapshot']
    if artifact_hashes(snapshot['path'])!=snapshot['files']:raise ValueError('Local tokenizer bytes changed')
    for name,key in (('work-caps.json','work_caps'),('io-caps.json','snapshot_io_contract')):
        if json.loads((places['evidence']/name).read_text())!=record['limits'][key]:raise ValueError('Frozen chosen caps changed')


def retain_expectation(record,role,result,*,cursor=None):
    output=locations(record['mode'],record['run_id'])['outputs'][role]
    endpoint=record['limits']['updates'] if cursor is None else cursor
    path=output/'checkpoints'/f'completed-{endpoint:06d}.pt'
    header,_=read_bounded_json(str(path)+'.commit.json')
    receipt,_=read_bounded_json(str(path)+'.work.json');validate_work_receipt(receipt)
    if (header!=receipt['snapshot'] or header['phase']!='completed' or header['completed_updates']!=endpoint
            or (cursor is None and result['completed_recovery']['path']!=str(path))):
        raise ValueError('Independent chosen snapshot header/receipt/cursor mismatch')
    contract=bounded_contract(output/'recovery-contract.json')
    prefix=locations(record['mode'],record['run_id'])['evidence']/f'{role}-completed-{endpoint:06d}'
    retain(str(prefix)+'-contract.json',contract,compact=True)
    retain(str(prefix)+'-work-receipt.json',receipt);retain(str(prefix)+'-header.json',header)
    metadata={key:_digest(str(prefix)+'-'+suffix+'.json') for key,suffix in
        (('contract','contract'),('work_receipt','work-receipt'),('header','header'))}
    value=dict(path=str(path),header=header,metadata=metadata,role=role,group=group_for(record['mode'],role))
    retain(str(prefix)+'-expectation.json',value);return value


def verify_expectation(record,expectation,*,final=False):
    places=locations(record['mode'],record['run_id']);role=expectation['role']
    cursor=2 if final else 1
    if (record['mode']!='replay' or role not in ('clean','source','resumed')
            or expectation['group']!=group_for(record['mode'],role)
            or expectation['path']!=str(places['outputs'][role]/'checkpoints'/f'completed-{cursor:06d}.pt')
            or expectation['header']['phase']!='completed' or expectation['header']['completed_updates']!=cursor
            or (not final and role!='source')):raise ValueError('Only fixed independent completed chosen expectations supported')
    prefix=places['evidence']/f'{role}-completed-{cursor:06d}'
    for key,suffix in (('contract','contract'),('work_receipt','work-receipt'),('header','header')):
        binding=expectation['metadata'][key]
        if binding['path']!=str(prefix)+'-'+suffix+'.json' or _digest(binding['path'])!=binding:
            raise ValueError('Independently retained chosen metadata changed')
    retained,_=read_bounded_json(expectation['metadata']['header']['path'])
    if retained!=expectation['header']:raise ValueError('Retained chosen header differs from expectation')


def comparison_digest(record,expectation):
    from dongxi_llms.batched_cache_lab import digest
    from dongxi_llms.training_snapshot import inspect_snapshot,load_snapshot
    verify_expectation(record,expectation,final=True)
    contract=bounded_contract(expectation['metadata']['contract']['path'])
    receipt,_=read_bounded_json(expectation['metadata']['work_receipt']['path'])
    journal=locations(record['mode'],record['run_id'])['journals'][expectation['group']]
    work=io=None;started=time.monotonic()
    def guard():
        available=next(int(row.split()[1])*1024 for row in Path('/proc/meminfo').read_text().splitlines() if row.startswith('MemAvailable:'))
        if available<25*GIB or time.monotonic()-started>600:raise RuntimeError('CPU chosen comparison reserve/deadline crossed')
    try:
        work,io,hook=chosen().open_chosen_resume_budgets(contract=contract,receipt=receipt,
            work_path=journal/'work.jsonl',io_path=journal/'io.jsonl',invocation_id='independent-chosen-final-comparison',
            expected_sha256=expectation['header']['payload_sha256'],expected_bytes=expectation['header']['payload_bytes'])
        kwargs=dict(expected_sha256=expectation['header']['payload_sha256'],expected_bytes=expectation['header']['payload_bytes'],
            expected_contract=contract,max_bytes=MAX_SNAPSHOT,guard=guard,io_budget=hook)
        inspect_snapshot(expectation['path'],**kwargs)
        payload=load_snapshot(expectation['path'],**kwargs);state=payload['state']
        work.validate_snapshot(state['work_ledger']);guard();components=component_fingerprints(state);guard()
        result=dict(components=components,completed=state['completed'],contract_sha256=canonical_hash(contract),
            work_ledger=work.snapshot(),io_ledger=io.snapshot(),
            scope='shared CPU inspect/load charged to I/O9; typed hashes outside chosen19/I/O9; no CUDA RNG application')
        del state,payload;gc.collect();return result
    finally:
        for ledger in (io,work):
            if ledger is not None:ledger.close()


def component_fingerprints(state):
    from dongxi_llms.batched_cache_lab import digest
    if set(state)!=set(COMPONENTS)|set(EXCLUDED):raise ValueError('Unexpected numerical chosen state schema')
    return {key:digest(state[key]) for key in COMPONENTS}


def comparison_child(mode,run_id):
    if mode!='replay':raise ValueError('Only fixed chosen replay supports comparison')
    evidence=locations(mode,run_id)['evidence'];record=json.loads((evidence/'preparation.json').read_text())
    verify_prepared(record);expectations=json.loads((evidence/'comparison-expectations.json').read_text())
    if set(expectations)!=set(('clean','source','resumed')):raise ValueError('Three fixed final expectations required')
    results={role:comparison_digest(record,expectations[role]) for role in ('clean','source','resumed')}
    actual={role:json.loads((locations(mode,run_id)['outputs'][role]/'result.json').read_text()) for role in results}
    checks=dict(equal_numerical_components=results['clean']['components']==results['source']['components']==results['resumed']['components'],
        unchanged_reference=all(row.get('reference_sha256')==row.get('final_reference_sha256')
            and row.get('reference_has_gradients') is False for row in actual.values())
            and len({row['reference_sha256'] for row in actual.values()})==1,
        completed2=all(row['completed']==2 for row in results.values()),
        same_scientific_contract=len({row['contract_sha256'] for row in results.values()})==1,
        clean_journals=all(not row[key][field] for row in results.values()
            for key in ('work_ledger','io_ledger') for field in ('open_tickets','failed_tickets')))
    verify_prepared(record);retain(evidence/'comparison.json',dict(status='passed' if all(checks.values()) else 'failed',
        checks=checks,results=results,excluded_operational_fields=EXCLUDED))
    return 0 if all(checks.values()) else 1


def training_child(mode,run_id,role,declaration):
    """Actual CUDA execution occurs only in this closed externally owned child."""
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from dongxi_llms.batched_cache_lab import digest
    from dongxi_llms.training_snapshot import inspect_snapshot
    places=locations(mode,run_id);group=group_for(mode,role);output=places['outputs'][role]
    record=json.loads((places['evidence']/'preparation.json').read_text());verify_prepared(record)
    _new_target(str(output));output.mkdir(mode=0o700);(output/'checkpoints').mkdir(mode=0o700)
    journal=IdentityJournal(output,dict(mode=mode,run_id=run_id,role=role,operator_declaration=declaration,
        preparation_sha256=record['preparation_sha256'],recipe=record['recipe'],limits=record['limits']))
    work=io=None;loop=None;started=time.monotonic();minimum_available=None
    def guard():
        nonlocal minimum_available
        available=next(int(row.split()[1])*1024 for row in Path('/proc/meminfo').read_text().splitlines() if row.startswith('MemAvailable:'))
        minimum_available=available if minimum_available is None else min(minimum_available,available)
        if available<25*GIB or time.monotonic()-started>record['limits']['external_seconds']:
            raise RuntimeError('Chosen CUDA child reserve/deadline crossed')
        if shutil.disk_usage(ROOT/'outputs').free<25*GIB:raise RuntimeError('Chosen retained-output disk reserve crossed')
    try:
        guard()
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('Actual CUDA/BF16 support required')
        native=bridge().native_runner();control=chosen()
        tokenizer=AutoTokenizer.from_pretrained(record['geometry']['tokenizer_snapshot']['path'],local_files_only=True)
        saved=AutoTokenizer.from_pretrained(record['parent_binding']['path'],local_files_only=True)
        interface,adoption=native.verify_parent_tokenizer(Path(record['parent_binding']['path']),saved,tokenizer,
            tokenizer_id=MODEL,tokenizer_revision=REVISION)
        data=encoded_dataset(tokenizer,record['input_bindings'])
        expected_data={key:deepcopy(data[key]) for key in record['geometry']['chosen_dataset']}
        if interface!=record['geometry']['interface'] or expected_data!=record['geometry']['chosen_dataset']:
            raise ValueError('Actual child tokenizer/encoded geometry differs from preparation')
        environment=environment_identity(ENVIRONMENT_LOCK)
        parent_files={str(Path(record['parent_binding']['path'])/name):value for name,value in record['parent_binding']['files'].items()}
        sources={name:value['sha256'] for name,value in record['source_bindings'].items()}
        inputs={value['path']:value['sha256'] for value in record['input_bindings'].values()}
        expectation=expected_contract=None;hook=None
        if role=='resumed':
            expectation=json.loads((places['evidence']/'resume-expectation.json').read_text());verify_expectation(record,expectation)
            expected_contract=bounded_contract(expectation['metadata']['contract']['path'])
            if (expected_contract['parent_files']!=parent_files or expected_contract['inputs']!=inputs
                    or expected_contract['sources']!=sources or expected_contract['environment']!=environment
                    or expected_contract['recipe']!=record['recipe']
                    or expected_contract['interface_sha256']!=data['interface_sha256']
                    or expected_contract['dataset']!={key:data[key] for key in expected_contract['dataset']}
                    or expected_contract['work_budget']!=control.chosen_work_budget_contract(record['limits']['work_caps'],JOURNAL_BYTES)
                    or expected_contract['snapshot_io_budget']!=record['limits']['snapshot_io_contract']):
                raise ValueError('Independently retained chosen contract disagrees before model loading')
            receipt,_=read_bounded_json(expectation['metadata']['work_receipt']['path'])
            work,io,hook=control.open_chosen_resume_budgets(contract=expected_contract,receipt=receipt,
                work_path=places['journals'][group]/'work.jsonl',io_path=places['journals'][group]/'io.jsonl',
                invocation_id=journal.invocation_id,expected_sha256=expectation['header']['payload_sha256'],
                expected_bytes=expectation['header']['payload_bytes'])
            inspect_snapshot(expectation['path'],expected_sha256=expectation['header']['payload_sha256'],
                expected_bytes=expectation['header']['payload_bytes'],expected_contract=expected_contract,
                max_bytes=MAX_SNAPSHOT,guard=guard,io_budget=hook)
        journal.attach(dict(source_bindings=record['source_bindings'],input_bindings=record['input_bindings'],
            parent_binding=record['parent_binding'],environment=environment,interface=interface,
            device=dict(mode='cuda',name=torch.cuda.get_device_name(),runtime=torch.version.cuda),
            expected_resume=expectation,preparation_sha256=record['preparation_sha256']))
        journal.stage('loading-fixed-FP32-policy-reference');torch.manual_seed(1818);guard()
        model=AutoModelForCausalLM.from_pretrained(record['parent_binding']['path'],local_files_only=True,
            torch_dtype=torch.float32,attn_implementation='sdpa').cuda();guard()
        reference=AutoModelForCausalLM.from_pretrained(record['parent_binding']['path'],local_files_only=True,
            torch_dtype=torch.float32,attn_implementation='sdpa').cuda().eval().requires_grad_(False);guard()
        model.config.use_cache=False;reference.config.use_cache=False;model.gradient_checkpointing_enable()
        for network in (model,reference):
            for module in network.modules():
                if isinstance(module,torch.nn.Dropout):module.p=0.
        optimizer=torch.optim.AdamW(model.parameters(),lr=5e-7,weight_decay=.01)
        loop=control.ChosenSFTLoop(model,optimizer,torch.Generator().manual_seed(1818),data['train'],
            pad_id=data['pad_id'],stop_ids=data['stop_ids'],accumulation=4,updates=record['limits']['updates'],
            device='cuda:0',guard=guard,autocast_dtype=torch.bfloat16,terminal_suffix_ids=data['terminal_suffix_ids'])
        contract=control.make_chosen_recovery_contract(loop,dataset=data,tokenizer=tokenizer,settings=record['recipe'],
            parent_files=parent_files,inputs=inputs,sources=sources,environment=environment,
            work_budget=control.chosen_work_budget_contract(record['limits']['work_caps'],JOURNAL_BYTES),
            snapshot_io_budget=record['limits']['snapshot_io_contract'])
        if expected_contract is not None and contract!=expected_contract:raise ValueError('Actual fresh chosen parent/model/optimizer contract differs')
        science=canonical_hash(contract)
        if role!='resumed':
            work=WorkLedger.create(places['journals'][group]/'work.jsonl',limits=record['limits']['work_caps'],
                contract_sha256=science,max_bytes=JOURNAL_BYTES,invocation_id=journal.invocation_id)
            io=WorkLedger.create(places['journals'][group]/'io.jsonl',limits=contract['snapshot_io_budget']['limits'],
                contract_sha256=io_ledger_contract_sha256(contract['snapshot_io_budget'],science),
                max_bytes=JOURNAL_BYTES,invocation_id=journal.invocation_id)
            hook=SnapshotIOBudget(io,contract=contract['snapshot_io_budget'],scientific_contract_sha256=science)
        loop.bind_recovery(contract,work_ledger=work,io_budget=hook,tokenizer=tokenizer)
        retain(output/'recovery-contract.json',contract,compact=True)
        if role=='resumed':
            journal.stage('restoring-actual-completed1');verify_expectation(record,expectation)
            payload=loop.restore(expectation['path'],expected_sha256=expectation['header']['payload_sha256'],
                expected_bytes=expectation['header']['payload_bytes']);del payload;gc.collect()
            if loop.completed!=1:raise ValueError('Fixed chosen replay resumes only completed1')
        restored=loop.completed;reference_hash=digest(reference.state_dict());guard()
        def commit():
            journal.stage('saving-completed-boundary',completed=loop.completed)
            path=output/'checkpoints'/f'completed-{loop.completed:06d}.pt'
            header=loop.save(path,parent_invocation=journal.invocation_id)
            return dict(path=str(path),completed_updates=loop.completed,header=header,
                work_receipt_path=str(path)+'.work.json')
        def generation(stage):
            return native.generate_dpo_panel(model,tokenizer,data['evaluation'],data['prefixes'],cap=64,
                stop_ids=data['stop_ids'],device='cuda:0',autocast_factory=lambda:torch.autocast('cuda',dtype=torch.bfloat16),
                guard=guard,work_ledger=work,row_sink=lambda row:native.append_metric(output/f'evaluation-{stage}-raw.jsonl',row))
        latest=commit();devices=list(range(torch.cuda.device_count()))
        journal.stage('baseline-generation-after-durable-boundary')
        with torch.random.fork_rng(devices=devices):before=generation('before')
        retain(output/'evaluation-before.json',before)
        for update in range(restored+1,record['limits']['updates']+1):
            journal.stage('chosen-completed-update',next_update=update)
            row=loop.completed_update()
            if update%record['limits']['checkpoint_every']==0 or update==record['limits']['updates']:latest=commit()
            native.append_metric(output/'metrics.jsonl',dict(row,invocation_id=journal.invocation_id,
                last_committed_update=latest['completed_updates'],snapshot=latest['path']))
        if latest['completed_updates']!=record['limits']['updates']:raise ValueError('Final diagnostics require durable complete boundary')
        journal.stage('final-accounted-diagnostics')
        with torch.random.fork_rng(devices=devices):
            validation=native.score_dpo_panel(model,reference,data['validation'],pad_id=data['pad_id'],device='cuda:0',
                beta=.1,autocast_factory=lambda:torch.autocast('cuda',dtype=torch.bfloat16),guard=guard,work_ledger=work)
            after=generation('after')
        retain(output/'evaluation-after.json',dict(validation=validation,after=after))
        if any(row.get('error') is not None for row in before+after):raise RuntimeError('Actual bounded chosen generation failed')
        journal.stage('exporting-full-HF-policy-after-final-commit');guard()
        model.save_pretrained(output/'policy',safe_serialization=True);tokenizer.save_pretrained(output/'policy')
        retain(output/'policy'/'course-genealogy.json',dict(kind='full-HF-chosen-policy',parent_checkpoint=record['parent_binding']['path'],
            parent_checkpoint_sha256=record['parent_binding']['files'],tokenizer=MODEL,tokenizer_revision=REVISION,
            checkpoint_interface=interface,template_sha256=interface['template_sha256'],legacy_adoption=adoption,
            completed_recovery=latest,scientific_contract_sha256=science,objective='chosen-only mean valid response-token NLL'))
        final_reference=digest(reference.state_dict());guard();verify_prepared(record)
        result=dict(completed_recovery=latest,restored_update=restored,before=before,after=after,validation=validation,
            reference_sha256=reference_hash,final_reference_sha256=final_reference,
            reference_has_gradients=any(parameter.grad is not None for parameter in reference.parameters()),
            cumulative_work_ledger=work.snapshot(),snapshot_io_ledger=io.snapshot(),
            minimum_sampled_mem_available_bytes=minimum_available,wall_seconds=time.monotonic()-started,
            scientific_contract_sha256=science,split_evidence=data['split'],
            artifact_scope=record['limits']['artifact_scope'],hash_scope=record['limits']['independent_comparison']['scope'])
        retain(output/'result.json',result);journal.complete(completed=loop.completed)
        return 0
    except BaseException as error:
        journal.fail(error)
        retain(output/'failure.json',dict(status='failed',type=type(error).__name__,message=str(error),
            attempts=[] if loop is None else loop.attempts,
            work_ledger=None if work is None else work.snapshot(),io_ledger=None if io is None else io.snapshot(),
            scope='failed/later physical journal reservations retained; no implicit cap refill'))
        raise
    finally:
        for ledger in (io,work):
            if ledger is not None:ledger.close()


def execute(record,declaration):
    if type(declaration) is not str or not 12<=len(declaration)<=4096:raise ValueError('Bounded goal-authorized chosen declaration required')
    places=locations(record['mode'],record['run_id']);evidence=places['evidence'];results={};expectations={};invocations=[]
    try:
        for role in places['outputs']:
            verify_prepared(record);_new_target(str(places['outputs'][role]))
            argv=record['commands'][role]+['--operator-declaration',declaration]
            retain(evidence/f'launch-{role}.json',dict(argv=argv,preparation_sha256=record['preparation_sha256']))
            result=_supervise(argv,evidence/('supervision-'+role),native=True,seconds=record['limits']['external_seconds'],
                probe=_native_probe,operator_declaration=declaration,native_stage=record['limits']['native_stage'])
            retain(evidence/f'returned-supervision-{role}.json',result);invocations.append(dict(role=role,result=result));verify_prepared(record)
            if result['status']!='completed' or result['actual_exit_code']!=0:raise RuntimeError('Actual chosen child failed; spending retained')
            actual=json.loads((places['outputs'][role]/'result.json').read_text());results[role]=actual
            expectations[role]=retain_expectation(record,role,actual)
            if role=='source':retain(evidence/'resume-expectation.json',retain_expectation(record,role,actual,cursor=1))
        checks=dict(all_actual_children_completed=True);comparison=None
        if record['mode']=='replay':
            retain(evidence/'comparison-expectations.json',expectations)
            argv=[INTERPRETER,str(ROOT/'scripts/run_native_chosen_stages.py'),'--mode','replay','--run-id',record['run_id'],'--comparison-child']
            retain(evidence/'launch-comparison.json',dict(argv=argv,scope='CPU final inspect/load charged to unchanged physical journals'))
            result=_supervise(argv,evidence/'supervision-comparison',native=True,seconds=600,probe=_native_probe,operator_declaration=declaration)
            retain(evidence/'returned-supervision-comparison.json',result);invocations.append(dict(role='cpu-comparison',result=result))
            if result['status']!='completed' or result['actual_exit_code']!=0:raise RuntimeError('Actual chosen comparison failed; spending retained')
            comparison=json.loads((evidence/'comparison.json').read_text());checks.update(comparison['checks'])
            project=lambda path:[{key:row[key] for key in ('update','loss','gradient_norm','indices','work')}
                for row in (json.loads(line) for line in (path/'metrics.jsonl').read_text().splitlines())]
            clean,source,resumed=(project(places['outputs'][role]) for role in ('clean','source','resumed'))
            checks['exact_numerical_metric_tail']=len(clean)==2 and clean==source and resumed==source[1:]
        else:
            actual=results['pilot'];rows=[json.loads(line) for line in (places['outputs']['pilot']/'metrics.jsonl').read_text().splitlines()]
            checks.update(completed100=actual['completed_recovery']['completed_updates']==100,
                exactly100_metrics=[row['update'] for row in rows]==list(range(1,101)),
                unchanged_reference=actual['reference_sha256']==actual['final_reference_sha256'] and actual['reference_has_gradients'] is False,
                clean_journals=all(not actual[key][field] for key in ('cumulative_work_ledger','snapshot_io_ledger')
                    for field in ('open_tickets','failed_tickets')))
        verify_prepared(record)
        exports={role:dict(path=str(path/'policy'),files=artifact_hashes(path/'policy')) for role,path in places['outputs'].items()}
        summary=dict(status='passed' if all(checks.values()) else 'failed',checks=checks,parent_binding=record['parent_binding'],
            invocations=invocations,results=results,comparison=comparison,exports=exports,science=record['science'],scope=record['scope'])
        retain(evidence/'acceptance.json',summary)
        if summary['status']!='passed':raise RuntimeError('Actual chosen acceptance failed; evidence retained')
        print(json.dumps(dict(status='passed',mode=record['mode'],evidence=str(evidence))));return 0
    except BaseException as error:
        retain(evidence/'failure.json',dict(status='failed',type=type(error).__name__,message=str(error),invocations=invocations,
            scope='all failed/later reservations preserved; no implicit restart or cap refill'));raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',required=True,choices=('replay','pilot'));parser.add_argument('--run-id',required=True)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--operator-declaration')
    parser.add_argument('--training-child',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--comparison-child',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--role',choices=('clean','source','resumed','pilot'),help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.comparison_child:
        if args.execute or args.training_child or args.role or args.operator_declaration:parser.error('Comparison has a closed separate route')
        return comparison_child(args.mode,args.run_id)
    if args.training_child:
        if args.execute or not args.role or not args.operator_declaration or not 12<=len(args.operator_declaration)<=4096:
            parser.error('Training child requires its fixed role and bounded declaration')
        return training_child(args.mode,args.run_id,args.role,args.operator_declaration)
    if args.role:parser.error('Role belongs only to the fixed child route')
    if args.execute and (not args.operator_declaration or not 12<=len(args.operator_declaration)<=4096):parser.error('Bounded goal-authorized declaration required')
    record=prepare(args.mode,args.run_id)
    if args.execute:return execute(record,args.operator_declaration)
    print(json.dumps(dict(status='prepared-not-executed',evidence=record['evidence'])));return 0


if __name__=='__main__':raise SystemExit(main())
