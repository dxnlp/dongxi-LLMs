#!/usr/bin/env python3
"""Fixed common20 post-pilot RLVR evaluation; default is tokenizer-only prepare.

Only accepted G4/G8 sixteen-update run-01 policies are selectable. Each group/cap
cell has900 external seconds for four original full-support sampled seeds plus
greedy, all twenty original items, native Instruct template and thinking off.
No weights load on preparation, download, training, recipe selection or retry.
The separate CPU comparison aligns unchanged Instruct/G4/G8 within each cap and
keeps train-overlap diagnostics, failed/capped attempts and measured costs.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
from functools import lru_cache
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dongxi_llms.native_profile_supervisor import _digest,_native_probe,_new_target,_supervise
from dongxi_llms.reasoning_generation import (SOURCE_FILES as GENERATION_SOURCES,
    freeze_local_contract,load_local_tokenizer,run_generation,serialize_prompt)
from dongxi_llms.reasoning_evaluation import replay_records,paired_group_bootstrap
from dongxi_llms.run_identity import artifact_hashes,canonical_hash,TOKENIZER_PATTERNS,assert_compatible

GROUPS=(4,8)
BUDGETS=(32,128)
SEEDS=(1009,1019,1029,1039)
SECONDS=900
GIB=1024**3
CONTEXT=512
EVENT_BYTES=256*1024**2
INTERPRETER='/home/dongxi/dgx-spark-dongxi/.venv/bin/python'
ENVIRONMENT_LOCK='/home/dongxi/dgx-spark-dongxi/uv.lock'
PILOT_CHECKS=('pilot_completed_horizon','original_prompt_and_source_contract','no_open_or_failed_journal_tickets')
BASELINE_CHECKS=('actual_exit0','all_five_cells_completed','all_original_responses','no_unreadable_jsonl')
EVALUATION_CHECKS=BASELINE_CHECKS
SOURCES=tuple(dict.fromkeys((*GENERATION_SOURCES,
    'scripts/run_native_rlvr_evaluation.py','tests/test_native_rlvr_evaluation.py',
    'scripts/run_native_reasoning_baselines.py','tests/test_native_reasoning_baselines.py',
    'scripts/run_native_rlvr_stages.py','tests/test_native_rlvr_stages.py',
    'src/dongxi_llms/qwen_rlvr_lab.py','src/dongxi_llms/grpo_lab.py',
    'src/dongxi_llms/staged_campaign.py','src/dongxi_llms/native_profile_supervisor.py',
    'src/dongxi_llms/campaign_supervisor.py')))


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


@lru_cache(maxsize=1)
def baseline():return module('rlvr_eval_original_baseline',ROOT/'scripts/run_native_reasoning_baselines.py')


@lru_cache(maxsize=1)
def stages():return module('rlvr_eval_actual_pilot',ROOT/'scripts/run_native_rlvr_stages.py')


def retain(path,value):
    with Path(path).open('x',encoding='utf-8') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False)
        handle.write('\n');handle.flush();os.fsync(handle.fileno())


def paths(group,budget,run_id):
    if (type(group) is not int or group not in GROUPS or type(budget) is not int or budget not in BUDGETS
            or type(run_id) is not str or re.fullmatch(r'run-[0-9]{2}',run_id) is None):
        raise ValueError('Only G4/G8,32/128 and fixed run-NN selectors are supported')
    stem=f'native-rlvr-evaluation-g{group}-cap{budget}-20261005-{run_id}'
    return dict(evidence=ROOT/'experiments/reports'/stem,output=ROOT/'outputs'/stem)


def source_bindings():return {name:_digest(str(ROOT/name)) for name in SOURCES}


def original_panel():return baseline().logical_panel()


def response_bindings(record):
    """Fixed retained record/summary paths, including optional failure/events."""
    bindings={}
    for cell in record['cells']:
        for name in ('responses.jsonl','summary.json','events.jsonl','input-identity.json','observed-interface.json','failure.json'):
            path=Path(cell['output'])/name
            if name in ('responses.jsonl','summary.json') or path.exists():
                bindings[cell['id']+'/'+name]=_digest(str(path),maximum=EVENT_BYTES if name=='events.jsonl' else 64*1024**2)
    return bindings


def fixed_limits(budget):
    return dict(external_seconds=900,reserve_bytes=25*GIB,context_window=512,cells=5,items_per_cell=20,
        planned_responses=100,maximum_emitted_tokens=100*budget,event_file_bytes=EVENT_BYTES,ordinary_output_file_bytes=64*1024**2)


def generation_identity(record):
    """Join every retained response/partial cost to frozen model and inputs.

    Missing or malformed identity evidence is a failed gate, not a deletion of
    its raw bytes or observed costs. No environment/Git/hardware probe is run.
    """
    model=record['local_model_binding'];tokenizers=artifact_hashes(model['path'],patterns=TOKENIZER_PATTERNS)
    items_path=Path(record['evidence'])/'items.json';cells=[]
    def input_key(path):
        path=Path(path).resolve()
        return str(path.relative_to(ROOT.resolve())) if path.is_relative_to(ROOT.resolve()) else str(path)
    for cell in record['cells']:
        output=Path(cell['output']);contract=json.loads(Path(cell['contract_path']).read_text());settings=contract['settings']
        interface=settings['generation']['interface'];expected_checkpoint='local-hf-sha256:'+canonical_hash(
            dict(files=model['files'],tokenizer=tokenizers,interface=interface['interface_sha256']))
        expected_inputs={input_key(items_path):record['input_bindings']['items']['sha256'],
            input_key(cell['contract_path']):record['input_bindings']['contract/'+cell['id']]['sha256']}
        expected_sources={name:record['source_bindings'][name]['sha256'] for name in GENERATION_SOURCES}
        issues=[];identity={};summary={};rows=[];partials=[]
        def read_json(name):
            path=output/name
            if not path.exists():return {}
            _digest(str(path),maximum=64*1024**2)
            try:return json.loads(path.read_text())
            except (ValueError,OSError):issues.append('unreadable '+name);return {}
        def read_jsonl(name):
            path=output/name;values=[]
            if path.exists():
                _digest(str(path),maximum=EVENT_BYTES if name=='events.jsonl' else 64*1024**2)
                for number,line in enumerate(path.read_text().splitlines(),1):
                    if not line.strip():continue
                    try:values.append(json.loads(line))
                    except ValueError:issues.append(f'unreadable {name}:{number}')
            return values
        identity=read_json('input-identity.json');summary=read_json('summary.json')
        rows=read_jsonl('responses.jsonl');partials=[event.get('record') for event in read_jsonl('events.jsonl') if event.get('stage')=='partial_response']
        try:
            lock=identity['environment']['environment_lock']
            valid=(identity.get('schema_version')==1
                and identity.get('identity_sha256')==canonical_hash({key:value for key,value in identity.items() if key!='identity_sha256'})
                and identity.get('checkpoint_path')==str(Path(model['path']).resolve()) and identity.get('checkpoint_files')==model['files']
                and identity.get('selected_tokenizer_path')==str(Path(model['path']).resolve()) and identity.get('selected_tokenizer_files')==tokenizers
                and identity.get('config')==settings and identity.get('checkpoint_interface')==interface
                and identity.get('device',{}).get('mode')==settings['generation']['device'] and identity.get('command')==record['argv']
                and identity.get('source_sha256')==expected_sources and identity.get('input_sha256')==expected_inputs
                and lock.get('status')=='hashed' and lock.get('path')==str(Path(record['input_bindings']['environment_lock']['path']).resolve())
                and lock.get('sha256')==record['input_bindings']['environment_lock']['sha256']
                and Path(identity['environment']['interpreter']).resolve()==Path(record['input_bindings']['interpreter']['path']).resolve())
            if not valid:issues.append('input identity does not join frozen model/source/settings/input bytes')
            assert_compatible(interface,read_json('observed-interface.json'),compare_source=False)
        except (KeyError,TypeError,ValueError):issues.append('missing or invalid retained input/observed interface identity')
        encodings={row['item_id']:row for row in cell['observed_prompt_encodings']}
        for row in rows+partials:
            try:
                encoded=encodings[row['item_id']]
                if (row.get('checkpoint_id')!=expected_checkpoint or row.get('input_identity_sha256')!=identity.get('identity_sha256')
                        or row.get('input_sha256')!=expected_inputs or row.get('contract_id')!=contract['identity']
                        or row.get('settings_sha256')!=canonical_hash(settings) or row.get('checkpoint_interface_sha256')!=interface['interface_sha256']
                        or row.get('prompt_token_ids')!=encoded['token_ids'] or row.get('serialized_prompt')!=encoded['serialized_prompt']):
                    issues.append('raw response/partial cost does not join selected model/input/encoding')
            except (KeyError,TypeError):issues.append('malformed raw response/partial identity')
        complete=(summary.get('status')=='completed' and summary.get('record_count')==20 and len(rows)==20
            and summary.get('contract_id')==contract['identity'] and summary.get('checkpoint_id')==expected_checkpoint
            and summary.get('identity_sha256')==identity.get('identity_sha256') and summary.get('inputs_unchanged') is True and not issues)
        cells.append(dict(id=cell['id'],complete=complete,expected_checkpoint_id=expected_checkpoint,
            identity_sha256=identity.get('identity_sha256'),issues=issues,raw_records=len(rows),partial_events=len(partials)))
    return dict(complete=all(cell['complete'] for cell in cells),cells=cells,
        scope='Raw finalized and partial cost records joined to retained input identity, exact selected export and original prompt encoding')


def baseline_gate(budget):
    evidence=baseline().paths('instruct-thinking-off',budget,'run-01')['evidence']
    record=json.loads((evidence/'preparation.json').read_text())
    accepted=json.loads((evidence/'acceptance.json').read_text());baseline().verify_prepared(record)
    if (accepted.get('status')!='passed' or any(accepted.get('checks',{}).get(key) is not True for key in BASELINE_CHECKS)
            or accepted.get('actual_exit_code')!=0 or accepted.get('row')!='instruct-thinking-off'
            or accepted.get('budget')!=budget or record.get('row')!='instruct-thinking-off'
            or record.get('budget')!=budget or record.get('run_id')!='run-01'
            or accepted.get('local_model_binding')!=record.get('local_model_binding')):
        raise ValueError('Actual unchanged Instruct thinking-off baseline100 required for this cap')
    outputs=response_bindings(record)
    if (baseline().output_summary(record)!=accepted.get('evaluation') or not generation_identity(record)['complete']
            or response_bindings(record)!=outputs):
        raise ValueError('Original baseline retained rows/costs differ from accepted replay')
    return dict(preparation=_digest(str(evidence/'preparation.json')),acceptance=_digest(str(evidence/'acceptance.json')),
        model=record['local_model_binding'],cells=record['cells'],logical_panel_sha256=record['logical_panel_sha256'],outputs=outputs)


def selected_policy(group):
    paths(group,32,'run-00')
    locations=stages().paths('pilot',group,'run-01');evidence=locations['evidence'];policy=locations['pilot']/'policy'
    declaration=json.loads((evidence/'preparation.json').read_text())
    accepted=json.loads((evidence/'acceptance.json').read_text());stages().verify_prepared(declaration)
    invocations=accepted.get('invocations',[]);report=accepted.get('reports',{}).get('pilot',{})
    if (accepted.get('status')!='passed' or accepted.get('stage')!='pilot' or accepted.get('group')!=group
            or any(accepted.get('checks',{}).get(key) is not True for key in PILOT_CHECKS)
            or declaration.get('stage')!='pilot' or declaration.get('group')!=group or declaration.get('run_id')!='run-01'
            or declaration.get('recipe')!=stages().fixed_recipe('pilot')
            or declaration['recipe'].get('updates')!=16 or declaration['locations'].get('pilot')!=str(locations['pilot'])
            or len(invocations)!=1 or invocations[0].get('role')!='pilot'
            or invocations[0].get('status')!='completed' or invocations[0].get('actual_exit_code')!=0
            or report.get('status')!='completed' or report.get('committed_completed_updates')!=16
            or accepted.get('local_model_binding')!=declaration.get('local_model_binding')
            or report.get('local_source_hashes')!=declaration['local_model_binding']['files']):
        raise ValueError('Exact actual accepted16-update G4/G8 pilot required, never a selected replacement')
    if json.loads((locations['pilot']/'report.json').read_text())!=report:raise ValueError('Actual accepted pilot report bytes differ')
    files=artifact_hashes(policy);genealogy=json.loads((policy/'course-genealogy.json').read_text())
    export=accepted.get('exports',{}).get('pilot')
    if (not isinstance(export,dict) or export.get('path')!=str(policy) or export.get('files')!=files
            or export.get('completed_updates')!=16 or export!=stages().pilot_export_binding(declaration,report)):
        raise ValueError('Selected export is not the actual accepted committed16 policy bytes/state')
    if (not files or 'config.json' not in files or not any(name.endswith('.safetensors') for name in files)
            or 'adapter_config.json' in files or genealogy.get('kind')!='full-HF-model'
            or genealogy.get('objective')!='course-response-mean-GRPO'
            or genealogy.get('parent_local_source_hashes')!=declaration['local_model_binding']['files']
            or genealogy.get('upstream_revision_metadata')!=stages().REVISION
            or genealogy.get('checkpoint_interface')!=declaration['observed_geometry']['checkpoint_interface']
            or genealogy.get('template_sha256')!=declaration['observed_geometry']['prompt_contract']['template_sha256']):
        raise ValueError('Actual full RLVR export/genealogy/native interface required')
    return dict(path=str(policy),model=f'actual-native-rlvr-g{group}-pilot16-run-01',files=files,
        upstream_parent=declaration['local_model_binding'],checkpoint_interface=genealogy['checkpoint_interface'],genealogy=genealogy,
        selection='predeclared16-update run-01 final; no quality selection',
        preparation=_digest(str(evidence/'preparation.json')),acceptance=_digest(str(evidence/'acceptance.json')),
        report=_digest(str(locations['pilot']/'report.json')),accepted_export=export)


def fixed_inputs(budget):
    inputs=deepcopy(baseline().fixed_inputs())
    # Capture the actual interpreter/lock used by this owned adapter separately.
    inputs.update(interpreter=_digest(INTERPRETER,executable=True),environment_lock=_digest(ENVIRONMENT_LOCK))
    gate=baseline_gate(budget)
    inputs.update(original_baseline_preparation=gate['preparation'],original_baseline_acceptance=gate['acceptance'])
    inputs.update({'original_baseline_output/'+key:value for key,value in gate['outputs'].items()})
    return inputs


def fixed_argv(group,budget,run_id):
    paths(group,budget,run_id)
    return [INTERPRETER,str(ROOT/'scripts/run_native_rlvr_evaluation.py'),'--group-size',str(group),
        '--budget',str(budget),'--run-id',run_id,'--child']


def native_interface(policy,tokenizer,budget):
    actual_settings=baseline().settings('instruct-thinking-off',budget,tokenizer,'sample',SEEDS[0])
    frozen=freeze_local_contract(original_panel()['items'],actual_settings,tokenizer)['settings']['generation']['interface']
    trained=policy['checkpoint_interface']
    if (trained['tokenizer']!=frozen['tokenizer'] or trained['template_sha256']!=frozen['template_sha256']):
        raise ValueError('Actual trained native tokenizer/template differs from common baseline evaluation')
    gate=baseline_gate(budget)
    if any(policy['upstream_parent'].get(key)!=gate['model'].get(key) for key in ('path','model','revision','files')):
        raise ValueError('Pilot did not start at the exact common cached Instruct baseline')
    return frozen


def prepare(group,budget,run_id):
    locations=paths(group,budget,run_id)
    for path in locations.values():_new_target(str(path))
    evidence=locations['evidence'];evidence.mkdir(mode=0o700);sources=inputs=policy=None
    try:
        sources,inputs,policy,panel=source_bindings(),fixed_inputs(budget),selected_policy(group),original_panel()
        tokenizer=load_local_tokenizer(policy['path']);interface=native_interface(policy,tokenizer,budget)
        retain(evidence/'items.json',panel['items']);retain(evidence/'logical-panel.json',panel)
        cells=[];gate=baseline_gate(budget)
        for decoding,seed in [('sample',seed) for seed in SEEDS]+[('greedy',SEEDS[0])]:
            name=f'{decoding}-{seed}'
            contract=freeze_local_contract(panel['items'],baseline().settings('instruct-thinking-off',budget,tokenizer,decoding,seed),tokenizer)
            contract['panel_binding']=dict(logical_contract_sha256=panel['logical_contract_sha256'],original_items_sha256=panel['items_sha256'],
                row='instruct-thinking-off',prompt_policy='Original prompt text unchanged; chat adds only the declared serialization')
            contract['identity']=canonical_hash({key:value for key,value in contract.items() if key!='identity'})
            baseline_cell=next(row for row in gate['cells'] if row['id']==name)
            if contract['identity']!=baseline_cell['contract_id']:raise ValueError('Exact common baseline decoding/interface contract differs')
            encodings=[]
            for item in panel['items']:
                text,ids=serialize_prompt(tokenizer,item,contract['settings'])
                if len(ids)+budget>CONTEXT:raise ValueError('Original prompt plus evaluation cap exceeds context')
                encodings.append(dict(item_id=item['id'],serialized_prompt=text,token_ids=ids))
            if encodings!=baseline_cell['observed_prompt_encodings']:raise ValueError('Common original prompt token IDs differ from baseline')
            retain(evidence/(name+'-contract.json'),contract)
            cells.append(dict(id=name,mode=decoding,seed=seed,contract_id=contract['identity'],
                contract_path=str(evidence/(name+'-contract.json')),output=str(locations['output']/name),observed_prompt_encodings=encodings))
        for key,name in (('items','items.json'),('logical_panel','logical-panel.json')):inputs[key]=_digest(str(evidence/name))
        for cell in cells:inputs['contract/'+cell['id']]=_digest(cell['contract_path'])
        if sources!=source_bindings() or policy!=selected_policy(group) or any(inputs.get(key)!=value for key,value in fixed_inputs(budget).items()):
            raise ValueError('Source/policy/input bytes changed during tokenizer-only preparation')
        record=dict(schema='dongxi-fixed-native-rlvr-evaluation-v1',group=group,budget=budget,run_id=run_id,
            utc=datetime.now(timezone.utc).isoformat(),evidence=str(evidence),output=str(locations['output']),
            source_bindings=sources,input_bindings=inputs,local_model_binding=policy,checkpoint_interface=interface,
            logical_panel_sha256=canonical_hash(panel),cells=cells,argv=fixed_argv(group,budget,run_id),execution_requested=False,
            limits=fixed_limits(budget),
            boundaries=['Original20 items and four original sampled seeds plus greedy; no best-of-N or quality selection',
                'Native Instruct thinking-off template, original prompt IDs and baseline contract identities verified before weights',
                'Original train/RLVR problem-overlap rows retained as diagnostics and excluded from heldout summaries',
                'Run-generation retained stops/errors/raw continuations/behavior probabilities and entered/completed forward costs',
                'Policy export hashes captured fresh after actual16 acceptance; no runner four-item diagnostics substituted',
                'Full-prefix no-KV generation;900 external seconds and25GiB sampled reserve are not logical/physical quotas',
                'Each original100xcap128 journal reader allows256MiB retained partial events (quadratic token/logp prefixes); ordinary output files64MiB',
                'Cap32/128 are separate rows; raw thinking output is graded unchanged; no automatic retry'])
        record['preparation_sha256']=canonical_hash(record);retain(evidence/'preparation.json',record);return record
    except BaseException as error:
        retain(evidence/'preparation-failure.json',dict(status='failed',type=type(error).__name__,message=str(error),
            source_bindings=sources,input_bindings=inputs,selected_policy=policy,scope='No CUDA child or model weights loaded'));raise


def verify_prepared(record):
    locations=paths(record['group'],record['budget'],record['run_id'])
    if (record.get('schema')!='dongxi-fixed-native-rlvr-evaluation-v1'
            or record.get('preparation_sha256')!=canonical_hash({key:value for key,value in record.items() if key!='preparation_sha256'})
            or record['argv']!=fixed_argv(record['group'],record['budget'],record['run_id'])
            or record.get('limits')!=fixed_limits(record['budget'])
            or any(record[key]!=str(value) for key,value in locations.items())):raise ValueError('Closed RLVR evaluation command/selector changed')
    if source_bindings()!=record['source_bindings']:raise ValueError('Bound RLVR evaluation source changed')
    if selected_policy(record['group'])!=record['local_model_binding']:raise ValueError('Accepted policy/export bytes changed')
    panel=original_panel();evidence=locations['evidence']
    if canonical_hash(panel)!=record['logical_panel_sha256'] or json.loads((evidence/'items.json').read_text())!=panel['items']:
        raise ValueError('Original annotated logical20 panel changed')
    expected=deepcopy(fixed_inputs(record['budget']))
    for key,name in (('items','items.json'),('logical_panel','logical-panel.json')):expected[key]=_digest(str(evidence/name))
    expected_cells=[(f'sample-{seed}','sample',seed) for seed in SEEDS]+[(f'greedy-{SEEDS[0]}','greedy',SEEDS[0])]
    if [(cell['id'],cell['mode'],cell['seed']) for cell in record['cells']]!=expected_cells:raise ValueError('Original evaluation cell coverage changed')
    gate=baseline_gate(record['budget'])
    for cell in record['cells']:
        path=evidence/(cell['id']+'-contract.json')
        if cell['contract_path']!=str(path) or cell['output']!=str(locations['output']/cell['id']):raise ValueError('Fixed evaluation cell path changed')
        expected['contract/'+cell['id']]=_digest(str(path));contract=json.loads(path.read_text());replay_records(panel['items'],[],contract)
        original=next(row for row in gate['cells'] if row['id']==cell['id'])
        if contract['identity']!=cell['contract_id'] or cell['contract_id']!=original['contract_id'] or cell['observed_prompt_encodings']!=original['observed_prompt_encodings']:
            raise ValueError('Exact common original cell/interface/encoding binding changed')
    if expected!=record['input_bindings']:raise ValueError('Bound original input/contract bytes changed')


def summary(record):
    identity=generation_identity(record) # bound event/raw readers before legacy replay
    value=baseline().output_summary(record);value['generation_identity']=identity;return value


def child(group,budget,run_id):
    evidence=paths(group,budget,run_id)['evidence'];record=json.loads((evidence/'preparation.json').read_text())
    launch=json.loads((evidence/'launch.json').read_text())
    if launch.get('preparation_sha256')!=record.get('preparation_sha256') or launch.get('argv')!=record['argv']:
        raise ValueError('Independently retained closed RLVR evaluation launch required')
    verify_prepared(record)
    for name in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE'):os.environ[name]='1'
    output=_new_target(record['output']);output.mkdir(mode=0o700)
    try:
        for cell in record['cells']:
            verify_prepared(record)
            result=run_generation(root=ROOT,items_path=evidence/'items.json',contract_path=cell['contract_path'],
                checkpoint=record['local_model_binding']['path'],output=cell['output'],environment_lock=ENVIRONMENT_LOCK,allow_cuda=True)
            verify_prepared(record)
            if result['status']!='completed' or result['record_count']!=20:raise RuntimeError('Actual original twenty-item cell incomplete; retained partial costs remain')
        retain(output/'child-summary.json',summary(record));return 0
    except BaseException as error:
        retain(output/'child-failure.json',dict(status='failed',type=type(error).__name__,message=str(error)));raise


def execute(record,declaration):
    if type(declaration) is not str or not 12<=len(declaration)<=4096:raise ValueError('Bounded goal-authorized RLVR evaluation declaration required')
    evidence=Path(record['evidence']);supervision=None
    try:
        verify_prepared(record);_new_target(record['output'])
        retain(evidence/'launch.json',dict(argv=record['argv'],operator_declaration=declaration,preparation_sha256=record['preparation_sha256']))
        supervision=_supervise(record['argv'],evidence/'supervision',native=True,seconds=900,probe=_native_probe,operator_declaration=declaration)
        retain(evidence/'returned-supervision.json',supervision);verify_prepared(record)
        actual=summary(record)
        checks=dict(actual_exit0=supervision['status']=='completed' and supervision['actual_exit_code']==0,
            all_five_cells_completed=actual['generation_identity']['complete'] and all(cell['summary'] is not None and cell['summary']['status']=='completed'
                and cell['records']==20 and not cell['missing_item_ids'] for cell in actual['cells']),
            all_original_responses=actual['missing_responses']==0,no_unreadable_jsonl=not actual['unreadable_jsonl_lines'])
        retain(evidence/'closing-bindings.json',dict(status='unchanged',source_bindings=record['source_bindings'],
            input_bindings=record['input_bindings'],local_model_binding=record['local_model_binding']))
        accepted=dict(status='passed' if all(checks.values()) else 'failed',group=record['group'],budget=record['budget'],
            checks=checks,local_model_binding=record['local_model_binding'],evaluation=actual,supervision=supervision,boundaries=record['boundaries'])
        retain(evidence/'acceptance.json',accepted)
        if accepted['status']!='passed':raise RuntimeError('Actual RLVR evaluation incomplete; every failed/capped record and known cost retained')
        print(json.dumps(dict(status='passed',group=record['group'],budget=record['budget'],evidence=str(evidence))));return 0
    except BaseException as error:
        retain(evidence/'failure.json',dict(status='failed',type=type(error).__name__,message=str(error),
            supervision_retained=supervision is not None,scope='No automatic retry/cap/model change; partial outputs remain'));raise


def compare_cap(budget,run_id):
    paths(4,budget,run_id);sources=source_bindings();gate=baseline_gate(budget);original=json.loads(Path(gate['acceptance']['path']).read_text())
    rows_by_policy={'unchanged-instruct':original['evaluation']['rows']};bindings={'unchanged-instruct':gate}
    for group in GROUPS:
        evidence=paths(group,budget,run_id)['evidence'];record=json.loads((evidence/'preparation.json').read_text());verify_prepared(record)
        accepted=json.loads((evidence/'acceptance.json').read_text())
        supervision=accepted.get('supervision',{})
        if (accepted.get('status')!='passed' or accepted.get('group')!=group or accepted.get('budget')!=budget
                or supervision.get('status')!='completed' or supervision.get('actual_exit_code')!=0
                or json.loads((evidence/'returned-supervision.json').read_text())!=supervision
                or any(accepted.get('checks',{}).get(key) is not True for key in EVALUATION_CHECKS)
                or accepted.get('local_model_binding')!=record['local_model_binding']):
            raise ValueError('Both complete actual RLVR evaluations required for common20 comparison')
        actual=summary(record)
        if actual!=accepted['evaluation'] or not actual['generation_identity']['complete']:
            raise ValueError('Actual retained evaluation model/input identities or rows/costs differ from accepted output replay')
        label=f'rlvr-g{group}';rows_by_policy[label]=actual['rows']
        bindings[label]=dict(preparation=_digest(str(evidence/'preparation.json')),acceptance=_digest(str(evidence/'acceptance.json')),
            supervision_receipt=_digest(str(evidence/'returned-supervision.json')),
            model=record['local_model_binding'],sources=record['source_bindings'],inputs=record['input_bindings'],outputs=response_bindings(record))
    cells=[];expected_items={item['id'] for item in original_panel()['items']}
    def metrics(rows):
        return dict(records=len(rows),correct=sum(row['correct'] for row in rows),
            natural_stops=sum(row['natural_termination'] for row in rows),truncated=sum(row['truncated'] for row in rows),
            statuses=dict(Counter(row['status'] for row in rows)),costs={key:sum(row['cost'][key] for row in rows)
                for key in ('generation_tokens','attempted_forward_tokens','model_forward_tokens','attempted_forward_calls','forward_calls','wall_seconds')})
    for mode,seed in [('sample',seed) for seed in SEEDS]+[('greedy',SEEDS[0])]:
        cell=f'{mode}-{seed}';aligned={label:[row for row in rows if row['cell_id']==cell] for label,rows in rows_by_policy.items()}
        if any(len(rows)!=20 or {row['item_id'] for row in rows}!=expected_items for rows in aligned.values()):
            raise ValueError('Exact common20 item/seed coverage required, not four diagnostic prompts')
        if len({row['contract_id'] for rows in aligned.values() for row in rows})!=1:raise ValueError('Common original contract identities differ')
        heldout={label:[row for row in rows if row['panel_annotation']['campaign_role']=='heldout-controlled-slice'
            and not row['panel_annotation']['rlvr_train_problem_overlap']] for label,rows in aligned.items()}
        cells.append(dict(id=cell,contract_id=aligned['unchanged-instruct'][0]['contract_id'],
            all_original={label:metrics(rows) for label,rows in aligned.items()},heldout={label:metrics(rows) for label,rows in heldout.items()},
            paired_heldout={label:paired_group_bootstrap(heldout['unchanged-instruct'],heldout[label],draws=2000,seed=1010,metric='correct')
                for label in ('rlvr-g4','rlvr-g8')},
            paired_scope='Each seed/greedy contract remains separate; small authored source clusters are descriptive, not population superiority'))
    report=dict(schema='dongxi-common20-native-rlvr-comparison-v1',budget=budget,run_id=run_id,
        logical_panel=original_panel(),bindings=bindings,cells=cells,
        all_original={label:metrics(rows) for label,rows in rows_by_policy.items()},
        heldout={label:metrics([row for row in rows if row['panel_annotation']['campaign_role']=='heldout-controlled-slice'
            and not row['panel_annotation']['rlvr_train_problem_overlap']]) for label,rows in rows_by_policy.items()},
        planned_attempts_per_policy=100,known_overlap_item_ids=[item['id'] for item in original_panel()['items'] if item['rlvr_train_problem_overlap']],
        boundaries=['Every original20 row, failure and cap retained; known train/RLVR overlap excluded from heldout statistics',
            'Cap budgets are never pooled; sampled seeds and greedy contract panels kept distinct',
            'No quality selection, rerun, reward-only shortcut or four-item runner diagnostic substitute',
            'Exact actual model/export/input/template/source bindings retained separately for unchanged Instruct/G4/G8'])
    if source_bindings()!=sources or baseline_gate(budget)!=gate:raise ValueError('Comparison source/baseline bytes changed')
    for group in GROUPS:
        evidence=paths(group,budget,run_id)['evidence'];record=json.loads((evidence/'preparation.json').read_text());verify_prepared(record)
        bound=bindings[f'rlvr-g{group}']
        if (response_bindings(record)!=bound['outputs'] or _digest(str(evidence/'acceptance.json'))!=bound['acceptance']
                or _digest(str(evidence/'returned-supervision.json'))!=bound['supervision_receipt']
                or _digest(str(evidence/'preparation.json'))!=bound['preparation']):raise ValueError('Comparison accepted rows/metadata changed')
    path=ROOT/'experiments/reports'/f'native-rlvr-common20-cap{budget}-20261005-{run_id}.json'
    retain(path,report);print(json.dumps(dict(status='compared',budget=budget,path=str(path))));return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group-size',type=int,choices=GROUPS);parser.add_argument('--budget',type=int,choices=BUDGETS,required=True)
    parser.add_argument('--run-id',required=True);parser.add_argument('--execute',action='store_true');parser.add_argument('--operator-declaration')
    parser.add_argument('--child',action='store_true',help=argparse.SUPPRESS);parser.add_argument('--compare',action='store_true')
    args=parser.parse_args(argv)
    if args.compare:
        if args.group_size or args.execute or args.child or args.operator_declaration:parser.error('CPU comparison takes only budget and run-id')
        return compare_cap(args.budget,args.run_id)
    if args.group_size is None:parser.error('Fixed G4/G8 selector required')
    if args.child:
        if args.execute or args.operator_declaration:parser.error('Internal child has a closed prepared route')
        return child(args.group_size,args.budget,args.run_id)
    if args.execute and (not args.operator_declaration or not 12<=len(args.operator_declaration)<=4096):parser.error('Bounded goal-authorized declaration required')
    record=prepare(args.group_size,args.budget,args.run_id)
    if args.execute:return execute(record,args.operator_declaration)
    print(json.dumps(dict(status='prepared-not-executed',evidence=record['evidence'])));return 0


if __name__=='__main__':raise SystemExit(main())
