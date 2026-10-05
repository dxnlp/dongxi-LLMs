#!/usr/bin/env python3
"""Closed common full400/chosen100/DPO100 evaluation; default prepares only.

Each producer has one900-second externally owned child and a sampled25GiB
reserve. Original4 validation pairs/4 independent location prompts,120 assistant
publication items and20 annotated reasoning items remain separate populations.
Retention uses common greedy64/custom saved-template settings, not the original
reasoning32/128 publication experiment. No arbitrary checkpoint or acquisition.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dongxi_llms.native_profile_supervisor import _digest,_native_probe,_new_target,_supervise
from dongxi_llms.run_identity import artifact_hashes,canonical_hash,TOKENIZER_PATTERNS
from dongxi_llms.reasoning_generation import (SOURCE_FILES as GENERATION_SOURCES,
    freeze_local_contract,load_local_tokenizer,run_generation,serialize_prompt)
from dongxi_llms.reasoning_evaluation import replay_records
from dongxi_llms.staged_campaign import evaluation_contracts

GIB=1024**3
PYTHON='/home/dongxi/dgx-spark-dongxi/.venv/bin/python'
LOCK='/home/dongxi/dgx-spark-dongxi/uv.lock'
ROLES=('unchanged','chosen100','dpo100')
PANELS=('location','assistant','reasoning')
COUNTS=dict(location=4,assistant=120,reasoning=20)
SECONDS=900
CAP=64
CONTEXT=512
JSONL_BYTES=16*1024**2
EVENT_BYTES=64*1024**2
RECEIPT_BYTES=16*1024**2
SOURCES=tuple(dict.fromkeys((*GENERATION_SOURCES,
    'scripts/run_native_preference_evaluation.py','tests/test_native_preference_evaluation.py',
    'scripts/run_native_dpo_stages.py','scripts/run_native_chosen_stages.py',
    'scripts/run_chapter11_spark_dpo.py','src/dongxi_llms/dpo_lab.py',
    'src/dongxi_llms/native_profile_supervisor.py','src/dongxi_llms/campaign_supervisor.py',
    'src/dongxi_llms/staged_campaign.py')))


def retain(path,value):
    with Path(path).open('x',encoding='utf-8') as handle:
        json.dump(value,handle,indent=2,ensure_ascii=False,allow_nan=False)
        handle.write('\n');handle.flush();os.fsync(handle.fileno())


def append(handle,value):
    handle.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
    handle.flush();os.fsync(handle.fileno())


def native():
    spec=importlib.util.spec_from_file_location('fixed_preference_native',ROOT/'scripts/run_chapter11_spark_dpo.py')
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def parent_adapter():
    spec=importlib.util.spec_from_file_location('fixed_preference_parent',ROOT/'scripts/run_native_dpo_stages.py')
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def producer_adapter(kind):
    if kind=='dpo':return parent_adapter()
    if kind!='chosen':raise ValueError('Only the two fixed preference adapters are supported')
    spec=importlib.util.spec_from_file_location('fixed_preference_chosen',ROOT/'scripts/run_native_chosen_stages.py')
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def paths(run_id):
    if type(run_id) is not str or re.fullmatch(r'run-[0-9]{2}',run_id) is None:
        raise ValueError('Only a fixed run-NN evidence identifier is supported')
    stem='native-preference-evaluation-20261005-'+run_id
    return dict(evidence=ROOT/'experiments/reports'/stem,output=ROOT/'outputs'/stem)


def source_bindings():return {name:_digest(str(ROOT/name)) for name in SOURCES}


def input_bindings():
    files={name:ROOT/'fixtures/chapter11'/(name+'.jsonl') for name in ('train','validation','evaluation')}
    files.update(protocol=ROOT/'fixtures/matched-chosen-sft/protocol.json',
        template=ROOT/'experiments/data/instruction_interface_v1.jinja',
        math_items=ROOT/'fixtures/reasoning-controls/math_items.json',
        assistant_card=ROOT/'experiments/data/instruction-interface-v1-data-card.json',
        interpreter=Path(PYTHON),environment_lock=Path(LOCK))
    files.update({f'assistant_{name}':ROOT/'outputs/course-sft-interface-v1'/(name+'.jsonl')
        for name in ('train','dev','test')})
    return {key:_digest(str(path),executable=key=='interpreter') for key,path in files.items()}


def producer_bindings():
    parent=parent_adapter().parent_binding();fixed_inputs=input_bindings()
    result={'unchanged':dict(path=parent['path'],files=parent['files'],parent=parent)}
    for role,kind in (('chosen100','chosen'),('dpo100','dpo')):
        directory=ROOT/'experiments/reports'/f'native-{kind}-pilot-20261005-run-01'
        acceptance=json.loads((directory/'acceptance.json').read_text())
        required=('all_actual_children_completed','completed100','exactly100_metrics','clean_journals')
        if kind=='chosen':required+=('unchanged_reference',)
        if (acceptance.get('status')!='passed' or acceptance.get('parent_binding')!=parent
                or not acceptance.get('checks') or not all(value is True for value in acceptance['checks'].values())
                or any(acceptance.get('checks',{}).get(key) is not True for key in required)):
            raise ValueError('Actual same-parent accepted100 policy required: '+role)
        invocations=acceptance.get('invocations',[])
        if (len(invocations)!=1 or invocations[0].get('role')!='pilot'
                or invocations[0].get('result',{}).get('status')!='completed'
                or invocations[0]['result'].get('actual_exit_code')!=0):
            raise ValueError('Actual pilot100 exit0 receipt required: '+role)
        prepared=json.loads((directory/'preparation.json').read_text());adapter=producer_adapter(kind)
        adapter.verify_prepared(prepared)
        roles=('train','validation','evaluation','interpreter','environment_lock')+(('protocol',) if kind=='chosen' else ())
        if (prepared.get('mode')!='pilot' or prepared.get('run_id')!='run-01'
                or prepared.get('limits',{}).get('updates')!=100 or prepared.get('parent_binding')!=parent
                or any(prepared['input_bindings'].get(key)!=fixed_inputs[key] for key in roles)
                or adapter.observed_geometry(parent,prepared['input_bindings'])!=prepared['geometry']):
            raise ValueError('Actual pilot source/input/interface/encoded geometry changed: '+role)
        exported=acceptance['exports']['pilot']
        expected=ROOT/'outputs'/f'native-{kind}-pilot-20261005-run-01-pilot'/'policy'
        actual=json.loads((expected.parent/'result.json').read_text())
        metrics=[json.loads(line) for line in (expected.parent/'metrics.jsonl').read_text().splitlines() if line.strip()]
        genealogy=json.loads((expected/'course-genealogy.json').read_text())
        if (exported['path']!=str(expected) or artifact_hashes(expected)!=exported['files']
                or acceptance.get('results',{}).get('pilot')!=actual
                or [row.get('update') for row in metrics]!=list(range(1,101))
                or actual.get('reference_has_gradients') is not False
                or any(actual.get(key,{}).get(field)!=[] for key in ('cumulative_work_ledger','snapshot_io_ledger')
                    for field in ('open_tickets','failed_tickets'))
                or genealogy.get('parent_checkpoint')!=parent['path']
                or genealogy.get('parent_checkpoint_sha256')!=parent['files']
                or genealogy.get('checkpoint_interface')!=prepared['geometry']['interface']
                or genealogy.get('completed_recovery')!=actual.get('completed_recovery')
                or genealogy.get('kind')!=('full-HF-chosen-policy' if kind=='chosen' else 'full-HF-DPO-policy')
                or genealogy.get('completed_recovery',{}).get('completed_updates')!=100):
            raise ValueError('Actual final100 export/genealogy differs: '+role)
        result[role]=dict(path=str(expected),files=exported['files'],genealogy=genealogy,
            acceptance=_digest(str(directory/'acceptance.json')),
            preparation=_digest(str(directory/'preparation.json')),
            actual_result=_digest(str(expected.parent/'result.json')),
            actual_metrics=_digest(str(expected.parent/'metrics.jsonl')),parent=parent,
            continuity=dict(interface=prepared['geometry']['interface'],
                encoded_validation_sha256=prepared['geometry']['encoded_validation_sha256'],
                tokenizer_snapshot=prepared['geometry']['tokenizer_snapshot']))
    return result


def panels(inputs):
    protocol=json.loads(Path(inputs['protocol']['path']).read_text())
    rows={name:native().load_rows(inputs[name]['path']) for name in ('train','validation','evaluation')}
    if [len(rows[key]) for key in ('train','validation','evaluation')]!=[8,4,4]:
        raise ValueError('Original8/4/4 location fixtures required; no invented8-item test panel')
    if set(protocol['groups'])!={row['id'] for split in rows.values() for row in split}:
        raise ValueError('Original scenario-group sidecar does not bind all location rows')
    location=[dict(id=row['id'],source_group=protocol['groups'][row['id']],task='location',split='publication',
        prompt=row['prompt'][-1]['content'],messages=deepcopy(row['prompt']),kind='text',
        reference=row['expected'],format_policy='any',extraction='whole') for row in rows['evaluation']]
    card=json.loads(Path(inputs['assistant_card']['path']).read_text());splits={}
    for name,count,groups in (('train',240,80),('dev',60,20),('test',120,40)):
        binding=inputs['assistant_'+name];split=native().load_rows(binding['path']);splits[name]=split
        if (binding['sha256']!=card['splits'][name]['sha256'] or len(split)!=count
                or len({row['group'] for row in split})!=groups):
            raise ValueError('Original assistant data-card population/bytes changed')
    for a,b in (('train','dev'),('train','test'),('dev','test')):
        if {r['group'] for r in splits[a]} & {r['group'] for r in splits[b]}:
            raise ValueError('Assistant source groups overlap')
    assistant=[dict(id=row['id'],source_group=row['group'],task=row['family'],split='publication',
        prompt=row['messages'][-2]['content'],messages=deepcopy(row['messages'][:-1]),kind='text',
        reference=row['messages'][-1]['content'],format_policy='any',extraction='whole') for row in splits['test']]
    reasoning=deepcopy(evaluation_contracts(ROOT)['reasoning-panel-v1']['items'])
    if len(reasoning)!=20:raise ValueError('Original annotated twenty-item retention panel required')
    return dict(location=location,assistant=assistant,reasoning=reasoning),rows


def generation_settings(tokenizer,template):
    stops=[tokenizer.eos_token_id];end=tokenizer.convert_tokens_to_ids('<|im_end|>')
    if end is None or end==tokenizer.unk_token_id:raise ValueError('Real original turn stop required')
    return dict(template_id='original-selected-full400-preference-retention',thinking_mode='template-default',
        decoding=dict(mode='greedy',seed=1010,temperature=1.,top_k=None,top_p=1.),
        stopping=dict(eos_token_ids=stops,turn_stop_token_ids=[] if end in stops else [end],
            pad_token_id=tokenizer.pad_token_id),max_new_tokens=CAP,
        generation=dict(input_mode='chat',template=template,context_window=CONTEXT,samples=1,
            device='cuda',dtype='bfloat16',add_special_tokens=False,max_run_seconds=SECONDS,
            scoring_text='decode_without_terminal_stop'))


def command(role,run_id):
    paths(run_id)
    if role not in ROLES:raise ValueError('Only the three fixed accepted producers are supported')
    return [PYTHON,str(ROOT/'scripts/run_native_preference_evaluation.py'),'--run-id',run_id,'--child',role]


def limits(encoded):
    return dict(seconds_each=SECONDS,reserve_bytes=25*GIB,context=CONTEXT,output_cap=CAP,
        validation_pairs=4,validation_policy_calls=8,validation_reference_calls=8,
        validation_positions_each=sum(len(branch[0])-1 for pair in encoded for branch in pair),
        planned_responses_each=sum(COUNTS.values()),maximum_emitted_tokens_each=sum(COUNTS.values())*CAP,
        retained_reader_bytes=dict(response_or_pair_jsonl=JSONL_BYTES,event_jsonl=EVENT_BYTES,input_identity=RECEIPT_BYTES))


def prepare(run_id):
    places=paths(run_id)
    for path in places.values():_new_target(str(path))
    evidence=places['evidence'];evidence.mkdir(mode=0o700)
    try:
        sources,inputs,producers=source_bindings(),input_bindings(),producer_bindings()
        items,rows=panels(inputs);dpo=native();reference=producers['unchanged']['path']
        template=dpo.restore_saved_template(reference)
        if template!=Path(inputs['template']['path']).read_text():raise ValueError('Original selected-parent custom template required')
        tokenizer=load_local_tokenizer(reference);settings=generation_settings(tokenizer,template)
        contracts={};interfaces={}
        for role in ROLES:
            proposed=load_local_tokenizer(producers[role]['path'])
            observed,_=dpo.verify_parent_tokenizer(Path(reference),tokenizer,proposed,
                tokenizer_id=parent_adapter().MODEL,tokenizer_revision=parent_adapter().REVISION)
            # Check actual exported saved semantics, not only the adopted parent serialization.
            actual,_=dpo.verify_parent_tokenizer(Path(producers[role]['path']),
                load_local_tokenizer(producers[role]['path']),proposed,
                tokenizer_id=parent_adapter().MODEL,tokenizer_revision=parent_adapter().REVISION)
            if observed!=actual:raise ValueError('Producer saved template/tokenizer differs from selected parent')
            if role!='unchanged' and actual!=producers[role]['continuity']['interface']:
                raise ValueError('Producer accepted training interface differs from evaluation')
            interfaces[role]=actual
        encoded=[dpo.encode_pair(tokenizer,row,CONTEXT) for row in rows['validation']]
        for role in ('chosen100','dpo100'):
            if dpo.state_digest(encoded)!=producers[role]['continuity']['encoded_validation_sha256']:
                raise ValueError('Producer accepted validation masks differ from evaluation')
        grouped=[[dict(row,group=json.loads(Path(inputs['protocol']['path']).read_text())['groups'][row['id']])
            for row in rows[name]] for name in ('train','validation','evaluation')]
        encoded_train=[dpo.encode_pair(tokenizer,row,CONTEXT) for row in rows['train']]
        prefixes=[tokenizer.apply_chat_template(row['prompt'],tokenize=True,
            add_generation_prompt=True,enable_thinking=False,return_dict=False) for row in rows['evaluation']]
        split=dpo.check_split_interfaces(tokenizer,grouped,[encoded_train,encoded,prefixes])
        for panel in PANELS:
            retain(evidence/(panel+'-items.json'),items[panel])
            contract=freeze_local_contract(items[panel],settings,tokenizer)
            for item in items[panel]:
                _,ids=serialize_prompt(tokenizer,item,contract['settings'])
                if len(ids)+CAP>CONTEXT:raise ValueError('Original retention prefix plus64 must fit without truncation')
            retain(evidence/(panel+'-contract.json'),contract);contracts[panel]=contract
            inputs[panel+'/items']=_digest(str(evidence/(panel+'-items.json')))
            inputs[panel+'/contract']=_digest(str(evidence/(panel+'-contract.json')))
        record=dict(schema='dongxi-fixed-native-preference-evaluation-v1',run_id=run_id,
            utc=datetime.now(timezone.utc).isoformat(),locations={k:str(v) for k,v in places.items()},
            source_bindings=sources,input_bindings=inputs,producers=producers,interfaces=interfaces,
            encoded_validation=encoded,validation_rows=rows['validation'],scenario_groups=json.loads(Path(inputs['protocol']['path']).read_text())['groups'],
            split_evidence=split,
            contracts={key:value['identity'] for key,value in contracts.items()},
            commands={role:command(role,run_id) for role in ROLES},
            limits=limits(encoded),
            precision='Likelihood: FP32 policy/reference with BF16 CUDA autocast/native single-shift masks; generation: all three BF16 loaded policies, common greedy64',
            scope='Four separate frozen slices. No history/checkpoint/quality selection, universal capability score, rationale faithfulness or original32/128 reasoning-panel completion.')
        record['preparation_sha256']=canonical_hash(record)
        retain(evidence/'preparation.json',record);verify_prepared(record);return record
    except BaseException as error:
        retain(evidence/'preparation-failure.json',dict(type=type(error).__name__,message=str(error)));raise


def verify_prepared(record):
    if (record.get('preparation_sha256')!=canonical_hash({k:v for k,v in record.items() if k!='preparation_sha256'})
            or record['locations']!={k:str(v) for k,v in paths(record['run_id']).items()}
            or record.get('schema')!='dongxi-fixed-native-preference-evaluation-v1'
            or record['commands']!={role:command(role,record['run_id']) for role in ROLES}
            or record['limits']!=limits(record['encoded_validation'])):
        raise ValueError('Closed preparation/selector/command changed')
    if source_bindings()!=record['source_bindings'] or producer_bindings()!=record['producers']:
        raise ValueError('Accepted producer/reference/source bytes changed')
    current=input_bindings()
    if any(record['input_bindings'].get(key)!=binding for key,binding in current.items()):
        raise ValueError('Original fixed input role changed')
    for key,binding in record['input_bindings'].items():
        if _digest(binding['path'],executable=key=='interpreter')!=binding:raise ValueError('Frozen input/contract bytes changed')
    evidence=paths(record['run_id'])['evidence'];tokenizer=load_local_tokenizer(record['producers']['unchanged']['path'])
    dpo=native();interface,_=dpo.verify_parent_tokenizer(Path(record['producers']['unchanged']['path']),tokenizer,tokenizer,
        tokenizer_id=parent_adapter().MODEL,tokenizer_revision=parent_adapter().REVISION)
    original,rows=panels(current)
    groups=json.loads(Path(current['protocol']['path']).read_text())['groups']
    encoded=[dpo.encode_pair(tokenizer,row,CONTEXT) for row in rows['validation']]
    if (canonical_hash(encoded)!=canonical_hash(record['encoded_validation']) or rows['validation']!=record['validation_rows']
            or groups!=record['scenario_groups'] or any(value!=interface for value in record['interfaces'].values())
            or any(dpo.state_digest(encoded)!=record['producers'][role]['continuity']['encoded_validation_sha256']
                for role in ('chosen100','dpo100'))):
        raise ValueError('Original native validation IDs/masks/interface changed')
    for panel in PANELS:
        items=json.loads((evidence/(panel+'-items.json')).read_text());contract=json.loads((evidence/(panel+'-contract.json')).read_text())
        expected=freeze_local_contract(items,generation_settings(tokenizer,tokenizer.chat_template),tokenizer)
        if (items!=original[panel] or len(items)!=COUNTS[panel] or contract!=expected or record['contracts'][panel]!=contract['identity']):
            raise ValueError('Original fixed greedy64 contract changed: '+panel)


def pair_scores(record,root,guard):
    import torch
    from transformers import AutoModelForCausalLM
    role=root.name;dpo=native();cost={name:dict(attempted_calls=0,forward_calls=0,attempted_positions=0,forward_positions=0)
        for name in ('policy','reference')}
    began=time.monotonic();guard()
    model=AutoModelForCausalLM.from_pretrained(record['producers'][role]['path'],local_files_only=True,
        trust_remote_code=False,torch_dtype=torch.float32,attn_implementation='sdpa').cuda().eval()
    guard();reference=AutoModelForCausalLM.from_pretrained(record['producers']['unchanged']['path'],local_files_only=True,
        trust_remote_code=False,torch_dtype=torch.float32,attn_implementation='sdpa').cuda().eval().requires_grad_(False)
    dpo.effective_model_contract(model);dpo.effective_model_contract(reference)
    def wrapper(network,kind):
        def forward(ids,attention):
            guard();row=cost[kind];row['attempted_calls']+=1;row['attempted_positions']+=ids.numel()
            retain_cost();result=network(input_ids=ids,attention_mask=attention,use_cache=False)
            torch.cuda.synchronize();row['forward_calls']+=1;row['forward_positions']+=ids.numel();retain_cost();return result
        return forward
    def retain_cost():
        with (root/'pair-cost-events.jsonl').open('a',encoding='utf-8') as handle:
            append(handle,dict(cost=cost,wall_seconds=time.monotonic()-began))
    try:
        with (root/'pair-scores.jsonl').open('x') as handle,torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for row,pair in zip(record['validation_rows'],record['encoded_validation']):
                branches=[dpo.collate([pair],branch,record['interfaces'][role]['tokenizer']['special_ids']['pad_token_id'],'cuda') for branch in (0,1)]
                loss,observed=dpo.model_pair_loss(wrapper(model,'policy'),wrapper(reference,'reference'),*branches,beta=.1)
                result=dict(id=row['id'],source_group=record['scenario_groups'][row['id']],
                    chosen_logp=float(observed['chosen_logp'][0]),rejected_logp=float(observed['rejected_logp'][0]),
                    reference_relative_margin=float(observed['margin'][0]),
                    reference_relative_logp_margin=float(observed['margin'][0])/.1,beta=.1,loss=float(loss),
                    chosen_targets=sum(pair[0][1][1:]),rejected_targets=sum(pair[1][1][1:]),
                    mask_scope='Exact original body+terminal marker+separator; one causal shift')
                if any(not math.isfinite(result[key]) for key in ('chosen_logp','rejected_logp','reference_relative_margin','reference_relative_logp_margin','loss')):
                    raise ValueError('Nonfinite actual pair likelihood/margin')
                append(handle,result)
        retain(root/'pair-summary.json',dict(status='completed',pairs=4,cost=cost,wall_seconds=time.monotonic()-began,
            reference_has_gradients=any(p.grad is not None for p in reference.parameters()),
            scope='Forward geometry, not FLOPs or allocation quota; model loading and scalar grading outside calls'))
    finally:
        del model,reference;gc.collect();torch.cuda.empty_cache()


def child(run_id,role):
    if role not in ROLES:raise ValueError('Unknown fixed producer')
    places=paths(run_id);evidence=places['evidence'];record=json.loads((evidence/'preparation.json').read_text())
    launch=json.loads((evidence/(role+'-launch.json')).read_text())
    if launch.get('argv')!=record['commands'][role] or launch.get('preparation_sha256')!=record['preparation_sha256']:
        raise ValueError('Parent-retained fixed launch required')
    verify_prepared(record)
    for name in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE'):os.environ[name]='1'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    root=_new_target(str(places['output']/role));root.mkdir(mode=0o700);began=time.monotonic();minimum_available=None
    def guard():
        nonlocal minimum_available
        available=next(int(row.split()[1])*1024 for row in Path('/proc/meminfo').read_text().splitlines() if row.startswith('MemAvailable:'))
        minimum_available=available if minimum_available is None else min(minimum_available,available)
        if available<25*GIB or time.monotonic()-began>SECONDS:raise RuntimeError('Actual preference child sampled reserve/deadline crossed')
    try:
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('Actual CUDA/BF16 support required')
        pair_scores(record,root,guard);verify_prepared(record)
        for panel in PANELS:
            guard()
            result=run_generation(root=ROOT,items_path=evidence/(panel+'-items.json'),contract_path=evidence/(panel+'-contract.json'),
                checkpoint=record['producers'][role]['path'],output=root/panel,environment_lock=LOCK,allow_cuda=True)
            verify_prepared(record)
            if result['status']!='completed' or result['record_count']!=COUNTS[panel]:raise RuntimeError('Original common retention panel incomplete')
        retain(root/'child-result.json',dict(status='completed',producer=role,planned_responses=144,
            minimum_sampled_mem_available_bytes=minimum_available,wall_seconds=time.monotonic()-began))
        return 0
    except BaseException as error:
        retain(root/'child-failure.json',dict(type=type(error).__name__,message=str(error),wall_seconds=time.monotonic()-began,
            minimum_sampled_mem_available_bytes=minimum_available,scope='All pair/generation attempts and costs remain retained'));raise


def jsonl(path,errors=None):
    if not Path(path).exists():return []
    maximum=EVENT_BYTES if Path(path).name=='events.jsonl' else JSONL_BYTES
    if Path(path).stat().st_size>maximum:raise ValueError('Bounded per-panel retained JSONL required')
    rows=[]
    for number,line in enumerate(Path(path).read_text().splitlines(),1):
        if not line.strip():continue
        try:rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            if errors is not None:errors.append(dict(path=str(path),line=number,error=str(error),
                scope='Original unreadable bytes remain retained; not a completed record'))
    return rows


def grouped_numeric_difference(left,right,metric,*,draws=2000,seed=1010):
    if type(draws) is not int or not 1<=draws<=100000:raise ValueError('Bounded bootstrap draws required')
    if set(left)!=set(right) or any(left[key].get(metric) is None or right[key].get(metric) is None for key in left):
        return dict(status='missing-pairs',difference=None,interval95=None)
    if not left:return dict(status='empty-slice',difference=None,interval95=None)
    groups=defaultdict(list)
    for key in left:
        if left[key]['source_group']!=right[key]['source_group']:raise ValueError('Paired source group changed')
        a,b=left[key][metric],right[key][metric]
        if (type(a) not in (int,float,bool) or type(b) not in (int,float,bool)
                or not math.isfinite(a) or not math.isfinite(b)):raise ValueError('Finite paired metric values required')
        groups[left[key]['source_group']].append(float(b)-float(a))
    values=list(groups.values());rng=random.Random(seed);samples=[]
    for _ in range(draws):
        resampled=[values[rng.randrange(len(values))] for _ in values]
        samples.append(sum(sum(v) for v in resampled)/sum(len(v) for v in resampled))
    samples.sort()
    return dict(status='complete-pairs',pairs=len(left),source_groups=len(groups),
        difference=sum(sum(v) for v in values)/len(left),interval95=[samples[int(.025*(draws-1))],samples[int(.975*(draws-1))]],
        scope='Source-group percentile uncertainty on this fixed diagnostic population, not general capability')


def grouped_difference(left,right,*,draws=2000,seed=1010):
    return grouped_numeric_difference(left,right,'correct',draws=draws,seed=seed)


def identity_receipt_check(record,role,panel,contract,generation_summary,rows,*,require_summary=True):
    """Join the actual saved invocation receipt, not only two copied SHA strings."""
    evidence=paths(record['run_id'])['evidence'];path=paths(record['run_id'])['output']/role/panel/'input-identity.json'
    if not path.exists():return False,dict(status='missing',path=str(path))
    try:
        binding=_digest(str(path),maximum=RECEIPT_BYTES);identity=json.loads(path.read_text())
        if type(identity) is not dict:raise ValueError('Saved identity receipt must be a dictionary')
        def input_key(value):
            value=Path(value).resolve();root=ROOT.resolve()
            return str(value.relative_to(root)) if value.is_relative_to(root) else str(value)
        inputs={input_key(record['input_bindings'][panel+'/'+kind]['path']):record['input_bindings'][panel+'/'+kind]['sha256']
            for kind in ('items','contract')}
        tokenizer_files=artifact_hashes(record['producers'][role]['path'],patterns=TOKENIZER_PATTERNS)
        checkpoint='local-hf-sha256:'+canonical_hash(dict(files=record['producers'][role]['files'],
            tokenizer=tokenizer_files,interface=contract['settings']['generation']['interface']['interface_sha256']))
        items={item['id']:item for item in json.loads((evidence/(panel+'-items.json')).read_text())}
        def row_join(row):
            item=items.get(row.get('item_id'))
            return (item is not None and row.get('checkpoint_id')==checkpoint and row.get('contract_id')==contract['identity']
                and row.get('settings_sha256')==canonical_hash(contract['settings'])
                and row.get('checkpoint_interface_sha256')==contract['settings']['generation']['interface']['interface_sha256']
                and all(row.get(key)==item[key] for key in ('source_group','task','split'))
                and row.get('sample_index')==0 and row.get('input_identity_sha256')==identity.get('identity_sha256')
                and row.get('input_sha256')==inputs)
        lock=record['input_bindings']['environment_lock'];environment=identity.get('environment',{})
        selected_lock=environment.get('environment_lock',{})
        valid=(identity.get('schema_version')==1
            and identity.get('identity_sha256')==canonical_hash({key:value for key,value in identity.items() if key!='identity_sha256'})
            and (not require_summary or identity.get('identity_sha256')==generation_summary.get('identity_sha256'))
            and identity.get('checkpoint_path')==str(Path(record['producers'][role]['path']).resolve())
            and identity.get('checkpoint_files')==record['producers'][role]['files']
            and identity.get('selected_tokenizer_path')==str(Path(record['producers'][role]['path']).resolve())
            and identity.get('selected_tokenizer_files')==tokenizer_files
            and identity.get('input_sha256')==inputs
            and identity.get('source_sha256')=={key:record['source_bindings'][key]['sha256'] for key in GENERATION_SOURCES}
            and identity.get('config')==contract['settings']
            and identity.get('checkpoint_interface')==contract['settings']['generation']['interface']
            and identity.get('device',{}).get('mode')==contract['settings']['generation']['device']
            and identity.get('command')==record['commands'][role]
            and environment.get('interpreter')==record['input_bindings']['interpreter']['path']
            and selected_lock.get('status')=='hashed' and selected_lock.get('path')==str(Path(lock['path']).resolve())
            and selected_lock.get('sha256')==lock['sha256']
            and all(row_join(row) for row in rows)
            and _digest(str(path),maximum=RECEIPT_BYTES)==binding)
        return valid,dict(status='bound' if valid else 'invalid-joins',path=str(path),artifact=binding,
            scope='Receipt self-SHA and exact selected checkpoint/tokenizer/source/input/lock/config/interface/closed-command joins')
    except (OSError,ValueError,TypeError,KeyError) as error:
        return False,dict(status='unreadable-or-invalid',path=str(path),error=str(error),scope='Original receipt bytes remain retained')


def consume(record):
    places=paths(record['run_id']);evidence=places['evidence'];panels_report={};pairs={};checks={};unreadable=[];pair_costs={}
    validation_ids={row['id'] for row in record['validation_rows']}
    for role in ROLES:
        pairs[role]=jsonl(places['output']/role/'pair-scores.jsonl',unreadable)
        events=jsonl(places['output']/role/'pair-cost-events.jsonl',unreadable)
        pair_costs[role]=events[-1] if events else None
        checks[role+'/all_validation_pairs']=[row['id'] for row in pairs[role]]==[row['id'] for row in record['validation_rows']]
        checks[role+'/frozen_validation_groups']=all(row.get('id') in validation_ids
            and row.get('source_group')==record['scenario_groups'][row['id']] for row in pairs[role])
        summary_path=places['output']/role/'pair-summary.json'
        summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
        checks[role+'/pair_reference_no_grad']=summary.get('reference_has_gradients') is False
        checks[role+'/pair_completed_and_costed']=(summary.get('status')=='completed' and summary.get('pairs')==4
            and all(summary.get('cost',{}).get(network)==dict(attempted_calls=8,forward_calls=8,
                attempted_positions=record['limits']['validation_positions_each'],
                forward_positions=record['limits']['validation_positions_each']) for network in ('policy','reference'))
            and pair_costs[role] is not None and pair_costs[role]['cost']==summary.get('cost'))
        child_path=places['output']/role/'child-result.json'
        completed=json.loads(child_path.read_text()) if child_path.exists() else {}
        checks[role+'/child_complete']=completed.get('status')=='completed' and completed.get('planned_responses')==144
    for panel in PANELS:
        items=json.loads((evidence/(panel+'-items.json')).read_text());contract=json.loads((evidence/(panel+'-contract.json')).read_text())
        by_role={};slices=defaultdict(set);partial={};summaries={};receipts={};partial_receipts={}
        for item in items:
            slices['task/'+item['task']].add(item['id'])
            if panel=='reasoning':
                slices['campaign-role/'+item['campaign_role']].add(item['id'])
                slices['family/'+item['family']].add(item['id'])
                slices['template/'+item['template_id']].add(item['id'])
        for role in ROLES:
            rows=jsonl(places['output']/role/panel/'responses.jsonl',unreadable)
            graded=replay_records(items,rows,contract)['rows'];raw={row['item_id']:row for row in rows};index={row['item_id']:row for row in graded}
            unfinished={};all_partial=[]
            for event in jsonl(places['output']/role/panel/'events.jsonl',unreadable):
                if event.get('stage')=='partial_response':
                    value=event['record'];all_partial.append(value)
                    if value['item_id'] not in raw:unfinished[(value['item_id'],value['sample_index'])]=value
            partial[role]=list(unfinished.values())
            checks[role+'/'+panel+'/coverage']=len(rows)==COUNTS[panel] and len(raw)==COUNTS[panel]
            checks[role+'/'+panel+'/no_response_errors']=all(row['error'] is None for row in rows)
            expected_checkpoint='local-hf-sha256:'+canonical_hash(dict(files=record['producers'][role]['files'],
                tokenizer=artifact_hashes(record['producers'][role]['path'],patterns=TOKENIZER_PATTERNS),
                interface=contract['settings']['generation']['interface']['interface_sha256']))
            summary_path=places['output']/role/panel/'summary.json'
            generation_summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
            receipt_ok,receipts[role]=identity_receipt_check(record,role,panel,contract,generation_summary,rows)
            partial_ok,partial_receipts[role]=identity_receipt_check(record,role,panel,contract,{},all_partial,require_summary=False)
            checks[role+'/'+panel+'/partial_identity_join']=not all_partial or partial_ok
            partial_receipts[role]['partial_records_checked']=len(all_partial)
            partial[role]=[dict(value,provenance_status='bound-selected-policy' if partial_ok else 'unverified-retained-cost')
                for value in unfinished.values()]
            checks[role+'/'+panel+'/generation_identity_complete']=(generation_summary.get('status')=='completed'
                and generation_summary.get('record_count')==COUNTS[panel]
                and generation_summary.get('contract_id')==contract['identity']
                and generation_summary.get('checkpoint_id')==expected_checkpoint and generation_summary.get('inputs_unchanged') is True
                and receipt_ok and all(row['checkpoint_id']==expected_checkpoint
                    and row['input_identity_sha256']==generation_summary.get('identity_sha256') for row in rows))
            by_role[role]={item['id']:dict(item_id=item['id'],source_group=item['source_group'],task=item['task'],
                correct=(None if item['id'] not in raw else index[item['id']]['correct'] if panel=='reasoning'
                    else raw[item['id']]['error'] is None and raw[item['id']]['response_text'].strip()==item['reference']),
                status='missing' if item['id'] not in index else index[item['id']]['status'],
                natural_stop=None if item['id'] not in index else index[item['id']]['natural_termination'],
                format_valid=None if item['id'] not in index else index[item['id']]['format_valid'],
                truncated=None if item['id'] not in index else index[item['id']]['truncated'],
                cost=None if item['id'] not in index else index[item['id']]['cost']) for item in items}
            completed=[value for value in by_role[role].values() if value['cost'] is not None]
            cost_keys=set(key for value in completed for key in value['cost'])|{'wall_seconds','generation_tokens','scoring_tokens'}
            summaries[role]=dict(planned=COUNTS[panel],completed=len(completed),missing=COUNTS[panel]-len(completed),
                correct=sum(value['correct'] is True for value in completed),natural_stops=sum(value['natural_stop'] is True for value in completed),
                truncated=sum(value['truncated'] is True for value in completed),
                completed_cost={key:dict(known_total=sum(value['cost'].get(key) or 0 for value in completed),
                    unknown_completed_rows=sum(value['cost'].get(key) is None for value in completed)) for key in sorted(cost_keys)})
        differences={role:{name:grouped_difference({key:by_role['unchanged'][key] for key in ids},
            {key:by_role[role][key] for key in ids}) for name,ids in slices.items()} for role in ('chosen100','dpo100')}
        ancillary={role:{name:{metric:grouped_numeric_difference({key:by_role['unchanged'][key] for key in ids},
            {key:by_role[role][key] for key in ids},metric) for metric in ('format_valid','natural_stop','truncated')}
            for name,ids in slices.items()} for role in ('chosen100','dpo100')}
        panels_report[panel]=dict(rows=by_role,incomplete_attempts=partial,
            summaries=summaries,input_identity_receipts=receipts,partial_input_identity_receipts=partial_receipts,
            ancillary_grouped_differences_vs_unchanged=ancillary,
            cost_boundary='Completed-record and latest nonfinal costs are separate; partial policy attribution requires bound receipt joins, otherwise costs remain unverified; killed in-flight work may be unobserved',
            slices={key:sorted(ids) for key,ids in slices.items()},differences_vs_unchanged=differences,
            scope='Reasoning20 greedy64 saved-custom-template retention, not original32/128 native-instruct publication' if panel=='reasoning'
                else 'Original frozen '+panel+' population; strict whole-response equality separate from generic format grading')
    likelihood_differences={}
    expected_ids=[row['id'] for row in record['validation_rows']]
    paired={role:{key:next((row for row in pairs[role] if row['id']==key),
        dict(id=key,source_group=record['scenario_groups'][key])) for key in expected_ids} for role in ROLES}
    for role in ('chosen100','dpo100'):
        valid=checks['unchanged/frozen_validation_groups'] and checks[role+'/frozen_validation_groups']
        likelihood_differences[role]={metric:(grouped_numeric_difference(paired['unchanged'],paired[role],metric) if valid
            else dict(status='invalid-frozen-groups',difference=None,interval95=None))
            for metric in ('chosen_logp','rejected_logp','reference_relative_margin','reference_relative_logp_margin')}
    checks['no_unreadable_jsonl']=not unreadable
    return dict(checks=checks,validation_likelihood_and_reference_margin=pairs,latest_pair_costs=pair_costs,
        validation_grouped_differences_vs_unchanged=likelihood_differences,
        independent_panels=panels_report,unreadable_jsonl=unreadable,
        scope='Separate likelihood/margin/location/assistant/reasoning slices; negative differences preserved, no universal pooled score')


def execute(record,declaration):
    if type(declaration) is not str or not 12<=len(declaration)<=4096:raise ValueError('Recorded bounded goal-authorized scope required')
    places=paths(record['run_id']);evidence=places['evidence'];_new_target(str(places['output']));places['output'].mkdir(mode=0o700);invocations=[]
    try:
        for role in ROLES:
            verify_prepared(record);argv=record['commands'][role]
            retain(evidence/(role+'-launch.json'),dict(argv=argv,operator_declaration=declaration,preparation_sha256=record['preparation_sha256']))
            result=_supervise(argv,evidence/(role+'-supervision'),native=True,seconds=SECONDS,probe=_native_probe,operator_declaration=declaration)
            retain(evidence/(role+'-returned-supervision.json'),result);invocations.append(dict(producer=role,result=result));verify_prepared(record)
            if result['status']!='completed' or result['actual_exit_code']!=0:raise RuntimeError('Actual preference child failed; all spending retained')
        summary=consume(record);verify_prepared(record)
        retain(evidence/'comparison.json',summary)
        status='passed' if all(summary['checks'].values()) else 'failed'
        retain(evidence/'acceptance.json',dict(status=status,checks=summary['checks'],invocations=invocations,
            producers=record['producers'],scope=record['scope'],precision=record['precision']))
        if status!='passed':raise RuntimeError('Actual common comparison incomplete; retained negative/failed evidence')
        return 0
    except BaseException as error:
        consumer_failure=None
        try:retain(evidence/'partial-comparison.json',consume(record))
        except BaseException as secondary:
            consumer_failure=dict(type=type(secondary).__name__,message=str(secondary),
                scope='Raw bytes remain retained; failed consumer does not replace the original invocation failure')
            retain(evidence/'partial-consumer-failure.json',consumer_failure)
        retain(evidence/'failure.json',dict(type=type(error).__name__,message=str(error),invocations=invocations,
            partial_consumer_failure=consumer_failure,
            scope='Original failed/partial pair and generation costs remain retained; no automatic retry or cap refill'));raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-id',required=True)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--operator-declaration')
    parser.add_argument('--child',choices=ROLES,help=argparse.SUPPRESS);args=parser.parse_args(argv)
    if args.child:
        if args.execute or args.operator_declaration:parser.error('Only fixed parent-retained child route is supported')
        return child(args.run_id,args.child)
    if args.execute and (not args.operator_declaration or not 12<=len(args.operator_declaration)<=4096):parser.error('Recorded bounded execution scope required')
    record=prepare(args.run_id)
    if args.execute:return execute(record,args.operator_declaration)
    print(json.dumps(dict(status='prepared-not-executed',evidence=record['locations']['evidence'])));return 0


if __name__=='__main__':raise SystemExit(main())
