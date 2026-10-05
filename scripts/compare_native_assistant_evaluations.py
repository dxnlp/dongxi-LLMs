#!/usr/bin/env python3
"""CPU-only original120/40 Base/full400/merged-LoRA400 saved-record comparison.

No model/tokenizer loading, generation, acquisition, GPU or arbitrary paths.
Strict answer equality and generic parser replay remain distinct estimands.
Paired intervals require every original item and its actual matched prompt IDs.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import run_native_assistant_evaluation as publication
from dongxi_llms.native_profile_supervisor import _digest,_new_target
from dongxi_llms.reasoning_evaluation import replay_records,paired_group_bootstrap
from dongxi_llms.reasoning_generation import validate_settings,attempt_seed
from dongxi_llms.run_identity import canonical_hash,validate_interface,environment_identity,artifact_hashes,TOKENIZER_PATTERNS

LABELS=('base','full400','lora400-fp32')
PAIRS=(('base','full400'),('base','lora400-fp32'),('full400','lora400-fp32'))
RUN_ID='run-01'
COMPARISON_RUN_ID='run-02'
DRAWS=2000
SEED=1010
MAX_BYTES=64*1024**2
SOURCES=('scripts/compare_native_assistant_evaluations.py','tests/test_native_assistant_comparison.py',
    'scripts/run_native_assistant_evaluation.py','src/dongxi_llms/reasoning_evaluation.py',
    'src/dongxi_llms/reasoning_generation.py','src/dongxi_llms/evaluation_lab.py',
    'src/dongxi_llms/run_identity.py','src/dongxi_llms/native_profile_supervisor.py')


def retain(path,value):
    with Path(path).open('x',encoding='utf-8') as handle:
        json.dump(value,handle,indent=2,allow_nan=False);handle.write('\n')
        handle.flush();os.fsync(handle.fileno())


def bytes_identity(path):
    """Stable bounded regular saved bytes, including an empty failed journal."""
    import stat
    path=Path(path);fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        before=os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0<=before.st_size<=MAX_BYTES:
            raise ValueError('Bounded regular saved comparison input required')
        chunks=[];count=0
        while block:=os.read(fd,1024**2):
            count+=len(block)
            if count>MAX_BYTES:raise ValueError('Saved comparison input exceeds byte ceiling')
            chunks.append(block)
        raw=b''.join(chunks);after=os.fstat(fd)
        if (len(raw)!=before.st_size or (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=
                (after.st_size,after.st_mtime_ns,after.st_ctime_ns)):
            raise ValueError('Saved comparison bytes changed during inspection')
    finally:os.close(fd)
    return raw,dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def parse_json(raw):
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('Duplicate saved JSON key')
            result[key]=value
        return result
    return json.loads(raw,object_pairs_hook=unique,
        parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite saved JSON')))


def json_input(path):
    raw,binding=bytes_identity(path);return parse_json(raw),binding


def jsonl_input(path):
    raw,binding=bytes_identity(path);lines=raw.splitlines(keepends=True)
    tail=b'' if not lines or lines[-1].endswith(b'\n') else lines.pop()
    return [parse_json(line) for line in lines],tail,binding


def report_paths(label):
    if label not in LABELS:raise ValueError('Only the three fixed publication selectors are supported')
    return ROOT/'experiments/reports'/f'native-assistant-publication-20261005-{label}-{RUN_ID}'


def expected_checkpoint_identity(files,tokenizer_files,interface):
    return 'local-hf-sha256:'+canonical_hash(dict(files=files,tokenizer=tokenizer_files,
        interface=interface['interface_sha256']))


def validate_loaded_identity(records,loaded,expected):
    if (len(loaded)!=1 or loaded[0].get('dtype')!='torch.bfloat16' or loaded[0].get('device')!='cuda' or
            loaded[0].get('checkpoint_id')!=expected or any(row['checkpoint_id']!=expected for row in records)):
        raise ValueError('Actual accepted checkpoint identity/BF16 CUDA model-load event differs')


def source_bindings():
    return {name:_digest(str(ROOT/name)) for name in SOURCES}


def logical_contract(contract,items):
    replay_records(items,[],contract)
    settings=validate_settings(contract['settings'])
    expected=dict(mode='greedy',seed=SEED,temperature=1,top_k=None,top_p=1)
    generation=settings['generation']
    if (settings['template_id']!='original-instruction-interface-v1-publication' or
            settings['thinking_mode']!='template-default' or settings['decoding']!=expected or
            settings['stopping']!={'eos_token_ids':[151643],'turn_stop_token_ids':[151645],'pad_token_id':151643} or
            settings['max_new_tokens']!=64 or generation['input_mode']!='chat' or
            generation['context_window']!=512 or generation['samples']!=1 or generation['device']!='cuda' or
            generation['dtype']!='bfloat16' or generation['add_special_tokens'] is not False or
            generation['max_run_seconds']!=900 or generation['scoring_text']!='decode_without_terminal_stop' or
            hashlib.sha256(generation['template'].encode()).hexdigest()!=
                hashlib.sha256((ROOT/'experiments/data/instruction_interface_v1.jinja').read_bytes()).hexdigest()):
        raise ValueError('Original common greedy64/template/stop/BF16 publication contract required')
    interface=validate_interface(generation['interface'])
    # Source IDs/classes are physical provenance, not tokenizer encoding rules.
    generation['interface']={key:deepcopy(interface[key]) for key in
        ('schema_version','tokenizer','template_sha256','generation_stop_ids','interface_sha256')}
    return dict(schema_version=contract['schema_version'],parser_version=contract['parser_version'],
        rubric_version=contract['rubric_version'],suite_sha256=contract['suite_sha256'],settings=settings)


def validate_original_items(items,original):
    if (items!=original or len(items)!=120 or len({row['id'] for row in items})!=120 or
            len({row['source_group'] for row in items})!=40 or Counter(row['task'] for row in items)!=
                {'copy':40,'reverse':40,'extract':40}):
        raise ValueError('Unchanged original120-item/40-source-group held-out panel required')


def strict_rows(items,records):
    index={item['id']:item for item in items}
    return [dict(publication.strict_result(row,index[row['item_id']]),sample_id=row['sample_id'],
        physical_contract_id=row['contract_id'],physical_checkpoint_id=row['checkpoint_id'],
        error=row['error'],cost=deepcopy(row['cost'])) for row in records]


def summaries(rows,*,strict):
    def one(selected):
        costs={key:dict(known_sum=sum(row['cost'][key] for row in selected if row['cost'].get(key) is not None),
            unknown_rows=sum(row['cost'].get(key) is None for row in selected))
            for key in sorted({key for row in selected for key in row['cost']})}
        return dict(attempts=len(selected),successes=sum(bool(row['exact'] if strict else row['task_success']) for row in selected),
            errors=sum(row['error'] is not None for row in selected),
            natural_stops=sum(row['stop_reason'] in ('eos','turn_stop') for row in selected),
            caps=sum(row['truncated'] for row in selected),stop_reasons=dict(Counter(row['stop_reason'] for row in selected)),
            costs=costs,scope='Overlapping logical counts and summed attempt time; not FLOPs, unique work or optimized inference throughput')
    return dict(overall=one(rows),by_task={task:one([row for row in rows if row.get('task')==task])
        for task in ('copy','reverse','extract')})


def analyze_panel(label,items,contract,records):
    logical=logical_contract(contract,items);generic=replay_records(items,records,contract)
    index={item['id']:item for item in items};seen=set();prefixes={}
    for row in records:
        if row['item_id'] in seen:raise ValueError('Only one actual greedy attempt per original item is supported')
        seen.add(row['item_id']);seed=attempt_seed(SEED,row['item_id'],0)
        if (row.get('sample_index')!=0 or row.get('attempt_seed')!=seed or
                row['sample_id']!=f'sample-0-seed-{seed}' or row.get('decoding')!=contract['settings']['decoding']):
            raise ValueError('Actual single frozen greedy attempt identity required')
        if row.get('prompt_token_ids') is not None:
            prefixes[row['item_id']]=row['prompt_token_ids']
    if len({row['checkpoint_id'] for row in records})>1:raise ValueError('One physical checkpoint per selector required')
    strict=strict_rows(items,records)
    for row in strict:row['task']=index[row['item_id']]['task']
    complete=len(seen)==120 and len(prefixes)==120 and all(prefixes[key] for key in prefixes)
    encoded=dict(items_sha256=canonical_hash(items),prompt_token_ids={key:prefixes[key] for key in sorted(prefixes)})
    return dict(label=label,physical_contract_id=contract['identity'],logical_contract=logical,
        logical_contract_sha256=canonical_hash(logical),encoded_contract=encoded,
        encoded_contract_sha256=canonical_hash(encoded) if complete else None,
        complete_original_coverage=complete,missing_item_ids=sorted(set(index)-seen),
        raw_records=deepcopy(records),strict=dict(rows=strict,summary=summaries(strict,strict=True)),
        generic=dict(replay=generic,summary=summaries(generic['rows'],strict=False)),
        scope='Generic parser behavior is not strict case-sensitive whole decoded-answer equality')


def paired_comparisons(panels):
    if set(panels)!=set(LABELS):raise ValueError('All three predetermined selectors required')
    logical={panel['logical_contract_sha256'] for panel in panels.values()}
    if len(logical)!=1:raise ValueError('Common logical encoding/template/decoding/precision contract mismatch')
    executions={panel.get('producer_execution_contract_sha256') for panel in panels.values()}
    if len(executions)!=1:raise ValueError('Actual generation source/environment/backend contract mismatch')
    result=[]
    for left,right in PAIRS:
        a,b=panels[left],panels[right]
        if not a['complete_original_coverage'] or not b['complete_original_coverage']:
            result.append(dict(left=left,right=right,status='unaligned-no-interval',intervals=None,
                reason='Actual original120 greedy item/prompt coverage is incomplete'));continue
        if a['encoded_contract_sha256']!=b['encoded_contract_sha256']:
            raise ValueError('Actual matched prompt token IDs differ')
        common=canonical_hash(dict(logical_contract_sha256=a['logical_contract_sha256'],
            encoded_contract_sha256=a['encoded_contract_sha256'],
            producer_execution_contract_sha256=a.get('producer_execution_contract_sha256')))
        def view(panel,kind):
            rows=panel['strict']['rows'] if kind=='strict' else panel['generic']['replay']['rows']
            return [dict(item_id=row['item_id'],sample_id=row['sample_id'],source_group=row['source_group'],
                contract_id=common,physical_contract_id=panel['physical_contract_id'],
                correct=bool(row['exact'] if kind=='strict' else row['correct']),
                task_success=bool(row['exact'] if kind=='strict' else row['task_success']),
                format_valid=False if kind=='strict' else bool(row['format_valid'])) for row in rows]
        intervals={}
        for kind,metrics in (('strict',('correct',)),('generic',('correct','task_success','format_valid'))):
            av,bv=view(a,kind),view(b,kind)
            if (len(av)!=120 or len(bv)!=120 or len({row['source_group'] for row in av})!=40 or
                    {(r['item_id'],r['sample_id'],r['source_group']) for r in av}!=
                    {(r['item_id'],r['sample_id'],r['source_group']) for r in bv}):
                raise ValueError('Bootstrap requires truly aligned120 greedy attempts/40 source groups')
            intervals[kind]={metric:paired_group_bootstrap(av,bv,draws=DRAWS,seed=SEED,metric=metric) for metric in metrics}
        result.append(dict(left=left,right=right,status='aligned-descriptive',common_analysis_contract_sha256=common,
            physical_contract_ids={left:a['physical_contract_id'],right:b['physical_contract_id']},
            analysis_identity_scope='Derived common logical+actual encoded contract for paired rows only; original physical receipts/records are unchanged',
            intervals=intervals,limits='Descriptive fixed source-cluster intervals; no universal model ranking or population improvement claim'))
    return result


def load_publication(label,original,bindings):
    root=report_paths(label);files={}
    for role,name in (('preparation','preparation.json'),('launch','launch.json'),('supervision','returned-supervision.json'),
            ('terminal','supervision/result.json'),('items','items.json'),('contract','contract.json'),
            ('generation_contract','generation/contract.json'),('identity','generation/input-identity.json'),
            ('observed_interface','generation/observed-interface.json')):
        files[role],binding=json_input(root/name);bindings[str(root/name)]=binding
    items=files['items'];validate_original_items(items,original)
    contract=files['contract'];prepared=files['preparation'];identity=files['identity'];supervised=files['supervision']
    if (files['generation_contract']!=contract or files['launch']['command']!=prepared['command'] or
            supervised.get('child_command')!=prepared['command'] or supervised.get('status') not in ('completed','failed') or
            type(supervised.get('actual_exit_code')) is not int or supervised.get('actual_native_profile_executed') is not True or
            any(files['terminal'].get(key)!=supervised.get(key) for key in ('status','actual_exit_code','child_command','child_pid','stop_reason')) or
            identity.get('identity_sha256')!=canonical_hash({k:v for k,v in identity.items() if k!='identity_sha256'}) or
            identity.get('checkpoint_files')!=prepared['checkpoint_files'] or
            prepared.get('contract')!=contract['identity'] or prepared.get('items')!=120 or prepared.get('source_groups')!=40):
        raise ValueError('Actual closed publication identity/launch/outcome binding differs')
    parent=publication.checkpoint(label,RUN_ID)
    command=[publication.PYTHON,str(ROOT/'scripts/generate_reasoning_records.py'),'--checkpoint',str(parent),
        '--items',str(root/'items.json'),'--contract',str(root/'contract.json'),
        '--output',str(root/'generation'),'--environment-lock',publication.LOCK,'--allow-cuda']
    if (prepared['command']!=command or identity.get('checkpoint_path')!=str(parent) or identity.get('command')!=command or
            identity.get('config')!=contract['settings'] or
            identity.get('selected_tokenizer_files')!=
                {name:value for name,value in prepared['checkpoint_files'].items() if name in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json','chat_template.jinja','vocab.json','merges.txt','tokenizer.model')}):
        raise ValueError('Actual selected model/tokenizer/configuration identity differs')
    actual_files=artifact_hashes(parent)
    if actual_files!=prepared['checkpoint_files']:raise ValueError('Actual publication checkpoint bytes changed')
    bindings[str(parent)]=dict(kind='artifact-file-map',path=str(parent),files=actual_files)
    if files['observed_interface']!=contract['settings']['generation']['interface']:
        raise ValueError('Observed publication tokenizer interface differs from its contract')
    for name,value in prepared['bindings'].items():
        if _digest(name,executable=name==publication.PYTHON)!=value:
            raise ValueError('Actual publication source/input bindings changed')
        bindings[name]=value
    for name,value in identity['source_sha256'].items():
        if _digest(str(ROOT/name))['sha256']!=value:
            raise ValueError('Actual generation implementation bytes differ from the input receipt')
    expected_inputs={str((root/name).relative_to(ROOT)):bindings[str(root/name)]['sha256']
        for name in ('items.json','contract.json')}
    if identity.get('input_sha256')!=expected_inputs:
        raise ValueError('Actual producer item/contract byte identity differs')
    environment=identity['environment']
    if (environment.get('interpreter')!=publication.PYTHON or
            environment.get('environment_lock',{}).get('path')!=publication.LOCK or
            environment['environment_lock'].get('sha256')!=prepared['bindings'][publication.LOCK]['sha256'] or
            identity.get('device')!={'mode':'cuda'}):
        raise ValueError('Actual producer interpreter/lock/device identity differs')
    if label!='base':
        if publication.parent_evidence(label,RUN_ID,parent)!=prepared['parent_evidence']:
            raise ValueError('Actual accepted parent/merge evidence differs from publication preparation')
        exported=prepared['parent_evidence']['exported_policy']
        bindings[exported['path']]=dict(kind='artifact-file-map',path=exported['path'],files=exported['files'])
        for key in ('acceptance','merge_acceptance'):
            value=prepared['parent_evidence'].get(key)
            if value is not None:
                if _digest(value['path'])!=value:raise ValueError('Publication parent outcome receipt changed')
                bindings[value['path']]=value
    records,tail,binding=jsonl_input(root/'generation/responses.jsonl');bindings[binding['path']]=binding
    if any(row['input_identity_sha256']!=identity['identity_sha256'] or
            row.get('input_sha256')!=identity['input_sha256'] for row in records):
        raise ValueError('Raw response actual input identity differs from retained receipt')
    panel=analyze_panel(label,items,contract,records)
    events,event_tail,binding=jsonl_input(root/'generation/events.jsonl');bindings[binding['path']]=binding
    loaded=[event for event in events if event.get('stage')=='model_loaded']
    expected=expected_checkpoint_identity(actual_files,artifact_hashes(parent,patterns=TOKENIZER_PATTERNS),
        contract['settings']['generation']['interface'])
    validate_loaded_identity(records,loaded,expected)
    execution=dict(source_sha256=identity['source_sha256'],environment=environment,
        model={key:loaded[0][key] for key in ('parameter_count','device','dtype','model_class')},
        attention_implementation='eager: explicit source-bound producer request, not observed per-call kernel',
        scope='Common producer implementation/runtime identity; excludes checkpoint weights and physical paths')
    panel['producer_execution_contract']=execution
    panel['producer_execution_contract_sha256']=canonical_hash(execution)
    panel['generation_event_evidence']=dict(binding=binding,stage_counts=dict(Counter(event['stage'] for event in events)),
        model_loaded=loaded[0],failure_events=[event for event in events if 'failure' in event],
        partial_prefixes=[event for event in events if event.get('stage')=='partial_response' and
            event.get('record',{}).get('item_id') not in {row['item_id'] for row in records}],
        incomplete_tail=dict(bytes=len(event_tail),sha256=hashlib.sha256(event_tail).hexdigest(),hex=event_tail.hex()),
        scope='Actual append-only events remain byte-bound in place; finalized raw records retained separately')
    if event_tail:panel['complete_original_coverage']=False
    for role,name in (('summary','generation/summary.json'),('generic','generation/evaluation.json'),
            ('strict','strict-results.json'),('failure','failure.json'),('generation_failure','generation/failure.json')):
        path=root/name
        if path.exists():
            value,binding=json_input(path);bindings[str(path)]=binding;files[role]=value
    if 'generic' in files and files['generic']!=panel['generic']['replay']:
        raise ValueError('Actual generic replay does not reproduce from retained raw records')
    if 'strict' in files:
        expected=[publication.strict_result(row,{item['id']:item for item in items}[row['item_id']]) for row in records]
        if files['strict'].get('rows')!=expected or files['strict'].get('exact')!=sum(row['exact'] for row in expected):
            raise ValueError('Actual strict producer results differ from decoded-answer replay')
    if 'summary' in files and (files['summary'].get('identity_sha256')!=identity['identity_sha256'] or
            files['summary'].get('contract_id')!=contract['identity'] or files['summary'].get('record_count')!=len(records) or
            files['summary'].get('inputs_unchanged') is not True):
        raise ValueError('Actual generation completion identity differs')
    panel['outcome_receipts']={key:files[key] for key in ('supervision','summary','strict','failure','generation_failure') if key in files}
    panel['physical_input_identity']=identity
    panel['unparsed_incomplete_tail']=dict(bytes=len(tail),sha256=hashlib.sha256(tail).hexdigest(),hex=tail.hex())
    if tail:panel['complete_original_coverage']=False
    return panel


def development_nll(bindings):
    rows={}
    dev=ROOT/'outputs/course-sft-interface-v1/dev.jsonl';raw,dev_binding=bytes_identity(dev);bindings[str(dev)]=dev_binding
    card,card_binding=json_input(ROOT/'experiments/data/instruction-interface-v1-data-card.json')
    bindings[card_binding['path']]=card_binding
    if (dev_binding['sha256']!=card['splits']['dev']['sha256'] or
            len(raw.splitlines())!=60 or len({parse_json(line)['group'] for line in raw.splitlines()})!=20):
        raise ValueError('Original60-item/20-source-group development NLL population required')
    for mode in ('full','lora'):
        stem=f'native-sft-{mode}-pilot400-20261005-{RUN_ID}'
        report=ROOT/'experiments/reports'/stem;output=ROOT/'outputs'/stem
        values={}
        for role,path in (('acceptance',report/'acceptance.json'),('result',output/'result.json'),('config',output/'config.json')):
            values[role],binding=json_input(path);bindings[str(path)]=binding
        accepted,result,config=(values[key] for key in ('acceptance','result','config'))
        if (accepted.get('status')!='passed' or not accepted.get('checks') or any(value is not True for value in accepted['checks'].values()) or
                accepted['result']!=result or result.get('updates')!=400 or result.get('resume_checkpoint_update')!=0 or
                config.get('mode')!=mode or config.get('updates')!=400 or
                config.get('dev_sha256')!=dev_binding['sha256'] or config.get('dev')!=str(dev) or
                config.get('template_sha256')!=hashlib.sha256((ROOT/'experiments/data/instruction_interface_v1.jinja').read_bytes()).hexdigest()):
            raise ValueError('Actual accepted fresh400 training outcome required for separate development NLL')
        for key in ('initial_dev_nll','final_dev_nll'):
            if type(result.get(key)) not in (int,float) or not math.isfinite(result[key]) or result[key]<0:
                raise ValueError('Finite observed initial/final development NLL required')
        rows[mode]=dict(initial_dev_nll=result['initial_dev_nll'],final_dev_nll=result['final_dev_nll'],
            initial_to_final_delta=result['final_dev_nll']-result['initial_dev_nll'],
            completed_updates=400,training_targets=result['supervised_targets_this_invocation'],
            processed_positions=result['cumulative_processed_positions'],invocation_id=result['invocation_id'],
            dev_sha256=config['dev_sha256'],template_sha256=config['template_sha256'],
            source=config['dtype'],scope='Observed teacher-forced original60-item development NLL; LoRA parent result before FP32 merge, not publication accuracy')
    if len({row['dev_sha256'] for row in rows.values()})!=1 or len({row['template_sha256'] for row in rows.values()})!=1:
        raise ValueError('Development NLL populations/templates differ')
    return dict(rows=rows,scope='Separate observed initial/final development metric; not held-out120-item generation and no bootstrap NLL claim')


def verify_bindings(sources,inputs):
    if source_bindings()!=sources:raise ValueError('Comparison source changed')
    for path,expected in inputs.items():
        if expected.get('kind')=='artifact-file-map':
            if artifact_hashes(path)!=expected['files']:raise ValueError('Actual comparison model artifact bytes changed')
            continue
        observed=(_digest(path,executable=path==publication.PYTHON or path==sys.executable)
            if 'actual_path' in expected else bytes_identity(path)[1])
        if observed!=expected:raise ValueError('Actual comparison input/outcome receipt changed')


def run():
    evidence=ROOT/'experiments/reports'/f'native-assistant-comparison-20261005-{COMPARISON_RUN_ID}'
    _new_target(str(evidence));sources=source_bindings();inputs={}
    for path in (ROOT/'outputs/course-sft-interface-v1/test.jsonl',ROOT/'experiments/data/instruction-interface-v1-data-card.json',
            ROOT/'experiments/data/instruction_interface_v1.jinja',ROOT/'uv.lock'):
        inputs[str(path)]=bytes_identity(path)[1]
    inputs[sys.executable]=_digest(sys.executable,executable=True)
    original=publication.publication_items()
    panels={label:load_publication(label,original,inputs) for label in LABELS}
    nll=development_nll(inputs);comparisons=paired_comparisons(panels)
    verify_bindings(sources,inputs)
    evidence.mkdir(mode=0o700)
    try:
        retain(evidence/'preparation.json',dict(schema='dongxi-native-assistant-comparison-v1',
            source_bindings=sources,input_outcome_bindings=inputs,environment=environment_identity(ROOT/'uv.lock'),
            selectors=LABELS,pairs=PAIRS,bootstrap=dict(draws=DRAWS,seed=SEED,unit='original source_group',descriptive=True),
            scope='Saved actual receipts/records only; no model loading, generation, universal ranking or new scientific gate'))
        retain(evidence/'comparison.json',dict(status='complete-descriptive' if all(panel['complete_original_coverage'] for panel in panels.values()) else 'incomplete-publication',
            panels=panels,paired_comparisons=comparisons,development_nll=nll,
            common_logical_contract_sha256=panels['base']['logical_contract_sha256'],
            boundaries=['Strict whole decoded-answer equality and generic parser replay are separate',
                'Original raw tokens, stop tokens, errors, costs and physical receipts are retained unchanged',
                'Common analysis identity is derived; actual checkpoint/input identities need not be identical',
                'Fixed source-cluster CIs are descriptive; few task templates limit generalization',
                'Development teacher-forced NLL is separate from held-out generation; no universal ranking']))
        verify_bindings(sources,inputs)
        retain(evidence/'closing-bindings.json',dict(status='unchanged',source_bindings=sources,input_outcome_bindings=inputs))
        print(json.dumps(dict(status='completed',evidence=str(evidence))),flush=True);return 0
    except BaseException as error:
        retain(evidence/'failure.json',dict(type=type(error).__name__,message=str(error),scope='Comparison invalid; raw producer receipts remain unchanged'))
        raise


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    return run()


if __name__=='__main__':raise SystemExit(main())
