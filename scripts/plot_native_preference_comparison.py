#!/usr/bin/env python3
"""Read-only CPU figures of the fixed, accepted three-arm common comparison.

No model/tokenizer load, weights, acquisition, GPU, new bootstrap or producer
mutation. Run only after the actual run01 comparison has passed. Displayed
values are rejoined to retained raw inputs with the unchanged bounded grader.
Accepted weight inventories remain receipt declarations, not fresh weight hashes.
"""
import argparse
from fnmatch import fnmatch
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dongxi_llms.reasoning_evaluation import replay_records
from dongxi_llms.run_identity import canonical_hash,TOKENIZER_PATTERNS

RUN_ID='run-01'
STEM='native-preference-evaluation-20261005-'+RUN_ID
ROLES=('unchanged','chosen100','dpo100')
COUNTS=dict(location=4,assistant=120,reasoning=20)
LABELS=('Full400 reference','Chosen-only100','DPO100')
PRECISION='Likelihood: FP32 policy/reference with BF16 CUDA autocast/native single-shift masks; generation: all three BF16 loaded policies, common greedy64'
FIGURE_LABELS=dict(
    likelihood='FP32 policy/reference weights; BF16 CUDA autocast; SDPA (backend not forced); original single-shift masks',
    retention='Separate frozen panels; BF16-loaded CUDA policies; eager attention; common greedy64/custom saved template')
GIB=1024**3
PYTHON='/home/dongxi/dgx-spark-dongxi/.venv/bin/python'
LOCK='/home/dongxi/dgx-spark-dongxi/uv.lock'
JSON_BYTES=16*1024**2
EVENT_BYTES=64*1024**2
GENERATION_SOURCES=('src/dongxi_llms/reasoning_generation.py','src/dongxi_llms/reasoning_evaluation.py',
    'src/dongxi_llms/run_identity.py','src/dongxi_llms/sampling_likelihood_lab.py','scripts/generate_reasoning_records.py')


def parse(raw):
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('Duplicate figure input key')
            result[key]=value
        return result
    return json.loads(raw,object_pairs_hook=unique,
        parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite figure input')))


def regular(path,maximum=JSON_BYTES,*,keep=False):
    path=Path(path);fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        before=os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0<=before.st_size<=maximum:
            raise ValueError('Bounded regular figure input required')
        digest=hashlib.sha256();chunks=[];count=0
        while chunk:=os.read(fd,min(1024**2,maximum-count+1)):
            count+=len(chunk)
            if count>maximum:raise ValueError('Figure input byte ceiling crossed')
            digest.update(chunk)
            if keep:chunks.append(chunk)
        after=os.fstat(fd);current=path.lstat()
        fields=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        if count!=before.st_size or fields(before)!=fields(after) or fields(after)!=fields(current):
            raise ValueError('Figure input changed while reading')
    finally:os.close(fd)
    return b''.join(chunks),dict(path=str(path),bytes=count,sha256=digest.hexdigest())


class Inputs:
    def __init__(self):self.bindings={};self.maximums={}
    def read(self,path,maximum=JSON_BYTES,*,keep=True):
        raw,binding=regular(path,maximum,keep=keep);key=str(path)
        if key in self.bindings and self.bindings[key]!=binding:raise ValueError('Figure input changed between reads')
        self.bindings[key]=binding;self.maximums[key]=maximum;return raw
    def json(self,path):return parse(self.read(path))
    def rows(self,path):
        raw=self.read(path)
        if not raw or not raw.endswith(b'\n'):raise ValueError('Complete nonempty raw JSONL required')
        rows=[parse(line) for line in raw.splitlines()]
        if any(type(row) is not dict for row in rows):raise ValueError('Raw JSONL objects required')
        return rows
    def bound(self,binding,expected=None,*,executable=False):
        if type(binding) is not dict or not {'path','bytes','sha256'}<=set(binding):
            raise ValueError('Actual byte binding required')
        path=Path(binding['path'])
        if not path.is_absolute() or (expected is not None and path!=Path(expected)):
            raise ValueError('Closed input binding path changed')
        if not path.is_relative_to(ROOT) and not executable and path!=Path(LOCK):
            raise ValueError('Input binding leaves fixed repository/lock')
        actual=path
        if executable:
            actual=path.resolve(strict=True)
            if str(actual)!=binding.get('actual_path',str(path)):raise ValueError('Interpreter link identity changed')
        self.read(actual,EVENT_BYTES,keep=False)
        observed=self.bindings[str(actual)]
        if (observed['bytes'],observed['sha256'])!=(binding['bytes'],binding['sha256']):
            raise ValueError('Actual source/input/producer metadata bytes changed')
    def verify(self):
        for path,binding in self.bindings.items():
            if regular(path,self.maximums[path])[1]!=binding:raise ValueError('Figure input changed during rendering')
        return dict(self.bindings)


def require(condition,message):
    if not condition:raise ValueError(message)


def fixed_paths():return ROOT/'experiments/reports'/STEM,ROOT/'outputs'/STEM


def producer_metadata(record,inputs):
    producers=record['producers'];parent=producers['unchanged']['parent']
    full=ROOT/'outputs/native-sft-full-pilot400-20261005-run-01/policy'
    require(set(producers)==set(ROLES) and producers['unchanged']['path']==str(full)
        and producers['unchanged']['files']==parent['files'] and parent['path']==str(full),
        'Exactly the selected actual full400/chosen100/DPO100 producers required')
    full_receipt=ROOT/'experiments/reports/native-sft-full-pilot400-20261005-run-01/acceptance.json'
    inputs.bound(parent['acceptance'],full_receipt);accepted=inputs.json(full_receipt)
    full_checks=('completed400','exactly400_metrics','original_encoded_geometry','no_open_or_failed_work',
        'fresh_pinned_base','five_completed_snapshots','export_kind','loadable_full_policy')
    exported=accepted.get('exported_policy',{})
    require(accepted.get('status')=='passed' and accepted.get('checks')
        and all(value is True for value in accepted['checks'].values())
        and all(accepted['checks'].get(key) is True for key in full_checks)
        and accepted.get('result',{}).get('status')=='completed' and accepted['result'].get('updates')==400
        and type(accepted.get('actual_exit_code')) is int and accepted['actual_exit_code']==0
        and exported.get('path')==str(full) and exported.get('files')==parent['files']
        and exported.get('genealogy')==parent.get('genealogy')==inputs.json(full/'course-genealogy.json'),
        'Actual accepted400 parent inventory required; no historical producer-source rewrite')
    for role,kind in (('chosen100','chosen'),('dpo100','dpo')):
        item=producers[role];directory=ROOT/'experiments/reports'/f'native-{kind}-pilot-20261005-run-01'
        policy=ROOT/'outputs'/f'native-{kind}-pilot-20261005-run-01-pilot/policy'
        require(item['path']==str(policy) and item['parent']==parent,'Matched selected-parent final100 export required')
        for field,path in (('acceptance',directory/'acceptance.json'),('preparation',directory/'preparation.json'),
                ('actual_result',policy.parent/'result.json'),('actual_metrics',policy.parent/'metrics.jsonl')):
            inputs.bound(item[field],path)
        acceptance=inputs.json(directory/'acceptance.json');prepared=inputs.json(directory/'preparation.json')
        result=inputs.json(policy.parent/'result.json');metrics=inputs.rows(policy.parent/'metrics.jsonl')
        calls=acceptance.get('invocations',[])
        required=('all_actual_children_completed','completed100','exactly100_metrics','clean_journals')
        if kind=='chosen':required+=('unchanged_reference',)
        require(acceptance.get('status')=='passed' and acceptance.get('checks')
            and all(value is True for value in acceptance['checks'].values())
            and all(acceptance['checks'].get(key) is True for key in required)
            and acceptance.get('parent_binding')==parent and len(calls)==1 and calls[0].get('role')=='pilot'
            and calls[0].get('result',{}).get('status')=='completed'
            and type(calls[0]['result'].get('actual_exit_code')) is int and calls[0]['result']['actual_exit_code']==0
            and acceptance.get('exports',{}).get('pilot',{}).get('path')==str(policy)
            and acceptance['exports']['pilot'].get('files')==item['files']
            and acceptance.get('results',{}).get('pilot')==result
            and prepared.get('parent_binding')==parent and prepared.get('mode')=='pilot' and prepared.get('run_id')==RUN_ID
            and prepared.get('limits',{}).get('updates')==100
            and [row.get('update') for row in metrics]==list(range(1,101))
            and result.get('completed_recovery',{}).get('completed_updates')==100
            and result.get('reference_has_gradients') is False,'Actual accepted complete100 producer metadata required')
        genealogy=inputs.json(policy/'course-genealogy.json')
        require(genealogy==item['genealogy'] and genealogy.get('parent_checkpoint')==str(full)
            and genealogy.get('parent_checkpoint_sha256')==parent['files']
            and genealogy.get('completed_recovery')==result['completed_recovery'],
            'Actual exported100 genealogy changed')


def generation_join(record,role,panel,contract,items,inputs):
    evidence,output=fixed_paths();folder=output/role/panel
    raw=inputs.rows(folder/'responses.jsonl');summary=inputs.json(folder/'summary.json')
    identity=inputs.json(folder/'input-identity.json');producer=record['producers'][role]
    tokenizer={name:sha for name,sha in producer['files'].items() if any(fnmatch(name,p) for p in TOKENIZER_PATTERNS)}
    interface=contract['settings']['generation']['interface']['interface_sha256']
    checkpoint='local-hf-sha256:'+canonical_hash(dict(files=producer['files'],tokenizer=tokenizer,interface=interface))
    input_map={str(evidence.relative_to(ROOT)/(panel+'-'+kind+'.json')):record['input_bindings'][panel+'/'+kind]['sha256']
        for kind in ('items','contract')}
    lock=record['input_bindings']['environment_lock']
    environment=identity.get('environment',{});selected_lock=environment.get('environment_lock',{})
    require(identity.get('schema_version')==1
        and identity.get('identity_sha256')==canonical_hash({k:v for k,v in identity.items() if k!='identity_sha256'})
        and identity.get('checkpoint_path')==producer['path'] and identity.get('checkpoint_files')==producer['files']
        and identity.get('selected_tokenizer_path')==producer['path'] and identity.get('selected_tokenizer_files')==tokenizer
        and identity.get('input_sha256')==input_map and identity.get('config')==contract['settings']
        and identity.get('checkpoint_interface')==contract['settings']['generation']['interface']
        and identity.get('source_sha256')=={key:record['source_bindings'][key]['sha256'] for key in GENERATION_SOURCES}
        and identity.get('device',{}).get('mode')==contract['settings']['generation']['device']
        and identity.get('command')==record['commands'][role]
        and environment.get('interpreter')==record['input_bindings']['interpreter']['path']
        and selected_lock.get('status')=='hashed' and selected_lock.get('path')==str(Path(lock['path']).resolve())
        and selected_lock.get('sha256')==lock['sha256']
        and summary.get('status')=='completed' and summary.get('record_count')==COUNTS[panel]
        and summary.get('contract_id')==contract['identity'] and summary.get('checkpoint_id')==checkpoint
        and summary.get('identity_sha256')==identity['identity_sha256'] and summary.get('inputs_unchanged') is True,
        'Complete selected-policy generation identity/summary join required')
    require(len(raw)==len(items)==COUNTS[panel] and [row['item_id'] for row in raw]==[item['id'] for item in items],
        'Complete ordered frozen response population required')
    for row in raw:
        require(row.get('checkpoint_id')==checkpoint and row.get('sample_index')==0
            and row.get('input_identity_sha256')==identity['identity_sha256'] and row.get('input_sha256')==input_map
            and row.get('error') is None,'Raw response does not join accepted selected policy')
    graded=replay_records(items,raw,contract)['rows'];mapped={row['item_id']:row for row in graded}
    projections={item['id']:dict(item_id=item['id'],source_group=item['source_group'],task=item['task'],
        correct=(mapped[item['id']]['correct'] if panel=='reasoning' else
            mapped[item['id']]['response_text'].strip()==item['reference']),
        status=mapped[item['id']]['status'],natural_stop=mapped[item['id']]['natural_termination'],
        format_valid=mapped[item['id']]['format_valid'],truncated=mapped[item['id']]['truncated'],
        cost=mapped[item['id']]['cost']) for item in items}
    inputs.read(folder/'events.jsonl',EVENT_BYTES,keep=False)
    return projections,dict(planned=COUNTS[panel],completed=len(raw),missing=0,
        correct=sum(row['correct'] is True for row in projections.values()),
        natural_stops=sum(row['natural_stop'] is True for row in projections.values()),
        truncated=sum(row['truncated'] is True for row in projections.values()))


def collect(inputs):
    evidence,output=fixed_paths();prepared=inputs.json(evidence/'preparation.json')
    comparison=inputs.json(evidence/'comparison.json');accepted=inputs.json(evidence/'acceptance.json')
    require(not (evidence/'failure.json').exists() and accepted.get('status')=='passed'
        and accepted.get('checks') and all(v is True for v in accepted['checks'].values())
        and comparison.get('checks')==accepted['checks'] and not comparison.get('unreadable_jsonl'),
        'Actual passed complete common comparison required')
    required_checks={'no_unreadable_jsonl'}
    for role in ROLES:
        required_checks.update(role+'/'+key for key in ('all_validation_pairs','frozen_validation_groups',
            'pair_reference_no_grad','pair_completed_and_costed','child_complete'))
        required_checks.update(role+'/'+panel+'/'+key for panel in COUNTS for key in
            ('coverage','no_response_errors','partial_identity_join','generation_identity_complete'))
    require(required_checks<=set(accepted['checks']),'All original common-comparison checks required')
    require(prepared.get('schema')=='dongxi-fixed-native-preference-evaluation-v1' and prepared.get('run_id')==RUN_ID
        and prepared.get('preparation_sha256')==canonical_hash({k:v for k,v in prepared.items() if k!='preparation_sha256'})
        and prepared.get('locations')==dict(evidence=str(evidence),output=str(output))
        and prepared.get('precision')==accepted.get('precision')==PRECISION
        and accepted.get('producers')==prepared.get('producers'), 'Fixed common preparation/producers/precision required')
    require(set(prepared['interfaces'])==set(ROLES) and len({v['interface_sha256'] for v in prepared['interfaces'].values()})==1,
        'Same frozen selected-parent interface required for all three arms')
    required={*GENERATION_SOURCES,'scripts/run_native_preference_evaluation.py'}
    require(required<=set(prepared['source_bindings']),'Actual declared evaluation source bindings required')
    for name,binding in prepared['source_bindings'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Closed producer source name required')
        inputs.bound(binding,ROOT/name)
    require(prepared['input_bindings']['interpreter']['path']==PYTHON
        and prepared['input_bindings']['environment_lock']['path']==LOCK,'Original interpreter/lock roles required')
    fixed_inputs={name:ROOT/'fixtures/chapter11'/(name+'.jsonl') for name in ('train','validation','evaluation')}
    fixed_inputs.update(protocol=ROOT/'fixtures/matched-chosen-sft/protocol.json',
        template=ROOT/'experiments/data/instruction_interface_v1.jinja',
        math_items=ROOT/'fixtures/reasoning-controls/math_items.json',
        assistant_card=ROOT/'experiments/data/instruction-interface-v1-data-card.json')
    fixed_inputs.update({f'assistant_{name}':ROOT/'outputs/course-sft-interface-v1'/(name+'.jsonl')
        for name in ('train','dev','test')})
    fixed_inputs.update({panel+'/'+kind:evidence/(panel+'-'+kind+'.json') for panel in COUNTS for kind in ('items','contract')})
    require(all(prepared['input_bindings'].get(name,{}).get('path')==str(path) for name,path in fixed_inputs.items()),
        'Original fixed data/template/panel input roles required')
    for name,binding in prepared['input_bindings'].items():inputs.bound(binding,executable=name=='interpreter')
    producer_metadata(prepared,inputs)
    invocations=accepted.get('invocations',[])
    require([row.get('producer') for row in invocations]==list(ROLES),'All three actual evaluation children required')
    limits=prepared['limits']
    require(limits==dict(seconds_each=900,reserve_bytes=25*GIB,context=512,output_cap=64,
        validation_pairs=4,validation_policy_calls=8,validation_reference_calls=8,
        validation_positions_each=sum(len(branch[0])-1 for pair in prepared['encoded_validation'] for branch in pair),
        planned_responses_each=144,maximum_emitted_tokens_each=144*64,
        retained_reader_bytes=dict(response_or_pair_jsonl=JSON_BYTES,event_jsonl=EVENT_BYTES,input_identity=JSON_BYTES)),
        'Unchanged common panel/caps required')
    pairs=comparison['validation_likelihood_and_reference_margin'];display={};panel_display={}
    require(set(pairs)==set(ROLES),'Exactly three actual likelihood arms required')
    expected_ids=[row['id'] for row in prepared['validation_rows']]
    require(len(expected_ids)==len(set(expected_ids))==4,'Exactly four original validation pairs required')
    require(inputs.rows(prepared['input_bindings']['validation']['path'])==prepared['validation_rows']
        and inputs.json(prepared['input_bindings']['protocol']['path'])['groups']==prepared['scenario_groups'],
        'Frozen original validation rows/scenario groups differ')
    for role,invocation in zip(ROLES,invocations):
        command=prepared['commands'][role];returned=inputs.json(evidence/(role+'-returned-supervision.json'))
        terminal=inputs.json(evidence/(role+'-supervision/result.json'));launch=inputs.json(evidence/(role+'-launch.json'))
        # The retained final record precedes the writer ACK/helper/queue cleanup.
        # Compare stable scientific/process fields, and require the later actual
        # returned ACK. Do not pretend legitimate post-write fields were saved.
        terminal_fields=('schema','status','child_command','child_pid','actual_exit_code','stop_reason','failure',
            'journal_error','cleanup_errors','actual_native_profile_executed','limits','minimum_sampled_available_bytes')
        require(command==[prepared['input_bindings']['interpreter']['path'],str(ROOT/'scripts/run_native_preference_evaluation.py'),
                '--run-id',RUN_ID,'--child',role]
            and launch.get('argv')==command and launch.get('preparation_sha256')==prepared['preparation_sha256']
            and all(terminal.get(key)==returned.get(key) for key in terminal_fields)
            and returned==invocation.get('result') and returned.get('status')=='completed'
            and returned.get('schema')=='dongxi-native-profile-watchdog-result-v1' and returned.get('final_record_retained') is True
            and type(returned.get('actual_exit_code')) is int and returned['actual_exit_code']==0
            and returned.get('actual_native_profile_executed') is True and type(returned.get('child_pid')) is int
            and returned['child_pid']>0 and returned.get('child_command')==command and not returned.get('cleanup_errors')
            and returned.get('limits',{}).get('external_seconds')==900 and returned['limits'].get('reserve_bytes')==25*GIB
            and returned.get('failure') is None and returned.get('journal_error') is None,
            'Actual completed exit0 returned/supervision/launch join required')
        raw=inputs.rows(output/role/'pair-scores.jsonl');summary=inputs.json(output/role/'pair-summary.json')
        costs=inputs.rows(output/role/'pair-cost-events.jsonl');child=inputs.json(output/role/'child-result.json')
        require(raw==pairs[role] and [row['id'] for row in raw]==expected_ids and summary.get('status')=='completed'
            and summary.get('pairs')==4 and summary.get('reference_has_gradients') is False
            and costs[-1].get('cost')==summary.get('cost') and comparison['latest_pair_costs'][role]==costs[-1]
            and child.get('status')=='completed' and child.get('producer')==role and child.get('planned_responses')==144,
            'Complete raw4 likelihood/child/cost join required')
        require(all(summary['cost'].get(network)==dict(attempted_calls=8,forward_calls=8,
            attempted_positions=limits['validation_positions_each'],forward_positions=limits['validation_positions_each'])
            for network in ('policy','reference')),'Original completed pair-forward geometry required')
        for index,row in enumerate(raw):
            for key in ('chosen_logp','rejected_logp','reference_relative_margin','reference_relative_logp_margin','loss'):
                require(type(row.get(key)) in (float,int) and math.isfinite(row[key]),'Finite raw likelihood required')
            pair=prepared['encoded_validation'][index]
            require(row.get('source_group')==prepared['scenario_groups'][row['id']] and row.get('beta')==.1
                and row['chosen_logp']<=0 and row['rejected_logp']<=0
                and row.get('chosen_targets')==sum(pair[0][1][1:]) and row.get('rejected_targets')==sum(pair[1][1][1:])
                and math.isclose(row['reference_relative_margin'],.1*row['reference_relative_logp_margin'],rel_tol=1e-6,abs_tol=1e-7),
                'Frozen pair group/mask/beta/logp geometry differs')
        display[role]={metric:sum(row[metric] for row in raw)/4 for metric in
            ('chosen_logp','rejected_logp','reference_relative_logp_margin')}
    for role in ROLES:
        for row,base in zip(pairs[role],pairs['unchanged']):
            expected=(row['chosen_logp']-row['rejected_logp'])-(base['chosen_logp']-base['rejected_logp'])
            require(math.isclose(row['reference_relative_logp_margin'],expected,rel_tol=1e-5,abs_tol=1e-4),
                'Unscaled margin does not join the common frozen full400 reference')
    for panel,count in COUNTS.items():
        items=inputs.json(evidence/(panel+'-items.json'));contract=inputs.json(evidence/(panel+'-contract.json'))
        settings=contract['settings'];generation=settings['generation']
        require(len(items)==count and contract['suite_sha256']==canonical_hash(items)
            and contract['identity']==canonical_hash({k:v for k,v in contract.items() if k!='identity'})
            and prepared['contracts'][panel]==contract['identity'] and settings['max_new_tokens']==64
            and settings['decoding']==dict(mode='greedy',seed=1010,temperature=1.,top_k=None,top_p=1.)
            and settings['template_id']=='original-selected-full400-preference-retention'
            and settings['thinking_mode']=='template-default' and generation['dtype']=='bfloat16'
            and generation['context_window']==512 and generation['input_mode']=='chat'
            and generation['template']==inputs.read(prepared['input_bindings']['template']['path']).decode('utf-8')
            and generation['interface']['interface_sha256']==prepared['interfaces']['unchanged']['interface_sha256'],
            'Same original common greedy64 saved-template contract required')
        observed=comparison['independent_panels'][panel];panel_display[panel]={}
        require(set(observed['rows'])==set(ROLES) and set(observed['summaries'])==set(ROLES),'All three matched retention arms required')
        for role in ROLES:
            rows,counts=generation_join(prepared,role,panel,contract,items,inputs)
            require(rows==observed['rows'][role] and not observed['incomplete_attempts'][role]
                and all(observed['summaries'][role].get(key)==value for key,value in counts.items()),
                'Displayed retention counts differ from complete actual raw rows')
            panel_display[panel][role]=counts
    return dict(likelihood=display,retention=panel_display,precision=PRECISION,producer_declared_scope=prepared['scope'],
        figure_label_scopes=FIGURE_LABELS,contracts=prepared['contracts'],producer_inventories=prepared['producers'],
        scope='Four validation pairs; location4, assistant120 and annotated reasoning20 remain separate. Greedy64 custom-template retention is not original32/128 native-Instruct publication. No pooled score, new bootstrap, FLOPs, quality selection or fresh weight hash.')


def render(data,directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=('#304A60','#B17332','#7A4869');x=list(range(3))
    fixture='AUTHORED CPU FIXTURE — not model results\n' if 'AUTHORED CPU FIXTURE' in data['producer_declared_scope'] else ''
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    try:
        for offset,key,label in ((-.16,'chosen_logp','Chosen'),(.16,'rejected_logp','Rejected')):
            axes[0].bar([v+offset for v in x],[data['likelihood'][role][key] for role in ROLES],.3,label=label)
        axes[0].set_ylabel('Mean log probability (nats / complete answer sequence)')
        axes[0].set_title('Absolute likelihood — exact4 validation pairs')
        axes[1].bar(x,[data['likelihood'][role]['reference_relative_logp_margin'] for role in ROLES],color=colors)
        axes[1].axhline(0,color='gray',linewidth=.8)
        axes[1].set_ylabel('Mean unscaled reference-relative logp margin (nats)')
        axes[1].set_title('(chosen − rejected) − frozen Full400 difference')
        for axis in axes:axis.set_xticks(x,LABELS,rotation=12);axis.grid(axis='y',alpha=.15)
        axes[0].legend(frameon=False)
        fig.suptitle(fixture+data['figure_label_scopes']['likelihood'],fontsize=9)
        fig.savefig(directory/'likelihood-and-margin.png',dpi=160)
    finally:plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    try:
        for axis,panel in zip(axes,('assistant','reasoning')):
            for offset,key,label,color in ((-.24,'correct','Correct','#304A60'),(0,'natural_stops','Natural stop','#658F71'),
                    (.24,'truncated','Capped','#B17332')):
                axis.bar([v+offset for v in x],[data['retention'][panel][role][key] for role in ROLES],.23,label=label,color=color)
            axis.set_xticks(x,LABELS,rotation=12);axis.set_ylim(0,COUNTS[panel]);axis.set_ylabel('Responses (counts, not independent populations)')
            axis.set_title('Assistant120: strict whole answers' if panel=='assistant' else 'Reasoning20: annotated bounded-parser diagnostic')
            axis.grid(axis='y',alpha=.15)
        axes[0].legend(frameon=False,fontsize=8)
        fig.suptitle(fixture+data['figure_label_scopes']['retention'],fontsize=9)
        fig.savefig(directory/'separate-retention-panels.png',dpi=160)
    finally:plt.close(fig)


def retain(path,value):
    with Path(path).open('x',encoding='utf-8') as handle:
        json.dump(value,handle,indent=2,ensure_ascii=False,allow_nan=False);handle.write('\n');handle.flush();os.fsync(handle.fileno())


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    output=parser.parse_args(argv).output
    require(output.is_absolute() and '..' not in output.parts and output.parent==ROOT/'experiments/reports'
        and not output.parent.is_symlink() and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,119}',output.name),
        'Exclusive direct report output directory required')
    if output.exists() or output.is_symlink():raise FileExistsError(output)
    output.mkdir(mode=0o700);inputs=Inputs()
    try:
        own=Path(__file__).resolve();inputs.read(own);inputs.read(own.parents[1]/'tests/test_native_preference_figure.py')
        inputs.read(own.parents[1]/'src/dongxi_llms/evaluation_lab.py')
        data=collect(inputs);before=dict(inputs.bindings);inputs.verify();render(data,output);after=inputs.verify()
        plots={name:regular(output/name)[1] for name in ('likelihood-and-margin.png','separate-retention-panels.png')}
        retain(output/'inputs.json',dict(status='completed',input_bindings_before=before,input_bindings_after=after,
            figures=plots,displayed_data=data,cpu_boundary='Read/hash metadata/raw journals and unchanged bounded scalar grading plus Matplotlib only; no model/weight/tokenizer/GPU/network; outside producer journals'))
        inputs.verify()
        retain(output/'acceptance.json',dict(status='passed',inputs_unchanged=True,receipt=regular(output/'inputs.json')[1],figures=plots))
    except BaseException as error:
        retain(output/'failure.json',dict(type=type(error).__name__,message=str(error),input_bindings=inputs.bindings,
            scope='Invalid/incomplete figure prefix retained; no producer mutation or replacement of failed/capped records'));raise
    print(output);return 0


if __name__=='__main__':raise SystemExit(main())
