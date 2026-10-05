#!/usr/bin/env python3
"""Archive only the fixed actual four-checkpoint story publication measurements.

Standard library only: no producer import, model/corpus loading, rating creation,
network, Torch, Git or GPU work. Existing inputs are read-only. --output must be
a new directory under experiments/reports; failures and successes never overwrite.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import stat

ROOT=Path(__file__).resolve().parents[1]
RUN_ID='20261005-01'
STEM='native-story-publication-'+RUN_ID
ARMS=('control','half-lr')
UPDATES=(0,400,4000,8000,14000)
AVAILABLE=tuple(f'{arm}-update-{update:06d}' for arm in ARMS for update in (0,400))
ALL_CHECKPOINTS=tuple(f'{arm}-update-{update:06d}' for arm in ARMS for update in UPDATES)
ORIGINAL_CONTRACT_SHA='96155a17e1b1cfe065f005ae61b2c640b5b172509eba7162404bd7737119e659'
INTERPRETER='/home/dongxi/dgx-spark-dongxi/.venv/bin/python'
LOCK='/home/dongxi/dgx-spark-dongxi/uv.lock'
MIB=1024**2
META_BYTES=16*MIB
SUPERVISION_BYTES=256*MIB
TOTAL_BYTES=256*MIB
ARCHIVE_SOURCES=('scripts/archive_native_story_publication.py','tests/test_native_story_publication_archive.py')
EVALUATION_SOURCES=('scripts/run_native_story_evaluation.py','src/dongxi_llms/story_rubric.py')
TRAINING_SOURCES=('scripts/train_stories.py','scripts/run_deterministic_story_child.py','scripts/run_native_story_stages.py',
    'src/dongxi_llms/stories_training.py','src/dongxi_llms/stories_data.py','src/dongxi_llms/decoder_lab.py',
    'src/dongxi_llms/pretraining_lab.py','src/dongxi_llms/story_work_budget.py','src/dongxi_llms/work_budget.py',
    'src/dongxi_llms/run_identity.py','src/dongxi_llms/snapshot_io_budget.py','src/dongxi_llms/native_profile_supervisor.py',
    'src/dongxi_llms/campaign_supervisor.py','src/dongxi_llms/staged_campaign.py',
    'experiments/specs/2026-10-04-staged-spark-campaign.md')
EXECUTION=dict(attention_declaration='pytorch-sdpa-auto-causal',
    kernel_selection='automatic SDPA dispatch; not a MATH-only override or measured per-call kernel',
    checkpoint_storage='torch.float32',loaded_model_weights='torch.float32',forward_autocast='CUDA torch.bfloat16',
    likelihood_probability_math='torch.float64',activation_checkpointing=False,deterministic_entry_used=False)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def require(condition,message):
    if not condition:raise ValueError(message)


def digest(path,maximum=META_BYTES):
    path=Path(path)
    require(not path.is_symlink(),'Archive inputs must be regular, not symlink replacements')
    with path.open('rb') as handle:
        before=os.fstat(handle.fileno());require(stat.S_ISREG(before.st_mode) and 0<before.st_size<=maximum,'Bounded nonempty archive input required')
        hashed=hashlib.sha256()
        for block in iter(lambda:handle.read(MIB),b''):hashed.update(block)
        after=os.fstat(handle.fileno())
    require((before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_size,after.st_mtime_ns,after.st_ctime_ns),
        'Archive input changed while hashing')
    return dict(path=str(path),bytes=before.st_size,sha256=hashed.hexdigest())


def decode(raw):
    def unique(pairs):
        value={}
        for key,item in pairs:
            require(key not in value,'Duplicate JSON field in actual evidence');value[key]=item
        return value
    return json.loads(raw,object_pairs_hook=unique,
        parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite JSON evidence')))


class Capture:
    def __init__(self):self.bindings={};self.total=0
    def bind(self,path,maximum=META_BYTES):
        path=Path(path);bound=digest(path,maximum);key=str(path)
        if key not in self.bindings:
            self.total+=bound['bytes'];require(self.total<=TOTAL_BYTES,'Aggregate archive input envelope exceeded')
            self.bindings[key]=dict(bound,maximum_bytes=maximum)
        else:require({k:v for k,v in self.bindings[key].items() if k!='maximum_bytes'}==bound,'Archive input changed between reads')
        return bound
    def raw(self,path,maximum=META_BYTES):
        before=self.bind(path,maximum);raw=Path(path).read_bytes()
        require(len(raw)==before['bytes'] and hashlib.sha256(raw).hexdigest()==before['sha256'],'Archive input changed during read')
        self.bind(path,maximum);return raw
    def json(self,path,maximum=META_BYTES):return decode(self.raw(path,maximum))
    def verify(self):
        for bound in self.bindings.values():
            require(digest(bound['path'],bound['maximum_bytes'])=={k:v for k,v in bound.items() if k!='maximum_bytes'},
                'Bound archive source/input bytes changed before publication')


def verify_sources(capture,prepared):
    bound=prepared['bindings'];training=bound['training'];evaluation=bound['evaluation']
    require(set(training['sources'])==set(TRAINING_SOURCES) and set(evaluation)==set(EVALUATION_SOURCES),'Original actual source roles changed')
    for group in (training['sources'],evaluation):
        for name,expected in group.items():
            path=ROOT/name;actual=capture.bind(path)
            require(expected['path']==str(path) and (actual['bytes'],actual['sha256'])==(expected['bytes'],expected['sha256']),
                'Actual producer/evaluation source changed: '+name)
    for key,path in (('interpreter',INTERPRETER),('environment_lock',LOCK)):
        expected=training[key];resolved=Path(path).resolve();actual=capture.bind(resolved,64*MIB)
        require(expected['path']==path and expected.get('actual_path',str(resolved))==str(resolved)
            and (actual['bytes'],actual['sha256'])==(expected['bytes'],expected['sha256']),'Actual interpreter/lock bytes changed')
    for name in ARCHIVE_SOURCES:capture.bind(ROOT/name)


def validate_nll(nll,child,matched):
    require(set(nll)=={'train','valid'},'Actual train/valid NLL documents required')
    for split,seed in (('train',409),('valid',909)):
        row=nll[split];ids=row['selected_window_ids']
        require(type(row['nll']) in (int,float) and math.isfinite(row['nll']) and row['nll']>=0,'Invalid finite target-weighted NLL')
        require(type(row['windows']) is int and row['windows']==64 and len(ids)==64 and len(set(ids))==64
            and all(type(value) is int and value>=0 for value in ids) and type(row['selection_seed']) is int and row['selection_seed']==seed
            and type(row['physical_positions']) is int and row['physical_positions']==65536 and row['complete_prepared_split'] is False
            and type(row['valid_targets']) is int and 1<=row['valid_targets']<=65536,'Original matched64-window NLL geometry required, not entire corpus')
        require(row['evaluation_execution']==EXECUTION and row['observed_logits_dtypes']==['torch.bfloat16'],
            'Original automatic-SDPA publication precision/interface changed')
        geometry={key:row[key] for key in ('selected_window_ids','selection_seed','valid_targets','physical_positions','windows')}
        if split in matched:require(matched[split]==geometry,'NLL windows/targets must match across all four checkpoints')
        else:matched[split]=geometry
    work=child['work'];require(not work['open_tickets'] and not work['failed_tickets'],'Failed/open actual publication work cannot be archived as complete')
    completed=work['completed']
    require(all(type(completed[key]) is int for key in ('evaluation_windows','evaluation_panels','evaluation_valid_targets',
        'policy_forward_calls','policy_forward_positions','generation_sequences')) and completed['evaluation_windows']==128 and completed['evaluation_panels']==2
        and completed['evaluation_valid_targets']==sum(nll[split]['valid_targets'] for split in nll)
        and completed['policy_forward_calls']==8 and completed['policy_forward_positions']==131072
        and completed['generation_sequences']==48,'Actual NLL/child measurement counters differ')


def collect(capture):
    evidence=ROOT/'experiments/reports'/STEM;output=ROOT/'outputs'/STEM
    prepared=capture.json(evidence/'preparation.json');coverage=capture.json(evidence/'coverage.json')
    plans=capture.json(evidence/'checkpoints.json');contract=capture.json(evidence/'contract.json');interface=capture.json(evidence/'interface.json')
    require(prepared['schema']=='dongxi-fixed-native-story-evaluation-v1' and prepared['run_id']==RUN_ID
        and prepared['preparation_sha256']==canonical_hash({k:v for k,v in prepared.items() if k!='preparation_sha256'}),'Actual fixed preparation identity changed')
    verify_sources(capture,prepared)
    require(contract==prepared['contract'] and contract['logical_contract_sha256']==ORIGINAL_CONTRACT_SHA
        and canonical_hash({k:v for k,v in contract.items() if k!='logical_contract_sha256'})==ORIGINAL_CONTRACT_SHA
        and interface==prepared['interface'] and interface['evaluation_execution']==EXECUTION
        and prepared['execution_labels']['evaluation']==EXECUTION,'Original story publication contract/interface changed')
    for name,expected in prepared['files'].items():
        require(name in ('contract.json','interface.json','work-caps.json','checkpoints.json') and capture.bind(evidence/name)==expected,
            'Actual prepared publication file bytes changed')
    require(set(prepared['files'])=={'contract.json','interface.json','work-caps.json','checkpoints.json'}
        and capture.json(evidence/'work-caps.json')==prepared['caps'],'Original prepared publication cap files required')
    for entry in prepared['execution_labels']['actual_producer_entries'].values():
        require(entry['event']=='actual-story-deterministic-entry' and entry['attention_backend']=='SDPA MATH only'
            and entry['deterministic_algorithms'] is True and entry['tf32'] is False,'Actual MATH-only training provenance required, separate from publication')
    require(len(contract['items'])==12 and len(contract['decoding'])==4 and contract['predetermined_updates']==list(UPDATES),
        'Original12x4x10 publication geometry required')
    require(plans==prepared['checkpoint_plan'] and [plan['checkpoint_id'] for plan in plans]==list(ALL_CHECKPOINTS),
        'Ten unique predetermined checkpoint plans required')
    declared=prepared['checkpoints'];require(set(declared)==set(ALL_CHECKPOINTS),'Actual checkpoint declarations changed')
    require(tuple(cid for cid in ALL_CHECKPOINTS if declared[cid]['status']=='available')==AVAILABLE,'Exactly four original available checkpoints required')
    for cid in ALL_CHECKPOINTS:
        path=evidence/(cid+'-declaration.json');declaration=capture.json(path)
        plan=next(row for row in plans if row['checkpoint_id']==cid)
        require(declaration==declared[cid] and capture.bind(path)['sha256']==plan['checkpoint_files'][path.name],
            'Actual checkpoint declaration changed')
    for name,expected in coverage['artifact_identities'].items():
        require(name in ('contract.json','checkpoints.json','records.jsonl') and capture.bind(evidence/name)==expected,
            'Actual coverage/checkpoint/record byte identity changed')
    require(set(coverage['artifact_identities'])=={'contract.json','checkpoints.json','records.jsonl'},'Original coverage hash roles required')
    raw=capture.raw(evidence/'records.jsonl');require(raw.endswith(b'\n'),'Incomplete actual record line')
    records=[decode(line) for line in raw.splitlines()];index={};items={item['id']:item for item in contract['items']};plan_map={plan['checkpoint_id']:plan for plan in plans}
    for row in records:
        cid=row['checkpoint_id'];item=items[row['item_id']];d=row['decoding_index'];rid=f'{cid}-{item["id"]}-d{d}'
        require(cid in AVAILABLE and type(d) is int and 0<=d<4 and row['attempt_index']==0 and rid==row['record_id'] and rid not in index,
            'Duplicate/unknown actual record or missing original cell')
        require(row['contract_sha256']==ORIGINAL_CONTRACT_SHA and row['checkpoint_sha256']==canonical_hash(plan_map[cid])
            and row['opening_sha256']==hashlib.sha256(item['prompt'].encode()).hexdigest() and row['source_group']==item['source_group']
            and row['text_sha256']==hashlib.sha256(row['text'].encode()).hexdigest(),'Actual record checkpoint/opening/text identity differs')
        require(row['error'] is None and row['stop_reason'] in ('natural-eos','token-cap') and row['generated_tokens']==len(row['token_ids'])
            and type(row['generated_tokens']) is int and all(type(token) is int and 0<=token<=50256 for token in row['token_ids'])
            and row['cost']['generation_tokens']==row['generated_tokens'],'Failed/invalid actual generation record')
        index[rid]=row
    expected=[dict(checkpoint_id=cid,item_id=item['id'],decoding_index=d,
        record_id=(f'{cid}-{item["id"]}-d{d}' if cid in AVAILABLE else None)) for cid in ALL_CHECKPOINTS for item in contract['items'] for d in range(4)]
    require(coverage['status']=='generated-awaiting-independent-ratings' and coverage['expected_cells']==expected
        and all(type(coverage[key]) is int for key in ('actual_records','missing_cells','generation_failures'))
        and len(records)==coverage['actual_records']==192 and coverage['missing_cells']==288 and coverage['generation_failures']==0
        and Counter(row['checkpoint_id'] for row in records)==Counter({cid:48 for cid in AVAILABLE}),'Actual4/192/288 coverage changed')
    invocations={row['checkpoint_id']:row for row in coverage['invocations']}
    require(len(coverage['invocations'])==10 and set(invocations)==set(ALL_CHECKPOINTS),'Duplicate/missing actual supervision coverage')
    measurements={};matched={}
    for cid in ALL_CHECKPOINTS:
        invocation=invocations[cid]
        if cid not in AVAILABLE:
            require(invocation['status']=='missing-checkpoint' and invocation['actual_exit_code'] is None,'Missing checkpoint falsely claimed executed');continue
        supervision=capture.json(evidence/(cid+'-returned-supervision.json'),SUPERVISION_BYTES)
        command=[INTERPRETER,str(ROOT/'scripts/run_native_story_evaluation.py'),'--run-id',RUN_ID,'--checkpoint-child',cid]
        require(invocation==dict(checkpoint_id=cid,**supervision) and supervision['status']=='completed' and supervision['actual_exit_code']==0
            and supervision['schema']=='dongxi-native-profile-watchdog-result-v1' and supervision['actual_native_profile_executed'] is True
            and supervision['child_command']==command and supervision['limits']['external_seconds']==900
            and supervision['limits']['reserve_bytes']==25*1024**3,'Actual completed publication supervisor receipt required')
        child=capture.json(output/cid/'child-result.json');nll=capture.json(output/cid/'fixed-nll.json')
        require(child['status']=='completed' and child['checkpoint_id']==cid and child['evaluation_execution']==EXECUTION,
            'Actual completed selected child measurements required')
        require(child['work']['limits']==prepared['caps']['limits'],'Actual child work limits differ from prepared caps')
        validate_nll(nll,child,matched)
        plan=plan_map[cid];require(plan['interface_sha256']==canonical_hash(interface)
            and plan['generation_source_sha256']=={name:prepared['bindings']['evaluation'][name]['sha256'] for name in EVALUATION_SOURCES},
            'Actual generation source/interface plan differs')
        receipt_path=evidence/(cid+'-receipt.json');receipt=capture.json(receipt_path)
        require(receipt['completed_updates']==declared[cid]['update'] and receipt['payload_sha256']==declared[cid]['payload_identity']['sha256']
            and receipt['payload_bytes']==declared[cid]['payload_identity']['bytes'],
            'Actual independent checkpoint receipt differs')
        require(capture.bind(receipt_path)==declared[cid]['receipt_identity']
            and capture.bind(receipt_path)['sha256']==plan['checkpoint_files'][receipt_path.name],'Actual checkpoint receipt byte identity changed')
        measurements[cid]=dict(fixed_nll=nll,child_result=child,returned_supervision=supervision,checkpoint_receipt=receipt)
    return dict(schema='dongxi-native-story-publication-archive-v1',status='archived-actual-available-slice',run_id=RUN_ID,
        actual_available_checkpoints=4,actual_records=192,planned_records=480,missing_records=288,
        raw=dict(preparation=prepared,coverage=coverage,checkpoints=plans,contract=contract,interface=interface,measurements=measurements),
        record_counts=dict(Counter(row['checkpoint_id'] for row in records)),matched_nll_geometry=matched,
        scopes=['Each NLL is target-weighted over64 fixed matched windows per train/valid split, not the entire corpus or generated-story quality',
            'Observed training producer entry is deterministic SDPA MATH; publication declares automatic SDPA dispatch, not measured per-call kernels',
            'Four available0/400 checkpoints and192 responses only; six later checkpoints/288 cells remain missing, original14000 horizon incomplete',
            'Raw NLL, child measurements and four returned supervision documents retained without rating creation, re-review or quality inference',
            'Archive freshly binds retained metadata/measurements/records/source bytes before and after; existing model/corpus receipt declarations are preserved without body rehash/loading'])


def archive(output):
    output=Path(os.path.abspath(output));reports=(ROOT/'experiments/reports').resolve()
    require(output.parent.resolve()==output.parent and reports in output.parents,'New explicit experiments/reports directory required')
    output.mkdir(mode=0o700);capture=Capture()
    try:
        report=collect(capture);capture.verify();report['input_bindings']=capture.bindings;report['archive_sha256']=canonical_hash(report)
        with (output/'archive.json').open('x',encoding='utf-8') as handle:
            json.dump(report,handle,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':'));handle.write('\n');handle.flush();os.fsync(handle.fileno())
        capture.verify()
        with (output/'acceptance.json').open('x',encoding='utf-8') as handle:
            json.dump(dict(status='passed',archive=digest(output/'archive.json',TOTAL_BYTES),archive_sha256=report['archive_sha256'],
                inputs_unchanged=True,scope='Metadata archival acceptance only; no new scientific completion or quality evidence'),handle,allow_nan=False)
        return report
    except BaseException as error:
        with (output/'failure.json').open('x',encoding='utf-8') as handle:
            json.dump(dict(status='failed',type=type(error).__name__,message=str(error),input_bindings=capture.bindings,
                scope='Original publication/source inputs untouched; any partial archive remains unaccepted'),handle,allow_nan=False)
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    args=parser.parse_args(argv);report=archive(args.output)
    print(json.dumps(dict(status=report['status'],archive=str(Path(args.output)/'archive.json'),archive_sha256=report['archive_sha256'])));return 0


if __name__=='__main__':raise SystemExit(main())
