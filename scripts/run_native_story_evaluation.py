#!/usr/bin/env python3
"""Generate the original fixed story panel from the two declared pilot arms.

Preparation reads/hashes local bytes but never loads weights. Execution uses one
fresh, externally supervised child per available predetermined checkpoint.
Missing weights/attempts remain missing cells. No quality ratings are produced.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
import run_native_story_stages as training
from dongxi_llms.native_profile_supervisor import _digest, _native_probe, _supervise
from dongxi_llms.run_identity import canonical_hash
from dongxi_llms.staged_campaign import evaluation_contracts
from dongxi_llms.story_rubric import UPDATES, DECODING, validate_contract, _record
from dongxi_llms.story_work_budget import (
    STORY_WORK_KEYS, StoryWorkBudget, story_work_contract, read_story_receipt, checked_payload,
    payload_handle)
from dongxi_llms.snapshot_io_budget import read_bounded_json

ARMS = ('control','half-lr')
PILOT_RUN_ID = '20261005-01'
PYTHON = training.PYTHON
EOS = 50256
CONTEXT = 1024
CAP = 256
NLL_WINDOWS = 64
SOURCES = ('scripts/run_native_story_evaluation.py','src/dongxi_llms/story_rubric.py')
# The actual97.95s profile produced807,234B of aggregate supervision metadata;
# this separate bounded role must accommodate the declared14400s pilot. It does
# not enlarge checkpoint receipts, snapshot metadata, model work or GPU budgets.
PRODUCER_CONTROL_BYTES = 256*1024**2
DECLARED_TRAINING_ATTENTION = 'pytorch-sdpa-auto-causal'
EVALUATION_EXECUTION = dict(attention_declaration='pytorch-sdpa-auto-causal',
    kernel_selection='automatic SDPA dispatch; not a MATH-only override or measured per-call kernel',
    checkpoint_storage='torch.float32',loaded_model_weights='torch.float32',
    forward_autocast='CUDA torch.bfloat16',likelihood_probability_math='torch.float64',
    activation_checkpointing=False,deterministic_entry_used=False)


def retain(path,value):
    training.retain(path,value)


def append(handle,value):
    handle.write(json.dumps(value,sort_keys=True,allow_nan=False)+'\n')
    handle.flush();os.fsync(handle.fileno())


def locations(run_id):
    if type(run_id) is not str or re.fullmatch(r'[a-z0-9][a-z0-9-]{0,47}',run_id) is None:
        raise ValueError('Bounded plain evaluation run ID required')
    name='native-story-publication-'+run_id
    return ROOT/'experiments/reports'/name,ROOT/'outputs'/name


def checkpoint_id(arm,update):
    if arm not in ARMS or type(update) is not int or update not in UPDATES:
        raise ValueError('Only fixed arms and predetermined checkpoints are permitted')
    return arm+'-update-'+format(update,'06d')


def source_bindings():
    return dict(training=training.bindings(), evaluation={name:_digest(str(ROOT/name)) for name in SOURCES})


def producer_control_bytes(path):
    """Separate bounded aggregate supervisor role, not snapshot bootstrap caps."""
    with payload_handle(path) as (handle,before):
        if not 0<before.st_size<=PRODUCER_CONTROL_BYTES:
            raise ValueError('Bounded nonempty producer control artifact required')
        raw=handle.read(PRODUCER_CONTROL_BYTES+1);after=os.fstat(handle.fileno())
        if (len(raw)!=before.st_size or len(raw)>PRODUCER_CONTROL_BYTES or
                (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns)):
            raise ValueError('Producer control bytes changed during inspection')
    return raw,dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def producer_json(path):
    raw,identity=producer_control_bytes(path)
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('Duplicate producer control key')
            result[key]=value
        return result
    value=json.loads(raw,object_pairs_hook=unique,
        parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite producer control')))
    if type(value) is not dict:raise ValueError('Producer control object required')
    return value,identity


def deterministic_producer_receipts(arm,parent_evidence,parent_output,bound):
    """Bind actual terminal producer evidence without demanding14k success.

    A completed independently receipted checkpoint from a capped/failed pilot
    remains usable. The parent invocation's negative outcome is never rewritten.
    """
    stage='story-'+arm+'-pilot'
    producer,prepared_identity=producer_json(parent_evidence/'preparation.json')
    if (producer.get('stage')!=stage or producer.get('bindings')!=bound['training'] or
            producer.get('run_id')!=PILOT_RUN_ID or producer.get('preparation_sha256')!=canonical_hash(
                {key:value for key,value in producer.items() if key!='preparation_sha256'})):
        raise ValueError('Actual fixed pilot preparation/source/input identity required')
    argv=training.command(stage,parent_evidence,parent_output,'pilot')
    if producer.get('commands')!={'pilot':argv}:
        raise ValueError('Actual fixed deterministic pilot command required')
    tranche=producer.get('execution_tranche',{})
    if (tranche.get('requested_stop')!=400 or tranche.get('total_horizon')!=14000 or
            tranche.get('full_schedule_completion') is not False):
        raise ValueError('Predeclared first400 tranche within original14000 horizon required')
    launch,launch_identity=producer_json(parent_evidence/'launch-pilot.json')
    if launch.get('argv')!=argv or launch.get('preparation_sha256')!=producer['preparation_sha256']:
        raise ValueError('Actual pilot launch differs from bound preparation')
    returned,returned_identity=producer_json(parent_evidence/'returned-supervision-pilot.json')
    final,final_identity=producer_json(parent_evidence/'supervision-pilot/result.json')
    terminal_fields=('schema','status','child_command','child_pid','actual_exit_code','stop_reason',
        'actual_native_profile_executed','limits','minimum_sampled_available_bytes')
    if (any(final.get(key)!=returned.get(key) for key in terminal_fields) or
            returned.get('schema')!='dongxi-native-profile-watchdog-result-v1' or
            returned.get('status') not in ('completed','failed') or
            returned.get('actual_native_profile_executed') is not True or
            returned.get('child_command')!=argv or type(returned.get('child_pid')) is not int or
            type(returned.get('actual_exit_code')) is not int or
            returned.get('limits',{}).get('external_seconds')!=14400 or
            returned['limits'].get('reserve_bytes')!=25*training.GIB):
        raise ValueError('Actual terminal owned pilot supervision required')
    raw,stdout_identity=producer_control_bytes(parent_evidence/'supervision-pilot/stdout.txt')
    first=raw.split(b'\n',1)[0]
    if len(first)>16384:raise ValueError('Bounded deterministic entry event required')
    entry=json.loads(first)
    if (type(entry) is not dict or set(entry)!={'event','torch_version','deterministic_algorithms',
            'cublas_workspace','attention_backend','tf32'} or
            entry['event']!='actual-story-deterministic-entry' or
            entry['deterministic_algorithms'] is not True or entry['cublas_workspace']!=':4096:8' or
            entry['attention_backend']!='SDPA MATH only' or entry['tf32'] is not False or
            type(entry['torch_version']) is not str or not entry['torch_version']):
        raise ValueError('Observed deterministic MATH producer entry required')
    files=dict(preparation=prepared_identity,launch=launch_identity,
        returned_supervision=returned_identity,terminal_supervision=final_identity,stdout=stdout_identity)
    completion=None
    if (parent_output/'pilot/completion.json').exists():
        actual,files['completion']=producer_json(parent_output/'pilot/completion.json')
        completion={key:actual.get(key) for key in ('completed_updates','cumulative_targets',
            'cumulative_processed_positions','requested_stop_reached','schedule_complete',
            'stopped_for_time_budget','stopped_for_target_budget')}
    return dict(files=files,
        declared_checkpoint_attention=DECLARED_TRAINING_ATTENTION,observed_entry=entry,
        declared_execution_tranche=tranche,actual_completion=completion,
        invocation=dict(status=returned['status'],actual_exit_code=returned['actual_exit_code'],
            stop_reason=returned['stop_reason'],child_pid=returned['child_pid']),
        scope='Actual producer terminal outcome; first400 tranche/available save is not whole14000 success')


def work_contract(prefix_lengths):
    if len(prefix_lengths)!=12 or any(type(n) is not int or not 1<=n<=CONTEXT-CAP for n in prefix_lengths):
        raise ValueError('Twelve actual prefixes must accommodate the complete frozen output cap')
    caps=dict.fromkeys(STORY_WORK_KEYS,0)
    caps.update(model_initializations=1,restore_operations=1,
        evaluation_panels=2,evaluation_windows=2*NLL_WINDOWS,
        evaluation_valid_targets=2*NLL_WINDOWS*CONTEXT,
        policy_forward_calls=2*(NLL_WINDOWS//16),policy_forward_positions=2*NLL_WINDOWS*CONTEXT,
        generation_sequences=48,generation_calls=48*CAP,generation_tokens=48*CAP,
        generation_positions=4*sum(CAP*n+CAP*(CAP-1)//2 for n in prefix_lengths),
        multinomial_draws=36*CAP)
    return story_work_contract(caps,8*1024**2)


def make_record(contract,plan,item,index,prompt_count,ids,model_logps,text,stop,error,cost):
    record=dict(record_id=plan['checkpoint_id']+'-'+item['id']+'-d'+str(index),
        contract_sha256=contract['logical_contract_sha256'],checkpoint_id=plan['checkpoint_id'],
        checkpoint_sha256=canonical_hash(plan),item_id=item['id'],source_group=item['source_group'],
        opening_sha256=hashlib.sha256(item['prompt'].encode()).hexdigest(),decoding_index=index,attempt_index=0,
        text=text,text_sha256=hashlib.sha256(text.encode()).hexdigest(),token_ids=ids,prompt_tokens=prompt_count,
        generated_tokens=len(ids),selected_likelihoods=model_logps,stop_reason=stop,
        truncated=stop in ('token-cap','context-cap'),error=error,cost=cost)
    _record(record,contract,{item['id']:item},{plan['checkpoint_id']:plan})
    return record


def token_distribution(logits,recipe,generator=None):
    """Float64 full-vocabulary probabilities; raw and behavior logp differ."""
    import torch
    if logits.ndim!=1 or logits.numel()!=50257 or not bool(torch.isfinite(logits).all()):
        raise ValueError('Finite complete GPT-2 vocabulary logits required')
    raw=torch.log_softmax(logits.double(),dim=-1)
    if recipe['mode']=='greedy':
        token=int(logits.argmax().item());behavior=0.
    else:
        log_behavior=torch.log_softmax(logits.double()/recipe['temperature'],dim=-1)
        probabilities=log_behavior.exp()
        if not bool(torch.isfinite(probabilities).all()) or not bool((probabilities>0).all()):
            raise ValueError('Floating-point full-support sampling is unrepresentable; no truncation fallback')
        token=int(torch.multinomial(probabilities,1,generator=generator).item())
        behavior=float(log_behavior[token].item())
    model=float(raw[token].item())
    if not math.isfinite(model) or not math.isfinite(behavior) or model>0 or behavior>0:
        raise ValueError('Selected model/behavior logp is invalid')
    return token,model,behavior


def prepare(run_id):
    evidence,output=locations(run_id)
    if evidence.exists() or output.exists() or evidence.is_symlink() or output.is_symlink():
        raise ValueError('Evaluation evidence/output must both be new')
    bound=source_bindings()
    contract=evaluation_contracts(ROOT)['story-publication-v1'];validate_contract(contract)
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(training.DATA/'tokenizer.json'))
    if tokenizer.get_vocab_size()!=50257 or tokenizer.token_to_id('<|endoftext|>')!=EOS:
        raise ValueError('Observed GPT-2 vocabulary/BOS/EOS mismatch')
    prefixes={item['id']:[EOS,*tokenizer.encode(item['prompt'],add_special_tokens=False).ids] for item in contract['items']}
    if any(EOS in ids[1:] for ids in prefixes.values()):raise ValueError('Opening contains a special EOS')
    caps=work_contract([len(ids) for ids in prefixes.values()])
    interface=dict(mode='raw-story-prefix',template=None,context=CONTEXT,output_cap=CAP,
        tokenizer_sha256=training.DATA_HASHES['tokenizer.json'],vocab=50257,bos_id=EOS,eos_id=EOS,
        observed_publication_prefixes=prefixes,logical_contract_sha256=contract['logical_contract_sha256'],
        evaluation_execution=EVALUATION_EXECUTION)
    evidence.mkdir(mode=0o700);output.mkdir(mode=0o700)
    retain(evidence/'contract.json',contract);retain(evidence/'interface.json',interface)
    retain(evidence/'work-caps.json',caps)
    plans=[];checkpoints={};producers={}
    for arm in ARMS:
        parent_evidence,parent_output=training.locations('story-'+arm+'-pilot',PILOT_RUN_ID)
        candidate_payloads=[parent_output/'pilot'/('update-'+format(update,'06d')+'.pt') for update in UPDATES]
        available=any(payload.exists() and Path(str(payload)+'.work.json').exists() for payload in candidate_payloads)
        producers[arm]=deterministic_producer_receipts(arm,parent_evidence,parent_output,bound) if available else None
        for update in UPDATES:
            cid=checkpoint_id(arm,update);payload=parent_output/'pilot'/('update-'+format(update,'06d')+'.pt')
            declaration=dict(checkpoint_id=cid,arm=arm,update=update,expected_payload=str(payload),
                status='missing-checkpoint',payload_identity=None,receipt_identity=None,producer_receipts_sha256=None,
                missing_reason='missing-payload-or-independent-completed-receipt')
            frozen=evidence/(cid+'-receipt.json')
            if payload.exists() and Path(str(payload)+'.work.json').exists():
                receipt=read_story_receipt(Path(str(payload)+'.work.json'))
                if receipt['completed_updates']!=update or receipt['payload_bytes']>2*training.GIB:
                    raise ValueError('Independent receipt differs from predetermined checkpoint')
                # Preserve independent metadata before any checkpoint body read.
                retain(frozen,receipt)
                actual=training.identity(payload)
                if (actual['sha256'],actual['bytes'])!=(receipt['payload_sha256'],receipt['payload_bytes']):
                    raise ValueError('Checkpoint bytes differ from independently retained receipt')
                declaration.update(status='available',payload_identity=actual,receipt_identity=training.identity(frozen),
                    producer_receipts_sha256=canonical_hash(producers[arm]),missing_reason=None)
            path=evidence/(cid+'-declaration.json');retain(path,declaration)
            files={path.name:training.identity(path)['sha256']}
            if declaration['status']=='available':
                files[payload.name]=declaration['payload_identity']['sha256']
                files[frozen.name]=declaration['receipt_identity']['sha256']
            plan=dict(checkpoint_id=cid,update=update,provenance='model-generated',checkpoint_files=files,
                interface_sha256=canonical_hash(interface),generation_source_sha256={name:value['sha256'] for name,value in bound['evaluation'].items()})
            plans.append(plan);checkpoints[cid]=declaration
    retain(evidence/'checkpoints.json',plans)
    record=dict(schema='dongxi-fixed-native-story-evaluation-v1',run_id=run_id,
        utc=datetime.now(timezone.utc).isoformat(),bindings=bound,contract=contract,interface=interface,
        caps=caps,producers=producers,checkpoint_plan=plans,checkpoints=checkpoints,
        execution_labels=dict(checkpoint_declared_training_attention=DECLARED_TRAINING_ATTENTION,
            actual_producer_entries={arm:None if value is None else value['observed_entry'] for arm,value in producers.items()},
            evaluation=EVALUATION_EXECUTION),
        files={name:training.identity(evidence/name) for name in ('contract.json','interface.json','work-caps.json','checkpoints.json')},
        ceilings=dict(checkpoints=10,external_seconds_each=900,all_generation_sequences=480,
            all_generation_tokens=122880,host_reserve_bytes=25*training.GIB,
            producer_control_bytes_each=PRODUCER_CONTROL_BYTES),
        scope='Original complete planned panel; missing payload declarations are not weights. No ratings, selected checkpoint, acquisition or whole-split NLL claim')
    record['preparation_sha256']=canonical_hash(record);retain(evidence/'preparation.json',record)
    return evidence,output,record


def assert_current(prepared,evidence):
    if source_bindings()!=prepared['bindings']:raise ValueError('Evaluation source/data/environment changed')
    for name,expected in prepared['files'].items():
        if training.identity(evidence/name)!=expected:raise ValueError('Prepared evaluation declaration changed')
    for producer in prepared.get('producers',{}).values():
        if producer is not None:
            for expected in producer['files'].values():
                if training.identity(expected['path'],maximum=PRODUCER_CONTROL_BYTES)!=expected:
                    raise ValueError('Actual producer receipt changed')
    for declaration in prepared['checkpoints'].values():
        if declaration['status']=='available':
            for expected in (declaration['payload_identity'],declaration['receipt_identity']):
                if training.identity(expected['path'])!=expected:raise ValueError('Bound checkpoint/receipt bytes changed')


def validate_checkpoint_state(state,declaration,prepared):
    import torch
    from dongxi_llms.stories_training import baseline
    expected=asdict(baseline(CONTEXT));contract=state.get('contract',{})
    recipe=contract.get('recipe',{});arm=declaration['arm']
    fixed=dict(total_updates=14000,warmup=200,peak_lr=1.5e-4 if arm=='half-lr' else 3e-4,
        floor_lr=1.5e-5 if arm=='half-lr' else 3e-5,microbatch=16,accumulation=1,
        seed=909,bf16=True,activation_checkpointing=True)
    if (type(state.get('step')) is not int or state.get('step')!=declaration['update'] or contract.get('model')!=expected or
            contract.get('data')!=training.DATA_HASHES['manifest.json'] or contract.get('device')!='cuda' or
            contract.get('torch')!=str(torch.__version__) or contract.get('attention')!=DECLARED_TRAINING_ATTENTION or
            any(recipe.get(k)!=v for k,v in fixed.items()) or contract.get('valid_target_budget')!=50_000_000):
        raise ValueError('Observed checkpoint does not implement the fixed fresh pilot arm/interface')
    expected_sources={'stories_training.py','stories_data.py','decoder_lab.py','pretraining_lab.py',
        'story_work_budget.py','work_budget.py','run_identity.py','snapshot_io_budget.py','train_stories.py'}
    if set(contract.get('implementation',{}))!=expected_sources:
        raise ValueError('Complete actual checkpoint producer implementation binding required')
    for name,expected_sha in contract.get('implementation',{}).items():
        source=ROOT/('scripts' if name=='train_stories.py' else 'src/dongxi_llms')/name
        if _digest(str(source))['sha256']!=expected_sha:raise ValueError('Checkpoint producer implementation changed')
    receipt=read_story_receipt(declaration['receipt_identity']['path'])
    if canonical_hash(contract)!=receipt['contract_sha256'] or state.get('story_work_prefix')!=receipt['work_prefix']:
        raise ValueError('Loaded payload differs from retained independent science/work receipt')
    producer=prepared.get('producers',{}).get(arm)
    if producer is not None and (contract.get('torch')!=producer['observed_entry']['torch_version'] or
            declaration.get('producer_receipts_sha256')!=canonical_hash(producer)):
        raise ValueError('Checkpoint differs from actual observed producer entry')


def fixed_nll(model,windows,seed,budget,device):
    import torch
    from torch.nn import functional as F
    ids=torch.randperm(len(windows),generator=torch.Generator().manual_seed(seed))[:NLL_WINDOWS].tolist()
    total=0.;targets=0;logits_dtypes=set()
    costs=dict(evaluation_panels=1,evaluation_windows=NLL_WINDOWS,evaluation_valid_targets=NLL_WINDOWS*CONTEXT,
        policy_forward_calls=NLL_WINDOWS//16,policy_forward_positions=NLL_WINDOWS*CONTEXT)
    with budget.operation(costs,'fixed-story-evaluation-nll') as work:
        work.enter(evaluation_panels=1)
        for start in range(0,len(ids),16):
            x,y=windows.batch(ids[start:start+16]);count=int((y!=-100).sum())
            units=dict(evaluation_windows=x.shape[0],evaluation_valid_targets=count,
                policy_forward_calls=1,policy_forward_positions=x.numel())
            work.enter(**units)
            with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):logits=model(x.to(device))
            logits_dtypes.add(str(logits.dtype))
            value=F.cross_entropy(logits.float().reshape(-1,50257),y.to(device).reshape(-1),ignore_index=-100,reduction='sum')
            if not bool(torch.isfinite(value)):raise ValueError('Nonfinite fixed-panel NLL')
            total+=float(value);targets+=count;work.done(**units)
        work.done(evaluation_panels=1)
    return dict(nll=total/targets,valid_targets=targets,windows=len(ids),selected_window_ids=ids,
        selection_seed=seed,physical_positions=len(ids)*CONTEXT,complete_prepared_split=False,
        observed_logits_dtypes=sorted(logits_dtypes),evaluation_execution=EVALUATION_EXECUTION,
        scope='Fixed matched target-weighted NLL, separate from generated story quality')


def child(run_id,cid):
    evidence,output=locations(run_id);prepared,_=read_bounded_json(evidence/'preparation.json')
    declaration=prepared['checkpoints'].get(cid)
    if declaration is None or declaration['status']!='available':raise ValueError('Only an available fixed checkpoint can execute')
    assert_current(prepared,evidence)
    root=output/cid;root.mkdir(mode=0o700)
    science=dict(story_work_budget=prepared['caps'],checkpoint=declaration,interface=prepared['interface'],
        generation_source=prepared['bindings']['evaluation'],nll=dict(train_seed=409,dev_seed=909,windows_each=NLL_WINDOWS))
    budget=StoryWorkBudget.create(root/'work.jsonl',contract=prepared['caps'],scientific_contract=science,
        invocation_id='publication-'+str(os.getpid()))
    import torch
    from dongxi_llms.stories_training import StoriesDecoder
    from dongxi_llms.decoder_lab import DecoderConfig
    from dongxi_llms.stories_data import Windows
    from tokenizers import Tokenizer
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise ValueError('Actual CUDA/BF16 is required')
        receipt=read_story_receipt(declaration['receipt_identity']['path'])
        with budget.operation(dict(restore_operations=1),'publication-checkpoint-inspection') as work:
            work.enter(restore_operations=1)
            with checked_payload(declaration['expected_payload'],receipt) as handle:state=torch.load(handle,map_location='cpu',weights_only=True)
            validate_checkpoint_state(state,declaration,prepared);work.done(restore_operations=1)
        with budget.operation(dict(model_initializations=1),'publication-exact-model-load') as work:
            work.enter(model_initializations=1)
            model=StoriesDecoder(DecoderConfig(**state['contract']['model']),activation_checkpointing=False)
            values=state['model']
            if not values or any(not isinstance(value,torch.Tensor) or value.dtype!=torch.float32 or
                    not bool(torch.isfinite(value).all()) for value in values.values()):
                raise ValueError('Finite actual FP32 checkpoint parameter storage required')
            model.load_state_dict(values,strict=True);model=model.cuda().eval()
            retain(root/'execution-identity.json',dict(
                checkpoint_declared_training_attention=state['contract']['attention'],
                checkpoint_declared_training_recipe_bf16=state['contract']['recipe']['bf16'],
                actual_training_entry=prepared['producers'][declaration['arm']]['observed_entry'],
                producer_receipts=prepared['producers'][declaration['arm']]['files'],
                evaluation=EVALUATION_EXECUTION,observed_checkpoint_parameter_dtypes=sorted({str(value.dtype) for value in values.values()}),
                observed_model_parameter_dtypes=sorted({str(value.dtype) for value in model.parameters()}),
                observed_evaluation_sdpa_flags=dict(flash=torch.backends.cuda.flash_sdp_enabled(),
                    memory_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),math=torch.backends.cuda.math_sdp_enabled()),
                observed_evaluation_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                torch_version=str(torch.__version__),scope='SDPA enable flags are not measured per-forward kernel selection'))
            del values,state
            work.done(model_initializations=1)
        tokenizer=Tokenizer.from_file(str(training.DATA/'tokenizer.json'))
        plan=next(p for p in prepared['checkpoint_plan'] if p['checkpoint_id']==cid)
        observations={}
        for split,seed in (('train',409),('valid',909)):
            observations[split]=fixed_nll(model,Windows(training.DATA,split),seed,budget,'cuda')
        retain(root/'fixed-nll.json',observations)
        logits_dtypes=set()
        with (root/'records.jsonl').open('x') as records,(root/'token-trace.jsonl').open('x') as trace:
            for item in prepared['contract']['items']:
                prefix=prepared['interface']['observed_publication_prefixes'][item['id']]
                for index,recipe in enumerate(DECODING):
                    rid=cid+'-'+item['id']+'-d'+str(index);began=time.monotonic()
                    ids=[];logps=[];behavior=[];attempted=forwarded=calls=0
                    append(trace,dict(record_id=rid,phase='start',item_id=item['id'],decoding_index=index,prompt_tokens=len(prefix),elapsed_seconds=0.))
                    costs=dict(generation_sequences=1,generation_calls=CAP,generation_tokens=CAP,
                        generation_positions=CAP*len(prefix)+CAP*(CAP-1)//2,
                        multinomial_draws=CAP if index else 0)
                    error=None;stop='token-cap'
                    try:
                        with budget.operation(costs,'publication-generation-'+rid) as work:
                            work.enter(generation_sequences=1)
                            generator=None if index==0 else torch.Generator(device='cuda').manual_seed(recipe['seed'])
                            current=torch.tensor([prefix],device='cuda')
                            for position in range(CAP):
                                n=current.numel();attempted+=n
                                work.enter(generation_calls=1,generation_positions=n)
                                append(trace,dict(record_id=rid,phase='entered-forward',positions=n,elapsed_seconds=time.monotonic()-began))
                                with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                                    logits=model.lm_head(model.features(current)[:,-1])[0]
                                logits_dtypes.add(str(logits.dtype))
                                torch.cuda.synchronize();forwarded+=n;calls+=1
                                work.done(generation_calls=1,generation_positions=n);work.enter(generation_tokens=1)
                                if index:work.enter(multinomial_draws=1)
                                token,logp,behavior_logp=token_distribution(logits,recipe,generator)
                                if index:work.done(multinomial_draws=1)
                                work.done(generation_tokens=1)
                                ids.append(token);logps.append(logp);behavior.append(behavior_logp)
                                append(trace,dict(record_id=rid,phase='token',token_id=token,model_logp=logp,
                                    behavior_logp=behavior_logp,observed_logits_dtype=str(logits.dtype),elapsed_seconds=time.monotonic()-began))
                                current=torch.cat((current,torch.tensor([[token]],device='cuda')),dim=1)
                                if token==EOS:stop='natural-eos';break
                            work.done(generation_sequences=1)
                    except Exception as failure:
                        error=type(failure).__name__+': '+str(failure)[:2048];stop='failure'
                        append(trace,dict(record_id=rid,phase='error',error=error,elapsed_seconds=time.monotonic()-began))
                        if ids and ids[-1]==EOS:stop='natural-eos';error=None
                    cost=dict(generation_tokens=len(ids),wall_seconds=time.monotonic()-began,prompt_tokens=len(prefix),
                        forward_positions=forwarded,attempted_forward_positions=attempted,forward_calls=calls)
                    record=make_record(prepared['contract'],plan,item,index,len(prefix),ids,logps,tokenizer.decode(ids),stop,error,cost)
                    append(records,record)
        retain(root/'child-result.json',dict(status='completed',checkpoint_id=cid,work=budget.ledger.snapshot(),
            evaluation_execution=EVALUATION_EXECUTION,observed_generation_logits_dtypes=sorted(logits_dtypes),
            cpu_pid=os.getpid(),cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            scope='Actual fixed publication generation and matched64-window train/dev NLL; no ratings'))
        return 0
    finally:budget.close()


def json_lines(path):
    if not path.exists():return []
    raw=path.read_bytes()
    if len(raw)>16*1024**2:raise ValueError('Bounded per-checkpoint raw records required')
    return [json.loads(line) for line in raw.splitlines(keepends=True) if line.endswith(b'\n')]


def record_file_identity(path):
    """An empty response file is measured missing evidence, not invalid bytes."""
    raw=path.read_bytes()
    if len(raw)>16*1024**2:raise ValueError('Combined response artifact exceeds its fixed byte envelope')
    return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def interrupted_records(prepared,cid,root,supervision):
    """Keep only actually started attempts; unattempted cells stay null."""
    records=json_lines(root/'records.jsonl');seen={r['record_id'] for r in records}
    traces={}
    for event in json_lines(root/'token-trace.jsonl'):traces.setdefault(event['record_id'],[]).append(event)
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(training.DATA/'tokenizer.json'))
    plan=next(p for p in prepared['checkpoint_plan'] if p['checkpoint_id']==cid)
    for rid,events in traces.items():
        if rid in seen:continue
        start=events[0]
        if start.get('phase')!='start':raise ValueError('Interrupted trace lacks initial actual geometry')
        item=next(i for i in prepared['contract']['items'] if i['id']==start['item_id'])
        tokens=[event for event in events if event['phase']=='token']
        ids=[event['token_id'] for event in tokens];logps=[event['model_logp'] for event in tokens]
        reason=supervision.get('stop_reason')
        stop='deadline' if reason=='external-deadline' else ('resource-stop' if reason=='host-reserve-below-threshold' else 'failure')
        error=None if stop!='failure' else 'Owned checkpoint child failed: '+str(reason or supervision.get('failure') or supervision.get('actual_exit_code'))
        if ids and ids[-1]==EOS:stop='natural-eos';error=None
        elif len(ids)==CAP:stop='token-cap';error=None
        entered=[event for event in events if event['phase']=='entered-forward']
        # A token confirms its forward returned; entered work may exceed it.
        cost=dict(generation_tokens=len(ids),prompt_tokens=start['prompt_tokens'],
            wall_seconds=max(event['elapsed_seconds'] for event in events),forward_calls=len(tokens),
            forward_positions=sum(event['positions'] for event in entered[:len(tokens)]),
            attempted_forward_positions=sum(event['positions'] for event in entered))
        records.append(make_record(prepared['contract'],plan,item,start['decoding_index'],start['prompt_tokens'],
            ids,logps,tokenizer.decode(ids),stop,error,cost))
    return records


def run(run_id,declaration):
    if type(declaration) is not str or not 12<=len(declaration)<=4096:raise ValueError('Retained operator scope required')
    evidence,output,prepared=prepare(run_id);results=[];records=[]
    try:
        for cid,checkpoint in prepared['checkpoints'].items():
            if checkpoint['status']!='available':
                results.append(dict(checkpoint_id=cid,status='missing-checkpoint',actual_exit_code=None));continue
            assert_current(prepared,evidence)
            argv=[PYTHON,str(Path(__file__).resolve()),'--run-id',run_id,'--checkpoint-child',cid]
            retain(evidence/(cid+'-launch.json'),dict(argv=argv,operator_declaration=declaration,
                preparation_sha256=prepared['preparation_sha256']))
            supervised=_supervise(argv,evidence/(cid+'-supervision'),native=True,seconds=900,
                probe=_native_probe,operator_declaration=declaration)
            retain(evidence/(cid+'-returned-supervision.json'),supervised)
            results.append(dict(checkpoint_id=cid,**supervised))
            records.extend(interrupted_records(prepared,cid,output/cid,supervised))
            assert_current(prepared,evidence)
        with (evidence/'records.jsonl').open('x') as handle:
            for record in records:append(handle,record)
        expected=[dict(checkpoint_id=plan['checkpoint_id'],item_id=item['id'],decoding_index=index,
            record_id=next((r['record_id'] for r in records if r['checkpoint_id']==plan['checkpoint_id'] and
                r['item_id']==item['id'] and r['decoding_index']==index),None))
            for plan in prepared['checkpoint_plan'] for item in prepared['contract']['items'] for index in range(4)]
        summary=dict(status='generated-awaiting-independent-ratings',expected_cells=expected,
            missing_cells=sum(row['record_id'] is None for row in expected),actual_records=len(records),
            generation_failures=sum(row['stop_reason']=='failure' for row in records),invocations=results,
            comparisons=[[checkpoint_id('control',update),checkpoint_id('half-lr',update)] for update in UPDATES],
            artifact_identities={name:(record_file_identity(evidence/name) if name=='records.jsonl' else training.identity(evidence/name))
                for name in ('contract.json','checkpoints.json','records.jsonl')},
            scope=prepared['scope'])
        retain(evidence/'coverage.json',summary)
        print(json.dumps(dict(status=summary['status'],actual_records=len(records),missing_cells=summary['missing_cells'],evidence=str(evidence))))
        return 0
    except BaseException as failure:
        retain(evidence/'failure.json',dict(type=type(failure).__name__,message=str(failure),
            completed_or_failed_checkpoint_ids=[r['checkpoint_id'] for r in results],scope='Raw outputs and failed attempts retained; no automatic retries'))
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--operator-declaration')
    parser.add_argument('--checkpoint-child',help=argparse.SUPPRESS)
    args=parser.parse_args(argv);locations(args.run_id)
    if args.checkpoint_child:
        if args.execute or args.operator_declaration:parser.error('Child mode cannot receive execution overrides')
        if args.checkpoint_child not in {checkpoint_id(arm,update) for arm in ARMS for update in UPDATES}:
            parser.error('Only ten fixed checkpoint children are allowed')
        return child(args.run_id,args.checkpoint_child)
    if args.execute:return run(args.run_id,args.operator_declaration)
    if args.operator_declaration:parser.error('Execution text requires --execute')
    evidence,_,prepared=prepare(args.run_id)
    print(json.dumps(dict(status='prepared-not-executed',evidence=str(evidence),preparation_sha256=prepared['preparation_sha256'])))
    return 0


if __name__=='__main__':raise SystemExit(main())
