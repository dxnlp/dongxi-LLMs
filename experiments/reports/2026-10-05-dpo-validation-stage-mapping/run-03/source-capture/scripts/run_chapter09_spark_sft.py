"""Bounded opt-in CUDA SFT runner for the Chapter 9 original instruction suite.

No execution on import. Revision must be an exact repository commit. See the
chapter lab for data generation, profile gates and evidence limits.
"""
import argparse
from contextlib import nullcontext
from copy import deepcopy
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import stat
import time
from uuid import uuid4

import torch
from torch.nn import functional as F
from dongxi_llms.run_identity import (IdentityJournal, artifact_hashes, cached_snapshot,
    assert_compatible, canonical_hash, collect_run_identity, prepare_model_snapshot, TOKENIZER_PATTERNS,
    tokenizer_interface, write_json)
from dongxi_llms.training_snapshot import inspect_snapshot, load_snapshot, save_snapshot
from dongxi_llms.work_budget import WorkLedger,WorkBudgetExceeded,validate_limits,MAX_JOURNAL_BYTES

FAILURE_OUTPUT=None
FAILURE_JOURNAL=None
ACTIVE_WORK_LEDGER=None


def _fixed_plain_equal(actual,expected):
    """Compare only the fixed observed metadata shape; no unbounded JSON hash."""
    if isinstance(expected,dict):
        if not isinstance(actual,dict) or len(actual)!=len(expected):return False
        return all(key in actual and _fixed_plain_equal(actual[key],value) for key,value in expected.items())
    if type(expected) in (list,tuple):
        return type(actual) is type(expected) and len(actual)==len(expected) and all(
            _fixed_plain_equal(first,second) for first,second in zip(actual,expected))
    if type(actual) is not type(expected):return False
    if type(actual) is str and len(actual)!=len(expected):return False
    if type(actual) is int and actual.bit_length()>63:return False
    return actual==expected


WORK_KEYS=('train_updates','sampled_examples','selector_steps','valid_targets',
    'logical_sequence_tokens','policy_forward_calls','policy_forward_positions',
    'evaluation_calls','evaluation_positions','generation_calls',
    'generation_position_upper_bound','generation_tokens',
    'recovery_validation_operations','recovery_history_rows','recovery_tensor_elements','recovery_rng_states')


def work_budget_contract(limits,max_bytes):
    if type(limits) is not dict or len(limits)!=len(WORK_KEYS) or set(limits)!=set(WORK_KEYS):
        raise ValueError('Exact sixteen-dimensional SFT v2 caps required; budgeted v1 is not migrated')
    validate_limits(limits)
    if type(max_bytes) is not int or not 1<=max_bytes<=MAX_JOURNAL_BYTES:
        raise ValueError('Exact SFT work dimensions and bounded journal bytes required')
    return dict(schema='dongxi-sft-logical-work-v2',limits=dict(limits),max_bytes=max_bytes,
        reservation='whole known operation, permanent/no refund',
        scope='logical input positions/calls and runner semantic validation; not generic load/tree/hash/serialization/FLOPs/physical quota')


def read_work_limits_file(path):
    """Initial bounded nonblocking regular-file read with no-follow ancestors."""
    path=Path(os.path.abspath(path));handles=[]
    try:
        parent=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW);handles.append(parent)
        for component in path.parent.parts[1:]:
            parent=os.open(component,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent);handles.append(parent)
        descriptor=os.open(path.name,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW,dir_fd=parent);handles.append(descriptor)
        before=os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):raise ValueError('SFT work limits require a regular no-follow file')
        if before.st_size>65536:raise ValueError('SFT work limits exceed64KiB')
        raw=os.pread(descriptor,65537,0);after=os.fstat(descriptor)
        if len(raw)>65536 or len(raw)!=before.st_size or (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):
            raise ValueError('SFT work limits changed during bounded read')
        return raw
    finally:
        for descriptor in reversed(handles):os.close(descriptor)


def verify_work_limits_file(path,expected_sha256):
    if type(expected_sha256) is not str or re.fullmatch('[0-9a-f]{64}',expected_sha256) is None:
        raise ValueError('Expected exact parsed SFT work-limit byte digest required')
    if hashlib.sha256(read_work_limits_file(path)).hexdigest()!=expected_sha256:
        raise ValueError('SFT parsed work-limit bytes changed before tokenizer/model work')


def bind_bounded_work_limits(identity,path,expected_sha256):
    """Bind the parsed cap role without passing its path to generic hashing."""
    if type(expected_sha256) is not str or re.fullmatch('[0-9a-f]{64}',expected_sha256) is None:
        raise ValueError('Expected exact parsed SFT work-limit byte digest required')
    raw=read_work_limits_file(path);observed=hashlib.sha256(raw).hexdigest()
    if observed!=expected_sha256:
        raise ValueError('SFT parsed work-limit bytes changed before tokenizer/model work')
    identity['bounded_work_limits']=dict(path=os.path.abspath(path),sha256=observed,bytes=len(raw),
        reader='bounded-nonblocking-no-follow-regular')
    identity['identity_sha256']=canonical_hash({key:value for key,value in identity.items() if key!='identity_sha256'})
    return identity


def verify_bounded_work_limits_identity(identity,path,expected_sha256):
    """Cap-only closing gate; never reopen this role through generic hashing."""
    recorded=identity.get('bounded_work_limits')
    current=bind_bounded_work_limits({},path,expected_sha256)['bounded_work_limits']
    if recorded!=current or identity.get('identity_sha256')!=canonical_hash(
            {key:value for key,value in identity.items() if key!='identity_sha256'}):
        raise ValueError('SFT bounded work-limit identity changed before closing')


class _WorkAttempt:
    def __init__(self,ledger,costs,operation):
        self.ledger=ledger;self.ticket=None
        self.entered={key:0 for key in WORK_KEYS};self.success=dict(self.entered)
        if ledger is not None:self.ticket=ledger.reserve(costs,operation=operation)

    def before(self,**costs):
        candidate=dict(self.entered)
        for key,value in costs.items():candidate[key]+=value
        if self.ledger is not None:self.ledger.assert_within(self.ticket,candidate)
        self.entered=candidate

    def after(self,**costs):
        for key,value in costs.items():self.success[key]+=value

    def complete(self):
        if self.ledger is not None:self.ledger.complete(self.ticket,self.success)

    def fail(self,error):
        if self.ledger is not None and not self.ledger.poisoned:
            self.ledger.fail(self.ticket,type(error).__name__+': '+str(error)[:470],
                known_actual=self.success,attempted=self.entered)

def available_gib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])/1024**2
    raise RuntimeError('Linux MemAvailable measurement unavailable')


def read_records(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not records or len({row['id'] for row in records}) != len(records):
        raise ValueError('Need nonempty unique-ID records')
    return records


def encode_record(tokenizer, record, max_length):
    messages = record['messages']
    if not messages or messages[-1]['role'] != 'assistant':
        raise ValueError('Training record needs a completed assistant answer')
    def render(value, generation=False):
        return tokenizer.apply_chat_template(value, tokenize=True, add_generation_prompt=generation,
                                             enable_thinking=False,return_dict=False)
    ids = render(messages)
    if len(ids) > max_length:
        raise ValueError(f"Overlength record {record['id']}: reject, never silently truncate")
    labels = [-100]*len(ids)
    for index, message in enumerate(messages):
        if message['role'] != 'assistant':
            continue
        prefix, through = render(messages[:index], True), render(messages[:index+1])
        # Refuse heuristic span labels if the tokenizer/template changes a prefix.
        if ids[:len(through)] != through or through[:len(prefix)] != prefix:
            raise ValueError('Template is not prefix-compatible; supply an audited template')
        labels[len(prefix):len(through)] = ids[len(prefix):len(through)]
    if not any(token != -100 for token in labels[1:]):
        raise ValueError('No surviving answer targets')
    return dict(id=record['id'], ids=ids, labels=labels)


def collate(records, pad_id, device):
    length=max(len(record['ids']) for record in records)
    ids=torch.full((len(records),length),pad_id,dtype=torch.long,device=device)
    labels=torch.full_like(ids,-100); attention=torch.zeros_like(ids)
    for row,record in enumerate(records):
        ids[row,:len(record['ids'])]=torch.tensor(record['ids'],device=device)
        labels[row,:len(record['ids'])]=torch.tensor(record['labels'],device=device)
        attention[row,:len(record['ids'])]=1
    return ids,labels,attention


def summed_loss(model,batch):
    ids,labels,attention=batch
    logits=model(input_ids=ids,attention_mask=attention,use_cache=False).logits[:,:-1].float()
    targets=labels[:,1:]
    return F.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets.reshape(-1),
                           reduction='sum',ignore_index=-100)


def stable_sft_contract(config):
    """Scientific identity, not invocation paths, deadline or checkpoint cadence."""
    operational = {'train','dev','template','environment_lock','resume','output',
                   'resume_contract',
                   'resume_sha256','resume_bytes','snapshot_max_bytes',
                   'runtime_seconds','reserve_gib','checkpoint_every','allow_download',
                   'work_limits','work_journal','work_journal_max_bytes'}
    return dict(kind='chapter09-sft-completed-v1',
                config={key:value for key,value in config.items() if key not in operational})


def observed_numerical_environment(environment):
    """Observed versions/platform, excluding operational interpreter/lock paths."""
    return {key:deepcopy(environment[key])
            for key in ('python_version','platform','machine','packages')}


def scientific_model_config(model):
    """Keep actual architecture/behavior, excluding only its operational location."""
    config=deepcopy(model.config.to_dict())
    config.pop('_name_or_path',None)
    return config


class SFTLoop:
    """The runner's actual completed-update path, with a CPU testable device boundary.

    No model acquisition or implicit checkpoint work. An interrupted update is
    poisoned until a durable completed snapshot is restored; it cannot be saved.
    """
    def __init__(self, model, optimizer, train, pad_id, *, microbatch, accumulation,
                 seed, device, autocast_dtype=None, guard=None, total_updates=2000):
        if any(type(value) is not int or value <= 0 for value in (microbatch,accumulation)):
            raise ValueError('Positive integer microbatch/accumulation required')
        if type(seed) is not int or type(pad_id) is not int or pad_id < 0 or not train:
            raise ValueError('Need an integer seed/pad and nonempty encoded records')
        if type(total_updates) is not int or not 1 <= total_updates <= 2000:
            raise ValueError('Fixed SFT horizon must lie within 1..2000 updates')
        self.model,self.optimizer=model,optimizer
        self.train=deepcopy(train)
        if len({row['id'] for row in train}) != len(train):
            raise ValueError('Encoded training IDs must be unique')
        for row in train:
            if (len(row['ids']) < 2 or len(row['ids']) != len(row['labels'])
                    or not any(token != -100 for token in row['labels'][1:])):
                raise ValueError('Each encoded record requires surviving shifted targets')
        self.pad_id,self.microbatch,self.accumulation=pad_id,microbatch,accumulation
        self.seed,self.device=seed,torch.device(device)
        self.total_updates=total_updates
        self.autocast_dtype=autocast_dtype
        self.guard=guard or (lambda:None)
        self.parameters=[parameter for parameter in model.parameters() if parameter.requires_grad]
        if not self.parameters:
            raise ValueError('No trainable policy parameters')
        self.parameter_names=[name for name,p in model.named_parameters() if p.requires_grad]
        names_by_id={id(p):name for name,p in model.named_parameters() if p.requires_grad}
        self.optimizer_parameters=[p for group in optimizer.param_groups for p in group['params']]
        if (len(self.optimizer_parameters)!=len(self.parameters)
                or {id(p) for p in self.optimizer_parameters}!={id(p) for p in self.parameters}):
            raise ValueError('Optimizer must cover each actual trainable parameter exactly once')
        optimizer_names=[[names_by_id[id(p)] for p in group['params']] for group in optimizer.param_groups]
        order=list(range(len(train))); random.Random(seed).shuffle(order)
        self.order=order
        self.cursor=self.update=self.targets=self.positions=0
        self.last_metric=None
        self.history=[]
        self.incomplete_update=False
        self.work_ledger=None
        # Ragged work repeats with the cyclic microbatch starting offset.
        # Prefix sums avoid O(updates^2) history validation on a real pilot.
        target_counts=[sum(token!=-100 for token in row['labels'][1:]) for row in self.train]
        self._work_prefix=[(0,0)]
        for batch in range(len(train)//math.gcd(len(train),microbatch)):
            indices=[self.order[(batch*microbatch+i)%len(train)] for i in range(microbatch)]
            previous=self._work_prefix[-1]
            self._work_prefix.append((previous[0]+sum(target_counts[i] for i in indices),
                                      previous[1]+microbatch*max(len(self.train[i]['ids']) for i in indices)))
        self.loop_contract=dict(encoded_records_sha256=canonical_hash(self.train),pad_id=pad_id,
            microbatch=microbatch,accumulation=accumulation,seed=seed,device=str(self.device),
            total_updates=total_updates,
            autocast_dtype=str(autocast_dtype),model_class=type(model).__name__,
            model_config_sha256=canonical_hash(scientific_model_config(model)),
            model_layout={key:[list(tensor.shape),str(tensor.dtype)]
                          for key,tensor in model.state_dict().items()},
            parameter_names=self.parameter_names,optimizer_parameter_names=optimizer_names)
        # Metadata captured at construction, not from untrusted saved lengths.
        # These are declared semantic-check slots, not FLOPs or serializer work.
        self._validation_model_elements={key:tensor.numel() for key,tensor in model.state_dict().items()}
        self._validation_adam_elements=[parameter.numel() for parameter in self.optimizer_parameters]
        original_groups=self.optimizer.state_dict()['param_groups']
        self._validation_group_layout=[(len(group),len(group['params'])) for group in original_groups]
        self._validation_cpu_rng_shape=torch.get_rng_state().shape
        self._validation_cuda_rng_shapes=[state.shape for state in torch.cuda.get_rng_state_all()] if self.device.type=='cuda' else []

    def autocast(self):
        return (nullcontext() if self.autocast_dtype is None else
                torch.autocast(self.device.type,dtype=self.autocast_dtype))

    def completed_update(self):
        if self.incomplete_update:
            raise RuntimeError('Interrupted update requires restore from a durable completed snapshot')
        if self.update>=self.total_updates:
            raise ValueError('Fixed SFT update horizon exhausted')
        # Observe the known cyclic window without advancing its active cursor.
        preview=[[self.train[self.order[(self.cursor+step*self.microbatch+i)%len(self.order)]]
                  for i in range(self.microbatch)] for step in range(self.accumulation)]
        costs=dict(train_updates=1,sampled_examples=self.microbatch*self.accumulation,
            selector_steps=self.microbatch*self.accumulation,
            valid_targets=sum(sum(value!=-100 for value in row['labels'][1:]) for rows in preview for row in rows),
            logical_sequence_tokens=sum(len(row['ids']) for rows in preview for row in rows),
            policy_forward_calls=self.accumulation,
            policy_forward_positions=sum(self.microbatch*max(len(row['ids']) for row in rows) for rows in preview))
        attempt=_WorkAttempt(self.work_ledger,costs,'sft-update')
        try:
            result=self._completed_update(attempt)
            attempt.after(train_updates=1);attempt.complete();return result
        except BaseException as error:
            # No failed partially backward/applied update can be saved or reused.
            self.incomplete_update=True;attempt.fail(error);raise

    def _completed_update(self,attempt):
        self.guard(); self.model.train(); self.incomplete_update=True
        window=[]
        for _ in range(self.accumulation):
            attempt.before(sampled_examples=self.microbatch,selector_steps=self.microbatch)
            records=[self.train[self.order[(self.cursor+i)%len(self.order)]] for i in range(self.microbatch)]
            self.cursor+=self.microbatch
            attempt.after(sampled_examples=self.microbatch,selector_steps=self.microbatch)
            metadata=dict(valid_targets=sum(sum(value!=-100 for value in row['labels'][1:]) for row in records),
                          logical_sequence_tokens=sum(len(row['ids']) for row in records))
            attempt.before(**metadata);attempt.after(**metadata)
            window.append(collate(records,self.pad_id,self.device))
        count=sum(int((batch[1][:,1:]!=-100).sum()) for batch in window)
        positions=sum(batch[0].numel() for batch in window)
        if count <= 0:
            raise ValueError('No surviving answer targets in update')
        self.optimizer.zero_grad(set_to_none=True); recorded_sum=0.
        for batch in window:
            self.guard()
            geometry=dict(policy_forward_calls=1,policy_forward_positions=batch[0].numel())
            attempt.before(**geometry)
            with self.autocast():
                loss=summed_loss(self.model,batch)
            attempt.after(**geometry)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            (loss/count).backward(); recorded_sum+=float(loss.detach())
        norm=torch.nn.utils.clip_grad_norm_(self.parameters,1.)
        if not torch.isfinite(norm):
            raise RuntimeError('Nonfinite accumulated gradient')
        attempt.before(train_updates=1);self.optimizer.step()
        if not all(bool(torch.isfinite(parameter).all()) for parameter in self.parameters):
            raise RuntimeError('Nonfinite updated policy')
        self.update+=1; self.targets+=count; self.positions+=positions
        self.last_metric=dict(update=self.update,answer_nll=recorded_sum/count,targets=count,
            gradient_norm=float(norm),cursor=self.cursor,cumulative_supervised_targets=self.targets,
            processed_positions=positions,cumulative_processed_positions=self.positions)
        self.history.append(deepcopy(self.last_metric))
        self.incomplete_update=False
        return deepcopy(self.last_metric)

    def _expected_work(self, update):
        cursor=update*self.microbatch*self.accumulation
        cycles,remainder=divmod(update*self.accumulation,len(self._work_prefix)-1)
        cycle,tail=self._work_prefix[-1],self._work_prefix[remainder]
        return cursor,cycles*cycle[0]+tail[0],cycles*cycle[1]+tail[1]

    def snapshot_state(self):
        if self.incomplete_update:
            raise RuntimeError('Cannot snapshot an interrupted or partially applied update')
        result=dict(model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),
            loop_contract=self.loop_contract,order=list(self.order),cursor=self.cursor,
            targets=self.targets,positions=self.positions,metric=deepcopy(self.last_metric),
            history=deepcopy(self.history),
            model_training=self.model.training,
            rng=dict(python=random.getstate(),torch=torch.get_rng_state(),
                     cuda=torch.cuda.get_rng_state_all() if self.device.type=='cuda' else []))
        if self.work_ledger is not None:result['work_ledger']=self.work_ledger.snapshot()
        return result

    def bind_work_ledger(self,ledger,contract):
        budget=contract['config'].get('work_budget')
        if not isinstance(ledger,WorkLedger) or budget is None or ledger.limits!=budget['limits'] or ledger.max_bytes!=budget['max_bytes'] or ledger.contract_sha256!=canonical_hash(contract):
            raise ValueError('SFT work ledger must bind the exact scientific contract')
        if budget!=work_budget_contract(budget['limits'],budget['max_bytes']):raise ValueError('Unsupported SFT work schema')
        self.work_ledger=ledger

    def validate_payload(self,payload):
        if type(payload) is not dict or payload.get('phase') != 'completed':
            raise ValueError('SFT accepts only completed-update snapshots')
        state=payload.get('state'); update=payload.get('completed_updates')
        if type(update) is not int or not 0 <= update <= self.total_updates:
            raise ValueError('Invalid completed SFT update')
        if type(state) is not dict:
            raise ValueError('Plain bounded SFT state required')
        expected_keys={'model','optimizer','loop_contract','order','cursor','targets','positions',
                       'metric','history','model_training','rng'}
        contract=payload.get('contract',{})
        if type(contract) is not dict or type(contract.get('config',{})) is not dict:
            raise ValueError('Plain SFT scientific contract required')
        if contract.get('config',{}).get('work_budget') is not None:
            expected_keys.add('work_ledger')
            if self.work_ledger is None:raise ValueError('Budgeted SFT restore requires its active cumulative journal')
            self.bind_work_ledger(self.work_ledger,payload['contract'])
            self.work_ledger.validate_snapshot(state.get('work_ledger'))
        elif self.work_ledger is not None:raise ValueError('Cannot silently apply an unbudgeted SFT snapshot to a budgeted loop')
        if len(state)!=len(expected_keys) or set(state)!=expected_keys:
            raise ValueError('SFT loop/layout/order contract mismatch')
        # Bound hostile lengths without traversing history, tensor values or RNG.
        if (type(state['history']) is not list or len(state['history'])!=update
                or type(state['order']) is not list or len(state['order'])!=len(self.order)
                or type(state['loop_contract']) is not dict or len(state['loop_contract'])!=len(self.loop_contract)
                or not isinstance(state['model'],dict) or len(state['model'])!=len(self._validation_model_elements)
                or type(state['optimizer']) is not dict or len(state['optimizer'])!=2
                or set(state['optimizer'])!={'state','param_groups'}
                or type(state['optimizer']['state']) is not dict
                or len(state['optimizer']['state'])!=(len(self.parameters) if update else 0)
                or type(state['optimizer']['param_groups']) is not list
                or len(state['optimizer']['param_groups'])!=len(self._validation_group_layout)):
            raise ValueError('SFT semantic-validation container lengths differ from fixed layout')
        for group,(key_count,param_count) in zip(state['optimizer']['param_groups'],self._validation_group_layout):
            if (type(group) is not dict or len(group)!=key_count or type(group.get('params')) is not list
                    or len(group['params'])!=param_count):
                raise ValueError('SFT optimizer container lengths differ from fixed layout')
        rng=state['rng']
        if (type(rng) is not dict or len(rng)!=3 or set(rng)!={'python','torch','cuda'}
                or type(rng['python']) is not tuple or len(rng['python'])!=3
                or type(rng['python'][1]) is not tuple or len(rng['python'][1])!=625
                or type(rng['cuda']) is not list or len(rng['cuda'])!=len(self._validation_cuda_rng_shapes)):
            raise ValueError('SFT RNG container lengths differ from fixed layout')
        costs=dict(recovery_validation_operations=1,recovery_history_rows=update,
            recovery_tensor_elements=sum(self._validation_model_elements.values())
                +(sum(1+2*count for count in self._validation_adam_elements) if update else 0)
                +self._validation_cpu_rng_shape.numel()+sum(shape.numel() for shape in self._validation_cuda_rng_shapes),
            recovery_rng_states=2+len(self._validation_cuda_rng_shapes))
        attempt=_WorkAttempt(self.work_ledger,costs,'sft-runner-semantic-validation')
        try:
            attempt.before(recovery_validation_operations=1)
            self._validate_payload_semantics(payload,attempt)
            attempt.after(recovery_validation_operations=1);attempt.complete()
        except BaseException as error:
            attempt.fail(error);raise

    def _validate_payload_semantics(self,payload,attempt):
        state=payload['state'];update=payload['completed_updates']
        if (not _fixed_plain_equal(state['loop_contract'],self.loop_contract)
                or type(state['order']) is not list
                or any(type(value) is not int for value in state['order'])
                or state['order']!=self.order):
            raise ValueError('SFT loop/layout/order contract mismatch')
        if type(update) is not int or not 0 <= update <= self.total_updates:
            raise ValueError('Invalid completed SFT update')
        expected=self._expected_work(update)
        actual=(state['cursor'],state['targets'],state['positions'])
        if any(type(value) is not int for value in actual) or actual!=expected:
            raise ValueError('SFT update/cursor/target/position counters disagree')
        metric=state['metric']
        history=state['history']
        if type(history) is not list or len(history)!=update:
            raise ValueError('Completed SFT history length differs from update count')
        if update==0:
            if metric is not None:
                raise ValueError('Initial SFT snapshot must not invent a completed metric')
        else:
            self._bounded_metric(metric)
        for number,row in enumerate(history,1):
            attempt.before(recovery_history_rows=1)
            self._bounded_metric(row)
            before=self._expected_work(number-1);after=self._expected_work(number)
            if (type(row) is not dict or set(row)!=set(self._metric_keys())
                    or any(type(row[key]) is not int for key in self._metric_keys()
                           if key not in ('answer_nll','gradient_norm'))
                    or any(type(row[key]) is not float for key in ('answer_nll','gradient_norm'))
                    or row['update']!=number or row['cursor']!=after[0]
                    or row['cumulative_supervised_targets']!=after[1]
                    or row['cumulative_processed_positions']!=after[2]
                    or row['targets']!=after[1]-before[1]
                    or row['processed_positions']!=after[2]-before[2]
                    or not math.isfinite(row['answer_nll']) or row['answer_nll']<0
                    or not math.isfinite(row['gradient_norm']) or row['gradient_norm']<0):
                raise ValueError('Saved committed SFT metric disagrees with counters')
            attempt.after(recovery_history_rows=1)
        if update and not _fixed_plain_equal(metric,history[-1]):
            raise ValueError('Last committed metric differs from full history')
        if type(state['model_training']) is not bool:
            raise ValueError('Invalid saved model mode')
        current=self.model.state_dict()
        if set(state['model']) != set(current):
            raise ValueError('Saved policy state keys differ')
        for key,value in current.items():
            units=self._validation_model_elements[key]
            attempt.before(recovery_tensor_elements=units)
            saved=state['model'][key]
            if (not isinstance(saved,torch.Tensor) or saved.shape!=value.shape or saved.dtype!=value.dtype
                    or saved.layout!=value.layout):
                raise ValueError('Saved policy tensor layout differs')
            if not bool(torch.isfinite(saved).all()):
                raise ValueError('Nonfinite saved policy tensor')
            attempt.after(recovery_tensor_elements=units)
        original=self.optimizer.state_dict(); saved=state['optimizer']
        if (set(saved)!={'state','param_groups'}
                or not _fixed_plain_equal(saved['param_groups'],original['param_groups'])):
            raise ValueError('Saved optimizer groups/settings differ')
        ids=[key for group in original['param_groups'] for key in group['params']]
        if (any(type(key) is not int for key in saved['state'])
                or set(saved['state']) != (set(ids) if update else set())
                or len(ids)!=len(self.parameters)):
            raise ValueError('Saved optimizer state coverage differs')
        for key,parameter in zip(ids,self.optimizer_parameters):
            if update==0:
                break
            moment=saved['state'][key]
            if type(moment) is not dict or len(moment)!=3 or set(moment)!={'step','exp_avg','exp_avg_sq'}:
                raise ValueError('Unexpected AdamW state')
            attempt.before(recovery_tensor_elements=1)
            if (not isinstance(moment['step'],torch.Tensor) or moment['step'].shape!=torch.Size([])
                    or moment['step'].layout!=torch.strided
                    or moment['step'].dtype not in (torch.float32,torch.float64)
                    or float(moment['step'])!=update):
                raise ValueError('AdamW step differs from completed updates')
            attempt.after(recovery_tensor_elements=1)
            for name in ('exp_avg','exp_avg_sq'):
                units=parameter.numel();attempt.before(recovery_tensor_elements=units)
                tensor=moment[name]
                if (not isinstance(tensor,torch.Tensor) or tensor.shape!=parameter.shape
                        or tensor.dtype!=parameter.dtype or tensor.layout!=parameter.layout):
                    raise ValueError('AdamW moment tensor layout differs')
                if not bool(torch.isfinite(tensor).all()):
                    raise ValueError('Nonfinite AdamW moment')
                if name=='exp_avg_sq' and bool((tensor<0).any()):
                    raise ValueError('Negative AdamW second moment')
                attempt.after(recovery_tensor_elements=units)
        rng=state['rng']
        if type(rng) is not dict or len(rng)!=3 or set(rng)!={'python','torch','cuda'}:
            raise ValueError('Incomplete SFT RNG state')
        try:
            attempt.before(recovery_rng_states=1)
            py=rng['python']
            if (type(py) is not tuple or len(py)!=3 or type(py[0]) is not int or py[0]!=3
                    or type(py[1]) is not tuple or len(py[1])!=625
                    or any(type(value) is not int for value in py[1])
                    or any(not 0<=value<2**32 for value in py[1][:-1])
                    or not 0<=py[1][-1]<=624
                    or (py[2] is not None and (type(py[2]) is not float or not math.isfinite(py[2])))):
                raise ValueError('Invalid Python RNG representation')
            random.Random().setstate(rng['python'])
            attempt.after(recovery_rng_states=1)
            units=self._validation_cpu_rng_shape.numel()
            attempt.before(recovery_rng_states=1,recovery_tensor_elements=units)
            cpu=rng['torch']
            if (not isinstance(cpu,torch.Tensor) or cpu.dtype!=torch.uint8 or cpu.device.type!='cpu'
                    or cpu.shape!=self._validation_cpu_rng_shape or cpu.layout!=torch.strided):
                raise ValueError('Invalid CPU RNG tensor layout')
            torch.Generator(device='cpu').set_state(rng['torch'])
            attempt.after(recovery_rng_states=1,recovery_tensor_elements=units)
            if type(rng['cuda']) is not list:
                raise ValueError('CUDA RNG collection must be a list')
            if self.device.type=='cpu' and rng['cuda']!=[]:
                raise ValueError('CPU snapshot contains CUDA RNG')
            if self.device.type=='cuda':
                current=torch.cuda.get_rng_state_all()
                if len(rng['cuda'])!=len(current):
                    raise ValueError('CUDA RNG device count differs')
                for number,(saved,expected) in enumerate(zip(rng['cuda'],current)):
                    units=self._validation_cuda_rng_shapes[number].numel()
                    attempt.before(recovery_rng_states=1,recovery_tensor_elements=units)
                    if (not isinstance(saved,torch.Tensor) or saved.dtype!=torch.uint8
                            or saved.device.type!='cpu' or saved.ndim!=1 or saved.shape!=expected.shape
                            or saved.layout!=torch.strided):
                        raise ValueError('Invalid CUDA RNG tensor layout')
                    # A temporary generator validates bytes without changing any
                    # live model/global RNG. This precedes all state application.
                    torch.Generator(device=f'cuda:{number}').set_state(saved)
                    attempt.after(recovery_rng_states=1,recovery_tensor_elements=units)
        except (TypeError,RuntimeError) as error:
            raise ValueError('Invalid SFT RNG state') from error

    @staticmethod
    def _metric_keys():
        return ('update','answer_nll','targets','gradient_norm','cursor',
                'cumulative_supervised_targets','processed_positions','cumulative_processed_positions')

    def _bounded_metric(self,row):
        expected=self._metric_keys()
        if (type(row) is not dict or len(row)!=len(expected)
                or any(key not in row for key in expected)):
            raise ValueError('Saved SFT metric requires bounded flat fixed fields')
        for key in expected:
            value=row[key]
            if key in ('answer_nll','gradient_norm'):
                if type(value) is not float or not math.isfinite(value):
                    raise ValueError('Saved SFT metric requires finite scalar fields')
            elif type(value) is not int or not 0<=value<=2**63-1:
                raise ValueError('Saved SFT metric requires bounded exact integer fields')

    def save(self,path,*,contract,parent_invocation,max_bytes):
        state=self.snapshot_state()
        payload=dict(phase='completed',completed_updates=self.update,state=state,contract=contract)
        self.validate_payload(payload)
        if self.work_ledger is not None:
            # Commit the completed save-validation spending, not its earlier
            # prevalidation prefix. No second semantic validation is performed.
            state['work_ledger']=self.work_ledger.snapshot()
        return save_snapshot(path,contract=contract,state=state,completed_updates=self.update,
                             parent_invocation=parent_invocation,max_bytes=max_bytes,guard=self.guard)

    def restore(self,path,*,contract,expected_sha256,expected_bytes,max_bytes):
        payload=load_snapshot(path,expected_sha256=expected_sha256,expected_bytes=expected_bytes,
            expected_contract=contract,max_bytes=max_bytes,validate_payload=self.validate_payload,guard=self.guard)
        state=payload['state']
        self.model.load_state_dict(state['model']); self.optimizer.load_state_dict(state['optimizer'])
        self.cursor,self.targets,self.positions=state['cursor'],state['targets'],state['positions']
        self.update=payload['completed_updates']; self.last_metric=deepcopy(state['metric'])
        self.history=deepcopy(state['history'])
        random.setstate(state['rng']['python']); torch.set_rng_state(state['rng']['torch'])
        if self.device.type=='cuda':
            torch.cuda.set_rng_state_all(state['rng']['cuda'])
        self.model.train(state['model_training']); self.incomplete_update=False
        return payload


def run_loop_with_recovery(loop,*,contract,output,parent_invocation,max_bytes,
                           checkpoint_every,metric_sink,baseline_observer,
                           final_observer,exporter):
    """Runner-owned save/observer ordering; callbacks never authorize model jobs."""
    if type(checkpoint_every) is not int or checkpoint_every < 1:
        raise ValueError('Positive checkpoint interval required')
    output=Path(output)
    latest=None
    def commit():
        nonlocal latest
        path=output/f'checkpoint-{loop.update:06d}.pt'
        header=loop.save(path,contract=contract,parent_invocation=parent_invocation,max_bytes=max_bytes)
        latest=dict(path=str(path),header=header)
        write_json(output/'latest-completed-snapshot.json',latest)
        return latest
    commit()
    rng=loop.snapshot_state()['rng']
    initial=baseline_observer()
    random.setstate(rng['python']); torch.set_rng_state(rng['torch'])
    if loop.device.type=='cuda':
        torch.cuda.set_rng_state_all(rng['cuda'])
    while loop.update < loop.total_updates:
        row=loop.completed_update()
        if loop.update%checkpoint_every==0 or loop.update==loop.total_updates:
            commit()
        metric_sink(row)
    final=final_observer()
    exporter()
    return dict(initial=initial,final=final,latest_completed_snapshot=latest)


def evaluate_sft_panel(loop,records,*,row_sink=None):
    """Reserve the entire original single-record NLL panel before a forward."""
    if not records:raise ValueError('Nonempty SFT evaluation panel required')
    costs=dict(valid_targets=sum(sum(value!=-100 for value in row['labels'][1:]) for row in records),
        logical_sequence_tokens=sum(len(row['ids']) for row in records),
        policy_forward_calls=len(records),policy_forward_positions=sum(len(row['ids']) for row in records),
        evaluation_calls=len(records),evaluation_positions=sum(len(row['ids']) for row in records))
    attempt=_WorkAttempt(loop.work_ledger,costs,'sft-nll-panel')
    try:
        loop.model.eval();loss_sum,count=0.,0
        with torch.no_grad(),loop.autocast():
            for record in records:
                loop.guard();batch=collate([record],loop.pad_id,loop.device)
                targets=int((batch[1][:,1:]!=-100).sum());positions=batch[0].numel()
                metadata=dict(valid_targets=targets,logical_sequence_tokens=len(record['ids']))
                attempt.before(**metadata);attempt.after(**metadata)
                geometry=dict(policy_forward_calls=1,policy_forward_positions=positions,
                              evaluation_calls=1,evaluation_positions=positions)
                attempt.before(**geometry)
                try:
                    loss=summed_loss(loop.model,batch);attempt.after(**geometry)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite SFT evaluation loss')
                except BaseException as error:
                    if row_sink is not None:row_sink(dict(id=record['id'],loss_sum=None,valid_targets=targets,
                        positions=positions,error=dict(type=type(error).__name__,message=str(error)[:512]),
                        cost_boundary='entered/successful logical work in journal; internal failed-call FLOPs unknown'))
                    raise
                value=float(loss);loss_sum+=value;count+=targets
                if row_sink is not None:row_sink(dict(id=record['id'],loss_sum=value,valid_targets=targets,positions=positions))
        attempt.complete();return loss_sum/count
    except BaseException as error:
        attempt.fail(error);raise
    finally:loop.model.train()


def generation_upper(prefixes,cap):
    if type(cap) is not int or cap<=0 or not prefixes or any(not prefix for prefix in prefixes):
        raise ValueError('Nonempty prompts and positive generation cap required')
    # The actual dispatch stays cached. This intentionally conservative uncached
    # envelope is not reported as actual cached input positions or as FLOPs.
    positions=sum(cap*len(prefix)+cap*(cap-1)//2 for prefix in prefixes)
    return dict(policy_forward_calls=len(prefixes)*cap,policy_forward_positions=positions,
        evaluation_calls=len(prefixes)*cap,evaluation_positions=positions,
        generation_calls=len(prefixes),generation_position_upper_bound=positions,
        generation_tokens=len(prefixes)*cap,logical_sequence_tokens=sum(len(prefix)+cap for prefix in prefixes))


def generate_sft_panel(loop,tokenizer,records,prefixes,*,cap,stop_ids,end_message_id,row_sink=None):
    """Retain original cached full/LoRA dispatch and raw errors within a panel cap."""
    if len(records)!=len(prefixes) or not stop_ids:raise ValueError('Aligned generation panel and explicit stops required')
    attempt=_WorkAttempt(loop.work_ledger,generation_upper(prefixes,cap),'sft-generation-panel')
    # PEFT.generate delegates to its adapter-injected underlying HF model.
    # Hook that executed model, not only the unused top-level PEFT.forward.
    network=loop.model.get_base_model() if hasattr(loop.model,'peft_config') else loop.model
    original=network.forward;had_forward='forward' in network.__dict__;previous=network.__dict__.get('forward')
    @wraps(original)
    def bounded_forward(*args,**kwargs):
        ids=kwargs.get('input_ids',args[0] if args else None)
        if not isinstance(ids,torch.Tensor) or ids.ndim!=2 or ids.shape[0]!=1:
            raise WorkBudgetExceeded('Unknown SFT generation geometry refused before backend')
        loop.guard();positions=ids.numel()
        geometry=dict(policy_forward_calls=1,policy_forward_positions=positions,
            evaluation_calls=1,evaluation_positions=positions,generation_position_upper_bound=positions)
        attempt.before(**geometry);result=original(*args,**kwargs);attempt.after(**geometry);return result
    try:
        loop.model.eval();network.forward=bounded_forward;rows=[]
        with torch.no_grad():
            for record,prefix in zip(records,prefixes):
                loop.guard();started=time.monotonic();continuation=None;stop=None
                ids=torch.tensor([prefix],device=loop.device)
                attempt.before(generation_calls=1,logical_sequence_tokens=len(prefix))
                attempt.after(logical_sequence_tokens=len(prefix))
                try:
                    generated=loop.model.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=cap,
                        do_sample=False,num_beams=1,num_return_sequences=1,guidance_scale=1.0,
                        pad_token_id=tokenizer.pad_token_id,eos_token_id=stop_ids,use_cache=True)
                    continuation=generated[0,len(prefix):].tolist()
                    stop=continuation[-1] if continuation and continuation[-1] in stop_ids else None
                    if len(continuation)>cap:raise WorkBudgetExceeded('SFT decoder returned beyond declared cap')
                    attempt.before(generation_tokens=len(continuation),logical_sequence_tokens=len(continuation))
                    attempt.after(generation_calls=1,generation_tokens=len(continuation),logical_sequence_tokens=len(continuation))
                    answer=tokenizer.decode(continuation,skip_special_tokens=True).strip()
                    retained=dict(id=record['id'],prompt_ids=list(prefix),generated_ids=continuation,
                        text=tokenizer.decode(continuation,skip_special_tokens=False),
                        reference=record['messages'][-1]['content'],exact_match=answer==record['messages'][-1]['content'],
                        stopped_on_end=bool(continuation and continuation[-1]==end_message_id),final_stop_id=stop,
                        stop_reason='declared-stop' if stop is not None else 'max_new_tokens',truncated=stop is None,
                        generated_tokens=len(continuation),prompt_tokens=len(prefix),elapsed_seconds=time.monotonic()-started,error=None)
                except BaseException as error:
                    retained=dict(id=record['id'],prompt_ids=list(prefix),generated_ids=continuation,text=None,
                        reference=record['messages'][-1]['content'],exact_match=None,final_stop_id=stop,
                        stop_reason='error',generation_stop_reason=None if continuation is None else 'declared-stop' if stop is not None else 'max_new_tokens',
                        truncated=None if continuation is None else stop is None,
                        generated_tokens=None if continuation is None else len(continuation),prompt_tokens=len(prefix),
                        elapsed_seconds=time.monotonic()-started,error=dict(type=type(error).__name__,message=str(error)[:512]),
                        cost_boundary='entered/successful logical geometry; returned IDs retained; failed internal partial outputs/FLOPs unknown')
                    if row_sink is not None:row_sink(retained)
                    raise  # preserve original SFT job failure, not silent error-as-success
                rows.append(retained)
                if row_sink is not None:row_sink(retained)
                loop.guard()
        attempt.complete();return rows
    except BaseException as error:
        attempt.fail(error);raise
    finally:
        if had_forward:network.forward=previous
        else:network.__dict__.pop('forward',None)
        loop.model.train()


def verify_base_tokenizer(base_tokenizer, proposed, *, template, stop_ids,
                          tokenizer_id, base_revision, tokenizer_revision):
    """A separately pinned tokenizer must preserve the pretrained base's meanings."""
    if base_tokenizer.pad_token_id is None:
        base_tokenizer.pad_token=base_tokenizer.eos_token
    expected=tokenizer_interface(base_tokenizer,template=template,stop_ids=stop_ids,
        tokenizer_id=tokenizer_id,tokenizer_revision=base_revision)
    observed=tokenizer_interface(proposed,template=template,stop_ids=stop_ids,
        tokenizer_id=tokenizer_id,tokenizer_revision=tokenizer_revision)
    # Different revision declarations may have identical verified semantics.
    assert_compatible(expected,observed,compare_source=False)
    return observed


def _main(argv=None):
    global FAILURE_OUTPUT, FAILURE_JOURNAL,ACTIVE_WORK_LEDGER
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default='Qwen/Qwen3-0.6B-Base')
    parser.add_argument('--revision',required=True)
    parser.add_argument('--tokenizer-revision',required=True)
    parser.add_argument('--template',type=Path,required=True)
    parser.add_argument('--train',type=Path,required=True)
    parser.add_argument('--dev',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['full','lora'],default='full')
    parser.add_argument('--updates',type=int,default=20)
    parser.add_argument('--microbatch',type=int,default=1)
    parser.add_argument('--accumulation',type=int,default=4)
    parser.add_argument('--max-length',type=int,default=256)
    parser.add_argument('--learning-rate',type=float,default=2e-5)
    parser.add_argument('--runtime-seconds',type=int,default=900)
    parser.add_argument('--reserve-gib',type=float,default=25.)
    parser.add_argument('--seed',type=int,default=1212)
    parser.add_argument('--rank',type=int,default=8)
    parser.add_argument('--checkpoint-every',type=int,default=20)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--resume-contract',type=Path,help='Independently retained scientific recovery contract')
    parser.add_argument('--resume-sha256',help='Independently expected committed payload SHA256')
    parser.add_argument('--resume-bytes',type=int,help='Independently expected committed payload byte size')
    parser.add_argument('--snapshot-max-bytes',type=int,help='Required explicit per-snapshot resource bound')
    parser.add_argument('--work-limits',type=Path,required=True,help='Immutable JSON with all sixteen SFT v2 work caps')
    parser.add_argument('--work-journal-max-bytes',type=int,required=True,help='Positive bounded work journal, at most64MiB')
    parser.add_argument('--work-journal',type=Path,help='Same physical private work journal REQUIRED on resume')
    parser.add_argument('--environment-lock',type=Path,required=True)
    parser.add_argument('--allow-download',action='store_true',help='Explicit opt-in; cached inputs are the default')
    args=parser.parse_args(argv)
    for revision in (args.revision,args.tokenizer_revision):
        if not re.fullmatch('[0-9a-f]{40}',revision):
            raise ValueError('Model and tokenizer revisions must be exact 40-character commit SHAs')
    if args.model not in ('Qwen/Qwen3-0.6B-Base','Qwen/Qwen3-1.7B-Base'):
        raise ValueError('Declared course run supports the two named base checkpoints')
    if min(args.updates,args.microbatch,args.accumulation,args.runtime_seconds,args.max_length,args.checkpoint_every)<=0:
        raise ValueError('Positive budgets required')
    if args.reserve_gib<25 or args.max_length>2048 or args.updates>2000:
        raise ValueError('Course bounds require >=25 GiB reserve, length<=2048, updates<=2000')
    if not math.isfinite(args.learning_rate) or args.learning_rate<=0 or not math.isfinite(args.reserve_gib):
        raise ValueError('Finite positive optimizer/resource settings required')
    if args.snapshot_max_bytes is not None and not 0<args.snapshot_max_bytes<=2**63-1:
        raise ValueError('Explicit snapshot byte bound must lie within 1..2**63-1')
    output=args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new empty output directory for every invocation, including resume')
    if args.resume and (not args.resume_sha256 or args.resume_bytes is None):
        raise ValueError('Resume requires independently expected SHA256 and byte size')
    if args.resume and args.resume_contract is None:
        raise ValueError('Resume requires independently retained --resume-contract')
    if args.resume and args.work_journal is None:
        raise ValueError('Resume requires the same physical --work-journal')
    if not args.resume and (args.resume_sha256 is not None or args.resume_bytes is not None
                            or args.resume_contract is not None):
        raise ValueError('Resume identity supplied without a resume payload')
    from dongxi_llms.work_budget import _decode
    raw_limits=read_work_limits_file(args.work_limits)
    parsed_limits_sha256=hashlib.sha256(raw_limits).hexdigest()
    budget=work_budget_contract(_decode(raw_limits),args.work_journal_max_bytes)
    output.mkdir(parents=True,exist_ok=True,mode=0o700)
    info=output.stat()
    if info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)&0o077:
        raise ValueError('SFT work journal requires a private owned output directory')
    FAILURE_OUTPUT=output/'failure.json'
    journal=FAILURE_JOURNAL=IdentityJournal(output, vars(args))
    root=Path(__file__).resolve().parents[1]
    sources=[Path(__file__),root/'src/dongxi_llms/run_identity.py',
             root/'src/dongxi_llms/training_snapshot.py',root/'src/dongxi_llms/work_budget.py']
    # Caps are intentionally NOT a generic input: its blocking file reader
    # would bypass the regular-file/nonblocking gate after a path replacement.
    inputs=[args.train,args.dev,args.template]+([args.resume,args.resume_contract] if args.resume else [])
    initial_identity=collect_run_identity(root,source_files=sources,input_files=inputs,
        environment_lock=args.environment_lock,config=vars(args),device={'mode':'cpu','name':'preflight'})
    bind_bounded_work_limits(initial_identity,args.work_limits,parsed_limits_sha256)
    journal.attach(initial_identity)
    journal.stage('hardware_preflight')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA with verified BF16 support required')
    if args.snapshot_max_bytes is None:
        raise ValueError('Production snapshot creation requires explicit --snapshot-max-bytes')
    if available_gib()<args.reserve_gib:
        raise RuntimeError('Host memory reserve unavailable before load')
    start=time.monotonic(); minimum_available=available_gib(); total_targets=0
    invocation_id=journal.invocation_id
    def guard():
        nonlocal minimum_available
        current=available_gib(); minimum_available=min(minimum_available,current)
        if current<args.reserve_gib: raise RuntimeError('Host memory reserve violated')
        if time.monotonic()-start>args.runtime_seconds: raise TimeoutError('Declared runtime cap reached')
    journal.stage('reading_data')
    train_records,dev_records=read_records(args.train),read_records(args.dev)
    if {record['group'] for record in train_records}&{record['group'] for record in dev_records}:
        raise ValueError('Train/development source groups overlap')
    from transformers import AutoTokenizer,AutoModelForCausalLM
    import transformers
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    guard()
    journal.stage('loading_tokenizer')
    verify_work_limits_file(args.work_limits,parsed_limits_sha256)
    tokenizer=AutoTokenizer.from_pretrained(args.model,revision=args.tokenizer_revision,
                                           local_files_only=not args.allow_download)
    guard()
    tokenizer.chat_template=args.template.read_text()
    for marker in ('<|im_start|>','<|im_end|>'):
        encoded=tokenizer.encode(marker,add_special_tokens=False)
        if len(encoded)!=1 or encoded[0]==tokenizer.unk_token_id:
            raise ValueError('Template requires existing single-token message markers; adding vocabulary is out of scope')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token=tokenizer.eos_token
    train=[encode_record(tokenizer,record,args.max_length) for record in train_records]
    dev=[encode_record(tokenizer,record,args.max_length) for record in dev_records]
    end_message_id=tokenizer.convert_tokens_to_ids('<|im_end|>')
    stop_ids=sorted({value for value in (end_message_id,tokenizer.eos_token_id) if value is not None})
    interface=tokenizer_interface(tokenizer,template=tokenizer.chat_template,stop_ids=stop_ids,
        tokenizer_id=args.model,tokenizer_revision=args.tokenizer_revision)
    config={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items() if key not in ('resume','output')}
    config.update(torch_version=str(torch.__version__),transformers_version=transformers.__version__,
                  template_sha256=hashlib.sha256(args.template.read_bytes()).hexdigest(),
                  train_sha256=hashlib.sha256(args.train.read_bytes()).hexdigest(),
                  dev_sha256=hashlib.sha256(args.dev.read_bytes()).hexdigest(),
                  encoded_train_sha256=canonical_hash(train),encoded_dev_sha256=canonical_hash(dev),
                  gpu=torch.cuda.get_device_name(),dtype='BF16 weights/autocast; FP32 cross entropy',
                  attention_backend='sdpa',loss_policy='assistant body + template end tokens; one explicit shift')
    config.update(work_budget=budget,work_limits_sha256=parsed_limits_sha256,
                  generation_dispatch='single-sequence greedy; use_cache=True; conservative uncached reservation',
                  generation_max_new_tokens=64)
    config['generation_stop_ids']=stop_ids
    config['checkpoint_interface']=interface
    config['source_sha256']=initial_identity['source_sha256']
    config['environment_lock_sha256']=initial_identity['environment']['environment_lock']['sha256']
    config['numerical_environment']=observed_numerical_environment(initial_identity['environment'])
    journal.stage('identifying_local_inputs',checkpoint_interface=interface)
    base_snapshot=prepare_model_snapshot(args.model,args.revision,allow_download=args.allow_download)
    guard()
    base_tokenizer=AutoTokenizer.from_pretrained(base_snapshot,local_files_only=True)
    verify_base_tokenizer(base_tokenizer,tokenizer,template=tokenizer.chat_template,
        stop_ids=stop_ids,tokenizer_id=args.model,base_revision=args.revision,
        tokenizer_revision=args.tokenizer_revision)
    identity=collect_run_identity(root,source_files=sources,input_files=inputs,checkpoint=base_snapshot,
        environment_lock=args.environment_lock,interface=interface,config=config,
        device={'mode':'cuda','name':torch.cuda.get_device_name(),'cuda_runtime':torch.version.cuda},before_chunk=guard)
    bind_bounded_work_limits(identity,args.work_limits,parsed_limits_sha256)
    if (identity['source_sha256']!=initial_identity['source_sha256']
            or identity['input_sha256']!=initial_identity['input_sha256']
            or identity['environment']['environment_lock']['sha256']
                !=initial_identity['environment']['environment_lock']['sha256']
            or observed_numerical_environment(identity['environment'])!=config['numerical_environment']):
        raise ValueError('SFT source/input/lock/environment changed during pre-load preparation')
    tokenizer_snapshot=cached_snapshot(args.model,args.tokenizer_revision,'tokenizer_config.json')
    identity['tokenizer_files']=artifact_hashes(tokenizer_snapshot,guard,patterns=TOKENIZER_PATTERNS)
    identity['identity_sha256']=canonical_hash({k:v for k,v in identity.items() if k!='identity_sha256'})
    config['base_checkpoint_files']=identity['checkpoint_files']
    journal.attach(identity)
    recovery_contract=stable_sft_contract(config)
    if args.resume:
        journal.stage('inspecting_completed_snapshot')
        if args.resume_contract.stat().st_size>65536:
            raise ValueError('Retained SFT contract exceeds bounded JSON size')
        retained=json.loads(args.resume_contract.read_text())
        if retained!=recovery_contract:
            raise ValueError('Current observable SFT science differs from retained contract')
        inspect_snapshot(args.resume,expected_sha256=args.resume_sha256,expected_bytes=args.resume_bytes,
            expected_contract=recovery_contract,max_bytes=args.snapshot_max_bytes,guard=guard)
    write_json(output/'recovery-contract.json',recovery_contract)
    journal.stage('loading_model')
    verify_work_limits_file(args.work_limits,parsed_limits_sha256)
    model=AutoModelForCausalLM.from_pretrained(base_snapshot,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda()
    guard()
    if model.get_input_embeddings().weight.shape[0]<=interface['tokenizer']['max_token_id']:
        raise ValueError('Actual tokenizer IDs exceed model embedding rows')
    journal.stage('configuring_training')
    model.config.use_cache=False
    model.gradient_checkpointing_enable()
    if args.mode=='lora':
        from peft import LoraConfig,get_peft_model
        model=get_peft_model(model,LoraConfig(r=args.rank,lora_alpha=args.rank,lora_dropout=0.,
                                             target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM',revision=args.revision))
        model.enable_input_require_grads()
    parameters=[parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=args.learning_rate,weight_decay=0.)
    guard()
    loop=SFTLoop(model,optimizer,train,tokenizer.pad_token_id,microbatch=args.microbatch,
                 accumulation=args.accumulation,seed=args.seed,device='cuda',
                 autocast_dtype=torch.bfloat16,guard=guard,total_updates=args.updates)
    work_path=args.work_journal or output/'work-ledger.jsonl'
    options=dict(limits=budget['limits'],contract_sha256=canonical_hash(recovery_contract),
                 max_bytes=budget['max_bytes'],invocation_id=invocation_id)
    ACTIVE_WORK_LEDGER=(WorkLedger.open if args.resume else WorkLedger.create)(work_path,**options)
    loop.bind_work_ledger(ACTIVE_WORK_LEDGER,recovery_contract)
    first_update=0
    if args.resume:
        journal.stage('restoring_checkpoint')
        restored=loop.restore(args.resume,contract=recovery_contract,expected_sha256=args.resume_sha256,
                              expected_bytes=args.resume_bytes,max_bytes=args.snapshot_max_bytes)
        first_update=loop.update
        if first_update>args.updates:
            raise ValueError('Saved update exceeds the fixed scientific horizon')
        if loop.last_metric is not None:
            write_json(output/'restored-committed-metric.json',dict(
                source_parent_invocation=restored['parent_invocation'],metric=loop.last_metric,
                resumed_invocation_id=invocation_id))
        write_json(output/'restored-committed-history.json',dict(
            source_parent_invocation=restored['parent_invocation'],history=loop.history,
            resumed_invocation_id=invocation_id))
    write_json(output/'config.json',config)
    guard()
    evaluation_records=dev_records[:8]
    evaluation_prefixes=[tokenizer.apply_chat_template(record['messages'][:-1],tokenize=True,
        add_generation_prompt=True,enable_thinking=False,return_dict=False) for record in evaluation_records]
    if any(not prefix or len(prefix)+64>args.max_length for prefix in evaluation_prefixes):
        raise ValueError('Fixed64-token evaluation plus prompt exceeds declared context; never silently shorten')
    def retain(stage,row):
        with (output/f'{stage}-raw.jsonl').open('a') as handle:
            handle.write(json.dumps(row,allow_nan=False)+'\n');handle.flush();os.fsync(handle.fileno())
    def evaluate(stage):
        return evaluate_sft_panel(loop,dev,row_sink=lambda row:retain(stage+'-nll',row))
    def generate_samples(stage):
        return generate_sft_panel(loop,tokenizer,evaluation_records,evaluation_prefixes,
            cap=64,stop_ids=stop_ids,end_message_id=end_message_id,row_sink=lambda row:retain(stage+'-generation',row))
    def baseline_observer():
        journal.stage('baseline_evaluation')
        result=dict(nll=evaluate('baseline'),samples=generate_samples('baseline'))
        journal.stage('training')
        return result
    def final_observer():
        journal.stage('final_evaluation')
        return dict(nll=evaluate('final'),samples=generate_samples('final'))
    def exporter():
        journal.stage('exporting_checkpoint'); guard()
        verify_bounded_work_limits_identity(identity,args.work_limits,parsed_limits_sha256)
        tokenizer.save_pretrained(output/'tokenizer')
        model.save_pretrained(output/'policy',safe_serialization=True)
        tokenizer.save_pretrained(output/'policy')
        (output/'policy'/'course-genealogy.json').write_text(json.dumps(
            dict(kind='full-HF-model' if args.mode=='full' else 'PEFT-adapter-requires-pinned-base',
                 base_model=args.model,base_revision=args.revision,mode=args.mode,
                 template_sha256=config['template_sha256'],data_sha256=config['train_sha256'],
                 checkpoint_interface=interface,base_checkpoint_files=identity['checkpoint_files'],
                 run_identity_file=journal.path.name),indent=2)+'\n')
    with (output/'metrics.jsonl').open('a') as log:
        def record_metric(record):
            nonlocal total_targets
            total_targets+=record['targets']
            log.write(json.dumps(dict(record,invocation_id=invocation_id,resume_checkpoint_update=first_update,
                                      elapsed_seconds=time.monotonic()-start))+'\n'); log.flush()
        lifecycle=run_loop_with_recovery(loop,contract=recovery_contract,output=output,
            parent_invocation=invocation_id,max_bytes=args.snapshot_max_bytes,
            checkpoint_every=args.checkpoint_every,metric_sink=record_metric,
            baseline_observer=baseline_observer,final_observer=final_observer,exporter=exporter)
    verify_bounded_work_limits_identity(identity,args.work_limits,parsed_limits_sha256)
    initial,baseline_samples=lifecycle['initial']['nll'],lifecycle['initial']['samples']
    final,samples=lifecycle['final']['nll'],lifecycle['final']['samples']
    result=dict(status='completed',invocation_id=invocation_id,resume_checkpoint_update=first_update,
                updates=args.updates,initial_dev_nll=initial,final_dev_nll=final,
                supervised_targets_this_invocation=total_targets,elapsed_seconds=time.monotonic()-start,
                cumulative_supervised_targets=loop.targets,cumulative_processed_positions=loop.positions,
                latest_completed_snapshot=lifecycle['latest_completed_snapshot'],
                minimum_sampled_memavailable_gib=minimum_available,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                trainable_parameters=sum(p.numel() for p in parameters),
                baseline_samples=baseline_samples,samples=samples,
                generation_scope='first eight development IDs, deterministic cached greedy64; not the frozen publication suite',
                cumulative_work_ledger=ACTIVE_WORK_LEDGER.snapshot(),work_journal=str(work_path),
                work_scope='permanent conservative reservations; logical positions/calls, not FLOPs/backward recomputation')
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    journal.complete(result_file='result.json',minimum_sampled_memavailable_gib=minimum_available)
    print(json.dumps(result,indent=2))


def main(argv=None):
    global FAILURE_JOURNAL, FAILURE_OUTPUT,ACTIVE_WORK_LEDGER
    FAILURE_JOURNAL=FAILURE_OUTPUT=None
    ACTIVE_WORK_LEDGER=None
    try:
        return _main(argv)
    except BaseException as error:
        if FAILURE_JOURNAL is not None:
            FAILURE_JOURNAL.fail(error)
        raise
    finally:
        if ACTIVE_WORK_LEDGER is not None:ACTIVE_WORK_LEDGER.close()


if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        if FAILURE_OUTPUT is not None:
            FAILURE_OUTPUT.write_text(json.dumps(dict(status='failed',exception_type=type(exc).__name__,
                                                     evidence='partial logs retained; no completion claim'),indent=2)+'\n')
        raise
