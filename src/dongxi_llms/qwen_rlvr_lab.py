"""Optional synchronous HF causal-model RLVR adapter, offline local weights only.

CPU tests construct an original randomly initialized tiny HF causal model.
No pretrained Qwen execution, CUDA profiling or language capability is implied.
Collection uses temperature1/full support; response-only token masks include EOS.
"""
from copy import deepcopy
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import stat
import sys
import time
from uuid import uuid4

import torch

from dongxi_llms.grpo_lab import (clipped_objective, exact_kl,
                                  group_advantages, verify_integer)
from dongxi_llms.run_identity import (collect_run_identity, parent_interface,
                                     tokenizer_interface, canonical_hash, artifact_hashes,
                                     file_digest, ARTIFACT_PATTERNS)
from dongxi_llms.batched_cache_lab import digest as state_digest
from dongxi_llms.training_snapshot import inspect_snapshot, load_snapshot, save_snapshot
from dongxi_llms.work_budget import (WorkLedger, WorkBudgetExceeded, validate_limits,
                                     MAX_JOURNAL_BYTES, _decode)
from dongxi_llms.snapshot_io_budget import (SnapshotIOBudget, validate_io_contract,
    io_ledger_contract_sha256, read_bounded_json, validate_work_receipt)

BUDGET_KEYS = ('train_updates','collections','source_examples','generated_slots',
    'valid_response_tokens','multinomial_draws','response_targets',
    'generation_forward_calls','generation_forward_positions',
    'policy_forward_calls','policy_forward_positions','reference_forward_calls',
    'reference_forward_positions','old_policy_forward_calls','old_policy_forward_positions',
    'evaluation_examples','evaluation_calls','evaluation_positions','evaluation_tokens',
    'recovery_validation_operations','recovery_validation_calls',
    'recovery_validation_positions','recovery_multinomial_draws')


def read_work_limits_file(path):
    """Bound bytes and refuse links/special files before parsing or output work."""
    path=Path(os.path.abspath(path));descriptors=[]
    try:
        parent=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW);descriptors.append(parent)
        for component in path.parent.parts[1:]:
            parent=os.open(component,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent)
            descriptors.append(parent)
        descriptor=os.open(path.name,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW,dir_fd=parent)
        descriptors.append(descriptor);before=os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):raise ValueError('Work limits must be a regular no-follow file')
        if before.st_size>65536:raise ValueError('Work cap input must be at most64KiB')
        raw=os.pread(descriptor,65537,0);after=os.fstat(descriptor)
        if len(raw)>65536 or len(raw)!=before.st_size or (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):
            raise ValueError('Work cap input exceeded its bound or changed during read')
        return raw
    finally:
        for descriptor in reversed(descriptors):os.close(descriptor)


def work_budget_contract(limits,max_bytes):
    validate_limits(limits)
    if set(limits)!=set(BUDGET_KEYS) or type(max_bytes) is not int or not 1<=max_bytes<=MAX_JOURNAL_BYTES:
        raise ValueError('Exact RLVR dimensions and bounded work-journal bytes required')
    return dict(schema='rlvr-logical-work-v1',limits=dict(limits),max_bytes=max_bytes,
        reservations='permanent conservative upper bounds before whole known operation',
        scope='logical calls/positions/actions/checks, not FLOPs/recompute/physical quotas')


def validate_work_contract(value):
    if (type(value) is not dict or len(value)!=5
            or set(value)!={'schema','limits','max_bytes','reservations','scope'}
            or value!=work_budget_contract(value['limits'],value['max_bytes'])):
        raise ValueError('Exact current RLVR23 work contract required; no silent migration')
    return deepcopy(value)


def validate_rlvr_io_budget(contract,io_budget,*,max_bytes=None):
    """A declared accounted contract cannot use the unhooked teaching path."""
    if 'snapshot_io_contract' not in contract:
        if io_budget is not None:raise ValueError('I/O hook absent from RLVR science')
        return
    declared=validate_io_contract(contract['snapshot_io_contract'])
    if (not isinstance(io_budget,SnapshotIOBudget) or io_budget.contract!=declared
            or io_budget.science_sha256!=canonical_hash(contract)):
        raise ValueError('Declared RLVR snapshot I/O requires its matching bound hook')
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes!=declared['envelope']['max_payload_bytes']):
        raise ValueError('RLVR snapshot byte bound differs from declared I/O envelope')
    io_budget._check_identity()


def verify_bounded_metadata(metadata,guard=lambda:None):
    for role,expected in metadata.items():
        guard()
        _,observed=(read_bounded_json(expected['path'],maximum=1024**2)
            if role=='resume_contract' else read_bounded_json(expected['path']))
        if observed!=expected:
            raise ValueError('Bounded snapshot bootstrap metadata changed: '+role)


def bind_bounded_metadata(identity,metadata,*,expected_snapshot=None,verified_snapshot=None):
    verify_bounded_metadata(metadata)
    identity['bounded_snapshot_metadata']=deepcopy(metadata)
    if expected_snapshot is not None:
        identity['expected_resume_snapshot']=dict(expected_snapshot,
            boundary='independently supplied expectation; not actual payload verification')
    if verified_snapshot is not None:
        identity['verified_resume_snapshot']=dict(verified_snapshot,
            boundary='actual shared inspection after independent journal admission')
    identity['identity_sha256']=canonical_hash({k:v for k,v in identity.items() if k!='identity_sha256'})
    return identity


def reject_payload_identity_aliases(args,source_files=()):
    """Metadata-only first-read gate; ordinary identity hashing stays excluded."""
    if not args.resume:return
    payload=os.path.abspath(args.resume)
    try:
        info=os.stat(payload);payload_inode=(info.st_dev,info.st_ino)
    except FileNotFoundError:
        # Missing bytes remain a charged shared-read failure, not a hidden read.
        payload_inode=None
    roles=[('source',Path(path)) for path in source_files]
    for role,path in (('template',args.template),('environment-lock',args.environment_lock)):
        if path is not None:roles.append((role,Path(path)))
    roles.extend(('parent-artifact',path) for pattern in ARTIFACT_PATTERNS
                 for path in args.model_dir.glob(pattern))
    for role,path in roles:
        if os.path.abspath(path)==payload:
            raise ValueError('Resume payload aliases a generic identity role: '+role)
        if payload_inode is not None:
            try:info=os.stat(path)
            except FileNotFoundError:continue
            if (info.st_dev,info.st_ino)==payload_inode:
                raise ValueError('Resume payload inode aliases a generic identity role: '+role)


def validate_resume_receipt(*,contract,budget,io_contract,receipt,expected_sha256,expected_bytes,updates):
    """Bounded metadata gates precede output, journal and model creation."""
    budget=validate_work_contract(budget);io_contract=validate_io_contract(io_contract)
    receipt=validate_work_receipt(receipt);header=receipt['snapshot']
    _integer(updates,'updates',1)
    if (updates>16 or type(contract) is not dict or contract.get('schema')!='rlvr-runner-recovery-v1'
            or type(contract.get('updates')) is not int or contract['updates']!=updates
            or header['completed_updates']>updates
            or (header['phase']=='pending' and header['completed_updates']==updates)):
        raise ValueError('RLVR resume phase/cursor exceeds the fixed completed/pending horizon')
    if (contract.get('work_budget')!=budget or contract.get('snapshot_io_contract')!=io_contract
            or header['contract_sha256']!=canonical_hash(contract)
            or header['payload_sha256']!=expected_sha256 or type(expected_bytes) is not int
            or header['payload_bytes']!=expected_bytes
            or expected_bytes>io_contract['envelope']['max_payload_bytes']
            or receipt['io_contract_sha256']!=canonical_hash(io_contract)
            or receipt['runner_work_prefix'] is None):
        raise ValueError('Independent RLVR receipt differs from science/caps/payload/runner expectation')
    return receipt


def open_snapshot_budgets(*,contract,budget,io_contract,receipt,work_path,io_path,
                          invocation_id,expected_sha256=None,expected_bytes=None):
    """Open both physical journal prefixes before any shared payload/model work."""
    budget=validate_work_contract(budget);io_contract=validate_io_contract(io_contract)
    if contract.get('work_budget')!=budget or contract.get('snapshot_io_contract')!=io_contract:
        raise ValueError('RLVR journal caps differ from scientific contract')
    if os.path.abspath(work_path)==os.path.abspath(io_path):
        raise ValueError('RLVR logical-work and snapshot I/O journals must be separate files')
    if receipt is not None:
        receipt=validate_resume_receipt(contract=contract,budget=budget,io_contract=io_contract,
            receipt=receipt,expected_sha256=expected_sha256,expected_bytes=expected_bytes,
            updates=contract['updates'])
    science=canonical_hash(contract);work=io_ledger=None
    try:
        options=dict(limits=budget['limits'],contract_sha256=science,max_bytes=budget['max_bytes'],
                     invocation_id=invocation_id)
        io_options=dict(limits=io_contract['limits'],contract_sha256=io_ledger_contract_sha256(io_contract,science),
                        max_bytes=io_contract['max_journal_bytes'],invocation_id=invocation_id)
        if receipt is None:
            work=WorkLedger.create(work_path,**options);io_ledger=WorkLedger.create(io_path,**io_options)
        else:
            work=WorkLedger.open(work_path,expected_snapshot=receipt['runner_work_prefix'],**options)
            io_ledger=WorkLedger.open(io_path,expected_snapshot=receipt['io_prefix'],**io_options)
        hook=SnapshotIOBudget(io_ledger,contract=io_contract,scientific_contract_sha256=science,
                              expected_receipt=receipt)
        return work,io_ledger,hook
    except BaseException:
        try:
            if io_ledger is not None:io_ledger.close()
        finally:
            if work is not None:work.close()
        raise


def verify_work_limits_file(path,expected_sha256):
    observed=hashlib.sha256(read_work_limits_file(path)).hexdigest()
    if observed!=expected_sha256:
        raise ValueError('Work cap bytes changed after their bounded initial parse')
    return observed


def attach_work_limits_identity(identity,root,path,expected_sha256):
    verify_work_limits_file(path,expected_sha256)
    absolute=Path(os.path.abspath(path))
    try:key=str(absolute.relative_to(Path(root)))
    except ValueError:key=str(absolute)
    identity['input_sha256'][key]=expected_sha256
    identity['bounded_input_files']={key:dict(role='rlvr-work-limits',max_bytes=65536,
        reader='regular no-follow nonblocking stable bounded bytes')}
    identity['identity_sha256']=canonical_hash({k:v for k,v in identity.items() if k!='identity_sha256'})
    return identity


class _WorkAttempt:
    def __init__(self,ledger,costs,operation):
        self.ledger=ledger;self.ticket=None
        self.entered={key:0 for key in BUDGET_KEYS};self.success=dict(self.entered)
        if ledger is not None:
            if set(ledger.limits)!=set(BUDGET_KEYS):raise ValueError('Wrong runner work dimensions')
            self.ticket=ledger.reserve(costs,operation=operation)

    def before(self,**delta):
        candidate=dict(self.entered)
        for key,value in delta.items():
            _integer(value,key);candidate[key]+=value
        if self.ledger is not None:self.ledger.assert_within(self.ticket,candidate)
        self.entered=candidate

    def after(self,**delta):
        for key,value in delta.items():
            _integer(value,key);self.success[key]+=value
            if self.success[key]>self.entered[key]:raise ValueError('Successful work without entered work')

    def complete(self):
        if self.ledger is not None:self.ledger.complete(self.ticket,self.success)

    def fail(self,error):
        if self.ledger is not None and not self.ledger.poisoned:
            self.ledger.fail(self.ticket,(type(error).__name__+': '+str(error))[:512],
                             known_actual=self.success,attempted=self.entered)


def collection_upper(prompt_length,cap,group):
    positions=group*(cap*prompt_length+cap*(cap-1)//2);score=group*(prompt_length+cap-1)
    return dict(collections=1,source_examples=1,generated_slots=group*cap,
        valid_response_tokens=group*cap,multinomial_draws=group*cap,
        generation_forward_calls=cap,generation_forward_positions=positions,
        policy_forward_calls=cap+1,policy_forward_positions=positions+score,
        old_policy_forward_calls=1,old_policy_forward_positions=score)


def pool_validation_upper(pool,*,policy=False):
    if type(pool) is not dict or not {'prompt_ids','responses'}<=set(pool):
        raise ValueError('Invalid validation pool before reservation')
    prompt=pool['prompt_ids'];responses=pool['responses']
    if (not isinstance(prompt,torch.Tensor) or prompt.ndim!=2 or prompt.shape[0]!=1
            or not isinstance(responses,torch.Tensor) or responses.ndim!=2
            or min(responses.shape)<1 or prompt.shape[1]<1):
        raise ValueError('Invalid validation geometry before reservation')
    group,steps=responses.shape;positions=group*(prompt.shape[1]+steps-1)
    result=dict(recovery_multinomial_draws=group*steps)
    if policy:result.update(policy_forward_calls=1,policy_forward_positions=positions,
        recovery_validation_calls=1,recovery_validation_positions=positions)
    return result


def application_upper(pool,*,validate=False):
    group,steps=pool['responses'].shape;positions=group*(pool['prompt_ids'].shape[1]+steps-1)
    result=dict(train_updates=1,response_targets=int(pool['mask'].sum()),
        policy_forward_calls=1,policy_forward_positions=positions,
        reference_forward_calls=1,reference_forward_positions=positions)
    if validate:
        for key,value in pool_validation_upper(pool,policy=True).items():result[key]=result.get(key,0)+value
        result['recovery_validation_operations']=1
    return result


def model_logits(model, ids):
    value = model(input_ids=ids, use_cache=False)
    return value.logits.float()


def available_gib():
    text = Path("/proc/meminfo").read_text()
    match = re.search(r"^MemAvailable:\s+(\d+) kB$", text, re.MULTILINE)
    if not match:
        raise RuntimeError("Cannot inspect MemAvailable on this platform")
    return int(match.group(1))*1024/2**30


def guard_memory(reserve=25.):
    measured = available_gib()
    if measured < reserve:
        raise RuntimeError(f"MemAvailable {measured:.2f} GiB below {reserve} GiB reserve")
    return measured


def collect_rollouts(model, reference, prompt_ids, expected, decode, eos_id,
                     generator, group_size=4, max_new_tokens=32,
                     before_stage=None, context_length=None, stop_ids=None,
                     source_id="wrapper-prompt", version=0, cursor=0, attempt=None,
                     work_ledger=None,_work_attempt=None):
    if (type(group_size) is not int or group_size<2 or type(max_new_tokens) is not int
            or max_new_tokens<1 or prompt_ids.ndim!=2 or prompt_ids.shape[0]!=1 or prompt_ids.shape[1]<1):
        raise ValueError('Invalid collection geometry before reservation')
    work=_work_attempt or _WorkAttempt(work_ledger,collection_upper(prompt_ids.shape[1],max_new_tokens,group_size),'rlvr-collection')
    try:
        work.before(collections=1,source_examples=1);work.after(source_examples=1)
        pool=_collect_rollouts(model,reference,prompt_ids,expected,decode,eos_id,generator,
            group_size,max_new_tokens,before_stage,context_length,stop_ids,source_id,version,cursor,attempt,work)
        work.after(collections=1)
        if _work_attempt is None:work.complete()
        return pool
    except BaseException as error:
        if _work_attempt is None:work.fail(error)
        raise


def _collect_rollouts(model, reference, prompt_ids, expected, decode, eos_id,
                     generator, group_size, max_new_tokens, before_stage, context_length,
                     stop_ids, source_id, version, cursor, attempt, work):
    """Collect a whole detached pool; partial attempts are not recovery state."""
    if (type(group_size) is not int or group_size < 2 or type(max_new_tokens) is not int
            or max_new_tokens < 1 or prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1
            or prompt_ids.shape[1] < 1 or prompt_ids.dtype != torch.long):
        raise ValueError("Need one prompt, G>=2 and positive generation cap")
    if type(eos_id) is not int or eos_id < 0:
        raise ValueError("A valid explicit EOS token ID is required")
    if context_length is not None and prompt_ids.shape[1]+max_new_tokens > context_length:
        raise ValueError("Prompt plus response cap exceeds model context length")
    stops_allowed = tuple(sorted(set(stop_ids or (eos_id,))))
    if any(type(token) is not int or token < 0 for token in stops_allowed):
        raise ValueError("Stop IDs must be explicit nonnegative integers")
    if type(expected) is not int or type(version) is not int or type(cursor) is not int:
        raise ValueError("Integer expected/version/cursor required")
    check = before_stage or (lambda: None)
    rng_before = generator.get_state().clone().cpu()
    policy_hash, reference_hash = state_digest(model.state_dict()), state_digest(reference.state_dict())
    model.eval()  # Disable dropout for behavior/current ratio alignment.
    reference.eval()
    prompts = prompt_ids.repeat(group_size, 1)
    current = prompts.clone()
    finished = torch.zeros(group_size, dtype=torch.bool, device=prompts.device)
    tokens, masks = [], []
    observed = attempt if attempt is not None else {}
    observed.update(stage="generation", sampled_steps=[], successful_generation_forward_calls=0,
                    successful_generation_forward_positions=0, successful_multinomial_draws=0,
                    successful_old_policy_forward_calls=0, successful_old_policy_forward_positions=0,
                    failure_cost_boundary="failed-call internal work unavailable; no mid-generation resume")
    with torch.no_grad():
        for _ in range(max_new_tokens):
            check()
            delta=dict(policy_forward_calls=1,policy_forward_positions=current.numel(),
                       generation_forward_calls=1,generation_forward_positions=current.numel())
            work.before(**delta)
            probabilities = model_logits(model, current)[:, -1].softmax(-1)
            work.after(**delta)
            observed["successful_generation_forward_calls"] += 1
            observed["successful_generation_forward_positions"] += current.numel()
            delta=dict(multinomial_draws=group_size,generated_slots=group_size,
                       valid_response_tokens=int((~finished).sum()))
            work.before(**delta)
            next_token = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
            work.after(**delta)
            observed["successful_multinomial_draws"] += group_size
            next_token = torch.where(finished, torch.full_like(next_token, eos_id), next_token)
            tokens.append(next_token)
            masks.append(~finished)
            observed["sampled_steps"].append(dict(ids=next_token.cpu().tolist(), valid=(~finished).cpu().tolist()))
            finished |= torch.isin(next_token, next_token.new_tensor(stops_allowed))
            current = torch.cat((current, next_token[:, None]), -1)
            if bool(finished.all()):
                break
    responses, mask = torch.stack(tokens, -1), torch.stack(masks, -1)
    start = prompts.shape[1]-1
    inputs = torch.cat((prompts, responses), -1)[:, :-1]
    observed['stage']='old-policy-score'
    check()
    with torch.no_grad():
        delta=dict(policy_forward_calls=1,policy_forward_positions=inputs.numel(),
                   old_policy_forward_calls=1,old_policy_forward_positions=inputs.numel())
        work.before(**delta)
        old_logits = model_logits(model, inputs)[:, start:]
        work.after(**delta)
        observed['successful_old_policy_forward_calls']=1
        observed['successful_old_policy_forward_positions']=inputs.numel()
        old_logp = old_logits.log_softmax(-1).gather(-1, responses[..., None]).squeeze(-1)
    texts, rewards, stops = [], [], []
    for row, valid in zip(responses, mask):
        active = row[valid].tolist()
        stop_positions = [i for i, token in enumerate(active) if token in stops_allowed]
        terminated = bool(stop_positions)
        text = decode(active[:stop_positions[0]] if terminated else active)
        texts.append(text)
        # A length-capped response is explicitly not accepted as a complete answer.
        rewards.append(float(terminated and verify_integer(text, expected)))
        stops.append("eos" if terminated else "token-limit")
    reward = torch.tensor(rewards, device=prompts.device)
    advantage = group_advantages(reward[None]).flatten()
    observed["stage"] = "complete-pool"
    pool = dict(schema="rlvr-complete-pool-v1", source_id=source_id, version=version,
        cursor_before=cursor, cursor_after=cursor+1, expected=expected,
        prompt_ids=prompt_ids.detach().cpu().clone(), responses=responses.detach().cpu(),
        mask=mask.detach().cpu(), old_logp=old_logp.detach().cpu(),
        texts=texts, rewards=rewards, stops=stops,
        final_stop_ids=[int(row[valid][-1]) if stop=="eos" else None
                        for row,valid,stop in zip(responses,mask,stops)],
        advantages=advantage.detach().cpu(), policy_sha256=policy_hash,
        reference_sha256=reference_hash, rng_before=rng_before,
        rng_after=generator.get_state().clone().cpu(),
        work=collection_work(prompt_ids.shape[1], responses.shape[1], group_size, int(mask.sum())))
    pool["pool_sha256"] = state_digest(pool)
    return pool


def collection_work(prompt_length, steps, group, valid):
    return dict(collections=1, generated_slots=group*steps, valid_response_tokens=valid,
                multinomial_draws=group*steps, generation_forward_calls=steps,
                generation_forward_positions=group*(steps*prompt_length+steps*(steps-1)//2),
                old_policy_forward_calls=1, old_policy_forward_positions=group*(prompt_length+steps-1))


def apply_rollouts(model, reference, pool, optimizer, beta=.02, before_stage=None, attempt=None,
                   work_ledger=None, _work_attempt=None):
    work=_work_attempt or _WorkAttempt(work_ledger,application_upper(pool),'rlvr-application')
    try:
        result=_apply_rollouts(model,reference,pool,optimizer,beta,before_stage,attempt,work)
        if _work_attempt is None:work.complete()
        return result
    except BaseException as error:
        if _work_attempt is None:work.fail(error)
        raise


def _apply_rollouts(model, reference, pool, optimizer, beta, before_stage, attempt, work):
    """Apply one retained whole pool without sampling; original reference is fixed."""
    check = before_stage or (lambda: None)
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("Finite nonnegative KL beta required")
    if pool["policy_sha256"] != state_digest(model.state_dict()) or pool["reference_sha256"] != state_digest(reference.state_dict()):
        raise ValueError("Stale policy or refreshed original reference")
    if any(parameter.requires_grad for parameter in reference.parameters()):
        raise ValueError("Original RLVR reference must be frozen")
    model.eval(); reference.eval()
    device = next(model.parameters()).device
    responses, mask, old_logp, advantage = (pool[key].to(device) for key in ("responses", "mask", "old_logp", "advantages"))
    prompts = pool["prompt_ids"].to(device).repeat(responses.shape[0],1)
    inputs = torch.cat((prompts,responses),-1)[:,:-1]
    start = prompts.shape[1]-1
    observed=attempt if attempt is not None else {}
    observed.update(stage='reference-score',successful_reference_forward_calls=0,
                    successful_policy_forward_calls=0,successful_reference_forward_positions=0,
                    successful_policy_forward_positions=0,
                    failure_cost_boundary='failed-call/backward/optimizer internal work unavailable')
    check()
    with torch.no_grad():
        delta=dict(reference_forward_calls=1,reference_forward_positions=inputs.numel())
        work.before(**delta)
        ref_logits = model_logits(reference, inputs)[:, start:]
        work.after(**delta)
    observed['successful_reference_forward_calls']=1;observed['successful_reference_forward_positions']=inputs.numel()
    check()
    observed['stage']='current-policy-score'
    delta=dict(policy_forward_calls=1,policy_forward_positions=inputs.numel(),response_targets=int(mask.sum()))
    work.before(**delta)
    logits = model_logits(model, inputs)[:, start:]
    work.after(**delta)
    observed['successful_policy_forward_calls']=1;observed['successful_policy_forward_positions']=inputs.numel()
    logp = logits.log_softmax(-1).gather(-1, responses[..., None]).squeeze(-1)
    ratio_error = float((logp.detach()-old_logp).abs()[mask].max())
    kl = exact_kl(logits, ref_logits)
    loss = clipped_objective(logp, old_logp, advantage, mask, kl=kl, beta=beta)
    optimizer.zero_grad(set_to_none=True)
    observed['stage']='backward'
    loss.backward()
    gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
    if not torch.isfinite(loss) or not torch.isfinite(torch.tensor(gradient)):
        raise RuntimeError("Nonfinite loss or gradient")
    check()
    observed['stage']='optimizer'
    work.before(train_updates=1)
    optimizer.step()
    work.after(train_updates=1)
    observed['stage']='completed-application'
    return {"loss": float(loss.detach()), "reward": sum(pool["rewards"])/responses.shape[0],
            "gradient_norm": gradient, "initial_log_ratio_max_abs": ratio_error,
            "exact_kl_nats_before_update": float((kl.detach()*mask).sum()/mask.sum()),
            "valid_response_tokens": int(mask.sum()), "texts": pool["texts"],
            "responses": responses.tolist(), "rewards": pool["rewards"], "stops": pool["stops"],
            "advantages": advantage.tolist()}


def update_model(model, reference, prompt_ids, expected, decode, eos_id,
                 optimizer, generator, group_size=4, max_new_tokens=32,
                 beta=.02, before_stage=None, context_length=None, stop_ids=None,work_ledger=None):
    """Preserved one-prompt collection/application numerical wrapper."""
    pool = collect_rollouts(model, reference, prompt_ids, expected, decode, eos_id,
        generator, group_size, max_new_tokens, before_stage, context_length, stop_ids,work_ledger=work_ledger)
    return apply_rollouts(model, reference, pool, optimizer, beta, before_stage,work_ledger=work_ledger)


def effective_config(model):
    """Exclude only HF's operational load location, not architecture/settings."""
    value = deepcopy(model.config.to_dict())
    value.pop("_name_or_path", None)
    return json.loads(json.dumps(value, allow_nan=False))


def observable_contract(*, parent_files, sources, inputs, environment, interface,
                        train, evaluation, seed, updates, group_size, max_new_tokens,
                        lr, beta, eos_id, stop_ids, context_length, dtype, device,work_budget=None,
                        snapshot_io_contract=None):
    interface = deepcopy(interface)
    # A local base path is provenance, not a token meaning. Its bytes are parent_files.
    local_id=interface.get("source", {}).get("tokenizer_id")
    if type(local_id) is str and local_id.startswith("/"):
        interface["source"]["tokenizer_id"] = "verified-local-parent-bytes"
    result=dict(schema="rlvr-runner-recovery-v1", parent_files=parent_files,
        sources=sources, inputs=inputs, environment=environment, interface=interface,
        encoded_train_sha256=state_digest(train), encoded_evaluation_sha256=state_digest(evaluation),
        train=deepcopy(train), evaluation=deepcopy(evaluation), seed=seed, updates=updates,
        group_size=group_size, max_new_tokens=max_new_tokens, lr=lr, beta=beta,
        eos_id=eos_id, stop_ids=list(stop_ids), context_length=context_length,
        dtype=dtype, device=device, objective="response-mean clipped GRPO; population advantages; exact reference KL",
        support="full-support temperature1", cap_reward=0., weight_decay=0., clip=1.,
        likelihood_validation_atol=1e-5)
    if work_budget is not None:result['work_budget']=validate_work_contract(work_budget)
    if snapshot_io_contract is not None:result['snapshot_io_contract']=validate_io_contract(snapshot_io_contract)
    return result


def full_contract(observed, loop):
    value = deepcopy(observed)
    value.update(loop_contract=deepcopy(loop.loop_contract), reference_sha256=loop.reference_sha256)
    return json.loads(json.dumps(value,allow_nan=False))


def verify_observable_contract(expected, observed):
    latent = {"loop_contract", "reference_sha256"}
    if type(expected) is not dict or set(expected) != set(observed) | latent:
        raise ValueError("Unknown/incomplete retained RLVR contract")
    if {key:value for key,value in expected.items() if key not in latent} != observed:
        raise ValueError("RLVR observable source/lock/parent/interface/data/recipe changed before model load")


def _integer(value, name, minimum=0):
    if type(value) is not int or not minimum <= value <= 2**63-1:
        raise ValueError(f"Invalid integer {name}")


def _add(total, delta):
    for key,value in delta.items():
        _integer(value,key)
        total[key] = total.get(key,0)+value


class RLVRLoop:
    """Actual cyclic runner loop with complete-pool and complete-update boundaries.

    Validation forwards on saved pending policy state are separate recovery work,
    not training work or a model-scale memory/performance claim.
    """
    def __init__(self, model, reference, optimizer, generator, train, decode, *,
                 eos_id, stop_ids, group_size, max_new_tokens, updates, seed,
                 context_length, beta=.02, guard=None,work_budget=None,work_ledger=None):
        for name,value,minimum in (("updates",updates,1),("group_size",group_size,2),
                                  ("max_new_tokens",max_new_tokens,1),("seed",seed,0),("eos_id",eos_id,0)):
            _integer(value,name,minimum)
        if updates>16 or group_size>8 or max_new_tokens>64 or not train or not math.isfinite(beta) or beta<0:
            raise ValueError("Invalid bounded RLVR recipe")
        self.model,self.reference,self.optimizer,self.generator=model,reference,optimizer,generator
        self.device=next(model.parameters()).device
        self.train,self.decode=deepcopy(train),decode
        self.eos_id,self.stop_ids=eos_id,sorted(set(stop_ids))
        if eos_id not in self.stop_ids or any(type(i) is not int or i<0 for i in stop_ids):
            raise ValueError("Explicit integer stops including EOS required")
        self.group_size,self.max_new_tokens,self.updates,self.seed=group_size,max_new_tokens,updates,seed
        self.context_length,self.beta,self.guard=context_length,beta,guard or (lambda:None)
        self.vocab=model.config.vocab_size
        if len({row['source_id'] for row in train})!=len(train):
            raise ValueError("Distinct frozen source IDs required")
        for row in train:
            if set(row)!={'source_id','prompt','prompt_ids','expected'} or type(row['source_id']) is not str or not row['source_id']:
                raise ValueError("Malformed frozen source row")
            if type(row['expected']) is not int or type(row['prompt']) is not str or not row['prompt_ids']:
                raise ValueError("Malformed source prompt/target")
            if any(type(token) is not int or not 0<=token<self.vocab for token in row['prompt_ids']):
                raise ValueError("Invalid prompt token IDs")
            if len(row['prompt_ids'])+max_new_tokens>context_length:
                raise ValueError("Frozen source exceeds context")
        self.parameters=list(model.parameters())
        names={id(p):name for name,p in model.named_parameters()}
        actual=[p for group in optimizer.param_groups for p in group['params']]
        if len(actual)!=len(self.parameters) or {id(p) for p in actual}!={id(p) for p in self.parameters}:
            raise ValueError("Optimizer must cover actual policy parameters exactly once")
        self.optimizer_parameters=actual
        if any(p.requires_grad for p in reference.parameters()):
            raise ValueError("Original reference must already be frozen")
        self.reference_sha256=state_digest(reference.state_dict())
        aliases={}
        for name,p in model.named_parameters(remove_duplicate=False):aliases.setdefault(id(p),[]).append(name)
        self.loop_contract=dict(model_config=effective_config(model),reference_config=effective_config(reference),
            model_class=type(model).__module__+'.'+type(model).__qualname__,
            reference_class=type(reference).__module__+'.'+type(reference).__qualname__,
            parameter_aliases=[group for group in aliases.values() if len(group)>1],
            attention_backend=str(model.config._attn_implementation),reference_attention_backend=str(reference.config._attn_implementation),
            layout={k:[list(v.shape),str(v.dtype)] for k,v in model.state_dict().items()},
            optimizer_names=[[names[id(p)] for p in group['params']] for group in optimizer.param_groups],
            optimizer_groups=[{k:v for k,v in group.items() if k!='params'} for group in optimizer.param_groups],
            train_sha256=state_digest(train),seed=seed,updates=updates,group_size=group_size,
            max_new_tokens=max_new_tokens,eos_id=eos_id,stop_ids=self.stop_ids,context_length=context_length,
            beta=beta,device=str(self.device),rollout_rng_bytes=generator.get_state().numel(),
            torch_rng_bytes=torch.get_rng_state().numel(),cuda_rng_count=torch.cuda.device_count() if self.device.type=='cuda' else 0)
        self.loop_contract=json.loads(json.dumps(self.loop_contract,allow_nan=False))
        if work_ledger is not None and work_budget is None:
            work_budget=work_budget_contract(work_ledger.limits,work_ledger.max_bytes)
        self.work_budget=None if work_budget is None else validate_work_contract(work_budget)
        self.work_ledger=work_ledger
        self.work_contract_sha256=None
        if self.work_budget is not None:self.loop_contract['work_budget']=deepcopy(self.work_budget)
        self.completed=self.cursor=0
        self.pending=None;self.history=[];self.collection_work={};self.application_work={}
        self.poisoned=False;self.attempts=[]

    def bind_work_ledger(self,ledger,contract):
        if (self.work_budget is None or contract.get('work_budget')!=self.work_budget
                or ledger.limits!=self.work_budget['limits'] or ledger.max_bytes!=self.work_budget['max_bytes']
                or ledger.contract_sha256!=canonical_hash(contract)):
            raise ValueError('Active work journal must match frozen RLVR scientific contract')
        self.work_ledger=ledger
        self.work_contract_sha256=canonical_hash(contract)

    def _budget_ready(self):
        if self.work_budget is not None and self.work_ledger is None:
            raise ValueError('Declared RLVR budget requires the retained active work journal')
        if self.work_ledger is not None and (self.work_budget is None
                or self.work_contract_sha256 is None or self.work_ledger.contract_sha256!=self.work_contract_sha256
                or self.work_ledger.limits!=self.work_budget['limits'] or self.work_ledger.max_bytes!=self.work_budget['max_bytes']):
            raise ValueError('Active work budget changed')

    def collect(self):
        if state_digest(self.train)!=self.loop_contract['train_sha256']:
            raise ValueError("Frozen encoded source records changed")
        if self.poisoned or self.pending is not None or self.completed>=self.updates:
            raise ValueError("Restore poisoned state or consume pending pool before collecting")
        self._budget_ready()
        row=self.train[self.cursor%len(self.train)]
        work=_WorkAttempt(self.work_ledger,collection_upper(len(row['prompt_ids']),self.max_new_tokens,self.group_size),'rlvr-collection')
        attempt=dict(source_id=row['source_id'],version=self.completed,cursor=self.cursor)
        self.attempts.append(attempt);self.poisoned=True
        try:
            pool=collect_rollouts(self.model,self.reference,torch.tensor([row['prompt_ids']],device=self.device),
                row['expected'],self.decode,self.eos_id,self.generator,self.group_size,self.max_new_tokens,
                self.guard,self.context_length,self.stop_ids,row['source_id'],self.completed,self.cursor,attempt,
                _work_attempt=work)
            work.complete()
        except BaseException as error:
            attempt['error']=dict(type=type(error).__name__,message=str(error));work.fail(error);raise
        self.pending=pool;self.cursor+=1;_add(self.collection_work,pool['work'])
        self.poisoned=False
        return deepcopy(pool)

    def apply_pending(self):
        if self.poisoned:
            raise ValueError("Interrupted update requires a durable restore")
        if self.pending is None:
            self.collect()
        self._budget_ready()
        work=_WorkAttempt(self.work_ledger,application_upper(self.pending,validate=True),'rlvr-application')
        self.poisoned=True
        pool=self.pending
        attempt=dict(kind='application',version=self.completed,pool_sha256=pool['pool_sha256'])
        self.attempts.append(attempt)
        try:
            work.before(recovery_validation_operations=1)
            self.validate_pool(self.pending,self.completed,self.completed,policy=self.model.state_dict(),_work_attempt=work)
            work.after(recovery_validation_operations=1)
            result=apply_rollouts(self.model,self.reference,pool,self.optimizer,self.beta,self.guard,attempt,_work_attempt=work)
            if not all(bool(torch.isfinite(p).all()) for p in self.parameters):
                raise ValueError("Nonfinite updated policy")
            work.complete()
        except BaseException as error:
            attempt['error']=dict(type=type(error).__name__,message=str(error));work.fail(error);raise
        self.completed+=1
        positions=self.group_size*(len(self.train[(self.completed-1)%len(self.train)]['prompt_ids'])+pool['responses'].shape[1]-1)
        _add(self.application_work,dict(updates=1,policy_forward_calls=1,reference_forward_calls=1,
                                       policy_forward_positions=positions,reference_forward_positions=positions))
        result.update(update=self.completed,behavior_version=self.completed-1,policy_version=self.completed,
            source_id=pool['source_id'],prompt=self.train[(self.completed-1)%len(self.train)]['prompt'],
            collection=deepcopy(pool),policy_sha256_after=state_digest(self.model.state_dict()))
        self.history.append(result);self.pending=None;self.poisoned=False
        return deepcopy(result)

    def validate_rng(self, value, *, rollout=False):
        size=self.loop_contract['rollout_rng_bytes' if rollout else 'torch_rng_bytes']
        if not isinstance(value,torch.Tensor) or value.dtype!=torch.uint8 or value.ndim!=1 or value.numel()!=size:
            raise ValueError("Invalid RNG shape/dtype")
        try:
            torch.Generator(device=self.device if rollout else 'cpu').set_state(value.cpu())
        except RuntimeError as error:
            raise ValueError("Invalid RNG bytes") from error

    def validate_pool(self,pool,version,cursor,*,policy=None,_work_attempt=None):
        self._budget_ready()
        costs=pool_validation_upper(pool,policy=policy is not None)
        costs['recovery_validation_operations']=1
        work=_work_attempt or _WorkAttempt(self.work_ledger,costs,'rlvr-pool-validation')
        try:
            if _work_attempt is None:work.before(recovery_validation_operations=1)
            self._validate_pool(pool,version,cursor,policy=policy,work=work)
            if _work_attempt is None:
                work.after(recovery_validation_operations=1);work.complete()
        except BaseException as error:
            if _work_attempt is None:work.fail(error)
            raise

    def _validate_pool(self,pool,version,cursor,*,policy=None,work):
        fields={'schema','source_id','version','cursor_before','cursor_after','expected','prompt_ids','responses','mask',
                'old_logp','texts','rewards','stops','final_stop_ids','advantages','policy_sha256','reference_sha256',
                'rng_before','rng_after','work','pool_sha256'}
        if type(pool) is not dict or set(pool)!=fields or pool['schema']!='rlvr-complete-pool-v1':
            raise ValueError("Invalid whole-pool schema")
        if state_digest({k:v for k,v in pool.items() if k!='pool_sha256'})!=pool['pool_sha256']:
            raise ValueError("Pending pool content identity changed")
        for key in ('version','cursor_before','cursor_after','expected'):
            _integer(pool[key],key,-2**63 if key=='expected' else 0)
        row=self.train[cursor%len(self.train)]
        if (pool['version']!=version or pool['cursor_before']!=cursor or pool['cursor_after']!=cursor+1
                or pool['source_id']!=row['source_id'] or pool['expected']!=row['expected']
                or pool['reference_sha256']!=self.reference_sha256):
            raise ValueError("Pool source/version/cursor/original reference mismatch")
        prompt=pool['prompt_ids'];responses=pool['responses'];mask=pool['mask'];old=pool['old_logp']
        if not isinstance(prompt,torch.Tensor) or prompt.dtype!=torch.long or prompt.tolist()!=[row['prompt_ids']]:
            raise ValueError("Pool prompt IDs changed")
        if (not isinstance(responses,torch.Tensor) or responses.dtype!=torch.long or responses.ndim!=2
                or responses.shape[0]!=self.group_size or not 1<=responses.shape[1]<=self.max_new_tokens
                or bool((responses<0).any()) or bool((responses>=self.vocab).any())):
            raise ValueError("Pool response IDs/geometry changed")
        if not isinstance(mask,torch.Tensor) or mask.dtype!=torch.bool or mask.shape!=responses.shape:
            raise ValueError("Invalid stop-inclusive mask")
        if int(mask.sum(-1).max())!=responses.shape[1]:
            raise ValueError("Pool contains sampled steps after every row already stopped")
        if not isinstance(old,torch.Tensor) or old.dtype!=torch.float32 or old.shape!=responses.shape or old.requires_grad or not bool(torch.isfinite(old).all()) or bool((old>0).any()):
            raise ValueError("Invalid detached old selected likelihoods")
        for key in ('texts','rewards','stops','final_stop_ids'):
            if type(pool[key]) is not list or len(pool[key])!=self.group_size:
                raise ValueError("Pool row evidence mismatch")
        for i,tokens in enumerate(responses.tolist()):
            ends=[j for j,t in enumerate(tokens) if t in self.stop_ids]
            length=ends[0]+1 if ends else len(tokens)
            valid=[j<length for j in range(len(tokens))]
            if mask[i].tolist()!=valid or (ends and any(t!=self.eos_id for t in tokens[length:])) or (not ends and len(tokens)!=self.max_new_tokens):
                raise ValueError("Mask must include first stop and exclude EOS fill/cap mismatch")
            text=self.decode(tokens[:length-1] if ends else tokens)
            reward=float(bool(ends) and verify_integer(text,row['expected']))
            if (type(pool['rewards'][i]) is not float or pool['rewards'][i]!=reward
                    or pool['texts'][i]!=text or pool['stops'][i]!=('eos' if ends else 'token-limit')
                    or pool['final_stop_ids'][i]!=(tokens[length-1] if ends else None)
                    or (ends and type(pool['final_stop_ids'][i]) is not int)):
                raise ValueError("Raw text/reward/stop evidence changed")
        advantage=group_advantages(torch.tensor(pool['rewards'],device=self.device)[None]).flatten().cpu()
        if not isinstance(pool['advantages'],torch.Tensor) or pool['advantages'].dtype!=torch.float32 or pool['advantages'].requires_grad or not torch.equal(pool['advantages'],advantage):
            raise ValueError("Population group advantages changed")
        expected_work=collection_work(len(row['prompt_ids']),responses.shape[1],self.group_size,int(mask.sum()))
        if type(pool['work']) is not dict or set(pool['work'])!=set(expected_work) or any(type(v) is not int for v in pool['work'].values()) or pool['work']!=expected_work:
            raise ValueError("Collected rectangular/valid work changed")
        self.validate_rng(pool['rng_before'],rollout=True);self.validate_rng(pool['rng_after'],rollout=True)
        # num_samples=1 uses the same rectangular RNG geometry, regardless of
        # probabilities. Retain both states and test actual engine consumption.
        replay=torch.Generator(device=self.device).set_state(pool['rng_before'].cpu())
        probabilities=torch.full((self.group_size,self.vocab),1/self.vocab,device=self.device,dtype=torch.float32)
        for _ in range(responses.shape[1]):
            work.before(recovery_multinomial_draws=self.group_size)
            torch.multinomial(probabilities,1,generator=replay)
            work.after(recovery_multinomial_draws=self.group_size)
        if not torch.equal(replay.get_state().cpu(),pool['rng_after'].cpu()):
            raise ValueError("Pool RNG after state differs from rectangular sampling work")
        if policy is not None:
            if pool['policy_sha256']!=state_digest(policy):
                raise ValueError("Pending pool belongs to another policy")
            self.guard()
            inputs=torch.cat((prompt.repeat(self.group_size,1),responses),-1)[:,:-1].to(self.device)
            parameters={k:v.to(self.device) for k,v in policy.items()}
            # Saved weights are temporarily bound, never applied to the live optimizer.
            mode=self.model.training
            try:
                self.model.eval()
                with torch.no_grad(),torch.random.fork_rng(devices=list(range(self.loop_contract['cuda_rng_count']))):
                    delta=dict(policy_forward_calls=1,policy_forward_positions=inputs.numel(),
                        recovery_validation_calls=1,recovery_validation_positions=inputs.numel())
                    work.before(**delta)
                    value=torch.func.functional_call(self.model,parameters,(),dict(input_ids=inputs,use_cache=False),tie_weights=False).logits.float()[:,prompt.shape[1]-1:]
                    work.after(**delta)
                    logp=value.log_softmax(-1).gather(-1,responses.to(self.device)[...,None]).squeeze(-1).cpu()
            finally:
                self.model.train(mode)
            if not torch.allclose(logp,old,atol=1e-5,rtol=0):
                raise ValueError("Old log probabilities disagree with saved pending policy")

    def snapshot_state(self):
        if self.poisoned:
            raise ValueError("Cannot save interrupted collection/partial update")
        self._budget_ready()
        result=dict(policy=self.model.state_dict(),reference=self.reference.state_dict(),optimizer=self.optimizer.state_dict(),
            loop_contract=deepcopy(self.loop_contract),cursor=self.cursor,collection_work=deepcopy(self.collection_work),
            application_work=deepcopy(self.application_work),history=deepcopy(self.history),pending=deepcopy(self.pending),
            rng=dict(python=random.getstate(),torch=torch.get_rng_state(),rollout=self.generator.get_state(),
                     cuda=torch.cuda.get_rng_state_all() if self.device.type=='cuda' else []))
        if self.work_ledger is not None:result['work_ledger']=self.work_ledger.snapshot()
        return result

    def validate_payload(self,payload):
        self._budget_ready()
        if type(payload) is not dict or not {'state','completed_updates','phase'}<=set(payload):
            raise ValueError('Invalid RLVR payload before reservation')
        state=payload['state'];k=payload['completed_updates'];_integer(k,'completed_updates')
        keys={'policy','reference','optimizer','loop_contract','cursor','collection_work','application_work','history','pending','rng'}
        if self.work_ledger is not None:keys.add('work_ledger')
        if type(state) is not dict or set(state)!=keys or state['loop_contract']!=self.loop_contract:
            raise ValueError('RLVR state schema/effective configuration changed')
        if self.work_ledger is not None:
            self.work_ledger.validate_snapshot(state.get('work_ledger'))
        elif 'work_ledger' in state:raise ValueError('Unbudgeted state cannot silently adopt a work journal')
        if k>self.updates or type(state.get('history')) is not list or len(state['history'])!=k:
            raise ValueError('Invalid bounded history before validation reservation')
        costs=dict(recovery_validation_operations=1)
        if any(type(row) is not dict or 'collection' not in row for row in state['history']):
            raise ValueError('Malformed retained history before validation reservation')
        pools=[(row['collection'],False) for row in state['history']]
        if state.get('pending') is not None:pools.append((state['pending'],True))
        for pool,policy in pools:
            for key,value in pool_validation_upper(pool,policy=policy).items():costs[key]=costs.get(key,0)+value
        work=_WorkAttempt(self.work_ledger,costs,'rlvr-payload-validation')
        try:
            work.before(recovery_validation_operations=1)
            self._validate_payload(payload,work)
            work.after(recovery_validation_operations=1);work.complete()
        except BaseException as error:work.fail(error);raise

    def _validate_payload(self,payload,work):
        state=payload['state'];k=payload['completed_updates'];_integer(k,'completed_updates')
        keys={'policy','reference','optimizer','loop_contract','cursor','collection_work','application_work','history','pending','rng'}
        if self.work_ledger is not None:keys.add('work_ledger')
        if set(state)!=keys or state['loop_contract']!=self.loop_contract or k>self.updates or state_digest(self.train)!=self.loop_contract['train_sha256']:
            raise ValueError("RLVR state schema/effective configuration changed")
        pending=state['pending'];phase='pending' if pending is not None else 'completed'
        if payload['phase']!=phase or (pending is not None and k==self.updates):
            raise ValueError("RLVR pending/completed phase mismatch")
        _integer(state['cursor'],'cursor')
        if state['cursor']!=k+int(pending is not None) or type(state['history']) is not list or len(state['history'])!=k:
            raise ValueError("Completed/pending source cursor/history mismatch")
        for role in ('policy','reference'):
            actual=self.model.state_dict()
            if not isinstance(state[role],dict) or set(state[role])!=set(actual):
                raise ValueError("Saved model keys changed")
            for name,tensor in state[role].items():
                if not isinstance(tensor,torch.Tensor) or tensor.shape!=actual[name].shape or tensor.dtype!=actual[name].dtype or not bool(torch.isfinite(tensor).all()):
                    raise ValueError("Saved model tensor layout/content invalid")
            for group in self.loop_contract['parameter_aliases']:
                if any(not torch.equal(state[role][group[0]],state[role][name]) for name in group[1:]):
                    raise ValueError("Saved tied parameter aliases disagree before application")
        if state_digest(state['reference'])!=self.reference_sha256:
            raise ValueError("Original reference reset/tampered")
        collection={};application={}
        previous_rng=torch.Generator(device=self.device).manual_seed(self.seed).get_state().cpu()
        previous_policy=self.reference_sha256
        for number,row in enumerate(state['history'],1):
            fields={'loss','reward','gradient_norm','initial_log_ratio_max_abs','exact_kl_nats_before_update',
                    'valid_response_tokens','texts','responses','rewards','stops','advantages','update',
                    'behavior_version','policy_version','source_id','prompt','collection','policy_sha256_after'}
            if type(row) is not dict or set(row)!=fields or type(row.get('update')) is not int or row['update']!=number:
                raise ValueError("Invalid committed history update")
            pool=row['collection'];self.validate_pool(pool,number-1,number-1,_work_attempt=work)
            if pool['policy_sha256']!=previous_policy or type(row.get('policy_sha256_after')) is not str or re.fullmatch('[0-9a-f]{64}',row['policy_sha256_after']) is None:
                raise ValueError("Historical behavior/policy digest lineage changed")
            previous_policy=row['policy_sha256_after']
            if not torch.equal(pool['rng_before'],previous_rng):
                raise ValueError("Historical rollout RNG chain changed")
            previous_rng=pool['rng_after'];_add(collection,pool['work'])
            positions=self.group_size*(pool['prompt_ids'].shape[1]+pool['responses'].shape[1]-1)
            _add(application,dict(updates=1,policy_forward_calls=1,reference_forward_calls=1,
                                  policy_forward_positions=positions,reference_forward_positions=positions))
            for key in ('loss','reward','gradient_norm','initial_log_ratio_max_abs','exact_kl_nats_before_update'):
                if type(row[key]) is not float or not math.isfinite(row[key]):
                    raise ValueError("Invalid numeric committed metric")
            if (row['texts']!=pool['texts'] or row['responses']!=pool['responses'].tolist()
                    or row['rewards']!=pool['rewards'] or row['advantages']!=pool['advantages'].tolist()
                    or row['stops']!=pool['stops'] or type(row['valid_response_tokens']) is not int
                    or row['valid_response_tokens']!=int(pool['mask'].sum())
                    or row['reward']!=sum(pool['rewards'])/self.group_size
                    or type(row['behavior_version']) is not int or row['behavior_version']!=number-1
                    or type(row['policy_version']) is not int or row['policy_version']!=number
                    or row['source_id']!=pool['source_id'] or row['prompt']!=self.train[(number-1)%len(self.train)]['prompt']
                    or row['gradient_norm']<0 or row['initial_log_ratio_max_abs']<0):
                raise ValueError("Committed metric/raw pool disagreement")
        if k and state['history'][-1]['policy_sha256_after']!=state_digest(state['policy']):
            raise ValueError("Last metric is not this completed policy")
        if not k and state_digest(state['policy'])!=self.reference_sha256:
            raise ValueError("Initial policy differs from original parent reference")
        if pending is not None:
            self.validate_pool(pending,k,k,policy=state['policy'],_work_attempt=work)
            if not torch.equal(pending['rng_before'],previous_rng):
                raise ValueError("Pending RNG is not the next declared source draw")
            previous_rng=pending['rng_after'];_add(collection,pending['work'])
        for key,expected in (('collection_work',collection),('application_work',application)):
            value=state[key]
            if type(value) is not dict or set(value)!=set(expected) or any(type(v) is not int for v in value.values()) or value!=expected:
                raise ValueError("Cumulative collection/application work mismatch")
        rng=state['rng']
        if type(rng) is not dict or set(rng)!={'python','torch','rollout','cuda'}:
            raise ValueError("Incomplete applicable RNG state")
        self.validate_rng(rng['torch']);self.validate_rng(rng['rollout'],rollout=True)
        if not torch.equal(rng['rollout'].cpu(),previous_rng):
            raise ValueError("Cursor and retained rollout RNG disagree")
        try:random.Random().setstate(rng['python'])
        except (TypeError,ValueError) as error:raise ValueError("Malformed Python RNG") from error
        if type(rng['cuda']) is not list or len(rng['cuda'])!=self.loop_contract['cuda_rng_count']:
            raise ValueError("Invalid CUDA RNG count/type")
        for i,value in enumerate(rng['cuda']):
            size=torch.cuda.get_rng_state(i).numel()
            if not isinstance(value,torch.Tensor) or value.dtype!=torch.uint8 or value.ndim!=1 or value.numel()!=size:
                raise ValueError("Invalid CUDA RNG layout before model application")
            try:torch.Generator(device=f'cuda:{i}').set_state(value.cpu())
            except RuntimeError as error:raise ValueError("Invalid CUDA RNG bytes") from error
        saved=state['optimizer'];original=self.optimizer.state_dict()
        if type(saved) is not dict or set(saved)!={'state','param_groups'} or saved['param_groups']!=original['param_groups']:
            raise ValueError("Optimizer recipe/group layout changed")
        ids=[i for group in saved['param_groups'] for i in group['params']]
        if any(type(i) is not int for i in ids) or any(type(i) is not int for i in saved['state']) or set(saved['state'])!=(set(ids) if k else set()):
            raise ValueError("Optimizer parameter ID coverage changed")
        for i,param in zip(ids,self.optimizer_parameters):
            if not k:break
            moment=saved['state'][i]
            if set(moment)!={'step','exp_avg','exp_avg_sq'}:
                raise ValueError("Unsupported AdamW moment schema")
            step=moment['step']
            if not isinstance(step,torch.Tensor) or step.ndim!=0 or step.dtype not in (torch.float32,torch.float64) or float(step)!=k:
                raise ValueError("Adam step must equal the completed cursor")
            for key in ('exp_avg','exp_avg_sq'):
                value=moment[key]
                if not isinstance(value,torch.Tensor) or value.shape!=param.shape or value.dtype!=param.dtype or not bool(torch.isfinite(value).all()) or (key=='exp_avg_sq' and bool((value<0).any())):
                    raise ValueError("AdamW moment layout/content invalid")

    def save(self,path,*,contract,parent_invocation,max_bytes,io_budget=None,work_receipt_path=None):
        validate_rlvr_io_budget(contract,io_budget,max_bytes=max_bytes)
        if io_budget is None and work_receipt_path is not None:
            raise ValueError('External work receipt requires the accounted RLVR snapshot hook')
        self._validate_work_contract(contract)
        state=self.snapshot_state();phase='pending' if self.pending is not None else 'completed'
        self.validate_payload(dict(state=state,phase=phase,completed_updates=self.completed))
        return save_snapshot(path,contract=contract,state=state,phase=phase,completed_updates=self.completed,
                             parent_invocation=parent_invocation,max_bytes=max_bytes,guard=self.guard,
                             io_budget=io_budget,work_receipt_path=work_receipt_path)

    def restore(self,path,*,contract,expected_sha256,expected_bytes,max_bytes,io_budget=None):
        validate_rlvr_io_budget(contract,io_budget,max_bytes=max_bytes)
        self._validate_work_contract(contract)
        payload=load_snapshot(path,expected_contract=contract,expected_sha256=expected_sha256,
            expected_bytes=expected_bytes,max_bytes=max_bytes,validate_payload=self.validate_payload,guard=self.guard,
            io_budget=io_budget)
        state=payload['state']
        self.model.load_state_dict(state['policy']);self.reference.load_state_dict(state['reference'])
        self.reference.eval().requires_grad_(False);self.optimizer.load_state_dict(state['optimizer'])
        self.completed=payload['completed_updates'];self.cursor=state['cursor']
        self.collection_work=deepcopy(state['collection_work']);self.application_work=deepcopy(state['application_work'])
        self.history=deepcopy(state['history']);self.pending=deepcopy(state['pending'])
        random.setstate(state['rng']['python']);torch.set_rng_state(state['rng']['torch']);self.generator.set_state(state['rng']['rollout'])
        if self.device.type=='cuda':torch.cuda.set_rng_state_all(state['rng']['cuda'])
        self.poisoned=False;self.model.eval()
        return payload

    def _validate_work_contract(self,contract):
        if self.work_budget is not None and (contract.get('work_budget')!=self.work_budget
                or canonical_hash(contract)!=self.work_contract_sha256):
            raise ValueError('Snapshot contract must match active work journal before work')
        if self.work_budget is None and 'work_budget' in contract:
            raise ValueError('Unbudgeted loop cannot claim budgeted snapshots')


def run_loop_with_recovery(loop,*,contract,output,parent_invocation,max_bytes,
                           metric_sink=lambda row:None,on_commit=lambda receipt:None,
                           baseline=lambda:None,final=lambda:None,export=lambda:None,io_budget=None):
    """Commit initial/pending/completed state before observers and metric writes."""
    validate_rlvr_io_budget(contract,io_budget,max_bytes=max_bytes)
    directory=Path(output)/'snapshots';directory.mkdir(exist_ok=False)
    def commit():
        phase='pending' if loop.pending is not None else 'completed'
        path=directory/f'{phase}-{loop.completed:06d}.pt'
        work_receipt_path=Path(str(path)+'.work.json') if io_budget is not None else None
        header=loop.save(path,contract=contract,parent_invocation=parent_invocation,max_bytes=max_bytes,
                         io_budget=io_budget,work_receipt_path=work_receipt_path)
        receipt=dict(path=str(path.resolve()),**header)
        if work_receipt_path is not None:receipt['work_receipt_path']=str(work_receipt_path.resolve())
        on_commit(receipt)
        return receipt
    latest=commit()
    with torch.random.fork_rng(devices=list(range(loop.loop_contract['cuda_rng_count']))):initial=baseline()
    while loop.completed<loop.updates:
        if loop.pending is None:
            loop.collect();latest=commit()
        row=loop.apply_pending();latest=commit();metric_sink(row)
    with torch.random.fork_rng(devices=list(range(loop.loop_contract['cuda_rng_count']))):ending=final()
    export()
    return dict(initial=initial,final=ending,latest=latest)


def prompt_contract(tokenizer, mode, genealogy=None, template=None):
    """Freeze raw/base or chat/SFT formatting and stopping identity, reject mismatch."""
    if mode not in ("raw", "chat") or tokenizer.eos_token_id is None:
        raise ValueError("Need a declared prompt mode and EOS identity")
    if template is not None:
        tokenizer.chat_template = template
    if mode == "raw" and genealogy is not None:
        raise ValueError("An SFT genealogy requires its saved chat template; raw prompting is a mismatch")
    stops = [tokenizer.eos_token_id]
    template_hash = None
    if mode == "chat":
        if not isinstance(tokenizer.chat_template, str) or not tokenizer.chat_template:
            raise ValueError("Chat mode requires one explicit saved or supplied template")
        template_hash = hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
        if genealogy is not None and genealogy.get("template_sha256") != template_hash:
            raise ValueError("Template differs from the SFT checkpoint genealogy")
        marker = tokenizer.encode("<|im_end|>", add_special_tokens=False)
        if len(marker) != 1 or marker[0] == tokenizer.unk_token_id:
            raise ValueError("Course chat contract needs the existing single-token im_end marker")
        stops.append(marker[0])
    def encode(prompt):
        if mode == "chat":
            ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
        else:
            ids = tokenizer.encode(prompt, add_special_tokens=False)
        return torch.tensor([ids], dtype=torch.long)
    return encode, sorted(set(stops)), {"prompt_mode": mode, "template_sha256": template_hash,
                                      "generation_stop_ids": sorted(set(stops)),
                                      "enable_thinking": False if mode == "chat" else None}


@torch.no_grad()
def evaluate_model(model, pairs, encode, decode, stop_ids, max_new_tokens,
                   context_length, device="cpu", before_stage=None, on_row=None,work_ledger=None):
    """Reserve an entire encoded greedy panel; partial failures stay observed."""
    if type(max_new_tokens) is not int or max_new_tokens<1 or not pairs:
        raise ValueError('Nonempty evaluation panel and positive response cap required')
    model.eval();encoded=[]
    for a,b in pairs:
        prompt=f'Return only the integer answer. {a} + {b} ='
        try:
            ids=encode(prompt).to(device)
            if ids.ndim!=2 or ids.shape[0]!=1 or ids.shape[1]<1 or ids.dtype!=torch.long:
                raise ValueError('One nonempty integer evaluation prompt required')
            if ids.shape[1]+max_new_tokens>context_length:
                raise ValueError('Evaluation prompt plus response cap exceeds model context')
            encoded.append(ids.clone())
        except BaseException as error:
            if on_row is not None:on_row(dict(prompt=prompt,expected=a+b,prompt_ids=None,text=None,
                tokens=[],raw_tokens=[],stop='error',correct=None,final_stop_id=None,truncated=None,
                error=dict(type=type(error).__name__,message=str(error)),generated_tokens=0,
                successful_forward_calls=0,successful_forward_positions=0,
                cost_boundary='encoding/pre-reservation failure; no model call'))
            raise
    positions=sum(max_new_tokens*ids.shape[1]+max_new_tokens*(max_new_tokens-1)//2 for ids in encoded)
    calls=len(encoded)*max_new_tokens
    work=_WorkAttempt(work_ledger,dict(evaluation_examples=len(encoded),evaluation_calls=calls,
        evaluation_positions=positions,evaluation_tokens=calls,policy_forward_calls=calls,
        policy_forward_positions=positions,generation_forward_calls=calls,
        generation_forward_positions=positions),'rlvr-evaluation-panel')
    try:
        result=_evaluate_model(model,pairs,encoded,decode,stop_ids,max_new_tokens,device,before_stage,on_row,work)
        work.complete();return result
    except BaseException as error:work.fail(error);raise


def _evaluate_model(model,pairs,encoded,decode,stop_ids,max_new_tokens,device,before_stage,on_row,work):
    """Frozen greedy exact-integer panel; all failures/truncations retained."""
    model.eval()
    check = before_stage or (lambda: None)
    rows = []
    for (a,b),encoded_ids in zip(pairs,encoded):
        prompt = f"Return only the integer answer. {a} + {b} ="
        row=dict(prompt=prompt,expected=a+b,prompt_ids=None,text=None,tokens=[],raw_tokens=[],
                 stop='in-progress',correct=None,final_stop_id=None,truncated=None,error=None,
                 successful_forward_calls=0,successful_forward_positions=0,
                 cost_boundary='successful calls only; failed-call internal work unavailable')
        output,ended=[],False
        try:
            ids=encoded_ids.clone();row['prompt_ids']=ids[0].tolist()
            work.before(evaluation_examples=1);work.after(evaluation_examples=1)
            for _ in range(max_new_tokens):
                check()
                delta=dict(policy_forward_calls=1,policy_forward_positions=ids.numel(),
                    generation_forward_calls=1,generation_forward_positions=ids.numel(),
                    evaluation_calls=1,evaluation_positions=ids.numel(),evaluation_tokens=1)
                work.before(**delta)
                token=int(model_logits(model,ids)[0,-1].argmax())
                work.after(**delta)
                row['successful_forward_calls']+=1;row['successful_forward_positions']+=ids.numel()
                row['raw_tokens'].append(token)
                if token in stop_ids:
                    ended=True;row['final_stop_id']=token;break
                output.append(token);row['tokens']=list(output)
                ids=torch.cat((ids,ids.new_tensor([[token]])),-1)
            text=decode(output)
            row.update(text=text,tokens=output,stop='declared-stop' if ended else 'token-limit',
                       correct=bool(ended and verify_integer(text,a+b)),truncated=not ended)
        except BaseException as error:
            row.update(stop='error',error=dict(type=type(error).__name__,message=str(error)),
                       generated_tokens=len(row['raw_tokens']))
            if on_row is not None:on_row(row)
            raise
        row['generated_tokens']=len(row['raw_tokens']);rows.append(row)
        if on_row is not None:
            on_row(rows[-1])
    return {"mode": "greedy-frozen-original-arithmetic", "n": len(rows),
            "accuracy": sum(row["correct"] for row in rows)/len(rows), "rows": rows}


def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_identity_files(identity,root,guard=lambda:None):
    verify_bounded_metadata(identity.get('bounded_snapshot_metadata',{}),guard)
    for group in ('source_sha256','input_sha256'):
        for name,expected in identity[group].items():
            path=Path(name);path=path if path.is_absolute() else Path(root)/path
            if group=='input_sha256' and name in identity.get('bounded_input_files',{}):
                guard();observed=verify_work_limits_file(path,expected)
            else:observed=file_digest(path,guard)
            if observed!=expected:
                raise ValueError("Actual source/input bytes changed")
    lock=identity['environment']['environment_lock']
    if file_digest(lock['path'],guard)!=lock['sha256'] or artifact_hashes(identity['checkpoint_path'],guard)!=identity['checkpoint_files']:
        raise ValueError("Actual lock/original parent bytes changed")


def jsonable(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().tolist()
    if isinstance(value,dict):return {key:jsonable(child) for key,child in value.items()}
    if isinstance(value,(tuple,list)):return [jsonable(child) for child in value]
    return value


class RunEvidence:
    """Persist progress in one newly created run directory, including failures.

    Each JSON snapshot is replaced atomically. Completed updates also append to
    a JSONL log. This journal is evidence persistence, not checkpoint recovery.
    """
    def __init__(self, output, config):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.invocation_id='rlvr-'+uuid4().hex
        self.data = {"status": "running", "stage": "created", "config": config,
                     "invocation_id":self.invocation_id,
                     "records": [], "initial_heldout_partial": []}
        self.flush()

    def flush(self):
        self.data["journal_seconds"] = time.monotonic()-self.started
        report = self.output/"report.json"
        temporary = self.output/"report.json.tmp"
        temporary.write_text(json.dumps(self.data, indent=2, allow_nan=False)+"\n")
        temporary.replace(report)
        status = {key: self.data[key] for key in ("status", "stage", "journal_seconds")}
        status["invocation_completed_records"] = len(self.data["records"])
        status["completed_updates"] = self.data.get("latest_durable_snapshot", {}).get(
            "completed_updates", len(self.data["records"]))
        status["durable_phase"] = self.data.get("latest_durable_snapshot", {}).get("phase")
        if "failure" in self.data:
            status["failure"] = self.data["failure"]
        temporary = self.output/"status.json.tmp"
        temporary.write_text(json.dumps(status, indent=2)+"\n")
        temporary.replace(self.output/"status.json")

    def stage(self, name, **fields):
        self.data.update(fields)
        self.data["stage"] = name
        self.flush()

    def baseline_row(self, row):
        self.data["initial_heldout_partial"].append(row)
        self.flush()

    def final_row(self, row):
        self.data.setdefault("final_heldout_partial", []).append(row)
        self.flush()

    def record(self, row):
        row=jsonable(row)
        with (self.output/"metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False)+"\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.data["records"].append(row)
        self.flush()

    def fail(self, error):
        self.data["status"] = "failed"
        if getattr(self,'work_ledger',None) is not None:
            self.data['work_ledger_current']=work_observation(self.work_ledger)
        if getattr(self,'snapshot_io_ledger',None) is not None:
            self.data['snapshot_io_ledger_current']=work_observation(self.snapshot_io_ledger)
        self.data["failure"] = {"type": type(error).__name__, "message": str(error),
                                "stage": self.data["stage"]}
        self.flush()

    def complete(self, manifest):
        self.data.update(manifest)
        self.data.update(status="completed", stage="completed")
        self.flush()


def snapshot_bootstrap_args(args,budget):
    """Bounded independent metadata; never read or generically hash the payload."""
    parsed,metadata=read_bounded_json(args.snapshot_io_limits)
    io_contract=validate_io_contract(parsed)
    if (type(args.snapshot_max_bytes) is not int or not 1<=args.snapshot_max_bytes<=2**63-1
            or io_contract['envelope']['max_payload_bytes']!=args.snapshot_max_bytes):
        raise ValueError('Explicit RLVR snapshot bound must match the snapshot I/O envelope')
    metadata={'snapshot_io_limits':metadata}
    receipt=retained=expected=None
    if args.resume:
        if args.resume_io_receipt is None:
            raise ValueError('Resume requires independently retained --resume-io-receipt')
        retained,metadata['resume_contract']=read_bounded_json(args.resume_contract,maximum=1024**2)
        value,metadata['resume_io_receipt']=read_bounded_json(args.resume_io_receipt)
        receipt=validate_resume_receipt(contract=retained,budget=budget,io_contract=io_contract,
            receipt=value,expected_sha256=args.resume_sha256,expected_bytes=args.resume_bytes,updates=args.updates)
        marker,metadata['resume_marker']=read_bounded_json(Path(str(args.resume)+'.commit.json'))
        validate_work_receipt(dict(receipt,snapshot=marker))
        if marker!=receipt['snapshot']:
            raise ValueError('Bounded RLVR marker differs from independent snapshot work receipt')
        expected=dict(path=os.path.abspath(args.resume),sha256=args.resume_sha256,bytes=args.resume_bytes)
    elif args.resume_io_receipt is not None:
        raise ValueError('Snapshot work receipt supplied without a resume payload')
    verify_bounded_metadata(metadata)
    return io_contract,metadata,receipt,retained,expected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True, help="Recorded immutable upstream commit SHA")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2323)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--prompt-mode", choices=("raw", "chat"), default="chat")
    parser.add_argument("--template", type=Path,
                        help="Explicit template if absent locally; must match a saved SFT genealogy")
    parser.add_argument("--max-seconds", type=float, default=600.,
                        help="Wall-clock cap checked between loading/generation/backward stages")
    parser.add_argument("--environment-lock", type=Path,
                        help="Required for actual execution; existing selected environment lock")
    parser.add_argument("--allow-legacy-interface", action="store_true",
                        help="Explicitly adopt an older course checkpoint's actual saved tokenizer")
    parser.add_argument("--resume",type=Path)
    parser.add_argument("--resume-contract",type=Path,help="Independently retained scientific contract")
    parser.add_argument("--resume-sha256")
    parser.add_argument("--resume-bytes",type=int)
    parser.add_argument("--snapshot-max-bytes",type=int,help="Explicit trusted-local save/load size envelope")
    parser.add_argument('--work-limits',type=Path,required=True,help='Exact immutable JSON of all23 logical work caps')
    parser.add_argument('--work-journal-max-bytes',type=int,required=True,help='Explicit journal envelope <=64MiB')
    parser.add_argument('--work-journal',type=Path,help='Same physical retained journal REQUIRED on resume')
    parser.add_argument('--snapshot-io-limits',type=Path,required=True,
                        help='Complete explicit nine-dimensional I/O contract with whole-operation envelope')
    parser.add_argument('--snapshot-io-ledger',type=Path,required=True,
                        help='Separate physical retained I/O journal; the same file is required on resume')
    parser.add_argument('--resume-io-receipt',type=Path,
                        help='Independent payload/science/I/O/runner-prefix receipt required on resume')
    args = parser.parse_args(argv)
    raw_limits=read_work_limits_file(args.work_limits)
    budget=work_budget_contract(_decode(raw_limits),args.work_journal_max_bytes)
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("revision must be a full immutable 40-character commit SHA")
    if not (1 <= args.updates <= 16 and 2 <= args.group_size <= 8 and 1 <= args.max_new_tokens <= 64):
        parser.error("bounded lab: updates1..16, group2..8, generation1..64")
    if not args.model_dir.is_dir() or args.output.exists() or not math.isfinite(args.lr) or args.lr <= 0 or not 0 < args.max_seconds <= 1800:
        parser.error("need existing local model dir, positive lr and unused output path")
    if args.resume and any(value is None for value in (args.resume_contract,args.resume_sha256,args.resume_bytes)):
        parser.error("Resume requires independently retained contract/SHA256/exact bytes")
    if args.resume and args.work_journal is None:
        parser.error('Resume requires the same physical retained --work-journal')
    if not args.resume and any(value is not None for value in (args.resume_contract,args.resume_sha256,args.resume_bytes)):
        parser.error("Resume expectations require an explicit payload")
    io_contract,metadata,receipt,retained,expected=snapshot_bootstrap_args(args,budget)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    journal = RunEvidence(args.output, config)
    journal.work_budget=budget;journal.work_ledger=journal.snapshot_io_ledger=None
    journal.snapshot_io_contract=io_contract;journal.bootstrap_metadata=metadata
    journal.independent_receipt=receipt;journal.retained_contract=retained
    journal.expected_resume_snapshot=expected
    journal.work_limits_sha256=hashlib.sha256(raw_limits).hexdigest()
    try:
        execute_run(args, journal)
    except BaseException as error:
        journal.fail(error)
        raise
    finally:
        try:
            if journal.snapshot_io_ledger is not None:journal.snapshot_io_ledger.close()
        finally:
            if journal.work_ledger is not None:journal.work_ledger.close()


def work_observation(ledger):
    try:return ledger.snapshot()
    except BaseException as error:return dict(status='unavailable',error=dict(type=type(error).__name__,message=str(error)),
        boundary='last durable prefix is separate; uncertain journal failure is not a reset')


def execute_run(args, journal):
    started = time.monotonic()
    minimum_available = float("inf")
    def resource_guard():
        nonlocal minimum_available
        if time.monotonic()-started >= args.max_seconds:
            raise RuntimeError("Bounded RLVR wall-clock budget exhausted")
        measured = guard_memory()
        minimum_available = min(minimum_available, measured)
        return measured
    journal.stage("identity-preflight")
    budget=getattr(journal,'work_budget',None)
    if budget is None:
        raw_limits=read_work_limits_file(args.work_limits)
        budget=work_budget_contract(_decode(raw_limits),args.work_journal_max_bytes)
        journal.work_limits_sha256=hashlib.sha256(raw_limits).hexdigest()
    budget=validate_work_contract(budget)
    io_contract=getattr(journal,'snapshot_io_contract',None)
    if io_contract is None:
        io_contract,metadata,receipt,retained,expected_payload=snapshot_bootstrap_args(args,budget)
        journal.snapshot_io_contract=io_contract;journal.bootstrap_metadata=metadata
        journal.independent_receipt=receipt;journal.retained_contract=retained
        journal.expected_resume_snapshot=expected_payload
    io_contract=validate_io_contract(io_contract)
    metadata=journal.bootstrap_metadata;receipt=journal.independent_receipt
    retained=journal.retained_contract;expected_payload=journal.expected_resume_snapshot
    verify_bounded_metadata(metadata)
    parsed_cap_hash=journal.work_limits_sha256
    if args.environment_lock is None:
        raise ValueError("Actual model execution requires --environment-lock")
    root = Path(__file__).resolve().parents[2]
    sources = [Path(__file__), root/"src/dongxi_llms/grpo_lab.py",
               root/"src/dongxi_llms/run_identity.py",root/"src/dongxi_llms/training_snapshot.py",
               root/"src/dongxi_llms/batched_cache_lab.py",root/'src/dongxi_llms/work_budget.py',
               root/'src/dongxi_llms/snapshot_io_budget.py',root/'src/dongxi_llms/artifact_budget.py']
    reject_payload_identity_aliases(args,sources)
    # The cap input has its own bounded reader, including identity and closure.
    inputs = [args.template] if args.template is not None else []
    initial_identity = collect_run_identity(root, source_files=sources, input_files=inputs,
        environment_lock=args.environment_lock, config=journal.data["config"],
        device={"mode": "cpu", "name": "preflight"})
    journal.stage("before-load", run_identity=initial_identity,parsed_work_limits_sha256=parsed_cap_hash)
    verify_work_limits_file(args.work_limits,parsed_cap_hash)
    attach_work_limits_identity(initial_identity,root,args.work_limits,parsed_cap_hash)
    bind_bounded_metadata(initial_identity,metadata,expected_snapshot=expected_payload)
    journal.stage('before-load',run_identity=initial_identity)
    if type(args.snapshot_max_bytes) is not int or not 1<=args.snapshot_max_bytes<=2**63-1:
        raise ValueError("Actual snapshot execution requires explicit positive --snapshot-max-bytes")
    resource_guard()
    verify_bounded_metadata(metadata,resource_guard)
    from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version
    journal.stage("tokenizer-load")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    resource_guard()
    if tokenizer.eos_token_id is None:
        raise ValueError("the supplied tokenizer has no EOS token ID")
    if (args.model_dir/"adapter_config.json").exists():
        raise ValueError("Supply a full or explicitly merged model, not a PEFT adapter directory")
    genealogy_path = args.model_dir/"course-genealogy.json"
    genealogy = json.loads(genealogy_path.read_text()) if genealogy_path.is_file() else None
    encode, stop_ids, interface = prompt_contract(tokenizer, args.prompt_mode, genealogy,
        args.template.read_text() if args.template is not None else None)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    template = tokenizer.chat_template if args.prompt_mode == "chat" else None
    if genealogy is not None:
        checkpoint_interface, adoption = parent_interface(args.model_dir, tokenizer,
            template=template, stop_ids=stop_ids, allow_legacy=args.allow_legacy_interface)
    else:
        checkpoint_interface = tokenizer_interface(tokenizer, template=template, stop_ids=stop_ids,
            tokenizer_id=str(args.model_dir.resolve()), tokenizer_revision=args.revision)
        adoption = {"legacy_adoption": False, "source": "local raw/base input, not a course parent"}
    reject_payload_identity_aliases(args,sources)
    identity = collect_run_identity(root, source_files=sources, input_files=inputs,
        checkpoint=args.model_dir, environment_lock=args.environment_lock,
        interface=checkpoint_interface, config=journal.data["config"],
        device={"mode": args.device, "name": "CPU" if args.device == "cpu" else "declared CUDA"},
        before_chunk=resource_guard)
    attach_work_limits_identity(identity,root,args.work_limits,parsed_cap_hash)
    bind_bounded_metadata(identity,metadata,expected_snapshot=expected_payload)
    if args.device=='cuda':
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Declared CUDA mode requires verified BF16 support")
        identity['device'].update(name=torch.cuda.get_device_name(),cuda_runtime=torch.version.cuda,
                                  gpu_driver=identity['gpu_driver'])
        identity['identity_sha256']=canonical_hash({k:v for k,v in identity.items() if k!='identity_sha256'})
    journal.stage('parent-identity-observed-before-data-validation',run_identity=identity,
                  checkpoint_interface=checkpoint_interface,interface=interface)
    verify_work_limits_file(args.work_limits,parsed_cap_hash)
    pairs = ((2, 3), (4, 5), (3, 6), (1, 7))
    heldout_pairs = ((2, 6), (3, 4), (5, 5), (7, 2))
    def source_rows(values,role):
        return [dict(source_id=f'{role}-{i}',prompt=f'Return only the integer answer. {a} + {b} =',
            prompt_ids=encode(f'Return only the integer answer. {a} + {b} =').flatten().tolist(),expected=a+b)
            for i,(a,b) in enumerate(values)]
    train_rows,evaluation_rows=source_rows(pairs,'train'),source_rows(heldout_pairs,'heldout')
    if {tuple(row['prompt_ids']) for row in train_rows}&{tuple(row['prompt_ids']) for row in evaluation_rows}:
        raise ValueError("Actual encoded train/evaluation prompts collide")
    declared_config=json.loads((args.model_dir/'config.json').read_text())
    context=declared_config.get('max_position_embeddings')
    if type(context) is not int or context<1:
        raise ValueError("Cannot determine original parent context before load")
    environment=identity['environment']
    science_environment={key:environment[key] for key in ('python_version','platform','machine','packages')}
    science_environment['lock_sha256']=environment['environment_lock']['sha256']
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    observed=observable_contract(parent_files=identity['checkpoint_files'],sources=identity['source_sha256'],
        inputs={'work-limits':parsed_cap_hash,'snapshot-io-limits':metadata['snapshot_io_limits']['sha256'],
                **({'template':file_digest(args.template,resource_guard)} if args.template is not None else {})},
        environment=science_environment,interface=checkpoint_interface,train=train_rows,evaluation=evaluation_rows,
        seed=args.seed,updates=args.updates,group_size=args.group_size,max_new_tokens=args.max_new_tokens,
        lr=args.lr,beta=.02,eos_id=tokenizer.eos_token_id,stop_ids=stop_ids,context_length=context,
        dtype=str(dtype),device=identity['device'],work_budget=budget,snapshot_io_contract=io_contract)
    work_ledger=io_ledger=io_hook=None
    if args.resume:
        journal.stage('resume-pre-model-contract-and-byte-check')
        verify_observable_contract(retained,observed)
        reject_payload_identity_aliases(args,sources)
        verify_identity_files(identity,root,resource_guard)
        work_ledger,io_ledger,io_hook=open_snapshot_budgets(contract=retained,budget=budget,
            io_contract=io_contract,receipt=receipt,work_path=args.work_journal,io_path=args.snapshot_io_ledger,
            invocation_id=journal.invocation_id,expected_sha256=args.resume_sha256,expected_bytes=args.resume_bytes)
        journal.work_ledger=work_ledger;journal.snapshot_io_ledger=io_ledger
        inspected=inspect_snapshot(args.resume,expected_sha256=args.resume_sha256,expected_bytes=args.resume_bytes,
            expected_contract=retained,max_bytes=args.snapshot_max_bytes,guard=resource_guard,io_budget=io_hook)
        bind_bounded_metadata(identity,metadata,expected_snapshot=expected_payload,verified_snapshot=inspected)
        journal.stage('resume-bytes-observed',run_identity=identity,verified_resume_snapshot=inspected)
    verify_bounded_metadata(metadata,resource_guard)
    journal.stage("model-load", interface=interface, checkpoint_interface=checkpoint_interface,
                  run_identity=identity, parent_genealogy=genealogy, legacy_adoption=adoption)
    if args.device == "cuda":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Declared CUDA mode requires verified BF16 support")
        torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True,
                                                torch_dtype=dtype, attn_implementation="sdpa").to(args.device)
    resource_guard()
    journal.stage("reference-copy")
    reference = deepcopy(model).eval()
    resource_guard()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    torch.manual_seed(args.seed)
    if set(pairs) & set(heldout_pairs):
        raise RuntimeError("Frozen arithmetic prompt pairs overlap")
    context = getattr(model.config, "max_position_embeddings", None)
    if context is None:
        raise ValueError("cannot determine the supplied model's context bound")
    decode = lambda tokens: tokenizer.decode(tokens, skip_special_tokens=True)
    loop=RLVRLoop(model,reference,optimizer,generator,train_rows,decode,eos_id=tokenizer.eos_token_id,
        stop_ids=stop_ids,group_size=args.group_size,max_new_tokens=args.max_new_tokens,updates=args.updates,
        seed=args.seed,context_length=int(context),guard=resource_guard,work_budget=budget)
    contract=full_contract(observed,loop)
    if retained is not None and retained!=contract:
        raise ValueError('Actual policy/reference/effective config/optimizer contract changed')
    work_path=args.work_journal
    if work_path is None:
        work_directory=args.output/'work-accounting';work_directory.mkdir(mode=0o700,exist_ok=False)
        work_path=work_directory/'journal.jsonl'
    if not args.resume:
        work_ledger,io_ledger,io_hook=open_snapshot_budgets(contract=contract,budget=budget,
            io_contract=io_contract,receipt=None,work_path=work_path,io_path=args.snapshot_io_ledger,
            invocation_id=journal.invocation_id)
        journal.work_ledger=work_ledger;journal.snapshot_io_ledger=io_ledger
    loop.bind_work_ledger(work_ledger,contract)
    journal.stage('work-journals-opened',work_journal=str(work_path.resolve()),work_budget=budget,
        snapshot_io_journal=str(args.snapshot_io_ledger.resolve()),snapshot_io_contract=io_contract)
    if retained is not None:
        journal.stage('restoring-completed-or-pending-state')
        payload=loop.restore(args.resume,contract=contract,expected_sha256=args.resume_sha256,
            expected_bytes=args.resume_bytes,max_bytes=args.snapshot_max_bytes,io_budget=io_hook)
        journal.stage('retained-parent-history',restored_completed=loop.completed,
                      restored_phase=payload['phase'],retained_committed_history=jsonable(loop.history))
    (args.output/'recovery-contract.json').write_text(json.dumps(contract,indent=2,allow_nan=False)+'\n')
    def committed(receipt):
        journal.stage('durable-'+receipt['phase'],latest_durable_snapshot=receipt)
    def baseline():
        journal.stage("initial-evaluation", frozen_panel_sha256=hashlib.sha256(repr(heldout_pairs).encode()).hexdigest())
        panel=evaluate_model(model,heldout_pairs,encode,decode,stop_ids,args.max_new_tokens,int(context),
            args.device,resource_guard,on_row=journal.baseline_row,work_ledger=work_ledger)
        journal.stage('ready-for-updates',initial_heldout=panel)
        return panel
    def final():
        journal.stage('final-evaluation')
        return evaluate_model(model,heldout_pairs,encode,decode,stop_ids,args.max_new_tokens,int(context),
                              args.device,resource_guard,on_row=journal.final_row,work_ledger=work_ledger)
    try:
        lifecycle=run_loop_with_recovery(loop,contract=contract,output=args.output,parent_invocation=journal.invocation_id,
            max_bytes=args.snapshot_max_bytes,metric_sink=journal.record,on_commit=committed,baseline=baseline,final=final,
            io_budget=io_hook)
    except BaseException:
        journal.stage('interrupted-attempt',partial_collection_attempts=jsonable(loop.attempts),
            live_state_poisoned=loop.poisoned,live_completed_updates=loop.completed,
            work_ledger_current=work_observation(work_ledger),
            snapshot_io_ledger_current=work_observation(io_ledger),
            boundary='replay only from last durable receipt; failed-call cost unavailable')
        raise
    initial_panel,final_panel=lifecycle['initial'],lifecycle['final']
    records = journal.data["records"]
    journal.stage("source-hashing", final_heldout=final_panel)
    reject_payload_identity_aliases(args,sources)
    verify_identity_files(identity,root,resource_guard)
    manifest = {"mode": "local-model-bounded-smoke", "upstream_revision": args.revision,
                "revision_evidence": "user-supplied metadata; local file hashes identify actual input bytes",
                "local_source_hashes": identity["checkpoint_files"],
                "run_identity": identity, "checkpoint_interface": checkpoint_interface,
                "config": {key: str(value) if isinstance(value, Path) else value
                           for key, value in vars(args).items()},
                "torch_version": torch.__version__, "transformers_version": transformers_version,
                "python_version": platform.python_version(), "machine": platform.machine(),
                "device_name": torch.cuda.get_device_name() if args.device == "cuda" else "CPU",
                "attention_backend": "Transformers SDPA; actual dispatched kernel requires profiling",
                "dtype": str(dtype), "minimum_sampled_memavailable_gib": minimum_available,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
                "source_command": identity['command'],
                "interface": interface, "parent_genealogy": genealogy,
                "seconds": time.monotonic()-started, "records": records,
                "frozen_panel_sha256": hashlib.sha256(repr(heldout_pairs).encode()).hexdigest(),
                "initial_heldout": initial_panel, "final_heldout": final_panel,
                "evaluation": "four original held-out integer prompts; descriptive pilot, no broad reasoning claim",
                "recovery": "completed and whole-pool pending resume; mid-generation continuation excluded",
                "latest_durable_snapshot":lifecycle['latest'],"cumulative_collection_work":loop.collection_work,
                "cumulative_application_work":loop.application_work,"committed_completed_updates":loop.completed}
    manifest['cumulative_work_ledger']=work_ledger.snapshot()
    manifest['work_journal']=str(work_path.resolve())
    manifest['cumulative_snapshot_io_ledger']=io_ledger.snapshot()
    manifest['snapshot_io_journal']=str(args.snapshot_io_ledger.resolve())
    manifest['snapshot_io_scope']='shared inspect/load/save visits only; caller capture, metadata/journal, application and artifact inventories excluded'
    journal.stage("checkpoint-export", **manifest)
    resource_guard()
    model.save_pretrained(args.output/"policy", safe_serialization=True)
    tokenizer.save_pretrained(args.output/"policy")
    (args.output/"policy"/"course-genealogy.json").write_text(json.dumps({
        "kind": "full-HF-model", "objective": "course-response-mean-GRPO",
        "parent_local_source_hashes": manifest["local_source_hashes"],
        "upstream_revision_metadata": args.revision, "template_sha256": interface["template_sha256"],
        "checkpoint_interface": checkpoint_interface, "legacy_adoption": adoption,
        "evaluation_sha256": manifest["frozen_panel_sha256"],
        "data_sha256": hashlib.sha256(repr(pairs).encode()).hexdigest()}, indent=2)+"\n")
    manifest["seconds"] = time.monotonic()-started
    journal.complete(manifest)
    print(json.dumps({"output": str(args.output), "updates": len(records),
                      "seconds": manifest["seconds"]}))


if __name__ == "__main__":
    main()
