#!/usr/bin/env python3
"""Optional bounded full-parameter DPO on a local SFT checkpoint, no implicit run.

    PYTHONPATH=src python scripts/run_chapter11_spark_dpo.py --help

JSONL pairs: id, prompt (list of role/content messages), chosen, rejected.
Independent generation JSONL: id, prompt (messages), expected (exact text).
This runner deliberately rejects truncation and template-prefix mismatches.
"""
import argparse
from contextlib import nullcontext
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time
import torch
from dongxi_llms.batched_cache_lab import digest as state_digest
from dongxi_llms.dpo_lab import model_pair_loss
from dongxi_llms.run_identity import (IdentityJournal, assert_compatible,
    canonical_hash, collect_run_identity, parent_interface, tokenizer_interface, write_json)
from dongxi_llms.training_snapshot import inspect_snapshot, load_snapshot, save_snapshot
from dongxi_llms.work_budget import WorkLedger, WorkBudgetExceeded, validate_limits, MAX_JOURNAL_BYTES
from dongxi_llms.artifact_budget import ArtifactBudget, MAX_JOURNAL_BYTES as MAX_ARTIFACT_JOURNAL_BYTES
from dongxi_llms.snapshot_io_budget import (SnapshotIOBudget, validate_io_contract,
    io_ledger_contract_sha256, read_bounded_json, read_work_receipt,validate_work_receipt)

_OUTPUT = None
_JOURNAL = None
_ATTEMPTED_WORK = None
_WORK_LEDGER = None
_ARTIFACT_LEDGER = None
_IO_LEDGER = None

BUDGET_KEYS = ("train_updates", "sampled_examples", "sampler_draws", "valid_targets",
               "logical_sequence_tokens", "policy_forward_calls", "policy_forward_positions",
               "reference_forward_calls", "reference_forward_positions", "evaluation_calls",
               "evaluation_positions", "generation_calls", "generation_position_upper_bound",
               "generation_tokens", "recovery_validation_operations", "recovery_history_rows",
               "recovery_tensor_elements", "recovery_rng_states", "recovery_sampler_draws")
RECOVERY_BUDGET_KEYS = BUDGET_KEYS[14:]
WORK_SCHEMA = "dongxi-dpo-logical-work-v2"


def read_work_limits_file(path):
    """Bounded initial CLI read; reject no-peer FIFOs before any read occurs."""
    path=Path(os.path.abspath(path));descriptors=[]
    try:
        parent=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        descriptors.append(parent)
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


def work_budget_contract(limits, max_bytes):
    validate_limits(limits)
    if set(limits) != set(BUDGET_KEYS) or type(max_bytes) is not int or not 1 <= max_bytes <= MAX_JOURNAL_BYTES:
        raise ValueError("Exact DPO dimensions and bounded work-journal bytes required")
    return dict(schema=WORK_SCHEMA, limits=dict(limits), max_bytes=max_bytes,
                reservations="permanent conservative upper bounds before whole known operation",
                scope="logical model positions/calls and semantic recovery verification units; not generic loader/serialization, FLOPs/backward recomputation or physical quotas")


def recovery_validation_upper(contract, completed):
    """Bound one semantic panel from expected layout, never inspecting tensors.

    Tensor units count a declared verifier envelope, not measured instructions:
    reference byte hashing + policy/reference finite panels, Adam finite/sign
    panels/steps, CPU RNG probes/equality buffers and CUDA RNG probes. Generic
    restricted-loader tree/finite checks before our callback remain excluded.
    """
    def integer(value, name, maximum=2**63-1, minimum=0):
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError("Invalid bounded recovery metadata: "+name)
        return value
    if type(contract) is not dict:
        raise ValueError("Plain expected recovery contract required")
    endpoint=integer(contract.get("updates"),"updates",1000,1)
    integer(completed,"completed",endpoint)
    accumulation=integer(contract.get("accumulation"),"accumulation",16,1)
    samples=contract.get("sample_work")
    if type(samples) is not list or not 1<=len(samples)<=4096:
        raise ValueError("Bounded expected sampler panel required")
    policy=contract.get("policy_shapes");optimizer=contract.get("optimizer_shapes")
    if type(policy) is not dict or not 1<=len(policy)<=4096 or type(optimizer) is not list or not 1<=len(optimizer)<=4096:
        raise ValueError("Bounded expected tensor layouts required")
    def elements(shape):
        if type(shape) is not dict or len(shape)!=2 or set(shape)!={"shape","dtype"}:
            raise ValueError("Exact expected tensor metadata required")
        dimensions=shape["shape"]
        if (type(dimensions) is not list or len(dimensions)>8
                or type(shape["dtype"]) is not str or not 1<=len(shape["dtype"])<=64):
            raise ValueError("Bounded expected tensor shape/dtype required")
        total=1
        for dimension in dimensions:
            total=integer(total*integer(dimension,"shape dimension",2**31-1),"tensor elements")
        return total
    if any(type(name) is not str or not 1<=len(name)<=256 for name in policy):
        raise ValueError("Bounded expected tensor names required")
    p=sum(elements(shape) for shape in policy.values())
    o=sum(elements(shape) for shape in optimizer)
    rng=integer(contract.get("torch_rng_bytes"),"CPU RNG bytes",65536,1)
    count=integer(contract.get("cuda_rng_count"),"CUDA RNG count",16)
    cuda=contract.get("cuda_rng_bytes")
    if type(cuda) is not list or len(cuda)!=count:
        raise ValueError("Exact bounded CUDA RNG layout required")
    cuda_elements=sum(integer(value,"CUDA RNG bytes",1024*1024,1) for value in cuda)
    costs=dict(recovery_validation_operations=1,recovery_history_rows=completed,
        recovery_tensor_elements=3*p+(3*o+len(optimizer) if completed else 0)+4*rng+cuda_elements,
        recovery_rng_states=2+count,recovery_sampler_draws=completed*accumulation)
    validate_limits(costs)
    return costs


def verify_parsed_work_limits(path, parsed_sha256, guard=lambda:None):
    guard()
    if hashlib.sha256(read_work_limits_file(path)).hexdigest()!=parsed_sha256:
        raise ValueError('Work limits bytes changed after the bounded initial parse')


def bind_parsed_work_limits(identity,path,parsed_sha256,guard=lambda:None):
    verify_parsed_work_limits(path,parsed_sha256,guard)
    identity['bounded_work_limits']={str(Path(os.path.abspath(path))):parsed_sha256}
    identity['identity_sha256']=canonical_hash({key:value for key,value in identity.items() if key!='identity_sha256'})
    return identity


def bind_bounded_metadata(identity, metadata, *, expected_snapshot=None, verified_snapshot=None):
    """Record bounded bootstrap bytes, never label expected payload bytes measured."""
    identity['bounded_snapshot_metadata']=dict(metadata)
    if expected_snapshot is not None:
        identity['expected_resume_snapshot']=dict(expected_snapshot,
            boundary='independently supplied expectation; not actual payload verification')
    if verified_snapshot is not None:
        identity['verified_resume_snapshot']=dict(verified_snapshot,
            boundary='actual shared inspection after independent journal admission')
    identity['identity_sha256']=canonical_hash({key:value for key,value in identity.items() if key!='identity_sha256'})
    return identity


def verify_bounded_metadata(metadata, guard=lambda:None):
    for name, expected in metadata.items():
        guard()
        _, observed=read_bounded_json(expected['path'])
        if observed!=expected:
            raise ValueError('Bounded snapshot bootstrap metadata changed: '+name)


def open_resume_budgets(*, contract, budget, io_contract, receipt, work_path, io_path,
                        invocation_id, expected_sha256, expected_bytes):
    """Bind independently retained prefixes before any payload/model operation."""
    if type(contract) is not dict or type(budget) is not dict or budget.get('schema')!=WORK_SCHEMA:
        raise ValueError('Plain retained science and exact DPO v2 work contract required')
    receipt=validate_work_receipt(receipt)
    io_contract=validate_io_contract(io_contract)
    if budget!=work_budget_contract(budget.get('limits'),budget.get('max_bytes')):
        raise ValueError('Retained DPO work schema/caps cannot be silently migrated')
    header=receipt['snapshot']
    if (type(contract.get('updates')) is not int or not 1<=contract['updates']<=1000
            or header['phase']!='completed' or header['completed_updates']>contract['updates']):
        raise ValueError('DPO resume requires a bounded completed boundary within the fixed endpoint')
    science=canonical_hash(contract)
    if contract.get('work_budget')!=budget or contract.get('snapshot_io_budget')!=io_contract:
        raise ValueError('Retained DPO science changes independently supplied work/I/O caps')
    if (header['contract_sha256']!=science or header['payload_sha256']!=expected_sha256
            or type(expected_bytes) is not int or header['payload_bytes']!=expected_bytes
            or receipt['runner_work_prefix'] is None):
        raise ValueError('Independent snapshot receipt differs from science/payload/runner expectation')
    work=io_ledger=None
    try:
        work=WorkLedger.open(work_path,limits=budget['limits'],contract_sha256=science,
            max_bytes=budget['max_bytes'],invocation_id=invocation_id,
            expected_snapshot=receipt['runner_work_prefix'])
        io_ledger=WorkLedger.open(io_path,limits=io_contract['limits'],
            contract_sha256=io_ledger_contract_sha256(io_contract,science),
            max_bytes=io_contract['max_journal_bytes'],invocation_id=invocation_id,
            expected_snapshot=receipt['io_prefix'])
        hook=SnapshotIOBudget(io_ledger,contract=io_contract,
            scientific_contract_sha256=science,expected_receipt=receipt)
        return work,io_ledger,hook
    except BaseException:
        if io_ledger is not None:io_ledger.close()
        if work is not None:work.close()
        raise


def snapshot_artifact_contract(*, max_bytes, max_entries, journal_max_bytes, payload_max_bytes):
    values=(max_bytes,max_entries,journal_max_bytes,payload_max_bytes)
    if any(type(value) is not int or not 1<=value<=2**63-1 for value in values):
        raise ValueError('Exact positive snapshot artifact capacities required')
    if not 4096<=journal_max_bytes<=min(max_bytes,MAX_ARTIFACT_JOURNAL_BYTES):
        raise ValueError('Snapshot artifact journal must fit the fixed byte envelope')
    return dict(schema='dongxi-dpo-snapshot-artifacts-v1',max_bytes=max_bytes,
        max_entries=max_entries,journal_max_bytes=journal_max_bytes,payload_max_bytes=payload_max_bytes,
        scope='direct snapshot payload/marker/staging pathnames only; not all outputs or a physical quota')


def validate_snapshot_artifact_ledger(contract, ledger):
    budget=contract.get('snapshot_artifact_budget')
    if budget is None:
        if ledger is not None:raise ValueError('Unbudgeted contract cannot adopt snapshot artifacts silently')
        return
    if not isinstance(ledger,ArtifactBudget):
        raise ValueError('Active snapshot artifact ledger required before state application')
    identity=ledger.identity
    if (identity['campaign_id']!=canonical_hash(contract)
            or any(identity[key]!=budget[key] for key in ('max_bytes','max_entries','journal_max_bytes'))):
        raise ValueError('Active snapshot artifact identity/capacity differs from DPO contract')


class _WorkAttempt:
    def __init__(self, ledger, costs, operation):
        self.ledger=ledger;self.ticket=None
        self.entered={key:0 for key in BUDGET_KEYS};self.success=dict(self.entered)
        if ledger is not None:self.ticket=ledger.reserve(costs,operation=operation)

    def before(self, **costs):
        candidate=dict(self.entered)
        for key,value in costs.items():candidate[key]+=value
        if self.ledger is not None:self.ledger.assert_within(self.ticket,candidate)
        self.entered=candidate

    def after(self, **costs):
        for key,value in costs.items():self.success[key]+=value

    def complete(self):
        if self.ledger is not None:self.ledger.complete(self.ticket,self.success)

    def fail(self, error):
        if self.ledger is not None and not self.ledger.poisoned:
            self.ledger.fail(self.ticket,type(error).__name__+": "+str(error)[:470],
                             known_actual=self.success,attempted=self.entered)


def update_upper(encoded_train, accumulation):
    rows=[example_work(encoded) for encoded in encoded_train]
    if not rows or type(accumulation) is not int or accumulation<=0:
        raise ValueError("Nonempty data and positive accumulation required before reservation")
    return dict(train_updates=1,sampled_examples=accumulation,sampler_draws=accumulation,
        valid_targets=accumulation*max(row["chosen_targets"]+row["rejected_targets"] for row in rows),
        logical_sequence_tokens=accumulation*max(row["logical_sequence_tokens"] for row in rows),
        policy_forward_calls=2*accumulation,reference_forward_calls=2*accumulation,
        policy_forward_positions=accumulation*max(row["policy_forward_positions"] for row in rows),
        reference_forward_positions=accumulation*max(row["reference_forward_positions"] for row in rows))

WORK_KEYS = ("sampled_pairs", "sampler_draws", "chosen_targets", "rejected_targets",
             "logical_sequence_tokens", "policy_forward_calls", "reference_forward_calls",
             "policy_forward_positions", "reference_forward_positions")


def zero_work():
    return {key: 0 for key in WORK_KEYS}


def example_work(encoded):
    chosen, rejected = encoded
    return dict(sampled_pairs=1, sampler_draws=1,
                chosen_targets=sum(chosen[1][1:]), rejected_targets=sum(rejected[1][1:]),
                logical_sequence_tokens=len(chosen[0])+len(rejected[0]),
                policy_forward_calls=2, reference_forward_calls=2,
                policy_forward_positions=len(chosen[0])+len(rejected[0])-2,
                reference_forward_positions=len(chosen[0])+len(rejected[0])-2)


def add_work(total, delta):
    for key in WORK_KEYS:
        total[key] += delta[key]


def completed_dpo_update(model, reference, optimizer, sampler, encoded_train, *,
                         pad_id, accumulation, beta, device, update,
                         autocast_factory=nullcontext, guard=lambda: None,
                         attempted_work=None, work_ledger=None):
    """Reserve the entire unshortened objective before any draw or forward."""
    attempt=_WorkAttempt(work_ledger,update_upper(encoded_train,accumulation),"dpo-update")
    try:
        result=_completed_dpo_update(model,reference,optimizer,sampler,encoded_train,
            pad_id=pad_id,accumulation=accumulation,beta=beta,device=device,update=update,
            autocast_factory=autocast_factory,guard=guard,attempted_work=attempted_work,attempt=attempt)
        attempt.after(train_updates=1);attempt.complete();return result
    except BaseException as error:
        attempt.fail(error);raise


def _completed_dpo_update(model, reference, optimizer, sampler, encoded_train, *,
                         pad_id, accumulation, beta, device, update,
                         autocast_factory, guard, attempted_work, attempt):
    """The actual runner update, also exercised unchanged by local CPU tests.

    Returned work is completed supervision. Mutable attempted_work charges each
    draw/forward before its call, retaining known attempted work on an exception.
    No partial backward or optimizer state returned here is a recovery boundary.
    """
    if type(accumulation) is not int or accumulation <= 0 or not encoded_train:
        raise ValueError("Nonempty training data and positive accumulation required")
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("Finite positive DPO beta required")
    if any(parameter.requires_grad for parameter in reference.parameters()):
        raise ValueError("DPO reference parameters must remain frozen")
    guard()
    model.train(); reference.eval()
    optimizer.zero_grad(set_to_none=True)
    losses, margins, indices, work = [], [], [], zero_work()
    def wrapper(network, role):
        def call(ids, attention):
            guard()
            costs={role+"_forward_calls":1,role+"_forward_positions":ids.numel()}
            attempt.before(**costs)
            if attempted_work is not None:
                attempted_work[role+"_forward_calls"] += 1
                attempted_work[role+"_forward_positions"] += ids.numel()
            result=network(input_ids=ids, attention_mask=attention, use_cache=False)
            attempt.after(**costs);return result
        return call
    for _ in range(accumulation):
        guard()
        # Draw allowance was reserved as part of the whole update, not after draw.
        attempt.before(sampled_examples=1,sampler_draws=1)
        index = int(torch.randint(len(encoded_train), (), generator=sampler))
        attempt.after(sampled_examples=1,sampler_draws=1)
        indices.append(index)
        delta = example_work(encoded_train[index])
        selected=dict(valid_targets=delta["chosen_targets"]+delta["rejected_targets"],
                      logical_sequence_tokens=delta["logical_sequence_tokens"])
        attempt.before(**selected);attempt.after(**selected)
        if attempted_work is not None:
            for key in ("sampled_pairs", "sampler_draws", "chosen_targets", "rejected_targets", "logical_sequence_tokens"):
                attempted_work[key] += delta[key]
        branches = [collate([encoded_train[index]], branch, pad_id, device) for branch in (0, 1)]
        with autocast_factory():
            loss, observations = model_pair_loss(wrapper(model, "policy"), wrapper(reference, "reference"),
                                                *branches, beta=beta)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite DPO loss")
        (loss/accumulation).backward()
        losses.append(float(loss.detach()))
        margins.append(float(observations["margin"].mean()))
        add_work(work, delta)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    attempt.before(train_updates=1)
    optimizer.step()
    return dict(update=update, loss=sum(losses)/len(losses), margin=sum(margins)/len(margins),
                gradient_norm=float(norm), indices=indices, work=work)


def parameter_contract(model):
    return {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in model.state_dict().items()}


def json_plain(value):
    return json.loads(json.dumps(value, allow_nan=False))


def optimizer_metadata_equal(actual,expected):
    """Compare only bounded observed group structure, without recursive encoding.

    Expected JSON lists may correspond to native optimizer tuples (betas). No
    unexpected caller container is traversed merely to normalize it to JSON.
    """
    remaining=[256]
    def same(left,right,depth=0):
        remaining[0]-=1
        if remaining[0]<0 or depth>8:return False
        if type(right) is dict:
            if type(left) is not dict or len(left)!=len(right) or len(right)>32:return False
            return all(key in left and same(left[key],value,depth+1) for key,value in right.items())
        if type(right) in (list,tuple):
            if type(left) not in (list,tuple) or len(left)!=len(right) or len(right)>32:return False
            return all(same(a,b,depth+1) for a,b in zip(left,right))
        if type(left) is not type(right):return False
        if type(left) is int and not -(2**63-1)<=left<=2**63-1:return False
        if type(left) is str and (len(left)!=len(right) or len(left)>256):return False
        if type(left) is float and not math.isfinite(left):return False
        return left==right
    return same(actual,expected)


def effective_model_contract(model):
    """Bind architecture and functional attention dropout, not only tensor sizes."""
    config = model.config.to_dict()
    config.pop("_name_or_path", None)  # invocation path is not an architecture
    dropout = {}
    for name, module in model.named_modules():
        for attribute in ("p", "attention_dropout", "resid_dropout", "hidden_dropout"):
            value = getattr(module, attribute, None)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(value) or value != 0:
                    raise ValueError("DPO recovery requires all observed dropout mechanisms disabled")
                dropout[name+":"+attribute] = float(value)
    for name, value in config.items():
        if "dropout" in name and isinstance(value, (int, float)) and value != 0:
            raise ValueError("DPO recovery requires zero dropout in the effective configuration")
    return json_plain(dict(config=config, module_dropout=dropout,
        attention_implementation=model.config._attn_implementation,
        gradient_checkpointing=bool(model.is_gradient_checkpointing),
        trainable_parameters=[name for name, value in model.named_parameters() if value.requires_grad],
        cache="use_cache=False on each training/reference/evaluation call"))


def observable_recovery_contract(*, parent_files, inputs, sources, environment, interface,
                                encoded_train, encoded_valid, evaluation_prefixes,
                                seed, updates, accumulation, beta, lr, max_length,
                                max_new_tokens, device, work_budget=None, snapshot_artifact_budget=None,
                                snapshot_io_budget=None):
    expected_budget=None
    if work_budget is not None:
        if type(work_budget) is not dict or work_budget.get('schema')!=WORK_SCHEMA:
            raise ValueError('Supplied budget requires explicit DPO v2 schema; no old-cap migration')
        expected_budget=work_budget_contract(work_budget.get('limits'),work_budget.get('max_bytes'))
        if work_budget!=expected_budget:
            raise ValueError('Supplied DPO v2 work contract differs from exact frozen schema')
    result=json_plain(dict(schema="dongxi-dpo-completed-v1", parent_files=parent_files,
                input_hashes=inputs, source_hashes=sources, environment=environment,
                checkpoint_interface=interface,
                torch_rng_bytes=torch.get_rng_state().numel(),
                cuda_rng_count=torch.cuda.device_count() if device.get("mode") == "cuda" else 0,
                cuda_rng_bytes=[value.numel() for value in torch.cuda.get_rng_state_all()] if device.get("mode") == "cuda" else [],
                encoded_train_sha256=state_digest(encoded_train),
                encoded_valid_sha256=state_digest(encoded_valid),
                evaluation_prefix_sha256=state_digest(evaluation_prefixes),
                sample_work=[example_work(row) for row in encoded_train],
                seed=seed, updates=updates, accumulation=accumulation, beta=beta,
                optimizer={"name": "AdamW", "lr": lr, "weight_decay": .01, "clip": 1.},
                max_length=max_length, max_new_tokens=max_new_tokens,
                device=device, objective="mean pairs; summed completion plus template endings; one causal shift",
                dropout="disabled", reference="original parent; fixed and detached"))
    if expected_budget is not None:
        result['work_budget']=expected_budget
    if snapshot_artifact_budget is not None:
        result['snapshot_artifact_budget']=snapshot_artifact_contract(**{key:snapshot_artifact_budget[key]
            for key in ('max_bytes','max_entries','journal_max_bytes','payload_max_bytes')})
    if snapshot_io_budget is not None:
        result['snapshot_io_budget']=validate_io_contract(snapshot_io_budget)
    return result


def make_recovery_contract(*, parent_files, inputs, sources, environment, interface,
                           model, reference, optimizer, encoded_train, encoded_valid, evaluation_prefixes,
                           seed, updates, accumulation, beta, lr, max_length, max_new_tokens, device, work_budget=None,
                           snapshot_artifact_budget=None,snapshot_io_budget=None):
    """Stable science identity; no output paths, deadline or snapshot cadence."""
    result = observable_recovery_contract(parent_files=parent_files, inputs=inputs, sources=sources,
                environment=environment, interface=interface, encoded_train=encoded_train,
                encoded_valid=encoded_valid, evaluation_prefixes=evaluation_prefixes,
                seed=seed, updates=updates, accumulation=accumulation, beta=beta, lr=lr,
                max_length=max_length, max_new_tokens=max_new_tokens, device=device,work_budget=work_budget,
                snapshot_artifact_budget=snapshot_artifact_budget,snapshot_io_budget=snapshot_io_budget)
    result.update(policy_shapes=parameter_contract(model),
                optimizer_shapes=[{"shape": list(value.shape), "dtype": str(value.dtype)} for value in model.parameters()],
                optimizer_parameter_names=[name for name, _ in model.named_parameters()],
                optimizer_group={key: value for key, value in optimizer.param_groups[0].items() if key != "params"},
                reference_sha256=state_digest(reference.state_dict()))
    result.update(policy_model=effective_model_contract(model), reference_model=effective_model_contract(reference))
    return json_plain(result)


def verify_observable_contract(expected, observed):
    latent = {"policy_shapes", "optimizer_shapes", "optimizer_parameter_names", "optimizer_group", "reference_sha256", "policy_model", "reference_model"}
    if not isinstance(expected, dict) or set(expected) != set(observed) | latent:
        raise ValueError("Unknown or incomplete independently retained DPO contract")
    if {key: value for key, value in expected.items() if key not in latent} != observed:
        raise ValueError("Observable source/data/parent/interface/recipe contract mismatch before model load")


def verify_identity_files(identity, root, guard=lambda: None):
    """Re-observe source/input/lock/parent bytes, not a descriptive stage label."""
    root = Path(root)
    for path,expected in identity.get('bounded_work_limits',{}).items():
        verify_parsed_work_limits(path,expected,guard)
    verify_bounded_metadata(identity.get('bounded_snapshot_metadata',{}),guard)
    for group in ("source_sha256", "input_sha256"):
        for name, expected in identity[group].items():
            path = Path(name)
            if not path.is_absolute():
                path = root/path
            if file_digest(path, guard) != expected:
                raise ValueError("Actual source/input bytes changed during the invocation")
    lock = identity["environment"]["environment_lock"]
    if file_digest(lock["path"], guard) != lock["sha256"]:
        raise ValueError("Actual selected environment lock changed")
    if identity.get("checkpoint_path"):
        from dongxi_llms.run_identity import artifact_hashes
        if artifact_hashes(identity["checkpoint_path"], guard) != identity["checkpoint_files"]:
            raise ValueError("Actual parent checkpoint artifacts changed")


def write_exclusive_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")
        handle.flush(); os.fsync(handle.fileno())


def snapshot_state(model, reference, optimizer, sampler, *, completed, counters, history, resume_parent=None,
                   work_ledger=None,snapshot_artifact_budget=None):
    result=dict(schema="dongxi-dpo-state-v1", policy=model.state_dict(), reference=reference.state_dict(),
                optimizer=optimizer.state_dict(), sampler_rng=sampler.get_state(),
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if next(model.parameters()).device.type == "cuda" else [],
                completed=completed, counters=dict(counters), history=list(history), resume_parent=resume_parent)
    if work_ledger is not None:result['work_ledger']=work_ledger.snapshot()
    if snapshot_artifact_budget is not None:result['snapshot_artifact_ledger']=snapshot_artifact_budget.receipt()
    return result


def validate_dpo_payload(payload, contract, work_ledger=None,snapshot_artifact_budget=None):
    """Permanently reserve each semantic panel before inspecting tensor values.

    Structural refusals and journal-prefix authentication must precede an open
    recovered ledger's reservation. Generic shared-loader validation before the
    callback and journal integrity work are outside these semantic units.
    Duplicate load-callback/restore checks deliberately cost separate panels.
    """
    if type(payload) is not dict or not isinstance(payload.get("state"),dict):
        raise ValueError("Plain completed payload/state required")
    state=payload["state"];completed=state.get("completed")
    costs=recovery_validation_upper(contract,completed)
    if (type(payload.get("completed_updates")) is not int or payload["completed_updates"]!=completed
            or payload.get("phase")!="completed" or type(state.get("history")) is not list
            or len(state["history"])!=completed or len(state)>14):
        raise ValueError("Bounded completed state/history structure required before validation")
    if 'work_budget' in contract:
        budget=contract['work_budget']
        if (type(budget) is not dict or budget.get('schema')!=WORK_SCHEMA
                or budget!=work_budget_contract(budget.get('limits'),budget.get('max_bytes'))):
            raise ValueError('Explicit DPO v2 validation-work budget required; no old-cap migration')
        if (not isinstance(work_ledger,WorkLedger) or work_ledger.limits!=budget['limits']
                or work_ledger.max_bytes!=budget['max_bytes']
                or work_ledger.contract_sha256!=canonical_hash(contract)):
            raise ValueError('Active cumulative ledger must match frozen DPO contract before validation')
        work_ledger.validate_snapshot(state.get('work_ledger'))
    elif work_ledger is not None:
        raise ValueError('Unbudgeted contract cannot adopt a cumulative ledger silently')
    attempt=_WorkAttempt(work_ledger,costs,'dpo-recovery-validation')
    try:
        # Conservative panel work is entered, not fabricated completed scans,
        # when a malformed value or probe fails. Reservations never refund.
        attempt.before(**costs)
        _validate_dpo_payload(payload,contract,work_ledger,snapshot_artifact_budget)
        attempt.after(**costs);attempt.complete()
    except BaseException as error:
        attempt.fail(error);raise


def _validate_dpo_payload(payload, contract, work_ledger=None,snapshot_artifact_budget=None):
    """Semantic counters/reference validation, inside the reserved panel."""
    state = payload["state"]
    required = {"schema", "policy", "reference", "optimizer", "sampler_rng", "torch_rng", "cuda_rng",
                "completed", "counters", "history", "resume_parent"}
    if 'work_budget' in contract:
        required.add('work_ledger')
    validate_snapshot_artifact_ledger(contract,snapshot_artifact_budget)
    if 'snapshot_artifact_budget' in contract:
        required.add('snapshot_artifact_ledger')
        snapshot_artifact_budget.validate_receipt(state.get('snapshot_artifact_ledger'))
    if set(state) != required or state["schema"] != "dongxi-dpo-state-v1" or payload["phase"] != "completed":
        raise ValueError("Unsupported DPO completed state")
    completed = state["completed"]
    if type(completed) is not int or not 0 <= completed <= contract["updates"] or completed != payload["completed_updates"]:
        raise ValueError("Invalid completed DPO update boundary")
    # Check actual tensor metadata against the reserved expected geometry before
    # hashing/scanning values; an oversized malformed reference must not be read.
    for key in ("policy", "reference"):
        if (not isinstance(state[key], dict) or len(state[key])!=len(contract["policy_shapes"])
                or set(state[key]) != set(contract["policy_shapes"])):
            raise ValueError("Policy/reference tensor keys differ from the parent architecture")
        for name, tensor in state[key].items():
            shape = contract["policy_shapes"][name]
            if (not isinstance(tensor, torch.Tensor) or tensor.layout!=torch.strided or tensor.is_quantized
                    or tensor.is_complex() or list(tensor.shape) != shape["shape"] or str(tensor.dtype) != shape["dtype"]):
                raise ValueError("Policy/reference tensor shape or dtype changed")
            if not torch.isfinite(tensor).all():
                raise ValueError("Nonfinite saved policy/reference tensor")
    if state_digest(state["reference"]) != contract["reference_sha256"]:
        raise ValueError("Original frozen DPO reference changed")
    if not isinstance(state["history"], list) or len(state["history"]) != completed:
        raise ValueError("Committed DPO numerical history differs from completed cursor")
    if not isinstance(state["counters"], dict) or len(state["counters"])!=len(WORK_KEYS) or set(state["counters"]) != set(WORK_KEYS) or any(type(value) is not int or not 0<=value<=2**63-1 for value in state["counters"].values()):
        raise ValueError("Cumulative DPO work counters must be exact nonnegative integers")
    counters, replay = zero_work(), torch.Generator().manual_seed(contract["seed"])
    for number, row in enumerate(state["history"], 1):
        if not isinstance(row, dict) or len(row)!=6 or set(row) != {"update", "loss", "margin", "gradient_norm", "indices", "work"} or type(row["update"]) is not int or row["update"] != number:
            raise ValueError("Invalid committed DPO metric row")
        if type(row["indices"]) is not list or len(row["indices"]) != contract["accumulation"] or any(type(value) is not int or not 0 <= value < len(contract["sample_work"]) for value in row["indices"]):
            raise ValueError("DPO indices must be bounded exact integers")
        if not isinstance(row["work"], dict) or len(row["work"])!=len(WORK_KEYS) or set(row["work"]) != set(WORK_KEYS) or any(type(value) is not int or not 0<=value<=2**63-1 for value in row["work"].values()):
            raise ValueError("DPO row work must be exact nonnegative integers")
        expected_indices = [int(torch.randint(len(contract["sample_work"]), (), generator=replay))
                            for _ in range(contract["accumulation"])]
        if row["indices"] != expected_indices:
            raise ValueError("Retained DPO indices differ from the declared sampler stream")
        work = zero_work()
        for index in expected_indices:
            add_work(work, contract["sample_work"][index])
        if row["work"] != work:
            raise ValueError("Retained completion/forward work differs from actual encoded masks")
        if any(type(row[key]) not in (int, float)
                or (type(row[key]) is int and not -(2**63-1)<=row[key]<=2**63-1)
                or not math.isfinite(row[key]) for key in ("loss", "margin", "gradient_norm")):
            raise ValueError("Nonfinite DPO metric")
        if row["loss"] < 0 or row["gradient_norm"] < 0:
            raise ValueError("DPO loss and gradient norm must be nonnegative")
        add_work(counters, work)
    if state["counters"] != counters or counters["sampler_draws"] != completed*contract["accumulation"]:
        raise ValueError("Inconsistent completed DPO work/draw counters")
    for name in ("sampler_rng", "torch_rng"):
        value = state[name]
        if not isinstance(value, torch.Tensor) or value.layout!=torch.strided or value.is_quantized or value.dtype != torch.uint8 or value.ndim != 1 or value.numel() != contract["torch_rng_bytes"]:
            raise ValueError("Invalid Torch RNG state")
        try:
            torch.Generator().set_state(value.cpu())
        except (RuntimeError, ValueError) as error:
            raise ValueError("Malformed Torch RNG bytes") from error
    if not torch.equal(state["sampler_rng"], replay.get_state()):
        raise ValueError("DPO sampler RNG differs from the completed draw cursor")
    if not isinstance(state["cuda_rng"], list) or len(state["cuda_rng"]) != contract["cuda_rng_count"] or any(not isinstance(value, torch.Tensor) or value.layout!=torch.strided or value.is_quantized or value.dtype != torch.uint8 or value.ndim != 1 or value.numel() != contract["cuda_rng_bytes"][index] for index, value in enumerate(state["cuda_rng"])):
        raise ValueError("Invalid CUDA RNG state")
    for index, value in enumerate(state["cuda_rng"]):
        try:
            torch.Generator(device=f"cuda:{index}").set_state(value.cpu())
        except (RuntimeError, ValueError) as error:
            raise ValueError("Malformed CUDA RNG bytes before state application") from error
    opt = state["optimizer"]
    if not isinstance(opt, dict) or len(opt)!=2 or set(opt) != {"state", "param_groups"} or not isinstance(opt["state"], dict) or len(opt["state"])>len(contract["optimizer_shapes"]) or not isinstance(opt["param_groups"], list) or len(opt["param_groups"]) != 1 or not isinstance(opt["param_groups"][0], dict) or len(opt["param_groups"][0])>32:
        raise ValueError("Invalid single-group AdamW state")
    group = opt["param_groups"][0]
    if not optimizer_metadata_equal({key: value for key, value in group.items() if key != "params"},contract["optimizer_group"]):
        raise ValueError("Saved optimizer changes the DPO recipe")
    parameters = list(range(len(contract["optimizer_shapes"])))
    if len(contract["optimizer_parameter_names"]) != len(parameters) or len(set(contract["optimizer_parameter_names"])) != len(parameters):
        raise ValueError("Optimizer parameter order is not fully named")
    if type(group["params"]) is not list or len(group["params"])!=len(parameters) or any(type(value) is not int for value in group["params"]) or group["params"] != parameters or any(type(value) is not int for value in opt["state"]) or set(opt["state"]) != (set(parameters) if completed else set()):
        raise ValueError("AdamW parameter or moment coverage differs from completed state")
    for index, moments in opt["state"].items():
        if type(moments) is not dict or len(moments)!=3 or set(moments) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("Unsupported saved AdamW moments")
        step = moments["step"]
        if not isinstance(step, torch.Tensor) or step.layout!=torch.strided or step.is_quantized or step.ndim != 0 or step.dtype != torch.float32 or float(step) != completed:
            raise ValueError("AdamW step disagrees with completed update cursor")
        shape = contract["optimizer_shapes"][index]
        for key in ("exp_avg", "exp_avg_sq"):
            value = moments[key]
            if not isinstance(value, torch.Tensor) or value.layout!=torch.strided or value.is_quantized or value.is_complex() or list(value.shape) != shape["shape"] or str(value.dtype) != shape["dtype"] or not torch.isfinite(value).all():
                raise ValueError("AdamW moment shape/dtype/finite state changed")
            if key == "exp_avg_sq" and (value < 0).any():
                raise ValueError("AdamW second moment must be nonnegative")


def restore_dpo_payload(payload, *, contract, model, reference, optimizer, sampler,work_ledger=None,
                        snapshot_artifact_budget=None):
    validate_dpo_payload(payload, contract,work_ledger,snapshot_artifact_budget)
    state = payload["state"]
    model.load_state_dict(state["policy"])
    reference.load_state_dict(state["reference"]); reference.eval().requires_grad_(False)
    optimizer.load_state_dict(state["optimizer"])
    sampler.set_state(state["sampler_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state["completed"], dict(state["counters"]), list(state["history"])


def append_metric(path, row):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, allow_nan=False, sort_keys=True)+"\n")
        handle.flush(); os.fsync(handle.fileno())


def train_completed_updates(model, reference, optimizer, sampler, encoded_train, *, contract,
                            output, invocation_id, pad_id, device, checkpoint_every, max_bytes,
                            completed=0, counters=None, history=None, parent_checkpoint=None,
                            autocast_factory=nullcontext, guard=lambda: None, attempted_work=None,
                            before_training=None,work_ledger=None,snapshot_artifact_budget=None,
                            snapshot_io_budget=None):
    """Periodically commit the actual training loop before metric/evaluation export.

    A save or metric failure leaves prior exclusive committed files unchanged.
    Completed history in the snapshot is authoritative, not an appended attempt.
    """
    output = Path(output)
    counters, history = dict(counters or zero_work()), list(history or [])
    if snapshot_io_budget is not None:
        if (not isinstance(snapshot_io_budget,SnapshotIOBudget)
                or contract.get('snapshot_io_budget')!=snapshot_io_budget.contract
                or snapshot_io_budget.science_sha256!=canonical_hash(contract)):
            raise ValueError('Actual DPO loop requires the independently bound snapshot I/O contract')
    elif 'snapshot_io_budget' in contract:
        raise ValueError('Accounted DPO science cannot silently use unhooked saves')
    validate_snapshot_artifact_ledger(contract,snapshot_artifact_budget)
    if snapshot_artifact_budget is None:
        checkpoints = output/"checkpoints"
        checkpoints.mkdir(exist_ok=False)
    else:
        if max_bytes!=contract['snapshot_artifact_budget']['payload_max_bytes']:
            raise ValueError('Snapshot payload envelope changed from the recovery contract')
        # A fresh invocation has unique filenames inside the same retained root.
        from uuid import uuid4
        checkpoints=snapshot_artifact_budget.root
        snapshot_namespace=uuid4().hex
    latest = parent_checkpoint
    if completed:
        for row in history:
            append_metric(output/"recovered-metrics.jsonl", dict(row, evidence="retained-parent-committed-row"))
    def commit(number):
        nonlocal latest
        guard()
        state = snapshot_state(model, reference, optimizer, sampler,
                               completed=number, counters=counters, history=history, resume_parent=parent_checkpoint,
                               work_ledger=work_ledger,snapshot_artifact_budget=snapshot_artifact_budget)
        synthetic = {"state": state, "phase": "completed", "completed_updates": number}
        validate_dpo_payload(synthetic, contract,work_ledger,snapshot_artifact_budget)
        # Commit must carry its own completed validation charge, not the older
        # pre-validation prefix from snapshot_state(). Loader/restore later retain
        # all further panels in the same journal instead of refunding them.
        if work_ledger is not None:state['work_ledger']=work_ledger.snapshot()
        name=f'completed-{number:06d}.pt' if snapshot_artifact_budget is None else f'{snapshot_namespace}-completed-{number:06d}.pt'
        path = checkpoints/name
        header = save_snapshot(path, contract=contract, state=state, completed_updates=number,
                               parent_invocation=invocation_id, max_bytes=max_bytes, guard=guard,
                               artifact_budget=snapshot_artifact_budget,io_budget=snapshot_io_budget,
                               work_receipt_path=Path(str(path)+'.work.json') if snapshot_io_budget is not None else None)
        latest = dict(path=str(path.resolve()), **header)
        if snapshot_io_budget is not None:
            latest['work_receipt_path']=str(Path(str(path)+'.work.json').resolve())
        return latest
    # Also copy a resumed boundary into its new invocation before diagnostics.
    commit(completed)
    baseline = None
    if before_training is not None:
        devices = list(range(contract["cuda_rng_count"]))
        with torch.random.fork_rng(devices=devices):
            baseline = before_training()
    for number in range(completed+1, contract["updates"]+1):
        row = completed_dpo_update(model, reference, optimizer, sampler, encoded_train,
                    pad_id=pad_id, accumulation=contract["accumulation"], beta=contract["beta"],
                    device=device, update=number, autocast_factory=autocast_factory,
                    guard=guard, attempted_work=attempted_work,work_ledger=work_ledger)
        add_work(counters, row["work"]); history.append(row)
        if number % checkpoint_every == 0 or number == contract["updates"]:
            commit(number)
        append_metric(output/"metrics.jsonl", dict(row, invocation_id=invocation_id,
                      last_committed_update=latest["completed_updates"],
                      snapshot=str(latest["path"])))
    return dict(completed=contract["updates"], counters=counters, history=history, checkpoint=latest, baseline=baseline)


def finalize_after_commit(training, *, evaluate, export, guard=lambda: None, cuda_devices=()):
    """Evaluation/export cannot precede the final durable completed boundary."""
    if training["checkpoint"]["completed_updates"] != training["completed"]:
        raise ValueError("Final evaluation/export requires the completed durable boundary")
    guard()
    with torch.random.fork_rng(devices=list(cuda_devices)):
        result = evaluate()
    guard()
    export()
    return result


def score_dpo_panel(model, reference, encoded_valid, *, pad_id, device, beta,
                    autocast_factory=nullcontext, guard=lambda:None,work_ledger=None):
    """Reserve the complete validation panel before either network executes."""
    if not encoded_valid:raise ValueError('Nonempty validation panel required')
    costs={key:0 for key in BUDGET_KEYS}
    for encoded in encoded_valid:
        row=example_work(encoded)
        costs['valid_targets']+=row['chosen_targets']+row['rejected_targets']
        costs['logical_sequence_tokens']+=row['logical_sequence_tokens']
        for role in ('policy','reference'):
            costs[role+'_forward_calls']+=2
            costs[role+'_forward_positions']+=row[role+'_forward_positions']
            costs['evaluation_calls']+=2
            costs['evaluation_positions']+=row[role+'_forward_positions']
    attempt=_WorkAttempt(work_ledger,costs,'dpo-validation-panel')
    def wrapper(network,role):
        def call(ids,attention):
            guard()
            delta={role+'_forward_calls':1,role+'_forward_positions':ids.numel(),
                   'evaluation_calls':1,'evaluation_positions':ids.numel()}
            attempt.before(**delta)
            result=network(input_ids=ids,attention_mask=attention,use_cache=False)
            attempt.after(**delta);return result
        return call
    try:
        model.eval();reference.eval();result=[]
        with torch.no_grad(),autocast_factory():
            for encoded in encoded_valid:
                guard();row=example_work(encoded)
                metadata=dict(valid_targets=row['chosen_targets']+row['rejected_targets'],
                              logical_sequence_tokens=row['logical_sequence_tokens'])
                attempt.before(**metadata);attempt.after(**metadata)
                branches=[collate([encoded],branch,pad_id,device) for branch in (0,1)]
                loss,observations=model_pair_loss(wrapper(model,'policy'),wrapper(reference,'reference'),*branches,beta=beta)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite validation loss')
                result.append(dict(loss=float(loss),margin=float(observations['margin'].mean())))
        attempt.complete();return result
    except BaseException as error:
        attempt.fail(error);raise


def generation_upper(prefixes,cap):
    if type(cap) is not int or cap<=0 or not prefixes or any(not prefix for prefix in prefixes):
        raise ValueError('Nonempty prompts and positive generation cap required')
    positions=sum(cap*len(prefix)+cap*(cap-1)//2 for prefix in prefixes)
    return dict(policy_forward_calls=len(prefixes)*cap,policy_forward_positions=positions,
        evaluation_calls=len(prefixes)*cap,evaluation_positions=positions,
        generation_calls=len(prefixes),generation_position_upper_bound=positions,
        generation_tokens=len(prefixes)*cap,
        logical_sequence_tokens=sum(len(prefix)+cap for prefix in prefixes))


def generate_dpo_panel(model,tokenizer,evaluation,prefixes,*,cap,stop_ids,device,
                       autocast_factory=nullcontext,guard=lambda:None,work_ledger=None,row_sink=None):
    """Greedy, single-sequence, uncached HF generation with pre-forward bounds.

    Successful raw continuation includes stopping IDs. Failed backend forwards
    retain entered logical geometry, not invented internal FLOPs/token outputs.
    """
    if len(evaluation)!=len(prefixes) or not stop_ids:raise ValueError('Aligned panel and explicit stops required')
    attempt=_WorkAttempt(work_ledger,generation_upper(prefixes,cap),'dpo-generation-panel')
    original=model.forward;rows=[];panel_error=None
    @wraps(original)
    def bounded_forward(*args,**kwargs):
        ids=kwargs.get('input_ids',args[0] if args else None)
        if not isinstance(ids,torch.Tensor) or ids.ndim!=2 or ids.shape[0]!=1:
            raise WorkBudgetExceeded('Unknown generation forward geometry refused before backend')
        guard();positions=ids.numel()
        delta=dict(policy_forward_calls=1,policy_forward_positions=positions,
            evaluation_calls=1,evaluation_positions=positions,generation_position_upper_bound=positions)
        attempt.before(**delta)
        result=original(*args,**kwargs)
        attempt.after(**delta);return result
    had_forward='forward' in model.__dict__;previous=model.__dict__.get('forward')
    try:
        model.eval();model.forward=bounded_forward
        with torch.inference_mode(),autocast_factory():
            for row,prefix in zip(evaluation,prefixes):
                guard();ids=torch.tensor([prefix],device=device);started=time.monotonic()
                attempt.before(generation_calls=1,logical_sequence_tokens=len(prefix))
                attempt.after(logical_sequence_tokens=len(prefix))
                fatal=None;continuation=None;stop=None
                try:
                    sampled=model.generate(ids,attention_mask=torch.ones_like(ids),do_sample=False,
                        num_beams=1,num_return_sequences=1,guidance_scale=1.0,
                        max_new_tokens=cap,use_cache=False,pad_token_id=tokenizer.pad_token_id,eos_token_id=stop_ids)
                    continuation=sampled[0,len(prefix):].tolist()
                    stop=continuation[-1] if continuation and continuation[-1] in stop_ids else None
                    if len(continuation)>cap:raise WorkBudgetExceeded('Decoder emitted beyond declared cap')
                    attempt.before(generation_tokens=len(continuation),logical_sequence_tokens=len(continuation))
                    attempt.after(generation_calls=1,generation_tokens=len(continuation),logical_sequence_tokens=len(continuation))
                    text=tokenizer.decode(continuation,skip_special_tokens=True)
                    retained=dict(id=row['id'],generated=text,raw_text=tokenizer.decode(continuation,skip_special_tokens=False),
                        generated_ids=continuation,prompt_ids=list(prefix),expected=row['expected'],
                        exact_match=text.strip()==row['expected'].strip(),stop_reason='declared-stop' if stop is not None else 'max_new_tokens',
                        final_stop_id=stop,truncated=stop is None,prompt_tokens=len(prefix),generated_tokens=len(continuation),
                        elapsed_seconds=time.monotonic()-started,error=None)
                except Exception as error:
                    panel_error=error
                    if isinstance(error,WorkBudgetExceeded):fatal=error
                    retained=dict(id=row['id'],prompt_ids=list(prefix),generated_ids=continuation,raw_text=None,generated=None,
                        expected=row['expected'],exact_match=None,stop_reason='error',truncated=None if continuation is None else stop is None,
                        generation_stop_reason=None if continuation is None else 'declared-stop' if stop is not None else 'max_new_tokens',
                        final_stop_id=stop,prompt_tokens=len(prefix),generated_tokens=None if continuation is None else len(continuation),elapsed_seconds=time.monotonic()-started,
                        error=dict(type=type(error).__name__,message=str(error)[:512]),
                        cost_boundary='entered/successful logical geometry retained; returned tokens retained if available; failed internal partial tokens/FLOPs unavailable')
                rows.append(retained)
                if row_sink is not None:row_sink(retained)
                if fatal is not None:raise fatal
                guard()
        if panel_error is None:attempt.complete()
        else:attempt.fail(panel_error)
        return rows
    except BaseException as error:
        attempt.fail(error);raise
    finally:
        if had_forward:model.forward=previous
        else:del model.forward


def check_split_interfaces(tokenizer, splits, encoded_splits):
    """Observed prefix collisions and available groups; missing groups stay a gap."""
    prefixes, groups = [], []
    for rows, encoded in zip(splits, encoded_splits):
        prefixes.append({tuple(tokenizer.apply_chat_template(row["prompt"], tokenize=True,
                        add_generation_prompt=True, enable_thinking=False, return_dict=False)) for row in rows})
        groups.append({row["group"] for row in rows if "group" in row})
        if any(not isinstance(row["group"], str) or not row["group"] for row in rows if "group" in row):
            raise ValueError("Explicit DPO source groups must be nonempty strings")
    for a, b in ((0, 1), (0, 2), (1, 2)):
        if prefixes[a] & prefixes[b]:
            raise ValueError("Actual encoded prompts overlap across DPO splits")
        if groups[a] & groups[b]:
            raise ValueError("Explicit source groups overlap across DPO splits")
    return dict(source_groups="fully-provided-disjoint" if all("group" in row for rows in splits for row in rows)
                else "missing-legacy-groups; source-independent split evidence still pending",
                encoded_prompt_collisions="none observed")


def file_digest(path, guard=None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            if guard is not None:
                guard()
            digest.update(chunk)
    return digest.hexdigest()


def host_available():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable unavailable; use the Linux Spark lane")


def load_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Need nonempty rows with unique ids")
    return rows


def restore_saved_template(checkpoint):
    """Read current HF Jinja storage, with a legacy JSON-template fallback.

    save_pretrained normally writes chat_template.jinja and removes the template
    from tokenizer_config.json. Reading only the latter breaks SFT -> DPO.
    """
    checkpoint = Path(checkpoint)
    jinja_path = checkpoint / "chat_template.jinja"
    config_path = checkpoint / "tokenizer_config.json"
    if jinja_path.is_file():
        template = jinja_path.read_text()
    elif config_path.is_file():
        template = json.loads(config_path.read_text()).get("chat_template")
    else:
        template = None
    if not isinstance(template, str) or not template:
        raise ValueError("SFT checkpoint has no single audited saved chat template")
    genealogy_path = checkpoint / "course-genealogy.json"
    if genealogy_path.is_file():
        expected = json.loads(genealogy_path.read_text()).get("template_sha256")
        actual = hashlib.sha256(template.encode()).hexdigest()
        if expected != actual:
            raise ValueError("Saved template differs from the checkpoint genealogy")
    return template


def verify_parent_tokenizer(checkpoint, saved, provided, *, tokenizer_id,
                            tokenizer_revision, allow_legacy=False):
    """Verify actual saved semantics, then the proposed downstream tokenizer."""
    template = restore_saved_template(checkpoint)
    def stops(tokenizer):
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        result = [tokenizer.eos_token_id]
        end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if end is not None and end != tokenizer.unk_token_id and end not in result:
            result.append(end)
        return result
    expected, adoption = parent_interface(checkpoint, saved, template=template,
        stop_ids=stops(saved), tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision, allow_legacy=allow_legacy)
    provided.chat_template = template
    observed = tokenizer_interface(provided, template=template, stop_ids=stops(provided),
        tokenizer_id=tokenizer_id, tokenizer_revision=tokenizer_revision)
    assert_compatible(expected, observed)
    return observed, adoption


def encode_pair(tokenizer, row, limit):
    prompt = row["prompt"]
    if not isinstance(prompt, list) or not prompt or prompt[-1]["role"] == "assistant":
        raise ValueError("prompt must be messages ending before the assistant response")
    prefix = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
    branches = []
    for key in ("chosen", "rejected"):
        if not isinstance(row[key], str) or not row[key].strip():
            raise ValueError("Both answers must be nonempty strings")
        ids = tokenizer.apply_chat_template(prompt + [{"role": "assistant", "content": row[key]}],
                                             tokenize=True, add_generation_prompt=False, enable_thinking=False, return_dict=False)
        if ids[:len(prefix)] != prefix:
            raise ValueError("Template does not preserve generation prefix; inspect the pinned tokenizer")
        if len(ids) > limit or len(ids) <= len(prefix):
            raise ValueError("Sequence exceeds limit or has an empty completion; filter explicitly")
        branches.append((ids, [False] * len(prefix) + [True] * (len(ids) - len(prefix))))
    return branches


def collate(encoded, branch, pad, device):
    records = [item[branch] for item in encoded]
    length = max(len(ids) for ids, _ in records)
    ids, attention, mask = [], [], []
    for tokens, completion in records:
        n = length - len(tokens)
        ids.append(tokens + [pad] * n)
        attention.append([1] * len(tokens) + [0] * n)
        mask.append(completion + [False] * n)
    return (torch.tensor(ids, device=device), torch.tensor(attention, device=device),
            torch.tensor(mask, dtype=torch.bool, device=device))


def _main(argv=None):
    global _OUTPUT, _JOURNAL, _ATTEMPTED_WORK, _WORK_LEDGER, _ARTIFACT_LEDGER, _IO_LEDGER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Local HF SFT checkpoint directory")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-revision", required=True, help="Exact 40-character HF commit")
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--evaluation", required=True, help="Independent prompt/expected JSONL")
    parser.add_argument("--output", required=True, help="New output directory; must not exist")
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--beta", type=float, default=.1)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1818)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--snapshot-max-bytes", type=int, required=True,
                        help="Explicit trusted-local save/load envelope; profile real checkpoint overhead separately")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-sha256", help="Independently retained snapshot payload SHA256, not a header shortcut")
    parser.add_argument("--resume-bytes", type=int)
    parser.add_argument("--resume-contract", type=Path,
                        help="Separately retained recovery-contract.json; required for resume")
    parser.add_argument('--work-limits',type=Path,required=True,
                        help='Immutable JSON of all 19 cumulative model/semantic-validation work caps; old14 refuses')
    parser.add_argument('--work-journal-max-bytes',type=int,required=True,
                        help='Explicit positive retained journal envelope, at most64MiB')
    parser.add_argument('--work-journal',type=Path,
                        help='Private owned journal path; REQUIRED same physical journal for resume')
    parser.add_argument('--snapshot-artifact-max-bytes',type=int,required=True,
                        help='Explicit aggregate snapshot pathname capacity, not a whole-output quota')
    parser.add_argument('--snapshot-artifact-max-entries',type=int,required=True)
    parser.add_argument('--snapshot-artifact-journal-max-bytes',type=int,required=True)
    parser.add_argument('--snapshot-artifact-root',type=Path,
                        help='New private flat root; REQUIRED same retained root for resume')
    parser.add_argument('--snapshot-artifact-receipt',type=Path,
                        help='Separately retained immutable ledger identity/prefix JSON; required for resume')
    parser.add_argument('--snapshot-io-limits',type=Path,required=True,
                        help='Explicit separate nine-dimensional v1 I/O contract with whole-operation envelope')
    parser.add_argument('--snapshot-io-ledger',type=Path,required=True,
                        help='Private persistent I/O journal; resume requires the same physical journal')
    parser.add_argument('--resume-io-receipt',type=Path,
                        help='Independent bounded payload/science/I/O/runner-prefix receipt required for resume')
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--environment-lock", required=True)
    parser.add_argument("--allow-legacy-interface", action="store_true",
                        help="Explicit adoption of actual saved tokenizer semantics, not old revision proof")
    args = parser.parse_args(argv)
    from dongxi_llms.work_budget import _decode
    raw_limits=read_work_limits_file(args.work_limits)
    parsed_limits_sha256=hashlib.sha256(raw_limits).hexdigest()
    budget=work_budget_contract(_decode(raw_limits),args.work_journal_max_bytes)
    io_value,io_identity=read_bounded_json(args.snapshot_io_limits)
    io_contract=validate_io_contract(io_value)
    if io_contract['envelope']['max_payload_bytes']!=args.snapshot_max_bytes:
        parser.error('Snapshot I/O payload envelope must equal the explicit snapshot byte bound')
    metadata={'snapshot_io_limits':io_identity}
    artifact_budget=snapshot_artifact_contract(max_bytes=args.snapshot_artifact_max_bytes,
        max_entries=args.snapshot_artifact_max_entries,journal_max_bytes=args.snapshot_artifact_journal_max_bytes,
        payload_max_bytes=args.snapshot_max_bytes)
    artifact_receipt=expected_contract=io_receipt=None
    if not (len(args.tokenizer_revision) == 40 and all(c in "0123456789abcdef" for c in args.tokenizer_revision)):
        parser.error("tokenizer revision must be an immutable 40-character lowercase commit")
    if not 1 <= args.updates <= 1000 or not 1 <= args.accumulation <= 16 or not math.isfinite(args.beta) or args.beta <= 0 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("Invalid bounded update/accumulation/beta/lr contract")
    if not 1 <= args.checkpoint_every <= 1000 or not 1 <= args.snapshot_max_bytes <= 2**63-1:
        parser.error("Positive bounded snapshot interval and explicit byte envelope required")
    if args.resume:
        if any(value is None for value in (args.resume_contract,args.resume_bytes,args.resume_sha256,
                args.work_journal,args.snapshot_artifact_root,args.snapshot_artifact_receipt,args.resume_io_receipt)):
            parser.error("Resume requires retained contract, SHA256, bytes, same journals and artifact root/receipt plus I/O work receipt")
        expected_contract,metadata['resume_contract']=read_bounded_json(args.resume_contract)
        artifact_receipt,metadata['snapshot_artifact_receipt']=read_bounded_json(args.snapshot_artifact_receipt)
        _,metadata['resume_io_receipt']=read_bounded_json(args.resume_io_receipt)
        io_receipt=read_work_receipt(args.resume_io_receipt)
        verify_bounded_metadata(metadata)
    elif any(value is not None for value in (args.resume_contract,args.resume_bytes,args.resume_sha256,
            args.snapshot_artifact_receipt,args.resume_io_receipt)):
        parser.error("Resume expectations require an explicit snapshot")
    if not 16 <= args.max_length <= 1024 or not 1 <= args.max_new_tokens <= 128:
        parser.error("Length limits exceed this teaching runner's bounds")
    if not Path(args.checkpoint).is_dir() or Path(args.output).exists():
        parser.error("Require existing local checkpoint and new output directory")
    checkpoint = Path(args.checkpoint)
    if (checkpoint / "adapter_config.json").exists() or not (checkpoint / "config.json").is_file():
        parser.error("DPO requires a full or merged HF checkpoint; merge a LoRA adapter explicitly first")
    if not any(checkpoint.glob("*.safetensors")) and not any(checkpoint.glob("pytorch_model*.bin")):
        parser.error("Full HF model weights are missing from the checkpoint")
    output = Path(args.output)
    output.mkdir(parents=True,mode=0o700)
    _OUTPUT = output
    journal = _JOURNAL = IdentityJournal(output, vars(args))
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), root / "src/dongxi_llms/dpo_lab.py",
               root / "src/dongxi_llms/run_identity.py", root / "src/dongxi_llms/training_snapshot.py",
               root / "src/dongxi_llms/batched_cache_lab.py",root/'src/dongxi_llms/work_budget.py',
               root/'src/dongxi_llms/artifact_budget.py',root/'src/dongxi_llms/snapshot_io_budget.py']
    # Cap bytes use the bounded role-specific reader, never the generic hasher.
    inputs = [args.train, args.validation, args.evaluation]
    expected_payload=(dict(path=str(args.resume),payload_sha256=args.resume_sha256,
        payload_bytes=args.resume_bytes) if args.resume else None)
    journal.attach(bind_bounded_metadata(bind_parsed_work_limits(collect_run_identity(root, source_files=sources, input_files=inputs,
        environment_lock=args.environment_lock, config=vars(args),
        device={"mode": "cpu", "name": "preflight"}),args.work_limits,parsed_limits_sha256),
        metadata,expected_snapshot=expected_payload))
    verify_parsed_work_limits(args.work_limits,parsed_limits_sha256)
    journal.stage("hardware_preflight")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or host_available() < 25 * 1024**3:
        raise RuntimeError("Require BF16-capable CUDA Spark and at least 25 GiB available host memory")
    start = time.monotonic()
    minimum_available = host_available()
    write_json(output / "status.json", {"status": "running", "config": vars(args)})
    def guard():
        nonlocal minimum_available
        minimum_available = min(minimum_available, host_available())
        if minimum_available < 25 * 1024**3 or time.monotonic() - start > 1800:
            raise RuntimeError("Sampled host memory reserve or 30-minute whole-run wall cap crossed")
    guard()
    journal.stage("reading_data")
    train, valid, evaluation = [load_rows(p) for p in (args.train, args.validation, args.evaluation)]
    if len(train) > 4096 or len(valid) > 128 or len(evaluation) > 64:
        raise ValueError("Teaching runner limits: 4096 train, 128 validation, 64 independent prompts")
    sets = [set(row["id"] for row in rows) for rows in (train, valid, evaluation)]
    if any(sets[a] & sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("IDs overlap across train, validation and independent evaluation")
    prompts = [set(json.dumps(row["prompt"], sort_keys=True) for row in rows) for rows in (train, valid, evaluation)]
    if any(prompts[a] & prompts[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Exact rendered-message prompts overlap across splits")
    from transformers import AutoTokenizer, AutoModelForCausalLM, __version__ as transformers_version
    verify_parsed_work_limits(args.work_limits,parsed_limits_sha256,guard)
    journal.stage("checking_parent_interface")
    saved_tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.tokenizer_revision,
                                             local_files_only=not args.allow_download)
    interface, adoption = verify_parent_tokenizer(checkpoint, saved_tokenizer, tokenizer,
        tokenizer_id=args.tokenizer, tokenizer_revision=args.tokenizer_revision,
        allow_legacy=args.allow_legacy_interface)
    stop_ids = interface["generation_stop_ids"]
    identity = bind_bounded_metadata(bind_parsed_work_limits(collect_run_identity(root, source_files=sources, input_files=inputs,
        checkpoint=checkpoint, environment_lock=args.environment_lock,
        interface=interface, config=vars(args), device={"mode": "cuda",
        "name": torch.cuda.get_device_name(), "cuda_runtime": torch.version.cuda}, before_chunk=guard),
        args.work_limits,parsed_limits_sha256,guard),metadata,expected_snapshot=expected_payload)
    journal.attach(identity)
    journal.stage("parent_interface_verified", legacy_adoption=adoption)
    write_json(output / "identity.json", identity)
    encoded_train = [encode_pair(tokenizer, r, args.max_length) for r in train]
    encoded_valid = [encode_pair(tokenizer, r, args.max_length) for r in valid]
    split_evidence = check_split_interfaces(tokenizer, (train, valid, evaluation),
                                           (encoded_train, encoded_valid, []))
    evaluation_prefixes = [tokenizer.apply_chat_template(row["prompt"], tokenize=True,
                             add_generation_prompt=True, enable_thinking=False, return_dict=False) for row in evaluation]
    if any(not prefix or len(prefix)+args.max_new_tokens > args.max_length for prefix in evaluation_prefixes):
        raise ValueError("Independent prompt plus generation exceeds declared context")
    environment = identity["environment"]
    science_environment = {key: environment[key] for key in ("python_version", "platform", "machine", "packages")}
    science_environment["environment_lock_sha256"] = environment["environment_lock"]["sha256"]
    science_environment["gpu_driver"] = identity["gpu_driver"]
    contract_options = dict(parent_files=identity["checkpoint_files"],
        inputs={"train": file_digest(args.train, guard), "validation": file_digest(args.validation, guard),
                "evaluation": file_digest(args.evaluation, guard),'work_limits':parsed_limits_sha256,
                'snapshot_io_limits':io_identity['sha256']}, sources=identity["source_sha256"],
        environment=science_environment, interface=interface, encoded_train=encoded_train,
        encoded_valid=encoded_valid, evaluation_prefixes=evaluation_prefixes, seed=args.seed,
        updates=args.updates, accumulation=args.accumulation, beta=args.beta, lr=args.lr,
        max_length=args.max_length, max_new_tokens=args.max_new_tokens, device=identity["device"],work_budget=budget,
        snapshot_artifact_budget=artifact_budget,snapshot_io_budget=io_contract)
    work_ledger=io_ledger=io_hook=None
    if args.resume:
        journal.stage("resume-pre-model-byte-and-observable-contract-check")
        verify_identity_files(identity, root, guard)
        verify_observable_contract(expected_contract, observable_recovery_contract(**contract_options))
        _WORK_LEDGER,_IO_LEDGER,io_hook=open_resume_budgets(contract=expected_contract,budget=budget,
            io_contract=io_contract,receipt=io_receipt,work_path=args.work_journal,
            io_path=args.snapshot_io_ledger,invocation_id=journal.invocation_id,
            expected_sha256=args.resume_sha256,expected_bytes=args.resume_bytes)
        work_ledger,io_ledger=_WORK_LEDGER,_IO_LEDGER
        inspected=inspect_snapshot(args.resume, expected_sha256=args.resume_sha256,
            expected_bytes=args.resume_bytes, expected_contract=expected_contract,
            max_bytes=args.snapshot_max_bytes, guard=guard,io_budget=io_hook)
        bind_bounded_metadata(identity,metadata,expected_snapshot=expected_payload,verified_snapshot=inspected)
        journal.attach(identity);write_json(output/'identity.json',identity)
    guard()
    torch.manual_seed(args.seed)
    verify_parsed_work_limits(args.work_limits,parsed_limits_sha256,guard)
    journal.stage("loading_policy_and_reference")
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, local_files_only=True,
             torch_dtype=torch.float32, attn_implementation="sdpa").cuda()
    guard()
    reference = AutoModelForCausalLM.from_pretrained(args.checkpoint, local_files_only=True,
             torch_dtype=torch.float32, attn_implementation="sdpa").cuda().eval().requires_grad_(False)
    guard()
    model.config.use_cache = False
    reference.config.use_cache = False
    model.gradient_checkpointing_enable()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    rng = torch.Generator().manual_seed(args.seed)
    contract = make_recovery_contract(model=model, reference=reference, optimizer=optimizer, **contract_options)
    if args.resume and contract!=expected_contract:
        raise ValueError('Actual original reference/model/optimizer layout differs from independently retained contract')
    work_path=args.work_journal or output/'work-ledger.jsonl'
    ledger_options=dict(limits=budget['limits'],contract_sha256=canonical_hash(contract),
        max_bytes=budget['max_bytes'],invocation_id=journal.invocation_id)
    if not args.resume:
        _WORK_LEDGER=work_ledger=WorkLedger.create(work_path,**ledger_options)
        _IO_LEDGER=io_ledger=WorkLedger.create(args.snapshot_io_ledger,limits=io_contract['limits'],
            contract_sha256=io_ledger_contract_sha256(io_contract,canonical_hash(contract)),
            max_bytes=io_contract['max_journal_bytes'],invocation_id=journal.invocation_id)
        io_hook=SnapshotIOBudget(io_ledger,contract=io_contract,scientific_contract_sha256=canonical_hash(contract))
    artifact_root=args.snapshot_artifact_root or output/'checkpoints'
    if args.resume:
        _ARTIFACT_LEDGER=snapshot_ledger=ArtifactBudget.restore(artifact_root,expected_receipt=artifact_receipt)
    else:
        _ARTIFACT_LEDGER=snapshot_ledger=ArtifactBudget.create(artifact_root,campaign_id=canonical_hash(contract),
            **{key:artifact_budget[key] for key in ('max_bytes','max_entries','journal_max_bytes')})
    validate_snapshot_artifact_ledger(contract,snapshot_ledger)
    completed, counters, history, parent_checkpoint = 0, zero_work(), [], None
    if args.resume:
        journal.stage("restoring-completed-dpo-state")
        verify_identity_files(identity, root, guard)
        payload = load_snapshot(args.resume, expected_sha256=args.resume_sha256,
            expected_bytes=args.resume_bytes, expected_contract=contract,
            max_bytes=args.snapshot_max_bytes, validate_payload=lambda saved: validate_dpo_payload(saved, contract,work_ledger,snapshot_ledger),
            guard=guard,io_budget=io_hook)
        completed, counters, history = restore_dpo_payload(payload, contract=contract,
                              model=model, reference=reference, optimizer=optimizer, sampler=rng,
                              work_ledger=work_ledger,snapshot_artifact_budget=snapshot_ledger)
        parent_checkpoint = dict(path=str(args.resume.resolve()), payload_sha256=args.resume_sha256,
                                payload_bytes=args.resume_bytes, completed_updates=completed,
                                producing_invocation=payload["parent_invocation"])
    write_exclusive_json(output/"recovery-contract.json", contract)
    _ATTEMPTED_WORK = zero_work()
    journal.data["attempted_training_work_this_invocation"] = _ATTEMPTED_WORK
    guard()
    def independent_evaluate(stage):
        return generate_dpo_panel(model,tokenizer,evaluation,evaluation_prefixes,
            cap=args.max_new_tokens,stop_ids=stop_ids,device='cuda',
            autocast_factory=lambda:torch.autocast('cuda',dtype=torch.bfloat16),guard=guard,
            work_ledger=work_ledger,row_sink=lambda row:append_metric(output/f'evaluation-{stage}-raw.jsonl',row))
    def baseline():
        journal.stage("baseline_evaluation", restored_update=completed)
        rows = independent_evaluate('before')
        write_exclusive_json(output/"evaluation-before.json", rows)
        return rows
    journal.stage("training-and-completed-snapshots")
    training = train_completed_updates(model, reference, optimizer, rng, encoded_train,
        contract=contract, output=output, invocation_id=journal.invocation_id,
        pad_id=tokenizer.pad_token_id, device="cuda", checkpoint_every=args.checkpoint_every,
        max_bytes=args.snapshot_max_bytes, completed=completed, counters=counters, history=history,
        parent_checkpoint=parent_checkpoint, autocast_factory=lambda: torch.autocast("cuda", dtype=torch.bfloat16),
        guard=guard, attempted_work=_ATTEMPTED_WORK, before_training=baseline,work_ledger=work_ledger,
        snapshot_artifact_budget=snapshot_ledger,snapshot_io_budget=io_hook)
    before = training["baseline"]
    def final_evaluation():
        journal.stage("final_evaluation", durable_completed_update=training["completed"], checkpoint=training["checkpoint"])
        validation=score_dpo_panel(model,reference,encoded_valid,pad_id=tokenizer.pad_token_id,
            device='cuda',beta=args.beta,autocast_factory=lambda:torch.autocast('cuda',dtype=torch.bfloat16),
            guard=guard,work_ledger=work_ledger)
        after = independent_evaluate('after')
        result = {"validation": validation, "after": after}
        write_exclusive_json(output/"evaluation-after.json", result)
        return result
    def export():
        journal.stage("exporting_checkpoint", durable_completed_update=training["completed"])
        model.save_pretrained(output / "policy", safe_serialization=True)
        tokenizer.save_pretrained(output / "policy")
        write_exclusive_json(output / "policy" / "course-genealogy.json", {
         "kind": "full-HF-DPO-policy", "parent_checkpoint": str(checkpoint.resolve()),
         "parent_checkpoint_sha256": identity["checkpoint_files"],
         "tokenizer": args.tokenizer, "tokenizer_revision": args.tokenizer_revision,
         "template_sha256": interface["template_sha256"],
         "checkpoint_interface": interface, "legacy_adoption": adoption,
         "run_identity_file": journal.path.name,
         "training_input_sha256": identity["input_sha256"], "completed_recovery": training["checkpoint"]})
    final = finalize_after_commit(training, evaluate=final_evaluation, export=export,
                                 guard=guard, cuda_devices=range(contract["cuda_rng_count"]))
    journal.stage("rehashing-actual-source-input-lock-and-parent-files")
    verify_identity_files(identity, root, guard)
    result = {"before": before, "after": final["after"], "validation": final["validation"],
              "minimum_sampled_mem_available_bytes": minimum_available, "wall_seconds": time.monotonic()-start,
              "reference_has_gradients": any(p.grad is not None for p in reference.parameters()),
              "completed_recovery": training["checkpoint"], "restored_update": completed,
              "cumulative_committed_training_work": training["counters"],
              "attempted_training_work_this_invocation": _ATTEMPTED_WORK,
              "split_evidence": split_evidence,
              "before_scope": "restored completed boundary" if args.resume else "original parent",
              'cumulative_work_ledger':work_ledger.snapshot(),'work_journal':str(work_path.resolve()),
              'snapshot_artifact_ledger':snapshot_ledger.receipt(),
              'snapshot_artifact_root':str(snapshot_ledger.root),
              'snapshot_artifact_usage':snapshot_ledger.usage(),
              'snapshot_io_ledger':io_ledger.snapshot(),'snapshot_io_journal':str(Path(args.snapshot_io_ledger).resolve()),
              'snapshot_io_scope':'shared operations only; metadata/journal/artifact inventory and caller capture excluded',
              "generation_cost_scope": "permanent full-panel reservations; entered/successful logical uncached forward geometry; not FLOPs/recompute/internal failed tokens"}
    write_exclusive_json(output/"result.json", result)
    write_json(output/"status.json", {"status": "completed", "updates": args.updates,
                                     "wall_seconds": time.monotonic()-start})
    journal.complete(result_file="result.json",minimum_sampled_mem_available_bytes=minimum_available)
    print(json.dumps({"output": str(output), "updates": args.updates, "status": "completed"}))


def main(argv=None):
    global _OUTPUT, _JOURNAL, _ATTEMPTED_WORK,_WORK_LEDGER,_ARTIFACT_LEDGER,_IO_LEDGER
    _OUTPUT = _JOURNAL = None
    _ATTEMPTED_WORK = None
    _WORK_LEDGER = None
    _ARTIFACT_LEDGER = None
    _IO_LEDGER = None
    try:
        return _main(argv)
    except BaseException as error:
        if _JOURNAL is not None:
            _JOURNAL.fail(error)
        raise
    finally:
        if _WORK_LEDGER is not None:_WORK_LEDGER.close()
        if _ARTIFACT_LEDGER is not None:_ARTIFACT_LEDGER.close()
        if _IO_LEDGER is not None:_IO_LEDGER.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if _OUTPUT is not None:
            (_OUTPUT / "status.json").write_text(json.dumps({"status": "failed",
                "error_type": type(error).__name__, "error": str(error), "partial_outputs_retained": True}))
        raise
