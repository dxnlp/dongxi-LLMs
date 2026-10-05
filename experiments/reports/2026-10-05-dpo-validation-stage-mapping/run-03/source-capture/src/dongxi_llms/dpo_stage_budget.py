"""Pure fixed-DPO work mapping; requirements and fixture receipts never launch.

Supplied encodings are byte-bound observations, not authenticated tokenization.
A real runner must independently re-encode and compare before using this plan.
Only authored fixture scope exists; no model/tokenizer imports or execution API.
"""
from copy import deepcopy
import hashlib
import json

from . import production_stage as stage
from .run_identity import canonical_hash, validate_interface

STAGES = ("assistant-dpo-smoke", "assistant-dpo-recovery", "assistant-dpo-pilot")
TRAIN_KEYS = ("train_updates", "sampled_examples", "sampler_draws", "valid_targets",
        "logical_sequence_tokens", "policy_forward_calls", "policy_forward_positions",
        "reference_forward_calls", "reference_forward_positions", "evaluation_calls",
        "evaluation_positions", "generation_calls", "generation_position_upper_bound",
        "generation_tokens")
VALIDATION_KEYS = ("recovery_validation_operations", "recovery_history_rows",
                   "recovery_tensor_elements", "recovery_rng_states", "recovery_sampler_draws")
KEYS = TRAIN_KEYS + VALIDATION_KEYS
MAX_POSITIONS = 200_000
MAX_ATTEMPTS = 4
GENERATION_CAP = 64
INPUT_ROLES = {"train": "data", "validation": "validation", "evaluation": "evaluation",
               "tokenizer": "tokenizer", "template": "template"}
LAYOUT_FIELDS = ("schema", "policy_shapes", "optimizer_shapes", "optimizer_parameter_names",
                 "torch_rng_bytes", "cuda_rng_count", "cuda_rng_bytes", "policy_model_sha256",
                 "reference_model_sha256")
DTYPES = ("torch.float32", "torch.float64", "torch.float16", "torch.bfloat16",
          "torch.int64", "torch.int32", "torch.int16", "torch.int8", "torch.uint8", "torch.bool")


def observe_layout(metadata):
    """Seal plain supplied observations; this does not authenticate a model."""
    stage._exact(metadata, LAYOUT_FIELDS, "validation layout observation")
    _layout(dict(metadata, layout_sha256="0"*64), verify_hash=False)
    result = deepcopy(metadata)
    result["layout_sha256"] = canonical_hash(metadata)
    _layout(result)
    return result


def _layout(value, *, verify_hash=True):
    stage._exact(value, (*LAYOUT_FIELDS, "layout_sha256"), "validation layout")
    if value["schema"] != "dongxi-dpo-validation-layout-v1":
        raise ValueError("Explicit known observed validation layout required")
    for key in ("policy_model_sha256", "reference_model_sha256"):
        stage._sha(value[key], key)
    policy, optimizer, names = value["policy_shapes"], value["optimizer_shapes"], value["optimizer_parameter_names"]
    if (type(policy) is not dict or not 1 <= len(policy) <= 4096 or type(optimizer) is not list
            or not 1 <= len(optimizer) <= 4096 or type(names) is not list or len(names) != len(optimizer)):
        raise ValueError("Bounded actual policy and optimizer layout required")
    if any(type(name) is not str or not 1 <= len(name) <= 256 for name in policy):
        raise ValueError("Bounded actual state names required")
    if any(type(name) is not str or name not in policy for name in names) or len(set(names)) != len(names):
        raise ValueError("Unique ordered actual optimizer parameter names required")
    def elements(item):
        stage._exact(item, ("shape", "dtype"), "actual tensor layout")
        if type(item["shape"]) is not list or len(item["shape"]) > 8 or type(item["dtype"]) is not str or item["dtype"] not in DTYPES:
            raise ValueError("Known bounded actual shape and dtype required")
        total = 1
        for size in item["shape"]:
            stage._integer(size, "actual dimension")
            if size > 2**31-1: raise ValueError("Actual dimension too large")
            total *= size; stage._integer(total, "actual tensor elements")
        return total
    p = sum(elements(item) for item in policy.values()); o = sum(elements(item) for item in optimizer)
    stage._integer(p, "policy elements", 1); stage._integer(o, "optimizer elements", 1)
    if any(item != policy[name] for name, item in zip(names, optimizer)):
        raise ValueError("Observed optimizer shapes do not bind ordered policy parameters")
    rng, count, cuda = value["torch_rng_bytes"], value["cuda_rng_count"], value["cuda_rng_bytes"]
    stage._integer(rng, "CPU RNG bytes", 1); stage._integer(count, "CUDA RNG count")
    if rng > 65536 or count > 16 or type(cuda) is not list or len(cuda) != count:
        raise ValueError("Bounded exact actual RNG layout required")
    for size in cuda:
        stage._integer(size, "CUDA RNG bytes", 1)
        if size > 1024*1024: raise ValueError("CUDA RNG layout exceeds declared bound")
    if verify_hash and canonical_hash({key: item for key, item in value.items() if key != "layout_sha256"}) != value["layout_sha256"]:
        raise ValueError("Validation layout hash differs")
    return p, o, len(optimizer), rng, sum(cuda), count


def validation_upper(layout, completed, accumulation):
    """Independent pure arithmetic for one runner-owned semantic panel."""
    stage._integer(completed, "completed validation cursor")
    stage._integer(accumulation, "validation accumulation", 1)
    if completed > 1000 or accumulation > 16: raise ValueError("Bounded validation recipe required")
    p, o, n, rng, cuda, count = _layout(layout)
    result = zero()
    result.update(recovery_validation_operations=1, recovery_history_rows=completed,
        recovery_tensor_elements=3*p+(3*o+n if completed else 0)+4*rng+cuda,
        recovery_rng_states=2+count, recovery_sampler_draws=completed*accumulation)
    return _limits(result)


def validation_schedule(layout, *, updates, accumulation, checkpoint_every, diagnostic_loads, attempts):
    for name, amount in (("updates", updates), ("checkpoint cadence", checkpoint_every),
                         ("diagnostic loads", diagnostic_loads), ("attempt capacity", attempts)):
        stage._integer(amount, name, 1)
    if updates > 1000 or checkpoint_every > 1000 or diagnostic_loads > 16 or attempts > MAX_ATTEMPTS:
        raise ValueError("Explicit bounded actual validation schedule required")
    cursors = sorted({0, updates, *range(checkpoint_every, updates+1, checkpoint_every)})
    panels = {cursor: validation_upper(layout, cursor, accumulation) for cursor in cursors}
    final = _times(panels[updates], diagnostic_loads)
    fresh = _sum(*(panels[cursor] for cursor in cursors), final)
    resumed = []
    for cursor in cursors:
        costs = _sum(_times(panels[cursor], 3), *(panels[k] for k in cursors if k > cursor), final)
        resumed.append(dict(durable_completed=cursor, limits=costs))
    envelope = {key: max(row["limits"][key] for row in resumed) for key in KEYS}
    return dict(checkpoint_every=checkpoint_every, commit_cursors=cursors,
        diagnostic_loads_per_attempt=diagnostic_loads, fresh_limits=fresh,
        resumed_at_boundaries=resumed, resumed_componentwise_upper=envelope,
        required_limits=_sum(fresh, _times(envelope, attempts-1)),
        boundary="callback + restore + new invocation commit; componentwise envelope is not one measured trajectory")


def zero():
    return {key: 0 for key in KEYS}


def _limits(value):
    stage._exact(value, KEYS, "actual DPO work caps")
    for key, amount in value.items(): stage._integer(amount, key)
    return deepcopy(value)


def _sum(*vectors):
    result = zero()
    for vector in vectors:
        for key, amount in vector.items():
            result[key] += amount
            stage._integer(result[key], key)
    return result


def _times(vector, factor):
    return _limits({key: amount*factor for key, amount in vector.items()})


def _sealed(value):
    result = deepcopy(value)
    result["requirements_sha256"] = canonical_hash(value)
    return result


def _validate_requirements(value):
    fields = ("schema", "scope", "stage_id", "preparation_sha256", "science", "observations",
              "schedule", "operations", "required_limits", "coarse_resource_ceilings",
              "boundaries", "production_ready", "launch_authorized", "requirements_sha256")
    stage._exact(value, fields, "DPO requirements")
    if (value["schema"] != "dongxi-dpo-work-requirements-v2" or value["scope"] != stage.FIXTURE_SCOPE
        or value["stage_id"] not in STAGES or value["production_ready"] is not False
        or value["launch_authorized"] is not False):
        raise ValueError("Only fixed authored requirements are supported")
    body = {key: item for key, item in value.items() if key != "requirements_sha256"}
    if canonical_hash(body) != value["requirements_sha256"]:
        raise ValueError("Requirements changed after geometry binding")
    _limits(value["required_limits"])
    _layout(value["observations"]["validation_layout"])
    schedule = value["schedule"]; recovery = schedule["recovery_validation"]
    expected = validation_schedule(value["observations"]["validation_layout"],
        updates=value["science"]["proposed_budget"]["maximum_updates"],
        accumulation=value["science"]["proposed_budget"]["accumulation"],
        checkpoint_every=recovery["checkpoint_every"], diagnostic_loads=recovery["diagnostic_loads_per_attempt"],
        attempts=schedule["complete_attempt_capacity"])
    if recovery != expected: raise ValueError("Recovery validation schedule changed")
    operations = value["operations"]
    for key in ("one_update", "one_validation_panel", "one_generation_panel"):
        _limits(operations[key])
        if any(operations[key][name] for name in VALIDATION_KEYS):
            raise ValueError("Model work cannot hide recovery verification units")
    model_work = _times(_sum(_times(operations["one_update"], value["science"]["proposed_budget"]["maximum_updates"]),
        operations["one_validation_panel"], _times(operations["one_generation_panel"], 2)), schedule["complete_attempt_capacity"])
    if value["required_limits"] != _sum(model_work, recovery["required_limits"]):
        raise ValueError("Complete model and validation requirements differ")
    return value


def _messages(row):
    prompt = row.get("prompt")
    if type(prompt) is not list or not 1 <= len(prompt) <= 32:
        raise ValueError("Bounded actual message prompts required")
    for message in prompt:
        stage._exact(message, ("role", "content"), "actual message")
        if (message["role"] not in ("system", "user", "assistant") or type(message["content"]) is not str
            or not 0 < len(message["content"]) <= 8192):
            raise ValueError("Plain bounded message content required")
    if prompt[-1]["role"] == "assistant": raise ValueError("Prompt must precede assistant response")
    return canonical_hash(prompt)


def _rows(raw, *, kind, maximum):
    result = [stage._strict_json(line) for line in raw.splitlines() if line.strip()]
    if not 1 <= len(result) <= maximum: raise ValueError("Actual panel size is outside runner bounds")
    ids, prompts, groups = [], [], []
    required = {"id", "prompt", "expected"} if kind == "evaluation" else {"id", "prompt", "chosen", "rejected"}
    for row in result:
        if type(row) is not dict or set(row) not in (required, required | {"group"}):
            raise ValueError("Actual row schema differs from the fixed DPO interface")
        if type(row["id"]) is not str or not 0 < len(row["id"]) <= 128: raise ValueError("Bounded row ID required")
        ids.append(row["id"]); prompts.append(_messages(row))
        for key in ("expected",) if kind == "evaluation" else ("chosen", "rejected"):
            if type(row[key]) is not str or not row[key].strip() or len(row[key]) > 32768:
                raise ValueError("Plain nonempty bounded response required")
        if "group" in row:
            if type(row["group"]) is not str or not 0 < len(row["group"]) <= 128: raise ValueError("Bounded source group required")
            groups.append(row["group"])
    if len(set(ids)) != len(ids): raise ValueError("Repeated actual row IDs")
    return result, dict(ids=ids, prompts=prompts, groups=groups,
                       source_groups="provided" if len(groups) == len(ids) else "missing; transfer evidence pending")


def _ids(value, interface, maximum):
    if type(value) not in (list, tuple) or not 1 <= len(value) <= maximum:
        raise ValueError("Nonempty plain token-ID sequence required")
    if any(type(number) is not int or not 0 <= number <= interface["tokenizer"]["max_token_id"] for number in value):
        raise ValueError("IDs must be exact in-interface integers")
    return list(value)


def _pairs(values, expected_count, interface, length, position_count):
    if type(values) not in (list, tuple) or len(values) != expected_count:
        raise ValueError("Encoded pair order/count differs from actual input bytes")
    normalized, prefixes, works = [], [], []
    for pair in values:
        if type(pair) not in (list, tuple) or len(pair) != 2: raise ValueError("Chosen/rejected pair required")
        branches, pair_prefix = [], []
        for branch in pair:
            if type(branch) not in (list, tuple) or len(branch) != 2: raise ValueError("IDs and completion mask required")
            ids = _ids(branch[0], interface, length); mask = branch[1]
            position_count[0] += len(ids)
            if position_count[0] > MAX_POSITIONS: raise ValueError("Supplied geometry exceeds bounded mapping envelope")
            if (type(mask) not in (list, tuple) or len(mask) != len(ids)
                or any(type(flag) is not bool for flag in mask) or not 2 <= len(ids) <= length):
                raise ValueError("One untruncated sequence and exact bool mask required")
            mask = list(mask)
            if not any(mask) or mask[0]: raise ValueError("Nonempty completion after a prompt required")
            first = mask.index(True)
            if mask != [False]*first+[True]*(len(mask)-first): raise ValueError("Completion mask must be one suffix")
            pair_prefix.append(ids[:first]); branches.append([ids, mask])
        if pair_prefix[0] != pair_prefix[1]: raise ValueError("Chosen/rejected actual prefixes differ")
        normalized.append(branches); prefixes.append(pair_prefix[0])
        works.append(dict(targets=sum(sum(branch[1][1:]) for branch in branches),
                          sequence_positions=sum(len(branch[0]) for branch in branches),
                          scored_positions=sum(len(branch[0])-1 for branch in branches)))
    return normalized, prefixes, works


def build_requirements(preparation, *, artifact_root, encoded_train, encoded_valid,
                       evaluation_prefixes, interface_sha256, encoder_source_sha256,
                       attempt_allowance, validation_layout, checkpoint_every,
                       diagnostic_loads_per_attempt, generation_cap=GENERATION_CAP):
    """Derive explicit whole-job caps from currently pinned supplied observations.

    The caller's encoding observations are NOT authenticated by this function.
    A live tokenizer/runner must independently reproduce them before execution.
    """
    if type(preparation) is not dict or preparation.get("stage_id") not in STAGES:
        raise ValueError("Only the three fixed DPO training stages are supported")
    binding = preparation.get("bindings")
    if type(binding) is not dict or binding.get("scope") != stage.FIXTURE_SCOPE:
        raise RuntimeError("Actual production encoding/preparation is not supported here")
    stage._exact(binding, ("scope","files","parent_slots","interface","limits","checkpoint_genealogy","boundary"), "preparation bindings")
    # Reconstruct through the original bounded reader, not an unchecked hash field.
    supplied = dict(scope=binding["scope"], files=binding["files"], parent_slots=binding["parent_slots"],
                    interface_role="interface", limits=binding["limits"])
    current = stage.prepare_stage(preparation["stage_id"], bindings=supplied, artifact_root=artifact_root)
    if preparation != current: raise ValueError("Preparation/source/input identities are stale or changed")
    stage._integer(attempt_allowance, "complete-attempt capacity", 1)
    if attempt_allowance > MAX_ATTEMPTS: raise ValueError("Bounded complete-attempt capacity required")
    if type(generation_cap) is not int or generation_cap != GENERATION_CAP:
        raise ValueError("Fixed current DPO generation cap is64; a different protocol requires revision")
    interface = validate_interface(binding["interface"])
    stage._sha(interface_sha256,"observed interface");stage._sha(encoder_source_sha256,"observed encoder source")
    if interface_sha256 != interface["interface_sha256"]: raise ValueError("Observed interface differs from preparation")
    source = current["source_files"]["scripts/run_chapter11_spark_dpo.py"]["sha256"]
    if encoder_source_sha256 != source: raise ValueError("Actual encoder source observation differs")
    input_identities, raw_inputs = {}, {}
    for name, role in INPUT_ROLES.items():
        if role not in binding["files"]: raise ValueError("Actual data and tokenizer artifacts must be byte-bound")
        expected = binding["files"][role]
        raw, identity = stage._read(artifact_root, expected["path"], cap=stage.MAX_ARTIFACT_BYTES,
            expected_bytes=expected["bytes"], expected_sha256=expected["sha256"])
        input_identities[name] = identity; raw_inputs[name] = raw
    if input_identities["template"]["sha256"] != interface["template_sha256"]:
        raise ValueError("Actual template bytes differ from the observed interface")
    raw_rows, orders = {}, {}
    for name, maximum in (("train",4096),("validation",128),("evaluation",64)):
        raw_rows[name], orders[name] = _rows(raw_inputs[name], kind=name, maximum=maximum)
    for left, right in (("train","validation"),("train","evaluation"),("validation","evaluation")):
        for key in ("ids", "prompts", "groups"):
            if set(orders[left][key]) & set(orders[right][key]): raise ValueError("Actual source splits overlap")
    science = current["science"]; budget = science["proposed_budget"]
    updates, accumulation, length = budget["maximum_updates"], budget["accumulation"], budget["sequence_positions_per_example_max"]
    recovery = validation_schedule(validation_layout, updates=updates, accumulation=accumulation,
        checkpoint_every=checkpoint_every, diagnostic_loads=diagnostic_loads_per_attempt, attempts=attempt_allowance)
    position_count=[0]
    train, train_prefixes, train_work = _pairs(encoded_train, len(raw_rows["train"]), interface, length, position_count)
    valid, valid_prefixes, valid_work = _pairs(encoded_valid, len(raw_rows["validation"]), interface, length, position_count)
    if type(evaluation_prefixes) not in (list, tuple) or len(evaluation_prefixes) != len(raw_rows["evaluation"]):
        raise ValueError("Actual generation panel order/count differs")
    prefixes = [_ids(prefix, interface, length) for prefix in evaluation_prefixes]
    if any(len(prefix)+generation_cap > length or len(prefix)>binding["limits"]["max_prompt_positions"] for prefix in prefixes):
        raise ValueError("Actual generation prefix/cap exceeds declared context/prompt bounds")
    for left, right in ((train_prefixes,valid_prefixes),(train_prefixes,prefixes),(valid_prefixes,prefixes)):
        if {tuple(p) for p in left} & {tuple(p) for p in right}: raise ValueError("Actual encoded prompts overlap")
    positions = position_count[0]+sum(map(len,prefixes))
    if positions > MAX_POSITIONS: raise ValueError("Supplied geometry exceeds bounded mapping envelope")
    update = zero(); update.update(train_updates=1, sampled_examples=accumulation, sampler_draws=accumulation,
        valid_targets=accumulation*max(row["targets"] for row in train_work),
        logical_sequence_tokens=accumulation*max(row["sequence_positions"] for row in train_work),
        policy_forward_calls=2*accumulation, reference_forward_calls=2*accumulation,
        policy_forward_positions=accumulation*max(row["scored_positions"] for row in train_work),
        reference_forward_positions=accumulation*max(row["scored_positions"] for row in train_work))
    validation = zero(); validation.update(valid_targets=sum(row["targets"] for row in valid_work),
        logical_sequence_tokens=sum(row["sequence_positions"] for row in valid_work),
        policy_forward_calls=2*len(valid), reference_forward_calls=2*len(valid), evaluation_calls=4*len(valid),
        policy_forward_positions=sum(row["scored_positions"] for row in valid_work),
        reference_forward_positions=sum(row["scored_positions"] for row in valid_work),
        evaluation_positions=2*sum(row["scored_positions"] for row in valid_work))
    growing = sum(generation_cap*len(prefix)+generation_cap*(generation_cap-1)//2 for prefix in prefixes)
    generation = zero(); generation.update(policy_forward_calls=len(prefixes)*generation_cap, policy_forward_positions=growing,
        evaluation_calls=len(prefixes)*generation_cap, evaluation_positions=growing, generation_calls=len(prefixes),
        generation_position_upper_bound=growing, generation_tokens=len(prefixes)*generation_cap,
        logical_sequence_tokens=sum(len(prefix)+generation_cap for prefix in prefixes))
    required = _sum(_times(_sum(_times(update,updates),validation,_times(generation,2)),attempt_allowance), recovery["required_limits"])
    training_targets = attempt_allowance*updates*update["valid_targets"]
    ceilings = deepcopy(binding["limits"])
    checks = {"max_valid_training_targets":training_targets, "max_policy_forward_positions":required["policy_forward_positions"],
        "max_reference_forward_positions":required["reference_forward_positions"],
        "max_evaluation_forward_positions":required["evaluation_positions"],
        "max_generation_forward_positions":required["generation_position_upper_bound"], "max_response_slots":required["generation_tokens"]}
    if any(amount>ceilings[key] for key,amount in checks.items()):
        raise ValueError("Actual whole-job requirements exceed independently prepared coarse ceilings")
    return _sealed(dict(schema="dongxi-dpo-work-requirements-v2", scope=stage.FIXTURE_SCOPE,
        stage_id=current["stage_id"], preparation_sha256=current["preparation_sha256"], science=deepcopy(science),
        observations=dict(input_files=input_identities, interface_sha256=interface_sha256, encoder_source_sha256=source,
            row_order=orders, encoded_train_sha256=canonical_hash(train), encoded_valid_sha256=canonical_hash(valid),
            evaluation_prefix_sha256=canonical_hash(prefixes), supplied_positions=positions,
            validation_layout=deepcopy(validation_layout),
            supplier_provenance="not authenticated; live source re-encoding remains required"),
        schedule=dict(complete_attempt_capacity=attempt_allowance, training_update_reservations=updates*attempt_allowance,
            baseline_generation_panels=attempt_allowance, final_generation_panels=attempt_allowance,
            validation_panels=attempt_allowance, generation_cap=generation_cap,
            recovery_validation=recovery,
            retry_permission=False, invocation_counter_enforced=False),
        operations=dict(one_update=update,one_validation_panel=validation,one_generation_panel=generation,
            training_valid_target_upper=training_targets,validation_target_upper=attempt_allowance*validation["valid_targets"],
            historical_single_branch_geometry=budget["maximum_padded_training_positions"],
            actual_training_policy_upper=attempt_allowance*updates*update["policy_forward_positions"],
            actual_training_reference_upper=attempt_allowance*updates*update["reference_forward_positions"]),
        required_limits=required, coarse_resource_ceilings=ceilings,
        boundaries=["logical positions/calls, not FLOPs/backward dispatch", "permanent whole-operation reservation; no refund",
            "requirements are not approval; actual production scope unsupported",
            "runner semantic verification units only; generic loader/serialization and physical containment excluded"], production_ready=False,launch_authorized=False))


def approve_fixture(requirements, *, limits, receipt_root, receipt_ref):
    """Require a NEW independently retained full-vector fixture receipt."""
    _validate_requirements(requirements); limits = _limits(limits)
    if any(limits[key]<amount for key,amount in requirements["required_limits"].items()):
        raise ValueError("An explicit actual-work cap is insufficient; coarse geometry is not approval")
    if limits != requirements["required_limits"]:
        raise ValueError("Extra capacity requires a separately declared attempt schedule, not cap overrides")
    record, identity = stage._receipt(receipt_root,receipt_ref)
    expected = dict(schema="dongxi-dpo-work-approval-v2",scope=stage.FIXTURE_SCOPE,stage_id=requirements["stage_id"],
        preparation_sha256=requirements["preparation_sha256"],requirements_sha256=requirements["requirements_sha256"],
        limits_sha256=canonical_hash(limits),decision="fixture-work-cap-validation-only")
    if record != expected: raise ValueError("New full-vector receipt must bind exact current requirements; production refuses")
    payload = json.dumps(limits,sort_keys=True,separators=(",",":"),allow_nan=False).encode()+b"\n"
    result = dict(schema="dongxi-dpo-work-mapping-v2",scope=stage.FIXTURE_SCOPE,requirements=deepcopy(requirements),
        limits=limits,work_limits_sha256=hashlib.sha256(payload).hexdigest(),work_limits_bytes=len(payload),
        approval_receipt=identity,production_ready=False,launch_authorized=False,jobs_started=0,launch_command=None)
    result["mapping_sha256"] = canonical_hash(result)
    return result


def work_limits_bytes(mapping):
    """Return the exact existing runner cap schema, without authority metadata."""
    if type(mapping) is not dict or mapping.get("schema")!="dongxi-dpo-work-mapping-v2" or mapping.get("scope")!=stage.FIXTURE_SCOPE:
        raise ValueError("Only a validated authored fixture mapping is supported")
    stage._exact(mapping,("schema","scope","requirements","limits","work_limits_sha256","work_limits_bytes","approval_receipt",
        "production_ready","launch_authorized","jobs_started","launch_command","mapping_sha256"),"work mapping")
    body={key:value for key,value in mapping.items() if key!="mapping_sha256"}
    if canonical_hash(body)!=mapping.get("mapping_sha256"):raise ValueError("Mapping bytes changed")
    _validate_requirements(mapping["requirements"]);limits=_limits(mapping["limits"])
    if limits!=mapping["requirements"]["required_limits"]:raise ValueError("Mapped caps no longer match declared requirements")
    if mapping["production_ready"] is not False or mapping["launch_authorized"] is not False or type(mapping["jobs_started"]) is not int or mapping["jobs_started"]!=0 or mapping["launch_command"] is not None:
        raise ValueError("Mapping cannot carry production authority")
    raw=json.dumps(limits,sort_keys=True,separators=(",",":"),allow_nan=False).encode()+b"\n"
    if hashlib.sha256(raw).hexdigest()!=mapping["work_limits_sha256"] or len(raw)!=mapping["work_limits_bytes"]:
        raise ValueError("Consumable work-limits bytes differ")
    return raw


def assert_reencoded(requirements, fresh_requirements):
    """Compare independently recomputed observations; do not authenticate a caller."""
    _validate_requirements(requirements);_validate_requirements(fresh_requirements)
    if requirements!=fresh_requirements:raise ValueError("Fresh actual geometry/source/input/interface/schedule differs")
    return True
