"""Pure checkpoint-IO allowances, separate from model and semantic work.

Supplied envelopes are not authenticated observations. This calculator neither
opens files nor reserves a ledger, launches a model, or grants run authority.
Its units match a declared shared-core operation envelope, not physical quotas.
"""
from copy import deepcopy
import hashlib
import json

SCHEMA = "dongxi-snapshot-io-requirements-v1"
SCOPE = "authored logical checkpoint-IO schedule; not a production mapping"
MAX_INT = 2**63 - 1
ENVELOPE_KEYS = ("max_payload_bytes", "max_tree_nodes", "max_tensor_elements",
                 "max_tensor_bytes", "max_primitive_bytes")
KEYS = ("snapshot_inspect_operations", "snapshot_load_operations",
        "snapshot_save_operations", "snapshot_hash_bytes", "snapshot_tree_nodes",
        "snapshot_tensor_elements", "snapshot_primitive_bytes", "snapshot_clone_bytes",
        "snapshot_serialization_bytes")
FIELDS = ("schema", "scope", "envelope", "schedule", "operations", "fresh_limits",
          "resumed_at_boundaries", "resumed_componentwise_upper", "required_limits",
          "production_ready", "launch_authorized", "requirements_sha256")
SCHEDULE_KEYS = ("updates", "checkpoint_every", "diagnostic_loads_per_attempt",
                 "complete_attempt_capacity", "commit_cursors")


def _integer(value, name, minimum=0, maximum=MAX_INT):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an exact bounded integer")


def _exact(value, fields, name):
    # Bound length before traversing untrusted mapping keys.
    if type(value) is not dict or len(value) != len(fields) or set(value) != set(fields):
        raise ValueError(f"Exact {name} fields required")


def validate_envelope(envelope):
    """Validate a supplied whole-operation envelope without scanning a tensor."""
    _exact(envelope, ENVELOPE_KEYS, "snapshot envelope")
    for key in ENVELOPE_KEYS:
        _integer(envelope[key], key, 1 if key in ENVELOPE_KEYS[:2] else 0)
    if envelope["max_tree_nodes"] > 1_000_000:
        raise ValueError("Snapshot node envelope exceeds the shared consumer bound")
    if any(envelope[key] > envelope["max_payload_bytes"]
           for key in ("max_tensor_bytes", "max_primitive_bytes")):
        raise ValueError("Tensor/primitive byte envelope exceeds payload envelope")
    return deepcopy(envelope)


def zero():
    return {key: 0 for key in KEYS}


def _vector(value):
    _exact(value, KEYS, "snapshot work vector")
    for key, amount in value.items():
        _integer(amount, key)
    return deepcopy(value)


def operation_cost(envelope, operation, *, payload_bytes=None):
    """One admitted operation; read size comes from an independent expectation.

    The schedule uses maximum payload size, while an actual read can reserve its
    exact size. Save reserves a maximum before cloning or serialization begins.
    """
    envelope = validate_envelope(envelope)
    if type(operation) is not str or operation not in ("inspect", "load", "save"):
        raise ValueError("Known snapshot operation required")
    if operation == "save":
        if payload_bytes is not None:
            raise ValueError("Save must reserve the maximum before serialization")
        byte_count = envelope["max_payload_bytes"]
    else:
        _integer(payload_bytes, "independently expected payload bytes", 1,
                 envelope["max_payload_bytes"])
        byte_count = payload_bytes
    result = zero()
    result[f"snapshot_{operation}_operations"] = 1
    result.update(snapshot_hash_bytes=byte_count,
                  snapshot_tree_nodes=envelope["max_tree_nodes"],
                  snapshot_primitive_bytes=envelope["max_primitive_bytes"])
    if operation in ("load", "save"):
        result["snapshot_tensor_elements"] = envelope["max_tensor_elements"]
    if operation == "save":
        result.update(snapshot_clone_bytes=envelope["max_tensor_bytes"],
                      snapshot_serialization_bytes=envelope["max_payload_bytes"])
    return _vector(result)


def _sum(*vectors):
    result = zero()
    for vector in vectors:
        _vector(vector)
        for key in KEYS:
            result[key] += vector[key]
            _integer(result[key], key)
    return result


def _times(vector, factor):
    _integer(factor, "operation multiplier")
    return _vector({key: amount*factor for key, amount in _vector(vector).items()})


def _hash(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_requirements(envelope, *, updates, checkpoint_every,
                       diagnostic_loads_per_attempt, complete_attempt_capacity):
    """Declare a separate bounded IO schedule; this grants no execution authority."""
    envelope = validate_envelope(envelope)
    for key, amount, maximum in (
            ("updates", updates, 1000), ("checkpoint cadence", checkpoint_every, 1000),
            ("diagnostic loads", diagnostic_loads_per_attempt, 16),
            ("complete attempt capacity", complete_attempt_capacity, 4)):
        _integer(amount, key, 1, maximum)
    cursors = sorted({0, updates, *range(checkpoint_every, updates+1, checkpoint_every)})
    operations = {name: operation_cost(envelope, name,
                        **({"payload_bytes": envelope["max_payload_bytes"]} if name != "save" else {}))
                  for name in ("inspect", "load", "save")}
    final = _times(operations["load"], diagnostic_loads_per_attempt)
    fresh = _sum(_times(operations["save"], len(cursors)), final)
    resumed = []
    for cursor in cursors:
        # One inspected+loaded parent; restore's semantic recheck does not read.
        save_count = 1 + sum(number > cursor for number in cursors)
        costs = _sum(operations["inspect"], operations["load"],
                     _times(operations["save"], save_count), final)
        resumed.append({"durable_completed": cursor, "limits": costs})
    upper = {key: max(row["limits"][key] for row in resumed) for key in KEYS}
    result = dict(schema=SCHEMA, scope=SCOPE, envelope=envelope,
        schedule=dict(updates=updates, checkpoint_every=checkpoint_every,
                      diagnostic_loads_per_attempt=diagnostic_loads_per_attempt,
                      complete_attempt_capacity=complete_attempt_capacity,
                      commit_cursors=cursors), operations=operations,
        fresh_limits=fresh, resumed_at_boundaries=resumed,
        resumed_componentwise_upper=upper,
        required_limits=_sum(fresh, _times(upper, complete_attempt_capacity-1)),
        production_ready=False, launch_authorized=False)
    result["requirements_sha256"] = _hash(result)
    return result


def validate_requirements(requirements):
    """Recompute every call and dimension; rehashing an omission does not repair it."""
    _exact(requirements, FIELDS, "standalone snapshot requirements")
    if (requirements["schema"] != SCHEMA or requirements["scope"] != SCOPE
            or requirements["production_ready"] is not False
            or requirements["launch_authorized"] is not False):
        raise ValueError("Only separate logical snapshot requirements are supported")
    schedule = requirements["schedule"]
    _exact(schedule, SCHEDULE_KEYS, "snapshot schedule")
    expected = build_requirements(requirements["envelope"], **{key: schedule[key]
        for key in SCHEDULE_KEYS if key != "commit_cursors"})
    # Validate sizes/types before equality/hash; bool must never alias a count.
    if (type(schedule["commit_cursors"]) is not list
            or len(schedule["commit_cursors"]) != len(expected["schedule"]["commit_cursors"])
            or any(type(number) is not int for number in schedule["commit_cursors"])):
        raise ValueError("Bounded exact commit cursors required")
    _exact(requirements["operations"], ("inspect", "load", "save"), "snapshot operations")
    for vector in requirements["operations"].values():
        _vector(vector)
    for key in ("fresh_limits", "resumed_componentwise_upper", "required_limits"):
        _vector(requirements[key])
    rows = requirements["resumed_at_boundaries"]
    if type(rows) is not list or len(rows) != len(expected["resumed_at_boundaries"]):
        raise ValueError("Exact bounded resumed boundary list required")
    for row in rows:
        _exact(row, ("durable_completed", "limits"), "resumed snapshot boundary")
        _integer(row["durable_completed"], "resumed cursor", 0, 1000)
        _vector(row["limits"])
    if requirements != expected:
        raise ValueError("Snapshot schedule, requirements or digest changed")
    return deepcopy(expected)


def assert_limits(requirements, limits):
    """Check explicit separate allowance; neither reserve a ticket nor refill caps."""
    requirements = validate_requirements(requirements)
    limits = _vector(limits)
    short = [key for key in KEYS if limits[key] < requirements["required_limits"][key]]
    if short:
        raise ValueError("Insufficient separate snapshot allowance: " + ",".join(short))
    return limits
