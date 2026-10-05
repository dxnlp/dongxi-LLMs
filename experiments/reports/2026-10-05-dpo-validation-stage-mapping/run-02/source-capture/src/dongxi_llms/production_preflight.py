"""Nonexecuting backend contract validator; only authored CPU fixtures exist.

No provider probes, processes, signals, platform writes or model launches occur.
Validating a claimed timeout/quota is not implementing that timeout/quota.
Actual platform readiness refuses until a separately authorized adapter exists.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import re

from .run_identity import canonical_hash
from .staged_campaign import stage_rows

GIB = 1024**3
MAX_RESPONSE_BYTES = 8192
MAX_MEMBERS = 128
FIXTURE_SCOPE = "authored-local-fixture"
GATES = ("containment", "artifact_quota", "bounded_observers", "external_watchdog", "gpu_clearance")
ADAPTERS = {
    "containment": "fixture-private-boundary-v1",
    "artifact_quota": "fixture-hard-quota-v1",
    "bounded_observers": "fixture-bounded-observer-v1",
    "external_watchdog": "fixture-independent-watchdog-v1",
    "gpu_clearance": "fixture-gpu-owner-inspection-v1",
}


def _exact(value, keys, name):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(f"{name} fields must match the declared schema")


def _integer(value, name, *, minimum=0, maximum=2**63-1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} requires a bounded exact integer")


def _hash(value, name):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} requires a lowercase SHA256/nonce")


def _finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class PreflightPolicy:
    stage_id: str
    preparation_sha256: str
    observation_nonce: str
    instance_id: str
    owner_uid: int
    hard_memory_bytes: int
    hard_pid_limit: int
    artifact_bytes: int
    artifact_entries: int
    deadline_seconds: float
    minimum_host_available_bytes: int = 25 * GIB
    maximum_observation_age_seconds: float = 2.
    maximum_observer_seconds: float = .5

    def validate(self):
        if self.stage_id not in {row["id"] for row in stage_rows()}:
            raise ValueError("Unknown campaign stage")
        _hash(self.preparation_sha256, "preparation identity")
        _hash(self.observation_nonce, "observation nonce")
        if type(self.instance_id) is not str or re.fullmatch(r"dongxi-stage-[0-9a-f]{32}", self.instance_id) is None:
            raise ValueError("Only an exact private opaque stage instance is allowed")
        _integer(self.owner_uid, "owner UID", minimum=1)
        for name in ("hard_memory_bytes", "hard_pid_limit", "artifact_bytes", "artifact_entries"):
            _integer(getattr(self, name), name, minimum=1)
        _integer(self.minimum_host_available_bytes, "host reserve", minimum=25*GIB)
        for name in ("deadline_seconds", "maximum_observation_age_seconds", "maximum_observer_seconds"):
            _finite(getattr(self, name), name)
            if getattr(self, name) == 0:
                raise ValueError(f"{name} must be positive")
        if self.maximum_observation_age_seconds > 2 or self.maximum_observer_seconds > .5:
            raise ValueError("Fixture freshness/observer bounds cannot be relaxed")
        return self


def _strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate observation key")
        result[key] = value
    return result


def parse_observation(raw):
    """Bound bytes before strict JSON parsing; only a shallow fixed schema follows."""
    if type(raw) is not bytes or not 0 < len(raw) <= MAX_RESPONSE_BYTES:
        raise ValueError("Observation must be bounded UTF-8 bytes")
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_strict_pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    except (UnicodeError, RecursionError, json.JSONDecodeError) as error:
        raise ValueError("Malformed bounded observation") from error


def _verify_observation(policy, value, now):
    _exact(value, ("schema", "scope", "stage_id", "preparation_sha256", "nonce", "sequence",
                   "observed_at", "observer_seconds", "host", "boundary", "quota", "observer",
                   "watchdog", "gpu"), "observation")
    if (value["schema"] != "dongxi-preflight-observation-v1" or value["scope"] != FIXTURE_SCOPE
        or value["stage_id"] != policy.stage_id or value["preparation_sha256"] != policy.preparation_sha256
        or value["nonce"] != policy.observation_nonce):
        raise ValueError("Observation is not bound to this fixture preparation and challenge")
    _integer(value["sequence"], "sequence", minimum=1)
    _finite(value["observed_at"], "observation time")
    _finite(value["observer_seconds"], "observer elapsed")
    if not 0 <= now-value["observed_at"] <= policy.maximum_observation_age_seconds:
        raise ValueError("Observation is future-dated or stale")
    if value["observer_seconds"] > policy.maximum_observer_seconds:
        raise ValueError("Observer exceeded its declared bounded response interval")
    host = value["host"]
    _exact(host, ("available_bytes", "complete", "measurement"), "host")
    _integer(host["available_bytes"], "available host bytes")
    if (host["complete"] is not True or host["measurement"] != "injected-sample-not-continuous"
        or host["available_bytes"] < policy.minimum_host_available_bytes):
        raise ValueError("Host reserve or inspection failed")
    boundary = value["boundary"]
    _exact(boundary, ("adapter", "instance_id", "owner_uid", "exclusive", "membership_complete",
                       "members", "hard_memory_bytes", "hard_pid_limit"), "boundary")
    if (boundary["adapter"] != ADAPTERS["containment"] or boundary["instance_id"] != policy.instance_id
        or type(boundary["owner_uid"]) is not int or boundary["owner_uid"] != policy.owner_uid
        or boundary["exclusive"] is not True or boundary["membership_complete"] is not True):
        raise ValueError("Private ownership or complete containment evidence missing")
    _integer(boundary["hard_memory_bytes"], "memory ceiling", minimum=1)
    _integer(boundary["hard_pid_limit"], "PID ceiling", minimum=1)
    if boundary["hard_memory_bytes"] != policy.hard_memory_bytes or boundary["hard_pid_limit"] != policy.hard_pid_limit:
        raise ValueError("Containment bounds do not match preparation")
    members = boundary["members"]
    if type(members) is not list or len(members) > MAX_MEMBERS:
        raise ValueError("Bounded member list required")
    for member in members:
        _integer(member, "owned fixture member", minimum=1)
    if len(set(members)) != len(members) or len(members) > policy.hard_pid_limit:
        raise ValueError("Duplicate or excess members")
    quota = value["quota"]
    _exact(quota, ("adapter", "instance_id", "owner_uid", "exclusive", "hard_bytes", "hard_entries"), "quota")
    if (quota["adapter"] != ADAPTERS["artifact_quota"] or quota["instance_id"] != policy.instance_id
        or type(quota["owner_uid"]) is not int or quota["owner_uid"] != policy.owner_uid
        or quota["exclusive"] is not True):
        raise ValueError("Missing exclusive hard aggregate quota contract")
    for key in ("hard_bytes", "hard_entries"):
        _integer(quota[key], key, minimum=1)
    if quota["hard_bytes"] != policy.artifact_bytes or quota["hard_entries"] != policy.artifact_entries:
        raise ValueError("Artifact quota does not match preparation")
    observer = value["observer"]
    _exact(observer, ("adapter", "timeout_enforced", "maximum_seconds", "maximum_bytes"), "observer")
    _finite(observer["maximum_seconds"], "observer timeout")
    _integer(observer["maximum_bytes"], "observer byte limit", minimum=1)
    if (observer["adapter"] != ADAPTERS["bounded_observers"] or observer["timeout_enforced"] is not True
        or not 0 < observer["maximum_seconds"] <= policy.maximum_observer_seconds
        or observer["maximum_bytes"] > MAX_RESPONSE_BYTES
        or value["observer_seconds"] > observer["maximum_seconds"]):
        raise ValueError("Bounded observer backend contract missing")
    watchdog = value["watchdog"]
    _exact(watchdog, ("adapter", "instance_id", "independent", "armed", "deadline_seconds", "cleanup_bounded"), "watchdog")
    _finite(watchdog["deadline_seconds"], "watchdog deadline")
    if (watchdog["adapter"] != ADAPTERS["external_watchdog"] or watchdog["instance_id"] != policy.instance_id
        or watchdog["independent"] is not True or watchdog["armed"] is not True
        or watchdog["cleanup_bounded"] is not True or watchdog["deadline_seconds"] != policy.deadline_seconds):
        raise ValueError("Independent armed bounded-cleanup watchdog missing")
    gpu = value["gpu"]
    _exact(gpu, ("adapter", "ownership_complete", "unreadable", "conflicts", "active_contexts", "scope"), "GPU")
    _integer(gpu["unreadable"], "unreadable GPU records")
    if (gpu["adapter"] != ADAPTERS["gpu_clearance"] or gpu["ownership_complete"] is not True
        or gpu["unreadable"] != 0 or gpu["conflicts"] != [] or gpu["active_contexts"] != []
        or gpu["scope"] != "injected-no-real-device-inspected"):
        raise ValueError("GPU idle/ownership fixture evidence incomplete or conflicted")
    return value


def verify_preflight(policy, observations, *, now, scope=FIXTURE_SCOPE):
    """Validate two observations without launching, probing or granting authority.

    Only mocked adapters are implemented. Even a valid fixture cannot be relabeled
    actual production evidence. A future platform adapter needs actual deadline,
    quota, ownership, GPU and private-boundary tests under separately given scope.
    """
    if scope != FIXTURE_SCOPE:
        raise RuntimeError("Actual production backend adapters are not implemented or authorized")
    policy.validate()
    _finite(now, "verification time")
    if type(observations) not in (list, tuple) or len(observations) != 2:
        raise ValueError("Two independently sequenced bounded observations required")
    values = [_verify_observation(policy, parse_observation(raw), now) for raw in observations]
    first, last = values
    if last["sequence"] <= first["sequence"] or last["observed_at"] <= first["observed_at"]:
        raise ValueError("A repeated observation is not a recheck")
    for key in ("boundary", "quota", "observer", "watchdog"):
        if first[key] != last[key]:
            raise ValueError("Backend identity/membership changed between checks")
    if any(len(raw) > value["observer"]["maximum_bytes"] for raw, value in zip(observations, values)):
        raise ValueError("Response exceeds its observer-specific envelope")
    evidence = {"observations": values, "observation_sha256": [canonical_hash(v) for v in values],
                "minimum_sampled_host_available_bytes": min(v["host"]["available_bytes"] for v in values),
                "measurement_scope": "authored sanitized fixtures; not continuous or platform evidence"}
    backend = {key: last[key] for key in ("boundary", "quota", "observer", "watchdog", "gpu")}
    return dict(schema="dongxi-production-preflight-v1", stage_id=policy.stage_id,
                preparation_sha256=policy.preparation_sha256, scope=FIXTURE_SCOPE, status="ready",
                observation_nonce=policy.observation_nonce,
                observed_utc=datetime.fromtimestamp(last["observed_at"], timezone.utc).isoformat(),
                backend_contract_sha256=canonical_hash(backend), backend_gates={gate: "ready" for gate in GATES},
                evidence=evidence, production_ready=False, launch_authorized=False)


def verify_fixture_drain(policy, raw, *, actual_leader_exit):
    """Check a claimed drain separately from actual exit; never discover or signal."""
    policy.validate()
    value = parse_observation(raw)
    _exact(value, ("schema", "scope", "instance_id", "owner_uid", "membership_complete",
                   "remaining_members", "cleanup_errors", "boundary_removed"), "drain")
    _integer(actual_leader_exit, "leader exit", minimum=-255, maximum=255)
    if (value["schema"] != "dongxi-fixture-drain-v1" or value["scope"] != FIXTURE_SCOPE
        or value["instance_id"] != policy.instance_id or type(value["owner_uid"]) is not int
        or value["owner_uid"] != policy.owner_uid):
        raise ValueError("Drain belongs to a different or unowned boundary")
    if (value["membership_complete"] is not True or value["remaining_members"] != []
        or value["cleanup_errors"] != [] or value["boundary_removed"] is not True
        or actual_leader_exit != 0):
        raise ValueError("Leader exit alone cannot satisfy owned-boundary cleanup")
    return {"status": "fixture-drained", "scope": FIXTURE_SCOPE, "actual_leader_exit": actual_leader_exit,
            "instance_id": policy.instance_id, "production_verified": False, "signals_sent": 0}
