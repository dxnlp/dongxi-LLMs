"""Compile fixed campaign preparations without any execution or acquisition API.

Only authored local fixtures can pass receipt validation. A preparation is not
consent, a live backend probe, a checkpoint lineage or an executable command.
Actual production mode remains unsupported and fails closed.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import sys

from . import staged_campaign
from .production_preflight import PreflightPolicy, verify_preflight
from .run_identity import canonical_hash, validate_interface

FIXTURE_SCOPE = "authored-local-fixture"
MAX_INTEGER = 2**63 - 1
MAX_RECEIPT_BYTES = 64 * 1024
MAX_ARTIFACT_BYTES = 1024 * 1024
MAX_INVENTORY_BYTES = 4 * 1024 * 1024
MAX_INTERPRETER_BYTES = 32 * 1024 * 1024
STAGE_IDS = (
    "story-profile", "story-control-smoke", "story-control-recovery", "story-control-pilot",
    "story-half-lr-smoke", "story-half-lr-recovery", "story-half-lr-pilot", "story-paired-comparison",
    "assistant-profile-base06", "assistant-frozen-base", "assistant-sft-full-smoke",
    "assistant-sft-full-recovery", "assistant-sft-full-pilot", "assistant-sft-full-evaluation",
    "assistant-sft-lora-smoke", "assistant-sft-lora-recovery", "assistant-sft-lora-pilot",
    "assistant-lora-merge", "assistant-sft-lora-evaluation", "assistant-selected-sft-parent",
    "assistant-dpo-smoke", "assistant-dpo-recovery", "assistant-dpo-pilot",
    "assistant-chosen-sft-smoke", "assistant-chosen-sft-recovery", "assistant-chosen-sft-pilot",
    "assistant-preference-comparison", "assistant-dpo-chosen-nll-extension",
    "assistant-dpo-rehearsal-extension", "reasoning-profile-base", "reasoning-profile-instruct",
    "reasoning-baseline-base-raw", "reasoning-baseline-base-chat",
    "reasoning-baseline-instruct-thinking-off", "reasoning-baseline-instruct-thinking-on",
    "reasoning-rlvr-g4-smoke", "reasoning-rlvr-g4-recovery", "reasoning-rlvr-g4-pilot",
    "reasoning-rlvr-g8-smoke", "reasoning-rlvr-g8-recovery", "reasoning-rlvr-g8-pilot",
    "reasoning-g4-g8-comparison", "assistant-1p7-confirmation",
    "reasoning-multi-seed-confirmation", "capstone-branch-comparison",
)
LIMIT_KEYS = (
    "external_seconds", "reserve_bytes", "artifact_bytes", "artifact_entries",
    "per_file_bytes", "snapshot_max_bytes", "hard_memory_bytes", "hard_pid_limit",
    "max_valid_training_targets", "max_padded_training_positions",
    "max_policy_forward_positions", "max_reference_forward_positions",
    "max_evaluation_forward_positions", "max_generation_forward_positions",
    "max_response_slots", "max_prompt_positions", "private_instance_id", "owner_uid",
)
SOURCE_FILES = tuple(dict.fromkeys((*staged_campaign.SOURCE_FILES,
    "src/dongxi_llms/production_stage.py", "tests/test_production_stage.py",
    "src/dongxi_llms/production_preflight.py",
    "src/dongxi_llms/checkpoint_merge.py",
    "experiments/specs/2026-10-05-production-stage-preparation.md")))


def _exact(value, keys, name):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(f"{name} fields must match the exact schema")


def _integer(value, name, minimum=0):
    if type(value) is not int or not minimum <= value <= MAX_INTEGER:
        raise ValueError(f"{name} must be an exact bounded integer")


def _sha(value, name):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")


def _strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    except (UnicodeError, RecursionError, json.JSONDecodeError) as error:
        raise ValueError("Malformed bounded JSON") from error
    count = 0
    def visit(node, depth=0):
        nonlocal count
        count += 1
        if depth > 32 or count > 10000:
            raise ValueError("JSON structure exceeds the fixture envelope")
        if type(node) is dict:
            for key, item in node.items():
                if type(key) is not str:
                    raise ValueError("String JSON keys required")
                visit(item, depth+1)
        elif type(node) is list:
            for item in node: visit(item, depth+1)
        elif type(node) is float and not math.isfinite(node):
            raise ValueError("Finite JSON numbers required")
        elif type(node) not in (str, int, float, bool, type(None)):
            raise ValueError("Only plain JSON values accepted")
    visit(value)
    return value


def _relative(name):
    if type(name) is not str or len(name) > 256 or "\\" in name:
        raise ValueError("Bounded relative inventory name required")
    parts = name.split("/")
    if (PurePosixPath(name).is_absolute() or any(p in ("", ".", "..") for p in parts)
        or any(re.fullmatch(r"[A-Za-z0-9_.-]+", p) is None for p in parts)):
        raise ValueError("Inventory cannot supply arbitrary or escaping paths")
    return parts


def _read(root, name, *, cap, expected_bytes=None, expected_sha256=None):
    """Open each component under a retained directory FD, without following links."""
    parts = _relative(name)
    if expected_bytes is not None:
        _integer(expected_bytes, "expected file bytes", 1)
        if expected_bytes > cap: raise ValueError("File exceeds the fixture read envelope")
    if expected_sha256 is not None: _sha(expected_sha256, "expected file hash")
    root_parts = Path(root).absolute().parts
    if any(part in (".", "..") for part in root_parts):
        raise ValueError("Trusted root cannot contain traversal components")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in root_parts[1:]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = child
        for component in parts[:-1]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = child
        handle = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        try:
            info = os.fstat(handle)
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= cap:
                raise ValueError("Bounded nonempty regular file required")
            chunks, remaining = [], cap+1
            while remaining:
                chunk = os.read(handle, min(65536, remaining))
                if not chunk: break
                chunks.append(chunk); remaining -= len(chunk)
            raw = b"".join(chunks)
            after = os.fstat(handle)
            if len(raw) > cap or len(raw) != info.st_size or after.st_size != info.st_size:
                raise ValueError("File changed size during bounded read")
            digest = hashlib.sha256(raw).hexdigest()
            if expected_bytes is not None and len(raw) != expected_bytes:
                raise ValueError("Actual bytes differ from independently expected length")
            if expected_sha256 is not None and digest != expected_sha256:
                raise ValueError("Actual bytes differ from independently expected hash")
            return raw, {"path": name, "sha256": digest, "bytes": len(raw)}
        finally:
            os.close(handle)
    finally:
        os.close(fd)


@dataclass(frozen=True)
class ReceiptRef:
    name: str
    sha256: str
    bytes: int

    def validate(self):
        if type(self.name) is not str or re.fullmatch(r"[A-Za-z0-9_-]{1,80}", self.name) is None:
            raise ValueError("Receipt name cannot be a filesystem path")
        _sha(self.sha256, "receipt hash"); _integer(self.bytes, "receipt bytes", 1)
        if self.bytes > MAX_RECEIPT_BYTES: raise ValueError("Receipt exceeds bounded JSON envelope")


@dataclass(frozen=True)
class PrerequisiteRef:
    receipt: ReceiptRef
    preparation_sha256: str
    result_sha256: str
    parent_manifest_sha256: str


def _receipt(root, reference):
    if type(reference) is not ReceiptRef:
        raise ValueError("An independently retained ReceiptRef is required")
    reference.validate()
    raw, identity = _read(root, reference.name+".json", cap=MAX_RECEIPT_BYTES,
        expected_bytes=reference.bytes, expected_sha256=reference.sha256)
    return _strict_json(raw), identity


def _row(stage_id):
    if type(stage_id) is not str or stage_id not in STAGE_IDS:
        raise ValueError("Unknown fixed campaign stage ID")
    rows = staged_campaign.stage_rows()
    if tuple(row["id"] for row in rows) != STAGE_IDS:
        raise ValueError("Campaign allowlist changed; a declared compiler revision is required")
    return deepcopy(next(row for row in rows if row["id"] == stage_id))


def _runner(row):
    identifier, branch = row["id"], row["branch"]
    implemented = row["stage"] in ("profile", "smoke", "recovery", "pilot")
    if branch == "story":
        source = "scripts/train_stories.py"
        role = "fresh-story-training"
    elif "assistant-sft-" in identifier or identifier == "assistant-profile-base06":
        source = "scripts/run_chapter09_spark_sft.py"; role = "supervised-fine-tuning"
    elif identifier.startswith("assistant-dpo-") and not row["optional"]:
        source = "scripts/run_chapter11_spark_dpo.py"; role = "direct-preference-optimization"
    elif identifier.startswith("reasoning-rlvr-"):
        source = "src/dongxi_llms/qwen_rlvr_lab.py"; role = "reasoning-rlvr"
    elif identifier == "assistant-lora-merge":
        source = "src/dongxi_llms/checkpoint_merge.py"; role = "explicit-lora-merge"; implemented = False
    else:
        source = "src/dongxi_llms/reasoning_generation.py"; role = "stage-adapter-required"; implemented = False
    return {"source": source, "operation": row["stage"], "role": role,
        "adapter_status": "runner-source-exists-integration-pending" if implemented else "stage-adapter-unimplemented",
        "execution_api": None, "argv": None,
        "boundary": "Source binding only; profile, recovery and supervisor protocols require actual separately approved integration."}


def _limits(value, row):
    _exact(value, LIMIT_KEYS, "resource/work limits")
    for key in LIMIT_KEYS:
        if key == "private_instance_id": continue
        _integer(value[key], key, 1 if key not in (
            "max_valid_training_targets", "max_padded_training_positions", "max_policy_forward_positions",
            "max_reference_forward_positions", "max_evaluation_forward_positions", "max_generation_forward_positions",
            "max_response_slots") else 0)
    if value["reserve_bytes"] < 25 * 1024**3:
        raise ValueError("Host reserve cannot fall below25GiB")
    if not value["snapshot_max_bytes"] <= value["per_file_bytes"] <= value["artifact_bytes"]:
        raise ValueError("Snapshot/per-file/aggregate ceilings are inconsistent")
    if type(value["private_instance_id"]) is not str or re.fullmatch(r"dongxi-stage-[0-9a-f]{32}", value["private_instance_id"]) is None:
        raise ValueError("An exact private fixture instance is required")
    proposed = row["proposed_budget"]["external_seconds"]
    if proposed is not None and value["external_seconds"] > proposed:
        raise ValueError("Preparation cannot relax the fixed stage deadline")
    target_cap = row["proposed_budget"]["valid_training_target_cap"]
    if target_cap is not None and value["max_valid_training_targets"] > target_cap:
        raise ValueError("Preparation cannot relax the fixed valid-target ceiling")
    return deepcopy(value)


def work_envelope(row, *, prompt_positions=None):
    """Source-derived logical training bounds, never measured FLOPs or total job cost."""
    b = row["proposed_budget"]
    updates, length, group, cap = (b[k] for k in ("maximum_updates", "sequence_positions_per_example_max", "group_size", "max_new_tokens"))
    result = {"padded_geometry": b["maximum_padded_training_positions"],
        "valid_targets": "not inferred from padded geometry", "policy_scored_positions": None,
        "reference_scored_positions": None, "response_slots": b["maximum_attempted_response_tokens"],
        "generation_prefix_positions": None, "scope": "logical maximum training inputs only; evaluation, retries and backward recomputation excluded"}
    if updates is not None and length is not None:
        if row["id"].startswith("assistant-dpo-"):
            result["policy_scored_positions"] = result["reference_scored_positions"] = 2 * updates * b["accumulation"] * (length-1)
        else:
            result["policy_scored_positions"] = b["maximum_padded_training_positions"]
            result["reference_scored_positions"] = 0
    elif updates is not None and group is not None and cap is not None and prompt_positions is not None:
        prefix = updates * group * (cap*prompt_positions + cap*(cap-1)//2)
        rescoring = updates * group * (prompt_positions+cap-1)
        result.update(generation_prefix_positions=prefix,
            policy_scored_positions=prefix+2*rescoring, reference_scored_positions=rescoring)
    return result


def _bindings(value, artifact_root, row):
    _exact(value, ("scope", "files", "parent_slots", "interface_role", "limits"), "local bindings")
    if value["scope"] != FIXTURE_SCOPE:
        raise RuntimeError("Actual model/artifact binding is not implemented or authorized here")
    files = value["files"]
    if type(value["parent_slots"]) is not dict or type(value["interface_role"]) is not str:
        raise ValueError("Fixed interface and parent inventory mappings required")
    if any(type(role) is not str for role in value["parent_slots"].values()):
        raise ValueError("Parent roles must name string inventory entries")
    if type(files) is not dict or not 3 <= len(files) <= 32:
        raise ValueError("Bounded artifact inventory required")
    required = {"data", "interface", "parent"}
    if not required <= set(files): raise ValueError("Data, interface and parent fixture files required")
    identities, documents, total = {}, {}, 0
    for role, expected in files.items():
        if type(role) is not str or re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", role) is None:
            raise ValueError("Bounded artifact role required")
        _exact(expected, ("path", "sha256", "bytes"), "inventory entry")
        raw, identity = _read(artifact_root, expected["path"], cap=MAX_ARTIFACT_BYTES,
            expected_sha256=expected["sha256"], expected_bytes=expected["bytes"])
        total += len(raw)
        if total > MAX_INVENTORY_BYTES: raise ValueError("Total fixture input read ceiling exceeded")
        identities[role] = identity
        if role in (value["interface_role"], *value["parent_slots"].values()):
            documents[role] = _strict_json(raw)
    if value["interface_role"] != "interface": raise ValueError("Fixed interface role required")
    interface = validate_interface(documents["interface"])
    expected_slots = row["proposed_weight_start"]
    expected_slots = expected_slots if type(expected_slots) is list else [expected_slots]
    _exact(value["parent_slots"], expected_slots, "fixed parent slots")
    for slot, role in value["parent_slots"].items():
        if role not in documents: raise ValueError("Parent slot is absent from observed inventory")
        parent = documents[role]
        _exact(parent, ("schema", "scope", "stage_slot", "interface_sha256"), "authored parent metadata")
        if (parent["schema"] != "authored-parent-v1" or parent["scope"] != FIXTURE_SCOPE
            or parent["stage_slot"] != slot or parent["interface_sha256"] != interface["interface_sha256"]):
            raise ValueError("Parent slot/interface observation changed")
    return {"scope": FIXTURE_SCOPE, "files": identities, "parent_slots": deepcopy(value["parent_slots"]),
        "interface": deepcopy(interface), "limits": _limits(value["limits"], row),
        "checkpoint_genealogy": None, "boundary": "Authored metadata/data only; no actual model checkpoint loaded or authenticated."}


def prepare_stage(stage_id, *, bindings=None, artifact_root=None, root=staged_campaign.ROOT):
    row = _row(stage_id)
    contracts = staged_campaign.evaluation_contracts(Path(root))
    evaluation = contracts if row["evaluation_contract"] == "all-contracts-separately" else contracts[row["evaluation_contract"]]
    sources = {name: _read(root, name, cap=4*1024*1024)[1] for name in SOURCE_FILES}
    interpreter = Path(sys.executable).resolve()
    interpreter_identity = _read(interpreter.parent, interpreter.name, cap=MAX_INTERPRETER_BYTES)[1]
    observed = None
    if bindings is not None:
        if artifact_root is None: raise ValueError("Independent artifact root required")
        observed = _bindings(bindings, artifact_root, row)
    elif artifact_root is not None:
        raise ValueError("An artifact root alone cannot supply bindings")
    science = {key: row[key] for key in ("id", "branch", "stage", "prerequisite_stage_ids",
        "proposed_weight_start", "evaluation_contract", "proposed_budget", "intervention", "optional")}
    if row["branch"] == "story":
        historical = _strict_json(_read(root, staged_campaign.HISTORY, cap=4*1024*1024)[0])
        science["historical_model_recipe_declaration"] = historical["contract"]
    runner = _runner(row)
    if runner["source"] not in sources:
        sources[runner["source"]] = _read(root, runner["source"], cap=4*1024*1024)[1]
    envelope = work_envelope(row, prompt_positions=observed["limits"]["max_prompt_positions"] if observed else None)
    result = {"schema": "dongxi-stage-preparation-v1", "stage_id": stage_id,
        "status": "prepared-not-launch-authorized", "science": science, "evaluation": evaluation,
        "runner": runner, "source_files": sources, "lock_sha256": sources["uv.lock"]["sha256"],
        "interpreter": {"binary": interpreter_identity, "python": platform.python_version(),
            "platform": platform.platform(), "machine": platform.machine(),
            "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "tokenizers", "peft")}},
        "bindings": observed, "logical_work_envelope": envelope,
        "unresolved": ["compiler-to-runner cumulative work-contract integration and remaining runner coverage", "actual private containment/hard quota/backend",
            "real stage authority/acquisition/profile/recovery/evaluation"] +
            (["local files/interface/limits unbound"] if observed is None else []) +
            (["fixed stage adapter absent"] if runner["adapter_status"] == "stage-adapter-unimplemented" else []),
        "jobs_started": 0, "launch_authorized": False, "production_ready": False, "launch_command": None}
    result["preparation_sha256"] = canonical_hash(result)
    return result


def compile_stage(request, *, bindings, artifact_root, receipt_root, approval_ref,
                  prerequisite_refs, preflight_ref, observation_nonce, now,
                  root=staged_campaign.ROOT):
    """Validate fixture receipts; there is intentionally no callable runner result."""
    _exact(request, ("stage_id",), "compile request")
    if Path(receipt_root).resolve() == Path(artifact_root).resolve() or Path(artifact_root).resolve() in Path(receipt_root).resolve().parents:
        raise ValueError("Receipts must be independently retained outside input/output artifacts")
    preparation = prepare_stage(request["stage_id"], bindings=bindings, artifact_root=artifact_root, root=root)
    if preparation["runner"]["adapter_status"] == "stage-adapter-unimplemented":
        raise RuntimeError("This fixed stage has no complete current adapter")
    _sha(observation_nonce, "independent observation challenge")
    if type(now) not in (int, float) or not math.isfinite(now) or now < 0:
        raise ValueError("Explicit finite verification time required")
    expected_ids = preparation["science"]["prerequisite_stage_ids"]
    _exact(prerequisite_refs, expected_ids, "independent prerequisite references")
    receipts = {}
    parent_manifest = canonical_hash({slot: preparation["bindings"]["files"][role]
        for slot, role in preparation["bindings"]["parent_slots"].items()})
    for identifier, reference in prerequisite_refs.items():
        if type(reference) is not PrerequisiteRef: raise ValueError("Independent prerequisite identity required")
        for value in (reference.preparation_sha256, reference.result_sha256, reference.parent_manifest_sha256):
            _sha(value, "independent prerequisite identity")
        if reference.parent_manifest_sha256 != parent_manifest:
            raise ValueError("Prerequisite does not bind the actual child parent observations")
        record, identity = _receipt(receipt_root, reference.receipt)
        _exact(record, ("schema", "scope", "stage_id", "status", "preparation_sha256", "result_sha256", "parent_manifest_sha256"), "prerequisite receipt")
        if record != {"schema": "dongxi-stage-prerequisite-v1", "scope": FIXTURE_SCOPE,
            "stage_id": identifier, "status": "fixture-complete", "preparation_sha256": reference.preparation_sha256,
            "result_sha256": reference.result_sha256, "parent_manifest_sha256": reference.parent_manifest_sha256}:
            raise ValueError("Prerequisite receipt differs from independently expected stage/result/input identity")
        receipts[identifier] = identity
    preflight, preflight_identity = _receipt(receipt_root, preflight_ref)
    _exact(preflight, ("schema", "stage_id", "preparation_sha256", "scope", "status", "observation_nonce",
        "observed_utc", "backend_contract_sha256", "backend_gates", "evidence", "production_ready", "launch_authorized"), "preflight receipt")
    if (preflight["schema"] != "dongxi-production-preflight-v1" or preflight["stage_id"] != request["stage_id"]
        or preflight["preparation_sha256"] != preparation["preparation_sha256"] or preflight["scope"] != FIXTURE_SCOPE
        or preflight["status"] != "ready" or preflight["production_ready"] is not False or preflight["launch_authorized"] is not False
        or preflight["observation_nonce"] != observation_nonce):
        raise ValueError("Missing or substituted authored preflight binding")
    limits = preparation["bindings"]["limits"]
    policy = PreflightPolicy(request["stage_id"], preparation["preparation_sha256"], observation_nonce,
        limits["private_instance_id"], limits["owner_uid"], limits["hard_memory_bytes"], limits["hard_pid_limit"],
        limits["artifact_bytes"], limits["artifact_entries"], limits["external_seconds"], minimum_host_available_bytes=limits["reserve_bytes"])
    evidence = preflight["evidence"]
    _exact(evidence, ("observations", "observation_sha256", "minimum_sampled_host_available_bytes", "measurement_scope"), "preflight evidence")
    if type(evidence["observations"]) is not list: raise ValueError("Bounded authored observations required")
    validated = verify_preflight(policy, [json.dumps(v, allow_nan=False).encode() for v in evidence["observations"]], now=now)
    if canonical_hash(validated) != canonical_hash(preflight):
        raise ValueError("Preflight summary does not match its independently revalidated observations")
    approval, approval_identity = _receipt(receipt_root, approval_ref)
    _exact(approval, ("schema", "scope", "stage_id", "preparation_sha256", "decision", "limits_sha256",
        "prerequisite_receipts_sha256", "preflight_receipt_sha256"), "approval receipt")
    expected = {"schema": "dongxi-stage-approval-v1", "scope": FIXTURE_SCOPE, "stage_id": request["stage_id"],
        "preparation_sha256": preparation["preparation_sha256"], "decision": "fixture-validation-only",
        "limits_sha256": canonical_hash(limits), "prerequisite_receipts_sha256": {k: v["sha256"] for k, v in receipts.items()},
        "preflight_receipt_sha256": preflight_identity["sha256"]}
    if approval != expected: raise ValueError("Approval assertions differ from independently pinned fixture conditions")
    return {"schema": "dongxi-compiled-preparation-v1", "status": "validated-authored-fixture-preparation",
        "preparation": preparation, "receipt_identities": {"approval": approval_identity, "preflight": preflight_identity,
            "prerequisites": receipts}, "jobs_started": 0, "launch_authorized": False, "production_ready": False,
        "launch_command": None, "actual_external_result": None,
        "boundary": "Pinned authored fixture assertions only; no real authority, platform readiness, model compatibility or checkpoint genealogy."}
