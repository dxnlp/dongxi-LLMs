"""Bounded, immutable trusted-local training snapshots, not a pickle sandbox.

The caller supplies an independently retained byte digest, exact size, scientific
contract and explicit resource envelope. The adjacent JSON is only an untrusted
commit marker. Its receipt is never substituted for the caller's expectation.
Runners validate their own cursor, metric, parameter and pending-pool semantics
before applying the returned state. This module launches no model or process.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from uuid import uuid4

import torch

from .run_identity import canonical_hash
from .artifact_budget import ArtifactBudget

SCHEMA_VERSION = 1
HEADER_LIMIT = 65_536
HEADER_FIELDS = frozenset(("schema_version", "payload_sha256", "payload_bytes",
                          "contract_sha256", "phase", "completed_updates"))
PAYLOAD_FIELDS = frozenset(("schema_version", "contract", "state", "phase",
                           "completed_updates", "parent_invocation"))


def _call(guard):
    if guard is not None:
        guard()


def _positive(value, name):
    if type(value) is not int or value < 1 or value > 2**63 - 1:
        raise ValueError(f"{name} must be a bounded positive integer")


def _step(value):
    if type(value) is not int or not 0 <= value <= 2**63 - 1:
        raise ValueError("completed_updates must be a nonnegative integer")


def _phase(value):
    if value not in ("completed", "pending") or type(value) is not str:
        raise ValueError("Unknown snapshot phase")


def _sha(value, name):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"Invalid {name}")


def _safe_tree(value, *, max_bytes, guard=None, contract=False, clone=False):
    """Reject arbitrary objects/cycles; freeze state before serializing it.

    Only dense real tensors are supported. Serialized size is bounded separately;
    summed tensor storage is a conservative early gate, not an allocation sandbox.
    """
    ancestors = set()
    tensor_bytes = nodes = 0

    def visit(item, depth=0):
        nonlocal tensor_bytes, nodes
        _call(guard)
        nodes += 1
        if depth > 64 or nodes > 1_000_000:
            raise ValueError("Snapshot nesting/node bound exceeded")
        if isinstance(item, torch.Tensor):
            if contract or item.layout != torch.strided or item.is_quantized or item.is_complex():
                raise ValueError("Only dense real state tensors are supported")
            tensor_bytes += item.numel() * item.element_size()
            if tensor_bytes > max_bytes:
                raise ValueError("Tensor storage exceeds snapshot envelope")
            if item.is_floating_point() and not bool(torch.isfinite(item).all()):
                raise ValueError("Nonfinite snapshot tensor")
            return item.detach().cpu().clone() if clone else item
        if isinstance(item, dict) or type(item) in (list, tuple):
            marker = id(item)
            if marker in ancestors:
                raise ValueError("Cyclic snapshot container")
            ancestors.add(marker)
            try:
                if isinstance(item, dict):
                    result = {}
                    for key, child in item.items():
                        if type(key) not in ((str,) if contract else (str, int)):
                            raise ValueError("Unsupported snapshot mapping key")
                        result[key] = visit(child, depth + 1)
                else:
                    children = [visit(child, depth + 1) for child in item]
                    result = tuple(children) if type(item) is tuple else children
                return result
            finally:
                ancestors.remove(marker)
        if type(item) not in (str, int, float, bool, type(None)):
            raise ValueError("Unsupported snapshot object")
        if type(item) is float and not math.isfinite(item):
            raise ValueError("Nonfinite snapshot primitive")
        return item

    return visit(value)


def _contract(value, max_bytes, guard=None):
    if not isinstance(value, dict) or not value:
        raise ValueError("A nonempty scientific contract is required")
    return _safe_tree(value, max_bytes=max_bytes, guard=guard, contract=True)


class _BoundedWriter:
    def __init__(self, handle, limit, guard):
        self.handle, self.limit, self.guard = handle, limit, guard

    def write(self, data):
        _call(self.guard)
        if self.handle.tell() + len(data) > self.limit:
            raise ValueError("Serialized snapshot exceeds max_bytes")
        return self.handle.write(data)

    def flush(self):
        self.handle.flush()

    def tell(self):
        return self.handle.tell()

    def seek(self, offset, whence=0):
        position = self.handle.seek(offset, whence)
        if position > self.limit:
            raise ValueError("Snapshot seek exceeds max_bytes")
        return position


def _fsync_directory(directory):
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _commit_path(path):
    return Path(str(path) + ".commit.json")


def _open_regular(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("Snapshot must be a regular file")
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate snapshot header key")
        result[key] = value
    return result


def _read_header(path):
    with _open_regular(_commit_path(path)) as handle:
        if os.fstat(handle.fileno()).st_size > HEADER_LIMIT:
            raise ValueError("Snapshot header exceeds bound")
        raw = handle.read(HEADER_LIMIT + 1)
    if len(raw) > HEADER_LIMIT:
        raise ValueError("Snapshot header exceeds bound")
    try:
        header = json.loads(raw, object_pairs_hook=_unique_object,
                            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite header")))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("Malformed snapshot header") from error
    if type(header) is not dict or set(header) != HEADER_FIELDS:
        raise ValueError("Unknown/incomplete snapshot header fields")
    if type(header["schema_version"]) is not int or header["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unknown snapshot header schema")
    _positive(header["payload_bytes"], "payload_bytes")
    _sha(header["payload_sha256"], "payload_sha256")
    _sha(header["contract_sha256"], "contract_sha256")
    _step(header["completed_updates"])
    _phase(header["phase"])
    return header


def _verified_bytes(path, *, expected_sha256, expected_bytes, expected_contract,
                    max_bytes, guard, retain):
    _positive(max_bytes, "max_bytes")
    _positive(expected_bytes, "expected_bytes")
    _sha(expected_sha256, "expected_sha256")
    if expected_bytes > max_bytes:
        raise ValueError("Expected snapshot exceeds max_bytes")
    contract = _contract(expected_contract, max_bytes, guard)
    _call(guard)
    header = _read_header(path)
    if header["payload_sha256"] != expected_sha256 or header["payload_bytes"] != expected_bytes:
        raise ValueError("Header differs from independently expected bytes")
    if header["contract_sha256"] != canonical_hash(contract):
        raise ValueError("Snapshot scientific contract mismatch")
    digest = hashlib.sha256()
    buffer = io.BytesIO() if retain else None
    total = 0
    with _open_regular(path) as handle:
        if os.fstat(handle.fileno()).st_size != expected_bytes:
            raise ValueError("Actual snapshot size mismatch")
        while True:
            _call(guard)
            chunk = handle.read(min(1_048_576, expected_bytes - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > expected_bytes:
                raise ValueError("Actual snapshot exceeds expected bytes")
            digest.update(chunk)
            if buffer is not None:
                buffer.write(chunk)
    if total != expected_bytes or digest.hexdigest() != expected_sha256:
        raise ValueError("Snapshot byte identity mismatch")
    if buffer is not None:
        buffer.seek(0)
    return header, buffer, contract


def inspect_snapshot(path, *, expected_sha256, expected_bytes, expected_contract,
                     max_bytes, guard=None):
    """Verify actual bytes and metadata without deserializing a tensor."""
    header, _, _ = _verified_bytes(path, expected_sha256=expected_sha256,
        expected_bytes=expected_bytes, expected_contract=expected_contract,
        max_bytes=max_bytes, guard=guard, retain=False)
    return header


def load_snapshot(path, *, expected_sha256, expected_bytes, expected_contract,
                  max_bytes, validate_payload=None, guard=None):
    """Restricted load of the exact verified buffer; never re-open its path.

    Memory includes a buffer of the declared file size plus restored tensors.
    Explicit file/storage bounds and a callback do not sandbox malicious tensor
    metadata: use only locally owned receipts/artifacts, not hostile checkpoints.
    """
    header, buffer, contract = _verified_bytes(path, expected_sha256=expected_sha256,
        expected_bytes=expected_bytes, expected_contract=expected_contract,
        max_bytes=max_bytes, guard=guard, retain=True)
    try:
        _call(guard)
        payload = torch.load(buffer, map_location="cpu", weights_only=True)
    finally:
        buffer.close()
    if type(payload) is not dict or set(payload) != PAYLOAD_FIELDS:
        raise ValueError("Unknown/incomplete snapshot payload fields")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unknown snapshot payload schema")
    _step(payload["completed_updates"])
    _phase(payload["phase"])
    if payload["completed_updates"] != header["completed_updates"] or payload["phase"] != header["phase"]:
        raise ValueError("Snapshot header/payload progress mismatch")
    observed_contract = _contract(payload["contract"], max_bytes, guard)
    if observed_contract != contract or canonical_hash(observed_contract) != header["contract_sha256"]:
        raise ValueError("Snapshot payload contract mismatch")
    if type(payload["state"]) is not dict or type(payload["parent_invocation"]) is not str or not payload["parent_invocation"]:
        raise ValueError("Missing state/invocation binding")
    _safe_tree(payload["state"], max_bytes=max_bytes, guard=guard)
    if validate_payload is not None:
        validate_payload(payload)
    _call(guard)
    return payload


def save_snapshot(path, *, contract, state, completed_updates, parent_invocation,
                  phase="completed", max_bytes, guard=None, artifact_budget=None):
    """Freeze state and exclusively commit bytes, then a fsynced JSON receipt.

    Any failure keeps its partial data/header temporary file for diagnosis.
    Existing targets are never overwritten; resume must use a new evidence path.
    Successful return is after data/header/directory fsync. A failure after header
    publication may leave a readable marker: callers must record the failed fsync
    and verify the receipt instead of assuming either durable success or deletion.
    An optional cooperative budget reserves payload/marker/staging coexistence;
    it covers only its private-root writer, not arbitrary exports or children.
    """
    _positive(max_bytes, "max_bytes")
    _step(completed_updates)
    _phase(phase)
    if type(parent_invocation) is not str or not parent_invocation:
        raise ValueError("A nonempty parent_invocation is required")
    contract = _contract(contract, max_bytes, guard)
    if type(state) is not dict:
        raise ValueError("Snapshot state must be a dictionary")
    frozen = _safe_tree(state, max_bytes=max_bytes, guard=guard, clone=True)
    payload = dict(schema_version=SCHEMA_VERSION, contract=contract, state=frozen,
                   phase=phase, completed_updates=completed_updates,
                   parent_invocation=parent_invocation)
    path = Path(path)
    marker = _commit_path(path)
    if marker.exists() or marker.is_symlink():
        raise FileExistsError(marker)
    reservation = None
    if artifact_budget is not None:
        if not isinstance(artifact_budget, ArtifactBudget):
            raise ValueError("A verified cooperative artifact ledger is required")
        payload_name = artifact_budget.name_for(path)
        marker_name = artifact_budget.name_for(marker)
        temporary_name = payload_name + ".commit-" + uuid4().hex
        reservation = artifact_budget.reserve_bundle("snapshot-" + uuid4().hex,
            {payload_name: max_bytes, marker_name: HEADER_LIMIT, temporary_name: HEADER_LIMIT})
    _call(guard)
    if reservation is None:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        context = os.fdopen(descriptor, "wb")
    else:
        context = reservation.writer(payload_name)
    with context as handle:
        torch.save(payload, _BoundedWriter(handle, max_bytes, guard))
        _call(guard)
        handle.flush()
        os.fsync(handle.fileno())
    digest = hashlib.sha256()
    total = 0
    with _open_regular(path) as handle:
        while True:
            _call(guard)
            chunk = handle.read(1_048_576)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError("Written snapshot exceeds max_bytes")
            digest.update(chunk)
    header = dict(schema_version=SCHEMA_VERSION, payload_sha256=digest.hexdigest(),
                  payload_bytes=total, contract_sha256=canonical_hash(contract),
                  phase=phase, completed_updates=completed_updates)
    encoded = (json.dumps(header, sort_keys=True, allow_nan=False) + "\n").encode()
    _call(guard)
    if reservation is None:
        descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".commit-", dir=path.parent)
        context = os.fdopen(descriptor, "wb")
    else:
        temporary = artifact_budget.root / temporary_name
        context = reservation.writer(temporary_name)
    with context as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    _call(guard)
    # Hard-link publication is atomic and exclusive, unlike replacing a target.
    if reservation is None:
        os.link(temporary, marker, follow_symlinks=False)
    else:
        reservation.link(temporary_name, marker_name)
    _fsync_directory(path.parent)
    if reservation is None:
        os.unlink(temporary)  # Only the exact temporary file created above.
    else:
        reservation.remove_staging(temporary_name)
    _fsync_directory(path.parent)
    return header
