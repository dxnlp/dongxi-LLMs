"""Declared shared-reader work, not physical containment or pickle sandboxing.

The independent receipt is bounded bootstrap evidence. A bound physical journal
retains every later attempted charge. A checkpoint cannot contain its own future
save completion, so its receipt pins the prefix immediately before reservation.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from .run_identity import canonical_hash
from .work_budget import WorkLedger, WorkBudgetExceeded, integer, validate_limits, MAX_JOURNAL_BYTES

IO_KEYS = tuple('snapshot_' + name for name in (
    'inspect_operations', 'load_operations', 'save_operations', 'hash_bytes',
    'tree_nodes', 'tensor_elements', 'primitive_bytes', 'clone_bytes', 'serialization_bytes'))
ENVELOPE_KEYS = ('max_payload_bytes', 'max_tree_nodes', 'max_tensor_elements',
                 'max_tensor_bytes', 'max_primitive_bytes')
RECEIPT_SCHEMA = 'dongxi-snapshot-work-receipt-v1'
RECEIPT_LIMIT = 65536
HEADER_KEYS = {'schema_version', 'payload_sha256', 'payload_bytes', 'contract_sha256',
               'phase', 'completed_updates'}


def _sha(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('Exact SHA256 required for snapshot work binding')


def io_budget_contract(limits, envelope, max_journal_bytes):
    if type(limits) is not dict or len(limits) != len(IO_KEYS) or set(limits) != set(IO_KEYS):
        raise ValueError('Explicit exact nine-dimensional snapshot I/O caps required')
    validate_limits(limits)
    if type(envelope) is not dict or len(envelope) != len(ENVELOPE_KEYS) or set(envelope) != set(ENVELOPE_KEYS):
        raise ValueError('Explicit whole-operation snapshot envelope required')
    for key, value in envelope.items(): integer(value, key)
    if not 1 <= envelope['max_tree_nodes'] <= 1000000 or envelope['max_payload_bytes'] < 1:
        raise ValueError('Positive bounded node/payload envelopes required')
    if any(envelope[key] > envelope['max_payload_bytes'] for key in ('max_tensor_bytes', 'max_primitive_bytes')):
        raise ValueError('Tensor/primitive envelope exceeds explicit payload bound')
    integer(max_journal_bytes, 'max_journal_bytes')
    if not 1 <= max_journal_bytes <= MAX_JOURNAL_BYTES:
        raise ValueError('Positive snapshot I/O journal bound <=64MiB required')
    return dict(schema='dongxi-snapshot-io-work-v1', limits=dict(limits), envelope=dict(envelope),
                max_journal_bytes=max_journal_bytes,
                scope='shared payload hash/tree/finite/clone/serialization visits; not metadata/journal/caller/physical quota',
                reservation='whole operation, permanent/no refund')


def validate_io_contract(value):
    if type(value) is not dict or set(value) != {'schema', 'limits', 'envelope', 'max_journal_bytes', 'scope', 'reservation'}:
        raise ValueError('Explicit known snapshot I/O contract required')
    if value != io_budget_contract(value['limits'], value['envelope'], value['max_journal_bytes']):
        raise ValueError('Changed/obsolete snapshot I/O contract')
    return deepcopy(value)


def io_ledger_contract_sha256(contract, scientific_contract_sha256):
    _sha(scientific_contract_sha256)
    return canonical_hash(dict(io_contract=validate_io_contract(contract), science_sha256=scientific_contract_sha256))


def read_bounded_json(path):
    """Bootstrap only: <=64KiB regular file through no-follow ancestors."""
    path = Path(os.path.abspath(path)); handles = []
    try:
        parent = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW); handles.append(parent)
        for component in path.parent.parts[1:]:
            parent = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent); handles.append(parent)
        fd = os.open(path.name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=parent); handles.append(fd)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > RECEIPT_LIMIT:
            raise ValueError('Snapshot bootstrap requires bounded regular no-follow metadata')
        raw = os.pread(fd, RECEIPT_LIMIT + 1, 0); after = os.fstat(fd)
        if len(raw) != before.st_size or len(raw) > RECEIPT_LIMIT or (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError('Snapshot bootstrap metadata changed during bounded read')
    finally:
        for fd in reversed(handles): os.close(fd)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result: raise ValueError('Duplicate snapshot bootstrap key')
            result[key] = value
        return result
    try:
        value = json.loads(raw, object_pairs_hook=unique,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite snapshot bootstrap')))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError('Malformed snapshot bootstrap metadata') from error
    return value, dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
                      reader='bounded-nonblocking-no-follow-regular')


def _prefix(value):
    """Cheap bounded schema; authoritative prefix equality belongs to WorkLedger."""
    fields = {'schema', 'ledger_id', 'file_identity', 'contract_sha256', 'limits', 'max_bytes',
              'sequence', 'chain_sha256', 'open_tickets', 'failed_tickets', 'reserved', 'completed',
              'known_partial', 'attempted_upper', 'uncertain_upper'}
    if type(value) is not dict or len(value) != len(fields) or set(value) != fields:
        raise ValueError('Bounded independently retained work prefix required')
    if value['schema'] != 'dongxi-work-ledger-v1': raise ValueError('Unknown receipt work prefix')
    for key in ('contract_sha256', 'chain_sha256'): _sha(value[key])
    if type(value['ledger_id']) is not str or re.fullmatch('[0-9a-f]{32}', value['ledger_id']) is None:
        raise ValueError('Invalid receipt ledger ID')
    for key in ('sequence', 'max_bytes'): integer(value[key], key)
    limits = validate_limits(value['limits'])
    for key in ('reserved', 'completed', 'known_partial', 'attempted_upper', 'uncertain_upper'):
        if type(value[key]) is not dict or len(value[key]) != len(limits) or set(value[key]) != set(limits):
            raise ValueError('Changed receipt work dimensions')
        for dimension, amount in value[key].items(): integer(amount, dimension)
    for key in ('open_tickets', 'failed_tickets', 'file_identity'):
        if type(value[key]) is not list or len(value[key]) > 1024:
            raise ValueError('Bounded receipt ticket/identity arrays required')
        for amount in value[key]: integer(amount, key)
    if len(value['file_identity']) != 2: raise ValueError('Invalid physical receipt journal anchor')


def validate_work_receipt(value):
    fields = {'schema', 'snapshot', 'io_contract_sha256', 'io_prefix', 'runner_work_prefix'}
    if type(value) is not dict or len(value) != len(fields) or set(value) != fields or value['schema'] != RECEIPT_SCHEMA:
        raise ValueError('Explicit independent snapshot work receipt required')
    header = value['snapshot']
    if type(header) is not dict or len(header) != len(HEADER_KEYS) or set(header) != HEADER_KEYS:
        raise ValueError('Invalid independent snapshot header binding')
    if type(header['schema_version']) is not int or header['schema_version'] != 2:
        raise ValueError('Accounted receipt requires explicit snapshot v2')
    for key in ('payload_sha256', 'contract_sha256'): _sha(header[key])
    for key in ('payload_bytes', 'completed_updates'): integer(header[key], key)
    if header['payload_bytes'] < 1 or type(header['phase']) is not str or header['phase'] not in ('completed', 'pending'):
        raise ValueError('Invalid independent snapshot progress/size')
    _sha(value['io_contract_sha256']); _prefix(value['io_prefix'])
    if value['runner_work_prefix'] is not None: _prefix(value['runner_work_prefix'])
    encoded = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    if len(encoded) + 1 > RECEIPT_LIMIT: raise ValueError('Independent snapshot work receipt exceeds64KiB')
    return deepcopy(value)


def read_work_receipt(path):
    value, _ = read_bounded_json(path)
    return validate_work_receipt(value)


class SnapshotIOBudget:
    def __init__(self, ledger, *, contract, scientific_contract_sha256, expected_receipt=None):
        self.contract = validate_io_contract(contract); _sha(scientific_contract_sha256)
        if not isinstance(ledger, WorkLedger) or not ledger.bound:
            raise ValueError('Snapshot I/O needs an independently bound physical work journal')
        if (ledger.limits != self.contract['limits'] or ledger.max_bytes != self.contract['max_journal_bytes']
                or ledger.contract_sha256 != io_ledger_contract_sha256(self.contract, scientific_contract_sha256)):
            raise ValueError('Snapshot I/O journal/scientific contract differs')
        self.ledger = ledger; self.science_sha256 = scientific_contract_sha256
        self._bound_ledger = ledger; self._bound_science_sha256 = scientific_contract_sha256
        self._contract_sha256 = canonical_hash(self.contract)
        self.expected_receipt = None; self._receipt_sha256 = None; self.last_receipt = None
        if expected_receipt is not None: self.bind_receipt(expected_receipt)

    def bind_receipt(self, receipt):
        self._check_identity()
        receipt = validate_work_receipt(receipt)
        if receipt['io_contract_sha256'] != canonical_hash(self.contract) or receipt['snapshot']['contract_sha256'] != self.science_sha256:
            raise ValueError('Independent receipt scientific/I/O contract differs')
        self.ledger.validate_snapshot(receipt['io_prefix'])
        self.expected_receipt = receipt
        self._receipt_sha256 = canonical_hash(receipt)

    def _check_identity(self):
        if self.ledger is not self._bound_ledger or self.science_sha256 != self._bound_science_sha256:
            raise ValueError('Snapshot I/O bound journal/scientific identity inspection changed')
        validate_io_contract(self.contract)
        if canonical_hash(self.contract) != self._contract_sha256:
            raise ValueError('Snapshot I/O contract inspection map changed')

    def begin(self, operation, *, max_bytes, expected_sha256=None, expected_bytes=None):
        self._check_identity()
        if operation not in ('inspect', 'load', 'save'): raise ValueError('Unknown shared snapshot operation')
        envelope = self.contract['envelope']
        if max_bytes != envelope['max_payload_bytes'] or type(max_bytes) is not int:
            raise ValueError('Shared payload bound differs from declared I/O envelope')
        if operation != 'save':
            if self.expected_receipt is None: raise ValueError('Independently retained receipt required before payload work')
            # Public inspection data are not permission to change the already
            # bound expectation; exact types reject False/0 and True/1 aliases.
            validate_work_receipt(self.expected_receipt)
            if canonical_hash(self.expected_receipt) != self._receipt_sha256:
                raise ValueError('Bound snapshot work receipt inspection map changed')
            header = self.expected_receipt['snapshot']
            if expected_sha256 != header['payload_sha256'] or type(expected_bytes) is not int or expected_bytes != header['payload_bytes']:
                raise ValueError('Read parameters differ from independent receipt')
            if expected_bytes > max_bytes: raise ValueError('Expected payload exceeds I/O envelope')
        costs = dict.fromkeys(IO_KEYS, 0)
        costs['snapshot_' + operation + '_operations'] = 1
        costs['snapshot_hash_bytes'] = max_bytes if operation == 'save' else expected_bytes
        costs['snapshot_tree_nodes'] = envelope['max_tree_nodes']
        costs['snapshot_primitive_bytes'] = envelope['max_primitive_bytes']
        if operation != 'inspect': costs['snapshot_tensor_elements'] = envelope['max_tensor_elements']
        if operation == 'save':
            costs['snapshot_clone_bytes'] = envelope['max_tensor_bytes']
            costs['snapshot_serialization_bytes'] = max_bytes
        prefix = self.ledger.snapshot()
        return SnapshotIOAttempt(self, costs, operation, prefix)


class SnapshotIOAttempt:
    def __init__(self, budget, costs, operation, prefix):
        self.budget = budget; self.operation = operation; self.prefix = prefix
        self.ticket = budget.ledger.reserve(costs, operation='snapshot-' + operation)
        self._costs = tuple(costs[key] for key in IO_KEYS)
        self._tensor_bytes = 0
        self.entered = dict.fromkeys(IO_KEYS, 0); self.success = dict(self.entered)

    def before(self, **costs):
        candidate = dict(self.entered)
        for key, amount in costs.items():
            if key not in candidate: raise ValueError('Unknown snapshot visit dimension')
            integer(amount, key); candidate[key] += amount
        # The whole reservation was durably admitted above. Per-node guards use
        # its private fixed envelope, not repeated O(journal size) file hashing.
        if any(candidate[key] > bound for key, bound in zip(IO_KEYS, self._costs)):
            raise WorkBudgetExceeded('Shared snapshot next visit exceeds its whole-operation envelope')
        self.entered = candidate

    def after(self, **costs):
        for key, amount in costs.items(): self.success[key] += amount

    def visit(self, **costs):
        self.before(**costs); self.after(**costs)

    def admit_tensor_bytes(self, amount):
        integer(amount, 'tensor bytes')
        candidate = self._tensor_bytes + amount
        if candidate > self.budget.contract['envelope']['max_tensor_bytes']:
            raise WorkBudgetExceeded('Shared snapshot tensor bytes exceed envelope before finite scan/clone')
        self._tensor_bytes = candidate

    def complete(self): self.budget.ledger.complete(self.ticket, self.success)

    def fail(self, error):
        if not self.budget.ledger.poisoned:
            self.budget.ledger.fail(self.ticket, type(error).__name__ + ': ' + str(error)[:470],
                                    known_actual=self.success, attempted=self.entered)

    def payload_binding(self):
        return dict(schema=RECEIPT_SCHEMA, io_contract_sha256=canonical_hash(self.budget.contract), io_prefix=deepcopy(self.prefix))

    def verify_payload(self, payload):
        receipt = self.budget.expected_receipt
        binding = payload['snapshot_io']
        if type(binding) is not dict or len(binding) != 3 or set(binding) != {'schema', 'io_contract_sha256', 'io_prefix'}:
            raise ValueError('Malformed payload snapshot I/O binding')
        _prefix(binding['io_prefix']); _sha(binding['io_contract_sha256'])
        if binding != dict(schema=RECEIPT_SCHEMA, io_contract_sha256=receipt['io_contract_sha256'], io_prefix=receipt['io_prefix']):
            raise ValueError('Payload I/O prefix differs from independently retained receipt')
        prefix = payload['state'].get('work_ledger')
        if prefix is not None: _prefix(prefix)
        if prefix != receipt['runner_work_prefix']:
            raise ValueError('Payload runner-work prefix differs from independent receipt')

    def make_receipt(self, header, state):
        return validate_work_receipt(dict(schema=RECEIPT_SCHEMA, snapshot=header,
            io_contract_sha256=canonical_hash(self.budget.contract), io_prefix=deepcopy(self.prefix),
            runner_work_prefix=deepcopy(state.get('work_ledger'))))
