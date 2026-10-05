"""Persistent story MODEL-work accounting; not byte I/O or a physical quota.

Data preparation, selection, tokenization, hashes, serialization, checkpoint
backward recomputation and journal processing are outside these declared units.
The independent receipt binds a completed legacy-layout story payload to the
same physical retained journal before any payload read.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from .run_identity import canonical_hash
from .snapshot_io_budget import read_bounded_json
from .work_budget import WorkLedger, MAX_JOURNAL_BYTES, integer, validate_limits

STORY_WORK_KEYS = (
    'model_initializations', 'train_updates', 'training_windows',
    'training_valid_targets', 'policy_forward_calls', 'policy_forward_positions',
    'backward_calls', 'optimizer_calls', 'evaluation_panels', 'evaluation_windows',
    'evaluation_valid_targets', 'generation_sequences', 'generation_calls',
    'generation_positions', 'generation_tokens', 'multinomial_draws',
    'activation_panels', 'activation_positions', 'activation_block_applications',
    'save_operations', 'restore_operations',
)
SCHEMA = 'dongxi-story-logical-work-v1'
RECEIPT_SCHEMA = 'dongxi-story-work-receipt-v1'
SCOPE = 'declared-model-operations-not-flops-recomputation-io-or-physical-resources'


def story_work_contract(limits, max_journal_bytes):
    limits = validate_limits(limits)
    if set(limits) != set(STORY_WORK_KEYS):
        raise ValueError('Exact story logical-work dimensions required')
    integer(max_journal_bytes, 'max_journal_bytes')
    if not 1 <= max_journal_bytes <= MAX_JOURNAL_BYTES:
        raise ValueError('Explicit story journal bound <=64MiB required')
    return dict(schema=SCHEMA, limits=limits, max_journal_bytes=max_journal_bytes,
                scope=SCOPE, reservation='permanent-whole-operation-before-model-work')


def validate_story_contract(contract):
    if type(contract) is not dict or len(contract) != 5 or set(contract) != {
            'schema', 'limits', 'max_journal_bytes', 'scope', 'reservation'}:
        raise ValueError('Known explicit story work contract required')
    expected = story_work_contract(contract['limits'], contract['max_journal_bytes'])
    if contract != expected:
        raise ValueError('Changed or obsolete story work schema')
    return deepcopy(expected)


def validate_story_receipt(receipt):
    keys = {'schema', 'payload_sha256', 'payload_bytes', 'contract_sha256',
            'completed_updates', 'work_prefix'}
    if type(receipt) is not dict or len(receipt) != len(keys) or set(receipt) != keys:
        raise ValueError('Independent accounted story receipt required')
    if receipt['schema'] != RECEIPT_SCHEMA:
        raise ValueError('Unknown story receipt schema')
    for key in ('payload_sha256', 'contract_sha256'):
        if type(receipt[key]) is not str or re.fullmatch('[0-9a-f]{64}', receipt[key]) is None:
            raise ValueError('Independent story digest required')
    integer(receipt['payload_bytes'], 'payload_bytes')
    integer(receipt['completed_updates'], 'completed_updates')
    if receipt['payload_bytes'] == 0 or type(receipt['work_prefix']) is not dict:
        raise ValueError('Invalid accounted story receipt')
    return deepcopy(receipt)


def read_story_receipt(path):
    value, _ = read_bounded_json(path)
    return validate_story_receipt(value)


def read_story_limits(path):
    """Exact full contract, never inferred caps; bounded nonblocking metadata."""
    value, identity = read_bounded_json(path)
    return validate_story_contract(value), identity


def assert_resume_disjoint(payload, roles):
    """Metadata-only first-read gate; selected payload cannot hide in bootstrap.

    Exact paths and same-inode hardlinks reject. No-follow ancestry also rejects
    symlink bootstrap aliases. This is cooperative admission, not a hostile
    namespace or concurrent-replacement sandbox.
    """
    def metadata(path):
        path = Path(os.path.abspath(path)); handles = []
        try:
            parent = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            handles.append(parent)
            for part in path.parent.parts[1:]:
                parent = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
                handles.append(parent)
            info = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError('Regular no-follow bootstrap role required')
            return (info.st_dev, info.st_ino)
        except FileNotFoundError:
            return None
        finally:
            for fd in reversed(handles): os.close(fd)
    payload = Path(os.path.abspath(payload)); identity = metadata(payload)
    if identity is None: raise ValueError('Selected resume payload does not exist')
    for role in roles:
        if role is None: continue
        role = Path(os.path.abspath(role))
        if role == payload or metadata(role) == identity:
            raise ValueError('Selected resume payload aliases a bootstrap/journal/input role')


def manifest_input_roles(data):
    """Bounded manifest metadata only, before the native loader's full hashes."""
    root = Path(data)
    manifest, _ = read_bounded_json(root / 'manifest.json')
    if type(manifest) is not dict:
        raise ValueError('Prepared story manifest object required')
    roles = []
    for split in ('train', 'valid'):
        branch = manifest.get(split)
        if type(branch) is not dict or type(branch.get('files')) is not dict:
            raise ValueError('Prepared story manifest file roles required')
        for name in branch['files']:
            if type(name) is not str or not name:
                raise ValueError('Explicit prepared story input filename required')
            roles.append(root / name)
    return roles


@contextmanager
def payload_handle(path):
    """No-follow regular payload handle. No byte quota or hostile-file sandbox."""
    path = Path(os.path.abspath(path)); handles = []
    try:
        parent = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        handles.append(parent)
        for part in path.parent.parts[1:]:
            parent = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            handles.append(parent)
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        handles.append(fd)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError('Regular no-follow story payload required')
        with os.fdopen(os.dup(fd), 'rb') as stream:
            yield stream, before
    finally:
        for fd in reversed(handles): os.close(fd)


def checked_payload(path, receipt):
    """Hash the same open file used by restricted loading, AFTER ledger binding."""
    receipt = validate_story_receipt(receipt)
    @contextmanager
    def checked():
        with payload_handle(path) as (stream, before):
            if before.st_size != receipt['payload_bytes']:
                raise ValueError('Pinned story payload size changed')
            hasher = hashlib.sha256()
            while True:
                block = stream.read(1024 * 1024)
                if not block: break
                hasher.update(block)
            after = os.fstat(stream.fileno())
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError('Story payload changed during hash')
            if hasher.hexdigest() != receipt['payload_sha256']:
                raise ValueError('Pinned story payload bytes changed')
            stream.seek(0)
            yield stream
    return checked()


def publish_story_receipt(path, payload, scientific_contract, prefix, completed_updates):
    with payload_handle(payload) as (stream, info):
        hasher = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''): hasher.update(block)
        after = os.fstat(stream.fileno())
        if (info.st_size, info.st_mtime_ns, info.st_ctime_ns) != (
                after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError('Published story payload changed')
    receipt = validate_story_receipt(dict(schema=RECEIPT_SCHEMA,
        payload_sha256=hasher.hexdigest(), payload_bytes=info.st_size,
        contract_sha256=canonical_hash(scientific_contract),
        completed_updates=completed_updates, work_prefix=prefix))
    raw = (json.dumps(receipt, sort_keys=True, allow_nan=False) + '\n').encode()
    if len(raw) > 64 * 1024: raise ValueError('Story receipt exceeds bounded metadata')
    # No overwrite or silent tail repair. An incomplete file is retained and
    # fails the bounded strict JSON reader; this is not shared snapshot I/O.
    with Path(path).open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return receipt


class StoryOperation:
    def __init__(self, budget, costs, name):
        self.budget = budget
        self.ticket = budget.ledger.reserve(costs, operation=name)
        self.attempted, self.actual = {}, {}

    def enter(self, **units):
        proposed = dict(self.attempted)
        for key, count in units.items(): proposed[key] = proposed.get(key, 0) + count
        self.budget.ledger.assert_within(self.ticket, proposed)
        self.attempted = proposed

    def done(self, **units):
        proposed = dict(self.actual)
        for key, count in units.items(): proposed[key] = proposed.get(key, 0) + count
        self.budget.ledger.assert_within(self.ticket, proposed)
        if any(count > self.attempted.get(key, 0) for key, count in proposed.items()):
            raise ValueError('Successful work must have been entered')
        self.actual = proposed


class StoryWorkBudget:
    def __init__(self, ledger, *, contract, scientific_contract):
        self._contract = validate_story_contract(contract)
        self._science = deepcopy(scientific_contract)
        self._science_sha = canonical_hash(scientific_contract)
        self.ledger = ledger
        self.last_receipt = None
        self.check(scientific_contract)

    @property
    def contract(self): return deepcopy(self._contract)

    @classmethod
    def create(cls, path, *, contract, scientific_contract, invocation_id):
        contract = validate_story_contract(contract)
        ledger = WorkLedger.create(path, limits=contract['limits'],
            contract_sha256=canonical_hash(scientific_contract),
            max_bytes=contract['max_journal_bytes'], invocation_id=invocation_id)
        try: return cls(ledger, contract=contract, scientific_contract=scientific_contract)
        except BaseException: ledger.close(); raise

    @classmethod
    def open(cls, path, *, contract, scientific_contract, invocation_id, receipt):
        contract = validate_story_contract(contract)
        receipt = read_story_receipt(receipt) if isinstance(receipt, (str, Path)) else validate_story_receipt(receipt)
        if receipt['contract_sha256'] != canonical_hash(scientific_contract):
            raise ValueError('Story receipt science/caps/source changed')
        ledger = WorkLedger.open(path, limits=contract['limits'],
            contract_sha256=canonical_hash(scientific_contract),
            max_bytes=contract['max_journal_bytes'], invocation_id=invocation_id,
            expected_snapshot=receipt['work_prefix'])
        try: return cls(ledger, contract=contract, scientific_contract=scientific_contract)
        except BaseException: ledger.close(); raise

    def check(self, scientific_contract):
        if (canonical_hash(scientific_contract) != self._science_sha
                or scientific_contract.get('story_work_budget') != self._contract
                or self.ledger.contract_sha256 != self._science_sha
                or self.ledger.limits != self._contract['limits']
                or self.ledger.max_bytes != self._contract['max_journal_bytes']
                or not self.ledger.bound):
            raise ValueError('Accounted story science/hook/journal mismatch')
        self.ledger.snapshot()

    @contextmanager
    def operation(self, costs, name):
        self.check(self._science)
        work = StoryOperation(self, costs, name)
        try:
            yield work
            self.ledger.complete(work.ticket, work.actual)
        except BaseException as error:
            # Open/reservation records stay charged even if failure recording
            # itself cannot fit or fsync. Never convert failed work to a refund.
            self.ledger.fail(work.ticket, f'{type(error).__name__}: {error}'[:512],
                             known_actual=work.actual, attempted=work.attempted)
            raise

    def close(self): self.ledger.close()
