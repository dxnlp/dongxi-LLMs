"""Cooperative cumulative logical work, not FLOPs or a physical resource quota.

Capacity is permanently charged BEFORE work in a bounded fsynced append journal.
Snapshots attest a prefix; recovery reads all later attempted charges as well.
An inode anchor and single writer prevent output-path resets/copy forks. This is
trusted-local accounting, not protection against privileged filesystem rollback.
"""
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from uuid import uuid4

from .run_identity import canonical_hash

SCHEMA = 'dongxi-work-ledger-v1'
MAX_INT = 2**63-1
MAX_JOURNAL_BYTES = 64*1024**2


class WorkBudgetExceeded(RuntimeError):
    pass


class WorkLedgerError(RuntimeError):
    pass


def integer(value, name):
    if type(value) is not int or not 0 <= value <= MAX_INT:
        raise ValueError(f'{name} must be a bounded nonnegative exact integer')


def validate_limits(limits):
    if type(limits) is not dict or not limits or len(limits)>64:
        raise ValueError('Nonempty bounded dimension map required')
    for key,value in limits.items():
        if type(key) is not str or re.fullmatch('[a-z][a-z0-9_]{0,63}',key) is None:
            raise ValueError('Invalid work dimension')
        integer(value,key)
    return dict(limits)


def _vector(value, limits):
    if type(value) is not dict or not set(value) <= set(limits):
        raise ValueError('Unknown work dimensions')
    result={key:0 for key in limits}
    for key,amount in value.items():
        integer(amount,key);result[key]=amount
    return result


def _text(value,name,maximum=128):
    if type(value) is not str or not value or len(value)>maximum:
        raise ValueError(f'Invalid {name}')


def _sha(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}',value) is None:
        raise ValueError('Invalid independent contract/chain digest')


def _unique(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('Duplicate journal JSON key')
        result[key]=value
    return result


def _decode(raw):
    try:
        return json.loads(raw,object_pairs_hook=_unique,
            parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite journal')))
    except (UnicodeError,json.JSONDecodeError,RecursionError) as error:
        raise WorkLedgerError('Malformed retained work journal') from error


class WorkLedger:
    """One owned journal. Reservations never refund, even after early stopping.

    Existing journals opened without expected_snapshot remain unbound and cannot
    reserve until validate_snapshot is called before model state application.
    Raw incomplete records are retained and rejected; no automatic tail repair.
    """
    @classmethod
    def create(cls,path,*,limits,contract_sha256,max_bytes,invocation_id):
        obj=cls._handle(path,limits,contract_sha256,max_bytes,invocation_id,new=True)
        try:
            header=dict(schema=SCHEMA,ledger_id=uuid4().hex,contract_sha256=contract_sha256,
                limits=obj.limits,max_bytes=max_bytes,file_identity=obj.identity)
            obj._write(header)
            obj._init(header);obj.bound=True
            obj._check();os.fsync(obj.parents[-1][0])
            return obj
        except BaseException:
            obj.close();raise

    @classmethod
    def open(cls,path,*,limits,contract_sha256,max_bytes,invocation_id,expected_snapshot=None):
        obj=cls._handle(path,limits,contract_sha256,max_bytes,invocation_id,new=False)
        try:
            raw=os.pread(obj.fd,max_bytes+1,0)
            if len(raw)>max_bytes or not raw.endswith(b'\n'):
                raise WorkLedgerError('Oversized or incomplete retained journal; bytes preserved')
            lines=raw.splitlines(keepends=True)
            if not lines:raise WorkLedgerError('Empty retained journal')
            header=_decode(lines[0]);obj._init(header)
            if header['limits']!=obj.limits or header['contract_sha256']!=contract_sha256 or header['max_bytes']!=max_bytes:
                raise ValueError('Budget limits/contract/journal bound changed')
            for line in lines[1:]:obj._replay(_decode(line))
            obj.size=len(raw)
            obj._bytes_sha256=hashlib.sha256(raw).hexdigest()
            if expected_snapshot is not None:obj.validate_snapshot(expected_snapshot)
            return obj
        except BaseException:
            obj.close();raise

    @classmethod
    def _handle(cls,path,limits,contract_sha256,max_bytes,invocation_id,*,new):
        validate_limits(limits);_sha(contract_sha256);integer(max_bytes,'max_bytes')
        if not 1<=max_bytes<=MAX_JOURNAL_BYTES:raise ValueError('Positive journal byte bound <=64MiB required')
        _text(invocation_id,'invocation ID')
        obj=cls();obj.path=Path(os.path.abspath(path));obj.limits=dict(limits);obj.max_bytes=max_bytes
        obj.contract_sha256=contract_sha256;obj.invocation_id=invocation_id
        obj.poisoned=False;obj.closed=False;obj.bound=False;obj.size=0
        obj._bytes_sha256=hashlib.sha256(b'').hexdigest();obj._sealed=None
        flags=os.O_RDWR|os.O_NOFOLLOW|os.O_NONBLOCK|(os.O_CREAT|os.O_EXCL if new else 0)
        obj.parents=[];obj.fd=None
        try:
            parent=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            obj.parents.append((parent,None))
            for part in obj.path.parent.parts[1:]:
                child=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent)
                obj.parents.append((child,part));parent=child
            info=os.fstat(parent)
            if info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)&0o077:
                raise ValueError('Privately owned final work-journal directory required')
            obj.fd=os.open(obj.path.name,flags,0o600,dir_fd=parent)
            info=os.fstat(obj.fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1 or info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)!=0o600:
                raise ValueError('Owned private single-link regular journal required')
            fcntl.flock(obj.fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            obj.identity=[info.st_dev,info.st_ino]
            if info.st_size>max_bytes:raise WorkLedgerError('Retained journal exceeds byte bound')
        except BaseException:
            if obj.fd is not None:os.close(obj.fd)
            for descriptor,_ in obj.parents:os.close(descriptor)
            obj.closed=True;raise
        return obj

    def _init(self,header):
        if type(header) is not dict or set(header)!={'schema','ledger_id','contract_sha256','limits','max_bytes','file_identity'} or header['schema']!=SCHEMA:
            raise ValueError('Invalid immutable work header')
        if type(header['ledger_id']) is not str or re.fullmatch('[0-9a-f]{32}',header['ledger_id']) is None:
            raise ValueError('Invalid ledger identity')
        validate_limits(header['limits']);_sha(header['contract_sha256']);integer(header['max_bytes'],'max_bytes')
        if type(header['file_identity']) is not list or len(header['file_identity'])!=2 or any(type(i) is not int or i<0 for i in header['file_identity']) or header['file_identity']!=self.identity:
            raise ValueError('Copied/replaced work journal cannot fork its anchored budget')
        self.header=deepcopy(header);self.ledger_id=header['ledger_id'];self.sequence=0
        self.chain=canonical_hash(header);self.tickets={};self.prefixes={0:self._summary()}
        self._seal()

    def _seal(self):
        # Public inspection maps are not authority: accidental edits are refused,
        # not accepted as a refill. Private internals are trusted-local, not ACLs.
        self._sealed=canonical_hash(dict(header=self.header,limits=self.limits,
            tickets=self.tickets,prefixes=self.prefixes,sequence=self.sequence,chain=self.chain,
            ledger_id=self.ledger_id,contract=self.contract_sha256,max_bytes=self.max_bytes,
            identity=self.identity))

    def _check(self):
        if self.closed or self.poisoned:raise WorkLedgerError('Closed/poisoned work ledger')
        try:
            if self._sealed is not None:
                expected=self._sealed;self._seal()
                if self._sealed!=expected:raise ValueError('Accounting inspection state mutated')
            for index,(descriptor,name) in enumerate(self.parents):
                anchored=os.fstat(descriptor)
                current=os.stat('/',follow_symlinks=False) if index==0 else os.stat(name,dir_fd=self.parents[index-1][0],follow_symlinks=False)
                if not stat.S_ISDIR(current.st_mode) or (current.st_dev,current.st_ino)!=(anchored.st_dev,anchored.st_ino):
                    raise ValueError('Ancestor pathname changed')
            parent=os.fstat(self.parents[-1][0])
            if parent.st_uid!=os.getuid() or stat.S_IMODE(parent.st_mode)&0o077:raise ValueError('Private parent mode/owner changed')
            info=os.fstat(self.fd)
            named=os.stat(self.path.name,dir_fd=self.parents[-1][0],follow_symlinks=False)
            if (not stat.S_ISREG(named.st_mode) or [named.st_dev,named.st_ino]!=self.identity
                    or [info.st_dev,info.st_ino]!=self.identity or info.st_size!=self.size or info.st_nlink!=1
                    or info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)!=0o600):
                raise ValueError('Owned journal changed outside its writer')
            raw=os.pread(self.fd,self.max_bytes+1,0)
            if len(raw)!=self.size or hashlib.sha256(raw).hexdigest()!=self._bytes_sha256:
                raise ValueError('Same-inode journal bytes changed')
        except (OSError,ValueError) as error:
            self.poisoned=True;raise WorkLedgerError('Journal pathname/anchor/mode changed; detached writes refused') from error

    def _write(self,value):
        raw=(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
        self._check()
        if self.size+len(raw)>self.max_bytes:
            self.poisoned=True
            raise WorkLedgerError('Journal byte envelope crossed before append; prior bytes retained')
        try:
            os.lseek(self.fd,self.size,os.SEEK_SET)
            offset=0
            while offset<len(raw):
                written=os.write(self.fd,raw[offset:])
                if not written:raise OSError('Zero-byte ledger write')
                offset+=written
            os.fsync(self.fd)
            self.size+=len(raw)
            self._bytes_sha256=hashlib.sha256(os.pread(self.fd,self.size,0)).hexdigest()
        except BaseException as error:
            self.poisoned=True
            raise WorkLedgerError('Uncommitted/uncertain append retained; live ledger poisoned') from error

    def _summary(self):
        result={name:{key:0 for key in self.limits} for name in ('reserved','completed','known_partial','attempted_upper','uncertain_upper')}
        opens=[];failed=[]
        for number,ticket in self.tickets.items():
            for key,value in ticket['costs'].items():result['reserved'][key]+=value
            if ticket['status']=='complete':
                for key,value in ticket['actual'].items():result['completed'][key]+=value
                for key,value in ticket['actual'].items():result['attempted_upper'][key]+=value
            else:
                if ticket['status']=='open':opens.append(number)
                else:failed.append(number)
                for key,value in ticket['actual'].items():result['known_partial'][key]+=value
                attempted=ticket['costs'] if ticket['status']=='open' else ticket['attempted']
                for key,value in attempted.items():
                    result['attempted_upper'][key]+=value
                    result['uncertain_upper'][key]+=value-ticket['actual'][key]
        return dict(schema=SCHEMA,ledger_id=self.ledger_id,file_identity=list(self.identity),
            contract_sha256=self.contract_sha256,limits=dict(self.limits),max_bytes=self.max_bytes,
            sequence=self.sequence,chain_sha256=self.chain,open_tickets=opens,failed_tickets=failed,**result)

    def _replay(self,envelope):
        if type(envelope) is not dict or set(envelope)!={'record','sha256'}:raise ValueError('Invalid journal envelope')
        record=envelope['record'];_sha(envelope['sha256'])
        if type(record) is not dict or envelope['sha256']!=canonical_hash(record):raise ValueError('Changed journal record hash')
        if type(record.get('sequence')) is not int or record['sequence']!=self.sequence+1 or record.get('previous')!=self.chain:
            raise ValueError('Changed journal sequence/chain')
        _text(record.get('invocation_id'),'invocation ID');event=record.get('event')
        if event=='reserve':
            if set(record)!={'sequence','previous','invocation_id','event','ticket','operation','costs'} or type(record['ticket']) is not int or record['ticket']!=record['sequence']:
                raise ValueError('Invalid reservation record')
            _text(record['operation'],'operation')
            costs=_vector(record['costs'],self.limits)
            if record['costs']!=costs or not any(costs.values()):raise ValueError('Full positive reservation vector required')
            reserved=self._summary()['reserved']
            if any(reserved[key]+value>self.limits[key] for key,value in costs.items()):raise ValueError('Retained journal exceeds declared caps')
            self.tickets[record['ticket']]=dict(operation=record['operation'],invocation_id=record['invocation_id'],
                costs=costs,actual={key:0 for key in self.limits},status='open')
        elif event in ('complete','fail'):
            fields={'sequence','previous','invocation_id','event','ticket','actual'}|({'error','attempted'} if event=='fail' else set())
            if set(record)!=fields or type(record['ticket']) is not int or record['ticket'] not in self.tickets:
                raise ValueError('Invalid completion/failure record')
            ticket=self.tickets[record['ticket']]
            if ticket['status']!='open':raise ValueError('Ticket already finalized')
            actual=_vector(record['actual'],self.limits)
            if actual!=record['actual'] or any(actual[key]>ticket['costs'][key] for key in self.limits):raise ValueError('Actual work exceeds whole-operation reservation')
            if event=='fail':
                _text(record['error'],'error',512)
                attempted=_vector(record['attempted'],self.limits)
                if attempted!=record['attempted'] or any(not actual[key]<=attempted[key]<=ticket['costs'][key] for key in self.limits):
                    raise ValueError('Failure attempted/successful work exceeds reservation')
                ticket['attempted']=attempted
            ticket.update(actual=actual,status=event)
        else:raise ValueError('Unknown work event')
        self.sequence=record['sequence'];self.chain=envelope['sha256']
        self.prefixes[self.sequence]=self._summary()
        self._seal()

    def _append(self,event,**fields):
        self._check()
        if not self.bound:raise WorkLedgerError('Bind recovered snapshot before changing retained work')
        record=dict(sequence=self.sequence+1,previous=self.chain,invocation_id=self.invocation_id,event=event,**fields)
        envelope=dict(record=record,sha256=canonical_hash(record))
        # Validate before writing via a detached in-memory replay, not mutation.
        shadow=object.__new__(WorkLedger);shadow.__dict__=dict(self.__dict__)
        shadow.tickets=deepcopy(self.tickets);shadow.prefixes={}
        shadow._replay(envelope)
        self._write(envelope);self._replay(envelope)

    def reserve(self,costs,*,operation):
        self._check()
        if not self.bound:raise WorkLedgerError('Bind recovered snapshot before reserving new work')
        _text(operation,'operation');costs=_vector(costs,self.limits)
        if not any(costs.values()):raise ValueError('No-op reservation is not work')
        reserved=self._summary()['reserved']
        crossing=[key for key,value in costs.items() if reserved[key]+value>self.limits[key]]
        if crossing:raise WorkBudgetExceeded('Whole operation refused before work: '+','.join(crossing))
        ticket=self.sequence+1
        self._append('reserve',ticket=ticket,operation=operation,costs=costs)
        return ticket

    def assert_within(self,ticket,actual):
        self._check();integer(ticket,'ticket')
        if ticket not in self.tickets or self.tickets[ticket]['status']!='open':raise ValueError('Unknown/finalized ticket')
        actual=_vector(actual,self.limits)
        if any(actual[key]>self.tickets[ticket]['costs'][key] for key in self.limits):
            raise WorkBudgetExceeded('Known next call exceeds its whole-operation reservation')
        return actual

    def complete(self,ticket,actual):
        actual=self.assert_within(ticket,actual)
        self._append('complete',ticket=ticket,actual=actual)

    def fail(self,ticket,error,known_actual=None,attempted=None):
        actual=self.assert_within(ticket,{} if known_actual is None else known_actual)
        attempted=self.assert_within(ticket,self.tickets[ticket]['costs'] if attempted is None else attempted)
        _text(error,'error',512);self._append('fail',ticket=ticket,actual=actual,attempted=attempted,error=error)

    def snapshot(self):
        self._check();return deepcopy(self._summary())

    def validate_snapshot(self,state):
        self._check()
        if type(state) is not dict or type(state.get('sequence')) is not int or state['sequence'] not in self.prefixes:
            raise ValueError('Unknown work snapshot prefix')
        # Exact integer/type gates precede Python equality (True must not alias1).
        for key in ('sequence','max_bytes'):
            integer(state.get(key),key)
        for key in ('reserved','completed','known_partial','attempted_upper','uncertain_upper','limits'):
            value=state.get(key)
            if type(value) is not dict or set(value)!=set(self.limits):raise ValueError('Changed work snapshot dimensions')
            for dimension,amount in value.items():integer(amount,dimension)
        for key in ('open_tickets','failed_tickets','file_identity'):
            if type(state.get(key)) is not list or any(type(i) is not int or i<0 for i in state[key]):raise ValueError('Changed work snapshot IDs')
        if state!=self.prefixes[state['sequence']]:raise ValueError('Snapshot ledger identity/limits/prefix counters changed')
        self.bound=True
        return self.snapshot()  # includes all later attempted/uncertain work

    def close(self):
        if not self.closed:
            self.closed=True
            try:fcntl.flock(self.fd,fcntl.LOCK_UN)
            finally:
                try:os.close(self.fd)
                finally:
                    for descriptor,_ in self.parents:os.close(descriptor)

    def __enter__(self):return self
    def __exit__(self,*_):self.close()
