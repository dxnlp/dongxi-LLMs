"""Cooperative pathname byte/entry reservations, not a filesystem quota.

Only direct files in one private owned root are covered. A single advisory-locked
writer journals reservations before creation. Arbitrary library/child writes and
hostile same-UID actors require a separately verified physical backend.
"""
from __future__ import annotations

from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from uuid import uuid4

JOURNAL = ".artifact-ledger.jsonl"
MAX_JOURNAL_BYTES = 16 * 1024**2
IDENTITY_FIELDS = {"schema", "campaign_id", "ledger_id", "root_device", "root_inode",
                   "max_bytes", "max_entries", "journal_max_bytes"}


def _integer(value, name, minimum=0):
    if type(value) is not int or not minimum <= value <= 2**63-1:
        raise ValueError(f"Invalid {name}")


def _text(value, name, maximum=200):
    if type(value) is not str or not value or len(value) > maximum or any(ord(c)<32 or ord(c)==127 for c in value):
        raise ValueError(f"Invalid {name}")


def _name(value):
    _text(value, "artifact name")
    if value in (".", "..", JOURNAL) or "/" in value or "\\" in value:
        raise ValueError("Only direct private-root artifact names are supported")
    return value


def _sha(value):
    if type(value) is not str or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("Invalid artifact/journal digest")


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash(value):
    return hashlib.sha256(_encode(value)).hexdigest()


def _object(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError("Duplicate journal key")
        result[key]=value
    return result


def _directory(path):
    """Open each ancestor without following symlinks, then retain the root fd."""
    if ".." in Path(path).parts:raise ValueError("Ambiguous parent traversal is not supported")
    path=Path(os.path.abspath(path))
    fd=os.open("/",os.O_RDONLY|os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            next_fd=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
            os.close(fd);fd=next_fd
        return path,fd
    except BaseException:
        os.close(fd);raise


class ArtifactBudget:
    """One locked, persistent cooperative ledger with immutable capacities."""
    @classmethod
    def create(cls, root, *, campaign_id, max_bytes, max_entries,
               journal_max_bytes=32768, require_physical_backend=False):
        if type(require_physical_backend) is not bool:
            raise ValueError("Physical-backend requirement must be a boolean")
        if require_physical_backend:
            raise RuntimeError("No verified physical aggregate quota backend is available")
        _text(campaign_id,"campaign_id",128)
        _integer(max_bytes,"max_bytes",1);_integer(max_entries,"max_entries",1)
        _integer(journal_max_bytes,"journal_max_bytes",4096)
        if journal_max_bytes > MAX_JOURNAL_BYTES or max_bytes < journal_max_bytes:
            raise ValueError("Journal capacity must fit the immutable artifact cap")
        if ".." in Path(root).parts:raise ValueError("Ambiguous parent traversal is not supported")
        root=Path(os.path.abspath(root));parent,fd=_directory(root.parent)
        try:
            os.mkdir(root.name,0o700,dir_fd=fd);os.fsync(fd)
        finally:os.close(fd)
        value=cls._open(root,create=True)
        info=os.fstat(value.directory_fd)
        value._identity=dict(schema="cooperative-artifact-budget-v1",campaign_id=campaign_id,
            ledger_id=uuid4().hex,root_device=info.st_dev,root_inode=info.st_ino,
            max_bytes=max_bytes,max_entries=max_entries,journal_max_bytes=journal_max_bytes)
        try:
            value._append(dict(kind="header",identity=value.identity))
            os.fsync(value.directory_fd)
            value._inventory()
            return value
        except BaseException:
            value.close();raise

    @classmethod
    def restore(cls, root, *, expected_receipt, require_physical_backend=False):
        if type(require_physical_backend) is not bool or require_physical_backend:
            raise RuntimeError("No verified physical aggregate quota backend is available")
        if type(expected_receipt) is not dict or set(expected_receipt)!={"identity","sequence","head_sha256","journal_bytes"}:
            raise ValueError("Independent artifact identity and journal-prefix receipt required")
        identity=expected_receipt['identity'];cls._validate_identity(identity)
        _integer(expected_receipt['sequence'],'sequence');_integer(expected_receipt['journal_bytes'],'journal_bytes',1)
        _sha(expected_receipt['head_sha256'])
        value=cls._open(root,create=False)
        value._identity=deepcopy(identity)
        try:
            value._check_root()
            size=os.fstat(value.journal_fd).st_size
            if not expected_receipt['journal_bytes']<=size<=identity['journal_max_bytes']:
                raise ValueError("Artifact journal rollback/size limit")
            raw=os.pread(value.journal_fd,size,0)
            if len(raw)!=size or not raw.endswith(b"\n"):
                raise ValueError("Incomplete artifact journal tail retained; no automatic repair")
            prefix_found=False;offset=0
            for line in raw.splitlines(keepends=True):
                if len(line)>65536:raise ValueError("Artifact journal row exceeds bound")
                try:row=json.loads(line,object_pairs_hook=_object)
                except (UnicodeError,json.JSONDecodeError,RecursionError) as error:
                    raise ValueError("Malformed artifact journal") from error
                value._replay(row);offset+=len(line)
                if value.sequence==expected_receipt['sequence']:
                    prefix_found=(value.head==expected_receipt['head_sha256'] and offset==expected_receipt['journal_bytes'])
            if not prefix_found:raise ValueError("Independent artifact journal prefix differs")
            value._journal_bytes=size;value._journal_sha256=hashlib.sha256(raw).hexdigest()
            value._inventory()
            return value
        except BaseException:
            value.close();raise

    @staticmethod
    def _validate_identity(identity):
        if type(identity) is not dict or set(identity)!=IDENTITY_FIELDS or identity['schema']!='cooperative-artifact-budget-v1':
            raise ValueError("Unknown artifact identity schema")
        _text(identity['campaign_id'],'campaign_id',128)
        if type(identity['ledger_id']) is not str or re.fullmatch('[0-9a-f]{32}',identity['ledger_id']) is None:
            raise ValueError("Invalid ledger identity")
        for key in ('root_device','root_inode','max_bytes','max_entries','journal_max_bytes'):
            _integer(identity[key],key,0 if key=='root_device' else 1)
        if not 4096<=identity['journal_max_bytes']<=min(identity['max_bytes'],MAX_JOURNAL_BYTES):
            raise ValueError("Invalid fixed journal envelope")

    @classmethod
    def _open(cls,root,*,create):
        value=cls();value.root,value.directory_fd=_directory(root)
        value.journal_fd=None;value.closed=False;value.poisoned=False
        value._files={};value._operations=set();value.sequence=-1;value.head='0'*64
        value._journal_bytes=None;value._journal_sha256=None
        try:
            info=os.fstat(value.directory_fd)
            if info.st_uid!=os.getuid() or info.st_mode&0o077:
                raise ValueError("Artifact root must be private and owned by this UID")
            flags=os.O_RDWR|os.O_NOFOLLOW|os.O_NONBLOCK
            if create:flags|=os.O_CREAT|os.O_EXCL
            value.journal_fd=os.open(JOURNAL,flags,0o600,dir_fd=value.directory_fd)
            info=os.fstat(value.journal_fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or info.st_mode&0o077 or info.st_nlink!=1:
                raise ValueError("Artifact journal must be one private owned regular file")
            fcntl.flock(value.journal_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            return value
        except BaseException:
            value.close();raise

    def close(self):
        if getattr(self,'closed',False):return
        self.closed=True
        if self.journal_fd is not None:os.close(self.journal_fd)
        os.close(self.directory_fd)

    def __enter__(self):return self
    def __exit__(self,*args):self.close()

    @property
    def identity(self):return deepcopy(self._identity)

    @property
    def files(self):return deepcopy(self._files)

    def _check_root(self):
        if self.closed or self.poisoned:raise ValueError("Closed or interrupted artifact journal")
        _,fd=_directory(self.root)
        try:info=os.fstat(fd)
        finally:os.close(fd)
        if (info.st_dev,info.st_ino)!=(self.identity['root_device'],self.identity['root_inode']) or info.st_uid!=os.getuid() or info.st_mode&0o077:
            raise ValueError("Artifact root identity/ownership changed")
        journal=self._stat(JOURNAL);opened=os.fstat(self.journal_fd)
        if (journal is None or (journal.st_dev,journal.st_ino)!=(opened.st_dev,opened.st_ino)
                or not stat.S_ISREG(opened.st_mode) or opened.st_uid!=os.getuid()
                or opened.st_mode&0o077 or opened.st_nlink!=1):
            raise ValueError("Artifact journal path identity changed")
        if self._journal_bytes is not None:
            if opened.st_size!=self._journal_bytes:
                raise ValueError("Artifact journal changed outside its locked writer")
            raw=os.pread(self.journal_fd,self._journal_bytes,0)
            if len(raw)!=self._journal_bytes or hashlib.sha256(raw).hexdigest()!=self._journal_sha256:
                raise ValueError("Artifact journal bytes changed outside its locked writer")

    def name_for(self,path):
        self._check_root()
        if ".." in Path(path).parts:raise ValueError("Ambiguous parent traversal is not supported")
        path=Path(os.path.abspath(path))
        if path.parent!=self.root:raise ValueError("Artifact path is outside the flat private root")
        return _name(path.name)

    def _stat(self,name):
        try:return os.stat(name,dir_fd=self.directory_fd,follow_symlinks=False)
        except FileNotFoundError:return None

    def _digest_file(self,name):
        fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=self.directory_fd)
        digest=hashlib.sha256()
        try:
            info=os.fstat(fd);limit=self._files[name]['limit']
            if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or info.st_mode&0o077 or info.st_size>limit:
                raise ValueError("Unsafe or over-cap artifact before hashing")
            total=0
            while chunk:=os.read(fd,min(1048576,limit-total+1)):
                total+=len(chunk)
                if total>limit:raise ValueError("Artifact grew beyond its reserved capacity")
                digest.update(chunk)
        finally:os.close(fd)
        return digest.hexdigest()

    def _inventory(self):
        self._check_root();names=os.listdir(self.directory_fd);inodes={};actual_bytes=0
        for name in names:
            info=self._stat(name)
            if info is None:raise ValueError("Artifact inventory changed during inspection")
            if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or info.st_mode&0o077:
                raise ValueError("Unsafe artifact type/ownership/permissions")
            if name==JOURNAL:
                limit=self.identity['journal_max_bytes']
            else:
                record=self._files.get(name)
                if record is None or record['state'] in ('reserved','released'):
                    raise ValueError("Untracked artifact entry")
                limit=record['limit']
                if info.st_size>limit:raise ValueError("Artifact exceeds its reserved capacity")
                if record['state'] in ('sealed','removing') and self._digest_file(name)!=record['sha256']:
                    raise ValueError("Sealed artifact bytes changed")
                if record['state']=='linked':
                    source=self._stat(record['source'])
                    if source is None or (source.st_dev,source.st_ino)!=(info.st_dev,info.st_ino):
                        raise ValueError("Published artifact link identity changed")
            if info.st_size>limit:raise ValueError("Artifact exceeds its reserved capacity")
            actual_bytes+=info.st_size
            key=(info.st_dev,info.st_ino);inodes.setdefault(key,[]).append(info)
        for items in inodes.values():
            if any(item.st_nlink!=len(items) for item in items):
                raise ValueError("Artifact has unowned external hard links")
        for name,record in self._files.items():
            if record['state']=='sealed' and name not in names:
                raise ValueError("Sealed artifact is missing")
        return dict(actual_pathname_bytes=actual_bytes,actual_entries=len(names))

    def _event(self,event):
        # Validate on a copy before any irreversible journal write.
        old_files,old_ops=self._files,self._operations
        self._files=deepcopy(old_files);self._operations=set(old_ops)
        try:self._apply(event)
        finally:self._files,self._operations=old_files,old_ops
        self._append(event)

    def _append(self,event):
        self._check_root()
        row=dict(sequence=self.sequence+1,previous_sha256=self.head,event=event)
        row['sha256']=_hash(row);encoded=_encode(row)+b'\n'
        if len(encoded)>65536 or os.fstat(self.journal_fd).st_size+len(encoded)>self.identity['journal_max_bytes']:
            raise ValueError("Fixed artifact journal capacity exhausted")
        try:
            os.lseek(self.journal_fd,0,os.SEEK_END)
            view=memoryview(encoded)
            while view:
                count=os.write(self.journal_fd,view)
                if count<=0:raise OSError("Incomplete artifact journal write")
                view=view[count:]
            os.fsync(self.journal_fd)
            self._replay(row)
            self._journal_bytes=os.fstat(self.journal_fd).st_size
            self._journal_sha256=hashlib.sha256(os.pread(self.journal_fd,self._journal_bytes,0)).hexdigest()
        except BaseException:
            self.poisoned=True;raise

    def _replay(self,row):
        if type(row) is not dict or set(row)!={'sequence','previous_sha256','event','sha256'}:
            raise ValueError("Unknown artifact journal record")
        _integer(row['sequence'],'sequence');_sha(row['sha256']);_sha(row['previous_sha256'])
        if row['sequence']!=self.sequence+1 or row['previous_sha256']!=self.head or row['sha256']!=_hash({k:v for k,v in row.items() if k!='sha256'}):
            raise ValueError("Artifact journal hash/sequence chain changed")
        self._apply(row['event']);self.sequence=row['sequence'];self.head=row['sha256']

    def _apply(self,event):
        if type(event) is not dict or type(event.get('kind')) is not str:raise ValueError("Invalid artifact event")
        kind=event['kind']
        if kind=='header':
            if self.sequence!=-1 or set(event)!={'kind','identity'} or event['identity']!=self.identity:
                raise ValueError("Artifact immutable header differs")
            return
        if self.sequence<0:raise ValueError("Artifact journal is missing its header")
        if kind=='reserve':
            if set(event)!={'kind','operation','files'}:raise ValueError("Invalid reservation event")
            _text(event['operation'],'operation',128)
            files=event['files']
            if event['operation'] in self._operations or type(files) is not dict or not files:
                raise ValueError("Duplicate/empty artifact reservation")
            for name,limit in files.items():
                _name(name);_integer(limit,'artifact capacity')
                if name in self._files and self._files[name]['state']!='released':raise ValueError("Artifact name already reserved")
            used=self.identity['journal_max_bytes']+sum(r['limit'] for r in self._files.values() if r['state']!='released')
            count=1+sum(r['state']!='released' for r in self._files.values())
            if used+sum(files.values())>self.identity['max_bytes'] or count+len(files)>self.identity['max_entries']:
                raise ValueError("Whole artifact bundle exceeds immutable byte/entry cap")
            for name,limit in files.items():self._files[name]=dict(limit=limit,state='reserved',operation=event['operation'])
            self._operations.add(event['operation']);return
        if kind=='link':
            if set(event)!={'kind','source','target'}:raise ValueError("Invalid artifact link event")
            source,target=event['source'],event['target'];_name(source);_name(target)
            if self._files.get(source,{}).get('state')!='sealed' or self._files.get(target,{}).get('state')!='reserved':
                raise ValueError("Unreserved artifact publication")
            if self._files[source]['limit']>self._files[target]['limit']:raise ValueError("Published link exceeds capacity")
            self._files[target].update(state='linked',source=source);return
        if kind not in ('open','seal','remove','release'):raise ValueError("Unknown artifact event kind")
        expected={'kind','name','size','sha256'} if kind=='seal' else {'kind','name'}
        if set(event)!=expected:raise ValueError("Invalid artifact lifecycle event")
        name=_name(event['name']);record=self._files.get(name)
        if record is None:raise ValueError("Missing artifact reservation")
        if kind=='open':
            if record['state']!='reserved':raise ValueError("Artifact cannot be reopened")
            record['state']='opened'
        elif kind=='seal':
            _integer(event['size'],'sealed size');_sha(event['sha256'])
            if record['state'] not in ('opened','linked') or event['size']>record['limit']:
                raise ValueError("Invalid artifact seal")
            record.update(state='sealed',limit=event['size'],sha256=event['sha256'])
        elif kind=='remove':
            if record['state']!='sealed':raise ValueError("Only sealed redundant staging can be removed")
            record['state']='removing'
        else:
            if record['state'] not in ('reserved','opened','linked','removing'):raise ValueError("Cannot release retained sealed artifact")
            record['state']='released';record['limit']=0

    def reserve_bundle(self,operation,files):
        self._inventory()
        if type(files) is not dict:raise ValueError("Reservation needs a name/capacity mapping")
        normalized={_name(name):limit for name,limit in files.items()}
        for name in normalized:
            if self._stat(name) is not None:raise FileExistsError(self.root/name)
        self._event(dict(kind='reserve',operation=operation,files=normalized))
        return Reservation(self,operation,tuple(normalized))

    def receipt(self):
        self._inventory()
        return dict(identity=deepcopy(self.identity),sequence=self.sequence,head_sha256=self.head,
                    journal_bytes=os.fstat(self.journal_fd).st_size)

    def validate_receipt(self, expected_receipt):
        """Trust an old prefix without rewinding this locked ledger's later charges.

        Restore already replayed the full chain; a live writer retains its exact
        whole-journal digest. Validate a prefix at a complete row boundary, not
        just the latest head, without reopening the advisory lock or mutating state.
        """
        if type(expected_receipt) is not dict or set(expected_receipt)!={"identity","sequence","head_sha256","journal_bytes"}:
            raise ValueError("Independent artifact identity and journal-prefix receipt required")
        self._validate_identity(expected_receipt['identity'])
        _integer(expected_receipt['sequence'],'sequence')
        _integer(expected_receipt['journal_bytes'],'journal_bytes',1)
        _sha(expected_receipt['head_sha256'])
        self._inventory()
        if expected_receipt['identity']!=self.identity or expected_receipt['journal_bytes']>self._journal_bytes:
            raise ValueError("Active artifact ledger identity/capacity or prefix differs")
        raw=os.pread(self.journal_fd,expected_receipt['journal_bytes'],0)
        if len(raw)!=expected_receipt['journal_bytes'] or not raw.endswith(b'\n'):
            raise ValueError("Artifact receipt is not a complete retained journal prefix")
        previous='0'*64
        for sequence,line in enumerate(raw.splitlines(keepends=True)):
            if len(line)>65536:raise ValueError("Artifact journal row exceeds bound")
            try:row=json.loads(line,object_pairs_hook=_object)
            except (UnicodeError,json.JSONDecodeError,RecursionError) as error:
                raise ValueError("Malformed artifact journal prefix") from error
            if (type(row) is not dict or set(row)!={'sequence','previous_sha256','event','sha256'}
                    or type(row['sequence']) is not int or row['sequence']!=sequence
                    or row['previous_sha256']!=previous
                    or row['sha256']!=_hash({key:value for key,value in row.items() if key!='sha256'})):
                raise ValueError("Artifact journal prefix hash/sequence differs")
            previous=row['sha256']
        if sequence!=expected_receipt['sequence'] or previous!=expected_receipt['head_sha256']:
            raise ValueError("Independent artifact journal prefix differs")

    def usage(self):
        actual=self._inventory()
        return dict(**actual,reserved_pathname_bytes=self.identity['journal_max_bytes']+
                    sum(r['limit'] for r in self._files.values() if r['state']!='released'),
                    reserved_entries=1+sum(r['state']!='released' for r in self._files.values()))


class Reservation:
    def __init__(self,budget,operation,names):self.budget,self.operation,self.names=budget,operation,names

    def _name(self,name):
        name=_name(name);self.budget._check_root()
        if name not in self.names or self.budget.files[name]['operation']!=self.operation:
            raise ValueError("Artifact belongs to another reservation")
        return name

    def writer(self,name):return BoundedArtifactWriter(self,self._name(name))

    def seal(self,name):
        name=self._name(name);info=self.budget._stat(name)
        if info is None or not stat.S_ISREG(info.st_mode) or info.st_size>self.budget.files[name]['limit']:
            raise ValueError("Missing or over-cap regular artifact to seal")
        self.budget._event(dict(kind='seal',name=name,size=info.st_size,sha256=self.budget._digest_file(name)))
        self.budget._inventory()

    def link(self,source,target):
        source,target=self._name(source),self._name(target);self.budget._inventory()
        if self.budget._stat(target) is not None:raise FileExistsError(self.budget.root/target)
        self.budget._event(dict(kind='link',source=source,target=target))
        os.link(source,target,src_dir_fd=self.budget.directory_fd,dst_dir_fd=self.budget.directory_fd,follow_symlinks=False)
        os.fsync(self.budget.directory_fd);self.seal(target)

    def release_absent(self,name):
        name=self._name(name)
        if self.budget._stat(name) is not None:raise ValueError("Retained partial artifact cannot be released")
        self.budget._event(dict(kind='release',name=name))

    def remove_staging(self,name):
        name=self._name(name);self.budget._inventory();info=self.budget._stat(name)
        if info is None or info.st_nlink<2:raise ValueError("Only exact redundant hard-linked staging may be removed")
        self.budget._event(dict(kind='remove',name=name))
        os.unlink(name,dir_fd=self.budget.directory_fd);os.fsync(self.budget.directory_fd)
        self.release_absent(name)


class BoundedArtifactWriter:
    def __init__(self,reservation,name):
        self.reservation,self.name=reservation,name;self.budget=reservation.budget
        self.budget._inventory();self.budget._event(dict(kind='open',name=name))
        fd=os.open(name,os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=self.budget.directory_fd)
        self.handle=os.fdopen(fd,'w+b');self.limit=self.budget.files[name]['limit']

    def write(self,data):
        self.budget._check_root()
        if self.tell()+len(data)>self.limit:raise ValueError("Artifact stream exceeds reserved capacity")
        return self.handle.write(data)

    def seek(self,offset,whence=0):
        self.budget._check_root()
        if type(offset) is not int or type(whence) is not int or whence not in (0,1,2):raise ValueError("Invalid artifact seek")
        if whence==2:self.handle.flush()  # A buffered write changes the actual SEEK_END base.
        end=os.fstat(self.handle.fileno()).st_size
        target=offset+(self.tell() if whence==1 else end if whence==2 else 0)
        if not 0<=target<=self.limit:raise ValueError("Artifact seek exceeds reserved capacity")
        return self.handle.seek(offset,whence)

    def tell(self):return self.handle.tell()
    def fileno(self):return self.handle.fileno()
    def flush(self):self.handle.flush()
    def __enter__(self):return self
    def __exit__(self,kind,*args):
        try:
            if kind is None:self.flush();os.fsync(self.fileno());os.fsync(self.budget.directory_fd)
        finally:self.handle.close()
        if kind is None:self.reservation.seal(self.name)
