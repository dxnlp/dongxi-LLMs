#!/usr/bin/env python3
"""CPU-only immutable evidence snapshot for the fixed native campaign.

Invoke after the owned jobs have ended. No model/tokenizer load, GPU, network,
Git, new scientific gate, resampling, score generation or producer mutation.
Original45 planned rows remain archived unchanged; observations are separate.
Final export hashing is assembler CPU work outside the producer work journals.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import time
import zlib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dongxi_llms.staged_campaign import stage_rows
from dongxi_llms.run_identity import canonical_hash,environment_identity,ARTIFACT_PATTERNS
from dongxi_llms.native_profile_supervisor import _digest,_new_target

ARCHIVE='experiments/reports/2026-10-04-staged-campaign-preparation.json'
ARCHIVED_ROWS_SHA256='4b5b721efbe0d80385f515791aa43bbf8845a6576a9db28c80529b24b6996c72'
PYTHON='/home/dongxi/dgx-spark-dongxi/.venv/bin/python'
LOCK='/home/dongxi/dgx-spark-dongxi/uv.lock'
CACHE_ROOT=Path('/home/dongxi/.cache/huggingface/hub')
JSON_BYTES=64*1024**2
JOURNAL_BYTES=256*1024**2
EXPORT_FILE_BYTES=4*1024**3
# Assembled receipt copies are not a producer journal or a single JSON input.
SNAPSHOT_BYTES=1024**3
GZIP_LEVEL=9
MAX_FILES=6000
SOURCES=('scripts/assemble_native_campaign_evidence.py','tests/test_native_campaign_evidence.py',
    'src/dongxi_llms/staged_campaign.py','src/dongxi_llms/run_identity.py',
    'src/dongxi_llms/native_profile_supervisor.py','src/dongxi_llms/campaign_supervisor.py')
DOCUMENTS=(ARCHIVE,'experiments/specs/2026-10-04-staged-spark-campaign.md',
    'experiments/specs/2026-10-05-spark-campaign-continuation.md',
    'docs/COURSE_IMPROVEMENT_PLAN.md','docs/course_improvements.json',
    'experiments/specs/2026-10-05-story-publication-raters.json')
OPTIONAL_IDS=('assistant-dpo-chosen-nll-extension','assistant-dpo-rehearsal-extension',
    'assistant-1p7-confirmation','reasoning-multi-seed-confirmation')
STORY_RATING_FILES=('rating-codex-blind-reader-a.json','rating-codex-blind-reader-b.json')


def retain(path,value):
    with Path(path).open('x',encoding='utf-8') as handle:
        json.dump(value,handle,indent=2,ensure_ascii=False,allow_nan=False)
        handle.write('\n');handle.flush();os.fsync(handle.fileno())


def _archive_limit(maximum):
    maximum=SNAPSHOT_BYTES if maximum is None else maximum
    if type(maximum) is not int or not 0<maximum<=SNAPSHOT_BYTES:
        raise ValueError('Positive bounded archive-output ceiling required')
    return maximum


class _BoundedArchiveWriter:
    """Admit each output chunk before writing; failed prefixes remain local."""
    def __init__(self,handle,maximum):
        self.handle=handle;self.maximum=maximum;self.written=0
    def write(self,chunk):
        if self.written+len(chunk)>self.maximum:
            raise ValueError('Archive-output ceiling crossed')
        count=self.handle.write(chunk)
        if count!=len(chunk):raise OSError('Incomplete archive-output write')
        self.written+=count;return count
    def flush(self):return self.handle.flush()
    def tell(self):return self.written


def retain_snapshot(path,value,*,maximum=None):
    """Exclusive streaming JSON; the independent archive cap includes newline."""
    maximum=_archive_limit(maximum)
    encoder=json.JSONEncoder(indent=2,ensure_ascii=False,allow_nan=False)
    with Path(path).open('xb') as handle:
        writer=_BoundedArchiveWriter(handle,maximum)
        for chunk in encoder.iterencode(value):writer.write(chunk.encode('utf-8'))
        writer.write(b'\n');writer.flush();os.fsync(handle.fileno())


def compress_snapshot(source,target,*,maximum=None):
    """Stream a same-environment deterministic gzip, never extract or overwrite.

    Both raw bytes read and compressed bytes written have the independent cap.
    A completed companion does not substitute for the two closing byte hashes.
    """
    maximum=_archive_limit(maximum);source=Path(source)
    fd=os.open(source,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        before=os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0<=before.st_size<=maximum:
            raise ValueError('Bounded regular snapshot required')
        with Path(target).open('xb') as handle:
            writer=_BoundedArchiveWriter(handle,maximum);count=0
            with gzip.GzipFile(filename='',mode='wb',fileobj=writer,
                    compresslevel=GZIP_LEVEL,mtime=0) as compressed:
                while chunk:=os.read(fd,1024**2):
                    count+=len(chunk)
                    if count>maximum:raise ValueError('Snapshot read ceiling crossed')
                    compressed.write(chunk)
            after=os.fstat(fd);named=source.lstat()
            identity=lambda item:(item.st_dev,item.st_ino,item.st_size,item.st_mtime_ns,item.st_ctime_ns)
            if count!=before.st_size or identity(before)!=identity(after) or identity(before)!=identity(named):
                raise ValueError('Snapshot bytes/path changed during compression')
            writer.flush();os.fsync(handle.fileno())
    finally:os.close(fd)


def parse_json(raw):
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('Duplicate retained JSON key')
            result[key]=value
        return result
    return json.loads(raw,object_pairs_hook=unique,
        parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite retained JSON')))


def stream_binding(path,maximum=JOURNAL_BYTES):
    """Bound regular bytes, including an empty failed response journal."""
    path=Path(path);fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        before=os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0<=before.st_size<=maximum:
            raise ValueError('Bounded regular evidence/export file required: '+str(path))
        digest=hashlib.sha256();count=0
        while chunk:=os.read(fd,1024**2):
            count+=len(chunk)
            if count>maximum:raise ValueError('Evidence/export hash ceiling crossed')
            digest.update(chunk)
        after=os.fstat(fd)
        if count!=before.st_size or (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):
            raise ValueError('Evidence/export bytes changed during hashing')
    finally:os.close(fd)
    return dict(path=str(path),bytes=count,sha256=digest.hexdigest())


class SavedEvidence:
    """One read/hash per receipt path; shared planned rows never duplicate it."""
    def __init__(self):
        self.bindings={};self.documents={};self.missing=set();self.bytes_hashed=0
        self.directories={};self.cache_links={}
    def bind(self,path,maximum=JOURNAL_BYTES):
        path=Path(path);key=str(path)
        if key in self.bindings:return self.bindings[key]
        if not path.exists() and not path.is_symlink():self.missing.add(key);return None
        if len(self.bindings)>=MAX_FILES:raise ValueError('Closed evidence file-count ceiling crossed')
        value=stream_binding(path,maximum);self.bindings[key]=value;self.bytes_hashed+=value['bytes'];return value
    def read(self,path):
        path=Path(path);key=str(path)
        if key in self.documents:return self.documents[key]
        binding=self.bind(path,JSON_BYTES)
        if binding is None:return None
        try:
            with path.open('rb') as handle:raw=handle.read(JSON_BYTES+1)
            if len(raw)>JSON_BYTES or hashlib.sha256(raw).hexdigest()!=binding['sha256']:
                raise ValueError('Saved JSON bytes changed after stable hash')
            value=parse_json(raw)
        except (OSError,ValueError) as error:
            value=dict(retained_json_error=dict(type=type(error).__name__,message=str(error)),
                scope='Original failed/malformed bytes remain byte-bound; not interpreted as completed evidence')
        self.documents[key]=value;return value
    def directory(self,path):
        path=Path(path);key=str(path)
        if key in self.directories:return self.directories[key]
        if not path.exists():self.missing.add(key);return None
        value=directory_layout(path);self.directories[key]=value;return value
    def artifact(self,path,root):
        """Only pinned HF-cache links may resolve into that model's blobs."""
        path=Path(path);root=Path(root)
        if not path.is_symlink():return self.bind(path,EXPORT_FILE_BYTES)
        cache=CACHE_ROOT
        if not root.is_relative_to(cache) or root.parent.name!='snapshots':
            raise ValueError('Only pinned HF snapshot artifact links may be followed')
        target=path.resolve(strict=True)
        if not target.is_relative_to(root.parent.parent/'blobs'):
            raise ValueError('Pinned cache artifact link leaves its model blobs')
        link=dict(path=str(path),link_target=os.readlink(path),actual_path=str(target))
        self.cache_links[str(path)]=link
        bound=self.bind(target,EXPORT_FILE_BYTES)
        return dict(bound,path=str(path),actual_path=str(target),cache_link_target=link['link_target'])
    def verify(self):
        for path,value in self.bindings.items():
            if stream_binding(path,EXPORT_FILE_BYTES if value['bytes']>JOURNAL_BYTES else JOURNAL_BYTES)!=value:
                raise ValueError('Saved source/input/outcome/export changed: '+path)
        for path in self.missing:
            if Path(path).exists() or Path(path).is_symlink():raise ValueError('Previously missing evidence appeared during assembly: '+path)
        for path,value in self.directories.items():
            if directory_layout(path)!=value:raise ValueError('Retained directory inventory changed: '+path)
        for path,value in self.cache_links.items():
            if (not Path(path).is_symlink() or os.readlink(path)!=value['link_target'] or
                    str(Path(path).resolve(strict=True))!=value['actual_path']):
                raise ValueError('Pinned cache artifact link changed: '+path)


def directory_layout(path):
    """Names/types/sizes retain orphans without reading tensor bodies."""
    path=Path(path)
    if not path.is_dir() or path.is_symlink():raise ValueError('Actual non-symlink evidence directory required')
    result=[]
    for item in sorted(path.iterdir()):
        metadata=item.lstat()
        result.append(dict(name=item.name,kind='link' if item.is_symlink() else ('directory' if stat.S_ISDIR(metadata.st_mode) else 'file'),
            bytes=metadata.st_size if not stat.S_ISDIR(metadata.st_mode) else None))
    if len(result)>MAX_FILES:raise ValueError('Retained directory entry ceiling crossed')
    return result


def original_rows(archive,current):
    rows=archive['campaign']['stage_rows']
    if (len(rows)!=45 or len({row['id'] for row in rows})!=45 or
            canonical_hash(rows)!=ARCHIVED_ROWS_SHA256 or
            {row['id'] for row in rows if row['optional']}!=set(OPTIONAL_IDS)):
        raise ValueError('Original archived45-row comparison contract changed')
    if [row['id'] for row in rows]!=[row['id'] for row in current]:raise ValueError('Original stage identities/order changed')
    changes=[]
    for before,after in zip(rows,current):
        changed={key:dict(original=deepcopy(value),current=deepcopy(after.get(key)))
            for key,value in before.items() if value!=after.get(key)}
        extra=set(after)-set(before)
        if set(changed)-{'known_blocker'} or extra-{'execution_dependency_scopes'}:
            raise ValueError('Original budgets/dependencies/interventions/criteria changed: '+before['id'])
        changes.append(dict(stage_id=before['id'],documentation_changes=changed,
            execution_dependency_scopes=deepcopy(after.get('execution_dependency_scopes'))))
    return deepcopy(rows),changes


def catalog():
    """Closed dated producers; historical runs are retained, not selected by score."""
    entries=[]
    def add(identifier,stages,relative,kind,scope,*,outputs=(),selected=True):
        entries.append(dict(id=identifier,stage_ids=list(stages),path=str(ROOT/'experiments/reports'/relative),
            kind=kind,scope=scope,output_roots=[str(ROOT/'outputs'/path) for path in outputs],selected=selected))
    add('assistant-profile',('assistant-profile-base06',),'2026-10-05-native-base-profile','profile',
        'Actual original20 full-SFT resource/profile evidence; not a400 pilot',outputs=('native-base-profile-20261005-run01',))
    for arm in ('full','lora'):
        add('assistant-'+arm+'-replay',(f'assistant-sft-{arm}-smoke',f'assistant-sft-{arm}-recovery'),
            f'2026-10-05-native-sft-{arm}-replay','recovery',
            'Original20 source invocation and fresh checkpoint10-to20 numerical replay; shared receipt supports smoke/recovery, not two new jobs')
        stem=f'native-sft-{arm}-pilot400-20261005-run-01'
        add('assistant-'+arm+'-pilot',(f'assistant-sft-{arm}-pilot',),stem,'pilot','Actual fresh400 fixed-recipe training',outputs=(stem,))
    stem='native-sft-lora-pilot400-merged-20261005-run-01'
    add('assistant-lora-merge',('assistant-lora-merge',),stem,'merge',
        'Actual accepted LoRA400; CPU FP32 merge/storage, CUDA FP32 numerical verification, bitwise fresh reload',outputs=(stem,))
    for selector,stage in (('base','assistant-frozen-base'),('full400','assistant-sft-full-evaluation'),('lora400-fp32','assistant-sft-lora-evaluation')):
        stem=f'native-assistant-publication-20261005-{selector}-run-01'
        add('assistant-publication-'+selector,(stage,),stem,'evaluation',
            'Original120/40 held-out values; common template/greedy64/BF16, strict whole decoded answers and generic replay separate')
    add('assistant-selected-parent',('assistant-selected-sft-parent',),'native-sft-full-pilot400-20261005-run-01','genealogy',
        'Predeclared accepted full400 final, never selected by publication score')
    for number in (1,2):
        add('assistant-comparison-'+str(number),(),f'native-assistant-comparison-20261005-run-{number:02d}','cpu-comparison',
            'CPU saved-record descriptive120/40 comparison, separate observed60-item dev NLL',selected=number==2)
    for kind,arm in (('dpo','dpo'),('chosen','chosen-sft')):
        for mode in ('replay','pilot'):
            stages=(f'assistant-{arm}-smoke',f'assistant-{arm}-recovery') if mode=='replay' else (f'assistant-{arm}-pilot',)
            stem=f'native-{kind}-{mode}-20261005-run-01'
            roles=('clean','source','resumed') if mode=='replay' else ('pilot',)
            add(kind+'-'+mode+('-historical-01' if kind=='dpo' and mode=='replay' else ''),stages,stem,'recovery' if mode=='replay' else 'pilot',
                'Actual same immutable full400 parent/reference; original location masks, replay2 or fixed100, persistent spending',
                outputs=tuple(stem+'-'+role for role in roles),selected=not(kind=='dpo' and mode=='replay'))
            if kind=='dpo' and mode=='replay':
                stem='native-dpo-replay-20261005-run-02'
                add('dpo-replay-historical-02',stages,stem,'recovery',
                    'Retained fixed CPU8-entry replay2 after run01; clean/source completed, fresh-resume deadline failure with unknown native exit and cleanup failure remain historical; unchanged600s caps/method',
                    outputs=tuple(stem+'-'+role for role in roles),selected=False)
                stem='native-dpo-replay-20261005-run-03'
                add('dpo-replay',stages,stem,'recovery',
                    'Predeclared fresh CPU8-entry replay2 with explicit four-worker complete-file snapshot SHA scheduling; default remains serial. Execution-only configuration, unchanged600s caps/method/science; missing actual receipts remain missing',
                    outputs=tuple(stem+'-'+role for role in roles))
    stem='native-preference-evaluation-20261005-run-01'
    add('preference-comparison',('assistant-preference-comparison',),stem,'evaluation',
        'Original4 validation pairs and independent4 locations/assistant120/annotated reasoning20; common greedy64; no pooled score',outputs=(stem,))
    for arm in ('control','half-lr'):
        for number in (1,2,3):
            if number==1 and arm=='half-lr':continue
            stem=f'native-story-{arm}-recovery-20261005-{number:02d}'
            add(f'story-{arm}-recovery-{number}',(f'story-{arm}-smoke',f'story-{arm}-recovery'),stem,'recovery',
                'Actual fresh clean3/source3/fresh resume1-to3; shared persistent journal; numerical replay is a separate CPU child',
                outputs=(stem,),selected=number==3)
        stem=f'native-story-{arm}-pilot-20261005-01'
        add('story-'+arm+'-pilot',(f'story-{arm}-pilot',),stem,'story-first-tranche',
            'Predeclared first400 tranche of original14000/200-warmup schedule/fullcaps; not full14000 or50M target completion',outputs=(stem,))
    for number in (1,2,3):
        stem=f'native-story-profile-20261005-{number:02d}'
        add('story-profile-'+str(number),('story-profile',),stem,'profile','Actual40 profile; no profile-weight transfer',outputs=(stem,),selected=number==3)
    add('story-publication',('story-paired-comparison',),'native-story-publication-20261005-01','evaluation',
        'Original12 openings ×4 recipes ×5 checkpoints ×2 arms; actual missing later cells remain null',outputs=('native-story-publication-20261005-01',))
    add('story-ratings',('story-paired-comparison',),'native-story-publication-20261005-01/ratings-evaluation-01','cpu-assessment',
        'Actual two fresh independent AI-rater records only; no human-rating/perfect-blinding/model-ID authentication claim')
    add('story-blind-packet',(),'native-story-publication-20261005-01/blind-packet-01','cpu-assessment',
        'Actual packet/receipt provenance, never authored old rubric-control scores')
    for role in ('base-raw','base-chat','instruct-thinking-off','instruct-thinking-on'):
        for cap in (32,128):
            stem=f'native-reasoning-{role}-cap{cap}-20261005-run-01'
            stages=[f'reasoning-baseline-{role}']
            if cap==128 and role in ('base-raw','instruct-thinking-off'):
                stages.append('reasoning-profile-'+('base' if role=='base-raw' else 'instruct'))
            add(f'reasoning-{role}-cap{cap}',stages,stem,'evaluation',
                'Original20 ×4 sampled seeds +greedy; cap32/128 separate. Embedded measured inference/resource observations are not a new distinct profile child',outputs=(stem,))
    for group in (4,8):
        for mode in ('recovery','pilot'):
            stages=(f'reasoning-rlvr-g{group}-smoke',f'reasoning-rlvr-g{group}-recovery') if mode=='recovery' else (f'reasoning-rlvr-g{group}-pilot',)
            stem=f'native-rlvr-g{group}-{mode}-20261005-run-01'
            outputs=(stem+'-source',stem+'-completed-resume',stem+'-pending-resume') if mode=='recovery' else (stem,)
            add(f'rlvr-g{group}-{mode}',stages,stem,'recovery' if mode=='recovery' else 'pilot',
                'Pinned Instruct parent/native interface; actual completed+pending recovery2 or fresh16; G4/G8 equal updates, unequal rollout work',
                outputs=outputs,selected=not (group==4 and mode=='recovery'))
        if group==4:
            stem='native-rlvr-g4-recovery-20261005-run-02'
            add('rlvr-g4-recovery-retry02',('reasoning-rlvr-g4-smoke','reasoning-rlvr-g4-recovery'),stem,'recovery',
                'Explicit single fresh retry after controller lazy-import correction; original failed01 and its physical spending remain retained; unchanged scientific recipe and cleanup/deadline caps',
                outputs=(stem+'-source',stem+'-completed-resume',stem+'-pending-resume'))
        for cap in (32,128):
            stem=f'native-rlvr-evaluation-g{group}-cap{cap}-20261005-run-01'
            add(f'rlvr-g{group}-evaluation-cap{cap}-accepted-default-unrun',('reasoning-g4-g8-comparison',),stem,'evaluation',
                'Original accepted-only run01 route retained separately; not an observed-export launch',outputs=(stem,),selected=False)
            stem=f'native-rlvr-evaluation-g{group}-cap{cap}-20261005-run-02'
            add(f'rlvr-g{group}-evaluation-cap{cap}',('reasoning-g4-g8-comparison',),stem,'evaluation',
                'Explicit run02 consistency-observed actual16 export; original training supervision remains unchanged, including logging failure. Same original20/100 attempts/native Instruct contract; prospective1Hz memory sampling',outputs=(stem,))
    for cap in (32,128):
        add(f'rlvr-common20-cap{cap}-accepted-default-unrun',('reasoning-g4-g8-comparison',),f'native-rlvr-common20-cap{cap}-20261005-run-01.json','cpu-comparison',
            'Original accepted-only run01 comparison route retained, not silently replaced',selected=False)
        add(f'rlvr-common20-cap{cap}',('reasoning-g4-g8-comparison',),f'native-rlvr-common20-cap{cap}-20261005-run-02.json','cpu-comparison',
            'Explicit observed-export run02; actual unchanged Instruct/G4/G8 aligned by seed; original pilot supervision retained; known overlaps excluded from held-out headlines; caps never pooled')
    for stem in ('2026-10-05-native-lora-merge-reload','2026-10-05-native-lora-fp32-merge-reload'):
        add('historical-'+stem,(),stem,'historical-merge','Historical short-run BF16 failure/FP32 check, not selected400 export',selected=False)
    for kind in ('chosen','dpo'):
        for number in (0,99):
            add(f'historical-{kind}-prepare-{number}',(),f'native-{kind}-replay-20261005-run-{number:02d}',
                'historical-preparation','Retained preparation/refusal controls, not executed native child evidence',selected=False)
    return entries


def collect_entry(entry,saved):
    path=Path(entry['path']);refs=[];terminal=[];cpu=[];launches=[];failures=[]
    if path.suffix=='.json':
        if saved.read(path) is not None:refs.append(str(path))
    elif path.is_dir():
        saved.directory(path)
        candidates=sorted(set(path.glob('*.json'))|set(path.glob('generation/*.json')))
        for receipt in candidates:
            saved.read(receipt);refs.append(str(receipt))
            if 'launch' in receipt.name:launches.append(str(receipt))
            if 'failure' in receipt.name:failures.append(str(receipt))
            if receipt.name=='cpu-comparison-process.json':cpu.append(str(receipt))
        for directory in sorted(p for p in path.iterdir() if p.is_dir() and 'supervision' in p.name):
            saved.directory(directory)
            receipt=directory/'result.json';value=saved.read(receipt)
            if value is not None:refs.append(str(receipt))
            returned=path/('returned-'+directory.name+'.json')
            # The returned value includes the final writer ACK/error; result.json
            # may have been written before that acknowledgement failed.
            if str(returned) in saved.documents:terminal.append(str(returned))
            elif value is not None:terminal.append(str(receipt))
            for name in ('stdout.txt','stderr.txt','events.jsonl'):
                if (directory/name).exists():saved.bind(directory/name)
        for journal in path.glob('*.jsonl'):saved.bind(journal)
        if (path/'generation').is_dir():
            saved.directory(path/'generation')
            for journal in (path/'generation').glob('*.jsonl'):saved.bind(journal)
    else:saved.missing.add(str(path))
    output_roots=set(entry['output_roots'])
    # Legacy replay paths and physical journals are fixed in actual child argv.
    for receipt in terminal:
        value=saved.documents[receipt]
        if not isinstance(value,dict):continue
        command=value.get('child_command',[])
        for index,argument in enumerate(command[:-1]):
            if argument not in ('--output','--work-journal','--snapshot-io-ledger','--snapshot-artifact-root'):continue
            target=Path(command[index+1])
            if not target.is_absolute() or not target.is_relative_to(ROOT/'outputs'):continue
            if argument in ('--output','--snapshot-artifact-root'):output_roots.add(str(target))
            else:
                saved.bind(target);output_roots.add(str(target.parent))
    for output in output_roots:
        root=Path(output)
        if not root.is_dir():saved.missing.add(str(root));continue
        saved.directory(root)
        for name in ('checkpoints','snapshots'):
            if (root/name).is_dir():saved.directory(root/name)
        # Metadata/raw journals only, not tensor checkpoints or caches.
        roots=[root]+[p for p in root.iterdir() if p.is_dir() and p.name not in ('policy','checkpoints','snapshots')]
        if entry['id']=='preference-comparison':
            # Closed actual producer layout; never recurse into tensor/cache trees.
            for role in ('unchanged','chosen100','dpo100'):
                for panel in ('location','assistant','reasoning'):
                    folder=root/role/panel
                    if folder.is_dir():roots.append(folder)
                    else:saved.missing.add(str(folder))
        for folder in roots:
            saved.directory(folder)
            for receipt in sorted(folder.glob('*.json')):
                saved.read(receipt);refs.append(str(receipt))
                if 'failure' in receipt.name:failures.append(str(receipt))
            for journal in sorted(folder.glob('*.jsonl')):saved.bind(journal)
    if not refs:return None
    return dict(**deepcopy(entry),receipt_paths=sorted(set(refs)),terminal_supervisor_paths=sorted(set(terminal)),
        actual_observed_output_roots=sorted(output_roots),cpu_process_paths=sorted(set(cpu)),
        launch_paths=sorted(set(launches)),failure_paths=sorted(set(failures)),
        launches_without_terminal_receipt_count=max(0,len(set(launches))-len(set(terminal))))


def native_child(value):
    if not isinstance(value,dict):return False
    command=value.get('child_command',[])
    if '--comparison-child' in command or '--verify-child' in command:return False
    scripts={Path(part).name for part in command if isinstance(part,str)}
    recognized=bool(scripts&{'run_deterministic_story_child.py','run_chapter09_spark_sft.py',
        'run_chapter11_spark_dpo.py','run_native_dpo_cpu8_child.py','run_native_chosen_stages.py','generate_reasoning_records.py',
        'run_native_lora_merge.py','run_native_story_evaluation.py','run_native_reasoning_baselines.py',
        'run_native_rlvr_evaluation.py','run_native_preference_evaluation.py'}) or 'dongxi_llms.qwen_rlvr_lab' in command
    recognized=recognized or any(str(ROOT/'experiments/reports'/stem/'merge-child.py') in command for stem in
        ('2026-10-05-native-lora-merge-reload','2026-10-05-native-lora-fp32-merge-reload'))
    exit_observation=(type(value.get('actual_exit_code')) is int or
        (value.get('status')=='failed' and 'actual_exit_code' in value and value['actual_exit_code'] is None))
    return (value.get('actual_native_profile_executed') is True and type(value.get('child_pid')) is int
        and value['child_pid']>0 and exit_observation and recognized)


def child_accounting(observations,documents):
    model={};other={};cpu=set();outer=set()
    for observation in observations.values():
        if observation is None:continue
        if observation['kind']=='cpu-comparison':
            cpu.update(path for path in observation['receipt_paths'] if Path(path).name in ('comparison.json',) or 'native-rlvr-common20-' in path)
        cpu.update(observation['cpu_process_paths'])
        if observation['launch_paths'] or observation['terminal_supervisor_paths']:outer.add(observation['path'])
        for path in observation['terminal_supervisor_paths']:
            value=documents[path];target=model if native_child(value) else other
            if not isinstance(value,dict):value={}
            if '--comparison-child' in value.get('child_command',[]) or '--verify-child' in value.get('child_command',[]):cpu.add(path)
            target[path]=dict(receipt_path=path,stage_observation_ids=[],status=value.get('status'),
                actual_exit_code=value.get('actual_exit_code'),child_pid=value.get('child_pid'),
                command=value.get('child_command'),child_seconds=value.get('child_seconds'),
                seconds=value.get('seconds'),stop_reason=value.get('stop_reason'),cleanup_errors=deepcopy(value.get('cleanup_errors')),
                journal_error=deepcopy(value.get('journal_error')),final_record_retained=value.get('final_record_retained'),
                actual_exit_observation='unknown-not-observed' if value.get('actual_exit_code') is None else 'reported-native-exit',
                minimum_sampled_available_bytes=value.get('minimum_sampled_available_bytes'))
    for child in model.values():
        child['stage_observation_ids']=[key for key,value in observations.items() if value is not None and child['receipt_path'] in value['terminal_supervisor_paths']]
    return dict(distinct_native_model_children=len(model),native_model_children=list(model.values()),
        unclassified_or_non_native_supervisors=list(other.values()),
        native_child_statuses=dict(Counter(value['status'] for value in model.values())),
        distinct_cpu_comparison_receipts=len(cpu),cpu_comparison_receipt_paths=sorted(cpu),
        outer_adapter_receipt_roots=sorted(outer),outer_adapter_count=len(outer),
        scope='Counts distinct terminal supervisor receipt paths of actually launched model-directed native invocations, including failed children with an unknown exit; not proof every child reached a GPU forward. Unknown exits/cleanup failures are not replaced by outer-adapter exits. Reused planned-row/acceptance/returned references do not add jobs. CPU comparisons and outer adapters are separate')


def accepted_document(observation,documents):
    if observation is None:return None,None
    names=('acceptance-retry02.json','acceptance.json','coverage.json','comparison.json','failure.json','report.json')
    for name in names:
        path=str(Path(observation['path'])/name)
        if path in documents:return path,documents[path]
    if Path(observation['path']).suffix=='.json':return observation['path'],documents[observation['path']]
    return None,None


def outcome_status(value):
    return value.get('status') if isinstance(value,dict) else None


def stage_observations(rows,observations,documents):
    result={}
    for row in rows:
        attached=[key for key,value in observations.items() if value is not None and value['selected'] and row['id'] in value['stage_ids']]
        if not attached:
            if row['id']=='capstone-branch-comparison':
                relevant=('story-publication','story-ratings','assistant-comparison-2','preference-comparison',
                    'rlvr-common20-cap32','rlvr-common20-cap128')
                available=[key for key in relevant if observations.get(key) is not None]
                result[row['id']]=None if not available else dict(observation_ids=available,
                    missing_observation_ids=[key for key in relevant if key not in available],
                    interpretation='CPU assembly reuses separate branch evidence; no new model child, pooled capability score, package completion assertion or maximum-horizon success')
            else:result[row['id']]=None
            continue
        statuses={key:outcome_status(accepted_document(observations[key],documents)[1]) for key in attached}
        result[row['id']]=dict(observation_ids=attached,retained_outcome_statuses=statuses,
            interpretation='Scoped observations only; archived proposed maxima/gates/dependencies are not marked complete')
        if row['stage']=='pilot' and row['branch']=='story':
            result[row['id']].update(actual_scope='first400-tranche',full14000_schedule_completion=None,
                original_maximum_updates=row['proposed_budget']['maximum_updates'],later_checkpoint_observations={str(step):None for step in (4000,8000,14000)})
            acceptance=accepted_document(observations[attached[0]],documents)[1] or {}
            if not isinstance(acceptance,dict):acceptance={}
            result[row['id']]['reported_completion_level']=acceptance.get('completion_level')
            result[row['id']]['reported_full_schedule_complete']=acceptance.get('full_schedule_complete')
    return result


def inventory_candidates(observations,documents):
    candidates={}
    for identifier in ('assistant-full-pilot','assistant-lora-pilot','assistant-lora-merge','dpo-pilot','chosen-pilot','rlvr-g4-pilot','rlvr-g8-pilot'):
        observation=observations.get(identifier);receipt,value=accepted_document(observation,documents)
        if isinstance(value,dict) and value.get('status')=='passed':
            exported=value.get('exported_policy') or value.get('exports',{}).get('pilot')
            if exported:candidates[identifier]=dict(path=exported['path'],files=exported['files'],acceptance_receipt_path=receipt)
        if identifier.startswith('rlvr-') and observation is not None:
            path=str(Path(observation['path'])/'export-observation.json');observed=documents.get(path)
            if observed is not None:
                if (not isinstance(observed,dict) or observed.get('schema')!='dongxi-native-rlvr-observed-export-v1'
                        or observed.get('status')!='export-consistent-supervision-not-reclassified'
                        or observed.get('stage')!='pilot' or observed.get('group')!=int(identifier[6])
                        or observed.get('run_id')!='run-01'
                        or observed.get('observation_sha256')!=canonical_hash({key:item for key,item in observed.items() if key!='observation_sha256'})
                        or observed.get('checks')!={'pilot_completed_horizon':True,'original_prompt_and_source_contract':True,'no_open_or_failed_journal_tickets':True}
                        or observed.get('before_bindings')!=observed.get('after_bindings')
                        or observed.get('export_binding',{}).get('completed_updates')!=16):
                    raise ValueError('Explicit actual16 export-consistency observation required')
                exported=observed['export_binding']
                candidates[identifier]=dict(path=exported['path'],files=exported['files'],
                    export_observation_receipt_path=path,
                    training_supervision_status=observed.get('supervision_observation',{}).get('status'),
                    inventory_scope='Observed completed export; original pilot acceptance/supervision is not reclassified')
    for identifier in ('assistant-full-pilot','reasoning-instruct-thinking-off-cap32'):
        observation=observations.get(identifier)
        if observation is None:continue
        path=str(Path(observation['path'])/'preparation.json');value=documents.get(path,{})
        model=(value.get('local_base_binding') or value.get('local_model_binding')) if isinstance(value,dict) else None
        if model:candidates['base-parent' if identifier=='assistant-full-pilot' else 'instruct-parent']=dict(
            path=model['path'],files=model['files'],preparation_receipt_path=path)
    return candidates


def bind_inventories(candidates,saved):
    inventories={};seen={}
    for identifier,value in candidates.items():
        root=Path(value['path'])
        if not root.is_absolute() or (not root.is_relative_to(ROOT/'outputs') and not root.is_relative_to(CACHE_ROOT)):
            raise ValueError('Selected export must stay in actual output/pinned-cache roots')
        expected=value['files']
        if not expected or any(not re.fullmatch(r'[0-9a-f]{64}',sha) for sha in expected.values()):raise ValueError('Recorded actual export byte inventory required')
        if str(root) in seen and expected!=seen[str(root)]:raise ValueError('Shared selected output inventories conflict')
        seen[str(root)]=expected
        saved.directory(root)
        files={str(path.relative_to(root)) for pattern in ARTIFACT_PATTERNS for path in root.glob(pattern) if path.is_file()}
        if files!=set(expected):raise ValueError('Actual selected output file inventory changed: '+str(root))
        bound={}
        for name,sha in expected.items():
            path=root/name
            if not path.is_relative_to(root) or Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Recorded export path escapes selected output')
            binding=saved.artifact(path,root)
            if binding is None or binding['sha256']!=sha:raise ValueError('Actual selected output bytes differ from accepted inventory: '+str(path))
            bound[name]=binding
        genealogy=saved.read(root/'course-genealogy.json') if 'course-genealogy.json' in expected else None
        inventories[identifier]=dict(**value,actual_file_bindings=bound,genealogy=genealogy,
            scope='Fresh CPU streamed byte checks; no model deserialization. Hash work is outside producer journals')
    return inventories


def bind_story_checkpoints(saved):
    result={}
    for arm in ('control','half-lr'):
        for update in (0,400,4000,8000,14000):
            key=f'{arm}-update-{update:06d}'
            payload=ROOT/'outputs'/f'native-story-{arm}-pilot-20261005-01'/'pilot'/f'update-{update:06d}.pt'
            receipt=Path(str(payload)+'.work.json');result[key]=None
            if not receipt.exists():continue
            value=saved.read(receipt)
            if not isinstance(value,dict) or value.get('completed_updates')!=update:
                raise ValueError('Actual story independent completed checkpoint receipt required')
            bound=saved.bind(payload,EXPORT_FILE_BYTES)
            if bound is None or (bound['sha256'],bound['bytes'])!=(value.get('payload_sha256'),value.get('payload_bytes')):
                raise ValueError('Actual story payload differs from independent completed receipt')
            result[key]=dict(payload_binding=bound,independent_receipt_path=str(receipt),
                scope='Completed predetermined checkpoint only; no full14000 horizon or later missing save inferred')
    return result


def genealogy_edges(inventories):
    edges=[]
    for child,parent in (('assistant-full-pilot','base-parent'),('assistant-lora-pilot','base-parent'),
            ('assistant-lora-merge','assistant-lora-pilot'),('dpo-pilot','assistant-full-pilot'),
            ('chosen-pilot','assistant-full-pilot'),('rlvr-g4-pilot','instruct-parent'),('rlvr-g8-pilot','instruct-parent')):
        if child not in inventories:continue
        exported=inventories[child];ancestor=inventories.get(parent);genealogy=exported.get('genealogy') or {}
        joined=None
        if ancestor:
            if child=='assistant-lora-merge':joined=genealogy.get('parent_adapter')==ancestor['path'] and genealogy.get('adapter_files')==ancestor['files']
            elif child in ('dpo-pilot','chosen-pilot'):joined=genealogy.get('parent_checkpoint')==ancestor['path'] and genealogy.get('parent_checkpoint_sha256')==ancestor['files']
            elif child.startswith('rlvr-'):joined=genealogy.get('parent_local_source_hashes')==ancestor['files']
            else:joined=genealogy.get('base_checkpoint_files')==ancestor['files']
        edges.append(dict(parent=parent,child=child,parent_path=None if ancestor is None else ancestor['path'],
            child_path=exported['path'],recorded_genealogy=genealogy,exact_recorded_parent_inventory_join=joined,
            scope='Assistant Base and native Instruct are separate branches; no checkpoint substitution or universal ranking'))
    return edges


def provenance(observations,documents,saved):
    retained={};current={}
    for identifier,observation in observations.items():
        if observation is None:continue
        records={}
        for path in observation['receipt_paths']:
            value=documents.get(path)
            if not isinstance(value,dict):continue
            selected={key:deepcopy(value[key]) for key in ('source_bindings','input_bindings','bindings','environment','interfaces',
                'observed_geometry','local_model_binding','local_base_binding','parent_binding','precision','science',
                'cpu_settings','snapshot_hash_workers','scope','boundaries','completion_level','full_schedule_complete',
                'execution_tranche') if key in value}
            if selected:records[path]=selected
            def walk(node):
                if isinstance(node,dict):
                    if isinstance(node.get('path'),str) and re.fullmatch(r'[0-9a-f]{64}',str(node.get('sha256',''))):
                        target=Path(node['path'])
                        if target.is_absolute() and (target.is_relative_to(ROOT/'src') or target.is_relative_to(ROOT/'scripts') or str(target) in (PYTHON,LOCK)):
                            key=str(target)
                            if key not in current:
                                if key==PYTHON:current[key]=_digest(key,executable=True)
                                else:current[key]=saved.bind(target,JSON_BYTES)
                    for item in node.values():walk(item)
                elif isinstance(node,list):
                    for item in node:walk(item)
            walk(selected)
        retained[identifier]=records
    return dict(retained_producer_source_environment_interface_records=retained,current_source_environment_file_bindings=current,
        boundary='Historical producer bindings stay historical even when source later changed. Current assembly bindings do not retroactively relabel a historical pass as current-source proof. Retained CPU/thread/hash-worker settings are execution configuration, not ledger identity, scientific intervention or a quota change')


def completed_story_ratings(observations,documents,bindings):
    observation=observations.get('story-ratings')
    if observation is None:return None
    report_path=str(Path(observation['path'])/'report.json');receipt_path=str(Path(observation['path'])/'receipt.json')
    report=documents.get(report_path);receipt=documents.get(receipt_path)
    publication=observations.get('story-publication');bundle=observations.get('story-blind-packet')
    if not isinstance(report,dict) or not isinstance(receipt,dict) or publication is None or bundle is None:return None
    if (report.get('status')!='supplied-ratings-replayed' or report.get('provenance')!='supplied-model-records' or
            len(report.get('rating_documents',[]))!=2 or len(report.get('raters',[]))!=2 or
            any(rater.get('provenance')!='ai' for rater in report['raters'])):return None
    binding=bindings.get(report_path)
    if binding is None or receipt.get('artifact_sha256',{}).get('report.json')!=binding['sha256']:
        raise ValueError('Actual supplied-model rating report differs from completed offline receipt')
    packet_path=str(Path(bundle['path'])/'packet.json');book_path=str(Path(bundle['path'])/'private-codebook.json')
    bundle_receipt=documents.get(str(Path(bundle['path'])/'receipt.json'))
    packet=documents.get(packet_path);book=documents.get(book_path)
    if not all(isinstance(value,dict) for value in (packet,book,bundle_receipt)):return None
    for value,key in ((packet,'packet_sha256'),(book,'codebook_sha256'),(report,'report_sha256')):
        if value.get(key)!=canonical_hash({name:item for name,item in value.items() if name!=key}):
            raise ValueError('Actual story rating packet/codebook/report semantic hash mismatch')
    for path,name in ((packet_path,'packet.json'),(book_path,'private-codebook.json')):
        actual=bindings.get(path)
        if actual is None:return None
        if bundle_receipt.get('artifact_sha256',{}).get(name)!=actual['sha256']:
            raise ValueError('Actual story blind bundle artifact receipt mismatch')
    publication_inputs={str(Path(publication['path'])/name) for name in ('contract.json','checkpoints.json','records.jsonl')}
    publication_inputs.add(str(ROOT/'experiments/specs/2026-10-05-story-publication-raters.json'))
    bundle_inputs=bundle_receipt.get('input_sha256',{})
    if not isinstance(bundle_inputs,dict) or set(bundle_inputs)!=publication_inputs:
        raise ValueError('Actual story blind bundle must bind this publication and declared raters')
    report_inputs=receipt.get('input_sha256',{})
    if not isinstance(report_inputs,dict) or not {packet_path,book_path}<=set(report_inputs):
        raise ValueError('Actual story rating receipt must bind the retained packet and private codebook')
    rating_paths=set(report_inputs)-{packet_path,book_path}
    if rating_paths!={str(Path(bundle['path'])/name) for name in STORY_RATING_FILES}:
        raise ValueError('Actual story rating receipt requires both predeclared supplied rating files, not empty templates')
    for inputs in (bundle_inputs,report_inputs):
        for path,sha in inputs.items():
            actual=bindings.get(path)
            if actual is None:return None
            if actual['sha256']!=sha:raise ValueError('Actual story bundle/rating input receipt mismatch: '+path)
    supplied=[documents.get(path) for path in rating_paths]
    if any(not isinstance(value,dict) for value in supplied):return None
    if (report['packet_sha256']!=packet['packet_sha256'] or book['packet_sha256']!=packet['packet_sha256'] or
            report['codebook_sha256']!=book['codebook_sha256'] or report.get('rubric_sha256')!=book.get('rubric_sha256') or
            report.get('contract_sha256')!=book.get('contract_sha256') or report['raters']!=book.get('raters') or
            book.get('provenance')!='supplied-model-records'):
        raise ValueError('Actual story rating report belongs to another packet/codebook')
    if sorted(supplied,key=lambda value:value.get('rater_id',''))!=sorted(report['rating_documents'],key=lambda value:value.get('rater_id','')):
        raise ValueError('Actual story report does not retain these supplied rating documents')
    if (len({value.get('rater_id') for value in supplied})!=2 or
            {value.get('rater_id') for value in supplied}!={value.get('rater_id') for value in report['raters']} or
            any(value.get('packet_sha256')!=packet['packet_sha256'] or value.get('rubric_sha256')!=book['rubric_sha256'] for value in supplied)):
        raise ValueError('Actual supplied story ratings differ from retained packet/rubric/raters')
    return report


def bind_story_rating_inputs(observations,saved):
    """Follow only existing offline receipt JSON inputs inside this publication."""
    publication=observations.get('story-publication');ratings=observations.get('story-ratings')
    if publication is None or ratings is None:return
    receipt=saved.documents.get(str(Path(ratings['path'])/'receipt.json'))
    inputs=receipt.get('input_sha256',{}) if isinstance(receipt,dict) else {}
    if not isinstance(inputs,dict):return
    bundle=Path(publication['path'])/'blind-packet-01'
    allowed={bundle/name for name in ('packet.json','private-codebook.json',*STORY_RATING_FILES)}
    for path in allowed:
        if str(path) in inputs:saved.read(path)


def scientific_slices(observations,documents,bindings=None):
    def document(identifier,name):
        observation=observations.get(identifier)
        return None if observation is None else documents.get(str(Path(observation['path'])/name))
    story=document('story-publication','coverage.json')
    return dict(story_publication=story,story_ratings=completed_story_ratings(observations,documents,bindings or {}),
        story_later_checkpoints={str(update):None for update in (4000,8000,14000)},
        assistant_comparison=document('assistant-comparison-2','comparison.json'),
        preference_comparison=document('preference-comparison','comparison.json'),
        reasoning_original_four_interfaces={role:{str(cap):document(f'reasoning-{role}-cap{cap}','acceptance.json') for cap in (32,128)}
            for role in ('base-raw','base-chat','instruct-thinking-off','instruct-thinking-on')},
        rlvr_common20={str(cap):(None if observations.get(f'rlvr-common20-cap{cap}') is None else documents[observations[f'rlvr-common20-cap{cap}']['path']]) for cap in (32,128)},
        scope='Retained actual outcomes only; no scoring/rating code or new pooled capability percentage. Missing later/max-horizon/multi-seed/scale evidence is not implied by a scoped comparison')


def output_path(run_id):
    if type(run_id) is not str or re.fullmatch(r'run-[0-9]{2}',run_id) is None:raise ValueError('Closed run-NN snapshot identifier required')
    return ROOT/'experiments/reports'/('native-campaign-evidence-20261005-'+run_id)


def assemble(run_id):
    target=output_path(run_id);_new_target(str(target));sources={name:_digest(str(ROOT/name)) for name in SOURCES}
    saved=SavedEvidence();began=time.monotonic();target.mkdir(mode=0o700)
    try:
        for name in DOCUMENTS:saved.bind(ROOT/name,JSON_BYTES)
        archive=saved.read(ROOT/ARCHIVE);rows,changes=original_rows(archive,stage_rows())
        entries=catalog();observations={entry['id']:collect_entry(entry,saved) for entry in entries}
        bind_story_rating_inputs(observations,saved)
        inventories=bind_inventories(inventory_candidates(observations,saved.documents),saved)
        story_checkpoints=bind_story_checkpoints(saved)
        lineage=genealogy_edges(inventories);source_provenance=provenance(observations,saved.documents,saved)
        interpreter=_digest(sys.executable,executable=True);lock=saved.bind(Path(LOCK),JSON_BYTES)
        report=dict(schema='dongxi-native-campaign-evidence-snapshot-v1',status='assembled-scoped-observations-not-all-maxima-complete',
            utc=datetime.now(timezone.utc).isoformat(),run_id=run_id,
            original_archived_stage_rows=rows,original_stage_rows_sha256=canonical_hash(rows),
            original_archived_evaluation_contracts=deepcopy(archive['campaign']['evaluation_contracts']),
            current_source_stage_documentation_differences=changes,stage_observations=stage_observations(rows,observations,saved.documents),
            observations=observations,child_accounting=child_accounting(observations,saved.documents),
            actual_selected_output_inventories=inventories,actual_story_checkpoint_payloads=story_checkpoints,
            actual_genealogy=lineage,scientific_slices=scientific_slices(observations,saved.documents,saved.bindings),
            provenance=source_provenance,retained_receipt_documents=saved.documents,
            source_bindings=sources,input_outcome_output_bindings=saved.bindings,missing_evidence_paths=sorted(saved.missing),
            retained_directory_layouts=saved.directories,pinned_cache_link_bindings=saved.cache_links,
            assembler_environment=dict(identity=environment_identity(LOCK),interpreter_binding=interpreter,lock_binding=lock),
            assembler_cpu_hash_work=dict(initial_unique_streamed_bytes=saved.bytes_hashed,elapsed_seconds_before_closing=time.monotonic()-began,
                boundary='CPU snapshot read/hash/JSON work and both closing verification hash passes are outside producer journals; no models/GPU loaded'),
            cost_boundaries=['Native supervisor child counts, CPU comparison receipts and outer adapter counts are distinct',
                'Shared receipt paths and persistent journals are not added once per planned row or embedded acceptance copy',
                'Retained attempted/completed work, elapsed seconds, target/position counts and serialization counters have different overlapping scopes',
                'CUDA peaks and sampled host memory are per-child observations, never additive branch cost or continuous-memory guarantees',
                'Story first400 tranche is not the full14000 schedule/50M-target ceiling; later checkpoint cells remain null',
                'Original18-package criteria and optional planned extensions are not replaced by a45-max-stage completion checklist',
                'Base→full400→DPO/chosen and separate Instruct→G4/G8 genealogy remain separate; no universal pooled model rank'])
        saved.verify()
        for path,value in source_provenance['current_source_environment_file_bindings'].items():
            if value is not None and 'actual_path' in value and _digest(path,executable=True)!=value:
                raise ValueError('Producer interpreter bytes changed during assembly')
        if {name:_digest(str(ROOT/name)) for name in SOURCES}!=sources or _digest(sys.executable,executable=True)!=interpreter:
            raise ValueError('Assembler source/interpreter changed')
        retain_snapshot(target/'snapshot.json',report,maximum=SNAPSHOT_BYTES)
        compress_snapshot(target/'snapshot.json',target/'snapshot.json.gz',maximum=SNAPSHOT_BYTES)
        saved.verify()
        if {name:_digest(str(ROOT/name)) for name in SOURCES}!=sources or _digest(sys.executable,executable=True)!=interpreter:
            raise ValueError('Assembler source/interpreter changed during final write')
        for path,value in source_provenance['current_source_environment_file_bindings'].items():
            if value is not None and 'actual_path' in value and _digest(path,executable=True)!=value:
                raise ValueError('Producer interpreter bytes changed during final write')
        retain(target/'closing-bindings.json',dict(status='unchanged',source_bindings=sources,input_outcome_output_bindings=saved.bindings,
            snapshot_binding=stream_binding(target/'snapshot.json',SNAPSHOT_BYTES),
            snapshot_gzip_binding=stream_binding(target/'snapshot.json.gz',SNAPSHOT_BYTES),
            archive_output_limits=dict(uncompressed_bytes=SNAPSHOT_BYTES,compressed_bytes=SNAPSHOT_BYTES,
                decompressed_bytes=SNAPSHOT_BYTES,boundary='Separate CPU archive-output ceilings; producer/input limits unchanged'),
            compression=dict(format='gzip',level=GZIP_LEVEL,mtime=0,original_filename='',automatic_extraction=False,
                python_version=sys.version,zlib_compile_version=zlib.ZLIB_VERSION,zlib_runtime_version=zlib.ZLIB_RUNTIME_VERSION,
                deterministic_scope='Identical input in the recorded Python/zlib environment; no cross-version promise'),
            missing_evidence_paths=sorted(saved.missing),
            retained_directory_layouts=saved.directories,pinned_cache_link_bindings=saved.cache_links,
            assembler_seconds=time.monotonic()-began,scope='Actual metadata/output byte closure, not a new producer run, package-status update or all45 maxima success'))
        print(json.dumps(dict(status=report['status'],output=str(target),native_model_children=report['child_accounting']['distinct_native_model_children'])),flush=True)
        return 0
    except BaseException as error:
        retain(target/'failure.json',dict(type=type(error).__name__,message=str(error),retained_bindings=saved.bindings,
            retained_missing_paths=sorted(saved.missing),scope='Snapshot assembly invalid/incomplete; all producer/failure bytes and any prior snapshot remain unchanged'))
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-id',default='run-01')
    args=parser.parse_args(argv);return assemble(args.run_id)


if __name__=='__main__':raise SystemExit(main())
