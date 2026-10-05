#!/usr/bin/env python3
"""Exclusive bounded CPU reference collection; no external model or service."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests'));sys.path.insert(0,str(ROOT/'src'))
import test_rlvr_snapshot_io as control
from dongxi_llms import training_snapshot as shared


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def available():
    row=next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    return int(row.split()[1])/1024**2


def sources():
    return {name:digest(ROOT/name) for name in (*control.SOURCES,str(Path(__file__).relative_to(ROOT)))}


def child(command,output,*,deadline=180):
    reserve=available()
    if reserve<25:raise RuntimeError('Actual sampled host reserve below declared25GiB')
    start=time.monotonic()
    process=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,timeout=deadline)
    log=process.stdout+process.stderr
    with Path(output).open('x',encoding='utf8') as handle:
        handle.write(log);handle.flush();os.fsync(handle.fileno())
    matches=re.findall(r'^Ran (\d+) tests? in ([0-9.]+)s$',log,re.MULTILINE)
    count=int(matches[-1][0]) if matches else None
    seconds=float(matches[-1][1]) if matches else None
    return dict(command=command,actual_exit_code=process.returncode,seconds=time.monotonic()-start,
        deadline_seconds=deadline,available_before_gib=reserve,available_after_gib=available(),
        raw_log=dict(path=str(Path(output).relative_to(ROOT)),sha256=digest(output),bytes=Path(output).stat().st_size),
        actual_unittest_count=count,actual_unittest_seconds=seconds,
        actual_unittest_footer_ok=bool(re.search(r'^OK(?: \(skipped=\d+\))?$',log,re.MULTILINE)))


def partial_control(directory):
    directory.mkdir(mode=0o700);loop,contract=control.fixture()
    work,io,hook=control.create(directory/'journals',loop,contract)
    try:
        retained=control.save(loop,directory/'completed0.pt',contract,hook)
        before=digest(retained['path']);loop.collect();partial=directory/'failed-pending0.pt'
        def fail(value,handle,*args,**kwargs):
            handle.write(b'partial!');raise OSError('authored eight-byte partial serializer publication')
        try:
            with patch.object(shared.torch,'save',side_effect=fail):control.save(loop,partial,contract,hook)
        except OSError as error:failure=dict(type=type(error).__name__,message=str(error))
        else:raise AssertionError('Predeclared partial serializer failure absent')
        after=dict(model=work.snapshot(),io=io.snapshot())
        if digest(retained['path'])!=before or Path(str(partial)+'.commit.json').exists() or Path(str(partial)+'.work.json').exists():
            raise AssertionError('Failed publication damaged a committed boundary or forged a commit')
        result=dict(original_checkpoint=retained,original_payload_sha256=before,error=failure,
            actual_partial_bytes=partial.stat().st_size,partial_sha256=digest(partial),
            old_checkpoint_unchanged=True,partial_committed=False,partial_receipt_published=False,
            work=after['model'],io=after['io'])
        control.write(directory/'partial-publication.json',result);return result
    finally:io.close();work.close()


def main():
    if len(sys.argv)!=2:raise SystemExit('One new exclusive report directory required')
    expected='/tmp/dongxi-course-reproduction.ZfVaEu/venv'
    if sys.prefix!=expected:raise RuntimeError('Existing isolated CPU interpreter required')
    if control.torch.cuda.is_available() or control.torch.version.cuda is not None:
        raise RuntimeError('Actual CPU-only build required')
    output=Path(sys.argv[1]).resolve();output.mkdir(mode=0o700,parents=True,exist_ok=False)
    source_before=sources();control.write(output/'source-before.json',source_before)
    for name,sha in source_before.items():
        dest=output/'source-capture'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as handle:handle.write((ROOT/name).read_bytes())
        if digest(dest)!=sha:raise AssertionError('Archived source differs from measured bytes')
    archive=ROOT/control.ARCHIVE
    original=json.loads((archive/'manifest.json').read_text())
    if digest(archive/'qwen_rlvr_lab.py.txt')!=original['original_runner_sha256']:
        raise AssertionError('Original actor archive changed')
    report=dict(schema='dongxi-rlvr-snapshot-io-reference-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        command=list(sys.orig_argv),source_sha256=source_before,commands=[],arms=[],cli_controls=[],
        status='running',model_scale_jobs_started=0,production_ready=False,launch_authorized=False,
        environment=dict(executable=sys.executable,prefix=sys.prefix,python=platform.python_version(),
            platform=platform.platform(),machine=platform.machine(),cuda_available=False,
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            hf_offline=os.environ.get('HF_HUB_OFFLINE')=='1',threads=control.torch.get_num_threads(),
            packages={name:importlib.metadata.version(name) for name in ('torch','transformers','tokenizers','peft')}),
        scope='original random local Qwen RLVR/native completed+pending CPU replay; no pretrained or physical proof')
    control.write(output/'start.json',report)
    started=time.monotonic()
    try:
        panels=[('new-focused',['test_rlvr_snapshot_io']),
            ('existing-shared',['test_rlvr_runner_recovery','test_rlvr_work_budget','test_training_snapshot',
                'test_training_snapshot_failure_paths','test_snapshot_io_budget']),
            ('reader-lesson',['test_rlvr_snapshot_io_lab'])]
        for name,modules in panels:
            row=child([sys.executable,'-m','unittest',*modules,'-v'],output/(name+'.log'))
            report['commands'].append(row)
            if row['actual_exit_code'] or not row['actual_unittest_footer_ok']:
                raise RuntimeError('Actual bounded acceptance child failed: '+name)
        for seed in (2323,2324):
            for phase in ('completed','pending'):
                report['arms'].append(control.replay_arm(output/f'{seed}-{phase}',seed,phase))
        for exhausted in (True,False):
            report['cli_controls'].append(control.cli_control(output/('cli-exhausted' if exhausted else 'cli-inspected'),exhausted=exhausted))
        for index,fields in enumerate([dict(phase='unknown'),dict(completed_updates=5),dict(phase='pending',completed_updates=4)]):
            report['cli_controls'].append(control.cli_control(output/f'cli-phase-{index}',bootstrap_fields=fields))
        for role in ('template','environment-lock','parent-artifact'):
            for hardlink in (False,True):
                if role=='parent-artifact' and not hardlink:continue
                report['cli_controls'].append(control.cli_control(output/f'cli-alias-{role}-{hardlink}',alias_role=role,hardlink=hardlink))
        report['partial_publication']=partial_control(output/'partial-publication')
        report['native_header_publication_failure']=control.publication_control(output/'native-publication-failure')
        report['status']='pass'
    except BaseException as error:
        report['status']='failed';report['failure']=dict(type=type(error).__name__,message=str(error))
    report['source_sha256_after']=sources()
    report['sources_unchanged']=report['source_sha256']==report['source_sha256_after']
    if not report['sources_unchanged']:report['status']='failed'
    report['seconds']=time.monotonic()-started
    report['count_policy']='Actual component footers are separate and overlap the full course suite; no global unique-test sum.'
    report['limitations']=dict(caller_capture_application_inventory_metadata_journal_identity='excluded from shared9 units',
        scope='declared shared visits/bytes, not CPU instructions/time/physical quotas or hostile-payload sandbox',
        learner_day=9,bounded_packages_complete=13,all45_external_outcomes='null',new_checkpoint_genealogy=[],
        pretrained_gpu_mac_hosted_cross_machine='not executed')
    report['artifact_sha256']={str(p.relative_to(ROOT)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()}
    control.write(output/'verification.json',report)
    print(json.dumps(dict(status=report['status'],record=str((output/'verification.json').relative_to(ROOT)),
        source_count=len(source_before),commands=report['commands'],seconds=report['seconds']),sort_keys=True))
    if report['status']!='pass':raise SystemExit(1)


if __name__=='__main__':main()
