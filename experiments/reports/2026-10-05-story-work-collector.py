"""Exclusive CPU evidence collection for original persistent story work."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests'))
import test_stories_work_budget as controls
RESERVE_SAMPLES=[]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    with Path(path).open('x',encoding='utf8') as handle:
        handle.write(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
        handle.flush();os.fsync(handle.fileno())


def run(command,root):
    sampled=controls.story.available_gib();RESERVE_SAMPLES.append(sampled)
    if sampled<25:raise RuntimeError('Refusing bounded CPU child below25GiB sampled host reserve')
    started=time.monotonic()
    answer=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=180)
    footers=re.findall(r'Ran (\d+) tests? in ([0-9.]+)s\s+(OK|FAILED[^\n]*)',answer.stderr)
    footer=None if not footers else dict(test_count=int(footers[-1][0]),
        unittest_seconds=float(footers[-1][1]),outcome=footers[-1][2])
    result=dict(command=command,actual_exit_code=answer.returncode,
        seconds=time.monotonic()-started,stdout=answer.stdout,stderr=answer.stderr,
        deadline_seconds=180,actual_footer=footer,sampled_prechild_available_gib=sampled)
    write(root,result);return result


def tensors(path,value):
    with Path(path).open('xb') as handle:
        torch.save(value,handle);handle.flush();os.fsync(handle.fileno())


def original_equations(root):
    root.mkdir();results=[]
    for seed in (909,910):
        for mode in (False,True):
            arm=root/f'{seed}-{int(mode)}';arm.mkdir()
            archived,module=controls.plain(seed,mode,archived=True)
            current,_=controls.plain(seed,mode)
            actual,science,budget=controls.fixture(arm/'accounted',seed,mode)
            try:
                histories={name:[] for name in ('archived','current_unaccounted','accounted')}
                rows=[('archived',archived),('current_unaccounted',current),('accounted',actual)]
                for _ in range(5):
                    for name,session in rows:histories[name].append(controls.record(session.update()))
                    if len({controls.state_sha(session) for name,session in rows})!=1:
                        raise AssertionError('Original-equation parameter/Adam/stream/RNG comparison failed')
                if len({controls.original.state_digest(value) for value in histories.values()})!=1:
                    raise AssertionError('Original numerical histories differ')
                states={name:controls.state_sha(session) for name,session in rows}
                evaluations={name:session.evaluate(controls.original.AuthoredWindows()) for name,session in rows}
                activations={name:(module if name=='archived' else controls.story).activation_summary(
                    session,controls.original.AuthoredWindows()) for name,session in rows}
                if any(value!=evaluations['archived'] for value in evaluations.values()):
                    raise AssertionError('Original evaluation differs')
                if any(value!=activations['archived'] for value in activations.values()):
                    raise AssertionError('Original activation observation differs')
                for name,session in rows:tensors(arm/(name+'-numerical-state.pt'),controls.numerical(session))
                write(arm/'history.json',histories)
                row=dict(seed=seed,activation_checkpointing=mode,actual_successful_updates=5,
                    config=asdict(actual.model.cfg),recipe=asdict(actual.recipe),science=science,
                    valid_target_budget=1000,state_sha256=states,histories_equal=True,
                    evaluations=evaluations,activations=activations,work=budget.ledger.snapshot())
                write(arm/'comparison.json',row);results.append(row)
            finally:budget.ledger.close()
    return results


def native_sampling(root):
    root.mkdir();archived,module=controls.plain(archived=True,native_sampling=True)
    current,_=controls.plain(native_sampling=True)
    actual,science,budget=controls.fixture(root/'accounted',native_sampling=True)
    try:
        tok=controls.AuthoredTokenizer()
        answers={'archived':module.samples(archived,tok,max_new=3),
            'current_unaccounted':controls.story.samples(current,tok,max_new=3),
            'accounted':controls.story.samples(actual,tok,max_new=3)}
        if not all(answer==answers['archived'] for answer in answers.values()):
            raise AssertionError('Native original EOS/sampling comparison failed')
        row=dict(config=asdict(actual.model.cfg),recipe=asdict(actual.recipe),science=science,
            vocabulary=50257,eos=50256,temperature=.8,private_generator_seed=909,max_new=3,
            authored_prefix_lengths=[2,3,4],outputs=answers,exact_original_output=True,
            work=budget.ledger.snapshot(),natural_eos_sequences=sum(r['ended_with_eos'] for r in answers['accounted']),
            scope='Authored token-ID interface, not GPT-2 tokenizer or English-story quality')
        write(root/'sampling.json',row);return row
    finally:budget.ledger.close()


def failed_publication(root):
    root.mkdir();results=[]
    for kind in ('serialization','link','receipt'):
        arm=root/kind;session,science,budget=controls.fixture(arm)
        try:
            old=arm/'completed0.pt';retained=controls.save(session,old)
            original_bytes=digest(old);original_receipt=digest(str(old)+'.work.json')
            session.update();target=arm/'incomplete1.pt'
            def partial(state,handle):handle.write(b'partial');handle.flush();raise OSError('authored serializer')
            owner=torch if kind=='serialization' else (controls.story.os if kind=='link' else controls.story)
            method='save' if kind=='serialization' else ('link' if kind=='link' else 'publish_story_receipt')
            effect=partial if kind=='serialization' else OSError('authored publication refusal')
            try:
                with patch.object(owner,method,side_effect=effect):session.save(target)
            except OSError as error:failure=dict(type=type(error).__name__,message=str(error))
            else:raise AssertionError('Declared publication failure absent')
            if digest(old)!=original_bytes or digest(str(old)+'.work.json')!=original_receipt:
                raise AssertionError('Failed save damaged previous completed boundary')
            row=dict(kind=kind,error=failure,completed_updates=session.step,carried_receipt=retained,
                old_payload_sha256=original_bytes,old_receipt_sha256=original_receipt,
                last_receipt_unchanged=budget.last_receipt==retained,
                partial_payload_exists=target.exists(),partial_temporary_exists=target.with_suffix('.pt.tmp').exists(),
                partial_receipt_exists=Path(str(target)+'.work.json').exists(),work=budget.ledger.snapshot())
            write(arm/'failure.json',row);results.append(row)
        finally:budget.ledger.close()
    return results


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--development',action='store_true');args=parser.parse_args()
    root=Path(args.output).resolve();root.mkdir(mode=0o700,parents=True,exist_ok=False)
    sources=tuple(controls.SOURCES)+(str(Path(__file__).relative_to(ROOT)),)
    before={path:digest(ROOT/path) for path in sources}
    originals=root/'source-before';originals.mkdir()
    for path in sources:
        output=originals/(path+'.txt');output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('xb') as handle:handle.write((ROOT/path).read_bytes())
    gib=controls.story.available_gib()
    RESERVE_SAMPLES.append(gib)
    if gib<25:raise RuntimeError('Refusing CPU children below25GiB available host reserve')
    started=time.monotonic();panels={};arms=[];comparisons=[];sampling=None;publications=[];failure=None
    try:
        panels['new_controls']=run([sys.executable,'-m','unittest','-v','test_stories_work_budget'],root/'new-controls.json')
        if panels['new_controls']['actual_exit_code']:raise RuntimeError('New controls failed; retained raw footer')
        if not args.development:
            panels['existing_story_and_work']=run([sys.executable,'-m','unittest','-v',
                'test_stories_pipeline','test_stories_valid_target_budget','test_work_budget'],root/'existing-controls.json')
            if panels['existing_story_and_work']['actual_exit_code']:raise RuntimeError('Existing regressions failed')
            panels['reader_lesson']=run([sys.executable,'-m','unittest','-v','test_stories_work_lab'],root/'reader-lesson.json')
            if panels['reader_lesson']['actual_exit_code']:raise RuntimeError('Reader lesson controls failed')
            comparisons=original_equations(root/'original-equations')
            sampling=native_sampling(root/'native-sampling')
            publications=failed_publication(root/'failed-publication')
            for seed in (909,910):
                for mode in (False,True):
                    arms.append(controls.replay_arm(root/f'arm-{seed}-{int(mode)}',seed,mode))
    except BaseException as error:
        failure=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    after={path:digest(ROOT/path) for path in sources}
    artifacts={str(path.relative_to(ROOT)):dict(sha256=digest(path),bytes=path.stat().st_size)
        for path in sorted(root.rglob('*')) if path.is_file()}
    result=dict(schema='dongxi-story-work-cpu-reference-v1',development=args.development,
        status='passed' if failure is None and before==after else 'failed',failure=failure,
        source_sha256_before=before,source_sha256_after=after,source_stable=before==after,
        artifact_sha256=artifacts,panels=panels,arms=arms,original_equations=comparisons,
        native_sampling=sampling,failed_publication=publications,seconds=time.monotonic()-started,
        minimum_sampled_prelaunch_available_gib=min(RESERVE_SAMPLES+
            [arm['execution']['sampled_prechild_available_gib'] for arm in arms]),
        prelaunch_host_available_gib_samples=RESERVE_SAMPLES,
        collection_invocation=dict(orig_argv=list(sys.orig_argv),process_id=os.getpid(),
            python_executable=sys.executable,prefix=sys.prefix,python_version=platform.python_version(),
            platform=platform.platform(),machine=platform.machine(),torch=str(torch.__version__),
            observed_flags={key:os.environ.get(key) for key in ('PYTHONPATH','CUDA_VISIBLE_DEVICES',
                'HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}),
        external_jobs_launched=0,
        pending='pretrained/GPU/platform/byte-I/O/physical gates and all45 external outcomes remain pending')
    write(root/'verification.json',result)
    print(json.dumps(dict(status=result['status'],verification=str(root/'verification.json'),
        source_count=len(before),artifact_count=len(artifacts),failure=failure),sort_keys=True),flush=True)
    if result['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
