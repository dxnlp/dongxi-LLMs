#!/usr/bin/env python3
"""Run the declared CPU acceptance panel and retain each command's evidence.

This orchestrator does not install dependencies or register kernels. A caller
must supply an isolated locked environment and a kernel using that interpreter.
No source notebook, environment or old evidence directory is overwritten.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from dongxi_llms.course_manifest import load_manifest, select_notebooks

CPU_VERIFICATION_SCOPE = (
    'CPU material checks on the recorded invocation platform; '
    'not other-platform reproduction, pretrained/GPU outcomes or learner mastery'
)

PANEL = [
    'notebooks/day-01/04_checkpoint_interface.ipynb',
    'notebooks/day-03/01_logits_softmax_nll.ipynb',
    'notebooks/day-10/04_mathematical_grading_and_response_replay.ipynb',
    'notebooks/day-15/03_preference_collection_and_judges.ipynb',
    'notebooks/day-20/03_behavior_probabilities_and_support.ipynb',
]


def execute(command, output, *, env, timeout, cwd=ROOT):
    """Return actual process status/output even after a timeout or failed check."""
    began = time.perf_counter()
    timed_out = False
    process = None

    def stop_group():
        if process is None:
            return
        # Jupyter starts kernels in independent sessions, so a process-group
        # signal alone is insufficient. Snapshot this invocation's descendants
        # before stopping its supervisor; psutil guards PID reuse on signaling.
        try:
            children = psutil.Process(process.pid).children(recursive=True)
        except psutil.NoSuchProcess:
            children = []
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        # The supervisor may have exited before its kernel; terminate the
        # remaining group, not an unrelated process or broad host service.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        _, alive = psutil.wait_procs(children, timeout=2)
        for child in alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass

    try:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            stop_group()
            stdout, stderr = process.communicate()
        except BaseException:
            stop_group()
            raise
        code = process.returncode
    except OSError as error:
        code, stdout, stderr = None, '', repr(error)
    output.write_text(stdout + '\n--- stderr ---\n' + stderr)
    return dict(command=command, exit_code=code, timed_out=timed_out,
                seconds=time.perf_counter()-began, log=str(output),
                log_sha256=hashlib.sha256(output.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--kernel', required=True)
    parser.add_argument('--host-reserve-gib', type=float, default=25)
    parser.add_argument('--full-notebooks', action='store_true')
    parser.add_argument('--command-timeout', type=int, default=600)
    args = parser.parse_args()
    if (args.command_timeout <= 0 or not math.isfinite(args.host_reserve_gib)
            or args.host_reserve_gib <= 0):
        parser.error('Timeout/reserve must be positive and finite')
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose a new empty CPU evidence directory')
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT/'src'), CUDA_VISIBLE_DEVICES='',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    manifest = dict(date_utc=datetime.now(timezone.utc).isoformat(),
                    scope=CPU_VERIFICATION_SCOPE,
                    executable=sys.executable, python=platform.python_version(),
                    environment_prefix=sys.prefix,
                    platform=platform.platform(), machine=platform.machine(),
                    kernel=args.kernel, commands=[], status='running',
                    base_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                    dirty=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).splitlines(),
                    packages={name:metadata.version(name) for name in
                              ['torch','numpy','matplotlib','nbclient','nbformat','ipykernel','transformers','tokenizers','peft']},
                    input_hashes={path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
                                  for path in ['pyproject.toml','uv.lock','docs/course_manifest.json']})
    source_paths = sorted([*(ROOT/'src').rglob('*.py'), *(ROOT/'scripts').glob('*.py'),
                           *(ROOT/'tests').glob('*.py')])
    manifest['source_hashes'] = {path.relative_to(ROOT).as_posix():hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in source_paths}
    # Bind lesson inputs before testing so one receipt represents one revision.
    lesson_paths = sorted([*(ROOT/'book').rglob('*.md'),
                           *(ROOT/'notebooks').glob('day-*/*.ipynb'),
                           *(ROOT/'docs/runbooks').glob('*.md'),
                           ROOT/'scripts/book_prose_exemptions.json',
                           ROOT/'experiments/reports/2026-10-10-book-editorial-pass/run-01/notebook-contract-review.json'])
    manifest['lesson_hashes'] = {
        path.relative_to(ROOT).as_posix():hashlib.sha256(path.read_bytes()).hexdigest()
        for path in lesson_paths if path.exists()}
    report = output/'cpu-verification.json'
    report.write_text(json.dumps(manifest, indent=2)+'\n')
    try:
        registry = load_manifest(ROOT)
        selected = select_notebooks(registry, paths=None if args.full_notebooks else PANEL)
        manifest['selected_notebooks'] = [row['path'] for row in selected]
        commands = [
            ('tests', [sys.executable,'-m','unittest','discover','-s','tests','-q']),
            ('math', [sys.executable,'scripts/check_book_math.py']),
            ('prose', [sys.executable,'scripts/check_book_prose.py','--strict-narrative','--json']),
            ('routes', [sys.executable,'scripts/check_course_integrity.py','--json']),
            ('notebook-contracts', [sys.executable,'scripts/check_notebook_contracts.py','--json']),
            ('notebooks', [sys.executable,'scripts/verify_course_notebooks.py',
                           '--kernel',args.kernel,'--output',str(output/'notebooks'),
                           '--expected-prefix',sys.prefix,
                           '--host-reserve-gib',str(args.host_reserve_gib),
                           '--notebooks',*[row['path'] for row in selected]]),
        ]
        for name, command in commands:
            result = execute(command, output/f'{name}.log', env=env, timeout=args.command_timeout)
            result['name'] = name
            manifest['commands'].append(result)
            report.write_text(json.dumps(manifest, indent=2)+'\n')
            print(json.dumps({key:result[key] for key in ['name','exit_code','timed_out','seconds']}), flush=True)
        manifest['changed_sources_during_run'] = [path for path, digest in manifest['source_hashes'].items()
            if not (ROOT/path).exists() or hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != digest]
        manifest['changed_lessons_during_run'] = [path for path, digest in manifest['lesson_hashes'].items()
            if not (ROOT/path).exists() or hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != digest]
        manifest['status'] = 'passed' if all(row['exit_code']==0 and not row['timed_out']
            for row in manifest['commands']) and not manifest['changed_sources_during_run'] \
            and not manifest['changed_lessons_during_run'] else 'failed'
    except BaseException as error:
        manifest['status'], manifest['error'] = 'failed', repr(error)
        report.write_text(json.dumps(manifest, indent=2)+'\n')
        raise
    report.write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'CPU evidence: {report}', flush=True)
    if manifest['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
