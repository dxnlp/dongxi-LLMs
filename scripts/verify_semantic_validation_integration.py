#!/usr/bin/env python3
"""Retain bounded offline CPU acceptance; never launch model-scale work.

Run only in an already prepared isolated CPU environment. A new output directory
is mandatory. Source/document identity and available-memory observations are
checks, not physical containment or a whole-job resource quota.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
NOTEBOOKS = [
    'notebooks/day-01/04_checkpoint_interface.ipynb',
    'notebooks/day-03/01_logits_softmax_nll.ipynb',
    'notebooks/day-10/04_mathematical_grading_and_response_replay.ipynb',
    'notebooks/day-15/03_preference_collection_and_judges.ipynb',
    'notebooks/day-20/03_behavior_probabilities_and_support.ipynb',
]
LESSON = 'notebooks/day-25/02_ragged_kv_cache_and_exact_recovery.ipynb'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
        handle.flush(); os.fsync(handle.fileno())


def memory_gib():
    line = next(row for row in Path('/proc/meminfo').read_text().splitlines()
                if row.startswith('MemAvailable:'))
    return int(line.split()[1])/1024**2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kernel', required=True)
    parser.add_argument('--expected-prefix', type=Path, required=True)
    parser.add_argument('--host-reserve-gib', type=float, default=25.)
    parser.add_argument('--timeout', type=int, default=180)
    args = parser.parse_args()
    if not 0 < args.host_reserve_gib <= 1024 or not 1 <= args.timeout <= 180:
        parser.error('Explicit finite positive reserve and at most180s deadline required')
    if Path(sys.prefix).resolve() != args.expected_prefix.resolve():
        raise RuntimeError('Collector must use the declared existing isolated environment')
    import torch
    if torch.cuda.is_available() or torch.version.cuda is not None:
        raise RuntimeError('Actual CPU-only environment required')
    output = args.output.resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    paths = sorted(path for directory in ('src', 'scripts', 'tests', 'training')
                   for path in (ROOT/directory).rglob('*.py') if '__pycache__' not in path.parts)
    source_before = {str(path.relative_to(ROOT)):sha(path) for path in paths}
    input_paths = ['uv.lock', 'docs/course_manifest.json', 'docs/course_improvements.json']
    inputs = {name:sha(ROOT/name) for name in input_paths}
    before = memory_gib()
    report = dict(date_utc=datetime.now(timezone.utc).isoformat(), scope='Local isolated offline Linux ARM64 CPU only',
        command=list(sys.orig_argv), base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],cwd=ROOT,text=True).strip(),
        executable=sys.executable, prefix=sys.prefix, python=platform.python_version(), platform=platform.platform(),
        machine=platform.machine(), cuda_available=torch.cuda.is_available(), torch_cuda_build=torch.version.cuda,
        source_sha256_before=source_before, input_sha256=inputs, commands=[],
        notebook_scope='Five unchanged references plus the changed Day25 recovery/RNG/reader lesson; not all76 notebooks',
        minimum_sampled_available_gib=before, required_host_reserve_gib=args.host_reserve_gib,
        model_scale_jobs_started=0, production_ready=False, launch_authorized=False)
    write(output/'start.json', report)
    env = dict(os.environ, PYTHONPATH=str(ROOT/'src'), CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    common = [sys.executable, 'scripts/verify_course_notebooks.py', '--kernel', args.kernel,
        '--expected-prefix', str(args.expected_prefix), '--host-reserve-gib', str(args.host_reserve_gib),
        '--timeout', str(args.timeout)]
    commands = [
        ('tests', [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-q']),
        ('math', [sys.executable, 'scripts/check_book_math.py']),
        ('routes', [sys.executable, 'scripts/check_course_integrity.py', '--json']),
        ('notebooks', common+['--output',str(output/'notebooks'),'--notebooks',*NOTEBOOKS]),
        ('rng_lesson', common+['--output',str(output/'rng-lesson'),'--notebooks',LESSON]),
    ]
    for name, command in commands:
        sample = memory_gib(); report['minimum_sampled_available_gib'] = min(report['minimum_sampled_available_gib'],sample)
        if sample < args.host_reserve_gib:
            report['commands'].append(dict(name=name, command=command, started=False,
                error='Available-memory sample below declared reserve'))
            break
        started = time.monotonic(); log = output/(name+'.log'); stopped = None
        with log.open('xb') as handle:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                sample = memory_gib(); report['minimum_sampled_available_gib'] = min(report['minimum_sampled_available_gib'],sample)
                if time.monotonic()-started > args.timeout or sample < args.host_reserve_gib:
                    stopped = 'deadline' if time.monotonic()-started > args.timeout else 'sampled reserve'
                    # This process group was created above for this exact child.
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    break
                time.sleep(.2)
        row = dict(name=name, command=command, started=True, exit_code=process.returncode,
            seconds=time.monotonic()-started, stopped_by=stopped, log=str(log.relative_to(ROOT)), log_sha256=sha(log))
        report['commands'].append(row)
        print(json.dumps(row),flush=True)
        if process.returncode != 0: break
    report['source_sha256_after'] = {str(path.relative_to(ROOT)):sha(path) for path in paths}
    report['sources_unchanged'] = report['source_sha256_before']==report['source_sha256_after']
    report['inputs_unchanged'] = inputs=={name:sha(ROOT/name) for name in input_paths}
    report['status'] = 'passed' if (len(report['commands'])==len(commands)
        and all(row.get('exit_code')==0 for row in report['commands'])
        and report['sources_unchanged'] and report['inputs_unchanged']) else 'failed'
    write(output/'cpu-verification.json',report)
    print('Retained acceptance:',output/'cpu-verification.json',report['status'],flush=True)
    if report['status'] != 'passed': raise SystemExit(1)


if __name__ == '__main__':
    main()
