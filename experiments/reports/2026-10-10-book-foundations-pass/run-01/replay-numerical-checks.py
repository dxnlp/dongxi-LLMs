"""Replay the three independent arithmetic/API checks and preserve fresh logs.

Run with the goal-local locked CPU interpreter and PYTHONPATH=src. Each check
compares explicit arithmetic with the existing canonical implementation.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
RUN = Path(__file__).resolve().parent
env = dict(os.environ, PYTHONPATH=str(ROOT/'src'), CUDA_VISIBLE_DEVICES='',
           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')

def stable(value):
    if isinstance(value, dict):
        return {k: stable(v) for k, v in value.items() if k != 'elapsed_seconds'}
    if isinstance(value, list):
        return [stable(v) for v in value]
    return value

depth_before = json.loads((RUN/'depth-calculations.json').read_text())
operations_before = json.loads((RUN/'operations-numerical-results-v2.json').read_text())
routes = json.loads((RUN/'routes-edits.json').read_text())['numerical_verification']
checks = [
    ('depth', [sys.executable, str(RUN/'depth-calculations.py')], None,
     depth_before, RUN/'depth-calculations.json'),
    ('operations', [sys.executable, str(RUN/'operations-numerical-check.py')], None,
     operations_before, None),
    ('routes', [sys.executable, '-'], routes['program_source'], routes['stdout'], None),
]
records = []
for name, command, source, expected, result_file in checks:
    result = subprocess.run(command, input=source, env=env, cwd=ROOT, text=True,
                            capture_output=True, timeout=60)
    (RUN/f'root-{name}-replay.txt').write_text(result.stdout+'\n--- stderr ---\n'+result.stderr)
    if result.returncode:
        raise RuntimeError(f'{name}: numerical replay failed: {result.stderr}')
    actual = json.loads(result_file.read_text() if result_file else result.stdout)
    assert stable(actual) == stable(expected), f'{name}: replay results changed'
    records.append(dict(check=name, exit_code=result.returncode, status='passed',
                        agrees_with_initial_calculation=True, command=command,
                        embedded_program_sha256=hashlib.sha256(source.encode()).hexdigest() if source else None))
(RUN/'numerical-replay.json').write_text(json.dumps(dict(status='passed', checks=records,
    scope='Fresh root execution of independent explicit-arithmetic versus canonical-API/autograd checks; no model-scale run'), indent=2)+'\n')
print(json.dumps(dict(status='passed', replayed_checks=len(records))))
