import hashlib
import json
import os
from pathlib import Path
import sys
import time
import unittest

ROOT=Path.cwd();OUT=ROOT/'experiments/reports/2026-10-11-cs336-upgrade/run-01/systems'
patterns=['test_systems_performance_lab.py','test_online_attention_lab.py','test_scaling_lab.py']
suite=unittest.TestSuite(unittest.defaultTestLoader.discover('tests',pattern=p) for p in patterns)
start=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(suite)
files=[ROOT/'tests'/p for p in patterns]+[ROOT/'src/dongxi_llms'/p for p in ['systems_performance_lab.py','online_attention_lab.py','scaling_lab.py']]
receipt=dict(status='passed' if result.wasSuccessful() else 'failed',tests_run=result.testsRun,
             failures=len(result.failures),errors=len(result.errors),elapsed_seconds=time.monotonic()-start,
             executable=sys.executable,command='PYTHONPATH=src CUDA_VISIBLE_DEVICES=\'\' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /tmp/dongxi-cs336-g9l7v90r/venv/bin/python experiments/reports/2026-10-11-cs336-upgrade/run-01/systems/focused-tests.py',
             source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
             boundary='Independent closed counts, synchronization order, checkpoint recomputation/parity, SDPA output/gradient/masks, numerical gradcheck, known-surface recovery and negative fit/identity contracts; no GPU performance test')
(OUT/'focused-tests-v1.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
if not result.wasSuccessful():raise SystemExit(1)
