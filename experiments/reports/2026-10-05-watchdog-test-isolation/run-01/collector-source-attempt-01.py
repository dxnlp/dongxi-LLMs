"""Bounded inert CPU checks of test-only fresh-process watchdog isolation."""
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/home/dongxi/dongxi_ai/Dongxi_LLMs')
PYTHON = '/tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python'
OUTPUT = ROOT/'experiments/reports/2026-10-05-watchdog-test-isolation/run-01'
NAMES = ('tests/test_native_profile_supervisor.py',
         'src/dongxi_llms/native_profile_supervisor.py',
         'src/dongxi_llms/campaign_supervisor.py',
         'experiments/reports/2026-10-05-watchdog-test-isolation-collector.py')


def hashes():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in NAMES}


def run():
    OUTPUT.mkdir(mode=0o700, parents=True, exist_ok=False)
    before = hashes()
    environment = dict(os.environ, PYTHONPATH=str(ROOT/'src')+':'+str(ROOT/'tests'),
        CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    environment.pop('DONGXI_WATCHDOG_ISOLATED_WORKER', None)
    tree = ast.parse((ROOT/NAMES[0]).read_text())
    methods = {node.name: [child.name for child in node.body if isinstance(child, ast.FunctionDef)
                          and child.name.startswith('test_')]
               for node in tree.body if isinstance(node, ast.ClassDef) and node.name in ('PreparationTests', 'WatchdogTests')}
    assert sum(len(v) for v in methods.values()) == 26
    contaminated = (
        "import threading, unittest, sys; "
        "stop=threading.Event(); auxiliary=threading.Thread(target=stop.wait,name='authored-whole-suite-parent',daemon=True); "
        "auxiliary.start(); print('authored_parent_thread_count='+str(threading.active_count())); "
        "suite=unittest.defaultTestLoader.loadTestsFromName('test_native_profile_supervisor'); "
        "result=unittest.TextTestRunner(verbosity=1).run(suite); "
        "print('auxiliary_preserved_until_owned_shutdown='+str(auxiliary.is_alive())); "
        "stop.set(); auxiliary.join(timeout=1); "
        "sys.exit(0 if result.wasSuccessful() and not auxiliary.is_alive() else 1)"
    )
    results = []
    for label, command in (
        ('isolated', [PYTHON, '-m', 'unittest', '-q', 'test_native_profile_supervisor']),
        ('thread-contaminated', [PYTHON, '-c', contaminated])):
        directory = OUTPUT/label
        directory.mkdir(mode=0o700)
        evidence = directory/'evidence'
        evidence.mkdir(mode=0o700)
        scenario_env = dict(environment, DONGXI_WATCHDOG_EVIDENCE=str(evidence))
        started = time.monotonic()
        process = subprocess.run(command, cwd=ROOT, env=scenario_env, text=True,
            capture_output=True, timeout=90)
        elapsed = time.monotonic()-started
        for name, contents in (('stdout.txt', process.stdout), ('stderr.txt', process.stderr)):
            with (directory/name).open('x') as handle:
                handle.write(contents)
        transports = []
        for path in sorted(evidence.rglob('isolated-transport/*.json')):
            value = json.loads(path.read_text())
            worker = json.loads(value['stdout'])
            transports.append(dict(path=str(path.relative_to(ROOT)), method=value['method'],
                actual_exit_code=value['actual_exit_code'], elapsed_seconds=value['elapsed_seconds'],
                parent_thread_count=value['parent_thread_count'], worker=worker))
        assertions = dict(exit_zero=process.returncode == 0,
            footer_26='Ran 26 tests' in process.stderr, footer_ok=process.stderr.rstrip().endswith('OK'),
            worker_count=len(transports) == 15,
            every_fresh_worker_single_thread=all(t['worker']['initial_thread_count'] == 1 for t in transports),
            every_worker_success=all(t['actual_exit_code'] == 0 and t['worker']['successful'] for t in transports),
            regression_parent_threaded=any(t['parent_thread_count'] >= 2 and 'parent-auxiliary-regression' in t['path'] for t in transports))
        if label == 'thread-contaminated':
            assertions.update(all_parents_threaded=all(t['parent_thread_count'] >= 2 for t in transports),
                foreign_thread_preserved='auxiliary_preserved_until_owned_shutdown=True' in process.stdout)
        results.append(dict(label=label, command=command, actual_exit_code=process.returncode,
            elapsed_seconds=elapsed, assertions=assertions, transports=transports))
    after = hashes()
    receipt = dict(schema='dongxi-watchdog-test-isolation-verification-v1',
        finished_at_utc=datetime.now(timezone.utc).isoformat(), scenarios=results,
        test_methods=methods, existing_test_count=26, source_before=before, source_after=after,
        stable_source=before == after,
        unchanged_production_guard=after[NAMES[1]] == 'f84e68594f2c37a4b196647c4abf76bb26ff83fdd4e8bf7f25b10cae1d481706',
        supersedes_test_source_sha256='9ea0a42755c943a2fd1ce7c79a4ed62c1a2bb12e45ce6022d7fc230a890461b9',
        historical_failure='experiments/reports/2026-10-05-current-cpu-closure/run-02/tests.log',
        scope='Actual fresh inert CPU controls only; no native model/GPU invocation, production guard change, installation or Git action.')
    receipt['status'] = 'passed' if receipt['stable_source'] and receipt['unchanged_production_guard'] and all(
        all(value['assertions'].values()) for value in results) else 'failed'
    with (OUTPUT/'verification.json').open('x') as handle:
        handle.write(json.dumps(receipt, indent=2, sort_keys=True)+'\n')
    print(json.dumps(dict(status=receipt['status'], existing_test_count=26,
        scenarios=[dict(label=r['label'], actual_exit_code=r['actual_exit_code'], elapsed_seconds=r['elapsed_seconds'],
                        assertions=r['assertions']) for r in results], source_after=after), indent=2))
    return 0 if receipt['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(run())
