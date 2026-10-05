#!/usr/bin/env python3
"""Exclusive evidence for the native story-clock companion; no model acquisition."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import traceback

from run_cpu_verification import execute
from verify_semantic_validation_integration import memory_gib, sha, write

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
NOTEBOOK = 'notebooks/day-09/01_read_training_clocks.ipynb'
OLD_PREVIEW = 'notebooks/figures/chapter-06/day-09-01_read_training_clocks-01.png'
PREVIEW = 'notebooks/figures/chapter-06/day-09-01_read_training_clocks-02.png'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kernel', required=True)
    parser.add_argument('--expected-prefix', type=Path, required=True)
    parser.add_argument('--host-reserve-gib', type=float, required=True)
    args = parser.parse_args()
    if not 0 < args.host_reserve_gib <= 1024 or Path(sys.prefix).resolve() != args.expected_prefix.resolve():
        parser.error('Explicit finite reserve and the existing isolated environment required')
    import torch
    from dongxi_llms.stories_work_lab import work_recovery_example
    if torch.cuda.is_available() or torch.version.cuda is not None:
        raise RuntimeError('Actual CPU-only environment required')
    output = args.output.resolve(); output.mkdir(mode=0o700, parents=True, exist_ok=False)
    paths = ['src/dongxi_llms/stories_training.py', 'src/dongxi_llms/story_work_budget.py',
        'src/dongxi_llms/stories_work_lab.py', 'src/dongxi_llms/stories_data.py',
        'src/dongxi_llms/decoder_lab.py', 'src/dongxi_llms/pretraining_lab.py',
        'src/dongxi_llms/work_budget.py', 'src/dongxi_llms/snapshot_io_budget.py',
        'src/dongxi_llms/batched_cache_lab.py', 'tests/test_stories_work_lab.py',
        'scripts/verify_stories_work_lesson.py', 'scripts/run_cpu_verification.py',
        'scripts/verify_semantic_validation_integration.py', 'scripts/verify_course_notebooks.py',
        'experiments/specs/2026-10-05-story-work-reader-lesson.md',
        'experiments/specs/2026-10-05-story-persistent-work.md', NOTEBOOK,
        'docs/course_manifest.json', 'uv.lock', OLD_PREVIEW,
        'experiments/reports/2026-09-14-tinystories-learning-result.json']
    before = {p:sha(ROOT/p) for p in paths}
    record = dict(command=list(sys.orig_argv), executable=sys.executable, prefix=sys.prefix,
        machine=platform.machine(), python=platform.python_version(), torch=str(torch.__version__),
        cuda_available=False, source_sha256_before=before, old_preview_sha256=before[OLD_PREVIEW],
        scope='Actual original native story CPU work clocks; no corpus, GPU, quality or physical quota',
        required_host_reserve_gib=args.host_reserve_gib, minimum_sampled_available_gib=memory_gib(),
        status='running', commands=[], model_scale_jobs_started=0, launch_authorized=False)
    write(output/'start.json', record)
    env = dict(os.environ, PYTHONPATH=str(ROOT/'src')+os.pathsep+str(ROOT/'tests'),
        CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    try:
        if record['minimum_sampled_available_gib'] < args.host_reserve_gib:
            raise RuntimeError('Sampled available memory below reserve')
        write(output/'native-work-example.json', work_recovery_example())
        write(output/'lower-cap-retry-refusal.json', work_recovery_example(24))
        commands = [('tests', [sys.executable, '-m', 'unittest', '-v', 'test_stories_work_lab']),
            ('notebook', [sys.executable, 'scripts/verify_course_notebooks.py', '--kernel', args.kernel,
                '--expected-prefix', str(args.expected_prefix), '--host-reserve-gib', str(args.host_reserve_gib),
                '--timeout', '180', '--output', str(output/'notebook'), '--notebooks', NOTEBOOK])]
        for name, command in commands:
            sample = memory_gib()
            record['minimum_sampled_available_gib'] = min(sample, record['minimum_sampled_available_gib'])
            if sample < args.host_reserve_gib: raise RuntimeError('Sampled available memory below reserve')
            result = execute(command, output/(name+'.log'), env=env, timeout=180)
            result['name'] = name; record['commands'].append(result)
            if result['exit_code'] != 0 or result['timed_out']:
                raise RuntimeError('Retained lesson child failed: '+name)
        log = (output/'tests.log').read_text()
        footer = re.findall(r'^Ran (\d+) tests in ([0-9.]+)s$', log, re.MULTILINE)
        if len(footer) != 1 or not re.search(r'^OK$', log, re.MULTILINE):
            raise AssertionError('Actual successful unittest footer required')
        record['test_footer'] = dict(tests=int(footer[0][0]), seconds=float(footer[0][1]), result='OK')
        manifest = json.loads((output/'notebook/manifest.json').read_text())
        if manifest['failures'] or len(manifest['notebooks']) != 1:
            raise AssertionError('One successful fresh notebook required')
        row = manifest['notebooks'][0]
        if row['images'] != 2 or row['executed_code_cells'] != 5:
            raise AssertionError('Preserve original three code cells and add two reference cells/two total figures')
        executed = json.loads(Path(row['output']).read_text())
        images = [o['data']['image/png'] for c in executed['cells'] if c['cell_type']=='code'
            for o in c.get('outputs',[]) if 'image/png' in o.get('data',{})]
        encoded = images[-1]; encoded = ''.join(encoded) if isinstance(encoded, list) else encoded
        image = base64.b64decode(encoded, validate=True)
        if not image.startswith(b'\x89PNG\r\n\x1a\n'): raise AssertionError('New actual plot must be PNG')
        target = ROOT/PREVIEW
        if target.exists():
            if hashlib.sha256(image).hexdigest() != sha(target):
                raise FileExistsError('Do not overwrite an earlier generated preview')
        else:
            with target.open('xb') as handle: handle.write(image)
        record['preview'] = dict(path=PREVIEW, sha256=sha(target), bytes=target.stat().st_size)
        record['notebook_manifest_sha256'] = sha(output/'notebook/manifest.json')
        record['fresh_notebook'] = row
        record['status'] = 'passed'
    except BaseException as error:
        record['status'] = 'failed'; record['error'] = repr(error)
        record['traceback'] = traceback.format_exc()
    record['source_sha256_after'] = {p:sha(ROOT/p) for p in paths}
    record['sources_unchanged'] = before == record['source_sha256_after']
    record['old_preview_unchanged'] = sha(ROOT/OLD_PREVIEW) == before[OLD_PREVIEW]
    if not record['sources_unchanged'] or not record['old_preview_unchanged']: record['status'] = 'failed'
    write(output/'verification.json', record)
    print(json.dumps(dict(status=record['status'], report=str(output/'verification.json'),
                         tests=record.get('test_footer'), error=record.get('error'))), flush=True)
    if record['status'] != 'passed': raise SystemExit(1)


if __name__ == '__main__': main()
