#!/usr/bin/env python3
"""Execute course notebooks in fresh CPU kernels without overwriting sources.

Outputs, environment identity, source hashes and failures are retained in a
new run directory. --export-figures writes only generated reference PNG assets.
"""
import argparse
import ast
import base64
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import tempfile
import time
import sys

import nbformat
from nbclient import NotebookClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from dongxi_llms.course_manifest import load_manifest, select_notebooks

KERNEL_IDENTITY_PREFIX = '__DONGXI_KERNEL_IDENTITY__:'


def kernel_identity(notebook):
    """Read the execution-only identity preamble, not user source outputs."""
    for output in notebook.cells[0].get('outputs', []):
        if output.get('output_type') == 'stream':
            for line in output.get('text', '').splitlines():
                if line.startswith(KERNEL_IDENTITY_PREFIX):
                    return json.loads(line[len(KERNEL_IDENTITY_PREFIX):])
    raise RuntimeError('Fresh kernel did not return an execution identity')


def execution_preamble():
    return "\n".join([
        "get_ipython().run_line_magic('matplotlib', 'inline')",
        "import sys, platform, json, importlib.metadata, torch",
        "assert not torch.cuda.is_available(), 'CPU teaching lane unexpectedly sees CUDA'",
        "identity = dict(executable=sys.executable, prefix=sys.prefix, python=platform.python_version(), platform=platform.platform(), cuda_available=torch.cuda.is_available(), torch_cuda_build=torch.version.cuda, packages={name:importlib.metadata.version(name) for name in ['torch','numpy','matplotlib','nbformat','nbclient','ipykernel','transformers','tokenizers','peft']})",
        f"print({KERNEL_IDENTITY_PREFIX!r} + json.dumps(identity, sort_keys=True))",
    ])


def prepare_output(output):
    """Never overwrite an old artifact even if its manifest was lost/not written."""
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Choose a new empty verification directory')
    output.mkdir(parents=True, exist_ok=True)
    return output


def check_kernel_prefix(identity, expected_prefix):
    # Resolve environment directories, not interpreter symlinks: two different
    # venv/bin/python links may point to the same underlying binary.
    if expected_prefix is not None and Path(identity['prefix']).resolve() != Path(expected_prefix).resolve():
        raise RuntimeError('Selected kernel is not from the declared isolated environment')


def available_gib():
    path = Path('/proc/meminfo')
    if not path.exists():
        return None
    line = next(l for l in path.read_text().splitlines() if l.startswith('MemAvailable:'))
    return int(line.split()[1]) / 1024**2


def mark_unfinished_scaffolds(cells):
    """Mark only explicit, unfinished exercise cells in an execution copy."""
    skipped = []
    for index, cell in enumerate(cells):
        # A carried-over source tag cannot silently disable a complete reference.
        tags = cell.metadata.get('tags', [])
        if 'skip-verification' in tags:
            cell.metadata['tags'] = [tag for tag in tags if tag != 'skip-verification']
        if cell.cell_type != 'code' or 'learner-exercise' not in cell.metadata.get('tags', []):
            continue
        tree = ast.parse(cell.source)
        # Tensor[...,0] is complete Python, not an unfinished RHS placeholder.
        indexing_ellipses = {node for subscript in ast.walk(tree) if isinstance(subscript, ast.Subscript)
                            for node in ast.walk(subscript.slice)
                            if isinstance(node, ast.Constant) and node.value is Ellipsis}
        if any(isinstance(node, ast.Constant) and node.value is Ellipsis and node not in indexing_ellipses
               for node in ast.walk(tree)):
            cell.metadata.setdefault('tags', []).append('skip-verification')
            skipped.append(index)
    return skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--days', nargs='*', type=int)
    parser.add_argument('--notebooks', nargs='+', help='Explicit registered repository-relative paths')
    parser.add_argument('--lane', choices=['core', 'optional', 'extension'])
    parser.add_argument('--kernel', default='dgx-spark-native')
    parser.add_argument('--expected-prefix', type=Path, help='Require actual kernel sys.prefix to match this environment')
    parser.add_argument('--timeout', type=int, default=240)
    parser.add_argument('--host-reserve-gib', type=float, default=25,
                        help='CPU-reference host reserve; Spark default25GiB, small CI hosts may explicitly use2')
    parser.add_argument('--export-figures', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.host_reserve_gib) or args.host_reserve_gib <= 0:
        parser.error('--host-reserve-gib must be finite and positive')
    root = Path(__file__).resolve().parents[1]
    registry = load_manifest(root, check_files=False)
    selected = select_notebooks(registry, days=args.days, paths=args.notebooks, lane=args.lane)
    sources = [root / row['path'] for row in selected]
    entries = {row['path']:row for row in selected}
    output = prepare_output(args.output or Path(tempfile.mkdtemp(prefix='dongxi-course-check-')))
    started = time.perf_counter()
    manifest = {
        'date_utc': datetime.now(timezone.utc).isoformat(),
        'base_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'python': platform.python_version(), 'platform': platform.platform(),
        'kernel': args.kernel, 'scope': 'CPU teaching references and stored evidence',
        'expected_kernel_prefix':str(args.expected_prefix) if args.expected_prefix is not None else None,
        'course_manifest_sha256':hashlib.sha256((root/'docs/course_manifest.json').read_bytes()).hexdigest(),
        'lock_sha256':hashlib.sha256((root/'uv.lock').read_bytes()).hexdigest() if (root/'uv.lock').exists() else None,
        'selected_routes':selected,
        'notebooks': [], 'failures': [], 'minimum_observed_available_gib': None,
        'required_host_reserve_gib':args.host_reserve_gib,
    }
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYTHONPATH=str(root/'src'))
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for source in sources:
        record = {'path':str(source.relative_to(root))}
        nb = None
        try:
            record['sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
            before = available_gib()
            if before is not None:
                old = manifest['minimum_observed_available_gib']
                manifest['minimum_observed_available_gib'] = before if old is None else min(old,before)
                if before < args.host_reserve_gib:
                    raise RuntimeError(f'Host reserve below {args.host_reserve_gib} GiB before notebook execution')
            nb = nbformat.read(source, as_version=4)
            nbformat.validate(nb)
            # Execution-only preamble makes matplotlib figures visible even if
            # the caller selected a noninteractive shell backend. Cell indices
            # in the manifest remain indices into the original source notebook.
            source_cells = list(nb.cells)
            # Completed attempts and every complete reference still execute.
            skipped_scaffolds = mark_unfinished_scaffolds(source_cells)
            nb.cells.insert(0, nbformat.v4.new_code_cell(execution_preamble()))
            began = time.perf_counter()
            NotebookClient(nb, timeout=args.timeout, kernel_name=args.kernel,
                           skip_cells_with_tag='skip-verification',
                           resources={'metadata':{'path':str(source.parent)}}).execute(env=env)
            actual_kernel = kernel_identity(nb)
            check_kernel_prefix(actual_kernel, args.expected_prefix)
            target = output / source.parent.name / source.name
            target.parent.mkdir(parents=True,exist_ok=True)
            nbformat.write(nb,target)
            images = [(i-1,o.data['image/png']) for i,c in enumerate(nb.cells) if c.cell_type=='code'
                      for o in c.outputs if o.output_type in ('display_data','execute_result')
                      and 'image/png' in o.data]
            record.update(code_cells=sum(c.cell_type=='code' for c in source_cells),
                          kernel_identity=actual_kernel,
                          executed_code_cells=sum(c.cell_type=='code' for c in source_cells)-len(skipped_scaffolds),
                          skipped_unfinished_exercise_cells=skipped_scaffolds,
                          images=len(images), seconds=time.perf_counter()-began,
                          output=str(target), image_cells=[i for i,_ in images], previews=[])
            if args.export_figures:
                chapter = entries[record['path']]['chapter']
                directory = root / f'notebooks/figures/chapter-{chapter:02d}'
                directory.mkdir(parents=True,exist_ok=True)
                for index,(cell,data) in enumerate(images,1):
                    dest = directory / f'{source.parent.name}-{source.stem}-{index:02d}.png'
                    dest.write_bytes(base64.b64decode(data))
                    record['previews'].append({'cell':cell,'path':str(dest.relative_to(root)),
                                               'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
            manifest['notebooks'].append(record)
            print(json.dumps({'path':record['path'],'code_cells':record['code_cells'],
                              'images':record['images'],'status':'passed'}),flush=True)
        except Exception as error:
            failure = dict(record, error=repr(error))
            if nb is not None:
                target = output / 'failures' / source.parent.name / source.name
                target.parent.mkdir(parents=True,exist_ok=True)
                nbformat.write(nb,target)
                failure['partial_output'] = str(target)
            manifest['failures'].append(failure)
            print(json.dumps(manifest['failures'][-1]),flush=True)
        manifest['seconds'] = time.perf_counter()-started
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Verification manifest:',output/'manifest.json',flush=True)
    if manifest['failures']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
