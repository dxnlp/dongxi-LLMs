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
import os
from pathlib import Path
import platform
import re
import subprocess
import tempfile
import time

import nbformat
from nbclient import NotebookClient


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
        if cell.cell_type != 'code' or 'learner-exercise' not in cell.metadata.get('tags', []):
            continue
        if any(isinstance(node, ast.Constant) and node.value is Ellipsis
               for node in ast.walk(ast.parse(cell.source))):
            cell.metadata.setdefault('tags', []).append('skip-verification')
            skipped.append(index)
    return skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--days', nargs='*', type=int)
    parser.add_argument('--kernel', default='dgx-spark-native')
    parser.add_argument('--timeout', type=int, default=240)
    parser.add_argument('--export-figures', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sources = sorted((root / 'notebooks').glob('day-*/*.ipynb'))
    if args.days:
        sources = [p for p in sources if int(p.parent.name[4:]) in args.days]
    if not sources:
        raise ValueError('No matching notebooks')
    output = args.output or Path(tempfile.mkdtemp(prefix='dongxi-course-check-'))
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'manifest.json').exists():
        raise FileExistsError('Choose a new verification directory')
    started = time.perf_counter()
    manifest = {
        'date_utc': datetime.now(timezone.utc).isoformat(),
        'base_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'python': platform.python_version(), 'platform': platform.platform(),
        'kernel': args.kernel, 'scope': 'CPU teaching references and stored evidence',
        'notebooks': [], 'failures': [], 'minimum_observed_available_gib': None,
    }
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYTHONPATH=str(root/'src'))
    day_to_chapter = {1:1, 2:2, 3:3, 4:4, 5:5, 6:5, 7:5, 8:6, 9:6,
                      10:7, 11:8, 12:9, 13:9, 14:9, 15:10, 16:10, 17:11,
                      18:11, 19:12, 20:12, 21:12, 22:13, 23:13, 24:14,
                      25:14, 26:15, 27:15, 28:15}
    for source in sources:
        before = available_gib()
        if before is not None:
            old = manifest['minimum_observed_available_gib']
            manifest['minimum_observed_available_gib'] = before if old is None else min(old,before)
            if before < 25:
                raise RuntimeError('Host reserve below 25 GiB before notebook execution')
        record = {'path':str(source.relative_to(root)),
                  'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
        try:
            nb = nbformat.read(source, as_version=4)
            nbformat.validate(nb)
            # Execution-only preamble makes matplotlib figures visible even if
            # the caller selected a noninteractive shell backend. Cell indices
            # in the manifest remain indices into the original source notebook.
            source_cells = list(nb.cells)
            # Completed attempts and every complete reference still execute.
            skipped_scaffolds = mark_unfinished_scaffolds(source_cells)
            nb.cells.insert(0, nbformat.v4.new_code_cell(
                "get_ipython().run_line_magic('matplotlib', 'inline')"))
            began = time.perf_counter()
            NotebookClient(nb, timeout=args.timeout, kernel_name=args.kernel,
                           skip_cells_with_tag='skip-verification',
                           resources={'metadata':{'path':str(source.parent)}}).execute(env=env)
            target = output / source.parent.name / source.name
            target.parent.mkdir(parents=True,exist_ok=True)
            nbformat.write(nb,target)
            images = [(i-1,o.data['image/png']) for i,c in enumerate(nb.cells) if c.cell_type=='code'
                      for o in c.outputs if o.output_type in ('display_data','execute_result')
                      and 'image/png' in o.data]
            record.update(code_cells=sum(c.cell_type=='code' for c in source_cells),
                          executed_code_cells=sum(c.cell_type=='code' for c in source_cells)-len(skipped_scaffolds),
                          skipped_unfinished_exercise_cells=skipped_scaffolds,
                          images=len(images), seconds=time.perf_counter()-began,
                          output=str(target), image_cells=[i for i,_ in images], previews=[])
            if args.export_figures:
                day = int(source.parent.name[4:])
                directory = root / f'notebooks/figures/chapter-{day_to_chapter[day]:02d}'
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
            manifest['failures'].append({'path':record['path'],'error':repr(error)})
            print(json.dumps(manifest['failures'][-1]),flush=True)
        manifest['seconds'] = time.perf_counter()-started
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Verification manifest:',output/'manifest.json',flush=True)
    if manifest['failures']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
