"""Execute only Chapter 5 reference notebooks in fresh kernels.

Sources and learner edits are never overwritten. Executed copies and a JSON
manifest go to a newly allocated temporary directory printed at startup.
Run with the platform Python from any working directory. This script is not a
notebook server and leaves no persistent server running.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

import nbclient
import nbformat
from nbclient import NotebookClient
import torch
import matplotlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dongxi_llms.decoder_lab import fit_one_batch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    destination = Path(tempfile.mkdtemp(prefix='chapter5-reference-'))
    print(f'Executed copies and manifest: {destination}', flush=True)
    files = sorted(ROOT.glob('notebooks/day-05/*.ipynb'))
    files += sorted(ROOT.glob('notebooks/day-06/*.ipynb'))
    files += sorted(ROOT.glob('notebooks/day-07/*.ipynb'))
    if len(files) != 12:
        raise RuntimeError(f'Expected exactly 12 notebooks; found {len(files)}')
    mem = next((line for line in Path('/proc/meminfo').read_text().splitlines()
                if line.startswith('MemAvailable:')), 'not available') if Path('/proc/meminfo').exists() else 'not available'
    manifest = dict(timestamp=datetime.now(timezone.utc).isoformat(),
                    python=sys.version, torch=torch.__version__,
                    nbformat=nbformat.__version__, nbclient=nbclient.__version__,
                    matplotlib=matplotlib.__version__,
                    platform=platform.platform(), kernel='dgx-spark-native',
                    initial_mem_available=mem,
                    base_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                    source_sha256=sha(ROOT/'src/dongxi_llms/decoder_lab.py'),
                    visuals_sha256=sha(ROOT/'src/dongxi_llms/decoder_visuals.py'),
                    architecture_sha256=sha(ROOT/'src/dongxi_llms/decoder_architecture.py'),
                    audit_sha256=sha(ROOT/'src/dongxi_llms/decoder_audit.py'),
                    test_sha256=sha(ROOT/'tests/test_decoder_lab.py'),
                    notebooks=[])
    start = time.perf_counter()
    failed = False
    for path in files:
        item = dict(path=str(path.relative_to(ROOT)), sha256=sha(path))
        began = time.perf_counter()
        notebook = nbformat.read(path, as_version=4)
        try:
            nbformat.validate(notebook)
            NotebookClient(notebook, timeout=180, kernel_name='dgx-spark-native',
                           resources={'metadata': {'path': str(path.parent)}}).execute()
            images=sum('image/png' in o.get('data',{}) for c in notebook.cells
                       for o in c.get('outputs',[]))
            item.update(status='passed', code_cells=sum(c.cell_type=='code' for c in notebook.cells),
                        rendered_figures=images)
        except Exception as error:
            failed = True
            item.update(status='failed', error=str(error))
        item['elapsed_seconds'] = time.perf_counter()-began
        target = destination / path.parent.name
        target.mkdir(exist_ok=True)
        nbformat.write(notebook, target / path.name)
        manifest['notebooks'].append(item)
        print(f"{item['status']}: {item['path']} ({item['elapsed_seconds']:.2f}s)", flush=True)
        (destination/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    torch.set_num_threads(1)
    _, metrics = fit_one_batch()
    manifest['training'] = metrics
    manifest['elapsed_seconds'] = time.perf_counter()-start
    manifest['status'] = 'failed' if failed else 'passed'
    (destination/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k:v for k,v in metrics.items() if k != 'history'}), flush=True)
    print(f"Final manifest: {destination / 'manifest.json'}", flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
