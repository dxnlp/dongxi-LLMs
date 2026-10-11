"""Execute only new CS336 companions in the declared temporary CPU kernel.

Source notebooks stay output-free. Executed copies, extracted images and receipts
are new goal-local evidence. This is not a replacement course orchestrator.
"""
import base64
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import time

import nbformat
from nbclient import NotebookClient

ROOT=Path.cwd()
OUT=ROOT/'experiments/reports/2026-10-11-cs336-upgrade/run-01/systems'
NAMES=['01_resource_accounting','02_online_attention','03_scaling_prediction']
rows=[]
for name in NAMES:
    source=ROOT/f'notebooks/extensions/cs336/{name}.ipynb'
    start=time.monotonic();nb=nbformat.read(source,as_version=4)
    try:
        NotebookClient(nb,kernel_name='dongxi-cs336',timeout=120,resources={'metadata':{'path':str(source.parent)}}).execute()
        nbformat.write(nb,OUT/f'{name}-executed.ipynb')
        figures=[]
        for cell in nb.cells:
            for index,output in enumerate(cell.get('outputs',[])):
                data=output.get('data',{})
                if 'image/png' in data:
                    path=OUT/'figures'/f'{name}-{cell.id}-{index}.png';path.parent.mkdir(exist_ok=True)
                    path.write_bytes(base64.b64decode(data['image/png']))
                    figures.append({'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source_cell_id':cell.id})
        rows.append(dict(path=str(source.relative_to(ROOT)),status='passed',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),elapsed_seconds=time.monotonic()-start,code_cells=sum(c.cell_type=='code' for c in nb.cells),figures=figures))
    except Exception as error:
        rows.append(dict(path=str(source.relative_to(ROOT)),status='failed',error=repr(error),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        nbformat.write(nb,OUT/f'{name}-failed.ipynb')
        break
receipt=dict(command='JUPYTER_PATH=/tmp/dongxi-cs336-g9l7v90r/kernel/share/jupyter PYTHONPATH=src CUDA_VISIBLE_DEVICES=\'\' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /tmp/dongxi-cs336-g9l7v90r/venv/bin/python experiments/reports/2026-10-11-cs336-upgrade/run-01/systems/execute-notebooks.py',kernel='dongxi-cs336',expected_interpreter='/tmp/dongxi-cs336-g9l7v90r/venv/bin/python',rows=rows,boundary='Three new CPU notebooks only; output-free source retained; semantic conclusions require independent review')
(OUT/'notebook-execution-v1.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
if any(r['status']!='passed' for r in rows):raise SystemExit(1)
