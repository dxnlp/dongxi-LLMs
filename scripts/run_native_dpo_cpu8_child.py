#!/usr/bin/env python3
"""Fixed CPU8 entry for unchanged native DPO or its accounted CPU comparison.

Only CPU intra/inter-op widths are set before loading either target. No GEMM
precision, tensor math, native recipe, validation, generation or save changes.
The owning fixed wrapper binds the forwarded argv and retains the stdout witness.
"""
import json
import os
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
WITNESS_PREFIX='DONGXI_DPO_CPU8 '
CPU_SETTINGS=dict(omp_num_threads='8',torch_num_threads=8,torch_num_interop_threads=1)


def configure_cpu_threads():
    os.environ['OMP_NUM_THREADS']='8'
    import torch
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    actual=dict(omp_num_threads=os.environ['OMP_NUM_THREADS'],torch_num_threads=torch.get_num_threads(),
        torch_num_interop_threads=torch.get_num_interop_threads())
    if actual!=CPU_SETTINGS:raise RuntimeError('Actual native DPO CPU8 thread settings differ')
    return actual


def main(argv=None):
    arguments=list(sys.argv[1:] if argv is None else argv)
    actual=configure_cpu_threads()
    target=ROOT/'scripts'/('run_native_dpo_stages.py' if '--comparison-child' in arguments else 'run_chapter11_spark_dpo.py')
    witness=dict(schema='dongxi-fixed-native-dpo-cpu8-witness-v1',target=str(target),**actual)
    print(WITNESS_PREFIX+json.dumps(witness,sort_keys=True),flush=True)
    sys.argv=[str(target),*arguments]
    runpy.run_path(str(target),run_name='__main__')
    return 0


if __name__=='__main__':raise SystemExit(main())
