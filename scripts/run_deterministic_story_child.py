#!/usr/bin/env python3
"""Fixed deterministic entry for the native story recovery/comparison campaign."""
import json
import os
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    if len(sys.argv)<2 or sys.argv[1]!='train':
        raise ValueError('Only the fixed local train_stories train entry is supported')
    # Must precede the Torch import and first CUDA context initialization.
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    print(json.dumps(dict(event='actual-story-deterministic-entry',
        torch_version=torch.__version__,deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cublas_workspace=os.environ['CUBLAS_WORKSPACE_CONFIG'],
        attention_backend='SDPA MATH only',tf32=False)),flush=True)
    sys.argv=[str(ROOT/'scripts/train_stories.py')]+sys.argv[1:]
    with sdpa_kernel(SDPBackend.MATH):
        runpy.run_path(sys.argv[0],run_name='__main__')


if __name__=='__main__':
    main()
