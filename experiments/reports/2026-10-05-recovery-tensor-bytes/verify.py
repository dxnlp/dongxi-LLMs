"""Narrow CPU byte-parity and historical-snapshot verification, without Git."""
import ast
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import struct
import subprocess
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tests'))
import torch
from dongxi_llms.batched_cache_lab import (Generation,PolicySession,digest,
    IDENTITY_ALGORITHM,load_state,make_model,model_hash,tensor_tree_equal)
from test_batched_cache_lab import independent_v2


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    directory=Path(__file__).resolve().parent;output=directory/'results.json'
    if output.exists() or (directory/'failure.json').exists():raise FileExistsError('Preserve earlier verification output')
    torch.set_num_threads(1);started=time.perf_counter()
    historical=[]
    for d in (ROOT/'fixtures').glob('batched-cache-recovery*'):
        historical.extend(p for p in d.rglob('*') if p.is_file())
    historical.extend((ROOT/'experiments/reports').glob('2026-10-04-batched-cache-recovery*'))
    previous={str(p.relative_to(ROOT)):sha(p) for p in historical}
    current=[ROOT/'src/dongxi_llms/batched_cache_lab.py',ROOT/'tests/test_batched_cache_lab.py',
        ROOT/'experiments/specs/2026-10-05-recovery-tensor-bytes.md',Path(__file__).resolve()]
    current.extend(directory.glob('original-*.txt'))
    before={str(p.relative_to(ROOT)):sha(p) for p in current}
    result={'schema':'dongxi-recovery-tensor-bytes-v1','status':'running',
        'identity_algorithm':IDENTITY_ALGORITHM,'source_spec_snapshot_sha256':before,
        'historical_artifact_sha256':previous,'command':list(sys.orig_argv),
        'environment':{'executable':sys.executable,'python':platform.python_version(),
            'platform':platform.platform(),'torch':str(torch.__version__),
            'numpy':importlib.metadata.version('numpy'),'threads':torch.get_num_threads(),
            'cuda_available':torch.cuda.is_available(),'git':'not queried by this authorized narrow check',
            'cpu_lock_sha256':sha(ROOT/'uv.lock'),'lock_boundary':'byte identity, not new environment installation'},
        'previous_control_failure':{'panel_tests':30,'errors':1,'suite_seconds':1.299,
            'cause':'New tamper test omitted required expected_contract keyword; fixed before passing full verification',
            'not_a_tensor_hash_failure':True},
        'metadata_inspection_failure':'A prior read-only JSON projection assumed training was a dict; actual list confirmed before verification',
        'dtype_layout_parity':[],'bf16_bit_controls':[],'historical_snapshots':[],
        'historical_policy_continuations':[]}
    try:
        old_source=(directory/'original-batched_cache_lab.py.txt').read_text()
        node=next(n for n in ast.parse(old_source).body if isinstance(n,ast.FunctionDef) and n.name=='digest')
        ns={'hashlib':hashlib,'json':json,'torch':torch}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<retained original digest>','exec'),ns)
        old=ns['digest']
        dtypes=(torch.bool,torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64,
                torch.float16,torch.float32,torch.float64,torch.complex64,torch.complex128)
        for dtype in dtypes:
            base=torch.arange(12).reshape(3,4).to(dtype)
            layouts=(base[0,0],base[:0],base[:,0:0],base.reshape(-1),base,base.T,base[:,::2])
            for name,x in zip(('scalar','empty-row','empty-columns','vector','matrix','transpose','slice'),layouts):
                nested={'tensor':x,'children':[x.clone(),(True,3,None)],'empty':{}}
                row={'dtype':str(dtype),'layout':name,'shape':list(x.shape),'stride':list(x.stride()),
                    'old_tensor_sha256':old(x),'current_tensor_sha256':digest(x),
                    'tensor_original_parity':old(x)==digest(x)==independent_v2(x),
                    'nested_original_parity':old(nested)==digest(nested)==independent_v2(nested),
                    'logical_contiguous_parity':digest(x)==digest(x.contiguous())}
                result['dtype_layout_parity'].append(row)
                assert all(row[k] for k in ('tensor_original_parity','nested_original_parity','logical_contiguous_parity'))
        bits=[0x0000,0x8000,0x3f80,0xc000,0x7f80,0xff80,0x7fc1,0x0001]
        bf=torch.tensor([b if b<32768 else b-65536 for b in bits],dtype=torch.int16).view(torch.bfloat16).reshape(2,4)
        def packed(x):return b''.join(struct.pack('=H',word&65535) for word in x.detach().contiguous().reshape(-1).view(torch.int16).tolist())
        try:old(bf)
        except TypeError as error:result['retained_old_bf16_failure']={'type':type(error).__name__,'message':str(error)}
        else:raise AssertionError('Prechange BF16 failure control unexpectedly passed')
        assert packed(bf)==b''.join(struct.pack('=H',b) for b in bits)
        for name,x in zip(('scalar','empty-row','empty-columns','vector','matrix','transpose','slice'),
                         (bf[0,0],bf[:0],bf[:,0:0],bf.reshape(-1),bf,bf.T,bf[:,::2])):
            row={'layout':name,'shape':list(x.shape),'bit_payload_hex':packed(x).hex(),
                'current_sha256':digest(x),'independent_packed_bits_match':digest(x)==independent_v2(x,packed),
                'logical_contiguous_parity':digest(x)==digest(x.contiguous())}
            result['bf16_bit_controls'].append(row);assert row['independent_packed_bits_match'] and row['logical_contiguous_parity']
        report=json.loads((ROOT/'experiments/reports/2026-10-04-batched-cache-recovery-root-verification.json').read_text())['results']
        model=make_model();engine=Generation(model,greedy=False);contract=engine.contract()
        record=report['generation_snapshot']
        loaded=load_state(ROOT/record['path'],expected_file_sha256=record['file_sha256'],expected_contract=contract)
        content=deepcopy(loaded);expected=content.pop('state_sha256')
        row={'path':record['path'],'embedded_state_old_and_current_match':old(content)==digest(content)==expected}
        result['historical_snapshots'].append(row);assert row['embedded_state_old_and_current_match']
        continued=Generation.restore(model,loaded,expected_contract=contract).finish()
        assert continued==report['sampled'];result['generation_archived_continuation_exact']=True
        for training in report['training']:
            objective,seed=training['objective'],training['seed']
            baseline=PolicySession(objective,seed)
            for _ in range(6):baseline.update()
            assert baseline.history==training['history']
            expected_final=training['final_snapshot']
            final=load_state(ROOT/expected_final['path'],expected_file_sha256=expected_final['file_sha256'],expected_contract=baseline.contract)
            for boundary in ('final_snapshot','completed_snapshot','pending_snapshot'):
                metadata=training[boundary]
                state=load_state(ROOT/metadata['path'],expected_file_sha256=metadata['file_sha256'],expected_contract=baseline.contract)
                content=deepcopy(state);expected=content.pop('state_sha256')
                row={'path':metadata['path'],'embedded_state_old_and_current_match':old(content)==digest(content)==expected}
                result['historical_snapshots'].append(row);assert row['embedded_state_old_and_current_match']
                if boundary=='final_snapshot':continue
                session=PolicySession(objective,seed);session.restore(state)
                if boundary=='pending_snapshot':
                    with patch.object(session,'collect',side_effect=AssertionError('Pending must not resample')):session.update()
                while session.step<6:session.update()
                snapshot=session.snapshot()
                fields=('policy','reference','optimizer','stream','rollout_rng','torch_rng','step','history','pending')
                checks={k:tensor_tree_equal(snapshot[k],final[k]) for k in fields}
                row={'objective':objective,'seed':seed,'boundary':boundary,
                     'full_history_matches_archived':session.history==training['history'],
                     'final_policy_sha256':model_hash(session.model),'final_hash_matches_archived':model_hash(session.model)==training['final_sha256'],
                     'all_final_fields_byte_equal':checks,'pending_no_resampling':boundary=='pending_snapshot'}
                result['historical_policy_continuations'].append(row)
                assert row['full_history_matches_archived'] and row['final_hash_matches_archived'] and all(checks.values())
        command=[sys.executable,'-m','unittest','discover','-s','tests','-p','test_batched_cache_lab.py','-v']
        tests=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,timeout=60)
        result['focused_tests']={'command':command,'actual_exit_code':tests.returncode,'stdout':tests.stdout,'stderr':tests.stderr}
        assert tests.returncode==0
        result['historical_artifacts_unchanged']=all(sha(ROOT/p)==h for p,h in previous.items())
        result['current_sources_unchanged']=all(sha(ROOT/p)==h for p,h in before.items())
        assert result['historical_artifacts_unchanged'] and result['current_sources_unchanged']
        result.update(status='passed',wall_seconds=time.perf_counter()-started,
            scope='Dense CPU tensor identity only; existing float64 tiny recovery; no BF16 training or production/pretrained/CUDA recovery')
        with output.open('x')as handle:handle.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:result[k] for k in ('status','identity_algorithm','historical_artifacts_unchanged','current_sources_unchanged','generation_archived_continuation_exact','wall_seconds')},indent=2))
        print('matrix rows:',len(result['dtype_layout_parity']),'BF16 rows:',len(result['bf16_bit_controls']),'old snapshots:',len(result['historical_snapshots']),'policy continuations:',len(result['historical_policy_continuations']))
    except (Exception,KeyboardInterrupt)as error:
        result.update(status='failed',error={'type':type(error).__name__,'message':str(error)},wall_seconds=time.perf_counter()-started)
        with (directory/'failure.json').open('x')as handle:handle.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
        raise


if __name__=='__main__':main()
